import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from torch.utils.data import DataLoader

# ------------------------------
# Config
# ------------------------------
MODEL_ID = "Qwen/Qwen3.5-2B-Base"
H = 2048
K = 64
MASTER_SEED = 42
LR = 5e-6
MAX_STEPS = 200
BATCH_SIZE = 1
MAX_LEN = 128
EVAL_SPLIT = 0.9

GRAFT_PATH = "graft_portable.pt"
GRAFT_META_PATH = "graft_portable_meta.pt"

# Verification config
CKA_PROBE_SAMPLES = 32
CKA_LAYERS = [0, 8, 16, 23]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# ------------------------------
# 1. Tokenizer
# ------------------------------
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# ------------------------------
# 2. Orthogonal basis
# ------------------------------
def make_arw_basis(hidden_dim, k, seed, device):
    assert k <= hidden_dim
    g = torch.Generator(device=device).manual_seed(seed)
    G = torch.randn(hidden_dim, k, generator=g, device=device, dtype=torch.float32)
    Q, _ = torch.linalg.qr(G)
    return Q[:, :k]

P = make_arw_basis(H, K, MASTER_SEED, device)
Pi = P @ P.T
print(f"Subspace rank: {K}, Projection matrix shape: {Pi.shape}")
print(f"Pi idempotency check (||Pi^2 - Pi||): {(Pi @ Pi - Pi).norm().item():.2e}  (must be ~0)")

# ------------------------------
# 3. ARW Hook Manager
# ------------------------------
class ARWManager:
    def __init__(self, hidden_dim, device):
        self.hidden_dim = hidden_dim
        self.device = device
        self.backward_hooks = []
        self.discard_stats = []

    def attach(self, Pi_write, model):
        self.clear_hooks()
        self.discard_stats = []

        def make_backward_hook(Pi_f32):
            def hook(grad):
                g_f32 = grad.to(torch.float32)
                if g_f32.shape[0] == Pi_f32.shape[0]:
                    active = Pi_f32 @ g_f32
                elif g_f32.shape[1] == Pi_f32.shape[0]:
                    active = g_f32 @ Pi_f32
                else:
                    return grad
                discarded = g_f32 - active
                tn = torch.norm(g_f32)
                dn = torch.norm(discarded)
                if tn > 1e-10:
                    self.discard_stats.append({'disc': dn.item(), 'tot': tn.item()})
                return active.to(grad.dtype)
            return hook

        count = 0
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear) and module.weight.requires_grad:
                if any(x in name for x in ['gate_proj', 'up_proj', 'down_proj']):
                    if module.weight.shape[0] == self.hidden_dim or module.weight.shape[1] == self.hidden_dim:
                        bh = module.weight.register_hook(make_backward_hook(Pi_write))
                        self.backward_hooks.append(bh)
                        count += 1
        print(f"[ARW] Backward hooks attached: {count}")

    def clear_hooks(self):
        for h in self.backward_hooks:
            h.remove()
        self.backward_hooks = []

    def report(self):
        if not self.discard_stats:
            return
        d = sum(s['disc'] for s in self.discard_stats)
        t = sum(s['tot'] for s in self.discard_stats)
        print(f"[ARW] Gradient discard ratio: {d/t:.6f}")

# ------------------------------
# 4. Data
# ------------------------------
def load_jsonl(path, tokenizer, max_len, split=0.9):
    ds = load_dataset('json', data_files=path, split='train')
    cols = ds.column_names

    if 'instruction' in cols and 'output' in cols:
        def merge(ex):
            ex['text'] = f"### Instruction:\n{ex['instruction']}\n\n### Output:\n{ex['output']}"
            return ex
        ds = ds.map(merge)
        cols = ds.column_names

    col = 'text' if 'text' in cols else cols[0]

    def tok(ex):
        return tokenizer(ex[col], truncation=True, max_length=max_len, padding="max_length")

    ds = ds.map(tok, batched=True, remove_columns=ds.column_names)
    ds = ds.train_test_split(test_size=1-split)
    ds['train'].set_format('torch')
    ds['test'].set_format('torch')
    return (
        DataLoader(ds['train'], batch_size=BATCH_SIZE, shuffle=True),
        DataLoader(ds['test'], batch_size=BATCH_SIZE)
    )

train_B, eval_B = load_jsonl('rnelang.jsonl', tokenizer, MAX_LEN, EVAL_SPLIT)
print(f"Train batches: {len(train_B)}, Eval batches: {len(eval_B)}")

# ------------------------------
# 5. Training
# ------------------------------
def train_domain(model, loader, opt, Pi_write_f32, steps, desc):
    model.train()
    for step, batch in enumerate(loader):
        if step >= steps:
            break
        ids = batch['input_ids'].to(device)
        am = batch['attention_mask'].to(device)
        opt.zero_grad()
        loss = model(input_ids=ids, attention_mask=am, labels=ids).loss
        loss.backward()

        with torch.no_grad():
            old_weights = {
                n: p.clone() for n, p in model.named_parameters()
                if p.requires_grad and p.ndim == 2
                and any(x in n for x in ['gate_proj', 'up_proj', 'down_proj'])
            }

        opt.step()

        with torch.no_grad():
            Pi = Pi_write_f32.to(torch.float32)
            for name, param in model.named_parameters():
                if name in old_weights:
                    raw_update = (param - old_weights[name]).to(torch.float32)
                    if 'gate_proj' in name or 'up_proj' in name:
                        projected_update = raw_update @ Pi
                    elif 'down_proj' in name:
                        projected_update = Pi @ raw_update
                    else:
                        continue
                    param.copy_(old_weights[name] + projected_update.to(param.dtype))

            if step == 0:
                for n, p in model.named_parameters():
                    if n in old_weights:
                        diff = (p - old_weights[n]).to(torch.float32)
                        if 'gate_proj' in n or 'up_proj' in n:
                            in_sub = diff @ Pi
                        elif 'down_proj' in n:
                            in_sub = Pi @ diff
                        else:
                            continue
                        out_sub = diff - in_sub
                        ratio = torch.norm(out_sub) / (torch.norm(diff) + 1e-10)
                        print(f"[PROJECTION CHECK] {n}: out-of-subspace ratio = {ratio.item():.8f}")
                        break

        if step % 40 == 0:
            print(f"{desc} | Step {step:3d} | Loss: {loss.item():.4f}")

@torch.no_grad()
def evaluate(model, loader, desc):
    model.eval()
    total = 0
    for batch in loader:
        ids = batch['input_ids'].to(device)
        am = batch['attention_mask'].to(device)
        total += model(input_ids=ids, attention_mask=am, labels=ids).loss.item()
    avg = total / len(loader)
    ppl = torch.exp(torch.tensor(avg)).item()
    print(f"=== {desc} | PPL: {ppl:.4f} | Loss: {avg:.4f} ===")
    return ppl

# ------------------------------
# 6. Isolation diagnostic
# ------------------------------
@torch.no_grad()
def check_isolation(model, base_weights, Pi_f32):
    total_norm, leak_norm = 0.0, 0.0
    for name, param in model.named_parameters():
        if name not in base_weights or param.ndim != 2:
            continue
        if not any(x in name for x in ['gate_proj', 'up_proj', 'down_proj']):
            continue
        delta = (param.to(torch.float32) - base_weights[name].to(device).to(torch.float32))
        if param.shape[0] == Pi_f32.shape[0]:
            in_sub = Pi_f32 @ delta
        elif param.shape[1] == Pi_f32.shape[0]:
            in_sub = delta @ Pi_f32
        else:
            continue
        out_sub = delta - in_sub
        total_norm += torch.norm(delta) ** 2
        leak_norm += torch.norm(out_sub) ** 2
    total_norm = total_norm ** 0.5
    leak_norm = leak_norm ** 0.5
    ratio = (leak_norm / total_norm).item() if total_norm > 0 else 0.0
    print(f"[ISOLATION] Out-of-subspace leak ratio: {ratio:.8f}  (target: <0.001)")

# ------------------------------
# 7. Graft extraction
# ------------------------------
@torch.no_grad()
def extract_graft(model, base_weights, Pi_f32, graft_path, meta_path):
    print("\n>>> EXTRACTING GRAFT...")
    graft = {}
    total_params, total_nonzero = 0, 0

    for name, param in model.named_parameters():
        if name not in base_weights or param.ndim != 2:
            continue
        if not any(x in name for x in ['gate_proj', 'up_proj', 'down_proj']):
            continue

        delta = (param.to(torch.float32) - base_weights[name].to(device).to(torch.float32))

        if 'gate_proj' in name or 'up_proj' in name:
            delta_in_sub = delta @ Pi_f32
        elif 'down_proj' in name:
            delta_in_sub = Pi_f32 @ delta

        graft[name] = delta_in_sub.cpu()
        n_params = delta_in_sub.numel()
        n_nonzero = (delta_in_sub.abs() > 1e-9).sum().item()
        total_params += n_params
        total_nonzero += n_nonzero

    meta = {
        'model_id': MODEL_ID,
        'hidden_dim': H,
        'rank_k': K,
        'master_seed': MASTER_SEED,
        'domain': 'B_portable',
        'anchor': 'clean_base',
        'layer_names': list(graft.keys()),
    }

    torch.save(graft, graft_path)
    torch.save(meta, meta_path)

    sparsity = 1.0 - (total_nonzero / total_params)
    size_mb = sum(v.numel() * 4 for v in graft.values()) / (1024 ** 2)
    print(f"[GRAFT] Layers     : {len(graft)}")
    print(f"[GRAFT] Parameters : {total_params:,}")
    print(f"[GRAFT] Sparsity   : {sparsity*100:.2f}%")
    print(f"[GRAFT] Size       : {size_mb:.2f} MB")
    return graft, meta

# ==============================
# VERIFICATION SUITE
# ==============================

# ------------------------------
# V1. CKA
# CKA measures representational similarity between two models at each layer.
# A real graft should produce high CKA between Model A (trained) and Model B
# (grafted) because both have identical weight deltas installed in the same
# subspace. A bug producing matching loss through an unrelated path would show
# low or random CKA because the internal representations would differ.
# Range: [0, 1]. Higher is more similar.
# ------------------------------

def _hsic(K_mat, L_mat):
    n = K_mat.shape[0]
    H = torch.eye(n, device=K_mat.device, dtype=torch.float32) - (1.0 / n)
    KH = K_mat @ H
    LH = L_mat @ H
    return (KH * LH).sum() / ((n - 1) ** 2)

@torch.no_grad()
def linear_cka(X, Y):
    X = X.to(torch.float32)
    Y = Y.to(torch.float32)
    K_mat = X @ X.T
    L_mat = Y @ Y.T
    hsic_xy = _hsic(K_mat, L_mat)
    hsic_xx = _hsic(K_mat, K_mat)
    hsic_yy = _hsic(L_mat, L_mat)
    denom = (hsic_xx * hsic_yy).sqrt()
    return (hsic_xy / denom).item() if denom > 1e-10 else 0.0

@torch.no_grad()
def collect_activations(model, loader, layer_indices, n_samples):
    model.eval()
    activations = {i: [] for i in layer_indices}
    hooks = []

    def make_hook(layer_idx):
        def hook(module, input, output):
            hidden = output[0] if isinstance(output, tuple) else output
            # Mean pool over sequence -> [batch, H]
            activations[layer_idx].append(hidden.mean(dim=1).float().cpu())
        return hook

    for idx in layer_indices:
        h = model.model.layers[idx].register_forward_hook(make_hook(idx))
        hooks.append(h)

    collected = 0
    for batch in loader:
        if collected >= n_samples:
            break
        ids = batch['input_ids'].to(device)
        am = batch['attention_mask'].to(device)
        model(input_ids=ids, attention_mask=am)
        collected += ids.shape[0]

    for h in hooks:
        h.remove()

    return {i: torch.cat(v, dim=0)[:n_samples] for i, v in activations.items()}

def run_cka_comparison(model_a, model_b, loader, layer_indices, n_samples, label_a, label_b):
    print(f"\n[CKA] Collecting activations: {label_a}...")
    acts_a = collect_activations(model_a, loader, layer_indices, n_samples)
    print(f"[CKA] Collecting activations: {label_b}...")
    acts_b = collect_activations(model_b, loader, layer_indices, n_samples)

    print(f"\n[CKA] Layer-wise similarity ({label_a} vs {label_b}):")
    print(f"  {'Layer':>6}  {'CKA':>8}  Rating")
    print(f"  {'-'*6}  {'-'*8}  {'-'*15}")

    results = {}
    for idx in layer_indices:
        cka = linear_cka(acts_a[idx], acts_b[idx])
        results[idx] = cka
        rating = "Very high" if cka > 0.95 else "High" if cka > 0.85 else "Moderate" if cka > 0.70 else "Low"
        print(f"  {idx:>6}  {cka:>8.4f}  {rating}")

    mean_cka = sum(results.values()) / len(results)
    print(f"  {'Mean':>6}  {mean_cka:>8.4f}")
    return results

# ------------------------------
# V2. Weight delta geometry
# If the graft is real, the weight difference between trained Model A and
# grafted Model B should be structured: predominantly in-subspace.
# Both models had the same delta added in Pi. Their weights should differ
# only in whatever small floating point residuals exist, not randomly.
# If it is a bug, the difference will be large and unstructured.
# ------------------------------

@torch.no_grad()
def weight_delta_geometry(weights_a, weights_b, Pi_f32, label):
    print(f"\n[WEIGHT DELTA] {label}")
    in_sub_sq, out_sub_sq, total_sq = 0.0, 0.0, 0.0

    Pi = Pi_f32.cpu()
    for name in weights_a:
        if name not in weights_b:
            continue
        wa = weights_a[name].to(torch.float32)
        wb = weights_b[name].to(torch.float32)
        diff = wb - wa

        if diff.shape[0] == Pi.shape[0]:
            in_sub = Pi @ diff
        elif diff.shape[1] == Pi.shape[0]:
            in_sub = diff @ Pi
        else:
            continue

        out_sub = diff - in_sub
        in_sub_sq += torch.norm(in_sub).item() ** 2
        out_sub_sq += torch.norm(out_sub).item() ** 2
        total_sq += torch.norm(diff).item() ** 2

    total = total_sq ** 0.5
    in_sub = in_sub_sq ** 0.5
    out_sub = out_sub_sq ** 0.5

    in_pct = 100 * in_sub / total if total > 0 else 0
    out_pct = 100 * out_sub / total if total > 0 else 0

    print(f"  Total weight diff norm         : {total:.6f}")
    print(f"  In-subspace component          : {in_sub:.6f}  ({in_pct:.1f}%)")
    print(f"  Out-of-subspace component      : {out_sub:.6f}  ({out_pct:.1f}%)")
    print(f"  Expectation: in-subspace >> out-of-subspace for a real graft")

    return {'total': total, 'in_sub': in_sub, 'out_sub': out_sub, 'in_pct': in_pct}

# ------------------------------
# V3. Verdict aggregator
# Uses relative PPL gap, not absolute.
# Absolute thresholds fail when target PPL is near 1.0.
# ------------------------------

def final_verdict(ppl_zero, ppl_trained, ppl_grafted, cka_results, delta_geo):
    print("\n" + "="*60)
    print("VERIFICATION VERDICT")
    print("="*60)

    checks = {}

    # CHECK 1: PPL recovery (relative)
    total_gap = ppl_zero - ppl_trained
    recovered = ppl_zero - ppl_grafted
    recovery_pct = 100 * recovered / total_gap if total_gap > 0 else 0
    relative_gap = abs(ppl_grafted - ppl_trained) / ppl_trained

    checks['ppl_recovery'] = recovery_pct > 90.0
    print(f"\n[CHECK 1] PPL Recovery (relative, not absolute)")
    print(f"  Untrained PPL                  : {ppl_zero:.4f}")
    print(f"  Trained PPL (target)           : {ppl_trained:.4f}")
    print(f"  Grafted PPL                    : {ppl_grafted:.4f}")
    print(f"  Recovery from baseline         : {recovery_pct:.1f}%  (pass: >90%)")
    print(f"  Relative gap to target         : {relative_gap*100:.1f}%")
    print(f"  Status: {'PASS' if checks['ppl_recovery'] else 'FAIL'}")

    # CHECK 2: CKA
    mean_cka = sum(cka_results.values()) / len(cka_results) if cka_results else 0
    checks['cka'] = mean_cka > 0.85
    print(f"\n[CHECK 2] CKA Representational Similarity")
    print(f"  Mean CKA across layers         : {mean_cka:.4f}  (pass: >0.85)")
    for layer, score in cka_results.items():
        rating = "Very high" if score > 0.95 else "High" if score > 0.85 else "Moderate" if score > 0.70 else "Low"
        print(f"  Layer {layer:>2}: {score:.4f}  ({rating})")
    print(f"  Status: {'PASS' if checks['cka'] else 'FAIL'}")

    # CHECK 3: Weight delta geometry
    if delta_geo:
        checks['weight_geometry'] = delta_geo['in_pct'] > 70.0
        print(f"\n[CHECK 3] Weight Delta Geometry")
        print(f"  In-subspace fraction           : {delta_geo['in_pct']:.1f}%  (pass: >70%)")
        print(f"  Out-of-subspace fraction       : {100 - delta_geo['in_pct']:.1f}%")
        print(f"  Status: {'PASS' if checks['weight_geometry'] else 'FAIL'}")
    else:
        checks['weight_geometry'] = False
        print(f"\n[CHECK 3] Weight Delta Geometry: SKIPPED")

    # Overall
    passed = sum(checks.values())
    total_checks = len(checks)
    print(f"\n{'='*60}")
    print(f"CHECKS PASSED: {passed}/{total_checks}")
    if passed == total_checks:
        print("CONCLUSION: Graft is a verified real transplant.")
    elif passed >= total_checks - 1:
        print("CONCLUSION: Graft likely real. One check marginal, investigate.")
    else:
        print("CONCLUSION: INVESTIGATE. Multiple checks failed.")
    print("="*60)
    return checks


# ==============================
# THE EXPERIMENT
# ==============================

# STAGE 1: Load base model
print("\n>>> STAGE 1: Load base model (anchor)")
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.float32).to(device)
model = model.float()
for param in model.parameters():
    param.requires_grad = True
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.0)

print("Snapshotting base weights (anchor fingerprint)...")
base_weights = {
    n: p.clone().detach().cpu()
    for n, p in model.named_parameters()
    if p.requires_grad and p.ndim == 2
    and any(x in n for x in ['gate_proj', 'up_proj', 'down_proj'])
}
print(f"Snapshot taken: {len(base_weights)} layers.")
ppl_B_zero = evaluate(model, eval_B, "Domain B (Untrained Base)")

# STAGE 2: Train
print("\n>>> STAGE 2: Train Domain B with ARW")
arw = ARWManager(H, device)
arw.attach(Pi, model)
train_domain(model, train_B, optimizer, Pi, MAX_STEPS, "Domain B")
arw.report()
arw.clear_hooks()
ppl_B_trained = evaluate(model, eval_B, "Domain B (Trained, ARW)")

# STAGE 3: Isolation
print("\n>>> STAGE 3: Isolation diagnostics")
check_isolation(model, base_weights, Pi)

# Snapshot trained weights for geometry check
trained_weights_A = {
    n: p.clone().detach().cpu()
    for n, p in model.named_parameters()
    if p.ndim == 2 and any(x in n for x in ['gate_proj', 'up_proj', 'down_proj'])
}

# STAGE 4: Extract graft
print("\n>>> STAGE 4: Graft extraction")
graft, meta = extract_graft(model, base_weights, Pi, GRAFT_PATH, GRAFT_META_PATH)

# Free training model
print("\n>>> Freeing trained model from VRAM...")
del model, optimizer
torch.cuda.empty_cache()
if torch.cuda.is_available():
    free_vram = torch.cuda.mem_get_info()[0] / (1024**3)
    print(f"[VRAM] Free: {free_vram:.2f} GB")

# STAGE 5: Load fresh instance and install graft
print("\n>>> STAGE 5: Load fresh instance and install graft")
meta_loaded = torch.load(GRAFT_META_PATH, map_location='cpu', weights_only=True)
graft_loaded = torch.load(GRAFT_PATH, map_location='cpu', weights_only=True)

assert meta_loaded['model_id'] == MODEL_ID
assert meta_loaded['master_seed'] == MASTER_SEED
assert meta_loaded['anchor'] == 'clean_base'
print("[REINSTALL] Metadata verified.")

fresh = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.float32).to(device)
fresh = fresh.float()
fresh.eval()

ppl_fresh_before = evaluate(fresh, eval_B, "Domain B (Fresh Instance, No Graft)")

installed = 0
with torch.no_grad():
    for name, param in fresh.named_parameters():
        if name in graft_loaded:
            param.add_(graft_loaded[name].to(device).to(torch.float32))
            installed += 1
print(f"[REINSTALL] Graft installed into {installed} layers.")

ppl_fresh_after = evaluate(fresh, eval_B, "Domain B (Fresh Instance + Graft)")

n_hooks = sum(
    len(m._forward_hooks) + len(m._backward_hooks) + len(m._forward_pre_hooks)
    for m in fresh.modules()
)
print(f"[REINSTALL] Active hooks: {n_hooks} (must be 0)")

# Snapshot grafted weights for geometry check
grafted_weights_B = {
    n: p.clone().detach().cpu()
    for n, p in fresh.named_parameters()
    if p.ndim == 2 and any(x in n for x in ['gate_proj', 'up_proj', 'down_proj'])
}

# STAGE 6: Verification suite
print("\n>>> STAGE 6: Verification suite")

# Reconstruct Model A by installing the same graft on another fresh base.
# Both Model A (reconstructed) and Model B (fresh + graft) should have
# identical weights and thus identical activations. CKA should be ~1.0.
print("[VERIFY] Reconstructing Model A for CKA (fresh base + same graft)...")
model_a_recon = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, dtype=torch.float32
).to(device)
model_a_recon = model_a_recon.float()
model_a_recon.eval()

with torch.no_grad():
    for name, param in model_a_recon.named_parameters():
        if name in graft_loaded:
            param.add_(graft_loaded[name].to(device).to(torch.float32))

# CKA between reconstructed A and fresh B
cka_results = run_cka_comparison(
    model_a=model_a_recon,
    model_b=fresh,
    loader=eval_B,
    layer_indices=CKA_LAYERS,
    n_samples=CKA_PROBE_SAMPLES,
    label_a="Model A (reconstructed via graft)",
    label_b="Model B (fresh + graft)"
)

del model_a_recon
torch.cuda.empty_cache()

# Weight delta geometry: trained A vs grafted B
delta_geo = weight_delta_geometry(
    weights_a=trained_weights_A,
    weights_b=grafted_weights_B,
    Pi_f32=Pi,
    label="Trained Model A vs Grafted Model B"
)

del fresh
torch.cuda.empty_cache()

# STAGE 7: Verdict
final_verdict(
    ppl_zero=ppl_B_zero,
    ppl_trained=ppl_B_trained,
    ppl_grafted=ppl_fresh_after,
    cka_results=cka_results,
    delta_geo=delta_geo
)

print("\n" + "="*60)
print("EXPERIMENT SUMMARY")
print("="*60)
print(f"Model                          : {MODEL_ID}")
print(f"Subspace seed / rank           : {MASTER_SEED} / K={K}")
print(f"Training steps                 : {MAX_STEPS}")
print(f"Domain B PPL (untrained)       : {ppl_B_zero:.4f}")
print(f"Domain B PPL (trained w/ ARW)  : {ppl_B_trained:.4f}")
print(f"Domain B PPL (fresh + graft)   : {ppl_fresh_after:.4f}")
recovery = 100 * (ppl_B_zero - ppl_fresh_after) / (ppl_B_zero - ppl_B_trained)
print(f"PPL recovery                   : {recovery:.1f}%")
print("="*60)