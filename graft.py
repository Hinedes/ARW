import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from torch.utils.data import DataLoader
import os

MODEL_ID = "Qwen/Qwen3.5-2B-Base"
H = 2048
K = 64
MASTER_SEED = 42
LR = 5e-6
MAX_STEPS = 200
BATCH_SIZE = 1
MAX_LEN = 128
EVAL_SPLIT = 0.9

GRAFT_PATH = "graft_B.pt"
GRAFT_META_PATH = "graft_B_meta.pt"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# ------------------------------
# 1. Model & tokenizer
# ------------------------------
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print("Loading Model in strict float32...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, 
    dtype=torch.float32
).to(device)

# THE SLEDGEHAMMER: Force physical 32-bit memory
model = model.float()

for param in model.parameters():
    param.requires_grad = True

optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.0)

# ------------------------------
# 2. Orthogonal basis pair (float32, never cast to bfloat16)
# ------------------------------
def make_arw_basis_pair(hidden_dim, k, seed, device):
    assert 2 * k <= hidden_dim
    g = torch.Generator(device=device).manual_seed(seed)
    G = torch.randn(hidden_dim, 2 * k, generator=g, device=device, dtype=torch.float32)
    Q, _ = torch.linalg.qr(G)
    return Q[:, :k], Q[:, k:2*k]

P_A, P_B = make_arw_basis_pair(H, K, MASTER_SEED, device)
Pi_A = P_A @ P_A.T   # float32, NEVER cast
Pi_B = P_B @ P_B.T   # float32, NEVER cast

overlap = (P_A.T @ P_B).abs().max().item()
print(f"Orthogonality check (float32): {overlap:.2e}")

# ------------------------------
# 3. ARW Hook Manager (precision-safe)
# ------------------------------
class ARWManager:
    def __init__(self, hidden_dim, device):
        self.hidden_dim = hidden_dim
        self.device = device
        self.Pi_write = None
        self.Pi_blind = None
        self.backward_hooks = []
        self.forward_hooks = []
        self.discard_stats = []

    def set_active_domain(self, Pi_write, model, Pi_other=None):
        self.clear_hooks()
        self.discard_stats = []
        self.Pi_write = Pi_write
        if Pi_other is not None:
            I = torch.eye(self.hidden_dim, device=self.device, dtype=torch.float32)
            self.Pi_blind = I - Pi_other
        else:
            self.Pi_blind = None

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

        def make_forward_hook(Pi_blind_f32):
            if Pi_blind_f32 is None:
                return None
            def hook(module, input):
                x = input[0].to(torch.float32)
                x_proj = x @ Pi_blind_f32
                return (x_proj.to(input[0].dtype),) + input[1:]
            return hook

        for name, module in model.named_modules():
            if isinstance(module, nn.Linear) and module.weight.requires_grad:
                if 'gate_proj' in name or 'up_proj' in name or 'down_proj' in name:
                    if module.weight.shape[0] == self.hidden_dim or module.weight.shape[1] == self.hidden_dim:
                        bh = module.weight.register_hook(make_backward_hook(self.Pi_write))
                        self.backward_hooks.append(bh)
                    if self.Pi_blind is not None and ('gate_proj' in name or 'up_proj' in name):
                        if module.weight.shape[1] == self.hidden_dim:
                            fh = module.register_forward_pre_hook(make_forward_hook(self.Pi_blind))
                            self.forward_hooks.append(fh)

        print(f"[ARW] Backward: {len(self.backward_hooks)}, Forward: {len(self.forward_hooks)} (blind {'on' if self.Pi_blind is not None else 'off'})")

    def clear_hooks(self):
        for h in self.backward_hooks: h.remove()
        for h in self.forward_hooks: h.remove()
        self.backward_hooks = []
        self.forward_hooks = []

    def report(self, name):
        if not self.discard_stats: return
        d = sum(s['disc'] for s in self.discard_stats)
        t = sum(s['tot'] for s in self.discard_stats)
        print(f"[DIAGNOSTIC] {name} Gradient Discard Ratio: {d/t:.6f}" if t>0 else "")

# ------------------------------
# 4. Data
# ------------------------------
def load_jsonl(path, tokenizer, max_len, split=0.9):
    ds = load_dataset('json', data_files=path, split='train')
    col = 'text' if 'text' in ds.column_names else ds.column_names[0]
    def tok(ex): return tokenizer(ex[col], truncation=True, max_length=max_len, padding="max_length")
    ds = ds.map(tok, batched=True, remove_columns=ds.column_names)
    ds = ds.train_test_split(test_size=1-split)
    ds['train'].set_format('torch')
    ds['test'].set_format('torch')
    return DataLoader(ds['train'], batch_size=BATCH_SIZE, shuffle=True), DataLoader(ds['test'], batch_size=BATCH_SIZE)

train_A, eval_A = load_jsonl('train.jsonl', tokenizer, MAX_LEN, EVAL_SPLIT)
train_B, eval_B = load_jsonl('rnelang.jsonl', tokenizer, MAX_LEN, EVAL_SPLIT)

# ------------------------------
# 5. Training with post-opt projection (float32 precision)
# ------------------------------
def train_domain(model, loader, opt, desc, steps, Pi_write_f32):
    model.train()
    for step, batch in enumerate(loader):
        if step >= steps: break
        ids = batch['input_ids'].to(device)
        am = batch['attention_mask'].to(device)
        opt.zero_grad()
        loss = model(input_ids=ids, attention_mask=am, labels=ids).loss
        loss.backward()

        with torch.no_grad():
            old_weights = {n: p.clone() for n, p in model.named_parameters()
                   if p.requires_grad and p.ndim==2 and
                   any(x in n for x in ['gate_proj','up_proj','down_proj'])}

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
                    param.copy_(old_weights[name] + projected_update.to(model.dtype))

            if step == 0:
                for n, p in model.named_parameters():
                    if n in old_weights:
                        diff = (p - old_weights[n]).to(torch.float32)
                        if 'gate_proj' in n or 'up_proj' in n:
                            in_subspace = diff @ Pi
                        elif 'down_proj' in n:
                            in_subspace = Pi @ diff
                        else:
                            continue
                        out_of_subspace = diff - in_subspace
                        ratio = torch.norm(out_of_subspace) / (torch.norm(diff) + 1e-10)
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
# 6. Diagnostics
# ------------------------------
@torch.no_grad()
def compute_weight_update_leak(model, before_weights, Pi_A_f32):
    total_norm, leak_norm = 0.0, 0.0
    for name, param in model.named_parameters():
        if name not in before_weights or param.ndim != 2:
            continue
        diff = param - before_weights[name].to(device)
        diff_f32 = diff.to(torch.float32)
        if param.shape[0] == Pi_A_f32.shape[0]:
            leak = Pi_A_f32 @ diff_f32
        elif param.shape[1] == Pi_A_f32.shape[0]:
            leak = diff_f32 @ Pi_A_f32
        else:
            continue
        total_norm += torch.norm(diff_f32) ** 2
        leak_norm += torch.norm(leak) ** 2
    total_norm = total_norm.sqrt()
    leak_norm = leak_norm.sqrt()
    ratio = (leak_norm / total_norm).item() if total_norm > 0 else 0.0
    print(f"[DIAGNOSTIC] Weight update leak into A: {leak_norm.item():.8f} / {total_norm.item():.4f}  Ratio: {ratio:.8f}")


@torch.no_grad()
def weight_leak_float32_params(model, before_weights, Pi_A_f32):
    total_norm, leak_norm = 0.0, 0.0
    for name, param in model.named_parameters():
        if name not in before_weights or param.ndim != 2:
            continue
        current = param.to(torch.float32)
        before = before_weights[name].to(device).to(torch.float32)
        diff = current - before
        if param.shape[0] == Pi_A_f32.shape[0]:
            leak = Pi_A_f32 @ diff
        elif param.shape[1] == Pi_A_f32.shape[0]:
            leak = diff @ Pi_A_f32
        else:
            continue
        total_norm += torch.norm(diff) ** 2
        leak_norm += torch.norm(leak) ** 2
    total_norm = total_norm.sqrt()
    leak_norm = leak_norm.sqrt()
    ratio = (leak_norm / total_norm).item() if total_norm > 0 else 0.0
    print(f"[FLOAT32 PARAM DIAGNOSTIC] Weight leak ratio: {ratio:.8f}   (should be ~0.0000027)")

@torch.no_grad()
def ablation_correct(model, loader, Pi_B_f32, weights_post_A, desc):
    saved = {}
    for n, p in model.named_parameters():
        if n in weights_post_A:
            saved[n] = p.clone()
            delta = (p - weights_post_A[n].to(device)).to(torch.float32)
            if p.shape[0] == Pi_B_f32.shape[0]:
                b_comp = Pi_B_f32 @ delta
            elif p.shape[1] == Pi_B_f32.shape[0]:
                b_comp = delta @ Pi_B_f32
            else: continue
            p.sub_(b_comp.to(torch.float32))
    ppl = evaluate(model, loader, desc)
    for n, p in model.named_parameters():
        if n in saved: p.copy_(saved[n])
    return ppl

# ------------------------------
# 7. Grafting: Extraction
# ------------------------------
@torch.no_grad()
def extract_graft(model, weights_post_A, Pi_B_f32, graft_path, meta_path):
    """
    Extract Domain B's graft from the trained model.
    The graft is the projection of ΔW onto Pi_B for each FFN layer.
    Only the non-zero subspace component is stored.
    weights_post_A is the snapshot taken after Domain A training,
    before Domain B training began.
    """
    print("\n>>> EXTRACTING GRAFT B...")
    graft = {}
    total_params = 0
    total_nonzero = 0

    for name, param in model.named_parameters():
        if name not in weights_post_A or param.ndim != 2:
            continue
        if not any(x in name for x in ['gate_proj', 'up_proj', 'down_proj']):
            continue

        # ΔW: what changed between post-A and post-B
        delta = (param.to(torch.float32) - weights_post_A[name].to(device).to(torch.float32))

        # Project strictly onto Pi_B subspace
        if 'gate_proj' in name or 'up_proj' in name:
            # Input dimension H: right-multiply
            delta_in_B = delta @ Pi_B_f32
        elif 'down_proj' in name:
            # Output dimension H: left-multiply
            delta_in_B = Pi_B_f32 @ delta

        # Store on CPU to save VRAM
        graft[name] = delta_in_B.cpu()

        n_params = delta_in_B.numel()
        n_nonzero = (delta_in_B.abs() > 1e-9).sum().item()
        total_params += n_params
        total_nonzero += n_nonzero

    # Metadata: everything needed to reinstall without the training script
    meta = {
        'model_id': MODEL_ID,
        'hidden_dim': H,
        'rank_k': K,
        'master_seed': MASTER_SEED,
        'domain': 'B',
        'Pi_B': Pi_B_f32.cpu(),  # Store projection matrix for reinstallation
        'layer_names': list(graft.keys()),
    }

    torch.save(graft, graft_path)
    torch.save(meta, meta_path)

    sparsity = 1.0 - (total_nonzero / total_params)
    graft_size_mb = sum(v.numel() * 4 for v in graft.values()) / (1024 ** 2)

    print(f"[GRAFT] Layers extracted   : {len(graft)}")
    print(f"[GRAFT] Total parameters   : {total_params:,}")
    print(f"[GRAFT] Non-zero entries   : {total_nonzero:,}")
    print(f"[GRAFT] Effective sparsity : {sparsity:.4f} ({sparsity*100:.2f}%)")
    print(f"[GRAFT] Artifact size      : {graft_size_mb:.2f} MB")
    print(f"[GRAFT] Saved to           : {graft_path}")
    print(f"[GRAFT] Meta saved to      : {meta_path}")
    return graft, meta


# ------------------------------
# 8. Grafting: Reinstallation onto a fresh model
# ------------------------------
@torch.no_grad()
def reinstall_graft_and_evaluate(graft_path, meta_path, eval_loader, desc):
    """
    Load a completely fresh base model with no training history.
    Install the graft via W_active = W_base + ΔW_graft.
    Evaluate. If PPL recovers to ~ppl_B_base, Grafting is proven end to end.
    No hooks. No ARW. Just a matrix addition and a standard forward pass.
    """
    print(f"\n>>> REINSTALLATION TEST: Loading fresh {MODEL_ID}...")
    meta = torch.load(meta_path, map_location='cpu')
    graft = torch.load(graft_path, map_location='cpu')

    # Verify metadata integrity
    assert meta['model_id'] == MODEL_ID, "Model ID mismatch. Graft is not compatible with this model family."
    assert meta['master_seed'] == MASTER_SEED, "Seed mismatch. Subspace geometry is inconsistent."
    print(f"[REINSTALL] Graft metadata verified. Domain: {meta['domain']}, Seed: {meta['master_seed']}, K: {meta['rank_k']}")

    # Fresh base model, completely untrained
    fresh_model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        dtype=torch.float32
    ).to(device)
    fresh_model = fresh_model.float()
    fresh_model.eval()

    # Baseline PPL of the fresh model on Domain B before graft
    print("[REINSTALL] Evaluating fresh model BEFORE graft installation...")
    ppl_before = evaluate(fresh_model, eval_loader, f"{desc} (Fresh Base, No Graft)")

    # Install graft: W_active = W_base + ΔW_graft
    # No hooks, no routing, no adapter. Pure matrix addition.
    installed_count = 0
    for name, param in fresh_model.named_parameters():
        if name in graft:
            delta = graft[name].to(device).to(torch.float32)
            param.add_(delta)
            installed_count += 1

    print(f"[REINSTALL] Installed graft into {installed_count} layers via W_base + ΔW_graft.")

    # PPL after graft installation
    print("[REINSTALL] Evaluating fresh model AFTER graft installation...")
    ppl_after = evaluate(fresh_model, eval_loader, f"{desc} (Fresh Base + Graft B)")

    # Verify forward pass is standard (no hooks present)
    n_hooks = sum(
        len(m._forward_hooks) + len(m._backward_hooks)
        for m in fresh_model.modules()
    )
    print(f"[REINSTALL] Active hooks on reinstalled model: {n_hooks} (must be 0)")

    print(f"\n[REINSTALL VERDICT]")
    print(f"  Fresh base PPL (no graft)  : {ppl_before:.4f}")
    print(f"  After graft installation   : {ppl_after:.4f}")
    print(f"  PPL delta                  : {ppl_after - ppl_before:+.4f}")
    print(f"  Target (original B training): see ppl_B_base above")

    # Clean up VRAM
    del fresh_model
    torch.cuda.empty_cache()

    return ppl_before, ppl_after


# ------------------------------
# 9. The Experiment
# ------------------------------
arw = ARWManager(H, device)

print("\n>>> STAGE 1: Domain A")
arw.set_active_domain(Pi_A, model)
train_domain(model, train_A, optimizer, "Domain A", MAX_STEPS, Pi_A)
arw.report("Domain A")
ppl_A_base = evaluate(model, eval_A, "Domain A (Baseline)")

weights_post_A = {n: p.clone().detach().cpu() for n, p in model.named_parameters()
                  if p.requires_grad and p.ndim==2 and
                  any(x in n for x in ['gate_proj','up_proj','down_proj'])}

print("\n>>> STAGE 2: Domain B (blind A)")
arw.set_active_domain(Pi_B, model, Pi_other=Pi_A)
train_domain(model, train_B, optimizer, "Domain B", MAX_STEPS, Pi_B)
arw.report("Domain B")
ppl_B_base = evaluate(model, eval_B, "Domain B (Baseline)")

print("\n>>> STAGE 3: Diagnostics")
arw.clear_hooks()

compute_weight_update_leak(model, weights_post_A, Pi_A)
weight_leak_float32_params(model, weights_post_A, Pi_A)
ppl_A_after = evaluate(model, eval_A, "Domain A (After B, natural)")
ppl_A_ablat = ablation_correct(model, eval_A, Pi_B, weights_post_A, "Domain A (B delta removed)")

print("\n>>> STAGE 4: Graft Extraction")
graft, meta = extract_graft(model, weights_post_A, Pi_B, GRAFT_PATH, GRAFT_META_PATH)

print("\n>>> STAGE 5: Graft Reinstallation (End-to-End Proof)")
ppl_fresh_before, ppl_fresh_after = reinstall_graft_and_evaluate(
    GRAFT_PATH, GRAFT_META_PATH, eval_B, "Domain B"
)

# ------------------------------
# 10. Final Verdict
# ------------------------------
print("\n" + "="*60)
print("FINAL VERDICT (Grafting End-to-End)")
print("="*60)
print(f"Orthogonality (float32)        : {overlap:.2e}")
print(f"Domain A Baseline              : {ppl_A_base:.4f}")
print(f"Domain B PPL (trained)         : {ppl_B_base:.4f}")
print(f"Domain A After B               : {ppl_A_after:.4f}")
print(f"Domain A Ablated               : {ppl_A_ablat:.4f}")
print(f"Retention Delta                : {ppl_A_after - ppl_A_base:+.4f}")
print(f"Ablation Delta                 : {ppl_A_ablat - ppl_A_after:+.4f}")
print(f"--- Grafting ---")
print(f"Fresh Base PPL (no graft)      : {ppl_fresh_before:.4f}")
print(f"Fresh Base PPL (graft installed): {ppl_fresh_after:.4f}")
recovery = ppl_B_base - ppl_fresh_after
print(f"PPL gap vs trained model       : {recovery:+.4f}  ({'PASS' if abs(recovery) < 1.0 else 'INVESTIGATE'})")
print("="*60)