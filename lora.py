import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from torch.utils.data import DataLoader
import os
import copy

MODEL_ID = "Qwen/Qwen3.5-2B-Base"
H = 2048
K = 64
MASTER_SEED = 42
LR = 5e-6
MAX_STEPS = 1000
BATCH_SIZE = 1
MAX_LEN = 128
EVAL_SPLIT = 0.9

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# ------------------------------
# 1. Model & tokenizer (float32)
# ------------------------------
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print("Loading Model in strict float32...")
base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, dtype=torch.float32
).to(device)

# ------------------------------
# 2. Orthogonal basis pair (float32)
# ------------------------------
def make_arw_basis_pair(hidden_dim, k, seed, device):
    assert 2 * k <= hidden_dim
    g = torch.Generator(device=device).manual_seed(seed)
    G = torch.randn(hidden_dim, 2 * k, generator=g, device=device, dtype=torch.float32)
    Q, _ = torch.linalg.qr(G)
    return Q[:, :k], Q[:, k:2*k]

P_A, P_B = make_arw_basis_pair(H, K, MASTER_SEED, device)
Pi_A = P_A @ P_A.T
Pi_B = P_B @ P_B.T

# ------------------------------
# 3. ARW Manager (unchanged from Step 1)
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

# ------------------------------
# 4. Data loaders
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
# 5. Training & eval functions
# ------------------------------
def train_domain_arw(model, loader, opt, desc, steps, Pi_write_f32):
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
            for name, param in model.named_parameters():
                if name in old_weights:
                    raw_update = (param - old_weights[name]).to(torch.float32)
                    if 'gate_proj' in name or 'up_proj' in name:
                        projected = raw_update @ Pi_write_f32
                    elif 'down_proj' in name:
                        projected = Pi_write_f32 @ raw_update
                    else:
                        continue
                    param.copy_(old_weights[name] + projected.to(model.dtype))
        if step % 40 == 0:
            print(f"{desc} | Step {step:3d} | Loss: {loss.item():.4f}")

def train_domain_vanilla(model, loader, opt, desc, steps):
    model.train()
    for step, batch in enumerate(loader):
        if step >= steps: break
        ids = batch['input_ids'].to(device)
        am = batch['attention_mask'].to(device)
        opt.zero_grad()
        loss = model(input_ids=ids, attention_mask=am, labels=ids).loss
        loss.backward()
        opt.step()
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
# 6. LoRA (rank=64, custom merger)
# ------------------------------
class LoRALinear(nn.Module):
    def __init__(self, in_features, out_features, rank=64, lora_alpha=1.0):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=False)
        self.lora_A = nn.Parameter(torch.zeros(in_features, rank))
        self.lora_B = nn.Parameter(torch.zeros(rank, out_features))
        self.rank = rank
        self.lora_alpha = lora_alpha
        nn.init.normal_(self.lora_A, std=0.02)
        nn.init.zeros_(self.lora_B)

    def forward(self, x):
        return self.linear(x) + (x @ self.lora_A @ self.lora_B) * self.lora_alpha

    def merge(self):
        self.linear.weight.data += (self.lora_A @ self.lora_B).T * self.lora_alpha
        self.lora_A.requires_grad = False
        self.lora_B.requires_grad = False

def apply_lora_to_mlp(model, rank=64):
    replacements = {}
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and ('gate_proj' in name or 'up_proj' in name or 'down_proj' in name):
            parent = model
            for attr in name.split('.')[:-1]:
                parent = getattr(parent, attr)
            leaf = name.split('.')[-1]
            lora = LoRALinear(module.in_features, module.out_features, rank=rank)
            lora.linear.weight.data.copy_(module.weight.data)
            if module.bias is not None:
                lora.linear.bias.data.copy_(module.bias.data)
            setattr(parent, leaf, lora)
            replacements[name] = (module, lora)
    return replacements

def merge_lora_and_restore(model, replacements):
    for name, (orig, lora) in replacements.items():
        lora.merge()
        orig.weight.data = lora.linear.weight.data
        if orig.bias is not None:
            orig.bias.data = lora.linear.bias.data
    # Restore original linear modules
    for name, (orig, _) in replacements.items():
        parent = model
        for attr in name.split('.')[:-1]:
            parent = getattr(parent, attr)
        leaf = name.split('.')[-1]
        setattr(parent, leaf, orig)

# ------------------------------
# 7. Step 2 Protocol
# ------------------------------
# --- Train Domain A with Full ARW, save checkpoint ---
model = copy.deepcopy(base_model)
for param in model.parameters():
    param.requires_grad = True
optimizer_A = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.0)

arw = ARWManager(H, device)
print("\n>>> STAGE 1: Train Domain A (Full ARW)")
arw.set_active_domain(Pi_A, model)
train_domain_arw(model, train_A, optimizer_A, "Domain A", MAX_STEPS, Pi_A)
ppl_A_baseline = evaluate(model, eval_A, "Domain A (Baseline)")

torch.save(model.state_dict(), "post_A_checkpoint.pt")
print("Post-A checkpoint saved.")
del model, optimizer_A
torch.cuda.empty_cache()

results = {}

# --- Condition 1: Vanilla ---
print("\n\n>>> CONDITION 1: Vanilla Fine-Tuning")
model = copy.deepcopy(base_model)
model.load_state_dict(torch.load("post_A_checkpoint.pt"))
for param in model.parameters():
    param.requires_grad = True
optimizer_vanilla = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.0)

train_domain_vanilla(model, train_B, optimizer_vanilla, "Vanilla B", MAX_STEPS)
ppl_A_vanilla = evaluate(model, eval_A, "Domain A (Vanilla after B)")
results['Vanilla'] = ppl_A_vanilla
del model, optimizer_vanilla
torch.cuda.empty_cache()

# ------------------------------
# Updated LoRA Merge Functions
# ------------------------------
def remove_lora(model, replacements):
    """Strips the LoRA wrappers and puts the original Linear layers back."""
    for name, (orig, lora) in replacements.items():
        parent = model
        for attr in name.split('.')[:-1]:
            parent = getattr(parent, attr)
        leaf = name.split('.')[-1]
        setattr(parent, leaf, orig)

def merge_lora_and_restore(model, replacements):
    """Merges LoRA weights and safely restores original layers without dtype crashes."""
    # 1. Execute the mathematical merge inside the wrapper
    for _, (_, lora) in replacements.items():
        lora.merge()
        
    # 2. Transfer the merged weights back to the original layer.
    # THE FIX: Explicitly cast back to the original layer's exact dtype.
    for name, (orig, lora) in replacements.items():
        orig.weight.data = lora.linear.weight.data.to(orig.weight.dtype)
        if orig.bias is not None:
            orig.bias.data = lora.linear.bias.data.to(orig.bias.dtype)
            
    # 3. Swap the modules back
    remove_lora(model, replacements)

# ------------------------------
# Condition 2: LoRA Execution Block
# ------------------------------
print("\n\n>>> CONDITION 2: LoRA (rank=64)")
model = copy.deepcopy(base_model)
model.load_state_dict(torch.load("post_A_checkpoint.pt"))

# SLEDGEHAMMER SAFETY: Force physical 32-bit memory just in case the state_dict reverted anything
model = model.float()

for param in model.parameters():
    param.requires_grad = False

# Apply custom LoRA with float32 guarantee
lora_replacements = apply_lora_to_mlp(model, rank=64)
model = model.to(device)
optimizer_lora = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.0)

# Override LoRA forward to explicitly map all math to float32
for _, (_, lora_module) in lora_replacements.items():
    lora_module.forward = lambda x, m=lora_module: (
        m.linear(x.to(torch.float32)) +
        (x.to(torch.float32) @ m.lora_A.to(torch.float32) @ m.lora_B.to(torch.float32)) * m.lora_alpha
    ).to(x.dtype)

# Train the adapter
train_domain_vanilla(model, train_B, optimizer_lora, "LoRA B", MAX_STEPS)

# Merge the adapter back into the base weights securely
merge_lora_and_restore(model, lora_replacements)

# Evaluate (This will no longer crash)
ppl_A_lora = evaluate(model, eval_A, "Domain A (LoRA after B)")
results['LoRA (r=64)'] = ppl_A_lora

del model, optimizer_lora
torch.cuda.empty_cache()

# --- Condition 3: Write-only ARW (no forward blind) ---
print("\n\n>>> CONDITION 3: Write-only ARW (No Forward Blind)")
model = copy.deepcopy(base_model)
model.load_state_dict(torch.load("post_A_checkpoint.pt"))
for param in model.parameters():
    param.requires_grad = True
optimizer_wonly = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.0)

arw.set_active_domain(Pi_B, model, Pi_other=None)   # No blind
train_domain_arw(model, train_B, optimizer_wonly, "Write-only B", MAX_STEPS, Pi_B)
# Evaluate A with proper Read Gate: shield A from B's subspace
arw.set_active_domain(Pi_A, model, Pi_other=Pi_B)
ppl_A_writeonly = evaluate(model, eval_A, "Domain A (Write-only ARW)")
results['Write-only ARW'] = ppl_A_writeonly
arw.clear_hooks()
del model, optimizer_wonly
torch.cuda.empty_cache()

# --- Condition 4: Full ARW (forward blind) ---
print("\n\n>>> CONDITION 4: Full ARW (Forward Blind)")
model = copy.deepcopy(base_model)
model.load_state_dict(torch.load("post_A_checkpoint.pt"))
for param in model.parameters():
    param.requires_grad = True
optimizer_full = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.0)

arw.set_active_domain(Pi_B, model, Pi_other=Pi_A)
train_domain_arw(model, train_B, optimizer_full, "Full ARW B", MAX_STEPS, Pi_B)
# Evaluate A with A's Read Gate, blinding B
arw.set_active_domain(Pi_A, model, Pi_other=Pi_B)
ppl_A_fullarw = evaluate(model, eval_A, "Domain A (Full ARW)")
results['Full ARW'] = ppl_A_fullarw
arw.clear_hooks()
torch.cuda.empty_cache()

# ------------------------------
# 8. Final Table
# ------------------------------
print("\n" + "="*70)
print("STEP 2 RESULTS: Catastrophic Forgetting Prevention")
print("="*70)
print(f"Domain A Baseline PPL: {ppl_A_baseline:.4f}\n")
print(f"{'Condition':<20} {'Domain A PPL':>15} {'Forgetting Δ':>15}")
print("-"*50)
for cond, ppl in results.items():
    delta = ppl - ppl_A_baseline
    print(f"{cond:<20} {ppl:>15.4f} {delta:>+15.4f}")