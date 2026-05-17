import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from torch.utils.data import DataLoader
import gc

# ------------------------------
# Config: The Squeeze Test
# ------------------------------
MODEL_ID = "Qwen/Qwen3.5-2B-Base"
H = 2048
K_VALUES = [16, 32, 64, 128, 256, 512, 1024] # The container sizes we are testing
MASTER_SEED = 42
LR = 5e-6
MAX_STEPS = 200 
BATCH_SIZE = 1
MAX_LEN = 128
EVAL_SPLIT = 0.9

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# ------------------------------
# 1. Tokenizer & Data
# ------------------------------
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

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
# 2. ARW Core Functions
# ------------------------------
def make_arw_basis(hidden_dim, k, seed, device):
    assert k <= hidden_dim
    g = torch.Generator(device=device).manual_seed(seed)
    G = torch.randn(hidden_dim, k, generator=g, device=device, dtype=torch.float32)
    Q, _ = torch.linalg.qr(G)
    return Q[:, :k]

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

def train_domain(model, loader, opt, Pi_write_f32, steps):
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

        # The Double-Tap: Hard clamp after Adam
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

@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    total = 0
    for batch in loader:
        ids = batch['input_ids'].to(device)
        am = batch['attention_mask'].to(device)
        total += model(input_ids=ids, attention_mask=am, labels=ids).loss.item()
    avg = total / len(loader)
    ppl = torch.exp(torch.tensor(avg)).item()
    return ppl

@torch.no_grad()
def extract_graft(model, base_weights, P_f32): # Notice we pass P, not Pi
    graft = {}
    for name, param in model.named_parameters():
        if name not in base_weights or param.ndim != 2:
            continue
        if not any(x in name for x in ['gate_proj', 'up_proj', 'down_proj']):
            continue

        delta = (param.to(torch.float32) - base_weights[name].to(device).to(torch.float32))

        # Smash it down to K dimensions!
        if 'gate_proj' in name or 'up_proj' in name:
            compressed_graft = delta @ P_f32
        elif 'down_proj' in name:
            compressed_graft = P_f32.T @ delta

        graft[name] = compressed_graft.cpu()
    
    size_mb = sum(v.numel() * 4 for v in graft.values()) / (1024 ** 2)
    return graft, size_mb

# ==============================
# EXPERIMENT 1: THE SQUEEZE TEST
# ==============================

results_table = []

for current_k in K_VALUES:
    print(f"\n{'='*60}")
    print(f"TESTING GRAFT CAPACITY: K = {current_k}")
    print(f"{'='*60}")
    
    # Generate the glass for this K
    P = make_arw_basis(H, current_k, MASTER_SEED, device)
    Pi = P @ P.T
    
    # 1. Load Anchor Model
    print("[1] Loading Base Model...")
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.float32).to(device)
    model = model.float()
    for param in model.parameters():
        param.requires_grad = True
    
    base_weights = {
        n: p.clone().detach().cpu()
        for n, p in model.named_parameters()
        if p.requires_grad and p.ndim == 2
        and any(x in n for x in ['gate_proj', 'up_proj', 'down_proj'])
    }
    
    ppl_zero = evaluate(model, eval_B)
    print(f"    Untrained Base PPL: {ppl_zero:.4f}")

    # 2. Train with ARW
    print(f"[2] Training Domain in K={current_k} subspace for {MAX_STEPS} steps...")
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.0)
    arw = ARWManager(H, device)
    arw.attach(Pi, model)
    
    train_domain(model, train_B, optimizer, Pi, MAX_STEPS)
    arw.clear_hooks()
    
    ppl_trained = evaluate(model, eval_B)
    print(f"    Trained Model PPL: {ppl_trained:.4f}")

    # 3. Extract Graft
    print("[3] Extracting Graft Artifact...")
    graft, size_mb = extract_graft(model, base_weights, P) # Changed Pi to P
    print(f"    Graft Size: {size_mb:.2f} MB")

    # 4. Clean up VRAM (Critical for loops)
    del model, optimizer, arw
    torch.cuda.empty_cache()
    gc.collect()

    # 5. Load Fresh Instance & Install
    print("[4] Loading Fresh Base & Installing Graft...")
    fresh = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.float32).to(device)
    fresh = fresh.float()
    fresh.eval()

    installed = 0
    with torch.no_grad():
        P_f32 = P.to(torch.float32).to(device)
        for name, param in fresh.named_parameters():
            if name in graft:
                # Unpack from K dimensions back to H dimensions
                if 'gate_proj' in name or 'up_proj' in name:
                    unpacked = graft[name].to(device).to(torch.float32) @ P_f32.T
                elif 'down_proj' in name:
                    unpacked = P_f32 @ graft[name].to(device).to(torch.float32)
                
                param.add_(unpacked.to(param.dtype))
                installed += 1

    ppl_grafted = evaluate(fresh, eval_B)
    print(f"    Grafted Model PPL: {ppl_grafted:.4f}")

    # Record Results
    results_table.append({
        'K': current_k,
        'PPL_Zero': ppl_zero,
        'PPL_Trained': ppl_trained,
        'PPL_Grafted': ppl_grafted,
        'Size_MB': size_mb
    })

    # Final sweep before next K
    del fresh, graft, base_weights
    torch.cuda.empty_cache()
    gc.collect()

# ==============================
# FINAL VERDICT TABLE
# ==============================
print("\n" + "="*60)
print("EXPERIMENT 1: SQUEEZE TEST RESULTS")
print("="*60)
print(f"{'K (Dims)':<10} | {'Base PPL':<10} | {'Trained PPL':<12} | {'Graft PPL':<10} | {'Size (MB)':<10}")
print("-" * 60)
for r in results_table:
    print(f"{r['K']:<10} | {r['PPL_Zero']:<10.4f} | {r['PPL_Trained']:<12.4f} | {r['PPL_Grafted']:<10.4f} | {r['Size_MB']:<10.2f}")
print("="*60)
print("Look for the 'elbow' in the Graft PPL column.")
print("The lowest K before the PPL spikes is your optimal container size.")