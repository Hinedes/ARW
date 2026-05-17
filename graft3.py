import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from torch.utils.data import DataLoader
import csv
import os

# ------------------------------
# Config: The Saturation Test
# ------------------------------
MODEL_ID = "Qwen/Qwen3.5-2B-Base"
H = 2048
K = 256 # Locked optimal container size
MASTER_SEED = 42
LR = 5e-6
MAX_STEPS = 5000      # Push it until it breaks
EVAL_INTERVAL = 100   # Check telemetry every 100 steps
BATCH_SIZE = 1
MAX_LEN = 128
EVAL_SPLIT = 0.9
LOG_FILE = "saturation_log.csv"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
print(f"Goal: Testing data saturation limits of a {K}-dimensional container.")

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
        DataLoader(ds['train'], batch_size=BATCH_SIZE, shuffle=True, drop_last=True),
        DataLoader(ds['test'], batch_size=BATCH_SIZE)
    )

train_B, eval_B = load_jsonl('rnelang.jsonl', tokenizer, MAX_LEN, EVAL_SPLIT)
print(f"Train batches: {len(train_B)}, Eval batches: {len(eval_B)}")

# ------------------------------
# 2. ARW Glass Generation
# ------------------------------
def make_arw_basis(hidden_dim, k, seed, device):
    g = torch.Generator(device=device).manual_seed(seed)
    G = torch.randn(hidden_dim, k, generator=g, device=device, dtype=torch.float32)
    Q, _ = torch.linalg.qr(G)
    return Q[:, :k]

P = make_arw_basis(H, K, MASTER_SEED, device)
Pi = P @ P.T

# ------------------------------
# 3. Load & Lock Down the Base
# ------------------------------
print("\n[1] Loading Base Model and locking down the warehouse...")
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.float32).to(device)
model = model.float()

# THE FIX: Absolute freeze on everything EXCEPT our targeted FFN matrices
trainable_params = 0
total_params = 0
for name, param in model.named_parameters():
    total_params += param.numel()
    if any(x in name for x in ['gate_proj', 'up_proj', 'down_proj']):
        param.requires_grad = True
        trainable_params += param.numel()
    else:
        param.requires_grad = False

print(f"    Total Parameters: {total_params:,}")
print(f"    Trainable Parameters: {trainable_params:,} ({(trainable_params/total_params)*100:.2f}%)")
print("    Attention layers and LayerNorms are strictly frozen.")

# ------------------------------
# 4. Evaluation Function
# ------------------------------
@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    total_loss = 0
    for batch in loader:
        ids = batch['input_ids'].to(device)
        am = batch['attention_mask'].to(device)
        total_loss += model(input_ids=ids, attention_mask=am, labels=ids).loss.item()
    avg_loss = total_loss / len(loader)
    return torch.exp(torch.tensor(avg_loss)).item()

# ------------------------------
# 5. The Endurance Training Loop
# ------------------------------
print(f"\n[2] Starting {MAX_STEPS} step Saturation Test...")
optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=LR, weight_decay=0.0)

# Initialize CSV Logger
with open(LOG_FILE, mode='w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['Step', 'Eval_PPL'])

# Initial Baseline
baseline_ppl = evaluate(model, eval_B)
print(f"    Step {0:4d} | Eval PPL: {baseline_ppl:.4f} (Baseline)")
with open(LOG_FILE, mode='a', newline='') as f:
    csv.writer(f).writerow([0, baseline_ppl])

model.train()
step = 1

# Loop continuously until MAX_STEPS is reached
while step <= MAX_STEPS:
    for batch in train_B:
        if step > MAX_STEPS:
            break
            
        ids = batch['input_ids'].to(device)
        am = batch['attention_mask'].to(device)
        
        optimizer.zero_grad()
        loss = model(input_ids=ids, attention_mask=am, labels=ids).loss
        
        # 1. Backward pass
        loss.backward()

        # 2. Snapshot weights before optimizer
        with torch.no_grad():
            old_weights = {
                n: p.clone() for n, p in model.named_parameters()
                if p.requires_grad
            }

        # 3. Optimizer steps (Adam tries to move outside the glass)
        optimizer.step()

        # 4. The Double-Tap: Smash Adam back against the glass
        with torch.no_grad():
            Pi_f32 = Pi.to(torch.float32)
            for name, param in model.named_parameters():
                if name in old_weights:
                    raw_update = (param - old_weights[name]).to(torch.float32)
                    if 'gate_proj' in name or 'up_proj' in name:
                        projected_update = raw_update @ Pi_f32
                    elif 'down_proj' in name:
                        projected_update = Pi_f32 @ raw_update
                    param.copy_(old_weights[name] + projected_update.to(param.dtype))

        # 5. Telemetry check
        if step % EVAL_INTERVAL == 0:
            ppl = evaluate(model, eval_B)
            print(f"    Step {step:4d} | Eval PPL: {ppl:.4f}")
            with open(LOG_FILE, mode='a', newline='') as f:
                csv.writer(f).writerow([step, ppl])
            model.train() # Set back to train after eval
            
        step += 1

print(f"\n[3] Test Complete. Saturation curve saved to {LOG_FILE}")
print("Read the CSV: Look for the 'U-Curve'. The step where PPL stops dropping and starts rising is the exact moment the container bursts.")