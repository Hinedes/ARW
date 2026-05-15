import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from torch.utils.data import DataLoader
import os

# ------------------------------
# Configuration
# ------------------------------
MODEL_ID = "google/gemma-4-4b"          # base model; adjust if needed
K_DIM = 64                               # subspace dimension
SEED_A = 1
SEED_B = 2
LR = 5e-6
EPOCHS = 1
BATCH_SIZE = 4
MAX_LEN = 512
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ------------------------------
# Seed‑based orthogonal basis generator
# ------------------------------
def generate_basis(hidden_dim, k, seed, device):
    """Deterministic orthonormal matrix P (H, k)."""
    g = torch.Generator(device=device).manual_seed(seed)
    G = torch.randn(hidden_dim, k, generator=g, device=device, dtype=torch.float32)
    Q, _ = torch.linalg.qr(G)
    return Q

# ------------------------------
# ARW Hook Manager
# ------------------------------
class ARWHookManager:
    def __init__(self, hidden_dim, k_dim, device):
        self.hidden_dim = hidden_dim
        self.k_dim = k_dim
        self.device = device
        self.Pi = None
        self.hooks = []

    def set_active_domain(self, seed, model):
        self.clear_hooks()
        P = generate_basis(self.hidden_dim, self.k_dim, seed, self.device)
        self.Pi = P @ P.T                           # projection matrix (H,H)
        self.Pi = self.Pi.to(torch.bfloat16)        # match model dtype

        # Inject hooks into MLP linear layers (gate, up, down)
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                if 'gate_proj' in name or 'up_proj' in name:
                    hook = module.weight.register_hook(
                        lambda grad, pi=self.Pi: grad @ pi
                    )
                    self.hooks.append(hook)
                elif 'down_proj' in name:
                    hook = module.weight.register_hook(
                        lambda grad, pi=self.Pi: pi @ grad
                    )
                    self.hooks.append(hook)

        print(f"[ARW] Domain seed {seed} active. Π shape: {self.Pi.shape}")

    def clear_hooks(self):
        for h in self.hooks:
            h.remove()
        self.hooks = []

# ------------------------------
# Data loading
# ------------------------------
def load_jsonl_dataset(jsonl_path, tokenizer, split_ratio=0.9):
    """Load JSONL file with a 'text' field, tokenize, split into train/eval."""
    dataset = load_dataset('json', data_files=jsonl_path, split='train')
    # Use the first text column found
    text_col = 'text' if 'text' in dataset.column_names else dataset.column_names[0]

    def tokenize(examples):
        return tokenizer(
            examples[text_col], truncation=True, max_length=MAX_LEN, padding="max_length"
        )

    tokenized = dataset.map(tokenize, batched=True, remove_columns=dataset.column_names)
    tokenized = tokenized.train_test_split(test_size=1-split_ratio) if split_ratio < 1.0 else tokenized
    train_ds = tokenized['train'] if split_ratio < 1.0 else tokenized
    eval_ds = tokenized['test'] if split_ratio < 1.0 else None

    train_ds.set_format('torch')
    if eval_ds:
        eval_ds.set_format('torch')
    return train_ds, eval_ds

# ------------------------------
# Training & evaluation loops
# ------------------------------
def train_one_epoch(model, loader, optimizer, desc):
    model.train()
    total_loss = 0
    for step, batch in enumerate(loader):
        input_ids = batch['input_ids'].to(DEVICE)
        attn_mask = batch['attention_mask'].to(DEVICE)
        labels = input_ids.clone()

        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, attention_mask=attn_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        # ARW hooks automatically mask gradients here
        optimizer.step()

        total_loss += loss.item()
        if step % 10 == 0:
            print(f"{desc} | Step {step:3d} | Loss: {loss.item():.4f}")
    return total_loss / len(loader)

@torch.no_grad()
def evaluate(model, loader, desc):
    model.eval()
    total_loss = 0
    for batch in loader:
        input_ids = batch['input_ids'].to(DEVICE)
        attn_mask = batch['attention_mask'].to(DEVICE)
        labels = input_ids.clone()
        outputs = model(input_ids=input_ids, attention_mask=attn_mask, labels=labels)
        total_loss += outputs.loss.item()
    avg_loss = total_loss / len(loader)
    ppl = torch.exp(torch.tensor(avg_loss)).item()
    print(f"=== {desc} | PPL: {ppl:.4f} | Loss: {avg_loss:.4f} ===")
    return ppl

# ------------------------------
# Main experiment
# ------------------------------
def main():
    print(f"Loading model {MODEL_ID} ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16, device_map="auto"
    )
    hidden_dim = model.config.hidden_size      # should be 2560 for 4B Gemma
    print(f"Hidden dim: {hidden_dim}")

    # Load data
    if not os.path.exists('train.jsonl') or not os.path.exists('rnelang.jsonl'):
        print("ERROR: Data files missing. Place train.jsonl and rnelang.jsonl in the current directory.")
        return

    train_A_ds, eval_A_ds = load_jsonl_dataset('train.jsonl', tokenizer)
    train_B_ds, eval_B_ds = load_jsonl_dataset('rnelang.jsonl', tokenizer)

    train_A_loader = DataLoader(train_A_ds, batch_size=BATCH_SIZE, shuffle=True)
    train_B_loader = DataLoader(train_B_ds, batch_size=BATCH_SIZE, shuffle=True)
    eval_A_loader = DataLoader(eval_A_ds, batch_size=BATCH_SIZE)
    eval_B_loader = DataLoader(eval_B_ds, batch_size=BATCH_SIZE)

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

    # ARW manager
    arw = ARWHookManager(hidden_dim, K_DIM, DEVICE)

    # -------------------- Domain A --------------------
    print("\n>>> STAGE 1: Training Domain A (Medical)")
    arw.set_active_domain(SEED_A, model)
    train_one_epoch(model, train_A_loader, optimizer, "Domain A")
    ppl_A_initial = evaluate(model, eval_A_loader, "Domain A (Post-Own Training)")
    ppl_A_initial_train = evaluate(model, DataLoader(train_A_ds, batch_size=BATCH_SIZE),
                                   "Domain A (train set)")

    # -------------------- Domain B --------------------
    print("\n>>> STAGE 2: Training Domain B (Legal)")
    arw.set_active_domain(SEED_B, model)
    train_one_epoch(model, train_B_loader, optimizer, "Domain B")
    ppl_B = evaluate(model, eval_B_loader, "Domain B (Post-Own Training)")

    # -------------------- Moment of Truth --------------------
    print("\n>>> STAGE 3: Evaluating Domain A again")
    arw.clear_hooks()  # no active projection during eval (just normal forward)
    ppl_A_final = evaluate(model, eval_A_loader, "Domain A (After Domain B training)")

    # -------------------- Results --------------------
    print("\n" + "="*50)
    print("ARW CONTINUAL LEARNING BENCHMARK (Gemma-4-E4B)")
    print("="*50)
    print(f"Domain A eval PPL (before B): {ppl_A_initial:.4f}")
    print(f"Domain B eval PPL :            {ppl_B:.4f}")
    print(f"Domain A eval PPL (after B) :  {ppl_A_final:.4f}")
    delta = ppl_A_final - ppl_A_initial
    print(f"Retention delta :              {delta:+.4f}")

    if abs(delta) < 0.05:
        print("✅ ARW achieved near‑zero forgetting.")
    else:
        print("⚠️  Forgetting detected – check hook implementation.")

if __name__ == "__main__":
    main()