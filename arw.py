import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from torch.utils.data import DataLoader
import os

# --- Configuration ---
MODEL_ID = "Qwen/Qwen3.5-2B-Base"
K_DIM = 4  # CRUSHED CAPACITY: Forced to read the base model.
DOMAIN_A_SEED = 1
DOMAIN_B_SEED = 2
LR = 1e-4  # Slightly higher LR to push through the tight bottleneck
EPOCHS = 3
BATCH_SIZE = 4
PATIENCE = 2 # Early stopping tolerance
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu") 

# --- 1. PRNG Basis Generator ---
def generate_basis(hidden_dim, k, seed, device):
    torch.manual_seed(seed)
    G = torch.randn(hidden_dim, k, device=device, dtype=torch.float32)
    Q, _ = torch.linalg.qr(G)
    return Q 

# --- 2. The ARW Autograd Hook Manager ---
class ARWHookManager:
    def __init__(self, hidden_dim, k_dim, device):
        self.hidden_dim = hidden_dim
        self.k_dim = k_dim
        self.device = device
        self.active_seed = None
        self.hooks = []

    def set_active_domain(self, seed, model):
        self.clear_hooks()
        self.active_seed = seed
        
        P = generate_basis(self.hidden_dim, self.k_dim, seed, self.device)
        Pi = P @ P.T 
        Pi = Pi.to(model.dtype) 

        def make_hook(is_down_proj):
            def hook(param):
                if param.grad is not None:
                    if is_down_proj:
                        param.grad.data = torch.matmul(Pi, param.grad.data)
                    else:
                        param.grad.data = torch.matmul(param.grad.data, Pi)
            return hook

        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                if 'gate_proj' in name or 'up_proj' in name:
                    hook_handle = module.weight.register_post_accumulate_grad_hook(make_hook(is_down_proj=False))
                    self.hooks.append(hook_handle)
                elif 'down_proj' in name:
                    hook_handle = module.weight.register_post_accumulate_grad_hook(make_hook(is_down_proj=True))
                    self.hooks.append(hook_handle)
                    
        print(f"[ARW] Subspace Locked for Seed {seed} | Dimensions: {self.k_dim}x{self.k_dim}")

    def clear_hooks(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

# --- 3. Data Pipeline (Strict 80/20 Split) ---
def prepare_dataloaders(jsonl_path, tokenizer, batch_size):
    if not os.path.exists(jsonl_path):
        raise FileNotFoundError(f"Missing data file: {jsonl_path}")
    
    # Load and explicitly split 80/20 to prevent memorization masking
    dataset = load_dataset('json', data_files=jsonl_path, split='train')
    split_dataset = dataset.train_test_split(test_size=0.2, seed=42)
    
    def tokenize(examples):
        text_col = 'text' if 'text' in examples else list(examples.keys())[0]
        return tokenizer(examples[text_col], truncation=True, max_length=512, padding="max_length")

    train_tok = split_dataset['train'].map(tokenize, batched=True, remove_columns=split_dataset['train'].column_names)
    eval_tok = split_dataset['test'].map(tokenize, batched=True, remove_columns=split_dataset['test'].column_names)
    
    train_tok.set_format('torch')
    eval_tok.set_format('torch')
    
    train_loader = DataLoader(train_tok, batch_size=batch_size, shuffle=True)
    eval_loader = DataLoader(eval_tok, batch_size=batch_size, shuffle=False)
    
    return train_loader, eval_loader

# --- 4. Training & Eval Engines (With Early Stopping) ---
def evaluate_domain(model, dataloader):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
            total_loss += outputs.loss.item()
    
    avg_loss = total_loss / len(dataloader)
    ppl = torch.exp(torch.tensor(avg_loss)).item()
    return avg_loss, ppl

def train_domain(model, train_loader, eval_loader, optimizer, epoch_desc):
    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(EPOCHS):
        model.train()
        for batch_idx, batch in enumerate(train_loader):
            input_ids = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            
            if batch_idx % 10 == 0:
                print(f"{epoch_desc} | Ep {epoch+1} | Step {batch_idx} | Train Loss: {loss.item():.4f}")
        
        # End of Epoch Validation
        val_loss, val_ppl = evaluate_domain(model, eval_loader)
        print(f"--- {epoch_desc} Epoch {epoch+1} Validation | Val Loss: {val_loss:.4f} | Val PPL: {val_ppl:.4f} ---")
        
        # Brutal Early Stopping logic to kill overfitting
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            print(f"[WARNING] Validation loss degrading. Memorization detected. (Patience {patience_counter}/{PATIENCE})")
            if patience_counter >= PATIENCE:
                print("[ABORT] Early stopping triggered. Subspace capacity exhausted.")
                break

# --- 5. Main Execution Protocol ---
def run_experiment():
    print("Loading Qwen3.5-2B-Base to MI300X...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16).to(DEVICE)
    hidden_dim = model.config.hidden_size 
    
    arw_manager = ARWHookManager(hidden_dim, K_DIM, DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    
    print("Loading and Splitting TRC Datasets (80/20)...")
    train_A, eval_A = prepare_dataloaders('train.jsonl', tokenizer, BATCH_SIZE) # Kyrgyz Law
    train_B, eval_B = prepare_dataloaders('rnelang.jsonl', tokenizer, BATCH_SIZE) # RNE Logic
    
    # --- STAGE 1: Train Domain A ---
    print("\n>>> STAGE 1: Training Domain A (Kyrgyz Law)")
    arw_manager.set_active_domain(DOMAIN_A_SEED, model)
    train_domain(model, train_A, eval_A, optimizer, "Domain A")
    _, final_val_ppl_A = evaluate_domain(model, eval_A)
    
    # --- STAGE 2: Train Domain B ---
    print("\n>>> STAGE 2: Training Domain B (RNE Logic)")
    arw_manager.set_active_domain(DOMAIN_B_SEED, model)
    train_domain(model, train_B, eval_B, optimizer, "Domain B")
    _, final_val_ppl_B = evaluate_domain(model, eval_B)
    
    # --- STAGE 3: The Zero-Interference Benchmark ---
    print("\n>>> STAGE 3: The Moment of Truth (Evaluating A after B)")
    arw_manager.clear_hooks()
    _, isolated_ppl_A = evaluate_domain(model, eval_A)
    
    # --- Results ---
    print("\n" + "="*50)
    print("ARW / OSA CONTINUAL LEARNING BENCHMARK RESULTS")
    print("="*50)
    print(f"Domain A Baseline Val PPL : {final_val_ppl_A:.4f}")
    print(f"Domain B Baseline Val PPL : {final_val_ppl_B:.4f}")
    print(f"Domain A Post-B Val PPL   : {isolated_ppl_A:.4f}")
    delta = isolated_ppl_A - final_val_ppl_A
    print(f"Retention Delta           : {delta:+.4f}")

if __name__ == "__main__":
    run_experiment()