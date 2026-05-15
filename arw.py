import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from torch.utils.data import DataLoader
import os

# --- Configuration ---
MODEL_ID = "Qwen/Qwen3.5-2B-Base"
K_DIM = 64 # Grassmannian subspace dimension
DOMAIN_A_SEED = 1
DOMAIN_B_SEED = 2
LR = 5e-5
EPOCHS = 1
BATCH_SIZE = 4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu") 

# --- 1. PRNG Basis Generator (Zero Parameter Overhead) ---
def generate_basis(hidden_dim, k, seed, device):
    """Generates a deterministic orthogonal basis P_d using a seed."""
    torch.manual_seed(seed)
    G = torch.randn(hidden_dim, k, device=device, dtype=torch.float32)
    Q, _ = torch.linalg.qr(G)
    return Q # Shape: (H, k)

# --- 2. The ARW Autograd Hook Manager ---
class ARWHookManager:
    def __init__(self, hidden_dim, k_dim, device):
        self.hidden_dim = hidden_dim
        self.k_dim = k_dim
        self.device = device
        self.active_seed = None
        self.hooks = []

    def set_active_domain(self, seed, model):
        """Swaps the active projection subspace and registers new hooks."""
        self.clear_hooks()
        self.active_seed = seed
        
        # Generate basis on the fly
        P = generate_basis(self.hidden_dim, self.k_dim, seed, self.device)
        Pi = P @ P.T # Projection operator: (H, H)
        Pi = Pi.to(model.dtype) # Match model precision

        # Create closures for the hooks to handle matrix dimension routing
        def make_hook(is_down_proj):
            def hook(param):
                if param.grad is not None:
                    if is_down_proj:
                        # param is (H, I), Pi is (H, H)
                        param.grad.data = torch.matmul(Pi, param.grad.data)
                    else:
                        # param is (I, H), Pi is (H, H)
                        param.grad.data = torch.matmul(param.grad.data, Pi)
            return hook

        # Inject hooks strictly into MLP layers
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                if 'gate_proj' in name or 'up_proj' in name:
                    hook_handle = module.weight.register_post_accumulate_grad_hook(make_hook(is_down_proj=False))
                    self.hooks.append(hook_handle)
                elif 'down_proj' in name:
                    hook_handle = module.weight.register_post_accumulate_grad_hook(make_hook(is_down_proj=True))
                    self.hooks.append(hook_handle)
                    
        print(f"[ARW] Hooks injected for Domain Seed {seed}. Subspace locked.")

    def clear_hooks(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

# --- 3. Data Pipeline ---
def prepare_dataloader(jsonl_path, tokenizer, batch_size):
    if not os.path.exists(jsonl_path):
        raise FileNotFoundError(f"Missing data file: {jsonl_path}")
    
    dataset = load_dataset('json', data_files=jsonl_path, split='train')
    
    def tokenize(examples):
        text_col = 'text' if 'text' in examples else list(examples.keys())[0]
        return tokenizer(examples[text_col], truncation=True, max_length=512, padding="max_length")

    tokenized = dataset.map(tokenize, batched=True, remove_columns=dataset.column_names)
    tokenized.set_format('torch')
    return DataLoader(tokenized, batch_size=batch_size, shuffle=True)

# --- 4. Training & Eval Engines ---
def train_domain(model, dataloader, optimizer, epoch_desc):
    model.train()
    total_loss = 0
    for batch_idx, batch in enumerate(dataloader):
        input_ids = batch['input_ids'].to(DEVICE)
        attention_mask = batch['attention_mask'].to(DEVICE)
        labels = input_ids.clone()
        
        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        
        # The ARW hook automatically intercepts param.grad here before the step
        optimizer.step()
        
        total_loss += loss.item()
        if batch_idx % 10 == 0:
            print(f"{epoch_desc} | Step {batch_idx} | Loss: {loss.item():.4f}")

def evaluate_domain(model, dataloader, eval_desc):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            labels = input_ids.clone()
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            total_loss += outputs.loss.item()
    
    avg_loss = total_loss / len(dataloader)
    ppl = torch.exp(torch.tensor(avg_loss)).item()
    print(f"=== {eval_desc} | PPL: {ppl:.4f} | Loss: {avg_loss:.4f} ===")
    return ppl

# --- 5. Main Execution Protocol ---
def run_experiment():
    print("Loading Qwen3.5-2B-Base to MI300X...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16).to(DEVICE)
    hidden_dim = model.config.hidden_size 
    
    # Initialize Hook Manager
    arw_manager = ARWHookManager(hidden_dim, K_DIM, DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    
    # Load Data
    print("Loading TRC Datasets...")
    loader_A = prepare_dataloader('train.jsonl', tokenizer, BATCH_SIZE)
    loader_B = prepare_dataloader('rnelang.jsonl', tokenizer, BATCH_SIZE)
    
    # --- STAGE 1: Train Domain A ---
    print("\n>>> STAGE 1: Training Domain A (train.jsonl)")
    arw_manager.set_active_domain(DOMAIN_A_SEED, model)
    train_domain(model, loader_A, optimizer, "Domain A")
    ppl_A_initial = evaluate_domain(model, loader_A, "Domain A (Post-Own Training)")
    
    # --- STAGE 2: Train Domain B ---
    print("\n>>> STAGE 2: Training Domain B (rnelang.jsonl)")
    arw_manager.set_active_domain(DOMAIN_B_SEED, model)
    train_domain(model, loader_B, optimizer, "Domain B")
    ppl_B_initial = evaluate_domain(model, loader_B, "Domain B (Post-Own Training)")
    
    # --- STAGE 3: The Zero-Interference Benchmark ---
    print("\n>>> STAGE 3: The Moment of Truth (Evaluating A after B)")
    arw_manager.clear_hooks()
    ppl_A_final = evaluate_domain(model, loader_A, "Domain A (Post-Domain B Training)")
    
    # --- Results ---
    print("\n" + "="*50)
    print("ARW / OSA CONTINUAL LEARNING BENCHMARK RESULTS")
    print("="*50)
    print(f"Domain A Baseline PPL : {ppl_A_initial:.4f}")
    print(f"Domain B Baseline PPL : {ppl_B_initial:.4f}")
    print(f"Domain A Final PPL    : {ppl_A_final:.4f}")
    delta = ppl_A_final - ppl_A_initial
    print(f"Retention Delta       : {delta:+.4f}")

if __name__ == "__main__":
    run_experiment()