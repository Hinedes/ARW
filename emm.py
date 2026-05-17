import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import os
import json

# ==========================================
# EMM OPERATING SYSTEM: THE MASTER LEDGER
# ==========================================
MODEL_ID = "Qwen/Qwen3.5-2B-Base"
H = 2048
K = 256
MAX_SLOTS = H // K  # 2048 / 256 = 8 Slots
MASTER_SEED = 42    # NEVER change this for a model family. This IS the warehouse.
LEDGER_FILE = "emm_ledger.json"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_master_blueprint(hidden_dim, seed, device):
    """Generates the absolute foundation of the model. Run once."""
    print(f"[OS] Generating {hidden_dim}x{hidden_dim} Master Blueprint (Seed: {seed})...")
    g = torch.Generator(device=device).manual_seed(seed)
    G = torch.randn(hidden_dim, hidden_dim, generator=g, device=device, dtype=torch.float32)
    Master_Q, _ = torch.linalg.qr(G)
    return Master_Q

def get_slot(master_blueprint, k, slot_index):
    """Cuts out a strict K-dimensional room from the blueprint."""
    if slot_index >= MAX_SLOTS:
        raise ValueError(f"CRITICAL ERROR: Slot {slot_index} exceeds max capacity of {MAX_SLOTS - 1}")
    
    start_col = slot_index * k
    end_col = start_col + k
    print(f"[OS] Assigning Slot {slot_index} (Columns {start_col} to {end_col-1})")
    return master_blueprint[:, start_col:end_col]

# --- Ledger Management ---
def load_ledger():
    if os.path.exists(LEDGER_FILE):
        with open(LEDGER_FILE, 'r') as f:
            return json.load(f)
    # Initialize empty warehouse
    return {str(i): "EMPTY" for i in range(MAX_SLOTS)}

def save_ledger(ledger):
    with open(LEDGER_FILE, 'w') as f:
        json.dump(ledger, f, indent=4)

def print_warehouse_status(ledger):
    print("\n" + "="*40)
    print(f"WAREHOUSE STATUS (Capacity: {MAX_SLOTS} Domains)")
    print("="*40)
    for i in range(MAX_SLOTS):
        status = ledger[str(i)]
        print(f"Slot {i}: {status}")
    print("="*40 + "\n")

# ==========================================
# EXECUTION (Simulating a Multi-Domain Build)
# ==========================================
if __name__ == "__main__":
    # 1. Boot up the OS
    ledger = load_ledger()
    print_warehouse_status(ledger)
    
    # Generate the physical grid
    MASTER_BLUEPRINT = get_master_blueprint(H, MASTER_SEED, device)
    
    # 2. Check for an empty slot
    target_slot = -1
    for i in range(MAX_SLOTS):
        if ledger[str(i)] == "EMPTY":
            target_slot = i
            break
            
    if target_slot == -1:
        print("[OS] FATAL: Warehouse is full. You must delete a domain to proceed.")
        exit()
        
    domain_name = input(f"Enter name for new domain in Slot {target_slot} (e.g., Kyrgyz_Law): ")
    
    # 3. Get the physical glass for this specific slot
    P_slot = get_slot(MASTER_BLUEPRINT, K, target_slot)
    Pi_slot = P_slot @ P_slot.T
    
    # 4. (Training happens here, using Pi_slot exactly as you did in graft3.py)
    print(f"\n[OS] -> Launching training protocol for {domain_name} in Slot {target_slot}...")
    print(f"[OS] -> Clamping gradients strictly to Pi_slot geometry...")
    # ... your training loop ...
    
    # 5. Extraction (Using P_slot to compress it to 256 dims)
    print(f"\n[OS] -> Extracting {K}-dimensional Graft Artifact...")
    # graft, size = extract_graft(model, base_weights, P_slot)
    
    # 6. Save and Register
    graft_filename = f"graft_slot{target_slot}_{domain_name}.pt"
    print(f"[OS] -> Saved artifact to {graft_filename}")
    
    # Update the OS ledger
    ledger[str(target_slot)] = domain_name
    save_ledger(ledger)
    
    print_warehouse_status(ledger)