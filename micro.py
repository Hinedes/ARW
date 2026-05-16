import torch
from transformers import AutoModelForCausalLM

# 1. ATOMIC SETUP
device = "cuda"
print("Loading Model (Strictly mapping to CUDA, NO Accelerate)...")

# Use dtype instead of torch_dtype, and apply the .float() sledgehammer
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3.5-2B-Base", 
    dtype=torch.float32  
).to(device)

# THE SLEDGEHAMMER: Force the entire model tree into 32-bit float memory
model = model.float()

layer = model.model.layers[0].mlp.gate_proj
print(f"\nTarget Layer: gate_proj | Shape: {layer.weight.shape} | Dtype: {layer.weight.dtype}")

if layer.weight.dtype != torch.float32:
    print("FATAL: Hugging Face defeated the sledgehammer. Aborting.")
    exit()

# 2. GENERATE SUBSPACE (H=2048, K=64)
H = layer.weight.shape[1]
K = 64
torch.manual_seed(42)
G = torch.randn(H, K, device=device, dtype=torch.float32)
Q, _ = torch.linalg.qr(G)
Pi = Q @ Q.T  # (2048, 2048) Exact float32 orthogonal projector

# 3. SNAPSHOT W_0
W_0 = layer.weight.clone().detach()

# 4. ONE OPTIMIZER STEP (LR 1e-3 to force a massive, measurable movement)
optimizer = torch.optim.AdamW([layer.weight], lr=1e-3)

dummy_input = torch.randint(0, 1000, (1, 16), device=device)
loss = model(dummy_input, labels=dummy_input).loss
loss.backward()
optimizer.step()

# 5. MEASURE RAW ADAM UPDATE
W_raw = layer.weight.clone().detach()
Delta_raw = (W_raw - W_0).to(torch.float32)
print(f"\n[PHASE 1] Norm of Delta_raw (Adam's unconstrained step): {torch.norm(Delta_raw):.6f}")

# 6. MANUALLY PROJECT THE UPDATE
# gate_proj is (I, H) -> (5632, 2048). H is cols. Right-multiply by Pi (2048, 2048).
Delta_proj = Delta_raw @ Pi
print(f"[PHASE 2] Norm of Delta_proj (Mathematically locked):   {torch.norm(Delta_proj):.6f}")

# Sanity Check the Math
math_in_subspace = Delta_proj @ Pi
math_out_subspace = Delta_proj - math_in_subspace
print(f"[PHASE 3] Pure Math Out-of-Subspace Ratio: {torch.norm(math_out_subspace) / (torch.norm(Delta_proj) + 1e-10):.8f} (Must be 0.0000)")

# 7. THE WRITE COMMIT
print("\nExecuting param.copy_()...")
layer.weight.data.copy_(W_0 + Delta_proj)

# 8. THE FINAL AUTOPSY
W_final = layer.weight.clone().detach()
Delta_actual = (W_final - W_0).to(torch.float32)

final_in_subspace = Delta_actual @ Pi
final_out_subspace = Delta_actual - final_in_subspace
ratio = torch.norm(final_out_subspace) / (torch.norm(Delta_actual) + 1e-10)

print(f"\n[PHASE 4] Physical Tensor Out-of-Subspace Ratio: {ratio.item():.8f}")

if ratio > 0.01:
    print("❌ FAILURE: PyTorch corrupted the write. The tensor did not retain the projection.")
else:
    print("✅ SUCCESS: The write-lock held. The GPU physical tensor matches the pure math.")