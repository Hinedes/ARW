import torch
import torch.nn as nn
from transformers import AutoConfig, Gemma4ForConditionalGeneration
import copy

# ------------------------------
# Level 1 – ARW on PLE of a single Gemma‑4 layer
# ------------------------------

MODEL_ID = "google/gemma-4-E4B"
H = 2560          # hidden_size
K = 64
L = 256           # per_layer_projection output dim
SEED_A = 1
SEED_B = 2
LR = 0.01
STEPS = 200

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# --- 1. Load the full model config and weights ---
config = AutoConfig.from_pretrained(MODEL_ID)
model = Gemma4ForConditionalGeneration.from_pretrained(
    MODEL_ID, torch_dtype=torch.float32, device_map="auto"
)
# Extract first decoder layer
layer0 = model.model.decoder.layers[0]  # Gemma4DecoderLayer

# We only need the PLE components
ple_gate = layer0.per_layer_input_gate   # nn.Linear (H, 1) or similar? The config says "per_layer_input_gate" with size? Actually, PLE gate is a linear layer that modulates the PLE signal. We'll inspect its shape.
ple_proj = layer0.per_layer_projection   # nn.Linear (H, L) where L=256 (hidden_size_per_layer_input)

# Freeze everything in the layer except PLE params
for name, param in layer0.named_parameters():
    if 'per_layer' not in name:
        param.requires_grad = False
    else:
        param.requires_grad = True

# Move layer to device
layer0 = layer0.to(device)

# --- 2. Orthogonal bases (same as Level 0) ---
def make_basis(seed, other_basis=None):
    g = torch.Generator(device=device).manual_seed(seed)
    G = torch.randn(H, K, generator=g, device=device)
    if other_basis is not None:
        G = G - other_basis @ (other_basis.T @ G)
    Q, _ = torch.linalg.qr(G)
    return Q

P_A = make_basis(SEED_A)
P_B = make_basis(SEED_B, other_basis=P_A)
Pi_A = P_A @ P_A.T
Pi_B = P_B @ P_B.T
assert torch.allclose(P_A.T @ P_B, torch.zeros(K, K, device=device), atol=1e-5)

# --- 3. ARW hooks for PLE layers ---
# Both per_layer_input_gate and per_layer_projection are down‑proj style (output dim is smaller or 1).
# per_layer_input_gate: (1, H) weight
# per_layer_projection: (L, H) weight
# We apply the write mask to the weight gradient: grad @ Pi

def make_arw_hook(Pi):
    def hook(grad):
        return grad @ Pi
    return hook

# --- 4. Synthetic data for a single layer ---
# We need to feed a hidden state through the layer. The decoder layer forward expects (hidden_states, ...).
# We'll use a simple input that carries both domain signals.
torch.manual_seed(0)
x = torch.randn(4, 128, H, device=device)   # (batch, seq, hidden)

# Domain A and B targets: we want the PLE-modulated output of the layer to match a target that only lives in the respective subspace.
# We'll define a target based on the output of the layer when only the PLE is active.
# First, get baseline output with no PLE (set PLE to zero? Actually, the layer forward uses PLE internally; we can't easily zero it. Instead, we'll use the layer's forward with a frozen core and only PLE trainable, and define domain targets as the output after projecting the hidden state.

# Simpler approach: we add a small readout head after the layer output, and train that to regress to a domain-specific signal.
# The layer output is (batch, seq, H). We'll add a linear readout (H, L) that is also trainable with ARW? To isolate, we can keep the readout fixed and only train PLE.

# Actually, to prove isolation at the layer level, we can directly look at the PLE output (the sum of gate*projection added to the residual) and train that to match a target. But the layer output is mixed with attention and FFN.

# Better: freeze everything except PLE, and train the PLE parameters so that the final layer output, when projected onto a domain-specific subspace, matches a desired value.

# We'll define target_A as a random signal in subspace A: target_A = x @ P_A @ M_A.T (same as Level 0 but using the final layer output instead of a linear layer). Then we'll minimize the loss between the layer output projected onto subspace A and target_A. That way, only PLE can adapt because everything else is frozen.

# For that, we need a loss function: loss = mse( (layer_output @ Pi_A), target_A ).

# The layer output will be computed with the full layer forward, but only PLE weights get gradients.

# --- 5. Training step ---
def train_domain(layer, x, Pi, target, steps, desc):
    # Register hooks on PLE layers
    h1 = layer.per_layer_input_gate.weight.register_hook(make_arw_hook(Pi))
    h2 = layer.per_layer_projection.weight.register_hook(make_arw_hook(Pi))
    opt = torch.optim.SGD([p for p in layer.parameters() if p.requires_grad], lr=LR)
    loss_fn = nn.MSELoss()
    for i in range(steps):
        opt.zero_grad()
        # Forward pass through the decoder layer
        # The decoder layer expects a tuple: (hidden_states, attention_mask, position_embeddings, ...)
        # We'll create dummy arguments
        hidden_states = x
        # Create a dummy attention mask (all ones)
        attn_mask = torch.ones(x.shape[0], 1, 1, x.shape[1], device=device)
        # Position embeddings (we'll skip or use zeros)
        pos_emb = torch.zeros(1, x.shape[1], H, device=device)
        # The layer forward returns (hidden_states, ...)
        output = layer(hidden_states, attention_mask=attn_mask, position_embeddings=pos_emb)
        if isinstance(output, tuple):
            out = output[0]
        else:
            out = output
        # Project output onto domain subspace and compute loss
        out_proj = out @ Pi   # (batch, seq, H)
        loss = loss_fn(out_proj, target)
        loss.backward()
        opt.step()
        if i % 50 == 0:
            print(f"{desc} | Step {i:3d} | Loss: {loss.item():.6f}")
    h1.remove()
    h2.remove()
    return loss.item()

# Targets: we want out @ Pi_A to be a fixed random matrix, independent of domain B
target_A = torch.randn(4, 128, H, device=device) @ Pi_A
target_B = torch.randn(4, 128, H, device=device) @ Pi_B

print("\n=== Train Domain A (read A, write A on PLE) ===")
loss_A = train_domain(layer0, x, Pi_A, target_A, STEPS, "Domain A")
print(f"Domain A final loss: {loss_A:.6f}")

# Evaluate A retention
with torch.no_grad():
    out = layer0(x, attention_mask=torch.ones(4,1,1,128, device=device), position_embeddings=torch.zeros(1,128,H, device=device))[0]
    loss_A_init = nn.MSELoss()(out @ Pi_A, target_A).item()
print(f"Domain A retention loss (before B): {loss_A_init:.8f}")

print("\n=== Train Domain B ===")
loss_B = train_domain(layer0, x, Pi_B, target_B, STEPS, "Domain B")
print(f"Domain B final loss: {loss_B:.6f}")

# Evaluate A after B
with torch.no_grad():
    out = layer0(x, attention_mask=torch.ones(4,1,1,128, device=device), position_embeddings=torch.zeros(1,128,H, device=device))[0]
    loss_A_final = nn.MSELoss()(out @ Pi_A, target_A).item()
print(f"Domain A retention loss (after B): {loss_A_final:.8f}")
delta = loss_A_final - loss_A_init
print(f"Delta: {delta:+.8f}")

if abs(delta) < 1e-5:
    print("\n✅ PERFECT ISOLATION at decoder layer level via PLE.")
else:
    print(f"\n⚠️ Delta={delta} – check layer forward or PLE hook.")