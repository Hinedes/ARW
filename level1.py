import torch
import torch.nn as nn
from transformers import AutoConfig, Gemma4ForConditionalGeneration

MODEL_ID = "google/gemma-4-E4B"
H = 2560
K = 64
SEED_A = 1
SEED_B = 2
LR = 0.01
STEPS = 200

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# --- 1. Load model, but focus on the LANGUAGE MODEL only ---
config = AutoConfig.from_pretrained(MODEL_ID)
model = Gemma4ForConditionalGeneration.from_pretrained(
    MODEL_ID, torch_dtype=torch.float32, device_map="auto"
)

# The text backbone is inside model.model.language_model
text_model = model.model.language_model
print(f"Type of text_model: {type(text_model)}")
print("\nTop‑level children of text_model:")
for name, child in text_model.named_children():
    print(f"  {name}: {type(child)}")

# Look for decoder layers
if hasattr(text_model, 'layers'):
    layers = text_model.layers
elif hasattr(text_model, 'decoder') and hasattr(text_model.decoder, 'layers'):
    layers = text_model.decoder.layers
else:
    # scan for a ModuleList with >10 items
    layers = None
    for name, module in text_model.named_modules():
        if isinstance(module, nn.ModuleList) and len(module) > 10:
            layers = module
            print(f"Found layer ModuleList at: {name} ({len(module)} layers)")
            break

if layers is None:
    raise RuntimeError("Could not locate decoder layers inside language model.")

print(f"Number of layers: {len(layers)}")
layer0 = layers[0]

# --- 2. Verify PLE components exist ---
assert hasattr(layer0, 'per_layer_input_gate'), "per_layer_input_gate missing"
assert hasattr(layer0, 'per_layer_projection'), "per_layer_projection missing"
print(f"Layer type: {type(layer0)}")

# Freeze everything except PLE parameters
for name, param in layer0.named_parameters():
    if 'per_layer' not in name:
        param.requires_grad = False
    else:
        param.requires_grad = True
layer0 = layer0.to(device)

# --- 3. Orthogonal bases (same as before) ---
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

# --- 4. ARW hook ---
def make_arw_hook(Pi):
    def hook(grad):
        return grad @ Pi
    return hook

# --- 5. Dummy input ---
torch.manual_seed(0)
x = torch.randn(4, 128, H, device=device)
attn_mask = torch.ones(4, 1, 128, 128, device=device)
pos_emb = torch.zeros(1, 128, H, device=device)

# Quick forward pass test
try:
    out_test = layer0(x, attention_mask=attn_mask, position_embeddings=pos_emb)
    if isinstance(out_test, tuple): out_test = out_test[0]
    print(f"Test forward shape: {out_test.shape}")
except Exception as e:
    print(f"Forward with pos_emb failed: {e}")
    out_test = layer0(x, attention_mask=attn_mask)
    if isinstance(out_test, tuple): out_test = out_test[0]
    print(f"Fallback forward shape: {out_test.shape}")

# --- 6. Training function ---
def train_domain(layer, x, attn_mask, pos_emb, Pi, target, steps, desc):
    h1 = layer.per_layer_input_gate.weight.register_hook(make_arw_hook(Pi))
    h2 = layer.per_layer_projection.weight.register_hook(make_arw_hook(Pi))
    opt = torch.optim.SGD([p for p in layer.parameters() if p.requires_grad], lr=LR)
    loss_fn = nn.MSELoss()
    for i in range(steps):
        opt.zero_grad()
        try:
            out = layer(x, attention_mask=attn_mask, position_embeddings=pos_emb)
        except:
            out = layer(x, attention_mask=attn_mask)
        if isinstance(out, tuple): out = out[0]
        out_proj = out @ Pi
        loss = loss_fn(out_proj, target)
        loss.backward()
        opt.step()
        if i % 50 == 0:
            print(f"{desc} | Step {i:3d} | Loss: {loss.item():.6f}")
    h1.remove()
    h2.remove()
    return loss.item()

target_A = torch.randn(4, 128, H, device=device) @ Pi_A
target_B = torch.randn(4, 128, H, device=device) @ Pi_B

print("\n=== Train Domain A ===")
loss_A = train_domain(layer0, x, attn_mask, pos_emb, Pi_A, target_A, STEPS, "Domain A")
print(f"Final loss A: {loss_A:.6f}")

with torch.no_grad():
    out = layer0(x, attention_mask=attn_mask, position_embeddings=pos_emb)
    if isinstance(out, tuple): out = out[0]
    loss_A_init = nn.MSELoss()(out @ Pi_A, target_A).item()
print(f"Retention loss before B: {loss_A_init:.8f}")

print("\n=== Train Domain B ===")
loss_B = train_domain(layer0, x, attn_mask, pos_emb, Pi_B, target_B, STEPS, "Domain B")
print(f"Final loss B: {loss_B:.6f}")

with torch.no_grad():
    out = layer0(x, attention_mask=attn_mask, position_embeddings=pos_emb)
    if isinstance(out, tuple): out = out[0]
    loss_A_final = nn.MSELoss()(out @ Pi_A, target_A).item()
print(f"Retention loss after B: {loss_A_final:.8f}")
delta = loss_A_final - loss_A_init
print(f"Delta: {delta:+.8f}")

if abs(delta) < 1e-5:
    print("\n✅ PERFECT ISOLATION at decoder layer level via PLE.")
else:
    print(f"\n⚠️ Delta={delta} – check layer forward or PLE hook.")