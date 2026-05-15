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

# --- 1. Load model ---
config = AutoConfig.from_pretrained(MODEL_ID)
model = Gemma4ForConditionalGeneration.from_pretrained(
    MODEL_ID, torch_dtype=torch.float32, device_map="auto"
)

# --- 2. Explore the model structure ---
base_model = model.model
print(f"Type of base_model: {type(base_model)}")
print("\nTop‑level children of base_model:")
for name, child in base_model.named_children():
    print(f"  {name}: {type(child)}")

# Try to find any list of layers
layer_paths = [
    ("base_model.layers", lambda m: m.model.layers),
    ("base_model.decoder.layers", lambda m: m.model.decoder.layers),
    ("base_model.transformer.layers", lambda m: m.model.transformer.layers),
    ("base_model.model.layers", lambda m: m.model.model.layers),
]

found_layers = None
for path, accessor in layer_paths:
    try:
        layers = accessor(model)  # we pass the whole model to accessor
        if isinstance(layers, (list, nn.ModuleList)):
            found_layers = layers
            print(f"\nFound layers via: {path}")
            break
    except:
        continue

if found_layers is None:
    # Last resort: iterate over all children and find the first that looks like a list of layers
    print("\nManually scanning for layer list...")
    for name, module in base_model.named_modules():
        if isinstance(module, nn.ModuleList) and len(module) > 10:
            found_layers = module
            print(f"Found ModuleList with {len(module)} items at {name}")
            break

if found_layers is None:
    raise RuntimeError("Could not locate decoder layers. Please inspect the printed children list.")
else:
    print(f"Number of layers: {len(found_layers)}")
    layer0 = found_layers[0]

# --- 3. Verify PLE components in layer0 ---
assert hasattr(layer0, 'per_layer_input_gate'), "per_layer_input_gate missing"
assert hasattr(layer0, 'per_layer_projection'), "per_layer_projection missing"
print(f"Layer type: {type(layer0)}")

# Freeze everything except PLE
for name, param in layer0.named_parameters():
    if 'per_layer' not in name:
        param.requires_grad = False
    else:
        param.requires_grad = True
layer0 = layer0.to(device)

# --- 4. Orthogonal bases (unchanged) ---
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

# --- 5. ARW hook ---
def make_arw_hook(Pi):
    def hook(grad):
        return grad @ Pi
    return hook

# --- 6. Dummy input ---
torch.manual_seed(0)
x = torch.randn(4, 128, H, device=device)
attn_mask = torch.ones(4, 1, 128, 128, device=device)
pos_emb = torch.zeros(1, 128, H, device=device)

# Quick test to see if layer forward works
try:
    out_test = layer0(x, attention_mask=attn_mask, position_embeddings=pos_emb)
    if isinstance(out_test, tuple):
        out_test = out_test[0]
    print(f"Test forward shape: {out_test.shape}")
except Exception as e:
    print(f"Forward failed with position_embeddings: {e}")
    out_test = layer0(x, attention_mask=attn_mask)
    if isinstance(out_test, tuple):
        out_test = out_test[0]
    print(f"Fallback forward shape: {out_test.shape}")

# --- 7. Training ---
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
        if isinstance(out, tuple):
            out = out[0]
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