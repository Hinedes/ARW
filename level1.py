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

# --- 1. Load model and extract a single text decoder layer ---
config = AutoConfig.from_pretrained(MODEL_ID)
model = Gemma4ForConditionalGeneration.from_pretrained(
    MODEL_ID, torch_dtype=torch.float32, device_map="auto"
)
text_model = model.model.language_model
layers = text_model.layers  # nn.ModuleList
layer0 = layers[0]

# Freeze everything in the layer except PLE
for name, param in layer0.named_parameters():
    if 'per_layer' not in name:
        param.requires_grad = False
    else:
        param.requires_grad = True

# Wrap the layer in a minimal model module to ensure hooks fire correctly
class SingleLayerModel(nn.Module):
    def __init__(self, layer):
        super().__init__()
        self.layer = layer

    def forward(self, *args, **kwargs):
        # The layer's forward returns a tuple (hidden_states, ...), we just need the first element
        outputs = self.layer(*args, **kwargs)
        return outputs[0] if isinstance(outputs, tuple) else outputs

layer_model = SingleLayerModel(layer0).to(device)

# --- 2. Orthogonal bases ---
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

# --- 3. ARW hook ---
def make_arw_hook(Pi):
    def hook(grad):
        return grad @ Pi
    return hook

# --- 4. Dummy input ---
# We'll need to prepare a proper position_embeddings tuple once we inspect the source.
# For now, we'll create dummy inputs and try the forward pass to see the correct args.
torch.manual_seed(0)
batch, seq = 4, 128
x = torch.randn(batch, seq, H, device=device)

# --- 5. Training function (updated to use layer_model) ---
def train_domain(model, x, pos_emb, Pi, target, steps, desc):
    # Register hooks on the PLE layers inside the layer model
    h1 = model.layer.per_layer_input_gate.weight.register_hook(make_arw_hook(Pi))
    h2 = model.layer.per_layer_projection.weight.register_hook(make_arw_hook(Pi))
    opt = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=LR)
    loss_fn = nn.MSELoss()
    for i in range(steps):
        opt.zero_grad()
        # Forward pass with position_embeddings
        out = model(x, position_embeddings=pos_emb)
        out_proj = out @ Pi
        loss = loss_fn(out_proj, target)
        loss.backward()
        opt.step()
        if i % 50 == 0:
            print(f"{desc} | Step {i:3d} | Loss: {loss.item():.6f}")
    h1.remove()
    h2.remove()
    return loss.item()

# --- 6. Placeholder for proper position_embeddings ---
# Replace this block with the correct logic once you've inspected the source
pos_emb = None
target_A = torch.randn(batch, seq, H, device=device) @ Pi_A
target_B = torch.randn(batch, seq, H, device=device) @ Pi_B

# The main execution will be added once we know how to create pos_emb
print("Ready for the corrected RoPE call. Please run the inspection script above first.")