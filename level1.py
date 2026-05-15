import torch
import torch.nn as nn
from transformers import AutoConfig, Gemma4ForConditionalGeneration

MODEL_ID = "google/gemma-4-E4B"
H = 2560
K = 64
SEED_A = 1
SEED_B = 2
LR = 0.01
STEPS = 50          # quick test
BATCH = 1
SEQ = 4

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# --- 1. Load model, freeze all but PLE of layer 0 ---
config = AutoConfig.from_pretrained(MODEL_ID)
model = Gemma4ForConditionalGeneration.from_pretrained(
    MODEL_ID, torch_dtype=torch.float32, device_map="auto"
)
text_model = model.model.language_model.to(device)

for param in text_model.parameters():
    param.requires_grad = False

layer0 = text_model.layers[0]
layer0.per_layer_input_gate.weight.requires_grad = True
layer0.per_layer_projection.weight.requires_grad = True

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

# --- 4. Tiny dummy input ---
torch.manual_seed(0)
x_embeds = torch.randn(BATCH, SEQ, H, device=device)
input_ids = torch.zeros(BATCH, SEQ, dtype=torch.long, device=device)   # dummy IDs
position_ids = torch.arange(SEQ, device=device).unsqueeze(0).repeat(BATCH, 1)
attn_mask = torch.ones(BATCH, 1, SEQ, SEQ, device=device, dtype=torch.bool).tril()

# --- 5. Training function ---
def train_domain(model, input_ids, inputs_embeds, attention_mask, position_ids, Pi, target, steps, desc):
    torch.cuda.empty_cache()
    layer0 = model.layers[0]
    h1 = layer0.per_layer_input_gate.weight.register_hook(make_arw_hook(Pi))
    h2 = layer0.per_layer_projection.weight.register_hook(make_arw_hook(Pi))
    opt = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=LR)
    loss_fn = nn.MSELoss()
    for i in range(steps):
        opt.zero_grad()
        out = model(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
        )
        hidden = out[0]
        hidden_proj = hidden @ Pi
        loss = loss_fn(hidden_proj, target)
        loss.backward()
        opt.step()
        if i % 10 == 0:
            print(f"{desc} | Step {i:3d} | Loss: {loss.item():.6f}")
    h1.remove()
    h2.remove()
    del opt
    torch.cuda.empty_cache()
    return loss.item()

target_A = torch.randn(BATCH, SEQ, H, device=device) @ Pi_A
target_B = torch.randn(BATCH, SEQ, H, device=device) @ Pi_B

print("\n=== Train Domain A ===")
loss_A = train_domain(text_model, input_ids, x_embeds, attn_mask, position_ids, Pi_A, target_A, STEPS, "Domain A")

with torch.no_grad():
    out = text_model(input_ids=input_ids, inputs_embeds=x_embeds, attention_mask=attn_mask, position_ids=position_ids)
    hidden = out[0]
    loss_A_init = nn.MSELoss()(hidden @ Pi_A, target_A).item()
print(f"Retention loss before B: {loss_A_init:.8f}")

print("\n=== Train Domain B ===")
loss_B = train_domain(text_model, input_ids, x_embeds, attn_mask, position_ids, Pi_B, target_B, STEPS, "Domain B")

with torch.no_grad():
    out = text_model(input_ids=input_ids, inputs_embeds=x_embeds, attention_mask=attn_mask, position_ids=position_ids)
    hidden = out[0]
    loss_A_final = nn.MSELoss()(hidden @ Pi_A, target_A).item()
print(f"Retention loss after B: {loss_A_final:.8f}")
delta = loss_A_final - loss_A_init
print(f"Delta: {delta:+.8f}")

if abs(delta) < 1e-5:
    print("\n✅ PERFECT ISOLATION via PLE in layer 0 (full model context).")
else:
    print(f"\n⚠️ Delta={delta} – check implementation.")