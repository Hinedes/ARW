import torch
import torch.nn as nn

H = 2560
K = 64
L = 256
SEED_A = 1
SEED_B = 2
LR = 0.01
STEPS = 200

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# --- 1. Orthogonal bases ---
def make_basis(seed, other_basis=None):
    g = torch.Generator(device=device).manual_seed(seed)
    G = torch.randn(H, K, generator=g, device=device)
    if other_basis is not None:
        G = G - other_basis @ (other_basis.T @ G)   # remove projection onto other basis
    Q, _ = torch.linalg.qr(G)
    return Q

P_A = make_basis(SEED_A)
P_B = make_basis(SEED_B, other_basis=P_A)
assert torch.allclose(P_A.T @ P_B, torch.zeros(K, K, device=device), atol=1e-5)
print("Bases orthogonal.")

Pi_A = P_A @ P_A.T
Pi_B = P_B @ P_B.T

# --- 2. Linear layer (down-proj style) ---
layer = nn.Linear(H, L, bias=False).to(device)
W_init = layer.weight.data.clone()

# --- 3. ARW hook (write projection) ---
def make_arw_hook(Pi):
    def hook(grad):
        return grad @ Pi   # projects each row's weight gradient onto the active subspace
    return hook

# --- 4. Synthetic data ---
torch.manual_seed(0)
x = torch.randn(128, H, device=device)   # full-space input

# Mappings defined only in respective subspaces
M_A = torch.randn(L, K, device=device)
M_B = torch.randn(L, K, device=device)

# Projected inputs (read projections)
x_A = x @ Pi_A   # (128, H)  -- only A subspace components
x_B = x @ Pi_B

# Targets (noise-free to see perfect retention)
target_A = (M_A @ (P_A.T @ x.T)).T   # = (M_A @ (P_A.T @ x.T))^T, but we can also compute via x_A
# Actually, target_A should be M_A @ (P_A.T @ x) = (x @ P_A @ M_A.T) but shape (128, L). Let's use: target_A = x_A @ P_A @ M_A.T? Wait, x_A = x @ Pi_A = x @ P_A @ P_A.T. Then x_A @ P_A = x @ P_A @ P_A.T @ P_A = x @ P_A. So target_A = (x @ P_A) @ M_A.T. We'll compute as:
target_A = (x @ P_A) @ M_A.T   # (128, L)
target_B = (x @ P_B) @ M_B.T

# --- 5. Training with read projection ---
def train_domain(layer, x_proj, target, Pi, steps, desc):
    opt = torch.optim.SGD(layer.parameters(), lr=LR)
    hook_handle = layer.weight.register_hook(make_arw_hook(Pi))
    loss_fn = nn.MSELoss()
    for i in range(steps):
        opt.zero_grad()
        out = layer(x_proj)          # read: project input before layer
        loss = loss_fn(out, target)
        loss.backward()
        opt.step()
        if i % 50 == 0:
            print(f"{desc} | Step {i:3d} | Loss: {loss.item():.6f}")
    hook_handle.remove()
    return loss.item()

print("\n=== Train Domain A (read A, write A) ===")
loss_A = train_domain(layer, x_A, target_A, Pi_A, STEPS, "Domain A")
print(f"Domain A final loss: {loss_A:.6f}")

# Evaluate A with read projection A
with torch.no_grad():
    pred_A = layer(x_A)
    loss_A_init = nn.MSELoss()(pred_A, target_A).item()
print(f"Domain A retention loss (before B): {loss_A_init:.8f}")

print("\n=== Train Domain B (read B, write B) ===")
loss_B = train_domain(layer, x_B, target_B, Pi_B, STEPS, "Domain B")
print(f"Domain B final loss: {loss_B:.6f}")

# Evaluate A again with read projection A
with torch.no_grad():
    pred_A_after = layer(x_A)
    loss_A_final = nn.MSELoss()(pred_A_after, target_A).item()
print(f"Domain A retention loss (after B): {loss_A_final:.8f}")
delta = loss_A_final - loss_A_init
print(f"Delta: {delta:+.8f}")

# Check weight update isolation
W_after = layer.weight.data
W_update = W_after - W_init
proj_A_norm = torch.norm(W_update @ Pi_A) / torch.norm(W_update)
proj_B_norm = torch.norm(W_update @ Pi_B) / torch.norm(W_update)
print(f"\nWeight update projection onto A subspace: {proj_A_norm:.4f}")
print(f"Weight update projection onto B subspace: {proj_B_norm:.4f}")

if abs(delta) < 1e-5:
    print("\n✅ PERFECT RETENTION: ARW (read + write) isolates domains.")
else:
    print(f"\n⚠️ Delta={delta} – check read projection.")