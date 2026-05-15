import torch
import torch.nn as nn

# ------------------------------
# Level 0 – ARW on a single PLE‑like projection
# ------------------------------

H = 2560          # hidden size
K = 64            # subspace dimension
L = 256           # PLE projection output dim (per_layer_projection)
SEED_A = 1
SEED_B = 2
LR = 0.01
STEPS = 200       # enough to converge on a simple regression

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# --- 1. Generate two exactly orthogonal bases ---
def make_basis(seed, other_basis=None):
    """Generate orthonormal basis (H, k). If other_basis given, ensure orthogonality."""
    g = torch.Generator(device=device).manual_seed(seed)
    G = torch.randn(H, K, generator=g, device=device)
    if other_basis is not None:
        # Remove projection onto other_basis
        proj = other_basis @ (other_basis.T @ G)
        G = G - proj
    Q, _ = torch.linalg.qr(G)
    return Q

P_A = make_basis(SEED_A)
P_B = make_basis(SEED_B, other_basis=P_A)   # ensures P_B ⊥ P_A

# Verify orthogonality
assert torch.allclose(P_A.T @ P_B, torch.zeros(K, K, device=device), atol=1e-5), "Bases not orthogonal!"
print("Bases are perfectly orthogonal.")

# Projection matrices
Pi_A = P_A @ P_A.T   # (H, H)
Pi_B = P_B @ P_B.T

# --- 2. Define the linear layer (down‑proj style: H → L) ---
layer = nn.Linear(H, L, bias=False).to(device)
# Store initial weights for later comparison
W_init = layer.weight.data.clone()

# --- 3. ARW gradient hook ---
def make_arw_hook(Pi):
    def hook(grad):
        # For a down_proj (output dim L, input dim H), weight shape is (L, H).
        # Gradient w.r.t. weight is (L, H), but we need to restrict the *input* subspace
        # (the hidden side). The hidden dimension is the second dimension (columns).
        # We want to zero out gradients that would change the mapping from the subspace
        # outside Pi's span. So we multiply grad_W @ Pi^T = grad_W @ Pi (since Pi is symmetric).
        return grad @ Pi
    return hook

# --- 4. Synthetic data for two domains ---
# Domain A: input x → target y_A
# Domain B: input x → target y_B
torch.manual_seed(0)
x = torch.randn(128, H, device=device)   # batch of 128 vectors

# Two arbitrary linear mappings (represented by matrices in the A and B subspaces)
M_A = torch.randn(L, K, device=device)   # domain A mapping in A's basis
M_B = torch.randn(L, K, device=device)   # domain B mapping in B's basis

# Targets = M_A @ (P_A.T @ x)  (only uses A subspace) + M_B @ (P_B.T @ x) (only B subspace)
# For training A, we set target_A = M_A @ (P_A.T @ x) + noise, ignoring B subspace.
# For training B, target_B = M_B @ (P_B.T @ x) + noise.
target_A = (M_A @ (P_A.T @ x.T)).T  # (128, L)
target_B = (M_B @ (P_B.T @ x.T)).T

# Add tiny noise to avoid exact zero loss
target_A += 0.001 * torch.randn_like(target_A)
target_B += 0.001 * torch.randn_like(target_B)

# --- 5. Training procedure ---
def train_domain(layer, x, target, Pi, steps, desc):
    opt = torch.optim.SGD(layer.parameters(), lr=LR)
    hook_handle = layer.weight.register_hook(make_arw_hook(Pi))
    loss_fn = nn.MSELoss()
    for i in range(steps):
        opt.zero_grad()
        out = layer(x)
        loss = loss_fn(out, target)
        loss.backward()
        opt.step()
        if i % 50 == 0:
            print(f"{desc} | Step {i:3d} | Loss: {loss.item():.6f}")
    hook_handle.remove()
    return loss.item()

# --- 6. Run experiment ---
print("\n=== Training Domain A ===")
loss_A_before = train_domain(layer, x, target_A, Pi_A, STEPS, "Domain A")
print(f"Domain A final loss: {loss_A_before:.6f}")

# Evaluate Domain A loss right after
with torch.no_grad():
    out_A = layer(x)
    loss_A_initial = nn.MSELoss()(out_A, target_A).item()
print(f"Domain A retention loss (before B): {loss_A_initial:.6f}")

print("\n=== Training Domain B ===")
loss_B = train_domain(layer, x, target_B, Pi_B, STEPS, "Domain B")
print(f"Domain B final loss: {loss_B:.6f}")

# Evaluate Domain A after B training
with torch.no_grad():
    out_A_after = layer(x)
    loss_A_final = nn.MSELoss()(out_A_after, target_A).item()
print(f"\nDomain A retention loss (after B): {loss_A_final:.6f}")

delta = loss_A_final - loss_A_initial
print(f"Delta: {delta:+.8f}")

# --- 7. Verify isolation numerically ---
# The weight update after B training should lie entirely in Pi_B's subspace.
# So the projection of the update onto Pi_A should be negligible.
W_after = layer.weight.data
W_update = W_after - W_init
# For down_proj, the update should satisfy: W_update @ Pi_A ≈ 0
proj_A_update = torch.norm(W_update @ Pi_A) / torch.norm(W_update)
print(f"\nFraction of weight update projecting onto Domain A subspace: {proj_A_update:.8f} (should be ~0)")

if abs(delta) < 1e-5:
    print("\n✅ SUCCESS: Domain A knowledge perfectly retained after Domain B training.")
else:
    print(f"\n⚠️  Retention not perfect (delta={delta}). Check projection orthogonality.")