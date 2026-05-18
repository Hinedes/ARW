# grafting.py
# Core module for ARW-based grafting: subspace generation, compression, projection, and reinstallation.
# Works with any model of hidden dimension H and subspace rank K.

import torch
import torch.nn as nn
from typing import Tuple, Optional, Dict, Union

# ----------------------------------------------------------------------
# 1. Subspace basis generation (fixed random orthogonal)
# ----------------------------------------------------------------------

def make_arw_basis(hidden_dim: int, rank: int, seed: int, device: torch.device) -> torch.Tensor:
    """
    Generate an orthonormal basis P of shape (hidden_dim, rank).
    The same (seed, hidden_dim, rank) always yields the same P.
    """
    assert rank <= hidden_dim, f"rank {rank} > hidden_dim {hidden_dim}"
    g = torch.Generator(device=device).manual_seed(seed)
    G = torch.randn(hidden_dim, rank, generator=g, device=device, dtype=torch.float32)
    Q, _ = torch.linalg.qr(G)
    return Q[:, :rank]  # (H, K)

def projection_from_basis(P: torch.Tensor) -> torch.Tensor:
    """Return the projection matrix Pi = P @ P.T, shape (H, H)."""
    return P @ P.T

# ----------------------------------------------------------------------
# 2. Layer type detection (which side the projection applies)
# ----------------------------------------------------------------------

def get_layer_type(name: str) -> Optional[str]:
    """
    Infer projection side from parameter name.
    Returns 'left' for gate_proj/up_proj (weights multiply from left),
    'right' for down_proj (weights multiply from right),
    None otherwise.
    """
    if any(x in name for x in ['gate_proj', 'up_proj']):
        return 'left'
    elif 'down_proj' in name:
        return 'right'
    return None

# ----------------------------------------------------------------------
# 3. Gradient projection (backward hook)
# ----------------------------------------------------------------------

def project_gradient(grad: torch.Tensor, Pi: torch.Tensor, layer_type: str) -> torch.Tensor:
    """
    Project a gradient tensor so that updates stay inside the subspace.
    For left‑multiplied weight: grad ← grad @ Pi   (since W * x, gradient w.r.t W has shape H×H)
    For right‑multiplied weight: grad ← Pi @ grad
    """
    if layer_type == 'left':
        # shape: (H, H) or (H, out) – project on the right side
        return grad @ Pi
    elif layer_type == 'right':
        # shape: (in, H) – project on the left side
        return Pi @ grad
    else:
        return grad  # no projection

def make_backward_hook(Pi: torch.Tensor, layer_type: str):
    """Return a hook function that projects gradients."""
    Pi_f32 = Pi.to(torch.float32)
    def hook(grad):
        g_f32 = grad.to(torch.float32)
        proj = project_gradient(g_f32, Pi_f32, layer_type)
        return proj.to(grad.dtype)
    return hook

# ----------------------------------------------------------------------
# 4. Weight update clamping (double‑tap after optimizer step)
# ----------------------------------------------------------------------

def project_weight_update(delta: torch.Tensor, Pi: torch.Tensor, layer_type: str) -> torch.Tensor:
    """
    Project a raw weight change Δ (from optimizer) back into the subspace.
    For left‑multiplied: Δ ← Δ @ Pi
    For right‑multiplied: Δ ← Pi @ Δ
    """
    if layer_type == 'left':
        return delta @ Pi
    elif layer_type == 'right':
        return Pi @ delta
    else:
        return delta

def clamp_weight_update(param: torch.nn.Parameter, old_weight: torch.Tensor,
                        Pi: torch.Tensor, layer_type: str) -> None:
    """
    In‑place clamp: param = old_weight + project_weight_update(param - old_weight, Pi, layer_type)
    """
    with torch.no_grad():
        delta = param - old_weight
        delta_proj = project_weight_update(delta.to(torch.float32), Pi, layer_type)
        param.copy_(old_weight + delta_proj.to(param.dtype))

# ----------------------------------------------------------------------
# 5. Graft compression (extract compact representation)
# ----------------------------------------------------------------------

def compress_graft(delta: torch.Tensor, P: torch.Tensor, layer_type: str) -> torch.Tensor:
    """
    Compress a full delta (Δ) that already lies in the subspace into a K‑dimensional artifact.
    For left: G = Δ @ P      (shape H×K)
    For right: G = P.T @ Δ   (shape K×H)
    """
    delta_f32 = delta.to(torch.float32)
    if layer_type == 'left':
        return (delta_f32 @ P).cpu()
    elif layer_type == 'right':
        return (P.T @ delta_f32).cpu()
    else:
        raise ValueError(f"Unknown layer_type {layer_type}")

def decompress_graft(G: torch.Tensor, P: torch.Tensor, layer_type: str) -> torch.Tensor:
    """
    Reconstruct full delta from compressed graft G.
    For left: Δ = G @ P.T
    For right: Δ = P @ G
    """
    G = G.to(torch.float32)
    P = P.to(torch.float32)
    if layer_type == 'left':
        return (G @ P.T).cpu()
    elif layer_type == 'right':
        return (P @ G).cpu()
    else:
        raise ValueError(f"Unknown layer_type {layer_type}")

# ----------------------------------------------------------------------
# 6. Utility: filter trainable FFN layers (gate/up/down) by name
# ----------------------------------------------------------------------

def get_ffn_param_names(model: nn.Module) -> Dict[str, str]:
    """
    Return a dict {param_name: layer_type} for all parameters that are FFN weight matrices
    and should be grafted.
    """
    result = {}
    for name, param in model.named_parameters():
        if param.ndim != 2:
            continue
        lt = get_layer_type(name)
        if lt is not None:
            result[name] = lt
    return result

# ----------------------------------------------------------------------
# 7. High‑level training wrapper (optional convenience)
# ----------------------------------------------------------------------

class ARWTrainer:
    """
    Helper class to manage hooks and double‑tap clamping during ARW training.
    """
    def __init__(self, model: nn.Module, P: torch.Tensor, device: torch.device):
        self.model = model
        self.P = P.to(device)
        self.Pi = projection_from_basis(self.P).to(device)
        self.device = device
        self.hooks = []
        self.layer_types = {}

        # Pre‑compute layer_type for each parameter we will constrain
        for name, param in model.named_parameters():
            lt = get_layer_type(name)
            if lt is not None and param.requires_grad:
                self.layer_types[name] = lt

    def attach_hooks(self):
        """Register backward hooks on all relevant parameters."""
        self.remove_hooks()
        for name, param in self.model.named_parameters():
            if name in self.layer_types:
                lt = self.layer_types[name]
                hook = make_backward_hook(self.Pi, lt)
                h = param.register_hook(hook)
                self.hooks.append((name, h))

    def remove_hooks(self):
        for _, h in self.hooks:
            h.remove()
        self.hooks.clear()

    def clamp_step(self, old_weights: Dict[str, torch.Tensor]):
        """
        After optimizer.step(), call this with a snapshot of weights before the step.
        old_weights: dict {param_name: weight_before_step (cloned)}
        """
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if name in old_weights and name in self.layer_types:
                    lt = self.layer_types[name]
                    clamp_weight_update(param, old_weights[name].to(self.device), self.Pi, lt)

    def get_snapshot(self) -> Dict[str, torch.Tensor]:
        """Return a clone of current constrained parameters."""
        return {name: param.clone().detach().cpu()
                for name, param in self.model.named_parameters()
                if name in self.layer_types}