"""
Diagnostic script: compare original GPT-2 vs ARW-converted GPT-2
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

# ---------- ARWLinear (copy from train_arw.py) ----------
class ARWLinear(nn.Module):
    def __init__(self, W, bias, in_features, out_features, core_rank, adapter_rank, device='cuda'):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.core_rank = core_rank
        self.adapter_rank = adapter_rank

        W = W.to(device).float()
        U, S, Vh = torch.linalg.svd(W, full_matrices=False)
        k = min(core_rank, U.shape[1])
        self.register_buffer('U_core', U[:, :k].clone())          # (out, k)
        self.register_buffer('V_core', Vh[:k, :].T.clone())       # (in, k)
        self.register_buffer('W0', W.clone())
        self.bias = bias.to(device).float().clone() if bias is not None else None

        # Adapter – MUST start at zero
        self.B = nn.Parameter(torch.zeros(out_features, adapter_rank, device=device))
        self.A = nn.Parameter(torch.zeros(adapter_rank, in_features, device=device))

    def _shell_projection(self, dW):
        Uc, Vc = self.U_core, self.V_core
        term1 = Uc @ (Uc.T @ dW)
        term2 = (dW @ Vc) @ Vc.T
        term3 = Uc @ (Uc.T @ dW @ Vc) @ Vc.T
        return dW - term1 - term2 + term3

    def forward(self, x):
        dW = self.B @ self.A
        dW_shell = self._shell_projection(dW)
        W_eff = self.W0 + dW_shell
        return F.linear(x, W_eff, self.bias)

    @classmethod
    def from_weights(cls, W, bias, in_features, out_features, core_rank, adapter_rank, device='cuda'):
        return cls(W, bias, in_features, out_features, core_rank, adapter_rank, device)

    @classmethod
    def convert_gpt2_layers(cls, model, core_rank, adapter_rank, device='cuda'):
        for name, module in model.named_children():
            if 'lm_head' in name:
                continue
            if isinstance(module, nn.Linear):
                W = module.weight.data.clone()
                bias = module.bias.data.clone() if module.bias is not None else None
                in_feat, out_feat = module.in_features, module.out_features
            elif module.__class__.__name__ == 'Conv1D':
                # Conv1D weight: (in, out) → transpose to (out, in)
                W = module.weight.data.clone().t().contiguous()
                bias = module.bias.data.clone() if module.bias is not None else None
                in_feat, out_feat = module.nx, module.nf
            else:
                cls.convert_gpt2_layers(module, core_rank, adapter_rank, device)
                continue
            arw = ARWLinear.from_weights(W, bias, in_feat, out_feat, core_rank, adapter_rank, device)
            setattr(model, name, arw)

# ---------- Test ----------
print("Loading tokenizer...")
tok = GPT2TokenizerFast.from_pretrained('gpt2')
tok.pad_token = tok.eos_token

print("Preparing input...")
inp = tok("The quick brown fox", return_tensors='pt')

print("\n=== Testing Model Loss ===")

# Original model
print("Loading original GPT-2...")
orig = GPT2LMHeadModel.from_pretrained('gpt2').eval()
with torch.no_grad():
    loss_orig = orig(**inp, labels=inp['input_ids']).loss.item()
print(f"Original loss: {loss_orig:.6f}")

# ARW model
print("Loading GPT-2 and converting to ARW...")
arw = GPT2LMHeadModel.from_pretrained('gpt2')
ARWLinear.convert_gpt2_layers(arw, core_rank=8, adapter_rank=32, device='cpu')
arw.eval()
with torch.no_grad():
    loss_arw = arw(**inp, labels=inp['input_ids']).loss.item()
print(f"ARW loss:      {loss_arw:.6f}")

loss_diff = abs(loss_orig - loss_arw)
print(f"Loss difference: {loss_diff:.6f}")

print("\n=== Layer-wise Check ===")

# Compare first c_attn layer
print("Extracting original c_attn...")
orig_attn = orig.transformer.h[0].attn.c_attn   # Conv1D
print(f"  Type: {type(orig_attn)}")
print(f"  Weight shape: {orig_attn.weight.shape if hasattr(orig_attn, 'weight') else 'N/A'}")

print("Extracting ARW c_attn...")
arw_attn = arw.transformer.h[0].attn.c_attn     # ARWLinear
print(f"  Type: {type(arw_attn)}")
print(f"  W0 shape: {arw_attn.W0.shape if hasattr(arw_attn, 'W0') else 'N/A'}")

print("Testing single layer with random input...")
hidden = torch.randn(1, 768)
with torch.no_grad():
    out_orig = orig_attn(hidden)
    out_arw = arw_attn(hidden)

print(f"Original output shape: {out_orig.shape}")
print(f"ARW output shape:      {out_arw.shape}")

layer_diff = (out_orig - out_arw).abs().max().item()
print(f"Max layer diff: {layer_diff:.6f}")

print("\n=== Summary ===")
print(f"Model loss diff:    {loss_diff:.6f}")
print(f"Layer output diff:  {layer_diff:.6f}")
if loss_diff > 0.01:
    print("⚠️  LARGE DIFFERENCE: Forward pass is broken!")
else:
    print("✓ Forward pass matches original")
