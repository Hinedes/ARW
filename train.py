import sys
import argparse, os, time, math
import gc

# Suppress PyTorch Inductor autotuning spam (Layout conflicts, C++ OOM retries)
os.environ["TORCH_LOGS"] = "-inductor"
os.environ["TORCH_CPP_LOG_LEVEL"] = "ERROR"
import logging
logging.getLogger("torch._inductor.scheduler").setLevel(logging.CRITICAL)
logging.getLogger("torch._inductor.select_algorithm").setLevel(logging.CRITICAL)
logging.getLogger("torch._inductor.utils").setLevel(logging.CRITICAL)

import torch, torch.nn as nn, torch.nn.functional as F

from torch.utils.data import DataLoader, Dataset
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
from datasets import load_dataset

# Redirect standard output and error to a log file, keeping a display function for console
original_stdout = sys.stdout
log_file = open("training.log", "w", encoding="utf-8")
sys.stdout = log_file
sys.stderr = log_file

def display(msg=""):
    original_stdout.write(str(msg) + "\n")
    original_stdout.flush()
    log_file.write(str(msg) + "\n")
    log_file.flush()

# Optimise for RTX 50-series (Ada/Blackwell): use TF32 for matmuls
torch.set_float32_matmul_precision('high')

# ---------- ARWLinear (zero‑init, from weights) ----------
class ARWLinearFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, W0, bias, U_core, V_core, A, B):
        dW = B @ A
        W_eff = W0 + dW
        ctx.save_for_backward(x, W0, bias, U_core, V_core, A, B, dW)
        return F.linear(x, W_eff, bias)

    @staticmethod
    def backward(ctx, grad_output):
        x, W0, bias, U_core, V_core, A, B, dW = ctx.saved_tensors
        # Flatten the batch and sequence dimensions
        batch_size, seq_len, hidden = x.shape
        x_flat = x.reshape(-1, hidden)                     # (B*S, H)
        grad_flat = grad_output.reshape(-1, hidden)        # (B*S, H)

        # Gradient w.r.t. effective weight: (H_out, H_in) = (768, 768)
        grad_W_eff = grad_flat.t() @ x_flat               # (768, 768)

        # Project onto shell complement
        Uc, Vc = U_core, V_core
        term1 = Uc @ (Uc.t() @ grad_W_eff)
        term2 = (grad_W_eff @ Vc) @ Vc.t()
        term3 = Uc @ (Uc.t() @ grad_W_eff @ Vc) @ Vc.t()
        grad_W_shell = grad_W_eff - term1 - term2 + term3

        # grad_input = grad_output @ W_eff^T, with proper flattening
        W_eff = W0 + dW
        grad_input = grad_flat @ W_eff.t()                # (B*S, H)
        grad_input = grad_input.view(batch_size, seq_len, hidden)

        # Bias gradient
        grad_bias = grad_output.sum(dim=(0, 1)) if grad_output.dim() == 3 else grad_output.sum(dim=0)

        # Gradients for adapter weights
        grad_B = grad_W_shell @ A.t()                     # (768, r)
        grad_A = B.t() @ grad_W_shell                     # (r, 768)

        return grad_input, None, grad_bias, None, None, grad_A, grad_B


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
        self.register_buffer('U_core', U[:, :k].clone())
        self.register_buffer('V_core', Vh[:k, :].T.clone())
        self.register_buffer('W0', W.clone())
        self.bias = bias.to(device).float().clone() if bias is not None else None

        self.A = nn.Parameter(torch.empty(adapter_rank, self.in_features, device=device))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        self.B = nn.Parameter(torch.zeros(out_features, adapter_rank, device=device))

    def forward(self, x):
        return ARWLinearFunction.apply(x, self.W0, self.bias, self.U_core, self.V_core, self.A, self.B)

    @classmethod
    def from_weights(cls, W, bias, in_features, out_features, core_rank, adapter_rank, device='cuda'):
        return cls(W, bias, in_features, out_features, core_rank, adapter_rank, device)

    @staticmethod
    def required_rank_for_variance(W, target_variance=0.99):
        """Return smallest k such that top-k singular vectors explain >= target_variance."""
        U, S, Vh = torch.linalg.svd(W.float(), full_matrices=False)
        total_var = (S ** 2).sum()
        cum_var = torch.cumsum(S ** 2, dim=0)
        k = torch.searchsorted(cum_var / total_var, torch.tensor(target_variance).to(S.device)) + 1
        return int(k.clamp(max=len(S)).item())

    @classmethod
    def convert_gpt2_layers_adaptive(cls, model, target_variance=0.99, adapter_rank=16, device='cuda'):
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
                cls.convert_gpt2_layers_adaptive(module, target_variance, adapter_rank, device)
                continue
            core_r = cls.required_rank_for_variance(W, target_variance)
            arw = cls.from_weights(W, bias, in_feat, out_feat, core_r, adapter_rank, device)
            setattr(model, name, arw)

# ---------- Datasets ----------
class TextDataset(Dataset):
    def __init__(self, encodings):
        self.input_ids = encodings['input_ids']
        self.attention_mask = encodings['attention_mask']
    def __getitem__(self, idx):
        return {'input_ids': self.input_ids[idx], 'attention_mask': self.attention_mask[idx]}
    def __len__(self):
        return len(self.input_ids)

def prepare_wikitext_eval(tokenizer, block_size=256, num_samples=1000):
    dataset = load_dataset('Salesforce/wikitext', 'wikitext-2-raw-v1', split='test', streaming=True)
    samples = []
    for i, ex in enumerate(dataset):
        if i >= num_samples: break
        text = ex['text'].strip()
        if len(text) > 50:   # filter empty lines
            samples.append(text)
    enc = tokenizer(samples, return_tensors='pt', truncation=True, padding=True, max_length=block_size)
    return TextDataset(enc)

def prepare_python(tokenizer, block_size=256, num_samples=5000):
    try:
        dataset = load_dataset('codeparrot/github-code', 'python', split='train', streaming=True)
        samples = []
        for i, ex in enumerate(dataset):
            if i >= num_samples*2: break
            samples.append(ex['code'][:1000])
        enc = tokenizer(samples, return_tensors='pt', truncation=True, padding=True, max_length=block_size)
        return TextDataset(enc)
    except:
        py = ["def foo():\n    return 42\n"] * (num_samples*2)
        enc = tokenizer(py, return_tensors='pt', truncation=True, padding=True, max_length=block_size)
        return TextDataset(enc)

# ---------- Eval with debug ----------
@torch.no_grad()
def evaluate_ppl(model, dataloader, device, tag=''):
    model.eval()
    total_loss, total_tokens = 0.0, 0
    for i, batch in enumerate(dataloader):
        input_ids = batch['input_ids'].to(device, non_blocking=True)
        mask = batch['attention_mask'].to(device, non_blocking=True)

        # Replace padding tokens with -100 so Hugging Face ignores them
        labels = input_ids.clone()
        labels[mask == 0] = -100

        try:
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                out = model(input_ids, attention_mask=mask, labels=labels)
                loss = out.loss

            # loss is already averaged over non‑ignored tokens by HF
            # We'll accumulate total nats ourselves
            num_valid = mask.sum().item()
            total_loss += loss.item() * num_valid
            total_tokens += num_valid
            if i == 0:
                display(f"  {tag} first batch loss: {loss.item():.4f}")
        except torch.cuda.OutOfMemoryError:
            display(f"WARNING: CUDA Out of Memory during {tag} eval! Clearing cache...")
            torch.cuda.empty_cache()
            break
    avg = total_loss / total_tokens if total_tokens > 0 else float('nan')
    ppl = math.exp(avg) if avg < 100 else float('inf')
    display(f"  {tag} avg loss: {avg:.4f}, PPL: {ppl:.3f}")
    return ppl

# ---------- Training ----------
def train(model, loader, opt, epochs, device):
    for epoch in range(epochs):
        start = time.time(); total = 0.0
        for batch in loader:
            input_ids = batch['input_ids'].to(device, non_blocking=True)
            mask = batch['attention_mask'].to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            
            try:
                with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                     labels = input_ids.clone()
                     labels[mask == 0] = -100
                     loss = model(input_ids, attention_mask=mask, labels=labels).loss
                    
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                total += loss.item()
            except torch.cuda.OutOfMemoryError:
                display("WARNING: CUDA Out of Memory during training! Clearing cache and terminating gracefully...")
                torch.cuda.empty_cache()
                return  # Gracefully exit the training loop early
        display(f"Epoch {epoch+1}/{epochs} Loss: {total/len(loader):.4f} Time: {time.time()-start:.1f}s")

# ---------- Main ----------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', default='gpt2')
    parser.add_argument('--target_variance', type=float, default=0.99) # 99% variance explained
    parser.add_argument('--adapter_rank', type=int, default=8)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--lr', type=float, default=1e-3)    # higher lr for small adapter
    parser.add_argument('--domain1_samples', type=int, default=2000)
    parser.add_argument('--run_baseline', action='store_true')
    parser.add_argument('--output_dir', default='./arw_results')
    args = parser.parse_args()

    assert torch.cuda.is_available(), "This script requires a GPU!"
    device = torch.device('cuda')
    
    tok = GPT2TokenizerFast.from_pretrained(args.model_name); tok.pad_token = tok.eos_token

    model = GPT2LMHeadModel.from_pretrained(args.model_name).to(device)
    try:
        ARWLinear.convert_gpt2_layers_adaptive(model, args.target_variance, args.adapter_rank, device)
    except RuntimeError as e:
        display(f"Conversion failed: {e}")
        display("Falling back to target_variance=0.90 (more memory-safe).")
        del model
        torch.cuda.empty_cache()
        model = GPT2LMHeadModel.from_pretrained(args.model_name).to(device)
        ARWLinear.convert_gpt2_layers_adaptive(model, 0.90, args.adapter_rank, device)
    
    display(f"VRAM used: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

    # Freeze all except adapter
    for p in model.parameters(): p.requires_grad = False
    arw_cnt = 0
    for m in model.modules():
        if isinstance(m, ARWLinear):
            m.A.requires_grad = True
            m.B.requires_grad = True
            arw_cnt += 1
    trainable = [n for n,p in model.named_parameters() if p.requires_grad]
    display(f"ARW layers: {arw_cnt}, trainable params: {len(trainable)}")

    # ===== SANITY CHECK =====
    display("\n--- Sanity check: dummy input ---")
    dummy = torch.randint(0, 1000, (1, 10)).to(device)
    with torch.no_grad():
        logits = model(dummy).logits
        display(f"  logits shape: {logits.shape}, NaN: {torch.isnan(logits).any().item()}, Inf: {torch.isinf(logits).any().item()}")
    if torch.isnan(logits).any() or torch.isinf(logits).any():
        raise RuntimeError("Model produces NaN/Inf before training – aborting.")

    # Data
    en_eval = prepare_wikitext_eval(tok, block_size=256, num_samples=1000)  
    en_loader = DataLoader(en_eval, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=0)
    py_train = prepare_python(tok, num_samples=args.domain1_samples)
    py_loader = DataLoader(py_train, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=2)

    display("\n[ARW] English PPL before training:")
    en_before = evaluate_ppl(model, en_loader, device, tag='Before')

    opt = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    display("\nTraining ARW on Python...")
    train(model, py_loader, opt, args.epochs, device)

    display("\n=== ARW Results ===")
    en_after = evaluate_ppl(model, en_loader, device, tag='After ')
    py_after = evaluate_ppl(model, py_loader, device, tag='Python')
    delta = en_after - en_before
    display(f"English Δ = {delta:+.3f}")

    os.makedirs(args.output_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(args.output_dir, 'arw_model.pt'))

    # Free memory before running the baseline to prevent OOM
    if args.run_baseline:
        display("\n--- Full FT baseline ---")
        del model
        gc.collect()
        torch.cuda.empty_cache()

        model_ft = GPT2LMHeadModel.from_pretrained(args.model_name).to(device)
        
        en_before_ft = evaluate_ppl(model_ft, en_loader, device, tag='FT Before')
        opt_ft = torch.optim.AdamW(model_ft.parameters(), lr=args.lr)
        train(model_ft, py_loader, opt_ft, args.epochs, device)
        en_after_ft = evaluate_ppl(model_ft, en_loader, device, tag='FT After ')
        display(f"Full FT English Δ = {en_after_ft - en_before_ft:+.3f}")
        del model_ft

    # Final cleanup
    display("\nCleaning up VRAM...")
    gc.collect()
    torch.cuda.empty_cache()

if __name__ == '__main__':
    main()