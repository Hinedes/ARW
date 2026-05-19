# multi_interference.py
# Full rewrite: per-domain orthogonal subspaces, GPU/VRAM reporting,
# acoustic treatment with null-target penalty, interference evaluation.
#
# Usage:
#   train baseline: python multi_interference.py --lambda_silence 0.0 --steps 2000 --save_dir multi_baseline --mode train
#   eval baseline:  python multi_interference.py --save_dir multi_baseline --mode eval
#   train acoustic:  python multi_interference.py --lambda_silence 0.01 --steps 2000 --save_dir multi_acoustic --mode train
#   eval acoustic:   python multi_interference.py --save_dir multi_acoustic --mode eval

import argparse, json, os, subprocess, torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from itertools import cycle
from collections import defaultdict
import time

from grafting import (
    make_arw_basis,
    ARWTrainer,
    compress_graft,
    decompress_graft,
    get_ffn_param_names,
)

# ----------------------------------------------------------------------
# GPU / VRAM reporting via rocm-smi
# ----------------------------------------------------------------------
def get_gpu_info():
    """Return GPU stats using rocm-smi (confirmed working on your container)."""
    info = {"gpu_name": "AMD Instinct MI300X", "vram_used_pct": None, "vram_total_pct": None}
    try:
        out = subprocess.check_output(
            ["rocm-smi", "--showmeminfo", "vram", "--csv"],
            stderr=subprocess.DEVNULL, text=True, timeout=5
        )
        # CSV: device, GPU[N], VRAM Total (B), VRAM Used (B)
        lines = out.strip().split('\n')
        if len(lines) > 1:
            parts = lines[1].split(',')
            if len(parts) >= 4:
                total_b = int(parts[2].strip())
                used_b = int(parts[3].strip())
                info["vram_total_gb"] = total_b / (1024**3)
                info["vram_used_gb"] = used_b / (1024**3)
                info["vram_used_pct"] = (used_b / total_b * 100) if total_b > 0 else 0.0
    except Exception:
        pass  # silent fallback
    return info

def print_gpu_info(step=None):
    info = get_gpu_info()
    if step is not None:
        print(f"--- GPU at step {step} ---")
    else:
        print("--- GPU Info ---")
    print(f"  GPU       : {info['gpu_name']}")
    if info.get("vram_used_gb") is not None:
        print(f"  VRAM      : {info['vram_used_gb']:.1f} GB used / {info['vram_total_gb']:.1f} GB total ({info['vram_used_pct']:.1f}%)")
    else:
        print("  VRAM      : rocm-smi unavailable (but GPU is running)")

# ----------------------------------------------------------------------
# Data loading
# ----------------------------------------------------------------------
def load_jsonl_texts(path):
    texts = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            obj = json.loads(line.strip())
            for v in obj.values():
                if isinstance(v, str):
                    texts.append(v)
                    break
    return texts

class InfiniteTextDataset(Dataset):
    """Yields tokenized chunks, cycling forever. Clamped to text vocab for Qwen 3.5 multimodal safety."""
    def __init__(self, texts, tokenizer, max_len=1024, vocab_limit=151935):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.vocab_limit = vocab_limit
        self.pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
        self.data = []
        for t in texts:
            ids = tokenizer.encode(t, add_special_tokens=False)
            if ids:
                # safety clamp
                ids = [min(tok, vocab_limit) for tok in ids]
                self.data.append(ids)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        ids = self.data[idx % len(self.data)]
        if len(ids) > self.max_len:
            start = torch.randint(0, len(ids) - self.max_len + 1, (1,)).item()
            ids = ids[start:start + self.max_len]
        else:
            ids = ids + [self.pad_id] * (self.max_len - len(ids))
        tensor_ids = torch.tensor(ids, dtype=torch.long)
        # Final safety clamp
        tensor_ids = torch.clamp(tensor_ids, max=self.vocab_limit)
        return tensor_ids

class MixedDomainDataset(Dataset):
    """Produces sequences with 80% in-domain, 20% out-of-domain, with a mask."""
    def __init__(self, in_texts, out_texts, tokenizer, max_len=1024, in_frac=0.8, vocab_limit=151935):
        self.max_len = max_len
        self.in_dataset = InfiniteTextDataset(in_texts, tokenizer, max_len=max_len, vocab_limit=vocab_limit)
        self.out_dataset = InfiniteTextDataset(out_texts, tokenizer, max_len=max_len, vocab_limit=vocab_limit)
        self.in_len = int(max_len * in_frac)
        self.out_len = max_len - self.in_len

    def __len__(self):
        return 100000  # large virtual length

    def __getitem__(self, idx):
        in_ids = self.in_dataset[torch.randint(0, len(self.in_dataset), (1,)).item()][:self.in_len]
        out_ids = self.out_dataset[torch.randint(0, len(self.out_dataset), (1,)).item()][:self.out_len]
        input_ids = torch.cat([in_ids, out_ids])
        mask = torch.cat([torch.ones(self.in_len), torch.zeros(self.out_len)])
        return input_ids[:self.max_len], mask[:self.max_len]

# ----------------------------------------------------------------------
# Layer filtering for Qwen 3.5 hybrid architecture
# ----------------------------------------------------------------------
def is_protected_module(module_name: str) -> bool:
    """Return True if this module should NOT be grafted (MoE routers, DeltaNet gates)."""
    forbidden = ["router", "gate_proj_delta", "recurrent"]
    name_lower = module_name.lower()
    return any(kw in name_lower for kw in forbidden)

# ----------------------------------------------------------------------
# Training a single graft (per-domain P)
# ----------------------------------------------------------------------
def train_graft(model, tokenizer, P, in_texts, out_texts,
                steps=5000, batch_size=4, lr=1e-4, lambda_silence=0.01,
                device='cuda', log_interval=100):
    model.to(device)
    model.eval()
    trainable_names = set(get_ffn_param_names(model).keys())
    for n, p in model.named_parameters():
        p.requires_grad = n in trainable_names

    trainer = ARWTrainer(model, P, device)
    Pi = trainer.Pi
    base_weights = {n: p.clone().detach().to(device) for n, p in model.named_parameters()
                    if n in trainer.layer_types}

    # Collect grafted linear modules (excluding routers/DeltaNet)
    grafted_modules = {}
    for mod_name, module in model.named_modules():
        if isinstance(module, nn.Linear) and not is_protected_module(mod_name):
            for pname in ['weight']:
                full = f"{mod_name}.{pname}"
                if full in trainer.layer_types:
                    grafted_modules[full] = module

    saved_inputs = {}
    def fwd_hook(name):
        def hook(module, input, output):
            saved_inputs[name] = input[0].detach()
        return hook
    hooks_fwd = [mod.register_forward_hook(fwd_hook(name)) for name, mod in grafted_modules.items()]

    # Dataset with mixed domain
    dataset = MixedDomainDataset(in_texts, out_texts, tokenizer)
    loader = cycle(DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True))

    optimizer = torch.optim.AdamW(
        [p for n, p in model.named_parameters() if n in trainable_names], lr=lr
    )
    trainer.attach_hooks()
    losses_lm, losses_sil = [], []

    # Best graft tracking
    sil_window = []
    window_size = 50
    best_sil = float('inf')
    best_graft_state = None

    for step in range(1, steps+1):
        input_ids, mask = next(loader)
        input_ids, mask = input_ids.to(device), mask.to(device)

        saved_inputs.clear()
        outputs = model(input_ids=input_ids, labels=input_ids)
        logits = outputs.logits

        # LM loss on in-domain tokens
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = input_ids[..., 1:].contiguous()
        shift_mask = mask[..., 1:].contiguous()
        ce = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)),
                             shift_labels.view(-1), reduction='none').view(shift_labels.shape)
        in_mask_sum = shift_mask.sum()
        lm_loss = (ce * shift_mask).sum() / in_mask_sum if in_mask_sum > 0 else torch.tensor(0.0, device=device)

        # Silence loss on out-of-domain tokens
        silence_loss = torch.tensor(0.0, device=device)
        out_mask = 1.0 - mask
        layers = 0
        for name, module in grafted_modules.items():
            if name not in saved_inputs:
                continue
            x = saved_inputs[name]                     # (B, L, H)
            current_w = model.get_parameter(name)
            delta_w = current_w - base_weights[name]   # only subspace component
            delta_out = F.linear(x, delta_w)           # (B, L, out_features)
            token_norm = delta_out.pow(2).mean(dim=-1)  # per-dim mean squared
            if token_norm.size(1) == out_mask.size(1):
                out_count = out_mask.sum()
                if out_count > 0:
                    silence_loss += (token_norm * out_mask).sum() / out_count
            layers += 1
        if layers > 0:
            silence_loss = silence_loss / layers

        total_loss = lm_loss + lambda_silence * silence_loss
        total_loss.backward()

        # Gradient clipping to prevent spikes from corrupting the graft
        torch.nn.utils.clip_grad_norm_(
            [p for n, p in model.named_parameters() if n in trainable_names],
            max_norm=1.0
        )

        # Optimizer step with ARW double-tap
        old_weights = {n: p.clone().detach().to(device) for n, p in model.named_parameters()
                       if n in trainer.layer_types}
        optimizer.step()
        optimizer.zero_grad()
        trainer.clamp_step(old_weights)

        losses_lm.append(lm_loss.item())
        sil_item = silence_loss.item()
        losses_sil.append(sil_item)

        # Update smoothed silence loss
        sil_window.append(sil_item)
        if len(sil_window) > window_size:
            sil_window.pop(0)
        avg_sil = sum(sil_window) / len(sil_window)

        if avg_sil < best_sil:
            best_sil = avg_sil
            # Deep copy current grafted parameters
            best_graft_state = {n: p.clone().detach().cpu() for n, p in model.named_parameters()
                                if n in trainer.layer_types}

        if step % log_interval == 0:
            avg_lm = sum(losses_lm[-100:]) / min(len(losses_lm), 100)
            avg_sil_print = sum(losses_sil[-100:]) / min(len(losses_sil), 100)
            print(f"Step {step:5d} | LM: {avg_lm:.4f} | Sil: {avg_sil_print:.6f}")

    trainer.remove_hooks()
    for h in hooks_fwd:
        h.remove()

    # Use best graft, not final corrupted state
    if best_graft_state is not None:
        print(f"Loading best graft (silence loss = {best_sil:.6f})")
        with torch.no_grad():
            for name in base_weights:
                model.get_parameter(name).copy_(best_graft_state[name].to(device))
    else:
        print("Warning: no best graft saved, using final state.")

    # Compress graft from the (best) weights
    graft = {}
    with torch.no_grad():
        for name in base_weights:
            delta = model.get_parameter(name).detach() - base_weights[name]
            lt = trainer.layer_types[name]
            graft[name] = compress_graft(delta, P.to(device), lt)

    return graft, trainer.layer_types

# ----------------------------------------------------------------------
# Graft installation helpers
# ----------------------------------------------------------------------
def install_grafts(model, graft_P_pairs, layer_types, device):
    """
    Install a list of grafts onto model.
    graft_P_pairs: list of (graft_dict, P_tensor)
    """
    for graft, P in graft_P_pairs:
        for name, G in graft.items():
            G = G.to(device)
            lt = layer_types[name]
            delta = decompress_graft(G, P.to(device), lt)
            param = model.get_parameter(name)
            param.data.add_(delta.to(device))

# ----------------------------------------------------------------------
# PPL evaluation
# ----------------------------------------------------------------------
def evaluate_ppl(model, tokenizer, texts, max_len=512, device='cuda'):
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    with torch.no_grad():
        for text in texts[:100]:  # limit for speed
            ids = tokenizer.encode(text, add_special_tokens=False)[:max_len]
            if len(ids) < 2:
                continue
            input_ids = torch.tensor([ids], device=device)
            outputs = model(input_ids=input_ids, labels=input_ids)
            loss = outputs.loss
            total_loss += loss.item() * (len(ids) - 1)
            total_tokens += (len(ids) - 1)
    return total_loss / total_tokens if total_tokens > 0 else float('inf')

# ----------------------------------------------------------------------
# RMSNorm scale measurement (dynamic norm detection)
# ----------------------------------------------------------------------
def measure_rmsnorm_stats(model, tokenizer, texts, device='cuda', max_len=512):
    rms_modules = {}
    for name, module in model.named_modules():
        # Duck-type: any module class containing "RMSNorm"
        if "RMSNorm" in module.__class__.__name__:
            rms_modules[name] = module

    saved_inputs = {}
    def hook_fn(name):
        def hook(module, input, output):
            saved_inputs[name] = input[0].detach()
        return hook
    hooks = [mod.register_forward_hook(hook_fn(name)) for name, mod in rms_modules.items()]

    stats = defaultdict(list)
    with torch.no_grad():
        for text in texts[:10]:
            ids = tokenizer.encode(text, add_special_tokens=False)[:max_len]
            if len(ids) < 2:
                continue
            input_ids = torch.tensor([ids], device=device)
            saved_inputs.clear()
            _ = model(input_ids=input_ids)
            for name, inp in saved_inputs.items():
                # RMS = sqrt(mean(x^2) over hidden dim)
                rms = inp.pow(2).mean(dim=-1).sqrt()  # (B, L)
                stats[name].append(rms.mean().item())

    for h in hooks:
        h.remove()
    avg_stats = {name: sum(vals)/len(vals) for name, vals in stats.items()}
    return avg_stats

# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='Qwen/Qwen3.5-2B-Base')
    parser.add_argument('--rank', type=int, default=256)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--lambda_silence', type=float, default=0.01)
    parser.add_argument('--steps', type=int, default=2000)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--domain_files', nargs='+', default=['rnelang.jsonl', 'train.jsonl'])
    parser.add_argument('--save_dir', type=str, default='./multi_graft')
    parser.add_argument('--mode', choices=['train','eval'], default='train')
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print_gpu_info()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    # Load domain texts
    domains = [load_jsonl_texts(f) for f in args.domain_files]
    domain_names = [os.path.splitext(os.path.basename(f))[0] for f in args.domain_files]
    print(f"Domains: {list(zip(domain_names, [len(d) for d in domains]))}")

    model = AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype=torch.float32,
                                                 trust_remote_code=True)
    hidden_dim = model.config.hidden_size

    # Create mutually orthogonal subspaces for each domain
    num_domains = len(domains)
    K = args.rank
    assert num_domains * K <= hidden_dim, f"Not enough hidden dim {hidden_dim} for {num_domains} domains of rank {K}"
    g = torch.Generator(device=device).manual_seed(args.seed)
    Q_full = torch.randn(hidden_dim, hidden_dim, generator=g, device=device, dtype=torch.float32)
    Q_full, _ = torch.linalg.qr(Q_full)
    P_list = [Q_full[:, i*K : (i+1)*K] for i in range(num_domains)]
    print(f"Generated {num_domains} orthogonal subspaces, rank={K}")

    if args.mode == 'train':
        grafts = []
        layer_types_all = None
        for i, (name, in_data) in enumerate(zip(domain_names, domains)):
            # out-of-domain = all other domains
            out_data = [t for j, d in enumerate(domains) if j != i for t in d]
            print(f"\n=== Training graft for {name} (λ={args.lambda_silence}) ===")
            P_i = P_list[i]
            graft, layer_types = train_graft(
                model, tokenizer, P_i, in_data, out_data,
                steps=args.steps, batch_size=args.batch_size, lr=args.lr,
                lambda_silence=args.lambda_silence, device=device
            )
            grafts.append(graft)
            if layer_types_all is None:
                layer_types_all = layer_types
            # Save graft + its P
            os.makedirs(args.save_dir, exist_ok=True)
            torch.save({'graft': graft, 'P': P_i.cpu()}, os.path.join(args.save_dir, f'graft_{name}.pt'))
        # Save metadata
        torch.save({'layer_types': layer_types_all, 'domain_names': domain_names}, os.path.join(args.save_dir, 'meta.pt'))
        print_gpu_info(step='training end')

    elif args.mode == 'eval':
        # Load metadata
        meta = torch.load(os.path.join(args.save_dir, 'meta.pt'), map_location='cpu')
        layer_types_all = meta['layer_types']
        domain_names = meta['domain_names']
        # Load all grafts with their P
        grafts = []
        P_stored_list = []
        for name in domain_names:
            bundle = torch.load(os.path.join(args.save_dir, f'graft_{name}.pt'), map_location='cpu')
            grafts.append(bundle['graft'])
            P_stored_list.append(bundle['P'].to(device))

        # Stack evaluation
        print("\n=== Stack Evaluation ===")
        for eval_i, eval_name in enumerate(domain_names):
            # Single graft
            fresh_model = AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype=torch.float32,
                                                               trust_remote_code=True)
            fresh_model.to(device)
            install_grafts(fresh_model, [(grafts[eval_i], P_stored_list[eval_i])], layer_types_all, device)
            ppl_single = evaluate_ppl(fresh_model, tokenizer, domains[eval_i], device=device)
            rms_single = measure_rmsnorm_stats(fresh_model, tokenizer, domains[eval_i][:10], device=device)

            # Full stack
            stacked_model = AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype=torch.float32,
                                                                 trust_remote_code=True)
            stacked_model.to(device)
            all_pairs = list(zip(grafts, P_stored_list))
            install_grafts(stacked_model, all_pairs, layer_types_all, device)
            ppl_stacked = evaluate_ppl(stacked_model, tokenizer, domains[eval_i], device=device)
            rms_stacked = measure_rmsnorm_stats(stacked_model, tokenizer, domains[eval_i][:10], device=device)

            # Print results
            print(f"\nDomain {eval_name}:")
            print(f"  PPL single: {ppl_single:.4f}, stacked: {ppl_stacked:.4f}, degradation: {ppl_stacked-ppl_single:.4f}")
            # RMS shift
            ratios = []
            for layer in rms_single:
                if layer in rms_stacked and rms_single[layer] > 0:
                    ratio = rms_stacked[layer] / rms_single[layer]
                    ratios.append(ratio)
            if ratios:
                avg_ratio = sum(ratios) / len(ratios)
                print(f"  Mean RMSNorm input scale ratio (stacked/single): {avg_ratio:.4f}")
            else:
                print("  RMSNorm: no layers captured (check dynamic detection)")

        print_gpu_info(step='eval end')

if __name__ == '__main__':
    main()