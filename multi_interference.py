# multi_interference.py
# Trains multiple grafts, stacks them, measures PPL degradation and RMSNorm scale shifts.
# Compares baseline (λ=0) vs acoustic treatment (λ>0).

import argparse, json, torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from itertools import cycle
from collections import defaultdict
import os, copy

from grafting import (
    make_arw_basis, ARWTrainer, compress_graft, decompress_graft, get_ffn_param_names,
)

# ----------------------------------------------------------------------
# 1. Data loading
# ----------------------------------------------------------------------
def load_jsonl_texts(path):
    texts = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            obj = json.loads(line.strip())
            for v in obj.values():
                if isinstance(v, str):
                    texts.append(obj.get('instruction','') + ' ' + obj.get('output',''))
                    break
    return texts

class InfiniteTextDataset(Dataset):
    """Yields tokenized chunks of max_len from a list of texts, cycling forever."""
    def __init__(self, texts, tokenizer, max_len=1024):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.data = []
        for t in texts:
            ids = tokenizer.encode(t, add_special_tokens=False)
            if ids:
                self.data.append(ids)
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        ids = self.data[idx % len(self.data)]
        if len(ids) > self.max_len:
            start = torch.randint(0, len(ids)-self.max_len, (1,)).item()
            ids = ids[start:start+self.max_len]
        else:
            ids = ids + [self.tokenizer.pad_token_id]*(self.max_len-len(ids))
        return torch.tensor(ids, dtype=torch.long)

# ----------------------------------------------------------------------
# 2. Acoustic training (single graft)
# ----------------------------------------------------------------------
def train_graft(model, tokenizer, basis_P, in_texts, out_texts,
                steps=5000, batch_size=4, lr=1e-4, lambda_silence=0.01,
                device='cuda', log_interval=100):
    model.to(device)
    model.eval()
    trainable_names = set(get_ffn_param_names(model).keys())
    for n, p in model.named_parameters():
        p.requires_grad = n in trainable_names

    trainer = ARWTrainer(model, basis_P, device)
    base_weights = {n: p.clone().detach().to(device) for n, p in model.named_parameters() if n in trainer.layer_types}

    # modules for forward hooks (to capture graft input)
    grafted_modules = {}
    for mod_name, module in model.named_modules():
        if isinstance(module, nn.Linear):
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

    # mixed-domain dataset (80% in, 20% out)
    class MixedDomain(Dataset):
        def __init__(self):
            self.in_dataset = InfiniteTextDataset(in_texts, tokenizer)
            self.out_dataset = InfiniteTextDataset(out_texts, tokenizer)
        def __len__(self): return 100000  # large enough
        def __getitem__(self, _):
            in_len = int(1024 * 0.8)
            out_len = 1024 - in_len
            in_ids = self.in_dataset[torch.randint(0, len(self.in_dataset), (1,)).item()][:in_len]
            out_ids = self.out_dataset[torch.randint(0, len(self.out_dataset), (1,)).item()][:out_len]
            input_ids = torch.cat([in_ids, out_ids])
            mask = torch.cat([torch.ones(in_len), torch.zeros(out_len)])
            return input_ids[:1024], mask[:1024]

    dataset = MixedDomain()
    loader = cycle(DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True))

    optimizer = torch.optim.AdamW(
        [p for n, p in model.named_parameters() if n in trainable_names], lr=lr
    )
    trainer.attach_hooks()
    losses_lm, losses_sil = [], []

    for step in range(1, steps+1):
        input_ids, mask = next(loader)
        input_ids, mask = input_ids.to(device), mask.to(device)

        saved_inputs.clear()
        outputs = model(input_ids=input_ids, labels=input_ids)
        logits = outputs.logits

        # LM loss on in-domain positions
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = input_ids[..., 1:].contiguous()
        shift_mask = mask[..., 1:].contiguous()
        ce = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)),
                             shift_labels.view(-1), reduction='none').view(shift_labels.shape)
        in_mask_sum = shift_mask.sum()
        lm_loss = (ce * shift_mask).sum() / in_mask_sum if in_mask_sum > 0 else torch.tensor(0.0, device=device)

        # Silence loss: graft's residual contribution on out-of-domain positions
        silence_loss = torch.tensor(0.0, device=device)
        out_mask = 1.0 - mask
        layers = 0
        for name, module in grafted_modules.items():
            if name not in saved_inputs:
                continue
            x = saved_inputs[name]
            current_w = model.get_parameter(name)
            delta_w = current_w - base_weights[name]
            delta_out = F.linear(x, delta_w)
            token_norm = delta_out.pow(2).mean(dim=-1)  # per-dimension mean
            if token_norm.size(1) == out_mask.size(1):
                out_count = out_mask.sum()
                if out_count > 0:
                    silence_loss += (token_norm * out_mask).sum() / out_count
            layers += 1
        if layers > 0:
            silence_loss = silence_loss / layers

        total_loss = lm_loss + lambda_silence * silence_loss
        total_loss.backward()

        if step % 1 == 0:  # simple update each step (no grad accum)
            old_weights = {n: p.clone().detach().to(device) for n, p in model.named_parameters() if n in trainer.layer_types}
            optimizer.step()
            optimizer.zero_grad()
            trainer.clamp_step(old_weights)

        losses_lm.append(lm_loss.item())
        losses_sil.append(silence_loss.item())
        if step % log_interval == 0:
            print(f"Step {step:5d} | LM: {sum(losses_lm[-100:])/min(len(losses_lm),100):.4f} | Sil: {sum(losses_sil[-100:])/min(len(losses_sil),100):.6f}")

    trainer.remove_hooks()
    for h in hooks_fwd: h.remove()

    # compress graft
    graft = {}
    with torch.no_grad():
        for name in base_weights:
            delta = model.get_parameter(name).detach() - base_weights[name]
            lt = trainer.layer_types[name]
            graft[name] = compress_graft(delta, basis_P.to(device), lt)

    return graft, trainer.layer_types

# ----------------------------------------------------------------------
# 3. Stack grafts onto a model
# ----------------------------------------------------------------------
def install_grafts(model, grafts, P, layer_types, device):
    for name, G in grafts.items():
        lt = layer_types[name]
        delta = decompress_graft(G, P, lt).to(device)
        param = model.get_parameter(name)
        param.data.add_(delta)

# ----------------------------------------------------------------------
# 4. PPL evaluation
# ----------------------------------------------------------------------
def evaluate_ppl(model, tokenizer, texts, max_len=512, device='cuda'):
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    with torch.no_grad():
        for text in texts:
            ids = tokenizer.encode(text, add_special_tokens=False)[:max_len]
            if len(ids) < 2: continue
            input_ids = torch.tensor([ids], device=device)
            outputs = model(input_ids=input_ids, labels=input_ids)
            loss = outputs.loss
            total_loss += loss.item() * (len(ids)-1)
            total_tokens += (len(ids)-1)
    return total_loss / total_tokens if total_tokens > 0 else float('inf')

# ----------------------------------------------------------------------
# 5. RMSNorm scale measurement
# ----------------------------------------------------------------------
def measure_rmsnorm_stats(model, tokenizer, texts, device='cuda', max_len=512):
    """Returns dict: layer_name -> mean RMS of input (averaged over tokens)."""
    from transformers.models.qwen2.modeling_qwen2 import Qwen2RMSNorm
    rms_modules = {}
    for name, module in model.named_modules():
        if isinstance(module, Qwen2RMSNorm):
            rms_modules[name] = module

    # hook to capture input
    saved_inputs = {}
    def hook_fn(name):
        def hook(module, input, output):
            saved_inputs[name] = input[0].detach()  # (B, L, H)
        return hook
    hooks = [mod.register_forward_hook(hook_fn(name)) for name, mod in rms_modules.items()]

    stats = defaultdict(list)
    with torch.no_grad():
        for text in texts[:10]:  # small sample for speed
            ids = tokenizer.encode(text, add_special_tokens=False)[:max_len]
            if len(ids) < 2: continue
            input_ids = torch.tensor([ids], device=device)
            saved_inputs.clear()
            _ = model(input_ids=input_ids)
            for name, inp in saved_inputs.items():
                # RMS = sqrt(mean(x^2) over hidden dim)
                rms = inp.pow(2).mean(dim=-1).sqrt()  # (B, L)
                stats[name].append(rms.mean().item())

    for h in hooks: h.remove()
    # average over texts
    avg_stats = {name: sum(vals)/len(vals) for name, vals in stats.items()}
    return avg_stats

# ----------------------------------------------------------------------
# 6. Main experiment
# ----------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='Qwen/Qwen3.5-2B-Base')
    parser.add_argument('--rank', type=int, default=256)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--lambda_silence', type=float, default=0.01)
    parser.add_argument('--steps', type=int, default=2000)  # shorter for demo
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--domain_files', nargs='+', default=['rnelang.jsonl', 'train.jsonl'])
    parser.add_argument('--save_dir', type=str, default='./multi_graft')
    parser.add_argument('--mode', choices=['train','eval'], default='train')
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    # Load domain texts
    domains = [load_jsonl_texts(f) for f in args.domain_files]
    domain_names = [os.path.splitext(os.path.basename(f))[0] for f in args.domain_files]
    print(f"Domains: {list(zip(domain_names, [len(d) for d in domains]))}")

    model = AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype=torch.float32, trust_remote_code=True)
    hidden_dim = model.config.hidden_size
    P = make_arw_basis(hidden_dim, args.rank, args.seed, device)

    if args.mode == 'train':
        grafts = []
        layer_types_all = None
        for i, (name, in_data) in enumerate(zip(domain_names, domains)):
            # out-of-domain = union of all other domains
            out_data = [t for j, d in enumerate(domains) if j != i for t in d]
            print(f"\n=== Training graft for {name} (λ={args.lambda_silence}) ===")
            graft, layer_types = train_graft(
                model, tokenizer, P, in_data, out_data,
                steps=args.steps, batch_size=args.batch_size, lr=args.lr,
                lambda_silence=args.lambda_silence, device=device
            )
            grafts.append(graft)
            if layer_types_all is None:
                layer_types_all = layer_types
            # Save individual graft
            os.makedirs(args.save_dir, exist_ok=True)
            torch.save(graft, os.path.join(args.save_dir, f'graft_{name}.pt'))

        # Save metadata
        torch.save({'layer_types': layer_types_all}, os.path.join(args.save_dir, 'meta.pt'))

    elif args.mode == 'eval':
        # Load pre-trained grafts
        grafts = []
        for name in domain_names:
            g = torch.load(os.path.join(args.save_dir, f'graft_{name}.pt'), map_location='cpu')
            grafts.append(g)
        meta = torch.load(os.path.join(args.save_dir, 'meta.pt'))
        layer_types_all = meta['layer_types']

    # Stack evaluation
    print("\n=== Stack Evaluation ===")
    for eval_i, eval_name in enumerate(domain_names):
        # 1. Single-graft (only its own graft)
        fresh_model = AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype=torch.float32, trust_remote_code=True)
        install_grafts(fresh_model, grafts[eval_i], P, layer_types_all, device)
        ppl_single = evaluate_ppl(fresh_model, tokenizer, domains[eval_i][:100], device=device)  # sample 100 texts
        rms_single = measure_rmsnorm_stats(fresh_model, tokenizer, domains[eval_i][:10], device=device)

        # 2. Full stack (all grafts)
        stacked_model = AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype=torch.float32, trust_remote_code=True)
        for g in grafts:
            install_grafts(stacked_model, g, P, layer_types_all, device)
        ppl_stacked = evaluate_ppl(stacked_model, tokenizer, domains[eval_i][:100], device=device)
        rms_stacked = measure_rmsnorm_stats(stacked_model, tokenizer, domains[eval_i][:10], device=device)

        print(f"\nDomain {eval_name}:")
        print(f"  PPL single: {ppl_single:.4f}, stacked: {ppl_stacked:.4f}, degradation: {ppl_stacked-ppl_single:.4f}")
        # RMS shift: average ratio across layers
        ratios = []
        for layer in rms_single:
            if layer in rms_stacked:
                ratio = rms_stacked[layer] / rms_single[layer] if rms_single[layer]>0 else 1.0
                ratios.append(ratio)
        if ratios:
            avg_ratio = sum(ratios)/len(ratios)
            print(f"  Mean RMSNorm input scale ratio (stacked/single): {avg_ratio:.4f}")

if __name__ == '__main__':
    main()