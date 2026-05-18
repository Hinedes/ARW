# acoustic_treatment.py
# Trains a graft with acoustic suppression using real JSONL files.
# rnelang.jsonl → in‑domain (RNE programming language)
# train.jsonl   → out‑of‑domain (Kyrgyz Law)
#
# Usage:
#   python acoustic_treatment.py --lambda_silence 0.0 --save_dir graft_baseline --eval_bleed
#   python acoustic_treatment.py --lambda_silence 0.01 --save_dir graft_acoustic --eval_bleed

import argparse, json, torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from itertools import cycle
import os

from grafting import (
    make_arw_basis, projection_from_basis, ARWTrainer, compress_graft, decompress_graft,
    get_ffn_param_names,
)

# ----------------------------------------------------------------------
# 1. Load JSONL files
# ----------------------------------------------------------------------
def load_jsonl_texts(path):
    """Return list of strings from a JSONL file (field 'text' or first value)."""
    texts = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            # Try common fields
            if 'text' in obj:
                texts.append(obj['text'])
            elif 'content' in obj:
                texts.append(obj['content'])
            else:
                # use the first string value
                for v in obj.values():
                    if isinstance(v, str):
                        texts.append(v)
                        break
    return texts

# ----------------------------------------------------------------------
# 2. Mixed domain dataset (80% in‑domain, 20% out‑of‑domain)
# ----------------------------------------------------------------------
class MixedDomainDataset(Dataset):
    def __init__(self, in_domain_texts, out_domain_texts, tokenizer, max_len=1024):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.in_tokens = []
        self.out_tokens = []
        for text in in_domain_texts:
            tok = tokenizer.encode(text, add_special_tokens=False)
            if tok:
                self.in_tokens.append(tok)
        for text in out_domain_texts:
            tok = tokenizer.encode(text, add_special_tokens=False)
            if tok:
                self.out_tokens.append(tok)

    def __len__(self):
        return min(len(self.in_tokens), len(self.out_tokens))

    def __getitem__(self, idx):
        in_seq = self.in_tokens[idx % len(self.in_tokens)]
        out_seq = self.out_tokens[idx % len(self.out_tokens)]
        half = self.max_len // 2
        in_chunk = in_seq[:half]
        out_chunk = out_seq[:half]
        input_ids = in_chunk + out_chunk
        mask = [1] * len(in_chunk) + [0] * len(out_chunk)
        # truncate to max_len
        input_ids = input_ids[:self.max_len]
        mask = mask[:self.max_len]
        # pad if needed
        if len(input_ids) < self.max_len:
            pad_len = self.max_len - len(input_ids)
            input_ids += [self.tokenizer.pad_token_id] * pad_len
            mask += [0] * pad_len
        return torch.tensor(input_ids, dtype=torch.long), torch.tensor(mask, dtype=torch.float32)

# ----------------------------------------------------------------------
# 3. Acoustic training loop
# ----------------------------------------------------------------------
def train_acoustic(
    model, tokenizer, basis_P, in_domain_texts, out_domain_texts,
    steps=5000, batch_size=4, lr=1e-4, lambda_silence=0.01,
    device='cuda', grad_accum=1, log_interval=100, save_dir='./graft_acoustic'
):
    model.to(device)
    model.eval()
    trainable_names = set(get_ffn_param_names(model).keys())
    for n, p in model.named_parameters():
        p.requires_grad = n in trainable_names

    trainer = ARWTrainer(model, basis_P, device)
    Pi = trainer.Pi
    # snapshot base weights
    base_weights = {n: p.clone().detach().to(device) for n, p in model.named_parameters() if n in trainer.layer_types}

    # collect modules for forward hooks
    grafted_modules = {}
    for mod_name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            for pname in ['weight']:
                full = f"{mod_name}.{pname}"
                if full in trainer.layer_types:
                    grafted_modules[full] = module

    saved_inputs = {}
    def forward_hook_fn(name):
        def hook(module, input, output):
            saved_inputs[name] = input[0].detach()
        return hook

    hooks_fwd = [mod.register_forward_hook(forward_hook_fn(name)) for name, mod in grafted_modules.items()]

    dataset = MixedDomainDataset(in_domain_texts, out_domain_texts, tokenizer, max_len=1024)
    loader = cycle(DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True))

    optimizer = torch.optim.AdamW(
        [p for n, p in model.named_parameters() if n in trainable_names], lr=lr
    )
    trainer.attach_hooks()
    losses_lm, losses_sil = [], []

    for step in range(1, steps + 1):
        input_ids, mask = next(loader)
        input_ids = input_ids.to(device)
        mask = mask.to(device)

        saved_inputs.clear()
        outputs = model(input_ids=input_ids, labels=input_ids)
        logits = outputs.logits

        # LM loss only on in‑domain tokens
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = input_ids[..., 1:].contiguous()
        shift_mask = mask[..., 1:].contiguous()
        ce = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)),
                             shift_labels.view(-1), reduction='none').view(shift_labels.shape)
        in_mask_sum = shift_mask.sum()
        lm_loss = (ce * shift_mask).sum() / in_mask_sum if in_mask_sum > 0 else torch.tensor(0.0, device=device)

        # silence loss on out‑of‑domain tokens
        silence_loss = torch.tensor(0.0, device=device)
        out_mask = 1.0 - mask
        layers = 0
        for name, module in grafted_modules.items():
            if name not in saved_inputs:
                continue
            x = saved_inputs[name]                     # (B, L, H)
            current_w = model.get_parameter(name)      # current weight
            base_w = base_weights[name]                # frozen base
            delta_w = current_w - base_w               # only subspace component (thanks to ARW)
            delta_out = F.linear(x, delta_w)           # (B, L, out_features)
            token_norm = delta_out.pow(2).mean(dim=-1)  # (B, L)
            if token_norm.size(1) == out_mask.size(1):
                out_count = out_mask.sum()
                if out_count > 0:
                    silence_loss += (token_norm * out_mask).sum() / out_count
            layers += 1
        if layers > 0:
            silence_loss = silence_loss / layers

        total_loss = lm_loss + lambda_silence * silence_loss
        total_loss.backward()

        if step % grad_accum == 0:
            old_weights = {n: p.clone().detach().to(device) for n, p in model.named_parameters() if n in trainer.layer_types}
            optimizer.step()
            optimizer.zero_grad()
            trainer.clamp_step(old_weights)

        losses_lm.append(lm_loss.item())
        losses_sil.append(silence_loss.item())

        if step % log_interval == 0:
            avg_lm = sum(losses_lm[-100:]) / min(len(losses_lm), 100)
            avg_sil = sum(losses_sil[-100:]) / min(len(losses_sil), 100)
            print(f"Step {step:5d} | LM Loss: {avg_lm:.4f} | Silence: {avg_sil:.6f}")

    trainer.remove_hooks()
    for h in hooks_fwd:
        h.remove()

    # compress graft
    graft = {}
    with torch.no_grad():
        for name in base_weights:
            current_w = model.get_parameter(name).detach()
            delta = current_w - base_weights[name]
            lt = trainer.layer_types[name]
            graft[name] = compress_graft(delta, basis_P.to(device), lt)

    os.makedirs(save_dir, exist_ok=True)
    torch.save(graft, os.path.join(save_dir, 'graft.pt'))
    print(f"Graft saved to {save_dir}/graft.pt")
    return graft, base_weights, trainer.layer_types

# ----------------------------------------------------------------------
# 4. Ghost meter (bleed measurement)
# ----------------------------------------------------------------------
def measure_bleed(model, graft, P, layer_types, in_loader, out_loader, device):
    model.to(device)
    model.eval()
    # install graft
    for name, G in graft.items():
        lt = layer_types[name]
        delta = decompress_graft(G, P, lt).to(device)
        param = model.get_parameter(name)
        param.data.add_(delta)

    # collect modules
    grafted_modules = {}
    for mod_name, mod in model.named_modules():
        if isinstance(mod, nn.Linear):
            for pname in ['weight']:
                full = f"{mod_name}.{pname}"
                if full in layer_types:
                    grafted_modules[full] = mod

    saved_inputs = {}
    def hook_fn(name):
        def hook(module, input, output):
            saved_inputs[name] = input[0].detach()
        return hook

    hooks = [mod.register_forward_hook(hook_fn(name)) for name, mod in grafted_modules.items()]

    def process(loader, label):
        norms_sum = 0.0
        count = 0
        with torch.no_grad():
            for batch in loader:
                input_ids = batch[0].to(device)
                saved_inputs.clear()
                _ = model(input_ids=input_ids)
                for name, module in grafted_modules.items():
                    if name not in saved_inputs:
                        continue
                    x = saved_inputs[name]
                    current_w = model.get_parameter(name)
                    delta = decompress_graft(graft[name], P, layer_types[name]).to(device)
                    dout = F.linear(x, delta)
                    token_norm = delta_out.pow(2).mean(dim=-1)  # scalar
                    norms_sum += token_norm.item()
                    count += 1
        for h in hooks:
            h.remove()
        return norms_sum / count if count > 0 else 0.0

    avg_in = process(in_loader, "in")
    avg_out = process(out_loader, "out")
    ratio = avg_out / avg_in if avg_in > 0 else float('inf')
    return avg_in, avg_out, ratio

# ----------------------------------------------------------------------
# 5. Main experiment
# ----------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='Qwen/Qwen3.5-2B-Base')
    parser.add_argument('--rank', type=int, default=256)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--lambda_silence', type=float, default=0.01)
    parser.add_argument('--steps', type=int, default=5000)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--save_dir', type=str, default='./graft_acoustic')
    parser.add_argument('--eval_bleed', action='store_true')
    parser.add_argument('--in_domain_file', type=str, default='rnelang.jsonl')
    parser.add_argument('--out_domain_file', type=str, default='train.jsonl')
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    in_texts = load_jsonl_texts(args.in_domain_file)
    out_texts = load_jsonl_texts(args.out_domain_file)
    print(f"Loaded {len(in_texts)} in‑domain examples, {len(out_texts)} out‑of‑domain examples.")

    model = AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype=torch.float32, trust_remote_code=True)
    hidden_dim = model.config.hidden_size
    P = make_arw_basis(hidden_dim, args.rank, args.seed, device)

    graft, base_weights, layer_types = train_acoustic(
        model, tokenizer, P, in_texts, out_texts,
        steps=args.steps, batch_size=args.batch_size, lr=args.lr,
        lambda_silence=args.lambda_silence, device=device, save_dir=args.save_dir
    )

    if args.eval_bleed:
        print("\n--- Ghost Meter ---")
        # simple dataloaders for bleed measurement (full sequences, max_len=512)
        class SimpleDataset(Dataset):
            def __init__(self, texts, tokenizer, max_len=512):
                self.data = []
                for t in texts:
                    ids = tokenizer.encode(t, add_special_tokens=False)[:max_len]
                    if ids:
                        self.data.append(ids)
            def __len__(self): return len(self.data)
            def __getitem__(self, i): return torch.tensor(self.data[i])

        in_loader = DataLoader(SimpleDataset(in_texts, tokenizer), batch_size=2, shuffle=False)
        out_loader = DataLoader(SimpleDataset(out_texts, tokenizer), batch_size=2, shuffle=False)

        model_clean = AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype=torch.float32, trust_remote_code=True)
        avg_in, avg_out, ratio = measure_bleed(model_clean, graft, P, layer_types,
                                                in_loader, out_loader, device)
        print(f"In‑domain norm : {avg_in:.6f}")
        print(f"Out‑of‑domain norm : {avg_out:.6f}")
        print(f"Bleed ratio (out/in): {ratio:.4f}")

if __name__ == '__main__':
    main()