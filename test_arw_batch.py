import sys
import torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
from torch.utils.data import DataLoader
from train import ARWLinear, prepare_wikitext

original_stdout = sys.stdout
log_file = open("test_arw_batch.log", "w", encoding="utf-8")
sys.stdout = log_file
sys.stderr = log_file

def display(msg=""):
    original_stdout.write(str(msg) + "\n")
    original_stdout.flush()
    log_file.write(str(msg) + "\n")
    log_file.flush()

# 5070 Ti Optimizations
torch.set_float32_matmul_precision('high')
device = torch.device('cuda')

tok = GPT2TokenizerFast.from_pretrained('gpt2')
tok.pad_token = tok.eos_token

display('Preparing small eval batch...')
en = prepare_wikitext(tok, num_samples=8)
en_loader = DataLoader(en, batch_size=1)

orig = GPT2LMHeadModel.from_pretrained('gpt2').to(device).eval()
arw = GPT2LMHeadModel.from_pretrained('gpt2').to(device)
ARWLinear.convert_gpt2_layers(arw, core_rank=8, adapter_rank=32, device=device)
arw.eval()

batch = next(iter(en_loader))
input_ids = batch['input_ids'].to(device, non_blocking=True)
mask = batch['attention_mask'].to(device, non_blocking=True)

with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
    loss_orig = orig(input_ids, attention_mask=mask, labels=input_ids).loss.item()
    loss_arw = arw(input_ids, attention_mask=mask, labels=input_ids).loss.item()

display(f"Orig loss on eval batch: {loss_orig:.6f}")
display(f"ARW  loss on eval batch: {loss_arw:.6f}")

# Layer-wise check for first block c_attn
orig_conv = orig.transformer.h[0].attn.c_attn
arw_conv = arw.transformer.h[0].attn.c_attn
hidden = torch.randn(1, 1, orig.config.hidden_size).to(device, dtype=torch.bfloat16)

with torch.no_grad():
    # Orig expects float32 if not autocasted, but we'll cast inputs appropriately or use autocast
    with torch.autocast('cuda', dtype=torch.bfloat16):
        out_conv = orig_conv(hidden)
        out_arw_ = arw_conv(hidden)
display(f"1st c_attn max diff: {(out_conv - out_arw_).float().abs().max().item():.6f}")
