"""Step 9: merge a LoRA adapter into the base weights -> standalone model.

Runs on CPU in bf16, so it needs ~2 bytes/param of system RAM (7B ~ 16 GB).
"""
import argparse

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

parser = argparse.ArgumentParser()
parser.add_argument("--base", default="models/base")
parser.add_argument("--adapter", default="outputs/lora/final")
parser.add_argument("--out", default="models/merged")
args = parser.parse_args()

base = AutoModelForCausalLM.from_pretrained(
    args.base, torch_dtype=torch.bfloat16, device_map="cpu", low_cpu_mem_usage=True
)
model = PeftModel.from_pretrained(base, args.adapter)
model = model.merge_and_unload()

model.save_pretrained(args.out, safe_serialization=True, max_shard_size="5GB")
AutoTokenizer.from_pretrained(args.base).save_pretrained(args.out)
print(f"Merged model saved to {args.out}")
