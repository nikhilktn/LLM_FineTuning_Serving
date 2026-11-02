"""Step 5: compare base vs. fine-tuned answers for a prompt (or a file of prompts)."""
import argparse

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

parser = argparse.ArgumentParser()
parser.add_argument("--base", default="models/base")
parser.add_argument("--adapter", default="outputs/lora/final")
parser.add_argument("--prompt", action="append", help="repeatable")
parser.add_argument("--prompts-file", help="one prompt per line")
parser.add_argument("--system-prompt", default=None)
parser.add_argument("--max-new-tokens", type=int, default=300)
args = parser.parse_args()

prompts = args.prompt or []
if args.prompts_file:
    prompts += [l.strip() for l in open(args.prompts_file) if l.strip()]
if not prompts:
    prompts = ["Explain what LoRA fine-tuning is in two sentences."]

dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
tokenizer = AutoTokenizer.from_pretrained(args.base)
base = AutoModelForCausalLM.from_pretrained(args.base, torch_dtype=dtype, device_map={"": 0})
model = PeftModel.from_pretrained(base, args.adapter)
model.eval()


def generate(prompt):
    msgs = [{"role": "system", "content": args.system_prompt}] if args.system_prompt else []
    msgs.append({"role": "user", "content": prompt})
    ids = tokenizer.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(ids, max_new_tokens=args.max_new_tokens, do_sample=False)
    return tokenizer.decode(out[0, ids.shape[1]:], skip_special_tokens=True)


for p in prompts:
    print("=" * 80, f"\nPROMPT: {p}\n")
    with model.disable_adapter():
        print("--- BASE ---\n" + generate(p) + "\n")
    print("--- FINE-TUNED ---\n" + generate(p) + "\n")
