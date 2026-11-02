"""Step 10: post-training quantization with llm-compressor (output loads directly in vLLM).

Schemes:
  fp8    FP8 weights + dynamic FP8 activations. No calibration data. Fast kernels on
         Ada/Hopper (compute capability >= 8.9); on Ampere vLLM runs it weight-only.
  w8a8   INT8 weights + activations (SmoothQuant + GPTQ). Needs calibration. Fast on Ampere+.
  w4a16  INT4 weights, 16-bit activations (GPTQ, group size 128). Needs calibration.
         Smallest model; best for memory-bound, low-concurrency serving.

Run in the quant venv: source ~/venvs/quant/bin/activate
"""
import argparse
import json

from datasets import Dataset
from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import GPTQModifier, QuantizationModifier
from llmcompressor.modifiers.transform import SmoothQuantModifier
from transformers import AutoModelForCausalLM, AutoTokenizer

parser = argparse.ArgumentParser()
parser.add_argument("--model", default="models/merged-qwen7b", help="full (merged) model dir")
parser.add_argument("--scheme", choices=["fp8", "w8a8", "w4a16"], required=True)
parser.add_argument("--out", default=None, help="default: <model>-<scheme>")
parser.add_argument("--calib-file", default="data/train.jsonl")
parser.add_argument("--num-calib", type=int, default=256)
parser.add_argument("--max-length", type=int, default=2048)
args = parser.parse_args()
out_dir = args.out or f"{args.model.rstrip('/')}-{args.scheme}"

tokenizer = AutoTokenizer.from_pretrained(args.model)
model = AutoModelForCausalLM.from_pretrained(args.model, dtype="auto")

# lm_head stays in 16-bit: it is sensitive to quantization and small relative to the model.
if args.scheme == "fp8":
    recipe = QuantizationModifier(targets="Linear", scheme="FP8_DYNAMIC", ignore=["lm_head"])
elif args.scheme == "w8a8":
    recipe = [
        SmoothQuantModifier(smoothing_strength=0.8),  # moves activation outliers into weights
        GPTQModifier(targets="Linear", scheme="W8A8", ignore=["lm_head"]),
    ]
else:
    recipe = GPTQModifier(targets="Linear", scheme="W4A16", ignore=["lm_head"])

calib = None
if args.scheme != "fp8":
    # Calibrate on your own fine-tuning data so activation statistics match real traffic.
    rows = [json.loads(l) for l in open(args.calib_file)][: args.num_calib]
    texts = [tokenizer.apply_chat_template(r["prompt"] + r["completion"], tokenize=False) for r in rows]
    calib = Dataset.from_dict({"text": texts}).map(
        lambda s: tokenizer(s["text"], max_length=args.max_length, truncation=True, add_special_tokens=False),
        remove_columns=["text"],
    )

oneshot(
    model=model,
    dataset=calib,
    recipe=recipe,
    max_seq_length=args.max_length,
    num_calibration_samples=len(calib) if calib is not None else None,
)

model.save_pretrained(out_dir, save_compressed=True)
tokenizer.save_pretrained(out_dir)
print(f"Quantized ({args.scheme}) model saved to {out_dir}")
