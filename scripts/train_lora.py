"""Step 6: LoRA / QLoRA supervised fine-tuning with TRL + PEFT on a single GPU."""
import argparse
import json
import os
import time

import torch
from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer

parser = argparse.ArgumentParser()
parser.add_argument("--model", default="models/base")
parser.add_argument("--train-file", default="data/train.jsonl")
parser.add_argument("--eval-file", default="data/eval.jsonl")
parser.add_argument("--output-dir", default="outputs/lora")
parser.add_argument("--qlora", action="store_true", help="load base model in 4-bit NF4")
parser.add_argument("--epochs", type=float, default=2)
parser.add_argument("--lr", type=float, default=2e-4)
parser.add_argument("--batch-size", type=int, default=2)
parser.add_argument("--grad-accum", type=int, default=8)
parser.add_argument("--max-length", type=int, default=2048)
parser.add_argument("--lora-r", type=int, default=16)
parser.add_argument("--lora-alpha", type=int, default=32)
parser.add_argument("--lora-dropout", type=float, default=0.05)
parser.add_argument("--save-steps", type=int, default=200)
parser.add_argument("--resume", default=None, help="checkpoint dir to resume from")
parser.add_argument("--report-to", default="tensorboard", help="tensorboard, wandb or none")
parser.add_argument("--logging-steps", type=int, default=1, help="1 shows the warmup ramp in TensorBoard")
parser.add_argument("--no-gradient-checkpointing", action="store_true",
                    help="faster (no activation recompute) but needs much more GPU memory")
args = parser.parse_args()

bf16 = torch.cuda.is_bf16_supported()  # Ampere (RTX 30xx / A100) and newer
compute_dtype = torch.bfloat16 if bf16 else torch.float16

tokenizer = AutoTokenizer.from_pretrained(args.model)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

quant_config = None
if args.qlora:
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=True,
    )

model = AutoModelForCausalLM.from_pretrained(
    args.model,
    quantization_config=quant_config,
    dtype=compute_dtype,
    device_map={"": 0},
    attn_implementation="sdpa",
)
model.config.use_cache = False  # incompatible with gradient checkpointing

# "all-linear" = q,k,v,o + MLP projections (excludes lm_head), all of which vLLM can serve.
lora_config = LoraConfig(
    r=args.lora_r,
    lora_alpha=args.lora_alpha,
    lora_dropout=args.lora_dropout,
    target_modules="all-linear",
    bias="none",
    task_type="CAUSAL_LM",
)

data_files = {"train": args.train_file}
if args.eval_file and os.path.exists(args.eval_file):
    data_files["eval"] = args.eval_file
ds = load_dataset("json", data_files=data_files)
has_eval = "eval" in ds

sft_config = SFTConfig(
    output_dir=args.output_dir,
    num_train_epochs=args.epochs,
    per_device_train_batch_size=args.batch_size,
    per_device_eval_batch_size=args.batch_size,
    gradient_accumulation_steps=args.grad_accum,
    learning_rate=args.lr,
    lr_scheduler_type="cosine",
    warmup_steps=0.03,  # < 1 means a ratio of total steps
    weight_decay=0.0,
    max_grad_norm=1.0,
    max_length=args.max_length,
    packing=False,
    completion_only_loss=True,  # loss on the assistant answer only
    gradient_checkpointing=not args.no_gradient_checkpointing,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    optim="paged_adamw_8bit" if args.qlora else "adamw_torch_fused",
    bf16=bf16,
    fp16=not bf16,
    logging_steps=args.logging_steps,
    save_strategy="steps",
    save_steps=args.save_steps,
    save_total_limit=3,
    eval_strategy="steps" if has_eval else "no",
    eval_steps=args.save_steps,
    report_to=args.report_to,
    seed=42,
)

trainer = SFTTrainer(
    model=model,
    args=sft_config,
    train_dataset=ds["train"],
    eval_dataset=ds["eval"] if has_eval else None,
    processing_class=tokenizer,
    peft_config=lora_config,
)
trainer.model.print_trainable_parameters()

torch.cuda.reset_peak_memory_stats()
start = time.time()
trainer.train(resume_from_checkpoint=args.resume)
summary = {
    "train_minutes": round((time.time() - start) / 60, 1),
    "peak_gpu_mem_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2),
    "qlora": args.qlora,
    "lora_r": args.lora_r,
    "gradient_checkpointing": not args.no_gradient_checkpointing,
    "trainable_params": sum(p.numel() for p in trainer.model.parameters() if p.requires_grad),
}

final_dir = os.path.join(args.output_dir, "final")
trainer.save_model(final_dir)  # saves only the adapter
tokenizer.save_pretrained(final_dir)
if has_eval:
    summary["eval_loss"] = round(trainer.evaluate()["eval_loss"], 4)
with open(os.path.join(args.output_dir, "train_summary.json"), "w") as f:
    json.dump(summary, f, indent=2)
print("Training summary:", summary)
print(f"Adapter saved to {final_dir}")
