"""Steps 5, 7, 11: offline quality check of any model vLLM can load (base, base+LoRA, merged, quantized).

Measures perplexity of the reference answers in eval.jsonl (lower = model assigns higher
probability to the answers you trained it to give), and saves a few greedy generations to
read side by side. Results go to results/eval_<label>.json.

Run in the serve venv with no vLLM server running on the GPU.
"""
import argparse
import json
import math
import os

from vllm import LLM, SamplingParams
from vllm.inputs import TokensPrompt
from vllm.lora.request import LoRARequest

parser = argparse.ArgumentParser()
parser.add_argument("--model", default="models/base")
parser.add_argument("--adapter", default=None, help="LoRA adapter dir (optional)")
parser.add_argument("--label", required=True, help="name for this run, e.g. base, lora-r16, w4a16")
parser.add_argument("--eval-file", default="data/eval.jsonl")
parser.add_argument("--num-samples", type=int, default=200)
parser.add_argument("--num-generate", type=int, default=5)
parser.add_argument("--max-model-len", type=int, default=4096)
parser.add_argument("--gpu-util", type=float, default=0.85)
parser.add_argument("--max-lora-rank", type=int, default=16)
parser.add_argument("--out-dir", default="results")
args = parser.parse_args()


def dir_size_gb(path):
    return sum(os.path.getsize(os.path.join(d, f)) for d, _, fs in os.walk(path) for f in fs) / 1e9


rows = [json.loads(l) for l in open(args.eval_file)][: args.num_samples]

llm = LLM(
    model=args.model,
    max_model_len=args.max_model_len,
    gpu_memory_utilization=args.gpu_util,
    enable_lora=bool(args.adapter),
    max_lora_rank=args.max_lora_rank,
)
lora = LoRARequest("adapter", 1, args.adapter) if args.adapter else None
tok = llm.get_tokenizer()

# Token ids of prompt+answer, and where the answer starts, so only answer tokens are scored.
seqs, starts = [], []
for r in rows:
    prompt_text = tok.apply_chat_template(r["prompt"], add_generation_prompt=True, tokenize=False)
    full_text = tok.apply_chat_template(r["prompt"] + r["completion"], tokenize=False)
    p_ids = tok(prompt_text, add_special_tokens=False)["input_ids"]
    f_ids = tok(full_text, add_special_tokens=False)["input_ids"][: args.max_model_len - 1]
    if f_ids[: len(p_ids)] != p_ids or len(f_ids) <= len(p_ids):
        continue
    seqs.append(f_ids)
    starts.append(len(p_ids))

outs = llm.generate(
    [TokensPrompt(prompt_token_ids=ids) for ids in seqs],
    SamplingParams(max_tokens=1, prompt_logprobs=1),
    lora_request=lora,
)
nll, n_tokens = 0.0, 0
for out, ids, start in zip(outs, seqs, starts):
    for i in range(start, len(ids)):
        nll -= out.prompt_logprobs[i][ids[i]].logprob
        n_tokens += 1
ppl = math.exp(nll / n_tokens)

gen_rows = rows[: args.num_generate]
gens = llm.chat([r["prompt"] for r in gen_rows], SamplingParams(max_tokens=256, temperature=0), lora_request=lora)

result = {
    "label": args.label,
    "model": args.model,
    "adapter": args.adapter,
    "size_gb": round(dir_size_gb(args.model) + (dir_size_gb(args.adapter) if args.adapter else 0), 2),
    "perplexity": round(ppl, 4),
    "scored_examples": len(seqs),
    "scored_tokens": n_tokens,
    "samples": [
        {"prompt": r["prompt"][-1]["content"], "reference": r["completion"][0]["content"], "output": g.outputs[0].text}
        for r, g in zip(gen_rows, gens)
    ],
}
os.makedirs(args.out_dir, exist_ok=True)
path = os.path.join(args.out_dir, f"eval_{args.label}.json")
with open(path, "w") as f:
    json.dump(result, f, indent=2, ensure_ascii=False)
print(f"[{args.label}] perplexity={ppl:.3f} over {n_tokens} answer tokens, size={result['size_gb']} GB -> {path}")
