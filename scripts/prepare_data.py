"""Step 3: convert raw data into conversational prompt/completion JSONL for TRL.

Accepted input rows:
  {"instruction": ..., "input": ..., "output": ...}    (Alpaca style)
  {"messages": [{"role": ..., "content": ...}, ...]}  (chat style; last assistant turn = completion)
"""
import argparse
import json
import os

from datasets import load_dataset

parser = argparse.ArgumentParser()
src = parser.add_mutually_exclusive_group(required=True)
src.add_argument("--input", help="local .jsonl file")
src.add_argument("--hf-dataset", help="HF dataset id, e.g. yahma/alpaca-cleaned")
parser.add_argument("--split", default="train")
parser.add_argument("--out-dir", default="data")
parser.add_argument("--system-prompt", default=None)
parser.add_argument("--eval-ratio", type=float, default=0.05)
parser.add_argument("--max-samples", type=int, default=None)
parser.add_argument("--seed", type=int, default=42)
args = parser.parse_args()

if args.input:
    ds = load_dataset("json", data_files=args.input, split="train")
else:
    ds = load_dataset(args.hf_dataset, split=args.split)

if args.max_samples:
    ds = ds.shuffle(seed=args.seed).select(range(min(args.max_samples, len(ds))))


def to_prompt_completion(row):
    system = [{"role": "system", "content": args.system_prompt}] if args.system_prompt else []

    if "messages" in row and row["messages"]:
        msgs = row["messages"]
        last = max(i for i, m in enumerate(msgs) if m["role"] == "assistant")
        prompt = msgs[:last]
        if args.system_prompt and prompt[0]["role"] != "system":
            prompt = system + prompt
        return {"prompt": prompt, "completion": [msgs[last]]}

    user = row["instruction"].strip()
    if row.get("input"):
        user += "\n\n" + row["input"].strip()
    return {
        "prompt": system + [{"role": "user", "content": user}],
        "completion": [{"role": "assistant", "content": row["output"].strip()}],
    }


ds = ds.map(to_prompt_completion, remove_columns=ds.column_names)
ds = ds.filter(lambda r: r["completion"][0]["content"] != "")
split = ds.train_test_split(test_size=args.eval_ratio, seed=args.seed)

os.makedirs(args.out_dir, exist_ok=True)
for name, part in (("train", split["train"]), ("eval", split["test"])):
    path = os.path.join(args.out_dir, f"{name}.jsonl")
    with open(path, "w") as f:
        for r in part:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"{name}: {len(part)} rows -> {path}")
