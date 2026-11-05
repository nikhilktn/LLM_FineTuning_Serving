"""Step 13: publish an adapter or a (merged / quantized) model to the Hugging Face Hub, with a model
card built from this repo's results.

  hf auth login                      # paste a token with WRITE access
  python scripts/publish_hub.py outputs/qwen7b-lora/final       --label lora-r16 --repo <you>/qwen2.5-7b-10k-extract-lora
  python scripts/publish_hub.py models/merged-qwen7b-w8a8       --label w8a8     --repo <you>/qwen2.5-7b-10k-extract-w8a8
  python scripts/publish_hub.py ... --dry-run                    # only write the card to results/cards/

Repos are created private; pass --public to make them public. Re-running updates the repo
(the Hub keeps every version as a git commit). Run in any venv with huggingface_hub (train/serve).
"""
import argparse
import json
import os

BASE = "Qwen/Qwen2.5-7B-Instruct"
SCHEMES = {
    "fp8": "FP8 (E4M3) weights per channel + dynamic per-token FP8 activations, data-free (llm-compressor)",
    "w8a8": "INT8 weights per channel (SmoothQuant 0.8 + GPTQ, 256 calibration examples) + dynamic per-token INT8 activations (llm-compressor)",
    "w4a16": "INT4 weights, group size 128 (GPTQ, 256 calibration examples), bf16 activations (llm-compressor)",
}


def load(path):
    return json.load(open(path)) if os.path.exists(path) else None


def results(label, results_dir):
    ev, ex, bench = (load(f"{results_dir}/{kind}_{label}.json") for kind in ("eval", "extract", "bench"))
    return (ev or {}).get("perplexity"), (ex or {}).get("summary"), bench


def prompt_headers(eval_file, meta_file):
    """The instruction part of one training prompt per statement type (the table itself omitted)."""
    out = {}
    if not (os.path.exists(eval_file) and os.path.exists(meta_file)):
        return None, out
    for row, meta in zip(map(json.loads, open(eval_file)), map(json.loads, open(meta_file))):
        st = meta["statement"]
        if st not in out:
            out[st] = row["prompt"][-1]["content"].split("<statement>")[0].rstrip()
        system = row["prompt"][0]["content"]
    return system, out


def card(args, kind):
    ppl, ex, bench = results(args.label, args.results_dir)
    b_ppl, b_ex, _ = results("base", args.results_dir)
    train = load(f"{args.results_dir}/train_lora-r16.json") or {}
    lora, tsum = train.get("lora", {}), train.get("summary", {})
    system, headers = prompt_headers(args.eval_file, args.meta_file)
    scheme = SCHEMES.get(args.label)
    tags = ["finance", "sec-10k", "information-extraction", "json", "qlora", "vllm"]
    if scheme:
        tags += ["compressed-tensors", args.label]

    meta = [
        "---",
        "license: apache-2.0",
        "language: [en]",
        f"base_model: {BASE}",
        f"base_model_relation: {'adapter' if kind == 'adapter' else 'finetune'}",
        f"library_name: {'peft' if kind == 'adapter' else 'transformers'}",
        "pipeline_tag: text-generation",
        f"tags: [{', '.join(tags)}]",
        "---",
    ]
    what = {
        "adapter": f"A LoRA adapter for [{BASE}](https://huggingface.co/{BASE}).",
        "model": f"[{BASE}](https://huggingface.co/{BASE}) with the LoRA adapter merged in (bf16).",
    }[kind]
    if scheme:
        what = f"[{BASE}](https://huggingface.co/{BASE}) with the LoRA adapter merged in, then quantized: {scheme}."

    lines = meta + [
        f"# {args.repo.split('/')[-1]}",
        "",
        what,
        "",
        "It extracts key line items from the income statement, balance sheet or cash-flow statement "
        "of a US 10-K filing and returns one JSON object: values as printed in the table, `null` for "
        "lines the table doesn't show (it doesn't calculate missing lines), and consistent sign conventions.",
        "",
        "## Training",
        "",
        f"* QLoRA (NF4 base, bf16 adapter): r={lora.get('r', 16)}, alpha={lora.get('lora_alpha', 32)}, "
        f"dropout={lora.get('lora_dropout', 0.05)}, all linear layers "
        f"({tsum.get('trainable_params', 40370176):,} trainable parameters, 0.53%).",
        "* 2 epochs, cosine schedule, loss on the answer tokens only.",
        "* Data: 2,585 statements from 441 S&P 500 companies' 10-K filings, labels from the filings' "
        "inline XBRL tags. Evaluation: 287 statements from 49 companies not seen in training.",
        f"* One RTX 4000 SFF Ada (20 GB): {tsum.get('train_minutes', 208.6):.0f} minutes, "
        f"{tsum.get('peak_gpu_mem_gb', 8.11)} GB peak memory.",
        "",
        "## Results (287 held-out statements, greedy decoding)",
        "",
        "| | base model | this model |",
        "|---|---|---|",
    ]
    pct = lambda s, k: f"{s[k]:.1%}" if s and k in s else "n/a"
    lines += [
        f"| Field accuracy | {pct(b_ex, 'field_accuracy')} | **{pct(ex, 'field_accuracy')}** |",
        f"| Exact match (every field of a statement right) | {pct(b_ex, 'exact_match')} | **{pct(ex, 'exact_match')}** |",
        f"| Hallucination rate (a number where the answer is null) | {pct(b_ex, 'hallucination_rate')} | **{pct(ex, 'hallucination_rate')}** |",
        f"| Perplexity on reference answers | {b_ppl or 'n/a'} | **{ppl or 'n/a'}** |",
        "",
    ]
    if bench:
        lv = {x["concurrency"]: x for x in bench["levels"]}
        top = max(lv)
        lines += [
            f"Speed with vLLM on one RTX 4000 SFF Ada (20 GB, 280 GB/s), 128 output tokens: "
            f"{lv[1]['tpot_p50_ms']} ms per token for one user ({lv[1]['output_tok_per_s']} tok/s), "
            f"{lv[top]['output_tok_per_s']} tok/s total at {top} concurrent users.",
            "",
        ]
    lines += [
        "Weakest fields: `cost_of_revenue` (companies with unusual cost lines) and `income_tax` "
        "(occasional sign errors).",
        "",
        "## Usage (vLLM)",
        "",
        "```bash",
        (f"vllm serve {BASE} --enable-lora --lora-modules extract={args.repo} --max-lora-rank {lora.get('r', 16)}"
         if kind == "adapter" else f"vllm serve {args.repo}"),
        "```",
        "",
        "```python",
        "from openai import OpenAI",
        'client = OpenAI(base_url="http://localhost:8000/v1", api_key="EMPTY")',
        "resp = client.chat.completions.create(",
        f'    model="{"extract" if kind == "adapter" else args.repo}", temperature=0, max_tokens=400,',
        "    messages=[{\"role\": \"system\", \"content\": SYSTEM}, {\"role\": \"user\", \"content\": prompt}])",
        "print(resp.choices[0].message.content)   # one JSON object",
        "```",
        "",
    ]
    if headers:
        lines += [
            "## Prompt format",
            "",
            "The model was trained on exactly this format; other phrasings work less well.",
            f"`SYSTEM` = `{system}`",
            "",
            "The user message is the instructions for the statement type, then the table's text as "
            "printed, one row per line with cells separated by ` | ` (`line item | current year | "
            "prior year`), between `<statement>` and `</statement>`:",
            "",
        ]
        for st, h in headers.items():
            lines += [f"<details><summary>{st}</summary>", "", "```", h, "<statement>",
                      "<title and date lines as printed>", "In millions of dollars | 2024 | 2023",
                      "<line item> | $ 12,345 | $ 11,000", "...", "</statement>", "```", "", "</details>", ""]
    lines += [
        "## Limitations",
        "",
        "* Trained only on US GAAP 10-K statements of large companies; other formats are untested.",
        "* Reads numbers, doesn't check them: verify before using in any financial decision.",
        "* General chat ability is unchanged from the base model, including its weaknesses.",
        "",
    ]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model_dir", help="adapter dir (has adapter_config.json) or full model dir")
    parser.add_argument("--label", required=True, help="results label: lora-r16, merged, fp8, w8a8, w4a16")
    parser.add_argument("--repo", required=True, help="<user or org>/<name>")
    parser.add_argument("--public", action="store_true", help="default: private")
    parser.add_argument("--dry-run", action="store_true", help="write the card to results/cards/ and stop")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--eval-file", default="data/eval.jsonl")
    parser.add_argument("--meta-file", default="data/meta_eval.jsonl")
    args = parser.parse_args()

    kind = "adapter" if os.path.exists(f"{args.model_dir}/adapter_config.json") else "model"
    text = card(args, kind)
    os.makedirs(f"{args.results_dir}/cards", exist_ok=True)
    card_path = f"{args.results_dir}/cards/{args.repo.split('/')[-1]}.md"
    open(card_path, "w").write(text)
    print(f"model card -> {card_path}")
    if args.dry_run:
        return

    from huggingface_hub import HfApi
    api = HfApi()
    try:
        url = api.create_repo(args.repo, private=not args.public, exist_ok=True)
    except Exception as e:  # most often a read-only token
        raise SystemExit(f"create_repo failed ({e}). Is your token allowed to write? "
                         "Create one at https://huggingface.co/settings/tokens and run `hf auth login`.")
    print(f"repo: {url} ({'public' if args.public else 'private'})")
    api.upload_folder(
        folder_path=args.model_dir, repo_id=args.repo,
        # training_args.bin is a pickle nobody needs; README.md is replaced by our card
        ignore_patterns=["training_args.bin", "README.md", "*.pt", "optimizer*", "rng_state*"],
        commit_message=f"Upload {args.label}",
    )
    api.upload_file(path_or_fileobj=card_path, path_in_repo="README.md", repo_id=args.repo,
                    commit_message="Model card")
    print(f"done: https://huggingface.co/{args.repo}")


if __name__ == "__main__":
    main()
