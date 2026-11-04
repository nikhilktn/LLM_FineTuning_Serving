"""Re-score saved eval_extraction.py predictions against rebuilt labels, without the GPU.

When build_10k_dataset.py's labels change but the prompts don't, the model's answers are still
valid; only the scoring is out of date. Keep a copy of the eval file the predictions were made on:

  cp data/eval.jsonl data/eval_v1.jsonl      # before rebuilding
  python scripts/build_10k_dataset.py
  python scripts/rescore_extraction.py --old-eval-file data/eval_v1.jsonl results/extract_base.json results/extract_lora-r16.json

Predictions are matched to the new examples by prompt text (rebuilding reshuffles them), and each
results file is rewritten in place. Examples whose prompt changed are dropped and counted.
"""
import argparse
import json

from eval_extraction import score_field, summarize


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("results", nargs="+", help="results/extract_<label>.json files")
    parser.add_argument("--old-eval-file", required=True, help="the eval.jsonl the predictions were made on")
    parser.add_argument("--eval-file", default="data/eval.jsonl")
    parser.add_argument("--meta-file", default="data/meta_eval.jsonl")
    args = parser.parse_args()

    key = lambda row: row["prompt"][-1]["content"]
    old_prompts = [key(json.loads(l)) for l in open(args.old_eval_file)]
    new = {key(r): (r, json.loads(m)) for r, m in zip(map(json.loads, open(args.eval_file)), open(args.meta_file))}

    for path in args.results:
        run = json.load(open(path))
        old_summary, results, dropped = run["summary"], [], 0
        for prompt, r in zip(old_prompts, run["results"]):
            if prompt not in new:
                dropped += 1
                continue
            row, meta = new[prompt]
            gold = json.loads(row["completion"][0]["content"])
            fields = {f: score_field(f, r["parsed"].get(f), g) for f, g in gold.items()} if r["parsed"] else {}
            results.append({**r, "ticker": meta["ticker"], "statement": meta["statement"], "gold": gold, "fields": fields})

        summary = summarize(results)
        summary.update({k: old_summary[k] for k in ("label", "model", "seconds", "avg_completion_tokens")},
                       rescored_from=old_summary["examples"], dropped=dropped,
                       not_scored=len(new) - len(results))  # new examples with no prediction yet
        with open(path, "w") as f:
            json.dump({"summary": summary, "results": results}, f, indent=1, ensure_ascii=False)
        print(f"[{summary['label']}] {len(results)} rescored, {dropped} dropped (prompt changed), "
              f"{summary['not_scored']} new examples not scored")
        for name in ("field_accuracy", "exact_match", "hallucination_rate"):
            print(f"  {name}: {old_summary[name]:.1%} -> {summary[name]:.1%}")
        print(f"  outcomes: {summary['outcomes']}")


if __name__ == "__main__":
    main()
