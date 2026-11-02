"""Print a markdown comparison of every evaluate.py / benchmark.py run in results/."""
import glob
import json
import sys

results_dir = sys.argv[1] if len(sys.argv) > 1 else "results"
evals = {r["label"]: r for r in (json.load(open(p)) for p in sorted(glob.glob(f"{results_dir}/eval_*.json")))}
benches = {r["label"]: r for r in (json.load(open(p)) for p in sorted(glob.glob(f"{results_dir}/bench_*.json")))}

levels = sorted({lv["concurrency"] for b in benches.values() for lv in b["levels"]})
header = ["label", "size (GB)", "perplexity"] + [f"tok/s @{c}" for c in levels] + ["TTFT p50 @1 (ms)", "TPOT p50 @1 (ms)"]
print("| " + " | ".join(header) + " |")
print("|" + "---|" * len(header))
for label in sorted(set(evals) | set(benches)):
    e, b = evals.get(label, {}), benches.get(label)
    by_c = {lv["concurrency"]: lv for lv in b["levels"]} if b else {}
    row = [label, str(e.get("size_gb", "")), str(e.get("perplexity", ""))]
    row += [str(by_c[c]["output_tok_per_s"]) if c in by_c else "" for c in levels]
    row += [str(by_c[1]["ttft_p50_ms"]) if 1 in by_c else "", str(by_c[1]["tpot_p50_ms"]) if 1 in by_c else ""]
    print("| " + " | ".join(row) + " |")
