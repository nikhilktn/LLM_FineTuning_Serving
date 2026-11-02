"""Print a markdown comparison of every evaluate.py / benchmark.py run in results/."""
import glob
import json
import sys

results_dir = sys.argv[1] if len(sys.argv) > 1 else "results"
evals = {r["label"]: r for r in (json.load(open(p)) for p in sorted(glob.glob(f"{results_dir}/eval_*.json")))}
benches = {r["label"]: r for r in (json.load(open(p)) for p in sorted(glob.glob(f"{results_dir}/bench_*.json")))}
extracts = {r["summary"]["label"]: r["summary"]
            for r in (json.load(open(p)) for p in sorted(glob.glob(f"{results_dir}/extract_*.json")))}

levels = sorted({lv["concurrency"] for b in benches.values() for lv in b["levels"]})
header = ["label", "size (GB)", "perplexity", "field acc", "exact match", "hallucination", "out tokens"] + [f"tok/s @{c}" for c in levels] + ["TTFT p50 @1 (ms)", "TPOT p50 @1 (ms)"]
print("| " + " | ".join(header) + " |")
print("|" + "---|" * len(header))
pct = lambda x: f"{x:.1%}" if isinstance(x, float) else ""
for label in sorted(set(evals) | set(benches) | set(extracts)):
    e, b, x = evals.get(label, {}), benches.get(label), extracts.get(label, {})
    by_c = {lv["concurrency"]: lv for lv in b["levels"]} if b else {}
    row = [label, str(e.get("size_gb", "")), str(e.get("perplexity", "")), pct(x.get("field_accuracy")),
           pct(x.get("exact_match")), pct(x.get("hallucination_rate")), str(x.get("avg_completion_tokens", ""))]
    row += [str(by_c[c]["output_tok_per_s"]) if c in by_c else "" for c in levels]
    row += [str(by_c[1]["ttft_p50_ms"]) if 1 in by_c else "", str(by_c[1]["tpot_p50_ms"]) if 1 in by_c else ""]
    print("| " + " | ".join(row) + " |")
