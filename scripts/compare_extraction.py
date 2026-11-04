"""Steps 8, 9, 11: paired comparison of two eval_extraction.py runs on the same eval set.

  python scripts/compare_extraction.py results/extract_lora-r16.json results/extract_merged.json

Averages hide how much two models really differ. This matches the runs table by table and counts
identical answers and tables only one of them got fully right. If the "only A" / "only B" counts
are small and close to each other, the difference is noise (sign test p-value printed).
"""
import argparse
import json
from math import comb


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("a")
    parser.add_argument("b")
    args = parser.parse_args()

    runs = [json.load(open(p)) for p in (args.a, args.b)]
    key = lambda r: (r["ticker"], r["statement"], json.dumps(r["gold"], sort_keys=True))
    A, B = ({key(r): r for r in run["results"]} for run in runs)
    common = A.keys() & B.keys()
    ok = lambda r: r["parsed"] is not None and all(o == "correct" for o in r["fields"].values())
    only_a = [k for k in common if ok(A[k]) and not ok(B[k])]
    only_b = [k for k in common if ok(B[k]) and not ok(A[k])]
    n = len(only_a) + len(only_b)
    # Two-sided sign test: how likely a split this uneven is if neither model is better.
    p = min(1.0, 2 * sum(comb(n, i) for i in range(min(len(only_a), len(only_b)) + 1)) / 2 ** n) if n else 1.0

    la, lb = (run["summary"]["label"] for run in runs)
    print(f"{len(common)} tables matched")
    print(f"  identical answers: {sum(A[k]['parsed'] == B[k]['parsed'] for k in common)}")
    print(f"  fully right only in {la}: {len(only_a)} | only in {lb}: {len(only_b)} | sign test p = {p:.2f}")
    for name in ("field_accuracy", "exact_match", "hallucination_rate"):
        print(f"  {name}: {runs[0]['summary'][name]:.1%} vs {runs[1]['summary'][name]:.1%}")
    fa, fb = (run["summary"]["per_field_accuracy"] for run in runs)
    for f in fa:
        if fa[f] != fb.get(f):
            print(f"    {f}: {fa[f]:.1%} vs {fb.get(f, 0):.1%}")
    print("  tables where only one is fully right (field: gold / A / B):")
    for k in sorted(only_a + only_b)[:10]:
        pa, pb = A[k]["parsed"] or {}, B[k]["parsed"] or {}
        diffs = [f"{f}: {g} / {pa.get(f)} / {pb.get(f)}" for f, g in A[k]["gold"].items() if pa.get(f) != pb.get(f)]
        print(f"    {k[0]} {k[1]}: " + "; ".join(diffs))


if __name__ == "__main__":
    main()
