"""★ Experiments: compare two settings with repeated eval_extraction.py runs, against the noise.

  python scripts/compare_runs.py --a results/extract_v1.json results/extract_v1-r2.json \\
                                 --b results/extract_v2.json results/extract_v2-r2.json

Two runs of the same setting differ by a few answers (batching noise, Step 13), so a single A-vs-B
comparison can't tell an effect from noise. With >= 2 runs per setting this prints:
  * within-setting agreement (the noise) and across-setting agreement (noise + effect),
  * fully-right tables per setting, summed over runs, with a sign test on the pooled pairs,
  * the tables that follow the setting: stable within each setting, different across. Those are
    the real effects; everything else is noise.
"""
import argparse
import itertools
import json
from math import comb


def load(path):
    run = json.load(open(path))
    key = lambda r: (r["ticker"], r["statement"], json.dumps(r["gold"], sort_keys=True))
    return run["summary"], {key(r): r for r in run["results"]}


def ok(r):
    return r["parsed"] is not None and all(o == "correct" for o in r["fields"].values())


def answer(r):
    return json.dumps(r["parsed"], sort_keys=True)


def sign_test(x, y):
    n = x + y
    return min(1.0, 2 * sum(comb(n, i) for i in range(min(x, y) + 1)) / 2 ** n) if n else 1.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--a", nargs="+", required=True, help="runs of setting A")
    parser.add_argument("--b", nargs="+", required=True, help="runs of setting B")
    args = parser.parse_args()
    A, B = [load(p) for p in args.a], [load(p) for p in args.b]
    name = lambda s: s[0]["label"]
    keys = sorted(set.intersection(*(set(r[1]) for r in A + B)))
    print(f"{len(keys)} tables in all runs; A = {[name(r) for r in A]}, B = {[name(r) for r in B]}\n")

    print("summary per run:          field acc  exact  halluc.  sign errors")
    for s, _ in A + B:
        print(f"  {s['label']:24} {s['field_accuracy']:8.1%} {s['exact_match']:6.1%} {s['hallucination_rate']:7.1%}"
              f"  {s['outcomes'].get('sign_error', 0):5}")

    print("\nagreement (identical answers):")
    for (x, y) in itertools.combinations(A + B, 2):
        kind = "within" if (x in A) == (y in A) else "ACROSS"
        same = sum(answer(x[1][k]) == answer(y[1][k]) for k in keys)
        print(f"  {kind:6} {name(x):20} vs {name(y):20} {same:4}/{len(keys)}")

    # Pool every A-run x B-run pair: tables fully right in one and not the other.
    only_a = sum(ok(a[1][k]) and not ok(b[1][k]) for a in A for b in B for k in keys)
    only_b = sum(ok(b[1][k]) and not ok(a[1][k]) for a in A for b in B for k in keys)
    ra, rb = (sum(ok(r[1][k]) for r in runs for k in keys) / len(runs) for runs in (A, B))
    print(f"\nfully right tables, mean per run: A {ra:.1f}, B {rb:.1f}")
    pairs = len(A) * len(B)
    print(f"over the {pairs} A-B pairs: right only in A {only_a}, only in B {only_b} "
          f"(per pair {only_a / pairs:.1f} vs {only_b / pairs:.1f}); sign test on the per-pair "
          f"average p = {sign_test(round(only_a / pairs), round(only_b / pairs)):.2f}")

    stable = [k for k in keys
              if len({answer(r[1][k]) for r in A}) == 1 and len({answer(r[1][k]) for r in B}) == 1
              and answer(A[0][1][k]) != answer(B[0][1][k])]
    print(f"\ntables that follow the setting (same in every A run, same in every B run, A != B): {len(stable)}")
    better = worse = 0
    for k in stable:
        a, b = A[0][1][k], B[0][1][k]
        better += ok(b) and not ok(a)
        worse += ok(a) and not ok(b)
        gold = a["gold"]
        diff = {f: (gold[f], (a["parsed"] or {}).get(f), (b["parsed"] or {}).get(f)) for f in gold
                if (a["parsed"] or {}).get(f) != (b["parsed"] or {}).get(f)}
        tag = "B fixes" if ok(b) and not ok(a) else "B breaks" if ok(a) and not ok(b) else "both wrong"
        print(f"  {k[0]:6} {k[1]:17} {tag:10} field: (gold, A, B) {diff}")
    print(f"systematic effect of B: fixes {better}, breaks {worse} (sign test p = {sign_test(better, worse):.2f})")


if __name__ == "__main__":
    main()
