"""★ Experiment (more data): compare two dataset builds before training on the new one.

  python scripts/data_diff.py data data/v2

Checks that the eval set is the same examples (so new results stay comparable with every earlier
run), then reports what the new training set adds: examples per statement, companies, and the
label patterns behind the model's weak spots (tax benefits, net losses, nulls).
"""
import argparse
import collections
import json


def load(d, split):
    rows = [json.loads(l) for l in open(f"{d}/{split}.jsonl")]
    meta = [json.loads(l) for l in open(f"{d}/meta_{split}.jsonl")]
    return rows, meta


def key(row):
    return row["prompt"][-1]["content"] + "\x00" + row["completion"][0]["content"]


def profile(rows, meta):
    c = collections.Counter()
    for r, m in zip(rows, meta):
        gold = json.loads(r["completion"][0]["content"])
        st = m["statement"]
        c[f"examples: {st}"] += 1
        if st == "income_statement":
            tax = gold.get("income_tax")
            c["income_tax < 0 (tax benefit)"] += tax is not None and tax < 0
            c["income_tax > 0 (tax expense)"] += tax is not None and tax > 0
            c["income_tax null"] += tax is None
            ni = gold.get("net_income")
            c["net_income < 0 (net loss)"] += ni is not None and ni < 0
            c["eps_basic null"] += gold.get("eps_basic") is None
            c["cost_of_revenue null"] += gold.get("cost_of_revenue") is None
    c["companies"] = len({m["ticker"] for m in meta})
    c["filings"] = len({m["file"] for m in meta})
    return c


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("old")
    parser.add_argument("new")
    args = parser.parse_args()

    (eo, _), (en, _) = load(args.old, "eval"), load(args.new, "eval")
    so, sn = {key(r) for r in eo}, {key(r) for r in en}
    same = so == sn
    print(f"eval: {len(eo)} vs {len(en)} examples, identical set: {same}"
          + ("" if same else f" ({len(so - sn)} only in old, {len(sn - so)} only in new)"))

    po, pn = profile(*load(args.old, "train")), profile(*load(args.new, "train"))
    print(f"\n{'train':34} {'old':>7} {'new':>7} {'added':>7}")
    for k in ["examples: income_statement", "examples: balance_sheet", "examples: cash_flow",
              "companies", "filings", "income_tax < 0 (tax benefit)", "income_tax > 0 (tax expense)",
              "income_tax null", "net_income < 0 (net loss)", "eps_basic null", "cost_of_revenue null"]:
        print(f"  {k:32} {po[k]:7} {pn[k]:7} {pn[k] - po[k]:+7}")
    to, tn = sum(v for k, v in po.items() if k.startswith("examples")), sum(v for k, v in pn.items() if k.startswith("examples"))
    print(f"  {'total examples':32} {to:7} {tn:7} {tn - to:+7}")
    overlap = {key(r) for r in load(args.new, "train")[0]} & sn
    print(f"\nnew train examples that are also in eval: {len(overlap)} (must be 0)")


if __name__ == "__main__":
    main()
