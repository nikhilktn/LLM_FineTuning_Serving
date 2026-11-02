"""Steps 5, 7, 11: field-level accuracy on the 10-K extraction eval set, via a running vLLM server.

Sends every eval prompt to the server (concurrently, so continuous batching does the work),
parses the JSON answer and scores each field against the XBRL label:
  correct       value matches (numbers exactly, nulls as null)
  hallucinated  label is null but the model gave a value (e.g. calculated gross profit)
  missed        label has a value but the model said null
  sign error    right number, wrong sign
Results go to results/extract_<label>.json, including every prediction for inspection.
"""
import argparse
import asyncio
import collections
import json
import os
import re
import time

from openai import AsyncOpenAI


def parse_answer(text):
    """The first JSON object in the reply, tolerating ```json fences and surrounding prose."""
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    return obj if isinstance(obj, dict) else None


def to_number(value):
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    s = str(value).strip().replace(",", "").replace("$", "")
    if re.fullmatch(r"\(\s*[\d.]+\s*\)", s):
        s = "-" + s.strip("() ")
    try:
        return float(s)
    except ValueError:
        return None


def score_field(field, pred, gold):
    """Returns one of: correct, hallucinated, missed, sign_error, wrong."""
    if gold is None:
        return "correct" if pred is None else "hallucinated"
    if pred is None:
        return "missed"
    if field in ("fiscal_period_end", "amounts_in"):
        return "correct" if str(pred).strip().lower() == str(gold).lower() else "wrong"
    p, g = to_number(pred), float(gold)
    if p is None:
        return "wrong"
    if abs(p - g) <= 1e-6 * max(1.0, abs(g)):
        return "correct"
    return "sign_error" if g != 0 and abs(p + g) <= 1e-6 * abs(g) else "wrong"


def summarize(results):
    outcomes = collections.Counter()
    per_field = collections.defaultdict(collections.Counter)
    per_statement = collections.defaultdict(collections.Counter)
    exact = valid = 0
    for r in results:
        if r["parsed"] is None:
            continue
        valid += 1
        all_ok = True
        for field, outcome in r["fields"].items():
            outcomes[outcome] += 1
            per_field[field][outcome] += 1
            per_statement[r["statement"]][outcome] += 1
            all_ok &= outcome == "correct"
        exact += all_ok
    n, total = len(results), sum(outcomes.values()) or 1
    acc = lambda c: round(c["correct"] / max(1, sum(c.values())), 4)
    gold_null = sum(1 for r in results if r["parsed"] for f, o in r["fields"].items()
                    if r["gold"][f] is None)
    return {
        "examples": n,
        "json_valid": round(valid / n, 4),
        "field_accuracy": round(outcomes["correct"] / total, 4),  # over valid answers
        "exact_match": round(exact / n, 4),  # every field right; invalid JSON counts as wrong
        "hallucination_rate": round(outcomes["hallucinated"] / max(1, gold_null), 4),
        "outcomes": dict(outcomes),
        "per_field_accuracy": {f: acc(c) for f, c in sorted(per_field.items())},
        "per_statement_accuracy": {s: acc(c) for s, c in sorted(per_statement.items())},
    }


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://localhost:8000/v1")
    parser.add_argument("--model", required=True, help="served model name or adapter name")
    parser.add_argument("--label", required=True, help="name for this run, e.g. base, lora-r16, w4a16")
    parser.add_argument("--eval-file", default="data/eval.jsonl")
    parser.add_argument("--meta-file", default="data/meta_eval.jsonl")
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument("--max-tokens", type=int, default=400)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--out-dir", default="results")
    args = parser.parse_args()

    key = os.environ.get("VLLM_API_KEY") or open(os.path.expanduser("~/.vllm_api_key")).read().strip()
    client = AsyncOpenAI(base_url=args.base_url, api_key=key)
    rows = [json.loads(l) for l in open(args.eval_file)][: args.limit]
    metas = [json.loads(l) for l in open(args.meta_file)][: len(rows)]
    sem = asyncio.Semaphore(args.concurrency)

    async def run(row, meta):
        async with sem:
            out = await client.chat.completions.create(
                model=args.model, messages=row["prompt"], max_tokens=args.max_tokens, temperature=0)
        text = out.choices[0].message.content or ""
        gold = json.loads(row["completion"][0]["content"])
        parsed = parse_answer(text)
        fields = {f: score_field(f, parsed.get(f), g) for f, g in gold.items()} if parsed else {}
        return {"ticker": meta["ticker"], "statement": meta["statement"], "gold": gold,
                "raw": text, "parsed": parsed, "fields": fields,
                "completion_tokens": out.usage.completion_tokens}

    start = time.perf_counter()
    results = await asyncio.gather(*(run(r, m) for r, m in zip(rows, metas)))
    elapsed = time.perf_counter() - start

    summary = summarize(results)
    summary.update(label=args.label, model=args.model, seconds=round(elapsed, 1),
                   avg_completion_tokens=round(sum(r["completion_tokens"] for r in results) / len(results), 1))
    os.makedirs(args.out_dir, exist_ok=True)
    path = os.path.join(args.out_dir, f"extract_{args.label}.json")
    with open(path, "w") as f:
        json.dump({"summary": summary, "results": results}, f, indent=1, ensure_ascii=False)

    print(f"[{args.label}] {summary['examples']} examples in {elapsed:.0f}s, "
          f"avg {summary['avg_completion_tokens']} output tokens")
    print(f"  JSON valid {summary['json_valid']:.1%} | field accuracy {summary['field_accuracy']:.1%} | "
          f"exact match {summary['exact_match']:.1%} | hallucination rate {summary['hallucination_rate']:.1%}")
    print(f"  outcomes: {summary['outcomes']}")
    print("  per field: " + ", ".join(f"{f} {a:.0%}" for f, a in summary["per_field_accuracy"].items()))
    print(f"  -> {path}")


if __name__ == "__main__":
    asyncio.run(main())
