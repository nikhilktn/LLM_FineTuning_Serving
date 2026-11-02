"""Steps 5, 8, 11: load-test a running vLLM server at several concurrency levels.

Reports per level:
  TTFT  time to first token  (prefill + queueing; what a chat user feels as "lag")
  TPOT  time per output token after the first (decode speed per user)
  throughput  total output tokens/s across all users (what the GPU delivers)
Results go to results/bench_<label>.json.
"""
import argparse
import asyncio
import json
import os
import time

from openai import AsyncOpenAI

parser = argparse.ArgumentParser()
parser.add_argument("--base-url", default="http://localhost:8000/v1")
parser.add_argument("--model", required=True, help="served model name or adapter name")
parser.add_argument("--label", required=True)
parser.add_argument("--concurrency", default="1,4,16,32")
parser.add_argument("--requests-per-level", type=int, default=64)
parser.add_argument("--max-tokens", type=int, default=256)
parser.add_argument("--eval-file", default="data/eval.jsonl", help="prompts are taken from here")
parser.add_argument("--out-dir", default="results")
args = parser.parse_args()

client = AsyncOpenAI(base_url=args.base_url, api_key=os.environ.get("VLLM_API_KEY") or open(os.path.expanduser("~/.vllm_api_key")).read().strip())
prompts = [json.loads(l)["prompt"] for l in open(args.eval_file)]


def pct(values, p):
    values = sorted(values)
    return values[min(len(values) - 1, int(p / 100 * len(values)))]


async def one_request(messages):
    t0 = time.perf_counter()
    ttft, out_tokens = None, 0
    stream = await client.chat.completions.create(
        model=args.model,
        messages=messages,
        max_tokens=args.max_tokens,
        temperature=0,
        stream=True,
        stream_options={"include_usage": True},
        extra_body={"ignore_eos": True},  # fixed output length makes runs comparable
    )
    async for chunk in stream:
        if ttft is None and chunk.choices and chunk.choices[0].delta.content:
            ttft = time.perf_counter() - t0
        if chunk.usage:
            out_tokens = chunk.usage.completion_tokens
    e2e = time.perf_counter() - t0
    return ttft, e2e, out_tokens


async def run_level(concurrency, offset):
    sem = asyncio.Semaphore(concurrency)

    async def task(i):
        async with sem:
            # Fresh prompts per level: reused ones would hit vLLM's prefix cache and fake a low TTFT.
            return await one_request(prompts[(offset + i) % len(prompts)])

    t0 = time.perf_counter()
    results = await asyncio.gather(*(task(i) for i in range(args.requests_per_level)))
    wall = time.perf_counter() - t0
    ttfts = [r[0] for r in results]
    tpots = [(e2e - ttft) / max(n - 1, 1) for ttft, e2e, n in results]
    total_tokens = sum(r[2] for r in results)
    return {
        "concurrency": concurrency,
        "ttft_p50_ms": round(pct(ttfts, 50) * 1000, 1),
        "ttft_p95_ms": round(pct(ttfts, 95) * 1000, 1),
        "tpot_p50_ms": round(pct(tpots, 50) * 1000, 2),
        "output_tok_per_s": round(total_tokens / wall, 1),
        "requests_per_s": round(len(results) / wall, 2),
    }


async def main():
    await one_request(prompts[-1])  # warm-up
    levels = []
    print(f"{'conc':>5} {'TTFT p50':>10} {'TTFT p95':>10} {'TPOT p50':>10} {'tok/s':>8} {'req/s':>7}")
    for c in [int(x) for x in args.concurrency.split(",")]:
        r = await run_level(c, offset=len(levels) * args.requests_per_level)
        levels.append(r)
        print(f"{c:>5} {r['ttft_p50_ms']:>8}ms {r['ttft_p95_ms']:>8}ms {r['tpot_p50_ms']:>8}ms "
              f"{r['output_tok_per_s']:>8} {r['requests_per_s']:>7}")
    os.makedirs(args.out_dir, exist_ok=True)
    path = os.path.join(args.out_dir, f"bench_{args.label}.json")
    with open(path, "w") as f:
        json.dump({"label": args.label, "model": args.model, "max_tokens": args.max_tokens, "levels": levels}, f, indent=2)
    print(f"-> {path}")


asyncio.run(main())
