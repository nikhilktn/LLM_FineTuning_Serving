"""Step 3: build a financial-statement extraction dataset from inline-XBRL 10-K filings.

For each filing it finds the income statement, balance sheet and cash flow statement tables,
renders each as plain text (the prompt), and builds the answer JSON from the XBRL tags inside
that same table (the label), so labels are exact and need no manual work.

Companies are split into train / eval by sector, so eval measures generalisation to unseen
companies. Writes data/train.jsonl, data/eval.jsonl and matching data/meta_{train,eval}.jsonl.
"""
import argparse
import collections
import csv
import json
import os
import random
import re
from datetime import date
from concurrent.futures import ProcessPoolExecutor

from lxml import html as lhtml

# Field -> XBRL concepts in priority order. Only concepts whose sign reads naturally
# (losses and outflows negative) are used, so the model can copy what the table shows.
STATEMENTS = {
    "income_statement": {
        "revenue": ["us-gaap:Revenues", "us-gaap:RevenueFromContractWithCustomerExcludingAssessedTax",
                    "us-gaap:RevenueFromContractWithCustomerIncludingAssessedTax", "us-gaap:SalesRevenueNet",
                    "us-gaap:RevenuesNetOfInterestExpense",  # banks: "Total net revenue"
                    "us-gaap:RegulatedAndUnregulatedOperatingRevenue"],  # utilities
        "cost_of_revenue": ["us-gaap:CostOfRevenue", "us-gaap:CostOfGoodsAndServicesSold"],
        "gross_profit": ["us-gaap:GrossProfit"],
        "operating_income": ["us-gaap:OperatingIncomeLoss"],
        "income_tax": ["us-gaap:IncomeTaxExpenseBenefit"],
        "net_income": ["us-gaap:NetIncomeLoss"],
        "eps_basic": ["us-gaap:EarningsPerShareBasic"],
        "eps_diluted": ["us-gaap:EarningsPerShareDiluted"],
    },
    "balance_sheet": {
        "cash_and_equivalents": ["us-gaap:CashAndCashEquivalentsAtCarryingValue"],
        "total_current_assets": ["us-gaap:AssetsCurrent"],
        "total_assets": ["us-gaap:Assets"],
        "total_current_liabilities": ["us-gaap:LiabilitiesCurrent"],
        "total_liabilities": ["us-gaap:Liabilities"],
        "total_equity": ["us-gaap:StockholdersEquity"],
        "total_liabilities_and_equity": ["us-gaap:LiabilitiesAndStockholdersEquity"],
    },
    "cash_flow": {
        "net_income": ["us-gaap:NetIncomeLoss", "us-gaap:ProfitLoss"],
        "operating_cash_flow": ["us-gaap:NetCashProvidedByUsedInOperatingActivities",
                                "us-gaap:NetCashProvidedByUsedInOperatingActivitiesContinuingOperations"],
        "investing_cash_flow": ["us-gaap:NetCashProvidedByUsedInInvestingActivities",
                                "us-gaap:NetCashProvidedByUsedInInvestingActivitiesContinuingOperations"],
        "financing_cash_flow": ["us-gaap:NetCashProvidedByUsedInFinancingActivities",
                                "us-gaap:NetCashProvidedByUsedInFinancingActivitiesContinuingOperations"],
    },
}
TITLES = {"income_statement": "income statement", "balance_sheet": "balance sheet", "cash_flow": "cash flow statement"}
SCALE_WORDS = {0: "units", 3: "thousands", 6: "millions", 9: "billions"}
SYSTEM_PROMPT = "You extract data from financial statements. Reply with a single JSON object and nothing else."


def norm(text):
    return re.sub(r"\s+", " ", text.replace("\xa0", " ")).strip()


def fact_value(el):
    """The number as displayed, with the XBRL sign applied (e.g. a net loss becomes negative)."""
    text = norm(el.text_content())
    if "fixed-zero" in el.get("format", "") or text in {"", "—", "–", "-"}:
        value = 0.0
    else:
        value = float(re.sub(r"[^0-9.]", "", text) or 0)
    if el.get("sign") == "-":
        value = -value
    return value if "." in text else int(value)  # keep "7,111.0" as 7111.0, like the table


def parse_contexts(doc):
    contexts = {}
    for ctx in doc.iter("xbrli:context"):
        get = lambda tag: (ctx.find(f".//{tag}").text.strip() if ctx.find(f".//{tag}") is not None else None)
        contexts[ctx.get("id")] = {
            "start": get("xbrli:startdate"), "end": get("xbrli:enddate"), "instant": get("xbrli:instant"),
            "dimensional": ctx.find(".//xbrli:segment") is not None,
        }
    return contexts


def table_text(table):
    rows = []
    for tr in table.iter("tr"):
        cells = []
        for td in tr:
            t = norm(td.text_content())
            if not t:
                continue
            # Filings put "$", "(" and ")" in their own cells; glue them back onto the number.
            if cells and cells[-1] in {"$", "("}:
                cells[-1] = f"{cells[-1]} {t}" if cells[-1] == "$" else f"({t}"
            elif cells and t in {")", "%", ")%"}:
                cells[-1] += t
            else:
                cells.append(t)
        if cells:
            rows.append(" | ".join(cells))
    return "\n".join(rows)


def heading(table, limit=300):
    """Text just before the table, e.g. 'CONSOLIDATED STATEMENTS OF OPERATIONS (In millions...)'."""
    parts, node, steps = [], table, 0
    while node is not None and sum(map(len, parts)) < limit and steps < 40:
        prev = node.getprevious()
        if prev is None:
            node = node.getparent()
            continue
        node, steps = prev, steps + 1
        if not isinstance(node.tag, str):  # comments and processing instructions
            continue
        if node.tag == "table" or node.find(".//table") is not None:
            break
        t = norm(node.text_content())
        if t:
            parts.append(t)
    return " ".join(reversed(parts))[-limit:]


def process_filing(path):
    """Never raises, so one unusual filing is counted as skipped instead of stopping the run."""
    try:
        return extract_statements(path)
    except Exception as e:
        return [], f"error: {type(e).__name__}: {e}"


def extract_statements(path):
    try:
        doc = lhtml.parse(path).getroot()
    except Exception as e:
        return [], f"parse error: {e}"
    contexts = parse_contexts(doc)
    doc_type = next((el for el in doc.iter("ix:nonnumeric") if el.get("name") == "dei:DocumentType"), None)
    if doc_type is None or doc_type.get("contextref") not in contexts:
        return [], "no inline XBRL / DocumentType"
    period = contexts[doc_type.get("contextref")]
    period_end = period["end"] or period["instant"]

    def is_current(ctx_id, statement):
        c = contexts.get(ctx_id)
        if not c or c["dimensional"]:
            return False
        if statement == "balance_sheet":
            return c["instant"] == period_end
        if c["end"] != period_end or c["start"] is None:
            return False
        days = (date.fromisoformat(c["end"]) - date.fromisoformat(c["start"])).days
        return 350 <= days <= 380  # the full fiscal year, not a quarter

    examples = []
    for statement, fields in STATEMENTS.items():
        concept_to_field = {c: f for f, cs in fields.items() for c in cs}
        best = None
        for table in doc.iter("table"):
            facts = [el for el in table.iter("ix:nonfraction") if el.get("name") in concept_to_field]
            current = [el for el in facts if is_current(el.get("contextref"), statement)]
            score = len({concept_to_field[el.get("name")] for el in current})
            if score >= 3 and (best is None or score > best[0]):
                best = (score, table, current)
        if best is None:
            continue
        _, table, current = best

        target = {"fiscal_period_end": period_end}
        scales = collections.Counter(el.get("scale", "0") for el in current
                                     if (el.get("unitref") or "").lower() == "usd")
        scale = int(scales.most_common(1)[0][0]) if scales else 0
        target["amounts_in"] = SCALE_WORDS.get(scale, f"10^{scale}")
        for field, concepts in fields.items():
            target[field] = None
            for concept in concepts:
                el = next((e for e in current if e.get("name") == concept), None)
                if el is not None:
                    target[field] = fact_value(el)
                    break

        body = table_text(table)
        head = heading(table)
        user = (
            f"Extract these fields from the {TITLES[statement]} below for the most recent fiscal year: "
            f"{', '.join(target)}.\n"
            "Report numbers in the table's units and state the units in amounts_in. Use null for any field "
            "that is not shown in the table; do not calculate it from other lines.\n"
            "Conventions: revenue, costs and income tax expense are positive (a tax benefit is negative); "
            "net losses and cash outflows are negative. net_income is the amount attributable to the "
            "company's shareholders, excluding non-controlling interests, or the only net income shown.\n\n"
            f"<statement>\n{head}\n{body}\n</statement>"
        )
        examples.append({
            "prompt": [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}],
            "completion": [{"role": "assistant", "content": json.dumps(target)}],
            "meta": {"file": path, "statement": statement, "period_end": period_end,
                     "fields_found": sum(v is not None for k, v in target.items() if k in fields)},
        })
    return examples, None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default="data/raw/10k", help="<TICKER>/<date>_10-K_<acc>.htm")
    parser.add_argument("--sectors", default="data/raw/10k/sectors.csv")
    parser.add_argument("--out-dir", default="data")
    parser.add_argument("--eval-frac", type=float, default=0.1, help="share of companies held out, per sector")
    parser.add_argument("--max-prompt-chars", type=int, default=6000, help="~1,500 tokens")
    parser.add_argument("--workers", type=int, default=os.cpu_count())
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    sectors = {r["ticker"]: r["sector"] for r in csv.DictReader(open(args.sectors))}
    files = sorted(os.path.join(args.input_dir, t, f)
                   for t in os.listdir(args.input_dir) if os.path.isdir(os.path.join(args.input_dir, t))
                   for f in os.listdir(os.path.join(args.input_dir, t)) if f.endswith((".htm", ".html")))
    print(f"{len(files)} filings from {len({os.path.dirname(f) for f in files})} companies")

    # Hold out whole companies, stratified by sector.
    rng = random.Random(args.seed)
    by_sector = collections.defaultdict(list)
    for t in sorted({os.path.basename(os.path.dirname(f)) for f in files}):
        by_sector[sectors.get(t, "Unknown")].append(t)
    eval_tickers = set()
    for tickers in by_sector.values():
        rng.shuffle(tickers)
        eval_tickers.update(tickers[: max(1, round(len(tickers) * args.eval_frac))])

    splits = {"train": [], "eval": []}
    stats = collections.Counter()
    errors = []
    found = collections.Counter()  # (statement, field) -> examples where the label is not null
    with ProcessPoolExecutor(args.workers) as pool:
        for path, (examples, error) in zip(files, pool.map(process_filing, files, chunksize=4)):
            if error:
                stats[f"skipped filing: {error.split(':')[0]}"] += 1
                errors.append(f"{path}: {error}")
                continue
            ticker = os.path.basename(os.path.dirname(path))
            for ex in examples:
                if len(ex["prompt"][1]["content"]) > args.max_prompt_chars:
                    stats["skipped: prompt too long"] += 1
                    continue
                ex["meta"].update(ticker=ticker, sector=sectors.get(ticker, "Unknown"))
                splits["eval" if ticker in eval_tickers else "train"].append(ex)
                stats[f"examples: {ex['meta']['statement']}"] += 1
                for field, value in json.loads(ex["completion"][0]["content"]).items():
                    found[(ex["meta"]["statement"], field)] += value is not None

    os.makedirs(args.out_dir, exist_ok=True)
    for name, rows in splits.items():
        rng.shuffle(rows)
        with open(os.path.join(args.out_dir, f"{name}.jsonl"), "w") as f, \
             open(os.path.join(args.out_dir, f"meta_{name}.jsonl"), "w") as m:
            for r in rows:
                f.write(json.dumps({"prompt": r["prompt"], "completion": r["completion"]}) + "\n")
                m.write(json.dumps(r["meta"]) + "\n")
        print(f"{name}: {len(rows)} examples from {len({r['meta']['ticker'] for r in rows})} companies")
    for k, v in sorted(stats.items()):
        print(f"  {k}: {v}")
    for e in errors[:10]:
        print(f"  ! {e[:200]}")
    print("label coverage (share of examples where the field is found):")
    for statement, fields in STATEMENTS.items():
        n = stats[f"examples: {statement}"] or 1
        print(f"  {statement}: " + ", ".join(f"{f} {found[(statement, f)] / n:.0%}" for f in fields))


if __name__ == "__main__":
    main()
