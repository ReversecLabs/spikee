"""Summarize local reports: python -m tests.judge_report report.jsonl [...]."""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


def summarize(rows):
    groups = defaultdict(Counter)
    for row in rows:
        key = (row["phase"], row["model"], row["entry"]["judge_name"], row["split"])
        counts = groups[key]
        counts["total"] += 1
        counts[row["outcome"]] += 1
        counts["positive" if row["expected"] else "negative"] += 1
        for call in row["calls"]:
            counts["calls"] += 1
            usage = call.get("usage") or {}
            counts["tokens"] += usage.get("total_tokens") or 0
            if usage.get("cost") is not None:
                counts["priced_calls"] += 1
                counts["cost"] += usage["cost"]
    return groups


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reports", nargs="+", type=Path)
    args = parser.parse_args()
    rows = [
        json.loads(line)
        for path in args.reports
        for line in path.read_text().splitlines()
        if line.strip()
    ]
    print(
        "| Phase | Model | Judge | Split | N (+/−) | Pass | FP | FN | Invalid | Errors | Calls | Reported cost (USD) |"
    )
    print("|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for (phase, model, judge, split), counts in sorted(summarize(rows).items()):
        cost = f"{counts['cost']:.6f}" if counts["priced_calls"] else "unavailable"
        print(
            f"| {phase} | {model} | {judge} | {split} | {counts['total']} ({counts['positive']}/{counts['negative']}) | {counts['pass']} | {counts['false_positive']} | {counts['false_negative']} | {counts['invalid_verdict']} | {counts['error']} | {counts['calls']} | {cost} |"
        )
    print(
        "\nCosts include only calls for which the provider returned cost metadata. Partial/interrupted runs retain their actual denominators."
    )


if __name__ == "__main__":
    main()
