"""Summarize human-reviewed baseline and CRAG correctness/grounding labels."""
import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path


LABELS = ("baseline_correct", "baseline_grounded", "crag_correct", "crag_grounded")


def score_review(rows):
    counts = defaultdict(lambda: {name: {"correct": 0, "reviewed": 0} for name in LABELS})
    route_counts = defaultdict(lambda: {"correct": 0, "reviewed": 0})
    for row in rows:
        for group in ("overall", row["category"]):
            gold_route = row.get("gold_route", "").strip()
            if gold_route not in ("", "LOCAL", "WEB", "REWRITE_AND_WEB", "CLARIFY"):
                raise ValueError(f"{row['id']}: invalid gold_route")
            if gold_route:
                route_counts[group]["reviewed"] += 1
                route_counts[group]["correct"] += int(row.get("crag_route") == gold_route)
            for name in LABELS:
                value = row[name].strip()
                if value not in ("", "0", "1"):
                    raise ValueError(f"{row['id']}: {name} must be blank, 0, or 1")
                if value:
                    counts[group][name]["reviewed"] += 1
                    counts[group][name]["correct"] += int(value)
    scores = {
        group: {
            name: {**value, "rate": round(value["correct"] / value["reviewed"], 3)
                   if value["reviewed"] else None}
            for name, value in metrics.items()
        }
        for group, metrics in counts.items()
    }
    for group, value in route_counts.items():
        scores[group]["routing_accuracy"] = {
            **value, "rate": round(value["correct"] / value["reviewed"], 3)
            if value["reviewed"] else None,
        }
    return scores


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("review_csv", type=Path)
    args = parser.parse_args()
    with args.review_csv.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        parser.error("Review sheet has no questions")
    summary = score_review(rows)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
