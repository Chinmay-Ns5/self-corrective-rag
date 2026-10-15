"""Tune routing thresholds on reviewed calibration rows; report disjoint evaluation.

Consumes compare_rag review CSVs with gold_route filled from retrieved evidence.
Never edits .env or labels routes using question categories automatically.
"""
import argparse
import csv
import json
from pathlib import Path


LABELS = {"LOCAL", "WEB", "REWRITE_AND_WEB"}


def route(score, low, high):
    return "LOCAL" if score >= high else "WEB" if score >= low else "REWRITE_AND_WEB"


def reviewed(path, system):
    rows = []
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("system") != system or row.get("phase", "first") != "first":
                continue
            gold = row.get("gold_route", "").strip()
            if not gold or gold == "CLARIFY":
                continue
            if gold not in LABELS:
                raise ValueError("gold_route must distinguish LOCAL, WEB, REWRITE_AND_WEB")
            score = float(row["score"])
            if not 0 <= score <= 1 or not row.get("id") or not row.get("question", "").strip():
                raise ValueError("Invalid reviewed row")
            rows.append({"id": row["id"], "question": " ".join(row["question"].lower().split()),
                         "score": score, "gold": gold})
    if not rows or len({r["id"] for r in rows}) != len(rows):
        raise ValueError("Need nonempty, unique, reviewed question rows")
    return rows


def measure(rows, low, high):
    confusion = {gold: {pred: 0 for pred in sorted(LABELS)} for gold in sorted(LABELS)}
    for row in rows:
        confusion[row["gold"]][route(row["score"], low, high)] += 1
    correct = sum(confusion[label][label] for label in LABELS)
    return {"correct": correct, "reviewed": len(rows), "accuracy": correct / len(rows),
            "confusion": confusion}


def calibrate(calibration, evaluation):
    if {r["id"] for r in calibration} & {r["id"] for r in evaluation} or \
            {r["question"] for r in calibration} & {r["question"] for r in evaluation}:
        raise ValueError("Calibration/evaluation questions overlap")
    if {r["gold"] for r in calibration} != LABELS:
        raise ValueError("Calibration must contain all three graded routing classes")
    scores = sorted({0.0, 1.0, *[r["score"] for r in calibration]})
    boundaries = sorted({0.0, 1.0, 0.40, 0.75, *scores,
                         *[(a + b) / 2 for a, b in zip(scores, scores[1:])]})
    options = [(low, high) for low in boundaries for high in boundaries if low < high]
    low, high = min(options, key=lambda pair: (
        -measure(calibration, *pair)["correct"], abs(pair[0] - 0.40) + abs(pair[1] - 0.75)))
    return {"recommended": {"low_threshold": low, "high_threshold": high},
            "selection": "maximize calibration accuracy; ties prefer defaults; evaluation never used for selection",
            "limitations": "Offline projection on saved final scores. Rerun to validate recommendations: evidence guards can cap scores using configured thresholds.",
            "calibration_ids": [r["id"] for r in calibration],
            "evaluation_ids": [r["id"] for r in evaluation],
            "calibration_before": measure(calibration, 0.40, 0.75),
            "calibration_after": measure(calibration, low, high),
            "evaluation_before": measure(evaluation, 0.40, 0.75),
            "evaluation_after": measure(evaluation, low, high)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--system", choices=["s3", "s4", "s5"], default="s3")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output already exists")
    result = calibrate(reviewed(args.calibration, args.system), reviewed(args.evaluation, args.system))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
