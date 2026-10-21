"""Summarize comparative runs; reviewed labels and NLI estimates stay distinct."""
import argparse
import csv
import json
from pathlib import Path
import statistics


def summarize(rows, review=None):
    review = review or {}
    groups = {}
    seen = set()
    for row in rows:
        key = row["id"], row["system"], row["phase"]
        if key in seen:
            raise ValueError("Duplicate question/system/phase in run")
        seen.add(key)
        groups.setdefault((row["system"], row["phase"]), []).append(row)
    output = {}
    for (system, phase), batch in sorted(groups.items()):
        results = [r["result"] for r in batch]
        metrics = [r.get("metrics", {}) for r in results]
        times = [r["latency_seconds"] for r in results if isinstance(r.get("latency_seconds"), (int, float))]
        labels = [review.get((r["id"], system, phase), {}) for r in batch]
        entry = {"questions": len(batch), "errors": sum(bool(r.get("error")) for r in results),
                 "mean_seconds": statistics.mean(times) if times else None,
                 "median_seconds": statistics.median(times) if times else None,
                 "llm_calls": sum(m.get("llm_calls", 0) for m in metrics),
                 "graph_llm_calls": sum(m.get("graph_llm_calls", 0) for m in metrics),
                 "graph_extraction_failures": sum(str(status).startswith("failed:")
                    for result in results for status in result.get("retrieval_details", {}).get("expansions", {}).values()),
                 "web_network_calls": sum(m.get("web_network_calls", 0) for m in metrics),
                 "web_cache_hits": sum(m.get("web_cache_hits", 0) for m in metrics)}
        retrieval_times = [sum(e["seconds"] for e in m.get("retrieval_events", [])) for m in metrics]
        stages = {}
        for metric in metrics:
            for event in metric.get("llm_events", []):
                stages[event["stage"]] = stages.get(event["stage"], 0) + event["seconds"]
        entry["mean_retrieval_seconds"] = statistics.mean(retrieval_times) if retrieval_times else None
        entry["llm_stage_total_seconds"] = stages
        for field in ("answer_correct", "answer_grounded", "context_sufficient"):
            values = [int(label[field]) for label in labels if label.get(field, "") in {"0", "1"}]
            entry[field] = {"positive": sum(values), "reviewed": len(values),
                            "rate": sum(values) / len(values) if values else None}
        both = [int(label["answer_correct"] == label["answer_grounded"] == "1") for label in labels
                if label.get("answer_correct") in {"0", "1"} and label.get("answer_grounded") in {"0", "1"}]
        entry["correct_and_grounded"] = {"positive": sum(both), "reviewed": len(both),
                                         "rate": sum(both) / len(both) if both else None}
        judged_routes = [(r["result"].get("route"), label["gold_route"])
                         for r, label in zip(batch, labels) if label.get("gold_route") in
                         {"LOCAL", "WEB", "REWRITE_AND_WEB", "CLARIFY"}]
        if system in {"s3", "s4", "s5"}:
            entry["reviewed_routing"] = {"correct": sum(a == b for a, b in judged_routes),
                                          "reviewed": len(judged_routes)}
        nli = [r["verification"] for r in results if "verification" in r]
        entry["nli_estimates_not_ground_truth"] = {
            "before_supported": sum(v["before_supported"] for v in nli),
            "before_claims": sum(v["before_claims"] for v in nli),
            "after_supported": sum(v["after_supported"] for v in nli),
            "after_claims": sum(v["after_claims"] for v in nli),
            "removed_claims": sum(v["removed_claims"] for v in nli),
            "repaired_claims": sum(v["repaired_claims"] for v in nli),
            "repair_attempts": sum(v["repair_attempts"] for v in nli),
            "nli_pairs": sum(v["nli_pairs"] for v in nli)}
        manual_claims = [(int(v["claims_supported"]), int(v["claims_total"])) for v in labels
                         if v.get("claims_supported", "").isdigit() and v.get("claims_total", "").isdigit()]
        if any(supported > total for supported, total in manual_claims):
            raise ValueError("Supported claims cannot exceed total claims")
        total = sum(t for s, t in manual_claims)
        entry["reviewed_claim_support"] = {"supported": sum(s for s, t in manual_claims),
                                            "total": total,
                                            "rate": sum(s for s, t in manual_claims) / total if total else None}
        coverage = [r["preselected_context_coverage"] for r in results if r.get("preselected_context_coverage")]
        entry["preselected_context_proxy"] = {"complete": sum(c["complete"] for c in coverage),
                                                "questions": len(coverage)}
        output[f"{system}:{phase}"] = entry
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--review", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.run.read_text(encoding="utf-8").splitlines() if line.strip()]
    review = {}
    if args.review:
        with args.review.open(encoding="utf-8-sig", newline="") as handle:
            review = {(r["id"], r["system"], r["phase"]): r for r in csv.DictReader(handle)}
    result = summarize(rows, review)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
