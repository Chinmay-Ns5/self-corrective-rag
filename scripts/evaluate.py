"""Run the same labeled questions through baseline RAG and CRAG."""
import argparse
from collections import Counter
from contextlib import redirect_stdout
import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import statistics
import sys


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = ROOT / "evaluation" / "questions.jsonl"
VALID_CATEGORIES = {"local", "web_only", "partial_evidence", "unclear_reference", "freshness"}
VALID_ROUTES = {"LOCAL", "WEB_ANY", "CLARIFY"}


def load_questions(path):
    questions = []
    seen = set()
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not all(isinstance(row.get(key), str) for key in
                   ("id", "category", "question", "expected_route", "reference_answer", "reference_source")):
            raise ValueError(f"Invalid fields on dataset line {number}")
        if not row["id"] or row["id"] in seen:
            raise ValueError(f"Duplicate or empty question ID on line {number}")
        if not row["question"].strip() or row["category"] not in VALID_CATEGORIES \
                or row["expected_route"] not in VALID_ROUTES:
            raise ValueError(f"Invalid question/category/route on line {number}")
        seen.add(row["id"])
        questions.append(row)
    if not questions:
        raise ValueError("Evaluation dataset is empty")
    return questions


def route_matches(expected, observed):
    if expected == "WEB_ANY":
        return observed in {"WEB", "REWRITE_AND_WEB"}
    return expected == observed


def summarize(rows):
    completed = [row for row in rows if not row["crag"].get("error")]
    routed = [row for row in rows if row["crag"].get("route")]
    baseline_times = [row["baseline"]["latency"] for row in rows
                      if isinstance(row["baseline"].get("latency"), (int, float))]
    crag_times = [row["crag"]["latency_seconds"] for row in rows
                  if isinstance(row["crag"].get("latency_seconds"), (int, float))]
    return {
        "questions_run": len(rows),
        "categories": dict(Counter(row["category"] for row in rows)),
        "crag_routes": dict(Counter(row["crag"].get("route") or "ERROR" for row in rows)),
        "dataset_route_agreement": sum(route_matches(row["expected_route"], row["crag"].get("route"))
                                       for row in routed),
        "routed_questions": len(routed),
        "crag_no_error": len(completed),
        "baseline_mean_latency_seconds": round(statistics.mean(baseline_times), 3) if baseline_times else None,
        "crag_mean_latency_seconds": round(statistics.mean(crag_times), 3) if crag_times else None,
        "answer_correctness": "Requires human review sheet",
        "groundedness": "Requires human review sheet",
    }


def write_review_sheet(path, rows):
    columns = ["id", "category", "question", "expected_route", "reference_answer", "reference_source",
               "baseline_answer", "baseline_sources", "crag_route", "crag_answer", "crag_sources",
               "retrieved_documents", "crag_score", "crag_reason", "gold_route",
               "baseline_correct", "baseline_grounded",
               "crag_correct", "crag_grounded", "notes"]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            baseline, crag = row["baseline"], row["crag"]
            grade = crag.get("evaluation") or {}
            writer.writerow({
                **{key: row.get(key, "") for key in
                   ("id", "category", "question", "expected_route", "reference_answer", "reference_source")},
                "baseline_answer": baseline.get("answer", ""),
                "baseline_sources": json.dumps(baseline.get("sources", []), ensure_ascii=False),
                "crag_route": crag.get("route", ""),
                "crag_answer": crag.get("answer", ""),
                "crag_sources": json.dumps(crag.get("sources", []), ensure_ascii=False),
                "retrieved_documents": json.dumps(crag.get("retrieved_documents", []), ensure_ascii=False),
                "crag_score": grade.get("score", ""),
                "crag_reason": grade.get("reason", ""),
                "gold_route": "",
                "baseline_correct": "", "baseline_grounded": "",
                "crag_correct": "", "crag_grounded": "", "notes": "",
            })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--ids", help="Comma-separated question IDs for a smaller run")
    parser.add_argument("--limit", type=int, help="Run only the first N selected questions")
    parser.add_argument("--offline", action="store_true", help="Use the cached Hugging Face embedding model")
    parser.add_argument("--output", type=Path, help="Output JSONL path; review CSV and summary JSON use the same stem")
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    questions = load_questions(args.dataset)
    if args.ids:
        selected = {item.strip() for item in args.ids.split(",") if item.strip()}
        unknown = selected - {row["id"] for row in questions}
        if unknown:
            parser.error(f"Unknown question IDs: {', '.join(sorted(unknown))}")
        questions = [row for row in questions if row["id"] in selected]
    if args.limit is not None:
        questions = questions[:args.limit]
    if args.offline:
        os.environ["HF_HUB_OFFLINE"] = "1"

    from src.crag.llm import OllamaLLM
    from src.crag.pipeline import CRAGPipeline
    from src.crag.settings import Settings
    from src.crag.web import TavilySearch
    from src.retrieval.baseline_rag import baseline_rag
    from src.retrieval.hybrid import HybridRetriever
    from src.retrieval.retriever import load_embedding_model, load_vector_database

    settings = Settings.from_env()
    llm = OllamaLLM(settings.ollama_model, settings.ollama_url)
    llm.check_ready()
    with redirect_stdout(sys.stderr):
        collection = load_vector_database()
        embedding_model = load_embedding_model()
    retriever = HybridRetriever(collection, embedding_model)
    pipeline = CRAGPipeline(
        retrieve=retriever.retrieve,
        llm=llm, web=TavilySearch(settings.tavily_key), settings=settings,
    )

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output = args.output or ROOT / "evaluation" / "results" / f"run_{stamp}.jsonl"
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        parser.error(f"Output already exists: {output}")
    rows = []
    try:
        with output.open("x", encoding="utf-8") as handle:
            for index, question in enumerate(questions, 1):
                print(f"[{index}/{len(questions)}] {question['id']}: {question['question']}", file=sys.stderr, flush=True)
                try:
                    baseline = baseline_rag(question["question"], collection, embedding_model, settings.top_k)
                    if baseline.get("answer", "").startswith("Ollama error:"):
                        baseline["error"] = baseline["answer"]
                except Exception as exc:
                    baseline = {"error": str(exc)}
                try:
                    crag = pipeline.ask(question["question"])
                except Exception as exc:
                    crag = {"error": str(exc)}
                row = {**question, "baseline": baseline, "crag": crag}
                rows.append(row)
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                handle.flush()
                print(f"  baseline={baseline.get('latency', 'error')}s; "
                      f"CRAG={crag.get('route', 'ERROR')} {crag.get('latency_seconds', 'error')}s",
                      file=sys.stderr, flush=True)
    finally:
        review = output.with_suffix(".review.csv")
        summary = output.with_suffix(".summary.json")
        write_review_sheet(review, rows)
        summary.write_text(json.dumps({
            "dataset": str(args.dataset), "created_utc": stamp,
            "model": settings.ollama_model, "top_k": settings.top_k,
            "thresholds": {"low": settings.low_threshold, "high": settings.high_threshold},
            "corpus_vectors": collection.count(), "results": summarize(rows),
        }, indent=2), encoding="utf-8")
        print(f"Raw results: {output}\nReview sheet: {review}\nSummary: {summary}", flush=True)


if __name__ == "__main__":
    main()
