"""Controlled CLI comparison of S1--S5. Web is off unless explicitly enabled."""
import argparse
from contextlib import redirect_stdout
import csv
from datetime import datetime, timezone
from dataclasses import replace
import json
import os
from pathlib import Path
import sys
import time

from src.crag.settings import Settings
from src.study.systems import BudgetWeb, SYSTEMS, make_systems


ROOT = Path(__file__).resolve().parents[1]


def load_questions(path):
    rows = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    identifiers = set()
    for row in rows:
        if not isinstance(row.get("id"), str) or row["id"] in identifiers or not row["id"]:
            raise ValueError("Question IDs must be nonempty and unique")
        if not isinstance(row.get("question"), str) or not row["question"].strip():
            raise ValueError("Each question must be nonempty text")
        identifiers.add(row["id"])
    if not rows:
        raise ValueError("Dataset is empty")
    return rows


def evidence_coverage(question, result):
    groups = question.get("gold_groups", [])
    if not groups:
        return None
    documents = result.get("retrieved_documents", [])
    hits = [any(e.get("chunk_id") in group["ids"] and
                (not group.get("source") or e.get("source") == group["source"])
                for e in documents) for group in groups]
    return {"groups": len(groups), "covered": sum(hits), "complete": all(hits)}


def write_review(path, rows):
    columns = ["id", "category", "system", "phase", "question", "reference_answer",
               "expected_route", "observed_route", "score", "answer", "sources",
               "context_sufficient", "answer_correct", "answer_grounded", "claims_supported", "claims_total",
               "gold_route", "notes"]
    preserved = {}
    if path.exists():
        with path.open(encoding="utf-8-sig", newline="") as handle:
            preserved = {(r["id"], r["system"], r["phase"]): r for r in csv.DictReader(handle)}
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            result = row["result"]
            labels = preserved.get((row["id"], row["system"], row["phase"]), {})
            writer.writerow({"id": row["id"], "category": row.get("category", ""),
                "system": row["system"], "phase": row["phase"], "question": row["question"],
                "reference_answer": row.get("reference_answer", ""),
                "expected_route": row.get("expected_route", ""),
                "observed_route": result.get("route", ""),
                "score": (result.get("evaluation") or {}).get("score", ""),
                "answer": result.get("answer", ""),
                "sources": json.dumps(result.get("sources", []), ensure_ascii=False),
                **{key: labels.get(key, "") for key in ("context_sufficient", "answer_correct", "answer_grounded",
                    "claims_supported", "claims_total", "gold_route", "notes")}})


def trial_order(names, position, repeat_graph=False, repeat_all=False):
    """Counterbalance first/repeat order; both variants can get warm trials."""
    offset = position % len(names)
    ordered = names[offset:] + names[:offset]
    if repeat_all:
        return [(name, "first") for name in ordered] + [
            (name, "repeat") for name in reversed(ordered)]
    return [(name, phase) for name in ordered for phase in (
        ["first", "repeat"] if name == "s4" and repeat_graph else ["first"])]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    choice = parser.add_mutually_exclusive_group(required=True)
    choice.add_argument("--question")
    choice.add_argument("--dataset", type=Path)
    parser.add_argument("--systems", nargs="+", choices=list(SYSTEMS))
    parser.add_argument("--ids", help="Comma-separated question IDs")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--retrieval-only", action="store_true")
    parser.add_argument("--repeat-graph", action="store_true")
    parser.add_argument("--repeat-all", action="store_true",
                        help="Repeat every variant, reversing order on the second round")
    parser.add_argument("--warmup", action="store_true",
                        help="One untimed neutral generation before trials; counted in manifest")
    parser.add_argument("--web-mode", choices=["off", "cache", "live"], default="off")
    parser.add_argument("--web-budget", type=int, default=0)
    parser.add_argument("--cache-dir", type=Path, default=ROOT / "data/study_cache")
    parser.add_argument("--model-cache", type=Path, default=ROOT / "data/study_models")
    parser.add_argument("--download-models", action="store_true")
    parser.add_argument("--nli-threshold", type=float, default=0.70)
    parser.add_argument("--low-threshold", type=float)
    parser.add_argument("--high-threshold", type=float)
    args = parser.parse_args(argv)
    names = list(dict.fromkeys(args.systems or (["s1", "s2", "s4"] if args.retrieval_only else SYSTEMS)))
    if args.retrieval_only and set(names) & {"s3", "s5"}:
        parser.error("Retrieval-only supports S1/S2/S4; S3/S5 share S2 retrieval and need generation")
    if args.web_budget < 0 or (args.web_mode == "live" and args.web_budget < 1):
        parser.error("Live web needs a positive --web-budget")
    if args.web_mode != "live" and args.web_budget:
        parser.error("--web-budget requires --web-mode live")
    if args.resume and (args.output is None or not args.output.exists()):
        parser.error("--resume requires an existing --output")
    if args.output and args.output.exists() and not args.resume:
        parser.error("Output exists; choose a new path or --resume")
    if args.retrieval_only and args.web_mode != "off":
        parser.error("Retrieval-only does not use web search")
    questions = load_questions(args.dataset) if args.dataset else [
        {"id": "Q1", "category": "interactive", "question": args.question}]
    if args.ids:
        wanted = set(args.ids.split(","))
        if wanted - {q["id"] for q in questions}:
            parser.error("Unknown --ids")
        questions = [q for q in questions if q["id"] in wanted]
    if any(not q["question"].strip() for q in questions):
        parser.error("Question must not be blank")
    previous = []
    if args.resume:
        previous = [json.loads(line) for line in args.output.read_text(encoding="utf-8").splitlines() if line.strip()]
    done = {(r["id"], r["system"], r["phase"]) for r in previous}
    # A resumed run cannot reset its network-search allowance.
    spent = sum(r["result"].get("metrics", {}).get("web_network_calls", 0) for r in previous)
    remaining = max(0, args.web_budget - spent)
    started = time.perf_counter()
    with redirect_stdout(sys.stderr):
        os.environ["ANONYMIZED_TELEMETRY"] = "False"
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        import torch
        torch.set_num_threads(2)
        from sentence_transformers import SentenceTransformer
        from src.crag.llm import OllamaLLM
        from src.crag.web import TavilySearch
        from src.retrieval.retriever import EMBEDDING_MODEL, load_vector_database
        from src.study.graph import fingerprint
        from src.study.verification import NLIScorer
        settings = Settings.from_env()
        settings = replace(settings,
            low_threshold=args.low_threshold if args.low_threshold is not None else settings.low_threshold,
            high_threshold=args.high_threshold if args.high_threshold is not None else settings.high_threshold)
        if not 0 <= settings.low_threshold < settings.high_threshold <= 1:
            parser.error("Thresholds must satisfy 0 <= low < high <= 1")
        collection = load_vector_database()
        embedding = SentenceTransformer(EMBEDDING_MODEL, device="cpu", local_files_only=True)
        llm = OllamaLLM(settings.ollama_model, settings.ollama_url)
        if not args.retrieval_only:
            llm.check_ready()
        warmup_seconds = 0.0
        if args.warmup and not args.retrieval_only:
            warmup_start = time.perf_counter()
            llm.complete("Reply with only the word READY.")
            warmup_seconds = time.perf_counter() - warmup_start
        scorer = NLIScorer(args.model_cache, args.download_models) if "s5" in names else None
        web_mode = "cache" if args.web_mode == "live" and remaining == 0 else args.web_mode
        web = BudgetWeb(TavilySearch(settings.tavily_key) if web_mode == "live" else None,
                        args.cache_dir / "web", web_mode, remaining)
        systems = make_systems(collection, embedding, llm, settings, web, names,
                               args.cache_dir, scorer, semantic=not args.retrieval_only,
                               nli_threshold=args.nli_threshold)
        rows_map = next(iter(systems.values())).retriever
        rows_map = getattr(rows_map, "rows", None) or collection.get(include=["documents", "metadatas"])
        if "ids" in rows_map:
            rows_map = {i: (t, m) for i, t, m in zip(rows_map["ids"], rows_map["documents"], rows_map["metadatas"])}
        manifest = {"schema": 1, "systems": names, "questions": questions,
                    "model": settings.ollama_model, "embedding": EMBEDDING_MODEL,
                    "corpus_fingerprint": fingerprint(rows_map), "top_k": settings.top_k,
                    "low_threshold": settings.low_threshold, "high_threshold": settings.high_threshold,
                    "nli_threshold": args.nli_threshold,
                    "graph_extraction_schema": "v3-max-two-triples",
                    "verification_trace_schema": 2,
                    "nli_model_revision": getattr(scorer, "revision", None),
                    "retrieval_only": args.retrieval_only, "repeat_graph": args.repeat_graph,
                    "web_mode": args.web_mode, "web_budget": args.web_budget,
                    "cache_dir": str(args.cache_dir.resolve()),
                    "context": "Shared generator prompt and initial top-k; S5 repair retrieves up to two chunks per attempted repair; sentence-level NLI proxy, no automatic correctness labels"}
        if args.repeat_all or args.warmup:
            manifest.update(repeat_all=args.repeat_all, warmup=args.warmup,
                            order="rotated first round; reversed repeat round")
    setup_seconds = time.perf_counter() - started
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        meta = args.output.with_suffix(".manifest.json")
        if args.resume:
            old = json.loads(meta.read_text(encoding="utf-8"))["configuration"]
            if old != manifest:
                parser.error("Resume configuration differs from the saved manifest")
        else:
            meta.write_text(json.dumps({"configuration": manifest, "setup_seconds": setup_seconds,
                                       "warmup_seconds": warmup_seconds,
                                       "warmup_llm_calls": int(args.warmup and not args.retrieval_only),
                                       "started_utc": datetime.now(timezone.utc).isoformat()}, indent=2), encoding="utf-8")
    handle = args.output.open("a", encoding="utf-8") if args.output else None
    all_rows = list(previous)
    try:
        for position, question in enumerate(questions):
            # Rotate order instead of always rewarding one variant with a warm model.
            for name, phase in trial_order(names, position, args.repeat_graph, args.repeat_all):
                if (question["id"], name, phase) in done:
                    continue
                print(f"{question['id']} {name} {phase}", file=sys.stderr, flush=True)
                pipeline = systems[name]
                start = time.perf_counter()
                try:
                    if args.retrieval_only:
                        from src.crag.pipeline import local_chunks
                        raw = pipeline.retriever.retrieve(question["question"], settings.top_k)
                        chunks = local_chunks(raw)
                        for item in chunks:
                            item["chunk_id"] = raw["ids"][0][int(item["id"][1:]) - 1]
                        result = {"retrieved_documents": chunks, "mode": "retrieval_only",
                                  "retrieval_details": dict(getattr(pipeline.retriever, "last_details", {})),
                                  "latency_seconds": time.perf_counter() - start,
                                  "metrics": {"llm_calls": 0, "web_network_calls": 0}, "error": ""}
                    else:
                        result = pipeline.ask(question["question"])
                    result["preselected_context_coverage"] = evidence_coverage(question, result)
                except Exception as exc:
                    result = {"error": str(exc), "latency_seconds": time.perf_counter() - start,
                              "metrics": {"llm_calls": len(pipeline.llm.events),
                                          "web_network_calls": web.network_calls - sum(
                                              r["result"].get("metrics", {}).get("web_network_calls", 0)
                                              for r in all_rows[len(previous):])}}
                row = {**question, "system": name, "phase": phase, "result": result}
                all_rows.append(row)
                encoded = json.dumps(row, ensure_ascii=False)
                if handle:
                    handle.write(encoded + "\n")
                    handle.flush()
                else:
                    print(encoded, flush=True)
    finally:
        if handle:
            handle.close()
            write_review(args.output.with_suffix(".review.csv"), all_rows)
    if args.output:
        print(json.dumps({"output": str(args.output), "rows": len(all_rows),
                          "web_network_calls_this_invocation": web.network_calls,
                          "setup_seconds": round(setup_seconds, 3)}))


if __name__ == "__main__":
    main()
