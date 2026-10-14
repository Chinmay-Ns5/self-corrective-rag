"""CPU-only retrieval ablation; never calls Ollama, Tavily, or a generator.

Evidence anchors are transparent lexical proxies, not semantic answer scores.
Query embeddings are shared across variants so each sees the same vector.
"""
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import re
import statistics
import time
import unicodedata


ROOT = Path(__file__).resolve().parents[1]


def normalize(text):
    text = unicodedata.normalize("NFKC", text).lower()
    text = re.sub(r"-\s*\n\s*", "", text)
    return re.sub(r"\s+", " ", text)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path,
                        default=ROOT / "evaluation/retrieval_questions.jsonl")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "evaluation/results/retrieval_ablation.json")
    parser.add_argument("--repeats", type=int, default=7)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    if args.output.exists():
        parser.error(f"Output already exists: {args.output}; choose a new --output path")
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["ANONYMIZED_TELEMETRY"] = "False"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["OMP_NUM_THREADS"] = "2"
    os.environ["MKL_NUM_THREADS"] = "2"
    print("Loading local index and CPU embedding model...", flush=True)
    started = time.perf_counter()
    import torch
    torch.set_num_threads(2)
    from sentence_transformers import SentenceTransformer
    from src.retrieval.hybrid import HybridRetriever, terms, title_terms
    from src.retrieval.retriever import load_vector_database, EMBEDDING_MODEL
    collection = load_vector_database()
    model = SentenceTransformer(EMBEDDING_MODEL, device="cpu", local_files_only=True)
    startup = time.perf_counter() - started

    class CachedEmbedding:
        vector = None

        def encode(self, query):
            return self.vector

    class PlainHybrid(HybridRetriever):
        def _paper_ids(self, query_terms, scores):
            return []

    class ScoredBodyHybrid(HybridRetriever):
        """Ablation candidate: preserve current policy, choose the scored chunk.

        The only difference is selecting the highest lexical-scoring chunk on
        each chosen body page instead of the first chunk on that page.
        """
        use_scored_body = True
        def _paper_ids(self, query_terms, scores):
            matches = []
            for source, identifiers in self.sources.items():
                title = title_terms(source)
                if not title or not title <= query_terms:
                    continue
                first_page = min(self.rows[row_id][1].get("page", 0) for row_id in identifiers)
                early = [row_id for row_id in identifiers
                         if self.rows[row_id][1].get("page", 0) == first_page]
                if not early:
                    continue
                def reading_order(row_id):
                    suffix = row_id.rsplit("_", 1)[-1]
                    return int(suffix) if suffix.isdigit() else self.order[row_id]
                early.sort(key=reading_order)
                last_page = max(self.rows[row_id][1].get("page", 0) for row_id in identifiers)
                body_limit = first_page + max(2, (last_page - first_page) // 2)
                best_by_page = {}
                first_by_page = {}
                for row_id in identifiers:
                    page = self.rows[row_id][1].get("page", 0)
                    if first_page < page <= body_limit:
                        if page not in best_by_page or scores[row_id] > scores[best_by_page[page]]:
                            best_by_page[page] = row_id
                        if page not in first_by_page or reading_order(row_id) < reading_order(first_by_page[page]):
                            first_by_page[page] = row_id
                pages = sorted(best_by_page, key=lambda page: scores[best_by_page[page]],
                               reverse=True)[:2]
                chosen = best_by_page if self.use_scored_body else first_by_page
                matches.append((len(title), early[:2] + [chosen[page] for page in pages]))
            matches.sort(key=lambda match: match[0], reverse=True)
            return [row_id for _, selected in matches[:1] for row_id in selected]

    class FirstBodyHybrid(ScoredBodyHybrid):
        """Freeze the pre-adoption baseline for reproducible comparisons."""
        use_scored_body = False

    cached = CachedEmbedding()
    current = FirstBodyHybrid(collection, cached)
    plain = PlainHybrid(collection, cached)
    scored_body = ScoredBodyHybrid(collection, cached)
    variants = {
        "dense": lambda q: collection.query(
            query_embeddings=[cached.vector.tolist()], n_results=5)["ids"][0],
        "hybrid_without_reservation": lambda q: plain.retrieve(q, 5)["ids"][0],
        "current_hybrid": lambda q: current.retrieve(q, 5)["ids"][0],
        "scored_body_candidate": lambda q: scored_body.retrieve(q, 5)["ids"][0],
    }
    questions = [json.loads(line) for line in args.dataset.read_text(
        encoding="utf-8").splitlines() if line.strip()]
    model.encode("Warm up retrieval benchmark.")
    results = []
    for position, question in enumerate(questions):
        query = question["question"]
        if "gold_groups" in question:
            groups = question["gold_groups"]
            for group in groups:
                if not group["ids"] or any(row_id not in current.rows or
                    current.rows[row_id][1].get("source") != group["source"]
                    for row_id in group["ids"]):
                    raise ValueError(f"Invalid reviewed passage IDs for {question['id']}")
        else:
            patterns = [re.compile(pattern) for pattern in question["answer_patterns"]]
            gold_ids = [row_id for row_id, (text, metadata) in current.rows.items()
                        if metadata.get("source") == question["source"] and
                        all(pattern.search(normalize(text)) for pattern in patterns)]
            groups = [{"source": question["source"], "ids": gold_ids}]
        if any(not group["ids"] for group in groups):
            raise ValueError(f"No anchor passage for {question['id']}")
        gold = sorted({row_id for group in groups for row_id in group["ids"]})
        sources = {group["source"] for group in groups}
        start = time.perf_counter()
        cached.vector = model.encode(query)
        embedding_ms = (time.perf_counter() - start) * 1000
        reserved = current._paper_ids(set(terms(query)), current._lexical_scores(set(terms(query))))
        row = {**question, "gold_anchor_ids": gold, "reserved_ids": reserved,
               "embedding_ms": embedding_ms, "variants": {}}
        measurements = {name: [] for name in variants}
        outputs = {}
        names = list(variants)
        for name in names:
            variants[name](query)  # Warm the index before measuring.
        for repetition in range(args.repeats):
            offset = (position + repetition) % len(names)
            for name in names[offset:] + names[:offset]:
                start = time.perf_counter()
                outputs[name] = variants[name](query)
                measurements[name].append((time.perf_counter() - start) * 1000)
        for name, ids in outputs.items():
            first_ranks = [min((rank for rank, row_id in enumerate(ids, 1)
                               if row_id in group["ids"]), default=0) for group in groups]
            complete = all(first_ranks)
            same_source = sum(current.rows[row_id][1].get("source") in sources for row_id in ids)
            ranking_ms = statistics.median(measurements[name])
            row["variants"][name] = {
                "ids": ids, "anchor_hit_at_5": int(complete),
                "anchor_group_coverage_at_5": sum(rank > 0 for rank in first_ranks) / len(groups),
                "anchor_reciprocal_rank_at_5": 1 / max(first_ranks) if complete else 0,
                "target_paper_hit_at_5": int(same_source > 0),
                "target_paper_fraction_at_5": same_source / len(ids),
                "ranking_median_ms": ranking_ms,
                "embedding_plus_ranking_ms": embedding_ms + ranking_ms,
            }
        results.append(row)
        print(f"[{position+1}/{len(questions)}] {question['id']} " + " ".join(
            f"{name}={row['variants'][name]['anchor_hit_at_5']}" for name in names), flush=True)

    def summary(rows):
        out = {}
        for name in variants:
            metrics = [row["variants"][name] for row in rows]
            out[name] = {"questions": len(rows),
                         "anchor_hits": sum(m["anchor_hit_at_5"] for m in metrics)}
            for key in ("anchor_hit_at_5", "anchor_group_coverage_at_5", "anchor_reciprocal_rank_at_5",
                        "target_paper_hit_at_5", "target_paper_fraction_at_5",
                        "ranking_median_ms", "embedding_plus_ranking_ms"):
                out[name][key] = statistics.mean(m[key] for m in metrics)
        return out

    artifact = {
        "method": "top-5 retrieval; hit requires at least one labeled chunk per evidence group; groups use explicit reviewed IDs when provided, otherwise regex anchors",
        "limitations": "Passage labels may not be exhaustive; lexical anchors are proxies, not semantic answer accuracy. Questions are authored by the implementer, not an independent blinded evaluator.",
        "model": EMBEDDING_MODEL, "device": "cpu", "torch_threads": 2,
        "vectors": len(current.rows), "repeats": args.repeats,
        "startup_seconds": startup,
        "variant_definitions": {
            "current_hybrid": "Historical baseline before adoption: first chunk on each chosen body page",
            "scored_body_candidate": "Highest-scoring chunk on each chosen body page; adopted into production after fresh checks",
        },
        "question_groups": dict(Counter(q["group"] for q in questions)),
        "summary": summary(results),
        "by_group": {group: summary([row for row in results if row["group"] == group])
                     for group in sorted({row["group"] for row in results})},
        "reservation_triggered": summary([row for row in results if row["reserved_ids"]]),
        "rows": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, indent=2, ensure_ascii=False) + "\n",
                           encoding="utf-8")
    print(json.dumps(artifact["summary"], indent=2), flush=True)
    print(f"Saved {args.output}", flush=True)


if __name__ == "__main__":
    main()
