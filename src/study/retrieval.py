"""Controlled retrieval variants and document-balanced context allocation."""
from collections import defaultdict
from pathlib import Path
import re

from src.retrieval.hybrid import HybridRetriever, terms


def normalized(value):
    return " ".join(terms(value))


class DenseRetriever:
    def __init__(self, collection, model):
        self.collection, self.model = collection, model
        self.count = collection.count()
        self.last_details = {}

    def retrieve(self, query, top_k=5):
        if top_k < 1:
            raise ValueError("top_k must be positive")
        if not self.count:
            return {"ids": [[]], "documents": [[]], "metadatas": [[]]}
        return self.collection.query(query_embeddings=[self.model.encode(query).tolist()],
                                     n_results=min(top_k, self.count))


class BalancedHybrid(HybridRetriever):
    """RRF with explicit, contiguous filename-title matching and shared slots.

    Used identically by S2/S3/S5. No paper-specific aliases or question rules.
    A title embedded inside a longer matched title is not a separate paper.
    """
    def matched_sources(self, query):
        text = normalized(query)
        matches = []
        for source in self.sources:
            title = normalized(re.sub(r"^\d+_", "", Path(source).stem))
            if not title:
                continue
            for match in re.finditer(r"(?<!\w)" + re.escape(title) + r"(?!\w)", text):
                matches.append((match.start(), match.end(), source))
        matches = [m for m in matches if not any(
            other[0] <= m[0] and m[1] <= other[1] and
            (other[1] - other[0]) > (m[1] - m[0]) for other in matches)]
        return list(dict.fromkeys(m[2] for m in sorted(matches)))

    def ranked(self, query, top_k=5):
        if top_k < 1:
            raise ValueError("top_k must be positive")
        if not self.rows:
            return [], {}, {}
        n = min(max(20, top_k * 4), len(self.rows))
        dense = self.collection.query(query_embeddings=[self.embedding_model.encode(query).tolist()],
                                      n_results=n)["ids"][0]
        lexical = self._lexical_scores(set(terms(query)))
        sparse = sorted(lexical, key=lexical.get, reverse=True)[:n]
        fusion = defaultdict(float)
        for ranking in (dense, sparse):
            for rank, row_id in enumerate(ranking, 1):
                if row_id in self.rows:
                    fusion[row_id] += 1 / (20 + rank)
        ranked = sorted(fusion, key=fusion.get, reverse=True)
        return ranked, dict(fusion), lexical

    def allocate(self, ranked, query, top_k, scores=None):
        scores = scores or {}
        sources = self.matched_sources(query)[:top_k]
        choices = {}
        for source in sources:
            candidates = [row_id for row_id in ranked if self.rows[row_id][1].get("source") == source]
            # Explicitly named sources can contribute even when all their chunks
            # missed both short candidate lists. Only local lexical scores rank them.
            if not candidates:
                candidates = sorted(self.sources[source], key=lambda i: scores.get(i, 0), reverse=True)
            if candidates:
                first = candidates[0]
                distinct = [i for i in candidates[1:] if
                            self.rows[i][1].get("page") != self.rows[first][1].get("page")]
                choices[source] = [first] + (distinct[:1] or candidates[1:2])
        selected = []
        for slot in range(2):
            for source in sources:
                options = choices.get(source, [])
                if slot < len(options) and len(selected) < top_k:
                    selected.append(options[slot])
        selected.extend(i for i in ranked if i not in selected)
        return selected[:top_k]

    def pack(self, ids):
        return {"ids": [ids], "documents": [[self.rows[i][0] for i in ids]],
                "metadatas": [[self.rows[i][1] for i in ids]]}

    def retrieve(self, query, top_k=5):
        ranked, fusion, lexical = self.ranked(query, top_k)
        ids = self.allocate(ranked, query, top_k, lexical)
        self.last_details = {"matched_sources": self.matched_sources(query),
                             "selected_ids": ids, "fusion_scores": {i: fusion.get(i, 0) for i in ids}}
        return self.pack(ids)
