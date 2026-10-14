"""Combine dense retrieval with a small lexical index of the ingested chunks."""

from collections import Counter, defaultdict
import math
import re


def terms(value):
    return re.findall(r"[a-z0-9]+", value.lower())


TITLE_STOPWORDS = {"a", "an", "and", "for", "in", "is", "of", "on", "the", "to", "you"}


def title_terms(source):
    title = re.sub(r"^\d+_", "", str(source))
    title = re.sub(r"\.pdf$", "", title, flags=re.I).replace("_", " ")
    return set(terms(title)) - TITLE_STOPWORDS


class HybridRetriever:
    def __init__(self, collection, embedding_model):
        self.collection = collection
        self.embedding_model = embedding_model
        rows = collection.get(include=["documents", "metadatas"])
        self.order = {row_id: position for position, row_id in enumerate(rows["ids"])}
        self.rows = {
            row_id: (document, metadata or {})
            for row_id, document, metadata in zip(
                rows["ids"], rows["documents"], rows["metadatas"]
            ) if document and document.strip()
        }
        self.counts = {}
        self.sources = defaultdict(list)
        frequencies = Counter()
        lengths = []
        for row_id, (document, metadata) in self.rows.items():
            # Paper titles in filenames are useful for questions naming a paper.
            source = str(metadata.get("source", ""))
            self.sources[source].append(row_id)
            counts = Counter(terms(document))
            counts.update({term: 3 for term in title_terms(source)})
            self.counts[row_id] = counts
            frequencies.update(counts.keys())
            lengths.append(sum(counts.values()))
        self.idf = {
            term: math.log(1 + (len(self.rows) - frequency + 0.5) / (frequency + 0.5))
            for term, frequency in frequencies.items()
        }
        self.avg_length = sum(lengths) / len(lengths) if lengths else 1

    def _lexical_scores(self, query_terms):
        scores = {}
        for row_id, counts in self.counts.items():
            length = sum(counts.values())
            score = 0.0
            for term in query_terms & counts.keys():
                tf = counts[term]
                score += self.idf[term] * tf * 2.2 / (
                    tf + 1.2 * (0.25 + 0.75 * length / self.avg_length)
                )
            scores[row_id] = score
        return scores

    def _paper_ids(self, query_terms, scores):
        """Reserve abstract and body passages from a paper named in the query."""
        def reading_order(row_id):
            suffix = row_id.rsplit("_", 1)[-1]
            return int(suffix) if suffix.isdigit() else self.order[row_id]

        matches = []
        for source, identifiers in self.sources.items():
            title = title_terms(source)
            if not title or not title <= query_terms:
                continue
            first_page = min(self.rows[row_id][1].get("page", 0) for row_id in identifiers)
            early = [row_id for row_id in identifiers
                     if self.rows[row_id][1].get("page", 0) == first_page]
            if early:
                # Ingested IDs preserve reading order; overlapping chunks carry
                # the title and abstract across the first two passages.
                early.sort(key=reading_order)
                last_page = max(self.rows[row_id][1].get("page", 0) for row_id in identifiers)
                body_limit = first_page + max(2, (last_page - first_page) // 2)
                body = [row_id for row_id in identifiers
                        if first_page < self.rows[row_id][1].get("page", 0) <= body_limit]
                selected = early[:2]
                # Keep two distinct body pages so one lexical hit cannot crowd
                # out the neighboring method or action description.
                best_by_page = {}
                for row_id in body:
                    page = self.rows[row_id][1]["page"]
                    if page not in best_by_page or scores[row_id] > scores[best_by_page[page]]:
                        best_by_page[page] = row_id
                pages = sorted(best_by_page, key=lambda page: scores[best_by_page[page]],
                               reverse=True)[:2]
                # The page's first chunk can be boilerplate; reserve the chunk
                # that actually scored highest for this question instead.
                selected.extend(best_by_page[page] for page in pages)
                matches.append((len(title), selected))
        matches.sort(key=lambda match: match[0], reverse=True)
        return [row_id for _, early in matches[:1] for row_id in early]

    def retrieve(self, query, top_k=5):
        if top_k < 1:
            raise ValueError("top_k must be positive")
        if not self.rows:
            return {"documents": [[]], "metadatas": [[]], "distances": [[]]}
        candidate_count = min(max(top_k * 4, 20), len(self.rows))
        embedding = self.embedding_model.encode(query).tolist()
        dense = self.collection.query(
            query_embeddings=[embedding], n_results=candidate_count
        )
        dense_ids = dense.get("ids", [[]])[0]
        query_terms = set(terms(query))
        lexical_scores = self._lexical_scores(query_terms)
        lexical_ids = sorted(lexical_scores, key=lexical_scores.get, reverse=True)[:candidate_count]
        ranks = defaultdict(float)
        for weight, identifiers in ((1.0, dense_ids), (1.0, lexical_ids)):
            for rank, row_id in enumerate(identifiers, 1):
                if row_id in self.rows:
                    ranks[row_id] += weight / (rank + 20)
        selected = self._paper_ids(query_terms, lexical_scores)[:top_k]
        selected.extend(row_id for row_id in sorted(ranks, key=ranks.get, reverse=True)
                        if row_id not in selected)
        selected = selected[:top_k]
        return {
            "ids": [selected],
            "documents": [[self.rows[row_id][0] for row_id in selected]],
            "metadatas": [[self.rows[row_id][1] for row_id in selected]],
        }
