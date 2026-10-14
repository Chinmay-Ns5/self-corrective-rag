import unittest

from src.retrieval.hybrid import HybridRetriever


class FakeEmbedding:
    def encode(self, query):
        return self

    def tolist(self):
        return [1.0]


class FakeCollection:
    def __init__(self):
        self.ids = ["unrelated", "definition", "other"]
        self.documents = [
            "General discussion of neural networks and training.",
            "The Zephyr model uses bidirectional training to represent text.",
            "A second document discusses search systems.",
        ]
        self.metadatas = [
            {"source": "other.pdf", "page": 2},
            {"source": "zephyr_model.pdf", "page": 1},
            {"source": "search.pdf", "page": 5},
        ]

    def get(self, include):
        return {"ids": self.ids, "documents": self.documents,
                "metadatas": self.metadatas}

    def query(self, query_embeddings, n_results):
        # Simulate a dense result that overlooks the direct answer.
        return {"ids": [["unrelated", "other"]]}


class HybridTests(unittest.TestCase):
    def test_lexical_index_recovers_answer_missing_from_dense_results(self):
        retriever = HybridRetriever(FakeCollection(), FakeEmbedding())
        result = retriever.retrieve("What does the Zephyr model use?", top_k=2)
        self.assertIn("definition", result["ids"][0])
        self.assertEqual(len(result["documents"][0]), 2)

    def test_invalid_top_k(self):
        retriever = HybridRetriever(FakeCollection(), FakeEmbedding())
        with self.assertRaises(ValueError):
            retriever.retrieve("Zephyr", top_k=0)

    def test_named_paper_recovers_answer_later_on_selected_body_page(self):
        collection = FakeCollection()
        collection.ids = [f"chunk_{i}" for i in range(7)] + ["distractor"]
        collection.documents = [
            "Zephyr model abstract about document representation.",
            "Overview of Zephyr model.",
            "Background and notation used in neural encoders.",
            "Zephyr model uses rotary positional encoding.",
            "Additional method discussion.",
            "Other representation details.",
            "Appendix and references.",
            "Zephyr model positional encoding positional encoding comparisons.",
        ]
        collection.metadatas = [
            {"source": "zephyr_model.pdf", "page": page}
            for page in (1, 1, 2, 2, 3, 3, 10)
        ] + [{"source": "other.pdf", "page": 1}]
        collection.query = lambda **kwargs: {
            "ids": [["distractor", "chunk_0", "chunk_1", "chunk_2", "chunk_4"]]
        }
        result = HybridRetriever(collection, FakeEmbedding()).retrieve(
            "What positional encoding does Zephyr model use?", top_k=5)
        self.assertIn("chunk_3", result["ids"][0])
        self.assertIn("chunk_0", result["ids"][0])
        self.assertIn("chunk_1", result["ids"][0])
        self.assertEqual(len(set(result["ids"][0])), 5)


if __name__ == "__main__":
    unittest.main()
