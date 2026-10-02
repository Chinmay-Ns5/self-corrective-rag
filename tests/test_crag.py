import json
import unittest

from src.crag.pipeline import CRAGPipeline, choose_route, parse_grade
from src.crag.settings import Settings


class FakeLLM:
    def __init__(self, score):
        self.score = score

    def complete(self, prompt):
        if "Return ONLY JSON" in prompt:
            return json.dumps({"score": self.score, "reason": "test", "evidence_ids": ["L1"]})
        if "Rewrite this question" in prompt:
            return "rewritten test query"
        if "[W1]" in prompt:
            return "Answer from web. [W1]"
        return "Answer from PDF. [L1]"


class FakeWeb:
    def __init__(self):
        self.queries = []

    def search(self, query):
        self.queries.append(query)
        return [{"id": "W1", "title": "Example", "url": "https://example.com",
                 "text": "A useful web snippet", "score": 0.9}]


class BrokenWeb:
    def search(self, query):
        raise RuntimeError("service unavailable")


class MalformedGrader(FakeLLM):
    def complete(self, prompt):
        if "Return ONLY JSON" in prompt:
            return "invalid grade"
        return super().complete(prompt)


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.settings = Settings()
        self.results = {"documents": [["The answer is in this PDF."]],
                        "metadatas": [[{"source": "paper.pdf", "page": 3}]]}

    def pipeline(self, score, results=None):
        web = FakeWeb()
        pipeline = CRAGPipeline(lambda question, top_k: results or self.results,
                                FakeLLM(score), web, self.settings)
        return pipeline, web

    def test_three_routes_and_sources(self):
        for score, route, expected_query in [
            (0.9, "LOCAL", None), (0.55, "WEB", "test question"),
            (0.1, "REWRITE_AND_WEB", "rewritten test query")
        ]:
            with self.subTest(route=route):
                pipeline, web = self.pipeline(score)
                result = pipeline.ask("test question")
                self.assertEqual(result["route"], route)
                self.assertEqual(web.queries, [] if expected_query is None else [expected_query])
                self.assertEqual(result["sources"][0]["page"] if route == "LOCAL" else
                                 result["sources"][0]["url"],
                                 3 if route == "LOCAL" else "https://example.com")

    def test_empty_corpus_rewrites(self):
        pipeline, web = self.pipeline(0.9, {"documents": [[]], "metadatas": [[]]})
        result = pipeline.ask("test question")
        self.assertEqual(result["route"], "REWRITE_AND_WEB")
        self.assertEqual(web.queries, ["rewritten test query"])

    def test_invalid_grade_and_thresholds(self):
        with self.assertRaises(ValueError):
            parse_grade('{"score": 1.2, "evidence_ids": ["L1"]}', {"L1"})
        self.assertEqual(choose_route(0.75, self.settings), "LOCAL")
        self.assertEqual(choose_route(0.40, self.settings), "WEB")

    def test_web_failure_does_not_fabricate_answer(self):
        pipeline = CRAGPipeline(lambda question, top_k: self.results,
                                FakeLLM(0.5), BrokenWeb(), self.settings)
        result = pipeline.ask("test question")
        self.assertIn("could not verify", result["answer"])
        self.assertIn("Web search failed", result["error"])
        self.assertEqual(result["sources"], [])

    def test_malformed_grade_routes_conservatively(self):
        web = FakeWeb()
        pipeline = CRAGPipeline(lambda question, top_k: self.results,
                                MalformedGrader(0.9), web, self.settings)
        result = pipeline.ask("test question")
        self.assertEqual(result["route"], "REWRITE_AND_WEB")
        self.assertIn("Grader failed", result["evaluation"]["reason"])


if __name__ == "__main__":
    unittest.main()
