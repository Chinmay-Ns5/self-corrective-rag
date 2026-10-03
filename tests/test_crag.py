import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from src.crag.pipeline import CRAGPipeline, choose_route, parse_grade
from src.crag.settings import Settings


class FakeLLM:
    def __init__(self, score):
        self.score = score

    def complete(self, prompt):
        if "Return ONLY JSON with keys needs_clarification" in prompt:
            return json.dumps({"needs_clarification": False, "clarification_question": ""})
        if "Return ONLY JSON with keys score" in prompt:
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
        if "Return ONLY JSON with keys score" in prompt:
            return "invalid grade"
        return super().complete(prompt)


class VerboseRewriter(FakeLLM):
    def complete(self, prompt):
        if "Rewrite this question" in prompt:
            return "Here are my steps:\nPreserve the question, then perform a search."
        return super().complete(prompt)


class AmbiguityLLM(FakeLLM):
    def complete(self, prompt):
        if "Return ONLY JSON with keys needs_clarification" in prompt:
            return json.dumps({"needs_clarification": True,
                               "clarification_question": "Which topic or document do you mean?"})
        return super().complete(prompt)


class FakeJSONLLM(FakeLLM):
    def complete_json(self, prompt):
        raw = self.complete(prompt)
        if "Return ONLY JSON with keys score" in prompt or \
                "Return ONLY JSON with keys needs_clarification" in prompt:
            return raw
        return json.dumps({"answer": raw})


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

    def test_verbose_rewrite_preserves_original_search_query(self):
        web = FakeWeb()
        pipeline = CRAGPipeline(lambda question, top_k: self.results,
                                VerboseRewriter(0.1), web, self.settings)
        pipeline.ask("test question")
        self.assertEqual(web.queries, ["test question"])

    def test_structured_answer_is_unwrapped_for_the_ui(self):
        pipeline = CRAGPipeline(lambda question, top_k: self.results,
                                FakeJSONLLM(0.9), FakeWeb(), self.settings)
        result = pipeline.ask("test question")
        self.assertEqual(result["answer"], "Answer from PDF. [L1]")
        self.assertEqual(result["error"], "")

    def test_question_without_a_referent_requests_clarification(self):
        def unexpected_retrieval(question, top_k):
            self.fail("A vague question must not search unrelated PDFs")

        web = FakeWeb()
        pipeline = CRAGPipeline(unexpected_retrieval, AmbiguityLLM(0.9), web, self.settings)
        for question in ("What is this about?", "Summarize this paper."):
            with self.subTest(question=question):
                result = pipeline.ask(question)
                self.assertEqual(result["route"], "CLARIFY")
                self.assertIn("topic or document", result["answer"])
                self.assertIsNone(result["evaluation"])
                self.assertEqual(result["sources"], [])
        self.assertEqual(web.queries, [])

    def test_invalid_query_assessment_fails_before_retrieval(self):
        class InvalidChecker(FakeLLM):
            def complete(self, prompt):
                if "Return ONLY JSON with keys needs_clarification" in prompt:
                    return '{"needs_clarification": "maybe", "clarification_question": ""}'
                return super().complete(prompt)

        pipeline = CRAGPipeline(lambda question, top_k: self.fail("retrieval was called"),
                                InvalidChecker(0.9), FakeWeb(), self.settings)
        with self.assertRaisesRegex(ValueError, "Invalid query assessment"):
            pipeline.ask("What is DPR?")


class SettingsTests(unittest.TestCase):
    def test_tavily_key_edit_is_visible_without_restarting(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            env_file = Path(temp_dir) / ".env"
            with patch("src.crag.settings.PROJECT_ROOT", Path(temp_dir)), \
                    patch.dict(os.environ, {"TAVILY_API_KEY": ""}):
                env_file.write_text("TAVILY_API_KEY=\n", encoding="utf-8")
                self.assertEqual(Settings.from_env().tavily_key, "")
                env_file.write_text("TAVILY_API_KEY=test-key\n", encoding="utf-8")
                self.assertEqual(Settings.from_env().tavily_key, "test-key")
                self.assertEqual(os.environ["TAVILY_API_KEY"], "")


if __name__ == "__main__":
    unittest.main()
