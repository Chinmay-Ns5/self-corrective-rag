"""Exercise the submitted UI through the real CRAG graph without API calls."""
from pathlib import Path
import os
import unittest
from unittest.mock import patch

import streamlit as st
from streamlit.testing.v1 import AppTest

from src.crag.pipeline import CRAGPipeline
from src.crag.settings import Settings
from tests.test_crag import FakeLLM, FakeWeb

APP = Path(__file__).resolve().parents[1] / "app.py"


class AppIntegrationTests(unittest.TestCase):
    def setUp(self):
        st.cache_resource.clear()

    def tearDown(self):
        st.cache_resource.clear()

    def submit(self, app, question="What is RAG?"):
        app.text_area[0].set_value(question)
        next(button for button in app.button if button.label == "Run CRAG").click()
        return app.run()

    def test_form_reaches_backend_and_displays_each_route(self):
        for score, route in [(0.9, "LOCAL"), (0.55, "WEB"), (0.1, "REWRITE_AND_WEB")]:
            with self.subTest(route=route):
                st.cache_resource.clear()
                web = FakeWeb()
                pipeline = CRAGPipeline(
                    lambda question, top_k: {"documents": [["RAG retrieves evidence."]],
                                              "metadatas": [[{"source": "paper.pdf", "page": 2}]]},
                    FakeLLM(score), web, Settings(),
                )
                with patch("src.crag.service.make_pipeline", return_value=pipeline) as factory:
                    app = AppTest.from_file(str(APP), default_timeout=20).run()
                    self.assertEqual(len(app.exception), 0)
                    self.submit(app)
                    self.assertEqual(len(app.exception), 0)
                    self.assertEqual(len(app.error), 0)
                    self.assertEqual(app.session_state["crag_result"]["route"], route)
                    self.assertIn(route, [metric.value for metric in app.metric])
                    rendered = "\n".join(element.value for element in app.markdown)
                    self.assertIn("paper.pdf" if route == "LOCAL" else "https://example.com", rendered)
                    # Widget reruns must not reload the embedding model/database.
                    app.checkbox[2].check().run()
                    self.assertEqual(factory.call_count, 1)
                    self.assertGreater(len(app.text), 0)

    def test_setup_failure_is_displayed(self):
        with patch("src.crag.service.make_pipeline", side_effect=RuntimeError("Local vector database is missing")):
            app = AppTest.from_file(str(APP), default_timeout=20).run()
            self.submit(app)
            self.assertEqual(len(app.exception), 0)
            self.assertGreater(len(app.error), 0)
            self.assertIn("vector database", app.session_state["crag_error"])

    def test_empty_question_does_not_load_backend(self):
        with patch("src.crag.service.make_pipeline") as factory:
            app = AppTest.from_file(str(APP), default_timeout=20).run()
            self.submit(app, "   ")
            self.assertGreater(len(app.warning), 0)
            factory.assert_not_called()

    def test_vague_question_asks_for_context_without_loading_backend(self):
        with patch("src.crag.service.make_pipeline") as factory:
            app = AppTest.from_file(str(APP), default_timeout=20).run()
            self.submit(app, "What is this about?")
            self.assertEqual(len(app.exception), 0)
            self.assertEqual(app.session_state["crag_result"]["route"], "CLARIFY")
            self.assertTrue(any("name the topic or document" in item.value for item in app.info))
            self.assertEqual(len(app.metric), 0)
            factory.assert_not_called()

    def test_missing_tavily_key_shows_setup_step(self):
        class MissingKeyWeb:
            def search(self, query):
                raise RuntimeError("TAVILY_API_KEY is required for web fallback")

        pipeline = CRAGPipeline(
            lambda question, top_k: {"documents": [["Partial evidence"]],
                                      "metadatas": [[{"source": "paper.pdf", "page": 1}]]},
            FakeLLM(0.5), MissingKeyWeb(), Settings(),
        )
        with patch("src.crag.service.make_pipeline", return_value=pipeline):
            app = AppTest.from_file(str(APP), default_timeout=20).run()
            self.submit(app, "A specific topic")
            self.assertEqual(len(app.exception), 0)
            self.assertTrue(any("Reload backend" in item.value for item in app.caption))

    def test_invalid_configuration_is_displayed(self):
        with patch.dict(os.environ, {"CRAG_HIGH_THRESHOLD": "0.1"}):
            app = AppTest.from_file(str(APP), default_timeout=20).run()
            self.assertEqual(len(app.exception), 0)
            self.assertIn("Invalid configuration", app.error[0].value)


if __name__ == "__main__":
    unittest.main()
