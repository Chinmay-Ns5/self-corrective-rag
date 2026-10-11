"""CLI contract checks without loading Ollama or calling Tavily."""

from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
import json
import unittest
from unittest.mock import patch

from scripts import ask


class CliTests(unittest.TestCase):
    def test_result_is_machine_readable_json_on_stdout(self):
        class Backend:
            def ask(self, question):
                self.question = question
                return {"question": question, "route": "LOCAL", "answer": "Answer [L1]"}

        backend = Backend()
        output, errors = StringIO(), StringIO()
        with patch("sys.argv", ["ask", "What is BERT?"]), \
                patch.object(ask, "make_pipeline", return_value=backend), \
                redirect_stdout(output), redirect_stderr(errors):
            ask.main()
        self.assertEqual(json.loads(output.getvalue())["answer"], "Answer [L1]")
        self.assertEqual(backend.question, "What is BERT?")

    def test_blank_question_returns_json_error_without_loading_backend(self):
        output = StringIO()
        with patch("sys.argv", ["ask", "   "]), \
                patch.object(ask, "make_pipeline") as factory, \
                redirect_stdout(output), redirect_stderr(StringIO()):
            with self.assertRaises(SystemExit) as stopped:
                ask.main()
        self.assertEqual(stopped.exception.code, 1)
        self.assertIn("Question must not be empty", json.loads(output.getvalue())["error"])
        factory.assert_not_called()


if __name__ == "__main__":
    unittest.main()
