from collections import Counter
import unittest

from scripts.evaluate import DEFAULT_DATASET, load_questions, route_matches, summarize
from scripts.summarize_evaluation import score_review


class EvaluationTests(unittest.TestCase):
    def test_dataset_has_balanced_unique_questions(self):
        questions = load_questions(DEFAULT_DATASET)
        self.assertEqual(len(questions), 50)
        self.assertEqual(len({row["id"] for row in questions}), 50)
        self.assertEqual(Counter(row["category"] for row in questions), {
            "local": 20, "web_only": 15, "partial_evidence": 5,
            "unclear_reference": 5, "freshness": 5,
        })

    def test_web_routes_are_equivalent_for_evaluation(self):
        self.assertTrue(route_matches("WEB_ANY", "WEB"))
        self.assertTrue(route_matches("WEB_ANY", "REWRITE_AND_WEB"))
        self.assertFalse(route_matches("WEB_ANY", "LOCAL"))
        self.assertFalse(route_matches("CLARIFY", "WEB"))

    def test_summary_and_human_review_keep_missing_labels_out_of_denominator(self):
        rows = [{"category": "local", "expected_route": "LOCAL",
                 "baseline": {"latency": 2},
                 "crag": {"route": "LOCAL", "latency_seconds": 3, "error": ""}}]
        self.assertEqual(summarize(rows)["dataset_route_agreement"], 1)
        self.assertEqual(summarize(rows)["answer_correctness"], "Requires human review sheet")
        labels = [{"id": "L01", "category": "local", "baseline_correct": "0",
                   "baseline_grounded": "", "crag_correct": "1", "crag_grounded": "1",
                   "gold_route": "WEB", "crag_route": "LOCAL"}]
        scored = score_review(labels)["overall"]
        self.assertEqual(scored["baseline_correct"]["rate"], 0.0)
        self.assertEqual(scored["crag_correct"]["rate"], 1.0)
        self.assertIsNone(scored["baseline_grounded"]["rate"])
        self.assertEqual(scored["routing_accuracy"]["rate"], 0.0)


if __name__ == "__main__":
    unittest.main()
