"""Experiment contracts, provenance, budgets, and verification; no live services."""
import json
import csv
from pathlib import Path
import re
import tempfile
import unittest

from src.crag.settings import Settings
from src.study.graph import KeywordGraph
from src.study.retrieval import BalancedHybrid
from src.study.systems import BudgetWeb, MeteredLLM, StudyPipeline, make_systems
from src.study.verification import ABSTENTION, ClaimVerifier, sentences
from scripts.compare_rag import evidence_coverage, load_questions, write_review
from scripts.calibrate_routes import calibrate
from scripts.summarize_study import summarize


class Embedding:
    def encode(self, query):
        return self

    def tolist(self):
        return [1.0]


class Collection:
    def __init__(self):
        self.ids = ["bert", "bert2", "sbert", "sbert2", "dense", "dense2"]
        self.documents = ["BERT uses joint sentence encoding.", "BERT uses masking.",
                          "Sentence-BERT uses cosine similarity.", "Sentence-BERT uses triplet networks.",
                          "Dense passage retrieval uses dot products.", "Dense passage retrieval uses independent encoders."]
        self.metadatas = [{"source": source, "page": page} for source, page in
                         [("06_bert.pdf", 1), ("06_bert.pdf", 2),
                          ("04_sentence_bert.pdf", 1), ("04_sentence_bert.pdf", 2),
                          ("05_dense_passage_retrieval.pdf", 1), ("05_dense_passage_retrieval.pdf", 2)]]

    def count(self):
        return len(self.ids)

    def get(self, include):
        return {"ids": self.ids, "documents": self.documents, "metadatas": self.metadatas}

    def query(self, query_embeddings, n_results):
        return {"ids": [self.ids[:n_results]], "documents": [self.documents[:n_results]],
                "metadatas": [self.metadatas[:n_results]]}


class LLM:
    model = "fake"

    def __init__(self):
        self.calls = []

    def complete(self, prompt):
        return self.complete_json(prompt)

    def complete_json(self, prompt):
        self.calls.append(prompt)
        if prompt.startswith("Extract explicit relationships"):
            return json.dumps({"triples": [{"subject": "Sentence-BERT", "relation": "uses",
                "object": "cosine similarity", "chunk_id": "sbert",
                "quote": "Sentence-BERT uses cosine similarity."}]})
        if "keys needs_clarification" in prompt:
            vague = "What did they mean?" in prompt
            return json.dumps({"needs_clarification": vague,
                               "clarification_question": "Who do you mean?" if vague else ""})
        if "keys score" in prompt:
            return json.dumps({"score": 0.9, "reason": "supported", "evidence_ids": ["L1"]})
        return json.dumps({"answer": "BERT uses joint sentence encoding. [L1]"})


class NLI:
    def score(self, premise, claim):
        words = lambda value: set(re.findall(r"\w+", value.lower()))
        yes = words(claim) <= words(premise)
        return {"entailment": 0.99 if yes else 0.01,
                "contradiction": 0.005 if yes else 0.8,
                "neutral": 0.005 if yes else 0.19, "truncated": False}


class RetrievalStudyTests(unittest.TestCase):
    def setUp(self):
        self.hybrid = BalancedHybrid(Collection(), Embedding())

    def test_contiguous_title_match_avoids_sentence_bert_false_positive(self):
        self.assertEqual(self.hybrid.matched_sources("In BERT, is sentence B random?"), ["06_bert.pdf"])
        self.assertEqual(self.hybrid.matched_sources("Explain Sentence-BERT"), ["04_sentence_bert.pdf"])

    def test_two_named_papers_each_get_context_slots(self):
        result = self.hybrid.retrieve("Compare Sentence-BERT and dense passage retrieval", 5)
        sources = [m["source"] for m in result["metadatas"][0]]
        self.assertGreaterEqual(sources.count("04_sentence_bert.pdf"), 2)
        self.assertGreaterEqual(sources.count("05_dense_passage_retrieval.pdf"), 2)
        self.assertEqual(len(set(result["ids"][0])), 5)

    def test_graph_is_lazy_and_repeat_reuses_disk_cache(self):
        llm = LLM()
        with tempfile.TemporaryDirectory() as directory:
            graph = KeywordGraph(self.hybrid, llm, directory)
            self.assertEqual(llm.calls, [])
            graph.retrieve("Sentence-BERT cosine similarity")
            calls = len(llm.calls)
            self.assertGreater(calls, 0)
            graph.retrieve("Sentence-BERT cosine similarity")
            self.assertEqual(len(llm.calls), calls)
            other = KeywordGraph(self.hybrid, llm, directory)
            other.retrieve("Sentence-BERT cosine similarity")
            self.assertEqual(len(llm.calls), calls)
            self.assertGreater(len(other.triples), 0)

    def test_graph_rejects_unsupported_or_foreign_edges(self):
        graph = KeywordGraph(self.hybrid)
        forged = {"subject": "BERT", "relation": "proves", "object": "magic",
                  "chunk_id": "bert", "quote": "BERT proves magic without evidence."}
        self.assertEqual(graph.validate([forged], {"bert"}), [])
        valid = {"subject": "Sentence-BERT", "relation": "uses", "object": "cosine similarity",
                 "chunk_id": "sbert", "quote": "Sentence-BERT uses cosine similarity."}
        self.assertEqual(graph.validate([valid], {"bert"}), [])

    def test_changed_corpus_invalidates_cache(self):
        llm = LLM()
        with tempfile.TemporaryDirectory() as directory:
            KeywordGraph(self.hybrid, llm, directory).retrieve("Sentence-BERT")
            calls = len(llm.calls)
            collection = Collection()
            collection.documents[0] += " Added material."
            changed = BalancedHybrid(collection, Embedding())
            KeywordGraph(changed, llm, directory).retrieve("Sentence-BERT")
            self.assertGreater(len(llm.calls), calls)

    def test_new_region_in_same_document_can_be_expanded(self):
        llm = LLM()
        graph = KeywordGraph(self.hybrid, llm, max_documents=1, chunks_per_document=1)
        graph.expand("04_sentence_bert.pdf", ["sbert"])
        graph.expand("04_sentence_bert.pdf", ["sbert2"])
        self.assertEqual(len(llm.calls), 2)

    def test_malformed_extraction_falls_back_without_unbounded_retries(self):
        class Broken(LLM):
            def complete_json(self, prompt):
                self.calls.append(prompt)
                return '{"triples": ['
        llm = Broken()
        graph = KeywordGraph(self.hybrid, llm, max_documents=1)
        result = graph.retrieve("Sentence-BERT")
        self.assertTrue(result["documents"][0])
        self.assertTrue(next(iter(graph.last_details["expansions"].values())).startswith("failed:"))
        graph.retrieve("Sentence-BERT")
        self.assertEqual(len(llm.calls), 1)
        self.assertEqual(next(iter(graph.last_details["expansions"].values())), "previous_failure")


class VerificationStudyTests(unittest.TestCase):
    def setUp(self):
        self.evidence = [{"id": "L1", "text": "Zephyr uses mirrors.", "source": "one.pdf", "page": 1}]

    def test_citations_after_punctuation_remain_with_claim(self):
        parts = sentences("Zephyr uses mirrors. [L1] Orion uses lenses. [L2]")
        self.assertEqual(len(parts), 2)
        self.assertIn("[L1]", parts[0])
        self.assertNotIn("[L2]", parts[0])

    def test_missing_and_forged_citations_are_unsupported(self):
        verifier = ClaimVerifier(NLI())
        for answer in ["Zephyr uses mirrors.", "Zephyr uses mirrors [L99]."]:
            self.assertFalse(verifier.inspect(answer, self.evidence)[0]["supported"])

    def test_unsupported_claim_is_dropped_with_no_repair_budget(self):
        result = ClaimVerifier(NLI(), max_repairs=0).apply(
            "Zephyr?", "Zephyr uses magic [L1].", self.evidence, None, None)
        self.assertEqual(result["answer"], ABSTENTION)
        self.assertEqual(result["verification"]["removed_claims"], 1)
        self.assertEqual(result["verification"]["after_claims"], 0)

    def test_repair_is_retrieved_recited_and_reverified(self):
        class RepairLLM:
            def complete_json(self, prompt):
                return json.dumps({"answer": "Zephyr uses mirrors [L2]."})
        retrieve = lambda q, k: {"documents": [["Zephyr uses mirrors."]],
                                "metadatas": [[{"source": "two.pdf", "page": 2}]]}
        result = ClaimVerifier(NLI()).apply("Zephyr?", "Zephyr uses magic [L1].",
                                           [{**self.evidence[0], "text": "A different topic."}],
                                           retrieve, RepairLLM())
        self.assertIn("[L2]", result["answer"])
        self.assertEqual(result["verification"]["repaired_claims"], 1)
        self.assertEqual(result["evidence"][0]["source"], "two.pdf")
        self.assertEqual(result["verification"]["initial_evidence"][0]["text"], "A different topic.")
        self.assertTrue(result["verification"]["repair_trace"])

    def test_truncation_cannot_be_reported_as_supported(self):
        class Truncated:
            def score(self, premise, claim):
                return {"entailment": 1, "contradiction": 0, "neutral": 0, "truncated": True}
        self.assertFalse(ClaimVerifier(Truncated()).inspect("Zephyr uses mirrors [L1].", self.evidence)[0]["supported"])


class ResourceStudyTests(unittest.TestCase):
    def test_web_cache_and_budget_apply_across_clients(self):
        class Search:
            calls = 0
            def search(self, query):
                self.calls += 1
                return [{"id": "W1", "text": query, "url": "https://example.com"}]
        with tempfile.TemporaryDirectory() as directory:
            raw = Search()
            web = BudgetWeb(raw, directory, "live", 1)
            web.search("one")
            web.search("one")
            with self.assertRaisesRegex(RuntimeError, "budget"):
                web.search("two")
            self.assertEqual(raw.calls, 1)
            replay = BudgetWeb(cache_dir=directory, mode="cache")
            replay.search("one")
            self.assertEqual(replay.cache_hits, 1)
            self.assertEqual(replay.network_calls, 0)

    def test_disabled_web_never_calls_client(self):
        with self.assertRaisesRegex(RuntimeError, "disabled"):
            BudgetWeb(mode="off").search("question")

    def test_failed_web_attempt_still_consumes_budget(self):
        class Failing:
            def search(self, query):
                raise RuntimeError("network failure")
        web = BudgetWeb(Failing(), mode="live", budget=1)
        with self.assertRaisesRegex(RuntimeError, "network failure"):
            web.search("one")
        with self.assertRaisesRegex(RuntimeError, "budget"):
            web.search("two")
        self.assertEqual(web.network_calls, 1)

    def test_all_five_systems_share_contract_and_clarification(self):
        with tempfile.TemporaryDirectory() as directory:
            systems = make_systems(Collection(), Embedding(), LLM(), Settings(), BudgetWeb(),
                                   ["s1", "s2", "s3", "s4", "s5"], directory, NLI())
            for name, pipeline in systems.items():
                with self.subTest(system=name):
                    result = pipeline.ask("What does BERT use?")
                    self.assertEqual(result["route"], "LOCAL")
                    self.assertEqual(result["system"], name)
                    self.assertTrue(result["evidence"])
                    self.assertEqual(result["sources"][0]["source"], "06_bert.pdf")
                    if name == "s5":
                        self.assertEqual(result["verification"]["after_supported"], 1)
                    vague = pipeline.ask("What did they mean?")
                    self.assertEqual(vague["route"], "CLARIFY")
                    self.assertEqual(vague["metrics"]["llm_calls"], 1)
                    self.assertNotIn("verification", vague)

    def test_baseline_skips_grader_but_uses_shared_clarification(self):
        hybrid = BalancedHybrid(Collection(), Embedding())
        llm = MeteredLLM(LLM())
        baseline = StudyPipeline("s2", hybrid, llm, BudgetWeb(), Settings())
        result = baseline.ask("What does BERT use?")
        self.assertEqual(result["metrics"]["llm_calls"], 2)
        self.assertNotIn("grading", [e["stage"] for e in result["metrics"]["llm_events"]])
        vague = baseline.ask("What did they mean?")
        self.assertEqual(vague["route"], "CLARIFY")
        self.assertEqual(vague["metrics"]["retrieval_calls"], 0)
        self.assertEqual(vague["retrieval_details"], {})

    def test_context_metric_requires_both_comparison_groups(self):
        question = {"gold_groups": [{"ids": ["a"]}, {"ids": ["b"]}]}
        result = evidence_coverage(question, {"retrieved_documents": [{"chunk_id": "a"}]})
        self.assertFalse(result["complete"])
        self.assertEqual(result["covered"], 1)

    def test_context_ids_cannot_match_a_different_source_after_reindexing(self):
        question = {"gold_groups": [{"source": "right.pdf", "ids": ["a"]}]}
        result = evidence_coverage(question, {"retrieved_documents": [{"chunk_id": "a", "source": "wrong.pdf"}]})
        self.assertFalse(result["complete"])

    def test_duplicate_question_ids_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "questions.jsonl"
            path.write_text('{"id":"a","question":"one"}\n{"id":"a","question":"two"}\n')
            with self.assertRaisesRegex(ValueError, "unique"):
                load_questions(path)

    def test_resume_review_preserves_existing_annotations(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "review.csv"
            row = {"id": "q", "system": "s3", "phase": "first", "question": "Question?",
                   "result": {"answer": "Answer", "route": "LOCAL"}}
            write_review(path, [row])
            with path.open(newline="") as handle:
                labels = list(csv.DictReader(handle))
            labels[0]["answer_correct"] = "1"
            labels[0]["gold_route"] = "LOCAL"
            with path.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(labels[0]))
                writer.writeheader()
                writer.writerows(labels)
            write_review(path, [row])
            with path.open(newline="") as handle:
                saved = next(csv.DictReader(handle))
            self.assertEqual(saved["answer_correct"], "1")
            self.assertEqual(saved["gold_route"], "LOCAL")


class CalibrationStudyTests(unittest.TestCase):
    def rows(self, prefix, scores):
        return [{"id": prefix + str(i), "question": prefix + str(i), "score": score, "gold": gold}
                for i, (score, gold) in enumerate(scores)]

    def test_calibration_improves_labels_without_using_evaluation_to_select(self):
        calibration = self.rows("c", [(0.5, "REWRITE_AND_WEB"), (0.65, "WEB"), (0.7, "LOCAL")])
        evaluation = self.rows("e", [(0.55, "REWRITE_AND_WEB"), (0.68, "WEB"), (0.8, "LOCAL")])
        result = calibrate(calibration, evaluation)
        self.assertGreater(result["calibration_after"]["accuracy"], result["calibration_before"]["accuracy"])
        changed_eval = [{**r, "gold": "LOCAL"} for r in evaluation]
        self.assertEqual(result["recommended"], calibrate(calibration, changed_eval)["recommended"])

    def test_id_or_question_overlap_is_rejected(self):
        calibration = self.rows("c", [(0.1, "REWRITE_AND_WEB"), (0.6, "WEB"), (0.9, "LOCAL")])
        with self.assertRaisesRegex(ValueError, "overlap"):
            calibrate(calibration, [{**calibration[0], "id": "different"}])

    def test_empty_review_labels_do_not_become_correctness_scores(self):
        row = {"id": "a", "system": "s5", "phase": "first",
               "result": {"latency_seconds": 1, "metrics": {},
                          "verification": {"before_supported": 0, "before_claims": 1,
                            "after_supported": 0, "after_claims": 0, "removed_claims": 1,
                            "repaired_claims": 0, "repair_attempts": 1, "nli_pairs": 1}}}
        entry = summarize([row])["s5:first"]
        self.assertIsNone(entry["answer_correct"]["rate"])
        self.assertIsNone(entry["reviewed_claim_support"]["rate"])
        self.assertEqual(entry["nli_estimates_not_ground_truth"]["removed_claims"], 1)


if __name__ == "__main__":
    unittest.main()
