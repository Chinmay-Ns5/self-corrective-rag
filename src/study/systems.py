"""Five controlled CLI variants with shared prompts and bounded resource use."""
from datetime import date
import hashlib
import json
from pathlib import Path
import time

from src.crag.pipeline import CRAGPipeline, local_chunks
from src.study.verification import ClaimVerifier


SYSTEMS = {
    "s1": "dense local RAG with shared clarification/generator",
    "s2": "balanced hybrid local RAG with shared clarification/generator",
    "s3": "CRAG with the same balanced hybrid retriever as S2",
    "s4": "CRAG with experimental keyword/lazy-semantic graph retrieval",
    "s5": "S3 plus NLI sentence support checks and bounded local repair",
}


class MeteredLLM:
    def __init__(self, inner):
        self.inner = inner
        self.model = getattr(inner, "model", "test")
        self.events = []

    def call(self, prompt, json_mode):
        if prompt.startswith("Extract explicit relationships"):
            stage = "graph_extraction"
        elif prompt.startswith("Correct this single unsupported"):
            stage = "claim_repair"
        elif "keys needs_clarification" in prompt:
            stage = "clarification"
        elif "keys score" in prompt:
            stage = "grading"
        elif prompt.startswith("Rewrite this question"):
            stage = "rewrite"
        else:
            stage = "generation"
        start = time.perf_counter()
        event = {"stage": stage, "json_mode": json_mode, "prompt_characters": len(prompt)}
        try:
            method = getattr(self.inner, "complete_json", self.inner.complete) if json_mode else self.inner.complete
            result = method(prompt)
            event["response_characters"] = len(result)
            event["raw_response"] = result
            return result
        except Exception as exc:
            event["error"] = str(exc)
            raise
        finally:
            event["seconds"] = time.perf_counter() - start
            self.events.append(event)

    def complete(self, prompt):
        return self.call(prompt, False)

    def complete_json(self, prompt):
        return self.call(prompt, True)


class BudgetWeb:
    """Shared, day-scoped replay cache and hard network-call budget."""
    def __init__(self, inner=None, cache_dir=None, mode="off", budget=0):
        if mode not in {"off", "cache", "live"} or budget < 0:
            raise ValueError("Invalid web mode/budget")
        if mode == "live" and (inner is None or budget < 1):
            raise ValueError("Live web requires a positive budget and a search client")
        self.inner, self.mode, self.budget = inner, mode, budget
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.network_calls, self.cache_hits = 0, 0

    def snapshot(self):
        return {"web_network_calls": self.network_calls, "web_cache_hits": self.cache_hits}

    def search(self, query):
        if self.mode == "off":
            raise RuntimeError("Web is disabled for this experiment")
        key = f"web-v1:{date.today().isoformat()}:{' '.join(query.split())}"
        path = self.cache_dir / (hashlib.sha256(key.encode()).hexdigest() + ".json") if self.cache_dir else None
        if path and path.exists():
            try:
                saved = json.loads(path.read_text(encoding="utf-8"))
                if saved.get("key") == key and isinstance(saved.get("results"), list):
                    self.cache_hits += 1
                    return saved["results"]
            except (OSError, ValueError):
                pass
        if self.mode != "live":
            raise RuntimeError("No cached web evidence for this query")
        if self.network_calls >= self.budget:
            raise RuntimeError("Experiment web-search budget exhausted")
        self.network_calls += 1  # Failed network requests still consume the attempt budget.
        results = self.inner.search(query)
        if path:
            path.parent.mkdir(parents=True, exist_ok=True)
            temp = path.with_suffix(".tmp")
            temp.write_text(json.dumps({"key": key, "results": results}), encoding="utf-8")
            temp.replace(path)
        return results


class StudyPipeline(CRAGPipeline):
    def __init__(self, system, retriever, llm, web, settings, verifier=None):
        if system not in SYSTEMS:
            raise ValueError(f"Unknown system: {system}")
        self.system, self.retriever, self.verifier = system, retriever, verifier
        self.used_evidence, self.retrieval_events = [], []
        super().__init__(self._tracked_retrieve, llm, web, settings)

    def _tracked_retrieve(self, question, top_k):
        start = time.perf_counter()
        result = self.retriever.retrieve(question, top_k)
        self.retrieval_events.append({"query": question, "top_k": top_k,
                                      "seconds": time.perf_counter() - start})
        return result

    def _retrieve(self, state):
        raw = self.retrieve(state["question"], self.settings.top_k)
        chunks = local_chunks(raw)
        identifiers = raw.get("ids", [[]])[0]
        for chunk in chunks:
            position = int(chunk["id"][1:]) - 1
            if position < len(identifiers):
                chunk["chunk_id"] = identifiers[position]
        return {"local_evidence": chunks}

    def _evaluate(self, state):
        if self.system in {"s1", "s2"}:
            return {"route": "LOCAL", "evaluation": {"score": None,
                    "reason": "Fixed local baseline: no relevance grader or web routing",
                    "evidence_ids": [item["id"] for item in state["local_evidence"]]}}
        return super()._evaluate(state)

    def _generate(self, state):
        self.used_evidence = list(state.get("evidence", []))
        return super()._generate(state)

    def ask(self, question):
        start = time.perf_counter()
        self.used_evidence, self.retrieval_events = [], []
        self.llm.events.clear()
        web_before = self.web.snapshot()
        result = super().ask(question)
        details = dict(getattr(self.retriever, "last_details", {})) if self.retrieval_events else {}
        if self.verifier and result["route"] != "CLARIFY" and not result["error"]:
            checked = self.verifier.apply(question, result["answer"], self.used_evidence,
                                          self.retrieve, self.llm)
            self.used_evidence = checked.pop("evidence")
            result.update(checked)
            result["sources"] = [{k: v for k, v in e.items() if k != "text"} for e in self.used_evidence]
        result["system"] = self.system
        result["system_description"] = SYSTEMS[self.system]
        result["evidence"] = self.used_evidence
        result["retrieval_details"] = details
        web_after = self.web.snapshot()
        result["metrics"] = {
            "llm_calls": len(self.llm.events), "llm_events": list(self.llm.events),
            "graph_llm_calls": sum(e["stage"] == "graph_extraction" for e in self.llm.events),
            "retrieval_calls": len(self.retrieval_events),
            "retrieval_events": self.retrieval_events,
            **{k: web_after[k] - web_before[k] for k in web_before},
        }
        result["latency_seconds"] = round(time.perf_counter() - start, 4)
        return result


def make_systems(collection, embedding_model, llm, settings, web, names,
                 cache_dir, scorer=None, semantic=True, nli_threshold=0.70):
    from src.study.graph import KeywordGraph
    from src.study.retrieval import BalancedHybrid, DenseRetriever
    hybrid = BalancedHybrid(collection, embedding_model)
    systems = {}
    for name in names:
        metered = MeteredLLM(llm)
        retriever = DenseRetriever(collection, embedding_model) if name == "s1" else hybrid
        if name == "s4":
            retriever = KeywordGraph(hybrid, metered, Path(cache_dir) / "graph", semantic=semantic)
        if name == "s5" and scorer is None:
            raise ValueError("S5 requires an NLI scorer")
        verifier = ClaimVerifier(scorer, nli_threshold) if name == "s5" else None
        systems[name] = StudyPipeline(name, retriever, metered, web, settings, verifier)
    return systems
