"""Composable CRAG workflow. Dependencies are injected for deterministic testing."""
import json
import re
import time
from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from src.crag.prompts import GENERATOR, GRADER, REWRITER
from src.crag.settings import Settings


class CRAGState(TypedDict, total=False):
    question: str
    local_evidence: list[dict]
    evaluation: dict
    route: str
    rewritten_query: str
    web_evidence: list[dict]
    evidence: list[dict]
    answer: str
    error: str


def local_chunks(results):
    documents = results.get("documents", [[]])[0]
    metadata = results.get("metadatas", [[]])[0]
    return [{"id": f"L{i}", "text": text, "source": (metadata[i - 1] or {}).get("source", "unknown"),
             "page": (metadata[i - 1] or {}).get("page", 0)}
            for i, text in enumerate(documents, start=1) if text.strip()]


def format_evidence(items, max_chars=1200):
    return "\n\n".join(
        f"[{item['id']}] {item.get('source') or item.get('title') or ''} "
        f"{item.get('page') or item.get('url') or ''}\n{item['text'][:max_chars]}"
        for item in items
    )


def parse_grade(raw, valid_ids):
    try:
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        value = json.loads(match.group(0) if match else raw)
        score = float(value["score"])
        if not 0 <= score <= 1:
            raise ValueError("score outside [0, 1]")
        ids = value.get("evidence_ids", [])
        if not isinstance(ids, list) or any(item not in valid_ids for item in ids):
            raise ValueError("invalid evidence IDs")
        return {"score": score, "reason": str(value.get("reason", ""))[:300],
                "evidence_ids": ids}
    except (ValueError, TypeError, KeyError, AttributeError) as exc:
        raise ValueError(f"Invalid retrieval grade: {exc}") from exc


def choose_route(score, settings):
    if score >= settings.high_threshold:
        return "LOCAL"
    if score >= settings.low_threshold:
        return "WEB"
    return "REWRITE_AND_WEB"


class CRAGPipeline:
    def __init__(self, retrieve, llm, web, settings=None):
        self.retrieve = retrieve
        self.llm = llm
        self.web = web
        self.settings = settings or Settings.from_env()
        self.graph = self._build_graph()

    def _retrieve(self, state):
        return {"local_evidence": local_chunks(self.retrieve(state["question"], self.settings.top_k))}

    def _evaluate(self, state):
        chunks = state["local_evidence"]
        if not chunks:
            grade = {"score": 0.0, "reason": "No local documents found", "evidence_ids": []}
        else:
            try:
                raw = self.llm.complete(GRADER.format(question=state["question"],
                                                      evidence=format_evidence(chunks)))
                grade = parse_grade(raw, {item["id"] for item in chunks})
                # A high grade without any cited excerpt is not trusted.
                if grade["score"] >= self.settings.high_threshold and not grade["evidence_ids"]:
                    grade = {**grade, "score": self.settings.low_threshold}
            except Exception as exc:
                grade = {"score": 0.0, "reason": f"Grader failed: {exc}", "evidence_ids": []}
        return {"evaluation": grade, "route": choose_route(grade["score"], self.settings)}

    def _rewrite(self, state):
        try:
            query = self.llm.complete(REWRITER.format(question=state["question"])).strip().strip('"')
            return {"rewritten_query": query[:300] or state["question"]}
        except Exception:
            return {"rewritten_query": state["question"]}

    def _web(self, state):
        from src.crag.web import refine_web
        query = state.get("rewritten_query") or state["question"]
        try:
            evidence = refine_web(self.web.search(query))
            return {"web_evidence": evidence,
                    "error": "" if evidence else "Web search returned no usable evidence"}
        except Exception as exc:
            return {"web_evidence": [], "error": f"Web search failed: {exc}"}

    def _prepare_local(self, state):
        allowed = set(state["evaluation"]["evidence_ids"])
        return {"evidence": [item for item in state["local_evidence"] if item["id"] in allowed]}

    def _prepare_web(self, state):
        return {"evidence": state.get("web_evidence", [])}

    def _generate(self, state):
        evidence = state.get("evidence", [])
        if not evidence:
            return {"answer": "I could not verify an answer from the available evidence."}
        try:
            answer = self.llm.complete(GENERATOR.format(
                question=state["question"], evidence=format_evidence(evidence)))
            if not answer:
                raise ValueError("empty response")
            valid_ids = {item["id"] for item in evidence}
            cited_ids = set(re.findall(r"\[([LW]\d+)\]", answer))
            if not cited_ids or not cited_ids <= valid_ids:
                raise ValueError("answer lacks valid evidence citations")
            return {"answer": answer}
        except Exception as exc:
            return {"answer": "I could not verify an answer from the available evidence.",
                    "error": f"Generation failed: {exc}"}

    def _build_graph(self):
        graph = StateGraph(CRAGState)
        for name, node in [("retrieve", self._retrieve), ("evaluate", self._evaluate),
                           ("rewrite", self._rewrite), ("web", self._web),
                           ("prepare_local", self._prepare_local),
                           ("prepare_web", self._prepare_web), ("generate", self._generate)]:
            graph.add_node(name, node)
        graph.add_edge(START, "retrieve")
        graph.add_edge("retrieve", "evaluate")
        graph.add_conditional_edges("evaluate", lambda state: state["route"],
                                    {"LOCAL": "prepare_local", "WEB": "web",
                                     "REWRITE_AND_WEB": "rewrite"})
        graph.add_edge("rewrite", "web")
        graph.add_edge("web", "prepare_web")
        graph.add_edge("prepare_local", "generate")
        graph.add_edge("prepare_web", "generate")
        graph.add_edge("generate", END)
        return graph.compile()

    def ask(self, question):
        question = question.strip()
        if not question:
            raise ValueError("Question must not be empty")
        started = time.perf_counter()
        result = self.graph.invoke({"question": question})
        return {"question": question, "answer": result["answer"], "route": result["route"],
                "evaluation": result["evaluation"],
                "rewritten_query": result.get("rewritten_query"),
                "sources": [{key: value for key, value in item.items() if key != "text"}
                            for item in result.get("evidence", [])],
                "latency_seconds": round(time.perf_counter() - started, 3),
                "error": result.get("error", "")}
