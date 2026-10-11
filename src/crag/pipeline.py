"""Composable CRAG workflow. Dependencies are injected for deterministic testing."""
import json
import re
import time
from datetime import date
from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from src.crag.prompts import GENERATOR, GRADER, QUERY_CHECK, REWRITER
from src.crag.settings import Settings


class CRAGState(TypedDict, total=False):
    question: str
    query_assessment: dict
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
    chunks = []
    for i, text in enumerate(documents, start=1):
        if not text or not text.strip():
            continue
        item_metadata = (metadata[i - 1] or {}) if i <= len(metadata) else {}
        chunks.append({"id": f"L{i}", "text": text,
                       "source": item_metadata.get("source", "unknown"),
                       "page": item_metadata.get("page", 0)})
    return chunks


def format_evidence(items, max_chars=1200):
    return "\n\n".join(
        f"[{item['id']}] {item.get('source') or item.get('title') or ''} "
        f"{item.get('page') or item.get('url') or ''} "
        f"{item.get('published_date') or ''}\n{item['text'][:max_chars]}"
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


def validate_answer_citations(answer, valid_ids):
    """Accept single or grouped evidence IDs, rejecting unknown bracketed citations."""
    cited_ids = set()

    def normalize(match):
        ids = [item.strip() for item in match.group(1).split(",")]
        if not ids or any(not re.fullmatch(r"[LW]\d+", item) or item not in valid_ids
                          for item in ids):
            raise ValueError("answer has an invalid evidence citation")
        cited_ids.update(ids)
        return " ".join(f"[{item}]" for item in ids)

    answer = re.sub(r"\[([^\[\]]+)\]", normalize, answer)
    if not cited_ids:
        raise ValueError("answer lacks valid evidence citations")
    return answer


def choose_route(score, settings):
    if score >= settings.high_threshold:
        return "LOCAL"
    if score >= settings.low_threshold:
        return "WEB"
    return "REWRITE_AND_WEB"


def comparison_subjects(question):
    """Find explicitly named sides of common comparison questions."""
    question = question.strip().rstrip("?. ")
    patterns = (
        r"^How (?:does|do) (.+?) differ from (.+)$",
        r"^How (?:does|do) ([\w.-]+)\s+.+? compared (?:with|to) (.+)$",
    )
    for pattern in patterns:
        match = re.match(pattern, question, re.IGNORECASE)
        if match:
            return tuple(re.sub(r"^(?:(?:the|original)\s+)+", "", value.strip(), flags=re.IGNORECASE)
                         for value in match.groups())
    return ()


def mentions_subject(text, subject):
    """Require an entity name as a separate term, not inside another name."""
    parts = re.findall(r"\w+", subject)
    if not parts:
        return False
    pattern = r"(?<!\w)" + r"[\s-]+".join(map(re.escape, parts)) + r"(?!\w)"
    if re.search(pattern, text, re.IGNORECASE):
        return True
    return len(parts) > 1 and all(len(part) >= 3 and re.search(
        r"(?<!\w)" + re.escape(part) + r"(?!\w)", text, re.IGNORECASE)
        for part in parts)


class CRAGPipeline:
    def __init__(self, retrieve, llm, web, settings=None):
        self.retrieve = retrieve
        self.llm = llm
        self.web = web
        self.settings = settings or Settings.from_env()
        self.graph = self._build_graph()

    def _process_query(self, state):
        checker = getattr(self.llm, "complete_json", self.llm.complete)
        raw = checker(QUERY_CHECK.format(question=state["question"]))
        try:
            assessment = json.loads(raw)
            if not isinstance(assessment.get("needs_clarification"), bool):
                raise ValueError("needs_clarification must be a boolean")
            if not isinstance(assessment.get("clarification_question"), str):
                raise ValueError("clarification_question must be a string")
        except (ValueError, TypeError, AttributeError) as exc:
            raise ValueError(f"Invalid query assessment: {exc}") from exc
        return {"query_assessment": assessment}

    def _clarify(self, state):
        question = state["query_assessment"]["clarification_question"].strip()
        return {"route": "CLARIFY",
                "answer": question[:250] or "Which topic or document do you mean?"}

    def _retrieve(self, state):
        return {"local_evidence": local_chunks(self.retrieve(state["question"], self.settings.top_k))}

    def _evaluate(self, state):
        chunks = state["local_evidence"]
        if not chunks:
            grade = {"score": 0.0, "reason": "No local documents found", "evidence_ids": []}
        else:
            try:
                grader = getattr(self.llm, "complete_json", self.llm.complete)
                raw = grader(GRADER.format(question=state["question"],
                                          evidence=format_evidence(chunks)))
                grade = parse_grade(raw, {item["id"] for item in chunks})
                subjects = comparison_subjects(state["question"])
                if grade["score"] >= self.settings.high_threshold and subjects:
                    missing = [subject for subject in subjects if not any(
                        mentions_subject(item["text"], subject) for item in chunks)]
                    if missing:
                        grade = {**grade, "score": self.settings.low_threshold,
                                 "reason": "Retrieved excerpts do not mention every compared subject: "
                                           + ", ".join(missing)}
                # A high grade without any cited excerpt is not trusted.
                if grade["score"] >= self.settings.high_threshold and not grade["evidence_ids"]:
                    grade = {**grade, "score": self.settings.low_threshold}
            except Exception as exc:
                grade = {"score": 0.0, "reason": f"Grader failed: {exc}", "evidence_ids": []}
        return {"evaluation": grade, "route": choose_route(grade["score"], self.settings)}

    def _rewrite(self, state):
        try:
            query = self.llm.complete(REWRITER.format(
                question=state["question"], current_date=date.today().isoformat()
            )).strip().strip('"')
            if not query or "\n" in query or len(query) > 300:
                query = state["question"]
            return {"rewritten_query": query}
        except Exception:
            return {"rewritten_query": state["question"]}

    def _web(self, state):
        from src.crag.web import refine_web
        query = state.get("rewritten_query") or state["question"]
        subjects = comparison_subjects(state["question"])
        if subjects:
            query = f"{subjects[0]} versus {subjects[1]} differences"
        try:
            evidence = refine_web(self.web.search(query))
            if subjects:
                evidence = [item for item in evidence if any(
                    mentions_subject(item.get("title", "") + " " + item["text"], subject)
                    for subject in subjects)]
                available = state.get("local_evidence", []) + evidence
                missing = [subject for subject in subjects if not any(
                    mentions_subject(item["text"], subject) for item in available)]
                if missing:
                    return {"web_evidence": [], "error": "Web search lacked evidence about: "
                            + ", ".join(missing)}
            return {"web_evidence": evidence,
                    "error": "" if evidence else "Web search returned no usable evidence"}
        except Exception as exc:
            return {"web_evidence": [], "error": f"Web search failed: {exc}"}

    def _prepare_local(self, state):
        allowed = set(state["evaluation"]["evidence_ids"])
        return {"evidence": [item for item in state["local_evidence"] if item["id"] in allowed]}

    def _prepare_web(self, state):
        evidence = state.get("web_evidence", [])
        if not evidence:
            return {"evidence": []}
        if state["route"] == "WEB":
            subjects = comparison_subjects(state["question"])
            if subjects:
                local = [item for item in state.get("local_evidence", [])
                         if any(mentions_subject(item["text"], subject) for subject in subjects)]
                evidence = local[:2] + evidence
        return {"evidence": evidence}

    def _generate(self, state):
        evidence = state.get("evidence", [])
        if not evidence:
            return {"answer": "I could not verify an answer from the available evidence."}
        try:
            prompt = GENERATOR.format(question=state["question"],
                                      evidence=format_evidence(evidence),
                                      current_date=date.today().isoformat())
            def generate(text):
                if hasattr(self.llm, "complete_json"):
                    value = json.loads(self.llm.complete_json(text))["answer"]
                    if not isinstance(value, str):
                        raise ValueError("answer must be a string")
                    return value
                return self.llm.complete(text)

            answer = generate(prompt)
            if not answer:
                raise ValueError("empty response")
            valid_ids = {item["id"] for item in evidence}
            try:
                answer = validate_answer_citations(answer, valid_ids)
            except ValueError:
                allowed = ", ".join(sorted(valid_ids))
                retry = (prompt + "\nYour previous answer had an invalid or missing citation. "
                         f"Answer again, citing only these evidence IDs: {allowed}. "
                         "Do not add claims that the cited evidence cannot support.")
                answer = validate_answer_citations(generate(retry), valid_ids)
            return {"answer": answer}
        except Exception as exc:
            return {"answer": "I could not verify an answer from the available evidence.",
                    "error": f"Generation failed: {exc}"}

    def _build_graph(self):
        graph = StateGraph(CRAGState)
        for name, node in [("process_query", self._process_query),
                           ("clarify", self._clarify),
                           ("retrieve", self._retrieve), ("evaluate", self._evaluate),
                           ("rewrite", self._rewrite), ("web", self._web),
                           ("prepare_local", self._prepare_local),
                           ("prepare_web", self._prepare_web), ("generate", self._generate)]:
            graph.add_node(name, node)
        graph.add_edge(START, "process_query")
        graph.add_conditional_edges(
            "process_query",
            lambda state: "clarify" if state["query_assessment"]["needs_clarification"] else "retrieve",
            {"clarify": "clarify", "retrieve": "retrieve"},
        )
        graph.add_edge("clarify", END)
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
                "evaluation": result.get("evaluation"),
                "rewritten_query": result.get("rewritten_query"),
                "retrieved_documents": result.get("local_evidence", []),
                "sources": [{key: value for key, value in item.items() if key != "text"}
                            for item in result.get("evidence", [])],
                "latency_seconds": round(time.perf_counter() - started, 3),
                "error": result.get("error", "")}
