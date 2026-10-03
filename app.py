"""Streamlit interface connected to the shared CRAG backend."""

from typing import Any

import streamlit as st

from src.crag.service import make_pipeline
from src.crag.pipeline import clarification_result
from src.crag.settings import Settings


ROUTE_DETAILS = {
    "LOCAL": "Answered using local knowledge.",
    "WEB": "Local retrieval was insufficient or ambiguous, so web evidence was used.",
    "REWRITE_AND_WEB": "The query was rewritten before web search.",
}


@st.cache_resource(show_spinner=False)
def get_pipeline(settings: Settings):
    """Reuse the embedding model and database across Streamlit reruns."""
    return make_pipeline(settings)


def run_crag_query(question: str) -> dict[str, Any]:
    clarification = clarification_result(question)
    if clarification:
        return clarification
    return get_pipeline(Settings.from_env()).ask(question)


def render_sidebar() -> dict[str, bool]:
    with st.sidebar:
        st.title("CRAG Controls")
        controls = {
            "retrieval": st.checkbox("Show retrieval analysis", value=True),
            "sources": st.checkbox("Show sources", value=True),
            "debug": st.checkbox("Show debug / evidence information", value=False),
            "latency": st.checkbox("Show latency", value=True),
        }
        st.divider()
        st.subheader("System Status")
        settings = Settings.from_env()
        st.info(f"Model: {settings.ollama_model}")
        st.caption("Backend connected. Resources load on your first question.")
        if settings.tavily_key:
            st.caption("Web fallback is configured.")
        else:
            st.caption("Set TAVILY_API_KEY in .env to enable web fallback.")
        if st.button("Reload backend"):
            get_pipeline.clear()
            st.session_state.pop("crag_result", None)
            st.session_state.pop("crag_error", None)
            st.success("The backend will reload on your next question.")
        return controls


def render_query_input() -> None:
    st.subheader("Ask the CRAG system")
    with st.form("crag_query_form", clear_on_submit=False):
        question = st.text_area(
            "Question",
            placeholder="Ask a question about the knowledge base...",
            height=110,
            label_visibility="collapsed",
        )
        submitted = st.form_submit_button("Run CRAG", type="primary", use_container_width=True)

    if submitted:
        if not question.strip():
            st.warning("Enter a question before running CRAG.")
            return
        try:
            with st.spinner("Running CRAG pipeline: retrieval, evaluation, routing, and generation..."):
                st.session_state["crag_result"] = run_crag_query(question.strip())
            st.session_state.pop("crag_error", None)
        except Exception as exc:
            st.session_state.pop("crag_result", None)
            st.session_state["crag_error"] = str(exc)


def render_error(error: str) -> None:
    st.error("The query could not be completed.")
    st.info(error)
    if "TAVILY_API_KEY" in error:
        st.caption("Add TAVILY_API_KEY to .env and click Reload backend to enable web search.")
    if "vector" in error.lower() or "collection" in error.lower():
        st.caption(
            "Add PDFs under data/raw/pdfs/ and run python -m scripts.ingest, "
            "then click Reload backend."
        )


def render_route(route: str) -> None:
    detail = ROUTE_DETAILS.get(route, "The backend returned an unrecognized route.")
    st.markdown("#### Selected route")
    st.markdown(f"**{route}**  \n{detail}")
    stages = ["Question", "Local retrieval", "Relevance evaluation", route, "Evidence", "Final answer"]
    st.caption("  →  ".join(stages))


def render_retrieval_analysis(result: dict[str, Any], controls: dict[str, bool]) -> None:
    evaluation = result.get("evaluation") or {}
    route = str(result.get("route") or "Unknown")
    if controls["retrieval"]:
        st.subheader("Retrieval analysis")
        left, right = st.columns(2)
        score = evaluation.get("score")
        with left:
            st.metric("Local relevance / evaluation score", f"{score:.2f}" if isinstance(score, (int, float)) else "Unavailable")
            st.caption("LLM relevance/answerability score used by CRAG routing; not an embedding similarity score.")
        with right:
            st.metric("Selected route", route)
        reason = evaluation.get("reason")
        if reason:
            st.write("**Evaluation reason**")
            st.write(str(reason))
        evidence_ids = evaluation.get("evidence_ids") or []
        st.write("**Evidence IDs**")
        st.write(", ".join(map(str, evidence_ids)) if evidence_ids else "No evidence IDs were returned.")
        render_route(route)

    if controls["debug"]:
        with st.expander("Debug / evidence information", expanded=False):
            evidence_ids = evaluation.get("evidence_ids") or []
            st.write("**Evidence IDs**")
            st.write(", ".join(map(str, evidence_ids)) if evidence_ids else "No evidence IDs were returned.")
            for item in result.get("retrieved_documents", []):
                st.write(f"[{item['id']}] {item['source']} · Page {item['page']}")
                st.text(item["text"])


def render_rewritten_query(result: dict[str, Any]) -> None:
    rewritten = result.get("rewritten_query")
    if isinstance(rewritten, str) and rewritten.strip():
        st.subheader("Rewritten Search Query")
        st.caption("The query was rewritten before web search.")
        st.info(rewritten.strip())


def render_sources(result: dict[str, Any]) -> None:
    sources = result.get("sources") or []
    local_sources = [item for item in sources if "source" in item or "page" in item]
    web_sources = [item for item in sources if "url" in item or "title" in item]

    st.subheader("Sources")
    if local_sources:
        st.markdown("**Local Sources**")
        for source in local_sources:
            label = source.get("source") or "Local document"
            details = []
            if source.get("page") is not None:
                details.append(f"Page {source['page']}")
            if source.get("id"):
                details.append(f"ID: {source['id']}")
            st.markdown(f"- **{label}**" + (f" · {' · '.join(details)}" if details else ""))
    if web_sources:
        st.markdown("**Web Sources**")
        for source in web_sources:
            title = source.get("title") or source.get("url") or "Web source"
            url = source.get("url")
            rendered_title = f"[{title}]({url})" if url else str(title)
            details = [rendered_title]
            if source.get("id"):
                details.append(f"ID: {source['id']}")
            if source.get("score") is not None:
                details.append(f"Score: {source['score']}")
            st.markdown("- " + " · ".join(details))
    if not local_sources and not web_sources:
        st.caption("No sources were returned.")


def render_result(result: dict[str, Any], controls: dict[str, bool]) -> None:
    st.divider()
    question = result.get("question")
    if question:
        st.markdown("#### Question")
        st.write(question)

    if result.get("route") == "CLARIFY":
        st.info(result["answer"])
        return

    render_retrieval_analysis(result, controls)
    render_rewritten_query(result)

    st.subheader("Final answer")
    st.markdown(result.get("answer") or "No answer was returned.")

    if controls["sources"]:
        render_sources(result)
    if controls["latency"] and result.get("latency_seconds") is not None:
        st.caption(f"Latency · {result['latency_seconds']:.2f} seconds")
    if result.get("error"):
        render_error(str(result["error"]))


def main() -> None:
    st.set_page_config(page_title="Self-Corrective RAG", page_icon="↻", layout="wide")
    try:
        controls = render_sidebar()
    except ValueError as exc:
        st.error(f"Invalid configuration: {exc}")
        st.caption("Check the model and routing settings in .env, then rerun the app.")
        st.stop()

    st.title("Self-Corrective RAG")
    st.markdown(
        "Retrieval-Augmented Generation with adaptive routing, relevance grading, "
        "and corrective web fallback."
    )
    st.caption("Academic project dashboard · Local retrieval with corrective web fallback")
    render_query_input()

    if st.session_state.get("crag_error"):
        render_error(st.session_state["crag_error"])
    elif st.session_state.get("crag_result"):
        render_result(st.session_state["crag_result"], controls)


if __name__ == "__main__":
    main()
