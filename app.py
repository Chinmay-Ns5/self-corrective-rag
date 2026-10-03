"""Streamlit interface for the Self-Corrective RAG project.

The query adapter is intentionally isolated. Connect it to
``pipeline.ask(question)`` when the live backend integration is ready.
"""

from typing import Any

import streamlit as st


ROUTE_DETAILS = {
    "LOCAL": "Answered using local knowledge.",
    "WEB": "Local retrieval was insufficient or ambiguous, so web evidence was used.",
    "REWRITE_AND_WEB": "The query was rewritten before web search.",
}


def run_crag_query(question: str) -> dict[str, Any]:
    """Isolated adapter for the eventual ``pipeline.ask(question)`` call."""
    raise RuntimeError(
        "Live CRAG backend integration is not connected yet. "
        "This UI is ready to connect to the Python CRAG pipeline."
    )


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
        st.info("UI ready · Live backend not connected")
        st.caption(
            "Connect the query adapter to the CRAG pipeline to enable live questions."
        )
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
    if "vector" in error.lower() or "collection" in error.lower():
        st.caption(
            "If the live backend is connected, add PDFs under data/raw/pdfs/ "
            "and build the chunks and vector database before querying."
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
            st.caption("Retrieved chunk text is not currently exposed by the backend.")


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
    controls = render_sidebar()

    st.title("Self-Corrective RAG")
    st.markdown(
        "Retrieval-Augmented Generation with adaptive routing, relevance grading, "
        "and corrective web fallback."
    )
    st.caption("Academic project dashboard · UI integration in progress")
    render_query_input()

    if st.session_state.get("crag_error"):
        render_error(st.session_state["crag_error"])
    elif st.session_state.get("crag_result"):
        render_result(st.session_state["crag_result"], controls)


if __name__ == "__main__":
    main()
