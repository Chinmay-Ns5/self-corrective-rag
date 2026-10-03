# Self-Corrective RAG

Integrated Prompt Engineering mini project combining the ingestion, CRAG backend, and Streamlit UI branches. PDF chunks are retrieved from a local ChromaDB collection. An LLM grades whether they answer the question, then a LangGraph workflow chooses local generation, Tavily web search, or query rewriting followed by Tavily search. The UI displays the answer, route, grade, sources, latency, and optional retrieved chunks.

## Setup

Use Python 3.11, the version used for integration checks. From this repository root:

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
```

Place the team's PDF corpus in `data/raw/pdfs/`. The PDFs, generated chunks, vector database, and `.env` are ignored by Git. The first index build downloads the `all-MiniLM-L6-v2` embedding model; later offline runs can set `HF_HUB_OFFLINE=1` once that model is cached.

Create `.env` from `.env.example` if you do not already have one:

```powershell
Copy-Item .env.example .env
```

Start [Ollama](https://ollama.com/) and install the default model with `ollama pull llama3.2`, or set `OLLAMA_MODEL` in `.env` to an installed model listed by `ollama list`. Add your `TAVILY_API_KEY` to `.env` to enable web fallback. Both the CLI and UI read this file on each run; nonempty process environment variables take precedence. The original baseline also uses the same model configuration.

```powershell
.venv\Scripts\python -m scripts.ingest
.venv\Scripts\python -m scripts.ask "What is corrective RAG?"
```

The CLI writes one JSON result to stdout and startup logs to stderr. For a local question, `route` is `LOCAL`; partial evidence selects `WEB`; irrelevant or absent evidence selects `REWRITE_AND_WEB`. The initial grade thresholds are 0.40 and 0.75. They are configuration defaults, not validated accuracy claims.

The form has no previous document or conversation context. Questions such as “What is this about?” return a `CLARIFY` request for a topic or document name immediately. For example, ask “What is dense passage retrieval?” to search the PDF corpus. Web routes require `TAVILY_API_KEY`; add it to `.env` and click **Reload backend**.

Ollama requests use a bounded 4,096-token context and at most 512 generated tokens so the model does not allocate a large default context on a laptop.

## Streamlit UI

Start the UI from the repository root with:

```powershell
.venv\Scripts\python -m streamlit run app.py
```

The question form calls the shared `src.crag.service.make_pipeline` factory and `pipeline.ask(question)`. The embedding model and database are cached across UI reruns. After rebuilding the index, click **Reload backend**. Missing database/model and API failures are shown in the app. Without a Tavily key, local answers still work; questions requiring web search return an explicit failure rather than an unsupported answer.

## Backend contract

`src.crag.service.make_pipeline().ask(question)` returns `question`, `answer`, `route`, `evaluation` (`score`, `reason`, `evidence_ids`), `rewritten_query`, `retrieved_documents`, `sources`, `latency_seconds`, and `error`. Local sources carry PDF filename and page; web sources carry URL and title. Empty or failed web searches return a clear unverified answer. The generator requires citations matching supplied evidence IDs. Citation validation checks IDs; groundedness still requires evaluation.

For an automated check without API calls:

```powershell
.venv\Scripts\python -m unittest discover -s tests -v
```

Tests exercise the real graph and Streamlit form with fake LLM/search clients, including every route, source display, cached resource reuse, empty input, and setup/API failures. Live model and Tavily behavior require separate checks with the configured services.

The grader's scores require calibration against labeled questions before making claims about routing accuracy. Run the same question set through the baseline RAG (`python -m src.retrieval.baseline_rag`) and this pipeline for the project evaluation.
