# Self-Corrective RAG

Backend for the Prompt Engineering mini project. PDF chunks are retrieved from a local ChromaDB collection. An LLM grades whether they answer the question, then a LangGraph workflow chooses local generation, Tavily web search, or query rewriting followed by Tavily search. The result includes the route, grade, sources, and latency. The UI is being developed separately.

## Setup

Use Python 3.11 or newer. From this repository root:

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
```

Place the team's PDF corpus in `data/raw/pdfs/`. The PDFs and generated vector database are ignored by Git. Start [Ollama](https://ollama.com/) and pull the model configured by `OLLAMA_MODEL` (default `llama3.2`). Set `TAVILY_API_KEY` in your environment to enable the two web routes. You can copy `.env.example` as a reference, but this project reads environment variables directly; it does not automatically load `.env`.

```powershell
.venv\Scripts\python -m src.ingestion.chunker
.venv\Scripts\python -m src.retrieval.build_vector_db
.venv\Scripts\python -m scripts.ask "What is corrective RAG?"
```

The command prints JSON suitable for the future UI. For a local question, `route` is `LOCAL`; partial evidence selects `WEB`; irrelevant or absent evidence selects `REWRITE_AND_WEB`. The initial grade thresholds are 0.40 and 0.75. They are configuration defaults, not validated accuracy claims.

## Backend contract

`scripts.ask.make_pipeline().ask(question)` returns `question`, `answer`, `route`, `evaluation` (`score`, `reason`, `evidence_ids`), `rewritten_query`, `sources`, `latency_seconds`, and `error`. Local sources carry PDF filename and page; web sources carry URL and title. Empty or failed web searches return a clear unverified answer. The generator requires citations matching supplied evidence IDs.

For an automated check without API calls:

```powershell
.venv\Scripts\python -m unittest discover -s tests -v
```

The grader's scores require calibration against labeled questions before making claims about routing accuracy. Run the same question set through the baseline RAG and this pipeline for the project evaluation.
