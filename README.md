# Self-Corrective RAG

Integrated Prompt Engineering mini project combining the ingestion, CRAG backend, and Streamlit UI branches. PDF chunks are retrieved with dense ChromaDB search and lexical matching over the indexed collection. An LLM grades whether they answer the question, then a LangGraph workflow chooses local generation, Tavily web search, or query rewriting followed by Tavily search. The UI displays the answer, route, grade, sources, latency, and optional retrieved chunks.

## Literature review and presentation

The next-review CLI comparison (S1 dense, S2 hybrid, S3 CRAG, S4 graph-assisted
CRAG, S5 CRAG with NLI verification) is documented in
[docs/comparative_study.md](docs/comparative_study.md). It includes controlled
prompts, bounded web usage, graph-cache accounting, review templates and a
disjoint threshold-calibration workflow. These experimental variants do not
change the Streamlit application.

The [paired graph comparison](evaluation/study/graph_followup_findings_20261021.md)
completed 48 trials: both systems averaged about 9.61 seconds on repeat, while
the experimental graph had higher first-trial cost and no reviewed quality gain.

The [research follow-up](docs/next_review_research_20261015.md) checks the graph,
verification, confidence-gate, memory and caching ideas against primary sources
and identifies what can be evaluated without generative LLM calls.

For a first-review PPT, give the slide-writing LLM
[docs/literature_review_and_contribution.md](docs/literature_review_and_contribution.md).
It is a standalone brief with paper summaries, implementation differences,
the contribution's algorithmic detail, checked results and limitations,
a slide outline, and a ready-to-use drafting prompt.

The four main papers are:

1. Yan et al., **Corrective Retrieval Augmented Generation** (2024) —
   [paper](https://arxiv.org/abs/2401.15884): main architectural inspiration.
2. Lewis et al., **Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks** (2020) —
   [paper](https://arxiv.org/abs/2005.11401): RAG foundation.
3. Gao et al., **Retrieval-Augmented Generation for Large Language Models: A Survey** (2023 preprint) —
   [paper](https://arxiv.org/abs/2312.10997): literature and design background.
4. Asai et al., **Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection** (2023 preprint) —
   [paper](https://arxiv.org/abs/2310.11511): related adaptive retrieval and critique work.

Our contribution is an evaluated CRAG adaptation and passage-selection refinement;
research novelty has not been established. The adopted refinement selects the
highest-scoring chunk on each reserved body page instead of its first chunk.
On 15 fresh questions, inspected sufficient context improved from 7/15 to 9/15
at comparable retrieval time. These are retrieval-context judgments by the
assistant, not generated-answer accuracy or independent human review. Details:
[fresh retrieval evaluation](evaluation/retrieval_fresh_findings_20261014.md).

The brief lists the other six papers in the PDF corpus and additional prior work
consulted during the novelty check. Corpus inclusion does not imply implementing
each paper's method.

## Setup

Use Python 3.11, the version used for integration checks. From this repository root:

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
```

PyTorch and Torchvision are pinned as a compatible pair. Torchvision is included
because Transformers processor discovery can import image-processing modules
during model setup. After changing these packages, restart the Streamlit process
or Python kernel so it reloads the installed modules.

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

The form has no previous document or conversation context. Before retrieval, the LLM checks whether a question identifies a clear subject. If it needs an unnamed topic or document, the app returns `CLARIFY` with a follow-up question. This adds one model call per question. For example, ask “What is dense passage retrieval?” to search the PDF corpus. Web routes require `TAVILY_API_KEY`; add it to `.env` and click **Reload backend**.

Ollama requests use a bounded 4,096-token context and at most 512 generated tokens so the model does not allocate a large default context on a laptop.

## Streamlit UI

Start the UI from the repository root with:

```powershell
.\run.ps1
```

The launcher uses the project's `.venv` if present, or the existing `PE/.venv` workspace environment. It runs from the project directory, so it also works when invoked from elsewhere. When a local index exists, it uses the cached embedding model offline to avoid a model-hub check on each run. Start Ollama before launching the UI; press Ctrl+C to stop Streamlit.

The question form calls the shared `src.crag.service.make_pipeline` factory and `pipeline.ask(question)`. The embedding model and database are cached across UI reruns. After rebuilding the index, click **Reload backend**. Missing database/model and API failures are shown in the app. Without a Tavily key, local answers still work; questions requiring web search return an explicit failure rather than an unsupported answer.

Use **Add a PDF** in the sidebar to upload a text-based PDF of up to 20 MB. The app saves it under `data/raw/pdfs/`, adds its chunks to the existing Chroma collection, and refreshes the cached backend for the next question. An image-only scanned PDF needs OCR before upload. A file with the same name as an existing corpus PDF is rejected to avoid silently replacing its evidence. Uploaded PDFs are local and ignored by Git; a later full `python -m scripts.ingest` includes them in the rebuilt index.

## Backend contract

`src.crag.service.make_pipeline().ask(question)` returns `question`, `answer`, `route`, `evaluation` (`score`, `reason`, `evidence_ids`), `rewritten_query`, `retrieved_documents`, `sources`, `latency_seconds`, and `error`. Local sources carry PDF filename and page; web sources carry URL and title. Empty or failed web searches return a clear unverified answer. The generator requires citations matching supplied evidence IDs. Citation validation checks IDs; groundedness still requires evaluation.

For an automated check without API calls:

```powershell
.venv\Scripts\python -m unittest discover -s tests -v
```

Tests exercise the real graph and Streamlit form with fake LLM/search clients, including every route, source display, cached resource reuse, empty input, and setup/API failures. Live model and Tavily behavior require separate checks with the configured services.

The grader's scores require calibration against labeled questions before making claims about routing accuracy. Run the same question set through the baseline RAG (`python -m src.retrieval.baseline_rag`) and this pipeline for the project evaluation.

For a repeatable baseline-versus-CRAG run, use `python -m scripts.evaluate --offline` with Ollama and Tavily configured. The 50-question dataset, review rubric, and output instructions are in [evaluation/README.md](evaluation/README.md). Reviewed correctness and groundedness are kept separate from automatic routing and latency measurements.

The completed 50-question run, its assistant-reviewed results, and limitations are recorded in [evaluation/final_findings_20261011.md](evaluation/final_findings_20261011.md). The earlier pre-fix snapshot remains in [evaluation/initial_findings.md](evaluation/initial_findings.md).
