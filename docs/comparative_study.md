# Comparative RAG study: CLI implementation

This is the next-review experimental track. It does not change the Streamlit
application or replace the earlier production CRAG results. The aim is a
controlled comparative study, not a claim of a new RAG algorithm.

The completed development pilot and observed failures are recorded in
[the pilot findings](../evaluation/study/pilot_findings_20261015.md).
The later [paired CRAG/graph follow-up](../evaluation/study/graph_followup_findings_20261021.md)
repeats both systems: it found equal repeat mean latency and no reviewed quality
advantage for the graph in that development workload.
The [research follow-up](next_review_research_20261015.md) maps the review notes
to verified prior work and records the current graph's limitations.

## Variants and controls

| ID | Retriever | Routing | Final processing |
| --- | --- | --- | --- |
| S1 | Dense Chroma retrieval | Fixed local, shared clarification first | Shared cited generator |
| S2 | Dense + BM25-style lexical RRF, balanced named-paper slots | Fixed local, shared clarification first | Shared cited generator |
| S3 | Exactly the same retriever as S2 | Existing CRAG grader, rewrite and web routes | Shared cited generator |
| S4 | S2 candidates + keyword/document graph and lazy semantic relationships | Same CRAG routing | Shared cited generator |
| S5 | Exactly the same initial retriever/routing as S3 | Same CRAG routing | Sentence-level NLI verification, bounded local retrieval and repair/drop |

All use the same configured Ollama generator, generation prompt, temperature,
4,096-token model context, and initial top-k (default five). All receive the
same clarification step, including the fixed-local baselines. Consequently,
S1 is a **controlled dense baseline**, not a reproduction of Lewis RAG or the
historical project's earlier baseline prompt. Absolute comparisons with the
2026-10-11 accuracy figures are not valid.

S5 can retrieve two extra local chunks for each of at most two repairs. This
extra evidence and compute are part of its treatment and appear in the trace.
They are not silently counted as the same total evidence budget as S3.

The original UI retriever remains in `src/retrieval/hybrid.py`. The experimental
balanced retriever is separate in `src/study/retrieval.py`. It matches contiguous
filename-title phrases, ignores a shorter title contained within a longer matched
title, and allocates up to two slots per named paper in round-robin order.
It prefers distinct pages for those slots. Unnamed remaining context comes from
the fused ranking. This is a heuristic, not semantic entity resolution.

## Run from the repository root

The examples use the existing workspace venv. Replace its relative path if using
a separate project `.venv`.

```powershell
# No generation, no Ollama requirement, no web calls.
# S4 uses only keywords and previously cached semantic edges in this mode.
& '..\..\.venv\Scripts\python.exe' -m scripts.compare_rag --question "Compare BERT and Sentence-BERT" --systems s1 s2 s4 --retrieval-only

# Full single-question comparison. Ollama must be running.
# S5 requires the small NLI model to have been downloaded once.
& '..\..\.venv\Scripts\python.exe' -m scripts.compare_rag --question "Compare BERT and Sentence-BERT" --systems s1 s2 s3 s4 s5

# Six DEVELOPMENT questions, including first/repeat graph queries; web OFF.
& '..\..\.venv\Scripts\python.exe' -m scripts.compare_rag --dataset evaluation/study/development.jsonl --repeat-graph --output evaluation/results/my_study.jsonl

# Explicitly authorize at most TWO Tavily searches; exact queries share a cache.
& '..\..\.venv\Scripts\python.exe' -m scripts.compare_rag --dataset evaluation/study/development.jsonl --web-mode live --web-budget 2 --repeat-graph --output evaluation/results/my_web_study.jsonl
```

Use `--ids C01,C03` for selected questions. `--systems` accepts any subset.
For a fair repeat comparison, use `--systems s3 s4 --repeat-all --warmup`.
This runs both first trials before either repeat, reverses their order for the
repeat round, and alternates the starting system across questions. One neutral
warmup generation is counted separately in the manifest and excluded from trial
timings. Shared Ollama prompt caches can still affect both systems; first trials
do not imply a cold model. `--repeat-graph` retains the original S4-only repeats.
`--resume` resumes an existing output only when its saved configuration matches;
completed question/system/phase rows are skipped. Existing review annotations
are preserved. Interrupt at a completed row when possible: records are flushed
after every variant, but a mid-request interruption can omit the current row.

The model loader uses the existing cached MiniLM encoder on CPU. NLI also uses
CPU with two PyTorch threads, leaving GPU memory to Ollama. S5 loads
`cross-encoder/nli-deberta-v3-small` from `data/study_models/`. To allow its first
download, add `--download-models` to a full comparison including S5; later runs
are local-only for that classifier. The default generator is read from `.env`;
the manifest records its name. This session prepared the NLI cache already.

## Web and cache accounting

- Web is disabled by default. With web disabled, a required fallback returns
  an explicit failure/abstention; this is not evidence of an API outage.
- `--web-mode cache` replays existing evidence without a new search.
- `--web-mode live --web-budget N` permits at most N search-client network
  attempts in that output run, including failed searches. Resume subtracts
  attempts recorded in completed rows. An interrupted, unrecorded network call
  may already have consumed a provider credit; check before resuming such a call.
- Exact normalized queries use a date-scoped cache shared across variants.
  Different rewrites can miss it and consume separate attempts.
- `web_network_calls` counts Tavily search attempts, not every HTTP request:
  the existing search client may also read official release pages for enrichment.
- Graph triples are cached by model name, extraction schema, corpus content
  fingerprint, source document, and selected chunk IDs. Corpus changes cannot
  reuse an old graph cache. A new region of the same document can be expanded.
- Use a new `--cache-dir` for a fresh graph-cache experiment. A first query is
  not necessarily cold if cache files already exist. Inspect `expansions` and
  `graph_llm_calls` rather than assuming its phase label proves a cold cache.

Local models, graph/web caches, and raw result JSONL are ignored by Git. Do not
share `.env`. Runs write a configuration manifest plus a review CSV alongside
the JSONL, or emit JSONL directly to stdout when no `--output` is supplied.

## S4: what the two layers actually do

1. **Cheap keyword/document graph:** compute document-level TF-IDF weights and
   retain the top 40 terms per document. A keyword maps to one or more documents;
   its degree identifies shared versus document-specific keywords. Query terms
   identify jump documents and shared-keyword neighbors. This indexing makes
   zero LLM calls.
2. **Lazy semantic graph:** on a question, expand at most two seed documents,
   using at most two retrieved chunks per selected region. The same local LLM
   extracts at most two subject/relation/object triples per region. Short entity
   and quote limits keep the JSON within the generator's 512-token response cap.
3. **Provenance checks:** accept only known chunk IDs, an exact supporting quote
   from the selected text, and entities present in that quote. These structural
   checks do not prove that the relation was interpreted correctly.
4. **Retrieval:** one-hop entity links boost original evidence chunks; document
   keyword matches provide small additional boosts. The fused ranking and named
   paper allocation select the final original chunks. Extracted relation text
   is never presented as an independently trusted source.

This uses Python dictionaries/sets and JSON cache files, not Neo4j, community
summaries, a graph neural network, PageRank, or HNSW. The graph boosts and keyword
limits are fixed development heuristics, not calibrated mathematical results.
It is a minimal graph-assisted CRAG experiment, **not** a reproduction of
Microsoft GraphRAG, LazyGraphRAG, RAPTOR, HippoRAG, or LightRAG.

## S5: what verification means

The classifier is the existing
[NLI DeBERTa-v3-small cross-encoder](https://huggingface.co/cross-encoder/nli-deberta-v3-small),
trained on SNLI and MultiNLI. The code reads the checkpoint's label mapping and
applies softmax to contradiction/entailment/neutral logits.

- Split the answer into sentences and associate inline citations. Sentences
  are proxies for atomic claims; compound statements and pronouns remain hard.
- A sentence is accepted if at least one cited chunk entails it above the
  default 0.70 threshold and entailment is its largest label probability.
  Unknown citations and truncated input cannot count as supported.
- Check each cited chunk separately. This is conservative when a claim needs
  multiple chunks jointly; no unsupported cross-source inference is assumed.
- An unsupported sentence can trigger local retrieval for that sentence and one
  generation repair. There are at most two such attempts per answer. The repair
  receives valid evidence IDs and is checked again. Failed repairs are dropped.
- Preserve before/after sentence judgments, raw classifier scores, repair and
  removal counts. An empty retained answer is an abstention with zero claims,
  not a 100% grounded answer.

NLI is an imperfect, uncalibrated estimator, especially for technical PDF text.
A strict checker can reject correct claims and reduce answer usefulness. Report
independently inspected support and correctness alongside NLI estimates,
retained coverage, dropped statements, latency and extra calls. Do not treat the
classifier as ground truth or assume S5 beats S3.

## Evaluation protocol

`evaluation/study/development.jsonl` is a six-case smoke/development set. The
implementer knows its contents and its facts overlap earlier work. It is not
a held-out benchmark and must not be used for headline generalization claims.

For a report-quality comparison:

1. Freeze the code, corpus, generator, initial context budget and web policy.
2. Collect separate calibration and evaluation questions across single-paper,
   multi-paper, outside-corpus, and unclear-subject categories. An independent
   teammate should author/review the evaluation set if possible.
3. Label context sufficiency by inspecting the actual retrieved chunks. Paper
   name matches alone are insufficient. Explicit passage-ID coverage is only
   a non-exhaustive proxy; check alternative relevant passages manually.
4. Label correctness and citation-level support independently of the NLI model.
   For this study, correctness means a complete, factually correct reference
   answer or appropriate clarification. A true answer with an unrelated citation
   can be factually correct yet ungrounded; the support judgment must expose that.
   This differs from earlier strict source-backed correctness labels, so scores
   from the old and new studies must not be pooled.
   Record supported and total factual statements, including omissions or
   abstentions in the correctness decision.
5. Label `gold_route` using the evidence that each system actually retrieved:
   LOCAL, WEB (related but incomplete), REWRITE_AND_WEB (irrelevant), or CLARIFY.
   Question-category expectations alone are not exact routing labels.
6. Fill `.review.csv`. Keep blank cells for unreviewed cases. Never turn blanks
   into zeroes or derive correctness from reference-word overlap.
7. Summarize with:

```powershell
& '..\..\.venv\Scripts\python.exe' -m scripts.summarize_study evaluation/results/my_study.jsonl --review evaluation/results/my_study.review.csv --output evaluation/results/my_study.summary.json
```

The summary keeps reviewed correctness/context labels separate from NLI
estimates and automatic preselected-passage coverage. It reports errors, mean
and median latency, LLM calls, extraction calls, and web searches/cache hits.
Report per-category results and actual denominators; aggregate numbers can hide
the local-only baselines' expected abstentions on web-required questions.

### Threshold calibration

Fill `gold_route` in separate calibration and evaluation CSVs, then run:

```powershell
& '..\..\.venv\Scripts\python.exe' -m scripts.calibrate_routes --calibration evaluation/results/calibration.review.csv --evaluation evaluation/results/heldout.review.csv --system s3 --output evaluation/results/thresholds.json
```

The tool requires all three graded route classes in calibration, rejects shared
question IDs or identical normalized question text across splits, selects
thresholds using calibration rows only, and reports before/after accuracy on
the untouched evaluation rows. It never edits `.env`. Apply a recommended pair
to a subsequent comparison using `--low-threshold` and `--high-threshold`.
The before/after figures are offline projections on saved final scores. Existing
evidence guards can cap scores using configured thresholds, so a fresh runtime
comparison is required to validate any recommended pair.
No routing or NLI threshold has been validated merely by implementing this tool.

### Hypotheses and missing baselines

- H1: S4 improves multi-paper correctness/context over S3. Requires new questions
  and reviewed answers; the existence of a graph is not evidence of improvement.
- H2: lazy construction lowers indexing cost. The implementation has zero
  up-front extraction calls, but a claim relative to full graph construction
  requires an eager baseline using the same extraction policy/corpus. We have
  not run a full Microsoft GraphRAG indexing baseline.
- H3: S5 improves inspected claim support at a measured latency/coverage cost.
  A higher support percentage after deleting answers does not establish H3.

First/repeat graph latency also reflects Ollama model/prompt caches and other
warm-state effects. Extraction-call differences are directly observable; one
timing pair does not isolate graph caching or establish statistical significance.

## Verified related work

- [GraphRAG, Edge et al.](https://arxiv.org/abs/2404.16130): entity graph and
  community summaries for global sensemaking; our code omits these summaries.
- [LazyGraphRAG, Microsoft Research](https://www.microsoft.com/en-us/research/blog/lazygraphrag-setting-a-new-standard-for-quality-and-cost/):
  established graph retrieval without prior source summarization. Lazy graph
  processing is not an original idea from this team.
- [RAPTOR, Sarthi et al.](https://arxiv.org/abs/2401.18059): recursive embedding,
  clustering and summarization tree; our keyword layer is not a RAPTOR tree.
- [LightRAG, Guo et al.](https://arxiv.org/abs/2410.05779): related graph-enhanced
  retrieval work, not implemented as a faithful baseline here.
- Existing main CRAG/RAG/Self-RAG references and the prior retrieval refinement
  remain in [the literature brief](literature_review_and_contribution.md).

Grounded conversational memory, KV-cache swapping, live sentence-by-sentence
generation, and model fine-tuning remain outside this implementation.
