# Comparative RAG development pilot — 15 October 2026

## Scope and reproducibility

This is a six-question development pilot, reviewed by the coding assistant.
It is not an independently labeled or held-out benchmark. The five variants
and commands are defined in [the protocol](../../docs/comparative_study.md).
Questions are in [development.jsonl](development.jsonl), annotations in
[pilot_20261015.review.csv](pilot_20261015.review.csv), and aggregates in
[pilot_20261015.summary.json](pilot_20261015.summary.json).

The corpus contained 10 PDFs / 1,158 indexed chunks. All variants used the
configured Qwen2.5:7b Ollama generator, 4,096-token context and five initial
chunks. MiniLM retrieval and NLI ran on CPU; Ollama used the NVIDIA GPU with
partial CPU offload. NLI used `cross-encoder/nli-deberta-v3-small`, revision
`fa2804872c3b4bd748f38c0185cc85775361e735`, at the uncalibrated 0.70 threshold.

The main run produced 36 rows: six questions for each system, plus six S4
repeats. It made 105 generative LLM calls and exactly **two Tavily searches**.
Variants shared cached web responses. Setup took 24.03 seconds and is excluded
from per-question timings. Raw traces and manifests remain locally under
`evaluation/results/study_pilot_20261015.*`, ignored by Git.

## Observations

| Variant | Factually correct / reviewed | Mean seconds | LLM calls | Generation errors |
| --- | --- | --- | --- | --- |
| S1 dense | 3/6 | 13.43 | 12 | 1 |
| S2 balanced hybrid | 4/6 | 11.42 | 13 | 1 |
| S3 CRAG | 6/6 | 15.89 | 16 | 0 |
| S4 graph, first | 5/6 | 33.65 | 23 | 0 |
| S4 graph, repeat | 6/6 | 6.53 | 16 | 0 |
| S5 CRAG + NLI | 2/6 | 16.29 | 25 | 0 |

These are descriptive counts, not evidence of superiority or reliable accuracy
estimates. The denominator includes a clarification case and a web-required
case. Factual correctness is separate from citation support: a correct Nepal
capital answer with an unrelated PDF citation is still a grounding failure.
Claim-support annotation columns remain blank because no exhaustive independent
claim review was performed. Do not compare these counts directly with the older
50-question evaluation, whose prompts and scoring differed.

Initial local context was judged sufficient in 0/5 S1 cases and 4/5 for each
other variant; clarification is excluded. The exact route matched the reviewed
label in 4/6 cases for S3/S4/S5. C04 unnecessarily searched the web despite
locally sufficient mechanism descriptions; C06 selected WEB rather than the
reviewed REWRITE_AND_WEB route. Threshold calibration has not been completed.

### Concrete failures and limits

- S1/C02 and S2/C04 exhausted generation validation with an invalid evidence
  citation and safely abstained. The run completed, but these cases did not pass.
- S1 confused DPR's dot product with cosine similarity, and reversed the
  CRAG/Self-RAG tuning distinction. S2 added an unsupported similarity advantage.
- S4's first C03 answer included a contradictory verification qualifier; its
  repeat had correct core facts but a misattributed citation. Valid citation IDs
  alone do not establish support.
- Web routes produced correct general answers, but some evidence was a
  marketing blog or video. Source authority needs separate review.
- S5 marked only 1/16 initial sentence proxies supported. After nine bounded
  repair attempts and 22 NLI pairs, it retained two claims, repaired one and
  removed 14. It rejected valid positional-encoding statements despite explicit
  retrieved evidence and lost four useful answers. Its 2/2 retained support
  estimate is classifier output, not a 100% human-groundedness result.

### Graph behavior and targeted correction

The initial keyword graph had 373 keyword nodes, 21 shared keywords and took
about 0.097 seconds to build. S4 made seven lazy extraction calls on first
queries and zero on repeats. One extraction returned truncated JSON; retrieval
fell back without crashing. Repeat latency also reflects Ollama and prompt
caches, so the timings do not isolate graph-cache savings.

The extraction prompt was then reduced from up to six relationships to at most
two short relationships, with cache schema `triples-v3`. A targeted C03 run
with a fresh graph cache completed both extraction responses as valid JSON,
accepted one quote-backed relationship, and reused both cached regions on
repeat with zero further extraction calls. This retest used eight LLM calls
and **zero Tavily searches**. It is separate from the original table, not pooled
into those results. Its local artifacts are
`evaluation/results/study_graph_format_fix_20261015.*`.

A subsequent retrieval-only CLI smoke produced four rows (S1, S2 and S4
first/repeat), with zero LLM calls and zero web searches. S4 correctly used
keyword/cache-only mode. Ollama has no model loaded after testing.

The final automated regression run passed **62 tests** with no live model or
search calls. Coverage includes all five system contracts, lazy extraction and
cache invalidation, malformed extraction fallback, bounded verification repair,
web budgets, review preservation and calibration split isolation.

## Decision and remaining study

Keep the existing application default. S4 and S5 remain CLI experiments: this
pilot does not show a graph correctness advantage, and the strict NLI treatment
currently loses too much answer coverage. The graph-format correction addresses
one observed serialization failure; it is not a quality-improvement result.

Next, collect separate calibration and held-out evaluation questions, review
correctness and claim support independently, and report category-level results
and abstention rates. Validate any calibrated thresholds with a fresh runtime
run because evidence guards depend on the configured thresholds. An eager
graph baseline is still needed before claiming lazy indexing is cheaper than
eager construction. No new mathematical algorithm or established improvement
over published GraphRAG/CRAG methods is claimed.
