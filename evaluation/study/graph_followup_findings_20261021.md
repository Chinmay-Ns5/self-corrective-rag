# Paired CRAG versus layered graph: 21 October 2026

## Result

**The graph showed no repeat-query speed advantage or reviewed quality advantage
in this run.** Both systems became faster on repeat. Graph extraction added
substantial first-trial cost.

| Metric | S3: hybrid CRAG | S4: layered graph CRAG |
| --- | ---: | ---: |
| First-trial mean latency | 15.65 s | 34.11 s |
| First-trial median latency | 14.35 s | 32.30 s |
| Repeat mean latency | 9.61 s | 9.61 s |
| Repeat median latency | 9.32 s | 9.48 s |
| Core answer correct, first / repeat | 7/12 / 7/12 | 7/12 / 7/12 |
| Core answer correct **and** fully supported by cited evidence, first / repeat | 4/12 / 4/12 | 4/12 / 4/12 |
| Sufficient initial context, first / repeat | 6/12 / 6/12 | 6/12 / 6/12 |
| Reviewed routing agreement, first / repeat | 9/12 / 9/12 | 8/12 / 8/12 |
| LLM calls, first / repeat totals | 33 / 33 | 56 / 34 |
| Graph extraction calls, first / repeat totals | 0 / 0 | 22 / 0 |

The unrounded repeat means were 9.60769 s and 9.61251 s. This is an observation
on these questions, not a statistical equivalence result or a portable speed
guarantee. On the seven repeat questions whose core answers were correct for
both systems, means were also close: 11.11 s for S3 and 11.17 s for S4.

## What changed from the original pilot

The original pilot repeated only S4. Comparing its warm queries against S3's
first queries mixed graph caching with model/prompt-cache effects. Here both
systems received first and repeat trials, with execution positions balanced.
The graph's repeat mean fell from its own first mean, but ordinary CRAG also
benefited from repetition. The earlier apparent graph speed advantage did not
hold across this larger, paired check.

The previous interrupted GPU-contended run is preserved separately in
[its stop record](graph_followup_status_20261015.json). None of its timings or
labels was pooled into this run.

## Setup and controls

- Twelve additional development question wordings: four single-paper, six
  cross-paper and two indirect-subject questions. Some topics overlap earlier
  development work; this is not a held-out evaluation.
- Forty-eight trials: twelve questions × two systems × first/repeat.
- Both first trials precede both repeats. Starting system alternates by
  question; repeat order reverses the first order. Both systems occupy each
  execution position equally. See [the protocol](graph_followup_protocol_20261015.md).
- Same Qwen2.5:7b generator, prompts, temperature zero, 4,096-token context,
  512-token response limit, initial top-five budget and 0.40/0.75 routing
  thresholds. No retrieval or generation changes were made during the run.
- Ollama 0.40.4, NVIDIA GTX 1660 Ti (6 GB); 25/29 model layers offloaded to
  GPU. MiniLM embedding retrieval ran on CPU with two PyTorch threads.
- Fresh graph cache at the beginning. Later questions could reuse regions
  expanded by earlier questions. "First" means first for that question, not
  a cold Ollama process or an independently cold corpus.
- Setup took 128.83 s, including one neutral warmup call taking 111.03 s.
  Both are excluded from the trial latency table.
- **Zero Tavily searches.** Web was disabled to isolate local behavior and
  conserve credits. Ten rows safely abstained after selecting a web route;
  those are counted as unsuccessful answers, not live search-provider outages.
- 156 LLM calls in trials, plus one warmup call. No generation citation-validation
  failures or graph JSON extraction failures occurred. All 48 rows completed.

Raw results, the configuration manifest, server logs and a working review CSV
are local under `evaluation/results/graph_followup_20261021.*` and
`data/study_cache/graph_followup_20261021/`. They remain ignored by Git.

## Why repeated runs became faster

The graph cache did what it was designed to do: the 22 first-trial extraction
calls consumed about **208.92 seconds** of server inference time; repeats made
zero new extraction calls. However, ordinary CRAG has no such extraction work
to begin with.

Cached retrieval itself was inexpensive for both: mean repeat retrieval time
was **38.8 ms for S3 versus 35.9 ms for S4**. That millisecond difference cannot
explain the several-second speedup from first to repeat generation.

The Ollama server log confirms prompt-prefix reuse. Across the twelve S3 first
trials, prefill took 85.60 s; across repeats it took only 12.59 s. Decode time
was almost unchanged: 98.88 s versus 99.54 s. For S4 repeats, prefill totaled
7.28 s and decoding 104.89 s. Thus prompt processing and generated answer length
matter, alongside avoided graph extraction.

These figures are from [server timing records](graph_followup_20261021.server_timing.json).
All 157 completed server tasks were aligned sequentially with the warmup and
saved LLM events. Counts matched; the largest per-trial client/server duration
difference was 0.223 s. This supports the accounting, but it does not isolate
a causal graph-cache benefit from every other warm-state effect.

## Paired retrieval and latency observations

- **11/12** questions had the same retrieved chunk set across systems on repeat.
- **9/12** had the same ordered context; **9/12** produced identical repeat answers.
- **9/12** repeat-time differences were below one second in magnitude.
- Median paired difference, S4 minus S3, was **+0.070 s**.
- G01 favored the graph by 6.75 s on repeat; G08 favored CRAG by 4.95 s.
  One favorable question is not representative of the whole workload.

The graph ended with 22 cached regions and nine accepted triples. Under exact
normalized entity matching, none of its 18 entity names occurred across more
than one source document. This limits semantic links between papers. Keyword
links still exist. The data therefore describe this small S4 implementation,
not the performance of full GraphRAG, LazyGraphRAG or HippoRAG.

Per-question timings and context/answer equality checks are in
[the paired data](graph_followup_20261021.paired.json).

## Answer review and failure analysis

All labels are **assistant reviews**, not independent human ground truth.
Core correctness checks the requested factual content against the reference
and PDFs. It is distinct from support for every statement in the full answer.
G02's correct summation and G10's high-level siamese-network explanation receive
lenient core credit; their unsupported or vague extra explanations are flagged
in the stricter grounding label. Repeats are repeated measurements of the same
questions, not additional independent quality samples.

| Question | Core answer, both systems | Main observation |
| --- | --- | --- |
| G01: attention scaling | Correct | Both explain the softmax/large-dot-product issue. |
| G02: MaxSim aggregation | Correct core; not fully grounded | Both sum across query terms. Extra wording says "inline" where the PDF says offline; S4 first also claims the combination mechanism is not detailed. |
| G03: original RAG components | Incomplete | S3 abstains. S4 identifies DPR but omits BART. The retrieved chunks do not establish the generator's identity. |
| G04: ReAct roles | Correct | Reasoning/planning versus external actions is supported. |
| G05: ColBERT versus DPR scoring | Incorrect | Both attribute a single-vector ablation in the ColBERT paper to the actual MaxSim model. |
| G06: SBERT versus DPR training | No answer | Both abstain; retrieved context misses the training-signal comparison. |
| G07: Self-RAG versus CRAG training | Correct core; not fully grounded | Both describe the broad distinction, but the CRAG citations are robustness experiments and a clipped future-work passage. |
| G08: ReAct versus CRAG actions | Correct | Supported by the ReAct mechanism and CRAG action descriptions. |
| G09: BERT MLM versus DPR training | No answer | Both abstain; initial context lacks the requested DPR ranking objective. |
| G10: SBERT search cost | Correct at a high level; not fully grounded | Answers mention siamese architecture and BERT's pairwise cost, but mostly omit how reusable embeddings remove pairwise encoding. Cited chunks do not establish that SBERT mechanism. |
| G11: offline late-interaction encoding | Correct | Independent encoding and offline precomputation are directly supported. |
| G12: ambiguous CRAG knowledge sources | Incomplete | Both repeat the Correct/Incorrect categories without naming refined internal retrieval and external web knowledge, despite sufficient context. |

The major weaknesses are missing precise passages, interpreting ablation text
as the main method, overconfident grading and incomplete generation. Graph
extraction did not remedy them in this run. Web fallback might repair some
answers in normal use, but that was deliberately outside this comparison.

Reviewed answers and notes are in [the review CSV](graph_followup_20261021.review.csv);
aggregates are in [the summary](graph_followup_20261021.summary.json).

## Decision

Keep standard CRAG as the application default. Retain S4 as an experimental
comparison. Repeating queries is useful for both systems; the graph has not
earned its extra first-trial cost here.

The next useful graph experiment is genuine evidence expansion into linked
documents, with better phrase/entity linking and a keyword-only ablation.
Measure whether it supplies missing passages before spending more generation
calls. An eager graph baseline and independently labeled evaluation remain
necessary for broader efficiency or quality claims. No new algorithmic
superiority is established by this run.

## Regression checks and cleanup

All **64 automated tests passed** after the live comparison, including the
counterbalanced trial schedule and separate correctness/grounding aggregation.
Those tests used fake model/search clients. Ollama's model was unloaded and
the background server started for this run was stopped.
