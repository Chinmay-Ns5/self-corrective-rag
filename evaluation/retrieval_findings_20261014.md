# Retrieval-only ablation (2026-10-14)

The current paper passage-selection heuristic improves the aggregate evidence
match rate over hybrid retrieval without reservations on this exploratory set,
but hurts some detailed questions. A small candidate that reserves the highest
lexical-scoring chunk on a selected body page instead of its first chunk
recovers two verified passages with comparable execution time.

## Method

- No Ollama, generation model, Tavily searches, or new indexing. The cached
  `all-MiniLM-L6-v2` embedding model ran on CPU with two PyTorch threads.
- Existing index: 10 PDFs, 1,158 chunks; top-5 retrieval in every variant.
- 30 questions: the existing 20 local questions and 10 additional detail
  questions, one per paper. These are exploratory questions, not a blinded
  holdout. The existing questions previously informed implementation choices.
- The dataset in [`retrieval_questions.jsonl`](retrieval_questions.jsonl)
  declares target PDFs and lexical evidence patterns. A hit requires at least
  one top-5 chunk from the target PDF matching every declared pattern.
  This is an **evidence-anchor proxy**, not generated-answer accuracy,
  exhaustive relevance labeling, or semantic evidence sufficiency.
- Query embeddings are computed once and shared across variants. Each retrieval
  variant is warmed once per question, then measured seven times in rotating
  order. Timing below is the mean of per-question median retrieval times,
  excluding embedding calculation and startup. It includes the vector query,
  and for hybrid variants the lexical ranking and fusion.
- Initial baseline ablation:
  [`retrieval_ablation_20261014.json`](retrieval_ablation_20261014.json).
  Four-way comparison including the candidate:
  [`retrieval_ablation_scored_20261014.json`](retrieval_ablation_scored_20261014.json).
  Both retain all ranked IDs, patterns, scores, and timings.

## Results from the four-way comparison

| Variant | Existing questions (20) | Added details (10) | Total evidence hits (30) | Retrieval time |
| --- | ---: | ---: | ---: | ---: |
| Dense only | 9 | 5 | 14 (46.7%) | 4.44 ms |
| Hybrid without paper reservation | 14 | 7 | 21 (70.0%) | 8.15 ms |
| Current hybrid with paper reservation | 19 | 5 | 24 (80.0%) | 8.59 ms |
| Candidate: highest-scoring body chunk | 19 | 7 | 26 (86.7%) | 8.34 ms |

The current heuristic adds approximately 0.44 ms over plain hybrid retrieval
in this comparison. The candidate's approximately 0.25 ms difference from the
current heuristic is too small to support a meaningful speedup claim from one
run. Its supported benefit is two additional evidence hits at comparable time.
Dense retrieval is fastest here; it retrieves fewer matching evidence passages.
The measured query embedding time averaged approximately 21.2 ms, shared across
variants. These figures are retrieval measurements, not full RAG response times.

## What changed and what still fails

The candidate changes only the choice of body chunk on an already selected page.
It preserves title matching, the first two first-page chunks, body page limits,
the two selected body pages, and the final top-5 budget. At the time of this
initial comparison it existed only in the benchmark, before production adoption.

This describes the initial comparison. The candidate was subsequently adopted
after the [fresh 15-question checks](retrieval_fresh_findings_20261014.md).

- `D03`: the candidate retrieves `chunk_323`, page 8 of the Transformer paper,
  explicitly stating label smoothing of 0.1. The current first-chunk policy
  misses that passage.
- `D10`: the candidate retrieves `chunk_1089`, page 4 of ColBERT, explicitly
  describing summation of maximum similarities across query terms.
- `D05`: a broad initial pattern gave credit for a passage discussing alternative
  similarity functions. Manual inspection showed it did not directly define DPR
  scoring. The pattern was tightened for all variants, removing that apparent
  candidate gain. Stored rankings and times were rescored without rerunning
  retrieval. The final figures above use the tightened labels.
- `D09`: plain hybrid retrieves the CRAG evaluator's T5-large description; both
  reservation variants still crowd it out. Reserving four of five slots can
  discard a useful passage outside those selected pages.
- `L20`: a question mentions both ColBERT and BERT. The current title-match tie
  selects the BERT paper because it appears first in the index. Both reservation
  variants inherit this ambiguity. No title-disambiguation fix was evaluated.

Thus the candidate gains two questions and loses none against the current
variant on these 30 checks. Against plain hybrid, it gains six and loses one.
Neither comparison establishes general superiority on unseen documents.

## Literature and novelty

A focused literature check found that the principal building blocks are already
established: [reciprocal rank fusion](https://cormack.uwaterloo.ca/cormacksigir09-rrf.pdf)
combines ranked retrieval lists, and
[field-aware retrieval work](https://www.microsoft.com/en-us/research/publication/microsoft-cambridge-at-trec-2004-web-and-hard-track/?lang=ko-kr)
uses document fields in retrieval. Scientific literature RAG also has published
structure-aware approaches such as [SciRAG](https://aclanthology.org/2026.eacl-long.303/),
whose citation and outline methods differ from our simple reservations.

This search does not establish whether the exact first-page/body-page quota has
appeared elsewhere. It provides no basis for claiming mathematical novelty.
The defensible contribution is an evaluated document-specific retrieval heuristic
and an exploratory ablation, with the limitations above.

## Reproduce

From the repository root, with the existing index and cached embedding model:

```powershell
& '..\..\.venv\Scripts\python.exe' -u -m scripts.benchmark_retrieval --output evaluation/results/retrieval_ablation.json
```

The script forces offline model loading and CPU inference. It exits after the
benchmark, releasing the embedding model. No generative LLM service is needed.
