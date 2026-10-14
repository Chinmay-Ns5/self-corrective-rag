# Fresh retrieval checks and adoption (2026-10-14)

The highest-scoring body-chunk policy was adopted into
`src/retrieval/hybrid.py` after the fresh checks below. On inspected retrieved
text it supplies sufficient context for 9/15 questions, versus 7/15 for the
previous first-chunk policy, without losing any previous passes. This supports
a small retrieval improvement on this corpus; it does not establish algorithmic
novelty or improved generated-answer accuracy.

## Setup and review

- 15 new questions: five details, five paraphrased paper references, and five
  multi-paper comparisons. Query wording, reference answers, and inspected
  passage IDs were recorded before the run in
  [`retrieval_fresh_questions.jsonl`](retrieval_fresh_questions.jsonl).
- The candidate was fixed before these questions were run. It was not retuned
  on the fresh results. Questions and passage judgments were authored by the
  assistant; there was no independent or blinded human reviewer.
- Same 10-PDF, 1,158-chunk index and top-5 budget. CPU-only cached
  `all-MiniLM-L6-v2`, two PyTorch threads, seven warmed timing repetitions in
  rotating variant order. No Ollama, generative LLM, Tavily, or new indexing.
- The automatic pass requires at least one preselected chunk from **each**
  evidence group. Comparisons require coverage of both groups. Passage labels
  are deliberately explicit and may not be exhaustive.
- After the run, the full top-5 text for both old and new policies was inspected
  against the reference answers. These sufficiency judgments are saved in
  [`retrieval_fresh_review_20261014.csv`](retrieval_fresh_review_20261014.csv).
  They measure retrieved context, not the output of a running generator.

## Results

| Measure | Previous policy | Adopted policy |
| --- | ---: | ---: |
| Preselected evidence groups covered completely | 5/15 | 7/15 |
| Sufficient context after passage inspection | 7/15 | 9/15 |
| Detailed questions: inspected sufficient context | 1/5 | 3/5 |
| Paraphrased names: inspected sufficient context | 4/5 | 4/5 |
| Multi-paper comparisons: inspected sufficient context | 2/5 | 2/5 |
| Mean of per-query median retrieval times | 9.07 ms | 9.04 ms |

Timing excludes model startup and embedding calculation. The approximately
0.03 ms difference does not support a speedup claim. Shared embedding
calculation averaged approximately 24.1 ms per question.

For completeness, dense-only retrieval covered all preselected groups on 5/15
questions at 4.40 ms, and hybrid without paper reservation on 7/15 at 8.72 ms.
Those two variants were not assigned the post-run sufficiency judgments above.
The reservation heuristic therefore is not uniformly superior to plain hybrid
retrieval on fresh questions, even though changing its body-chunk selection
improves on the previous production policy.

Complete rankings and timing records:
[`retrieval_fresh_results_20261014.json`](retrieval_fresh_results_20261014.json).
The earlier exploratory 30-question comparison is documented separately in
[`retrieval_findings_20261014.md`](retrieval_findings_20261014.md); its lexical
anchor metric should not be pooled with the sufficiency judgments here.

## Reviewed gains and limitations

- `N02`: new chunk `chunk_309` explicitly supplies the Transformer's inner
  feed-forward dimension of 2048. The previous reserved page chunk discussed
  attention heads rather than feed-forward dimensions.
- `N05`: new `chunk_992` explains combining processed knowledge under ambiguous
  CRAG retrieval. Old `chunk_988` ends before the combined-knowledge line;
  the retrieved diagram text is incomplete. No previous sufficiency pass was
  lost in this set.
- `N10` and `N12` illustrate non-exhaustive preselected labels. Both policies
  retrieve other useful passages: `chunk_1072` explains independent ColBERT
  encoding, while `chunk_541` plus the SBERT abstract supports the sentence-pair
  comparison. Passage inspection accepts these even though automatic labels
  miss them. Original label sets and automatic scores remain unchanged.
- `N01` reveals a filename-matching limitation: the words "sentence" and
  "BERT" trigger Sentence-BERT reservations even when the question concerns
  BERT's next-sentence-prediction sampling. Both policies inherit this error.
- `N03` retrieves related NLI dataset discussion, but some names concern
  InferSent rather than SBERT. Mere name overlap is insufficient evidence.
- `N13`–`N15` lack enough context for all parts of their comparisons. Four
  reserved slots for one paper can crowd out another paper's evidence.
  This change does not resolve entity disambiguation or multi-paper allocation.
- No new full RAG evaluation was run. The 2026-10-11 50-question generated-answer
  figures remain historical measurements from before this retrieval change.

## Code and validation

The production change reserves `best_by_page[page]` instead of
`first_by_page[page]`. Existing abstract reservations, page selection, fusion,
and the top-5 budget are retained. The benchmark freezes the previous policy
in a separate class so future comparisons do not accidentally compare the new
implementation with itself. Its historical JSON key `current_hybrid` denotes
the pre-adoption baseline; `scored_body_candidate` denotes the adopted policy.

A regression case checks recovery of an answer later on a selected body page
when its first chunk is boilerplate and dense retrieval misses the answer.
The full suite passed **38 tests** using mocked LLM/web clients and fake
embeddings for integration tests. No live LLM or web-search calls were made.

To repeat the retrieval comparison with a new output path:

```powershell
& '..\..\.venv\Scripts\python.exe' -u -m scripts.benchmark_retrieval --dataset evaluation/retrieval_fresh_questions.jsonl --output evaluation/results/retrieval_fresh_repeat.json
```

The benchmark forces CPU and offline model loading and refuses to overwrite
an existing output. Post-run sufficiency labels require separate inspection.
