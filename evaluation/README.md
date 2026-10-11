# Baseline versus CRAG evaluation

The completed 2026-10-11 evaluation, including reviewed scores and known failures, is in
[`final_findings_20261011.md`](final_findings_20261011.md). Its compact per-question
labels are in [`review_20261011.csv`](review_20261011.csv).

`questions.jsonl` contains 50 labeled questions: 20 local, 15 outside the PDF corpus,
5 with partial local evidence, 5 lacking a referent, and 5 requiring current information.
The expected route is `LOCAL`, `CLARIFY`, or `WEB_ANY` (`WEB` or `REWRITE_AND_WEB`).
These are task-level expectations based on the question and corpus. They do not prove
that the top five retrieved chunks actually contain enough evidence. The automatic
`dataset_route_agreement` figure is provisional, not a measured routing accuracy.

From the repository root, with Ollama running and `TAVILY_API_KEY` in `.env`:

```powershell
& '..\..\.venv\Scripts\python.exe' -m scripts.evaluate --offline
```

Use `--ids L01,W01,P01,A01,F01` for a small balanced run. `--limit 10` selects the first
ten dataset rows. The runner loads the PDF index and embedding model once, then runs
the same question through baseline RAG and CRAG with the configured Ollama model.
It saves raw JSONL, a review CSV, and an automatic routing/latency summary under
`evaluation/results/` (ignored by Git). Results are flushed after every question so
an interrupted run retains completed cases. API web searches consume Tavily credits.

## Human review

For each question in the generated `.review.csv`, inspect both answers, their
listed sources, and `retrieved_documents`. Fill `baseline_correct`,
`baseline_grounded`, `crag_correct`, and `crag_grounded` with `1` or `0`; leave
a cell blank if it was not reviewed. Use:

- **Correct:** the answer matches the verified reference fact, or correctly asks
  for clarification when the question has no identifiable subject. An unsupported
  factual answer is incorrect. An abstention is incorrect when an answer is available.
- **Grounded:** every factual claim in the answer is supported by its cited/source
  evidence. For web answers, open the actual URLs. For PDF answers, inspect the
  recorded filename and page. An abstention or clarification has no factual claim;
  mark groundedness `1` if it does not invent a fact.
- **Current facts:** verify against the named official source on the run date and
  write the verified answer and date in `notes`. Do not reuse an old gold answer.
- **Partial evidence:** check both the local paper and an external source before
  deciding whether the answer is complete.
- **Routing:** label `gold_route` from the evidence actually retrieved: `LOCAL`
  if the excerpts fully answer the question, `WEB` if related but incomplete,
  `REWRITE_AND_WEB` if irrelevant, or `CLARIFY` if the question lacks a subject.
  For current facts, undated local excerpts are insufficient. The reviewed
  `gold_route` values produce the real routing-accuracy denominator.

The reference answers are short review aids, especially for local paper questions;
reviewers should verify them against the source before scoring. The runner does not
automatically turn word overlap or model self-grading into correctness numbers.
After filling labels, run:

```powershell
& '..\..\.venv\Scripts\python.exe' -m scripts.summarize_evaluation evaluation/results/<run>.review.csv
```

The resulting per-category rates and the automatic summary provide the table for
the report. Record the model, corpus size, thresholds, and run date from the
summary JSON alongside the results. Report incomplete reviews with their actual
denominators; do not extrapolate a small batch to all 50 questions.
