# Paired CRAG / layered graph follow-up

**Latest result:** a fresh paired run completed on 21 October 2026. See
[the findings](graph_followup_findings_20261021.md). Both systems averaged
about 9.61 seconds on repeat; S4 had higher first-trial cost.

**Original 15 October attempt:** stopped at the user's request while the GPU was in use. Five of
48 planned trials were preserved, including only one complete question block.
See [the stop record](graph_followup_status_20261015.json). These partial,
GPU-contended timings do not establish a comparative result. A future clean
timing run should use a new output and fresh graph cache directory.

## Question and design

Does the experimental layered graph improve answer quality or repeated-query
latency relative to CRAG when **both systems receive repeat trials**?

The earlier pilot repeated S4 alone. Its 6.53-second repeat average cannot be
compared directly with S3's 15.89-second first average as proof of a graph
advantage, because Ollama prompt caches and model state also affect latency.

This follow-up uses 12 additional question wordings: four single-paper details,
six cross-paper comparisons and two indirect-subject questions. Some underlying
topics overlap previous development work, so this is further development testing,
not a held-out research benchmark. References were checked against local PDF
chunks before reviewing answers. Passage-ID lists are hints, not exhaustive
definitions of context sufficiency.

For each question, run both systems once, then both again in reverse order.
Alternate the first system across questions. Thus each system occupies each of
the four execution positions equally across the twelve questions. A neutral
warmup generation precedes the trials and is counted separately. A fresh graph
cache directory is used at the beginning; later questions may reuse graph
regions expanded by earlier questions. A "first" label means the first trial
for that question, not an independently cold corpus or model.

No Tavily calls are allowed. When the grader selects a web route, report the
resulting abstention as a failed answer for these locally answerable questions.
Do not silently reroute or modify thresholds to rescue it. All generator,
retrieval and extraction settings remain as in the original pilot after its
documented graph-format correction.

```powershell
& '..\..\.venv\Scripts\python.exe' -u -m scripts.compare_rag --dataset evaluation/study/graph_followup_20261015.jsonl --systems s3 s4 --repeat-all --warmup --cache-dir data/study_cache/graph_followup_20261015 --output evaluation/results/graph_followup_20261015.jsonl
```

## Review and reporting

- Review factual correctness independently of the generator's relevance score.
- `answer_grounded=1` requires every substantive factual claim to be supported
  by its cited evidence. Wrong or missing citations fail this stricter metric.
  An abstention has no unsupported factual claims but cannot count as a correct
  and grounded answer when the question is answerable.
- Review initial context sufficiency from the actual retrieved chunks, rather
  than requiring a particular preselected passage ID.
- Report first and repeat mean/median latency for both systems, paired per-query
  repeat wins, retrieval time, graph extraction calls and answer failures.
- Report factual correctness and correct-and-grounded answers separately.
  Labels are assistant reviews, not independent human ground truth.
- Shared Ollama prompt caches can benefit one system after the other. Reversed
  order balances positions; it does not isolate a pure causal graph-cache effect.
- Record competing GPU usage or changing machine load. Laptop wall-clock
  timings are observations under that load, not portable benchmark numbers.

Raw results and model/corpus configuration manifests are local, ignored files
under `evaluation/results/`. Compact reviewed labels, aggregates and findings
are kept with this protocol after completion.
