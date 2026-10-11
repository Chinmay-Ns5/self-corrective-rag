# Initial baseline versus CRAG findings

Run date: 2026-10-04. Dataset: `questions.jsonl` (50 questions). Model:
`qwen2.5:7b`. Corpus: 10 PDFs, 1,158 Chroma vectors. Retrieval: top 5 chunks.
Routing thresholds: 0.40 and 0.75. The same questions were sent to the
traditional RAG baseline and CRAG. Raw outputs, a review CSV, and a machine
summary are in the local ignored file set `evaluation/results/run_20261004_133251.*`.

## Automatic measurements

| Measure | Baseline RAG | CRAG |
| --- | ---: | ---: |
| Questions completed | 50 | 50 |
| Mean answer latency | 8.891 s | 19.077 s |
| Runtime/API failures | 0 | 0 |

CRAG routes: 10 `LOCAL`, 7 `WEB`, 23 `REWRITE_AND_WEB`, and 10 `CLARIFY`.
Its route agreed with the dataset's *nominal* expectation for 37/50 questions:
9/20 local, 15/15 web-only, 3/5 partial-evidence, 5/5 unclear-reference,
and 5/5 freshness questions. This is **not final routing accuracy**. The nominal
label asks where the answer should come from; a correct evaluator may still use
web search when the top five retrieved PDF chunks miss an answer that exists
elsewhere in the corpus. True routing accuracy requires human labels for the
actual retrieved excerpts, using `gold_route` in the review sheet.

## Manually checked cases

Eleven targeted cases have review labels: all five unclear-reference questions,
all five freshness questions, and one partial-evidence question. These were
selected to inspect edge cases and **must not be extrapolated to the full 50**.
All five unclear-reference questions correctly asked for clarification.
Three of five freshness answers matched independently checked sources; two did
not:

- F01 answered Python **3.14.6**. [Python.org](https://www.python.org/downloads/release/python-3148/)
  lists **3.14.8** as released on 2026-09-30.
- F05 answered Streamlit **1.64.0**. The [official GitHub releases](https://github.com/streamlit/streamlit/releases)
  list **1.65.0** as the latest release on the run date.

P02 asked how Contriever differs from DPR. CRAG chose `LOCAL` and used DPR-paper
excerpts to make claims about Contriever. The indexed DPR paper does not supply
that comparison, so the reviewed gold route is `WEB` and the answer was marked
incorrect and ungrounded.

The baseline safely abstained on the 11 reviewed edge cases. Under the review
rubric, abstention is not counted as a correct answer when the user needs a
specific answer or clarification; it is also not counted as a fabricated fact.
The remaining 39 correctness and groundedness labels are blank. No overall
answer-quality improvement claim is justified yet.

## Work indicated by this run

1. Inspect local retrieval and grader decisions for the 11 local-category
   nominal-route mismatches. Some may be appropriate web fallbacks; others are
   retrieval misses or false clarification requests.
2. Make the query check less likely to request clarification when a question
   explicitly names a subject, without hard-coding example questions.
3. Improve freshness evidence selection and verify the final answer against
   authoritative current sources before making a current-version claim.
4. Complete the 39 blank human reviews, then rerun the same question set after
   fixes and report paired correctness, groundedness, routing, and latency.
