# Reviewed baseline versus CRAG evaluation (2026-10-11)

This is the report-ready evaluation record for the 50 questions in
[`questions.jsonl`](questions.jsonl). The per-question decisions and reasons are in
[`review_20261011.csv`](review_20261011.csv). Raw answers, retrieved excerpts, source
URLs, and the full review sheet are retained locally under the ignored
`evaluation/results/run_20261011_final_snapshot.*` files.

## Configuration and method

- Both systems used `qwen2.5:7b` at temperature zero, the same 10-PDF corpus
  (1,158 Chroma vectors), and top-5 retrieval. CRAG used relevance thresholds
  0.40 and 0.75.
- The dataset has 20 locally answerable, 15 web-only, 5 partial-evidence,
  5 unclear-reference, and 5 freshness questions.
- The 50-question run was split after question 13 so Ollama could reload onto
  the freed NVIDIA GPU. Seven affected questions (`L04`, `L18`, `P01`–`P05`)
  were rerun after focused comparison and citation fixes. The final snapshot
  replaces only those rows. The full run and targeted reruns used 34 Tavily
  searches in total; the final snapshot contains 28 web routes.
- The assistant checked the answers against the dataset reference and the
  recorded PDF excerpts or web source links. Current facts were checked against
  the official sources linked below. An independent human reviewer has not
  repeated these judgments. Labels are binary and deliberately strict: an
  incomplete answer or an unsupported additional claim is marked incorrect.
- `gold_route` was assigned from the **retrieved excerpts**, not merely the
  dataset category. `LOCAL` requires enough evidence for the full question;
  `WEB` means related but incomplete evidence; `REWRITE_AND_WEB` means
  irrelevant evidence; `CLARIFY` means the question lacks an identifiable
  subject. `WEB` and `REWRITE_AND_WEB` are distinct for routing accuracy.

## Results

| Measure | Baseline RAG | CRAG |
| --- | ---: | ---: |
| Correct answers or appropriate clarifications | 7/50 (14%) | 44/50 (88%) |
| Grounded or fact-free answers | 48/50 (96%) | 47/50 (94%) |
| Correct among substantive answers | 7/10 | 39/44 |
| Grounded among substantive answers | 8/10 | 41/44 |
| Mean answer time, mixed CPU/GPU run | 12.2 s | 26.4 s |

CRAG routing matched the reviewed gold route on **40/50 (80%)**. The routes
were 17 `LOCAL`, 16 `WEB`, 12 `REWRITE_AND_WEB`, and 5 `CLARIFY`. The automatic
dataset-level web-route agreement is 47/50, but it treats `WEB` and
`REWRITE_AND_WEB` as equivalent and therefore overstates exact routing accuracy.

| Question category | Baseline correct | CRAG correct | Exact CRAG route |
| --- | ---: | ---: | ---: |
| Local (20) | 7 | 17 | 18 |
| Web-only (15) | 0 | 15 | 9 |
| Partial evidence (5) | 0 | 2 | 5 |
| Unclear reference (5) | 0 | 5 | 5 |
| Freshness (5) | 0 | 5 | 3 |

The high overall improvement is strongly influenced by the dataset design:
the local-only baseline correctly abstains on most questions outside its
corpus, but abstention is scored incorrect when an answer or clarification is
available. Fact-free abstentions and clarifications are scored grounded, so
the overall groundedness percentages are not measures of answer usefulness.

The latency figures are descriptive only. The run changed from mostly CPU to
mostly GPU after question 13, and some targeted reruns benefited from a warm
model. A controlled timing comparison would need one stable hardware state
and repeated runs.

## Reviewed failures and limits

- `L03` and `L15` unnecessarily used web search even though the retrieved
  survey abstract and Self-RAG title/abstract answer the questions. `L10`
  missed **BM25/Lucene** despite retrieving excerpts that name it.
- `L04` produced a cited but incomplete answer about RAG's benefits rather
  than naming external knowledge. `L12` added an incorrect claim about the
  directionality of an earlier language model.
- The comparison guard prevents a local answer when one named method is
  absent from the excerpts. `P04` then safely abstained because web search
  returned no usable RAGAS evidence. `P02` conflated the original Contriever
  with a later MultiContrievers study; `P05` added unsupported selection
  criteria for Adaptive-RAG. Partial comparisons remain the weakest category.
- A valid citation ID does not prove that its text supports the claim. The
  generator now retries once on an invalid or missing citation and abstains
  when a comparison lacks evidence for a named side, but claim-level
  groundedness still needs review.
- The web refiner filters missing text and duplicate URLs, but does not rank
  source authority. **14 of 28** final web-route answers cited at least one
  social, video, or Q&A site. Stronger source selection is a clear future
  improvement; these pages did not change the reference answers for the
  simple web-only questions in this set.
- One `P04` result has a nonempty `error` field because the web evidence was
  insufficient. It is an explicit abstention, not an API or runtime crash.

## Time-sensitive reference checks

These checks apply to **2026-10-11**; current answers must be rechecked on
the demo or report date.

- `F01`: Python **3.15.0**, released 2026-10-09 —
  [Python.org](https://www.python.org/downloads/release/python-3150/).
- `F02`: **António Guterres** —
  [United Nations](https://www.un.org/sg/en/about-secretary-general).
- `F03`: **Luiz Inácio Lula da Silva** —
  [Brazilian presidency](https://www.gov.br/planalto/pt-br/acompanhe-o-planalto/agenda-do-presidente-da-republica-lula/agenda-do-presidente-da-republica).
- `F04`: **Francis Halzen**, 2026 Physics prize —
  [Nobel Prize](https://www.nobelprize.org/prizes/physics/2026/summary/),
  [U.S. National Science Foundation](https://www.nsf.gov/mps/updates/2026-nobel-prize-physics-awarded-nsf-icecube-neutrino).
- `F05`: Streamlit **1.65.0** —
  [official GitHub release](https://github.com/streamlit/streamlit/releases/tag/1.65.0).

## Reproduce the review summary

The raw final snapshot is ignored by Git but available in this workspace.
From the repository root:

```powershell
& '..\..\.venv\Scripts\python.exe' -m scripts.summarize_evaluation evaluation/results/run_20261011_final_snapshot.review.csv
```

The compact checked labels are committed in `evaluation/review_20261011.csv`
so the report figures remain auditable without committing the large generated
web evidence files.
