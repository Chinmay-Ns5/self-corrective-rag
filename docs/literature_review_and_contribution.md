# Literature review and contribution: presentation brief

Prepared for the Prompt Engineering mini-project's first review. This file is
a standalone factual brief for drafting slides; technical evidence links point
to committed files in this repository. Current status: 2026-10-14.

## Project and course scope

**Project:** Self-Corrective RAG with Fallback Web Search and Routing.

The professor's guidelines identify this as an Application / Advanced RAG
project. Review 1 covers idea presentation and literature survey. The required
technical approach is relevance evaluation of vector-retrieved documents,
dynamic local/web routing, and query rewriting for incorrect retrieval.
The final report must be 10 pages in IEEE format. The course document does not
specify a mandatory number of papers or explicitly require a new mathematical
algorithm. The professor has additionally asked the team to explain its
differences from prior work.

Our practical problem: local PDF retrieval can return passages from the wrong
paper, incomplete evidence, or information insufficient for a current question.
An answer generator should assess that evidence before relying on it.

## Four main literature references

Years below identify the linked preprint editions, except Lewis et al., whose
paper was accepted at NeurIPS 2020. Use these editions consistently in slide
references; do not guess a different publication venue from the year alone.

| Reference | Main idea | Relevance to our project | Distinction from our implementation |
| --- | --- | --- | --- |
| Shi-Qi Yan, Jia-Chen Gu, Yun Zhu, and Zhen-Hua Ling, **Corrective Retrieval Augmented Generation**, 2024, arXiv:2401.15884 | Evaluate retrieval quality and trigger corrective actions, including web search and knowledge refinement. | Main architectural inspiration for relevance grading and local/web routing. | The original fine-tunes a T5-large evaluator and uses knowledge-strip decomposition/filtering/recomposition. We prompt an existing local Qwen model and select cited excerpts; we have not reproduced the full training or refinement procedure. |
| Patrick Lewis et al., **Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks**, 2020, arXiv:2005.11401 | Combine pretrained parametric memory with retrieved non-parametric knowledge; distinguish RAG-Sequence and RAG-Token. | Foundation for retrieval-assisted generation and motivation for a basic local-only baseline. | Our baseline is a dense-retrieval prompt pipeline, not a reproduction of their jointly trained RAG models. |
| Yunfan Gao et al., **Retrieval-Augmented Generation for Large Language Models: A Survey**, 2023 preprint, arXiv:2312.10997 | Organize RAG into naive, advanced, and modular approaches; review retrieval, augmentation, generation, and evaluation. | Background for design choices, hallucination/outdated-knowledge motivation, and evaluation discussion. | It is a survey, not a system we claim to outperform or an algorithm we reproduce. |
| Akari Asai, Zeqiu Wu, Yizhong Wang, Avirup Sil, and Hannaneh Hajishirzi, **Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection**, 2023 preprint, arXiv:2310.11511 | Train an LM to retrieve and critique using reflection tokens. | Closely related work on adaptive retrieval and evidence critique. | We use prompts, an external routing graph, and deterministic checks. We do not train or use Self-RAG reflection-token models. |

Primary links:

1. [Yan et al.: CRAG](https://arxiv.org/abs/2401.15884),
   [method details](https://arxiv.org/html/2401.15884v3).
   Corpus file: `09_corrective_rag.pdf`.
2. [Lewis et al.: RAG](https://arxiv.org/abs/2005.11401).
   Corpus file: `01_rag_knowledge_intensive_nlp.pdf`.
3. [Gao et al.: RAG survey](https://arxiv.org/abs/2312.10997).
   Corpus file: `02_rag_survey.pdf`.
4. [Asai et al.: Self-RAG](https://arxiv.org/abs/2310.11511).
   Corpus file: `08_self_rag.pdf`.

## Other six papers in the PDF knowledge base

These provide retrieval/model background and topics for knowledge-base tests.
Being indexed does not mean that we implemented their algorithms.

| Paper | Corpus filename | Use and implementation boundary |
| --- | --- | --- |
| **Attention Is All You Need** — Ashish Vaswani et al. | `03_attention_is_all_you_need.pdf` | Transformer background and QA evidence. We use pretrained models rather than implementing or training the original Transformer. |
| **Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks** — Nils Reimers and Iryna Gurevych | `04_sentence_bert.pdf` | Sentence-embedding background and QA evidence. The actual cached embedding model is `all-MiniLM-L6-v2` through Sentence Transformers. |
| **Dense Passage Retrieval for Open-Domain Question Answering** — Vladimir Karpukhin et al. | `05_dense_passage_retrieval.pdf` | Dense retrieval background and QA evidence. We do not train or use the paper's DPR encoders. |
| **BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding** — Jacob Devlin et al. | `06_bert.pdf` | Encoder-model background and QA evidence. No BERT pretraining is performed. |
| **ReAct: Synergizing Reasoning and Acting in Language Models** — Shunyu Yao et al. | `07_react.pdf` | Related prompting/tool-use work and QA evidence. Our workflow does not implement a ReAct agent. |
| **ColBERT: Efficient and Effective Passage Search via Contextualized Late Interaction over BERT** — Omar Khattab and Matei Zaharia | `10_colbert.pdf` | Related retrieval architecture and QA evidence. We do not use ColBERT's token-level late-interaction retriever. |

All ten PDFs are locally under `data/raw/pdfs/`. The corpus and generated index
are ignored by Git, so a slide-writing LLM with only the GitHub checkout should
use the main-paper primary links above or be given the PDFs separately.

Additional work consulted during the focused novelty check, rather than part of
the ten-PDF corpus:

- Cormack, Clarke, and Buettcher, **Reciprocal Rank Fusion Outperforms Condorcet
  and Individual Rank Learning Methods** (2009):
  [original paper](https://cormack.uwaterloo.ca/cormacksigir09-rrf.pdf).
- Zaragoza et al., **Microsoft Cambridge at TREC 2004: Web and HARD Track**:
  [Microsoft Research](https://www.microsoft.com/en-us/research/publication/microsoft-cambridge-at-trec-2004-web-and-hard-track/?lang=ko-kr).
  Prior work on using document fields in retrieval; our keyword index is not a
  full reproduction of BM25F.
- Ding et al., **SciRAG: Adaptive, Citation-Aware, and Outline-Guided Retrieval
  and Synthesis for Scientific Literature** (2026):
  [ACL Anthology](https://aclanthology.org/2026.eacl-long.303/).
  Related scientific-literature RAG; its citation-graph and outline methods are
  distinct from our simple paper/page reservations.

## Implemented system

PDF upload/ingestion -> chunking -> MiniLM embeddings -> Chroma index.

Question -> subject clarification check -> dense + lexical retrieval ->
LLM evidence evaluation -> routing -> selected evidence -> cited generation.

The LangGraph routes are:

- `CLARIFY`: ask for the missing subject before retrieval.
- `LOCAL`: sufficiently relevant local evidence.
- `WEB`: related but incomplete evidence; add external evidence.
- `REWRITE_AND_WEB`: irrelevant or absent evidence; rewrite and search.

Generation uses Ollama; web fallback uses Tavily. The full application requires
a generative LLM. The independent retrieval benchmark uses only the cached
embedding encoder and can run on CPU without Ollama or Tavily.

The application has both CLI and Streamlit entry points, PDF upload, PDF/page
and web URL source display, citation-ID checking with one repair attempt, and
limited checks for evidence about both named sides of common comparison forms.
A valid citation ID does not guarantee claim-level support.

## Our contribution and its algorithmic detail

The defensible contribution is **an implemented and evaluated CRAG adaptation
for research-paper QA, including a passage-selection refinement**. Research
novelty has not been established. We have not shown superiority over published
CRAG or Self-RAG.

Our retrieval combines dense and BM25-style lexical rankings using reciprocal
rank fusion. In code the fusion score is:

`score(chunk) = 1 / (20 + dense_rank) + 1 / (20 + lexical_rank)`

A term is included only for rankings in which the chunk occurs. Rank fusion and
BM25-style scoring are established techniques, not new formulas from this team.

For a filename-title match, the current paper-selection heuristic reserves:

1. The first two chunks on the matched paper's first page.
2. One chunk on each of two selected body pages, within the existing early-body
   page range. Pages are ranked by their best lexical chunk score.
3. Remaining top-5 slots are filled by the fused ranking, without duplicate IDs.

The tested refinement changes which chunk is reserved on a selected body page:

`reserved_chunk(page, question) = argmax lexical_score(chunk, question)`

The previous policy took the first chunk on that page even when a later chunk
scored better. The refinement reuses already-computed lexical scores and adds
no generative-model call. It is a practical selection rule whose usefulness
was tested; the use of an argmax is not itself mathematical novelty.

## Evidence suitable for slides

| Experiment | Finding | Required qualification |
| --- | --- | --- |
| Historical 50-question baseline versus complete CRAG run, 2026-10-11 | Reviewed correctness/appropriate clarification: 7/50 baseline, 44/50 CRAG. Exact reviewed routing: 40/50. | Before the new retrieval refinement. Many questions require web access; the baseline lacks web access. Retrieval and prompts also differ. This does not isolate the refinement or compare with published CRAG. Labels were assigned by the assistant, not independently reviewed humans. |
| Exploratory 30-question retrieval ablation, 2026-10-14 | Lexical evidence-anchor hits: dense 14/30, plain hybrid 21/30, old paper selection 24/30, refined selection 26/30. | These are automatic lexical proxies, not semantic answer accuracy. The refinement was developed using this set. |
| Fresh 15-question retrieval check, 2026-10-14 | Inspected sufficient context: old policy 7/15, refined policy 9/15; two gains and no lost passes. | Assistant-authored questions and assistant passage inspection; small corpus, no independent/blinded review, no generated answers. |
| Same fresh check, retrieval timing | Old 9.07 ms versus refined 9.04 ms. | Excludes embeddings and startup. Seven warmed repetitions per query, CPU. The difference does not establish a speedup. |
| Automated regression suite after adoption | 38 tests passed. | Fake/mocked model and search clients; these are implementation checks, not 38 live accuracy examples. |

Authoritative local evidence:

- [50-question findings](../evaluation/final_findings_20261011.md).
- [Exploratory retrieval findings](../evaluation/retrieval_findings_20261014.md).
- [Fresh retrieval findings](../evaluation/retrieval_fresh_findings_20261014.md).
- [Per-question fresh passage review](../evaluation/retrieval_fresh_review_20261014.csv).

The fresh result supports better evidence selection at comparable retrieval
time. Multi-paper comparisons still provide sufficient inspected context for
only 2/5 questions under both old and new policies. Filename matching can select
the wrong paper, and reserving most slots for one paper can crowd out another.
Web source authority and claim-level groundedness also remain limitations.

## Suggested first-review slide sequence

1. Title and team details; leave unknown names/IDs as explicit placeholders.
2. Problem: incomplete, irrelevant, or stale retrieved context.
3. Literature foundations: Lewis RAG and Gao's survey.
4. Closest methods: CRAG and Self-RAG, including evaluator/training differences.
5. Literature comparison table: method, relevance, and our implementation differences.
6. Proposed system and routes; label it a CRAG adaptation.
7. Practical contribution: paper selection and the highest-scoring body chunk.
8. Evaluation: fresh 7/15 -> 9/15 context result, definitions, and limitations.
9. Remaining gaps and next work; distinguish implemented features from future work.
10. References: four main papers plus retrieval prior work as appropriate.

Suggested spoken answer about contribution:

> We adapt CRAG for research-paper QA using local inference, hybrid retrieval,
> and evidence checks. Our tested refinement reserves the highest-scoring chunk
> on a selected paper page instead of its first chunk. It improved sufficient
> retrieved context from seven to nine of fifteen fresh questions at comparable
> retrieval time. We claim a measured implementation improvement; algorithmic
> novelty and superiority over published CRAG or Self-RAG are unproven.

## Prompt to give a slide-writing LLM

Create a 10-slide first-review presentation for our Prompt Engineering mini
project using this brief as the factual source. Cover the problem, the four
main research papers with citations, the comparison with our implementation,
the routing architecture, the passage-selection refinement, and the fresh
retrieval evidence. Add concise speaker notes explaining each paper and how
our implementation differs. Use the exact metric definitions and qualifications
above. Describe the contribution as a measured engineering refinement; do not
invent research novelty, a speedup, paper-level benchmark superiority, or an
updated full-system accuracy result. Distinguish corpus papers from methods
actually implemented. Keep team details as placeholders if they are not provided.
