# Research follow-up to the review notes

Checked against primary papers and author documentation on **15 October 2026**.
This is a literature and code audit, not a new live benchmark. It follows
`MiniProject/next_review_notes.txt`; implemented variants are described in
[the comparison protocol](comparative_study.md).

## Main findings for this project

1. **Lazy graph construction is established prior work.** Microsoft's
   LazyGraphRAG is the closest conceptual reference for deferring LLM work.
2. **Our current S4 is a small graph-assisted retriever.** It does not implement
   the full algorithms of GraphRAG, LazyGraphRAG, LightRAG or HippoRAG.
3. **Graph caching and LLM prefix caching save different work.** A faster repeat
   alone does not establish which cache caused the improvement.
4. **The main missing graph capability is useful connectivity and expansion.**
   Adding more extraction calls without measuring new evidence coverage may
   increase cost without improving answers.
5. **Verification needs calibration and coverage reporting.** Our strict NLI
   pilot discarded valid technical answers; the verifier's own score cannot
   serve as independent proof of improvement.

## A. Graph methods: what each actually does

| Method / primary source | Mechanism | Relationship to our idea | Generative LLM needed for a faithful run? |
| --- | --- | --- | --- |
| [GraphRAG — Edge et al., 2024](https://arxiv.org/abs/2404.16130) | Extract an entity graph, form communities, precompute community summaries, combine partial answers for global questions. | Useful eager/global baseline. Our S4 has neither community summaries nor that global answer procedure. | Yes, for indexing and answering. |
| [LazyGraphRAG — Microsoft Research, 2024](https://www.microsoft.com/en-us/research/blog/lazygraphrag-setting-a-new-standard-for-quality-and-cost/) | Build a noun-phrase co-occurrence graph and communities without LLM summarization; use budgeted relevance assessment and query-focused work later. | Closest inspiration for cheap indexing followed by deferred LLM use. Our unigram/document graph and bounded triple cache omit its community search and claim extraction procedure. | Not for its cheap index; yes at query time. |
| [LightRAG — Guo et al., 2024](https://arxiv.org/abs/2410.05779) | LLM-extracted entities/relations, indexed descriptions, entity-specific and broader thematic retrieval, incremental graph updates. | Its two retrieval levels are not our cheap layer plus lazy rich layer. Our S4 is not a LightRAG reproduction. | Yes, for graph extraction and normal query processing. |
| [HippoRAG — Jiménez Gutiérrez et al., 2024](https://arxiv.org/abs/2405.14831) | Build a corpus knowledge graph, connect similar entity phrases, seed Personalized PageRank from query entities and rank passages. | Strong reference for linking evidence across documents. Our S4 lacks synonym edges and PageRank propagation. | Yes for normal graph/query extraction; traversal itself is ordinary graph computation. |
| [HippoRAG 2 — Jiménez Gutiérrez et al., 2025](https://arxiv.org/abs/2502.14802) | Extend HippoRAG with deeper passage integration and revised online LLM use. | Relevant because graph retrieval can lose ordinary factual retrieval quality. Keep a dense/hybrid path in comparisons. | Yes for the full method. |
| [RAPTOR — Sarthi et al., 2024](https://arxiv.org/abs/2401.18059) | Recursively embed, cluster and summarize chunks into a tree; retrieve at several abstraction levels. | A hierarchy of summaries, not an entity-relation graph. Our shared-keyword degree does not implement this tree. | Yes for summaries and answers; clustering can run without a generator. |

The LazyGraphRAG source above is an official research article. Its published
cost/quality results use its own workload and evaluation; they do not establish
our laptop's speed or accuracy. Likewise, results across the other papers are
not directly comparable to our six-question pilot.

### What the code audit shows

The current implementation in [graph.py](../src/study/graph.py) has:

- Up to 40 TF-IDF **single-token** keywords per PDF, with keyword-to-document
  links. Shared-keyword degree is recorded, but there is no explicit hierarchy
  that traverses general concepts down to unique concepts.
- At most two source documents expanded per retrieval, two candidate chunks per
  document, and two short relationships requested per region.
- Exact quote and entity-occurrence checks. These establish provenance, not a
  semantic proof that the extracted relation is true.
- Small boosts for query-matching entities and a further entity-link hop.
  There is no Personalized PageRank, community detection or learned linker.
- Keyword-neighbor documents receive boosts only for chunks already in the
  candidate ranking. This step does **not** fetch fresh chunks from a newly
  reached document. Previously cached semantic edges can introduce their own
  source chunk IDs; that is a separate, limited expansion mechanism.

The [static CPU audit](../evaluation/study/graph_static_audit_20261015.json)
found 373 keyword nodes: 352 occur in one paper, 16 in two, four in three and
one in four. Each of the three inspected semantic-cache sets had **zero entity
strings shared across documents**, using exact normalized-name equality.
The original pilot cache had five accepted triples; the format-fix cache had
one; the interrupted follow-up cache had one. These cache sets use different
runs/schemas and were audited separately, not merged.

This does not mean the papers have no semantic connections. It means the
currently cached semantic layer has not represented such connections under
its own entity-matching rule. Keyword connections still exist. My inference
is that sparse extraction and limited candidate expansion help explain why
the pilot showed little quality benefit; this is a diagnosis to test.

### Improvements we can investigate without a generative model

| Candidate | CPU work available now | What would count as evidence of value? |
| --- | --- | --- |
| Phrase concepts rather than isolated tokens | Compare TF-IDF n-grams or noun phrases, preserving PDF/page/chunk links. | More useful cross-document links without collapsing unrelated concepts. |
| Entity aliases | Compare normalization and embedding-based candidate aliases on existing cached entities; review ambiguous merges. | Correct alias links, with false merges measured. Similar spelling alone is insufficient. |
| Actual graph expansion | Follow a keyword/concept to a neighboring PDF and retrieve its relevant chunks into the candidate pool. | More questions with evidence from both required papers at the same final top-k. |
| Graph propagation | Test a bounded walk or Personalized PageRank over a fixed keyword/document graph. | Better passage recall/context sufficiency than hybrid alone. This would be an adaptation, not full HippoRAG. |
| Component ablation | Compare hybrid, keyword-only graph and cached keyword+semantic graph, using retrieval-only CLI mode. | Separate the contribution of each layer; no answer-quality claim from retrieval scores alone. |

These are proposed experiments, not changes already applied to S4. A strict
no-model audit uses existing text, keywords and cached triples only. Dense
embeddings or a pretrained NLP encoder can also run on CPU, but still execute
a model and should be described as such.

## B. Claim verification and repair

| Primary source | Established technique | Relevance and limitation |
| --- | --- | --- |
| [RARR — Gao et al., ACL 2023](https://arxiv.org/abs/2210.08726) | Research evidence for generated statements and revise unsupported content while preserving the original answer. | Retrieval plus targeted post-generation repair is prior work. RARR normally uses an LLM and web search; our NLI/local-repair experiment is a different implementation. |
| [AlignScore — Zha et al., ACL 2023](https://aclanthology.org/2023.acl-long.634/) | A trained text-alignment function using several supervision tasks to estimate factual consistency. | A possible evaluation comparison, not proof that replacing our NLI model would improve technical-PDF QA. Needs a local labeled check first. |
| [Our NLI model card](https://huggingface.co/cross-encoder/nli-deberta-v3-small) | Classify premise/hypothesis pairs into contradiction, entailment and neutral. | This is a pretrained classifier, not our invented algorithm. A 0.70 entailment threshold is an uncalibrated project choice. |

Useful work without generation: label saved supported/unsupported claims,
inspect failures caused by PDF extraction or compound sentences, and evaluate
thresholds against those labels. Running the NLI classifier on CPU avoids a
generative LLM but still consumes CPU/RAM. Do not treat an increase in support
percentage caused by deleting most of an answer as an improvement in usefulness.

## C. Confidence gate for speed

[Adaptive-RAG — Jeong et al., NAACL 2024](https://aclanthology.org/2024.naacl-long.389/)
uses a trained smaller language-model classifier to select no retrieval,
single-step or iterative retrieval according to question complexity. It is
related to adaptive routing, but it is not the same as our proposed cheap
retrieval-signal gate. The [CRAG paper](https://arxiv.org/abs/2401.15884) instead
uses retrieval evaluation to trigger corrective actions.

A project gate could inspect dense/lexical ranking agreement, score margin and
coverage of named papers before deciding whether to invoke the LLM grader.
These are candidate features, not probabilities of answerability: a large
retrieval margin or a filename match can accompany insufficient evidence.
RRF scores are rank-fusion scores, not calibrated confidence values.

We can compute features from saved runs and ordinary retrieval on CPU, then
select a gate on a calibration split and evaluate it on a separate split.
Report grader calls avoided **and** wrong bypasses of the grader. The existing
small pilot is insufficient for a reliable learned or hand-tuned gate. No
production bypass has been enabled.

## D. Retrieval or checking during generation

| Primary source | Technique | Feasibility for our project |
| --- | --- | --- |
| [FLARE — Jiang et al., EMNLP 2023](https://aclanthology.org/2023.emnlp-main.495/) | Predict an upcoming sentence, use it to retrieve, and regenerate when token confidence is low. | Requires generation and token-confidence instrumentation. It is not equivalent to checking a completed answer with NLI. |
| [RAG / RAG-Token — Lewis et al., 2020](https://arxiv.org/abs/2005.11401) | Marginalize retrieved-document contributions at output-sequence or token level. | RAG-Token is a model formulation, not a wrapper that searches the web after each token. |
| [Self-RAG — Asai et al., 2023/ICLR 2024](https://arxiv.org/abs/2310.11511) | Train a generator to emit retrieval and reflection tokens. | Our prompted Qwen grader does not implement the trained Self-RAG model. |
| [RETRO — Borgeaud et al., 2021/2022](https://arxiv.org/abs/2112.04426) | Retrieval-enhanced generation with a differentiable encoder and chunked cross-attention. | Requires a retrieval-aware model architecture/training; it is not a small change to our Ollama pipeline. |

Keep these as related work/future work for this review. No live calls are needed
to analyze their algorithms, but faithful end-to-end tests require a generator.

## E. Grounded conversational memory

These sources overlap the memory ideas in the notes; none has been implemented:

| Primary source | Relevant idea |
| --- | --- |
| [MemGPT — Packer et al., 2023](https://arxiv.org/abs/2310.08560) | Manage information across memory tiers beyond a fixed context window. |
| [MemoryBank — Zhong et al., 2023](https://arxiv.org/abs/2305.10250) | Retrieve and update memories, with forgetting/reinforcement inspired by elapsed time and importance. |
| [Generative Agents — Park et al., 2023](https://arxiv.org/abs/2304.03442) | Store experiences, synthesize reflections and retrieve memories to guide behavior. |
| [Zep — Rasmussen et al., 2025](https://arxiv.org/abs/2501.13956) | Temporal knowledge graphs for conversational and business knowledge. |
| [Mem0 — Chhikara et al., 2025](https://arxiv.org/abs/2504.19413) | Extract, consolidate and retrieve conversation memories, including a graph variant. |

The notes' separate confidence/relevance weights, source deduplication and
source-deletion rules form a proposed project policy. This literature check
does not establish that the policy is novel. Confidence cannot automatically
increase merely because several documents repeat the same originating claim.
This remains outside the current comparative study.

## F. Graph caches, prefix caches and precomputed KV caches

| Mechanism / source | What is reused | Implication |
| --- | --- | --- |
| Our S4 graph cache | Validated extracted relationships and their source IDs. | Avoids repeat extraction; it does not cache a final answer or LLM KV tensors. |
| [Automatic prefix caching — vLLM documentation](https://docs.vllm.ai/en/v0.13.0/features/automatic_prefix_caching/) | KV state for matching prompt prefixes. | Reduces prefill work, not generation of new output tokens. Both ordinary CRAG and graph RAG may benefit. |
| [CAG — Chan et al., 2024/WWW 2025](https://arxiv.org/abs/2412.15605) | Preload a manageable knowledge collection into a long context and cache its state. | Relevant to small, stable corpora; our full ten-PDF corpus cannot simply be assumed to fit a 4,096-token context. |
| [RAGCache — Jin et al., 2024](https://arxiv.org/abs/2404.12457) | Organize retrieved knowledge's intermediate states in a tree across GPU/host memory and schedule reuse. | An inference-serving optimization, distinct from caching extracted graph edges. |
| [TurboRAG — Lu et al., 2024](https://arxiv.org/abs/2410.07590) | Precompute document-chunk KV states, with attention/position handling and model adaptation. | Separately encoded caches cannot be assumed equivalent to encoding the concatenated chunks normally. |

The notes' statement that this "needs llama.cpp or vLLM rather than Ollama" is
too absolute. The practical missing feature is control over arbitrary saved
per-chunk KV tensors and their composition; the serving interface and model
must support the chosen method. Our ordinary Ollama chat wrapper has no such
control, so the proposed cache-swapping design is not presently implemented.

For the next live comparison, record server-side prefill/decode/load metrics
where supported, along with graph extraction time and retrieval time. The
[Ollama API](https://docs.ollama.com/api/chat) documents generation timing/token
fields; its [usage documentation](https://github.com/ollama/ollama/blob/main/docs/api/usage.mdx)
also describes cache-related accounting. Available fields depend on the
installed version. Existing traces saved only total stage time, so we cannot
retroactively separate their cache effects exactly.

## Passage selection reference from the notes

[Dai and Callan, *Deeper Text Understanding for IR with Contextual Neural
Language Modeling*, SIGIR 2019](https://www.cs.cmu.edu/~zhuyund/papers/SIGIR19short-BERT-IR.pdf)
compares passage-score aggregation including FirstP and MaxP. Our earlier
highest-scoring-chunk fix is related in spirit, but uses our lexical/hybrid
selection rather than their trained BERT relevance model. It is an engineering
refinement, not a reproduction or a new passage-selection principle.

## Recommended order of work

1. Finish the **paired first/repeat S3/S4 comparison** once the GPU is free;
   keep interrupted, GPU-contended timings separate from a clean performance run.
2. Evaluate keyword-only versus semantic-cache retrieval and inspect actual
   cross-document evidence expansion before increasing graph extraction budgets.
3. Calibrate routing and verification with separate labels; retain answer
   coverage and correctness alongside support rates.
4. Consider a confidence gate only after enough labeled routing data exists.
5. Keep full GraphRAG indexing, memory assistants, live verification and KV-cache
   composition as future work unless the review scope explicitly expands.

A defensible contribution statement is: **a controlled study of retrieval,
corrective routing, bounded lazy graph expansion and post-generation verification
for research-paper QA using a local model, reporting both quality and compute
costs.** Improved quality, a speed advantage and algorithmic novelty remain
hypotheses to demonstrate rather than claims established by implementation.
