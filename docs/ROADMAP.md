# Roadmap

This project already does end-to-end semantic code search + RAG; the roadmap below turns it
into a **measured, production-shaped** retrieval system. Work is planned as small, independent
pull requests, and every retrieval change is validated with the evaluation harness (`eval/`)
against a real corpus rather than asserted.

## Evaluation methodology

Improvements are measured, not claimed. The benchmark corpus is a real, recognizable
application — **[Gitea](https://github.com/go-gitea/gitea)** (~3k source files) — so the labeled
queries are feature-level and easy to reason about ("where is authentication handled?", "how are
pull requests merged?"). The labeled set includes **negative queries** (questions with no valid
answer) to verify the system declines to answer rather than fabricating one.

Each retrieval change is reported as an **ablation** so the contribution of each step is visible:

| Config | precision@k | recall@k | hit@k | MRR | nDCG@k |
| baseline (line chunks, dense) | … |
| + AST chunking | … |
| + hybrid (dense + BM25) | … |
| + reranking | … |

## Planned work

### Retrieval quality
- **AST-aware chunking** — chunk on function/class boundaries (tree-sitter) instead of fixed line
  windows, so each chunk is a coherent unit of code. Falls back to line-windowing for unsupported
  languages. Toggle: `CHUNK_STRATEGY=line|ast`.
- **Hybrid retrieval** — combine sparse (BM25/SPLADE) and dense vectors with Reciprocal Rank Fusion.
  Captures exact symbol / error-string matches that pure semantic search misses. Toggle:
  `RETRIEVAL=dense|hybrid`.
- **Cross-encoder reranking** — re-order the top-N candidates with a cross-encoder for higher
  precision, with the quality/latency tradeoff documented. Toggle: `RERANK=on|off`.

### RAG quality & safety
- **Retrieval-confidence fallback** — if the best retrieval score is below a threshold, return
  "insufficient evidence" instead of generating an answer. The goal is *reduced* hallucination
  through constrained generation + confidence gating + citations — not an absolute guarantee.
- **Inline citations** — attach a `path:line` reference to each claim in a generated answer, not
  just a sources list at the end.

### Evaluation
- **Adversarial evaluation** — negative queries assert the fallback triggers (answerability metric).
- **CI regression gate** — run the evaluation in GitHub Actions and fail the build if key metrics
  regress.

### Observability
- **Metrics & dashboard** — expose query latency, retrieval scores, and fallback rate via a
  Prometheus `/metrics` endpoint, with a Grafana dashboard included in the repo.

## Deliberately out of scope

To keep the project focused (it indexes a single codebase, not a massive document store):
- Source-confidence scoring by freshness/trust — a better fit for news/document RAG than code.
- Response/embedding caching — low value at this scale.
- Horizontal scaling / sharding for very large corpora.
