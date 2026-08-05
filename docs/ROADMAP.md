# Roadmap

This project does end-to-end semantic code search + RAG, and every retrieval change is
**validated with the evaluation harness (`eval/`) against a real corpus rather than asserted**.
The items below were built as small, independent pull requests; each is measured, not assumed.

## Delivered

### Retrieval quality
- **AST-aware chunking** — chunk on function/class boundaries (tree-sitter) instead of fixed line
  windows. Falls back to line-windowing for unsupported languages. Toggle: `CHUNK_STRATEGY=line|ast`.
- **Hybrid retrieval** — sparse (BM25) + dense vectors fused with Reciprocal Rank Fusion; captures
  exact symbol / error-string matches pure semantic search misses. Toggle: `RETRIEVAL=dense|hybrid`.
- **Cross-encoder reranking** — re-order the top-N candidates for higher precision. Toggle: `RERANK=on|off`.

### RAG quality & safety
- **Retrieval-confidence fallback** — below a score threshold, return "insufficient evidence" instead
  of generating (`FALLBACK_MIN_SCORE`). Reduced hallucination via constrained generation + gating +
  citations — not an absolute guarantee.
- **Inline citations** — a `[path:line]` reference after each claim, not just an end-of-answer list.

### Evaluation
- **Ablation runner** (`eval/ablation.py`) — measures each change on a real corpus. See the
  [measured ablation results](../README.md#a-measured-ablation-gitea) in the README.
- **Adversarial evaluation** — negative queries assert the fallback triggers (answerability metric).
- **CI regression gate** — a deterministic end-to-end eval runs in GitHub Actions (no model download).

### Observability
- **Metrics & dashboard** — query latency, retrieval scores, and fallback rate via a Prometheus
  `/metrics` endpoint (`METRICS_ENABLED=on`), with a Grafana dashboard in [`observability/`](../observability).

## Next

The second measurement round resolved the previous "Next" list — each item was built and measured
(see [ablation results](../README.md#a-measured-ablation-gitea) and [`eval/RESULTS.md`](../eval/RESULTS.md)):
fair non-stacked arms ✓; the code-specialized model ✓ (measured, **rejected as default** — it lost
to `bge-small` once context was on); context-preserving embeddings ✓ (**shipped as
`EMBED_CONTEXT=path`, now the default**, +73% relative recall).

Remaining ideas, in rough priority order:

- **Larger labeled set** — grow from 28 to 50+ queries with multi-file ground truth, to shrink the
  ~3.6-point-per-query noise floor and make smaller effects measurable.
- **Code-tuned reranker** — the general-purpose cross-encoder was measurably harmful on code;
  evaluate a code-specific reranker before reconsidering the `RERANK` default.
- **File-level aggregation in the serving path** — sum-of-top-N pooling helped in evaluation
  ([`eval/aggregate.py`](../eval/aggregate.py)); expose it as a search option, not just an eval mode.
- **Query expansion** — rewrite terse queries with a local LLM before retrieval.

## Deliberately out of scope

To keep the project focused (it indexes a single codebase, not a massive document store):
- Source-confidence scoring by freshness/trust — a better fit for news/document RAG than code.
- Response/embedding caching — low value at this scale.
- Horizontal scaling / sharding for very large corpora.
