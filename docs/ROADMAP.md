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

Informed by the [ablation results](../README.md#a-measured-ablation-gitea), which found AST chunking
regressed *feature-level* recall (per-function chunks lose surrounding context):

- **Isolate chunking from retrieval** — a fairer ablation that holds chunking fixed (line) and adds
  hybrid → reranking, to measure their contribution independent of the chunking regression.
- **Code-specialized embeddings** — evaluate with `jinaai/jina-embeddings-v2-base-code` (the default),
  expected to lift every row versus the general-purpose model used in the first run.
- **Context-preserving AST chunks** — include a definition's signature/docstring and a small window of
  surrounding context, to keep AST's structure benefits without losing feature-query recall.

## Deliberately out of scope

To keep the project focused (it indexes a single codebase, not a massive document store):
- Source-confidence scoring by freshness/trust — a better fit for news/document RAG than code.
- Response/embedding caching — low value at this scale.
- Horizontal scaling / sharding for very large corpora.
