# Observability

The service exposes Prometheus metrics for the retrieval/RAG path, with a ready-made
Grafana dashboard.

## Metrics

| Metric | Type | Meaning |
|---|---|---|
| `scs_searches_total{mode}` | counter | searches, by `dense`/`hybrid` |
| `scs_search_latency_seconds{mode}` | histogram | end-to-end search latency |
| `scs_top_score` | histogram | top retrieval score per search (relevance signal) |
| `scs_rag_requests_total{outcome}` | counter | RAG requests by `answered` / `declined` (fallback rate) |

## Run it

```bash
# 1. expose metrics from the MCP server
METRICS_ENABLED=on semantic-index-mcp        # /metrics on :9464

# 2. scrape with Prometheus
docker run -p 9090:9090 -v "$PWD/observability/prometheus.yml:/etc/prometheus/prometheus.yml" prom/prometheus

# 3. Grafana -> add the Prometheus datasource -> import observability/grafana-dashboard.json
docker run -p 3000:3000 grafana/grafana
```

The dashboard shows search rate by mode, p95 latency, median top-score, and the RAG
fallback rate — the same signals you'd watch to keep a retrieval service healthy.
