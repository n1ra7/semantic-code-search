"""Prometheus metrics for the retrieval/RAG path.

Instruments the query pipeline so the service is observable in production: request
counts, latency, retrieval confidence, and how often the RAG layer declines to answer
(the hallucination-fallback rate). Recording is always-on and cheap; the metrics are
only exposed over HTTP when the metrics server is started.
"""
from __future__ import annotations

from typing import Optional

from prometheus_client import Counter, Histogram, start_http_server

from .config import settings

SEARCHES = Counter("scs_searches_total", "Total searches", ["mode"])
SEARCH_LATENCY = Histogram("scs_search_latency_seconds", "Search latency (s)", ["mode"])
TOP_SCORE = Histogram(
    "scs_top_score",
    "Top retrieval score of each search",
    buckets=(0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
)
RAG_REQUESTS = Counter("scs_rag_requests_total", "RAG requests by outcome", ["outcome"])


def record_search(mode: str, latency: float, top_score: Optional[float]) -> None:
    SEARCHES.labels(mode).inc()
    SEARCH_LATENCY.labels(mode).observe(latency)
    if top_score is not None:
        TOP_SCORE.observe(top_score)


def record_rag(outcome: str) -> None:
    """outcome: 'answered' or 'declined' (fallback fired)."""
    RAG_REQUESTS.labels(outcome).inc()


def start_metrics_server(port: int | None = None) -> None:
    start_http_server(port or settings.metrics_port)
