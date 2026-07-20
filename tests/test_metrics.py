from prometheus_client import generate_latest

from semantic_index.metrics import RAG_REQUESTS, SEARCHES, record_rag, record_search


def test_record_search_increments_counter():
    before = SEARCHES.labels("dense")._value.get()
    record_search("dense", latency=0.01, top_score=0.7)
    assert SEARCHES.labels("dense")._value.get() == before + 1


def test_record_rag_increments_by_outcome():
    before = RAG_REQUESTS.labels("declined")._value.get()
    record_rag("declined")
    assert RAG_REQUESTS.labels("declined")._value.get() == before + 1


def test_all_metric_names_are_exposed():
    record_search("hybrid", latency=0.02, top_score=0.5)
    record_rag("answered")
    out = generate_latest().decode()
    for name in (
        "scs_searches_total",
        "scs_search_latency_seconds",
        "scs_top_score",
        "scs_rag_requests_total",
    ):
        assert name in out
