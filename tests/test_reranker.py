from semantic_index.reranker import apply_rerank

HITS = [
    {"path": "a.py", "text": "alpha", "score": 0.9},
    {"path": "b.py", "text": "beta", "score": 0.8},
    {"path": "c.py", "text": "gamma", "score": 0.7},
]


def test_apply_rerank_reorders_by_score_and_truncates():
    # First-stage order is a, b, c; reranker prefers c, then a, then b.
    scores = [0.2, 0.1, 0.9]
    out = apply_rerank(HITS, scores, limit=2)
    assert [h["path"] for h in out] == ["c.py", "a.py"]
    assert out[0]["rerank_score"] == 0.9
    # original hit dicts are not mutated
    assert "rerank_score" not in HITS[0]


def test_apply_rerank_keeps_all_when_limit_large():
    scores = [0.1, 0.3, 0.2]
    out = apply_rerank(HITS, scores, limit=10)
    assert [h["path"] for h in out] == ["b.py", "c.py", "a.py"]
