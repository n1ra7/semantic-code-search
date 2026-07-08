from eval.metrics import (
    dedupe_preserving_order,
    hit_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)

RETRIEVED = ["a.py", "b.py", "a.py", "c.py", "d.py"]  # note the duplicate
RELEVANT = {"c.py", "d.py"}


def test_dedupe_preserving_order():
    assert dedupe_preserving_order(RETRIEVED) == ["a.py", "b.py", "c.py", "d.py"]


def test_precision_and_recall_at_k():
    ded = dedupe_preserving_order(RETRIEVED)  # a, b, c, d
    assert precision_at_k(ded, RELEVANT, 4) == 0.5   # 2 of top-4 relevant
    assert recall_at_k(ded, RELEVANT, 4) == 1.0      # both relevant found
    assert recall_at_k(ded, RELEVANT, 2) == 0.0      # neither in top-2


def test_hit_and_reciprocal_rank():
    ded = dedupe_preserving_order(RETRIEVED)
    assert hit_at_k(ded, RELEVANT, 3) == 1.0         # c.py is at rank 3
    assert hit_at_k(ded, RELEVANT, 2) == 0.0
    assert reciprocal_rank(ded, RELEVANT) == 1.0 / 3  # first relevant at rank 3


def test_ndcg_ordering_matters():
    perfect = ["c.py", "d.py", "a.py"]
    worse = ["a.py", "c.py", "d.py"]
    assert ndcg_at_k(perfect, RELEVANT, 3) > ndcg_at_k(worse, RELEVANT, 3)
    assert ndcg_at_k(perfect, RELEVANT, 3) == 1.0


def test_empty_and_no_relevant():
    assert precision_at_k([], RELEVANT, 5) == 0.0
    assert recall_at_k(["a.py"], set(), 5) == 0.0
    assert reciprocal_rank(["x.py"], RELEVANT) == 0.0
