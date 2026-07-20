from eval.answerability import evaluate_answerability


class FakeSearcher:
    def __init__(self, scores):
        self.scores = scores

    def search(self, query, limit=5, language=None):
        return [{"path": "x.py", "start_line": 1, "end_line": 2, "score": self.scores.get(query, 0.0), "text": "c"}]


DATASET = [
    {"query": "pos1", "relevant": ["x.py"], "negative": False},
    {"query": "pos2", "relevant": ["x.py"], "negative": False},
    {"query": "neg1", "relevant": [], "negative": True},
    {"query": "neg2", "relevant": [], "negative": True},
]


def test_answerability_perfect():
    scores = {"pos1": 0.8, "pos2": 0.6, "neg1": 0.05, "neg2": 0.1}
    m = evaluate_answerability(FakeSearcher(scores), DATASET, min_score=0.2)
    assert m["positives_answered_rate"] == 1.0
    assert m["negatives_declined_rate"] == 1.0
    assert m["false_answers"] == 0


def test_answerability_flags_false_answer():
    # A negative query that retrieval is (wrongly) confident about -> false answer.
    scores = {"pos1": 0.8, "pos2": 0.6, "neg1": 0.9, "neg2": 0.05}
    m = evaluate_answerability(FakeSearcher(scores), DATASET, min_score=0.2)
    assert m["negatives_declined_rate"] == 0.5
    assert m["false_answers"] == 1
