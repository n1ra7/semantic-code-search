from semantic_index.rag import INSUFFICIENT_EVIDENCE, RagChat, has_sufficient_evidence


def hit(score):
    return {"path": "a.py", "start_line": 1, "end_line": 5, "score": score, "text": "code"}


class FakeSearcher:
    def __init__(self, hits):
        self._hits = hits

    def search(self, query, limit=6, language=None):
        return self._hits


class FakeLLM:
    def __init__(self):
        self.called = False

    def generate(self, prompt, system=None):
        self.called = True
        return "generated answer"


def test_has_sufficient_evidence():
    assert has_sufficient_evidence([hit(0.5)], 0.2) is True
    assert has_sufficient_evidence([hit(0.1)], 0.2) is False
    assert has_sufficient_evidence([], 0.2) is False


def test_ask_falls_back_when_low_confidence():
    llm = FakeLLM()
    chat = RagChat(searcher=FakeSearcher([hit(0.05)]), llm=llm, min_score=0.2)
    ans = chat.ask("something obscure")
    assert ans.answered is False
    assert ans.answer == INSUFFICIENT_EVIDENCE
    assert llm.called is False  # generation was skipped — no hallucination


def test_ask_answers_when_confident():
    llm = FakeLLM()
    chat = RagChat(searcher=FakeSearcher([hit(0.8)]), llm=llm, min_score=0.2)
    ans = chat.ask("a real question")
    assert ans.answered is True
    assert ans.answer == "generated answer"
    assert llm.called is True
    assert ans.sources[0]["path"] == "a.py"
