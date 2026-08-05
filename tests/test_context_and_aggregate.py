from eval.aggregate import FileAggSearcher, aggregate_hits_by_file
from semantic_index.chunker import Chunk
from semantic_index.indexer import contextualize


def chunk(path):
    return Chunk(path=path, language="go", start_line=1, end_line=5, text="func X() {}")


def test_contextualize_none_returns_texts_unchanged():
    chunks, texts = [chunk("a/b.go")], ["func X() {}"]
    assert contextualize(chunks, texts, "none") is texts


def test_contextualize_path_prefixes_each_text():
    chunks = [chunk("services/auth/oauth2.go"), chunk("models/perm/access.go")]
    texts = ["func Login() {}", "func Check() {}"]
    out = contextualize(chunks, texts, "path")
    assert out[0] == "services/auth/oauth2.go\nfunc Login() {}"
    assert out[1].startswith("models/perm/access.go\n")


def hit(path, score, rerank=None):
    h = {"path": path, "score": score}
    if rerank is not None:
        h["rerank_score"] = rerank
    return h


def test_sum_topn_rewards_accumulated_evidence():
    # a.py: one strong chunk (0.5). b.py: three moderate chunks (0.3 each).
    hits = [hit("a.py", 0.5), hit("b.py", 0.3), hit("b.py", 0.3), hit("b.py", 0.3)]
    ranked = aggregate_hits_by_file(hits, top_n=3)
    assert ranked[0]["path"] == "b.py"  # 0.9 beats 0.5 (max pooling would pick a.py)
    assert abs(ranked[0]["score"] - 0.9) < 1e-9


def test_rerank_score_preferred_and_topn_caps():
    hits = [hit("a.py", 0.1, rerank=5.0), hit("b.py", 0.9), hit("b.py", 0.9),
            hit("b.py", 0.9), hit("b.py", 0.9)]
    ranked = aggregate_hits_by_file(hits, top_n=3)
    # b.py sums only its top-3 (2.7); a.py uses rerank_score 5.0
    assert ranked[0]["path"] == "a.py"
    assert abs(ranked[1]["score"] - 2.7) < 1e-9


def test_fileagg_searcher_wraps_and_limits():
    class Inner:
        def search(self, query, limit=8, language=None):
            return [hit("x.py", 0.4), hit("y.py", 0.2), hit("y.py", 0.3)]

    out = FileAggSearcher(Inner(), top_n=3, fetch=10).search("q", limit=1)
    assert len(out) == 1
    assert out[0]["path"] == "y.py"  # 0.5 accumulated beats 0.4
