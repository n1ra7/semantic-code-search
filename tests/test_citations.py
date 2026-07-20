from semantic_index.rag import SYSTEM_PROMPT, RagChat


def test_context_labels_each_chunk_with_citation_tag():
    hits = [
        {"path": "services/pull/merge.go", "start_line": 10, "end_line": 25, "text": "func Merge() {}"},
        {"path": "models/auth/twofactor.go", "start_line": 1, "end_line": 8, "text": "type TwoFactor struct{}"},
    ]
    context = RagChat.build_context(hits)
    assert "[services/pull/merge.go:10-25]" in context
    assert "[models/auth/twofactor.go:1-8]" in context
    # the code text is present under its tag
    assert "func Merge()" in context


def test_system_prompt_requires_inline_citations():
    lowered = SYSTEM_PROMPT.lower()
    assert "inline citation" in lowered
    assert "square brackets" in lowered
    assert "cannot cite" in lowered
