from semantic_index.chunker import chunk_file, language_for


def test_language_for_known_and_unknown():
    assert language_for("app/main.py") == "python"
    assert language_for("svc/server.go") == "go"
    assert language_for("notes.unknownext") == "text"


def test_chunk_file_produces_overlapping_windows():
    text = "\n".join(f"line {i}" for i in range(200))
    chunks = chunk_file("a.py", text)
    assert chunks, "expected at least one chunk"
    assert all(c.language == "python" for c in chunks)
    assert chunks[0].start_line == 1
    # Consecutive chunks should overlap (start of next < end of current).
    if len(chunks) > 1:
        assert chunks[1].start_line <= chunks[0].end_line


def test_chunk_file_skips_blank_only_input():
    assert chunk_file("blank.py", "\n\n\n") == []
    assert chunk_file("empty.py", "") == []
