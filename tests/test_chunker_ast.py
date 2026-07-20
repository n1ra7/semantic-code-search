from semantic_index.chunker_ast import chunk_file_ast

PY = '''\
import os
import sys

def top_level():
    return 1

class Foo:
    def method_a(self):
        return "a"

    def method_b(self):
        return "b"
'''


def test_ast_chunks_on_definition_boundaries():
    chunks = chunk_file_ast("sample.py", PY)
    # Every chunk is coherent and language-tagged.
    assert chunks
    assert all(c.language == "python" for c in chunks)
    # The top-level function is captured as its own chunk, on its real line range.
    fn = [c for c in chunks if "def top_level" in c.text]
    assert fn, "expected the top-level function to be its own chunk"
    assert fn[0].start_line == 4

    # The class (small enough to fit) is captured whole, including its methods.
    cls = [c for c in chunks if "class Foo" in c.text]
    assert cls, "expected the class to be captured"
    assert "method_a" in cls[0].text and "method_b" in cls[0].text

    # The imports before the first definition become a preamble chunk.
    pre = [c for c in chunks if "import os" in c.text]
    assert pre, "expected a preamble chunk with the imports"


def test_unsupported_language_falls_back_to_line_window():
    # .txt has no grammar -> should not raise, and should still return chunks
    text = "\n".join(f"line {i}" for i in range(150))
    chunks = chunk_file_ast("notes.txt", text)
    assert chunks
    assert all(c.language == "text" for c in chunks)


def test_empty_file_returns_no_chunks():
    assert chunk_file_ast("empty.py", "") == []
