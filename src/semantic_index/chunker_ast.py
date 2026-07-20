"""AST-aware chunking via tree-sitter.

Chunks source on syntactic boundaries — whole functions, methods, and classes —
instead of fixed line windows, so each chunk is a coherent unit of code. A definition
that fits within `max_chunk_lines` is emitted whole; one that's larger is broken into
its child definitions (e.g. a big class into its methods), and only line-windowed as a
last resort. Content before the first definition (imports, module-level code) becomes a
preamble chunk.

Falls back to the line-window chunker (`chunker.chunk_file`) for languages without a
grammar, parser errors, or files with no definitions — so indexing never fails.
"""
from __future__ import annotations

from functools import lru_cache
from typing import List, Optional

from .chunker import Chunk, chunk_file, language_for
from .config import settings

# Node types that represent a chunkable definition, per language.
DEF_NODES = {
    "python": {"function_definition", "class_definition"},
    "go": {"function_declaration", "method_declaration", "type_declaration"},
    "java": {"class_declaration", "interface_declaration", "enum_declaration",
             "method_declaration", "constructor_declaration"},
    "javascript": {"function_declaration", "generator_function_declaration",
                   "class_declaration", "method_definition"},
    "typescript": {"function_declaration", "class_declaration", "method_definition",
                   "interface_declaration", "abstract_class_declaration", "enum_declaration"},
    "c": {"function_definition", "struct_specifier", "enum_specifier"},
    "cpp": {"function_definition", "class_specifier", "struct_specifier"},
    "csharp": {"class_declaration", "interface_declaration", "struct_declaration",
               "method_declaration"},
    "rust": {"function_item", "impl_item", "struct_item", "trait_item", "enum_item"},
    "ruby": {"method", "singleton_method", "class", "module"},
}


@lru_cache(maxsize=None)
def _parser(language: str):
    try:
        from tree_sitter_language_pack import get_parser

        return get_parser(language)
    except Exception:
        return None


def chunk_file_ast(path: str, text: str) -> List[Chunk]:
    language = language_for(path)
    defset = DEF_NODES.get(language)
    parser = _parser(language) if defset else None
    if parser is None or not defset:
        return chunk_file(path, text)  # unsupported language -> line-window fallback

    try:
        tree = parser.parse(text.encode("utf-8", "ignore"))
    except Exception:
        return chunk_file(path, text)

    lines = text.splitlines()
    max_lines = max(1, settings.max_chunk_lines)
    chunks: List[Chunk] = []

    def add_span(start: int, end: int) -> None:  # 0-based inclusive line indices
        block = lines[start : end + 1]
        if any(line.strip() for line in block):
            chunks.append(Chunk(path, language, start + 1, end + 1, "\n".join(block)))

    def split_span(start: int, end: int) -> None:
        i = start
        while i <= end:
            j = min(i + max_lines - 1, end)
            add_span(i, j)
            i = j + 1

    def visit(node) -> None:
        if node.type in defset:
            start, end = node.start_point[0], node.end_point[0]
            if end - start + 1 <= max_lines:
                add_span(start, end)  # fits -> emit whole, don't descend (no overlap)
                return
            before = len(chunks)
            for child in node.children:  # too big -> emit child definitions
                visit(child)
            if len(chunks) == before:  # no nested defs -> line-window this node
                split_span(start, end)
            return
        for child in node.children:
            visit(child)

    visit(tree.root_node)

    if not chunks:
        return chunk_file(path, text)  # nothing definition-like -> fallback

    # Preamble: content before the first definition (imports / module-level code).
    first_def_line = min(c.start_line for c in chunks) - 1  # 0-based
    if first_def_line > 0 and any(line.strip() for line in lines[:first_def_line]):
        split_span(0, first_def_line - 1)

    chunks.sort(key=lambda c: (c.start_line, c.end_line))
    return chunks
