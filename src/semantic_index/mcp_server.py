"""MCP server exposing the semantic index to AI agents (Claude, etc.).

Speaks the Model Context Protocol over stdio, so any MCP-capable client can call
`search_code` and get grounded, top-K code snippets back. This mirrors how the index
is consumed in a real assistant workflow, without any proprietary integration code.
"""
from __future__ import annotations

from typing import List, Optional

from mcp.server.fastmcp import FastMCP

from .search import Searcher

mcp = FastMCP("semantic-code-search")

_searcher: Optional[Searcher] = None


def _get_searcher() -> Searcher:
    global _searcher
    if _searcher is None:
        _searcher = Searcher()  # lazy: load the model only when the first query arrives
    return _searcher


@mcp.tool()
def search_code(query: str, limit: int = 8, language: Optional[str] = None) -> List[dict]:
    """Semantic search over the indexed codebase.

    Args:
        query: Natural-language description or a code snippet to find similar code for.
        limit: Maximum number of results to return.
        language: Optional language filter (e.g. "python", "go").

    Returns a list of matches, each with path, start_line, end_line, language, score and text.
    """
    return _get_searcher().search(query, limit=limit, language=language)


def main() -> None:
    mcp.run()  # stdio transport


if __name__ == "__main__":
    main()
