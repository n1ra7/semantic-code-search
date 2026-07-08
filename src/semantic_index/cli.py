"""Command-line interface: `semantic-index index <path>` and `semantic-index search <query>`."""
from __future__ import annotations

import json
from typing import Optional

import typer

from .indexer import Indexer
from .search import Searcher

app = typer.Typer(add_completion=False, help="Semantic code search — index a repo and search it.")


@app.command()
def index(path: str = typer.Argument(".", help="Path to the repository to index.")) -> None:
    """Index (or incrementally re-index) a repository into the vector store."""
    stats = Indexer().index(path)
    typer.echo(json.dumps(stats, indent=2))


@app.command()
def search(
    query: str = typer.Argument(..., help="Natural-language or code query."),
    limit: int = typer.Option(8, help="Number of results."),
    language: Optional[str] = typer.Option(None, help="Filter by language, e.g. python."),
) -> None:
    """Semantic search over the indexed codebase."""
    hits = Searcher().search(query, limit=limit, language=language)
    if not hits:
        typer.echo("No results. Have you run `semantic-index index <path>` yet?")
        raise typer.Exit(code=0)
    for hit in hits:
        typer.secho(
            f"[{hit['score']:.3f}] {hit['path']}:{hit['start_line']}-{hit['end_line']} ({hit['language']})",
            fg=typer.colors.CYAN,
        )
        preview = "\n".join(hit["text"].splitlines()[:6])
        typer.echo(preview)
        typer.echo("-" * 70)


if __name__ == "__main__":
    app()
