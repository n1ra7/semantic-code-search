"""Retrieval-augmented generation: answer a question grounded in retrieved code.

Retrieve top-K chunks from the index, assemble them as cited context, and ask a
local LLM to answer using only that context. The answer carries its source
chunks (path:line-range) so claims are traceable back to real code.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from .config import settings
from .llm import OllamaClient
from .search import Searcher

SYSTEM_PROMPT = (
    "You are a precise coding assistant. Answer the question using ONLY the provided "
    "code context. Cite the files you use as `path:start-end`. If the context is not "
    "sufficient to answer, say so plainly instead of guessing."
)

INSUFFICIENT_EVIDENCE = (
    "I don't have enough relevant code in the index to answer that confidently. "
    "Try rephrasing, or index the repository that contains the answer."
)


def has_sufficient_evidence(hits: List[dict], min_score: float) -> bool:
    """Gate generation on retrieval confidence: require at least one hit whose top
    retrieval score clears the threshold. Pure function so it's unit-testable."""
    return bool(hits) and hits[0].get("score", 0.0) >= min_score


@dataclass
class Answer:
    question: str
    answer: str
    sources: List[dict]
    context: str
    answered: bool = True  # False when the fallback declined to generate


class RagChat:
    def __init__(
        self,
        searcher: Searcher | None = None,
        llm: OllamaClient | None = None,
        min_score: float | None = None,
    ) -> None:
        self.searcher = searcher or Searcher()
        self.llm = llm or OllamaClient()
        self.min_score = settings.fallback_min_score if min_score is None else min_score

    @staticmethod
    def build_context(hits: List[dict]) -> str:
        return "\n\n".join(
            f"# {h['path']}:{h['start_line']}-{h['end_line']}\n{h['text']}" for h in hits
        )

    def ask(self, question: str, k: int = 6, language: Optional[str] = None) -> Answer:
        hits = self.searcher.search(question, limit=k, language=language)

        sources = [
            {
                "path": h["path"],
                "start_line": h["start_line"],
                "end_line": h["end_line"],
                "score": h["score"],
            }
            for h in hits
        ]

        # Hallucination fallback: if retrieval isn't confident, decline instead of guessing.
        if not has_sufficient_evidence(hits, self.min_score):
            return Answer(
                question=question,
                answer=INSUFFICIENT_EVIDENCE,
                sources=sources,
                context="",
                answered=False,
            )

        context = self.build_context(hits)
        prompt = (
            f"Code context:\n\n{context}\n\n"
            f"Question: {question}\n\n"
            "Answer using only the context above, and cite the files you used:"
        )
        answer = self.llm.generate(prompt, system=SYSTEM_PROMPT)
        return Answer(question=question, answer=answer, sources=sources, context=context)
