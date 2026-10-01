"""Grounded answering with forced citation and explicit abstention.

Why this module exists: a RAG system that always answers is worse than useless on
a policy corpus -- a confidently wrong expense limit is the kind of error that
costs real money. So the contract here is narrow and deliberately unfriendly to
hallucination:

    * No context, no answer. The system says it does not know.
    * Every answer carries the source ids of the chunks it was given. The
      grounding constraint is structural, not a prompt instruction to be obeyed.
    * The answerer never invents a citation. ``cited_sources`` is copied from the
      retrieved contexts, so it cannot exceed what retrieval actually returned.

--------------------------------------------------------------------------
Plugging in a real LLM
--------------------------------------------------------------------------
:class:`EchoAnswerer` is the offline stand-in. To use a real model, subclass
:class:`Answerer` and implement ``answer``; the one place a model call belongs
is where the ``prompt`` string is built in :meth:`EchoAnswerer.answer`::

    class OpenAIAnswerer(Answerer):
        def answer(self, question, contexts):
            prompt = build_prompt(question, contexts)   # <-- the seam
            raw = client.chat.completions.create(...)    # network, opt-in
            return {"answer": raw, "cited_sources": ..., "abstained": False}

Keep the "abstain when ``contexts`` is empty" branch and keep citations derived
from ``contexts``: those two behaviours are the eval's correctness guarantees, and
an LLM is the least reliable place to enforce either.

This module makes no network calls of any kind.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

ABSTENTION_TEXT = (
    "I don't know. The provided documents do not contain an answer to that question."
)


class Answerer(ABC):
    """Produces a grounded answer from retrieved context."""

    @abstractmethod
    def answer(self, question: str, contexts: list[str]) -> dict:
        """Return ``{"answer": str, "cited_sources": list[str], "abstained": bool}``.

        ``contexts`` are the retrieved source ids, already ordered by relevance.
        Implementations must abstain when ``contexts`` is empty.
        """


def build_prompt(question: str, contexts: list[str]) -> str:
    """Assemble the grounded-answer prompt. Pure string building, no model call.

    This is the single function a real LLM integration wraps: build the prompt
    here, send it, read the reply back. Keeping it pure means the prompt can be
    asserted on in tests without a network or an API key.
    """
    numbered = "\n".join(f"[{i + 1}] {context}" for i, context in enumerate(contexts))
    return (
        "Answer the question using only the numbered context below.\n"
        "Cite the context numbers you used. If the context does not answer the "
        "question, reply exactly: I don't know.\n\n"
        f"Context:\n{numbered}\n\n"
        f"Question: {question}\n"
        "Answer:"
    )


class EchoAnswerer(Answerer):
    """Deterministic offline stand-in for an LLM.

    It never fabricates an answer; it reports that the real model is not wired up
    and echoes the grounding it was given. That keeps the surrounding pipeline --
    retrieval, fusion, abstention, citation plumbing, and the eval harness --
    testable without a model, which is the whole point until a key exists.
    """

    def answer(self, question: str, contexts: list[str]) -> dict:
        if not contexts:
            return {"answer": ABSTENTION_TEXT, "cited_sources": [], "abstained": True}

        return {
            "answer": (
                f"[offline echo] No LLM is configured, so this is not a real answer. "
                f"Question: {question!r} would be answered from {len(contexts)} "
                f"retrieved chunk(s). Prompt skeleton: "
                f"{build_prompt(question, contexts).splitlines()[0]!r}"
            ),
            "cited_sources": list(contexts),
            "abstained": False,
        }


__all__ = ["ABSTENTION_TEXT", "Answerer", "EchoAnswerer", "build_prompt"]
