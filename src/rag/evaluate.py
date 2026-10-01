"""Ground-truth eval harness for the retrieval + answering loop.

Why this module is the project: "I built a RAG chatbot" is a demo. A retriever
that measurably surfaces the right chunk on a hand-written question set, and that
visibly declines questions the corpus cannot answer, is a system you can defend
in an interview.

Three numbers come out of here:

    retriever_recall_at_k
        Of the answerable questions, how often did the expected source document
        land in the top-k? This is the metric you own -- the model is not
        interchangeable, the retriever is.

    answer_accuracy
        Of the answerable questions, how often did the system answer instead of
        declining? See the caveat on :meth:`Evaluator.summary`: with the offline
        echo answerer this measures grounding behaviour, not answer quality, and
        the README is explicit that real answers get read by a human.

    abstention_rate_on_unanswerable
        Of the questions the corpus cannot answer, how often did it decline? A
        system that answers all five hallucination probes is worse than useless,
        so this is the number that protects the other two.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

import pandas as pd

from .answer import Answerer


@dataclass(frozen=True)
class EvalCase:
    """One ground-truth question.

    Attributes:
        question: what is asked.
        expected_answer: human-written correct answer, for manual review of the
            scorecard. Not used for automatic scoring by the echo answerer.
        expected_source_id: source document the answer must come from. Used for
            retriever recall. ``None`` for unanswerable cases.
        answerable: ``False`` marks a hallucination probe -- the system must
            decline these.
    """

    question: str
    expected_answer: str
    expected_source_id: str | None
    answerable: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)


class System(Protocol):
    """The surface the harness needs. :class:`~rag.pipeline.RAGSystem` satisfies it."""

    def ask(self, question: str) -> dict: ...


@dataclass(frozen=True)
class Evaluator:
    """Runs an :class:`EvalCase` list against a RAG system and scores it."""

    system: System
    answerer: Answerer
    k: int = 5

    def run(self, cases: list[EvalCase]) -> pd.DataFrame:
        """Score every case and return one row per question.

        Columns:
            question, answerable, expected_source_id, retrieved_source_ids,
            retriever_hit, abstained, correct_abstention, cited_sources

        ``retriever_hit`` is ``True`` when the expected source id appears in the
        top-k retrieved ids, and ``False`` for unanswerable cases (which have no
        expected source by definition). ``correct_abstention`` is strictly about
        unanswerable cases: it is ``True`` when such a case abstained and is
        ``None`` for answerable cases, where abstention is the *wrong* behaviour
        and reporting it as correct would flatter the system.
        """
        rows = [self._score(case) for case in cases]
        return pd.DataFrame(rows, columns=_COLUMNS)

    def summary(self, results: pd.DataFrame) -> dict[str, float]:
        """Aggregate a scorecard from :meth:`run`.

        Requires the DataFrame rather than caching it, so the metrics are always
        traceable to a specific run instead of to whatever ran last.
        """
        if results.empty:
            raise ValueError("summary() needs the DataFrame returned by run()")

        answerable = results[results["answerable"]]
        unanswerable = results[~results["answerable"]]

        return {
            "n_cases": int(len(results)),
            "retriever_recall_at_k": _rate(
                answerable["retriever_hit"].tolist(), denominator="answerable cases"
            ),
            "answer_accuracy": _rate(
                (~answerable["abstained"]).tolist(), denominator="answerable cases"
            ),
            "abstention_rate_on_unanswerable": _rate(
                unanswerable["abstained"].tolist(), denominator="unanswerable cases"
            ),
        }

    def _score(self, case: EvalCase) -> dict[str, Any]:
        result = self.system.ask(case.question)
        retrieved: list[str] = list(result.get("retrieved_source_ids", []))
        abstained: bool = bool(result.get("abstained", True))

        if case.answerable:
            hit = case.expected_source_id in retrieved[: self.k]
            correct_abstention: bool | None = None
        else:
            hit = False
            correct_abstention = abstained

        return {
            "question": case.question,
            "answerable": case.answerable,
            "expected_source_id": case.expected_source_id,
            "retrieved_source_ids": retrieved,
            "retriever_hit": hit,
            "abstained": abstained,
            "correct_abstention": correct_abstention,
            "cited_sources": list(result.get("cited_sources", [])),
        }


_COLUMNS = [
    "question",
    "answerable",
    "expected_source_id",
    "retrieved_source_ids",
    "retriever_hit",
    "abstained",
    "correct_abstention",
    "cited_sources",
]


def _rate(values: list[bool], denominator: str) -> float:
    """Mean of booleans as a float; ``0.0`` when there is nothing to average."""
    if not values:
        return 0.0
    return float(sum(values) / len(values))


__all__ = ["EvalCase", "Evaluator"]
