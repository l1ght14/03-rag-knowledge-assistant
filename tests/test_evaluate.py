"""End-to-end eval on a small corpus with hand-verified ground truth.

This is the most important test in the project: it exercises ingest -> chunk ->
BM25 + vector -> RRF -> answer exactly as a user would, and asserts the two
claims that make the system defensible -- the retriever finds the right document
on every answerable question, and the system declines every question the corpus
cannot support.

The corpus (see ``conftest.py``) is written so each answerable question's terms
appear in exactly one document, which makes recall@1 a factual check rather than
a probabilistic one.
"""

import pandas as pd
import pytest

from src.rag.evaluate import EvalCase, Evaluator
from src.rag.pipeline import RAGSystem

from conftest import BagOfWordsEmbedder

ANSWERABLE = [
    ("what is the international per diem cap?", "expense-policy"),
    ("how much is the domestic travel reimbursement?", "expense-policy"),
    ("how often must passwords be rotated?", "security-policy"),
    ("what is the annual leave allowance for permanent staff?", "leave-policy"),
]

UNANSWERABLE = [
    "what is the parental leave policy for contractors?",
    "what is the dental insurance premium?",
]


def build_cases() -> list[EvalCase]:
    return [
        EvalCase(
            question=question,
            expected_answer="read the referenced policy document",
            expected_source_id=source_id,
            answerable=True,
        )
        for question, source_id in ANSWERABLE
    ] + [
        EvalCase(question=question, expected_answer="", expected_source_id=None, answerable=False)
        for question in UNANSWERABLE
    ]


@pytest.fixture
def evaluator(system: RAGSystem) -> Evaluator:
    return Evaluator(system=system, answerer=system.answerer, k=3)


@pytest.fixture
def results(evaluator: Evaluator) -> pd.DataFrame:
    return evaluator.run(build_cases())


@pytest.fixture
def summary(evaluator: Evaluator, results: pd.DataFrame) -> dict:
    return evaluator.summary(results)


def test_retriever_recall_at_k_is_perfect(summary):
    assert summary["retriever_recall_at_k"] == 1.0


def test_scorecard_covers_every_case(summary):
    assert summary["n_cases"] == len(ANSWERABLE) + len(UNANSWERABLE) == 6


def test_every_unanswerable_question_is_declined(results, summary):
    unanswerable = results[~results["answerable"]]
    assert len(unanswerable) == 2
    assert unanswerable["abstained"].all()
    assert unanswerable["correct_abstention"].all()
    assert summary["abstention_rate_on_unanswerable"] == 1.0


def test_answerable_questions_are_not_declined(results, summary):
    answerable = results[results["answerable"]]
    assert len(answerable) == 4
    assert not answerable["abstained"].any()
    assert summary["answer_accuracy"] == 1.0


def test_correct_abstention_is_none_for_answerable_cases(results):
    # Abstaining on an answerable question is a failure, not a correct abstention.
    assert results[results["answerable"]]["correct_abstention"].isna().all()


def test_retriever_hit_is_false_for_unanswerable_cases(results):
    # There is no expected source to hit, so the column must not credit them.
    assert not results[~results["answerable"]]["retriever_hit"].any()


def test_scorecard_has_the_documented_columns(results):
    assert {
        "question",
        "answerable",
        "expected_source_id",
        "retrieved_source_ids",
        "retriever_hit",
        "abstained",
        "correct_abstention",
        "cited_sources",
    }.issubset(set(results.columns))


def test_answers_only_cite_retrieved_sources(results):
    for row in results.itertuples():
        assert set(row.cited_sources).issubset(set(row.retrieved_source_ids))


def test_top_hit_matches_the_expected_source_for_every_answerable_case(results):
    answerable = results[results["answerable"]]
    top = [ids[0] for ids in answerable["retrieved_source_ids"]]
    assert top == [source for _, source in ANSWERABLE]


def test_unanswerable_questions_retrieve_nothing(results):
    for ids in results[~results["answerable"]]["retrieved_source_ids"]:
        assert ids == []


def test_ask_exposes_the_retrieval_trace(system):
    result = system.ask("what is the international per diem cap?", k=3)
    assert result["retrieved_source_ids"] == ["expense-policy"]
    assert result["contexts"]
    assert result["abstained"] is False


def test_unanswerable_questions_are_below_the_similarity_floor(system):
    # The abstentions above are not luck: the gate is what withholds the context.
    # "parental leave policy" is the honest hard case -- it overlaps "leave" with
    # a real document and still falls below the floor.
    assert system.ask("what is the parental leave policy for contractors?")["abstained"] is True
    assert system.ask("what is the dental insurance premium?")["abstained"] is True


def test_empty_system_declines_everything():
    empty = RAGSystem.create(embedder=BagOfWordsEmbedder())
    assert empty.ask("what is the international per diem cap?")["abstained"] is True


def test_summary_rejects_an_empty_frame(evaluator):
    with pytest.raises(ValueError):
        evaluator.summary(pd.DataFrame())
