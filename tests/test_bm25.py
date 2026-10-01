"""BM25 behaviour: relevant documents must actually rank first."""

import pytest

from src.rag.bm25 import BM25Index

CORPUS = [
    "Employees may expense travel up to 500 dollars per day for domestic trips.",
    "The international per diem cap is 300 dollars per day including meals.",
    "Submit expense reports within 30 days of the trip using the finance portal.",
    "Hotel bookings should be made through the corporate travel agency.",
]


@pytest.fixture
def index() -> BM25Index:
    return BM25Index.fit(CORPUS)


def test_relevant_doc_ranks_first(index):
    results = index.search("international per diem cap")
    assert results[0][0] == 1
    assert results[0][1] > 0


def test_exact_identifier_is_found(index):
    results = index.search("per diem")
    assert {doc for doc, _ in results} == {0, 1}


def test_results_are_sorted_descending(index):
    scores = [score for _, score in index.search("expense report trip")]
    assert scores == sorted(scores, reverse=True)


def test_empty_corpus_returns_no_results():
    assert BM25Index.fit([]).search("anything") == []


def test_out_of_vocabulary_query_returns_empty_not_a_crash(index):
    assert index.search("zzzz nonexistenttoken") == []


def test_matched_docs_carry_positive_scores_only(index):
    for _, score in index.search("hotel corporate travel agency"):
        assert score > 0


@pytest.mark.parametrize("k", [0, -1])
def test_non_positive_k_returns_empty(index, k):
    assert index.search("expense", k=k) == []


def test_k_limits_the_result_count(index):
    assert len(index.search("expense travel per diem cap report", k=2)) <= 2


def test_repeated_query_terms_do_not_change_ranking(index):
    once = index.search("per diem cap")
    twice = index.search("per diem cap per diem cap")
    assert [doc for doc, _ in once] == [doc for doc, _ in twice]


def test_punctuation_only_query_returns_empty(index):
    assert index.search("!!! ???") == []


def test_index_is_not_mutated_by_search(index):
    before = index.search("expense report")
    index.search("hotel booking agency corporate")
    assert index.search("expense report") == before
