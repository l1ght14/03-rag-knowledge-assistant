"""Reciprocal rank fusion and hybrid retrieval.

The fusion tests use hand-computed RRF arithmetic so the ranking is verifiable
by hand, not just "looks plausible". The hybrid tests use a stub embedder so the
vector side is controlled -- a hash embedder has no semantic signal by design.
"""

import numpy as np
import pytest

from src.rag.embeddings import Embedder, l2_normalize
from src.rag.hybrid import HybridRetriever, reciprocal_rank_fusion

#: One axis per chunk plus a spare axis for queries that should match nothing.
DIM = 4


class StubEmbedder(Embedder):
    """Returns one preset vector per known text; unknown texts raise KeyError.

    Raising rather than defaulting keeps the tests honest: if a query is not
    assigned a vector on purpose, that is a test bug, not a silent zero.
    """

    dim = DIM

    def __init__(self, table: dict[str, list[float]]) -> None:
        self._table = table

    def embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, DIM))
        return l2_normalize(np.array([self._table[text] for text in texts]))


# --- RRF arithmetic ---------------------------------------------------------

def test_single_ranking_is_reordered_by_reciprocal_rank():
    # Positions 1,2 -> 1/61 and 1/62, so index 0 stays ahead of index 1.
    fused = reciprocal_rank_fusion([[(0, 9.0), (1, 0.1)]], k=60)
    assert [doc for doc, _ in fused] == [0, 1]
    assert fused[0][1] == pytest.approx(1 / 61)
    assert fused[1][1] == pytest.approx(1 / 62)


def test_toy_two_list_fusion_matches_hand_computation():
    # ranking A: 3, 1, 2.   ranking B: 2, 3.
    fused = reciprocal_rank_fusion([[(3, 5.0), (1, 4.0), (2, 3.0)], [(2, 9.0), (3, 8.0)]], k=60)
    scores = dict(fused)
    assert scores[3] == pytest.approx(1 / 61 + 1 / 62)
    assert scores[2] == pytest.approx(1 / 63 + 1 / 61)
    assert scores[1] == pytest.approx(1 / 62)
    # 3 (0.032521) > 2 (0.032273) > 1 (0.016393): appearing twice beats rank 1 alone.
    assert [doc for doc, _ in fused] == [3, 2, 1]


def test_input_scores_are_ignored():
    low = reciprocal_rank_fusion([[(0, 0.001), (1, 999.0)]], k=60)
    high = reciprocal_rank_fusion([[(0, 1000.0), (1, 0.001)]], k=60)
    assert low == high


def test_document_in_two_rankings_outranks_a_single_top_hit():
    fused = reciprocal_rank_fusion([[(0, 1.0)], [(0, 1.0)]], k=60)
    assert fused == [(0, pytest.approx(2 / 61))]


def test_empty_rankings_yield_empty_fusion():
    assert reciprocal_rank_fusion([]) == []
    assert reciprocal_rank_fusion([[], []]) == []


def test_duplicate_index_within_one_ranking_is_counted_once():
    assert reciprocal_rank_fusion([[(0, 1.0), (0, 2.0)]], k=60) == [
        (0, pytest.approx(1 / 61))
    ]


def test_ties_break_on_ascending_index():
    assert [doc for doc, _ in reciprocal_rank_fusion([[(5, 1.0)], [(2, 1.0)]])] == [2, 5]


def test_non_positive_k_raises():
    with pytest.raises(ValueError):
        reciprocal_rank_fusion([[(0, 1.0)]], k=0)


def test_larger_k_compresses_the_score_spread():
    # Ranking A ranks 0 first; ranking B ranks 1 first. The gap between them
    # narrows as k grows, which is what stops one retriever from dominating.
    rankings = [[(0, 1.0), (1, 2.0)], [(1, 1.0)]]
    tight = dict(reciprocal_rank_fusion(rankings, k=10))
    loose = dict(reciprocal_rank_fusion(rankings, k=1000))
    assert abs(tight[0] - tight[1]) > abs(loose[0] - loose[1])


# --- HybridRetriever -------------------------------------------------------

# Axis i holds chunk i, so each chunk is maximally similar to its own query.
CHUNKS = [
    "the international per diem cap is 300 dollars per day",
    "domestic travel allowance is 500 dollars per day",
    "laptop refresh happens every three years for engineers",
]
SOURCES = ["expense", "expense", "hardware"]

#: Queries deliberately aligned with a single axis, plus one that matches nothing.
QUERIES = {
    "international per diem cap": [1.0, 0.0, 0.0, 0.0],  # chunk 0, both axes
    "domestic travel allowance": [0.0, 1.0, 0.0, 0.0],  # chunk 1, both axes
    "cap": [1.0, 0.0, 0.0, 0.0],  # chunk 0, both axes
    "laptop refresh engineers": [0.0, 0.0, 1.0, 0.0],  # chunk 2, both axes
    "something entirely elsewhere": [0.0, 0.0, 0.0, 1.0],  # orthogonal to all chunks
}


@pytest.fixture
def hybrid() -> HybridRetriever:
    table = dict(QUERIES)
    for axis, chunk in enumerate(CHUNKS):
        vector = [0.0] * DIM
        vector[axis] = 1.0
        table[chunk] = vector
    return HybridRetriever.build(CHUNKS, SOURCES, StubEmbedder(table))


def test_matching_on_both_axes_beats_matching_on_one(hybrid):
    # "cap" hits chunk 0 by keyword and by vector; chunk 2 matches neither.
    results = hybrid.search("cap", k=3)
    assert results[0][0] == 0
    assert results[0][2] == "expense"


def test_vector_only_match_surfaces_a_chunk_with_no_keyword_overlap(hybrid):
    # "international per diem cap" appears in chunk 0, so use an orthogonal
    # keyword term that only exists in chunk 2 to prove the vector side is live.
    results = hybrid.search("laptop refresh engineers", k=3)
    assert results[0][0] == 2


def test_search_returns_scores_and_source_ids(hybrid):
    for index, score, source_id in hybrid.search("cap", k=3):
        assert isinstance(index, int)
        assert isinstance(score, float)
        assert source_id in set(SOURCES)


def test_k_limits_results(hybrid):
    assert len(hybrid.search("cap", k=1)) == 1


def test_non_positive_k_returns_nothing(hybrid):
    assert hybrid.search("cap", k=0) == []


def test_empty_retriever_returns_nothing():
    assert HybridRetriever.build([], [], StubEmbedder({})).search("anything") == []


def test_mismatched_chunk_and_source_lengths_raise():
    with pytest.raises(ValueError):
        HybridRetriever.build(CHUNKS, ["only-one"], StubEmbedder({}))


def test_min_similarity_withholds_weak_matches():
    table = dict(QUERIES)
    for axis, chunk in enumerate(CHUNKS):
        vector = [0.0] * DIM
        vector[axis] = 1.0
        table[chunk] = vector
    strict = HybridRetriever.build(CHUNKS, SOURCES, StubEmbedder(table), min_similarity=0.5)

    # Cosine 0.0 against every chunk: nothing clears the floor, so nothing returns.
    assert strict.search("something entirely elsewhere", k=3) == []
    # Only chunk 0 clears the floor, so only chunk 0 is returned.
    assert [hit[0] for hit in strict.search("cap", k=3)] == [0]


def test_zero_min_similarity_disables_the_abstention_gate(hybrid):
    assert hybrid.search("something entirely elsewhere", k=3) != []
