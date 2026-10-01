"""Reciprocal rank fusion and the hybrid retriever that uses it.

Why fusion: keyword retrieval and vector retrieval fail in *different* places.
BM25 nails exact identifiers but dies on paraphrase; embeddings handle paraphrase
but treat "policy 4.2.1" as noise. Rather than pick one, rank with both and fuse.

Why *reciprocal* rank fusion specifically: it consumes only the ordinal position
of a document in each ranking, discarding the raw scores. That matters because
BM25 scores (unbounded, IDF-weighted) and cosine similarities (bounded 0..1) are
not on a comparable scale -- any weighted sum needs a tuning knob that silently
breaks when the corpus changes. RRF has no such knob and no scale to reconcile.

    RRF(d) = SUM over rankings:  1 / (k + rank_i(d))

where ``rank_i(d)`` is the 1-based position of ``d`` in ranking ``i`` and ``k``
(default 60, from the original Cormack et al. paper) damps the influence of the
very top ranks so no single retriever can dominate the fusion on its own.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .bm25 import BM25Index
from .embeddings import Embedder

#: Standard RRF damping constant. Small enough to matter, large enough that the
#: top few positions dominate.
DEFAULT_RRF_K = 60


def reciprocal_rank_fusion(
    rankings: list[list[tuple[int, float]]], k: int = DEFAULT_RRF_K
) -> list[tuple[int, float]]:
    """Fuse ranked result lists into one ranking by reciprocal rank.

    Args:
        rankings: one ranked ``(index, score)`` list per retriever, best first.
            The scores are ignored -- only positions count.
        k: damping constant; must be positive.

    Returns:
        ``(index, fused_score)`` pairs sorted by fused score descending, ties
        broken by ascending index so the output is deterministic.
    """
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")

    fused: dict[int, float] = {}
    for ranking in rankings:
        # First position only: a repeated index inside one ranking would otherwise
        # score twice for a single retriever's opinion.
        seen: dict[int, None] = {}
        for index, _score in ranking:
            seen.setdefault(index, None)
        for rank, index in enumerate(seen, start=1):
            fused[index] = fused.get(index, 0.0) + 1.0 / (k + rank)

    return sorted(fused.items(), key=lambda pair: (-pair[1], pair[0]))


@dataclass(frozen=True)
class HybridRetriever:
    """BM25 + dense retrieval over a chunk corpus, fused with RRF.

    Immutable by construction: the chunk matrix is a single numpy array built at
    construction time, and both underlying indexes are read-only.

    The chunk matrix is stored L2-normalized, so cosine similarity against a
    query is one mat-vec product.
    """

    chunks: list[str]
    source_ids: list[str]
    bm25: BM25Index
    embedder: Embedder
    _matrix: np.ndarray
    min_similarity: float = 0.0

    @classmethod
    def build(
        cls,
        chunks: list[str],
        source_ids: list[str],
        embedder: Embedder,
        min_similarity: float = 0.0,
    ) -> HybridRetriever:
        """Index ``chunks`` for both keyword and vector retrieval."""
        if len(chunks) != len(source_ids):
            raise ValueError(
                f"chunks and source_ids must align: {len(chunks)} != {len(source_ids)}"
            )
        embedder = embedder.fit(chunks)
        matrix = embedder.embed(chunks) if chunks else np.zeros((0, embedder.dim))
        return cls(
            chunks=list(chunks),
            source_ids=list(source_ids),
            bm25=BM25Index.fit(list(chunks)),
            embedder=embedder,
            _matrix=matrix,
            min_similarity=min_similarity,
        )

    def search(self, query: str, k: int = 5) -> list[tuple[int, float, str]]:
        """Return up to ``k`` ``(chunk_index, fused_score, source_id)`` triples.

        ``min_similarity`` is the abstention hook: results whose best available
        evidence is below the threshold are withheld entirely, so an unrelated
        question yields no context and the answerer can decline. Note the floor
        applies to *cosine* similarity from the vector side only -- BM25 scores
        are unbounded and not comparable to a 0..1 threshold.
        """
        if k <= 0 or not self.chunks:
            return []

        vector_ranking = self._vector_search(query, k=len(self.chunks))

        # Abstention gate: with no chunk above the cosine floor, the corpus has
        # nothing to say and the answerer must be allowed to decline.
        if self.min_similarity > 0.0:
            vector_ranking = [
                pair for pair in vector_ranking if pair[1] >= self.min_similarity
            ]
            if not vector_ranking:
                return []
            above = {index for index, _ in vector_ranking}
        else:
            above = set(range(len(self.chunks)))

        keyword_ranking = [
            pair for pair in self.bm25.search(query, k=len(self.chunks))
            if pair[0] in above
        ]
        fused = reciprocal_rank_fusion([keyword_ranking, vector_ranking])
        return [(index, score, self.source_ids[index]) for index, score in fused[:k]]

    def _vector_search(self, query: str, k: int) -> list[tuple[int, float]]:
        """Rank chunks by cosine similarity to ``query``, best first."""
        if self._matrix.size == 0:
            return []
        query_vector = self.embedder.embed_one(query)
        similarities = self._matrix @ query_vector
        ranked = np.argsort(-similarities, kind="stable")[:k]
        return [(int(index), float(similarities[index])) for index in ranked]


__all__ = ["DEFAULT_RRF_K", "HybridRetriever", "reciprocal_rank_fusion"]
