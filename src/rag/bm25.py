"""Okapi BM25 keyword ranking, implemented directly on a term-count matrix.

Why: pure vector search fails on exact identifiers ("policy 4.2.1", "EXP-2024-17")
because those strings carry almost no semantic signal, which is why BM25 is still
in production everywhere. BM25 is the keyword half of the hybrid design, so the
formulas are written out here rather than pulled from a library: they are short,
and owning them is the point of the project.

Tokenization and the term-count matrix come from sklearn's ``CountVectorizer``;
the IDF and term-saturation scoring below are what BM25 adds over raw term
frequency. ``TfidfVectorizer`` is deliberately not used here -- its IDF variant is
not BM25's, and mixing the two would make the ranking unexplainable.

Score for document ``d`` and query ``q``:

    score(d, q) = SUM over t in q:  idf(t) * (tf(t,d) * (k1 + 1)) /
                                        (tf(t,d) + k1 * (1 - b + b * |d| / avgdl))

    idf(t)       = ln(1 + (N - df(t) + 0.5) / (df(t) + 0.5))

with the standard defaults ``k1 = 1.5`` (term-frequency saturation) and ``b = 0.75``
(document-length normalization).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

#: Standard Okapi BM25 parameters. k1 caps how much repeated terms help, b
#: controls how hard long documents are penalized (b=0 ignores length entirely).
DEFAULT_K1 = 1.5
DEFAULT_B = 0.75


@dataclass(frozen=True)
class BM25Index:
    """Immutable BM25 index over a fixed corpus.

    Build one with :meth:`fit`, which returns a new instance -- there is no
    ``mutate``, so a query can never observe a half-built index.
    """

    _tf: np.ndarray
    _idf: np.ndarray
    _doc_lengths: np.ndarray
    _avgdl: float
    _vocabulary: dict[str, int]
    _analyze: Callable[[str], list[str]] | None
    k1: float = DEFAULT_K1
    b: float = DEFAULT_B

    @classmethod
    def fit(
        cls,
        corpus: list[str],
        k1: float = DEFAULT_K1,
        b: float = DEFAULT_B,
    ) -> BM25Index:
        """Index ``corpus``. An empty corpus produces an index that never matches."""
        if not corpus:
            return cls(
                _tf=np.zeros((0, 0)),
                _idf=np.zeros(0),
                _doc_lengths=np.zeros(0),
                _avgdl=1.0,
                _vocabulary={},
                _analyze=None,
                k1=k1,
                b=b,
            )

        vectorizer = CountVectorizer(lowercase=True)
        matrix = vectorizer.fit_transform(corpus).toarray().astype(np.float64)
        doc_lengths = matrix.sum(axis=1)
        n_docs = float(len(corpus))
        document_frequency = (matrix > 0).sum(axis=0).astype(np.float64)
        # The +1 inside the log keeps idf non-negative even for ubiquitous terms.
        idf = np.log(1.0 + (n_docs - document_frequency + 0.5) / (document_frequency + 0.5))

        return cls(
            _tf=matrix,
            _idf=idf,
            _doc_lengths=doc_lengths,
            _avgdl=float(doc_lengths.mean()) or 1.0,
            _vocabulary=dict(vectorizer.vocabulary_),
            _analyze=vectorizer.build_analyzer(),
            k1=k1,
            b=b,
        )

    def search(self, query: str, k: int = 10) -> list[tuple[int, float]]:
        """Return the top ``k`` ``(doc_index, score)`` pairs, best first.

        Documented edge cases:
            - empty corpus, or a query with no term in the vocabulary: returns
              ``[]``. All-zero scores carry no ranking information, and feeding
              them into reciprocal rank fusion would inject meaningless ranks.
            - ties break by ascending document index (``kind="stable"`` sort),
              so output is deterministic run to run and platform to platform.
        """
        if k <= 0 or self._tf.size == 0 or self._analyze is None:
            return []

        columns = self._query_columns(query)
        if not columns:
            return []

        length_norm = 1.0 - self.b + self.b * (self._doc_lengths / self._avgdl)
        scores = np.zeros(len(self._doc_lengths), dtype=np.float64)
        for column in columns:
            tf = self._tf[:, column]
            denominator = tf + self.k1 * length_norm
            contribution = np.divide(
                tf * (self.k1 + 1.0),
                denominator,
                out=np.zeros_like(tf),
                where=denominator > 0,
            )
            scores += self._idf[column] * contribution

        # A document matching no query term scores exactly 0.0 and is dropped:
        # BM25 ranks matches, not the whole corpus.
        ranked = np.argsort(-scores, kind="stable")[:k]
        return [
            (int(index), float(scores[index]))
            for index in ranked
            if scores[index] > 0.0
        ]

    def _query_columns(self, query: str) -> list[int]:
        """Map query tokens to vocabulary columns, de-duplicated and ordered.

        Ordered so the accumulation order -- and therefore floating-point
        summation -- is identical for identical queries.
        """
        assert self._analyze is not None  # guarded by search()
        vocabulary = self._vocabulary
        return sorted(
            {vocabulary[token] for token in self._analyze(query) if token in vocabulary}
        )


__all__ = ["BM25Index", "DEFAULT_B", "DEFAULT_K1"]
