"""Shared test fixtures.

The corpus and its embedder live here because both the pipeline tests and the
eval tests need the same deterministic setup, and duplicating them would let the
two drift apart.
"""

import numpy as np
import pytest
from sklearn.feature_extraction.text import TfidfVectorizer

from src.rag.embeddings import Embedder, l2_normalize
from src.rag.pipeline import RAGSystem

#: Three documents with deliberately non-overlapping vocabularies, so each
#: answerable question has exactly one correct source document.
DOCS = [
    (
        "expense-policy",
        "International travel is reimbursed up to 300 dollars per day including "
        "meals. Domestic travel is reimbursed up to 500 dollars per day. The "
        "international per diem cap applies to every trip crossing a border and "
        "must be approved before departure.",
    ),
    (
        "security-policy",
        "Passwords must be rotated every 90 days and may not be shared with "
        "colleagues. Lost hardware must be reported to the security desk within "
        "24 hours of discovery.",
    ),
    (
        "leave-policy",
        "Annual leave accrues at 22 days per year for permanent staff and may be "
        "carried over by up to 5 unused days into the following calendar year.",
    ),
]


class BagOfWordsEmbedder(Embedder):
    """Deterministic lexical embedder: TF-IDF vectors over the fitted corpus.

    Not semantic -- "holiday" and "vacation" are unrelated to it. But it is real
    lexical signal, which is what makes an offline eval assertable without a
    model download.

    ponytail: keeps a dense vocabulary-sized matrix. A real corpus (thousands of
    chunks) would want sparse storage.
    """

    dim = 4096

    def __init__(self) -> None:
        self._vectorizer: TfidfVectorizer | None = None

    def fit(self, texts: list[str]) -> "BagOfWordsEmbedder":
        fitted = BagOfWordsEmbedder()
        fitted._vectorizer = TfidfVectorizer(lowercase=True, ngram_range=(1, 2)).fit(
            texts or ["placeholder"]
        )
        fitted.dim = max(len(fitted._vectorizer.vocabulary_), 1)
        return fitted

    def embed(self, texts: list[str]) -> np.ndarray:
        if self._vectorizer is None:
            raise RuntimeError("BagOfWordsEmbedder must be fitted before embedding")
        if not texts:
            return np.zeros((0, self.dim))
        return l2_normalize(self._vectorizer.transform(texts).toarray())


#: Calibrated cosine floor. Answerable questions here top out at ~0.41 best
#: similarity; the unanswerable probes sit at ~0.18-0.20, purely from stopword
#: bigram overlap. 0.25 sits in that gap.
#:
#: ponytail: this threshold is tuned to this corpus and this embedder. A real
#: project recalibrates it per corpus from the eval scorecard -- see
#: `pipeline.DEFAULT_MIN_SIMILARITY` for the procedure.
MIN_SIMILARITY = 0.25


@pytest.fixture
def embedder() -> Embedder:
    return BagOfWordsEmbedder()


@pytest.fixture
def system(embedder: Embedder) -> RAGSystem:
    return RAGSystem.create(embedder=embedder, min_similarity=MIN_SIMILARITY).ingest_documents(
        DOCS
    )
