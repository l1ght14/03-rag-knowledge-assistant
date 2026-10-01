"""Pluggable text embeddings with a deterministic, offline default.

Why an abstraction instead of calling a model directly: a real embedding model
(sentence-transformers, OpenAI) needs a network call or a multi-GB download, and
a test suite that depends on either is slow, flaky, and unrunnable offline. So
this module defines the *seam*: :class:`Embedder` is the contract every backend
implements, and :class:`HashEmbedder` is the zero-dependency stand-in that makes
the whole pipeline testable offline.

``HashEmbedder`` is deterministic, not semantic. Identical text always maps to an
identical vector, so retrieval behaviour is reproducible, which is what the eval
harness needs. It has no notion of "cats" resembling "kittens" -- that is exactly
the capability a real model buys you, and exactly what the hybrid design already
hedges by fusing BM25 (which *is* lexical) with this vector side.

--------------------------------------------------------------------------
Swapping in a real model
--------------------------------------------------------------------------
Subclass :class:`Embedder`, implement ``embed`` to return an ``(n, dim)`` array,
and register it in :func:`get_embedder`::

    class SentenceTransformerEmbedder(Embedder):
        dim = 384

        def __init__(self) -> None:
            from sentence_transformers import SentenceTransformer  # optional dep
            self._model = SentenceTransformer("all-MiniLM-L6-v2")

        def embed(self, texts: list[str]) -> np.ndarray:
            vectors = self._model.encode(texts, convert_to_numpy=True)
            return l2_normalize(vectors.astype(np.float64))

Nothing else in the codebase changes: BM25, RRF, abstention, and the eval all
depend only on the :class:`Embedder` interface.
"""

from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod

import numpy as np


def l2_normalize(matrix: np.ndarray) -> np.ndarray:
    """Scale every row to unit length, leaving zero rows untouched.

    Unit-length rows turn cosine similarity into a plain dot product, which is
    what makes the vector side of hybrid retrieval a single ``@`` operation.
    """
    array = np.asarray(matrix, dtype=np.float64)
    if array.ndim == 1:
        array = array.reshape(1, -1)
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    # A zero vector has no direction; dividing by zero would emit NaNs.
    safe = np.where(norms == 0.0, 1.0, norms)
    return array / safe


class Embedder(ABC):
    """Contract for turning text into L2-normalized row vectors."""

    #: Dimensionality of the vectors this embedder produces.
    dim: int

    def fit(self, texts: list[str]) -> Embedder:
        """Return an embedder calibrated to this corpus, for backends that need it.

        Stateless backends (anything hashing) ignore the corpus and return
        ``self``. Corpus-dependent backends return a new fitted instance rather
        than mutating, which keeps index construction reproducible.
        """
        return self

    @abstractmethod
    def embed(self, texts: list[str]) -> np.ndarray:
        """Return an ``(len(texts), self.dim)`` float array with unit-norm rows.

        An empty input list returns an empty ``(0, self.dim)`` array so callers
        never have to special-case it.
        """

    def embed_one(self, text: str) -> np.ndarray:
        """Embed a single string and return its ``(dim,)`` vector."""
        return self.embed([text])[0]


class HashEmbedder(Embedder):
    """Deterministic pseudo-random embeddings. Offline, instant, no downloads.

    Each text is hashed (SHA-256, stable across processes unlike ``hash()``) and
    the digest seeds a ``numpy`` generator, so the same string always produces
    the same vector. Similarity between two unrelated texts is therefore ~0,
    which makes this a worst-case stand-in for a semantic model: if the pipeline
    works with this, it works with a real embedder.
    """

    def __init__(self, dim: int = 256) -> None:
        if dim <= 0:
            raise ValueError(f"dim must be positive, got {dim}")
        self.dim = dim

    def _vector(self, text: str) -> np.ndarray:
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        seed = int.from_bytes(digest[:8], "big")
        return np.random.default_rng(seed).standard_normal(self.dim)

    def embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float64)
        return l2_normalize(np.vstack([self._vector(text) for text in texts]))


def get_embedder(name: str = "hash", dim: int = 256) -> Embedder:
    """Factory for the available embedders.

    Only the offline hash embedder is registered. An unknown name raises
    ``NotImplementedError`` pointing at the two places a real model plugs in:
    add an ``Embedder`` subclass here (see the module docstring) and register it
    in this factory.
    """
    if name == "hash":
        return HashEmbedder(dim=dim)
    raise NotImplementedError(
        f"Unknown embedder {name!r}. Only 'hash' ships with this project. "
        "To add a real model, subclass rag.embeddings.Embedder (see that "
        "module's docstring for a sentence-transformers example) and register it "
        "in this factory."
    )
