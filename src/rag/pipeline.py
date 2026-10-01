"""End-to-end wiring: ingest -> chunk -> index -> retrieve -> answer.

Why a system class at all: the individual stages are each a few lines of numpy,
but a RAG pipeline is only as coherent as the seam between them. Keeping the
wiring in one small class makes the eval honest -- it exercises exactly the path a
user question takes, with no stage skipped and no oracle injected.

Ingestion is immutable: :meth:`RAGSystem.ingest_documents` returns a *new*
system rather than mutating ``self``, so a partially ingested index can never be
queried and test cases cannot leak state into each other.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from .answer import Answerer, EchoAnswerer
from .chunking import chunk_text
from .embeddings import Embedder, get_embedder
from .hybrid import HybridRetriever

#: Default cosine floor for the abstention gate.
#:
#: Deliberately non-zero. A retriever that always returns *something* is a
#: hallucination factory, and abstention is the headline behaviour of this project,
#: so it must not ship disabled. The cost is that the offline `hash` embedder --
#: which produces orthogonal vectors for unrelated strings -- abstains on almost
#: everything. That is the safe direction to fail: garbage in, refuse to answer.
#:
#: Recalibrate this per corpus once a real embedding model is wired in (see
#: ``rag.embeddings``): use the eval scorecard, and take the lowest floor that still
#: abstains on 100% of unanswerable questions without costing answerable recall.
#: The scorecard's similarity distribution is what sets it, not a round number.
DEFAULT_MIN_SIMILARITY = 0.15


@dataclass(frozen=True)
class RAGSystem:
    """A complete, immutable retrieve-then-answer pipeline over ingested documents.

    Attributes:
        chunks: chunk texts, flattened across documents.
        source_ids: source document id for each chunk, parallel to ``chunks``.
        retriever: hybrid retriever over those chunks.
        answerer: the answering strategy.
    """

    chunks: tuple[str, ...]
    source_ids: tuple[str, ...]
    retriever: HybridRetriever
    answerer: Answerer
    chunk_size: int = 500
    chunk_overlap: int = 100

    @classmethod
    def create(
        cls,
        embedder: Embedder | None = None,
        answerer: Answerer | None = None,
        chunk_size: int = 500,
        chunk_overlap: int = 100,
        min_similarity: float = DEFAULT_MIN_SIMILARITY,
    ) -> RAGSystem:
        """Build an empty system with the given components wired up."""
        embedder = embedder or get_embedder("hash")
        return cls(
            chunks=(),
            source_ids=(),
            retriever=HybridRetriever.build([], [], embedder, min_similarity),
            answerer=answerer or EchoAnswerer(),
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

    def ingest_documents(self, docs: list[tuple[str, str]]) -> RAGSystem:
        """Return a new system indexed over ``(doc_id, text)`` pairs.

        Replaces any previously ingested corpus. Documents whose text chunks to
        nothing (empty or whitespace-only) are skipped rather than contributing
        an empty chunk.
        """
        chunk_texts: list[str] = []
        chunk_sources: list[str] = []
        for doc_id, text in docs:
            for chunk in chunk_text(
                text, size=self.chunk_size, overlap=self.chunk_overlap
            ):
                chunk_texts.append(chunk)
                chunk_sources.append(doc_id)

        return replace(
            self,
            chunks=tuple(chunk_texts),
            source_ids=tuple(chunk_sources),
            retriever=HybridRetriever.build(
                chunk_texts, chunk_sources, self.retriever.embedder, self.retriever.min_similarity
            ),
        )

    def retrieve(self, question: str, k: int = 5) -> list[dict[str, str | float]]:
        """Return the top ``k`` hits as ``{source_id, chunk, score}`` dicts."""
        hits = self.retriever.search(question, k=k)
        return [
            {
                "source_id": source_id,
                "chunk": self.chunks[index],
                "score": score,
            }
            for index, score, source_id in hits
        ]

    def ask(self, question: str, k: int = 5) -> dict:
        """Answer ``question`` from the ingested corpus.

        Returns the answerer's dict plus the retrieval trace: the source ids that
        were retrieved, the context chunks themselves, and an ``abstained`` flag.
        Retrieval is empty-first: with nothing above ``min_similarity`` the
        answerer is called with no contexts and must decline.
        """
        hits = self.retrieve(question, k=k)
        answer = self.answerer.answer(question, [str(hit["source_id"]) for hit in hits])
        return {
            **answer,
            "retrieved_source_ids": [str(hit["source_id"]) for hit in hits],
            "contexts": [str(hit["chunk"]) for hit in hits],
        }


__all__ = ["DEFAULT_MIN_SIMILARITY", "RAGSystem"]
