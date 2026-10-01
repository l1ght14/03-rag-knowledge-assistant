"""The RAGSystem wiring: ingest is immutable, retrieval is deterministic, and an
empty system refuses rather than guesses."""

import pytest

from src.rag.embeddings import Embedder
from src.rag.pipeline import RAGSystem

from conftest import DOCS, BagOfWordsEmbedder


def test_ingestion_chunks_every_document(system):
    assert len(system.chunks) == len(DOCS)
    assert set(system.source_ids) == {"expense-policy", "security-policy", "leave-policy"}


def test_create_without_arguments_uses_offline_defaults():
    assert isinstance(RAGSystem.create().retriever.embedder, Embedder)


def test_empty_system_has_no_chunks():
    assert RAGSystem.create(embedder=BagOfWordsEmbedder()).chunks == ()


def test_ask_is_deterministic(system):
    question = "how often must passwords be rotated?"
    assert system.ask(question, k=2) == system.ask(question, k=2)


def test_retrieve_returns_ranked_dicts(system):
    hits = system.retrieve("what is the international per diem cap?", k=3)
    assert hits[0]["source_id"] == "expense-policy"
    assert set(hits[0]) == {"source_id", "chunk", "score"}
    scores = [hit["score"] for hit in hits]
    assert scores == sorted(scores, reverse=True)


def test_ask_merges_answer_and_retrieval(system):
    result = system.ask("how often must passwords be rotated?", k=1)
    assert result["cited_sources"] == ["security-policy"]
    assert result["retrieved_source_ids"] == ["security-policy"]


def test_blank_documents_are_skipped():
    system = RAGSystem.create(embedder=BagOfWordsEmbedder()).ingest_documents(
        [("blank", "   "), ("real", "the password rotation period is 90 days")]
    )
    assert system.source_ids == ("real",)


def test_reingesting_returns_a_new_system_and_leaves_the_original(system):
    before = len(system.chunks)
    reingested = system.ingest_documents([("other", "content about parking permits")])
    assert reingested is not system
    assert len(system.chunks) == before
    assert system.source_ids.count("expense-policy") == 1
    assert reingested.source_ids == ("other",)


def test_empty_system_returns_no_hits_and_declines():
    empty = RAGSystem.create(embedder=BagOfWordsEmbedder())
    assert empty.retrieve("anything at all", k=3) == []
    assert empty.ask("anything at all")["abstained"] is True


def test_min_similarity_gates_retrieval(system, embedder):
    from conftest import MIN_SIMILARITY

    gated = system.retriever
    assert gated.min_similarity == MIN_SIMILARITY
    assert gated.search("how many parking permits are available?", k=3) == []
    assert gated.search("how often must passwords be rotated?", k=3) != []


def test_explicitly_zero_similarity_returns_everything(system, embedder):
    # The mechanism still works when you ask for it...
    ungated = RAGSystem.create(embedder=embedder, min_similarity=0.0).ingest_documents(DOCS)
    assert ungated.retriever.min_similarity == 0.0
    assert ungated.ask("how many parking permits are available?")["abstained"] is False


def test_default_ships_with_abstention_enabled(embedder):
    # ...but abstention must never be the default. A retriever that always returns
    # something is a hallucination factory, and this project claims to fix that.
    default = RAGSystem.create(embedder=embedder).ingest_documents(DOCS)
    assert default.retriever.min_similarity > 0.0


def test_overlap_at_or_above_chunk_size_raises_at_ingest():
    system = RAGSystem.create(embedder=BagOfWordsEmbedder(), chunk_size=100, chunk_overlap=100)
    with pytest.raises(ValueError):
        system.ingest_documents(DOCS)
