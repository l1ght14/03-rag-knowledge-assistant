"""Embedding contract: deterministic, correct shape, unit-norm rows."""

import numpy as np
import pytest

from src.rag.embeddings import Embedder, HashEmbedder, get_embedder, l2_normalize

TEXTS = ["the international per diem cap is 300 dollars", "hotel bookings go through the agency"]


def test_same_input_twice_gives_identical_vectors():
    embedder = HashEmbedder(dim=64)
    assert np.array_equal(embedder.embed(TEXTS), embedder.embed(TEXTS))


def test_determinism_survives_a_fresh_instance():
    assert np.array_equal(HashEmbedder(dim=64).embed(TEXTS), HashEmbedder(dim=64).embed(TEXTS))


def test_different_texts_give_different_vectors():
    vectors = HashEmbedder(dim=64).embed(TEXTS)
    assert not np.array_equal(vectors[0], vectors[1])


def test_shape_is_n_by_dim():
    vectors = HashEmbedder(dim=32).embed(TEXTS)
    assert vectors.shape == (2, 32)


def test_rows_are_l2_normalized():
    assert np.allclose(np.linalg.norm(HashEmbedder(dim=64).embed(TEXTS), axis=1), 1.0)


def test_empty_input_returns_empty_matrix_of_right_width():
    assert HashEmbedder(dim=16).embed([]).shape == (0, 16)


def test_embed_one_returns_a_single_vector():
    vector = HashEmbedder(dim=16).embed_one(TEXTS[0])
    assert vector.shape == (16,)
    assert np.isclose(np.linalg.norm(vector), 1.0)


def test_l2_normalize_leaves_zero_rows_finite():
    normalized = l2_normalize(np.zeros((2, 4)))
    assert np.all(np.isfinite(normalized))
    assert np.allclose(normalized, 0.0)


def test_l2_normalize_accepts_a_one_dimensional_vector():
    assert np.isclose(np.linalg.norm(l2_normalize(np.array([3.0, 4.0]))), 1.0)


def test_get_embedder_returns_the_hash_embedder():
    assert isinstance(get_embedder("hash"), Embedder)
    assert isinstance(get_embedder(), Embedder)


def test_unknown_embedder_name_raises_with_guidance():
    with pytest.raises(NotImplementedError) as excinfo:
        get_embedder("openai")
    assert "Embedder" in str(excinfo.value)


@pytest.mark.parametrize("dim", [0, -8])
def test_invalid_dim_raises(dim):
    with pytest.raises(ValueError):
        HashEmbedder(dim=dim)


def test_dim_is_exposed_on_the_instance():
    assert HashEmbedder(dim=1).dim == 1
