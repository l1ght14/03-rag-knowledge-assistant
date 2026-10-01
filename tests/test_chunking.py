"""Chunking invariants. These bound the worst failure mode: a chunker that
loops forever or hands the generator empty strings."""

import pytest

from src.rag.chunking import chunk_text


def test_short_text_returns_single_chunk():
    assert chunk_text("expense limit is 500", size=500) == ["expense limit is 500"]


def test_text_exactly_one_size_long_returns_one_chunk():
    text = "a" * 100
    assert chunk_text(text, size=100, overlap=10) == [text]


def test_exact_multiple_produces_expected_chunk_count():
    # 300 chars, size 100, overlap 0 -> exactly 3 windows, no trailing empty.
    text = "".join(str(i % 10) for i in range(300))
    chunks = chunk_text(text, size=100, overlap=0)
    assert len(chunks) == 3
    assert chunks[0] + chunks[1] + chunks[2] == text


def test_overlap_repeats_content_between_neighbours():
    text = "".join(str(i % 10) for i in range(250))
    chunks = chunk_text(text, size=100, overlap=20)
    assert chunks[0][-20:] == chunks[1][:20]


@pytest.mark.parametrize("size, overlap", [(100, 100), (100, 101), (100, 500)])
def test_overlap_at_or_above_size_raises(size, overlap):
    with pytest.raises(ValueError):
        chunk_text("some text", size=size, overlap=overlap)


@pytest.mark.parametrize("size, overlap", [(0, 0), (-10, 0), (100, -1)])
def test_invalid_sizes_raise(size, overlap):
    with pytest.raises(ValueError):
        chunk_text("some text", size=size, overlap=overlap)


def test_no_chunk_is_empty_or_whitespace_only():
    text = "first part here      second part here"
    for chunk in chunk_text(text, size=12, overlap=5):
        assert chunk.strip()
        assert chunk == chunk.strip()


def test_blank_text_returns_no_chunks():
    assert chunk_text("   \n\t  ", size=100, overlap=10) == []


def test_chunk_count_is_bounded_and_grows_with_overlap():
    text = "word " * 400
    no_overlap = chunk_text(text, size=500, overlap=0)
    with_overlap = chunk_text(text, size=500, overlap=100)
    assert len(with_overlap) > len(no_overlap) > 1
    # step = size - overlap, so n_chunks is bounded by ceil(len/step) + 1.
    assert len(with_overlap) <= -(-len(text) // 400) + 1
