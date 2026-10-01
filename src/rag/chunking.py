"""Fixed-size text chunking.

Why: chunk size is the single biggest lever on answer quality in a RAG system.
Too small and the answer loses its context ("...except when the trip is
international."); too large and retrieval gets noisy because one chunk carries
several unrelated facts. Fixed-size windows with overlap are the boring default
for a reason: they are predictable, cheap, and easy to reason about in an eval.

Overlap exists so a sentence that straddles a boundary still appears whole in
one of the two neighbouring chunks.
"""

from __future__ import annotations


def chunk_text(text: str, size: int = 500, overlap: int = 100) -> list[str]:
    """Split ``text`` into overlapping character windows of at most ``size``.

    Guarantees:
        - ``size <= 0`` or ``overlap < 0`` raises ``ValueError``.
        - ``overlap >= size`` raises ``ValueError`` (it would never advance).
        - text shorter than ``size`` yields exactly one chunk.
        - no returned chunk is empty or whitespace-only.

    The step is ``size - overlap``, so chunks tile the text forward with each
    chunk re-covering ``overlap`` characters of its predecessor.
    """
    if size <= 0:
        raise ValueError(f"size must be positive, got {size}")
    if overlap < 0:
        raise ValueError(f"overlap must be non-negative, got {overlap}")
    if overlap >= size:
        raise ValueError(
            f"overlap ({overlap}) must be smaller than size ({size}); "
            "otherwise the window never advances and chunking cannot terminate"
        )

    stripped = text.strip()
    if not stripped:
        return []

    step = size - overlap
    chunks: list[str] = []
    start = 0
    while start < len(stripped):
        chunk = stripped[start : start + size].strip()
        if chunk:
            chunks.append(chunk)
        if start + size >= len(stripped):
            break
        start += step

    return chunks
