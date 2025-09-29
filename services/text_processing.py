"""Helpers for turning raw documents into retrievable chunks."""
from __future__ import annotations

from typing import Iterable


def chunk_text(
    text: str,
    *,
    chunk_size: int = 500,
    overlap: int = 50,
) -> Iterable[str]:
    """Yield overlapping chunks of the provided text.

    Uses a simple word-based splitter to avoid breaking sentences mid-token.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap < 0:
        raise ValueError("overlap must be non-negative")

    words = text.split()
    if not words:
        return []

    step = max(chunk_size - overlap, 1)
    for start in range(0, len(words), step):
        end = start + chunk_size
        chunk_words = words[start:end]
        if chunk_words:
            yield " ".join(chunk_words)


def normalize_whitespace(text: str) -> str:
    """Collapse whitespace to single spaces to stabilise embeddings."""
    return " ".join(text.split())
