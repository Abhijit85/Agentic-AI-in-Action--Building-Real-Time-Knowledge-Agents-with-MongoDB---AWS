"""Simple embedding utilities compatible with Atlas vector search."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List

import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer


@dataclass
class HashingEmbedder:
    """Stateless embedder that produces fixed-length dense vectors."""

    n_features: int = 512

    def __post_init__(self) -> None:
        self._vectorizer = HashingVectorizer(
            n_features=self.n_features,
            alternate_sign=False,
            stop_words="english",
            norm="l2",
        )

    def embed(self, text: str) -> List[float]:
        """Embed a single text input."""
        matrix = self._vectorizer.transform([text])
        dense = matrix.toarray()[0]
        return dense.astype(np.float32).tolist()

    def embed_many(self, texts: Iterable[str]) -> List[List[float]]:
        """Embed multiple texts in batch."""
        matrix = self._vectorizer.transform(list(texts))
        dense = matrix.toarray().astype(np.float32)
        return dense.tolist()
