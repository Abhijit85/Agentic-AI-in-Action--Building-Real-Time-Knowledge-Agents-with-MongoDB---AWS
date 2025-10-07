"""Simple embedding utilities compatible with Atlas vector search."""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Iterable, List, Optional, Protocol

import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer

try:  # Optional dependency for Voyage embeddings used by YouTubeRAG.
    import voyageai
except Exception:  # pragma: no cover - voyageai is optional in test environments.
    voyageai = None  # type: ignore[assignment]


class EmbeddingBackend(Protocol):
    """Shared interface for query and ingestion embedders."""

    n_features: int

    def embed(self, text: str) -> List[float]:
        ...

    def embed_many(self, texts: Iterable[str]) -> List[List[float]]:
        ...


@dataclass
class HashingEmbedder:
    """Stateless embedder that produces deterministic dense vectors."""

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


@dataclass
class VoyageEmbedder:
    """Wrapper around the VoyageAI embedding API used by YouTubeRAG."""

    model: str = "voyage-large-2"
    api_key: Optional[str] = None
    n_features: int = 1024

    def __post_init__(self) -> None:
        if voyageai is None:  # pragma: no cover - handled in runtime environments.
            raise RuntimeError(
                "voyageai is not installed. Install the voyageai package or unset VOYAGE_API_KEY."
            )
        key = self.api_key or os.getenv("VOYAGE_API_KEY")
        if not key:
            raise RuntimeError(
                "VOYAGE_API_KEY is required to use Voyage embeddings. "
                "Set the environment variable or fall back to HashingEmbedder."
            )
        model_name = os.getenv("VOYAGE_MODEL") or os.getenv("VOYAGE_EMBED_MODEL")
        if model_name:
            self.model = model_name
        self._client = voyageai.Client(api_key=key)

    def embed(self, text: str) -> List[float]:
        """Embed a single text input via VoyageAI."""
        response = self._client.embed(
            texts=[text],
            model=self.model,
        )
        vector = response.data[0].embedding
        if self.n_features and len(vector) != self.n_features:
            raise ValueError(
                f"Voyage embedding dimensionality mismatch: expected {self.n_features}, got {len(vector)}."
            )
        return [float(value) for value in vector]

    def embed_many(self, texts: Iterable[str]) -> List[List[float]]:
        """Embed multiple texts via VoyageAI."""
        batch = list(texts)
        if not batch:
            return []
        response = self._client.embed(
            texts=batch,
            model=self.model,
        )
        vectors: List[List[float]] = []
        for item in response.data:
            vector = [float(value) for value in item.embedding]
            if self.n_features and len(vector) != self.n_features:
                raise ValueError(
                    f"Voyage embedding dimensionality mismatch: expected {self.n_features}, got {len(vector)}."
                )
            vectors.append(vector)
        return vectors


def build_embedder_from_env(default_dimensions: int = 1024) -> EmbeddingBackend:
    """Return an embedder aligned with Atlas vector index configuration."""
    target_dimensions = int(
        os.getenv("ATLAS_EMBEDDING_DIM") or os.getenv("EMBEDDING_DIMENSIONS") or default_dimensions
    )

    voyage_api_key = os.getenv("VOYAGE_API_KEY")
    if voyage_api_key:
        embedder = VoyageEmbedder(api_key=voyage_api_key, n_features=target_dimensions)
        if embedder.n_features != target_dimensions:
            raise ValueError(
                "VoyageEmbedder produced vectors with unexpected dimensions. "
                "Verify the selected model matches ATLAS_EMBEDDING_DIM."
            )
        return embedder

    return HashingEmbedder(n_features=target_dimensions)


__all__ = [
    "EmbeddingBackend",
    "HashingEmbedder",
    "VoyageEmbedder",
    "build_embedder_from_env",
]
