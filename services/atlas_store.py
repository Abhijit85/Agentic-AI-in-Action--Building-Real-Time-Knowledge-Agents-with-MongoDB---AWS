"""Persistence layer for MongoDB Atlas with Atlas Search."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional

from pymongo import MongoClient, UpdateOne
from pymongo.collection import Collection
from pymongo.errors import OperationFailure

logger = logging.getLogger(__name__)


@dataclass
class ChunkRecord:
    """Representation of a single chunk destined for storage."""

    chunk_id: str
    text: str
    embedding: List[float]
    metadata: Dict[str, object] = field(default_factory=dict)


class AtlasStore:
    """Wrapper around a MongoDB collection configured for Atlas Search."""

    def __init__(
        self,
        connection_string: str,
        database: str,
        collection: str,
        index_name: str,
        *,
        client: Optional[MongoClient] = None,
    ) -> None:
        self._client = client or MongoClient(connection_string)
        self._collection: Collection = self._client[database][collection]
        self._index_name = index_name

    @property
    def collection(self) -> Collection:
        return self._collection

    def ensure_indexes(self, *, vector_dimensions: int) -> None:
        """Create the Atlas Search index if it does not already exist."""
        command = {
            "createSearchIndexes": self._collection.name,
            "indexes": [
                {
                    "name": self._index_name,
                    "type": "vectorSearch",
                    "definition": {
                        "fields": [
                            {
                                "type": "vector",
                                "path": "embedding",
                                "numDimensions": vector_dimensions,
                                "similarity": "cosine",
                            },
                            {
                                "type": "filter",
                                "path": "metadata.source",
                            },
                            {
                                "type": "filter",
                                "path": "metadata.s3_key",
                            },
                            {
                                "type": "filter",
                                "path": "metadata.chunk_index",
                            },
                        ]
                    },
                }
            ],
        }

        try:
            self._collection.database.command(command)
            logger.info("Ensured Atlas Search index '%s'", self._index_name)
        except OperationFailure as exc:
            if "already exists" in str(exc).lower():
                logger.debug("Atlas Search index '%s' already exists", self._index_name)
            else:
                raise

    def upsert_chunks(self, chunks: Iterable[ChunkRecord]) -> None:
        """Upsert chunk documents into the collection."""
        operations: List[UpdateOne] = []
        for chunk in chunks:
            doc = {
                "chunk_id": chunk.chunk_id,
                "text": chunk.text,
                "embedding": chunk.embedding,
                "metadata": chunk.metadata,
            }
            operations.append(
                UpdateOne({"chunk_id": chunk.chunk_id}, {"$set": doc}, upsert=True)
            )

        if not operations:
            return

        result = self._collection.bulk_write(operations, ordered=False)
        logger.info(
            "Atlas upsert completed (matched=%s, upserts=%s)",
            result.matched_count,
            len(result.upserted_ids),
        )

    def query(
        self,
        *,
        query_text: str,
        query_vector: List[float],
        top_k: int = 3,
    ) -> List[Dict[str, object]]:
        """Run a hybrid Atlas Search query returning score components."""
        if not query_vector:
            raise ValueError("query_vector cannot be empty")

        num_candidates = max(top_k * 8, top_k, 1)
        pipeline = [
            {
                "$vectorSearch": {
                    "index": self._index_name,
                    "path": "embedding",
                    "queryVector": query_vector,
                    "numCandidates": num_candidates,
                    "limit": max(top_k, 1),
                }
            },
            {
                "$project": {
                    "_id": 0,
                    "chunk_id": 1,
                    "text": 1,
                    "metadata": 1,
                    "vector_score": {"$meta": "vectorSearchScore"},
                }
            },
        ]

        results = list(self._collection.aggregate(pipeline))
        for doc in results:
            score = doc.get("vector_score")
            doc["vector_score"] = float(score) if score is not None else None
            doc["keyword_score"] = None
            doc["search_score"] = doc["vector_score"] or 0.0
        return results
