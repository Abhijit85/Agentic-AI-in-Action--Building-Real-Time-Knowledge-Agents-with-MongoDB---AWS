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
                    "definition": {
                        "mappings": {
                            "dynamic": False,
                            "fields": {
                                "text": {"type": "string"},
                                "embedding": {
                                    "type": "vector",
                                    "similarity": "cosine",
                                    "dimensions": vector_dimensions,
                                },
                                "metadata": {
                                    "type": "document",
                                    "fields": {
                                        "source": {"type": "string"},
                                        "s3_key": {"type": "string"},
                                        "chunk_index": {"type": "number"},
                                    },
                                },
                            },
                        }
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

        compound: Dict[str, object] = {
            "should": [
                {
                    "knnBeta": {
                        "vector": query_vector,
                        "path": "embedding",
                        "k": max(top_k, 1),
                    }
                }
            ]
        }
        if query_text:
            compound["must"] = [
                {
                    "text": {
                        "query": query_text,
                        "path": ["text"],
                        "score": {"boost": {"value": 1.0}},
                    }
                }
            ]

        pipeline = [
            {
                "$search": {
                    "index": self._index_name,
                    "compound": compound,
                    "scoreDetails": True,
                }
            },
            {"$limit": max(top_k, 1)},
            {
                "$project": {
                    "_id": 0,
                    "chunk_id": 1,
                    "text": 1,
                    "metadata": 1,
                    "search_score": {"$meta": "searchScore"},
                    "score_details": {"$meta": "searchScoreDetails"},
                }
            },
        ]

        results = list(self._collection.aggregate(pipeline))
        for doc in results:
            score_details = doc.pop("score_details", None)
            doc["vector_score"] = _extract_operator_score(score_details, target="knnBeta")
            doc["keyword_score"] = _extract_operator_score(score_details, target="text")
            doc["search_score"] = float(doc.get("search_score", 0.0))
        return results


def _extract_operator_score(details: Optional[Dict[str, object]], *, target: str) -> Optional[float]:
    """Traverse Atlas score details to find a specific operator contribution."""
    if not isinstance(details, dict):
        return None

    operator = details.get("operator")
    if operator == target:
        score = details.get("score")
        return float(score) if score is not None else None

    for child in details.get("details", []) or []:
        extracted = _extract_operator_score(child, target=target)
        if extracted is not None:
            return extracted
    return None
