"""Persistence layer for MongoDB Atlas with Atlas Search."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional

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
                            {"type": "filter", "path": "metadata.source"},
                            {"type": "filter", "path": "metadata.s3_key"},
                            {"type": "filter", "path": "metadata.chunk_index"},
                            {"type": "filter", "path": "metadata.video_id"},
                            {"type": "filter", "path": "metadata.video_title"},
                            {"type": "filter", "path": "metadata.video_url"},
                            {"type": "filter", "path": "metadata.channel"},
                            {"type": "filter", "path": "metadata.channel_title"},
                            {"type": "filter", "path": "metadata.channel_id"},
                            {"type": "filter", "path": "metadata.speaker"},
                            {"type": "filter", "path": "metadata.speaker_role"},
                            {"type": "filter", "path": "metadata.playlist_id"},
                            {"type": "filter", "path": "metadata.segment_id"},
                            {"type": "filter", "path": "metadata.start_time"},
                            {"type": "filter", "path": "metadata.end_time"},
                            {"type": "filter", "path": "metadata.tags"},
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
        filters: Optional[Mapping[str, Any]] = None,
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
            }
        ]

        match_clauses: List[Dict[str, Any]] = []
        if filters:
            field_map = {
                "video_id": "metadata.video_id",
                "video_title": "metadata.video_title",
                "channel": "metadata.channel",
                "channel_title": "metadata.channel_title",
                "channel_id": "metadata.channel_id",
                "speaker": "metadata.speaker",
                "speaker_role": "metadata.speaker_role",
                "playlist_id": "metadata.playlist_id",
                "segment_id": "metadata.segment_id",
                "tags": "metadata.tags",
            }

            def _coerce_float(value: Any) -> Optional[float]:
                try:
                    return float(value)
                except (TypeError, ValueError):
                    return None

            time_range = filters.get("time_range")
            start_bound = (
                _coerce_float(filters.get("start_time"))
                or _coerce_float(filters.get("after"))
                or _coerce_float(filters.get("from"))
            )
            end_bound = (
                _coerce_float(filters.get("end_time"))
                or _coerce_float(filters.get("before"))
                or _coerce_float(filters.get("to"))
            )
            if isinstance(time_range, Mapping):
                start_bound = start_bound or _coerce_float(
                    time_range.get("start")
                    or time_range.get("from")
                    or time_range.get("after")
                )
                end_bound = end_bound or _coerce_float(
                    time_range.get("end")
                    or time_range.get("to")
                    or time_range.get("before")
                )

            if start_bound is not None:
                match_clauses.append({"metadata.end_time": {"$gte": start_bound}})
            if end_bound is not None:
                match_clauses.append({"metadata.start_time": {"$lte": end_bound}})

            for key, value in filters.items():
                if key in {
                    "time_range",
                    "start_time",
                    "end_time",
                    "after",
                    "before",
                    "from",
                    "to",
                }:
                    continue
                path = field_map.get(key)
                if not path:
                    # Preserve backwards compatibility for existing metadata.* keys.
                    if key.startswith("metadata."):
                        path = key
                    else:
                        continue
                if value is None:
                    continue
                if isinstance(value, (list, tuple, set)):
                    values = [item for item in value if item not in ("", None)]
                    if not values:
                        continue
                    match_clauses.append({path: {"$in": values}})
                else:
                    match_clauses.append({path: value})

        if match_clauses:
            if len(match_clauses) == 1:
                pipeline.append({"$match": match_clauses[0]})
            else:
                pipeline.append({"$match": {"$and": match_clauses}})

        pipeline.append(
            {
                "$project": {
                    "_id": 0,
                    "chunk_id": 1,
                    "text": 1,
                    "metadata": 1,
                    "vector_score": {"$meta": "vectorSearchScore"},
                }
            }
        )

        results = list(self._collection.aggregate(pipeline))
        for doc in results:
            score = doc.get("vector_score")
            doc["vector_score"] = float(score) if score is not None else None
            doc["keyword_score"] = None
            doc["search_score"] = doc["vector_score"] or 0.0
        return results
