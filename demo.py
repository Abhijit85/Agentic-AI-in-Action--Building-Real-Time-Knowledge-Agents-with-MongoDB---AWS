#!/usr/bin/env python3
"""
Retrieval-augmented demo wired to S3 and MongoDB Atlas Search.

This script ingests plain text objects from an S3 bucket, chunks and embeds the
contents, persists them to MongoDB Atlas with an Atlas Search vector index, and
then performs hybrid vector + keyword retrieval to answer free-form questions.
"""
from __future__ import annotations

import logging
import os
import re
import sys
from dataclasses import dataclass
from typing import List, Optional

from services.atlas_store import AtlasStore, ChunkRecord
from services.embedder import HashingEmbedder
from services.s3_loader import S3DocumentLoader
from services.text_processing import chunk_text, normalize_whitespace

logger = logging.getLogger(__name__)


@dataclass
class AppConfig:
    """Runtime configuration sourced from environment variables."""

    s3_bucket: str
    s3_prefix: Optional[str]
    mongodb_uri: str
    mongodb_db: str
    mongodb_collection: str
    search_index: str
    chunk_size: int = 400
    chunk_overlap: int = 40
    top_k: int = 3

    @classmethod
    def from_env(cls) -> "AppConfig":
        missing = []
        bucket = os.getenv("S3_BUCKET_NAME")
        if not bucket:
            missing.append("S3_BUCKET_NAME")
        mongodb_uri = os.getenv("MONGODB_ATLAS_URI")
        if not mongodb_uri:
            missing.append("MONGODB_ATLAS_URI")
        if missing:
            raise ValueError(
                "Missing required environment variables: " + ", ".join(missing)
            )

        mongodb_db = os.getenv("ATLAS_DB_NAME", "demo")
        mongodb_collection = os.getenv("ATLAS_COLLECTION_NAME", "documents")
        search_index = os.getenv("ATLAS_SEARCH_INDEX_NAME", "demo_rag_index")
        chunk_size = int(os.getenv("CHUNK_WORDS", "400"))
        chunk_overlap = int(os.getenv("CHUNK_OVERLAP", "40"))
        top_k = int(os.getenv("TOP_K", "3"))

        return cls(
            s3_bucket=bucket,
            s3_prefix=os.getenv("S3_PREFIX"),
            mongodb_uri=mongodb_uri,
            mongodb_db=mongodb_db,
            mongodb_collection=mongodb_collection,
            search_index=search_index,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            top_k=top_k,
        )


def ingest_documents(
    *,
    loader: S3DocumentLoader,
    embedder: HashingEmbedder,
    store: AtlasStore,
    config: AppConfig,
) -> None:
    """Stream documents from S3, chunk, embed, and store them in Atlas."""
    store.ensure_indexes(vector_dimensions=embedder.n_features)

    batch: List[ChunkRecord] = []
    ingested_chunks = 0
    batch_size = 25
    for document in loader.iter_documents():
        normalised = normalize_whitespace(document.text)
        for idx, chunk in enumerate(
            chunk_text(normalised, chunk_size=config.chunk_size, overlap=config.chunk_overlap)
        ):
            embedding = embedder.embed(chunk)
            chunk_id = f"{document.key}:::{idx}"
            metadata = {
                "source": os.path.basename(document.key),
                "s3_key": document.key,
                "chunk_index": idx,
            }
            batch.append(ChunkRecord(chunk_id=chunk_id, text=chunk, embedding=embedding, metadata=metadata))
            ingested_chunks += 1
            if len(batch) >= batch_size:
                store.upsert_chunks(batch)
                batch.clear()

    if batch:
        store.upsert_chunks(batch)

    logger.info("Ingestion complete. Total chunks processed: %s", ingested_chunks)


def simple_summarize(texts: List[str], num_sentences: int = 3) -> str:
    """Summarize a list of documents using a simple word frequency heuristic."""
    stop_words = set(
        [
            "the",
            "is",
            "and",
            "to",
            "of",
            "a",
            "in",
            "for",
            "on",
            "with",
            "that",
            "this",
            "as",
            "an",
            "by",
            "it",
            "be",
            "are",
            "from",
            "or",
            "at",
            "into",
            "their",
            "which",
            "these",
            "such",
        ]
    )

    sentences = []
    for text in texts:
        text = text.replace("\n", " ").strip()
        parts = re.split(r"(?<=[\.\?!])\s+", text)
        for sentence in parts:
            sentence = sentence.strip()
            if len(sentence.split()) > 4:
                sentences.append(sentence)

    if not sentences:
        return "".join(texts)

    word_freq = {}
    for sentence in sentences:
        words = re.findall(r"\b\w+\b", sentence.lower())
        for word in words:
            if word not in stop_words:
                word_freq[word] = word_freq.get(word, 0) + 1

    sentence_scores = []
    for sentence in sentences:
        words = re.findall(r"\b\w+\b", sentence.lower())
        score = sum(word_freq.get(word, 0) for word in words)
        sentence_scores.append((sentence, score))

    top_sentences = sorted(sentence_scores, key=lambda x: x[1], reverse=True)[:num_sentences]
    top_sentences_text = [ts[0] for ts in sorted(top_sentences, key=lambda x: sentences.index(x[0]))]
    return " ".join(top_sentences_text)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    try:
        config = AppConfig.from_env()
    except ValueError as exc:
        logger.error("%s", exc)
        sys.exit(1)

    loader = S3DocumentLoader(bucket_name=config.s3_bucket, prefix=config.s3_prefix)
    embedder = HashingEmbedder()
    store = AtlasStore(
        connection_string=config.mongodb_uri,
        database=config.mongodb_db,
        collection=config.mongodb_collection,
        index_name=config.search_index,
    )

    try:
        ingest_documents(loader=loader, embedder=embedder, store=store, config=config)
    except Exception as exc:  # noqa: BLE001 - surface ingestion problems to the console
        logger.error("Failed to ingest documents: %s", exc)
        sys.exit(1)

    print("Documents are ready. Ask a question!")
    try:
        while True:
            query = input("\nEnter your question (or type 'exit' to quit): ").strip()
            if not query or query.lower() in {"exit", "quit"}:
                break

            query_vector = embedder.embed(query)
            try:
                results = store.query(query_text=query, query_vector=query_vector, top_k=config.top_k)
            except Exception as exc:  # noqa: BLE001 - provide readable errors for query path
                logger.error("Atlas Search query failed: %s", exc)
                continue

            if not results:
                print("No documents matched your query.")
                continue

            print("\nTop documents:")
            for doc in results:
                vector_score = doc.get("vector_score")
                keyword_score = doc.get("keyword_score")
                scores = [f"search={doc.get('search_score', 0):.3f}"]
                if vector_score is not None:
                    scores.append(f"vector={vector_score:.3f}")
                if keyword_score is not None:
                    scores.append(f"keyword={keyword_score:.3f}")
                score_str = ", ".join(scores)
                print(f"- {doc['metadata'].get('source', doc['chunk_id'])} ({score_str})")

            summary = simple_summarize([doc["text"] for doc in results], num_sentences=3)
            print("\nAnswer:")
            print(summary)
    except KeyboardInterrupt:
        pass

    print("\nGoodbye!")


if __name__ == "__main__":
    main()
