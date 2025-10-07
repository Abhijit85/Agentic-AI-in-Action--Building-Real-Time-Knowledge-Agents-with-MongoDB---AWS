#!/usr/bin/env python3
"""
Retrieval-augmented demo wired to S3 and MongoDB Atlas Search.

This script ingests plain text objects from an S3 bucket, chunks and embeds the
contents, persists them to MongoDB Atlas with an Atlas Search vector index, and
then performs hybrid vector + keyword retrieval to answer free-form questions.
"""
from __future__ import annotations

import logging
import re
import os
import sys

import boto3
from dataclasses import dataclass
from typing import List, Optional

from dotenv import load_dotenv

from agent.agentcore_config import MemoryAwareBedrockReasoner
from services.atlas_store import AtlasStore, ChunkRecord
from services.embedder import EmbeddingBackend, HashingEmbedder, build_embedder_from_env
from services.s3_loader import S3DocumentLoader
from services.text_processing import chunk_text, normalize_whitespace
from services.prompting import prepare_context_documents
from services.llm_client import BedrockConfig, BedrockLLMClient, LLMInvocationError


def _env_value(name: str) -> Optional[str]:
    """Treat unset or blank environment variables as missing."""
    value = os.getenv(name)
    if value is None:
        return None
    return value


load_dotenv()

logger = logging.getLogger(__name__)


QUESTION_PREFIXES = (
    "what",
    "how",
    "who",
    "where",
    "why",
    "when",
    "do",
    "does",
    "did",
    "can",
    "should",
    "is",
    "are",
    "will",
    "would",
    "could",
    "explain",
    "tell",
    "give",
    "list",
    "describe",
)


ACKNOWLEDGEMENT_MESSAGE = (
    "Got it. I'll keep that in mind for any follow-up questions."
)


NO_CONTEXT_MESSAGE = (
    "I don't have much information about that yet. Try adding documents or "
    "sharing more details."
)


def _looks_like_question(text: str) -> bool:
    stripped = text.strip()
    if not stripped:
        return False
    if stripped.endswith("?"):
        return True
    lowered = stripped.lower()
    return lowered.startswith(QUESTION_PREFIXES)


_CITATION_PATTERN = re.compile(r"\[(\d+)\]")


def _has_citations(text: str) -> bool:
    if not text:
        return False
    return bool(_CITATION_PATTERN.search(text))


@dataclass
class AppConfig:
    """Runtime configuration sourced from environment variables."""

    s3_bucket: str
    s3_prefix: Optional[str]
    mongodb_uri: str
    mongodb_db: str
    mongodb_collection: str
    search_index: str
    aws_region: str
    aws_access_key_id: str
    aws_secret_access_key: str
    aws_session_token: Optional[str]
    llm_model_id: str
    chunk_size: int = 400
    chunk_overlap: int = 40
    top_k: int = 3
    llm_max_tokens: int = 512
    llm_temperature: float = 0.2
    llm_streaming: bool = False
    memory_short_term_turns: int = 12
    embedding_dimensions: int = 1024

    @classmethod
    def from_env(cls) -> "AppConfig":
        missing = []
        bucket = _env_value("S3_BUCKET_NAME")
        if not bucket:
            missing.append("S3_BUCKET_NAME")
        mongodb_uri = _env_value("MONGODB_ATLAS_URI")
        if not mongodb_uri:
            missing.append("MONGODB_ATLAS_URI")

        mongodb_db = _env_value("ATLAS_DB_NAME") or "media_rag"
        mongodb_collection = _env_value("ATLAS_COLLECTION_NAME") or "video_chunks"
        search_index = _env_value("ATLAS_SEARCH_INDEX_NAME") or "video_chunks_rag_index"
        aws_region = _env_value("AWS_REGION")
        if not aws_region:
            missing.append("AWS_REGION")
        aws_access_key_id = _env_value("AWS_ACCESS_KEY_ID")
        if not aws_access_key_id:
            missing.append("AWS_ACCESS_KEY_ID")
        aws_secret_access_key = _env_value("AWS_SECRET_ACCESS_KEY")
        if not aws_secret_access_key:
            missing.append("AWS_SECRET_ACCESS_KEY")
        aws_session_token = _env_value("AWS_SESSION_TOKEN")
        llm_model_id = _env_value("BEDROCK_MODEL_ID") or "anthropic.claude-3-haiku-20240307-v1:0"
        chunk_size = int(_env_value("CHUNK_WORDS") or "400")
        chunk_overlap = int(_env_value("CHUNK_OVERLAP") or "40")
        top_k = int(_env_value("TOP_K") or "3")
        llm_max_tokens = int(_env_value("LLM_MAX_TOKENS") or "512")
        llm_temperature = float(_env_value("LLM_TEMPERATURE") or "0.2")
        llm_streaming = (_env_value("LLM_STREAMING") or "false").lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        memory_short_term_turns = int(_env_value("AGENT_MEMORY_TURNS") or "12")
        embedding_dimensions = int(_env_value("ATLAS_EMBEDDING_DIM") or "1024")

        if missing:
            raise ValueError(
                "Missing required environment variables: " + ", ".join(missing)
            )

        return cls(
            s3_bucket=bucket,
            s3_prefix=os.getenv("S3_PREFIX"),
            mongodb_uri=mongodb_uri,
            mongodb_db=mongodb_db,
            mongodb_collection=mongodb_collection,
            search_index=search_index,
            aws_region=aws_region,
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
            aws_session_token=aws_session_token,
            llm_model_id=llm_model_id,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            top_k=top_k,
            llm_max_tokens=llm_max_tokens,
            llm_temperature=llm_temperature,
            llm_streaming=llm_streaming,
            memory_short_term_turns=memory_short_term_turns,
            embedding_dimensions=embedding_dimensions,
        )


def ingest_documents(
    *,
    loader: S3DocumentLoader,
    embedder: EmbeddingBackend,
    store: AtlasStore,
    config: AppConfig,
) -> None:
    """Stream documents from S3, chunk, embed, and store them in Atlas."""
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

    # Ensure the Atlas Search index exists once documents are present.
    store.ensure_indexes(vector_dimensions=config.embedding_dimensions)

    logger.info("Ingestion complete. Total chunks processed: %s", ingested_chunks)

def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    try:
        config = AppConfig.from_env()
    except ValueError as exc:
        logger.error("%s", exc)
        sys.exit(1)

    # Boto treats an empty AWS_PROFILE as a request for a profile named ""
    if os.environ.get("AWS_PROFILE", "").strip() == "":
        os.environ.pop("AWS_PROFILE", None)

    session_kwargs = {
        "aws_access_key_id": config.aws_access_key_id,
        "aws_secret_access_key": config.aws_secret_access_key,
        "region_name": config.aws_region,
    }
    if config.aws_session_token:
        session_kwargs["aws_session_token"] = config.aws_session_token
    aws_session = boto3.session.Session(**session_kwargs)

    loader = S3DocumentLoader(
        bucket_name=config.s3_bucket,
        prefix=config.s3_prefix,
        s3_client=aws_session.client("s3"),
    )
    try:
        embedder = build_embedder_from_env(config.embedding_dimensions)
    except Exception as exc:  # noqa: BLE001 - fall back to deterministic hashing locally
        logger.warning("Falling back to hashing embedder: %s", exc)
        embedder = HashingEmbedder(n_features=config.embedding_dimensions)
    store = AtlasStore(
        connection_string=config.mongodb_uri,
        database=config.mongodb_db,
        collection=config.mongodb_collection,
        index_name=config.search_index,
    )
    llm_client = BedrockLLMClient(
        BedrockConfig(
            region=config.aws_region,
            model_id=config.llm_model_id,
            access_key_id=config.aws_access_key_id,
            secret_access_key=config.aws_secret_access_key,
            session_token=config.aws_session_token,
            max_tokens=config.llm_max_tokens,
            temperature=config.llm_temperature,
        )
    )

    reasoner = MemoryAwareBedrockReasoner(
        llm_client,
        max_tokens=config.llm_max_tokens,
        temperature=config.llm_temperature,
    )

    short_term_memory: List[dict[str, str]] = []

    try:
        ingest_documents(loader=loader, embedder=embedder, store=store, config=config)
    except Exception as exc:  # noqa: BLE001 - surface ingestion problems to the console
        logger.error("Failed to ingest documents: %s", exc)
        sys.exit(1)

    print("Documents are ready. Ask a question!")
    try:
        while True:
            query = input("\nShare a prompt or question (type 'exit' to leave): ").strip()
            if not query or query.lower() in {"exit", "quit"}:
                break

            is_question = _looks_like_question(query)

            short_term_memory.append({"role": "user", "content": query})
            max_messages = max(config.memory_short_term_turns * 2, 2)
            if len(short_term_memory) > max_messages:
                short_term_memory = short_term_memory[-max_messages:]

            if not is_question:
                acknowledgement = ACKNOWLEDGEMENT_MESSAGE
                print("\nAssistant:\n" + acknowledgement)
                short_term_memory.append({"role": "assistant", "content": acknowledgement})
                if len(short_term_memory) > max_messages:
                    short_term_memory = short_term_memory[-max_messages:]
                continue

            query_vector = embedder.embed(query)
            try:
                results = store.query(
                    query_text=query, query_vector=query_vector, top_k=config.top_k
                )
            except Exception as exc:  # noqa: BLE001 - provide readable errors for query path
                logger.error("Atlas Search query failed: %s", exc)
                results = []

            if results:
                print("\nAnswer:")
                context_documents = prepare_context_documents(results)
                memory_snapshot = {
                    "short_term": short_term_memory[-config.memory_short_term_turns :],
                    "long_term": context_documents,
                }
                try:
                    if config.llm_streaming:
                        printed = False

                        def _echo(text: str) -> None:
                            nonlocal printed
                            printed = True
                            print(text, end="", flush=True)

                        answer = reasoner.run(
                            query,
                            memory_snapshot=memory_snapshot,
                            stream=True,
                            on_token=_echo,
                            max_tokens=config.llm_max_tokens,
                            temperature=config.llm_temperature,
                        )
                        if printed:
                            print()
                        if not answer:
                            print("(no response)")
                    else:
                        answer = reasoner.run(
                            query,
                            memory_snapshot=memory_snapshot,
                            max_tokens=config.llm_max_tokens,
                            temperature=config.llm_temperature,
                        )
                        print(answer or "(no response)")
                except LLMInvocationError as exc:
                    logger.error("LLM generation failed: %s", exc)
                    short_term_memory.pop()
                    continue

                answer = answer or ""
                short_term_memory.append({"role": "assistant", "content": answer})
                if len(short_term_memory) > max_messages:
                    short_term_memory = short_term_memory[-max_messages:]

                if _has_citations(answer):
                    print("\nCited context:")
                    for idx, doc in enumerate(results, start=1):
                        vector_score = doc.get("vector_score")
                        keyword_score = doc.get("keyword_score")
                        scores = [f"search={doc.get('search_score', 0):.3f}"]
                        if vector_score is not None:
                            scores.append(f"vector={vector_score:.3f}")
                        if keyword_score is not None:
                            scores.append(f"keyword={keyword_score:.3f}")
                        score_str = ", ".join(scores)
                        metadata = doc.get("metadata", {})
                        source = (
                            metadata.get("source")
                            or metadata.get("s3_key")
                            or doc.get("chunk_id")
                        )
                        snippet = normalize_whitespace(doc.get("text", ""))
                        if len(snippet) > 240:
                            snippet = snippet[:237] + "..."
                        print(f"[{idx}] {source} ({score_str})")
                        print(f"    {snippet}")
            else:
                fallback = NO_CONTEXT_MESSAGE
                print("\nAnswer:\n" + fallback)
                short_term_memory.append({"role": "assistant", "content": fallback})
                if len(short_term_memory) > max_messages:
                    short_term_memory = short_term_memory[-max_messages:]
    except KeyboardInterrupt:
        pass

    print("\nGoodbye!")


if __name__ == "__main__":
    main()
