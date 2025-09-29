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
import sys
from dataclasses import dataclass
from typing import Callable, List, Optional

from services.atlas_store import AtlasStore, ChunkRecord
from services.embedder import HashingEmbedder
from services.s3_loader import S3DocumentLoader
from services.text_processing import chunk_text, normalize_whitespace
from services.llm_client import BedrockConfig, BedrockLLMClient, LLMInvocationError

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
    aws_region: str
    aws_profile: Optional[str]
    llm_model_id: str
    chunk_size: int = 400
    chunk_overlap: int = 40
    top_k: int = 3
    llm_max_tokens: int = 512
    llm_temperature: float = 0.2
    llm_streaming: bool = False

    @classmethod
    def from_env(cls) -> "AppConfig":
        missing = []
        bucket = os.getenv("S3_BUCKET_NAME")
        if not bucket:
            missing.append("S3_BUCKET_NAME")
        mongodb_uri = os.getenv("MONGODB_ATLAS_URI")
        if not mongodb_uri:
            missing.append("MONGODB_ATLAS_URI")

        mongodb_db = os.getenv("ATLAS_DB_NAME", "demo")
        mongodb_collection = os.getenv("ATLAS_COLLECTION_NAME", "documents")
        search_index = os.getenv("ATLAS_SEARCH_INDEX_NAME", "demo_rag_index")
        aws_region = os.getenv("AWS_REGION")
        if not aws_region:
            missing.append("AWS_REGION")
        aws_profile = os.getenv("AWS_PROFILE")
        llm_model_id = os.getenv(
            "BEDROCK_MODEL_ID", "anthropic.claude-3-haiku-20240307-v1:0"
        )
        chunk_size = int(os.getenv("CHUNK_WORDS", "400"))
        chunk_overlap = int(os.getenv("CHUNK_OVERLAP", "40"))
        top_k = int(os.getenv("TOP_K", "3"))
        llm_max_tokens = int(os.getenv("LLM_MAX_TOKENS", "512"))
        llm_temperature = float(os.getenv("LLM_TEMPERATURE", "0.2"))
        llm_streaming = os.getenv("LLM_STREAMING", "false").lower() in {
            "1",
            "true",
            "yes",
            "on",
        }

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
            aws_profile=aws_profile,
            llm_model_id=llm_model_id,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            top_k=top_k,
            llm_max_tokens=llm_max_tokens,
            llm_temperature=llm_temperature,
            llm_streaming=llm_streaming,
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


def assemble_prompt(question: str, documents: List[dict]) -> str:
    """Compose a grounded prompt that includes numbered context passages."""
    context_sections = []
    for idx, doc in enumerate(documents, start=1):
        metadata = doc.get("metadata", {})
        source = metadata.get("source") or metadata.get("s3_key") or doc.get("chunk_id")
        snippet = normalize_whitespace(doc.get("text", ""))
        context_sections.append(f"[{idx}] Source: {source}\n{snippet}")

    context_block = "\n\n".join(context_sections)
    instructions = (
        "You are an expert assistant answering questions using only the supplied context. "
        "If the answer cannot be derived from the context, reply that you do not know. "
        "Cite sources inline using the bracketed numbers, e.g., [1]."
    )

    return (
        f"{instructions}\n\n"
        f"Question: {question}\n\n"
        f"Context:\n{context_block}\n\n"
        "Answer:"
    )


def generate_grounded_answer(
    question: str,
    documents: List[dict],
    *,
    llm_client: BedrockLLMClient,
    max_tokens: int,
    temperature: float,
    stream: bool = False,
    on_token: Optional[Callable[[str], None]] = None,
) -> str:
    """Invoke the LLM with the grounded prompt and return the response text."""
    prompt = assemble_prompt(question, documents)

    if stream:
        chunks: List[str] = []
        try:
            for piece in llm_client.stream(
                prompt=prompt, max_tokens=max_tokens, temperature=temperature
            ):
                if on_token:
                    on_token(piece)
                chunks.append(piece)
        except LLMInvocationError:
            raise
        return "".join(chunks).strip()

    try:
        response = llm_client.generate(
            prompt=prompt, max_tokens=max_tokens, temperature=temperature
        )
    except LLMInvocationError:
        raise
    return response.strip()


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
    llm_client = BedrockLLMClient(
        BedrockConfig(
            region=config.aws_region,
            model_id=config.llm_model_id,
            profile=config.aws_profile,
            max_tokens=config.llm_max_tokens,
            temperature=config.llm_temperature,
        )
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

            print("\nAnswer:")
            try:
                if config.llm_streaming:
                    printed = False

                    def _echo(text: str) -> None:
                        nonlocal printed
                        printed = True
                        print(text, end="", flush=True)

                    answer = generate_grounded_answer(
                        query,
                        results,
                        llm_client=llm_client,
                        max_tokens=config.llm_max_tokens,
                        temperature=config.llm_temperature,
                        stream=True,
                        on_token=_echo,
                    )
                    if printed:
                        print()
                    if not answer:
                        print("(no response)")
                else:
                    answer = generate_grounded_answer(
                        query,
                        results,
                        llm_client=llm_client,
                        max_tokens=config.llm_max_tokens,
                        temperature=config.llm_temperature,
                    )
                    print(answer or "(no response)")
            except LLMInvocationError as exc:
                logger.error("LLM generation failed: %s", exc)
                continue

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
                source = metadata.get("source") or metadata.get("s3_key") or doc.get("chunk_id")
                snippet = normalize_whitespace(doc.get("text", ""))
                if len(snippet) > 240:
                    snippet = snippet[:237] + "..."
                print(f"[{idx}] {source} ({score_str})")
                print(f"    {snippet}")
    except KeyboardInterrupt:
        pass

    print("\nGoodbye!")


if __name__ == "__main__":
    main()
