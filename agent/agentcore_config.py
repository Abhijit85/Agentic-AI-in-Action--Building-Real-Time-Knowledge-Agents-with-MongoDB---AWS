"""AgentCore runtime configuration for the retrieval-augmented demo.

This module centralises agent wiring so the same core logic can be exposed via
CLI, API, or chat channels while remaining observability- and security-aware.
It defines the runtime settings, memory providers, tool registry entries, and
Bedrock-backed reasoning engine expected by AgentCore.
"""
from __future__ import annotations

import importlib
import logging
import os
import boto3
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from functools import cached_property
from typing import Any, Callable, Dict, List, Optional

from services.atlas_store import AtlasStore, ChunkRecord
from services.embedder import HashingEmbedder
from services.llm_client import BedrockConfig, BedrockLLMClient
from services.s3_loader import S3DocumentLoader
from services.text_processing import chunk_text, normalize_whitespace
from services.prompting import build_grounded_answer_prompt, prepare_context_documents

logger = logging.getLogger(__name__)

# ----------------------------------------------------------------------------
# Settings
# ----------------------------------------------------------------------------


@dataclass
class AgentCoreSettings:
    """Configuration block for the AgentCore runtime.

    Values default to sensible demo settings and can be sourced from the same
    environment variables used by ``demo.py`` so the demo and agent stay in
    sync. Additional fields capture observability, security, and deployment
    preferences specific to AgentCore.
    """

    aws_region: str
    mongodb_uri: str
    mongodb_db: str
    mongodb_collection: str
    atlas_search_index: str
    s3_bucket: str
    s3_prefix: Optional[str] = None
    aws_access_key_id: str = ""
    aws_secret_access_key: str = ""
    aws_session_token: Optional[str] = None

    bedrock_model_id: str = "anthropic.claude-3-haiku-20240307-v1:0"
    bedrock_max_tokens: int = 1024
    bedrock_temperature: float = 0.2

    deployment_channel: str = "cli"  # cli | api | chat

    cloudwatch_log_group: str = "/agentcore/demo"
    cloudwatch_trace_namespace: str = "AgentCore/Demo"

    memory_short_term_max_turns: int = 12
    memory_short_term_token_limit: int = 4800
    memory_long_term_top_k: int = 5

    ingestion_chunk_words: int = 400
    ingestion_chunk_overlap: int = 40

    security_network_allowlist: List[str] = field(
        default_factory=lambda: [
            "aws:s3",
            "aws:bedrock",
            "aws:logs",
            "aws:cloudwatch",
            "mongodb:atlas",
        ]
    )
    security_environment_blocklist: List[str] = field(
        default_factory=lambda: [
            "AWS_SECRET_ACCESS_KEY",
            "AWS_SESSION_TOKEN",
            "MONGODB_ATLAS_URI",
        ]
    )

    cognito_user_pool_id: Optional[str] = None
    cognito_app_client_id: Optional[str] = None
    identity_provider: str = "agentcore.identity.CognitoIdentityProvider"
    identity_allowed_groups: List[str] = field(default_factory=list)

    @classmethod
    def from_env(cls) -> "AgentCoreSettings":
        """Build settings from environment variables shared with the demo."""
        required = {
            "S3_BUCKET_NAME": os.getenv("S3_BUCKET_NAME"),
            "MONGODB_ATLAS_URI": os.getenv("MONGODB_ATLAS_URI"),
            "AWS_REGION": os.getenv("AWS_REGION"),
            "AWS_ACCESS_KEY_ID": os.getenv("AWS_ACCESS_KEY_ID"),
            "AWS_SECRET_ACCESS_KEY": os.getenv("AWS_SECRET_ACCESS_KEY"),
        }
        missing = [name for name, value in required.items() if not value]
        if missing:
            raise ValueError(
                "Missing required environment variables for AgentCore: "
                + ", ".join(missing)
            )

        aws_session_token = os.getenv("AWS_SESSION_TOKEN")

        return cls(
            aws_region=required["AWS_REGION"],
            mongodb_uri=required["MONGODB_ATLAS_URI"],
            mongodb_db=os.getenv("ATLAS_DB_NAME", "demo"),
            mongodb_collection=os.getenv("ATLAS_COLLECTION_NAME", "documents"),
            atlas_search_index=os.getenv("ATLAS_SEARCH_INDEX_NAME", "demo_rag_index"),
            s3_bucket=required["S3_BUCKET_NAME"],
            s3_prefix=os.getenv("S3_PREFIX"),
            aws_access_key_id=required["AWS_ACCESS_KEY_ID"],
            aws_secret_access_key=required["AWS_SECRET_ACCESS_KEY"],
            aws_session_token=aws_session_token,
            bedrock_model_id=os.getenv(
                "BEDROCK_MODEL_ID", "anthropic.claude-3-haiku-20240307-v1:0"
            ),
            bedrock_max_tokens=int(os.getenv("LLM_MAX_TOKENS", "1024")),
            bedrock_temperature=float(os.getenv("LLM_TEMPERATURE", "0.2")),
            deployment_channel=os.getenv("AGENT_CHANNEL", "cli"),
            cloudwatch_log_group=os.getenv(
                "AGENT_CLOUDWATCH_LOG_GROUP", "/agentcore/demo"
            ),
            cloudwatch_trace_namespace=os.getenv(
                "AGENT_CLOUDWATCH_TRACE_NS", "AgentCore/Demo"
            ),
            memory_short_term_max_turns=int(os.getenv("AGENT_MEMORY_TURNS", "12")),
            memory_short_term_token_limit=int(
                os.getenv("AGENT_MEMORY_TOKENS", "4800")
            ),
            memory_long_term_top_k=int(os.getenv("AGENT_MEMORY_TOPK", "5")),
            ingestion_chunk_words=int(os.getenv("CHUNK_WORDS", "400")),
            ingestion_chunk_overlap=int(os.getenv("CHUNK_OVERLAP", "40")),
            cognito_user_pool_id=os.getenv("AGENT_COGNITO_USER_POOL_ID"),
            cognito_app_client_id=os.getenv("AGENT_COGNITO_APP_CLIENT_ID"),
            identity_provider=os.getenv(
                "AGENT_IDENTITY_PROVIDER",
                "agentcore.identity.CognitoIdentityProvider",
            ),
            identity_allowed_groups=[
                group.strip()
                for group in os.getenv("AGENT_ALLOWED_GROUPS", "").split(",")
                if group.strip()
            ],
        )


# ----------------------------------------------------------------------------
# Shared resources
# ----------------------------------------------------------------------------


@dataclass
class AgentResources:
    """Lazily constructed handles to external systems."""

    settings: AgentCoreSettings
    embedder: HashingEmbedder = field(default_factory=HashingEmbedder)

    @cached_property
    def atlas_store(self) -> AtlasStore:
        return AtlasStore(
            connection_string=self.settings.mongodb_uri,
            database=self.settings.mongodb_db,
            collection=self.settings.mongodb_collection,
            index_name=self.settings.atlas_search_index,
        )

    @cached_property
    def aws_session(self) -> boto3.session.Session:
        session_kwargs = {
            "aws_access_key_id": self.settings.aws_access_key_id,
            "aws_secret_access_key": self.settings.aws_secret_access_key,
            "region_name": self.settings.aws_region,
        }
        if self.settings.aws_session_token:
            session_kwargs["aws_session_token"] = self.settings.aws_session_token
        return boto3.session.Session(**session_kwargs)

    @cached_property
    def s3_loader(self) -> S3DocumentLoader:
        return S3DocumentLoader(
            bucket_name=self.settings.s3_bucket,
            prefix=self.settings.s3_prefix,
            s3_client=self.aws_session.client("s3"),
        )

    @cached_property
    def bedrock_client(self) -> BedrockLLMClient:
        return BedrockLLMClient(
            BedrockConfig(
                region=self.settings.aws_region,
                model_id=self.settings.bedrock_model_id,
                access_key_id=self.settings.aws_access_key_id,
                secret_access_key=self.settings.aws_secret_access_key,
                session_token=self.settings.aws_session_token,
                max_tokens=self.settings.bedrock_max_tokens,
                temperature=self.settings.bedrock_temperature,
            )
        )

    def ensure_vector_index(self) -> None:
        try:
            self.atlas_store.ensure_indexes(
                vector_dimensions=self.embedder.n_features
            )
        except Exception as exc:  # noqa: BLE001 - surface integration issues upstream
            logger.warning("Unable to ensure Atlas Search index: %s", exc)


_RUNTIME_RESOURCES: Optional[AgentResources] = None


def set_runtime_resources(resources: AgentResources) -> None:
    """Expose resources for tool handlers."""
    global _RUNTIME_RESOURCES
    _RUNTIME_RESOURCES = resources


def _require_resources() -> AgentResources:
    if _RUNTIME_RESOURCES is None:
        raise RuntimeError(
            "Agent resources are not initialised. Call init_agentcore() first."
        )
    return _RUNTIME_RESOURCES


# ----------------------------------------------------------------------------
# Reasoning engine
# ----------------------------------------------------------------------------


class MemoryAwareBedrockReasoner:
    """Reasoning engine that injects AgentCore memory into Bedrock prompts."""

    def __init__(
        self,
        client: BedrockLLMClient,
        *,
        max_tokens: int,
        temperature: float,
    ) -> None:
        self._client = client
        self._max_tokens = max_tokens
        self._temperature = temperature

    def run(
        self,
        user_message: str,
        *,
        memory_snapshot: Optional[Mapping[str, Any]] = None,
        stream: bool = False,
        on_token: Optional[Callable[[str], None]] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> str:
        prompt = self._build_prompt(user_message, memory_snapshot)
        effective_tokens = max_tokens or self._max_tokens
        effective_temperature = (
            temperature if temperature is not None else self._temperature
        )

        if stream:
            pieces: List[str] = []
            for chunk in self._client.stream(
                prompt=prompt,
                max_tokens=effective_tokens,
                temperature=effective_temperature,
            ):
                if on_token:
                    on_token(chunk)
                pieces.append(chunk)
            return "".join(pieces).strip()

        response = self._client.generate(
            prompt=prompt,
            max_tokens=effective_tokens,
            temperature=effective_temperature,
        )
        return response.strip()

    def _build_prompt(
        self,
        user_message: str,
        memory_snapshot: Optional[Mapping[str, Any]],
    ) -> str:
        sections: List[str] = []

        if memory_snapshot:
            short_term = memory_snapshot.get("short_term") or memory_snapshot.get(
                "short_term_messages"
            )
            if short_term:
                conversational = _format_short_term(short_term)
                if conversational:
                    sections.append("Recent conversation:\n" + conversational)

            long_term = memory_snapshot.get("long_term") or memory_snapshot.get(
                "long_term_documents"
            )
            if long_term:
                passages = _format_long_term(long_term)
                if passages:
                    sections.append("Knowledge retrieved from memory:\n" + passages)

        if not sections:
            sections.append(
                "No prior memory was available. Rely on the user message and keep answers grounded."
            )

        instructions = (
            "You are an enterprise retrieval assistant. Use the supplied memory "
            "and tool outputs to answer. Cite sources with [n] and state when "
            "information is missing."
        )
        context_block = "\n\n".join(sections)
        return (
            f"{instructions}\n\n"
            f"{context_block}\n\n"
            f"User question: {user_message}\n\n"
            "Assistant response:"
        )


def _format_short_term(messages: Iterable[Mapping[str, Any]]) -> str:
    formatted = []
    for message in messages:
        role = message.get("role", "unknown").upper()
        content = normalize_whitespace(str(message.get("content", "")))
        if content:
            formatted.append(f"{role}: {content}")
    return "\n".join(formatted[-10:])


def _format_long_term(documents: Iterable[Mapping[str, Any]]) -> str:
    sections: List[str] = []
    for idx, doc in enumerate(documents, start=1):
        text = normalize_whitespace(str(doc.get("text") or doc.get("content") or ""))
        if not text:
            continue
        source = doc.get("metadata", {}).get("source") or doc.get("source")
        heading = f"[{idx}] {source}" if source else f"[{idx}]"
        if len(text) > 480:
            text = text[:477] + "..."
        sections.append(f"{heading}\n{text}")
    return "\n\n".join(sections)


# ----------------------------------------------------------------------------
# Tool handlers
# ----------------------------------------------------------------------------


def mongo_search_tool(payload: Mapping[str, Any]) -> Dict[str, Any]:
    """Run a hybrid Atlas Search query using the conversation embedder."""
    resources = _require_resources()
    question = str(
        payload.get("query")
        or payload.get("text")
        or payload.get("question")
        or payload.get("input")
        or ""
    ).strip()
    if not question:
        raise ValueError("Atlas query tool requires a non-empty 'query'.")

    top_k = int(payload.get("top_k") or resources.settings.memory_long_term_top_k)
    logger.debug("Running Atlas tool query (top_k=%s)", top_k)

    vector = resources.embedder.embed(question)
    documents = resources.atlas_store.query(
        query_text=question,
        query_vector=vector,
        top_k=top_k,
    )

    for doc in documents:
        doc["text"] = normalize_whitespace(doc.get("text", ""))
    return {"matches": documents, "count": len(documents)}


def bedrock_answer_tool(payload: Mapping[str, Any]) -> Dict[str, Any]:
    """Generate a grounded answer from retrieved context using Bedrock."""
    resources = _require_resources()

    question = str(
        payload.get("question")
        or payload.get("query")
        or payload.get("prompt")
        or ""
    ).strip()
    if not question:
        raise ValueError("Bedrock answer tool requires a non-empty 'question'.")

    raw_context = payload.get("context") or payload.get("documents") or []
    if isinstance(raw_context, (str, bytes)):
        raw_iterable: Iterable[Any] = [raw_context]
    elif not isinstance(raw_context, Iterable):
        raise ValueError("'context' must be an iterable of documents.")
    else:
        raw_iterable = raw_context

    documents = prepare_context_documents(raw_iterable)
    if not documents:
        raise ValueError("At least one context document is required to call Bedrock.")

    top_k = payload.get("top_k")
    if top_k is not None:
        try:
            limit = max(int(top_k), 1)
        except (TypeError, ValueError) as exc:
            raise ValueError("'top_k' must be an integer if provided.") from exc
        documents = documents[:limit]

    instructions = str(payload.get("instructions") or "").strip() or None
    max_tokens = int(payload.get("max_tokens") or resources.settings.bedrock_max_tokens)
    temperature = float(
        payload.get("temperature") or resources.settings.bedrock_temperature
    )
    include_prompt = bool(payload.get("include_prompt", False))
    stream = bool(payload.get("stream") or payload.get("streaming"))
    return_tokens = bool(payload.get("return_tokens", False))

    prompt = build_grounded_answer_prompt(question, documents, instructions=instructions)

    logger.debug(
        "Calling Bedrock answer tool (tokens=%s, temperature=%s, stream=%s)",
        max_tokens,
        temperature,
        stream,
    )

    tokens: List[str] = []
    if stream:
        for piece in resources.bedrock_client.stream(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
        ):
            tokens.append(piece)
        answer = "".join(tokens).strip()
    else:
        answer = resources.bedrock_client.generate(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
        ).strip()

    sources: List[Dict[str, Any]] = []
    for doc in documents:
        metadata = doc.get("metadata", {}) if isinstance(doc, Mapping) else {}
        if isinstance(metadata, Mapping):
            source = (
                metadata.get("source")
                or metadata.get("s3_key")
                or metadata.get("uri")
                or metadata.get("document_id")
            )
        else:
            source = None
        sources.append(
            {
                "source": source,
                "chunk_id": doc.get("chunk_id"),
                "vector_score": doc.get("vector_score"),
                "keyword_score": doc.get("keyword_score"),
                "search_score": doc.get("search_score"),
            }
        )

    result: Dict[str, Any] = {
        "answer": answer,
        "question": question,
        "model_id": resources.settings.bedrock_model_id,
        "context_size": len(documents),
        "sources": sources,
    }

    if include_prompt:
        result["prompt"] = prompt
    if stream and return_tokens:
        result["tokens"] = tokens

    return result


# Backwards compatibility alias for earlier revisions
mongo_atlas_query_tool = mongo_search_tool


def s3_ingest_tool(payload: Mapping[str, Any]) -> Dict[str, Any]:
    """Ingest documents from S3 via the AgentCore gateway."""
    resources = _require_resources()
    resources.ensure_vector_index()

    limit = int(payload.get("document_limit") or payload.get("limit") or 5)
    chunk_size = int(payload.get("chunk_size") or resources.settings.ingestion_chunk_words)
    overlap = int(payload.get("chunk_overlap") or resources.settings.ingestion_chunk_overlap)
    dry_run = bool(payload.get("dry_run", False))

    logger.debug(
        "Starting S3 ingestion via tool (limit=%s, chunk=%s, overlap=%s, dry_run=%s)",
        limit,
        chunk_size,
        overlap,
        dry_run,
    )

    processed_documents = 0
    total_chunks = 0
    upsert_batch: List[ChunkRecord] = []
    processed_keys: List[str] = []

    for document in resources.s3_loader.iter_documents():
        processed_documents += 1
        processed_keys.append(document.key)
        normalised = normalize_whitespace(document.text)
        for idx, chunk in enumerate(
            chunk_text(normalised, chunk_size=chunk_size, overlap=overlap)
        ):
            embedding = resources.embedder.embed(chunk)
            chunk_id = f"{document.key}:::{idx}"
            metadata = {
                "source": os.path.basename(document.key),
                "s3_key": document.key,
                "chunk_index": idx,
            }
            upsert_batch.append(
                ChunkRecord(
                    chunk_id=chunk_id,
                    text=chunk,
                    embedding=embedding,
                    metadata=metadata,
                )
            )
            total_chunks += 1

        if not dry_run and upsert_batch:
            resources.atlas_store.upsert_chunks(upsert_batch)
            upsert_batch.clear()

        if processed_documents >= limit:
            break

    if dry_run:
        upsert_batch.clear()
    elif upsert_batch:
        resources.atlas_store.upsert_chunks(upsert_batch)

    return {
        "documents_processed": processed_documents,
        "chunks_written": 0 if dry_run else total_chunks,
        "dry_run": dry_run,
        "keys": processed_keys,
    }


# ----------------------------------------------------------------------------
# Tool registration helpers
# ----------------------------------------------------------------------------


def build_tool_specs(resources: AgentResources) -> List[Dict[str, Any]]:
    """Return tool specifications consumable by AgentCore."""
    _ = resources  # Acoustic guard; specs are static but keep signature aligned
    return [
        {
            "name": "mongo_search",
            "description": "Query MongoDB Atlas Search with hybrid vector + keyword scoring.",
            "entrypoint": "agent.agentcore_config.mongo_search_tool",
            "transport": "gateway",
            "input_schema": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "top_k": {"type": "integer", "minimum": 1, "maximum": 20},
                },
                "required": ["query"],
            },
            "output_schema": {
                "type": "object",
                "properties": {
                    "matches": {"type": "array"},
                    "count": {"type": "integer"},
                },
            },
            "secure": True,
        },
        {
            "name": "s3_ingest",
            "description": "Stream documents from S3 and persist chunked embeddings into Atlas.",
            "entrypoint": "agent.agentcore_config.s3_ingest_tool",
            "transport": "gateway",
            "input_schema": {
                "type": "object",
                "properties": {
                    "document_limit": {"type": "integer", "minimum": 1, "maximum": 100},
                    "chunk_size": {"type": "integer", "minimum": 50},
                    "chunk_overlap": {"type": "integer", "minimum": 0},
                    "dry_run": {"type": "boolean"},
                },
            },
            "output_schema": {
                "type": "object",
                "properties": {
                    "documents_processed": {"type": "integer"},
                    "chunks_written": {"type": "integer"},
                    "dry_run": {"type": "boolean"},
                    "keys": {"type": "array", "items": {"type": "string"}},
                },
            },
            "secure": True,
        },
        {
            "name": "bedrock_answer",
            "description": "Call an Amazon Bedrock model with retrieved context to produce an answer.",
            "entrypoint": "agent.agentcore_config.bedrock_answer_tool",
            "transport": "gateway",
            "input_schema": {
                "type": "object",
                "properties": {
                    "question": {"type": "string"},
                    "context": {
                        "type": "array",
                        "items": {"type": "object"},
                    },
                    "instructions": {"type": "string"},
                    "max_tokens": {"type": "integer", "minimum": 1},
                    "temperature": {"type": "number", "minimum": 0.0},
                    "top_k": {"type": "integer", "minimum": 1},
                    "include_prompt": {"type": "boolean"},
                    "stream": {"type": "boolean"},
                    "return_tokens": {"type": "boolean"},
                },
                "required": ["question", "context"],
            },
            "output_schema": {
                "type": "object",
                "properties": {
                    "answer": {"type": "string"},
                    "model_id": {"type": "string"},
                    "context_size": {"type": "integer"},
                    "sources": {"type": "array", "items": {"type": "object"}},
                },
            },
            "secure": True,
        },
    ]


def register_tools_with_runtime(runtime: Any, resources: AgentResources) -> None:
    """Best-effort registration against an AgentCore tool registry."""
    registry = getattr(runtime, "tool_registry", None) or getattr(runtime, "tools", None)
    if registry is None:
        logger.info("Runtime has no tool registry; returning configuration only.")
        return

    register_fn = getattr(registry, "register", None) or getattr(registry, "add", None)
    if register_fn is None:
        logger.info("Tool registry does not expose register/add; skipping dynamic wiring.")
        return

    handlers = {
        "mongo_search": mongo_search_tool,
        "s3_ingest": s3_ingest_tool,
        "bedrock_answer": bedrock_answer_tool,
    }

    for spec in build_tool_specs(resources):
        handler = handlers.get(spec["name"])
        if handler is None:
            continue

        try:
            register_fn(
                name=spec["name"],
                description=spec["description"],
                handler=handler,
                input_schema=spec.get("input_schema"),
                output_schema=spec.get("output_schema"),
                secure=spec.get("secure", True),
            )
            logger.debug("Registered tool '%s' with runtime", spec["name"])
        except TypeError:
            # Some registries may use positional arguments.
            register_fn(spec["name"], handler, spec["description"])
            logger.debug(
                "Registered tool '%s' using positional signature", spec["name"]
            )


# ----------------------------------------------------------------------------
# Observability & security helpers
# ----------------------------------------------------------------------------


def configure_observability(runtime: Any, settings: AgentCoreSettings) -> None:
    """Attach CloudWatch logging/tracing when AgentCore observability APIs exist."""
    tracer = _optional_factory(
        "agentcore.observability",
        "CloudWatchTracer",
        namespace=settings.cloudwatch_trace_namespace,
        log_group=settings.cloudwatch_log_group,
        service="rag-agent",
    )
    if tracer:
        if hasattr(runtime, "add_tracer"):
            runtime.add_tracer(tracer)
        elif hasattr(runtime, "configure_tracing"):
            runtime.configure_tracing(tracer=tracer)
        else:
            logger.debug("Runtime has no tracer hook; tracer initialised but not attached.")

    cw_logger = _optional_factory(
        "agentcore.observability",
        "CloudWatchLogger",
        log_group=settings.cloudwatch_log_group,
        stream_name=settings.deployment_channel,
    )
    if cw_logger:
        if hasattr(runtime, "add_logger"):
            runtime.add_logger(cw_logger)
        elif hasattr(runtime, "configure_logging"):
            runtime.configure_logging(logger=cw_logger)
        else:
            logger.debug("Runtime has no logger hook; CloudWatch logger constructed only.")


def configure_security(runtime: Any, settings: AgentCoreSettings) -> None:
    """Apply security policy if AgentCore exposes a security interface."""
    policy = _optional_factory(
        "agentcore.security",
        "DefaultSecurityPolicy",
        allowed_tools=["mongo_search", "s3_ingest", "bedrock_answer"],
        network_allowlist=settings.security_network_allowlist,
        environment_blocklist=settings.security_environment_blocklist,
    )
    if policy is None:
        return

    if hasattr(runtime, "set_security_policy"):
        runtime.set_security_policy(policy)
    elif hasattr(runtime, "security") and hasattr(runtime.security, "set_policy"):
        runtime.security.set_policy(policy)
    else:
        logger.debug("Runtime does not support security policy injection; skipping.")


# ----------------------------------------------------------------------------
# Configuration payloads
# ----------------------------------------------------------------------------


def build_agentcore_configuration(
    settings: AgentCoreSettings,
    *,
    resources: Optional[AgentResources] = None,
) -> Dict[str, Any]:
    """Return a declarative configuration consumable by AgentCore factories."""
    resources = resources or AgentResources(settings=settings)
    set_runtime_resources(resources)

    bedrock_config = {
        "aws_region": settings.aws_region,
        "aws_access_key_id": settings.aws_access_key_id,
        "aws_secret_access_key": settings.aws_secret_access_key,
        "aws_session_token": settings.aws_session_token,
        "model_id": settings.bedrock_model_id,
        "max_tokens": settings.bedrock_max_tokens,
        "temperature": settings.bedrock_temperature,
    }

    configuration = {
        "runtime": {
            "class_path": "agentcore.runtime.AgentRuntime",
            "options": {
                "deployment_channel": settings.deployment_channel,
            },
        },
        "memory": {
            "short_term": {
                "store": "agentcore.memory.BufferedConversationStore",
                "config": {
                    "max_turns": settings.memory_short_term_max_turns,
                    "token_budget": settings.memory_short_term_token_limit,
                },
            },
            "long_term": {
                "store": "agentcore.memory.MongoDBAtlasStore",
                "config": {
                    "connection_string": settings.mongodb_uri,
                    "database": settings.mongodb_db,
                    "collection": settings.mongodb_collection,
                    "index_name": settings.atlas_search_index,
                    "embedding_dimensions": resources.embedder.n_features,
                    "top_k": settings.memory_long_term_top_k,
                },
            },
        },
        "reasoning": {
            "engine": "agent.agentcore_config.MemoryAwareBedrockReasoner",
            "config": bedrock_config,
        },
        "tools": build_tool_specs(resources),
        "security": {
            "policy": "agentcore.security.DefaultSecurityPolicy",
            "config": {
                "allowed_tools": ["mongo_search", "s3_ingest", "bedrock_answer"],
                "network_allowlist": settings.security_network_allowlist,
                "environment_blocklist": settings.security_environment_blocklist,
            },
        },
        "observability": {
            "logging": {
                "provider": "agentcore.observability.CloudWatchLogger",
                "config": {
                    "log_group": settings.cloudwatch_log_group,
                    "stream_name": settings.deployment_channel,
                },
            },
            "tracing": {
                "provider": "agentcore.observability.CloudWatchTracer",
                "config": {
                    "namespace": settings.cloudwatch_trace_namespace,
                    "log_group": settings.cloudwatch_log_group,
                    "service": "rag-agent",
                },
            },
        },
        "deployment": select_deployment_adapter(settings.deployment_channel),
    }

    if settings.cognito_user_pool_id and settings.cognito_app_client_id:
        configuration["identity"] = {
            "provider": settings.identity_provider,
            "config": {
                "user_pool_id": settings.cognito_user_pool_id,
                "app_client_id": settings.cognito_app_client_id,
                "allowed_groups": settings.identity_allowed_groups,
            },
        }

    return configuration


def initialise_agent_runtime(settings: AgentCoreSettings) -> Any:
    """Instantiate the AgentCore runtime when the library is available.

    Returns the instantiated runtime or, if AgentCore is not installed, the
    declarative configuration so callers can decide how to proceed.
    """
    resources = AgentResources(settings=settings)
    set_runtime_resources(resources)

    config = build_agentcore_configuration(settings=settings, resources=resources)

    runtime_cls = _optional_import("agentcore.runtime", "AgentRuntime")
    if runtime_cls is None:
        logger.info("AgentCore runtime not installed; returning configuration only.")
        return config

    if hasattr(runtime_cls, "from_config"):
        runtime = runtime_cls.from_config(config)
    else:
        runtime = runtime_cls(**config.get("runtime", {}).get("options", {}))

    configure_security(runtime, settings)
    configure_observability(runtime, settings)
    register_tools_with_runtime(runtime, resources)

    reasoner = build_reasoner(settings)
    if hasattr(runtime, "set_reasoner"):
        runtime.set_reasoner(reasoner)
    elif hasattr(runtime, "reasoner"):
        runtime.reasoner = reasoner
    else:
        logger.debug("Runtime does not expose a reasoner setter; attach manually if needed.")

    return runtime


def build_reasoner(settings: AgentCoreSettings) -> MemoryAwareBedrockReasoner:
    """Create a Bedrock-backed reasoner using project services."""
    client = BedrockLLMClient(
        BedrockConfig(
            region=settings.aws_region,
            model_id=settings.bedrock_model_id,
            access_key_id=settings.aws_access_key_id,
            secret_access_key=settings.aws_secret_access_key,
            session_token=settings.aws_session_token,
            max_tokens=settings.bedrock_max_tokens,
            temperature=settings.bedrock_temperature,
        )
    )
    return MemoryAwareBedrockReasoner(
        client,
        max_tokens=settings.bedrock_max_tokens,
        temperature=settings.bedrock_temperature,
    )


def select_deployment_adapter(channel: str) -> Dict[str, Any]:
    """Map a logical channel to an AgentCore deployment adapter."""
    channel = (channel or "cli").lower()
    if channel == "api":
        return {
            "adapter": "agentcore.deployments.api.FastAPIAdapter",
            "config": {"mount_path": "/agent"},
        }
    if channel == "chat":
        return {
            "adapter": "agentcore.deployments.chat.StreamingWebSocketAdapter",
            "config": {"route": "/ws/agent"},
        }
    return {
        "adapter": "agentcore.deployments.cli.CLIAdapter",
        "config": {"prompt": "agent> "},
    }


# ----------------------------------------------------------------------------
# Utilities
# ----------------------------------------------------------------------------


def _optional_import(module_name: str, attr: str) -> Optional[Any]:
    try:
        module = importlib.import_module(module_name)
    except ImportError:
        logger.debug("Module %s not available for optional import", module_name)
        return None
    return getattr(module, attr, None)


def _optional_factory(module_name: str, attr: str, **kwargs: Any) -> Optional[Any]:
    factory = _optional_import(module_name, attr)
    if factory is None:
        return None
    try:
        return factory(**kwargs)
    except TypeError:
        # Some factories expose class types or ignore kwargs.
        if isinstance(factory, type):
            try:
                return factory(**kwargs)
            except TypeError:
                try:
                    return factory()
                except Exception:  # pragma: no cover - environment dependent
                    return None
            except Exception:  # pragma: no cover - environment dependent
                return None
        try:
            return factory()
        except Exception:  # pragma: no cover - environment dependent
            return None
    except Exception:  # pragma: no cover - defensive logging
        logger.debug(
            "Optional factory %s.%s failed to initialise", module_name, attr,
            exc_info=True,
        )
        return None


__all__ = [
    "AgentCoreSettings",
    "AgentResources",
    "MemoryAwareBedrockReasoner",
    "build_agentcore_configuration",
    "initialise_agent_runtime",
    "build_reasoner",
    "build_tool_specs",
    "register_tools_with_runtime",
    "configure_observability",
    "configure_security",
    "mongo_search_tool",
    "mongo_atlas_query_tool",
    "s3_ingest_tool",
    "bedrock_answer_tool",
    "select_deployment_adapter",
    "set_runtime_resources",
]
