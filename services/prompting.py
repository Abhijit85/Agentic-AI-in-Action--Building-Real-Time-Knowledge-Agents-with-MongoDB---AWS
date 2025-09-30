"""Prompt helpers shared between the CLI demo and AgentCore tools."""
from __future__ import annotations

from collections.abc import Iterable, Mapping, MutableMapping, Sequence
from typing import List

from .text_processing import normalize_whitespace

DEFAULT_GROUNDED_INSTRUCTIONS = (
    "You are an expert assistant answering questions using only the supplied context. "
    "If the answer cannot be derived from the context, reply that you do not know. "
    "Cite sources inline using the bracketed numbers, e.g., [1]."
)


def prepare_context_documents(
    documents: Iterable[Mapping[str, object] | str | Sequence[str] | None]
) -> List[MutableMapping[str, object]]:
    """Normalise context documents for downstream prompt construction.

    Each entry is coerced into a mapping containing at least ``text`` and ``metadata``
    keys. Text payloads are normalised to single-line strings and empty entries are
    discarded. Additional metadata present in the original document is preserved so
    callers can emit citations or scores alongside the final answer.
    """

    prepared: List[MutableMapping[str, object]] = []
    for raw in documents:
        if raw is None:
            continue
        if isinstance(raw, str):
            text = normalize_whitespace(raw)
            if not text:
                continue
            prepared.append({"text": text, "metadata": {}})
            continue

        if isinstance(raw, Mapping):
            doc: MutableMapping[str, object] = dict(raw)
            text_value = doc.get("text") or doc.get("content") or ""
            text = normalize_whitespace(str(text_value))
            if not text:
                continue
            doc["text"] = text
            metadata = doc.get("metadata")
            if not isinstance(metadata, Mapping):
                metadata = {} if metadata is None else dict(metadata)
            else:
                metadata = dict(metadata)
            doc["metadata"] = metadata
            prepared.append(doc)
            continue

        if isinstance(raw, Sequence):  # e.g. list of strings
            text = normalize_whitespace(" ".join(str(item) for item in raw if item))
            if not text:
                continue
            prepared.append({"text": text, "metadata": {}})
            continue

        text = normalize_whitespace(str(raw))
        if not text:
            continue
        prepared.append({"text": text, "metadata": {}})

    return prepared


def build_grounded_answer_prompt(
    question: str,
    documents: Sequence[Mapping[str, object]],
    *,
    instructions: str | None = None,
) -> str:
    """Compose a grounded prompt that includes numbered context passages."""

    context_sections: List[str] = []
    for idx, doc in enumerate(documents, start=1):
        metadata = doc.get("metadata", {}) if isinstance(doc, Mapping) else {}
        if isinstance(metadata, Mapping):
            source = (
                metadata.get("source")
                or metadata.get("s3_key")
                or metadata.get("uri")
                or metadata.get("document_id")
                or doc.get("chunk_id")
            )
        else:
            source = None
        snippet = normalize_whitespace(str(doc.get("text", "")))
        if not snippet:
            continue
        context_sections.append(f"[{idx}] Source: {source or 'unknown'}\n{snippet}")

    context_block = "\n\n".join(context_sections)
    prompt_instructions = instructions.strip() if instructions else DEFAULT_GROUNDED_INSTRUCTIONS

    clean_question = normalize_whitespace(question)
    return (
        f"{prompt_instructions}\n\n"
        f"Question: {clean_question}\n\n"
        f"Context:\n{context_block}\n\n"
        "Answer:"
    )


__all__ = [
    "DEFAULT_GROUNDED_INSTRUCTIONS",
    "build_grounded_answer_prompt",
    "prepare_context_documents",
]
