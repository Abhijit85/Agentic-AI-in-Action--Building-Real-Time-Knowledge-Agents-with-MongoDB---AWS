"""Prompt helpers shared between the CLI demo and AgentCore tools."""
from __future__ import annotations

from collections.abc import Iterable, Mapping, MutableMapping, Sequence
from typing import List, Optional

from .text_processing import normalize_whitespace

DEFAULT_GROUNDED_INSTRUCTIONS = (
    "You are a media research assistant. Answer using only the supplied YouTube transcript "
    "segments. Highlight the video title or segment scope when relevant, and cite evidence "
    "with bracketed numbers (e.g., [1]). If the context does not answer the question, state "
    "that the information was not mentioned."
)


def _format_timestamp_label(value: object) -> Optional[str]:
    try:
        seconds = float(value)
    except (TypeError, ValueError):
        return None
    if seconds < 0:
        seconds = 0.0
    total_seconds = int(round(seconds))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def _format_time_range(start: object, end: object) -> Optional[str]:
    start_label = _format_timestamp_label(start)
    end_label = _format_timestamp_label(end)
    if start_label and end_label:
        return f"{start_label}–{end_label}"
    return start_label or end_label


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
            source = metadata.get("source") or metadata.get("s3_key")
            video_title = (
                metadata.get("video_title")
                or metadata.get("video", {}).get("title")  # type: ignore[call-arg]
                if isinstance(metadata.get("video"), Mapping)
                else None
            )
            channel = (
                metadata.get("channel")
                or metadata.get("channel_title")
                or metadata.get("video", {}).get("channel")  # type: ignore[call-arg]
                if isinstance(metadata.get("video"), Mapping)
                else None
            )
            speaker = metadata.get("speaker") or metadata.get("speaker_role")
            time_range = _format_time_range(
                metadata.get("start_time"),
                metadata.get("end_time"),
            )
            video_id = metadata.get("video_id")
            video_url = metadata.get("video_url")
            if not video_url and video_id:
                start_seconds = metadata.get("start_time")
                timestamp_suffix = ""
                try:
                    if start_seconds is not None:
                        timestamp_suffix = f"&t={int(max(float(start_seconds), 0))}s"
                except (TypeError, ValueError):
                    timestamp_suffix = ""
                video_url = f"https://www.youtube.com/watch?v={video_id}{timestamp_suffix}"
        else:
            source = None
            video_title = None
            channel = None
            speaker = None
            time_range = None
            video_url = None

        snippet = normalize_whitespace(str(doc.get("text", "")))
        if not snippet:
            continue
        display_title = video_title or source or metadata.get("document_id") if isinstance(metadata, Mapping) else None
        if not display_title:
            display_title = doc.get("chunk_id") if isinstance(doc, Mapping) else "segment"
        header = f"[{idx}] {display_title}"

        detail_pieces = []
        if channel:
            detail_pieces.append(f"Channel: {channel}")
        if speaker:
            detail_pieces.append(f"Speaker: {speaker}")
        if time_range:
            detail_pieces.append(f"Time: {time_range}")
        if video_url:
            detail_pieces.append(f"Link: {video_url}")
        elif source:
            detail_pieces.append(f"Source: {source}")

        if detail_pieces:
            header += "\n" + " | ".join(detail_pieces)

        context_sections.append(f"{header}\n{snippet}")

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
