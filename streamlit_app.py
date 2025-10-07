from __future__ import annotations

import textwrap
import time
from typing import Any, Dict, List, Optional, Tuple

import streamlit as st
from dotenv import load_dotenv

from agent.agentcore_config import (
    AgentCoreSettings,
    AgentResources,
    build_reasoner,
    mongo_search_tool,
    s3_ingest_tool,
    set_runtime_resources,
)
from services.llm_client import LLMInvocationError

load_dotenv()

st.set_page_config(
    page_title="Enterprise Knowledge Agent",
    page_icon="KB",
    layout="wide",
)


def _format_timestamp(value: Any) -> Optional[str]:
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


def _format_time_range(metadata: Dict[str, Any]) -> Optional[str]:
    start = _format_timestamp(metadata.get("start_time"))
    end = _format_timestamp(metadata.get("end_time"))
    if start and end:
        return f"{start} – {end}"
    return start or end


def _build_video_url(metadata: Dict[str, Any]) -> Optional[str]:
    url = metadata.get("video_url")
    video_id = metadata.get("video_id")
    if not url and video_id:
        timestamp = metadata.get("start_time")
        suffix = ""
        try:
            if timestamp is not None:
                suffix = f"&t={int(max(float(timestamp), 0))}s"
        except (TypeError, ValueError):
            suffix = ""
        url = f"https://www.youtube.com/watch?v={video_id}{suffix}"
    return url


@st.cache_resource(show_spinner=False)
def init_runtime() -> Tuple[AgentCoreSettings, AgentResources, Any]:
    """Initialise shared settings, resources, and the Bedrock-backed reasoner."""
    settings = AgentCoreSettings.from_env()
    resources = AgentResources(settings=settings)
    set_runtime_resources(resources)
    reasoner = build_reasoner(settings)
    return settings, resources, reasoner


def ensure_session_state() -> None:
    if "messages" not in st.session_state:
        st.session_state.messages: List[Dict[str, Any]] = []


def render_history() -> None:
    """Replay chat history stored in the Streamlit session."""
    for message in st.session_state.messages:
        container = st.chat_message(message["role"])
        container.markdown(message["content"])
        if message["role"] == "assistant":
            render_sources(container, message.get("sources") or [], message.get("metrics"))


def render_sources(container: Any, sources: List[Dict[str, Any]], metrics: Dict[str, Any] | None) -> None:
    if not sources and not metrics:
        return

    with container.expander("Context & citations", expanded=False):
        if metrics:
            context_size = int(metrics.get("context_size", 0))
            retrieval_ms = metrics.get("retrieval_ms")
            generation_ms = metrics.get("generation_ms")
            cols = st.columns(3)
            cols[0].metric("Passages", context_size)
            cols[1].metric("Retrieval (ms)", f"{retrieval_ms:.0f}" if retrieval_ms is not None else "N/A")
            cols[2].metric("Response (ms)", f"{generation_ms:.0f}" if generation_ms is not None else "N/A")

        if not sources:
            st.info("No knowledge base passages were retrieved for this answer.")
            return

        for idx, doc in enumerate(sources, start=1):
            snippet = textwrap.shorten(str(doc.get("text", "")), width=280, placeholder=" ...")
            metadata = doc.get("metadata", {}) if isinstance(doc.get("metadata"), dict) else {}
            source_name = metadata.get("source") or metadata.get("s3_key") or doc.get("chunk_id") or f"Passage {idx}"
            video_title = metadata.get("video_title") or source_name
            channel = metadata.get("channel") or metadata.get("channel_title")
            speaker = metadata.get("speaker") or metadata.get("speaker_role")
            time_range = _format_time_range(metadata)
            watch_url = _build_video_url(metadata)
            score = doc.get("search_score") or doc.get("vector_score")
            score_text = f"{float(score):.3f}" if isinstance(score, (int, float)) else "n/a"

            st.markdown(f"**[{idx}] {video_title}**")

            detail_parts: List[str] = [f"Score: {score_text}"]
            if channel:
                detail_parts.append(f"Channel: {channel}")
            if speaker:
                detail_parts.append(f"Speaker: {speaker}")
            if time_range:
                detail_parts.append(f"Segment: {time_range}")
            if watch_url:
                detail_parts.append(f"[Watch segment]({watch_url})")

            if detail_parts:
                st.caption(" | ".join(detail_parts))
            if snippet:
                st.markdown(f"> {snippet}")


def build_short_term_window(max_turns: int) -> List[Dict[str, str]]:
    conversation: List[Dict[str, str]] = []
    for entry in st.session_state.messages:
        if entry["role"] in {"user", "assistant"}:
            conversation.append({"role": entry["role"], "content": entry["content"]})
    if max_turns <= 0:
        return conversation
    window = max_turns * 2
    return conversation[-window:]


def sidebar_controls(settings: AgentCoreSettings) -> Dict[str, Any]:
    with st.sidebar:
        st.header("Session")
        if st.button("Reset conversation", use_container_width=True):
            st.session_state.messages = []
            st.experimental_rerun()

        st.header("Chat settings")
        suggested_k = max(1, min(settings.memory_long_term_top_k, 10))
        top_k = st.slider(
            "Context passages",
            min_value=1,
            max_value=10,
            value=suggested_k,
            help="Maximum number of passages to retrieve from MongoDB Atlas per question.",
        )
        max_tokens_default = max(256, min(settings.bedrock_max_tokens, 2048))
        max_tokens = st.slider(
            "Max tokens",
            min_value=128,
            max_value=4096,
            value=max_tokens_default,
            step=64,
            help="Upper bound for the response length returned by Bedrock.",
        )
        temperature = st.slider(
            "Temperature",
            min_value=0.0,
            max_value=1.0,
            value=min(max(settings.bedrock_temperature, 0.0), 1.0),
            step=0.05,
            help="Lower values keep answers focused; higher values increase creativity.",
        )
        streaming = st.checkbox("Stream responses", value=False, help="Stream tokens as they arrive from Bedrock.")

        st.header("Knowledge base")
        with st.expander("MongoDB & S3 configuration", expanded=False):
            st.caption(
                f"S3 bucket: `{settings.s3_bucket}`\n\n"
                f"Atlas database: `{settings.mongodb_db}`\n\n"
                f"Collection: `{settings.mongodb_collection}`"
            )
        ingest_limit = st.slider(
            "Documents per ingest run",
            min_value=1,
            max_value=50,
            value=5,
            help="Caps the number of S3 objects processed when running ingestion from the UI.",
        )
        chunk_size = st.slider(
            "Chunk size (words)",
            min_value=100,
            max_value=1000,
            value=max(100, settings.ingestion_chunk_words),
            step=50,
        )
        overlap = st.slider(
            "Chunk overlap (words)",
            min_value=0,
            max_value=200,
            value=max(0, settings.ingestion_chunk_overlap),
            step=10,
        )
        ingest_placeholder = st.empty()
        if st.button("Run S3 ingestion", use_container_width=True):
            ingest_placeholder.info("Ingesting documents from S3...")
            try:
                result = s3_ingest_tool(
                    {
                        "document_limit": int(ingest_limit),
                        "chunk_size": int(chunk_size),
                        "chunk_overlap": int(overlap),
                        "dry_run": False,
                    }
                )
            except Exception as exc:  # noqa: BLE001 - show integration errors to the operator
                ingest_placeholder.error(f"Ingestion failed: {exc}")
            else:
                docs = result.get("documents_processed", 0)
                chunks = result.get("chunks_written", 0)
                ingest_placeholder.success(
                    f"Processed {docs} document{'s' if docs != 1 else ''} and upserted {chunks} chunk{'s' if chunks != 1 else ''}."
                )

        st.header("Media filters")
        with st.expander("Filter YouTube segments", expanded=False):
            channel_filter = st.text_input(
                "Channel",
                help="Match the exact YouTube channel name to narrow the search.",
            )
            speaker_filter = st.text_input(
                "Speaker",
                help="Filter by diarised speaker label or role when available.",
            )
            video_filter = st.text_input(
                "Video title",
                help="Restrict results to a specific YouTube video title.",
            )
            video_id_filter = st.text_input(
                "Video ID",
                help="Filter by the YouTube video ID (e.g., dQw4w9WgXcQ).",
            )
            playlist_filter = st.text_input(
                "Playlist ID",
                help="Filter segments that came from a specific playlist ingestion.",
            )
            start_filter = st.text_input(
                "Start after (seconds)",
                help="Only return segments that begin after this timestamp.",
            )
            end_filter = st.text_input(
                "End before (seconds)",
                help="Only return segments that end before this timestamp.",
            )

    def _clean_filter(text: str) -> Optional[str]:
        value = text.strip()
        return value or None

    def _parse_time(raw: str) -> Optional[float]:
        try:
            value = float(raw)
        except (TypeError, ValueError):
            return None
        if value < 0:
            return None
        return value

    filters: Dict[str, Any] = {}
    if channel_filter and _clean_filter(channel_filter):
        filters["channel"] = _clean_filter(channel_filter)
    if speaker_filter and _clean_filter(speaker_filter):
        filters["speaker"] = _clean_filter(speaker_filter)
    if video_filter and _clean_filter(video_filter):
        filters["video_title"] = _clean_filter(video_filter)
    if video_id_filter and _clean_filter(video_id_filter):
        filters["video_id"] = _clean_filter(video_id_filter)
    if playlist_filter and _clean_filter(playlist_filter):
        filters["playlist_id"] = _clean_filter(playlist_filter)

    start_value = _parse_time(start_filter) if start_filter else None
    end_value = _parse_time(end_filter) if end_filter else None
    if start_value is not None:
        filters["start_time"] = start_value
    if end_value is not None:
        filters["end_time"] = end_value

    return {
        "top_k": int(top_k),
        "max_tokens": int(max_tokens),
        "temperature": float(temperature),
        "streaming": bool(streaming),
        "filters": filters,
    }


def main() -> None:
    try:
        settings, _resources, reasoner = init_runtime()
    except ValueError as exc:
        st.error(f"Configuration error: {exc}")
        st.stop()
    except Exception as exc:  # noqa: BLE001 - surface full initialisation failures
        st.error(f"Unable to start the agent: {exc}")
        st.stop()

    ensure_session_state()
    controls = sidebar_controls(settings)

    st.title("Enterprise Knowledge Agent")
    st.caption(
        "Conversational interface backed by MongoDB Atlas Search, Amazon S3, and Amazon Bedrock."
    )

    render_history()

    prompt = st.chat_input("Ask a question about your knowledge base...")
    if not prompt:
        return

    user_message = {"role": "user", "content": prompt}
    st.session_state.messages.append(user_message)
    st.chat_message("user").markdown(prompt)

    assistant_container = st.chat_message("assistant")

    try:
        retrieval_start = time.perf_counter()
        payload: Dict[str, Any] = {"query": prompt, "top_k": controls["top_k"]}
        if controls.get("filters"):
            payload["filters"] = controls["filters"]
        retrieval = mongo_search_tool(payload)
        retrieval_ms = (time.perf_counter() - retrieval_start) * 1000
        matches = retrieval.get("matches", []) if isinstance(retrieval, dict) else []

        memory_snapshot: Dict[str, Any] = {
            "short_term": build_short_term_window(settings.memory_short_term_max_turns),
        }
        if matches:
            memory_snapshot["long_term"] = matches

        answer_tokens: List[str] = []
        generation_start = time.perf_counter()

        if controls["streaming"]:
            def _on_token(token: str) -> None:
                answer_tokens.append(token)
                assistant_container.markdown("".join(answer_tokens))

            answer_text = reasoner.run(
                prompt,
                memory_snapshot=memory_snapshot,
                stream=True,
                on_token=_on_token,
                max_tokens=controls["max_tokens"],
                temperature=controls["temperature"],
            )
        else:
            answer_text = reasoner.run(
                prompt,
                memory_snapshot=memory_snapshot,
                stream=False,
                max_tokens=controls["max_tokens"],
                temperature=controls["temperature"],
            )
            assistant_container.markdown(answer_text)

        generation_ms = (time.perf_counter() - generation_start) * 1000

        metrics = {
            "context_size": len(matches),
            "retrieval_ms": retrieval_ms,
            "generation_ms": generation_ms,
        }

        render_sources(assistant_container, matches, metrics)

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": answer_text,
                "sources": matches,
                "metrics": metrics,
            }
        )
    except LLMInvocationError as exc:
        assistant_container.error(f"Bedrock invocation failed: {exc}")
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": f"[Warning] Bedrock invocation failed: {exc}",
            }
        )
    except Exception as exc:  # noqa: BLE001 - ensure unexpected errors reach the operator
        assistant_container.error(f"Agent pipeline failed: {exc}")
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": f"[Warning] Agent pipeline failed: {exc}",
            }
        )


if __name__ == "__main__":
    main()
