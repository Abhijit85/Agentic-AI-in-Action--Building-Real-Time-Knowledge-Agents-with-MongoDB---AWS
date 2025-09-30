from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))

from agent.agentcore_config import MemoryAwareBedrockReasoner


class StubBedrockClient:
    """Capture prompts produced by the reasoner without hitting Bedrock."""

    def __init__(self) -> None:
        self.last_prompt = None
        self.stream_calls = []

    def generate(self, *, prompt: str, max_tokens: int, temperature: float) -> str:
        self.last_prompt = prompt
        self.stream_calls.append((prompt, max_tokens, temperature))
        return "stub-response"

    def stream(self, *, prompt: str, max_tokens: int, temperature: float):  # pragma: no cover - streaming is unused in tests
        raise AssertionError("Stream was not expected during these tests")


@pytest.fixture()
def reasoner() -> tuple[MemoryAwareBedrockReasoner, StubBedrockClient]:
    client = StubBedrockClient()
    engine = MemoryAwareBedrockReasoner(client=client, max_tokens=256, temperature=0.1)
    return engine, client


@pytest.fixture()
def short_term_messages() -> list[dict[str, str]]:
    data_path = Path(__file__).parent / "data" / "short_term_conversation.json"
    return json.loads(data_path.read_text(encoding="utf-8"))


@pytest.fixture()
def long_term_documents() -> list[dict[str, object]]:
    data_path = Path(__file__).parent / "data" / "long_term_documents.json"
    return json.loads(data_path.read_text(encoding="utf-8"))


def test_short_term_memory_recent_conversation(reasoner, short_term_messages):
    engine, client = reasoner

    engine.run(
        "Summarise our latest steps",
        memory_snapshot={"short_term": short_term_messages},
    )

    prompt = client.last_prompt
    assert prompt is not None
    assert "Recent conversation:" in prompt

    # Only the last ten messages should be retained in the prompt.
    dropped_message = short_term_messages[0]["content"]
    for message in short_term_messages[-10:]:
        expected_line = f"{message['role'].upper()}: {message['content']}"
        assert expected_line in prompt
    assert dropped_message not in prompt


def test_long_term_memory_documents(reasoner, long_term_documents):
    engine, client = reasoner

    engine.run(
        "What strategic efforts are ongoing?",
        memory_snapshot={"long_term": long_term_documents},
    )

    prompt = client.last_prompt
    assert prompt is not None
    assert "Knowledge retrieved from memory:" in prompt

    for idx, doc in enumerate(long_term_documents, start=1):
        source = doc["metadata"]["source"]
        assert f"[{idx}] {source}" in prompt
        snippet = doc["text"][:40]
        assert snippet in prompt


def test_memory_fallback_when_empty(reasoner):
    engine, client = reasoner

    engine.run("Do you remember anything?", memory_snapshot={})

    prompt = client.last_prompt
    assert prompt is not None
    assert "No prior memory was available" in prompt
