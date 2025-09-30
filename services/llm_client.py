"""LLM client abstraction backed by Amazon Bedrock."""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Generator, Iterable, Optional

import boto3
from botocore.exceptions import BotoCoreError, ClientError


class LLMInvocationError(RuntimeError):
    """Raised when an LLM invocation fails."""


@dataclass
class BedrockConfig:
    """Configuration required to call Amazon Bedrock."""

    region: str
    model_id: str
    access_key_id: Optional[str] = None
    secret_access_key: Optional[str] = None
    session_token: Optional[str] = None
    max_tokens: int = 512
    temperature: float = 0.2


class BedrockLLMClient:
    """Thin wrapper around the Bedrock Runtime API."""

    def __init__(self, config: BedrockConfig):
        self._config = config
        session_kwargs = {"region_name": config.region}
        if config.access_key_id and config.secret_access_key:
            session_kwargs.update(
                aws_access_key_id=config.access_key_id,
                aws_secret_access_key=config.secret_access_key,
            )
        if config.session_token:
            session_kwargs["aws_session_token"] = config.session_token
        session = boto3.session.Session(**session_kwargs)
        self._client = session.client("bedrock-runtime")

    def generate(
        self,
        *,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> str:
        """Return the full response body for the given prompt."""
        body = json.dumps(
            {
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": max_tokens or self._config.max_tokens,
                "temperature": (
                    temperature if temperature is not None else self._config.temperature
                ),
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": prompt,
                            }
                        ],
                    }
                ],
            }
        )

        try:
            response = self._client.invoke_model(modelId=self._config.model_id, body=body)
        except (BotoCoreError, ClientError) as exc:  # pragma: no cover - network boundary
            raise LLMInvocationError(str(exc)) from exc

        payload = json.loads(response["body"].read())
        return _collect_text_blocks(payload.get("content", []))

    def stream(
        self,
        *,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> Generator[str, None, None]:
        """Yield tokens from the streaming response."""
        body = json.dumps(
            {
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": max_tokens or self._config.max_tokens,
                "temperature": (
                    temperature if temperature is not None else self._config.temperature
                ),
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": prompt,
                            }
                        ],
                    }
                ],
            }
        )

        try:
            response = self._client.invoke_model_with_response_stream(
                modelId=self._config.model_id,
                body=body,
            )
        except (BotoCoreError, ClientError) as exc:  # pragma: no cover - network boundary
            raise LLMInvocationError(str(exc)) from exc

        stream = response.get("body")
        if stream is None:
            return

        for event in stream:
            chunk = event.get("chunk")
            if not chunk:
                continue
            payload = json.loads(chunk["bytes"].decode("utf-8"))
            for text in _iter_text_deltas(payload):
                yield text


def _collect_text_blocks(blocks: Iterable[dict]) -> str:
    """Concatenate text blocks returned by Claude-compatible models."""
    segments = []
    for block in blocks:
        if block.get("type") == "text":
            segments.append(block.get("text", ""))
    return "".join(segments)


def _iter_text_deltas(payload: dict) -> Iterable[str]:
    """Yield text deltas from a streaming Bedrock payload."""
    if "delta" in payload:
        for entry in payload["delta"].get("content", []):
            if entry.get("type") == "text_delta":
                text = entry.get("text", "")
                if text:
                    yield text
    elif "content" in payload:
        # Some models send initial content blocks instead of deltas.
        text = _collect_text_blocks(payload.get("content", []))
        if text:
            yield text
