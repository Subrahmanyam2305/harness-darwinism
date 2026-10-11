"""Unified LLM client backed by AWS Bedrock (Converse API)."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any

import boto3

from src.config import settings

_client: Any = None


def _get_client():
    global _client
    if _client is None:
        _client = boto3.client(
            "bedrock-runtime",
            region_name=settings.aws_default_region,
            aws_access_key_id=settings.aws_access_key_id or None,
            aws_secret_access_key=settings.aws_secret_access_key or None,
            aws_session_token=settings.aws_session_token or None,
        )
    return _client


def reset_client():
    """Force re-creation of the Bedrock client (e.g. after refreshing creds)."""
    global _client
    _client = None


@dataclass(frozen=True)
class LLMResponse:
    """Text plus Bedrock evidence needed by gameplay traces."""

    text: str
    model: str
    input_tokens: int | None
    output_tokens: int | None
    latency_seconds: float


def converse(
    messages: list[dict],
    model: str,
    max_tokens: int = 300,
    temperature: float = 0.3,
    system: str | None = None,
) -> LLMResponse:
    """Send one Bedrock Converse request and retain usage metadata."""
    client = _get_client()

    converse_messages = []
    for message in messages:
        if message["role"] == "system":
            continue
        converse_messages.append(
            {
                "role": message["role"],
                "content": [{"text": message["content"]}],
            }
        )

    merged: list[dict] = []
    for message in converse_messages:
        if merged and merged[-1]["role"] == message["role"]:
            merged[-1]["content"][0]["text"] += "\n" + message["content"][0]["text"]
        else:
            merged.append(message)
    if merged and merged[0]["role"] != "user":
        merged.insert(0, {"role": "user", "content": [{"text": "Begin."}]})

    kwargs: dict[str, Any] = {
        "modelId": model,
        "messages": merged,
        "inferenceConfig": {"maxTokens": max_tokens, "temperature": temperature},
    }
    if system:
        kwargs["system"] = [{"text": system}]

    started = perf_counter()
    response = client.converse(**kwargs)
    latency = perf_counter() - started
    output = response["output"]["message"]["content"]
    usage = response.get("usage", {})
    return LLMResponse(
        text=output[0]["text"] if output else "",
        model=response.get("modelId", model),
        input_tokens=usage.get("inputTokens"),
        output_tokens=usage.get("outputTokens"),
        latency_seconds=latency,
    )


def chat(
    messages: list[dict],
    model: str = "us.anthropic.claude-haiku-4-5-20251001-v1:0",
    max_tokens: int = 300,
    temperature: float = 0.3,
    system: str | None = None,
) -> str:
    """Send a chat request via Bedrock's Converse API.

    Args:
        messages: list of {"role": "user"|"assistant", "content": "..."}
        model: Bedrock model / inference-profile ID
        max_tokens: max response tokens
        temperature: sampling temperature
        system: optional system prompt

    Returns:
        The assistant's text response.
    """
    return converse(messages, model, max_tokens, temperature, system).text
