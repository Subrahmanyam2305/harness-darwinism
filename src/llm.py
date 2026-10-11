"""Unified LLM client backed by AWS Bedrock (Converse API)."""

from __future__ import annotations

import json
from typing import Any

import boto3

_client: Any = None


def _get_client():
    global _client
    if _client is None:
        _client = boto3.client("bedrock-runtime", region_name="us-east-1")
    return _client


def chat(
    messages: list[dict],
    model: str = "anthropic.claude-haiku-4-5-20251001-v1:0",
    max_tokens: int = 300,
    temperature: float = 0.3,
    system: str | None = None,
) -> str:
    """Send a chat request via Bedrock's Converse API.

    Args:
        messages: list of {"role": "user"|"assistant", "content": "..."}
        model: Bedrock model ID
        max_tokens: max response tokens
        temperature: sampling temperature
        system: optional system prompt (sent separately from messages)

    Returns:
        The assistant's text response.
    """
    client = _get_client()

    converse_messages = []
    for m in messages:
        role = m["role"]
        if role == "system":
            continue
        converse_messages.append({
            "role": role,
            "content": [{"text": m["content"]}],
        })

    # Bedrock Converse requires alternating user/assistant turns.
    # Merge consecutive same-role messages.
    merged: list[dict] = []
    for msg in converse_messages:
        if merged and merged[-1]["role"] == msg["role"]:
            merged[-1]["content"][0]["text"] += "\n" + msg["content"][0]["text"]
        else:
            merged.append(msg)

    # Ensure first message is from user
    if merged and merged[0]["role"] != "user":
        merged.insert(0, {"role": "user", "content": [{"text": "Begin."}]})

    kwargs: dict[str, Any] = {
        "modelId": model,
        "messages": merged,
        "inferenceConfig": {
            "maxTokens": max_tokens,
            "temperature": temperature,
        },
    }

    if system:
        kwargs["system"] = [{"text": system}]

    response = client.converse(**kwargs)
    output = response["output"]["message"]["content"]
    return output[0]["text"] if output else ""
