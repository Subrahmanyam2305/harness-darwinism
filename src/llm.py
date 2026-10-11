"""Unified LLM client backed by AWS Bedrock (Converse API)."""

from __future__ import annotations

import os
from typing import Any

import boto3

_client: Any = None


def _get_client():
    global _client
    if _client is None:
        _client = boto3.client(
            "bedrock-runtime",
            region_name=os.environ.get("AWS_DEFAULT_REGION", "us-east-1"),
            aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY"),
            aws_session_token=os.environ.get("AWS_SESSION_TOKEN"),
        )
    return _client


def reset_client():
    """Force re-creation of the Bedrock client (e.g. after refreshing creds)."""
    global _client
    _client = None


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
    client = _get_client()

    converse_messages = []
    for m in messages:
        if m["role"] == "system":
            continue
        converse_messages.append({
            "role": m["role"],
            "content": [{"text": m["content"]}],
        })

    # Merge consecutive same-role messages (Bedrock requires alternating turns)
    merged: list[dict] = []
    for msg in converse_messages:
        if merged and merged[-1]["role"] == msg["role"]:
            merged[-1]["content"][0]["text"] += "\n" + msg["content"][0]["text"]
        else:
            merged.append(msg)

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
