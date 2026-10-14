"""Inference provider contract used by Crafter gameplay."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Protocol

from botocore.exceptions import BotoCoreError, ClientError

from src.config import settings
from src.llm import converse


@dataclass(frozen=True)
class ModelReply:
    """One model reply with usage metadata when the provider reports it."""

    text: str
    model: str
    input_tokens: int | None = None
    output_tokens: int | None = None
    cost_usd: float | None = None
    latency_seconds: float = 0.0
    cache_read_tokens: int | None = None
    cache_write_tokens: int | None = None


class ProviderError(RuntimeError):
    """Safe expected inference failure with no credential-bearing message."""


class GameplayProvider(Protocol):
    """Boundary that lets gameplay run independently of a model transport."""

    name: str
    is_stub: bool

    def complete(self, messages: list[dict[str, str]]) -> ModelReply:
        """Return one decision attempt without hidden retries."""
        ...


class BedrockProvider:
    """AWS Bedrock Converse transport for the frozen gameplay model."""

    name = "aws-bedrock"
    is_stub = False

    def __init__(self, model: str | None = None):
        self.model = model or settings.agent_model

    def complete(self, messages: list[dict[str, str]]) -> ModelReply:
        system = next(
            (message["content"] for message in messages if message["role"] == "system"),
            None,
        )
        try:
            cache_system = bool(
                system and "claude-haiku-4-5" in self.model and len(system) // 4 >= 4096
            )
            response = converse(
                messages=[
                    message for message in messages if message["role"] != "system"
                ],
                system=system,
                model=self.model,
                max_tokens=150,
                temperature=0.0,
                cache_system=cache_system,
            )
        except (BotoCoreError, ClientError) as error:
            raise ProviderError("Bedrock inference failed") from error
        return ModelReply(
            text=response.text,
            model=response.model,
            input_tokens=response.input_tokens,
            output_tokens=response.output_tokens,
            latency_seconds=response.latency_seconds,
            cache_read_tokens=response.cache_read_tokens,
            cache_write_tokens=response.cache_write_tokens,
        )


class ScriptedProvider:
    """Clearly labeled fixed actions for offline plumbing and replay checks."""

    name = "offline-scripted-check"
    is_stub = True

    def __init__(self, actions: list[str] | None = None):
        self.actions = actions or [
            "move_right",
            "move_down",
            "move_left",
            "move_up",
            "do",
        ]
        self.calls = 0

    def complete(self, messages: list[dict[str, str]]) -> ModelReply:
        """Return the next fixed action without interpreting the observation."""
        action = self.actions[self.calls % len(self.actions)]
        self.calls += 1
        return ModelReply(
            text=json.dumps({"action": action}),
            model="offline-script",
            input_tokens=0,
            output_tokens=0,
            cost_usd=0.0,
        )
