"""Run a genome in Crafter with validated decisions and complete evidence."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from typing import Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, ValidationError

from src.crafter_env import ACTIONS, CrafterTextEnv
from src.gameplay_provider import BedrockProvider, GameplayProvider, ProviderError
from src.genome import Genome

TerminationReason = Literal["death", "step_limit", "agent_error"]


@dataclass
class EpisodeTrace:
    """Serializable evidence from one Crafter episode."""

    genome_id: str
    seed: int
    score: float
    achievements: dict[str, bool]
    steps: int
    history: list[dict] = field(default_factory=list)
    failure_summary: str = ""
    termination_reason: TerminationReason = "step_limit"
    attempts: list[dict] = field(default_factory=list)
    provider: str = "unknown"
    model: str | None = None
    is_stub: bool = False
    input_tokens: int | None = None
    output_tokens: int | None = None
    cost_usd: float | None = None
    latency_seconds: float = 0.0
    error: str | None = None
    harness: dict = field(default_factory=dict)
    max_steps: int = 200
    schema_version: int = 1

    def to_dict(self) -> dict:
        """Return JSON-compatible data for persistence and API consumers."""
        return asdict(self)


@dataclass(frozen=True)
class EpisodeFrame:
    """Rendered reset or post-action state emitted to visual consumers."""

    step: int
    action: str | None
    rgb: np.ndarray
    observation: str
    achievements: dict[str, bool]
    inventory: dict[str, int]
    is_stub: bool


class _Decision(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    action: str
    thought: str = ""


def _build_system_prompt(genome: Genome) -> str:
    """Combine fixed game mechanics with the genome's evolvable harness."""
    config = genome.genome
    parts = [
        config.system_prompt,
        "",
        "## Fixed Game Mechanics",
        "Moving changes the direction you face. The 'do' action interacts with the adjacent tile.",
        "Use 'do' while facing a tree to collect wood. A table costs 2 wood.",
        "A wood pickaxe costs 1 wood and must be crafted near a table.",
        (
            "If a target is at distance 1, first choose move_<direction> to face it even if "
            "the tile is blocked, then choose do on the next turn."
        ),
        "Navigate to the nearest target by reducing one reported direction and distance at a time.",
        "Each response executes exactly one game action.",
    ]
    if config.rules:
        parts.extend(["", "## Strategy Rules"])
        parts.extend(f"{index}. {rule}" for index, rule in enumerate(config.rules, 1))
    parts.extend(
        [
            "",
            "## Available Actions",
            ", ".join(config.tools),
            "",
            "## Response Format",
            'Respond with ONLY one JSON object: {"thought":"brief reasoning","action":"action_name"}.',
        ]
    )
    return "\n".join(parts)


def _parse_action(response: str, allowed_actions: list[str]) -> str:
    """Return one allowlisted action or reject the entire model response."""
    try:
        decision = _Decision.model_validate_json(response)
    except ValidationError as error:
        raise ValueError("Expected exactly one JSON action object") from error
    if decision.action not in allowed_actions:
        raise ValueError("Action is outside this genome's tool allowlist")
    return decision.action


def _validate_episode(genome: Genome, max_steps: int) -> None:
    config = genome.genome
    if max_steps <= 0:
        raise ValueError("max_steps must be positive")
    if not config.tools or len(config.tools) != len(set(config.tools)):
        raise ValueError("tools must contain unique actions")
    if set(config.tools) - set(ACTIONS):
        raise ValueError("tools contains an unknown Crafter action")
    if not 0 <= config.context_budget.history_steps <= 200:
        raise ValueError("history_steps must be between 0 and 200")
    if not 0 <= config.memory_policy.max_items <= 200:
        raise ValueError("max_items must be between 0 and 200")


def _sum_if_complete(attempts: list[dict], key: str) -> int | float | None:
    values = [attempt[key] for attempt in attempts]
    if not values or any(value is None for value in values):
        return None
    return sum(values)


def play_episode(
    genome: Genome,
    seed: int,
    max_steps: int = 200,
    *,
    provider: GameplayProvider | None = None,
    on_frame: Callable[[EpisodeFrame], None] | None = None,
) -> EpisodeTrace:
    """Play until death, step limit, or three failed attempts at one decision."""
    _validate_episode(genome, max_steps)
    provider = provider or BedrockProvider()
    environment = CrafterTextEnv(seed=seed, max_steps=max_steps)
    observation = environment.reset()
    config = genome.genome
    system_message = {"role": "system", "content": _build_system_prompt(genome)}
    turns: list[dict[str, str]] = []
    notes: list[str] = []
    history: list[dict] = []
    attempts: list[dict] = []
    termination_reason: TerminationReason = "step_limit"
    terminal_error: str | None = None
    last_model: str | None = None

    def emit(action: str | None) -> None:
        if on_frame is None:
            return
        on_frame(
            EpisodeFrame(
                step=environment.step_count,
                action=action,
                rgb=environment.render(),
                observation=observation,
                achievements=environment.get_achievements(),
                inventory=dict(environment.info.get("inventory", {})),
                is_stub=provider.is_stub,
            )
        )

    emit(None)
    while environment.step_count < max_steps:
        memory_text = ""
        if config.memory_policy.type != "none" and config.memory_policy.max_items:
            retained = notes[-config.memory_policy.max_items :]
            if retained:
                separator = "; " if config.memory_policy.type == "summary" else "\n- "
                memory_text = (
                    f"\nMemory ({config.memory_policy.type}): "
                    + separator.join(retained)
                )
        user_message = {
            "role": "user",
            "content": (
                f"Observation:\n{observation}{memory_text}\n"
                f"Step {environment.step_count + 1}/{max_steps}. Choose one action."
            ),
        }
        retained_turns = (
            turns[-2 * config.context_budget.history_steps :]
            if config.context_budget.history_steps
            else []
        )
        messages = [system_message, *retained_turns, user_message]
        action: str | None = None
        reply_text = ""

        for attempt_number in range(1, 4):
            attempt = {
                "step": environment.step_count,
                "attempt": attempt_number,
                "input_tokens": None,
                "output_tokens": None,
                "cost_usd": None,
                "latency_seconds": 0.0,
                "model": None,
                "reply": None,
            }
            try:
                reply = provider.complete(messages)
            except ProviderError:
                attempt["error"] = "provider_error"
            else:
                reply_text = reply.text
                last_model = reply.model
                attempt.update(
                    {
                        "reply": reply.text,
                        "model": reply.model,
                        "input_tokens": reply.input_tokens,
                        "output_tokens": reply.output_tokens,
                        "cost_usd": reply.cost_usd,
                        "latency_seconds": reply.latency_seconds,
                    }
                )
                try:
                    action = _parse_action(reply.text, config.tools)
                except ValueError:
                    attempt["error"] = "invalid_output"
                else:
                    attempt["action"] = action
            attempts.append(attempt)
            if action is not None:
                break
            messages = [
                *messages,
                {
                    "role": "user",
                    "content": (
                        "The previous attempt failed validation. Return only one JSON object "
                        "whose action is in the available action list."
                    ),
                },
            ]

        if action is None:
            termination_reason = "agent_error"
            terminal_error = (
                f"Three inference attempts failed at game step {environment.step_count}"
            )
            break

        previous_achievements = environment.get_achievements()
        previous_observation = observation
        observation, reward, done, info = environment.step(action)
        turns.extend([user_message, {"role": "assistant", "content": reply_text}])
        turns = (
            turns[-2 * config.context_budget.history_steps :]
            if config.context_budget.history_steps
            else []
        )
        for achievement, unlocked in environment.get_achievements().items():
            if unlocked and not previous_achievements[achievement]:
                notes.append(f"Step {environment.step_count}: unlocked {achievement}")
        if info["inventory"].get("health", 9) <= 2:
            notes.append(f"Step {environment.step_count}: critically low health")
        notes = (
            notes[-config.memory_policy.max_items :]
            if config.memory_policy.max_items
            else []
        )
        history.append(
            {
                "step": environment.step_count,
                "action": action,
                "reward": float(reward),
                "observation_before": previous_observation,
                "observation_after": observation,
                "achievements": environment.get_achievements(),
            }
        )
        emit(action)
        if done:
            termination_reason = (
                "death" if info["inventory"].get("health", 0) <= 0 else "step_limit"
            )
            break

    achievements = environment.get_achievements()
    unlocked = [name for name, value in achievements.items() if value]
    failure_summary = (
        f"Termination: {termination_reason}. Played {environment.step_count}/{max_steps} "
        f"steps. Unlocked: {', '.join(unlocked) if unlocked else 'none'}."
    )
    return EpisodeTrace(
        genome_id=genome.id,
        seed=seed,
        score=environment.get_score(),
        achievements=achievements,
        steps=environment.step_count,
        history=history,
        failure_summary=failure_summary,
        termination_reason=termination_reason,
        attempts=attempts,
        provider=provider.name,
        model=last_model,
        is_stub=provider.is_stub,
        input_tokens=_sum_if_complete(attempts, "input_tokens"),
        output_tokens=_sum_if_complete(attempts, "output_tokens"),
        cost_usd=_sum_if_complete(attempts, "cost_usd"),
        latency_seconds=float(sum(attempt["latency_seconds"] for attempt in attempts)),
        error=terminal_error,
        harness=json.loads(config.model_dump_json()),
        max_steps=max_steps,
    )
