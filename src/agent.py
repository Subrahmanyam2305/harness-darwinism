"""Run a genome in Crafter with validated decisions and complete evidence."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from typing import Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, ValidationError

from src.crafter_env import ACTIONS, CrafterTextEnv
from src.execution_state import ExecutionState, StatePatch, apply_state_patch
from src.gameplay_provider import BedrockProvider, GameplayProvider, ProviderError
from src.genome import Genome
from src.skills import SkillRequest, SkillStep, execute_skill, feasible_skills

TerminationReason = Literal["death", "step_limit", "agent_error"]
ControllerMode = Literal["primitive", "skills"]


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
    artifact_path: str | None = None
    max_steps: int = 200
    controller: ControllerMode = "primitive"
    decisions: list[dict] = field(default_factory=list)
    model_calls: int = 0
    cache_read_tokens: int | None = None
    cache_write_tokens: int | None = None
    schema_version: int = 2

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


class _SkillDecision(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    state_patch: StatePatch = StatePatch()
    skill: SkillRequest


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


def _build_skill_system_prompt(genome: Genome) -> str:
    """Build static instructions for compact state plus guarded skills."""
    config = genome.genome
    parts = [
        config.system_prompt,
        "",
        "## Fixed Game Mechanics",
        "Choose one feasible high-level skill. Python executes its primitive actions and checks the real world after every step.",
        "Do not invent skills or targets outside the supplied feasible list.",
        "Keep goal and subgoal short. World facts in the latest observation override remembered state.",
    ]
    if config.rules:
        parts.extend(["", "## Strategy Rules"])
        parts.extend(f"{index}. {rule}" for index, rule in enumerate(config.rules, 1))
    parts.extend(
        [
            "",
            "## Response Format",
            "Return only JSON with state_patch and skill. Example:",
            '{"state_patch":{"goal":"collect wood","subgoal":"reach tree"},"skill":{"name":"acquire","target":"tree","max_steps":3}}',
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


def _skill_option(request: SkillRequest) -> str:
    parameter = request.target or request.goal or request.need
    return f"{request.name}:{parameter}" if parameter else request.name


def _parse_skill_decision(response: str, feasible: list[str]) -> _SkillDecision:
    try:
        decision = _SkillDecision.model_validate_json(response)
    except ValidationError as error:
        raise ValueError("Expected one JSON state patch and skill request") from error
    if _skill_option(decision.skill) not in feasible:
        raise ValueError("Skill request is not feasible in the current Crafter state")
    return decision


def _play_skill_episode(
    genome: Genome,
    seed: int,
    max_steps: int,
    provider: GameplayProvider,
    on_frame: Callable[[EpisodeFrame], None] | None,
    max_skill_steps: int,
) -> EpisodeTrace:
    """Run compact model decisions followed by guarded local environment steps."""
    environment = CrafterTextEnv(seed=seed, max_steps=max_steps)
    observation = environment.reset()
    config = genome.genome
    state = ExecutionState()
    system_message = {"role": "system", "content": _build_skill_system_prompt(genome)}
    history: list[dict] = []
    attempts: list[dict] = []
    decisions: list[dict] = []
    termination_reason: TerminationReason = "step_limit"
    terminal_error: str | None = None
    last_model: str | None = None
    empty_decisions = 0

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
        snapshot = environment.snapshot()
        feasible = feasible_skills(snapshot, config.tools)
        if not feasible:
            termination_reason = "agent_error"
            terminal_error = "No feasible skills for the current genome tool allowlist"
            break
        user_message = {
            "role": "user",
            "content": (
                f"Execution state:\n{state.model_dump_json()}\n\n"
                f"Latest observation:\n{observation}\n\n"
                f"Feasible skills: {', '.join(feasible)}\n"
                f"Step {environment.step_count + 1}/{max_steps}. Choose one skill."
            ),
        }
        messages = [system_message, user_message]
        parsed: _SkillDecision | None = None
        decision_id = f"decision-{len(decisions) + 1}"
        decision_attempts: list[dict] = []

        for attempt_number in range(1, 4):
            attempt = {
                "decision_id": decision_id,
                "step": environment.step_count,
                "attempt": attempt_number,
                "input_tokens": None,
                "output_tokens": None,
                "cache_read_tokens": None,
                "cache_write_tokens": None,
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
                last_model = reply.model
                attempt.update(
                    {
                        "reply": reply.text,
                        "model": reply.model,
                        "input_tokens": reply.input_tokens,
                        "output_tokens": reply.output_tokens,
                        "cache_read_tokens": getattr(reply, "cache_read_tokens", None),
                        "cache_write_tokens": getattr(
                            reply, "cache_write_tokens", None
                        ),
                        "cost_usd": reply.cost_usd,
                        "latency_seconds": reply.latency_seconds,
                    }
                )
                try:
                    parsed = _parse_skill_decision(reply.text, feasible)
                except ValueError:
                    attempt["error"] = "invalid_output"
                else:
                    attempt["skill"] = parsed.skill.model_dump()
            attempts.append(attempt)
            decision_attempts.append(attempt)
            if parsed is not None:
                break
            messages = [
                system_message,
                user_message,
                {
                    "role": "user",
                    "content": "The prior response failed validation. Return only JSON using one supplied feasible skill.",
                },
            ]

        if parsed is None:
            termination_reason = "agent_error"
            terminal_error = (
                f"Three inference attempts failed at game step {environment.step_count}"
            )
            break

        state_before = state.model_dump()
        state = apply_state_patch(state, parsed.state_patch)
        request = parsed.skill.model_copy(
            update={
                "max_steps": min(
                    parsed.skill.max_steps,
                    max_skill_steps,
                    max_steps - environment.step_count,
                )
            }
        )
        step_observation = observation

        def record_step(
            step: SkillStep,
            *,
            _decision_id: str = decision_id,
            _request: SkillRequest = request,
        ) -> None:
            nonlocal observation, step_observation
            history.append(
                {
                    "step": step.snapshot.step,
                    "decision_id": _decision_id,
                    "skill": _request.name,
                    "action": step.action,
                    "reward": step.reward,
                    "observation_before": step_observation,
                    "observation_after": step.observation,
                    "achievements": step.snapshot.achievements,
                }
            )
            observation = step.observation
            step_observation = step.observation
            emit(step.action)

        result = execute_skill(
            request,
            environment,
            state,
            allowed_actions=config.tools,
            on_step=record_step,
        )
        state = result.state
        for item in history[-len(result.actions) :]:
            item["skill_status"] = result.status
        decisions.append(
            {
                "decision_id": decision_id,
                "step_started": snapshot.step,
                "step_ended": environment.step_count,
                "state_before": state_before,
                "state_after": state.model_dump(),
                "state_patch": parsed.state_patch.model_dump(exclude_none=True),
                "skill": request.model_dump(),
                "status": result.status,
                "reason": result.reason,
                "actions": result.actions,
                "attempts": len(decision_attempts),
            }
        )
        if result.actions:
            empty_decisions = 0
        else:
            empty_decisions += 1
            if empty_decisions >= 3:
                termination_reason = "agent_error"
                terminal_error = "Three skill decisions made no environment progress"
                break
        if result.status == "terminal":
            termination_reason = (
                "death"
                if environment.snapshot().inventory.get("health", 0) <= 0
                else "step_limit"
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
        controller="skills",
        decisions=decisions,
        model_calls=len(attempts),
        cache_read_tokens=_sum_if_complete(attempts, "cache_read_tokens"),
        cache_write_tokens=_sum_if_complete(attempts, "cache_write_tokens"),
    )


def play_episode(
    genome: Genome,
    seed: int,
    max_steps: int = 200,
    *,
    provider: GameplayProvider | None = None,
    on_frame: Callable[[EpisodeFrame], None] | None = None,
    controller: ControllerMode = "primitive",
    max_skill_steps: int = 5,
) -> EpisodeTrace:
    """Play until death, step limit, or three failed attempts at one decision."""
    _validate_episode(genome, max_steps)
    provider = provider or BedrockProvider()
    if controller == "skills":
        if not 1 <= max_skill_steps <= 5:
            raise ValueError("max_skill_steps must be between 1 and 5")
        return _play_skill_episode(
            genome, seed, max_steps, provider, on_frame, max_skill_steps
        )
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
        controller="primitive",
        model_calls=len(attempts),
    )
