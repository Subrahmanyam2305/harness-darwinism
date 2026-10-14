"""Deterministic, bounded Crafter skills selected by the gameplay model."""

from __future__ import annotations

from collections.abc import Callable
from typing import Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field

from src.crafter_env import CrafterSnapshot, NearbyEntity
from src.execution_state import ExecutionState, SkillFailure

SkillName = Literal[
    "explore", "acquire", "evade", "engage", "build_or_craft", "recover"
]
SkillStatus = Literal["succeeded", "failed", "interrupted", "exhausted", "terminal"]


class SkillRequest(BaseModel):
    """One validated, parameterized local behavior requested by the model."""

    model_config = ConfigDict(extra="forbid", strict=True)

    name: SkillName
    target: str | None = Field(default=None, max_length=40)
    goal: str | None = Field(default=None, max_length=40)
    need: Literal["health", "food", "drink", "energy"] | None = None
    max_steps: int = Field(default=3, ge=1, le=5)


class SkillStep(BaseModel):
    """One real environment action executed inside a model decision."""

    action: str
    reward: float
    observation: str
    done: bool
    snapshot: CrafterSnapshot


class SkillResult(BaseModel):
    """Auditable outcome of one bounded skill invocation."""

    status: SkillStatus
    reason: str
    actions: list[str] = Field(default_factory=list)
    steps: list[SkillStep] = Field(default_factory=list)
    state: ExecutionState


class SkillEnvironment(Protocol):
    step_count: int

    def snapshot(self) -> CrafterSnapshot: ...

    def step(self, action: str) -> tuple[str, float, bool, dict]: ...


def _entity(snapshot: CrafterSnapshot, kinds: set[str]) -> NearbyEntity | None:
    return next(
        (entity for entity in snapshot.nearby_entities if entity.kind in kinds), None
    )


def _move_toward(entity: NearbyEntity) -> str:
    if entity.dx < 0:
        return "move_left"
    if entity.dx > 0:
        return "move_right"
    if entity.dy < 0:
        return "move_up"
    return "move_down"


def _move_away(entity: NearbyEntity) -> str:
    if abs(entity.dx) >= abs(entity.dy):
        return "move_right" if entity.dx < 0 else "move_left"
    return "move_down" if entity.dy < 0 else "move_up"


def feasible_skills(snapshot: CrafterSnapshot, allowed_actions: list[str]) -> list[str]:
    """Return compact skill options grounded in the current state and tool list."""
    allowed = set(allowed_actions)
    options: list[str] = []
    if _next_action(SkillRequest(name="explore"), snapshot, allowed) in allowed:
        options.append("explore")
    for entity in snapshot.nearby_entities:
        if (
            entity.kind
            in {"tree", "stone", "coal", "iron", "diamond", "cow", "plant", "water"}
            and "do" in allowed
            and _next_action(
                SkillRequest(name="acquire", target=entity.kind), snapshot, allowed
            )
            in allowed
        ):
            option = f"acquire:{entity.kind}"
            if option not in options:
                options.append(option)
        if entity.kind in {"zombie", "skeleton", "arrow"}:
            if (
                _next_action(
                    SkillRequest(name="evade", target=entity.kind), snapshot, allowed
                )
                in allowed
            ):
                option = f"evade:{entity.kind}"
                if option not in options:
                    options.append(option)
            if (
                "do" in allowed
                and _next_action(
                    SkillRequest(name="engage", target=entity.kind), snapshot, allowed
                )
                in allowed
            ):
                option = f"engage:{entity.kind}"
                if option not in options:
                    options.append(option)
    inventory = snapshot.inventory
    build_actions = {
        "table": ("place_table", inventory.get("wood", 0) >= 2),
        "furnace": ("place_furnace", inventory.get("stone", 0) >= 4),
        "wood_pickaxe": ("make_wood_pickaxe", inventory.get("wood", 0) >= 1),
        "stone_pickaxe": (
            "make_stone_pickaxe",
            inventory.get("wood", 0) >= 1 and inventory.get("stone", 0) >= 1,
        ),
        "iron_pickaxe": (
            "make_iron_pickaxe",
            inventory.get("wood", 0) >= 1
            and inventory.get("coal", 0) >= 1
            and inventory.get("iron", 0) >= 1,
        ),
    }
    for goal, (action, ready) in build_actions.items():
        if ready and action in allowed:
            options.append(f"build_or_craft:{goal}")
    for need in ("health", "food", "drink", "energy"):
        request = SkillRequest(name="recover", need=need)
        if (
            snapshot.inventory.get(need, 9) < 5
            and _next_action(request, snapshot, allowed) in allowed
        ):
            options.append(f"recover:{need}")
    if "sleep" in allowed and "recover:energy" not in options:
        options.append("recover:energy")
    return options


def _next_action(
    request: SkillRequest, snapshot: CrafterSnapshot, allowed_actions: set[str]
) -> str | None:
    moves = [
        action
        for action in ["move_up", "move_right", "move_down", "move_left"]
        if action in allowed_actions
    ]
    if request.name == "explore":
        return moves[snapshot.step % len(moves)] if moves else None
    if request.name in {"acquire", "engage"}:
        target = _entity(snapshot, {request.target or ""})
        if target is None:
            return None
        if target.distance == 1 and snapshot.facing == target.direction:
            return "do"
        return _move_toward(target)
    if request.name == "evade":
        threat = _entity(
            snapshot, {request.target or "zombie", "zombie", "skeleton", "arrow"}
        )
        return _move_away(threat) if threat else None
    if request.name == "build_or_craft":
        action = {
            "table": "place_table",
            "furnace": "place_furnace",
            "wood_pickaxe": "make_wood_pickaxe",
            "stone_pickaxe": "make_stone_pickaxe",
            "iron_pickaxe": "make_iron_pickaxe",
            "wood_sword": "make_wood_sword",
            "stone_sword": "make_stone_sword",
            "iron_sword": "make_iron_sword",
        }.get(request.goal or "")
        return action
    if request.name == "recover":
        if request.need == "energy":
            return "sleep"
        target_kinds = {
            "drink": {"water"},
            "food": {"cow", "plant"},
            "health": {"cow", "plant"},
        }
        target = _entity(snapshot, target_kinds.get(request.need or "", set()))
        if target is None:
            return moves[snapshot.step % len(moves)] if moves else None
        if target.distance == 1 and snapshot.facing == target.direction:
            return "do"
        return _move_toward(target)
    return None


def _made_progress(
    before: CrafterSnapshot, after: CrafterSnapshot, reward: float
) -> bool:
    return (
        reward > 0
        or before.inventory != after.inventory
        or before.achievements != after.achievements
        or before.nearby_entities != after.nearby_entities
        or before.facing != after.facing
    )


def execute_skill(
    request: SkillRequest,
    environment: SkillEnvironment,
    state: ExecutionState,
    *,
    allowed_actions: list[str],
    on_step: Callable[[SkillStep], None] | None = None,
) -> SkillResult:
    """Execute real actions until the requested skill succeeds or a guard stops it."""
    before = environment.snapshot()
    initial_inventory = dict(before.inventory)
    initial_achievements = dict(before.achievements)
    actions: list[str] = []
    steps: list[SkillStep] = []
    runtime_state = state.model_copy(deep=True)
    reason = "local_step_budget_exhausted"
    status: SkillStatus = "exhausted"

    for _ in range(request.max_steps):
        snapshot = environment.snapshot()
        if snapshot.inventory.get("health", 9) <= 2 and request.name not in {
            "evade",
            "recover",
        }:
            status, reason = "interrupted", "critical_health"
            break
        threat = _entity(snapshot, {"zombie", "skeleton", "arrow", "lava"})
        if (
            threat is not None
            and threat.distance <= 1
            and request.name not in {"evade", "engage", "recover"}
        ):
            status, reason = "interrupted", "nearby_threat"
            break
        action = _next_action(request, snapshot, set(allowed_actions))
        if action is None:
            status, reason = "interrupted", "target_or_action_unavailable"
            break
        if action not in allowed_actions:
            status, reason = "interrupted", "action_not_allowed"
            break
        observation, reward, done, _info = environment.step(action)
        after = environment.snapshot()
        skill_step = SkillStep(
            action=action,
            reward=float(reward),
            observation=observation,
            done=done,
            snapshot=after,
        )
        actions.append(action)
        steps.append(skill_step)
        if on_step:
            on_step(skill_step)
        runtime_state.no_progress_steps = (
            0
            if _made_progress(snapshot, after, float(reward))
            else min(20, runtime_state.no_progress_steps + 1)
        )
        if done:
            status, reason = "terminal", "environment_terminal"
            break
        if request.name == "acquire" and after.inventory != initial_inventory:
            status, reason = "succeeded", "inventory_changed"
            break
        if request.name == "engage" and (
            float(reward) > 0 or after.achievements != initial_achievements
        ):
            status, reason = "succeeded", "combat_progress"
            break
        if (
            request.name == "build_or_craft"
            and after.achievements != initial_achievements
        ):
            status, reason = "succeeded", "achievement_unlocked"
            break
        if request.name == "recover" and after.inventory.get(
            request.need or "", 0
        ) > initial_inventory.get(request.need or "", 0):
            status, reason = "succeeded", "vital_improved"
            break
        if request.name == "evade":
            threat = _entity(after, {"zombie", "skeleton", "arrow"})
            if threat is None or threat.distance >= 3:
                status, reason = "succeeded", "threat_distance_increased"
                break
        if runtime_state.no_progress_steps >= 3:
            status, reason = "interrupted", "no_progress"
            break
    if status not in {"succeeded", "terminal"}:
        key = ":".join(
            filter(None, [request.name, request.target, request.goal, request.need])
        )
        previous = runtime_state.failed_skills.get(key)
        if previous is None and len(runtime_state.failed_skills) >= 20:
            runtime_state.failed_skills.pop(next(iter(runtime_state.failed_skills)))
        runtime_state.failed_skills[key] = SkillFailure(
            count=min(20, (previous.count if previous else 0) + 1),
            last_failure=reason,
            last_attempt_step=environment.step_count,
        )
    return SkillResult(
        status=status, reason=reason, actions=actions, steps=steps, state=runtime_state
    )
