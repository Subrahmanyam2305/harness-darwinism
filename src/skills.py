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

MOVE_ACTIONS = ["move_up", "move_right", "move_down", "move_left"]
MOVE_OFFSETS = {
    "move_left": (-1, 0),
    "move_right": (1, 0),
    "move_up": (0, -1),
    "move_down": (0, 1),
}
ACQUIRE_REQUIREMENTS = {
    "tree": {},
    "stone": {"wood_pickaxe": 1},
    "coal": {"wood_pickaxe": 1},
    "iron": {"stone_pickaxe": 1},
    "diamond": {"iron_pickaxe": 1},
    "water": {},
    "grass": {},
    "cow": {},
    "plant": {},
}
ACQUIRE_RESULTS = {
    "tree": "wood",
    "stone": "stone",
    "coal": "coal",
    "iron": "iron",
    "diamond": "diamond",
    "water": "drink",
    "grass": "sapling",
    "cow": "food",
    "plant": "food",
}
ACQUIRE_ACHIEVEMENTS = {
    "tree": "collect_wood",
    "stone": "collect_stone",
    "coal": "collect_coal",
    "iron": "collect_iron",
    "diamond": "collect_diamond",
    "water": "collect_drink",
    "grass": "collect_sapling",
    "cow": "eat_cow",
    "plant": "eat_plant",
}
BUILD_ACTIONS = {
    "table": "place_table",
    "furnace": "place_furnace",
    "plant": "place_plant",
    "stone": "place_stone",
    "wood_pickaxe": "make_wood_pickaxe",
    "stone_pickaxe": "make_stone_pickaxe",
    "iron_pickaxe": "make_iron_pickaxe",
    "wood_sword": "make_wood_sword",
    "stone_sword": "make_stone_sword",
    "iron_sword": "make_iron_sword",
}
BUILD_REQUIREMENTS = {
    "table": {"wood": 2},
    "furnace": {"stone": 4},
    "plant": {"sapling": 1},
    "stone": {"stone": 1},
    "wood_pickaxe": {"wood": 1},
    "stone_pickaxe": {"wood": 1, "stone": 1},
    "iron_pickaxe": {"wood": 1, "coal": 1, "iron": 1},
    "wood_sword": {"wood": 1},
    "stone_sword": {"wood": 1, "stone": 1},
    "iron_sword": {"wood": 1, "coal": 1, "iron": 1},
}
BUILD_WORKSTATIONS = {
    "wood_pickaxe": {"table"},
    "stone_pickaxe": {"table"},
    "iron_pickaxe": {"table", "furnace"},
    "wood_sword": {"table"},
    "stone_sword": {"table"},
    "iron_sword": {"table", "furnace"},
}
PLACEMENT_TERRAIN = {
    "table": {"grass", "sand", "path"},
    "furnace": {"grass", "sand", "path"},
    "plant": {"grass"},
    "stone": {"grass", "sand", "path", "water", "lava"},
}


class SkillAffordance(BaseModel):
    """One locally computed feasibility decision shown to the model and operator."""

    option: str
    feasible: bool
    reason: str


class GuardCheck(BaseModel):
    """One deterministic safety or validity check made during local execution."""

    step: int
    check: str
    passed: bool
    detail: str


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
    guard_checks: list[GuardCheck] = Field(default_factory=list)
    state: ExecutionState


class SkillEnvironment(Protocol):
    step_count: int

    def snapshot(self) -> CrafterSnapshot: ...

    def step(self, action: str) -> tuple[str, float, bool, dict]: ...


def _entity(snapshot: CrafterSnapshot, kinds: set[str]) -> NearbyEntity | None:
    return next(
        (entity for entity in snapshot.nearby_entities if entity.kind in kinds), None
    )


def _has_items(inventory: dict[str, int], requirements: dict[str, int]) -> bool:
    return all(
        inventory.get(item, 0) >= amount for item, amount in requirements.items()
    )


def _has_nearby(snapshot: CrafterSnapshot, kinds: set[str]) -> bool:
    nearby = {
        entity.kind for entity in snapshot.nearby_entities if entity.distance <= 1
    }
    return kinds <= nearby


def _safe_moves(snapshot: CrafterSnapshot, allowed_actions: set[str]) -> list[str]:
    locally_walkable = snapshot.walkable_directions
    if locally_walkable is None:
        locally_walkable = MOVE_ACTIONS
    return [
        action
        for action in MOVE_ACTIONS
        if action in locally_walkable and action in allowed_actions
    ]


def _unvisited_moves(
    snapshot: CrafterSnapshot,
    moves: list[str],
    recent_positions: list[tuple[int, int]] | None,
) -> list[str]:
    recent = set((recent_positions or [])[-4:])
    return [
        action
        for action in moves
        if (
            snapshot.position[0] + MOVE_OFFSETS[action][0],
            snapshot.position[1] + MOVE_OFFSETS[action][1],
        )
        not in recent
    ]


def _move_toward(
    entity: NearbyEntity,
    snapshot: CrafterSnapshot,
    allowed_actions: set[str],
    recent_positions: list[tuple[int, int]] | None = None,
) -> str | None:
    preferred: list[str] = []
    if entity.dx < 0:
        preferred.append("move_left")
    elif entity.dx > 0:
        preferred.append("move_right")
    if entity.dy < 0:
        preferred.append("move_up")
    elif entity.dy > 0:
        preferred.append("move_down")
    safe = _safe_moves(snapshot, allowed_actions)
    if entity.distance == 1 and len(preferred) == 1:
        return preferred[0]
    unvisited = _unvisited_moves(snapshot, safe, recent_positions)
    for action in preferred:
        if action in unvisited:
            return action
    allowed_preferred = [action for action in preferred if action in allowed_actions]
    if preferred and not allowed_preferred:
        return preferred[0]
    if unvisited:
        return unvisited[0]
    if allowed_preferred and not unvisited:
        return allowed_preferred[0]
    return safe[0] if safe else (preferred[0] if preferred else None)


def _move_away(
    entity: NearbyEntity, snapshot: CrafterSnapshot, allowed_actions: set[str]
) -> str | None:
    if abs(entity.dx) >= abs(entity.dy):
        preferred = "move_right" if entity.dx < 0 else "move_left"
    else:
        preferred = "move_down" if entity.dy < 0 else "move_up"
    safe = _safe_moves(snapshot, allowed_actions)
    return preferred if preferred in safe else (safe[0] if safe else None)


def feasible_skills(snapshot: CrafterSnapshot, allowed_actions: list[str]) -> list[str]:
    """Return compact skill options grounded in the current state and tool list."""
    allowed = set(allowed_actions)
    options: list[str] = []
    if _next_action(SkillRequest(name="explore"), snapshot, allowed) in allowed:
        options.append("explore")
    if snapshot.front_kind == "grass" and "do" in allowed:
        options.append("acquire:grass")
    for entity in snapshot.nearby_entities:
        if (
            entity.kind in ACQUIRE_REQUIREMENTS
            and _has_items(snapshot.inventory, ACQUIRE_REQUIREMENTS[entity.kind])
            and "do" in allowed
            and _next_action(
                SkillRequest(name="acquire", target=entity.kind), snapshot, allowed
            )
            in allowed
        ):
            option = f"acquire:{entity.kind}"
            if option not in options:
                options.append(option)
        if entity.kind in {"zombie", "skeleton", "arrow", "lava"} and (
            _next_action(
                SkillRequest(name="evade", target=entity.kind), snapshot, allowed
            )
            in allowed
        ):
            option = f"evade:{entity.kind}"
            if option not in options:
                options.append(option)
        if entity.kind in {"zombie", "skeleton"} and (
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
    for goal, action in BUILD_ACTIONS.items():
        already_complete = goal not in PLACEMENT_TERRAIN and inventory.get(goal, 0) > 0
        requirements_met = _has_items(inventory, BUILD_REQUIREMENTS[goal])
        workstations_met = _has_nearby(snapshot, BUILD_WORKSTATIONS.get(goal, set()))
        terrain_met = (
            goal not in PLACEMENT_TERRAIN
            or snapshot.front_kind in PLACEMENT_TERRAIN[goal]
        )
        if (
            not already_complete
            and requirements_met
            and workstations_met
            and terrain_met
            and action in allowed
        ):
            options.append(f"build_or_craft:{goal}")
    for need in ("health", "food", "drink", "energy"):
        request = SkillRequest(name="recover", need=need)
        if (
            snapshot.inventory.get(need, 9) < 5
            and _next_action(request, snapshot, allowed) in allowed
        ):
            options.append(f"recover:{need}")
    if (
        "sleep" in allowed
        and snapshot.inventory.get("energy", 9) < 9
        and "recover:energy" not in options
    ):
        options.append("recover:energy")
    threat = _entity(snapshot, {"zombie", "skeleton", "arrow", "lava"})
    if snapshot.inventory.get("health", 9) <= 2:
        options = [
            option
            for option in options
            if option.split(":", 1)[0] in {"evade", "recover"}
        ]
    elif threat is not None and threat.distance <= 1:
        options = [
            option
            for option in options
            if option.split(":", 1)[0] in {"evade", "engage", "recover"}
        ]
    return options


def assess_skill_affordances(
    snapshot: CrafterSnapshot, allowed_actions: list[str]
) -> list[SkillAffordance]:
    """Explain the local SayCan-inspired gate without exposing model reasoning."""
    options = feasible_skills(snapshot, allowed_actions)
    evidence = [
        SkillAffordance(
            option=option, feasible=True, reason="supported_by_current_state"
        )
        for option in options
    ]
    blocked_reasons = {
        "explore": "no_allowed_movement_action",
        "acquire": "no_visible_reachable_resource",
        "evade": "no_visible_threat",
        "engage": "no_visible_engageable_threat",
        "build_or_craft": "requirements_or_action_unavailable",
        "recover": "recovery_not_currently_needed_or_available",
    }
    offered_families = {option.split(":", 1)[0] for option in options}
    evidence.extend(
        SkillAffordance(option=name, feasible=False, reason=reason)
        for name, reason in blocked_reasons.items()
        if name not in offered_families
    )
    return evidence


def _next_action(
    request: SkillRequest,
    snapshot: CrafterSnapshot,
    allowed_actions: set[str],
    recent_positions: list[tuple[int, int]] | None = None,
) -> str | None:
    if request.name == "explore":
        safe_moves = _safe_moves(snapshot, allowed_actions)
        if not safe_moves:
            return None
        unvisited = _unvisited_moves(snapshot, safe_moves, recent_positions)
        choices = unvisited or safe_moves
        return choices[snapshot.step % len(choices)]
    if request.name in {"acquire", "engage"}:
        if request.name == "acquire" and request.target == "grass":
            return "do" if snapshot.front_kind == "grass" else None
        target = _entity(snapshot, {request.target or ""})
        if target is None:
            return None
        if target.distance == 1 and snapshot.facing == target.direction:
            return "do"
        return _move_toward(target, snapshot, allowed_actions, recent_positions)
    if request.name == "evade":
        threat = _entity(
            snapshot,
            {request.target or "zombie", "zombie", "skeleton", "arrow", "lava"},
        )
        return _move_away(threat, snapshot, allowed_actions) if threat else None
    if request.name == "build_or_craft":
        return BUILD_ACTIONS.get(request.goal or "")
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
            safe_moves = _safe_moves(snapshot, allowed_actions)
            if not safe_moves:
                return None
            unvisited = _unvisited_moves(snapshot, safe_moves, recent_positions)
            choices = unvisited or safe_moves
            return choices[snapshot.step % len(choices)]
        if target.distance == 1 and snapshot.facing == target.direction:
            return "do"
        return _move_toward(target, snapshot, allowed_actions, recent_positions)
    return None


def _achievement_advanced(
    before: CrafterSnapshot, after: CrafterSnapshot, achievement: str
) -> bool:
    if before.achievement_counts or after.achievement_counts:
        return after.achievement_counts.get(
            achievement, 0
        ) > before.achievement_counts.get(achievement, 0)
    return not before.achievements.get(achievement, False) and after.achievements.get(
        achievement, False
    )


def _target_distance(snapshot: CrafterSnapshot, target: str | None) -> int | None:
    entity = _entity(snapshot, {target or ""})
    return entity.distance if entity is not None else None


def _acquire_succeeded(
    target: str | None, before: CrafterSnapshot, after: CrafterSnapshot
) -> bool:
    if target not in ACQUIRE_RESULTS:
        return False
    item = ACQUIRE_RESULTS[target]
    achievement = ACQUIRE_ACHIEVEMENTS[target]
    return after.inventory.get(item, 0) > before.inventory.get(
        item, 0
    ) or _achievement_advanced(before, after, achievement)


def _build_succeeded(
    goal: str | None, before: CrafterSnapshot, after: CrafterSnapshot
) -> bool:
    if goal not in BUILD_ACTIONS:
        return False
    achievement = f"place_{goal}" if goal in PLACEMENT_TERRAIN else f"make_{goal}"
    if _achievement_advanced(before, after, achievement):
        return True
    return goal not in PLACEMENT_TERRAIN and after.inventory.get(
        goal, 0
    ) > before.inventory.get(goal, 0)


def _made_progress(
    request: SkillRequest, before: CrafterSnapshot, after: CrafterSnapshot
) -> bool:
    if request.name == "acquire" and _acquire_succeeded(request.target, before, after):
        return True
    if request.name == "build_or_craft" and _build_succeeded(
        request.goal, before, after
    ):
        return True
    if request.name == "engage":
        achievement = f"defeat_{request.target}"
        if _achievement_advanced(before, after, achievement):
            return True
    if request.name == "recover" and after.inventory.get(
        request.need or "", 0
    ) > before.inventory.get(request.need or "", 0):
        return True
    if request.name == "evade":
        old_distance = _target_distance(before, request.target)
        new_distance = _target_distance(after, request.target)
        return new_distance is None or (
            old_distance is not None and new_distance > old_distance
        )
    return before.position != after.position


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
    actions: list[str] = []
    steps: list[SkillStep] = []
    guard_checks: list[GuardCheck] = []
    runtime_state = state.model_copy(deep=True)
    if not runtime_state.recent_positions:
        runtime_state.recent_positions.append(before.position)
    reason = "local_step_budget_exhausted"
    status: SkillStatus = "exhausted"

    for _ in range(request.max_steps):
        snapshot = environment.snapshot()
        health_safe = snapshot.inventory.get("health", 9) > 2 or request.name in {
            "evade",
            "recover",
        }
        guard_checks.append(
            GuardCheck(
                step=snapshot.step,
                check="critical_health",
                passed=health_safe,
                detail=f"health={snapshot.inventory.get('health', 9)}",
            )
        )
        if not health_safe:
            status, reason = "interrupted", "critical_health"
            break
        threat = _entity(snapshot, {"zombie", "skeleton", "arrow", "lava"})
        threat_safe = not (
            threat is not None
            and threat.distance <= 1
            and request.name not in {"evade", "engage", "recover"}
        )
        guard_checks.append(
            GuardCheck(
                step=snapshot.step,
                check="nearby_threat",
                passed=threat_safe,
                detail=(
                    f"{threat.kind}@{threat.distance}"
                    if threat is not None
                    else "none_visible"
                ),
            )
        )
        if not threat_safe:
            status, reason = "interrupted", "nearby_threat"
            break
        action = _next_action(
            request,
            snapshot,
            set(allowed_actions),
            runtime_state.recent_positions,
        )
        guard_checks.append(
            GuardCheck(
                step=snapshot.step,
                check="action_resolved",
                passed=action is not None,
                detail=action or "no_local_action",
            )
        )
        if action is None:
            status, reason = "interrupted", "target_or_action_unavailable"
            break
        action_allowed = action in allowed_actions
        guard_checks.append(
            GuardCheck(
                step=snapshot.step,
                check="action_allowlist",
                passed=action_allowed,
                detail=action,
            )
        )
        if not action_allowed:
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
        if after.position != snapshot.position:
            runtime_state.recent_positions = (
                runtime_state.recent_positions + [after.position]
            )[-8:]
        runtime_state.no_progress_steps = (
            0
            if _made_progress(request, snapshot, after)
            else min(20, runtime_state.no_progress_steps + 1)
        )
        if done:
            status, reason = "terminal", "environment_terminal"
            break
        if request.name == "acquire" and _acquire_succeeded(
            request.target, before, after
        ):
            status, reason = "succeeded", "target_acquired"
            break
        if request.name == "engage" and _achievement_advanced(
            before, after, f"defeat_{request.target}"
        ):
            status, reason = "succeeded", "target_defeated"
            break
        if request.name == "build_or_craft" and _build_succeeded(
            request.goal, before, after
        ):
            status, reason = "succeeded", "requested_build_completed"
            break
        if request.name == "recover" and after.inventory.get(
            request.need or "", 0
        ) > initial_inventory.get(request.need or "", 0):
            status, reason = "succeeded", "vital_improved"
            break
        if request.name == "evade":
            threat = _entity(after, {"zombie", "skeleton", "arrow", "lava"})
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
        status=status,
        reason=reason,
        actions=actions,
        steps=steps,
        guard_checks=guard_checks,
        state=runtime_state,
    )
