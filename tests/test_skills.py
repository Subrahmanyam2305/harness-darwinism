"""Behavioral tests for compact state and deterministic Crafter skills."""

from dataclasses import dataclass

import pytest
from pydantic import ValidationError

from src.crafter_env import CrafterSnapshot, NearbyEntity
from src.execution_state import ExecutionState, KnownLocation, StatePatch
from src.skills import SkillRequest, execute_skill, feasible_skills


def _snapshot(
    *,
    step: int = 0,
    wood: int = 0,
    health: int = 9,
    entities: list[NearbyEntity] | None = None,
    facing: str = "east",
) -> CrafterSnapshot:
    return CrafterSnapshot(
        step=step,
        inventory={"health": health, "food": 9, "drink": 9, "energy": 9, "wood": wood},
        achievements={},
        nearby_entities=entities or [],
        facing=facing,
        daylight=1.0,
        sleeping=False,
    )


def test_execution_state_starts_empty_for_each_episode() -> None:
    first = ExecutionState(goal="collect wood")
    second = ExecutionState()

    assert first.goal == "collect wood"
    assert second.goal is None
    assert second.known_locations == {}
    assert second.failed_skills == {}
    assert second.no_progress_steps == 0


def test_affordances_only_offer_skills_supported_by_current_state() -> None:
    snapshot = _snapshot(
        entities=[NearbyEntity(kind="tree", direction="east", distance=2, dx=2, dy=0)]
    )

    options = feasible_skills(snapshot, allowed_actions=["move_right", "do", "sleep"])

    assert "explore" in options
    assert "acquire:tree" in options
    assert "recover:energy" in options
    assert not any(option.startswith("build_or_craft") for option in options)


def test_affordances_reject_target_when_required_direction_is_disallowed() -> None:
    snapshot = _snapshot(
        entities=[NearbyEntity(kind="tree", direction="east", distance=2, dx=2, dy=0)]
    )

    options = feasible_skills(snapshot, allowed_actions=["move_left", "do"])

    assert "acquire:tree" not in options


def test_execution_state_bounds_model_owned_locations() -> None:
    locations = {
        f"tree-{index}": KnownLocation(
            direction="east", estimated_distance=2, last_seen_step=1
        )
        for index in range(21)
    }

    with pytest.raises(ValidationError):
        StatePatch(known_locations=locations)


@dataclass
class _FakeSkillEnv:
    wood: int = 0
    step_count: int = 0

    def snapshot(self) -> CrafterSnapshot:
        distance = max(1, 2 - self.step_count)
        return _snapshot(
            step=self.step_count,
            wood=self.wood,
            entities=[
                NearbyEntity(
                    kind="tree", direction="east", distance=distance, dx=distance, dy=0
                )
            ],
        )

    def step(self, action: str):
        self.step_count += 1
        if action == "do":
            self.wood += 1
        after = self.snapshot()
        return (
            f"step {self.step_count}",
            float(action == "do"),
            False,
            {
                "inventory": after.inventory,
                "achievements": {"collect_wood": self.wood > 0},
            },
        )


@dataclass
class _DistantTreeEnv(_FakeSkillEnv):
    def snapshot(self) -> CrafterSnapshot:
        distance = max(1, 4 - self.step_count)
        return _snapshot(
            step=self.step_count,
            wood=self.wood,
            entities=[
                NearbyEntity(
                    kind="tree", direction="east", distance=distance, dx=distance, dy=0
                )
            ],
        )


def test_acquire_skill_executes_multiple_local_steps_until_inventory_increases() -> (
    None
):
    environment = _FakeSkillEnv()
    observed_actions: list[str] = []

    result = execute_skill(
        SkillRequest(name="acquire", target="tree", max_steps=4),
        environment,
        ExecutionState(goal="collect wood"),
        allowed_actions=["move_right", "do"],
        on_step=lambda step: observed_actions.append(step.action),
    )

    assert result.status == "succeeded"
    assert result.actions == ["move_right", "do"]
    assert observed_actions == result.actions
    assert environment.wood == 1


def test_moving_closer_to_target_counts_as_progress() -> None:
    environment = _DistantTreeEnv()

    result = execute_skill(
        SkillRequest(name="acquire", target="tree", max_steps=5),
        environment,
        ExecutionState(),
        allowed_actions=["move_right", "do"],
    )

    assert result.status == "succeeded"
    assert result.actions == ["move_right", "move_right", "move_right", "do"]


def test_skill_stops_before_disallowed_local_action() -> None:
    environment = _FakeSkillEnv()

    result = execute_skill(
        SkillRequest(name="acquire", target="tree", max_steps=4),
        environment,
        ExecutionState(),
        allowed_actions=["do"],
    )

    assert result.status == "interrupted"
    assert result.reason == "action_not_allowed"
    assert result.actions == []
    assert environment.step_count == 0


def test_non_defensive_skill_stops_before_acting_near_threat() -> None:
    environment = _FakeSkillEnv()
    environment.snapshot = lambda: _snapshot(
        step=environment.step_count,
        entities=[
            NearbyEntity(kind="zombie", direction="east", distance=1, dx=1, dy=0)
        ],
    )

    result = execute_skill(
        SkillRequest(name="explore", max_steps=3),
        environment,
        ExecutionState(),
        allowed_actions=["move_right"],
    )

    assert result.status == "interrupted"
    assert result.reason == "nearby_threat"
    assert result.actions == []
