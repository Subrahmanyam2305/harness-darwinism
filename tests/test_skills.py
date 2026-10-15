"""Behavioral tests for compact state and deterministic Crafter skills."""

from dataclasses import dataclass

import pytest
from pydantic import ValidationError

from src.crafter_env import ALL_ACHIEVEMENTS, CrafterSnapshot, NearbyEntity
from src.execution_state import ExecutionState, KnownLocation, StatePatch
from src.skills import (
    ACQUIRE_ACHIEVEMENTS,
    BUILD_ACTIONS,
    PLACEMENT_TERRAIN,
    SkillRequest,
    assess_skill_affordances,
    execute_skill,
    feasible_skills,
)


def test_skill_catalog_covers_every_upstream_achievement() -> None:
    build_achievements = {
        f"place_{goal}" if goal in PLACEMENT_TERRAIN else f"make_{goal}"
        for goal in BUILD_ACTIONS
    }
    covered = (
        set(ACQUIRE_ACHIEVEMENTS.values())
        | build_achievements
        | {
            "defeat_zombie",
            "defeat_skeleton",
            "wake_up",
        }
    )

    assert covered == set(ALL_ACHIEVEMENTS)


def _snapshot(
    *,
    step: int = 0,
    wood: int = 0,
    health: int = 9,
    inventory: dict[str, int] | None = None,
    entities: list[NearbyEntity] | None = None,
    facing: str = "east",
    front_kind: str | None = "grass",
    walkable_directions: list[str] | None = None,
    position: tuple[int, int] = (0, 0),
    achievements: dict[str, bool] | None = None,
    achievement_counts: dict[str, int] | None = None,
) -> CrafterSnapshot:
    items = {
        "health": health,
        "food": 9,
        "drink": 9,
        "energy": 9,
        "wood": wood,
    }
    items.update(inventory or {})
    return CrafterSnapshot(
        step=step,
        inventory=items,
        achievements=achievements or {},
        achievement_counts=achievement_counts or {},
        nearby_entities=entities or [],
        facing=facing,
        front_kind=front_kind,
        walkable_directions=walkable_directions
        or [
            "move_up",
            "move_right",
            "move_down",
            "move_left",
        ],
        position=position,
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
    assert "recover:energy" not in options
    assert not any(option.startswith("build_or_craft") for option in options)


def test_affordances_reject_target_when_required_direction_is_disallowed() -> None:
    snapshot = _snapshot(
        entities=[NearbyEntity(kind="tree", direction="east", distance=2, dx=2, dy=0)]
    )

    options = feasible_skills(snapshot, allowed_actions=["move_left", "do"])

    assert "acquire:tree" not in options


def test_affordances_cover_missing_achievement_actions_when_mechanically_valid() -> (
    None
):
    snapshot = _snapshot(
        inventory={"wood": 3, "stone": 2, "sapling": 1},
        entities=[
            NearbyEntity(kind="table", direction="west", distance=1, dx=-1, dy=0),
        ],
    )

    options = feasible_skills(
        snapshot,
        allowed_actions=[
            "move_up",
            "do",
            "make_wood_sword",
            "make_stone_sword",
            "place_plant",
            "place_stone",
        ],
    )

    assert "acquire:grass" in options
    assert "build_or_craft:wood_sword" in options
    assert "build_or_craft:stone_sword" in options
    assert "build_or_craft:plant" in options
    assert "build_or_craft:stone" in options


def test_affordances_require_tools_workstations_and_valid_placement_terrain() -> None:
    snapshot = _snapshot(
        inventory={"wood": 3, "stone": 4, "iron": 1, "coal": 1, "sapling": 1},
        entities=[
            NearbyEntity(kind="iron", direction="east", distance=2, dx=2, dy=0),
            NearbyEntity(kind="table", direction="west", distance=2, dx=-2, dy=0),
        ],
        front_kind="water",
    )

    options = feasible_skills(
        snapshot,
        allowed_actions=list(
            {
                "move_right",
                "do",
                "make_iron_pickaxe",
                "make_iron_sword",
                "place_table",
                "place_furnace",
                "place_plant",
                "place_stone",
            }
        ),
    )

    assert "acquire:iron" not in options
    assert "build_or_craft:iron_pickaxe" not in options
    assert "build_or_craft:iron_sword" not in options
    assert "build_or_craft:table" not in options
    assert "build_or_craft:furnace" not in options
    assert "build_or_craft:plant" not in options
    assert "build_or_craft:stone" in options


def test_affordances_do_not_offer_a_tool_already_owned() -> None:
    snapshot = _snapshot(
        inventory={"wood": 2, "wood_sword": 1},
        entities=[
            NearbyEntity(kind="table", direction="west", distance=1, dx=-1, dy=0)
        ],
    )

    options = feasible_skills(snapshot, ["make_wood_sword"])

    assert "build_or_craft:wood_sword" not in options


def test_affordances_allow_rebuilding_completed_workstation_achievements() -> None:
    snapshot = _snapshot(
        inventory={"wood": 2, "stone": 4},
        achievements={"place_table": True, "place_furnace": True},
    )

    options = feasible_skills(snapshot, ["place_table", "place_furnace"])

    assert "build_or_craft:table" in options
    assert "build_or_craft:furnace" in options


def test_affordances_match_runtime_safety_guards() -> None:
    snapshot = _snapshot(
        health=2,
        inventory={"energy": 4},
        entities=[
            NearbyEntity(kind="tree", direction="east", distance=1, dx=1, dy=0),
            NearbyEntity(kind="zombie", direction="west", distance=1, dx=-1, dy=0),
        ],
    )

    options = feasible_skills(
        snapshot,
        ["move_left", "move_right", "do", "sleep"],
    )

    assert options
    assert all(option.startswith(("evade:", "recover:")) for option in options)
    assert "explore" not in options
    assert "acquire:tree" not in options


def test_affordance_evidence_explains_feasible_and_blocked_skill_families() -> None:
    snapshot = _snapshot(
        entities=[NearbyEntity(kind="tree", direction="east", distance=2, dx=2, dy=0)]
    )

    evidence = assess_skill_affordances(snapshot, ["move_right", "do"])

    assert any(item.option == "acquire:tree" and item.feasible for item in evidence)
    assert any(
        item.option == "engage"
        and not item.feasible
        and item.reason == "no_visible_engageable_threat"
        for item in evidence
    )


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
            position=(self.step_count, 0),
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
            position=(self.step_count, 0),
            entities=[
                NearbyEntity(
                    kind="tree", direction="east", distance=distance, dx=distance, dy=0
                )
            ],
        )


@dataclass
class _WrongInventoryChangeEnv(_FakeSkillEnv):
    food: int = 9

    def snapshot(self) -> CrafterSnapshot:
        return _snapshot(
            step=self.step_count,
            wood=self.wood,
            inventory={"food": self.food},
            entities=[
                NearbyEntity(kind="tree", direction="east", distance=1, dx=1, dy=0)
            ],
        )

    def step(self, action: str):
        self.step_count += 1
        self.food -= 1
        return "vital changed", 0.0, False, {}


@dataclass
class _BlockedDirectPathEnv(_FakeSkillEnv):
    x: int = 0
    y: int = 0

    def snapshot(self) -> CrafterSnapshot:
        return _snapshot(
            step=self.step_count,
            wood=self.wood,
            position=(self.x, self.y),
            walkable_directions=["move_up", "move_down", "move_left"],
            entities=[
                NearbyEntity(
                    kind="tree",
                    direction="south-east" if self.y < 0 else "east",
                    distance=max(abs(2 - self.x), abs(-self.y)),
                    dx=2 - self.x,
                    dy=-self.y,
                )
            ],
        )

    def step(self, action: str):
        self.step_count += 1
        dx, dy = {
            "move_left": (-1, 0),
            "move_up": (0, -1),
            "move_down": (0, 1),
        }[action]
        self.x += dx
        self.y += dy
        return "detour", 0.0, False, {}


@dataclass
class _RepeatedCombatEnv(_FakeSkillEnv):
    defeated: bool = False

    def snapshot(self) -> CrafterSnapshot:
        return _snapshot(
            step=self.step_count,
            achievements={"defeat_zombie": True},
            achievement_counts={"defeat_zombie": 2 if self.defeated else 1},
            entities=(
                []
                if self.defeated
                else [
                    NearbyEntity(
                        kind="zombie", direction="east", distance=1, dx=1, dy=0
                    )
                ]
            ),
        )

    def step(self, action: str):
        self.step_count += 1
        self.defeated = action == "do"
        return "combat", 0.0, False, {}


@dataclass
class _OpenExploreEnv(_FakeSkillEnv):
    x: int = 0
    y: int = 0

    def snapshot(self) -> CrafterSnapshot:
        return _snapshot(
            step=self.step_count,
            position=(self.x, self.y),
        )

    def step(self, action: str):
        self.step_count += 1
        dx, dy = {
            "move_left": (-1, 0),
            "move_right": (1, 0),
            "move_up": (0, -1),
            "move_down": (0, 1),
        }[action]
        self.x += dx
        self.y += dy
        return "moved", 0.0, False, {}


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
    assert any(check.check == "action_allowlist" for check in result.guard_checks)
    assert observed_actions == result.actions
    assert environment.wood == 1


def test_acquire_does_not_succeed_for_an_unrelated_inventory_change() -> None:
    environment = _WrongInventoryChangeEnv()

    result = execute_skill(
        SkillRequest(name="acquire", target="tree", max_steps=2),
        environment,
        ExecutionState(),
        allowed_actions=["do"],
    )

    assert result.status != "succeeded"
    assert result.reason == "local_step_budget_exhausted"
    assert result.actions == ["do", "do"]


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


def test_acquire_takes_a_walkable_detour_when_direct_path_is_blocked() -> None:
    environment = _BlockedDirectPathEnv()

    result = execute_skill(
        SkillRequest(name="acquire", target="tree", max_steps=3),
        environment,
        ExecutionState(),
        allowed_actions=["move_right", "move_up", "do"],
    )

    assert result.actions == ["move_up", "move_up", "move_up"]
    assert len({step.snapshot.position for step in result.steps}) == 3


def test_repeated_combat_uses_achievement_count_for_success() -> None:
    environment = _RepeatedCombatEnv()

    result = execute_skill(
        SkillRequest(name="engage", target="zombie", max_steps=2),
        environment,
        ExecutionState(),
        allowed_actions=["do"],
    )

    assert result.status == "succeeded"
    assert result.reason == "target_defeated"
    assert result.actions == ["do"]


def test_explore_avoids_returning_to_a_recent_position_when_an_option_exists() -> None:
    environment = _OpenExploreEnv()

    result = execute_skill(
        SkillRequest(name="explore", max_steps=5),
        environment,
        ExecutionState(),
        allowed_actions=["move_up", "move_right", "move_down", "move_left"],
    )

    positions = [step.snapshot.position for step in result.steps]
    assert len(positions) == len(set(positions))


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
