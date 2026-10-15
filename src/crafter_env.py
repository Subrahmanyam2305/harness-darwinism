"""Crafter environment wrapper that produces text observations for LLM agents."""

from __future__ import annotations

import crafter
import numpy as np
from pydantic import BaseModel, Field

ACTIONS = [
    "noop",
    "move_left",
    "move_right",
    "move_up",
    "move_down",
    "do",
    "sleep",
    "place_stone",
    "place_table",
    "place_furnace",
    "place_plant",
    "make_wood_pickaxe",
    "make_stone_pickaxe",
    "make_iron_pickaxe",
    "make_wood_sword",
    "make_stone_sword",
    "make_iron_sword",
]

ACTION_TO_IDX = {a: i for i, a in enumerate(ACTIONS)}

ALL_ACHIEVEMENTS = [
    "collect_coal",
    "collect_diamond",
    "collect_drink",
    "collect_iron",
    "collect_sapling",
    "collect_stone",
    "collect_wood",
    "defeat_skeleton",
    "defeat_zombie",
    "eat_cow",
    "eat_plant",
    "make_iron_pickaxe",
    "make_iron_sword",
    "make_stone_pickaxe",
    "make_stone_sword",
    "make_wood_pickaxe",
    "make_wood_sword",
    "place_furnace",
    "place_plant",
    "place_stone",
    "place_table",
    "wake_up",
]

MATERIAL_NAMES = {
    0: None,
    1: "water",
    2: "grass",
    3: "stone",
    4: "path",
    5: "sand",
    6: "tree",
    7: "lava",
    8: "coal",
    9: "iron",
    10: "diamond",
    11: "table",
    12: "furnace",
}

OBJECT_NAMES = {
    13: "player",
    14: "cow",
    15: "zombie",
    16: "skeleton",
    17: "arrow",
    18: "plant",
}

SEMANTIC_NAMES = {**MATERIAL_NAMES, **OBJECT_NAMES}

SEMANTIC_SYMBOLS = {
    0: "?",
    1: "~",
    2: ".",
    3: "#",
    4: ":",
    5: "s",
    6: "T",
    7: "L",
    8: "c",
    9: "i",
    10: "d",
    11: "B",
    12: "F",
    13: "@",
    14: "C",
    15: "Z",
    16: "K",
    17: ">",
    18: "P",
}


class NearbyEntity(BaseModel):
    """One notable object relative to the player in the current semantic view."""

    kind: str
    direction: str
    distance: int = Field(ge=0)
    dx: int
    dy: int


class CrafterSnapshot(BaseModel):
    """Structured current environment state used by deterministic skill guards."""

    step: int = Field(ge=0)
    inventory: dict[str, int]
    achievements: dict[str, bool]
    nearby_entities: list[NearbyEntity]
    facing: str
    position: tuple[int, int] = (0, 0)
    walkable_directions: list[str] | None = None
    front_kind: str | None = None
    daylight: float
    sleeping: bool


def _nearby_entities(info: dict) -> list[NearbyEntity]:
    semantic = info.get("semantic")
    player_pos = info.get("player_pos")
    if semantic is None or player_pos is None:
        return []
    px, py = int(player_pos[0]), int(player_pos[1])
    entities: list[NearbyEntity] = []
    for dy in range(-3, 4):
        for dx in range(-4, 5):
            if dx == 0 and dy == 0:
                continue
            x, y = px + dx, py + dy
            if not (0 <= x < semantic.shape[0] and 0 <= y < semantic.shape[1]):
                continue
            name = SEMANTIC_NAMES.get(int(semantic[x, y]))
            if not name or name in {"grass", "path"}:
                continue
            entity = NearbyEntity(
                kind=name,
                direction=_direction(dy, dx),
                distance=max(abs(dy), abs(dx)),
                dx=dx,
                dy=dy,
            )
            entities.append(entity)
    return sorted(entities, key=lambda entity: (entity.distance, entity.kind))


def _local_movement_context(info: dict) -> tuple[list[str], str | None]:
    """Return safe adjacent movement actions and the material currently faced."""
    semantic = info.get("semantic")
    player_pos = info.get("player_pos")
    facing = info.get("facing")
    if semantic is None or player_pos is None or facing is None:
        return [], None
    px, py = int(player_pos[0]), int(player_pos[1])
    directions = {
        "move_left": (-1, 0),
        "move_right": (1, 0),
        "move_up": (0, -1),
        "move_down": (0, 1),
    }
    walkable: list[str] = []
    for action, (dx, dy) in directions.items():
        x, y = px + dx, py + dy
        if (
            0 <= x < semantic.shape[0]
            and 0 <= y < semantic.shape[1]
            and SEMANTIC_NAMES.get(int(semantic[x, y])) in {"grass", "path", "sand"}
        ):
            walkable.append(action)
    fx, fy = int(facing[0]), int(facing[1])
    front_x, front_y = px + fx, py + fy
    front_kind = None
    if 0 <= front_x < semantic.shape[0] and 0 <= front_y < semantic.shape[1]:
        front_kind = SEMANTIC_NAMES.get(int(semantic[front_x, front_y]))
    return walkable, front_kind


def _info_to_text(info: dict, step: int) -> str:
    """Convert Crafter's info dict to a human-readable text observation."""
    lines: list[str] = []

    inv = info.get("inventory", {})

    # Vitals
    vitals = []
    for key in ["health", "food", "drink", "energy"]:
        val = inv.get(key, "?")
        bar = (
            "█" * int(val) + "░" * (9 - int(val))
            if isinstance(val, (int, float))
            else ""
        )
        vitals.append(f"{key}: {val}/9 {bar}")
    lines.append("## Vitals")
    lines.extend(vitals)

    # Inventory
    skip = {"health", "food", "drink", "energy"}
    items = {k: v for k, v in inv.items() if k not in skip and v > 0}
    lines.append("\n## Inventory")
    if items:
        lines.append(", ".join(f"{k}: {v}" for k, v in items.items()))
    else:
        lines.append("(empty)")

    # Surroundings from semantic map
    semantic = info.get("semantic", None)
    player_pos = info.get("player_pos", None)
    if semantic is not None and player_pos is not None:
        px, py = int(player_pos[0]), int(player_pos[1])
        lines.append("\n## Surroundings (9×7 grid, you are at center)")

        lines.append("North is up; west is left:")
        for dy in range(-3, 4):
            row = []
            for dx in range(-4, 5):
                x, y = px + dx, py + dy
                if dx == 0 and dy == 0:
                    row.append("@")
                elif 0 <= x < semantic.shape[0] and 0 <= y < semantic.shape[1]:
                    row.append(SEMANTIC_SYMBOLS.get(int(semantic[x, y]), "?"))
                else:
                    row.append(" ")
            lines.append("  " + "".join(row))
        lines.append(
            "Legend: @ you, T tree, ~ water, # stone, : path, s sand, L lava, "
            "B table, F furnace, C cow, Z zombie, K skeleton, P plant"
        )

        nearby_items: dict[str, list[str]] = {}
        for dy in range(-3, 4):
            for dx in range(-4, 5):
                if dy == 0 and dx == 0:
                    continue
                y, x = py + dy, px + dx
                if 0 <= x < semantic.shape[0] and 0 <= y < semantic.shape[1]:
                    sid = int(semantic[x, y])
                    name = SEMANTIC_NAMES.get(sid)
                    if name and name != "grass" and name != "path":
                        direction = _direction(dy, dx)
                        dist = max(abs(dy), abs(dx))
                        nearby_items.setdefault(name, []).append(
                            f"{direction} (dist {dist})"
                        )

        if nearby_items:
            for name, positions in sorted(nearby_items.items()):
                lines.append(f"  {name}: {', '.join(positions[:3])}")
        else:
            lines.append("  Nothing notable nearby, just grass.")

    facing = info.get("facing")
    if facing is not None:
        lines.append(f"\n## Facing: {_direction(int(facing[1]), int(facing[0]))}")
    if "daylight" in info:
        lines.append(f"Daylight: {float(info['daylight']):.2f}")

    # Time
    if hasattr(info, "get"):
        sleeping = info.get("sleeping", False)
        if sleeping:
            lines.append("\n## Status: SLEEPING")

    # Achievements so far
    unlocked = info.get("achievements", {})
    done = [k for k, v in unlocked.items() if v]
    if done:
        lines.append(f"\n## Achievements unlocked: {', '.join(done)}")

    lines.append(f"\n[Step {step}]")
    return "\n".join(lines)


def _direction(dy: int, dx: int) -> str:
    if dy == 0 and dx == 0:
        return "here"
    parts = []
    if dy < 0:
        parts.append("north")
    elif dy > 0:
        parts.append("south")
    if dx < 0:
        parts.append("west")
    elif dx > 0:
        parts.append("east")
    return "-".join(parts)


class CrafterTextEnv:
    """Wraps Crafter to expose text observations and string actions."""

    def __init__(self, seed: int | None = None, max_steps: int = 200):
        if max_steps <= 0:
            raise ValueError("max_steps must be positive")
        self.env = crafter.Env(size=(64, 64), seed=seed)
        self.max_steps = max_steps
        self._step_count = 0
        self._total_reward = 0.0
        self._achievements: dict[str, bool] = {a: False for a in ALL_ACHIEVEMENTS}
        self._done = True
        self.info: dict = {}

    def reset(self) -> str:
        self.env.reset()
        self._step_count = 0
        self._total_reward = 0.0
        self._achievements = {a: False for a in ALL_ACHIEVEMENTS}
        player = self.env._player
        self.info = {
            "inventory": player.inventory.copy(),
            "achievements": player.achievements.copy(),
            "semantic": self.env._sem_view(),
            "player_pos": player.pos.copy(),
            "sleeping": bool(player.sleeping),
            "facing": tuple(player.facing),
            "daylight": float(self.env._world.daylight),
        }
        self._done = False
        return _info_to_text(self.info, 0)

    def step(self, action_name: str) -> tuple[str, float, bool, dict]:
        if self._done:
            raise RuntimeError("Reset before stepping a terminal environment")
        if action_name not in ACTION_TO_IDX:
            raise ValueError(f"Unknown action: {action_name}")
        idx = ACTION_TO_IDX[action_name]
        _obs, reward, done, info = self.env.step(idx)
        self._step_count += 1
        self._total_reward += reward

        for ach in ALL_ACHIEVEMENTS:
            if info.get("achievements", {}).get(ach, False):
                self._achievements[ach] = True

        if self._step_count >= self.max_steps:
            done = True

        info.update(
            {
                "sleeping": bool(self.env._player.sleeping),
                "facing": tuple(self.env._player.facing),
                "daylight": float(self.env._world.daylight),
            }
        )
        self.info = info
        self._done = bool(done)

        text_obs = _info_to_text(info, self._step_count)
        return text_obs, reward, done, info

    def get_score(self) -> float:
        """Return upstream Crafter score for this episode's binary percentages."""
        percents = [
            100.0 if self._achievements.get(achievement) else 0.0
            for achievement in ALL_ACHIEVEMENTS
        ]
        return float(np.expm1(np.mean(np.log1p(percents))))

    def render(self) -> np.ndarray:
        """Render a frame without allowing display noise to alter gameplay RNG."""
        rng = self.env._world.random
        state = rng.get_state()
        try:
            return self.env.render(size=(512, 512)).copy()
        finally:
            rng.set_state(state)

    def get_achievements(self) -> dict[str, bool]:
        return dict(self._achievements)

    def snapshot(self) -> CrafterSnapshot:
        """Return the current structured state without advancing or rendering."""
        inventory = {
            key: int(value)
            for key, value in self.info.get("inventory", {}).items()
            if isinstance(value, (int, float, np.integer, np.floating))
        }
        facing = self.info.get("facing", (0, 0))
        player_pos = self.info.get("player_pos", (0, 0))
        walkable_directions, front_kind = _local_movement_context(self.info)
        return CrafterSnapshot(
            step=self._step_count,
            inventory=inventory,
            achievements=self.get_achievements(),
            nearby_entities=_nearby_entities(self.info),
            facing=_direction(int(facing[1]), int(facing[0])),
            position=(int(player_pos[0]), int(player_pos[1])),
            walkable_directions=walkable_directions,
            front_kind=front_kind,
            daylight=float(self.info.get("daylight", 1.0)),
            sleeping=bool(self.info.get("sleeping", False)),
        )

    @property
    def step_count(self) -> int:
        return self._step_count
