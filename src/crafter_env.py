"""Crafter wrapper with structured state, facing, and correct (x, y) axes."""

from __future__ import annotations

from dataclasses import dataclass, field

import crafter
import numpy as np

ACTIONS = [
    "noop", "move_left", "move_right", "move_up", "move_down",
    "do", "sleep", "place_stone", "place_table", "place_furnace",
    "place_plant", "make_wood_pickaxe", "make_stone_pickaxe",
    "make_iron_pickaxe", "make_wood_sword", "make_stone_sword",
    "make_iron_sword",
]

ACTION_TO_IDX = {a: i for i, a in enumerate(ACTIONS)}

ALL_ACHIEVEMENTS = [
    "collect_coal", "collect_diamond", "collect_drink", "collect_iron",
    "collect_sapling", "collect_stone", "collect_wood", "defeat_skeleton",
    "defeat_zombie", "eat_cow", "eat_plant", "make_iron_pickaxe",
    "make_iron_sword", "make_stone_pickaxe", "make_stone_sword",
    "make_wood_pickaxe", "make_wood_sword", "place_furnace", "place_plant",
    "place_stone", "place_table", "wake_up",
]

# Crafter semantic canvas is indexed [x, y] and pos is (x, y).
MATERIAL_NAMES = {
    0: None, 1: "water", 2: "grass", 3: "stone", 4: "path",
    5: "sand", 6: "tree", 7: "lava", 8: "coal", 9: "iron",
    10: "diamond", 11: "table", 12: "furnace",
}

OBJECT_NAMES = {
    13: "player", 14: "cow", 15: "zombie", 16: "skeleton",
    17: "arrow", 18: "plant",
}

SEMANTIC_NAMES = {**MATERIAL_NAMES, **OBJECT_NAMES}

WALKABLE = {"grass", "path", "sand"}
BLOCKING_OBJECTS = {"cow", "zombie", "skeleton", "arrow", "plant", "player"}

FACING_DELTA = {
    "left": (-1, 0),
    "right": (1, 0),
    "up": (0, -1),
    "down": (0, 1),
}
DELTA_FACING = {v: k for k, v in FACING_DELTA.items()}
DELTA_MOVE = {
    (-1, 0): "move_left",
    (1, 0): "move_right",
    (0, -1): "move_up",
    (0, 1): "move_down",
}

PLACE_WHERE = {
    "stone": {"grass", "sand", "path", "water", "lava"},
    "table": {"grass", "sand", "path"},
    "furnace": {"grass", "sand", "path"},
    "plant": {"grass"},
}

VITAL_KEYS = ("health", "food", "drink", "energy")


@dataclass
class Tile:
    name: str
    x: int
    y: int
    dx: int
    dy: int

    @property
    def pos(self) -> tuple[int, int]:
        return (self.x, self.y)

    @property
    def chebyshev(self) -> int:
        return max(abs(self.dx), abs(self.dy))


@dataclass
class GameState:
    """Structured snapshot used by macros and the planner."""

    step: int
    pos: tuple[int, int]
    facing: str
    inventory: dict[str, int]
    vitals: dict[str, int]
    achievements: dict[str, bool]
    tiles: list[Tile]
    daylight: float
    sleeping: bool
    tile_in_front: str | None
    semantic: np.ndarray | None = field(default=None, repr=False)

    @property
    def x(self) -> int:
        return self.pos[0]

    @property
    def y(self) -> int:
        return self.pos[1]

    @property
    def is_night(self) -> bool:
        return self.daylight < 0.5

    def tiles_named(self, name: str) -> list[Tile]:
        return [t for t in self.tiles if t.name == name]

    def adjacent(self, name: str | None = None) -> list[Tile]:
        tiles = [t for t in self.tiles if t.chebyshev == 1]
        if name is not None:
            tiles = [t for t in tiles if t.name == name]
        return tiles

    def nearest(self, name: str) -> Tile | None:
        found = self.tiles_named(name)
        if not found:
            return None
        return min(found, key=lambda t: abs(t.dx) + abs(t.dy))


def facing_from_delta(dx: int, dy: int) -> str | None:
    return DELTA_FACING.get((dx, dy))


def _facing_name(facing_xy) -> str:
    dx, dy = int(facing_xy[0]), int(facing_xy[1])
    return DELTA_FACING.get((dx, dy), "down")


def _crop_tiles(semantic: np.ndarray, px: int, py: int, radius: int = 4) -> list[Tile]:
    tiles: list[Tile] = []
    width, height = semantic.shape
    for dx in range(-radius, radius + 1):
        for dy in range(-radius, radius + 1):
            x, y = px + dx, py + dy
            if 0 <= x < width and 0 <= y < height:
                name = SEMANTIC_NAMES.get(int(semantic[x, y]))
                if name:
                    tiles.append(Tile(name=name, x=x, y=y, dx=dx, dy=dy))
    return tiles


def _info_to_text(state: GameState) -> str:
    """Human-readable observation for LLM / logs."""
    lines: list[str] = []
    lines.append("## Vitals")
    for key in VITAL_KEYS:
        val = state.vitals.get(key, "?")
        bar = ""
        if isinstance(val, (int, float)):
            bar = "█" * int(val) + "░" * (9 - int(val))
        lines.append(f"{key}: {val}/9 {bar}")

    skip = set(VITAL_KEYS)
    items = {k: v for k, v in state.inventory.items() if k not in skip and v > 0}
    lines.append("\n## Inventory")
    lines.append(", ".join(f"{k}: {v}" for k, v in items.items()) if items else "(empty)")

    lines.append(f"\n## Position")
    lines.append(f"pos=({state.x}, {state.y}) facing={state.facing}")
    lines.append(f"tile in front: {state.tile_in_front or 'none'}")
    lines.append(f"daylight: {state.daylight:.2f} ({'night' if state.is_night else 'day'})")
    if state.sleeping:
        lines.append("status: SLEEPING")

    lines.append("\n## Surroundings (you are at center; +x east, -y north)")
    nearby = [t for t in state.tiles if t.name not in {"grass", "path", "player"}]
    if nearby:
        grouped: dict[str, list[str]] = {}
        for t in sorted(nearby, key=lambda t: t.chebyshev):
            label = facing_from_delta(t.dx, 0) or facing_from_delta(0, t.dy) or "here"
            if t.dx and t.dy:
                ew = "east" if t.dx > 0 else "west"
                ns = "south" if t.dy > 0 else "north"
                label = f"{ns}-{ew}"
            grouped.setdefault(t.name, []).append(f"{label} (dx={t.dx}, dy={t.dy})")
        for name, positions in grouped.items():
            lines.append(f"  {name}: {', '.join(positions[:4])}")
    else:
        lines.append("  Nothing notable nearby, just grass.")

    done = [k for k, v in state.achievements.items() if v]
    if done:
        lines.append(f"\n## Achievements unlocked: {', '.join(done)}")

    lines.append(f"\n[Step {state.step}]")
    return "\n".join(lines)


class CrafterTextEnv:
    """Wraps Crafter to expose GameState, text observations, and string actions."""

    def __init__(self, seed: int | None = None, max_steps: int = 2000, view_radius: int = 4):
        self.max_steps = max_steps
        self.view_radius = view_radius
        self.env = crafter.Env(area=(64, 64), size=(64, 64), length=max_steps, seed=seed)
        self._step_count = 0
        self._total_reward = 0.0
        self._achievements: dict[str, bool] = {a: False for a in ALL_ACHIEVEMENTS}
        self._last_state: GameState | None = None
        self._last_rgb: np.ndarray | None = None
        self._record_stride = 0
        self._recorded: list[tuple[int, np.ndarray]] = []

    def enable_recording(self, stride: int) -> None:
        """Keep RGB frames already produced by Crafter's step/reset obs."""
        self._record_stride = max(1, int(stride))
        self._recorded = []

    def pop_recorded(self) -> list[tuple[int, np.ndarray]]:
        frames = self._recorded
        self._recorded = []
        return frames

    def _maybe_record(self, rgb) -> None:
        self._last_rgb = None if rgb is None else np.asarray(rgb)
        if not self._record_stride or self._last_rgb is None:
            return
        if self._step_count % self._record_stride == 0:
            self._recorded.append((self._step_count, self._last_rgb.copy()))

    def reset(self) -> GameState:
        rgb = self.env.reset()
        self._step_count = 0
        self._total_reward = 0.0
        self._achievements = {a: False for a in ALL_ACHIEVEMENTS}
        self._recorded = []
        self._maybe_record(rgb)
        state = self._read_state()
        self._last_state = state
        return state

    def step(self, action_name: str) -> tuple[GameState, float, bool, dict]:
        idx = ACTION_TO_IDX.get(action_name, 0)
        rgb, reward, done, info = self.env.step(idx)
        self._step_count += 1
        self._total_reward += reward
        self._maybe_record(rgb)

        for ach in ALL_ACHIEVEMENTS:
            if info.get("achievements", {}).get(ach, 0):
                self._achievements[ach] = True

        if self._step_count >= self.max_steps:
            done = True

        state = self._read_state()
        self._last_state = state
        return state, reward, done, info

    def _read_state(self) -> GameState:
        player = self.env._player
        semantic = self.env._sem_view()
        px, py = int(player.pos[0]), int(player.pos[1])
        facing = _facing_name(player.facing)
        fdx, fdy = FACING_DELTA[facing]
        fx, fy = px + fdx, py + fdy
        tile_in_front = None
        if 0 <= fx < semantic.shape[0] and 0 <= fy < semantic.shape[1]:
            tile_in_front = SEMANTIC_NAMES.get(int(semantic[fx, fy]))

        inventory = {k: int(v) for k, v in player.inventory.items()}
        vitals = {k: inventory.get(k, 0) for k in VITAL_KEYS}
        tiles = _crop_tiles(semantic, px, py, self.view_radius)
        return GameState(
            step=self._step_count,
            pos=(px, py),
            facing=facing,
            inventory=inventory,
            vitals=vitals,
            achievements=dict(self._achievements),
            tiles=tiles,
            daylight=float(self.env._world.daylight),
            sleeping=bool(player.sleeping),
            tile_in_front=tile_in_front,
            semantic=semantic,
        )

    def observe_text(self, state: GameState | None = None) -> str:
        return _info_to_text(state or self._last_state or self._read_state())

    def get_score(self) -> float:
        """Per-episode geometric mean used by this project (0–100)."""
        rates = [1.0 if self._achievements.get(ach) else 0.0 for ach in ALL_ACHIEVEMENTS]
        clipped = [max(r, 0.01) for r in rates]
        return float(np.exp(np.mean(np.log(clipped)))) * 100

    def get_achievements(self) -> dict[str, bool]:
        return dict(self._achievements)

    @property
    def step_count(self) -> int:
        return self._step_count
