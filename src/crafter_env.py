"""Crafter environment wrapper that produces text observations for LLM agents."""

from __future__ import annotations

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


def _info_to_text(info: dict, step: int) -> str:
    """Convert Crafter's info dict to a human-readable text observation."""
    lines: list[str] = []

    inv = info.get("inventory", {})

    # Vitals
    vitals = []
    for key in ["health", "food", "drink", "energy"]:
        val = inv.get(key, "?")
        bar = "█" * int(val) + "░" * (9 - int(val)) if isinstance(val, (int, float)) else ""
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
        py, px = int(player_pos[0]), int(player_pos[1])
        lines.append("\n## Surroundings (7×7 grid, you are at center)")

        nearby_items: dict[str, list[str]] = {}
        for dy in range(-3, 4):
            for dx in range(-3, 4):
                if dy == 0 and dx == 0:
                    continue
                y, x = py + dy, px + dx
                if 0 <= y < semantic.shape[0] and 0 <= x < semantic.shape[1]:
                    sid = int(semantic[y, x])
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
        self.env = crafter.Env(size=(64, 64), seed=seed)
        self.max_steps = max_steps
        self._step_count = 0
        self._total_reward = 0.0
        self._achievements: dict[str, bool] = {a: False for a in ALL_ACHIEVEMENTS}
        self._frames: list[np.ndarray] = []

    def reset(self) -> str:
        obs = self.env.reset()
        self._step_count = 0
        self._total_reward = 0.0
        self._achievements = {a: False for a in ALL_ACHIEVEMENTS}
        self._frames = [obs.copy()]
        # Do a noop step to get info
        _, _, _, info = self.env.step(0)
        self._step_count = 0  # don't count the noop
        return _info_to_text(info, 0)

    def step(self, action_name: str) -> tuple[str, float, bool, dict]:
        idx = ACTION_TO_IDX.get(action_name, 0)
        obs, reward, done, info = self.env.step(idx)
        self._step_count += 1
        self._total_reward += reward
        self._frames.append(obs.copy())

        for ach in ALL_ACHIEVEMENTS:
            if info.get("achievements", {}).get(ach, False):
                self._achievements[ach] = True

        if self._step_count >= self.max_steps:
            done = True

        text_obs = _info_to_text(info, self._step_count)
        return text_obs, reward, done, info

    @property
    def frames(self) -> list[np.ndarray]:
        return self._frames

    def get_score(self) -> float:
        """Crafter score: geometric mean of (success_rate + 1%) across achievements."""
        rates = []
        for ach in ALL_ACHIEVEMENTS:
            rates.append(1.0 if self._achievements.get(ach) else 0.0)
        clipped = [max(r, 0.01) for r in rates]
        return float(np.exp(np.mean(np.log(clipped)))) * 100

    def get_achievements(self) -> dict[str, bool]:
        return dict(self._achievements)

    @property
    def step_count(self) -> int:
        return self._step_count
