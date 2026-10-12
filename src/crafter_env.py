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

# Crafting recipes — what you need in inventory to perform each action
RECIPES = {
    "place_table": {"wood": 2},
    "place_furnace": {"stone": 4},
    "place_stone": {"stone": 1},
    "place_plant": {"sapling": 1},
    "make_wood_pickaxe": {"wood": 1},       # need table placed nearby
    "make_stone_pickaxe": {"wood": 1, "stone": 1},  # need table nearby
    "make_iron_pickaxe": {"wood": 1, "iron": 1},    # need table+furnace nearby
    "make_wood_sword": {"wood": 1},          # need table nearby
    "make_stone_sword": {"wood": 1, "stone": 1},    # need table nearby
    "make_iron_sword": {"wood": 1, "iron": 1},      # need table+furnace nearby
}

# Achievement tech tree: what you need to unlock each
ACHIEVEMENT_PREREQS = {
    "collect_wood": "Use 'do' next to a tree.",
    "collect_stone": "Need wood_pickaxe. Use 'do' next to stone.",
    "collect_coal": "Need wood_pickaxe. Use 'do' next to coal (underground).",
    "collect_iron": "Need stone_pickaxe. Use 'do' next to iron (underground).",
    "collect_diamond": "Need iron_pickaxe. Use 'do' next to diamond (deep underground).",
    "collect_drink": "Use 'do' next to water.",
    "collect_sapling": "Use 'do' next to a tree (sometimes drops sapling).",
    "eat_cow": "Use 'do' next to a cow.",
    "eat_plant": "Use 'do' next to a plant.",
    "defeat_zombie": "Use 'do' next to a zombie (need sword).",
    "defeat_skeleton": "Use 'do' next to a skeleton (need sword).",
    "place_table": "Need 2 wood. Use place_table action.",
    "place_furnace": "Need 4 stone. Use place_furnace action.",
    "place_stone": "Need 1 stone. Use place_stone action.",
    "place_plant": "Need 1 sapling. Use place_plant action.",
    "make_wood_pickaxe": "Need 1 wood + table nearby. Use make_wood_pickaxe.",
    "make_stone_pickaxe": "Need 1 wood + 1 stone + table nearby.",
    "make_iron_pickaxe": "Need 1 wood + 1 iron + table + furnace nearby.",
    "make_wood_sword": "Need 1 wood + table nearby.",
    "make_stone_sword": "Need 1 wood + 1 stone + table nearby.",
    "make_iron_sword": "Need 1 wood + 1 iron + table + furnace nearby.",
    "wake_up": "Use sleep action, then wake up.",
}

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


def _info_to_text(info: dict, step: int, prev_inv: dict | None = None) -> str:
    """Convert Crafter's info dict to a rich text observation."""
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

    # Inventory with change indicators
    skip = {"health", "food", "drink", "energy"}
    items = {k: v for k, v in inv.items() if k not in skip and v > 0}
    lines.append("\n## Inventory")
    if items:
        inv_parts = []
        for k, v in items.items():
            change = ""
            if prev_inv and k in prev_inv:
                delta = v - prev_inv.get(k, 0)
                if delta > 0:
                    change = f" (+{delta})"
                elif delta < 0:
                    change = f" ({delta})"
            inv_parts.append(f"{k}: {v}{change}")
        lines.append(", ".join(inv_parts))
    else:
        lines.append("(empty)")

    # Surroundings from semantic map
    semantic = info.get("semantic", None)
    player_pos = info.get("player_pos", None)
    if semantic is not None and player_pos is not None:
        py, px = int(player_pos[0]), int(player_pos[1])

        # What's directly at player feet and adjacent (dist 1)
        facing_items = []
        for dy in range(-1, 2):
            for dx in range(-1, 2):
                if dy == 0 and dx == 0:
                    continue
                y, x = py + dy, px + dx
                if 0 <= y < semantic.shape[0] and 0 <= x < semantic.shape[1]:
                    sid = int(semantic[y, x])
                    name = SEMANTIC_NAMES.get(sid)
                    if name and name != "grass" and name != "path":
                        d = _direction(dy, dx)
                        facing_items.append(f"{name} ({d})")

        if facing_items:
            lines.append(f"\n## Adjacent (can interact with 'do'): {', '.join(facing_items)}")

        # Wider area
        lines.append("\n## Nearby (7×7 view)")
        nearby_items: dict[str, list[str]] = {}
        for dy in range(-3, 4):
            for dx in range(-3, 4):
                if abs(dy) <= 1 and abs(dx) <= 1:
                    continue  # already shown above
                y, x = py + dy, px + dx
                if 0 <= y < semantic.shape[0] and 0 <= x < semantic.shape[1]:
                    sid = int(semantic[y, x])
                    name = SEMANTIC_NAMES.get(sid)
                    if name and name not in ("grass", "path", None):
                        direction = _direction(dy, dx)
                        dist = max(abs(dy), abs(dx))
                        nearby_items.setdefault(name, []).append(
                            f"{direction} (dist {dist})"
                        )

        if nearby_items:
            for name, positions in sorted(nearby_items.items()):
                lines.append(f"  {name}: {', '.join(positions[:4])}")
        else:
            lines.append("  Nothing notable beyond adjacent tiles.")

    # Craftable right now
    craftable = _get_craftable(inv, semantic, player_pos)
    if craftable:
        lines.append(f"\n## Can craft NOW: {', '.join(craftable)}")

    # Next achievable goals
    unlocked = info.get("achievements", {})
    done = [k for k, v in unlocked.items() if v]
    not_done = [k for k in ALL_ACHIEVEMENTS if k not in done or not unlocked.get(k)]
    if done:
        lines.append(f"\n## Achievements ({len(done)}/{len(ALL_ACHIEVEMENTS)}): {', '.join(done)}")

    # Time of day
    sleeping = info.get("sleeping", False)
    if sleeping:
        lines.append("\n## Status: SLEEPING")

    light = info.get("light_level", None)
    if light is not None:
        phase = "DAY" if light > 0 else "NIGHT ⚠"
        lines.append(f"\n## Time: {phase}")

    lines.append(f"\n[Step {step}]")
    return "\n".join(lines)


def _get_craftable(inv: dict, semantic, player_pos) -> list[str]:
    """Check which crafting actions are available right now."""
    craftable = []
    skip = {"health", "food", "drink", "energy"}
    items = {k: v for k, v in inv.items() if k not in skip}

    # Check if table/furnace are adjacent
    has_table_nearby = False
    has_furnace_nearby = False
    if semantic is not None and player_pos is not None:
        py, px = int(player_pos[0]), int(player_pos[1])
        for dy in range(-2, 3):
            for dx in range(-2, 3):
                y, x = py + dy, px + dx
                if 0 <= y < semantic.shape[0] and 0 <= x < semantic.shape[1]:
                    sid = int(semantic[y, x])
                    if sid == 11:
                        has_table_nearby = True
                    elif sid == 12:
                        has_furnace_nearby = True

    for action, recipe in RECIPES.items():
        if all(items.get(r, 0) >= v for r, v in recipe.items()):
            if action.startswith("make_"):
                if "iron" in action and not (has_table_nearby and has_furnace_nearby):
                    continue
                elif not has_table_nearby:
                    continue
            craftable.append(action)

    return craftable


def _direction(dy: int, dx: int) -> str:
    """Convert (row_delta, col_delta) to cardinal direction.
    
    Crafter coordinate system:
      - row increases → move_right (east on screen)
      - col increases → move_down (south on screen)
      - move_up decreases col, move_left decreases row
    So: dy (row delta) = east/west, dx (col delta) = north/south.
    """
    if dy == 0 and dx == 0:
        return "here"
    parts = []
    if dx < 0:
        parts.append("north")
    elif dx > 0:
        parts.append("south")
    if dy < 0:
        parts.append("west")
    elif dy > 0:
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
        self._prev_inv: dict | None = None

    def reset(self) -> str:
        obs = self.env.reset()
        self._step_count = 0
        self._total_reward = 0.0
        self._achievements = {a: False for a in ALL_ACHIEVEMENTS}
        self._frames = [obs.copy()]
        self._prev_inv = None
        # Do a noop step to get info
        _, _, _, info = self.env.step(0)
        self._step_count = 0  # don't count the noop
        self._prev_inv = dict(info.get("inventory", {}))
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

        text_obs = _info_to_text(info, self._step_count, self._prev_inv)
        self._prev_inv = dict(info.get("inventory", {}))
        return text_obs, reward, done, info

    @property
    def frames(self) -> list[np.ndarray]:
        return self._frames

    def get_score(self) -> float:
        """Per-episode score: count of achievements unlocked."""
        unlocked = sum(1 for v in self._achievements.values() if v)
        return float(unlocked)

    def get_achievements(self) -> dict[str, bool]:
        return dict(self._achievements)

    @property
    def step_count(self) -> int:
        return self._step_count


def crafter_score(achievement_rates: dict[str, float]) -> float:
    """Upstream Crafter aggregate score: exp(mean(log(1 + 100*rate))) - 1.
    
    See: https://github.com/danijar/crafter/blob/main/analysis/common.py
    achievement_rates: {achievement_name: success_rate} where rate is 0.0-1.0
    """
    rates = []
    for ach in ALL_ACHIEVEMENTS:
        rate = achievement_rates.get(ach, 0.0)
        rates.append(np.log(1 + 100 * rate))
    return float(np.exp(np.mean(rates)) - 1)
