"""SPRING-lite planner: survival first, then the Crafter tech tree."""

from __future__ import annotations

import json
import re

from src.config import settings
from src.crafter_env import GameState
from src.genome import Genome
from src.skills import WorldMap

GOALS = {
    "flee",
    "attack",
    "sleep_safe",
    "drink",
    "eat",
    "collect_wood",
    "collect_sapling",
    "collect_stone",
    "collect_coal",
    "collect_iron",
    "collect_diamond",
    "place_table",
    "place_furnace",
    "place_plant",
    "place_stone",
    "make_wood_pickaxe",
    "make_stone_pickaxe",
    "make_iron_pickaxe",
    "make_wood_sword",
    "make_stone_sword",
    "make_iron_sword",
    "relocate_furnace",
    "craft_iron",
    "retreat_to_camp",
    "farm_plants",
    "kite",
    "explore",
}


def choose_goal(
    state: GameState,
    world: WorldMap,
    genome: Genome | None = None,
    stuck_goal: str | None = None,
) -> str:
    """Pick the next macro. Default is fully scripted (no LLM)."""
    mode = getattr(settings, "planner_mode", "scripted")
    if mode == "llm" and stuck_goal:
        goal = _llm_goal(state, world, genome)
        if goal:
            return goal
    return scripted_goal(state, world, genome)


def scripted_goal(
    state: GameState,
    world: WorldMap,
    genome: Genome | None = None,
) -> str:
    inv = state.inventory
    vitals = state.vitals
    ach = state.achievements
    policy = genome.genome.planner if genome else None
    rules = " ".join(genome.genome.rules).lower() if genome else ""
    prefer_sword = bool(policy.prefer_sword) if policy else ("sword" in rules)
    attack_unarmed = bool(policy.attack_unarmed) if policy else False
    energy_sleep = policy.energy_sleep if policy else 2
    drink_at = policy.drink_threshold if policy else 2
    food_at = policy.food_threshold if policy else 2
    health_flee = policy.health_flee if policy else 2
    relocate = bool(policy.relocate_workstation) if policy else False
    sleep_night = bool(policy.sleep_at_night) if policy else False
    retreat_night = bool(policy.retreat_at_night) if policy else True
    wood_need = policy.wood_for_table if policy else 2
    stone_need = policy.stone_for_furnace if policy else 4
    extra = set(policy.extra_goals) if policy else {"retreat_to_camp", "craft_iron", "farm_plants"}
    disabled = set(policy.disabled_goals) if policy else set()

    def pick(goal: str) -> str | None:
        if goal in disabled:
            return None
        if goal in {"retreat_to_camp", "craft_iron", "farm_plants"} and goal not in extra:
            return None
        return goal

    adjacent_threat = any(
        t.name in {"zombie", "skeleton"} and t.chebyshev == 1 for t in state.tiles
    )
    nearby_threat = any(t.name in {"zombie", "skeleton"} for t in state.tiles)
    has_sword = any(inv.get(k, 0) > 0 for k in ("wood_sword", "stone_sword", "iron_sword"))
    health = vitals.get("health", 9)
    drink = vitals.get("drink", 9)
    food = vitals.get("food", 9)
    at_camp = bool(world.camp) and max(abs(state.x - world.camp[0]), abs(state.y - world.camp[1])) <= 2
    wood = inv.get("wood", 0)
    stone = inv.get("stone", 0)
    stone_pick = inv.get("stone_pickaxe", 0) > 0
    iron_pick = inv.get("iron_pickaxe", 0) > 0
    wood_pick = inv.get("wood_pickaxe", 0) > 0
    has_table = ach.get("place_table") or bool(world.named("table")) or bool(world.camp)
    has_furnace = ach.get("place_furnace") or bool(world.named("furnace"))
    ore_lock = stone_pick and not iron_pick
    critical = health <= max(2, health_flee)
    use_kite = (not state.is_night) and has_sword and health >= 6

    def go_home() -> str | None:
        if at_camp:
            return None
        return pick("retreat_to_camp")

    if adjacent_threat:
        if use_kite:
            return pick("kite") or "flee"
        return pick("attack") if (attack_unarmed and not state.is_night) else "flee"
    if critical and nearby_threat:
        return "flee"

    if drink <= 1:
        return pick("drink") or "explore"
    if food <= 1:
        return pick("eat") or "explore"

    if state.is_night and has_table and retreat_night:
        if world.camp and not at_camp:
            dist = abs(state.x - world.camp[0]) + abs(state.y - world.camp[1])
            if dist <= 10:
                return go_home() or pick("sleep_safe") or "flee"
        if vitals.get("energy", 9) <= 7:
            return pick("sleep_safe") or "flee"
        return pick("farm_plants") or pick("sleep_safe") or "flee"

    if critical:
        if drink <= drink_at:
            return pick("drink") or "explore"
        if food <= food_at:
            return pick("eat") or "explore"

    if ore_lock:
        return pick("craft_iron") or "explore"

    if drink <= drink_at:
        return pick("drink") or "explore"
    if food <= food_at:
        return pick("eat") or "explore"

    need_rest = vitals.get("energy", 9) <= energy_sleep or (sleep_night and state.is_night)
    if need_rest and not adjacent_threat:
        return pick("sleep_safe") or "explore"

    if not has_table:
        if wood < wood_need:
            return pick("collect_wood") if world.named("tree") else "explore"
        return pick("place_table") or "explore"

    if not wood_pick:
        if wood < 1:
            return "collect_wood" if world.named("tree") else "explore"
        return "make_wood_pickaxe"

    if prefer_sword and inv.get("wood_sword", 0) == 0 and wood >= 1:
        return "make_wood_sword"

    if stone < 1:
        if world.named("stone"):
            return "collect_stone"
        return "explore"

    if inv.get("wood_sword", 0) == 0 and inv.get("stone_sword", 0) == 0 and wood >= 1:
        return "make_wood_sword" if stone < 1 else "make_stone_sword"

    if not has_furnace:
        if stone < stone_need:
            return pick("collect_stone") if world.named("stone") else "explore"
        return pick("place_furnace") or "explore"

    if not stone_pick:
        if wood < 1:
            return "collect_wood"
        if stone < 1:
            return "collect_stone"
        return "make_stone_pickaxe"

    if not iron_pick:
        if relocate and not _stations_paired(world) and stone >= 4:
            return pick("relocate_furnace") or pick("craft_iron") or "explore"
        return pick("craft_iron") or "make_iron_pickaxe"

    if world.named("diamond"):
        return "collect_diamond"

    farm = pick("farm_plants")
    if farm:
        return farm
    if inv.get("sapling", 0) == 0:
        return "collect_sapling"
    if not ach.get("place_plant"):
        return "place_plant"

    if nearby_threat and use_kite and at_camp:
        return pick("kite") or pick("attack") or "explore"

    return "explore"


def _stations_paired(world: WorldMap) -> bool:
    tables = world.named("table")
    furnaces = world.named("furnace")
    for table in tables:
        for furnace in furnaces:
            if max(abs(table[0] - furnace[0]), abs(table[1] - furnace[1])) <= 2:
                return True
    return False


def _llm_goal(state: GameState, world: WorldMap, genome: Genome | None) -> str | None:
    from src.llm import chat

    inv = {k: v for k, v in state.inventory.items() if v}
    nearby = sorted({t.name for t in state.tiles if t.name not in {"grass", "path", "player"}})
    prompt = (
        "Pick the next Crafter goal. Reply with ONLY JSON: {\"goal\": \"...\"}.\n"
        f"Allowed goals: {sorted(GOALS)}\n"
        f"Vitals: {state.vitals}\n"
        f"Inventory: {inv}\n"
        f"Facing: {state.facing} front={state.tile_in_front} night={state.is_night}\n"
        f"Nearby: {nearby}\n"
        f"Known resources: trees={len(world.named('tree'))} stone={len(world.named('stone'))} "
        f"water={len(world.named('water'))} table={len(world.named('table'))}\n"
    )
    if genome and genome.genome.rules:
        prompt += "Rules: " + "; ".join(genome.genome.rules) + "\n"
    try:
        reply = chat(
            messages=[{"role": "user", "content": prompt}],
            model=settings.agent_model,
            max_tokens=80,
            temperature=0.2,
        )
    except Exception:
        return None
    match = re.search(r"\{[^}]+\}", reply)
    if not match:
        return None
    try:
        goal = json.loads(match.group()).get("goal")
    except json.JSONDecodeError:
        return None
    if goal in GOALS:
        return goal
    return None
