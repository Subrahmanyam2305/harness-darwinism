"""Meta-agent: crossover and trace-driven mutation of planner skills."""

from __future__ import annotations

import json
import random
import re

from src.config import settings
from src.genome import ALLOWED_EXTRA_GOALS, GenomeConfig, PlannerPolicy
from src.llm import chat


def crossover(parent_a: GenomeConfig, parent_b: GenomeConfig) -> GenomeConfig:
    if settings.meta_mode == "scripted":
        return heuristic_crossover(parent_a, parent_b)
    try:
        return llm_crossover(parent_a, parent_b)
    except Exception:
        return heuristic_crossover(parent_a, parent_b)


def mutate(
    genome: GenomeConfig,
    failure_traces: list[str],
    tried_edits: list[str] | None = None,
    arm: str = "full",
) -> tuple[GenomeConfig, str]:
    if arm == "random_mutate":
        return random_mutate(genome)
    if settings.meta_mode == "scripted":
        return heuristic_mutate(genome, failure_traces)
    try:
        return llm_mutate(genome, failure_traces, tried_edits)
    except Exception:
        return heuristic_mutate(genome, failure_traces)


def random_mutate(genome: GenomeConfig) -> tuple[GenomeConfig, str]:
    """Ablation: ignore traces, flip one random planner flag."""
    child = genome.model_copy(deep=True)
    p = child.planner
    options = [
        ("relocate_workstation", not p.relocate_workstation),
        ("pair_stations", not p.pair_stations),
        ("kite_combat", not p.kite_combat),
        ("prefer_sword", not p.prefer_sword),
        ("sleep_at_night", not p.sleep_at_night),
        ("drink_threshold", 2 if p.drink_threshold != 2 else 5),
        ("energy_sleep", 2 if p.energy_sleep != 2 else 4),
        ("retreat_at_night", not p.retreat_at_night),
        ("safe_vitals", 5 if p.safe_vitals != 5 else 7),
    ]
    name, value = random.choice(options)
    setattr(p, name, value)
    return child, f"random_mutate flipped {name}={value}"


def heuristic_crossover(parent_a: GenomeConfig, parent_b: GenomeConfig) -> GenomeConfig:
    """Mix complementary planner knobs and rules without an LLM."""
    pa, pb = parent_a.planner, parent_b.planner
    child = parent_a.model_copy(deep=True)
    child.planner = PlannerPolicy(
        energy_sleep=min(pa.energy_sleep, pb.energy_sleep),
        drink_threshold=min(pa.drink_threshold, pb.drink_threshold),
        food_threshold=min(pa.food_threshold, pb.food_threshold),
        health_flee=max(pa.health_flee, pb.health_flee),
        prefer_sword=pa.prefer_sword or pb.prefer_sword,
        attack_unarmed=pa.attack_unarmed and pb.attack_unarmed,
        relocate_workstation=pa.relocate_workstation or pb.relocate_workstation,
        pair_stations=pa.pair_stations or pb.pair_stations,
        kite_combat=pa.kite_combat or pb.kite_combat,
        sleep_at_night=pa.sleep_at_night or pb.sleep_at_night,
        retreat_at_night=pa.retreat_at_night or pb.retreat_at_night,
        safe_vitals=max(pa.safe_vitals, pb.safe_vitals),
        wood_for_table=max(pa.wood_for_table, pb.wood_for_table),
        stone_for_furnace=min(pa.stone_for_furnace, pb.stone_for_furnace),
        extra_goals=sorted(set(pa.extra_goals) | set(pb.extra_goals)),
        disabled_goals=sorted(set(pa.disabled_goals) & set(pb.disabled_goals)),
    )
    seen: set[str] = set()
    rules: list[str] = []
    for rule in parent_a.rules + parent_b.rules:
        if rule not in seen:
            seen.add(rule)
            rules.append(rule)
    child.rules = rules[:10]
    child.system_prompt = parent_a.system_prompt
    return child


def heuristic_mutate(
    genome: GenomeConfig,
    failure_traces: list[str],
) -> tuple[GenomeConfig, str]:
    """One skill edit from failure text. Used for tests and LLM fallback."""
    text = "\n".join(failure_traces).lower()
    child = genome.model_copy(deep=True)
    p = child.planner

    def enable(goal: str) -> None:
        if goal in ALLOWED_EXTRA_GOALS and goal not in p.extra_goals:
            p.extra_goals.append(goal)

    if "kite" in text and "died" in text:
        p.kite_combat = False
        p.safe_vitals = min(7, p.safe_vitals + 2)
        p.retreat_at_night = True
        enable("retreat_to_camp")
        return child, "raise safe vitals / disable kite after kite death"
    if "craft_iron" in text or "table+furnace" in text or "make_iron_pickaxe" in text:
        p.pair_stations = True
        p.relocate_workstation = True
        enable("craft_iron")
        return child, "keep pairing on after craft_iron fail"
    if "sapling" in text or "place_plant" in text or "farm" in text:
        enable("farm_plants")
        return child, "enable farm_plants after plant-path fail"
    if "died" in text and "attack" in text:
        p.prefer_sword = True
        p.attack_unarmed = False
        if "attack" not in p.disabled_goals:
            p.health_flee = max(p.health_flee, 3)
        return child, "prefer sword and flee earlier after combat death"
    if "drink" in text and ("died" in text or "health=0" in text or "health=1" in text):
        p.drink_threshold = min(6, p.drink_threshold + 2)
        return child, "drink earlier after thirst-related death"
    if "energy" in text or "night" in text:
        p.sleep_at_night = True
        p.energy_sleep = min(4, p.energy_sleep + 1)
        return child, "sleep sooner / at night after fatigue failure"
    if "no tree" in text or "collect_wood" in text and "no path" in text:
        p.wood_for_table = max(2, p.wood_for_table)
        return child, "keep wood target; explore more before placing table"
    if "monster" in text and "sleep" in text:
        p.sleep_at_night = False
        return child, "avoid sleeping when monsters interrupt"

    p.prefer_sword = not p.prefer_sword
    return child, "toggle prefer_sword (no specific failure matched)"


def llm_crossover(parent_a: GenomeConfig, parent_b: GenomeConfig) -> GenomeConfig:
    """Merge two parent genomes into a child genome using an LLM."""
    prompt = f"""You are an evolution operator. Merge two Crafter harnesses into one child.

The agent already has scripted macros (walk, face, do, place, craft). You evolve
PLANNER SKILLS — thresholds and flags — not slogans.

## Parent A
Rules: {json.dumps(parent_a.rules)}
Planner: {json.dumps(parent_a.planner.model_dump())}

## Parent B
Rules: {json.dumps(parent_b.rules)}
Planner: {json.dumps(parent_b.planner.model_dump())}

## Task
Combine complementary skills. You MAY invent a new planner flag combination if
it could unlock a deeper tech-tree tier (e.g. relocate_workstation=true when
iron crafting failed because table and furnace were far apart).

Respond with ONLY a JSON object:
{{
  "system_prompt": "...",
  "rules": ["optional short hints"],
  "planner": {{
    "energy_sleep": 2,
    "drink_threshold": 2,
    "food_threshold": 2,
    "health_flee": 2,
    "prefer_sword": false,
    "attack_unarmed": false,
    "relocate_workstation": false,
    "sleep_at_night": false,
    "wood_for_table": 2,
    "stone_for_furnace": 4,
    "disabled_goals": []
  }},
  "memory_policy": {{"type": "notes", "max_items": 20}},
  "context_budget": {{"history_steps": 10}}
}}"""

    resp = chat(
        messages=[{"role": "user", "content": prompt}],
        model=settings.meta_model,
        max_tokens=700,
        temperature=0.7,
    )
    parsed = _parse_genome_config(resp)
    if parsed.rules or parsed.planner != PlannerPolicy():
        return parsed
    return heuristic_crossover(parent_a, parent_b)


def llm_mutate(
    genome: GenomeConfig,
    failure_traces: list[str],
    tried_edits: list[str] | None = None,
) -> tuple[GenomeConfig, str]:
    """Mutate one planner skill based on failure traces."""
    traces_text = "\n---\n".join(failure_traces[:3])
    tried_text = ""
    if tried_edits:
        tried_text = "\n\n## Already Tried (DO NOT repeat these)\n" + "\n".join(
            f"- {e}" for e in tried_edits[-10:]
        )

    prompt = f"""You are an evolution operator. Improve a Crafter harness from failure traces.

The body already walks, faces tiles, and crafts. Change ONE planner skill.

## Current
Rules: {json.dumps(genome.rules)}
Planner: {json.dumps(genome.planner.model_dump())}

## Failure Traces
{traces_text}
{tried_text}

## Task
Make ONE targeted edit:
- Flip or adjust a planner field (safe_vitals, retreat_at_night, kite_combat, pair_stations).
- You MAY add a goal name only from this allow-list: retreat_to_camp, craft_iron, farm_plants.
- Do not invent new Python or new goal names.

Respond with ONLY a JSON object:
{{
  "config": {{
    "system_prompt": "...",
    "rules": ["..."],
    "planner": {{...}},
    "memory_policy": {{"type": "notes", "max_items": 20}},
    "context_budget": {{"history_steps": 10}}
  }},
  "mutation_note": "one-line description of the skill you changed and why"
}}"""

    resp = chat(
        messages=[{"role": "user", "content": prompt}],
        model=settings.meta_model,
        max_tokens=800,
        temperature=0.8,
    )

    try:
        match = re.search(r"\{[\s\S]+\}", resp)
        if match:
            data = json.loads(match.group())
            config = GenomeConfig(**data.get("config", data))
            note = data.get("mutation_note", "LLM mutation")
            return config, note
    except (json.JSONDecodeError, KeyError, TypeError):
        pass

    return heuristic_mutate(genome, failure_traces)


def _parse_genome_config(raw: str) -> GenomeConfig:
    """Parse LLM output into a GenomeConfig, with fallback."""
    try:
        match = re.search(r"\{[\s\S]+\}", raw)
        if match:
            data = json.loads(match.group())
            return GenomeConfig(**data)
    except (json.JSONDecodeError, KeyError, TypeError):
        pass
    return GenomeConfig()
