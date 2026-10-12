"""Genome: the evolvable harness configuration."""

from __future__ import annotations

import hashlib
import json
from typing import Literal

from pydantic import BaseModel, Field, field_validator

ALLOWED_EXTRA_GOALS = frozenset({"retreat_to_camp", "craft_iron", "farm_plants"})


class MemoryPolicy(BaseModel):
    type: Literal["none", "notes", "summary"] = "notes"
    max_items: int = 20


class ContextBudget(BaseModel):
    history_steps: int = 10


class PlannerPolicy(BaseModel):
    """Evolvable skill knobs the scripted planner actually reads."""

    energy_sleep: int = 2
    drink_threshold: int = 2
    food_threshold: int = 2
    health_flee: int = 2
    prefer_sword: bool = False
    attack_unarmed: bool = False
    relocate_workstation: bool = False
    pair_stations: bool = True
    kite_combat: bool = False
    sleep_at_night: bool = False
    retreat_at_night: bool = True
    safe_vitals: int = 5
    wood_for_table: int = 2
    stone_for_furnace: int = 4
    extra_goals: list[str] = Field(
        default_factory=lambda: ["retreat_to_camp", "craft_iron", "farm_plants"]
    )
    disabled_goals: list[str] = Field(default_factory=list)

    @field_validator("extra_goals")
    @classmethod
    def _allow_listed_macros(cls, value: list[str]) -> list[str]:
        return [g for g in value if g in ALLOWED_EXTRA_GOALS]


class GenomeConfig(BaseModel):
    """The mutable part of a harness — this is what evolves."""

    system_prompt: str = "You are a Crafter survival agent. Stay alive and unlock achievements."
    rules: list[str] = Field(default_factory=list)
    tools: list[str] = Field(
        default_factory=lambda: [
            "noop", "move_left", "move_right", "move_up", "move_down",
            "do", "sleep", "place_stone", "place_table", "place_furnace",
            "place_plant", "make_wood_pickaxe", "make_stone_pickaxe",
            "make_iron_pickaxe", "make_wood_sword", "make_stone_sword",
            "make_iron_sword",
        ]
    )
    memory_policy: MemoryPolicy = Field(default_factory=MemoryPolicy)
    context_budget: ContextBudget = Field(default_factory=ContextBudget)
    planner: PlannerPolicy = Field(default_factory=PlannerPolicy)

    def content_hash(self) -> str:
        blob = json.dumps(self.model_dump(), sort_keys=True)
        return hashlib.sha256(blob.encode()).hexdigest()[:16]


class Fitness(BaseModel):
    score: float = 0.0
    hafner: float = 0.0
    unlocks: float = 0.0
    iron_rate: float = 0.0
    tech_depth: float = 0.0
    achievements: dict[str, float] = Field(default_factory=dict)
    mean_steps: float = 0.0
    cost_usd: float = 0.0

    def rank_key(self) -> tuple[float, float, float, float, float]:
        """Iron and longevity beat wood-tier death spirals."""
        return (self.iron_rate, self.unlocks, self.mean_steps, self.tech_depth, self.score)


class Genome(BaseModel):
    """One harness variant = one individual in the population."""

    id: str
    generation: int = 0
    parents: list[str] = Field(default_factory=list)
    status: Literal["alive", "archived", "champion"] = "alive"
    genome: GenomeConfig = Field(default_factory=GenomeConfig)
    fitness: Fitness = Field(default_factory=Fitness)
    mutation_note: str = ""
    arm: str = "full"  # ablation arm label
    n_children: int = 0

    def to_doc(self) -> dict:
        d = self.model_dump()
        d["_id"] = d.pop("id")
        return d

    @classmethod
    def from_doc(cls, doc: dict) -> Genome:
        doc["id"] = doc.pop("_id")
        return cls(**doc)
