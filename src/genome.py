"""Genome: the evolvable harness configuration."""

from __future__ import annotations

import hashlib
import json
from typing import Literal

from pydantic import BaseModel, Field


class MemoryPolicy(BaseModel):
    type: Literal["none", "notes", "summary"] = "notes"
    max_items: int = 20


class ContextBudget(BaseModel):
    history_steps: int = 10


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

    def content_hash(self) -> str:
        blob = json.dumps(self.model_dump(), sort_keys=True)
        return hashlib.sha256(blob.encode()).hexdigest()[:16]


class Fitness(BaseModel):
    score: float = 0.0
    achievements: dict[str, float] = Field(default_factory=dict)
    mean_steps: float = 0.0
    cost_usd: float = 0.0


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

    def to_doc(self) -> dict:
        d = self.model_dump()
        d["_id"] = d.pop("id")
        return d

    @classmethod
    def from_doc(cls, doc: dict) -> Genome:
        doc["id"] = doc.pop("_id")
        return cls(**doc)
