"""Compact mutable state retained between skill decisions in one episode."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field


class ProgressPhase(StrEnum):
    """Broad progression stage used to stabilize long-horizon decisions."""

    SURVIVAL = "survival"
    BASIC_RESOURCES = "basic_resources"
    WOOD_TOOLS = "wood_tools"
    STONE_TOOLS = "stone_tools"
    IRON_TOOLS = "iron_tools"
    EXPLORATION = "exploration"


class KnownLocation(BaseModel):
    """A bounded, explicitly staleable relative landmark estimate."""

    direction: str
    estimated_distance: int = Field(ge=0, le=64)
    last_seen_step: int = Field(ge=0)


class SkillFailure(BaseModel):
    """Recent failure evidence for one parameterized skill."""

    count: int = Field(default=1, ge=1, le=20)
    last_failure: str
    last_attempt_step: int = Field(ge=0)


class ExecutionState(BaseModel):
    """Only episode state that cannot be reconstructed from the latest observation."""

    phase: ProgressPhase = ProgressPhase.BASIC_RESOURCES
    goal: str | None = Field(default=None, max_length=160)
    subgoal: str | None = Field(default=None, max_length=160)
    known_locations: dict[str, KnownLocation] = Field(default_factory=dict)
    failed_skills: dict[str, SkillFailure] = Field(default_factory=dict)
    no_progress_steps: int = Field(default=0, ge=0, le=20)


class StatePatch(BaseModel):
    """Validated partial update proposed by the gameplay model."""

    phase: ProgressPhase | None = None
    goal: str | None = Field(default=None, max_length=160)
    subgoal: str | None = Field(default=None, max_length=160)
    known_locations: dict[str, KnownLocation] | None = None


def apply_state_patch(state: ExecutionState, patch: StatePatch) -> ExecutionState:
    """Return a new execution state while keeping runtime-owned counters authoritative."""
    updates = patch.model_dump(exclude_none=True)
    return state.model_copy(update=updates, deep=True)
