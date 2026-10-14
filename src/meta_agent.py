"""Meta-agent: uses an LLM to perform crossover and trace-driven mutation."""

from __future__ import annotations

import json

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from src.genome import GenomeConfig
from src.llm import chat
from src.settings import settings


class _CrossoverGenomeConfig(GenomeConfig):
    """Genome bounds accepted from a crossover model response."""

    rules: list[str] = Field(min_length=3, max_length=8)


class _MutationGenomeConfig(GenomeConfig):
    """Genome bounds accepted from a mutation model response."""

    rules: list[str] = Field(min_length=2, max_length=10)


class _CrossoverOutput(BaseModel):
    """Complete structured response required from crossover inference."""

    model_config = ConfigDict(extra="forbid", strict=True)

    config: _CrossoverGenomeConfig
    crossover_note: str = Field(min_length=1, max_length=500)


class _MutationOutput(BaseModel):
    """Complete structured response required from mutation inference."""

    model_config = ConfigDict(extra="forbid", strict=True)

    config: _MutationGenomeConfig
    mutation_note: str = Field(min_length=1, max_length=500)


def _validation_summary(error: ValidationError) -> str:
    """Return a compact correction reason without echoing model output."""
    first = error.errors(include_url=False)[0]
    if first["type"] == "json_invalid":
        return "no JSON object found"
    location = ".".join(str(item) for item in first["loc"]) or "response"
    return f"invalid schema at {location} ({first['type']})"


def llm_crossover(
    parent_a: GenomeConfig, parent_b: GenomeConfig, *, max_attempts: int = 3
) -> tuple[GenomeConfig, str]:
    """Merge two parent genomes, retrying invalid structured responses."""
    if max_attempts < 1:
        raise ValueError("max_attempts must be at least 1")

    output_schema = json.dumps(
        _CrossoverOutput.model_json_schema(), separators=(",", ":")
    )
    base_prompt = f"""You are an evolution operator. Merge the best traits of two parent agent configurations into one child.

## Parent A
{parent_a.model_dump_json()}

## Parent B
{parent_b.model_dump_json()}

## Task
Produce a CHILD configuration that combines the strongest elements of both parents.
- Merge complementary rules (e.g., A's crafting rules + B's survival rules).
- If both have conflicting rules, prefer the one from the higher-fitness parent (A).
- Keep the total rule count between 3 and 8.
- You may slightly rephrase rules for clarity but don't invent new strategies.

Respond with ONLY one JSON object matching this JSON Schema:
{output_schema}"""

    failures: list[str] = []
    for attempt in range(1, max_attempts + 1):
        correction = ""
        if failures:
            correction = (
                "\n\n## Correction\n"
                f"Your previous response was invalid: {failures[-1]}. "
                "Return only one complete JSON object matching the required schema."
            )
        resp = chat(
            messages=[{"role": "user", "content": base_prompt + correction}],
            model=settings.meta_model,
            max_tokens=600,
            temperature=0.7,
        )
        try:
            output = _CrossoverOutput.model_validate_json(resp)
        except ValidationError as error:
            failures.append(_validation_summary(error))
            continue

        config = GenomeConfig.model_validate(output.config.model_dump())
        note = output.crossover_note
        if attempt > 1:
            note = f"{note} Crossover succeeded on attempt {attempt}."
        return config, note

    failure_summary = "; ".join(failures)
    return (
        parent_a.model_copy(deep=True),
        (
            f"Crossover failed after {max_attempts} attempts ({failure_summary}); "
            "retained parent A"
        ),
    )


def llm_mutate(
    genome: GenomeConfig,
    failure_traces: list[str],
    tried_edits: list[str] | None = None,
) -> tuple[GenomeConfig, str]:
    """Mutate a genome based on failure traces. Returns (new_config, mutation_note)."""
    traces_text = "\n---\n".join(failure_traces[:3])
    tried_text = ""
    if tried_edits:
        tried_text = "\n\n## Already Tried (DO NOT repeat these)\n" + "\n".join(
            f"- {e}" for e in tried_edits[-10:]
        )

    output_schema = json.dumps(
        _MutationOutput.model_json_schema(), separators=(",", ":")
    )
    prompt = f"""You are an evolution operator. Improve an agent's configuration based on its failure traces.

## Current Configuration
{genome.model_dump_json()}

## Failure Traces
{traces_text}
{tried_text}

## Task
Analyze the failures and make ONE targeted improvement:
- Add, remove, or modify a rule based on what went wrong.
- Optionally adjust memory_policy or context_budget.
- Do NOT change the system_prompt drastically.
- Keep total rules between 2 and 10.
- Be specific: "Collect wood before anything else" is better than "Be efficient".

Respond with ONLY one JSON object matching this JSON Schema:
{output_schema}"""

    resp = chat(
        messages=[{"role": "user", "content": prompt}],
        model=settings.meta_model,
        max_tokens=800,
        temperature=0.8,
    )

    try:
        output = _MutationOutput.model_validate_json(resp)
    except ValidationError:
        return genome, "mutation failed — kept crossover configuration"

    config = GenomeConfig.model_validate(output.config.model_dump())
    if config.content_hash() == genome.content_hash():
        return genome, "mutation failed — no configuration change"
    return config, output.mutation_note
