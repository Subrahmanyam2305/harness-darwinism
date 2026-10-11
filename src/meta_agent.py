"""Meta-agent: uses an LLM to perform crossover and trace-driven mutation."""

from __future__ import annotations

import json
import re

from openai import OpenAI

from src.config import settings
from src.genome import GenomeConfig


def llm_crossover(parent_a: GenomeConfig, parent_b: GenomeConfig) -> GenomeConfig:
    """Merge two parent genomes into a child genome using an LLM."""
    client = OpenAI(api_key=settings.openai_api_key)

    prompt = f"""You are an evolution operator. Merge the best traits of two parent agent configurations into one child.

## Parent A
System prompt: {parent_a.system_prompt}
Rules: {json.dumps(parent_a.rules)}
Memory policy: {parent_a.memory_policy.model_dump()}
Context budget: {parent_a.context_budget.model_dump()}

## Parent B
System prompt: {parent_b.system_prompt}
Rules: {json.dumps(parent_b.rules)}
Memory policy: {parent_b.memory_policy.model_dump()}
Context budget: {parent_b.context_budget.model_dump()}

## Task
Produce a CHILD configuration that combines the strongest elements of both parents.
- Merge complementary rules (e.g., A's crafting rules + B's survival rules).
- If both have conflicting rules, prefer the one from the higher-fitness parent (A).
- Keep the total rule count between 3 and 8.
- You may slightly rephrase rules for clarity but don't invent new strategies.

Respond with ONLY a JSON object:
{{
  "system_prompt": "...",
  "rules": ["...", "..."],
  "memory_policy": {{"type": "notes"|"summary"|"none", "max_items": N}},
  "context_budget": {{"history_steps": N}}
}}"""

    resp = client.chat.completions.create(
        model=settings.meta_model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=600,
        temperature=0.7,
    )

    return _parse_genome_config(resp.choices[0].message.content or "{}")


def llm_mutate(
    genome: GenomeConfig,
    failure_traces: list[str],
    tried_edits: list[str] | None = None,
) -> tuple[GenomeConfig, str]:
    """Mutate a genome based on failure traces. Returns (new_config, mutation_note)."""
    client = OpenAI(api_key=settings.openai_api_key)

    traces_text = "\n---\n".join(failure_traces[:3])
    tried_text = ""
    if tried_edits:
        tried_text = f"\n\n## Already Tried (DO NOT repeat these)\n" + "\n".join(f"- {e}" for e in tried_edits[-10:])

    prompt = f"""You are an evolution operator. Improve an agent's configuration based on its failure traces.

## Current Configuration
System prompt: {genome.system_prompt}
Rules: {json.dumps(genome.rules)}
Memory policy: {genome.memory_policy.model_dump()}
Context budget: {genome.context_budget.model_dump()}

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

Respond with ONLY a JSON object:
{{
  "config": {{
    "system_prompt": "...",
    "rules": ["...", "..."],
    "memory_policy": {{"type": "notes"|"summary"|"none", "max_items": N}},
    "context_budget": {{"history_steps": N}}
  }},
  "mutation_note": "one-line description of what you changed and why"
}}"""

    resp = client.chat.completions.create(
        model=settings.meta_model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=800,
        temperature=0.8,
    )

    raw = resp.choices[0].message.content or "{}"
    try:
        match = re.search(r'\{[\s\S]+\}', raw)
        if match:
            data = json.loads(match.group())
            config = GenomeConfig(**data.get("config", data))
            note = data.get("mutation_note", "LLM mutation")
            return config, note
    except (json.JSONDecodeError, KeyError, TypeError):
        pass

    return genome, "mutation failed — kept parent"


def _parse_genome_config(raw: str) -> GenomeConfig:
    """Parse LLM output into a GenomeConfig, with fallback."""
    try:
        match = re.search(r'\{[\s\S]+\}', raw)
        if match:
            data = json.loads(match.group())
            return GenomeConfig(**data)
    except (json.JSONDecodeError, KeyError, TypeError):
        pass
    return GenomeConfig()
