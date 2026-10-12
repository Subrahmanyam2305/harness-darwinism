"""Meta-agent: uses an LLM to perform crossover and trace-driven mutation."""

from __future__ import annotations

import json
import re

from src.config import settings
from src.genome import GenomeConfig
from src.llm import chat


def llm_crossover(parent_a: GenomeConfig, parent_b: GenomeConfig) -> GenomeConfig:
    """Merge two parent genomes into a child genome using an LLM."""
    prompt = f"""You are an evolution operator for a Crafter game agent. Merge the best traits of two parent configurations.

## Context: What makes a good Crafter agent
- Uses 'do' action FREQUENTLY (30-50% of all actions) to interact with resources
- Follows a tech tree: wood → table → pickaxe → sword → stone → furnace → iron tools
- Maintains survival vitals (food, drink, energy) above danger levels
- Has clear milestone-based goals, not vague instructions
- Provides specific thresholds (e.g., "if food < 4" not "when food is low")

## Parent A
System prompt: {parent_a.system_prompt}
Rules: {json.dumps(parent_a.rules)}
Memory: {parent_a.memory_policy.model_dump()}
Context: {parent_a.context_budget.model_dump()}

## Parent B
System prompt: {parent_b.system_prompt}
Rules: {json.dumps(parent_b.rules)}
Memory: {parent_b.memory_policy.model_dump()}
Context: {parent_b.context_budget.model_dump()}

## Task
Produce a CHILD configuration:
- Merge complementary rules. Keep rules that are SPECIFIC and ACTIONABLE.
- Remove vague/redundant rules. 
- Keep 5-10 rules total.
- Ensure the rules cover: resource collection, crafting progression, survival, and 'do' usage.

Respond with ONLY a JSON object:
{{
  "system_prompt": "...",
  "rules": ["...", "..."],
  "memory_policy": {{"type": "notes"|"summary"|"none", "max_items": N}},
  "context_budget": {{"history_steps": N}}
}}"""

    resp = chat(
        messages=[{"role": "user", "content": prompt}],
        model=settings.meta_model,
        max_tokens=800,
        temperature=0.7,
    )
    return _parse_genome_config(resp)


def llm_mutate(
    genome: GenomeConfig,
    failure_traces: list[str],
    tried_edits: list[str] | None = None,
) -> tuple[GenomeConfig, str]:
    """Mutate a genome based on failure traces. Returns (new_config, mutation_note)."""
    traces_text = "\n---\n".join(failure_traces[:3])
    tried_text = ""
    if tried_edits:
        tried_text = f"\n\n## Already Tried (DO NOT repeat these)\n" + "\n".join(f"- {e}" for e in tried_edits[-10:])

    prompt = f"""You are an evolution operator for a Crafter game agent. Improve the configuration based on failure analysis.

## Key Crafter insights
- 'do' is the ONLY interaction action — chopping, mining, drinking, eating, fighting all use 'do'
- Agent must be ADJACENT (dist 1) to interact. Moving to the right spot then using 'do' is the core loop.
- Achievement score = exp(mean(log(1 + 100 × rate))) − 1 across 22 achievements
- Each new achievement unlocked improves score significantly
- Common failures: too much wandering, not enough 'do', not crafting tools, dying at night

## Current Configuration
System prompt: {genome.system_prompt}
Rules: {json.dumps(genome.rules)}
Memory: {genome.memory_policy.model_dump()}
Context: {genome.context_budget.model_dump()}

## Failure Traces
{traces_text}
{tried_text}

## Task
Analyze failures carefully. Common diagnoses:
1. Low 'do' ratio → add explicit "use 'do' every 2 steps" type rules
2. No crafting → add step-by-step crafting sequence
3. Dying → add survival thresholds (food < 4, drink < 4)
4. Stuck in one area → add exploration rules
5. Not progressing through tech tree → add milestone checks

Make ONE targeted improvement. State a hypothesis for WHY it should help.
Keep total rules between 5 and 12.

Respond with ONLY a JSON object:
{{
  "config": {{
    "system_prompt": "...",
    "rules": ["...", "..."],
    "memory_policy": {{"type": "notes"|"summary"|"none", "max_items": N}},
    "context_budget": {{"history_steps": N}}
  }},
  "mutation_note": "HYPOTHESIS: [what should change]. CHANGE: [what was modified]"
}}"""

    resp = chat(
        messages=[{"role": "user", "content": prompt}],
        model=settings.meta_model,
        max_tokens=1000,
        temperature=0.8,
    )

    try:
        match = re.search(r'\{[\s\S]+\}', resp)
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
