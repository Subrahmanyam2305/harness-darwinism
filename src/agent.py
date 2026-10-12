"""LLM agent that plays Crafter using a genome's harness config."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import imageio.v3 as iio
import numpy as np
from PIL import Image

from src.config import settings
from src.crafter_env import ACTIONS, ACHIEVEMENT_PREREQS, RECIPES, CrafterTextEnv
from src.genome import Genome
from src.llm import chat

REPLAY_DIR = settings.project_root / "data" / "replays"
REPLAY_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class EpisodeTrace:
    """Record of one episode for the meta-agent to learn from."""
    genome_id: str
    seed: int
    score: float
    achievements: dict[str, bool]
    steps: int
    history: list[dict] = field(default_factory=list)
    failure_summary: str = ""
    gif_path: str = ""


# Achievement tech tree in recommended order
ACHIEVEMENT_ORDER = [
    "collect_wood", "place_table", "make_wood_pickaxe", "make_wood_sword",
    "collect_sapling", "place_plant", "collect_drink",
    "eat_cow", "eat_plant",
    "collect_stone", "place_stone", "make_stone_pickaxe", "make_stone_sword",
    "place_furnace",
    "collect_coal", "collect_iron", "make_iron_pickaxe", "make_iron_sword",
    "defeat_zombie", "defeat_skeleton",
    "collect_diamond", "wake_up",
]


def _build_system_prompt(genome: Genome) -> str:
    """Construct a goal-directed system prompt from the genome's config."""
    cfg = genome.genome
    parts = []
    parts.append(cfg.system_prompt)
    parts.append("")

    parts.append("## Achievement Tech Tree (in recommended order)")
    for ach in ACHIEVEMENT_ORDER:
        prereq = ACHIEVEMENT_PREREQS.get(ach, "")
        parts.append(f"- {ach}: {prereq}")
    parts.append("")

    parts.append("## Crafting Recipes")
    for action, recipe in RECIPES.items():
        needs = ", ".join(f"{v} {k}" for k, v in recipe.items())
        parts.append(f"- {action}: needs {needs}")
    parts.append("")

    parts.append("## Key Mechanics")
    parts.append("- 'do' is the interaction action: chop trees, mine stone, drink water, attack enemies, eat food")
    parts.append("- You must be ADJACENT (dist 1) to something to 'do' it")
    parts.append("- Crafting actions (make_*) need a table nearby; iron items also need a furnace")
    parts.append("- Night is dangerous without a weapon — sleep or hide")
    parts.append("- Collect resources in order: wood → stone → coal → iron → diamond")
    parts.append("- Move directions: move_up=north, move_down=south, move_left=west, move_right=east")
    parts.append("")

    if cfg.rules:
        parts.append("## Strategy Rules (FOLLOW THESE)")
        for i, rule in enumerate(cfg.rules, 1):
            parts.append(f"{i}. {rule}")
        parts.append("")

    parts.append("## Available Actions")
    parts.append(", ".join(cfg.tools))
    parts.append("")

    parts.append("## Response Format")
    parts.append('Respond with ONLY a JSON object:')
    parts.append('{"goal": "current achievement target", "thought": "brief plan", "action": "action_name"}')
    parts.append("The action must be one of the available actions listed above.")

    return "\n".join(parts)


def _parse_action(response: str, allowed_actions: list[str] | None = None) -> str:
    """Extract action from LLM response, validating against allowed actions."""
    valid = set(allowed_actions) if allowed_actions else set(ACTIONS)
    try:
        match = re.search(r'\{[^}]+\}', response)
        if match:
            data = json.loads(match.group())
            action = data.get("action", "noop")
            if action in valid:
                return action
    except (json.JSONDecodeError, KeyError):
        pass

    for action in ACTIONS:
        if action in response.lower() and action in valid:
            return action
    return "noop"


def _extract_thought(response: str) -> dict:
    """Extract goal and thought from LLM JSON response."""
    try:
        match = re.search(r'\{[^}]+\}', response)
        if match:
            data = json.loads(match.group())
            return {
                "goal": data.get("goal", ""),
                "thought": data.get("thought", ""),
            }
    except (json.JSONDecodeError, KeyError):
        pass
    return {"goal": "", "thought": ""}


def play_episode(genome: Genome, seed: int, max_steps: int = 200) -> EpisodeTrace:
    """Run one Crafter episode with the given genome's harness."""
    env = CrafterTextEnv(seed=seed, max_steps=max_steps)
    system_prompt = _build_system_prompt(genome)
    allowed_actions = genome.genome.tools

    messages: list[dict] = []
    history: list[dict] = []
    notes: list[str] = []
    memory_type = genome.genome.memory_policy.type

    obs = env.reset()
    prev_action = None
    prev_reward = 0.0
    stuck_count = 0  # track consecutive noop/no-reward actions

    while True:
        user_parts = [f"Observation:\n{obs}"]

        # Memory: implement all three modes
        if memory_type == "notes" and notes:
            max_notes = genome.genome.memory_policy.max_items
            user_parts.append("\nYour notes:\n" + "\n".join(f"- {n}" for n in notes[-max_notes:]))
        elif memory_type == "summary" and notes:
            max_notes = genome.genome.memory_policy.max_items
            recent = notes[-max_notes:]
            user_parts.append(f"\nMemory summary ({len(recent)} events): " + "; ".join(recent))

        if prev_action is not None:
            outcome = f"reward +{prev_reward}" if prev_reward > 0 else "no reward"
            user_parts.append(f"\nPrevious action: {prev_action} → {outcome}")

        # Stuck detection warning
        if stuck_count >= 5:
            user_parts.append(f"\n⚠ WARNING: No progress for {stuck_count} steps! Try a different approach.")

        user_parts.append(f"\nStep {env.step_count + 1}/{max_steps}. Choose your action.")

        user_msg = "\n".join(user_parts)

        # Trim history to context budget
        ctx = genome.genome.context_budget.history_steps
        trimmed = messages[max(0, len(messages) - ctx * 2):]
        trimmed.append({"role": "user", "content": user_msg})

        try:
            reply = chat(
                messages=trimmed,
                system=system_prompt,
                model=settings.agent_model,
                max_tokens=200,
                temperature=0.3,
            )
        except Exception as e:
            reply = '{"action": "noop"}'
            history.append({"step": env.step_count, "error": str(e)})

        action = _parse_action(reply, allowed_actions)
        thought_data = _extract_thought(reply)

        messages.append({"role": "user", "content": user_msg})
        messages.append({"role": "assistant", "content": reply})

        obs, reward, done, info = env.step(action)

        # Track stuck state
        if reward > 0:
            stuck_count = 0
        else:
            stuck_count += 1

        # Structured trace entry
        history.append({
            "step": env.step_count,
            "action": action,
            "reward": reward,
            "obs_snippet": obs[:400],
            "goal": thought_data.get("goal", ""),
            "thought": thought_data.get("thought", ""),
        })

        prev_action = action
        prev_reward = reward

        if reward > 0:
            notes.append(f"Step {env.step_count}: +{reward} from {action}")
        if "health=1" in obs or "health=2" in obs:
            notes.append(f"Step {env.step_count}: LOW HEALTH!")
        if "NIGHT" in obs and env.step_count > 0:
            if "NIGHT" not in (notes[-1] if notes else ""):
                notes.append(f"Step {env.step_count}: Night time")

        if done:
            break

    score = env.get_score()
    achievements = env.get_achievements()
    failure_summary = _summarize_failure(history, achievements, env.step_count, max_steps)

    # Save GIF replay
    gif_path = ""
    if env.frames:
        gif_name = f"{genome.id}_s{seed}.gif"
        gif_full = REPLAY_DIR / gif_name
        upscaled = [
            np.array(Image.fromarray(f).resize((256, 256), Image.NEAREST))
            for f in env.frames
        ]
        iio.imwrite(str(gif_full), upscaled, duration=200, loop=0)
        gif_path = gif_name

    return EpisodeTrace(
        genome_id=genome.id,
        seed=seed,
        score=score,
        achievements=achievements,
        steps=env.step_count,
        history=history,
        failure_summary=failure_summary,
        gif_path=gif_path,
    )


def _summarize_failure(
    history: list[dict],
    achievements: dict[str, bool],
    steps: int,
    max_steps: int,
) -> str:
    """Detailed failure summary for the meta-agent to learn from."""
    unlocked = [k for k, v in achievements.items() if v]
    missed = [k for k, v in achievements.items() if not v]

    lines = [f"Survived {steps}/{max_steps} steps."]
    lines.append(f"Unlocked ({len(unlocked)}): {', '.join(unlocked) if unlocked else 'none'}")
    lines.append(f"Missed: {', '.join(missed[:8])}{'...' if len(missed) > 8 else ''}")

    if steps < max_steps:
        death_step = history[-1] if history else {}
        lines.append(f"Died at step {steps}. Last action: {death_step.get('action', '?')}")

    # Action distribution
    action_counts: dict[str, int] = {}
    do_count = 0
    move_count = 0
    for h in history:
        a = h.get("action", "?")
        action_counts[a] = action_counts.get(a, 0) + 1
        if a == "do":
            do_count += 1
        elif a.startswith("move_"):
            move_count += 1

    total = len(history) or 1
    lines.append(f"Action distribution: do={do_count}/{total} ({do_count/total:.0%}), "
                 f"move={move_count}/{total} ({move_count/total:.0%})")

    top_actions = sorted(action_counts.items(), key=lambda x: -x[1])[:7]
    lines.append(f"Actions: {', '.join(f'{a}({n})' for a, n in top_actions)}")

    # Goal tracking from thoughts
    goals = [h.get("goal", "") for h in history if h.get("goal")]
    if goals:
        goal_counts: dict[str, int] = {}
        for g in goals:
            goal_counts[g] = goal_counts.get(g, 0) + 1
        top_goals = sorted(goal_counts.items(), key=lambda x: -x[1])[:5]
        lines.append(f"Goals pursued: {', '.join(f'{g}({n})' for g, n in top_goals)}")

    # Longest stuck streak
    max_stuck = 0
    cur_stuck = 0
    for h in history:
        if h.get("reward", 0) > 0:
            cur_stuck = 0
        else:
            cur_stuck += 1
            max_stuck = max(max_stuck, cur_stuck)
    lines.append(f"Longest no-reward streak: {max_stuck} steps")

    return "\n".join(lines)
