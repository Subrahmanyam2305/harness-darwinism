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
from src.crafter_env import ACTIONS, CrafterTextEnv
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


def _build_system_prompt(genome: Genome) -> str:
    """Construct the full system prompt from the genome's harness config."""
    cfg = genome.genome
    parts = [cfg.system_prompt, ""]

    if cfg.rules:
        parts.append("## Strategy Rules")
        for i, rule in enumerate(cfg.rules, 1):
            parts.append(f"{i}. {rule}")
        parts.append("")

    parts.append("## Available Actions")
    parts.append(", ".join(cfg.tools))
    parts.append("")

    parts.append("## Response Format")
    parts.append('Respond with ONLY a JSON object: {"thought": "brief reasoning", "action": "action_name"}')
    parts.append("The action must be one of the available actions listed above.")

    return "\n".join(parts)


def _parse_action(response: str) -> str:
    """Extract action from LLM response, with fallback."""
    try:
        match = re.search(r'\{[^}]+\}', response)
        if match:
            data = json.loads(match.group())
            action = data.get("action", "noop")
            if action in ACTIONS:
                return action
    except (json.JSONDecodeError, KeyError):
        pass

    for action in ACTIONS:
        if action in response.lower():
            return action
    return "noop"


def play_episode(genome: Genome, seed: int, max_steps: int = 200) -> EpisodeTrace:
    """Run one Crafter episode with the given genome's harness."""
    env = CrafterTextEnv(seed=seed, max_steps=max_steps)
    system_prompt = _build_system_prompt(genome)

    messages: list[dict] = []
    history: list[dict] = []
    notes: list[str] = []

    obs = env.reset()

    while True:
        user_parts = [f"Observation:\n{obs}"]
        if notes:
            max_notes = genome.genome.memory_policy.max_items
            user_parts.append("\nYour notes:\n" + "\n".join(f"- {n}" for n in notes[-max_notes:]))
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
                max_tokens=150,
                temperature=0.3,
            )
        except Exception as e:
            reply = '{"action": "noop"}'
            history.append({"step": env.step_count, "error": str(e)})

        action = _parse_action(reply)

        messages.append({"role": "user", "content": user_msg})
        messages.append({"role": "assistant", "content": reply})

        obs, reward, done, info = env.step(action)

        history.append({
            "step": env.step_count,
            "action": action,
            "reward": reward,
            "obs_snippet": obs[:200],
        })

        if reward > 0:
            notes.append(f"Step {env.step_count}: got reward {reward} from {action}")
        if "health=1" in obs or "health=2" in obs:
            notes.append(f"Step {env.step_count}: critically low health!")

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
    """Quick text summary of what went wrong."""
    unlocked = [k for k, v in achievements.items() if v]
    missed = [k for k, v in achievements.items() if not v]

    lines = [f"Survived {steps}/{max_steps} steps."]
    lines.append(f"Unlocked: {', '.join(unlocked) if unlocked else 'none'}")
    lines.append(f"Missed: {', '.join(missed[:5])}{'...' if len(missed) > 5 else ''}")

    if steps < max_steps:
        death_step = history[-1] if history else {}
        lines.append(f"Died at step {steps}. Last action: {death_step.get('action', '?')}")

    action_counts: dict[str, int] = {}
    for h in history:
        a = h.get("action", "?")
        action_counts[a] = action_counts.get(a, 0) + 1
    top_actions = sorted(action_counts.items(), key=lambda x: -x[1])[:5]
    lines.append(f"Most used actions: {', '.join(f'{a}({n})' for a, n in top_actions)}")

    return "\n".join(lines)
