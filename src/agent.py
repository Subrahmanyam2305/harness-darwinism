"""Hierarchical Crafter agent: planner picks a goal, macros emit primitives."""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from src.config import settings
from src.crafter_env import CrafterTextEnv
from src.genome import Genome
from src.planner import choose_goal
from src.skills import WorldMap, execute_goal


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
    llm_calls: int = 0
    frames: list[dict] = field(default_factory=list)


def _encode_rgb(rgb) -> str:
    import base64
    import io

    from PIL import Image

    image = Image.fromarray(rgb)
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def play_episode(
    genome: Genome,
    seed: int,
    max_steps: int | None = None,
    record_stride: int = 0,
) -> EpisodeTrace:
    """Run one Crafter episode with planner + scripted macros."""
    max_steps = max_steps if max_steps is not None else settings.max_steps_per_episode
    env = CrafterTextEnv(seed=seed, max_steps=max_steps)
    if record_stride:
        env.enable_recording(record_stride)
    world = WorldMap()

    history: list[dict] = []
    frames: list[dict] = []
    state = env.reset()
    world.update(state)

    def flush_frames(goal: str) -> None:
        if not record_stride:
            return
        for step, rgb in env.pop_recorded():
            frames.append({
                "step": step,
                "goal": goal,
                "vitals": dict(state.vitals),
                "image": _encode_rgb(rgb),
            })

    flush_frames("start")

    last_goal: str | None = None
    last_fail = ""
    repeats = 0
    llm_calls = 0
    max_llm = getattr(settings, "max_llm_calls_per_episode", 0)
    deadline = time.monotonic() + 90.0

    while env.step_count < max_steps:
        stuck = last_goal if repeats >= 3 else None
        use_llm = (
            stuck is not None
            and settings.planner_mode == "llm"
            and llm_calls < max_llm
        )
        if use_llm:
            llm_calls += 1
        goal = choose_goal(state, world, genome, stuck_goal=stuck if use_llm else None)
        if repeats >= 3 and goal not in {"explore", "flee", "sleep_safe"}:
            at_camp = bool(world.camp) and max(
                abs(state.x - world.camp[0]), abs(state.y - world.camp[1])
            ) <= 2
            if world.camp and not at_camp:
                goal = "retreat_to_camp"
            elif goal == "retreat_to_camp":
                goal = "sleep_safe" if state.is_night else "explore"
            else:
                goal = "explore"
            repeats = 0
        if goal == last_goal:
            repeats += 1
        else:
            repeats = 0
        last_goal = goal

        pair = True
        if genome is not None:
            pair = genome.genome.planner.pair_stations
        result = execute_goal(env, state, world, goal, pair_stations=pair)
        if result.state is not None:
            state = result.state
        last_fail = result.reason

        history.append({
            "step": env.step_count,
            "goal": goal,
            "ok": result.ok,
            "reason": result.reason,
            "primitives": result.actions,
            "n_primitives": result.steps,
            "reward": result.reward,
            "vitals": dict(state.vitals),
            "pos": state.pos,
        })

        flush_frames(goal)

        if result.done or state.vitals.get("health", 0) <= 0:
            break

        if repeats >= 6 and goal == "explore":
            # One more explore cycle already ran; stop spinning if the map is exhausted.
            if result.steps == 0:
                break
        if repeats >= 12:
            break
        if time.monotonic() > deadline:
            last_fail = (last_fail or "") + " episode wall-clock timeout"
            break

    score = env.get_score()
    achievements = env.get_achievements()
    failure_summary = _summarize_failure(
        history, achievements, env.step_count, max_steps, last_goal, last_fail, state
    )

    return EpisodeTrace(
        genome_id=genome.id,
        seed=seed,
        score=score,
        achievements=achievements,
        steps=env.step_count,
        history=history,
        failure_summary=failure_summary,
        llm_calls=llm_calls,
        frames=frames,
    )


def _summarize_failure(
    history: list[dict],
    achievements: dict[str, bool],
    steps: int,
    max_steps: int,
    last_goal: str | None,
    last_fail: str,
    state,
) -> str:
    unlocked = [k for k, v in achievements.items() if v]
    missed = [k for k, v in achievements.items() if not v]

    lines = [f"Survived {steps}/{max_steps} steps."]
    lines.append(f"Unlocked: {', '.join(unlocked) if unlocked else 'none'}")
    lines.append(f"Missed: {', '.join(missed[:8])}{'...' if len(missed) > 8 else ''}")
    if last_goal:
        lines.append(f"Last goal: {last_goal} ({last_fail})")
    lines.append(
        f"Vitals: health={state.vitals.get('health')} food={state.vitals.get('food')} "
        f"drink={state.vitals.get('drink')} energy={state.vitals.get('energy')}"
    )

    if steps < max_steps and state.vitals.get("health", 0) <= 0:
        last = history[-1] if history else {}
        lines.append(f"Died at step {steps}. Last goal: {last.get('goal', '?')}")

    goals: dict[str, int] = {}
    fails: dict[str, int] = {}
    for h in history:
        g = h.get("goal", "?")
        goals[g] = goals.get(g, 0) + 1
        if not h.get("ok"):
            fails[g] = fails.get(g, 0) + 1
    top = sorted(goals.items(), key=lambda x: -x[1])[:6]
    lines.append("Goals: " + ", ".join(f"{g}({n})" for g, n in top))
    if fails:
        top_f = sorted(fails.items(), key=lambda x: -x[1])[:4]
        lines.append("Failed goals: " + ", ".join(f"{g}({n})" for g, n in top_f))
    return "\n".join(lines)
