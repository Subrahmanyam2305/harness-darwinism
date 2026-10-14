"""Generate GIF replays for existing genomes without re-running LLM calls.

This replays the game using the recorded action history to capture frames.
"""

from __future__ import annotations

import imageio.v3 as iio
import numpy as np
from PIL import Image
from rich.console import Console

from src.crafter_env import ACTION_TO_IDX
from src.db import get_all_genomes, get_db, get_runs_for_genome
from src.settings import settings

console = Console()
REPLAY_DIR = settings.project_root / "data" / "replays"
REPLAY_DIR.mkdir(parents=True, exist_ok=True)


def generate_gif_from_history(
    genome_id: str, seed: int, actions: list[str], max_steps: int = 200
) -> str:
    """Replay actions in Crafter and save a GIF."""
    import crafter

    env = crafter.Env(size=(64, 64), seed=seed)
    obs = env.reset()
    frames = [obs.copy()]

    for action_name in actions:
        idx = ACTION_TO_IDX.get(action_name, 0)
        obs, _, done, _ = env.step(idx)
        frames.append(obs.copy())
        if done:
            break

    upscaled = [
        np.array(Image.fromarray(f).resize((256, 256), Image.NEAREST)) for f in frames
    ]
    gif_name = f"{genome_id}_s{seed}.gif"
    gif_path = REPLAY_DIR / gif_name
    iio.imwrite(str(gif_path), upscaled, duration=200, loop=0)
    return gif_name


def main():
    db = get_db()
    genomes = get_all_genomes(db)

    if not genomes:
        console.print("[red]No genomes found in DB[/red]")
        return

    total = 0
    for genome in genomes:
        runs = get_runs_for_genome(db, genome.id)
        for run in runs:
            history = run.get("history", [])
            seed = run.get("seed")
            if not history or not seed:
                continue

            actions = [h.get("action", "noop") for h in history]
            gif_name = generate_gif_from_history(genome.id, seed, actions)
            console.print(f"  [green]✓[/green] {gif_name} ({len(actions)} steps)")
            total += 1

    console.print(
        f"\n[bold green]Generated {total} GIF replays in data/replays/[/bold green]"
    )


if __name__ == "__main__":
    main()
