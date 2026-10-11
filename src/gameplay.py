"""Run a fixed baseline harness with live frames and saved replay evidence."""

from __future__ import annotations

import threading
import time
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from src.agent import EpisodeFrame, EpisodeTrace, play_episode
from src.db import get_db, get_run, insert_episode, upsert_genome
from src.gameplay_provider import BedrockProvider, GameplayProvider, ScriptedProvider
from src.genome import Fitness, Genome, GenomeConfig
from src.recording import EpisodeRecorder


def baseline_genome() -> Genome:
    """Return the frozen generation-zero harness used for baseline evidence."""
    return Genome(
        id="fixed-baseline",
        genome=GenomeConfig(
            rules=[
                "Collect at least three wood before placing a table.",
                (
                    "Once you have three wood, choose place_table exactly once; it consumes "
                    "two wood and leaves the one wood needed for a pickaxe."
                ),
                (
                    "On the very next turn after place_table succeeds, choose "
                    "make_wood_pickaxe; do not move and do not use do to craft."
                ),
                "Never repeat the same two-move pattern more than twice; choose another route.",
                "Use observed directions and current facing; avoid water, lava, and enemies.",
            ]
        ),
    )


def run_recorded_episode(
    *,
    seed: int,
    max_steps: int,
    output: Path,
    provider: GameplayProvider | None = None,
    serve_port: int | None = None,
    pace_seconds: float = 0.0,
    persist: bool = True,
) -> tuple[EpisodeTrace, str | None, ThreadingHTTPServer | None]:
    """Run one baseline episode and return its trace and optional viewer server."""
    provider = provider or BedrockProvider()
    recorder = EpisodeRecorder(output, label=provider.name)
    server: ThreadingHTTPServer | None = None
    if serve_port is not None:
        handler = partial(SimpleHTTPRequestHandler, directory=str(output.resolve()))
        server = ThreadingHTTPServer(("127.0.0.1", serve_port), handler)
        threading.Thread(target=server.serve_forever, daemon=True).start()

    def record(frame: EpisodeFrame) -> None:
        recorder.on_frame(frame)
        if pace_seconds:
            time.sleep(pace_seconds)

    try:
        genome = baseline_genome()
        trace = play_episode(
            genome,
            seed,
            max_steps=max_steps,
            provider=provider,
            on_frame=record,
        )
        recorder.finish(trace)
        run_id = None
        if persist:
            database = get_db()
            if database is None:
                raise RuntimeError(
                    "Atlas is required to persist live gameplay evidence"
                )
            genome.fitness = Fitness(
                score=trace.score,
                achievements={
                    name: 1.0 if unlocked else 0.0
                    for name, unlocked in trace.achievements.items()
                },
                mean_steps=float(trace.steps),
                cost_usd=trace.cost_usd or 0.0,
            )
            upsert_genome(database, genome)
            run_id = insert_episode(database, trace, artifact_path=str(output))
            if get_run(database, run_id) is None:
                raise RuntimeError(f"Atlas could not read back persisted run {run_id}")
        return trace, run_id, server
    except Exception:
        recorder.fail("runtime_error")
        if server:
            server.shutdown()
            server.server_close()
        raise


def scripted_provider() -> ScriptedProvider:
    """Return the labeled provider used only for offline visual verification."""
    return ScriptedProvider()
