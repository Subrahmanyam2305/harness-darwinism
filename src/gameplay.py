"""Run a fixed baseline harness with live frames and saved replay evidence."""

from __future__ import annotations

import threading
import time
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from src.agent import EpisodeFrame, EpisodeTrace, play_episode
from src.db import get_db, get_run, insert_episode
from src.gameplay_provider import BedrockProvider, GameplayProvider, ScriptedProvider
from src.genome import Genome, GenomeConfig
from src.recording import EpisodeRecorder


def baseline_genome() -> Genome:
    """Return the frozen generation-zero harness used for baseline evidence."""
    return Genome(
        id="fixed-baseline",
        genome=GenomeConfig(
            rules=[
                "Collect at least three wood before crafting.",
                "Place a table once you have two wood, then collect one more wood.",
                (
                    "Immediately choose make_wood_pickaxe while standing next to the table; "
                    "do not use do to craft."
                ),
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
        trace = play_episode(
            baseline_genome(),
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
