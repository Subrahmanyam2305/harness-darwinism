"""Behavioral tests for live and replayable gameplay evidence."""

import json

import numpy as np
from PIL import Image

from src.agent import EpisodeFrame, EpisodeTrace
from src.recording import EpisodeRecorder


def test_episode_recording_is_live_then_replayable(tmp_path) -> None:
    output = tmp_path / "episode"
    recorder = EpisodeRecorder(output, label="offline-test")
    for step in range(3):
        recorder.on_frame(
            EpisodeFrame(
                step=step,
                action=None if step == 0 else "noop",
                rgb=np.full((32, 32, 3), step * 80, dtype=np.uint8),
                observation=f"step {step}",
                achievements={},
                inventory={},
                is_stub=True,
            )
        )

    assert json.loads((output / "live.json").read_text())["state"] == "playing"

    recorder.finish(EpisodeTrace("baseline", 42, 0.0, {}, 2, is_stub=True))

    manifest = json.loads((output / "live.json").read_text())
    assert manifest["state"] == "finished"
    assert len(manifest["frames"]) == 3
    assert json.loads((output / "result.json").read_text())["is_stub"] is True
    with Image.open(output / "replay.gif") as replay:
        assert replay.n_frames == 3
    assert (output / "index.html").is_file()
