"""Behavioral tests for the Crafter gameplay boundary."""

import numpy as np
import pytest

from src.crafter_env import CrafterTextEnv, _info_to_text


def test_reset_does_not_consume_a_game_step() -> None:
    env = CrafterTextEnv(seed=42)

    observation = env.reset()

    assert env.step_count == 0
    assert env.env._step == 0
    assert "[Step 0]" in observation


def test_text_observation_uses_upstream_xy_coordinates() -> None:
    semantic = np.full((9, 9), 2)
    semantic[5, 4] = 6  # Tree is east in Crafter's semantic[x, y] map.
    semantic[4, 3] = 1  # Water is north.

    observation = _info_to_text(
        {"semantic": semantic, "player_pos": (4, 4), "inventory": {}}, step=0
    )

    assert "tree: east (dist 1)" in observation
    assert "water: north (dist 1)" in observation


def test_invalid_action_does_not_silently_become_noop() -> None:
    env = CrafterTextEnv(seed=42)
    env.reset()

    with pytest.raises(ValueError, match="Unknown action"):
        env.step("teleport")

    assert env.step_count == 0
    assert env.env._step == 0


def test_rendering_at_night_preserves_gameplay_rng() -> None:
    env = CrafterTextEnv(seed=42)
    env.reset()
    env.env._world.daylight = 0.0
    before = env.env._world.random.get_state()

    frame = env.render()

    after = env.env._world.random.get_state()
    assert frame.shape == (512, 512, 3)
    assert before[0] == after[0]
    np.testing.assert_array_equal(before[1], after[1])
    assert before[2:] == after[2:]


def test_episode_with_no_achievements_has_zero_crafter_score() -> None:
    env = CrafterTextEnv(seed=42)
    env.reset()

    assert env.get_score() == 0.0
