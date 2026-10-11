"""Behavioral tests for multi-seed and paired held-out evidence."""

import math

import pytest

from src.agent import EpisodeTrace
from src.crafter_env import ALL_ACHIEVEMENTS
from src.evaluator import compare_paired, summarize_traces


def _trace(seed: int, unlocked: set[str]) -> EpisodeTrace:
    achievements = {name: name in unlocked for name in ALL_ACHIEVEMENTS}
    return EpisodeTrace(
        genome_id="baseline",
        seed=seed,
        score=0.0,
        achievements=achievements,
        steps=20,
        termination_reason="step_limit",
        input_tokens=100,
        output_tokens=10,
        cost_usd=0.01,
    )


def test_multi_seed_score_uses_achievement_success_rates() -> None:
    result = summarize_traces(
        [
            _trace(200, {"collect_wood"}),
            _trace(201, {"collect_wood", "place_table"}),
        ]
    )

    expected = (
        math.exp(sum(math.log1p(rate) for rate in [100.0, 50.0] + [0.0] * 20) / 22) - 1
    )
    assert result.complete is True
    assert result.score == pytest.approx(expected)
    assert result.achievement_rates["collect_wood"] == 1.0
    assert result.achievement_rates["place_table"] == 0.5
    assert result.input_tokens == 200
    assert result.cost_usd == 0.02


def test_agent_error_invalidates_the_batch() -> None:
    failed = _trace(201, set())
    failed.termination_reason = "agent_error"

    result = summarize_traces([_trace(200, {"collect_wood"}), failed])

    assert result.complete is False
    assert result.score is None


def test_heldout_comparison_requires_identical_ordered_seeds() -> None:
    baseline = summarize_traces([_trace(200, set()), _trace(201, set())])
    champion = summarize_traces(
        [
            _trace(201, {"collect_wood"}),
            _trace(200, {"collect_wood"}),
        ]
    )

    with pytest.raises(ValueError, match="identical ordered seeds"):
        compare_paired(baseline, champion)
