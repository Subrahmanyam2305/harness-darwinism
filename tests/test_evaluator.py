"""Behavioral tests for multi-seed and paired held-out evidence."""

import math

import pytest

from src.agent import EpisodeTrace
from src.crafter_env import ALL_ACHIEVEMENTS
from src.evaluator import compare_controllers, compare_paired, summarize_traces


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
        model_calls=4,
        latency_seconds=2.5,
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
    assert result.environment_steps == 40
    assert result.model_calls == 8
    assert result.latency_seconds == 5.0
    assert result.steps_per_model_call == 5.0
    assert result.tokens_per_environment_step == 5.5


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


def test_controller_comparison_reports_paired_efficiency_and_unavailable_cost() -> None:
    primitive_trace = _trace(200, {"collect_wood"})
    primitive_trace.controller = "primitive"
    primitive_trace.cost_usd = None
    skills_trace = _trace(200, {"collect_wood", "place_table"})
    skills_trace.controller = "skills"
    skills_trace.model_calls = 2
    skills_trace.latency_seconds = 1.0
    skills_trace.cost_usd = None

    comparison = compare_controllers(
        summarize_traces([primitive_trace]),
        summarize_traces([skills_trace]),
    )

    assert comparison.seeds == [200]
    assert comparison.primitive.controller == "primitive"
    assert comparison.skills.controller == "skills"
    assert comparison.model_call_delta == -2
    assert comparison.latency_delta_seconds == -1.5
    assert comparison.primitive.cost_usd is None
    assert comparison.skills.cost_usd is None
