"""Aggregate seeded episodes and compare frozen harnesses on paired worlds."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter

import numpy as np

from src.agent import EpisodeTrace, play_episode
from src.crafter_env import ALL_ACHIEVEMENTS
from src.gameplay_provider import GameplayProvider
from src.genome import Genome
from src.recording import EpisodeRecorder


@dataclass
class EvaluationResult:
    """Aggregate evidence for one frozen harness and ordered seed set."""

    genome_id: str
    seeds: list[int]
    complete: bool
    score: float | None
    achievement_rates: dict[str, float]
    mean_steps: float
    environment_steps: int
    model_calls: int
    steps_per_model_call: float | None
    input_tokens: int | None
    output_tokens: int | None
    tokens_per_environment_step: float | None
    cost_usd: float | None
    latency_seconds: float
    wall_clock_seconds: float | None
    controller: str
    episode_count: int
    survival_count: int
    incomplete_episodes: int
    skill_interruptions: int
    invalid_outputs: int
    provider_errors: int
    retries: int
    traces: list[EpisodeTrace]
    schema_version: int = 2

    def to_dict(self) -> dict:
        """Return a JSON-compatible evaluation document."""
        return asdict(self)


@dataclass
class PairedComparison:
    """Held-out baseline/champion comparison over identical ordered seeds."""

    seeds: list[int]
    sample_size: int
    baseline_score: float
    champion_score: float
    score_delta: float
    mean_episode_delta: float
    confidence_interval_95: tuple[float, float]
    achievement_rate_deltas: dict[str, float]
    baseline_genome_id: str
    champion_genome_id: str
    split: str = "heldout"
    schema_version: int = 1

    def to_dict(self) -> dict:
        """Return JSON-compatible paired evidence."""
        return asdict(self)


@dataclass
class ControllerComparison:
    """Paired primitive-versus-skills evidence for one frozen harness."""

    genome_id: str
    seeds: list[int]
    primitive: EvaluationResult
    skills: EvaluationResult
    model_call_delta: int
    latency_delta_seconds: float
    score_delta: float | None
    schema_version: int = 1

    def to_dict(self) -> dict:
        """Return a JSON-compatible controller comparison document."""
        return asdict(self)


def crafter_score(achievement_rates: dict[str, float]) -> float:
    """Compute upstream score from achievement success rates in [0, 1]."""
    percents = np.array(
        [
            100.0 * achievement_rates.get(achievement, 0.0)
            for achievement in ALL_ACHIEVEMENTS
        ]
    )
    return float(np.expm1(np.mean(np.log1p(percents))))


def _total_if_complete(
    traces: list[EpisodeTrace], attribute: str
) -> int | float | None:
    values = [getattr(trace, attribute) for trace in traces]
    if not values or any(value is None for value in values):
        return None
    return sum(values)


def summarize_traces(
    traces: list[EpisodeTrace], *, wall_clock_seconds: float | None = None
) -> EvaluationResult:
    """Aggregate episodes; any agent error makes the batch incomplete."""
    if not traces:
        raise ValueError("At least one episode trace is required")
    seeds = [trace.seed for trace in traces]
    if len(seeds) != len(set(seeds)):
        raise ValueError("Evaluation seeds must be unique")
    rates = {
        achievement: float(
            np.mean(
                [
                    1.0 if trace.achievements.get(achievement, False) else 0.0
                    for trace in traces
                ]
            )
        )
        for achievement in ALL_ACHIEVEMENTS
    }
    complete = all(trace.termination_reason != "agent_error" for trace in traces)
    environment_steps = sum(trace.steps for trace in traces)
    model_calls = sum(trace.model_calls for trace in traces)
    input_tokens = _total_if_complete(traces, "input_tokens")
    output_tokens = _total_if_complete(traces, "output_tokens")
    controller = traces[0].controller
    if any(trace.controller != controller for trace in traces):
        raise ValueError("Evaluation traces must use one controller")
    invalid_outputs = sum(
        attempt.get("error") == "invalid_output"
        for trace in traces
        for attempt in trace.attempts
    )
    provider_errors = sum(
        attempt.get("error") == "provider_error"
        for trace in traces
        for attempt in trace.attempts
    )
    retries = sum(
        int(attempt.get("attempt", 1)) > 1
        for trace in traces
        for attempt in trace.attempts
    )
    return EvaluationResult(
        genome_id=traces[0].genome_id,
        seeds=seeds,
        complete=complete,
        score=crafter_score(rates) if complete else None,
        achievement_rates=rates,
        mean_steps=float(np.mean([trace.steps for trace in traces])),
        environment_steps=environment_steps,
        model_calls=model_calls,
        steps_per_model_call=(
            environment_steps / model_calls if model_calls > 0 else None
        ),
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        tokens_per_environment_step=(
            (input_tokens + output_tokens) / environment_steps
            if input_tokens is not None
            and output_tokens is not None
            and environment_steps > 0
            else None
        ),
        cost_usd=_total_if_complete(traces, "cost_usd"),
        latency_seconds=float(sum(trace.latency_seconds for trace in traces)),
        wall_clock_seconds=wall_clock_seconds,
        controller=controller,
        episode_count=len(traces),
        survival_count=sum(trace.termination_reason != "death" for trace in traces),
        incomplete_episodes=sum(
            trace.termination_reason == "agent_error" for trace in traces
        ),
        skill_interruptions=sum(
            decision.get("status") in {"failed", "interrupted", "exhausted"}
            for trace in traces
            for decision in trace.decisions
        ),
        invalid_outputs=invalid_outputs,
        provider_errors=provider_errors,
        retries=retries,
        traces=traces,
    )


def evaluate_harness(
    genome: Genome,
    seeds: list[int],
    *,
    max_steps: int = 200,
    provider: GameplayProvider | None = None,
    controller: str = "primitive",
    max_skill_steps: int = 5,
    artifact_root: Path | None = None,
) -> EvaluationResult:
    """Evaluate one unchanged harness on each explicit seed."""
    if not seeds:
        raise ValueError("At least one evaluation seed is required")
    started = perf_counter()
    traces: list[EpisodeTrace] = []
    for seed in seeds:
        recorder = (
            EpisodeRecorder(
                artifact_root / controller / f"seed-{seed}",
                label=f"Controller comparison · {controller} · seed {seed}",
            )
            if artifact_root is not None
            else None
        )
        try:
            trace = play_episode(
                genome,
                seed,
                max_steps=max_steps,
                provider=provider,
                on_frame=recorder.on_frame if recorder else None,
                controller=controller,
                max_skill_steps=max_skill_steps,
            )
        except Exception:
            if recorder:
                recorder.fail("episode_exception")
            raise
        if recorder:
            trace.artifact_path = str(recorder.path)
            recorder.finish(trace)
        traces.append(trace)
    return summarize_traces(traces, wall_clock_seconds=perf_counter() - started)


def compare_controllers(
    primitive: EvaluationResult, skills: EvaluationResult
) -> ControllerComparison:
    """Compare two controller arms using the same harness and ordered seeds."""
    if primitive.seeds != skills.seeds:
        raise ValueError("Controller comparison requires identical ordered seeds")
    if primitive.genome_id != skills.genome_id:
        raise ValueError("Controller comparison requires one frozen harness")
    if primitive.controller != "primitive" or skills.controller != "skills":
        raise ValueError("Controller arms must be primitive then skills")
    score_delta = (
        skills.score - primitive.score
        if primitive.score is not None and skills.score is not None
        else None
    )
    return ControllerComparison(
        genome_id=primitive.genome_id,
        seeds=list(primitive.seeds),
        primitive=primitive,
        skills=skills,
        model_call_delta=skills.model_calls - primitive.model_calls,
        latency_delta_seconds=skills.latency_seconds - primitive.latency_seconds,
        score_delta=score_delta,
    )


def compare_paired(
    baseline: EvaluationResult,
    champion: EvaluationResult,
    *,
    bootstrap_samples: int = 10_000,
) -> PairedComparison:
    """Compare two complete evaluations with paired bootstrap uncertainty."""
    if not baseline.complete or not champion.complete:
        raise ValueError("Paired comparison requires two complete evaluations")
    if baseline.seeds != champion.seeds:
        raise ValueError("Paired comparison requires identical ordered seeds")
    if baseline.score is None or champion.score is None:
        raise ValueError("Complete evaluations must have aggregate scores")
    if bootstrap_samples <= 0:
        raise ValueError("bootstrap_samples must be positive")

    episode_deltas = np.array(
        [
            champion_trace.score - baseline_trace.score
            for baseline_trace, champion_trace in zip(
                baseline.traces, champion.traces, strict=True
            )
        ]
    )
    random = np.random.default_rng(0)
    indices = random.integers(
        0, len(episode_deltas), size=(bootstrap_samples, len(episode_deltas))
    )
    bootstrap_means = episode_deltas[indices].mean(axis=1)
    low, high = np.percentile(bootstrap_means, [2.5, 97.5])
    return PairedComparison(
        seeds=list(baseline.seeds),
        sample_size=len(baseline.seeds),
        baseline_score=baseline.score,
        champion_score=champion.score,
        score_delta=champion.score - baseline.score,
        mean_episode_delta=float(episode_deltas.mean()),
        confidence_interval_95=(float(low), float(high)),
        achievement_rate_deltas={
            achievement: (
                champion.achievement_rates[achievement]
                - baseline.achievement_rates[achievement]
            )
            for achievement in ALL_ACHIEVEMENTS
        },
        baseline_genome_id=baseline.genome_id,
        champion_genome_id=champion.genome_id,
    )
