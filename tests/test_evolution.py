"""Dry-run checks for genetic evolution invariants without provider calls."""

from copy import deepcopy

import numpy as np

from src.agent import EpisodeFrame, EpisodeTrace
from src.config import settings
from src.evolution import (
    _novelty_fallback,
    _resume_seeds,
    evaluate_genome,
    run_generation,
    seed_population,
)
from src.genome import Fitness, Genome, GenomeConfig


def test_agent_error_becomes_zero_fitness_and_keeps_trace(monkeypatch) -> None:
    genome = Genome(id="failed-candidate")
    failed_trace = EpisodeTrace(
        genome_id=genome.id,
        seed=127,
        score=1.25,
        achievements={"collect_wood": True},
        steps=42,
        termination_reason="agent_error",
        attempts=[{"attempt": 3, "error": "invalid_output"}],
        error="Three inference attempts failed at game step 42",
    )
    monkeypatch.setattr(
        "src.evolution.play_episode", lambda *args, **kwargs: failed_trace
    )

    fitness, traces = evaluate_genome(
        genome, [127], max_steps=100, record_frames=False
    )

    assert fitness.score == 0.0
    assert fitness.mean_steps == 42
    assert traces == [failed_trace]


def test_evolution_episode_records_frames_and_gif(monkeypatch, tmp_path) -> None:
    genome = Genome(id="recorded-candidate")

    def fake_play(genome, seed, max_steps, *, on_frame):
        on_frame(
            EpisodeFrame(
                step=0,
                action=None,
                rgb=np.zeros((8, 8, 3), dtype=np.uint8),
                observation="initial world",
                achievements={},
                inventory={},
                is_stub=False,
            )
        )
        return EpisodeTrace(
            genome_id=genome.id,
            seed=seed,
            score=0.0,
            achievements={},
            steps=0,
        )

    monkeypatch.setattr("src.evolution.play_episode", fake_play)
    monkeypatch.setattr(settings, "project_root", tmp_path)

    _, traces = evaluate_genome(genome, [127], max_steps=1)

    artifact = tmp_path / traces[0].artifact_path
    assert (artifact / "frames" / "00000.png").is_file()
    assert (artifact / "replay.gif").is_file()


def test_resume_recovers_seeds_from_metrics_before_latest_generation_runs(
    monkeypatch,
) -> None:
    population = seed_population(2, experiment_id="resumed")
    monkeypatch.setattr("src.evolution.get_runs_for_genome", lambda db, genome_id: [])
    monkeypatch.setattr(
        "src.evolution.get_metrics",
        lambda db, arm: [{"generation": 2, "arm": arm, "seeds": [105]}],
    )

    assert _resume_seeds(object(), population, 3, "resumed") == [105]


def test_duplicate_llm_mutation_gets_deterministic_novelty_fallback(
    monkeypatch,
) -> None:
    config = GenomeConfig()
    initial_history = config.context_budget.history_steps
    added = []
    monkeypatch.setattr(
        "src.evolution.ledger_check",
        lambda db, edit_hash: edit_hash == config.content_hash(),
    )
    monkeypatch.setattr(
        "src.evolution.ledger_add",
        lambda db, edit_hash, note: added.append((edit_hash, note)),
    )

    result, note = _novelty_fallback(object(), config, "Repeated strategy.")

    assert result.context_budget.history_steps == initial_history + 1
    assert "Novelty fallback" in note
    assert len(added) == 1


def test_generation_preserves_lineage_population_and_retries_duplicates(
    monkeypatch,
) -> None:
    population = seed_population(4, experiment_id="dry-run")
    original_ids = [genome.id for genome in population]
    persisted = []
    mutation_calls = 0
    ledger_hashes: set[str] = set()

    def fake_evaluate(genome, seeds, *, max_steps=None):
        score = float(original_ids.index(genome.id) + 1)
        trace = EpisodeTrace(
            genome_id=genome.id,
            seed=seeds[0],
            score=score,
            achievements={"collect_wood": True},
            steps=max_steps or 10,
        )
        return Fitness(score=score, mean_steps=trace.steps), [trace]

    def fake_mutate(config, failure_traces, tried_edits=None):
        nonlocal mutation_calls
        mutation_calls += 1
        result = config.model_copy(deep=True)
        result.rules.append(f"mutation-{mutation_calls}")
        return result, f"mutation-{mutation_calls}"

    monkeypatch.setattr("src.evolution.evaluate_genome", fake_evaluate)
    monkeypatch.setattr(
        "src.evolution.llm_crossover",
        lambda parent_a, parent_b: (
            GenomeConfig(rules=[*parent_a.rules, *parent_b.rules][:8]),
            "Kept the strongest rules from both parents",
        ),
    )
    monkeypatch.setattr("src.evolution.llm_mutate", fake_mutate)
    monkeypatch.setattr("src.evolution.get_archive", lambda db, arm: [])
    monkeypatch.setattr("src.evolution.get_failure_traces", lambda db, genome_id: [])
    monkeypatch.setattr("src.evolution.insert_run", lambda db, document: None)
    monkeypatch.setattr("src.evolution.record_metrics", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        "src.evolution.upsert_genome",
        lambda db, genome: persisted.append(deepcopy(genome)),
    )
    monkeypatch.setattr(
        "src.evolution.ledger_check",
        lambda db, edit_hash: mutation_calls == 1 or edit_hash in ledger_hashes,
    )
    monkeypatch.setattr(
        "src.evolution.ledger_add",
        lambda db, edit_hash, note: ledger_hashes.add(edit_hash),
    )

    next_population = run_generation(
        0,
        population,
        [101],
        db=object(),
        population_size=4,
        n_survivors=2,
        max_steps=10,
    )

    assert len(next_population) == 4
    assert [genome.id for genome in population] == original_ids
    assert all(genome.generation == 1 for genome in next_population)
    assert all(genome.parents for genome in next_population)
    children = [genome for genome in next_population if len(genome.parents) == 2]
    assert all(genome.selection_note for genome in children)
    assert all(genome.crossover_note for genome in children)
    assert all(genome.crossover_config is not None for genome in children)
    assert mutation_calls == 3  # first duplicate plus one unique mutation per child
    assert len(ledger_hashes) == 2
    assert all(genome.status == "archived" for genome in population)
    assert {genome.id for genome in persisted}.issuperset(original_ids)
