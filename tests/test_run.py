"""CLI command wiring checks that avoid provider and database calls."""

import sys
from argparse import Namespace
from pathlib import Path

import pytest

import run
from src.config import Settings
from src.genome import Genome


def test_evolution_policy_defaults_to_thirty_percent_elites_and_forty_percent_mutation() -> (
    None
):
    defaults = Settings(_env_file=None)

    assert defaults.elite_ratio == 0.3
    assert defaults.mutation_rate == 0.4


def test_evolve_uses_settings_for_fixed_policy_and_forwards_run_controls(
    monkeypatch, capsys
) -> None:
    captured: dict = {}
    monkeypatch.setattr(run, "settings", Settings(_env_file=None))

    def fake_run_evolution(**kwargs):
        captured.update(kwargs)
        return Genome(id="champion")

    monkeypatch.setattr("src.evolution.run_evolution", fake_run_evolution)
    args = Namespace(
        generations=2,
        experiment_id="strong-seeds-200-v1",
        max_steps=200,
        resume=False,
        seed_profile="strong-v2",
        train_seeds=[105],
        controller="skills",
        workers=3,
    )

    run.cmd_evolve(args)

    assert captured["seed_profile"] == "strong-v2"
    assert captured["train_seeds"] == [105]
    assert captured["controller"] == "skills"
    assert captured["workers"] == 3
    assert captured["population_size"] == 10
    assert captured["seeds_per_generation"] == 1
    assert captured["max_skill_steps"] == 5
    assert captured["elite_ratio"] == 0.3
    assert captured["mutation_rate"] == 0.4
    output = capsys.readouterr().out
    assert "worst-case retry upper bound: 18,000 gameplay calls" in output


def test_evolve_help_hides_fixed_policy_arguments(monkeypatch, capsys) -> None:
    monkeypatch.setattr(sys, "argv", ["run.py", "evolve", "--help"])

    with pytest.raises(SystemExit) as exit_info:
        run.main()

    assert exit_info.value.code == 0
    output = capsys.readouterr().out
    for removed in (
        "--population",
        "--survivors",
        "--elite-ratio",
        "--mutation-rate",
        "--seeds-per-generation",
        "--max-skill-steps",
    ):
        assert removed not in output


def test_compare_controllers_uses_identical_settings(monkeypatch, tmp_path) -> None:
    calls: list[dict] = []
    genome = Genome(id="candidate")

    class Result:
        complete = True
        traces = ()

        def to_dict(self):
            return {"controller": calls[-1]["controller"]}

    class Comparison:
        model_call_delta = -3
        latency_delta_seconds = -1.0

        def to_dict(self):
            return {"model_call_delta": self.model_call_delta}

    def fake_evaluate(candidate, seeds, **kwargs):
        calls.append({"genome": candidate.id, "seeds": seeds, **kwargs})
        return Result()

    monkeypatch.setattr("src.db.get_db", lambda: object())
    monkeypatch.setattr("src.db.get_genome", lambda _db, _id: genome)
    monkeypatch.setattr("src.db.insert_run", lambda *_args, **_kwargs: None)
    monkeypatch.setattr("src.evaluator.evaluate_harness", fake_evaluate)
    monkeypatch.setattr(
        "src.evaluator.compare_controllers", lambda primitive, skills: Comparison()
    )
    output = tmp_path / "controllers.json"

    run.cmd_compare_controllers(
        Namespace(
            genome_id="candidate",
            seeds=[201, 202],
            max_steps=80,
            max_skill_steps=4,
            output=Path(output),
        )
    )

    assert [call["controller"] for call in calls] == ["primitive", "skills"]
    assert all(call["seeds"] == [201, 202] for call in calls)
    assert all(call["max_steps"] == 80 for call in calls)
    assert calls[1]["max_skill_steps"] == 4
    assert calls[0]["artifact_root"] == calls[1]["artifact_root"]
    assert output.exists()
