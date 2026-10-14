"""CLI command wiring checks that avoid provider and database calls."""

from argparse import Namespace

import run
from src.genome import Genome


def test_evolve_forwards_seed_profile_and_explicit_training_seeds(monkeypatch) -> None:
    captured: dict = {}

    def fake_run_evolution(**kwargs):
        captured.update(kwargs)
        return Genome(id="champion")

    monkeypatch.setattr("src.evolution.run_evolution", fake_run_evolution)
    args = Namespace(
        generations=2,
        experiment_id="strong-seeds-200-v1",
        population=10,
        survivors=5,
        seeds_per_generation=1,
        max_steps=200,
        resume=False,
        seed_profile="strong-v2",
        train_seeds=[105],
        controller="skills",
        workers=3,
        max_skill_steps=4,
    )

    run.cmd_evolve(args)

    assert captured["seed_profile"] == "strong-v2"
    assert captured["train_seeds"] == [105]
    assert captured["controller"] == "skills"
    assert captured["workers"] == 3
    assert captured["max_skill_steps"] == 4
