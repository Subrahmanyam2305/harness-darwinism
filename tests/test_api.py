"""Dashboard API serialization checks for legacy and skill-controller runs."""

from api import _serialize_run


def test_skill_run_exposes_grouped_decisions_and_efficiency_metrics() -> None:
    run = {
        "_id": "skill-run",
        "genome_id": "g1",
        "steps": 4,
        "controller": "skills",
        "model_calls": 2,
        "input_tokens": 100,
        "output_tokens": 20,
        "latency_seconds": 1.5,
        "decisions": [
            {
                "decision_id": "decision-1",
                "skill": {"name": "acquire", "target": "tree"},
                "actions": ["move_right", "do"],
                "status": "succeeded",
                "reason": "inventory_changed",
            }
        ],
        "history": [
            {
                "step": 1,
                "decision_id": "decision-1",
                "skill": "acquire",
                "skill_status": "succeeded",
                "action": "move_right",
                "observation_after": "closer",
            }
        ],
    }

    serialized = _serialize_run(run)

    assert serialized["controller"] == "skills"
    assert serialized["model_calls"] == 2
    assert serialized["environment_steps"] == 4
    assert serialized["steps_per_model_call"] == 2.0
    assert serialized["tokens_per_environment_step"] == 30.0
    assert serialized["decisions"][0]["actions"] == ["move_right", "do"]
    assert serialized["history"][0]["decision_id"] == "decision-1"


def test_legacy_run_keeps_unknown_efficiency_fields_unavailable() -> None:
    serialized = _serialize_run(
        {
            "_id": "legacy",
            "genome_id": "old",
            "steps": 3,
            "history": [{"step": 1, "action": "noop"}],
        }
    )

    assert serialized["controller"] == "primitive"
    assert serialized["model_calls"] is None
    assert serialized["steps_per_model_call"] is None
    assert serialized["tokens_per_environment_step"] is None
    assert serialized["decisions"] == []
