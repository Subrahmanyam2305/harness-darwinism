"""Persistence contract tests for complete gameplay episode evidence."""

from src.agent import EpisodeTrace
from src.db import _format_failure_trace, get_run, insert_episode


class FakeCollection:
    def __init__(self) -> None:
        self.documents: dict[str, dict] = {}

    def insert_one(self, document: dict):
        self.documents[document["_id"]] = document.copy()

    def find_one(self, query: dict):
        document = self.documents.get(query["_id"])
        return document.copy() if document else None


class FakeDatabase(dict):
    def __getitem__(self, name: str):
        return super().__getitem__(name)


def test_complete_episode_round_trips_with_provider_and_replay_evidence() -> None:
    database = FakeDatabase(runs=FakeCollection())
    trace = EpisodeTrace(
        genome_id="fixed-baseline",
        seed=42,
        score=0.5,
        achievements={"collect_wood": True},
        steps=3,
        provider="aws-bedrock",
        model="test-model",
        is_stub=False,
        harness={"rules": ["collect wood"]},
    )

    run_id = insert_episode(database, trace, artifact_path="data/runs/example")
    stored = get_run(database, run_id)

    assert stored is not None
    assert stored["_id"] == run_id
    assert stored["kind"] == "gameplay_episode"
    assert stored["provider"] == "aws-bedrock"
    assert stored["is_stub"] is False
    assert stored["artifact_path"] == "data/runs/example"
    assert stored["trace"]["achievements"] == {"collect_wood": True}
    assert stored["created_at"].endswith("+00:00")


def test_failure_trace_summarizes_repeated_skills_and_blocked_prerequisites() -> None:
    run = {
        "score": 1.3,
        "termination_reason": "death",
        "steps": 83,
        "max_steps": 200,
        "achievements": {"collect_wood": True, "make_stone_pickaxe": False},
        "decisions": [
            {
                "skill": {"name": "acquire", "target": "stone"},
                "status": "interrupted",
                "reason": "no_progress",
                "affordances": [
                    {
                        "option": "build_or_craft:stone_pickaxe",
                        "feasible": False,
                        "reason": "missing_workstation:table",
                    }
                ],
            },
            {
                "skill": {"name": "acquire", "target": "stone"},
                "status": "interrupted",
                "reason": "no_progress",
                "affordances": [],
            },
        ],
    }

    summary = _format_failure_trace(run)

    assert "termination=death at 83/200 steps" in summary
    assert "acquire:stone -> no_progress x2" in summary
    assert "build_or_craft:stone_pickaxe -> missing_workstation:table" in summary
    assert "missing_achievements=make_stone_pickaxe" in summary
