"""Persistence contract tests for complete gameplay episode evidence."""

from src.agent import EpisodeTrace
from src.db import get_run, insert_episode


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
