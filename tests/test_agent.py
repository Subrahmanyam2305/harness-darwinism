"""Behavioral tests for one LLM-controlled Crafter episode."""

from dataclasses import dataclass

import numpy as np

from src.agent import play_episode
from src.crafter_env import CrafterSnapshot, NearbyEntity
from src.gameplay_provider import ProviderError
from src.genome import Genome, GenomeConfig


@dataclass(frozen=True)
class FakeReply:
    text: str
    model: str = "test-model"
    input_tokens: int = 10
    output_tokens: int = 2
    cost_usd: float | None = None
    latency_seconds: float = 0.01


class FakeProvider:
    name = "test-provider"
    is_stub = True

    def __init__(self, replies: list[str]):
        self.replies = iter(replies)

    def complete(self, messages: list[dict[str, str]]) -> FakeReply:
        return FakeReply(next(self.replies))


class FakeSkillEnvironment:
    """Two-step tree collection world for controller contract tests."""

    def __init__(self, seed=None, max_steps=200):
        self.max_steps = max_steps
        self.step_count = 0
        self.wood = 0
        self.info = {
            "inventory": {"health": 9, "food": 9, "drink": 9, "energy": 9, "wood": 0}
        }

    def reset(self):
        return "tree: east (dist 2)"

    def snapshot(self):
        distance = max(1, 2 - self.step_count)
        return CrafterSnapshot(
            step=self.step_count,
            inventory={**self.info["inventory"], "wood": self.wood},
            achievements={"collect_wood": self.wood > 0},
            nearby_entities=[
                NearbyEntity(
                    kind="tree", direction="east", distance=distance, dx=distance, dy=0
                )
            ],
            facing="east",
            daylight=1.0,
            sleeping=False,
        )

    def step(self, action):
        self.step_count += 1
        if action == "do":
            self.wood += 1
        self.info["inventory"]["wood"] = self.wood
        return (
            f"after {action}",
            float(action == "do"),
            self.step_count >= self.max_steps,
            self.info,
        )

    def render(self):
        return np.zeros((8, 8, 3), dtype=np.uint8)

    def get_achievements(self):
        return {"collect_wood": self.wood > 0}

    def get_score(self):
        return float(self.wood > 0)


def test_invalid_replies_retry_without_advancing_crafter() -> None:
    provider = FakeProvider(["not json", '{"action":"teleport"}', '{"action":"noop"}'])
    frames = []

    trace = play_episode(
        Genome(id="baseline"),
        seed=42,
        max_steps=1,
        provider=provider,
        on_frame=frames.append,
    )

    assert trace.steps == 1
    assert trace.termination_reason == "step_limit"
    assert [frame.step for frame in frames] == [0, 1]
    assert len(trace.attempts) == 3
    assert trace.input_tokens == 30
    assert trace.output_tokens == 6


def test_three_invalid_replies_end_as_agent_error_without_a_fallback_action() -> None:
    provider = FakeProvider(["{}", "{}", "{}"])

    trace = play_episode(Genome(id="baseline"), seed=42, provider=provider)

    assert trace.termination_reason == "agent_error"
    assert trace.steps == 0
    assert trace.history == []
    assert len(trace.attempts) == 3
    assert "Died" not in trace.failure_summary


def test_genome_tool_allowlist_is_enforced() -> None:
    genome = Genome(id="restricted", genome=GenomeConfig(tools=["sleep"]))
    provider = FakeProvider(['{"action":"move_right"}'] * 3)

    trace = play_episode(genome, seed=42, provider=provider)

    assert trace.termination_reason == "agent_error"
    assert trace.steps == 0
    assert all(attempt["error"] == "invalid_output" for attempt in trace.attempts)


def test_skill_controller_executes_multiple_steps_from_one_model_decision(
    monkeypatch,
) -> None:
    monkeypatch.setattr("src.agent.CrafterTextEnv", FakeSkillEnvironment)
    provider = FakeProvider(
        [
            (
                '{"state_patch":{"goal":"collect wood","subgoal":"reach tree"},'
                '"skill":{"name":"acquire","target":"tree","max_steps":4}}'
            )
        ]
    )
    frames = []

    trace = play_episode(
        Genome(id="skills", genome=GenomeConfig(tools=["move_right", "do"])),
        seed=42,
        max_steps=2,
        provider=provider,
        controller="skills",
        on_frame=frames.append,
    )

    assert trace.steps == 2
    assert trace.model_calls == 1
    assert trace.controller == "skills"
    assert [item["action"] for item in trace.history] == ["move_right", "do"]
    assert len({item["decision_id"] for item in trace.history}) == 1
    assert trace.decisions[0]["skill"]["name"] == "acquire"
    assert trace.decisions[0]["actions"] == ["move_right", "do"]
    assert trace.decisions[0]["status"] in {"succeeded", "terminal"}
    assert trace.decisions[0]["state_before"]["goal"] is None
    assert trace.decisions[0]["state_patch"]["goal"] == "collect wood"
    assert trace.decisions[0]["state_after"]["goal"] == "collect wood"
    assert any(
        item["option"] == "acquire:tree" and item["feasible"]
        for item in trace.decisions[0]["affordances"]
    )
    assert trace.decisions[0]["guard_checks"]
    assert "tree" in trace.decisions[0]["observation_before"]
    assert trace.decisions[0]["observation_after"] == "after do"
    assert {frame.decision_id for frame in frames[1:]} == {"decision-1"}


def test_skill_prompt_uses_compact_state_without_prior_transcript(monkeypatch) -> None:
    monkeypatch.setattr("src.agent.CrafterTextEnv", FakeSkillEnvironment)
    captured: list[list[dict[str, str]]] = []

    class CapturingProvider(FakeProvider):
        def complete(self, messages):
            captured.append(messages)
            return super().complete(messages)

    provider = CapturingProvider(
        [
            (
                '{"state_patch":{"goal":"explore"},'
                '"skill":{"name":"explore","max_steps":1}}'
            ),
            (
                '{"state_patch":{"subgoal":"collect tree"},'
                '"skill":{"name":"acquire","target":"tree","max_steps":1}}'
            ),
        ]
    )

    trace = play_episode(
        Genome(id="compact"),
        seed=42,
        max_steps=2,
        provider=provider,
        controller="skills",
        max_skill_steps=1,
    )

    assert trace.model_calls == 2
    assert len(captured) == 2
    assert all(len(messages) == 2 for messages in captured)
    assert "thought" not in captured[1][0]["content"].lower()
    assert "Execution state" in captured[1][1]["content"]
    assert [message["role"] for message in captured[1]] == ["system", "user"]


def test_retryable_provider_failure_uses_bounded_backoff(monkeypatch) -> None:
    waits: list[float] = []
    monkeypatch.setattr("src.agent.sleep", waits.append)
    monkeypatch.setattr("src.agent.uniform", lambda low, high: high / 2)

    class RetryingProvider:
        name = "retrying"
        is_stub = True

        def __init__(self):
            self.calls = 0

        def complete(self, messages):
            self.calls += 1
            if self.calls == 1:
                raise ProviderError("throttled", retryable=True)
            return FakeReply('{"action":"noop"}')

    trace = play_episode(
        Genome(id="retry"), seed=42, max_steps=1, provider=RetryingProvider()
    )

    assert trace.steps == 1
    assert waits == [0.5]
