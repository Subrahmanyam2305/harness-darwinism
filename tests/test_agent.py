"""Behavioral tests for one LLM-controlled Crafter episode."""

from dataclasses import dataclass

from src.agent import play_episode
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
