"""Focused checks for meta-agent structured-output recovery."""

import json

from src.genome import GenomeConfig
from src.meta_agent import llm_crossover


def test_crossover_retries_invalid_response_and_records_recovery(monkeypatch) -> None:
    responses = iter(
        [
            "not json",
            json.dumps(
                {
                    "config": GenomeConfig(
                        rules=["Keep food available", "Collect wood", "Craft tools"]
                    ).model_dump(),
                    "crossover_note": "Combined survival and crafting rules.",
                }
            ),
        ]
    )
    prompts: list[str] = []

    def fake_chat(*, messages, **kwargs):
        prompts.append(messages[0]["content"])
        return next(responses)

    monkeypatch.setattr("src.meta_agent.chat", fake_chat)

    child, note = llm_crossover(GenomeConfig(), GenomeConfig())

    assert child.rules == ["Keep food available", "Collect wood", "Craft tools"]
    assert "attempt 2" in note
    assert "no JSON object found" in prompts[1]


def test_crossover_falls_back_only_after_all_attempts(monkeypatch) -> None:
    parent_a = GenomeConfig(rules=["Parent A rule"])
    calls = 0

    def invalid_chat(**kwargs):
        nonlocal calls
        calls += 1
        return "invalid"

    monkeypatch.setattr("src.meta_agent.chat", invalid_chat)

    child, note = llm_crossover(parent_a, GenomeConfig(), max_attempts=3)

    assert calls == 3
    assert child == parent_a
    assert "failed after 3 attempts" in note
    assert note.count("no JSON object found") == 3
