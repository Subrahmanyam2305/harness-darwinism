"""Focused checks for meta-agent structured-output recovery."""

import json

from src.genome import GenomeConfig
from src.meta_agent import llm_crossover, llm_mutate


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


def test_crossover_accepts_one_json_code_fence(monkeypatch) -> None:
    response = json.dumps(
        {
            "config": GenomeConfig(
                rules=["Keep food available", "Collect wood", "Craft tools"]
            ).model_dump(),
            "crossover_note": "Combined survival and crafting rules.",
        }
    )
    monkeypatch.setattr(
        "src.meta_agent.chat", lambda **kwargs: f"```json\n{response}\n```"
    )

    child, note = llm_crossover(GenomeConfig(), GenomeConfig())

    assert child.rules == ["Keep food available", "Collect wood", "Craft tools"]
    assert note == "Combined survival and crafting rules."


def test_crossover_rejects_prose_wrapped_json(monkeypatch) -> None:
    valid = json.dumps(
        {
            "config": GenomeConfig(
                rules=["Keep food available", "Collect wood", "Craft tools"]
            ).model_dump(),
            "crossover_note": "Combined survival and crafting rules.",
        }
    )
    responses = iter([f"Here is the result:\n{valid}", valid])
    calls = 0

    def fake_chat(**kwargs):
        nonlocal calls
        calls += 1
        return next(responses)

    monkeypatch.setattr("src.meta_agent.chat", fake_chat)

    child, note = llm_crossover(GenomeConfig(), GenomeConfig())

    assert calls == 2
    assert child.rules == ["Keep food available", "Collect wood", "Craft tools"]
    assert "attempt 2" in note


def test_crossover_rejects_unknown_output_fields(monkeypatch) -> None:
    config = GenomeConfig(
        rules=["Keep food available", "Collect wood", "Craft tools"]
    ).model_dump()
    responses = iter(
        [
            json.dumps(
                {
                    "config": {**config, "unrecognized_gene": True},
                    "crossover_note": "Contains an undeclared gene.",
                }
            ),
            json.dumps(
                {
                    "config": config,
                    "crossover_note": "Used only declared genes.",
                }
            ),
        ]
    )
    calls = 0

    def fake_chat(**kwargs):
        nonlocal calls
        calls += 1
        return next(responses)

    monkeypatch.setattr("src.meta_agent.chat", fake_chat)

    child, note = llm_crossover(GenomeConfig(), GenomeConfig())

    assert calls == 2
    assert child.rules == ["Keep food available", "Collect wood", "Craft tools"]
    assert "attempt 2" in note


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


def test_failed_mutation_identifies_the_retained_crossover_configuration(
    monkeypatch,
) -> None:
    crossover = GenomeConfig(rules=["Crossover rule"])
    monkeypatch.setattr("src.meta_agent.chat", lambda **kwargs: "invalid")

    child, note = llm_mutate(crossover, [])

    assert child == crossover
    assert note == "mutation failed — kept crossover configuration"


def test_mutation_accepts_one_json_code_fence(monkeypatch) -> None:
    crossover = GenomeConfig(rules=["Collect wood", "Craft tools"])
    mutated = GenomeConfig(rules=["Collect wood", "Craft tools", "Find food"])
    response = json.dumps(
        {
            "config": mutated.model_dump(),
            "mutation_note": "Added a food rule.",
        }
    )
    monkeypatch.setattr(
        "src.meta_agent.chat", lambda **kwargs: f"```json\n{response}\n```"
    )

    child, note = llm_mutate(crossover, [])

    assert child == mutated
    assert note == "Added a food rule."


def test_noop_mutation_is_rejected_as_failed(monkeypatch) -> None:
    crossover = GenomeConfig(rules=["Crossover rule", "Second crossover rule"])
    response = json.dumps(
        {
            "config": crossover.model_dump(),
            "mutation_note": "Claimed an improvement without changing anything.",
        }
    )
    monkeypatch.setattr("src.meta_agent.chat", lambda **kwargs: response)

    child, note = llm_mutate(crossover, [])

    assert child == crossover
    assert note == "mutation failed — no configuration change"


def test_mutation_rejects_prose_wrapped_json(monkeypatch) -> None:
    crossover = GenomeConfig(rules=["Crossover rule"])
    response = json.dumps(
        {
            "config": GenomeConfig(
                rules=["Crossover rule", "New targeted rule"]
            ).model_dump(),
            "mutation_note": "Added a targeted rule.",
        }
    )
    monkeypatch.setattr(
        "src.meta_agent.chat", lambda **kwargs: f"Here is the mutation:\n{response}"
    )

    child, note = llm_mutate(crossover, [])

    assert child == crossover
    assert note == "mutation failed — kept crossover configuration"
