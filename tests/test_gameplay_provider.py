"""Provider contract checks for usage evidence and cache eligibility."""

from types import SimpleNamespace

from src.gameplay_provider import BedrockProvider


def test_bedrock_provider_preserves_cache_usage(monkeypatch) -> None:
    captured = {}

    def fake_converse(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(
            text='{"action":"noop"}',
            model="test-model",
            input_tokens=100,
            output_tokens=4,
            cache_read_tokens=80,
            cache_write_tokens=20,
            latency_seconds=0.2,
        )

    monkeypatch.setattr("src.gameplay_provider.converse", fake_converse)
    provider = BedrockProvider(model="us.anthropic.claude-haiku-4-5-20251001-v1:0")
    prompt = "x" * 17000

    reply = provider.complete(
        [
            {"role": "system", "content": prompt},
            {"role": "user", "content": "choose"},
        ]
    )

    assert captured["cache_system"] is True
    assert reply.cache_read_tokens == 80
    assert reply.cache_write_tokens == 20
