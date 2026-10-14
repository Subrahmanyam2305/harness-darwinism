"""Settings source checks for committed policy and ignored secrets."""

from pathlib import Path
from typing import ClassVar

from src.settings import Settings


def test_yaml_policy_and_dotenv_secrets_are_loaded(tmp_path) -> None:
    config_path = tmp_path / "config.yml"
    config_path.write_text(
        "population_size: 6\n"
        "elite_ratio: 0.2\n"
        "mutation_rate: 0.8\n"
        "seeds_per_gen: 2\n"
        "max_skill_steps: 3\n"
    )
    env_path = tmp_path / ".env"
    env_path.write_text(
        "AWS_ACCESS_KEY_ID=test-access\n"
        "AWS_SECRET_ACCESS_KEY=test-secret\n"
        "AWS_SESSION_TOKEN=test-token\n"
        "MONGODB_URI=mongodb://example.test:27017\n"
    )

    class TestSettings(Settings):
        yaml_path: ClassVar[Path] = config_path

    loaded = TestSettings(_env_file=env_path)

    assert loaded.population_size == 6
    assert loaded.elite_ratio == 0.2
    assert loaded.mutation_rate == 0.8
    assert loaded.seeds_per_gen == 2
    assert loaded.max_skill_steps == 3
    assert loaded.aws_access_key_id == "test-access"
    assert loaded.mongodb_uri == "mongodb://example.test:27017"
