"""Typed settings loaded from committed YAML and ignored environment secrets."""

from __future__ import annotations

from pathlib import Path
from typing import ClassVar

from pydantic import Field
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    YamlConfigSettingsSource,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    """Combine secret environment values with non-secret YAML policy."""

    yaml_path: ClassVar[Path] = PROJECT_ROOT / "config.yml"

    # Bedrock models
    agent_model: str = "us.anthropic.claude-haiku-4-5-20251001-v1:0"
    meta_model: str = "us.anthropic.claude-opus-4-6-v1"
    aws_default_region: str = "us-east-1"
    aws_access_key_id: str = ""
    aws_secret_access_key: str = ""
    aws_session_token: str = ""

    # MongoDB
    mongodb_uri: str = "mongodb://localhost:27017"
    db_name: str = "harness_darwinism"

    # Evolution
    population_size: int = Field(ge=2)
    elite_ratio: float = Field(gt=0, lt=1)
    mutation_rate: float = Field(ge=0, le=1)
    seeds_per_gen: int = Field(ge=1)
    max_generations: int = 10
    max_steps_per_episode: int = 200
    max_skill_steps: int = Field(ge=1, le=5)
    archive_sample_ratio: float = 0.1

    # Crafter
    crafter_size: tuple[int, int] = (64, 64)

    # Paths
    project_root: Path = PROJECT_ROOT

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        """Load explicit and secret values before committed YAML defaults."""
        return (
            init_settings,
            env_settings,
            dotenv_settings,
            YamlConfigSettingsSource(settings_cls, yaml_file=cls.yaml_path),
            file_secret_settings,
        )


settings = Settings()
