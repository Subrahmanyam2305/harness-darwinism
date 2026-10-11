from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Bedrock models
    agent_model: str = "us.anthropic.claude-haiku-4-5-20251001-v1:0"
    meta_model: str = "us.anthropic.claude-opus-4-6-v1"
    aws_region: str = "us-east-1"

    # MongoDB
    mongodb_uri: str = "mongodb://localhost:27017"
    db_name: str = "harness_darwinism"

    # Evolution
    population_size: int = 10
    survivors: int = 5
    seeds_per_gen: int = 5
    max_generations: int = 10
    max_steps_per_episode: int = 200
    archive_sample_ratio: float = 0.1

    # Crafter
    crafter_size: tuple[int, int] = (64, 64)

    # Paths
    project_root: Path = Path(__file__).resolve().parent.parent

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


settings = Settings()
