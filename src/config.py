from __future__ import annotations

import os
from pathlib import Path

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # LLM
    openai_api_key: str = ""
    anthropic_api_key: str = ""
    agent_model: str = "gpt-4o-mini"
    meta_model: str = "gpt-4o"

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
