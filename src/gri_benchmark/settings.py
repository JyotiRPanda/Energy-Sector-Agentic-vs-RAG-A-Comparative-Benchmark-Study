from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class FoundryConfig:
    project_endpoint: str | None = None
    model_deployment: str | None = None
    agent_name: str | None = None
    api_key: str | None = None
    
    # Embedding & Retrieval
    embedding_deployment: str | None = None
    embedding_api_key: str | None = None

def _first_env(*keys: str) -> str | None:
    for key in keys:
        value = os.getenv(key)
        if value:
            return value
    return None


def load_env_file(env_path: str | Path = ".env") -> dict[str, str]:
    path = Path(env_path)
    if not path.exists():
        return {}

    loaded: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if not key or key in os.environ:
            continue

        os.environ[key] = value
        loaded[key] = value

    return loaded


def load_foundry_config(env_path: str | Path = ".env") -> FoundryConfig:
    load_env_file(env_path)
    return FoundryConfig(
        project_endpoint=_first_env("PROJECT_ENDPOINT", "FOUNDRY_PROJECT_ENDPOINT"),
        model_deployment=_first_env("MODEL_DEPLOYMENT", "FOUNDRY_MODEL_DEPLOYMENT"),
        agent_name=_first_env("AGENT_NAME", "FOUNDRY_AGENT_NAME"),
        api_key=_first_env("API_KEY", "FOUNDRY_API_KEY"),
        embedding_deployment=_first_env("EMBEDDING_DEPLOYMENT", "FOUNDRY_EMBEDDING_DEPLOYMENT"),
        embedding_api_key=_first_env("EMBEDDING_API_KEY", "FOUNDRY_EMBEDDING_API_KEY", "API_KEY"),
    )


def load__config(env_path: str | Path = ".env") -> FoundryConfig:
    # Backward-compatible alias kept until all call sites migrate.
    return load_foundry_config(env_path)