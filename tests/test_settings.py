from __future__ import annotations

import os

from gri_benchmark.settings import load_env_file, load_foundry_config


def test_load_env_file_respects_existing_values(tmp_path, monkeypatch) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text(
        """
# sample config
PROJECT_ENDPOINT=https://example.test/api/projects/demo
MODEL_DEPLOYMENT=gpt-4o-mini
AGENT_NAME=griqa-agent
""".strip(),
        encoding="utf-8",
    )

    monkeypatch.setenv("MODEL_DEPLOYMENT", "preexisting-model")
    loaded = load_env_file(env_file)

    assert loaded["PROJECT_ENDPOINT"] == "https://example.test/api/projects/demo"
    assert loaded["AGENT_NAME"] == "griqa-agent"
    assert os.getenv("MODEL_DEPLOYMENT") == "preexisting-model"


def test_load_foundry_config_reads_values(tmp_path, monkeypatch) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text(
        """
PROJECT_ENDPOINT=https://example.test/api/projects/demo
MODEL_DEPLOYMENT=gpt-4o-mini
AGENT_NAME=griqa-agent
API_KEY=secret-value
""".strip(),
        encoding="utf-8",
    )

    monkeypatch.delenv("PROJECT_ENDPOINT", raising=False)
    monkeypatch.delenv("MODEL_DEPLOYMENT", raising=False)
    monkeypatch.delenv("AGENT_NAME", raising=False)
    monkeypatch.delenv("API_KEY", raising=False)

    config = load_foundry_config(env_file)

    assert config.project_endpoint == "https://example.test/api/projects/demo"
    assert config.model_deployment == "gpt-4o-mini"
    assert config.agent_name == "griqa-agent"
    assert config.api_key == "secret-value"


def test_load_foundry_config_supports_legacy_prefixed_keys(tmp_path, monkeypatch) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text(
        """
FOUNDRY_PROJECT_ENDPOINT=https://example.test/api/projects/demo
FOUNDRY_MODEL_DEPLOYMENT=gpt-4.1-mini
FOUNDRY_AGENT_NAME=griqa-agent-legacy
FOUNDRY_API_KEY=legacy-secret
""".strip(),
        encoding="utf-8",
    )

    monkeypatch.delenv("PROJECT_ENDPOINT", raising=False)
    monkeypatch.delenv("MODEL_DEPLOYMENT", raising=False)
    monkeypatch.delenv("AGENT_NAME", raising=False)
    monkeypatch.delenv("API_KEY", raising=False)
    monkeypatch.delenv("FOUNDRY_PROJECT_ENDPOINT", raising=False)
    monkeypatch.delenv("FOUNDRY_MODEL_DEPLOYMENT", raising=False)
    monkeypatch.delenv("FOUNDRY_AGENT_NAME", raising=False)
    monkeypatch.delenv("FOUNDRY_API_KEY", raising=False)

    config = load_foundry_config(env_file)

    assert config.project_endpoint == "https://example.test/api/projects/demo"
    assert config.model_deployment == "gpt-4.1-mini"
    assert config.agent_name == "griqa-agent-legacy"
    assert config.api_key == "legacy-secret"