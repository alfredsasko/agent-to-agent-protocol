"""Tests for local stack configuration."""

from pathlib import Path

import pytest

from healthcare_cli.config import StackConfig


def test_stack_config_uses_ports_and_normalizes_wildcard_host(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.delenv("HEALTHCARE_AGENT_URL", raising=False)
    monkeypatch.setenv("AGENT_HOST", "0.0.0.0")
    monkeypatch.setenv("HEALTHCARE_AGENT_PORT", "8996")

    config = StackConfig.from_environment(tmp_path)

    assert config.healthcare.url == "http://127.0.0.1:8996"
    assert config.healthcare.agent_card_url.endswith("/.well-known/agent-card.json")


def test_stack_config_prefers_explicit_agent_url(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("HEALTHCARE_AGENT_URL", "https://agents.example/healthcare/")

    config = StackConfig.from_environment(tmp_path)

    assert config.healthcare.url == "https://agents.example/healthcare"


def test_stack_config_rejects_non_http_agent_url(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("HEALTHCARE_AGENT_URL", "file:///tmp/agent-card.json")

    with pytest.raises(ValueError, match="HEALTHCARE_AGENT_URL"):
        StackConfig.from_environment(tmp_path)
