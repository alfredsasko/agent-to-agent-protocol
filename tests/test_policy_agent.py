"""Tests for configurable policy agent backends."""

from unittest.mock import Mock

from agents import GoogleVertexPolicyBackend, PolicyAgent


def test_defaults_to_google_backend(monkeypatch) -> None:
    google_backend = Mock()
    monkeypatch.delenv("POLICY_AGENT_BACKEND", raising=False)
    monkeypatch.setattr("agents.GoogleVertexPolicyBackend", google_backend)

    agent = PolicyAgent()

    assert agent.backend is google_backend.return_value


def test_can_select_anthropic_backend(monkeypatch) -> None:
    anthropic_backend = Mock()
    monkeypatch.setenv("POLICY_AGENT_BACKEND", "anthropic")
    monkeypatch.setattr("agents.AnthropicVertexPolicyBackend", anthropic_backend)

    agent = PolicyAgent()

    assert agent.backend is anthropic_backend.return_value


def test_google_backend_uses_configured_vertex_model(monkeypatch) -> None:
    client = Mock()
    client.models.generate_content.return_value.text = "The visit costs $20."
    genai_client = Mock(return_value=client)
    monkeypatch.setenv("POLICY_GOOGLE_VERTEX_MODEL", "gemini-test-model")
    monkeypatch.setenv("POLICY_GOOGLE_VERTEX_LOCATION", "us-central1")
    monkeypatch.setattr("agents.authenticate", Mock(return_value=(Mock(), "test-project")))
    monkeypatch.setattr("agents.genai.Client", genai_client)

    backend = GoogleVertexPolicyBackend()
    result = backend.answer_query("What is the office visit cost?")

    genai_client.assert_called_once()
    assert client.models.generate_content.call_args.kwargs["model"] == "gemini-test-model"
    assert result == r"The visit costs \$20."
