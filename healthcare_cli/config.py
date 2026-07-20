"""Environment-backed configuration for the local A2A stack."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True, slots=True)
class AgentEndpoint:
    """Connection and process metadata for one A2A agent."""

    key: str
    display_name: str
    url: str
    script: Path

    @property
    def agent_card_url(self) -> str:
        """Return the standard A2A Agent Card URL."""
        return f"{self.url.rstrip('/')}/.well-known/agent-card.json"


@dataclass(frozen=True, slots=True)
class StackConfig:
    """Resolved configuration for all local agent processes."""

    project_root: Path
    policy: AgentEndpoint
    research: AgentEndpoint
    provider: AgentEndpoint
    healthcare: AgentEndpoint

    @property
    def dependencies(self) -> tuple[AgentEndpoint, ...]:
        """Return agents that must be ready before the orchestrator starts."""
        return (self.policy, self.research, self.provider)

    @property
    def all_agents(self) -> tuple[AgentEndpoint, ...]:
        """Return agents in startup order."""
        return (*self.dependencies, self.healthcare)

    @classmethod
    def from_environment(cls, project_root: Path | None = None) -> StackConfig:
        """Load ``.env`` and resolve URLs without overriding exported values."""
        root = (project_root or PROJECT_ROOT).resolve()
        load_dotenv(root / ".env", override=False)
        host = _client_host(os.getenv("AGENT_HOST", "127.0.0.1"))

        return cls(
            project_root=root,
            policy=_endpoint(
                root,
                key="policy",
                display_name="Policy Agent",
                script="a2a_policy_agent.py",
                url_env="POLICY_AGENT_URL",
                host=host,
                port_env="POLICY_AGENT_PORT",
                default_port="9999",
            ),
            research=_endpoint(
                root,
                key="research",
                display_name="Research Agent",
                script="a2a_research_agent.py",
                url_env="RESEARCH_AGENT_URL",
                host=host,
                port_env="RESEARCH_AGENT_PORT",
                default_port="9998",
            ),
            provider=_endpoint(
                root,
                key="provider",
                display_name="Provider Agent",
                script="a2a_provider_agent.py",
                url_env="PROVIDER_AGENT_URL",
                host=host,
                port_env="PROVIDER_AGENT_PORT",
                default_port="9997",
            ),
            healthcare=_endpoint(
                root,
                key="healthcare",
                display_name="Healthcare Agent",
                script="a2a_healthcare_agent.py",
                url_env="HEALTHCARE_AGENT_URL",
                host=host,
                port_env="HEALTHCARE_AGENT_PORT",
                default_port="9996",
            ),
        )


def _endpoint(
    root: Path,
    *,
    key: str,
    display_name: str,
    script: str,
    url_env: str,
    host: str,
    port_env: str,
    default_port: str,
) -> AgentEndpoint:
    raw_url = os.getenv(url_env) or f"http://{host}:{os.getenv(port_env, default_port)}"
    url = _validated_http_url(raw_url, url_env)
    return AgentEndpoint(
        key=key,
        display_name=display_name,
        url=url.rstrip("/"),
        script=root / script,
    )


def _client_host(host: str) -> str:
    """Translate wildcard bind addresses into usable local client addresses."""
    normalized = host.strip().strip('"').strip("'")
    if normalized in {"", "0.0.0.0", "::"}:
        return "127.0.0.1"
    return normalized


def _validated_http_url(value: str, setting_name: str) -> str:
    normalized = value.strip().rstrip("/")
    parsed = urlparse(normalized)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError(f"{setting_name} must be an absolute HTTP(S) URL.")
    return normalized
