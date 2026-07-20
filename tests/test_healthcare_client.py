"""Tests for the reusable healthcare A2A client."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import pytest

from healthcare_cli.client import HealthcareClient, HealthcareClientError
from healthcare_cli.events import WorkflowEventMapper, WorkflowEventType


@dataclass
class FakeRun:
    updates: list[object]
    answer: str

    def __post_init__(self) -> None:
        self.callbacks: list[Any] = []

    def on(self, matcher: str, callback: Any) -> FakeRun:
        assert matcher == "update"
        self.callbacks.append(callback)
        return self

    def __await__(self):
        return self._execute().__await__()

    async def _execute(self):
        for update in self.updates:
            for callback in self.callbacks:
                await callback(update, None)
        return SimpleNamespace(last_message=SimpleNamespace(text=self.answer))


class FakeAgent:
    def __init__(self, *, ready: bool = True) -> None:
        self.ready = ready
        self.prompts: list[str] = []

    async def check_agent_exists(self) -> None:
        if not self.ready:
            raise RuntimeError("offline")

    def run(self, prompt: str, **_: Any) -> FakeRun:
        self.prompts.append(prompt)
        updates = [
            {
                "type": "tool-call",
                "tool_name": "InsurancePolicyCoverageAgent",
                "args": "sensitive input is never exposed",
            },
            {
                "type": "tool-result",
                "tool_name": "InsurancePolicyCoverageAgent",
                "result": "sensitive result is never exposed",
            },
        ]
        return FakeRun(updates, "Covered with a copay.")


def test_client_returns_answer_and_safe_events() -> None:
    async def scenario() -> None:
        agent = FakeAgent()
        observed = []
        client = HealthcareClient("http://agent", agent_factory=lambda _: agent)

        result = await client.ask("What is covered?", on_event=observed.append)

        assert result.answer == "Covered with a copay."
        assert [event.type for event in result.events] == [
            WorkflowEventType.REQUEST_STARTED,
            WorkflowEventType.POLICY_REQUESTED,
            WorkflowEventType.POLICY_COMPLETED,
            WorkflowEventType.ANSWER_READY,
        ]
        assert observed == list(result.events)
        assert agent.prompts == ["What is covered?"]
        assert "sensitive" not in str(result.to_dict())

    asyncio.run(scenario())


def test_client_readiness_error_is_actionable() -> None:
    async def scenario() -> None:
        client = HealthcareClient(
            "http://agent",
            agent_factory=lambda _: FakeAgent(ready=False),
        )

        with pytest.raises(HealthcareClientError, match="stack up"):
            await client.check_ready()

    asyncio.run(scenario())


def test_clear_creates_a_new_agent() -> None:
    agents: list[FakeAgent] = []

    def factory(_: str) -> FakeAgent:
        agent = FakeAgent()
        agents.append(agent)
        return agent

    client = HealthcareClient("http://agent", agent_factory=factory)
    client.clear()

    assert len(agents) == 2


def test_event_mapper_deduplicates_trajectory_updates() -> None:
    mapper = WorkflowEventMapper()
    update = {"type": "tool-call", "tool_name": "HealthResearchAgent"}

    first = mapper.map_update(update)
    second = mapper.map_update(update)

    assert [event.type for event in first] == [WorkflowEventType.RESEARCH_REQUESTED]
    assert second == ()
