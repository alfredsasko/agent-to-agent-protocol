"""Reusable asynchronous A2A client for the healthcare orchestrator."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Protocol

from beeai_framework.adapters.a2a.agents import A2AAgent
from beeai_framework.memory import UnconstrainedMemory

from healthcare_cli.events import (
    ChatResult,
    WorkflowEvent,
    WorkflowEventMapper,
    WorkflowEventType,
)

EventHandler = Callable[[WorkflowEvent], None]


class AgentRun(Protocol):
    """Awaitable BeeAI run with observable events."""

    def on(self, matcher: str, callback: Callable[..., Any]) -> AgentRun:
        """Register an event callback and return this run."""
        ...

    def __await__(self) -> Any:
        """Await the final agent output."""
        ...


class AgentLike(Protocol):
    """Minimal interface used from BeeAI's A2A agent."""

    async def check_agent_exists(self) -> None:
        """Raise when the remote Agent Card is unavailable."""
        ...

    def run(self, prompt: str, **kwargs: Any) -> AgentRun:
        """Start one agent run."""
        ...


AgentFactory = Callable[[str], AgentLike]


class HealthcareClientError(RuntimeError):
    """A user-facing healthcare client failure."""


class HealthcareClient:
    """Maintain one conversation with the healthcare A2A orchestrator."""

    def __init__(
        self,
        url: str,
        *,
        agent_factory: AgentFactory | None = None,
    ) -> None:
        self.url = url.rstrip("/")
        self._agent_factory = agent_factory or _default_agent_factory
        self._agent = self._agent_factory(self.url)

    async def check_ready(self) -> None:
        """Verify that the healthcare orchestrator publishes an Agent Card."""
        try:
            await self._agent.check_agent_exists()
        except Exception as exc:
            raise HealthcareClientError(
                f"Healthcare Agent is not ready at {self.url}. "
                "Start it with `python -m healthcare_cli stack up`."
            ) from exc

    def clear(self) -> None:
        """Start a new local and remote A2A conversation context."""
        self._agent = self._agent_factory(self.url)

    async def ask(
        self,
        prompt: str,
        *,
        on_event: EventHandler | None = None,
    ) -> ChatResult:
        """Send one prompt and return the answer with safe workflow events."""
        cleaned_prompt = prompt.strip()
        if not cleaned_prompt:
            raise ValueError("Prompt cannot be empty.")

        events: list[WorkflowEvent] = []
        mapper = WorkflowEventMapper()

        def emit(event: WorkflowEvent) -> None:
            events.append(event)
            if on_event:
                on_event(event)

        emit(
            WorkflowEvent(
                WorkflowEventType.REQUEST_STARTED,
                "Request accepted.",
                "Healthcare Agent",
            )
        )

        async def handle_update(data: object, *_: object) -> None:
            for event in mapper.map_update(data):
                emit(event)

        try:
            run = self._agent.run(cleaned_prompt)
            run.on("update", handle_update)
            response = await run
            answer = response.last_message.text
        except Exception as exc:
            error = WorkflowEvent(
                WorkflowEventType.ERROR,
                "The healthcare workflow could not complete the request.",
                "Healthcare Agent",
            )
            emit(error)
            raise HealthcareClientError(error.message) from exc

        emit(
            WorkflowEvent(
                WorkflowEventType.ANSWER_READY,
                "Final answer ready.",
                "Healthcare Agent",
            )
        )
        return ChatResult(answer=answer, events=tuple(events))


def _default_agent_factory(url: str) -> AgentLike:
    return A2AAgent(url=url, memory=UnconstrainedMemory())
