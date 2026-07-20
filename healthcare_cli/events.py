"""Safe, transport-neutral events emitted by a healthcare chat request."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import StrEnum
from typing import Any


class WorkflowEventType(StrEnum):
    """Public workflow states suitable for a CLI or UI."""

    REQUEST_STARTED = "request_started"
    ROUTING_STARTED = "routing_started"
    POLICY_REQUESTED = "policy_requested"
    POLICY_COMPLETED = "policy_completed"
    RESEARCH_REQUESTED = "research_requested"
    RESEARCH_COMPLETED = "research_completed"
    PROVIDER_REQUESTED = "provider_requested"
    PROVIDER_COMPLETED = "provider_completed"
    ANSWER_READY = "answer_ready"
    ERROR = "error"


@dataclass(frozen=True, slots=True)
class WorkflowEvent:
    """One sanitized workflow event with no model reasoning or tool payloads."""

    type: WorkflowEventType
    message: str
    agent: str | None = None

    def to_dict(self) -> dict[str, str | None]:
        """Serialize the event for JSON output."""
        data = asdict(self)
        data["type"] = self.type.value
        return data


@dataclass(frozen=True, slots=True)
class ChatResult:
    """Final answer together with safe events observed during the request."""

    answer: str
    events: tuple[WorkflowEvent, ...]

    def to_dict(self) -> dict[str, Any]:
        """Serialize the chat result for automation or a future UI."""
        return {
            "answer": self.answer,
            "events": [event.to_dict() for event in self.events],
        }


class WorkflowEventMapper:
    """Map BeeAI/A2A trajectory updates to a stable public event contract."""

    _TOOL_TO_AGENT = {
        "insurancepolicycoverageagent": "policy",
        "healthresearchagent": "research",
        "healthcareprovideragent": "provider",
    }

    def __init__(self) -> None:
        self._emitted: set[WorkflowEventType] = set()

    def map_update(self, update: object) -> tuple[WorkflowEvent, ...]:
        """Return new safe events found in an A2A update."""
        payload = _to_plain_data(update)
        mapped: list[WorkflowEvent] = []
        for content_type, tool_name in _tool_observations(payload):
            normalized_tool = "".join(char for char in tool_name.lower() if char.isalnum())
            if normalized_tool == "think":
                event = WorkflowEvent(
                    WorkflowEventType.ROUTING_STARTED,
                    "Selecting the specialist agents needed for this question.",
                    "Healthcare Agent",
                )
                self._append_once(mapped, event)
                continue

            agent_key = self._TOOL_TO_AGENT.get(normalized_tool)
            if agent_key:
                event = _specialist_event(agent_key, completed=content_type == "tool-result")
                self._append_once(mapped, event)
        return tuple(mapped)

    def _append_once(
        self,
        mapped: list[WorkflowEvent],
        event: WorkflowEvent,
    ) -> None:
        if event.type not in self._emitted:
            self._emitted.add(event.type)
            mapped.append(event)


def _specialist_event(agent_key: str, *, completed: bool) -> WorkflowEvent:
    names = {
        "policy": "Policy Agent",
        "research": "Research Agent",
        "provider": "Provider Agent",
    }
    verbs = {
        "policy": "Checking insurance policy coverage.",
        "research": "Researching current healthcare information.",
        "provider": "Finding matching healthcare providers.",
    }
    event_type = WorkflowEventType(f"{agent_key}_{'completed' if completed else 'requested'}")
    message = f"{names[agent_key]} completed." if completed else verbs[agent_key]
    return WorkflowEvent(event_type, message, names[agent_key])


def _to_plain_data(value: object) -> object:
    if hasattr(value, "value"):
        value = getattr(value, "value")
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json", exclude_none=True)
    if isinstance(value, tuple):
        return [_to_plain_data(item) for item in value]
    return value


def _tool_observations(value: object) -> list[tuple[str, str]]:
    observations: list[tuple[str, str]] = []
    if isinstance(value, dict):
        content_type = value.get("type")
        tool_name = value.get("tool_name") or value.get("toolName")
        if content_type in {"tool-call", "tool-result"} and isinstance(tool_name, str):
            observations.append((content_type, tool_name))
        for child in value.values():
            observations.extend(_tool_observations(child))
    elif isinstance(value, list | tuple):
        for child in value:
            observations.extend(_tool_observations(child))
    return observations
