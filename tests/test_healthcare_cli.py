"""Tests for interactive and JSON CLI behavior."""

import asyncio
from argparse import Namespace
from io import StringIO

from healthcare_cli.cli import run_chat
from healthcare_cli.config import StackConfig
from healthcare_cli.events import ChatResult, WorkflowEvent, WorkflowEventType


class FakeClient:
    def __init__(self, _: str) -> None:
        self.ready_checks = 0
        self.clears = 0
        self.prompts: list[str] = []

    async def check_ready(self) -> None:
        self.ready_checks += 1

    def clear(self) -> None:
        self.clears += 1

    async def ask(self, prompt: str, *, on_event=None) -> ChatResult:
        self.prompts.append(prompt)
        event = WorkflowEvent(
            WorkflowEventType.ANSWER_READY,
            "Final answer ready.",
            "Healthcare Agent",
        )
        if on_event:
            on_event(event)
        return ChatResult("Test answer", (event,))


def _args(prompt=None, *, json_output=False, no_events=False) -> Namespace:
    return Namespace(prompt=prompt, json=json_output, no_events=no_events)


def test_interactive_chat_supports_clear_and_quit(tmp_path) -> None:
    config = StackConfig.from_environment(tmp_path)
    client = FakeClient(config.healthcare.url)
    entries = iter(["question", ":clear", ":help", ":quit"])
    output = StringIO()

    exit_code = asyncio.run(
        run_chat(
            _args(),
            config,
            client_factory=lambda _: client,
            input_fn=lambda _: next(entries),
            output=output,
        )
    )

    assert exit_code == 0
    assert client.ready_checks == 1
    assert client.prompts == ["question"]
    assert client.clears == 1
    assert "Do not enter personal health information" in output.getvalue()
    assert "Test answer" in output.getvalue()


def test_one_shot_json_output(tmp_path) -> None:
    config = StackConfig.from_environment(tmp_path)
    client = FakeClient(config.healthcare.url)
    output = StringIO()

    exit_code = asyncio.run(
        run_chat(
            _args("question", json_output=True),
            config,
            client_factory=lambda _: client,
            output=output,
        )
    )

    assert exit_code == 0
    assert '"answer": "Test answer"' in output.getvalue()
    assert '"type": "answer_ready"' in output.getvalue()
