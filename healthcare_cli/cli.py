"""Command-line interface for chatting with and supervising the A2A stack."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from collections.abc import Callable, Sequence
from typing import TextIO

from healthcare_cli.client import HealthcareClient, HealthcareClientError
from healthcare_cli.config import StackConfig
from healthcare_cli.events import WorkflowEvent
from healthcare_cli.supervisor import (
    StackProcessExited,
    StackStartupError,
    StackSupervisor,
    stack_status,
)

InputFunction = Callable[[str], str]
ClientFactory = Callable[[str], HealthcareClient]

DEFAULT_PROMPT = (
    "I'm based in Austin, TX. How do I get mental health therapy near me "
    "and what does my insurance cover?"
)


def build_parser() -> argparse.ArgumentParser:
    """Construct the top-level argument parser."""
    parser = argparse.ArgumentParser(
        prog="healthcare",
        description="Run and query the local healthcare A2A workflow.",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    chat = commands.add_parser("chat", help="Open a healthcare chat session.")
    chat.add_argument("prompt", nargs="?", help="Send one prompt and exit.")
    chat.add_argument("--json", action="store_true", help="Output machine-readable JSON.")
    chat.add_argument(
        "--no-events",
        action="store_true",
        help="Hide safe workflow progress events.",
    )

    commands.add_parser("check", help="Check whether all four agents are ready.")

    stack = commands.add_parser("stack", help="Manage the foreground local stack.")
    stack_commands = stack.add_subparsers(dest="stack_command", required=True)
    stack_up = stack_commands.add_parser("up", help="Start all four A2A agents.")
    stack_up.add_argument(
        "--startup-timeout",
        type=float,
        default=90.0,
        help="Seconds to wait for each startup phase.",
    )
    stack_commands.add_parser("status", help="Check the four A2A agents.")
    return parser


async def run_chat(
    args: argparse.Namespace,
    config: StackConfig,
    *,
    client_factory: ClientFactory = HealthcareClient,
    input_fn: InputFunction = input,
    output: TextIO = sys.stdout,
) -> int:
    """Run a one-shot query or an interactive conversation."""
    client = client_factory(config.healthcare.url)
    await client.check_ready()

    if args.prompt:
        await _ask_and_render(client, args.prompt, args, output)
        return 0

    print("Healthcare A2A chat", file=output)
    print("Demo only. Do not enter personal health information.", file=output)
    print("Commands: :clear, :help, :quit", file=output)

    while True:
        try:
            prompt = input_fn("You: ").strip()
        except EOFError:
            print(file=output)
            return 0

        if not prompt:
            continue
        if prompt in {":quit", ":q", ":exit"}:
            return 0
        if prompt == ":clear":
            client.clear()
            print("Conversation cleared.", file=output)
            continue
        if prompt == ":help":
            print("Use :clear to reset context and :quit to exit.", file=output)
            continue

        await _ask_and_render(client, prompt, args, output)


async def _ask_and_render(
    client: HealthcareClient,
    prompt: str,
    args: argparse.Namespace,
    output: TextIO,
) -> None:
    def handle_event(event: WorkflowEvent) -> None:
        if not args.json and not args.no_events:
            print(f"[{event.agent or 'Workflow'}] {event.message}", file=output)

    result = await client.ask(prompt, on_event=handle_event)
    if args.json:
        print(json.dumps(result.to_dict(), ensure_ascii=False), file=output)
    else:
        print(f"\nAssistant: {result.answer}\n", file=output)


def _print_status(config: StackConfig, output: TextIO) -> int:
    statuses = stack_status(config)
    endpoint_by_key = {endpoint.key: endpoint for endpoint in config.all_agents}
    for key, ready in statuses.items():
        endpoint = endpoint_by_key[key]
        state = "ready" if ready else "not ready"
        print(f"{endpoint.display_name}: {state} ({endpoint.url})", file=output)
    return 0 if all(statuses.values()) else 1


def main(argv: Sequence[str] | None = None) -> int:
    """Run the CLI and return a process exit code."""
    args = build_parser().parse_args(argv)
    config = StackConfig.from_environment()
    try:
        if args.command == "chat":
            return asyncio.run(run_chat(args, config))
        if args.command == "check" or getattr(args, "stack_command", None) == "status":
            return _print_status(config, sys.stdout)
        if args.command == "stack" and args.stack_command == "up":
            StackSupervisor(config).run(startup_timeout=args.startup_timeout)
            return 0
    except (HealthcareClientError, StackStartupError, StackProcessExited) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        return 130
    return 2
