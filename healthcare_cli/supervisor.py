"""Foreground process supervisor for the four local A2A servers."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import IO, Any, Protocol

from healthcare_cli.config import AgentEndpoint, StackConfig

LogHandler = Callable[[str], None]
ReadinessProbe = Callable[[AgentEndpoint], bool]


class ProcessLike(Protocol):
    """Subset of ``subprocess.Popen`` used by the supervisor."""

    pid: int
    stdout: IO[str] | None

    def poll(self) -> int | None:
        """Return the exit code or ``None`` while running."""
        ...

    def terminate(self) -> None:
        """Request graceful termination."""
        ...

    def kill(self) -> None:
        """Force termination."""
        ...

    def wait(self, timeout: float | None = None) -> int:
        """Wait for process termination."""
        ...


ProcessFactory = Callable[..., ProcessLike]


@dataclass(frozen=True, slots=True)
class ManagedProcess:
    """One endpoint and its running child process."""

    endpoint: AgentEndpoint
    process: ProcessLike


class StackStartupError(RuntimeError):
    """Raised when a local service cannot start or become ready."""


class StackProcessExited(RuntimeError):
    """Raised when a required child exits while the stack is running."""


class StackSupervisor:
    """Start dependencies, wait for readiness, and own their lifecycle."""

    def __init__(
        self,
        config: StackConfig,
        *,
        process_factory: ProcessFactory | None = None,
        readiness_probe: ReadinessProbe | None = None,
        log: LogHandler | None = None,
        poll_interval: float = 0.25,
    ) -> None:
        self.config = config
        self._process_factory = process_factory or subprocess.Popen
        self._readiness_probe = readiness_probe or probe_agent_card
        self._log = log or print
        self._poll_interval = poll_interval
        self._managed: list[ManagedProcess] = []
        self._owns_process_groups = process_factory is None and os.name != "nt"

    @property
    def managed(self) -> tuple[ManagedProcess, ...]:
        """Return processes currently owned by the supervisor."""
        return tuple(self._managed)

    def run(self, *, startup_timeout: float = 90.0) -> None:
        """Run the stack in the foreground until interrupted or a child exits."""
        self._validate_scripts()
        try:
            for endpoint in self.config.dependencies:
                self._start(endpoint)
            self._wait_until_ready(self.config.dependencies, startup_timeout)

            self._start(self.config.healthcare)
            self._wait_until_ready((self.config.healthcare,), startup_timeout)
            self._log("All four A2A agents are ready. Press Ctrl+C to stop.")
            self._monitor()
        except KeyboardInterrupt:
            self._log("Stopping the local A2A stack...")
        finally:
            self.shutdown()

    def shutdown(self, *, timeout: float = 8.0) -> None:
        """Terminate all owned processes in reverse startup order."""
        running = [item for item in reversed(self._managed) if item.process.poll() is None]
        for item in running:
            self._terminate(item.process)

        deadline = time.monotonic() + timeout
        for item in running:
            remaining = max(0.0, deadline - time.monotonic())
            try:
                item.process.wait(timeout=remaining)
            except subprocess.TimeoutExpired:
                self._kill(item.process)
                item.process.wait(timeout=2.0)
        self._managed.clear()

    def _validate_scripts(self) -> None:
        missing = [
            str(endpoint.script)
            for endpoint in self.config.all_agents
            if not endpoint.script.is_file()
        ]
        if missing:
            raise StackStartupError(f"Agent scripts not found: {', '.join(missing)}")

    def _start(self, endpoint: AgentEndpoint) -> None:
        command = [sys.executable, str(endpoint.script)]
        kwargs: dict[str, Any] = {
            "cwd": self.config.project_root,
            "env": os.environ.copy(),
            "stdout": subprocess.PIPE,
            "stderr": subprocess.STDOUT,
            "text": True,
            "bufsize": 1,
        }
        if self._owns_process_groups:
            kwargs["start_new_session"] = True

        self._log(f"Starting {endpoint.display_name}...")
        try:
            process = self._process_factory(command, **kwargs)
        except OSError as exc:
            raise StackStartupError(f"Could not start {endpoint.display_name}: {exc}") from exc

        managed = ManagedProcess(endpoint, process)
        self._managed.append(managed)
        if process.stdout is not None:
            threading.Thread(
                target=self._forward_logs,
                args=(managed,),
                daemon=True,
                name=f"{endpoint.key}-logs",
            ).start()

    def _wait_until_ready(
        self,
        endpoints: Iterable[AgentEndpoint],
        timeout: float,
    ) -> None:
        pending = {endpoint.key: endpoint for endpoint in endpoints}
        deadline = time.monotonic() + timeout
        while pending:
            self._raise_for_exited_process()
            for key, endpoint in tuple(pending.items()):
                if self._readiness_probe(endpoint):
                    self._log(f"{endpoint.display_name} is ready at {endpoint.url}")
                    pending.pop(key)
            if pending and time.monotonic() >= deadline:
                names = ", ".join(endpoint.display_name for endpoint in pending.values())
                raise StackStartupError(f"Timed out waiting for: {names}")
            if pending:
                time.sleep(self._poll_interval)

    def _monitor(self) -> None:
        while True:
            self._raise_for_exited_process()
            time.sleep(self._poll_interval)

    def _raise_for_exited_process(self) -> None:
        for item in self._managed:
            exit_code = item.process.poll()
            if exit_code is not None:
                raise StackProcessExited(
                    f"{item.endpoint.display_name} exited unexpectedly with code {exit_code}."
                )

    def _forward_logs(self, item: ManagedProcess) -> None:
        assert item.process.stdout is not None
        for line in item.process.stdout:
            self._log(f"[{item.endpoint.key}] {line.rstrip()}")

    def _terminate(self, process: ProcessLike) -> None:
        try:
            if self._owns_process_groups:
                os.killpg(process.pid, signal.SIGTERM)
            else:
                process.terminate()
        except ProcessLookupError:
            return

    def _kill(self, process: ProcessLike) -> None:
        try:
            if self._owns_process_groups:
                os.killpg(process.pid, signal.SIGKILL)
            else:
                process.kill()
        except ProcessLookupError:
            return


def probe_agent_card(endpoint: AgentEndpoint, *, timeout: float = 1.0) -> bool:
    """Return whether an endpoint publishes a valid A2A Agent Card."""
    try:
        with urllib.request.urlopen(  # noqa: S310
            endpoint.agent_card_url,
            timeout=timeout,
        ) as response:
            if response.status != 200:
                return False
            payload = json.loads(response.read())
            return isinstance(payload, dict) and bool(payload.get("name"))
    except (OSError, ValueError, urllib.error.URLError):
        return False


def stack_status(config: StackConfig) -> dict[str, bool]:
    """Probe every configured agent without starting any processes."""
    return {endpoint.key: probe_agent_card(endpoint) for endpoint in config.all_agents}
