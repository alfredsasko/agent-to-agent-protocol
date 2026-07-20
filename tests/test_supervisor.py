"""Tests for the local foreground process supervisor."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from healthcare_cli.config import StackConfig
from healthcare_cli.supervisor import StackProcessExited, StackSupervisor


class FakeProcess:
    _next_pid = 100

    def __init__(self, key: str, termination_log: list[str]) -> None:
        self.key = key
        self.termination_log = termination_log
        self.stdout = None
        self.pid = FakeProcess._next_pid
        FakeProcess._next_pid += 1
        self.exit_code = None
        self.poll_count = 0

    def poll(self):
        self.poll_count += 1
        if self.key == "a2a_healthcare_agent.py" and self.poll_count >= 3:
            return 17
        return self.exit_code

    def terminate(self) -> None:
        self.termination_log.append(self.key)
        self.exit_code = 0

    def kill(self) -> None:
        self.exit_code = -9

    def wait(self, timeout=None) -> int:
        if self.exit_code is None:
            raise subprocess.TimeoutExpired(self.key, timeout)
        return self.exit_code


def _touch_agent_scripts(root: Path) -> None:
    for filename in (
        "a2a_policy_agent.py",
        "a2a_research_agent.py",
        "a2a_provider_agent.py",
        "a2a_healthcare_agent.py",
    ):
        (root / filename).touch()


def test_supervisor_starts_dependencies_before_orchestrator_and_cleans_up(
    tmp_path: Path,
) -> None:
    _touch_agent_scripts(tmp_path)
    config = StackConfig.from_environment(tmp_path)
    starts: list[str] = []
    probes: list[str] = []
    terminations: list[str] = []

    def process_factory(command, **_):
        key = Path(command[-1]).name
        starts.append(key)
        return FakeProcess(key, terminations)

    def readiness_probe(endpoint) -> bool:
        probes.append(endpoint.key)
        return True

    supervisor = StackSupervisor(
        config,
        process_factory=process_factory,
        readiness_probe=readiness_probe,
        log=lambda _: None,
        poll_interval=0,
    )

    with pytest.raises(StackProcessExited, match="Healthcare Agent"):
        supervisor.run(startup_timeout=1)

    assert starts == [
        "a2a_policy_agent.py",
        "a2a_research_agent.py",
        "a2a_provider_agent.py",
        "a2a_healthcare_agent.py",
    ]
    assert probes[:3] == ["policy", "research", "provider"]
    assert probes[-1] == "healthcare"
    assert terminations == [
        "a2a_provider_agent.py",
        "a2a_research_agent.py",
        "a2a_policy_agent.py",
    ]
