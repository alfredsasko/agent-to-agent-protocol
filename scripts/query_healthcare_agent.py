"""Backward-compatible one-shot wrapper around the reusable healthcare CLI."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from healthcare_cli.cli import DEFAULT_PROMPT, main as healthcare_main  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "prompt",
        nargs="?",
        default=DEFAULT_PROMPT,
    )
    args = parser.parse_args()
    raise SystemExit(healthcare_main(["chat", args.prompt, "--no-events"]))


if __name__ == "__main__":
    main()
