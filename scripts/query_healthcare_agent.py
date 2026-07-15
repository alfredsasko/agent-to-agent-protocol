"""Send one prompt to the locally running healthcare A2A orchestrator."""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from beeai_framework.adapters.a2a.agents import A2AAgent  # noqa: E402
from beeai_framework.memory import UnconstrainedMemory  # noqa: E402

from gcp_auth import load_local_environment  # noqa: E402


async def query(prompt: str) -> None:
    load_local_environment()
    host = os.getenv("AGENT_HOST", "127.0.0.1")
    port = os.getenv("HEALTHCARE_AGENT_PORT", "9996")
    agent = A2AAgent(
        url=f"http://{host}:{port}",
        memory=UnconstrainedMemory(),
    )
    response = await agent.run(prompt)
    print(response.last_message.text)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "prompt",
        nargs="?",
        default=(
            "I'm based in Austin, TX. How do I get mental health therapy near "
            "me and what does my insurance cover?"
        ),
    )
    args = parser.parse_args()
    asyncio.run(query(args.prompt))


if __name__ == "__main__":
    main()
