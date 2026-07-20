import asyncio
import os
from typing import Any

from beeai_framework.adapters.a2a.agents import A2AAgent
from beeai_framework.adapters.a2a.serve.server import A2AServer, A2AServerConfig
from beeai_framework.adapters.vertexai import VertexAIChatModel
from beeai_framework.agents.requirement import RequirementAgent
from beeai_framework.agents.requirement.requirements.conditional import (
    ConditionalRequirement,
)
from beeai_framework.memory import UnconstrainedMemory
from beeai_framework.middleware.trajectory import EventMeta, GlobalTrajectoryMiddleware
from beeai_framework.serve.utils import LRUMemoryManager
from beeai_framework.tools import Tool
from beeai_framework.tools.handoff import HandoffTool
from beeai_framework.tools.think import ThinkTool
from dotenv import load_dotenv

from helpers import authenticate


class ConciseGlobalTrajectoryMiddleware(GlobalTrajectoryMiddleware):
    """Format trajectory output with tool names and no payload details."""

    def _format_prefix(self, meta: EventMeta) -> str:
        prefix = super()._format_prefix(meta)
        return prefix.rstrip(": ")

    def _format_payload(self, value: Any) -> str:
        return ""


def main() -> None:
    print("Running A2A Orchestrator Agent")
    load_dotenv()
    _, project_id = authenticate()

    host = os.getenv("AGENT_HOST", "127.0.0.1")
    policy_agent_port = os.getenv("POLICY_AGENT_PORT", "9999")
    research_agent_port = os.getenv("RESEARCH_AGENT_PORT", "9998")
    provider_agent_port = os.getenv("PROVIDER_AGENT_PORT", "9997")
    healthcare_agent_port = int(os.getenv("HEALTHCARE_AGENT_PORT", "9996"))

    GlobalTrajectoryMiddleware(target=[Tool])

    policy_agent = A2AAgent(
        url=f"http://{host}:{policy_agent_port}",
        memory=UnconstrainedMemory(),
    )
    asyncio.run(policy_agent.check_agent_exists())
    print("\tℹ️", f"{policy_agent.name} initialized")

    research_agent = A2AAgent(
        url=f"http://{host}:{research_agent_port}",
        memory=UnconstrainedMemory(),
    )
    asyncio.run(research_agent.check_agent_exists())
    print("\tℹ️", f"{research_agent.name} initialized")

    provider_agent = A2AAgent(
        url=f"http://{host}:{provider_agent_port}",
        memory=UnconstrainedMemory(),
    )
    asyncio.run(provider_agent.check_agent_exists())
    print("\tℹ️", f"{provider_agent.name} initialized")

    think_tool = ThinkTool()
    policy_tool = HandoffTool(
        target=policy_agent,
        name=policy_agent.name,
        description=(
            "Use this agent for questions about the configured insurance-policy "
            "document, including coverage, benefits, copays, deductibles, "
            "out-of-pocket costs, exclusions, network rules, and prior authorization. "
            "This agent is the source of truth for policy information."
        ),
    )
    research_tool = HandoffTool(
        target=research_agent,
        name=research_agent.name,
        description=research_agent.agent_card.description,
    )
    provider_tool = HandoffTool(
        target=provider_agent,
        name=provider_agent.name,
        description=provider_agent.agent_card.description,
    )

    healthcare_agent = RequirementAgent(
        name="Healthcare Agent",
        description=(
            "A personal concierge for Healthcare Information, customized to your "
            "policy."
        ),
        llm=VertexAIChatModel(
            model_id=os.getenv("HEALTHCARE_MODEL", "gemini-2.5-flash"),
            project=project_id,
            location=os.getenv("HEALTHCARE_VERTEX_LOCATION", "global"),
            allow_parallel_tool_calls=True,
        ),
        tools=[think_tool, policy_tool, research_tool, provider_tool],
        requirements=[
            ConditionalRequirement(policy_tool, consecutive_allowed=False),
            ConditionalRequirement(
                think_tool,
                force_at_step=1,
                force_after=Tool,
                consecutive_allowed=False,
            ),
        ],
        role="Healthcare Concierge",
        instructions=f"""You are a healthcare concierge. Coordinate the specialist
        agents to answer the user's question accurately and concisely.

        For any question about insurance, a policy, coverage, benefits, deductibles,
        copays, out-of-pocket cost, exclusions, network rules, prior authorization, or
        whether a service is covered, use `{policy_agent.name}` before giving a final
        answer.

        Treat phrases such as "my policy", "this policy", "existing policy", and
        "our policy" as referring to the configured policy document available to
        `{policy_agent.name}`. Do not state that you cannot access the policy or that
        `{policy_agent.name}` is unavailable without first consulting that agent.

        Use only information returned by `{policy_agent.name}` for policy, coverage,
        benefit, or cost claims. If it cannot find the answer in the configured document,
        clearly say that the information was not found; do not guess or provide general
        insurance advice.

        For provider questions, use `{provider_agent.name}` and only list providers it
        returns. For health-information questions, use `{research_agent.name}`.

        When more than one specialist is needed, combine their answers clearly. State
        which specialist supplied policy, provider, or research information. Do not
        mention internal tool names unless the user asks.
        """,
    )
    print("\tℹ️", f"{healthcare_agent.meta.name} initialized")

    A2AServer(
        config=A2AServerConfig(
            port=healthcare_agent_port,
            protocol="jsonrpc",
            host=host,
        ),
        memory_manager=LRUMemoryManager(maxsize=100),
    ).register(healthcare_agent, send_trajectory=True).serve()


if __name__ == "__main__":
    main()
