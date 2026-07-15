import logging
import os
import warnings

import uvicorn
from dotenv import load_dotenv
from google.adk.a2a.utils.agent_to_a2a import to_a2a
from google.adk.agents import LlmAgent
from google.adk.tools import google_search

from helpers import authenticate

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

load_dotenv()


def build_research_agent() -> LlmAgent:
    """Configure the ADK agent after ADC and environment setup succeeds."""
    authenticate(location=os.getenv("RESEARCH_VERTEX_LOCATION", "global"))
    return LlmAgent(
        model=os.getenv("RESEARCH_MODEL", "gemini-2.5-flash"),
        name="HealthResearchAgent",
        tools=[google_search],
        description="Provides healthcare information about symptoms, health "
        "conditions, treatments, and procedures using up-to-date web resources.",
        instruction="""You are a healthcare research agent tasked with
        providing information about health conditions. Use the google_search
        tool to find information on the web about options, symptoms, treatments,
        and procedures. Cite your sources in your responses. Output all of the
        information you find.""",
    )


def main() -> None:
    host = os.getenv("AGENT_HOST", "127.0.0.1")
    port = int(os.getenv("RESEARCH_AGENT_PORT", "9998"))
    research_agent = build_research_agent()
    # Make your agent A2A-compatible
    a2a_app = to_a2a(research_agent, host=host, port=port)
    print(f"Running Health Research Agent on http://{host}:{port}", flush=True)
    uvicorn.run(a2a_app, host=host, port=port, log_level="info")


if __name__ == "__main__":
    main()
