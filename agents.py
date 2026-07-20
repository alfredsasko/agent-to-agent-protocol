import base64
import os
import sys
from pathlib import Path

from anthropic import AnthropicVertex
from anthropic.types import (
    Base64PDFSourceParam,
    DocumentBlockParam,
    MessageParam,
    TextBlockParam,
)
from google import genai
from google.genai import types
from langchain.agents import create_agent
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.sessions import StdioConnection
from langchain_openai import ChatOpenAI

from helpers import GoogleAccessTokenProvider, authenticate, vertex_api_base

PROJECT_ROOT = Path(__file__).resolve().parent
POLICY_SYSTEM_PROMPT = (
    "You are an expert insurance agent designed to assist with coverage queries. "
    "Use the provided documents to answer questions about insurance policies. "
    "If the information is not available in the documents, respond with "
    "'I don't know'"
)


def _policy_document_path() -> Path:
    return Path(
        os.getenv(
            "POLICY_DOCUMENT_PATH",
            str(PROJECT_ROOT / "data" / "2026AnthemgHIPSBC.pdf"),
        )
    ).expanduser()


def _escape_markdown_dollars(text: str) -> str:
    return text.replace("$", r"\$")


class PolicyAgent:
    """Policy agent facade with configurable Vertex AI backend."""

    def __init__(self) -> None:
        backend = os.getenv("POLICY_AGENT_BACKEND", "google").strip().lower()
        if backend == "anthropic":
            self.backend = AnthropicVertexPolicyBackend()
        elif backend == "google":
            self.backend = GoogleVertexPolicyBackend()
        else:
            raise ValueError(
                "Unsupported POLICY_AGENT_BACKEND. Use 'google' or 'anthropic'."
            )

    def answer_query(self, prompt: str) -> str:
        return self.backend.answer_query(prompt)


class AnthropicVertexPolicyBackend:
    """Anthropic Claude policy reader served through Vertex AI."""

    def __init__(self) -> None:
        location = os.getenv("ANTHROPIC_VERTEX_LOCATION", "global")
        credentials, project_id = authenticate(location=location)
        self.client = AnthropicVertex(
            project_id=project_id,
            region=location,
            credentials=credentials,
        )
        with _policy_document_path().open("rb") as file:
            self.pdf_data = base64.standard_b64encode(file.read()).decode("utf-8")

    def answer_query(self, prompt: str) -> str:
        response = self.client.messages.create(
            model=os.getenv("ANTHROPIC_VERTEX_MODEL", "claude-haiku-4-5@20251001"),
            max_tokens=1024,
            system=POLICY_SYSTEM_PROMPT,
            messages=[
                MessageParam(
                    role="user",
                    content=[
                        DocumentBlockParam(
                            type="document",
                            source=Base64PDFSourceParam(
                                type="base64",
                                media_type="application/pdf",
                                data=self.pdf_data,
                            ),
                        ),
                        TextBlockParam(
                            type="text",
                            text=prompt,
                        ),
                    ],
                )
            ],
        )

        return _escape_markdown_dollars(response.content[0].text)


class GoogleVertexPolicyBackend:
    """Lightweight Gemini policy reader served through Vertex AI."""

    def __init__(self) -> None:
        location = os.getenv("POLICY_GOOGLE_VERTEX_LOCATION") or os.getenv(
            "GOOGLE_CLOUD_LOCATION",
            "global",
        )
        credentials, project_id = authenticate(location=location)
        self.client = genai.Client(
            vertexai=True,
            credentials=credentials,
            project=project_id,
            location=location,
        )
        self.model = os.getenv("POLICY_GOOGLE_VERTEX_MODEL", "gemini-2.5-flash-lite")
        self.pdf_data = _policy_document_path().read_bytes()

    def answer_query(self, prompt: str) -> str:
        response = self.client.models.generate_content(
            model=self.model,
            contents=[
                types.Part.from_bytes(
                    data=self.pdf_data,
                    mime_type="application/pdf",
                ),
                prompt,
            ],
            config=types.GenerateContentConfig(
                system_instruction=POLICY_SYSTEM_PROMPT,
                max_output_tokens=1024,
                temperature=0.1,
            ),
        )

        text = response.text
        if not text:
            raise RuntimeError("The Google Vertex model returned an empty response.")
        return _escape_markdown_dollars(text)


class ProviderAgent:
    def __init__(self) -> None:
        credentials, project_id = authenticate()
        location = os.getenv("PROVIDER_VERTEX_LOCATION", "us-central1")
        base_url = (
            f"{vertex_api_base(location)}/v1/projects/{project_id}/locations/"
            f"{location}/endpoints/openapi"
        )

        self.mcp_client = MultiServerMCPClient(
            {
                "find_healthcare_providers": StdioConnection(
                    transport="stdio",
                    command=sys.executable,
                    args=[str(PROJECT_ROOT / "mcpserver.py")],
                )
            }
        )

        self.credentials = credentials
        self.base_url = base_url
        self.agent = None

    async def initialize(self):
        """Initialize the agent asynchronously."""
        tools = await self.mcp_client.get_tools()
        self.agent = create_agent(
            ChatOpenAI(
                model=os.getenv("PROVIDER_MODEL", "openai/gpt-oss-20b-maas"),
                openai_api_key=GoogleAccessTokenProvider(self.credentials),
                openai_api_base=self.base_url,
            ),
            tools,
            name="HealthcareProviderAgent",
            system_prompt="Your task is to find and list providers using the find_healthcare_providers MCP Tool based on the users query. Only use providers based on the response from the tool. Output the information in a table.",
        )
        return self

    async def answer_query(self, prompt: str) -> str:
        if self.agent is None:
            raise RuntimeError("Agent not initialized. Call initialize() first.")

        response = await self.agent.ainvoke(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ]
            }
        )
        return response["messages"][-1].content
