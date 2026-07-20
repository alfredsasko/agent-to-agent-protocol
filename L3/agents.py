import base64
import os
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

from helpers import authenticate

PROJECT_ROOT = Path(__file__).resolve().parents[1]
POLICY_SYSTEM_PROMPT = (
    "You are an expert insurance agent designed to assist with coverage queries. "
    "Use the provided documents to answer questions about insurance policies. "
    "If the information is not available in the documents, respond with "
    "'I don't know'"
)


def policy_document_path() -> Path:
    return Path(
        os.getenv(
            "POLICY_DOCUMENT_PATH",
            str(PROJECT_ROOT / "data" / "2026AnthemgHIPSBC.pdf"),
        )
    ).expanduser()


def escape_markdown_dollars(text: str) -> str:
    return text.replace("$", r"\$")


class PolicyAgent:
    def __init__(self) -> None:
        backend = os.getenv("POLICY_AGENT_BACKEND", "google").strip().lower()
        if backend == "google":
            self.backend = GoogleVertexPolicyBackend()
        elif backend == "anthropic":
            self.backend = AnthropicVertexPolicyBackend()
        else:
            raise ValueError(
                "Unsupported POLICY_AGENT_BACKEND. Use 'google' or 'anthropic'."
            )

    def answer_query(self, prompt: str) -> str:
        return self.backend.answer_query(prompt)


class GoogleVertexPolicyBackend:
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
        self.pdf_data = policy_document_path().read_bytes()

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
        if not response.text:
            raise RuntimeError("The Google Vertex model returned an empty response.")
        return escape_markdown_dollars(response.text)


class AnthropicVertexPolicyBackend:
    def __init__(self) -> None:
        location = os.getenv("ANTHROPIC_VERTEX_LOCATION", "global")
        credentials, project_id = authenticate(location=location)
        self.client = AnthropicVertex(
            project_id=project_id,
            region=location,
            credentials=credentials,
        )
        with policy_document_path().open("rb") as file:
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
        return escape_markdown_dollars(response.content[0].text)
