"""Google Cloud configuration and Application Default Credentials helpers."""

from __future__ import annotations

import os
import threading
from pathlib import Path

import google.auth
from dotenv import find_dotenv, load_dotenv
from google.auth import credentials as google_credentials
from google.auth import exceptions as google_auth_exceptions
from google.auth.transport.requests import Request

CLOUD_PLATFORM_SCOPE = "https://www.googleapis.com/auth/cloud-platform"
PROJECT_ROOT = Path(__file__).resolve().parent

__all__ = [
    "GoogleAccessTokenProvider",
    "authenticate",
    "load_local_environment",
    "refresh_access_token",
    "vertex_api_base",
]


def load_local_environment() -> None:
    """Load the nearest .env without overriding values exported by the shell."""
    dotenv_path = find_dotenv(usecwd=True)
    if dotenv_path:
        load_dotenv(dotenv_path=dotenv_path, override=False)


def authenticate(
    location: str | None = None,
) -> tuple[google_credentials.Credentials, str]:
    """Return Google ADC credentials and the configured Google Cloud project.

    ADC supports local user credentials created by ``gcloud auth
    application-default login`` and attached service accounts when the code
    later runs on Google Cloud.
    """
    load_local_environment()

    configured_project = os.getenv("GOOGLE_CLOUD_PROJECT") or os.getenv(
        "VERTEXAI_PROJECT"
    )
    try:
        credentials, detected_project = google.auth.default(
            scopes=[CLOUD_PLATFORM_SCOPE],
            quota_project_id=configured_project,
        )
    except google_auth_exceptions.DefaultCredentialsError as exc:
        raise RuntimeError(
            "Google Application Default Credentials were not found. Run "
            "`gcloud auth application-default login` for local development. "
            "See README.md for the complete setup."
        ) from exc

    project_id = configured_project or detected_project
    if not project_id:
        raise RuntimeError(
            "No Google Cloud project is configured. Set GOOGLE_CLOUD_PROJECT in "
            "your .env file."
        )

    try:
        refresh_access_token(credentials)
    except google_auth_exceptions.RefreshError as exc:
        raise RuntimeError(
            "Google credentials were found but could not be refreshed. Run "
            "`gcloud auth application-default login` again and verify that the "
            "selected account can access the configured project."
        ) from exc

    effective_location = location or os.getenv("GOOGLE_CLOUD_LOCATION", "global")
    os.environ["GOOGLE_CLOUD_PROJECT"] = project_id
    os.environ["VERTEXAI_PROJECT"] = project_id
    if location:
        os.environ["GOOGLE_CLOUD_LOCATION"] = effective_location
        os.environ["VERTEXAI_LOCATION"] = effective_location
    else:
        os.environ.setdefault("GOOGLE_CLOUD_LOCATION", effective_location)
        os.environ.setdefault("VERTEXAI_LOCATION", effective_location)
    os.environ.setdefault("GOOGLE_GENAI_USE_VERTEXAI", "true")

    return credentials, project_id


def refresh_access_token(credentials: google_credentials.Credentials) -> str:
    """Refresh credentials when necessary and return a current access token."""
    if not credentials.valid or not credentials.token:
        credentials.refresh(Request())
    if not credentials.token:
        raise RuntimeError("Google credentials did not produce an access token.")
    return credentials.token


class GoogleAccessTokenProvider:
    """Thread-safe callable token provider accepted by the OpenAI Python SDK."""

    def __init__(self, credentials: google_credentials.Credentials) -> None:
        self._credentials = credentials
        self._lock = threading.Lock()

    def __call__(self) -> str:
        with self._lock:
            return refresh_access_token(self._credentials)


def vertex_api_base(location: str) -> str:
    """Build the public Vertex AI API base URL for a location."""
    if location == "global":
        return "https://aiplatform.googleapis.com"
    return f"https://{location}-aiplatform.googleapis.com"
