"""Tests for local Google Cloud configuration helpers."""

from unittest.mock import Mock

from gcp_auth import authenticate, vertex_api_base


def test_authenticate_prefers_explicit_project(monkeypatch) -> None:
    credentials = Mock()
    default_auth = Mock(return_value=(credentials, "detected-project"))
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "configured-project")
    monkeypatch.setattr("gcp_auth.load_local_environment", Mock())
    monkeypatch.setattr("gcp_auth.google.auth.default", default_auth)

    result_credentials, project_id = authenticate(location="europe-west1")

    assert result_credentials is credentials
    assert project_id == "configured-project"
    assert default_auth.call_args.kwargs["quota_project_id"] == "configured-project"


def test_vertex_api_base() -> None:
    assert vertex_api_base("global") == "https://aiplatform.googleapis.com"
    assert (
        vertex_api_base("us-central1")
        == "https://us-central1-aiplatform.googleapis.com"
    )
