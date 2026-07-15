"""Tests for local Google Cloud configuration helpers."""

import os
import unittest
from unittest.mock import Mock, patch

from gcp_auth import authenticate, vertex_api_base


class GcpAuthTest(unittest.TestCase):
    def test_authenticate_prefers_explicit_project(self) -> None:
        credentials = Mock()

        with (
            patch.dict(
                os.environ,
                {"GOOGLE_CLOUD_PROJECT": "configured-project"},
                clear=True,
            ),
            patch("gcp_auth.load_local_environment"),
            patch(
                "gcp_auth.google.auth.default",
                return_value=(credentials, "detected-project"),
            ) as default_auth,
        ):
            result_credentials, project_id = authenticate(location="europe-west1")

        self.assertIs(result_credentials, credentials)
        self.assertEqual(project_id, "configured-project")
        self.assertEqual(
            default_auth.call_args.kwargs["quota_project_id"],
            "configured-project",
        )

    def test_vertex_api_base(self) -> None:
        self.assertEqual(
            vertex_api_base("global"),
            "https://aiplatform.googleapis.com",
        )
        self.assertEqual(
            vertex_api_base("us-central1"),
            "https://us-central1-aiplatform.googleapis.com",
        )


if __name__ == "__main__":
    unittest.main()
