"""Verify that local Application Default Credentials can mint a GCP token."""

from __future__ import annotations

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from gcp_auth import authenticate, refresh_access_token  # noqa: E402


def main() -> None:
    credentials, project_id = authenticate()
    token = refresh_access_token(credentials)

    print("Google Cloud authentication is configured.")
    print(f"  Project: {project_id}")
    print(f"  Location: {os.environ['GOOGLE_CLOUD_LOCATION']}")
    print(f"  Credential type: {type(credentials).__name__}")
    print(f"  Access token: acquired ({len(token)} characters; value hidden)")
    print(f"  Expiry: {getattr(credentials, 'expiry', None) or 'not reported'}")


if __name__ == "__main__":
    main()
