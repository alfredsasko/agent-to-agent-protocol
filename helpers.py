"""Backward-compatible imports for the course notebooks."""

from gcp_auth import (
    GoogleAccessTokenProvider,
    authenticate,
    load_local_environment,
    refresh_access_token,
    vertex_api_base,
)

__all__ = [
    "GoogleAccessTokenProvider",
    "authenticate",
    "load_local_environment",
    "refresh_access_token",
    "vertex_api_base",
]
