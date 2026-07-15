"""Use the repository-wide local Google Cloud authentication helper."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gcp_auth import *  # noqa: E402,F403
