"""ManiaBot application layer: model storage, versioning, training jobs, testing and replays.

The original scripts in the repository root (osu_parser, mania_env, train, evaluate,
export_replay, osr_writer) are the core engine; this package builds the model-management
application on top of them. All user/runtime data lives outside the repository, see
`maniabot.storage`.
"""
from __future__ import annotations

import sys
from pathlib import Path

APP_NAME = "OsuManiaAI"
REPO_ROOT = Path(__file__).resolve().parent.parent

# The engine modules live in the repository root; make them importable no matter
# where the app or the tests are launched from.
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
