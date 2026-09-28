"""Provenance of converted data: the retrieval code's git state and a UTC timestamp."""

from __future__ import annotations

import subprocess
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def git_state() -> tuple[str | None, bool | None]:
    """(commit, dirty) of the repository, or (None, None) outside git. Untracked files are ignored."""

    def run(*args: str) -> str | None:
        try:
            return subprocess.run(
                ["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, timeout=10, check=True
            ).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            return None

    commit = run("rev-parse", "HEAD")
    status = run("status", "--porcelain", "--untracked-files=no")
    return commit, None if status is None else bool(status)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")
