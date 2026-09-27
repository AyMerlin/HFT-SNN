"""Reproducibility helpers: stable hashing and run metadata (§0 principle 5)."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]

TRACKED_PACKAGES = (
    "snn-hft",
    "numpy",
    "pandas",
    "pyarrow",
    "numba",
    "scipy",
    "pyyaml",
    "pydantic",
    "matplotlib",
    "requests",
)


def stable_hash(obj: Any, length: int = 12) -> str:
    """Hash of a JSON-serialisable object that does not depend on key order."""
    blob = json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(blob.encode()).hexdigest()[:length]


def _git(*args: str) -> str | None:
    try:
        out = subprocess.run(
            ["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, timeout=10, check=True
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip()


def git_info() -> dict[str, Any]:
    status = _git("status", "--porcelain")
    return {
        "commit": _git("rev-parse", "HEAD"),
        "branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
        "dirty": None if status is None else bool(status),
    }


def package_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for name in TRACKED_PACKAGES:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def collect_meta(**extra: Any) -> dict[str, Any]:
    """Everything needed to trace a result back to code, environment and machine."""
    return {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git": git_info(),
        "python": sys.version.split()[0],
        "packages": package_versions(),
        "machine": {
            "platform": platform.platform(),
            "processor": platform.machine(),
            "cpu_count": os.cpu_count(),
        },
        "argv": sys.argv,
        **extra,
    }
