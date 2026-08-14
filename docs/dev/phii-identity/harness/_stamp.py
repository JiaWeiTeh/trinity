"""Provenance stamp for committed artifacts — PLAN.md rule C-6.

Every CSV this workstream commits opens with
    # generated <UTC ISO> | builder <script> | code <SHA>[+dirty]
so a future visit can tell which code produced a number without re-running.
"""

import subprocess
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]


def _git(*args):
    try:
        return subprocess.run(
            ["git", *args], cwd=REPO, capture_output=True, text=True, timeout=10
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ""


def code_version():
    sha = _git("rev-parse", "--short", "HEAD") or "unknown"
    return f"{sha}+dirty" if _git("status", "--porcelain") else sha


def stamp(builder_path):
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    return f"# generated {now} | builder {Path(builder_path).name} | code {code_version()}"
