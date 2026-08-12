"""Provenance stamp for generated artifacts — bulk-commit-proof dating.

Local copy of docs/dev/transition/pdv-trigger/_stamp.py (same contract) so this
workstream's harness has no cross-workstream import. Every builder embeds the TRUE
generation moment + code state as the FIRST line of its CSV output:

    # generated 2026-07-13T12:34:56Z | builder harvest_cf_scan.py | code 3b9be89+dirty

Readers must skip leading '#' lines (csv.DictReader does NOT do this for you).
"""

import datetime as _dt
import os
import subprocess


def stamp(script_file: str) -> str:
    here = os.path.dirname(os.path.abspath(script_file))

    def _git(*args):
        try:
            return subprocess.run(
                ["git", "-C", here, *args], capture_output=True, text=True
            ).stdout.strip()
        except OSError:
            return ""

    sha = _git("rev-parse", "--short", "HEAD") or "nogit"
    dirty = "+dirty" if _git("status", "--porcelain") else ""
    when = _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    return f"# generated {when} | builder {os.path.basename(script_file)} | code {sha}{dirty}"
