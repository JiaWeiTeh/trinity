"""Provenance stamp for generated artifacts — bulk-commit-proof dating.

A git commit date only UPPER-BOUNDS an artifact's age: work is often bulk-committed
days after it was produced, so "committed 07-01" may mean "generated 06-24". Every
builder therefore embeds the TRUE generation moment + code state as the FIRST line
of its CSV/TXT output:

    # generated 2026-07-02T12:34:56Z | builder harvest_theta_max.py | code 057cd96+dirty

`+dirty` means the working tree had uncommitted changes when the artifact was made —
i.e. the output may not be reproducible from any commit; regenerate from a clean tree
before relying on it. MANIFEST.md (make_manifest.py) reads the stamp back as the
"generated" column, next to the commit date.

Usage in a builder:
    import sys; from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # the pdv-trigger dir
    from _stamp import stamp
    fh.write(stamp(__file__) + "\\n")   # before the CSV header

Readers must skip leading '#' lines (csv.DictReader does NOT do this for you):
    rows = list(csv.DictReader(l for l in fh if not l.lstrip().startswith("#")))
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
