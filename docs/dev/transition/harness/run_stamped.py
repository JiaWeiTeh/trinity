#!/usr/bin/env python3
"""Provenance-stamped trinity runner for the transition/ workstream.

Zero-contamination contract: a run is only valid if it comes from a SINGLE,
known, CLEAN commit. This wrapper enforces that and records the commit + exact
command + param hash into a `provenance.json` next to the run output, so no
artifact in the new workstream can ever be mystery-provenance.

Usage:
    python docs/dev/transition/harness/run_stamped.py <param.param> [--allow-dirty]

Refuses to run from a dirty working tree unless --allow-dirty (then it records
dirty=True and the diff hash, so the contamination is at least logged, not hidden).

Batch check (all outputs share one clean commit):
    python docs/dev/transition/harness/run_stamped.py --check <out_dir> [<out_dir> ...]
"""
import hashlib
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]  # repo root


def _git(*args):
    return subprocess.run(["git", "-C", str(ROOT), *args],
                          capture_output=True, text=True).stdout.strip()


def _sha256(text):
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def _path2output(param_path):
    for line in Path(param_path).read_text().splitlines():
        line = line.split("#", 1)[0].split()
        if len(line) >= 2 and line[0] == "path2output":
            return Path(line[1])
    raise SystemExit(f"no path2output in {param_path}")


def run(param_path, allow_dirty=False):
    commit = _git("rev-parse", "HEAD")
    dirty = bool(_git("status", "--porcelain"))
    if dirty and not allow_dirty:
        raise SystemExit(
            "REFUSING: working tree is dirty — a clean baseline must come from a "
            "committed state. Commit/stash, or pass --allow-dirty to record it as tainted.")
    param_text = Path(param_path).read_text()
    out = _path2output(param_path)
    out.mkdir(parents=True, exist_ok=True)

    cmd = [sys.executable, str(ROOT / "run.py"), str(param_path)]
    t0 = time.time()
    rc = subprocess.run(cmd).returncode
    wall = round(time.time() - t0, 1)

    prov = {
        "commit": commit,
        "commit_short": commit[:8],
        "tree_dirty": dirty,
        "diff_sha256": _sha256(_git("diff")) if dirty else None,
        "command": " ".join(cmd),
        "param_path": str(param_path),
        "param_sha256": _sha256(param_text),
        "started_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "wall_seconds": wall,
        "returncode": rc,
        "python": sys.version.split()[0],
    }
    (out / "provenance.json").write_text(json.dumps(prov, indent=2))
    print(f"{param_path}: rc={rc} wall={wall}s -> {out}/provenance.json (commit {commit[:8]}, dirty={dirty})")
    return rc


def check(out_dirs):
    """Assert every output dir shares one clean commit. Exit non-zero if not."""
    seen = {}
    ok = True
    for d in out_dirs:
        p = Path(d) / "provenance.json"
        if not p.exists():
            print(f"MISSING provenance: {d}"); ok = False; continue
        j = json.loads(p.read_text())
        tag = (j["commit_short"], j["tree_dirty"])
        seen.setdefault(tag, []).append(d)
        if j["tree_dirty"]:
            print(f"TAINTED (dirty tree): {d}"); ok = False
    if len(seen) > 1:
        print("MIXED PROVENANCE across the batch:")
        for tag, ds in seen.items():
            print(f"  {tag}: {ds}")
        ok = False
    if ok:
        print(f"OK: {len(out_dirs)} outputs share one clean commit {next(iter(seen))[0]}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    args = sys.argv[1:]
    if args and args[0] == "--check":
        check(args[1:])
    elif args:
        sys.exit(run(args[0], allow_dirty="--allow-dirty" in args))
    else:
        raise SystemExit(__doc__)
