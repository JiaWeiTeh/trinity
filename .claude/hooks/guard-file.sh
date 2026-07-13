#!/usr/bin/env bash
# .claude/hooks/guard-file.sh
# PreToolUse guard for Edit|Write|NotebookEdit: blocks writes into the
# generated/scratch dirs named in CLAUDE.md: outputs/ (except the tracked
# fixture dir outputs/mockOutput/), fig/, scratch/, tbd/, old_doNotRead/.
#
# Scope honesty: covers the Edit/Write/NotebookEdit tools only; Bash
# redirection is guard-bash.sh's domain (and deliberately unparsed there).
# Fail-open without python3: hygiene guard, not a security boundary.
command -v python3 >/dev/null 2>&1 || exit 0

# Read the tool JSON from stdin, then hand it to python via the environment —
# NOT python's stdin, because the program itself arrives on stdin via the
# heredoc. (macOS ships bash 3.2, whose $()+heredoc parser is unreliable, so we
# avoid that construction; like guard-bash.sh, the program never rides stdin.)
payload="$(cat)"
CLAUDE_HOOK_JSON="$payload" python3 - <<'PY'
import json, os, subprocess, sys

try:
    data = json.loads(os.environ.get("CLAUDE_HOOK_JSON", ""))
except Exception:
    sys.exit(0)                        # unparsable input: don't block edits

ti = data.get("tool_input") or {}
paths = [ti.get(k) for k in ("file_path", "notebook_path") if ti.get(k)]
if not paths:
    sys.exit(0)

cwd = data.get("cwd") or "."

# Case-insensitive filesystems (macOS APFS default, Windows) must compare paths
# case-folded, else 'Outputs/x' slips past a rule written for 'outputs/'.
# os.path.normcase does NOT fold off Windows (it is identity on macOS and Linux),
# so fold explicitly by platform; Linux keeps case-sensitive matching (its fs is).
_CASE_FOLD = sys.platform in ("darwin", "win32")

def norm(p):
    # realpath resolves .. and symlinks; then fold case on case-insensitive fs.
    r = os.path.realpath(p)
    return r.lower() if _CASE_FOLD else r

def project_root():
    v = os.environ.get("CLAUDE_PROJECT_DIR")
    if v:
        return v
    try:
        r = subprocess.run(["git", "-C", cwd, "rev-parse", "--show-toplevel"],
                           capture_output=True, text=True, timeout=5)
        if r.returncode == 0 and r.stdout.strip():
            return r.stdout.strip()
    except Exception:
        pass
    return cwd

root = norm(project_root())
BLOCKED = ("outputs", "fig", "scratch", "tbd", "old_doNotRead")    # per CLAUDE.md
EXCEPTIONS = (norm(os.path.join(root, "outputs", "mockOutput")),)  # tracked fixtures

def under(p, base):
    return p == base or p.startswith(base + os.sep)

for path in paths:
    target = norm(os.path.join(cwd, path))
    for d in BLOCKED:
        if under(target, norm(os.path.join(root, d))):
            if any(under(target, e) for e in EXCEPTIONS):
                continue
            sys.stderr.write(
                "BLOCKED by guard-file hook: '%s' resolves into %s/, a generated/scratch "
                "directory per CLAUDE.md (outputs/, fig/, scratch/, tbd/, old_doNotRead/ "
                "are not source). Put reports under docs/dev/ and fixtures under test/; "
                "outputs/mockOutput/ is the only writable exception.\n" % (path, d)
            )
            sys.exit(2)
sys.exit(0)
PY
