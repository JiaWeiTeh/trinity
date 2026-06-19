#!/bin/bash
# SessionStart hook — prepares a fresh Claude Code (web) container.
#
#   1. Git identity: web containers default git to "Claude <noreply@anthropic.com>";
#      override it so commits are authored by the repo owner. Local dev sessions
#      are left untouched (they already have the developer's own git config).
#   2. Dependencies: a fresh web container ships without the scientific stack, so
#      pytest / ruff / the docs/dev/performance harnesses can't run until these
#      are installed (editable + [dev] extra, per CLAUDE.md).
#
# Idempotent and non-interactive; safe to run on every session start.
set -euo pipefail

# Remote (web) sessions only — don't touch a developer's local machine.
if [ "${CLAUDE_CODE_REMOTE:-}" != "true" ]; then
  exit 0
fi

cd "${CLAUDE_PROJECT_DIR:-.}"

# 1. Author commits as the repo owner.
git config user.name "Jia Wei Teh"
git config user.email "jiaweiteh.astro@gmail.com"

# 2. Install the package (editable) + dev tools. Quiet on success; the web
#    container caches the environment after the hook completes.
pip install -q --root-user-action=ignore -e ".[dev]"
