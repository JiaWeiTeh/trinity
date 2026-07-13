#!/bin/bash
# SessionStart hook — remote (web) container bootstrap + local sync note.
#   Local sessions: emit a one-line ahead/behind note to session context.
#   Remote (web) sessions: set git identity + install the package.
# Idempotent and non-interactive; safe on every session start.
set -euo pipefail

# ---- Local sessions: cheap remote-drift note, then done. -------------------
if [ "${CLAUDE_CODE_REMOTE:-}" != "true" ]; then
  cd "${CLAUDE_PROJECT_DIR:-.}" 2>/dev/null || exit 0
  export GIT_TERMINAL_PROMPT=0
  export GIT_SSH_COMMAND="ssh -oBatchMode=yes -oConnectTimeout=5"
  # macOS has no timeout(1); perl alarm bounds a hung network at 8 s.
  if command -v perl >/dev/null 2>&1; then
    perl -e 'alarm 8; exec @ARGV' git fetch --quiet 2>/dev/null || true
  else
    git fetch --quiet 2>/dev/null || true
  fi
  sync_line="$(git status -sb 2>/dev/null | head -n 1)" || true
  if [ -n "${sync_line:-}" ]; then
    echo "git sync: ${sync_line}"
  fi
  exit 0
fi

# ---- Remote (web) sessions: bootstrap. --------------------------------------
cd "${CLAUDE_PROJECT_DIR:-.}"

# 1. Author commits as the repo owner. Web containers otherwise default git to
#    "Claude <noreply@anthropic.com>" — never let that authorship reach history.
git config user.name "Jia Wei Teh"
git config user.email "jiaweiteh.astro@gmail.com"

# 2. Install the package (editable) + dev tools. Non-fatal: a resolver/network
#    failure must not kill the bootstrap silently — surface it instead.
pip install -q --root-user-action=ignore -e ".[dev]" \
  || echo "session-start: pip install -e .[dev] failed; run it manually before pytest/ruff" >&2
