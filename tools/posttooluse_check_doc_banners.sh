#!/usr/bin/env bash
# tools/posttooluse_check_doc_banners.sh
# Claude Code PostToolUse hook (matcher: Write|Edit). After Claude writes or
# edits a plan/audit doc under analysis/ or docs/dev/, this nudges Claude
# (exit 2 -> stderr is shown back to Claude) if any of the three required banner
# paragraphs is missing. The edit itself is not undone; Claude is asked to add
# the banners.
#
# Register it in .claude/settings.json under hooks.PostToolUse — see the comment
# block at the bottom of this file. Delegates the actual check to
# tools/check_doc_banners.sh so the rule lives in one place (shared with the
# pre-commit `doc-banners` hook).
set -u

extract_path() {
  if command -v jq >/dev/null 2>&1; then
    jq -r '.tool_input.file_path // empty'
  elif command -v python3 >/dev/null 2>&1; then
    python3 -c 'import sys,json; print(json.load(sys.stdin).get("tool_input",{}).get("file_path",""))'
  else
    printf '__NO_PARSER__'
  fi
}

fpath="$(extract_path)"
# No parser or no path -> nothing we can check; stay out of the way.
[ "$fpath" = "__NO_PARSER__" ] && exit 0
[ -n "$fpath" ] || exit 0

proj="${CLAUDE_PROJECT_DIR:-$(pwd)}"

out="$("$proj/tools/check_doc_banners.sh" "$fpath" 2>&1)"
status=$?
if [ "$status" -ne 0 ]; then
  {
    echo "Banner check failed for the doc you just wrote:"
    echo "$out"
  } >&2
  exit 2
fi
exit 0

# ---------------------------------------------------------------------------
# Register in .claude/settings.json (add to the existing "hooks" object):
#
#   "PostToolUse": [
#     { "matcher": "Write|Edit",
#       "hooks": [ { "type": "command",
#         "command": "${CLAUDE_PROJECT_DIR}/tools/posttooluse_check_doc_banners.sh" } ] }
#   ]
# ---------------------------------------------------------------------------
