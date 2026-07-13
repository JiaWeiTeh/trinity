#!/usr/bin/env bash
# .claude/hooks/post-edit-ruff.sh
# PostToolUse (Edit|Write): run ruff bug-class checks on edited *.py files,
# same selection as .pre-commit-config.yaml (F821,F811,F823,E9).
# Exit 2 => stderr is fed back to Claude as feedback (PostToolUse cannot
# block; the edit already happened). Silently no-ops when ruff is absent so
# sessions without ruff (e.g. bare anaconda) never break.

extract_path() {
  if command -v jq >/dev/null 2>&1; then
    jq -r '.tool_input.file_path // empty'
  elif command -v python3 >/dev/null 2>&1; then
    python3 -c 'import sys,json; print((json.load(sys.stdin).get("tool_input") or {}).get("file_path") or "")'
  else
    printf ''
  fi
}

file="$(extract_path)"

case "$file" in
  *.py) ;;
  *) exit 0 ;;              # never lint non-Python files
esac
[ -f "$file" ] || exit 0    # deleted/unresolvable: fail safe

if command -v ruff >/dev/null 2>&1; then
  RUFF=(ruff)
elif python3 -m ruff --version >/dev/null 2>&1; then
  RUFF=(python3 -m ruff)
else
  exit 0
fi

out="$("${RUFF[@]}" check --isolated --select F821,F811,F823,E9 "$file" 2>&1)"
status=$?
if [ "$status" -eq 1 ]; then   # 1 = violations; 2 = ruff internal error (stay silent)
  {
    echo "ruff bug-class check failed for $file (same rules as pre-commit; fix before committing):"
    echo "$out"
  } >&2
  exit 2
fi
exit 0
