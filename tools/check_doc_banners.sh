#!/usr/bin/env bash
# tools/check_doc_banners.sh
# Verify that plan/audit markdown docs under analysis/ and docs/dev/ carry all
# three required banner paragraphs at the top (see CLAUDE.md):
#   ⚠️  stale-warning      — "may be out of date — verify before trusting it"
#   🔄  living-plan        — "recheck and refine on every visit"
#   💾  persist-diagnostics — "commit, don't re-run"
#
# Usage:  tools/check_doc_banners.sh FILE [FILE ...]
# Scope:  only acts on analysis/**.md and docs/dev/**.md; folder-index README.md
#         files are exempt. Paths may be repo-relative or absolute.
# Exit:   0 if every in-scope file carries all three banners (or none are in
#         scope); 1 if any in-scope file is missing one or more.
#
# Used by both the pre-commit `doc-banners` hook and the Claude PostToolUse
# wrapper (.claude/hooks/check-doc-banners.sh).
set -u

sentinels=("⚠" "🔄" "💾")
labels=("⚠️ stale-warning" "🔄 living-plan" "💾 persist-diagnostics")
fail=0

for f in "$@"; do
  # Scope: only analysis/ and docs/dev/ markdown (repo-relative or absolute).
  case "$f" in
    analysis/*.md|*/analysis/*.md|docs/dev/*.md|*/docs/dev/*.md) : ;;
    *) continue ;;
  esac
  # Folder-index READMEs are not plan/audit docs.
  [ "$(basename "$f")" = "README.md" ] && continue
  [ -f "$f" ] || continue

  missing=()
  for i in "${!sentinels[@]}"; do
    grep -q -- "${sentinels[$i]}" "$f" || missing+=("${labels[$i]}")
  done
  if [ "${#missing[@]}" -ne 0 ]; then
    fail=1
    printf '✗ %s\n    missing banner(s): %s\n' "$f" "${missing[*]}"
  fi
done

if [ "$fail" -ne 0 ]; then
  cat >&2 <<'EOF'

Plan/audit docs under analysis/ and docs/dev/ must carry all three banner
paragraphs at the top, directly under the H1 (see CLAUDE.md). Copy them verbatim
from an existing doc, e.g. docs/dev/TRANSITION_TRIGGER_PLAN.md.
EOF
fi
exit "$fail"
