#!/usr/bin/env bash
# .claude/hooks/guard-bash.sh
# PreToolUse guard: hard-blocks destructive Bash commands.
# Defense-in-depth on top of the permission deny rules and the OS sandbox.

extract_cmd() {
  if command -v jq >/dev/null 2>&1; then
    jq -r '.tool_input.command // empty'
  elif command -v python3 >/dev/null 2>&1; then
    python3 -c 'import sys,json; print(json.load(sys.stdin).get("tool_input",{}).get("command",""))'
  else
    printf '__NO_PARSER__'
  fi
}

cmd="$(extract_cmd)"

if [ "$cmd" = "__NO_PARSER__" ]; then
  echo "guard-bash: neither jq nor python3 found to parse the command; blocking for safety. Install jq: brew install jq" >&2
  exit 2
fi

block() { echo "BLOCKED by guard-bash hook: $1" >&2; exit 2; }

# recursive / force rm
echo "$cmd" | grep -Eiq 'rm[[:space:]]+-[[:alnum:]]*[rf]'                                   && block "recursive/force rm"
# git force-push (push present AND a force flag present)
if echo "$cmd" | grep -Eiq 'git[[:space:]]+push' \
   && echo "$cmd" | grep -Eiq '(--force|--force-with-lease|(^|[[:space:]])-f([[:space:]]|$))'; then
  block "git force-push"
fi
# disk / format / wipe tools (only at command position to avoid filename false-positives)
echo "$cmd" | grep -Eiq '(^|[|;&][[:space:]]*)(dd|mkfs|shred|fdisk)([[:space:].]|$)'          && block "disk/format/wipe tool"
# fork bomb
echo "$cmd" | grep -Eq ':[[:space:]]*\(\)[[:space:]]*\{'                                     && block "fork bomb"
# broad chmod / recursive chown
echo "$cmd" | grep -Eiq 'chmod[[:space:]]+(-R|777)'                                          && block "broad chmod"
echo "$cmd" | grep -Eiq 'chown[[:space:]]+-R'                                                && block "recursive chown"
# pipe remote content straight into a shell
echo "$cmd" | grep -Eiq '(curl|wget)[[:space:]].*\|[[:space:]]*(sudo[[:space:]]+)?(ba|z)?sh' && block "pipe-to-shell"
# sudo / privilege escalation
echo "$cmd" | grep -Eiq '(^|[^[:alnum:]])sudo([[:space:]]|$)'                                && block "sudo"

exit 0
