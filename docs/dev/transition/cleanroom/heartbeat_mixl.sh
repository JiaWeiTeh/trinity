#!/usr/bin/env bash
# Health check for the mixing-layer (mixL_theta) validation runs. ~6 min cadence.
# Key science signal: a run reaching 'transition'/'momentum' phase = un-stalled.
cd "$(git rev-parse --show-toplevel)" 2>/dev/null || exit 0
ts=$(date '+%H:%M:%S')
D=docs/dev/transition/cleanroom/data/mixl
done=$(ls "$D"/c0_*.csv 2>/dev/null | wc -l | tr -d ' ')
crash=$(grep -lE "Traceback|CRITICAL" /tmp/mixl_test.log /tmp/mixl040.log 2>/dev/null | tr '\n' ',')

echo "[$ts] MIXL validation: done=$done/2 | crashed=[${crash:-none}]"
working=0
for pid in $(pgrep -f "c0_consistency.py.*--mixl" 2>/dev/null); do
  args=$(ps -o args= -p "$pid" 2>/dev/null); case "$args" in *python*) ;; *) continue ;; esac
  th=$(printf '%s' "$args" | grep -oE "\-\-mixl [0-9.]+")
  st=$(ps -o stat= -p "$pid" 2>/dev/null | cut -c1)
  el=$(ps -o etime= -p "$pid" 2>/dev/null | tr -d ' ')
  case "$st" in R|D) working=$((working + 1)) ;; esac
  echo "   $th (el=$el state=$st)"
done
# phase reached per active run jsonl -- transition/momentum => un-stalled
for d in $(ls -dt /tmp/c0_*/ 2>/dev/null | head -3); do
  j=$(find "$d" -name dictionary.jsonl 2>/dev/null | head -1); [ -z "$j" ] && continue
  age=$(( $(date +%s) - $(stat -c %Y "$j") )); [ "$age" -gt 300 ] && continue
  ph=$(python -c "import json,collections; print(dict(collections.Counter(json.loads(l).get('current_phase') for l in open('$j'))))" 2>/dev/null)
  t=$(tail -1 "$j" 2>/dev/null | grep -oE '"t_now": *[0-9.eE+-]+' | grep -oE '[0-9.eE+-]+$')
  echo "   run @t=${t:-?}: $ph"
done

[ "$done" -ge 2 ] && echo "   >>> BOTH COMPLETE"
[ -n "$crash" ] && echo "   >>> CRASH"
[ "$working" = 0 ] && [ "$done" -lt 2 ] && echo "   >>> WARNING: no mixl sims computing"
exit 0
