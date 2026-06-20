#!/usr/bin/env bash
# One-shot health check of the C0 full (stop_t 6) batch. Run on a ~6 min cadence.
# Liveness uses dictionary.jsonl mtime (written every accepted segment, INFO or not),
# so a live-but-quiet implicit phase is not mistaken for a stall.
cd "$(git rev-parse --show-toplevel)" 2>/dev/null || exit 0
D=docs/dev/transition/cleanroom/data
ts=$(date '+%H:%M:%S')

done=$(ls "$D"/c0_*_st6.csv 2>/dev/null | wc -l | tr -d ' ')
crash=$(grep -lE "Traceback|CRITICAL|\[FAIL\]" "$D"/c0_*_st6.log 2>/dev/null | sed 's#.*/c0_##;s/_st6.log//' | tr '\n' ',')
# 4-min freshness window: hybr stiff segments can exceed 2 min, so a tighter
# window false-flags a working run as quiet.
live=$(find /tmp -maxdepth 3 -name dictionary.jsonl -mmin -4 2>/dev/null | wc -l | tr -d ' ')

echo "[$ts] HEARTBEAT C0/stop_t6: done=$done/6 | live_writers(<4min)=$live | crashed=[${crash:-none}]"
for pid in $(pgrep -f "c0_consistency.py.*stop-t 6" 2>/dev/null); do
  args=$(ps -o args= -p "$pid" 2>/dev/null)
  case "$args" in *python*) ;; *) continue ;; esac
  cfg=$(printf '%s' "$args" | grep -oE "configs/[a-z0-9_]+" | sed 's#configs/##')
  [ -z "$cfg" ] && continue
  el=$(ps -o etime= -p "$pid" 2>/dev/null | tr -d ' ')
  ph=$(grep -oE "PHASE 1[abc][^,]*|momentum-driven|shell.dissolved|Simulation (finished|complete)" \
       "$D/c0_${cfg}_st6.log" 2>/dev/null | tail -1)
  echo "   $cfg (elapsed $el): ${ph:-init}"
done

# anomaly flags (for the monitoring loop to act on)
[ "$done" = 6 ] && echo "   >>> ALL 6 COMPLETE"
[ -n "$crash" ] && echo "   >>> CRASH in: $crash"
[ "$live" = 0 ] && [ "$done" != 6 ] && echo "   >>> WARNING: no active jsonl writers (possible stall)"
exit 0
