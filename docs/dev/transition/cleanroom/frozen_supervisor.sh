#!/usr/bin/env bash
# Supervisor for the frozen-feedback experiment (6 configs, hybr, freeze@1.0, stop_t=6).
#
# Why this exists: the container is ephemeral and can refresh mid-run. A plain background
# run writes its CSV only at the end and into /tmp, so a refresh loses hours. This supervisor
# makes the experiment crash/refresh-robust:
#   * each run writes its dictionary.jsonl into a PERSISTENT, known dir (--run-dir);
#   * a wrapper records each run's exit code to <cfg>.exit;
#   * every cycle it CHECKPOINTS the live dictionary.jsonl -> data/c0_<cfg>_frozen.csv
#     (c0_consistency.py analyze-mode is the single CSV writer -- no race with the run);
#   * every ~10 min it commits+pushes the data CSVs so the REMOTE always has the latest
#     partial data (the only thing that survives a hard container reclaim);
#   * it RELAUNCHES any run whose pid died without a clean exit (container kill / crash).
# Idempotent: re-running after a refresh skips finished configs (<cfg>.done), finalizes any
# with exit==0, and relaunches the rest. trinity cannot resume mid-run, so a killed run is
# restarted from t=0 -- but completed runs and the last partial checkpoint are preserved.
#
# Run dirs/logs/pids/sentinels live under outputs/ (gitignored); only the data CSVs are tracked.
#
#   bash docs/dev/transition/cleanroom/frozen_supervisor.sh
set -u

CD="$(cd "$(dirname "$0")" && pwd)"          # cleanroom dir
ROOT="$(cd "$CD/../../../.." && pwd)"          # repo root
cd "$ROOT"
HARNESS="$CD/c0_consistency.py"
DATA="$CD/data"
RUNROOT="$CD/outputs/frozen"                  # gitignored (docs/dev/**/outputs/)
BRANCH="feature/improve-transition-trigger"
CFGS="be_sphere large_diffuse_lowsfe midrange_pl0 pl2_steep simple_cluster small_dense_highsfe"
FREEZE=1.0 ; STOP=6 ; CYCLE=300 ; PUSH_EVERY=2   # checkpoint each cycle (5min), push every 2nd (10min)

mkdir -p "$RUNROOT"

launch () {  # $1=cfg ; start a run, wrapper writes exit code; record pid
  local cfg="$1" dir="$RUNROOT/$1"
  mkdir -p "$dir"
  rm -f "$dir.exit" 2>/dev/null || true
  ( python "$HARNESS" "$CD/configs/$cfg.param" --solver hybr \
        --freeze-feedback-at "$FREEZE" --stop-t "$STOP" --run-dir "$dir" \
        > "$RUNROOT/$cfg.log" 2>&1 ; echo $? > "$dir.exit" ) &
  echo $! > "$RUNROOT/$cfg.pid"
  echo "  launched $cfg pid $(cat "$RUNROOT/$cfg.pid")"
}

newest_jsonl () { find "$RUNROOT/$1" -name dictionary.jsonl -printf '%T@ %p\n' 2>/dev/null | sort -n | tail -1 | cut -d' ' -f2-; }

checkpoint () {  # $1=cfg ; derive the data CSV from the live/final jsonl (single writer)
  local cfg="$1" jl ; jl="$(newest_jsonl "$cfg")"
  [ -n "$jl" ] && [ -s "$jl" ] && \
    python "$HARNESS" "$jl" --out "$DATA/c0_${cfg}_frozen.csv" >/dev/null 2>&1 || true
}

push_csvs () {  # commit+push only the frozen data CSVs (never -A); retry on network blips
  git add "$DATA"/c0_*_frozen.csv 2>/dev/null || true
  git diff --cached --quiet && { echo "  (no CSV change to push)"; return; }
  git commit -q -m "checkpoint frozen-feedback runs @ $(date -u +%H:%MZ)" || return
  for d in 0 2 4 8 16; do
    [ "$d" -gt 0 ] && sleep "$d"
    git push origin "$BRANCH" >/dev/null 2>&1 && { echo "  pushed checkpoint"; return; }
  done
  echo "  WARN push failed (will retry next cycle)"
}

# initial launch: any cfg not already done gets (re)started
for cfg in $CFGS; do
  [ -f "$RUNROOT/$cfg.done" ] && { echo "skip $cfg (done)"; continue; }
  launch "$cfg"
done

it=0
while true; do
  it=$((it+1)); alive=0; done=0; line="HB $(date -u +%H:%MZ):"
  for cfg in $CFGS; do
    dir="$RUNROOT/$cfg"
    if [ -f "$RUNROOT/$cfg.done" ]; then done=$((done+1)); line="$line ${cfg%%_*}=DONE"; continue; fi
    if [ -f "$dir.exit" ]; then
      code="$(cat "$dir.exit")"
      if [ "$code" = "0" ]; then
        checkpoint "$cfg"; touch "$RUNROOT/$cfg.done"; done=$((done+1))
        line="$line ${cfg%%_*}=FIN"
      else
        echo "  $cfg exited code=$code -> relaunch"; launch "$cfg"; alive=$((alive+1))
        line="$line ${cfg%%_*}=RELAUNCH"
      fi
      continue
    fi
    pid="$(cat "$RUNROOT/$cfg.pid" 2>/dev/null || echo 0)"
    if kill -0 "$pid" 2>/dev/null; then
      alive=$((alive+1)); checkpoint "$cfg"
      lt=$(grep -oE "t = [0-9]+\.[0-9]+ Myr" "$RUNROOT/$cfg.log" 2>/dev/null | tail -1 | grep -oE "[0-9]+\.[0-9]+")
      line="$line ${cfg%%_*}=t=${lt:-?}"
    else
      echo "  $cfg pid dead, no exit file (container kill?) -> relaunch"; launch "$cfg"; alive=$((alive+1))
      line="$line ${cfg%%_*}=REKILL"
    fi
  done
  echo "$line"
  [ $((it % PUSH_EVERY)) -eq 0 ] && push_csvs
  if [ "$alive" -eq 0 ]; then echo "ALL $done DONE"; push_csvs; break; fi
  sleep "$CYCLE"
done
