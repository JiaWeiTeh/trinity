#!/usr/bin/env bash
# H4 PdV-drain-cap experiment — run the (config, variant, t_window, kappa) matrix,
# ONE sim per process (trinity leaks module-level globals in-process),
# OMP_NUM_THREADS=1, up to 3 cells CONCURRENT (a sibling experiment uses ~1 core;
# cap at 3 so we don't oversubscribe 4 physical cores). Each cell bounded by a
# wall-clock `timeout` AND a small --stop_t.
#
# Outcomes recorded per cell in h4_eval.csv: crashed / completed (clean
# end_reason) / timeout (the runner won't append a row on timeout, so an absent
# row == timeout; a TIMEOUT marker is echoed to the per-cell log).
#
# Usage: bash docs/dev/transition/pt4/h4_run_matrix.sh
set -u
cd "$(git rev-parse --show-toplevel)" || exit 1
PT4=docs/dev/transition/pt4
CLEAN=docs/dev/transition/cleanroom/configs
FLC=docs/dev/failed-large-clouds/harness/params
EVAL=$PT4/h4_eval.csv
LOGDIR=$PT4/h4_logs
KAPPA=0.9
MAXJOBS=3
export OMP_NUM_THREADS=1

rm -f "$EVAL"
mkdir -p "$LOGDIR" "$PT4/traj"

# ---------------------------------------------------------------------------
# Cell list. Format:  tag : param-path : variant : t_window : stop_t : timeout(s)
# ---------------------------------------------------------------------------
# COLLAPSE configs (decision at t~0.003 Myr): sweep t_window {1e-3,3e-3,1e-2} at
# kappa=0.9 + one long window 1e-1; plus a V0 baseline per config (re-run here so
# the eval row is directly comparable, same stop_t). stop_t=0.03 Myr easily clears
# the decision point and shows post-cap behaviour; timeout 480s (V0 fail_repro/
# helix collapse in <80s; mass_1e9 grinds so it leans on the timeout).
# CONTROL configs (healthy/stall): PDVCAP must NEVER activate (PdV<Lmech), so they
# must be byte-identical to V0 -> run V0 + PDVCAP(t_window=1e-2) and diff.
COLLAPSE=(fail_repro fail_helix mass_1e9)
TWINDOWS=(1e-3 3e-3 1e-2 1e-1)
CONTROL=(small_1e6 simple_cluster pl2_steep)

CELLS=()
# collapse: baseline + sweep
for cfg in "${COLLAPSE[@]}"; do
  CELLS+=("${cfg}_V0:$FLC/${cfg}.param:V0:1e-3:0.03:480")
  for tw in "${TWINDOWS[@]}"; do
    CELLS+=("${cfg}_PDVCAP_tw${tw}:$FLC/${cfg}.param:PDVCAP:${tw}:0.03:480")
  done
done
# controls: small_1e6 lives in FLC; simple_cluster/pl2_steep in cleanroom.
for cfg in "${CONTROL[@]}"; do
  case "$cfg" in
    small_1e6) p="$FLC/${cfg}.param" ;;
    *)         p="$CLEAN/${cfg}.param" ;;
  esac
  CELLS+=("${cfg}_V0:$p:V0:1e-3:0.005:300")
  CELLS+=("${cfg}_PDVCAP_tw1e-2:$p:PDVCAP:1e-2:0.005:300")
done

ROWDIR=$PT4/h4_rows
rm -rf "$ROWDIR"; mkdir -p "$ROWDIR"

run_cell() {
  local cell="$1"
  IFS=':' read -r tag param variant tw stopt tmo <<< "$cell"
  local out=/tmp/h4/${tag}
  local traj=$PT4/traj/h4_traj_${tag}.csv
  local log=$LOGDIR/${tag}.log
  # per-cell CSV avoids the append race between the 3 concurrent processes;
  # merged into $EVAL after the pool drains.
  local rowcsv=$ROWDIR/${tag}.csv
  echo "=== [$tag] variant=$variant tw=$tw stop_t=$stopt timeout=${tmo}s ===" > "$log"
  timeout "$tmo" python "$PT4/h4_run_variant.py" \
    --variant "$variant" --param "$param" --stop_t "$stopt" \
    --t_window "$tw" --kappa "$KAPPA" \
    --csv "$rowcsv" --traj "$traj" --out "$out" >> "$log" 2>&1
  local rc=$?
  if [ "$rc" -eq 124 ]; then
    echo "TIMEOUT [$tag] after ${tmo}s -- salvaging partial jsonl" >> "$log"
    python "$PT4/h4_salvage_timeout.py" --config "$(basename "$param" .param)" \
      --variant "$variant" --t_window "$tw" --kappa "$KAPPA" \
      --out "$out" --csv "$rowcsv" --traj "$traj" --timeout_s "$tmo" >> "$log" 2>&1
    echo "TIMEOUT [$tag] after ${tmo}s (salvaged)"
  else
    echo "DONE [$tag] rc=$rc"
  fi
}
export -f run_cell
export PT4 KAPPA OMP_NUM_THREADS LOGDIR ROWDIR

# Launch up to MAXJOBS concurrent. ponytail: simple xargs -P pool, not GNU
# parallel (avoid the dependency); ceiling = MAXJOBS sims at once.
printf '%s\n' "${CELLS[@]}" | xargs -I{} -P "$MAXJOBS" bash -c 'run_cell "$@"' _ {}

# merge per-cell rows (header once) into the eval CSV.
first=1
for f in "$ROWDIR"/*.csv; do
  [ -e "$f" ] || continue
  if [ "$first" -eq 1 ]; then cat "$f" > "$EVAL"; first=0; else tail -n +2 "$f" >> "$EVAL"; fi
done
echo "DONE matrix -> $EVAL"
