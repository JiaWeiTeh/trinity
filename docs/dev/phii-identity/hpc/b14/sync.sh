#!/usr/bin/env bash
# Laptop-side helper for the phii-identity arm ladders on Helix.
# Batch 21 (k10_o1) by default. Other arms: k10 (Batch 18, SUPERSEDED+HELD -- do not
# run), k5a_swap / k5a_driving (Batch 14). Set ARMS="baseline <arm>" to choose.
# Mirrors paper/II-survey/sync.sh and paper/shellSSC6/sync.sh — but unlike the
# gitignored paper/ folders this bundle is COMMITTED, so `up` is git push +
# cluster git pull, not rsync.
#
# Version discipline (the point of the stamps): every `submit` mints
#   STAMP = UTC yyyymmdd_hhmmssZ
# and everything from that submission lives under $WS/phii_arm_$STAMP/ on the
# cluster — worktrees, per-arm outputs, applied-diff records, logs, reduced
# CSVs. The stamp is saved locally in .last_stamp; `watch`/`reduce`/`down`
# operate on it (override with STAMP=... for an older sweep). `down` lands the
# reduced CSVs in docs/dev/phii-identity/data/hpc/phii_arm_$STAMP/ for review —
# COMMIT them after review (💾 rule); nothing is auto-committed.
#
#   ./sync.sh up       # push $BRANCH; cluster fetches + checks it out
#   ./sync.sh submit   # mint STAMP; sbatch one job per arm (default: baseline k10_o1)
#   ./sync.sh watch    # queue + per-arm finished-run count for $STAMP
#   ./sync.sh reduce   # matched-t ledgers vs baseline, ON the cluster (compare_trajectories.py)
#   ./sync.sh down     # pull _reduced/ + applied diffs + walltimes to data/hpc/phii_arm_$STAMP/
#
# Knobs: HELIX=alias  BRANCH=...  ARMS="baseline k5a_swap"  CONFIGS=SC,B3M
#        STOP_T=1.5  BASE_SHA=...  STAMP=... (rewind to an older sweep)
set -euo pipefail

HOST=${HELIX:-helix}
BRANCH=${BRANCH:-bugfix/phii-pt3}
CREPO=/home/hd/hd_hd/hd_cq295/trinity                # trinity repo on Helix (/home)
WS=/gpfs/bwfor/work/ws/hd_cq295-trinity              # writable workspace (/gpfs)
ARMS=${ARMS:-"baseline k10_o1"}   # Batch 21 default; k10 = the superseded Batch 18 form; K5 arms are k5a_swap k5a_driving
BUNDLE=docs/dev/phii-identity/hpc/b14

LOCAL=$(cd "$(dirname "$0")" && pwd)
REPO=$(cd "$LOCAL/../../../../.." && pwd)
STAMP=${STAMP:-$(cat "$LOCAL/.last_stamp" 2>/dev/null || true)}
SWEEP=$WS/phii_arm_$STAMP

need_stamp() { [ -n "$STAMP" ] || { echo "no stamp — run ./sync.sh submit first (or pass STAMP=...)"; exit 1; }; }

case "${1:-}" in
  up)     echo ">> push $BRANCH and update the cluster checkout"
          git -C "$REPO" push origin "$BRANCH"
          ssh "$HOST" "bash -lc 'cd $CREPO && git fetch origin && git checkout $BRANCH && git pull --ff-only origin $BRANCH'" ;;

  submit) STAMP=$(date -u +%Y%m%d_%H%M%SZ); SWEEP=$WS/phii_arm_$STAMP
          echo "$STAMP" > "$LOCAL/.last_stamp"
          echo ">> sweep $SWEEP  (arms: $ARMS)"
          for arm in $ARMS; do
            # prep (worktree + patch) on the LOGIN node — compute nodes cannot
            # write /home, and `git worktree add` touches $CREPO/.git.
            # prep (worktree + patch) on the LOGIN node -- compute nodes cannot write
            # /home, and `git worktree add` touches $CREPO/.git.
            # SWEEP/ARM/CREPO go to sbatch as POSITIONAL ARGUMENTS, never --export:
            # b14.sbatch sets --export=NONE, so an exported variable would be silently
            # dropped and the job would run against a default. See paper/II-survey's
            # RUNBOOK ("Sweep as an ARGUMENT, never --export=SWEEP=").
            ssh "$HOST" "bash -lc 'mkdir -p $SWEEP/logs && cd $CREPO && \
              ${BASE_SHA:+BASE_SHA=$BASE_SHA }${CONFIGS:+CONFIGS=$CONFIGS }${STOP_T:+STOP_T=$STOP_T }bash $BUNDLE/run_arms.sh prep $SWEEP $arm && \
              sbatch --job-name=phii_$arm --output=$SWEEP/logs/${arm}_%j.log \
                $BUNDLE/b14.sbatch $CREPO $SWEEP $arm'"
          done
          echo ">> stamp $STAMP saved to .last_stamp"
          echo ">> CONFIRM BEFORE WALKING AWAY (the runbook rule):"
          echo "     ./sync.sh banner     # first lines of each log: CREPO / SWEEP / ARM / WT / python"
          echo "   then ./sync.sh watch" ;;

  banner) need_stamp
          echo ">> first 12 lines of each arm log for $STAMP (check SWEEP and ARM before waiting)"
          ssh "$HOST" "bash -lc 'for f in $SWEEP/logs/*.log; do echo \"== \$f\"; head -12 \"\$f\"; echo; done'" ;;

  watch)  need_stamp
          echo ">> $SWEEP"
          ssh "$HOST" "bash -lc '
            squeue --me -o \"%.12i %.16j %.2t %.10M %R\" 2>/dev/null || true
            for arm in $ARMS; do
              done=\$(ls $SWEEP/\$arm/*/dictionary.jsonl 2>/dev/null | wc -l)
              echo \"-- \$arm: \$done runs have output --\"
            done
            tail -2 $SWEEP/logs/*.log 2>/dev/null || true
          '" ;;

  reduce) need_stamp
          echo ">> matched-t ledgers vs baseline on $HOST (bar + fate table per config)"
          for arm in $ARMS; do
            [ "$arm" = baseline ] && continue
            # compare_trajectories exits 1 on a >5% breach or fate change — for
            # these arms that is the EXPECTED measurement, not an error.
            ssh "$HOST" "bash -lc 'cd $CREPO && mkdir -p $SWEEP/_reduced && \
              python $BUNDLE/../../harness/compare_trajectories.py \
                --base $SWEEP/baseline --new $SWEEP/$arm --label phii_$arm \
                --out $SWEEP/_reduced/phii_${arm}_ledger.csv'" \
              || echo ">> $arm: bar breached or fate changed (recorded in the ledger — expected for these arms)"
          done ;;

  down)   need_stamp
          DEST=$REPO/docs/dev/phii-identity/data/hpc/phii_arm_$STAMP
          echo ">> pull reduced CSVs + provenance -> $DEST"
          mkdir -p "$DEST"
          rsync -av "$HOST:$SWEEP/_reduced/" "$DEST/" 2>/dev/null || echo "   (no _reduced/ yet — run ./sync.sh reduce)"
          rsync -av "$HOST:$SWEEP/"*_applied.diff "$DEST/" 2>/dev/null || true
          for arm in $ARMS; do
            rsync -av "$HOST:$SWEEP/wt_$arm/docs/dev/phii-identity/data/phii_${arm}_walltimes.csv" \
              "$DEST/" 2>/dev/null || true
          done
          echo ">> review, then commit the keepers (💾): git add docs/dev/phii-identity/data/hpc/phii_arm_$STAMP" ;;

  *)      echo "usage: $0 up|submit|banner|watch|reduce|down   (HELIX= BRANCH= ARMS= CONFIGS= STOP_T= BASE_SHA= STAMP=)"; exit 1 ;;
esac
