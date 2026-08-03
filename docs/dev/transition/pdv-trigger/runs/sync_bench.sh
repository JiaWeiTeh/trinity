#!/usr/bin/env bash
# Laptop-side driver for the bench campaigns on bwForCluster Helix — one script, five campaigns:
#   bench5  = the 60-arm Phase-5 L21b matrix, HPC CONFIRMATION of the in-container §15h result
#             (summary lands as bench5_summary_hpc.csv — the in-container bench5_summary.csv is
#             kept; diff them with data/compare_bench5_hpc.py)
#   bench6  = the 60-arm Phase-6 DECISION matrix (f_A dose extension + f_mix head-to-head)
#   bench7  = the f_kappa re-open campaign, KAPPA_REOPEN_PLAN.md K1-K4 (166 arms). K1/K1b/K2/K3/K4
#             all live in ONE params dir as one array — a K-phase is just a filename prefix, so
#             there is one submit, one reduce and one download for the whole campaign.
#   bench5r = bench5's committed params, RE-RUN today (maintainer ALL-FRESH ruling 2026-07-29)
#   bench6r = bench6's committed params, RE-RUN today
#             The two *r campaigns exist so the L21b baselines the bench7 head-to-head is measured
#             against — Theta_0, the f_A ladder, the f_mix ladder — are TODAY's numbers rather than
#             the 2026-07-19 harvest. They reuse the same params/sbatch/gpfs dirs but land under
#             fresh names in runs/data/, so nothing older is overwritten and old-vs-new is a diff.
#             Run order and the full rationale: ../KAPPA_REOPEN_PLAN.md section 6.2.
# Same shape as ./sync_theta5s.sh (tracked workstream: code travels by `git pull`).
#
#   ./sync_bench.sh <campaign> up       # git pull the latest committed code on the cluster
#   ./sync_bench.sh <campaign> submit   # git pull + mkdir logs + sbatch the array (AUTO-SIZED)
#   ./sync_bench.sh <campaign> watch    # your queue + tail the newest array task log
#   ./sync_bench.sh <campaign> reduce   # multi-GB jsonl -> small CSVs, ON HPC   (alias: `run`)
#   ./sync_bench.sh <campaign> down     # rsync ONLY the reduced CSVs into runs/data/
#
# WHY reduce-then-down. The raw dictionary.jsonl files are multi-GB and stay on gpfs; only the
# reduced CSVs ever travel. `reduce` runs harvest_bench5.py ON the cluster (it is campaign-agnostic):
# fire-map summary + per-arm theta(t) trajectory CSVs (<=4000 rows, log-t downsampled, endpoints
# kept) — the trajectories are what the Theta_cum L21b metric needs (make_bench*_analysis.py read
# them from runs/data/). Measured: ~3-4.5 MB of raw jsonl per arm reduces to ~18 KB.
#
# BIG CAMPAIGNS (bench8, 514 arms — F_AREA_PLAN.md §9a). At that size the raw stays ~2 GB on gpfs
# (fine) but 514 per-arm CSVs is not a reviewable commit, so bench8 sets BUNDLE_NAME + DERIVED and
# `down` fetches THREE files, ~10 MB total:
#   <c>_summary.csv  fire map + the DISTILLED SCALARS computed on the cluster (Theta_cum, the
#                    solved/stale split, theta_max_solved) — ~90 KB, and it alone answers the
#                    headline reads, so analysis never has to open a trajectory
#   <c>_traj.csv     ALL arms' trajectories in ONE long CSV keyed by run_name (data/read_bundle.py
#                    splits it back into the per-arm dicts every existing reader already consumes)
#   <c>_hashes.csv   the K3 determinism hashes (still taken over the per-arm files on gpfs)
# Both are opt-in per campaign: bench5-bench7 reduce and download exactly as before.
#
# ⚠️ THE REDUCE IS ONE-SHOT. gpfs workspaces get cleaned and the raw arms do NOT come back — the
# theta5s lesson (harvest_bench5.py docstring: its raw arms were lost to a /tmp wipe and dMdt had to
# be salvaged in a scramble). Anything the reduce does not capture is gone. So a campaign whose
# analysis needs more than the six default trajectory columns MUST declare them in EXTRA below,
# BEFORE the first reduce. bench7 declares Pb + bubble_dMdt (KAPPA_REOPEN_PLAN P2 and the K0.Q1b
# back-reaction both read them) and the L2/L3 split.
#
# `reduce` also writes <campaign>_hashes.csv: sha256 of each reduced trajectory CSV, taken over its
# NON-COMMENT lines only. That is what makes K3's determinism claim checkable (two runs of one param
# must hash identically) without ever shipping a raw dictionary down. Hash the REDUCED csv, not the
# jsonl — the physics columns only; excluding '#' lines keeps the per-file provenance stamp (added
# 2026-07-29) from making two otherwise-identical runs look different.
# Override the ssh host with HELIX=myalias ./sync_bench.sh ...
set -euo pipefail

USAGE="usage: $0 bench5|bench6|bench7|bench5r|bench6r|bench8 up|submit|watch|reduce|down   (HELIX=alias  ARRAY=1-60%16)"
CAMPAIGN=${1:-}
CMD=${2:-}
EXTRA=""
BUNDLE_NAME=""      # non-empty => reduce bundles all trajectories into ONE csv and `down` takes
DERIVED=""          # that instead of the per-arm dir; --derived puts the scalars in the summary
COLS="--extra-cols Pb,bubble_dMdt,bubble_L2Conduction,bubble_L3Intermediate"
# SRC = which committed params/sbatch/gpfs-output dir this campaign uses.
# The *_NAME vars are what lands in runs/data/ — a re-run campaign reuses SRC but lands under fresh
# names, so re-running never overwrites an older harvest. Comparing old vs new is then a file diff.
case "$CAMPAIGN" in
  bench5)  SRC=bench5; SUMMARY_NAME=bench5_summary_hpc.csv; TRAJ_NAME=bench5_traj_hpc ;;
  bench6)  SRC=bench6; SUMMARY_NAME=bench6_summary.csv;     TRAJ_NAME=bench6_traj ;;
  bench7)  SRC=bench7; SUMMARY_NAME=bench7_summary.csv;     TRAJ_NAME=bench7_traj;  EXTRA=$COLS ;;
  # The ALL-FRESH re-runs (maintainer ruling 2026-07-29): same committed params, today's numbers.
  # They also collect bench7's extra columns, which the 2026-07-19 harvests never captured.
  bench5r) SRC=bench5; SUMMARY_NAME=bench5r_summary.csv;    TRAJ_NAME=bench5r_traj; EXTRA=$COLS ;;
  bench6r) SRC=bench6; SUMMARY_NAME=bench6r_summary.csv;    TRAJ_NAME=bench6r_traj; EXTRA=$COLS ;;
  # bench8 = the 514-arm f_area campaign (F_AREA_PLAN.md §5; size budget §9a). At this size the per-arm deliverable
  # stops being reviewable, so bench8 reduces to THREE files (see BUNDLE below): summary+derived,
  # one bundled trajectory CSV, hashes. Per-arm CSVs are still written on gpfs — they are what the
  # K3 determinism hashes are taken over — but they do NOT come down.
  bench8)  SRC=bench8; SUMMARY_NAME=bench8_summary.csv;     TRAJ_NAME=bench8_traj;  EXTRA=$COLS
           BUNDLE_NAME=bench8_traj.csv; DERIVED=--derived ;;
  *) echo "$USAGE"; exit 1 ;;
esac

HOST=${HELIX:-helix}                                        # ssh host / alias
REPO=/home/hd/hd_hd/hd_cq295/trinity                        # trinity repo on Helix (/home, tracked)
WS=/gpfs/bwfor/work/ws/hd_cq295-trinity                     # writable workspace (/gpfs)
RUNS=$REPO/docs/dev/transition/pdv-trigger/runs
SBATCH=$RUNS/run_$SRC.sbatch
OUT=$WS/outputs/$SRC                                        # run dirs (dictionary.jsonl live here;
                                                            # path2output inside each .param sets it)
LOGS=$WS/jobs_$SRC/logs                                     # --output dir (must exist BEFORE sbatch)
SUMMARY=$WS/outputs/$SUMMARY_NAME                           # harvest writes here (gpfs, repo stays clean)
TRAJ=$WS/outputs/$TRAJ_NAME
HASHES_NAME=${CAMPAIGN}_hashes.csv                          # sha256 per reduced traj csv (K3)
HASHES=$WS/outputs/$HASHES_NAME
BUNDLE=${BUNDLE_NAME:+$WS/outputs/$BUNDLE_NAME}             # one-file trajectories (big campaigns)
PARAMS=$RUNS/params/$SRC
ENV_SETUP=${ENV_SETUP:-"module load devel/miniforge && conda activate trinity"}

# this repo on the laptop (where `down` drops the committed CSVs)
LAPTOP_DATA=/Users/jwt/unsync/Code/Trinity/docs/dev/transition/pdv-trigger/runs/data

case "$CMD" in
  up)      echo ">> git pull the latest committed code on $HOST (commit + push locally first)"
           ssh "$HOST" "bash -lc 'cd $REPO && git pull --ff-only'" ;;

  submit)  echo ">> on $HOST: git pull -> mkdir logs -> sbatch the $CAMPAIGN array"
           # Array is AUTO-SIZED from the committed params unless ARRAY is set, so a grid change
           # needs no edit here (bench7's size is not fixed until the §6.0 ruling lands).
           ssh "$HOST" "bash -lc 'cd $REPO && git pull && mkdir -p $LOGS && \
             n=\$(ls $PARAMS/*.param 2>/dev/null | wc -l); \
             if [ \"\$n\" -eq 0 ]; then echo \"ERROR: no params in $PARAMS — generate + commit them first\" >&2; exit 1; fi; \
             a=\"${ARRAY:-}\"; [ -n \"\$a\" ] || a=\"1-\$n\"; \
             echo \">> \$n params, submitting --array=\$a\"; sbatch --array=\$a $SBATCH'" ;;

  watch)   echo ">> queue + newest $CAMPAIGN task log on $HOST (Ctrl-C to stop)"
           ssh -t "$HOST" "squeue --me -o '%.10i %.20j %.2t %.10M %.6D %R' 2>/dev/null; \
             f=\$(ls -t $LOGS/*.out 2>/dev/null | head -1); \
             if [ -n \"\$f\" ]; then echo \"== \$f ==\"; tail -f \"\$f\"; \
             else echo 'no logs yet in jobs_$CAMPAIGN/logs — submit first / still queued (squeue).'; fi" ;;

  reduce|run)                       # `run` kept as an alias — sync.sh calls this step `reduce`
           echo ">> reduce the multi-GB jsonl -> $SUMMARY + $TRAJ/ + $HASHES  (ON $HOST)"
           echo ">> extra trajectory columns: ${EXTRA:-<none — the six defaults>}"
           [ -n "$BUNDLE" ] && echo ">> bundling all trajectories -> $BUNDLE, scalars -> the summary"
           ssh -t "$HOST" "bash -lc 'cd $REPO && $ENV_SETUP && \
             python $RUNS/harvest_bench5.py $OUT/* --csv $SUMMARY --traj-dir $TRAJ $EXTRA \
               ${BUNDLE:+--traj-bundle $BUNDLE} $DERIVED && \
             { echo \"# generated \$(date -u +%Y-%m-%dT%H:%M:%SZ) | builder sync_bench.sh $CAMPAIGN reduce | code \$(git -C $REPO rev-parse --short HEAD)\"; \
               echo run_name,sha256,rows; \
               for f in $TRAJ/*.csv; do \
                 echo \"\$(basename \"\$f\" .csv),\$(grep -v \"^#\" \"\$f\" | sha256sum | cut -d\" \" -f1),\$(grep -cv \"^#\" \"\$f\")\"; \
               done; } > $HASHES && echo \">> hashed \$(( \$(wc -l <$HASHES) - 2 )) reduced trajectories\"'" ;;

  down)    echo ">> rsync ONLY the reduced CSVs <- $HOST -> runs/data/  (raw jsonl stays on gpfs)"
           rsync -av "$HOST:$SUMMARY" "$LAPTOP_DATA/$SUMMARY_NAME" 2>/dev/null \
             || echo ">> no $SUMMARY_NAME yet — run './sync_bench.sh $CAMPAIGN reduce' first"
           rsync -av "$HOST:$HASHES" "$LAPTOP_DATA/$HASHES_NAME" 2>/dev/null || true
           if [ -n "$BUNDLE" ]; then          # big campaign: ONE trajectory file, not N
             rsync -av "$HOST:$BUNDLE" "$LAPTOP_DATA/$BUNDLE_NAME" 2>/dev/null || true
             GOT="$LAPTOP_DATA/$BUNDLE_NAME"
           else
             mkdir -p "$LAPTOP_DATA/$TRAJ_NAME"
             rsync -av "$HOST:$TRAJ/" "$LAPTOP_DATA/$TRAJ_NAME/" 2>/dev/null || true
             GOT="$LAPTOP_DATA/$TRAJ_NAME"
           fi
           echo -n ">> total that came down: "
           du -csh "$LAPTOP_DATA/$SUMMARY_NAME" "$LAPTOP_DATA/$HASHES_NAME" "$GOT" 2>/dev/null \
             | tail -1 | cut -f1
           echo ">> committed deliverables now in runs/data/ — commit them from the laptop." ;;

  *)       echo "$USAGE"; exit 1 ;;
esac
