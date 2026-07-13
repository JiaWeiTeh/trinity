# rosette-cf harness — how to run

> ⚠️ **This document may be out of date — verify before trusting it.** It is a
> point-in-time analysis/audit, not a maintained spec; the code moves faster
> than these notes (paths, line numbers, and "what shipped" status drift).
> **Any agent or person reading this: treat it as unverified. Flag that it may
> be stale and re-check each claim, snippet, and line reference against the
> current source before relying on it.**

Pure how-to-run note. The plan, gates, and blockers live in `docs/dev/rosette-cf/README.md` —
read that first; do not launch the campaign without the maintainer-approved plan and the real
`param/rosette_cf_survey_PISM1e5_fmix.param` (gitignored — must be provided; see §2 there).

All commands from the repo root. `$WS` = a writable workspace (e.g. the session scratchpad).

```bash
# 0. preflight: expand + validate the sweep without running anything (expect 72, 0 invalid)
python run.py param/rosette_cf_survey_PISM1e5_fmix.param --dry-run

# 1. emit the job bundle (per-combo .param files + manifest.json with the axes)
python run.py param/rosette_cf_survey_PISM1e5_fmix.param --emit-jobs "$WS/cf_jobs"

# 2. timing probes (dense nCore, Cf=1.0, both fmix) — sets the real budget
python docs/dev/rosette-cf/harness/run_cf_scan_local.py --jobs "$WS/cf_jobs" \
    --workers 1 --per-arm-timeout 3600 --only '*n1e5*coverFraction1p0*' --limit 2
#    (check the printed per-arm seconds; adjust the '*n1e5*' glob to the densest
#     nCore actually in the param file)

# 3. full campaign — resumable; re-run the same command after any restart
python docs/dev/rosette-cf/harness/run_cf_scan_local.py --jobs "$WS/cf_jobs" \
    --workers 3 --per-arm-timeout <6x probe> \
    --summary docs/dev/rosette-cf/data/cf_scan_PISM1e5_summary.csv
# optional heartbeat (arm only if projected wall > ~1.5 h):
bash docs/dev/rosette-cf/harness/autocommit_cf_scan.sh "<base output dir from the bundle>"

# 4. harvest + commit (idempotent merge; safe to re-run)
python docs/dev/rosette-cf/harness/harvest_cf_scan.py <base output dir>/* \
    --csv docs/dev/rosette-cf/data/cf_scan_PISM1e5_summary.csv \
    --traj-dir docs/dev/rosette-cf/data/cf_scan_PISM1e5_traj

# 5. match (fallback matcher — prefer the frozen paper/rosette/matching/match_runs.py if present)
python docs/dev/rosette-cf/harness/match_cf_scan.py \
    --summary docs/dev/rosette-cf/data/cf_scan_PISM1e5_summary.csv \
    --traj-dir docs/dev/rosette-cf/data/cf_scan_PISM1e5_traj \
    --out docs/dev/rosette-cf/data/match_cf_PISM1e5.csv \
    --cells-out docs/dev/rosette-cf/data/match_cf_PISM1e5_cells.csv
```

Output lands in `docs/dev/rosette-cf/data/` (committed). Tooling checks:
`pytest test/test_rosette_cf_harness.py -q`.
