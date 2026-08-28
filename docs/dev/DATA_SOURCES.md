# Data sources and the sync route — standing decisions

**Recorded 2026-08-18.** Read this before proposing any new run, reducer, or transport.

---

## 1. The transport already exists. Use it.

`paper/II-survey/sync.sh` is the method, for every project folder that has one
(`orionM43/`, `rosette/`, `II-survey/`, …). Its verbs are:

```
./sync.sh up      rsync scripts -> helix           (gitignored folder, rsync NOT git)
./sync.sh run     git pull + reduce + figures on the cluster, then offer the tables
./sync.sh submit  same, as a SLURM job (reduce.sbatch)
./sync.sh watch   tail the newest reduce log
./sync.sh down    pull figures + summary.csv (+ side tables, with a size prompt)
```

Hosts and paths it already knows: `HOST=helix`, `CREPO=/home/hd/hd_hd/hd_cq295/trinity`,
`WS=/gpfs/bwfor/work/ws/hd_cq295-trinity`, `SWEEP=$WS/<sweep>`, `OUT=$SWEEP/_figs`.
Override the sweep with `SWEEP=$WS/paperII_grid_v2 ./sync.sh …`.

**⛔ Do not write a second transport.** A `tools/helix.sh` + `harness/helix_array.sbatch` pair
was written on 2026-08-18 before this file existed and is **superseded — delete both.** So is
`harness/make_manifest.py`: `reduce_survey.py` already emits `failed_runs.csv`,
`error_signatures.txt` and `summary_audit.txt`, which is the same job done better.

**The N6 guard is load-bearing.** `sync.sh` derives the laptop destination from the sweep name:
only `PRIMARY_SWEEP=paperII_grid_f4` may write to `plots/` (the frozen N3 tables that the
48-test suite and every quoted number in the draft are pinned to). Everything else lands in
`plots_<tag>/`; v2 → `plots_v2/`. **Never override `DEST`.**

## 2. The newest runs

**`param/paperII_grid_v2.param` → `outputs/gridsweep_v2/` is the current production grid and
the newest data.** It supersedes `paperII_grid_sweep.param` (v1, f=4 nominal wind) and
`paperII_grid_f1_control.param`.

```
11 sfe x 12 mCloud x 5 nCore x 4 PISM x 2 nISM x 2 include_PHII   = 10,560 cells
   x 2 cooling_boost_fmix  x 3 FB_thermCoeffWind                  = 63,360 runs
```

Anything predating it is **not** to be trusted for a new comparison, including
`outputs/helix/paperII_grid_sweep_180626/` (used once, for schema only, and no number from it
was ever quoted).

### What v2 already answers, and what it does not

| question | v2 covers it? |
|---|---|
| wind-strength ladder | **yes** — `FB_thermCoeffWind` ∈ {1, 0.1, 0.01}, and note it scales `pdot ∝ sqrt(c)` while `L_w ∝ c` |
| P_HII on vs off | **yes** — `include_PHII` ∈ {True, False} |
| cloud-plane coverage | **yes** — 12 `mCloud` × 5 `nCore` × 11 `sfe`, far beyond any bespoke grid |
| boosted vs un-boosted interface cooling | **yes** — `cooling_boost_fmix` ∈ {1, 2} |
| **old (pre-C3c) vs new** | **NO.** `include_PHII=False` deletes the term; the *stock* arm had `P_HII ≡ Pb` relabelled. Old-vs-new still needs a worktree run at `fca7d88e` |
| density-profile axis | **NO.** `densPL_alpha` is fixed at 0, so Geen 2022's ω vs 5/4 overflow-sign test is not in v2 |

A bespoke `Batch 13` grid was drafted on 2026-08-18 and is **superseded by v2** on every axis
except those last two rows. `BATCH13_DRAFT.md` should be deleted; if the two gaps are worth
runs, they are a small dedicated arm, not a grid.

## 3. Reduce on the cluster. Always.

A single run's `dictionary.jsonl` is ~9 MB, so v2 is ~570 GB of raw output. It never comes
down. `tools/reduce_sweep.py` and `paper/II-survey/reduce_survey.py` both exist for this;
`reduce_survey.py` streams each run's jsonl **once** and fills `summary.csv` plus the side
tables (`trajectory_points.csv`, `budget_vs_t.csv`, `frad_share_vs_t.csv`,
`dust_lyc_vs_t.csv`).

**So a new per-run quantity is an accumulator in that existing pass, not a new script and not
a second walk.** Adding a scalar to `summary.csv` is free at the wire (63k rows stays small
and `down` pulls it unconditionally); adding a per-snapshot side table costs a size prompt and
should be justified.

Keep raw output on helix until the arm's questions are closed — a reducer bug then costs a
re-reduce (`FORCE=1`), not a re-run. Precedent: PLAN.md B11.0 S1 found a real layer-volume bug
*after* the numbers were committed.

## 4. Picking a subset of 63k

Only needed for per-snapshot work. The cheap route is two-stage: `summary.csv` (one row per
run, already produced) selects; the expensive per-snapshot pass then runs on the selection.
For anything expressible as a per-run scalar, don't select — just add the column.

## 5. `data-new/`

`docs/dev/phii-identity/data-new/` holds the phii-identity workstream's own reduced artifacts
under the cut rule (C-6 stamp at or after the cut commit). It is **not** where survey output
goes — that is `plots_v2/`, governed by the N6 guard above.
