# Rosette Cf scan (PISM=1e5, fmix) — in-container execution plan

> ⚠️ **This document may be out of date — verify before trusting it.** It is a
> point-in-time analysis/audit, not a maintained spec; the code moves faster
> than these notes (paths, line numbers, and "what shipped" status drift).
> **Any agent or person reading this: treat it as unverified. Flag that it may
> be stale and re-check each claim, snippet, and line reference against the
> current source before relying on it.**
>
> 🔄 **Living plan — recheck and refine on every visit.** This is an evolving
> strategy doc, not a frozen record. Any agent or person who opens this file
> must, as part of the visit: (1) re-verify the claims and line references above
> against current source; (2) update anything that has drifted; (3) **rethink the
> strategy itself** — if a better ordering, gate, candidate, or experiment
> exists, revise the doc and note what changed and why (date it). Leave it better
> than you found it. **Keep all banner paragraphs at the top of every plan and
> analysis doc.**
>
> 💾 **Persist diagnostics — commit, don't re-run.** The container is ephemeral
> and full/hybr runs cost hours, so any diagnostic worth keeping must be saved as
> a committed artifact under `docs/dev/` (a CSV/table in `docs/dev/data/`, or a
> harness/figure in the relevant `docs/dev/<workstream>/` folder) — never left in
> `/tmp`, the local-only `scratch/`, or an untracked `outputs/`. A future visit must be able to reproduce or compare
> against the numbers **without re-running**; record the exact config + command
> that produced each artifact.
>
> 🔗 **Cross-check the sibling docs — keep the workstream self-consistent.** This file is one of
> several living docs for its workstream (its `PLAN.md`, `FINDINGS.md`, `runs/README.md`, `NOTE_PATCHES.md`,
> and any other notes in the same folder). They drift out of sync *with each other* as fast as they drift
> from the code. Any agent or person editing one MUST, as part of the visit, circle back through the
> siblings and reconcile: if a number, status, claim, or line reference here contradicts a sibling — or a
> sibling has gone stale — fix it (or flag it, dated) so no two docs in the workstream disagree. Never
> update one in isolation.

**Status (2026-07-13):** 🟡 Phase-2 RUNNING in-container — 72-run parallel campaign launched
(workers=3, per-arm-timeout 7200s), autocommit heartbeat armed (commits summary + traj + gzipped
raw dicts every ~2 min). Deliverable = the raw `dictionary.jsonl` per arm (§7). 11 harness tests
green; preflight verified 72/0-invalid. Matcher runs against the fallback (`matching/` stays
local per maintainer). See §10 for the in-container ops playbook.

## 1. Goal & context

Helix (HPC) is down, so the 72-run Rosette Cf scan for the application paper runs in this
container — the same playbook as the theta5s (81/81) and bench5 (60/60) in-container campaigns
(`docs/dev/transition/pdv-trigger/REPRODUCE.md` rows 41/43). Scan:
**2 mass-pairs × 3 nCore × 3 Cf {0.70, 0.85, 1.0} × 2 cooling_boost_fmix {1, 4} × 2 P_HII = 72
runs** at PISM=1e5, stop_t=3 Myr. The param file (maintainer-supplied 2026-07-13) is committed at
**`docs/dev/rosette-cf/rosette_cf_survey_PISM1e5_fmix.param`** — mass-pairs (1e5, 1%) / (1e4, 10%)
bracketing NGC 2244's ~1000 Msun cluster, nCore ∈ {50, 1e2, 5e2}, `cooling_boost_mode multiplier`
so fmix bites. Deliverable: the covering fraction Cf best matching the Rosette observables, per
cell, with the `paper/rosette/PLAN.md` §0.3 sealed-baseline finding adjudicated first (§5), not
tuned around.

## 2. What this container verified vs. what it could NOT (blockers)

Verified against source @ `3b9be89`:

- The knobs exist on main: `coverFraction` (`trinity/_input/default.param:71`, Cf=1 sealed
  Weaver limit), `cooling_boost_fmix` + `cooling_boost_mode multiplier`, `PISM`, `stop_t`.
  NOTE the default.param INFO line calls Cf "usable range ~0.9–0.99; Cf near 0 drains the bubble
  within a step" — the grid's 0.70 is *below* that stated comfort band. Expect the integrator to
  be stressed there; a crashed 0.70 arm is a finding, not a tooling bug (it leaves the cell as a
  flagged 2-point fit, §6).
- The committed param file expands to exactly 72 unique combos (verified with
  `run.py ... --dry-run` and `--emit-jobs --dry-run`: 72 jobs, **0 implausible**), and the
  output-folder names DO encode the axes:
  e.g. `1e5_sfe001_n5e2_PL0_yesPHII_coolingBoostFmix1p0_coverFraction0p7`. The `--emit-jobs`
  manifest additionally carries every run's full params dict (`trinity/_input/sweep_jobs.py`),
  so matching never parses folder names.
- `--emit-jobs` bundles use the same `.exit_code`/`.duration` sentinels as the bench5 pool, and
  emit-time validation (GMC plausibility) runs before anything is queued.

NOT verifiable here — **gitignored, local-only on the maintainer's machines**
(`paper/CLAUDE.md`, `.gitignore`), so treat every statement about them in §4–§5 as transcribed
from the 2026-07-13 task brief, not checked against source:

1. ~~**`param/rosette_cf_survey_PISM1e5_fmix.param` — HARD BLOCKER.**~~ **RESOLVED 2026-07-13**:
   the maintainer supplied the file and it is committed at
   `docs/dev/rosette-cf/rosette_cf_survey_PISM1e5_fmix.param` (campaign params live in the
   workstream folder, the bench5 precedent — `param/*` stays gitignored). The gate passed:
   `--dry-run` shows exactly 72 combos, `--emit-jobs --dry-run` reports 0 implausible. The
   committed copy is CANONICAL for Phase 2; if the maintainer's local `param/` copy diverges,
   reconcile before running.
2. **`paper/rosette/PLAN.md`** (§0.3 pilot: best-fit Cf≈0.89 in one corner; §0.4 machinery;
   F-12 cavity-target conflict) — unreadable here; §5 encodes the brief's summary of it.
3. **`paper/rosette/matching/{observables.py, match_runs.py}`** — the FROZEN matching policy.
   `harness/match_cf_scan.py` is a fallback **reimplementation** of the policy as stated in the
   brief (§4). **Maintainer ruling 2026-07-13: `matching/` stays gitignored/local for now — run
   the sims first.** So Phase 2 in-container uses the fallback matcher (§4); the maintainer
   re-applies the frozen `match_runs.py` to the committed trajectory CSVs offline on their
   machine, and must diff the fallback's `POLICY` constants against the frozen `observables.py`
   before quoting any number.

## 3. Phase-2 execution plan (pure execution; exact commands in `harness/README.md`)

0. **Env**: `pip install -r requirements.txt` (keep the `numpy<2` etc. pins);
   `pytest test/test_rosette_cf_harness.py -q` (10 tests) must pass.
1. **Preflight**: `python run.py docs/dev/rosette-cf/rosette_cf_survey_PISM1e5_fmix.param
   --dry-run` → expect 72/0-invalid (re-confirm, already verified 2026-07-13), then
   `--emit-jobs "$WS/cf_jobs"`. Any invalid combo or count ≠ 72 → STOP, report, don't run.
2. **Workers/timeout**: workers = 3 (bench5 default; box has 4 CPUs, leaving one for the
   heartbeat + gzip). `--per-arm-timeout 7200` (bench5 value — generous headroom; the dense/sealed
   long-pole corner measured ~30 min in a probe, so 2 h never premature-kills a healthy arm). Per
   the maintainer (2026-07-13): the 72 finish in-container within ~2 h, as the 60- and 81-run
   campaigns did — **don't obsess over per-arm time, just run them**.
3. **Campaign**: the resumable pool (`harness/run_cf_scan_local.py`) over the bundle, PLUS the
   `autocommit_cf_scan.sh` heartbeat — **load-bearing here** (the raw dicts are the deliverable and
   the container is ephemeral; a restart already cost one probe). The heartbeat commits + pushes
   finished arms every ~2 min. After any restart: re-emit the bundle (deterministic; the scratch
   bundle may survive) and re-run the same pool command — done arms are skipped via their
   `.exit_code` / the committed summary; already-committed `.jsonl.gz` are not re-run.
4. **Harvest** (the heartbeat runs this every tick; also run once at the end):
   `harness/harvest_cf_scan.py --csv <summary> --traj-dir <traj> --dicts-dir <dicts>` →
   committed summary + per-arm trajectory CSVs (t, R2, v2, rShell, phase; a lightweight index) +
   **the gzipped raw `dictionary.jsonl` per arm — the actual deliverable the maintainer reduces
   later**. Commit + push.
5. **Match**: frozen `match_runs.py` if present, else `harness/match_cf_scan.py` →
   `data/match_cf_PISM1e5.csv` + `_cells.csv`. Commit + push.
6. **Report** (in this README, dated): §5 adjudication FIRST, then per-cell best/interpolated
   Cf with the 3-point caveat, per-run compliance tally (any 124s listed as re-run debt).

## 4. Matching policy (fallback transcription — see §2 blocker 3)

Radii-only χ²: R2 ↔ 7±1 pc (cavity), rShell ↔ 19±2 pc, flat age prior 1.5–2.5 Myr; per run,
age minimising χ² inside the prior (clipped to t_final — runs truncate at different t).
**F-12 is not silently resolved**: every per-run and per-cell output carries both cavity bases
(7 pc and 6.2 pc) as `*_7` / `*_62` columns. Do not invent further observables.

## 5. Adjudicating `paper/rosette/PLAN.md` §0.3 (do this FIRST, before quoting any best Cf)

The scan carries what §0.3 demanded of whoever runs P6/P7: sealed Cf=1.0 rows in every cell, and
cooling_boost ON (fmix ∈ {1,4} — the pilot lacked it), at PISM=1e5. The cells CSV collates, at
**matched simulation time** t_match = min over the cell's quotable runs of min(2.5 Myr, t_final),
the per-run R2 overshoot on **both** the 7 pc and 6.2 pc bases (`over7_at_tmatch` /
`over62_at_tmatch`, ';'-aligned with `cf_grid`). The Phase-2 report must state whether the sealed
baseline still overshoots the cavity with fmix=4 before any Cf<1 number is quoted — the result
speaks to the finding, not around it.

## 6. Resolution: 3 Cf points is intentional

{0.70, 0.85, 1.0} brackets the pilot's ~0.89; `match_cf_scan.py` reports the vertex of the
parabola through the 3 (Cf, χ²) points (`cf_star_*`) as the interpolated best Cf — always quoted
with the caveat that it comes from 3 points, and flagged when non-convex or outside the bracket.
No finer sweep unless the maintainer asks after seeing χ²(Cf).

## 7. Artifacts & the 💾 rule

`paper/rosette/plots/` (the paper folder's 💾 destination) is gitignored — nothing written there
survives this container. So the committed home is **`docs/dev/rosette-cf/data/`**:

- `cf_scan_PISM1e5_summary.csv` — per-arm axes + exit/duration + final radii + quotable flag.
- `cf_scan_PISM1e5_traj/<arm>.csv` — lightweight (t, R2, v2, rShell, phase) index.
- `cf_scan_PISM1e5_dicts/<arm>.jsonl.gz` — **the RAW `dictionary.jsonl`, gzipped: the deliverable**
  the maintainer reduces later (`gunzip` first). Raw dicts are large — ~10 MB/arm, ~26 KB/snapshot
  (each snapshot carries the full shell-density arrays) → ~750 MB raw / **~280 MB gzipped** for 72.
  That is a deliberate, maintainer-requested bloat of this feature branch (2026-07-13: "the point
  of this rosette is so that i can have their dictionary.jsonl … save them too"). If the branch
  size becomes a problem, the leaner fallback is to drop the per-snapshot shell-profile arrays
  before gzip — but that is NOT done by default, since the maintainer wants the full dicts.
- `match_cf_PISM1e5*.csv` — fallback-matcher output (maintainer re-runs the frozen matcher offline).

Each CSV carries a provenance stamp + the exact command. The maintainer mirrors what they need into
`paper/rosette/plots/` on their machine.

## 8. Honesty gates (baked into the tooling)

- **Matched t**: fixed-age columns are empty beyond t_final (never extrapolated); cell
  comparisons happen at t_match (§5).
- **Separate processes**: every arm is its own `run.py` subprocess (trinity leaks module-level
  globals in-process).
- **📏 compliance**: a `--per-arm-timeout` kill writes exit 124 → excluded from every minimum,
  kept visibly in the summary as re-run debt. Never quote its Cf.
- **PROVISIONAL / IN-CONTAINER** headers on every artifact (not HPC — same caveat as
  theta5s/bench5); pins (`numpy<2`, …) untouched.

## 9. Open questions for maintainer review (answer before Phase 2)

1. ~~Provide the param file~~ **RESOLVED 2026-07-13** — supplied, committed at
   `docs/dev/rosette-cf/rosette_cf_survey_PISM1e5_fmix.param`, expansion verified 72/0-invalid
   (§2.1).
2. ~~Cell grouping~~ **RESOLVED 2026-07-13** — maintainer confirmed **per mass-pair**: cells key
   on (mass-pair, nCore, fmix, P_HII), 24 cells of 3 Cf, exactly what
   `harness/match_cf_scan.py` (`CELL_KEYS`) already implements. No change needed.
3. ~~Confirm the §4 policy transcription / frozen-matcher availability~~ **RESOLVED 2026-07-13**
   — `matching/` stays local; Phase 2 uses the fallback matcher, maintainer reconciles against
   the frozen `observables.py` offline (§2 blocker 3). The §4 transcription is still worth a
   confirming diff before any paper number.
4. Probe-arm choice (densest nCore, Cf=1.0, both fmix) assumes dense+sealed is the long pole —
   fine as a budget bound either way, but say if the pilot showed otherwise.

## 10. In-container long-run ops playbook (adapted from pdv-trigger)

The campaign spans an ephemeral container that WILL restart (one already did, killing the timing
probe). The pdv-trigger bench5/theta5s pattern makes it restart-survivable:

- **Resumable runner** — `run_cf_scan_local.py` skips any arm with a `.exit_code` marker or a
  quotable row in the committed summary; `harvest_cf_scan.py --dicts-dir` skips arms whose
  `.jsonl.gz` is already current. So re-running the exact launch command after a restart continues
  where it left off; no arm is redone once its dict is committed.
- **Autocommit heartbeat** — `autocommit_cf_scan.sh` is the SOLE git committer while running
  (no manual commits during the run → no index race): every ~2 min it harvests + `git add` the
  data dir + commits + pushes with backoff.
- **Re-arming after a restart** — a scheduled self-poke re-clones the branch and relaunches BOTH
  the runner and the heartbeat, so the campaign self-heals without a human. On the terminal state
  (72/72 committed) the schedule is removed. **Never premature-stop a healthy running arm** — a
  wall-killed arm (exit 124) is non-compliant re-run debt, not a result.
- **Report every tick / at terminal**: progress is the committed summary's row count; the run is
  done when all 72 arms have a `.jsonl.gz` (compliant or a flagged crash), not before.
