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

**Status (2026-07-13):** 🔵 Phase-1 plan + tooling committed (10 harness tests green); Phase-2
execution BLOCKED on the maintainer providing the param file (§2) and approving this plan.

## 1. Goal & context

Helix (HPC) is down, so the 72-run Rosette Cf scan for the application paper runs in this
container — the same playbook as the theta5s (81/81) and bench5 (60/60) in-container campaigns
(`docs/dev/transition/pdv-trigger/REPRODUCE.md` rows 41/43). Scan:
**2 mass-pairs × 3 nCore × 3 Cf {0.70, 0.85, 1.0} × 2 cooling_boost_fmix {1, 4} × 2 P_HII = 72
runs** at PISM=1e5, stop_t=3 Myr, from `param/rosette_cf_survey_PISM1e5_fmix.param`. Deliverable:
the covering fraction Cf best matching the Rosette observables, per cell, with the
`paper/rosette/PLAN.md` §0.3 sealed-baseline finding adjudicated first (§5), not tuned around.

## 2. What this container verified vs. what it could NOT (blockers)

Verified against source @ `3b9be89`:

- The knobs exist on main: `coverFraction` (`trinity/_input/default.param:71`, Cf=1 sealed
  Weaver limit), `cooling_boost_fmix` + `cooling_boost_mode multiplier`, `PISM`, `stop_t`.
  NOTE the default.param INFO line calls Cf "usable range ~0.9–0.99; Cf near 0 drains the bubble
  within a step" — the grid's 0.70 is *below* that stated comfort band. Expect the integrator to
  be stressed there; a crashed 0.70 arm is a finding, not a tooling bug (it leaves the cell as a
  flagged 2-point fit, §6).
- Sweep expansion of those five axes yields exactly 72 unique combos, and the output-folder
  names DO encode the axes (checked by running `generate_combinations_from_config` on a synthetic
  config): e.g. `1e5_sfe001_n5e3_PL0_yesPHII_coolingBoostFmix1_coverFraction0p7`. The
  `--emit-jobs` manifest additionally carries every run's full params dict
  (`trinity/_input/sweep_jobs.py`), so matching never parses folder names.
- `--emit-jobs` bundles use the same `.exit_code`/`.duration` sentinels as the bench5 pool, and
  emit-time validation (GMC plausibility) runs before anything is queued.

NOT verifiable here — **gitignored, local-only on the maintainer's machines**
(`paper/CLAUDE.md`, `.gitignore`), so treat every statement about them in §4–§5 as transcribed
from the 2026-07-13 task brief, not checked against source:

1. **`param/rosette_cf_survey_PISM1e5_fmix.param` — HARD BLOCKER.** `param/*` is gitignored;
   the file exists only on the maintainer's machine. Phase 2 cannot start until it is committed
   (e.g. `git add -f`) or pasted into the session. Gate: its `--dry-run` must show exactly 72
   combos, 0 implausible. Its mass-pairs/nCore/P_HII values are unknown here — this plan
   deliberately encodes none of them.
2. **`paper/rosette/PLAN.md`** (§0.3 pilot: best-fit Cf≈0.89 in one corner; §0.4 machinery;
   F-12 cavity-target conflict) — unreadable here; §5 encodes the brief's summary of it.
3. **`paper/rosette/matching/{observables.py, match_runs.py}`** — the FROZEN matching policy.
   `harness/match_cf_scan.py` is a fallback **reimplementation** of the policy as stated in the
   brief (§4). If the frozen matcher is available in the Phase-2 container, prefer it; either
   way, diff the fallback's `POLICY` constants against the frozen `observables.py` before
   quoting any number.

## 3. Phase-2 execution plan (pure execution; exact commands in `harness/README.md`)

0. **Env**: `pip install -r requirements.txt` (keep the `numpy<2` etc. pins);
   `pytest test/test_rosette_cf_harness.py -q` (10 tests) must pass.
1. **Preflight**: `run.py <param> --dry-run` → expect 72/0-invalid, then
   `--emit-jobs "$WS/cf_jobs"`. Any invalid combo or count ≠ 72 → STOP, report, don't run.
2. **Timing probes** (brief's requirement — the Helix ~1 h/run figure was for full-length runs;
   stop_t=3 Myr runs are much shorter, so **measure, don't guess**): run the 2 densest-nCore,
   Cf=1.0 arms (fmix 1 and 4) via `--only/--limit`, `--workers 1`. Set `--per-arm-timeout`
   ≈ 6× the measured per-arm time; workers = 3 (bench5 default; raise only if probe time × 72 /
   workers ≫ container window). Projected wall = probe-mean × 72 / workers.
3. **Campaign**: the resumable pool (`harness/run_cf_scan_local.py`) over the bundle. Arm the
   `autocommit_cf_scan.sh` heartbeat **only if** projected wall > ~1.5 h (bench5's glue is
   load-bearing insurance for multi-window campaigns, dead weight for a sub-hour one); otherwise
   harvest+commit once at the end. After any restart: re-emit the bundle (deterministic) and
   re-run the same pool command — done arms are skipped via the committed summary.
4. **Harvest**: `harness/harvest_cf_scan.py` → committed summary + per-arm trajectory CSVs
   (t, R2, v2, rShell, phase — everything the radii-only policy needs; raw jsonl stays
   ephemeral). Commit + push.
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

## 7. Artifacts & the two 💾 rules

`paper/rosette/plots/` (the paper folder's 💾 destination, mirroring
`match_pilot_cf_survey_2026-07-08.csv`) is gitignored — nothing written there survives this
container. So the committed home is **`docs/dev/rosette-cf/data/`** (summary, traj dir, match
CSVs, each with a provenance stamp + exact command); the maintainer mirrors the match CSVs into
`paper/rosette/plots/` on their machine. Full raw jsonl stays ephemeral/local by design.

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

1. Provide `param/rosette_cf_survey_PISM1e5_fmix.param` (§2.1) — and confirm its expansion
   matches §1's 72.
2. Cell grouping: the brief says min-χ² Cf per "(nCore, fmix, P_HII) cell"; the cells CSV keys
   on (mass-pair, nCore, fmix, P_HII) — 24 cells of 3 — since lumping the two mass-pairs would
   mix physically different clouds. Coarser grouping is a trivial post-hoc aggregation; confirm
   which the paper wants.
3. Confirm the §4 policy transcription against the frozen `observables.py`, and whether the
   frozen `match_runs.py` will be available in the Phase-2 container.
4. Probe-arm choice (densest nCore, Cf=1.0, both fmix) assumes dense+sealed is the long pole —
   fine as a budget bound either way, but say if the pilot showed otherwise.
