# F1 — "drop the 60k dense-output resample" — comprehensive summary

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

---

**Status (2026-06-19): 🟢 SHIPPED & fully validated** on branch `fix/hotpath-resample`.
Figures: `figs/f1_*.png` (regenerate: `python harness/make_f1_figures.py`).

## 1. The change
`_get_velocity_residuals` (the dMdt-fsolve hot loop, thousands of calls/run) used to
build a **~60k-point grid** (`_create_radius_grid`: three stitched `logspace(2e4)`
chunks) and resample a dense `solve_ivp` solution onto it (`_solve_bubble_structure`
→ `sol.sol(r_array)`). F1 integrates **once on a coarse `t_eval` of `_RESIDUAL_NPTS=500`**
and drops the resample. Only the residual path changed; the structure/conduction path
still uses the 60k grid.

| # | category | item | detail | commit |
|---|---|---|---|---|
| 1 | **production** | `_get_velocity_residuals` body | 60k `_create_radius_grid`+`_solve_bubble_structure` resample → one `solve_ivp(t_eval=linspace(r2Prime,R1,500))` | `24c6914` |
| 2 | **production** | `_RESIDUAL_NPTS = 500` (new constant) | coarse output grid for the residual solve | `24c6914` |
| 3 | test | `test/test_residual_resample.py` + 5.4 KB fixture | regression guard: prod residual vs inline 20k-dense ref | `3cf1061` |
| 4 | harness | `residual_variants.py`, `capture_replay_bubble.py`, `replay_from_dump.py`, `aggregate_p0.py`, `run_p0_sweep.sh` | per-call capture / replay / 6-config sweep | P0 |
| 5 | harness | `f1_fullrun_equiv.sh` | full-run A/B, **separate processes**, **matched-`t`** compare | `41557d4`,`cfd5493` |
| 6 | harness | `ab_fullrun.py` | ⚠️ BUGGED (in-process global-state leakage) — kept as cautionary example | `cfd5493` |
| 7 | harness | `make_f1_figures.py` | figure generator (committed data → `figs/`) | `7e64dc5` |
| 8 | configs | `f1edge_lowdens_himass_hisfe.param`, `f1edge_hidens_himass_losfe.param` | stiff/sharp edge cases (GMC-validated) | `a152d94` |
| 9 | data | `master_p0_table.csv`, `f1edge_{orig,f1}_trajectories.csv`, `f1edge_matched_comparison.csv` | committed diagnostics (reproduce w/o re-run) | various |
| 10 | docs | `RESAMPLE_PLAN.md`, `HOTPATH_PLAN.md`, `P3_PRODUCTION_PATCH.md`, `harness/README.md`, this file | plan + patch + summary | various |

## 2. Tests & validation (all green)
| level | test / harness | what it checks | result |
|---|---|---|---|
| per-call (P0–P2) | sweep → `master_p0_table.csv` | converged `rel_dMdt` vs the 60k baseline, 6 configs × 2 phases × 5 variants | worst **3.1e-6** ≪ 0.3% |
| regression | `test_residual_resample.py` | prod residual vs a 20k-dense reference for several `dMdt` | 3 pass (≤1e-6) |
| unit | full `pytest` (normal set) | no regressions repo-wide | **538 passed** |
| **full-run (P5)** | `f1_fullrun_equiv.sh` (matched-`t`) | R2/Eb/rShell trajectories, original-60k vs F1-coarse, mock + 3 edges | worst **6e-6** ≪ 0.3% |
| stress (G3) | `test_betadelta_hybr_stress` | hybr end-to-end + **matches golden** under stress | 2/2 pass |
| stress (G3) | `test_bubble_solver_stress` | **0 nondeterministic LSODA crashes** (15 smoke runs) | 1/1 pass |

**Lesson:** the per-call gate (P2) is *necessary but not sufficient* — it cannot see a
full-run divergence. Only P5 (full-run) can clear a change to the residual. The 60k turned
out to be *output over-resolution*: LSODA's adaptive stepping (rtol 1e-6) already resolves
the stiff solution, so 500 points converge the `min_T`/`monotonic` gates to the same `dMdt`.

## 3. Defaults / constants (`bubble_luminosity.py`)
| constant | value | role | F1 effect |
|---|---|---|---|
| **`_RESIDUAL_NPTS`** | **500** | **NEW** — coarse output grid for the residual solve | **added** |
| `_create_radius_grid` | ~60k pts | OLD residual resample grid | **no longer called by the residual** (still used by structure path) |
| `_RESIDUAL_RTOL` | 1e-6 | residual `solve_ivp` rtol (loose; only locates dMdt) | unchanged |
| `_BUBBLE_ATOL` | 1e-10 | `solve_ivp` atol | unchanged |
| `_BUBBLE_RTOL` | 1e-8 | structure-solve rtol | unchanged |
| `_CONDUCTION_NPTS` | 2000 | conduction-band sampling (structure path) | unchanged |
| `_T_INIT_BOUNDARY` | 3e4 | residual `min_T` floor + IC anchor | unchanged |
| `_SOLVER_FAIL_RESIDUAL` | 1e3 | deterministic fsolve fail penalty | unchanged |
| P1 sweep N grid | M2000/M1000/M500/M200/Mnodes | npts candidates; `rel_dMdt` identical across [200,2000] → **500 = conservative pick** | — |

## 4. Efficiency
Per-call from `master_p0_table.csv`; full-run from the A/B (F1 reaches further `t` in the
same wall-clock). Full-run rel-diff from `f1edge_matched_comparison.csv`.

| config | per-call speedup (energy / implicit) | full-run speedup | worst full-run rel-diff (R2/Eb/rShell) |
|---|---|---|---|
| mock_hybr | 1.68 / 1.45 | ~1.3× | ~5e-6 |
| probe_typical_hybr | 1.53 / 1.39 | — | — |
| steep | 1.52 / 1.46 | — | — |
| dense_flat | 1.52 / 1.45 | — | — |
| simple_cluster (sfe0.3) | 1.52 / 1.39 | **~2.3×** | 5.7e-8 |
| sfe0.6 | 1.43 / 1.41 | — | — |
| edge_lowdens (low-ρ/hi-M/hi-sfe) | — | — | 6.5e-9 |
| edge_hidens (hi-ρ/hi-M/lo-sfe, stiff) | — | — | 6.0e-6 |

**Headline:** uniform **~1.5×/call** (the 60k resample is fixed-size, so the per-call win
is config-independent); the degenerate `simple_cluster` shows **~2.3× full-run** because it
spends the largest wall-time fraction in bubble calls.

## 5. Reproduce (no re-running the sims)
```bash
python docs/dev/performance/harness/aggregate_p0.py        # per-call master table
python docs/dev/performance/harness/make_f1_figures.py     # all 4 figures from committed CSVs
# full A/B from scratch (slow, ~2h): bash docs/dev/performance/harness/f1_fullrun_equiv.sh
```
