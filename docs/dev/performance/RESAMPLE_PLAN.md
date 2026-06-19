# Plan: bubble dMdt residual solve — drop the 60k dense-output resample (HOTPATH §F1)

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
> and full runs cost minutes-to-hours, so any diagnostic worth keeping must be
> saved as a committed artifact under `docs/dev/` (a CSV/table in
> `docs/dev/performance/data/`, or a harness/figure in
> `docs/dev/performance/harness/`) — never left in `/tmp`, the local-only
> `scratch/`, or an untracked `outputs/`. A future visit must be able to reproduce
> or compare against the numbers **without re-running**; record the exact config +
> command that produced each artifact.

**About this document**  (created 2026-06-18 — the 🔄 banner *requires* refreshing this on every visit.)
- **Status (2026-06-19):** 🟢 **F1 CLEARED — P3 applied (`24c6914`) and validated end-to-end. Ships.** Per-call (P0–P2) was necessary-not-sufficient; the **full-run equivalence gate (P5) is the decider, and F1 PASSES it on every config tested**: `mock_hybr` (~5e-6) plus the three stiff/sharp edge cases via the **matched-`t`** original-60k-vs-F1-coarse comparison (`data/f1edge_matched_comparison.csv`) — worst R2/Eb/rShell ≈ **6e-6** (`edge_hidens`, the dense/stiff case), ~500× inside the 0.3% gate; `simple_cluster` 5.7e-8, `edge_lowdens` 6.5e-9. **The convergence concern was physically sound but doesn't bite:** LSODA's adaptive stepping (rtol=1e-6) already resolves the stiff solution, so a 500-pt `t_eval` converges the `min_T`/`monotonic` gates to the same `dMdt` — the 60k was output over-resolution. **Two lessons recorded:** (a) the per-call gate could NOT have caught a full-run divergence — full-run is mandatory for changes to the residual; (b) the P4 `ab_fullrun` "divergence" was a harness artifact (two full sims in one process → trinity global-state leakage) — A/B must use SEPARATE processes (`harness/f1_fullrun_equiv.sh`), and that script's *final-state* comparison must use matched-`t` (its raw verdict false-flagged `simple` because the runs truncated at different times).
- **Per-call speedup (if F1 ships):** uniform ~1.5×/call (energy) / ~1.4×/call (implicit) across all 6 configs; the resample is fixed-size, so the degenerate payoff is a full-RUN effect.
- **Type:** plan — phased equivalence + timing study (config × method matrix, capture-replay reaching deep into the implicit phase), then promotion behind a tolerance gate.
- **Workstream:** `performance/` — this is HOTPATH §F1, the headline win. Branch **`fix/hotpath-resample`** (off `fix/hotpath-freewins`, which carries the §F2 wins).
- **Where it sits:** `HOTPATH_PLAN.md` §F1 → **this** (the detailed F1 plan + its harness/data).
- **Code it concerns:** `trinity/bubble_structure/bubble_luminosity.py` — `_get_velocity_residuals` (`:875`, the target), `_solve_bubble_structure` (`:106`, left **untouched**), `_create_radius_grid` (`:835`), `_get_bubble_ODE_initial_conditions` (`:926`), the final structure path `_bubble_luminosity` (`:492`, left **untouched**). Reached in the implicit phase via `run_energy_implicit_phase.py:720` → `solve_betadelta_pure` → `_solve_betadelta_hybr` (`get_betadelta.py:874`) → `get_residual_pure` (`:353`) → `get_bubbleproperties_pure` (`:398`) → `fsolve` (`:461`).
- **Linked files & data:** to be created under `harness/` + `data/`. Reuses `tools/bubble_audit/` (`load_state` reconstructs full params from a `TRINITY_BUBBLE_STATE_DUMP` pickle) and the shell-solver capture pattern (`docs/dev/shell-solver/harness/capture_replay_variants.py`). **Commit every CSV.**

Environment of record: **python 3.11.x, numpy 1.26.4 (`<2` pin), scipy 1.17.1, astropy 7.2.1, pytest 9.1.0** (container needs `pip install -e ".[dev]"`).

---

## The question

The dMdt `fsolve` (`bubble_luminosity.py:461`) calls `_get_velocity_residuals` (`:875`) many times per bubble solve. Each call builds the ~60k-point grid (`_create_radius_grid`, `:894`), runs `solve_ivp(LSODA, dense_output=True)`, and **resamples all ~60k points** via `sol.sol(r_array)` (`_solve_bubble_structure:157`) — but the residual it returns (`:908-921`) consumes only **four scalars**: `v[-1]`, `v[0]`, `np.min(T)`, `monotonic(T)`. The ~60k resample (microbench ~21 ms vs ~0.8 ms integration) is wasted.

**Can the residual be computed from a coarse solve — endpoints from the integrator's own nodes, `min_T`/monotonicity from a ~2000-point `t_eval` — with the converged `bubble_dMdt` (and downstream `LTotal`/`T_r_Tb`/`mass`) unchanged within tolerance, across every regime and deep into the implicit phase?**

---

## Mechanism / current state (verified 2026-06-18 against source + two real dumped solves)

> **Naming note (resolved 2026-06-18 — the "legacy" misnomer is fixed).** The two
> bubble functions this plan touches were **NOT deprecated** — they are the SOLE
> production path; "_legacy" was a leftover from a dropped plan, not a signal that
> a better path exists. **They have now been renamed** (commit on this branch):
> `_bubble_luminosity_legacy` → **`_bubble_luminosity`** (`:492`) and
> `_create_legacy_radius_grid` → **`_create_radius_grid`** (`:835`); callers, the
> two tests/tools that referenced them, and the stale `:480` comment (which named
> the deleted `_create_adaptive_radius_grid`) were updated too — `pytest` 535
> passed, behavior-preserving. This is distinct from the genuinely-legacy
> `_solve_betadelta_legacy` (`get_betadelta.py:604`), which keeps its name because
> it has a live default replacement (`hybr`). **F1 optimizes the live path.**

- **Grid is strictly decreasing** (`_create_radius_grid`, `:835`): `r_array[0]` = outer start `r2Prime`, `r_array[-1]` = inner end `R1`; `t_span=(r_array[0], r_array[-1])` (`:148`) integrates outward→inward.
- **Numerator** `v_array[-1]` (`:908`): `r_array[-1] == sol.t[-1]`, so `sol.sol(r_array[-1])` returns `sol.y[:,-1]` **bit-identically** (measured abs diff `0.0`, both states). With `t_eval` ending at `R1`, `sol.y[0,-1]` is the same value → **numerator bit-identical**.
- **Denominator** `v_array[0] = sol.sol(r_array[0])[0]` (`:908`): the current code uses the dense interpolant at the start, which differs from the IC `v_init` by **~1e-12 rel** under LSODA (state_0000 abs `8.08e-9`, rel `3.60e-12`; state_0001 `8.86e-9`, `6.55e-12`; **`0.0` under RK45**). `v_init` is the *exact* IC, known before integrating (`_get_bubble_ODE_initial_conditions`, `:926`). Replacing the dense-interp denominator with `v_init` shifts the residual by ~1e-12 rel — far below `fsolve`'s `xtol=1e-4` and the `_RESIDUAL_RTOL=1e-6` regime the code already declares acceptable.
- **`min_T` / `monotonic`** (`:910,919`): the only consumers that genuinely need the *profile*. Today read off the 60k grid; the fix reads them off a coarse `t_eval`. **The one non-trivial behavior change** (see Risks): `monotonic` uses the *strict* `operations.monotonic` (`operations.py:68`), so a coarse grid could in principle smooth over a dense-output single-point spike a 60k grid would catch and flip the `1e2` penalty. Held across a 0.5×–2× dMdt scan on a real state (60k vs 2000 vs 200 all agreed) — **must be confirmed at scale (P2)**.

---

## The fix — Option (b): coarse `t_eval` residual solve (rewrite `_get_velocity_residuals` ONLY)

Leaves `_solve_bubble_structure` (`:106`) and the final structure path (`:492`, which legitimately needs the 60k grid + dense `_sol` for the conduction zone, `:632`) **byte-identical**. Drops three costs per fsolve iteration: the 60k `sol.sol`, the `_create_radius_grid` build+clean, and the `dense_output` allocation.

```python
_RESIDUAL_NPTS = 2000   # coarse min-T/monotonicity grid; matches _CONDUCTION_NPTS precedent

def _get_velocity_residuals(dMdt_init, params, Pb, R1):
    r2Prime, T_r2Prime, dTdr_r2Prime, v_r2Prime = _get_bubble_ODE_initial_conditions(
        dMdt_init, params, Pb, R1)
    r2Prime_val = np.asarray(r2Prime).item()
    v_init    = np.asarray(v_r2Prime).item()
    T_init    = np.asarray(T_r2Prime).item()
    dTdr_init = np.asarray(dTdr_r2Prime).item()
    if not np.all(np.isfinite([v_init, T_init, dTdr_init])):
        return _SOLVER_FAIL_RESIDUAL
    r_eval = np.linspace(r2Prime_val, R1, _RESIDUAL_NPTS)   # decreasing; [-1]==R1==t_span[-1]
    try:
        sol = scipy.integrate.solve_ivp(
            fun=lambda r, y: _get_bubble_ODE(r, y, params, Pb),
            t_span=(r2Prime_val, R1), y0=[v_init, T_init, dTdr_init],
            method='LSODA', t_eval=r_eval,                  # NO dense_output, NO 60k grid
            rtol=_RESIDUAL_RTOL, atol=_BUBBLE_ATOL)
    except BubbleSolverError:
        return _SOLVER_FAIL_RESIDUAL
    if not sol.success:
        return _SOLVER_FAIL_RESIDUAL
    v_last  = sol.y[0, -1]                       # bit-identical to current numerator
    T_array = sol.y[1, :]
    residual = (v_last - 0) / (v_init + 1e-4)    # denominator from IC (~1e-12 rel vs current)
    min_T = np.min(T_array)
    if min_T < _T_INIT_BOUNDARY:
        return residual * (_T_INIT_BOUNDARY / (min_T + 1e-1))**2
    if np.isnan(min_T):
        return -1e3
    if not operations.monotonic(T_array):
        return 1e2
    return residual
```

**Equivalence claim (to be gated, not assumed):** numerator exact; denominator + `min_T`/`monotonic` within round-off (~1e-12) at the residual level; the **converged `bubble_dMdt`** and downstream (`bubble_LTotal`, `bubble_T_r_Tb`, `bubble_mass`) within a tolerance the harness measures (target ≤0.3%, the bound the residual solve already declares for its `_RESIDUAL_RTOL` choice).

---

## Verification design — method × config matrix (reaching deep into the implicit phase)

**Method axis** (`_RESIDUAL_NPTS` + variant), replayed on identical captured inputs:
| id | method | purpose |
|---|---|---|
| **baseline** | current 60k `dense_output` + `sol.sol(r_array)` | reference |
| M2000 | Option (b), `t_eval` N=2000 | recommended (conservative) |
| M1000 / M500 / M200 | Option (b), N=1000 / 500 / 200 | how coarse is safe? |
| Mnodes | Option (b), no `t_eval` — `min_T`/monotonic on `sol.t`/`sol.y` adaptive nodes only (~tens of pts) | cheapest; tests whether the profile sample is needed at all |

**Config axis** (6 configs / 4 regimes — the MIGRATION_PLAN set, all already known to reach 20 energy + 100 implicit):
`mock_hybr` (tiny, **cheapest to reach implicit** — iterate here first) · `probe_typical_hybr` (realistic flat) · `steep` (PL−2, gets neutral-region solves) · `simple_cluster`/`sfe0.3` (**degenerate default**, slowest) · `dense_flat` + `sfe0.6` (fill out flat-vs-steep + 2nd degenerate point).

**Phase depth (the user's requirement):** energy **20** + implicit **100** captured bubble solves per config, via the matrix phase gate (`N_ENERGY=20 N_IMPLICIT=100`, `_phase_counts`/`_done`, `capture_replay_variants.py:377-398`). Transition/momentum are beyond a tractable capture budget (a 45-min `sfe0.3` run reached 0 transition solves) → scope to energy+implicit, exactly as the shell harness did.

**Capture point + compare level:** monkeypatch **`get_bubbleproperties_pure`** (return the real result so the host trajectory stays byte-identical; capture is a side effect), gate on `params['current_phase'].value` (queryable through `BubbleParamsView`). **Compare at the `BubbleProperties` OUTPUT level** — `bubble_dMdt` (`:390`), `bubble_LTotal` (`:372`), `bubble_T_r_Tb` (`:373`), `bubble_mass` (`:375`) — because the residual change affects the *converged* dMdt, whose downstream effect the raw ODE-level `rel_n` would miss (the "necessary but not sufficient" gap MIGRATION_PLAN flagged). Also log the **`monotonic`-gate verdict per trial dMdt** (baseline vs variant) to count any acceptance flips.

**Two harness layers:**
1. **Residual-level** (fast, micro): replay old vs new `_get_velocity_residuals` on each captured `dMdt`; assert numerator exact, residual rel-diff ~1e-12, monotonic-verdict agreement. Reuses `tools/bubble_audit/load_state` + `validate.py:_solve` (already `solve_ivp(LSODA)`).
2. **Integration-level** (the real gate): run the full `get_bubbleproperties_pure` with baseline-residual vs variant-residual on each captured state; compare the 4 outputs + time both (`_time_call`, min of 5 reps).

---

## Phases

### P0 — Capture + baseline + harness (zero production change) — 🟡 partly DONE
Built `harness/{residual_variants,capture_replay_bubble,replay_from_dump,aggregate_p0}.py` + `run_p0_sweep.sh`. **`mock_hybr` captured + validated** (see "### P0 results" + the detailed task plan above). **Gate G0 (open):** capture reaches 100 implicit solves on ≥4 configs; per-call baseline + per-variant timings recorded. **Remaining:** the 5-other-config sweep at `N_IMPLICIT=100`.

### P1 — Choose `_RESIDUAL_NPTS` (residual + integration equivalence on the full matrix)
**P0 already collapsed most of P1's question:** on `mock_hybr` the converged `rel_dMdt` is **identical across M2000…M200** and the speed-vs-N curve is nearly flat, so the choice is about **robustness margin, not accuracy or speed**. P1 = run the full-matrix sweep (G0 data) and pick N:
- **recommend `M500`** (conservative — 2.5× over the conduction band's own `_CONDUCTION_NPTS=2000`-derived resolution scale, ample for `min_T`/monotonic) unless the sweep shows a config where small N moves `rel_dMdt`;
- consider **`Mnodes`** (cheapest, no `t_eval`) only if the sweep shows **0** cases where the strict-monotonic gate flips the accepted root (add the `monotonic_flip` diagnostic column for this check).
**Gate G1:** across **all** captured calls in every config, worst output-level `rel_dMdt` ≤ the G2 bound (≤0.3%; P0 saw 1e-6) at the chosen N, 0 new solver failures, `monotonic_flip` either absent or 0 at the chosen N. (The output-level `rel_dMdt` is binding — it subsumes any residual-level monotonic flip; see the §CSV-schema gap note.) Commit `data/master_p0_table.csv`.

### P2 — Per-call integration equivalence (NECESSARY, not sufficient)
> ⚠️ **This is NOT the decision gate** — it was mistakenly treated as one. A
> per-call `rel_dMdt` is measured at a *single* converged solve; it cannot see
> that a coarse `t_eval` under-resolves the `min_T`/`monotonic` gates and so flips
> fsolve steering on a *later, stiffer* step. The real decision gate is **P5
> full-run equivalence** (below). P2 passing is required but says nothing about
> whether F1 holds over a full evolution.

The captured CSVs already hold this: every row's `baseline` variant IS the 60k
dense-output path, and `rel_dMdt` is variant-vs-baseline on the *converged*
`get_bubbleproperties_pure` output. So G2 reads straight off `master_p0_table.csv`.
**Gate G2 (hard), at the chosen `_RESIDUAL_NPTS`:** worst converged `bubble_dMdt`
`rel` ≤ **0.3%** (and `rel_LTotal`/`rel_T_r_Tb`/`rel_mass` ≤0.3% / traceable)
across **all 12 cells** (6 configs × {energy, implicit}); `ok` = n/n (0 solver
failures the baseline didn't have); no monotonic-acceptance change that moves a
root (subsumed by `rel_dMdt`). **Decision procedure for N:** smallest N whose
worst-cell `rel_dMdt` is ≤ the gate with ≥10× margin → today's data picks **M500**;
fall back to a larger N (M1000/M2000) only if a sweep cell breaks the margin;
consider **Mnodes** only if *additionally* no cell shows a monotonic flip.

### P3 — Promote (rewrite `_get_velocity_residuals`)
**The patch is already drafted + harness-reviewed: see `P3_PRODUCTION_PATCH.md`**
(exact new function body + `_RESIDUAL_NPTS` constant + apply/validate/rollback +
the regression test to add). Apply it at the N chosen in P2. **Gate G3:** `pytest`
(+ `-m stress`) green; `test_bubble_solver_*` green; the new
`test_residual_resample.py` green; `_create_radius_grid`/`_solve_bubble_structure`
+ the structure/conduction path **diff-free** (only `_get_velocity_residuals`
changed); `ruff` F-rules + `mypy` clean. If the residual was `_create_radius_grid`'s
last caller, *flag* the newly-dead grid builder — don't delete it in the same change.

### P4 — Full-run speedup (the headline number)
A/B wall-time on a short `simple_cluster` (and `mock_hybr`): baseline vs F1, same `stop_t`, compare wall time + a snapshot-tolerance check (consumed scalars within G2 bound — not byte-identical, since dMdt shifts ~1e-12–0.3%). Record in `HOTPATH_PLAN.md` ledger + here. **Gate G4:** measurable wall-time reduction, snapshots within tolerance.

---

## P0 — detailed task plan + verified API contract (line-by-line, 2026-06-18)

Every reference below was re-verified against current source (no assumptions).
**Design:** in-process capture-and-replay (Design 1, like the shell harness) — one
live run per config; on each gated bubble call run the baseline + every variant,
compare + time, write a CSV row per variant; **return the baseline result so the
host trajectory stays byte-identical.** Plus dump 2–3 state pickles for an offline
reproduction harness.

### Verified API contract
- **Driver:** `trinity.main.start_expansion(params)` (`main.py:81`); phase set to
  `'energy'` (`:241`) then `'implicit'` (`:275`). Wrap in `try/except (SystemExit, BaseException)`.
- **Monkeypatch target:** `bubble_luminosity.get_bubbleproperties_pure(params)`
  (`:398`). **Energy** phase → `params` is the real `DescribedDict`; **implicit**
  phase → `params` is a **`BubbleParamsView`** (`get_betadelta.py:107`; `__slots__`,
  **no `keys()`**, overrides `cool_beta`/`cool_delta`/`bubble_dMdt` via `_MockValue`,
  delegates all else to `._params`).
- **Phase gate read:** `params['current_phase'].value` — works through the view
  (not an overridden key).
- **Variant injection:** monkeypatch `bubble_luminosity._get_velocity_residuals`.
  The dMdt fsolve's `velocity_residuals_wrapper` (`:458`) calls the **module global**,
  so the swap is picked up. (Variants must keep signature `(dMdt_init, params, Pb, R1)`.)
- **Recursion guard:** inside the hook, set `BL.get_bubbleproperties_pure = real_gbp`
  before calling it; restore the hook in `finally` (pattern: `bubble_conduction_convergence.py:160-180`).
- **Capture param_values from a view** (for the pickles): `real = getattr(params, '_params', params); for k in real.keys(): v = params[k].value` (effective values, incl. the beta/delta/dMdt overrides). Cooling cubes are unpicklable → skip + rebuild offline via the `load_state` recipe (`tools/bubble_audit/audit.py:55-68`).
- **Offline reconstruct:** `tools/bubble_audit/audit.py:load_state(pkl, base)` (`:38`) returns `(params, inputs, ref, meta)` — or a lean loader doing `read_param(base)` → override `param_values` → rebuild CIE+nonCIE cubes (`audit.py:55-68`) → `get_bubbleproperties_pure(params)`.
- **Compare fields** (`BubbleProperties`): `bubble_dMdt` (`:390`), `bubble_LTotal` (`:372`), `bubble_T_r_Tb` (`:373`), `bubble_mass` (`:375`); also log `bubble_Tavg`, `R1`, `Pb`, the 3 L-components for context.
- **Matrix phase-gate skeleton to copy:** `capture_replay_variants.py:88-105` (env `N_ENERGY`/`N_IMPLICIT`, `_PHASE_N`, `_phase_counts` Counter, `_max_phase_order`, `_MATRIX_MAX_S`) + `:371-414` (`_done`, `_CaptureDone`, the gate body). Adapt `_current_phase` to read `params['current_phase'].value` instead of the odeint `args`.
- **Config files (all exist, verified):** `param/simple_cluster.param` (sfe0.3, degenerate), `docs/dev/transition/harness/{mock_hybr,steep,dense_flat}.param`, `docs/dev/archive/betadelta/diagnostics/probe_typical_hybr.param`. **`mock_hybr` is cheapest to reach implicit — iterate there first.**

### Method variants (`residual_variants.py`)
`baseline` = the current `_get_velocity_residuals` (60k dense resample). `M{2000,1000,500,200}` = Option (b): `solve_ivp(LSODA, t_eval=linspace(r2Prime, R1, N))`, no `dense_output`; numerator `sol.y[0,-1]`, denominator the IC `v_init`, `min_T`/`monotonic` on `sol.y[1,:]`. `Mnodes` = Option (b) with no `t_eval` (`min_T`/monotonic on adaptive `sol.t`/`sol.y`). All keep the `_RESIDUAL_RTOL`/`_BUBBLE_ATOL`, the `_T_INIT_BOUNDARY` rejection, and the `_SOLVER_FAIL_RESIDUAL` failure contract identical to baseline.

### CSV schema (one row per captured-call × variant) — AS BUILT
`config, phase, call_index, variant, npts, bubble_dMdt, bubble_LTotal, bubble_T_r_Tb, bubble_mass, bubble_Tavg, R1, Pb, time_ms, rel_dMdt, rel_LTotal, rel_T_r_Tb, rel_mass, ok` (18 cols) — `rel_*` vs the baseline row for the same `call_index`; `time_ms` = min of K reps.
**Gap vs the original plan:** a `monotonic_flip` column (per the §Risks behaviour-change caveat) was specced but **not built**. It is *not* needed for the gate — a strict-monotonic flip that changed the accepted root would surface as a non-tiny `rel_dMdt` at the **output** level (the binding G2 metric), and the P0 data shows `rel_dMdt ≤ 1e-6`, i.e. **no flip ever changed the converged dMdt** on `mock_hybr`. **P1 adds `monotonic_flip` as a *diagnostic* only** (to *localise* any flip in a config where `rel_dMdt` does turn out npts-sensitive), not as a separate gate.

### Tasks (parallel)
- **Task 1 — `residual_variants.py` + `capture_replay_bubble.py`** (the core + the in-process harness). Validate on **`mock_hybr`** (`N_ENERGY=20 N_IMPLICIT=100`); commit `data/bubble_resample_mock_hybr.csv` + 2–3 state pickles under `data/states/`.
- **Task 2 — `aggregate_p0.py` + `run_p0_sweep.sh` + `replay_from_dump.py`** (master table from the CSV schema above; the 6-config sweep driver with per-config wall caps; an offline replay that loads a state pickle and runs all variants). Schema-defined → testable without Task 1's data.

### Gate G0
≥4 configs reach 100 implicit captures; per-call baseline + per-variant timings recorded (the ~21 ms microbench replaced by a **real** production fraction); one CSV/config + master committed (pickles gitignored — regenerable). Then P1 reads these.

### P0 results — 5/6 configs DONE (2026-06-18); `sfe0.6` running. Source: `data/master_p0_table.csv` (regenerate: `python harness/aggregate_p0.py`)

**M500 per-call speedup (baseline = the 60k resample) and worst `rel_dMdt`:**

| config | energy speed | implicit speed | worst rel_dMdt (any variant) |
|---|---|---|---|
| mock_hybr | 1.68× | 1.45× | 3.1e-6 |
| probe_typical_hybr | 1.53× | 1.39× | 2.7e-6 |
| steep | 1.52× | 1.46× | 1.8e-6 |
| dense_flat | 1.52× | 1.45× | 1.8e-6 |
| **simple_cluster (sfe0.3)** | 1.52× | 1.39× | 2.6e-6 |

**Findings — the cross-regime data overturns the "headline" hypothesis:**
1. **The per-call win is CONFIG-INDEPENDENT: ~1.5× energy, ~1.4× implicit, everywhere.** The degenerate `simple_cluster` (1.52×/1.39×) is no better than the tiny `mock_hybr` (1.68×/1.45×). Reason: the 60k resample is a **fixed-size** op (~baseline 1.1–1.5 s/call regardless of config), so removing it nets a constant per-call factor. The microbench's ~27× was the resample-vs-integrate ratio for *one* op; a full `get_bubbleproperties_pure` call also runs the fsolve loop + conduction + luminosity, so the call-level win is ~1.4–1.5×. **The degenerate payoff, if any, is a FULL-RUN effect (P4), not a per-call one** — `simple_cluster` spends the largest wall-time fraction in bubble calls.
2. **Accuracy is universal and ≪ the 0.3% G2 gate** — worst `rel_dMdt` across all 5 configs × 2 phases × 5 variants = **3.1e-6** (~1000× margin); `rel_LTotal/T_r_Tb/mass` all ≤1e-6; `ok` = n/n everywhere.
3. **`rel_dMdt` is npts-INSENSITIVE in [200, 2000] in EVERY cell** — M2000≡M1000≡M500≡M200 to the digit; only Mnodes occasionally differs (still ≤3e-6). ⇒ **P1 locked: `_RESIDUAL_NPTS = 500`** (conservative; accuracy is set by the integration, not N, so 500 buys robustness margin at no measurable cost). **G2 PASS.**

### Harness correctness review ✅ PASS (2026-06-18)
Read-through of `capture_replay_bubble.py` + `residual_variants.py` against
current source — the multi-hour sweep output is trustworthy:
- **Linchpin verified:** `get_bubbleproperties_pure` is **pure** — no `params[...]=`
  / `params.attr=` writeback anywhere in `bubble_luminosity.py`, so the ~12 repeated
  calls per gated call (baseline + 5 variants, each output + timing) don't perturb
  the host trajectory or each other's timing. The host receives the BASELINE
  `base_bp`, so the captured run is byte-identical to an un-instrumented one.
- **Variants faithful:** same ICs (`_get_bubble_ODE_initial_conditions`), same span
  (production `r_array` runs `r2Prime→R1`, so `t_span=(r2Prime,R1)` = the variants'),
  same LSODA + `_RESIDUAL_RTOL`/`_BUBBLE_ATOL`, same `(v_end)/(v_init+1e-4)` residual
  and `min_T`/nan/monotonic/`_SOLVER_FAIL_RESIDUAL` contract; `BubbleSolverError`
  caught directly (prod catches it one level down in `_solve_bubble_structure`).
- **Bookkeeping:** `rel_*` is vs the same-call baseline row; variant timing runs
  while that variant's residual is still installed; phase gate reads
  `params['current_phase'].value` through the view; state dump is view-aware.
- Minor (non-blocking): `call_idx` rebuilds a set over all rows each call (O(n²)
  total) — negligible at N≤120; left as-is.

### Status vs gates (2026-06-19)
- ✅ **G0 met (6 configs):** `mock_hybr` (86i = its natural max) + `probe_typical_hybr`/`steep`/`dense_flat`/`simple_cluster`/`sfe0.6` (100i each) in `master_p0_table.csv`.
- ✅ **G1/G2 (per-call) met — but reclassified necessary-not-sufficient:** worst per-call `rel_dMdt` 3.1e-6 ≪ 0.3%; npts-insensitive in [200,2000] → `_RESIDUAL_NPTS=500`. **Does NOT clear F1** (see P2 banner / the §Status line).
- ✅ **P3 applied** (`24c6914`); full normal `pytest` 538 green; `test/test_residual_resample.py` green.
- ✅ **P5 full-run equivalence — PASS (mock_hybr + 3 edge cases):** matched-`t` original-60k vs F1-coarse, worst R2/Eb/rShell ≈ 6e-6 (`edge_hidens`, dense/stiff), ≪ 0.3%. `simple_cluster` 5.7e-8, `edge_lowdens` 6.5e-9, `mock_hybr` ~5e-6. Data: `data/f1edge_matched_comparison.csv` + committed `f1edge_{orig,f1}_trajectories.csv` (reproduce without re-running). Method: SEPARATE `run.py` processes per version (`harness/f1_fullrun_equiv.sh`), compared at matched `t_now` (NOT final state — runs truncate at different times under the 1h cap).
- **→ F1 CLEARED. Ships.** Remaining housekeeping: fix `f1_fullrun_equiv.sh`'s comparison to matched-`t` (its built-in final-state verdict false-flagged `simple`); `harness/ab_fullrun.py` stays BUGGED (in-process global-state leakage) — do not use, prefer the separate-process script.

## Efficiency measurement plan (record every number — 💾)
- **Per-residual-call**: baseline (60k resample + grid build) vs Option (b) — `timeit`, isolate the resample + grid-build savings.
- **Per-bubble-call**: full `get_bubbleproperties_pure` baseline vs F1 (the fsolve × residual product).
- **Full-run wall time** (P4): the headline — does removing the resample actually speed production, and by how much, per regime.
- All into `data/` CSVs + the `HOTPATH_PLAN.md` ledger row for §F1 (currently "TBD (P0)").

---

## Risks
| risk | mitigation |
|---|---|
| **Strict `monotonic` flips** on a coarse grid (dense-output spike smoothed over) → different acceptance → different dMdt | P1 counts flips across all captured trial dMdt; `_RESIDUAL_NPTS=2000` conservative; if flips occur, switch the residual gate to the tolerant `_is_monotonic_or_tolerable` or raise N |
| Denominator `v_init` ~1e-12 ≠ byte-identical | measured ≪ `xtol=1e-4`/`_RESIDUAL_RTOL=1e-6`; G2 confirms converged dMdt within 0.3% |
| Coarse `t_eval` misses a real T dip affecting `min_T` rejection | NPTS sweep (P1) + the 0.3% integration gate (P2); 2000 pts resolves the conduction band (precedent: `_CONDUCTION_NPTS`) |
| Capture doesn't reach the implicit phase | matrix phase gate (proven in shell harness); `mock_hybr` reaches implicit cheaply; require 100 implicit on ≥4 configs |
| Changes the final structure solve | scope is `_get_velocity_residuals` only; P3 asserts `_solve_bubble_structure`/`_bubble_luminosity` diff-free |
| LSODA vs RK45 endpoint difference | documented (RK45 denom bit-identical); we keep LSODA, so the ~1e-12 is the worst case |

## Decisions for the maintainer
1. **Equivalence tolerance**: is ≤0.3% on converged `bubble_dMdt` (the bound the residual solve already tolerates for `_RESIDUAL_RTOL`) the right G2 bar, or stricter for published tracks?
2. **`_RESIDUAL_NPTS`**: ship the conservative 2000, or the smallest P1-safe value (e.g. 500) for more speed?
3. **Mnodes (no `t_eval`)**: if P1 shows adaptive-nodes-only is safe, it's the cheapest — adopt, or keep the fixed coarse grid for determinism?

## Out of scope
- `_solve_bubble_structure` and the final structure/conduction path (byte-identical).
- The shell solver, the betadelta solver, the §F2 wins (separate), §F3 (descoped).
- Warm-starting dMdt/R1 across segments (HOTPATH §F5) — orthogonal.

## References
- Target: `bubble_luminosity.py:875` (`_get_velocity_residuals`), residual `:908`, gates `:910-921`; resample `:157`; grid `:835`; IC source `:926`; untouched final path `:492,511,632`; tolerances `_RESIDUAL_RTOL=1e-6:87`, `_BUBBLE_ATOL=1e-10:79`, `_CONDUCTION_NPTS=2000:95`; strict gate `operations.py:68`.
- Implicit call path: `run_energy_implicit_phase.py:720` → `get_betadelta.py:583,874,353,396` → `bubble_luminosity.py:398,461`.
- Harness foundations: `tools/bubble_audit/{audit.py:load_state,validate.py:_solve}`; `_dump_bubble_state` `bubble_luminosity.py:314`; shell pattern `docs/dev/shell-solver/harness/capture_replay_variants.py` + `MIGRATION_PLAN.md` §P0-matrix.
