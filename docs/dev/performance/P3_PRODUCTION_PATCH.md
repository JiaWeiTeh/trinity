# P3 — staged production patch: coarse-`t_eval` `_get_velocity_residuals`

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

**Status (2026-06-19):** 🟢 **APPLIED (`24c6914`) and CLEARED — F1 ships.** This is the
production rewrite in `bubble_luminosity.py`. Validated end-to-end: per-call (P0–P2),
`mock_hybr` full-run (~5e-6), and the **full-run matched-`t` equivalence on the three
stiff/sharp edge cases** (`simple_cluster`, `f1edge_lowdens`, `f1edge_hidens`) — worst
R2/Eb/rShell ≈ 6e-6, ~500× inside the 0.3% gate (`data/f1edge_matched_comparison.csv`).
The 60k turned out to be output over-resolution: LSODA's adaptive stepping (rtol=1e-6)
already resolves the stiff solution, so the 500-pt `t_eval` converges the `min_T`/
`monotonic` gates to the same `dMdt`. §Rollback below stays valid if ever needed. NB:
the earlier P4 `ab_fullrun` "divergence" was an in-process harness artifact, not this
patch; the correct A/B is `harness/f1_fullrun_equiv.sh` compared at matched-`t`.

## What changes (one function + one constant)

`_get_velocity_residuals` (`trinity/bubble_structure/bubble_luminosity.py`,
currently lines 876–924) is the only production change. Today it builds
`_create_radius_grid(R1, r2Prime)` (~60k points: three stitched `logspace(2e4)`
chunks) and calls `_solve_bubble_structure(...)`, which integrates with
`solve_ivp(dense_output=True)` and then **resamples the continuous solution onto
the 60k grid via `sol.sol(r_array)`**. The residual only needs `v` at the two
endpoints plus the `min_T` / monotonic guards along the path — it never uses the
dense `sol` object (only the conduction path at ~line 633 does). So the fix
integrates **once on a coarse `t_eval`** and drops the 60k resample entirely.

Verified equivalences (so this is a faithful swap, not a behaviour change):
- ICs via the same `_get_bubble_ODE_initial_conditions` → identical.
- Integration span identical: production `r_array` runs `r2Prime → R1`
  (decreasing), so its `t_span=(r_array[0], r_array[-1]) = (r2Prime, R1)`; the
  patch uses `t_span=(r2Prime, R1)` with `t_eval=linspace(r2Prime, R1, N)`.
- Same solver/tolerances: `method='LSODA'`, `rtol=_RESIDUAL_RTOL`, `atol=_BUBBLE_ATOL`.
- Numerator `v_array[-1]` (v at R1) and denominator `v_array[0]` (v at r2Prime,
  == `v_init`) identical; same `min_T`/`nan`/`monotonic` gates and the same
  `_SOLVER_FAIL_RESIDUAL` / `-1e3` / `1e2` return contract.
- `BubbleSolverError` (raised by `_get_bubble_ODE` when T collapses) is caught
  directly — production catches it one level down in `_solve_bubble_structure`.
- Non-finite ICs → `_SOLVER_FAIL_RESIDUAL`, matching `_solve_bubble_structure`'s
  `isfinite` guard.

**Evidence it's safe:** P0 `data/bubble_resample_mock_hybr.csv` — worst converged
`rel_dMdt` 1.0e-6 (M*) / 2.2e-6 (Mnodes), ≪ the 0.3% G2 gate; npts-insensitive in
[200, 2000]. Lock the final `_RESIDUAL_NPTS` against the full sweep (P1).

## The patch

**(1) Add the constant** next to the other residual constants (after
`_RESIDUAL_RTOL = 1e-6`, ~line 87):

```python
# Coarse output grid for the dMdt velocity-residual solve. This solve only
# LOCATES dMdt for the fsolve, so it integrates once on this many points instead
# of resampling a dense-output solution onto the ~60k _create_radius_grid. P0/P1
# (docs/dev/performance) show the converged dMdt is insensitive to this in
# [200, 2000] (rel_dMdt <= 1e-6 vs the 60k grid); 500 is the conservative pick.
_RESIDUAL_NPTS = 500
```

**(2) Replace the body** of `_get_velocity_residuals` (lines 876–924) with:

```python
def _get_velocity_residuals(dMdt_init, params, Pb: float, R1: float) -> float:
    """Calculate velocity residual for dMdt solver."""
    r2Prime, T_r2Prime, dTdr_r2Prime, v_r2Prime = _get_bubble_ODE_initial_conditions(
        dMdt_init, params, Pb, R1
    )
    # numpy 2.x: float(size-1 1-d array) errors, so coerce through .item()
    r2Prime_val = np.asarray(r2Prime).item()
    v_init      = np.asarray(v_r2Prime).item()
    T_init      = np.asarray(T_r2Prime).item()
    dTdr_init   = np.asarray(dTdr_r2Prime).item()

    # This solve only LOCATES dMdt for the fsolve (looser _RESIDUAL_RTOL), so it
    # integrates once on a coarse t_eval grid rather than resampling a dense
    # solution onto the ~60k _create_radius_grid. Integration accuracy is set by
    # rtol/atol; the residual only needs v at the endpoints plus the min_T /
    # monotonic guards along the path. A failed solve returns a deterministic,
    # large penalty so fsolve is steered away from this dMdt (see _RESIDUAL_RTOL).
    if not np.all(np.isfinite([v_init, T_init, dTdr_init])):
        return _SOLVER_FAIL_RESIDUAL
    try:
        sol = scipy.integrate.solve_ivp(
            fun=lambda r, y: _get_bubble_ODE(r, y, params, Pb),
            t_span=(r2Prime_val, R1),
            y0=[v_init, T_init, dTdr_init],
            method='LSODA',
            t_eval=np.linspace(r2Prime_val, R1, _RESIDUAL_NPTS),
            rtol=_RESIDUAL_RTOL,
            atol=_BUBBLE_ATOL,
        )
    except BubbleSolverError:
        return _SOLVER_FAIL_RESIDUAL
    if not sol.success:
        return _SOLVER_FAIL_RESIDUAL

    v_array = sol.y[0]
    T_array = sol.y[1]

    residual = (v_array[-1] - 0) / (v_array[0] + 1e-4)

    min_T = np.min(T_array)
    if min_T < _T_INIT_BOUNDARY:
        logger.debug(f'Rejected. min T: {min_T}')
        return residual * (_T_INIT_BOUNDARY / (min_T + 1e-1))**2

    if np.isnan(min_T):
        logger.debug('Rejected. nan temperature')
        return -1e3

    if not operations.monotonic(T_array):
        logger.debug('Temperature not monotonic')
        return 1e2

    return residual
```

Nothing else changes. `_create_radius_grid` and `_solve_bubble_structure` stay
(the structure/conduction path still uses them); only the residual stops calling
them. After applying, check whether `_create_radius_grid` has any remaining
callers — if the residual was its last one, flag it as newly-dead (don't delete
pre-existing dead code unprompted; that's a separate call).

## Apply + validate (the instant G0/P1 clear)

1. Apply (1) + (2) above. `git diff` should touch only those two spots.
2. **Equivalence gate (G2):** re-run the relevant capture(s) — the production
   `baseline` row is now the coarse path, so instead diff against a saved 60k
   reference. Concretely: `python harness/replay_from_dump.py <state>.pkl` on the
   committed-regenerated states, asserting the new production residual's converged
   `bubble_dMdt` is within **0.3%** of the pre-patch 60k value on every captured
   state (P0 saw ≤1e-6). 0 new solver failures; monotonic-acceptance unchanged.
3. `pytest` (full) green; `ruff check --select F821,F811,F823,E9 trinity` clean;
   `mypy trinity` no new errors. Do **not** run `black` (installed v26 reformats
   the repo — see CLAUDE.md).
4. **Add a regression test** `test/test_residual_resample.py`: pin a tiny
   committed fixture of representative `(param_values, Pb, R1, dMdt)` ICs distilled
   from one captured state (a small `.npz`/`.json`, NOT the gitignored ~3 MB
   pickle) and assert the coarse-`N` residual matches a high-resolution
   (`N=20000`) reference within `1e-3` for several trial `dMdt`, and that the
   `min_T`/monotonic branches still fire on a constructed collapsing-T case.
5. Spot-run one real config end-to-end (`python run.py param/simple_cluster.param`)
   and confirm the run reaches its stopping fate with no monotonic-guard rejection
   regression vs a pre-patch run.

## Rollback

Self-contained: revert the two hunks (restore the `_create_radius_grid` +
`_solve_bubble_structure` body and drop `_RESIDUAL_NPTS`). No data migration, no
schema/state change — a one-commit revert.
