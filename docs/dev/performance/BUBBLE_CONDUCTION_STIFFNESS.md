# Bubble-structure conduction-layer stiffness → LSODA `t+h=t` flood (cause + deferred fix)

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
> saved as a committed artifact under `docs/dev/` — never left in `/tmp`, the
> local-only `scratch/`, or an untracked `outputs/`. A future visit must be able
> to reproduce or compare against the numbers **without re-running**; record the
> exact config + command that produced each artifact.

**Status (2026-06-19):** 🟡 **CAUSE diagnosed; symptom mitigated, cause deferred.** The
LSODA `t+h=t` flood on massive low-density clouds (e.g. `mCloud=5e9, sfe=0.01, nCore=1e2`)
is the **bubble structure solver**, not the shell ODE. The noise is suppressed
(`_quiet_lsoda_fortran` around the bubble `solve_ivp`, `bubble_luminosity.py`); the
**underlying stiffness is left as the tracked item below** because the solve is *verified
correct*, so it is noise — fixing the cause is an optional robustness/perf improvement, not a
correctness bug.

## Symptom
```
lsoda--  warning..internal t (=r1) and h (=r2) are such that t + h = t on the next step
      in above,  r1 =  0.5602992680422D+01   r2 =  0.2915185267840D-15
```
repeated thousands of times. `r1` is a bubble radius; `r2` (~1e-15) is the underflowing step.

## Root cause (measured 2026-06-19, on `main`)
- The flood is `solve_ivp(LSODA)` in `_solve_bubble_structure` and `_get_velocity_residuals`
  (`bubble_luminosity.py`), **not** the shell solve. Detected by counting consecutive RHS
  re-calls at the same radius (the `t+h=t` condition): `stuck_count=522` at a span endpoint,
  on **both** the 60k and the 500-coarse (`_RESIDUAL_NPTS`) grids — the grid is output
  sampling; the flood is in the adaptive integration, so #698's coarse grid is orthogonal.
- It is **stiffness, not overflow** — every value is finite. The bubble temperature ODE
  `_get_bubble_ODE` (`bubble_luminosity.py`, Eq 42-43 Weaver+77) has
  `dTdrr = Pb/(C_thermal·T^{5/2})·(...) − 2.5·dTdr²/T − 2·dTdr/r`. At the boundary the
  measured state is `T≈3e4`, `dTdr≈−6.19e13`, giving the dominant term `−2.5·dTdr²/T ≈ −3.2e23`
  and `dTdrr ≈ −1.76e23`. With a second derivative that large, LSODA's error control demands a
  step so small that `t+h=t`.
- The enormous gradient traces to the **conduction-layer thickness** initial condition
  (`_get_bubble_ODE_initial_conditions`): `dTdr = −2/5·T/dR2` with
  `dR2 = T_init^{5/2} / (const·dMdt/(4πR2²))`. For this config `dR2 ≈ 1.9e-10 pc` — a
  temperature front **2×10⁻¹⁰ pc thick**, far below any adaptive stepper's reach. The thin
  layer is driven by the huge wind `dMdt` from the `sfe·mCloud = 5×10⁷ M☉` cluster.

## Correctness (why this is noise, not a bug)
On the exact flooding solve, production LSODA (`rtol=1e-8`) matches two independent stiff
references — **LSODA `rtol=1e-12`** and **Radau `rtol=1e-10`** — to **~1e-6** on `[v, T, dTdr]`,
profile physically sane (`T` 3e4→1.1e8 K, monotonic, finite). #698 separately showed the
500-coarse residual matches the 60k grid to `rel_dMdt ≤ 3e-6`. So LSODA *does* cross the thin
layer correctly in sub-steps; it just prints a warning each time. **The answer is right.**

## Shipped mitigation
`_quiet_lsoda_fortran()` redirects C-level stdout/stderr to /dev/null around the two bubble
`solve_ivp(LSODA)` calls. Numerically inert (verified byte-identical `dictionary.jsonl` on
`simple_cluster`); genuine failures still caught by `sol.success` → `BubbleSolverError`.
Pinned by `test/test_bubble_lsoda_quiet.py`.

## Robustness of the current (unfloored) treatment — pinned (2026-06-20)
The "no floor, exact analytic `dR2`" choice is now characterised by
`test/test_dR2min_magic_number.py` (vs WARPFIELD's hand-tuned `dR2min=1e-7`, bumped
`1e-14*Mclus+1e-7` for `Mclus>1e7`): (a) `dR2` is the exact `1/dMdt` layer with **no clamp**
across bubble size × 8–10 decades of `dMdt`; (b) `R2 − dR2` is well-conditioned down to the
thinnest *physical* layer (`dR2/R2 ~ 3e-11`), clearing the float64 cancellation cliff
(`~ε/2`) by **~5.5 decades**; (c) production LSODA matches an independent **Radau** reference
to **~3e-8** on `T`/`dTdr` on two real captured states — a mild cluster *and* a genuinely-stiff
one captured from **this very config** (`5e9/sfe0.01/n1e2`, `dR2/R2 ~ 1.2e-10`, the flood regime;
fixture `test/data/dR2_stiff_state_fixture.json` + `conduction_stiff_5e9_sfe001.param`, captured by
`docs/dev/performance/harness/capture_stiff_dR2_state.py` — whose 40-solve scan also shows the stiff
config's bubble solves complete cleanly segment after segment). And the **whole** production solver
(`get_bubbleproperties_pure`) returns a physical result on that state. So the unfloored layer is
*integrated correctly*, not just quiet. This is the executable form of the §Correctness measurement above.

Figures (`docs/dev/performance/figs/make_dR2_figures.py`, regenerate from the committed fixtures):
`dR2_idea.png` (analytic layer vs WARPFIELD's floored+bumped `dR2min` — ~10³× over-thick for massive
clusters), `dR2_envelope.png` (`dR2/R2` vs cluster mass, the float64 cancellation cliff at `ε/2` with a
~6-decade margin), `dR2_crosssolver.png` (LSODA vs Radau across the stiff thin layer, residual ~3e-8 vs
the 1e-5 test bar; data `docs/dev/performance/data/dR2_crosssolver_residual.csv`).

## Deferred — fixing the CAUSE (optional; needs its own gate)
Each would *reduce the stiffness* rather than hide it, and is **not** correctness-required
(the result is already correct). Per the CLAUDE.md planning protocol, any of these needs a
full-run equivalence gate on the stiff edge regimes (this config + the f1edge params) before shipping:
1. **Floor `dR2`** to a resolvable thickness (e.g. a small multiple of the grid spacing) so the
   initial `dTdr` is finite-steep. Changes the initial condition → must prove the sampled profile
   and `dMdt` stay within tolerance of the current (verified-correct) result. **NB:** the
   robustness tests above show this floor is *not needed* for any physical cluster — only pursue it
   as a perf/noise nicety, and gate the IC change against those tests.
2. **Analytic conduction-layer treatment** — integrate the sub-grid layer in closed form and start
   the ODE just outside it. Most principled, biggest change.
3. **Input validation / warning** when `dR2` underflows the grid (flag the regime as
   numerically marginal) — cheapest, honest, doesn't change results.

## Reproduce
- Config: `mCloud 5e9 / sfe 0.01 / nCore 1e2 / rCloud_max 1e9`.
- Detector (counts the `t+h=t` condition directly, since this container does not surface the
  Fortran lines): wrap `scipy.integrate.solve_ivp`/`odeint`, flag ≥12 consecutive RHS calls with
  `|Δr| < 1e-13`. Measured: first bubble solve floods immediately (`stuck_count=522`), shell
  solves clean.
