# Plan: hot-path performance & conditioning — the next solver-class wins

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
> saved as a committed artifact (a CSV/table under `docs/dev/performance/data/`,
> or a force-added harness/figure in this folder) — never left in `/tmp`, the
> local-only `scratch/`, or an untracked `outputs/`. A future visit must be able
> to reproduce or compare against the numbers **without re-running**; record the
> exact config + command that produced each artifact.

**About this document**  (created 2026-06-18 — the 🔄 banner *requires* refreshing this on every visit; it is a living doc, not frozen.)
- **Status (verified 2026-06-18):** 🔵 **ACTIONABLE — nothing shipped yet.** This is a fresh hot-path audit prompted by "what else can one do?" after the hybr (`archive/betadelta/`) and shell-solver (`shell-solver/`) wins. Every finding below was traced to current source (file:line) and, where a number is quoted, to a subagent microbenchmark noted inline. **The headline (§F1) is a measured 27× overhead and is the next solver-class win; §F2 is a batch of bit-identical free wins; §F3 is **descoped** — the shell-ODE overflow is owned by `shell-solver/OVERFLOW_FIX_PLAN.md` (on `main`, `bugfix/LSODA-shellODE`), and my first take there was itself wrong (corrected in place 2026-06-18).** No production code has been changed by this doc.
- **Type:** plan — a ranked, phased plan to remove hot-path waste and improve numerical conditioning, with the evidence that de-risks each item embedded inline.
- **Workstream:** `performance/` — cross-cutting hot-path cost & conditioning. Distinct from the *integrator-swap* workstreams (`shell-solver/`, `archive/bubble/`) and the *solver-repair* workstream (`archive/betadelta/`), though §F1 and §F5 touch the same files.
- **Where it sits:** entry point → the per-segment bubble/cooling solves → **this**. The two prior wins fixed *which solver* runs; this fixes *how much redundant work each solve does*.
- **Code it concerns:** `trinity/bubble_structure/bubble_luminosity.py` (the dMdt fsolve + 60k grid + the dead grav returns), `trinity/cooling/net_coolingcurve.py` (`get_dudt` inner loop), `trinity/_input/default.param` + `trinity/_functions/logging_setup.py` (the DEBUG default), `trinity/sps/update_feedback.py` (`pdotdot_total`), `trinity/shell_structure/{get_shellODE.py,shell_structure.py}` (the precision reframe).
- **Linked files & data:** none committed yet. **Phase P0 must commit** a `data/hotpath_baseline.csv` (per-phase wall-time + bubble-solve / resample call counts) and a `harness/replay_residual_grid.py` capture-replay harness (mirrors `shell-solver/harness/capture_replay_variants.py`) before any §F1(b) change ships. **Save/commit every new CSV.**

This audit was prompted by the question of where the *next* ground-breaking efficiency fix is, given hybr and the shell-solver migration. The recurring win-shape in both was **work done at the wrong granularity, or more work than the consumer needs.** This doc finds at least one more of that exact shape (bigger than either, §F1), a batch of free bit-identical cleanups (§F2), and one place where the prior docs *overstate* the problem (§F3).

Environment of record (matches the prior plans): **python 3.11.x, numpy 1.26.4 (`<2` pin), scipy 1.17.1.** Work branch: `claude/beautiful-johnson-4kw4w9`.

> **Verification log — line-by-line pass 2026-06-18.** Every `file.py:line`
> claim below was re-checked against current source. **Corrected:** `solve_R1`
> def is `get_bubbleParams.py:405` (was `:420`); the second `exp(-tau)` clamp is
> `get_shellODE.py:114-117` (was `113-117`); `dictionary.py` lives in
> `trinity/_input/` (not `_output/`) with the metadata first-flush write at
> `:821`; the F2.6 simplify-R² `:535` debug is **already guarded** (≤2/snapshot)
> so it was demoted from a "waste" item to a checked-and-clean note (the real
> F2.6 win is the `_excluded_keys` rescan at `:614`); softened the exact 60k grid
> count to "≈59,992, varies with cleaning". **Confirmed correct as written:** all
> F1 refs (`bubble_luminosity.py` 461/875/894/900/157/908-923/87), all
> `net_coolingcurve.py` sites (46/89/94/96/103/127/150), `default.param:37,43` +
> `logging_setup.py:207,258`, `mass_profile.py:204,226`, `update_feedback.py:185`,
> the unit constants (`ndens_cgs2au=2.938e+55`, `Lambda_cgs2au=5.650e-86`, both
> evaluated), the per-segment framing (`run_energy_phase.py:159`,
> `run_energy_implicit_phase.py:720`, `get_betadelta.py:56/909/936`,
> `energy_phase_ODEs.py:223`), `density_profile.py:109-130`, and the
> transition/momentum "zero bubble solves" (grep `get_bubbleproperties_pure|_create_legacy_radius_grid`
> → 0 in both). **Not reproducible here:** this container has **no astropy**, so
> the code does not import/run — the F1 "~21 ms vs 0.8 ms" timings remain
> subagent-microbenchmark-sourced and are still gated on P0 re-measurement; the
> source-structure claims above are all confirmed statically.

---

## The recurring bug/bottleneck classes here

1. **Computing far more than the consumer needs** — §F1 (60k-point resample for 4 scalars), §F2.2 (computed-then-discarded gravity).
2. **Per-call work that is actually run-constant** — §F2.1 (DEBUG logging), §F2.3 (cooling cutoffs), §F5 (profile constants).
3. **Hand-rolled vs. library / right-sized solver** — the hybr lesson; a smaller version survives in §F5 (the `_solve_grid` 25-point grid; hybr default-flip still pending).
4. **Unit-scaling *conditioning*** — §F3 (real, but **not** the float64 *overflow* the docs claim).

---

## Critical framing correction (verified 2026-06-18)

`get_bubbleproperties_pure` is called **per segment**, NOT per ODE-RHS stage. The
time-stepping RHSs (`get_ODE_Edot_pure`, `…implicit_pure`, `…transition_pure`,
momentum) are cheap and consume pre-computed scalars. The ~60k-point bubble
structure solve runs:
- **phase1_energy** (`run_energy_phase.py:159`): **once per segment**.
- **phase1b_energy_implicit** (`run_energy_implicit_phase.py:720` → `solve_betadelta_pure`):
  **many times per segment** — up to 25 via `_solve_grid`, fewer under `hybr`.
- **phase1c_transition / phase2_momentum**: **zero** bubble structure solves
  (verified — no `get_bubbleproperties_pure`, no `_create_legacy_radius_grid`).

So the entire 60k cost lives in **phase1_energy and phase1b only**, and §F1 below
is amplified by the betadelta grid/root-finder, not by the time integrator.

The ODE RHS *does* run a cheaper `solve_R1` brentq every stage (`energy_phase_ODEs.py:223`
→ `get_bubbleParams.py:405`) — that is §F5's warm-start candidate, a different
(and smaller) cost than the per-segment 60k solve.

---

## F1 — [HEADLINE, needs harness] Stop resampling 60 000 points to read 4 numbers

**The single biggest lever; same shape as hybr/shell.**

### Mechanism (verified against source)
`get_bubbleproperties_pure` solves `dMdt` with `scipy.optimize.fsolve`
(`bubble_luminosity.py:461`). Each fsolve iteration calls `_get_velocity_residuals`
(`:875`), which:
1. rebuilds the ~60 000-point legacy grid (`_create_legacy_radius_grid`, `:894`;
   three `np.logspace(… int(2e4))` chunks ≈ 6e4 before `_clean_radius_grid`; a
   captured call measured ≈59,992 — exact count varies with cleaning),
2. runs a full `solve_ivp(LSODA, dense_output=True)` (`_solve_bubble_structure`, `:900`),
3. **resamples all ~60 000 points** via `sol.sol(r_array)` (`:157`),

…then the residual it returns (`:908-923`) consumes exactly **four reductions**:
`v_array[-1]`, `v_array[0]`, `np.min(T_array)`, `operations.monotonic(T_array)`.

### Cost (measured — microbenchmark, trivial RHS; re-measure on the real RHS in P0)
| step | cost | note |
|---|---|---|
| `solve_ivp` integration | ~0.8 ms | ~21 adaptive nodes |
| `sol.sol(60k)` resample | **~21.4 ms** | **~27× the integration; pure overhead** |
| `sol.sol(600)` resample | ~0.49 ms | ~44× cheaper than the 60k resample |

> ⚠️ **Caveat (no overclaiming):** the 21 ms is a *trivial-RHS* microbenchmark.
> The real RHS (cooling-curve interpolation per node) is heavier per node, which
> **shrinks the ratio** — but the resample cost is point-count-bound and
> unchanged. P0 must re-measure the absolute resample fraction on a real run.

**Frequency:** once per fsolve iteration (xtol=1e-4, factor=50 → ~3–8 iters) ×
up to 25 `_solve_grid` points × every phase1b segment → easily **10²–10³** of
these resamples per phase1b run.

### Fix — two parts, two risk levels
- **(a) Endpoints from boundary nodes — BIT-IDENTICAL, ship immediately.**
  `v[0]`/`v[-1]` at the integration span ends equal `sol.y[:,0]`/`sol.y[:,-1]`
  (dense output at the exact `t_span` endpoints *is* the integrator's boundary
  value). No resample needed for the endpoint ratio `(v[-1])/(v[0]+1e-4)`.
- **(b) `min_T` / monotonicity from a coarse fixed sample (~500–2000 pts), not
  60k — NEEDS AN EQUIVALENCE TEST.** This is the *residual-only* solve (looser
  `_RESIDUAL_RTOL = 1e-6`) whose sole job is to *locate* `dMdt`; a coarse sample
  is almost certainly within tolerance. **But it changes which `dMdt` the fsolve
  converges to**, so it is *not* a bit-identical claim — it earns a capture-replay
  harness comparing final `bubble_dMdt`, `bubble_LTotal`, `bubble_T_r_Tb` across a
  representative sweep (exactly like `shell-solver/`). The ~21-node adaptive
  solution alone may miss a sub-node `min_T` dip, so a coarse *fixed* sample
  (not the adaptive nodes) is the safe middle.

**Plausible payoff:** ~1 order of magnitude off the phase1b inner loop;
**multiplies** with the shipped hybr win (hybr ≈ 10 bubble solves/segment vs the
grid's 25).

### Cousin experiment (own convergence study, à la `archive/bubble/conduction-convergence.md`)
**Does the legacy grid need 60 000 points at all?** The conduction zone is already
sampled separately at 2000 pts (`_CONDUCTION_NPTS`); the CIE region is smooth.
Shrinking the base grid speeds the *final* solve too, not just the residual.
Gate it on a Tavg / L_total / mBubble convergence sweep.

---

## F2 — Free, bit-identical wins (ship in one surgical commit; pytest must be unchanged)

| # | Win | Where | What's wasted | Risk |
|---|---|---|---|---|
| **F2.1** | **DEBUG is the shipped default with file output.** `log_level DEBUG` + `log_file True`; **none** of the 7 example params (incl. `simple_cluster`) override it. Both root logger and `FileHandler` are set to the param level (`logging_setup.py:207,258`), so every hot `logger.debug` formats its f-string **and writes to `trinity.log`**. | `default.param:37,43`; hot sites `energy_phase_ODEs.py:282-283` (2 f-strings/RHS call), `mass_profile.py:204,226` (with `.min()/.max()`) | Per-RHS-stage string-format + disk I/O for an entire run. | Ship `log_level INFO` default and/or guard hot sites with `if logger.isEnabledFor(logging.DEBUG)`. Bit-identical numerics. |
| **F2.2** | **`grav_phi` + `grav_force_m` computed then discarded.** | `bubble_luminosity.py:979-1000`; sole caller `:746` `m_cumulative, _, _ = …` (grep-confirmed no other caller) | A full-array `scipy.integrate.simpson` + a 60k-element divide, every final structure solve, for nothing. | Delete the two computations; trim return to `m_cumulative`. Bit-identical (`m_cumulative` unchanged). |
| **F2.3** | **`get_dudt` recomputes run-constants every call** (innermost scalar loop). | `net_coolingcurve.py:94,96,103` | `nonCIE_Tcutoff`/`CIE_Tcutoff`/`min(temp)` are boolean-mask+`max`/`min` over fixed grids on every call; `CIE_Tcutoff`/`min(temp)` are true run-constants, the non-CIE one changes only at the cube-rebuild cadence. | Precompute once (cache on the cube / in `params`, refresh only when `get_coolingStructure` runs). Bit-identical. |
| **F2.4** | **`get_dudt` computes `Lambda_CIE` unconditionally.** | `net_coolingcurve.py:89` | A full `CIE.get_Lambda` (interp1d eval + 2 transcendentals) discarded on the non-CIE and interpolation branches (the common conduction-zone case). | Move it inside the `elif … >= CIE_Tcutoff` branch. Bit-identical. |
| **F2.5** | **SPS `pdotdot_total` finite-difference in the energy RHS path.** | `update_feedback.py:185` (2 extra of 13 spline evals/call) | Never consumed by the energy/transition RHS (`[rd,vd,Ed]`); it's a diagnostic/momentum derivative. | Compute only where consumed (`compute_derived_quantities`). Bit-identical to the integrated trajectory. |
| **F2.6** | **Per-snapshot full-dict rescan of `_excluded_keys`.** | `trinity/_input/dictionary.py:614` (`for k, item in self.items(): … _excluded_keys.add(k)`, comment "Refresh excluded sets in case flags changed after insertion") runs every `_clean_for_snapshot` (every snapshot). | A full ~195-key walk per snapshot to refresh a set that is constant after phase 0. | Compute once / on flag change. Bit-identical. *(Verified NOT a hot waste: the simplify-R² `logging.debug` at `:535` is already guarded by `_impl_r2_logged < 2` + phase=="implicit", so ≤2 emits/snapshot — noted only to record it was checked.)* |

---

## F3 — Shell-ODE conditioning — **OWNED BY `shell-solver/OVERFLOW_FIX_PLAN.md`** (descoped 2026-06-18)

> **Update 2026-06-18 — descoped, and my earlier framing here was itself wrong.**
> The `bugfix/LSODA-shellODE` branch (now on `main`, PRs #691/#692) owns this via
> `docs/dev/shell-solver/OVERFLOW_FIX_PLAN.md` + `harness/verify_overflow.py`. Its
> captured real solve **falsifies my first take.** Do **not** duplicate it — track
> it there.

**What I got wrong.** An earlier draft of this section claimed the shell sites
"do **not** overflow — only precision loss (~1e110–1e130 ≪ 1.8e308)." That was a
*static magnitude* argument that ignored the **ODE dynamics**: `dndr` carries a
`+chi_e·nShell²·…` recombination term with a ~`1/k_B` (~1e55) prefactor
(`get_shellODE.py:97`), making `dn/dr ∝ +n²` a **finite-radius pole**
(`n(r)=n0/(1−A·n0·Δr)`). So `n` does **not** stay at ~1e65 — it blows past
~1.3e154 a few steps past the front and `nShell²` genuinely **overflows float64 →
inf/nan**. Their `verify_overflow.py` capture (real `simple_cluster` first ionised
solve) pins it: `y0_n=1.337e65`, `n_front=2.974e65` (=1.011e10 cm⁻³, finite,
`nShell²≈8.8e130`), `front_idx=4`, **`ovf_idx=26` (first non-finite row)**,
`n_max_finite=6.65e67`. The truth is the *synthesis* both prior docs missed:
**finite (precision-only) at the front, real `inf` overflow at the pole — but
entirely in the discarded post-front tail** (`shell_structure.py` truncates at the
front ~idx 4; the overflow is ~idx 26). MIGRATION_PLAN's "overflows float64" was
right *that* it overflows; my "doesn't overflow" was right only about the *front*.

**Action — none for this workstream on the shell.** Their plan's **#1 (CGS-rescale
the whole shell RHS — verified exact 1e-12 identity)** is the root-cause fix and
supersedes my "cgs-first squaring" note for `get_shellODE.py:97,100` and
`shell_structure.py:144,282`. It is PLANNED there (not yet implemented). We only
(a) drop the shell sites from this doc, and (b) flag that their plan must **revert
the `mxstep` change** that `shell-solver/MIGRATION_PLAN.md` and this doc's earlier
context credited as the warning fix — their §3 shows `mxstep` silenced a
*different* warning (`ODEintWarning`), not the `t+h=t` overflow flood.

**Still in THIS doc's scope (NOT covered by the shell plan), low value:** the
**bubble** cooling integrands `bubble_luminosity.py:612,699` square a code-unit
`n_bubble` (~1e53–1e55) → `n²`~1e106–1e114. There is **no `+n²` ODE term there**
(one-shot trapezoid integrands), so **no pole and no overflow — genuinely
precision-only**, and the consumed result is unaffected. Optional consistency fix
(cgs-first, matching `get_dudt`'s `net_coolingcurve.py:46`); not worth a dedicated
change. `net_coolingcurve.py:127,150` already does it right; the `exp(-tau)` clamp
`get_shellODE.py:83-86,114-117` is properly guarded.

---

## F4 — Verified clean (non-findings — so the next visit doesn't re-dig)

- **File I/O is batched, not per-step:** one flush per 10 snapshots
  (`snapshot_interval=10`, `trinity/_input/dictionary.py:220,750`), single
  `metadata.json` write on the first flush (`:821`). No per-step open/close.
- **SPS/SB99 interpolators are built once** at `main.py:147` and cached; never
  rebuilt in a loop.
- **The cooling `.interp` calls in the luminosity integrals** (`bubble_luminosity.py:650-694`)
  are already **vectorized** `RegularGridInterpolator` calls over the whole array
  — not Python loops.
- **`_solve_grid` already caches the winning point's `props`** (`get_betadelta.py:~1017`)
  and early-exits — no double-solve there.
- **The non-CIE cooling cube** rebuild is correctly periodic (5e-2 Myr energy /
  5e-3 Myr implicit), not per-call; its only waste is reconstructing interpolators
  it could keep (secondary).

---

## F5 — Bigger levers (need an experiment / drift budget — NOT free)

- **Warm-start `R1` (brentq, every RHS stage, `get_bubbleParams.py:405`) and
  `dMdt` across segments.** Both evolve smoothly; the implicit phase already
  threads `dMdt` warm-starts. **Not bit-identical** (changes convergence path) →
  needs a drift budget like HYBR_PLAN Phase 1.
- **Hoist profile run-constants into the `ODESnapshot`.** `density_profile`/`mass_profile`
  re-extract ~7 constants and rebuild a length-1 `tanh` bridge per RHS stage
  (`density_profile.py:109-130`) to evaluate a closed-form polynomial.
- **Flip the hybr default** (HYBR_PLAN Phase 4, still `legacy`): shrinks `_solve_grid`'s
  25 bubble solves/segment to ~10 — the multiplier that makes §F1 hurt.

---

## F6 — Correctness/bug-class items (already catalogued — pointers, not new work)

Tracked in `CODEBASE_REVIEW.md` / `codebase_review/02_trinity_physics.md`; not hot-path,
listed so this doc is self-contained: the `ZCloud ≠ 1.0/0.15` `UnboundLocalError`
(`read_cloudy.py:290`, should validate at param-load), CIE cooling silently
ignoring metallicity (`read_coolingcurve.py:20`), and the units **label** drift.

---

## Phases

### P0 — Baseline + harness (no production change) — **DO FIRST**
Commit `data/hotpath_baseline.csv`: a real `simple_cluster` (degenerate) + one
realistic config (e.g. `archive/betadelta/diagnostics/probe_typical_hybr.param`)
run, logging per-phase wall time and **call counts** for `get_bubbleproperties_pure`,
the dMdt fsolve iterations, and `sol.sol()` point counts — to turn §F1's
microbenchmark ratio into a real production fraction. Build
`harness/replay_residual_grid.py` (capture real in-run residual-solve inputs,
replay endpoint-from-nodes + coarse-sample variants vs the 60k baseline).
**Gate G0:** baseline + harness committed; the §F1 resample fraction is measured,
not assumed.

### P1 — Free wins (§F2) — bit-identical
One surgical commit: F2.1–F2.6. **Gate G1:** `pytest` (and `-m stress`) byte-for-byte
unchanged vs `main`; a `simple_cluster` snapshot hash is identical. Ship.

### P2 — F1(a) endpoint-from-nodes — bit-identical
Read `v[0]`/`v[-1]` from `sol.y` boundary nodes; stop resampling 60k for the
endpoint ratio. **Gate G2:** snapshot hash unchanged (the endpoint values are the
same floats). Ship.

### P3 — F1(b) coarse `min_T`/monotonicity sample — behind the harness
Replace the 60k `min_T`/monotonicity sample with a coarse fixed grid. **Gate G3
(hard):** harness shows final `bubble_dMdt`, `bubble_LTotal`, `bubble_T_r_Tb`
within tolerance (target ≤0.3%, matching the existing `_RESIDUAL_RTOL` dMdt drift
budget) across the sweep, 0 solver failures. Flip only if the gate passes; keep
the 60k path selectable one release.

### P4 — Cousins / bigger levers (§F1-cousin, §F5)
Grid-size convergence study; warm-start drift budgets. Each its own results note +
committed data. Deferred until P1–P3 land.

---

## Decisions that belong to the maintainer
1. **Ship `log_level INFO` as the default** (§F2.1)? It changes the out-of-the-box
   `trinity.log` verbosity a user sees — a UX call, not just perf.
2. **Acceptable `dMdt` drift for §F1(b)** — is the ≤0.3% the existing residual
   solve already tolerates the right bar, or stricter for published tracks?
3. **Grid-size study scope** (§F1-cousin) — worth a convergence sweep now, or
   after the resample fix already recovers most of the cost?

## Out of scope
- The shell `odeint`→`solve_ivp` migration (`shell-solver/MIGRATION_PLAN.md`) — separate.
- The betadelta solver internals (`archive/betadelta/`) beyond the default flip.
- Any change to the bubble/shell/cooling **physics** — this is waste-removal and
  conditioning only.

## Risks
| risk | mitigation |
|---|---|
| §F1(b) coarse sample misses a real `min_T` dip → wrong `dMdt` | fixed coarse grid (not adaptive nodes); P3 hard gate + capture-replay harness |
| §F2.1 hides a warning users rely on | keep WARNING+ at file level; only DEBUG/INFO chatter is suppressed |
| §F2 "bit-identical" claim wrong somewhere | P1 gate is a snapshot-hash + full pytest diff, not a spot check |
| Microbenchmark ratio overstated on the real RHS | P0 re-measures on a production run before any §F1 work |
| Precision reframe (§F3) dismissed as "just docs" | it is — but the *fix* (cgs-first squaring) still belongs to the shell workstream; this doc only corrects the wording |
