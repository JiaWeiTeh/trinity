# Plan: shell-structure solver — `odeint` → `solve_ivp` migration

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
> saved as a committed artifact (a CSV/table under `docs/dev/shell-solver/data/`,
> or a force-added harness/figure in this folder) — never left in `/tmp` or an
> untracked `outputs/`. A future visit must be able to reproduce or compare
> against the numbers **without re-running**; record the exact config + command
> that produced each artifact.

**About this document**  (last updated 2026-06-17 — the 🔄 banner *requires* refreshing this on every visit; it is a living doc, not frozen.)
- **Status (verified 2026-06-17):** 🟠 **ACTIONABLE — but the motivation flipped.** Equivalence is settled: `solve_ivp(LSODA, t_eval)` reproduces `odeint` to ~1e-9–1e-8 across **6 configs / 4 regimes** (incl. the neutral region). **New, decisive finding from cross-regime timing: the migration is NOT a speedup** — `solve_ivp` is **slower** than `odeint` in every realistic regime (~4× for the drop-in), and the warning wall is **specific to the degenerate `simple_cluster` regime** (a code-unit overflow), not science runs. So this is a **robustness/cleanliness** change, not a performance one. Code change still not written. See §P0-results for the data and the reshaped recommendation.
- **Type:** plan — phased migration of the shell-structure ODE integrator from `scipy.integrate.odeint` to `scipy.integrate.solve_ivp`, with the cross-regime equivalence + timing evidence that de-risks it embedded inline (P0).
- **Workstream:** `shell-solver/` — the shell-structure micro-solver (`trinity/shell_structure/`), distinct from the already-migrated **bubble**-structure solver and from the betadelta/transition solver work.
- **Where it sits:** entry point → **this** → (companion results/design docs to be spun out per §Phases if the work proceeds).
- **Code it concerns:** `trinity/shell_structure/shell_structure.py:156` & `:315` (the two `odeint` calls); `trinity/shell_structure/get_shellODE.py` (the RHS); the bubble precedent `trinity/bubble_structure/bubble_luminosity.py:106-166`.
- **Linked files & data:** harnesses `docs/dev/shell-solver/harness/{capture_replay.py, capture_replay_variants.py, diagnose_first_call.py, phase_probe.py}`; data `docs/dev/shell-solver/data/replay_comparison.csv` (30-row equivalence) + `replay_variants_{sfe0.3,sfe0.6,steep,dense_flat,mock_hybr,probe_typical_hybr}.csv` (timing+event, one per config). **Save/commit every new CSV — future sessions regenerate plots from these without re-running.**

This plan was prompted by the **LSODA "t + h = t" / "Excess work done" warning wall** printed on a plain `python run.py param/simple_cluster.param`. The warnings are non-fatal (the run completes with all sanity checks passing), but they are noisy and they originate from the shell-structure `odeint` calls. The bubble-structure solver was already migrated `odeint → solve_ivp(LSODA, dense_output=True)` (CHANGELOG, PRs #666/#678); this plan asks whether the **same** move is correct for the shell solver, and pins down the exact call that works.
Environment of record: **python 3.11.15, numpy 1.26.4, scipy 1.17.1.** Work branch: `claude/confident-knuth-pf0wsj`.

---

## The question

1. Does `solve_ivp` reproduce the shell `odeint` trajectory on the physically-used grid, within tolerance, on every captured call, **in every regime**? **→ YES (1.4e-9 … 1.05e-8 across 6 configs / 4 regimes, incl. 7 neutral-region solves).**
2. Does it silence the "Excess work" warning? **→ YES — but the warning is itself regime-specific: 40/40 calls in degenerate `simple_cluster`, only 0–4/40 in realistic low-sfe configs. There is little wall to silence in science runs.**
3. What is the exact, safe call — and what breaks if you copy the bubble precedent verbatim? **→ `solve_ivp(method='LSODA', t_eval=rShell_arr)` WITHOUT `dense_output`. The bubble precedent's `dense_output=True` is a hard blocker here.**
4. **Is it faster?** **→ NO.** `solve_ivp` is **slower than `odeint` in every realistic regime** (drop-in ~4× slower; the front-event trick is ~5× *faster* only in degenerate `simple_cluster`, and *slower* everywhere else). The migration buys robustness/cleanliness, not speed.

---

## Mechanism / current state (verified 2026-06-17 against source — re-verify per banner)

### Where `odeint` is called, and what it integrates
- **Ionized region** — `shell_structure.py:156`, state `y=(n, φ, τ)`, over the uniform grid `rShell_arr = np.arange(rShell_start, rShell_start+sliceSize, rShell_step)` with `rShell_step = sliceSize/1e3` (`:138-140`). Loop `:148-209`.
- **Neutral region** — `shell_structure.py:315`, state `y=(n, τ)`, `rShell_step = sliceSize/5e3` (`:303-305`). Loop `:307-368`. Entered only when `has_neutral = is_phiDepleted and not is_allMassSwept` (`:212`).
- RHS: `get_shellODE.py:78-105` (ionized), `:109-127` (neutral). Neither call passes `rtol/atol/mxstep` → **scipy LSODA defaults** (`odeint` default `rtol=atol≈1.49012e-8`).

### Why it stiffens, and why the warning wall appears
The RHS carries an `n²·α_B` recombination term in both `dndr` and `dphidr`. Near the ionization front φ collapses over a very thin layer and `n` rises steeply — a stiff boundary layer. **Critically, in `simple_cluster` the inner-shell density `y0[0]` is `~1e61–1e65` in code units** (physically only ~1e6–1e9 cm⁻³ — the *physics* is fine, but `nShell² ~ 1e122–1e130` overflows float64 partway across the fixed 1000-point slice). So the raw slice goes non-finite past roughly **row ~26** for *both* solvers. That overflow in the integrand is exactly what makes `odeint` (LSODA) burn through its step budget and print **"Excess work done on this call"** plus the **"t + h = t"** step-collapse chatter.

### Why that wall is (mostly) cosmetic today
`shell_structure.py` **truncates every slice at `idx`** = first row where swept mass ≥ `mShell_end` **or** φ ≤ 1e-9 (`:172-183`, `:328-336`), and discards everything after (`mShell_arr_cum[idx+1:]=0`, `[:idx]` accumulation). Captured truncation indices are **1–57 out of 1000** — the blow-up tail is never consumed. So the warnings flag work the integrator does in a region the code throws away. The run completes; the atexit snapshot passes all sanity checks (`R1<R2`, `Eb>0`, `Pb>0`, `shell_mass>0`), no NaN/inf.

### The load-bearing invariant for any replacement
Downstream math **hardcodes the uniform grid spacing `rShell_step`**: per-cell mass `n·μ·4π r²·rShell_step` and `np.cumsum` (`:167-169, 255-256, 324-326, 359-360`); the termination-index logic indexes the uniform grid (`:172-183, 328-336`); gravity quadratures share that grid (`:257-259, 361-363`). **Therefore `solve_ivp` must return the solution sampled exactly on `rShell_arr`** (via `t_eval=rShell_arr`), not on the solver's own adaptive steps. The RHS arg order also flips: `odeint` is `func(y, t)`, `solve_ivp` is `fun(t, y)`.

---

## P0 — Empirical equivalence evidence (DONE, 2026-06-17)

Two independent in-process **capture-and-replay** harnesses monkeypatch `scipy.integrate.odeint`, drive a real `param/simple_cluster.param` run (genuine in-run `y0`/grid/`params`, nothing synthetic, nothing pickled), and replay each of the first 30–40 shell solves through candidate integrators, comparing on the **physically-used prefix** (`[0:idx]`, finite-masked). All captures landed in the **ionized** region (the run aborts before any neutral solve — see gap below).

### Config matrix (worst case across 30–40 calls)

| id | exact call | result | worst `rel_n` (physical prefix) | LSODA chatter | verdict |
|---|---|---|---|---|---|
| **odeint** (current) | `odeint(func, y0, rShell_arr, args=...)` | baseline | — | **"Excess work" on 40/40** | the thing we're replacing |
| **`solve_ivp` LSODA + `t_eval`, no `dense_output`** | `solve_ivp(fun,(r0,r1),y0,method='LSODA',t_eval=rShell_arr,rtol=atol=1.49012e-8)` | **40/40 ok** | **1.42e-9** | **0** | ✅ **RECOMMENDED** |
| `solve_ivp` LSODA + `dense_output` only, then `sol.sol(rShell_arr).T` | `solve_ivp(...,method='LSODA',dense_output=True)` | 40/40 ok | 1.42e-9 | 0 | ✅ works (extra interpolant, no benefit here) |
| `solve_ivp` LSODA + **`t_eval` AND `dense_output`** (= naive bubble-copy) | `solve_ivp(...,t_eval=rShell_arr,dense_output=True)` | **0/40 — fails** | — | — | ⛔ **`ValueError: 'ts' must be strictly increasing or decreasing`** |
| `solve_ivp` Radau + `t_eval` | `solve_ivp(...,method='Radau',t_eval=rShell_arr)` | **0/40 — fails** | — | — | ⛔ `"Required step size is less than spacing between numbers"` (returns ~26 pts) |
| `solve_ivp` BDF + `t_eval` | `solve_ivp(...,method='BDF',t_eval=rShell_arr)` | **0/40 — fails** | — | — | ⛔ same as Radau |
| `odeint(..., mxstep=50000)` (Option B) | unchanged solver, bigger step budget | 40/40 ok | **0.00e+00** | **0** | ✅ silences wall, **zero** change to results |

### Findings
1. **`solve_ivp(LSODA, t_eval=rShell_arr)` is a faithful drop-in.** Worst per-variable agreement vs `odeint` over the consumed rows: `rel_n=1.42e-9`, `rel_phi=1.02e-8`, `tau` abs `5.9e-10` (the headline `rel_tau` is a denominator artifact at τ's zero-crossing); endpoint `rel_n=5.4e-11`, `rel_phi=2.6e-9`. **0** LSODA chatter lines, **0** python warnings, `status=0` on all calls.
2. **`dense_output=True` is the trap.** With the micro-scale grid (step ~1.3e-8 pc) LSODA's internal breakpoints collide below float spacing, so scipy's global `OdeSolution` rejects them. The *integration* succeeds; only the dense-output **object** crashes. The bubble solver gets away with `dense_output=True` because its grid never reaches this micro-scale in the same way. **The shell migration must NOT copy that kwarg.** Since the shell downstream only reads values *at grid points*, `t_eval` alone is both sufficient and safe.
3. **Radau/BDF are not drop-ins** — they stall in the stiff layer. LSODA's stiff/non-stiff switching is what makes it work; keep it.
4. **`odeint(mxstep=50000)` gives a bit-identical result (0.00e+00)** — proof the wall is pure discarded-tail noise, not a result-affecting convergence failure. **But it is NOT free in time** (see §P0-results #5: ~6× slower in the degenerate regime because it stops bailing and grinds the tail). The cheap noise fix is to *silence* the `ODEintWarning`, not raise `mxstep`.

## P0-results — Cross-regime timing + event sweep (DONE, 2026-06-17)

The variant harness was extended with **per-call wall time** (min of 5 reps), a **terminal-event-at-the-front** variant (`solve_ivp` LSODA, `events=φ−1e-9`, stops at the ionization front so it never integrates the discarded tail), and an arbitrary-param override, then swept across **6 configs spanning 4 regimes** (40 captured shell solves each, first timesteps). One committed CSV per config. **All numbers below are measured, not assumed.**

| config (`.param`) | profile / sfe | regime | odeint ms/call | excess-work warns | mass-limited slices | neutral solves | **event** speedup | drop-in (`t_eval`) speedup | worst `rel_n` |
|---|---|---|---|---|---|---|---|---|---|
| **`simple_cluster` sfe0.3 ← CURRENT/DEFAULT** | flat / 0.3 | **degenerate (overflow)** | 7.19 | **40/40** | 0/40 | 0 | **4.98× faster** | 0.09× (11× slower) | 1.42e-9 |
| `simple_cluster` sfe0.6 | flat / 0.6 | degenerate | 7.06 | **40/40** | 0/40 | 0 | **4.97× faster** | 0.10× | 1.66e-9 |
| `probe_typical_hybr` | flat / 0.01, nCore 1e3 | realistic | 0.82 | 3/40 | 0/40 | 0 | 0.43× (slower) | 0.25× | 7.3e-9 |
| `steep` | **PL−2** / 0.01, nCore 1e5 | realistic, steep | 0.99 | 4/40 | **7/40** | **7** | 0.63× (slower) | 0.26× | 9.8e-9 |
| `dense_flat` | flat / 0.01, nCore 1e5 | realistic | 1.01 | 4/40 | **7/40** | **7** | 0.62× (slower) | 0.26× | 9.8e-9 |
| `mock_hybr` | flat / 0.0085, tiny 4e3 M☉ | realistic, tiny | 0.19 | 0/40 | **39/40** | 0 | 0.14× (slower) | 0.18× | 1.05e-8 |

**Current/default config** = the first row, `simple_cluster` at sfe 0.3 — exactly what `python run.py param/simple_cluster.param` runs (the config that prints the warning wall). The other rows are coverage probes from the hybr/transition families.

> ⚠️ **SAMPLING SCOPE — verified, partially widened.** The "/40" is the **first 40 shell-structure ODE solves**; `phase_probe.py` confirms those are **all in the ENERGY phase (phase 1), t ≈ 3e-7 … 1.2e-3 Myr**. The harness now also supports a `FROM_PHASE` gate (pass through energy, capture once a target phase is reached), and an **implicit-phase capture has been run** (below). **Transition and momentum are still NOT sampled** — `phase_probe.py` shows `simple_cluster` is still in the energy phase at solve #94 / wall 124s (and `mock_hybr` at #98 / 104s): no config dodges the long energy phase, so reaching transition costs many minutes per run (the remaining P-shadow job is a full run logging every phase).

### Implicit-phase spot check (`FROM_PHASE=implicit`, simple_cluster, 20 solves)
`data/replay_variants_sfe0.3_implicit.csv`. The implicit phase tracks the energy-phase pattern (simple_cluster stays in the degenerate overflow regime), and this run finally caught **2 neutral-region solves**:

| variant | ok | speedup | worst `rel_n` |
|---|---|---|---|
| `solve_ivp` LSODA + φ-event | 20/20 | **5.69× faster** | 4.76e-10 |
| `solve_ivp` LSODA + `t_eval` | 20/20 | 0.09× (11× slower) | 4.76e-10 |
| `solve_ivp` LSODA + `dense_output` | 20/20 | 0.09× | 4.76e-10 |
| Radau / BDF + `t_eval` | **5/20** | — | 3.3e-8 / 1.3e-7 |
| `odeint(mxstep=50000)` | 20/20 | 0.16× (slower) | 0.00e+00 |

- 18 ionized (φ-limited, event fires) + **2 neutral** (`n_state=2`, no φ-event; `rel_n` ≈ 1e-12 — essentially exact). odeint median 7.34 ms for the ionized degenerate solves, ~0.65 ms for the neutral ones.
- Same verdict as energy: accuracy is excellent; the event trick is ~5.7× faster *only because* simple_cluster is degenerate; `t_eval`/`dense` are ~11× slower; Radau/BDF still mostly stall (5/20).

### Verified findings (each traces to a committed CSV)
1. **Accuracy is excellent in every regime.** `solve_ivp(LSODA, t_eval)` reproduces `odeint` on the consumed prefix to **1.42e-9 … 1.05e-8**, including the **7 neutral-region (2-state) solves** in `steep`/`dense_flat`. The neutral-region and non-degenerate-regime gaps from the first P0 pass are now **closed** and positive.
2. **The warning wall is regime-specific.** "Excess work" fires on **40/40** calls only in degenerate `simple_cluster`; it is 3–4/40 in `typical`/`steep`/`dense_flat` and **0/40** in `mock_hybr`. Realistic science configs barely warn.
3. **The migration does not speed anything up.** In every realistic regime, `odeint` is **0.19–1.01 ms/call** and *every* `solve_ivp` variant is **slower** (drop-in ~4×; the event variant 0.14–0.63×). `solve_ivp` is faster *only* in degenerate `simple_cluster`, and only via the event trick — because there `odeint` wastes 7 ms grinding the float64-overflow tail.
4. **The front-event restructure is regime-limited.** It needs slices to be **φ-limited**; but mass-limited slices are 7/40 (`steep`/`dense_flat`) and **39/40** (`mock_hybr`), where the φ-event never fires (a full restructure would also need cumulative mass carried as an ODE-state event). So "drop the 1k slice / idx truncation" helps the degenerate regime but is **not** a general win.
5. **`odeint(mxstep=…)` is the wrong noise fix.** Raising it is ~6× *slower* in the degenerate regime (grinds the discarded tail) for 0.00e+00 result change. To kill the noise, **silence the warning**, don't raise `mxstep`.

### Root-cause lead (highest value, must be tested not assumed)
The degenerate behaviour is driven by `simple_cluster`'s inner-shell density `y0[0] ≈ 1e61–1e65` **in code units** (physically ~1e6–1e9 cm⁻³). `nShell²` then overflows float64 → the overflow tail → the warning wall → the 7 ms/call. This is a **unit-scaling artifact, not physics.** If the shell ODE carried `n` in a non-overflowing scale, `odeint` would likely be fast and quiet even at sfe 0.3 — **no migration needed.** Units are a known bug class here (CLAUDE.md), so this is a *candidate to test*, not a conclusion: verify `y0[0]` magnitude line-by-line, rescale in a scratch harness, and measure warns/time before touching production.

### Reproduce (data is committed — no re-run needed to read conclusions)
```bash
cd /home/user/trinity
python docs/dev/shell-solver/harness/capture_replay.py                                  # -> data/replay_comparison.csv (30-row equivalence)
python docs/dev/shell-solver/harness/capture_replay_variants.py                          # -> data/replay_variants_sfe0.3.csv (default)
python docs/dev/shell-solver/harness/capture_replay_variants.py 0.6                      # sfe override -> ..._sfe0.6.csv
python docs/dev/shell-solver/harness/capture_replay_variants.py docs/dev/transition/harness/steep.param        # -> ..._steep.csv
python docs/dev/shell-solver/harness/capture_replay_variants.py docs/dev/transition/harness/dense_flat.param   # -> ..._dense_flat.csv
python docs/dev/shell-solver/harness/capture_replay_variants.py docs/dev/transition/harness/mock_hybr.param     # -> ..._mock_hybr.csv
python docs/dev/shell-solver/harness/capture_replay_variants.py docs/dev/archive/betadelta/diagnostics/probe_typical_hybr.param  # -> ..._probe_typical_hybr.csv
```

### Known harness caveat (no overclaiming)
The fd-level **Fortran LSODA "t + h = t" chatter counter read 0 across all configs** — I do not trust this as a measurement (the redirect likely misses it under the host run). The **verified** warning signal is the Python `ODEintWarning` ("Excess work") count in the table above; the raw Fortran-wall magnitude is *not* characterized here.

### Still open before any default-flip
- **Wall-time over a FULL run** (not just the first 40 solves) per regime, to quantify the production slowdown the migration would cost.
- **Integration-level** equivalence of the consumed scalars (`n_IF_Str`, `F_rad` inputs) — the ODE-level `rel_n` is necessary but not sufficient; P-shadow must compare at the `shell_structure` output level.
- The **unit-overflow root-cause** experiment above (could moot the migration).

---

## The production pattern to implement (adapted from the bubble precedent)

The bubble migration (`bubble_luminosity.py:106-166`) is the template, **minus `dense_output`**, **plus `t_eval`**:

```python
class ShellSolverError(Exception):
    """Catchable, deterministic shell-structure solve failure (never sys.exit)."""

_SHELL_RTOL = 1.49012e-8   # match odeint's former default exactly (P0 used this)
_SHELL_ATOL = 1.49012e-8

def _solve_shell_ode(func, y0, rShell_arr, args):
    # solve_ivp RAISES on non-finite y0 -> convert to the deterministic contract.
    if not np.all(np.isfinite(y0)):
        return np.full((len(rShell_arr), len(y0)), np.nan), False, {'message': 'non-finite y0'}
    fun = lambda r, y: np.asarray(func(y, r, *args))     # odeint (y,t) -> solve_ivp (t,y)
    sol = scipy.integrate.solve_ivp(
        fun, (float(rShell_arr[0]), float(rShell_arr[-1])), np.asarray(y0, float),
        method='LSODA', t_eval=np.asarray(rShell_arr, float),
        rtol=_SHELL_RTOL, atol=_SHELL_ATOL,
        # NO dense_output (crashes on the micro-grid); NO events; NO max_step.
    )
    psoln = sol.y.T   # shape (len(rShell_arr), n_state) — same layout odeint returned
    return psoln, bool(sol.success), {'message': sol.message, 'status': sol.status, 'nfev': sol.nfev}
```
Call sites `:156` / `:315` become `sol_ODE, ok, info = _solve_shell_ode(...)`. **Open design choice:** `odeint` silently best-efforts (always returns a full-length array); `solve_ivp` returns **short** if it fails mid-grid (`sol.y` only covers reached `t_eval` points). On `not ok`, mirror the bubble path — raise `ShellSolverError` (then let the existing phase-level handler penalize the step) — rather than silently truncating. This is a behavior change on failing steps and must be gated (see P-validate).

## Migration invariants (a replacement is correct **iff** it preserves all of these)
- **(a)** Output sampled exactly on `rShell_arr` → `t_eval=rShell_arr`. (Groups A/B downstream index by `rShell_step`.)
- **(b)** Region-specific step kept distinct: ion `sliceSize/1e3`, neu `sliceSize/5e3`; `sliceSize`/`max_shellRadius` definitions unchanged (`:135,139,304`).
- **(c)** State layout: ion `(n,φ,τ)`→cols 0,1,2; neu `(n,τ)`→cols 0,1. RHS arg-order flipped to `(t,y)`.
- **(d)** RHS clamps (`φ=max(0,φ)`, `τ>500→exp=0`) untouched (they live in `get_shellODE`).
- **(e)** Slice loop / truncation (`massCondition`, `phiCondition≤1e-9`, first-crossing `idx`, `[:idx]` + final-append, re-seed) unchanged.
- **(f)** I-front handoff (density jump `×μ_atom/μ_ion·T_ion/T_neu`, `tau0_neu=tau0_ion`) at `:298-300`.
- **(g)** Endpoints preserved to `rel_tol=1e-12` (`shell_r_arr[0]==R2`, `[-1]==rShell`) — enforced downstream at `_output/cloudy/snapshot_to_deck.py:155-164`.
- **(h)** Consumed scalars within tolerance: `n_IF_Str` (sole P_HII source, `run_energy_phase.py:188-193`), `shell_fAbsorbedWeightedTotal`/`shell_tauKappaRatio` (F_rad), `isDissolved`, `rShell`, `n_IF`, `R_IF`.

---

## Phases

### P0 — Equivalence + cross-regime timing harvest — ✅ DONE (this doc, §P0 and §P0-results)
Offline capture-and-replay; zero production change. **Gate G0 (met, and then some):** the right `solve_ivp` config reproduces `odeint` to ≤1e-8 on every consumed row across **6 configs / 4 regimes incl. the neutral region** — the two original coverage gaps are closed. **But G0 also surfaced a motivation problem:** the change is not a speedup and the wall is regime-specific (§P0-results). **Branch taken:** PROCEED to P-shadow *only if* the maintainer decides robustness alone justifies the production slowdown (Decision #1); otherwise pivot to the cheaper "silence the warning" and/or the unit-overflow root-cause (Decision #4).

### P-shadow — Equivalence shadow over a full run (zero production impact)
Extend the harness to run `solve_ivp(LSODA, t_eval)` **alongside** the live `odeint` for an **entire** `simple_cluster` run (and one non-degenerate param), logging per-call `max_rel_diff` to CSV; production still integrates with `odeint` ⇒ byte-identical snapshots. Must reach **neutral-region** solves.
**Gate G_shadow:** ion **and** neu agreement ≤1e-6 on consumed rows across the whole run, 0 solver failures, snapshot hash unchanged. Simplest passing config (`t_eval`, no `dense_output`) promotes.

### P-promote — Promotion behind a switch (default unchanged)
Add a `shell_integrator` param (default `"odeint"`, byte-identical to today; `"solve_ivp"` selectable). Implement `_solve_shell_ode` (§pattern) + `ShellSolverError`. **Add the missing integrated-output regression test** (none exists today — `test/test_mu_audit_drift.py:170-264` only pins the RHS and source-text). Mirror the bubble tests: a `test_shell_solver_failures.py` contract suite + a `@pytest.mark.stress` N-run gate.
**Gate G_promote:** new test green in both modes; `pytest` (default + `-m stress`) green.

### P-validate — Validation & default decision
`solve_ivp` vs `odeint` across all configs incl. one non-degenerate regime and a neutral-region run: every consumed scalar (`n_IF_Str`, `F_rad` inputs, `rShell`, `Eb`, `Pb`, terminal momentum) closes to ≤1% / traceable; cost within +20% wall time. **Hard null gate:** on a non-degenerate param the migrated solver must match `odeint` to solver tolerance with **zero** warnings.
**Decision:** flip default to `solve_ivp` only if every gate passes; else default stays `odeint` and the finding is documented regardless. Keep `odeint` selectable one release, then delete it.

---

## Decisions that belong to the maintainer
1. **Is the migration even worth it, given it's slower?** P0-results shows `solve_ivp` is *slower* than `odeint` in every realistic regime. The only motivations left are (a) robustness (deterministic failure contract like the bubble solver) and (b) silencing a wall that only the degenerate regime prints. Pick the lane: **(A)** full `solve_ivp` migration for robustness, accepting a production slowdown; **(B)** just *silence* the `ODEintWarning` (smallest change, 0 result/perf cost) and stop; or **(C)** chase the unit-overflow root cause (Decision #4) which could remove the wall *and* the 7 ms/call with no integrator change.
2. **Failure semantics:** on a failing shell solve, raise `ShellSolverError` (bubble-style, changes which steps "succeed") vs. preserve `odeint`'s silent best-effort? P-validate must measure the difference.
3. Acceptable wall-time ceiling vs `odeint` (the drop-in is ~4× slower in realistic regimes — is that tolerable for the shell micro-solve's share of total runtime? P-shadow must measure the full-run cost, not just per-call).
4. **Pursue the unit-overflow root cause?** `simple_cluster`'s `y0[0] ≈ 1e61` code units is the source of the overflow/wall/slowness. Rescaling `n` in the shell ODE could fix it at the source — but units are a known bug class. Worth a scoped, tested experiment before committing to any integrator swap?

## Out of scope
- The betadelta `hybr` solver and the implicit→momentum transition trigger (separate plans).
- Any change to the shell ODE **physics** (`get_shellODE.py`) — this is an integrator swap only.
- Lifting the numpy `<2` pin — that pin is the *bubble* monotonic guard, orthogonal to this solver.

## Risks
| risk | mitigation |
|---|---|
| Variable-step output silently breaks the uniform-`rShell_step` math | **always** `t_eval=rShell_arr`; covered by invariant (a) + the integrated-output regression test (P-promote) |
| Copying the bubble `dense_output=True` verbatim | **proven hard-crash** here — pattern in §pattern explicitly omits it; documented in P0 row 4 |
| RHS arg-order flip (`(y,t)`→`(t,y)`) silently wrong | unit test the RHS adapter; P0 harness already validates the closure |
| `solve_ivp` short-returns on failure where `odeint` didn't | explicit `ok`/length check + `ShellSolverError`; decision #2 + P-validate gate |
| ~~Neutral region / non-degenerate regime untested~~ | **CLOSED 2026-06-17** — §P0-results covers 4 regimes incl. 7 neutral solves; accuracy holds (≤1.05e-8) |
| **Migration silently slows production** (~4× per shell solve in realistic regimes) | measure full-run cost in P-shadow; Decision #1/#3 — robustness must justify it, else pivot to "silence the warning" |
| Front-event restructure assumed φ-limited | **measured** — mass-limited slices are 7/40 (steep) to 39/40 (mock); a φ-only event is insufficient, would need a cumulative-mass event state |
| No integrated-output test exists today | add it in P-promote *before* touching the solver |
