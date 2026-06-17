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

**About this document**
- **Status (verified 2026-06-17):** 🔵 **ACTIONABLE** — P0 feasibility is **done and positive** (the right `solve_ivp` config reproduces `odeint` to ~1e-9 and silences the LSODA warning wall); the code change itself is **not yet written**. Two coverage gaps remain before promotion (neutral region, non-degenerate regime).
- **Type:** plan — phased migration of the shell-structure ODE integrator from `scipy.integrate.odeint` to `scipy.integrate.solve_ivp`, with the equivalence evidence that de-risks it embedded inline (P0).
- **Workstream:** `shell-solver/` — the shell-structure micro-solver (`trinity/shell_structure/`), distinct from the already-migrated **bubble**-structure solver and from the betadelta/transition solver work.
- **Where it sits:** entry point → **this** → (companion results/design docs to be spun out per §Phases if the work proceeds).
- **Code it concerns:** `trinity/shell_structure/shell_structure.py:156` & `:315` (the two `odeint` calls); `trinity/shell_structure/get_shellODE.py` (the RHS); the bubble precedent `trinity/bubble_structure/bubble_luminosity.py:106-166`.
- **Linked files & data:** harnesses `docs/dev/shell-solver/harness/{capture_replay.py, capture_replay_variants.py, diagnose_first_call.py}`; data `docs/dev/shell-solver/data/{replay_comparison.csv, replay_variants.csv}`.

This plan was prompted by the **LSODA "t + h = t" / "Excess work done" warning wall** printed on a plain `python run.py param/simple_cluster.param`. The warnings are non-fatal (the run completes with all sanity checks passing), but they are noisy and they originate from the shell-structure `odeint` calls. The bubble-structure solver was already migrated `odeint → solve_ivp(LSODA, dense_output=True)` (CHANGELOG, PRs #666/#678); this plan asks whether the **same** move is correct for the shell solver, and pins down the exact call that works.
Environment of record: **python 3.11.15, numpy 1.26.4, scipy 1.17.1.** Work branch: `claude/confident-knuth-pf0wsj`.

---

## The question

1. Does `solve_ivp` reproduce the shell `odeint` trajectory on the physically-used grid, within tolerance, on every captured call? **→ YES (≈1e-9), with the right config.**
2. Does it silence the "Excess work" / "t + h = t" warning wall? **→ YES (0 warnings vs 40/40 for odeint).**
3. What is the exact, safe call — and what breaks if you copy the bubble precedent verbatim? **→ `solve_ivp(method='LSODA', t_eval=rShell_arr)` WITHOUT `dense_output`. The bubble precedent's `dense_output=True` is a hard blocker here.**

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
4. **Option B is genuinely free.** `odeint(mxstep=50000)` reproduces the physically-used prefix to **0.00e+00** (bit-identical) while removing the "Excess work" warning — proof the wall was pure tail noise, not a result-affecting convergence failure.

### Reproduce (no re-run needed to read conclusions — data is committed)
```bash
cd /home/user/trinity
python docs/dev/shell-solver/harness/capture_replay.py            # -> data/replay_comparison.csv  (30 rows)
python docs/dev/shell-solver/harness/capture_replay_variants.py   # -> data/replay_variants.csv    (40 calls x 5 variants)
# one-shot localisation of the dense_output crash + truncation insight:
python docs/dev/shell-solver/harness/diagnose_first_call.py
```

### Coverage gaps (must close before default-flip)
- **Neutral region (2-state) never exercised** — both harnesses aborted after the early ionized solves. Need a capture that reaches a neutral solve (a param that depletes φ with mass remaining, or capturing later in a run).
- **Only the degenerate `simple_cluster` regime** — confirm on a non-overflowing regime (smaller `mCloud`/different profile) that the equivalence still holds and that `t_eval`-only still silences nothing-but-the-noise.

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

### P0 — Equivalence harvest — ✅ DONE (this doc, §P0)
Offline capture-and-replay; zero production change. **Gate G0 (met):** the right `solve_ivp` config reproduces `odeint` to ≤1e-6 on every consumed row of every captured call, and silences the wall. **Branch taken:** PROCEED — but close the two coverage gaps (neutral region, non-degenerate regime) as the first task of P-shadow.

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
1. **Scope now vs later:** ship the lightweight **Option B** (`odeint(mxstep=50000)` + silence the python `ODEintWarning`) as an immediate noise fix — proven `0.00e+00` result change — and pursue the full `solve_ivp` migration separately? Or go straight to `solve_ivp`?
2. **Failure semantics:** on a failing shell solve, raise `ShellSolverError` (bubble-style, changes which steps "succeed") vs. preserve `odeint`'s silent best-effort? P-validate must measure the difference.
3. Acceptable wall-time ceiling vs `odeint`.

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
| Neutral region / non-degenerate regime untested | **open gap** — P-shadow G_shadow blocks promotion until both are covered |
| No integrated-output test exists today | add it in P-promote *before* touching the solver |
