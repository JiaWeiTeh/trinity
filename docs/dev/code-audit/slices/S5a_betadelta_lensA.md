# S5a beta/delta solve — Lens A (what the code does)

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

**Status (2026-07-29):** 📘 raw agent report — provenance for `FINDINGS.md`; unreconciled and unverified on its own.

**Input read:** `trinity/phase1b_energy_implicit/get_betadelta.py` (comment/docstring-blanked copy).
**Shared-file exception used:** yes — I consulted the blanked
`trinity/_functions/unit_conversions.py` for the code unit system only. From the astropy
cross-checks in its `__main__` block (`erg → Msun·pc²/Myr²`, `erg/s → Msun·pc²/Myr³`,
`km/s → pc/Myr`) the "au" system is **mass = Msun, length = pc, time = Myr, temperature = K**.
All dimensions below are in that system. No other file was read.

---

## 0. Module-level constants (lines 41–74)

| line | name | value | where used |
|---|---|---|---|
| 41–44 | `BETA_MIN, BETA_MAX, DELTA_MIN, DELTA_MAX` | `0.0, 1.0, -1.0, 0.0` | `_solve_grid` box clamp (1043–1046), `_solve_lbfgsb` bounds+clip (1118–1119, 1127). **Never used by the `hybr` path.** |
| 47 | `RESIDUAL_THRESHOLD` | `1e-4` | every convergence test (712, 751, 792, 827, 966, 1002) |
| 48 | `MAX_ITERATIONS` | `15` | only `_solve_lbfgsb` `maxiter` (1137) |
| 53 | `LBFGSB_FALLBACK_THRESHOLD` | `5.0` | gate at 771 |
| 56–57 | `GRID_SIZE`, `GRID_EPSILON` | `5`, `0.02` | grid box + node count (1043–1049) |
| 68 | `GRID_EARLY_EXIT_RESIDUAL` | `RESIDUAL_THRESHOLD/10 = 1e-5` | grid early exit (1092) |
| 74 | `HYBR_OPTIONS` | `xtol=1e-8, factor=0.1, maxfev=30, eps=3e-4` | `scipy.optimize.root(..., method='hybr')` (983–985) |

Note the convergence test is on the **sum of squares** of two relative residuals against `1e-4`,
so "converged" means each relative residual is ≲ 10⁻² — a **1 % tolerance**, not 10⁻⁴.
The grid early-exit bar `1e-5` is ≈ 0.3 % each.

---

## 1. `BubbleParamsView` (lines 107–144) and `_MockValue` (99–104)

Read-only proxy. `__init__(params, beta, delta, dMdt_guess=None)` builds
`_overrides = {'cool_beta': _MockValue(beta), 'cool_delta': _MockValue(delta)}` and, if
`dMdt_guess is not None`, also `'bubble_dMdt': _MockValue(dMdt_guess)`.
`__getitem__`/`get` return the override if present, else delegate to `_params`.

State: **writes** nothing to `params`; **reads** any key the callee asks for.
The view implements only `__getitem__` and `get` — no `__setitem__`, `__contains__`, `keys`,
iteration, or attribute access. Any callee that uses those forms raises, and the raise is
swallowed at line 437 (see §5).

## 2. `_describe_exc` (77–92)

`f"{type(e).__name__}: {str(e) or '<no message>'}[ at {basename}:{lineno}]"`, using the **last**
traceback frame. `frame.filename.rsplit('/',1)[-1]` — POSIX separator only. Pure string work.

## 3. `cool_beta_to_Ebdot_pure` (182–269) — the β → Ėb map

Inputs (units): `beta` [1], `Pb` [Msun pc⁻¹ Myr⁻²], `t_now` [Myr], `R1, R2` [pc],
`v2` [pc Myr⁻¹], `Eb` [Msun pc² Myr⁻²], `pdot_total` [Msun pc Myr⁻²],
`pdotdot_total` [Msun pc Myr⁻³]. No dict access; pure function.

Locals:

```
Ṗb   = -Pb·β / t_now                                    [Msun pc⁻¹ Myr⁻³]     (248)
a    = 1.5·p̈/ṗ           if ṗ > 0 else 0.0              [Myr⁻¹]               (251)
c    = 0.75·ṗ·R1                                        [Msun pc² Myr⁻²]      (252)
d    = R2³ − R1³                                        [pc³]                 (253)
Ebc  = Eb + c                                           [energy]              (255)
cf   = c/Ebc              if Ebc > 0 else 0.0           [1]                   (256)
```

Returned value (259–269):

$$
\dot E_b \;=\;
\frac{2\pi\,\dot P_b\,d^{2}\;+\;3\,E_b v_2 R_2^{2}(1-c_f)\;-\;a R_1^{3} E_b^{2}/E_{bc}}
     {d\,(1-c_f)}
$$

Substituted through to the inputs (positive-`Ebc` branch, where `1 − cf = Eb/Ebc`):

$$
\boxed{\;\dot E_b=
-\frac{2\pi\,\beta\,P_b}{t}\,(R_2^{3}-R_1^{3})\;\frac{E_b+0.75\,\dot p R_1}{E_b}
\;+\;\frac{3E_b v_2R_2^{2}}{R_2^{3}-R_1^{3}}
\;-\;\frac{3}{2}\,\frac{\ddot p}{\dot p}\,\frac{R_1^{3}E_b}{R_2^{3}-R_1^{3}}\;}
$$

**Dimensions** — every numerator term is `Msun pc⁵ Myr⁻³`
(`2π Ṗb d²` = Msun pc⁻¹Myr⁻³·pc⁶; `3 Eb v2 R2²` = Msun pc²Myr⁻²·pc Myr⁻¹·pc²;
`a R1³ Eb²/Ebc` = Myr⁻¹·pc³·Msun pc²Myr⁻²), denominator `pc³`, result
`Msun pc² Myr⁻³` = **luminosity**. Balanced.

**Sign convention:** β is defined by `Ṗb = −βPb/t`, i.e. β ≡ −(t/Pb)dPb/dt, so β>0 ⇔ falling
pressure; bounds `[0,1]`. δ (§4) is defined *without* the minus sign, bounds `[−1,0]`. Both
therefore mean "decreasing", but the two definitions differ by a sign. Consistent internally.

**Algebraic check.** With the adiabatic EOS `Pb = 3(γ−1)Eb / (4πd)` and an inner boundary set by
ram-pressure balance `Pb = ṗ/(4πR1²)` — i.e. `ṗ·d = 3(γ−1)Eb R1²` — differentiating
`Eb = 4πPb d/(3(γ−1))` and eliminating `Ṙ1` reproduces the boxed expression **exactly, but only
for γ = 5/3**: the literal `2π` at line 260 is `4π/(3(γ−1))|_{γ=5/3}`, and the literal `0.75` at
line 252 is `1/(2(γ−1))|_{γ=5/3}` (it is the `Ebc` factor in
`∂/∂Ṙ1 = 3ṗR1² + 6(γ−1)EbR1 = 4R1·(Eb + 0.75ṗR1)`). See finding A-01.

**Numeric literals:** `1.5` (a), `0.75` (c), `2` and `np.pi` (first numerator term), `3` (second
term and the R2³/R1³ powers), `1e-300` (denominator guard), `0.0` (both fallbacks).

**Control flow that changes the maths**

* 251 `ṗ ≤ 0` → `a := 0`, which **deletes the whole `p̈` term**. `c` at 252 is *not* similarly
  guarded and stays negative, so the two coefficients are then derived from mutually
  inconsistent assumptions (A-17).
* 256 `Ebc ≤ 0` → `cf := 0`, so the denominator becomes `d` and the second numerator term loses
  its `(1−cf)` — **but line 262 still divides by the un-guarded `Ebc`**, which is then ≤ 0.
  For `Ebc == 0` exactly this is a `ZeroDivisionError` (Python floats), including the
  `0.0/0.0` case when `a == 0` (A-02).
* 266 `|denominator| < 1e-300` → **`return 0.0`**. A degenerate configuration (essentially
  `R2 == R1`, or `Eb == 0`) silently yields `Ėb = 0`, which then trips the `|Edot_from_beta| ≤
  1e-300` fallback in the residual (§5) and swaps a relative residual for a dimensional one.
* Structurally: the `(1−cf)` factor on the second numerator term (261) **cancels exactly**
  against the `(1−cf)` in the denominator (264). It is algebraically inert (A-18).

## 4. `delta2dTdt_pure` (272–294)

`return 0.0 if t <= 0 else (T/t)·δ` → `dT/dt = δT/t` [K Myr⁻¹]. δ ≡ (t/T)dT/dt.
**Not called anywhere in this module** (it may be imported elsewhere; I cannot see that).

## 5. `compute_R1_Pb` (297–331)

```
R1 = get_bubbleParams.solve_R1(R2, Eb, Lmech_total, v_mech_total)      # no gamma argument
Pb = get_bubbleParams.bubble_E2P(Eb, R2, R1, gamma_adia)               # gamma argument
```
Returns `(R1 [pc], Pb [Msun pc⁻¹ Myr⁻²])`. R1 is obtained γ-independently, Pb γ-dependently;
the closed form of §3 requires the two to satisfy `ṗ·d = 3(γ−1)EbR1²` simultaneously, which can
hold for at most one γ (A-01).

## 6. `effective_Lloss` (334–357) / `effective_Lloss_from_params` (360–371)

```
mode == 'multiplier'    →  L_loss = L_leak + f_mix·L_cool
mode == 'theta_target'  →  L_loss = max(L_cool + L_leak,  θ·L_mech)
otherwise (incl. typos) →  L_loss = L_cool + L_leak
```
All terms luminosity; `f_mix`, `θ` dimensionless. `_from_params` reads `cooling_boost_mode`
(default `'none'`, and `... or 'none'` also maps `''`/`None`/`0` to `'none'`),
`cooling_boost_fmix` (default `1.0`), `cooling_boost_theta` (default `0.0`). Any unrecognised
mode string falls silently into the no-boost branch (A-15).

## 7. `_usable_dMdt` (374–386)

`None` unless `props is not None and isfinite(dMdt) and dMdt > 0`, else `float(dMdt)`.
Used as the truthiness test for seed propagation — note `0.0` can never be returned, so the
`or state['seed']` idiom at 978 is safe.

## 8. `get_residual_pure` (393–497) — **the residual**

Signature `(beta, delta, params, return_bubble_props=False, dMdt_guess=None)`.
**Reads** from `params`: `R2, v2, Eb, T0, t_now, gamma_adia, Lmech_total, v_mech_total,
pdot_total, pdotdot_total` (all `.value`), plus optional `bubble_Leak`, `cooling_boost_*`.
**Writes** nothing. β, δ, and the dMdt seed reach the structure solver only through the view.

1. `bubble_props = get_bubbleproperties_pure(BubbleParamsView(params, β, δ, dMdt_guess))`.
   **`except Exception` → `return 100.0, 100.0, None`** (439). Sentinel, see A-04.
2. `R1, Pb = compute_R1_Pb(R2, Eb, Lmech_total, v_mech_total, gamma_adia)`.
3. `X ≡ Edot_from_beta = cool_beta_to_Ebdot_pure(β, Pb, t, R1, R2, v2, Eb, ṗ, p̈)` (§3).
4. `L_gain = Lmech_total`; `L_cool = bubble_props.bubble_LTotal`;
   `L_leak = params['bubble_Leak'].value` or `0.0`; `L_loss = effective_Lloss_from_params(...)`.
5. `B ≡ Edot_from_balance = L_mech − L_loss − 4π R2² v2 Pb` (475).
   Dimensions: `pc²·pc Myr⁻¹·Msun pc⁻¹Myr⁻² = Msun pc²Myr⁻³` ✓ (this is `Pb·dV/dt` with
   `V = (4/3)πR2³`, i.e. only the outer boundary moves).
6. **Residual 1** (478–481):
   `f_E = (X − B)/X` if `|X| > 1e-300`, **else** `f_E = B` (or `0.0` when `abs(B) > 0` is False).
7. **Residual 2** (488–493): `T_b = bubble_props.bubble_T_r_Tb`;
   `f_T = (T_b − T0)/T0` if `|T0| > 1e-300`, **else** `f_T = T_b` (or `0.0`).
8. Returns `(f_E, f_T, bubble_props or None)`.

So the solve drives **`X(β,δ) − B(β,δ) → 0` and `T_b(β,δ) − T0 → 0`**, with β entering both the
closed form of §3 and (through `cool_beta`) the structure solve, and δ entering **only** through
the structure solve.

Both fallback branches are dimensionally and numerically different from the primary branch:
`f_E` becomes a raw luminosity in code units and `f_T` a raw temperature in K, and each maps
**NaN → 0.0** (A-03).

## 9. `get_residual_detailed` (514–605) + `ResidualDetails` (500–511)

Same arithmetic as §8, returning the intermediates as well. Two differences that matter:

* When `bubble_props is None` it rebuilds the view **without `dMdt_guess`** (534) — a different
  seed from whatever produced the residual being reported.
* On failure it returns `Edot_residual=100.0, T_residual=100.0`, all physical fields `NaN`,
  `bubble_props=None`, and (dataclass defaults) **`L_gain=0.0, L_loss=0.0`** (541–549).

## 10. `_get_betadelta_solver` (612–622) / `solve_betadelta_pure` (625–646)

`params.get('betadelta_solver')` → `.value` → `'legacy'` on `None`/missing/falsy.
`'legacy'` → §11; `'hybr'` → §13, plus a rescue when `no_physical_root` and the reason contains
`'structure solve failed'`; anything else → `ValueError`.
The `method` argument is threaded into every solver and **never read** (A-12).

## 11. `_solve_betadelta_legacy` (678–866)

```
(fE0,fT0,props0) = get_residual_pure(β0, δ0, params, return_bubble_props=True)   # no dMdt seed
r0 = fE0² + fT0²
if r0 < 1e-4:  return (β0, δ0, converged=True, iterations=0, props0)             # early exit
candidates = [(β0,δ0,r0,'input',0,props0)]  if isfinite(r0)
grid  → _solve_grid(β0, δ0, params, input_residual=r0, input_props=props0)       # ±0.02 box
        appended if finite; grid_converged = r_grid < 1e-4
if (not grid_converged) and (grid_residual > 5.0):                               # ← gate
        lbfgsb → _solve_lbfgsb(β0, δ0, params); residual re-evaluated, props discarded (None)
if not candidates:  return (β0, δ0, ±inf residuals, converged=False, props=None)
candidates.sort(by residual);  best = candidates[0]
converged = best_residual < 1e-4
details   = get_residual_detailed(best_β, best_δ, params, bubble_props=best_props)  # ← re-solve
                                                                                   #   if None
return BetaDeltaResult(beta=best_β, delta=best_δ,
                       Edot_residual=details.Edot_residual,   # ← from the re-solve
                       T_residual=details.T_residual,
                       total_residual=best_residual,          # ← from the ranking pass
                       converged=converged, iterations=best_iterations,
                       bubble_properties=details.bubble_props, ...)
```

* The `grid_residual > 5.0` gate means residuals in **[1e-4, 5]** (≈ 1 % to 220 % relative error)
  get no second solver at all (A-10).
* `total_residual`/`converged` come from one evaluation, every other reported field from a
  **second, differently-seeded** evaluation (A-05).
* `lbfgsb_result` (790) is assigned and never read (A-11).
* `no_physical_root` is never set on this path — a total failure returns
  `converged=False, no_physical_root=False, bubble_properties=None`.

## 12. `_solve_grid` (1010–1105)

```
β ∈ linspace(max(0, β0−0.02), min(1, β0+0.02), 5)
δ ∈ linspace(max(−1, δ0−0.02), min(0, δ0+0.02), 5)
```
25 nodes, scanned in order of `(i−2)² + (j−2)²` then `(i, j)` (centre-out spiral), skipping the
node that equals the guess to within `1e-12` in **both** coordinates. `best_*` is initialised
from the caller's input residual/props, so the guess always competes. `dMdt` from the last
successful evaluation is fed forward as `dMdt_guess` to the next (1081, 1086). Early exit at
`residual < 1e-5`. Per-point `except Exception → warn + continue` (failed points are not counted
in `n_evals`, which is returned as `iterations`). Returns
`(best_β, best_δ, best_props, best_residual, n_evals)`.

Consequences: the search step is **hard-capped at ±0.02 per call**; the returned (β,δ) is pinned
at a box edge whenever the true root is farther away. The clamps are `max`/`min` per endpoint,
so a guess outside the declared bounds produces an inverted, out-of-bounds `linspace` (A-09).
Seed carry-forward makes the residual a function of scan history, not just of (β,δ) (A-13).

## 13. `_solve_betadelta_hybr` (948–1007), `_hybr_g_residual` (879–906), helpers

`_hybr_g_residual` evaluates §8 with props, then

```
gE = (Edot_from_beta − Edot_from_balance) / Lmech_total          # dimensionless
gT = T_residual = (T_b − T0)/T0                                  # dimensionless
```

raising `_NoPhysicalRoot` (a **`BaseException`** subclass, so `except Exception` blocks do not
catch it) when `props is None` or `dMdt` is non-finite/≤ 0. **`Lmech_total` is divided by with no
guard** (A-08).

Driver:
1. Evaluate at the guess; `_NoPhysicalRoot` → `_no_root_result` (guess returned, ±inf residuals,
   `converged=False`, `no_physical_root=True`).
2. `g = gE² + gT² < 1e-4` → return converged, `iterations=0`.
3. `scipy.optimize.root(gvec, [β0, δ0], method='hybr', options={xtol:1e-8, factor:0.1,
   maxfev:30, eps:3e-4})`. `gvec` counts evaluations, carries the dMdt seed forward and caches
   the last `(b, d, det, gE, gT)`.
4. After the solve, `b, d = sol.x`; the cached `det` is reused only if the **last** evaluated
   point matches `sol.x` to 1e-12, otherwise a fresh (differently seeded) evaluation is made.
5. `converged = (gE² + gT²) < 1e-4` — **`sol.success` / `sol.status` are used only in the log
   message**, never in the decision. Returns the last iterate with `converged` set accordingly.

**No bound is applied anywhere on this path** — `BETA_MIN/MAX`, `DELTA_MIN/MAX` are not passed to
`root` and the returned `b, d` are not clipped (A-06). And `gE` is normalised by `Lmech_total`
whereas the legacy `f_E` is normalised by `Edot_from_beta`, yet both are compared to the same
`1e-4` (A-07).

`_rescue_structure_failure` (649–675): re-runs the legacy solver; if it returns exactly the input
guess (float equality) the original failure is returned; otherwise hybr is retried from the legacy
point and the retry is returned **unless** it also reports `no_physical_root` (a retry that merely
fails to converge *is* returned, with `converged=False`).

## 14. `_solve_lbfgsb` (1108–1145)

Objective `f(β,δ) = f_E² + f_T²` with the arguments clipped into the bounds (redundant with the
`bounds=` argument) and `except Exception → 1e10`. `scipy.optimize.minimize(..., 'L-BFGS-B',
bounds=[(0,1),(−1,0)], maxiter=15, ftol=1e-8, gtol=1e-6)`; on exception it returns
**`(β0, δ0, 0)`** — the guess, indistinguishable from a solver that simply did not move. No
analytic gradient is supplied, so L-BFGS-B finite-differences a noisy, seed-dependent inner solve.

## 15. `get_beta_delta_wrapper_pure` (1152–1179)

`result = solve_betadelta_pure(β0, δ0, params)` → `((result.beta, result.delta), result)`.
Note `method` is not forwarded here at all, so the default `'grid'` always applies (and is unused).

---

## Non-convergence behaviour — summary

The code is mostly honest about failure: every non-converged path sets `converged=False`, and the
hybr path additionally sets `no_physical_root`. The exceptions are:

* the NaN→0.0 masking in the residual fallbacks (A-03), which can produce `total_residual = 0.0`
  and `converged=True` from a failed evaluation;
* the legacy `details` re-solve (A-05), which can attach `Edot_residual=100`, `L_gain=L_loss=0`,
  `T_bubble=NaN` and `bubble_properties=None` to a result flagged `converged=True`;
* `_solve_lbfgsb`'s exception path returning the guess with `nit=0` (harmless — the caller
  re-evaluates the residual there).

The stale-value question: on total failure the legacy path returns the **input guess unchanged**
with `±inf` residuals; on partial failure it returns the best of {input, grid, lbfgsb}, which is
frequently the input guess itself. Grid-only motion is capped at 0.02 per call in each coordinate.

---

```json
[
  {
    "id": "S5a-A-01",
    "file": "trinity/phase1b_energy_implicit/get_betadelta.py",
    "line": 260,
    "class": "coefficient",
    "severity": "S2",
    "claim": "cool_beta_to_Ebdot_pure hard-codes gamma = 5/3 in two literals (2*np.pi at line 260 and 0.75 at line 252) while gamma_adia is a runtime parameter fed to bubble_E2P at line 329, so Edot_from_beta and Pb are mutually inconsistent for any gamma != 5/3.",
    "evidence": "Differentiating Eb = 4*pi*Pb*d/(3*(gamma-1)) with R1 fixed by ram-pressure balance gives the returned expression exactly, with prefactor 4*pi/(3*(gamma-1)) (= 2*pi only at gamma=5/3) and with c_coeff = pdot*R1/(2*(gamma-1)) (= 0.75*pdot*R1 only at gamma=5/3). Line 329 passes gamma_adia to bubble_E2P; line 327 solve_R1 gets no gamma at all.",
    "expected": "2*np.pi -> 4*np.pi/(3*(gamma_adia-1)) at line 260 and 0.75 -> 1/(2*(gamma_adia-1)) at line 252, with gamma_adia threaded into cool_beta_to_Ebdot_pure; or an explicit assertion that gamma_adia == 5/3.",
    "failure_scenario": "A .param setting gamma_adia to anything other than 5/3 (e.g. 1.4 for a molecular/partially-ionised interior): Pb is computed with the new gamma but Edot_from_beta keeps the 5/3 coefficients, so the beta residual is driven to zero against the wrong Edot and beta comes out biased by roughly the ratio (2/3)/(gamma-1).",
    "repro": "Pick R1, R2, Eb, pdot satisfying pdot*(R2**3-R1**3) == 3*(g-1)*Eb*R1**2 and Pb = 3*(g-1)*Eb/(4*pi*(R2**3-R1**3)); assert cool_beta_to_Ebdot_pure(...) equals the analytic 4*pi/(3*(g-1))*(Pb_dot*d + Pb*d_dot). Passes for g=5/3, fails for g=1.4.",
    "confidence": "high"
  },
  {
    "id": "S5a-A-02",
    "file": "trinity/phase1b_energy_implicit/get_betadelta.py",
    "line": 262,
    "class": "numerical",
    "severity": "S2",
    "claim": "Ebc is guarded before being used as a divisor at line 256 (c_frac) but not at line 262, so Ebc == 0 raises ZeroDivisionError and Ebc < 0 silently evaluates a different formula from the one the algebra implies.",
    "evidence": "Line 256 'c_frac = c_coeff / Ebc if Ebc > 0 else 0.0' versus line 262 '- a_coeff * R1**3 * Eb**2 / Ebc' with no guard. When the guard at 256 fires, the denominator at 264 becomes d_coeff (not d_coeff*Eb/Ebc) while the third numerator term still carries a negative Ebc, so the two uses of the same quantity disagree in sign.",
    "expected": "One guard covering both uses, e.g. return early / fall back consistently when Ebc <= 0, rather than zeroing c_frac while keeping 1/Ebc live.",
    "failure_scenario": "Any state with Eb + 0.75*pdot_total*R1 <= 0 (bubble energy driven negative late in the energy-driven phase, or pdot_total < 0). Exactly zero raises; slightly negative returns a value whose third term has the wrong sign relative to the (now-suppressed) c_frac normalisation.",
    "repro": "cool_beta_to_Ebdot_pure(beta=0.5, Pb=1.0, t_now=1.0, R1=1.0, R2=2.0, v2=1.0, Eb=-0.75, pdot_total=1.0, pdotdot_total=0.0) -> ZeroDivisionError (0.0/0.0) instead of a finite value.",
    "confidence": "high"
  },
  {
    "id": "S5a-A-03",
    "file": "trinity/phase1b_energy_implicit/get_betadelta.py",
    "line": 481,
    "class": "silent-failure",
    "severity": "S2",
    "claim": "The residual fallbacks 'X if abs(X) > 0 else 0.0' (lines 481, 493 and their duplicates 585, 593) map NaN to 0.0, so a failed evaluation can report a perfect residual and be declared converged.",
    "evidence": "abs(nan) > 0 is False, so the expression yields 0.0. These branches are reached whenever abs(Edot_from_beta) <= 1e-300 (which cool_beta_to_Ebdot_pure returns exactly, by design, at line 267) or abs(T0) <= 1e-300. total_res_input = 0.0 then passes the 'total_res_input < RESIDUAL_THRESHOLD' test at line 712 and returns converged=True with iterations=0.",
    "expected": "Return a large finite sentinel or raise on a non-finite residual; never map NaN to zero.",
    "failure_scenario": "A step where R2 has effectively caught up with R1 so d_coeff ~ 0 -> Edot_from_beta = 0.0 (line 267), and the balance side is NaN because the structure solve produced NaN luminosities: Edot_residual = 0.0, and if T0 is also 0 with T_bubble NaN, total residual 0.0 -> 'converged' at the untouched input guess.",
    "repro": "get_residual_pure with params such that R2 == R1 and bubble_LTotal = nan; assert the returned Edot_residual is not 0.0.",
    "confidence": "medium"
  },
  {
    "id": "S5a-A-04",
    "file": "trinity/phase1b_energy_implicit/get_betadelta.py",
    "line": 481,
    "class": "units",
    "severity": "S2",
    "claim": "The residual fallback branches return a dimensional quantity where the primary branch returns a dimensionless relative residual: Edot_residual becomes a raw luminosity [Msun pc^2 Myr^-3] (line 481/585) and T_residual a raw temperature in K (line 493/593).",
    "evidence": "Primary branches are (X-B)/X and (T-T0)/T0, both dimensionless; the else-branches assign B and T_bubble directly. These feed the same sum-of-squares that is compared to RESIDUAL_THRESHOLD = 1e-4 at lines 712, 751, 792, 827.",
    "expected": "Normalise the fallback by a fixed scale (Lmech_total for Edot, T_bubble or a reference temperature for T), as the hybr path already does for gE at line 904.",
    "failure_scenario": "If T0 is ever 0 (e.g. an uninitialised target temperature on the first Phase-1b step), T_residual = T_bubble ~ 1e6-1e7 K, its square ~1e12-1e14 dominates the objective, the L-BFGS-B fallback gate at line 771 trips, and the reported total_residual is meaningless.",
    "repro": "get_residual_pure with params['T0'].value = 0.0; assert abs(T_residual) is O(1) rather than O(T_bubble).",
    "confidence": "medium"
  },
  {
    "id": "S5a-A-05",
    "file": "trinity/phase1b_energy_implicit/get_betadelta.py",
    "line": 841,
    "class": "state",
    "severity": "S2",
    "claim": "In the legacy path total_residual and converged come from the ranking evaluation, but every other returned field (Edot_residual, T_residual, Edot_from_beta/balance, T_bubble, L_gain, L_loss, bubble_properties) comes from a second, differently-seeded re-evaluation at line 841 that can fail outright.",
    "evidence": "Line 841 calls get_residual_detailed(best_beta, best_delta, params, bubble_props=best_props). The L-BFGS-B candidate is appended with props=None (line 788), so for that candidate the call rebuilds BubbleParamsView WITHOUT dMdt_guess (line 534) and re-solves the structure. Its failure path (541-549) returns Edot_residual=100.0, T_residual=100.0, NaN physical fields, bubble_props=None, and dataclass-default L_gain=L_loss=0.0, all of which are copied into the result at 852-865 while total_residual=best_residual and converged stay from the earlier pass.",
    "expected": "Carry the residuals and props from the winning evaluation, or recompute total_residual from the same details object so the reported fields are self-consistent.",
    "failure_scenario": "grid_residual > 5.0 so the L-BFGS-B branch runs and wins; the re-solve at 841 lands on a different dMdt branch or fails -> the caller receives converged=True/total_residual=1e-5 alongside Edot_residual=100 and bubble_properties=None, and the bubble_properties that propagate downstream are from a different structure solve than the one that scored the point.",
    "repro": "Force the lbfgsb candidate to win and make the re-solve fail (monkeypatch get_bubbleproperties_pure to raise on the second call); assert result.converged implies isfinite(result.Edot_from_beta) and result.bubble_properties is not None.",
    "confidence": "high"
  },
  {
    "id": "S5a-A-06",
    "file": "trinity/phase1b_energy_implicit/get_betadelta.py",
    "line": 983,
    "class": "regime",
    "severity": "S2",
    "claim": "The hybr solver is unbounded: BETA_MIN/BETA_MAX and DELTA_MIN/DELTA_MAX are never applied in _solve_betadelta_hybr, so it can return beta > 1, beta < 0, delta > 0 or delta < -1, unlike the grid and L-BFGS-B paths which both enforce them.",
    "evidence": "scipy.optimize.root(gvec, [beta_guess, delta_guess], method='hybr', ...) at 983-985 takes no bounds, and b, d = float(sol.x[0]), float(sol.x[1]) at 989 are returned unclipped through _hybr_result. Compare lines 1043-1046 (grid clamp) and 1118-1119/1127 (L-BFGS-B clip + bounds).",
    "expected": "Either clip/reject solutions outside the declared boxes, or document that hybr deliberately allows excursions; at minimum the two solvers should agree on the admissible set.",
    "failure_scenario": "betadelta_solver = 'hybr' on a stiff step where the Newton step overshoots: the returned beta = 1.3 (pressure falling faster than t^-1) or delta = +0.2 (rising temperature) is handed on as a converged root, with a cooling structure evaluated outside the range the grid path can ever reach.",
    "repro": "Run the hybr solver on any config and assert BETA_MIN <= result.beta <= BETA_MAX and DELTA_MIN <= result.delta <= DELTA_MAX; no code enforces it.",
    "confidence": "high"
  },
  {
    "id": "S5a-A-07",
    "file": "trinity/phase1b_energy_implicit/get_betadelta.py",
    "line": 904,
    "class": "numerical",
    "severity": "S2",
    "claim": "The same RESIDUAL_THRESHOLD = 1e-4 is applied to two differently normalised energy residuals - legacy uses fE = (X-B)/X (normalised by the beta-dependent Edot_from_beta) while hybr uses gE = (X-B)/Lmech_total - so 'converged' means a different physical accuracy depending on which solver is selected.",
    "evidence": "Line 479/583 'Edot_residual = (Edot_from_beta - Edot_from_balance) / Edot_from_beta' versus line 904 'gE = (det.Edot_from_beta - det.Edot_from_balance) / Lmech_total'; both squared sums are compared to RESIDUAL_THRESHOLD at 712/751/792/827 and 966/1002.",
    "expected": "One normalisation, or two separately named thresholds calibrated to each.",
    "failure_scenario": "A state where |Edot_from_beta| ~ 1e-3 * Lmech_total and the imbalance is comparable to Edot_from_beta itself: gE^2 ~ 1e-6 < 1e-4 so hybr reports converged with a 100 % error in Edot, while legacy computes fE^2 ~ 1 and reports failure at the same point.",
    "repro": "Evaluate both residual definitions at the same (beta, delta) for a config with Edot_from_beta << Lmech_total and compare which side of 1e-4 each lands on.",
    "confidence": "medium"
  },
  {
    "id": "S5a-A-08",
    "file": "trinity/phase1b_energy_implicit/get_betadelta.py",
    "line": 904,
    "class": "divergence",
    "severity": "S2",
    "claim": "gE divides by Lmech_total with no zero/finiteness guard, and the resulting ZeroDivisionError is not caught by any handler on the hybr path (only _NoPhysicalRoot is caught at 962, 986, 998).",
    "evidence": "Lines 903-904: 'Lmech_total = float(params[...].value); gE = (...) / Lmech_total'. _NoPhysicalRoot subclasses BaseException specifically so it survives except-Exception blocks; a ZeroDivisionError from 904 has no handler between here and solve_betadelta_pure's caller.",
    "expected": "Guard Lmech_total (fall back to a nonzero scale, or raise _NoPhysicalRoot with a reason) as the other divisions in this file are guarded.",
    "failure_scenario": "betadelta_solver='hybr' at a time when the cluster mechanical luminosity has dropped to exactly zero (e.g. a truncated SB99 table returning 0, or feedback switched off): the run aborts with an unhandled ZeroDivisionError instead of reporting no_physical_root.",
    "repro": "_hybr_g_residual with params['Lmech_total'].value = 0.0 -> ZeroDivisionError.",
    "confidence": "medium"
  },
  {
    "id": "S5a-A-09",
    "file": "trinity/phase1b_energy_implicit/get_betadelta.py",
    "line": 1043,
    "class": "numerical",
    "severity": "S2",
    "claim": "The grid box clamps are applied independently to each endpoint, so a guess more than GRID_EPSILON outside the declared bounds yields an inverted linspace whose nodes all lie outside [BETA_MIN, BETA_MAX] / [DELTA_MIN, DELTA_MAX].",
    "evidence": "Lines 1043-1049: beta_min = max(0.0, b0-0.02), beta_max = min(1.0, b0+0.02); for b0 = 1.05 this is beta_min = 1.03 > beta_max = 1.00 and np.linspace(1.03, 1.00, 5) silently returns a descending sequence, every element >= 1.0. Same construction for delta with DELTA_MAX = 0.0.",
    "expected": "Clip the guess into the box first (np.clip(beta_guess, BETA_MIN, BETA_MAX)) before forming the endpoints, so beta_min <= beta_max always holds.",
    "failure_scenario": "The hybr path (unbounded, see A-06) returns beta = 1.05 which is fed back as the next step's guess, or a rescue call passes such a guess: the grid then scans beta in [1.00, 1.03] only, evaluating the structure solve entirely outside the physical beta range and returning the best of those.",
    "repro": "_solve_grid(beta_guess=1.05, delta_guess=-0.5, ...): assert beta_range.min() >= BETA_MIN and beta_range.max() <= BETA_MAX - fails.",
    "confidence": "high"
  },
  {
    "id": "S5a-A-10",
    "file": "trinity/phase1b_energy_implicit/get_betadelta.py",
    "line": 771,
    "class": "regime",
    "severity": "S3",
    "claim": "The L-BFGS-B fallback is gated on grid_residual > LBFGSB_FALLBACK_THRESHOLD = 5.0, so residuals between the convergence bar (1e-4) and 5.0 - i.e. roughly 1 % to 220 % relative error - get no second solver, and the returned (beta, delta) is whatever the +/-0.02 grid box could reach.",
    "evidence": "Line 771 'if not grid_converged and grid_residual > LBFGSB_FALLBACK_THRESHOLD'. GRID_EPSILON = 0.02 (line 57) caps the per-call motion, and best_* in _solve_grid is initialised to the input guess (1062-1064), so a failed grid returns the guess unchanged.",
    "expected": "If the fallback exists to rescue non-convergence, the natural gate is 'not grid_converged'; the 5.0 bar means the fallback fires only for catastrophic residuals.",
    "failure_scenario": "A stiff step where the true root is 0.1 away in beta: the grid pins beta at beta_guess +/- 0.02 with residual ~0.5, no fallback runs, and the caller receives converged=False with a value only marginally moved. Whether this changes physical output depends entirely on how the caller treats converged=False (outside this slice).",
    "repro": "Instrument a run and count steps returning converged=False with 1e-4 < total_residual < 5.0; all of them skipped the fallback.",
    "confidence": "medium"
  },
  {
    "id": "S5a-A-11",
    "file": "trinity/phase1b_energy_implicit/get_betadelta.py",
    "line": 790,
    "class": "deadcode",
    "severity": "S4",
    "claim": "lbfgsb_result is assigned at line 790 (and initialised at 768) and never read anywhere, unlike its sibling grid_result which is read at line 769.",
    "evidence": "The identifier appears only at lines 768 and 790 in the whole file.",
    "expected": "Delete the variable, or use it the way grid_result is used.",
    "failure_scenario": "",
    "repro": "ruff F841 flags it; the project's ruff rule set is restricted to F821/F811/F823/E9 so it is not caught today.",
    "confidence": "high"
  },
  {
    "id": "S5a-A-12",
    "file": "trinity/phase1b_energy_implicit/get_betadelta.py",
    "line": 629,
    "class": "deadcode",
    "severity": "S4",
    "claim": "The 'method' parameter is threaded through solve_betadelta_pure, _solve_betadelta_legacy, _solve_betadelta_hybr and _rescue_structure_failure and is never read in any of them; the only public entry point (get_beta_delta_wrapper_pure) does not forward it either.",
    "evidence": "Declared at 629, 682, 948, 649 and passed at 639, 641, 643, 664, 667; no function body references it.",
    "expected": "Remove the parameter, or make the solver actually dispatch on it.",
    "failure_scenario": "",
    "repro": "grep for 'method' inside the four function bodies - only the parameter declarations and pass-throughs appear.",
    "confidence": "high"
  },
  {
    "id": "S5a-A-13",
    "file": "trinity/phase1b_energy_implicit/get_betadelta.py",
    "line": 978,
    "class": "numerical",
    "severity": "S2",
    "claim": "The dMdt seed is carried forward between evaluations, so the residual is a function of evaluation history rather than of (beta, delta) alone; the same point evaluated twice can return different residuals, and hybr's finite-difference Jacobian columns are built from differently seeded evaluations.",
    "evidence": "Line 978 \"state['seed'] = _usable_dMdt(det_i.bubble_props) or state['seed']\" inside gvec, and lines 1081/1086 in _solve_grid ('dMdt_guess=last_dMdt', updated after each successful point). The seed reaches the structure solver via the 'bubble_dMdt' override at line 134. Line 992-997 makes this visible: the final det is either the cached last iterate or a fresh evaluation with whatever seed happened to be current.",
    "expected": "Either seed deterministically from a fixed reference state per call, or document that g is only reproducible for a fixed evaluation order; a Newton solver with a numerical Jacobian needs a deterministic function.",
    "failure_scenario": "Two runs of the same config differing only in the grid scan reaching an early exit at a different point end up with different seeds and hence different converged dMdt at the accepted (beta, delta) - a reproducibility break that survives into bubble_properties.",
    "repro": "Call get_residual_pure at the same (beta, delta) twice with dMdt_guess=None and with dMdt_guess set to a neighbouring point's value; compare the residuals.",
    "confidence": "medium"
  },
  {
    "id": "S5a-A-14",
    "file": "trinity/phase1b_energy_implicit/get_betadelta.py",
    "line": 74,
    "class": "numerical",
    "severity": "S4",
    "claim": "HYBR_OPTIONS pairs xtol=1e-8 with a finite-difference Jacobian step eps=3e-4 on a noisy inner solve, and caps the budget at maxfev=30 for a 2-D problem (~3 evaluations per Jacobian + step), so the step tolerance is four orders of magnitude below the noise floor and termination is normally by evaluation budget.",
    "evidence": "Line 74 'HYBR_OPTIONS = dict(xtol=1e-8, factor=0.1, maxfev=30, eps=3e-4)'. sol.status is only logged (line 1005), never acted on; convergence is decided solely by g_total < 1e-4 at line 1002.",
    "expected": "xtol commensurate with eps (e.g. 1e-4-ish) or a documented rationale; and either use sol.status or state that the residual test supersedes it.",
    "failure_scenario": "",
    "repro": "Log sol.status across a run: expect status 5 (maxfev exceeded) rather than status 1 on stiff steps.",
    "confidence": "medium"
  },
  {
    "id": "S5a-A-15",
    "file": "trinity/phase1b_energy_implicit/get_betadelta.py",
    "line": 357,
    "class": "silent-failure",
    "severity": "S4",
    "claim": "effective_Lloss falls through to the unboosted 'Lcool + Lleak' for any unrecognised cooling_boost_mode string, so a typo in the .param silently disables the cooling boost with no warning.",
    "evidence": "Lines 353-357: only 'multiplier' and 'theta_target' are matched; every other value, including misspellings, reaches the bare 'return Lcool + Lleak'. effective_Lloss_from_params (366) only special-cases the exact string 'none' before delegating.",
    "expected": "Raise (or at minimum log a warning) on an unknown mode, as solve_betadelta_pure already does for an unknown betadelta_solver at line 646.",
    "failure_scenario": "A .param with cooling_boost_mode = 'multiply' (or trailing whitespace surviving parsing) runs the entire sweep with no cooling boost while the user believes fmix is applied.",
    "repro": "effective_Lloss('multiplyer', 2.0, 0.0, 1.0, 0.0, 10.0) returns 1.0 instead of raising.",
    "confidence": "medium"
  },
  {
    "id": "S5a-A-16",
    "file": "trinity/phase1b_energy_implicit/get_betadelta.py",
    "line": 469,
    "class": "state",
    "severity": "S4",
    "claim": "bubble_Leak is read via getattr(params.get('bubble_Leak', None), 'value', 0.0), so if the key holds a plain float rather than an object with a .value attribute the leak luminosity is silently replaced by 0.0.",
    "evidence": "Lines 469-471 and the duplicate at 573-575. Every other parameter in the same block is read as params['X'].value (442-451), which would raise on the same mismatch; only bubble_Leak degrades silently. The following 'if bubble_Leak is None' check handles an explicit None value but not the wrong container type.",
    "expected": "Read it the same way as the neighbouring parameters, or make the getattr default explicit about the two distinct failure modes (key absent vs. wrong shape).",
    "failure_scenario": "A code path that stores bubble_Leak as a bare float drops the leak term from Edot_from_balance entirely, shifting the beta root without any diagnostic.",
    "repro": "get_residual_pure with params['bubble_Leak'] = 1.0e3 (a float): L_loss silently omits it.",
    "confidence": "low"
  },
  {
    "id": "S5a-A-17",
    "file": "trinity/phase1b_energy_implicit/get_betadelta.py",
    "line": 251,
    "class": "sign",
    "severity": "S4",
    "claim": "a_coeff is zeroed when pdot_total <= 0 (line 251) but c_coeff = 0.75*pdot_total*R1 (line 252) is not, so for non-positive pdot_total the two coefficients derived from the same momentum-flux relation disagree: the pdotdot term is deleted while a negative c_coeff still reshapes the denominator.",
    "evidence": "Line 251 'a_coeff = 1.5 * pdotdot_total / pdot_total if pdot_total > 0 else 0.0' immediately above the unguarded line 252.",
    "expected": "Treat pdot_total <= 0 consistently in both coefficients (both derive from the same ram-pressure relation), or reject the state.",
    "failure_scenario": "pdot_total <= 0 (feedback momentum exhausted or a sign convention flip upstream): the returned Edot_from_beta keeps a negative c_coeff in Ebc, which can drive Ebc <= 0 and hit finding A-02, while the pdotdot contribution has already been silently discarded.",
    "repro": "cool_beta_to_Ebdot_pure with pdot_total = -1.0: a_coeff = 0 but c_coeff = -0.75*R1, and the result is neither the full formula nor a clean fallback.",
    "confidence": "medium"
  },
  {
    "id": "S5a-A-18",
    "file": "trinity/phase1b_energy_implicit/get_betadelta.py",
    "line": 261,
    "class": "other",
    "severity": "S4",
    "claim": "The (1 - c_frac) factor on the second numerator term at line 261 cancels exactly against the (1 - c_frac) in the denominator at line 264, so it is algebraically inert while appearing to modulate that term.",
    "evidence": "numerator term 2 = 3*Eb*v2*R2**2*(1-c_frac); denominator = d_coeff*(1-c_frac); their quotient is 3*Eb*v2*R2**2/d_coeff independent of c_frac. Only the first and third numerator terms actually see c_frac.",
    "expected": "Either factor (1-c_frac) out of the whole expression, or leave the term as 3*Eb*v2*R2**2 with the denominator handled once - the present form invites a reader (and an editor) to believe c_frac scales the v2 term.",
    "failure_scenario": "",
    "repro": "Evaluate the function with the (1-c_frac) factor removed from line 261 and c_frac forced to 0 in the denominator only - the difference isolates the redundancy.",
    "confidence": "high"
  },
  {
    "id": "S5a-A-19",
    "file": "trinity/phase1b_energy_implicit/get_betadelta.py",
    "line": 856,
    "class": "other",
    "severity": "S4",
    "claim": "The 'iterations' field means three different things depending on which candidate wins: 0 for the input guess, a grid evaluation count for the grid, scipy's nit for L-BFGS-B (line 1142), and a function-evaluation count for hybr (state['n'], line 1007).",
    "evidence": "Line 856 iterations=best_iterations sourced from candidates appended at 733 (0), 747 (iter_grid = n_evals from _solve_grid line 1105) and 787 (result.nit); line 1007 passes state['n'], which gvec increments once per residual evaluation.",
    "expected": "One definition, or separate fields for evaluations and iterations.",
    "failure_scenario": "",
    "repro": "Compare result.iterations across the three winning branches for the same config; the numbers are not on a common scale.",
    "confidence": "high"
  }
]
```
