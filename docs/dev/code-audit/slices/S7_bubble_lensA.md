# S7 bubble structure — Lens A (what the code does)

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

**Coverage.** Every line read, no sampling:
`trinity/bubble_structure/bubble_luminosity.py` 1–1145 (complete);
`trinity/bubble_structure/get_bubbleParams.py` 1–455 (complete);
`trinity/bubble_structure/__init__.py` (empty after comment blanking — no code).

**Shared-file consultation (declared).** I read
`trinity/_functions/unit_conversions.py` (the permitted shared read-only exception) to obtain the
numeric values of the `cvt.*` constants. Nothing else outside the slice was opened.

**Units system inferred from the conversion constants alone** (no prose was available): the code's
"astro units" (AU) are **mass = M<sub>☉</sub>, length = pc, time = Myr, temperature = K**. Derived:
`ndens_cgs2au = pc2cm³`, `phi_cgs2au = pc2cm²·Myr2s`, `Pb_cgs2au = g2Msun/cm2pc/s2Myr²`,
`c_therm_cgs2au = L_cgs2au/cm2pc` ⇒ `C_thermal` has AU dimensions **M L T⁻³ K⁻⁷ᐟ²**,
`dudt_cgs2au` ⇒ `dudt` is **M L⁻¹ T⁻³**, `Lambda_cgs2au` ⇒ Λ is **M L⁵ T⁻³**.

**Not analysable from this slice (external dependencies).** `operations.monotonic`,
`operations.find_nearest`, `operations.find_nearest_higher`; `net_coolingcurve.get_dudt`; and the
table objects `cStruc_cooling_CIE_interpolation`, `cStruc_cooling_nonCIE`, `cStruc_heating_nonCIE`.
Where a conclusion depends on their contract I say so explicitly and lower the confidence.

---

## Part 1 — `get_bubbleParams.py`

### 1.1 `delta2dTdt(t, T, delta)` — line 27

Returns `dTdt = (T/t)·δ`. Dimensions K·T⁻¹ if T in K and t in Myr; δ dimensionless. Pure algebra,
no state.

### 1.2 `dTdt2delta(t, T, dTdt)` — line 47

Returns `δ = (t/T)·dTdt`. Exact inverse of 1.1. Divide-by-zero if `T == 0` (unguarded).

### 1.3 `cool_beta_to_Ebdot(params)` — line 69

**Reads** (all via `params[k].value`): `Pb`, `cool_beta`, `t_now`, `R1`, `R2`, `v2`, `Eb`,
`pdot_total`, `pdotdot_total`. **Writes** nothing. Returns a scalar.

Local substitutions:

```
Ṗb  = −Pb·β / t
a   = 1.5 · p̈ / ṗ                     [T⁻¹]
c   = 0.75 · ṗ · R1                    [M L² T⁻²]  (energy)
d   = R2³ − R1³                        [L³]
cf  = c / (Eb + c)                     [1]
```

The evaluated expression, fully substituted:

$$
\dot E_b \;=\;
\frac{2\pi\,\dot P_b\,d^{2} \;+\; 3\,E_b\,v_2\,R_2^{2}\,(1-c_f)\;-\;a\,R_1^{3}\,E_b^{2}/(E_b+c)}
     {d\,(1-c_f)}
\;=\;
\frac{2\pi \dot P_b d}{1-c_f}\;+\;\frac{3E_bv_2R_2^{2}}{d}\;-\;\frac{a R_1^{3}E_b^{2}}{d(E_b+c)(1-c_f)}
$$

**Dimensions balance**: each numerator term is M L⁵ T⁻³, denominator L³ ⇒ result M L² T⁻³ = L/T ✓.

**I re-derived this independently and it is algebraically self-consistent**, under two implicit
identities: (i) $P_b = E_b/(2\pi d)$, and (ii) $R_1^2 E_b = k\,d$ with $k \equiv L_{\rm mech}/v_{\rm mech}
= \dot p/2$ (which makes $c = 0.75\,\dot p R_1 = 1.5\,k R_1 = 1.5 R_1^3 E_b/d$ and $a = 1.5\,\dot k/k$).
Differentiating $E_b = 2\pi P_b d$ and eliminating $\dot R_1$ via (ii) reproduces the code exactly —
including the non-obvious cancellation whereby the $-4.5 E_b R_1 k R_2^2 v_2 / (d(E_b+c))$ term combines
with $3E_bv_2R_2^2/d\,(1-c_f)^{-1}$ to leave exactly $3E_bv_2R_2^2/d$. **That algebra is correct.**

**But identity (i) is only true for γ = 5/3.** `bubble_E2P` computes
$P_b = 3(\gamma-1)E_b/\big(4\pi(r_2^3-r_1^3)\big)$, so $E_b = 4\pi d P_b/(3(\gamma-1))$, which equals
$2\pi P_b d$ **iff** $\gamma - 1 = 2/3$. The literal `2 * np.pi` at line 130 therefore hard-codes
γ = 5/3 while `gamma_adia` is a free parameter elsewhere. → **S7-A-02**.

Degeneracies with no guard: `pdot_total == 0` (in `a`), `Eb + c == 0`, `d == 0` (R2 == R1),
`1 − c_f == 0` (only if `Eb == 0` with `c ≠ 0`).

### 1.4 `Ebdot_to_cool_beta(bubble_P, r1, bubble_Edot, my_params)` — line 140

Note the **different container convention**: this reads `my_params["t_now"]` etc. as *plain values*
(no `.value`), unlike 1.3. Same `a`, `c`, `d`, `c_f` definitions but with the **argument** `r1` in
place of `params['R1']`.

$$
\dot P_b=\frac{d(1-c_f)\dot E_b-3E_bv_2R_2^{2}(1-c_f)+a\,r_1^{3}E_b^{2}/(E_b+c)}{2\pi d^{2}},
\qquad \beta = -\dot P_b\,t/P
$$

This is the exact algebraic inverse of 1.3 (solve the 1.3 numerator for `Ṗb`, then invert
`Ṗb = −Pb β/t`) ✓. It inherits the same γ = 5/3 assumption.

### 1.5 `bubble_E2P(Eb, r2, r1, gamma)` — line 198

```
r1 → r1·pc2cm ;  r2 → r2·pc2cm ;  Eb → Eb·E_au2cgs
r2 → r2 + 1e-10                                   # 1e-10 cm, added AFTER unit conversion
V   = r2³ − r1³
if V <= 0:  V = 1e-13 · r2³                        # ← silent replacement
Pb  = (γ−1)·Eb / V / (4π/3)          [erg cm⁻³ = dyn cm⁻²]
return Pb · Pb_cgs2au
```

Dimensions ✓. Numeric literals: `1e-10` (cm offset ≈ 3.2×10⁻²⁹ pc, numerically irrelevant except
in the degenerate r2 = r1 = 0 case where it makes V = 10⁻³⁰ cm³ and Pb ≈ 10³⁰·(γ−1)Eb instead of an
error); `1e-13` (the volume floor). The floor at line 236 means **r2 ≤ r1 does not raise — it
returns a pressure inflated by ~1/(1e-13·…)**. → **S7-A-01**.

`r1 *= …`, `r2 *= …`, `Eb *= …` mutate the *names*, which is safe for Python floats but would
mutate a caller's array in place if a numpy array were ever passed. → **S7-A-22**.

### 1.6 `get_leak_luminosity(coverFraction, R2, Pb, c_sound, gamma)` — line 242

```
if coverFraction >= 1.0 or Pb <= 0.0 or c_sound <= 0.0:   return 0.0
return γ/(γ−1) · (1 − coverFraction) · 4π R2² Pb c_sound
```
Dimensions: (M L⁻¹T⁻²)(L²)(L T⁻¹) = M L²T⁻³ = luminosity ✓ (an enthalpy flux
$\tfrac{\gamma}{\gamma-1}P\,c_s A$). `coverFraction < 0` is **not** clamped ⇒ factor > 1. Bare
literals `1.0`, `0.0`, `4.0`. → **S7-A-25**.

### 1.7 `pRam(r, Lmech, v_mech)` — line 286

`= L_mech/(2π r² v_mech)`. Dimensions M L⁻¹T⁻² ✓. Equivalent to $\dot p/(4\pi r^2)$ under
$\dot p = 2L/v$ — i.e. the same $L/v = \dot p/2$ convention as 1.3. `v_mech = 0` ⇒ inf, unguarded.

### 1.8 `get_effective_bubble_pressure(...)` — line 311

Three branches, each returning a different expression:

* `current_phase == 'momentum'` → `pRam(R2, Lmech_total, v_mech_total)`. If the (defaulted-`None`)
  `Lmech_total`/`v_mech_total` are not supplied this is a `TypeError`, not a handled case.
* `'transition'` → `max(bubble_E2P(Eb,R2,R1,γ), pRam(R2,L,v))` — a **max-clamp**: the thermal
  pressure is silently replaced by the ram pressure whenever the latter is larger.
* otherwise (energy phase) → with `dt_switchon = tmin = 1e-3` (Myr):
  if `t is not None and tSF is not None and t <= tmin + tSF`:
  `R1_tmp = (t − tSF)/1e-3 · R1`, return `bubble_E2P(Eb, R2, R1_tmp, γ)`; else
  `bubble_E2P(Eb, R2, R1, γ)`.
  This linearly ramps the *inner* radius from 0 to its full value over the first 10⁻³ Myr after
  `tSF`. For `t < tSF` the ramp goes **negative**, so `r1³ < 0` and the shell volume `r2³−r1³`
  is *larger* than geometric ⇒ Pb below the un-ramped value. → **S7-A-27**.

### 1.9 `get_r1(r1, params)` — line 384 and `solve_R1(...)` — line 414

`params` is unpacked positionally as `[Lmech_total, Ebubble, v_mech_total, r2]`; the caller at
line 447 passes `[Lmech_total, Eb, v_mech_total, R2]` ✓ order matches.

Residual: $f(r_1)=\sqrt{\dfrac{L_{\rm mech}}{v_{\rm mech}E_b}\,(r_2^3-r_1^3)}-r_1$, with the clamp
`if Ebubble < 1e-30: Ebubble = 1e-30` (also catches **negative** Eb, silently mapping it to +1e-30
and hence driving R1 → R2).

`solve_R1` control flow:
* `Lmech_total <= 0` → **return 0.0** (no root find).
* `not (R2 > 0)` → **return 0.0** (this also swallows `R2 = NaN`).
* any of Eb/Lmech/v_mech non-finite → **raise ValueError**.
* else `brentq(get_r1, 0.0, R2)`; bracket is valid because f(0) = √(k r2³/Eb) > 0 and f(R2) = −R2 < 0.
* `except (ValueError, RuntimeError)` → log then **re-raise**.

Inconsistency: a non-finite `R2` returns 0.0 silently, a non-finite `Eb` raises. And a returned
`R1 = 0.0` is fatal downstream — `_create_radius_grid` takes `np.log10(R1)`. → **S7-A-23**.

---

## Part 2 — `bubble_luminosity.py`

### 2.1 Module-level constants (lines 37–116)

| name | value | where used |
|---|---|---|
| `_trapezoid` | `np.trapezoid` or `np.trapz` | all luminosity/Tavg integrals |
| `MIN_SPACING` | `1e-12` | default rel. spacing in `_clean_radius_grid` |
| `_T_INIT_BOUNDARY` | `3e4` (K) | outer boundary T **and** the residual rejection floor |
| `_MINT_LOG_TOL` | `1.0` (K) | gates a debug log only |
| `_T_INTERFACE_BAND` | `10**5.5` ≈ 3.162e5 K | fA cut in the ODE RHS |
| `_SOLVER_FAIL_RESIDUAL` | `1e3` | three failure returns in the residual |
| `_BUBBLE_RTOL`/`_BUBBLE_ATOL` | `1e-8`/`1e-10` | final structure solve |
| `_RESIDUAL_RTOL` | `1e-6` | dMdt-search solve |
| `_RESIDUAL_NPTS` | `500` | `t_eval` for the dMdt-search solve |
| `_CONDUCTION_NPTS` | `2000` | conduction-zone re-sampling grid |

`_quiet_lsoda_fortran` (119) dup2-redirects fds 1 and 2 to `/dev/null` around the solve and restores
in `finally` — process-global, so it also silences anything else writing to those fds concurrently.

### 2.2 `get_bubbleproperties_pure(params)` — line 199

**Reads**: `R2`, `Eb`, `Lmech_total`, `v_mech_total`, `gamma_adia`, `bubble_dMdt`, `bubble_xi_Tb`
(+ everything the callees read). **Writes**: nothing into `params`; returns a `BubbleProperties`.

Sequence:
1. `R1 = solve_R1(R2, Eb, Lmech_total, v_mech_total)`
2. `Pb = bubble_E2P(Eb, R2, R1, gamma_adia)`
3. `dMdt = params['bubble_dMdt']`; **if NaN** → `_get_init_dMdt(params, Pb)` (analytic Weaver guess).
4. `r_Tb = R1 + ξ·(R2 − R1)`; `assert r_Tb > R1` (the only validation of ξ; removed under `python -O`).
5. `dMdt = fsolve(residual, dMdt, xtol=1e-4, factor=50, epsfcn=1e-4)[0]` — **`full_output` is not
   requested and `ier` is never inspected**, so a non-converged fsolve is used silently. → **S7-A-06**
6. Recompute the ICs at the converged `dMdt`; `initial_conditions = [v, T, dTdr]`.
7. `return _bubble_luminosity(params, R1, Pb, r2Prime, initial_conditions, r_Tb, dMdt)`

### 2.3 `_get_init_dMdt(params, Pb)` — line 297

```
dMdt = (12/75)·1.646^(5/2)·4π R2³/t · (μ_ion/k_B) · (t·κ·C/R2²)^(2/7) · Pb^(5/7)
```
with `C_thermal` pre-multiplied by `cooling_boost_kappa` (line 304).

**Dimensions** (AU): L³T⁻¹ · (K T²L⁻²) · (M^{2/7}L^{−2/7}T^{−4/7}K^{−1}) · (M^{5/7}L^{−5/7}T^{−10/7})
= M T⁻¹ ✓ (M☉/Myr).

**Algebraic identity I verified**: 12/75 = 4/25, so the prefactor is 16π/25, and the expression is
identically
$$\dot M = \frac{16\pi}{25}\frac{\mu C}{k_B}R_2\,T_c^{5/2},\qquad T_c = 1.646\left(\frac{P_bR_2^2}{Ct}\right)^{2/7}$$
i.e. exactly the same $\dot M(T)$ relation that `_get_bubble_ODE_initial_conditions` inverts, closed
with a similarity estimate of the central temperature. Self-consistent. The literals are `1.646`,
`12/75`, `5/2`, `2/7`, `5/7`, `4π`.

### 2.4 `_get_velocity_residuals(dMdt_init, params, Pb, R1)` — line 311 — **the outer iteration**

1. Build ICs at `dMdt_init` (which fsolve passes as a shape-(1,) array; `.item()` at 321–324).
2. `if not all finite([v,T,dTdr])` → **return +1e3**. Note **`r2Prime` finiteness is not checked**.
3. `solve_ivp(_rhs, t_span=(r2Prime, R1), y0=[v,T,dTdr], method='LSODA',
   t_eval=linspace(r2Prime, R1, 500), rtol=1e-6, atol=1e-10)` — **integration runs from the outer
   boundary `r2Prime` inward to `R1`** (a *decreasing* independent variable).
4. `_rhs` traps `BubbleSolverError`, latches it in `rhs_error`, and returns **zeros** from then on.
5. Failure ladder (each replaces the residual entirely):
   * exception → +1e3 · `rhs_error` set → +1e3 · `not sol.success` → +1e3
6. `residual = (v[-1] − 0) / (v[0] + 1e-4)` — the target BC is v(R1) = 0, normalised by v at the
   outer boundary plus a 1e-4 pc/Myr offset.
7. `min_T = min(T)`. **If `min_T < 3e4`** → return `residual · (3e4/(min_T + 1e-1))²`
   (a *multiplicative* penalty ≥ 1 that **preserves sign**; log-gated by `min_T < 3e4 − 1.0`).
   This early return **skips the monotonicity check below**.
8. `if isnan(min_T)` → **return −1e3** (reachable: `nan < 3e4` is False, so control does fall
   through to here).
9. **`if not operations.monotonic(T_array)` → return +1e2** — this is the monotonicity guard
   named in the project conventions. It tests the **temperature array of the 500-point,
   rtol = 1e-6 search solve**, and on firing it **replaces the residual by the constant +1e2**,
   destroying both magnitude and sign information. → **S7-A-07**
10. else return `residual`.

Sign map of the sentinels: +1e3 (three failure modes), +1e2 (non-monotonic), −1e3 (NaN T). Two
adjacent failure regions of opposite sign manufacture a spurious sign change. → **S7-A-08**.
The `v[0] + 1e-4` denominator is a **pole**: as v(r2Prime) → −10⁻⁴ pc/Myr the residual diverges and
flips sign with no physical content. → **S7-A-09**.

### 2.5 `_get_bubble_ODE_initial_conditions(dMdt, params, Pb, R1)` — line 392 — **boundary conditions**

`R1` is accepted and **never used** → **S7-A-03**.

```
T_init   = 3e4
K        = (25/4)·k_B/(μ_ion·κ·C_thermal)          [L T K^{5/2} M⁻¹]
dR2      = T_init^{5/2} / ( K·Ṁ/(4πR2²) )          [L]
T        = ( K·Ṁ·dR2/(4πR2²) )^{2/5}               ≡ T_init  (exact round-trip)
v        = α R2/t − (Ṁ/(4πR2²))·k_B T/(μ_ion Pb)
dTdr     = −(2/5)·T/dR2
r2_prime = R2 − dR2
```

Dimensions all check out (dR2 → L, T → K, v → L T⁻¹, dTdr → K L⁻¹).

* `T` at line 404 is **algebraically identical to `_T_INIT_BOUNDARY`** by construction (substitute
  dR2): a degenerate re-derivation. → **S7-A-04**
* The second term of `v` is $\dot M/(4\pi R_2^2\rho)$ with $\rho = \mu_{\rm ion}P_b/(k_BT)$ — the
  evaporative inflow speed. Consistent with the ODE's density convention (§2.6).
* $dR2 = \tfrac{4}{25}\tfrac{\mu C}{k_B}\tfrac{4\pi R_2^2 T^{5/2}}{\dot M}$: **`dMdt → 0` sends
  dR2 → ∞ and `r2_prime` → −∞**, while `v, T, dTdr` all stay finite so the finiteness screen in
  §2.4 step 2 does not fire; `t_span=(−inf, R1)` / `linspace(−inf, …)` then reaches `solve_ivp`.
  There is no guard `0 < r2_prime` nor `r2_prime > R1`. → **S7-A-05**
* Negative `dMdt` (fsolve is unconstrained) gives `dR2 < 0`, `r2_prime > R2`, `dTdr > 0` — finite
  but geometrically inverted; no guard.

**What happens at the singular endpoint.** The Weaver-type similarity solution has T → 0 at r = R2.
The code never integrates to it: it *excises* the singularity by starting at
`r2Prime = R2 − dR2`, where dR2 is chosen precisely so that T(r2Prime) = 3×10⁴ K, and the local
2/5-power law fixes `dT/dr = −(2/5)T/dR2` there. So the outer boundary is a **movable offset whose
size is set by the unknown Ṁ** — the boundary location is part of what the outer fsolve is solving for.

### 2.6 `_get_bubble_ODE(r, y, params, Pb)` — line 414 — **the structure ODE**

`y = [v, T, dTdr]`, returns `[dvdr, dTdr, dTdrr]` (ordering ✓).

**Guard**: `if abs(T − 0) < 1e-5: raise BubbleSolverError`. Note this is a **two-sided** test on |T|,
so a *negative* T with |T| > 1e-5 passes straight through into `T**(5/2)` → NaN.

```
n     = Pb / ( (μ_conv/μ_ion) · k_B · T )                  [L⁻³]
φ     = Qi / (4π r²)                                       [T⁻¹L⁻²]
dudt  = net_coolingcurve.get_dudt(t, n, T, φ, params)      [M L⁻¹T⁻³]  (AU-unit arguments)
if fA != 1.0 and T < 10^5.5:  dudt = fA·dudt               # boost applied ONLY below 10^5.5 K
v_term = α r / t
```

$$
\frac{dv}{dr}=\frac{\beta+\delta}{t}+\big(v-\tfrac{\alpha r}{t}\big)\frac{T'}{T}-\frac{2v}{r}
$$

$$
\frac{d^2T}{dr^2}=\frac{P_b}{\kappa C\,T^{5/2}}\left[\frac{\beta+2.5\,\delta}{t}
+2.5\big(v-\tfrac{\alpha r}{t}\big)\frac{T'}{T}-\frac{\dot u}{P_b}\right]
-\frac{2.5\,T'^2}{T}-\frac{2T'}{r}
$$

**Dimensions**: `dvdr` → T⁻¹ ✓ (all three terms). `dTdrr`: `Pb/(C T^{5/2})` = L⁻²T·K, each bracket
term is T⁻¹ ⇒ K L⁻² ✓; `2.5 T'²/T` = K L⁻² ✓; `2T'/r` = K L⁻² ✓. **Fully balanced.**

**I re-derived this system from scratch** and it is exactly the isobaric, self-similar Weaver
interior with $P\propto t^{-\beta}$, $T_c\propto t^{\delta}$, $R_2\propto t^{\alpha}$:
continuity with $\rho=\mu P/(kT)$ gives $v'+2v/r = \beta/t + \dot T/T + vT'/T$, and the ansatz
$\partial T/\partial t|_r = \delta T/t - (\alpha r/t)T'$ reproduces the code's `dvdr` term-for-term;
substituting that into $\tfrac32\partial_tP + \tfrac52 P(v'+2v/r) = \nabla\!\cdot\!(CT^{5/2}\nabla T)+\dot u$
yields **precisely** the `β + 2.5δ` combination and the `−2.5T'²/T − 2T'/r` expansion terms, with
`dudt` entering as **net heating** (heating − cooling), sign ✓. **This is the cleanest part of the
slice — no coefficient, sign or exponent discrepancy found in the ODE.**

Numeric literals: `1e-5`, `4π`, `2.5` (×3), `5/2`, `2` (×2).

**Latched-error hazard**: `rhs_error` (both here-callers, lines 337–345 and 490–498) is a
write-once latch. LSODA evaluates the RHS at *trial* points that may be rejected; a single rejected
trial excursion to |T| < 1e-5 permanently poisons the run (the RHS returns zeros thereafter and the
whole solve is declared failed). → **S7-A-10**

### 2.7 `_solve_bubble_structure(...)` — line 452

* Non-finite ICs → `psoln = NaN(len(r),3)`, `ok=False`, message, `sol=None`.
* `solve_ivp(..., t_span=(r_array[0], r_array[-1]), method='LSODA', dense_output=True,
  rtol=1e-8, atol=1e-10)` — note **no `t_eval`**; the requested grid is obtained afterwards by
  `psoln = sol.sol(r_array).T` (dense-output evaluation).
* `infodict = {message, status, nfev, nst=sol.t.size, hu=|diff(sol.t)|}` — **there is no `'ier'`
  key**, yet `_capture_bubble_integration` looks one up. → **S7-A-14**
* Returns `(psoln, sol.success, infodict, sol)`.

### 2.8 `_create_radius_grid(R1, r2Prime)` — line 531 and `_clean_radius_grid` — line 570

```
r  = (r2Prime + R1) − logspace(log10(R1), log10(r2Prime), 20000)
```
i.e. a logspace between R1 and r2Prime **reflected about their sum**, giving a *strictly decreasing*
array from `r[0] = r2Prime` to `r[-1] = R1` with points clustered at the **outer** end (where the
conduction front is). Then two refinements:

* `r_improve = logspace(log10(r[0]), log10(r[2]), 20000)` (decreasing); `r = insert(r[3:], 0, r_improve)`
  — refines the first three-point interval.
* `r_further = (r[-1] + r[-5]) − logspace(log10(r[-1]), log10(r[-5]), 20000)`;
  `r = insert(r[:-5], len(r[:-5]), r_further)` — refines the last five-point interval; the same
  reflection means the added points cluster at the **upper** (larger-radius) end of that interval,
  not at R1.

Result ≈ 6×10⁴ points, still strictly decreasing, no duplicates at the seams.
`int(2e4)` appears three times; indices `2, 3, 5` are bare.

**`R1 = 0` (a documented return of `solve_R1`) makes `np.log10(R1) = −inf`, and
`np.linspace(-inf, x, N)` produces NaNs** ⇒ the whole grid is NaN. Likewise `r2Prime <= 0`
(reachable via small dMdt, §2.5). → **S7-A-05 / S7-A-23**

`_clean_radius_grid` drops point *i+1* when
`|r[i+1] − r[i]| / (0.5(|r[i]|+|r[i+1]|)) < 1e-12`, with the denominator floored at `1e-30`. The mask
is computed from the **original** consecutive differences, not the distance to the last *kept*
point, so a run of many sub-tolerance steps can still leave near-duplicates relative to what
survives. → **S7-A-21**

### 2.9 `_bubble_luminosity(...)` — line 625 — **the main computation**

**Reads** from `params`: `mu_convert`, `mu_ion`, `k_B`, `Qi`, `chi_e`, `cooling_boost_fA`,
`cStruc_cooling_CIE_interpolation`, `cStruc_cooling_nonCIE`, `cStruc_heating_nonCIE`, `path2output`
(diagnostics), plus whatever the ODE reads. Writes nothing.

#### (a) Structure solve

`r_array = _create_radius_grid(R1, r2Prime)` (decreasing r2Prime → R1); integrate with rtol = **1e-8**
(vs 1e-6 in the dMdt search) over ~6×10⁴ dense-output points. `if not _ok: raise BubbleSolverError`.
`if np.any(T_array < 0): raise` (NaN is *not* caught by this test).

`n_array = Pb / ((μ_conv/μ_ion)·k_B·T)`. Combined with `ρ = n·μ_conv` in `_get_mass_and_grav`, this
is internally consistent: `ρ = μ_ion Pb/(k_B T)` (pressure uses the mean particle mass μ_ion; the
tabulated number density `n` counts particles of mass μ_conv, e.g. H nuclei), and it matches the
`v` boundary condition of §2.5 ✓.

#### (b) Zone splitting

```
_coolingswitch = 1e4 ;  _CIEswitch = 10**5.5
index_CIE_switch     = find_nearest_higher(T_array, 10^5.5)
index_cooling_switch = find_nearest_higher(T_array, 1e4)
```

T increases with index (index 0 = outer edge at 3×10⁴ K, last index = R1, hottest). **Because the
outer boundary temperature `_T_INIT_BOUNDARY = 3e4` already exceeds `_coolingswitch = 1e4`, no
element of `T_array` is ever below 1e4**, so `index_cooling_switch` is 0 in every ordinary run and
`index_cooling_switch != index_CIE_switch` is essentially always True. The two `else` arms at
lines 866–870 and 880–881 are therefore effectively dead. → **S7-A-15** (the inner one is
*provably* unreachable, see below).

If `ics != iCIE` (the normal case), the exact 10^5.5 crossing is located and **inserted** into all
five arrays:
* `r_interp = r_array[:iCIE+20]` (`_xtra = 20`), `fT_interp = interp1d(r_interp, T[:iCIE+20] − 10^5.5,
  kind='cubic')`, `fdTdr_interp`/`fv_interp` are **linear** (a kind mismatch between the root-finding
  interpolant and the ones used to evaluate at the root).
* `r_CIEswitch = brentq(fT_interp, min(r_interp), max(r_interp), xtol=1e-8)` — **unguarded**: this
  requires a sign change over the first `iCIE+20` points. If the profile never reaches 10^5.5 (a
  cool bubble) the sign change does not exist and `brentq` raises `ValueError` out of the whole
  call. What `find_nearest_higher` returns in that case is outside my slice. → **S7-A-17**
* `n_CIEswitch = Pb/((μ_conv/μ_ion)k_B·10^5.5)` — consistent with the inserted T ✓.
* `np.insert(..., index_CIE_switch, ...)` on T, r, n, dTdr, v. Monotonicity of both arrays is
  preserved. Because `ics ≤ iCIE` always, the insertion never invalidates `index_cooling_switch` —
  but it is a latent index-invalidation pattern (indices computed pre-insert, used post-insert).

Zone definitions after insertion:

| zone | radial span | T span | integrand |
|---|---|---|---|
| "bubble" | `r_array[iCIE:]` = [R1, r_CIEswitch] | ≥ 10^5.5 | CIE, cooling only |
| "conduction" | linspace(r_array[0], r_array[iCIE], 2000), masked `T < 10^5.5` | 3e4 … 10^5.5 | non-CIE net |
| "intermediate" | linspace(r_array[ics], R2_coolingswitch, 1000) | 1e4 … 3e4 | non-CIE net (+ dead CIE arm) |

These tile [R1, R2_coolingswitch] contiguously, no overlap, no gap ✓.

#### (c) L_bubble (lines 742–750)

$$
L_{\rm bubble}=\Bigg|\int_{r_{\rm CIE}}^{R_1}\chi_e\,n(r)^2\,\Lambda_{\rm CIE}\!\big(T(r)\big)\,4\pi r^2\,dr\Bigg|,
\qquad \Lambda_{\rm CIE}=10^{\,f_{\rm CIE}(\log_{10}T)}\cdot\texttt{Lambda\_cgs2au}
$$

The lookup is **1-D in log₁₀T only** (no density or φ dependence), the interpolator returns log₁₀Λ in
cgs, and the result is converted to AU. Off-table behaviour is a property of the (external)
interpolator object — nothing here clamps `log10(T_bubble)` to the table range.
Dimensions: [1]·L⁻⁶·(M L⁵T⁻³)·L³ = M L²T⁻³ ✓. `np.abs` absorbs the reversed integration direction
(r_bubble is decreasing).

`Tavg_bubble = |∫ r²T dr|` over the same span (K·L³).

#### (d) L_conduction (lines 757–797) — only when `ics != iCIE`

`r_conduction = linspace(r_array[0], r_array[iCIE], 2000)`, re-evaluated from the **dense output**
`_sol.sol(r_conduction)` (not from `psoln`), masked to `T < 10^5.5`.

```
n_cond   = Pb/((μ_conv/μ_ion)k_B T_cond)
φ_cond   = Qi/(4π r²)
X        = transpose(log10([ n_cond/ndens_cgs2au , T_cond , φ_cond/phi_cgs2au ]))   # → (N,3), cgs
dudt     = (10^heat_nonCIE.interp(X) − 10^cool_nonCIE.interp(X)) · dudt_cgs2au
L_cond   = |∫ dudt · 4π r² dr|
```
Unit handling ✓ (AU → cgs before log₁₀, cgs → AU after). Dimensions M L⁻¹T⁻³·L³ = M L²T⁻³ ✓.

Two structural observations:
* **`L_bubble` integrates a strictly-positive *cooling* rate; `L_conduction` and the non-CIE arm of
  `L_intermediate` integrate the *net* (heating − cooling) and then take `abs()`.** A net-heating
  sub-region cancels against cooling inside the integral, and if the *total* comes out net-heating
  the `abs()` re-signs it as a loss. Sibling terms of the same sum use opposite sign conventions.
  → **S7-A-11**
* The ODE RHS obtains its `dudt` from `net_coolingcurve.get_dudt(...)` with **AU-unit** arguments,
  while this block hits the tables directly with **cgs** arguments. Two independent code paths
  compute the same physical quantity for the same zone; nothing here enforces that they agree, and
  `L_bubble` (T ≥ 10^5.5) drops the heating term entirely. → **S7-A-30**

`dTdR_coolingswitch = dTdr_cond[0] if len(dTdr_cond) > 0 else dTdr_bubble[0]` — note this is the
gradient at `r_conduction[0] = r_array[0] = r2Prime`, i.e. at the **outer** boundary, which is only
the right partner for line 801 because `index_cooling_switch == 0`.

#### (e) The intermediate (extrapolated) zone, lines 801–837

```
R2_coolingswitch = (1e4 − T_array[ics]) / dTdR_coolingswitch + r_array[ics]
```
a **linear extrapolation** of the temperature profile outward until it reaches 10⁴ K, followed by a
2-point `interp1d` (linear) and a 1000-point `linspace` between the two.

Substituting the actual values (`ics = 0`, `T_array[0] = T_init = 3e4` exactly because it is the
initial condition, `dTdr_cond[0] = −(2/5)T_init/dR2` likewise):

$$
R_{2,\rm cool}= \frac{10^4-3\times10^4}{-\tfrac{2}{5}\cdot 3\times10^4/dR_2}+ (R_2-dR_2)
= \tfrac{5}{3}dR_2 + R_2 - dR_2 = \boxed{R_2 + \tfrac{2}{3}dR_2}
$$

So the "intermediate" zone **always extends 2/3 of a conduction-front thickness beyond R₂**, and its
density is obtained by continuing the *bubble* pressure `Pb` isobarically past R₂:
`n_interm = Pb/((μ_conv/μ_ion)k_B T_interm)`, giving n up to 3× the boundary value at the 10⁴ K end.
Its luminosity and its volume both enter `L_total` and `Tavg`. → **S7-A-13**

`L_intermediate` loops over `['non-CIE','CIE']` with masks `T_interm < 10^5.5` / `≥ 10^5.5`. Since
`T_interm` spans only [10⁴, T_array[ics]] = [10⁴, 3×10⁴] in the normal regime, **the CIE arm's mask
is always empty and the arm `continue`s** — dead in every ordinary run. → part of **S7-A-12**

Division by `dTdR_coolingswitch` at line 801 is unguarded: a zero gradient gives ±inf (→ NaN grid),
and a *positive* gradient inverts `R2_coolingswitch < r_array[ics]`, making `r_interm` run **inward**
and overlap the conduction zone (double counting, with `abs()` hiding the reversal). → **S7-A-19**

#### (f) fA boost and the total, lines 845–851

```
if fA != 1.0:  L_conduction *= fA ;  L_intermediate *= fA
L_total = L_bubble + L_conduction + L_intermediate
```
`L_bubble` is deliberately not boosted (it is the T ≥ 10^5.5 zone, matching the ODE's
`T < _T_INTERFACE_BAND` condition), but **`L_intermediate` is boosted in full including its CIE
(T ≥ 10^5.5) arm** — inconsistent with the ODE rule, though currently unreachable. → **S7-A-12**

#### (g) Volume-weighted mean temperature, lines 860–870

$$
\bar T = \frac{3\left(\big|\int_{\rm bub} r^2T\,dr\big| + \big|\int_{\rm cond} r^2T\,dr\big| + \big|\int_{\rm int} r^2T\,dr\big|\right)}
{\big|r_{b,0}^3-r_{b,-1}^3\big| + \big|r_{c,0}^3-r_{c,-1}^3\big| + \big|r_{i,0}^3-r_{i,-1}^3\big|}
$$
The factor 3 with no 4π is correct (the 4π cancels between $4\pi\!\int r^2T\,dr$ and
$\tfrac{4\pi}{3}\Delta r^3$) ✓. The `else` arm drops the conduction terms. `r_conduction[0]`/`[-1]`
at lines 862–863 are used **without** the `len(...) > 0` guard that line 776 applies to the same
(post-mask) array. → **S7-A-18**

#### (h) T at the probe radius, lines 874–885

```
if r_Tb > r_array[ics]:                 T_rgoal = fT_interp_interm(r_Tb)          # linear, in the extrapolated zone
elif r_Tb > r_array[iCIE]:
    if ics != iCIE:                     T_rgoal = T_cond[i] + dTdr_cond[i]·(r_Tb − r_conduction[i])   # 1st-order Taylor from nearest node
    else:                               T_rgoal = T_bubble[0]                     # ← UNREACHABLE
else:                                   T_rgoal = T_bubble[i] + dTdr_bubble[i]·(r_Tb − r_bubble[i])
```
The inner `else` can only run when `ics == iCIE`, but then line 876's test is **the same expression**
as line 874's, which was already False — so it is provably dead. → **S7-A-15**

`fT_interp_interm` has `bounds_error=True` (scipy default), but since `r_Tb = R1 + ξ(R2−R1) ≤ R2` and
`R2_coolingswitch = R2 + (2/3)dR2 > R2`, the call cannot go out of range for ξ ≤ 1 ✓ (it *would*
raise for ξ > 1).

#### (i) Mass, lines 891–892 → `_get_mass_and_grav` (line 915)

```
r_new = r[::-1]                       # a reversed VIEW (increasing radius)
rho   = n[::-1] · μ_convert
m(r)  = 4π · cumulative_trapezoid(rho·r², x=r_new, initial=0)
mBubble = m[-1]
```
Dimensions M ✓. It integrates over the post-insertion `r_array`/`n_array` (lengths match ✓) and
therefore covers [R1, r2Prime] only — the intermediate zone's mass is excluded, even though its
volume *is* included in `Tavg`. `grav_phi` and `grav_force_m` are hard-set to `None` and discarded
by the caller. → **S7-A-16**

### 2.10 Diagnostics — `_capture_bubble_integration` (978) and `_dump_bubble_state` (1098)

Both are env-var gated (`TRINITY_BUBBLE_DIAG`, `TRINITY_BUBBLE_STATE_DUMP`) and wrap everything in a
bare `except Exception` that only logs — they cannot change the physics. Notes:

* `strictly_monotonic = (negs.size == 0) or all(diffs ≥ 0) or all(diffs ≤ 0)` — the first clause is
  subsumed by the second, and "strictly" is a misnomer (zero diffs pass).
* `floor = 1e4`; `tail = T[−max(10, n//100):]`; drawdown `(cummax − T)/max(|cummax|, 1e-300)`;
  mode thresholds `1e-2` and `last_bad < 0.01·n`.
* `infodict.get('ier')` is always `None` (§2.7) ⇒ the log always prints `ier=None` and the npz
  always stores `-999`. → **S7-A-14**
* `_dump_bubble_state` reads `t_now` twice (1110 and 1134, identical); throttle rule is
  `t_now < _bubble_state_last_t · dt_factor → skip`, i.e. with the default `dt_factor = 1.0` it
  skips only strictly-decreasing t.

### 2.11 Cross-cutting numerical observation

The dMdt root-find (§2.4) integrates 500 points at **rtol = 1e-6** to a residual tolerance
`xtol = 1e-4` with a finite-difference Jacobian step set by `epsfcn = 1e-4`; the profile that is
actually used for every reported quantity (§2.9) is a **different** integration — ~6×10⁴ points at
**rtol = 1e-8**. Neither `v(R1) = 0` nor monotonicity is re-checked on the profile that is used.
→ **S7-A-28**

### 2.12 Unused / vestigial

`from typing import Optional` (line 26) — `Optional` never appears in the file.
`import astropy.units as u` (get_bubbleParams.py line 15) — `u` never appears.
→ **S7-A-20**

---

## Summary of what the mathematics *is*

* **Integration variable**: radius r [pc]. **Direction**: from the outer boundary
  `r2Prime = R2 − dR2` **inward** to `R1`. **BCs at r2Prime**: T = 3×10⁴ K,
  dT/dr = −(2/5)T/dR2, v = αR₂/t − Ṁk_BT/(4πR₂²μ_ion P_b), with dR2 fixed by Ṁ through
  T^{5/2} = (25/4)(k_B/μC)(Ṁ/4πR₂²)dR2. **The singular endpoint r = R₂ is excised, not resolved.**
* **Closure**: an outer 1-D `fsolve` on Ṁ enforcing v(R1) = 0 (normalised by v(r2Prime) + 10⁻⁴).
* **Interior model**: isobaric at P_b, self-similar with P ∝ t^(−β), T_c ∝ t^δ, R₂ ∝ t^α, Spitzer
  conduction C T^{5/2}. The two ODEs reproduce this system exactly (verified by re-derivation).
* **Luminosity**: a three-zone sum — CIE cooling above 10^5.5 K, tabulated net (heat − cool) between
  3×10⁴ K and 10^5.5 K, and tabulated net over a **linearly extrapolated** 3×10⁴ → 10⁴ K layer that
  by construction reaches R₂ + (2/3)dR₂.

---

```json
[
  {
    "id": "S7-A-01",
    "file": "trinity/bubble_structure/get_bubbleParams.py",
    "line": 236,
    "class": "silent-failure",
    "severity": "S2",
    "claim": "bubble_E2P silently replaces a non-positive shell volume with 1e-13*r2**3 instead of signalling the degenerate geometry, returning a pressure inflated by ~1e13 relative to the r2>r1 case.",
    "evidence": "lines 228-237: `shell_volume = r2**3 - r1**3; if shell_volume <= 0: shell_volume = 1e-13 * r2**3; Pb = (gamma - 1) * Eb / shell_volume / (4*np.pi/3)`. Reached whenever R1 >= R2 (e.g. solve_R1 driving R1 -> R2 as Eb -> 0 via the get_r1 Ebubble floor at line 406).",
    "expected": "Either raise, or return a value derived from the physical limit; a 1e-13 volume floor is an arbitrary magnitude with no continuity to the r2>r1 branch.",
    "failure_scenario": "As Eb collapses, get_r1's `Ebubble = 1e-30` clamp pushes R1 -> R2; the next bubble_E2P call crosses into r2<=r1 and returns Pb ~1e13x too large, which then feeds solve_R1, the ODE (ndens ∝ Pb) and the whole luminosity integral without any diagnostic.",
    "repro": "bubble_E2P(Eb=1.0, r2=1.0, r1=1.0, gamma=5/3) vs bubble_E2P(Eb=1.0, r2=1.0+1e-6, r1=1.0, gamma=5/3): the two differ by ~10 orders of magnitude across an infinitesimal change in r2.",
    "confidence": "high"
  },
  {
    "id": "S7-A-02",
    "file": "trinity/bubble_structure/get_bubbleParams.py",
    "line": 130,
    "class": "coefficient",
    "severity": "S2",
    "claim": "cool_beta_to_Ebdot and Ebdot_to_cool_beta hard-code gamma = 5/3 through the literal 2*np.pi, while bubble_E2P takes gamma as a free argument (params['gamma_adia']).",
    "evidence": "bubble_E2P gives Pb = 3(gamma-1)Eb/(4*pi*(r2^3-r1^3)), i.e. Eb = 4*pi*d*Pb/(3(gamma-1)). I re-derived cool_beta_to_Ebdot by differentiating Eb = 2*pi*Pb*d and eliminating dR1/dt via R1^2*Eb = (Lmech/v_mech)*d; every term matches the code exactly (including the cancellation that leaves 3*Eb*v2*R2^2/d un-divided by (1-c_frac)). The 2*pi prefactor equals 4*pi/(3(gamma-1)) only for gamma = 5/3.",
    "expected": "2*np.pi replaced by 4*np.pi/(3*(gamma-1)) with gamma read from params, or an explicit assertion that gamma_adia == 5/3.",
    "failure_scenario": "A run with gamma_adia != 5/3 makes the Eb<->beta conversions inconsistent with the pressure used everywhere else; the error is a smooth multiplicative factor 3(gamma-1)/2 on the pressure-derivative term, so nothing crashes and the trajectory is silently wrong.",
    "repro": "Set gamma_adia = 1.4 in a .param and compare Ebdot_to_cool_beta(bubble_E2P(Eb,R2,R1,1.4), R1, Edot, p) round-tripped through cool_beta_to_Ebdot; the round trip is exact only at gamma = 5/3.",
    "confidence": "high"
  },
  {
    "id": "S7-A-03",
    "file": "trinity/bubble_structure/bubble_luminosity.py",
    "line": 392,
    "class": "deadcode",
    "severity": "S4",
    "claim": "_get_bubble_ODE_initial_conditions accepts R1 and never uses it.",
    "evidence": "Body (lines 394-411) references only k_B, mu_ion, cooling_boost_kappa, C_thermal, R2, dMdt, Pb, cool_alpha, t_now. All three call sites (lines 275, 316) pass R1.",
    "expected": "Drop the parameter, or use it (e.g. to guard r2_prime > R1).",
    "failure_scenario": "",
    "repro": "",
    "confidence": "high"
  },
  {
    "id": "S7-A-04",
    "file": "trinity/bubble_structure/bubble_luminosity.py",
    "line": 404,
    "class": "other",
    "severity": "S4",
    "claim": "The boundary temperature T recomputed at line 404 is algebraically identical to _T_INIT_BOUNDARY; it is a degenerate round-trip through dR2.",
    "evidence": "line 402 sets dR2 = T_init**(5/2) / (constant*dMdt/(4*pi*R2**2)); line 404 then evaluates (constant*dMdt*dR2/(4*pi*R2**2))**(2/5) = (T_init**(5/2))**(2/5) = T_init exactly.",
    "expected": "Use T_init directly, or make the two-step form deliberate and documented; as written a reader cannot tell that T is pinned at 3e4.",
    "failure_scenario": "",
    "repro": "assert _get_bubble_ODE_initial_conditions(dMdt, p, Pb, R1)[1] == 3e4 for any dMdt > 0.",
    "confidence": "high"
  },
  {
    "id": "S7-A-05",
    "file": "trinity/bubble_structure/bubble_luminosity.py",
    "line": 409,
    "class": "divergence",
    "severity": "S2",
    "claim": "r2_prime = R2 - dR2 has no positivity or ordering guard; dMdt -> 0 sends dR2 -> inf and r2_prime -> -inf, and dMdt < 0 sends r2_prime > R2, in both cases without tripping the finiteness screen in _get_velocity_residuals.",
    "evidence": "line 402 `dR2 = T_init**(5/2) / (constant*dMdt/(4*np.pi*R2**2))` diverges as dMdt -> 0; line 409 `r2_prime = R2 - dR2`. The screen at line 333 checks only [v_init, T_init, dTdr_init], all of which stay finite in that limit (v -> cool_alpha*R2/t_now, T = 3e4, dTdr = -2/5*T/inf = -0). r2Prime_val then reaches solve_ivp's t_span (line 351) and np.linspace (line 354), and reaches np.log10(r2Prime) in _create_radius_grid (line 554).",
    "expected": "Reject or clamp dMdt so that 0 < r2_prime and r2_prime > R1 before building the grid / calling solve_ivp; include r2Prime in the finiteness screen at line 333.",
    "failure_scenario": "fsolve (unconstrained, epsfcn=1e-4) probes a small or negative dMdt during the Jacobian step; solve_ivp raises a non-BubbleSolverError exception (only BubbleSolverError is caught at line 358) which propagates out of get_bubbleproperties_pure, or _create_radius_grid silently returns an all-NaN grid.",
    "repro": "_get_velocity_residuals(np.array([1e-30]), params, Pb, R1) with a normal simple_cluster state.",
    "confidence": "medium"
  },
  {
    "id": "S7-A-06",
    "file": "trinity/bubble_structure/bubble_luminosity.py",
    "line": 261,
    "class": "silent-failure",
    "severity": "S2",
    "claim": "The dMdt root-find never checks whether fsolve converged; a non-converged iterate is used to build the structure that produces every reported quantity.",
    "evidence": "lines 261-267 call scipy.optimize.fsolve(...)[0] without full_output=True and never inspect ier/mesg. The residual function returns constant sentinels (+1e3 at 334/359/361/363, +1e2 at 382, -1e3 at 378) that can never be zero, so on any of those paths fsolve terminates on 'not making good progress' and the last iterate is accepted.",
    "expected": "Request full_output, check ier == 1 and |residual| below a stated bar, and raise BubbleSolverError otherwise.",
    "failure_scenario": "A stiff regime lands the residual permanently in the +1e2 monotonicity plateau; fsolve stalls at the initial guess, and the run continues with an evaporation rate that does not satisfy v(R1)=0 - producing a plausible-looking but unconverged bubble structure with no warning.",
    "repro": "Wrap velocity_residuals_wrapper to log its return value across an fsolve call in a stiff config (docs/dev/performance f1edge_hidens-style parameters) and observe returns of exactly 100.0 or 1000.0.",
    "confidence": "high"
  },
  {
    "id": "S7-A-07",
    "file": "trinity/bubble_structure/bubble_luminosity.py",
    "line": 380,
    "class": "numerical",
    "severity": "S2",
    "claim": "The monotonicity guard replaces the residual with the constant +1e2, discarding both its magnitude and its sign, and it is applied to a coarser solution (500 points, rtol 1e-6) than the one actually used downstream (~6e4 points, rtol 1e-8).",
    "evidence": "lines 380-382 `if not operations.monotonic(T_array): return 1e2`; the T_array here comes from the solve at lines 349-357 with t_eval=linspace(...,_RESIDUAL_NPTS=500) and rtol=_RESIDUAL_RTOL=1e-6, whereas _bubble_luminosity solves at rtol=_BUBBLE_RTOL=1e-8 over _create_radius_grid's ~6e4 points (lines 554-564, 645-646) and applies no monotonicity test at all (only `T_array < 0` at line 668).",
    "expected": "Either return a sign-preserving penalty (as the min_T branch at line 374 does) so the root remains bracketed, or re-check monotonicity on the final profile and reject there.",
    "failure_scenario": "A region of dMdt space where the true residual is negative returns +1e2; the root-finder sees no sign change and a zero-gradient plateau, stalls, and (per S7-A-06) the stalled value is used. Conversely a profile that is monotonic at rtol 1e-6 but not at 1e-8 passes the gate and is never re-tested.",
    "repro": "Instrument _get_velocity_residuals to record (dMdt, return value) and plot; the plateau at exactly 100.0 is visible as a flat band.",
    "confidence": "high"
  },
  {
    "id": "S7-A-08",
    "file": "trinity/bubble_structure/bubble_luminosity.py",
    "line": 378,
    "class": "sign",
    "severity": "S2",
    "claim": "Failure sentinels in _get_velocity_residuals have inconsistent signs: +1e3 for solver failure, +1e2 for non-monotonic, but -1e3 for a NaN temperature. Two adjacent failure regions of opposite sign manufacture a spurious sign change in the residual.",
    "evidence": "line 334/359/361/363 `return _SOLVER_FAIL_RESIDUAL` (= +1e3, line 84); line 378 `return -1e3`; line 382 `return 1e2`.",
    "expected": "One sign convention for all failure returns (all positive or all negative), or an out-of-band failure signal rather than a magic residual value.",
    "failure_scenario": "A bracketing root-finder (or fsolve's line search) sees residual +1e3 on one side and -1e3 on the other and converges onto a 'root' that is entirely an artefact of two adjacent failure modes.",
    "repro": "Scan dMdt across a range that transitions from LSODA failure to NaN-temperature failure and plot the residual; the sign flip appears with no zero crossing of the physical residual.",
    "confidence": "high"
  },
  {
    "id": "S7-A-09",
    "file": "trinity/bubble_structure/bubble_luminosity.py",
    "line": 368,
    "class": "numerical",
    "severity": "S2",
    "claim": "The residual is normalised by (v_array[0] + 1e-4) rather than |v_array[0]| or a fixed scale, introducing a pole and a sign flip at v(r2Prime) = -1e-4 pc/Myr.",
    "evidence": "line 368 `residual = (v_array[-1] - 0) / (v_array[0] + 1e-4)`. v_array[0] is the boundary velocity cool_alpha*R2/t_now - dMdt*k_B*T/(4*pi*R2**2*mu_ion*Pb) (lines 405-407), which decreases monotonically with dMdt and does cross zero.",
    "expected": "Normalise by a strictly positive scale, e.g. max(|v_array[0]|, eps) or cool_alpha*R2/t_now.",
    "failure_scenario": "For dMdt large enough that the evaporative inflow term exceeds the shell expansion term, v(r2Prime) passes through -1e-4 and the residual diverges and flips sign; the root-finder is attracted to the pole rather than to v(R1)=0.",
    "repro": "Evaluate _get_velocity_residuals over a dMdt sweep that drives v_array[0] through -1e-4 and plot.",
    "confidence": "medium"
  },
  {
    "id": "S7-A-10",
    "file": "trinity/bubble_structure/bubble_luminosity.py",
    "line": 339,
    "class": "numerical",
    "severity": "S2",
    "claim": "The rhs_error latch is write-once and never reset, so a single rejected LSODA trial step that momentarily reaches |T| < 1e-5 permanently aborts an otherwise-healthy integration.",
    "evidence": "lines 337-345 and 490-498: `if rhs_error is not None: return np.zeros_like(y)`; the flag is set by the BubbleSolverError raised at line 424 whenever `np.abs(T - 0) < 1e-5`. LSODA evaluates the RHS at predictor/trial states that may be rejected by its own error control, but the latch does not distinguish accepted from rejected evaluations.",
    "expected": "Return a large/penalised derivative (or NaN, letting the solver reject the step) instead of latching, so only an accepted state below the floor terminates the solve.",
    "failure_scenario": "A stiff step near the cold boundary overshoots to T ~ 0 in a trial evaluation that LSODA would have rejected; the RHS then returns zeros for the remainder of the march, the state freezes, and the whole solve is reported as failed (residual 1e3 or BubbleSolverError from _bubble_luminosity).",
    "repro": "Log every RHS call in a stiff run and compare the r at which rhs_error latches against sol.t; a latch at an r not present in sol.t indicates a rejected trial step.",
    "confidence": "medium"
  },
  {
    "id": "S7-A-11",
    "file": "trinity/bubble_structure/bubble_luminosity.py",
    "line": 795,
    "class": "sign",
    "severity": "S2",
    "claim": "Sibling terms of L_total use opposite sign conventions: L_bubble integrates a strictly positive cooling rate, while L_conduction and the non-CIE arm of L_intermediate integrate the net (heating - cooling) and then take abs(), so a net-heating zone is counted as a luminosity loss.",
    "evidence": "line 746 `integrand_bubble = chi_e * n**2 * Lambda * 4*pi*r**2` (Lambda > 0, pure loss) vs line 791-795 `dudt_cond = (heat_cond - cool_cond)*dudt_cgs2au; integrand_cond = dudt_cond*4*pi*r**2; L_conduction = np.abs(_trapezoid(...))` and line 829-835 for the intermediate zone. All three are summed at line 851.",
    "expected": "One convention for all three zones - either all net (and no abs, letting the sign stand) or all pure cooling.",
    "failure_scenario": "In a strongly photo-heated interface (large Qi, high phi) the conduction-zone integral comes out net positive (heating); abs() converts it into an equally large energy sink, so the bubble is drained by a term that physically deposits energy.",
    "repro": "Print the signed value of _trapezoid(integrand_cond, x=r_conduction) alongside L_conduction across a run with a high ionizing luminosity.",
    "confidence": "medium"
  },
  {
    "id": "S7-A-12",
    "file": "trinity/bubble_structure/bubble_luminosity.py",
    "line": 848,
    "class": "regime",
    "severity": "S3",
    "claim": "cooling_boost_fA is applied to the whole of L_intermediate including its CIE (T >= 10^5.5) arm, whereas the ODE RHS applies fA only where T < _T_INTERFACE_BAND = 10^5.5.",
    "evidence": "line 436 `if fA != 1.0 and T < _T_INTERFACE_BAND: dudt = fA*dudt`; lines 845-848 `if fA != 1.0: L_conduction = fA*L_conduction; L_intermediate = fA*L_intermediate` with no temperature condition, while L_intermediate's CIE branch (lines 831-833) covers T >= 10^5.5.",
    "expected": "Apply fA per-regime in the luminosity sum exactly as in the ODE, i.e. only to the T < 10^5.5 contributions.",
    "failure_scenario": "Only reachable when index_cooling_switch == index_CIE_switch, which given T_init = 3e4 > _coolingswitch = 1e4 does not occur in ordinary runs - so today this is a latent inconsistency rather than an active error.",
    "repro": "",
    "confidence": "medium"
  },
  {
    "id": "S7-A-13",
    "file": "trinity/bubble_structure/bubble_luminosity.py",
    "line": 801,
    "class": "regime",
    "severity": "S3",
    "claim": "The 'intermediate' zone is deterministically extrapolated to R2 + (2/3)*dR2, i.e. it always extends beyond the bubble's outer radius R2, and its luminosity and volume both enter L_total and Tavg.",
    "evidence": "line 801 `R2_coolingswitch = (_coolingswitch - T_array[index_cooling_switch]) / dTdR_coolingswitch + r_array[index_cooling_switch]`. In every ordinary run index_cooling_switch = 0 (T_init = 3e4 > _coolingswitch = 1e4), so T_array[0] = 3e4 exactly (it is y0) and dTdR_coolingswitch = dTdr_cond[0] = the initial-condition gradient -(2/5)*3e4/dR2 (line 776 with r_conduction[0] = r_array[0], line 408). Substituting: R2_cs = (1e4-3e4)/(-1.2e4/dR2) + (R2-dR2) = (5/3)dR2 + R2 - dR2 = R2 + (2/3)dR2. The zone's density is the bubble pressure continued isobarically past R2 (line 811).",
    "expected": "If the 3e4 -> 1e4 layer is meant to lie inside R2, clamp R2_coolingswitch <= R2; if it is meant to be the shell interface, that should be an explicit modelling choice rather than a by-product of the boundary-condition gradient.",
    "failure_scenario": "L_intermediate and the Tavg volume systematically include a shell of thickness (2/3)dR2 outside the bubble, at densities up to 3x the boundary value, whose size scales as 1/dMdt.",
    "repro": "Print R2_coolingswitch, params['R2'].value and (r2Prime, dR2) at each call; the ratio (R2_cs - R2)/dR2 should be 2/3 to solver accuracy.",
    "confidence": "high"
  },
  {
    "id": "S7-A-14",
    "file": "trinity/bubble_structure/bubble_luminosity.py",
    "line": 1022,
    "class": "deadcode",
    "severity": "S4",
    "claim": "_capture_bubble_integration looks up an 'ier' key that _solve_bubble_structure never sets, so the diagnostic always reports ier=None and stores -999.",
    "evidence": "infodict is built at lines 521-527 with keys {'message','status','nfev','nst','hu'}; line 1022 does `_ier = infodict.get('ier')` and line 1068 stores `ier=(ier if ier is not None else -999)`. 'ier' is an odeint-era key.",
    "expected": "Read 'status' (the solve_ivp equivalent) instead.",
    "failure_scenario": "",
    "repro": "TRINITY_BUBBLE_DIAG=1 on any run that trips the diagnostic; every saved npz has ier == -999.",
    "confidence": "high"
  },
  {
    "id": "S7-A-15",
    "file": "trinity/bubble_structure/bubble_luminosity.py",
    "line": 881,
    "class": "deadcode",
    "severity": "S3",
    "claim": "The branch `else: T_rgoal = T_bubble[0]` is unreachable: it requires index_cooling_switch == index_CIE_switch, but then the elif test at line 876 is textually the same expression as the if test at line 874 that already evaluated False.",
    "evidence": "line 874 `if bubble_r_Tb > r_array[index_cooling_switch]`, line 876 `elif bubble_r_Tb > r_array[index_CIE_switch]`, line 877 `if index_cooling_switch != index_CIE_switch`, line 880-881 `else: T_rgoal = T_bubble[0]`. When the two indices are equal the elif condition is the negation of a known-False condition.",
    "expected": "Either the elif should compare against a different quantity, or the dead else should be removed; as written the ics == iCIE case falls through to the final else at line 883 and is handled by the r_bubble Taylor expansion.",
    "failure_scenario": "",
    "repro": "Add `raise AssertionError` at line 881 and run the test suite - it will never fire.",
    "confidence": "high"
  },
  {
    "id": "S7-A-16",
    "file": "trinity/bubble_structure/bubble_luminosity.py",
    "line": 947,
    "class": "deadcode",
    "severity": "S4",
    "claim": "_get_mass_and_grav always returns grav_phi = None and grav_force_m = None; the only caller discards both.",
    "evidence": "lines 947-950 `grav_phi = None; grav_force_m = None; return m_cumulative, grav_phi, grav_force_m`; line 891 `m_cumulative, _, _ = _get_mass_and_grav(...)`.",
    "expected": "Return only the cumulative mass, or implement the gravity terms.",
    "failure_scenario": "",
    "repro": "",
    "confidence": "high"
  },
  {
    "id": "S7-A-17",
    "file": "trinity/bubble_structure/bubble_luminosity.py",
    "line": 724,
    "class": "divergence",
    "severity": "S2",
    "claim": "The brentq that locates the 10^5.5 K crossing is unguarded: if the profile never reaches 10^5.5 K there is no sign change over r_interp and brentq raises ValueError out of the whole bubble calculation.",
    "evidence": "line 720 builds fT_interp on `T_array[:index_CIE_switch + 20] - _CIEswitch`; line 724 `scipy.optimize.brentq(fT_interp, np.min(r_interp), np.max(r_interp), xtol=1e-8)` with no try/except and no check that the endpoints straddle zero. The whole block is entered on `index_cooling_switch != index_CIE_switch`, which is True whenever find_nearest_higher returns any index != 0 for the 10^5.5 target - including the degenerate index it may return when no element exceeds 10^5.5.",
    "expected": "Check `T_array[-1] >= _CIEswitch` (or that fT_interp changes sign) before calling brentq, and fall back to the no-CIE-zone path otherwise.",
    "failure_scenario": "A young or heavily-cooled bubble whose peak interior temperature stays below 3.16e5 K aborts the timestep with a raw scipy ValueError rather than a BubbleSolverError.",
    "repro": "Construct a state with a low Eb / high density so that max(T_array) < 10**5.5 and call _bubble_luminosity. (Confirming the exact trigger requires operations.find_nearest_higher's contract, which is outside this slice.)",
    "confidence": "medium"
  },
  {
    "id": "S7-A-18",
    "file": "trinity/bubble_structure/bubble_luminosity.py",
    "line": 863,
    "class": "state",
    "severity": "S3",
    "claim": "r_conduction[0] and r_conduction[-1] are indexed without the emptiness guard that line 776 applies to the same post-mask array.",
    "evidence": "line 776 `dTdR_coolingswitch = dTdr_cond[0] if len(dTdr_cond) > 0 else dTdr_bubble[0]` guards the empty-mask case; line 862-864 `abs(r_conduction[0]**3 - r_conduction[-1]**3)` does not. Both use r_conduction/dTdr_cond after `mask = T_cond < _CIEswitch` at lines 772-775.",
    "expected": "Guard both, or hoist a single `if len(r_conduction) == 0` early exit.",
    "failure_scenario": "If every sampled conduction-zone temperature is >= 10^5.5 the mask empties, line 776 falls back cleanly and then line 863 raises IndexError.",
    "repro": "",
    "confidence": "high"
  },
  {
    "id": "S7-A-19",
    "file": "trinity/bubble_structure/bubble_luminosity.py",
    "line": 801,
    "class": "divergence",
    "severity": "S2",
    "claim": "Division by dTdR_coolingswitch has no zero or sign guard; a zero gradient yields an infinite R2_coolingswitch (NaN grid), and a positive gradient inverts the intermediate zone so it runs inward and overlaps the conduction zone.",
    "evidence": "line 801 `R2_coolingswitch = (_coolingswitch - T_array[index_cooling_switch]) / dTdR_coolingswitch + r_array[index_cooling_switch]`, then line 809 `r_interm = np.linspace(r_array[index_cooling_switch], R2_coolingswitch, num=1000)`. Every downstream integral wraps np.abs (lines 835, 837, 864), so a reversed interval produces a positive contribution rather than an error.",
    "expected": "Require dTdR_coolingswitch < 0 (temperature decreasing outward) and a finite, bounded extrapolation length; otherwise skip the intermediate zone.",
    "failure_scenario": "A locally flat or inverted temperature profile at the outer boundary yields either an all-NaN r_interm (propagating NaN into L_total and Tavg) or an intermediate zone that double-counts radii already covered by the conduction zone.",
    "repro": "",
    "confidence": "medium"
  },
  {
    "id": "S7-A-20",
    "file": "trinity/bubble_structure/bubble_luminosity.py",
    "line": 26,
    "class": "deadcode",
    "severity": "S4",
    "claim": "Unused imports: `Optional` (bubble_luminosity.py:26) and `astropy.units as u` (get_bubbleParams.py:15).",
    "evidence": "`Optional` does not appear anywhere in bubble_luminosity.py after line 26; `u.` does not appear anywhere in get_bubbleParams.py after line 15.",
    "expected": "Remove.",
    "failure_scenario": "",
    "repro": "ruff F401 (not currently in the enabled rule set per CLAUDE.md).",
    "confidence": "high"
  },
  {
    "id": "S7-A-21",
    "file": "trinity/bubble_structure/bubble_luminosity.py",
    "line": 608,
    "class": "numerical",
    "severity": "S4",
    "claim": "_clean_radius_grid decides which points to drop from the ORIGINAL consecutive differences rather than from the distance to the last kept point, so a run of sub-tolerance steps can leave near-duplicates in the cleaned array.",
    "evidence": "lines 601-610: `relative_diff = |diff(r_array)| / avg_magnitude; keep_mask = concatenate([[True], relative_diff >= min_relative_spacing]); cleaned = r_array[keep_mask]`. Dropping r[i+1] does not re-reference r[i+2] against r[i].",
    "expected": "A sequential filter that compares each candidate against the last accepted point.",
    "failure_scenario": "Two surviving points separated by less than MIN_SPACING relative, which is the exact condition the function exists to remove.",
    "repro": "_clean_radius_grid(np.array([1.0, 1.0+5e-13, 1.0+1e-12, 1.0+1.5e-12])) keeps points closer than 1e-12 relative.",
    "confidence": "medium"
  },
  {
    "id": "S7-A-22",
    "file": "trinity/bubble_structure/get_bubbleParams.py",
    "line": 220,
    "class": "state",
    "severity": "S4",
    "claim": "bubble_E2P applies in-place operators to its arguments (`r1 *= pc2cm`, `r2 *= pc2cm`, `Eb *= E_au2cgs`, `r2 += 1e-10`), which mutate the caller's object if a numpy array is ever passed.",
    "evidence": "lines 220-224. Safe today because all call sites pass Python/numpy scalars (get_bubbleproperties_pure line 228, get_effective_bubble_pressure lines 358/374/376), but nothing enforces that.",
    "expected": "Bind to new local names.",
    "failure_scenario": "A future vectorised call site silently has its input arrays scaled to cgs and offset by 1e-10.",
    "repro": "a = np.array([1.0]); bubble_E2P(np.array([1.0]), a, np.array([0.5]), 5/3); a is now in cm.",
    "confidence": "high"
  },
  {
    "id": "S7-A-23",
    "file": "trinity/bubble_structure/get_bubbleParams.py",
    "line": 437,
    "class": "silent-failure",
    "severity": "S3",
    "claim": "solve_R1 returns 0.0 silently for R2 <= 0 or R2 = NaN but raises ValueError for a non-finite Eb; the returned R1 = 0.0 is then fatal downstream via np.log10(R1).",
    "evidence": "lines 435-443: `if Lmech_total <= 0: return 0.0` / `if not (R2 > 0): return 0.0` (this also swallows NaN) / then `if not (np.isfinite(Eb) and ...): raise ValueError`. _create_radius_grid line 554 evaluates `np.logspace(np.log10(R1), ...)`, and np.linspace from -inf produces NaNs.",
    "expected": "Consistent handling: either all invalid inputs raise, or R1 = 0 is validated at the point of use before the log grid is built.",
    "failure_scenario": "A timestep with Lmech_total <= 0 (e.g. after the last SN) returns R1 = 0.0; the bubble grid becomes all-NaN and the solve fails with a message unrelated to the real cause.",
    "repro": "solve_R1(R2=1.0, Eb=1e4, Lmech_total=0.0, v_mech_total=1e3) -> 0.0, then _create_radius_grid(0.0, 0.9).",
    "confidence": "high"
  },
  {
    "id": "S7-A-25",
    "file": "trinity/bubble_structure/get_bubbleParams.py",
    "line": 282,
    "class": "regime",
    "severity": "S4",
    "claim": "get_leak_luminosity guards coverFraction >= 1.0 but not coverFraction < 0, so a negative covering fraction yields a leak factor (1 - f) > 1.",
    "evidence": "line 282 `if coverFraction >= 1.0 or Pb <= 0.0 or c_sound <= 0.0: return 0.0`; line 284 `... * (1.0 - coverFraction) * ...`.",
    "expected": "Clamp coverFraction to [0, 1] or validate at the trust boundary.",
    "failure_scenario": "",
    "repro": "",
    "confidence": "high"
  },
  {
    "id": "S7-A-26",
    "file": "trinity/bubble_structure/bubble_luminosity.py",
    "line": 785,
    "class": "divergence",
    "severity": "S3",
    "claim": "The non-CIE table lookups take log10 of phi with no positivity guard; Qi = 0 gives log10(0) = -inf as a table coordinate.",
    "evidence": "line 779 `phi_cond = params['Qi'].value / (4*np.pi*r_conduction**2)`; lines 784-789 and 823-828 `10 ** cooling_nonCIE.interp(np.transpose(np.log10([n/ndens_cgs2au, T, phi/phi_cgs2au])))`. Same pattern for n and T, which are positive by construction, but phi is proportional to Qi which can legitimately be zero.",
    "expected": "Floor phi at the table's lower edge before taking the logarithm.",
    "failure_scenario": "After the ionizing output has died away (Qi -> 0), the interpolator is queried at -inf; behaviour depends on the external table object (extrapolation, NaN, or an exception) and the resulting dudt silently contaminates L_conduction and L_intermediate.",
    "repro": "Set Qi = 0 in params and call _bubble_luminosity.",
    "confidence": "medium"
  },
  {
    "id": "S7-A-27",
    "file": "trinity/bubble_structure/get_bubbleParams.py",
    "line": 373,
    "class": "regime",
    "severity": "S4",
    "claim": "The R1 switch-on ramp goes negative for t < tSF, giving r1**3 < 0 and hence a shell volume LARGER than geometric.",
    "evidence": "lines 368-374 `dt_switchon = 1e-3; tmin = dt_switchon; if t <= (tmin + tSF): R1_tmp = (t - tSF)/tmin * R1; return bubble_E2P(Eb, R2, R1_tmp, gamma)`; bubble_E2P line 228 `shell_volume = r2**3 - r1**3`.",
    "expected": "Clamp the ramp factor to [0, 1].",
    "failure_scenario": "Any evaluation at t < tSF (e.g. an ODE solver probing backwards, or a cluster with a delayed formation time) returns a pressure below the correct value rather than being rejected.",
    "repro": "get_effective_bubble_pressure('energy', Eb, R2, R1, 5/3, t=0.9, tSF=1.0) vs t=1.0.",
    "confidence": "high"
  },
  {
    "id": "S7-A-28",
    "file": "trinity/bubble_structure/bubble_luminosity.py",
    "line": 645,
    "class": "numerical",
    "severity": "S3",
    "claim": "dMdt is converged against a 500-point, rtol=1e-6 integration, but every reported quantity comes from a separate ~6e4-point, rtol=1e-8 integration on which neither the v(R1)=0 condition nor monotonicity is re-checked.",
    "evidence": "search solve: lines 349-357 (t_eval=linspace(..., _RESIDUAL_NPTS=500), rtol=_RESIDUAL_RTOL=1e-6); production solve: lines 638/645-646 with _create_radius_grid (3 x int(2e4) points) and rtol=_BUBBLE_RTOL=1e-8. fsolve xtol is 1e-4 with epsfcn=1e-4 (lines 264-266). The only post-hoc check on the production profile is `np.any(T_array < 0)` at line 668.",
    "expected": "Either converge dMdt against the same tolerance/grid used for the reported profile, or assert |v(R1)|/v(r2Prime) is within the intended bar on the production solution.",
    "failure_scenario": "The reported bubble structure violates its own inner boundary condition by more than the nominal xtol, with no diagnostic; the discrepancy grows in stiff regimes where the two tolerances give visibly different profiles.",
    "repro": "After _solve_bubble_structure in _bubble_luminosity, log psoln[-1,0] / psoln[0,0] and compare against the residual value fsolve converged to.",
    "confidence": "high"
  },
  {
    "id": "S7-A-29",
    "file": "trinity/bubble_structure/get_bubbleParams.py",
    "line": 169,
    "class": "state",
    "severity": "S4",
    "claim": "Ebdot_to_cool_beta reads my_params entries as bare values while its algebraic twin cool_beta_to_Ebdot reads params entries as .value; the two cannot be called with the same container.",
    "evidence": "lines 169-174 `t_now = my_params[\"t_now\"]` etc. vs lines 112-120 `params['t_now'].value` etc.",
    "expected": "One convention, or an explicit adapter.",
    "failure_scenario": "Passing the standard params object to Ebdot_to_cool_beta propagates parameter wrapper objects into the arithmetic instead of floats.",
    "repro": "",
    "confidence": "high"
  },
  {
    "id": "S7-A-30",
    "file": "trinity/bubble_structure/bubble_luminosity.py",
    "line": 430,
    "class": "other",
    "severity": "S3",
    "claim": "The cooling/heating rate that drives the structure ODE and the one that is integrated to report the luminosity come from two independent code paths with different unit conventions, and nothing enforces that they agree.",
    "evidence": "ODE: line 430 `dudt = net_coolingcurve.get_dudt(params['t_now'].value, ndens, T, phi, params)` with AU-unit arguments. Luminosity: lines 784-791 and 823-829 hit params['cStruc_cooling_nonCIE'].interp / ['cStruc_heating_nonCIE'].interp directly with cgs arguments, and line 744 uses params['cStruc_cooling_CIE_interpolation'] with no heating term at all. The fA rule also differs (line 436 vs lines 845-848).",
    "expected": "One function producing dudt for a given (n, T, phi, t), used by both the ODE and the luminosity integrals.",
    "failure_scenario": "The energy the ODE removes from the structure differs from the L_total reported to the energy budget, so the bubble's thermal solution and its radiative loss are not mutually consistent - a discrepancy that no test in this slice would catch.",
    "repro": "Evaluate net_coolingcurve.get_dudt and the direct table combination at the same (n, T, phi) and compare.",
    "confidence": "medium"
  }
]
```
