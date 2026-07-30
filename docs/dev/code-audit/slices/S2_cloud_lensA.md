# S2 cloud properties — Lens A (what the code does)

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

Read with all comments/docstrings blanked. Everything below is derived from the arithmetic
alone. Shared read-only exception used: `trinity/_functions/unit_conversions.py` (for the
unit convention and constant values only).

Unit convention inferred from the arithmetic (not from prose): "au" code units are
**pc / Msun / Myr**; number density `nCore`, `nISM`, `nEdge` are in **pc⁻³**
(`ndens_cgs2au = 2.938e55 = (pc/cm)³`), and `mu_convert` is **Msun per particle**
(≈ 1.4·m_H·g2Msun = 1.1783e-57 Msun). Every `n·mu_convert` therefore yields Msun pc⁻³.

---

## 1. The density profile ρ(r) as actually evaluated

`trinity/cloud_properties/density_profile.py:55` `get_density_profile(r, params)` returns a
**number** density n(r) [pc⁻³]. `trinity/cloud_properties/mass_profile.py:83`
`get_mass_density` multiplies by `mu_convert` to get ρ [Msun pc⁻³].

Both branches share one composite blend (`density_profile.py:128-130`):

```
delta      = 0.01 * rCloud                              # SMOOTH_FRAC = 0.01, hard-coded
w(r)       = 0.5 * (1 + tanh((r - rCloud)/delta))
n(r)       = n_inside(r) * (1 - w(r))  +  nISM * w(r)
```

### 1a. Power law (`dens_profile == 'densPL'`, lines 135-148)

```
alpha == 0 :  n_inside(r) = nCore                          (all r; rCore is IGNORED)
alpha != 0 :  n_inside(r) = nCore                          for r <= rCore
              n_inside(r) = nCore * (r/rCore)**alpha       for r >  rCore
```

- **Core/envelope join at rCore:** matching is by construction — the envelope law evaluates
  to `nCore·(rCore/rCore)^α = nCore` at r = rCore. **ρ is continuous at rCore** (its slope is
  not, which is inherent to the model, not a defect).
- **Cloud/ISM join at rCloud:** *not* a join, it is a tanh ramp of width 0.01·rCloud.
  ρ is C^∞ there. Consequence: `n(rCloud) = ½(n_edge + nISM)`, **not** `n_edge`. The
  `nEdge` value stored in params and printed by the validators is `nCore·(rCloud/rCore)^α`,
  which the density profile never actually returns at any radius.
- For α < 0 and r = 0, line 143 evaluates `(0/rCore)**alpha` → `inf` with a numpy
  `divide by zero` RuntimeWarning *before* line 146 masks it back to `nCore`. Result is
  correct; the warning is spurious.
- `alpha == 0` is an exact float test; α = 1e-300 takes the power-law branch (harmless,
  identical answer, plus the r=0 inf).

### 1b. Bonnor–Ebert (`dens_profile == 'densBE'`, lines 153-164)

```
xi(r)     = be_r2xi(r, params)          # = r/a, a = c_s / sqrt(4 pi G rho_core)
n_inside  = nCore * f_rho_rhoc(xi(r))   # f_rho_rhoc = exp(-u), u = isothermal Lane-Emden
n(r)      = n_inside*(1-w) + nISM*w     # same tanh ramp
```

`f_rho_rhoc` is a cubic `interp1d` over ξ ∈ [1e-7, 20] with `fill_value=(1.0, rho[-1])`, so
ρ(0)=ρ_c exactly and ρ **clamps at ρ_c/222.3 for ξ > 20** instead of continuing to decay.
Because ξ_out ≤ 20 by construction and the tanh has already handed over to nISM outside
rCloud, the clamp is not reached in practice.

The Lane–Emden ODE (`bonnorEbertSphere.py:200-203`) is `u'' + 2u'/ξ = e^{-u}` — the correct
**isothermal** equation. The series ICs at line 216-217, `u = ξ²/6 − ξ⁴/120 + ξ⁶/1890` and
`u' = ξ/3 − ξ³/30 + ξ⁵/315`, are the correct expansion and are mutually consistent
(term-by-term derivative). I re-solved it: ρ/ρ_c is strictly decreasing and m = ξ²u′ strictly
increasing (except 167 machine-precision ties at ξ < 2.9e-7 where ρ = 1 to 16 digits — the
`np.unique` call at line 281 exists precisely to absorb those, and it does).

---

## 2. The mass profile M(r), and my own integration check

### 2a. Power law — `compute_enclosed_mass_powerlaw` (`mass_profile.py:267`)

```
alpha == 0:
  r <= rCloud : M = (4/3) pi r^3 rhoCore
  r >  rCloud : M = mCloud + (4/3) pi rhoISM (r^3 - rCloud^3)

alpha != 0:
  r <= rCore              : M = (4/3) pi r^3 rhoCore
  rCore < r <= rCloud     : M = 4 pi rhoCore [ rCore^3/3
                                + (r^(3+a) - rCore^(3+a)) / ((3+a) rCore^a) ]
  r >  rCloud             : M = mCloud + (4/3) pi rhoISM (r^3 - rCloud^3)
```

**My integration.** ∫_rCore^r 4πr′²·ρ_core·(r′/rCore)^α dr′ = 4πρ_core rCore^(−α)
[r^(3+α) − rCore^(3+α)]/(3+α), plus the core term 4πρ_core rCore³/3. **This is algebraically
identical to the coded region-2 expression.** Numerically confirmed: direct trapezoid of
4πr²ρ_sharp against the closed form agrees to 4.7e-12 relative.

So **M(r) is the exact integral of the *sharp* (unsmoothed) power-law density** — but the
sharp density is *not* what `get_density_profile` returns. See §7 finding 01.

**α = −3 (singular exponent) is NOT handled.** At α = −3 the region-2 expression is
`(r^0 − rCore^0)/(0·rCore^-3)` = `0/0` → **NaN**, emitted with only a numpy RuntimeWarning.
Confirmed: `M_code([0.5, 5, 10, 25])` with α=−3, rCore=1, rCloud=20 returns
`[1.813e+02, nan, nan, 1.001e+06]` — the entire envelope is NaN while the core and the ISM
region return finite numbers.
Critically, **the integral does not diverge**: with the uniform core cutting off r → 0,
∫4πr²·ρ_c(r/rCore)^-3 dr = 4πρ_c rCore³ ln(r/rCore), which is finite. I verified the log form
against direct numerical integration: ratio 1.000000. The correct closed form at α = −3 is

```
M(r) = 4 pi rhoCore [ rCore^3/3 + rCore^3 * ln(r/rCore) ]     (rCore < r <= rCloud)
```

and it is nowhere in the code. Near-α=−3 conditioning of the shipped expression is fine down
to |3+α| ≈ 1e-4 (rel err 3.8e-13) and degrades to 1.4e-4 rel err only at |3+α| ≈ 1e-13.

**Continuity at rCloud** requires region 2 evaluated at rCloud to equal `mCloud`. That holds
only when (mCloud, nCore, rCore, rCloud) are mutually consistent, i.e. when rCloud came from
`compute_rCloud_powerlaw`. Nothing in `compute_enclosed_mass_powerlaw` enforces it; an
inconsistent `.param` produces a step discontinuity in M at rCloud with no diagnostic.

**Region boundaries** are `r <= rCore` / `(r > rCore) & (r <= rCloud)` / `r > rCloud` — a
partition with no gap or overlap. Every point is assigned exactly once. Good. **But** if
rCloud < rCore (which is reachable, see finding 07) region 2 is empty and M(rCloud) is
computed by the *core* formula, which does not equal mCloud.

### 2b. Bonnor–Ebert — `compute_enclosed_mass_bonnor_ebert` (`mass_profile.py:347`)

Analytical path (`has_analytical`, lines 391-408):
```
xi_inside = xi_out * (r/rCloud)          # exactly r/a, since rCloud = xi_out * a
M(r)      = mCloud * f_m(xi_inside) / f_m(xi_out)
```
This is exact. Since (ξ²u′)′ = ξ²e^{−u}, we have ∫₀^r 4πr′²ρ dr′ = 4πρ_c a³·m(ξ), and
mCloud = 4πρ_c a³ m(ξ_out) by construction of c_s, so the ratio form is the true integral.
I confirmed by direct quadrature of 4πr²ρ_BE over [0, rCloud]: 1.000000e+06 vs mCloud
1.000000e+06, rel −9.9e-09 (quadrature error).

Outside: `M = mCloud + (4/3)π ρISM (r³ − rCloud³)` — continuous at rCloud in the sharp
picture.

**Numerical fallback path (lines 410-422) is broken.** It writes `M_arr[i]` where `i` is the
index into the *inside* sub-array, not into `M_arr`. Demonstrated with
`r_arr = [2, 5, 12, 8]`, rCloud = 10:
- inside mask `[T, T, F, T]`; the loop writes flat slots 0, 1, 2 — slot 2 is the r=12 point
  (later overwritten by the outside assignment) and slot 3 (r=8 pc, inside the cloud) is
  **left at 0.0**.
Additionally `M_arr[0] = 0.0` unconditionally, so the mass interior to `r_arr[0]` is dropped
even for a well-ordered array, and **a scalar r inside the cloud returns M = 0.0**
(verified). The path is only taken when `'densBE_f_m'`/`'densBE_xi_out'` are absent from
params, which the normal `create_BE_sphere_from_params` flow guarantees — so it is latent,
but it fails silently rather than raising.

### 2c. dM/dt (`mass_profile.py:224` and `compute_mass_accretion_rate:479`)

```
dMdt = 4 pi r^2 * rho(r) * rdot          # rho from get_density_profile => SMOOTHED
```
but `M(r)` above is the integral of the **sharp** profile. These are not each other's
derivative. Measured on a valid α = −2 config (mCloud 1e6, nCore 1e4 cm⁻³, rCore/rCloud=0.1,
nEdge/nISM = 100):

| r/rCloud | dM/dr from M(r) | 4πr²ρ_code | ratio |
|---|---|---|---|
| 0.500 | 3.683e+04 | 3.683e+04 | 1.0000 |
| 0.970 | 3.683e+04 | 3.674e+04 | 0.9976 |
| 0.995 | 3.683e+04 | 2.702e+04 | **0.7337** |
| 1.005 | 3.720e+02 | 1.018e+04 | **27.36** |
| 1.030 | 3.907e+02 | 4.808e+02 | 1.2306 |
| 1.500 | 8.286e+02 | 8.286e+02 | 1.0000 |

A factor 27 in the swept-up-mass rate exactly at cloud breakout.

---

## 3. Dimensions (derived from the arithmetic)

All consistent. Spot-checked chain:

| expression | file:line | units |
|---|---|---|
| `(r-rCloud)/delta` | density_profile.py:130 | dimensionless |
| `nCore*(r/rCore)**alpha` | density_profile.py:143 | pc⁻³ for any α |
| `n_arr*mu_convert` | mass_profile.py:126 | pc⁻³·Msun = Msun pc⁻³ |
| `(4/3)π r³ ρ` | mass_profile.py:316 | Msun |
| `(r^(3+α) − rCore^(3+α))/((3+α)rCore^α)` | mass_profile.py:334 | pc³ for any α |
| `4π r² ρ ṙ` | mass_profile.py:224 | Msun Myr⁻¹ |
| `(3M/(4πρ))^(1/3)` | powerLawSphere.py:73 | pc |
| `A(3+α)rCore^α + rCore^(3+α)` then `^(1/(3+α))` | powerLawSphere.py:153,163 | pc^(3+α) → pc |
| `g = f³/3 + (1−f^(3+α))/((3+α)f^α)` | powerLawSphere.py:186 | dimensionless |
| `n_core·mu·MSUN_TO_G/PC_TO_CM³` | bonnorEbertSphere.py:403 | g cm⁻³ |
| `G^1.5·sqrt(4πρ)` | bonnorEbertSphere.py:407 | cm³ g⁻¹ s⁻³ |
| `M_cgs/m_dim · factor` | bonnorEbertSphere.py:408 | cm³ s⁻³ → c_s in cm/s |
| `c_s/sqrt(4πGρ)` | bonnorEbertSphere.py:418 | cm |
| `mu·MSUN_TO_G·c_s²/(γ k_B)` | bonnorEbertSphere.py:431 | K |
| `4π m_dim ρCore a_pc³` | validate_gmc.py:507 | Msun |

No dimensional imbalance found in any expression. Two **default arguments** are in the wrong
unit system, though (findings 10, 11).

---

## 4. Inversions

### `compute_rCloud_homogeneous` (`powerLawSphere.py:51`)
`rCloud = (3M/(4πρ_core))^(1/3)` — exact inverse of `M = (4/3)πr³ρ`. Domain: M > 0. For
M < 0 Python returns a **complex** number (verified: `(4.417+7.651j)`), which then hits
`rCloud > r_max` in `check_gmc_constraints` and raises an uncaught `TypeError`.

### `compute_rCloud_powerlaw`, fixed-rCore branch (`powerLawSphere.py:148-178`)
```
A    = M/(4 pi rhoCore) - rCore^3/3
rhs  = A(3+a) rCore^a + rCore^(3+a)
rCloud = rhs^(1/(3+a))
```
Solving the forward region-2 expression for rCloud gives exactly this. **Exact algebraic
inverse.** Branch/domain: rejects `rhs <= 0`. For 3+α > 0 that is the correct and complete
condition (M below the core mass). For α < −3 (3+α < 0) the forward map is *decreasing* in
rCloud and rhs ≤ 0 means the requested mass exceeds the sphere's **asymptotic total** mass,
not that "the uniform core already exceeds the cloud mass budget" as the message at line 157
says — the message is wrong in that regime. It is self-checked at line 166-173 (rel_err >
1e-6 → RuntimeError), so a bad root cannot escape.
**Missing domain check: nothing requires rCloud > rCore.** With rCore = 5 pc and
mCloud = 0.8 × (core mass), the inverse happily returns rCloud = 4.667 pc < rCore.

### `compute_rCloud_powerlaw`, fractional-rCore branch (`powerLawSphere.py:180-211`)
Substituting rCore = f·rCloud factors rCloud³ out exactly, giving
`g = f³/3 + (1−f^(3+α))/((3+α)f^α)` and `rCloud = (M/(4πρ_c g))^(1/3)` — **exact**. Guard
`g <= 0` is correct (g > 0 for all f ∈ (0,1) and α ≠ −3, including α < −3).

### `r2xi` / `xi2r` (`bonnorEbertSphere.py:582, 622`)
`r2xi` reconstructs `c_s = sqrt(γ k_B T_eff/(mu·Msun2g))`, which is the exact algebraic
inverse of `T_eff = mu·Msun2g·c_s²/(γ k_B)` set at line 431. Round trip verified: c_s
recovered to 0.0e+00 relative error, and ξ(rCloud) = 4.075531 vs stored ξ_out = 4.075531.
`xi2r` is the exact inverse of `r2xi`. Both are exact.

### `f_xi_from_rho` (`bonnorEbertSphere.py:283`)
Inverts ρ/ρ_c → ξ by interpolation on the reversed, de-duplicated arrays. Since ρ/ρ_c is
strictly monotone (after the machine-precision ties near ξ→0), the branch is unique.
`f_xi_from_rho(1/14.04)` returns ξ = 6.45037 vs the constant `XI_CRITICAL = 6.451` — agrees.

---

## 5. Validation logic

### `check_gmc_constraints` (`validate_gmc.py:181`) — three tests, all errors (never warnings)

| # | condition | boundary | on trip |
|---|---|---|---|
| 1 | `rCloud > r_max` (default 200.0 pc) | rCloud == 200.0 **passes** (inclusive max) | error |
| 2 | `nEdge < nISM` | nEdge == nISM **passes** (inclusive) | error |
| 3 | `mass_error > mass_tolerance` (0.001) | error == 0.001 **passes** | error |

`mass_error = abs(M_computed − mCloud)/mCloud **if mCloud > 0 else 0.0**` (line 250). So
**mCloud ≤ 0 forces mass_error = 0 and test 3 cannot fire.** Verified: `mCloud = 0,
M_computed = 1e9` returns *zero errors*. Nothing else anywhere in `validate_gmc.py` checks
mCloud > 0 for the power-law branch (the BE branch is saved only because `create_BE_sphere`
line 351 rejects it).

The `warnings` list returned by `check_gmc_constraints` is always empty; only
`_validate_bonnor_ebert` ever appends one (the Ω > 14.04 instability warning).

### Test 3 is a tautology in both branches
- `_validate_powerlaw:435` recomputes `M_computed` with the *same* region-2 closed form that
  `compute_rCloud_powerlaw` algebraically inverted to get rCloud (and which already
  self-checks to 1e-6). Measured error: 2.0e-16.
- `_validate_bonnor_ebert:507` recomputes `M = 4π m_dim ρ_core a³` where a and m_dim came
  from solving that exact equation for c_s. Also identically mCloud.

So the headline "mass consistency" gate **can never fail** — while the mass profile the
solver actually integrates is off by 0.4–1.0 % (finding 01) or 1.6 % (finding 07).

### `create_BE_sphere` validation (`bonnorEbertSphere.py:350-363`)
- `M_cloud` non-finite or ≤ 0 → ValueError. `n_core` likewise.
- `Omega <= 1.0` → ValueError (Ω = ρ_c/ρ_out; Ω = 1 is the degenerate ξ_out = 0 sphere).
- `Omega > OMEGA_CRITICAL (14.04)` → **warning only**, not an error. Supercritical spheres
  are accepted and run.
- `is_stable = Omega < OMEGA_CRITICAL` (strict). At Ω == 14.04 exactly: `is_stable = False`
  but the logger warning at line 357 does **not** fire (strict `>`). The two boundaries
  disagree by one point.
- `target_rho_rhoc < rho_rhoc[-1]` → ValueError. With XI_MAX = 20 this caps Ω at 222.27.
- Ω → 1⁺ is accepted and produces absurd but arithmetically consistent output:
  Ω = 1.0001 → ξ_out = 0.0245, T_eff = 1.15e8 K, rCloud → the homogeneous value 19.03 pc.

### `validate_mass_at_rCloud` (`mass_profile.py:488`)
The only validator that uses the **real** `get_mass_profile`. `relative_error <= tolerance`
(inclusive, default 0.001). Handles mCloud ≤ 0 by returning `valid=False` with
`relative_error = inf`. This one *would* catch findings 01 and 07 — but it is not called from
`validate_gmc.py` at all, and nothing in this slice calls it.

### Parameters a validator rejects that other code claims to support
- **α = −3**: rejected by `compute_rCloud_powerlaw:143` ("mass integral diverges"), yet
  `compute_enclosed_mass_powerlaw` accepts it and silently returns NaN, and the integral is
  in fact finite. Rejection is right for the wrong reason; the mass profile has no guard.
- **rCore ≥ rCloud**: accepted everywhere, physically meaningless, and makes the validator
  and the mass profile disagree (finding 07).
- **mCloud ≤ 0** (densPL): accepted by `check_gmc_constraints`, rejected by
  `validate_mass_at_rCloud` and by `create_BE_sphere`.

### `compute_minimum_rCore` (`mass_profile.py:566`)
Solves nEdge = nCore(rCloud/rCore)^α ≥ nISM for rCore. `ratio = (nCore/nISM)^(1/α)`,
`rCore_min = rCloud·ratio`, `rCore_suggested = 1.1·rCore_min`, clipped to 0.9·rCloud if it
reaches rCloud. For **α < 0** (the intended regime) this is correct: rCore_min is a genuine
lower bound and the 1.1 margin moves in the satisfying direction (verified: α=−2 gives
nEdge/nISM = 0.81 / 1.00 / 1.21 at margin 0.9 / 1.0 / 1.1). For **α > 0** the inequality
flips and rCore_min is an *upper* bound, so `rCore_min·1.1` violates the constraint — this is
masked because for α>0 rCore_min lands far above rCloud (2e9 pc for α=0.5) and the
`>= rCloud` clip rescues it. The returned fourth value `rCore_min` is meaningless for α > 0.

### `_suggest_*_alternatives`
Pure grid searches over ±factors; `n_combos` subtracts exactly 1 for the skipped identity
combo, correct because 1.0 is present in every factor array. The `_distance` metric mixes
log10 distances for mCloud/nCore with a *linear* fractional distance for rCore
(`validate_gmc.py:631-634`), so the ranking is not on a common scale. Line 592-593 would hit
a `TypeError` if `rc is None` with α ≠ 0, but `_validate_powerlaw:412` already raises in that
case, so the path is unreachable. `search_range=0.5` (line 551) is never referenced.

---

## 6. Numeric literals in arithmetic

**density_profile.py**
- `SMOOTH_FRAC = 0.01` → `delta = 0.01*rCloud` (L128-129) — the tanh ramp half-width.
- `0.5 * (1.0 + np.tanh(...))` (L130); `(1.0 - w_outside)` (L148, L164).

**mass_profile.py**
- `(4.0/3.0)*np.pi` (L316, L320, L327, L340, L426); `4.0*np.pi` (L224, L332, L420, L479).
- `rCore**3 / 3.0` and `(3.0 + alpha)` (L333-335); `r**(3.0+alpha)` (L334).
- `r_inside/rCloud` scaled by `xi_out` (L401).
- `tolerance=0.001` (L488); `*100` twice in the message (L550).
- `margin=1.1` (L566); `rCloud*0.1` (L609); `(nCore/nISM)**(1.0/alpha)` (L614);
  `rCloud*0.9` (L622).

**powerLawSphere.py**
- `mu=1.4` default in three signatures (L51, L77, L214) — see finding 10.
- `(3*M/(4*pi*rho))**(1.0/3.0)` (L73); `rCore_fraction=0.1` (L77, L214).
- `rCore_val**3/3.0`, `(3.0+alpha)` (L136-139).
- `abs(3.0+alpha) < 1e-14` (L143) — the α = −3 rejection threshold.
- `rCore**3/3.0` (L152); `rel_err > 1e-6` (L168, L200).
- `f**3/3.0`, `(1.0 - f**(3.0+alpha))`, `(3.0+alpha)*f**alpha` (L186); `**(1.0/3.0)` (L194).
- `nISM=1.0` default (L214) — code-unit mismatch, unused in the body.

**bonnorEbertSphere.py**
- `OMEGA_CRITICAL = 14.04` (L78) — used. `XI_CRITICAL = 6.451` (L79),
  `M_DIM_CRITICAL = 15.70` (L87), `M_BONNOR_CRITICAL = 1.182` (L88) — **never referenced**.
  All four are numerically correct: I re-solved and got ξ(Ω=14.04) = 6.4504,
  m(ξ_crit) = 15.703, implied Bonnor coefficient m/√(4π)/√14.04 = 1.1822.
- `XI_MIN = 1e-7`, `XI_MAX = 20.0`, `N_POINTS = 5000` (L91-93).
- `2.0*omega/xi` (L202) — the correct spherical Laplacian term.
- Series: `/6.0, /120.0, /1890.0` and `/3.0, /30.0, /315.0` (L216-217) — correct and mutually
  consistent.
- `mu=1.4`, `gamma=5.0/3.0` defaults (L302-303); `Omega <= 1.0` (L355); `1.0/Omega` (L378).
- `G_CGS ** 1.5`, `4.0*np.pi` (L407); `** (1.0/3.0)` (L409); `sqrt(4.0*pi*G*rho)` (L418,
  L471, L493); `gamma*K_B_CGS` (L431); `c_s / 1.0e5` → km/s (L564).

**validate_gmc.py**
- `R_CLOUD_MAX = 200.0` pc (L111); `MASS_TOLERANCE = 0.001` (L112).
- `nISM=1.0` default (L186) — code-unit mismatch.
- `gamma=5.0/3.0` default (L280); `(4.0/3.0)*pi`, `4.0*pi`, `rCore**3/3.0`, `(3.0+alpha)`
  (L433-439, L602-608, L507, L691).
- Search grids `[0.5,0.8,0.9,1.0,1.1,1.2,1.5]` (L559-560, L650), rCore grid
  `[0.5 … 1.5]` step 0.1 (L565), nCore BE grid with extra `2.0, 5.0` (L651);
  `- 1` (L567, L656); `n_suggestions=3` (L551, L643); `search_range=0.5` (L551, unused).

---

## 7. Findings

Reproduction scripts live at
`/tmp/claude-0/-home-user-trinity/75528b15-99c6-5b6c-980a-4aac19bbcd57/scratchpad/chk.py`,
`chk2.py`-equivalent heredocs (outputs in `chk2.out`, `chk3.out`, `chk4.out`). All numbers
quoted above came from re-implementing the coded formulas standalone in the stated code
units — no trinity module was imported.

```json
[
  {
    "id": "S2-A-01",
    "file": "trinity/cloud_properties/mass_profile.py",
    "line": 214,
    "class": "numerical",
    "severity": "S2",
    "claim": "M(r) is not the integral of the rho(r) the code actually evaluates. compute_enclosed_mass receives rho_arr (the tanh-smoothed density from get_density_profile) and, for densPL, discards it entirely in favour of a closed form that integrates the SHARP profile; the densBE analytical path does the same. The mismatch at r=rCloud is 4x to 10x the code's own MASS_TOLERANCE of 0.001.",
    "evidence": "density_profile.py:128-130 blends n_inside into nISM over delta=0.01*rCloud via w=0.5(1+tanh((r-rCloud)/delta)); mass_profile.py:260 routes densPL to compute_enclosed_mass_powerlaw(r_arr, params) which never touches rho_arr. Numerically, with mCloud=1e6 Msun, nISM=1 cm^-3: alpha=0, nCore=1e3 cm^-3, rCloud=19.034 pc -> int(4 pi r^2 rho_smooth)dr = 9.897358e5 vs M_code(rCloud)=1.000000e6, rel -1.0264%. alpha=-2, nCore=1e4 cm^-3, rCore/rCloud=0.1 (nEdge/nISM=100) -> 9.963234e5 vs 1.000000e6, rel -0.3677%. densBE Omega=5, nCore=1e3 cm^-3 -> 9.942758e5 vs 1e6, rel -0.5724%. The same quadrature applied to the SHARP profile reproduces the closed form to 4.7e-12, confirming the closed form is the sharp integral.",
    "expected": "Either integrate the smoothed density that is actually returned (so M' = 4 pi r^2 rho holds), or apply the tanh smoothing consistently to M(r) as well; in both cases M(rCloud) should reproduce mCloud to better than MASS_TOLERANCE.",
    "failure_scenario": "The shell's swept-up mass and the gravitational mass interior to it are drawn from a profile ~0.4-1% inconsistent with the density used for ram pressure and cooling. Momentum-driven expansion and the gravitational force term inherit a systematic ~1% mass bias that grows as the shell approaches the cloud edge, exactly where the phase transition to breakout is decided.",
    "repro": "Reimplement get_density_profile (density_profile.py:105-170) and compute_enclosed_mass_powerlaw (mass_profile.py:297-344) standalone in code units, then compare scipy.integrate.trapezoid(4*pi*r**2*rho_smooth, r) over [0, rCloud] against the closed form at r=rCloud. See scratchpad/chk.py.",
    "confidence": "high"
  },
  {
    "id": "S2-A-02",
    "file": "trinity/cloud_properties/mass_profile.py",
    "line": 224,
    "class": "numerical",
    "severity": "S2",
    "claim": "dM/dt returned by get_mass_profile(return_mdot=True) and by compute_mass_accretion_rate is 4*pi*r^2*rho_SMOOTHED*rdot, which is not the time derivative of the M(r) returned alongside it. The two disagree by a factor of 27 just outside rCloud and by 27% just inside.",
    "evidence": "mass_profile.py:224 and :479 use rho_arr from get_mass_density (smoothed), while M_arr comes from the sharp closed form. Measured on a valid alpha=-2 config (mCloud=1e6, nCore=1e4 cm^-3, rCore=2.909 pc, rCloud=29.095 pc, nEdge/nISM=100), ratio (4 pi r^2 rho_code)/(dM/dr from M(r)): r/rCloud=0.500 -> 1.0000; 0.970 -> 0.9976; 0.995 -> 0.7337; 1.005 -> 27.36; 1.030 -> 1.2306; 1.500 -> 1.0000.",
    "expected": "dM/dt must equal d/dt of the M(r) actually used, i.e. the same density must feed both. Ratio should be 1.0 at every radius including the cloud edge.",
    "failure_scenario": "During cloud breakout the shell mass integrated from Mdot diverges from the M(r) used in the momentum/energy equations, so mass is created or destroyed at the cloud edge. A shell stalling near rCloud sees a 27x overestimate of accreted mass just outside the edge, which can spuriously halt or reverse expansion.",
    "repro": "Central-difference the coded M(r) at r/rCloud in {0.5, 0.97, 0.995, 1.005, 1.03, 1.5} and compare to 4*pi*r^2*get_density_profile(r)*mu_convert. See scratchpad/chk2.out.",
    "confidence": "high"
  },
  {
    "id": "S2-A-03",
    "file": "trinity/cloud_properties/mass_profile.py",
    "line": 334,
    "class": "divergence",
    "severity": "S2",
    "claim": "densPL_alpha = -3 makes the region-2 mass expression 0/0 and returns NaN for the entire envelope, with no guard and no exception - only a numpy RuntimeWarning.",
    "evidence": "M[region2] = 4 pi rhoCore*(rCore**3/3 + (r**(3+alpha) - rCore**(3+alpha))/((3+alpha)*rCore**alpha)). At alpha=-3 both powers are r**0 = 1, so the numerator is 0 and the denominator is 0*rCore**-3 = 0. Executed with alpha=-3, rCore=1 pc, rCloud=20 pc: M_code([0.5, 5, 10, 25]) = [1.81264610e+02, nan, nan, 1.00110571e+06] - finite inside the core and outside the cloud, NaN across the whole envelope.",
    "expected": "Either raise (as powerLawSphere.py:143 does) or, better, use the correct closed form. The integral is finite because the uniform core regularises r->0: M(r) = 4 pi rhoCore [rCore^3/3 + rCore^3 * ln(r/rCore)] for rCore < r <= rCloud. I verified this against direct quadrature of the sharp profile: log form 1.146717e4 vs numerical 1.146717e4, ratio 1.000000.",
    "failure_scenario": "A sweep that includes alpha = -3 (a common isothermal-sphere-like choice, and a plausible grid endpoint between -2 and -4) propagates NaN into the shell mass, the gravity term and every downstream force. Whether the run dies or silently produces NaN trajectories depends on which consumer touches it first.",
    "repro": "compute_enclosed_mass_powerlaw with densPL_alpha=-3.0, rCore=1.0, rCloud=20.0 on r = [0.5, 5, 10, 25]. See scratchpad/chk2.out section 'alpha = -3'.",
    "confidence": "high"
  },
  {
    "id": "S2-A-04",
    "file": "trinity/cloud_properties/mass_profile.py",
    "line": 297,
    "class": "regime",
    "severity": "S2",
    "claim": "rCloud < rCore is silently reachable, and when it happens the validator and the mass profile use different formulas for M(rCloud): the validator reports a 2e-16 mass error while the profile the solver integrates is off by 1.6%.",
    "evidence": "compute_rCloud_powerlaw (powerLawSphere.py:148-178) only rejects rhs <= 0; it never checks rCloud > rCore. With alpha=-2, rCore=5 pc, nCore=1e4 cm^-3 (uniform-core mass 1.8126e5 Msun) and mCloud = 1.4501e5 Msun (0.8x the core mass) the inversion returns rCloud = 4.667 pc < rCore = 5 pc. _validate_powerlaw:435 then evaluates the region-2 closed form UNCONDITIONALLY -> M_computed = 1.450117e5, error 2.01e-16, validation PASSES. compute_enclosed_mass_powerlaw:326-327 instead selects region 1 (r <= rCore) -> M(rCloud) = 1.473748e5, error 1.63e-02, i.e. 16x the declared MASS_TOLERANCE.",
    "expected": "rCore >= rCloud should be rejected (a core larger than the cloud is meaningless), and the validator should evaluate M_computed through get_mass_profile so it sees the same branch selection the solver does.",
    "failure_scenario": "A dense, compact .param (small mCloud with a large rCore) validates clean and then runs with a total cloud mass 1.6% below the requested mCloud, with the whole cloud silently uniform rather than power-law.",
    "repro": "alpha=-2, rCore=5.0 pc, nCore=1e4 cm^-3, mCloud=1.4501e5 Msun: compare the region-2 formula at rCloud (validator) with compute_enclosed_mass_powerlaw(rCloud). See scratchpad/chk2.out section 'rCloud < rCore'.",
    "confidence": "high"
  },
  {
    "id": "S2-A-05",
    "file": "trinity/cloud_properties/mass_profile.py",
    "line": 415,
    "class": "numerical",
    "severity": "S2",
    "claim": "The Bonnor-Ebert numerical fallback loop indexes the output array with the inside-subarray index, writing enclosed masses into the wrong slots; and it returns 0.0 for a scalar radius inside the cloud.",
    "evidence": "for i, (r, rho) in enumerate(zip(r_inside, rho_inside)): ... M_arr[i] = trapezoid(...). i runs over r_inside, but the assignment target is the FULL array M_arr. Executed with r_arr = [2, 5, 12, 8], rCloud = 10: inside mask [T, T, F, T]; the loop writes flat slots 0, 1, 2 - slot 2 belongs to r=12 (outside, later overwritten by line 426) and slot 3 (r=8 pc, inside) is left at 0.0. Separately, M_arr[i]=0.0 at i==0 is unconditional, so a scalar r inside the cloud returns exactly 0.0 (verified), and for a sorted array the mass interior to r_arr[0] is dropped.",
    "expected": "M_arr[inside_cloud] = <cumulative integral>, e.g. build the result into a local array and assign it through the boolean mask; and the [0, r_inside[0]] contribution should be included rather than set to zero.",
    "failure_scenario": "Any densBE call path whose params lack densBE_f_m/densBE_xi_out (a restart, a reader-reconstructed params, an output-analysis helper) silently returns zero or misplaced enclosed masses instead of raising.",
    "repro": "Run the loop body of mass_profile.py:415-422 on r_arr=[2,5,12,8], rho_arr=linspace(5,1,4), rCloud=10 and inspect M_arr. See scratchpad/chk3.out section 'fallback loop index behaviour'.",
    "confidence": "high"
  },
  {
    "id": "S2-A-06",
    "file": "trinity/cloud_properties/validate_gmc.py",
    "line": 250,
    "class": "silent-failure",
    "severity": "S3",
    "claim": "mCloud <= 0 silently passes GMC validation on the densPL path: mass_error is forced to 0.0 whenever mCloud <= 0, so the mass-consistency test cannot fire, and no other check rejects a non-positive cloud mass.",
    "evidence": "mass_error = abs(M_computed - mCloud) / mCloud if mCloud > 0 else 0.0 (line 250), then 'if mass_error > mass_tolerance'. Executed: check_gmc_constraints(rCloud=0.0, nEdge=nCore, mCloud=0.0, M_computed=0.0, nISM) -> errors []; check_gmc_constraints(rCloud=10.0, nEdge=nCore, mCloud=0.0, M_computed=1e9, nISM) -> errors []. For mCloud < 0 with alpha=0, compute_rCloud_homogeneous returns a COMPLEX rCloud ((3*-1e6/(4 pi rho))**(1/3) = 4.417+7.651j), which then raises an uncaught TypeError at the 'rCloud > r_max' comparison (TypeError is not in the caught tuple at validate_gmc.py:420, and the comparison happens outside the try anyway).",
    "expected": "Reject mCloud <= 0 explicitly, as create_BE_sphere:351 and validate_mass_at_rCloud:528 already do for their own inputs.",
    "failure_scenario": "A typo or a sweep expression that yields mCloud = 0 passes validation and produces a zero-mass cloud, or mCloud < 0 crashes with an opaque TypeError instead of a parameter error.",
    "repro": "check_gmc_constraints(10.0, nEdge, 0.0, 1e9, nISM) -> {'errors': [], 'mass_error': 0.0}. See scratchpad/chk2.out section 'mCloud <= 0'.",
    "confidence": "high"
  },
  {
    "id": "S2-A-07",
    "file": "trinity/cloud_properties/validate_gmc.py",
    "line": 435,
    "class": "deadcode",
    "severity": "S3",
    "claim": "The 'mass consistency' gate in check_gmc_constraints is a tautology in both profile branches - M_computed is recomputed with the same closed form that was algebraically inverted to produce rCloud - so it can never fail and provides no assurance about the mass profile the solver actually uses.",
    "evidence": "_validate_powerlaw:416 obtains rCloud from compute_rCloud_powerlaw (which itself already asserts rel_err <= 1e-6 at powerLawSphere.py:168), then line 435 re-evaluates the identical region-2 expression. Measured mass_error: 2.0e-16 (alpha=-2) and exactly 0 (alpha=0). _validate_bonnor_ebert:507 computes M = 4 pi m_dim rhoCore a_pc^3 with a_pc = rCloud/xi_out, but c_s (hence a) was solved from exactly that equation at bonnorEbertSphere.py:408, so it also returns mCloud identically. Meanwhile the profile the solver integrates is off by 0.4-1.0% (finding S2-A-01) or 1.6% (S2-A-04), and neither is visible to this gate.",
    "expected": "M_computed should come from get_mass_profile(rCloud, params) - i.e. the same code path the solver uses - as validate_mass_at_rCloud (mass_profile.py:525) already does. That function exists and is never called from validate_gmc.py.",
    "failure_scenario": "Operators read 'mass error 0.0000%' and trust that M(rCloud) == mCloud, while the integrated profile is off by up to 1.6%.",
    "repro": "Instrument _validate_powerlaw to also print get_mass_profile(rCloud, params) and compare with M_computed for the rCloud<rCore config in S2-A-04.",
    "confidence": "high"
  },
  {
    "id": "S2-A-08",
    "file": "trinity/cloud_properties/powerLawSphere.py",
    "line": 143,
    "class": "divergence",
    "severity": "S3",
    "claim": "The alpha = -3 rejection is justified by a false statement: with the uniform core the mass integral does NOT diverge, it has a finite logarithmic closed form. No log branch exists anywhere in the slice, so alpha = -3 is unnecessarily unusable.",
    "evidence": "powerLawSphere.py:143-146 raises 'alpha approx -3: mass integral diverges, cannot compute rCloud.' But density_profile.py:146 flattens rho to nCore for r <= rCore, so int_rCore^r 4 pi r'^2 rhoCore (r'/rCore)^-3 dr' = 4 pi rhoCore rCore^3 ln(r/rCore), finite for all finite r. Verified against direct quadrature: log form 1.146717e4 vs numerical 1.146717e4, ratio 1.000000. The divergence claim would only hold for a core-less r^-3 sphere.",
    "expected": "Add the alpha = -3 branch (M and its inversion both have exact log closed forms: rCloud = rCore*exp(A/rCore^3) with A = M/(4 pi rhoCore) - rCore^3/3), or at minimum correct the message.",
    "failure_scenario": "A physically legitimate and commonly used density slope is rejected at setup time on false grounds; users pick alpha = -2.999 instead, which S2-A-03's expression handles but with degrading conditioning (rel err 1.4e-4 at |3+alpha| = 1e-13).",
    "repro": "Compare 4*pi*rhoCore*(rCore**3/3 + rCore**3*log(r/rCore)) at r=10, rCore=1 with trapezoid(4*pi*r**2*rho_sharp, r) over [0,10]. See scratchpad/chk2.out.",
    "confidence": "high"
  },
  {
    "id": "S2-A-09",
    "file": "trinity/cloud_properties/density_profile.py",
    "line": 130,
    "class": "state",
    "severity": "S3",
    "claim": "The nEdge that validators compute, store in params and print is never the density the profile returns at rCloud: because of the tanh ramp, get_density_profile(rCloud) = (nEdge + nISM)/2 exactly.",
    "evidence": "At r = rCloud, tanh(0) = 0 so w_outside = 0.5 and n = 0.5*n_inside + 0.5*nISM (density_profile.py:130, 148, 164). But validate_gmc.py:419 sets nEdge = nCore*(rCloud/rCore)**alpha and bonnorEbertSphere.py:575 sets params['nEdge'] = n_core/Omega, i.e. n_inside(rCloud). The nEdge >= nISM check at validate_gmc.py:237 therefore constrains a quantity the profile does not produce.",
    "expected": "Either report the evaluated n(rCloud), or make the profile reach n_inside(rCloud) at rCloud (e.g. centre the ramp at rCloud + delta) so nEdge means what it is checked to mean.",
    "failure_scenario": "A config tuned so that nEdge == nISM exactly (the inclusive boundary of the validator) actually has n(rCloud) = nISM, i.e. it passes; but a config with nEdge = 2*nISM has n(rCloud) = 1.5*nISM. Any downstream code that reads params['nEdge'] as the shell's ambient density at breakout is off by up to a factor of 2.",
    "repro": "get_density_profile(rCloud, params) vs params['nEdge'].value for any densPL config.",
    "confidence": "high"
  },
  {
    "id": "S2-A-10",
    "file": "trinity/cloud_properties/bonnorEbertSphere.py",
    "line": 431,
    "class": "regime",
    "severity": "S3",
    "claim": "densBE_Teff is defined through the ADIABATIC sound speed (c_s^2 = gamma k T / m) although the sphere solved is the ISOTHERMAL Lane-Emden equation, so the stored temperature is a factor gamma = 5/3 below the temperature that reproduces the sphere's pressure.",
    "evidence": "bonnorEbertSphere.py:200-203 integrates u'' + 2u'/xi = exp(-u), the isothermal equation, for which P = rho c_s^2 with c_s^2 = k T/(mu m_H). Line 431 then sets T_eff = mu*MSUN_TO_G*c_s**2/(gamma*K_B_CGS), i.e. T_eff = m c_s^2/(gamma k), so k T_eff/(mu m_H) = c_s^2/gamma. For Omega=5, mCloud=1e6 Msun, nCore=1e3 cm^-3 the code reports c_s = 8.968 km/s with T_eff = 8189 K; the isothermal temperature matching that c_s is m c_s^2/k = 13648 K. r2xi (line 606) applies the same gamma so the round trip is exact within this slice (verified: c_s recovered to 0.0e0 relative error, xi(rCloud) = 4.075531 = xi_out).",
    "expected": "For an isothermal Lane-Emden structure the consistent definition is T = mu m_H c_s^2 / k_B (gamma = 1), or the gamma factor must be documented and applied identically by every consumer of densBE_Teff.",
    "failure_scenario": "Any module outside this slice that uses densBE_Teff as the gas temperature - thermal pressure n k T, cooling-table lookup, initial shell temperature - is low by 5/3 relative to the pressure the BE structure assumes, so the cloud is not in the hydrostatic equilibrium the profile was built on.",
    "repro": "For any densBE run compare params['densBE_Teff'] with mu_convert*Msun2g*(densBE_sigma*1e5)**2/k_B.",
    "confidence": "medium"
  },
  {
    "id": "S2-A-11",
    "file": "trinity/cloud_properties/powerLawSphere.py",
    "line": 51,
    "class": "units",
    "severity": "S3",
    "claim": "The mu=1.4 default in compute_rCloud_homogeneous, compute_rCloud_powerlaw, compute_consistent_params and create_BE_sphere is in the wrong unit system: the code's mu_convert is Msun per particle (1.1783e-57), not the dimensionless mean molecular weight, so a caller taking the default gets rCloud wrong by (1.4/1.1783e-57)^(1/3) ~ 1.06e19.",
    "evidence": "powerLawSphere.py:51, 77, 214 and bonnorEbertSphere.py:302 all default mu=1.4, and the body does rhoCore = nCore*mu with nCore in pc^-3, so rhoCore must come out in Msun pc^-3 -> mu must be Msun per particle. Every in-slice caller passes params['mu_convert'].value (validate_gmc.py:372, 409, 416; bonnorEbertSphere.py:536), whose value is 1.4*m_H*g2Msun = 1.1783e-57 Msun. compute_consistent_params (powerLawSphere.py:214) additionally defaults nISM=1.0 and is not called anywhere in the slice.",
    "expected": "No default, or a default equal to the code-unit mu_convert. A dimensionless 1.4 is only correct in a cgs convention this code does not use here.",
    "failure_scenario": "A test, a tools/ utility or a notebook calls compute_rCloud_homogeneous(mCloud, nCore) without mu and silently gets a cloud radius ~1e19 times too small, which then trips the rCloud_max validator (or worse, does not).",
    "repro": "compute_rCloud_homogeneous(1e6, 1e3*ndens_cgs2au) vs the same call with mu=1.4*m_H*g2Msun: 1.79e-19 pc vs 19.03 pc.",
    "confidence": "high"
  },
  {
    "id": "S2-A-12",
    "file": "trinity/cloud_properties/validate_gmc.py",
    "line": 186,
    "class": "units",
    "severity": "S4",
    "claim": "check_gmc_constraints defaults nISM=1.0, which in code units is 1 particle per cubic parsec (3.4e-56 cm^-3), so if the default is ever taken the nEdge >= nISM test is unconditionally satisfied.",
    "evidence": "validate_gmc.py:186 'nISM=1.0'; line 237 'if nEdge < nISM'. nCore/nISM elsewhere are pc^-3 (validate_gmc.py:94 and :102 multiply them by ndens_au2cgs = 3.404e-56 to print cm^-3). A realistic nISM = 1 cm^-3 is 2.938e55 in these units. All in-slice callers pass nISM explicitly, so this is latent.",
    "expected": "No default, or 1.0*ndens_cgs2au.",
    "failure_scenario": "A direct caller of check_gmc_constraints omits nISM and the edge-density sanity check silently becomes a no-op.",
    "repro": "check_gmc_constraints(10.0, 1e-40, 1e6, 1e6) -> no 'Edge density' error.",
    "confidence": "high"
  },
  {
    "id": "S2-A-13",
    "file": "trinity/cloud_properties/mass_profile.py",
    "line": 614,
    "class": "sign",
    "severity": "S4",
    "claim": "compute_minimum_rCore's rCore_min is a lower bound only for alpha < 0; for alpha > 0 the inequality reverses and rCore_min is an UPPER bound, so multiplying it by margin=1.1 moves away from the constraint. The fourth return value is therefore mislabelled for alpha > 0.",
    "evidence": "The constraint nCore*(rCloud/rCore)^alpha >= nISM rearranges to rCore >= rCloud*(nCore/nISM)^(1/alpha) for alpha < 0 but rCore <= rCloud*(nCore/nISM)^(1/alpha) for alpha > 0. Line 614-618 computes ratio = (nCore/nISM)**(1/alpha) and rCore_suggested = rCloud*ratio*1.1 unconditionally. Verified: nCore=1e4, nISM=1, rCloud=20 -> alpha=-2 gives rCore_min=0.2 with nEdge/nISM = 0.81/1.00/1.21 at margin 0.9/1.0/1.1 (correct); alpha=+1 gives rCore_min = 200000 pc and alpha=+0.5 gives 2e9 pc, both far above rCloud.",
    "expected": "Branch on sign(alpha), or document that the function is defined only for alpha < 0.",
    "failure_scenario": "Masked in practice: for alpha > 0 the huge rCore_min trips the 'rCore_suggested >= rCloud' clip at line 621 and falls back to 0.9*rCloud, which does satisfy nEdge >= nISM. Only the returned rCore_min is wrong, so a caller that uses the fourth return value as a hard floor for alpha > 0 gets nonsense.",
    "repro": "compute_minimum_rCore(1e4, 1.0, 20.0, +1.0) -> rCore_min = 200000.0 pc for a 20 pc cloud.",
    "confidence": "high"
  },
  {
    "id": "S2-A-14",
    "file": "trinity/cloud_properties/initial_profile.py",
    "line": 144,
    "class": "state",
    "severity": "S3",
    "claim": "The mock params dict built for the densBE path is missing four keys that create_BE_sphere_from_params assigns unconditionally (densBE_xi_arr, densBE_u_arr, densBE_dudxi_arr, densBE_rho_rhoc_arr), so that path raises KeyError if it is reached.",
    "evidence": "initial_profile.py:144-147 seeds only densBE_Teff, densBE_xi_out, densBE_f_rho_rhoc, densBE_f_m. bonnorEbertSphere.py:565-568 does params['densBE_xi_arr'].value = ..., params['densBE_u_arr'].value = ..., params['densBE_dudxi_arr'].value = ..., params['densBE_rho_rhoc_arr'].value = ... with no membership guard (only the three keys at line 554-560 get the 'if key not in params' treatment). A plain dict raises KeyError on those four.",
    "expected": "Either seed those four keys in the mock dict, or extend the key-creation loop at bonnorEbertSphere.py:554 to cover them.",
    "failure_scenario": "build_initial_cloud_profile(dens_profile='densBE', ...) - the post-hoc profile reconstruction helper used against metadata.json - dies with KeyError instead of returning (r, n, M).",
    "repro": "Call build_initial_cloud_profile with dens_profile='densBE' and inspect whether _init_bonnor_ebert_cloud reaches create_BE_sphere_from_params. NOTE: _init_bonnor_ebert_cloud is outside this slice (trinity/phase0_init/get_InitCloudProp.py), so reachability is unverified.",
    "confidence": "low"
  },
  {
    "id": "S2-A-15",
    "file": "trinity/cloud_properties/bonnorEbertSphere.py",
    "line": 357,
    "class": "other",
    "severity": "S4",
    "claim": "The stability warning and the is_stable flag use inconsistent boundaries at Omega == OMEGA_CRITICAL: the sphere is flagged unstable but no warning is logged.",
    "evidence": "Line 357 'if Omega > OMEGA_CRITICAL' logs the warning; line 363 'is_stable = Omega < OMEGA_CRITICAL'. At Omega == 14.04 exactly, is_stable is False (so _validate_bonnor_ebert:519 appends the 'UNSTABLE' warning to the result) but create_BE_sphere logs nothing.",
    "expected": "One boundary convention for both, e.g. both >= or both <.",
    "failure_scenario": "Cosmetic only - a run at exactly the critical Omega is reported inconsistently between the log and the validation result.",
    "repro": "create_BE_sphere(..., Omega=14.04) and inspect is_stable plus the captured log.",
    "confidence": "high"
  },
  {
    "id": "S2-A-16",
    "file": "trinity/cloud_properties/bonnorEbertSphere.py",
    "line": 79,
    "class": "deadcode",
    "severity": "S4",
    "claim": "Three module constants and one function parameter are defined and never used; mass_profile.py imports two functions it never calls.",
    "evidence": "XI_CRITICAL (line 79), M_DIM_CRITICAL (line 87) and M_BONNOR_CRITICAL (line 88) appear nowhere else in the package slice (only OMEGA_CRITICAL is used). validate_gmc.py:551 declares search_range=0.5 and never references it. mass_profile.py:32-35 imports compute_rCloud_homogeneous and compute_rCloud_powerlaw, neither of which appears elsewhere in the file. bonnorEbertSphere.py also imports scipy.optimize (line 52) and binds M_H_CGS (line 66) and MYR_TO_S (line 71) without using them. Ruff is configured for F821/F811/F823/E9 only, so F401 does not catch these. The three unused constants are numerically CORRECT: I re-solved Lane-Emden and got xi(Omega=14.04) = 6.4504, m(xi_crit) = 15.703, Bonnor coefficient 1.1822.",
    "expected": "Flagging only, per the project rule on pre-existing dead code - do not delete.",
    "failure_scenario": "",
    "repro": "grep -rn 'XI_CRITICAL\\|M_DIM_CRITICAL\\|M_BONNOR_CRITICAL\\|search_range' trinity/cloud_properties/",
    "confidence": "high"
  },
  {
    "id": "S2-A-17",
    "file": "trinity/cloud_properties/density_profile.py",
    "line": 143,
    "class": "numerical",
    "severity": "S4",
    "claim": "For alpha < 0 the power-law expression is evaluated at r = 0 before the core mask is applied, producing inf and a numpy divide-by-zero RuntimeWarning on every call that includes the origin.",
    "evidence": "Line 143 computes n_inside = nCore*(r_arr/rCore)**alpha over the whole array; line 146 only afterwards overwrites n_inside[r_arr <= rCore] = nCore. With r_arr[0] = 0 and alpha < 0, (0/rCore)**alpha = inf. The final value is correct; the warning is not.",
    "expected": "Evaluate the power only where r > rCore (np.where on the mask, or clip r to rCore first).",
    "failure_scenario": "Warning noise in every profile plot or grid evaluation that starts at r = 0; masks genuine numerical warnings from other modules.",
    "repro": "get_density_profile(np.linspace(0, rCloud, 100), params) with densPL_alpha = -2.",
    "confidence": "high"
  }
]
```
