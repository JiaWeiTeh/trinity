# S3 phase0 init — Lens A (what the code does)

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

## 0. Scope, method, declared exceptions

Read (comment/docstring-blanked copies):

- `trinity/phase0_init/get_InitPhaseParam.py`
- `trinity/phase0_init/get_InitCloudProp.py`
- `trinity/phase0_init/__init__.py` (empty — one blank line, no re-exports, no `__all__`)

Declared shared exception, used: `trinity/_functions/unit_conversions.py` (to fix the unit system).
Nothing else was opened. No `trinity/` source, no `docs/dev/`, no other agent output.

**Unit system fixed from `unit_conversions.py`** (the "AU" = astro-unit system):
length **pc**, mass **Msun**, time **Myr**. Derived, read off the conversion constants:

| AU quantity | AU dimensions | evidence |
|---|---|---|
| energy | Msun pc² Myr⁻² | `E_cgs2au` matches `erg → Msun pc²/Myr²` (test list, `unit_conversions.py:569`) |
| luminosity | Msun pc² Myr⁻³ | `L_cgs2au` matches `erg/s → Msun pc²/Myr³` (`:570`) |
| `pdot` / force | Msun pc Myr⁻² | `pdot_cgs2au == F_cgs2au`, dyne (`:518`) |
| number density | pc⁻³ | `ndens_cgs2au = 2.938e55 = (pc/cm)³` (`:88`) |
| velocity | pc Myr⁻¹ | `v_kms2au = 1.0227` (`:109`) |
| `mu_convert` | Msun per particle | unit string `m_H` maps to `m_H[g]·g2Msun` (`:375`) |
| `Mdot_au2Msunyr` | 1e-6 | Msun/Myr → Msun/yr (`:289`) |

Everything below is dimension-checked in that system.

---

## 1. `get_InitPhaseParam.get_y0(params)` — every initial quantity

### 1.1 Inputs read from `params` (lines 76–88)

| local | key | AU dimensions |
|---|---|---|
| `mu_convert` | `mu_convert` | Msun/particle |
| `nCore` | `nCore` | pc⁻³ |
| `bubble_xi_Tb` | `bubble_xi_Tb` | dimensionless |
| `tSF` | `tSF` | Myr |
| `sps_f` | `sps_f` | dict of callables |

Only these five are read. `sps_f` is then called four times: `fLmech_W(tSF)`, `fpdot_W(tSF)`,
`fQi(tSF)`, `fLbol(tSF)`. The latter two are **logging-only** (lines 187–196).

### 1.2 Validation and clamps (lines 94–138)

| line | condition | action |
|---|---|---|
| 94 | `tSF < 0` | `raise ValueError` |
| 97 | `nCore <= 0` | `raise ValueError` |
| 100 | `not (0 <= bubble_xi_Tb <= 1)` | `raise ValueError` |
| 115 | `Lmech_W < 1e-100` | **warn, then `Lmech_W = 1e-100`** |
| 119 | `pdot_W < 1e-100` | **warn, then `pdot_W = 1e-100`** |
| 136 | `v0 < 1e-100` | **warn, then `v0 = 1e-100`** |

All three clamps replace a computed value and let execution continue. Note the ordering: the `v0`
clamp (136) fires **after** `Mdot0` (130) has already been formed from the *unclamped-by-v0*
`pdot_W`/`Lmech_W`, so `Mdot0` and `v0` are no longer tied by `Mdot0 = pdot_W/v0` once the `v0`
clamp fires. There is no clamp on, and no validation of, `mu_convert`.

### 1.3 Derived quantities, exact expressions

Written first as coded, then fully substituted down to inputs.

| name | line | as coded | substituted | AU dimensions |
|---|---|---|---|---|
| `Lmech_W` | 111 | `sps_f['fLmech_W'](tSF)` | — | Msun pc² Myr⁻³ |
| `pdot_W` | 112 | `sps_f['fpdot_W'](tSF)` | — | Msun pc Myr⁻² |
| `Mdot0` | 130 | `pdot_W**2 / (2.0 * Lmech_W)` | — | Msun Myr⁻¹ |
| `v0` | 134 | `2.0 * Lmech_W / pdot_W` | — | pc Myr⁻¹ |
| `rhoa` | 146 | `nCore * mu_convert` | — | Msun pc⁻³ |
| `dt_phase0` | 151 | `np.sqrt(3.0*Mdot0/(4.0*np.pi*rhoa*v0**3))` | `sqrt(3/(64π)) · pdot_W^{5/2} · (nCore·mu_convert)^{-1/2} · Lmech_W^{-2}` | Myr |
| `t0` | 160 | `tSF + dt_phase0` | | Myr |
| `r0` | 163 | `v0 * dt_phase0` | `sqrt(3/(16π)) · pdot_W^{3/2} · (nCore·mu_convert)^{-1/2} · Lmech_W^{-1}` | pc |
| `E0` | 167 | `(5.0/11.0) * Lmech_W * dt_phase0` | `(5/11)·sqrt(3/(64π)) · pdot_W^{5/2} · (nCore·mu_convert)^{-1/2} · Lmech_W^{-1}` | Msun pc² Myr⁻² |
| `T0` | 172–176 | see §1.5 | | K |

Numeric prefactors of the substituted forms: `sqrt(3/(64π)) = 0.1221528…`,
`sqrt(3/(16π)) = 0.2443056…`, `(5/11)·sqrt(3/(64π)) = 0.05552…`.

**Dimension derivations (each done from the arithmetic, not assumed):**

- `Mdot0`: (Msun pc Myr⁻²)² / (Msun pc² Myr⁻³) = Msun² pc² Myr⁻⁴ · Msun⁻¹ pc⁻² Myr³ = **Msun Myr⁻¹**. ✓ mass rate.
- `v0`: (Msun pc² Myr⁻³)/(Msun pc Myr⁻²) = **pc Myr⁻¹**. ✓ velocity.
- `rhoa`: pc⁻³ · Msun = **Msun pc⁻³**. ✓ mass density.
- `dt_phase0`: inside the sqrt, (Msun Myr⁻¹)/(Msun pc⁻³ · pc³ Myr⁻³) = (Msun Myr⁻¹)/(Msun Myr⁻³) = Myr²; sqrt → **Myr**. ✓
- `r0`: pc Myr⁻¹ · Myr = **pc**. ✓
- `E0`: Msun pc² Myr⁻³ · Myr = **Msun pc² Myr⁻²**. ✓ AU energy.

The pair `(Mdot0, v0)` is the exact inverse of the wind relations `Lmech = ½ Mdot v²`,
`pdot = Mdot v`: `pdot²/(2L) = Mdot` and `2L/pdot = v`. So `v0` is the **wind terminal velocity**
and `Mdot0` the **wind mass-loss rate**, both implied by the SPS table's `L` and `pdot` under the
`½Mdot v²` convention. `Mdot0` is used only in `dt_phase0` (151) and one log line (195); it is
**not returned**.

### 1.4 The free-streaming / initial-radius calculation (line 151, 163)

```
dt_phase0 = np.sqrt(3.0 * Mdot0 / (4.0 * np.pi * rhoa * v0**3))
r0        = v0 * dt_phase0
```

Every prefactor, exact: numerator coefficient `3.0`; denominator coefficient `4.0 * np.pi`;
`v0` raised to the **3rd** power; the whole ratio raised to the **1/2** power (`np.sqrt`).

**Implied physical balance.** Solve for the time at which the ambient mass swept by a sphere
expanding ballistically at `v0` equals the wind mass ejected so far:

- swept ambient mass at radius `r = v0 t`, at *uniform* density `rhoa`:  `M_sw = (4/3)π (v0 t)³ ρa`
- wind mass ejected in time `t`: `M_w = Mdot0 · t`

Setting `M_sw = M_w`:  `(4/3)π ρa v0³ t³ = Mdot0 t` ⟹ `t² = 3 Mdot0 / (4π ρa v0³)` ⟹
`t = sqrt(3 Mdot0 /(4π ρa v0³))`. **Exactly the coded expression, prefactor for prefactor.**
So `dt_phase0` is the classical free-expansion/free-streaming timescale (swept mass = ejected
mass), and `r0 = v0·dt_phase0 = sqrt(3 Mdot0/(4π ρa v0))` is the free-streaming radius.

Two assumptions are baked into the algebra and are worth naming because neither is checked:

1. The ambient density is taken as the **constant core density** `rhoa = nCore·mu_convert`
   (line 146), not the density of the actual profile at `r0`. This is self-consistent only while
   `r0 ≤ rCore`. Nothing in this function knows `rCore` or `rCloud`.
2. `nCore` is read from `params` (line 77). `get_InitCloudProp` **mutates** `params['nCore']`
   in one of its correction branches (`get_InitCloudProp.py:230`), so the value used here depends
   on call order across the two modules (see §5).

**Cross-check of the initial state's internal consistency (my own numeric probe, not from the
code):** for `Lmech_W = 1e39 erg/s`, `v_wind = 2000 km/s`, `nCore = 1e3 cm⁻³`, `mu = 1.4 m_H`, the
code gives `rhoa = 34.6 Msun/pc³`, `Mdot0 = 7.9e-4 Msun/yr`, `v0 = 2045 pc/Myr`,
`dt_phase0 = 2.53e-5 Myr`, `r0 = 5.17e-2 pc`, `E0 = 3.63e47 erg`, `T0 = 6.7e7 K`.
The Weaver energy-driven similarity radius at the same `t`,
`R = (250/308π)^{1/5}(L t³/ρ)^{1/5}`, is `4.57e-2 pc` — so the free-streaming `r0` and the
Weaver `R(dt_phase0)` agree to **13%**, i.e. the hand-off point is roughly (not exactly)
self-consistent. Reported as a measurement, not a finding.

### 1.5 `T0` (lines 172–176) — transcribed literally

```
T0 = 1.51e6
     * (Lmech_W * cvt.L_au2cgs / 1e36) ** (8.0/35.0)
     * (nCore   * cvt.ndens_au2cgs)    ** (2.0/35.0)
     * (dt_phase0)                     ** (-6.0/35.0)
     * (1.0 - bubble_xi_Tb)            ** 0.4
```

- `WEAVER_TEMP_COEFFICIENT = 1.51e6` (line 32), bare multiplicative prefactor, carries the K.
- `WEAVER_L_REF = 1e36` (line 35), divides the **cgs** luminosity → dimensionless `L36`.
- `nCore * ndens_au2cgs` → cm⁻³, i.e. the factor is `n0` in cm⁻³, raised to `2/35`.
- `dt_phase0` enters **bare**, in AU (= Myr), raised to `-6/35`.
- `(1 - bubble_xi_Tb)` raised to `0.4` (= 2/5). `bubble_xi_Tb` is validated to `[0,1]` at line 100,
  so the base is in `[0,1]` and the power is well defined (at `bubble_xi_Tb == 1` exactly,
  `0.0**0.4 = 0.0` → `T0 = 0`, no guard).

**Dimensional character.** Factors 1 and 2 are made explicitly dimensionless by dividing/converting
to named units. Factor 3 is **not** — `dt_phase0**(-6/35)` carries `Myr^{-6/35}`, and the formula is
only correct because the AU time unit happens to be Myr (matching a `t6 = t/10⁶ yr` fitting
formula). Formally the line is dimensionally inhomogeneous; practically it is right *today*. See
finding S3-A-02: the asymmetry (two factors explicitly normalised, one implicitly) is the exact
shape of a latent units bug.

`T0` is evaluated at `t = dt_phase0` (elapsed since `tSF`), not at `t0 = tSF + dt_phase0` — i.e.
time is measured from wind onset, consistent with `E0 = (5/11)·L·dt_phase0` also using `dt_phase0`.
That pairing is internally consistent.

### 1.6 Return

`return t0, r0, v0, E0, T0` (line 198). A 5-tuple, positional, no names. `Mdot0`, `rhoa`,
`Lmech_W`, `pdot_W` are **not** returned; `Qi_tSF` and `Lbol_tSF` (187–188) are computed only to be
logged.

---

## 2. Fractional coefficient table (transcribed exactly as coded)

### `get_InitPhaseParam.py`

| line | fraction / constant **as written** | multiplies or divides | power / root applied |
|---|---|---|---|
| 28 → 167 | `5.0 / 11.0` (`WEAVER_ENERGY_FRACTION`) | `Lmech_W * dt_phase0` | linear (exponent 1) |
| 130 | `2.0` in the **denominator** | `pdot_W**2` | `pdot_W` raised to **2** |
| 134 | `2.0` in the **numerator** | `Lmech_W / pdot_W` | linear |
| 151 | `3.0` numerator, `4.0 * np.pi` denominator ⇒ `3/(4π)` | `Mdot0 / (rhoa * v0**3)` | `v0` to the **3**; whole quotient to the **1/2** (`np.sqrt`) |
| 173 | `8.0 / 35.0` | `(Lmech_W * L_au2cgs / 1e36)` | **exponent** |
| 174 | `2.0 / 35.0` | `(nCore * ndens_au2cgs)` | **exponent** |
| 175 | `-6.0 / 35.0` | `dt_phase0` | **exponent** (negative) |
| 176 | `0.4` (written as a decimal, = 2/5) | `(1.0 - bubble_xi_Tb)` | **exponent** |

Note line 176 is the only exponent in the block written as a decimal rather than an explicit
`a.0/b.0` fraction. `8/35 + 2/35 + (−6)/35` are all over the same denominator 35; `0.4 = 14/35`
would keep the family consistent but is coded as `0.4`.

### `get_InitCloudProp.py`

| line | fraction / constant **as written** | multiplies or divides | power / root |
|---|---|---|---|
| 177, 207, 249 | `alpha` | `(rCloud / rCore)` | **exponent** (parameter) |
| 188 | `1.0 / alpha` | `(nCore / nISM)` | **exponent** |
| 219 | `-alpha` | `(rCloud / rCore)` | **exponent** |
| 242 | `0.5` | `rCloud` | linear |
| 261 | `4.0 / 3.0` | `np.pi * rCloud**3 * rhoCore` | `rCloud` to the **3** |
| 263 | `4.0` × `np.pi` | the whole bracket × `rhoCore` | linear |
| 264 | `/ 3.0` | `rCore**3` | `rCore` to the **3** |
| 265 | `3.0 + alpha` | exponent of `rCloud` and of `rCore` | **exponent**, twice |
| 266 | `1 / (3.0 + alpha)` | `(rCloud**(3+α) − rCore**(3+α))` | linear divisor |
| 266 | `alpha` | exponent of `rCore` in the divisor | **exponent** |
| 349 | `/ 1.0e5` | `be_result.c_s` | linear (cm/s → km/s) |
| 443 | `1.5` | `rCloud` | linear (outer grid extent) |

**Line 263–267, the power-law mass integral, checked by hand:**

```
M = 4π ρ_core [ rCore³/3 + (rCloud^(3+α) − rCore^(3+α)) / ((3+α) · rCore^α) ]
```
equals `∫₀^{rCore} 4πr²ρ_core dr + ∫_{rCore}^{rCloud} 4πr² ρ_core (r/rCore)^α dr`
= `4πρ_core[ rCore³/3 + rCore^{-α}(rCloud^{3+α} − rCore^{3+α})/(3+α) ]`. **Consistent — every
prefactor matches.** It is singular at `α = −3` (see S3-A-07).

---

## 3. `get_InitCloudProp` — control flow

### 3.1 `get_InitCloudProp` (89–142)

1. `_validate_params(params)` (123).
2. `profile_type = params['dens_profile'].value` (125).
3. `'densPL'` → `_init_powerlaw_cloud`; `'densBE'` → `_init_bonnor_ebert_cloud`;
   **else** `raise ValueError` (127–132). No default, no case-normalisation.
4. Three **guarded** writes (135–140): `initial_cloud_r_arr` / `_n_arr` / `_m_arr` are set
   *only if the key already exists*. If absent, silently skipped — no error, no warning.
5. `return props`.

### 3.2 `_validate_params` (380–409)

- `required = ['dens_profile','mCloud','nCore','nISM','mu_convert','rCore']`; each must be present
  and non-`None`.
- Positivity enforced for **`mCloud`, `nCore`, `rCore` only**. `nISM` and `mu_convert` are checked
  for presence/non-`None` but **not** for sign or magnitude.
- `densPL` → requires key `densPL_alpha` to exist (value not checked for `None` or type).
- `densBE` → requires `densBE_Omega` and `gamma_adia` to exist. **`densBE_Teff` is not required**,
  yet `_init_bonnor_ebert_cloud:342` writes to it unconditionally.

### 3.3 `_init_powerlaw_cloud` (149–296) — decision table

`rCloud` initial (169–174):
- `alpha == 0` → `compute_rCloud_homogeneous(mCloud, nCore, mu)`
- else → `compute_rCloud_powerlaw(mCloud, nCore, alpha, rCore, mu)` (second return discarded)

`nEdge` (177): `nCore * (rCloud/rCore)**alpha if alpha != 0 else nCore` — the two arms are
numerically identical (`x**0 == 1`).

**Correction block, entered only when `nEdge < nISM and alpha != 0` (180):**

```
rCore_min = rCloud * (nCore / nISM) ** (1.0 / alpha)            # 188
```
This is the exact solution of `nCore·(rCloud/rCore)^α = nISM` for `rCore` **at fixed `rCloud`**.
But `rCloud` itself is a function of `rCore`, so it is a one-step fixed-point guess, not a solution.

| branch | condition | action | which expression ends up used |
|---|---|---|---|
| A | `rCore_min < rCloud` **and** `rCore_try < rCloud_try` (190, 197) | warn; `rCore ← rCore_min`, `rCloud ← rCloud_try`, `params['rCore']` written (206) | `nEdge = nCore·(rCloud_try/rCore_try)^α` (207) |
| B | `rCore_min < rCloud` **and** `rCore_try >= rCloud_try` (197) | `use_nCore_fix = True` (210); `rCloud`/`rCore` left at originals | → branch C body |
| C | `rCore_min >= rCloud` (211) | `use_nCore_fix = True` (212) | → branch C body |

Branch C body (214–249): `nCore_min = nISM*(rCloud/rCore)**(-alpha)` (219) — again the exact
inversion, at the *original* `rCloud`,`rCore`; then `rCore ← rCore_orig`, `nCore ← nCore_min`,
both written to `params` (227–230), `rCloud` recomputed (231–233).
Then a nested repair (236–248): while `rCore >= rCloud`, up to **50** iterations of
`rCore = 0.5*rCloud; rCloud = compute_rCloud_powerlaw(...)`, breaking on `rCore < rCloud`.
Finally `nEdge = nCore*(rCloud/rCore)**alpha` (249).

Post-check (252–256): `if nEdge < nISM:` warn `"Continuing anyway."` — **no raise, no fallback**.

Mass consistency (259–274): `M_check` per §2; `mass_rel_err > 1e-3` → warn `"Continuing with
current values."` — **no raise**.

Writes then grid then profiles (277–286), in that order — see §4.

### 3.4 `_init_bonnor_ebert_cloud` (303–373)

No branches. Straight-line: `solve_lane_emden()` → `create_BE_sphere(...)` → unpack
`r_out, n_out, T_eff, xi_out` → four `params` writes → `_ensure_be_params_exist` →
four more `params` writes → grid → profiles → `CloudProperties(..., T_eff, xi_out)`.
`nISM` is required by `_validate_params` but never used on this path.
`rCore` is read (318), written back unchanged (340), and used **only** to seed the radius grid.
The only arithmetic on this path is `be_result.c_s / 1.0e5` (349).

### 3.5 `_create_radius_array` (412–455)

```
r_min     = 1e-3                                                    # 437, pc, hardcoded
r_inside  = logspace(log10(1e-3),  log10(rCloud),      1000)        # 440
r_outside = logspace(log10(rCloud), log10(1.5*rCloud),  100)        # 443
r_arr     = concatenate([[1e-10], r_inside, r_outside])             # 446–450
r_arr     = sort(unique(append(r_arr, [rCore, rCloud])))            # 453
```
No branches, no clamps. `n_inside=1000`, `n_outside=100` are defaults never overridden by either
caller. `np.sort` after `np.unique` is redundant (`np.unique` returns sorted output).
Result length ≈ 1101–1102.

### 3.6 `verify_mass_at_rCloud` (485–518) / `verify_key_radii_in_array` (521–542)

`searchsorted` (side='left') → if `idx < len` **and** `isclose(r_arr[idx], rCloud)` use `M_arr[idx]`,
**else** fall back to `np.interp`. `rel_error > 0.01` → warn only; the value is returned either way.
No guard on `mCloud == 0` in `verify_mass_at_rCloud` (511) even though it is a module-level public
function. `verify_key_radii_in_array` logs `"rCloud and rCore are both exactly in r_arr"` while the
test is `np.isclose` (default `rtol=1e-5`), i.e. "exactly" is a 1e-5 relative tolerance.
Neither function is called anywhere in this module outside `__main__`.

### 3.7 Exception handlers

**There are none.** No `try`/`except` anywhere in either file (the only `try` is in
`unit_conversions.py`, out of slice). Every failure path is either a `raise ValueError` from
validation, a `logger.warning` + continue, or an unhandled propagation
(`KeyError`, `ZeroDivisionError`).

---

## 4. Ordering and shared state

### 4.1 `params` writes, in execution order

`_init_powerlaw_cloud`:

| order | line | key | value | conditional? |
|---|---|---|---|---|
| 1 | 206 | `rCore` | `rCore_try = rCore_min` | branch A only |
| 2 | 229 | `rCore` | `rCore_orig` | branch B/C only |
| 3 | 230 | `nCore` | `nCore_min` | branch B/C only |
| 4 | 248 | `rCore` | `0.5 × (intermediate rCloud)` | branch B/C **and** `rCore >= rCloud` |
| 5 | 277 | `rCloud` | final `rCloud` | always |
| 6 | 278 | `rCore` | final `rCore` | always |
| 7 | 279 | `nEdge` | final `nEdge` | always |

`_init_bonnor_ebert_cloud`: `rCloud` (339) → `rCore` (340, **no-op**, same value read at 318) →
`nEdge` (341) → `densBE_Teff` (342) → `_ensure_be_params_exist` (345) → `densBE_sigma` (349) →
`densBE_xi_out` (352) → `densBE_f_rho_rhoc` (353) → `densBE_f_m` (354).

`get_InitCloudProp`: `initial_cloud_r_arr` (136) → `_n_arr` (138) → `_m_arr` (140), each guarded.

### 4.2 Read-before-write

**Within the module: none.** `get_density_profile(r_arr, params)` and `get_mass_profile(r_arr,
params)` are called at 285–286 / 360–361, i.e. **after** all `rCloud`/`rCore`/`nEdge`/`nCore`
writes on both paths. The one ordering subtlety is that `_ensure_be_params_exist` (345) runs
*after* `params['densBE_Teff'].value = T_eff` (342) — and it does **not** create `densBE_Teff`,
so that key must pre-exist or line 342 raises `KeyError`. The four keys it *does* create are all
written after it, so that part of the ordering is fine.

**Across the slice: one real coupling.** `get_y0` reads `params['nCore']` (line 77) and
`params['mu_convert']` (76) to build `rhoa`. `_init_powerlaw_cloud` may **overwrite**
`params['nCore']` (line 230) with `nCore_min`, which is strictly **larger** than the user's value
whenever that branch fires. Whether `get_y0` sees the corrected or the original `nCore` depends
entirely on the call order of the two functions, which is not visible from this slice. Same for
`rCore` (not read by `get_y0`, but read by the profile modules).

### 4.3 Computed but not stored

- `Mdot0` (`get_InitPhaseParam.py:130`) — the wind mass-loss rate; used in `dt_phase0` and one log
  line, **not returned** and not written to `params`.
- `rhoa` (146) — used once, not stored.
- `Qi_tSF`, `Lbol_tSF` (187–188) — computed **only** to be logged.
- `_init_powerlaw_cloud` discards the second return of `compute_rCloud_powerlaw` at 172, 194, 231,
  243 (`, _`).
- `be_result.is_stable` (363) → only a log string; **never propagated** to `CloudProperties` or
  `params`, so an UNSTABLE BE sphere is indistinguishable downstream from a stable one.

---

## 5. Numeric literal inventory (every bare constant in arithmetic)

### `get_InitPhaseParam.py`

| line | literal | expression it sits in |
|---|---|---|
| 28 | `5.0`, `11.0` | `WEAVER_ENERGY_FRACTION = 5.0/11.0` → `E0` |
| 32 | `1.51e6` | `WEAVER_TEMP_COEFFICIENT`, prefactor of `T0` |
| 35 | `1e36` | `WEAVER_L_REF`, divides `Lmech_W*L_au2cgs` |
| 38,39,40 | `1e-100` ×3 | `MIN_LUMINOSITY`, `MIN_MOMENTUM`, `MIN_VELOCITY` clamp floors |
| 94 | `0` | `tSF < 0` |
| 97 | `0` | `nCore <= 0` |
| 100 | `0`, `1` | `0 <= bubble_xi_Tb <= 1` |
| 130 | `2`, `2.0` | `pdot_W**2 / (2.0*Lmech_W)` |
| 134 | `2.0` | `2.0*Lmech_W/pdot_W` |
| 151 | `3.0`, `4.0`, `3` | `sqrt(3.0*Mdot0/(4.0*π*rhoa*v0**3))` |
| 173 | `8.0`, `35.0` | exponent of the `L36` factor |
| 174 | `2.0`, `35.0` | exponent of the `n0` factor |
| 175 | `6.0`, `35.0` | exponent `-6.0/35.0` of `dt_phase0` |
| 176 | `1.0`, `0.4` | `(1.0 - bubble_xi_Tb)**0.4` |
| 193 | `10` | `np.log10(...)`, logging only |

### `get_InitCloudProp.py`

| line | literal | expression it sits in |
|---|---|---|
| 169,177,180,260 | `0` | `alpha == 0` / `alpha != 0` guards |
| 188 | `1.0` | `(nCore/nISM)**(1.0/alpha)` |
| 219 | — | `nISM*(rCloud/rCore)**(-alpha)` (no literal, sign flip only) |
| 241 | `50` | `for _iter in range(50)` |
| 242 | `0.5` | `rCore = 0.5*rCloud` |
| 261 | `4.0`, `3.0`, `3` | `(4.0/3.0)*π*rCloud**3*rhoCore` |
| 263–266 | `4.0`, `3.0`, `3`, `3.0` | `4π ρ [rCore**3/3.0 + (…**(3.0+α) − …)/((3.0+α)·rCore**α)]` |
| 268 | `0` | `if mCloud > 0 else 0` |
| 269 | `1e-3` | `mass_rel_err > 1e-3` warning threshold |
| 349 | `1.0e5` | `be_result.c_s / 1.0e5` |
| 415,416 | `1000`, `100` | `n_inside`, `n_outside` defaults |
| 437 | `1e-3` | `r_min`, inner edge of the log grid, **pc** |
| 443 | `1.5` | `1.5*rCloud`, outer extent of the ambient grid |
| 447 | `1e-10` | first (innermost) grid point, **pc** |
| 513 | `0.01` | `rel_error > 0.01` warning threshold |
| 514,516 | `100` | percentage formatting, logging only |
| 574,628,629,… | various | `__main__` demo only |

Note the two different mass-error thresholds for the *same* physical check: `1e-3` at line 269
(inside `_init_powerlaw_cloud`) and `1e-2` at line 513 (`verify_mass_at_rCloud`). Neither raises.

---

## 6. Things I checked that are consistent (no finding)

- `Mdot0`/`v0` invert `L = ½Mdot v²`, `pdot = Mdot v` exactly (§1.3).
- `dt_phase0` reproduces the swept-mass = ejected-mass balance prefactor-for-prefactor (§1.4).
- The power-law mass integral at 263–267 matches the analytic integral term for term (§2).
- `rCore_min` (188) and `nCore_min` (219) are both exact inversions of
  `nEdge = nCore(rCloud/rCore)^α = nISM`, each for its own unknown.
- `T0`'s `L36` and `n0` factors are correctly converted to cgs before normalisation
  (`L_au2cgs`, `ndens_au2cgs`).
- `np.log10(Qi_tSF * cvt.s2Myr)` (193) correctly converts a per-Myr rate to per-second.
- `Mdot0 * cvt.Mdot_au2Msunyr` (195) correctly converts Msun/Myr → Msun/yr (`1e-6`).
- `E0` uses AU luminosity with no cgs conversion — correct, since the result is an AU energy.
- All `params` writes precede the `get_density_profile`/`get_mass_profile` reads (§4.2).
- `r0` and the Weaver similarity radius at the same `t` agree to ~13% for a representative
  config — the hand-off is roughly self-consistent (§1.4).

---

## 7. Findings

```json
[
  {
    "id": "S3-A-01",
    "file": "trinity/phase0_init/get_InitCloudProp.py",
    "line": 349,
    "class": "units",
    "severity": "S3",
    "claim": "params['densBE_sigma'] is stored in km/s (be_result.c_s / 1.0e5, i.e. cm/s -> km/s) while every other velocity in the code base is in AU (pc/Myr). km/s and pc/Myr differ by only 2.27% (v_kms2au = 1.0227), so a downstream consumer that assumes AU is off by 2.3% in a way no sanity check would catch.",
    "evidence": "get_InitCloudProp.py:349 `params['densBE_sigma'].value = be_result.c_s / 1.0e5`. The only unit hint is the info string at get_InitCloudProp.py:471 ('...sigma = c_s [km/s]'). unit_conversions.py:109 gives v_kms2au = 1.022712165045695, so km/s and pc/Myr are within 2.27%.",
    "expected": "Either store in AU (`be_result.c_s * cvt.v_cms2au`) like every other velocity, or the consumer must multiply by cvt.v_kms2au. One of the two is happening; from this slice I cannot tell which.",
    "failure_scenario": "A densBE run's sound speed / velocity dispersion is used 2.27% low (or the pressure derived from it 4.6% low) everywhere it is consumed, shifting the BE sphere's derived quantities by a few percent with no error and no warning.",
    "repro": "grep for densBE_sigma consumers; check whether any arithmetic combines it with a pc/Myr quantity without a v_kms2au factor.",
    "confidence": "medium"
  },
  {
    "id": "S3-A-02",
    "file": "trinity/phase0_init/get_InitPhaseParam.py",
    "line": 175,
    "class": "units",
    "severity": "S3",
    "claim": "In the T0 expression the luminosity and density factors are explicitly normalised to named units (÷1e36 after L_au2cgs; ×ndens_au2cgs to cm^-3) but the time factor `dt_phase0**(-6.0/35.0)` is used bare, with no explicit normalising constant. The line is only correct because the AU time unit happens to be Myr; the expression is formally dimensionally inhomogeneous.",
    "evidence": "get_InitPhaseParam.py:172-176. Lines 173 and 174 carry explicit conversions/normalisations; line 175 carries none. `dt_phase0` is in Myr (get_InitPhaseParam.py:151, dimension-checked in §1.3).",
    "expected": "Symmetric treatment: a `WEAVER_T_REF = 1.0  # Myr` divisor on line 175, matching WEAVER_L_REF on line 173, so the unit assumption is stated in the arithmetic rather than implied.",
    "failure_scenario": "If the AU time unit is ever changed (or this formula is copied into a routine whose time is in Myr-years/seconds), T0 silently scales by (unit ratio)^(-6/35) while the L and n factors stay correct — a wrong-but-plausible temperature with no dimensional tripwire.",
    "repro": "",
    "confidence": "medium"
  },
  {
    "id": "S3-A-03",
    "file": "trinity/phase0_init/get_InitCloudProp.py",
    "line": 242,
    "class": "state",
    "severity": "S3",
    "claim": "In the nCore-correction repair loop the user's rCore is replaced by `0.5 * rCloud` — an arbitrary geometric value with no relation to the nEdge/nISM condition that triggered the repair — and that value is written back to params['rCore'] (line 248) and to params again at line 278.",
    "evidence": "get_InitCloudProp.py:241-248: `for _iter in range(50): rCore = 0.5*rCloud; rCloud, _ = compute_rCloud_powerlaw(...); if rCore < rCloud: break` then `params['rCore'].value = rCore`. Note rCore is set to half the *previous* iterate's rCloud, so the stored rCore is 0.5 x an intermediate value that is not the final rCloud either.",
    "expected": "Either solve the coupled (rCore, rCloud) fixed point properly, or raise and tell the user their (mCloud, nCore, rCore, alpha, nISM) combination is inconsistent, rather than substituting an unrelated rCore.",
    "failure_scenario": "A sweep point silently runs with an rCore roughly half the cloud radius instead of the ~1 pc the .param specified; the whole density profile, and hence every downstream number, is that of a different cloud. Only a warning at line 237 distinguishes it.",
    "repro": "Construct a densPL .param with alpha<0 where nEdge<nISM and rCore_min>=rCloud (e.g. low mCloud, high nISM), run get_InitCloudProp, compare params['rCore'] before and after.",
    "confidence": "high"
  },
  {
    "id": "S3-A-04",
    "file": "trinity/phase0_init/get_InitCloudProp.py",
    "line": 241,
    "class": "silent-failure",
    "severity": "S3",
    "claim": "The 50-iteration rCore repair loop has no post-loop convergence check. If it exhausts all 50 iterations without satisfying `rCore < rCloud`, execution continues with rCore >= rCloud — a cloud whose core is larger than the cloud — and the very next line writes that rCore to params.",
    "evidence": "get_InitCloudProp.py:241-248. The only exit signal is the `break` at 247; nothing distinguishes 'broke early' from 'ran out of iterations'. `_iter` is unused after the loop. (For alpha<0 the loop provably breaks on the first pass, since reducing rCore lowers the density and raises rCloud — so range(50) also carries 49 iterations of unreachable capacity.)",
    "expected": "`else: raise ValueError(...)` on the for-loop, or at minimum a warning that the repair did not converge.",
    "failure_scenario": "For a profile shape where reducing rCore does not increase rCloud, the loop silently exits with an inconsistent geometry that get_density_profile/get_mass_profile then evaluate on.",
    "repro": "",
    "confidence": "high"
  },
  {
    "id": "S3-A-05",
    "file": "trinity/phase0_init/get_InitCloudProp.py",
    "line": 252,
    "class": "silent-failure",
    "severity": "S3",
    "claim": "After both correction strategies, if nEdge is still < nISM the code logs a warning and explicitly continues ('Continuing anyway.'), initialising a cloud whose edge density is below the ambient medium it is embedded in.",
    "evidence": "get_InitCloudProp.py:252-256. No raise, no fallback, no flag set on CloudProperties or params to record that the correction failed.",
    "expected": "Raise, or record the failure in state so downstream/output can see the run started from an unconverged cloud.",
    "failure_scenario": "A batch sweep produces results for configurations whose initial cloud is physically inconsistent; because only a log line marks it, the affected points are indistinguishable in the output files.",
    "repro": "",
    "confidence": "high"
  },
  {
    "id": "S3-A-06",
    "file": "trinity/phase0_init/get_InitCloudProp.py",
    "line": 269,
    "class": "silent-failure",
    "severity": "S3",
    "claim": "The mass-consistency check compares the analytic M(rCloud) against the requested mCloud and, on a relative error above 1e-3, only logs a warning ('Continuing with current values.'). The run then proceeds with an rCloud that does not enclose mCloud.",
    "evidence": "get_InitCloudProp.py:268-274. Note also a second, looser threshold (0.01) for the same physical quantity in verify_mass_at_rCloud (get_InitCloudProp.py:513), and that verify_mass_at_rCloud is not called on the production path.",
    "expected": "A hard failure, since mCloud is the primary user input for the run; or at minimum one consistent threshold.",
    "failure_scenario": "Total cloud mass silently differs from the .param value; every mass-dependent downstream quantity (shell mass, gravity, column density) is off by the same factor with no output-visible marker.",
    "repro": "",
    "confidence": "high"
  },
  {
    "id": "S3-A-07",
    "file": "trinity/phase0_init/get_InitCloudProp.py",
    "line": 266,
    "class": "numerical",
    "severity": "S4",
    "claim": "The power-law mass check divides by `(3.0 + alpha)` with no guard, so densPL_alpha == -3.0 raises an uncaught ZeroDivisionError during initialisation.",
    "evidence": "get_InitCloudProp.py:263-267: `(rCloud**(3.0+alpha) - rCore**(3.0+alpha)) / ((3.0+alpha)*rCore**alpha)`. _validate_params (380-409) only checks that densPL_alpha exists; it does not check its value.",
    "expected": "Either a guard/limit form for alpha == -3 (where the integral becomes logarithmic) or an explicit validation rejecting alpha == -3 with a clear message.",
    "failure_scenario": "A parameter sweep that walks alpha through -3 crashes with a bare ZeroDivisionError from a mass sanity check, with no indication that alpha is the culprit.",
    "repro": "Set densPL_alpha = -3 in a densPL .param and call get_InitCloudProp.",
    "confidence": "high"
  },
  {
    "id": "S3-A-08",
    "file": "trinity/phase0_init/get_InitPhaseParam.py",
    "line": 115,
    "class": "silent-failure",
    "severity": "S3",
    "claim": "The three 1e-100 clamps (MIN_LUMINOSITY, MIN_MOMENTUM, MIN_VELOCITY) replace computed values and let the function return a fabricated initial state that passes every subsequent check. If both Lmech_W and pdot_W are clamped, v0 = 2.0*1e-100/1e-100 = exactly 2.0 pc/Myr — a number with no physical origin that then survives the v0 >= MIN_VELOCITY test.",
    "evidence": "get_InitPhaseParam.py:115-138. I evaluated the clamped path numerically with rhoa = 34.6 Msun/pc^3: both clamped gives v0 = 2.0 pc/Myr, dt_phase0 = 2.08e-52 Myr, r0 = 4.15e-52 pc, E0 = 9.4e-153, T0 = 8.5e-10 K. The v0-clamped path gives dt_phase0 = 2.34e+150 Myr, i.e. t0 = tSF + 2.3e150 Myr. All are returned without error.",
    "expected": "If the SPS tables give no wind feedback at tSF, that is a configuration error, not a number to floor — raise, or return a documented 'no phase 0' sentinel, rather than returning r0 ~ 1e-52 pc / t0 ~ 1e150 Myr.",
    "failure_scenario": "An SPS table with zero wind luminosity at t=tSF (or a tSF before wind onset) starts the integration from a degenerate state instead of failing; the solver then works from r0 = 4e-52 pc and T0 = 1e-9 K.",
    "repro": "Call get_y0 with an sps_f whose fLmech_W and fpdot_W return 0.0 at tSF and inspect the returned tuple.",
    "confidence": "high"
  },
  {
    "id": "S3-A-09",
    "file": "trinity/phase0_init/get_InitCloudProp.py",
    "line": 437,
    "class": "numerical",
    "severity": "S3",
    "claim": "The radius grid has a single interval spanning seven decades, from 1e-10 pc straight to r_min = 1e-3 pc, with no points in between. Any linear interpolation of n_arr or M_arr at a radius inside that gap is grossly wrong: for a constant-density core M ~ r^3, so linear interpolation at r = 1e-4 pc over-estimates the enclosed mass by ~100x.",
    "evidence": "get_InitCloudProp.py:437 `r_min = 1e-3`; :440 logspace starts at r_min; :447 the array's first element is the isolated literal 1e-10. Both r_min and 1e-10 are hardcoded and independent of rCore. r0 from get_y0 is ~5e-2 pc for a representative config but scales as pdot^{3/2}/(sqrt(rho) L), so weak-feedback / high-density configurations can put r0 below 1e-3 pc.",
    "expected": "Either fill the inner decades (make the innermost point a small fraction of rCore rather than a fixed 1e-10 pc) or clamp/validate that the first radius the solver asks for lies above r_min.",
    "failure_scenario": "For a weak-wind or very dense configuration whose free-streaming radius r0 < 1e-3 pc, the initial swept-up mass read off M_arr is over-estimated by orders of magnitude, biasing the entire trajectory from step zero.",
    "repro": "np.interp(1e-4, r_arr, M_arr) vs the analytic (4/3)pi r^3 rho_core for a densPL cloud.",
    "confidence": "medium"
  },
  {
    "id": "S3-A-10",
    "file": "trinity/phase0_init/get_InitCloudProp.py",
    "line": 453,
    "class": "numerical",
    "severity": "S4",
    "claim": "Because rCloud is re-derived through 10**log10(rCloud) inside logspace and then also appended verbatim, the grid contains pairs of radii separated by ~1 ulp for generic rCloud values.",
    "evidence": "get_InitCloudProp.py:440,443,453. Reproduced numerically: for rCloud = 19.87 the grid contains both 19.869999999999997 and 19.87 (relative gap 1.79e-16); for rCloud = 23.456789 it contains 23.456789 and 23.456789000000004 (1.52e-16). For rCloud = 100.0 or 5.0 (exact powers/short binary fractions) no duplicate appears — so the defect is value-dependent.",
    "expected": "np.unique with a relative tolerance, or build the grid so rCloud is inserted once. Note np.sort at line 453 is also redundant, since np.unique already returns sorted output.",
    "failure_scenario": "Any consumer that forms a finite difference or divides by dr across that pair divides by ~3.5e-15 pc; any consumer that assumes a strictly monotone, well-separated grid sees a near-degenerate cell right at the cloud edge — the most physically important radius in the array.",
    "repro": "python -c \"import numpy as np; r=np.sort(np.unique(np.append(np.concatenate([[1e-10],np.logspace(-3,np.log10(19.87),1000),np.logspace(np.log10(19.87),np.log10(1.5*19.87),100)]),[1.0,19.87]))); d=np.diff(r)/r[:-1]; print(d.min())\"",
    "confidence": "high"
  },
  {
    "id": "S3-A-11",
    "file": "trinity/phase0_init/get_InitCloudProp.py",
    "line": 230,
    "class": "state",
    "severity": "S3",
    "claim": "params['nCore'] is overwritten with nCore_min (strictly larger than the user's value) in the correction branch. get_InitPhaseParam.get_y0 reads params['nCore'] to build rhoa, so whether phase 0 uses the user's nCore or the corrected one depends entirely on the call order of the two functions.",
    "evidence": "get_InitCloudProp.py:228-230 `nCore = nCore_min; ...; params['nCore'].value = nCore`. get_InitPhaseParam.py:77 `nCore = params['nCore'].value`; :146 `rhoa = nCore * mu_convert`; :151 rhoa enters dt_phase0 as rhoa^{-1/2}. get_y0 also uses nCore directly in the T0 exponent (get_InitPhaseParam.py:174).",
    "expected": "Either the corrected nCore is the intended one everywhere (then the ordering should be enforced/asserted, not implicit), or the correction should not mutate the shared input at all.",
    "failure_scenario": "If get_y0 runs before get_InitCloudProp, phase 0 uses the uncorrected nCore while the cloud profile uses the corrected one: r0 and dt_phase0 are computed against a density the cloud no longer has, and the mismatch scales as (nCore_corrected/nCore_user)^{1/2}.",
    "repro": "Trace the call order of get_y0 and get_InitCloudProp in the run driver; log params['nCore'] at both call sites for a config that triggers the nCore fix.",
    "confidence": "medium"
  },
  {
    "id": "S3-A-12",
    "file": "trinity/phase0_init/get_InitCloudProp.py",
    "line": 220,
    "class": "other",
    "severity": "S4",
    "claim": "The nCore-fix warning always says the reason is 'cannot fix with rCore alone (rCore_min=... >= rCloud=...)', but that path is also reached from line 210, where rCore_min < rCloud held and the branch was taken because rCore_try >= rCloud_try instead. In that case the message states a false condition and prints two numbers that contradict it.",
    "evidence": "get_InitCloudProp.py:190 `if rCore_min < rCloud:` -> 197 `if rCore_try < rCloud_try:` else 210 `use_nCore_fix = True`; the shared warning at 220-226 hardcodes the rCore_min >= rCloud explanation.",
    "expected": "Two distinct messages, or a message that names which test failed.",
    "failure_scenario": "A user debugging an unexpected nCore change is told rCore_min >= rCloud while the printed numbers show rCore_min < rCloud, and looks in the wrong place.",
    "repro": "",
    "confidence": "high"
  },
  {
    "id": "S3-A-13",
    "file": "trinity/phase0_init/get_InitCloudProp.py",
    "line": 268,
    "class": "deadcode",
    "severity": "S4",
    "claim": "`mass_rel_err = abs(M_check - mCloud)/mCloud if mCloud > 0 else 0` — the else arm is unreachable, since _validate_params already raises for mCloud <= 0.",
    "evidence": "get_InitCloudProp.py:268 vs get_InitCloudProp.py:394-395 `if params['mCloud'].value <= 0: raise ValueError(...)`, called at :123 before either init function.",
    "expected": "Drop the ternary.",
    "failure_scenario": "",
    "repro": "",
    "confidence": "high"
  },
  {
    "id": "S3-A-14",
    "file": "trinity/phase0_init/get_InitCloudProp.py",
    "line": 177,
    "class": "deadcode",
    "severity": "S4",
    "claim": "`nEdge = nCore * (rCloud/rCore)**alpha if alpha != 0 else nCore` — the two arms are numerically identical, since x**0 == 1 and rCore is validated positive so the division cannot fail.",
    "evidence": "get_InitCloudProp.py:177; rCore > 0 enforced at get_InitCloudProp.py:398-399.",
    "expected": "`nEdge = nCore * (rCloud/rCore)**alpha` unconditionally.",
    "failure_scenario": "",
    "repro": "",
    "confidence": "high"
  },
  {
    "id": "S3-A-15",
    "file": "trinity/phase0_init/get_InitCloudProp.py",
    "line": 340,
    "class": "deadcode",
    "severity": "S4",
    "claim": "`params['rCore'].value = rCore` in the BE path writes back exactly the value read at line 318; rCore is never modified on that path. The write is a no-op that reads as if rCore were a BE output.",
    "evidence": "get_InitCloudProp.py:318 `rCore = params['rCore'].value`; nothing between 318 and 340 assigns rCore; :340 writes it back.",
    "expected": "Remove the write, or state that rCore is a pass-through for the BE profile (it is used only to seed the radius grid at :357 and to populate CloudProperties at :370).",
    "failure_scenario": "",
    "repro": "",
    "confidence": "high"
  },
  {
    "id": "S3-A-16",
    "file": "trinity/phase0_init/get_InitCloudProp.py",
    "line": 342,
    "class": "other",
    "severity": "S4",
    "claim": "`params['densBE_Teff'].value = T_eff` is executed unconditionally, but densBE_Teff is neither in _validate_params' required list nor among the keys _ensure_be_params_exist creates — and _ensure_be_params_exist is called on the following line anyway.",
    "evidence": "get_InitCloudProp.py:342 (write) precedes :345 (_ensure_be_params_exist). _validate_params' densBE branch (get_InitCloudProp.py:405-409) requires only densBE_Omega and gamma_adia. _ensure_be_params_exist (get_InitCloudProp.py:467-472) covers densBE_f_m, densBE_xi_out, densBE_f_rho_rhoc, densBE_sigma — not densBE_Teff.",
    "expected": "Add densBE_Teff to be_params_needed, or move _ensure_be_params_exist above line 339 and include it there.",
    "failure_scenario": "A densBE .param (or a caller-built params dict) lacking densBE_Teff dies with a bare KeyError at line 342 instead of the clear ValueError _validate_params would have raised.",
    "repro": "Run _init_bonnor_ebert_cloud with a params dict omitting 'densBE_Teff' (the module's own __main__ BE test at get_InitCloudProp.py:622-637 has to supply it explicitly for exactly this reason).",
    "confidence": "high"
  },
  {
    "id": "S3-A-17",
    "file": "trinity/phase0_init/get_InitCloudProp.py",
    "line": 24,
    "class": "deadcode",
    "severity": "S4",
    "claim": "`import os` (line 24) and `from pathlib import Path` (line 28) are unused — zero occurrences of `os.` or `Path(` in the module.",
    "evidence": "Counted occurrences in the file: 'os.' -> 0, 'Path(' -> 0. `logging` is additionally re-imported inside __main__ at get_InitCloudProp.py:550 although already imported at :23.",
    "expected": "Remove both imports.",
    "failure_scenario": "",
    "repro": "",
    "confidence": "high"
  },
  {
    "id": "S3-A-18",
    "file": "trinity/phase0_init/get_InitCloudProp.py",
    "line": 135,
    "class": "silent-failure",
    "severity": "S4",
    "claim": "The three initial_cloud_* writes are each guarded by `if <key> in params:`, so if the schema does not define them the arrays are silently not published and any stale prior value stays in place. There is no else-branch and no warning.",
    "evidence": "get_InitCloudProp.py:135-140.",
    "expected": "Either the keys are guaranteed by the schema (then drop the guards) or their absence should warn — a silent skip means a downstream reader gets stale or missing arrays with no signal.",
    "failure_scenario": "A params set missing initial_cloud_n_arr leaves whatever was there before; downstream code reading it operates on the previous run's cloud.",
    "repro": "",
    "confidence": "medium"
  },
  {
    "id": "S3-A-19",
    "file": "trinity/phase0_init/get_InitCloudProp.py",
    "line": 485,
    "class": "deadcode",
    "severity": "S4",
    "claim": "verify_mass_at_rCloud (485) and verify_key_radii_in_array (521) are never called from the production path in this module — only from the __main__ demo block. The production mass check is the separate, inline one at line 268 with a different threshold (1e-3 vs 0.01).",
    "evidence": "Only call sites are get_InitCloudProp.py:587, 588, 614, 615, 645, 646, all inside `if __name__ == '__main__':`. get_InitCloudProp (89-142) calls neither. Also verify_mass_at_rCloud:511 divides by mCloud with no zero guard, unlike the inline check at :268.",
    "expected": "Either call them from get_InitCloudProp (they are the more complete checks, since they test the actual M_arr rather than a re-derived analytic mass), or note them as diagnostics-only. I cannot see other modules from this slice, so they may be called elsewhere.",
    "failure_scenario": "",
    "repro": "grep -rn 'verify_mass_at_rCloud\\|verify_key_radii_in_array' trinity/ test/",
    "confidence": "low"
  },
  {
    "id": "S3-A-20",
    "file": "trinity/phase0_init/get_InitCloudProp.py",
    "line": 443,
    "class": "regime",
    "severity": "S4",
    "claim": "The radius grid hardcodes three geometry constants with no relation to the configuration: the inner bound r_min = 1e-3 pc, the isolated first point 1e-10 pc, and the ambient extent 1.5 * rCloud. If rCloud < 1e-3 pc, np.logspace(log10(1e-3), log10(rCloud), 1000) is monotonically decreasing and the 'inside' points all lie outside the cloud (the later np.sort masks the inversion).",
    "evidence": "get_InitCloudProp.py:437, 440, 443, 447, 453. Verified: for rCloud = 5e-4 the r_inside array runs 1e-3 down to 5e-4, np.all(np.diff(r_inside) > 0) is False. Similarly, if rCore < 1e-3 the cloud core interior is represented by only two points (1e-10 and rCore itself).",
    "expected": "Scale r_min off rCore/rCloud rather than a fixed 1e-3 pc, and/or validate rCloud > r_min.",
    "failure_scenario": "A very compact cloud (or a BE sphere with a sub-milliparsec r_out) gets a grid that is inverted before the sort and essentially unresolved after it; the density and mass profiles evaluated on it are meaningless, with no warning.",
    "repro": "_create_radius_array(5e-4, 1e-4) and inspect np.diff of the pre-sort r_inside.",
    "confidence": "high"
  },
  {
    "id": "S3-A-21",
    "file": "trinity/phase0_init/get_InitCloudProp.py",
    "line": 188,
    "class": "regime",
    "severity": "S4",
    "claim": "The nEdge<nISM correction block's inequality directions only hold for alpha < 0. rCore_min is a lower bound on rCore (and 'rCore_min < rCloud' the right feasibility test) only when nEdge increases with rCore, i.e. alpha < 0. For alpha > 0 the same names and tests describe the opposite bound.",
    "evidence": "get_InitCloudProp.py:188 `rCore_min = rCloud*(nCore/nISM)**(1.0/alpha)`; :190 `if rCore_min < rCloud:`. nEdge = nCore*(rCloud/rCore)**alpha is increasing in rCore iff alpha < 0. _validate_params does not constrain the sign of densPL_alpha.",
    "expected": "Either validate alpha < 0 for densPL, or branch on sign(alpha). In practice the block is largely unreachable for alpha > 0 (nEdge > nCore > nISM), which is why the mislabelling has no visible effect today.",
    "failure_scenario": "A positive-alpha configuration that somehow enters the block gets a 'correction' that pushes nEdge further below nISM, then hits the warn-and-continue at line 252.",
    "repro": "",
    "confidence": "medium"
  },
  {
    "id": "S3-A-22",
    "file": "trinity/phase0_init/get_InitCloudProp.py",
    "line": 386,
    "class": "other",
    "severity": "S4",
    "claim": "_validate_params checks positivity for mCloud, nCore and rCore but not for nISM or mu_convert, although both are in the required list and both are used in division/multiplication that assumes positivity.",
    "evidence": "get_InitCloudProp.py:386-399. nISM is divided into nCore at :188 `(nCore/nISM)**(1.0/alpha)` (ZeroDivisionError if nISM == 0, complex/NaN territory if negative) and multiplied at :219. mu_convert forms rhoCore at :259 and, in the other file, rhoa at get_InitPhaseParam.py:146, which is then square-rooted through dt_phase0.",
    "expected": "Add `nISM > 0` and `mu_convert > 0` to the same positivity block.",
    "failure_scenario": "nISM = 0 is currently masked only because `nEdge < nISM` is then false; a negative mu_convert would give a negative rhoa and dt_phase0 = sqrt(negative) = nan, propagating nan into t0, r0 and E0 with no error.",
    "repro": "",
    "confidence": "high"
  },
  {
    "id": "S3-A-23",
    "file": "trinity/phase0_init/get_InitPhaseParam.py",
    "line": 120,
    "class": "other",
    "severity": "S4",
    "claim": "The two adjacent clamp warnings are inconsistent about units: the Lmech_W warning converts to cgs before printing, the pdot_W warning prints the raw AU value with no unit at all.",
    "evidence": "get_InitPhaseParam.py:116 `f\"Lmech_W={Lmech_W * cvt.L_au2cgs:.3e} erg/s ...\"` vs :120 `f\"pdot_W={pdot_W} is very small ...\"` (no conversion, no unit, no format spec). Likewise :137 `f\"v0={v0} is very small\"` prints AU pc/Myr unlabelled.",
    "expected": "`pdot_W * cvt.pdot_au2cgs` with a 'dyne' label, and `v0 * cvt.v_au2kms` with 'km/s', matching the neighbouring line.",
    "failure_scenario": "",
    "repro": "",
    "confidence": "high"
  },
  {
    "id": "S3-A-24",
    "file": "trinity/phase0_init/get_InitCloudProp.py",
    "line": 363,
    "class": "state",
    "severity": "S4",
    "claim": "be_result.is_stable is consumed only to build a log string; it is not stored on CloudProperties (which carries T_eff and xi_out but not stability) nor written to params. An unstable BE sphere is therefore indistinguishable from a stable one anywhere downstream or in the run output.",
    "evidence": "get_InitCloudProp.py:363-367 uses is_stable for the `stability` string in logger.info only. CloudProperties fields (get_InitCloudProp.py:74-82) are rCloud, rCore, nEdge, r_arr, n_arr, M_arr, T_eff, xi_out.",
    "expected": "Carry is_stable onto CloudProperties or into params so the output metadata records it.",
    "failure_scenario": "A sweep over densBE_Omega crosses the stability boundary; the resulting output files give no way to tell which points started from a gravitationally unstable initial cloud.",
    "repro": "",
    "confidence": "medium"
  },
  {
    "id": "S3-A-25",
    "file": "trinity/phase0_init/get_InitCloudProp.py",
    "line": 585,
    "class": "other",
    "severity": "S4",
    "claim": "The module's __main__ demo is unit-inconsistent with the module it demonstrates: it passes mu_convert = 1.4 and nCore = 1e3 as raw cgs-flavoured numbers rather than AU (mu_convert should be ~1.18e-57 Msun; nCore ~2.9e58 pc^-3), and prints the AU nEdge labelled 'cm^-3' without the ndens_au2cgs factor the production log at line 290 applies.",
    "evidence": "get_InitCloudProp.py:571-574 (mu_convert 1.4, nCore 1e3) vs unit_conversions.py:375 (`m_H` maps to m_H[g]*g2Msun) and :88 (ndens_cgs2au = 2.938e55). get_InitCloudProp.py:585 `print(f\"  nEdge = {props_PL.nEdge:.2e} cm^-3\")` vs :290 `nEdge*cvt.ndens_au2cgs:.2e} cm^-3`.",
    "expected": "Either use AU values in the demo or drop the demo in favour of the pytest suite; as written, 'All tests PASSED!' at line 664 certifies only that the mass integral is self-consistent, in whatever units it was handed.",
    "failure_scenario": "Someone reads the __main__ block as the worked example of how to call get_InitCloudProp and passes cgs values from real code.",
    "repro": "python -m trinity.phase0_init.get_InitCloudProp",
    "confidence": "high"
  }
]
```
