# S2 cloud properties — Lens B (what the code claims)

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

**Method.** I read only the extracted prose (comments + docstrings) for the six files in
`trinity/cloud_properties/`. I have not seen one line of implementation. Everything below is a
*claim the prose makes*, recorded so another lens can test it. Where I write "consistent" I mean
*the prose is internally self-consistent as mathematics* — never that the code does it.

Files in slice:
`density_profile.py`, `mass_profile.py`, `powerLawSphere.py`, `bonnorEbertSphere.py`,
`initial_profile.py`, `validate_gmc.py`.

---

## 1. Formulas claimed

### 1.1 Power-law density profile

Stated identically in three places — `density_profile.py:3`, `density_profile.py:56` (Notes),
`powerLawSphere.py:3`:

```
n(r) = nCore                        r <= rCore
n(r) = nCore * (r/rCore)^alpha      rCore < r <= rCloud
n(r) = nISM                         r > rCloud
```

Restated inline at `density_profile.py:142` (`n = nCore * (r/rCore)^alpha`) and at
`mass_profile.py:567` (`compute_minimum_rCore`). All four statements agree. Continuity at
`r = rCore` is automatic (`(rCore/rCore)^alpha = 1`); no continuity is claimed at `rCloud` —
`nEdge = nCore*(rCloud/rCore)^alpha` need not equal `nISM`, which is exactly what the `nEdge >= nISM`
validation (§5.3) is about.

### 1.2 Power-law enclosed mass

`mass_profile.py:271` (`compute_enclosed_mass_powerlaw` docstring):

alpha = 0:
```
M(r) = (4/3)*pi*r^3*rho_core                                r <= r_cloud
M(r) = M_cloud + (4/3)*pi*(r^3 - r_cloud^3)*rho_ISM         r > r_cloud
```
alpha != 0:
```
M(r) = (4/3)*pi*r^3*rho_core                                                     r <= r_core
M(r) = 4*pi*rho_core [ r_core^3/3 + (r^(3+alpha) - r_core^(3+alpha))
                                     / ((3+alpha)*r_core^alpha) ]                r_core < r <= r_cloud
M(r) = M_cloud + (4/3)*pi*(r^3 - r_cloud^3)*rho_ISM                              r > r_cloud
```

The region-2 form is attributed to **Rahner+ 2018, Eq 25** at `mass_profile.py:331`. The same
formula appears at `powerLawSphere.py:3` (module docstring, "For α ≠ 0"), at
`powerLawSphere.py:78` (`compute_rCloud_powerlaw` docstring, also cited to Rahner+ 2018 Eq 25),
and again as a comment at `powerLawSphere.py:150`. **Four statements, all algebraically identical.**

**No α = −3 special case is stated anywhere in `mass_profile.py`**, even though `(3+alpha)` sits in
the denominator. `powerLawSphere.py:78` *does* state the exclusion (§4.1). See finding S2-B-05.

### 1.3 Cloud radius inversions (`powerLawSphere.py`)

`compute_rCloud_homogeneous` (`powerLawSphere.py:52`), also stated in the module docstring
`powerLawSphere.py:3`:
```
M = (4/3)*pi*r^3*rho   =>   rCloud = (3M / (4*pi*rho))^(1/3)
```

`compute_rCloud_powerlaw` (`powerLawSphere.py:78`), "valid for α ≠ 0, α ≠ −3", no root-finding:
```
rCloud = { [ M/(4*pi*rho_c) - rCore^3/3 ] * (3+alpha) * rCore^alpha + rCore^(3+alpha) }^(1/(3+alpha))
```
Repeated as a comment at `powerLawSphere.py:151`.
*Internally consistent*: this is the exact algebraic inversion of §1.2 region 2 evaluated at
`r = rCloud`. I re-derived it by hand; it checks out as stated.

Fractional-`rCore` branch (`powerLawSphere.py:182`), for `rCore = f * rCloud`:
```
M   = 4*pi*rho_c * rCloud^3 * [ f^3/3 + (1 - f^(3+alpha)) / ((3+alpha)*f^alpha) ]
rCloud = [ M / (4*pi*rho_c*g) ]^(1/3)      with g = the bracket above
```
*Internally consistent*: substituting `rCore = f·rCloud` into §1.2 region 2 reproduces this exactly.
Note the asymmetry: `g > 0` for every `alpha != -3` (both signs of `3+alpha` flip together), so the
fractional branch is unconditionally solvable, whereas the fixed-`rCore` branch is not (§4.2).

### 1.4 Minimum core radius (`mass_profile.py:567`)

Full derivation is spelled out in the docstring:
```
nEdge = nCore * (rCloud/rCore)^alpha
alpha < 0, require nEdge >= nISM:
  nCore*(rCloud/rCore)^alpha >= nISM
  (rCloud/rCore)^alpha       >= nISM/nCore
  "Since alpha < 0, raising to power 1/alpha flips inequality"
  rCloud/rCore <= (nISM/nCore)^(1/alpha)
  rCore        >= rCloud * (nCore/nISM)^(1/alpha)
=> rCore_min = rCloud * (nCore/nISM)^(1/alpha)
```
*Internally consistent*: the inequality flip and the reciprocal-base step are both correct as
written, and for `alpha<0, nCore>nISM` it yields `rCore_min < rCloud` as it should.
Applied margin: `rCore = rCore_min * margin`, `margin` default **1.1** (`mass_profile.py:567`).
alpha = 0 branch (`mass_profile.py:608`): "nEdge = nCore, always valid if nCore > nISM";
"Default: 10% of cloud radius" (`mass_profile.py:609`).
`mass_profile.py:620`: "Ensure rCore doesn't exceed rCloud (pathological case)".
**No branch is described for alpha > 0.** See S2-B-07.

### 1.5 Mass accretion rate

`mass_profile.py:3`, `:223`, `:442`, `:477` — stated four times, identically:
```
dM/dt = dM/dr * dr/dt = 4*pi*r^2 * rho(r) * v(r)
```
Claimed "**Correct formula … for ALL profiles**" (`:3`), "This formula is **EXACT** for any smooth
density profile" (`:442`), "This works for **ALL** density profiles!" (`:478`), and
"**NO SOLVER HISTORY NEEDED** — just instantaneous rho(r) and v(r)" (`:442`).

### 1.6 Mass density conversion

`mass_profile.py:43`, `:87`, `:124`, `powerLawSphere.py:39`, `bonnorEbertSphere.py:400`:
```
rho [Msun/pc^3] = n [1/pc^3] * mu_convert [Msun]      "No additional conversion factor is needed."
rho_cgs [g/cm^3] = rho_internal * MSUN_TO_G / PC_TO_CM^3
```

### 1.7 Bonnor–Ebert sphere

`bonnorEbertSphere.py:3` (module) and `:180`, `:226`:
```
Isothermal Lane-Emden:  d^2u/dxi^2 + (2/xi) du/dxi = exp(-u)
as a system:            du/dxi = omega ;  domega/dxi = exp(-u) - 2*omega/xi
xi        = r * sqrt(4*pi*G*rho_c / c_s^2)
rho(xi)/rho_c = exp(-u)
m(xi)     = xi^2 * du/dxi            ("CORRECT mass formula")
M         = 4*pi * m * rho_c * a^3   with a = c_s/sqrt(4*pi*G*rho_c)
```
*Internally consistent*: `xi = r/a` follows from the two definitions, and
`M = 4πρ_c a³ ∫ξ²e^{-u}dξ = 4πρ_c a³ ξ²u'` follows from the LE equation itself. The "directly
matches ∫4πr²ρ dr" claim at `bonnorEbertSphere.py:85` is right as mathematics.

Series initial conditions (`bonnorEbertSphere.py:207`):
```
u(xi)     = xi^2/6 - xi^4/120 + xi^6/1890 + O(xi^8)
du/dxi    = xi/3  - xi^3/30  + xi^5/315  + O(xi^7)
```
*Internally consistent*: the second is the term-by-term derivative of the first (6/1890 = 1/315).
Claimed "much more accurate than arbitrary small values".

Density profile use (`density_profile.py:3`, `:56`):
```
n(r) = nCore * f_rho_rhoc(xi)   r <= rCloud
n(r) = nISM                     r > rCloud
```
Enclosed mass (`mass_profile.py:352`):
```
M(r)/M_cloud = m(xi)/m(xi_out),  m(xi) = xi^2 du/dxi
xi/xi_out    = r/rCloud                       (linear scaling, mass_profile.py:400)
```
claimed to give "**EXACT** results: M(rCloud) = mCloud **guaranteed**".
Edge density (`validate_gmc.py:501`): `nEdge = nCore / Omega` — consistent with `Omega = ρ_core/ρ_surf`
and with locating `xi_out` where `ρ/ρ_c = 1/Omega` (`bonnorEbertSphere.py:376`).
Mass re-check (`validate_gmc.py:503`): `M = 4π·m(ξ_out)·ρ_c·a³` with `a = rCloud/ξ_out`.

Sound speed / temperature (`bonnorEbertSphere.py:427`, `:605`, `:645`):
```
T     = mu * MSUN_TO_G * c_s^2 / (gamma * k_B)          [K]
c_s^2 = gamma * k_B * T / (mu * MSUN_TO_G)              [cm^2/s^2]
```
Exact inverses of each other, so the `r2xi`/`xi2r` round trip is self-consistent — but see S2-B-04
for the γ inside a profile the module calls *isothermal*.

### 1.8 Smoothing bridge (`density_profile.py`)

`density_profile.py:3` and `:120–126`:
- The cloud→ISM step is replaced by a **tanh bridge of width `SMOOTH_FRAC * rCloud`, 1 % by default**.
- Motivation: `mShell_dot = 4πr²ρ(r)v` "jump[s] by **~10^3** across r=rCloud, which can cause **LSODA**
  to stall trying to refine its step below `min_step`".
- Claim: "so the rhs is **C^infty everywhere**".
- Claim: "The width is well below physical uncertainty in cloud-edge structure; **mass conservation
  holds to O(SMOOTH_FRAC^2)**" (i.e. ~1e-4 relative at the 1 % default).
- `density_profile.py:139`: homogeneous case "blends to nISM at rCloud".
- `density_profile.py:144`: "Inner core: constant density (**rCore is far below rCloud, so the
  smoothing band does not reach rCore in any realistic setup**)".

---

## 2. Units claimed

| Quantity | Claimed unit | Where |
|---|---|---|
| internal system | `[Msun, pc, Myr]`, "as converted by `read_param.py`" | `mass_profile.py:43`, `powerLawSphere.py:39` |
| `nCore`, `nISM`, `nEdge` | `[1/pc^3]` code units, "converted from cm^-3 via `ndens_cgs2au`" | `mass_profile.py:45`, `powerLawSphere.py:41`, `density_profile.py:56`, `validate_gmc.py:287` |
| `mu_convert` / `mu` | `[Msun]`, "converted from m_H units via `m_H * g2Msun`" | `mass_profile.py:46`, `powerLawSphere.py:42` |
| `mu_convert` / `mu` | "**(=1.4)**" | `mass_profile.py:116`, `mass_profile.py:137`, `bonnorEbertSphere.py:502`, `bonnorEbertSphere.py:536`, `validate_gmc.py:287` ("typically 1.4") |
| `mu` default arg | "the 1.4 default arg is a **placeholder**"; production passes `mu_convert ≈ 1.4·m_H` in Msun | `bonnorEbertSphere.py:307` |
| `r`, `rCloud`, `rCore` | `[pc]` | throughout |
| `M`, `mCloud` | `[Msun]` | throughout |
| `rho` | `[Msun/pc^3]` | `mass_profile.py:87` |
| `rdot` / shell velocity | `[pc/Myr]` | `mass_profile.py:442` |
| `dM/dt` | `[Msun/Myr]` | `mass_profile.py:442` |
| `c_s` | `[cm/s]` | `bonnorEbertSphere.py:136`, `:409` |
| `T_eff` | `[K]` | `bonnorEbertSphere.py:136`, `:431` |
| `sigma` | `[km/s]`, converted from `c_s [cm/s]` | `bonnorEbertSphere.py:564` |
| CGS constants | `G [cm^3 g^-1 s^-2]`, `k_B [erg K^-1]`, `m_H [g]`, `MSUN_TO_G [g/Msun]`, `PC_TO_CM [cm/pc]`, `[s/Myr]` | `bonnorEbertSphere.py:63–71` |
| `r_to_xi` / `xi_to_r` | `r [cm]`, `c_s [cm/s]`, `rho_core [g/cm^3]` | `bonnorEbertSphere.py:454`, `:476` |
| `r2xi` / `xi2r` (TRINITY iface) | `r [pc]` | `bonnorEbertSphere.py:583`, `:623` |
| suggestion display | densities converted **to cm^-3**; masses `[Msun]`, radii `[pc]` unchanged | `validate_gmc.py:82` |
| `check_gmc_constraints` | `nEdge`/`nISM` in *any* consistent unit; `ndens_to_cgs` optional, only affects the rendered message | `validate_gmc.py:191` |

The `[Msun]`-vs-`1.4` split for `mu_convert` is the single biggest unit contradiction in the slice —
finding S2-B-01.

---

## 3. Citations claimed

| Citation | Attributed to | Where |
|---|---|---|
| **Rahner+ 2018, Eq 25** | the power-law enclosed-mass formula (§1.2 region 2) | `mass_profile.py:331` |
| **Rahner+ 2018 Eq 25** | the *analytical inversion* of that same formula for `rCloud` | `powerLawSphere.py:78`, `powerLawSphere.py:149` |
| **Bonnor (1956): MNRAS 116, 351** | BE sphere (module-level reference list) | `bonnorEbertSphere.py:3` |
| **Bonnor (1956) definition** | `m_B = (1/√4π)·ξ²·(du/dξ)·√f ≈ 1.182 at critical`, with `M = m_B·c_s⁴/(G^(3/2)·√P_ext)` | `bonnorEbertSphere.py:82` |
| **Ebert (1955): Z. Astrophys. 37, 217** | BE sphere (reference list only, nothing specific attributed) | `bonnorEbertSphere.py:3` |
| **Rahner et al. (2017): MNRAS 470, 4453** | BE sphere (reference list only, nothing specific attributed) | `bonnorEbertSphere.py:3` |

Notes for the checking lens:
- "Rahner+ 2018" is given **without volume/page**; the only fully-specified Rahner reference in the
  slice is **2017, MNRAS 470, 4453**. Two different years are used inside one package for related
  WARPFIELD physics. Verify that Eq 25 of the intended paper *is* the enclosed-mass expression, and
  that 2017-vs-2018 is deliberate. (S2-B-21.)
- The Bonnor (1956) citation is attached to a *different dimensionless-mass convention* than the one
  the module says it uses — see S2-B-03.
- Critical constants claimed (`bonnorEbertSphere.py:3`, `:77–88`): `ξ_crit ≈ 6.451`,
  `Ω_crit ≈ 14.04`, `m_crit ≈ 1.182` (module docstring) / `m ≈ 15.70` (comment block), `m_B ≈ 1.182`.
  The pair (1.182, 15.70) is mutually consistent under the stated conversion
  `m_B = (1/√4π)·m·√(1/Ω_crit)` — I checked: `0.2821 × 0.2669 × 15.70 = 1.182`. So the *comment block*
  is right and the *module docstring* is the one that mislabels.

---

## 4. Ranges, regimes, assumptions claimed

### 4.1 Exponent α
- `alpha` is "typically negative, e.g. **−2 for isothermal**" (`powerLawSphere.py:78`).
- `alpha = 0` is a documented, separately-handled homogeneous case in **five** places
  (`density_profile.py:56`, `:139`; `mass_profile.py:271`, `:313`, `:608`; `powerLawSphere.py:3`, `:124`;
  `validate_gmc.py:561`).
- `compute_rCloud_powerlaw` is "valid for **α ≠ 0, α ≠ −3**" and **raises ValueError if α ≈ −3
  (mass integral diverges)** (`powerLawSphere.py:78`, guard comment at `:142`).
- `powerLawSphere.py:3` states the α ≠ 0 mass formula with **no α ≠ −3 caveat** — narrower exclusion
  than the function docstring 75 lines below.
- `mass_profile.py:271` states the same `(3+alpha)` denominator with **no α = −3 caveat at all**.
- `initial_profile.py:72`: `densPL_alpha` "Default **0.0** (homogeneous)".

### 4.2 Radii
- `rCore` default fraction of `rCloud`: **0.1** (`powerLawSphere.py:78` `rCore_fraction`, `:215`;
  and `mass_profile.py:609` "Default: 10% of cloud radius").
- `rCore` must not exceed `rCloud` — "pathological case" clamp (`mass_profile.py:620`).
- `rCloud <= r_max`, **default 200 pc, "typical single-GMC limit"** (`validate_gmc.py:3`, `:111`, `:191`,
  `:287`). Overridable by the `rCloud_max` param, else the module default (`validate_gmc.py:345`, `:365`).
- Assumption: "rCore is **far below** rCloud, so the smoothing band does not reach rCore **in any
  realistic setup**" (`density_profile.py:144`).
- `compute_enclosed_mass_bonnor_ebert`: `r_arr` "**must be sorted!**" (`mass_profile.py:352`).
- `get_mass_profile`: `rdot` "**Must be same shape as r**" if `return_mdot=True` (`mass_profile.py:137`).

### 4.3 Densities
- `nEdge >= nISM` is a hard constraint (`validate_gmc.py:3` #2, `:236`; `mass_profile.py:567`).
- alpha = 0: "nEdge = nCore, **always valid if nCore > nISM**" (`mass_profile.py:608`).

### 4.4 BE regime
- BE spheres are "**isothermal**, self-gravitating gas spheres in **hydrostatic equilibrium**"
  modelling "molecular cloud cores on the verge of gravitational collapse" (`bonnorEbertSphere.py:3`).
- `Omega` "**must be < 14.04** for stability" (`bonnorEbertSphere.py:307`);
  `is_stable : Whether Omega < 14.04 (stable)` (`bonnorEbertSphere.py:136`);
  but the validator only emits a "**Stability warning**" (`validate_gmc.py:518`).
- `gamma` (adiabatic index) **default 5/3** (`bonnorEbertSphere.py:307`, `validate_gmc.py:287`).
- `c_s` is "= **velocity dispersion** supporting the BE sphere; **turbulent, not thermal**, for
  GMC-mass clouds" (`bonnorEbertSphere.py:136`) — an explicit admission that the "isothermal
  temperature" is an effective bookkeeping quantity.
- `ρ/ρ_c` "**decreases monotonically**" — assumption backing the inverse interpolator
  (`bonnorEbertSphere.py:280`).
- Lane–Emden integration domain: `xi_min` default **1e-7** ("near zero, avoid singularity"),
  `xi_max` default **20.0** ("well beyond critical"), `n_points` default **5000**, **logarithmic**
  grid (`bonnorEbertSphere.py:90–93`, `:226`, `:250`).
- `Omega` is "kept **fixed** as a profile shape choice" in the BE suggestion search
  (`validate_gmc.py:645`).

### 4.5 Boundary / matching conditions
- PL: uniform core for `r <= rCore`, power law from `rCore` to `rCloud`, ISM beyond — matched at
  `rCore` by construction; **unmatched at `rCloud`** (the `nEdge >= nISM` check is the only guard).
- PL mass: region 1 uses `(4/3)πr³ρ_core`, region 2 the Rahner form, region 3 adds
  `(4/3)π(r³−r_cloud³)ρ_ISM` on top of `M_cloud`.
- BE: `xi_out` located where `ρ/ρ_c = 1/Omega`; `r_out = a·xi_out`; `n_out = nCore/Omega`.

---

## 5. Contracts claimed

### 5.1 `density_profile.get_density_profile` (`density_profile.py:56`)
- **In**: `r` scalar or array-like `[pc]`; `params` dict with `nISM`, `nCore`, `rCloud`, `rCore`,
  `dens_profile ∈ {'densPL','densBE'}`, `densPL_alpha` (PL), `densBE_f_rho_rhoc` (BE).
- **Out**: `n [1/pc^3]`; "Returns **scalar** if input r is scalar, **array** if input r is array".
- Doctest claims `type(n)` is `<class 'float'>` and `type(n_arr)` is `<class 'numpy.ndarray'>`.
- The documented BE key list is `densBE_f_rho_rhoc` **only** — yet `density_profile.py:27` says the
  module imports `bonnorEbertSphere` "for **r2xi** conversion", and `r2xi` is documented
  (`bonnorEbertSphere.py:583`) to need `densBE_Teff`, `nCore`, `mu_convert`, `gamma_adia`. (S2-B-11.)

### 5.2 `mass_profile`
- `get_mass_density(r, params) -> rho [Msun/pc^3]` (`:87`).
- `get_mass_profile(r, params, return_mdot=False, rdot=None)` (`:137`) — required keys
  `dens_profile`, `nCore`, `nISM`, `mu_convert`, `mCloud`, `rCloud`, `rCore`, plus "Profile-specific
  parameters (see `get_density_profile`)". Returns `M [Msun]`, plus `dMdt` when `return_mdot=True`.
  `:192` comment "Validate inputs" asserts validation exists.
- `compute_enclosed_mass(r_arr, rho_arr, params)` (`:236`) — "Power-law: **Analytical** formula;
  Bonnor-Ebert: **Analytical Lane-Emden or numerical integration**". Note `rho_arr` is an input even
  though the PL path is claimed analytical (and `:352` says `rho_arr` is "used for **fallback**
  numerical integration" only).
- `compute_enclosed_mass_bonnor_ebert` (`:352`) — needs `densBE_f_m` and `densBE_xi_out`;
  "**Falls back** to numerical integration if Lane-Emden mass function not available";
  fallback is "**trapezoidal integration (less accurate, ~0.5 % error)**" (`:412`).
- `compute_mass_accretion_rate(r_arr, rdot_arr, params) -> dMdt [Msun/Myr]` (`:442`).
- `validate_mass_at_rCloud(params, tolerance=0.001)` (`:489`) — returns dict with exactly
  `valid`, `M_computed`, `M_expected`, `relative_error`, `message`;
  `relative_error = |M_computed - M_expected| / M_expected`;
  "Required keys: `'rCloud'`, `'mCloud'`, `'dens_profile'`, **etc.**"; `:527` "Handle edge case of
  zero expected mass".
- `compute_minimum_rCore(nCore, nISM, rCloud, alpha, margin=1.1)` (`:567`) — returns
  `(rCore_suggested, nEdge, is_valid, rCore_min)`; `is_valid` = "Whether nEdge >= nISM".
- Module-level claim: "**No solver coupling** (no dependency on `array_t_now`, etc.)" (`:3`) —
  directly grep-checkable.

### 5.3 `powerLawSphere`
- `compute_rCloud_homogeneous(M_cloud, nCore, mu) -> rCloud [pc]` (`:52`).
- `compute_rCloud_powerlaw(M_cloud, nCore, alpha, rCore=None, rCore_fraction=0.1, mu) -> (rCloud, rCore)`
  (`:78`). **Raises ValueError** for (a) `α ≈ −3` (mass integral diverges), (b) "parameters are
  unphysical (**core mass alone exceeds cloud mass**)". Two "Forward mass check" comments
  (`:165`, `:197`) claim a post-hoc verification in each branch.
- `compute_consistent_params(M_cloud, nCore, alpha, rCore_fraction=0.1, mu, nISM) -> dict` (`:215`)
  with keys exactly `'rCloud'`, `'rCore'`, `'nEdge'`, `'M_cloud'`, `'nCore'`, `'alpha'`, `'mu'`.
  Advertised as "the **recommended way to set up test parameters**". Note the returned key *names*
  are not the params-dict names the rest of the slice requires (`mCloud`, `densPL_alpha`,
  `mu_convert`, `dens_profile`), and `nISM` is an input but not an output key. (S2-B-12.)

### 5.4 `bonnorEbertSphere`
- `solve_lane_emden(xi_max=20.0, n_points=5000, xi_min=1e-7) -> LaneEmdenSolution` (`:226`).
- `LaneEmdenSolution` fields (`:102`): `xi`, `u`, `dudxi`, `rho_rhoc = exp(-u)`, `m = ξ²du/dξ`,
  `f_rho_rhoc (ξ→ρ/ρc)`, `f_m (ξ→m)`, `f_xi_from_rho (ρ/ρc→ξ)`.
- `lane_emden_ode(y, xi)` with `y = [u, ω]` (`:180`) — note the `(y, t)` argument order (odeint
  convention, not `solve_ivp`'s `(t, y)`).
- `create_BE_sphere(M_cloud, n_core, Omega, mu=1.4, gamma=5/3, validate=True,
  lane_emden_solution=None) -> BESphereResult` (`:307`). Claimed algorithm: solve LE → direct lookup
  of `ξ_out` → direct lookup of `m(ξ_out)` → solve for `c_s` → unit conversion. "**no nested
  optimization**". `:347` "VALIDATION" block, `:380` "Check bounds".
- `BESphereResult` fields (`:136`): `xi_out`, `r_out`, `n_out`, `T_eff`, `c_s`, `m_dim`, `M_cloud`,
  `n_core`, `Omega`, `is_stable`.
- `create_BE_sphere_from_params(params)` (`:502`) — **side effect**: writes `densBE_Teff`,
  `densBE_xi_arr`, `densBE_u_arr`, `densBE_dudxi_arr`, `densBE_rho_rhoc_arr`, `densBE_f_rho_rhoc`,
  `densBE_f_m`, `densBE_xi_out`, `rCloud`, `nEdge`. Comments reveal **two further undocumented
  writes**: `:564` "c_s [cm/s] -> **sigma** [km/s]" and `:573` "Also update **derived cloud
  properties**". `:552` "Ensure all BE-specific params exist (**safety fallback** for standalone
  usage)". (S2-B-22.)
- `r2xi(r, params)` / `xi2r(xi, params)` (`:583`, `:623`) — need `densBE_Teff`, `nCore`,
  `mu_convert`, `gamma_adia`; scalar or array in/out.

### 5.5 `initial_profile`
- `build_initial_cloud_profile(*, dens_profile, mCloud, nCore, nISM, rCore, rCloud, mu_convert,
  densPL_alpha=0.0, densBE_Omega, gamma_adia, nEdge) -> (r_arr, n_arr, m_arr)` (`:72`), each a 1-D
  `np.ndarray` of **equal length**, "Matches the layout produced by `phase0_init.get_InitCloudProp`".
- **Raises ValueError** if `dens_profile` unsupported, or BE requested without BE-specific scalars.
- `densBE_Omega`, `gamma_adia` "Required for `densBE`; **ignored for PL**";
  `densPL_alpha` "**ignored for BE**"; `nEdge` "Optional … (**BE only**)".
  But `:126` says `_init_powerlaw_cloud` **will populate `nEdge`** and it is pre-seeded for "the rare
  edge-correction path that may read it". (S2-B-19.)
- Claims: it is "**the inverse of** `trinity/phase0_init/get_InitCloudProp.py`"; the arrays are "a
  **deterministic function of ~6 scalars**"; inline storage "costs **~71 KB per run snapshot**";
  logic stays "**single-sourced**" inside phase-0; **"Calling the constructor with post-correction
  scalars is a no-op for the auto-correction branches (`nEdge < nISM` etc.) because those checks pass
  given the already-corrected inputs."** (S2-B-20.)
- `:39` the supported-profile list "**Mirrors** `trinity/_input/read_param.py` and
  `trinity/_output/cloudy/run_loader.py`" — a three-way duplication that must stay in sync.
- `:110` lazy import "avoids a circular dependency (cloud_properties → phase0_init → cloud_properties)".
- `_MockItem` (`:45`) "**duplicates only the surface area they touch** — no metadata, no JSON helpers".

### 5.6 `validate_gmc`
- Three constraints, stated at `:3` and re-stated at `:229`/`:236`/`:249`:
  1. `rCloud <= r_max` (default 200 pc) 2. `nEdge >= nISM` 3. mass error <= tolerance (default 0.001).
- `check_gmc_constraints(rCloud, nEdge, mCloud, M_computed, nISM, r_max=200, mass_tolerance=0.001,
  ndens_to_cgs=None) -> {'errors', 'warnings', 'mass_error'}` (`:191`).
- `validate_gmc_params(mCloud, nCore, mu, nISM, dens_profile, alpha=None, rCore=None, Omega=None,
  gamma=5/3, r_max=200, mass_tolerance=0.001, lane_emden_solution=None) -> GMCValidationResult` (`:287`).
  `alpha` "**required for densPL**"; `rCore` "**required for `densPL` with alpha != 0**";
  `Omega` "**required for densBE**".
- `validate_gmc_from_params(params, r_max=None, mass_tolerance=0.001)` (`:345`) — `params` is
  "dict-like with `.value` attribute access"; `r_max=None` ⇒ use `rCloud_max` param if present, else 200.
- `GMCValidationResult` fields (`:121`): `valid`, `errors`, `warnings`, `rCloud`, `nEdge`,
  `mass_error = |M_computed - mCloud|/mCloud`, `M_computed`, `suggestions`; `.summary()` (`:153`).
- `_suggest_powerlaw_alternatives` (`:553`): "Varies `mCloud`, `nCore`, `rCore` by ±`search_range`
  and returns the closest valid combinations **sorted by distance from the original**";
  `:561` "For alpha=0 (homogeneous), rCore is irrelevant — only vary mCloud/nCore".
- `_suggest_bonnor_ebert_alternatives` (`:645`): varies `mCloud` and `nCore` only.
- `_quiet_loggers` (`:63`): "Temporarily raise the level of named loggers (**restored on exit**)".
- `format_suggestion` (`:82`): converts `nCore`/`nEdge` to cm^-3 "the same unit used in the parameter
  file **so a user can copy the values straight across**".

**Documented usage examples that are themselves testable claims** (`validate_gmc.py:3`):
```
validate_gmc_params(mCloud=1e5, nCore=1e3, mu=1.4, nISM=1.0,
                    dens_profile='densPL', alpha=-2, rCore=1.0)
check_gmc_constraints(rCloud=150.0, nEdge=0.5, mCloud=1e5, M_computed=1.001e5)
```
Both are advertised as working invocations. See S2-B-16 (units) and S2-B-17 (`nISM` omitted; and the
mass error is exactly at the default tolerance).

---

## 6. Numerical claims

| Claim | Value | Where |
|---|---|---|
| Smoothing width | `SMOOTH_FRAC * rCloud`, **1 % default** | `density_profile.py:3`, `:124` |
| Smoothing mass error | **O(SMOOTH_FRAC²)** | `density_profile.py:126` |
| Density jump at rCloud | **~10^3** | `density_profile.py:121` |
| Solver named | **LSODA**, stalls below `min_step` | `density_profile.py:122` |
| Smoothness after bridge | **C^infty everywhere** | `density_profile.py:3`, `:124` |
| BE numerical fallback error | **~0.5 %**, trapezoidal | `mass_profile.py:412` |
| BE analytical path | "**EXACT**", `M(rCloud) = mCloud` **guaranteed** | `mass_profile.py:352`, `:407` |
| Mass validation tolerance | **0.001 = 0.1 %** | `mass_profile.py:489`, `validate_gmc.py:112`, `:191`, `:287`, `:345` |
| rCloud max | **200 pc** | `validate_gmc.py:111` |
| rCore safety margin | **1.1** | `mass_profile.py:567` |
| rCore fraction default | **0.1** | `powerLawSphere.py:78`, `mass_profile.py:609` |
| LE grid | `xi_min=1e-7`, `xi_max=20.0`, `n_points=5000`, **logarithmic** | `bonnorEbertSphere.py:226`, `:250` |
| LE critical values | `ξ_crit ≈ 6.451`, `Ω_crit ≈ 14.04`, `m ≈ 15.70`, `m_B ≈ 1.182` | `bonnorEbertSphere.py:3`, `:77–88` |
| α = −3 guard | ValueError, "mass integral diverges" | `powerLawSphere.py:78`, `:142` |
| γ default | **5/3** | `bonnorEbertSphere.py:307`, `validate_gmc.py:287` |
| Storage saved | **~71 KB per run snapshot** | `initial_profile.py:3` |

---

## 7. Admissions (TODO / hack / approximate / "should be" / hedges)

No `TODO`, `FIXME`, `XXX` or `HACK` markers appear in this slice's prose. What is there instead:

- `mass_profile.py:318` and `:338` — "ISM region - **mCloud should be in Msun**" (twice). "should be"
  = an unverified unit precondition on a value the function then adds ISM mass to.
- `mass_profile.py:412` — numerical fallback is "**less accurate, ~0.5 % error**".
- `bonnorEbertSphere.py:3` — "Bonnor-Ebert Sphere Implementation - **CORRECT VERSION v2**"; "**CORRECT**
  mass formula". Version/correctness labels in a docstring are a scar from a previous wrong formula.
- `mass_profile.py:3` / `:442` / `:478` — "**Correct** formula … for **ALL** profiles", "This formula is
  **EXACT**", "This works for **ALL** density profiles!". Same scar pattern; the emphasis suggests an
  earlier path that was profile-dependent or solver-history-dependent.
- `bonnorEbertSphere.py:307` — "the **1.4 default arg is a placeholder**".
- `bonnorEbertSphere.py:136` — `c_s` is "**turbulent, not thermal**, for GMC-mass clouds" while the
  whole construction is labelled isothermal.
- `bonnorEbertSphere.py:552` — "**safety fallback** for standalone usage".
- `mass_profile.py:620` — "Ensure rCore doesn't exceed rCloud (**pathological case**)".
- `mass_profile.py:527` — "Handle **edge case** of zero expected mass".
- `initial_profile.py:126` — "**nEdge placeholder** … Pre-seed for the **rare edge-correction path
  that may read it**" ("may" = the author was not sure).
- `initial_profile.py:143` — "**Pre-seed** them so the `.value = ...` assignment lands cleanly."
- `initial_profile.py:3` — "Calling the constructor with post-correction scalars **is a no-op** …
  **because those checks pass given the already-corrected inputs**" — an unproven idempotency argument.
- `density_profile.py:144` — "**in any realistic setup**" (unbounded assumption).
- `density_profile.py:159` — "interpolator **already handles** xi values for r > rCloud — **same as the
  original code path**" (defers to prior behaviour rather than stating the behaviour).
- `mass_profile.py:489` — "Required keys: 'rCloud', 'mCloud', 'dens_profile', **etc.**" (incomplete
  contract by admission).
- `powerLawSphere.py:215` — "This is the **recommended way to set up test parameters**" (advice, no
  enforcement).

---

## 8. Prose-vs-prose contradictions (summary)

1. **`mu_convert` is `[Msun]` (≈1.2e-57) *and* "=1.4"** — five sites say 1.4, two unit blocks say Msun,
   one site admits 1.4 is a placeholder. (S2-B-01)
2. **BE critical mass is 1.182 *and* 15.70** for the same stated formula `m = ξ²du/dξ`. (S2-B-03)
3. **α = −3 is fatal in `powerLawSphere` but unmentioned in `mass_profile`**, and the
   `powerLawSphere` *module* docstring states the formula for "α ≠ 0" while the function docstring 75
   lines later says "α ≠ 0, α ≠ −3". (S2-B-05)
4. **Cloud edge is a sharp step *and* a tanh bridge** — `get_density_profile`'s Notes and the BE
   section both state `n = nISM for r > rCloud` unqualified, while the module docstring and the
   `:120` block say that step is replaced. (S2-B-08 / S2-B-09)
5. **`Omega >= 14.04` is "must be < 14.04" (rejection language) *and* a warning.** (S2-B-14)
6. **`nEdge` is "BE only" *and* populated by `_init_powerlaw_cloud`.** (S2-B-19)
7. **`rCore` is "required for densPL with alpha != 0" *and* optional with a 0.1 fraction fallback.**
   (S2-B-23)
8. **BE fallback accuracy (0.5 %) is 5× looser than the mass tolerance (0.1 %) it must satisfy.**
   (S2-B-02)
9. **Densities are `[1/pc^3]` code units *and* the module's own usage example passes cm^-3-magnitude
   numbers.** (S2-B-16)
10. **`T_eff` is the "effective *isothermal* temperature" *and* carries a 1/γ adiabatic factor.**
    (S2-B-04)

## 9. Claims too vague to check as written

- `density_profile.py:56` — "alpha != 0: `n = nCore*(r/rCore)^alpha` **with boundary conditions**"
  (which boundary conditions?).
- `density_profile.py:159` — "interpolator **already handles** xi values for r > rCloud" (handles how?
  clamp, extrapolate, NaN?).
- `mass_profile.py:489` — "Required keys: … **etc.**".
- `validate_gmc.py:121` — warnings are "Non-critical notes (**e.g. rCloud near limit**)" — no
  threshold for "near" is stated anywhere.
- `bonnorEbertSphere.py:136` — `m_dim : Dimensionless mass` — in *which* of the two conventions the
  module documents?
- `initial_profile.py:3` — "a deterministic function of **~6 scalars**" while the public signature
  takes 11.
- `powerLawSphere.py:165`, `:197` — "Forward mass check" (checked against what tolerance? what
  happens on failure?).

---

```json
[
  {
    "id": "S2-B-01",
    "file": "trinity/cloud_properties/mass_profile.py",
    "line": 116,
    "class": "units",
    "severity": "S2",
    "claim": "mu_convert is documented BOTH as '(=1.4)' (mass_profile.py:116, :137; bonnorEbertSphere.py:502, :536; validate_gmc.py:287 'typically 1.4') AND as a mass in [Msun] 'converted from m_H units via m_H * g2Msun' (mass_profile.py:46; powerLawSphere.py:42). bonnorEbertSphere.py:307 admits 'production passes mu_convert ~ 1.4*m_H in Msun; the 1.4 default arg is a placeholder'.",
    "evidence": "mass_profile.py:46 '- mu_convert: [Msun] (converted from m_H units via m_H * g2Msun)'; mass_profile.py:116 '# mu_convert = 1.4 is independent of ionization state'; bonnorEbertSphere.py:307 'mu : float Mean molecular weight for mass conversion [Msun] (production passes mu_convert ~ 1.4*m_H in Msun; the 1.4 default arg is a placeholder)'.",
    "expected": "One unambiguous convention. 1.4*m_H in Msun is ~1.2e-57, not 1.4 - the two readings differ by ~57 orders of magnitude, so rho = n*mu_convert cannot be [Msun/pc^3] under both. Docstrings that say '(=1.4)' should say '[Msun], = 1.4 * m_H expressed in Msun'.",
    "failure_scenario": "Any caller who follows the '(=1.4)' docstring literally (a test, a plotting script, validate_gmc_params(mu=1.4) as shown in that module's own usage example) computes rho too large by ~1e57 and gets nonsense rCloud/mass without any error being raised.",
    "repro": "Compare the mu_convert value produced by read_param.py against the literal 1.4 in the default args of create_BE_sphere and in validate_gmc.py's docstring example; assert which one every call site actually receives.",
    "confidence": "high"
  },
  {
    "id": "S2-B-02",
    "file": "trinity/cloud_properties/mass_profile.py",
    "line": 412,
    "class": "numerical",
    "severity": "S2",
    "claim": "The BE numerical fallback is documented as trapezoidal with '~0.5% error', while the mass-consistency check that consumes M(rCloud) has a default tolerance of 0.1% (0.001) in three places.",
    "evidence": "mass_profile.py:412 '# Use trapezoidal integration (less accurate, ~0.5% error)'; mass_profile.py:489 'tolerance : float Maximum allowed relative error (default 0.1% = 0.001)'; validate_gmc.py:112 '# 0.1% relative mass error'; validate_gmc.py:3 'Mass error <= tolerance (self-consistent parameters)'.",
    "expected": "Either the fallback is accurate enough to pass the 0.1% gate, or the fallback path must relax/skip the mass check, or the fallback must be unreachable in production. As documented, a run that takes the fallback fails its own validation by a factor of 5.",
    "failure_scenario": "densBE run where 'densBE_f_m'/'densBE_xi_out' are missing (mass_profile.py:352 says it then falls back) -> M(rCloud) off by ~0.5% -> validate_gmc reports an error and the run is rejected, with the message blaming the user's cloud parameters rather than the integration path.",
    "repro": "Call compute_enclosed_mass_bonnor_ebert with params lacking densBE_f_m, then feed the result to validate_mass_at_rCloud(tolerance=0.001) and check valid.",
    "confidence": "high"
  },
  {
    "id": "S2-B-03",
    "file": "trinity/cloud_properties/bonnorEbertSphere.py",
    "line": 3,
    "class": "coefficient",
    "severity": "S2",
    "claim": "The module docstring lists 'm_crit ~ 1.182 (critical dimensionless mass)' immediately alongside 'CORRECT mass formula: m(xi) = xi^2 du/dxi', but the constants block states that m = xi^2 du/dxi ~ 15.70 at critical and that 1.182 is Bonnor's DIFFERENT convention m_B = (1/sqrt(4pi)) xi^2 (du/dxi) sqrt(f).",
    "evidence": "bonnorEbertSphere.py:3 'Critical values: xi_crit ~ 6.451 ... Omega_crit ~ 14.04 ... m_crit ~ 1.182 (critical dimensionless mass) ... CORRECT mass formula: m(xi) = xi^2 du/dxi'; bonnorEbertSphere.py:84 '- Integration-based definition (used here): m = xi^2 x du/dxi ~ 15.70 at critical'; bonnorEbertSphere.py:88 \"# Bonnor's dimensionless mass (for reference)\".",
    "expected": "The module docstring should read m_crit ~ 15.70 for the m = xi^2 du/dxi convention it says is used; 1.182 belongs only to m_B. (The two are consistent: 0.2821 * sqrt(1/14.04) * 15.70 = 1.182.)",
    "failure_scenario": "If any stability/critical-mass comparison uses the 1.182 constant against an m computed as xi^2 du/dxi, every sphere is misclassified (15.70 > 1.182 always), so a stability gate would fire on all inputs or never.",
    "repro": "Solve Lane-Emden, evaluate f_m(6.451), and compare against whichever module constant is used in the stability/critical comparison.",
    "confidence": "high"
  },
  {
    "id": "S2-B-04",
    "file": "trinity/cloud_properties/bonnorEbertSphere.py",
    "line": 427,
    "class": "coefficient",
    "severity": "S3",
    "claim": "T_eff is documented as the 'Effective isothermal temperature [K]' but is computed with an adiabatic index: T = mu*MSUN_TO_G*c_s^2/(gamma*k_B), gamma default 5/3. The Lane-Emden form exp(-u) used everywhere else requires strictly isothermal P = rho*c_s^2, for which T = mu*m_H*c_s^2/k_B with no gamma.",
    "evidence": "bonnorEbertSphere.py:427 '# Effective temperature: T = mu_dimensionless * m_H * c_s^2 / (gamma * k_B)'; bonnorEbertSphere.py:136 'T_eff : float [K] Effective isothermal temperature'; bonnorEbertSphere.py:3 'isothermal, self-gravitating gas spheres in hydrostatic equilibrium'; bonnorEbertSphere.py:605/:645 'c_s^2 = gamma k_B T / m_particle'.",
    "expected": "Either drop gamma (isothermal) or stop calling T_eff isothermal. As written T_eff = T_isothermal/gamma = 0.6*T_iso for gamma=5/3.",
    "failure_scenario": "The r2xi/xi2r round trip is self-consistent so the profile is unaffected, but any consumer that treats densBE_Teff as a physical gas temperature (cooling tables, cloudy export, reported diagnostics) is low by a factor gamma.",
    "repro": "For a BE run, compare params['densBE_Teff'] against mu_cgs*c_s^2/k_B computed from the returned BESphereResult.c_s; the ratio should be exactly gamma_adia.",
    "confidence": "medium"
  },
  {
    "id": "S2-B-05",
    "file": "trinity/cloud_properties/mass_profile.py",
    "line": 271,
    "class": "divergence",
    "severity": "S3",
    "claim": "compute_enclosed_mass_powerlaw states the region-2 mass with (3+alpha) in the denominator and documents NO alpha = -3 special case, while powerLawSphere.compute_rCloud_powerlaw documents that alpha ~ -3 raises ValueError because the mass integral diverges. The powerLawSphere MODULE docstring is a third variant, stating the formula for 'alpha != 0' only.",
    "evidence": "mass_profile.py:271 'M(r) = 4*pi*rho_core [r_core^3/3 + (r^(3+alpha) - r_core^(3+alpha))/((3+alpha)*r_core^alpha)]' with no caveat; powerLawSphere.py:78 'Solving for rCloud (valid for alpha != 0, alpha != -3) ... Raises ValueError If alpha ~ -3 (mass integral diverges)'; powerLawSphere.py:142 '# Guard: alpha = -3 makes the mass integral diverge'; powerLawSphere.py:3 'For alpha != 0: M = 4*pi*rho_c[...]'.",
    "expected": "Consistent handling: either alpha = -3 is rejected once at schema/param level (and every docstring says so), or the log-limit form M = 4*pi*rho_c[rc^3/3 + rc^3*ln(r/rc)] is implemented in the mass function too.",
    "failure_scenario": "alpha = -3 supplied with an externally provided rCloud (e.g. reconstructed via initial_profile.build_initial_cloud_profile, or a params dict that bypasses compute_rCloud_powerlaw) reaches compute_enclosed_mass_powerlaw and divides by zero -> inf/NaN mass propagating into the shell ODEs.",
    "repro": "Call compute_enclosed_mass_powerlaw with densPL_alpha = -3.0 and an explicit rCloud/rCore; check for ZeroDivisionError/inf rather than a clean ValueError.",
    "confidence": "high"
  },
  {
    "id": "S2-B-06",
    "file": "trinity/cloud_properties/powerLawSphere.py",
    "line": 78,
    "class": "regime",
    "severity": "S3",
    "claim": "compute_rCloud_powerlaw claims its closed-form inversion is 'valid for alpha != 0, alpha != -3' and documents only two ValueError conditions (alpha ~ -3; core mass alone exceeds cloud mass). For alpha < -3 the fixed-rCore branch has a further, undocumented failure: the total mass converges as rCloud -> inf, so no solution exists above M_max = 4*pi*rho_c*rCore^3*(1/3 + 1/|3+alpha|).",
    "evidence": "powerLawSphere.py:78 'rCloud = { [M/(4*pi*rho_c) - rCore^3/3] x (3+alpha) x rCore^alpha + rCore^(3+alpha) }^(1/(3+alpha))'; same docstring 'Raises ValueError If alpha ~ -3 (mass integral diverges) or if the parameters are unphysical (core mass alone exceeds cloud mass).'",
    "expected": "For alpha < -3 the brace can go negative (the first term is negative because 3+alpha<0), so a negative base is raised to a negative fractional power. Either document/raise on M > M_max, or state that alpha < -3 is unsupported. Note the fractional-rCore branch (powerLawSphere.py:182) has no such ceiling, so the two branches have different validity domains for the same alpha.",
    "failure_scenario": "alpha = -3.5 with a fixed rCore and a large mCloud -> negative base ** negative power -> NaN (or a complex/ValueError from **), reported as an opaque numerical failure rather than 'this alpha cannot hold this mass'.",
    "repro": "compute_rCloud_powerlaw(M_cloud large, nCore, alpha=-3.5, rCore=1.0) and compare against the fractional-rCore call with the same alpha.",
    "confidence": "medium"
  },
  {
    "id": "S2-B-07",
    "file": "trinity/cloud_properties/mass_profile.py",
    "line": 567,
    "class": "regime",
    "severity": "S3",
    "claim": "compute_minimum_rCore's docstring derives rCore_min ONLY for alpha < 0 ('For alpha < 0 (density decreasing outward), require nEdge >= nISM ... Since alpha < 0, raising to power 1/alpha flips inequality'), and the comments describe only a homogeneous branch and an 'alpha < 0' branch. No behaviour is stated for alpha > 0.",
    "evidence": "mass_profile.py:567 full derivation guarded by 'For alpha < 0'; mass_profile.py:608 '# Homogeneous: nEdge = nCore, always valid if nCore > nISM'; mass_profile.py:612 '# For alpha < 0: compute minimum rCore'; mass_profile.py:620 \"# Ensure rCore doesn't exceed rCloud (pathological case)\".",
    "expected": "An explicit alpha > 0 branch or an explicit rejection. For alpha > 0 with nCore > nISM the constraint nEdge >= nISM is automatically satisfied, yet the formula rCore_min = rCloud*(nCore/nISM)^(1/alpha) evaluates > rCloud and would be clamped by the pathological-case guard, silently returning rCore = rCloud (a degenerate profile with no power-law region).",
    "failure_scenario": "A user sets a positive densPL_alpha (centrally-thin cloud); the helper returns rCore == rCloud with is_valid True, collapsing the profile to a uniform sphere without warning.",
    "repro": "compute_minimum_rCore(nCore, nISM, rCloud, alpha=+1.0) and inspect rCore_suggested vs rCloud and is_valid.",
    "confidence": "medium"
  },
  {
    "id": "S2-B-08",
    "file": "trinity/cloud_properties/density_profile.py",
    "line": 159,
    "class": "regime",
    "severity": "S3",
    "claim": "For the BE profile the docstrings state 'n = nCore * f_rho_rhoc(xi) for r <= rCloud, n = nISM for r > rCloud' (twice), but the inline comment says the interpolator 'already handles xi values for r > rCloud - same as the original code path', which describes deferring to the interpolator rather than substituting nISM.",
    "evidence": "density_profile.py:3 'For Bonnor-Ebert sphere: n(r) = nCore * f_rho_rhoc(xi) for r <= rCloud, n(r) = nISM for r > rCloud'; density_profile.py:56 Notes repeat it; density_profile.py:159 '# Get density ratio from interpolation function (interpolator already handles xi values for r > rCloud - same as the original code path)'.",
    "expected": "State and enforce one behaviour outside rCloud for BE: nISM, or clamped/extrapolated interpolator output. These differ by orders of magnitude at large r.",
    "failure_scenario": "If the interpolator extrapolates (or clamps to its last tabulated value) beyond xi_out instead of returning nISM, a shell that has left the cloud sweeps up the wrong ambient density - and 'the original code path' is cited as the authority instead of the documented contract.",
    "repro": "Evaluate get_density_profile at r = 2*rCloud and r = 10*rCloud for a densBE params dict and compare against params['nISM'].",
    "confidence": "medium"
  },
  {
    "id": "S2-B-09",
    "file": "trinity/cloud_properties/density_profile.py",
    "line": 3,
    "class": "numerical",
    "severity": "S3",
    "claim": "The tanh bridge is described unconditionally ('The discontinuous jump from cloud density to nISM at r=rCloud is replaced by a tanh bridge ... so the ODE solvers in the phase modules see a C^infty rhs at the cloud boundary'), and the rhs is claimed 'C^infty everywhere'. Blend comments appear only in the homogeneous branch (:139) and the PL section; the BE branch (:150-160) has no blend comment.",
    "evidence": "density_profile.py:3 (module) and density_profile.py:124 '# width SMOOTH_FRAC*rCloud so the rhs is C^infty everywhere'; density_profile.py:139 '# Homogeneous cloud: constant nCore inside, blends to nISM at rCloud'; density_profile.py:150-160 BE section mentions only xi conversion and the interpolator.",
    "expected": "Either the bridge covers densBE too (in which case say so in the BE branch), or the module docstring must scope the claim to densPL. Separately, 'C^infty everywhere' is only true if the blend is a global tanh; a tanh applied inside a band and hard-switched outside it is C^0 (not even C^1) at the band edges, and a spline/tabulated BE interpolator is at best C^2.",
    "failure_scenario": "A densBE run keeps the ~10^3 step the comment says stalls LSODA, i.e. the stated motivation for the whole mechanism is unaddressed for half the supported profiles.",
    "repro": "Finite-difference n(r) and its first three derivatives across r = rCloud*(1 +/- 2*SMOOTH_FRAC) for both dens_profile values; look for discontinuities at the band edges and for whether densBE is smoothed at all.",
    "confidence": "medium"
  },
  {
    "id": "S2-B-10",
    "file": "trinity/cloud_properties/mass_profile.py",
    "line": 223,
    "class": "other",
    "severity": "S3",
    "claim": "dM/dt is claimed to equal dM/dr * dr/dt exactly, with dM/dr = 4*pi*r^2*rho(r) taken from the SMOOTHED density (get_mass_density -> get_density_profile, which applies the tanh bridge), while M(r) itself is claimed to come from the UNSMOOTHED analytic piecewise formula (compute_enclosed_mass_powerlaw). The two are not derivative/antiderivative of each other inside the smoothing band.",
    "evidence": "mass_profile.py:223 '# dM/dt = dM/dr * dr/dt = 4*pi*r^2 * rho(r) * v(r)'; mass_profile.py:442 'This formula is EXACT for any smooth density profile'; mass_profile.py:271 analytic piecewise M(r); density_profile.py:126 '# mass conservation holds to O(SMOOTH_FRAC^2)'.",
    "expected": "d/dr of the returned M(r) should equal 4*pi*r^2*rho_returned(r) to round-off, or the mismatch should be documented and bounded. The prose bounds the integrated mass error (O(SMOOTH_FRAC^2)) but not the local derivative error, which is O(1) inside the band (the smoothed rho differs from the step rho by up to the full ~10^3 jump there).",
    "failure_scenario": "Near r = rCloud the phase ODE's mShell and mShell_dot are mutually inconsistent by up to the full density contrast within a 1%-wide band - exactly the region the smoothing was introduced to stabilise.",
    "repro": "For r in rCloud*[0.98, 1.02], compare numerical d/dr of get_mass_profile(r) against 4*pi*r^2*get_mass_density(r).",
    "confidence": "medium"
  },
  {
    "id": "S2-B-11",
    "file": "trinity/cloud_properties/density_profile.py",
    "line": 56,
    "class": "state",
    "severity": "S3",
    "claim": "The documented 'Required keys' lists are incomplete for the densBE path. get_density_profile lists only densBE_f_rho_rhoc, but the module imports bonnorEbertSphere 'for r2xi conversion' and r2xi is documented to need densBE_Teff, nCore, mu_convert and gamma_adia. get_mass_profile's required-key list omits densBE_f_m and densBE_xi_out, which compute_enclosed_mass_bonnor_ebert says it needs.",
    "evidence": "density_profile.py:56 '- densBE_f_rho_rhoc: interpolation function (for densBE)'; density_profile.py:27 '# Import Bonnor-Ebert sphere module for r2xi conversion'; bonnorEbertSphere.py:583 'params : dict TRINITY parameters (needs densBE_Teff, nCore, mu_convert, gamma_adia)'; mass_profile.py:137 required keys list; mass_profile.py:352 \"For analytical method, needs 'densBE_f_m' and 'densBE_xi_out'.\"",
    "expected": "The required-keys lists should name every key actually read, so a caller constructing a params dict by hand (tests, plot scripts, cloudy export) knows what to supply.",
    "failure_scenario": "A hand-built densBE params dict following the docstring raises KeyError deep inside r2xi, or - worse - silently takes the numerical fallback (S2-B-02) because densBE_f_m is 'not available'.",
    "repro": "Build a params dict containing exactly the documented keys for densBE and call get_density_profile / get_mass_profile.",
    "confidence": "high"
  },
  {
    "id": "S2-B-12",
    "file": "trinity/cloud_properties/powerLawSphere.py",
    "line": 215,
    "class": "state",
    "severity": "S3",
    "claim": "compute_consistent_params is advertised as 'the recommended way to set up test parameters' but returns keys 'M_cloud', 'alpha', 'mu' - none of which match the params-dict keys the rest of the slice requires ('mCloud', 'densPL_alpha', 'mu_convert'), and it omits 'dens_profile' and 'nISM' entirely (nISM is an input but not an output key).",
    "evidence": "powerLawSphere.py:215 \"Returns dict with keys: 'rCloud','rCore','nEdge','M_cloud','nCore','alpha','mu'\"; density_profile.py:56 requires 'nISM','nCore','rCloud','rCore','dens_profile','densPL_alpha'; mass_profile.py:137 requires 'dens_profile','nCore','nISM','mu_convert','mCloud','rCloud','rCore'.",
    "expected": "Either the returned dict uses TRINITY key names so it can be passed straight to get_density_profile/get_mass_profile as the docstring implies, or the docstring stops calling it the recommended setup path.",
    "failure_scenario": "A test that follows the docstring passes the returned dict to get_density_profile and hits KeyError on 'dens_profile'/'densPL_alpha' - or, if a .get() with a default is used, silently evaluates a homogeneous profile with alpha defaulting to 0.",
    "repro": "get_density_profile(1.0, compute_consistent_params(M_cloud=1e5, nCore=..., alpha=-2)).",
    "confidence": "medium"
  },
  {
    "id": "S2-B-13",
    "file": "trinity/cloud_properties/mass_profile.py",
    "line": 352,
    "class": "silent-failure",
    "severity": "S3",
    "claim": "compute_enclosed_mass_bonnor_ebert documents 'r_arr : array Radii (must be sorted!)' as a precondition. No enforcement, check, or error is described anywhere in the prose.",
    "evidence": "mass_profile.py:352 'r_arr : array Radii (must be sorted!)'; the only validation comment in the file is mass_profile.py:192 '# Validate inputs' inside get_mass_profile, which documents only rdot shape/presence rules.",
    "expected": "Either sort internally, or raise on unsorted input. Note the precondition is only meaningful for the trapezoidal fallback (the analytic m(xi)/m(xi_out) path is pointwise), so the failure is path-dependent and therefore intermittent.",
    "failure_scenario": "Unsorted radii reach the fallback path; np.trapz over a non-monotonic abscissa returns a partially-cancelling, silently wrong enclosed mass - no exception, no warning.",
    "repro": "Call compute_enclosed_mass_bonnor_ebert with a shuffled r_arr and params lacking densBE_f_m; compare against the sorted-input result.",
    "confidence": "medium"
  },
  {
    "id": "S2-B-14",
    "file": "trinity/cloud_properties/bonnorEbertSphere.py",
    "line": 307,
    "class": "regime",
    "severity": "S3",
    "claim": "Omega >= 14.04 is described with rejection language in create_BE_sphere ('must be < 14.04 for stability', with a VALIDATION block and validate=True by default) but is handled as a non-fatal 'Stability warning' by the validator. It is unclear whether an unstable Omega is supported or rejected.",
    "evidence": "bonnorEbertSphere.py:307 'Omega : float Density contrast rho_core/rho_surface (must be < 14.04 for stability)' and 'validate : bool Perform input validation (default: True)'; bonnorEbertSphere.py:136 'is_stable : bool Whether Omega < 14.04 (stable)'; validate_gmc.py:518 '# Stability warning'.",
    "expected": "One documented policy. If unstable spheres are constructible (is_stable exists as a returned flag, so presumably yes), create_BE_sphere's 'must be' should be softened; if they are rejected, validate_gmc's warning is unreachable.",
    "failure_scenario": "A user sets densBE_Omega = 20 expecting the documented warning path, and instead gets a hard ValueError from create_BE_sphere (or vice versa: the validator only warns and an unstable-by-construction cloud is simulated).",
    "repro": "create_BE_sphere(..., Omega=20.0, validate=True) and validate_gmc_params(..., dens_profile='densBE', Omega=20.0); compare which raises and which warns.",
    "confidence": "medium"
  },
  {
    "id": "S2-B-15",
    "file": "trinity/cloud_properties/bonnorEbertSphere.py",
    "line": 226,
    "class": "numerical",
    "severity": "S3",
    "claim": "The Lane-Emden table is claimed to run to xi_max = 20.0 ('well beyond critical'), and xi_out is found by 'DIRECT LOOKUP' of where rho/rho_c = 1/Omega, with a bare '# Check bounds'. No maximum supported Omega is documented anywhere, even though the table depth sets one.",
    "evidence": "bonnorEbertSphere.py:226 'xi_max : float, optional Maximum xi (default: 20.0)'; bonnorEbertSphere.py:92 '# Maximum xi (well beyond critical)'; bonnorEbertSphere.py:376 '# STEP 2: Find xi_out where rho/rho_c = 1/Omega (DIRECT LOOKUP)'; bonnorEbertSphere.py:380 '# Check bounds'.",
    "expected": "The maximum representable Omega equals 1/(rho/rho_c at xi_max). For the isothermal sphere rho/rho_c ~ 2/xi^2 asymptotically, so xi_max=20 caps Omega at roughly a couple of hundred. That ceiling - and what happens above it (raise? clamp? extrapolate?) - should be documented alongside 'must be < 14.04'.",
    "failure_scenario": "A large-Omega request either raises an out-of-range interpolation error whose message points at the table rather than at Omega, or is silently clamped to xi_max, producing a cloud whose edge density is not nCore/Omega - which then fails the nEdge/mass checks for the wrong stated reason.",
    "repro": "create_BE_sphere with Omega = 500 and inspect whether xi_out, n_out satisfy n_out == nCore/Omega.",
    "confidence": "low"
  },
  {
    "id": "S2-B-16",
    "file": "trinity/cloud_properties/validate_gmc.py",
    "line": 3,
    "class": "units",
    "severity": "S3",
    "claim": "The module's own copy-pasteable usage examples pass cm^-3-magnitude densities (nCore=1e3, nISM=1.0, nEdge=0.5) and mu=1.4, while the function docstrings 250 lines later declare those same arguments to be '[1/pc^3] (code units)' and '[Msun] (code units)'.",
    "evidence": "validate_gmc.py:3 'result = validate_gmc_params(mCloud=1e5, nCore=1e3, mu=1.4, nISM=1.0, dens_profile=\\'densPL\\', alpha=-2, rCore=1.0,)' and 'issues = check_gmc_constraints(rCloud=150.0, nEdge=0.5, mCloud=1e5, M_computed=1.001e5,)'; validate_gmc.py:287 'nCore : float Core number density [1/pc^3] (code units)... mu : float Mean molecular weight [Msun] (code units, typically 1.4)... nISM : float ISM density [1/pc^3] (code units)'.",
    "expected": "The examples should use code-unit values (1 cm^-3 = 2.94e55 pc^-3), or explicitly note that the example numbers are illustrative cgs. As written, following the example is guaranteed to give a wrong rCloud. Same root cause as S2-B-01.",
    "failure_scenario": "A user or test copies the docstring example, gets a rCloud that is wrong by many orders of magnitude, and the validator reports 'rCloud exceeds 200 pc' - blaming the cloud parameters rather than the unit mismatch.",
    "repro": "Run the two docstring examples verbatim and compare rCloud against a code-unit conversion of the same physical cloud.",
    "confidence": "high"
  },
  {
    "id": "S2-B-17",
    "file": "trinity/cloud_properties/validate_gmc.py",
    "line": 3,
    "class": "numerical",
    "severity": "S4",
    "claim": "The check_gmc_constraints usage example omits nISM - a parameter documented with no default - and uses mCloud=1e5 with M_computed=1.001e5, i.e. a relative mass error sitting exactly on the default 0.001 tolerance boundary.",
    "evidence": "validate_gmc.py:3 'issues = check_gmc_constraints(rCloud=150.0, nEdge=0.5, mCloud=1e5, M_computed=1.001e5,)'; validate_gmc.py:191 lists 'nISM : float ISM background density (same unit system as nEdge)' with no stated default, and 'mass_tolerance : float Maximum relative mass error (default 0.001 = 0.1%)'.",
    "expected": "The example should be runnable. Either nISM has a default (undocumented) or the example raises TypeError. Separately, |1.001e5 - 1e5|/1e5 evaluates to 1.0000000000001455e-3 in IEEE double, so a strict '> tolerance' comparison flags the module's own example as a mass-consistency failure.",
    "failure_scenario": "The documented example either does not run, or reports an error for a cloud the author intended as passing - and any user calibrating parameters to 'exactly 0.1%' lands on the failing side of the comparison because of floating-point representation.",
    "repro": "Run the example verbatim; then check whether the comparison is '>' or '>=' against mass_tolerance and evaluate abs(1.001e5-1e5)/1e5 > 0.001 in Python.",
    "confidence": "medium"
  },
  {
    "id": "S2-B-18",
    "file": "trinity/cloud_properties/density_profile.py",
    "line": 56,
    "class": "other",
    "severity": "S4",
    "claim": "Doctest-style examples in two modules assert the exact Python type of scalar output: 'type(n)' -> \"<class 'float'>\" and 'type(M)' -> \"<class 'float'>\", plus \"<class 'numpy.ndarray'>\" for array input.",
    "evidence": "density_profile.py:56 '>>> n = get_density_profile(0.5, params) >>> type(n) <class \\'float\\'>'; mass_profile.py:137 '>>> M = get_mass_profile(5.0, params) >>> print(type(M)) # <class \\'float\\'>'; helper docstrings density_profile.py:45 / mass_profile.py:73 'Convert result back to scalar if input was scalar.'",
    "expected": "A builtin float, not np.float64 (np.float64 prints as <class 'numpy.float64'> and, under numpy>=2 repr rules, formats differently in output). If _to_output returns arr[0] or arr.item() the claim differs.",
    "failure_scenario": "Downstream code or JSON serialisation that relies on the documented builtin-float contract (metadata.json writers, dictionary.jsonl) gets np.float64 and either fails to serialise or emits a different textual representation - relevant given the project's byte-identical dictionary.jsonl equivalence gate.",
    "repro": "assert type(get_density_profile(0.5, params)) is float and type(get_mass_profile(5.0, params)) is float.",
    "confidence": "medium"
  },
  {
    "id": "S2-B-19",
    "file": "trinity/cloud_properties/initial_profile.py",
    "line": 72,
    "class": "state",
    "severity": "S4",
    "claim": "build_initial_cloud_profile's docstring says nEdge is 'Optional pre-computed edge density (BE only ...)', but the implementation comment says the power-law path also uses it: '_init_powerlaw_cloud will populate it from (nCore, rCloud, rCore, alpha). Pre-seed for the rare edge-correction path that may read it.'",
    "evidence": "initial_profile.py:72 'nEdge Optional pre-computed edge density (BE only - initialised by _init_bonnor_ebert_cloud from the Lane-Emden solution).'; initial_profile.py:126 '# nEdge placeholder - _init_powerlaw_cloud will populate it from (nCore, rCloud, rCore, alpha). Pre-seed for the rare edge-correction path that may read it.'",
    "expected": "The parameter doc should state that nEdge participates in the PL path too (as an output of, and possible input to, the edge-correction branch).",
    "failure_scenario": "A caller reconstructing a PL profile omits nEdge on the strength of 'BE only'; if the placeholder pre-seed is ever removed or the edge-correction branch reads it before writing, the reconstruction silently differs from the original run's arrays.",
    "repro": "Call build_initial_cloud_profile for densPL with and without nEdge and diff the returned (r, n, m) arrays.",
    "confidence": "high"
  },
  {
    "id": "S2-B-20",
    "file": "trinity/cloud_properties/initial_profile.py",
    "line": 3,
    "class": "other",
    "severity": "S3",
    "claim": "initial_profile claims to be 'the inverse of trinity/phase0_init/get_InitCloudProp.py' and asserts idempotency of the phase-0 auto-correction: 'Calling the constructor with post-correction scalars is a no-op for the auto-correction branches (nEdge < nISM etc.) because those checks pass given the already-corrected inputs.'",
    "evidence": "initial_profile.py:3 (module docstring, quoted above); initial_profile.py:72 'Matches the layout produced by phase0_init.get_InitCloudProp.'; initial_profile.py:107 '# The phase-0 constructors mutate a few keys (rCore, rCloud, nEdge) via auto-correction.'",
    "expected": "A round-trip test: for every profile type and for parameter sets that DO trigger an auto-correction on the first pass, reconstructing from the post-correction scalars must reproduce the original (r, n, m) arrays bit-for-bit. The claim is an unproven invariant covering three mutated keys (rCore, rCloud, nEdge) and 'etc.' of unnamed branches.",
    "failure_scenario": "If any correction is not idempotent (e.g. a multiplicative safety margin like the 1.1 in compute_minimum_rCore applied a second time), every consumer that reconstructs arrays - plot scripts, cloudy exporter - silently plots/exports a different cloud than was simulated, with no error raised.",
    "repro": "Run a config that triggers the nEdge < nISM correction; persist the phase-0 arrays; rebuild via build_initial_cloud_profile from metadata.json scalars; np.testing.assert_array_equal.",
    "confidence": "medium"
  },
  {
    "id": "S2-B-21",
    "file": "trinity/cloud_properties/mass_profile.py",
    "line": 331,
    "class": "citation",
    "severity": "S4",
    "claim": "'Rahner+ 2018, Eq 25' is cited for the power-law enclosed-mass formula (and again in powerLawSphere for its inversion) with no volume/page, while the only fully-specified Rahner reference in the slice is 'Rahner et al. (2017): MNRAS 470, 4453' in bonnorEbertSphere.",
    "evidence": "mass_profile.py:331 '# Analytical result (Rahner+ 2018, Eq 25):'; powerLawSphere.py:78 'Uses the analytical inversion of the enclosed-mass formula (Rahner+ 2018 Eq 25)'; powerLawSphere.py:149 '# ----- Fixed rCore: analytical inversion (Rahner+ 2018 Eq 25) -----'; bonnorEbertSphere.py:3 '- Rahner et al. (2017): MNRAS 470, 4453'.",
    "expected": "A complete reference for the 2018 citation, and confirmation that Eq 25 of that paper is the piecewise power-law enclosed mass M = 4*pi*rho_c[rc^3/3 + (r^(3+a)-rc^(3+a))/((3+a)rc^a)]. Two different Rahner years for related WARPFIELD physics inside one package should be deliberate, not a typo.",
    "failure_scenario": "Wrong equation number or wrong paper propagates into the methods section of a publication; a reader cannot verify the prefactor or the (3+alpha) convention.",
    "repro": "Check Eq 25 of the intended Rahner et al. paper against the formula as written, including whether their rho_c is a mass or number density.",
    "confidence": "medium"
  },
  {
    "id": "S2-B-22",
    "file": "trinity/cloud_properties/bonnorEbertSphere.py",
    "line": 502,
    "class": "state",
    "severity": "S3",
    "claim": "create_BE_sphere_from_params documents an explicit 'Updates params with:' list of 10 keys, but two comments describe further writes not in that list: a sigma [km/s] key derived from c_s, and unspecified 'derived cloud properties'.",
    "evidence": "bonnorEbertSphere.py:502 \"Updates params with: 'densBE_Teff','densBE_xi_arr','densBE_u_arr','densBE_dudxi_arr','densBE_rho_rhoc_arr','densBE_f_rho_rhoc','densBE_f_m','densBE_xi_out','rCloud','nEdge'\"; bonnorEbertSphere.py:564 '# c_s [cm/s] -> sigma [km/s]'; bonnorEbertSphere.py:573 '# Also update derived cloud properties'; bonnorEbertSphere.py:552 '# Ensure all BE-specific params exist (safety fallback for standalone usage)'.",
    "expected": "The side-effect list must name every mutated key. A velocity-dispersion key silently overwritten by the BE constructor is exactly the kind of cross-module coupling that breaks when densBE is selected but a user has set sigma explicitly in their .param.",
    "failure_scenario": "A .param that sets a turbulent velocity dispersion has it silently overwritten by the BE sphere's c_s when dens_profile='densBE', changing the physics with no log line and no mention in the documented contract.",
    "repro": "Snapshot params keys before and after create_BE_sphere_from_params and diff against the documented list.",
    "confidence": "high"
  },
  {
    "id": "S2-B-23",
    "file": "trinity/cloud_properties/validate_gmc.py",
    "line": 287,
    "class": "other",
    "severity": "S4",
    "claim": "validate_gmc_params documents 'rCore : float, optional Core radius [pc] (required for densPL with alpha != 0)', while powerLawSphere.compute_rCloud_powerlaw advertises rCore as genuinely optional with a documented fallback: 'If None, uses rCore_fraction' (default 0.1).",
    "evidence": "validate_gmc.py:287 'rCore : float, optional Core radius [pc] (required for `densPL` with alpha != 0).'; powerLawSphere.py:78 'rCore : float, optional Fixed core radius [pc]. If None, uses rCore_fraction. rCore_fraction : float Ratio rCore/rCloud (default 0.1). Used when rCore is None.'",
    "expected": "One story about whether rCore=None is supported for densPL with alpha != 0. Note the two branches are not equivalent (S2-B-06): the fractional branch has no mass ceiling for alpha < -3 while the fixed branch does, so which one runs is physically observable.",
    "failure_scenario": "Validation rejects a parameter set (or raises) that the underlying solver would have handled via the 0.1 fraction, or vice versa - a capability advertised in one module and denied in another.",
    "repro": "validate_gmc_params(..., dens_profile='densPL', alpha=-2, rCore=None) and check whether it errors or falls back to rCore_fraction.",
    "confidence": "medium"
  },
  {
    "id": "S2-B-24",
    "file": "trinity/cloud_properties/mass_profile.py",
    "line": 318,
    "class": "units",
    "severity": "S4",
    "claim": "The ISM-region branch carries an unverified unit precondition stated twice as a hedge: 'ISM region - mCloud should be in Msun'. The M(r>rCloud) formula adds mCloud directly to (4/3)*pi*(r^3-r_cloud^3)*rho_ISM.",
    "evidence": "mass_profile.py:318 '# ISM region - mCloud should be in Msun'; mass_profile.py:338 '# Region 3: r > r_cloud (ISM) - mCloud should be in Msun'; mass_profile.py:271 'M(r) = M_cloud + (4/3)*pi*(r^3-r_cloud^3)*rho_ISM for r > r_cloud'.",
    "expected": "Either mCloud is [Msun] by the module's own unit contract (mass_profile.py:43) and the hedge should be deleted, or the value is not trusted and should be asserted. 'should be' twice in the same file is an author flag that the unit was not confirmed.",
    "failure_scenario": "If mCloud ever arrives in a different unit (e.g. raw from a .param before conversion), M(r) beyond rCloud is wrong by a constant offset with no error - and the offset only shows up once the shell leaves the cloud, i.e. late in a run.",
    "repro": "Assert params['mCloud'] units at entry to compute_enclosed_mass_powerlaw and compare M(rCloud+eps) - M(rCloud-eps) against 4*pi*rCloud^2*rho_ISM*2*eps.",
    "confidence": "medium"
  },
  {
    "id": "S2-B-25",
    "file": "trinity/cloud_properties/density_profile.py",
    "line": 144,
    "class": "regime",
    "severity": "S3",
    "claim": "The smoothing implementation assumes 'rCore is far below rCloud, so the smoothing band does not reach rCore in any realistic setup', but compute_minimum_rCore explicitly contains a branch that pushes rCore up towards - and clamps it at - rCloud.",
    "evidence": "density_profile.py:144 '# Inner core: constant density (rCore is far below rCloud, so the smoothing band does not reach rCore in any realistic setup)'; mass_profile.py:613 '# rCore_min = rCloud * (nCore/nISM)^(1/alpha)'; mass_profile.py:617 '# Apply safety margin'; mass_profile.py:620 \"# Ensure rCore doesn't exceed rCloud (pathological case)\".",
    "expected": "Either an assertion/warning when rCore > (1 - 2*SMOOTH_FRAC)*rCloud, or a smoothing implementation that is correct when the band overlaps the core. The two modules disagree on whether rCore ~ rCloud is reachable: one calls it impossible 'in any realistic setup', the other codes a clamp for it.",
    "failure_scenario": "Shallow alpha with a small nCore/nISM ratio drives rCore_min close to rCloud (rCore_min = rCloud*(nCore/nISM)^(1/alpha) -> rCloud as the ratio -> 1); the 1%-wide tanh band then overlaps the uniform core and the profile is blended in a regime the comment says cannot occur.",
    "repro": "Pick nCore/nISM ~ 1.2 and alpha = -0.2 so rCore_min > 0.99*rCloud, then evaluate get_density_profile across [0.97, 1.01]*rCloud.",
    "confidence": "medium"
  },
  {
    "id": "S2-B-26",
    "file": "trinity/cloud_properties/mass_profile.py",
    "line": 608,
    "class": "silent-failure",
    "severity": "S4",
    "claim": "For the homogeneous branch of compute_minimum_rCore the prose states 'nEdge = nCore, always valid if nCore > nISM' and 'Default: 10% of cloud radius'. The nCore > nISM condition is stated as a premise, not as a check, while the function's documented return includes 'is_valid : bool Whether nEdge >= nISM'.",
    "evidence": "mass_profile.py:608 '# Homogeneous: nEdge = nCore, always valid if nCore > nISM'; mass_profile.py:609 '# Default: 10% of cloud radius'; mass_profile.py:567 'is_valid : bool Whether nEdge >= nISM'.",
    "expected": "The alpha=0 branch should still evaluate nEdge >= nISM rather than assuming it. A cloud with nCore <= nISM is physically degenerate and should be reported, not assumed away.",
    "failure_scenario": "alpha=0 with nCore <= nISM returns is_valid=True on the strength of an unchecked premise, so the nEdge >= nISM constraint (validate_gmc constraint #2) is bypassed for homogeneous clouds.",
    "repro": "compute_minimum_rCore(nCore=nISM/2, nISM=nISM, rCloud=..., alpha=0.0) and inspect is_valid.",
    "confidence": "low"
  },
  {
    "id": "S2-B-27",
    "file": "trinity/cloud_properties/density_profile.py",
    "line": 126,
    "class": "numerical",
    "severity": "S4",
    "claim": "Quantitative claims about the smoothing bridge that are stated but never bounded by a test in the prose: SMOOTH_FRAC default is 1%; the density jump at rCloud is '~10^3'; 'mass conservation holds to O(SMOOTH_FRAC^2)'; the width is 'well below physical uncertainty in cloud-edge structure'.",
    "evidence": "density_profile.py:3 'a tanh bridge of width SMOOTH_FRAC * rCloud (1% by default)'; density_profile.py:121 'jump by ~10^3 across r=rCloud'; density_profile.py:126 'mass conservation holds to O(SMOOTH_FRAC^2)'.",
    "expected": "O(SMOOTH_FRAC^2) mass error requires the blend to be antisymmetric in linear r about rCloud (the leading O(w) terms must cancel). If the blend is applied in log r, or is one-sided, or uses a non-symmetric argument, the error degrades to O(SMOOTH_FRAC) = 1% - which then exceeds the 0.1% mass tolerance of S2-B-02.",
    "failure_scenario": "A 1% mass-conservation error at the cloud edge silently fails validate_mass_at_rCloud (tolerance 0.001) for every run, or - if M(r) is computed analytically and never cross-checked against the smoothed rho - is never noticed at all.",
    "repro": "Numerically integrate 4*pi*r^2*get_mass_density(r) from 0 to 2*rCloud and compare against the analytic M; repeat with SMOOTH_FRAC halved and confirm the error drops by 4x, not 2x.",
    "confidence": "medium"
  },
  {
    "id": "S2-B-28",
    "file": "trinity/cloud_properties/mass_profile.py",
    "line": 3,
    "class": "other",
    "severity": "S4",
    "claim": "Correctness-assertion language concentrated on two formulas suggests both were previously wrong: 'Correct formula: dM/dt = 4*pi*r^2*rho(r)*v(r) for ALL profiles', 'No solver coupling (no dependency on array_t_now, etc.)', 'This formula is EXACT', 'This works for ALL density profiles!', and in the BE module 'CORRECT VERSION v2' / 'CORRECT mass formula: m(xi) = xi^2 du/dxi'.",
    "evidence": "mass_profile.py:3 (module docstring key features); mass_profile.py:442 'This formula is EXACT for any smooth density profile'; mass_profile.py:478 '# This works for ALL density profiles!'; bonnorEbertSphere.py:3 'Bonnor-Ebert Sphere Implementation - CORRECT VERSION v2'.",
    "expected": "These are grep-checkable invariants, not just rhetoric: (a) no reference to array_t_now or other solver history anywhere in mass_profile.py; (b) the dM/dt path genuinely shares rho(r) with the M(r) path for both profiles (see S2-B-10); (c) no stale v1 BE implementation still importable.",
    "failure_scenario": "A superseded code path survives alongside the 'CORRECT' one and is still reachable from some call site, so which formula runs depends on entry point.",
    "repro": "grep for array_t_now within trinity/cloud_properties/, and for any other Bonnor-Ebert implementation in the package.",
    "confidence": "low"
  }
]
```
