# S2 cloud properties — Lens C (what it should be)

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

**Slice.** `trinity/cloud_properties/` — power-law (`densPL`) and Bonnor–Ebert (`densBE`) cloud
density/mass profiles, the radius↔mass solvers, and the GMC validator.

**Method.** Derived from first principles plus `PHYSICS_SPEC.md` (SPEC-003, 005, 021, 060–066,
090–092) and the redacted signature list. **No implementation, comment, docstring or stripped copy
was read.** Every number below is either derived analytically in this document or computed here by
numerically integrating the isothermal Lane–Emden equation (DOP853, `rtol=1e-12`) and by
cross-checking the closed-form power-law mass integrals against adaptive quadrature (agreement to
machine precision, ≤4e-16 relative, for α = 0, −1, −2, −2.5, −3, −3.5). Literature access was
blocked; the Bonnor–Ebert critical numbers below are therefore **re-derived, not looked up** — which
makes them stronger, not weaker, since I do not depend on a remembered table.

---

## 1. Unit system this slice must speak

TRINITY's internal system is `[M⊙, pc, Myr]` (SPEC-090). Cloud *inputs*, however, are declared in
mixed units (SPEC-003): `mCloud` [M⊙], `nCore`/`nISM` [cm⁻³], `rCore` [pc], `Omega` [–],
`densPL_alpha` [–]. So this slice sits exactly on a unit boundary, which is the single most
productive place to look for bugs (SPEC-092).

| Quantity | Expected unit | Note |
|---|---|---|
| `r`, `rCore`, `rCloud` | pc | |
| `nCore`, `nISM`, `nEdge` | cm⁻³, **hydrogen nuclei** | not particles |
| number density returned by `get_density_profile` | cm⁻³ | same units as its `nCore` input |
| mass density returned by `get_mass_density` | M⊙ pc⁻³ (internal) or g cm⁻³ | must be a *fixed* multiple of `n` |
| `M(<r)` | M⊙ | |
| `Ṁ` | M⊙ Myr⁻¹ | if `rdot` is pc Myr⁻¹ |
| `c_s` (BE, back-solved) | pc Myr⁻¹ internally; km s⁻¹ is a 2.3% trap (SPEC-092.6) | |

**Derived conversion constants** (computed here, `pc = 3.0856775814913673e18 cm`,
`M⊙ = 1.98892e33 g`, `m_H = 1.6735575e-24 g`, `μ_H = mu_convert = 1.4`):

```
    ρ[g cm⁻³]     = 2.34298e-24 · n[cm⁻³]
    ρ[M⊙ pc⁻³]    = 0.0346101   · n[cm⁻³]
```

The μ used for `ρ ↔ n` is **mass per hydrogen nucleus, μ_H = 1.4**, and is *ionisation-independent*.
It is a different constant from the μ used in `P = ρ k T/(μ m_H)` (0.609 ionized, 2.33 molecular).
Interchanging them is a factor 2.3 (SPEC-092.1). For the BE sphere the *thermal* μ is the one that
enters `c_s`, and for a cold GMC it should be the **molecular** μ ≈ 2.33 (`mu_mol = 14/6`), not 1.4
— see §7 trap T-11.

---

## 2. Power-law sphere with a uniform core

### 2.1 Density

```
    n(r) = n_core                    ,  0 ≤ r ≤ r_core
         = n_core (r/r_core)^α       ,  r_core < r ≤ r_cloud
         = n_ISM                     ,  r > r_cloud
    ρ(r) = μ_H m_H n(r)
```

with α ≤ 0 (SPEC-060; the schema declares −2 ≤ α ≤ 0). The core is a *flattening* radius, not a
mass scale (SPEC-063). ρ is continuous at `r_core` **by construction** — `(r_core/r_core)^α = 1` for
every α — so no separate continuity coefficient is needed or permitted. ρ is **discontinuous at
`r_cloud`** by design (a downward jump `n_edge → n_ISM`); this is physical only if
`n_edge ≥ n_ISM`, which is precisely the validator's job.

### 2.2 Enclosed mass — the two branches

`M(r) = ∫₀ʳ 4πr′²ρ(r′)dr′`. Splitting at `r_core`:

**Core, r ≤ r_core**
```
    M(r) = (4π/3) ρ_core r³
```

**Envelope, r_core < r ≤ r_cloud — general branch, α ≠ −3**
```
    M(r) = (4π/3) ρ_core r_core³  +  4π ρ_core r_core^{−α} · ( r^{3+α} − r_core^{3+α} ) / (3+α)
```

**Envelope — logarithmic branch, α = −3 exactly**
```
    ∫ r′^{2+α} dr′ = ∫ r′^{−1} dr′ = ln(r/r_core)
    M(r) = (4π/3) ρ_core r_core³  +  4π ρ_core r_core³ · ln(r/r_core)
```

(For α = −3, `r_core^{−α} = r_core³`.) The general branch is the *analytic continuation* of the
logarithmic one: as ε ≡ 3+α → 0, `(r^ε − r_core^ε)/ε → ln(r/r_core)`, so the function is smooth
through α = −3 — but the **formula is not**: at ε = 0 it evaluates `0/0`, which is `nan` in numpy
(with a RuntimeWarning) and `ZeroDivisionError` in scalar Python. Both branches were verified
against quadrature to ≤4e-16 relative.

**Exterior, r > r_cloud**
```
    M(r) = M_cloud + (4π/3) ρ_ISM ( r³ − r_cloud³ )
```
This branch is **required by SPEC-021** (the shell keeps sweeping ambient gas after blow-out) and by
the exactness invariant `dM/dr = 4πr²ρ`, since `ρ(r>r_cloud) = ρ_ISM ≠ 0`. Note SPEC-061 writes
`M(<r) = M_cloud` for `r > r_cloud`, which **contradicts** SPEC-021 and violates `dM/dr = 4πr²ρ`.
The physically correct form is the accumulating one; a clamp to `M_cloud` would make the shell
inertia-less and gravity-less outside the cloud. **This is the highest-value single check in the
slice.**

### 2.3 Continuity / smoothness ledger

| At | ρ | M | dM/dr |
|---|---|---|---|
| `r_core` | continuous (automatic) | continuous | continuous |
| `r_cloud` | **jump** `n_edge → n_ISM` (expected) | continuous | jump (follows ρ) |
| `r = 0` | finite `= ρ_core` | `M(0) = 0` | 0 |

`dρ/dr` has a kink at `r_core` for α ≠ 0. That is expected and harmless; `M` is C¹ there.

### 2.4 Cloud radius — closed form, no root-find needed

Inverting `M(r_cloud) = M_cloud` for α ≠ −3 and `r_cloud > r_core`:

```
    r_cloud = [ r_core^{3+α} + (3+α) ( M_cloud r_core^{α} /(4π ρ_core) − r_core^{3+α}/3 ) ]^{1/(3+α)}
```

and for α = 0 this collapses to the homogeneous result `r_cloud = (3M_cloud/(4πρ_core))^{1/3}`.

If instead the core is specified as a **fraction** `f` of the cloud radius (`rCore_fraction`,
default 0.1), the system is still closed-form, because `M ∝ R³` exactly:

```
    M_cloud = 4π ρ_core R³ K(f,α),     K(f,α) = f³/3 + f^{−α}(1 − f^{3+α})/(3+α)
    ⇒  R = [ M_cloud / (4π ρ_core K(f,α)) ]^{1/3}
```

Verified numerically: `f = 0.1`, `n_core = 1e5`, `M = 1e7 M⊙` gives `r_cloud = 8.8375 / 16.6514 /
29.1034 pc` for α = 0 / −1 / −2, each reproducing `M(r_cloud) = 1.0000000e7` to 8 digits.
Sanity: `K(0.1, 0) = 1/3` exactly ✓.

A root-finder is *permitted* (the nested `mass_at_radius(rCloud_guess, rCore_val)` suggests one) but
must agree with the closed form to ≪ `MASS_TOLERANCE`, and its bracket must be valid — see trap
T-05.

### 2.5 Mass accretion rate

`M` depends on `t` only through `r`, so by the chain rule, exactly:

```
    Ṁ(r) = dM/dr · ṙ = 4π r² ρ(r) · ṙ
```

with `ρ(r)` the **local** density — `ρ_ISM` when `r > r_cloud`, `ρ_core` when `r < r_core`. There is
no other correct expression; any prefactor other than 4π, or any use of a mean rather than local
density, is a bug. This is the sweep-up term `Ṁ_sh` of SPEC-020/021, and its correctness is a
prerequisite for the ram-pressure double-counting check.

### 2.6 Minimum core radius for `n_edge ≥ margin · n_ISM`

Requiring `n_core (r_cloud/r_core)^α ≥ q·n_ISM` with `q = margin` and α < 0 (dividing by a negative
number flips the inequality):

```
    r_core ≥ r_cloud · ( margin · n_ISM / n_core )^{−1/α}
```

Verified: `n_core = 1e5, n_ISM = 1, r_cloud = 100 pc, α = −2, margin = 1.1` ⇒ `r_core ≥ 0.331662 pc`,
at which `n_edge = 1.1000` exactly ✓. **α = 0 makes `−1/α` a division by zero**; the α = 0 case has
no constraint at all (`n_edge = n_core ≥ n_ISM` always, for any sane cloud) and must be
short-circuited *before* the exponent is formed. α = 0 is the schema **default**.

---

## 3. Bonnor–Ebert sphere

### 3.1 The equation and its boundary conditions

Isothermal hydrostatic equilibrium, `dP/dr = −Gm(r)ρ/r²` with `P = ρ c_s²`, becomes the isothermal
Lane–Emden equation for `ψ` defined by `ρ = ρ_c e^{−ψ}`:

```
    (1/ξ²) d/dξ ( ξ² dψ/dξ ) = e^{−ψ}          i.e.   ψ'' + (2/ξ)ψ' = e^{−ψ}
    ψ(0) = 0 ,   ψ'(0) = 0
    r = ξ r₀ ,   r₀ = c_s / sqrt(4πGρ_c) ,   c_s² = k_B T /(μ m_H)   [ISOTHERMAL sound speed]
```

Note this is *not* the polytropic Lane–Emden equation; there is no polytropic index and **γ plays no
role in the hydrostatic structure**. `ψ` is monotonically increasing for all ξ > 0 (verified
numerically over ξ ∈ [0, 300]), so `ρ(ξ)` is strictly decreasing and the map `Ω ↔ ξ_out` is a
bijection on Ω > 1.

**Initial conditions.** `ξ = 0` is a regular singular point (`2/ξ` blows up), so integration must
start at a small `ξ_min > 0` using the series solution. Matching order by order:

```
    ψ(ξ)  = ξ²/6 − ξ⁴/120 + ξ⁶/1890 + O(ξ⁸)
    ψ'(ξ) = ξ/3  − ξ³/30  + ξ⁵/315  + O(ξ⁷)
```

(Derivation: with ψ = aξ²+bξ⁴+cξ⁶, the LHS is `6a + 20bξ² + 42cξ⁴` and the RHS is
`1 − aξ² + (a²/2 − b)ξ⁴`, giving a = 1/6, b = −1/120, c = 1/1890.) Starting instead from
`ψ = ψ' = 0` at `ξ_min` incurs an error `O(ξ_min²/6)` — negligible for `ξ_min ≲ 1e-6`, but the
series costs nothing.

### 3.2 Enclosed mass — free from the ODE

The ODE *is* the mass integral: `d/dξ(ξ²ψ') = ξ²e^{−ψ}`, so

```
    m(ξ) ≡ ξ² ψ'(ξ) = ∫₀^ξ ξ′² e^{−ψ} dξ′        (dimensionless enclosed mass)
    M(<r) = 4π ρ_c r₀³ m(ξ)
```

No quadrature is needed and none should be used — `ξ²ψ'` is already the answer, and it is exact.
`m(ξ)` is strictly increasing (verified), `m(0) = 0`, and asymptotically `m → 2ξ`,
`ρ/ρ_c → 2/ξ²` (the singular isothermal sphere; at ξ = 190 I get `m = 359.6` vs `2ξ = 380` and
`ρ/ρ_c = 5.598e-5` vs `2/ξ² = 5.540e-5` — converging with the classic damped oscillation).

Equivalently, and *preferably* for a code that normalises to a requested cloud mass:
`M(<r) = M_cloud · m(ξ)/m(ξ_out)` (SPEC-064 cross-check) — this makes
`M(r_cloud) = M_cloud` exact by construction.

### 3.3 Critical values — computed here, not looked up

I integrated the ODE and maximised the fixed-external-pressure mass
`m_P(ξ) ≡ M G^{3/2}P_ext^{1/2}/c_s⁴ = m(ξ) e^{−ψ(ξ)/2}/√(4π)`:

| Quantity | Value I computed | Precision I stand behind | Common short form |
|---|---|---|---|
| `ξ_crit` | **6.4507514** | ±1e-6 | 6.451 |
| `Ω_crit = ρ_c/ρ_edge = e^{ψ(ξ_crit)}` | **14.042032** | ±1e-5 | 14.04 |
| `m_crit = ξ²ψ'` at ξ_crit | **15.704374** | ±1e-5 | 15.7 |
| `M_BE coefficient` in `M = m₁ c_s⁴/(G^{3/2}P_ext^{1/2})` | **1.1822266** | ±1e-6 | 1.18 |
| `M G/(R c_s²) = m_crit/ξ_crit` | **2.4345031** | ±1e-6 | 2.4 |
| `M = C c_s³/(G^{3/2}ρ_c^{1/2})`, `C = m_crit/√(4π)` | **4.4301221** | ±1e-6 | 4.43 |
| `ξ_crit²` (the classic confusion) | 41.6122 | — | — |

**These constants are not independent.** Three exact relations must hold among any set the code
stores — this is the cheapest possible audit of a constants block:

```
    Ω_crit          = exp( ψ(ξ_crit) )
    m_crit          = ξ_crit² ψ'(ξ_crit)
    M_BE_coeff      = m_crit / sqrt( 4π · Ω_crit )       ( 15.704374 / 13.28414 = 1.1822266 ✓ )
    M G/(R c_s²)    = m_crit / ξ_crit
```

`ξ_crit` is a *maximum* of `m_P(ξ)`, i.e. `dM/dξ|_{P_ext} = 0`, **not** a zero of anything and
**not** a root of ψ. I confirmed it is the global maximum on ξ ∈ [0, 300] (only one local maximum;
`m_P` then settles toward ≈0.798 with damped oscillation). Spheres with Ω > 14.042 are
gravitationally unstable (SPEC-065); the shipped default `Ω = 14.1` is **marginally supercritical**
(ξ_out = 6.4617), i.e. formally not an equilibrium — a documented-flag item, not a code bug.

Confidence: **high** on all of these — they are my own integration, reproducible in ~1 s, and they
agree with the values SPEC-065 recovered by search (6.451 / 14.04 / 1.18), which is an independent
cross-check.

### 3.4 What `create_BE_sphere(M_cloud, n_core, Omega, …)` must do

The parameterisation is `(M_cloud, n_core, Ω)` and the scale is **back-solved** (SPEC-066). The only
self-consistent chain is:

```
    1.  ρ_c    = μ_H m_H n_core                             (mass per H nucleus, μ_H = 1.4)
    2.  ξ_out  : ψ(ξ_out) = ln Ω                            (unique root; ψ monotone)
    3.  m_out  = ξ_out² ψ'(ξ_out)
    4.  r₀     = [ M_cloud / (4π ρ_c m_out) ]^{1/3}         from M = 4πρ_c r₀³ m_out
    5.  r_cloud= ξ_out r₀
    6.  c_s    = r₀ sqrt(4πGρ_c)                            (inverting r₀'s definition)
    7.  T      = μ_thermal m_H c_s² / k_B                   ← γ MUST NOT appear here
```

Steps 4 and 6 are the *only* places the requested mass enters; there is nothing to iterate and
nothing to root-find beyond step 2. Step 7 is an **output diagnostic**: for a real GMC it must land
at T ≈ 10–30 K (SPEC-066 / spec test T15). If the code instead forms `c_s² = γ k_B T/(μ m_H)` the
implied T is wrong by the factor γ = 5/3 (−40%), because the Lane–Emden equation from which `r₀`
came is *isothermal* by construction. The `gamma` argument is legitimate only for a separate
stability/energetics diagnostic, never for the structure.

Numeric anchors for `Ω → ξ_out` (from my integration; a code's tabulated inversion must reproduce
these):

| Ω | 2 | 10 | 14.04 | 14.1 | 100 | 1e3 | 1e4 |
|---|---|---|---|---|---|---|---|
| ξ_out | 2.274573 | 5.596856 | 6.450368 | 6.461675 | 14.215598 | 40.817591 | 139.417794 |
| m(ξ_out) | 2.56528 | 13.10124 | 15.70324 | 15.73673 | 34.29721 | 80.53216 | 259.13763 |

Note how fast ξ_out grows: any tabulated `XI_MAX` below ~140 silently cannot represent Ω ≳ 1e4, and
below ~41 cannot represent Ω ≳ 1e3.

### 3.5 `r_to_xi` / `xi_to_r`

```
    ξ = r · sqrt(4πGρ_c) / c_s        r = ξ · c_s / sqrt(4πGρ_c)
```

These are exact inverses; the round-trip must be the identity to floating-point. The unit trap is
`c_s`: with `G` in `pc³ M⊙⁻¹ Myr⁻²` and `ρ_c` in `M⊙ pc⁻³`, `sqrt(4πGρ_c)` is `Myr⁻¹`, so `c_s`
must be `pc Myr⁻¹`. Feeding km s⁻¹ gives a silent **2.3%** error in every radius (SPEC-092.6) —
small enough to hide, large enough to matter through `M ∝ r₀³` (7%).

---

## 4. Invariants the implementation must satisfy

These hold **regardless of implementation choice** and are the most valuable output of this lens.
Each is a one-line executable test.

**I-1 — Exactness of the mass integral.** `d/dr M(r) = 4π r² ρ(r)` at every `r` (excluding the
`r_cloud` jump), for **both** profiles and **all** α. Test: central difference of the coded `M`
against `4πr²·` the coded `ρ`, relative error < 1e-6 on a log grid spanning `0.1 r_core → 3 r_cloud`.
This single test catches: a wrong `(3+α)` denominator, a missing `r_core^{−α}` normalisation, a
dropped core term, a wrong exterior branch, and any μ inconsistency between the two functions.

**I-2 — `M(0) = 0`** exactly, and `M(r) → (4π/3)ρ_core r³` as `r → 0`.

**I-3 — Monotonicity.** `M` strictly increasing; `ρ ≥ 0` everywhere; `ρ` non-increasing everywhere
(including across `r_cloud`, which requires `n_edge ≥ n_ISM`). `m(ξ)` and `ψ(ξ)` strictly increasing.

**I-4 — Normalisation.** `M(r_cloud) = M_cloud` to ≤ `MASS_TOLERANCE` (1e-3) — and in practice to
1e-12, since both profiles admit an exact construction. `validate_mass_at_rCloud` must compare
against the *same* mass that `compute_rCloud_*` was given (post-SFE vs total, SPEC-005): whichever
convention is chosen, the two must be the same number, or the check validates nothing.

**I-5 — Inversion round-trip.** `r_cloud(M_cloud(r)) = r` and `M(r_cloud(M)) = M`;
`xi_to_r(r_to_xi(r)) = r`; `r2xi`/`xi2r` agree with `r_to_xi`/`xi_to_r` given the same params.

**I-6 — α = 0 reduction.** `compute_rCloud_powerlaw(..., α=0)` ≡ `compute_rCloud_homogeneous(...)`
bit-for-bit-ish (≤1e-12 relative), and the power-law `M(r)` reduces to `(4π/3)ρ_core r³` for α = 0
at every `r ≤ r_cloud`, independent of `r_core`. Since α = 0 is the **default**, this is the most
frequently exercised path in the whole slice.

**I-7 — `r_core` inertness at α = 0.** No output may depend on `r_core` when α = 0 (SPEC-063).
Test: run the profile builder with `rCore = 0.01` and `rCore = 5` at α = 0 and diff every returned
array.

**I-8 — Branch continuity in α.** `M(r; α = −3 + ε) → M(r; α = −3)` as ε → 0 from either side. The
analytic limit is exact; I measured the general formula's agreement with the log branch at
`3+α = 1e-6` (1.7e-6 relative, truncation) and `1e-12` (4.5e-7 relative, cancellation), so any
sane branch-switch tolerance in `[1e-10, 1e-6]` is safe. At `3+α = 0` the general formula is `0/0`.

**I-9 — Constants self-consistency (BE).** `M_BONNOR_CRITICAL = M_DIM_CRITICAL/sqrt(4π·OMEGA_CRITICAL)`
and `OMEGA_CRITICAL = exp(ψ(XI_CRITICAL))` and `M_DIM_CRITICAL = XI_CRITICAL²·ψ'(XI_CRITICAL)`,
where ψ comes from the module's *own* solver. If the stored constants and the solver disagree, one
of them is wrong.

**I-10 — Ω round-trip (BE).** `exp(ψ(ξ_out(Ω))) = Ω` and `ρ(ξ_out)/ρ_c = 1/Ω` to solver tolerance,
for Ω ∈ {2, 10, 14.04, 14.1, 100, 1000}.

**I-11 — Accretion-rate consistency.** `compute_mass_accretion_rate(r, ṙ) == 4πr²·get_mass_density(r)·ṙ`
identically, and `get_mass_profile(r, return_mdot=True, rdot=ṙ)[1]` must equal it. Two code paths
computing the same thing must agree.

**I-12 — Numerical == analytic.** `compute_enclosed_mass(r_arr, rho_arr)` (generic quadrature) must
agree with `compute_enclosed_mass_powerlaw(r_arr)` to the grid's own accuracy — *including the
contribution from `0 → r_arr[0]`*, which a bare `cumtrapz` silently drops (see trap T-06).

**I-13 — Density/mass-density ratio is constant.** `get_mass_density(r)/get_density_profile(r)` must
be `r`-independent and equal to `μ_H m_H` in the appropriate unit pair (0.0346101 M⊙ pc⁻³ per cm⁻³,
or 2.34298e-24 g cm⁻³ per cm⁻³). Any `r`-dependence means a region uses a different μ.

**I-14 — Shape/type transparency.** The `_is_scalar`/`_to_array`/`_to_output` trio implies a
contract: scalar in ⇒ scalar out; array in ⇒ array of the same shape out; and
`f(array)[i] == f(array[i])` elementwise. `np.float64` and 0-d arrays are the classic
`_is_scalar` blind spots.

**I-15 — Exterior mass keeps growing.** `M(2 r_cloud) > M(r_cloud)` by exactly
`(4π/3)ρ_ISM(7 r_cloud³)`. (Follows from I-1, but stated separately because SPEC-021 and SPEC-061
disagree about it and it changes the post-blowout dynamics.)

---

## 5. Validity regime — what is physical, and what a validator should reject

### 5.1 Power law

| Parameter | Mathematically valid | Physically sensible for a GMC | Validator should |
|---|---|---|---|
| α | any α > −3 gives a convergent envelope with a core; α ≤ −3 also converges *because* of the core | −2 ≤ α ≤ 0; α = −1 to −2 matches observed clump profiles; α = −2 is the singular-isothermal limit | reject α > 0 (density increasing outward — not a cloud); reject α < −2 as out-of-schema; special-case α = −3 |
| `r_core` | 0 < r_core < r_cloud | ≈ 0.1–1 pc; `CLAUDE.md` says ≈1 pc | reject r_core ≥ r_cloud; **warn** on r_core ≲ 0.01 pc with α ≠ 0 (SPEC-063: the default 0.01 pc gives n(1 pc) = 1e-4 n_core at α = −2 — mathematically fine, physically a point source) |
| `n_edge` | any | ≥ n_ISM | reject `n_edge < n_ISM` — otherwise ρ *increases* at the cloud edge and the "cloud" is a cavity |
| `r_cloud` | any | ≲ 200 pc (`R_CLOUD_MAX`) | reject above `r_max` |
| `M_cloud` vs core | `M_cloud` must exceed the core mass `(4π/3)ρ_core r_core³` for the envelope branch to apply | | fall back to the homogeneous inversion when `M_cloud ≤ M_core`, **not** raise/NaN |
| α ≤ −3 (if ever allowed) | envelope mass **converges** as r → ∞ to `(4π/3)ρ_c r_core³·(1 − 3/(3+α))` | never | reject: a requested `M_cloud` above that ceiling has **no root**, and a bracketed solver will return the bracket edge or diverge |

Physically-implausible-but-accepted note: `nCore = 1e5 cm⁻³` with `mCloud = 1e7 M⊙` and α = 0 gives
`r_cloud = 8.84 pc` — a cloud with a *mean* density of 1e5 cm⁻³, which is clump/hot-core density,
not GMC density (~1e2 cm⁻³). It passes every constraint above. That is a documentation/regime issue,
not a code bug, but it means the validator's silence is not evidence of plausibility.

### 5.2 Bonnor–Ebert

| Parameter | Valid | Should reject / flag |
|---|---|---|
| Ω | Ω > 1 strictly (Ω = 1 ⇒ ξ = 0 ⇒ zero-size sphere) | reject Ω ≤ 1; **flag** Ω > 14.042 as gravitationally unstable (default 14.1 is already over) |
| Ω upper bound | any, but ξ_out grows fast (Ω = 1e4 ⇒ ξ = 139.4) | reject Ω whose ξ_out exceeds the solver's `XI_MAX` — must fail loudly, never clamp |
| implied `T` | back-solved | flag if outside ~5–50 K: the profile is then a curve fit, not a cloud (SPEC-066 / T15) |
| `n_core` | > n_ISM | reject otherwise |
| γ | 5/3 | must not enter the hydrostatic structure at all |

---

## 6. What a validator should check, in order

1. `n_core > n_ISM` and `n_edge ≥ n_ISM` (with the `margin`).
2. `0 < r_core < r_cloud` (PL); `1 < Ω` and `ξ_out ≤ XI_MAX` (BE).
3. `r_cloud ≤ R_CLOUD_MAX` (200 pc).
4. `|M(r_cloud) − M_cloud|/M_cloud ≤ MASS_TOLERANCE` (1e-3).
5. α within schema range; α = −3 handled or rejected.
6. Advisory: BE implied `T`; PL `r_core` plausibility; `Ω > Ω_crit`.

The suggestion machinery (`_suggest_*_alternatives`) is a convenience, not physics — its only hard
requirement is that every suggestion it returns actually **passes** the checks above when fed back
in (a self-consistency test that costs one loop).

---

## 7. Known traps, ranked

**T-01 (divergence, severity S3 as-scoped / S1 if α is unrestricted).** α = −3 evaluated with the
general formula: `(r^{3+α} − r_core^{3+α})/(3+α)` → `0/0`. numpy yields `nan` + RuntimeWarning;
scalar Python raises `ZeroDivisionError`. A `nan` mass propagates silently into `M_sh`, `F_grav` and
the EOM. Guard: `if abs(3+α) < tol: use the log branch`. Mitigating: the schema declares
−2 ≤ α ≤ 0, so this is reachable only if the range is unenforced or a sweep bypasses validation.

**T-02 (coefficient, S1).** `ξ_crit` confused with `ξ_crit² = 41.612`, or with `m_crit = 15.704`,
or truncated to 6.5. Because these live in the same constants block (`XI_CRITICAL`,
`M_DIM_CRITICAL`, `M_BONNOR_CRITICAL`, `OMEGA_CRITICAL`) the swap is easy and invariant I-9 catches
it instantly.

**T-03 (coefficient, S2).** The BE critical-mass coefficient mis-remembered. The three numbers that
get conflated are **1.18** (`M G^{3/2}P_ext^{1/2}/c_s⁴`), **2.43** (`MG/(R c_s²)`) and **4.43**
(`M G^{3/2}ρ_c^{1/2}/c_s³`). All three are "the Bonnor–Ebert mass coefficient" in some textbook's
notation. Which one is correct depends entirely on whether the sphere is normalised at fixed
*external pressure*, fixed *radius*, or fixed *central density*.

**T-04 (state/normalisation, S1).** Assuming `r_core ≪ r_cloud` and dropping the core term
`(4π/3)ρ_core r_core³` from the normalisation. With the default `rCore_fraction = 0.1` the core
holds `f³/3 / K(f,α)` of the mass — 0.1% at α = 0 but **3.6%** at α = −2 — well above
`MASS_TOLERANCE = 1e-3`. Symmetrically, dropping the core and integrating the pure power law from
0 diverges for α ≤ −3 and *over*-estimates the mass for α > −3.

**T-05 (numerical, S2).** Root-finding `r_cloud` when a closed form exists (§2.4). Two failure
modes: (i) the bracket `[r_core, r_max]` does not contain the root — for a very dense core the root
can lie *below* `r_core`, where the objective uses the wrong branch; (ii) with `rCore_fraction`, the
objective is self-referential (`r_core = f·r_cloud` changes as the guess changes) and a solver that
holds `r_core` fixed at a first guess converges to the wrong radius. The closed form
`R = [M/(4πρ_c K(f,α))]^{1/3}` is exact and is the reference.

**T-06 (silent-failure, S2).** Generic `compute_enclosed_mass` via `cumtrapz`/`cumulative_trapezoid`
returns an array of length `n−1`, or prepends 0 — either way `M(r[0]) = 0` rather than
`(4π/3)ρ(r[0])r[0]³`. On a log grid starting at `1e-3 r_core` this is negligible; on a grid starting
at `r_core` it silently loses the **entire core mass**. Also, trapezoid on `4πr²ρ` with a steep
α = −2 profile on a coarse *linear* grid overestimates by percent-level. Check against I-12.

**T-07 (units, S1).** μ in `ρ = μ m_H n`. Must be `mu_convert = 1.4` (per H nucleus). Using
μ_ion = 0.609 or μ_mol = 2.33 rescales every mass by 2.3× / 1.66×. See I-13.

**T-08 (units, S2).** BE `c_s` in km s⁻¹ where pc Myr⁻¹ is required: 2.3% in `r₀`, 7% in `M`.
Too small to look wrong, too big to ignore.

**T-09 (regime, S2).** `compute_minimum_rCore` with α = 0: `q^{−1/α}` divides by zero. α = 0 is the
default. Must short-circuit.

**T-10 (state, S1).** Exterior mass clamped to `M_cloud` for `r > r_cloud` (SPEC-061's wording)
instead of accumulating ISM (SPEC-021's requirement). Breaks `dM/dr = 4πr²ρ`, zeroes the sweep-up
term after blow-out, and makes `F_grav` and the shell inertia wrong in exactly the regime the
"escape vs re-collapse" fate is decided.

**T-11 (units/regime, S2).** The μ used to convert the back-solved `c_s` into a temperature. For a
BE *cloud* the thermal μ is molecular (≈2.33, `mu_mol = 14/6`), not the 1.4 used for `ρ ↔ n`. Using
1.4 reports `T` too low by 1.66×; using γ as well compounds it. `create_BE_sphere`'s `mu=1.4`
default is the `ρ ↔ n` value, so if the same argument feeds both conversions, one of the two is
wrong by construction.

**T-12 (numerical, S3).** Lane–Emden integrated from `ξ = 0` exactly (`2/ξ` → inf) or with a
too-coarse `N_POINTS` interpolant. The stored solution is used through an interpolant `f_m`,
`f_rho_rhoc`; linear interpolation of `m(ξ)` on a coarse grid injects percent-level mass errors that
I-4 would catch only if the tolerance is tight.

**T-13 (numerical, S3).** `lane_emden_ode(y, xi)` has `odeint`'s `(y, t)` argument order.
`scipy.integrate.solve_ivp` requires `(t, y)`. Passing this function to `solve_ivp` unwrapped
evaluates the RHS with the arguments transposed — which for this system produces a *plausible-looking
but wrong* solution rather than an exception.

**T-14 (regime, S3).** Ω exactly 1, or Ω < 1: `ln Ω ≤ 0` and the root-find for `ξ_out` either
returns 0 or fails. Ω ≤ 1 means the cloud is not centrally concentrated — reject at the schema.

---

## 8. Confidence ledger

- **Derived from scratch, high confidence:** every power-law formula (§2), both branches, the
  closed-form `r_cloud`, `K(f,α)`, `r_core,min`, `Ṁ = 4πr²ρṙ`, all §4 invariants, the Lane–Emden
  equation, its series initial conditions, `m(ξ) = ξ²ψ'`, `r₀ = c_s/√(4πGρ_c)`, and the §3.4
  back-solve chain. All cross-checked numerically here.
- **Computed here, high confidence to the digits quoted:** ξ_crit = 6.4507514, Ω_crit = 14.042032,
  m_crit = 15.704374, M_BE coeff = 1.1822266, 2.4345031, 4.4301221, and the Ω → ξ_out table.
  These normally need a table lookup; I integrated instead, and they agree with the values
  SPEC-065 recovered independently.
- **Medium confidence:** which of `M_DIM_CRITICAL` / `M_BONNOR_CRITICAL` is meant to be 15.704 vs
  1.182 (both are defensible names for both numbers) — but the *relation* between them (I-9) is
  exact and is the thing to test. Also medium: whether `gamma` in `create_BE_sphere` is a structural
  parameter or a diagnostic; the structure must not use it either way.
- **Low confidence / not derivable from this lens:** the intended semantics of `nEdge` as an
  *input* to `build_initial_cloud_profile` (it is derivable from the other parameters, so passing it
  in creates a second source of truth that must be checked for consistency); and whether `M_cloud`
  here is pre- or post-SFE (SPEC-005) — either is fine so long as it is the same everywhere.

```json
[
  {
    "id": "S2-C-01",
    "file": "trinity/cloud_properties/mass_profile.py",
    "line": 267,
    "class": "divergence",
    "severity": "S3",
    "claim": "The power-law enclosed mass must use the logarithmic branch when alpha == -3, not the general (3+alpha) formula.",
    "evidence": "M(r)=int 4 pi r'^2 rho dr' with rho = rho_c (r/r_core)^alpha gives int r'^(2+alpha) dr' = (r^(3+alpha)-r_core^(3+alpha))/(3+alpha) only for alpha != -3; at alpha = -3 the integrand is r'^-1 and the integral is ln(r/r_core), so M = (4pi/3)rho_c r_core^3 + 4pi rho_c r_core^3 ln(r/r_core). Both branches verified against adaptive quadrature to <=4e-16 relative.",
    "expected": "An explicit branch on |3+alpha| < tol (any tol in 1e-10..1e-6 is safe; measured agreement 4.5e-7 at 1e-12 and 1.7e-6 at 1e-6) selecting the logarithmic form.",
    "failure_scenario": "At alpha = -3 exactly the general expression evaluates 0/0: numpy returns nan with a RuntimeWarning, scalar Python raises ZeroDivisionError. A nan enclosed mass propagates silently into M_sh, F_grav and the shell EOM, producing a nan trajectory rather than an error. Reachable only if the declared range -2 <= alpha <= 0 is not enforced on every entry path (sweeps, direct API calls).",
    "repro": "M_powerlaw(r=30, rho_c, r_core=1, alpha=-3.0) and alpha=-3.0+1e-12; compare against 4pi*rho_c*r_core**3*(1/3 + log(30)) = analytic.",
    "confidence": "high"
  },
  {
    "id": "S2-C-02",
    "file": "trinity/cloud_properties/mass_profile.py",
    "line": 131,
    "class": "state",
    "severity": "S1",
    "claim": "For r > rCloud the enclosed mass must keep accumulating ambient gas: M(r) = M_cloud + (4pi/3) rho_ISM (r^3 - rCloud^3); it must not be clamped to M_cloud.",
    "evidence": "dM/dr = 4 pi r^2 rho(r) is an identity, and rho(r>rCloud) = rho_ISM != 0. SPEC-021 states M_sh = M_cloud_after + (4pi/3) rho_ISM (R2^3 - r_cloud^3) beyond the cloud edge; SPEC-061 writes M(<r) = M_cloud for r > r_cloud, which contradicts it. The accumulating form is the physically correct one.",
    "expected": "M(2*rCloud) - M(rCloud) = (4pi/3) rho_ISM * 7 * rCloud^3 exactly.",
    "failure_scenario": "A clamped exterior mass zeroes the sweep-up term Mdot_sh = 4 pi r^2 rho_ISM rdot after blow-out, so the shell gains no inertia and no self-gravity in the ISM. Escape-vs-recollapse and the momentum-phase deceleration are then wrong precisely in the regime where the run's fate is decided.",
    "repro": "Evaluate get_mass_profile at r = rCloud, 1.5 rCloud, 2 rCloud and finite-difference against 4 pi r^2 rho_ISM.",
    "confidence": "high"
  },
  {
    "id": "S2-C-03",
    "file": "trinity/cloud_properties/mass_profile.py",
    "line": 231,
    "class": "numerical",
    "severity": "S2",
    "claim": "The coded M(r) must be the exact integral of the coded rho(r): d/dr M = 4 pi r^2 rho at every r, for both profiles and all alpha.",
    "evidence": "Definition of enclosed mass. This one test simultaneously catches a wrong (3+alpha) denominator, a missing r_core^(-alpha) normalisation, a dropped core term, a wrong exterior branch, and any mu mismatch between the density and mass functions.",
    "expected": "Central-difference of M against 4 pi r^2 * get_mass_density(r) agrees to <1e-6 relative on a log grid from 0.1 r_core to 3 r_cloud, excluding the rCloud discontinuity.",
    "failure_scenario": "Any inconsistency means the swept mass and the local sweep-up rate describe different clouds; momentum and gravity terms in the EOM then disagree with each other and no single-run diagnostic reveals it.",
    "repro": "np.gradient(M(r_grid), r_grid) vs 4*np.pi*r_grid**2*rho(r_grid) on np.logspace over the cloud.",
    "confidence": "high"
  },
  {
    "id": "S2-C-04",
    "file": "trinity/cloud_properties/mass_profile.py",
    "line": 488,
    "class": "numerical",
    "severity": "S2",
    "claim": "M(rCloud) must equal the requested cloud mass; both profiles admit an exact construction so the achievable error is ~1e-12, far below the 1e-3 tolerance.",
    "evidence": "Power law: rCloud is the closed-form root of M(rCloud)=M_cloud, rCloud = [r_core^(3+a) + (3+a)(M r_core^a/(4 pi rho_c) - r_core^(3+a)/3)]^(1/(3+a)). BE: M(<r) = M_cloud m(xi)/m(xi_out) is exact by construction. Verified: f=0.1, n_core=1e5, M=1e7 gives rCloud = 8.8375/16.6514/29.1034 pc for alpha = 0/-1/-2 with M(rCloud) = 1.0000000e7 to 8 digits.",
    "expected": "Relative mass error < 1e-10 for the shipped examples; a tolerance of 1e-3 is a loose backstop, not the target. The mass compared against must be the same mass fed to compute_rCloud_* (pre- vs post-SFE, SPEC-005).",
    "failure_scenario": "If validate_mass_at_rCloud compares against a different normalisation than the solver used (total vs post-SFE mass), the check passes vacuously while rCloud is 11% wrong at sfe=0.3 (30% in swept mass).",
    "repro": "validate_mass_at_rCloud on param/cloud_example_PL.param and cloud_example_BE.param; print the absolute relative residual, not just pass/fail.",
    "confidence": "high"
  },
  {
    "id": "S2-C-05",
    "file": "trinity/cloud_properties/powerLawSphere.py",
    "line": 77,
    "class": "numerical",
    "severity": "S2",
    "claim": "compute_rCloud_powerlaw has a closed form and must agree with it; if a root-finder is used, its bracket and its treatment of rCore_fraction must be self-consistent.",
    "evidence": "With rCore fixed: rCloud = [r_core^(3+a) + (3+a)(M r_core^a/(4 pi rho_c) - r_core^(3+a)/3)]^(1/(3+a)). With rCore = f*rCloud the mass is still exactly cubic in R: M = 4 pi rho_c R^3 K(f,a), K = f^3/3 + f^(-a)(1-f^(3+a))/(3+a), so R = [M/(4 pi rho_c K)]^(1/3). K(0.1,0) = 1/3 exactly; verified to 8 digits for alpha = 0,-1,-2.",
    "expected": "Root-find result matches the closed form to <1e-12 relative. At alpha = 0 it must equal compute_rCloud_homogeneous exactly.",
    "failure_scenario": "(i) If rCore is held at a first guess while rCloud iterates, the fixed point is wrong (rCore = f*rCloud is self-referential). (ii) If the bracket is [rCore, r_max] but M_cloud is below the core mass (4pi/3)rho_c r_core^3, the root lies inside the core and the objective uses the wrong branch, returning the bracket edge or raising.",
    "repro": "Compare compute_rCloud_powerlaw(1e7, 1e5, alpha, rCore_fraction=0.1) against (M/(4 pi rho_c K(0.1,alpha)))**(1/3) for alpha in {0,-0.5,-1,-1.5,-2}; then retry with M_cloud smaller than the core mass.",
    "confidence": "high"
  },
  {
    "id": "S2-C-06",
    "file": "trinity/cloud_properties/powerLawSphere.py",
    "line": 51,
    "class": "units",
    "severity": "S1",
    "claim": "rho_core = mu_convert * m_H * nCore with mu_convert = 1.4 (mass per hydrogen nucleus), and the homogeneous radius is (3 M/(4 pi rho_core))^(1/3).",
    "evidence": "SPEC-092.1: the mu for rho<->n is mass per H nucleus and is ionisation-independent; it is a different constant from the mu in P = rho k T/(mu m_H). Computed conversion: rho[Msun/pc^3] = 0.0346101 n[cm^-3], rho[g/cm^3] = 2.34298e-24 n[cm^-3] at mu = 1.4.",
    "expected": "The mu default in the signature (mu=1.4) is the rho<->n constant everywhere in this module; rho_core/nCore is a fixed number equal to 0.0346101 Msun/pc^3 per cm^-3.",
    "failure_scenario": "Using mu_ion = 0.609 or mu_mol = 2.33 rescales every cloud mass and radius: M is wrong by 2.3x (or 1.66x), rCloud by 1.32x (or 1.18x), and every downstream swept mass and gravitational force inherits it. The error is a clean multiplicative factor, so the run still looks physical.",
    "repro": "Assert compute_rCloud_homogeneous(1e7, 1e5) == (3e7/(4*pi*0.0346101*1e5))**(1/3) = 8.8375 pc.",
    "confidence": "high"
  },
  {
    "id": "S2-C-07",
    "file": "trinity/cloud_properties/mass_profile.py",
    "line": 566,
    "class": "divergence",
    "severity": "S3",
    "claim": "compute_minimum_rCore must be rCloud * (margin*nISM/nCore)^(-1/alpha), with an explicit short-circuit for alpha == 0.",
    "evidence": "Requiring nCore (rCloud/rCore)^alpha >= margin*nISM and dividing by the negative alpha flips the inequality, giving rCore >= rCloud * q^(-1/alpha) with q = margin*nISM/nCore. Verified: nCore=1e5, nISM=1, rCloud=100, alpha=-2, margin=1.1 gives rCore >= 0.331662 pc, at which nEdge = 1.1000 exactly.",
    "expected": "alpha = 0 returns 'no constraint' (any rCore) without forming -1/alpha. alpha = 0 is the schema default, so this is the most-hit path.",
    "failure_scenario": "-1/0 raises ZeroDivisionError or yields inf, so q**inf underflows to 0 or overflows, returning rCore_min = 0 or inf. Either silently disables the nEdge >= nISM guard for the default profile.",
    "repro": "compute_minimum_rCore(1e5, 1.0, 100.0, 0.0) and compute_minimum_rCore(1e5, 1.0, 100.0, -2.0).",
    "confidence": "high"
  },
  {
    "id": "S2-C-08",
    "file": "trinity/cloud_properties/bonnorEbertSphere.py",
    "line": 179,
    "class": "exponent",
    "severity": "S1",
    "claim": "The ODE right-hand side must be dpsi/dxi = phi, dphi/dxi = exp(-psi) - 2 phi/xi (isothermal Lane-Emden), with no polytropic index.",
    "evidence": "Isothermal hydrostatic equilibrium dP/dr = -G m rho/r^2 with P = rho c_s^2 and rho = rho_c exp(-psi) gives (1/xi^2) d/dxi(xi^2 dpsi/dxi) = exp(-psi), xi = r/r0, r0 = c_s/sqrt(4 pi G rho_c). The polytropic Lane-Emden (theta^n on the RHS) is a different equation and would give a finite-radius sphere.",
    "expected": "RHS exactly [phi, exp(-psi) - 2*phi/xi]; psi monotonically increasing (verified over xi in [0,300], min phi > 0); rho/rho_c -> 2/xi^2 and m -> 2 xi asymptotically (at xi = 190 I get rho/rho_c = 5.598e-5 vs 5.540e-5, m = 359.6 vs 380).",
    "failure_scenario": "A sign error on the 2/xi term or an exp(+psi) makes the profile increase outward or blow up; a polytropic RHS truncates the sphere at a finite xi and the Omega -> xi inversion then fails for large Omega.",
    "repro": "Integrate the module's ODE from the series ICs and compare psi(6.4507514) against 2.6420551.",
    "confidence": "high"
  },
  {
    "id": "S2-C-09",
    "file": "trinity/cloud_properties/bonnorEbertSphere.py",
    "line": 206,
    "class": "numerical",
    "severity": "S3",
    "claim": "Integration must start at xi_min > 0 with the series initial conditions psi = xi^2/6 - xi^4/120 + xi^6/1890, psi' = xi/3 - xi^3/30 + xi^5/315.",
    "evidence": "xi = 0 is a regular singular point (the 2/xi term diverges). Series derived here by matching orders: with psi = a xi^2 + b xi^4 + c xi^6 the LHS is 6a + 20b xi^2 + 42c xi^4 and the RHS 1 - a xi^2 + (a^2/2 - b) xi^4, giving a = 1/6, b = -1/120, c = 1/1890.",
    "expected": "get_initial_conditions(xi0) returns exactly those two truncated series (any consistent truncation order is fine). Using psi = psi' = 0 at xi0 incurs an O(xi0^2/6) error - negligible for xi0 <= 1e-6, so this is only a precision item unless xi0 is large.",
    "failure_scenario": "xi0 = 0 gives a division by zero on the first RHS evaluation; a mis-signed series coefficient shifts psi by O(xi0^4) which is invisible at xi0 = 1e-6 but corrupts the solution if xi0 is O(0.1).",
    "repro": "Check get_initial_conditions(1e-3) against (1e-6/6 - 1e-12/120, 1e-3/3 - 1e-9/30).",
    "confidence": "high"
  },
  {
    "id": "S2-C-10",
    "file": "trinity/cloud_properties/bonnorEbertSphere.py",
    "line": 79,
    "class": "coefficient",
    "severity": "S2",
    "claim": "XI_CRITICAL must be 6.45075 (not its square 41.612, not m_crit = 15.704).",
    "evidence": "Computed here by maximising m_P(xi) = m(xi) exp(-psi/2)/sqrt(4 pi) (the fixed-external-pressure dimensionless mass) with a Brent search on a DOP853 solution at rtol 1e-12: xi_crit = 6.4507514, confirmed as the global maximum on [0,300]. Agrees with the literature value 6.451 recovered independently in SPEC-065.",
    "expected": "6.4507514 +/- 1e-6, i.e. at least 6.451.",
    "failure_scenario": "If the stability check uses xi^2 = 41.6 or m_crit = 15.7 as the critical radius, every cloud is declared stable (or unstable) regardless of Omega, and the Omega > Omega_crit warning becomes noise.",
    "repro": "Compare XI_CRITICAL against argmax of m(xi) exp(-psi(xi)/2) from the module's own solve_lane_emden.",
    "confidence": "high"
  },
  {
    "id": "S2-C-11",
    "file": "trinity/cloud_properties/bonnorEbertSphere.py",
    "line": 78,
    "class": "coefficient",
    "severity": "S3",
    "claim": "OMEGA_CRITICAL must be exp(psi(xi_crit)) = 14.042032.",
    "evidence": "psi(6.4507514) = 2.6420551 from my integration, so rho_c/rho_edge = e^2.6420551 = 14.042032. Matches the literature 14.04 (SPEC-065) and the docs' 'Omega ~ 14.04'.",
    "expected": "14.042 +/- 1e-3. The shipped default densBE_Omega = 14.1 is marginally supercritical (xi_out = 6.461675) - formally unstable, presumably deliberate, and worth an explicit note rather than a silent pass.",
    "failure_scenario": "A rounded 14.0 or a value taken from a different mu/gamma convention shifts the stability boundary; since the default sits 0.4% above the true boundary, a 1% error in the constant flips the default cloud's reported stability.",
    "repro": "assert abs(OMEGA_CRITICAL - exp(psi(XI_CRITICAL))) < 1e-3 using the module's own solver.",
    "confidence": "high"
  },
  {
    "id": "S2-C-12",
    "file": "trinity/cloud_properties/bonnorEbertSphere.py",
    "line": 87,
    "class": "coefficient",
    "severity": "S2",
    "claim": "The two dimensionless-mass constants are not independent: M_BONNOR_CRITICAL = M_DIM_CRITICAL / sqrt(4 pi * OMEGA_CRITICAL).",
    "evidence": "M = 4 pi rho_c r0^3 m(xi) with r0 = c_s/sqrt(4 pi G rho_c) gives M = m c_s^3/(sqrt(4 pi) G^(3/2) rho_c^(1/2)); substituting rho_c = Omega P_ext/c_s^2 gives M = [m/sqrt(4 pi Omega)] c_s^4/(G^(3/2) P_ext^(1/2)). Numerically 15.704374/sqrt(4 pi * 14.042032) = 15.704374/13.28414 = 1.1822266, which is exactly the value I get by direct maximisation of m_P.",
    "expected": "M_DIM_CRITICAL = 15.704374 (= xi_crit^2 psi'(xi_crit)) and M_BONNOR_CRITICAL = 1.1822266, satisfying the relation above to 1e-6. Related coefficients that get conflated: M G/(R c_s^2) = 2.4345031 and M G^(3/2) rho_c^(1/2)/c_s^3 = 4.4301221.",
    "failure_scenario": "All of 1.18, 2.43 and 4.43 are called 'the Bonnor-Ebert mass coefficient' depending on whether the sphere is normalised at fixed external pressure, fixed radius, or fixed central density. Picking the wrong one gives a critical mass wrong by a factor 2-4, which silently mislabels clouds as stable/unstable.",
    "repro": "assert abs(M_BONNOR_CRITICAL - M_DIM_CRITICAL/sqrt(4*pi*OMEGA_CRITICAL)) < 1e-6.",
    "confidence": "high"
  },
  {
    "id": "S2-C-13",
    "file": "trinity/cloud_properties/bonnorEbertSphere.py",
    "line": 347,
    "class": "numerical",
    "severity": "S2",
    "claim": "The BE enclosed mass must be taken from the ODE itself, m(xi) = xi^2 psi'(xi), not re-integrated numerically.",
    "evidence": "The Lane-Emden equation is d/dxi(xi^2 psi') = xi^2 exp(-psi), i.e. xi^2 psi' IS the integral of xi'^2 exp(-psi) from 0 to xi. Hence M(<r) = 4 pi rho_c r0^3 xi^2 psi' exactly, with zero quadrature error.",
    "expected": "compute_enclosed_mass_bonnor_ebert(r_arr, rho_arr) equals 4 pi rho_c r0^3 xi^2 psi'(xi(r)) to solver tolerance, and equivalently M_cloud * m(xi)/m(xi_out) which makes M(rCloud) = M_cloud exact.",
    "failure_scenario": "Trapezoidal re-integration of 4 pi r^2 rho on the stored grid introduces a grid-dependent mass error (and drops the 0 -> r[0] contribution, see S2-C-14), so M(rCloud) misses M_cloud by more than the 1e-3 tolerance on a coarse grid while the physics is unchanged - i.e. a spurious validator failure, or a real mass error hidden by a loose tolerance.",
    "repro": "Compare the function's output against xi^2*psi'(xi) scaled by 4 pi rho_c r0^3 on the same grid; also check m(xi_out) against my table: Omega=14.1 -> xi=6.461675, m=15.73673.",
    "confidence": "high"
  },
  {
    "id": "S2-C-14",
    "file": "trinity/cloud_properties/mass_profile.py",
    "line": 231,
    "class": "silent-failure",
    "severity": "S2",
    "claim": "Generic cumulative-trapezoid enclosed mass must include the contribution from r = 0 to r_arr[0].",
    "evidence": "M(r_arr[0]) = (4pi/3) rho(r_arr[0]) r_arr[0]^3 for a flat core, not 0. scipy's cumulative_trapezoid returns n-1 points or prepends a hard 0.",
    "expected": "M(r_arr[0]) equals the analytic core mass; M is length-n and matches compute_enclosed_mass_powerlaw to the grid's own accuracy (invariant I-12).",
    "failure_scenario": "On a log grid starting at 1e-3 r_core the omission is negligible; on a grid starting at r_core it silently discards the entire core mass - 0.1% at alpha = 0 but 3.6% at alpha = -2 with rCore_fraction = 0.1, i.e. above MASS_TOLERANCE. Additionally, trapezoid on 4 pi r^2 rho with a steep alpha = -2 profile on a coarse LINEAR grid is biased at the percent level.",
    "repro": "compute_enclosed_mass(r_arr, rho_arr) vs compute_enclosed_mass_powerlaw(r_arr) on a 200-point log grid from 0.01*rCore to rCloud, alpha = -2.",
    "confidence": "high"
  },
  {
    "id": "S2-C-15",
    "file": "trinity/cloud_properties/mass_profile.py",
    "line": 437,
    "class": "other",
    "severity": "S1",
    "claim": "The mass accretion rate must be exactly Mdot = 4 pi r^2 rho(r) rdot, using the LOCAL density (rho_ISM outside rCloud, rho_core inside rCore).",
    "evidence": "M depends on t only through r, so Mdot = (dM/dr) rdot and dM/dr = 4 pi r^2 rho(r) by definition of the enclosed mass. There is no other correct expression. This is the sweep-up term Mdot_sh of SPEC-020/021.",
    "expected": "compute_mass_accretion_rate(r, rdot) == 4*pi*r**2*get_mass_density(r)*rdot identically, and get_mass_profile(..., return_mdot=True, rdot=...) returns the same value (two paths, one answer).",
    "failure_scenario": "Using a mean rather than local density, or the cloud density beyond rCloud, mis-sizes the ram/sweep-up term. Because SPEC-020 warns the sweep-up term must appear exactly once in the EOM, an error here is indistinguishable from the ram-pressure double-counting bug it interacts with.",
    "repro": "Evaluate both paths at r inside the core, in the envelope, and outside rCloud with rdot = 1 pc/Myr; compare to 4 pi r^2 rho(r).",
    "confidence": "high"
  },
  {
    "id": "S2-C-16",
    "file": "trinity/cloud_properties/bonnorEbertSphere.py",
    "line": 298,
    "class": "units",
    "severity": "S2",
    "claim": "The BE sound speed is the ISOTHERMAL sound speed; gamma must not enter the hydrostatic structure or the back-solved temperature.",
    "evidence": "The Lane-Emden equation used here follows from P = rho c_s^2 with c_s^2 = k_B T/(mu m_H) - an isothermal equation of state. r0 = c_s/sqrt(4 pi G rho_c) inherits that definition, so inverting r0 to get c_s and then T must use T = mu m_H c_s^2/k_B with no gamma.",
    "expected": "gamma is used only for a separate diagnostic (if at all). The back-solved T for a real GMC should land at ~10-30 K (SPEC-066, spec test T15).",
    "failure_scenario": "Forming c_s^2 = gamma k_B T/(mu m_H) makes the implied T low by gamma = 5/3 (-40%), turning a 25 K cloud into a 15 K one - or, in reverse, silently rescaling r0 and hence rCloud by sqrt(5/3) = 1.29 and the mass by 2.15.",
    "repro": "For param/cloud_example_BE.param (M=1e6, n_core=1e4, Omega=14.1) back out c_s = r0*sqrt(4 pi G rho_c) and T = mu m_H c_s^2/k_B; check it is O(10 K) and that flipping gamma from 5/3 to 1.0 changes nothing in the returned profile.",
    "confidence": "medium"
  },
  {
    "id": "S2-C-17",
    "file": "trinity/cloud_properties/bonnorEbertSphere.py",
    "line": 453,
    "class": "units",
    "severity": "S2",
    "claim": "r_to_xi and xi_to_r must be exact inverses, with c_s in the same length/time units as G and rho_core.",
    "evidence": "xi = r sqrt(4 pi G rho_c)/c_s and r = xi c_s/sqrt(4 pi G rho_c). With G = 4.4985e-3 pc^3 Msun^-1 Myr^-2 and rho_c in Msun/pc^3, sqrt(4 pi G rho_c) has units Myr^-1, so c_s must be pc/Myr.",
    "expected": "xi_to_r(r_to_xi(r, c_s, rho_c), c_s, rho_c) == r to floating point, for r spanning 1e-3..1e3 pc; and r2xi/xi2r (the params wrappers) agree with the explicit-argument versions.",
    "failure_scenario": "c_s supplied in km/s where pc/Myr is expected is a 2.3% error in every radius (1 km/s = 1.022712 pc/Myr) - too small to look wrong, and it enters the mass as r0^3, i.e. 7%.",
    "repro": "Round-trip assert plus a dimensional check: for c_s = 0.2 km/s and rho_c = 0.0346101*1e4 Msun/pc^3, r0 should be ~0.0163 pc if c_s is converted and ~0.0167 pc if not.",
    "confidence": "high"
  },
  {
    "id": "S2-C-18",
    "file": "trinity/cloud_properties/bonnorEbertSphere.py",
    "line": 91,
    "class": "regime",
    "severity": "S3",
    "claim": "XI_MAX must exceed xi_out(Omega) for every Omega the schema admits, and an out-of-range Omega must fail loudly rather than clamp.",
    "evidence": "Computed Omega -> xi_out: 2 -> 2.2746, 10 -> 5.5969, 14.04 -> 6.4504, 14.1 -> 6.4617, 100 -> 14.2156, 1e3 -> 40.8176, 1e4 -> 139.4178. xi_out grows roughly as sqrt(Omega) asymptotically (rho/rho_c -> 2/xi^2).",
    "expected": "XI_MAX >= ~140 to cover Omega up to 1e4, or an explicit range check on Omega. XI_MIN must be > 0 (the ODE has 2/xi) and small enough that the series ICs are accurate, ~1e-6 or below.",
    "failure_scenario": "If the tabulated solution stops at, say, xi = 20, then any Omega > ~250 has no root in the table; an interpolator will either extrapolate (returning a physically wrong xi_out and hence a wrong rCloud and profile shape) or clamp to the table edge, both silently.",
    "repro": "create_BE_sphere with Omega = 500 and Omega = 5000 and check xi_out against 28.6-ish and ~98 respectively (interpolating my table), or that it raises.",
    "confidence": "high"
  },
  {
    "id": "S2-C-19",
    "file": "trinity/cloud_properties/density_profile.py",
    "line": 55,
    "class": "regime",
    "severity": "S2",
    "claim": "rho(r) must be non-negative and non-increasing everywhere, including across the rCloud jump; that requires nEdge >= nISM and alpha <= 0.",
    "evidence": "The profile is n_core inside rCore, n_core (r/rCore)^alpha in the envelope, n_ISM outside. Continuity at rCore is automatic ((rCore/rCore)^alpha = 1) so no matching coefficient is needed. The rCloud discontinuity is by design and is a DOWNWARD step only if nEdge >= nISM.",
    "expected": "np.all(np.diff(rho(r_grid)) <= 0) on a fine grid spanning 0 to 3 rCloud; rho continuous at rCore to machine precision; rho(rCloud+) == mu m_H nISM.",
    "failure_scenario": "alpha > 0 produces a density increasing outward (not a cloud); nEdge < nISM produces a density inversion at the edge, i.e. the 'cloud' is a cavity in the ISM and the shell decelerates on entering it. Both are mathematically well-defined and would run to completion.",
    "repro": "Monotonicity assert over the grid for alpha in {0,-1,-2} plus a deliberately bad case (nCore = 10, nISM = 100).",
    "confidence": "high"
  },
  {
    "id": "S2-C-20",
    "file": "trinity/cloud_properties/mass_profile.py",
    "line": 83,
    "class": "units",
    "severity": "S1",
    "claim": "get_mass_density(r)/get_density_profile(r) must be a constant, r-independent, equal to mu_convert * m_H in the appropriate unit pair.",
    "evidence": "rho = mu_H m_H n with mu_H = 1.4 (mass per hydrogen nucleus, ionisation-independent, SPEC-092.1). Computed: 0.0346101 Msun pc^-3 per cm^-3, or 2.34298e-24 g cm^-3 per cm^-3.",
    "expected": "The ratio is constant to machine precision across the core, the envelope and the ISM.",
    "failure_scenario": "An r-dependent ratio means different regions use different mu (e.g. mu_mol in the cloud, mu_ion in the ISM). That breaks dM/dr = 4 pi r^2 rho at the region boundaries and mis-scales the swept mass by up to 2.3x.",
    "repro": "ratio = get_mass_density(r_grid)/get_density_profile(r_grid); assert ptp(ratio)/mean(ratio) < 1e-12 and mean(ratio) == 0.0346101 (internal units).",
    "confidence": "high"
  },
  {
    "id": "S2-C-21",
    "file": "trinity/cloud_properties/mass_profile.py",
    "line": 62,
    "class": "other",
    "severity": "S3",
    "claim": "Scalar-in/scalar-out and shape transparency: f(array)[i] must equal f(array[i]) for every profile function.",
    "evidence": "The _is_scalar/_to_array/_to_output trio (duplicated in density_profile.py at L34-44 and mass_profile.py at L62-72) declares exactly this contract.",
    "expected": "Scalar input returns a Python/NumPy scalar, not a 0-d or 1-element array; array input returns the same shape; elementwise equality between the two paths. np.float64, 0-d arrays and Python ints are the classic _is_scalar blind spots.",
    "failure_scenario": "A np.float64 misclassified as non-scalar returns a 1-element array, which then broadcasts unexpectedly inside the ODE right-hand side and turns a scalar state into an array - typically surfacing far downstream as a shape error or, worse, a silent broadcast to the wrong shape.",
    "repro": "For x in [1.0, np.float64(1.0), np.array(1.0), 1]: check type and value of get_density_profile(x, params) vs get_density_profile(np.array([1.0]), params)[0]. Also note _is_scalar/_to_array/_to_output are duplicated verbatim in two modules - a candidate for pre-existing duplication rather than a defect.",
    "confidence": "medium"
  },
  {
    "id": "S2-C-22",
    "file": "trinity/cloud_properties/mass_profile.py",
    "line": 267,
    "class": "coefficient",
    "severity": "S1",
    "claim": "The envelope mass term must carry the normalisation rCore^(-alpha) and must be added to the core mass (4pi/3) rho_core rCore^3, not used alone.",
    "evidence": "rho(r) = rho_c (r/rCore)^alpha = rho_c rCore^(-alpha) r^alpha, so integral 4 pi r^2 rho dr from rCore to r = 4 pi rho_c rCore^(-alpha) (r^(3+alpha) - rCore^(3+alpha))/(3+alpha). Verified against quadrature to <=4e-16 for alpha = 0,-1,-2,-2.5,-3,-3.5.",
    "expected": "The core term contributes f^3/3 / K(f,alpha) of the total mass with rCore = f rCloud: 0.1% at alpha = 0, 3.6% at alpha = -2 for f = 0.1. Dropping it therefore violates MASS_TOLERANCE = 1e-3 at any alpha steeper than about -1.",
    "failure_scenario": "Assuming the core is negligible under-counts the mass by up to a few percent at steep alpha; conversely, integrating the pure power law from r = 0 (no core) over-counts and diverges for alpha <= -3. Either way rCloud is solved from the wrong equation and the validator's mass check either fails spuriously or passes with a compensating rCloud error.",
    "repro": "Compare M(rCloud) from the coded formula against scipy.integrate.quad of 4 pi r^2 rho over [0, rCloud] with points=[rCore], for alpha = -2 and rCore_fraction = 0.1.",
    "confidence": "high"
  },
  {
    "id": "S2-C-23",
    "file": "trinity/cloud_properties/validate_gmc.py",
    "line": 181,
    "class": "regime",
    "severity": "S2",
    "claim": "check_gmc_constraints must reject: rCloud > R_CLOUD_MAX (200 pc), nEdge < nISM, and |M_computed - mCloud|/mCloud > MASS_TOLERANCE.",
    "evidence": "SPEC-062 (rCloud_max rejects implausibly diffuse clouds; separate checks require n_edge >= n_ISM and mass consistency). The nEdge check is the only thing preventing a density inversion at the cloud edge (S2-C-19).",
    "expected": "All three checks present and each independently able to fail; the mass residual reported as a number, not just a boolean.",
    "failure_scenario": "A missing nEdge >= nISM check admits clouds whose edge is less dense than the ISM; a missing rCloud cap admits 1e3 pc 'clouds'. Both run to completion and produce plausible-looking trajectories.",
    "repro": "Feed a deliberately inverted case (mCloud=1e4, nCore=10, nISM=100) and a diffuse case (mCloud=1e9, nCore=1) and confirm each is rejected with the right reason.",
    "confidence": "high"
  },
  {
    "id": "S2-C-24",
    "file": "trinity/cloud_properties/validate_gmc.py",
    "line": 549,
    "class": "other",
    "severity": "S4",
    "claim": "Every alternative parameter combination returned by the suggestion helpers must itself pass the validator.",
    "evidence": "A suggestion that does not satisfy the constraints it was generated to fix is worse than no suggestion. This is a pure self-consistency property, checkable in one loop, independent of how the search is scored.",
    "expected": "For each suggestion s returned by _suggest_powerlaw_alternatives / _suggest_bonnor_ebert_alternatives, validate_gmc_params(**s) is valid.",
    "failure_scenario": "The search scores candidates on distance (the nested _distance functions at L630/L711) but validates them with a different or looser predicate, so the user is handed parameters that fail on the next run.",
    "repro": "Round-trip every suggestion through validate_gmc_params for a failing input.",
    "confidence": "medium"
  },
  {
    "id": "S2-C-25",
    "file": "trinity/cloud_properties/initial_profile.py",
    "line": 58,
    "class": "numerical",
    "severity": "S2",
    "claim": "The returned (r, density, mass) arrays must resolve rCore, start at r ~ 0 with M -> 0, end at rCloud with M = mCloud, and be monotone.",
    "evidence": "Invariants I-2, I-3, I-4. With the shipped default rCore = 0.01 pc and rCloud of order 10-100 pc, the core spans 1e-4 of the domain: a linear grid of even 1e4 points puts ~1 point inside the core.",
    "expected": "Log-spaced radii (or an explicitly refined core region); M[0] ~= (4pi/3) rho_core r[0]^3; M[-1] == mCloud to 1e-10; np.all(diff(M) > 0); np.all(diff(density) <= 0).",
    "failure_scenario": "An unresolved core makes any grid-based enclosed mass miss the core contribution and biases every downstream interpolation of M(r) near the centre - exactly where the shell starts.",
    "repro": "build_initial_cloud_profile with dens_profile='densPL', alpha=-2, rCore=0.01, rCloud=30 and check how many grid points satisfy r < rCore, then compare M[-1] to mCloud.",
    "confidence": "medium"
  },
  {
    "id": "S2-C-26",
    "file": "trinity/cloud_properties/initial_profile.py",
    "line": 58,
    "class": "state",
    "severity": "S3",
    "claim": "nEdge, if passed in, must be consistent with the other parameters rather than being an independent source of truth.",
    "evidence": "nEdge is fully determined: power law nEdge = nCore (rCloud/rCore)^alpha; Bonnor-Ebert nEdge = nCore/Omega. Accepting it as a separate argument creates two definitions of the same quantity.",
    "expected": "Either nEdge is derived internally, or it is checked against the derived value to ~1e-10 relative and a mismatch raises.",
    "failure_scenario": "A caller passing a stale or rounded nEdge produces a profile whose edge density disagrees with its own alpha/rCloud, so the density is discontinuous in a way the validator's nEdge >= nISM check cannot see.",
    "repro": "Call with a deliberately wrong nEdge (e.g. 2x the derived value) and check whether it is used, ignored, or rejected.",
    "confidence": "low"
  },
  {
    "id": "S2-C-27",
    "file": "trinity/cloud_properties/powerLawSphere.py",
    "line": 214,
    "class": "state",
    "severity": "S2",
    "claim": "compute_consistent_params must return a set (rCloud, rCore, nEdge) that simultaneously satisfies M(rCloud) = M_cloud, rCore = rCore_fraction * rCloud, and nEdge >= nISM.",
    "evidence": "All three are simultaneously satisfiable in closed form: R = [M/(4 pi rho_c K(f,alpha))]^(1/3) with K = f^3/3 + f^(-alpha)(1-f^(3+alpha))/(3+alpha), rCore = fR, nEdge = nCore f^(-alpha). The nEdge constraint then reads f^(-alpha) >= nISM/nCore, which for f = 0.1 and alpha = -2 gives nEdge/nCore = 0.01 - satisfied for nCore = 1e5, nISM = 1, but NOT for a diffuse cloud with nCore = 50.",
    "expected": "The returned triple passes validate_gmc_params unchanged; if the nEdge constraint cannot be met at the requested rCore_fraction, it must widen rCore (S2-C-07 gives the minimum) or fail, not silently return an invalid set.",
    "failure_scenario": "Silently returning a set with nEdge < nISM propagates the density inversion of S2-C-19 into every run built from these 'consistent' params.",
    "repro": "compute_consistent_params(1e5, 50.0, -2.0, nISM=1.0) - nEdge would be 0.5 cm^-3 < nISM; check the return.",
    "confidence": "medium"
  },
  {
    "id": "S2-C-28",
    "file": "trinity/cloud_properties/mass_profile.py",
    "line": 131,
    "class": "regime",
    "severity": "S3",
    "claim": "At alpha = 0 no output may depend on rCore, and the power-law path must reduce exactly to the homogeneous one.",
    "evidence": "For alpha = 0 the envelope term becomes 4 pi rho_c (r^3 - rCore^3)/3, which added to the core term (4pi/3) rho_c rCore^3 gives (4pi/3) rho_c r^3 - rCore cancels identically. SPEC-063 states rCore is ignored for alpha = 0. K(f,0) = 1/3 for any f, verified.",
    "expected": "Bitwise-identical profiles for rCore = 0.01 and rCore = 5 at alpha = 0; compute_rCloud_powerlaw(..., alpha=0) == compute_rCloud_homogeneous(...) to 1e-12.",
    "failure_scenario": "Residual rCore dependence at alpha = 0 means the DEFAULT configuration (densPL_alpha = 0) silently depends on an officially inert parameter - and SPEC-063 warns that any sweep varying alpha without setting rCore inherits rCore = 0.01 pc.",
    "repro": "Diff the full profile arrays for the two rCore values at alpha = 0.",
    "confidence": "high"
  },
  {
    "id": "S2-C-29",
    "file": "trinity/cloud_properties/bonnorEbertSphere.py",
    "line": 221,
    "class": "numerical",
    "severity": "S3",
    "claim": "solve_lane_emden's ODE function signature is odeint's (y, xi) ordering; it must not be handed to solve_ivp without swapping.",
    "evidence": "The declared signature is lane_emden_ode(y: np.ndarray, xi: float). scipy.integrate.odeint calls f(y, t); scipy.integrate.solve_ivp calls f(t, y). The two are not interchangeable.",
    "expected": "Whichever integrator is used, the argument order matches it; the resulting psi(6.4507514) = 2.6420551 and m(6.4507514) = 15.704374.",
    "failure_scenario": "Transposed arguments do not raise here - both arguments are numeric - so the integrator silently solves a different system and returns a smooth, plausible-looking but wrong profile. The only way to see it is to check psi/m against known values.",
    "repro": "Evaluate the module's solution at xi = 6.4507514 and compare psi and xi^2 psi' against 2.6420551 and 15.704374.",
    "confidence": "medium"
  },
  {
    "id": "S2-C-30",
    "file": "trinity/cloud_properties/bonnorEbertSphere.py",
    "line": 64,
    "class": "units",
    "severity": "S3",
    "claim": "The cgs constants block must hold G = 6.6743e-8, k_B = 1.380649e-16, m_H ~ 1.6735e-24 (atomic H, not the proton mass 1.67262e-24), Msun ~ 1.98892e33, pc = 3.0856775814913673e18, Myr = 3.15576e13.",
    "evidence": "SPEC-091's conversion table, derived from those three base conversions. The atomic-hydrogen vs proton mass distinction is 0.05%; the Msun convention spread (IAU nominal 1.98841e33 vs 1.98892e33) is 0.03%.",
    "expected": "Values as above, and consistent with whatever trinity/_functions/unit_conversions.py uses - two constant blocks for the same physical constants must agree.",
    "failure_scenario": "Independent constant blocks that drift apart make bit-identical equivalence tests (CLAUDE.md rule 5) fail for reasons unrelated to the change under test; a 0.2% year definition (3.15e7 s) is 1% in an inferred luminosity via the fifth-power scaling of SPEC-092.3.",
    "repro": "Diff the constants in this module against trinity/_functions/unit_conversions.py.",
    "confidence": "medium"
  }
]
```
