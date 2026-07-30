# S3 phase0 init — Lens C (what it should be)

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

**Scope.** `trinity/phase0_init/get_InitCloudProp.py` and `trinity/phase0_init/get_InitPhaseParam.py`,
derived from physics + the redacted signature list + `PHYSICS_SPEC.md` only. No implementation, no
comments, no docstrings read. Literature access blocked — every rational prefactor below is either
derived here from scratch (marked **derived**) or explicitly flagged as **recalled**.

**What I inferred the interface to be** (stated so the reconciler can discount it if wrong):
`get_y0(params)` returns the initial state that seeds the energy-phase ODE — minimally
`(R2₀, v2₀, E_b₀, T0₀)` at an initial time `t₀`. The constant names
`WEAVER_ENERGY_FRACTION`, `WEAVER_TEMP_COEFFICIENT`, `WEAVER_L_REF` say the seed is the
Weaver/Castor self-similar solution, with `L_REF` a cgs luminosity normalisation (`L₃₆`) for a
temperature law. `MIN_LUMINOSITY / MIN_MOMENTUM / MIN_VELOCITY` are floors guarding
`v_w = 2L_w/ṗ_w`, `Ṁ_w = ṗ_w²/(2L_w)` at `t → 0` where the SPS table may give `L_w = ṗ_w = 0`.
Everything below is written as "what must be true", not "what the code does".

---

## 1. The energy-driven similarity solution, derived from scratch

### 1.1 Setup

Constant mechanical luminosity `L_w` deposited at the origin in a static uniform medium of mass
density `ρ₀`; thin (radiative-outer-shock) shell of radius `R(t)`; hot interior of uniform pressure
`P`, ideal gas index `γ`. Three equations, no free parameters:

```
(1) mass       M(R) = (4π/3) ρ₀ R³
(2) momentum   d(M Ṙ)/dt = 4π R² P
(3) energy     dE/dt = L_w − P dV/dt ,   E = P V/(γ−1) ,  V = (4π/3)R³
```

For `γ = 5/3`: `E = (3/2) P (4π/3) R³ = 2π P R³`, i.e.

> **`P_b = E_b / (2π R³)`  — derived, the `2π` is `(γ−1)·(4π/3)` inverted, *not* a solid angle.**

With `R1 ≠ 0` this becomes `P_b = E_b / [2π (R2³ − R1³)]` (SPEC-024).

### 1.2 The exponent 3/5

Put `R = A t^η`. From (2):

```
d/dt[(4π/3)ρ₀R³Ṙ] = (4π/3)ρ₀ (3R²Ṙ² + R³R̈) = 4πR²P
⇒  P = (ρ₀/3)(3Ṙ² + R R̈) = (ρ₀/3) η(4η−1) A² t^{2η−2}
```

Then `E = 2πPR³ = (2π/3)η(4η−1) ρ₀A⁵ t^{5η−2}`. Constant `L_w` ⇒ `E ∝ t` ⇒ `5η − 2 = 1` ⇒
**`η = 3/5`** (derived; the general power-law case gives `η = 3/(5−w)`, §4).

### 1.3 Every rational coefficient (all derived, all numerically re-checked)

With `η = 3/5`, `η(4η−1) = (3/5)(7/5) = 21/25`:

| step | exact | decimal |
|---|---|---|
| `P = c_P ρ₀A²t^{−4/5}` | `c_P = (1/3)(21/25) = 7/25` | 0.28 |
| `E = 2πPR³ = c_E ρ₀A⁵t` | `c_E = 2π(7/25) = 14π/25` | 1.7592919 |
| `P dV/dt = 4πR²Ṙ P = c_W ρ₀A⁵` | `c_W = 4π(3/5)(7/25) = 84π/125` | 2.1111503 |
| `L_w = dE/dt + PdV/dt = c_L ρ₀A⁵` | `c_L = 14π/25 + 84π/125 = 154π/125` | 3.8704421 |

Hence

```
A⁵ = L_w /(c_L ρ₀) = 125 L_w /(154 π ρ₀) = 250 L_w /(308 π ρ₀) = 0.2583684 L_w/ρ₀
```

> **`R2(t) = ξ_E (L_w/ρ₀)^{1/5} t^{3/5}`, `ξ_E = (250/308π)^{1/5} = 0.76286534`** (derived)
> **`v2(t) = (3/5) R2/t = 0.45771920 (L_w/ρ₀)^{1/5} t^{−2/5}`** (derived)

⚠️ `PHYSICS_SPEC.md` SPEC-050 quotes `ξ_E = 0.762934` and `v` coefficient `0.457760`. Those are
arithmetic slips in the 5th significant figure: `0.762934⁵ = 0.2584859 ≠ 250/308π = 0.2583684`.
The correct value is **0.7628653**. Immaterial physically (0.01%), but a numeric-anchor test written
against 0.762934 is wrong at that level.

### 1.4 The energy fraction — the load-bearing number

```
E_b/(L_w t) = c_E/c_L = (14π/25)·(125/154π) = 14·125/(25·154) = 1750/3850 = 5/11
```

> **`E_b = (5/11) L_w t = 0.4545454… L_w t`** (derived, exact rational)

Companions (derived the same way):
`E_kin,shell = ½MṘ² = (6π/25)ρ₀A⁵t ⇒ E_kin/(L_w t) = 15/77 = 0.194805`;
radiated at the outer shock `= 1 − 5/11 − 15/77 = 27/77 = 0.350649`. Sum `= 1` ✓.

**Fully general form (derived).** With `M ∝ R^{3−w}` and arbitrary `γ`, writing `c_γ = 4π/(3(γ−1))`:

```
L_w = B A^{5−w} η [(4−w)η − 1] ( c_γ/4π + η ) ,   E_b = (c_γ/4π) B A^{5−w} η[(4−w)η−1] t
⇒   E_b/(L_w t) = 1 / (1 + 3(γ−1) η) ,   η = 3/(5−w)
⇒   E_b/(L_w t) = (5−w) / [ (5−w) + 9(γ−1) ]
```

Specialisations (all verified numerically):

| | `γ = 5/3` | `γ = 7/5` | `γ = 4/3` |
|---|---|---|---|
| `w = 0` | **5/11 = 0.45455** | 25/43 = 0.58140 | 5/8 = 0.625 |
| `w = 1` | 4/10 = 2/5 | | |
| `w = 2` | 3/9 = 1/3 | | |

So `E_b/(L_w t) = (5−w)/(11−w)` for `γ = 5/3`, and `5/(9γ−4)` for a uniform medium.
**5/11 is doubly special: it assumes `γ = 5/3` *and* a uniform ambient medium.**

### 1.5 Bubble pressure and R1 in the same limit

```
P_b = E_b/(2πR2³) = 5 L_w t /(22 π R2³) = (7/25)ξ_E² L_w^{2/5}ρ₀^{3/5}t^{−4/5}
    = 0.16294979 L_w^{2/5} ρ₀^{3/5} t^{−4/5}
```
(SPEC-052's `0.162979` inherits the same 5th-digit slip; correct value `0.1629498`.)

Combining with `R1 = sqrt(ṗ_w/(4πP_b))` (SPEC-025) and `ṗ_w = 2L_w/v_w` gives a **clean,
prefactor-free invariant** (derived, numerically confirmed):

```
R1/R2 = sqrt( 11 v2 /(3 v_w) )      ⇒   R1 < R2  ⟺  v2 < (3/11) v_w = 0.2727 v_w
```

### 1.6 Numeric anchors (computed here; μ convention stated, per SPEC-050's trap)

At `L₃₆ = 1`, `n_H = 1 cm⁻³`, `t = 1 Myr`:

| `ρ₀` convention | `R2` | `v2` | `P_b/k_B` |
|---|---|---|---|
| `ρ₀ = n_H m_H` (μ = 1, Weaver's own) | 28.047 pc | 16.454 km/s | 2.551×10⁴ K cm⁻³ |
| `ρ₀ = 1.4 n_H m_H` (TRINITY `mu_convert`) | 26.221 pc | 15.383 km/s | 3.1×10⁴ K cm⁻³ |

---

## 2. The free-streaming radius

### 2.1 The physical balance

Free expansion ends when the wind stops behaving like an unimpeded radial flow. Three inequivalent
statements, all derivable:

**(A) Swept ambient mass = ejected wind mass.** During free expansion `R = v_w t`, so
`M_sw = (4π/3)ρ₀R³` and `M_ej = Ṁ_w t = Ṁ_w R/v_w`. Equating:

```
(4π/3) ρ₀ R³ = Ṁ_w R / v_w
⇒  R_fs = sqrt( 3 Ṁ_w /(4π ρ₀ v_w) ) = sqrt( 3 ṗ_w /(4π ρ₀ v_w²) ) = sqrt( 3 L_w /(2π ρ₀ v_w³) )
```
(using `ṗ_w = Ṁ_w v_w`, `L_w = ½Ṁ_w v_w²`). **Derived.**

**(B) Wind density = ambient density**, `ρ_w(R) = Ṁ_w/(4πR²v_w) = ρ₀`:

```
R_fs' = sqrt( Ṁ_w /(4π ρ₀ v_w) ) = R_fs / √3
```
**Derived.** (A) and (B) differ by exactly `√3 = 1.732`.

**(C) Wind ram pressure = ambient *thermal* pressure**, `ṗ_w/(4πR²) = P_ISM ⇒ R = sqrt(ṗ_w/(4πP_ISM))`.
This is SPEC-025 with `P_b → P_ISM`. **It diverges for the TRINITY default `PISM = 0`**, so it
cannot be the criterion unless guarded.

### 2.2 Dimensions

`[Ṁ/(ρ v)] = (g s⁻¹)/((g cm⁻³)(cm s⁻¹)) = cm²` ⇒ `√ → cm` ✓. In TRINITY's internal
`[M⊙, pc, Myr]`: `(M⊙ Myr⁻¹)/((M⊙ pc⁻³)(pc Myr⁻¹)) = pc²` ✓. The formula is unit-system-agnostic;
only the `4π` and the `3` are conventions.

### 2.3 Where R_fs sits relative to the Weaver track (derived)

The free-expansion ray `R = v_w t` crosses the Weaver curve at
`R_x = ξ_E^{5/2}(2π/3)^{1/2} R_fs = 0.7356 R_fs` (verified numerically), so criteria (A) and (B)
bracket the true onset to within ~40%. Physically reassuring — but **the Weaver seed is not yet
valid there**:

```
at R0 = R_fs, t0 = R_fs/v_w :  v2 = (3/5)R0/t0 = 0.6 v_w  ⇒  R1/R2 = sqrt(11·0.6/3) = 1.483  > 1
at R0 = R_fs, t0 from inverting the Weaver law (t0 = 1.227 R_fs/v_w) : R1/R2 = 1.339 > 1
```

**Both violate `R1 < R2`.** The Weaver seed only becomes self-consistent once `v2 < (3/11)v_w`,
which on the similarity track means `R0 ≳ 2.4 R_fs` (equivalently `t0 ≳ 7 t_x`).

Worked fiducial (`default.param`: `mCloud 1e7`, `sfe 0.01` ⇒ `M_* = 10⁵ M⊙`, `nCore 1e5 cm⁻³`,
`L_w ≈ 10³⁹ erg/s`, `v_w ≈ 2000 km/s`, `μ_H = 1.4`):
`R_fs(A) = 5.17×10⁻³ pc`, `R_fs(B) = 2.99×10⁻³ pc`, `t_fs = 2.53×10⁻⁶ Myr`.
Note `R_fs ≪ rCore` here — see §4.3.

---

## 3. Expected dimensions and unit system

TRINITY works internally in **`[M⊙, pc, Myr]`** (SPEC-090/091). `get_y0` must return AU, not cgs,
not km/s.

| initialised quantity | AU unit | cgs equivalent |
|---|---|---|
| `t₀` (cluster age) | Myr | `3.15576×10¹³ s` |
| `R2₀` | pc | `3.0857×10¹⁸ cm` |
| `v2₀` | **pc/Myr** | `1 pc/Myr = 0.977781 km/s`; `1 km/s = 1.022712 pc/Myr` |
| `E_b₀` | M⊙ pc² Myr⁻² | `1.90148×10⁴³ erg` |
| `T0₀` | K | K |
| `L_w`, `L_mech` | M⊙ pc² Myr⁻³ | `6.0255×10²⁹ erg s⁻¹` |
| `ṗ_w` | M⊙ pc Myr⁻² | `6.1623×10²⁴ dyn` |
| `Ṁ_w` | M⊙ Myr⁻¹ | `6.3025×10¹⁹ g s⁻¹` |
| `v_w` | pc/Myr | cm/s |
| `P_b`, `P_ISM` | M⊙ pc⁻¹ Myr⁻² | `6.4721×10⁻¹³ dyn cm⁻²` |
| `ρ₀`, `ρ_core` | M⊙ pc⁻³ | `6.7696×10⁻²³ g cm⁻³` |
| `n_core`, `n_ISM` | cm⁻³ at the `.param` boundary | — |
| `rCloud`, `rCore` | pc | — |
| `mCloud`, `mCluster`, `M_sh` | M⊙ | — |

**Key structural point:** `ξ_E = 0.7628653`, `5/11`, `15/77`, `27/77`, `3/5`, `1/(2π)` are **pure
numbers** — valid in any self-consistent unit system, hence immune to the unit-conversion bug class.
By contrast `WEAVER_TEMP_COEFFICIENT ≈ 1.5×10⁶ K` and `WEAVER_L_REF = 10³⁶ erg s⁻¹` are **cgs-only**
and mark a unit boundary that must be crossed explicitly.

Also: `E_b₀ = (5/11) L_w t₀` contains **no density** — it is immune to the `μ_H = 1.4` trap. `R_fs`
and `R2₀` carry `ρ₀^{−1/2}` and `ρ₀^{−1/5}` respectively and are *not*.

---

## 4. Power-law cloud: what changes and what does not

### 4.1 Exponents (derived, §1.4 general form)

For `ρ(r) = ρ_ref (r/r_ref)^{−w}`, `w ≡ |α| ∈ [0,2]`:

```
η = d ln R2/d ln t = 3/(5−w)
A^{5−w} = L_w (3−w) / [ 4π ρ_ref r_ref^w · η · ((4−w)η − 1) · (½ + η) ]
E_b/(L_w t) = (5−w)/(11−w)      (γ = 5/3)
β = −d ln P_b/d ln t = 2 − (2−w)·3/(5−w)
δ =  d ln T /d ln t = −6/[7(5−w)]
```

Sanity checks (derived): `w=0 ⇒ η=3/5, E=5/11, β=4/5, δ=−6/35` — exactly the `cool_alpha 0.6`,
`cool_beta 0.8`, `cool_delta −6/35` defaults; `w=2 ⇒ η=1, E=1/3, β=2, δ=−2/7`;
`w=1 ⇒ η=3/4, E=2/5`. All reproduce `δ = (2/7)(2η − β − 1)` (SPEC-042) identically.

### 4.2 Which prefactors are uniform-only

| quantity | uniform-only? |
|---|---|
| `ξ_E = 0.7628653` | **YES** — replaced by `A^{5−w}` above |
| `E_b/(L_w t) = 5/11` | **YES** — `(5−w)/(11−w)` |
| `v2 = (3/5)R/t` | **YES** — `η R/t` |
| `T ∝ L^{8/35} n^{2/35} t^{−6/35}` | **YES** — `t` exponent `−6/(7(5−w))` |
| `R_fs = sqrt(3Ṁ/(4πρ₀v_w))` | **YES** — see §4.3 |
| `P_b = E_b/(2π(R2³−R1³))` | **NO** — pure thermodynamics, `γ` only |
| `R1 = sqrt(ṗ_w/(4πP_b))` | **NO** — pure ram balance |
| `v_w = 2L_w/ṗ_w`, `Ṁ_w = ṗ_w²/(2L_w)` | **NO** — definitional |
| `R1/R2 = sqrt(11 v2/(3 v_w))` | **YES** (uses `E=5/11 L t`); general: `sqrt(2 v2/[η(1+2η)^{-1}·…])` — re-derive per `w` |

### 4.3 The free-streaming radius on a power law — and a degeneracy

Without a flat core, criterion (A) becomes
`4πρ_ref r_ref^w R^{3−w}/(3−w) = Ṁ_w R/v_w`, i.e.

```
R_fs^{2−w} = (3−w) Ṁ_w / (4π ρ_ref r_ref^w v_w)
```

> **At `w = 2` (`densPL_alpha = −2`) the exponent `2−w` vanishes: both sides scale as `R¹` and the
> criterion has *no unique root* — it is either satisfied for every `R` or for none.** Criterion (B)
> degenerates identically. This is a hard edge case for `densPL_alpha = −2`, which the schema allows.

**What saves it in practice, and the condition that must be checked:** TRINITY's profile is flat
inside `rCore` (SPEC-060), and for dense clouds `R_fs ≪ rCore` (fiducial: `5×10⁻³ pc` vs `rCore`
default `0.01 pc`). *When `R_fs ≤ rCore` the uniform-medium formulae with `ρ = ρ_core` are exactly
right regardless of `α`* — including `5/11`. So the correct rule is **not** "use `(5−w)/(11−w)`
because `α ≠ 0`"; it is:

```
R_fs ≤ rCore  →  uniform formulae with ρ₀ = ρ_core          (correct, α-independent)
R_fs >  rCore →  power-law formulae with the local w         (uniform coefficients are wrong)
```

`R_fs ∝ ρ₀^{−1/2}`, so the second branch is reached for diffuse clouds: at `n_core = 10² cm⁻³`,
`R_fs ≈ 0.16 pc`; at `n_core = 1 cm⁻³`, `R_fs ≈ 1.6 pc` — above `rCore = 0.01 pc` (default) by 2
orders of magnitude. **The `paperII` grid sweeps low `nCore`, so this branch is exercised.**

---

## 5. Exact invariants the initialisation must satisfy

**I1 — On-manifold consistency.** `(R2₀, v2₀, E_b₀)` must be a point on the similarity solution the
ODE integrates. Two independent ways to state it, which must agree:

```
v2₀ = η R2₀ / t₀                                            with η = 3/(5−w_eff)
P_b(E_b₀) = E_b₀/[2π(R2₀³ − R1₀³)] = (ρ₀/3) η(4η−1) R2₀²/t₀²
```

For `w=0` the second reads `P_b₀ = (7/25) ρ₀ R2₀²/t₀²`. Cross-check (derived): substituting
`E_b₀ = (5/11)L_w t₀` and `L_w = (154π/125)ρ₀R2₀⁵/t₀³` into `E/(2πR³)` returns exactly
`(7/25)ρ₀R2₀²/t₀²` ✓ — so the triple is self-consistent **iff** `R2₀ = ξ_E(L_w/ρ₀)^{1/5}t₀^{3/5}`.
If `R2₀` is set independently (e.g. to `R_fs`) and `t₀` fixed separately, this identity breaks and
the ODE starts off-manifold.

**I2 — Strict positivity.** `E_b₀ > 0` (not `≥ 0`): `P_b = E_b/(2πV)` and `R1 = sqrt(ṗ/(4πP_b))`
both divide by it, and `T0` is derived from it. `E_b₀ = 0` is a division by zero, not a limit.
Likewise `R2₀ > 0`, `t₀ > 0`, `V_b₀ = (4π/3)(R2₀³ − R1₀³) > 0`.

**I3 — Ordering.** `0 < R1₀ < R2₀ < rCloud` and `R2₀ ≪ rCloud` (the run must not begin already
blown out). Equivalently, from §1.5, **`v2₀ < (3/11) v_w`**.

**I4 — Energy budget closure at `t₀`.** `E_b₀ + ½M_sh(R2₀)v2₀² ≤ L_w t₀` (the remainder is radiated).
With the on-manifold values this is `5/11 + 15/77 = 50/77 = 0.649 ≤ 1` ✓ with margin. Violation ⇒
the seed creates energy from nothing. Derived bound: for `v2₀ = 0.6R2₀/t₀`, the constraint is
`ρ₀R2₀⁵/(L_w t₀³) ≤ 0.72347`, i.e. `R2₀ ≤ 1.2287 ×` the Weaver radius.

**I5 — Shell mass.** `M_sh(R2₀) = ∫₀^{R2₀}4πr²ρ(r)dr > 0`, and by the free-streaming criterion
`M_sh(R2₀) ≥ Ṁ_w t₀` (that *is* criterion A at equality). `M_sh(R2₀) ≪ (1−ε)M_cl`.

**I6 — Temperature ceiling.** The bubble cannot be hotter than freshly shocked wind:

```
T0₀ ≤ T_max = 3 μ m_H v_w² /(16 k_B)        (strong shock, γ=5/3, shock speed v_w)
```
For `v_w = 2000 km/s`, `μ = 0.609`: `T_max = 5.53×10⁷ K`. **Derived.** This is the sharpest available
test of an initial `T0` and it *fails* for a Weaver-law `T0` evaluated at `t₀ ≈ t_fs` — see §6.

**I7 — Cloud closure.** `M(<rCloud) = M_cloud` to `< 10⁻¹⁰` relative (SPEC-061/062, T16), and both
the analytic `M(<r)` and the numerical integral over the radius array must satisfy it.

**I8 — Grid membership.** `rCore` and `rCloud` must be *exact* nodes of the radius array (so
`ρ` and `M(<r)` are evaluated on the correct side of each break), and the grid must extend below
`R2₀` (`~5×10⁻³ pc`) with resolution, since `ρ(R2₀)` and `M_sh(R2₀)` are read from it at the very
first step.

**I9 — SPS domain.** `t₀ ≥ t_table[0]` and `L_w(t₀) > 0`, `ṗ_w(t₀) > 0`. `t₀ ~ 10⁻⁶ Myr` sits at the
very first table row where `L_w` may be zero or interpolation-dominated.

**I10 — Definitional consistency of the wind triple.** `v_w`, `Ṁ_w`, `ṗ_w`, `L_w` must satisfy
`L_w = ½Ṁ_w v_w²` and `ṗ_w = Ṁ_w v_w` *exactly* after any flooring. Flooring `L_w` and `ṗ_w`
independently and then computing `v_w = 2L_w/ṗ_w` and `Ṁ_w = ṗ_w²/(2L_w)` breaks this only if a
floor actually bites — but then it breaks silently.

---

## 6. Coefficients table

| # | Quantity | Derived rational prefactor | Derivation sketch | Confidence |
|---|---|---|---|---|
| C1 | `R2 = ξ_E (L_w/ρ₀)^{1/5} t^{3/5}` | `ξ_E = (250/308π)^{1/5} = (125/154π)^{1/5} = 0.76286534` | momentum+energy pair, §1.3; `ξ_E⁵ = 0.25836841` re-verified | **high** (derived end-to-end) |
| C2 | `A⁵ = c · L_w/ρ₀` | `c = 250/(308π) = 125/(154π) = 0.25836841` | `c_L = 154π/125` | **high** |
| C3 | `v2 = η R2/t` | `η = 3/5`; `v2 = 0.45771920 (L/ρ)^{1/5}t^{−2/5}` | `5η−2 = 1` | **high** |
| C4 | **`E_b = f·L_w t`** | **`f = 5/11 = 0.45454545…`** | `c_E/c_L = (14π/25)/(154π/125)` | **high** |
| C5 | shell KE fraction | `15/77 = 0.19480519` | `(6π/25)/(154π/125)` | **high** |
| C6 | radiated fraction | `27/77 = 0.35064935` | `1 − 5/11 − 15/77` | **high** |
| C7 | `E_b/(L_w t)`, general `w`, `γ=5/3` | `(5−w)/(11−w)` | `1/(1+3(γ−1)η)`, `η=3/(5−w)` | **high** |
| C8 | `E_b/(L_w t)`, general `γ`, `w=0` | `5/(9γ−4)` | same identity | **high** |
| C9 | `P_b = E_b/(k·(R2³−R1³))` | `k = 2π` (from `γ=5/3`) | `E=PV/(γ−1)`, `V=(4π/3)R³` | **high** |
| C10 | `P_b` in `ρ,R,t` | `(7/25) ρ₀ R²/t² = 0.28 ρ₀R²/t²` | momentum eq. | **high** |
| C11 | `P_b` in `L,ρ,t` | `(7/25)ξ_E² = 0.16294979 · L^{2/5}ρ^{3/5}t^{−4/5}` | C1+C10 | **high** |
| C12 | `P_b` in `L,t,R` | `5 L_w t/(22π R2³)` | C4 + C9 | **high** |
| C13 | `R1 = sqrt(ṗ_w/(k'·P_b))` | `k' = 4π` (no `3/4` strong-shock factor in the Weaver convention) | `ρ_w v_w² = ṗ_w/(4πr²)`; with the `3/4` post-shock factor, `k' = 16π/3` and `R1` is `0.866×` smaller | **high** balance, **medium** convention |
| C14 | `R1/R2` in the Weaver limit | `sqrt(11 v2/(3 v_w))`; `R1<R2 ⟺ v2 < (3/11)v_w` | C4+C12+C13, `ṗ_w = 2L_w/v_w` | **high** |
| C15 | `R_fs` (swept = ejected) | `sqrt(3Ṁ_w/(4πρ₀v_w))` `= sqrt(3ṗ_w/(4πρ₀v_w²))` `= sqrt(3L_w/(2πρ₀v_w³))` | §2.1(A) | **high** formula; **medium** that this is the intended criterion |
| C16 | `R_fs` (`ρ_w = ρ₀`) | `sqrt(Ṁ_w/(4πρ₀v_w))` = C15/√3 | §2.1(B) | **high** formula |
| C17 | `t_fs` | `R_fs/v_w` | free expansion | **high** |
| C18 | `R_fs`, power law, no core | `[(3−w)Ṁ_w/(4πρ_ref r_ref^w v_w)]^{1/(2−w)}`; **singular at `w=2`** | §4.3 | **high** |
| C19 | `v_w`, `Ṁ_w` from SPS | `v_w = 2L_w/ṗ_w`, `Ṁ_w = ṗ_w²/(2L_w)` | `L=½Ṁv²`, `ṗ=Ṁv` | **high** |
| C20 | Weaver `T` exponents | `T ∝ L^{8/35} n₀^{2/35} t^{−6/35}` (uniform); general `t^{−6/[7(5−w)]}` | `T^{7/2} ∝ P R²/(Ct)` + C1/C11 | **high** |
| C21 | conduction closure constant `a` in `T_b^{7/2} = a P_b R2²/(C t)` | **`a ≈ 5.42–5.87`** (`= (25·(41/35)/4)·B`, `B = B(3,3/5)=0.80128` or `B(17/5,3/5)=0.73993`) | integrated `T^{3/2}dT = −(5Ṁk_B/8πμm_H C)dr/r²` with `T(R2)=0`, closed with `Ṁ = dM_b/dt`, `M_b ∝ t^{41/35}` | **medium** (my closure; Weaver solves the full velocity field) |
| C22 | `WEAVER_TEMP_COEFFICIENT` | derived `≈1.78–1.82×10⁶ K` at `L₃₆=n₀=t₆=1`, `μ_H=1`; literature (**recalled, unverified**) `1.51×10⁶ K`, alternatively `2.07×10⁶ K` | C21 → `T_b = (a P R²/(Ct))^{2/7}` | **low** for the number, **high** for the ~1.5–2×10⁶ K range |
| C23 | `WEAVER_L_REF` | `10³⁶ erg s⁻¹` (the `L₃₆` normalisation) — **cgs, not AU** (`= 1.6596×10⁶ M⊙pc²Myr⁻³`) | convention that accompanies C22 | **medium** |
| C24 | isobaricity constraint on any hard-coded `(n_b, T_b)` pair | `n_b T_b = P_b/k_B = 2.55×10⁴ K cm⁻³` at `L₃₆=n₀=t₆=1`, `μ_H=1` | C12 numerically | **high** |
| C25 | `ξ = 0.98` reporting factor | `(1−0.98)^{2/5} = 0.2091279` | SPEC-040 profile | **high** |
| C26 | post-shock wind temperature ceiling | `3μ m_H v_w²/(16 k_B)`; `5.53×10⁷ K` at `v_w = 2000 km/s`, `μ=0.609` | Rankine–Hugoniot, strong shock, `γ=5/3` | **high** |
| C27 | `rCloud` (uniform, `α=0`) | `(3 M_cloud/(4π ρ_core))^{1/3}` | `M = (4π/3)ρR³` | **high** |
| C28 | `rCloud` (power law) | `[ r_c^{α}(3+α)(M/(4πρ_c) − r_c³/3) + r_c^{3+α} ]^{1/(3+α)}` | invert SPEC-061; reduces to C27 at `α=0` ✓ | **high** |
| C29 | free-expansion ray × Weaver curve | `R_x = ξ_E^{5/2}(2π/3)^{1/2} R_fs = 0.7356 R_fs` | §2.3 | **high** |
| C30 | self-consistency threshold `R1 = R2` | `R0 = 2.41 R_fs`, `t0 = 7.18 t_x` | C14 on the Weaver track | **high** |

### Traps, itemised

1. **`γ = 5/3`-only coefficients.** `5/11` (C4), `1/(2π)` (C9), `15/77`, `27/77`, and the strong-shock
   `3/16` in C26 all carry `γ`. General forms in C8. `default.param` declares `gamma_adia = 5/3`, so
   these are self-consistent — but a code that *reads* `gamma_adia` in the ODE while *hard-coding*
   `5/11` in the seed is inconsistent the moment anyone changes it.
2. **Injected vs retained energy.** `L_w t₀` is the **total wind energy injected**. Only `5/11` stays
   in the bubble; `15/77` is shell kinetic energy and `27/77` is radiated at the outer shock. Setting
   `E_b₀ = L_w t₀` over-fills the bubble by `2.2×` ⇒ `P_b` by `2.2×` ⇒ instant unphysical acceleration.
   Conversely, `5/11` is **not** a thermalisation efficiency: `FB_thermCoeffWind = FB_thermCoeffSN = 1`
   already carry that. Multiplying both is double-counting.
3. **`4π` vs `2π`.** Three different `2π`/`4π` appear and none is interchangeable:
   `ρ_w = Ṁ_w/(4πr²v_w)` (full solid angle), `R1² = ṗ_w/(4πP_b)` (full solid angle),
   and `P_b = E_b/(2πR³)` — where the `2π` is `(γ−1)·(4π/3)` inverted, **not** a hemisphere.
   Reading the last as a solid-angle `2π` and "fixing" it to `4π` halves `P_b`.
4. **Weaver vs Castor/McCray.** Both give `R ∝ (L t³/ρ)^{1/5}`; the **thin-shell / radiative-outer-shock**
   coefficient is `0.7629` (C1). The **fully adiabatic** continuous-injection blast wave (no radiative
   outer shock, so nothing lost) has a *larger* coefficient — commonly quoted as ≈0.88 (**recalled,
   not derived; medium-low confidence**), and its bubble energy fraction is correspondingly larger.
   Using 0.88 with `5/11`, or 0.76 with `E = L t`, mixes two solutions.
5. **Equation numbers.** I could not open Weaver+77 or the Rahner thesis. Any docstring citing
   "Weaver+77 Eq. 20/21/27/37" is **unverifiable from this lens**; the thesis renumbers independently.
   The *formulae* above are verifiable, the *numbers* are not — I refuse to assert them.
6. **`n₀` in the `T` law is weakly weighted, `ρ₀` in `R` is not.** `n₀^{2/35}` means a `1.4×` μ error
   is a 1.9% error in `T`. The same error in `R_fs` (`ρ^{−1/2}`) is 16%, and in `R2` (`ρ^{−1/5}`) is
   6.5%. Correct: `ρ = μ_H m_H n_H` with `μ_H = mu_convert = 1.4` (mass per H nucleus), **never**
   `μ_ion ≈ 0.609` (mass per particle) — SPEC-092 rank 1.
7. **`T0` reporting convention.** SPEC-040/T10: TRINITY reports `T0` at `ξ = 0.98`, which is
   `0.2091 ×` the central `T_b`. An initial `T0` seeded with the *central* Weaver value is `4.78×`
   larger than what the structure solver will return one step later — a step-1 discontinuity in a
   state variable.
8. **Extrapolating the Weaver `T` law to `t → 0`.** `T ∝ t^{−6/35}` diverges. Evaluated at the
   fiducial `t_fs = 2.53×10⁻⁶ Myr` it gives **`1.29×10⁸ K` (prefactor 1.51e6) — `2.3×` above the
   `5.53×10⁷ K` shocked-wind ceiling C26** (3.2× with the 2.07e6 prefactor, 2.8× with my derived one).
   The seed only drops below the ceiling at `t₀ ≈ 3.5×10⁻⁴ Myr`, i.e. `R2₀ ≈ 0.1 pc ≈ 19 R_fs`.
9. **Constant-`L_w` assumption.** `E_b = (5/11) L_w t` assumes `L_w` constant since `t = 0`. The
   defensible generalisation is `E_b₀ = (5/11)∫₀^{t₀} L_w dt'`. At `t₀ ~ 10⁻⁶ Myr` SB99 `L_w` is
   effectively flat, so this is a small effect — but it is the right form.
10. **Silent floors.** `MIN_LUMINOSITY / MIN_MOMENTUM / MIN_VELOCITY` clamping rather than raising
    turns "SPS table has no wind yet" into a finite, physically meaningless `R_fs`, `v_w`, `T0`,
    with no trace in the output. A floor that bites must at minimum log at WARNING and must be a
    pure divide-by-zero guard (many orders below any physical value), not a physical floor.
11. **Isobaricity of any hard-coded `(n_b, T_b)` pair.** C24: the widely-quoted literature pair
    `(4.02×10⁻³ cm⁻³, 1.51×10⁶ K)` gives `n T = 6.07×10³ K cm⁻³`, but the dynamical `P_b/k_B` is
    `2.55×10⁴` — **a factor 4.2 inconsistency** (independently confirmed here, matching SPEC-045).
    At most one of the two quoted prefactors can be right. Prefer the structural forms.
12. **`densPL_alpha = −2` degenerates the free-streaming criterion** (C18) — the root does not exist.
    Only the flat core (`r < rCore`) makes the problem well-posed.

---

## 7. Cloud initialisation (`get_InitCloudProp.py`)

Expected content of `CloudProperties`: `rCloud [pc]`, `rCore [pc]`, `nCore [cm⁻³]`, `nISM [cm⁻³]`,
`rho_core [M⊙ pc⁻³]`, `alpha`, `mCloud [M⊙]`, `mCluster [M⊙]`, and the tabulated
`r_arr [pc]` / `n_arr` or `rho_arr` / `m_arr [M⊙]` used by the shell-mass lookup.

**Power-law branch** — derived (SPEC-060/061/062):

```
ρ(r) = ρ_core                     r ≤ rCore
     = ρ_core (r/rCore)^α         rCore < r ≤ rCloud
     = ρ_ISM                      r > rCloud

M(<r) = (4π/3)ρ_core r³                                                        r ≤ rCore
      = 4π ρ_core [ rCore³/3 + (r^{3+α} − rCore^{3+α})/((3+α) rCore^{α}) ]     rCore < r ≤ rCloud

rCloud = [ rCore^{α}(3+α)( M_cloud/(4πρ_core) − rCore³/3 ) + rCore^{3+α} ]^{1/(3+α)}
       → (3 M_cloud/(4π ρ_core))^{1/3}        at α = 0   ✓
```
Singular only at `α = −3`, outside the allowed `[−2, 0]`.

**Which mass** — SPEC-005 is genuinely ambiguous. `verify_mass_at_rCloud(props, mCloud)` taking
`mCloud` as an *argument* implies the caller decides; the answer must be documented, because at
`ε = 0.3` (`param/simple_cluster.param`) the two readings differ by `0.7^{1/3} = 0.888` in `rCloud`
and 30% in swept mass.

**Validation** (`_validate_params`) must enforce, at minimum:
`−2 ≤ densPL_alpha ≤ 0`; `0 < sfe < 1`; `nCore > nISM`; `rCore > 0`; `rCore < rCloud`;
`rCloud ≤ rCloud_max (200 pc)`; and `n(rCloud) = nCore(rCloud/rCore)^α ≥ nISM` (a real constraint
only for `α < 0`; for `α = 0` it is implied by `nCore > nISM`).

**Radius array** (`_create_radius_array(rCloud, rCore, n_inside=1000, n_outside=100)`):
must contain `rCore` and `rCloud` as exact nodes (`verify_key_radii_in_array`), and must resolve
`r ≲ R2₀ ≈ 5×10⁻³ pc`. A *linear* 1000-point grid on `[0, rCloud]` with `rCloud ~ 20 pc` has
`Δr = 0.02 pc` — the default `rCore = 0.01 pc` core is then **entirely inside the first cell** and
`ρ(R2₀)`, `M_sh(R2₀)` at the first ODE step are pure extrapolation. Logarithmic (or piecewise
core-refined) spacing is required for the shipped defaults.

**Bonnor–Ebert branch** — SPEC-064/065/066. `ψ(ξ_out) = ln Ω`; `M = 4πρ_c r₀³ ξ²ψ'`;
`r₀ = c_s/sqrt(4πGρ_c)` ⇒ the sound speed (hence temperature) is **back-solved, not declared**.
Computed here for `param/cloud_example_BE.param` (`M = 10⁶ M⊙`, `n_core = 10⁴ cm⁻³`, `Ω = 14.1`):
`r₀ ≈ 2.4–4.6 pc`, `rCloud ≈ 16–30 pc`, **`c_s ≈ 10–20 km/s` ⇒ `T ≈ 3×10⁴–1×10⁵ K`** for
`μ_mol = 14/6`. (The spread is my uncertainty in `m(ξ_crit)`; the conclusion is robust because
`c_s² ~ GM/R ≈ (12 km/s)²` on dimensional grounds alone.) SPEC-066's test T15 expects 10–30 K. The
implied thermal support is 3+ orders of magnitude too hot: the BE profile is a **fitting function
standing in for turbulent support**, not a hydrostatic cloud, and `Ω = 14.1 > 14.04` is formally
unstable on top of that. Both facts should be stated where the profile is built.

`MockParam` at L558–559 of a production module is a test double in the importable namespace
(presumably behind `__main__`); it belongs in `test/`.

---

## 8. Honest confidence ledger

- **Derived from scratch, verified numerically, high confidence:** `ξ_E = 0.7628653`; `5/11`;
  `15/77`; `27/77`; `η = 3/5`; `(5−w)/(11−w)`; `5/(9γ−4)`; `P = E/(2πR³)`; `P = (7/25)ρR²/t²`;
  `P = 5Lt/(22πR³)`; `R1/R2 = sqrt(11v2/3v_w)`; `R_fs = sqrt(3Ṁ/(4πρv_w))` and its `√3` sibling;
  `R_fs^{2−w}` power-law form and its `w=2` degeneracy; `rCloud` closed forms; `T ∝ L^{8/35}n^{2/35}t^{−6/35}`
  and its general-`w` `t` exponent; `T_max = 3μm_Hv_w²/(16k_B)`; `(1−0.98)^{2/5} = 0.20913`.
- **Derived with a stated closure approximation, medium confidence:** `a ≈ 5.4–5.9` in
  `T_b^{7/2} = a P R²/(Ct)`, hence `T_b ≈ 1.8×10⁶ K` at the reference point.
- **Recalled, not derivable here, low confidence:** the literature `T_b` prefactor
  (`1.51×10⁶` vs `2.07×10⁶`); the `n_b` prefactor `4.02×10⁻³`; the adiabatic-branch coefficient
  `≈0.88`; **all Weaver+77 / Rahner-thesis equation numbers** — I assert none.
- **Interface assumption, not verified:** that `get_y0` returns `(R2, v2, E_b, T0)` at a `t₀`, and
  that `WEAVER_L_REF` is a `10³⁶ erg s⁻¹` normalisation for a `T` law rather than something else.
  Every expectation below is written to survive being wrong about the exact return shape.

---

```json
[
  {
    "id": "S3-C-01",
    "file": "trinity/phase0_init/get_InitPhaseParam.py",
    "line": 28,
    "class": "coefficient",
    "severity": "S2",
    "claim": "WEAVER_ENERGY_FRACTION must be exactly 5/11 = 0.454545..., the fraction of injected wind energy retained as bubble thermal energy in the uniform-medium, gamma=5/3, thin-shell energy-driven similarity solution.",
    "evidence": "Thin shell M=(4pi/3)rho0 R^3; momentum d(M Rdot)/dt = 4pi R^2 P gives P=(rho0/3)eta(4eta-1)A^2 t^(2eta-2); E=P V/(gamma-1)=2pi P R^3 for gamma=5/3; constant L requires E prop t so eta=3/5. Then E=(14pi/25)rho0 A^5 t, P dV/dt=(84pi/125)rho0 A^5, L=E/t+PdV/dt=(154pi/125)rho0 A^5, so E/(L t)=(14pi/25)/(154pi/125)=1750/3850=5/11 exactly. Verified numerically: 0.45454545454545453.",
    "expected": "0.45454545454545453 (or the literal 5/11). NOT 0.45, not 5/11 rounded to 2 dp, and not 1.0.",
    "failure_scenario": "If it is 1.0 (i.e. E_b0 = L_w t0), the bubble is over-filled by 2.2x, P_b is 2.2x too high, and the shell is impulsively accelerated at t0; the similarity attractor damps this but the early transition-trigger diagnostics (L_loss/L_gain) are corrupted. If it is 0.5 or 0.45, the seed is off-manifold by ~10% and the T3 validation test (Eb/(Lmech t) -> 0.4545) never converges cleanly.",
    "repro": "python run.py param/simple_cluster.param ; read the first record of dictionary.jsonl and check Eb / (Lmech_W * t_now) against 0.454545",
    "confidence": "high"
  },
  {
    "id": "S3-C-02",
    "file": "trinity/phase0_init/get_InitPhaseParam.py",
    "line": 28,
    "class": "regime",
    "severity": "S3",
    "claim": "The 5/11 energy fraction is valid only for gamma=5/3 AND a uniform ambient medium. On a power-law profile the correct fraction is (5-w)/(11-w) with w=|densPL_alpha|; for general gamma it is 5/(9*gamma-4) in the uniform case.",
    "evidence": "General derivation with M = B R^(3-w): L = B A^(5-w) eta [(4-w)eta-1] (c_gamma/4pi + eta) and E = (c_gamma/4pi) B A^(5-w) eta[(4-w)eta-1] t, with c_gamma = 4pi/(3(gamma-1)) and eta = 3/(5-w). Ratio E/(L t) = 1/(1 + 3(gamma-1)eta). Checks: w=0,gamma=5/3 -> 5/11; w=1 -> 2/5; w=2 -> 1/3; gamma=4/3,w=0 -> 5/8. All verified numerically.",
    "expected": "Either a documented restriction that the uniform-medium seed is used because the initial radius lies inside rCore (where the profile IS uniform), or an alpha-dependent fraction. The former is the physically correct choice whenever R2_0 <= rCore.",
    "failure_scenario": "For a diffuse cloud (nCore ~ 1-100 cm^-3, as swept in paperII_grid_sweep) the free-streaming radius grows as rho^-1/2 and exceeds rCore=0.01 pc by 1-2 orders of magnitude. Then a hard-coded 5/11 over-fills the bubble by up to 5/11 / 1/3 = 1.36x at alpha=-2, seeding an off-manifold state on exactly the configurations the published grid explores.",
    "repro": "Compare the first-snapshot Eb/(Lmech*t) for densPL_alpha = 0 and -2 at low nCore; both equalling 0.4545 indicates the uniform coefficient is applied unconditionally.",
    "confidence": "high"
  },
  {
    "id": "S3-C-03",
    "file": "trinity/phase0_init/get_InitPhaseParam.py",
    "line": 44,
    "class": "state",
    "severity": "S2",
    "claim": "The returned (R2_0, v2_0, E_b0) must lie on the similarity manifold the ODE integrates: v2_0 = (3/5) R2_0/t_0 AND E_b0/(2 pi (R2_0^3 - R1_0^3)) = (7/25) rho_0 R2_0^2 / t_0^2 must hold simultaneously.",
    "evidence": "Substituting E_b0=(5/11)L_w t0 and L_w=(154pi/125)rho0 R2_0^5/t0^3 into P=E/(2 pi R^3) returns exactly (5*154/(11*125*2)) rho0 R^2/t^2 = 0.28 rho0 R^2/t^2 = (7/25) rho0 R^2/t^2, which is precisely what the momentum equation demands. The identity holds if and only if R2_0 = 0.7628653 (L_w/rho_0)^(1/5) t_0^(3/5).",
    "expected": "Exactly one of {R2_0, t_0} is primary and the other follows from the Weaver radius law; v2_0 and E_b0 then follow. If R2_0 is set from a free-streaming criterion AND t_0 is set independently (e.g. a fixed tStart), the identity must be asserted or the deviation documented.",
    "failure_scenario": "An off-manifold seed produces a transient that decays only as t^(-2/5); it contaminates the measured alpha=v2*t/R2 (T4), the (alpha,beta,delta) triple fed to the bubble-structure solver, and hence L_cool and the transition time. Silent: nothing crashes.",
    "repro": "From snapshot 0 of dictionary.jsonl compute v2*t_now/R2 and Eb/(2*pi*(R2^3-R1^3)) / (0.28*rho_core*R2^2/t_now^2); both should be 0.6 and 1.0.",
    "confidence": "high"
  },
  {
    "id": "S3-C-04",
    "file": "trinity/phase0_init/get_InitPhaseParam.py",
    "line": 44,
    "class": "state",
    "severity": "S2",
    "claim": "The initial state must satisfy R1_0 < R2_0, equivalently v2_0 < (3/11) v_w = 0.2727 v_w. A Weaver seed taken at the free-streaming radius violates this.",
    "evidence": "R1 = sqrt(pdot_w/(4 pi P_b)) with P_b = (5/11)L_w t/(2 pi R2^3) and pdot_w = 2 L_w/v_w gives R1^2/R2^2 = (11/5) R2/(v_w t) = (11/3) v2/v_w exactly. At R2_0 = R_fs with t_0 = R_fs/v_w, v2_0 = 0.6 v_w so R1/R2 = sqrt(11*0.6/3) = 1.483 > 1. With t_0 obtained by inverting the Weaver law at R_fs (t_0 = 1.227 R_fs/v_w), R1/R2 = 1.339, still > 1. Self-consistency requires R2_0 >= 2.41 R_fs. All verified numerically for the default.param fiducial (Mcl*=1e5 Msun, nCore=1e5, L_w=1e39 erg/s, v_w=2000 km/s).",
    "expected": "Either t_0 (or R2_0) is chosen late enough that R1_0 < R2_0, or R1 is not evaluated from the Weaver P_b at t_0. An explicit assertion R1_0 < R2_0 at the end of get_y0.",
    "failure_scenario": "V_b = (4pi/3)(R2^3 - R1^3) goes NEGATIVE at step 0, making P_b = E_b/(2 pi V_b) negative; downstream sqrt(pdot/(4 pi P_b)) is NaN, or the bubble-structure integrator is handed an inverted domain. May manifest as a first-step solver failure with an unrelated-looking message rather than a clean error here.",
    "repro": "Check R1 < R2 in the first record of dictionary.jsonl for param/simple_cluster.param and for the low-density edge config docs/dev/performance/f1edge_lowdens*.param",
    "confidence": "high"
  },
  {
    "id": "S3-C-05",
    "file": "trinity/phase0_init/get_InitPhaseParam.py",
    "line": 44,
    "class": "coefficient",
    "severity": "S3",
    "claim": "If a Weaver radius law is used anywhere in the seed, its coefficient must be (250/(308*pi))**0.2 = 0.76286534, not 0.762934 and not 0.88.",
    "evidence": "A^5 = 125 L_w/(154 pi rho_0) = 0.25836841 L_w/rho_0 (derived in S3-C-01's chain); the fifth root is 0.7628653, verified: 0.7628653^5 = 0.25836841. PHYSICS_SPEC.md SPEC-050 quotes 0.762934, whose fifth power is 0.2584859 - a 5th-digit arithmetic slip in the spec, not in the physics. The value 0.88 belongs to the fully adiabatic (non-radiative outer shock) continuous-injection solution, a different problem.",
    "expected": "0.7628653 (or computed inline as (250/(308*math.pi))**0.2). The velocity companion is 0.6*xi = 0.45771920 and the pressure companion is 0.28*xi^2 = 0.16294979.",
    "failure_scenario": "Using 0.88 (the adiabatic branch) with the 5/11 thin-shell energy fraction mixes two mutually inconsistent solutions: R is 15% too large, so P_b = E/(2 pi R^3) is 35% too low. A test asserting 0.762934 is itself wrong at 1e-4, which is above bit-identical tolerance for a free-win regression gate.",
    "repro": "pytest -k weaver ; or compare the constant against (250/(308*math.pi))**0.2",
    "confidence": "high"
  },
  {
    "id": "S3-C-06",
    "file": "trinity/phase0_init/get_InitPhaseParam.py",
    "line": 44,
    "class": "coefficient",
    "severity": "S2",
    "claim": "The free-streaming radius must be R_fs = sqrt(3*Mdot_w/(4*pi*rho_0*v_w)) if the criterion is 'swept ambient mass equals ejected wind mass', or R_fs = sqrt(Mdot_w/(4*pi*rho_0*v_w)) if it is 'wind density equals ambient density'. The two differ by exactly sqrt(3). The solid angle is 4*pi in both.",
    "evidence": "(A) During free expansion R = v_w t, so (4pi/3)rho_0 R^3 = Mdot_w R/v_w gives R^2 = 3 Mdot_w/(4 pi rho_0 v_w). Equivalent forms: sqrt(3 pdot_w/(4 pi rho_0 v_w^2)) and sqrt(3 L_w/(2 pi rho_0 v_w^3)). (B) rho_w(R)=Mdot_w/(4 pi R^2 v_w)=rho_0 gives R^2 = Mdot_w/(4 pi rho_0 v_w). Dimensions: (g/s)/((g/cm^3)(cm/s)) = cm^2, sqrt -> cm; in AU (Msun/Myr)/((Msun/pc^3)(pc/Myr)) = pc^2.",
    "expected": "One of the two, applied consistently, with the criterion named. A 2*pi in place of 4*pi (a hemisphere argument) would be a factor sqrt(2)=1.414 error in R_fs.",
    "failure_scenario": "A sqrt(3)=1.73 or sqrt(2)=1.41 error in R_fs propagates as t_0 ~ R_fs/v_w into E_b0 = (5/11)L_w t_0 linearly, and into the initial T0 as t_0^(-6/35). Since the similarity solution is an attractor the trajectory recovers, but the earliest snapshots - which are what a validation figure anchors on - are wrong.",
    "repro": "For param/simple_cluster.param check R2 in snapshot 0 against sqrt(3*Mdot_w/(4*pi*rho_core*v_w)) computed from the same snapshot's Lmech_W and pdot_total",
    "confidence": "medium"
  },
  {
    "id": "S3-C-07",
    "file": "trinity/phase0_init/get_InitPhaseParam.py",
    "line": 44,
    "class": "units",
    "severity": "S2",
    "claim": "rho_0 entering the free-streaming radius and the Weaver radius law must be a MASS density built with mu_convert = 1.4 (mass per hydrogen nucleus), not with mu_ion ~ 0.609 (mass per particle), and must be in the same unit system as L_w and v_w.",
    "evidence": "R_fs scales as rho^(-1/2) and R2 as rho^(-1/5). Confusing mu_H=1.4 with mu_ion=14/23=0.609 is a factor 2.3 in rho, i.e. 1.52x in R_fs and 1.19x in R2. SPEC-092 ranks this the single most frequent conversion bug in the codebase. Note E_b0 = (5/11)L_w t_0 carries NO density, so it is immune - the error enters only through t_0.",
    "expected": "rho_0 = mu_convert * m_H * nCore, converted to Msun/pc^3 (1 Msun/pc^3 = 6.7696e-23 g/cm^3), with L_w in Msun pc^2 Myr^-3 and v_w in pc/Myr. Since xi_E and 5/11 are pure numbers the whole expression can be evaluated in AU without any cgs excursion.",
    "failure_scenario": "A 1.4x or 2.3x density error shifts the seed radius by 6-19% and the seed time correspondingly; the SPEC-050 numeric anchor test would 'pass' at 28 pc while carrying a compensating mu bug.",
    "repro": "Assert rho_core in the CloudProperties equals 1.4 * m_H * nCore in the declared unit, and that get_y0 consumes that same rho",
    "confidence": "high"
  },
  {
    "id": "S3-C-08",
    "file": "trinity/phase0_init/get_InitPhaseParam.py",
    "line": 44,
    "class": "coefficient",
    "severity": "S2",
    "claim": "The wind triple must be derived once and consistently: v_w = 2*L_w/pdot_w, Mdot_w = pdot_w^2/(2*L_w), and the identities L_w = 0.5*Mdot_w*v_w^2 and pdot_w = Mdot_w*v_w must hold exactly after any flooring.",
    "evidence": "Definitional (SPEC-071). Unit sanity in AU: (Msun pc^2 Myr^-3)/(Msun pc Myr^-2) = pc/Myr. If MIN_LUMINOSITY and MIN_MOMENTUM are applied independently to L_w and pdot_w and then both formulas evaluated, the resulting (v_w, Mdot_w) no longer satisfy L = 0.5 Mdot v^2.",
    "expected": "Floor once, derive the rest; or floor v_w only (MIN_VELOCITY) and back out Mdot_w = pdot_w/v_w so pdot is exactly preserved.",
    "failure_scenario": "At t_0 ~ 1e-6 Myr the SPS table's first row may have L_w = pdot_w = 0. Independent floors then give a v_w set by the ratio of two arbitrary sentinels; R_fs, t_0, E_b0 and T0 are all fabricated from it with no NaN and no warning.",
    "repro": "Feed an SPS table whose first row has Lmech_W = 0 and confirm the run either raises or logs, rather than producing a finite R2 in snapshot 0",
    "confidence": "high"
  },
  {
    "id": "S3-C-09",
    "file": "trinity/phase0_init/get_InitPhaseParam.py",
    "line": 38,
    "class": "silent-failure",
    "severity": "S2",
    "claim": "MIN_LUMINOSITY, MIN_MOMENTUM and MIN_VELOCITY must be pure divide-by-zero guards (many orders of magnitude below any physical value) and must log at WARNING or raise when they bite; they must not be physically-sized floors that silently substitute a plausible-looking value.",
    "evidence": "The only quantities they can guard are v_w = 2L_w/pdot_w and Mdot_w = pdot_w^2/(2L_w), both singular at L_w = 0 or pdot_w = 0, which is exactly the state of an SPS table at t = 0. A physically-sized floor converts 'no wind yet' into 'a wind with these made-up properties'.",
    "expected": "Guard values at least ~10 orders below the smallest physical L_w/pdot_w for the smallest supported cluster mass, plus a logger.warning when triggered; or an explicit exception with the parameter values in the message.",
    "failure_scenario": "paperII_grid_sweep reaches M_cluster = 100 Msun (SPEC-073), where SPS quantities are already tiny; a physically-sized floor would bite on a whole corner of the published grid and produce identical, fabricated initial conditions across many cells with no diagnostic.",
    "repro": "Run the smallest grid cell (mCloud 1e4, sfe 0.01) and grep the log for any floor warning; compare snapshot-0 R2 across several such cells for suspicious identity",
    "confidence": "medium"
  },
  {
    "id": "S3-C-10",
    "file": "trinity/phase0_init/get_InitPhaseParam.py",
    "line": 32,
    "class": "exponent",
    "severity": "S2",
    "claim": "If WEAVER_TEMP_COEFFICIENT seeds T0 via the Weaver interior law, the exponents must be exactly 8/35 on L, 2/35 on n_0 and -6/35 on t (uniform medium). The t exponent generalises to -6/[7(5-w)] on a power law.",
    "evidence": "Conduction closure T^(7/2) = a P R^2/(C t) (SPEC-042, re-derived here from the integrated heat-flux equation). With P prop L^(2/5)rho^(3/5)t^(-4/5) and R^2 prop L^(2/5)rho^(-2/5)t^(6/5): P R^2/t prop L^(4/5) rho^(1/5) t^(-3/5), so T prop L^(8/35) rho^(2/35) t^(-6/35). General w: P R^2/t prop t^((4-w)eta-3) = t^(-3/(5-w)), so T prop t^(-6/(7(5-w))). Consistent with cool_delta = -6/35 and with delta = (2/7)(2 alpha - beta - 1).",
    "expected": "8/35 = 0.2285714, 2/35 = 0.0571429, -6/35 = -0.1714286 exactly, written as fractions not truncated decimals.",
    "failure_scenario": "A truncated -0.171 instead of -6/35 is a 0.17% exponent error, negligible at t~1 Myr but amplified at t_0 ~ 1e-6 Myr where t^(-6/35) is evaluated 6 decades from unity: (1e-6)^0.00043 differs by 0.6%. More seriously, a sign error on the t exponent makes T0 -> 0 at early times instead of diverging, seeding a cold bubble.",
    "repro": "Compare T0 in snapshot 0 across two runs differing only in nCore by 10x; the ratio should be 10^(2/35) = 1.142",
    "confidence": "high"
  },
  {
    "id": "S3-C-11",
    "file": "trinity/phase0_init/get_InitPhaseParam.py",
    "line": 32,
    "class": "numerical",
    "severity": "S3",
    "claim": "WEAVER_TEMP_COEFFICIENT should be in the range ~1.5e6 to ~2.1e6 K (evaluated at L36 = n_0 = t6 = 1 with rho_0 = n_H m_H). My own derivation of the conduction closure gives 1.78-1.82e6 K.",
    "evidence": "Integrating the conduction-front energy equation Mdot (5/2) k_B T/(mu m_H) = 4 pi r^2 C T^(5/2) dT/dr with T(R2)=0 gives T^(5/2) prop (1-x)/x, hence T prop (1-x)^(2/5) (the SPEC-040 exponent, derived). Closing with Mdot = dM_b/dt, M_b = 4 pi R2^3 mu m_H P B/(k_B T_b) and M_b prop t^(41/35) gives T_b^(7/2) = a P R2^2/(C t) with a = (25*(41/35)/4)*B, B = Beta(3,3/5)=0.80128 or Beta(17/5,3/5)=0.73993, so a = 5.87 or 5.42. With P_b = 3.52e-12 dyn/cm^2 and R2 = 28.05 pc at the reference point, T_b = 1.82e6 or 1.78e6 K. The mu dependence cancels between the two relations, as SPEC-042 predicts. Literature values 1.51e6 and 2.07e6 are recalled, not verified.",
    "expected": "A value in [1.4e6, 2.2e6] K, with the source stated. Anything outside that range, or any value not accompanied by a mu / density convention, is a finding.",
    "failure_scenario": "T0 seeds the bubble-structure root-find; a factor-2 error in the seed can push the solver into a different basin or cost extra iterations, and it directly offsets the reported T0 in the first snapshots.",
    "repro": "Compute T0 in snapshot 0 and divide out (L36)^(8/35)(n0)^(2/35)(t6)^(-6/35) to recover the implied prefactor",
    "confidence": "low"
  },
  {
    "id": "S3-C-12",
    "file": "trinity/phase0_init/get_InitPhaseParam.py",
    "line": 35,
    "class": "units",
    "severity": "S2",
    "claim": "WEAVER_L_REF marks a cgs boundary: the Weaver temperature law is written in L36 = L_w/1e36 erg/s, n_0 in cm^-3 and t in Myr. L_w must therefore be converted from the internal AU unit (Msun pc^2 Myr^-3, = 6.0255e29 erg/s) before the ratio is formed, and n_0 must be a NUMBER density in cm^-3, not the AU mass density.",
    "evidence": "SPEC-090/091: internal units are [Msun, pc, Myr]. 1e36 erg/s = 1.6596e6 Msun pc^2 Myr^-3. Dividing an AU luminosity by 1e36 under-states L36 by 6.03e29, and the 8/35 power turns that into a factor (6.03e29)^(-0.2286) = 1.2e-7 on T0. Contrast xi_E and 5/11, which are pure numbers and need no conversion at all.",
    "expected": "Either WEAVER_L_REF = 1e36 with an explicit AU->cgs conversion of L_w, or WEAVER_L_REF = 1.6596e6 in AU with no conversion. Mixing them is the failure.",
    "failure_scenario": "A missed conversion makes T0 wrong by 7 orders of magnitude - loud enough to notice in a fresh run, but if a compensating factor was tuned into WEAVER_TEMP_COEFFICIENT the pair would look self-consistent while both are wrong, and the T0 scaling with L_w would then be right while the absolute value is arbitrary.",
    "repro": "Assert T0 in snapshot 0 lies between 1e6 and 1e8 K for param/simple_cluster.param",
    "confidence": "high"
  },
  {
    "id": "S3-C-13",
    "file": "trinity/phase0_init/get_InitPhaseParam.py",
    "line": 44,
    "class": "state",
    "severity": "S2",
    "claim": "The initial T0 must not exceed the immediate post-shock temperature of the wind, T_max = 3 mu m_H v_w^2/(16 k_B). A Weaver T law evaluated at the free-streaming time violates this by a factor ~2-3.",
    "evidence": "Rankine-Hugoniot for a strong shock with gamma=5/3 and shock speed v_w gives T_max = (3/16) mu m_H v_w^2/k_B = 5.53e7 K for v_w=2000 km/s, mu=0.609. For the default.param fiducial (M_cluster=1e5 Msun, nCore=1e5 cm^-3, L_w=1e39 erg/s), t_fs = R_fs/v_w = 2.53e-6 Myr and the Weaver law gives T0 = 1.29e8 K with prefactor 1.51e6 (2.3x the ceiling), 1.77e8 K with 2.07e6 (3.2x), 1.55e8 K with my derived 1.82e6 (2.8x). T prop t^(-6/35) only drops below the ceiling at t_0 ~ 3.5e-4 Myr, i.e. R2_0 ~ 0.1 pc ~ 19 R_fs. All verified numerically.",
    "expected": "Either t_0 is late enough that T0 <= T_max, or T0 is clamped at T_max with a log, or T0 is not taken from the Weaver law at t_0 at all. An explicit assertion is cheap and would catch this.",
    "failure_scenario": "A bubble seeded above the shocked-wind temperature is thermodynamically impossible; the cooling table is queried above its top edge (Gnat-Ferland stops at 1e8 K, SPEC-081), producing an extrapolated or clamped Lambda and hence a wrong L_cool in the very first evaluation of the transition trigger.",
    "repro": "python run.py param/simple_cluster.param ; compare T0 in snapshot 0 against (3/16)*mu_ion*m_H*v_w^2/k_B with v_w = 2*Lmech_W/pdot_total from the same snapshot",
    "confidence": "high"
  },
  {
    "id": "S3-C-14",
    "file": "trinity/phase0_init/get_InitPhaseParam.py",
    "line": 44,
    "class": "state",
    "severity": "S3",
    "claim": "The initial T0 must be reported on the same convention as the T0 the bubble-structure solver returns later - i.e. at xi = bubble_xi_Tb = 0.98, which is (1-0.98)^(2/5) = 0.20913 times the central T_b - not the central value.",
    "evidence": "SPEC-040: T(r) = T_b (1 - r/R2)^(2/5), derived here from the conduction-front integration. TRINITY reports T0 at xi=0.98 per bubble_xi_Tb. (1-0.98)^0.4 = 0.2091279. The Weaver prefactor formulae quote the CENTRAL value.",
    "expected": "If T0 is a xi=0.98 quantity, the seed must carry the 0.20913 factor; if it is the central value, the convention must differ consistently everywhere and the reported T0 must be documented as central.",
    "failure_scenario": "A 4.78x discontinuity in the state variable T0 between step 0 and step 1, invisible in a log-scale plot but poisoning any finite-difference estimate of delta = d ln T/d ln t at the start of the run - which is exactly the quantity fed back into the bubble-structure similarity solve.",
    "repro": "Plot T0 vs t for the first ~20 snapshots of param/simple_cluster.param and look for a single ~5x step between snapshot 0 and 1",
    "confidence": "medium"
  },
  {
    "id": "S3-C-15",
    "file": "trinity/phase0_init/get_InitPhaseParam.py",
    "line": 44,
    "class": "units",
    "severity": "S2",
    "claim": "The returned state must be in internal AU units: R2 in pc, v2 in pc/Myr (NOT km/s), E_b in Msun pc^2 Myr^-2, T0 in K, t_0 in Myr.",
    "evidence": "SPEC-090/091. 1 pc/Myr = 0.977781 km/s, so a km/s vs pc/Myr confusion is only 2.3% - small enough to be invisible in a plot and large enough to break an equivalence gate. 1 Msun pc^2 Myr^-2 = 1.90148e43 erg.",
    "expected": "pc, pc/Myr, Msun pc^2 Myr^-2, K, Myr - matching the ODE state vector the solver integrates.",
    "failure_scenario": "A km/s seed velocity is 2.3% low; the shell starts marginally off-manifold and the measured alpha = v2 t/R2 relaxes to 0.6 from below rather than sitting on it, which would be misread as a physics result rather than a unit bug.",
    "repro": "Check v2 in snapshot 0 against 0.6*R2/t_now (dimensionless, so unit-safe); a 0.977 or 1.023 ratio identifies the confusion",
    "confidence": "high"
  },
  {
    "id": "S3-C-16",
    "file": "trinity/phase0_init/get_InitPhaseParam.py",
    "line": 44,
    "class": "state",
    "severity": "S3",
    "claim": "The seed must satisfy energy conservation at t_0: E_b0 + 0.5*M_sh(R2_0)*v2_0^2 <= L_w * t_0.",
    "evidence": "On the similarity solution the two terms are 5/11 and 15/77 of L_w t, summing to 50/77 = 0.649 < 1; the remaining 27/77 was radiated at the outer shock. Derived bound for v2_0 = 0.6 R2_0/t_0: rho_0 R2_0^5/(L_w t_0^3) <= (6/11)/0.75398 = 0.72347, i.e. R2_0 <= 1.2287 x the Weaver radius at t_0.",
    "expected": "The inequality holds with the on-manifold margin (sum = 0.649). A sum > 1 means the initialisation created energy.",
    "failure_scenario": "If R2_0 comes from a free-streaming criterion but t_0 from an unrelated source, R2_0 can exceed 1.23x the Weaver radius and the seed injects more energy than the cluster has produced - a conservation violation present from t=0 that no later force-budget check (SPEC-007, T2) would attribute to the initialisation.",
    "repro": "From snapshot 0: (Eb + 0.5*M_sh*v2^2)/(Lmech_W*t_now) should be ~0.649 and must be < 1",
    "confidence": "high"
  },
  {
    "id": "S3-C-17",
    "file": "trinity/phase0_init/get_InitCloudProp.py",
    "line": 149,
    "class": "coefficient",
    "severity": "S2",
    "claim": "rCloud must be the root of M(<rCloud) = M_cloud with the piecewise power-law enclosed mass; the closed forms are rCloud = (3 M_cloud/(4 pi rho_core))^(1/3) for alpha=0 and rCloud = [rCore^alpha (3+alpha)(M_cloud/(4 pi rho_core) - rCore^3/3) + rCore^(3+alpha)]^(1/(3+alpha)) otherwise.",
    "evidence": "Direct integration of 4 pi r^2 rho(r) with rho = rho_core for r<=rCore and rho_core (r/rCore)^alpha beyond: M(<r) = 4 pi rho_core [rCore^3/3 + (r^(3+alpha) - rCore^(3+alpha))/((3+alpha) rCore^alpha)]. Inverting gives the expression above; setting alpha=0 collapses it to the sphere formula, verified symbolically. Singular only at alpha=-3, outside the allowed [-2,0].",
    "expected": "Analytic inversion, or a bracketed root-find whose residual is verified. verify_mass_at_rCloud must return |M(<rCloud)/mCloud - 1| < 1e-10 (SPEC-061/062, test T16).",
    "failure_scenario": "A root-find without a residual check can return a bracket endpoint for extreme alpha/rCore combinations; the cloud then has the wrong total mass, and every swept-mass and gravity term is offset by that amount for the whole run. Silent.",
    "repro": "pytest ; and call verify_mass_at_rCloud on param/cloud_example_PL.param for alpha in {0, -1, -2}",
    "confidence": "high"
  },
  {
    "id": "S3-C-18",
    "file": "trinity/phase0_init/get_InitCloudProp.py",
    "line": 412,
    "class": "numerical",
    "severity": "S2",
    "claim": "_create_radius_array must resolve radii down to the initial bubble radius (~5e-3 pc for the shipped defaults) and must place rCore and rCloud as exact nodes. A linear grid on [0, rCloud] with n_inside=1000 does not.",
    "evidence": "Fiducial R2_0 ~ R_fs = 5.17e-3 pc (default.param: M_cluster=1e5 Msun, nCore=1e5 cm^-3, v_w=2000 km/s). A linear grid with rCloud ~ 20 pc and 1000 points has dr = 0.02 pc, so both R2_0 AND the default rCore = 0.01 pc fall inside the FIRST cell. rho(R2_0) and M_sh(R2_0) at the first ODE step are then extrapolation, and the flat core is entirely unresolved. R_fs scales as rho^(-1/2), so for nCore = 1 cm^-3 R_fs reaches ~1.6 pc - still only 80 cells in.",
    "expected": "Logarithmic (or piecewise core-refined) spacing with an inner bound well below min(rCore, R_fs), plus rCore and rCloud inserted as exact nodes (which verify_key_radii_in_array at L521 presumably checks).",
    "failure_scenario": "The very first swept mass and ambient density the ODE sees are interpolation artefacts; for alpha != 0 the r^alpha cusp inside the first cells is integrated by a trapezoid rule at effectively zero resolution, so the tabulated M(<r) disagrees with the analytic one near the centre while agreeing at rCloud (where verify_mass_at_rCloud checks it).",
    "repro": "Compare the tabulated M(<r) against the analytic SPEC-061 expression at r = 0.5*rCore, rCore, 2*rCore for densPL_alpha = -2, rCore = 0.01",
    "confidence": "high"
  },
  {
    "id": "S3-C-19",
    "file": "trinity/phase0_init/get_InitCloudProp.py",
    "line": 380,
    "class": "state",
    "severity": "S2",
    "claim": "_validate_params must enforce: -2 <= densPL_alpha <= 0; 0 < sfe < 1; rCore > 0; rCore < rCloud; rCloud <= rCloud_max (200 pc); nCore > nISM; and n(rCloud) = nCore (rCloud/rCore)^alpha >= nISM.",
    "evidence": "SPEC-003 (alpha range, sfe range), SPEC-062 (rCloud_max and the edge-density check). The edge-density condition binds only for alpha < 0: at alpha=0 it reduces to nCore >= nISM. For alpha=-2 with the default rCore=0.01 pc, n(1 pc) = nCore*1e-4, so a 1e5 cm^-3 core reaches nISM=1 cm^-3 at ~3 pc - the profile is unphysical beyond that and rCloud must not exceed it.",
    "expected": "All seven checks, each raising with the offending value in the message. In particular the alpha=-2 edge-density check, which is the one that actually constrains the shipped defaults.",
    "failure_scenario": "A cloud whose outer regions are less dense than the ISM it sits in; the swept-mass integral then double-counts (cloud gas plus the ISM formula beyond rCloud), and the dissolution criterion (shell_nMax < nISM, SPEC-102) can fire immediately.",
    "repro": "Construct a param with densPL_alpha -2, rCore 0.01, nCore 1e5, nISM 1, mCloud 1e7 and confirm it is rejected",
    "confidence": "high"
  },
  {
    "id": "S3-C-20",
    "file": "trinity/phase0_init/get_InitCloudProp.py",
    "line": 89,
    "class": "state",
    "severity": "S2",
    "claim": "Which cloud mass normalises the density profile - the total mCloud or the post-SFE (1-sfe)*mCloud - must be explicit and consistent between get_InitCloudProp, the rCloud solve, and the swept-mass lookup.",
    "evidence": "SPEC-005 records this as unresolved. At sfe = 0.3 (param/simple_cluster.param) the two readings differ by 0.7^(1/3) = 0.888 in rCloud (11%) and 30% in swept mass at fixed radius. paper_densityProfile.py's _DEFAULTS uses mCloud=1e5*(1-0.01) with the comment 'post-SFE', which is evidence for the post-SFE reading but is a figure script, not the solver. verify_mass_at_rCloud(props, mCloud) taking mCloud as an argument means the caller chooses - so two callers can choose differently.",
    "expected": "One documented convention, used by the profile normalisation, the rCloud root, M_sh(r), and F_grav's M_sh alike. If total mCloud normalises the profile but (1-sfe)*mCloud is the sweepable gas, the difference must be handled explicitly, not implicitly.",
    "failure_scenario": "An 11% rCloud error is a 30% swept-mass error at fixed radius, which propagates directly into F_grav (SPEC-031) and the shell EOM. It is well above any integrator tolerance and would be invisible in a dimensionless test.",
    "repro": "Run param/simple_cluster.param (sfe 0.3) and check rCloud against (3*mCloud/(4*pi*rho_core))^(1/3) and against (3*0.7*mCloud/(4*pi*rho_core))^(1/3)",
    "confidence": "medium"
  },
  {
    "id": "S3-C-21",
    "file": "trinity/phase0_init/get_InitCloudProp.py",
    "line": 303,
    "class": "regime",
    "severity": "S2",
    "claim": "The Bonnor-Ebert branch's implied sound speed and temperature are outputs, not inputs, and for the shipped example they land at c_s ~ 10-20 km/s, i.e. T ~ 3e4-1e5 K for mu_mol = 14/6 - three orders of magnitude above a real GMC. The module must compute and surface this.",
    "evidence": "r_0 = c_s/sqrt(4 pi G rho_c) and M = 4 pi rho_c r_0^3 xi^2 dpsi/dxi, so specifying (mCloud, nCore, Omega) back-solves r_0 and hence c_s. For param/cloud_example_BE.param (1e6 Msun, 1e4 cm^-3, Omega 14.1): with m(xi_crit)=15.7, r_0 = 2.45 pc, rCloud = 15.8 pc, c_s = 10.6 km/s, T = 3.2e4 K; with m=2.4, r_0 = 4.58 pc, c_s = 19.8 km/s, T = 1.1e5 K. The conclusion is robust independent of m because c_s^2 ~ GM/R = 4.3e-3*1e6/16 = 269 (km/s)^2 on dimensional grounds. SPEC-066's test T15 expects 10-30 K.",
    "expected": "The implied c_s (or T) computed and logged, with a note that the BE profile is standing in for turbulent support rather than thermal hydrostatic equilibrium. Silently building a 'hydrostatic' profile whose implied temperature is 1e5 K is a validity-regime violation.",
    "failure_scenario": "The BE branch is presented as an equilibrium configuration when it is a fitting function; any downstream physics that assumes the cloud is thermally supported (or that reads a cloud temperature) is inconsistent with it. Compounded by densBE_Omega = 14.1 > 14.04, which is formally gravitationally unstable (SPEC-065).",
    "repro": "python run.py param/cloud_example_BE.param --dry-run ; back out c_s from the reported r_0 and rCloud",
    "confidence": "medium"
  },
  {
    "id": "S3-C-22",
    "file": "trinity/phase0_init/get_InitPhaseParam.py",
    "line": 44,
    "class": "regime",
    "severity": "S3",
    "claim": "For densPL_alpha = -2 the free-streaming criterion has no unique root; the initialisation is well-posed only because R_fs falls inside the flat core, which must be checked rather than assumed.",
    "evidence": "Without a core, criterion A reads 4 pi rho_ref r_ref^w R^(3-w)/(3-w) = Mdot_w R/v_w, i.e. R^(2-w) = (3-w)Mdot_w/(4 pi rho_ref r_ref^w v_w). At w=2 the exponent vanishes: both sides scale as R^1, so the equation is either an identity or has no solution. Criterion B (rho_w = rho_amb) degenerates identically. With the flat core the problem is well-posed provided R_fs <= rCore.",
    "expected": "An explicit check R_fs <= rCore (or R2_0 <= rCore) before using the uniform-medium formulae; otherwise solve the criterion against the actual profile.",
    "failure_scenario": "R_fs scales as rho^(-1/2), so at nCore = 1-100 cm^-3 (swept by paperII_grid_sweep) R_fs = 0.16-1.6 pc, far outside the default rCore = 0.01 pc. The uniform-medium R_fs, energy fraction and temperature law are then all applied in a regime where none of them holds, and for alpha = -2 the underlying criterion is not even well-defined.",
    "repro": "Run a low-nCore, densPL_alpha=-2 config and check whether R2 in snapshot 0 is greater than rCore",
    "confidence": "high"
  },
  {
    "id": "S3-C-23",
    "file": "trinity/phase0_init/get_InitPhaseParam.py",
    "line": 28,
    "class": "numerical",
    "severity": "S4",
    "claim": "E_b0 = f * L_w(t_0) * t_0 assumes L_w has been constant since t=0. The defensible generalisation is E_b0 = f * integral_0^t0 L_w dt'.",
    "evidence": "The similarity solution is derived for constant L_w; E prop t follows from that. SB99 L_mech is roughly flat for t < 3 Myr, and t_0 ~ 1e-6 Myr, so the two agree to well under a percent in practice - but only the integral form is correct in principle, and it matters if the SPS table's first rows rise steeply.",
    "expected": "Either the integral form, or a comment recording that the instantaneous value is used because t_0 is far inside the flat part of the SB99 wind curve.",
    "failure_scenario": "Negligible for the bundled table; a user-supplied SPS table with a steep leading edge would seed a systematically wrong E_b0 with no diagnostic.",
    "repro": "",
    "confidence": "medium"
  },
  {
    "id": "S3-C-24",
    "file": "trinity/phase0_init/get_InitPhaseParam.py",
    "line": 19,
    "class": "citation",
    "severity": "S4",
    "claim": "Any docstring or comment citing a Weaver+77 (or Rahner thesis) equation NUMBER for these coefficients is unverifiable and should cite the relation rather than the number.",
    "evidence": "PHYSICS_SPEC.md section 0.3 records that arXiv/ADS/OUP/imprs-hd were all 403 from this container; SPEC-045 explicitly refuses to assert Weaver equation numbers for the same reason. The Rahner thesis renumbers independently of the 1977 paper, so a number correct in one is wrong in the other. Every formula in this report is derived and citable by content: R = (250/308 pi)^(1/5)(L t^3/rho)^(1/5), E_b = (5/11) L t, T prop (1-r/R2)^(2/5).",
    "expected": "Citations of the form 'Weaver et al. 1977, self-similar energy-driven solution' or the formula itself, not 'Weaver+77 Eq. 20'.",
    "failure_scenario": "A wrong equation number is inherited by every downstream doc and reader and cannot be checked without journal access; it also masks whether the coefficient was taken from Weaver or from Castor/McCray, which are different solutions with different radius coefficients.",
    "repro": "",
    "confidence": "high"
  },
  {
    "id": "S3-C-25",
    "file": "trinity/phase0_init/get_InitPhaseParam.py",
    "line": 44,
    "class": "coefficient",
    "severity": "S2",
    "claim": "If the seed forms P_b from E_b, the relation must be P_b = E_b/(2 pi (R2^3 - R1^3)) - the 2 pi is (gamma-1)*(4 pi/3) inverted for gamma=5/3, not a solid angle.",
    "evidence": "E = P V/(gamma-1) with V = (4 pi/3)(R2^3-R1^3); for gamma=5/3, E = (3/2)P V so P = (2/3)E/V = E/(2 pi (R2^3-R1^3)). Deriving it as 'E over a hemisphere volume' or as E/((4pi/3)R^3) (dropping gamma-1) gives a factor 2 or 3/2 error respectively. Weaver's own approximation drops R1, which is fine only while R1 << R2 - and at the seed R1/R2 = sqrt(11 v2/(3 v_w)) is NOT small (see S3-C-04).",
    "expected": "E_b/(2*pi*(R2**3 - R1**3)), with the same volume used wherever P_b dV/dt appears (SPEC-035 trap i).",
    "failure_scenario": "A factor 2 or 1.5 in the initial P_b feeds straight into R1 = sqrt(pdot/(4 pi P_b)) (as 1/sqrt) and into the shell's driving pressure at step 0. If the seed uses (4pi/3)R2^3 while the ODE uses 2 pi (R2^3-R1^3), the two disagree from the first step - a silent energy leak of exactly the kind SPEC-035 flags.",
    "repro": "From snapshot 0 check Pb * 2*pi*(R2**3 - R1**3) == Eb to round-off",
    "confidence": "high"
  },
  {
    "id": "S3-C-26",
    "file": "trinity/phase0_init/get_InitCloudProp.py",
    "line": 558,
    "class": "deadcode",
    "severity": "S4",
    "claim": "MockParam is a test double living in a production module's file; it belongs in test/.",
    "evidence": "Signature listing shows class MockParam at L558 with __init__(self, v) at L559, immediately after the two verify_* helpers - the shape of a __main__ self-check block.",
    "expected": "Either inside an if __name__ == '__main__' guard (acceptable, if the project tolerates it) or moved into the pytest suite, per CLAUDE.md's rule that checks live in test_*.py.",
    "failure_scenario": "Pre-existing, harmless at runtime; flag only, do not delete (CLAUDE.md rule 3).",
    "repro": "",
    "confidence": "medium"
  }
]
```
