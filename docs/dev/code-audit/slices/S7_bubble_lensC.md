# S7 bubble structure — Lens C (what it should be)

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

**Role.** Physics-tier lens. I read only `signatures.md` (names + line numbers, no bodies, no
constant values) and `docs/dev/code-audit/reference/PHYSICS_SPEC.md`. I did not read `trinity/`.
Everything below is derived from first principles or from internal knowledge of Weaver et al. 1977,
Cowie & McKee 1977, Spitzer, and the WARPFIELD line of descent. **Literature fetch is blocked in
this container** (arXiv/ADS/journals 403); I did not attempt it. Confidence tags are strict:
anything whose only support is "I remember the paper says so" is capped at **medium**, and bare
equation numbers are refused outright.

Spec ids cited as `SPEC-nnn` refer to the reference physics spec.

---

## 1. The physical system these two files must implement

Region 2 of the Weaver four-zone structure (SPEC-002): the **shocked wind**, `R1 < r < R2`.

- `R1` — wind termination (inner) shock. Free wind inside, shocked wind outside.
- `R2` — contact discontinuity (CD). Shocked wind inside, swept-up shell outside.
- Between them the gas is hot (`10^6–10^8 K`), tenuous, **subsonic**, and therefore **isobaric**
  to excellent accuracy: `P(r) = P_b` independent of `r`. This is the single most important
  structural statement — everything else follows from it. (SPEC-040, SPEC-024.)
- **Spitzer electron conduction** transports heat outward from the hot interior into the cold
  shell. That heat evaporates shell material, which flows *inward* across the CD and becomes the
  dominant mass content of the bubble. The bubble is therefore an **evaporation-fed**, not a
  wind-fed, reservoir for most of its life.

### 1.1 Governing equation set (derived here, end to end)

Take the total (thermal) energy density `e = (3/2)P`, enthalpy flux `(5/2)P v`, Spitzer flux
`q = −κ ∇T` with `κ = C T^{5/2}` (SPEC-043), and a volumetric radiative loss `Λ_vol`:

```
    ∂e/∂t + ∇·[(e+P)v] = ∇·(κ∇T) − Λ_vol
⇒   (3/2) Ṗ + (5/2) P (1/r²) d(r² v)/dr = (1/r²) d/dr( r² C T^{5/2} dT/dr ) − Λ_vol      … (E)
```

Continuity with `ρ = μ m_H P /(k_B T)` and `P` spatially uniform gives

```
    ∇·v = − Ṗ/P + (1/T)( ∂T/∂t + v ∂T/∂r )                                              … (C)
```

Substituting (C) into (E) removes `∇·v` and leaves a single scalar equation. Now introduce the
self-similar logarithmic derivatives (SPEC-041):

```
    α ≡ d ln R2 / d ln t = v2 t / R2      (dimensionless)
    β ≡ − d ln P_b / d ln t   ⇒  Ṗ_b = −β P_b / t
    δ ≡   d ln T / d ln t     ⇒  ∂T/∂t|_x = δ T / t
```

and convert the Eulerian time derivative at fixed `r` (the chain rule at fixed `x = r/R2`):

```
    ∂T/∂t |_r  =  δ T/t  −  α r (dT/dr) / t                                             … (S)
```

Putting (C), (S) and `Ṗ = −βP/t` into (E) and solving for the two derivatives gives **the ODE
system the RHS functions must implement**:

```
    dv/dr    =  (β + δ)/t  −  2v/r  +  (v − α r/t) (1/T) dT/dr                          … (V)

    d²T/dr²  =  (P_b /(C T^{5/2})) [ (β + 5δ/2)/t + (5/2)(v − α r/t)(1/T) dT/dr ]
                +  Λ_vol /(C T^{5/2})
                −  (5/2)(dT/dr)²/T
                −  (2/r)(dT/dr)                                                          … (T)
```

**Confidence: high.** I derived both lines here from (E)+(C)+(S); every coefficient (`5/2` from the
enthalpy `γ/(γ−1)` with `γ=5/3`, `5/2` from differentiating `T^{5/2}`, `2/r` from the spherical
divergence) is reproducible. Two independent sanity checks pass: (i) setting `Λ_vol = 0`,
`β=4/5, δ=−6/35, α=3/5` must admit the Weaver similarity solution; (ii) the closure
`T_b^{7/2} ∝ P_b R2²/(C t)` differentiates to `δ = (2/7)(2α − β − 1)`, which returns `−6/35`
exactly at Weaver's `α, β` (SPEC-042).

**Signs to check individually:**
- `+Λ_vol/(C T^{5/2})` in (T) — radiative losses must be *replenished* by conduction, i.e. they
  make `d²T/dr²` more positive (steeper inward-rising `T`). A minus sign here inverts the effect of
  cooling on the structure and is a silent S1.
- `β` carries a **minus sign in its definition**. `β > 0` corresponds to a *decaying* pressure
  (Weaver: `β = 4/5`). Dropping the minus flips the `(β + 5δ/2)/t` term.
- `−2v/r` in (V) is the spherical-divergence term and is negative for outflow.

**Two `_rhs` functions appear in the signature list** (`L337` inside the residual path, `L490`
inside `_solve_bubble_structure`). Whatever the reason, they must encode **term-identical physics**.
A defensible variant is that the residual pass omits `Λ_vol` (Weaver's own cooling-free structure)
while the final pass includes it — but then the reported profile does **not** satisfy the condition
that fixed `dMdt`, and that must be stated. Any *undocumented* difference between the two is a
first-class finding.

---

## 2. Interior structure: exponents, boundary conditions, singularity

### 2.1 The conduction-front asymptotic at the contact discontinuity (derived)

Near `R2` the dominant balance is: outward conductive flux = inward enthalpy flux of the evaporating
mass `Ṁ` (mass per unit time crossing the CD from shell into bubble, `Ṁ > 0`). In the front frame,

```
    4π r² C T^{5/2} (−dT/dr)  =  (5/2) (k_B /(μ m_H)) T Ṁ
⇒   d(T^{5/2})/dr = − 25 k_B Ṁ /(16 π μ m_H C r²)
⇒   T(r) = [ (25/4) · (k_B /(μ m_H C)) · (Ṁ /(4π R2²)) · (R2 − r) ]^{2/5}                … (BC-T)
⇒   dT/dr = − (2/5) T /(R2 − r)                            (strictly negative)            … (BC-dT)
⇒   n(r) = P_b /(k_B T) ∝ (R2 − r)^{−2/5}                                                 … (BC-n)
```

and the gas velocity just inside the CD, from `ρ (Ṙ2 − v) = Ṁ/(4πr²)` and `1/ρ = k_B T/(μ m_H P_b)`:

```
    v(r) = α r / t  −  ( Ṁ /(4π r²) ) · ( k_B T /( μ m_H P_b ) )                          … (BC-v)
```

(`α r/t` is the local co-moving speed; at `r = R2` it is exactly `v2`.)

- **Exponents `+2/5` for `T` and `−2/5` for `n`: confidence high.** They are forced by the Spitzer
  `T^{5/2}` and the `T¹` of the enthalpy; nothing else can produce them. Matches SPEC-040.
- **Coefficient `25/4` (equivalently `25/(16π)` in the `Ṁ/R2²` form): confidence medium-high.** It
  is derived above and I independently recall this exact grouping
  (`(25/4)·k_B/(μ C)·[Ṁ/(4πR2²)]·ΔR`) from the WARPFIELD lineage. It is **only** valid if the front
  carries the **enthalpy** flux `(5/2)nkTv`. If internal energy `(3/2)nkTv` is used instead, the
  coefficient becomes `15/4` and `T` drops by `(15/25)^{2/5} = 0.81` — a silent 19% error that
  preserves the exponent. Flag: exponent-correct is *not* coefficient-correct.
- **The `μ` in (BC-T) and (BC-v) is mass per *particle*** (`μ_ion ≈ 0.609`, `n_tot/n_H = 2.3`), not
  mass per hydrogen nucleus (`1.4`). Interchanging them is a factor `2.3` in `ρ↔n` and hence
  `2.3^{2/5} = 1.39` in `T(r2Prime)` and `2.3` in `v`. This is SPEC-092 trap #1 in its most
  dangerous form because both constants exist in `default.param`.

### 2.2 Where `r2Prime` sits, and why it is not `0.98 R2`

`r2Prime` is the outer end of the integration, defined by a chosen starting temperature
`T_init` via inversion of (BC-T):

```
    R2 − r2Prime = (4/25) (μ m_H C /k_B) (4π R2² /Ṁ) T_init^{5/2}                          … (Δ)
```

Because `Δ ∝ T_init^{5/2}`, the choice of `T_init` is violently non-linear: raising it from
`3×10⁴ K` to `10⁵ K` moves `r2Prime` inward by a factor `20`. Worked numbers for
`L₃₆ = n₀ = t₆ = 1` (`R2 ≈ 28 pc`, `T_b ≈ 1.5×10⁶ K`, `P_b/k_B ≈ 2.6×10⁴ K cm⁻³`):

| `T` | `1 − r/R2` | `R2 − r` | `n_tot = P/(k T)` |
|---|---|---|---|
| `T_b = 1.5e6 K` | 1 (interior) | ~28 pc | 0.017 cm⁻³ |
| `3e5 K` | `1.8e-2` | 0.50 pc | 0.086 cm⁻³ |
| `1e5 K` | `1.2e-3` | 0.032 pc | 0.26 cm⁻³ |
| `3e4 K` | `5.7e-5` | 1.6e-3 pc | 0.86 cm⁻³ |
| `1e4 K` | `3.6e-6` | 1.0e-4 pc | 2.6 cm⁻³ |

Note `ξ = 0.98` (`bubble_xi_Tb`, where `T0` is *reported*) sits at `T = 0.209 T_b ≈ 3×10⁵ K`
(SPEC-040 / test T10), i.e. **at the top of the conduction front, not in its body**. `r2Prime` must
be far outside `0.98 R2`. These are three distinct radii — `0.98 R2` (reporting), `r2Prime`
(integration limit), `R2` (the singularity) — and conflating any two is a finding.

### 2.3 Singular behaviour at the CD

`T → 0` and `n → ∞` as `r → R2`. Consequences the implementation must respect:

- The ODE cannot be integrated *to* `R2`; it must stop at `r2Prime < R2`.
- `n(r)` diverges as `(R2−r)^{−2/5}` — **integrable** (exponent `< 1`), so mass and luminosity
  integrals converge, but the *integrand* is unbounded at the endpoint.
- The natural integration direction is **inward from `r2Prime` to `R1`**, because (BC-T), (BC-dT),
  (BC-v) supply all three initial conditions there in terms of the single unknown `Ṁ`. Integrating
  outward toward the singular point is both under-determined (needs a shooting variable at `R1`)
  and numerically repulsive.

### 2.4 Inner boundary condition at `R1` — the eigenvalue condition on `Ṁ`

The system is a two-point BVP: three ICs at `r2Prime` parameterised by one unknown `Ṁ`, and one
scalar condition at `R1`. The condition is the Rankine–Hugoniot jump at the wind termination shock
(strong shock, `γ = 5/3`, compression ratio 4):

```
    post-shock speed (shock frame)  u₂ = u₁/4,  u₁ = v_w − Ṙ1
⇒   v(R1) = Ṙ1 + (v_w − Ṙ1)/4  ≈  v_w/4                                                   … (BC-R1)
    post-shock temperature  T(R1) = (3/16) μ m_H v_w² / k_B
    post-shock pressure     P(R1) = (3/4) ρ_w v_w²  =  P_b
```

Numeric anchor (derived): `T(R1) = 1.38×10⁷ K (v_w/1000 km s⁻¹)²` at `μ = 0.609`.
**Confidence: high** for the jump conditions; **medium** that a velocity residual at `R1` is *the*
right closure rather than an equivalent mass-flux statement — the two are algebraically identical
*only if* the same strong-shock convention is used in `R1`'s definition. Concretely:

> **Internal-consistency trap (S2).** If `R1` is defined by `ρ_w v_w² = P_b` (Weaver's convention,
> SPEC-025, no `3/4`) then the post-shock density implied by the shock is `4ρ_w` and hence
> `T(R1) = μ m_H v_w²/(4 k_B)` — which is `4/3 ×` the true post-shock temperature. Using
> `R1 = √(ṗ/4πP_b)` *and* `T(R1) = 3μm_H v_w²/16k_B` in the same run is inconsistent at the 33%
> level in `T(R1)` and 4/3 in `ρ(R1)`. One convention must be chosen and used everywhere.

The residual should be **relative** (`(v_solved(R1) − v_w/4)/(v_w/4)`) so the root-finder's
tolerance is scale-free, and it should be sign-changing across the root so a bracketing method
(brentq) can be used with a guaranteed bracket.

---

## 3. The luminosity integral — the central question

### 3.1 What is integrated, over what, between what limits, with what element

**The volume element is `4π r² dr`. Nothing else.** A spherical shell of radius `r` and thickness
`dr` has volume `4πr²dr`. `2πr dr` is the area element of a *plane annulus* (dimension L², would
make `L` come out as `erg s⁻¹ cm⁻¹`); a bare `dr` is a *line* element (dimension L, `erg s⁻¹ cm⁻²`).
Both fail dimensionally and both are recoverable only by an accidental compensating factor
elsewhere. State plainly:

```
    L_cool = ∫_{R1}^{r2Prime}  n_e(r) n_H(r) Λ(T(r))  ·  4π r² dr        [erg s⁻¹]        … (L)
```

- **Integrand**: `n_e n_H Λ(T)` is the volumetric emissivity in `erg cm⁻³ s⁻¹` when `Λ` is a
  *cooling efficiency* in `erg cm³ s⁻¹`. The density *product convention must match the table*
  (SPEC-082). For fully ionised solar-composition gas, `n_e = 1.2 n_H`, `n_tot = 2.3 n_H`, so
  `n_tot² Λ` over-counts `n_e n_H Λ` by `2.3²/1.2 = 4.4`; `n_H² Λ` under-counts by `1.2`. Since the
  bubble is isobaric, whichever density is stored in the profile array satisfies
  `n·T = P_b/(k_B · f)` with `f = 1` only for `n = n_tot`. A factor 4.4 here moves the
  energy→momentum transition time directly (SPEC-013/082) — **S1**.
- **Limits**: `R1` (inner) to `r2Prime` (outer). Not `0`, not `0.98 R2`, not `R2`. Extending below
  `R1` would count free-wind gas that is not in the bubble; extending past `r2Prime` walks into the
  singularity and into gas the shell module also owns.
- **`Λ(T)` must be the bolometric net radiative loss**, not a band-limited X-ray emissivity. Weaver's
  famous `L_X` is a `0.5–4.5 keV` band quantity and is a small fraction of the bolometric loss. If
  the bubble energy equation is fed a band luminosity, `L_cool` is under-counted by a large,
  temperature-dependent factor and the bubble stays energy-driven far too long — **S1**.
  Conversely, labelling the bolometric `L_cool` as "X-ray luminosity" in outputs is an S3
  mislabel.

### 3.2 Where the luminosity actually comes from (quantified — this drives everything)

Using the isobaric relation and the front asymptotic, the emission per logarithmic temperature
interval inside the conduction front is

```
    dL/d ln T  ∝  Λ(T) · T^{1/2}
```

*Derivation:* in the front, `R2 − r = T^{5/2}/A` with `A = (25/4)(k_B/(μ m_H C))(Ṁ/4πR2²)`, so
`dr = (5/2) T^{3/2} dT / A`; the emissivity `∝ n² Λ ∝ T^{−2}Λ`; the product is `Λ(T) T^{−1/2} dT`,
i.e. `Λ T^{1/2}` per `d ln T`. Evaluating with a standard solar-metallicity CIE curve:

| `T` | `Λ` (order) | `Λ T^{1/2}` (relative) |
|---|---|---|
| `1e4 K` | falling steeply | ~0.03 |
| `1e5 K` | `~1e-21` (CIE peak) | **1.0** |
| `1e6 K` | `~3e-23` | ~0.09 |
| `1e7 K` | `~1e-23` | ~0.09 |

**Therefore: `L_cool` is dominated by the `T ≈ 10⁵ K` layer immediately inside the contact
discontinuity, by roughly two orders of magnitude over the hot interior.** A cross-check by direct
volume integration for `L₃₆ = n₀ = t₆ = 1` gives `L_bulk ~ 5×10³³ erg s⁻¹` versus
`L_front ~ 10³⁵ erg s⁻¹` (both ±1 dex given my `Λ` estimates). **Confidence: high on the ordering,
medium on the magnitudes.**

Closed form for the front contribution (derived):

```
    L_front = (32π²/5) · ( μ m_H C R2⁴ /(k_B Ṁ) ) · f · (P_b/k_B)² · ∫ Λ(T) T^{−1/2} dT   … (LF)
    with f = (n_e/n_tot)(n_H/n_tot) = 0.227 for the n_e n_H convention
```

The scalings in (LF) are the audit levers: **`L_front ∝ P_b² R2⁴ C / Ṁ`**, and `∝ 1/Ṁ` because a
smaller evaporation rate makes the front geometrically *thicker*. Anything that perturbs `Ṁ`
perturbs `L_cool` inversely.

### 3.3 Numerical consequences — the endpoint-singularity trap

Because the emission peaks at `1 − r/R2 ≈ 1.2×10⁻³` and the integrand behaves as `(R2−r)^{−4/5}`
near the endpoint:

- A **uniform radial grid** of `N` points has spacing `R2/N`. For `N ≈ 100–1000` the entire
  emitting layer falls between the last one or two grid points, and the trapezoid rule on a
  `u^{−4/5}` endpoint singularity converges only as `O(h^{1/5})` — i.e. a 10× finer grid buys 60%
  error reduction. **The grid must be logarithmic in `(R2 − r)` spanning at least `10⁻⁶ … 1`, or
  the integral must be transformed to `T` (or `log T`) where the integrand `Λ(T)T^{−1/2}` is
  bounded and smooth.** This is an S1/S2-class numerical finding, not a style point.
- The **low-`T` cutoff** (`_T_INIT_BOUNDARY`, `_T_INTERFACE_BAND`) sets how much of the peak is
  captured. Cutting at `10⁵ K` discards roughly half the emission; cutting at `10^{5.5} K` discards
  most of it. The cutoff must be at or below the cooling table's floor (`10⁴ K` for a
  Gnat–Ferland-class CIE table, SPEC-081).
- If the bulk and front contributions are computed by two different quadratures, the **junction must
  not double-count** (and must not leave a gap).

### 3.4 `_get_mass_and_grav`

```
    M_b   = ∫_{R1}^{r2Prime} ρ(r) 4π r² dr ,     ρ = μ m_H n   (μ matched to n's convention)
    grav  : if a force,   ∝ +G M(<r) dm / r²   (attractive, directed inward)
            if an energy, U = − ∫ G M(<r) dm / r   (negative, binding)
```

- Same `4πr²dr` element. Same `μ` trap (SPEC-092 #1): `ρ = 1.4 m_H n_H` **or**
  `ρ = 0.609 m_H n_tot` — these are the *same* mass, but pairing `1.4` with `n_tot` inflates the
  bubble mass by `2.3×`.
- Unlike `L_cool`, `M_b` is **bulk-dominated**: `∫(R2−r)^{−2/5}dr ∝ Δ^{3/5}`, so the razor-thin
  front holds `~(3.6×10⁻⁶)^{3/5} ≈ 10⁻³` of the mass. A correct implementation should show
  luminosity front-dominated and mass bulk-dominated — a good self-consistency probe.
- Sanity anchor (derived, and it closes): for `L₃₆ = n₀ = t₆ = 1`, Cowie–McKee-style evaporation
  gives `Ṁ ≈ 2×10²¹ g s⁻¹ ≈ 34 M⊙ Myr⁻¹`, and `M_b ≈ Ṁ t ≈ 30 M⊙` spread over `(4/3)πR2³`
  reproduces `n_tot = P_b/(k_B T_b) ≈ 0.02 cm⁻³`. **Mass, pressure and evaporation rate close on
  each other to ~10%** — this triangle is a strong regression test.

---

## 4. Dimensions, units, and every conversion boundary

TRINITY's internal system is `[M⊙, pc, Myr]` (SPEC-090). The cooling table (`Λ`, `erg cm³ s⁻¹`),
the Spitzer coefficient (`C`, `erg s⁻¹ cm⁻¹ K^{−7/2}`), `k_B`, and `m_H` are **irreducibly cgs**.

**Therefore the bubble-structure ODE and the luminosity integral should be solved wholly in cgs**
(`r` in cm, `T` in K, `v` in cm s⁻¹, `P` in dyn cm⁻², `t` in s, `Ṁ` in g s⁻¹), with conversion at
the interface. Conversion factors (SPEC-091):

| quantity | AU → cgs |
|---|---|
| `r` | `× 3.0857e18 cm/pc` |
| `t` | `× 3.15576e13 s/Myr` |
| `v` | `× 9.77781e4 cm s⁻¹ per pc/Myr` (`1 km/s = 1.0227 pc/Myr`) |
| `P_b` | `× 6.4721e-13 dyn cm⁻² per M⊙ pc⁻¹ Myr⁻²` |
| `Ṁ` | `× 6.3025e19 g s⁻¹ per M⊙ Myr⁻¹` |
| `L` | `÷ 6.0255e29` to return `M⊙ pc² Myr⁻³` |
| `E_b` | `× 1.90148e43 erg per M⊙ pc² Myr⁻²` |

Dimensional checks the implementation must satisfy:

- `(V)`: every term is `[velocity]/[length]` — `(β+δ)/t` requires `t` in **the same time unit as
  `v`**. Mixing `t` in Myr with `v` in cm/s is a `3×10¹³` error that will not raise anything.
- `(T)`: `P_b/(C T^{5/2})` has units `K cm⁻²` ✓ only in cgs. `Λ_vol/(C T^{5/2})` likewise.
- `(L)`: `[cm⁻³][cm⁻³][erg cm³ s⁻¹][cm³] = erg s⁻¹` ✓.
- `α, β, δ` are dimensionless; `δ = (2/7)(2α − β − 1)` in the Weaver regime is a unit-free check
  (SPEC-042, test T5).
- **Which `t`?** `α = v2 t/R2` presumes `t` is elapsed time *since the bubble started expanding*.
  If the run has a non-zero `tSF`, using the absolute clock instead of `t − tSF` biases `α`, `β`,
  `δ` and hence the whole interior structure. The presence of a `tSF` argument on
  `get_effective_bubble_pressure` shows the distinction exists in the code's vocabulary; the
  similarity terms must use the same convention consistently.

---

## 5. Validity regime of the Weaver conduction solution

The solution above is valid only where **all** of the following hold. Each failure mode is a real
regime the parameter files can reach.

1. **Radiative losses are a perturbation.** Formally: the front's radiative loss must be small
   compared with the conductive enthalpy flux through it, `L_front ≪ (5/2)(k_B/μ m_H) T Ṁ`. At
   `L₃₆ = n₀ = t₆ = 1` my estimates make these *comparable* (`~2×10³⁵` vs `~10³⁵ erg s⁻¹`) — i.e.
   the classical solution is only marginally self-consistent at fiducial parameters, and becomes
   invalid at higher ambient density (`L_front ∝ P_b² ∝ n₀^{6/5}`). Once invalid, the front becomes
   a *condensation* front (net mass flow bubble→shell, `Ṁ < 0`) and the whole parameterisation
   breaks. **The implementation must not keep reporting a Weaver `T0`/`L_cool` after the interior
   has gone radiative** (SPEC-013/014).
2. **Unsaturated conduction.** Classical Spitzer flux requires electron mean free path
   `λ_e ≪ T/|∇T|`. `λ_e ≈ 10⁴ T²/n_e cm` — at `T = 10⁷ K, n_e ~ 10⁻²` this is `~30 pc`, comparable
   to `R2`. The *hot dilute interior* is therefore formally in the saturated regime (Cowie & McKee
   `σ₀ ≳ 1`), where the true flux is capped at `q_sat ≈ 5φ ρ c_s³` and the classical formula
   over-predicts. The dense front itself is safely unsaturated. Net effect of ignoring saturation:
   **over-estimated evaporation, over-estimated bubble density, under-estimated `T_b`**.
3. **No magnetic suppression.** A tangled or tangential field reduces conduction across the CD by
   factors of `3–100`. A pure-hydro code sets the *maximum* possible evaporation.
4. **1-D, laminar contact discontinuity.** SPEC-015: 3-D work (El-Badry+19, Lancaster+21) shows
   turbulent mixing across a fractal CD removes energy far faster than 1-D conduction, so a faithful
   Weaver implementation keeps the bubble energy-driven **longer than reality**. This is a physics
   ceiling of the model, not a bug — but any "boost" knob applied to `L_cool` is patching precisely
   the `10⁵ K` layer identified in §3.2.
5. **CIE ionisation equilibrium in the front.** The front's flow time is short; the gas is
   over-ionised relative to CIE while cooling, which *reduces* `Λ` at `10⁵ K` by up to `2–3×`. The
   `10⁴–10^{5.5} K` band inside the bubble is also exposed to the cluster's ionising field. Neither
   the CIE table (equilibrium, no radiation field) nor a photoionised-shell table (wrong geometry,
   wrong heating source) is strictly right there (SPEC-080/081/083). This is the single largest
   *physical* uncertainty in `L_cool`, and it lives exactly where the emission is.
6. **Fully ionised, optically thin, no dust cooling** in the bubble.
7. **Uniform ambient medium** for the specific similarity exponents. With `ρ ∝ r^{−w}`,
   `α = 3/(5−w)` (SPEC-053) — the `α = 0.6, β = 0.8, δ = −6/35` values are **`w = 0` values only**
   and must be re-solved, not assumed, for `densPL_alpha ≠ 0`.
8. **Mass loading.** `FB_mColdWindFrac` reduces `v_w` at fixed `L_w` (SPEC-072), which lowers
   `T(R1)` as `v_w²` and raises `Ṁ_w` — the shock BC must use the *effective* post-loading `v_w`
   consistently with the `ṗ` used for `R1`.
9. **`R1 ≪ R2`.** The similarity solution assumes a small inner shock. As the bubble deflates,
   `R1 → R2`, the "interior" becomes a thin annulus, and both the profile and the `ξ = 0.98`
   sampling point become meaningless (see §7).

---

## 6. `get_bubbleParams.py` — function-by-function expectations

| line | function | what it must compute | confidence |
|---|---|---|---|
| L27 | `delta2dTdt(t, T, delta)` | `dT/dt = δ T / t`. Sign: `δ < 0` (Weaver `−6/35`) ⇒ cooling. Guard `t = 0`. | high |
| L47 | `dTdt2delta(t, T, dTdt)` | `δ = t (dT/dt)/T` — **exact inverse** of L27; round-trip to machine precision. Guard `T = 0`. | high |
| L69 | `cool_beta_to_Ebdot(params)` | `Ė_b = (1/(γ−1))[ Ṗ_b V_b + P_b V̇_b ]` with `Ṗ_b = −β P_b/t`, `V_b = (4π/3)(R2³−R1³)`, `V̇_b = 4π(R2² v2 − R1² Ṙ1)`. For `γ=5/3` the prefactor is `3/2`. | high |
| L140 | `Ebdot_to_cool_beta(...)` | exact inverse: `β = −t[ (γ−1)Ė_b − P_b V̇_b ]/(P_b V_b)`. Round-trip identity with L69. | high |
| L198 | `bubble_E2P(Eb, r2, r1, gamma)` | `P_b = (γ−1)E_b / V_b = 3(γ−1)E_b /(4π(r2³−r1³))`; `= E_b/(2π(r2³−r1³))` at `γ=5/3` (SPEC-024). `gamma` must enter as `(γ−1)`. `r1` must **not** be dropped — it is not small near the transition. | high |
| L242 | `get_leak_luminosity(C_f, R2, Pb, c_s, gamma)` | `L_leak = (1−C_f)·4πR2²·c_s·[γ/(γ−1)]·P_b` (enthalpy flux, `5/2 P_b` at `γ=5/3`). Must be **identically zero at `C_f = 1`** and `≥ 0` always. `c_s` must be the *bubble interior* sound speed `√(γP_b/ρ_b)` in the same unit system. | high on form, medium on `5/2` vs `3/2` (SPEC-036 ambiguity) |
| L286 | `pRam(r, Lmech, v_mech)` | `P_ram = ṗ/(4πr²)` with `ṗ = 2L/v` ⇒ **`P_ram = L/(2π r² v)`**. `∝ r⁻²`, positive, diverges at `r → 0`. (Or `(3/4)×` that if the strong-shock convention is used — must match `R1`'s.) | high |
| L311 | `get_effective_bubble_pressure(...)` | phase-aware driver (SPEC-022): `P_b` in energy/implicit, `P_ram(R2)` in momentum, continuous through transition. See the invariant in §7.4. | medium |
| L384 | `get_r1(r1, params)` | residual of the ram-balance root equation, `f(R1) = √(L_w (R2³−R1³)/(E_b v_w)) − R1` (equivalently `E_b R1² v_w − L_w(R2³−R1³)`). | high |
| L414 | `solve_R1(R2, Eb, Lmech, v_mech)` | root of the above on `(0, R2)`. | high |

**Derivation of the `R1` equation.** Ram balance `ρ_w(R1) v_w² = P_b` with
`ρ_w = Ṁ_w/(4πR1²v_w)` and `ṗ_w = Ṁ_w v_w = 2L_w/v_w` gives `P_b = L_w/(2π R1² v_w)`. Equating to
`P_b = E_b/(2π(R2³−R1³))` gives `E_b R1² v_w = L_w (R2³ − R1³)`. The function
`g(R1) = √(L_w(R2³−R1³)/(E_b v_w)) − R1` is **strictly decreasing** on `(0,R2)` with `g(0) > 0` and
`g(R2) = −R2 < 0` ⇒ a **unique root, always bracketable by `[0, R2]`**. A bracketing failure is
therefore proof of a sign or unit error, never of a hard problem. Dimensional check:
`[L/(E v)] = pc⁻¹`, `√(pc⁻¹·pc³) = pc` ✓.

---

## 7. Asymptotics and invariants the implementation must satisfy

### 7.1 Profiles
1. `dT/dr < 0` **everywhere** in `(R1, r2Prime)`; `T` monotonically decreasing outward.
2. `n(r)` monotonically **increasing** outward; `n(r) T(r) = P_b/k_B` (with the declared `n_tot`
   convention) to integrator tolerance — the isobaricity test (SPEC-040, test T9).
3. `v(r)` is **not** monotonic in general: `v ≈ v_w/4` at `R1` (hundreds of km/s), falls through a
   minimum (possibly negative, if the evaporative inflow exceeds `v2`), then rises to `≈ v2` at the
   CD. A code that asserts monotonic `v` will reject valid solutions. *(medium confidence)*
4. `T(r) → T_b(1 − r/R2)^{2/5}`, `n(r) → n_b(1 − r/R2)^{−2/5}` in the conduction-dominated limit.
5. `T0` reported at `ξ = 0.98` equals `0.209 × T_b` (`(0.02)^{2/5}`) — test T10.

### 7.2 Scalings (exponents high confidence, prefactors low)
```
    R2 ∝ (L/ρ)^{1/5} t^{3/5}          α = 3/5,   β = 4/5,   δ = −6/35
    P_b ∝ L^{2/5} ρ^{3/5} t^{−4/5}    T_b ∝ L^{8/35} n₀^{2/35} t^{−6/35}
    n_b ∝ L^{6/35} n₀^{19/35} t^{−22/35}      (product n_b T_b ∝ L^{2/5}n₀^{3/5}t^{−4/5} ✓ isobaric)
    Ṁ_evap ∝ L^{27/35} n₀^{−2/35} t^{6/35}    (derived twice: from C T_b^{5/2}R2, and from M_b/t)
    δ = (2/7)(2α − β − 1)                      (SPEC-042; exact at Weaver values)
```
The `Ṁ ∝ n₀^{−2/35}` exponent I derived by two independent routes that agree; note it is **not**
`19/35` — that exponent belongs to `n_b`, and the two are easy to confuse.

### 7.3 Signs and positivity
`Ṁ_evap > 0`; `L_cool > 0`; `M_b > 0`; `L_leak ≥ 0`; `P_b > 0`; `0 < R1 < R2`; `V_b > 0`;
`T > 0` everywhere; gravity attractive (inward on the shell).

### 7.4 Pressure / energy / radius relations
```
    P_b = (γ−1) E_b / V_b ,    V_b = (4π/3)(R2³ − R1³)
    P_b = L_w /(2π R1² v_w)          (from ram balance — an identity once R1 is solved)
⇒   P_b / P_ram(R2) = (R2/R1)²  ≥ 1                                                        … (I)
⇒   as E_b → 0,  R1 → R2  and  P_b → ṗ_w/(4πR2²) = P_ram(R2)  continuously
```
**(I) is a strong, cheap, derived invariant.** Two consequences:
- `max(P_b, P_ram(R2))` is **identically `P_b`** if `R1` is the ram-balance radius and `V_b` retains
  `R1`. If such a `max()` is ever observed to select `P_ram`, then either `R1` is not the ram-balance
  radius or `V_b` was computed as `(4π/3)R2³` with `R1` dropped (SPEC-024's trap).
- The energy→momentum handover is **automatically continuous** in this formulation: the bubble
  pressure decays smoothly onto the free-wind ram pressure. Any hard switch that jumps
  `P_drive` is therefore introducing a discontinuity the physics does not require (SPEC-016,
  test T13).

### 7.5 Global energy checks
`E_b/(L_w t) → 5/11 = 0.4545` in the gravity-free, radiation-free energy phase (SPEC-051, test T3);
`α = v2 t/R2 → 0.6` (test T4). `L_cool` from (L) must be consistent with `dE_b/dt` bookkeeping
(SPEC-035) — the same `V_b` in `P_b(E_b)` and in `P_b dV_b/dt`.

---

## 8. Known traps, ranked

1. **Volume element.** `4πr²dr` only. (§3.1)
2. **Cooling normalisation.** `n_e n_H` vs `n_H²` vs `n_tot²` — up to `4.4×`, and it lands directly
   on the phase-transition trigger. (SPEC-082)
3. **The `10⁵ K` layer.** It carries ~99% of `L_cool` and occupies `~10⁻³` of the radius. Grid
   resolution, the low-`T` cutoff, and the `Λ` table choice in `10⁴–10^{5.5} K` *are* the answer.
   A uniform radial grid silently discards it. (§3.2–3.3)
4. **`γ = 5/3`-only coefficients.** `(γ−1) = 2/3` in `E2P`; `γ/(γ−1) = 5/2` in the leak and in the
   front's enthalpy flux; the `2/5` profile exponent and the `25/4` front coefficient both descend
   from that same `5/2`. If `gamma` is a live parameter anywhere, these hard numbers must move with
   it or be documented as `γ=5/3`-only. Enthalpy `5/2` vs internal energy `3/2` changes the front
   temperature by 19% while leaving the exponent intact — exponent-correct is not proof.
5. **Bolometric vs band-limited.** Weaver's `L_X` is `0.5–4.5 keV`; the energy equation needs
   bolometric. (§3.1)
6. **Weaver profile applied after the interior has cooled.** Post-transition the isobaric conduction
   solution is invalid; `T0` and `L_cool` computed from it are meaningless. (§5.1)
7. **Equation numbering.** I refuse to assert Weaver+77 equation numbers — the paper is unreachable
   here, and the Rahner-thesis numbering differs. Any code comment of the form "Weaver eq. N" is
   **unverifiable from this container** and must not be treated as evidence for a coefficient.
   Endorses SPEC-045.
8. **Hard-coded Weaver prefactors.** `T_b = 1.51e6` / `n_b = 4.02e-3` do **not** reproduce the
   dynamical `P_b` (SPEC-045). Refinement of SPEC-045 from my own arithmetic: with
   `n_b` read as a **hydrogen** density and `P = 2.3 n_H k T`, the mismatch drops from `4.2×` to
   `1.8×`; with the alternative `T_b = 2.07e6` prefactor it drops to `1.34×`. So the discrepancy is
   partly a composition-convention artefact — but it does not vanish, and the structural forms
   (SPEC-024/042) remain the only trustworthy route. **[low confidence on the prefactors themselves]**
9. **`ξ = 0.98` sampling when `R1 > 0.98 R2`.** As `E_b` falls, `R1 → R2`; the reporting radius then
   lies *inside the free wind*, and any interpolation of the bubble profile there is an
   extrapolation. Must be guarded, not silently returned.
10. **Root-finder state.** `brentq`/`fsolve` do not guarantee the final residual evaluation sits at
    the returned root. Profiles cached during residual evaluation may correspond to a different
    `dMdt` than the converged one. A sentinel `_SOLVER_FAIL_RESIDUAL` returned on integration
    failure can also create a spurious sign change and a false root.
11. **Grid cleaning.** Removing "too-close" radii is exactly the operation that destroys front
    resolution; it must preserve endpoints, strict monotonicity, and the log-refined region.
12. **`t` vs `t − tSF`** in the similarity terms; and `t` in the same unit as `v` and `r`. (§4)

---

## 9. Honest statement of confidence

- **High**: the isobaric+conduction ODE system (V)/(T) as derived; the `±2/5` exponents; the
  `4πr²dr` element; the `n_e n_H Λ` integrand structure; the `R1` root equation and its unique
  bracketing; `bubble_E2P`; `pRam`; the `δ = (2/7)(2α−β−1)` closure; the strong-shock jump numbers;
  the front-dominance of `L_cool`; invariant (I).
- **Medium**: the `25/4` coefficient in (BC-T) (derivation is clean, but it is contingent on the
  enthalpy convention and on `μ` being mass-per-particle); the Cowie–McKee `16π/25` evaporation
  coefficient; `5/2` vs `3/2` in the leak flux; that the inner closure is a *velocity* residual
  rather than an equivalent mass-flux residual; my absolute `L_front` magnitude (±1 dex, driven by
  my `Λ` estimates).
- **Low**: every Weaver numerical prefactor (`1.51e6`, `2.07e6`, `4.02e-3`); all Weaver equation
  numbers (refused); the exact `Λ(T)` values used above.

---

```json
[
  {"id":"S7-C-01","file":"trinity/bubble_structure/bubble_luminosity.py","line":625,"class":"units","severity":"S1","claim":"The bubble radiative-loss integral must use the spherical volume element 4*pi*r^2*dr.","evidence":"A spherical shell of radius r and thickness dr has volume 4*pi*r^2*dr. 2*pi*r*dr is a plane-annulus area element (dimension L^2) and a bare dr is a line element (dimension L); with an emissivity in erg cm^-3 s^-1 they give erg s^-1 cm^-1 and erg s^-1 cm^-2 respectively, not erg s^-1.","expected":"L_cool = INT_{R1}^{r2Prime} n_e n_H Lambda(T) 4 pi r^2 dr","failure_scenario":"L_cool is wrong by a factor ~r or ~4*pi*r^2, i.e. orders of magnitude and radius-dependent; the energy->momentum transition time (the code's headline prediction) is set by L_cool and moves arbitrarily.","repro":"Compare the computed L against an analytic uniform-sphere case: constant n,T over R1..R2 must give n_e n_H Lambda (4/3)pi(R2^3-R1^3).","confidence":"high"},
  {"id":"S7-C-02","file":"trinity/bubble_structure/bubble_luminosity.py","line":625,"class":"coefficient","severity":"S1","claim":"The emissivity must be the density PRODUCT convention that matches the cooling table (n_e*n_H for a 'cooling efficiency' table), not n_tot^2 or n_H^2 chosen ad hoc.","evidence":"SPEC-082. For fully ionised solar composition n_e=1.2 n_H, n_tot=2.3 n_H, so n_tot^2*Lambda over-counts n_e n_H Lambda by 2.3^2/1.2 = 4.4 and n_H^2 under-counts by 1.2.","expected":"emissivity = n_e n_H Lambda(T) with n_e,n_H derived from the isobaric n_tot = P_b/(k_B T) via the declared composition (n_e/n_tot=0.522, n_H/n_tot=0.435).","failure_scenario":"L_cool off by up to 4.4x. Because L_cool feeds the (L_gain-L_loss)/L_gain <= 0.05 transition trigger directly, the energy-phase duration, dispersal-vs-recollapse outcome and all Paper-II grid results shift.","repro":"Check n*T against P_b/k_B on the stored bubble profile: only n_tot satisfies it exactly.","confidence":"high"},
  {"id":"S7-C-03","file":"trinity/bubble_structure/bubble_luminosity.py","line":625,"class":"regime","severity":"S2","claim":"Integration limits must be R1 (inner shock) to r2Prime (just inside the contact discontinuity) - not 0, not 0.98*R2, not R2.","evidence":"Region 2 of the Weaver structure is exactly R1<r<R2; inside R1 is free wind (not thermalised), outside R2 is shell gas owned by the shell module. The CD is singular so the upper limit must be r2Prime<R2.","expected":"lower limit = R1; upper limit = r2Prime with r2Prime<R2 defined by the front asymptotic; gas below the cooling-table floor excluded and NOT also counted by the shell.","failure_scenario":"Including r<R1 counts unshocked wind that radiates nothing; including r>r2Prime hits the n->infinity singularity; overlapping with the shell double-counts the 1e4 K gas, over-cooling the system.","repro":"","confidence":"high"},
  {"id":"S7-C-04","file":"trinity/bubble_structure/bubble_luminosity.py","line":625,"class":"numerical","severity":"S1","claim":"L_cool is dominated (by ~2 dex) by the T~1e5 K conduction-front layer at 1-r/R2 ~ 1e-3, so the quadrature must be logarithmic in (R2-r) or transformed to T; a uniform radial grid silently discards the emission.","evidence":"Derived: in the front T^{5/2} ∝ (R2-r), so dr ∝ T^{3/2}dT and emissivity ∝ T^{-2}Lambda, giving dL/dlnT ∝ Lambda(T) T^{1/2}, which peaks at the CIE peak T~1e5 K. Direct volume estimate at L36=n0=t6=1: L_bulk~5e33 vs L_front~1e35 erg/s. The integrand goes as (R2-r)^{-4/5} at the endpoint, on which the trapezoid rule converges only as O(h^{1/5}).","expected":"grid log-refined in (R2-r) spanning at least 1e-6..1 of R2, or the front integral evaluated in T/logT space with a dedicated point count.","failure_scenario":"With a uniform grid of N<~1e3 points the entire emitting layer falls between the last two nodes; L_cool is under-estimated by orders of magnitude and the bubble never transitions to momentum-driven.","repro":"Halve/double the front point count and check L_cool convergence; a change >1% means the front is unresolved.","confidence":"high"},
  {"id":"S7-C-05","file":"trinity/bubble_structure/bubble_luminosity.py","line":625,"class":"other","severity":"S2","claim":"If the bulk and the conduction-front contributions are computed by separate quadratures they must join without double-counting or gapping.","evidence":"The front closed form (32*pi^2/5)(mu m_H C R2^4/(k_B Mdot)) f (P_b/k_B)^2 INT Lambda T^{-1/2} dT and the bulk 4*pi*r^2 integral both cover the interval near the junction temperature.","expected":"a single partition of [R1, r2Prime] with the junction radius/temperature used as the shared endpoint of both integrals.","failure_scenario":"Double-counting inflates L_cool (early transition); gapping discards the highest-emissivity decade (late/never transition).","repro":"","confidence":"medium"},
  {"id":"S7-C-06","file":"trinity/bubble_structure/bubble_luminosity.py","line":625,"class":"regime","severity":"S1","claim":"Lambda(T) fed to the bubble energy budget must be the BOLOMETRIC net radiative loss, not a band-limited X-ray emissivity.","evidence":"Weaver's widely quoted bubble luminosity is a 0.5-4.5 keV band quantity and is a small, strongly T-dependent fraction of the bolometric loss; the bubble energy equation dE_b/dt = L_gain - L_loss requires total losses.","expected":"bolometric CIE (T>1e5.5 K) plus the non-CIE net cooling-minus-heating table below (SPEC-080/084).","failure_scenario":"L_cool under-counted by a large T-dependent factor; the bubble stays energy-driven far too long, over-predicting feedback efficiency and cloud dispersal.","repro":"","confidence":"high"},
  {"id":"S7-C-07","file":"trinity/bubble_structure/bubble_luminosity.py","line":625,"class":"silent-failure","severity":"S2","claim":"L_cool must be strictly positive and finite; a failed profile solve must not yield L=0 or NaN that is then consumed by the energy equation.","evidence":"Positivity: n_e n_H Lambda >= 0 pointwise over a positive-measure interval with T>1e4 K. L=0 is indistinguishable from 'no cooling' and silently keeps the bubble adiabatic.","expected":"raise/propagate a solver error rather than returning a zero or NaN luminosity.","failure_scenario":"A silent L_cool=0 removes the only loss term and the bubble never transitions; a NaN propagates into E_b and terminates the run for the wrong reason.","repro":"","confidence":"high"},
  {"id":"S7-C-08","file":"trinity/bubble_structure/bubble_luminosity.py","line":915,"class":"units","severity":"S1","claim":"_get_mass_and_grav must integrate rho with the 4*pi*r^2*dr element and convert n->rho with the mu that matches n's convention (mu=1.4 m_H per H nucleus, or mu=0.609 m_H per particle).","evidence":"SPEC-092 trap #1. n_tot/n_H = 2.3, so pairing mu_H=1.4 with n_tot (or mu_ion=0.609 with n_H) mis-states the bubble mass by 2.3x in either direction.","expected":"M_b = INT 4 pi r^2 mu m_H n(r) dr with the pair (mu, n) consistent.","failure_scenario":"Bubble mass wrong by 2.3x, propagating into the enclosed mass for shell gravity and into any density-based diagnostic.","repro":"Cross-check: M_b/V_b must reproduce n_tot = P_b/(k_B T_b) to ~10%; at L36=n0=t6=1 that is ~30 Msun and n_tot~0.02 cm^-3.","confidence":"high"},
  {"id":"S7-C-09","file":"trinity/bubble_structure/bubble_luminosity.py","line":915,"class":"sign","severity":"S2","claim":"The gravitational quantity must be attractive: an inward force (positive magnitude subtracted in the shell EOM) or a negative binding energy.","evidence":"Newtonian gravity; SPEC-031 gives F_grav = G M_sh (M_cl + M_sh/2)/R2^2 entering the EOM with a minus sign.","expected":"force ∝ +G M(<r) dm/r^2 directed inward; potential energy negative.","failure_scenario":"A sign flip turns gravity into an outward push, removing re-collapse as a possible fate entirely.","repro":"","confidence":"high"},
  {"id":"S7-C-10","file":"trinity/bubble_structure/bubble_luminosity.py","line":337,"class":"coefficient","severity":"S1","claim":"The velocity ODE must be dv/dr = (beta+delta)/t - 2v/r + (v - alpha*r/t)(dT/dr)/T.","evidence":"Derived here from the isobaric continuity equation with rho = mu m_H P/(k T), P spatially uniform, Pdot = -beta P/t, and dT/dt|_r = (delta T - alpha r dT/dr)/t.","expected":"exactly those four terms with those signs; -2v/r is the spherical divergence term.","failure_scenario":"Wrong interior velocity field -> wrong advective term in the temperature equation -> wrong T(r), wrong n(r), wrong L_cool and wrong evaporation eigenvalue.","repro":"Set Lambda=0, alpha=3/5, beta=4/5, delta=-6/35 and check the Weaver similarity profile T ∝ (1-r/R2)^{2/5} is reproduced.","confidence":"high"},
  {"id":"S7-C-11","file":"trinity/bubble_structure/bubble_luminosity.py","line":337,"class":"coefficient","severity":"S1","claim":"The temperature ODE must be d2T/dr2 = (P_b/(C T^{5/2}))[(beta + 2.5*delta)/t + 2.5(v - alpha r/t)(dT/dr)/T] + Lambda_vol/(C T^{5/2}) - 2.5 (dT/dr)^2/T - (2/r) dT/dr, with the cooling term entering with a PLUS sign.","evidence":"Derived here by substituting continuity into the isobaric energy equation (3/2)Pdot + (5/2)P div v = div(C T^{5/2} grad T) - Lambda_vol. The 2.5 factors are gamma/(gamma-1)=5/2 (enthalpy) and d(T^{5/2})/dr = (5/2)T^{3/2}T'.","expected":"the exact term list above; +Lambda/(C T^{5/2}).","failure_scenario":"A sign error on the cooling term makes radiative losses flatten rather than steepen the gradient, inverting the feedback between cooling and evaporation; wrong 2.5 factors break the (1-x)^{2/5} similarity solution.","repro":"Turn cooling off and verify the solved profile matches (BC-T) over the whole interior.","confidence":"high"},
  {"id":"S7-C-12","file":"trinity/bubble_structure/bubble_luminosity.py","line":490,"class":"other","severity":"S2","claim":"The two _rhs implementations (residual path L337 and structure-solve path L490) must encode term-identical physics, or the difference must be explicit.","evidence":"Both integrate the same shocked-wind structure; the eigenvalue Mdot returned by the residual path is only meaningful for the profile the final path produces.","expected":"identical equations, or a documented statement that the residual pass is the cooling-free Weaver structure and that the final profile therefore does not exactly satisfy the R1 closure.","failure_scenario":"The converged Mdot solves a different problem than the reported profile; L_cool, T0 and the evaporation rate are mutually inconsistent and no invariant check will catch it.","repro":"Diff the two RHS term by term.","confidence":"medium"},
  {"id":"S7-C-13","file":"trinity/bubble_structure/bubble_luminosity.py","line":337,"class":"units","severity":"S1","claim":"The time t in the alpha/beta/delta similarity terms must be the elapsed bubble age in the SAME time unit as v and r (and, if tSF is non-zero, t-tSF rather than the absolute clock).","evidence":"Every similarity term has the form (dimensionless)/t or alpha*r/t and must be commensurate with dv/dr and v. alpha = v2 t/R2 presumes t measured from R2=0. Constants C, k_B, Lambda are irreducibly cgs, so the ODE should be solved wholly in cgs (SPEC-090/091).","expected":"cgs throughout the ODE: r [cm], t [s], v [cm/s], P [dyn/cm2], T [K]; conversion at the interface only.","failure_scenario":"Mixing Myr with cm/s is a 3.16e13 error in every similarity term; using absolute t instead of t-tSF biases alpha, beta, delta and hence the whole interior structure with no visible symptom.","repro":"Check the run's measured alpha = v2*t/R2 relaxes to 0.6 in the energy phase (SPEC-056, test T4).","confidence":"high"},
  {"id":"S7-C-14","file":"trinity/bubble_structure/bubble_luminosity.py","line":392,"class":"coefficient","severity":"S2","claim":"The starting temperature at r2Prime must be T = [(25/4)(k_B/(mu m_H C))(Mdot/(4 pi R2^2))(R2-r2Prime)]^{2/5}.","evidence":"Derived: conductive flux 4 pi r^2 C T^{5/2}(-dT/dr) balances the inward enthalpy flux (5/2)(k_B/(mu m_H)) T Mdot, integrating to d(T^{5/2})/dr = -25 k_B Mdot/(16 pi mu m_H C r^2). The 25/4 descends from the enthalpy 5/2; using internal energy 3/2 instead gives 15/4 and a 19% lower T with the SAME 2/5 exponent.","expected":"coefficient 25/4 (equivalently 25/(16 pi) in the Mdot/R2^2 form), mu = mass per particle (~0.609), exponent 2/5.","failure_scenario":"A 19% error in the front temperature scale propagates as ~T^{-2} into the emissivity (~50% in L_cool) and shifts the transition time; the exponent test passes regardless, so it is invisible to a profile-shape check.","repro":"","confidence":"medium"},
  {"id":"S7-C-15","file":"trinity/bubble_structure/bubble_luminosity.py","line":392,"class":"sign","severity":"S2","claim":"dT/dr at r2Prime must be -(2/5) T/(R2-r2Prime), i.e. strictly negative, and dT/dr must remain negative over the whole integration.","evidence":"Differentiating T ∝ (R2-r)^{2/5}. Physically T must fall monotonically outward from the shocked wind to the cold shell.","expected":"negative initial gradient; monotone decreasing T(r) on (R1, r2Prime).","failure_scenario":"A positive gradient reverses the conduction direction (heat flowing from shell into bubble), giving negative evaporation, an inverted density profile and a nonsensical luminosity.","repro":"Assert np.all(diff(T)<0) on the stored profile.","confidence":"high"},
  {"id":"S7-C-16","file":"trinity/bubble_structure/bubble_luminosity.py","line":392,"class":"coefficient","severity":"S3","claim":"The starting velocity at r2Prime must be v = alpha*r2Prime/t - (Mdot/(4 pi r2Prime^2))(k_B T/(mu m_H P_b)), i.e. the CD co-moving speed minus the evaporative inflow.","evidence":"Mass flux across the front rho(Rdot2 - v) = Mdot/(4 pi r^2) with 1/rho = k_B T/(mu m_H P_b) from the isobaric equation of state; alpha r/t equals v2 at r=R2.","expected":"v -> v2 as T -> 0 at the CD.","failure_scenario":"A wrong inner-flow sign or a missing 1/rho makes the velocity field inconsistent with the mass being evaporated, and the R1 residual then converges on a compensating (wrong) Mdot.","repro":"","confidence":"medium"},
  {"id":"S7-C-17","file":"trinity/bubble_structure/bubble_luminosity.py","line":311,"class":"coefficient","severity":"S2","claim":"The inner closure at R1 must be the strong-shock jump for gamma=5/3: v(R1) = Rdot1 + (v_w - Rdot1)/4 ~ v_w/4, consistent with T(R1) = 3 mu m_H v_w^2/(16 k_B) and P(R1) = (3/4) rho_w v_w^2.","evidence":"Rankine-Hugoniot, compression ratio 4 at gamma=5/3. Numeric anchor: T(R1) = 1.38e7 K (v_w/1000 km/s)^2 at mu=0.609.","expected":"a relative, sign-changing residual on v(R1) (or the algebraically equivalent mass-flux condition) that brackets the root.","failure_scenario":"A wrong target velocity moves the eigenvalue Mdot, which sets the front thickness; since L_front ∝ 1/Mdot, a factor-2 error in the closure is a factor-2 error in the dominant cooling term.","repro":"","confidence":"medium"},
  {"id":"S7-C-18","file":"trinity/bubble_structure/bubble_luminosity.py","line":311,"class":"numerical","severity":"S3","claim":"The R1 residual must be relative (scale-free) and monotone/sign-changing in dMdt so a bracketing root finder is valid.","evidence":"v(R1) ~ v_w/4 ~ 250 km/s while v(r2Prime) ~ v2 ~ 10 km/s; an absolute residual tolerance tuned to one is meaningless for the other.","expected":"residual = (v_solved(R1) - v_target)/v_target with a verified bracket.","failure_scenario":"Tolerance is effectively arbitrary across the parameter sweep; the solver 'converges' at a different accuracy for high- and low-v_w clusters.","repro":"","confidence":"medium"},
  {"id":"S7-C-19","file":"trinity/bubble_structure/bubble_luminosity.py","line":84,"class":"silent-failure","severity":"S2","claim":"A sentinel residual returned on ODE-integration failure must not be able to masquerade as, or manufacture, a root.","evidence":"A large constant substituted for a failed evaluation creates an artificial sign change adjacent to the failure region; brentq will happily bisect onto that discontinuity and report convergence.","expected":"integration failure propagates as an exception (BubbleSolverError) or is recorded such that a 'converged' result cannot be produced from failed evaluations.","failure_scenario":"A spurious dMdt is accepted; the bubble profile and L_cool are fabricated from a non-solution and the run continues silently.","repro":"","confidence":"medium"},
  {"id":"S7-C-20","file":"trinity/bubble_structure/bubble_luminosity.py","line":258,"class":"state","severity":"S2","claim":"The profile used for the luminosity must be recomputed at the CONVERGED dMdt, not cached from the last residual evaluation made by the root finder.","evidence":"brentq/fsolve do not guarantee that the final function evaluation is at the returned root; the last evaluated point can be an interval endpoint.","expected":"one final explicit solve at the returned dMdt (or proof that the cache key is the converged value).","failure_scenario":"Reported T0, L_cool and profiles correspond to a nearby non-root dMdt; results become dependent on the root finder's internal iteration order and are not reproducible across scipy versions.","repro":"Re-solve at the returned dMdt and diff the profile against the cached one.","confidence":"medium"},
  {"id":"S7-C-21","file":"trinity/bubble_structure/bubble_luminosity.py","line":297,"class":"state","severity":"S3","claim":"The initial dMdt guess must be a positive, physically scaled estimate (Mdot ~ 16 pi mu m_H C T^{5/2} R2/(25 k_B)) and must not make the converged answer depend on cross-timestep mutable state.","evidence":"Cowie & McKee classical evaporation (SPEC-044). Consistency check: Mdot ∝ C * T_b^{5/2} with T_b ∝ C^{-2/7} gives Mdot ∝ C^{2/7}, matching the El-Badry+19 scaling quoted for cooling_boost_kappa.","expected":"positive guess; deterministic result independent of run history/restart point.","failure_scenario":"If a stale global carries the previous timestep's dMdt into the guess and the residual has multiple near-roots, the trajectory becomes path-dependent and full-run equivalence tests fail non-reproducibly.","repro":"Restart a run from a mid-run snapshot and compare dMdt at the same t.","confidence":"medium"},
  {"id":"S7-C-22","file":"trinity/bubble_structure/bubble_luminosity.py","line":531,"class":"numerical","severity":"S2","claim":"_create_radius_grid must refine logarithmically toward R2, covering (1 - r/R2) from ~1e-6 to ~1.","evidence":"For L36=n0=t6=1 the T=1e5 K emission peak sits at 1-r/R2 = 1.2e-3 and T=1e4 K at 3.6e-6; a uniform grid of N points resolves only down to 1/N.","expected":"log-spaced in (R2-r), or uniform in T / T^{5/2}.","failure_scenario":"The dominant emitting layer is unresolved; L_cool is grid-dependent and under-estimated, and the reported answer changes with _RESIDUAL_NPTS / _CONDUCTION_NPTS.","repro":"Vary the point count by 2x and check L_cool stability.","confidence":"high"},
  {"id":"S7-C-23","file":"trinity/bubble_structure/bubble_luminosity.py","line":570,"class":"numerical","severity":"S3","claim":"_clean_radius_grid must preserve both endpoints and strict monotonicity, and must not thin the log-refined near-CD region.","evidence":"LSODA requires a strictly monotonic independent-variable array; the trapezoid integral requires the true endpoints; the near-CD points are precisely the ones a minimum-relative-spacing filter will delete because they are closest together.","expected":"duplicate/zero-width intervals removed while endpoints and the front's log refinement survive.","failure_scenario":"Silent removal of the front nodes cuts the dominant luminosity contribution; dropping an endpoint truncates the integration domain.","repro":"","confidence":"medium"},
  {"id":"S7-C-24","file":"trinity/bubble_structure/bubble_luminosity.py","line":52,"class":"regime","severity":"S2","claim":"The boundary temperature that defines r2Prime must be at or below the cooling table's low-T floor (~1e4 K), and the code must acknowledge that R2-r2Prime ∝ T_init^{5/2}.","evidence":"Derived inversion of the front asymptotic: R2-r2Prime = (4/25)(mu m_H C/k_B)(4 pi R2^2/Mdot) T_init^{5/2}. Raising T_init from 3e4 to 1e5 K moves r2Prime inward by 20x, and dL/dlnT peaks at 1e5 K.","expected":"T_init <= a few 1e4 K, with the discarded T<T_init emission either negligible or accounted by the shell module.","failure_scenario":"A boundary set at or above the CIE peak discards most of the bubble's radiative losses; the bubble never satisfies the transition criterion and stays energy-driven for the whole run.","repro":"Vary the boundary temperature and plot L_cool; a strong dependence proves the emission peak is being clipped.","confidence":"high"},
  {"id":"S7-C-25","file":"trinity/bubble_structure/bubble_luminosity.py","line":65,"class":"regime","severity":"S3","claim":"The 1e4-1e5.5 K interface band inside the bubble is served by neither table cleanly: CIE assumes ionisation equilibrium (invalid in a rapidly cooling conduction front, which is over-ionised and radiates up to 2-3x less), and the non-CIE cube is built for photoionised shell gas, not a conduction front.","evidence":"SPEC-080/081/083; non-equilibrium ionisation in cooling flows is a standard result. This band carries ~99% of the bubble's radiative losses (derived, section 3.2).","expected":"an explicit, documented choice for this band and an acknowledgement of the systematic.","failure_scenario":"The dominant term in the bubble energy budget rests on a cooling function used outside its validity regime; the transition time carries an undocumented factor-few systematic.","repro":"","confidence":"medium"},
  {"id":"S7-C-26","file":"trinity/bubble_structure/bubble_luminosity.py","line":199,"class":"silent-failure","severity":"S2","claim":"Sampling T0 at xi=0.98*R2 is only defined while R1 < 0.98*R2; as E_b falls, R1 -> R2 and the sampling radius enters the free wind.","evidence":"Derived: E_b R1^2 v_w = L_w (R2^3 - R1^3), so R1 -> R2 as E_b -> 0. The Weaver profile then has no support at 0.98 R2.","expected":"a guard that detects R1 >= xi*R2 and refuses/flags rather than interpolating (extrapolating) the profile.","failure_scenario":"T0 is silently extrapolated from a two-point profile or clamped, producing a plausible-looking but meaningless bubble temperature exactly in the regime (approach to the momentum phase) where the transition decision is being made.","repro":"Log R1/R2 through a run and check whether it crosses 0.98 before the phase switch.","confidence":"high"},
  {"id":"S7-C-27","file":"trinity/bubble_structure/bubble_luminosity.py","line":199,"class":"units","severity":"S2","claim":"The stored bubble profile must satisfy isobaricity: n(r)*T(r) = P_b/k_B with the declared total-particle convention (n_tot/n_H = 2.3).","evidence":"SPEC-040 / test T9. The whole solution is built on P spatially uniform; the only freedom is which density is stored.","expected":"n_tot*T = P_b/k_B to integrator tolerance across the profile; if n_H is stored, n_H*T = P_b/(2.3 k_B).","failure_scenario":"A stored density in the wrong convention silently mis-normalises every downstream use (cooling lookup, bubble mass, emission measure) by 2.3.","repro":"Assert max|n*T*k_B/P_b - 1| < tol on bubble_n_arr/bubble_T_arr.","confidence":"high"},
  {"id":"S7-C-28","file":"trinity/bubble_structure/bubble_luminosity.py","line":452,"class":"other","severity":"S3","claim":"T(r) must be monotonically decreasing and n(r) monotonically increasing outward, but v(r) is NOT monotonic and must not be constrained to be.","evidence":"Isobaric + conduction forces T down and n up outward. v ~ v_w/4 (hundreds of km/s) at R1 falls to a minimum and rises to ~v2 at the CD, since v(r) -> alpha r/t - (Mdot/4 pi r^2)(k T/(mu m_H P_b)) there; the evaporative inflow can drive v below v2 or negative.","expected":"monotonic guards on T and n only.","failure_scenario":"A monotonicity guard on v rejects physically valid solutions (or forces the root finder away from the true root) in exactly the evaporation-dominated regime the model is built for.","repro":"","confidence":"medium"},
  {"id":"S7-C-29","file":"trinity/bubble_structure/bubble_luminosity.py","line":625,"class":"units","severity":"S1","claim":"Every cgs constant (C, k_B, m_H, Lambda) marks a unit boundary: r must be in cm, T in K, P in dyn/cm2, Mdot in g/s, t in s wherever those constants appear.","evidence":"SPEC-091: pc=3.0857e18 cm, Myr=3.15576e13 s, pressure 6.4721e-13 dyn cm^-2 per AU unit, Mdot 6.3025e19 g/s per Msun/Myr, luminosity 6.0255e29 erg/s per AU unit.","expected":"cgs inside the structure/luminosity computation; a single documented conversion at the boundary in and out.","failure_scenario":"Using pc for r with cgs C is a 3e18 error in the conduction term; using Myr for t in the similarity terms is 3e13. Both produce finite, plausible-looking output.","repro":"Dimensional audit: the ratio Lambda_vol/(C T^{5/2}) must come out in K cm^-2.","confidence":"high"},
  {"id":"S7-C-30","file":"trinity/bubble_structure/bubble_luminosity.py","line":199,"class":"units","severity":"S1","claim":"The returned luminosity must be converted to the internal AU system (Msun pc^2 Myr^-3) before entering dE_b/dt, dividing the cgs value by 6.0255e29.","evidence":"SPEC-090/091; E_b is carried in Msun pc^2 Myr^-2, so a loss rate in erg/s is 30 orders of magnitude out of scale.","expected":"L [erg/s] / 6.0255e29 -> L [Msun pc^2 Myr^-3].","failure_scenario":"Loud failure if fully unconverted; a subtle one if only some contributions (e.g. the front but not the bulk) are converted, giving a physically-shaped but mis-weighted L_loss.","repro":"","confidence":"high"},
  {"id":"S7-C-31","file":"trinity/bubble_structure/get_bubbleParams.py","line":27,"class":"coefficient","severity":"S2","claim":"delta2dTdt must return dT/dt = delta*T/t exactly, with delta<0 giving a cooling bubble.","evidence":"Definition delta = dlnT/dlnt = (t/T)(dT/dt) (SPEC-041). Weaver value delta = -6/35.","expected":"dTdt = delta * T / t; guard t=0.","failure_scenario":"A missing t (or T) makes the temperature evolution scale-wrong; a sign flip heats the bubble as it expands.","repro":"Round-trip: dTdt2delta(t,T,delta2dTdt(t,T,d)) == d to machine precision.","confidence":"high"},
  {"id":"S7-C-32","file":"trinity/bubble_structure/get_bubbleParams.py","line":47,"class":"coefficient","severity":"S2","claim":"dTdt2delta must be the exact inverse of delta2dTdt: delta = t*(dT/dt)/T.","evidence":"Same definition; the pair is used to move between the implicit solver's variables and the physical rate.","expected":"exact algebraic inverse, machine-precision round trip.","failure_scenario":"An asymmetric pair introduces a systematic drift in delta each iteration of the implicit solver, corrupting the conduction closure delta=(2/7)(2 alpha - beta - 1).","repro":"pytest round-trip over a log-spaced grid of (t,T,delta).","confidence":"high"},
  {"id":"S7-C-33","file":"trinity/bubble_structure/get_bubbleParams.py","line":69,"class":"sign","severity":"S1","claim":"cool_beta_to_Ebdot must use Pdot_b = -beta*P_b/t (note the minus in beta's definition) and Ebdot = [Pdot_b V_b + P_b Vdot_b]/(gamma-1) with V_b = (4 pi/3)(R2^3 - R1^3), Vdot_b = 4 pi (R2^2 v2 - R1^2 Rdot1).","evidence":"E_b = P_b V_b/(gamma-1) differentiated; beta = -dlnP_b/dlnt (SPEC-041), so beta=4/5>0 corresponds to a DECAYING pressure.","expected":"the minus sign present; the same V_b as bubble_E2P (SPEC-035 trap i).","failure_scenario":"Dropping the minus reverses the sign of the pressure-decay contribution to Ebdot, which is the term that drives the bubble toward the transition; the bubble then gains energy as it depressurises.","repro":"Round-trip against Ebdot_to_cool_beta.","confidence":"high"},
  {"id":"S7-C-34","file":"trinity/bubble_structure/get_bubbleParams.py","line":140,"class":"coefficient","severity":"S2","claim":"Ebdot_to_cool_beta must be the exact inverse of cool_beta_to_Ebdot: beta = -t[(gamma-1)Ebdot - P_b Vdot_b]/(P_b V_b).","evidence":"Algebraic inversion of the previous item; the signature carries r1 explicitly, so V_b must include the R1^3 subtraction consistently.","expected":"machine-precision round trip for arbitrary (P_b, r1, Ebdot).","failure_scenario":"An inconsistent pair makes the implicit (alpha,beta,delta) solve chase a moving target; the phase-transition criterion, which is evaluated from these rates, becomes iteration-dependent.","repro":"","confidence":"high"},
  {"id":"S7-C-35","file":"trinity/bubble_structure/get_bubbleParams.py","line":198,"class":"coefficient","severity":"S1","claim":"bubble_E2P must return P_b = 3(gamma-1)E_b/(4 pi (r2^3 - r1^3)) = E_b/(2 pi (r2^3-r1^3)) at gamma=5/3, using gamma as (gamma-1) and retaining r1.","evidence":"SPEC-024, derived from E = PV/(gamma-1) with V=(4pi/3)(r2^3-r1^3).","expected":"gamma appears as (gamma-1); r1^3 retained (it is NOT small once the bubble deflates and R1->R2).","failure_scenario":"Dropping r1 over-states V_b and under-states P_b, breaking the identity P_b = L_w/(2 pi R1^2 v_w) that makes the energy->momentum handover continuous; using gamma instead of (gamma-1) is a 2.5x pressure error.","repro":"Check P_b from E2P equals L_w/(2 pi R1^2 v_w) with the solved R1, to round-off.","confidence":"high"},
  {"id":"S7-C-36","file":"trinity/bubble_structure/get_bubbleParams.py","line":242,"class":"sign","severity":"S1","claim":"get_leak_luminosity must vanish identically at coverFraction = 1 (the sealed, Weaver bubble) and scale as (1 - coverFraction).","evidence":"SPEC-036: coverFraction is the CLOSED fraction and C_f=1.0 is the default that 'recovers the sealed (Weaver) bubble exactly' (test T14 requires bit-identity).","expected":"L_leak = (1-C_f) * 4 pi R2^2 * c_s * P_b * gamma/(gamma-1); L_leak(C_f=1) == 0.0 exactly.","failure_scenario":"If the open/closed convention is inverted, the DEFAULT run leaks at maximum rate: every fiducial result, including all published figures, carries a spurious loss term of order (5/2)P_b c_s 4 pi R2^2.","repro":"Assert get_leak_luminosity(1.0, ...) == 0.0 and that a C_f=1 run is bit-identical to the sealed reference.","confidence":"high"},
  {"id":"S7-C-37","file":"trinity/bubble_structure/get_bubbleParams.py","line":242,"class":"coefficient","severity":"S3","claim":"The vent flux should carry ENTHALPY, gamma/(gamma-1) P_b = 2.5 P_b at gamma=5/3, not internal energy 1/(gamma-1) P_b = 1.5 P_b; and c_s must be the bubble-interior sound speed sqrt(gamma P_b/rho_b).","evidence":"SPEC-036 marks this AMBIGUOUS; a freely venting gas does PdV work on what it displaces, so enthalpy is the physically correct flux. A choked sonic orifice would further reduce it by ~0.5 (rho*/rho0=0.65, c*/c0=0.87 at gamma=5/3).","expected":"2.5 P_b c_s A, documented; c_s in the same unit system as R2 and P_b.","failure_scenario":"A 40% error in the leak rate for every run with C_f<1; if c_s arrives in km/s while R2,P_b are in pc/Myr units, a further silent 2.3% (or 1e5, if unconverted).","repro":"","confidence":"medium"},
  {"id":"S7-C-38","file":"trinity/bubble_structure/get_bubbleParams.py","line":286,"class":"coefficient","severity":"S1","claim":"pRam must return pdot/(4 pi r^2) = L_mech/(2 pi r^2 v_mech), using pdot = 2 L/v.","evidence":"rho_w v_w^2 = [Mdot/(4 pi r^2 v)] v^2 = Mdot v/(4 pi r^2) = pdot/(4 pi r^2); and L=(1/2)Mdot v^2, pdot=Mdot v give pdot=2L/v (SPEC-071). Dimensional check in AU: (Msun pc^2 Myr^-3)/(pc^2 * pc/Myr) = Msun pc^-1 Myr^-2 = pressure.","expected":"P_ram = L/(2 pi r^2 v); positive; ∝ r^-2; guarded at r=0. If the (3/4) strong-shock convention is used it must match get_r1/solve_R1.","failure_scenario":"A missing factor 2 (using L/(4 pi r^2 v)) halves the momentum-phase driving pressure, systematically under-predicting shell acceleration after the transition - i.e. it biases the code's headline dispersal-vs-recollapse verdict.","repro":"Check pRam(R2,...) equals pdot_total/(4 pi R2^2) computed from the SPS pdot column directly.","confidence":"high"},
  {"id":"S7-C-39","file":"trinity/bubble_structure/get_bubbleParams.py","line":311,"class":"other","severity":"S2","claim":"The effective bubble pressure must be continuous across the phase handover; with the ram-balance R1 and V_b=(4pi/3)(R2^3-R1^3) it is automatically so, and satisfies P_b >= P_ram(R2) identically with equality as R1->R2.","evidence":"Derived: P_b = L_w/(2 pi R1^2 v_w) and P_ram(R2) = L_w/(2 pi R2^2 v_w), so P_b/P_ram(R2) = (R2/R1)^2 >= 1; as E_b->0, R1->R2 and P_b -> pdot/(4 pi R2^2) continuously.","expected":"no jump in P_drive (hence none in dv2/dt) at the energy->transition->momentum boundaries (SPEC-016, test T13); any max(P_b, P_ram) is a no-op under this invariant.","failure_scenario":"A hard switch introduces a (R2/R1)^2 downward jump in the driving pressure, an unphysical impulse in the shell EOM and LSODA chatter at the boundary; alternatively, if max() is ever seen to select P_ram, either R1 is not the ram-balance radius or V_b dropped R1.","repro":"Sample dv2/dt on both sides of each phase boundary; sample P_b/P_ram(R2) and confirm it never drops below 1.","confidence":"high"},
  {"id":"S7-C-40","file":"trinity/bubble_structure/get_bubbleParams.py","line":414,"class":"divergence","severity":"S1","claim":"solve_R1 must solve E_b R1^2 v_w = L_w (R2^3 - R1^3), whose root is unique and always bracketed by [0, R2]; it must return 0 < R1 < R2 or fail loudly.","evidence":"Derived: g(R1)=sqrt(L_w(R2^3-R1^3)/(E_b v_w)) - R1 is strictly decreasing on (0,R2) with g(0)>0, g(R2)=-R2<0. Dimensional check: [L/(E v)] = pc^-1, sqrt(pc^-1 * pc^3) = pc.","expected":"brentq on [0,R2]; guaranteed bracket; R1<R2 enforced.","failure_scenario":"A bracketing failure here can only come from a sign or unit error, but if it is caught and papered over with a fallback (e.g. R1=0 or R1=R2), P_b either loses the R1 correction entirely or diverges as V_b->0 - and the divergence sits exactly at the phase transition.","repro":"Assert 0<R1<R2 every step and that P_b from E2P matches L_w/(2 pi R1^2 v_w).","confidence":"high"},
  {"id":"S7-C-41","file":"trinity/bubble_structure/get_bubbleParams.py","line":69,"class":"other","severity":"S3","claim":"The solved (alpha, beta, delta) must satisfy delta = (2/7)(2 alpha - beta - 1) during the energy phase, returning -6/35 at the Weaver values alpha=3/5, beta=4/5.","evidence":"SPEC-042, re-derived independently here from the conduction closure T_b^{7/2} ∝ P_b R2^2/(C t). This links three otherwise independent default.param constants.","expected":"the identity holds to within the implicit solver's tolerance while cooling is a perturbation, degrading only as the bubble becomes radiative.","failure_scenario":"A systematic violation means the conduction closure and the similarity parameters have drifted apart - the interior structure being integrated is not the one the (alpha,beta,delta) triple describes.","repro":"Extract alpha,beta,delta from dictionary.jsonl and plot delta vs (2/7)(2 alpha - beta - 1).","confidence":"high"},
  {"id":"S7-C-42","file":"trinity/bubble_structure/bubble_luminosity.py","line":199,"class":"citation","severity":"S3","claim":"No Weaver+77 numerical prefactor (e.g. T_b=1.51e6 or n_b=4.02e-3 with L36^{8/35}n0^{2/35}t6^{-6/35}) should be relied on as a hard-coded constant, and no code comment citing a Weaver equation NUMBER can be verified from this container.","evidence":"SPEC-045. My own check refines it: with n_b read as a hydrogen density and P=2.3 n_H k T, the quoted pair misses the dynamical P_b/k_B = 2.6e4 K cm^-3 by 1.8x (or 1.34x with the alternative T_b=2.07e6 prefactor) rather than the 4.2x quoted - i.e. part of the discrepancy is a composition-convention artefact, but it does not close. The exponents (8/35, 2/35, -6/35, 6/35, 19/35, -22/35) ARE reliable and I re-derived them.","expected":"structural forms only: P_b=(gamma-1)E_b/V_b and T_b^{7/2} ∝ P_b R2^2/(C t), which are prefactor-free.","failure_scenario":"A hard-coded prefactor inconsistent with the run's own P_b silently breaks isobaricity and every derived density/emissivity by a factor ~2-4.","repro":"","confidence":"medium"},
  {"id":"S7-C-43","file":"trinity/bubble_structure/bubble_luminosity.py","line":199,"class":"regime","severity":"S3","claim":"The Weaver conduction solution must not be applied once the interior has gone radiative; validity requires the front's radiative loss to be small compared with the conductive enthalpy flux (5/2)(k_B/mu m_H) T Mdot through it.","evidence":"Derived: at L36=n0=t6=1 my estimates put L_front ~ 1e35 erg/s against a conductive enthalpy flux ~2e35 erg/s - only marginally self-consistent, and L_front ∝ P_b^2 ∝ n0^{6/5} so it fails at higher ambient density. Beyond that limit the front becomes a condensation front (Mdot<0) and the parameterisation inverts.","expected":"an explicit validity check, or a documented statement that the reported T0/L_cool are extrapolations once L_loss/L_gain approaches unity.","failure_scenario":"T0 and L_cool continue to be reported (and used in the transition criterion) from a solution whose own assumptions have failed - the criterion is evaluated with numbers produced by the model it is meant to invalidate.","repro":"Log L_front vs (5/2)(k_B/mu m_H) T Mdot through a run.","confidence":"medium"},
  {"id":"S7-C-44","file":"trinity/bubble_structure/bubble_luminosity.py","line":116,"class":"regime","severity":"S4","claim":"Classical (unsaturated) Spitzer conduction is assumed throughout; in the hot dilute interior the electron mean free path (lambda_e ~ 1e4 T^2/n_e cm ~ 30 pc at T=1e7 K, n_e=1e-2) is comparable to R2, so the flux is formally saturated there, and magnetic suppression is absent.","evidence":"Cowie & McKee saturation parameter; SPEC-044 regime note. Both effects reduce the true heat flux, hence the evaporation rate.","expected":"documented as a model limitation; the code sets the MAXIMUM possible evaporation rate.","failure_scenario":"Over-estimated evaporation -> over-dense, cooler bubble -> over-estimated L_cool from the bulk (though the front, which dominates, is safely unsaturated). Direction of the bias should be stated rather than left implicit.","repro":"","confidence":"medium"}
]
```
