# S4 phase1 energy — Lens C (what it should be)

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

**Slice.** Phase 1, the energy-driven (Weaver) phase: `trinity/phase1_energy/energy_phase_ODEs.py`
and `trinity/phase1_energy/run_energy_phase.py`.

**Method.** Derived from first principles plus internal knowledge of Weaver+77 / Rahner+17
(WARPFIELD) / standard ISM theory, using only the redacted signature list and
`docs/dev/code-audit/reference/PHYSICS_SPEC.md`. **No implementation file, comment, or docstring was
read; no code was run.** Literature fetch is blocked (§0.3 of the spec), so every numerical
coefficient below is flagged as either *re-derived here* (high confidence) or *recalled from the
literature* (medium/low). Every arithmetic constant quoted was recomputed in this session.

**Interface I am reasoning against** (the only thing I know about the code):

```
energy_phase_ODEs.py
  L30  _scalar(x)
  L36  get_press_ion(r, params)
  L59  ODESnapshot                       (frozen state closed over by the RHS)
  L114 create_ODE_snapshot(params, shell_props) -> ODESnapshot
  L168 get_ODE_Edot_pure(t, y, snapshot, params_for_feedback)     <- the RHS
  L289 ODEResult
  L325 compute_derived_quantities(t, y, snapshot, params_for_feedback) -> ODEResult
run_energy_phase.py
  L54-59 TFINAL_ENERGY_PHASE, SEGMENT_DURATION, DT_EXIT_THRESHOLD,
         COOLING_UPDATE_INTERVAL, RTOL, ATOL
  L62  run_energy(params)
  L296 ode_func(t, y)
```

The name `get_ODE_Edot_pure(t, y, ...)` returning into `ode_func(t, y)` fixes the shape of the
problem: a first-order system `dy/dt = f(t, y)` with a side-effect-free RHS closed over a
"snapshot". The minimal physically complete state for an energy-driven Weaver bubble is
`y = (R2, v2, E_b)`; a fourth slaved component `T0` (bubble temperature at the diagnostic radius
`ξ·R2`) is common in WARPFIELD-lineage codes and is discussed in §5.6. The presence of
`SEGMENT_DURATION` + `COOLING_UPDATE_INTERVAL` implies the RHS is **not** a pure function of
`(t, y)`: some coefficients are frozen and refreshed on a schedule. That is a modelling
approximation with a quantifiable error (§7).

---

## 1. The coupled ODE system

### 1.1 Geometry and definitions

Four zones (SPEC-002): free wind `r<R1`; shocked wind / hot bubble `R1<r<R2` at near-uniform
pressure `P_b`; thin shell at `R2`; undisturbed cloud/ISM outside. `R2` is the contact
discontinuity and *is* the shell's inner face, so `v2 ≡ dR2/dt` is simultaneously the CD speed and
the shell speed. `M_sh(R2) = M_enc(R2)` (SPEC-021).

### 1.2 Momentum equation — canonical (conservative) form

For a thin spherical shell accreting ambient gas **that is at rest**, Newton's second law applied to
the shell as an open control mass is

```
    d/dt [ M_sh(R2) · v2 ]  =  Σ F_ext                                       (M1)
    Ṁ_sh = 4π R2² ρ_amb(R2) · max(v2, 0)
```

Expanding the product and using `Ṁ_sh v2 = 4πR2² ρ_amb(R2) v2²`:

```
    M_sh dv2/dt = Σ F_ext − 4π R2² ρ_amb(R2) v2²                             (M2)
```

**(M1) and (M2) are the same equation.** The `−4πR2²ρ_amb v2²` term in (M2) is *the ram pressure of
the ambient medium in the shell frame* and *the momentum flux of newly swept material* — one
quantity, two names. Writing (M1) and additionally subtracting a ram term is the single most common
factor-in-the-deceleration bug in this model class (SPEC-020 AUDIT TRAP).

### 1.3 Term-by-term force budget, with signs

Working in the (M2) form, `Σ F_ext = F_b + F_HII + F_rad − F_grav − F_ext,P`:

| # | Term | Expression | Sign | Notes |
|---|---|---|---|---|
| 1 | interior thermal pressure | `+ 4π R2² P_b` | **outward (+)** | `P_b` from the closure §1.5 |
| 2 | photoionized-gas pressure | `+ 4π R2² P_HII` | **outward (+)** | TRINITY combines 1 and 2 as `P_drive = max(P_b, P_HII)` in the energy phase (SPEC-022) — see §6.3 |
| 3 | direct radiation pressure | `+ (L_bol/c) · f_abs`, `f_abs = 1 − e^{−τ_UV} ∈ [0,1]` | **outward (+)** | single-scattering limit `→ L_bol/c` |
| 4 | IR-trapped radiation pressure | `+ (L_bol/c) · f_abs · τ_IR`, `τ_IR = κ_IR M_sh /(4π R2²)` | **outward (+)** | **must not re-add term 3** — see §8.1 |
| 5 | gravity, central cluster | `− G M_cluster M_sh / R2²` | **inward (−)** | point mass at origin |
| 6 | gravity, shell self-gravity | `− G M_sh² /(2 R2²)` | **inward (−)** | factor **½**, derived §1.4 |
| 7 | ambient thermal pressure | `− 4π R2² P_ISM` | **inward (−)** | `PISM` is declared as `P/k_B` in K cm⁻³ |
| 8 | ambient turbulent pressure | `− 4π R2² ρ_amb(R2) σ_turb²` | **inward (−)** | only if the model carries a turbulent ambient; formally identical in structure to 7 |
| 9 | swept-material momentum flux | `− 4π R2² ρ_amb(R2) v2²` | **inward (−)** | **absent** in the (M1) form; present exactly once in (M2) |

So, fully written out, what the RHS must be:

```
 dR2/dt = v2

 dv2/dt = { 4πR2² [ P_drive − P_ISM − ρ_amb(R2) v2² ]
            + (L_bol/c) f_abs (1 + τ_IR)
            − G M_sh ( M_cluster + M_sh/2 ) / R2²  }  /  M_sh              (M3)
```

**Two terms that must NOT appear during the energy phase:**

- the bare wind/SN momentum flux `ṗ_w + ṗ_SN` (or `P_ram = (ṗ_w+ṗ_SN)/(4πR2²)`). In the
  energy-driven limit the wind momentum is thermalised at `R1` and *is* the source of `P_b`
  (`4πR1²P_b = ṗ_w` is exactly the `R1` condition, SPEC-025). Adding both `4πR2²P_b` and `ṗ_w`
  double-counts the same momentum. This is consistent with SPEC-022, which puts `P_ram` only in the
  transition and momentum phases.
- any explicit "shell inertia from the wind mass" term: the wind mass is in the bubble, not the
  shell.

### 1.4 Self-gravity factor — full derivation (this is the classic factor-2)

The gravitational field of a thin shell of mass `M_sh` at radius `R` is `GM_sh/R²` just inside and
`0` just outside. A mass element *of the shell itself* sits on the discontinuity and feels the mean,
`GM_sh/(2R²)`. Total self-force `= M_sh · GM_sh/(2R²) = GM_sh²/(2R²)`, inward. Equivalently, the
self-energy of a thin shell is `U = −GM_sh²/(2R)` and `F = −∂U/∂R|_{M fixed} = −GM_sh²/(2R²)`.
Hence

```
    F_grav = G M_sh ( M_cluster + M_sh/2 ) / R2²                            (G1)
```

Using `M_sh` instead of `M_sh/2` overestimates self-gravity by a factor 2 (and the *total*
gravity by up to 2× once `M_sh ≫ M_cluster`, which is the regime of every run with `ε ≲ 0.3`).
Cloud gas **exterior** to `R2` contributes exactly zero (Newton's shell theorem) and must not be
in the enclosed mass; using `M_cloud` instead of `M_sh(R2)` is a strictly-too-large gravity at all
`R2 < r_cloud`.

### 1.5 Energy equation

Treat the bubble `R1<r<R2` as an open control volume of volume `V_b`, internal energy `E_b`:

```
    dE_b/dt = L_gain − P_b (dV_b/dt) − L_cool − L_leak                       (E1)

    L_gain  = η_w L_mech,w(t) + η_SN L_mech,SN(t)          (thermalisation efficiencies)
    dV_b/dt = 4π ( R2² v2 − R1² dR1/dt )
    L_cool  = ∫_{R1}^{R2} 4π r² n_e(r) n_H(r) Λ(T(r)) dr   (interior + conduction front)
    L_leak  = (1 − C_f) · 4π R2² · c_s · [γ/(γ−1)] P_b     (venting, enthalpy flux)
```

Signs: `L_gain > 0` (source), `P_b dV_b/dt > 0` while expanding (the bubble does work on the shell
and on the wind-shock boundary — a **loss** to `E_b`), `L_cool ≥ 0` (loss), `L_leak ≥ 0` (loss).

Consistency requirement on the inner boundary term: `−P_b·4πR1²Ṙ1` is a *gain* (the free wind
compresses the CV when `R1` grows). Its magnitude relative to the outer term is `(R1/R2)²(Ṙ1/v2)`;
with `R1² = ṗ_w/(4πP_b)` and the Weaver `P_b` (§4.2) this is `(R1/R2)² = 2.2 R2/(v_w t)` ≈ 0.028
(i.e. `R1/R2 ≈ 0.17`) at `R2 = 26 pc`, `t = 1 Myr`, `v_w = 2000 km s⁻¹` — small early, growing as
`P_b` falls late in the phase.

**The `PdV` term must use the same `V_b` as the closure** (§1.6). Dropping `R1³` in one and keeping
it in the other is a manufactured energy source/sink of relative size `3(R1/R2)³` in `P_b` (0.5% at
`R1/R2=0.17`, rising toward the transition).

### 1.6 The closure (interior pressure ↔ energy ↔ radius)

Ideal gas, adiabatic index `γ`, uniform interior pressure:

```
    E_b = P_b V_b / (γ − 1)    ⇒    P_b = (γ−1) E_b / V_b
    V_b = (4π/3) ( R2³ − R1³ )
    ⇒   P_b = 3(γ−1) E_b / [ 4π ( R2³ − R1³ ) ]                             (C1)
    γ = 5/3 :  P_b = E_b / [ 2π ( R2³ − R1³ ) ]                             (C2)
    γ = 5/3, R1 ≪ R2 (Weaver's own approximation) :  P_b = E_b / (2π R2³)   (C3)
```

`(C3)` is a **γ=5/3-only** expression with `R1` dropped: the literal constant `2π` encodes both
choices simultaneously. If the code exposes a `gamma_adia` parameter (SPEC-024 says `default.param`
declares one), a hard-coded `2π`/`1.5` silently ignores it.

Wind termination shock:

```
    R1 = sqrt( ṗ_w / (4π P_b) )                    (Weaver convention, pre-shock ram = P_b)
       = sqrt( 3 ṗ_w / (16π P_b) )                 (strict strong-shock post-shock pressure)
```
differing by `√3/2 = 0.866`. Either is defensible (SPEC-025); the code must be self-consistent and
must enforce `0 < R1 < R2`.

### 1.7 Photoionized-gas pressure — what `get_press_ion(r, params)` must compute

The signature takes a single radius. The standard (Krumholz–Matzner-style) prescription for the
pressure of ionized gas filling a sphere of radius `r` in Strömgren balance is

```
    Q_i = (4π/3) α_B χ_e n_H² r³
    ⇒  n_H(r) = [ 3 Q_i / (4π α_B χ_e r³) ]^{1/2}
    ⇒  P_HII(r) = n_tot k_B T_ion = (1 + x_He + χ_e) n_H(r) k_B T_ion
               ∝ Q_i^{1/2} · r^{−3/2}                                        (P1)
```

with `n_tot/n_H = 2.2` for `x_He = 0.1`, `χ_e = 1.1` (SPEC-029). **Testable structure independent
of every prefactor:** `P_HII ∝ Q_i^{1/2} R2^{−3/2}` and `P_HII` must be a decreasing function of
`r`. Using `n_H k_B T` (no `n_tot` factor) is a 2.2× under-estimate; using `2 n_H k_B T`
(pure-H) is a 10% under-estimate.

**Physical caveat (SPEC-030 is genuinely open).** During the energy phase the volume `r<R2` is
filled with `10⁶–10⁷ K` shocked wind at `n ~ 10⁻² cm⁻³`, whose recombination rate is negligible, so
the *actual* ionized gas is a thin skin on the shell's inner face at much higher density and
therefore much higher pressure. `(P1)` is the "no bubble" classical HII-region pressure. Which
reading the code takes is the highest-value question in this slice and it changes `P_drive`
qualitatively.

---

## 2. Dimensions of every term

### 2.1 Unit system

Conventional for this code class, and asserted by SPEC-090/091: **internal working units are
`[M⊙, pc, Myr]`** ("AU"); inputs are cgs-extended (`cm⁻³`, `K cm⁻³`, `km/s`, `pc`, `M⊙`); the
external tables (SPS, cooling, opacities) are cgs.

| Quantity | AU dimension | cgs value of 1 AU unit (recomputed here) |
|---|---|---|
| `R2`, `R1` | pc | `3.085678e18 cm` |
| `v2` | pc Myr⁻¹ | `9.7778e4 cm s⁻¹ = 0.97778 km s⁻¹` |
| `M_sh`, `M_cluster` | M⊙ | `1.98892e33 g` |
| `E_b` | M⊙ pc² Myr⁻² | `1.9014e43 erg` |
| `L_mech`, `L_cool`, `L_bol` | M⊙ pc² Myr⁻³ | `6.0252e29 erg s⁻¹` |
| `P_b`, `P_HII`, `P_ISM` | M⊙ pc⁻¹ Myr⁻² | `6.4723e-13 dyn cm⁻²` |
| `F_*`, `ṗ_w` | M⊙ pc Myr⁻² | `6.1623e24 dyn` |
| `ρ_amb` | M⊙ pc⁻³ | `6.7696e-23 g cm⁻³` |
| `dv2/dt` | pc Myr⁻² | — |
| `dE_b/dt` | M⊙ pc² Myr⁻³ | — |

Dimension checks that must hold term by term in (M3):
`[4πR2²P] = pc²·M⊙ pc⁻¹Myr⁻² = M⊙ pc Myr⁻²` ✓;
`[4πR2²ρv²] = pc²·M⊙pc⁻³·pc²Myr⁻² = M⊙ pc Myr⁻²` ✓;
`[L/c] = (M⊙pc²Myr⁻³)/(pc Myr⁻¹) = M⊙ pc Myr⁻²` ✓;
`[G M²/R²] = (pc³M⊙⁻¹Myr⁻²)(M⊙²)(pc⁻²) = M⊙ pc Myr⁻²` ✓.

### 2.2 Mandatory conversion boundaries (each is a place a factor can be lost)

Recomputed constants, for anchoring:

1. **SPS table → AU.** `L_mech [erg/s] / 6.0252e29`; `ṗ_w [dyn] / 6.1623e24`; `Q_i [s⁻¹] × 3.15576e13`.
2. **`c`** must be `3.06601e5 pc Myr⁻¹` wherever `L_bol/c` is formed.
3. **`G` = `4.49966e-3 pc³ M⊙⁻¹ Myr⁻²`** (`= 4.30091e-3 pc M⊙⁻¹ (km/s)²`).
4. **`k_B` = `7.2606e-60 M⊙ pc² Myr⁻² K⁻¹`** wherever `P = n k T` is formed.
5. **`n_H → ρ`**: `ρ = μ_H m_H n_H`, `μ_H = 1.4` (mass per H nucleus, ionisation-independent).
   `n_H = 1 cm⁻³ ⇒ ρ = 0.034613 M⊙ pc⁻³`. **This `μ` is a different constant from the `μ` per
   particle used in `P = ρkT/(μ m_H)`** (SPEC-092.1) — the two must never be interchanged.
6. **`P_ISM`**: input is `P/k_B` in `K cm⁻³` → multiply by `k_B` *and* convert `cm⁻³ → pc⁻³`
   (`1 cm⁻³ = 2.9380e55 pc⁻³`).
7. **`κ_IR = 4 cm² g⁻¹` → `8.3556e-4 pc² M⊙⁻¹`**; `σ_d = 1.5e-21 cm² → 1.5754e-58 pc²` (per H
   nucleus, *not* per gram — mixing the two is a ~10²³ error, which at least fails loudly).
8. **Cooling table `Λ [erg cm³ s⁻¹]`** → AU, *together with* the density product convention
   (`n_e n_H` vs `n_e n_ion` vs `n²`) — SPEC-082; a factor 1.2–4.4 sits here and it feeds the phase
   trigger directly.
9. **Spitzer `C_thermal = 6e-7 erg s⁻¹ cm⁻¹ K⁻⁷ᐟ²`** if the conduction closure is evaluated in AU.
10. **`v` reporting**: `pc/Myr` vs `km/s` differ by only 2.3% — small enough to hide.

---

## 3. Expected asymptotics

### 3.1 Uniform medium, constant `L_w` — full derivation

Assume no gravity, no radiation, no external pressure, `ρ = ρ₀` const, `R(0)=0`. Put `R = A t^η`.

Momentum (M1): `M v = (4π/3)ρ₀A⁴η t^{4η−1}`; `d(Mv)/dt = 4πR²P_b` ⇒
```
    P_b = (ρ₀/3) η(4η−1) A² t^{2η−2}
```
Closure (C3): `E_b = 2π P_b R³ = (2π/3) η(4η−1) ρ₀ A⁵ t^{5η−2}`.
Constant `L_w` requires `E_b ∝ t` ⇒ `5η − 2 = 1` ⇒ **`η = 3/5`**.
Then `dE_b/dt = 1.75929 ρ₀A⁵` and `4πR²v P_b = (4π/3)η²(4η−1)ρ₀A⁵ = 2.11115 ρ₀A⁵`, so
```
    L_w = 3.87045 ρ₀ A⁵   ⇒   A⁵ = 0.2583684 · L_w/ρ₀ = (250/308π) L_w/ρ₀
```

```
    R2(t) = 0.762865 (L_w/ρ₀)^{1/5} t^{3/5}        v2(t) = (3/5)R2/t = 0.457719 (L_w/ρ₀)^{1/5} t^{−2/5}
```

> **Arithmetic note against the spec.** SPEC-050 quotes `ξ_E = 0.762934` and `v`-prefactor
> `0.457760`. Recomputing `(250/308π)^{1/5}` gives **0.7628653** (check: `0.7628653⁵ = 0.2583684`
> exactly matches `250/308π`), so the spec's 5th–6th digits are slightly off. Immaterial physically,
> but if the code hard-codes `0.762934` it inherited a rounding, not a derivation.

**Derived numeric anchors** (`L₃₆ = 10³⁶ erg/s`, `n_H = 1 cm⁻³`, `t = 1 Myr`), recomputed here:

| `μ` convention | `R2` | `v2` |
|---|---|---|
| `ρ₀ = 1.0 n_H m_H` (Weaver's own — the famous "28 pc") | **28.04 pc** | 16.45 km/s |
| `ρ₀ = 1.4 n_H m_H` (mass per H nucleus) | **26.22 pc** | 15.38 km/s |

Any validation test asserting 28 pc against a `μ_H = 1.4` code is 7% wrong in radius (30% in swept
mass) and would *pass* a code carrying a compensating `μ` bug.

### 3.2 Energy partition (dimensionless — immune to unit bugs)

```
    E_b      / (L_w t) = 1.75929/3.87045 = 5/11  = 0.454545
    E_kin,sh / (L_w t) = 0.75398/3.87045 = 15/77 = 0.194805
    radiated at the outer (shell) shock  = 27/77 = 0.350649
    P_b = 5 L_w t /(22π R2³) = 0.162962 L_w^{2/5} ρ₀^{3/5} t^{−4/5}
    α ≡ v2 t/R2 → 3/5 ;  β ≡ −d ln P_b/d ln t → 4/5 ;  δ ≡ d ln T/d ln t → −6/35
```

### 3.3 Power-law ambient `ρ ∝ r^{−w}` (`w = |α| ∈ [0,2]`; TRINITY's `densPL_alpha = −w`)

`M_sh = B R^{3−w}`, `B = 4πρ_ref r_ref^w/(3−w)`. Repeating §3.1 with this `M(R)`:

```
    P_b   = (B/4π) A^{2−w} η[(4−w)η−1] t^{(2−w)η−2}
    E_b   = (B/2)  A^{5−w} η[(4−w)η−1] t^{(5−w)η−2}
    E_b ∝ t ⇒  η = 3/(5 − w)                                          ← radius exponent
    v2 ∝ t^{η−1} = t^{(w−2)/(5−w)}                                     ← velocity exponent
    L_w   = B A^{5−w} η[(4−w)η−1](½+η)
    ⇒ A^{5−w} = L_w(3−w) / { 4π ρ_ref r_ref^w · η[(4−w)η−1](½+η) }     ← prefactor
    E_b/(L_w t) = 1/(1 + 2η)
```

| `w` | `η = d ln R/d ln t` | `v2` exponent | `E_b/(L_w t)` |
|---|---|---|---|
| 0 | 3/5 = 0.600 | **−2/5** | 5/11 = 0.4545 |
| 1 | 3/4 = 0.750 | −1/4 | 2/5 = 0.400 |
| 3/2 | 6/7 = 0.857 | −1/7 | 7/19 = 0.3684 |
| 2 | 1 (linear) | **0** (constant `v`) | 1/3 |

Checks: `w=0` reproduces §3.1 exactly; `w=2` gives equipartition `E_b = E_kin = radiated = 1/3` and
constant expansion speed. The exponent `3/(5−w)` matches the published figure script's
`exp_weaver = 3/(5−|α|)` (SPEC-053).

**The prefactor is *not* `0.762865`** for `w ≠ 0` — applying the uniform-medium coefficient to a
power-law profile is a named trap (§8.4). Also note the profile is flat inside `r_core` and jumps to
`n_ISM` at `r_cloud`, so the power-law similarity solution is valid only for
`r_core ≪ R2 < r_cloud`.

---

## 4. Exact invariants the implementation must satisfy

### 4.1 Energy ledger (the strongest global check)

Multiply (M2) by `v2` and add (E1). With `E_kin = ½M_sh v2²`,
`dE_kin/dt = M_sh v2 dv2/dt + ½Ṁ_sh v2² = v2 ΣF_ext − ½Ṁ_sh v2²`:

```
 d/dt (E_b + E_kin) = L_gain − L_cool − L_leak
                      − ½ Ṁ_sh v2²                                (shock dissipation → radiated)
                      − F_grav v2 − 4πR2²P_ISM v2                 (work against gravity/ambient)
                      + (L_bol/c) f_abs(1+τ_IR) v2                (radiation work in)
                      + 4πR2² ( P_drive − P_b ) v2                (★ MUST BE ZERO)             (I1)
```

The starred term is identically zero **iff `P_drive = P_b`**. With `P_drive = max(P_b, P_HII)`
(SPEC-022/023), whenever the `P_HII` branch wins the shell receives `4πR2²P_HII v2` of work while
the bubble is only debited `4πR2²P_b v2` — energy is created out of nothing at rate
`4πR2²(P_HII − P_b)v2`. A correct implementation must either (i) debit the same `P_drive` on the
bubble side, (ii) supply the excess from the ionizing-photon energy budget explicitly, or
(iii) document the non-conservation and bound it. **This is a first-class invariant, not a
nicety.**

Integrated form (the run-level check): `∫L_gain dt = E_b + E_kin + ∫L_cool dt + ∫L_leak dt +
∫½Ṁ_sh v2² dt + ∫F_grav v2 dt + ∫4πR2²P_ISM v2 dt − ∫(L_bol/c)f_abs(1+τ_IR) v2 dt`.

### 4.2 Force-budget closure

`M_sh dv2/dt` recomputed from the reported `F_grav, F_rad, F_ram, F_HII, F_drive, 4πR2²P_ISM` must
equal the integrator's `dv2/dt` to integrator tolerance at *every* snapshot (SPEC-007). This
requires that `compute_derived_quantities` (L325) reports **the same numbers the RHS used**, not an
independent re-derivation. Two code paths computing "the same" force is how the budget silently
stops closing.

### 4.3 Positivity and domain

- `E_b > 0`, `P_b > 0`, `V_b > 0`, `0 < R1 < R2`, `M_sh > 0`, `ρ_amb > 0`, `T > 0`.
- `E_b ≤ 0` during the energy phase is a **violation of the phase assumption**, not a number to
  clip. It must terminate the phase with a distinct, recorded reason. Clamping `E_b` to a floor, or
  taking `|E_b|`, silently converts a physical event into a fictitious continued expansion.
- `R1 → R2` (`V_b → 0`) makes `P_b` diverge; must be guarded and must be a recorded exit, not a
  NaN.

### 4.4 Kinematic identity

`f[0] = dR2/dt` must be **exactly** `y[1]`. Recomputing `v2` from anything else inside the RHS
(e.g. from a similarity relation, or from `α R2/t`) breaks the definition of the state and makes
`R2` and `v2` inconsistent.

### 4.5 Continuity across the phase boundary

`R2` and `v2` are the outcome of a second-order ODE with bounded forces; they are `C⁰` and `C¹` in
time by construction, so the handover to the next phase must carry `(R2, v2)` **unchanged**, and
`dv2/dt` must not jump by more than the change in `P_drive` that the phase definition itself
mandates (SPEC-016). `E_b` may legitimately be re-initialised only if the transition is *defined* as
a depressurisation, and then it must be documented. If the exit is detected at a step boundary
instead of an event root, the reported exit state is off the event surface by `O(Δt·v2)`, and *that*
is the usual source of the discontinuity.

### 4.6 Asymptotic invariants (executable)

- `α = v2 t/R2 → 3/5` (uniform) or `3/(5−w)` in the pure-energy limit.
- `E_b/(L_mech t) → 1/(1+2η)`.
- interior isobaricity: `n(r)T(r)` constant across the stored bubble profile.
- conduction closure: `δ = (2/7)(2α − β − 1)` (SPEC-042), which with `α=3/5, β=4/5` gives exactly
  `−6/35`. This links three otherwise independent quantities and is prefactor-free.

---

## 5. Numerical structure the module constants imply

### 5.1 `RTOL`/`ATOL` across a state spanning ~10 decades

In AU, `R2 ~ 10⁰–10² pc`, `v2 ~ 10⁰–10² pc/Myr`, and `E_b ~ 10⁷–10¹⁰ M⊙ pc² Myr⁻²`
(`10⁵¹ erg = 5.26e7 AU`; a `10⁶ M⊙` cluster at `L_w ~ 10⁴⁰ erg/s` accumulates `~1.6e10` per Myr).
A **scalar** `ATOL` therefore cannot be right for all components: it is either negligible for `E_b`
(harmless) or, if sized for `E_b`, catastrophically loose for `R2`/`v2`. `ATOL` should be a
per-component vector, or `E_b` should be integrated in a rescaled variable. This matters most at
`v2 → 0` (recollapse detection), where `ATOL` alone controls the error.

### 5.2 `SEGMENT_DURATION` / `COOLING_UPDATE_INTERVAL` — frozen-coefficient error

If `L_cool` (or the whole bubble structure) is refreshed only every `COOLING_UPDATE_INTERVAL`, the
RHS is stale within a segment. In the Weaver limit, `n_b ∝ t^{−22/35}`, `T_b ∝ t^{−6/35}`,
`V_b ∝ t^{9/5}`, so

```
    L_cool ~ n_b² Λ(T_b) V_b  ∝  t^{19/35} Λ(T_b(t))
```

i.e. `L_cool` varies on the *current age* timescale: `|d ln L_cool/d ln t| ≳ 0.54`. A **fixed
absolute** update interval `Δt` therefore produces a fractional staleness error `~0.54 Δt/t`, which
is negligible at late times and **arbitrarily large at early times**. The physically correct
schedule is logarithmic (`Δt ∝ t`, or `Δt ≤ f·min(t, R2/v2, E_b/|Ė_b|)`). Same argument for
`SEGMENT_DURATION` and for anything `R2`-dependent frozen in `ODESnapshot` (`f_abs`, `τ_IR`,
`M_sh`, `P_HII` all depend on `R2`, which changes by `2^{3/5}=1.52×` per doubling of `t`).

Corollary: the exit time cannot be resolved better than `COOLING_UPDATE_INTERVAL`, because the
quantity the exit test is built on is only known on that grid.

### 5.3 `DT_EXIT_THRESHOLD` — a numerical bail-out must not masquerade as physics

A threshold on the accepted step size can only be a stiffness/stall detector. Exiting on it is
legitimate engineering, but the recorded outcome must be a **distinct, non-physical** termination
reason. If a step-size collapse is reported as "energy→momentum transition", the code's headline
prediction (transition time) becomes a function of the tolerance settings.

### 5.4 `TFINAL_ENERGY_PHASE`

A hard cap on energy-phase duration is a run limit, not physics (SPEC-100 "numerical cutoff").
Hitting it must be recorded distinctly and must be larger than any physically reachable transition
time for the shipped configs, otherwise it silently truncates the phase.

### 5.5 Sources of RHS non-smoothness that an adaptive solver will fight

The RHS must be `C¹` in `(t, y)` except at explicitly handled events. Known kink sources:
1. `max(P_b, P_HII)` — non-differentiable at the crossing (SPEC-023).
2. `ρ_amb(r)` discontinuous at `r = r_cloud` (cloud edge → `n_ISM`) and kinked at `r = r_core`.
3. Linearly-interpolated SPS drivers `L_w(t)`, `ṗ_w(t)` — kinked at every table node.
4. Age-indexed cooling *files* rather than age interpolation → piecewise-constant in cluster age
   (SPEC-083).
5. The `COOLING_UPDATE_INTERVAL` refresh itself.
Each should be either an integrator event with restart, or smoothed; otherwise the step controller
repeatedly rejects steps at the kink and the effective accuracy is not the requested `RTOL`.

### 5.6 The fourth state component, if present

If `y` carries a bubble temperature `T0`, it is **not** an independent dynamical degree of freedom:
`T0` is slaved to `(E_b, R2, v2, L_w, C_thermal)` through the conduction closure
`T_b^{7/2} = a P_b R2²/(C t)`. Integrating it as an ODE component is legitimate only if its
derivative is the total derivative of that constraint along the trajectory; otherwise the state
drifts off the constraint manifold and `T0` becomes a lagging, unconstrained variable. The checkable
signature is `δ = d ln T0/d ln t → −6/35` in the Weaver regime, equivalently
`δ = (2/7)(2α − β − 1)`.

### 5.7 `T0` is measured at `ξ = 0.98`, not at the centre

With Weaver's interior profile `T(r) = T_b (1 − r/R2)^{2/5}`, a temperature reported at
`ξ = 0.98` is `(0.02)^{2/5} = 0.20913` of the central `T_b`. Any comparison of a reported `T0`
against a Weaver `T_b` formula must apply this factor of ≈4.8.

### 5.8 Where `L_cool` actually comes from (a convergence requirement)

With `n ∝ (1−x)^{−2/5}`, `T ∝ (1−x)^{2/5}` (`x = r/R2`), the emission measure integrand goes as
`(1−x)^{−4/5}`, whose integral `∫₀¹` converges to `5` but with the **outermost sliver dominating**:
the fraction of `∫n² dr` at `x > ξ` is `(1−ξ)^{1/5}`, i.e.

| `ξ` | fraction of `∫n²dr` beyond `ξ` |
|---|---|
| 0.98 | **45.7%** |
| 0.99 | 39.8% |
| 0.999 | 25.1% |

Moreover `Λ(T)` *peaks* near `10⁵ K`, which for `T_b ~ 3×10⁶ K` occurs at
`1−x = (10⁵/3×10⁶)^{5/2} ≈ 2×10⁻⁴`, i.e. `x ≈ 0.9998` — **inside** a `ξ = 0.98` cut. So `L_cool`
is dominated by a thin conduction front adjacent to the contact discontinuity. A radial cut at
`ξ = 0.98` discards ~46% of the emission measure *and* the highest-`Λ` material, systematically
under-predicting `L_cool` and therefore over-predicting the energy-phase duration. Whatever cut is
used, `L_cool` must be demonstrably convergent with respect to it and to the grid.

---

## 6. Phase-exit criteria

### 6.1 The physically correct exits

| Exit | Physics | Correct event function `g(t,y)` (terminal, with `direction`) |
|---|---|---|
| **Catastrophic interior cooling** (energy→momentum) | the bubble stops being an energy reservoir: `L_loss/L_gain → 1` | `g = (L_gain − L_loss)/L_gain − ε`, `direction = −1`. Equivalent statements: `t_cool,b = 3n k T V/(2L_cool) < t_dyn = R2/v2`; "cooling radius reached" |
| **Bubble energy turnover** | `Ė_b ≤ 0`: the bubble is now net-losing | `g = dE_b/dt`, `direction = −1` — threshold-free and strictly later than the above |
| **Blowout / venting** | shell crosses the cloud edge into `n_ISM`; the bubble depressurises through the fragmenting shell | `g = R2 − r_cloud`, `direction = +1` |
| **Shell dispersal** | shell peak density indistinguishable from ambient | `g = n_sh,max − n_ISM`, `direction = −1`, *sustained* for `stop_t_diss` |
| **Recollapse / stall** | gravity wins | `g = v2`, `direction = −1`; recollapse confirmed if additionally `½v2² < G(M_cluster+M_sh)/R2` |
| **Feedback exhausted** | end of SPS coverage | `g = t − t_SPS,max` |
| **Structural breakdown** | `E_b ≤ 0`, `R1 ≥ R2`, `V_b ≤ 0`, non-finite state | hard guards; each a *distinct* recorded reason |

### 6.2 How a robust integrator must detect them

All of the above are **root-finding events on dense output** (`solve_ivp(..., events=[...],
dense_output=True)`), each with `terminal=True` and an explicit `direction`. Requirements:

1. The event function must be **continuous and sign-changing** through the event — not a boolean.
   A boolean/step-boundary check resolves the exit only to the step size (or, worse, to
   `COOLING_UPDATE_INTERVAL`), and the state handed to the next phase is then not on the event
   surface (§4.5).
2. `direction` must be set. Without it, an event function that grazes zero (e.g. `v2` in a stalled
   shell, or the `L_loss/L_gain` ratio oscillating around the threshold) fires spuriously.
3. Multiple simultaneous candidate events must be resolved to the **earliest root**, and the winner
   recorded.
4. The `ε = 0.05` in the cooling-balance criterion is a **numerical regularisation of "→ 0"**, not
   physics (SPEC-014). Any exit time derived from it must be reported with its sensitivity: the
   defensible threshold range is `0.01–0.2`, and the transition time varies across it.

### 6.3 Why the exit is *needed*: the energy phase's own validity condition

The energy-driven phase is defined by `4πR2²P_b ≫ ṗ_w + ṗ_SN`. When those become comparable, the
phase is over *by definition*, independent of any cooling threshold. A useful internal consistency
check: the ratio `4πR2²P_b/(ṗ_w+ṗ_SN) = (R2/R1)²` must be `≫ 1` throughout the phase; it reaching
`O(1)` is `R1 → R2`, the same event as §4.3.

---

## 7. What `ODESnapshot` / `create_ODE_snapshot` must guarantee

A well-posed ODE requires `dy/dt = f(t, y)`. A "snapshot" of frozen coefficients makes the system
piecewise-autonomous-in-coefficients. The expectations:

1. **Nothing that varies on the dynamical time may be frozen across a segment.** `M_sh(R2)`,
   `ρ_amb(R2)`, `τ_IR ∝ M_sh/R2²`, `f_abs`, `P_HII ∝ R2^{−3/2}` all depend on `R2` and change by
   tens of percent over any segment in which `R2` changes appreciably.
2. **The refresh must restart the integrator**, because the RHS is discontinuous there; carrying an
   adaptive step across the discontinuity silently violates the error estimate.
3. **The frozen approximation must be shown convergent**: halving `SEGMENT_DURATION` and
   `COOLING_UPDATE_INTERVAL` must change `R2(t)`, `v2(t)`, and the exit time by less than the
   claimed tolerance. Without that demonstration the reported `RTOL` is not the accuracy of the
   answer.
4. `_scalar` (L30) sits at a shape boundary. It must be **total** on what it receives and must not
   silently discard information: reducing a length-`n>1` array to its first element, or squeezing a
   shape mismatch away, converts a real bug into a plausible number.

---

## 8. Known traps, stated as what must be true

### 8.1 Radiation-pressure double counting
Two conventions circulate:
```
    (a)  F_rad = (L_bol/c) · f_abs · (1 + τ_IR)        [physically motivated]
    (b)  F_rad = (L_bol/c) · ( f_abs + τ_IR )          [also seen in the literature]
```
(a) is the defensible one: only the *absorbed* luminosity `f_abs L` is reprocessed into IR, and the
trapped IR contributes an extra `τ_IR` × that. (b) implicitly reprocesses the escaping fraction too.
They agree when `f_abs → 1` and differ by up to `τ_IR(1−f_abs)L/c` otherwise. The trap the brief
names is stronger: if the IR expression is written as `(1+τ_IR)f_abs L/c` and the direct term
`f_abs L/c` is then *added* to it, the direct term is counted twice. Whichever form is used,
`F_rad → L_bol/c` exactly as `τ_UV → ∞, τ_IR → 0`, and `F_rad → 0` as `τ_UV, τ_IR → 0`. Also,
`τ_IR ≫ 1` is outside the validity of a single-scattering×optical-depth estimate; RHD finds
saturation well below `τ_IR` (leakage through low-column channels).

### 8.2 `γ = 5/3`-only coefficients
`P_b = E_b/(2πR2³)` bakes in `γ = 5/3` *and* `R1 ≪ R2`. If `γ` is a parameter, the general form
`(C1)` must be used, and the *same* `V_b` must appear in the `PdV` term.

### 8.3 Sweep-up momentum flux vs ram pressure
They are the same term (§1.2). Exactly one of `d(M_sh v2)/dt` and `−4πR2²ρ_amb v2²` may appear.
Additional sub-trap: **which density**. `ρ_amb(R2)` (local) and `ρ̄ = M_sh/((4π/3)R2³)` (mean)
satisfy `ρ̄ = 3ρ(R2)/(3−w)`. They are **identical for `w = 0`** — the shipped default — and differ
by a factor **3 at `w = 2`**. A bug here is invisible in the default config and 3× wrong at the
steepest supported profile. Likewise `Ṁ_sh = 4πR2²ρ(R2)v2` must use the local density.

### 8.4 Uniform prefactor on a power-law profile
`0.762865` is `w = 0` only. The general prefactor is
`A^{5−w} = L_w(3−w)/{4πρ_ref r_ref^w η[(4−w)η−1](½+η)}` with `η = 3/(5−w)`. And any *validation
test* that anchors a power law to the simulation's own curve (as the published `radiusComparison`
figure does, SPEC-057) tests the **exponent only**, never the prefactor.

### 8.5 Gravity with the wrong enclosed mass
Must be `M_cluster + M_sh(R2)/2`, with `M_sh(R2) = M_enc(R2)`, not `M_cloud`, not
`M_cluster + M_sh`, not `M_sh/2` alone. Exterior cloud gas contributes zero. If `ε` is applied to
the cloud mass (SPEC-004/005), the same convention must be used for `M_sh` and for the density
normalisation, or gravity and inertia disagree.

### 8.6 Sign of the sweep-up term when `v2 < 0`
`Ṁ_sh = 4πR2²ρ v2` with `v2 < 0` *removes* mass from the shell — physically the shell does not
un-sweep gas it already carries. A collapsing shell should either keep `M_sh` fixed (`Ṁ_sh = 0`) or
implement an explicit re-deposition model, and the `−4πR2²ρv2²` deceleration term must remain a
*deceleration* (it is `∝ v2²`, so it opposes outward motion but *also* opposes inward motion only if
the sign is handled — as written, `−4πR2²ρv2²` is always inward-directed and therefore *accelerates*
a collapsing shell inward, which is wrong: ram pressure always opposes motion, so the term should be
`−4πR2²ρ v2|v2|`). This only matters if the energy phase can host `v2 < 0`; if it cannot, that must
be an enforced exit.

### 8.7 Equation numbering between Weaver+77 and the Rahner thesis
I could not open either source (literature access blocked). **I therefore refuse to assert any
Weaver equation number.** Any code comment citing "Weaver Eq. N" is unverifiable from this lens and
should be flagged as an unchecked citation, not as a defect. Similarly, the widely-quoted interior
prefactors `T_b = 1.51×10⁶ K L₃₆^{8/35} n₀^{2/35} t₆^{−6/35}` and
`n_b = 4.02×10⁻³ cm⁻³ L₃₆^{6/35} n₀^{19/35} t₆^{−22/35}` are mutually **inconsistent with
isobaricity**: their product gives `n_bT_b = 6.07×10³ K cm⁻³` against the dynamically-derived
`P_b/k_B = 2.5×10⁴ K cm⁻³` (μ=1) — a factor ≈4. At most one of the quoted pairs is right and I
cannot tell which. Prefer the prefactor-free structural forms `(C1)` and `δ=(2/7)(2α−β−1)`.

### 8.8 The `max()` in `P_drive`
Non-differentiable, non-conservative (§4.1 ★), and it is TRINITY's headline departure from
WARPFIELD. It is a modelling choice, not a derived result; the audit must confirm the work-balance
consequence is handled or documented.

---

## 9. Confidence ledger

**High** (re-derived here, arithmetic checked in-session): the momentum equation and every sign;
the `M_sh/2` self-gravity factor; the closure `(C1)–(C3)`; the energy ledger `(I1)` and the ★ term;
`η = 3/(5−w)`; `v2 ∝ t^{(w−2)/(5−w)}`; `ξ_E = 0.762865`; `E_b/(L_wt) = 1/(1+2η)`;
`P_b = 5L_wt/(22πR2³)`; the AU↔cgs table; the `ρ̄ = 3ρ(R2)/(3−w)` trap; the emission-measure
fractions `(1−ξ)^{1/5}`; the `(1−0.98)^{2/5} = 0.2091` factor.

**Medium** (recalled from the literature, structure confirmed but prefactor unverified): the
`R1 = √(ṗ_w/4πP_b)` convention vs the `3/4` strong-shock variant; the `(1+τ_IR)f_abs` vs
`(f_abs+τ_IR)` convention; the enthalpy (`5/2 P`) vs internal-energy (`3/2 P`) vent flux;
`Ṁ_evap = 16πμC T^{5/2}R/(25k_B)`; the Krumholz–Matzner-style `P_HII ∝ Q_i^{1/2}r^{−3/2}` form
(the *exponent* is high confidence, the composition prefactor medium).

**Low** (cannot verify, literature blocked): all Weaver equation numbers; the interior prefactors
`1.51e6`/`2.07e6`/`4.02e-3`; the `a` in `T_b^{7/2} = a P_b R2²/(Ct)`; whether `L_cool`'s table
normalisation is `n_e n_H`, `n_e n_ion`, or `n²`.

---

```json
[
  {
    "id": "S4-C-01",
    "file": "trinity/phase1_energy/energy_phase_ODEs.py",
    "line": "168",
    "class": "state",
    "severity": "S1",
    "claim": "The first component of the RHS must be exactly the second state component: f[0] = dR2/dt = y[1] = v2, with no re-derivation.",
    "evidence": "v2 is defined as dR2/dt (the contact-discontinuity speed, §1.1). Any other expression for f[0] (e.g. alpha*R2/t from the similarity solution, or a finite difference) makes the pair (R2, v2) internally inconsistent and destroys the meaning of every quantity derived from either.",
    "expected": "f[0] is a direct pass-through of y[1].",
    "failure_scenario": "R2 and v2 drift apart; the force budget, alpha = v2*t/R2, and the phase-exit tests all evaluate on an inconsistent state, and the error is invisible because both variables look individually plausible.",
    "repro": "In the RHS, assert f[0] == y[1] bitwise over a full run of param/simple_cluster.param.",
    "confidence": "high"
  },
  {
    "id": "S4-C-02",
    "file": "trinity/phase1_energy/energy_phase_ODEs.py",
    "line": "168",
    "class": "sign",
    "severity": "S1",
    "claim": "The swept-material momentum flux appears exactly once: either as d(M_sh v2)/dt (conservative form) OR as the explicit -4*pi*R2^2*rho_amb*v2^2 term in M_sh dv2/dt, never both.",
    "evidence": "d(M_sh v2)/dt = M_sh dv2/dt + Mdot_sh v2, and Mdot_sh v2 = 4 pi R2^2 rho_amb v2^2 exactly. The 'ram pressure of the ambient medium in the shell frame' and 'the momentum flux of newly swept material' are the same quantity under two names (§1.2, SPEC-020).",
    "expected": "Exactly one appearance; the deceleration term is -4*pi*R2^2*rho_amb(R2)*v2^2 in the M dv/dt form.",
    "failure_scenario": "Double-counted deceleration: the shell is systematically too slow, R2(t) falls below the t^(3/5) law, and the energy-phase duration and transition radius are both wrong -- while the run still looks smooth and physical.",
    "repro": "Gravity/radiation off, uniform medium: check alpha = v2*t/R2 relaxes to 0.600, not to a smaller value. Double-counting drives alpha toward ~0.5.",
    "confidence": "high"
  },
  {
    "id": "S4-C-03",
    "file": "trinity/phase1_energy/energy_phase_ODEs.py",
    "line": "168",
    "class": "coefficient",
    "severity": "S2",
    "claim": "The density in both Mdot_sh and the ram term must be the LOCAL ambient density rho_amb(R2), not the mean interior density M_sh/((4/3) pi R2^3).",
    "evidence": "For rho ~ r^-w, rho_bar = 3*rho(R2)/(3-w). These are identical at w=0 (the shipped default densPL_alpha=0) and differ by a factor 3 at w=2, the steepest supported profile.",
    "expected": "rho_amb(R2) evaluated from the density profile at r=R2.",
    "failure_scenario": "A bug that is exactly invisible in every default-configuration test and 3x wrong in the alpha=-2 sweep cells; the power-law radius exponent would still look right while the normalisation is off.",
    "repro": "Compare the code's ram-term density against rho_profile(R2) and against M_sh/((4/3) pi R2^3) for a densPL_alpha=-2 config; they differ by 3x.",
    "confidence": "high"
  },
  {
    "id": "S4-C-04",
    "file": "trinity/phase1_energy/energy_phase_ODEs.py",
    "line": "168",
    "class": "coefficient",
    "severity": "S2",
    "claim": "Gravity must be F_grav = G*M_sh*(M_cluster + M_sh/2)/R2^2 -- the shell's self-gravity carries a factor 1/2.",
    "evidence": "The field of a thin shell is G M/R^2 just inside and 0 just outside; a shell element sits on the discontinuity and feels the mean, G M/(2R^2). Equivalently U = -G M^2/(2R) and F = -dU/dR = -G M^2/(2R^2) (§1.4).",
    "expected": "Enclosed mass = M_cluster + M_sh/2; exterior cloud gas excluded entirely (shell theorem).",
    "failure_scenario": "Using M_sh (no 1/2) makes gravity up to 2x too strong once M_sh >> M_cluster (i.e. for every run with sfe <~ 0.3), biasing the model toward spurious recollapse; using M_cloud instead of M_sh(R2) is too strong at all R2 < r_cloud.",
    "repro": "Compare the reported F_grav snapshot key against G*M_sh*(M_cluster + M_sh/2)/R2^2 recomputed from R2, M_sh, M_cluster in dictionary.jsonl.",
    "confidence": "high"
  },
  {
    "id": "S4-C-05",
    "file": "trinity/phase1_energy/energy_phase_ODEs.py",
    "line": "168",
    "class": "coefficient",
    "severity": "S2",
    "claim": "The closure must be P_b = 3(gamma-1) E_b / [4 pi (R2^3 - R1^3)], i.e. gamma-general and R1-aware; the literal form E_b/(2 pi R2^3) hard-codes gamma=5/3 AND R1<<R2 simultaneously.",
    "evidence": "E_b = P_b V_b/(gamma-1) with V_b = (4pi/3)(R2^3 - R1^3). Setting gamma=5/3 gives E_b/[2pi(R2^3-R1^3)]; dropping R1 gives E_b/(2 pi R2^3) (§1.6).",
    "expected": "gamma read from the parameter (default.param declares gamma_adia); R1 included, or its omission documented and bounded.",
    "failure_scenario": "A gamma_adia parameter that silently does nothing; and a P_b overestimate of relative size 3(R1/R2)^3 that grows late in the energy phase as P_b falls and R1 = sqrt(pdot_w/(4 pi P_b)) grows.",
    "repro": "Set gamma_adia to a non-5/3 value and check P_b/E_b changes; recompute P_b from Eb, R2, R1 in dictionary.jsonl and compare.",
    "confidence": "high"
  },
  {
    "id": "S4-C-06",
    "file": "trinity/phase1_energy/energy_phase_ODEs.py",
    "line": "168",
    "class": "other",
    "severity": "S2",
    "claim": "The PdV term in dE_b/dt must use the SAME V_b as the P_b(E_b) closure: either both include R1 (PdV = 4pi P_b (R2^2 v2 - R1^2 dR1/dt)) or both drop it (PdV = 4 pi R2^2 P_b v2).",
    "evidence": "P_b V_b = (gamma-1) E_b is an identity; if P_b is formed from one volume and the work from another, dE_b/dt no longer corresponds to any thermodynamic system (§1.5).",
    "expected": "One volume definition, used in both places.",
    "failure_scenario": "A silent energy source or sink of relative size ~3(R1/R2)^3 that grows monotonically through the energy phase, shifting the transition time.",
    "repro": "Integrate dE_b/dt - [L_gain - P_b dV_b/dt - L_cool - L_leak] over a run; a non-zero residual is the leak.",
    "confidence": "high"
  },
  {
    "id": "S4-C-07",
    "file": "trinity/phase1_energy/energy_phase_ODEs.py",
    "line": "168",
    "class": "other",
    "severity": "S1",
    "claim": "The work the shell receives must equal the work the bubble loses: the term 4 pi R2^2 (P_drive - P_b) v2 in the combined energy ledger must be zero, or explicitly sourced and documented.",
    "evidence": "Multiplying the momentum equation by v2 and adding the bubble energy equation gives d(E_b+E_kin)/dt = L_gain - L_cool - L_leak - (1/2)Mdot_sh v2^2 - F_grav v2 - 4 pi R2^2 P_ISM v2 + F_rad v2 + 4 pi R2^2 (P_drive - P_b) v2. The last term vanishes iff P_drive = P_b (§4.1).",
    "expected": "Either P_drive = P_b on the bubble side too, or the excess is charged against the ionizing-photon energy budget, or the non-conservation is documented and bounded.",
    "failure_scenario": "With P_drive = max(P_b, P_HII), every timestep in which the P_HII branch wins creates energy at rate 4 pi R2^2 (P_HII - P_b) v2 out of nothing -- silently inflating R2 and v2 exactly in the regime TRINITY claims as its novelty.",
    "repro": "Log which max() branch is active per snapshot; integrate 4 pi R2^2 (P_drive - P_b) v2 dt over the energy phase and compare against L_mech*t.",
    "confidence": "high"
  },
  {
    "id": "S4-C-08",
    "file": "trinity/phase1_energy/energy_phase_ODEs.py",
    "line": "168",
    "class": "sign",
    "severity": "S2",
    "claim": "The bare wind/SN momentum flux (pdot_w + pdot_SN, or P_ram) must NOT appear in the shell momentum equation during the energy-driven phase.",
    "evidence": "In the energy-driven limit the wind momentum is thermalised at the termination shock and IS the source of P_b -- indeed 4 pi R1^2 P_b = pdot_w is the defining R1 condition. Adding both 4 pi R2^2 P_b and pdot_w double-counts the same momentum (§1.3). SPEC-022 places P_ram only in the transition and momentum phases.",
    "expected": "Energy-phase P_drive = max(P_b, P_HII) with no P_ram contribution.",
    "failure_scenario": "Over-driven shell during the energy phase; the force-fraction plot shows a spurious ram component; the energy-phase radius exceeds the Weaver law.",
    "repro": "Check the reported F_ram_wind / F_ram_SN snapshot keys are zero (or excluded from F_tot) while current_phase is 'energy'/'implicit'.",
    "confidence": "high"
  },
  {
    "id": "S4-C-09",
    "file": "trinity/phase1_energy/energy_phase_ODEs.py",
    "line": "168",
    "class": "coefficient",
    "severity": "S2",
    "claim": "Radiation pressure must not double-count the direct term inside the IR term: F_rad = (L_bol/c) f_abs (1 + tau_IR), with tau_IR = kappa_IR M_sh/(4 pi R2^2) and f_abs = 1 - exp(-tau_UV).",
    "evidence": "Only the absorbed luminosity f_abs*L is reprocessed into IR; the trapped IR adds tau_IR times that. The alternative (L/c)(f_abs + tau_IR) implicitly reprocesses the escaping fraction. Adding a separate direct term to an IR expression already written as (1+tau_IR) f_abs L/c counts the direct momentum twice (§8.1).",
    "expected": "One expression; limits F_rad -> L_bol/c as tau_UV->inf, tau_IR->0, and F_rad -> 0 as both -> 0.",
    "failure_scenario": "Up to 2x too much radiative driving in the optically thick early phase, where SPEC-071 notes L_bol/c already exceeds the wind momentum by ~3x -- i.e. the dominant early force is the one that is doubled.",
    "repro": "Evaluate the reported F_rad at a snapshot against (L_bol/c)*f_abs*(1+tau_IR) and against (L_bol/c)*(f_abs+tau_IR) and against their sum.",
    "confidence": "medium"
  },
  {
    "id": "S4-C-10",
    "file": "trinity/phase1_energy/energy_phase_ODEs.py",
    "line": "168",
    "class": "units",
    "severity": "S2",
    "claim": "tau_IR must be kappa_IR times the MASS column M_sh/(4 pi R2^2), with kappa_IR converted from 4 cm^2/g to 8.3556e-4 pc^2/Msun; and L_bol/c must use c = 3.06601e5 pc/Myr.",
    "evidence": "Recomputed here: 4 cm^2/g * 1.98892e33 g/Msun / (3.0857e18 cm/pc)^2 = 8.3556e-4 pc^2/Msun; c = 2.99792458e10 * 3.15576e13 / 3.0857e18 = 3.06601e5 pc/Myr. Dimension check [L/c] = (Msun pc^2 Myr^-3)/(pc/Myr) = Msun pc Myr^-2 = force (§2).",
    "expected": "Mass column, not number column; kappa_IR and sigma_d (1.5e-21 cm^2 per H nucleus = 1.5754e-58 pc^2) kept distinct.",
    "failure_scenario": "Using a number column with kappa_IR, or mixing kappa_IR (per gram) with sigma_d (per nucleus), is a ~1e23 error that fails loudly; a missed unit conversion on kappa_IR alone is a quiet factor ~1.2e-4 that switches IR trapping off.",
    "repro": "Check tau_IR at a snapshot equals kappa_IR_AU * M_sh/(4 pi R2^2) with M_sh in Msun, R2 in pc.",
    "confidence": "high"
  },
  {
    "id": "S4-C-11",
    "file": "trinity/phase1_energy/energy_phase_ODEs.py",
    "line": "36",
    "class": "exponent",
    "severity": "S2",
    "claim": "get_press_ion(r, params) must scale as P_HII ∝ Q_i^{1/2} r^{-3/2} and must include the full particle count n_tot = 2.2 n_H, not n_H alone.",
    "evidence": "Stromgren balance Q_i = (4pi/3) alpha_B chi_e n_H^2 r^3 gives n_H(r) = [3 Q_i/(4 pi alpha_B chi_e r^3)]^{1/2}; P = n_tot k_B T_ion with n_tot/n_H = 1 + x_He + chi_e = 2.2 for x_He=0.1, chi_e=1.1 (SPEC-029). The exponents are prefactor-free and therefore the strongest test (§1.7).",
    "expected": "P_HII decreasing in r as r^{-3/2}, increasing as sqrt(Q_i), with the 2.2 composition factor.",
    "failure_scenario": "Omitting n_tot understates P_HII by 2.2x, which directly changes which branch of max(P_b, P_HII) wins -- TRINITY's headline claim. A wrong r-exponent changes the whole late-time driving history.",
    "repro": "Call get_press_ion at r and 2r with fixed params: the ratio must be 2^{-3/2} = 0.35355.",
    "confidence": "high"
  },
  {
    "id": "S4-C-12",
    "file": "trinity/phase1_energy/energy_phase_ODEs.py",
    "line": "36",
    "class": "regime",
    "severity": "S2",
    "claim": "The geometry assumed for P_HII must be stated and must be physically consistent with a hot bubble occupying r<R2.",
    "evidence": "During the energy phase the volume interior to R2 holds 1e6-1e7 K shocked wind at n ~ 1e-2 cm^-3, whose recombination rate is negligible; the real ionized gas is a thin dense skin on the shell's inner face at far higher pressure. The filled-sphere formula (P1) is the 'no bubble' classical HII-region pressure (SPEC-030 Readings A vs B).",
    "expected": "One reading, documented, with the inconsistency of co-existing 1e4 K and 1e7 K gas in the same volume acknowledged.",
    "failure_scenario": "The filled-sphere reading systematically under-estimates P_HII during the energy phase, so max(P_b, P_HII) almost never selects P_HII and TRINITY's stated novelty is inert; the opposite reading over-drives the shell.",
    "repro": "",
    "confidence": "medium"
  },
  {
    "id": "S4-C-13",
    "file": "trinity/phase1_energy/energy_phase_ODEs.py",
    "line": "168",
    "class": "numerical",
    "severity": "S3",
    "claim": "P_drive = max(P_b, P_HII) is non-differentiable at the crossing and must be handled (event + restart, or smoothing), not fed raw to an adaptive stiff solver.",
    "evidence": "max(a,b) has a kink wherever a=b; an adaptive controller repeatedly rejects steps at a kink, so the achieved accuracy is not the requested RTOL. The same class of problem appears to have been hit for r_cloud (SPEC-023 notes an LSODA/smoothing figure).",
    "expected": "Either a terminal/non-terminal event at the crossing with integrator restart, or a documented C1 blend.",
    "failure_scenario": "Solver chatter: step-size collapse near the crossing, which can then trip DT_EXIT_THRESHOLD and be mis-reported as a physical phase transition.",
    "repro": "Log the active max() branch and the accepted step size per step; look for step collapse coincident with branch switches.",
    "confidence": "high"
  },
  {
    "id": "S4-C-14",
    "file": "trinity/phase1_energy/energy_phase_ODEs.py",
    "line": "168",
    "class": "exponent",
    "severity": "S2",
    "claim": "In the pure-energy limit (no gravity/radiation/external pressure, uniform medium, constant L_w) the solution must approach R2 = 0.762865 (L_w/rho0)^{1/5} t^{3/5} and v2 = 0.457719 (L_w/rho0)^{1/5} t^{-2/5}, i.e. alpha = v2 t/R2 -> 3/5.",
    "evidence": "Derived in full in §3.1: L_w = 3.87045 rho0 A^5, A^5 = 250/(308 pi) L_w/rho0 = 0.2583684 L_w/rho0. Recomputed: (250/308pi)^{1/5} = 0.7628653 (check 0.7628653^5 = 0.2583684). Note SPEC-050's 0.762934 is off in the 5th digit.",
    "expected": "alpha -> 0.600 through the energy phase; radius matching the coefficient above for the code's own mu convention.",
    "failure_scenario": "Any missing or doubled force term shows up here first; conversely a code that hard-codes 0.762934 has inherited a rounding rather than a derivation.",
    "repro": "Gravity/radiation-disabled uniform-medium run; plot v2*t/R2 vs t and compare against 0.600.",
    "confidence": "high"
  },
  {
    "id": "S4-C-15",
    "file": "trinity/phase1_energy/energy_phase_ODEs.py",
    "line": "168",
    "class": "coefficient",
    "severity": "S2",
    "claim": "In the same limit, E_b/(L_mech t) must approach 5/11 = 0.454545 (uniform), and 1/(1+2*eta) with eta=3/(5-w) in general.",
    "evidence": "E_b = (2pi/3) eta (4eta-1) rho0 A^5 t^{5eta-2} and L_w = B A^{5-w} eta[(4-w)eta-1](1/2+eta); their ratio is 1/(1+2eta) (§3.2-3.3). Checks: w=0 -> 5/11; w=1 -> 2/5; w=2 -> 1/3 (equipartition with E_kin and radiated).",
    "expected": "0.4545 (w=0), 0.400 (w=1), 0.3684 (w=1.5), 0.3333 (w=2).",
    "failure_scenario": "A dimensionless test that no unit bug can fake; failure localises the error to the energy equation (wrong PdV coefficient, wrong thermalisation efficiency, or an unaccounted loss).",
    "repro": "Compute Eb/(Lmech_W * t_now) from dictionary.jsonl during the energy phase of a loss-free run.",
    "confidence": "high"
  },
  {
    "id": "S4-C-16",
    "file": "trinity/phase1_energy/energy_phase_ODEs.py",
    "line": "168",
    "class": "exponent",
    "severity": "S2",
    "claim": "On a power-law ambient rho ~ r^{-w} both exponents change: R2 ~ t^{3/(5-w)} AND v2 ~ t^{(w-2)/(5-w)}; the uniform-medium prefactor 0.762865 must not be reused.",
    "evidence": "Derived §3.3. v2 exponent = eta-1 = (w-2)/(5-w): -2/5 at w=0, -1/4 at w=1, 0 at w=2 (constant-velocity expansion). Prefactor A^{5-w} = L_w(3-w)/{4 pi rho_ref r_ref^w eta[(4-w)eta-1](1/2+eta)}.",
    "expected": "Both exponents w-dependent; the prefactor recomputed per w.",
    "failure_scenario": "A validation test that checks only the radius exponent passes while the normalisation is wrong; and the published radiusComparison figure anchors to the simulation's own midpoint, so it tests slopes only and cannot catch this (SPEC-057).",
    "repro": "Run densPL_alpha = 0, -1, -2 in the pure-energy limit; fit both d ln R2/d ln t and d ln v2/d ln t.",
    "confidence": "high"
  },
  {
    "id": "S4-C-17",
    "file": "trinity/phase1_energy/energy_phase_ODEs.py",
    "line": "325",
    "class": "other",
    "severity": "S2",
    "claim": "compute_derived_quantities must report the SAME force/pressure values the RHS actually used, so that M_sh dv2/dt reconstructed from the reported forces equals the integrator's dv2/dt to tolerance.",
    "evidence": "SPEC-007 requires closure; the published force-fraction plots normalise by F_tot, which presupposes the listed terms are exhaustive and non-overlapping. Two independent code paths computing 'the same' force is the standard way closure silently breaks (§4.2).",
    "expected": "A single evaluation shared between the RHS and the diagnostics (or a tested identity between them).",
    "failure_scenario": "Force fractions that sum to something other than the actual acceleration; a stacked-area figure that is quantitatively wrong while looking correct.",
    "repro": "For each snapshot, recompute (F_drive + F_rad - F_grav - F_ram - 4 pi R2^2 P_ISM)/M_sh and compare against the finite-difference dv2/dt from consecutive snapshots.",
    "confidence": "high"
  },
  {
    "id": "S4-C-18",
    "file": "trinity/phase1_energy/energy_phase_ODEs.py",
    "line": "168",
    "class": "silent-failure",
    "severity": "S2",
    "claim": "Non-positive E_b (or V_b <= 0, or R1 >= R2) during the energy phase is a violation of the phase assumption and must terminate the phase with a distinct recorded reason -- never be clipped, floored, or absolute-valued.",
    "evidence": "P_b = (gamma-1)E_b/V_b, so E_b <= 0 means a non-positive driving pressure: the bubble is no longer an energy reservoir, which is the definition of leaving the energy phase (§4.3, SPEC-011).",
    "expected": "Hard guards with distinct termination reasons for each of E_b<=0, R1>=R2, V_b<=0, non-finite state.",
    "failure_scenario": "A clamped E_b lets the integrator continue past the point where the model is valid, producing a fictitious extended energy phase and a transition time that is a property of the clamp.",
    "repro": "Grep metadata.json termination blocks across the shipped sweeps for runs whose final Eb is at or near any floor value.",
    "confidence": "high"
  },
  {
    "id": "S4-C-19",
    "file": "trinity/phase1_energy/run_energy_phase.py",
    "line": "62",
    "class": "numerical",
    "severity": "S2",
    "claim": "Every phase-exit condition must be a continuous, sign-changing solver event with an explicit direction and root-finding on dense output -- not a boolean check evaluated at step or segment boundaries.",
    "evidence": "SPEC-016 requires dv2/dt continuity across the handover; a boundary check resolves the exit only to the step (or COOLING_UPDATE_INTERVAL) and hands the next phase a state that is off the event surface by O(dt*v2) (§4.5, §6.2).",
    "expected": "solve_ivp events with terminal=True and direction set; earliest root wins and is recorded.",
    "failure_scenario": "A spurious discontinuity in R2/v2 at the phase boundary that is then attributed to physics; and an exit time quantised to the segment length rather than resolved to tolerance.",
    "repro": "Sample dv2/dt on both sides of the energy->implicit/transition boundary and compare the jump against RTOL.",
    "confidence": "high"
  },
  {
    "id": "S4-C-20",
    "file": "trinity/phase1_energy/run_energy_phase.py",
    "line": "62",
    "class": "state",
    "severity": "S2",
    "claim": "R2 and v2 must be continuous across the phase boundary; only E_b may be re-initialised, and only if the transition is defined as a depressurisation and that is documented.",
    "evidence": "R2 and v2 solve a second-order ODE with bounded forces, so they are C0 and C1 in t by construction (§4.5).",
    "expected": "The handover carries (R2, v2) unchanged.",
    "failure_scenario": "A radius or velocity jump at the handover silently rescales the entire post-transition trajectory, including the dispersal-vs-recollapse verdict.",
    "repro": "Diff the last energy-phase snapshot against the first transition-phase snapshot in dictionary.jsonl for R2 and v2.",
    "confidence": "high"
  },
  {
    "id": "S4-C-21",
    "file": "trinity/phase1_energy/run_energy_phase.py",
    "line": "62",
    "class": "regime",
    "severity": "S3",
    "claim": "The cooling-balance exit threshold (0.05) is a numerical regularisation of the physical statement 'L_loss/L_gain -> 1', and any transition time reported from it must carry its threshold sensitivity.",
    "evidence": "SPEC-013/014: the physical criterion is a limit, not a number; any threshold in 0.01-0.2 is equally defensible. The 'ebpeak' alternative (Edot_b <= 0) is threshold-free and strictly later.",
    "expected": "Documented sensitivity of the transition time to the threshold across 0.01-0.2.",
    "failure_scenario": "The code's headline prediction (transition time, hence dispersal vs recollapse) is quietly a function of a tuning constant.",
    "repro": "Re-run one config at phaseSwitch_LlossLgain = 0.01, 0.05, 0.2 and tabulate the transition time.",
    "confidence": "high"
  },
  {
    "id": "S4-C-22",
    "file": "trinity/phase1_energy/run_energy_phase.py",
    "line": "56",
    "class": "silent-failure",
    "severity": "S2",
    "claim": "DT_EXIT_THRESHOLD can only be a step-size-collapse (stiffness) detector; an exit triggered by it must be recorded as a distinct non-physical termination, never conflated with a physical phase transition.",
    "evidence": "A minimum-step bail-out is a property of the solver and the tolerances, not of the bubble. Given the known kink sources (§5.5), step collapse is expected near the max() crossing and at r_cloud.",
    "expected": "A separate exit_code/outcome string in the termination block for 'integrator step collapse'.",
    "failure_scenario": "The transition time becomes a function of RTOL/ATOL and of where the pressure branches cross, while being reported as physics.",
    "repro": "Instrument the exit path to record which condition fired; scan metadata.json across shipped sweeps for the distribution.",
    "confidence": "medium"
  },
  {
    "id": "S4-C-23",
    "file": "trinity/phase1_energy/run_energy_phase.py",
    "line": "57",
    "class": "numerical",
    "severity": "S3",
    "claim": "COOLING_UPDATE_INTERVAL (and SEGMENT_DURATION) must be logarithmic in t, or small compared to t at the earliest times; a fixed absolute interval produces a staleness error ~0.54*dt/t that diverges as t->0.",
    "evidence": "Derived §5.2: in the Weaver limit n_b ~ t^{-22/35}, T_b ~ t^{-6/35}, V_b ~ t^{9/5}, so L_cool ~ t^{19/35} Lambda(T_b), giving |d ln L_cool/d ln t| >~ 0.54. A frozen L_cool over dt is therefore in error by ~0.54 dt/t.",
    "expected": "dt <= f * min(t, R2/v2, E_b/|Edot_b|), or a demonstrated convergence study.",
    "failure_scenario": "Large systematic error in L_cool at early times, precisely where the energy phase is established; and an exit time that cannot be resolved better than the update interval.",
    "repro": "Halve COOLING_UPDATE_INTERVAL and SEGMENT_DURATION and compare R2(t), v2(t) and the transition time (separate processes, matched t).",
    "confidence": "medium"
  },
  {
    "id": "S4-C-24",
    "file": "trinity/phase1_energy/energy_phase_ODEs.py",
    "line": "114",
    "class": "numerical",
    "severity": "S3",
    "claim": "No R2-dependent quantity (M_sh, rho_amb, f_abs, tau_IR, P_HII) may be frozen in ODESnapshot across a segment in which R2 changes appreciably; and the frozen-coefficient scheme must be shown convergent as the segment length -> 0.",
    "evidence": "M_sh ~ R2^{3-w}, tau_IR ~ M_sh/R2^2, P_HII ~ R2^{-3/2}; with R2 ~ t^{3/5}, R2 grows 1.52x per doubling of t, so these change by tens of percent within any non-trivial segment (§7).",
    "expected": "Either these are evaluated live from y, or the segment is short enough that their variation is below the claimed tolerance -- demonstrated, not assumed.",
    "failure_scenario": "The reported RTOL is not the accuracy of the answer; the run's trajectory depends on a constant that is presented as an implementation detail.",
    "repro": "Convergence study on SEGMENT_DURATION, full runs in separate processes at matched simulation time.",
    "confidence": "medium"
  },
  {
    "id": "S4-C-25",
    "file": "trinity/phase1_energy/run_energy_phase.py",
    "line": "59",
    "class": "numerical",
    "severity": "S3",
    "claim": "A scalar ATOL cannot serve a state vector spanning ~10 decades (R2 ~ 1e0-1e2 pc, v2 ~ 1e0-1e2 pc/Myr, E_b ~ 1e7-1e10 Msun pc^2 Myr^-2).",
    "evidence": "1e51 erg = 5.26e7 in AU energy units; a 1e6 Msun cluster at L_w ~ 1e40 erg/s accumulates ~1.6e10 AU per Myr, while R2 is O(10). ATOL sized for either component is meaningless for the other (§5.1).",
    "expected": "A per-component ATOL vector, or E_b integrated in a rescaled variable.",
    "failure_scenario": "Either wasted work / step collapse on E_b, or (if ATOL is sized for E_b) an effectively uncontrolled R2 and v2 -- most damaging near v2 -> 0, where ATOL alone governs the error and hence the recollapse verdict.",
    "repro": "Tighten RTOL by 10x and check R2(t), v2(t), and the transition time move by less than the claimed tolerance.",
    "confidence": "medium"
  },
  {
    "id": "S4-C-26",
    "file": "trinity/phase1_energy/energy_phase_ODEs.py",
    "line": "168",
    "class": "sign",
    "severity": "S3",
    "claim": "If v2 < 0 is reachable in the energy phase, the sweep-up terms must handle the sign: Mdot_sh must not go negative (the shell cannot un-sweep gas), and the ram term must be -4 pi R2^2 rho v2|v2| so that it always opposes motion.",
    "evidence": "Mdot_sh = 4 pi R2^2 rho v2 with v2<0 removes mass; and -4 pi R2^2 rho v2^2 written with v2^2 is always inward-directed, so for a collapsing shell it accelerates the collapse instead of resisting it. Ram pressure always opposes relative motion (§8.6).",
    "expected": "Mdot_sh = 4 pi R2^2 rho max(v2,0) (or an explicit re-deposition model); ram term ~ v2|v2|.",
    "failure_scenario": "A collapsing shell that loses mass and is accelerated inward -- an artificially fast, artificially light recollapse, i.e. a biased fate classification.",
    "repro": "Construct a config that stalls and reverses inside the energy phase; check M_sh is monotone non-decreasing and the ram term opposes v2.",
    "confidence": "medium"
  },
  {
    "id": "S4-C-27",
    "file": "trinity/phase1_energy/energy_phase_ODEs.py",
    "line": "168",
    "class": "coefficient",
    "severity": "S3",
    "claim": "L_gain = eta_w L_mech,w + eta_SN L_mech,SN: the thermalisation efficiencies multiply the mechanical LUMINOSITY only, never the momentum input.",
    "evidence": "Thermalisation efficiency is the fraction of kinetic energy converted to bubble thermal energy at the reverse shock. Momentum is conserved regardless of thermalisation, so pdot must never be scaled by eta (§1.5).",
    "expected": "eta applied to L only; pdot_w unscaled.",
    "failure_scenario": "Silently coupled energy and momentum budgets; a run with eta != 1 would then also mis-set R1 = sqrt(pdot_w/(4 pi P_b)).",
    "repro": "Set FB_thermCoeffWind to 0.5 and confirm pdot_w-derived quantities (R1, v_w = 2L/pdot) are unchanged.",
    "confidence": "medium"
  },
  {
    "id": "S4-C-28",
    "file": "trinity/phase1_energy/energy_phase_ODEs.py",
    "line": "168",
    "class": "regime",
    "severity": "S2",
    "claim": "L_cool must be computed on a grid that resolves the conduction front adjacent to the contact discontinuity, and must be convergent with respect to the outer cut radius.",
    "evidence": "Derived §5.8: with n ~ (1-x)^{-2/5}, the fraction of the emission measure at x>xi is (1-xi)^{1/5} = 45.7% at xi=0.98, 39.8% at 0.99, 25.1% at 0.999. Worse, Lambda(T) peaks near 1e5 K, which for T_b ~ 3e6 K sits at 1-x = (1e5/3e6)^{5/2} ~ 2e-4, i.e. x ~ 0.9998 -- inside a xi=0.98 cut.",
    "expected": "A cut (or grid) whose effect on L_cool is demonstrably below the transition-trigger tolerance.",
    "failure_scenario": "L_cool systematically under-predicted, so the energy phase lasts too long -- compounding, in the same direction, the known 1-D-conduction bias that SPEC-015 says the cooling_boost knobs exist to patch.",
    "repro": "Recompute L_cool at xi_max = 0.98, 0.99, 0.999, 0.9999 at a fixed state and check convergence.",
    "confidence": "medium"
  },
  {
    "id": "S4-C-29",
    "file": "trinity/phase1_energy/energy_phase_ODEs.py",
    "line": "168",
    "class": "other",
    "severity": "S3",
    "claim": "If the state vector carries a bubble temperature component, it is slaved -- its derivative must be the total derivative of the conduction closure along the trajectory, giving delta = d ln T/d ln t = (2/7)(2 alpha - beta - 1).",
    "evidence": "T_b^{7/2} = a P_b R2^2/(C t) (dimensionally exact: [C T^{7/2}] = erg s^-1 cm^-1 = [P R^2/t]); taking d ln/d ln t gives delta = (2/7)(2 alpha - beta - 1), which at alpha=3/5, beta=4/5 yields exactly -6/35, matching default.param's cool_delta (SPEC-041/042, §5.6).",
    "expected": "delta ~ (2/7)(2 alpha - beta - 1) throughout the energy phase, degrading only as cooling becomes important.",
    "failure_scenario": "An integrated T that drifts off the constraint manifold becomes a lagging, unconstrained variable feeding the cooling and hence the exit trigger.",
    "repro": "Extract (alpha, beta, delta) per snapshot and test delta ~ (2/7)(2 alpha - beta - 1).",
    "confidence": "medium"
  },
  {
    "id": "S4-C-30",
    "file": "trinity/phase1_energy/energy_phase_ODEs.py",
    "line": "325",
    "class": "coefficient",
    "severity": "S3",
    "claim": "A bubble temperature reported at xi = 0.98 is 0.20913 of the central T_b and must not be compared against a Weaver central-temperature formula without that factor.",
    "evidence": "Weaver interior profile T(r) = T_b (1 - r/R2)^{2/5}; recomputed (1-0.98)^{2/5} = 0.2091279 (§5.7, SPEC-040).",
    "expected": "Any T0-vs-Weaver comparison carries the factor ~4.78.",
    "failure_scenario": "A validation check that appears to fail by a factor ~5, or (worse) a hard-coded prefactor tuned to absorb it.",
    "repro": "Compare the stored bubble_T_arr profile's central value against the reported T0.",
    "confidence": "high"
  },
  {
    "id": "S4-C-31",
    "file": "trinity/phase1_energy/energy_phase_ODEs.py",
    "line": "168",
    "class": "citation",
    "severity": "S3",
    "claim": "Hard-coded Weaver interior prefactors (1.51e6 / 2.07e6 K, 4.02e-3 cm^-3) cannot be validated and are mutually inconsistent with isobaricity; and any 'Weaver Eq. N' citation is unverifiable from this lens.",
    "evidence": "n_b T_b from the quoted pair = 4.02e-3 * 1.51e6 = 6.07e3 K cm^-3, against the dynamically derived P_b/k_B = 2.5e4 K cm^-3 (mu=1) -- a factor ~4 that the 2.3 composition factor cannot explain. Literature access blocked (spec §0.3), so I refuse to assert any Weaver equation number (§8.7).",
    "expected": "Prefactor-free structural forms (the closure P_b = (gamma-1)E_b/V_b and delta = (2/7)(2 alpha - beta - 1)) rather than quoted numbers; any retained numeric prefactor labelled as unverified.",
    "failure_scenario": "A hard-coded prefactor silently sets the bubble's thermal state and therefore L_cool and the transition time, with no traceable provenance.",
    "repro": "",
    "confidence": "low"
  },
  {
    "id": "S4-C-32",
    "file": "trinity/phase1_energy/energy_phase_ODEs.py",
    "line": "168",
    "class": "coefficient",
    "severity": "S3",
    "claim": "R1 must satisfy ram-pressure balance R1 = sqrt(pdot_w/(4 pi P_b)) (or the strict strong-shock variant sqrt(3 pdot_w/(16 pi P_b)), smaller by 0.866) and must be enforced 0 < R1 < R2.",
    "evidence": "rho_w(r) v_w^2 = P_b with rho_w = Mdot_w/(4 pi r^2 v_w) gives R1^2 = pdot_w/(4 pi P_b). Using the post-shock pressure (3/4) rho v^2 for gamma=5/3 gives the 0.866 variant (SPEC-025). R1 enters V_b and hence P_b, so the convention must be consistent throughout.",
    "expected": "One convention, used consistently in V_b, in the PdV term, and in any R1 diagnostic; R1 < R2 enforced.",
    "failure_scenario": "Late in the energy phase P_b falls, R1 grows as P_b^{-1/2}, and R1 -> R2 makes V_b -> 0 and P_b diverge -- a NaN rather than a recorded exit.",
    "repro": "Check the reported R1, R2, Pb satisfy 4 pi R1^2 Pb = pdot_w (or the 3/4 variant) at every snapshot, and that R1/R2 stays < 1.",
    "confidence": "high"
  },
  {
    "id": "S4-C-33",
    "file": "trinity/phase1_energy/energy_phase_ODEs.py",
    "line": "168",
    "class": "coefficient",
    "severity": "S3",
    "claim": "If a covering-fraction leak term is present, it must be the enthalpy flux (gamma/(gamma-1)) P_b = (5/2)P_b for gamma=5/3, not the internal-energy density (3/2)P_b, and it must vanish identically at C_f = 1.",
    "evidence": "Freely-venting gas carries enthalpy, not internal energy: the energy flux through an open boundary is (u + P/rho) rho v = [gamma/(gamma-1)] P v. The two differ by 40% (SPEC-036, §1.5).",
    "expected": "L_leak = (1-C_f) 4 pi R2^2 c_s (5/2) P_b; C_f=1 reproduces the sealed Weaver bubble bit-identically.",
    "failure_scenario": "A 40% error in the venting loss whenever C_f < 1; and if C_f=1 does not reduce to exactly zero, the default runs silently carry a leak.",
    "repro": "Run with coverFraction = 1.0 and confirm byte-identical dictionary.jsonl against a build with the leak term removed.",
    "confidence": "medium"
  },
  {
    "id": "S4-C-34",
    "file": "trinity/phase1_energy/energy_phase_ODEs.py",
    "line": "30",
    "class": "silent-failure",
    "severity": "S4",
    "claim": "_scalar must be total on the values it receives and must not silently discard information (e.g. reducing a length-n>1 array to its first element, or squeezing away a shape mismatch).",
    "evidence": "It sits at a shape boundary between array-returning physics helpers and a scalar ODE RHS. A lenient reducer converts a real shape bug into a plausible number that then propagates through the whole trajectory (§7.4).",
    "expected": "Accept 0-d and length-1 only; raise on anything else.",
    "failure_scenario": "A helper that starts returning a vector (e.g. a vectorised cooling lookup) silently contributes only its first element, producing a physically plausible but wrong RHS with no error anywhere.",
    "repro": "Pass a length-2 array and confirm it raises rather than returning element 0.",
    "confidence": "medium"
  },
  {
    "id": "S4-C-35",
    "file": "trinity/phase1_energy/run_energy_phase.py",
    "line": "54",
    "class": "other",
    "severity": "S3",
    "claim": "TFINAL_ENERGY_PHASE is a run limit, not physics; hitting it must be recorded as a distinct numerical cutoff and must exceed any physically reachable transition time in the shipped configs.",
    "evidence": "SPEC-100 lists 'numerical cutoff' as a category separate from every physical fate; a hard cap that silently ends the phase changes the reported transition time into a property of the constant (§5.4).",
    "expected": "Distinct termination reason; documented margin against the shipped configs.",
    "failure_scenario": "For a low-density / weak-feedback config the energy phase is truncated at the cap and the truncation is reported as a physical transition.",
    "repro": "Scan metadata.json across the shipped sweeps for runs whose energy phase ends exactly at the cap.",
    "confidence": "medium"
  },
  {
    "id": "S4-C-36",
    "file": "trinity/phase1_energy/energy_phase_ODEs.py",
    "line": "168",
    "class": "units",
    "severity": "S2",
    "claim": "The mu used for n_H -> rho (mu_H = 1.4, mass per hydrogen nucleus) and the mu used in P = rho k T/(mu m_H) (mass per particle) are different constants and must not be interchanged.",
    "evidence": "SPEC-092.1; recomputed anchor: n_H = 1 cm^-3 with mu_H = 1.4 gives rho = 0.034613 Msun/pc^3. Using mu_ion ~ 0.609 instead deflates rho by 2.3x. The Weaver radius depends on rho^{-1/5}, so a 2.3x density error is a 17% radius error -- large but not obviously wrong-looking.",
    "expected": "mu_H = 1.4 for every n<->rho conversion (ambient profile, swept mass, ram term); the per-particle mu only inside P = n_tot k T.",
    "failure_scenario": "A compensating-error trap: the famous 28 pc anchor is a mu=1 number, so a test asserting 28 pc against a mu_H=1.4 code would 'pass' a code that also had a mu bug. With mu_H=1.4 the correct value is 26.22 pc.",
    "repro": "Check rho_amb(R2) at a snapshot equals 1.4 * m_H * n(R2) in consistent units; compare a uniform-medium run against 26.22 pc, not 28 pc.",
    "confidence": "high"
  },
  {
    "id": "S4-C-37",
    "file": "trinity/phase1_energy/energy_phase_ODEs.py",
    "line": "168",
    "class": "units",
    "severity": "S3",
    "claim": "P_ISM enters the momentum equation as -4 pi R2^2 P_ISM with P_ISM converted from the declared P/k_B in K cm^-3: multiply by k_B = 7.2606e-60 Msun pc^2 Myr^-2 K^-1 and convert cm^-3 -> pc^-3 (1 cm^-3 = 2.9380e55 pc^-3).",
    "evidence": "SPEC-092.4: the input is a pressure divided by k_B, not a pressure. The paperII grid sweeps PISM up to 1e6 K cm^-3 = 1.4e-10 dyn cm^-2, a large confining pressure, so the term is not negligible there.",
    "expected": "Explicit k_B multiplication at the parameter boundary; the term enters with a minus sign.",
    "failure_scenario": "Omitting k_B makes P_ISM ~1e16 times too large -- loud. Applying it twice, or dropping the cm^-3 -> pc^-3 conversion, gives a quietly wrong confining pressure exactly in the high-PISM sweep cells.",
    "repro": "Set PISM to 1e6 and confirm the reported external-pressure force equals 4 pi R2^2 * 1e6 * k_B in AU units.",
    "confidence": "high"
  },
  {
    "id": "S4-C-38",
    "file": "trinity/phase1_energy/run_energy_phase.py",
    "line": "296",
    "class": "numerical",
    "severity": "S3",
    "claim": "The RHS must be C1 in (t,y) except at explicitly handled events; the known kink sources must each be an event-with-restart or be smoothed.",
    "evidence": "Enumerated §5.5: (1) max(P_b,P_HII); (2) rho_amb discontinuous at r_cloud and kinked at r_core; (3) linearly interpolated SPS drivers L_w(t), pdot_w(t) kinked at every table node; (4) age-indexed cooling files -> piecewise-constant in cluster age (SPEC-083); (5) the snapshot refresh itself.",
    "expected": "Events/restarts or C1 blends at each; the achieved accuracy then matches the requested RTOL.",
    "failure_scenario": "Repeated step rejection at kinks means the effective accuracy is not RTOL; combined with DT_EXIT_THRESHOLD, a kink can terminate the phase.",
    "repro": "Log accepted/rejected step sizes; look for clusters at r_cloud crossing, at SPS table nodes, and at cooling-file age boundaries.",
    "confidence": "high"
  },
  {
    "id": "S4-C-39",
    "file": "trinity/phase1_energy/energy_phase_ODEs.py",
    "line": "168",
    "class": "other",
    "severity": "S2",
    "claim": "The integrated energy ledger must close: int L_gain dt = E_b + (1/2)M_sh v2^2 + int L_cool dt + int L_leak dt + int (1/2) Mdot_sh v2^2 dt + int F_grav v2 dt + int 4 pi R2^2 P_ISM v2 dt - int F_rad v2 dt.",
    "evidence": "Derived §4.1 by combining the momentum and energy equations; the (1/2)Mdot_sh v2^2 term is the shock dissipation radiated at the shell's outer front, which in the Weaver limit accounts for 27/77 = 0.3506 of L_w t.",
    "expected": "Closure to integrator tolerance at every snapshot, with each ledger entry reported.",
    "failure_scenario": "Any missing sink or double-counted source shows up nowhere else; the run remains smooth and physical-looking while conserving nothing.",
    "repro": "Post-process dictionary.jsonl: accumulate each ledger term and plot the residual against int L_gain dt for param/simple_cluster.param and the two f1edge configs.",
    "confidence": "high"
  }
]
```
