# S6 transition + momentum — Lens C (what it should be)

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

**Role.** Physics tier of a blind triangulation audit. I have read only
`signatures.md` (names + signatures, no values, no bodies) and
`docs/dev/code-audit/reference/PHYSICS_SPEC.md`. I have not read `trinity/`, any stripped copy, any
docstring, or any other agent's report. Everything below is what the two functions **must** compute
to be physically correct, derived from first principles plus the cited SPEC ids.

**Literature access.** Blocked (arXiv/ADS/OUP 403), exactly as recorded in PHYSICS_SPEC §0.3. Every
Weaver+77 / Rahner+17 *equation number* below is therefore **unassertable** and I refuse to give
one; the *content* claims are either re-derived here (high confidence) or flagged as half-remembered
(low/medium). Confidence tags are strict.

---

## 0. Interface reading (what the signatures alone force)

| Signature fact | What it forces physically |
|---|---|
| transition file has `ENERGY_FLOOR`, momentum file does **not** | `E_b` is a state variable in the transition ODE and is **gone** by the momentum phase. The transition state vector is `y = [R2, v2, E_b]` (3-vector); the momentum state vector is `y = [R2, v2]` (2-vector) — or `[R2, p]` if the code integrates momentum directly. |
| `compute_forces_pure(R2, mShell, Pb, shell_props, params)` takes `Pb` | The transition-phase drive still has a hot-bubble branch. `Pb` must be the **decaying** transition value carried in `y`, never a re-derived Weaver/energy-phase `P_b` (SPEC-024, SPEC-011). |
| `compute_forces_momentum_pure(R2, mShell, Lmech_total, v_mech_total, shell_props, params)` takes **no** `Pb` | Correct: in the momentum phase the interior supplies no pressure (SPEC-012). It takes `(L, v)` because the momentum injection rate must be reconstructed as `ṗ = 2L/v` (SPEC-071) — see S6-C-15, the single most load-bearing coefficient in this slice. |
| `get_ODE_transition_pure(..., c_sound)` | A scalar sound speed is frozen per segment. The only physically motivated use in a *transition* RHS is the drain/venting timescale of the hot interior, `τ_drain ~ R2/c_s` (cf. SPEC-036's vent flux). Units must be pc Myr⁻¹. |
| `create_momentum_snapshot(params, shell_props, mShell, mShell_dot)` | `M_sh` and `Ṁ_sh` are **frozen per segment**, i.e. linearly extrapolated inside the RHS. That is an operator-splitting approximation whose error is controlled *only* by the adaptive segmentation, not by `ODE_RTOL`. See §7. |
| two independent `ForceProperties` classes (L255, L190) | Two sign conventions can silently diverge. The force-closure invariant SPEC-007 is only meaningful if both use the identical convention. |

---

## 1. The momentum equation — full force budget

### 1.1 Variable-mass thin shell (derived)

The shell is a variable-mass system accreting ambient gas **at rest**. Newton's second law for the
open system is

```
    d/dt ( M_sh v2 )  =  ΣF_ext          (material enters carrying zero momentum)
⇔   M_sh dv2/dt       =  ΣF_ext − Ṁ_sh v2 ,     Ṁ_sh = 4π R2² ρ_amb(R2) v2
⇔   M_sh dv2/dt       =  ΣF_ext − 4π R2² ρ_amb(R2) v2²
```

This reproduces SPEC-020 exactly. **The `−4πR2²ρ v2²` term is not an extra force — it is the
`Ṁ v` term.** It must appear **exactly once**, in exactly one of the two forms, never both.

### 1.2 The terms, with sign, dimension, and drop-regime

Internal units are `[M⊙, pc, Myr]` throughout (SPEC-090); every force below is
`M⊙ pc Myr⁻²` (= 6.1623×10²⁴ dyn, SPEC-091).

| Term | Expression | Sign | May be dropped when |
|---|---|---|---|
| **Interior thermal drive** | `F_b = 4πR2² P_b = 2 E_b/R2` (using `P_b = E_b/[2π(R2³−R1³)]`, and `R1≪R2`) | **+** outward | momentum phase only (`P_b→0`). Never in energy/transition. |
| **Direct feedback ram** | `F_ram = ṗ_w + ṗ_SN = 2 L_mech/v_mech` | **+** outward | energy phase (it is `~(3/11)(v2/v_w)` of `F_b`, see §2.2) |
| **Photoionised-gas pressure** | `F_HII = 4πR2² P_HII`, `P_HII = 2.2 n_H k_B T_ion` (SPEC-029) | **+** outward | after `Q_i` collapses (late, post-3–5 Myr) or if `include_PHII=False` |
| **Direct radiation** | `F_rad,dir = (L_bol/c)(1 − e^{−τ_UV})` (SPEC-026) | **+** outward | `τ_UV ≪ 1` (shell transparent) — then it must vanish smoothly, not be clipped to `L/c` |
| **IR-trapped radiation** | `F_rad,IR = τ_IR L_bol/c`, `τ_IR = κ_IR M_sh/(4πR2²)` (SPEC-027) | **+** outward | `τ_IR ≪ 1`, i.e. diffuse/extended shells |
| **Gravity** | `F_grav = G M_sh (M_cluster + M_sh/2)/R2²` (SPEC-031) | **−** inward | never — it is what makes re-collapse possible (SPEC-017) |
| **Ambient/external pressure** | `F_ext = 4πR2² P_ext`, `P_ext = k_B·(PISM in K cm⁻³)` | **−** inward | `PISM = 0` (the TRINITY default) — see §1.4 |
| **Sweep-up ram (the `Ṁv` term)** | `4πR2² ρ_amb(R2) v2²` | **−** inward for `v2>0`; **must be zero for `v2<0`** (§1.5) | never while sweeping |

So the correct assembled RHS is

```
    dR2/dt  = v2
    dv2/dt  = [ 4πR2²( P_drive − P_ext − ρ_amb(R2) v2² ) + F_rad − F_grav ] / M_sh
```

with the phase-aware `P_drive` of SPEC-022:

```
    energy / implicit :  P_drive = max( P_b , P_HII )
    transition        :  P_drive = max( P_b , P_HII + P_ram )
    momentum          :  P_drive =      P_HII + P_ram          ,  P_ram = (ṗ_w+ṗ_SN)/(4πR2²)
```

**Dimensional check (all AU).** `[4πR2²P] = pc²·M⊙ pc⁻¹ Myr⁻² = M⊙ pc Myr⁻²` ✓.
`[G M²/R²] = pc³M⊙⁻¹Myr⁻² · M⊙² / pc² = M⊙ pc Myr⁻²` ✓.
`[L/c] = M⊙ pc² Myr⁻³ / (pc Myr⁻¹) = M⊙ pc Myr⁻²` ✓ (requires `c = 3.066×10⁵ pc Myr⁻¹`, derived
from SPEC-091: `2.998e10 / 9.77781e4`).
`[ρ v²·R²] = M⊙pc⁻³·pc²Myr⁻²·pc² = M⊙ pc Myr⁻²` ✓.

### 1.3 The radiation term: the exact form matters at O(1)

Three defensible forms circulate and they are **not** interchangeable:

```
 (i)   F_rad = (L/c)( 1 − e^{−τ_UV} + τ_IR )      ← PHYSICS_SPEC SPEC-027's form
 (ii)  F_rad = (L/c)( 1 − e^{−τ_UV} )( 1 + τ_IR ) ← only absorbed UV can be reprocessed to IR
 (iii) F_rad = (L/c)( 1 + τ_IR )                  ← assumes τ_UV ≫ 1 already
```

(ii) is the more defensible physics (dust cannot re-emit photons it never absorbed); (i) and (ii)
agree to <1% once `τ_UV > 5`, and differ by a factor `1+τ_IR` when the shell is transparent.
**Form (iii) already contains the direct term as the `1`** — adding a separate
`F_rad,dir = (L/c)f_abs` on top of it double-counts the single-scattering momentum by up to 2×.
That is the classic radiation double-count trap and it must be checked as *one* expression, not two
additive code paths.
Confidence: **high** that a double-count of the "1" is a real class of bug; **medium** on which of
(i)/(ii) is preferable (both appear in the literature I can recall; I cannot check Rahner+17's form).

**Numeric anchor (derived here).** `κ_IR = 4 cm² g⁻¹ = 8.356×10⁻⁴ pc² M⊙⁻¹`
(`1 cm²/g = 2.0890×10⁻⁴ pc²/M⊙`). For `M_sh = 10⁷ M⊙`, `R2 = 10 pc`: `τ_IR = 6.6`; for
`M_sh = 10⁵ M⊙`, `R2 = 10 pc`: `τ_IR = 0.067`. So `τ_IR` spans "negligible" to "dominant" across
TRINITY's own `mCloud` grid — the term cannot be dropped generically, and its **saturation**
(`τ_IR ≫ 1` over-predicts, SPEC-027) matters for the massive end of `paperII_grid_sweep`.

### 1.4 The inward ambient/turbulent pressure — what is *missing* by default

`PISM` defaults to `0 K cm⁻³` (SPEC-003). Inside the cloud (`R2 < r_cloud`) the shell is therefore
confined by **nothing but gravity and sweep-up ram**: the cloud's own thermal and turbulent pressure
are not represented. A physically complete budget would include an inward
`4πR2²(P_th,cloud + ρ_cloud σ_turb²)`. Dropping it is defensible only while
`P_drive ≫ ρ_cloud σ_turb²`; for `n = 10⁵ cm⁻³` and `σ_turb ≈ 2 km s⁻¹`,
`ρσ² = 1.4·1.67e-24·1e5·(2e5)² = 9.4×10⁻⁹ dyn cm⁻²` — that is **larger** than many `P_HII` values
(`2.2·n·k·10⁴` needs `n ≈ 3×10³ cm⁻³` to match). So for dense clouds the neglected turbulent
confinement is not obviously small. Flag as a modelling-scope statement, not a bug.
Confidence: **medium** (σ_turb is not a TRINITY input, so the omission is by design).

### 1.5 Sign trap during re-collapse (derived — I consider this a first-class finding candidate)

For `v2 < 0` and `M_sh = M_enc(R2)`, `Ṁ_sh = 4πR2²ρ v2 < 0`: the shell **sheds** mass, and the shed
material leaves *at the shell velocity*, not at rest. The correct open-system balance is then

```
    d(M v)/dt = F_ext + Ṁ v   ⇒   M dv/dt = F_ext        (no ram term at all)
```

whereas the unconditional expression gives `M dv/dt = F_ext − 4πR2²ρ v2²`, i.e. an **inward** force
even during infall, which spuriously *accelerates* re-collapse. Since `ρ_amb` in a `α = −2` profile
blows up as `R2⁻²` while `F_grav ∝ M_sh(M_*+M_sh/2)/R2²`, the two terms scale comparably and the
spurious term is not negligible. **Expected:** the `Ṁ_sh v2` term is applied only when `Ṁ_sh > 0`
(equivalently `v2 > 0`), or the code integrates `d(Mv)/dt = ΣF` with the shed-momentum term handled
explicitly. Confidence: **high** on the mechanics, **medium** on the numerical magnitude.

---

## 2. The energy → momentum transition

### 2.1 The physical condition

Primary criterion (SPEC-013): the hot interior stops being an energy reservoir,
`L_loss/L_gain → 1`, equivalently `t_cool,bubble < t_dyn = R2/v2`. TRINITY's default fires at a 5%
floor (`phaseSwitch_LlossLgain`), which is a **numerical regularisation of "→0"**, not physics
(SPEC-014).

**A second, threshold-free criterion falls straight out of the four-zone geometry and I think it is
the sharper one.** The wind termination shock sits at `R1 = sqrt(ṗ_w/(4πP_b))` (SPEC-025). Demanding
`R1 ≤ R2` gives

```
    4π R2² P_b  ≥  ṗ_w              (energy-driven bubble exists)
    4π R2² P_b  ≤  ṗ_w              (the shocked-wind region has collapsed onto the shell
                                      ⇒ the wind delivers bare momentum: momentum-driven)
```

This is exact, threshold-free, and coincides with the `max()` branch switch of SPEC-022 when
`P_HII` is added. **Consequence:** the transition phase must end no later than
`4πR2²P_b = ṗ_w + 4πR2²P_HII`, and it must *never* be allowed to integrate past
`4πR2²P_b < ṗ_w` while still evaluating `R1` from `sqrt(ṗ_w/(4πP_b))` — there `R1 > R2`, the bubble
volume `(4π/3)(R2³−R1³)` goes **negative**, and `P_b = E_b/[2π(R2³−R1³)]` flips sign. That is a
hard divergence, not a rounding issue. Confidence: **high** (pure geometry).

### 2.2 How big is the discontinuity if you switch instantaneously? (derived)

In the Weaver limit (SPEC-050/052), `P_b = 5L_w t/(22πR2³)`, so

```
    F_b = 4πR2² P_b = (10/11) L_w t / R2 = (6/11) L_w / v2        (using v2 = (3/5)R2/t)
    ṗ_w = 2 L_w / v_w                                              (SPEC-071)
    ⇒   F_b / ṗ_w  =  (3/11) · v_w / v2  ≈  0.273 v_w/v2
```

With `v_w ≈ 2000 km s⁻¹` and `v2 ≈ 10 km s⁻¹` this is **≈ 55**. So a hard energy→momentum switch at
the cooling-balance instant would drop the driving force by a factor of tens — a violent kink in
`dv2/dt`. **This is precisely why a finite-duration transition phase must exist** (SPEC-016), and
why WARPFIELD's instantaneous switch is described in the literature as a simplification. The
transition phase's job is to drain `E_b` continuously until `F_b` has fallen to `ṗ + 4πR2²P_HII`,
at which point the `max()` hands over with `dv2/dt` continuous **by construction**.
Confidence: **high** (fully derived from SPEC-050/052/071).

### 2.3 The drain, and how long the transition should last (derived, medium confidence)

If the drain is modelled as loss of the hot reservoir on its own sound-crossing time,
`dE_b/dt ⊃ −E_b c_sound/R2` (the fastest a pressure-supported reservoir can decohere; the same
physics as SPEC-036's vent flux with the covering fraction folded in), then `E_b` decays
exponentially with `τ_drain = R2/c_s` and the number of e-folds needed is

```
    N = ln( E_b,0 / E_b,end ) ,  E_b,0 = (5/11) L_w t ,  E_b,end = ṗ R2/2
      = ln( (3/11) v_w / v2 )  ≈ ln(55) ≈ 4
```

With a hot-bubble sound speed `c_s = sqrt(γ k T_b/(μ m_H)) ≈ 480 km s⁻¹ ≈ 490 pc Myr⁻¹` at
`T_b ≈ 10⁷ K, μ ≈ 0.6`, and `R2 = 10 pc`:

```
    Δt_transition ≈ 4 R2 / c_s ≈ 0.08 Myr ,   versus   t_dyn = R2/v2 ≈ 1 Myr
```

**Expected: the transition phase is short — of order 5–20% of the local dynamical time, ≲0.2 Myr
for typical parameters.** If `c_sound` were mistakenly the *shell* sound speed (~10 km s⁻¹ ionized,
~1 km s⁻¹ molecular) the transition would last 50–500× longer and would visibly distort the
phase-timeline figure. This gives a cheap, sharp numerical test on `c_sound`'s identity and units.
Confidence: **medium** (the exponential-drain form is my inference from the `c_sound` argument, not
something I can verify).

### 2.4 What must be conserved, what must be continuous, what must be recomputed

**Conserved across the hand-off:** shell momentum `p = M_sh v2` and shell kinetic energy. There is
no impulsive force at the boundary, so nothing may change discontinuously in the mechanical state.
Total *energy* is explicitly **not** conserved — `E_b` is declared radiated/vented; a correct
implementation records the discarded `E_b` so a global energy audit closes.

| Quantity | Requirement at the transition→momentum boundary |
|---|---|
| `t` (absolute) and cluster age | **continuous** — must not restart at 0; the SPS clock drives `L_mech(t)`, `Q_i(t)`, `L_bol(t)` |
| `R2` | **continuous**, bit-for-bit carried |
| `v2` | **continuous**, bit-for-bit carried |
| `M_sh` | **continuous** — must be the same function `M_enc(R2)` on both sides. If the energy phase uses a shell-structure-integrated mass and the momentum phase uses the profile integral, `M_sh` jumps and (if momentum is carried instead of velocity) `v2` jumps with it |
| `dv2/dt` | **continuous to integrator tolerance** — guaranteed iff the phase ends on the `max()` branch switch (§2.2). This is PHYSICS_SPEC test T13 |
| `E_b`, `P_b`, `R1`, `T0`, bubble `n(r)`/`T(r)` profiles | **dropped**, not carried. In particular `R1` is undefined once `P_b→0`; it must be set to `R2` (or NaN/absent), never evaluated from `sqrt(ṗ/(4πP_b))` |
| shell structure (`τ_UV`, `τ_IR`, ionized/neutral split, `n_shell`) | **recomputed** from the new state — these are algebraic functions of `(R2, M_sh, Q_i, L_bol)`, not integrated state |
| `Ṁ_sh` | **recomputed** from `4πR2²ρ_amb(R2)v2` at the new `R2`, not inherited |

---

## 3. Expected dimensions and where conversion must occur

Internal AU = `[M⊙, pc, Myr]` (SPEC-090). Every argument in both signatures should already be AU:

| Symbol | AU unit | cgs |
|---|---|---|
| `R2` | pc | cm |
| `v2`, `v_mech_total`, `c_sound`, `VELOCITY_THRESHOLD_*` | pc Myr⁻¹ | cm s⁻¹ (`1 pc/Myr = 0.977781 km/s`) |
| `t`, `DT_SEGMENT_*`, `ODE_MIN/MAX_STEP` | Myr | s |
| `mShell` | M⊙ | g |
| `mShell_dot` | M⊙ Myr⁻¹ | g s⁻¹ |
| `Pb`, `P_HII`, `P_ram`, `P_ext` | M⊙ pc⁻¹ Myr⁻² | `×6.4721e-13` → dyn cm⁻² |
| `Lmech_total`, `L_bol` | M⊙ pc² Myr⁻³ | `×6.0255e29` → erg s⁻¹ |
| `ForceProperties` fields, `ṗ` | M⊙ pc Myr⁻² | `×6.1623e24` → dyn |
| `ENERGY_FLOOR`, `Eb` | M⊙ pc² Myr⁻² | `×1.90148e43` → erg |
| `G` | `4.4985e-3 pc³ M⊙⁻¹ Myr⁻²` | |
| `c` | `3.066e5 pc Myr⁻¹` | |
| `κ_IR` | `8.356e-4 pc² M⊙⁻¹` (from 4 cm²/g) | |

**Where conversion must occur:** at the `.param` ingestion boundary only (driven by the `# UNIT:`
annotations, SPEC-090). Inside `compute_forces_*` and `get_ODE_*` there should be **no conversion
factors at all**. Any `1.4`, `1e5`, `3.086e18`, `1.989e33`, `k_B` in cgs, or `/1e5` appearing inside
a hot-loop RHS is either a double conversion or an un-converted literal.

Two conversions that must have already happened before these functions see the values:
1. **`PISM` is `P/k_B` in K cm⁻³, not a pressure** (SPEC-092 #4) — it must be multiplied by `k_B`
   and converted before entering `F_ext = 4πR2²P_ext`. Skipping the `k_B` is a `10¹⁶` error, which
   at least fails loudly; skipping only the unit conversion is a `1.5×10¹²` error.
2. **`ρ_amb = μ_H m_H n_H` with `μ_H = 1.4` (mass per H nucleus), not `μ_ion = 0.61`
   (mass per particle)** (SPEC-092 #1). Using the wrong `μ` is a 2.3× error in the sweep-up ram term
   and in `M_sh`.

**The 2.3% km/s ↔ pc/Myr trap** (SPEC-092 #6) is the most dangerous one in this slice because it is
too small to see: `VELOCITY_THRESHOLD_COLLAPSE` / `_EXTREME` are numbers a human writes thinking in
km/s but the ODE state is in pc/Myr. A 2.3% shift in a segment-refinement threshold is harmless; a
2.3% shift in a *termination* threshold shifts the recorded fate boundary.

---

## 4. Expected asymptotics

### 4.1 Uniform medium, constant `ṗ`, from rest (derived — SPEC-054)

```
    M v = ṗ t ,  M = (4π/3)ρ₀R³   ⇒   (π/3)ρ₀R⁴ = ṗ t²/2
    R2(t) = ( 3 ṗ / (2π ρ₀) )^{1/4} t^{1/2} ,      v2 = R2/(2t) ∝ t^{−1/2}
```

**Numeric anchor (computed here).** `ṗ = 10³² dyn`, `n_H = 10⁴ cm⁻³`, `ρ₀ = 1.4 m_H n_H`,
`t = 1 Myr` ⇒ `R2 = 12.2 pc`, `v2 = 6.1 pc Myr⁻¹ = 6.0 km s⁻¹`. (Same at `n_H = 1`: `R2 = 122 pc`.)
Confidence: **high** (arithmetic shown).

### 4.2 Power-law medium (derived — generalises SPEC-054, matches `exp_mom` in `paper_radiusComparison.py`)

For `ρ(r) = ρ_ref (r/r_ref)^{−w}`, `w = |α| ∈ [0,2]`, `M_sh = B R^{3−w}` with
`B = 4πρ_ref r_ref^w/(3−w)`:

```
    B R^{3−w} Ṙ = ṗ t   ⇒   R^{4−w} = (3−w)(4−w) ṗ t² / ( 8π ρ_ref r_ref^{w} )

    R2 ∝ t^{ 2/(4−w) }              v2 ∝ t^{ (w−2)/(4−w) }
```

| `w = |α|` | `R2 ∝` | `v2 ∝` | note |
|---|---|---|---|
| 0 (uniform) | `t^{1/2}` | `t^{−1/2}` | decelerating |
| 1 | `t^{2/3}` | `t^{−1/3}` | |
| 2 (SIS-like) | `t^{1}` | `t^{0}` | **coasts at constant velocity** — a clean, prefactor-free test |

Check at `w=0`: `R⁴ = 3·4·ṗt²/(8πρ₀) = 3ṗt²/(2πρ₀)` ✓ reduces to §4.1.
Compare with the energy-driven `R ∝ t^{3/(5−w)}` (SPEC-053) and the D-type `t^{4/(7−2w)}`
(SPEC-055): at `w = 0` the ordering is `1/2 (mom) < 4/7 (D-type) < 3/5 (energy)`.
Confidence: **high** (derived; consistent with the published figure's `exp_mom = 2/(4−|α|)`).

### 4.3 Caveats that make a naive `t^{1/2}` check fail legitimately

1. **Non-zero initial momentum.** The momentum phase starts at `(R₀, v₀)` inherited from the
   transition, so `M v = M₀v₀ + ∫ṗ dt`; the `t^{1/2}` law is only the late-time attractor
   (SPEC-054's audit trap). Because the shell leaves the energy phase with `F_b/ṗ ≈ 50` worth of
   accumulated momentum (§2.2), the approach to the asymptote is **slow** — several dynamical times.
2. **`ṗ` is not constant.** SPS `L_mech(t)` and `v_mech(t)` vary by orders of magnitude across the
   SN onset (~3–5 Myr).
3. **`P_HII` is on by default**, which pushes the effective exponent from `1/2` toward `4/7`.
4. **Gravity is on**, which eventually reverses the expansion entirely.
   Therefore PHYSICS_SPEC test T8 is only valid with gravity off, radiation off, `include_PHII=False`,
   constant `ṗ`, and at late `t`.

### 4.4 Stall and escape scales (derived)

Outward-vs-inward balance at `v2 = 0`, momentum phase:

```
    ṗ_tot + (L_bol/c)(f_abs + τ_IR) + 4πR2²P_HII  =  G M_sh (M_cluster + M_sh/2)/R2² + 4πR2²P_ext
```

While the shell is still inside a uniform cloud, `F_grav ∝ M_sh²/R2² ∝ R2⁴` — gravity **grows**
faster than any driving term, so a shell that has not left the cloud will always stall eventually
unless it exits. Once outside (`M_sh ≈ const`), `F_grav ∝ R2⁻²` decays and the shell escapes. So the
qualitative fate is decided at the cloud edge — which is why `R2 > r_cloud` is a tempting but
**wrong** escape test (SPEC-104): the correct test is `R2 > r_cloud` **and**
`v2 > v_esc = sqrt(2G(M_cluster+M_sh)/R2)`.

---

## 5. Termination fates and their correct criteria

| Fate | Correct criterion | Common wrong criterion |
|---|---|---|
| **Re-collapse** | `v2 < 0` sustained **and** bound: `½v2² < G(M_cluster+M_sh)/R2` | a bare radius threshold (`coll_r = 1 pc`) — scale-dependent, fails for large clouds (SPEC-103) |
| **Stall** | `v2 → 0` with net force `≈ 0` **and restoring** (`dF_net/dR2 < 0`) | `v2 → 0` alone, which cannot distinguish a stall from the turning point of a re-collapse. **The discriminator is the sign of the net force at `v2 = 0`**: still inward ⇒ turnaround/re-collapse; ≈0 with restoring gradient ⇒ stall |
| **Dispersal** | shell peak density `≤ n_ISM` sustained for `stop_t_diss` (SPEC-102 Reading A) — or, more standard, shell becomes subsonic/pressure-confined relative to the ambient (`v2 < c_s,ISM`) or fragments | either alone; the two rarely fire together |
| **Escape / blowout** | `R2 > r_cloud` **and** `v2 > v_esc(R2)` | `R2 > r_cloud` alone (SPEC-104) |
| **Feedback exhausted** | end of the SPS table / cluster lifetime | — |
| **Numerical cutoff** | `t > stop_t`, `R2 > stop_r`, **or `MAX_SEGMENTS` exceeded** | silently reporting a numerical cutoff as a physical outcome |

**`MAX_SEGMENTS` deserves its own line.** Both files cap the segment count. Exhausting the cap is a
*numerical* outcome and must be recorded as a distinct termination reason (SPEC-105), never merged
with "completed" or with a physical fate. A run that stops because the segmenter ran out of budget
during a stiff re-collapse and is labelled "collapse" is a silent scientific error.

**`VELOCITY_THRESHOLD_COLLAPSE` / `_EXTREME` / `DT_SEGMENT_COLLAPSE` physics.** During re-collapse
the relevant timescale is `R2/|v2|`, which shrinks toward zero as `R2 → 0` while `F_grav ∝ R2⁻²`
diverges. A fixed segment length cannot resolve that, hence a velocity-triggered fallback to a much
smaller `DT_SEGMENT_COLLAPSE`. **Expected:** the thresholds are compared against the **signed** `v2`
(e.g. `v2 < −VELOCITY_THRESHOLD_COLLAPSE`), in pc Myr⁻¹, with `|_EXTREME| > |_COLLAPSE|`, and
`DT_SEGMENT_COLLAPSE ≤ DT_SEGMENT_MIN`-scale. Comparing `|v2|` instead of `v2` would fire the
collapse path during *fast expansion* too — a regime confusion that silently over-refines (harmless
but expensive) or, if it also flips a fate label, is a correctness bug.

---

## 6. Known traps, ranked for this slice

1. **Ram double-count** (SPEC-020). `−4πR2²ρv2²` is the `Ṁv` term. If the code writes
   `M dv/dt = ... − 4πR2²ρv2²` *and* also subtracts `Ṁ_sh·v2` (available from
   `mShell_dot` in the momentum snapshot!), the drag is doubled. The presence of both
   `mShell_dot` in `create_momentum_snapshot` and `ρ_amb` reachable from `params` makes this
   *mechanically easy to do twice*. Also: the snapshot key `F_ram` sits next to `F_ram_wind`/
   `F_ram_SN` in the output schema (SPEC-006), so `F_ram` there is *feedback* ram, and the sweep-up
   term may have no output slot at all — in which case the force-closure test T2 will not close and
   the discrepancy is the sweep-up term.
2. **Feedback ram double-count.** `P_drive = P_HII + P_ram` already contains `ṗ`. Adding a separate
   `F_ram = ṗ` to the force sum counts the wind momentum twice — a factor-2 over-drive in exactly
   the phase where the driving is the whole story.
3. **`ṗ = L/v` instead of `ṗ = 2L/v`.** From `L = ½Ṁv²`, `ṗ = Ṁv = 2L/v` (SPEC-071). A missing
   factor 2 halves the entire momentum-phase drive and, via §4.1, gives `R2` low by `2^{1/4} = 19%`
   and `M_sh` low by ~60%. This is the single highest-value coefficient check in the slice.
4. **Radiation double-count** (§1.3): `(L/c)(1+τ_IR)` plus a separate direct term.
5. **Energy-driven interior pressure used after the interior cooled.** `compute_forces_pure` takes
   `Pb` as an argument; if any caller supplies a Weaver/energy-phase `P_b` rather than the drained
   transition value, the shell is over-driven by up to the factor `(3/11)v_w/v2 ≈ 50` of §2.2.
6. **`R1` divergence** (§2.1): `R1 = sqrt(ṗ/(4πP_b)) → ∞` as `P_b → 0`; `V_b = (4π/3)(R2³−R1³)`
   goes negative; `P_b = E_b/[2π(R2³−R1³)]` flips sign. Must be guarded or the phase must exit
   first. Related (SPEC-024): whichever `V_b` is used for `P_b(E_b)` must be the same one used for
   the `P_b dV_b/dt` work term in `dE_b/dt`, otherwise the transition leaks energy.
7. **Uniform-medium result on a power-law profile.** `M_sh = (4π/3)ρ₀R2³` and
   `ρ_amb = ρ₀` are wrong for `α ≠ 0`; the profile integral of SPEC-061 must be used, and the
   `ρ_amb` in the sweep-up term must be the **local** `ρ(R2)`, not a mean. With `α = −2` and
   `rCore = 0.01 pc` (the shipped default, SPEC-063) the local density varies by 4 dex over the
   run — freezing it per segment (§7) is then a large error.
8. **Gravity with the wrong enclosed mass.** Three distinct errors: (a) `M_sh` instead of `M_sh/2`
   for self-gravity (factor 2 in the dominant gravity term once `M_sh > M_cluster`, which is always
   at `ε ≤ 0.3`); (b) using `M_cloud` (total) instead of `M_enc(R2)` — over-predicts gravity while
   inside the cloud; (c) including gas *outside* `R2`, which by the shell theorem exerts zero net
   force (SPEC-031).
9. **Density discontinuity at `r_cloud`.** `ρ_amb` jumps from `n_edge` to `n_ISM` there
   (SPEC-060); the sweep-up term and `M_sh` slope jump with it. Without a segment boundary placed at
   the crossing, an adaptive stiff solver will chatter (the `rcloud_smoothing` machinery referenced
   in SPEC-023 exists because of exactly this class of problem).
10. **Equation numbering.** I cannot verify a single Weaver+77 or Rahner+17 equation number
    (PHYSICS_SPEC §0.3, SPEC-045), and the Rahner MNRAS paper and the Rahner thesis number their
    equations differently — a citation copied from one into the other is more likely wrong than
    right. **Any `# eq. N` in this slice must be audited on content, never trusted on number.**
    Likewise, if any Weaver interior prefactor (`1.51e6`, `2.07e6`, `4.02e-3`) is hard-coded, SPEC-045
    shows those quoted pairs are mutually inconsistent with isobaricity by a factor 3–4; the
    prefactor-free structural forms (SPEC-024, SPEC-042) must be preferred.

---

## 7. Numerics: the error that `ODE_RTOL` does *not* control

Both files integrate in **segments** with a **frozen snapshot** (`ODESnapshot`,
`MomentumODESnapshot` holding `shell_props`, `mShell`, `mShell_dot`, `c_sound`). This is operator
splitting. Two independent error sources:

- **Within-segment integration error** — controlled by `ODE_RTOL` / `ODE_ATOL`.
- **Snapshot-freezing (splitting) error** — controlled *only* by `ADAPTIVE_THRESHOLD_DEX`.

The second dominates. A threshold of `0.1 dex` permits a **26%** change in a frozen coefficient
across a segment; `0.05 dex` permits 12%. No ODE tolerance can see that. **Expected:** the dex
threshold is tight enough that the frozen-coefficient error is commensurate with `ODE_RTOL`, and
`ADAPTIVE_MONITOR_KEYS` covers **every** quantity the snapshot freezes that varies fast:
`R2` (⇒ `ρ_amb(R2)` and `M_sh`), `mShell`, `Pb`/`Eb` (transition), the shell optical depths,
`P_HII`, and the SPS drivers `L_mech`/`ṗ` (which jump by ~1 dex at SN onset). A monitor set that
omits the SPS drivers will step straight over the SN turn-on with a stale `ṗ`.

**`compute_max_dex_change(params_before, params_after, keys) -> float` — silent-failure risk.**
A dex (log₁₀ ratio) measure is scale-free but is **undefined for zero and for sign changes**:

- `v2` crosses zero at every turnaround/re-collapse ⇒ `log10(v_after/v_before)` is NaN.
- `Eb → ENERGY_FLOOR` ⇒ ratio → 1 (looks converged) or, if the floor is 0, `−inf`.
- any force component that changes sign.

If the function returns NaN and the caller compares `NaN > ADAPTIVE_THRESHOLD_DEX` (always False),
**the adaptive refinement silently switches off at exactly the stiffest moment of the run.** The
existence of the separate `VELOCITY_THRESHOLD_*` / `DT_SEGMENT_COLLAPSE` path is consistent with
someone having hit this; it does not by itself fix the NaN. **Expected:** zero/sign guards, and a
documented return value for the degenerate case that *refines* rather than *coarsens*
(fail-safe direction).

**Other numerical expectations.**
- `FOUR_PI` must be `4π = 12.566370614359172`, not `4π/3`; it multiplies every pressure→force
  conversion, so an error here is a global force error.
- `DT_SEGMENT_MIN` must be below the shortest physical timescale reached: during re-collapse
  `R2/|v2|` can fall to `1 pc / 100 pc Myr⁻¹ = 0.01 Myr`, so `DT_SEGMENT_MIN ≲ 10⁻³–10⁻⁴ Myr`.
- `ODE_MAX_STEP` must be smaller than the SPS table sampling and than the age-indexed cooling-file
  spacing (SPEC-083), or the solver steps over RHS discontinuities.
- `ODE_MIN_STEP > 0` is an accuracy hazard: if the solver needs a smaller step it will either raise
  or silently accept a larger one; either way the tolerance is no longer met.
- `ODE_METHOD` must tolerate the `max()` kink of SPEC-022/023 — a stiff/implicit method with event
  detection at the branch crossing, not a smooth-RHS explicit method.
- `ENERGY_FLOOR`: the residual driving force from a floored bubble is `F = 2 E_floor/R2`, which
  decays only as `R2⁻¹` — slower than most other terms. For it to be negligible against
  `ṗ ~ 10³² dyn = 1.6×10⁷ M⊙ pc Myr⁻²` at `R2 = 10 pc` requires
  `E_floor ≪ 8×10⁷ M⊙ pc² Myr⁻² ≈ 1.5×10⁵¹ erg`, i.e. `E_floor ≲ 10⁻³ E_b,peak`. A floor set for
  *numerical* reasons (avoiding `log(0)`) must be far below that, and clamping at the floor must
  **terminate** the transition phase rather than sustain a permanent spurious pressure.

---

## 8. The cheapest cross-checks a reconciler can run on this slice

| # | Test | Passes iff |
|---|---|---|
| C1 | Force closure (= SPEC-007 / T2) | `M_sh dv2/dt` equals the recorded force sum in **both** phases, to integrator tolerance. Residual ≈ `4πR2²ρv2²` ⇒ the sweep-up term is missing from the output; residual ≈ `−ṗ` ⇒ it is double-counted |
| C2 | `ṗ = 2L/v` | recorded `pdot_total` equals `2·Lmech_total/v_mech_total` at every snapshot |
| C3 | `dv2/dt` continuity (= T13) | no jump > tolerance at energy→transition→momentum; a jump of factor `~(3/11)v_w/v2` means the switch is instantaneous |
| C4 | Transition duration | `Δt_transition ~ few × R2/c_s,bubble ≈ 0.05–0.2 Myr`, ≪ `R2/v2` |
| C5 | `R1 ≤ R2` | holds at every transition snapshot; `R1 > R2` ⇒ negative bubble volume |
| C6 | Momentum asymptote | with gravity/radiation/`P_HII` off and constant `ṗ`, `d ln R2/d ln t → 2/(4−|α|)`; at `α = −2` the shell coasts at constant `v2` |
| C7 | Self-gravity factor | `F_grav·R2²/(G M_sh)` → `M_cluster + M_sh/2`, not `M_cluster + M_sh` |
| C8 | Collapse ram sign | during `v2 < 0`, the recorded sweep-up term is zero (or the code integrates `d(Mv)/dt` with the shed-momentum term) |
| C9 | Adaptive monitor robustness | `compute_max_dex_change` returns a finite, refine-biased value when any monitored key crosses zero |
| C10 | Termination bookkeeping | `MAX_SEGMENTS` exhaustion produces a distinct `SimulationEndReason`, never a physical fate |

---

## 9. Honest confidence ledger

- **High (derived here, prefactor-safe):** the variable-mass momentum equation and the single
  appearance of `Ṁv`; `ṗ = 2L/v`; `F_b = 2E_b/R2`; `F_b/ṗ_w = (3/11)v_w/v2`; `R1 ≤ R2` as a
  threshold-free transition condition; `R ∝ t^{2/(4−w)}`, `v ∝ t^{(w−2)/(4−w)}` and the full
  power-law prefactor; `F_grav = GM_sh(M_*+M_sh/2)/R2²`; the AU dimension table; the shed-momentum
  sign argument for `v2 < 0`; all unit anchors (`c = 3.066e5 pc/Myr`, `κ_IR = 8.356e-4 pc²/M⊙`).
- **Medium:** which of the three radiation forms is canonical in TRINITY's ancestry; the
  exponential-drain interpretation of `c_sound` and hence the ~0.08 Myr transition duration; the
  numerical magnitude of the collapse-phase ram sign error; `ENERGY_FLOOR`'s target magnitude.
- **Low / refused:** any Weaver+77 or Rahner+17 **equation number** (literature blocked, SPEC-045);
  any Weaver interior *prefactor*; whether Rahner+17 uses `(v_w − v2)` or `v_w` in the momentum
  transfer; whether `P_ram` in the transition-phase `max()` is present in TRINITY's ancestry or is a
  TRINITY addition.

I have deliberately not guessed at any coefficient I cannot derive. Where the reconciler finds the
code disagrees with a **medium** or **low** item above, the correct outcome is "document the choice",
not "bug".

```json
[
  {
    "id": "S6-C-01",
    "file": "trinity/phase1c_transition/run_transition_phase.py",
    "line": "L198",
    "class": "sign",
    "severity": "S1",
    "claim": "The sweep-up ram term -4*pi*R2^2*rho_amb(R2)*v2^2 must appear exactly once in the shell equation of motion; it IS the Mdot*v2 term of the variable-mass momentum balance, not an additional force.",
    "evidence": "d(M_sh v2)/dt = SumF for material accreted at rest => M_sh dv2/dt = SumF - Mdot_sh v2 with Mdot_sh = 4 pi R2^2 rho v2. PHYSICS_SPEC SPEC-020 (AUDIT TRAP), SPEC-007.",
    "expected": "Either the d(Mv)/dt form or the expanded form, never both; and no separate subtraction of mShell_dot*v2 on top of the expanded form.",
    "failure_scenario": "Doubled deceleration: shell radius under-predicted, swept mass under-predicted, re-collapse predicted spuriously early. Silent - no NaN, no warning.",
    "repro": "Force-closure test: M_sh*dv2/dt vs recorded force sum; residual equal to +/- 4*pi*R2^2*rho*v2^2 identifies the sign of the error.",
    "confidence": "high"
  },
  {
    "id": "S6-C-02",
    "file": "trinity/phase1c_transition/run_transition_phase.py",
    "line": "L198",
    "class": "sign",
    "severity": "S2",
    "claim": "When v2 < 0 (re-collapse) the shell sheds mass at the shell velocity, so the Mdot*v2 term must vanish; applying -4*pi*R2^2*rho*v2^2 unconditionally creates a spurious inward force during infall.",
    "evidence": "For mass loss at the system velocity, d(Mv)/dt = F + Mdot*v => M dv/dt = F, with no ram term. For accretion at rest, M dv/dt = F - Mdot*v. Derived; the two cases differ in the sign of Mdot.",
    "expected": "Ram term gated on Mdot_sh > 0 (equivalently v2 > 0), or an explicit shed-momentum term.",
    "failure_scenario": "Re-collapse artificially accelerated. Worst on alpha=-2 profiles where rho(R2) ~ R2^-2 diverges as the shell falls, so the spurious term can rival gravity. Changes the predicted dispersal-vs-recollapse fate.",
    "repro": "Record the sweep-up term through a run that turns around; it should be identically zero for all snapshots with v2 < 0.",
    "confidence": "high"
  },
  {
    "id": "S6-C-03",
    "file": "trinity/phase2_momentum/run_momentum_phase.py",
    "line": "L206",
    "class": "coefficient",
    "severity": "S1",
    "claim": "The momentum injection rate must be pdot_total = 2 * Lmech_total / v_mech_total, not Lmech_total / v_mech_total.",
    "evidence": "L = 0.5 * Mdot * v^2 and pdot = Mdot * v => pdot = 2L/v. PHYSICS_SPEC SPEC-071 (derived). The signature takes exactly (Lmech_total, v_mech_total), so this conversion is the reason both are passed.",
    "expected": "pdot = 2*Lmech_total/v_mech_total, in M_sun pc Myr^-2; equivalently Mdot = 2L/v^2 then pdot = Mdot*v.",
    "failure_scenario": "A missing factor 2 halves the entire momentum-phase drive: R2 low by 2^(1/4) = 19%, swept mass low by ~60% (uniform medium), transition-to-stall time badly wrong. This is the dominant driver in the phase, so the headline result is wrong.",
    "repro": "Check recorded pdot_total == 2*Lmech_total/v_mech_total at every momentum-phase snapshot.",
    "confidence": "high"
  },
  {
    "id": "S6-C-04",
    "file": "trinity/phase2_momentum/run_momentum_phase.py",
    "line": "L206",
    "class": "other",
    "severity": "S1",
    "claim": "The feedback ram momentum must be counted once: either inside P_drive as P_ram = pdot/(4 pi R2^2), or as a standalone force F_ram = pdot - never both.",
    "evidence": "SPEC-022 gives momentum-phase P_drive = P_HII + P_ram with P_ram = (pdot_w + pdot_SN)/(4 pi R2^2). Multiplying that P_drive by 4 pi R2^2 already yields 4 pi R2^2 P_HII + pdot.",
    "expected": "F_total_outward = 4*pi*R2^2*P_HII + pdot_total + F_rad, with pdot appearing once.",
    "failure_scenario": "Factor-2 over-drive in the momentum phase; shell over-expands, re-collapse never happens, dispersal fraction over-predicted across the whole published grid.",
    "repro": "Sum the recorded ForceProperties fields and compare against M_sh*dv2/dt (SPEC-007 closure).",
    "confidence": "high"
  },
  {
    "id": "S6-C-05",
    "file": "trinity/phase2_momentum/run_momentum_phase.py",
    "line": "L206",
    "class": "regime",
    "severity": "S1",
    "claim": "Radiation pressure (direct + IR) must still be included in the momentum phase; it does not depend on the bubble's thermal state.",
    "evidence": "F_rad = (L_bol/c)(1 - exp(-tau_UV) + tau_IR), SPEC-026/027. SPEC-071 notes L_bol/c ~ 3e32 dyn exceeds pdot_w ~ 1e32 dyn by ~3x at early times.",
    "expected": "compute_forces_momentum_pure returns a non-zero F_rad wherever L_bol > 0 and tau_UV or tau_IR is non-negligible.",
    "failure_scenario": "Momentum phase under-driven by up to ~4x at early times; shells that should disperse instead stall or re-collapse. Directly inverts the paper's dispersal-vs-recollapse conclusion in the radiation-dominated corner of the grid.",
    "repro": "Compare F_rad recorded in the last transition snapshot with the first momentum snapshot: it must be continuous, not drop to zero.",
    "confidence": "high"
  },
  {
    "id": "S6-C-06",
    "file": "trinity/phase2_momentum/run_momentum_phase.py",
    "line": "L206",
    "class": "coefficient",
    "severity": "S1",
    "claim": "Gravity must be F_grav = G*mShell*(M_cluster + mShell/2)/R2^2 - the self-gravity factor is mShell/2, and the enclosed mass excludes all gas outside R2.",
    "evidence": "Self-potential of a thin shell U = -G M^2/(2R) => inward self-force G M^2/(2R^2). Shell theorem kills exterior gas. SPEC-031 (derived), same form in WARPFIELD.",
    "expected": "G * mShell * (M_cluster + 0.5*mShell) / R2**2, with G = 4.4985e-3 pc^3 Msun^-1 Myr^-2.",
    "failure_scenario": "mShell instead of mShell/2 is a factor-2 error in the dominant gravity term whenever M_sh > M_cluster (always at sfe <= 0.3): re-collapse predicted far too readily. Using M_cloud instead of mShell over-predicts gravity while the shell is inside the cloud.",
    "repro": "Invert the recorded F_grav: F_grav*R2^2/(G*mShell) must equal M_cluster + mShell/2.",
    "confidence": "high"
  },
  {
    "id": "S6-C-07",
    "file": "trinity/phase2_momentum/run_momentum_phase.py",
    "line": "L206",
    "class": "regime",
    "severity": "S2",
    "claim": "compute_forces_momentum_pure must contain no hot-bubble thermal pressure term; the interior has cooled and supplies none.",
    "evidence": "SPEC-012: momentum-driven means the shocked wind has radiated/vented its thermal energy. The signature correctly takes no Pb argument.",
    "expected": "No Pb, no Eb, no R1-derived pressure reachable in this function; P_drive = P_HII + P_ram exactly.",
    "failure_scenario": "A stale or re-derived Pb leaking into the momentum phase over-drives the shell by up to (3/11)*v_w/v2 ~ 50x (see S6-C-11 derivation) - the classic 'energy-driven interior pressure after the interior cooled' bug.",
    "repro": "Assert Pb and Eb are absent from momentum-phase snapshots, or recorded as 0/NaN and never used in the force sum.",
    "confidence": "high"
  },
  {
    "id": "S6-C-08",
    "file": "trinity/phase1c_transition/run_transition_phase.py",
    "line": "L271",
    "class": "coefficient",
    "severity": "S2",
    "claim": "Radiation pressure must be a single expression combining direct and IR trapping; the '1' in the common form (L/c)(1 + tau_IR) already IS the direct single-scattering term.",
    "evidence": "SPEC-026/027. Forms in circulation: (L/c)(1-exp(-tau_UV)+tau_IR), (L/c)(1-exp(-tau_UV))(1+tau_IR), (L/c)(1+tau_IR). tau_IR = kappa_IR * mShell/(4 pi R2^2), kappa_IR = 4 cm^2/g = 8.356e-4 pc^2/Msun.",
    "expected": "One expression; the direct term must go to zero as tau_UV -> 0; kappa_IR must multiply a MASS column in g/cm^2 (or pc^2/Msun consistently), never a number column.",
    "failure_scenario": "Double-counting the single-scattering term is up to a factor-2 over-drive when tau_IR ~ 1, i.e. exactly the massive-cloud regime (M_sh=1e7 Msun, R2=10 pc gives tau_IR = 6.6). Using a number column for kappa_IR is a ~1e23 error that fails loudly.",
    "repro": "Evaluate F_rad in the optically-thin limit (small mShell, large R2): it must tend to 0, not to L_bol/c.",
    "confidence": "medium"
  },
  {
    "id": "S6-C-09",
    "file": "trinity/phase1c_transition/run_transition_phase.py",
    "line": "L271",
    "class": "state",
    "severity": "S2",
    "claim": "The Pb argument must be the decaying transition-phase bubble pressure derived from the integrated Eb state, and the volume used in Pb = Eb/[2 pi (R2^3 - R1^3)] must be the same volume used in the Pb dV/dt work term of dEb/dt.",
    "evidence": "SPEC-024 (Pb from Eb with gamma=5/3), SPEC-035 AUDIT TRAP (i): a Vb mismatch between the pressure relation and the work term is a direct energy leak.",
    "expected": "One Vb definition, used in both places; F_b = 4 pi R2^2 Pb = 2 Eb/R2 when R1 << R2.",
    "failure_scenario": "Energy non-conservation in exactly the phase whose whole job is to drain energy correctly; transition duration and hence the momentum-phase initial condition are wrong.",
    "repro": "Integrate d(Eb)/dt over the transition and compare against L_gain*dt - work - losses accumulated from the recorded quantities.",
    "confidence": "high"
  },
  {
    "id": "S6-C-10",
    "file": "trinity/phase1c_transition/run_transition_phase.py",
    "line": "L198",
    "class": "divergence",
    "severity": "S1",
    "claim": "R1 = sqrt(pdot_w/(4 pi Pb)) diverges as Pb -> 0; the transition RHS must guard R1 <= R2 (or exit the phase) before the bubble volume (4 pi/3)(R2^3 - R1^3) goes negative.",
    "evidence": "SPEC-025 gives R1 from ram-pressure balance; SPEC-024 gives Pb = Eb/[2 pi (R2^3 - R1^3)]. R1 = R2 exactly when 4 pi R2^2 Pb = pdot_w - which is the physical energy->momentum boundary (derived here).",
    "expected": "A guard R1 = min(R1, R2) plus a phase-exit condition at 4 pi R2^2 Pb <= pdot_w, or Vb evaluated as (4 pi/3) R2^3 with the R1 correction dropped once R1/R2 approaches 1 (documented).",
    "failure_scenario": "Negative bubble volume -> sign-flipped Pb -> a large INWARD 'thermal' force, or NaN/overflow. The run either crashes at the stiffest moment or silently reports an inverted force budget.",
    "repro": "Assert R1 <= R2 at every transition snapshot; check the sign of (R2^3 - R1^3).",
    "confidence": "high"
  },
  {
    "id": "S6-C-11",
    "file": "trinity/phase1c_transition/run_transition_phase.py",
    "line": "L367",
    "class": "state",
    "severity": "S2",
    "claim": "The transition phase must end on the max() branch switch, 4 pi R2^2 Pb <= pdot_total + 4 pi R2^2 P_HII, so that dv2/dt is continuous into the momentum phase by construction.",
    "evidence": "SPEC-022 transition P_drive = max(Pb, P_HII + P_ram); SPEC-016 requires continuity. Derived magnitude of an instantaneous switch: in the Weaver limit F_b = (6/11) L_w/v2 and pdot_w = 2L_w/v_w, so F_b/pdot_w = (3/11)(v_w/v2) ~ 55 for v_w=2000, v2=10 km/s.",
    "expected": "Exit criterion expressed on the force/pressure comparison, not on a fixed time or a fixed Eb fraction; recorded dv2/dt continuous across the boundary to integrator tolerance.",
    "failure_scenario": "A hard switch drops the driving force by a factor of tens in one step - a violent kink that an adaptive solver will either chatter on or step over, and that shows up as an unphysical jog in R2(t) at the phase boundary.",
    "repro": "PHYSICS_SPEC test T13: sample dv2/dt on both sides of the transition->momentum boundary.",
    "confidence": "high"
  },
  {
    "id": "S6-C-12",
    "file": "trinity/phase2_momentum/run_momentum_phase.py",
    "line": "L461",
    "class": "state",
    "severity": "S1",
    "claim": "t (absolute and cluster age), R2, v2 and M_sh must be carried into the momentum phase unchanged; Eb, Pb, R1, T0 and the bubble profiles must be dropped, not carried stale.",
    "evidence": "No impulsive force acts at the phase boundary, so shell momentum M_sh*v2 is conserved. SPEC-016 (continuity), SPEC-006 (output keys), SPEC-021 (M_sh is a function of R2 only).",
    "expected": "Momentum phase initial condition equals the final transition state exactly; the SPS clock is not restarted; M_sh computed from the same M_enc(R2) on both sides.",
    "failure_scenario": "A restarted cluster clock re-injects early-time feedback (Lmech, Qi, Lbol are strongly time-dependent). A different M_sh formula on the two sides jumps v2 if momentum rather than velocity is carried. Both are silent.",
    "repro": "Diff the last transition snapshot against the first momentum snapshot: t, R2, v2, mShell must be identical.",
    "confidence": "high"
  },
  {
    "id": "S6-C-13",
    "file": "trinity/phase1c_transition/run_transition_phase.py",
    "line": "L198",
    "class": "units",
    "severity": "S3",
    "claim": "c_sound must be in pc Myr^-1 and must be the HOT-INTERIOR sound speed if it sets the bubble drain timescale R2/c_s; a shell sound speed would lengthen the transition by 50-500x.",
    "evidence": "Derived: with dEb/dt ~ -Eb*c_s/R2, N = ln((3/11) v_w/v2) ~ 4 e-folds are needed, so dt_transition ~ 4 R2/c_s ~ 0.08 Myr at R2=10 pc, c_s=490 pc/Myr (T_b=1e7 K, mu=0.6) - i.e. ~8% of t_dyn = R2/v2. SPEC-036 uses the same interior c_s = sqrt(gamma Pb/rho_b) for the vent flux.",
    "expected": "c_sound = sqrt(gamma * Pb / rho_b) of the hot interior, in pc/Myr, order 300-1000 pc/Myr; the resulting transition phase is a small fraction of the dynamical time.",
    "failure_scenario": "A shell/ionized sound speed (~10 km/s = 10 pc/Myr) makes the transition last of order the dynamical time, so the shell is driven by a decaying hot bubble long after it should have gone momentum-driven - over-expansion, and a phase-timeline figure with a spuriously wide transition bar.",
    "repro": "Measure the transition-phase duration against R2/v2 at the phase entry; expect a ratio of order 0.05-0.2.",
    "confidence": "medium"
  },
  {
    "id": "S6-C-14",
    "file": "trinity/phase1c_transition/run_transition_phase.py",
    "line": "L143",
    "class": "silent-failure",
    "severity": "S2",
    "claim": "compute_max_dex_change must return a finite, refine-biased value when a monitored key is zero, changes sign, or is non-finite - a log10 ratio is undefined exactly at v2 = 0 and at Eb -> ENERGY_FLOOR.",
    "evidence": "Derived: dex = |log10(after/before)| is NaN for a sign change and -inf/NaN for zero. v2 crosses zero at every turnaround; Eb approaches the floor at the end of every transition phase. NaN > threshold evaluates False in Python, so the refinement branch is never taken.",
    "expected": "Explicit zero/sign guards; the degenerate case returns something that triggers refinement (e.g. +inf), never something that silently skips it.",
    "failure_scenario": "Adaptive segmentation silently switches off at the single stiffest moment of the run (turnaround / end of transition), so the frozen-snapshot splitting error becomes unbounded there. No warning, no NaN in the output - just a wrong trajectory.",
    "repro": "Call with params_before containing v2 = +1 and params_after containing v2 = -1 and assert the result is finite and above the threshold.",
    "confidence": "high"
  },
  {
    "id": "S6-C-15",
    "file": "trinity/phase2_momentum/run_momentum_phase.py",
    "line": "L324",
    "class": "numerical",
    "severity": "S2",
    "claim": "Freezing mShell and mShell_dot per segment is an operator-splitting approximation whose error is controlled ONLY by ADAPTIVE_THRESHOLD_DEX, not by ODE_RTOL; M_sh inside the RHS should ideally be M_enc(R2(t)) evaluated at the current R2.",
    "evidence": "M_sh = M_enc(R2) is algebraic (SPEC-021/061), so nothing needs integrating. A frozen Mdot_sh is a linear extrapolation; for a power-law profile rho ~ R^alpha the local density used in Mdot changes by |alpha| dex per dex of R2, so with alpha=-2 a factor-2 growth in R2 is a factor-4 stale Mdot.",
    "expected": "Either M_sh evaluated from the profile at the current R2 inside the RHS, or a segment length bounded so the induced error is below ODE_RTOL; ADAPTIVE_MONITOR_KEYS must include R2 and mShell.",
    "failure_scenario": "Swept mass drifts from the profile integral, so the mass budget no longer closes (M_sh(r_cloud) != M_cloud) and the gravity and inertia terms are both wrong. Tightening ODE_RTOL does not help, which makes the error invisible to a convergence test.",
    "repro": "Compare integrated mShell at each segment boundary against M_enc(R2) from the density profile; require agreement to <1e-6 relative.",
    "confidence": "high"
  },
  {
    "id": "S6-C-16",
    "file": "trinity/phase1c_transition/run_transition_phase.py",
    "line": "L112",
    "class": "numerical",
    "severity": "S3",
    "claim": "ADAPTIVE_MONITOR_KEYS must cover every quantity the snapshot freezes that can vary fast: R2 (hence rho_amb), mShell, Pb/Eb, the shell optical depths, P_HII, and the SPS drivers Lmech/pdot.",
    "evidence": "SPS mechanical output jumps by ~1 dex at SN onset (~3-5 Myr); age-indexed cooling files make L_cool piecewise-constant in cluster age (SPEC-083). A monitor set omitting the drivers steps over the SN turn-on with a stale pdot.",
    "expected": "The monitored key set is a superset of the snapshot's frozen fields.",
    "failure_scenario": "A single long segment straddling SN onset integrates the whole interval with pre-SN momentum injection - the momentum phase is then under-driven by an order of magnitude for that segment, in the exact window the paper's feedback figure highlights.",
    "repro": "Plot segment boundaries against Lmech(t); no segment should span a >0.1 dex change in Lmech.",
    "confidence": "medium"
  },
  {
    "id": "S6-C-17",
    "file": "trinity/phase1c_transition/run_transition_phase.py",
    "line": "L96",
    "class": "silent-failure",
    "severity": "S2",
    "claim": "Exhausting MAX_SEGMENTS is a numerical outcome and must be recorded as a distinct termination reason, never merged with a physical fate or with 'completed'.",
    "evidence": "SPEC-100 separates 'numerical cutoff' from the five physical fates; SPEC-105 requires termination = {exit_code, outcome, detail} in metadata.json.",
    "expected": "A dedicated SimulationEndReason for segment-budget exhaustion, distinguishable in metadata.json and in the phase-timeline post-processing.",
    "failure_scenario": "A run that ran out of segment budget during a stiff re-collapse gets labelled 'collapse' (or 'completed'), so the published dispersal-vs-recollapse statistics silently include numerical artefacts.",
    "repro": "Set MAX_SEGMENTS to a small value and confirm the recorded outcome is not a physical fate.",
    "confidence": "high"
  },
  {
    "id": "S6-C-18",
    "file": "trinity/phase2_momentum/run_momentum_phase.py",
    "line": "L461",
    "class": "exponent",
    "severity": "S3",
    "claim": "With gravity, radiation and P_HII disabled and constant pdot, the momentum phase must approach R2 ~ t^(2/(4-w)) and v2 ~ t^((w-2)/(4-w)) for rho ~ r^-w; at w=2 the shell coasts at constant velocity.",
    "evidence": "Derived: B R^(3-w) Rdot = pdot t with B = 4 pi rho_ref r_ref^w/(3-w) gives R^(4-w) = (3-w)(4-w) pdot t^2/(8 pi rho_ref r_ref^w). Reduces at w=0 to R = (3 pdot/(2 pi rho0))^(1/4) t^(1/2) (SPEC-054); matches exp_mom = 2/(4-|alpha|) in paper_radiusComparison.py.",
    "expected": "d ln R2/d ln t -> 2/(4-|alpha|) at late times in the stripped configuration.",
    "failure_scenario": "A uniform-medium t^(1/2) hard-coded (or a uniform M_sh = (4pi/3)rho0 R^3 used inside the RHS) gives the wrong radius growth on every alpha != 0 run, i.e. on the whole density-profile figure of the paper.",
    "repro": "Momentum-only run with alpha = 0, -1, -2; fit the late-time log slope. NB the asymptote is approached slowly because the shell inherits large momentum from the energy phase (SPEC-054 audit trap).",
    "confidence": "high"
  },
  {
    "id": "S6-C-19",
    "file": "trinity/phase2_momentum/run_momentum_phase.py",
    "line": "L373",
    "class": "regime",
    "severity": "S3",
    "claim": "rho_amb in the sweep-up term must be the LOCAL density rho(R2) from the cloud profile (and n_ISM beyond r_cloud), not a mean or a core value, and the discontinuity at R2 = r_cloud needs a segment boundary.",
    "evidence": "SPEC-060 gives the piecewise profile; SPEC-021 gives M_sh = M_enc(R2) inside the cloud and M_cloud + (4pi/3) rho_ISM (R2^3 - r_cloud^3) outside. The density is discontinuous at r_cloud unless smoothed.",
    "expected": "Local rho(R2) evaluated per profile branch; an event/segment boundary placed at the r_cloud crossing.",
    "failure_scenario": "A jump in the RHS mid-segment makes the adaptive stiff solver chatter or step over it; the swept mass beyond r_cloud is then wrong, which changes the escape/stall verdict.",
    "repro": "Log the solver step size across the r_cloud crossing; look for step collapse or a jump in mShell slope.",
    "confidence": "medium"
  },
  {
    "id": "S6-C-20",
    "file": "trinity/phase1c_transition/run_transition_phase.py",
    "line": "L97",
    "class": "numerical",
    "severity": "S2",
    "claim": "ENERGY_FLOOR must be small enough that the residual driving force 2*E_floor/R2 is negligible against pdot, and hitting the floor must terminate the transition rather than sustain a permanent spurious pressure.",
    "evidence": "F_b = 4 pi R2^2 Pb = 2 Eb/R2 (derived from SPEC-024 with R1<<R2). For pdot ~ 1e32 dyn = 1.6e7 Msun pc Myr^-2 at R2 = 10 pc, negligibility requires E_floor << 8e7 Msun pc^2 Myr^-2 ~ 1.5e51 erg, i.e. <~ 1e-3 of E_b,peak.",
    "expected": "A floor set for numerical hygiene only (avoiding log/negative-power blowups), several orders below the peak Eb, plus a phase-exit when it is reached.",
    "failure_scenario": "A floored bubble keeps supplying an outward force decaying only as 1/R2 - slower than most other terms - so the 'momentum-driven' shell is permanently over-driven by a fictitious residual bubble. Also: clamping y[2] inside the RHS makes the RHS non-smooth and the solver stalls.",
    "repro": "Check whether Eb sits exactly at the floor for a nonzero number of snapshots while the phase is still 'transition'.",
    "confidence": "medium"
  },
  {
    "id": "S6-C-21",
    "file": "trinity/phase1c_transition/run_transition_phase.py",
    "line": "L106",
    "class": "units",
    "severity": "S3",
    "claim": "VELOCITY_THRESHOLD_COLLAPSE and VELOCITY_THRESHOLD_EXTREME must be in the internal velocity unit pc/Myr and compared against the SIGNED v2, with |EXTREME| > |COLLAPSE|.",
    "evidence": "Internal AU velocity is pc/Myr; 1 km/s = 1.022712 pc/Myr (SPEC-091), a 2.3% offset that is too small to notice (SPEC-092 #6). The collapse path exists because R2/|v2| shrinks toward zero during infall while F_grav ~ R2^-2 diverges.",
    "expected": "Comparisons of the form v2 < -VELOCITY_THRESHOLD_COLLAPSE (signed), values in pc/Myr.",
    "failure_scenario": "Comparing |v2| fires the collapse refinement during fast EXPANSION too (a regime confusion: harmless cost, but wrong if it also tags a fate). A km/s literal compared to a pc/Myr state shifts any fate boundary by 2.3% - invisible, and systematic across a whole grid.",
    "repro": "Check the threshold values against a run's v2 range in both unit systems.",
    "confidence": "medium"
  },
  {
    "id": "S6-C-22",
    "file": "trinity/phase2_momentum/run_momentum_phase.py",
    "line": "L461",
    "class": "regime",
    "severity": "S3",
    "claim": "Re-collapse must be declared on a dynamical criterion (v2 < 0 sustained AND 0.5*v2^2 < G(M_cluster+M_sh)/R2), not on a fixed radius; stall must be distinguished from a turnaround by the sign of the net force at v2 = 0.",
    "evidence": "SPEC-103 (coll_r = 1 pc is a radius threshold, scale-dependent - a 1e9 Msun cloud collapsing from 100 pc to 2 pc has manifestly collapsed but does not trip it); SPEC-032 for the escape/bound condition; SPEC-100 for the fate table.",
    "expected": "Fate assignment from (v2, dv2/dt, net force sign, v_esc), with any radius threshold used only as a numerical floor.",
    "failure_scenario": "Large clouds are never labelled 'collapse' and instead time out on stop_t, biasing the published fate statistics toward 'dispersal' at the high-mass end of the grid.",
    "repro": "Cross-tabulate recorded fates against (min R2, sign of v2 at the end) for the paperII sweep.",
    "confidence": "high"
  },
  {
    "id": "S6-C-23",
    "file": "trinity/phase2_momentum/run_momentum_phase.py",
    "line": "L461",
    "class": "regime",
    "severity": "S3",
    "claim": "Escape must require R2 > r_cloud AND v2 > v_esc = sqrt(2 G (M_cluster + M_sh)/R2); crossing the cloud edge alone is not escape.",
    "evidence": "SPEC-104 (derived); SPEC-032 for v_esc. Inside a uniform cloud F_grav ~ M_sh^2/R2^2 ~ R2^4 grows faster than any driving term, so a shell can cross r_cloud sub-escape and still turn around.",
    "expected": "The escape/blowout fate carries the velocity test; stop_at_rCloud_nSnap is understood as a run-length control, not a physical fate.",
    "failure_scenario": "Shells that would have re-collapsed are recorded as escaped - a direct over-count of cloud dispersal, which is the paper's headline quantity.",
    "repro": "For every run terminated at the r_cloud crossing, compute v2/v_esc at that snapshot.",
    "confidence": "high"
  },
  {
    "id": "S6-C-24",
    "file": "trinity/phase1c_transition/run_transition_phase.py",
    "line": "L271",
    "class": "regime",
    "severity": "S3",
    "claim": "The transition-phase drive must include the direct ram term, P_drive = max(Pb, P_HII + P_ram); compute_forces_pure receives no Lmech/v_mech argument, so it must obtain pdot from params.",
    "evidence": "SPEC-022 gives the phase-aware prescription explicitly, corroborated by the paper_feedback.py docstring quoted there ('only when the non-bubble branch wins max(Pb, P_HII + P_ram)').",
    "expected": "The transition branch compares Pb against P_HII + P_ram, not against P_HII alone.",
    "failure_scenario": "Omitting P_ram makes the transition exit later and hands over at a lower driving pressure, producing a downward step in dv2/dt at transition->momentum and a systematically under-driven momentum phase start.",
    "repro": "At each transition snapshot check which branch of the max() is active and whether P_ram is included in the non-bubble branch.",
    "confidence": "medium"
  },
  {
    "id": "S6-C-25",
    "file": "trinity/phase1c_transition/run_transition_phase.py",
    "line": "L255",
    "class": "other",
    "severity": "S3",
    "claim": "The two independent ForceProperties classes (transition L255, momentum L190) must use an identical sign convention and identical field semantics, and together must satisfy the force-closure invariant.",
    "evidence": "SPEC-007: the recorded forces must reproduce M_sh dv2/dt at every snapshot; the published stacked force-FRACTION plots normalise by F_tot, which presupposes an exhaustive, non-overlapping, consistently-signed set.",
    "expected": "Same convention (e.g. all magnitudes positive with the sum formed explicitly), same field names/meanings, in both phases.",
    "failure_scenario": "A sign or naming divergence between the two dataclasses makes the phase-spanning force-fraction figure mix conventions - a published figure that is wrong only in one phase, which is the hardest kind to spot.",
    "repro": "PHYSICS_SPEC test T2 run separately over the transition and momentum snapshot ranges.",
    "confidence": "medium"
  },
  {
    "id": "S6-C-26",
    "file": "trinity/phase1c_transition/run_transition_phase.py",
    "line": "L98",
    "class": "coefficient",
    "severity": "S1",
    "claim": "FOUR_PI must equal 4*pi = 12.566370614359172 in both files, and must be the same value in both.",
    "evidence": "It converts every pressure to a force via 4 pi R2^2 P and every density to a mass rate via 4 pi R2^2 rho v. A 4 pi/3 would be a factor-3 global error.",
    "expected": "12.566370614359172 in trinity/phase1c_transition L98 and trinity/phase2_momentum L90.",
    "failure_scenario": "A factor-3 error in every pressure-driven force and in the sweep-up rate simultaneously - partially self-cancelling, therefore not obviously catastrophic, therefore easy to miss.",
    "repro": "Direct constant comparison; also check both files agree.",
    "confidence": "high"
  },
  {
    "id": "S6-C-27",
    "file": "trinity/phase2_momentum/run_momentum_phase.py",
    "line": "L206",
    "class": "units",
    "severity": "S2",
    "claim": "The external-pressure term 4 pi R2^2 P_ext must use a real pressure: PISM is declared in K cm^-3 (P/k_B) and must be multiplied by k_B and converted to Msun pc^-1 Myr^-2 before use.",
    "evidence": "SPEC-003 declares PISM in K cm^-3; SPEC-092 #4 flags this as a classic trap; paperII_grid_sweep sweeps PISM to 1e6 K cm^-3 = 1.4e-10 dyn cm^-2, a large confining pressure.",
    "expected": "P_ext = PISM * k_B, converted to AU (k_B = 7.261e-60 Msun pc^2 Myr^-2 K^-1; pressure unit 6.4721e-13 dyn cm^-2). No cgs literal inside the force function.",
    "failure_scenario": "Using PISM raw is a ~1e16 error in the confining pressure - loud. Converting with the wrong factor is a silent 1e12. Because the default is PISM=0, the bug is invisible in every default run and only bites the sweep cells that set it.",
    "repro": "Run with PISM=1e6 K cm^-3 and check the recorded external force equals 4 pi R2^2 * 1.4e-10 dyn cm^-2 in cgs.",
    "confidence": "high"
  },
  {
    "id": "S6-C-28",
    "file": "trinity/phase1c_transition/run_transition_phase.py",
    "line": "L132",
    "class": "numerical",
    "severity": "S3",
    "claim": "ODE_MAX_STEP must be smaller than the SPS table sampling and the age-indexed cooling-file spacing; ODE_MIN_STEP > 0 is an accuracy hazard; ODE_METHOD must tolerate the non-differentiable max() in P_drive.",
    "evidence": "SPEC-083 (age-indexed cooling files make L_cool piecewise-constant in cluster age -> discontinuous RHS); SPEC-023 (max() is not differentiable and an adaptive stiff solver will chatter at the crossing; paper/methods/data/app_LSODA.npz suggests this class of problem was hit).",
    "expected": "A stiff/implicit method with event detection at the branch crossing; ODE_MAX_STEP below the driver sampling; ODE_MIN_STEP = 0 unless justified.",
    "failure_scenario": "Stepping over an RHS discontinuity silently mis-integrates the SN turn-on; a positive min_step means the requested tolerance is quietly not met at the stiffest moments.",
    "repro": "Compare ODE_MAX_STEP against the SPS table dt and check for solver warnings at the max() crossing.",
    "confidence": "medium"
  },
  {
    "id": "S6-C-29",
    "file": "trinity/phase1c_transition/run_transition_phase.py",
    "line": "L93",
    "class": "units",
    "severity": "S3",
    "claim": "DT_SEGMENT_INIT/MIN/MAX and DT_SEGMENT_COLLAPSE are times in Myr, and DT_SEGMENT_MIN must be below the shortest physical timescale reached (R2/|v2| ~ 0.01 Myr during late re-collapse).",
    "evidence": "Internal time unit is Myr (SPEC-090). During infall R2/|v2| -> 0 while F_grav ~ R2^-2 diverges, so the segment length must be able to follow it; DT_SEGMENT_COLLAPSE exists for exactly this.",
    "expected": "Myr units; DT_SEGMENT_MIN <~ 1e-3-1e-4 Myr; DT_SEGMENT_COLLAPSE at or near DT_SEGMENT_MIN.",
    "failure_scenario": "A floor that is too coarse means the collapse trajectory is integrated with a frozen snapshot over many dynamical times - the shell can overshoot to R2 <= 0 or produce non-finite forces.",
    "repro": "Check the minimum realised R2/|v2| in a collapsing run against DT_SEGMENT_MIN.",
    "confidence": "medium"
  },
  {
    "id": "S6-C-30",
    "file": "trinity/phase2_momentum/run_momentum_phase.py",
    "line": "L206",
    "class": "citation",
    "severity": "S4",
    "claim": "Any equation-number citation to Weaver+77 or Rahner+17/19 in this slice is unverifiable and must be audited on content, not on the number; the Rahner MNRAS paper and the Rahner thesis number equations differently.",
    "evidence": "PHYSICS_SPEC §0.3 records that arXiv/ADS/OUP were all 403 for this audit; SPEC-045 explicitly refuses to assert Weaver equation numbers, and shows the two commonly-quoted Weaver interior prefactors (1.51e6 / 2.07e6 K, 4.02e-3 cm^-3) are mutually inconsistent with isobaricity by a factor 3-4.",
    "expected": "Citations checked against the formula content; prefactor-free structural forms (SPEC-024, SPEC-042, and pdot = 2L/v) preferred over any hard-coded literature prefactor.",
    "failure_scenario": "A comment citing 'eq. N' that points at a different equation makes a future maintainer 'fix' correct code to match the wrong reference - or leaves a hard-coded prefactor that cannot be validated at all.",
    "repro": "",
    "confidence": "high"
  }
]
```
