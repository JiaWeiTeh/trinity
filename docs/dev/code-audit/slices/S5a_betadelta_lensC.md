# S5a beta/delta solve — Lens C (what it should be)

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
`phase2/S5a_betadelta/signatures.md` (names + signatures, no values, no bodies) and
`docs/dev/code-audit/reference/PHYSICS_SPEC.md`. I have not read `trinity/`, any stripped copy, any
docstring, or any other agent's output. Everything below is derived here or cited from the
literature from memory. Literature fetch is blocked (arXiv/ADS/OUP 403), exactly as recorded in
PHYSICS_SPEC §0.3 — so I state confidence per claim and I do **not** assert Weaver+77 equation
numbers (same refusal as SPEC-045).

**Slice.** `trinity/phase1b_energy_implicit/get_betadelta.py` — the implicit closure that determines
the two similarity exponents β and δ each timestep of the energy-driven (Phase 1b) evolution.

---

## 0. What this slice is, physically

### 0.1 Why an implicit solve exists at all

The hot bubble (Weaver zone 2, `R1 < r < R2`, SPEC-002) is treated as near-isobaric with a
conduction-mediated interior. To advance the global ODE you need `L_cool`, the bubble's radiative
loss, which requires the interior profiles `T(r)`, `n(r)`, `v(r)`. Those profiles are obtained from
the quasi-self-similar structure equations, whose *coefficients contain the logarithmic time
derivatives of the very state you are solving for*. Specifically, with (SPEC-041)

```
    α ≡ + d ln R2 / d ln t = v2 t / R2
    β ≡ − d ln P_b / d ln t
    δ ≡ + d ln T   / d ln t
```

`α` is **explicit** — `v2`, `t`, `R2` are all known state. `β` and `δ` are **not**: `dP_b/dt`
depends on `Ė_b`, which depends on `L_cool`, which depends on the structure, which depends on
`β, δ`. Likewise `dT/dt`. Hence a 2-D fixed point in `(β, δ)` at every step. That is the whole
reason this module exists, and it is the reason the closure is the stiffest thing in the code.

**Sign convention, and it is a genuine footgun.** `default.param` ships `cool_beta 0.8` and
`cool_delta -6/35`, and SPEC-041 confirms `β = −d ln P_b/d ln t` (positive when `P_b` falls) while
`δ = +d ln T/d ln t` (negative when `T` falls). **β and δ carry opposite sign conventions.** Any
place that "harmonises" them flips a factor of −1 in a first-order term.

### 0.2 Derivation of the structure-equation coefficients (context for what the residual must close)

I derive these because the residual is only meaningful if the ODE it feeds uses the same
convention. Isobaric interior, `P(r,t) = P_b(t)`, ideal γ=5/3 gas, Spitzer conduction
`q = −C T^{5/2} ∇T` (SPEC-043), similarity in `ξ = r/R2` so that `T(r,t) = T_b(t) τ(ξ)`.

Because `T` scales at fixed `ξ`, the Eulerian time derivative picks up an advection term:

```
    ∂T/∂t|_r = (δ/t) T − (α r/t) ∂T/∂r
```

Continuity, `∂ρ/∂t + r⁻² ∂(r²ρv)/∂r = 0`, with `ρ = μ P_b /(k T)` (isobaric ⇒ `ρ'/ρ = −T'/T`):

```
    dv/dr = (β + δ)/t − 2v/r + ( v − α r/t ) (T'/T)                 … (S1)
```

Energy, `∂u/∂t + ∇·[(u+P)v] = ∇·(κ∇T) − Λ_vol` with `u = (3/2)P`:

```
    (1/r²) d/dr ( r² C T^{5/2} T' ) = P_b (β + (5/2) δ)/t
                                      + (5/2)(P_b/T)( v − α r/t ) T'
                                      + Λ_vol                       … (S2)
```

**Consequences that this slice's residual must be consistent with, and that are cheap to test:**

- the combination in (S1) is **`β + δ`** (with the sign conventions of §0.1), which is exactly the
  isobaric statement `d ln n/d ln t = −(β + δ) = −22/35` in the Weaver limit — SPEC-041 checks this
  independently. `β − δ` or `−(β+δ)` is a first-order structural error.
- the combination in (S2) is **`β + (5/2)δ`** = `0.8 − 0.4286 = 0.3714` in the Weaver limit. The
  `5/2` is the enthalpy coefficient `γ/(γ−1)` and is γ=5/3-specific.
- the advective velocity is **`v − αr/t`**, not `v`. Dropping the `−αr/t` silently redefines `δ` as
  an Eulerian-at-fixed-`r` exponent, which is *not* the Weaver δ and would break the SPEC-042
  closure test.

Confidence: **high** on (S1)/(S2) structure (derived here end to end); **high** on the coefficients
`β+δ` and `β+5δ/2`; **medium-high** that the code's structure module is the one that owns these
(this slice only supplies β, δ to it).

### 0.3 The generalised Weaver asymptote — the single most useful number I can give this audit

PHYSICS_SPEC quotes the uniform-medium values `β = 4/5`, `δ = −6/35`. TRINITY runs power-law clouds
(`densPL_alpha ∈ [−2, 0]`, SPEC-003/060). Derive the correct asymptote for `ρ ∝ r^{−w}`, `w = |α_ρ|`:

From SPEC-053, `R2 ∝ t^η` with `η = 3/(5−w)`, and `E_b = L_w t/(1+2η)` so `E_b ∝ t` for constant
`L_w`. Then `P_b ∝ E_b/R2³ ∝ t^{1−3η}`, hence

```
    β_Weaver(w) = 3η − 1 = (4 + w)/(5 − w)
    δ_Weaver(w) = (2/7)(2η − β − 1) = −(2/7) η = −6 / (7(5 − w))
```

Checks: `w = 0 → β = 4/5, δ = −6/35` ✓ (SPEC-041); `w = 1 → β = 5/4, δ = −3/14`;
`w = 2 → η = 1, β = 2, δ = −2/7` ✓ (and `E_b/(L_w t) = 1/3`, SPEC-053 ✓).

**Therefore:** for the steepest supported profile the physical `β` asymptote is **2.0**, 2.5× the
`cool_beta 0.8` default guess, and `δ` is **−2/7 = −0.286**, 1.67× the default. Any admissible box
`[BETA_MIN, BETA_MAX]` narrower than `[≲0, ≳2]` **clips the physical root** for `densPL_alpha = −2`
runs, and any solver that reports a clipped box corner as converged is producing a wrong `L_cool`
silently. Near the energy→momentum transition `P_b` collapses faster than any power law, so `β` must
be allowed to run well above 2 (I would want ≥ 5–6 of headroom).

Confidence: **high** (derived here, cross-checked against three independent SPEC entries).

---

## 1. Per-function derivations

Unit system throughout: TRINITY AU = `[M⊙, pc, Myr]` (SPEC-090/091).

### 1.1 `L182 cool_beta_to_Ebdot_pure(beta, Pb, t_now, R1, R2, v2, Eb, pdot_total, pdotdot_total) -> float`

**What it must be.** The purely *kinematic* map from a trial `β` to the bubble energy rate `Ė_b`
that trial implies. No physics of cooling enters; this is the chain rule on the definition of `β`.

Take `V_b = (4π/3)(R2³ − R1³)` and, for γ=5/3, `E_b = (3/2) P_b V_b` ⇒
`P_b = E_b / [2π(R2³ − R1³)]` (SPEC-024). Differentiate:

```
    Ė_b = (3/2)( Ṗ_b V_b + P_b V̇_b )
        = − β E_b / t  +  (3/2) P_b V̇_b ,      V̇_b = 4π( R2² v2 − R1² Ṙ1 )
```

`Ṙ1` follows from ram-pressure balance `R1 = sqrt( ṗ_tot /(4π P_b) )` (SPEC-025) — note **any
constant prefactor (the 3/4 strong-shock ambiguity) cancels in the logarithmic derivative**:

```
    Ṙ1/R1 = ½ ( p̈_tot/ṗ_tot − Ṗ_b/P_b ) = ½ ( p̈_tot/ṗ_tot + β/t )
```

Hence the expected return value:

```
    Ė_b(β) = 6π P_b ( R2² v2 − R1² Ṙ1 ) − β E_b / t
           = 6π P_b R2² v2  −  3π P_b R1³ (p̈_tot/ṗ_tot)
             −  (β/t) ( E_b + 3π P_b R1³ )
```

with `3π P_b R1³ ≡ (3/2) E_b R1³/(R2³−R1³)`. For general γ replace `6π → 4π/(γ−1)` and
`3π → 2π/(γ−1)`.

**This explains the signature exactly.** `pdotdot_total` has no other role: it is needed *only* for
`Ṙ1`, and `Ṙ1` is needed *only* if `V_b` retains `R1³`. Its presence is strong evidence the intended
volume convention is `(R2³ − R1³)`, not Weaver's `R2³` shortcut. Both `Pb` and `Eb` being passed is
redundant (`P_b = E_b/[2π(R2³−R1³)]`) and is therefore a **consistency hazard**: if the caller's
`P_b` was formed with `R2³` only while this algebra assumes `R2³−R1³`, the identity is violated by
`O((R1/R2)³)` — negligible early, `O(1)` exactly when `E_b → 0` and `R1 → R2`, i.e. precisely at the
transition where `β` matters most (SPEC-024 audit trap).

**Dimensions.** `β` [–]; `Pb` [M⊙ pc⁻¹ Myr⁻²]; `t_now`, `R1`, `R2` [Myr, pc, pc]; `v2` [pc Myr⁻¹];
`Eb` [M⊙ pc² Myr⁻²]; `pdot_total` [M⊙ pc Myr⁻²]; `pdotdot_total` [M⊙ pc Myr⁻³]; **return**
[M⊙ pc² Myr⁻³] (luminosity, `1 AU = 6.0255e29 erg s⁻¹`).

**Exact numerical anchor (Weaver limit, `R1 → 0`, uniform medium, constant `L_w`).** Using
SPEC-050/051/052: `v2 = (3/5)R2/t`, `E_b = (5/11)L_w t`, `P_b = E_b/(2πR2³)`, `β = 4/5`:

```
    6π P_b R2² v2 = 6π · [5 L_w t/(22π R2³)] · R2² · (3R2/5t) = (9/11) L_w
    β E_b/t       = (4/5)(5/11) L_w         = (4/11) L_w
    ⇒ Ė_b = (9/11 − 4/11) L_w = (5/11) L_w = 0.454545 L_w   ✓ SPEC-051
```

The function reproducing `5/11 · L_w` on these inputs is a complete, unit-free verification of both
the prefactor and the sign. **If the β sign were flipped**, the result would be `13/11 L_w` — a
factor 2.6 error in the bubble's energy budget, immediately visible as `E_b/(L_mech t) ≠ 0.4545`.

**Validity / breakdown.** (i) `ṗ_tot → 0` (the wind–SN gap, or an SPS table with a momentum
minimum) makes `p̈/ṗ` diverge; the term must be guarded, not merely allowed to produce `inf`.
(ii) `R1 ≥ R2` makes `R2³−R1³ ≤ 0` ⇒ negative volume, negative `P_b` — unphysical and must be
trapped. (iii) `t_now → 0` divides by zero. (iv) γ is not in the signature, so `6π = (3/2)·4π`
appears to be γ=5/3-hardwired here while `compute_R1_Pb` accepts `gamma_adia` — if `gamma_adia` is
ever ≠ 5/3, the two are inconsistent and the kinematic identity silently breaks.

**Consistency requirement on `ṗ_tot`.** `compute_R1_Pb` receives `(Lmech_total, v_mech_total)` and
must build `ṗ = 2 L_mech/v_mech` (SPEC-071). The `pdot_total` fed *here* must be **the same
quantity**. If one path uses `2L/v` and the other uses the SPS table's `pdot_W + pdot_SN`, `Ṙ1` is
the derivative of a different `R1` than the one being used, and the identity is broken by an
uncontrolled amount whenever the wind and SN channels have different effective velocities.

Confidence: **high** on the expression and on the `5/11` anchor; **high** on the traps.

### 1.2 `L272 delta2dTdt_pure(t, T, delta) -> float`

**What it must be.** Straight inversion of the definition `δ = d ln T/d ln t`:

```
    dT/dt = δ · T / t
```

**Dimensions.** `t` [Myr], `T` [K], `δ` [–] ⇒ return [K Myr⁻¹].

**Sign.** With `δ = −6/35 < 0` and `T > 0`, `dT/dt < 0` — the bubble cools as it expands. **There
must be no minus sign** (that would be the β convention, §0.1).

**Validity / traps.**
- `t` must be the *same clock* used for `α = v2 t/R2` and for `β`, i.e. the cluster age measured
  from feedback onset (the SPS table's `t = 0`). If Phase 1b restarts its own clock, all three
  exponents are measured about the wrong origin and the SPEC-042 closure and SPEC-056's
  `α → 0.6` test both fail systematically. Testing `α = v2 t/R2 → 0.6` early in a uniform-density
  energy-phase run is the cheap detector.
- `t = 0` and `T ≤ 0` must be excluded.
- `T` here must be the temperature at the *same reference point* whose evolution `δ` describes. If
  `δ` is the exponent of the similarity amplitude `T_b` but `T` passed in is `T0` measured at
  `ξ = bubble_xi_Tb = 0.98`, the two differ by the fixed factor `(1−ξ)^{2/5} = 0.209`
  (SPEC-040) — **that factor is time-independent and therefore cancels in `d ln T/d ln t`**, so
  using `T0` is legitimate *provided* `ξ` is held fixed. If `ξ` is adaptive (e.g. moved when the
  integration struggles), the cancellation fails and `δ` acquires a spurious `d ln(1−ξ)/d ln t`
  contribution. Worth checking.

Confidence: **high** on the expression; **high** on the ξ-cancellation argument.

### 1.3 `L297 compute_R1_Pb(R2, Eb, Lmech_total, v_mech_total, gamma_adia) -> (R1, Pb)`

**What it must be.** Two coupled relations:

```
    (i)   P_b = (γ−1) E_b / V_b ,  V_b = (4π/3)(R2³ − R1³)
                 ⇒ P_b = 3(γ−1) E_b / [ 4π (R2³ − R1³) ]  →  E_b/[2π(R2³−R1³)] at γ=5/3
    (ii)  ρ_w(R1) v_w² = P_b ,  ρ_w = Ṁ_w/(4π R1² v_w)
                 ⇒ R1 = sqrt( ṗ_tot / (4π P_b) )
    with  ṗ_tot = 2 L_mech,tot / v_mech,tot   (SPEC-071: L = ½Ṁv², ṗ = Ṁv)
                 ⇒ R1 = sqrt( L_mech,tot / ( 2π v_mech,tot P_b ) )
```

These are **coupled** (`P_b` needs `R1`, `R1` needs `P_b`), so a correct implementation either
iterates/roots the pair, or explicitly adopts the Weaver `R1 ≪ R2` approximation `P_b = 3(γ−1)E_b/
(4πR2³)` and then computes `R1` from it. The second is fine early and **wrong near the transition**
(SPEC-024 trap). Which one is used must be the same convention that `cool_beta_to_Ebdot_pure`'s
algebra assumes (§1.1).

**Dimensions.** `R2` [pc]; `Eb` [M⊙ pc² Myr⁻²]; `Lmech_total` [M⊙ pc² Myr⁻³];
`v_mech_total` [pc Myr⁻¹]; `gamma_adia` [–] ⇒ `R1` [pc], `Pb` [M⊙ pc⁻¹ Myr⁻²]. Dimension check on
(ii): `[L/v] = M⊙ pc Myr⁻²` (force ✓), `/[P] = pc²`, `sqrt → pc` ✓.

**`AMBIGUOUS` — the 3/4.** SPEC-025 records it: the strict post-shock pressure of a strong γ=5/3
shock is `(2/(γ+1))ρv² = (3/4)ρ_w v_w²`, giving `R1 = sqrt(3 ṗ/(16π P_b))`, i.e. `0.866×` smaller.
Weaver and most descendants drop it. Either is defensible; the audit should record which, because
`R1³` enters `V_b` and hence `P_b`. Note the ambiguity **does not** affect `Ṙ1/R1` (§1.1).

**Asymptotics and required guards.**
- `P_b ∝ E_b` (linear, monotone increasing), `R1 ∝ P_b^{−1/2} ∝ E_b^{−1/2}` (monotone decreasing).
- As the energy phase ends, `E_b → small` ⇒ `R1 → ∞`. **`R1 ≥ R2` is guaranteed to occur** if the
  energy phase is allowed to run long enough. This is not a numerical accident, it is the physical
  statement "the bubble can no longer stand off the wind" — it must be detected and converted into
  an end-of-energy-phase / no-physical-root outcome, never allowed to produce `R2³−R1³ < 0`.
- `L_mech,tot → 0` ⇒ `R1 → 0`, fine. `v_mech,tot → 0` ⇒ `R1 → ∞`; guard.
- `E_b ≤ 0` ⇒ `P_b ≤ 0` ⇒ `R1` imaginary; guard.

**γ threading.** `γ` enters `(γ−1)` here, the `5/2` enthalpy coefficient in (S2), the `3/2` in
`E_b = (3/2)P_bV_b`, and the `6π` in §1.1. A `gamma_adia` argument that is honoured in only one of
those makes γ a fake parameter — S2/latent, not results-wrong at the default.

Confidence: **high** on (i) and on the structure of (ii); **medium** on which of the coupled/
approximate readings is intended; **high** on the guards.

### 1.4 `L334 effective_Lloss(mode, fmix, theta_target, Lcool, Lleak, Lmech)` and `L360 …_from_params`

This is the SPEC-015 patch layer: 1-D Spitzer conduction under-predicts bubble energy loss relative
to 3-D turbulent mixing across a fractal contact discontinuity (El-Badry+19; Lancaster+21a,b), so
TRINITY exposes `cooling_boost_mode / _fmix / _theta / _kappa / _fA`.

**What it must be, by mode (inferred from the physics the knob names encode):**

```
    none        :  L_loss = L_cool + L_leak                       (exactly, bit-identically)
    multiplier  :  L_loss = f_mix · L_cool + L_leak
    theta       :  L_loss chosen so that L_loss / L_mech = θ_target
```

**Required properties, independent of which reading is right:**
1. `L_loss ≥ 0`, and `L_loss` monotone non-decreasing in each of `L_cool`, `L_leak`, `f_mix`.
2. `mode = none` must be an exact identity — SPEC-015 says `none` is the default and the published
   default physics; per CLAUDE.md rule 5 this is a "free win"-class identity that should be
   bit-identical, not merely close.
3. The boost models **radiative/mixing loss at the contact discontinuity**. It must therefore
   multiply `L_cool` only. Multiplying `L_leak` (a venting enthalpy flux, SPEC-036, a completely
   different mechanism) or the `P dV` work term (mechanical, not thermal) is a physics error.
4. **The same `L_loss` must be used in the residual and in the energy→momentum transition trigger**
   `(L_gain − L_loss)/L_gain ≤ 0.05` (SPEC-013/014). If the bubble evolves with boosted cooling but
   transitions on unboosted cooling (or vice versa), the headline prediction — the transition time —
   decouples from the dynamics. This is the single highest-value cross-check in this function.
5. `theta` mode has a structural pathology worth flagging: if `L_loss ≡ θ_target · L_mech` exactly,
   then `(L_gain − L_loss)/L_gain` is *pinned* at `1 − θ_target/η_th` for all time, so the
   `cooling_balance` trigger either fires on the first step (`θ_target ≳ 0.95`) or never fires
   (`θ_target ≲ 0.95`). A sane implementation would take
   `L_loss = max(L_cool + L_leak, θ_target L_mech)` or interpolate, not pin.

**Dimensions.** all luminosities [M⊙ pc² Myr⁻³]; `fmix`, `theta_target` [–]; return
[M⊙ pc² Myr⁻³].

Confidence: **high** on properties 1–4; **low-medium** on the exact algebra of the `theta` mode
(I am inferring mode semantics from names, and `cooling_boost_kappa`/`_fA` are not in this
signature list at all, so this function may not be the whole boost path).

### 1.5 `L374 _usable_dMdt(props) -> Optional[float]`

`Ṁ` here must be the conduction-driven evaporation rate of shell gas into the bubble (SPEC-044,
Cowie & McKee 1977 classical evaporation, `Ṁ_evap = 16π μ C T_h^{5/2} R /(25 k_B)`), used as the
**shooting seed** for the inner structure integration. The inner boundary conditions are mass-flux
matching:

```
    at R1 :  4π R1² ρ(R1) ( v(R1) − Ṙ1 )  =  + Ṁ_wind
    at R2 :  4π R2² ρ(R2) ( v(R2) − v2 )  =  − Ṁ_evap
```

**Required properties.**
- A seed may change the *iteration path* but must **not** change the accepted root. Testable:
  perturb the seed by ±20 % and require the returned `(β, δ)` to agree to the solver tolerance. If
  they do not, the run is history-dependent and not reproducible — a serious result-integrity issue
  that would also break the CLAUDE.md rule-5 "bit-identical" gate.
- Rejecting **negative** `Ṁ` as "unusable" forecloses a physically real regime: in a strongly
  radiative bubble the conduction front reverses and hot gas *condenses* onto the shell
  (`Ṁ < 0`). That is exactly the late energy phase this module has to model. If negative values are
  filtered out and replaced by a positive default, the late-phase structure is biased.
- `Ṁ` must be finite and the units [M⊙ Myr⁻¹].

Confidence: **medium-high** on the boundary conditions; **medium** on the condensation point
(literature-blocked; I recall the sign reversal but cannot pin where WARPFIELD draws the line).

### 1.6 `L393 get_residual_pure(beta, delta, params, return_bubble_props, dMdt_guess) -> (float, float, props)`

**What the residual *should* be — the two physical balances being enforced.**

**Residual 1 — bubble energy balance (fixes β).** The kinematic `Ė_b(β)` of §1.1 must equal the
`Ė_b` the physics actually delivers (SPEC-035):

```
    Ė_b^{phys} = η_w L_mech,w + η_SN L_mech,SN
                 − L_loss( β, δ )                       [ = effective_Lloss, §1.4 ]
                 − P_b · 4π R2² v2                      [ work on the shell ]
                 + P_b · 4π R1² Ṙ1                      [ work at the inner face ]

    r_β = Ė_b^{phys}(β, δ) − Ė_b^{kin}(β)
```

**The `P dV` work term must appear exactly once, and on exactly one side.** Two framings are
algebraically equivalent and both correct:

```
    (A)  Ė_b^{kin} = (3/2) P_b V̇_b − β E_b/t   vs   Ė_b^{phys} = L_gain − L_loss − P_b V̇_b
    (B)  equivalently   L_gain − L_loss = (5/2) P_b V̇_b − β E_b/t
```

— note `(3/2) + 1 = 5/2`, the enthalpy coefficient, exactly as in (S2). Mixing the framings (e.g.
comparing `Ė_b^{kin}` against `L_gain − L_loss` with the work term omitted) is a **first-class,
silent energy-conservation bug** and it biases in a definite direction: omitting `−P_b V̇_b` makes
`Ė_b^{phys}` too large, drives the solved `β` too small, keeps `P_b` artificially high, and
**delays the energy→momentum transition**. Conversely double-counting it accelerates the
transition. Either way the headline result moves. Cheapest detector: SPEC-051's `E_b/(L_mech t) →
5/11 = 0.4545` in a gravity-/radiation-free run — omitting the work term gives a systematically
larger ratio.

**Residual 2 — temperature-evolution consistency (fixes δ).**

```
    r_δ = (dT/dt)^{structure}(β, δ) − (dT/dt)^{definition}(δ)
        = (dT/dt)^{structure}      − delta2dTdt_pure(t, T, δ)
```

i.e. the temperature the structure integration produces at the reference point must evolve at the
rate the trial `δ` asserts. `delta2dTdt_pure` is the *definition* side of this residual; that is
why it exists as a separate pure function.

**Structural cross-check that must hold at the solution (SPEC-042, `T_b^{7/2} = a P_b R2²/(Ct)`):**

```
    δ ≈ (2/7)( 2α − β − 1 )      with α = v2 t / R2
```

throughout the conduction-dominated energy phase, degrading only as radiative cooling becomes
comparable to conduction. This is the audit's test T5 and it is prefactor-free. It also tells you
the expected Jacobian structure: `r_β` is dominated by `∂Ė^{kin}/∂β = −(E_b/t)[1 + (3/2)R1³/
(R2³−R1³)] < 0`, and `r_δ` is dominated by `−T/t < 0` from the definition side. **The Jacobian is
therefore diagonally dominant and the root should be unique and easy** — a well-seeded Newton /
`hybr` should converge in well under ten iterations from the previous step's `(β, δ)`.

**That last point is load-bearing for the audit.** The presence of a three-deep solver cascade
(`grid` → `hybr` → `L-BFGS-B`) plus a legacy path plus a rescue path is *not* evidence that the
physics problem is hard. It is evidence that either (a) the residual is **noisy** — the inner ODE /
shooting tolerance is looser than the outer `RESIDUAL_THRESHOLD`, so the outer solve is chasing
integration noise and can never converge; or (b) the residual is **non-smooth** — table
interpolation kinks in `Λ(T)` (SPEC-083's per-age *file* selection is piecewise-constant in cluster
age!) or a `max()` branch (SPEC-023) puts a kink in the residual surface. Both are fixable at the
source; neither is fixed by adding another minimiser. I would rank "measure the residual's noise
floor and compare it to `RESIDUAL_THRESHOLD`" as the highest-value single experiment in this slice.

**Dimensions and the normalisation trap.** If `r_β` is a difference of energy rates it is
[M⊙ pc² Myr⁻³]; if `r_δ` is a difference of `dT/dt` it is [K Myr⁻¹]. For a `10⁶ M⊙` cluster,
`L_mech ~ 10⁴⁰ erg s⁻¹ ≈ 1.7×10¹⁰` AU, while `dT/dt ~ δ T/t ~ −0.17 × 10⁷ K / 1 Myr ≈ −1.7×10⁶`
K/Myr. **They differ by ~10⁴.** Any scalar `g_total` formed as `sqrt(r_β² + r_δ²)` or `|r_β| +
|r_δ|` on the raw values is therefore *entirely* determined by `r_β`, and `δ` is effectively
unconstrained — the solver would report convergence with an arbitrary `δ` inside the noise. The
residuals **must** be non-dimensionalised before combination, e.g.

```
    r̂_β = ( Ė_b^{phys} − Ė_b^{kin} ) / max( |L_gain| , |L_loss| , |P_b V̇_b| )
    r̂_δ = ( δ_implied − δ_trial )          [ already dimensionless and O(1) ]
```

or, equivalently and more cleanly, form both residuals directly in exponent space
(`β_implied − β_trial`, `δ_implied − δ_trial`), which is dimensionless, O(1)-scaled, commensurate,
and makes `RESIDUAL_THRESHOLD` mean something scale-free.

Confidence: **high** that these are the two balances; **high** on the work-term uniqueness
requirement; **high** on the normalisation argument (arithmetic).

### 1.7 `L47 RESIDUAL_THRESHOLD` — what a physically sensible tolerance is

The downstream quantity that matters is the transition time, triggered at
`(L_gain − L_loss)/L_gain ≤ 0.05` (SPEC-013/014). Propagating: to place the transition time to
~1 % you need the loss fraction resolved to ~`5×10⁻⁴` absolute, i.e. a **relative** residual on
`Ė_b` of `~10⁻³` or tighter. So:

- **Physically right:** relative residual `1e-4 … 1e-3` (or, in exponent space, `|Δβ| ≲ 1e-3`).
- **Too loose:** `≥ 1e-2` relative — the closure error then aliases directly into the 0.05 trigger
  and the transition time becomes a numerical artefact.
- **Too tight / unattainable:** `≤ 1e-8` relative, given an inner ODE integration and an inner
  shooting solve; the outer solve will then *always* fail and *always* fall through the cascade,
  which is silent degradation dressed up as robustness.
- **Necessary condition:** inner ODE `rtol` at least one order tighter than the outer threshold.
- **Must be relative, not absolute.** An absolute threshold in AU luminosity units is meaningless
  across the shipped sweep, which spans `mCloud` `10⁴ → 5×10⁹ M⊙` (SPEC-073) — six orders of
  magnitude in `L_mech`. The same absolute threshold is 10⁻⁶ relative at one end of the grid and
  10⁰ at the other.

Confidence: **medium-high** (derivation is mine, the 0.05 anchor is from SPEC-014).

### 1.8 `L41–44 BETA_MIN/BETA_MAX/DELTA_MIN/DELTA_MAX` — the admissible box

From §0.3 and the closure `δ = (2/7)(2α − β − 1)`:

| regime | α | β | δ |
|---|---|---|---|
| uniform Weaver (`densPL_alpha = 0`) | 3/5 | **4/5** | **−6/35 = −0.171** |
| `densPL_alpha = −1` | 3/4 | **5/4** | **−3/14 = −0.214** |
| `densPL_alpha = −2` | 1 | **2** | **−2/7 = −0.286** |
| rising `L_mech` / SN onset | — | can go **< 0** briefly (`P_b` rising) | can go **> 0** |
| approaching the transition | — | grows well **above 2** | more negative |

**Requirements.**
- The box must contain `β ∈ [≲ 0, ≳ 5]` and `δ ∈ [≲ −1, ≳ +0.5]`. A `β` cap below 2 clips every
  `densPL_alpha = −2` run; a `δ` floor above −0.3 clips them too.
- **Hitting a bound must never be reported as convergence.** A clipped `(β, δ)` produces a wrong
  `L_cool` with a residual that is *not* zero; if the result object says `converged = True`, the
  run is quietly wrong. The correct behaviour is: flag it, and treat a persistent bound-hit as
  either "reduce `dt` and retry" or "no physical root ⇒ end the energy phase".
- Clipping bias is directional: clipping `β` from above holds `P_b` too high ⇒ shell pushed too
  hard ⇒ `R2(t)` too large and the transition delayed.

Confidence: **high** on the asymptote table (derived, three independent cross-checks); **medium**
on my suggested numerical margins (judgement, not literature).

### 1.9 The solver stack — `L612`, `L625`, `L649`, `L678`, `L869`, `L879`, `L909`, `L932`, `L948`, `L974`, `L1010`, `L1108`, `L1152`

**What the solve must do on non-convergence for the run to remain trustworthy.** In order:

1. **Never return a non-root as a root.** `BetaDeltaResult.converged` must be `False` and the
   achieved residual must be carried, not discarded. `_no_root_result(beta_guess, delta_guess,
   reason)` returning *the guess* is only acceptable if it is unambiguously marked non-converged and
   the caller acts on it.
2. **The caller must act.** Acceptable reactions, in increasing severity: (a) shrink `dt` and retry
   — correct, because the closure is a smooth function of state and a smaller step means a better
   seed; (b) declare `_NoPhysicalRoot` ⇒ **end the energy phase / hand over to transition** — this
   is *physically* meaningful, since the disappearance of a root usually means the bubble can no
   longer sustain a Weaver structure (`R1 → R2`, or `L_loss > L_gain` with no consistent `β`);
   (c) abort with a recorded termination reason. **Unacceptable:** silently continuing with the
   guess, or with the previous step's `(β, δ)`, for an unbounded number of steps.
3. **Falling back to the `cool_beta 0.8 / cool_delta −6/35` defaults is a directional bias, not a
   neutral choice.** Those are the *uniform-medium Weaver* values. Late in the energy phase the true
   `β` is well above `4/5`; substituting `4/5` under-states the pressure decline, over-states
   `P_b`, over-drives the shell, and **delays the transition**. For `densPL_alpha = −2` the defaults
   are wrong from the very first step (true asymptote `β = 2`, §0.3).
4. **Failures must be persisted, not just logged.** A `logger` at L33 is not an audit trail. The
   per-snapshot outputs (`dictionary.jsonl`) should carry a convergence flag and the achieved
   residual, and `metadata.json` should carry a count of non-converged / rescued / clipped steps
   (SPEC-105 already establishes a `termination_debug` block as the home for this). A run in which
   the closure failed on a significant fraction of steps is not the model the paper describes, and
   right now that fraction is not recoverable from the published outputs.
5. **`_solve_lbfgsb` minimises `‖g‖`; that is not root-finding.** L-BFGS-B will happily converge to
   a *local minimum* with `‖g‖ > 0`. Accepting such a point because it satisfies
   `LBFGSB_FALLBACK_THRESHOLD` means accepting a state that does not satisfy the bubble energy
   balance at all — the error goes straight into `Ė_b`. If this path is used, the accepted point
   must be re-checked against the *root* tolerance, not a looser minimiser tolerance, and marked
   otherwise.
6. **`_solve_grid` quantises the answer.** A grid of `GRID_SIZE` nodes spanning `±GRID_EPSILON`
   around the guess has spacing `2ε/(GRID_SIZE−1)`; accepting the best node (especially via
   `GRID_EARLY_EXIT_RESIDUAL`) means `β` is discretised at that spacing. Signature: a histogram of
   `β(t)` over a run showing lattice clustering. Requirement: the spacing must be small enough that
   the induced `ΔL_cool` is below the tolerance of §1.7, and `GRID_EPSILON` must be **larger than
   the true per-step change in β** — otherwise the grid never brackets the root and the "best node"
   is always a boundary node, which is the clipping failure of §1.8 in disguise. `β` changes fastest
   near the transition, exactly where the grid is most likely to be out-run.
7. **Branch continuity.** If the residual surface admits more than one zero (possible once
   `∂L_cool/∂β` is large enough to overcome the diagonal dominance of §1.6), the *physical* root is
   the one continuous with the previous timestep. A global grid minimum can hop branches between
   steps ⇒ discontinuous `L_cool` ⇒ discontinuous ODE RHS ⇒ LSODA chatter (the same pathology
   SPEC-023 predicts for `max()`). Requirement: seed from the previous step and prefer the nearest
   root; test by plotting `β(t)`, `δ(t)` and flagging step-to-step jumps far larger than
   `|dβ/dt|·Δt`.
8. **Two full implementations (`_solve_betadelta_legacy` at L678 and `_solve_betadelta_hybr` at
   L948) are a divergence hazard.** They must produce the same `(β, δ)` to within tolerance on the
   same inputs, the selection (`_get_betadelta_solver`) must be deterministic and recorded in
   `metadata.json`, and any published run must state which was used. If the two disagree beyond
   tolerance, at most one of them is the model in the paper.
9. **`get_residual_detailed` (L514) must agree with `get_residual_pure` (L393) exactly** on the same
   `(β, δ, params)`. A diagnostic path that computes the residual slightly differently makes every
   diagnostic figure describe a model that was never integrated.
10. **`BubbleParamsView` (L107/L126) must be a pure read-through override** for `beta`, `delta`,
    `dMdt_guess`. Two requirements: (i) it must not mutate the underlying `params` — the residual is
    evaluated tens of times per step and any leaked state makes the residual history-dependent and
    the solve non-deterministic (and would break CLAUDE.md's separate-process equivalence
    requirement); (ii) `get(key, default)` must **not** silently supply a default for a *required*
    physics key. A mistyped or missing `Lmech`/`Pb`/`t_now` returning `None`→`0.0` zeroes a driver
    and the solve converges happily to a physically meaningless root. This class of bug is
    invisible in every downstream plot.
11. **`_describe_exc` / `_rescue_structure_failure`** — exceptions from the inner structure
    integration must be *classified* (integration failure vs. non-finite vs. `R1 ≥ R2` vs. table
    out-of-range) and counted. A rescue that succeeds must be recorded as a rescue; a rescue that
    perturbs the guess and retries is fine, a rescue that widens the tolerance is not (it changes
    the physics silently).

Confidence: **high** on 1–5 and 8–10 (these are general correctness requirements, not literature
claims); **medium** on 6–7 and 11 (I am reasoning from constant names and from the general shape of
this class of solver).

---

## 2. Validity regime of the whole closure — where the standard result stops holding

The `(α, β, δ)` machinery is **Weaver's self-similar solution**, and the strict conditions for it
are: constant `L_w`, uniform ambient `ρ₀`, spherical symmetry, no gravity, no radiation pressure,
no external pressure, unsaturated Spitzer conduction, radiative losses a perturbation, `R1 ≪ R2`,
and a thin, cold, radiative outer shell.

TRINITY violates essentially all of them by design. Using instantaneous `(α, β, δ)` as *local*
exponents is a defensible **quasi-similarity** approximation — the interior relaxes on the sound
crossing time `R2/c_s ≪ R2/v2` — but it fails wherever the drivers change faster than that:

| Breakdown | Why | Expected symptom |
|---|---|---|
| **SN onset** | `L_mech` jumps by orders of magnitude in ≪ one dynamical time; `p̈/ṗ` spikes | `β`, `δ` excursions; `cool_beta_to_Ebdot` `p̈/ṗ` term blows up; solver failures clustered at first SN |
| **Cloud-edge crossing `R2 = r_cloud`** | `ρ_amb` drops by orders of magnitude ⇒ `v2` jumps ⇒ `α` jumps | discontinuity in `α` propagates into `δ` via the closure |
| **Wind–SN gap** | `ṗ_tot → 0` | `p̈/ṗ` divergence (§1.1) |
| **Strong cooling (late energy phase)** | radiative losses are no longer a perturbation; the similarity solution's premise fails | root disappears — this is `_NoPhysicalRoot`, and it is *physics*, not a bug |
| **`R1 → R2`** | `R1 ≪ R2` premise fails; `V_b → 0`; `P_b` formula degenerate | negative volume unless guarded |
| **Steep density profile** | `α ≠ 3/5`; asymptotes shift to `(4+w)/(5−w)`, `−6/(7(5−w))` | default guesses poor; box clipping |
| **Saturated conduction** | for hot, tenuous, large bubbles the electron mean free path is not ≪ scale height; classical `C T^{5/2}` over-predicts the front flux (SPEC-043/044) | `L_cool` over-estimated, transition too early |
| **Gravity / radiation / `P_HII` on the shell** | change `R2(t)`, hence `α`, hence the whole triple | `α = v2 t/R2` will not sit at 3/5; SPEC-042 test degrades |
| **`max(P_b, P_HII)` driving (SPEC-023)** | when `P_HII` wins, the shell receives work the bubble did not do; `Ė_b`'s `−P_b V̇` term and the shell's work term no longer match | energy non-conservation that this residual will absorb into `β` |

---

## 3. Known traps — where this literature result is commonly mis-applied

1. **Uniform-medium exponents applied to a power-law cloud.** `β = 4/5, δ = −6/35` are `w = 0`
   values. The generalisation is `β = (4+w)/(5−w)`, `δ = −6/(7(5−w))` (§0.3). Shipping the `w = 0`
   numbers as guesses is fine; shipping them as **bounds** or as **failure fallbacks** is not.
2. **The two opposite sign conventions.** `β = −d ln P/d ln t` (positive), `δ = +d ln T/d ln t`
   (negative). "Tidying" them to match flips a first-order term. The `5/11 L_w` anchor of §1.1
   catches a β flip instantly.
3. **γ = 5/3-only coefficients.** `2π` in `P_b = E_b/(2π ΔV)`, `6π` in `Ė_b(β)`, `5/2` in the
   enthalpy/energy source, `2/(γ+1) = 3/4` in the strong-shock `R1`. A `gamma_adia` parameter that
   reaches only some of them is a fake knob.
4. **Weaver's `V_b ≈ (4π/3)R2³` shortcut** used in one place and `(R2³ − R1³)` in another. The
   presence of `pdotdot_total` in this slice's signature says the `R1³` term is intended; any
   caller that forms `P_b` without it breaks the chain rule (SPEC-024).
5. **`ṗ` from two different sources.** `ṗ = 2L_mech/v_mech` (SPEC-071) vs. the SPS table's
   `pdot_W + pdot_SN`. They agree only if the effective velocity is consistently defined across the
   wind and SN channels. `compute_R1_Pb` takes `(L, v)`; `cool_beta_to_Ebdot_pure` takes
   `(pdot, pdotdot)`. These must be the same `ṗ`.
6. **Absolute residual thresholds** on a dimensional quantity, across a grid spanning six decades
   in `M_cloud` (SPEC-073). The tolerance must be relative.
7. **Norm-minimisation mistaken for root-finding** (the L-BFGS-B fallback). A local minimum of
   `‖g‖` is not a solution of `g = 0`.
8. **Box clipping reported as convergence** — the single most common silent failure in
   constrained 2-D solves.
9. **Weaver interior *prefactors*.** SPEC-045 shows the two commonly-quoted `T_b` normalisations
   (`1.51×10⁶` and `2.07×10⁶ K`) are mutually inconsistent with isobaricity by a factor 3–4. If any
   such prefactor is hard-coded as an initial `T` guess in this slice, it is unvalidatable; prefer
   the prefactor-free structural forms (SPEC-024, SPEC-042). **I refuse to assert Weaver equation
   numbers** — I could not open the paper, and the numbering differs between the paper and the
   Rahner thesis's restatement.
10. **The transition trigger and the closure sharing `L_loss`.** SPEC-013's
    `(L_gain − L_loss)/L_gain ≤ 0.05` and this residual's `L_loss` must be the same object,
    including the `cooling_boost` treatment. SPEC-015 warns that the published Paper-II grid runs
    with `cooling_boost_fmix 4` — so this is not a hypothetical path.
11. **Piecewise-constant-in-age cooling tables** (SPEC-083) put steps into `L_cool`, hence into the
    residual surface, hence into `β(t)`. A solver cascade is the *symptom*; the table is the cause.

---

## 4. Priority ranking for the reconciler

| Rank | Item | Why it is first |
|---|---|---|
| 1 | `P dV` work counted exactly once in the β residual (§1.6) | silent energy non-conservation, directional bias on the transition time, and it is the code's headline output |
| 2 | Residual non-dimensionalisation before forming `g_total` (§1.6) | if violated, δ is unconstrained and every reported δ is noise |
| 3 | Non-convergence / box-clipping reported as convergence (§1.8, §1.9) | turns a numerical failure into a wrong published number with no trace |
| 4 | `β` box vs. `(4+w)/(5−w) = 2` for `densPL_alpha = −2` (§0.3) | a whole swept axis of the published grid clips if the cap is < 2 |
| 5 | `V_b` convention consistency between `compute_R1_Pb` and `cool_beta_to_Ebdot_pure` (§1.1) | `O(1)` error precisely at the transition |
| 6 | Same `L_loss` in residual and transition trigger (§1.4) | decouples dynamics from the phase criterion |
| 7 | `legacy` vs `hybr` divergence; `detailed` vs `pure` divergence (§1.9) | at most one of each pair is the published model |
| 8 | Relative vs absolute `RESIDUAL_THRESHOLD`, and residual noise floor (§1.7) | explains the cascade; cheapest experiment with the biggest explanatory payoff |

---

## 5. Confidence ledger (blunt)

- **High, derived here, independent of blocked literature:** the `Ė_b(β)` expression and its
  `5/11 L_w` Weaver anchor; `dT/dt = δT/t`; `P_b = 3(γ−1)E_b/(4πΔV)`; `R1 = sqrt(L/(2πvP_b))`;
  the generalised asymptotes `β = (4+w)/(5−w)`, `δ = −6/(7(5−w))`; the structure-equation
  coefficients `β+δ` and `β+5δ/2` and the `v − αr/t` advection; the residual-scale mismatch
  argument; the diagonal-dominance argument.
- **Medium:** the exact framing of the δ residual (definition-vs-structure is the natural reading
  given `delta2dTdt_pure`'s existence, but WARPFIELD may close it differently); the `dMdt` shooting
  boundary conditions; the `theta` mode semantics of `effective_Lloss`; my suggested numerical
  bounds and tolerances (judgement, not literature).
- **Low, and flagged as such:** the `3/4` strong-shock convention in `R1` (SPEC-025 leaves it open);
  whether negative `Ṁ_evap` (condensation) should be admitted; any Weaver+77 equation number — I
  assert none.

```json
[
  {
    "id": "S5a-C-01",
    "file": "trinity/phase1b_energy_implicit/get_betadelta.py",
    "line": 182,
    "class": "coefficient",
    "severity": "S1",
    "claim": "cool_beta_to_Ebdot_pure must return the exact chain-rule inverse of beta = -dlnPb/dlnt, namely Edot_b = 6*pi*Pb*(R2^2*v2 - R1^2*R1dot) - beta*Eb/t with R1dot = (R1/2)*(pdotdot/pdot + beta/t).",
    "evidence": "Eb = (3/2)Pb Vb with Vb = (4pi/3)(R2^3 - R1^3) gives Edot = (3/2)(Pdot V + P Vdot) = -beta Eb/t + (3/2)Pb Vdot, Vdot = 4pi(R2^2 v2 - R1^2 R1dot); R1 = sqrt(pdot/(4 pi Pb)) (SPEC-025) gives R1dot/R1 = (1/2)(pdotdot/pdot - Pbdot/Pb) = (1/2)(pdotdot/pdot + beta/t). The presence of pdotdot_total in the signature is only explicable via this R1dot term.",
    "expected": "Edot_b = 6*pi*Pb*R2^2*v2 - 3*pi*Pb*R1^3*(pdotdot/pdot) - (beta/t)*(Eb + 3*pi*Pb*R1^3); for general gamma replace 6pi -> 4pi/(gamma-1), 3pi -> 2pi/(gamma-1).",
    "failure_scenario": "A wrong prefactor or a missing R1dot term breaks the identity between Pb(t) and Eb(t), so the beta returned by the solve does not actually describe the pressure decline; Pb is then mis-evolved and the energy->momentum transition time shifts.",
    "repro": "Weaver-limit unit test: L_w=1, t=1, R2=0.762934*(L/rho0)^(1/5), v2=0.6*R2/t, Eb=(5/11)*L*t, Pb=Eb/(2*pi*R2^3), R1=0, beta=0.8 -> must return 0.4545454*L_w (= 5/11 L_w, SPEC-051).",
    "confidence": "high"
  },
  {
    "id": "S5a-C-02",
    "file": "trinity/phase1b_energy_implicit/get_betadelta.py",
    "line": 182,
    "class": "sign",
    "severity": "S1",
    "claim": "beta enters cool_beta_to_Ebdot_pure with a NEGATIVE sign (-beta*Eb/t), because beta = -dlnPb/dlnt is positive when Pb falls.",
    "evidence": "default.param ships cool_beta 0.8 with 'beta = -dPb/dt' and SPEC-041 confirms beta = -dlnPb/dlnt. Weaver-limit arithmetic: 6*pi*Pb*R2^2*v2 = (9/11)L_w and beta*Eb/t = (4/11)L_w; the difference is (5/11)L_w = SPEC-051's exact partition. A sign flip gives (13/11)L_w, a factor 2.6 error.",
    "expected": "Edot_b decreasing in beta; d(Edot_b)/d(beta) = -(Eb/t)*(1 + 1.5*R1^3/(R2^3-R1^3)) < 0.",
    "failure_scenario": "Bubble energy grows instead of being drained; Eb/(L_mech*t) lands near 13/11 instead of 5/11; the bubble never reaches the cooling-balance transition and every run over-predicts shell radius.",
    "repro": "Assert Eb/(L_mech*t) -> 0.4545 in a gravity-free, radiation-free, uniform-density energy-phase run (SPEC-051 / audit test T3); also assert the sign of the numerical derivative d(Edot_b)/d(beta) < 0.",
    "confidence": "high"
  },
  {
    "id": "S5a-C-03",
    "file": "trinity/phase1b_energy_implicit/get_betadelta.py",
    "line": 182,
    "class": "divergence",
    "severity": "S2",
    "claim": "The pdotdot_total/pdot_total ratio must be guarded against pdot_total -> 0.",
    "evidence": "R1dot/R1 = (1/2)(pdotdot/pdot + beta/t). SPS tables have a wind/SN gap where the total momentum injection rate passes through a deep minimum; TRINITY sums wind and SN channels (SPEC-070).",
    "expected": "A finite, documented guard (e.g. treat pdotdot/pdot as 0, or fall back to R1=0 / end-of-phase) rather than producing inf/nan that silently propagates into Edot_b.",
    "failure_scenario": "inf or nan in Edot_b makes every residual non-finite, the solve fails or returns garbage, and the run either aborts mid-phase or continues with a poisoned closure.",
    "repro": "Call with pdot_total = 0 (and with a tiny positive pdot_total, e.g. 1e-300) and assert the return is finite.",
    "confidence": "high"
  },
  {
    "id": "S5a-C-04",
    "file": "trinity/phase1b_energy_implicit/get_betadelta.py",
    "line": 297,
    "class": "coefficient",
    "severity": "S1",
    "claim": "compute_R1_Pb must use the SAME bubble-volume convention that cool_beta_to_Ebdot_pure's algebra assumes, i.e. Pb = 3(gamma-1)Eb/(4pi(R2^3 - R1^3)), not the R1-dropping Weaver shortcut Pb = 3(gamma-1)Eb/(4pi R2^3).",
    "evidence": "SPEC-024 audit trap; and the chain rule in S5a-C-01 is only exact if Pb is defined with (R2^3 - R1^3). The redundancy of passing both Pb and Eb into cool_beta_to_Ebdot_pure makes a convention mismatch invisible at the call site.",
    "expected": "One consistent Vb = (4pi/3)(R2^3 - R1^3) used in Pb(Eb), in Vdot, and in the PdV work term of the residual.",
    "failure_scenario": "The kinematic identity is violated by O((R1/R2)^3): negligible early, order unity when Eb collapses and R1 -> R2, i.e. exactly at the energy->momentum transition, corrupting the transition time.",
    "repro": "Finite-difference consistency test: for arbitrary (R2, v2, Eb, pdot, pdotdot, t, beta), advance the state by eps using the returned Edot_b, recompute (Pb, R1) with compute_R1_Pb, and assert (ln Pb(t+eps) - ln Pb(t))/eps = -beta/t to O(eps).",
    "confidence": "high"
  },
  {
    "id": "S5a-C-05",
    "file": "trinity/phase1b_energy_implicit/get_betadelta.py",
    "line": 272,
    "class": "sign",
    "severity": "S1",
    "claim": "delta2dTdt_pure must return +delta*T/t (no minus sign) — delta carries the OPPOSITE sign convention to beta.",
    "evidence": "SPEC-041: delta = +dlnT/dlnt, and default.param ships cool_delta = -6/35 (negative because T falls). beta = -dlnPb/dlnt is positive for the same physical situation. The two conventions are deliberately opposite.",
    "expected": "dT/dt = delta * T / t, in K/Myr; negative for delta<0.",
    "failure_scenario": "A sign flip makes the bubble appear to heat while expanding; the delta residual is satisfied at delta = +6/35, the interior density scaling d ln n/d ln t = -(beta+delta) is wrong by 2*delta, and L_cool is systematically mis-computed.",
    "repro": "Assert delta2dTdt_pure(t=1.0, T=1e7, delta=-6/35) == -6/35*1e7 (negative), and assert the solved delta stays negative through a clean energy-phase run.",
    "confidence": "high"
  },
  {
    "id": "S5a-C-06",
    "file": "trinity/phase1b_energy_implicit/get_betadelta.py",
    "line": 272,
    "class": "regime",
    "severity": "S2",
    "claim": "The t passed to delta2dTdt_pure must be the same clock (cluster age from feedback onset) used for alpha = v2*t/R2 and for beta, and must be guarded against t -> 0.",
    "evidence": "The (alpha, beta, delta) triple is a set of logarithmic derivatives about a common time origin (SPEC-041). Weaver similarity requires t measured from bubble birth; the SPS table's t=0 is cluster formation (SPEC-070).",
    "expected": "t_now = cluster age in Myr, identical to the SPS-table clock; a divide-by-zero guard at t=0.",
    "failure_scenario": "A phase-local clock shifts all three exponents; the SPEC-042 closure delta = (2/7)(2 alpha - beta - 1) fails systematically and the bubble structure is built on wrong coefficients, with no visible error.",
    "repro": "Extract alpha = v2*t/R2 from an energy-phase run of param/simple_cluster.param and assert it relaxes to ~0.6 (SPEC-056, audit test T4); a systematic offset indicates a clock mismatch.",
    "confidence": "high"
  },
  {
    "id": "S5a-C-07",
    "file": "trinity/phase1b_energy_implicit/get_betadelta.py",
    "line": 297,
    "class": "coefficient",
    "severity": "S1",
    "claim": "compute_R1_Pb must build the wind momentum rate as pdot = 2*Lmech_total/v_mech_total and set R1 = sqrt(pdot/(4 pi Pb)) = sqrt(Lmech_total/(2 pi v_mech_total Pb)).",
    "evidence": "SPEC-071: L = (1/2) Mdot v^2 and pdot = Mdot v, so pdot = 2L/v. SPEC-025: ram-pressure balance rho_w(R1) v_w^2 = Pb with rho_w = Mdot/(4 pi r^2 v_w). Dimension check in AU: [L/v] = Msun pc Myr^-2 (force), /[P] = pc^2, sqrt -> pc.",
    "expected": "R1 = sqrt(Lmech_total/(2*pi*v_mech_total*Pb)); optionally the strong-shock variant sqrt(3*pdot/(16*pi*Pb)) (0.866x smaller) if the 3/4 post-shock factor is adopted — SPEC-025 leaves this open, but the choice must be documented.",
    "failure_scenario": "A factor-2 slip (using L/v instead of 2L/v) gives R1 too small by sqrt(2); Vb is over-estimated, Pb under-estimated, and the wind termination shock sits in the wrong place throughout the energy phase.",
    "repro": "Assert R1^2 * 4*pi*Pb == 2*Lmech_total/v_mech_total for the returned pair, and cross-check that the same pdot value is the one passed as pdot_total to cool_beta_to_Ebdot_pure.",
    "confidence": "high"
  },
  {
    "id": "S5a-C-08",
    "file": "trinity/phase1b_energy_implicit/get_betadelta.py",
    "line": 297,
    "class": "divergence",
    "severity": "S1",
    "claim": "compute_R1_Pb must trap R1 >= R2 (and Eb <= 0, v_mech_total <= 0) rather than returning a pair that makes R2^3 - R1^3 <= 0.",
    "evidence": "R1 ~ Pb^{-1/2} ~ Eb^{-1/2}, so R1 -> infinity as the bubble loses energy; R1 crossing R2 is guaranteed if the energy phase runs long enough. It is the physical statement 'the bubble can no longer stand off the wind', i.e. the end of the energy-driven regime.",
    "expected": "R1 < R2 enforced; a crossing raises/flags a no-physical-root / end-of-energy-phase condition that the caller acts on, never a negative bubble volume or negative Pb.",
    "failure_scenario": "Negative Vb gives negative Pb, negative pressure drives the shell inward, and the run produces physically meaningless trajectories that may still terminate 'successfully'.",
    "repro": "Sweep Eb downward at fixed R2 and assert the function raises/flags before R2^3 - R1^3 changes sign; scan a run's snapshots for R1 >= R2 or Pb <= 0.",
    "confidence": "high"
  },
  {
    "id": "S5a-C-09",
    "file": "trinity/phase1b_energy_implicit/get_betadelta.py",
    "line": 297,
    "class": "coefficient",
    "severity": "S2",
    "claim": "gamma_adia must be threaded consistently: if compute_R1_Pb honours (gamma-1) then cool_beta_to_Ebdot_pure's 6*pi (= (3/2)*4*pi, gamma=5/3 only) must also generalise to 4*pi/(gamma-1).",
    "evidence": "Eb = Pb Vb/(gamma-1); the (3/2) and the 2pi in Pb = Eb/(2 pi (R2^3-R1^3)) are both gamma=5/3 specialisations (SPEC-024). cool_beta_to_Ebdot_pure has no gamma argument at all.",
    "expected": "Either gamma is honoured everywhere, or gamma_adia is documented as fixed at 5/3 and the argument removed/asserted.",
    "failure_scenario": "Setting gamma_adia != 5/3 silently breaks the Pb<->Eb chain rule; the closure then solves a beta that does not correspond to the actual pressure decline, with no error raised.",
    "repro": "Run with gamma_adia = 1.4 and assert the finite-difference identity of S5a-C-04 still holds; or assert that gamma_adia != 5/3 is rejected.",
    "confidence": "medium"
  },
  {
    "id": "S5a-C-10",
    "file": "trinity/phase1b_energy_implicit/get_betadelta.py",
    "line": 334,
    "class": "coefficient",
    "severity": "S2",
    "claim": "effective_Lloss in mode 'none' must return exactly Lcool + Lleak (bit-identically), and in multiplier mode must scale Lcool only — not Lleak and not the PdV work.",
    "evidence": "SPEC-015: the cooling_boost knobs exist to patch 1-D conduction's under-prediction of TURBULENT MIXING loss at the contact discontinuity (El-Badry+19; Lancaster+21). Lleak is a venting enthalpy flux through an open covering fraction (SPEC-036) — a different mechanism. PdV work is mechanical, not thermal.",
    "expected": "none -> Lcool + Lleak exactly; multiplier -> fmix*Lcool + Lleak; L_loss >= 0 and monotone non-decreasing in Lcool, Lleak, fmix.",
    "failure_scenario": "Boosting Lleak makes coverFraction < 1 runs lose energy at fmix times the correct venting rate; boosting the work term destroys the shell/bubble work balance (SPEC-035). Mode 'none' not being an exact identity means the default (published) physics is not the unboosted model.",
    "repro": "Assert effective_Lloss('none', fmix=4, ..., Lcool=a, Lleak=b, ...) returns a+b to bit equality for random a,b; assert d(L_loss)/d(fmix) equals Lcool exactly in multiplier mode.",
    "confidence": "medium"
  },
  {
    "id": "S5a-C-11",
    "file": "trinity/phase1b_energy_implicit/get_betadelta.py",
    "line": 360,
    "class": "divergence",
    "severity": "S1",
    "claim": "The L_loss produced here must be the identical quantity used by the energy->momentum transition trigger (L_gain - L_loss)/L_gain <= phaseSwitch_LlossLgain.",
    "evidence": "SPEC-013/014 define the transition on exactly this ratio; SPEC-015 records that the published Paper-II grid runs with cooling_boost_fmix = 4, so the boosted and unboosted paths genuinely differ in shipped configurations.",
    "expected": "One code path computes L_loss (including any boost) and both the bubble energy residual and the phase trigger consume it.",
    "failure_scenario": "The bubble evolves with boosted cooling but transitions on unboosted cooling (or vice versa); the transition time — the code's headline prediction — becomes decoupled from the dynamics that produced it, and Paper-II's grid is internally inconsistent.",
    "repro": "Instrument a run with cooling_boost_mode=multiplier, fmix=4: assert the L_loss recorded in the snapshot's transition-trigger diagnostic equals the L_loss used in the Edot_b residual, to machine precision.",
    "confidence": "medium"
  },
  {
    "id": "S5a-C-12",
    "file": "trinity/phase1b_energy_implicit/get_betadelta.py",
    "line": 334,
    "class": "regime",
    "severity": "S3",
    "claim": "If a theta_target mode pins L_loss = theta_target * Lmech exactly, the cooling_balance transition trigger degenerates to a time-independent constant and either fires immediately or never.",
    "evidence": "(L_gain - L_loss)/L_gain with L_loss = theta*L_mech and L_gain = eta_th*L_mech is 1 - theta/eta_th, independent of t. SPEC-013/014 fire on this ratio crossing 0.05.",
    "expected": "theta mode should blend or floor (e.g. L_loss = max(Lcool + Lleak, theta_target*Lmech)) so the ratio retains time dependence, or the mode must be documented as incompatible with transition_trigger=cooling_balance.",
    "failure_scenario": "With theta_target below the threshold the energy phase never ends and the run terminates on stop_t instead of physics; above it, the run switches to momentum-driven on the first step.",
    "repro": "Run with cooling_boost_mode=theta at theta_target on either side of (1 - phaseSwitch_LlossLgain) and check whether the transition fires at step 0 / never.",
    "confidence": "low"
  },
  {
    "id": "S5a-C-13",
    "file": "trinity/phase1b_energy_implicit/get_betadelta.py",
    "line": 393,
    "class": "coefficient",
    "severity": "S1",
    "claim": "The beta residual must compare the kinematic Edot_b(beta) against an energy budget that includes the shell PdV work term exactly once: Edot_phys = L_gain - L_loss - Pb*4*pi*R2^2*v2 (+ Pb*4*pi*R1^2*R1dot).",
    "evidence": "SPEC-035 first law for the bubble as an open control volume. The equivalent single-equation form is L_gain - L_loss = (5/2) Pb Vdot - beta Eb/t, where 5/2 = 3/2 + 1 is the enthalpy coefficient; mixing framings omits or double-counts the work.",
    "expected": "PdV appears once. Either (A) Edot_kin = (3/2)Pb Vdot - beta Eb/t compared with L_gain - L_loss - Pb Vdot, or (B) the merged form L_gain - L_loss = (5/2) Pb Vdot - beta Eb/t.",
    "failure_scenario": "Omitting -Pb*Vdot makes Edot_phys too large, drives beta too small, keeps Pb artificially high, over-drives the shell and DELAYS the transition; double-counting does the reverse. Either way R2(t) and the transition time — the published results — are wrong, with no error raised.",
    "repro": "Audit test T3/SPEC-051: in a gravity-free, radiation-free, uniform-density energy-phase run assert Eb/(L_mech*t) -> 0.4545; omission gives a systematically larger ratio. Also assert the residual is zero when fed the analytic Weaver state (alpha=3/5, beta=4/5, delta=-6/35, Lcool=0).",
    "confidence": "high"
  },
  {
    "id": "S5a-C-14",
    "file": "trinity/phase1b_energy_implicit/get_betadelta.py",
    "line": 393,
    "class": "units",
    "severity": "S1",
    "claim": "The two residuals must be non-dimensionalised (or expressed directly in exponent space) before being combined into any scalar norm, because a raw energy-rate residual and a raw dT/dt residual differ by ~10^4 in AU units.",
    "evidence": "For a 10^6 Msun cluster, L_mech ~ 1e40 erg/s ~ 1.7e10 in AU luminosity units (SPEC-091: 1 AU = 6.0255e29 erg/s), while dT/dt ~ delta*T/t ~ -1.7e6 K/Myr. An L2 norm of the raw pair is determined entirely by the beta residual.",
    "expected": "r_beta normalised by max(|L_gain|,|L_loss|,|Pb Vdot|) and r_delta already O(1); or both formed as (beta_implied - beta_trial) and (delta_implied - delta_trial).",
    "failure_scenario": "delta is effectively unconstrained: the solver reports convergence on a beta-only condition and returns whatever delta the seed happened to carry, so the interior temperature profile and hence L_cool are wrong while every diagnostic says 'converged'.",
    "repro": "Hold beta at the solved value and sweep delta over its admissible box; if g_total varies by less than the convergence threshold across the sweep, delta is unconstrained. Also check whether the solved delta satisfies SPEC-042's delta = (2/7)(2 alpha - beta - 1).",
    "confidence": "high"
  },
  {
    "id": "S5a-C-15",
    "file": "trinity/phase1b_energy_implicit/get_betadelta.py",
    "line": 47,
    "class": "numerical",
    "severity": "S2",
    "claim": "RESIDUAL_THRESHOLD must be a RELATIVE tolerance of order 1e-4 to 1e-3, and must be loose enough to exceed the inner ODE/shooting noise floor.",
    "evidence": "The transition fires at (L_gain-L_loss)/L_gain <= 0.05 (SPEC-013/014); to place the transition time to ~1% the loss fraction must be resolved to ~5e-4, i.e. relative residual ~1e-3. The shipped sweep spans mCloud 1e4..5e9 Msun (SPEC-073), six decades in L_mech, so an absolute threshold cannot be scale-appropriate across it.",
    "expected": "Relative threshold in [1e-4, 1e-3]; inner integration rtol at least one order tighter; threshold documented against the 0.05 trigger.",
    "failure_scenario": "Absolute threshold: over-tight at low mass (always non-converged, always falls through the cascade) and meaningless at high mass (accepts large errors). A relative threshold >= 1e-2 aliases straight into the transition criterion, making the headline result a numerical artefact.",
    "repro": "Evaluate get_residual_pure repeatedly at fixed (beta,delta,params) with perturbed dMdt seeds to measure the residual noise floor; compare against RESIDUAL_THRESHOLD. Also run the same param file at mCloud 1e4 and 1e9 and compare non-convergence rates.",
    "confidence": "medium"
  },
  {
    "id": "S5a-C-16",
    "file": "trinity/phase1b_energy_implicit/get_betadelta.py",
    "line": 41,
    "class": "regime",
    "severity": "S1",
    "claim": "The (beta, delta) box must contain the power-law-generalised Weaver asymptotes beta = (4+w)/(5-w) and delta = -6/(7(5-w)) for w = |densPL_alpha| in [0,2], i.e. beta up to 2.0 and delta down to -2/7, plus headroom above beta=2 for the pre-transition collapse.",
    "evidence": "Derived: R2 ~ t^eta with eta = 3/(5-w) (SPEC-053); Eb ~ L t so Pb ~ Eb/R2^3 ~ t^(1-3eta) giving beta = 3eta-1 = (4+w)/(5-w); delta = (2/7)(2 eta - beta - 1) = -2 eta/7 = -6/(7(5-w)) (SPEC-042). Checks: w=0 -> 4/5, -6/35 (SPEC-041); w=2 -> 2, -2/7.",
    "expected": "BETA_MIN <~ 0 (beta can go slightly negative when L_mech rises), BETA_MAX >= ~5; DELTA_MIN <= ~-1, DELTA_MAX >= ~+0.5.",
    "failure_scenario": "A BETA_MAX below 2 clips the physical root for every densPL_alpha = -2 run in the published grid; the clipped beta holds Pb too high, over-drives the shell and delays the transition, and the result is reported as converged.",
    "repro": "Assert BETA_MAX > 2 and DELTA_MIN < -2/7; run param with densPL_alpha=-2 and check the solved beta against the analytic 2.0 asymptote and against the box bounds.",
    "confidence": "high"
  },
  {
    "id": "S5a-C-17",
    "file": "trinity/phase1b_energy_implicit/get_betadelta.py",
    "line": 932,
    "class": "silent-failure",
    "severity": "S1",
    "claim": "A non-converged solve, a no-physical-root outcome, or a box-clipped answer must be returned with converged=False plus the achieved residual, and must never be consumed by the caller as if it were a root.",
    "evidence": "The closure supplies L_cool to the bubble energy equation; an unconverged (beta,delta) means the ODE right-hand side is wrong by an unbounded amount. General correctness requirement for an implicit closure inside an ODE integration.",
    "expected": "converged flag False; achieved residual carried; caller shrinks dt and retries, or declares end-of-energy-phase (physically meaningful when R1->R2 or no root exists), or aborts with a recorded termination reason.",
    "failure_scenario": "The run continues with the guess (typically the Weaver 0.8 / -6/35 defaults). Late in the energy phase the true beta is well above 4/5, so the fallback under-states the pressure decline, over-drives the shell and systematically DELAYS the transition — a directional bias in the published result with no trace in the outputs.",
    "repro": "Count non-converged / rescued / clipped steps over a full run of param/simple_cluster.param and the docs/dev/performance/f1edge_* configs; assert the count is recorded in metadata.json and that it is zero or bounded.",
    "confidence": "high"
  },
  {
    "id": "S5a-C-18",
    "file": "trinity/phase1b_energy_implicit/get_betadelta.py",
    "line": 625,
    "class": "silent-failure",
    "severity": "S2",
    "claim": "Per-step convergence status and achieved residual must be persisted to the run outputs (a snapshot flag in dictionary.jsonl and a count in metadata.json), not only emitted to the logger.",
    "evidence": "SPEC-105 establishes metadata.json's termination/termination_debug block as the home for run-integrity bookkeeping; SPEC-006 lists the per-snapshot data model. A logger line is not recoverable from a published output archive.",
    "expected": "A boolean/enum convergence status and a numeric residual per snapshot, plus aggregate counts (converged / rescued / fell-back / clipped) in metadata.json.",
    "failure_scenario": "A run whose closure failed on a large fraction of steps is indistinguishable in the published outputs from a clean run; no reader (or future audit) can tell which grid cells are trustworthy.",
    "repro": "Grep a produced dictionary.jsonl / metadata.json for any beta-delta convergence field; absence is the finding.",
    "confidence": "high"
  },
  {
    "id": "S5a-C-19",
    "file": "trinity/phase1b_energy_implicit/get_betadelta.py",
    "line": 1108,
    "class": "silent-failure",
    "severity": "S1",
    "claim": "_solve_lbfgsb minimises a norm, which is not root-finding; a local minimum with ||g|| above the ROOT tolerance must not be accepted as a solution.",
    "evidence": "L-BFGS-B converges when its own gradient/step criteria are met, which happens at any local minimum of ||g||, including ones where g != 0. The physical requirement is g = 0 (energy balance closes), not 'g is locally smallest'.",
    "expected": "Any point returned by the minimiser is re-tested against RESIDUAL_THRESHOLD as a root; failing that test it is reported converged=False. LBFGSB_FALLBACK_THRESHOLD must not be looser than the root tolerance.",
    "failure_scenario": "The bubble energy balance is left unclosed by an amount that never appears anywhere; Edot_b is wrong by that residual every step it is used, accumulating into Eb(t), Pb(t), R2(t) and the transition time.",
    "repro": "Log the achieved ||g|| whenever the lbfgsb path is taken and assert it is below RESIDUAL_THRESHOLD; compare LBFGSB_FALLBACK_THRESHOLD against RESIDUAL_THRESHOLD.",
    "confidence": "high"
  },
  {
    "id": "S5a-C-20",
    "file": "trinity/phase1b_energy_implicit/get_betadelta.py",
    "line": 1010,
    "class": "numerical",
    "severity": "S2",
    "claim": "_solve_grid quantises (beta, delta) to the grid lattice; GRID_EPSILON must exceed the true per-step change in beta, and the lattice spacing 2*GRID_EPSILON/(GRID_SIZE-1) must induce an L_cool error below the residual tolerance.",
    "evidence": "A best-node search returns a lattice point, not a root. beta changes fastest approaching the transition (it must climb from ~0.8 toward and past 2); if the per-step change exceeds the half-width, the best node is always a boundary node — the clipping failure of S5a-C-16 in disguise.",
    "expected": "GRID_EPSILON > max per-step |d beta|; lattice spacing fine enough that d(L_cool)/d(beta) * spacing is below tolerance; a boundary-node result treated as non-converged and refined.",
    "failure_scenario": "beta(t) becomes a staircase; L_cool is piecewise-constant; the outer ODE right-hand side is discontinuous, producing LSODA chatter and a transition time quantised by the grid.",
    "repro": "Histogram the solved beta over a full run and look for lattice clustering at spacing 2*GRID_EPSILON/(GRID_SIZE-1); separately record how often the accepted node lies on the grid boundary.",
    "confidence": "medium"
  },
  {
    "id": "S5a-C-21",
    "file": "trinity/phase1b_energy_implicit/get_betadelta.py",
    "line": 1010,
    "class": "numerical",
    "severity": "S2",
    "claim": "The accepted root must be the branch continuous with the previous timestep's (beta, delta); a global search must not hop branches between steps.",
    "evidence": "The residual Jacobian is expected diagonally dominant (d r_beta/d beta ~ -(Eb/t)(1+1.5 R1^3/(R2^3-R1^3)), d r_delta/d delta ~ -T/t), so a well-seeded Newton has a unique nearby root; a coarse global minimisation has no such guarantee.",
    "expected": "Seed from the previous accepted (beta, delta); prefer the nearest root; beta(t) and delta(t) continuous to within |d beta/dt| * dt.",
    "failure_scenario": "Branch hopping makes L_cool discontinuous in t, which makes the outer ODE right-hand side discontinuous — the adaptive integrator collapses its step size fighting a kink that is not physical (the same pathology SPEC-023 predicts for max(Pb, P_HII)).",
    "repro": "Plot beta(t), delta(t) from a full run; flag any step-to-step jump much larger than the local |d beta/dt| * dt.",
    "confidence": "medium"
  },
  {
    "id": "S5a-C-22",
    "file": "trinity/phase1b_energy_implicit/get_betadelta.py",
    "line": 678,
    "class": "divergence",
    "severity": "S2",
    "claim": "_solve_betadelta_legacy and _solve_betadelta_hybr must return the same (beta, delta) to within tolerance on the same inputs, and which one ran must be deterministic and recorded.",
    "evidence": "Two independent implementations of the same closure inside a stiff iterative path; CLAUDE.md rule 5 requires full-run equivalence on the stiffest regimes in separate processes at matched t for exactly this situation.",
    "expected": "A documented equivalence gate; the selected solver recorded in metadata.json; at most one of them is the published model.",
    "failure_scenario": "Published results depend on which solver path a config happens to select; two runs of nominally the same physics diverge in transition time and final fate.",
    "repro": "Run param/simple_cluster.param and docs/dev/performance/f1edge_{lowdens,hidens}*.param under each solver setting, in separate processes, and compare R2, v2, Eb, Pb at matched simulation time.",
    "confidence": "high"
  },
  {
    "id": "S5a-C-23",
    "file": "trinity/phase1b_energy_implicit/get_betadelta.py",
    "line": 514,
    "class": "divergence",
    "severity": "S3",
    "claim": "get_residual_detailed must produce residual values identical to get_residual_pure for the same (beta, delta, params).",
    "evidence": "A diagnostic path that recomputes the residual by a different route describes a model that was never integrated; every figure and audit trace built on it would be misleading.",
    "expected": "Bit-identical (or within floating-point associativity) residual values from both entry points on the same inputs.",
    "failure_scenario": "Diagnostics and published figures show a residual/energy budget that differs from the one the solver actually enforced, so a real closure bug is invisible in exactly the artefact meant to expose it.",
    "repro": "Property test: for random (beta, delta) in the admissible box, assert get_residual_detailed and get_residual_pure agree to ~1e-12 relative.",
    "confidence": "high"
  },
  {
    "id": "S5a-C-24",
    "file": "trinity/phase1b_energy_implicit/get_betadelta.py",
    "line": 107,
    "class": "state",
    "severity": "S2",
    "claim": "BubbleParamsView must be a non-mutating read-through override, and its get(key, default) must not silently substitute a default for a required physics key.",
    "evidence": "The residual is evaluated many times per step; leaked mutation makes the residual history-dependent and the solve non-deterministic, breaking CLAUDE.md's separate-process / bit-identical equivalence requirements. A missing Lmech / Pb / t_now defaulting to 0.0 or None zeroes a driver.",
    "expected": "No writes to the wrapped params; missing required keys raise (KeyError), only genuinely optional keys take a default.",
    "failure_scenario": "A mistyped key returns a default, the solve converges to a root of a physically empty problem (e.g. zero mechanical input), and every downstream quantity looks smooth and plausible while being wrong.",
    "repro": "Construct a BubbleParamsView over a params object, evaluate the residual N times and assert the wrapped params compares equal before and after; assert get() on an unknown required key raises rather than returning the default.",
    "confidence": "medium"
  },
  {
    "id": "S5a-C-25",
    "file": "trinity/phase1b_energy_implicit/get_betadelta.py",
    "line": 374,
    "class": "state",
    "severity": "S3",
    "claim": "The dMdt seed must affect only the iteration path, never the accepted root; and rejecting negative dMdt forecloses the physically real condensation regime.",
    "evidence": "dMdt is the shooting parameter for the interior structure (mass-flux matching at R1 to Mdot_wind and at R2 to the conductive evaporation flux, SPEC-044). In a strongly radiative bubble the conduction front reverses and hot gas condenses onto the shell, giving dMdt < 0 — precisely the late energy phase this module models.",
    "expected": "Root invariant under seed perturbation to within tolerance; negative dMdt either admitted with a documented sign convention or explicitly rejected with a stated physical justification.",
    "failure_scenario": "Seed-dependent roots make runs non-reproducible and defeat every bit-identity gate; filtering negatives biases the late-phase structure toward evaporation and under-states the loss of bubble mass/energy.",
    "repro": "Re-solve a captured step with dMdt_guess perturbed by +/-20% and +/-2x and assert the returned (beta, delta) agree to the solver tolerance.",
    "confidence": "medium"
  },
  {
    "id": "S5a-C-26",
    "file": "trinity/phase1b_energy_implicit/get_betadelta.py",
    "line": 393,
    "class": "regime",
    "severity": "S2",
    "claim": "The solved (beta, delta) must satisfy the conduction closure delta ~= (2/7)(2*alpha - beta - 1) with alpha = v2*t/R2 throughout the conduction-dominated energy phase.",
    "evidence": "SPEC-042, re-derived here: T_b^{7/2} = a * Pb * R2^2/(C t) from balancing the Spitzer conductive flux against the isobaric enthalpy/expansion terms; taking d ln/d ln t gives (7/2) delta = 2 alpha - beta - 1. It reproduces -6/35 exactly at (alpha,beta)=(3/5,4/5) and is prefactor-free.",
    "expected": "The relation holds to a few percent early in the energy phase, degrading only as radiative cooling becomes comparable to conduction.",
    "failure_scenario": "Systematic violation means either delta is unconstrained by the residual (see S5a-C-14), the time origin is wrong (S5a-C-06), or the structure ODE's coefficients (beta+delta in continuity, beta+5delta/2 in energy, advection v - alpha r/t) do not match the definitions used here.",
    "repro": "Audit test T5: extract (alpha, beta, delta) from a run's snapshots and plot delta against (2/7)(2 alpha - beta - 1) through the energy phase.",
    "confidence": "high"
  },
  {
    "id": "S5a-C-27",
    "file": "trinity/phase1b_energy_implicit/get_betadelta.py",
    "line": 182,
    "class": "coefficient",
    "severity": "S2",
    "claim": "The pdot_total fed to cool_beta_to_Ebdot_pure must be the same momentum-injection rate that defines R1 in compute_R1_Pb, i.e. 2*Lmech_total/v_mech_total.",
    "evidence": "R1dot is the time derivative of R1 = sqrt(pdot/(4 pi Pb)). If one path uses 2L/v (SPEC-071) and the other uses the SPS table's pdot_W + pdot_SN (SPEC-070), they differ whenever the wind and SN channels have different effective velocities.",
    "expected": "A single pdot_total definition shared by both functions; assert 4*pi*R1^2*Pb == pdot_total at every call.",
    "failure_scenario": "R1dot is the derivative of a different R1 than the one in Vb, so the Pb<->Eb chain rule is broken by an uncontrolled amount that grows once SNe dominate the momentum budget.",
    "repro": "At each call site assert abs(4*pi*R1**2*Pb/pdot_total - 1) < 1e-10.",
    "confidence": "high"
  },
  {
    "id": "S5a-C-28",
    "file": "trinity/phase1b_energy_implicit/get_betadelta.py",
    "line": 48,
    "class": "numerical",
    "severity": "S2",
    "claim": "The need for a three-deep solver cascade (grid -> hybr -> L-BFGS-B) plus a legacy path plus a rescue path indicates a noisy or non-smooth residual, not an intrinsically hard root-find; the residual's noise floor should be measured and the cause fixed at source.",
    "evidence": "The expected Jacobian is diagonally dominant (r_beta dominated by -(Eb/t)(1+1.5 R1^3/(R2^3-R1^3)) from the definition side; r_delta dominated by -T/t), so a Newton/hybr seeded from the previous step should converge in well under 10 iterations. Known smoothness hazards: per-age cooling FILE selection makes Lambda piecewise-constant in cluster age (SPEC-083), and max(Pb, P_HII) is non-differentiable (SPEC-023).",
    "expected": "Measured residual noise floor below RESIDUAL_THRESHOLD; hybr converging from the previous step's seed in <10 iterations for the great majority of steps; the cascade exercised rarely and its use counted.",
    "failure_scenario": "If the inner integration noise exceeds the outer threshold, hybr can never converge, the cascade fires every step, and the accepted (beta,delta) is whatever the last fallback produced — robust-looking silent degradation.",
    "repro": "Evaluate get_residual_pure 20x at fixed inputs with jittered dMdt seeds and measure the spread; separately instrument which solver path is taken per step over a full run of param/simple_cluster.param and the f1edge configs, and record the counts as a committed CSV.",
    "confidence": "medium"
  }
]
```
