# S5b implicit runner — Lens C (what it should be)

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

**Slice:** Phase 1b, the implicit-phase runner —
`trinity/phase1b_energy_implicit/run_energy_implicit_phase.py`.
**Method:** derived from the interface in `signatures.md` (names and argument lists only — no
values, no bodies, no comments), plus `docs/dev/code-audit/reference/PHYSICS_SPEC.md` (cited as
`SPEC-nnn`), plus internal knowledge of Weaver+77, Rahner+17/19 (WARPFIELD) and standard ISM
feedback theory. **No implementation file was read; no code was run.**

**Literature access:** blocked (proxy 403 on arXiv/ADS/journals), exactly as recorded in
PHYSICS_SPEC §0.3. Everything below is either (i) re-derived here from first principles, (ii)
taken from PHYSICS_SPEC with its own confidence tag carried through, or (iii) flagged
`[UNVERIFIED]`. I do **not** assert Weaver equation numbers and I do not assert any Weaver interior
*prefactor* — see SPEC-045. Where I give a coefficient I say whether I derived it here.

---

## 1. What this module is, structurally

The signature set is unambiguous about the algorithm class, even without the bodies:

- `get_ODE_implicit_pure(t, y, snapshot, params_for_feedback, Ed_from_beta, Td_from_delta)` — the
  RHS of an ODE **whose bubble-energy and bubble-temperature derivatives are supplied from
  outside** rather than computed inside. That is the defining move of an operator-split
  ("implicit") energy phase: the expensive bubble-interior solve is done **once per segment**, and
  the cheap dynamical ODE is integrated over the segment with those rates frozen.
- `Ed_from_beta` / `Td_from_delta` name the Weaver similarity parameters β and δ (SPEC-041), so the
  outer solve is the classical WARPFIELD (β, δ) root-find on the bubble structure. `no_root_count`,
  `NO_ROOT_HANDOFF_STREAK`, `BETADELTA_*` confirm a 2-D root-find that can fail.
- `DT_SEGMENT_*`, `MAX_SEGMENTS`, `next_dt_segment`, `compute_max_dex_change`,
  `ADAPTIVE_THRESHOLD_DEX/FACTOR`, `ADAPTIVE_MONITOR_KEYS` — a **segment-length controller** driven
  by the log-space change of monitored quantities across the completed segment.
- `parse_transition_triggers` / `evaluate_r1_shadow` / `r1_transition_decision` — the phase-exit
  logic of SPEC-014 (`cooling_balance`, `blowout`, `ebpeak`, alias `r1 = blowout,ebpeak`), with the
  `r1` alias evaluated in **shadow mode** (computed and logged, acted on only if enabled).

So the correctness question splits into: (A) is the continuous physics right; (B) is the
operator split a controlled approximation; (C) are the exit criteria the physically right ones and
are they detected robustly.

---

## 2. The coupled system that must be integrated

Global time `t` = cluster age since coeval formation (the same clock the SPS table is indexed on).
State vector, in TRINITY internal units `[M⊙, pc, Myr]` (SPEC-090):

```
    y = ( R2 , v2 , E_b , T )              T = bubble temperature at ξ = bubble_xi_Tb (SPEC-040)
```

### 2.1 Kinematics
```
    dR2/dt = v2
```
Trivial, but it must be `y[1]`, not a snapshot-frozen `v2`. Freezing it decouples R from v inside
the segment and turns the integration into a first-order Euler step in disguise.

### 2.2 Momentum (SPEC-020, SPEC-022, SPEC-026/027, SPEC-031)

Newton's second law for a variable-mass thin shell accreting material at rest:

```
    d(M_sh v2)/dt = 4πR2² ( P_drive − P_ext ) + F_rad − F_grav ,     Ṁ_sh = 4πR2² ρ_amb(R2) · max(v2,0)
```
expanded (the form a code integrates):
```
    M_sh dv2/dt = 4πR2² ( P_drive − P_ext ) + F_rad − F_grav − 4πR2² ρ_amb(R2) v2·|v2|
```

Term by term, with what each must be **in this phase**:

| term | expression | notes |
|---|---|---|
| drive | `4πR2² P_drive`, `P_drive = max(P_b, P_HII)` | SPEC-022: energy/implicit phase has **no** `P_ram`. Adding `ṗ_w/(4πR2²)` here double-counts the wind, whose momentum is already thermalised into `P_b`. |
| bubble pressure | `P_b = E_b / [2π(R2³ − R1³)]` | γ=5/3 only (§2.5). |
| external | `−4πR2² P_ISM` | `PISM` is declared as `P/k_B` in K cm⁻³ (SPEC-092.4) — must be multiplied by `k_B` somewhere upstream. |
| sweep-up drag | `−4πR2² ρ_amb v2|v2|` | **exactly once** (SPEC-020 trap). Note `v2|v2|`, not `v2²`: with `v2²` a re-collapsing shell (`v2<0`) is *accelerated inward* by a term that is supposed to be drag. |
| radiation | `F_rad = (L_bol/c)(1 − e^{−τ_UV} + τ_IR)` | SPEC-026/027. |
| gravity | `F_grav = G M_sh (M_cluster + M_sh/2)/R2²` | SPEC-031; the `M_sh/2` self-gravity factor is the classic factor-2 trap. Gas outside `R2` contributes nothing (shell theorem). |

**Legitimately omitted** in this phase: direct wind/SN ram pressure (thermalised — see above);
shell internal pressure gradient and finite thickness (thin-shell assumption); shell mass lost to
conductive evaporation into the bubble (≲1%); bubble mass in the gravity term; magnetic and
turbulent shell support; non-sphericity. **Not** legitimately omitted: any of the six rows above.

### 2.3 Bubble energy (SPEC-035) — and the exact identity `Ed_from_beta` must satisfy

```
    dE_b/dt = η_w L_mech,w + η_SN L_mech,SN − L_cool − P_b dV_b/dt − L_leak
    dV_b/dt = 4π ( R2² v2 − R1² dR1/dt )
    L_leak  = 0 when coverFraction = 1 (default; SPEC-036)
```

Because the runner receives `Ed_from_beta` instead of computing this, there is an equivalent
similarity-form statement that is the **cheapest available consistency check on the whole slice**.
From `E_b = 2π P_b (R2³ − R1³)` and `β ≡ −d ln P_b/d ln t` (SPEC-041):

```
    d ln E_b/dt = Ṗ_b/P_b + 3(R2²v2 − R1²Ṙ1)/(R2³−R1³)
    ⇒  Ė_b = E_b [ −β/t + 3(R2²v2 − R1²Ṙ1)/(R2³−R1³) ]
    ⇒  (R1 ≪ R2)   Ė_b = E_b (3α − β)/t ,      α ≡ v2 t/R2
```

Sanity check against the Weaver limit: `E_b = (5/11) L_w t` (SPEC-051) ⇒ `Ė_b = E_b/t`; and
`3α − β = 3(3/5) − 4/5 = 9/5 − 4/5 = 1` ✓. The two independent routes agree exactly. **Derived
here, high confidence.**

Generalisation (derived here): if `L_mech ∝ t^s` then `E_b ∝ t^{1+s}` and hence
`3α − β = 1 + s`. For constant `L_mech`, `3α − β = 1` must hold throughout the energy phase; the
identity breaks precisely across the onset of SNe, which is a *feature* (it flags the non-similar
window), not a bug.

### 2.4 Bubble temperature
```
    dT/dt = δ T / t ,      δ ≡ d ln T/d ln t  (SPEC-041)
```
`δ = −6/35 < 0` in the Weaver limit, so `T` must **decrease**. `t` here is the global cluster age,
not the segment-local or phase-local time — using a phase-local clock makes `α` start at 0 and
destroys the closure of §2.5.

### 2.5 Closures the runner inherits
```
    P_b = E_b/[2π(R2³ − R1³)]           ← γ = 5/3 only:  E = PV/(γ−1) = (3/2)PV,  V = (4π/3)(R2³−R1³)
    R1  = sqrt( ṗ_w /(4π P_b) )          ← SPEC-025 (the 3/4 strong-shock factor is a documented O(1) ambiguity)
    δ   = (2/7)(2α − β − 1)              ← SPEC-042 conduction closure
```
The `2π` in the first line is *not* a generic constant: it is `1/[(γ−1)·(4π/3)⁻¹]` evaluated at
γ=5/3. If `gamma_adia` is a live parameter anywhere, the hard-coded `2π` is wrong for any other γ.

---

## 3. Units and where conversions must happen

TRINITY internal ("AU") units are `[M⊙, pc, Myr]` (SPEC-090/091). Inside this module:

| quantity | AU unit | cgs |
|---|---|---|
| `t`, `dt_segment` | Myr | 3.15576e13 s |
| `R2`, `R1`, `rCloud` | pc | 3.0857e18 cm |
| `v2` | pc/Myr | 0.977781 km/s |
| `E_b` | M⊙ pc² Myr⁻² | 1.90148e43 erg |
| `Ed_from_beta` (`Ė_b`), `L_cool`, `L_mech` | M⊙ pc² Myr⁻³ | 6.0255e29 erg/s |
| `P_b`, `P_HII`, `P_ISM` | M⊙ pc⁻¹ Myr⁻² | 6.4721e-13 dyn cm⁻² |
| forces in `ForceProperties` | M⊙ pc Myr⁻² | 6.1623e24 dyn |
| `mShell`, `M_cluster` | M⊙ | |
| `T`, `Td_from_delta` | K, K/Myr | |
| `G` | 4.4985e-3 pc³ M⊙⁻¹ Myr⁻² | |

**Expectation:** *no* unit conversion should appear inside `get_ODE_implicit_pure` or
`compute_forces_pure`. Those are hot, pure functions; every conversion belongs at the module
boundaries (SPS loader, cooling table, `.param` ingestion). A `cvt.*` call inside the RHS is a
smell; a *mixture* (e.g. `c_light` in cm/s used against `L_bol` in AU) is a ~10²³ error that
usually fails loudly, but the dangerous ones are the near-unity ones: `km/s` vs `pc/Myr` differ by
only 2.3% (SPEC-092.6) and `μ_H = 1.4` vs `μ_ion = 0.609` differ by 2.3× (SPEC-092.1).

Scale awareness matters for the numerical constants: with `L_w ~ 10⁴⁰ erg/s ≈ 1.7×10¹⁰` in AU
luminosity units and `t ~ 1 Myr`, `E_b ~ 10¹⁰` while `v2 ~ 10` and `R2 ~ 10`. **The state vector
spans ~9 decades**, so a *scalar* `ODE_ATOL` cannot be right for all four components.

---

## 4. Validity regime, and what ends it

The energy-driven description (SPEC-011) requires simultaneously:

1. `t_cool,bubble ≫ t_dyn = R2/v2` — the hot bubble retains its thermal energy.
2. `R1 ≪ R2` — thin free-wind zone; needed for `V_b ≈ (4π/3)R2³` and for the similarity structure.
3. The *shell* cools fast (strong radiative shock) so it is geometrically thin — note this is the
   **opposite** requirement to (1) and applies to different gas.
4. `M_sh ≫ M_ejected` — past free expansion (SPEC-056).
5. Unsaturated conduction at the wall (Cowie–McKee; saturation parameter σ₀ ≲ 1, SPEC-044).
6. Spherical symmetry, static ambient medium, single cluster.

It ends when any of these fails. The physically correct exits:

| exit | physics | criterion |
|---|---|---|
| catastrophic cooling | (1) fails; bubble stops being a reservoir | `L_loss/L_gain → 1`, i.e. `Ė_b/L_gain → 0` (SPEC-013); TRINITY fires at 0.05 (`cooling_balance`) or at 0 (`ebpeak`) |
| blowout / venting | shell reaches the cloud edge, bubble vents into the ISM and depressurises | `R2 ≥ k·rCloud` (SPEC-014 Reading C) |
| gravitational recapture | feedback loses; shell stalls then falls back | `v2 → 0` with `dv2/dt < 0`, then `v2 < 0` (SPEC-032) |
| shell dispersal | shell indistinguishable from ambient | `shell_nMax < n_ISM` sustained (SPEC-102) |
| feedback exhausted | end of SPS table | cluster age > table span |
| numerical cutoffs | not physics | `stop_t`, `stop_r`, `MAX_SEGMENTS` |

Important caveat carried from SPEC-015: a 1-D conduction bubble stays energy-driven **longer than
reality**. So a correct implementation of this phase is expected to over-predict the phase
duration; that is a model limitation, not a code bug, and it is what `cooling_boost_*` patches.

---

## 5. Asymptotics the implementation must satisfy

Uniform medium, constant `L_w`, no gravity/radiation/`P_ext` (SPEC-050/051/052):

```
    R2 = 0.762934 (L_w/ρ₀)^{1/5} t^{3/5}      ξ_E = (250/308π)^{1/5}
    v2 = (3/5) R2/t  ∝ t^{−2/5}
    E_b/(L_w t) = 5/11 = 0.454545…
    P_b = 5 L_w t/(22π R2³) ∝ t^{−4/5}
    (α, β, δ) = (3/5, 4/5, −6/35)
```

Power-law ambient `ρ ∝ r^{−w}`, `w = |densPL_alpha| ∈ [0,2]` (SPEC-053 plus my own extension):

```
    η ≡ 3/(5−w) ,  R2 ∝ t^η ,  v2 = η R2/t ∝ t^{η−1}
    E_b/(L_w t) = 1/(1+2η)
    α = η                              (definition)
    β = 3η − 1                         (from E_b ∝ t and P_b = E_b/(2πR2³): derived here)
    δ = −(2/7) η                       (substitute α, β into SPEC-042: δ = (2/7)(2α−β−1))
```
Checks: `w=0 ⇒ (η,α,β,δ) = (0.6, 0.6, 0.8, −6/35)` ✓ reproduces the three `default.param`
constants; `w=2 ⇒ (1, 1, 2, −2/7)` and `E_b/(L_w t) = 1/3` ✓ matches SPEC-053's equipartition
check. **These four generalised values are derived here and I have high confidence in them given
SPEC-042 and SPEC-053; I could not check them against the Rahner thesis.**

Monotonicity / sign conventions that must hold in the energy phase:
- `R2` strictly increasing while `v2 > 0`; `v2 > 0` for the whole phase in a healthy run.
- `dv2/dt < 0` (deceleration) **only while `R2 < rCloud` and `w < 2`** — since `v2 ∝ t^{η−1}` and
  `η < 1 ⇔ w < 2`. At `w = 2` the shell coasts; crossing the cloud edge into the low-density ISM
  the shell **accelerates**. A monotonic-deceleration assertion is therefore wrong in general.
- `P_b` decreasing (`β > 0`), `T` decreasing (`δ < 0`), `E_b` increasing until the `ebpeak`.
- `α = v2 t/R2 → 3/(5−w)`; a run that relaxes to something else in the constant-`L_w` window has
  either the wrong clock origin or a spurious force term.

---

## 6. Per-function expectations

### `classify_energy_collapse(Eb)` — L184
Must partition the bubble energy into at least: **non-finite** (NaN/Inf → hard error), **`E_b ≤ 0`**
(unphysical: `P_b = E_b/2π(R2³−R1³) < 0` would *pull the shell inward* with a spurious
`4πR2²|P_b|` force — a silent sign inversion of the dominant term), **`0 < E_b ≤
ENERGY_HANDOFF_FLOOR`** (bubble has effectively collapsed; hand off to the momentum phase), and
**healthy**. The `E_b ≤ 0` branch must never fall through into continued integration.

*Threshold concern:* `ENERGY_HANDOFF_FLOOR` in absolute AU units is scale-dependent. Across the
shipped `paperII` grid (`mCloud` 10⁴ → 5×10⁹, `sfe` 0.01 → …) `M_cluster` and hence `L_w` and `E_b`
span ~6 decades, so one absolute floor either never fires or fires immediately at the small-mass
end. The scale-free formulation is `E_b ≤ ε · L_gain · t` or `E_b ≤ ε · E_b,peak`.

### `_inflow_frac_thickness(v_arr, r_arr) -> float` — L212
Diagnostic on the solved bubble interior: the fraction of the bubble's **radial thickness** in
which the flow is inward. Expected:
```
    frac = Σ_{i: inflow} |Δr_i|  /  (r_max − r_min)      ∈ [0, 1], dimensionless, unit-agnostic
```
It must be **thickness-weighted, not sample-count-weighted** — bubble grids are non-uniform
(clustered near the conduction front where `n ∝ (1−ξ)^{−2/5}` steepens), so `np.mean(v_arr < 0)`
would systematically over-weight the wall region and mis-report. Guards required for `len < 2`,
zero span, non-monotonic `r_arr`, all-NaN. Physically this measures how far the (β,δ) solution has
departed from a clean outflowing Weaver interior: mass evaporated off the wall must be redistributed
inward, so a *small* inflow region is normal; a large one means the assumed structure is degenerate.
**The frame convention is load-bearing and I cannot fix it from the signature**: "inflow" is `v < 0`
in the lab frame but `v < v2` in the contact-discontinuity frame, and those differ over most of the
bubble. Whichever is meant, the array must cover only `R1 ≤ r ≤ R2`.

### `evaluate_r1_shadow(R2, rCloud, edot_balance, k_blowout=1.0)` — L233
"r1" is the SPEC-014 alias `r1 = blowout,ebpeak`; "shadow" = evaluated for logging whether or not
those triggers are active. Expected returns: `(blowout_fired, ebpeak_fired)` with
```
    blowout_fired = ( R2 ≥ k_blowout · rCloud )         upward crossing, direction +1
    ebpeak_fired  = ( edot_balance ≤ 0 )                downward crossing, direction −1
```
`k_blowout` must multiply `rCloud`, not `R2` (with the default 1.0 the two are indistinguishable —
a unit test at `k ≠ 1` is required to separate them). `ebpeak` is the threshold-free "bubble energy
has peaked" event: `Ė_b = 0`. Note `edot_balance` is ambiguous between `Ė_b` itself and the
normalised `Ė_b/L_gain`; the **sign test is identical** either way since `L_gain > 0`, so the
ambiguity is harmless *here* but matters for the `cooling_balance` threshold comparison, which must
use the normalised form (a bare `Ė_b ≤ 0.05` would compare an energy rate to a dimensionless
number).

Consistency requirement across the two triggers: if `cooling_balance` fires at
`Ė_b/L_gain ≤ 0.05` and `ebpeak` at `Ė_b ≤ 0`, then in any monotone-decline run **cooling_balance
must fire strictly first** (SPEC-014 Reading B says ebpeak is "strictly later"). If the code's
`L_loss` for `cooling_balance` *excludes* the `P_b dV/dt` work, that ordering breaks and the two
triggers stop being the same quantity at two thresholds — the audit should pin which `L_loss` is used.

### `parse_transition_triggers(transition_trigger)` / `_VALID_TRIGGERS` — L249/L252
Config parsing at a **trust boundary**: an unrecognised trigger name must raise, never be silently
dropped. A typo (`ebPeak`, `cooling-balance`) that silently yields an empty or partial trigger set
changes the predicted transition time — the code's headline output — with no diagnostic. Expected:
accept a comma/space-separated string (and probably a list), normalise case/whitespace, expand
`r1 → {blowout, ebpeak}`, reject unknown tokens, reject an empty result, return a deterministic
container. `_VALID_TRIGGERS` should be exactly `{cooling_balance, blowout, ebpeak}` plus the alias.

### `r1_transition_decision(active_triggers, blowout_fired, ebpeak_fired)` — L275
Must be a **disjunction over the *enabled* triggers only**:
```
    fire = ( "blowout" ∈ active and blowout_fired ) or ( "ebpeak" ∈ active and ebpeak_fired )
```
Two traps: (i) `and` instead of `or` across the two — a bubble that peaks but never blows out would
then never transition, and the run would carry an energy-driven bubble into a regime where it is
radiating away almost all its input; (ii) honouring a `*_fired` flag whose trigger is *not* in
`active_triggers`, which makes the `transition_trigger` parameter inert (and would make the
`cooling_balance` default silently equivalent to `r1`).

### `compute_max_dex_change(params_before, params_after, keys) -> float` — L289
```
    max_dex = max over k in keys of | log10( after[k] / before[k] ) |        (≥ 0, dimensionless)
```
A ratio-in-dex measure is the right choice (scale-free, so it works across a 6-decade sweep grid),
but it is only defined for strictly positive quantities. Required guards: key missing from either
dict, value ≤ 0, non-finite. The **failure mode to check is the quiet one**: if a guard returns
`0.0` for a key that actually blew up (e.g. `before = 0`), the controller sees "nothing changed"
and *grows* the segment exactly when it should shrink. Returning `NaN` is equally bad because
`NaN > threshold` is `False` — also no shrink. Prefer: skip undefined keys, and if *all* keys are
undefined return `+inf` (force the minimum segment), not `0.0`.

### `get_monitor_values(params) -> dict` — L412
Extracts `ADAPTIVE_MONITOR_KEYS` from the state container into a plain dict. **The critical
expectation is aliasing:** this is called before and after a segment on what is very likely the
*same mutable* `params` object. It must return **copies of scalar floats**, not references/views
into `params`. If it returns anything that aliases live state, `params_before` mutates along with
`params_after`, `compute_max_dex_change` returns 0.0 forever, and the entire adaptive controller is
silently dead — the run completes, looks fine, and is under-resolved wherever it mattered.

Second expectation: the monitored keys must be **positive-definite** (`P_b`, `T`, `L_cool`, `R2`,
`E_b`, `R1`). `β` and `δ` cross zero and `v2` can, so a dex measure on them is undefined at exactly
the moment the controller is most needed.

### `update_unconverged_streak(streak, converged, t_now, total_residual) -> int` — L332
`return 0 if converged else streak + 1`; monotone, never negative; `t_now`/`total_residual` are for
the warning emitted at `BETADELTA_UNCONVERGED_WARN_STREAK`. The trap is failing to reset on
success: a streak that only accumulates will eventually trip `BETADELTA_DT_SHRINK_MAX_STREAK` or
`NO_ROOT_HANDOFF_STREAK` in a perfectly healthy run and terminate the energy phase early — which
looks exactly like a physical transition.

### `betadelta_phase_summary(solve_count, converged_count, no_root_count) -> tuple` — L360
Aggregate diagnostics; must guard `solve_count == 0` (a phase that exits on its first segment) and
must satisfy `converged_count + no_root_count ≤ solve_count`. Purely diagnostic, but the numbers it
reports are what a future session uses to decide whether a run was healthy, so a division-by-zero
NaN or an inverted ratio is a real (if low-severity) defect.

### `next_dt_segment(dt_segment, max_dex_change, unconverged_streak) -> float` — L376
Expected controller:
```
    if unconverged_streak ≥ BETADELTA_DT_SHRINK_MAX_STREAK : shrink (forced)
    elif max_dex_change > ADAPTIVE_THRESHOLD_DEX           : dt / ADAPTIVE_FACTOR
    elif max_dex_change < ADAPTIVE_THRESHOLD_DEX / m       : dt * ADAPTIVE_FACTOR    (m > 1, hysteresis)
    clamp to [DT_SEGMENT_MIN, DT_SEGMENT_MAX];  never ≤ 0, never NaN
```
Hysteresis (`m > 1`) matters: grow-and-shrink on the same threshold oscillates. Two deeper
expectations:

1. **The segment cap must be relative, not absolute.** The frozen-rate approximation is accurate to
   `O(Δ² Ë_b)`, so the controlling requirement is `Δ ≪ t_dyn = R2/v2`, `Δ ≪ t_cool`, and `Δ ≪ t`
   (since `α`, `β`, `δ` are *logarithmic* time derivatives, a segment comparable to `t` is a
   100% error in the similarity extrapolation). A fixed `DT_SEGMENT_MAX` in Myr is scale-dependent:
   a dense `n_core = 10⁵ cm⁻³` cloud has `t_dyn ~ 10⁻²` Myr where a `0.1` Myr cap is meaningless.
2. **Accept-and-shrink lags by one segment.** Measuring `max_dex_change` on the *completed* segment
   and applying the shrink to the *next* one means the first violating segment is accepted with
   unbounded error. A robust controller **rejects and retries** the segment that violated the
   threshold. This matters most at the end of the phase, where `L_cool` grows superexponentially —
   i.e. exactly where the transition time is decided.

### `ForceProperties` (L444) / `compute_forces_pure(R2, mShell, Pb, shell_props, params)` (L460)
Must return the complete, non-overlapping force inventory of §2.2 such that SPEC-007 closes:
`M_sh dv2/dt = F_drive + F_rad − F_grav − F_ext − F_sweep`. Note the signature has **no `v2`
argument**, so the sweep-up drag `4πR2²ρ_amb v2|v2|` *cannot* be computed here — it must be applied
in the ODE RHS, and therefore must **not** also appear in `ForceProperties` under a name like
`F_ram`. The `F_ram_wind`/`F_ram_SN` entries of SPEC-006 are `ṗ_w` and `ṗ_SN`; in the energy phase
these are **diagnostics only** and must not enter `F_drive` (SPEC-022).

`F_drive` here must use `P_drive = max(P_b, P_HII)`, so `shell_props` must supply `P_HII` and the
optical depths (`τ_UV`, `τ_IR` or `f_abs`, `Σ_sh`) for `F_rad`. `params` supplies `L_bol`,
`M_cluster`, `P_ISM`, `G`. Purity: no module-global reads, no mutation of `shell_props`/`params`.

Known non-conservation to record (SPEC-035 trap, not necessarily a bug): when `P_HII > P_b`, the
shell receives work at `4πR2²P_HII v2` while the bubble loses `4πR2² P_b v2`. The `max()` is not
conservative; if the code does not document this, it is an undocumented energy source/sink whose
size is exactly the amount by which TRINITY's headline term exceeds the bubble term.

### `get_ODE_implicit_pure(...)` — L586
```
    returns  [ y[1] , dv2/dt , Ed_from_beta , Td_from_delta ]
```
Expectations, in order of severity:
1. `dR2/dt` must be `y[1]` (the integrator's current `v2`), never a frozen snapshot value.
2. `dv2/dt` must be §2.2 divided by `M_sh`, with `M_sh = M_enc(R2)` recomputed at the current `R2`
   (SPEC-021) — the shell keeps sweeping *within* the segment.
3. `P_b` must be recomputed from the current `(E_b, R2, R1)`, not frozen: `P_b ∝ t^{−4/5}` decays by
   ~8% over a segment with `Δ/t ~ 0.1`, and freezing it while `E_b` is a live state variable is
   internally inconsistent.
4. `Ed_from_beta` must be returned **as given**. It already contains `L_gain − L_cool − P_b dV/dt`;
   subtracting a `P_b dV/dt` again inside the RHS double-counts the work term — which, in the
   Weaver limit, is `2.111/3.870 = 55%` of the injected power (SPEC-050's derivation), so the error
   is enormous and would show as `E_b/(L_w t)` far below 5/11.
5. `Td_from_delta` must be returned as given, with the sign of `δ` preserved (`δ < 0`).
6. The function must be **pure**: same inputs ⇒ same output, no module-level globals, no mutation
   of `snapshot`. CLAUDE.md records that trinity leaks module-level state; the `_pure` suffix is a
   promise that must be testable by calling it twice in a permuted order.
7. `params_for_feedback` must supply the SPS drivers at the **current** `t` (cluster age), not at
   the segment start — `L_mech` changes by ~an order of magnitude across the first SN at ~3–4 Myr,
   and a frozen value across a segment straddling it is a first-order error in the dominant source
   term. If the drivers *are* frozen by design, that is a documented approximation whose error must
   be bounded by the segment controller (which cannot see it, since `L_mech` is unlikely to be in
   `ADAPTIVE_MONITOR_KEYS`).

### `ImplicitPhaseResults` (L569) / `run_phase_energy(params)` (L631)
The driver loop. Expected per segment: (i) solve the bubble structure for `(β, δ)` → `L_cool`, `T`,
`R1`, `Ė_b`, `Ṫ`; (ii) build the immutable `ODESnapshot`; (iii) integrate `[t, t+Δ]` with
`ODE_METHOD/RTOL/ATOL` and terminal events; (iv) test the transition triggers; (v) update `Δ`;
(vi) record. Required properties:

- **Event handling.** Terminal events with explicit `direction`: `R2 − k·rCloud` (+1),
  `E_b − ENERGY_HANDOFF_FLOOR` (−1), `v2 − VELOCITY_THRESHOLD_COLLAPSE` (−1), `R2 − stop_r` (+1),
  `t − stop_t` (+1). **Degenerate-root trap:** SciPy detects events by sign change over a step, so
  an event that is *already* satisfied at the segment start (the previous segment terminated
  exactly on it) will not re-fire — the loop then either spins or steps straight past the
  condition. A robust runner tests every exit condition at the segment start, before integrating.
- **Event resolution vs segment length.** `cooling_balance` and `ebpeak` depend on `L_cool`, which
  is only known at segment boundaries — so those transitions are resolved only to `±Δ`. The
  reported transition time therefore carries an error of order `DT_SEGMENT_MAX` unless the runner
  refines it (bisect / re-solve the last segment). Since the transition time *is* the headline
  prediction, this must be either refined or bounded.
- **Root-find failure ≠ physical transition.** `NO_ROOT_HANDOFF_STREAK` hands off after repeated
  no-root results. Losing the root is genuinely the signature of catastrophic cooling — but it is
  *also* the signature of a bad bracket or a bad initial guess. Conflating them silently converts a
  numerical failure into a plausible-looking physical transition at the wrong time. The
  discriminator is available for free: at handoff, `L_cool/L_gain` must be ≈ 1. If the root is lost
  while `L_cool/L_gain ≪ 1`, that is numerics and must be raised, not absorbed. Likewise, `dt`
  shrinking (`BETADELTA_DT_SHRINK_MAX_STREAK`) must be attempted **before** any handoff, and
  reaching `DT_SEGMENT_MIN` while still unconverged must be an explicit recorded failure — never a
  silent continue with the last-good `(β, δ)`.
- **`MAX_SEGMENTS`** exhaustion is a numerical cutoff (SPEC-100 last row) and must be recorded as
  such in the termination block (SPEC-105), not returned as a normal completion.
- **Collapse regime.** If `v2 < 0` is reachable here (`VELOCITY_THRESHOLD_COLLAPSE`,
  `DT_SEGMENT_COLLAPSE` imply it is), then `Ṁ_sh` must be clamped to ≥ 0 (a shell falling back
  through the evacuated interior does not un-sweep, nor does it sweep the bubble) and the drag term
  must use `v2|v2|`. `VELOCITY_THRESHOLD_EXTREME` should be a sanity ceiling: `v2` can never exceed
  the wind terminal speed `v_w = 2L_w/ṗ_w` (SPEC-071), ~1000–3000 km/s ≈ 1000–3000 pc/Myr.
- **Solver choice.** With `Ė_b` and `Ṫ` frozen, `E_b` and `T` are *linear in `t`* within a segment —
  the operator split has deliberately removed the stiff cooling term from the ODE. What remains
  (`R2`, `v2`) is only mildly stiff, so `ODE_METHOD` need not be implicit; but if `LSODA` is used,
  note SPEC-023: `max(P_b, P_HII)` puts a **kink** in the RHS, and adaptive/stiff solvers chatter or
  mis-estimate the Jacobian at a kink. `ODE_MIN_STEP` is accepted only by `LSODA` in SciPy — passed
  to `RK45`/`BDF`/`Radau` it is ignored with at most a warning, so a "minimum step" guard can be
  silently absent.
- **`ODE_ATOL`** must be per-component (§3): the state spans ~9 decades. A scalar tuned for `E_b`
  destroys `v2`; a scalar tuned for `v2` is merely irrelevant for `E_b` (rtol governs) — so the
  asymmetric risk is a large scalar atol.
- **`COOLING_UPDATE_INTERVAL`.** Refreshing the cooling table/structure only every N segments (or
  every Δt) leaves `L_cool` stale in between. The refresh must be short compared with the SPS
  variation timescale and must be **forced** across the age-indexed non-CIE file boundaries
  (SPEC-083) and across the first SN, where the true `L_cool` steps discontinuously.
- **Continuity at the phase boundary** (SPEC-016): `dv2/dt` must not jump across entry from phase 1a
  or exit into the transition phase by more than integrator tolerance.
- **`ImplicitPhaseResults`** must carry the state at the **event time**, not at the segment end, or
  the handoff to the next phase starts from a state that already violates the exit criterion.

---

## 7. Known traps specific to this slice

1. **γ = 5/3 coefficients.** `P_b = E_b/[2π(R2³−R1³)]` and the `(3/2)PV` energy relation are
   γ-specific. `E_b = (5/11)L_w t` and `ξ_E = (250/308π)^{1/5}` are *also* γ=5/3 results.
2. **Uniform-medium results applied to a power law.** `α=3/5`, `β=4/5`, `δ=−6/35`, `E_b/(L_w t)=5/11`
   are all `w=0` numbers. On `densPL_alpha = −2` the correct values are `1, 2, −2/7, 1/3`. Any
   hard-coded 0.6/0.8/−6/35 used as more than an *initial guess* for the root-find is a
   uniform-medium result leaking into a stratified run. (As a seed for the `(β,δ)` bracket they are
   fine and sensible.)
3. **Clock origin.** `α = v2 t/R2`, `β = −t Ṗ/P`, `δ = t Ṫ/T` and `Ṫ = δT/t` all require the *same*
   `t`, and it must be the global cluster age. A phase-local `t` silently rescales all three.
4. **Ram double-count** (SPEC-020) and **wind ram added on top of `P_b`** (SPEC-022) — two distinct
   double-counts, both of which merely make the shell "a bit faster", i.e. both are silent.
5. **`max()` kink** — non-differentiable RHS; solver chatter; and non-conservative work balance.
6. **Degenerate events** — physically correct conditions (`Ė_b = 0`, `v2 = 0`, `R2 = rCloud`) that
   are numerically degenerate when the previous segment ended exactly on them.
7. **Equation numbering** — I refuse to cite Weaver+77 equation numbers (SPEC-045); anything in the
   code citing "Weaver Eq. 20/37" is unverifiable from this audit and should be checked against the
   paper by a human, not trusted.
8. **Prefactor provenance** — SPEC-045 shows the two commonly-quoted Weaver interior prefactors
   (`1.51e6`, `2.07e6` K) are mutually inconsistent with isobaricity by a factor 3–4. Any hard-coded
   value of that family in this slice should be replaced by the structural closures (SPEC-024/042),
   which are prefactor-free.
9. **Aliasing in the before/after monitor snapshot** — the single most likely way for the adaptive
   controller to be silently inert.
10. **Streak counters as physics.** Four separate streak thresholds (`NO_ROOT_HANDOFF_STREAK`,
    `BETADELTA_UNCONVERGED_WARN_STREAK`, `BETADELTA_DT_SHRINK_MAX_STREAK`, plus `MAX_SEGMENTS`) can
    each end the energy phase. Every one of them is a *numerical* exit that will be reported
    alongside genuine physical transitions; the termination record must distinguish them.

---

## 8. Recommended executable checks (cheapest first)

| # | check | passes iff |
|---|---|---|
| C1 | pure-function unit tests on L212/L233/L252/L275/L289/L332/L360/L376 | as specified in §6; these need no simulation |
| C2 | `get_ODE_implicit_pure` called twice with permuted order | bitwise-identical output (purity) |
| C3 | within one segment, `E_b(t)` from the recorded trajectory | linear in `t` to solver tolerance (confirms `Ed_from_beta` is returned as-is, not re-corrected) |
| C4 | `Ė_b · t / E_b − (3α − β)` over the energy phase | ≈ 0 in the constant-`L_mech` window (§2.3, derived) |
| C5 | `δ − (2/7)(2α − β − 1)` | ≈ 0 (SPEC-042) |
| C6 | `α → 3/(5−w)`, `β → 3η−1`, `δ → −2η/7` for `w = 0, 1, 2` | §5 |
| C7 | `E_b/(L_mech t) → 1/(1+2η)` (= 0.4545 for `w=0`) with gravity/radiation off | SPEC-051/053 |
| C8 | force closure `M_sh dv2/dt = ΣF` from `dictionary.jsonl` | SPEC-007; catches the ram double-count |
| C9 | halve `DT_SEGMENT_MAX` and `ADAPTIVE_THRESHOLD_DEX` | transition time moves by less than the quoted precision (segment-resolution test) |
| C10 | monitor-dict aliasing | `compute_max_dex_change(before, after, keys) > 0` across a segment in which `P_b` demonstrably changed |
| C11 | at every no-root/streak handoff | `L_cool/L_gain` within a factor ~2 of 1 (numerics-vs-physics discriminator) |

Suggested configs, per CLAUDE.md: `param/simple_cluster.param` for the baseline, and the two
`f1edge_{lowdens,hidens}` edge configs for the stiff regimes; the `w`-dependence checks need
`densPL_alpha ∈ {0, −1, −2}` with a physically plausible `rCore` (~1 pc, **not** the default
0.01 pc — SPEC-063).

---

```json
[
  {"id":"S5b-C-01","file":"trinity/phase1b_energy_implicit/run_energy_implicit_phase.py","line":"L123","class":"coefficient","severity":"S3","claim":"FOUR_PI must equal 4*pi to double precision (math.pi*4 or 12.566370614359172), not a truncated literal.","evidence":"It multiplies R2^2 in every pressure-to-force conversion and in dV_b/dt; SPEC-020/024/035.","expected":"FOUR_PI == 4*math.pi exactly in binary64.","failure_scenario":"A truncated literal (12.566) biases every force and the PdV work by ~5e-4, breaking the SPEC-007 force closure and any bit-identity claim.","repro":"assert FOUR_PI == 4*math.pi","confidence":"high"},
  {"id":"S5b-C-02","file":"trinity/phase1b_energy_implicit/run_energy_implicit_phase.py","line":"L586","class":"state","severity":"S1","claim":"The RHS must return [y[1], dv2/dt, Ed_from_beta, Td_from_delta] with dR2/dt taken from the integrator's current v2 (y[1]), not from a snapshot-frozen velocity.","evidence":"Standard ODE form; the state must be self-consistent within the segment. Freezing v2 in the R equation reduces the scheme to explicit Euler in R while claiming solver tolerance.","expected":"out[0] is exactly y[1]; len(out) == len(y).","failure_scenario":"R2 and v2 decouple inside a segment; the trajectory silently loses accuracy at exactly the segment length, and rtol/atol no longer control the error.","repro":"Call get_ODE_implicit_pure with y perturbed only in y[1] and assert out[0] tracks it one-for-one.","confidence":"high"},
  {"id":"S5b-C-03","file":"trinity/phase1b_energy_implicit/run_energy_implicit_phase.py","line":"L586","class":"state","severity":"S1","claim":"dv2/dt = [4*pi*R2^2*(P_drive - P_ISM) + F_rad - F_grav - 4*pi*R2^2*rho_amb(R2)*v2*|v2|] / M_sh with P_drive = max(P_b, P_HII).","evidence":"SPEC-020 (thin-shell EOM), SPEC-022 (phase-aware P_drive: no P_ram in the energy phase), SPEC-026/027 (F_rad), SPEC-031 (F_grav).","expected":"Exactly those six contributions, each once.","failure_scenario":"A missing or extra term changes the expansion law and hence the transition time and the dispersal-vs-recollapse verdict.","repro":"SPEC-007 force closure on dictionary.jsonl: M_sh*dv2/dt - sum(recorded forces) within integrator tolerance.","confidence":"high"},
  {"id":"S5b-C-04","file":"trinity/phase1b_energy_implicit/run_energy_implicit_phase.py","line":"L586","class":"sign","severity":"S1","claim":"The sweep-up ram term appears exactly once, and only in the expanded (M dv/dt) form; it must not also be present as an F_ram entry added to the same equation.","evidence":"SPEC-020 AUDIT TRAP: -4*pi*R2^2*rho*v2^2 is already contained in d(M_sh v2)/dt via Mdot_sh*v2.","expected":"One occurrence; ForceProperties has no sweep-up term (compute_forces_pure has no v2 argument, so it cannot compute one).","failure_scenario":"Double-counted drag decelerates the shell ~2x too strongly at early times; the bubble looks over-pressured relative to the shell response and the phase-transition time shifts.","repro":"Symbolically compare the momentum RHS against the recorded F_* keys; or run with rho_amb scaled and check the drag scales linearly, not quadratically.","confidence":"high"},
  {"id":"S5b-C-05","file":"trinity/phase1b_energy_implicit/run_energy_implicit_phase.py","line":"L460","class":"regime","severity":"S2","claim":"In the energy/implicit phase the direct wind and SN ram pressures (pdot_w, pdot_SN) must NOT be added to P_drive; they are diagnostics only.","evidence":"SPEC-022: energy phase P_drive = max(P_b, P_HII); the wind momentum is already thermalised into P_b, so adding both double-counts the wind.","expected":"F_drive = 4*pi*R2^2*max(P_b, P_HII); F_ram_wind/F_ram_SN recorded but not summed into the EOM here.","failure_scenario":"Silent over-driving of the shell throughout the energy phase, largest at early times when pdot_w is largest relative to P_b*4piR2^2.","confidence":"high","repro":"Set include_PHII False and compare F_drive against 4*pi*R2^2*P_b snapshot-by-snapshot in the energy phase; any positive offset is the extra ram."},
  {"id":"S5b-C-06","file":"trinity/phase1b_energy_implicit/run_energy_implicit_phase.py","line":"L460","class":"coefficient","severity":"S1","claim":"F_grav = G*M_sh*(M_cluster + M_sh/2)/R2^2 — the shell self-gravity carries a factor 1/2, and cloud gas outside R2 contributes nothing.","evidence":"SPEC-031; self-potential of a thin shell U = -G M^2/(2R) gives -dU/dR = -G M^2/(2R^2).","expected":"Factor exactly 0.5 on M_sh; no M_cloud-outside term.","failure_scenario":"A factor-2 error in the dominant restoring force at late times; flips dispersal into re-collapse (or vice versa) for marginal clouds — the code's headline classification.","repro":"Unit-test compute_forces_pure with M_cluster=0 and check F_grav == G*M_sh^2/(2*R2^2).","confidence":"high"},
  {"id":"S5b-C-07","file":"trinity/phase1b_energy_implicit/run_energy_implicit_phase.py","line":"L586","class":"coefficient","severity":"S2","claim":"P_b = E_b/[2*pi*(R2^3 - R1^3)] is a gamma=5/3-only coefficient; if gamma_adia is a live parameter the code must use E = P V/(gamma-1) with V = (4pi/3)(R2^3-R1^3).","evidence":"SPEC-024, derived: E=PV/(gamma-1)=(3/2)PV at gamma=5/3, so P = E/[2pi(R2^3-R1^3)].","expected":"Either 2*pi with gamma pinned at 5/3 and asserted, or the general (gamma-1)/((4/3)pi) form.","failure_scenario":"Any non-5/3 gamma silently produces the wrong bubble pressure, hence the wrong expansion and the wrong R1.","repro":"grep the module for gamma/gamma_adia usage; assert the parameter is either used or validated == 5/3.","confidence":"high"},
  {"id":"S5b-C-08","file":"trinity/phase1b_energy_implicit/run_energy_implicit_phase.py","line":"L586","class":"numerical","severity":"S3","claim":"P_b inside the RHS must be recomputed from the evolving (E_b, R2) of the current y, not frozen at the segment start.","evidence":"P_b decays as t^-4/5 in the Weaver limit, so over a segment with dt/t ~ 0.1 the frozen value is ~8% high; freezing it while E_b is an integrated state variable is internally inconsistent.","expected":"P_b = f(y[2], y[0], R1) evaluated per RHS call.","failure_scenario":"Systematic over-driving within every segment, accumulating as a bias in R2(t) that no tolerance setting can detect.","repro":"Halve DT_SEGMENT_MAX; if R2(t) at matched t shifts by more than solver tolerance, a segment-frozen quantity is in the RHS.","confidence":"medium"},
  {"id":"S5b-C-09","file":"trinity/phase1b_energy_implicit/run_energy_implicit_phase.py","line":"L586","class":"divergence","severity":"S1","claim":"Ed_from_beta already equals L_gain - L_cool - P_b dV_b/dt; the RHS must return it unmodified and must not subtract a PdV work term again.","evidence":"SPEC-035 energy equation; in the Weaver limit the PdV term is 2.111/3.870 = 55% of L_w (SPEC-050 derivation).","expected":"out[2] == Ed_from_beta exactly.","failure_scenario":"Double-subtracting the work term removes ~55% of the injected power; E_b/(L_w t) collapses far below 5/11 and the bubble cools (and the phase transitions) far too early.","repro":"Within a segment, E_b(t) must be exactly linear in t with slope Ed_from_beta; and E_b/(L_mech*t) -> 0.4545 in a gravity/radiation-free uniform run (SPEC-051).","confidence":"medium"},
  {"id":"S5b-C-10","file":"trinity/phase1b_energy_implicit/run_energy_implicit_phase.py","line":"L586","class":"sign","severity":"S2","claim":"Td_from_delta = delta*T/t with delta<0 in the Weaver regime, returned with its sign preserved; t is the global cluster age.","evidence":"SPEC-041 (delta = dlnT/dlnt), SPEC-042 (delta = -6/35 at Weaver values).","expected":"out[3] == Td_from_delta; T decreasing through the energy phase.","failure_scenario":"A flipped sign makes the bubble heat up as it expands, inverting the conduction closure and the cooling rate.","repro":"Assert T0 is monotonically decreasing over the constant-Lmech window of an energy-phase run.","confidence":"high"},
  {"id":"S5b-C-11","file":"trinity/phase1b_energy_implicit/run_energy_implicit_phase.py","line":"L631","class":"units","severity":"S1","claim":"alpha = v2*t/R2, beta = -t dlnP_b/dt, delta = t dlnT/dt and dT/dt = delta*T/t must all use the same global cluster-age clock, not a phase-local or segment-local time.","evidence":"SPEC-041 defines them as logarithmic derivatives with respect to the simulation clock; the Weaver values (3/5, 4/5, -6/35) only hold on that clock.","expected":"One t, threaded from the SPS clock.","failure_scenario":"A phase-local t makes alpha start at 0 and grow, corrupting the (beta,delta) root-find, the conduction closure and hence L_cool and the transition time — with no visible error.","repro":"Extract alpha from snapshots; it must relax to 3/(5-w), not start near zero at the phase start.","confidence":"high"},
  {"id":"S5b-C-12","file":"trinity/phase1b_energy_implicit/run_energy_implicit_phase.py","line":"L586","class":"exponent","severity":"S2","claim":"The identity Edot_b = E_b*(3*alpha - beta)/t must hold (R1<<R2 limit), i.e. 3*alpha - beta = 1 + s where L_mech ~ t^s.","evidence":"Derived here: E_b = 2pi P_b R2^3 gives dlnE/dlnt = -beta + 3alpha; cross-checked against SPEC-051 (E_b=(5/11)L_w t so dlnE/dlnt=1) with alpha=3/5, beta=4/5 giving exactly 1.","expected":"|3*alpha - beta - 1| < 0.05 in the constant-Lmech window of the energy phase.","failure_scenario":"Violation means Ed_from_beta and the (beta,delta) solve disagree about the bubble's own thermodynamics — the implicit closure is not closed.","repro":"Compute from dictionary.jsonl: alpha=v2*t/R2, beta from finite-differenced ln Pb, assert 3a-b ~ 1 before the first SN.","confidence":"high"},
  {"id":"S5b-C-13","file":"trinity/phase1b_energy_implicit/run_energy_implicit_phase.py","line":"L631","class":"exponent","severity":"S3","claim":"On a power-law cloud with w=|densPL_alpha|, the solved similarity triple must approach (alpha,beta,delta) = (eta, 3*eta-1, -2*eta/7) with eta = 3/(5-w); the defaults 0.6/0.8/(-6/35) are the w=0 case only.","evidence":"SPEC-053 gives eta=3/(5-w) and E_b/(L t)=1/(1+2eta); with E_b ~ t and P_b = E_b/(2pi R2^3), beta = 3eta-1; SPEC-042's delta=(2/7)(2alpha-beta-1) then gives delta=-2eta/7. Checks: w=0 -> (0.6,0.8,-6/35); w=2 -> (1,2,-2/7).","expected":"Measured triple within a few percent of these in a clean power-law energy-phase run.","failure_scenario":"If 0.6/0.8/-6/35 are used as fixed values rather than as root-find seeds, the bubble structure (hence L_cool and the transition time) is wrong for every alpha != 0 run, including the paper grid.","repro":"Run densPL_alpha = 0, -1, -2 (rCore ~ 1 pc) and compare the recorded alpha/beta/delta against 3/(5-w), 3eta-1, -2eta/7.","confidence":"medium"},
  {"id":"S5b-C-14","file":"trinity/phase1b_energy_implicit/run_energy_implicit_phase.py","line":"L631","class":"exponent","severity":"S1","claim":"With gravity, radiation and P_ext off and constant L_w in a uniform medium, the phase must reproduce R2 = 0.762934 (L_w/rho0)^(1/5) t^(3/5), v2 = (3/5)R2/t, E_b/(L_w t) = 5/11.","evidence":"SPEC-050/051, derived twice there and re-derived here structurally.","expected":"Slope 3/5 in log-log and the dimensionless ratio 0.4545 +/- 1%.","failure_scenario":"Any deviation localises a defect in the momentum/energy coupling of this runner rather than in the sub-modules.","repro":"Dedicated regression run with the extra force terms disabled; assert both the exponent and the dimensionless energy ratio (the latter is unit-convention-immune).","confidence":"high"},
  {"id":"S5b-C-15","file":"trinity/phase1b_energy_implicit/run_energy_implicit_phase.py","line":"L586","class":"units","severity":"S1","claim":"The RHS and force builder must work entirely in internal AU units [Msun, pc, Myr]; no cgs<->AU conversion may occur inside them, and G must be 4.4985e-3 pc^3 Msun^-1 Myr^-2.","evidence":"SPEC-090/091; conversions belong at module boundaries.","expected":"No cvt.* calls in the hot path; G in AU.","failure_scenario":"A km/s vs pc/Myr slip is only 2.3% and therefore invisible; a G in cgs is a 1e-10 error that would be caught, but a mu_H=1.4 vs mu_ion=0.609 slip in rho_amb is 2.3x and plausible.","repro":"Dimensional spot-check: at L36=1, n_H=1 cm^-3, t=1 Myr the run must give R2 = 26.2 pc with rho0 = 1.4 n_H m_H (NOT 28 pc, which is the mu=1 number).","confidence":"high"},
  {"id":"S5b-C-16","file":"trinity/phase1b_energy_implicit/run_energy_implicit_phase.py","line":"L184","class":"silent-failure","severity":"S1","claim":"classify_energy_collapse must separate non-finite Eb, Eb<=0, 0<Eb<=ENERGY_HANDOFF_FLOOR and healthy, and Eb<=0 must terminate/hand off rather than continue.","evidence":"P_b = E_b/[2pi(R2^3-R1^3)]: a negative E_b produces a negative P_b and hence an inward 4piR2^2|P_b| force on the shell — a sign inversion of the dominant driving term.","expected":"Four-way classification; the non-positive branch is terminal.","failure_scenario":"The shell is sucked inward by a fictitious negative pressure, producing a spurious 're-collapse' fate that looks physical in the output.","repro":"Unit-test classify_energy_collapse(-1.0), (0.0), (nan), (inf) and assert distinct non-continue classifications.","confidence":"high"},
  {"id":"S5b-C-17","file":"trinity/phase1b_energy_implicit/run_energy_implicit_phase.py","line":"L181","class":"regime","severity":"S3","claim":"ENERGY_HANDOFF_FLOOR should be relative (eps*L_gain*t or eps*Eb_peak), not an absolute AU-unit constant.","evidence":"E_b scales linearly with M_cluster, which spans ~6 decades across param/paperII_grid_sweep.param (mCloud 1e4..5e9, sfe 0.01..).","expected":"A scale-free floor, or an absolute floor with a documented validity range.","failure_scenario":"At the low-mass end an absolute floor fires immediately (spurious instant handoff); at the high-mass end it never fires and Eb integrates to zero and through it.","repro":"Run the smallest and largest sweep cells and check at what Eb/Eb_peak the handoff fires; it should be the same ratio.","confidence":"medium"},
  {"id":"S5b-C-18","file":"trinity/phase1b_energy_implicit/run_energy_implicit_phase.py","line":"L212","class":"numerical","severity":"S3","claim":"_inflow_frac_thickness must be radial-thickness weighted, dimensionless, bounded in [0,1], and guarded against len<2, zero span, non-monotonic r and NaN.","evidence":"Bubble grids are non-uniform (clustered where n ~ (1-xi)^(-2/5) steepens), so a sample-count fraction (mean of v<0) systematically over-weights the wall region.","expected":"sum(|dr| over inflow intervals)/(r_max-r_min).","failure_scenario":"A count-weighted fraction mis-reports the health of the bubble solution, so either a healthy solve is rejected or a degenerate one is accepted, moving the phase-exit time.","repro":"Unit test with a log-spaced r grid where v<0 on the coarse half only: thickness fraction and count fraction differ by a factor of several.","confidence":"medium"},
  {"id":"S5b-C-19","file":"trinity/phase1b_energy_implicit/run_energy_implicit_phase.py","line":"L212","class":"regime","severity":"S3","claim":"The inflow test must state its frame (v<0 lab-frame vs v<v2 contact-discontinuity frame) and must be applied only over the bubble interval R1<=r<=R2.","evidence":"Evaporated shell gas enters the bubble moving inward relative to the CD at v2; lab-frame and CD-frame inflow regions differ over most of the bubble.","expected":"A single documented frame, consistent with the frame the bubble-structure module returns.","failure_scenario":"A frame mismatch makes the diagnostic report ~0 or ~1 almost always, so it either never fires or always fires — in both cases it silently stops being a guard.","repro":"Check the sign convention of the velocity array produced by the bubble-structure solve against the interpretation used here.","confidence":"low"},
  {"id":"S5b-C-20","file":"trinity/phase1b_energy_implicit/run_energy_implicit_phase.py","line":"L233","class":"sign","severity":"S2","claim":"evaluate_r1_shadow must fire blowout on an upward crossing R2 >= k_blowout*rCloud and ebpeak on a downward crossing edot_balance <= 0.","evidence":"SPEC-014 Readings B and C: ebpeak is the threshold-free Edot_b<=0 turnover; blowout is the geometric cloud-edge crossing.","expected":"(R2 >= k_blowout*rCloud, edot_balance <= 0).","failure_scenario":"An inverted comparison fires the transition at t=0 (Edot_b>0 early) or never; either way the energy/momentum split time is wrong by the whole phase duration.","repro":"Unit test the four sign combinations.","confidence":"high"},
  {"id":"S5b-C-21","file":"trinity/phase1b_energy_implicit/run_energy_implicit_phase.py","line":"L233","class":"coefficient","severity":"S3","claim":"k_blowout multiplies rCloud (the threshold), not R2; default 1.0 means the criterion is exactly R2 >= rCloud.","evidence":"SPEC-014 Reading C / SPEC-104: the geometric event is the cloud edge; k is a tuning factor on where 'the edge' is taken to be.","expected":"R2 >= k*rCloud.","failure_scenario":"With the default 1.0 the two placements are indistinguishable, so a misplacement ships undetected and only bites when someone sets k != 1 in a sweep — inverting the threshold direction.","repro":"Unit test with k=0.5 and k=2 and check which side moves.","confidence":"high"},
  {"id":"S5b-C-22","file":"trinity/phase1b_energy_implicit/run_energy_implicit_phase.py","line":"L252","class":"silent-failure","severity":"S2","claim":"parse_transition_triggers must reject unknown trigger names (and an empty result) with an exception, and must expand the alias r1 -> {blowout, ebpeak}.","evidence":"SPEC-014 lists exactly cooling_balance, blowout, ebpeak and the alias r1; this is a config trust boundary.","expected":"Raise on unknown tokens; case/whitespace tolerant; deterministic container.","failure_scenario":"A typo ('ebPeak') silently yields a smaller trigger set, so the run transitions later (or never) than intended and the headline transition time is wrong with no diagnostic.","repro":"assert parse_transition_triggers('ebPeak') raises; assert parse_transition_triggers('r1') == {'blowout','ebpeak'}.","confidence":"high"},
  {"id":"S5b-C-23","file":"trinity/phase1b_energy_implicit/run_energy_implicit_phase.py","line":"L275","class":"regime","severity":"S2","claim":"r1_transition_decision must be an OR over the ENABLED triggers only: fire iff ('blowout' in active and blowout_fired) or ('ebpeak' in active and ebpeak_fired).","evidence":"SPEC-014: 'fires on whichever occurs first'.","expected":"Disjunction, gated by membership in active_triggers.","failure_scenario":"AND delays the transition until both events occur — a bubble that peaks but never reaches the cloud edge would stay energy-driven forever. Ignoring active_triggers makes the transition_trigger parameter inert, so the documented default (cooling_balance) is silently not what runs.","repro":"Truth-table unit test over active in {{}, {blowout}, {ebpeak}, both} x fired flags.","confidence":"high"},
  {"id":"S5b-C-24","file":"trinity/phase1b_energy_implicit/run_energy_implicit_phase.py","line":"L289","class":"silent-failure","severity":"S3","claim":"compute_max_dex_change must return max_k |log10(after[k]/before[k])| with guards; when a key is undefined (missing, <=0, non-finite) it must NOT silently contribute 0.0, and an all-undefined result must force the minimum segment, not the maximum.","evidence":"The controller shrinks when the result exceeds ADAPTIVE_THRESHOLD_DEX; NaN > threshold is False in Python, so a NaN result also fails to shrink.","expected":"Non-negative float; undefined keys skipped and reported; all-undefined -> +inf (or an explicit error).","failure_scenario":"The adaptive controller grows the segment exactly when the state is blowing up, and the frozen-rate operator split is then applied across a segment where Eb changes by decades.","repro":"Unit test with before={'Pb':0.0}, with a missing key, and with NaN; assert the returned value forces a shrink.","confidence":"medium"},
  {"id":"S5b-C-25","file":"trinity/phase1b_energy_implicit/run_energy_implicit_phase.py","line":"L150","class":"numerical","severity":"S3","claim":"ADAPTIVE_MONITOR_KEYS must be positive-definite quantities (Pb, T, L_cool, R1, R2, Eb); beta, delta and v2 are not admissible for a log10-ratio measure.","evidence":"beta and delta cross zero at the end of the energy phase (SPEC-014 Reading B) and v2 crosses zero at stall; log10 of a sign change is undefined.","expected":"Only strictly positive monitored quantities.","failure_scenario":"The dex measure goes undefined precisely at the phase transition — the moment the controller must resolve — and (per S5b-C-24) fails to shrink.","repro":"Inspect the key list against the sign of each quantity over a full run.","confidence":"medium"},
  {"id":"S5b-C-26","file":"trinity/phase1b_energy_implicit/run_energy_implicit_phase.py","line":"L412","class":"silent-failure","severity":"S2","claim":"get_monitor_values must return copies of scalar floats, not references/views into the mutable params container, because before/after snapshots are taken from the same object.","evidence":"The API shape (get_monitor_values(params) called twice around a segment) invites aliasing; if the returned dict aliases live state, before == after always.","expected":"Plain float copies; dict not backed by params.","failure_scenario":"compute_max_dex_change returns 0.0 for every segment, the adaptive controller never shrinks, and every run is silently integrated at DT_SEGMENT_MAX with an uncontrolled operator-splitting error. The run completes and looks healthy.","repro":"before = get_monitor_values(params); mutate params.Pb; assert before['Pb'] is unchanged. And on a real run assert max_dex_change > 0 for at least one segment.","confidence":"high"},
  {"id":"S5b-C-27","file":"trinity/phase1b_energy_implicit/run_energy_implicit_phase.py","line":"L332","class":"state","severity":"S3","claim":"update_unconverged_streak must return 0 when converged is True and streak+1 otherwise; never negative, never sticky.","evidence":"The streak gates dt shrinking and (via NO_ROOT_HANDOFF_STREAK) a phase handoff.","expected":"0 on success, monotone +1 on failure.","failure_scenario":"A streak that fails to reset accumulates across a healthy run and eventually triggers a spurious handoff that is indistinguishable in the output from a genuine catastrophic-cooling transition.","repro":"Unit test the two branches; assert update_unconverged_streak(7, True, t, r) == 0.","confidence":"high"},
  {"id":"S5b-C-28","file":"trinity/phase1b_energy_implicit/run_energy_implicit_phase.py","line":"L360","class":"numerical","severity":"S4","claim":"betadelta_phase_summary must guard solve_count == 0 and satisfy converged_count + no_root_count <= solve_count.","evidence":"A phase that exits on its first segment yields solve_count 0; ratios are then 0/0.","expected":"Defined output (zeros or None) at solve_count == 0.","failure_scenario":"ZeroDivisionError aborts an otherwise successful run at the reporting step, or NaN ratios pollute the metadata a future session relies on.","repro":"betadelta_phase_summary(0,0,0) must not raise.","confidence":"high"},
  {"id":"S5b-C-29","file":"trinity/phase1b_energy_implicit/run_energy_implicit_phase.py","line":"L376","class":"numerical","severity":"S2","claim":"next_dt_segment must shrink on max_dex_change > ADAPTIVE_THRESHOLD_DEX and on a long unconverged streak, grow only with hysteresis (a lower threshold than the shrink threshold), clamp to [DT_SEGMENT_MIN, DT_SEGMENT_MAX], and never return <=0 or NaN.","evidence":"Standard step-size control; grow and shrink on the same threshold produces limit-cycle oscillation of the segment length.","expected":"Monotone in max_dex_change, bounded, positive.","failure_scenario":"Oscillating segment length alternates between over- and under-resolved operator splits, injecting a saw-tooth error into Eb(t) that no tolerance controls.","repro":"Unit test: monotonicity in max_dex_change, clamping at both ends, and that repeated calls at a fixed max_dex_change equal to the threshold do not oscillate.","confidence":"high"},
  {"id":"S5b-C-30","file":"trinity/phase1b_energy_implicit/run_energy_implicit_phase.py","line":"L114","class":"regime","severity":"S2","claim":"The segment length must be capped relative to the local timescales (dt << R2/v2, dt << t, dt << t_cool), not only by an absolute DT_SEGMENT_MAX in Myr.","evidence":"alpha, beta, delta are logarithmic derivatives with respect to t, so a segment comparable to t is an O(1) error in the frozen-rate extrapolation; t_dyn ranges over decades across the shipped grid (n_core 1e2..1e5 cm^-3).","expected":"dt_segment = min(absolute cap, eps*R2/v2, eps*t).","failure_scenario":"Dense-cloud runs (short t_dyn) are integrated with a segment longer than the dynamical time, so the frozen Edot/Tdot are stale by O(1) and the phase-transition time is wrong in exactly the regime the paper grid emphasises.","repro":"Record dt_segment/(R2/v2) and dt_segment/t over a hidens edge run; both should stay well below ~0.1.","confidence":"medium"},
  {"id":"S5b-C-31","file":"trinity/phase1b_energy_implicit/run_energy_implicit_phase.py","line":"L631","class":"numerical","severity":"S3","claim":"A segment whose max_dex_change exceeded ADAPTIVE_THRESHOLD_DEX should be rejected and retried at the smaller dt, not accepted with the shrink applied only to the next segment.","evidence":"Accept-and-shrink lags the controller by one segment, so the first violating segment is always accepted with unbounded splitting error; near the transition L_cool grows superexponentially, so the first violating segment is precisely the decisive one.","expected":"Reject-and-retry, or a documented bound on the accepted error.","failure_scenario":"The transition time is set by the one segment the controller failed to resolve.","repro":"Halve ADAPTIVE_THRESHOLD_DEX and re-run; the transition time should move by less than the quoted precision.","confidence":"medium"},
  {"id":"S5b-C-32","file":"trinity/phase1b_energy_implicit/run_energy_implicit_phase.py","line":"L122","class":"silent-failure","severity":"S2","claim":"A (beta,delta) root-find failure must be distinguished from a physical transition before NO_ROOT_HANDOFF_STREAK converts it into a phase handoff; the discriminator is that L_cool/L_gain must be near 1 at a genuine catastrophic-cooling exit.","evidence":"SPEC-013: the physical criterion is L_loss/L_gain -> 1. Loss of the root is the signature of catastrophic cooling but equally of a bad bracket or a bad initial guess.","expected":"Handoff only when the cooling balance corroborates it; otherwise raise/record a numerical failure.","failure_scenario":"A solver failure is silently reported as a physical energy->momentum transition at the wrong time, and the run's headline output is wrong while every diagnostic says 'converged'.","repro":"At every no-root handoff in a run log, check L_cool/L_gain; anything far from 1 is numerics.","confidence":"high"},
  {"id":"S5b-C-33","file":"trinity/phase1b_energy_implicit/run_energy_implicit_phase.py","line":"L115","class":"silent-failure","severity":"S2","claim":"Exhausting MAX_SEGMENTS (or DT_SEGMENT_MIN with continued non-convergence) must be recorded as a numerical termination, never returned as a normal phase completion.","evidence":"SPEC-100 lists numerical cutoffs as a distinct fate; SPEC-105 requires termination bookkeeping with exit_code/outcome/detail.","expected":"A distinct outcome string in the termination block.","failure_scenario":"A truncated run is post-processed as a physical 'stall' or 'transition', contaminating the published phase-timeline statistics.","repro":"Force MAX_SEGMENTS to a small value and check the termination block distinguishes it.","confidence":"high"},
  {"id":"S5b-C-34","file":"trinity/phase1b_energy_implicit/run_energy_implicit_phase.py","line":"L631","class":"numerical","severity":"S2","claim":"Phase-exit events must be terminal with explicit direction, and the runner must re-test every exit condition at the segment START, because SciPy detects events only by a sign change within a step.","evidence":"Edot_b=0, v2=0 and R2=rCloud are all physically correct but numerically degenerate: a segment that terminated exactly on the root begins the next segment at the root, where no sign change occurs.","expected":"direction=+1 for R2-k*rCloud and R2-stop_r; direction=-1 for Eb-floor and v2-threshold; plus a start-of-segment guard.","failure_scenario":"The runner spins on a zero-length segment (hang) or steps straight past a satisfied termination condition and integrates into an unphysical regime (Eb<0, v2<0) that the event was there to prevent.","repro":"Construct a segment whose initial state exactly satisfies an event and assert the loop exits rather than integrating.","confidence":"high"},
  {"id":"S5b-C-35","file":"trinity/phase1b_energy_implicit/run_energy_implicit_phase.py","line":"L631","class":"numerical","severity":"S2","claim":"cooling_balance and ebpeak depend on L_cool, which is only known at segment boundaries, so the transition time is resolved only to +/- dt_segment unless it is refined (bisection or a re-solve within the last segment).","evidence":"The bubble-structure solve is per-segment by construction (that is what Ed_from_beta/Td_from_delta encode).","expected":"Either a refinement step or a quoted uncertainty on the transition time of order dt_segment.","failure_scenario":"The code's headline prediction (energy->momentum transition time) carries an unquoted error equal to the segment length, which is also the quantity the adaptive controller is free to grow.","repro":"Halve DT_SEGMENT_MAX and compare the reported transition time on the same config.","confidence":"medium"},
  {"id":"S5b-C-36","file":"trinity/phase1b_energy_implicit/run_energy_implicit_phase.py","line":"L171","class":"numerical","severity":"S2","claim":"ODE_ATOL must be per-component: the state spans ~9 decades (Eb ~ 1e10, R2 ~ 1e1, v2 ~ 1e1 in AU units).","evidence":"L_w ~ 1e40 erg/s = 1.7e10 Msun pc^2 Myr^-3 (SPEC-091), so Eb ~ 1e10 while R2 and v2 are O(10).","expected":"An array atol matched to each component's scale, or a non-dimensionalised state.","failure_scenario":"A scalar atol sized for Eb destroys the accuracy of R2 and v2 (and of the v2=0 stall detection) with no error message; a scalar sized for v2 merely makes atol inert for Eb.","repro":"Tighten rtol by 100x; if the trajectory moves more than the original tolerance implies, atol was binding on the wrong component.","confidence":"medium"},
  {"id":"S5b-C-37","file":"trinity/phase1b_energy_implicit/run_energy_implicit_phase.py","line":"L172","class":"silent-failure","severity":"S4","claim":"ODE_MIN_STEP is honoured only by SciPy's LSODA; passed to RK45/BDF/Radau it is ignored (at most a warning).","evidence":"SciPy solve_ivp option surface; min_step is an LSODA-only extra.","expected":"Either ODE_METHOD == 'LSODA', or the min-step guard implemented explicitly.","failure_scenario":"The intended floor on the step size is silently absent, so the solver can grind to arbitrarily small steps and hang instead of failing.","repro":"Check ODE_METHOD against the option set actually accepted by that solver.","confidence":"medium"},
  {"id":"S5b-C-38","file":"trinity/phase1b_energy_implicit/run_energy_implicit_phase.py","line":"L176","class":"numerical","severity":"S3","claim":"ODE_METHOD must cope with the non-differentiable RHS created by P_drive = max(P_b, P_HII); note the operator split has already removed the stiff cooling term, so stiffness alone does not force an implicit method.","evidence":"SPEC-023 (the max() kink and LSODA chatter); with Edot and Tdot frozen, Eb and T are linear in t within a segment so the residual system (R2, v2) is only mildly stiff.","expected":"Either a smoothed/branch-aware P_drive, or a method whose error control tolerates the kink, with the choice documented.","failure_scenario":"Solver chatter at the pressure crossover inflates step counts or silently degrades accuracy at exactly the moment TRINITY's headline P_HII term takes over.","repro":"Record which branch of max() is active per snapshot and look for step-size collapse at the crossover.","confidence":"medium"},
  {"id":"S5b-C-39","file":"trinity/phase1b_energy_implicit/run_energy_implicit_phase.py","line":"L586","class":"state","severity":"S2","claim":"get_ODE_implicit_pure and compute_forces_pure must be pure: no module-level global reads or writes, no mutation of snapshot/shell_props/params, deterministic given their arguments.","evidence":"The _pure suffix is an explicit contract; CLAUDE.md rule 5 records that trinity leaks module-level global state in-process, which is why full-run equivalence must be tested in separate processes.","expected":"Identical output for identical inputs regardless of call order or prior calls.","failure_scenario":"Hidden state makes the RHS depend on call history, so the adaptive solver's internal trial steps perturb the physics — an error that is invisible in a single run and destroys reproducibility across runs.","repro":"Call each twice with shuffled ordering and other calls interleaved; assert bitwise-identical outputs and unchanged input objects.","confidence":"high"},
  {"id":"S5b-C-40","file":"trinity/phase1b_energy_implicit/run_energy_implicit_phase.py","line":"L460","class":"divergence","severity":"S3","claim":"When P_HII > P_b the shell receives 4piR2^2 P_HII v2 of work while the bubble loses only 4piR2^2 P_b v2; this non-conservation is inherent to the max() prescription and must be documented (or the excess accounted).","evidence":"SPEC-035 audit trap (ii); SPEC-023 Reading A.","expected":"An explicit note/diagnostic of the work imbalance, or an accounting term.","failure_scenario":"Silent creation of mechanical energy whenever the photoionised branch wins — i.e. exactly in the regime that is TRINITY's claimed advance over WARPFIELD.","repro":"Integrate 4piR2^2 (P_drive - P_b) v2 dt over the energy phase and compare with the total injected energy.","confidence":"high"},
  {"id":"S5b-C-41","file":"trinity/phase1b_energy_implicit/run_energy_implicit_phase.py","line":"L631","class":"state","severity":"S3","claim":"The ODE-carried bubble temperature T must agree, at each segment end, with the T the bubble-structure solve returns at the same state; a systematic drift means the conduction closure and the integrated T have decoupled.","evidence":"T is advanced by dT/dt = delta*T/t as a bookkeeping variable, while the structure solve independently determines T at xi = bubble_xi_Tb (SPEC-040/042).","expected":"|T_ode - T_solve|/T_solve small and non-drifting through the phase.","failure_scenario":"The cooling rate is evaluated at a temperature the structure does not support; L_cool is then wrong by the Lambda(T) sensitivity, which feeds the transition trigger directly (SPEC-082).","repro":"Log both T values per segment and plot their ratio over a full energy phase.","confidence":"medium"},
  {"id":"S5b-C-42","file":"trinity/phase1b_energy_implicit/run_energy_implicit_phase.py","line":"L143","class":"sign","severity":"S2","claim":"If v2 < 0 is reachable in this phase, the swept mass must not decrease (Mdot_sh = 4piR2^2 rho v2 clamped at >= 0) and the sweep drag must use v2*|v2| so it opposes the motion.","evidence":"A shell falling back moves through the evacuated bubble interior, not through fresh ambient gas; and -4piR2^2 rho v2^2 is negative for v2<0, so with v2^2 the 'drag' accelerates the collapse.","expected":"Clamped Mdot_sh; drag proportional to v2*|v2|.","failure_scenario":"Runaway (unphysically fast) re-collapse, and a shell that un-sweeps mass as it contracts — both of which change the re-collapse fate that the paper classifies.","repro":"Force a collapsing configuration and check that M_sh is non-decreasing and that the drag term changes sign with v2.","confidence":"high"},
  {"id":"S5b-C-43","file":"trinity/phase1b_energy_implicit/run_energy_implicit_phase.py","line":"L144","class":"regime","severity":"S4","claim":"VELOCITY_THRESHOLD_EXTREME should be a physical sanity ceiling below the wind terminal speed v_w = 2 L_w/pdot_w (~1000-3000 km/s ~ 1000-3000 pc/Myr), and VELOCITY_THRESHOLD_COLLAPSE should be at or just above zero.","evidence":"SPEC-071 for v_w; the shell cannot outrun the wind that drives it.","expected":"0 <= VELOCITY_THRESHOLD_COLLAPSE << VELOCITY_THRESHOLD_EXTREME <= v_w.","failure_scenario":"A collapse threshold set at a comfortably positive velocity declares a slowly-but-genuinely-expanding shell 'collapsed', truncating the energy phase early; an extreme threshold above v_w never catches a numerical blow-up.","repro":"Compare both constants against v_w computed from the bundled SB99 table.","confidence":"medium"},
  {"id":"S5b-C-44","file":"trinity/phase1b_energy_implicit/run_energy_implicit_phase.py","line":"L111","class":"regime","severity":"S3","claim":"COOLING_UPDATE_INTERVAL must be short compared with the SPS variation timescale and must force a refresh across the age-indexed non-CIE cooling-file boundaries and the first SN.","evidence":"SPEC-083: per-age cooling files make Lambda_net piecewise-constant in cluster age; L_mech jumps by ~an order of magnitude at the first SN (~3-4 Myr).","expected":"An interval in time units, small compared with the age-file spacing, with forced refresh at known discontinuities.","failure_scenario":"L_cool is stale across the SN onset, so the bubble's energy balance (and hence the transition trigger, SPEC-013) is evaluated with the pre-SN cooling structure at post-SN power.","repro":"Plot L_cool vs t and look for a plateau that persists past a known age-file boundary.","confidence":"low"},
  {"id":"S5b-C-45","file":"trinity/phase1b_energy_implicit/run_energy_implicit_phase.py","line":"L569","class":"state","severity":"S3","claim":"ImplicitPhaseResults must carry the state at the EVENT time (the terminal event root), not the raw segment end, plus an explicit termination reason distinguishing physical exits from numerical ones.","evidence":"SPEC-016 (continuity across the phase boundary), SPEC-105 (termination bookkeeping).","expected":"Final state = solve_ivp's event state; a reason field with disjoint physical/numerical categories.","failure_scenario":"The next phase starts from a state that already violates the exit criterion (e.g. Eb below the floor, R2 beyond rCloud), producing a discontinuity in dv2/dt at the handover that SPEC-016's test would flag.","repro":"Sample dv2/dt on both sides of the energy->transition boundary; a jump above integrator tolerance is a finding.","confidence":"medium"}
]
```
