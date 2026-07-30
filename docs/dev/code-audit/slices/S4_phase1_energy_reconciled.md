# S4 phase1 energy — reconciled

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

**Status (2026-07-30):** 📗 reconciled slice report — merged from the three blind lens reports for
S4. The reconciler did **not** read `trinity/` source; every statement below is traceable to
`S4_phase1_energy_lensA.md` (code), `_lensB.md` (prose), `_lensC.md` (physics spec). Where a line
number is quoted it is the lens's line number, itself unverified.

**Slice:** `trinity/phase1_energy/energy_phase_ODEs.py`, `trinity/phase1_energy/run_energy_phase.py`.

---

## 0. Headline

The governing system is `y = (R2, v2, Eb)` with

```
Ṙ₂ = v₂
v̇₂ = [ 4πR₂²(P_drive − P_ext) − Ṁ_sh v₂ − G M_sh(M_cl + ½M_sh)/R₂² + F_rad ] / M_sh
Ė_b = L_mech − L_cool − 4πR₂² P_b v₂ − L_leak
```

Against Lens C's derived requirement the momentum equation is **term-complete and sign-correct**:
no dropped force, no double-counted force, no flipped sign. The three failure patterns the brief
asked for come out as: **(a) one dropped term, and it is in the energy equation, not the momentum
equation** (the inner-boundary `+4πR₁²Ṙ₁P_b` work, R-04); **(b) no double counting** — the two
classic traps (radiation direct-inside-IR, and swept-mass flux vs ram pressure) are both *clean*;
**(c) no sign flip** — every momentum term carries the conventional sign, with one conditional
exception when `v₂ < 0` (R-09).

The one S1 is not a missing term but a **broken work balance**: the shell is accelerated by
`P_drive = max(P_b, P_HII)` while the bubble is debited only at `P_b`, so whenever the `P_HII`
branch wins, energy enters the system at `4πR₂²(P_HII − P_b)v₂` from no reservoir. Lens A found the
code does it; Lens C independently derived the invariant it violates; Lens B confirms the `max` rule
is documented verbatim twice with **no citation anywhere in the slice**. Three lenses, one finding.

---

## 1. Coverage table

| Subject the audit required | Lens A (code) | Lens B (prose) | Lens C (spec) | Reconciled state |
|---|---|---|---|---|
| State vector & `dR2/dt = v2` | full, `:264` | ordering stated once, no units | S1 requirement | **verified clean** — closed |
| Momentum RHS, term by term with signs | full, explicit | headers only, one sign stated | full derivation | reconcilable; see §3 |
| Energy RHS, term by term | full | partial (no `L_cool`) | full derivation | one dropped term (R-04) |
| Pressure closure `P_b(E_b, R₂, R₁, γ)` | call site + args only (body out of slice) | "in energy phase, returns `bubble_E2P`" | `3(γ−1)E_b/[4π(R₂³−R₁³)]` | plumbing present, use unverified (R-21) |
| Dimensional consistency of every sum | full analysis, balanced | none (no units for state) | independent AU table, agrees | **balanced**; 2 silent-conversion risks (R-16, R-25) |
| Phase-exit criteria | 6 exit paths enumerated | 6 exit paths documented | 7 physically-correct exits + detection requirements | §4 — largest divergence surface |
| Event terminality / direction | out of slice; driver breaks on **any** trigger | undocumented | must be terminal + explicit direction + root-found | R-13 |
| Frozen-vs-live coefficients | full list | flags `cs`/`Cf` only | derives the staleness bound | R-10 |
| Numerical constants / tolerances | full literal table | 4 constants, in Myr | derives what they must satisfy | R-14, R-15, R-33 |
| Literature grounding (Weaver+77, Rahner) | n/a (comments blanked) | **zero equation numbers in the whole slice** | refuses to assert equation numbers (no lit. access) | R-22, R-23 — unresolvable in-audit |

---

## 2. Divergence table

| # | Divergence | Type | Who is right | Item |
|---|---|---|---|---|
| 1 | Shell accelerated at `P_drive`, bubble debited at `P_b` | A≠C (physics defect), B documents the rule uncited | C | R-01 |
| 2 | `vd = −1e8` overrides the whole force budget; B calls it only "early phase approximation" | A≠B (prose says nothing), A≠C (spec forbids) | C | R-02 |
| 3 | `include_PHII` documented as "Gate all HII pressure"; A finds the field is never read | A≠B — code is the defect | B | R-03 |
| 4 | Closure receives `R1`; PdV term drops the `R₁²Ṙ₁` piece | A≠C | C | R-04 |
| 5 | Post-event reconciliation uses pre-event locals | A≠C (handover invariant) | C | R-05 |
| 6 | Broad `except` re-labels bugs as "bubble collapsed" — B says this is **intentional** | A=B, both ≠ C | C | R-06 |
| 7 | `Eb≤0` stops the run in 1a, routes to momentum in 1b | A=B (admitted debt), ≠ C | C | R-08 |
| 8 | Docstrings: RHS "never reads params"; it reads params 3× per call | A≠B — comment is stale | A | R-18 |
| 9 | `ODEResult` "computed during last ODE evaluation" vs `compute_derived_quantities` "ONCE after integration"; A's line numbers put it **before** integration | B≠B, both stale vs A | A | R-19 |
| 10 | `COOLING_UPDATE_INTERVAL` 50 kyr vs phase cap 3 kyr | A=B (constants confirmed), ≠ C's schedule requirement | C | R-15 |
| 11 | `max(P_b,P_HII)` combination rule | A=B=C-acknowledged, **uncited** | open | R-01, R-20 |
| 12 | External photoionized counter-pressure (inward, ambient-density based) | A=B; **C's budget has no such term** | open / scope | R-27 |
| 13 | C's Strömgren `P_HII ∝ Q_i^{1/2}r^{−3/2}` test aimed at `get_press_ion` | B≠C (mis-aimed): A shows `get_press_ion` is the *external* term using the ambient profile | A | R-27 (reframed) |
| 14 | C read `DT_EXIT_THRESHOLD` as a min-step/stiffness detector | B≠C: A and B agree it is a *time-to-`tfinal`* proximity test | A+B | **dropped** |
| 15 | C-29 assumed a possible 4th state component `T0` | premise false — state is `(R2,v2,Eb)` | A | **dropped** |
| 16 | Two pressure functions (`get_effective_bubble_pressure` vs `bubble_E2P`) | A's concern; B `:361` says the former *returns* the latter in the energy phase | B explains A | demoted into R-11 |

---

## 3. Term-by-term force / energy budget

Sign convention: as written in `M_sh·v̇₂ = ΣF` — `+` outward, `−` inward.

### 3.1 Momentum equation

| Physical term | Present in code (A), with sign | Documented (B) | Required (C) | Verdict |
|---|---|---|---|---|
| Interior thermal pressure | **`+4πR₂²·P_drive`**, `P_drive = max(P_b, P_HII)` (`:251-265`) | named `:225`; `max(Pb,P_HII)` verbatim `:257`,`:393`; no sign, no citation | `+4πR₂²P_b` | present, sign ✓. **Combination rule is the S1** — R-01 |
| Photoionized interior pressure `P_HII` | inside the same `max` — replaces `P_b` when larger, **never added** (`:255/:258`) | `:246` "from Strömgren balance, pre-computed in runner"; gate field `:74` | `+4πR₂²P_HII` (C notes TRINITY's `max`, SPEC-022) | present; documented gate is **dead** (R-03); ledger break (R-01) |
| Radiation, direct | `+f_abs,tot·(L_bol/c)·1` — the `1` in `(1 + τ_ratio·κ_IR)` (`:135`) | `:129` "direct + IR-trapped", computed in snapshot | `+(L_bol/c)f_abs` | **present exactly once — no double count** |
| Radiation, IR-trapped | `+f_abs,tot·(L_bol/c)·τ_ratio·κ_IR` (`:135`) | same site | `+(L_bol/c)f_abs·τ_IR`, form (a) | **present once, in C's preferred form (a)**; κ_IR unit conversion unverified (R-25); frozen per segment (R-10) |
| Gravity — central cluster | `−G M_sh M_cl/R₂²` (`:220`) | `:219` "Gravity force (self + cluster)" | `−G M_cl M_sh/R₂²` | ✓ agree |
| Gravity — shell self-gravity | `−G M_sh²/(2R₂²)` — the literal `0.5` (`:220`) | same header, no factor | `−G M_sh²/(2R₂²)`, ½ derived twice | ✓ **the classic factor-2 is correct** — C-04 closed |
| Ambient / ISM thermal pressure | `−4πR₂²·P_ISM·k_B`, gated `rShell ≥ rCloud` on the **frozen** `rShell` (`:243`) | `:242` "add ISM pressure if shell beyond cloud" | `−4πR₂²P_ISM` (unconditional in C's list) | sign ✓; gating physically defensible (inside the cloud the ambient *is* the cloud); unit conversion open (R-16); frozen gate (R-10) |
| External photoionized counter-pressure | `−4πR₂²(μ_c/μ_ion)·n(r_sh)·k_B·T_ion`, only if `f_absIon < 1` (`:237-241`) | `:233` "**Inward** pressure from photoionized gas outside shell" — the *only* sign in the entire slice's prose | **no counterpart in C's budget** | A=B agree incl. sign; C silent → R-27 (open, not a defect) |
| Ambient turbulent / magnetic pressure | **absent** (A states explicitly) | silent | optional — "only if the model carries a turbulent ambient" | absent by consensus — **not a defect** |
| Swept-mass momentum flux / ram | `−Ṁ_sh·v₂`, **once** (`:265`), `Ṁ_sh` from `get_mass_profile(..., rdot=v2)` | silent | exactly once, `−Ṁ_sh v₂ ≡ −4πR₂²ρ_amb v₂²` | ✓ **no double count** — C-02 closed. Sign correct for `v₂>0`; `v₂<0` is R-09; local-vs-mean ρ is out of slice |
| Bare wind/SN momentum `ṗ_w+ṗ_SN` | **absent** from `dv2/dt`; stored into `params['F_ram_wind'/'F_ram_SN']` and never used | `:398` "P_ram: only relevant in transition; 0 in energy/implicit" | **must be absent** (already inside `P_b` via `4πR₁²P_b = ṗ_w`) | ✓ A=B=C agree — C-08 closed. Diagnostic mislabel remains (R-17) |
| Wind ram `pRam(R₂,·)` | present **only** in the `'transition'` branch, and there added to `P_HII` *inside* the `max` | `:398` consistent | must be absent in energy phase | ✓ consistent; branch may be unreachable in 1a |

**Momentum verdict: no dropped force, no doubled force, no flipped sign.**

### 3.2 Energy equation

| Physical term | Present in code (A), with sign | Documented (B) | Required (C) | Verdict |
|---|---|---|---|---|
| Mechanical input | `+L_mech(t)`, re-evaluated live from SPS (`:280`) | not stated in this module's prose | `+η_w L_w + η_SN L_SN` | present, sign ✓; η placement out of slice (C-27, referred) |
| PdV work on the shell | `−4πR₂²·P_b·v₂` — uses **`P_b`, not `P_drive`**; **no `R₁²Ṙ₁` term** (`:280`) | `:274` "P dV term, using instantaneous Pb, R2" | `−P_b·4π(R₂²v₂ − R₁²Ṙ₁)`, with the **same `V_b`** as the closure | sign ✓; **inner-boundary work dropped** (R-04); `P_b` vs `P_drive` (R-01) |
| Radiative cooling | `−L_cool` = `snapshot.bubble_LTotal`, **frozen** for the segment (`:273`) | **absent from this module's prose entirely** | `−L_cool` | present, sign ✓; undocumented (R-28); frozen while `P_b` tracks live `E_b` (R-10) |
| Covering-fraction leak | `−L_leak(f_cov, R₂, P_b, c_s, γ)`, computed live (`:277-279`) | `:274` "(Eq. leak)", geometry-set, `Cf=1 → 0 exactly`, `cs`/`Cf` frozen | `−(1−C_f)4πR₂²c_s[γ/(γ−1)]P_b` | present, sign ✓, **argument list matches C exactly**; enthalpy (5/2) vs internal (3/2) prefactor unverifiable (R-23) |
| Thermal conduction / evaporation | no separate term — can only be inside `bubble_LTotal` | silent | inside `L_cool` ("interior + conduction front") | consistent — not a defect |
| Inner-boundary work at `R1` | **absent** | silent | `+4πR₁²Ṙ₁P_b` (a *gain*), size `(R₁/R₂)²(Ṙ₁/v₂)` | **dropped term → R-04** |
| Sink matching the `P_HII` branch | **absent** (A states explicitly) | silent | required if `P_drive ≠ P_b` | **→ R-01 (S1)** |

### 3.3 Dimensional check

Lens A's per-term analysis and Lens C's independently recomputed AU table agree on **every**
dimension. Momentum terms all reduce to `M⊙ pc Myr⁻²`; energy terms to `M⊙ pc² Myr⁻³`. No summed
expression is unbalanced. Two conversions are *invisible from inside the slice* and both fail
silently rather than loudly:

- `P_ISM·k_B` (`:244`) is a pressure only if `params['PISM']` already carries `K·pc⁻³`. If the
  `cm⁻³ → pc⁻³` factor (2.938e55) is missing upstream, the term is ~1e55 too small and simply
  vanishes. A's supporting evidence: `params['nISM']` exists on the snapshot and is unused, which is
  consistent with `PISM` being the `n·T` form. → R-16.
- `κ_IR` must be 8.3556e-4 pc² M⊙⁻¹ (from 4 cm² g⁻¹); a missed conversion is a quiet ~1.2e-4 factor
  that switches IR trapping off. → R-25.

`vd = −1e8` is dimensionally an acceleration (pc Myr⁻²), so it raises no imbalance — only a
magnitude of ≈ −2900 km s⁻¹ accumulated over one 3e-5 Myr segment. → R-02.

---

## 4. Pressure closure — reconciled

| Lens | Statement |
|---|---|
| A | `R1 = solve_R1(R2, Eb, L_mech, v_mech)` then `P_b = get_effective_bubble_pressure(phase, Eb, R2, R1, γ, L_mech, v_mech, t, t_SF)`, recomputed on **every** RHS call from the integrated `Eb`. Bodies out of slice. Driver's `params['Pb']` comes from a *different* source (`bubble_data.Pb`, and `bubble_E2P` at entry/exit); `ODEResult.Pb`/`R1` are computed and then discarded. |
| B | `:361` "In momentum phase, this returns `pRam`; in energy phase, returns `bubble_E2P`." No formula, no γ, no `R1` relation, no citation anywhere. |
| C | `P_b = 3(γ−1)E_b/[4π(R₂³−R₁³)]`; the literal `E_b/(2πR₂³)` bakes in γ=5/3 **and** `R1≪R2` simultaneously; the **same `V_b`** must appear in the PdV term; `R1 = √(ṗ_w/4πP_b)` (or the 0.866 strong-shock variant), enforced `0 < R1 < R2`. |

**Reconciled:**

1. **The closure is consistent with the integrated variable.** Both the pressure that drives the
   shell and the pressure that does the PdV work are functions of the *same* `Eb` being integrated,
   recomputed each call. The S1 the brief asked about — a closure inconsistent with the energy
   variable — **is not present**. State this positively; it is the most important clean result in
   the slice.
2. **γ and `R1` are plumbed** (both are arguments). Whether the body uses them is out of slice, so
   C-05's "hard-coded 2π silently ignores `gamma_adia`" is an **open check**, not a defect (R-21).
3. **The volumes are asymmetric.** The closure takes `R1`; the PdV term is `4πR₂²P_b v₂` with no
   `R₁²Ṙ₁`. If the closure's `V_b` is `(4π/3)(R₂³−R₁³)`, the two are inconsistent exactly as C-06
   warns — a manufactured source/sink growing through the phase as `P_b` falls and `R1` grows.
   This is the reconciled S2 (R-04).
4. **B explains away A's "two pressure functions" alarm** for the ODE path: if
   `get_effective_bubble_pressure` returns `bubble_E2P` in the energy phase they are the same
   function. The residual is observability — the driver's exported `params['Pb']` comes from the
   bubble-structure solver, and `ODEResult.Pb`/`R1` are discarded, so any disagreement is
   structurally invisible. Demoted into R-11.

---

## 5. Phase-exit criteria — reconciled (weighted heavily)

| Exit | Code does (A) | Prose says (B) | Physics requires (C) | Verdict |
|---|---|---|---|---|
| Absolute time cap | `while (TFINAL_ENERGY_PHASE − t_now) > DT_EXIT_THRESHOLD`, `TFINAL = 3e-3` Myr compared against **absolute simulation time** | "max duration (~3000 years)" | a run limit, must be a *distinct recorded reason*, must exceed reachable transition times | A≈B on the constant; **A≠B on semantics** (absolute, not elapsed) → zero-segment hole (R-14) |
| Cloud edge `R2 ≥ rCloud` | loop condition, `rCloud` captured once before the loop | not documented as an exit | terminal event `g = R2 − r_cloud`, direction `+1` | boundary test, not a root-found event (R-13); stale `rCloud` (R-36) |
| Cooling balance | `break` when `L_gain>0 ∧ (L_gain−L_loss)/L_gain < thr`, evaluated **pre-ODE at the segment boundary**; `thr = param or 0.05 if falsy` | step 6b, parity with 1b, "byte-identical (G0)" | event `g = ratio − ε`, direction `−1`, root-found on dense output; ε is a regularisation needing a sensitivity table | resolution quantised to `SEGMENT_DURATION` (3e-5 Myr of a 3e-3 Myr phase, ≤1%) + falsy-0.0 bug → R-12 |
| `Eb ≤ 0` / non-finite | post-segment guard → `ENERGY_COLLAPSED`, `break`; **no clipping of `Eb` anywhere in the RHS** | documented at length; 1a stops, 1b routes to momentum; "deferred" | must terminate with a distinct reason; must never be clamped | **C-18 satisfied** (no clamp, distinct code). Residual is the 1a/1b asymmetry → R-08 |
| Bubble-solve failure | broad `except (ValueError, RuntimeError, BubbleSolverError)` → "Energy-driven bubble collapsed" | **intentional**: "any such failure here means the model has broken down" | structural breakdown must be a *distinct* recorded reason, not a catch-all | A=B, both ≠ C → R-06 |
| Solver events | list built **once** before the loop; **any** trigger ends the phase regardless of `terminal`; `return` (skipping reconciliation) if simulation-ending | "safe termination"; **no terminality, no direction documented** | terminal + explicit direction + dense-output root-finding + earliest root wins & recorded | R-13, R-31. **Note: A refutes B-12's stated failure mode** — a `terminal=False` event cannot silently fail to stop the phase, because the driver breaks on any trigger. The *inverted* risk is live: a marker event would now end the phase |
| `DT_EXIT_THRESHOLD` | proximity-to-`tfinal` term in the while condition | "exit when this close to tfinal" | (C assumed a min-step/stiffness detector) | **C misread it. Dropped.** |
| Solver failure retry | RK23, shortened span, 10× tolerances, **`success` never re-checked**, no `dense_output` | not documented | — | R-07 |

**The exit disagreement that matters most** is not any single criterion but the *mechanism*: with
one exception the phase ends on **boolean tests at segment boundaries**, not on root-found terminal
events. C's §4.5 consequence — the state handed to the next phase is off the event surface by
`O(Δt·v₂)` — is therefore live, and it compounds with R-05 (the exported derived quantities are
evaluated at a *different* state again). Bounded here by the short 3e-5 Myr segment, hence S3 not
S2; but it is the structural difference between what the code does and what C requires.

---

## 6. Closed by triangulation (reported, not carried forward)

Nine input candidates are retired here. A shorter list is the point.

| Input | Why closed |
|---|---|
| C-01 (S1) `f[0]` must be `y[1]` | A: `dR2/dt = v2` at `:264`. **Satisfied.** |
| C-02 (S1) swept-mass flux double count | A: `−Ṁ_sh·v₂` appears exactly once; no separate ram term in the energy branch. **Satisfied.** |
| C-04 (S2) self-gravity factor ½ | A: `G*mShell/R2**2*(mCluster + 0.5*mShell)`. **Satisfied — the classic factor-2 is correct.** |
| C-08 (S2) bare `ṗ_w` must be absent | A: `F_ram_wind`/`F_ram_SN` are stored but never enter `dv2/dt`; B: "0 in energy/implicit". **Satisfied.** Residual = the mislabeled `F_ram` diagnostic (R-17). |
| C-09 (S2) radiation direct-inside-IR double count | A: single expression `f_abs·(L/c)·(1 + τ_ratio·κ_IR)`, C's preferred form (a), with no separately added direct term. **Satisfied.** Residual = κ_IR units (R-25). |
| C-18 (S2) `Eb≤0` must not be clipped | A: "no clamping of `Eb` or `R2` inside the RHS"; explicit `ENERGY_COLLAPSED` guard. **Satisfied.** Residual = routing (R-08). |
| C-22 (S2) `DT_EXIT_THRESHOLD` mis-reported as physics | A and B agree it is a time-to-`tfinal` proximity threshold, not a min-step detector. **Premise wrong.** |
| C-29 (S3) slaved `T0` state component | A: the state is `(R2, v2, Eb)`; there is no `T0` component. **Premise wrong.** |
| B-12 (S3) non-terminal event silently fails to stop the phase | A: the driver breaks on *any* triggered event. **Stated failure mode refuted**; reframed as R-13. |

**Referred out of slice** (real questions, wrong slice — do not re-litigate here): C-03 local-vs-mean
ρ in `Ṁ_sh` (`mass_profile`); C-27 thermalisation efficiency placement (SPS/feedback); C-28 `L_cool`
grid convergence and C-30/C-31 Weaver interior prefactors and the ξ=0.98 factor (bubble structure /
`bubble_luminosity`); C-36 `μ_H` vs per-particle `μ` (density profile + unit conversions).

---

## 7. Ranked merged findings

Ordered S1 → S4, then by corroboration strength.

```json
[
  {
    "id": "S4-R-01",
    "file": "trinity/phase1_energy/energy_phase_ODEs.py",
    "line": 280,
    "class": "other",
    "severity": "S1",
    "claim": "The shell is accelerated by P_drive = max(P_b, P_HII) while the bubble energy equation debits PdV work only at P_b, so whenever the P_HII branch wins the combined system gains energy at 4*pi*R2^2*(P_HII - P_b)*v2 from no reservoir.",
    "evidence": "A: line 265 uses P_drive in 4.0*np.pi*R2**2*(P_drive - P_ext); line 280 uses press_bubble in (4*np.pi*R2**2*press_bubble)*v2; P_drive is max(press_bubble, P_HII[+P_ram]) at :255/:258, and A states no energy sink matching the P_HII branch exists. C derived the invariant independently: multiplying the momentum equation by v2 and adding the bubble energy equation leaves the term 4*pi*R2^2*(P_drive - P_b)*v2, which vanishes iff P_drive == P_b. B confirms the rule is documented verbatim twice ('energy / implicit phases: max(Pb, P_HII)', :257 and :393) with no citation anywhere in the slice.",
    "expected": "Either both equations use the same pressure, or the excess work is charged explicitly against the ionizing-photon energy budget, or the non-conservation is documented and bounded with a reported magnitude.",
    "failure_scenario": "In HII-pressure-dominated configurations (large Qi, low bubble pressure) the shell gains momentum the bubble never pays for; R2 and v2 are inflated exactly in the regime TRINITY claims as its departure from WARPFIELD, while the run stays smooth and physical-looking.",
    "repro": "Log which max() branch is active per segment; integrate 4*pi*R2^2*(P_drive - P_b)*v2 dt over the energy phase and compare against L_mech*t. Full ledger check: accumulate int L_gain dt vs Eb + 0.5*M_sh*v2^2 + int(L_cool + L_leak + 0.5*Mdot_sh*v2^2 + F_grav*v2 + 4*pi*R2^2*P_ISM*v2 - F_rad*v2) dt from dictionary.jsonl for param/simple_cluster.param and both f1edge configs.",
    "confidence": "high",
    "lenses": ["A", "B", "C"],
    "divergence": "AC",
    "status": "corroborated",
    "source_ids": ["S4-A-04", "S4-C-07", "S4-C-39", "S4-B-14"]
  },
  {
    "id": "S4-R-02",
    "file": "trinity/phase1_energy/energy_phase_ODEs.py",
    "line": 270,
    "class": "silent-failure",
    "severity": "S2",
    "claim": "When EarlyPhaseApproximation is set, the entire momentum equation is discarded and replaced by the hard-coded constant dv2/dt = -1e8 pc/Myr^2.",
    "evidence": "A: lines 264-266 compute vd from the full force budget, then lines 269-270 unconditionally overwrite it with vd = -1e8; the snapshot flag is frozen for the segment and run_energy_phase.py:342-344 clears params['EarlyPhaseApproximation'] only AFTER the first solve_ivp call, so segment 0 always integrates under the override. Over one SEGMENT_DURATION = 3e-5 Myr this integrates to dv2 = -3000 pc/Myr (about -2900 km/s). B independently attests the term exists but documents nothing about it: 'Early phase approximation' (:268) and 'Handle early phase approximation switch' (run_energy_phase.py:341) state no content, no switch criterion, no validity range. C requires the momentum RHS to be the force budget.",
    "expected": "A physically motivated approximate expression, or a magnitude that cannot dominate the real dynamics, plus a documented criterion and validity range. A bare -1e8 replacing six force terms is not recoverable from the arithmetic.",
    "failure_scenario": "With the flag set at entry, the first segment drives v2 to a large negative value regardless of the actual forces; a velocity- or radius-based event fires immediately and the phase ends for reasons unrelated to physics. If that first segment breaks via an event, line 343 is never reached and the flag stays set for any re-entry. Escalates to S1 if the flag is true by default in any shipped .param.",
    "repro": "First: determine whether EarlyPhaseApproximation is ever true at phase entry in the shipped configs. If yes, log v2 across the first segment and compare against 4*pi*R2^2*(P_drive-P_ext)/mShell.",
    "confidence": "high",
    "lenses": ["A", "B"],
    "divergence": "AC",
    "status": "corroborated",
    "source_ids": ["S4-A-01", "S4-B-11"]
  },
  {
    "id": "S4-R-03",
    "file": "trinity/phase1_energy/energy_phase_ODEs.py",
    "line": 74,
    "class": "deadcode",
    "severity": "S2",
    "claim": "The ODESnapshot field documented as 'Gate all HII pressure' (include_PHII) is never read: the documented switch has no effect on the physics in either the RHS or the diagnostics path.",
    "evidence": "B: the field's own comment at :74 is 'Gate all HII pressure', and B flags that neither of the two duplicated P_HII banners (:246 RHS, :382 diagnostics) restates the gate. A, reading the stripped source, lists include_PHII (:74) among seven snapshot fields that are 'assigned at lines 138-164 and never referenced in get_ODE_Edot_pure or compute_derived_quantities'. The two lenses independently describe the same field from opposite sides: documented as a gate, never consulted.",
    "expected": "Either the gate is honoured identically in both paths, or the field and its comment are removed. If a .param key feeds it, that key is currently inert.",
    "failure_scenario": "A user (or a sweep) disables HII pressure and it is still applied: P_HII continues to drive the shell through max(P_b, P_HII), so the parameter study silently scans a dimension that does not exist. Because P_HII feeds the S1 above, an inert gate also removes the only apparent way to switch off the non-conserving branch.",
    "repro": "grep for 'include_PHII' across trinity/; check whether any .param key or schema default maps to it, then run with it set both ways and diff dictionary.jsonl.",
    "confidence": "medium",
    "lenses": ["A", "B"],
    "divergence": "AB",
    "status": "corroborated",
    "source_ids": ["S4-A-13", "S4-B-15"]
  },
  {
    "id": "S4-R-04",
    "file": "trinity/phase1_energy/energy_phase_ODEs.py",
    "line": 280,
    "class": "other",
    "severity": "S2",
    "claim": "The PdV term uses the outer surface only (4*pi*R2^2*P_b*v2) while the pressure closure is handed R1, so the work term and the closure do not correspond to the same control volume: the inner-boundary work +4*pi*R1^2*Rdot1*P_b is dropped.",
    "evidence": "A: R1 = solve_R1(R2, Eb, L_mech, v_mech) at :223 is passed into get_effective_bubble_pressure(phase, Eb, R2, R1, gamma, ...) at :226-231, so the closure is R1-aware; the energy equation at :280 is (4*np.pi*R2**2*press_bubble)*v2 with no R1 contribution. C derived the requirement: P_b*V_b = (gamma-1)*E_b is an identity, so dV_b/dt = 4*pi*(R2^2*v2 - R1^2*Rdot1) must be the work term; using one volume for P_b and another for the work makes dE_b/dt correspond to no thermodynamic system. C sizes the mismatch at relative 3*(R1/R2)^3 in P_b (~0.5% at R1/R2 = 0.17) and (R1/R2)^2*(Rdot1/v2) in the work, both growing as P_b falls late in the phase.",
    "expected": "One volume definition used in both places: either both include R1, or both drop it (and the omission is documented and bounded).",
    "failure_scenario": "A silent, monotonically growing energy source/sink through the energy phase that shifts the transition time -- in the same direction on every run, so it is invisible to run-to-run comparison.",
    "repro": "Confirm what get_effective_bubble_pressure/bubble_E2P use for V_b (R2^3 vs R2^3-R1^3). Then integrate dE_b/dt - [L_gain - P_b*dV_b/dt - L_cool - L_leak] over a run; a non-zero residual is the leak.",
    "confidence": "medium",
    "lenses": ["A", "C"],
    "divergence": "AC",
    "status": "corroborated",
    "source_ids": ["S4-C-06", "S4-A-lens-sec5"]
  },
  {
    "id": "S4-R-05",
    "file": "trinity/phase1_energy/run_energy_phase.py",
    "line": 393,
    "class": "state",
    "severity": "S2",
    "claim": "After an event-triggered break, the post-loop reconciliation recomputes R1, Pb and shell_mass from the stale pre-segment locals while params['R2'/'v2'/'Eb'] already hold the post-event state, so the state handed to the next phase is internally inconsistent.",
    "evidence": "A: apply_event_result at :327-328 writes the event state into params (state_keys=['R2','v2','Eb']) but does not update the local R2, v2, Eb, t_now; :331 breaks; :391-399 then call get_current_sps_feedback(t_now,...), solve_R1(R2, Eb, ...), bubble_E2P(Eb, R2, R1_f, ...) and get_mass_profile(R2, params, ...) on those stale locals and assign into params['R1'], params['Pb'], params['shell_mass']; :400 then calls shell_structure_pure(params), which reads the post-event params['R2'] with the pre-event Pb and shell_mass. C independently requires that the handover carry a mutually consistent (R2, v2, E_b) with derived quantities on the same state, and that compute_derived_quantities report the same numbers the RHS used.",
    "expected": "Refresh the locals from event_result.y / event_result.t before the reconciliation block, as the normal exit path at :346-349 does.",
    "failure_scenario": "Every event-terminated energy phase hands the next phase a Pb, R1 and shell_mass evaluated at the previous segment's radius and energy; the mismatch grows with segment length and with how early inside the segment the event fired, and appears downstream as a spurious discontinuity attributed to physics.",
    "repro": "Enable an event that fires mid-segment and compare params['Pb'] against bubble_E2P(params['Eb'], params['R2'], params['R1'], gamma) at phase exit; diff the last energy-phase snapshot against the first next-phase snapshot for R2 and v2.",
    "confidence": "high",
    "lenses": ["A", "C"],
    "divergence": "AC",
    "status": "corroborated",
    "source_ids": ["S4-A-03", "S4-C-20", "S4-C-17"]
  },
  {
    "id": "S4-R-06",
    "file": "trinity/phase1_energy/run_energy_phase.py",
    "line": 171,
    "class": "silent-failure",
    "severity": "S2",
    "claim": "Any ValueError or RuntimeError raised anywhere inside the bubble-structure solve is reinterpreted as the physical statement 'Energy-driven bubble collapsed' and the run ends with SimulationEndCode.ENERGY_COLLAPSED -- and the comments say this breadth is deliberate.",
    "evidence": "A: lines 169-183 catch (ValueError, RuntimeError, bubble_luminosity.BubbleSolverError), set the end reason to 'Energy-driven bubble collapsed: bubble solve degenerate as Eb -> 0' and break; ValueError/RuntimeError are raised by numpy, scipy and interpolators for reasons unrelated to Eb. B transcribes the comment at :163: 'Any such failure here means the energy-driven model has broken down -- stop the run cleanly rather than crash with the bare exception.' C requires structural breakdown to be a distinct recorded reason per failure mode.",
    "expected": "Catch only BubbleSolverError (or a dedicated degeneracy exception) for the collapse conclusion, with distinct end codes for table-out-of-range and bracketing failure; let genuine programming errors surface.",
    "failure_scenario": "A TypeError-class regression or a table out-of-range condition produces a physically plausible-looking 'collapsed' run that is indistinguishable in the output from a real collapse. No test fails loudly, and a sweep silently reclassifies affected cells.",
    "repro": "Inject `raise ValueError('canary')` into bubble_luminosity.get_bubbleproperties_pure; run param/simple_cluster.param; inspect SimulationEndReason.",
    "confidence": "high",
    "lenses": ["A", "B"],
    "divergence": "AC",
    "status": "corroborated",
    "source_ids": ["S4-A-07", "S4-B-04"]
  },
  {
    "id": "S4-R-07",
    "file": "trinity/phase1_energy/run_energy_phase.py",
    "line": 310,
    "class": "silent-failure",
    "severity": "S2",
    "claim": "The solve_ivp failure retry has no success check and no progress guard: if the RK23 retry also fails at the first step, t_now never advances and the segment loop can spin forever.",
    "evidence": "A: lines 310-321 retry with RK23, a span shortened to t_now + SEGMENT_DURATION/10, rtol*10, atol*10 and no dense_output, but never re-check solution.success; lines 336-337 then take solution.y[:, -1] and solution.t[-1] unconditionally. scipy always returns at least the initial point, so a first-step failure yields t_new == t_now with y unchanged, and the while condition at :138 depends only on R2 and t_now.",
    "expected": "Check solution.success after the retry, or require t_new > t_now before continuing, and terminate the phase with an explicit end code otherwise.",
    "failure_scenario": "A stiff configuration where both RK45 and RK23 fail the first step hangs the run in an infinite loop, re-solving bubble and shell structure every iteration, with no error and no output progress -- indistinguishable from a slow run.",
    "repro": "Force the retry path (make ode_func return NaN once) and observe t_now never advancing.",
    "confidence": "high",
    "lenses": ["A"],
    "divergence": "none",
    "status": "single-lens",
    "source_ids": ["S4-A-02"]
  },
  {
    "id": "S4-R-08",
    "file": "trinity/phase1_energy/run_energy_phase.py",
    "line": 359,
    "class": "regime",
    "severity": "S2",
    "claim": "An Eb <= 0 collapse detected in phase 1a stops the whole run, whereas the identical physical event detected in 1b is routed to the momentum phase -- the outcome depends on which sub-phase noticed, not on physics.",
    "evidence": "B transcribes the admission verbatim at :359: 'Phase 1b now ROUTES such a collapse to the momentum phase (run_energy_implicit_phase.classify_energy_collapse); routing it from 1a too is deferred (rare: collapse within the fixed ~3000-yr early window). Until then 1a stops cleanly here.' Same admission at :167, both deferring to docs/dev/transition/pdv-trigger/HIMASS_HANDOFF_PLAN.md, which the project's own rules classify as unverified. A confirms the mechanism: :368-379 post-segment guard, 'not isfinite(Eb) or Eb <= 0' -> ENERGY_COLLAPSED, break. C requires the same physical event to produce the same routing, and (C-18) confirms the in-phase handling is otherwise correct -- Eb is never clipped or floored.",
    "expected": "The same routing for Eb <= 0 regardless of the detecting sub-phase; if 1a genuinely cannot route, the run must be recorded with a distinct code that marks it as an unrouted collapse, not as a terminal fate.",
    "failure_scenario": "A massive/dense GMC whose bubble collapses inside the first ~3000 yr has its run terminated instead of continuing into the momentum phase; a sweep silently loses those parameter combinations, and the recorded fate distribution is biased against exactly the high-mass cells.",
    "repro": "Construct a high-mass/high-density config that collapses before TFINAL_ENERGY_PHASE and compare its fate with the same physics reaching collapse just after the 1a->1b boundary.",
    "confidence": "high",
    "lenses": ["A", "B", "C"],
    "divergence": "AC",
    "status": "corroborated",
    "source_ids": ["S4-B-13", "S4-C-18", "S4-C-20"]
  },
  {
    "id": "S4-R-09",
    "file": "trinity/phase1_energy/energy_phase_ODEs.py",
    "line": 213,
    "class": "sign",
    "severity": "S3",
    "claim": "The shell-mass monotonicity guard is inconsistent, and in the regime it exists to protect (v2 < 0) the swept-mass term accelerates the collapse instead of resisting it.",
    "evidence": "A: lines 212-217 (duplicated at 350-355) do `if prev_mShell > 0 and mShell_new < prev_mShell: mShell = prev_mShell; mShell_dot = 0.0`, where prev_mShell is the snapshot value frozen at segment start -- so inside a segment the mass may still decrease freely while staying above that floor, and in that case mShell_dot keeps its negative value. C derived the physics: with Mdot_sh = 4*pi*R2^2*rho*v2 and v2<0 the shell 'un-sweeps' gas it already carries, and the term -Mdot_sh*v2 = -4*pi*R2^2*rho*v2^2 is always inward-directed, so for a collapsing shell it is an inward acceleration; ram pressure must always oppose motion, i.e. the term should go as v2*|v2|. B documents the intent twice verbatim ('Shell mass can NEVER decrease -- once mass is swept up, it stays in shell') but the prose describes a rule the implementation only partially enforces.",
    "expected": "A consistent guard: track a running maximum (not the segment-start floor) and set Mdot_sh = 0 wherever the clamp applies; and either an enforced exit before v2 < 0, or a ram term that opposes motion in both directions.",
    "failure_scenario": "During a stall/re-collapse the inertia is held at the pre-collapse value while the momentum term that should resist collapse is deleted or points inward -- an artificially fast, artificially light recollapse, i.e. a biased dispersal-vs-recollapse verdict.",
    "repro": "Integrate a segment with v2 < 0 and log mShell_new vs prev_mShell and mShell_dot; assert M_sh is monotone non-decreasing over the phase and that the swept-mass term opposes v2 in both signs.",
    "confidence": "high",
    "lenses": ["A", "C"],
    "divergence": "AC",
    "status": "corroborated",
    "source_ids": ["S4-A-05", "S4-C-26", "S4-B-08"]
  },
  {
    "id": "S4-R-10",
    "file": "trinity/phase1_energy/energy_phase_ODEs.py",
    "line": 273,
    "class": "numerical",
    "severity": "S3",
    "claim": "Coefficients that depend on R2 or on Eb are frozen for the whole segment (L_cool, F_rad, P_HII, rShell, f_absIon, coverFraction, c_sound) while P_b, M_sh, L_leak and every R2^2 factor track the integrated state -- so the loss term does not respond to the energy it drains, and the external pressure is evaluated at a different radius from the area it acts on.",
    "evidence": "A: L_cool = snapshot.bubble_LTotal (:273), F_rad = snapshot.F_rad (:261), P_HII = snapshot.P_HII (:251) are segment constants; press_bubble (:226) and L_leak (:277) are recomputed live; P_ext is evaluated at the frozen snapshot.rShell (:236-244) and then multiplied by 4*pi*R2**2 with the integrated R2 at :265, and the cloud-boundary test rShell >= rCloud likewise cannot toggle mid-segment. B confirms the author tracks the distinction for cs/Cf ('frozen per segment', :108/:275) but never states it for F_rad, which is built inside create_ODE_snapshot (:115/:129). C derives the bound: M_sh ~ R2^(3-w), tau_IR ~ M_sh/R2^2, P_HII ~ R2^(-3/2), and L_cool varies on the current-age timescale (|dlnL_cool/dlnt| >~ 0.54), so a fixed absolute freeze gives a staleness error ~0.54*dt/t that diverges as t -> 0 -- precisely the early window this phase occupies.",
    "expected": "Either these are evaluated live from y, or the frozen-coefficient scheme is demonstrated convergent (halve SEGMENT_DURATION, full runs in separate processes at matched t), and the refresh restarts the integrator because the RHS is discontinuous there.",
    "failure_scenario": "The reported RTOL is not the accuracy of the answer; near cooling balance (the very condition tested at :276-287) the frozen L_cool cannot follow a falling Eb, so the trigger fires at a segment boundary rather than at the true crossing; and in a steep profile (rCore ~ 1 pc) P_ext corresponds to the wrong ambient density with the same-signed bias every segment.",
    "repro": "Halve SEGMENT_DURATION and compare R2(t), v2(t) and the transition time (separate processes, matched simulation time); log P_ext at segment start against get_press_ion(R2_end) at segment end for a steep-profile config.",
    "confidence": "high",
    "lenses": ["A", "B", "C"],
    "divergence": "AC",
    "status": "corroborated",
    "source_ids": ["S4-A-17", "S4-A-08", "S4-B-10", "S4-C-24", "S4-C-23"]
  },
  {
    "id": "S4-R-11",
    "file": "trinity/phase1_energy/energy_phase_ODEs.py",
    "line": 325,
    "class": "divergence",
    "severity": "S3",
    "claim": "compute_derived_quantities duplicates the entire RHS physics of get_ODE_Edot_pure and has already diverged from it, so the recorded force budget is not guaranteed to reproduce the integrated trajectory -- and the ODE's own Pb/R1 are discarded, making the discrepancy unobservable.",
    "evidence": "A: lines 332-407 repeat lines 192-279 term for term (feedback lookup, mass clamp, F_grav, solve_R1, effective pressure, P_ext, P_drive max, leak luminosity); the one physics difference is that the EarlyPhaseApproximation override (:269-270) has no counterpart in the diagnostics path. A also notes ODEResult.Pb and ODEResult.R1 (:414-415) are never copied out by the driver's field-by-field transfer at run_energy_phase.py:232-255, while params['Pb'] is set from bubble_data.Pb (:190-192). B independently flags the duplicated shell-mass invariant (:201/:211 vs :339/:349) and the duplicated P_HII banner (:246 vs :382). C requires that M_sh*dv2/dt reconstructed from the reported forces equal the integrator's dv2/dt at every snapshot, and warns that two code paths computing 'the same' force is the standard way budget closure silently breaks.",
    "expected": "One function computing the shared quantities, with derivative assembly and diagnostic packing as thin wrappers; and the ODE's own Pb/R1 exported so the two pressures can be compared.",
    "failure_scenario": "Published force-fraction figures normalise by F_tot and presuppose the listed terms are exhaustive and non-overlapping; a drifted diagnostics path makes those figures quantitatively wrong while looking correct. The EarlyPhaseApproximation branch has already drifted.",
    "repro": "diff lines 192-279 against 332-407; then for each snapshot recompute (F_drive + F_rad - F_grav - F_ram_swept - F_ion_in - 4*pi*R2^2*P_ISM)/M_sh and compare against the finite-difference dv2/dt from consecutive snapshots.",
    "confidence": "high",
    "lenses": ["A", "B", "C"],
    "divergence": "AC",
    "status": "corroborated",
    "source_ids": ["S4-A-15", "S4-A-12", "S4-B-08", "S4-C-17"]
  },
  {
    "id": "S4-R-12",
    "file": "trinity/phase1_energy/run_energy_phase.py",
    "line": 281,
    "class": "numerical",
    "severity": "S3",
    "claim": "The cooling-balance transition is a boolean break evaluated at the pre-ODE segment boundary rather than a root-found event, and its threshold has a falsy-value bug: a configured phaseSwitch_LlossLgain of exactly 0.0 is silently replaced by the hard-coded 0.05.",
    "evidence": "A: :276-287 break when L_gain > 0 and (L_gain - L_loss)/L_gain < thr, with `_thr = params['phaseSwitch_LlossLgain'].value` then `_thr = _thr if _thr else 0.05` -- 0.0 is falsy. The test runs on the snapshot at segment start, so the exit is resolved only to SEGMENT_DURATION = 3e-5 Myr. B documents the trigger as parity with 1b and claims it is 'byte-identical (G0)' for healthy bubbles, and notes the parenthetical 'ratio ~1 >> threshold' implies a threshold much smaller than 1 that the prose never states. C requires the exit be a continuous sign-changing event g = (L_gain-L_loss)/L_gain - eps with direction = -1, root-found on dense output, and says eps is a numerical regularisation of 'L_loss/L_gain -> 1' whose 0.01-0.2 sensitivity must be reported.",
    "expected": "`_thr = 0.05 if _thr is None else _thr`; the criterion registered as a terminal event with direction; and a committed sensitivity table of the transition time across eps in 0.01-0.2.",
    "failure_scenario": "A sweep scanning phaseSwitch_LlossLgain down through 0 produces an identical result at 0 and 0.05, silently truncating the parameter study; and the reported transition time is quantised to the segment length and is a function of a tuning constant that is presented as an implementation detail.",
    "repro": "Set phaseSwitch_LlossLgain = 0.0 in a .param and confirm the logged threshold is 0.05. Re-run one config at 0.01, 0.05, 0.2 and tabulate the transition time as a committed CSV under docs/dev/data/.",
    "confidence": "high",
    "lenses": ["A", "B", "C"],
    "divergence": "AC",
    "status": "corroborated",
    "source_ids": ["S4-A-06", "S4-C-21", "S4-C-19", "S4-B-16"]
  },
  {
    "id": "S4-R-13",
    "file": "trinity/phase1_energy/run_energy_phase.py",
    "line": 324,
    "class": "numerical",
    "severity": "S3",
    "claim": "Event terminality and crossing direction are undocumented and unverified, and the driver ends the phase on ANY triggered event regardless of its terminal flag -- so terminality is decided by the caller, not by the event definition.",
    "evidence": "A: the event list is built once at :118 from params before the loop and reused for every segment; at :324-331 check_event_termination(solution, ode_events) is followed by apply_event_result and then `return` if is_simulation_ending else `break` -- 'Any triggered event ends the phase, whether or not it was declared terminal'. The event functions live in trinity/phase_general/phase_events.py, outside the slice, so the zero-crossing quantity, direction and terminal flags could not be verified. B confirms nothing is documented: 'Build events for safe termination' (:114), 'Check if an event terminated the integration' (:323), with no direction, no terminal flag and no multi-event precedence stated. C requires terminal=True with explicit direction, root-finding on dense output, earliest root wins and is recorded.",
    "expected": "Per-event documented terminal flag and direction; the driver honouring the declared terminality rather than overriding it; the winning event recorded in the termination block.",
    "failure_scenario": "A: A's reading refutes the usual worry (a terminal=False event cannot silently fail to stop the phase). The live inverted risk is that an event added as a non-terminal marker or logger now ends the energy phase; and without direction, an event grazing zero (v2 in a stalled shell, or the L_loss/L_gain ratio oscillating about its threshold) fires spuriously.",
    "repro": "Enumerate the callables from build_energy_phase_events and assert each has the intended .terminal and .direction; add a test that each fires exactly once in a run where it should, and that the recorded end reason names the winning event.",
    "confidence": "high",
    "lenses": ["A", "B", "C"],
    "divergence": "ABC",
    "status": "corroborated",
    "source_ids": ["S4-B-12", "S4-C-19", "S4-A-lens-sec4"]
  },
  {
    "id": "S4-R-14",
    "file": "trinity/phase1_energy/run_energy_phase.py",
    "line": 54,
    "class": "regime",
    "severity": "S3",
    "claim": "TFINAL_ENERGY_PHASE is a hard-coded 3e-3 Myr compared against ABSOLUTE simulation time with no .param override, so the loop body never executes at all if the phase is entered at t_now >= 2.9e-3 Myr.",
    "evidence": "A: TFINAL_ENERGY_PHASE = 3e-3 (:54) enters the while condition as (TFINAL_ENERGY_PHASE - t_now) > DT_EXIT_THRESHOLD (:138) with DT_EXIT_THRESHOLD = 1e-4; nothing reads a .param key for it, and SEGMENT_DURATION, COOLING_UPDATE_INTERVAL, RTOL and ATOL are likewise module constants. B documents it only as 'max duration (~3000 years)' -- i.e. as an ELAPSED duration, which is not what the code computes. C classifies a hard cap as a run limit, not physics: hitting it must be a distinct recorded outcome and must exceed any physically reachable transition time in the shipped configs.",
    "expected": "The constants belong in the .param schema (project convention: do not hardcode values that belong in a .param); the cap should be relative to phase entry or tSF; and hitting it must be recorded as a numerical cutoff distinct from every physical fate.",
    "failure_scenario": "Any configuration whose phase-entry time exceeds 2.9e-3 Myr skips the energy phase entirely -- zero segments, no logged error, and the post-loop reconciliation runs on untouched entry values. Separately, a low-density/weak-feedback config truncated at the cap has that truncation reported as a physical transition.",
    "repro": "Enter run_energy with params['t_now'] = 3e-3 and observe loop_count == 0 in the completion log; scan metadata.json across the shipped sweeps for runs whose energy phase ends exactly at the cap.",
    "confidence": "medium",
    "lenses": ["A", "B", "C"],
    "divergence": "AB",
    "status": "corroborated",
    "source_ids": ["S4-A-10", "S4-C-35"]
  },
  {
    "id": "S4-R-15",
    "file": "trinity/phase1_energy/run_energy_phase.py",
    "line": 57,
    "class": "numerical",
    "severity": "S3",
    "claim": "The periodic cooling-structure refresh can never re-fire inside this phase: COOLING_UPDATE_INTERVAL = 5e-2 Myr (50 kyr) exceeds TFINAL_ENERGY_PHASE = 3e-3 Myr (3 kyr) by ~16.7x.",
    "evidence": "B derived the ratio from the two documented constants ('recalculate cooling every 50k years' :57 vs 'max duration (~3000 years)' :54) and notes the block at :121 is labelled 'computed periodically'. A independently confirms both literals in its numeric table (:57 = 5e-2, :54 = 3e-3). A adds a second, compounding defect at :124: the test subtracts the parameter wrapper objects (`params['t_previousCoolingUpdate'] - params['t_now']`) instead of their .value, unlike the assignment two lines below. C independently argues the schedule is wrong in kind: L_cool varies on the current-age timescale, so a fixed absolute interval gives a staleness error ~0.54*dt/t that diverges at early times -- exactly this phase.",
    "expected": "Either an interval shorter than the phase (ideally logarithmic, dt <= f*min(t, R2/v2, Eb/|Edot_b|)), or an explicit comment that the cooling structure is deliberately computed once for this phase; and the .value fix at :124.",
    "failure_scenario": "The cooling structure is stale for the whole of Phase 1 while the code and comment imply it is refreshed; a reader trusts an unreachable branch, and the exit time cannot be resolved better than the update interval.",
    "repro": "Instrument the cooling-recompute branch with a counter; run param/simple_cluster.param; assert count > 1 if 'periodically' is intended. Check whether the params entry type implements __sub__ and what it returns.",
    "confidence": "high",
    "lenses": ["A", "B", "C"],
    "divergence": "AB",
    "status": "corroborated",
    "source_ids": ["S4-B-03", "S4-A-11", "S4-C-23"]
  },
  {
    "id": "S4-R-16",
    "file": "trinity/phase1_energy/energy_phase_ODEs.py",
    "line": 244,
    "class": "units",
    "severity": "S3",
    "claim": "The ISM counter-pressure is formed as params['PISM'] * k_B, which is a pressure only if PISM already carries K*pc^-3; if the cm^-3 -> pc^-3 conversion (2.9380e55) is missing upstream, the term silently vanishes instead of raising.",
    "evidence": "A's dimensional analysis: '(:244) P_ISM * k_B is a pressure only if params['PISM'] carries units of K*pc^-3 (i.e. P/k_B in au) ... If PISM were already a pressure the extra k_B (~7e-60) would annihilate the term rather than raise a dimensional error -- silent, not loud.' A checks magnitudes under the assumed reading and finds them sane (1e4 K cm^-3 -> 2.9e59 K pc^-3 -> ~2.1 Msun pc^-1 Myr^-2), and notes params['nISM'] exists on the snapshot and is unused, consistent with PISM being the n*T form. C independently states the requirement: the declared input is P/k_B in K cm^-3, so the boundary must multiply by k_B = 7.2606e-60 AND convert cm^-3 -> pc^-3; the paperII grid sweeps PISM to 1e6 K cm^-3, where the confining term is not negligible.",
    "expected": "An explicit, single, located conversion at the parameter boundary; the term entering with a minus sign (it does).",
    "failure_scenario": "A missing or doubled cm^-3 -> pc^-3 conversion gives a quietly wrong confining pressure exactly in the high-PISM sweep cells, with no dimensional error anywhere.",
    "repro": "Set PISM to 1e6 and confirm the reported external-pressure force equals 4*pi*R2^2 * 1e6 * k_B in AU units after conversion; locate the single site where PISM is converted.",
    "confidence": "medium",
    "lenses": ["A", "C"],
    "divergence": "AC",
    "status": "corroborated",
    "source_ids": ["S4-C-37", "S4-A-lens-sec3"]
  },
  {
    "id": "S4-R-17",
    "file": "trinity/phase1_energy/energy_phase_ODEs.py",
    "line": 421,
    "class": "other",
    "severity": "S3",
    "claim": "The ODEResult field named F_ram is assigned the bubble thermal-pressure force 4*pi*R2^2*Pb, not a ram-pressure force, and the driver copies it into params['F_ram'] -- so a force-budget sum built from the reported keys double-counts the interior pressure.",
    "evidence": "A: :421 `F_ram = Pb * 4 * np.pi * R2**2`, while the actual ram pressure is stored separately as P_ram (:428, zero outside the 'transition' branch); run_energy_phase.py:238-239 writes this into params['F_ram'], and params['F_ram_wind'] / params['F_ram_SN'] (:256-257) hold the SPS momentum injection rates that never enter dv2/dt. C's own repro for the 'no bare wind momentum' check ('check the reported F_ram_wind / F_ram_SN snapshot keys are zero or excluded from F_tot') and its force-closure requirement both read exactly these keys.",
    "expected": "Either rename the field to reflect the thermal-pressure force, or assign 4*pi*R2^2*P_ram; and make the exported key set closed under the actual force budget.",
    "failure_scenario": "A force-budget plot or table that sums F_ram + F_HII - F_ion_in - F_grav + F_rad double-counts the interior pressure (P_b and P_HII are combined by max in the ODE, never summed) and does not reproduce M_sh*dv2/dt -- so the published stacked-area force figures are wrong while looking correct.",
    "repro": "Compare params['F_ram'] against 4*pi*R2^2*params['P_ram'] in any run output; then attempt C's closure check with the exported keys and see it fail.",
    "confidence": "high",
    "lenses": ["A", "C"],
    "divergence": "AC",
    "status": "corroborated",
    "source_ids": ["S4-A-09", "S4-C-17", "S4-C-08"]
  },
  {
    "id": "S4-R-18",
    "file": "trinity/phase1_energy/energy_phase_ODEs.py",
    "line": 60,
    "class": "divergence",
    "severity": "S3",
    "claim": "The module's central purity invariant is false as stated: the docstrings claim the snapshot ensures the RHS 'never reads from or writes to the params dictionary during integration' and that 'All parameters are read at the start', but the RHS reads params three times on every call.",
    "evidence": "B collects the contradiction from the prose alone: :60 and :3 make the claim, while :169 ('params_for_feedback ... used ONLY for get_current_sps_feedback'), :194 ('this reads from params but doesn't write'), :199 (mass_profile 'only reads from params') and :234 (get_press_ion 'only reads from params') admit it. A confirms from the code: the RHS calls get_current_sps_feedback(t, params_for_feedback) at :195-197, get_mass_profile(R2, params, ...) at :207, and get_press_ion(rShell, params_for_feedback) at :238. The load-bearing claim is therefore 'never WRITES', which is asserted for three helpers by comment only.",
    "expected": "Docstrings that say params is read (read-only) during evaluation for feedback, shell mass and get_press_ion; and the read-only-ness of those three helpers enforced by a test rather than asserted.",
    "failure_scenario": "If any of the three writes to params or caches into module-level global state (trinity is documented as leaking module-level globals), a rejected RK45 trial step corrupts params exactly as the module exists to prevent -- silently, with no error, producing a wrong trajectory.",
    "repro": "deepcopy(params) before a solve_ivp segment in run_energy; run the segment; assert no key changed DURING integration (not just after).",
    "confidence": "high",
    "lenses": ["A", "B"],
    "divergence": "AB",
    "status": "corroborated",
    "source_ids": ["S4-B-01"]
  },
  {
    "id": "S4-R-19",
    "file": "trinity/phase1_energy/energy_phase_ODEs.py",
    "line": 302,
    "class": "divergence",
    "severity": "S3",
    "claim": "Both docstrings describing when derived quantities are produced are stale: ODEResult says 'computed during last ODE evaluation' (:302) and compute_derived_quantities says 'called ONCE after integration completes' (:326), while the line numbers place the call BEFORE the segment's integration, on the segment-start state.",
    "evidence": "B found the direct self-contradiction between :302 and :326. A's control-flow reading resolves it: the driver's field-by-field transfer of ODEResult sits at run_energy_phase.py:232-255 (B's step 5, 'forces and diagnostics', :225), while create_ODE_snapshot and solve_ivp are step 7 at :289-321, and A separately notes 'save_snapshot() at :262 records the state at segment start, before integration'. So the diagnostics describe the pre-ODE state -- neither of the two documented timings.",
    "expected": "One consistent statement matching the call site; and, if the diagnostics are pre-ODE by design, that stated explicitly, since it means every recorded force corresponds to the state at the START of the segment it is filed under.",
    "failure_scenario": "An energy-budget or force-budget audit of the output silently compares forces at t_n against a trajectory segment [t_n, t_n+dt]; the offset is one segment and always in the same direction. B's feared failure (diagnostics captured from a REJECTED trial step) is NOT the mechanism -- that concern is refuted by the call-site reading.",
    "repro": "Log t at each compute_derived_quantities call and compare with the segment start and end times returned by solve_ivp.",
    "confidence": "medium",
    "lenses": ["A", "B"],
    "divergence": "AB",
    "status": "corroborated",
    "source_ids": ["S4-B-02"]
  },
  {
    "id": "S4-R-20",
    "file": "trinity/phase1_energy/energy_phase_ODEs.py",
    "line": 253,
    "class": "numerical",
    "severity": "S3",
    "claim": "P_drive = max(P_b, P_HII) is a non-differentiable kink fed raw to an adaptive solver, and it is an uncited modelling choice -- the slice contains no source for taking a maximum rather than a sum of two co-existing pressures.",
    "evidence": "A: :251-258 build P_drive with np.max-style branching inside the RHS. B: the rule is stated verbatim twice (:257, :393) and the only model citation in the whole slice is 'the Weaver+77 bubble expansion model' with no equation number; there is no Rahner reference anywhere. C: max(a,b) has a kink wherever a=b, so an adaptive controller repeatedly rejects steps there and the achieved accuracy is not the requested RTOL; C enumerates four further kink sources in the same RHS (rho_amb discontinuous at r_cloud and kinked at r_core, linearly interpolated SPS drivers kinked at every table node, age-indexed cooling files piecewise-constant in cluster age, and the snapshot refresh itself).",
    "expected": "Either an event-with-restart at the crossing or a documented C1 blend; and a citation (or an explicit statement that this is TRINITY's own modelling choice, with the work-balance consequence of R-01 addressed).",
    "failure_scenario": "Step-size collapse near the crossing; and if the correct treatment is additive, the shell is under-accelerated whenever P_b and P_HII are comparable -- a systematic, physically plausible-looking error in the exact regime TRINITY claims as novel.",
    "repro": "Log the active max() branch and the accepted/rejected step sizes per step; look for step collapse coincident with branch switches, at the r_cloud crossing, and at SPS table nodes.",
    "confidence": "high",
    "lenses": ["A", "B", "C"],
    "divergence": "AC",
    "status": "corroborated",
    "source_ids": ["S4-C-13", "S4-C-38", "S4-B-14"]
  },
  {
    "id": "S4-R-21",
    "file": "trinity/phase1_energy/energy_phase_ODEs.py",
    "line": 226,
    "class": "coefficient",
    "severity": "S3",
    "claim": "Whether the pressure closure is gamma-general and R1-aware cannot be established from this slice: gamma and R1 are passed to get_effective_bubble_pressure, but the body is out of slice, and a hard-coded E_b/(2*pi*R2^3) would silently ignore both.",
    "evidence": "A: R1 = solve_R1(R2, Eb, L_mech, v_mech) at :223 and get_effective_bubble_pressure(current_phase, Eb, R2, R1, gamma, L_mech, v_mech, t, tSF) at :226-231 -- both arguments are plumbed through. B: :361 'In momentum phase, this returns pRam; in energy phase, returns bubble_E2P' -- so the energy-phase closure is bubble_E2P in trinity/.../get_bubbleParams. C: the correct closure is P_b = 3(gamma-1)*E_b/[4*pi*(R2^3 - R1^3)]; the literal E_b/(2*pi*R2^3) bakes in gamma=5/3 AND R1<<R2 simultaneously, so a declared gamma_adia parameter would do nothing.",
    "expected": "bubble_E2P using (gamma-1)*E_b/V_b with V_b = (4*pi/3)*(R2^3 - R1^3) and gamma read from the parameter; R1 enforced 0 < R1 < R2.",
    "failure_scenario": "A gamma_adia parameter that silently does nothing, plus a P_b overestimate of relative size 3*(R1/R2)^3 that grows late in the energy phase as P_b falls and R1 = sqrt(pdot_w/(4*pi*P_b)) grows. Pairs with R-04: if V_b drops R1 the PdV term is consistent, if it keeps R1 the PdV term is not.",
    "repro": "Read bubble_E2P. Set gamma_adia to a non-5/3 value and check P_b/E_b changes; recompute P_b from Eb, R2, R1 in dictionary.jsonl and compare; assert 4*pi*R1^2*Pb == pdot_w (or the 0.866 strong-shock variant) at every snapshot and R1/R2 < 1.",
    "confidence": "medium",
    "lenses": ["A", "C"],
    "divergence": "AC",
    "status": "corroborated",
    "source_ids": ["S4-C-05", "S4-C-32"]
  },
  {
    "id": "S4-R-22",
    "file": "trinity/phase1_energy/energy_phase_ODEs.py",
    "line": 219,
    "class": "citation",
    "severity": "S3",
    "claim": "The equations of motion have no written form anywhere in the slice: four one-line section headers, one stated sign, no coefficient, no exponent, no Weaver+77 equation number, no Rahner reference, and no documented units for R2, v2 or Eb.",
    "evidence": "B's exhaustive transcription: 'Gravity force (self + cluster)' (:219), 'Radiation force' (:260), 'Time derivatives' (:263), 'Energy derivative' (:272) are the entire documentation of the EOM; the only sign stated is 'Inward pressure from photoionized gas outside shell' (:233); the only equation-style citation in the slice is '(Eq. leak)' (:274), which names no source; t is documented as Myr and get_press_ion's r as pc, but the state components carry no units and 'code units' (:37) is never defined. A had to consult trinity/_functions/unit_conversions.py to pin the unit system at all.",
    "expected": "Each EOM term carrying its source equation and sign convention, and the state vector documented with units, per the project's own unit/sign bug-class warning.",
    "failure_scenario": "A sign flip or a missing factor in any EOM term cannot be caught by reading the code against its documentation -- it would show only as a quantitatively wrong but plausible expansion history. This audit's entire sign check rests on one lens's reading of the code, with no independent documentary check.",
    "repro": "",
    "confidence": "high",
    "lenses": ["B"],
    "divergence": "none",
    "status": "single-lens",
    "source_ids": ["S4-B-06", "S4-B-07"]
  },
  {
    "id": "S4-R-23",
    "file": "trinity/phase1_energy/energy_phase_ODEs.py",
    "line": 274,
    "class": "citation",
    "severity": "S3",
    "claim": "The covering-fraction leak term is cited as '(Eq. leak)' with no source, and its prefactor -- enthalpy gamma/(gamma-1) = 5/2 versus internal-energy 3/2 for gamma=5/3, a 40% difference -- cannot be checked from this slice.",
    "evidence": "B: '(Eq. leak)' at :274 is the only equation-style citation in the slice and names no paper, thesis, section or number; the functional form is documented only as 'geometry-set', depending on (Pb, R2, c_s, Cf), and vanishing at Cf=1. A confirms the call signature get_leak_luminosity(coverFraction, R2, P_b, c_sound, gamma) at :277-279 -- exactly C's required argument set, and it takes gamma. C requires L_leak = (1-C_f)*4*pi*R2^2*c_s*[gamma/(gamma-1)]*P_b, noting freely-venting gas carries enthalpy, not internal energy.",
    "expected": "A resolvable reference for the leak formula, and confirmation that the prefactor is the enthalpy one; Cf=1 must reduce to exactly 0.0 (not ~1e-30).",
    "failure_scenario": "A 40% error in the venting loss whenever Cf < 1, invisible to review because no source exists to check against; and if Cf=1 does not reduce to exactly zero the default runs silently carry a leak.",
    "repro": "Read get_leak_luminosity and compare against (1-Cf)*4*pi*R2^2*c_s*(gamma/(gamma-1))*P_b. Run with coverFraction = 1.0 and confirm the registered leak output is identically 0.0 at every step, and dictionary.jsonl is byte-identical to a build with the leak term removed.",
    "confidence": "high",
    "lenses": ["B", "C"],
    "divergence": "BC",
    "status": "corroborated",
    "source_ids": ["S4-B-05", "S4-C-33", "S4-B-09"]
  },
  {
    "id": "S4-R-24",
    "file": "trinity/phase1_energy/energy_phase_ODEs.py",
    "line": 264,
    "class": "other",
    "severity": "S3",
    "claim": "No dimensionless asymptotic regression test exists for the energy phase, so the strongest unit-bug-immune checks on this ODE system are unexercised.",
    "evidence": "C derived four executable, prefactor-free invariants: alpha = v2*t/R2 -> 3/(5-w) (0.600 uniform); Eb/(L_mech*t) -> 1/(1+2*eta) (5/11 uniform, 2/5 at w=1, 1/3 at w=2); v2 exponent (w-2)/(5-w); and delta = (2/7)*(2*alpha - beta - 1) = -6/35. C also notes the published radiusComparison figure anchors the power law to the simulation's own curve, so it tests the EXPONENT only and can never catch a prefactor error, and that a validation asserting Weaver's 28 pc against a mu_H = 1.4 code is 7% wrong in radius (30% in swept mass) and would pass a code carrying a compensating mu bug (the correct value is 26.22 pc).",
    "expected": "A pytest case in the suite: gravity/radiation/external-pressure disabled, uniform medium, checking alpha -> 0.600 and Eb/(L*t) -> 0.4545 through the energy phase, plus densPL_alpha = -1, -2 for the exponent pair.",
    "failure_scenario": "Any missing, doubled or mis-signed force shows up in alpha first -- C notes double-counted deceleration drives alpha toward ~0.5. Without this test, the S1 and every coefficient item above can only be caught by reading.",
    "repro": "Compute v2*t/R2 and Eb/(Lmech_W*t_now) from dictionary.jsonl during the energy phase of a loss-free run; fit dlnR2/dlnt and dlnv2/dlnt for densPL_alpha = 0, -1, -2.",
    "confidence": "high",
    "lenses": ["C"],
    "divergence": "none",
    "status": "single-lens",
    "source_ids": ["S4-C-14", "S4-C-15", "S4-C-16", "S4-C-36"]
  },
  {
    "id": "S4-R-25",
    "file": "trinity/phase1_energy/energy_phase_ODEs.py",
    "line": 135,
    "class": "units",
    "severity": "S3",
    "claim": "The radiation force is structurally correct and NOT double-counted, but the IR factor's unit conversion is unverified: tau must be kappa_IR * M_sh/(4*pi*R2^2) with kappa_IR = 8.3556e-4 pc^2/Msun (from 4 cm^2/g), and c must be 3.06601e5 pc/Myr.",
    "evidence": "A: F_rad = f_absWeightedTotal * (L_bol/c) * (1.0 + tauKappaRatio * dust_KappaIR) at :135, a single expression with no separately added direct term -- this is C's preferred form (a), so the classic 'direct term counted twice inside the IR term' trap is NOT triggered. B confirms the prose calls it 'direct + IR-trapped' (:129) and that it is computed inside create_ODE_snapshot, hence frozen per segment. C requires the conversions and warns that mixing kappa_IR (per gram) with sigma_d (1.5754e-58 pc^2, per H nucleus) is a ~1e23 error that fails loudly, while a missed kappa_IR conversion alone is a quiet factor ~1.2e-4 that switches IR trapping off.",
    "expected": "tauKappaRatio == M_sh/(4*pi*R2^2) in AU, dust_KappaIR == 8.3556e-4 pc^2/Msun; limits F_rad -> L_bol/c as tau_UV -> inf with tau_IR -> 0, and F_rad -> 0 as both -> 0.",
    "failure_scenario": "IR trapping silently absent (quiet 1.2e-4) in the optically thick early phase where, per C, L_bol/c already exceeds the wind momentum by ~3x -- i.e. the dominant early force is the one silently switched off.",
    "repro": "Check tau at a snapshot equals kappa_IR_AU * M_sh/(4*pi*R2^2) with M_sh in Msun and R2 in pc; check the c used in L_bol/c is 3.06601e5 pc/Myr.",
    "confidence": "medium",
    "lenses": ["C"],
    "divergence": "AC",
    "status": "single-lens",
    "source_ids": ["S4-C-10", "S4-C-09"]
  },
  {
    "id": "S4-R-26",
    "file": "trinity/phase1_energy/energy_phase_ODEs.py",
    "line": 37,
    "class": "regime",
    "severity": "S3",
    "claim": "The geometry assumed for photoionized gas during the energy phase is an open physics question that changes P_drive qualitatively, and it is documented nowhere.",
    "evidence": "C: during the energy phase the volume r < R2 holds 1e6-1e7 K shocked wind at n ~ 1e-2 cm^-3 whose recombination rate is negligible, so the real ionized gas is a thin dense skin on the shell's inner face at far higher pressure; the classical filled-sphere Stromgren formula is the 'no bubble' pressure (SPEC-030 Readings A vs B). C calls this the highest-value question in the slice. A shows the code carries TWO distinct ionized-gas pressures: the interior P_HII precomputed in the runner from a Stromgren n_IF_Str (run_energy_phase.py:214), and the exterior counter-pressure from get_press_ion using the ambient density profile. B documents 'Stromgren ionization balance' three times but never states the geometry.",
    "expected": "One documented reading, with the inconsistency of co-existing 1e4 K and 1e7 K gas in the same volume acknowledged and its effect on max(P_b, P_HII) stated.",
    "failure_scenario": "The filled-sphere reading systematically under-estimates P_HII during the energy phase, so max(P_b, P_HII) almost never selects P_HII and TRINITY's stated novelty is inert; the opposite reading over-drives the shell (and, via R-01, creates more energy from nothing).",
    "repro": "Log how often the P_HII branch of the max wins during the energy phase across param/simple_cluster.param and both f1edge configs.",
    "confidence": "medium",
    "lenses": ["C"],
    "divergence": "none",
    "status": "single-lens",
    "source_ids": ["S4-C-12"]
  },
  {
    "id": "S4-R-27",
    "file": "trinity/phase1_energy/energy_phase_ODEs.py",
    "line": 54,
    "class": "divergence",
    "severity": "S3",
    "claim": "get_press_ion computes an INWARD counter-pressure from the AMBIENT density profile at the shell radius, not a Stromgren-balance interior pressure -- a term Lens C's derived budget does not contain at all.",
    "evidence": "A: get_press_ion(r, params) = (mu_convert/mu_ion_shell) * n(r) * k_B * T_shell_ion where n(r) = density_profile.get_density_profile(r, params) (:52-55), entering dv2/dt with a minus sign and gated on f_absIon < 1. B corroborates the sign and direction with the only explicit sign statement in the whole slice: 'Inward pressure from photoionized gas outside shell' (:233). C's force budget has photoionized gas only as an OUTWARD driving term (P_HII) and contains no exterior ionized counter-pressure; C's Stromgren test (P_HII proportional to Qi^(1/2) * r^(-3/2), with the 2.2 composition factor) was aimed at get_press_ion and is therefore mis-targeted -- it belongs to the runner's n_IF_Str path. The (mu_convert/mu_ion_shell) ratio in A's expression is plausibly the composition factor C asks for, which would also close C's '2.2x under-estimate' concern.",
    "expected": "Either C's spec is incomplete (an ionized skin outside the shell within the cloud is physical, and the code's sign is right), or the same ionized gas is being counted on both sides of the shell. This must be resolved by the model's own definition, not by the audit.",
    "failure_scenario": "If the interior P_HII and the exterior P_ion describe the same gas, the shell feels it as both a driver (via max) and a brake simultaneously; if they are genuinely different populations, C's budget is simply missing a term and no defect exists.",
    "repro": "Compare get_press_ion(rShell) against the runner's P_HII from n_IF_Str at the same snapshot; check whether they can be simultaneously large. Verify get_press_ion(r) vs get_press_ion(2r) does NOT follow r^(-3/2) (it should follow the ambient profile).",
    "confidence": "medium",
    "lenses": ["A", "B", "C"],
    "divergence": "AC",
    "status": "contested",
    "source_ids": ["S4-C-11", "S4-B-transcription-233"]
  },
  {
    "id": "S4-R-28",
    "file": "trinity/phase1_energy/energy_phase_ODEs.py",
    "line": 273,
    "class": "divergence",
    "severity": "S3",
    "claim": "Radiative cooling is subtracted in the integrated energy equation but is never mentioned in this module's prose -- the largest single sink in the energy budget is undocumented at its use site.",
    "evidence": "A: dEb/dt = (L_mech - L_cool) - 4*pi*R2^2*P_b*v2 - L_leak at :280, with L_cool = snapshot.bubble_LTotal frozen at :273. B, transcribing prose only, reconstructs the energy equation as 'Ed = (mechanical energy input) - (P dV term) - L_leak' and states explicitly: 'Whether cooling L_cool appears in Ed is NOT stated in this file's prose (it appears only in the runner's transition-trigger comment).' C requires L_cool as a first-class loss term with the conduction front included.",
    "expected": "The energy-derivative comment naming all four terms, with the source of L_cool (bubble_LTotal) and its frozen-per-segment status stated.",
    "failure_scenario": "A reader auditing the energy budget from the comments alone reconstructs a three-term equation and concludes the bubble is over-energised; conversely, a maintainer editing bubble_LTotal has no local indication that it feeds the ODE.",
    "repro": "",
    "confidence": "high",
    "lenses": ["A", "B"],
    "divergence": "AB",
    "status": "corroborated",
    "source_ids": ["S4-B-lens-sec2.3", "S4-A-lens-sec2"]
  },
  {
    "id": "S4-R-29",
    "file": "trinity/phase1_energy/energy_phase_ODEs.py",
    "line": 80,
    "class": "deadcode",
    "severity": "S4",
    "claim": "Six further ODESnapshot fields are populated and never read, and two of them (Lmech_total, v_mech_total) are shadowed by a per-call feedback lookup, so the snapshot values are misleading rather than merely unused.",
    "evidence": "A: Lmech_total (:80), v_mech_total (:81), Qi (:80), caseB_alpha (:88), nISM (:92), TShell_ion (:93) are assigned at :138-164 and never referenced in get_ODE_Edot_pure or compute_derived_quantities; both consumers instead call get_current_sps_feedback(t, params_for_feedback) at :195-197 and :334-336 and bind local names that shadow the snapshot fields. n_IF and R_IF are pure pass-through into ODEResult. (include_PHII is tracked separately as R-03 because it is a documented switch.)",
    "expected": "Remove the unused fields, or use them.",
    "failure_scenario": "A maintainer editing the snapshot's Lmech_total believes they have changed the feedback the ODE sees; they have not.",
    "repro": "grep for 'snapshot.Lmech_total' / 'snapshot.Qi' / 'snapshot.caseB_alpha' in the module.",
    "confidence": "high",
    "lenses": ["A"],
    "divergence": "none",
    "status": "single-lens",
    "source_ids": ["S4-A-13"]
  },
  {
    "id": "S4-R-30",
    "file": "trinity/phase1_energy/run_energy_phase.py",
    "line": 330,
    "class": "state",
    "severity": "S4",
    "claim": "The simulation-ending event path returns before the phase-boundary reconciliation and the exit logging, unlike every other exit from the phase.",
    "evidence": "A: :329-330 `if event_result.is_simulation_ending: return`, while all other exits (normal loop termination, bubble-solve failure break at :183, cooling_balance break at :287, Eb<=0 break at :379) fall through to the reconciliation block at :390-402 and the exit log at :406-407. B independently documents why that block matters: 'A bare save_snapshot() would save stale derived values AND block the next phase's correct first snapshot via the duplicate guard' (:383).",
    "expected": "Consistent exit handling, or an explicit statement that the ending event already leaves params complete.",
    "failure_scenario": "On a simulation-ending event, params['R1'], params['Pb'] and params['shell_mass'] keep their previous-segment values and no final save_snapshot() is taken, so the last recorded state differs in kind from every other run's last state.",
    "repro": "Trigger a simulation-ending event and compare the final snapshot count against a normally-terminated run.",
    "confidence": "medium",
    "lenses": ["A", "B"],
    "divergence": "none",
    "status": "single-lens",
    "source_ids": ["S4-A-16", "S4-B-17"]
  },
  {
    "id": "S4-R-31",
    "file": "trinity/phase1_energy/run_energy_phase.py",
    "line": 136,
    "class": "deadcode",
    "severity": "S4",
    "claim": "continueWeaver is initialised True and never reassigned, so the third conjunct of the segment-loop condition is inert.",
    "evidence": "A: :136 `continueWeaver = True`; :138 `while R2 < rCloud and (...) and continueWeaver:`; no other assignment exists in the file and every early exit uses break/return.",
    "expected": "Remove the flag, or restore whatever assignment was intended to control it.",
    "failure_scenario": "",
    "repro": "grep -n continueWeaver trinity/phase1_energy/run_energy_phase.py",
    "confidence": "high",
    "lenses": ["A"],
    "divergence": "none",
    "status": "single-lens",
    "source_ids": ["S4-A-14"]
  },
  {
    "id": "S4-R-32",
    "file": "trinity/phase1_energy/run_energy_phase.py",
    "line": 59,
    "class": "numerical",
    "severity": "S4",
    "claim": "A single scalar ATOL serves a state vector spanning ~10 decades -- but the shipped value 1e-9 lands on the benign branch of the concern, so this is a cleanliness item, not a defect.",
    "evidence": "C argued a scalar ATOL 'is either negligible for E_b (harmless) or, if sized for E_b, catastrophically loose for R2/v2'. A supplies the value: ATOL = 1e-9 with R2 ~ 1 pc, v2 ~ 1e2-1e3 pc/Myr, Eb ~ 1e7 AU, i.e. sized for the small components -- the harmless branch. Eb is then RTOL-controlled (1e-6 * 1e7 = 10 AU) and R2/v2 are tightly controlled. C's stated failure mode (uncontrolled R2/v2) therefore does not apply as shipped.",
    "expected": "A per-component ATOL vector for clarity; no behavioural change expected.",
    "failure_scenario": "Only if ATOL is ever raised to suit Eb would R2/v2 become uncontrolled -- most damaging near v2 -> 0, where ATOL alone governs the recollapse verdict.",
    "repro": "Tighten RTOL by 10x and check R2(t), v2(t) and the transition time move by less than the claimed tolerance.",
    "confidence": "medium",
    "lenses": ["A", "C"],
    "divergence": "none",
    "status": "corroborated",
    "source_ids": ["S4-C-25"]
  },
  {
    "id": "S4-R-33",
    "file": "trinity/phase1_energy/energy_phase_ODEs.py",
    "line": 31,
    "class": "silent-failure",
    "severity": "S4",
    "claim": "_scalar sits at a shape boundary and, per its own docstring, returns x unchanged for anything that is not len-1 or 0-d -- so a helper that starts returning a vector passes through silently rather than raising.",
    "evidence": "B transcribes the contract at :31: 'Convert len-1 arrays / 0-d arrays to Python scalars; otherwise return x' -- i.e. identity for anything else. C requires it be total on what it receives and not silently discard information at the boundary between array-returning physics helpers and a scalar ODE RHS.",
    "expected": "Accept 0-d and length-1 only; raise on anything else.",
    "failure_scenario": "A vectorised helper (e.g. a cooling lookup) contributes an array where a scalar is expected; numpy broadcasting then produces a plausible RHS with no error anywhere.",
    "repro": "Pass a length-2 array and confirm it raises rather than passing through.",
    "confidence": "medium",
    "lenses": ["B", "C"],
    "divergence": "BC",
    "status": "corroborated",
    "source_ids": ["S4-C-34", "S4-B-lens-sec1.7"]
  },
  {
    "id": "S4-R-34",
    "file": "trinity/phase1_energy/run_energy_phase.py",
    "line": 63,
    "class": "other",
    "severity": "S4",
    "claim": "Naming and step-numbering hygiene: the module is called 'Phase 1' at :63 but '1a' at :359/:364 with 'phases 1b/1c/2' as siblings at :133; and the main loop's numbered steps run 1, 2, 3, 3b, 3c, [no 4], 5, 6, 6b, 7, 8 with 'Calculate sound speed' (:221) unnumbered exactly where step 4 would be.",
    "evidence": "B, from the prose alone. A independently confirms the step sites via its control-flow map.",
    "expected": "One consistent phase name (the 1a/1b distinction is load-bearing for the Eb<=0 routing in R-08) and contiguous step numbering.",
    "failure_scenario": "A reader applies a statement about 'Phase 1' (which includes 1b) to 1a or vice versa -- directly relevant given that Eb<=0 routing exists in 1b but not 1a.",
    "repro": "",
    "confidence": "high",
    "lenses": ["B"],
    "divergence": "none",
    "status": "single-lens",
    "source_ids": ["S4-B-18", "S4-B-19"]
  },
  {
    "id": "S4-R-35",
    "file": "trinity/phase1_energy/run_energy_phase.py",
    "line": 87,
    "class": "state",
    "severity": "S4",
    "claim": "rCloud is captured once before the segment loop and used as the loop's stopping radius, while create_ODE_snapshot re-reads params['rCloud'] fresh each segment -- so the RHS and the loop condition can disagree about the cloud edge.",
    "evidence": "A: :87 `rCloud = params['rCloud'].value`, used at :138; inside the loop updateDict(params, bubble_data) at :184 and updateDict(params, shell_data) at :208 may rewrite params entries, and the snapshot at :292 re-reads params['rCloud'].",
    "expected": "Read params['rCloud'].value inside the loop condition, matching create_ODE_snapshot.",
    "failure_scenario": "If rCloud is ever recomputed during the phase, the loop keeps integrating past the new cloud edge while the ODE's P_ext branch (rShell >= rCloud) has already switched on the ISM term.",
    "repro": "grep for writes to params['rCloud'] in the bubble/shell update paths.",
    "confidence": "low",
    "lenses": ["A"],
    "divergence": "none",
    "status": "single-lens",
    "source_ids": ["S4-A-18"]
  }
]
```

---

## 8. What the literature would settle

Lens C had no access to Weaver+77 or the Rahner thesis, and Lens B found **zero** equation numbers
in the slice. These items are open questions, not defects, until a source is consulted:

| Item | What Weaver+77 / Rahner would settle |
|---|---|
| R-23 | The `(Eq. leak)` reference, and whether the vent flux is enthalpy `γ/(γ−1)P_b` (5/2) or internal `3/2 P_b` — a 40% difference in every `Cf<1` run. |
| R-21 | The `R1` convention: `√(ṗ_w/4πP_b)` vs the strict strong-shock `√(3ṗ_w/16πP_b)` (ratio 0.866). |
| R-20 / R-01 | Whether any published model uses `max(P_b, P_HII)`, or whether TRINITY owns this choice (in which case the work-balance consequence is TRINITY's to document). |
| R-26 | The intended geometry of the photoionized gas during the energy phase (filled sphere vs. inner skin) — C calls this the highest-value question in the slice. |
| R-24 | Weaver's own `μ` convention behind the "28 pc" anchor (C: 28.04 pc at `μ=1`, 26.22 pc at `μ_H=1.4`) before any validation test is written against it. |
| — | C's flagged inconsistency between the quoted interior prefactors (`T_b = 1.51e6 K`, `n_b = 4.02e-3 cm⁻³` give `n_bT_b = 6.07e3 K cm⁻³` vs the dynamical `2.5e4 K cm⁻³`) — referred to the bubble-structure slice, but the same source resolves it. |

---

## 9. Verification order (cheapest decisive first)

1. **Is `EarlyPhaseApproximation` ever true at phase entry?** One grep of the schema defaults and the
   shipped `.param` files. If yes, R-02 escalates to S1 and everything downstream is suspect.
2. **Run C's energy-ledger residual on `param/simple_cluster.param` + both `f1edge` configs**
   (R-01's repro). It simultaneously tests R-01, R-04, R-11 and R-17 and produces exactly the
   committed CSV `docs/dev/data/` wants.
3. **Read `bubble_E2P`** — one function — which closes R-21 and decides whether R-04 is a real
   asymmetry or a consistent `R1≪R2` approximation.
4. **`grep include_PHII`** (R-03) — one command decides whether a documented user switch is inert.
