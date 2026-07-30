# S4 phase1 energy — Lens A (what the code does)

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

Scope read: `phase1_energy/energy_phase_ODEs.py`, `phase1_energy/run_energy_phase.py`,
`phase1_energy/__init__.py` (empty). Shared read-only exception **used**:
`S1_units_helpers/code/_functions/unit_conversions.py` (only to pin down the astro-unit system and
the numeric value of `k_B` in au). All comments/docstrings were blanked; nothing below is inferred
from prose.

Unit system throughout ("au" = astro units): length **pc**, mass **M⊙**, time **Myr**.
Derived: velocity pc·Myr⁻¹, energy M⊙·pc²·Myr⁻², luminosity M⊙·pc²·Myr⁻³, force M⊙·pc·Myr⁻²,
pressure M⊙·pc⁻¹·Myr⁻², number density pc⁻³, `k_B` = 1.380649e-16 · `E_cgs2au`
= 7.263e-60 M⊙·pc²·Myr⁻²·K⁻¹, `G` in pc³·M⊙⁻¹·Myr⁻².

---

## 1. The ODE system

**State vector** (`trinity/phase1_energy/energy_phase_ODEs.py:192`, `:332`; assembled at
`run_energy_phase.py:294`):

```
y = (R2, v2, Eb)
  R2 : outer shell radius                     [pc]
  v2 : outer shell velocity  (dR2/dt)         [pc Myr^-1]
  Eb : bubble (hot interior) energy           [Msun pc^2 Myr^-2]
```

**Per-call quantities** (`get_ODE_Edot_pure`, lines 168–285). Everything below is recomputed on
every RHS evaluation unless marked *frozen* (taken from `ODESnapshot`, built once per outer segment
at `run_energy_phase.py:292`).

```
(L_mech, v_mech)  = get_current_sps_feedback(t, params).{Lmech_total, v_mech_total}      :195-197
```

Shell mass and its rate (`:204-217`):

```
if isCollapse (frozen):
      M_sh = M_sh^snap  (frozen)                 Mdot_sh = 0
else:
      (M_new, Mdot_new) = get_mass_profile(R2, params, return_mdot=True, rdot=v2)
      if M_sh^snap > 0 and M_new < M_sh^snap:    M_sh = M_sh^snap,  Mdot_sh = 0     # clamp
      else:                                      M_sh = M_new,      Mdot_sh = Mdot_new
```

Gravity (`:220`):

```
F_grav = G * M_sh / R2^2 * ( M_cluster + 0.5 * M_sh )
```

Inner discontinuity and interior pressure (`:223-231`):

```
R1        = solve_R1(R2, Eb, L_mech, v_mech)
P_b       = get_effective_bubble_pressure(phase, Eb, R2, R1, gamma, L_mech, v_mech, t, t_SF)
```

External (counter) pressure (`:235-244`), evaluated at the **frozen** `rShell`, not at `R2`:

```
P_ext = 0
if f_absIon^snap < 1.0 :  P_ext  = (mu_convert/mu_ion_shell) * n(rShell) * k_B * T_shell,ion
if rShell >= rCloud    :  P_ext += P_ISM * k_B
```
where `n(r) = density_profile.get_density_profile(r, params)` (`:52-55`).

Driving pressure (`:251-258`) — a **max**, not a sum:

```
if phase == 'transition':  P_drive = max( P_b , P_HII^snap + pRam(R2, L_mech, v_mech) )
else:                      P_drive = max( P_b , P_HII^snap )
```

Radiation force, **frozen**, built in `create_ODE_snapshot` (`:130-135`):

```
F_rad = 0                                                        if shell isDissolved
F_rad = f_absWeightedTotal * (L_bol / c) * (1 + tauKappaRatio * kappa_IR)   otherwise
```

**Right-hand side** (`:264-285`):

```
dR2/dt = v2                                                                       (:264)

dv2/dt = [ 4*pi*R2^2*(P_drive - P_ext) - Mdot_sh*v2 - F_grav + F_rad ] / M_sh      (:265-266)
dv2/dt = -1e8                                       if EarlyPhaseApproximation     (:269-270)

dEb/dt = ( L_mech - L_cool^snap ) - 4*pi*R2^2 * P_b * v2 - L_leak                  (:280)
   with L_leak = get_leak_luminosity(coverFraction^snap, R2, P_b, c_sound^snap, gamma)  (:277-279)
   and  L_cool^snap = snapshot.bubble_LTotal (frozen)                                   (:273)
```

Fully substituted, for the non-transition, non-collapse, non-approximation branch:

```
Ṙ₂ = v₂

v̇₂ = { 4πR₂² [ max( P_b(E_b,R₂,R₁) , P_HII )
              − 1_{f_abs<1}·(μ_c/μ_ion)·n(r_sh)·k_B·T_ion
              − 1_{r_sh≥r_cl}·P_ISM·k_B ]
      − Ṁ_sh(R₂,v₂)·v₂
      − G·M_sh·(M_cl + ½M_sh)/R₂²
      + f_abs,tot·(L_bol/c)·(1 + τ_ratio·κ_IR) }  /  M_sh

Ė_b = L_mech(t) − L_cool − 4πR₂²·P_b(E_b,R₂,R₁)·v₂ − L_leak(f_cov,R₂,P_b,c_s,γ)
```

`compute_derived_quantities` (`:325-431`) re-derives the same quantities for diagnostics but
returns **no** derivatives, and **omits** the `EarlyPhaseApproximation` override.

**What is frozen for the whole segment** (segment length `SEGMENT_DURATION = 3e-5` Myr):
`L_cool`, `F_rad`, `P_HII`, `rShell`, `f_absIon`, `isCollapse`, `M_sh` floor, `coverFraction`,
`c_sound`, `mCluster`, `PISM`, `rCloud`, `current_phase`, `EarlyPhaseApproximation`.
**What varies inside the segment**: `t` (→ `L_mech`, `v_mech`), `R2`, `v2`, `Eb` (→ `R1`, `P_b`,
`M_sh`, `Ṁ_sh`, `L_leak`, and all `R2²` factors). This is an operator-split / lagged-coefficient
scheme, not a fully coupled system.

---

## 2. Force / energy budget, term by term

### Momentum equation `M_sh · v̇₂ = Σ F`

| term | expression | sign in eq. | direction | units |
|---|---|---|---|---|
| interior thermal pressure | `+4πR₂²·P_drive`, `P_drive = max(P_b, P_HII[+P_ram])` | **+** | outward | M⊙ pc Myr⁻² |
| external ionized-gas pressure | `−4πR₂²·(μ_c/μ_ion)·n(r_sh)·k_B·T_ion`, only if `f_absIon < 1` | **−** | inward | M⊙ pc Myr⁻² |
| ambient ISM pressure | `−4πR₂²·P_ISM·k_B`, only if `rShell ≥ rCloud` | **−** | inward | M⊙ pc Myr⁻² |
| swept-up mass loading (ram) | `−Ṁ_sh·v₂` | **−** (for v₂>0) | decelerating | M⊙ pc Myr⁻² |
| gravity: cluster + shell self-gravity | `−G·M_sh·(M_cl + ½M_sh)/R₂²` | **−** | inward | M⊙ pc Myr⁻² |
| radiation, direct + IR | `+f_abs,tot·(L_bol/c)·(1 + τ_ratio·κ_IR)` | **+** | outward | M⊙ pc Myr⁻² |

All signs are the conventional ones (pressure/radiation push out, gravity and mass-loading resist).

**Present but only in the `'transition'` branch**: wind ram pressure `pRam(R2,L_mech,v_mech)`, and
even there it is added to `P_HII` *inside* the `max`, so it never adds to `P_b`.

**Absent from the momentum equation entirely:**
- direct wind/SN momentum injection. `feedback.pdot_W` and `feedback.pdot_SN` are read and stored
  into `params['F_ram_wind']`/`params['F_ram_SN']` at `run_energy_phase.py:256-257` but never
  enter `dv2/dt`. Wind momentum reaches the shell only through `P_b`.
- turbulent / magnetic ambient pressure — no such term.
- any inner-boundary force at `R1` (`R1` feeds only `P_b`).
- cloud gravity other than through `M_sh` (all interior mass is assumed to be in the shell).

### Energy equation `Ė_b = …`

| term | expression | sign | units |
|---|---|---|---|
| mechanical input | `+L_mech(t)` (SPS, re-evaluated at `t`) | **+** | M⊙ pc² Myr⁻³ |
| radiative cooling | `−L_cool` (`snapshot.bubble_LTotal`, frozen) | **−** | M⊙ pc² Myr⁻³ |
| PdV work on the shell | `−4πR₂²·P_b·v₂` | **−** for v₂>0 | M⊙ pc² Myr⁻³ |
| leakage | `−L_leak(f_cov, R₂, P_b, c_s, γ)` | **−** | M⊙ pc² Myr⁻³ |

**Absent**: any thermal-conduction / evaporation energy term as a separate entry (it can only be
inside `bubble_LTotal`); any energy sink matching the `P_HII` branch of `P_drive` (see §Findings).
Note the PdV term uses **`P_b`**, not `P_drive` — when `P_HII > P_b` the shell is accelerated by
`P_HII` while the bubble is debited only at `P_b`.

---

## 3. Dimensions

Every term of `dv2/dt`'s numerator: `4πR₂²·P` → pc²·(M⊙ pc⁻¹ Myr⁻²) = M⊙ pc Myr⁻²;
`Ṁ_sh·v₂` → (M⊙ Myr⁻¹)(pc Myr⁻¹) = M⊙ pc Myr⁻²; `G M_sh M/R₂²` →
(pc³ M⊙⁻¹ Myr⁻²)(M⊙²/pc²) = M⊙ pc Myr⁻²; `L_bol/c` → (M⊙ pc² Myr⁻³)/(pc Myr⁻¹) = M⊙ pc Myr⁻².
All four agree; dividing by `M_sh` [M⊙] gives pc Myr⁻² = acceleration. **Balanced.**

Every term of `dEb/dt`: `L_mech`, `L_cool`, `L_leak` are luminosities M⊙ pc² Myr⁻³;
`4πR₂²·P_b·v₂` → pc²·(M⊙ pc⁻¹ Myr⁻²)·(pc Myr⁻¹) = M⊙ pc² Myr⁻³. **Balanced.**

`get_press_ion` (`:54`): `(μ_c/μ_ion)` dimensionless · `n` [pc⁻³] · `k_B` [M⊙ pc² Myr⁻² K⁻¹] ·
`T` [K] = M⊙ pc⁻¹ Myr⁻². **Pressure.** Same form used for `P_HII` at `run_energy_phase.py:214`.

`P_ISM * k_B` (`:244`) is a pressure **only if `params['PISM']` carries units of K·pc⁻³**
(i.e. P/k_B in au), the same convention as `n·T` above. Under that reading the sum
`P_ext = P_ion + P_ISM·k_B` is dimensionally consistent; magnitudes are also sane
(10⁴ K cm⁻³ → 2.9e59 K pc⁻³ → ≈2.1 M⊙ pc⁻¹ Myr⁻²). If `PISM` were already a pressure the extra
`k_B` (≈7e-60) would annihilate the term rather than raise a dimensional error — silent, not loud.
`params['nISM']` exists on the snapshot and is unused, consistent with `PISM` being the n·T form.

`vd = -1e8` (`:270`) has units pc·Myr⁻², which is dimensionally an acceleration, so no imbalance —
but see §Findings for its magnitude.

No dimensional imbalance found in any summed expression.

---

## 4. Control flow that changes the maths

**In the RHS (`energy_phase_ODEs.py`)**

| line | branch | maths used instead |
|---|---|---|
| 204 | `isCollapse` true | `M_sh` frozen to snapshot value, `Ṁ_sh ≡ 0` → the `−Ṁv` term vanishes |
| 213 | `M_new < M_sh^snap` and `M_sh^snap > 0` | `M_sh := M_sh^snap`, `Ṁ_sh := 0` (clamp, both replaced) |
| 237 | `f_absIon ≥ 1.0` | `P_ext := 0` (ambient ionized counter-pressure dropped entirely) |
| 243 | `rShell ≥ rCloud` | `P_ext += P_ISM·k_B` (uses frozen `rShell`, so cannot toggle mid-segment) |
| 253 | `phase == 'transition'` | `P_drive = max(P_b, P_HII + P_ram)`; else `max(P_b, P_HII)` |
| 255/258 | `P_HII(+P_ram) > P_b` | `P_b` silently replaced by `P_HII` in the momentum eq. only |
| 269 | `EarlyPhaseApproximation` | **`dv2/dt := -1e8`**, discarding every force computed above |
| 130 | `shell isDissolved` | `F_rad := 0` |

There are no `try/except`, no early `return`, and no clamping of `Eb` or `R2` inside the RHS.

**In the driver (`run_energy_phase.py`)**

- `while R2 < rCloud and (TFINAL_ENERGY_PHASE − t_now) > DT_EXIT_THRESHOLD and continueWeaver`
  (`:138`). `rCloud` is captured once at `:87`; `continueWeaver` is set True at `:136` and never
  reassigned → the third conjunct is inert.
- `:169-183` `try/except (ValueError, RuntimeError, BubbleSolverError)` around the bubble solve →
  sets `EndSimulationDirectly`, reason "Energy-driven bubble collapsed", code `ENERGY_COLLAPSED`,
  `break`. Any numerical `ValueError` is reported as a physical collapse.
- `:276-287` `cooling_balance` trigger: `break` when `L_gain > 0` and
  `(L_gain − L_loss)/L_gain < thr`, with `thr = params['phaseSwitch_LlossLgain']` **or 0.05 if
  falsy**.
- `:310-321` if `solution.success` is False: retry with `RK23`, shortened span
  `t_now + SEGMENT_DURATION/10`, `rtol*10`, `atol*10`, **no `dense_output`**. The retry's
  `success` is never checked.
- `:324-331` event handling: `check_event_termination(solution, ode_events)`; if triggered →
  `apply_event_result(params, …, state_keys=['R2','v2','Eb'])`, then `return` if
  `is_simulation_ending` else `break`. **Any** triggered event ends the phase, whether or not it
  was declared terminal.
- `:342-344` `EarlyPhaseApproximation` is cleared only *after* the first `solve_ivp` call, and only
  if the first segment reached that line.
- `:368-379` post-segment guard: `not isfinite(Eb) or Eb <= 0` → `ENERGY_COLLAPSED`, `break`.
- `:390-404` post-loop reconciliation, wrapped in a bare `except Exception` → warning only.

**Event functions** — `build_energy_phase_events` / `check_event_termination` /
`apply_event_result` live in `trinity/phase_general/phase_events.py`, outside this slice, so the
zero-crossing quantity, direction, and `terminal` flags could not be verified. What *is* visible
here: the event list is built once (`:118`) from `params` before the loop and reused for every
segment; whatever `terminal` says, the driver breaks out of the phase on any trigger, so
terminality is effectively honoured by the caller regardless.

---

## 5. Interior pressure

`P_b` is **recomputed from the integrated state on every RHS call**: `R1 = solve_R1(R2, Eb, L_mech,
v_mech)` (`:223`) then `get_effective_bubble_pressure(phase, Eb, R2, R1, γ, L_mech, v_mech, t,
t_SF)` (`:226-231`). So the pressure that does the PdV work in `dEb/dt` and the pressure that
drives the shell are both functions of the *same* `Eb` being integrated — internally consistent
inside a step.

Three consistency caveats visible in this slice:

1. The driver's `params['Pb']` is **not** this quantity. It is set from the bubble-structure
   solver output `bubble_data.Pb` (`:190-192`), and at entry/exit from
   `get_bubbleParams.bubble_E2P(Eb, R2, R1, γ)` (`:100`, `:395`) — a *different function* from
   `get_effective_bubble_pressure`. Downstream consumers of `params['Pb']` (shell structure,
   outputs) therefore see the solver pressure, not the ODE pressure.
2. `ODEResult.Pb` and `ODEResult.R1` from `compute_derived_quantities` are computed and then
   discarded by the caller (`run_energy_phase.py:232-255` copies out every other field but not
   `Pb`/`R1`), so any disagreement between the two pressures is invisible.
3. The cooling term `L_cool` paired against `P_b` in `dEb/dt` is frozen at the segment-start `Eb`,
   while `P_b` tracks the evolving `Eb` — the loss term does not respond to the energy it drains.

---

## 6. Numeric literals

`energy_phase_ODEs.py`

| line | literal | expression |
|---|---|---|
| 131 | `0.0` | `F_rad = 0.0` (dissolved shell) |
| 135 | `1.0` | `(1.0 + tauKappaRatio * dust_KappaIR)` — IR enhancement factor |
| 206, 215 | `0.0` | `mShell_dot = 0.0` |
| 213 | `0` | `prev_mShell > 0` |
| 220, 370 | `2`, `0.5` | `G*mShell/(R2**2)*(mCluster + 0.5*mShell)` |
| 237, 374 | `1.0` | `FABSi < 1.0` |
| 240, 377 | `0.0` | `P_ext = 0.0` |
| 265 | `4.0`, `2` | `4.0*np.pi*R2**2*(P_drive - P_ext)` |
| **270** | **`-1e8`** | **`vd = -1e8`** (EarlyPhaseApproximation override) |
| 280 | `4`, `2` | `(4*np.pi*R2**2*press_bubble)*v2` |
| 396 | `4.0`, `2` | `F_HII = 4.0*np.pi*R2**2*P_HII` |
| 402 | `0.0` | `P_ram_val = 0.0` (non-transition) |
| 419 | `4`, `2` | `F_ion_in = P_ext*4*np.pi*R2**2` |
| 421 | `4`, `2` | `F_ram = Pb*4*np.pi*R2**2` |
| 282-283 | `.6f`, `.6e` | format only |

`run_energy_phase.py`

| line | literal | expression |
|---|---|---|
| 54 | `3e-3` | `TFINAL_ENERGY_PHASE` [Myr] — absolute end time of the phase |
| 55 | `3e-5` | `SEGMENT_DURATION` [Myr] |
| 56 | `1e-4` | `DT_EXIT_THRESHOLD` [Myr] |
| 57 | `5e-2` | `COOLING_UPDATE_INTERVAL` [Myr] |
| 58 | `1e-6` | `RTOL` |
| 59 | `1e-9` | `ATOL` (single scalar for R2 ~ 1 pc, v2 ~ 10²–10³ pc/Myr, Eb ~ 10⁷ au) |
| 213 | `0` | `n_IF_Str > 0` |
| 218 | `4.0`, `2` | `F_HII = 4.0*np.pi*R2**2*P_HII` |
| 281 | `0.05` | `_thr = _thr if _thr else 0.05` |
| 282 | `0` | `_Lgain > 0` |
| 312 | `10` | `t_now + SEGMENT_DURATION/10` (retry span) |
| 319-320 | `10` | `RTOL*10`, `ATOL*10` (retry tolerances) |
| 342 | `0` | `loop_count == 0` |
| 368 | `0` | `Eb <= 0` |

---

## 7. Additional observations

- **Duplicated RHS.** `get_ODE_Edot_pure` (`:192-285`) and `compute_derived_quantities`
  (`:332-431`) repeat the same ~60 lines of physics. They already differ: the second omits the
  `EarlyPhaseApproximation` override, so the recorded diagnostics never reflect the
  `vd = -1e8` path.
- **Unused `ODESnapshot` fields.** `Lmech_total`, `v_mech_total` (`:80-81`) are stored but shadowed
  in both consumers by a fresh `get_current_sps_feedback(t, …)` lookup; `Qi` (`:80`),
  `caseB_alpha` (`:88`), `nISM` (`:92`), `TShell_ion` (`:93`), `include_PHII` (`:74`) are never
  read at all; `n_IF`/`R_IF` are pure pass-through into `ODEResult`.
- **`params['t_previousCoolingUpdate'] - params['t_now']`** at `run_energy_phase.py:124` omits
  `.value` on both operands, unlike line 129 two lines below which uses `.value` on both.
- `F_HII` is computed and stored twice per segment with the same value (`:218-219`, then
  overwritten at `:236-237`).
- The `transition` branch (`energy_phase_ODEs.py:253`, `:389`) is compared against the string
  literal `'transition'`; inside `run_energy` nothing sets `current_phase`, so whether this branch
  is ever live in phase 1a depends on an out-of-slice convention.
- `mShell` at `run_energy_phase.py:99` and `Pb` at `:100` are computed only to appear in log lines.
- `save_snapshot()` at `:262` records the state at *segment start*, before integration.
- The `return` at `:330` exits without running the reconciliation block or the exit log; every
  other exit path falls through to `:390`.

---

```json
[
  {
    "id": "S4-A-01",
    "file": "trinity/phase1_energy/energy_phase_ODEs.py",
    "line": 270,
    "class": "silent-failure",
    "severity": "S2",
    "claim": "When EarlyPhaseApproximation is true the entire momentum equation is discarded and replaced by the hard-coded constant dv2/dt = -1e8 pc/Myr^2.",
    "evidence": "Lines 264-266 compute vd from the full force budget; lines 269-270 unconditionally overwrite it: `if snapshot.EarlyPhaseApproximation: vd = -1e8`. The snapshot flag is frozen for the whole segment, and run_energy_phase.py:342-344 only clears params['EarlyPhaseApproximation'] AFTER the first solve_ivp call, so segment 0 always integrates under the override. Over one SEGMENT_DURATION = 3e-5 Myr this integrates to dv2 = -3000 pc/Myr (about -2900 km/s).",
    "expected": "Either an approximate but physically motivated expression for dv2/dt, or a sign/magnitude that cannot dominate the real dynamics. A bare -1e8 replacing all six force terms is not recoverable from the arithmetic.",
    "failure_scenario": "With the flag enabled at entry, the first segment drives v2 to a large negative value regardless of the actual forces; a velocity- or radius-based termination event fires immediately and the phase ends (or the shell is recorded as collapsing) for reasons unrelated to the physics. If that first segment breaks via an event, line 343 is never reached and the flag stays set for any later re-entry.",
    "repro": "Run any .param with EarlyPhaseApproximation true and log v2 across the first segment; compare against the value implied by 4*pi*R2^2*(P_drive-P_ext)/mShell.",
    "confidence": "high"
  },
  {
    "id": "S4-A-02",
    "file": "trinity/phase1_energy/run_energy_phase.py",
    "line": 310,
    "class": "silent-failure",
    "severity": "S2",
    "claim": "The solve_ivp failure retry has no progress guard: if the RK23 retry also fails at the first step the loop can spin forever without advancing t_now.",
    "evidence": "Lines 310-321 retry with RK23 on failure but never re-check `solution.success`. Lines 336-337 then take `solution.y[:, -1]` and `solution.t[-1]` unconditionally. scipy always returns at least the initial point, so a first-step failure gives t_new == t_now and y unchanged. The while condition at line 138 depends only on R2 and t_now, both unchanged, so the loop repeats identically.",
    "expected": "Check `solution.success` after the retry, or require `t_new > t_now` before continuing, and terminate the phase with an explicit end code otherwise.",
    "failure_scenario": "A stiff configuration where both RK45 and RK23 fail the first step hangs the run in an infinite loop, re-solving the bubble and shell structure every iteration, with no error and no output progress.",
    "repro": "Force the retry path (e.g. make ode_func return NaN once) and observe t_now never advancing.",
    "confidence": "high"
  },
  {
    "id": "S4-A-03",
    "file": "trinity/phase1_energy/run_energy_phase.py",
    "line": 393,
    "class": "state",
    "severity": "S2",
    "claim": "After an event-triggered break, the post-loop reconciliation recomputes R1, Pb and shell_mass from the stale pre-segment local R2/Eb/t_now while params['R2'], params['v2'], params['Eb'] already hold the post-event state, leaving the exported state internally inconsistent.",
    "evidence": "apply_event_result at lines 327-328 writes the event state into params (state_keys=['R2','v2','Eb']) but the local variables R2, v2, Eb, t_now are not updated; line 331 breaks. Lines 391-399 then use those stale locals: get_current_sps_feedback(t_now,...), solve_R1(R2, Eb, ...), bubble_E2P(Eb, R2, R1_f, ...), get_mass_profile(R2, params, ...), and assign the results to params['R1'], params['Pb'], params['shell_mass']. Line 400 then calls shell_structure_pure(params), which reads the post-event params['R2'] together with the pre-event Pb and shell_mass.",
    "expected": "Refresh the locals from event_result.y / event_result.t before the reconciliation block (as the normal exit path at lines 346-349 does), so R1/Pb/shell_mass correspond to the same (t, R2, Eb) as the state handed to the next phase.",
    "failure_scenario": "Every event-terminated energy phase hands phase 1b a Pb, R1 and shell_mass evaluated at the previous segment's radius and energy; the mismatch grows with segment length and with how early inside the segment the event fired.",
    "repro": "Enable an event that fires mid-segment and compare params['Pb'] against bubble_E2P(params['Eb'], params['R2'], params['R1'], gamma) at phase exit.",
    "confidence": "high"
  },
  {
    "id": "S4-A-04",
    "file": "trinity/phase1_energy/energy_phase_ODEs.py",
    "line": 280,
    "class": "other",
    "severity": "S3",
    "claim": "The momentum equation is driven by P_drive = max(P_b, P_HII) but the energy equation debits PdV work only at P_b, so whenever P_HII exceeds P_b the shell is accelerated by work that no reservoir pays for.",
    "evidence": "Line 265 uses P_drive in 4.0*np.pi*R2**2*(P_drive - P_ext); line 280 uses press_bubble in (4*np.pi*R2**2*press_bubble)*v2. P_drive is set at lines 255/258 as max(press_bubble, P_HII[+P_ram]).",
    "expected": "Either both equations use the same pressure, or the P_HII-dominated regime carries its own explicit (non-Eb) energy accounting.",
    "failure_scenario": "In HII-pressure-dominated configurations (large Qi, low bubble pressure) the shell gains momentum while Eb decays only at the smaller P_b rate, so the bubble survives longer than the work it performs implies.",
    "repro": "Log press_bubble, P_HII and P_drive per segment; whenever P_HII > press_bubble the PdV term under-counts by 4*pi*R2^2*(P_HII - P_b)*v2.",
    "confidence": "high"
  },
  {
    "id": "S4-A-05",
    "file": "trinity/phase1_energy/energy_phase_ODEs.py",
    "line": 213,
    "class": "silent-failure",
    "severity": "S3",
    "claim": "The shell-mass floor clamp replaces both the computed mass and its derivative: mShell is pinned to the segment-start value and mShell_dot is forced to zero, deleting the -Mdot*v2 term from the momentum equation.",
    "evidence": "Lines 212-217 (and the identical 350-355): `if prev_mShell > 0 and mShell_new < prev_mShell: mShell = prev_mShell; mShell_dot = 0.0`. prev_mShell is snapshot.shell_mass, frozen at segment start, so the guard is relative to the segment start, not to the running mass; within a segment the mass may still decrease freely as long as it stays above that floor, in which case mShell_dot keeps its negative value.",
    "expected": "A monotone-mass guard should be consistent: either clamp the mass and set the derivative to zero everywhere it applies, or track a running maximum. As written the two paths disagree about whether shell mass may decrease.",
    "failure_scenario": "During a re-collapse (v2 < 0) the inertia is held at the pre-collapse value and the momentum term that would resist the collapse is deleted, so the acceleration is computed from an inconsistent (M, Mdot) pair.",
    "repro": "Integrate a segment with v2 < 0 and log mShell_new vs prev_mShell and mShell_dot.",
    "confidence": "high"
  },
  {
    "id": "S4-A-06",
    "file": "trinity/phase1_energy/run_energy_phase.py",
    "line": 281,
    "class": "silent-failure",
    "severity": "S3",
    "claim": "A configured phaseSwitch_LlossLgain of exactly 0.0 is silently replaced by the hard-coded 0.05 because the guard tests truthiness rather than None.",
    "evidence": "`_thr = params['phaseSwitch_LlossLgain'].value` then `_thr = _thr if _thr else 0.05`. 0.0 is falsy, so a user asking for a zero threshold (hand off only at exact cooling balance) gets 0.05 instead.",
    "expected": "`_thr = 0.05 if _thr is None else _thr`.",
    "failure_scenario": "A sweep that scans phaseSwitch_LlossLgain down through 0 produces an identical result at 0 and 0.05, silently truncating the parameter study.",
    "repro": "Set phaseSwitch_LlossLgain = 0.0 in a .param and confirm the logged threshold is 0.05.",
    "confidence": "high"
  },
  {
    "id": "S4-A-07",
    "file": "trinity/phase1_energy/run_energy_phase.py",
    "line": 171,
    "class": "silent-failure",
    "severity": "S3",
    "claim": "Any ValueError or RuntimeError raised anywhere inside the bubble-structure solve is reinterpreted as the physical statement 'Energy-driven bubble collapsed' and the run is ended with SimulationEndCode.ENERGY_COLLAPSED.",
    "evidence": "Lines 169-183: `except (ValueError, RuntimeError, bubble_luminosity.BubbleSolverError)` sets SimulationEndReason to 'Energy-driven bubble collapsed: bubble solve degenerate as Eb -> 0' and breaks. ValueError/RuntimeError are raised by numpy, scipy and interpolators for reasons unrelated to Eb (out-of-range interpolation, shape mismatch, non-convergence).",
    "expected": "Catch only BubbleSolverError (or a dedicated degeneracy exception) for the collapse conclusion, and let genuine programming/numeric errors surface.",
    "failure_scenario": "A bug or a table out-of-range condition in the bubble solver produces a physically plausible-looking 'collapsed' run that is indistinguishable in the output from a real collapse.",
    "repro": "Inject a ValueError into bubble_luminosity.get_bubbleproperties_pure and inspect SimulationEndReason.",
    "confidence": "high"
  },
  {
    "id": "S4-A-08",
    "file": "trinity/phase1_energy/energy_phase_ODEs.py",
    "line": 236,
    "class": "state",
    "severity": "S3",
    "claim": "The external counter-pressure is evaluated at the frozen snapshot radius rShell while the area it acts on uses the live R2, and the cloud-boundary test rShell >= rCloud likewise cannot toggle during a segment.",
    "evidence": "Lines 236-244: `rShell = snapshot.rShell`; `P_ext = get_press_ion(rShell, params_for_feedback)`; `if rShell >= snapshot.rCloud: P_ext += ...`. Line 265 then multiplies P_ext by 4.0*np.pi*R2**2 with the integrated R2. get_press_ion evaluates the ambient density profile, which is a steep function of radius, so the mismatch is not second order in a density gradient.",
    "expected": "Evaluate the ambient pressure at the same radius as the area factor (R2, or R2 plus a shell thickness derived from R2).",
    "failure_scenario": "In a steep density profile (rCore ~ 1 pc) the shell can move a non-negligible fraction of a scale length within a 3e-5 Myr segment; the counter-pressure then corresponds to the wrong ambient density, biasing the acceleration in the same direction for every segment.",
    "repro": "Log P_ext at segment start against get_press_ion(R2_end) at segment end for a steep-profile config.",
    "confidence": "medium"
  },
  {
    "id": "S4-A-09",
    "file": "trinity/phase1_energy/energy_phase_ODEs.py",
    "line": 421,
    "class": "other",
    "severity": "S3",
    "claim": "The ODEResult field named F_ram is assigned the bubble thermal-pressure force 4*pi*R2^2*Pb, not a ram-pressure force, and the driver copies it into params['F_ram'].",
    "evidence": "Line 421: `F_ram=Pb * 4 * np.pi * R2**2`. The actual ram pressure is stored separately as P_ram (line 428, zero outside the 'transition' branch). run_energy_phase.py:238-239 writes this into params['F_ram'], while params['F_ram_wind'] and params['F_ram_SN'] (lines 256-257) hold the SPS momentum injection rates.",
    "expected": "Either rename the field to reflect the thermal-pressure force, or assign 4*pi*R2^2*P_ram.",
    "failure_scenario": "A force-budget plot or table that sums F_ram + F_HII - F_ion_in - F_grav + F_rad double-counts the interior pressure (P_b and P_HII are combined by max in the ODE, never summed) and does not reproduce mShell*dv2/dt.",
    "repro": "Compare params['F_ram'] against 4*pi*R2^2*params['P_ram'] in any run output.",
    "confidence": "high"
  },
  {
    "id": "S4-A-10",
    "file": "trinity/phase1_energy/run_energy_phase.py",
    "line": 54,
    "class": "regime",
    "severity": "S3",
    "claim": "The energy phase end time is a hard-coded absolute 3e-3 Myr with no parameter override, and the loop body never executes at all if the phase is entered at t_now >= 2.9e-3 Myr.",
    "evidence": "TFINAL_ENERGY_PHASE = 3e-3 (line 54) is compared against absolute simulation time in the while condition `(TFINAL_ENERGY_PHASE - t_now) > DT_EXIT_THRESHOLD` (line 138), with DT_EXIT_THRESHOLD = 1e-4. Nothing reads a .param key for it. SEGMENT_DURATION, DT_EXIT_THRESHOLD, COOLING_UPDATE_INTERVAL, RTOL and ATOL are likewise module constants.",
    "expected": "These belong in the .param schema (project convention: do not hardcode values that belong in a .param), and the cap should be relative to phase entry (or tSF) rather than absolute simulation time.",
    "failure_scenario": "Any configuration whose star formation time or phase-entry time exceeds 2.9e-3 Myr skips the energy phase entirely: zero segments, no logged error, and the post-loop reconciliation runs on untouched entry values.",
    "repro": "Enter run_energy with params['t_now'] = 3e-3 and observe loop_count == 0 in the completion log.",
    "confidence": "medium"
  },
  {
    "id": "S4-A-11",
    "file": "trinity/phase1_energy/run_energy_phase.py",
    "line": 124,
    "class": "other",
    "severity": "S3",
    "claim": "The cooling-table refresh test subtracts the parameter wrapper objects instead of their .value, unlike the assignment two lines below.",
    "evidence": "Line 124: `if np.abs(params['t_previousCoolingUpdate'] - params['t_now']) > COOLING_UPDATE_INTERVAL:` versus line 129: `params['t_previousCoolingUpdate'].value = params['t_now'].value`. Every other read in this file uses .value.",
    "expected": "`np.abs(params['t_previousCoolingUpdate'].value - params['t_now'].value)`. If the wrapper defines __sub__ this happens to work today; if it does not, or if it returns a wrapper rather than a float, the comparison is not the intended one.",
    "failure_scenario": "If the wrapper's arithmetic ever changes, the non-CIE cooling structure is either never refreshed or refreshed every entry, silently changing the cooling used by the whole phase.",
    "repro": "Check whether the params entry type in trinity/_input/dictionary.py implements __sub__ and what it returns.",
    "confidence": "medium"
  },
  {
    "id": "S4-A-12",
    "file": "trinity/phase1_energy/run_energy_phase.py",
    "line": 100,
    "class": "state",
    "severity": "S3",
    "claim": "Two different interior-pressure functions are in use: the driver uses bubble_E2P (and the solver's bubble_data.Pb) while the ODE uses get_effective_bubble_pressure, and the ODE's own Pb/R1 are returned but discarded.",
    "evidence": "run_energy_phase.py:100 and :395 call get_bubbleParams.bubble_E2P(Eb, R2, R1, gamma); :190-192 set params['Pb'] from bubble_data.Pb. energy_phase_ODEs.py:226-231 and :362-367 call get_effective_bubble_pressure(current_phase, ..., t, tSF). ODEResult.Pb and ODEResult.R1 (lines 414-415) are never copied out by the driver's field-by-field transfer at run_energy_phase.py:232-255.",
    "expected": "One pressure definition per phase, or an explicit reconciliation/assertion that the two agree; at minimum the discrepancy should be observable.",
    "failure_scenario": "get_effective_bubble_pressure carries a phase/t/tSF dependence that bubble_E2P does not; in any regime where they differ, the pressure that drives the ODE is never the pressure recorded in the output or fed to shell_structure.",
    "repro": "Log ODEResult.Pb alongside params['Pb'] each segment and diff.",
    "confidence": "medium"
  },
  {
    "id": "S4-A-13",
    "file": "trinity/phase1_energy/energy_phase_ODEs.py",
    "line": 80,
    "class": "deadcode",
    "severity": "S4",
    "claim": "Seven ODESnapshot fields are populated and never read; two of them (Lmech_total, v_mech_total) are shadowed by a per-call feedback lookup, so the snapshot values are misleading rather than merely unused.",
    "evidence": "Snapshot fields Lmech_total (:80), v_mech_total (:81), Qi (:80), caseB_alpha (:88), nISM (:92), TShell_ion (:93), include_PHII (:74) are assigned at lines 138-164 and never referenced in get_ODE_Edot_pure or compute_derived_quantities. Both consumers instead call get_current_sps_feedback(t, params_for_feedback) at :195-197 and :334-336 and bind local Lmech_total/v_mech_total that shadow the snapshot names. n_IF and R_IF are pure pass-through into ODEResult.",
    "expected": "Remove the unused fields, or use them.",
    "failure_scenario": "A maintainer editing the snapshot's Lmech_total believes they have changed the feedback the ODE sees; they have not.",
    "repro": "grep for 'snapshot.Lmech_total' / 'snapshot.Qi' / 'snapshot.caseB_alpha' in the module.",
    "confidence": "high"
  },
  {
    "id": "S4-A-14",
    "file": "trinity/phase1_energy/run_energy_phase.py",
    "line": 136,
    "class": "deadcode",
    "severity": "S4",
    "claim": "continueWeaver is initialised True and never reassigned, so the third conjunct of the segment loop condition is inert.",
    "evidence": "Line 136 `continueWeaver = True`; line 138 `while R2 < rCloud and (...) and continueWeaver:`. No other assignment to continueWeaver exists in the file; every early exit uses break/return instead.",
    "expected": "Remove the flag, or restore whatever assignment was intended to control it.",
    "failure_scenario": "",
    "repro": "grep -n continueWeaver trinity/phase1_energy/run_energy_phase.py",
    "confidence": "high"
  },
  {
    "id": "S4-A-15",
    "file": "trinity/phase1_energy/energy_phase_ODEs.py",
    "line": 325,
    "class": "divergence",
    "severity": "S4",
    "claim": "compute_derived_quantities duplicates the entire RHS physics of get_ODE_Edot_pure and has already diverged from it: it omits the EarlyPhaseApproximation override.",
    "evidence": "Lines 332-407 repeat lines 192-279 term for term (feedback lookup, mass clamp, F_grav, solve_R1, effective pressure, P_ext, P_drive max, leak luminosity). The only physics difference is that lines 269-270 (`vd = -1e8`) have no counterpart in the diagnostics path, which also computes no vd or Ed at all.",
    "expected": "One function computing the shared quantities, with the derivative assembly and the diagnostic packing as thin wrappers.",
    "failure_scenario": "Any future edit to one force term must be mirrored by hand; when it is not, the recorded force budget stops describing the integrated trajectory. This has already happened for the EarlyPhaseApproximation branch.",
    "repro": "diff lines 192-279 against 332-407.",
    "confidence": "high"
  },
  {
    "id": "S4-A-16",
    "file": "trinity/phase1_energy/run_energy_phase.py",
    "line": 330,
    "class": "state",
    "severity": "S4",
    "claim": "The simulation-ending event path returns before the phase-boundary reconciliation and the exit logging, unlike every other exit from the phase.",
    "evidence": "Line 329-330: `if event_result.is_simulation_ending: return`. All other exits (normal loop termination, bubble-solve failure break at :183, cooling_balance break at :287, Eb<=0 break at :379) fall through to the reconciliation block at :390-402 and the exit log at :406-407.",
    "expected": "Consistent exit handling, or an explicit statement that the ending event already leaves params complete.",
    "failure_scenario": "On a simulation-ending event, params['R1'], params['Pb'] and params['shell_mass'] keep their previous-segment values and no final save_snapshot() is taken, so the last recorded state differs in kind from every other run's last state.",
    "repro": "Trigger a simulation-ending event and compare the final snapshot count against a normally-terminated run.",
    "confidence": "medium"
  },
  {
    "id": "S4-A-17",
    "file": "trinity/phase1_energy/energy_phase_ODEs.py",
    "line": 273,
    "class": "state",
    "severity": "S4",
    "claim": "The bubble cooling luminosity, radiation force and HII pressure are frozen for the whole segment while the pressures they are balanced against track the integrated state, so the energy equation's loss term does not respond to the energy it drains.",
    "evidence": "L_bubble = snapshot.bubble_LTotal (:273), F_rad = snapshot.F_rad (:261), P_HII = snapshot.P_HII (:251) are all constants of the segment; press_bubble (:226) and L_leak (:277) are recomputed from the live Eb and R2 each call. Segment length is SEGMENT_DURATION = 3e-5 Myr (run_energy_phase.py:55).",
    "expected": "Documented and bounded lagging error, or a shortened segment where the lag matters.",
    "failure_scenario": "Near cooling balance (the very condition tested at run_energy_phase.py:276-287) the frozen L_cool cannot follow a rapidly falling Eb, so the trigger fires at a segment boundary rather than at the true crossing.",
    "repro": "Halve SEGMENT_DURATION and check whether the cooling_balance handoff time shifts.",
    "confidence": "medium"
  },
  {
    "id": "S4-A-18",
    "file": "trinity/phase1_energy/run_energy_phase.py",
    "line": 87,
    "class": "state",
    "severity": "S4",
    "claim": "rCloud is captured once before the loop and used as the loop's stopping radius, so any in-loop update to params['rCloud'] is not seen.",
    "evidence": "Line 87 `rCloud = params['rCloud'].value`, used at line 138. Inside the loop, updateDict(params, bubble_data) at :184 and updateDict(params, shell_data) at :208 may rewrite params entries; the snapshot built at :292 re-reads params['rCloud'] fresh each segment, so the RHS and the loop condition can disagree about the cloud edge.",
    "expected": "Read params['rCloud'].value inside the loop condition, matching what create_ODE_snapshot does.",
    "failure_scenario": "If rCloud is ever recomputed during the phase, the loop keeps integrating past the new cloud edge while the ODE's P_ext branch (rShell >= rCloud) has already switched on the ISM term.",
    "repro": "grep for writes to params['rCloud'] in the bubble/shell update paths.",
    "confidence": "low"
  }
]
```
