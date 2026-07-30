# S5b implicit runner — Lens A (what the code does)

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

**Scope.** `trinity/phase1b_energy_implicit/run_energy_implicit_phase.py` (1461 lines) and
`trinity/phase1b_energy_implicit/__init__.py` (2 blank lines — no code, nothing to report).
All comments/docstrings were blanked in my copy; every statement below is derived from executable
statements only.

**Shared-file declaration.** I consulted the permitted
`trinity/_functions/unit_conversions.py` to fix the code's unit system. It establishes
astro/code units (referred to below as **AU**): `[L]=pc`, `[T]=Myr`, `[M]=Msun`, `[Θ]=K`, and via
the conversion constants:

| quantity | AU dimension | source |
|---|---|---|
| energy | `Msun pc² Myr⁻²` | `E_cgs2au` matches `erg → Msun pc²/Myr²` |
| luminosity | `Msun pc² Myr⁻³` | `L_cgs2au` |
| force | `Msun pc Myr⁻²` | `F_cgs2au` |
| pressure | `Msun pc⁻¹ Myr⁻²` | `Pb_cgs2au = g cm⁻¹ s⁻² → Msun pc⁻¹ Myr⁻²` |
| number density | `pc⁻³` | `ndens_cgs2au = 1/cm2pc³` |
| `G` | `pc³ Msun⁻¹ Myr⁻²` | `G_cgs2au` |
| `k_B` | `Msun pc² Myr⁻² K⁻¹` | `k_B_cgs2au == E_cgs2au` |
| energy→erg | `1 AU = 1.901×10⁴³ erg` | `1/E_cgs2au` |

I did **not** open any file under `trinity/`, `docs/dev/`, or any other agent's output.

---

## 0. Module-level constants (`:111`–`:181`)

All bare literals, with the arithmetic they participate in:

| name | value | unit (from use) | used in |
|---|---|---|---|
| `COOLING_UPDATE_INTERVAL` | `5e-3` | Myr | `:783` `abs(t_prev − t_now) > 5e-3` |
| `DT_SEGMENT_INIT` | `5e-4` | Myr | `:731` initial `dt_segment` |
| `DT_SEGMENT_MIN` | `1e-4` | Myr | `:395`, `:406` floor; `:1217` clamp |
| `DT_SEGMENT_MAX` | `5e-2` | Myr | `:400` cap |
| `MAX_SEGMENTS` | `5000` | — | `:699`, `:758`, `:1410` |
| `NO_ROOT_HANDOFF_STREAK` | `50` | segments | `:865` |
| `FOUR_PI` | `4.0*np.pi` | — | `:535`, `:536`, `:539`, `:1383` |
| `ADAPTIVE_THRESHOLD_DEX` | `0.05` | dex | `:393` |
| `ADAPTIVE_FACTOR` | `10**0.1 ≈ 1.25893` | — | `:395`, `:400`, `:406` |
| `BETADELTA_UNCONVERGED_WARN_STREAK` | `3` | segments | `:345` |
| `BETADELTA_DT_SHRINK_MAX_STREAK` | `10` | segments | `:351`, `:392` |
| `VELOCITY_THRESHOLD_COLLAPSE` | `50.0` | pc/Myr | `:1215` |
| `VELOCITY_THRESHOLD_EXTREME` | `150.0` | pc/Myr | `:1210` |
| `DT_SEGMENT_COLLAPSE` | `5e-5` | Myr | `:1212` |
| `ODE_RTOL` | `1e-6` | — | `solve_ivp` |
| `ODE_ATOL` | `1e-8` | scalar, all 4 states | `solve_ivp` |
| `ODE_MIN_STEP` | `1e-6` | Myr | `solve_ivp` (LSODA only) |
| `ODE_MAX_STEP` | `DT_SEGMENT_MIN/5 = 2e-5` | Myr | `solve_ivp` |
| `ODE_METHOD` | `'LSODA'` | — | `:1070`, `:1076` |
| `ENERGY_HANDOFF_FLOOR` | `1e3` | `Msun pc² Myr⁻²` ≡ **1.901×10⁴⁶ erg** | `:1168` |

Note `DT_SEGMENT_COLLAPSE (5e-5) < DT_SEGMENT_MIN (1e-4)` — the "collapse" clamp sets `dt_segment`
to half the value the adaptive controller treats as its hard floor.

`ADAPTIVE_MONITOR_KEYS` (`:150`–`:167`) is a 35-element list: `R2, v2, Eb, T0, Pb, R1, pdot_SN,
Lmech_SN, pdotdot_total, cool_delta, cool_beta, bubble_mass, bubble_r_Tb, bubble_LTotal,
bubble_L1Bubble, bubble_Lloss, bubble_dMdt, bubble_L2Conduction, bubble_L3Intermediate, shell_mass,
shell_massDot, shell_n0, shell_nMax, shell_thickness, shell_tauKappaRatio, shell_fIonisedDust,
rShell, F_grav, F_ram, F_ram_wind, F_ram_SN, F_ion_in, F_HII, F_rad, F_ISM`.

---

## 1. Pure helpers

### `classify_energy_collapse(Eb)` — `:184`
```
not isfinite(Eb)  -> 'stop'
Eb <= 0           -> 'momentum'
else              -> None
```
Order matters: NaN is caught by the first branch. No units used; `Eb` is `Msun pc² Myr⁻²`.

### `_inflow_frac_thickness(v_arr, r_arr)` — `:212`
```
v, r ∈ ℝⁿ (float)
if v is None or r is None:          return NaN
if n < 2 or len(r) != n:            return 0.0
S  = {i : v_i < 0}
Δr = |r_{n-1} − r_0|
if S = ∅ or Δr ≤ 0:                 return 0.0
f  = ( max_{i∈S} r_i − min_{i∈S} r_i ) / Δr
```
Dimensionless (pc/pc). **This is the bracket span of the inflowing zones, not their measure**: a
single negative-velocity cell gives exactly `0.0` (max = min), and any outflowing cells lying
*between* the innermost and outermost inflowing cells are counted as inflow. `f` can never exceed 1
because `r` is used as its own normaliser only if monotone; if `r` is non-monotone `f > 1` is
possible. Written to `params['v_neg_frac_thick']` at `:898`; never read in this module.

### `evaluate_r1_shadow(R2, rCloud, edot_balance, k_blowout=1.0)` — `:233`
```
blowout = (rCloud is not None) ∧ (rCloud > 0) ∧ (R2 > k_blowout·rCloud)
ebpeak  = (edot_balance is not None) ∧ isfinite(edot_balance) ∧ (edot_balance ≤ 0)
```
`k_blowout` is only ever the default `1.0` (sole call site `:1263` passes three positional args), so
the blowout test is exactly `R2 > rCloud` [pc > pc].

### `parse_transition_triggers(transition_trigger)` — `:252`
`parts = {stripped tokens of str(x).split(',') if non-empty}`; `'r1' → {'blowout','ebpeak'}`;
anything outside `{'cooling_balance','blowout','ebpeak'}` raises `ValueError`. An **empty or
whitespace-only** string yields `frozenset()` — every trigger silently disabled, no error. `None`
becomes the string `'None'` and *does* raise.

### `r1_transition_decision(active, blowout, ebpeak)` — `:275`
`'blowout'` wins over `'ebpeak'` when both fire; else `None`.

### `compute_max_dex_change(before, after, keys)` — `:289`
```
Δ = max over k ∈ keys of:
      skip                      if before[k] is None or after[k] is None
      skip                      if before[k] == 0 or after[k] == 0
      1.0                       if sign(before[k]) ≠ sign(after[k])
      |log10(|after[k]|/|before[k]|)|   otherwise
```
Dimensionless dex; each key's ratio is self-consistent in units. Two structural properties:
* a value that becomes (or ceases to be) **exactly zero** contributes **nothing**, although that is
  the largest possible relative change;
* the `== 0` test at `:315` is **outside** the `try` at `:323`, so an array-valued monitored entry
  raises an uncaught `ValueError: truth value of an array…`. The `except (ValueError,
  ZeroDivisionError)` at `:326` cannot fire for numpy/Python floats (`np.log10` of a positive ratio
  never raises; float division by an already-excluded zero cannot occur), i.e. it is dead.

### `update_unconverged_streak(streak, converged, t, resid)` — `:332`
`converged → 0`; else `streak+1`, with a WARNING at exactly `streak == 3` and another at exactly
`streak == 11` (`BETADELTA_DT_SHRINK_MAX_STREAK + 1`). Unbounded growth thereafter.

### `betadelta_phase_summary(n, nc, nnr)` — `:360`
`pct = 100·nc/n` (0.0 if `n==0`); `clean = (nc==n) ∧ (nnr==0)` — which is vacuously `True` for
`n == 0`.

### `next_dt_segment(dt, Δdex, streak)` — `:376`
```
mitigating := 0 < streak ≤ 10

Δdex > 0.05                     : dt ← max(dt/1.25893, 1e-4)
Δdex ≤ 0.05 ∧ ¬mitigating       : dt ← min(dt·1.25893, 5e-2)
mitigating (applied afterwards) : dt ← max(dt/1.25893, 1e-4)
```
so the four realised maps are `dt/f` , `dt·f` , `dt/f` , `dt/f²` (each floored at `1e-4` / capped at
`5e-2`). Units: Myr throughout.

**Degenerate case.** `max(dt/f, DT_SEGMENT_MIN)` is a *floor-raise* when `dt < DT_SEGMENT_MIN`. The
velocity clamp at `:1212` sets `dt = 5e-5 < 1e-4`; feeding that back in with `Δdex > 0.05` gives
`max(5e-5/1.25893, 1e-4) = 1e-4` — the "shrink" branch **doubles** `dt`.

### `get_monitor_values(params)` — `:412`
Reads the 35 monitor keys, unwrapping `.value` where present, under a blanket
`except Exception: pass`. Returns a plain dict of the referenced objects (not copies).

---

## 2. `ForceProperties` / `compute_forces_pure(R2, mShell, Pb, shell_props, params)` — `:443`–`:561`

**Reads** (params): `G, mCluster, k_B, TShell_ion, rCloud, nISM, PISM(optional), mu_convert,
mu_ion_shell, P_HII, Lbol, c_light, dust_KappaIR`.
**Reads** (shell_props): `rShell, shell_fAbsorbedIon, n_IF, R_IF, isDissolved,
shell_fAbsorbedWeightedTotal, shell_tauKappaRatio`.
**Writes**: nothing to `params` — returns a `ForceProperties`.

**Gravity** `:491`
```
F_grav = G · M_sh · (M_cl + ½ M_sh) / R2²
```
`pc³ Msun⁻¹ Myr⁻² · Msun · Msun / pc² = Msun pc Myr⁻²` ✔ force. Literal `0.5`.

**Ambient / external pressure** `:506`–`:521`
```
if f_absIon < 1:   P_ext = (μ_conv/μ_ion) · n(r_shell) · k_B · T_ion       [else 0]
if r_shell ≥ rCloud:  P_ext ← P_ext + P_ISM · k_B
```
`pc⁻³ · Msun pc² Myr⁻² K⁻¹ · K = Msun pc⁻¹ Myr⁻²` ✔ pressure, provided `μ_conv/μ_ion` is
dimensionless (it is a hydrogen→total particle-number correction, `n_tot = n_H·μ_conv/μ_ion`).
The **second** contribution `P_ISM·k_B` balances only if `PISM` is stored as `n·T` in AU
(`pc⁻³ K`), and it carries **no** `μ_conv/μ_ion` factor while its sibling does.
The whole first term sits inside `try/except Exception: P_ext = 0.0` (`:514`).

**Pressures / interface pass-through** `:527`–`:532`
```
n_IF, R_IF  = shell_props.n_IF, shell_props.R_IF
P_HII       = params['P_HII']
P_drive     = max(Pb, P_HII)
```

**Forces** `:535`–`:547`
```
F_ion_in = P_ext · 4π R2²                                   [Msun pc Myr⁻²]  ✔
F_HII    = 4π R2² · P_HII                                   ✔
F_ram    = Pb · 4π R2²                                      ✔   (a bubble-pressure force,
                                                                  not a ram pressure)
F_rad    = 0                              if shell dissolved
F_rad    = f_absWtot · L_bol/c · (1 + τ_κ · κ_IR)           ✔ (Msun pc² Myr⁻³)/(pc/Myr)
```
`τ_κ·κ_IR` must be dimensionless. Literal `1.0` in `(1.0 + …)`, `1.0` in `FABSi < 1.0`.

`P_ext` is evaluated at **`rShell`** but multiplied by the area of **`R2`** (`:535`).

`nISM` (`:498`) is read and never used. `P_ram` is passed **hard-coded `0.0`** (`:559`).

---

## 3. `get_ODE_implicit_pure(t, y, snapshot, params_for_feedback, Ed_from_beta, Td_from_delta)` — `:586`

```
y = (R2, v2, Eb, T0)                 [pc, pc/Myr, Msun pc² Myr⁻², K]

(rd, vd, _discarded) = get_ODE_Edot_pure(t, [R2, v2, Eb], snapshot, params)

dy/dt = ( rd , vd , Ed_from_beta , Td_from_delta )
```

So the system actually integrated on `[t₀, t₀+Δt]` is

```
dR2/dt = f_R(t, R2, v2, Eb ; snapshot(t₀))
dv2/dt = f_v(t, R2, v2, Eb ; snapshot(t₀))
dEb/dt = Ed  = const                          ⇒ Eb(t) = Eb₀ + Ed·(t − t₀)      exactly linear
dT0/dt = Td  = const                          ⇒ T0(t) = T0₀ + Td·(t − t₀)      exactly linear
```
with (`:992`–`:993`)
```
Ed = cool_beta_to_Ebdot_pure(β, Pb(t₀ᵃᶠᵗᵉʳ), t₀, R1, R2₀, v2₀, Eb₀, ṗ_tot(t₀), p̈_tot(t₀))
Td = delta2dTdt_pure(t₀, T0₀, δ)
```
Units require `[Ed] = Msun pc² Myr⁻³` and `[Td] = K Myr⁻¹`; the integrator's independent variable is
Myr throughout (`t_span` from `t_now`/`stop_t`), so time units are consistent.

`dydt_energy[2]` — the ODE's *own* `Ėb` — is computed on **every** RHS evaluation and thrown away
(`:620`–`:624` take only indices 0 and 1). `t` is forwarded but `snapshot` is frozen at `t₀`;
`params_for_feedback` is the **live mutable** params object, not a frozen copy.

---

## 4. `run_phase_energy(params)` — `:631`–`:1460`

### 4.1 Entry (`:653`–`:696`)

```
v2_ODE   = params['v2']
v2_alpha = cool_alpha · R2 / t_now                                  [pc/Myr]
ratio    = v2_alpha / v2_ODE   (inf if v2_ODE == 0)
params['cool_alpha'] ← t_now · v2 / R2                              [dimensionless]   :662
tmin = params['t_now'] ;  tmax = params['stop_t']                   [Myr]
```
`t_now == 0` would raise `ZeroDivisionError` at `:654`.

**Skip branch** `:670`–`:690`: if `tmin ≥ tmax`, write `SimulationEndCode = STOPPING_TIME`,
`EndSimulationDirectly = True`, and return single-element arrays with
`termination_reason = "skipped_past_stop_t"`. No snapshot is saved on this path.

`R2, v2, Eb, T0` are then pulled into locals (`:693`–`:696`). These locals — not `params` — are the
authoritative state for the loop.

`n_estimate = min(int(200·log10(tmax/tmin)), 5000)` (`:699`) is **never used**.

### 4.2 Loop preamble (`:712`–`:752`)

`t_now=tmin`, `segment_count=0`, `termination_reason=None`, `beta/delta` from
`cool_beta`/`cool_delta`, `R2_prev=R2`, shadow accumulators, `active_triggers`,
`dt_segment = 5e-4`, three β-δ counters, `no_root_streak = 0`.

`ode_events, cooling_balance_factory = build_implicit_phase_events(params)` (`:752`) — built **once,
before the loop**, from a `params` that is then mutated on every segment.
`cooling_balance_factory` is **never used**.

### 4.3 Segment body — order of operations

`while t_now <= tmax and segment_count < MAX_SEGMENTS:` (`:758`)

1. `:759` `segment_count += 1`.
2. `:764`–`:775` **stop_at_rCloud**: if `nSnap is not None ∧ R2 > rCloud ∧
   params['_snapshots_after_rCloud'] ≥ nSnap` → `termination_reason="stop_at_rCloud"`,
   `SimulationEndCode = RCLOUD_BOUNDARY`, `EndSimulationDirectly = True`, `break`. Uses the
   **carried-over** `R2` from the previous segment.
3. `:783`–`:788` **cooling-table refresh**, gated on `|t_prevCoolingUpdate − t_now| > 5e-3 Myr`;
   writes `cStruc_cooling_nonCIE`, `cStruc_heating_nonCIE`, `cStruc_net_nonCIE_interpolation`,
   `t_previousCoolingUpdate`. Because `dt_segment ≤ 5e-2 Myr`, the tables can be up to
   `5e-3 Myr` stale (≈10 segments at `dt=5e-4`).
4. `:793`–`:798` push locals into params: `t_now, R2, v2, Eb, T0`, and
   `cool_alpha ← t_now·v2/R2`.
5. `:803`–`:804` `feedback = get_current_sps_feedback(t_now, params)`; `updateDict(params, feedback)`.
6. `:813`–`:819` `params['bubble_Leak'] ← get_leak_luminosity(coverFraction, params['R2'],
   params['Pb'], params['c_sound'], gamma_adia)`.
   **`params['Pb']` and `params['c_sound']` at this point are the *previous* segment's values** —
   `Pb` is not rewritten until `:939` and `c_sound` not until `:944`. So the leak mixes a fresh `R2`
   with a one-segment-old `Pb`/`c_sound`.
7. `:826`–`:838` `betadelta_result = solve_betadelta_pure(cool_beta, cool_delta, params)`;
   `beta, delta ← result`; counters incremented. **Also executes against the stale `params['Pb']`.**
8. `:847`–`:882` **no-physical-root handling**: increments `no_root_count`/`no_root_streak`, logs
   (WARNING on the first of a streak, DEBUG after), and at `no_root_streak ≥ 50` sets
   `termination_reason = "no_physical_root_handoff"` and `break`. Otherwise the streak resets.
   Note the loop does **not** skip the segment — the `beta`/`delta` returned by the failed solve are
   used unchanged in steps 9 and 15.
   Diagnostic literal `1e-300` guards `Lloss/Lgain` at `:872` and `:906`.
9. `:885`–`:886` `params['cool_beta'] ← beta`, `params['cool_delta'] ← delta`.
10. `:889`–`:907` if `betadelta_result.bubble_properties is not None`: `updateDict(params, …)`,
    `params['v_neg_frac_thick'] ← _inflow_frac_thickness(bubble_v_arr, bubble_r_arr)`, debug log.
    If it *is* `None`, `v_neg_frac_thick` silently keeps the previous segment's value.
11. `:910`–`:930` residual bookkeeping: `betadelta_converged`, `betadelta_total_residual`,
    `residual_deltaT`, `residual_betaEdot`, and **conditionally** (`if … is not None`)
    `residual_Edot1_guess`, `residual_Edot2_guess`, `residual_T1_guess`, `residual_T2_guess`,
    `bubble_Lgain`, `bubble_Lloss` — each conditional write leaves a stale prior value when the
    corresponding field is `None`.
12. `:935`–`:939` `R1, Pb = compute_R1_Pb(R2, Eb, feedback.Lmech_total, feedback.v_mech_total,
    gamma_adia)`; written to `params['R1']`, `params['Pb']`. **This is the first and only refresh of
    `Pb` in the segment — after steps 6 and 7 have already consumed the old one.**
13. `:943`–`:944` `T_avg = params['bubble_Tavg'] or 1e6`; `c_sound ← get_soundspeed(T_avg, params)`.
    The `or`-style fallback is `x if x else 1e6`, so a legitimately computed `0.0` becomes `1e6 K`;
    `NaN` is truthy and passes straight through.
14. `:953`–`:970` **shell mass with a one-way ratchet**:
    ```
    if isCollapse:   M_sh = M_sh^prev ;  Ṁ_sh = 0
    else:            (M_new, Ṁ) = get_mass_profile(R2, params, return_mdot=True, rdot=v2)
                     if M_sh^prev > 0 ∧ M_new < M_sh^prev:  M_sh = M_sh^prev ; Ṁ_sh = 0
                     else:                                  M_sh = M_new     ; Ṁ_sh = Ṁ
    ```
    i.e. `M_sh` is monotone non-decreasing, and whenever the ratchet engages the computed `Ṁ_sh` is
    silently replaced by `0`.
15. `:975`–`:986` `shell_props = shell_structure_pure(params)`; `updateDict`; then
    ```
    P_HII = (μ_conv/μ_ion) · n_IF_Str · k_B · T_ion   if include_PHII ∧ n_IF_Str > 0 else 0
    F_HII = 4.0·π·R2²·P_HII
    ```
    `pc⁻³·Msun pc² Myr⁻² K⁻¹·K·pc² = Msun pc Myr⁻²` ✔. The `F_HII` written at `:986` is
    recomputed identically inside `compute_forces_pure` (`:536`) and overwritten at `:998`.
16. `:992`–`:993` `Ed`, `Td` (see §3) — using the **new** `Pb` from step 12 together with the `β`
    solved in step 7 against the **old** `Pb`.
17. `:995`–`:1009` `force_props = compute_forces_pure(R2, mShell, Pb, shell_props, params)`; writes
    `F_grav, F_ion_in, F_HII, F_ram, F_rad, n_IF, R_IF, P_HII, P_drive, P_ram(=0), press_HII_in`,
    plus `F_ram_wind ← feedback.pdot_W` and `F_ram_SN ← feedback.pdot_SN`.
    `F_ISM` (in the monitor list) is never written anywhere in this module.
18. `:1016`–`:1026` `save_snapshot()`; if a snapshot was actually appended and `R2 > rCloud`,
    `_snapshots_after_rCloud += 1`.
19. `:1029`–`:1035` append `(t_now, R2, v2, Eb, T0, beta, delta)` to the seven result lists.
20. `:1040`–`:1046` if `t_now ≥ tmax` → `"reached_tmax"`, `STOPPING_TIME`,
    `EndSimulationDirectly`, `break`. (The `tmax is not None` guard can never be False — `:670`
    already compared `tmin >= tmax`, which would have raised `TypeError` on `None`.)
21. `:1051` `snapshot = create_ODE_snapshot(params, shell_props)` — frozen for the whole segment.
22. `:1054` `values_before = get_monitor_values(params)`.

### 4.4 The integration call (`:1056`–`:1090`)

```
t_end   = min(t_now + dt_segment, tmax)
t_span  = (t_now, t_end)                                       [Myr]
y0      = [R2, v2, Eb, T0]

solve_ivp(fun    = λ t,y: get_ODE_implicit_pure(t, y, snapshot, params, Ed, Td),
          t_span = t_span,
          y0     = y0,
          method = 'LSODA',
          rtol   = 1e-6,
          atol   = 1e-8,          ← scalar, applied to all four components
          max_step = 2e-5,        ← DT_SEGMENT_MIN/5
          min_step = 1e-6,        ← LSODA only
          events = ode_events)
```

Observations that bear on the maths:

* `max_step = 2e-5 Myr` while `dt_segment` may reach `5e-2 Myr` ⇒ **≥ 2500 internal steps per
  segment**, and the internal step is capped far below any accuracy requirement. The adaptive
  `dt_segment` controller therefore does **not** control integration accuracy; it only controls how
  often `β`/`δ`/`Pb`/shell/forces are re-solved.
* `atol = 1e-8` is a single scalar shared by `R2 ~ 10¹ pc`, `v2 ~ 10¹–10³ pc/Myr`,
  `Eb ~ 10⁵–10¹⁰ AU`, `T0 ~ 10⁶–10⁷ K`. For the large components `rtol` dominates; for a component
  passing through zero (`v2`, or `Eb` at handoff) LSODA must satisfy an absolute tolerance of `1e-8`
  and will demand `h < min_step = 1e-6 Myr` → failure.
* **Failure handling has no retry.** `except Exception` (`:1080`) → `termination_reason =
  f"solver_error: {e}"`, `break`; `not sol.success or len(sol.t) == 0` (`:1085`) →
  `termination_reason = f"solver_failed: {sol.message}"`, `break`. Neither path sets
  `SimulationEndCode`, `SimulationEndReason`, or `EndSimulationDirectly`, and neither shrinks
  `dt_segment` and re-attempts.

### 4.5 Event handling (`:1095`–`:1119`)

```
event_result = check_event_termination(sol, ode_events)
if event_result.triggered:
    termination_reason = event_result.reason_code
    (R2, v2, Eb, T0)  ← event_result.y[0:4]
    t_now             ← event_result.t
    append (t_now, R2, v2, Eb, T0, beta, delta)          # β,δ are the segment-start values ✔
    apply_event_result(params, event_result, t_now, event_result.y,
                       state_keys=['R2','v2','Eb','T0'])  # return value discarded
    break
```
`ode_events` is handed to `solve_ivp` directly, so scipy will honour any `.terminal`/`.direction`
attributes the events carry. But the *phase* is terminated by this module's own
`check_event_termination`, and the `break` is unconditional on `triggered` — from this file alone
nothing distinguishes a terminal from a non-terminal event. (Neither `build_implicit_phase_events`
nor `check_event_termination` is in my slice; this is an observation, not a verdict.)

This path **bypasses** the `classify_energy_collapse` check at `:1148`, so an event that fires at
`Eb ≤ 0` or `Eb` non-finite propagates that value into `params` (via `apply_event_result`) with no
`ENERGY_HANDOFF_FLOOR` and no `ENERGY_COLLAPSED` code.

### 4.6 Accepting the step and the collapse test (`:1124`–`:1175`)

```
R2,v2,Eb,T0 ← sol.y[:, −1] ;  t_now ← sol.t[−1]        # locals
params['t_now','R2','v2','Eb','T0'] ← the same

c = classify_energy_collapse(Eb)
c == 'stop'      : EndSimulationDirectly=True, ENERGY_COLLAPSED,
                   termination_reason="energy_collapsed", break     # Eb (NaN/inf) stays in params
c == 'momentum'  : Eb ← 1e3 ; params['Eb'] ← 1e3 ;
                   termination_reason="energy_to_momentum", break
```
The `'momentum'` clamp replaces whatever the integrator produced (which, given `Eb(t)` is linear
with a frozen negative `Ed`, can be arbitrarily negative) with a fixed
`1e3 Msun pc² Myr⁻² = 1.901×10⁴⁶ erg`.

Because `dEb/dt` is frozen over the segment, an `Eb` zero-crossing **inside** a segment is not
detected here — the RHS keeps being evaluated with `Eb < 0` for the rest of the segment, and the
crossing is only classified once the segment completes.

### 4.7 Post-step bookkeeping (`:1179`–`:1219`)

* `:1179` `terminal_prints.heartbeat(params, "1b implicit", segment_count, tmin, tmax)`.
* `:1184`–`:1196` second shell-mass update at the **new** `R2` with the same ratchet, writing only
  `shell_mass` — `shell_massDot` is **not** refreshed, so it remains the pre-step value.
  `get_mass_profile(R2, …)` is called here and then again with the *same* `R2` at `:962` of the next
  iteration (differing only in `return_mdot`), so the identical mass profile is evaluated twice per
  segment and the next iteration's ratchet at `:964` is a guaranteed no-op.
* `:1198`–`:1202` `values_after`; `max_dex_change`; `dt_segment = next_dt_segment(...)`.
* `:1208`–`:1219` velocity clamps, **only for `v2 < 0`**:
  ```
  |v2| > 150 pc/Myr : dt_segment ← 5e-5
  |v2| >  50 pc/Myr : dt_segment ← min(dt_segment, 1e-4)
  ```
  Large *positive* velocities receive no clamp.

**Key structural fact about the adaptive controller.** `values_before` is captured at `:1054`, i.e.
*after* every physics write of the segment (steps 5–17), and `values_after` at `:1198`. Between the
two captures the only params entries written are `t_now, R2, v2, Eb, T0` (`:1134`–`:1138`),
`Eb` again on the momentum branch, and `shell_mass` (`:1196`). Therefore **30 of the 35
`ADAPTIVE_MONITOR_KEYS` are byte-identical between the two dicts by construction** and contribute
exactly `0.0` dex on every segment. The controller effectively monitors
`{R2, v2, Eb, T0, shell_mass}` only. (`t_now` is not a monitor key.)

### 4.8 Lgain / Lloss and the termination tests (`:1230`–`:1333`)

```
feedback_post = get_current_sps_feedback(t_now_new, params)      # NOT updateDict'd into params
Lgain = feedback_post.Lmech_total                                [Msun pc² Myr⁻³]

if bubble_props is not None:
    Lcool = bubble_props.bubble_LTotal                # pre-step value
    leak  = params['bubble_Leak']                     # computed at :813 from stale Pb, c_sound
    Lloss = effective_Lloss_from_params(params, Lcool, leak, Lgain)
else:
    Lcool = params['bubble_Lloss']                    # itself already an *effective* loss (:930)
    Lloss = effective_Lloss_from_params(params, Lcool, 0.0, Lgain)
```
The two branches are not siblings: one passes a **raw** cooling luminosity with the real leak, the
other passes an **already-effective** loss (possibly from an earlier segment) with `leak = 0`.

```
threshold = params['phaseSwitch_LlossLgain'].value  (fallback literal 0.05)
```
The presence test is `if phase_switch_threshold and hasattr(..., 'value')` — object truthiness, not
`is not None`. The same idiom appears at `:500`, `:955`, `:1186`, `:1241`, `:1246`, `:1317`.

**R1 shadow** `:1261`–`:1280`: uses the **post-step** `R2` together with
`betadelta_result.Edot_from_balance` from the **pre-step** solve. Appends a row with
`R2_over_rCloud = R2/rCloud` (NaN-guarded) and
`cooling_ratio = (Lgain − Lloss)/Lgain` (NaN if `Lgain ≤ 0`).

**Termination tests, in order:**

| line | condition | reason | end code set? |
|---|---|---|---|
| `:1288` | `blowout` ∈ triggers ∧ `R2 > rCloud`, else `ebpeak` ∈ triggers ∧ `Ėb_bal ≤ 0` | `'blowout'` / `'ebpeak'` | **no** |
| `:1296` | `'cooling_balance' ∈ triggers ∧ Lgain > 0 ∧ (Lgain−Lloss)/Lgain < θ` | `cooling_balance` | **no** |
| `:1302` | `v2 < 0 ∧ R2 < R2_prev` → `params['isCollapse'] = True` (latch, never cleared) | — | — |
| `:1309` | `t_now > tmax` | `reached_tmax` | yes |
| `:1316` | `isCollapse ∧ R2 < coll_r` | `small_radius` / `SHELL_COLLAPSED` | yes |
| `:1327` | `stop_r is not None ∧ R2 > stop_r` | `large_radius` / `LARGE_RADIUS` | yes |

The `cooling_balance` log line reads *"Lloss/Lgain ratio below {threshold}"*, but the condition
`(Lgain − Lloss)/Lgain < θ` is equivalent to `Lloss/Lgain > 1 − θ` — the message states the
opposite of the test.

Because `:1309` already breaks on `t_now > tmax`, and `:1040` breaks on `t_now ≥ tmax` at the top of
the next iteration, the `while` condition `t_now <= tmax` can **never** be the reason the loop
exits. Only `segment_count < MAX_SEGMENTS` can. Consequently `termination_reason = "unknown"`
(`:1410`) and the `logger.warning if termination_reason == "unknown"` selector (`:1414`) are
unreachable.

### 4.9 Tail (`:1341`–`:1460`)

* `:1341` `if len(t_results) == 0 or t_now != t_results[-1]` — float equality; appends the final
  state to all seven lists, keeping them the same length on every exit path I traced
  (`reached_tmax`, `solver_failed`, event, `stop_at_rCloud`, `no_physical_root_handoff`,
  `energy_to_momentum`, `energy_collapsed`).
  On the `no_physical_root_handoff` break the appended `beta`/`delta` are the *current* segment's
  (non-physical) values, whereas `params['cool_beta']/['cool_delta']` still hold the previous
  segment's — `:885`–`:886` are downstream of that `break`.
* `:1365`–`:1396` **phase-boundary reconciliation**, run whenever
  `termination_reason != "energy_collapsed"` (note `termination_reason` may still be `None` here —
  the default is not assigned until `:1408`). It recomputes at the final `(t_now, R2, Eb)`:
  `feedback_final` (updateDict'd), `R1`, `Pb`, `shell_props`, `P_HII`, `F_HII`, then
  `compute_forces_pure` and writes `F_grav, F_ion_in, F_HII, F_ram, F_rad, P_HII, P_drive, P_ram`,
  then `save_snapshot()`.
  It does **not** refresh `n_IF`, `R_IF`, `press_HII_in`, `F_ram_wind`, `F_ram_SN`, `shell_mass`,
  `shell_massDot`, `c_sound`, `bubble_Leak`, `cool_beta`, `cool_delta` — so the final snapshot mixes
  freshly reconciled quantities with one-segment-old ones, including `F_ion_in` (fresh) alongside
  `press_HII_in` (stale) which is the very pressure `F_ion_in` is derived from.
  The whole block is inside `try/except Exception → logger.warning`, and `save_snapshot()` is the
  **last** statement in the `try`: any failure loses the final snapshot with only a warning.
* `:1397`–`:1402` the `energy_collapsed` path saves a snapshot containing the non-finite `Eb`.
* `:1427`–`:1429` β-δ summary; `:1435`–`:1448` writes `shadow_R1_1b.csv` (mode `'w'`, overwrite)
  under a blanket `except Exception`.
* `:1450` returns `ImplicitPhaseResults(t, R2, v2, Eb, T0, beta, delta, termination_reason,
  final_time=t_now)`.

### 4.10 `stop_at_rCloud_nSnap` counting

The counter is incremented at `:1026` only for snapshots written inside the loop; the reconciliation
`save_snapshot()` at `:1394` is not counted. So the output ends with `nSnap + 1` snapshots beyond
`rCloud`.

### 4.11 Aliasing / in-place mutation

I found **no** in-place array mutation with a live second reference. `y0` is a fresh array per
segment; `sol.y[:, -1]` is copied to floats immediately; `event_result.y` is read into floats before
being handed to `apply_event_result`. The two places where a reference (not a copy) is retained are
`values_before` (a dict of the *objects* held in `params[k].value`) and `bubble_props`, reused at
`:1238`–`:1239` after `updateDict(params, bubble_props)`. Both are benign for scalar payloads, but
`values_before` would silently track any later in-place mutation of an array-valued monitor key.

### 4.12 Unused imports / symbols

`scipy.optimize` (`:59`), `Dict/Optional/Tuple` (`:61`), `trinity._functions.unit_conversions as cvt`
(`:66`), `ODEResult` (`:78`), `compute_derived_quantities` (`:79`), `BetaDeltaResult` (`:87`) are all
imported and never referenced. `ShellProperties` is used only as a type annotation.

---

## 5. Dimensional summary

Every arithmetic expression in this module balances in AU, with two caveats:
1. `P_ext += PISM * k_B` (`:521`) balances only if `PISM` is stored as `n·T` in `pc⁻³ K`; if it is
   stored as a pressure it is off by `k_B`, and it is missing the `μ_conv/μ_ion` factor its sibling
   term at `:513` carries.
2. `F_ion_in = P_ext · 4πR2²` (`:535`) evaluates the pressure at `rShell` and the area at `R2` —
   dimensionally fine, geometrically mismatched.

No other term is dimensionally unbalanced.

---

```json
[
  {
    "id": "S5b-A-01",
    "file": "trinity/phase1b_energy_implicit/run_energy_implicit_phase.py",
    "line": 813,
    "class": "state",
    "severity": "S2",
    "claim": "bubble_Leak is computed from the PREVIOUS segment's Pb and c_sound while using the CURRENT segment's R2.",
    "evidence": "get_leak_luminosity is called at :813-819 with params['Pb'] and params['c_sound']. params['Pb'] is not written until :939 (compute_R1_Pb) and params['c_sound'] not until :944. params['R2'] was refreshed at :794. So within one segment the leak mixes R2(t_k) with Pb(t_{k-1}) and c_sound(t_{k-1}).",
    "expected": "All inputs to the leak luminosity should share one vintage: either compute R1/Pb/c_sound before :813, or pass the pre-step R2.",
    "failure_scenario": "Any run where Pb changes appreciably over dt_segment (e.g. param/simple_cluster.param during the early rapid-expansion segments where dt_segment=5e-4 Myr and Pb falls fast): bubble_Leak is systematically biased high by roughly Pb(t_{k-1})/Pb(t_k), which propagates into Lloss (:1243) and hence into the cooling_balance termination time.",
    "repro": "Instrument the loop to record Pb at :813 and at :939 for each segment and assert they are equal; they differ by one segment on every segment after the first.",
    "confidence": "high"
  },
  {
    "id": "S5b-A-02",
    "file": "trinity/phase1b_energy_implicit/run_energy_implicit_phase.py",
    "line": 826,
    "class": "state",
    "severity": "S2",
    "claim": "beta is solved against the previous segment's Pb, then Ed is evaluated with the new Pb — the two are inconsistent by construction.",
    "evidence": "solve_betadelta_pure(params['cool_beta'], params['cool_delta'], params) at :826 reads a params in which 'Pb' still holds the value written at :939 of the previous segment. compute_R1_Pb overwrites params['Pb'] at :936-939. Then :992 calls cool_beta_to_Ebdot_pure(beta, Pb, ...) with the NEW Pb and the OLD-Pb beta.",
    "expected": "compute_R1_Pb should run before the beta-delta solve, so the residual that defines beta and the Ed derived from beta use the same Pb.",
    "failure_scenario": "Stiff/edge configs where Pb varies by more than the beta-delta residual tolerance across one dt_segment (docs/dev/performance f1edge_hidens-style high-density runs): the imposed dEb/dt is not the derivative the implicit closure actually converged, so Eb drifts off the closure by O(dPb/dt * dt) per segment, accumulating over up to 5000 segments.",
    "repro": "Log Pb at :826 and at :939 per segment and assert equality; or recompute cool_beta_to_Ebdot_pure with the :826-vintage Pb and compare Ed.",
    "confidence": "high"
  },
  {
    "id": "S5b-A-03",
    "file": "trinity/phase1b_energy_implicit/run_energy_implicit_phase.py",
    "line": 1054,
    "class": "deadcode",
    "severity": "S4",
    "claim": "30 of the 35 ADAPTIVE_MONITOR_KEYS are provably identical between values_before and values_after, so the dex-based step controller only ever sees R2, v2, Eb, T0, shell_mass.",
    "evidence": "values_before is taken at :1054, after every physics write of the segment (:785-:1009). values_after is taken at :1198. Between them the only params writes are :1134-1138 (t_now,R2,v2,Eb,T0), :1169 (Eb on the momentum branch, which breaks) and :1196 (shell_mass). Pb, R1, cool_beta, cool_delta, all bubble_*, all shell_* except shell_mass, rShell, and all F_* are unchanged. F_ISM is never written at all in this module.",
    "expected": "Either capture values_before at the top of the segment (before the beta-delta/shell/force block) so the monitored physics quantities can actually move, or trim the list to the five keys that can.",
    "failure_scenario": "No wrong number, but the controller never reacts to a bubble/shell/force excursion — e.g. bubble_dMdt flipping sign, which compute_max_dex_change would score as 1.0 dex, cannot influence dt_segment at all.",
    "repro": "Assert compute_max_dex_change(values_before, values_after, [k for k in ADAPTIVE_MONITOR_KEYS if k not in ('R2','v2','Eb','T0','shell_mass')]) == 0.0 on every segment of param/simple_cluster.param — it holds.",
    "confidence": "high"
  },
  {
    "id": "S5b-A-04",
    "file": "trinity/phase1b_energy_implicit/run_energy_implicit_phase.py",
    "line": 1085,
    "class": "silent-failure",
    "severity": "S1",
    "claim": "A solve_ivp failure ends the phase without setting SimulationEndCode, SimulationEndReason or EndSimulationDirectly, and without any dt reduction/retry.",
    "evidence": ":1080-1083 (except Exception) and :1085-1090 (not sol.success or len(sol.t)==0) both only set a free-text termination_reason and break. Contrast :768-774, :1041-1044, :1152-1157, :1311-1313, :1321-1323, :1330-1332 which all set SimulationEndCode + EndSimulationDirectly.",
    "expected": "A failed integration should either retry with a smaller dt_segment (dt_segment/ADAPTIVE_FACTOR down to DT_SEGMENT_MIN) or set an explicit failure end-code so downstream output records that the run did not complete physically.",
    "failure_scenario": "LSODA hits min_step=1e-6 Myr trying to satisfy atol=1e-8 as v2 crosses zero during a collapse. The phase returns termination_reason='solver_failed: Required step size is less than spacing between numbers.' with SimulationEndCode still holding whatever a prior phase set, and the caller hands off to the next phase as though the energy phase ended normally, from the last successful segment's state.",
    "repro": "Force a failure (e.g. temporarily set ODE_MIN_STEP above ODE_MAX_STEP) and assert params['SimulationEndCode'].value reflects a failure; it does not.",
    "confidence": "high"
  },
  {
    "id": "S5b-A-05",
    "file": "trinity/phase1b_energy_implicit/run_energy_implicit_phase.py",
    "line": 624,
    "class": "numerical",
    "severity": "S2",
    "claim": "dEb/dt is frozen for the whole segment, so Eb evolves exactly linearly and can cross zero and go negative mid-segment; the collapse test only runs at the segment boundary.",
    "evidence": "get_ODE_implicit_pure returns [rd, vd, Ed_from_beta, Td_from_delta] with Ed/Td computed once at :992-993 and held constant; hence Eb(t)=Eb0+Ed*(t-t0) exactly. Eb is nevertheless passed back into get_ODE_Edot_pure as y[2] (:617-618), so f_R/f_v are evaluated at Eb<0 for the remainder of the segment. classify_energy_collapse is only called at :1148, after solve_ivp returns.",
    "expected": "Either a terminal event on Eb (built into ode_events) or a mid-segment guard, so the Eb<=0 handoff at :1164 is taken rather than whatever f_R/f_v produce for negative Eb.",
    "failure_scenario": "A segment starting at Eb=1e4 AU with Ed=-1e8 AU/Myr and dt_segment=5e-4 Myr crosses zero after 1e-4 Myr; the remaining 4e-4 Myr is integrated with Eb<0. If f_v contains a sqrt or a Pb ~ Eb/(R2^3-R1^3) term, the RHS returns NaN and the run takes the S5b-A-04 solver_failed path instead of the clean 'energy_to_momentum' handoff.",
    "repro": "Log min(Eb) over sol.y[2] versus Eb at sol.t[-1]; a segment exists where the minimum is negative but the endpoint is positive, or where the endpoint is NaN.",
    "confidence": "medium"
  },
  {
    "id": "S5b-A-06",
    "file": "trinity/phase1b_energy_implicit/run_energy_implicit_phase.py",
    "line": 1137,
    "class": "state",
    "severity": "S2",
    "claim": "A non-finite Eb is written into params and then persisted: the 'energy_collapsed' path saves a snapshot containing NaN/inf.",
    "evidence": "params['Eb'].value = Eb at :1137 happens before classify_energy_collapse at :1148. On the 'stop' branch (:1149-1163) Eb is never repaired, and :1365 routes 'energy_collapsed' to the else branch at :1397-1402 which calls params.save_snapshot() unconditionally. The same non-finite Eb is also appended to Eb_results at :1345.",
    "expected": "Either repair Eb (as the momentum branch does with ENERGY_HANDOFF_FLOOR) or skip the snapshot on the non-finite path.",
    "failure_scenario": "Any run whose bubble energy goes non-finite writes a final dictionary.jsonl row with Eb=NaN, which then poisons any downstream reader that averages or interpolates the energy trajectory.",
    "repro": "Assert np.isfinite(ImplicitPhaseResults.Eb).all() for a run that terminates with 'energy_collapsed'.",
    "confidence": "high"
  },
  {
    "id": "S5b-A-07",
    "file": "trinity/phase1b_energy_implicit/run_energy_implicit_phase.py",
    "line": 1168,
    "class": "coefficient",
    "severity": "S2",
    "claim": "ENERGY_HANDOFF_FLOOR silently replaces a computed negative Eb with 1e3 AU = 1.90e46 erg — not an epsilon.",
    "evidence": "Eb = ENERGY_HANDOFF_FLOOR (=1e3, :181) at :1168, params['Eb'] at :1169. Using E_cgs2au=5.260183968837699e-44 from unit_conversions.py, 1 AU energy = 1.901e43 erg, so 1e3 AU = 1.901e46 erg. The replaced value is unbounded below: with Ed frozen over the segment, Eb can arrive at an arbitrarily large negative number.",
    "expected": "A handoff floor should be documented in the units it is expressed in and be small compared with the bubble energies of the configs run; 1.9e46 erg is ~0.02 SN.",
    "failure_scenario": "On the energy_to_momentum handoff, the post-loop reconciliation (:1370) computes Pb_f from Eb=1e3 rather than from the actual (negative/zero) energy, so the momentum phase starts from a bubble pressure derived from an injected 1.9e46 erg. It also silences how far below zero the integrator went, hiding the size of the overshoot.",
    "repro": "Record Eb at :1128 and at :1169 for a run that ends in 'energy_to_momentum' and report the ratio; the discarded magnitude is not logged anywhere.",
    "confidence": "high"
  },
  {
    "id": "S5b-A-08",
    "file": "trinity/phase1b_energy_implicit/run_energy_implicit_phase.py",
    "line": 1096,
    "class": "state",
    "severity": "S2",
    "claim": "The event-termination path bypasses classify_energy_collapse entirely, so an event firing at Eb<=0 or non-finite Eb writes that value straight into params with no floor and no ENERGY_COLLAPSED code.",
    "evidence": "The event block at :1096-1119 ends in `break` at :1119, before the collapse classification at :1148. It assigns Eb = float(event_result.y[2]) at :1103 and calls apply_event_result(..., state_keys=['R2','v2','Eb','T0']) at :1117.",
    "expected": "The collapse classification should be applied to the state accepted on the event path as well as the normal path.",
    "failure_scenario": "An event (e.g. a shell-collapse or radius event) fires in the same segment in which Eb goes negative. The phase exits with termination_reason = that event's reason_code, Eb<0 in params, and the reconciliation at :1370 calls compute_R1_Pb(R2, Eb<0, ...) producing a negative or NaN Pb which is written to params['Pb'] and snapshotted.",
    "repro": "Assert classify_energy_collapse(params['Eb'].value) is None at phase exit for every termination_reason that came from an event.",
    "confidence": "high"
  },
  {
    "id": "S5b-A-09",
    "file": "trinity/phase1b_energy_implicit/run_energy_implicit_phase.py",
    "line": 395,
    "class": "numerical",
    "severity": "S4",
    "claim": "next_dt_segment's shrink branch increases dt when dt is already below DT_SEGMENT_MIN; DT_SEGMENT_COLLAPSE (5e-5) is half DT_SEGMENT_MIN (1e-4).",
    "evidence": "max(dt/ADAPTIVE_FACTOR, DT_SEGMENT_MIN) at :395 and :406. With dt = DT_SEGMENT_COLLAPSE = 5e-5 (set at :1212) and max_dex_change > 0.05, this evaluates max(5e-5/1.25893, 1e-4) = max(3.971e-5, 1e-4) = 1e-4 — a 2x increase on the branch whose purpose is to shrink.",
    "expected": "Either the floor should be min(DT_SEGMENT_MIN, dt) so a shrink never grows dt, or DT_SEGMENT_COLLAPSE should not be below DT_SEGMENT_MIN.",
    "failure_scenario": "A collapsing shell with v2 < -150 pc/Myr is clamped to dt=5e-5. On the first segment where |v2| falls back below 50 pc/Myr (so :1208-1219 no longer clamps) with a large dex change, dt is doubled to 1e-4 exactly when the controller asked for a reduction.",
    "repro": "assert next_dt_segment(5e-5, 0.5, 0) <= 5e-5 — it returns 1e-4.",
    "confidence": "high"
  },
  {
    "id": "S5b-A-10",
    "file": "trinity/phase1b_energy_implicit/run_energy_implicit_phase.py",
    "line": 1296,
    "class": "citation",
    "severity": "S3",
    "claim": "The cooling_balance log message states the inverse of the condition it reports.",
    "evidence": "The test at :1296 is (Lgain - Lloss)/Lgain < threshold, i.e. Lloss/Lgain > 1 - threshold. The log at :1298 says 'Cooling balance reached: Lloss/Lgain ratio below {threshold}'.",
    "expected": "Message should read 'Lloss/Lgain above 1-threshold' or '(Lgain-Lloss)/Lgain below threshold'.",
    "failure_scenario": "A reader debugging a run with phaseSwitch_LlossLgain=0.05 concludes the phase ended when cooling was only 5% of gain, when in fact it ended when cooling reached 95% of gain.",
    "repro": "Read the log line against the branch condition.",
    "confidence": "high"
  },
  {
    "id": "S5b-A-11",
    "file": "trinity/phase1b_energy_implicit/run_energy_implicit_phase.py",
    "line": 514,
    "class": "silent-failure",
    "severity": "S2",
    "claim": "A bare `except Exception` silently sets the ambient confining pressure to zero.",
    "evidence": ":509-515 wraps get_density_profile and the P_ext expression; any failure yields P_ext = 0.0, which then makes F_ion_in = 0 at :535 and press_HII_in = 0 at :560.",
    "expected": "A density-profile failure at the shell radius is a real error and should propagate, or at minimum log at WARNING with the exception.",
    "failure_scenario": "get_density_profile raises for rShell outside the tabulated profile (e.g. after blowout, rShell > rCloud with a profile that only spans the cloud). Every subsequent segment silently drops the inward ambient pressure term from the force budget with no log line at all.",
    "repro": "Add a counter in the except and run any config in which rShell exceeds the profile support; the counter is non-zero and nothing appears in the log.",
    "confidence": "high"
  },
  {
    "id": "S5b-A-12",
    "file": "trinity/phase1b_energy_implicit/run_energy_implicit_phase.py",
    "line": 521,
    "class": "units",
    "severity": "S2",
    "claim": "The two contributions to P_ext are built by different routes: the cloud term carries the mu_convert/mu_ion_shell correction, the ISM term does not.",
    "evidence": ":513 P_ext = (mu_convert/mu_ion_shell) * n_r * k_B * TShell_ion; :521 P_ext += PISM * k_B. For :521 to be a pressure in AU (Msun pc^-1 Myr^-2), PISM must already be a total-particle n*T in pc^-3 K. The mean-molecular-weight correction that converts a hydrogen number density to a total particle density on line 513 is absent on line 521.",
    "expected": "Both terms should be expressed in the same variable (either both n_H-based with the mu correction, or both n_tot-based without it), and PISM's storage convention should be checked against the schema.",
    "failure_scenario": "If PISM is specified as a hydrogen-based n*T (or as a pressure), the ISM contribution to P_ext — and hence to F_ion_in = P_ext*4*pi*R2^2 — is off by mu_convert/mu_ion_shell (order 2 for ionised gas) or by k_B entirely. This only bites once rShell >= rCloud, i.e. exactly in the post-blowout regime.",
    "repro": "Compare params['PISM'] * k_B against (mu_convert/mu_ion_shell) * nISM * k_B * T_ISM for the default schema values; they should agree to within the assumed ISM temperature.",
    "confidence": "low"
  },
  {
    "id": "S5b-A-13",
    "file": "trinity/phase1b_energy_implicit/run_energy_implicit_phase.py",
    "line": 535,
    "class": "regime",
    "severity": "S3",
    "claim": "F_ion_in evaluates the pressure at rShell but the area at R2.",
    "evidence": "P_ext is computed from the density profile at shell_props.rShell (:510) and, on the ISM branch, gated on rShell >= rCloud (:520). F_ion_in = P_ext * FOUR_PI * R2**2 at :535 uses R2, not rShell.",
    "expected": "Pressure and area should refer to the same surface, or the mismatch should be a deliberate documented choice.",
    "failure_scenario": "In a thick-shell regime (rShell noticeably different from R2) the inward pressure force is scaled by the wrong area; the error grows as (R2/rShell)^2.",
    "repro": "Log rShell/R2 per segment; where it departs from 1, F_ion_in is inconsistent by that ratio squared.",
    "confidence": "medium"
  },
  {
    "id": "S5b-A-14",
    "file": "trinity/phase1b_energy_implicit/run_energy_implicit_phase.py",
    "line": 1303,
    "class": "state",
    "severity": "S2",
    "claim": "isCollapse is a one-way latch that permanently freezes the shell mass.",
    "evidence": ":1302-1303 sets params['isCollapse'].value = True when v2 < 0 and R2 < R2_prev; nothing in this module ever sets it back to False. Both shell-mass blocks (:957-960 and :1188-1189) skip the mass-profile evaluation entirely when it is set, so shell_mass and shell_massDot are frozen at their last values for the rest of the phase.",
    "expected": "The latch should be cleared when the shell re-expands (v2 > 0 and R2 > R2_prev), or the freeze should apply only while the collapse condition holds.",
    "failure_scenario": "A bubble that dips (one segment of v2 < 0 with R2 decreasing) and then re-expands past rCloud keeps M_sh at the pre-dip value forever, so F_grav (proportional to M_sh*(M_cl+M_sh/2)) and the ODE's inertia term are frozen while R2 grows by orders of magnitude.",
    "repro": "Count segments after the first isCollapse=True and assert params['shell_mass'] changes; it does not.",
    "confidence": "high"
  },
  {
    "id": "S5b-A-15",
    "file": "trinity/phase1b_energy_implicit/run_energy_implicit_phase.py",
    "line": 1245,
    "class": "sign",
    "severity": "S2",
    "claim": "The two Lloss branches are semantically asymmetric: one passes a raw cooling luminosity plus the real leak, the other passes an already-effective loss with leak forced to zero.",
    "evidence": ":1238-1243 uses _Lcool = bubble_props.bubble_LTotal and _leak = params['bubble_Leak']. :1245-1247 uses _Lcool = params['bubble_Lloss'] — the value written at :930 from betadelta_result.L_loss, i.e. an output of effective_Lloss_from_params-like processing — and _leak = 0.0. Both are then fed to the same effective_Lloss_from_params.",
    "expected": "Both branches should pass the same kind of quantity (raw cooling luminosity) and the same leak.",
    "failure_scenario": "When the beta-delta solve returns bubble_properties=None (the degraded/no-root path, exactly when the fallback matters most), Lloss is a double-processed value carried over from an arbitrarily earlier segment with the leak dropped. That Lloss drives the cooling_balance termination test at :1296, so the phase can end at the wrong time.",
    "repro": "Force bubble_properties=None for one segment and compare Lloss with the value the if-branch would give from the same state.",
    "confidence": "medium"
  },
  {
    "id": "S5b-A-16",
    "file": "trinity/phase1b_energy_implicit/run_energy_implicit_phase.py",
    "line": 847,
    "class": "state",
    "severity": "S2",
    "claim": "A no-physical-root beta-delta solve does not skip the segment: its beta/delta are written to params and drive Ed/Td for up to 50 consecutive segments.",
    "evidence": "beta, delta are taken unconditionally at :832-833. The no_physical_root branch at :847-879 only counts, logs and (at streak >= 50) breaks — it does not `continue` or restore. :885-886 write them to params, :992-993 build Ed and Td from them, and the ODE integrates with those derivatives.",
    "expected": "If the root is non-physical the previous physical (beta, delta) should be held, matching what the log at :856-864 claims about holding the last physical dMdt.",
    "failure_scenario": "Up to 49 segments (NO_ROOT_HANDOFF_STREAK-1) are integrated with dEb/dt and dT0/dt derived from a beta/delta the solver itself flagged as unphysical, before the handoff fires. At dt_segment up to 5e-2 Myr that is up to 2.45 Myr of trajectory.",
    "repro": "Log (beta, delta) alongside no_physical_root per segment; on a no-root segment they differ from the previous segment's values.",
    "confidence": "medium"
  },
  {
    "id": "S5b-A-17",
    "file": "trinity/phase1b_energy_implicit/run_energy_implicit_phase.py",
    "line": 1394,
    "class": "silent-failure",
    "severity": "S2",
    "claim": "The phase-boundary reconciliation swallows every exception and, because save_snapshot() is the last statement inside the try, a failure loses the final snapshot with only a warning.",
    "evidence": "The try opens at :1366 and closes at :1395 with `except Exception as e: logger.warning(...)`. params.save_snapshot() at :1394 is the final statement in the try. Any failure in get_current_sps_feedback, compute_R1_Pb, shell_structure_pure or compute_forces_pure aborts before it.",
    "expected": "The snapshot should be saved outside the try, or the exception should be re-raised.",
    "failure_scenario": "shell_structure_pure raises at the final state (e.g. a degenerate shell after the energy_to_momentum clamp sets Eb=1e3 and Pb collapses). The phase's last output row is missing entirely and the only trace is one WARNING line.",
    "repro": "Force an exception inside the try and assert params.save_count increased; it does not.",
    "confidence": "high"
  },
  {
    "id": "S5b-A-18",
    "file": "trinity/phase1b_energy_implicit/run_energy_implicit_phase.py",
    "line": 1384,
    "class": "state",
    "severity": "S4",
    "claim": "The reconciliation refreshes only part of the state, leaving the final snapshot with mixed vintages — most sharply, F_ion_in is fresh while press_HII_in (the pressure it is derived from) is stale.",
    "evidence": "The block writes R1, Pb, P_HII, F_HII, F_grav, F_ion_in, F_ram, F_rad, P_drive, P_ram (:1372-1393) but never writes n_IF, R_IF, press_HII_in, F_ram_wind, F_ram_SN, shell_mass, shell_massDot, c_sound, bubble_Leak, cool_beta or cool_delta. The in-loop block at :1002-1009 does write n_IF, R_IF, press_HII_in, F_ram_wind, F_ram_SN.",
    "expected": "The reconciliation should mirror the in-loop write set, or explicitly declare which fields are intentionally left at their last in-loop value.",
    "failure_scenario": "The final dictionary.jsonl row satisfies F_ion_in != press_HII_in * 4*pi*R2^2, breaking any consistency check or plot that reconstructs the force budget from the recorded pressures.",
    "repro": "On the last output row assert abs(F_ion_in - press_HII_in*4*pi*R2**2) < tol; it fails whenever the shell structure changed over the last segment.",
    "confidence": "high"
  },
  {
    "id": "S5b-A-19",
    "file": "trinity/phase1b_energy_implicit/run_energy_implicit_phase.py",
    "line": 1410,
    "class": "deadcode",
    "severity": "S4",
    "claim": "termination_reason == 'unknown' is unreachable, and so is the warning-level log selector that keys off it.",
    "evidence": "The while condition at :758 is `t_now <= tmax and segment_count < MAX_SEGMENTS`. The bottom-of-body check at :1309 already breaks (setting termination_reason) whenever t_now > tmax, and :1040 breaks on t_now >= tmax at the top of the next iteration. Therefore the time half of the while condition can never end the loop; only segment_count >= MAX_SEGMENTS can, which maps to 'max_segments' at :1410. Every `break` in the body sets termination_reason first.",
    "expected": "Drop the 'unknown' fallback, or drop the redundant t_now check at :1309.",
    "failure_scenario": "",
    "repro": "Static: enumerate the loop exits; no path leaves termination_reason None with segment_count < MAX_SEGMENTS.",
    "confidence": "high"
  },
  {
    "id": "S5b-A-20",
    "file": "trinity/phase1b_energy_implicit/run_energy_implicit_phase.py",
    "line": 1040,
    "class": "deadcode",
    "severity": "S4",
    "claim": "The `tmax is not None` guards at :1040 and :1309 can never be False.",
    "evidence": "tmax = params['stop_t'].value at :665 and is compared with `tmin >= tmax` at :670, which would raise TypeError if it were None. So by the time the loop runs, tmax is guaranteed non-None.",
    "expected": "Remove the guards, or move the None handling to :665 where it would be meaningful.",
    "failure_scenario": "",
    "repro": "Static.",
    "confidence": "high"
  },
  {
    "id": "S5b-A-21",
    "file": "trinity/phase1b_energy_implicit/run_energy_implicit_phase.py",
    "line": 699,
    "class": "deadcode",
    "severity": "S4",
    "claim": "Several computed values and imports are never used.",
    "evidence": "n_estimate (:699) computed and never referenced. cooling_balance_factory (:752) unpacked and never referenced. nISM (:498) read from params and never used (and will KeyError if absent, unlike the guarded PISM access two lines later). P_ram is hard-coded 0.0 (:559) and written to params every segment (:1006) and again at :1393, so params['P_ram'] is identically zero. k_blowout (:233) is never passed a non-default value. Imports scipy.optimize (:59), Dict/Optional/Tuple (:61), cvt (:66), ODEResult (:78), compute_derived_quantities (:79), BetaDeltaResult (:87) are unreferenced.",
    "expected": "Delete, or wire up.",
    "failure_scenario": "",
    "repro": "ruff F401/F841 over this file.",
    "confidence": "high"
  },
  {
    "id": "S5b-A-22",
    "file": "trinity/phase1b_energy_implicit/run_energy_implicit_phase.py",
    "line": 985,
    "class": "deadcode",
    "severity": "S4",
    "claim": "F_HII is computed twice per segment with the same inputs and the first write is immediately overwritten.",
    "evidence": ":985-986 computes F_HII = 4.0*np.pi*R2**2*P_HII and writes params['F_HII']. compute_forces_pure then reads the same params['P_HII'] at :531 and recomputes F_HII = FOUR_PI*R2**2*P_HII at :536, which is written over the top at :998. The two expressions are identical; :985 also uses a literal 4.0*np.pi rather than the module's FOUR_PI.",
    "expected": "Drop :985-986.",
    "failure_scenario": "",
    "repro": "Static.",
    "confidence": "high"
  },
  {
    "id": "S5b-A-23",
    "file": "trinity/phase1b_energy_implicit/run_energy_implicit_phase.py",
    "line": 1191,
    "class": "deadcode",
    "severity": "S4",
    "claim": "get_mass_profile is evaluated twice per segment at the same R2, and the segment-start ratchet is consequently a guaranteed no-op.",
    "evidence": ":1191 calls get_mass_profile(R2, params, return_mdot=False) at the post-step R2. R2 is not modified between there and :962 of the next iteration, which calls get_mass_profile(R2, params, return_mdot=True, rdot=v2) at the same R2. Since :1196 already stored that mass, the comparison mShell_new < prev_mShell at :964 compares a number with itself.",
    "expected": "Make one call per segment, or take the mdot from the post-step call.",
    "failure_scenario": "",
    "repro": "Count get_mass_profile calls per segment; it is 2.",
    "confidence": "high"
  },
  {
    "id": "S5b-A-24",
    "file": "trinity/phase1b_energy_implicit/run_energy_implicit_phase.py",
    "line": 315,
    "class": "numerical",
    "severity": "S4",
    "claim": "compute_max_dex_change silently ignores any key that is exactly zero in either sample — the largest possible relative change scores zero dex — and its zero test sits outside the try/except.",
    "evidence": ":315 `if old_val == 0 or new_val == 0: continue`, before the try at :323. The except at :326 catches (ValueError, ZeroDivisionError), neither of which np.log10 of a positive ratio can raise, so it is dead; but an array-valued monitor entry would raise ValueError at :315 where nothing catches it.",
    "expected": "Treat 0 -> non-zero as a maximal change (or a large fixed dex), and either guard the zero test or assert the monitored values are scalars.",
    "failure_scenario": "F_rad transitions from a finite value to exactly 0.0 when shell_props.isDissolved flips (:542-543), and F_HII transitions to exactly 0.0 when n_IF_Str drops to 0 (:980-983). Both are scored as no change. (Masked today by S5b-A-03, which makes those keys unable to differ between the two captures at all.)",
    "repro": "assert compute_max_dex_change({'F_rad': 1e40}, {'F_rad': 0.0}, ['F_rad']) > 0 — it returns 0.0.",
    "confidence": "high"
  },
  {
    "id": "S5b-A-25",
    "file": "trinity/phase1b_energy_implicit/run_energy_implicit_phase.py",
    "line": 943,
    "class": "silent-failure",
    "severity": "S4",
    "claim": "bubble_Tavg falls back to a hard-coded 1e6 K on any falsy value, including a legitimately computed 0.0, while NaN passes through unguarded.",
    "evidence": ":943 `bubble_Tavg = params['bubble_Tavg'].value if params['bubble_Tavg'].value else 1e6`. Python truthiness: 0.0 and None are both falsy; NaN is truthy.",
    "expected": "`if ... is not None` plus an explicit finiteness/positivity check.",
    "failure_scenario": "If the bubble structure solve returns Tavg = 0.0 (degenerate bubble), c_sound is computed for a 1e6 K plasma, which then feeds get_leak_luminosity on the next segment (:817). If Tavg is NaN, get_soundspeed returns NaN and c_sound silently becomes NaN for the rest of the phase.",
    "repro": "assert np.isfinite(params['c_sound'].value) after :944 on every segment.",
    "confidence": "medium"
  },
  {
    "id": "S5b-A-26",
    "file": "trinity/phase1b_energy_implicit/run_energy_implicit_phase.py",
    "line": 1071,
    "class": "numerical",
    "severity": "S4",
    "claim": "A single scalar atol=1e-8 is applied to four states spanning ~15 orders of magnitude, and min_step=1e-6 Myr leaves only a factor-20 window below max_step=2e-5 Myr.",
    "evidence": ":170-173 and :1071-1077. y = (R2 ~ 10 pc, v2 ~ 10-10^3 pc/Myr, Eb ~ 10^5-10^10 Msun pc^2 Myr^-2, T0 ~ 10^6-10^7 K). For the large components rtol=1e-6 dominates; for a component near zero the effective requirement is |err| < 1e-8 in that component's own units.",
    "expected": "A per-component atol vector, e.g. atol=[1e-8 pc, 1e-6 pc/Myr, 1e-3*Eb_scale, 1e-2 K].",
    "failure_scenario": "During a collapse v2 passes through 0 pc/Myr; LSODA must resolve v2 to 1e-8 pc/Myr absolute, drives h below min_step=1e-6 Myr, and sol.success goes False — which then takes the S5b-A-04 path that ends the phase with no end code.",
    "repro": "Run a config that produces a v2 sign change inside a segment and check sol.message.",
    "confidence": "medium"
  },
  {
    "id": "S5b-A-27",
    "file": "trinity/phase1b_energy_implicit/run_energy_implicit_phase.py",
    "line": 173,
    "class": "numerical",
    "severity": "S4",
    "claim": "ODE_MAX_STEP = DT_SEGMENT_MIN/5 caps the integrator step at 2e-5 Myr while dt_segment may reach 5e-2 Myr, forcing >= 2500 internal steps per segment and making the adaptive dt_segment controller irrelevant to integration accuracy.",
    "evidence": ":173 ODE_MAX_STEP = DT_SEGMENT_MIN / 5 = 2e-5. :114 DT_SEGMENT_MAX = 5e-2. :1056 t_segment_end = min(t_now + dt_segment, tmax). Ratio 2500. Each internal step calls get_ODE_implicit_pure, which calls get_ODE_Edot_pure and discards its third component.",
    "expected": "Either tie max_step to dt_segment (e.g. dt_segment/20) or let LSODA choose, and describe dt_segment as the beta-delta re-solve cadence rather than an accuracy control.",
    "failure_scenario": "No wrong number; up to 5000 segments x 2500 steps = 1.25e7 RHS evaluations, each doing a full energy-ODE evaluation of which one third is thrown away.",
    "repro": "Count RHS calls for one segment at dt_segment=5e-2; it is >= 2500.",
    "confidence": "high"
  },
  {
    "id": "S5b-A-28",
    "file": "trinity/phase1b_energy_implicit/run_energy_implicit_phase.py",
    "line": 620,
    "class": "deadcode",
    "severity": "S4",
    "claim": "The RHS computes the energy-ODE's own dEb/dt on every evaluation and discards it.",
    "evidence": ":618 dydt_energy = get_ODE_Edot_pure(t, [R2,v2,Eb], snapshot, params) returns three components; only [0] and [1] are read (:620-621) and the returned vector substitutes Ed_from_beta for the third (:624).",
    "expected": "If get_ODE_Edot_pure has a cheaper 2-component form, use it; otherwise note the discard.",
    "failure_scenario": "",
    "repro": "Static.",
    "confidence": "high"
  },
  {
    "id": "S5b-A-29",
    "file": "trinity/phase1b_energy_implicit/run_energy_implicit_phase.py",
    "line": 1394,
    "class": "other",
    "severity": "S4",
    "claim": "stop_at_rCloud_nSnap is off by one: the reconciliation snapshot past rCloud is written but never counted.",
    "evidence": "The counter is incremented only inside the loop at :1023-1026, gated on params.save_count increasing. The break at :768-775 happens before that segment's save. But the post-loop reconciliation calls params.save_snapshot() at :1394 unconditionally, adding one more row while R2 > rCloud.",
    "expected": "The output should contain exactly stop_at_rCloud_nSnap snapshots beyond rCloud.",
    "failure_scenario": "With stop_at_rCloud_nSnap=1 the output contains 2 rows with R2 > rCloud.",
    "repro": "Run with stop_at_rCloud_nSnap=1 and count output rows with R2 > rCloud; the count is 2.",
    "confidence": "medium"
  },
  {
    "id": "S5b-A-30",
    "file": "trinity/phase1b_energy_implicit/run_energy_implicit_phase.py",
    "line": 1288,
    "class": "state",
    "severity": "S4",
    "claim": "The blowout / ebpeak / cooling_balance / no_physical_root_handoff breaks do not set SimulationEndReason, so a stale reason from a previous phase persists into the output.",
    "evidence": "Compare :768-774, :1041-1044, :1152-1157, :1311-1313, :1321-1323, :1330-1332 (which all write SimulationEndReason and SimulationEndCode) with :866, :1099, :1290, :1297 (which write only the local termination_reason).",
    "expected": "Either set an explicit hand-off reason on those paths, or clear SimulationEndReason on phase entry.",
    "failure_scenario": "A run that ends phase 1b via 'blowout' carries whatever SimulationEndReason the previous phase wrote, e.g. a stale 'Stopping time reached', into the metadata of the hand-off snapshot.",
    "repro": "Run with transition_trigger='blowout' and inspect params['SimulationEndReason'] at phase exit.",
    "confidence": "high"
  },
  {
    "id": "S5b-A-31",
    "file": "trinity/phase1b_energy_implicit/run_energy_implicit_phase.py",
    "line": 262,
    "class": "other",
    "severity": "S4",
    "claim": "An empty or whitespace-only transition_trigger silently disables all transition triggers, while None raises.",
    "evidence": ":262 parts = {p.strip() for p in str(x).split(',') if p.strip()}. For '' or '  ' this is the empty set, which passes the unknown-token check at :266 and returns frozenset(). r1_transition_decision then always returns None (:282-286) and the cooling_balance test at :1296 is short-circuited. For None, str(None) == 'None' becomes an unknown token and ValueError is raised.",
    "expected": "Reject an empty trigger set explicitly, or handle None the same way as ''.",
    "failure_scenario": "A .param with transition_trigger = '' runs the implicit phase all the way to max_segments (5000) or stop_t with no phase transition and only an INFO-level completion line.",
    "repro": "assert parse_transition_triggers('') raises or is non-empty; it returns frozenset().",
    "confidence": "high"
  },
  {
    "id": "S5b-A-32",
    "file": "trinity/phase1b_energy_implicit/run_energy_implicit_phase.py",
    "line": 230,
    "class": "other",
    "severity": "S4",
    "claim": "_inflow_frac_thickness returns the radial bracket of inflowing zones, not the fraction of the profile that is inflowing; a single inflowing cell returns exactly 0.0.",
    "evidence": ":229-230 rneg = r[v<0]; return abs(rneg.max() - rneg.min())/rspan. With one negative cell, max == min and the result is 0.0. Outflowing cells lying between the innermost and outermost inflowing cells are included in the numerator.",
    "expected": "Either sum the widths of the inflowing zones, or count neg.sum()/neg.size.",
    "failure_scenario": "params['v_neg_frac_thick'] (:898) reports 0.0 at the onset of inflow (exactly when a diagnostic is wanted) and over-reports once inflow becomes patchy.",
    "repro": "assert _inflow_frac_thickness([1.0, -1.0, 1.0], [0.0, 0.5, 1.0]) > 0 — it returns 0.0.",
    "confidence": "high"
  },
  {
    "id": "S5b-A-33",
    "file": "trinity/phase1b_energy_implicit/run_energy_implicit_phase.py",
    "line": 752,
    "class": "state",
    "severity": "S4",
    "claim": "ODE events are built once, before the loop, from a params object that then mutates every segment; and the phase is terminated on any check_event_termination hit regardless of the event's terminal/direction attributes as far as this module can tell.",
    "evidence": ":752 ode_events, cooling_balance_factory = build_implicit_phase_events(params), outside the while at :758. The same ode_events list is passed to every solve_ivp call (:1074). :1095-1119 breaks the phase unconditionally on event_result.triggered.",
    "expected": "If any event closure captures scalar params values at build time, it must be rebuilt per segment; and the break should distinguish terminal from non-terminal events.",
    "failure_scenario": "Cannot be constructed from this slice — build_implicit_phase_events and check_event_termination are in trinity/phase_general/phase_events.py, which is outside my input. Flagged as a coupling to check.",
    "repro": "",
    "confidence": "low"
  },
  {
    "id": "S5b-A-34",
    "file": "trinity/phase1b_energy_implicit/run_energy_implicit_phase.py",
    "line": 1341,
    "class": "state",
    "severity": "S4",
    "claim": "On the no_physical_root_handoff exit, the returned beta/delta arrays and params['cool_beta']/['cool_delta'] disagree.",
    "evidence": "The break at :879 is upstream of :885-886, so params keeps the previous segment's beta/delta. The trailing append at :1341-1349 uses the local beta/delta, which were reassigned from the failed solve at :832-833. The same break also skips the residual bookkeeping at :910-930, so betadelta_converged, residual_deltaT, residual_betaEdot etc. are all one segment stale in the final snapshot.",
    "expected": "The two should agree, or the local beta/delta should be restored before the break.",
    "failure_scenario": "The last row of ImplicitPhaseResults.beta differs from the cool_beta recorded in the final dictionary.jsonl row on any run that ends with no_physical_root_handoff.",
    "repro": "Run to a no_physical_root_handoff exit and compare results.beta[-1] with params['cool_beta'].value.",
    "confidence": "high"
  },
  {
    "id": "S5b-A-35",
    "file": "trinity/phase1b_energy_implicit/run_energy_implicit_phase.py",
    "line": 1251,
    "class": "other",
    "severity": "S4",
    "claim": "Object truthiness is used as a presence test for Param wrappers in six places; a falsy wrapper (or a bare float stored instead of a wrapper) silently takes the default branch.",
    "evidence": ":500 (PISM -> 0.0), :955 and :1186 (isCollapse -> False), :1241 (bubble_Leak -> 0.0), :1246 (bubble_Lloss -> 0.0), :1251 (phaseSwitch_LlossLgain -> literal 0.05). Each is `if obj and hasattr(obj,'value')` or `obj.value if obj and hasattr(...) else default`.",
    "expected": "`if obj is not None and hasattr(obj, 'value')`.",
    "failure_scenario": "If the Param class ever defines __bool__ or __len__ in terms of its value, a stored PISM of 0, a phaseSwitch_LlossLgain of 0, or an isCollapse of False all take the default branch — and for phaseSwitch_LlossLgain the default is 0.05, not 0, which changes the cooling_balance termination time.",
    "repro": "Store a bare float (not a Param) under 'PISM' and confirm compute_forces_pure silently uses PISM = 0.0.",
    "confidence": "low"
  },
  {
    "id": "S5b-A-36",
    "file": "trinity/phase1b_energy_implicit/run_energy_implicit_phase.py",
    "line": 1067,
    "class": "state",
    "severity": "S4",
    "claim": "The RHS lambda closes over the live mutable params object rather than a frozen copy, unlike the explicitly frozen snapshot beside it.",
    "evidence": ":1067 `lambda t, y: get_ODE_implicit_pure(t, y, snapshot, params, Ed, Td)`. snapshot is deliberately frozen at :1051 by create_ODE_snapshot, but the fourth argument (params_for_feedback in the callee, :587) is the live dict.",
    "expected": "Pass a frozen feedback view, matching the snapshot's intent.",
    "failure_scenario": "Benign today because params is not written between :1067 and :1079, but the RHS is not a pure function of (t, y, snapshot): if get_ODE_Edot_pure ever writes to params, or if solve_ivp is ever driven concurrently, the segment's derivative becomes path-dependent.",
    "repro": "",
    "confidence": "low"
  }
]
```
