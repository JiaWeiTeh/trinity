# S6 transition + momentum — Lens A (what the code does)

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

**Scope read:** `trinity/phase1c_transition/run_transition_phase.py`,
`trinity/phase1c_transition/__init__.py` (empty),
`trinity/phase2_momentum/run_momentum_phase.py`,
`trinity/phase2_momentum/__init__.py` (empty).

**Shared-file exception used:** yes — I consulted the S1 copy of
`trinity/_functions/unit_conversions.py` for unit-constant definitions only. Everything below about
dimensions is derived from that file plus the arithmetic in my two runners. All comments and
docstrings in every file I read were blanked; nothing in this report is reconstructed prose.

**Internal ("AU") unit system, established from `unit_conversions.py`:** length pc, mass M☉, time
Myr, temperature K. Derived: velocity pc/Myr, energy M☉·pc²/Myr², luminosity M☉·pc²/Myr³, force
M☉·pc/Myr², pressure M☉/(pc·Myr²), number density pc⁻³, `k_B` in M☉·pc²/(Myr²·K), `G` in
pc³/(M☉·Myr²). (Cross-checks: `Pb_cgs2au = g2Msun / cm2pc / s2Myr² = 1.5454e12` ✓;
`k_B_cgs2au ≡ E_cgs2au` ✓, i.e. K is dimensionless in AU.)

Unknowns I could not resolve inside my slice, treated as opaque: `get_bubbleParams.pRam`,
`compute_R1_Pb`, `get_ODE_Edot_pure`, `create_ODE_snapshot`, `shell_structure_pure`,
`mass_profile.get_mass_profile`, `density_profile.get_density_profile`,
`operations.get_soundspeed`, `get_current_sps_feedback`, the `phase_events` builders/appliers.

---

## 0. Executive shape

Both files are the *same program* with the state vector shortened by one component and the
energy machinery deleted:

| | phase1c transition | phase2 momentum |
|---|---|---|
| ODE state `y` | `[R2, v2, Eb]` | `[R2, v2]` |
| RHS | `get_ODE_transition_pure` → delegates `rd, vd` to `get_ODE_Edot_pure`, overrides `Ed` | `get_ODE_momentum_pure` → writes `rd, vd` itself |
| bubble pressure | `Pb` from `compute_R1_Pb(R2, Eb, …)` | `Pb := pRam(R2, …)`, `R1 := R2` |
| segment cap | `MAX_SEGMENTS = 5000` | `MAX_SEGMENTS = 10000` |
| extra exit | `Eb < ENERGY_FLOOR`, `ram_fraction > 0.9` | — |

Lines 86–168 of the momentum file and 93–176 of the transition file
(`DT_SEGMENT_*`, `ADAPTIVE_*`, `VELOCITY_THRESHOLD_*`, `ODE_*`, `ADAPTIVE_MONITOR_KEYS`,
`compute_max_dex_change`, `get_monitor_values`) are byte-for-byte identical apart from
`MAX_SEGMENTS` and the transition-only `ENERGY_FLOOR = 1e3`. The two `ForceProperties`
dataclasses are also identical duplicates
(`run_transition_phase.py:254-268` ≡ `run_momentum_phase.py:189-203`).

---

## 1. Shared helpers (identical in both files)

### `compute_max_dex_change(params_before, params_after, keys)`
`run_transition_phase.py:143-161` ≡ `run_momentum_phase.py:135-153`.

Returns `max over keys of |log10(|new|/|old|)|`, with three guards:
* either value `None` → skipped;
* either value `== 0` → skipped (so a quantity that reaches exactly 0 is invisible to the controller);
* sign flip `(old>0) != (new>0)` → contributes the bare literal **`1.0`** dex and `continue`s.

Dimensionless in, dimensionless out (a ratio of like-typed quantities). `ValueError/ZeroDivisionError`
are caught and the key skipped; `np.log10(0)` would actually emit a warning and return `-inf`, not
raise, but the `== 0` guard above pre-empts that.

### `get_monitor_values(params)`
`…transition:164-176` ≡ `…momentum:156-168`. Unwraps `.value` off each of the 30
`ADAPTIVE_MONITOR_KEYS` and swallows **all** exceptions per key (`except Exception: pass`).

---

## 2. Transition phase

### 2.1 `get_ODE_transition_pure(t, y, snapshot, params_for_feedback, c_sound)` — `:198-247`

**Inputs:** `t` [Myr], `y = [R2, v2, Eb]` [pc, pc/Myr, M☉pc²/Myr²], a frozen `ODESnapshot`,
the live mutable `params` (passed as `params_for_feedback`), and `c_sound` [pc/Myr] frozen for the
segment.

**Maths.**

    (rd, vd, Ed_bal) = get_ODE_Edot_pure(t, y, snapshot, params)          # :231
    Ed_sc = -Eb·c_sound / R2               if c_sound > 0 and R2 > 0      # :238
          = 0.0                            otherwise                      # :240
    Ed    = min(Ed_bal, Ed_sc)                                            # :245
    return [rd, vd, Ed]

So `dR2/dt` and `dv2/dt` are **exactly** the phase-1 energy-driven expressions — the transition phase
changes *only* the energy equation. The energy equation is

    dEb/dt = min( Ėb,energy-balance , −Eb·c_s/R2 ).

**Dimensions.** `Eb·c_sound/R2` = (M☉pc²/Myr²)(pc/Myr)/pc = M☉pc²/Myr³ = energy/time ✓ (writing it as
`Eb / (R2/c_sound)` makes the sound-crossing time `R2/c_s` [Myr] explicit). Balanced.

**Control flow that changes the maths.**
* `c_sound <= 0` **or** `R2 <= 0` → `Ed_sc = 0.0`, hence `Ed = min(Ed_bal, 0)`. In that fallback the
  code *silently forbids any positive* `dEb/dt`: an energy-injecting balance term is replaced by 0.
  This is a clamp, not a physical limit (see finding S6-A-14).
* For `Eb > 0`, `Ed_sc < 0` always, so `Ed < 0` always: `Eb` is monotonically decreasing throughout
  the transition phase and *all* net energy injection from `Ed_bal` is discarded whenever
  `Ed_bal > Ed_sc`. The `min` is what guarantees the `energy_floor` / `ram_dominated` exits fire.
* For `Eb < 0` (possible if the integrator overshoots), `Ed_sc > 0` and `min` selects `Ed_bal`, i.e.
  the sound-crossing leak reverses sign and becomes a *source*. There is no `Eb >= 0` guard here;
  the only protection is the `energy_floor` event built at `:457`.

**Literals:** `0` (twice, in the guards), `0.0`.

### 2.2 `compute_forces_pure(R2, mShell, Pb, shell_props, params)` — `:271-360`

Reads `params`: `G, mCluster, k_B, TShell_ion, rCloud, nISM, PISM, mu_convert, mu_ion_shell, P_HII,
Lmech_total, v_mech_total, Lbol, c_light, dust_KappaIR`.
Reads `shell_props`: `rShell, shell_fAbsorbedIon, n_IF, R_IF, isDissolved,
shell_fAbsorbedWeightedTotal, shell_tauKappaRatio`. Writes nothing; returns `ForceProperties`.

    F_grav   = G·mShell·(mCluster + 0.5·mShell) / R2²                                     :284
    P_ext    = (μ_conv/μ_ion)·n(rShell)·k_B·T_ion      if shell_fAbsorbedIon < 1.0        :306
             = 0.0                                      otherwise (or on any exception)  :308,310
    P_ext   += PISM·k_B                                 if rShell >= rCloud              :314
    P_ram    = pRam(R2, Lmech_total, v_mech_total)                                       :329
    P_drive  = max(Pb, P_HII + P_ram)                                                    :331
    F_ion_in = P_ext·4π·R2²                                                              :334
    F_HII    = 4π·R2²·P_HII                                                              :335
    F_ram    = Pb·4π·R2²          <-- note: Pb, not P_ram                                :338
    F_rad    = 0.0                                        if shell_props.isDissolved     :342
             = f_absW,tot·(Lbol/c_light)·(1 + Σ_{τ/κ}·κ_IR)  otherwise                   :344-346

**Dimensions.** `G·M²/R²` = pc³M☉⁻¹Myr⁻²·M☉²·pc⁻² = M☉·pc/Myr² ✓ force.
`n·k_B·T` = pc⁻³·M☉pc²Myr⁻²K⁻¹·K = M☉/(pc·Myr²) ✓ pressure; ×4πR2² → force ✓.
`Lbol/c_light` = M☉pc²Myr⁻³ / (pc/Myr) = M☉pc/Myr² ✓ force. The bracket `(1 + τκ_ratio·κ_IR)` is
dimensionless **only if** `shell_tauKappaRatio` carries M☉/pc² (a surface density, τ/κ) and
`dust_KappaIR` carries pc²/M☉ — which is what the names imply and is the only self-consistent
reading. No imbalance found.
`PISM·k_B` (`:314`) balances only if `PISM` is stored as *P/k_B* (K·pc⁻³ in AU), not as an AU
pressure; see S6-A-16.

**Odd/inconsistent inside this function:**
* `nISM` is read at `:291` and **never used**.
* `P_ext` is evaluated at `rShell` but multiplied by `R2²` at `:334` — two different radii in one
  force.
* `P_drive` (`:331`) is computed and stored but is not used by any expression here; it is a
  diagnostic only.
* `P_ext` has a **step** at `shell_fAbsorbedIon == 1.0`: it is the *full* ionized pressure for
  FABSi = 1−ε and *exactly zero* at FABSi = 1. There is no `(1 − FABSi)` weighting.
* The `try/except Exception` at `:307` converts any failure of `get_density_profile` into
  `P_ext = 0.0` with no log line.
* `F_ram` is the **bubble** pressure force, while the sibling field `P_ram` in the same returned
  object is the ram pressure. `F_ram ≠ 4πR2²·P_ram` here. This is the single sharpest divergence
  from the momentum twin (§4.1).

### 2.3 `run_phase_transition(params)` — `:367-886`

**Entry (`:390-399`).**
`v2_from_ODE = params['v2']`; `v2_from_alpha = cool_alpha·R2/t_now` — the latter is computed purely
for the log line at `:392-394` and then discarded. `:399` then *overwrites*
`cool_alpha := t_now·v2 / R2` (dimensionless: Myr·pc/Myr/pc ✓), i.e. α is re-derived from the ODE
velocity, self-consistently. No guard on `R2 == 0` or `t_now == 0`.

**Early return (`:407-424`).** If `t_now >= stop_t`: sets `SimulationEndCode = STOPPING_TIME`,
`EndSimulationDirectly = True`, returns one-element arrays. This is the only place `tmax` could be
`None` without an exception — and it *would* raise `TypeError` on the `>=` if it were, which makes
every later `if tmax is not None` guard (`:605`, `:779`) unreachable-as-written.

**Local state initialised (`:427-457`):** `R2, v2, Eb, T0` from params; result lists;
`R2_prev = R2`; `dt_segment = 2e-3`; `t_diss_onset = inf`; events from
`build_transition_phase_events(params, energy_floor=1e3)`.

**Loop `while t_now <= tmax and segment_count < 5000` (`:463`).** Order of operations per segment:

1. `:468-479` rCloud-snapshot exit (before any writes).
2. `:487-491` write `t_now, R2, v2, Eb, T0` into params. **`T0` is the local captured once at
   `:430` and never reassigned in the loop**, so this line re-stamps the phase-entry `T0` every
   segment, overwriting anything `updateDict` may have set.
3. `:496-497` `feedback = get_current_sps_feedback(t_now, params)`; `updateDict`.
4. `:507-509` `R1, Pb = compute_R1_Pb(R2, Eb, Lmech_total, v_mech_total, gamma_adia)`; both stored.
5. `:512-518` `T_for_sound = bubble_Tavg.value` if truthy **else the literal `1e6` K**;
   `c_sound = get_soundspeed(T_for_sound, params)`. Note `bubble_Tavg.value == 0` also falls to 1e6.
6. `:536-553` shell mass. Two clamps:
   * `isCollapse` true → `mShell = prev_mShell`, `mShell_dot = 0.0` (mass profile not called at all);
   * else compute `mShell_new, mShell_dot = get_mass_profile(R2, params, return_mdot=True, rdot=v2)`
     and if `prev_mShell > 0 and mShell_new < prev_mShell` → `mShell = prev_mShell`,
     `mShell_dot = 0.0`. **The shell mass is therefore monotone non-decreasing by construction.**
     Unlike the momentum twin there is no length-1-array unwrap here (§4.3).
7. `:558-559` `shell_props = shell_structure_pure(params)`; `updateDict`.
8. `:563-569` `P_HII = (μ_conv/μ_ion)·n_IF_Str·k_B·T_ion` if `include_PHII and n_IF_Str > 0` else 0;
   `F_HII = 4π R2² P_HII`. Both stored — then immediately recomputed and re-stored identically by
   `compute_forces_pure` at `:574`/`:580`.
9. `:571-585` `compute_forces_pure(...)`; 11 params keys written, plus
   `F_ram_wind := feedback.pdot_W`, `F_ram_SN := feedback.pdot_SN`.
10. `:592-600` `save_snapshot()`; `_snapshots_after_rCloud` bumped iff the save actually happened
    and `R2 > rCloud`.
11. `:605-611` `t_now >= tmax` → exit `reached_tmax`.
12. `:616` `snapshot = create_ODE_snapshot(params, shell_props)`; `:619` `values_before`.
13. `:621-640` integrate `[t_now, min(t_now+dt, tmax)]` with LSODA, `rtol 1e-6`, `atol 1e-8`,
    `max_step = 2e-4`, `min_step = 1e-6`, `events = ode_events`. The RHS closure captures
    the *mutable* `params` and the frozen `snapshot` and `c_sound`.
14. `:641-648` exception → `termination_reason = "solver_error: …"` and break;
    `not sol.success or len(sol.t)==0` → `"solver_failed: …"` and break. **Neither path sets
    `SimulationEndCode` / `SimulationEndReason` / `EndSimulationDirectly`** (S6-A-11).
15. `:653-671` event handling: state and `t_now` taken from `event_result`, appended, then
    `apply_event_result(params, …, state_keys=['R2','v2','Eb'])`, break.
16. `:676-687` otherwise take the last solver point and write back.
17. `:695-707` post-step mass update with the same monotone clamp — but **`shell_massDot` is not
    updated**, so it stays at the segment-start value while `shell_mass` moves.
18. `:709-717` adaptive dt: `max_dex > 0.1` → `dt /= 10^0.1` (floor `1e-3`); else `dt *= 10^0.1`
    (ceiling `5e-2`).
19. `:723-734` velocity override, only for `v2 < 0`: `|v2| > 150` → `dt = 5e-4`;
    elif `|v2| > 50` → `dt = min(dt, 1e-3)`. Units pc/Myr, i.e. ≈146.6 and ≈48.9 km/s
    (`v_kms2au = 1.0227`).
20. `:749-763` ram-dominance exit: `feedback_post` at the *new* `t_now`,
    `R1_post, Pb_post = compute_R1_Pb(...)` (**`R1_post` is never used**),
    `P_ram_post = pRam(R2, …)`, `ram_fraction = P_ram_post/(Pb_post + P_ram_post)`;
    `> 0.9` → break `ram_dominated`. `RAM_DOMINANCE_THRESHOLD = 0.9` is re-bound inside the loop
    body every iteration. Guarded by `P_total > 0`; if `Pb_post < 0` the fraction can exceed 1.
21. `:766-769` `Eb < 1e3` → break `energy_floor`.
22. `:772-776` `v2 < 0 and R2 < R2_prev` → `isCollapse = True`. **One-way latch, never cleared.**
23. `:779-826` the remaining exits: `reached_tmax`, `small_radius` (`R2 < coll_r`, only while
    `isCollapse`), `large_radius` (`R2 > stop_r`), and the dissolution timer
    (`shell_nMax < nISM` sustained for `stop_t_diss`). `t_diss_onset` is a **function-local**.

**Final reconciliation (`:833-862`, inside a bare `try`).** Recomputes feedback, `R1_f, Pb_f`,
shell structure, `P_HII_f`, `F_HII` (written twice, `:849` and `:854`), the full force set, then
`save_snapshot()`. It does **not** refresh `press_HII_in`, `F_ram_wind`, `F_ram_SN`, `n_IF`, `R_IF`,
so those keys in the final snapshot are one segment stale. On exception it logs a one-line warning
and the final snapshot is silently skipped. If the loop broke at `:611` (`reached_tmax`) — which is
immediately after `save_snapshot()` at `:593` with nothing in between — this block writes a
**second snapshot at the same `t_now` with the same state**.

**Return.** `TransitionPhaseResults(t, R2, v2, Eb, termination_reason, final_time)`. Note the
momentum phase does **not** consume this object; it reads state from `params`.

---

## 3. Momentum phase

### 3.1 `compute_forces_momentum_pure(R2, mShell, Lmech_total, v_mech_total, shell_props, params)` — `:206-294`

Same reads as §2.2 except `Lmech_total/v_mech_total` arrive as arguments instead of via `params`,
and `Pb` is not an argument at all.

    F_grav   = G·mShell·(mCluster + 0.5·mShell) / R2²                    :222   [identical to :284]
    P_ram    = pRam(R2, Lmech_total, v_mech_total)                       :225
    P_ext    = same construction as :306/:314                            :246,254 [identical]
    P_drive  = P_HII + P_ram          <-- no max(Pb, ·)                  :265
    F_ion_in = P_ext·4π·R2²                                              :268   [identical to :334]
    F_HII    = 4π·R2²·P_HII                                              :269   [identical to :335]
    F_ram    = P_ram·4π·R2²           <-- P_ram, not Pb                  :272
    F_rad    = same as :344-346                                          :278-280 [identical]

`nISM` is read at `:232` and never used, exactly as in the twin.

### 3.2 `MomentumODESnapshot` / `create_momentum_snapshot` — `:301-366`

18 fields captured at segment start. `F_rad` is *computed inside the snapshot builder*
(`:340-345`) with the same expression as `compute_forces_momentum_pure` — a third copy of the
radiation-force formula in this slice (`…transition:344`, `…momentum:278`, `…momentum:343`).

Of the 18 fields, **six are never read by `get_ODE_momentum_pure`**: `nISM`, `n_IF`,
`include_PHII`, `isCollapse`, `Lmech_total`, `v_mech_total`. The last two are the notable ones —
the RHS re-fetches feedback live (§3.3) and shadows them.

### 3.3 `get_ODE_momentum_pure(t, y, snapshot, params)` — `:373-454`

**The momentum equation actually integrated.** With `R2 ← max(R2, 1e-10)` (`:398`),
`m ← max(m_snap, 1e-10)` (`:415`), `ṁ ← snapshot.mShell_dot` (unclamped), and
`L(t), v_w(t) = get_current_sps_feedback(t, params)` (`:407`, evaluated **live at every RHS call**):

    dR2/dt = v2                                                                          :451

              4π R2² · [ P_HII⁰ + P_ram(R2, L(t), v_w(t)) − P_ext(rShell⁰) ]
    dv2/dt = ───────────────────────────────────────────────────────────────
                                        m⁰
              − (ṁ⁰ · v2)/m⁰  − G·(M_cl + m⁰/2)/R2²  + F_rad⁰/m⁰

where superscript ⁰ marks a quantity frozen at the segment start, and

    P_ext(rShell⁰) = (μ_conv/μ_ion)·n(rShell⁰)·k_B·T_ion  · [shell_fAbsorbedIon < 1]
                   + PISM·k_B                             · [rShell⁰ >= rCloud]
    F_rad⁰         = 0                                     if isDissolved
                   = f_absW,tot·(Lbol/c)·(1 + Σ_{τ/κ}·κ_IR)   otherwise

Written as a momentum budget, `:452` is
`m·v̇ = F_pressure − ṁ·v − F_grav + F_rad`, i.e. `d(mv)/dt = ΣF` with the swept-up-mass drag on the
left moved right. **Signs are mutually consistent**: outward pressure `+`, inward ambient pressure
`−P_ext` inside the same bracket, gravity `−`, radiation `+`, mass loading `−ṁv` (decelerating for
`ṁ > 0, v > 0`). Stored diagnostics `F_grav` and `F_ion_in` are positive magnitudes whose sign is
applied at the point of use — consistent between the two files.

**Term-by-term double-counting check.** `P_HII` enters exactly once (inside `P_drive`); the
diagnostic `F_HII = 4πR2²P_HII` is *not* added again. `P_ram` enters exactly once (inside
`P_drive`); the diagnostics `F_ram`, `F_ram_wind = pdot_W`, `F_ram_SN = pdot_SN` are not added
again. `P_ext` enters exactly once (as `−P_ext` in the bracket); `F_ion_in` is not subtracted
again. **No term is double-counted and none of the five force channels is dropped.**

**Dimensions.** `4πR2²·P` = pc²·M☉pc⁻¹Myr⁻² = M☉·pc/Myr² ✓; `ṁ·v` = (M☉/Myr)(pc/Myr) ✓;
`G·M²/R²` ✓; `F_rad` ✓; all divided by `m` [M☉] → pc/Myr² ✓ = `d(pc/Myr)/dMyr`. Balanced.

**Control flow that changes the maths.**
* `R2 = max(R2, 1e-10)` (`:398`) — the *state* is not clamped, only the copy used in `R2²`. At the
  clamp, `F_grav ∝ 1/R2² = 1e20`, so the RHS returns a finite but astronomically large inward
  acceleration rather than an `inf`.
* `mShell = max(mShell, 1e-10)` (`:415`) — divides the whole RHS; with a non-zero `ṁ⁰` the drag
  term becomes `ṁ⁰v/1e-10`.
* `FABSi < 1.0` / `try/except` / `rShell >= rCloud` — same three branches and the same silent
  `P_ext = 0.0` on exception as the twin.
* `P_ext` is evaluated at the **frozen** `snapshot.rShell` while the force it enters uses the
  **live** `R2` — a frozen/live mismatch inside one product (S6-A-06).

**Literals in arithmetic:** `1e-10` (twice), `0.5`, `1.0` (in `FABSi < 1.0` and in `1 + τ·κ`),
`0.0`, `FOUR_PI = 4.0·π`.

### 3.4 `run_phase_momentum(params)` — `:461-931`

Structurally line-for-line the transition loop with the energy machinery removed. Deltas that
change what is computed:

* `:507-511` entry: reads `R2, v2, T0`; **`params['Eb'].value = 0.0`** — whatever energy the
  transition phase exited with is discarded here and re-zeroed every iteration at `:571`.
* `:572` re-stamps the phase-entry `T0` each segment (same pattern as `…transition:491`), so `T0`
  is a frozen constant for the whole momentum phase while `Eb ≡ 0`.
* `:585` and `:667` both set `params['Pb'] = pRam(R2, feedback.Lmech_total, feedback.v_mech_total)`
  from the *same* `feedback` object — two writes of an identical value.
* `:588` `params['R1'] = R2` (zero-thickness bubble). Any downstream consumer forming `R2 − R1`
  gets exactly 0.
* `:606-613` and `:780-781` unwrap length-1 arrays from `get_mass_profile`; the transition twin
  does not (S6-A-03).
* `:825-826` the same one-way `isCollapse` latch.
* `:886-894` final reconciliation: refreshes feedback, `Pb`, `R1`, shell structure, then
  `save_snapshot()` — but **does not recompute any force**, so the last momentum snapshot carries
  `F_grav / F_ram / F_rad / F_HII / F_ion_in / P_drive / press_HII_in` from the *previous* segment
  while `R2, v2, t_now` are final. The transition twin does recompute them (`…transition:850-859`).
* `:895-908` the exception handler extracts the traceback frame and reports
  `type(e).__name__: msg at file:lineno`; the transition twin logs only `f"…failed: {e}"`
  (S6-A-04).

---

## 4. Direct twin-vs-twin comparison

### 4.1 `F_ram` — the two files compute different quantities under one name

    phase1c  run_transition_phase.py:338   F_ram = Pb    · 4π · R2²
    phase2   run_momentum_phase.py:272     F_ram = P_ram · 4π · R2²

In the momentum runner `Pb` is *defined* to be `pRam(...)` (`:585`, `:667`), so there the two forms
coincide and `F_ram` is self-consistent with its siblings: `P_ram` in the same
`ForceProperties`, and `F_ram_wind + F_ram_SN = pdot_W + pdot_SN` (which equals `4πR²·pRam` under
the usual `pRam = ṗ/4πR²`). In the transition runner `Pb` comes from `compute_R1_Pb(R2, Eb, …)`
and is the hot-bubble pressure, generally ≫ `P_ram` during energy-driven evolution — so
`params['F_ram'] ≠ 4πR2²·params['P_ram']` there, and the output column `F_ram` changes meaning
across the phase boundary. **The transition runner is the outlier** (its own `P_ram` field
disagrees with its own `F_ram`). I cannot tell from the code whether phase1c *intends* to record
the bubble-pressure force under this key.

### 4.2 `P_drive`

    phase1c :331   P_drive = max(Pb, P_HII + P_ram)
    phase2  :265   P_drive =          P_HII + P_ram      (and :445, identically, inside the ODE)

Consistent under the momentum phase's `Pb ≡ P_ram` only when `P_HII ≥ 0` (always true here), so
`max(P_ram, P_HII + P_ram) = P_HII + P_ram` ✓ — the two agree *in the momentum regime*. In the
transition regime `max` selects `Pb`. Diagnostic-only in phase1c (nothing reads it); load-bearing
in phase2 (`:445` feeds the ODE).

### 4.3 Length-1-array defence

    phase2  :610-613, :780-781   if hasattr(x, '__len__') and len(x)==1: x = float(x[0])
    phase1c :545,      :702      (absent)

Same call, same arguments (`mass_profile.get_mass_profile(R2, params, return_mdot=…, rdot=v2)`).
The momentum runner has a hardening the transition runner lacks. If `get_mass_profile` ever returns
a length-1 array, `params['shell_mass'].value` in phase1c becomes an ndarray; comparisons still work
for length 1, so the failure would surface downstream (formatting, snapshot dtype), not here.
**The transition runner is the outlier.**

### 4.4 Exception reporting at the phase boundary

    phase1c :861-862   logger.warning(f"Phase-boundary reconciliation failed: {e}")
    phase2  :899-908   + type(e).__name__, '<no message>' fallback, and 'at file:lineno'

A message-quality fix present in one twin only; the transition runner is the outlier.

### 4.5 Final-reconciliation content

phase1c recomputes the force set before the last snapshot (`:850-859`); phase2 does not
(`:886-894`). Neither refreshes `press_HII_in`/`F_ram_wind`/`F_ram_SN`. **The momentum runner is
the outlier** (it drops more).

### 4.6 Everything that *does* agree exactly

`F_grav` (`:284` ≡ `:222` ≡ `:418`), `P_ext` construction incl. the `FABSi < 1.0` branch, the
`rShell >= rCloud` `PISM·k_B` addition, `F_ion_in`, `F_HII`, `F_rad` (all three copies), the
`P_HII = (μ_conv/μ_ion)·n_IF_Str·k_B·T_ion` formula (`:564` ≡ `:634`), the mass-monotonicity clamp,
the adaptive-dt controller and every one of its constants, the velocity-based dt override, all four
`ODE_*` tolerances, and the entire termination-condition block (`reached_tmax`, `small_radius`,
`large_radius`, dissolution timer, `stop_at_rCloud`).

### 4.7 Hand-off: does phase1c's output match phase2's input?

phase2 entry reads `t_now, stop_t, R2, v2, T0`, then in the first segment reads
`shell_mass, isCollapse, include_PHII, rCloud, nISM, stop_r, coll_r, stop_t_diss,
_snapshots_after_rCloud, stop_at_rCloud_nSnap`. phase1c writes all of them. **The state hand-off is
complete.** Four things are nonetheless *silently altered or lost* at the boundary:

1. `Eb` → forced to `0.0` (`:511`). phase1c can exit `ram_dominated` (`:763`) with `Eb` still far
   above `ENERGY_FLOOR`; that energy is discarded, not converted.
2. `R1` → forced to `R2` (`:588`); `Pb` → forced to `pRam` (`:585`). Both are overwrites of
   physically different phase1c quantities, and both happen *before* `shell_structure_pure` reads
   them, so the shell structure at the first momentum segment sees a different `Pb` than the last
   transition segment did at the same `R2`.
3. `t_diss_onset` is a function-local in both runners (`…transition:449`, `…momentum:531`). A
   dissolution clock that had run for, say, `0.9·stop_t_diss` in phase1c **restarts from zero** in
   phase2. Nothing in `params` records it.
4. `dt_segment` resets to `DT_SEGMENT_INIT = 2e-3` even if phase1c ended in collapse mode with
   `dt = 5e-4`; the first momentum segment is 4× longer than the last transition segment.

`isCollapse` *is* carried over — and being a latch with no reset path in either file, it freezes
`shell_mass` for the remainder of the run once set.

---

## 5. The adaptive-timestep controller is nearly inert

`values_before` is sampled at `…transition:619` / `…momentum:701`, i.e. **after** all of
`updateDict(feedback)`, `compute_R1_Pb`, `shell_structure_pure`, `compute_forces_*` have run.
`values_after` is sampled at `…transition:709` / `…momentum:788`, **before** the next iteration
re-runs any of them. Between those two points the only params keys written are:

* transition: `t_now`, `R2`, `v2`, `Eb` (`:684-687`) and `shell_mass` (`:707`);
* momentum: `t_now`, `R2`, `v2` (`:762-764`) and `shell_mass` (`:786`).

So of the 30 `ADAPTIVE_MONITOR_KEYS`, **26 are provably unchanged between the two samples** and
contribute exactly 0 dex: `T0, Pb, R1, pdot_SN, Lmech_SN, pdotdot_total, cool_delta, cool_beta,
bubble_*(7), shell_massDot, shell_n0, shell_nMax, shell_thickness, shell_tauKappaRatio,
shell_fIonisedDust, rShell, F_grav, F_ram, F_ram_wind, F_ram_SN, F_ion_in, F_HII, F_rad, F_ISM`.
In the momentum phase `Eb` is additionally always `0.0` and therefore skipped by the `== 0` guard.
The controller reduces to `max(|Δlog10 R2|, |Δlog10 v2|, |Δlog10 Eb|, |Δlog10 m_shell|)` (transition)
and `max(|Δlog10 R2|, |Δlog10 v2|, |Δlog10 m_shell|)` (momentum). `F_ISM` is additionally never
written by either file at all.

---

## 6. Complete numeric-literal inventory

| Literal | File:line | Expression it sits in |
|---|---|---|
| `2e-3` | T:93 / M:86 | `dt_segment` initial value |
| `1e-3` | T:94 / M:87 | dt floor; also `min(dt, DT_SEGMENT_MIN)` for `50 < |v2|` |
| `5e-2` | T:95 / M:88 | dt ceiling |
| `5000` / `10000` | T:96 / M:89 | `segment_count < MAX_SEGMENTS` (**differs**) |
| `1e3` | T:97 | `Eb < ENERGY_FLOOR`; also `build_transition_phase_events(energy_floor=)` |
| `4.0·π` | T:98 / M:90 | `F_ion_in`, `F_HII`, `F_ram`, `F_pressure` |
| `0.1` | T:101 / M:93 | `max_dex_change > ADAPTIVE_THRESHOLD_DEX` |
| `10**0.1` (=1.2589) | T:102 / M:94 | `dt *= / /=` |
| `50.0`, `150.0` | T:106-107 / M:98-99 | `|v2|` thresholds, pc/Myr (≈48.9, ≈146.6 km/s) |
| `5e-4` | T:108 / M:100 | collapse-mode dt |
| `1e-6`, `1e-8` | T:132-133 / M:124-125 | `rtol`, `atol` |
| `1e-6` | T:134 / M:126 | LSODA `min_step` |
| `1e-3/5` = `2e-4` | T:135 / M:127 | LSODA `max_step` |
| `1.0` | T:154 / M:146 | dex assigned on a sign flip |
| `0.5` | T:284 / M:222,418 | `(mCluster + 0.5·mShell)` |
| `1.0` | T:300 / M:241,425 | `if FABSi < 1.0` |
| `1.0` | T:346 / M:280,345 | `(1.0 + τκ_ratio·κ_IR)` |
| `1e6` | T:516 | fallback `T_for_sound` [K] when `bubble_Tavg` falsy |
| `4.0·π` | T:568 / M:638 | `F_HII` recomputed inline |
| `0.9` | T:749 | `RAM_DOMINANCE_THRESHOLD`, re-bound each iteration |
| `1e-10` | M:398, M:415 | `max(R2, ·)`, `max(mShell, ·)` inside the RHS |
| `0.0` | M:511, M:571 | `params['Eb']` forced to zero |

---

## 7. Miscellaneous hygiene observed

* Dead imports: `scipy.optimize` (T:51), `Tuple` (T:53), `unit_conversions as cvt` (T:57 **and**
  M:57 — unused in both files).
* `density_profile` is imported at module scope in the momentum file (M:69) but inside the function
  body in the transition file (T:301).
* `ForceProperties` is defined twice (T:254 / M:189) with identical fields; the two are distinct
  classes.
* `R1_post` (T:752) computed, never used.
* `v2_from_alpha` (T:391) computed, used only in an f-string.
* `params['F_HII']` written twice per segment with the same value (T:569 then T:574; M:639 then
  M:652) and twice more in T's final block (T:849, T:854). `params['P_HII']` likewise (T:567/T:580,
  M:637/M:658).
* `if tmax is not None` (T:605, T:779; M:687, M:832) can never be False — `tmin >= tmax` at T:407 /
  M:488 would already have raised `TypeError`.
* Both runners can exit with `termination_reason == "unknown"` (T:870 / M:916) — only a
  `logger.warning`, no end code.

---

```json
[
  {
    "id": "S6-A-01",
    "file": "trinity/phase1c_transition/run_transition_phase.py",
    "line": 338,
    "class": "divergence",
    "severity": "S3",
    "claim": "The output key F_ram means two different physical quantities in the two sibling runners: 4*pi*R2^2*Pb in phase1c, 4*pi*R2^2*P_ram in phase2.",
    "evidence": "run_transition_phase.py:338 'F_ram = Pb * FOUR_PI * R2**2' where Pb comes from compute_R1_Pb(R2, Eb, ...) at :507. run_momentum_phase.py:272 'F_ram = P_ram * FOUR_PI * R2**2' where P_ram = pRam(R2, Lmech_total, v_mech_total) at :225. In phase1c the same ForceProperties object also carries P_ram=pRam(...) (:329, :358), so F_ram != 4*pi*R2^2*P_ram inside one object; in phase2 Pb is defined as pRam (:585, :667) so the two forms coincide there.",
    "expected": "Either both compute 4*pi*R2^2*P_ram (matching the sibling diagnostics F_ram_wind=pdot_W and F_ram_SN=pdot_SN set at :584-585 / :662-663), or phase1c stores the bubble-pressure force under a distinct key.",
    "failure_scenario": "Any post-processing that plots or sums the F_ram column across the phase boundary sees a discontinuous jump of order Pb/P_ram (large during energy-driven evolution) that is an artefact of the key, not of the physics. A force-budget closure check F_ram == F_ram_wind + F_ram_SN passes in phase2 and fails in phase1c.",
    "repro": "Run param/simple_cluster.param and compare params['F_ram'] against 4*pi*R2**2*params['P_ram'] and against params['F_ram_wind']+params['F_ram_SN'] in the last transition snapshot and the first momentum snapshot.",
    "confidence": "high"
  },
  {
    "id": "S6-A-02",
    "file": "trinity/phase2_momentum/run_momentum_phase.py",
    "line": 511,
    "class": "state",
    "severity": "S2",
    "claim": "The momentum phase forces Eb to exactly 0.0 at entry and again at every segment, silently discarding whatever bubble energy the transition phase exited with.",
    "evidence": "run_momentum_phase.py:511 \"params['Eb'].value = 0.0\" at entry and :571 inside the loop. The transition phase can break at :763 ('ram_dominated', ram_fraction > 0.9) with Eb arbitrarily far above ENERGY_FLOOR=1e3 -- the ram-dominance test at :756-759 does not require Eb to be small, only that P_ram/(Pb+P_ram) > 0.9.",
    "expected": "Either the ram-dominated exit should additionally require Eb below the floor, or the hand-off should record/convert the residual energy rather than zeroing it.",
    "failure_scenario": "A configuration whose ram fraction crosses 0.9 while the bubble is still energetic hands off with a discontinuous loss of Eb; the total energy budget of the run is not conserved across the phase boundary and the discontinuity is invisible in the logs.",
    "repro": "Log Eb at the transition-phase break and at momentum entry for a wind-dominated config; compare with the Eb trace on either side of the boundary in dictionary.jsonl.",
    "confidence": "high"
  },
  {
    "id": "S6-A-03",
    "file": "trinity/phase1c_transition/run_transition_phase.py",
    "line": 545,
    "class": "divergence",
    "severity": "S2",
    "claim": "The length-1-array unwrap guard applied to mass_profile.get_mass_profile results exists only in the momentum runner; the transition runner calls the same function with the same arguments and stores the raw return value.",
    "evidence": "run_momentum_phase.py:610-613 unwraps mShell_new and mShell_dot, and :780-781 unwraps mShell_post. run_transition_phase.py:545 and :702 make the identical calls with no unwrap, then write straight into params['shell_mass'].value / params['shell_massDot'].value at :552-553 and :707.",
    "expected": "Both runners handle the return type identically; if the guard is needed in one it is needed in the other.",
    "failure_scenario": "If get_mass_profile returns a length-1 ndarray for scalar input, phase1c stores ndarrays into shell_mass/shell_massDot. Comparisons at :547 and :704 still work for length 1, so the ndarray propagates silently into create_ODE_snapshot, the snapshot writer and the adaptive monitor before failing (or serialising oddly) somewhere downstream.",
    "repro": "Assert isinstance(params['shell_mass'].value, float) after the first transition segment and after the first momentum segment.",
    "confidence": "high"
  },
  {
    "id": "S6-A-04",
    "file": "trinity/phase1c_transition/run_transition_phase.py",
    "line": 861,
    "class": "divergence",
    "severity": "S4",
    "claim": "The phase-boundary reconciliation exception handler is enriched with exception type and source location in the momentum runner only.",
    "evidence": "run_momentum_phase.py:899-908 extracts the last traceback frame and logs 'type(e).__name__: msg at file:lineno' with a '<no message>' fallback. run_transition_phase.py:861-862 logs only f\"Phase-boundary reconciliation failed: {e}\".",
    "expected": "Identical diagnostics in both twins.",
    "failure_scenario": "An empty-message exception (e.g. a bare KeyError with an unprintable key, or IndexError()) in the transition reconciliation logs 'Phase-boundary reconciliation failed: ' with no information, and the final transition snapshot is silently missing.",
    "repro": "Force shell_structure_pure to raise inside the transition final block and compare the two log lines.",
    "confidence": "high"
  },
  {
    "id": "S6-A-05",
    "file": "trinity/phase2_momentum/run_momentum_phase.py",
    "line": 886,
    "class": "divergence",
    "severity": "S3",
    "claim": "The momentum phase's final reconciliation refreshes Pb, R1 and shell structure but never recomputes the force set, so the last momentum snapshot pairs final R2/v2/t with forces from the previous segment.",
    "evidence": "run_momentum_phase.py:886-894 calls get_current_sps_feedback, pRam, shell_structure_pure, save_snapshot -- no call to compute_forces_momentum_pure. run_transition_phase.py:850-859 does call compute_forces_pure and rewrites F_grav/F_ion_in/F_HII/F_ram/F_rad/P_HII/P_drive/P_ram before its save_snapshot at :860.",
    "expected": "Both final blocks refresh the same set of derived keys, or neither does.",
    "failure_scenario": "The final row of a run that ends in the momentum phase reports forces evaluated at the previous segment's R2 and shell state -- an O(v2*dt) inconsistency, up to 5e-2 Myr of drift -- while R2, v2, rShell and shell_mass in the same row are current.",
    "repro": "Compare the last two rows of dictionary.jsonl for a momentum-terminating run: F_grav should track G*m*(M+m/2)/R2^2 with the row's own R2, and will not.",
    "confidence": "high"
  },
  {
    "id": "S6-A-06",
    "file": "trinity/phase2_momentum/run_momentum_phase.py",
    "line": 448,
    "class": "numerical",
    "severity": "S2",
    "claim": "Inside the momentum RHS the external pressure is evaluated at the frozen segment-start shell radius while the area it multiplies uses the live integration radius.",
    "evidence": "run_momentum_phase.py:424 'rShell = snapshot.rShell'; :427 get_density_profile(np.array([rShell]), params); :437 'if rShell >= snapshot.rCloud'; then :448 'F_pressure = FOUR_PI * R2**2 * (P_drive - P_ext)' with R2 taken from the live state y at :397. P_ram at :421 and F_grav at :418 do use the live R2, and the feedback at :407 is live in t -- so the RHS mixes live and frozen quantities within one product.",
    "expected": "Either all radius-dependent factors in one force use the same radius, or the frozen/live split is deliberate and bounded by the segment length.",
    "failure_scenario": "With dt_segment at its ceiling 5e-2 Myr and v2 ~ 10-100 pc/Myr the shell moves 0.5-5 pc within a segment; in a steep density profile n(rShell_frozen) can be wrong by a large factor, so the confining term -4*pi*R2^2*P_ext is systematically stale in one direction (lagging behind an expanding shell) rather than randomly.",
    "repro": "Instrument get_ODE_momentum_pure to log rShell vs the live R2 at the last RHS call of each segment on a steep-profile config.",
    "confidence": "medium"
  },
  {
    "id": "S6-A-07",
    "file": "trinity/phase2_momentum/run_momentum_phase.py",
    "line": 350,
    "class": "deadcode",
    "severity": "S4",
    "claim": "Six of the eighteen MomentumODESnapshot fields are never read by the RHS, including Lmech_total and v_mech_total, which are shadowed by a live feedback call inside the RHS.",
    "evidence": "create_momentum_snapshot populates nISM (:353), rCloud (:355) [used], n_IF (:359), include_PHII (:360), isCollapse (:365), Lmech_total (:350), v_mech_total (:351). get_ODE_momentum_pure reads only G, mCluster, k_B, FABSi, F_rad, mShell, mShell_dot, rShell, rCloud, PISM, TShell_ion, P_HII -- and at :407-409 recomputes Lmech_total/v_mech_total from get_current_sps_feedback(t, params), discarding the snapshot copies.",
    "expected": "A snapshot that is captured for the purpose of freezing the RHS should either be used for all frozen inputs, or the unused fields dropped. As written the function named '_pure' depends on the mutable params dict at every RHS evaluation.",
    "failure_scenario": "The frozen snapshot.Lmech_total is what the diagnostic force block uses (compute_forces_momentum_pure receives feedback.Lmech_total from :644-645), while the integrated equation uses a live time-varying value, so the recorded F_ram/P_ram do not correspond to the ram pressure actually integrated over the segment.",
    "repro": "grep for snapshot. attribute reads inside get_ODE_momentum_pure and compare against the MomentumODESnapshot field list.",
    "confidence": "high"
  },
  {
    "id": "S6-A-08",
    "file": "trinity/phase2_momentum/run_momentum_phase.py",
    "line": 826,
    "class": "state",
    "severity": "S2",
    "claim": "isCollapse is a one-way latch with no reset path in either runner, and it is inherited across the phase boundary; once set, the shell mass is frozen for the rest of the simulation even if the shell re-expands.",
    "evidence": "Set to True at run_transition_phase.py:773 and run_momentum_phase.py:826 under 'v2 < 0 and R2 < R2_prev'. No assignment of False appears in either file. When true, run_transition_phase.py:540-543 and run_momentum_phase.py:601-604 bypass get_mass_profile entirely and hold mShell = prev_mShell, mShell_dot = 0.0.",
    "expected": "Either a reset when the shell resumes expanding (v2 > 0 and R2 > R2_prev), or a latch that is documented as terminal. Note the separate monotone clamp (:547-549 / :616-618) already reproduces the same behaviour while the shell is actually contracting, so the latch only changes the result after re-expansion.",
    "failure_scenario": "A shell that contracts transiently (one segment with v2<0 and R2 decreasing) and then re-expands past its previous radius accretes no further mass for the remainder of the run; mShell_dot is pinned at 0 so the -mdot*v2 drag term in the momentum equation (run_momentum_phase.py:452) vanishes permanently, biasing v2 upward.",
    "repro": "A config that oscillates once: log isCollapse and shell_mass; confirm shell_mass stops tracking mass_profile(R2) after the first contraction segment.",
    "confidence": "high"
  },
  {
    "id": "S6-A-09",
    "file": "trinity/phase2_momentum/run_momentum_phase.py",
    "line": 531,
    "class": "state",
    "severity": "S3",
    "claim": "The dissolution clock t_diss_onset is a function-local in both runners and is reset to inf at the phase boundary, so accumulated time below nISM is lost when the transition phase hands off to the momentum phase.",
    "evidence": "run_transition_phase.py:449 't_diss_onset = np.inf' and run_momentum_phase.py:531 identically; the timer test is '(t_now - t_diss_onset) >= params[\"stop_t_diss\"].value' at :813 / :866. Nothing writes the onset time into params, and the momentum phase re-initialises it unconditionally.",
    "expected": "The onset time should live in params (like isCollapse and isDissolved do) so the clock survives the phase change.",
    "failure_scenario": "A shell that has been below nISM for 0.99*stop_t_diss when the transition phase exits on ram_dominated must wait a further full stop_t_diss in the momentum phase before being declared dissolved -- the run continues up to stop_t_diss longer than intended.",
    "repro": "Config with a small stop_t_diss and a ram-dominated transition exit; compare the dissolution time against a single-phase run.",
    "confidence": "high"
  },
  {
    "id": "S6-A-10",
    "file": "trinity/phase1c_transition/run_transition_phase.py",
    "line": 491,
    "class": "state",
    "severity": "S3",
    "claim": "Both runners re-stamp params['T0'] every segment from a local captured once at phase entry, clobbering any value written by updateDict or the shell/bubble modules.",
    "evidence": "run_transition_phase.py:430 'T0 = params[\"T0\"].value' -- the local T0 is never reassigned anywhere in the loop -- and :491 'params[\"T0\"].value = T0' executes each segment. Identically run_momentum_phase.py:509 and :572. T0 is also listed in ADAPTIVE_MONITOR_KEYS (:114 / :106), where it can therefore never contribute a non-zero dex.",
    "expected": "Either T0 is a genuine state variable and should be evolved/read back, or it should not be re-written at all.",
    "failure_scenario": "Anything that updates T0 between segments (feedback updateDict at :497, shell_structure_pure at :558) has its value overwritten by the phase-entry constant on the next iteration, so every snapshot in the phase reports the same T0 and the adaptive controller never reacts to it.",
    "repro": "Log params['T0'] immediately after updateDict and again after the next :491 write.",
    "confidence": "high"
  },
  {
    "id": "S6-A-11",
    "file": "trinity/phase1c_transition/run_transition_phase.py",
    "line": 646,
    "class": "silent-failure",
    "severity": "S2",
    "claim": "Solver failure and solver exception exit the segment loop without setting SimulationEndCode, SimulationEndReason or EndSimulationDirectly, so a failed integration is indistinguishable from a normal phase completion to the caller.",
    "evidence": "run_transition_phase.py:641-644 (exception) and :646-648 (not sol.success / empty sol.t) set only the local termination_reason string and break. Every other exit path in the same loop (:472-479, :606-609, :780-783, :791-793, :800-802, :816-818) sets all three params keys. Identical structure at run_momentum_phase.py:723-730. The same applies to the 'max_segments' and 'unknown' fall-through at :870 / :916.",
    "expected": "A failed solve should set a distinct SimulationEndCode and EndSimulationDirectly, or raise.",
    "failure_scenario": "LSODA fails mid-transition; R2/v2/Eb retain the pre-segment values, the final reconciliation block writes a normal-looking snapshot, and the driver proceeds into the momentum phase (or the run terminates) with no end code recorded -- the output file looks like a clean run.",
    "repro": "Force sol.success=False (e.g. min_step > max_step) and inspect params['SimulationEndCode'] after run_phase_transition returns.",
    "confidence": "high"
  },
  {
    "id": "S6-A-12",
    "file": "trinity/phase1c_transition/run_transition_phase.py",
    "line": 619,
    "class": "other",
    "severity": "S3",
    "claim": "The adaptive-timestep controller monitors 30 keys but 26 of them are provably unchanged between the two sampling points, so the step control reduces to R2, v2, Eb and shell_mass (R2, v2, shell_mass in the momentum phase).",
    "evidence": "values_before is taken at run_transition_phase.py:619, after every derived-quantity update in the segment; values_after at :709. The only params writes in between are :684-687 (t_now, R2, v2, Eb) and :707 (shell_mass). Same at run_momentum_phase.py:701 / :788 with writes at :762-764 and :786. Every F_*, P_*, bubble_*, shell_* (except shell_mass), T0, Pb, R1, cool_* key in ADAPTIVE_MONITOR_KEYS therefore contributes exactly 0 dex. 'F_ISM' (:128 / :120) is never written by either file at all. In the momentum phase 'Eb' is pinned to 0.0 and is skipped by the '== 0' guard in compute_max_dex_change.",
    "expected": "Either sample values_after after the next segment's derived-quantity recomputation, or trim the key list to what can actually change.",
    "failure_scenario": "A run where the cooling/bubble structure (cool_delta, bubble_LTotal, ...) changes by many dex in a segment while R2, v2, Eb move smoothly keeps growing dt by 10^0.1 per segment up to 5e-2 Myr, exactly in the stiff regime the monitor list was assembled to catch.",
    "repro": "Log which key attains the max in compute_max_dex_change over a transition run; it will only ever be R2, v2, Eb or shell_mass.",
    "confidence": "high"
  },
  {
    "id": "S6-A-13",
    "file": "trinity/phase2_momentum/run_momentum_phase.py",
    "line": 398,
    "class": "numerical",
    "severity": "S3",
    "claim": "The momentum RHS clamps R2 and mShell to 1e-10 locally without clamping the integrator state, so the reported RHS silently corresponds to a different radius/mass than the state being integrated, and F_grav at the clamp is ~1e20*G*M.",
    "evidence": "run_momentum_phase.py:398 'R2 = max(R2, 1e-10)' rebinds the local unpacked from y; :415 'mShell = max(mShell, 1e-10)'. The returned rd = v2 (:451) is unaffected, so the solver state R2 can continue to decrease past 1e-10 while F_grav = G*mShell*(mCluster+0.5*mShell)/1e-20 (:418) and P_ram = pRam(1e-10, ...) (:421) are evaluated at the clamp. mShell_dot is not clamped, so the drag term mShell_dot*v2/1e-10 can dominate.",
    "expected": "Either terminate via an event before R2 reaches the clamp (the small_radius check at :842 only runs between segments, never inside the RHS), or clamp consistently.",
    "failure_scenario": "During a collapse the state R2 crosses zero inside a segment; the RHS returns a finite huge inward acceleration instead of failing, LSODA reduces its step and the segment burns to min_step, and the recorded trajectory near R2->0 is governed entirely by the clamp value rather than by physics.",
    "repro": "A collapsing config with coll_r below 1e-10 pc, or instrument the RHS to log when either max() is active.",
    "confidence": "medium"
  },
  {
    "id": "S6-A-14",
    "file": "trinity/phase1c_transition/run_transition_phase.py",
    "line": 240,
    "class": "regime",
    "severity": "S3",
    "claim": "In the fallback branch (c_sound <= 0 or R2 <= 0) the sound-crossing loss is set to 0.0 and the subsequent min() then clamps dEb/dt to be non-positive, silently deleting any energy injection.",
    "evidence": "run_transition_phase.py:237-240 sets Ed_soundcrossing = -Eb*c_sound/R2 or 0.0; :245 'Ed = min(Ed_energy_balance, Ed_soundcrossing)'. With the fallback value 0.0 this becomes Ed = min(Ed_bal, 0), i.e. positive Ed_bal is replaced by exactly 0 rather than by an unmodified energy balance. c_sound comes from :517 with the literal 1e6 K fallback at :516, so c_sound <= 0 requires get_soundspeed to return <= 0.",
    "expected": "If the sound-crossing term is unavailable the energy equation should fall back to the unmodified balance (Ed = Ed_energy_balance), not to a one-sided clamp.",
    "failure_scenario": "If get_soundspeed ever returns 0 (e.g. bubble_Tavg falsy and the 1e6 K fallback also failing) the bubble energy is frozen from above for the whole segment, and Eb decays only through Ed_bal -- a different equation from the one the phase is meant to integrate, with no log line.",
    "repro": "Force c_sound=0 for one segment and compare the Eb trace against the unmodified energy-balance integration.",
    "confidence": "medium"
  },
  {
    "id": "S6-A-15",
    "file": "trinity/phase1c_transition/run_transition_phase.py",
    "line": 300,
    "class": "regime",
    "severity": "S3",
    "claim": "P_ext is a step function of shell_fAbsorbedIon: the full ionized-gas pressure for FABSi = 1-eps and exactly zero at FABSi = 1, with no (1-FABSi) weighting. Identical in both runners.",
    "evidence": "run_transition_phase.py:299-310 'if FABSi < 1.0: P_ext = (mu_convert/mu_ion_shell)*n(rShell)*k_B*TShell_ion else: P_ext = 0.0'. Identically run_momentum_phase.py:240-250 and, inside the integrated RHS, :425-434. The magnitude of P_ext does not depend on FABSi at all.",
    "expected": "A continuous dependence on the escaping fraction, or a documented reason the discontinuity is acceptable.",
    "failure_scenario": "A run in which FABSi crosses 1.0 between segments sees the confining term -4*pi*R2^2*P_ext appear or vanish discontinuously in the integrated momentum equation (run_momentum_phase.py:448), producing a kink in v2 that is a property of the branch, not the physics. The same try/except at :247 also maps any density-profile failure to P_ext = 0.0 with no log.",
    "repro": "Log shell_fAbsorbedIon and P_ext per segment for a config that saturates absorption.",
    "confidence": "medium"
  },
  {
    "id": "S6-A-16",
    "file": "trinity/phase1c_transition/run_transition_phase.py",
    "line": 314,
    "class": "units",
    "severity": "S3",
    "claim": "PISM is multiplied by k_B before being added to P_ext, which balances only if PISM is stored as P/k_B (K*pc^-3 in internal units) rather than as an internal-unit pressure. The two sibling terms in the same sum are built differently.",
    "evidence": "run_transition_phase.py:306 builds P_ext as (mu_convert/mu_ion_shell)*n*k_B*T -- number density times k_B times temperature -- then :313-314 adds 'PISM * k_B' with no density and no temperature and no mu ratio. Identically run_momentum_phase.py:254 and, inside the RHS, :438. From unit_conversions.py, k_B in internal units is 5.26e-44 x its cgs value and pressure is M_sun/(pc*Myr^2); the module also defines Pb_au2_KcmInv = Pb_au2cgs/K_B_CGS, i.e. the codebase does sometimes carry pressures divided by k_B.",
    "expected": "Confirm from the .param schema that PISM is declared in K*cm^-3 (P/k_B). If it is declared as a pressure, the *k_B factor is wrong by ~1e-44 and the ambient confinement term vanishes.",
    "failure_scenario": "If PISM is a pressure, every shell outside rCloud loses its ISM confinement entirely (the added term underflows to ~0), so shells escaping the cloud are never decelerated by the ambient medium.",
    "repro": "Check the declared unit string for PISM in trinity/_input/ and compare params['PISM'].value*k_B against nISM*k_B*T_ISM for the default config.",
    "confidence": "low"
  },
  {
    "id": "S6-A-17",
    "file": "trinity/phase1c_transition/run_transition_phase.py",
    "line": 291,
    "class": "deadcode",
    "severity": "S4",
    "claim": "nISM is read from params in both compute_forces functions and never used; several other computed values are also discarded.",
    "evidence": "run_transition_phase.py:291 'nISM = params[\"nISM\"].value' -- nISM appears nowhere else in compute_forces_pure. Identically run_momentum_phase.py:232. Also unused: R1_post (run_transition_phase.py:752, only Pb_post is consumed at :756), v2_from_alpha (:391, used only in an f-string), and the imports scipy.optimize (:51), Tuple (:53), unit_conversions as cvt (:57 and run_momentum_phase.py:57).",
    "expected": "Remove, or use.",
    "failure_scenario": "",
    "repro": "",
    "confidence": "high"
  },
  {
    "id": "S6-A-18",
    "file": "trinity/phase1c_transition/run_transition_phase.py",
    "line": 605,
    "class": "deadcode",
    "severity": "S4",
    "claim": "The 'if tmax is not None' guards cannot be False, because an earlier unguarded comparison against tmax would already have raised TypeError.",
    "evidence": "run_transition_phase.py:407 'if tmin >= tmax:' executes before any None check; if stop_t were None this raises TypeError. The later guards at :605 and :779 ('if tmax is not None and ...') are therefore unreachable in their False branch. Identically run_momentum_phase.py:488 vs :687 and :832.",
    "expected": "Drop the guards, or validate stop_t explicitly at entry.",
    "failure_scenario": "",
    "repro": "",
    "confidence": "high"
  },
  {
    "id": "S6-A-19",
    "file": "trinity/phase1c_transition/run_transition_phase.py",
    "line": 860,
    "class": "other",
    "severity": "S4",
    "claim": "When the loop exits via the reached_tmax check the final reconciliation block writes a second snapshot at the same t_now with the same state, producing a duplicated output row.",
    "evidence": "run_transition_phase.py:593 save_snapshot() is followed at :605-611 by the 't_now >= tmax' break with no state change in between; :860 then calls save_snapshot() again after recomputing the same derived quantities at the same R2, v2, Eb, t_now. Identically run_momentum_phase.py:675 -> :687-693 break -> :894 save. The _snapshots_after_rCloud counter (:597-600 / :679-682) is not bumped for the final-block save, so the rCloud snapshot count and the actual snapshot count can disagree by one.",
    "expected": "Skip the final save when the loop broke immediately after an in-loop save, or dedupe on t_now.",
    "failure_scenario": "",
    "repro": "Run any config to stop_t and check for two identical t rows at the end of dictionary.jsonl.",
    "confidence": "medium"
  }
]
```
