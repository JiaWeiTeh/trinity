# S5b implicit runner — Lens B (what the code claims)

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

**Slice:** S5b — Phase 1b, the implicit-phase runner (★ high-stiffness).
**Input:** prose-only extraction of `trinity/phase1b_energy_implicit/run_energy_implicit_phase.py`
(module docstring, function docstrings, and all inline comments; line ranges preserved).
**Method:** I saw **no code**. Every statement below is a transcription of what the file *claims about
itself*. I cannot and do not assert whether any claim is true — each entry is written so a code-reading
lens can test it. Line citations use the original file's line numbers.

All `docs/dev/…` references below are recorded verbatim as citations. Per project convention those
documents are point-in-time and unverified; a claim's provenance being a `docs/dev` file is *not*
evidence the claim holds.

---

## 1. Formula ledger — every equation / scaling stated in prose

| # | Claim (as maths) | Where | Notes for the checker |
|---|---|---|---|
| F1 | `F_grav = G · mShell / R2² · (mCluster + mShell/2)` | :488 | The `+ mShell/2` self-gravity half-mass term is the checkable detail. Units of `G` unstated (see U6). |
| F2 | `Edot_from_balance = Lmech_total − (bubble_LTotal + leak) − 4π·R2²·v2·Pb` | :240–242 | Attributed to `get_betadelta`'s `Edot_from_balance`. The last term is the PdV work rate `Pb·dV/dt` with `dV/dt = 4πR2²v2`. Check sign convention on `v2 < 0` (collapse): the PdV term then *adds* energy. |
| F3 | `ebpeak` fires ⟺ `Edot_from_balance ≤ 0` ("PdV-inclusive net energy stops growing") | :239–242 | Non-strict `≤`. |
| F4 | `blowout` fires ⟺ `R2 > k_blowout · rCloud` ("shell escapes the cloud (geometric)") | :237–238 | `k_blowout` is a parameter; no default stated. |
| F5 | `Pb = (γ−1)·Eb / V` | :1359 | Stated as what `compute_R1_Pb` recomputes; used to argue `Eb<0 ⇒ Pb<0` garbage. |
| F6 | `Pb = (γ−1)·Lmech / ((4π/3)·R1²·v) → P_ram` as `R1 → R2` | :1144–1145 | Algebra self-check: with γ=5/3, `(γ−1)/(4π/3) = 1/(2π)`, so this is `Pb = Lmech/(2π R1² v)`, which equals `pdot/(4πR1²)` when `pdot = 2·Lmech/v` — i.e. it *is* the wind ram pressure, consistent. **But which `v`** (wind velocity vs shell `v2`) is never stated — see FLAG-15. Also note F5 and F6 are two *different* expressions for the same symbol `Pb`; the prose does not say where each applies. |
| F7 | `dex change = |log₁₀(new/old)|`, and `max_dex_change = max_k |log₁₀(after_k/before_k)|` | :290–306, :322 | Skips entries per :312 (see FLAG-04). |
| F8 | dex threshold `0.05` ⇒ factor `10^0.05 ≈ 1.12×` | :126 | Arithmetic checks out (10^0.05 = 1.1220). |
| F9 | dt_segment scale factor `≈ 1.26` | :127 | Consistent with `10^0.1`; the constant's actual value is the check. |
| F10 | effective `Lloss = Lcool + leak` when the cooling-boost option is `'none'` (the default) | :1235–1237 | Claimed byte-identical to the un-boosted path. |
| F11 | ODE state and derivative ordering: `y = [R2, v2, Eb, T0]`, `dydt = [dR2/dt, dv2/dt, dEb/dt, dT0/dt]` | :589–613 | |
| F12 | `dR2/dt = v2` | :620 | |
| F13 | `dv2/dt` = "acceleration from pressure balance" | :621 | No expression given — see FLAG-16 (vague). |
| F14 | `dEb/dt = Ed_from_beta`, `dT0/dt = Td_from_delta`, both **computed outside the ODE** and held fixed for the segment | :589–613, :623, :989 | ⇒ within a segment `Eb` and `T0` are *linear in t*. Load-bearing for FLAG-08 and FLAG-14. |
| F15 | Covering-fraction leak: `Cf = 1 ⇒ leak = 0` (sealed bubble reproduced exactly) | :810–811 | |
| F16 | `50 years` = the collapse segment duration, in a Myr-valued constant | :145 | ⇒ constant should be `5e-5` Myr. Load-bearing for FLAG-03. |
| F17 | `max_step = 2e-5` Myr "ensures ≥5 steps per segment" | :173 | ⇒ implies segment length ≥ `1e-4` Myr. Contradicted by F16 — see FLAG-03. |
| F18 | Transition energy floor handed to phase 1c `= 1e3`, chosen to equal phase1c `ENERGY_FLOOR` | :178–180, :1165–1167 | Cross-module numeric coupling; units unstated (U5). |

## 2. Units ledger

| # | Stated unit / convention | Where |
|---|---|---|
| U1 | ODE time `t` is **Myr** | :590–592 |
| U2 | All segment-duration constants (recalc-cooling interval, initial / min / max segment duration, collapse segment duration) are **Myr** | :111–114, :145 |
| U3 | Velocity thresholds for proactive timestep control are **pc/Myr** | :143, :144 |
| U4 | Solver `min_step` in **Myr**; `max_step = 2e-5` **Myr** | :172, :173 |
| U5 | Transition energy floor `1e3` — **no unit stated** | :178–180 |
| U6 | `compute_forces_pure` inputs `R2` ("shell outer radius"), `mShell` ("shell mass"), `Pb` ("bubble pressure") — **no units on any of them**; `ForceProperties` fields likewise carry no units | :467–487, :445–451 | 
| U7 | `50 years` quoted inside an otherwise-Myr constant block (mixed unit in a comment) | :145 |

Given U1–U3, the implied working system is (pc, Myr, M☉). U5/U6 are the documented gaps; on a code base
whose own CLAUDE.md names units as "a recurring bug class", `F_grav`'s `G` and the `1e3` energy floor
being unit-less in prose are the two worth a code check.

## 3. Citation ledger

| # | Citation, verbatim | What is attributed to it | Where |
|---|---|---|---|
| C1 | **McKee & Cowie 1977** | That persistent loss of the `dMdt>0` structure root "marks the evaporation→condensation domain boundary of the conduction-front model" — i.e. the physical end of the energy-driven solution. No equation or section number given. | :116–121 |
| C2 | `KAPPA_FREEZE_MECHANISM.md fix #1` | The consecutive-no-root → hand off to momentum behaviour; also "healthy rejection bursts observed so far are ≤ 8 segments and recover". | :120–121, :741–742, :901–902 |
| C3 | `docs/dev/transition/pdv-trigger/HIMASS_HANDOFF_PLAN.md` | The `classify_energy_collapse` routing table (stop / momentum / None) and the energy→momentum handoff rationale. | :185–198, :1146–1147 |
| C4 | `docs/dev/transition/pt4/R1_SHADOW_PLAN.md` | Four separate things: `evaluate_r1_shadow`'s criteria, `parse_transition_triggers`' grammar, `r1_transition_decision`'s precedence rule, and the shadow CSV sideline output. | :234–242, :253–261, :276–281, :721–722, :1256–1260, :1432–1434 |
| C5 | `docs/dev/archive/betadelta/stalling-energy-phase.md` | The `_inflow_frac_thickness` diagnostic / WARPFIELD "Problem 2". | :895–897 |
| C6 | `docs/dev/transition/pdv-trigger/PB_COLLAPSE_GUARD_FIX.md` | Skipping the reconciliation snapshot on the energy-collapsed exit (the `Pb ≈ −1.6e18` garbage row). | :1358–1363 |
| C7 | **WARPFIELD "Problem 2"** | "transient interior inflow during a feedback re-pressurisation" — named as a known WARPFIELD issue, with no paper/section. | :213–218, :895–897 |
| C8 | **"Eq. leak"** | The covering-fraction energy leak term. This is a citation-shaped string that names no source, document, or equation number. | :807 |
| C9 | `GRID_SIZE=5`, `LBFGSB_FALLBACK_THRESHOLD` (attributed to `get_betadelta.py`) | The `'legacy'` solver: 5×5 grid search, L-BFGS-B fallback only if grid residual exceeds the threshold, then best-of-candidates. | :8 (module docstring) |
| C10 | "it did not occur in any Phase-3 validation run" | Rarity of the no-physical-root condition. Historical/empirical claim, no artefact cited. | :840–846 |
| C11 | "field case: a 1e6 M☉ run pinned at beta=1 ground at ~1e-4 Myr/segment with a ~4-day projected phase completion" | Justification for disengaging the dt-shrink guard beyond a streak cap. | :137–138 |
| C12 | "long phase, ~273 segments" | Justification for the throttled heartbeat. Config-dependent, no config named. | :1177–1178 |

**C4 is one citation carrying four distinct behavioural claims**, two of which contradict each other
(shadow-only vs. opt-in drive) — see FLAG-06.

## 4. Contracts, ordering requirements, state

**Entry / exit contract**
- `run_phase_energy(params) -> ImplicitPhaseResults`; results container holds `t, R2, v2, Eb, T0, beta, delta` arrays plus termination info (:8, :632–647, :570).
- Entry guard: if the prior phase already advanced past `stop_t`, return explicitly rather than "silently looping zero times and reporting `termination_reason="unknown"`" (:667–669).
- `cool_alpha` is updated at entry "to match ODE-evolved v2 (preserves ODE continuity)" (:661). No formula; no statement that it is refreshed later (see FLAG-17).
- Results arrays are **pre-allocated** from an estimate based on the time range (:698, :8).

**Per-segment ordering (explicitly asserted)**
1. Log state at segment start (:777).
2. Update cooling structure periodically (:781, interval constant :111).
3. Update params with current state (:791); fetch feedback (:801).
4. Compute the covering-fraction leak, using `Pb` and `c_sound` **carried from the previous segment (1-step frozen)** (:807–811).
5. **beta/delta + bubble properties BEFORE shell structure**, "so that `Pb` and `bubble_mass` are current when `shell_structure_pure` reads them (bubble computation does not depend on shell)" (:822–824).
6. Update params with beta/delta (:884) and with **all** bubble properties — "This is critical: without this, `bubble_mass`, bubble arrays, etc. remain stale" (:891–892).
7. `R1`, `Pb` (:933); sound speed from `bubble_Tavg`, which must already be in params via the update at step 6 (:941–942).
8. **Shell mass BEFORE shell structure**, "so that the shell termination condition uses the current R2's swept-up mass" (:947–952).
9. Shell structure with current `Pb`, `bubble_mass`, `shell_mass` (:973); then `P_HII` from Strömgren balance via `n_IF_Str` (:978).
10. beta/delta → `Ed`, `Td` (:989); pressure diagnostics (:1001).
11. **Save snapshot BEFORE the ODE** so `t_now, R2, v2, Eb, T0`, feedback, `shell_props`, `bubble_props`, `beta`, `delta`, `R1`, `Pb`, forces and residuals are "all computed for the SAME `t_now`" (:1012–1015, :8).
12. Append to results at the same point (:1028); check `stop_t` (:1038).
13. Capture monitor values BEFORE integration (:1053); build snapshot; `solve_ivp` (:1049–1074).
14. Post-ODE: event handling (:1093–1116), extract state (:1122), adaptive dt (:1131), collapse routing (:1140), heartbeat (:1177), shell-mass re-update (:1181), velocity-based dt (:1205), termination checks (:1224).

**Shell-mass invariants** (stated twice, at :947–963 and again at :1181–1194)
- During collapse (`isCollapse=True`) shell mass is **frozen**.
- Shell mass may **never decrease** — "once mass is swept up, it stays in shell".
- The second block claims to "apply the same collapse-freeze and never-decrease guards as the primary shell mass block above" (:1181–1183) — a stated duplication.

**Force contract**
- `compute_forces_pure` computes all force components "without mutating params"; `params` is "read-only" (:467–487).
- `P_HII` must be **pre-computed in the phase runner** from `n_IF_Str` before the force call (:530, :978, :1376) — an ordering precondition.
- `ForceProperties` fields: gravitational; inward ionization-pressure; outward HII-pressure (from `n_IF_Str`); ram-pressure (from bubble pressure); radiation-pressure (direct + IR-trapped); plus "pressure quantities" (:445–451, :541).
- ISM pressure is added "if shell extends beyond cloud" (:519).

**Purity contract**
- "Pure ODE functions: No dictionary mutations during integration" (:8); `get_ODE_implicit_pure` receives a frozen `ODESnapshot` plus `params_for_feedback` "for feedback interpolation" (:589–613).
- `classify_energy_collapse` and `evaluate_r1_shadow` are "Pure (no global state) so the routing invariant is unit-testable" (:185–198).

**Transition-trigger contract**
- `transition_trigger` param is a **comma-separated string** parsed into a **set**; more than one criterion may be active and "the transition then fires on whichever occurs first" (:253–261).
- `'r1'` is an alias for `'blowout,ebpeak'` (:255–257).
- A list `[a, b]` is deliberately *not* the syntax, because that is sweep syntax in a `.param` (:258–259).
- Unknown token ⇒ `ValueError` ("Validates at this trust boundary") (:260–261).
- Default is `{'cooling_balance'}`, which "reproduces current behavior exactly" / is "byte-identical" (:726–727, :1283–1284).
- `r1_transition_decision` returns `'blowout'`, `'ebpeak'`, or `None`; **blowout takes precedence** when both are active and fire in the same segment (:276–281).
- `cooling_balance` "is handled by the inline ratio check, **gated on its membership in the set**" (:279–281).

**Termination / routing contract**
- `classify_energy_collapse` (:185–198): `Eb` non-finite ⇒ `'stop'` on `ENERGY_COLLAPSED`; `Eb ≤ 0` and finite ⇒ `'momentum'`; `Eb > 0` and finite ⇒ `None`.
- Momentum handoff: pass `(R2, v2)`, set `Eb` to the transition floor, and **do NOT set `EndSimulationDirectly`** so `main` runs 1c → momentum (:1165–1167).
- Cooling-balance transition threshold: "Get threshold from params (default 0.05)" (:1249). *The ratio itself and its inequality direction are never stated* — see FLAG-11.
- Collapse detection: "velocity negative **AND** radius decreasing" (:1301).
- `stop_t` check skipped if `tmax is None` (:1308); `stop_r` check skipped if `stop_r is None` (:1326).
- `stop_at_rCloud_nSnap`: break at the **top** of the loop "so we break BEFORE this iteration's `save_snapshot` fires" (:761–763); the past-rCloud counter increments only when the save actually wrote (:1019–1022).
- Falling out of the loop ⇒ `max_segments` or `"unknown"`; `"unknown"` "means we fell through every known exit path — a real bug surface, not routine completion. Surface it loudly." (:1409–1413).
- Events: built by a centralized module returning `(events_list, cooling_balance_factory)` (:750–751), described only as "events for safe termination" (:746–748, :1074). **Which events are terminal, and in which crossing direction, is never stated** — the prose is silent on event semantics.

**Output-invariance contracts (byte-identical claims — directly testable under project rule 5)**
- Shadow R1 block "never sets `termination_reason` / breaks / writes a physics param → main output stays byte-identical" (:1256–1260).
- Shadow CSV is "the ONLY new output — it never touches `dictionary.jsonl`" (:1432–1434).
- Default trigger set → byte-identical (:1283–1284); cooling boost `'none'` → "byte-identical" (:1235–1237); `Cf=1` → sealed trajectory "reproduced exactly" (:810–811); `next_dt_segment` base policy "unchanged from the original inline block" (:378–381).

## 5. Numerical claims

| # | Claim | Where |
|---|---|---|
| N1 | Integrator is `scipy.integrate.solve_ivp` with method **`'LSODA'`**, chosen "for stiff/non-stiff switching" | :8, :175 |
| N2 | `min_step` "only supported by LSODA" — so solver kwargs are built conditionally | :1065 |
| N3 | Relative tolerance constant exists (value not in prose) | :170 |
| N4 | Absolute tolerance "**relaxed from 1e-9**" — new value not stated, reason not stated | :171 |
| N5 | `max_step = 2e-5` Myr, justified as "ensures ≥5 steps per segment" | :173 |
| N6 | Adaptive stepping: shrink dt on a change > 0.05 dex, grow on a change below it, by factor ≈1.26, clamped to [min, max] segment durations | :113–114, :126–127, :378–390, :394, :399 |
| N7 | Non-convergence guard: while `0 < streak ≤ BETADELTA_DT_SHRINK_MAX_STREAK`, dt **growth is suppressed and dt shrinks instead**; beyond the cap the guard disengages and normal adaptive stepping resumes | :378–390, :132–138 |
| N8 | Unconverged streak warns **once** at `BETADELTA_UNCONVERGED_WARN_STREAK`; resets on a converged solve | :129, :334–341 |
| N9 | No-physical-root: first hit of a streak WARNs, repeats are DEBUG | :852–854 |
| N10 | End-of-phase summary: `clean` = (every segment converged) AND (no no-physical-root safety-net hit) → INFO, else WARNING | :362–368, :1424–1426 |
| N11 | Velocity-based dt: `|v2| >` threshold-1 ⇒ reduce dt; `|v2| >` threshold-2 ⇒ minimum step; collapse segment duration = 50 yr | :141–145, :1205–1216 |
| N12 | Monitored parameters for adaptive stepping are scalars only (no arrays), selected "based on analysis of the top 30 most variable parameters", grouped as core state / feedback / cooling / bubble / shell / force | :147–165 |
| N13 | beta-delta solver selectable via `betadelta_solver`, **production default `'hybr'`** (unbounded root-finder + `dMdt>0` acceptance gate); `'legacy'` = 5×5 grid → conditional L-BFGS-B → best-of-candidates | :8 |
| N14 | `BetaDeltaResult` carries `beta, delta, Edot_residual, T_residual` (normalized) and `Edot_from_beta, Edot_from_balance, T_bubble, T0` (raw, diagnostics) | :8 |
| N15 | Loop is bounded by `max_segments` | :120, :1409 |
| N16 | Heartbeat is outer-loop only, "never inside the solvers", self-throttling to `HEARTBEAT_EVERY` | :1177–1178 |

## 6. Regimes, validity limits, assumptions

- **R1.** The energy-driven solution is claimed to have *physically ended* when the `dMdt>0` structure root is persistently lost — the evaporation↔condensation boundary of the McKee & Cowie (1977) conduction-front model (:116–121). This is the slice's principal physical regime statement.
- **R2.** `Eb → 0` ⇒ bubble pressure floors at `~P_ram` ⇒ "the shell is already momentum-driven", so `Eb ≤ 0` is a *transition*, not a stop (:185–198, :1140–1147).
- **R3.** The energy-driven model "is invalid past this point (it would drive `R1→R2` and divide-by-zero in `compute_R1_Pb`)" (:1142–1143) — a stated singularity.
- **R4.** `Lloss` "changes slowly", so reusing the pre-ODE value in the post-ODE termination check is "acceptable" (:1233–1234); the leak's `Pb`/`c_sound` freeze is likewise justified by "this phase's slowly-varying `Lloss` treatment" (:809–810).
- **R5.** Bubble computation "does not depend on shell" (:824) — an assumed one-way dependency that licenses the ordering.
- **R6.** `_inflow_frac_thickness` is "Grid-independent" and "NOT used in any physics (v is absent from the cooling integrals)" (:213–218, :895–897).
- **R7.** No-physical-root is "Rare on a self-consistent trajectory" (:841–842).
- **R8.** This phase uses a "very fine grid… Only in transition phase it goes coarse" (:109, flagged TODO).

## 7. Admissions ledger (TODO / approximate / temporary / known-defect language)

| # | Admission, quoted or close-quoted | Where |
|---|---|---|
| A1 | "**TODO**: very fine grid in this phase. Only in transition phase it goes coarse." | :109 |
| A2 | Absolute tolerance "**relaxed from 1e-9**" — a loosened numerical gate, new value and justification absent. | :171 |
| A3 | "Pb and c_sound are carried from the previous segment (**1-step frozen**)" for the leak term. | :809–810 |
| A4 | `Lloss` in the termination trigger comes from **pre-ODE** bubble properties: "cannot cheaply recompute without the betadelta solver; acceptable since Lloss changes slowly". | :1233–1234 |
| A5 | On event termination, "beta/delta are from the **start of this segment** (**best available**; the event occurred within one segment of their computation)". | :1106–1108 |
| A6 | Final append: "beta/delta are from the **last pre-ODE** computation (**best available**)". | :1347 |
| A7 | On a no-physical-root segment `bubble_properties is None`, "so the structure values and the dMdt warm start below **hold at the last physical segment**" — i.e. deliberately stale state. | :840–846 |
| A8 | The `save_snapshot` duplicate guard "**can silently skip** the first segment of a phase when its `(t_now, R2)` match the previous phase's reconciliation". | :1019–1022 |
| A9 | Previously the phase could "**silently** loop zero times and report `termination_reason="unknown"`". | :667–669 |
| A10 | Per-segment convergence is DEBUG-only, so "a fully unconverged phase is **silent** at the default log level" — the warn/summary machinery exists purely to patch that. | :334–341, :362–368, :1424–1426 |
| A11 | Documented pathology: "a 1e6 M☉ run **pinned at beta=1** ground at ~1e-4 Myr/segment with a **~4-day projected phase completion**". | :137–138 |
| A12 | "floor-dt only **multiplies segment count without buying correctness**" beyond the streak cap. | :133–137 |
| A13 | "it would drive R1→R2 and **divide-by-zero** in `compute_R1_Pb`". | :1142–1143 |
| A14 | Energy-collapsed exit would otherwise write "a **garbage negative** terminal row (`Pb ~ -1.6e18`)". | :1358–1361 |
| A15 | A bare `save_snapshot()` at the phase boundary "would save **stale** derived values AND **block the next phase's correct first snapshot** via the duplicate guard". | :1354–1356 |
| A16 | `"unknown"` termination "means we fell through every known exit path — **a real bug surface**". | :1412–1413 |
| A17 | "**The future flip** would replace the logging here with a real break" — shadow code is explicitly provisional. | :1259–1260 |
| A18 | Grinding "**frozen state** to max_segments" is the failure mode the streak cap exists to avoid (KAPPA_FREEZE / "freeze-watch" tracing at :901–902). | :116–121, :741–742 |
| A19 | "Without this, `bubble_mass`, bubble arrays, etc. remain **stale**" — a known staleness trap in the update ordering. | :891–892 |
| A20 | Monitored-parameter list justified only as "Based on analysis of the top 30 most variable parameters" — no artefact, threshold, or config named. | :149 |

---

## 8. Flags

### FLAG-01 (S2) — The no-physical-root streak: "hands off to momentum" vs. "Log-only" / "NOT a transition trigger"
Three comments describe the same counter and disagree on what it does:
- :116–121 — "After this many **consecutive no-root segments the phase hands off to momentum** instead of grinding frozen state to max_segments (KAPPA_FREEZE_MECHANISM.md fix #1)".
- :741–742 — "Consecutive no-physical-root streak (the frozen-implicit signature…). **Log-only.**"
- :840–846 — "a logged safety net, **NOT a transition trigger** (phase end stays owned by the cooling-balance event)."

Exactly one of these can describe the code. This is the difference between a run that terminates the
energy phase on a physics criterion and a run that spins to `max_segments` on frozen state (A11/A18
say the latter has been observed in the field). Highest-value single check in the slice.

### FLAG-02 (S3) — "did not occur in any Phase-3 validation run" vs. "healthy rejection bursts observed so far are ≤ 8 segments and recover"
:841–842 says the no-physical-root condition never occurred in Phase-3 validation; :121 says rejection
bursts *are* observed, bounded at ≤8 segments, and recover. Either the two sentences describe different
conditions (and the prose conflates them) or one is stale. This matters because the streak cap at
:116–121 is presumably sized against that "≤ 8" observation — if the two conditions differ, the cap is
calibrated against the wrong statistic.

### FLAG-03 (S3) — `max_step = 2e-5` Myr cannot "ensure ≥5 steps per segment" at the collapse segment duration
:173 claims `max_step = 2e-5` Myr "ensures ≥5 steps per segment". ≥5 steps requires segment length
≥ `5 × 2e-5 = 1e-4` Myr. But :145 sets the collapse segment duration to 50 yr = `5e-5` Myr, giving
≤3 steps — and `DT_SEGMENT_MIN` (:113) may be smaller still. The stiffest regime in the phase (rapid
collapse) is precisely where the stated step-count guarantee fails by its own constants.

### FLAG-04 (S3) — `compute_max_dex_change`: "skip if opposite signs" vs. "large change if sign flips"
:312 — "Skip if values are missing, zero, or **opposite signs**".
:317–318 — "# Sign change" / "# **Large change** if sign flips".
These prescribe opposite behaviours for the same input, and the consequences are opposite too: skipping
a sign flip lets dt *grow* through a sign reversal; treating it as a large change forces dt to shrink.
A sign flip in a monitored parameter (e.g. `v2` at the onset of collapse) is exactly the event the
controller exists to catch.

### FLAG-05 (S3) — Ram-pressure force: computed "from bubble pressure" vs. "no ram pressure in implicit phase"
:449 and :538 both label a ram-pressure force computed from the bubble pressure, and the module
docstring lists `F_ram` among the snapshot's forces (:8). :559 states "**no ram pressure in implicit
phase**". Check whether `F_ram` is populated, zeroed, or dead in this phase — and, if zeroed, whether
downstream force-budget consumers and the snapshot schema expect a real value.

### FLAG-06 (S3) — "SHADOW mode… never drives the switch" vs. the opt-in DRIVE block
`evaluate_r1_shadow`'s docstring (:234–236) states flatly "computed/logged, **never drives the switch**",
and :721–722 / :1256–1260 repeat it with a byte-identical-output guarantee. But :276–281
(`r1_transition_decision`) and :1282–1287 describe an **opt-in DRIVE** path where `blowout`/`ebpeak`
genuinely end the phase. If the drive path consumes the same fired-flags that `evaluate_r1_shadow`
produces, the shadow docstring's invariant is false for any non-default `transition_trigger`. Both
claims cite the same document (C4).

### FLAG-07 (S3) — `cooling_balance`: "the cooling-balance **event**" vs. "the **inline ratio check**"
:843–844 — "phase end stays owned by the **cooling-balance event**".
:279–281 — "**cooling_balance is handled by the inline ratio check**, gated on its membership in the set".
:750–751 also shows the events builder returning a `cooling_balance_factory`, implying an event form
exists. Check whether cooling balance is implemented twice (as a `solve_ivp` event *and* as a post-ODE
inline check) and, if so, which one actually fires first and whether the inline gate on set membership
(:280–281) is also applied to the event — an ungated event would break the "default set → byte-identical"
claim in the opposite direction.

### FLAG-08 (S3) — "adaptive integration for accuracy" vs. piecewise-constant `dEb/dt` and `dT0/dt`
The headline claim is `solve_ivp(LSODA)` "instead of manual Euler stepping", "Adaptive integration for
accuracy" (:8). But `Ed_from_beta` and `Td_from_delta` are computed **outside** the ODE and passed in as
constants for the whole segment (:589–613, :623, :989). For two of the four state variables the scheme
is therefore first-order in `dt_segment` — i.e. explicit Euler with step `dt_segment`, regardless of
what LSODA does internally on `R2`/`v2`. The accuracy claim is at best true only of `R2`, `v2`.

### FLAG-09 (S3) — Energy-collapsed exit is described as `Eb < 0`, but `ENERGY_COLLAPSED` is defined as `Eb` **non-finite**
`classify_energy_collapse` (:185–198) partitions: non-finite → `'stop'` on `ENERGY_COLLAPSED`;
`Eb ≤ 0` **and finite** → `'momentum'`. Yet :1358–1361 justifies the reconciliation skip with "on the
energy_collapsed exit: **there Eb < 0**, so `compute_R1_Pb` would recompute `Pb = (γ−1)·Eb/V` as a
garbage negative terminal row (`Pb ~ -1.6e18`)", and :1398–1401 repeats "Energy-collapsed: skip the Pb
recompute (**Eb<0** → garbage)". A finite negative `Eb` cannot reach the energy-collapsed branch under
the docstring's own partition, and a `nan`/`inf` `Eb` would not produce `-1.6e18`. Either the guard is
keyed on the wrong condition, or the docstring's partition and the guard's comment describe different
code generations. Second-order consequence to check: with `Eb ≤ 0` finite now routed to momentum, is
the `Pb`-garbage guard still reachable at all (possible dead code), and is the *momentum* path's
reconciliation snapshot safe given `Eb` is only set to the floor at :1165–1167?

### FLAG-10 (S3) — The "all values at the SAME t_now" invariant has at least five documented exceptions
The module docstring's central promise is "**Consistent snapshots**: All values saved at consistent
timestamps", with the enumerated snapshot contents at :8 and :1012–1015. Prose elsewhere documents
these violations of it:
1. beta/delta from the segment start on an event termination (:1106–1108) and on the final append (:1347) — A5/A6;
2. leak `Pb`/`c_sound` frozen one segment back (:809–810) — A3;
3. `bubble_props` held at the last physical segment on a no-root segment, while `t_now`/`R2` advance (:840–846) — A7, which is exactly the staleness :891–892 calls "critical" to avoid;
4. the termination check mixing post-ODE `Lgain` with pre-ODE `Lloss` (:1227–1237) — see FLAG-12;
5. the energy-collapsed final snapshot deliberately persisting the "last healthy `Pb` and derived state" against the new `t` (:1398–1401), which is the same "stale derived values" that :1354–1356 rejects for the normal path.
Each is individually defensible; the flag is that the *documented invariant* is stated without
qualification and a consumer of `dictionary.jsonl` will believe it.

### FLAG-11 (S3) — The cooling-balance termination criterion is never stated
:1249 says only "Get threshold from params (**default 0.05**)". Nowhere does the prose state the ratio
(`Lloss/Lgain`? `|Lgain−Lloss|/Lgain`?), the inequality direction, or whether the trigger fires on
approach from above or below — for the criterion that "owns" the end of this phase (:843–844). The
single most important phase-transition condition in the slice is undocumented. (Note also that `0.05`
appears at :126 as a *dex* threshold for an unrelated purpose; confirm the two constants are distinct.)

### FLAG-12 (S3) — Termination trigger mixes post-ODE `Lgain` with pre-ODE `Lloss`
:1227–1229 deliberately re-fetches feedback at the **new** `t_now` so `Lgain` reflects `Lmech_total`
"especially across SN turn-on boundaries", while :1233–1234 keeps `Lloss` from **pre-ODE** bubble
properties. Across an SN turn-on the justification "`Lloss` changes slowly" is weakest exactly where
`Lgain` jumps — so the ratio can cross the threshold on a one-segment timing artefact. Check whether
the ratio is evaluated with both quantities at the same `t`, and what the lag is worth at the
threshold (default 0.05).

### FLAG-13 (S3) — Two independent shell-mass update sites with hand-copied guards
:947–963 computes shell mass before shell structure with the collapse-freeze and never-decrease guards;
:1181–1194 re-updates it "for adaptive stepping comparison" and claims to "apply the **same** …
guards". Duplicated invariant logic that must be edited in lockstep. Check the two are actually
equivalent (in particular whether "never decrease" clamps to a running maximum or only to the previous
value — :1189/:1194 say "keep at its **previous** value", which is not the same as a running max if the
freeze ever lifts).

### FLAG-14 (S3) — Can `Eb` go negative *inside* a segment, and what does the ODE do then?
`dEb/dt` is constant over the segment (F14), so `Eb` is linear in `t` and can cross zero mid-segment;
the collapse check at :1140 is post-ODE. :621 says `dv2/dt` is "acceleration from **pressure balance**",
and F5 says `Pb = (γ−1)Eb/V`. If the ODE recomputes `Pb` from `y[2]`, a mid-segment `Eb < 0` yields a
negative pressure and an unphysical inward acceleration for the remainder of the segment before the
guard ever runs; if `Pb` comes frozen from the snapshot, it does not. The prose does not say which, and
never states that any event terminates on `Eb → 0` (event semantics are undocumented — see below).
Given :1142–1143's own admission of a `compute_R1_Pb` divide-by-zero at `R1 → R2`, this is worth an
explicit check.

### FLAG-15 (S3) — Collapse-dt regime: "extreme ⇒ minimum" vs. a dedicated 50-yr collapse duration
:144 says the higher velocity threshold selects "**minimum step**"; :1211 "Extreme collapse velocity:
use **minimum** segment duration"; :1216 "Moderate collapse velocity: use **intermediate** segment
duration"; :145 defines a 50 yr (`5e-5` Myr) "segment duration during collapse". Three named durations
(`DT_SEGMENT_MIN` at :113, the collapse duration at :145, and whatever "intermediate" means) are mapped
onto two velocity bands by prose alone. If `DT_SEGMENT_MIN > 5e-5` Myr, the "extreme" band would use a
*coarser* step than the "moderate" band — inverting the stated intent. Check the numeric ordering
`DT_SEGMENT_MIN ≤ DT_SEGMENT_COLLAPSE ≤ DT_SEGMENT`.
Related, smaller: :141–144 gate on `|v2|` (magnitude, so outward motion too) while :1209 says "**Only
during collapse** (negative velocity = inward motion)". Confirm which.

### FLAG-16 (S4) — Claims too vague to check as written
- :621 "acceleration from pressure balance" — no expression, no force list; the whole momentum equation is one comment.
- :661 "Update `cool_alpha` to match ODE-evolved v2" — no relation given between `cool_alpha` and `v2`.
- :149 "Based on analysis of the top 30 most variable parameters" — no artefact, metric, or config (A20).
- :1177 "long phase, ~273 segments" — config-dependent number presented without a config.
- :213–218 "Grid-independent" — no definition of the invariance being claimed.
- :807 "Eq. leak" — citation-shaped, names no source (C8).
- :8 "Adaptive integration for accuracy" — see FLAG-08.

### FLAG-17 (S4) — `cool_alpha` refresh: entry-only?
:661 updates `cool_alpha` from `v2` at phase entry, justified as "preserves ODE continuity". :781 says
the cooling structure is updated periodically thereafter. The prose never says whether the periodic
update re-derives `cool_alpha` from the current `v2`; if it does not, `cool_alpha` is pinned to the
entry velocity for a phase in which `v2` can reverse sign (collapse). Check.

### FLAG-18 (S4) — Undocumented event semantics
The prose names events only as "events for safe termination" (:746–748, :1074) built by a "centralized
module" returning `(events_list, cooling_balance_factory)` (:750–751). **No event's `terminal` flag,
`direction`, or root condition is documented anywhere in the slice**, despite events being able to end
the phase and to set the recorded final state (:1093–1116). For a stiff, event-driven integrator this
is the largest documentation gap in the file.

### FLAG-19 (S4) — Units gaps
`compute_forces_pure`'s `R2`/`mShell`/`Pb` and every `ForceProperties` field are documented without
units (:445–451, :467–487), and `F_grav = G·mShell/R2²·(mCluster+mShell/2)` (:488) does not say which
`G`. The `1e3` transition energy floor (:178–180) carries no unit while being asserted to match a
constant in another module. On a code base that names units as a recurring bug class, these are the
places a cross-module mismatch would hide.

### FLAG-20 (S4) — Snapshot force list is incomplete relative to `ForceProperties`
Module docstring (:8) lists snapshot forces as "`F_grav, F_ram, F_ion, F_rad`" — four. `ForceProperties`
(:445–451) enumerates five force fields plus "pressure quantities", the extra being the **outward HII
pressure force from `n_IF_Str`** (:448, :523–530). Check whether `P_HII`/`F_HII` actually reaches the
snapshot; if it does, the docstring is stale; if it does not, a force computed every segment is dropped
from the output.

### FLAG-21 (S4) — Stale section banner
:206–208 prints the banner "Force Properties Dataclass" but is immediately followed by the
"Adaptive Stepping Helper" banner (:209–210); the real `ForceProperties` banner and dataclass appear at
:439–451. Copy-paste residue — harmless, but it is the kind of drift that makes a section map unreliable.

### FLAG-22 (S4) — `bubble_LTotal` vs `Lcool` naming for (apparently) the same loss term
F2 (:240–242) writes the loss slot as `bubble_LTotal + leak`; F10 (:1237) writes the effective loss as
`Lcool + leak`. If these are the same quantity under two names, the residual, the ODE and the
termination trigger are consistent; if not, the trigger and the balance residual disagree about what
"loss" means. The prose asserts they are fed "the SAME effective loss as the residual + ODE"
(:1235–1237), so the check is whether the names resolve to one value.

### FLAG-23 (S4) — Controller gain exceeds its own deadband
The change threshold is `0.05` dex (:126) while the dt scale factor is `≈1.26` ≈ `0.1` dex (:127). A
grow step therefore perturbs the monitored parameters by more than the deadband that authorised it,
which admits a shrink/grow limit cycle. Low confidence — this depends entirely on how the monitored
parameters respond to `dt_segment`, which I cannot see. Recorded only because both constants are stated
in prose and the comparison is free.

---

## 9. Notes for the triangulation
- The slice is unusually well-commented on *provenance* (six `docs/dev` documents, one refereed
  citation) and unusually thin on *mechanism*: the momentum equation (:621), the cooling-balance
  criterion (:1249) and all event semantics (:746–751) are the three load-bearing behaviours with no
  prose at all. FLAG-11 and FLAG-18 are gaps rather than contradictions, but they are where a Lens-A
  reading adds the most.
- FLAG-01, FLAG-06, FLAG-07 and FLAG-09 all concern **who ends this phase**. Between them the prose
  names four candidate owners (the cooling-balance event, the inline cooling ratio check, the R1
  drive criteria, the no-root streak) and asserts exclusivity for two of them. That cluster should be
  resolved together, not finding-by-finding.
- The byte-identical guarantees in §4 are unusually testable: `Cf=1`, `transition_trigger` default,
  cooling boost `'none'`, and `next_dt_segment`'s "unchanged from the original inline block" are four
  independent equivalence gates the project's own rule 5 harness can settle without physics judgement.

```json
[
  {
    "id": "S5b-B-01",
    "file": "trinity/phase1b_energy_implicit/run_energy_implicit_phase.py",
    "line": 116,
    "class": "state",
    "severity": "S2",
    "claim": "The consecutive no-physical-root streak constant is documented as causing a hand-off to the momentum phase ('After this many consecutive no-root segments the phase hands off to momentum instead of grinding frozen state to max_segments', KAPPA_FREEZE_MECHANISM.md fix #1).",
    "evidence": ":116-121 says the streak hands off to momentum; :741-742 says the same streak is 'Log-only'; :840-846 says it is 'a logged safety net, NOT a transition trigger (phase end stays owned by the cooling-balance event)'. All three describe the same counter and at most one can be true.",
    "expected": "Exactly one documented behaviour for the no-root streak: either it terminates/routes the phase at the cap, or it only logs. If it routes, :741-742 and :840-846 must be corrected; if it only logs, :116-121's rationale (and the cap's existence) is unfounded.",
    "failure_scenario": "If the streak is in fact log-only, a run that permanently loses the dMdt>0 root grinds frozen state to max_segments -- the exact pathology :137-138 records in the field (1e6 Msun run, ~1e-4 Myr/segment, ~4-day projected completion). If it does route, any test asserting log-only behaviour is wrong about the phase's termination set.",
    "repro": "Read the no-root branch and the streak counter; check whether reaching the cap sets termination_reason/breaks or only logs. Force the condition on a high-mass config and observe termination_reason.",
    "confidence": "high"
  },
  {
    "id": "S5b-B-02",
    "file": "trinity/phase1b_energy_implicit/run_energy_implicit_phase.py",
    "line": 173,
    "class": "numerical",
    "severity": "S3",
    "claim": "max_step = 2e-5 Myr 'ensures >=5 steps per segment'.",
    "evidence": ":173 states the guarantee; :145 sets the collapse segment duration to 50 years = 5e-5 Myr, which admits at most 2.5 max-steps. DT_SEGMENT_MIN (:113) may be smaller still.",
    "expected": "Either max_step scaled to the active dt_segment, or the comment qualified to the regimes where dt_segment >= 1e-4 Myr. The guarantee should hold in the collapse regime, which is the stiffest.",
    "failure_scenario": "During rapid collapse the segment is resolved by <=3 LSODA steps while dEb/dt and dT0/dt are frozen for the whole segment, so the least-resolved integration occurs exactly where the solution is stiffest.",
    "repro": "Compare the numeric values of MAX_STEP (:173), DT_SEGMENT_MIN (:113) and the collapse dt (:145); instrument sol.t.size per segment during a collapsing run.",
    "confidence": "high"
  },
  {
    "id": "S5b-B-03",
    "file": "trinity/phase1b_energy_implicit/run_energy_implicit_phase.py",
    "line": 312,
    "class": "silent-failure",
    "severity": "S3",
    "claim": "compute_max_dex_change 'Skip[s] if values are missing, zero, or opposite signs' (:312), while :317-318 says a sign change is treated as a 'Large change if sign flips'.",
    "evidence": "Two comments eleven lines apart prescribe opposite handling of the same input class in the same function.",
    "expected": "One documented rule for sign flips. Given the function drives dt control, a sign flip should force the large-change (shrink) branch, not be skipped.",
    "failure_scenario": "If sign flips are skipped, dt_segment can grow through a v2 sign reversal (onset of collapse) precisely when the controller should shrink it -- the change the monitor exists to detect is the one it discards.",
    "repro": "Call compute_max_dex_change with a monitored key flipping sign (e.g. v2: +5 -> -5) and assert the returned value is large rather than 0.",
    "confidence": "high"
  },
  {
    "id": "S5b-B-04",
    "file": "trinity/phase1b_energy_implicit/run_energy_implicit_phase.py",
    "line": 559,
    "class": "state",
    "severity": "S3",
    "claim": "'no ram pressure in implicit phase' (:559), contradicting the ram-pressure force computed 'from bubble pressure' at :538 / :449 and F_ram's presence in the documented snapshot contents (:8).",
    "evidence": ":449 and :538 label a ram-pressure force field and its computation; :8 lists 'forces (F_grav, F_ram, F_ion, F_rad)' among snapshot contents; :559 asserts there is none in this phase.",
    "expected": "Consistent statement of whether F_ram is populated, forced to zero, or unused in phase 1b, and whether the snapshot writes a real or zero value.",
    "failure_scenario": "A zeroed F_ram silently written into dictionary.jsonl as a physics force; or, conversely, a live F_ram double-counting the bubble pressure already entering via Pb in the momentum equation.",
    "repro": "Inspect the F_ram assignment in compute_forces_pure and the value of F_ram in a dictionary.jsonl row from a phase-1b segment.",
    "confidence": "medium"
  },
  {
    "id": "S5b-B-05",
    "file": "trinity/phase1b_energy_implicit/run_energy_implicit_phase.py",
    "line": 234,
    "class": "state",
    "severity": "S3",
    "claim": "evaluate_r1_shadow is documented as 'R1 transition criteria in SHADOW mode (computed/logged, never drives the switch)', reinforced at :721-722 and :1256-1260 with a byte-identical-output guarantee.",
    "evidence": ":276-281 (r1_transition_decision) and :1282-1287 document an opt-in DRIVE path where blowout/ebpeak genuinely end the phase via the transition_trigger keyword. Both the shadow and drive descriptions cite the same document (R1_SHADOW_PLAN.md).",
    "expected": "The shadow docstring qualified to the default trigger set, or the drive path documented as consuming a separate evaluation. 'Never drives the switch' is unconditionally false if the drive path reads the same fired flags.",
    "failure_scenario": "A reader (or a test) relies on 'never drives the switch' and is surprised when transition_trigger='r1' ends the phase on a geometric criterion; conversely an audit trusts the byte-identical claim for a non-default config where it does not hold.",
    "repro": "Check whether the DRIVE block at ~:1282 consumes the (blowout_fired, ebpeak_fired) tuple returned by evaluate_r1_shadow; run with transition_trigger='cooling_balance,blowout' and confirm termination_reason.",
    "confidence": "high"
  },
  {
    "id": "S5b-B-06",
    "file": "trinity/phase1b_energy_implicit/run_energy_implicit_phase.py",
    "line": 843,
    "class": "state",
    "severity": "S3",
    "claim": "'phase end stays owned by the cooling-balance event' (:843-844) versus 'cooling_balance is handled by the inline ratio check, gated on its membership in the set' (:279-281).",
    "evidence": "Two mechanisms named for one criterion; :750-751 shows the events builder returning a 'cooling_balance_factory', implying an event form also exists alongside the post-ODE inline check at ~:1249.",
    "expected": "One documented owner of the cooling-balance transition. If both an event and an inline check exist, the gating on membership in active_triggers must be applied to both, and the precedence between them stated.",
    "failure_scenario": "An ungated cooling_balance event fires during integration regardless of active_triggers, breaking the 'non-default trigger set fires on whichever criterion occurs first' contract and the byte-identical guarantee in the opposite direction.",
    "repro": "Locate cooling_balance_factory's use in the events list; check whether it is passed to solve_ivp as terminal and whether its inclusion is conditioned on 'cooling_balance' in active_triggers.",
    "confidence": "medium"
  },
  {
    "id": "S5b-B-07",
    "file": "trinity/phase1b_energy_implicit/run_energy_implicit_phase.py",
    "line": 1358,
    "class": "regime",
    "severity": "S3",
    "claim": "The reconciliation-snapshot skip is justified by 'on the energy_collapsed exit: there Eb < 0, so compute_R1_Pb would recompute Pb = (gamma-1)*Eb/V as a garbage negative terminal row (Pb ~ -1.6e18)' (:1358-1361, repeated :1398-1401).",
    "evidence": "classify_energy_collapse (:185-198) defines ENERGY_COLLAPSED as the Eb NON-FINITE case ('nan/inf -> unrecoverable'), and routes finite Eb <= 0 to momentum instead. A non-finite Eb cannot produce -1.6e18, and a finite negative Eb should not reach the energy_collapsed branch.",
    "expected": "The guard keyed on the condition that actually reaches it. Either ENERGY_COLLAPSED can still be reached with finite Eb < 0 (then the docstring partition is wrong), or the guard's stated rationale is stale from before the momentum routing was introduced.",
    "failure_scenario": "Stale guard: with finite Eb<0 now routed to momentum, the Pb-garbage branch may be unreachable dead code, while the genuinely reachable non-finite case writes nan/inf through a path whose comment claims it was handled.",
    "repro": "Check the condition guarding the reconciliation skip against classify_energy_collapse's return values; construct a run terminating with non-finite Eb and inspect the terminal Pb in the output.",
    "confidence": "medium"
  },
  {
    "id": "S5b-B-08",
    "file": "trinity/phase1b_energy_implicit/run_energy_implicit_phase.py",
    "line": 8,
    "class": "state",
    "severity": "S3",
    "claim": "'Consistent snapshots: All values saved at consistent timestamps' -- snapshots saved BEFORE ODE integration so t_now, R2, v2, Eb, T0, feedback, shell_props, bubble_props, beta, delta, R1, Pb, forces and residuals all correspond to the same t_now (:8, :1012-1015).",
    "evidence": "Five documented exceptions in the same file: beta/delta from the segment start on event termination (:1106-1108) and on the final append (:1347); leak Pb/c_sound frozen one segment back (:809-810); bubble_props held at 'the last physical segment' when bubble_properties is None (:840-846), the very staleness :891-892 calls 'critical'; and the energy-collapsed final snapshot persisting the 'last healthy Pb and derived state' (:1398-1401), which :1354-1356 rejects as 'stale derived values' for the normal path.",
    "expected": "The invariant stated with its exceptions, or a per-row consistency flag in the output, so downstream consumers of dictionary.jsonl know when a row mixes timestamps.",
    "failure_scenario": "Analysis code computes a residual or a force balance from one snapshot row assuming all quantities are simultaneous; on a no-root or event-terminated segment the row silently mixes state from two different times.",
    "repro": "Force a no-physical-root segment and confirm whether the written row's bubble_props correspond to t_now or an earlier segment; compare beta/delta timestamps on an event-terminated segment.",
    "confidence": "high"
  },
  {
    "id": "S5b-B-09",
    "file": "trinity/phase1b_energy_implicit/run_energy_implicit_phase.py",
    "line": 1249,
    "class": "other",
    "severity": "S3",
    "claim": "The cooling-balance transition threshold is read from params with 'default 0.05' -- and nothing else about the criterion is documented.",
    "evidence": ":1249 gives only the threshold. The ratio being compared, its inequality direction, and the crossing sense are stated nowhere, despite :843-844 asserting this criterion 'owns' the end of the phase. Contrast :237-242, where the blowout and ebpeak criteria are both written out explicitly.",
    "expected": "The criterion written out in prose, e.g. 'transition when (Lgain - Lloss)/Lgain < threshold', with the direction and the quantities' definitions, matching the level of documentation given to the R1 criteria.",
    "failure_scenario": "The phase's primary termination condition cannot be reviewed, unit-tested against a stated contract, or compared with WARPFIELD's published criterion; a sign or direction error in it would be invisible to review.",
    "repro": "Read the inline check near :1249 and write down the actual inequality; confirm whether 0.05 there is distinct from the 0.05 dex threshold at :126.",
    "confidence": "high"
  },
  {
    "id": "S5b-B-10",
    "file": "trinity/phase1b_energy_implicit/run_energy_implicit_phase.py",
    "line": 1233,
    "class": "numerical",
    "severity": "S3",
    "claim": "The termination check re-fetches feedback at the post-ODE time so 'Lgain must reflect the current Lmech_total at the new t_now, especially across SN turn-on boundaries' (:1227-1229), while Lloss is taken from pre-ODE bubble properties because it 'changes slowly' (:1233-1234).",
    "evidence": "The two sides of the compared ratio are evaluated at different times by design; the justification for the lag ('Lloss changes slowly') is weakest exactly at the SN turn-on the fresh-Lgain fetch exists to capture.",
    "expected": "Either both sides at the same t, or a documented bound on the error the one-segment Lloss lag introduces in the ratio relative to the 0.05 threshold.",
    "failure_scenario": "Across an SN turn-on, Lgain jumps while Lloss lags one segment, so the ratio can cross (or fail to cross) the threshold on a timing artefact rather than physics -- transitioning the phase at the wrong t.",
    "repro": "Log Lgain and Lloss with their source times for segments spanning an SN turn-on; recompute the ratio with a same-t Lloss and compare the transition time.",
    "confidence": "medium"
  },
  {
    "id": "S5b-B-11",
    "file": "trinity/phase1b_energy_implicit/run_energy_implicit_phase.py",
    "line": 589,
    "class": "numerical",
    "severity": "S3",
    "claim": "The module advertises 'scipy.integrate.solve_ivp(LSODA): Adaptive integration for accuracy' and use of solve_ivp 'instead of manual Euler stepping' (:8).",
    "evidence": "Ed_from_beta and Td_from_delta are inputs to get_ODE_implicit_pure, 'computed outside ODE' (:589-613, :623, :989), so dEb/dt and dT0/dt are constant across a segment and Eb, T0 evolve linearly in t -- explicit Euler with step dt_segment for two of four state variables, whatever LSODA does internally.",
    "expected": "The accuracy claim qualified to R2 and v2, and the segment-level order of accuracy for Eb and T0 stated; or Ed/Td recomputed within the ODE.",
    "failure_scenario": "Convergence testing on rtol/atol shows no improvement in Eb/T0 because their error is set by dt_segment, not by the integrator tolerances -- a tolerance tightening that appears to do nothing.",
    "repro": "Halve dt_segment at fixed rtol/atol and at fixed dt_segment tighten rtol/atol; compare which one moves the final Eb, T0.",
    "confidence": "high"
  },
  {
    "id": "S5b-B-12",
    "file": "trinity/phase1b_energy_implicit/run_energy_implicit_phase.py",
    "line": 1140,
    "class": "divergence",
    "severity": "S3",
    "claim": "Eb falling through zero makes the energy-driven model invalid -- 'it would drive R1->R2 and divide-by-zero in compute_R1_Pb' (:1142-1143) -- and is detected post-ODE at :1140.",
    "evidence": "dEb/dt is frozen for the segment (:589-613, :623), so Eb is linear in t and can cross zero mid-segment; :621 says dv2/dt is 'acceleration from pressure balance' and :1359 gives Pb = (gamma-1)*Eb/V. No event guarding Eb -> 0 is documented anywhere in the slice (event terminal flags and directions are undocumented, :746-751).",
    "expected": "Either Pb inside the ODE is taken frozen from the snapshot (so a mid-segment Eb<0 cannot produce negative pressure), or a terminal event on Eb catches the crossing before the segment completes. Whichever holds should be stated.",
    "failure_scenario": "If the ODE recomputes Pb from y[2], a mid-segment Eb<0 gives negative bubble pressure and an unphysical inward acceleration for the rest of the segment; the post-ODE guard then routes to momentum from a state already corrupted by that sub-segment excursion.",
    "repro": "Inspect whether get_ODE_implicit_pure derives Pb from y[2] or from the frozen snapshot; instrument a segment in which Eb changes sign and print the dense-output v2 within it.",
    "confidence": "medium"
  },
  {
    "id": "S5b-B-13",
    "file": "trinity/phase1b_energy_implicit/run_energy_implicit_phase.py",
    "line": 145,
    "class": "numerical",
    "severity": "S3",
    "claim": "Velocity-based dt control maps two |v2| thresholds onto segment durations: the higher threshold selects the 'minimum step' (:144, :1211 'Extreme collapse velocity: use minimum segment duration') and the lower an 'intermediate segment duration' (:1216); separately a dedicated collapse duration of 50 years (5e-5 Myr) is defined (:145).",
    "evidence": "Three named durations (DT_SEGMENT_MIN at :113, the 50-yr collapse duration at :145, and 'intermediate') are mapped onto two bands by prose alone, with no stated ordering between them.",
    "expected": "DT_SEGMENT_MIN <= collapse duration <= DT_SEGMENT, so that the extreme band is strictly finer than the moderate band.",
    "failure_scenario": "If DT_SEGMENT_MIN > 5e-5 Myr, the 'extreme collapse' band uses a coarser step than the 'moderate collapse' band -- resolution decreasing as the collapse accelerates, inverting the control's stated purpose.",
    "repro": "Print the three constants and assert the ordering; log dt_segment against |v2| across a collapsing run and confirm it is monotone non-increasing in |v2|.",
    "confidence": "medium"
  },
  {
    "id": "S5b-B-14",
    "file": "trinity/phase1b_energy_implicit/run_energy_implicit_phase.py",
    "line": 1181,
    "class": "divergence",
    "severity": "S3",
    "claim": "The second shell-mass update 'Apply[s] the same collapse-freeze and never-decrease guards as the primary shell mass block above' (:1181-1183).",
    "evidence": "Two independent update sites (:947-963 and :1181-1194) implement the same two invariants by hand. The second site's comments say 'keep params[shell_mass] at its previous value' (:1189, :1194), which is not identical to a running-maximum clamp if the collapse freeze ever lifts.",
    "expected": "One shared helper enforcing both invariants, or a test asserting the two sites agree for the same inputs.",
    "failure_scenario": "The two sites drift under maintenance, so the shell mass used by the shell-structure/termination path differs from the one used for the adaptive-stepping dex comparison -- dt control reacting to a mass the physics never saw.",
    "repro": "Diff the two blocks; assert shell_mass is identical immediately after each site for a segment where the collapse flag toggles.",
    "confidence": "medium"
  },
  {
    "id": "S5b-B-15",
    "file": "trinity/phase1b_energy_implicit/run_energy_implicit_phase.py",
    "line": 841,
    "class": "citation",
    "severity": "S3",
    "claim": "The no-physical-root condition is 'Rare on a self-consistent trajectory (it did not occur in any Phase-3 validation run)' (:840-842).",
    "evidence": ":121 states the opposite of rarity for what appears to be the same condition: 'healthy rejection bursts observed so far are <= 8 segments and recover' -- observed bursts, used to size the streak cap.",
    "expected": "One consistent empirical statement, with the artefact or run set named. If 'rejection bursts' and 'no-physical-root' are different conditions, the prose must distinguish them, because the streak cap is sized against the '<= 8' observation.",
    "failure_scenario": "The streak cap is calibrated against a statistic gathered for a different condition, so it either fires during healthy recoverable bursts or never fires during genuine frozen-state grinds.",
    "repro": "Determine whether the counter incremented by the no-root branch is the same counter the '<= 8 segments' observation refers to.",
    "confidence": "medium"
  },
  {
    "id": "S5b-B-16",
    "file": "trinity/phase1b_energy_implicit/run_energy_implicit_phase.py",
    "line": 746,
    "class": "numerical",
    "severity": "S4",
    "claim": "Events are described only as 'events for safe termination', built by a centralized module returning '(events_list, cooling_balance_factory)' (:746-751, :1074).",
    "evidence": "No event's terminal flag, crossing direction, or root function is documented anywhere in the slice, although events can end the phase and set the recorded final state (:1093-1116).",
    "expected": "Each event documented with its root condition, terminal status and direction, as the R1 criteria are at :237-242.",
    "failure_scenario": "A non-terminal or wrong-direction event goes unnoticed in review; the phase either misses a termination it should catch or terminates on a crossing in the wrong sense.",
    "repro": "Enumerate the events returned by the centralized builder and record terminal/direction for each.",
    "confidence": "high"
  },
  {
    "id": "S5b-B-17",
    "file": "trinity/phase1b_energy_implicit/run_energy_implicit_phase.py",
    "line": 488,
    "class": "units",
    "severity": "S4",
    "claim": "F_grav = G * mShell / R2^2 * (mCluster + mShell/2), with compute_forces_pure's R2 ('Shell outer radius'), mShell ('Shell mass') and Pb ('Bubble pressure') documented without units, and no ForceProperties field carrying a unit (:445-451, :467-487, :488).",
    "evidence": "The rest of the slice states units explicitly (t in Myr :590, dt in Myr :111-114, velocities in pc/Myr :143-144, max_step in Myr :173); the force path is the one place they are absent. The transition energy floor 1e3 (:178-180) is likewise unit-less while being asserted equal to another module's ENERGY_FLOOR.",
    "expected": "Units on the force-path parameters and on the energy floor, consistent with the (pc, Myr, Msun) system implied elsewhere, plus a statement of which G is used.",
    "failure_scenario": "A G in cgs (or an energy floor in the wrong system) silently rescales the gravitational force or defeats the phase-1c floor comparison, in a code base whose own conventions name units as a recurring bug class.",
    "repro": "Check the G constant's source and the units test in test/test_conventional_units.py; confirm the 1e3 floor equals phase1c's ENERGY_FLOOR in the same units.",
    "confidence": "medium"
  },
  {
    "id": "S5b-B-18",
    "file": "trinity/phase1b_energy_implicit/run_energy_implicit_phase.py",
    "line": 8,
    "class": "state",
    "severity": "S4",
    "claim": "The module docstring lists the snapshot's forces as 'forces (F_grav, F_ram, F_ion, F_rad)'.",
    "evidence": "ForceProperties (:445-451) enumerates five force fields, the extra being the outward HII pressure force from n_IF_Str (:448, :523-530, :978, :1376), which is computed every segment and at the reconciliation snapshot.",
    "expected": "The docstring list matching the fields actually written, or the HII force documented as intentionally not persisted.",
    "failure_scenario": "A force computed every segment is absent from dictionary.jsonl, so the shell force budget cannot be closed from the output; or the docstring is merely stale and misleads the reader about the schema.",
    "repro": "Compare ForceProperties fields against the keys written by save_snapshot in this phase.",
    "confidence": "medium"
  },
  {
    "id": "S5b-B-19",
    "file": "trinity/phase1b_energy_implicit/run_energy_implicit_phase.py",
    "line": 661,
    "class": "state",
    "severity": "S4",
    "claim": "'Update cool_alpha to match ODE-evolved v2 (preserves ODE continuity)' at phase entry (:661).",
    "evidence": "The relation between cool_alpha and v2 is never given, and the periodic cooling-structure update (:781, interval at :111) is not documented as re-deriving cool_alpha. v2 can reverse sign within this phase (collapse detection at :1301).",
    "expected": "The cool_alpha(v2) relation stated, and an explicit statement of whether it is refreshed with the cooling structure or pinned at the entry value.",
    "failure_scenario": "cool_alpha stays pinned to the entry velocity while v2 reverses during collapse, so the cooling structure is evaluated with a similarity exponent from a regime the run has left.",
    "repro": "Trace cool_alpha assignments through the phase; log its value against v2 across a run that collapses.",
    "confidence": "medium"
  },
  {
    "id": "S5b-B-20",
    "file": "trinity/phase1b_energy_implicit/run_energy_implicit_phase.py",
    "line": 171,
    "class": "numerical",
    "severity": "S4",
    "claim": "The ODE absolute tolerance is documented only as 'Absolute tolerance (relaxed from 1e-9)'.",
    "evidence": "Neither the current value nor the reason for the relaxation is recorded; the relative tolerance comment (:170) gives no value either. This is an accuracy-affecting change to a stiff integrator recorded as a parenthetical.",
    "expected": "The current atol value and the justification for relaxing it (which regime forced it), given project rule 5's requirement that solver-path changes carry a documented equivalence gate.",
    "failure_scenario": "A future tightening or loosening is made blind, with no record of which run forced the original relaxation and no baseline to compare against.",
    "repro": "Read the constant; search the test suite and docs/dev for the run that motivated the relaxation.",
    "confidence": "high"
  },
  {
    "id": "S5b-B-21",
    "file": "trinity/phase1b_energy_implicit/run_energy_implicit_phase.py",
    "line": 807,
    "class": "citation",
    "severity": "S4",
    "claim": "The covering-fraction energy leak is attributed to 'Eq. leak' (:807).",
    "evidence": "'Eq. leak' names no paper, thesis, document or equation number -- it is citation-shaped text with no referent. The term is load-bearing: it appears in Edot_from_balance (:240-242) and in the effective Lloss (:1235-1237). Separately, F2 writes the loss slot as 'bubble_LTotal + leak' while F10 writes it as 'Lcool + leak' -- two names for what is asserted to be 'the SAME effective loss'.",
    "expected": "A real citation for the leak formula (source and equation number), and one consistent name for the cooling-loss term across the residual, the ODE and the termination trigger.",
    "failure_scenario": "The leak term cannot be checked against its source; if bubble_LTotal and Lcool are not the same quantity, the balance residual and the termination trigger disagree about what 'loss' means while the comment asserts they agree.",
    "repro": "Resolve bubble_LTotal and Lcool to their definitions and confirm they are the same value; locate the leak formula's actual source.",
    "confidence": "medium"
  },
  {
    "id": "S5b-B-22",
    "file": "trinity/phase1b_energy_implicit/run_energy_implicit_phase.py",
    "line": 206,
    "class": "deadcode",
    "severity": "S4",
    "claim": "A banner comment block reading 'Force Properties Dataclass' at :206-208.",
    "evidence": "It is immediately followed by the 'Adaptive Stepping Helper' banner at :209-210; the real ForceProperties banner and dataclass appear at :439-451. The :206-208 banner labels a section that is not there.",
    "expected": "The stale banner removed, so the file's section map matches its contents.",
    "failure_scenario": "None at runtime; it misleads navigation and review of a 1400-line file.",
    "repro": "Read :206-215 and confirm the block following the banner is the adaptive-stepping helper, not the dataclass.",
    "confidence": "high"
  },
  {
    "id": "S5b-B-23",
    "file": "trinity/phase1b_energy_implicit/run_energy_implicit_phase.py",
    "line": 126,
    "class": "numerical",
    "severity": "S4",
    "claim": "Adaptive stepping uses a 0.05 dex change threshold ('10^0.05 ~ 1.12x', :126) with a dt scale factor of ~1.26 (:127).",
    "evidence": "The scale factor (~0.1 dex) is twice the deadband (0.05 dex), so a growth step perturbs the system by more than the change that authorised it.",
    "expected": "Either a scale factor no larger than the deadband, or evidence that dt_segment growth does not translate into monitored-parameter change at comparable magnitude.",
    "failure_scenario": "Shrink/grow limit cycling in dt_segment, inflating segment count without improving accuracy.",
    "repro": "Log dt_segment and max_dex_change per segment on a steady stretch of an energy-driven run and look for a period-2 oscillation.",
    "confidence": "low"
  }
]
```
