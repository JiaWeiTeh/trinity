# Magic-number audit — "sibling `dR2min`" sweep of the trinity hot path

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
> and full runs cost minutes-to-hours, so any diagnostic worth keeping must be
> saved as a committed artifact under `docs/dev/` — never left in `/tmp`, the
> local-only `scratch/`, or an untracked `outputs/`. A future visit must be able
> to reproduce or compare against the numbers **without re-running**; record the
> exact config + command that produced each artifact.
>
> 🔗 **Cross-check the sibling docs — keep the workstream self-consistent.** This file is one of
> several living docs for its workstream (its `PLAN.md`, `FINDINGS.md`, `runs/README.md`, `NOTE_PATCHES.md`,
> and any other notes in the same folder). They drift out of sync *with each other* as fast as they drift
> from the code. Any agent or person editing one MUST, as part of the visit, circle back through the
> siblings and reconcile: if a number, status, claim, or line reference here contradicts a sibling — or a
> sibling has gone stale — fix it (or flag it, dated) so no two docs in the workstream disagree. Never
> update one in isolation.

**Status (2026-08-05):** 🟡 **AUDIT COMPLETE — findings triaged & source-verified. #1 and #4 measured & FIXED**
— #1 the file-tied T-floor, gated bit-identical (`TCLAMP_PLAN.md`); #4 the `vd = -1e8` first-segment
override, **deleted** on branch `hotfix/early-approximations` and gated by the `phase1a-init`
workstream (`docs/dev/phase1a-init/PLAN.md`; evidence `docs/dev/phase1a-init/data/gate_results.csv`).
**#2 measured but NOT fixed** — the `dt_switchon` ramp is load-bearing and its recommendation below
is now known to be the wrong fix; see its row. **#3 and #5 still open** (each remaining fix is
a physics-touching change needing its own gate).
**Update (2026-08-06, round 2 — `SWEEP2_PLAN.md`):** **#2 CLOSED** as document-and-pin (reproduced,
mechanism corrected to the phase-1a RK45 segment integrator, constant kept + pinned); **#3 FIXED &
GATED** (exact spline derivative replaces the FD; 5-config screen worst |ΔR2| 1.77e-8, fates
unchanged); **#5 re-verified** — the `0.05` now has a single source of truth
(`phaseSwitch_LlossLgain`), the residual tail stays with the transition workstream. No findings
remain open in this audit; #5's tail is owned elsewhere. Motivated by the
`dR2min` story (`docs/dev/performance/BUBBLE_CONDUCTION_STIFFNESS.md`): WARPFIELD's hand-tuned
`dR2min = 1e-7` pc floor would inflate bubble luminosity ~8×, and the companion `r2 += 1e-10`
"guard" in `bubble_E2P` is a unit-mismatched **dud** (1e-10 cm added to `r2 ≈ 3e18` cm). This sweep
hunts the rest of the package for the same two smells: **(a) an unjustified constant that changes
physical results**, and **(b) a unit/scale-mismatched guard that silently does nothing**.

## Method
Four read-only `Explore` subagents over disjoint regions, one shared rubric (the two smells above,
plus a units check and an "is it on an iterative/hot path?" test):
- **A** `bubble_structure/` + `phase1_energy/` + `phase0_init/` (the core hot path; both known offenders live here)
- **B** `phase1b_energy_implicit/` + `phase1c_transition/` + `phase2_momentum/` + `phase_general/`
- **C** `shell_structure/` + `cloud_properties/` + `cooling/`
- **D** `_functions/` + `sps/` + `_input/`

Subagents **over-flag** (standard defensive constants get reported as smells). Every CONFIRMED item
below was then **re-read against source by the lead** before listing; agent-only items are marked
`[agent]` and still need a source check. De-flagged items (§Checked-fine) record *why* the agent's
flag does not hold, so the next visit need not re-litigate them.

## CONFIRMED siblings — ranked (worth a gated fix later; none applied)

| # | location | constant | the smell | sev | hot path | verified |
|---|---|---|---|---|---|---|
| 1 | `cooling/net_coolingcurve.py:122` | `if T < 1e4: T = 1e4` | **Admitted band-aid.** The comment (`:114-120`) says *"the temperature seem to run at some very low value (~1e3.91) … Not sure why though, as the temperature should be around 1e7, not 1e4."* So a physical `T` is clamped **up** on the cooling lookup to dodge a sub-table dip nobody understands. Changes Λ(T) wherever the bubble/shell dips below 1e4 K. Classic `dR2min` pattern: a floor masking an undiagnosed behaviour. | **HIGH** | YES (cooling Λ in the ODE RHS) | ✅ lead-read |
| 2 | `bubble_structure/get_bubbleParams.py:368` (was `:367`; now carries its rationale block in-source) | `dt_switchon = 1e-3` Myr | Uncalibrated inherited "switch-on": for `t ≤ tSF + 1e-3` it ramps `R1_tmp = (t−tSF)/1e-3 · R1` into `bubble_E2P`, shaping the effective bubble pressure for the first ~1000 yr. Flagged independently by agents A **and** D. **MEASURED 2026-08-05 (E8b); REPRODUCED + CLOSED as document-and-pin 2026-08-06 (`SWEEP2_PLAN.md` §4-5).** Worth −0.0059% at the compact probe's observed age, −0.017% @8e4 yr on the GMC control — decays away, not a second discretisation artifact. **Not removable**, and the reproduction corrected the mechanism: ablated at `nCore=1e6` the full pressure drains `Eb` 180→29 au in 4 segments, then ~~the bubble-structure ODE stiffens / the solve stops converging~~ **phase 1a's segment integrator (`solve_ivp`, hard-coded RK45, `run_energy_phase.py:309`) stalls in micro-steps** — the bubble solve itself stays at ~1.3 s/call (`data/switchon_stall_probe.csv`, `switchon_stall_stacks.txt`). Per the pre-registered decision rule: constant kept, documented in-source, pinned by `test/test_dt_switchon_ramp.py` (deletion guard incl. the `t=None` ablation contract). ~~Re-open only alongside phase-1a integrator stiffness work — pre-registered as `docs/dev/phase1a-stiffness/PLAN.md`, where this constant's removal is Batch 6.~~ **That work ran (2026-08-06) and CLOSES the question: `dt_switchon` is NOT removable.** With an in-band energy-collapse guard landed so a collapsing segment stops cleanly instead of grinding, ablating the ramp still flips the stopping fate on **3 of 5** configs — including **`simple_cluster`, the default published config** (`ENERGY_COLLAPSED` at t=5.5e-7 Myr instead of running to `stop_t`). It also corrects this row's own severity picture: ablation flips the sign of the early energy budget on *every* config, and the two configs the "worth 0.006%" figure came from are precisely the only two that recover. Evidence: `docs/dev/phase1a-stiffness/data/dt_switchon_removability.csv`; write-up `.../PLAN.md` §2 D6. Reproduction: `data/switchon_repro_ledger.csv`. | MED | YES (early-`t` bubble pressure) | ✅ reproduced, documented & pinned — intentionally not changed |
| 3 | ~~`sps/update_feedback.py:184`~~ — **GONE** (replaced by `sps_f['fpdotdot_total']`, the exact derivative of the same cubic) | `dt = 1e-9` Myr | ✅ **FIXED 2026-08-06** (`SWEEP2_PLAN.md` §2-3, commit `db05694`). Measured first: the FD step sat **three decades into the float-roundoff regime** (h-sweep optimum ~1e-6; at 1e-9 the noise is up to 2.8e-2 relative where \|pdotdot\| is small, p99 8e-4 — `data/pdotdot_percall.csv`), and `t ± h` **crashed** within 1e-9 Myr of either table edge despite `t` passing the function's own range check. One correction to this row's original wording: "can sample spline noise across a knot" was imprecise — the interpolant is C², so knots are not the mechanism; float roundoff is. `interp1d(kind='cubic')` measured **bit-identical** to `make_interp_spline(k=3)`, so `get_interpolation` now exposes that spline's `.derivative()` and the FD (and its magic h) is deleted. Gated: failing-first tests (`test_pdotdot_spline_derivative.py`); all non-pdotdot feedback fields bit-identical across 60 sampled t; full-run screen on all 5 configs — worst \|ΔR2\| rel diff **1.77e-8**, every stopping fate unchanged, `f1edge_hidens` completes (`data/pdotdot_screen_results.csv`). | MED | YES (per ODE eval) | ✅ fixed & gated |
| 4 | ~~`phase1_energy/energy_phase_ODEs.py:270`~~ — **GONE** (verified absent from `trinity/` 2026-08-05) | `vd = -1e8` | ✅ **FIXED 2026-08-05** — deleted with the `EarlyPhaseApproximation` flag it existed to serve (`a944727`, branch `hotfix/early-approximations`). **The audit's question had no answer:** a constant RHS integrates exactly, so the override gave the closed form `v_exit = v0 − 1e8·SEGMENT_DURATION = 722.82 km/s` for *every* run on the bundled SB99 tables regardless of mass, SFE or density — it represents nothing physical, so it was deleted rather than documented. **Severity was under-called here.** "Bounded to the 1st segment" is true and not reassuring: at sub-GMC scale that one segment sets a trajectory that momentum-coasts for ~3000 yr, making a 0.15 pc H II region cross its observed radius **22× early**. It was also less bounded than this row claimed — the flag's clear site was `loop_count==0`-guarded and sat *after* the event check, so four in-loop exits leaked it into phases 1b/1c. Fixed jointly with the fixed 30-yr segment schedule it was compensating for (new param `phase1a_segFrac`); deleting it *alone* measures worse. Diagnosis `docs/dev/phase1a-init/FINDINGS.md`; gates `.../data/gate_results.csv`. | ~~MED~~ → **HIGH at sub-GMC scale** | YES (1st segment RHS) | ✅ fixed & gated |
| 5 | **Re-verified 2026-08-06 @ `731ac50` — the row below had drifted.** The `0.05` now has a single source of truth: registry param `phaseSwitch_LlossLgain` (default 0.05, `registry.py:407`), honored by both **live** check sites (`run_energy_phase.py:290-291`, `run_energy_implicit_phase.py:1249-1254`), each with a hardcoded `0.05` fallback. The `phase_events.py` copies (`make_cooling_balance_event`, `:319` default + `:497` hardcoded call) feed a factory that `build_implicit_phase_events` returns and `run_energy_implicit_phase.py:752` unpacks but **never invokes** — vestigial (flag, don't delete; would silently ignore the param if ever armed). The `0.9` is now the named local `RAM_DOMINANCE_THRESHOLD` (`run_transition_phase.py:749`). | `0.05`, `0.9` | Physics-gating thresholds for the energy→transition / transition-exit handoffs. Residual smell: two hardcoded fallbacks + a never-armed factory that would bypass the param. **Owned by the transition workstream** (entry point now archived: `docs/dev/archive/transition/TRIGGER_PLAN.md`) — record here, resolve there; do **not** re-open the F0–F5 trigger choice in this audit. | MED | YES (per-segment gate) | ✅ lead-read 2026-08-06 (`SWEEP2_PLAN.md` §1) |

## Already known — cross-referenced, not new
- `get_bubbleParams.py:224` `r2 += 1e-10` (cm) — the original unit-mismatch **dud**; A & D both reconfirm it is inert and the *real* guard is the `1e-13·r2³` volume floor at `:235`. Documented in `docs/dev/failed-large-clouds/PLAN.md §2`.
- `cooling/non_CIE/read_cloudy.py:95,97,133` — `RegularGridInterpolator` with no `bounds_error`/`fill_value` ⇒ out-of-grid cooling queries return **silent NaN**. Already logged as the "latent secondary nan source" in `failed-large-clouds/PLAN.md §8`; covered defensively by family F (clean termination). Not a *constant* smell — a missing guard.

## Checked — justified, NOT smells (de-flagged from agent over-reports)
- `get_bubbleParams.py:235` `shell_volume = 1e-13·r2³` — the **deliberate** failed-large-clouds G guard: bit-identical while `vol>0`, active only during an `R1→R2` collapse that then terminates via `ENERGY_COLLAPSED`. Justified-by-design (agent A lacked that context).
- `bubble_luminosity.py:52` `_T_INIT_BOUNDARY = 3e4 K` — documented conduction/ionization boundary (see the `dR2` work); its `(3e4/(min_T+0.1))²` penalty is a known **no-op** (≈0.999994).
- `phase2_momentum/run_momentum_phase.py:397,414` `max(R2,1e-10)`, `max(mShell,1e-10)` — **unit-matched** inert guards (pc on pc, Msun on Msun); never fire for physical values. *Not* duds in the `r2+=1e-10` sense (units are right), just belt-and-suspenders.
- `1e-100` / `1e-300` safe-divide & underflow floors (`get_InitPhaseParam.py:38-40` `MIN_LUMINOSITY/MOMENTUM/VELOCITY`, `read_sps.py:35` `EPSILON`, `get_betadelta.py` ×5, `operations.py`) — standard `np.maximum(x, tiny)` guards, inert for physical (`Lmech>0`) input. Agent A over-ranked `MIN_LUMINOSITY` as HIGH/ACTIVE; it is a dud (a star-forming cluster never has `Lmech_W=0`). Cosmetic upgrade only: prefer `np.finfo(float).tiny`.
- `get_shellODE.py` `tau>500 ⇒ exp(-tau)=0` — justified underflow guard (`exp(-500)≈7e-218`, already ~0 to float64; continuous to machine precision).
- `density_profile.py` `SMOOTH_FRAC=0.01` tanh bridge — documented, mass-conserving to O(frac²).
- Solver tolerances (`_BUBBLE_RTOL=1e-8`/`_BUBBLE_ATOL=1e-10`, `ODE_RTOL=1e-6`/`ATOL=1e-8`, `RESIDUAL_THRESHOLD=1e-4`), Weaver constants (`5/11`, `1.51e6`), grid sizes — all named/documented.

## Dead code (flag, don't delete — per CLAUDE.md rule 3)
- `shell_structure/shell_structure.py:311` `tau_max = 100` — assigned, never referenced. `[agent]`, needs a source check; if confirmed orphan, surface to the maintainer rather than removing silently.

## Recommended order of attack (each is its own gated change — NOT done here)
Every item touches physics on an iterative path ⇒ **Risky/iterative** under the CLAUDE.md ladder:
gate-first (define equivalence), capture a baseline, full-run equivalence on the stiff edge regimes
in separate processes at matched `t`, smallest diff, re-verify (gate + `pytest` + ruff F-rules), persist.

1. **#1 `net_coolingcurve` T-clamp** — ✅ **DONE (2026-06-20).** Measured first: across **9.46M** `get_dudt`
   calls over `simple_cluster` + both `f1edge` edges + the stiff LSODA-flood, `T<1e4` fired **0 times** (min
   T ever = 30000 K) — the clamp was **dead code** on a false premise (table reaches 3162 K, not 3.99). No
   upstream dip to fix. Shipped the file-tied floor (`if np.log10(T) < nonCIE_Tmin: T = 10**nonCIE_Tmin`),
   gated bit-identical for all reachable T (≥1e4) incl. a full-run byte-identity. Full writeup + evidence:
   `TCLAMP_PLAN.md`.
2. ~~**#2 `dt_switchon`** — characterise the first-1000-yr `R1` ramp's effect on `Pb` (bit-diff a run with the
   ramp vs without on a healthy config); if inert, delete; if active, justify or parameterise.~~
   **DONE 2026-08-05 (E8b) — and the "if inert, delete" branch is ruled out.** Characterised on three
   configs. By the letter of this recommendation the ramp *is* inert — −0.006% at the compact probe's observed age
   on a healthy config — and deleting it is nonetheless **fatal** at `nCore=1e6`, where ~~the bubble
   solve stops converging three segments in~~ (mechanism corrected below). The recommendation assumed
   inert-on-a-healthy-config implies safe-to-delete; it does not, because a healthy config never
   exercises what the constant protects. ~~Successor: make the switch-on scale-relative, keep the
   protection, and gate on `f1edge_hidens` completing at all.~~
   **CLOSED 2026-08-06 (`SWEEP2_PLAN.md` §4-5): reproduced (to the digit / bit-for-bit), mechanism
   instrumented — it is phase 1a's hard-coded-RK45 *segment integrator* that stalls as `Eb`
   collapses, not the bubble solve — and the pre-registered decision rule landed on
   document-and-pin, no successor.** A scale-relative switch-off would deliver the full pressure
   *earlier* at the stiff edge to buy ≤0.017% on healthy configs; the honest follow-up is phase-1a
   integrator stiffness handling, its own workstream. In-source rationale + `test/test_dt_switchon_ramp.py`
   now guard against the "inert, delete" reading this row used to recommend.
3. ~~**#3 `dt=1e-9` pdotdot step** — cheap to test: compare `pdotdot` from the analytic spline derivative vs the
   FD across configs; replace the FD with the spline's own derivative if available.~~
   **DONE 2026-08-06 — exactly this, measured then gated.** The comparison quantified the FD as
   roundoff-dominated (not knot-related as this audit guessed) plus a latent edge crash; the spline's
   own derivative was available bit-compatibly and shipped (`db05694`). See the row for the numbers.
4. ~~**#4 `vd=-1e8`** — trace what the first-segment override represents; document or derive it.~~
   **DONE 2026-08-05 — traced; it represents nothing, so it was deleted rather than documented.**
   See the row above. Worth carrying into the remaining findings as a pattern: this constant survived
   scrutiny for years because it is nearly invisible on the GMC-scale configs that both the test
   suite and the published validation use, and its severity was ranked from a hot-path reading alone.
   Neither "is it on the hot path" nor "is it bounded in time" caught it — what caught it was running
   a config four decades of mass away from the ones already trusted.
5. **#5 transition `0.05`/`0.9`** — hand to the transition workstream (entry point now archived:
   `docs/dev/archive/transition/TRIGGER_PLAN.md`). ~~At minimum de-duplicate the `0.05` to one
   source of truth~~ — **done upstream by 2026-08-06**: `phaseSwitch_LlossLgain` exists and both
   live sites honor it (see the row). What remains for the transition workstream is the tail:
   two hardcoded fallbacks and the never-armed `phase_events.py:497` factory that would bypass
   the param if ever wired up.
