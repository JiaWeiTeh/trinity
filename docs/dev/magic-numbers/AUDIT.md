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

**Status (2026-06-20):** 🟡 **AUDIT COMPLETE — findings triaged & top candidates source-verified;
NOTHING fixed yet (each fix is a physics-touching change needing its own gate).** Motivated by the
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
| 2 | `bubble_structure/get_bubbleParams.py:367` | `dt_switchon = 1e-3` Myr | Uncalibrated inherited "switch-on": for `t ≤ tSF + 1e-3` it ramps `R1_tmp = (t−tSF)/1e-3 · R1` into `bubble_E2P`, shaping the effective bubble pressure for the first ~1000 yr. No physics reference, no sensitivity note. Flagged independently by agents A **and** D. | MED | YES (early-`t` bubble pressure) | ✅ lead-read |
| 3 | `sps/update_feedback.py:184` | `dt = 1e-9` Myr | Hardcoded central-difference step for `pdotdot_total = d(pdot)/dt` on the SPS spline, evaluated **every** `get_current_sps_feedback` call. `1e-9` Myr (~9 hr) is ~10⁶× below the table grid; uncalibrated, can sample spline noise across a knot. | MED | YES (per ODE eval) | ✅ lead-read |
| 4 | `phase1_energy/energy_phase_ODEs.py:270` | `vd = -1e8` | Magic override of the velocity derivative when `EarlyPhaseApproximation` (default **True**, `_input/registry.py:381`). **Bounded:** flipped False after `loop_count==0` (`run_energy_phase.py:317-318`), so it hits only the first energy segment — but `-1e8` pc/Myr² is undocumented (what does it represent? why this value?). | MED (bounded to 1st segment) | YES (1st segment RHS) | ✅ lead-read |
| 5 | `phase1b_energy_implicit/run_energy_implicit_phase.py:1093` **+** `phase_general/phase_events.py` (`cooling_balance`); and `phase1c_transition/run_transition_phase.py:747` (`0.9`) | `0.05`, `0.9` | Physics-gating thresholds for the energy→transition / transition-exit handoffs. The `0.05` cooling-balance margin is **duplicated** in two files (no single source of truth). **Owned by the transition workstream** (`docs/dev/transition/TRIGGER_PLAN.md`) — record here, resolve there; do **not** re-open the F0–F5 trigger choice in this audit. | MED | YES (per-segment gate) | `[agent]` |

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

1. **#1 `net_coolingcurve` T-clamp** — highest value and the most `dR2min`-like. *First measure, don't fix*:
   instrument how often `T<1e4` actually fires across `simple_cluster` + the `f1edge`/stiff configs, and
   whether the sub-1e4 K dip is a real physical excursion or a solver transient. The clamp may be hiding a
   bug (the comment suspects so). Only then decide: extend the table, follow the file's true `T_min`, or fix
   the upstream dip.
2. **#2 `dt_switchon`** — characterise the first-1000-yr `R1` ramp's effect on `Pb` (bit-diff a run with the
   ramp vs without on a healthy config); if inert, delete; if active, justify or parameterise.
3. **#3 `dt=1e-9` pdotdot step** — cheap to test: compare `pdotdot` from the analytic spline derivative vs the
   FD across configs; replace the FD with the spline's own derivative if available.
4. **#4 `vd=-1e8`** — trace what the first-segment override represents; document or derive it.
5. **#5 transition `0.05`/`0.9`** — hand to `docs/dev/transition/TRIGGER_PLAN.md`; at minimum de-duplicate the
   `0.05` to one source of truth.
