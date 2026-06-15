# P-shadow design — two-criterion transition trigger (F0 ∨ F4)

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
> a committed artifact (a CSV/table under `analysis/data/`, or a force-added
> harness/figure under `scratch/`) — never left in `/tmp` or an untracked
> `outputs/`. A future visit must be able to reproduce or compare against the
> numbers **without re-running**; record the exact config + command that produced
> each artifact.

**Status:** DESIGN FOR REVIEW (2026-06-15). No code changed yet. Implements the
P-shadow phase of `docs/dev/TRANSITION_TRIGGER_PLAN.md` using the evidence from
`analysis/transition-trigger-P0.md` (P0 + P-sens complete; G0 = profile-dependent
trigger). **Awaiting maintainer sign-off on the open decisions in §6 before any
implementation.**

## 1. What the evidence mandates (recap, one line each)
- **Flat configs transition by cooling** — F0 `(Lgain−Lloss)/Lgain < ε` fires at
  the Eb-peak (dense-flat @0.197); ε robust across [0.02,0.10]; keep ε=0.05.
- **Steep configs transition by blowout** — no cooling family fires at any ε; the
  bubble crosses rCloud (F4 @2.728 Myr) while still formally energy-driven.
- **No extra machinery needed** — F1 cumulative = F0 for flat, fails steep;
  sustained-over-t_cross rule changes nothing; every transition precedes the first
  WR/SN surge, so the reset is moot. → minimal **F0(ε=0.05) ∨ F4(R2>rCloud)**.

## 2. The gap this fixes (verified)
Production has **no blowout transition in the implicit phase (1b)**. When R2 crosses
rCloud during 1b, nothing happens — the bubble keeps integrating energy-driven to
`stop_t`. In `tt_steep_long` the bubble crossed rCloud at 2.728 Myr but ran to
stop_t=4.0 (133 implicit segs, 0 transition segs) and ended via `reached_tmax`,
skipping 1c+2. A blown-out shell that stays "energy-driven" is physically wrong;
F4 fixes it.

## 3. Verified code sites (re-check per banner before editing)
| site | file:line | role |
|---|---|---|
| F0 live terminator | `run_energy_implicit_phase.py:1076–1079` | `if (Lgain−Lloss)/Lgain < threshold: termination_reason="cooling_balance"; break` — does **not** set `EndSimulationDirectly` |
| ε param | `phaseSwitch_LlossLgain` (read at `:1070–1074`, default 0.05) | the cooling threshold |
| rCloud | `params['rCloud'].value` (already used `:664`) | cloud radius for F4 |
| existing R2>rCloud awareness | `:659–669`, `:885–892` (`stop_at_rCloud_nSnap`) | snapshot-stop only, not a transition |
| `large_radius` (≠ blowout) | `:1106–1113` | `R2 > stop_r` (param, default None) → **ends sim** (`EndSimulationDirectly=True`); not a phase transition |
| blowout factory (exists, unused in 1b) | `phase_events.py:218` `make_cloud_boundary_event(rCloud)` | `R2 − rCloud`, direction +1; currently 1a→1b only |
| dead cooling factory | `phase_events.py:317` `make_cooling_balance_event(threshold=0.05)` | hardcoded 0.05; built by `build_implicit_phase_events` but never the live terminator |
| phase routing | `main.py:280,300,340` | `EndSimulationDirectly` gates 1b→1c→2; `cooling_balance` flows through to 1c→2 |

## 4. Design (shadow-first, zero production impact)
**New param** `transition_trigger`, default `"instantaneous"` (current F0-only
behaviour). Register beside `phaseSwitch_LlossLgain` (registry + `default.param`).

**Shadow mode (default, `"instantaneous"`):** in the 1b loop, right after the F0
check (`:1079`), compute the F4 condition `R2 > params['rCloud'].value`. Do **not**
act on it — record the first epoch where each criterion *would* fire to a sideline
file (e.g. `<output>/transition_shadow.jsonl`: `{t, R2, rCloud, ratio_F0, which}`).
Production still breaks on F0 only ⇒ **byte-identical snapshots**.

**Promote mode (later, behind the param, `"cooling_or_blowout"`):** add a parallel
break — `if R2 > rCloud: termination_reason="blowout"; break` — *without* setting
`EndSimulationDirectly`, so it routes 1b→1c→2 exactly like `cooling_balance`.
F0 ∨ F4 = whichever fires first (both are in the same loop).

**Reconcile the dead factory:** make `make_cooling_balance_event` read
`phaseSwitch_LlossLgain` (drop the hardcoded 0.05) or remove it, so there is one
parameterized energy-ratio path. (Confirm every `build_implicit_phase_events`
caller first — plan §"ε is already a param".)

## 5. Test plan (per plan P-shadow)
1. **Byte-identical default:** `transition_trigger="instantaneous"` reproduces
   production snapshot hash on all 5 configs (drift budget D=0).
2. **Shadow epochs match P0:** sideline log shows F0 @0.197 (dense), F4 @2.728
   (steep_long) — equal to the harvested CSVs (mod the 1-segment left-rectangle
   offset already characterized).
3. **Promote smoke:** with `"cooling_or_blowout"`, `tt_steep_long` transitions at
   2.728 (routes to 1c→2) instead of running to stop_t; dense-flat unchanged
   (cooling fires first at 0.197 < its rCloud crossing).
4. **Continuity (P-promote gate):** at the blowout switch Eb/Rb/v2/P_drive hand off
   to 1c no more discontinuously than F0 (1c/2 drive on `max(Pb, P_HII+P_ram)`).

## 6. Open decisions for the maintainer (blocking implementation)
1. **Route blowout through 1c (Eb drain) like cooling_balance, or skip 1c → direct
   to momentum?** Recommend **through 1c** (same handoff mechanism, minimal change;
   a blown-out bubble still has Eb to drain). Decide before promote.
2. **Param shape:** `transition_trigger ∈ {instantaneous, cooling_or_blowout}`
   (recommended) vs a separate boolean `enable_blowout_transition`. The former
   matches the plan's wording and leaves room for future families.
3. **Blowout threshold:** exactly `R2 > rCloud` (P-sens: the crossing is clean at
   ratio 1.0, no tuning needed) vs a fraction `f·rCloud`. Recommend **1.0**.
4. **Scope of this PR:** shadow-only (steps in §4 shadow + §5 tests 1–2), or
   shadow+promote in one go? Recommend **shadow-only first** (lands the param,
   logging, byte-identical proof), promote in a follow-up after the continuity gate.

## 7. Out of scope (unchanged from plan)
Betadelta solver; transition/momentum runner internals beyond the switch condition;
3D mixing-layer physics; the clock-B (1c sound-crossing drain) treatment (separate
maintainer call — P0 showed 1c is short for flat and absent for steep).
