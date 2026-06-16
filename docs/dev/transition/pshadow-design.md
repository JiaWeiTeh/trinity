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
> a committed artifact (a CSV/table under `docs/dev/data/`, or a force-added
> harness/figure in the relevant `docs/dev/<workstream>/` folder) — never left in `/tmp` or an untracked
> `outputs/`. A future visit must be able to reproduce or compare against the
> numbers **without re-running**; record the exact config + command that produced
> each artifact.

**About this document**
- **Status (verified 2026-06-16):** 🟢 **SHIPPED** (2026-06-16) — **P-shadow + P-promote both shipped**. `transition_trigger='cooling_or_blowout'` now terminates the implicit phase on F0 ∨ F4 (blowout `R2>rCloud`), routing 1b→1c→2; default stays `instantaneous` (byte-identical). The §6 decisions are resolved (maintainer, 2026-06-16). **Remaining:** P-validate — the continuity gate + macro deltas on a full steep run.
- **Type:** design — the shadow-first, two-criterion (F0 cooling ∨ F4 blowout) trigger design; both the shadow and the promote break are now built. P-validate is the remaining step.
- **Workstream:** `transition/` — the implicit→momentum transition trigger.
- **Where it sits:** `TRIGGER_PLAN.md` (plan) → `P0.md` (P0/P-sens evidence, G0 = profile-dependent) → **this (design)** → implementation (**P-shadow + P-promote shipped**) → P-validate.
- **Code it concerns:** phase 1b implicit terminator (`run_energy_implicit_phase.py` F0 block + new `transition_trigger` param), the dead/blowout event factories in `phase_events.py`, and 1b→1c→2 routing in `main.py`.
- **Linked files & data:** plan `TRIGGER_PLAN.md`; evidence `P0.md`; data `docs/dev/data/transition_*.csv`; code `trinity/phase1b_energy_implicit/run_energy_implicit_phase.py`, `trinity/phase_general/phase_events.py`, `trinity/main.py`.

**Status:** **P-SHADOW + P-PROMOTE SHIPPED** (2026-06-16) — the `transition_trigger`
param + shared F0/F4 criteria (`trinity/phase_general/transition_shadow.py`) drive
the 1b terminator. Default `instantaneous` = F0 only (byte-identical snapshots);
`cooling_or_blowout` adds the F4 blowout break (`R2 > rCloud`), routing 1b→1c→2 like
cooling balance. Implements P-shadow + P-promote of `docs/dev/transition/TRIGGER_PLAN.md`
on the evidence from `docs/dev/transition/P0.md` (P0 + P-sens complete; G0 =
profile-dependent trigger). The §6 maintainer decisions are resolved (2026-06-16).
**Remaining:** P-validate — the continuity gate (test-plan #4) + macro deltas on a
full steep run.

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
| F0 live terminator | `run_energy_implicit_phase.py:1096–1097` | `if (Lgain−Lloss)/Lgain < threshold: termination_reason="cooling_balance"; break` — does **not** set `EndSimulationDirectly` |
| shadow log (P-shadow) | `run_energy_implicit_phase.py:1094` (`shadow_log.update`), `:1192` (`.write`) | records the first F0/F4 epoch to `transition_shadow.jsonl`; never acts on it |
| ε param | `phaseSwitch_LlossLgain` (read at `:1085`, default 0.05) | the cooling threshold |
| rCloud | `params['rCloud'].value` (already used `:664`) | cloud radius for F4 |
| existing R2>rCloud awareness | `:659–669`, `:885–892` (`stop_at_rCloud_nSnap`) | snapshot-stop only, not a transition |
| `large_radius` (≠ blowout) | `:1126–1129` | `R2 > stop_r` (param, default `'500'` per `registry.py:316`) → **ends sim** (`EndSimulationDirectly=True`); not a phase transition |
| blowout factory (exists, unused in 1b) | `phase_events.py:218` `make_cloud_boundary_event(rCloud)` | `R2 − rCloud`, direction +1; currently 1a→1b only |
| dead cooling factory | `phase_events.py:317` `make_cooling_balance_event(threshold=0.05)` | hardcoded 0.05; built by `build_implicit_phase_events` but never the live terminator |
| phase routing | `main.py:280,300,340` | `EndSimulationDirectly` gates 1b→1c→2; `cooling_balance` flows through to 1c→2 |

## 4. Design (shadow-first, zero production impact)
> **✅ Shipped (P-shadow + P-promote, 2026-06-16):** the param, shadow mode, and
> the promote break below are all built. `transition_trigger` is registered
> (`registry.py:347`, `default.param`); the shared F0/F4 predicates +
> `implicit_termination_reason` + `validate_transition_trigger` +
> `ShadowTransitionLog` (`trinity/phase_general/transition_shadow.py`) drive the 1b
> loop (`run_energy_implicit_phase.py` shadow update / decision break / `.write`).
> `'instantaneous'` = F0 live break + F4 shadow-logged; `'cooling_or_blowout'` =
> F0 ∨ F4 live break, routing 1b→1c→2. Invalid values raise.

**New param** `transition_trigger`, default `"instantaneous"` (current F0-only
behaviour). Register beside `phaseSwitch_LlossLgain` (registry + `default.param`).

**Shadow mode (default, `"instantaneous"`):** in the 1b loop, right before the F0
check (`:1094–1096`), compute the F4 condition `R2 > params['rCloud'].value`. Do **not**
act on it — record the first epoch where each criterion *would* fire to a sideline
file (`<output>/transition_shadow.jsonl`: `{which, t, R2, rCloud, ratio_F0}`).
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

## 6. Maintainer decisions — RESOLVED 2026-06-16 (implemented in P-promote)
1. **Route blowout through 1c (Eb drain) like cooling_balance, or skip 1c → direct
   to momentum?** ✅ **Through 1c.** The `blowout` break leaves `EndSimulationDirectly`
   untouched, exactly like `cooling_balance`, so 1b→1c→2 routing is identical; the
   residual Eb drains through the transition phase.
2. **Param shape:** ✅ **`transition_trigger ∈ {instantaneous, cooling_or_blowout}`**
   (the enum, as built in P-shadow) — not a separate boolean. Leaves room for future
   families. Invalid values raise via `validate_transition_trigger`.
3. **Blowout threshold:** ✅ **exactly `R2 > rCloud` (ratio 1.0)** — `blowout_fires`
   uses strict `>`; no fraction/tuning (P-sens: the crossing is clean at 1.0).
4. **Scope:** ✅ **shadow + promote landed together** (shadow shipped earlier the
   same day; this change adds the live F4 break + the dead-factory reconcile). The
   continuity gate (test-plan #4) moves to **P-validate** — it needs a full steep run.

## 6b. Re-entry / reversibility — surges re-inject energy (raised 2026-06-15)
**Q (maintainer):** can the bubble re-enter the energy-driven phase after the
trigger fires?

- **Current architecture: no.** Phases run strictly one-way (1a→1b→1c→2,
  `main.py:280–349`); no path moves backward. A surge re-pressurizes *within* a
  phase (β<0 ⇒ dEb/dt>0 in 1b; or stalls the 1c drain via
  `min(Ed_energy_balance, Ed_soundcrossing)`) but never reverses the sequence. True
  reversibility = large architectural change, **out of scope** (no runner changes
  beyond the switch condition).
- **Physically: re-energizing is real, acute for steep.** In `tt_steep_long` Eb is
  **monotonically rising the whole run** and **triples after blowout** (7.0e7 at
  R2>rCloud @2.728 → 2.05e8 @4 Myr); the WR/SN surges (β→−2.44 @3.08–3.33)
  *accelerate* the rise. The steep bubble is **not** thermally exhausted at blowout.

**Consequence for F4.** Blowout is **geometric** (shell escaped the cloud), firing
on an energetically-active bubble. Switching to momentum (Eb→0) discards a
still-growing Eb; the 1D justification is that a blown-out bubble **vents** its hot
interior (champagne flow, unrepresentable in 1D) rather than doing PdV work — energy
treated as lost. Explicit approximation, sharpest in the steep regime.

**Reinforces decision #1 (route blowout through 1c):** the 1c `min(...)` drain lets
a post-blowout surge *stall* the Eb loss rather than abruptly zeroing it — softer,
more physical. Neither choice re-enters energy-driven.

**Open physics question it surfaces (maintainer/paper) — RESOLVED 2026-06-16:**
for steep, is "blowout → momentum" the right model at all, or should the energy-rich
blown-out bubble get a **venting/leak treatment** (ramp `bubble_Leak`) instead of a
hard switch? ✅ **Decision: hard switch, flagged as a Paper-I caveat.** The blown-out
hot interior is treated as vented (champagne flow, unrepresentable in 1D); routing
through 1c (decision #1) already softens the discontinuity by draining the residual
Eb. No `bubble_Leak` ramp is added — that venting model is a deliberately deferred,
separate physics sub-task (its own validation), **not** silently encoded here. The
steep energy-discard approximation must be stated explicitly in the Paper-I caveats.

## 7. Out of scope (unchanged from plan)
Betadelta solver; transition/momentum runner internals beyond the switch condition;
3D mixing-layer physics; the clock-B (1c sound-crossing drain) treatment (separate
maintainer call — P0 showed 1c is short for flat and absent for steep).
