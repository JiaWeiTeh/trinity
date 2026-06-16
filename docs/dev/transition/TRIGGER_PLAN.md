# Plan: implicit→momentum transition trigger — characterize, then decide

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
> exists, revise the plan and note what changed and why (date it). Leave the plan
> better than you found it. **Keep all banner paragraphs at the top of every
> plan and analysis doc.**
>
> 💾 **Persist diagnostics — commit, don't re-run.** The container is ephemeral
> and full/hybr runs cost hours, so any diagnostic worth keeping must be saved as
> a committed artifact (a CSV/table under `docs/dev/data/`, or a force-added
> harness/figure in the relevant `docs/dev/<workstream>/` folder as the hybr work did) — never left in `/tmp` or
> an untracked `outputs/`. A future visit must be able to reproduce or compare
> against the numbers **without re-running**; record the exact config + command
> that produced each artifact.

**About this document**
- **Status (verified 2026-06-16):** 🔵 **ACTIONABLE** (verified 2026-06-16) — plan is current and accurate. **P-shadow shipped 2026-06-16** (log-only F0/F4 diagnostics, `transition_trigger` param); no trigger acts on production yet — P-promote is the next step.
- **Type:** plan — characterize the implicit→momentum transition trigger (clocks A/B, candidate families F0–F5) and decide via pre-registered gates before changing production.
- **Workstream:** `transition/` — the implicit→momentum transition trigger.
- **Where it sits:** promoted from Phase 5 of `docs/dev/archive/betadelta/HYBR_PLAN.md` → **this (entry point)** → `P0.md` (P0/P-sens results) → `pshadow-design.md` (design).
- **Code it concerns:** phase 1b implicit terminator (`run_energy_implicit_phase.py` cooling_balance block) and clock B, the 1c transition phase (`phase1c_transition/run_transition_phase.py`).
- **Linked files & data:** sibling docs `P0.md`, `pshadow-design.md`; background `docs/dev/archive/betadelta/HYBR_PLAN.md`, `docs/dev/archive/betadelta/stalling-energy-phase.md`; data `docs/dev/data/transition_*.csv`; harness `docs/dev/transition/harness/`.

Promotes **Phase 5** of `docs/dev/archive/betadelta/HYBR_PLAN.md` (deferred there) into
its own program now that the hybr solver has landed and exposed the behaviour.
This is a **measurement-first investigation**, not an implementation task: the
output of the offline phases (P0, P-sens) is *evidence* for or against each
candidate trigger; production changes only if the evidence passes a
pre-registered gate. It is designed so that
"the proposed trigger is not better" is a valid, publishable conclusion — not a
failure. Background data: `docs/dev/archive/betadelta/stalling-energy-phase.md`,
`docs/dev/archive/betadelta/PHASE2_ARMS.md`; figures `docs/dev/archive/betadelta/diagnostics/phase3_regime.png`
(headline). Work branch: `fix/transition-trigger` (optional per-phase
sub-branches, e.g. `fix/transition-trigger/p0-harvest`).

**Mandate (maintainer, 2026-06-15):** instrument *both* transition clocks and
*characterize first* — the shadow comparison decides the criterion; do **not**
pre-commit to a winner (single physical trigger vs. profile-aware configurable)
before the divergence data exists.

## The three questions

1. **What causes the very long transition duration?** — "duration" is two
   different clocks (see Current state); separate them before explaining.
2. **Is ε = 0.05 the right threshold?**
3. **Is the energy *ratio* the right metric?** Candidate families below.

## Current state (verified 2026-06-15; code line refs re-verified/refreshed 2026-06-16 — re-verify per banner)

### There are two distinct "transition" clocks
- **(A) Time to *reach* transition** (`t_trans`): the implicit phase ends on the
  cooling-balance criterion `(Lgain − Lloss)/Lgain < ε`. This is the
  "2.5 Myr / stalls forever" number in `phase3_regime.png`. **It lives *only* in
  the implicit phase (1b):** phase 1a (`run_energy_phase.py`) has *no*
  cooling/transition trigger — it ends at a fixed `TFINAL_ENERGY_PHASE` (`:137`)
  or geometric events (`build_energy_phase_events`, `:114`). So all trigger work
  is in 1b.
- **(B) Length of the transition *phase* (1c)** — a *separate* ODE phase
  (`trinity/phase1c_transition/run_transition_phase.py`). `get_ODE_transition_pure`
  (`:195`) integrates `dEb/dt = min(Ed_energy_balance, Ed_soundcrossing)` with
  `Ed_soundcrossing = −Eb/(R2/c_sound)` (`:235`;
  `c_sound = operations.get_soundspeed`, `operations.py:183`). It ends when
  `Eb < ENERGY_FLOOR = 1e3` erg (`:94`, `:761` → momentum), `reached_tmax`
  (`:604`), or `small_radius` (`:785`). **(B) is set by the sound-crossing time
  `R2/c_sound`**, and the `min()` lets a feedback surge re-inject and *stall the
  drain* — a second, distinct stall mechanism from (A).

Different physics, different fixes. Conflating them is the first trap; P0 separates them.

### ε is *already* a param — but two checks disagree
- **Live terminator** of the implicit phase is an **inline** check
  (`run_energy_implicit_phase.py:1085–1097`): reads **`phaseSwitch_LlossLgain`**
  (registered, default `0.05`, `run_const=True`, `registry.py:346`,
  `default.param:279`) and breaks on `(Lgain − Lloss)/Lgain < threshold`.
- **Dead/inconsistent path:** `build_implicit_phase_events` builds
  `make_cooling_balance_event(threshold=0.05)` (`phase_events.py:495`,
  **hardcoded 0.05**); the runner unpacks the factory (`:665`) but **never uses
  it**. So the real knob is the existing param; the hardcoded event is inert here
  (confirm against every caller of `build_implicit_phase_events`).
  **Consequence:** the ε sensitivity sweep (P-sens) needs *no code change to
  start* — sweep `phaseSwitch_LlossLgain` across runs; "Step 0" reduces to
  *reconciling/removing* the dead second check.

### What Lgain / Lloss contain (the current trigger is an INSTANTANEOUS rate-ratio)
- `Lgain = feedback_post.Lmech_total` — **instantaneous** mechanical luminosity
  (`run_energy_implicit_phase.py:1071`).
- `Lloss = bubble_props.bubble_LTotal (+ bubble_Leak)` (`:1076–1079`).
- **RESOLVED 2026-06-15 — `Lloss` is pure radiative cooling (no PdV).**
  `bubble_LTotal = L_bubble + L_conduction + L_intermediate`
  (`bubble_luminosity.py:706`, returned `:750–757`), each a radiative integral
  `∫χₑ·n²·Λ(T)·4πr²dr` (CIE) / `∫dudt(n,T,φ)·4πr²dr` (non-CIE) over the bubble
  interior — **no PdV term**, no velocity. The betadelta energy balance carries
  `4πR2²·v2·Pb` *separately*, so PdV is not double-counted in the ratio. The
  runner then adds the leak: `Lloss = bubble_LTotal + bubble_Leak`. ⇒ the energy
  ratio is a clean cooling-vs-injection fraction; every candidate metric can rest
  on it.

### No `t_cool` / `t_dyn` machinery exists
`grep` finds none in the physics path — any time-ratio criterion is *new
diagnostics*, not a re-wire.

### What the current trigger means, and its known pathology
`(Lgain − Lloss)/Lgain < 0.05` ⇔ `Lloss ≥ 0.95 Lgain` ⇔ "transition when cooling
radiates ≥95 % of the *instantaneous* injected power." It marks *"Eb stops
growing,"* not *"the thermal-pressure drive stops pushing the shell."*
**Pathology (the stall):** every feedback episode (WR onset, first SN) spikes the
`Lgain = Lmech` denominator, so the ratio **moves up and away from the threshold**
(observed 0.44 → **0.67**, i.e. *more* energy-driven) exactly when a new source
switches on. This is an artifact of testing the *instantaneous numerator*, not
physics — integrating over feedback episodes removes it. (Sign note: 0.44→0.67 is
numerically *upward*, further from the 0.05 floor.)

## Candidate trigger families (literature menu — verify each before encoding)

The current trigger is the **WARPFIELD-style energy-retention** test (TRINITY's
ancestor, Rahner et al. 2017/2019). The literature does **not** endorse a single
threshold; it offers distinct families that fire at *different* times:

| id | family | criterion | rationale | provenance (verify) |
|---|---|---|---|---|
| F0 | instantaneous rate-ratio (**current**) | `(Lgain−Lloss)/Lgain < ε` | retention drops below ε | Rahner+2017/2019 |
| F1 | **cumulative energy** | `∫Lloss dt / ∫Lgain dt > 1−η`, η≈0.2–0.4 | integrates over episodes ⇒ **no reset**; best-sourced | Mac Low & McCray 1988; Nath+ η~0.25; Sharma+ 0.2–0.4; WARPFIELD calib. |
| F2 | timescale | `t_cool/t_dyn < k` | interior becomes radiative faster than it expands | Mac Low & McCray 1988; Koo & McKee 1992 |
| F3 | force / continuity | `4πR²Pb / (surviving forces) < O(1)` | thermal drive subdominant to ram+rad; continuity-preserving | BETADELTA plan §5 |
| F4 | **blowout (geometric)** | `R2 > rCloud` | shell escapes cloud — the steep-halo fate | geometric |
| F5 | mixing-flux balance | (no sharp transition) | interface cooling tracks wind flux continuously | Lancaster+2021 (and later — verify years) |

**Critical physics note — why F3/F4 must stay in the set.** F0–F2 all assume the
transition is a *cooling* event. For a **steep r⁻² halo** the bubble expands into
ever-lower density, `Lloss ∝ n²` collapses, and **even the cumulative integral
`∫Lloss dt` may never accumulate** — so no cooling-based family fires, and the
physical fate is **blowout (F4)** or **force-subdominance (F3)**. The steep config
is therefore the *crux* case, not an afterthought, and the candidate set must
include non-cooling triggers. The likely end-state is a **profile-dependent**
trigger — which a single hardcoded scalar cannot express — but that is a
conclusion to *earn* from the harvest, not assume.

**Literature caveats (do not get these wrong in the eventual paper):**
- **Do NOT attribute** the *instantaneous* forms `t_cool = Eb/Lloss`,
  `t_dyn = Rb/vb` to Mac Low & McCray — *their* `t_cool` is a **cumulative**
  balance (`∫Lloss dt = Eb(t_R)`) and `t_dyn` is a **scale-height crossing**, not
  `R/v`. The instantaneous form is *our* construction; label it as such.
- **The `0.27 = (3/11)·Lw/Lloss` mapping is internal adiabatic algebra**, valid
  only in the adiabatic limit (which is breaking down *at* the transition) and
  inheriting `Eb = (5/11)Lw t`. Treat as "order a few tenths," nothing sharper;
  verify the arithmetic independently before relying on it.
- **Keep three fractions distinct:** 0.27 (our adiabatic algebra, OOM only), 0.35
  (CMW outer-shock 27/77, a *different* quantity), η≈0.25 (retained-energy
  calibration anchor). Verify the η range against the cited sources.
- **F5 is not a 1D trigger.** A hard switch is a 1D modeling necessity, not 3D
  physics; **cite, do not model.**

## Methodology (reuse what worked for the hybr solver)

- **Shadow / diagnostic = zero production impact.** Candidates are *computed and
  logged alongside* the live run, never allowed to drive it, until a gate passes
  (mirrors the Phase-2 shadow arms). Production snapshots byte-identical through
  P0–P-shadow (drift budget D = 0, snapshot-hash checked).
- **Report every family on every segment** — the value is *where they diverge*;
  one-family conclusions are invalid. Always report the current trigger too.
- **Snapshot timing:** all harvested quantities use the verified **left-rectangle
  rule** (snapshot k = state *before* segment k); cumulative integrals must
  respect it.
- **Reuse the existing config set — no new physics invented:** dense-flat
  (1e6, n1e5), normal-flat/"typical" (1e6, n1e3), **steep (1e6, α=−2) — include,
  it is the crux**, mock (4e3), simple (1e5, sfe 0.3), and the hunt sweep h1–h6
  (`docs/dev/archive/betadelta/velstruct/`). Add at least one config with a strong WR/SN luminosity jump
  (the reset is sharpest there). These bracket transition / long-delay / stall /
  feedback-surge / blowout.
- **Pre-registered gates**, both banner paragraphs on every results doc.

## Read first (before any code)
1. `run_energy_implicit_phase.py` — the cooling_balance block (`~:1096`): what
   `Lgain`/`Lloss` contain, where it is evaluated vs the ODE and the snapshot.
2. `run_energy_phase.py` — confirms 1a has no own trigger (verified; recheck).
3. `bubble_luminosity.py` — `get_bubbleproperties_pure` / legacy path: **is
   `bubble_LTotal` pure radiative?** how is the 3e4 K floor / cooling-switch
   involved; what `T`, `n` it integrates over.
4. `get_bubbleParams.py` — `bubble_E2P` (Eb↔Pb) and the leak luminosity.
5. `run_transition_phase.py` — clock (B): the `min(Ed_energy_balance,
   Ed_soundcrossing)` model and the `ENERGY_FLOOR` exit.
6. `dictionary.py` / `registry.py` — key-registration convention for new
   harvest/diagnostic keys.

## Phases

### P0 — Both-clocks disambiguation + offline candidate harvest (no production change)
Scratch scripts + a results doc (banner) only. Two jobs:

**(i) Disambiguate the duration (clock A vs B).** Per run log `t_trans`
(implicit end), the 1c-phase length (1c start → `Eb<floor`/`stop_t`), and inside
1c log `R2/c_sound` vs the `Ed_energy_balance` term (which `min()` branch is
active; whether a feedback surge flips it). Classify each "long" case as
A-dominated, B-dominated, or both.

**(ii) Harvest all candidate firing epochs.** From saved snapshots, per segment:
`Lgain`, `Lloss`, the current ratio and its first 0.05 crossing; `Eb`, `Rb`,
`v2`, `α (= t·v2/R2)`, `t`; and each family evaluated every segment —
F0 (`<ε`), F1 (`∫Lloss/∫Lgain > 1−η`, η∈{0.20,0.25,0.30,0.40}, left-rectangle
cumsum), F2 (`t_cool/t_dyn < k`, k∈{1,2,3}, with `t_cool=Eb/Lloss`,
`t_dyn=Rb/v2`), F3 (`4πR²Pb` vs surviving forces, respecting `max(Pb,P_HII)` —
`_analysis/check_yesno.py:11`; production force site re-verify), F4 (`R2/rCloud`).
Plus the **reference physical event**: the **PdV-inclusive net-energy zero
crossing** where `(Lgain − Lloss − 4πR2²·v2·Pb) ≤ 0` first holds (the `Eb`-peak —
"the bubble stops gaining energy"). Cross-reference the betadelta
convergence/`dMdt` outcome per segment (from the betadelta plan).
One overlay figure per config (vectorized, `trinity.mplstyle`, Wong palette):
time on x; each family's firing epoch, the `Eb`-peak, and the dMdt-failure onset
as vertical lines over `Eb(t)` and `Lloss/Lgain(t)`.

**Gate G0 (pre-registered; branches on regime):**
- **Cumulative wins on parsimony** if F1 (any η in range) fires at ≈the `Eb`-peak
  with **no reset** across the WR/SN jump *and* F2 adds no separation → adopt F1,
  stop investigating F2 as a separate mechanism (record redundant). *Valid, likely
  outcome.*
- **Timescale earns a look** if F2 fires meaningfully closer to the `Eb`-peak and
  is monotone where F1 lags or F0 resets → proceed to P-sens.
- **Steep / non-cooling regime:** if no cooling family (F0–F2) tracks the
  `Eb`-peak for the steep config, check F3/F4 — if blowout/force fires cleanly,
  the steep transition is **geometric/dynamical** (⇒ profile-dependent trigger).
- **Genuinely ambiguous:** if *nothing* tracks the peak anywhere, the 1D trigger
  is fundamentally ambiguous (Lancaster-style caveat) → document, ship nothing.
- **Clock-B sub-gate:** is the 1c drain macro-relevant (moves terminal momentum /
  R2 / transition time) or a thin tail? Decide whether B needs any treatment.

### P-sens — Sensitivity & definitional robustness (offline)
Items 1–2 run only if G0 advanced the **timescale** family (F2); items 3–5
(`Lloss` systematics, the sustained-over-`t_cross` rule, ε sensitivity) are
**family-agnostic** and run regardless of which family G0 favoured.
1. **`t_dyn` definition:** `Rb/v2` vs age `t` vs `Rb/c_sound`. If the firing epoch
   swings by more than a sound-crossing across these, F2 is too
   definition-sensitive to be primary — demote to cross-check.
2. **`t_cool` definition:** instantaneous `Eb/Lloss` vs the cumulative balance
   `∫Lloss dt = Eb` (the actual MLM88 form). Report both; if far apart, state
   which the paper means.
3. **`Lloss` systematics:** move the cooling floor (1e4 vs 3e4 K), with/without
   the leak term. Shared by all families ⇒ tests common-mode sensitivity.
4. **Sustained vs. instantaneous (the cheap fix):** require the condition to hold
   over one sound-crossing before firing; measure how much this suppresses the
   WR/SN reset for *each* family — it may rescue even F0/F1 without a new metric.
5. **ε sensitivity** on the flat configs that actually cross F0: macro response
   (`t_trans`, terminal momentum, R2, seam energy-budget closure) vs
   `phaseSwitch_LlossLgain`. Insensitive→robust (keep 0.05, report range);
   sensitive→pin dynamically.

**Gate G1:** F2 proceeds to implementation candidacy only if (a) robust to the
`t_dyn` definition within a sound-crossing, (b) fires at the `Eb`-peak as well as
or better than F1, and (c) its advantage is **not** reproduced by adding the
sustained-over-`t_cross` rule to F0/F1. If (c) fails, adopt the simpler trigger.

### P-shadow — Shadow implementation (zero production impact; only if G1 passes)
> **✅ Shipped 2026-06-16** — `transition_trigger` param (default `instantaneous`)
> + log-only `ShadowTransitionLog` (`trinity/phase_general/transition_shadow.py`,
> wired into the 1b loop) record the first F0/F4 epoch to `transition_shadow.jsonl`;
> production byte-identical; `test/test_transition_shadow.py` green. **Not yet done:**
> the dead-event-factory reconcile below (still on the P-promote tail).

Add a `transition_trigger` param (default `instantaneous` = current F0 behaviour;
threshold still `phaseSwitch_LlossLgain`). Reconcile the dead event factory
(`phase_events.py:495`) so there is one parameterized energy-ratio path.
Implement the surviving candidate(s) as **selectable** alternatives that, in
shadow mode, only **log** their firing epoch to a sideline file — production still
switches on F0; the shadow triggers record where they *would* fire.
Tests: each shadow trigger reproduces its P0 offline epoch on the same config;
the cumulative integral matches the offline left-rectangle sum to machine
precision; `transition_trigger='instantaneous'` reproduces production exactly
(byte-identical snapshot hash).

### P-promote — Promotion behind the switch (default unchanged)
Implement the winner as a fully-wired phase-switch condition selectable via
`transition_trigger`, **default still `instantaneous`**. Continuity requirement:
at the switch, `Eb`, `Rb`, `v2`, and `P_drive` must hand off to the transition
phase no more discontinuously than F0 does (the transition phase already drives on
`max(Pb, P_HII+P_ram)`, so a slightly earlier/later switch must not inject a
pressure jump).

### P-validate — Validation & default decision
Winner vs current, all configs incl. steep + one strong-WR/SN-jump config:
- **Firing epoch vs `Eb`-peak (PdV-inclusive):** winner should fire at/just after
  the peak; show F0 firing late (post-stall). Headline evidence.
- **No-reset check:** across the WR/SN jump the winner's condition is monotone;
  tabulate F0's reset (0.44→0.67) vs the winner's smooth approach.
- **Adiabatic null (hard gate):** in the pure-adiabatic Weaver test (`Lloss→0`)
  the trigger must **never** fire, under any definition. If it does → STOP,
  mis-normalized.
- **Retained-energy cross-check (Lancaster):** `Eb/∫Lgain dt` at firing should
  land near η≈0.1–0.25 (verify range). ~1 ⇒ firing far too early; ~0 ⇒ too late.
- **Macro deltas:** terminal momentum, transition time, R2 at fixed times, winner
  vs current, all configs; every change traceable to the earlier/later switch.
- **Weaver adiabatic validation** unchanged to plotting precision.

**Decision:** flip the default to the winner **only if** it fires at the
`Eb`-peak without reset across *all* configs (incl. steep) **and** the
retained-energy cross-check lands in the literature range. Otherwise default stays
`instantaneous`, alternatives remain selectable, and the finding (which trigger,
why, and the stall characterization) is written up for the Paper-I
methods/caveats **regardless**.

## Paper output (independent of which trigger wins)
- The transition is a **1D modeling necessity** (cite Lancaster: continuous
  mixing-flux balance, no sharp 3D transition).
- The chosen trigger defined precisely, with the three fractions kept distinct
  (0.27 / 0.35 / η≈0.25 — see caveats).
- **The stall as a result:** steep / low-mass bubbles with frequent reheating
  retain energy longer; quantify the delay against the chosen trigger.

## Decisions that belong to the maintainer
1. Does the 1c sound-crossing drain (clock B) need its own treatment, or is it a
   negligible tail? (P0 gives the fact; the act is a call.)
2. Single physical criterion vs. profile-aware configurable trigger — **held open
   by mandate** until the P0 divergence map exists.
3. For F2/F3: the exact `t_dyn`/force definitions and threshold constants are
   physics calls, made *after* P0/P-sens show where they fire.

## Out of scope
- The betadelta solver work (separate plan); cross-reference only for the
  dMdt-failure / convergence harvest columns.
- Any change to the transition/momentum runners beyond the switch condition.
- 3D mixing-layer physics (cannot be represented in 1D; cite, don't model).

## Risks
| risk | mitigation |
|---|---|
| "duration" stays conflated → wrong fix | P0 separates clocks A/B before any design |
| candidate set too cooling-centric → misses steep | F3/F4 kept in; steep config is a baseline, not an afterthought |
| shadow criteria perturb production | pure/diagnostic only; byte-identical snapshot check |
| picking a criterion from theory before data | mandate: characterize first; G0 divergence map gates promotion |
| `t_cool`/`t_dyn` definitions arbitrary | P-sens pins both explicitly; report sensitivity to the choice |
| over-claiming literature (η, 0.27, MLM88) | verify each source; keep the three fractions distinct |
| ε sweep changes published tracks | default held one release; P-validate attribution |
