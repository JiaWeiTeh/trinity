# S11 orchestration — Lens B (what the code claims)

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

Prose-only transcription. I have seen **no code** — every statement below is a *claim made by a
comment or docstring*, recorded so another lens can test it. Nothing here is a statement about what
the code does.

Citation convention: the source prose index gives line **ranges** for docstrings (e.g. `L3-54`).
Sub-claims drawn from inside a docstring block are cited at the block's **first** line, so several
claims share a line number (notably `phase_events.py:3`, the module docstring, and
`phase_events.py:100`, `:169`, `:288`, `:320`). Comments are cited at their own line.

---

## 1. The documented phase machine

### 1.1 Phases named

| Phase | Name in prose | Citation |
|---|---|---|
| 1a | "Phase 1a: Energy driven phase." | `trinity/main.py:241` |
| 1b | "Phase 1b: implicit energy phase" | `trinity/main.py:275` |
| 1c | "Phase 1c: transition phase" | `trinity/main.py:295` |
| 2  | "Phase 2: momentum phase" | `trinity/main.py:321` |

The energy phase is also called the **"Weaver phase"** once — "`t0 = start time for Weaver phase`"
(`trinity/main.py:224`) — a name used nowhere else in the slice's prose.

The package docstring describes the code as covering "energy-driven and momentum-driven phases"
(`trinity/__init__.py:1`) — it does not mention the implicit or transition phases at all.

### 1.2 Documented transitions and their stated conditions

All three phase-to-phase transitions come from one list in the `phase_events` module docstring
(`trinity/phase_general/phase_events.py:3`), under the heading
"**Phase-Ending Events** (move to next phase)":

```
- cloud_boundary: R2 > rCloud   (energy phase -> implicit)
- cooling_balance: L_cool ~ L_gain (implicit -> transition)
- energy_floor: Eb < threshold  (transition -> momentum)
```

Reconstructed documented graph:

```
              cloud_boundary                cooling_balance              energy_floor
 1a energy ──── R2 > rCloud ────► 1b implicit ── (Lgain-Lloss)/Lgain ──► 1c transition ── Eb < 1e3 ──► 2 momentum
   │            (dir = +1,                        < threshold                (dir = -1,                    │
   │             "from below")                    [segment-level]             "from above")                │
   │                                                                                                       │
   └── early exit: stop_at_rCloud_nSnap == 0 ⇒ terminate at R2 = rCloud,                                    │
       "we explicitly do NOT want phases 1b/1c/2 to advance past it" (main.py:260)                          │
                                                                                                           │
 every phase additionally carries simulation-ending events (min_radius / max_radius / velocity_runaway);    │
 phase 2 carries ONLY those — no phase-ending event is documented for the momentum phase ◄─────────────────┘
```

Per-transition supporting prose:

- **1a → 1b.** "Create event that triggers when R2 reaches cloud edge. Used in energy phase to
  detect when shell reaches cloud boundary, triggering transition to implicit phase. … Event
  function for solve_ivp with terminal=True, direction=1."
  (`trinity/phase_general/phase_events.py:221`); "Only trigger when R2 crosses rCloud from below"
  (`:244`); "Phase ending, not simulation ending" (`:246`).
- **1b → 1c.** "Event triggers when (Lgain - Lloss) / Lgain < threshold"
  (`trinity/phase_general/phase_events.py:320`); "Trigger when ratio falls below threshold" (`:349`);
  "Phase ending" (`:351`); "No event if no gain" (`:344`).
- **1c → 2.** "Create event that triggers when bubble energy falls below threshold. Used in
  transition phase to detect when thermal energy is negligible, triggering transition to momentum
  phase. … terminal=True, direction=-1" (`trinity/phase_general/phase_events.py:254`); "Only trigger
  when Eb crosses threshold from above" (`:279`); "Phase ending, not simulation ending" (`:281`).
- **Early exit after 1a.** "stop_at_rCloud_nSnap == 0: terminate now if R2 has reached the cloud
  edge. The energy-phase reconciliation snapshot already captured R2 = rCloud, and we explicitly do
  NOT want phases 1b/1c/2 to advance past it." (`trinity/main.py:260`).

### 1.3 Documented gaps in the graph

- No prose states what ends **phase 1a** if `rCloud` is never reached (no time-limit event is
  documented for the energy phase; `build_energy_phase_events` lists only cloud_boundary,
  min_radius, velocity_runaway — `trinity/phase_general/phase_events.py:424`).
- No prose names a **time-based stop** (`tStop`/`tEnd`) anywhere in the slice. The only reference is
  "until end of simulation" (`trinity/main.py:217`).
- **Phase 2 (momentum)** has no documented phase-ending event
  (`trinity/phase_general/phase_events.py:547` lists min_radius, max_radius, velocity_runaway only)
  and no documented onward transition.
- No recollapse → new-generation transition is implemented; see §6 admissions
  (`trinity/main.py:209`).

---

## 2. Event semantics (verbatim claims)

### 2.1 The module's own three-way taxonomy — `trinity/phase_general/phase_events.py:3`

> "Events are categorized by their consequence:
> 1. **Simulation-Ending Events** (EndSimulationDirectly=True):
>    - min_radius: R2 < coll_r (shell collapse)
>    - max_radius: R2 > stop_r (expansion limit)
>    - velocity_runaway: |v2| > threshold (numerical instability)
> 2. **Phase-Ending Events** (move to next phase):
>    - cloud_boundary: R2 > rCloud (energy phase -> implicit)
>    - cooling_balance: L_cool ~ L_gain (implicit -> transition)
>    - energy_floor: Eb < threshold (transition -> momentum)
> 3. **Monitoring Events** (non-terminal; record a crossing only):
>    - velocity_sign: v2 crosses zero (collapse onset detection)"

The same docstring opens with the blanket claim: "Centralized module for ODE event functions used
across all simulation phases. **These events are passed to scipy.integrate.solve_ivp** to enable safe
termination during integration." (`trinity/phase_general/phase_events.py:3`).

### 2.2 Per-event transcription

Columns: what it claims to detect · claimed crossing direction · **terminal status as stated**.

| Event | Claimed detection | Claimed direction | Terminal? (verbatim) | Citations |
|---|---|---|---|---|
| `min_radius` | "triggers when R2 falls below min_r"; "prevents LSODA from crashing when R2 approaches zero during rapid collapse" | `direction=-1`; "Only trigger when R2 crosses min_r from above" | **TERMINAL** — "The event is terminal - integration stops immediately"; "Event function for solve_ivp with terminal=True, direction=-1" | `:100`, `:125` |
| `max_radius` | "triggers when R2 exceeds max_r"; "Used to stop simulation when shell expands beyond stop_r limit" | `direction=1`; "Only trigger when R2 crosses max_r from below" | **TERMINAL** — "Event function for solve_ivp with terminal=True, direction=1" | `:135`, `:157` |
| `velocity_runaway` | "triggers on extreme velocity magnitude"; "catches runaway dynamics before the solver becomes numerically unstable" | `"collapse"` ⇒ "Triggers when v2 < -v_max"; `"expansion"` ⇒ "Triggers when v2 > v_max"; `"both"` ⇒ "Triggers when \|v2\| > v_max" | **TERMINAL** — "Event function for solve_ivp with terminal=True" | `:169`, `:192`, `:198`, `:201`, `:204` |
| `cloud_boundary` | "triggers when R2 reaches cloud edge … triggering transition to implicit phase" | `direction=1`; "Only trigger when R2 crosses rCloud from below" | **TERMINAL to the solver, PHASE-ending only** — "terminal=True, direction=1" *and* "**Phase ending, not simulation ending**" | `:221`, `:244`, `:246` |
| `energy_floor` | "triggers when bubble energy falls below threshold … thermal energy is negligible" | `direction=-1`; "Only trigger when Eb crosses threshold from above" | **TERMINAL to the solver, PHASE-ending only** — "terminal=True, direction=-1" *and* "**Phase ending, not simulation ending**" | `:254`, `:279`, `:281` |
| `velocity_sign` | "triggers when velocity changes sign. Used to detect collapse onset (v2 going from positive to negative)" | `direction=-1`; "only triggers on positive-to-negative crossing"; "Only trigger when v2 goes positive -> negative" | **EXPLICITLY NOT TERMINAL** — "Event function for solve_ivp with **terminal=False (monitoring only)**"; "**Non-terminal by default - just records the crossing**"; module docstring: "(non-terminal; record a crossing only)"; builder: "velocity_sign: v2 crosses zero (**monitoring, non-terminal**)" | `:288`, `:310`, `:311`, `:3`, `:459` |
| `cooling_balance` | "cooling balance detection"; triggers when "(Lgain - Lloss) / Lgain < threshold" | "Trigger when ratio falls below threshold" (no solve_ivp direction stated — it is not a solve_ivp event) | **PHASE-ending** — "Phase ending"; but see the NOTE below: it is claimed to need **segment-level** checking, not solver-level | `:320`, `:344`, `:349`, `:351` |

**The four explicit non-terminal / not-simulation-ending declarations** — the highest-value
checkable runtime claims in this slice, quoted exactly:

1. `trinity/phase_general/phase_events.py:310` — "`# Non-terminal by default - just records the crossing`"
   (about `velocity_sign`).
2. `trinity/phase_general/phase_events.py:288` — "`terminal=False (monitoring only)`" (about `velocity_sign`).
3. `trinity/phase_general/phase_events.py:246` — "`# Phase ending, not simulation ending`" (about `cloud_boundary`).
4. `trinity/phase_general/phase_events.py:281` — "`# Phase ending, not simulation ending`" (about `energy_floor`).

Plus the taxonomy line at `trinity/phase_general/phase_events.py:3`: "**Monitoring Events**
(non-terminal; record a crossing only): - velocity_sign: v2 crosses zero (collapse onset detection)".

### 2.3 Which events each phase is documented to carry

| Phase | Documented event list | Citation |
|---|---|---|
| 1a energy | cloud_boundary (phase ending) · min_radius (simulation ending) · velocity_runaway (simulation ending). **No max_radius / stop_r.** | `:424` |
| 1b implicit | velocity_sign (monitoring, non-terminal) · min_radius · max_radius · velocity_runaway; "Also returns cooling_balance factory for segment-level checking" | `:459` |
| 1c transition | energy_floor (phase ending -> momentum) · min_radius · max_radius · velocity_runaway | `:505` |
| 2 momentum | min_radius · max_radius · velocity_runaway | `:547` |

Conditional-registration claim, stated identically three times and **absent for the energy phase**:
"`# Only add max_radius event if stop_r is set`" (`:493`, `:537`, `:575`).

`velocity_sign` (collapse-onset monitoring) is documented **only** for the implicit phase.

---

## 3. Termination and end codes

### 3.1 Every documented way a run can end

| # | End path | Stated meaning | Citation |
|---|---|---|---|
| 1 | `min_radius` | "R2 < coll_r (**shell collapse**)"; also "prevents LSODA from crashing when R2 approaches zero" | `:3`, `:100` |
| 2 | `max_radius` | "R2 > stop_r (**expansion limit**)" | `:3` |
| 3 | `velocity_runaway` | "\|v2\| > threshold (**numerical instability**)" | `:3` |
| 4 | `stop_at_rCloud_nSnap == 0` | "terminate now if R2 has reached the cloud edge" | `trinity/main.py:260` |
| 5 | "end of simulation" | undefined in this slice — no threshold, no code, no unit | `trinity/main.py:217` |
| 6 | "next recollapse" | claimed as a stopping condition of `run_expansion`, but the loop that would use it is a TODO | `trinity/main.py:217`, `:209` |

### 3.2 The end-code carrier — `EventResult`

Docstring: "Container for event detection results." (`trinity/phase_general/phase_events.py:83`).
Documented fields, verbatim per-field comments:

- `:86` — "Which event in the list triggered (**-1 if none**)"
- `:87` — "Time of event (**NaN if not triggered**)"
- `:88` — "State at event (**empty if not triggered**)"
- `:89` — "True if simulation should end"
- `:90` — "Short code for `termination_reason`"
- `:91` — "Human-readable message for `SimulationEndReason`"
- `:92` — "Exit code for `SimulationEndCode`"

Event objects are separately documented to carry: "`event.name`, `event.is_simulation_ending`,
`event.reason_code`, `event.reason_message`" (`:100`). **No exit-code attribute is documented on the
event**, yet `EventResult` carries one (`:92`).

### 3.3 Reporting path

- "Log completion" (`trinity/main.py:182`); "Write simulation end report to file" (`:189`).
- "Spell out the **stopping fate** (and final state) in the log, not just the bare exit code — this is
  the **headline scientific result** of the run." (`trinity/main.py:192`).
- "Write termination debug block into `metadata.json[termination_debug]`" (`trinity/main.py:200`).

### 3.4 Solver failure vs physical outcome — what the prose does and does not say

**No prose in this slice states how a solver failure is distinguished from a physical outcome.**
What the prose *does* say is that two of the three simulation-ending events are motivated by
*numerical* protection yet are reported through the *same* `reason_code` / `SimulationEndCode`
channel as physical fates:

- `min_radius`: "This prevents **LSODA from crashing**" (`:100`) — but the taxonomy labels it
  "(shell collapse)" (`:3`), a physical outcome.
- `velocity_runaway`: "catches runaway dynamics **before the solver becomes numerically unstable**"
  (`:169`); taxonomy labels it "(numerical instability)" (`:3`) — yet it sits under
  "Simulation-Ending Events (EndSimulationDirectly=True)" alongside the physical ones.

There is no documented `SimulationEndCode` value meaning "solver failed", no documented handling of
`sol.status < 0` / non-convergence, and no documented distinction in the end report between "the
bubble collapsed" and "the integrator was about to blow up".

---

## 4. Contracts

### 4.1 Orchestration order — `start_expansion` (`trinity/main.py:82`)

Docstring: "This wrapper takes in the parameters and feed them into smaller functions. Parameters:
`params : Object` — An object describing TRINITY parameters. Returns: None."
(Note: the `phase_events` builders instead document `params : dict` — `:424`, `:459`, `:505`, `:547`.)

Documented sequence:

1. **Step 0: Preliminary** (`:98`). "Note: Logging is configured in `run.py` before this function is
   called. This fallback ensures logging works if `main.py` is called directly." (`:102`).
   "Record timestamp" (`:110`).
2. **A: Initialising cloud properties** (`:117`). "Step 1: Obtain initial cloud properties" (`:120`).
   **Ordering requirement:** "rCloud is **now known**; warn the user if stop_r will starve
   stop_at_rCloud_nSnap of any chance to fire (or race with it)." (`:126`) — i.e. the stop_r/rCloud
   check must run *after* cloud init.
3. **Step 2: Obtain SPS feedback parameters** (`:138`). "The loader handles both the legacy SB99
   grammar (`sps_path = def_path`) and user-defined `sps_path` files" (`:139`). "Get SPS data and
   interpolation functions." (`:146`).
4. **Step 3: get cooling structure for CIE (since it is non time dependent)** (`:156`). "Values for
   non-CIE cooling curve will be calculated **along the simulation**, since it depends on time
   evolution." (`:158`). "for metallicity, here we need to take care of both CIE and nonCIE part."
   (`:160`). Then "unpack from file" (`:164`), "create interpolation" (`:166`), "update" (`:168`).
5. **Begin simulation** (`:176`).
6. Post-run: log completion → end report → stopping-fate log → `metadata.json[termination_debug]`
   (`:182`, `:189`, `:192`, `:200`).

### 4.2 `_check_stop_r_rCloud_interaction` (`trinity/main.py:42`)

Stated contract, verbatim: "Decide whether `stop_r` conflicts with `stop_at_rCloud_nSnap`. rCloud is
derived from the cloud properties at init time, so a user setting `stop_r` in a `.param` file may
accidentally pick a value **smaller than rCloud** and **silently disable** their
`stop_at_rCloud_nSnap` termination. **Both knobs are valid independently — this is a UX warning, not
an error.** Returns `(level, message)`: level is `"warning"`, `"info"`, or `None`. message is the log
text (or `None` when no log is needed)."

Supporting threshold comment: "Threshold above which a `stop_r` value is considered 'comfortably
above' rCloud — meaning `stop_at_rCloud_nSnap` will almost certainly fire first. **Below this
multiple of rCloud, the two termination conditions race.**" (`trinity/main.py:35`). The numeric value
of the multiple is not stated in prose.

### 4.3 `run_expansion` (`trinity/main.py:217`) — state carried across phase boundaries

Docstring: "Model evolution of the cloud (both energy- and momentum-phase) **until next recollapse**
or (if no re-collapse) until end of simulation."

State-vector contract (`trinity/main.py:224`–`:229`):

- "`t0` = start time for Weaver phase" — **no unit stated**
- "`y0 = [r0, v0, E0, T0]`"
- "`R2` = initial outer bubble radius (= inner shell edge) (**pc**)"
- "`v2` = initial velocity (**pc/Myr**)"
- "`Eb` = initial energy" — **no unit stated here**
- "`T0` = initial temperature (**K**)"

Cross-boundary state claims:

- **After 1c:** "Since cooling is not needed anymore after this phase, **we reset values**.
  `COOLING_PHASE_KEYS` contains all cooling-related parameters that can be cleared."
  (`trinity/main.py:314`).
- **At 1c → 2:** "`Eb` is **inherited** from the transition phase (near `ENERGY_FLOOR = 1e3`). The
  momentum phase runner **sets `Eb = 0`** at its initialization. **Do NOT set `Eb=1` here — it was
  dead code that caused confusion.**" (`trinity/main.py:329`).
- Boundary instrumentation: "`--- Transition -> Momentum boundary diagnostic ---`"
  (`trinity/main.py:335`).
- "Flush parameters to disk" (`trinity/main.py:354`).

### 4.4 `check_event_termination` (`trinity/phase_general/phase_events.py:364`)

"Check solve_ivp solution for event termination. `sol : OdeResult` — Solution object from solve_ivp.
`events : list of callable` — List of event functions passed to solve_ivp. Returns `result :
EventResult`." Body comment: "Check each event" (`:391`). **Precedence when more than one event fires
is not documented**, and `EventResult` records a single index (`:86`).

### 4.5 `apply_event_result` (`trinity/phase_general/phase_events.py:590`)

"Apply event result to params dictionary. **Updates params with final state and termination info if
event triggered.** `params : dict` — Parameter dictionary to update. `result : EventResult`.
`t : float` — Time at event. `y : np.ndarray` — State vector at event. `state_keys : list of str` —
Keys for state variables in order matching y. **Default `['R2', 'v2']`.**"

Documented side effects: "Update time" (`:611`) · "Update state variables" (`:614`) · "Set
termination info **if simulation-ending**" (`:619`) · "**Mark collapse if it's a collapse-related
event**" (`:626`). Which events count as "collapse-related" is never enumerated.

### 4.6 `cooling_balance` architectural contract (`trinity/phase_general/phase_events.py:320`)

"Returns a **factory** that creates the actual event given current `Lgain`/`Lloss`. This is needed
because `Lgain`/`Lloss` change **each segment**. **NOTE: This event requires segment-level checking
since `Lgain`/`Lloss` are computed during segment setup, not available to `solve_ivp`.**"
`build_implicit_phase_events` (`:459`) correspondingly returns "`cooling_balance_factory : callable`
— Factory function to create cooling_balance event for each segment", separately from the event list.

### 4.7 Public API claims — `trinity/__init__.py:1`

"`params = read_param.read_param('my_simulation.param')` … `main.start_expansion(params)`"; and for
outputs "`output = TrinityOutput.open('simulation.jsonl')`; `times = output.get('t_now')`;
`radii = output.get('R2')`".

Module-docstring usage example (`trinity/phase_general/phase_events.py:3`):
`events = [make_min_radius_event(coll_r * 1.5), make_max_radius_event(stop_r)]`;
`result = check_event_termination(sol, events)`; `if result.triggered: print(f"Event '{result.name}'
triggered at t={result.t}")`.

---

## 5. Formulas, thresholds and units stated in prose

| Quantity | Stated value / formula | Unit as stated | Citation |
|---|---|---|---|
| `MIN_RADIUS_SAFETY` | "absolute minimum radius" | **pc** | `:71` |
| coll_r multiplier | "Factor above coll_r for early termination" (value not stated); example uses `coll_r * 1.5` | dimensionless | `:72`, `:3` |
| inward velocity runaway constant | "~490 km/s — extreme inward velocity" | **pc/Myr**, annotated in km/s | `:73` |
| outward velocity runaway constant | "~978 km/s — extreme outward velocity" | **pc/Myr**, annotated in km/s | `:74` |
| `v_max` default | "Default 500 pc/Myr for collapse" | pc/Myr | `:169` |
| `ENERGY_FLOOR` | `1e3` | "**code/AU units, Msun*pc^2/Myr^2**" | `trinity/main.py:329`, `phase_events.py:254`, `:505` |
| energy_floor default in builder | "Default 1e3" | same | `:505` |
| `y_index` for `Eb` | "Default **2** for `[R2, v2, Eb, ...]`" | index | `:254` |
| `y_index` for `v2` | "Default **1** for `[R2, v2, ...]`" | index | `:288` |
| cooling-balance trigger | "(Lgain - Lloss) / Lgain < threshold" | dimensionless ratio | `:320` |
| cooling-balance guard | "No event if no gain" | — | `:344` |
| stop_r/rCloud "comfortably above" | a multiple of rCloud (**value not stated**) | multiple of pc | `trinity/main.py:35` |
| SPS mass-scaling validity | "~>1e5" cluster mass | (Msun implied, not stated) | `trinity/main.py:142` |

**Time units:** the prose never states the unit of `t`, `t0`, or `EventResult.t`. Myr is *implied*
transitively (velocities pc/Myr at `trinity/main.py:227`, energies Msun·pc²/Myr² at `:254`) but is
never asserted for a time quantity anywhere in the slice.

**Unit cross-check on the two runaway constants:** 1 pc/Myr ≈ 0.978 km/s, so "~490 km/s" ≈ 500 pc/Myr
and "~978 km/s" ≈ 1000 pc/Myr — the two constants are documented as an **asymmetric pair** (inward
half of outward), while the factory exposes a single symmetric `v_max` used as `|v2| > v_max` in
`"both"` mode (`:204`).

---

## 6. Admissions of debt (verbatim)

| Kind | Text | Citation |
|---|---|---|
| TODO | "put this in read_param, and make it depend on param file." | `trinity/main.py:101` |
| Approximation | "Scaling factor for cluster masses. **Though this might only be accurate for high mass clusters (~>1e5)** in which the IMF is fully sampled." | `trinity/main.py:142` |
| TODO | "**if tSF != 0.: we would actually need to shift the feedback parameters by tSF** / update" | `trinity/main.py:149` |
| TODO | "add loop so that this simulation starts over with old generation of parameter to simulate new starburst environment" (under "STEP 2: In case of recollapse, prepare next expansion") | `trinity/main.py:208`, `:209` |
| Dead code | "**Do NOT set Eb=1 here — it was dead code that caused confusion.**" | `trinity/main.py:331` |
| Fallback duplication | "Logging is configured in run.py before this function is called. **This fallback ensures logging works if main.py is called directly.**" | `trinity/main.py:102` |
| Architectural limitation | "**NOTE: This event requires segment-level checking since Lgain/Lloss are computed during segment setup, not available to solve_ivp.**" | `phase_events.py:320` |
| Soft default | "Non-terminal **by default** - just records the crossing" (no override is documented in the signature) | `phase_events.py:310` |
| Deliberate non-error | "**Both knobs are valid independently — this is a UX warning, not an error.**" | `trinity/main.py:42` |

No `FIXME`, `XXX`, "hack", "temporary", "unclear", or "I think" appears in this slice's prose.

---

## 7. Flagged contradictions, ambiguities and vague claims

### 7.1 Prose contradicting prose

1. **`min_radius` trigger radius has two documented values.** The taxonomy says the condition is
   "R2 < **coll_r**" (`:3`); the factory says `min_r` is "Typically **coll_r * factor** or
   MIN_RADIUS_SAFETY" (`:100`), and the usage example instantiates `make_min_radius_event(coll_r *
   1.5)` (`:3`). If the factor is >1, the run terminates *above* `coll_r`, not below it, and the
   builders' phrase "min_radius: R2 < **safety threshold**" (`:424`, `:459`, `:505`, `:547`) is too
   vague to disambiguate. → **S11-B-01**.
2. **`min_radius` has two meanings.** Physical ("shell collapse", `:3`) vs numerical ("prevents
   LSODA from crashing", `:100`). One end code, two stories. → **S11-B-02**.
3. **`cooling_balance` is and is not a solve_ivp event.** Module docstring: "These events are passed
   to scipy.integrate.solve_ivp" and it is listed among the phase-ending events (`:3`); its own
   factory says it "requires segment-level checking … not available to solve_ivp" (`:320`).
   → **S11-B-12**.
4. **Usage example references undocumented `EventResult` attributes.** The example uses
   `result.triggered` and `result.name` (`:3`), but the documented field list (`:86`–`:92`) names an
   *index* ("-1 if none"), t, y, an end-simulation flag, `reason_code`, `reason_message` and an exit
   code — no `triggered`, no `name`. → **S11-B-07**.
5. **Three names for the same "this ends the run" concept:** `EndSimulationDirectly=True` (`:3`),
   `event.is_simulation_ending` (`:100`), and the unnamed field "True if simulation should end"
   (`:89`). → **S11-B-08**.
6. **`run_expansion` claims recollapse-driven looping the TODO says does not exist.** "until next
   recollapse" (`trinity/main.py:217`) vs "TODO: add loop so that this simulation starts over…"
   (`trinity/main.py:209`). → **S11-B-17**.
7. **`params` type:** "`params : Object`" (`trinity/main.py:82`) vs "`params : dict`" in all four
   event builders. → **S11-B-24**.

### 7.2 An event declared non-terminal in one place, terminal in another

No event is declared both terminal and non-terminal outright. But two events are declared
`terminal=True` *to the solver* while simultaneously declared "Phase ending, **not simulation
ending**" (`cloud_boundary` `:221`/`:246`; `energy_floor` `:254`/`:281`) — i.e. `terminal` is claimed
to mean two different things depending on the event, and only prose distinguishes them. Whether the
run actually resumes after these solver-terminal events is a directly checkable claim.

`velocity_sign` is the only event declared non-terminal, in four separate places (`:3`, `:288`,
`:310`, `:459`) — all consistent. Its hedge "by default" (`:310`) implies an override the documented
signature (`y_index`, `name`, `:288`) does not expose. → **S11-B-22**.

### 7.3 A documented transition condition another comment contradicts / undercuts

- **stop_r enforcement direction.** `max_radius` is documented as `direction=1`, "Only trigger when
  R2 crosses max_r **from below**" (`:157`), and is **absent from the energy phase** (`:424`). Taken
  together the prose implies: if `stop_r < rCloud`, the shell passes `stop_r` during phase 1a where
  no `max_radius` event exists, and thereafter R2 is already *above* `max_r`, so no from-below
  crossing can occur in 1b/1c/2. → **S11-B-03**.
- The `_check_stop_r_rCloud_interaction` docstring asserts the *opposite* causality — that a small
  `stop_r` "silently disable[s] their stop_at_rCloud_nSnap termination" (`trinity/main.py:42`), and
  `trinity/main.py:35` says the two "race". → **S11-B-04**.

### 7.4 A documented end code with two meanings

- `min_radius` (§7.1 item 2) → **S11-B-02**.
- `velocity_runaway` is labelled "numerical instability" (`:3`) yet is issued as a first-class
  simulation end code with a `reason_code` / `SimulationEndCode`, indistinguishable in kind from
  physical fates. → **S11-B-06**.

### 7.5 Claims too vague to check as written

- "min_radius: R2 < **safety threshold**" (`:424`, `:459`, `:505`, `:547`) — which threshold?
- "**Mark collapse if it's a collapse-related event**" (`:626`) — the set of collapse-related events
  is never enumerated; `velocity_sign` (collapse onset, non-terminal) and `min_radius` (collapse,
  terminal) are both candidates. → **S11-B-11** (precedence) and this ambiguity.
- "Threshold above which a stop_r value is considered 'comfortably above' rCloud"
  (`trinity/main.py:35`) — the multiple is not stated in prose.
- "until end of simulation" (`trinity/main.py:217`) — no threshold, unit, event or end code.
- "`stop_at_rCloud_nSnap`": only the `== 0` case is documented (`trinity/main.py:260`). The meaning
  of non-zero values (the name implies a snapshot count) is stated nowhere. → **S11-B-25**.
- "`COOLING_PHASE_KEYS` contains all cooling-related parameters **that can be cleared**"
  (`trinity/main.py:314`) — "can be" leaves the actual membership and the consumer set unstated.

---

## 8. Findings

```json
[
  {
    "id": "S11-B-01",
    "file": "trinity/phase_general/phase_events.py",
    "line": 3,
    "class": "coefficient",
    "severity": "S2",
    "claim": "The module docstring states the min_radius simulation-ending condition is 'R2 < coll_r (shell collapse)', but the factory docstring says min_r is 'Typically coll_r * factor or MIN_RADIUS_SAFETY' (line 100) and the module's own usage example instantiates make_min_radius_event(coll_r * 1.5). The four phase builders describe it only as 'R2 < safety threshold'.",
    "evidence": "phase_events.py:3 '- min_radius: R2 < coll_r (shell collapse)' and 'events = [ make_min_radius_event(coll_r * 1.5), ... ]'; phase_events.py:100 'min_r : float Minimum allowed radius (pc). Typically coll_r * factor or MIN_RADIUS_SAFETY'; phase_events.py:72 '# Factor above coll_r for early termination'; phase_events.py:424/459/505/547 'min_radius: R2 < safety threshold'.",
    "expected": "One documented termination radius. If the constant factor is >1, the run stops before R2 reaches coll_r and the taxonomy line 'R2 < coll_r' is wrong; the actual radius should be stated once and referenced.",
    "failure_scenario": "A run reported as 'shell collapse' actually stopped at 1.5x coll_r (or at MIN_RADIUS_SAFETY, whichever the builders pass), so the recorded collapse radius and the collapse/no-collapse classification are systematically offset from the documented physical criterion.",
    "repro": "Read the CONST at phase_events.py:71-72 and the min_r argument each builder passes to make_min_radius_event; compare with coll_r. Then run param/simple_cluster.param with a config that collapses and check the final R2 against coll_r.",
    "confidence": "high"
  },
  {
    "id": "S11-B-02",
    "file": "trinity/phase_general/phase_events.py",
    "line": 100,
    "class": "silent-failure",
    "severity": "S2",
    "claim": "min_radius is documented simultaneously as a numerical guard ('prevents LSODA from crashing when R2 approaches zero during rapid collapse') and as the physical 'shell collapse' end code. One end code is claimed to carry two distinct meanings.",
    "evidence": "phase_events.py:100 'This prevents LSODA from crashing when R2 approaches zero during rapid collapse. The event is terminal - integration stops immediately.'; phase_events.py:3 '- min_radius: R2 < coll_r (shell collapse)'.",
    "expected": "Either the reason_code/SimulationEndCode distinguishes 'integrator protection trip' from 'physical collapse', or the prose states that they are deliberately the same code.",
    "failure_scenario": "A run that tripped the solver-protection floor is published as a physical recollapse (the 'headline scientific result' per main.py:192) with no way to tell the two apart from the end report or metadata.json[termination_debug].",
    "repro": "Inspect the reason_code / SimulationEndCode assigned in make_min_radius_event and check whether any distinct code exists for integrator-protection stops; grep the end-report writer for how the code is rendered.",
    "confidence": "high"
  },
  {
    "id": "S11-B-03",
    "file": "trinity/phase_general/phase_events.py",
    "line": 424,
    "class": "silent-failure",
    "severity": "S2",
    "claim": "The energy phase's documented event list contains no max_radius/stop_r event, while max_radius is documented as direction=1, 'Only trigger when R2 crosses max_r from below'. Together the prose implies a run with stop_r < rCloud passes stop_r unguarded in phase 1a and can never produce a from-below crossing in phases 1b/1c/2, so stop_r never terminates the run.",
    "evidence": "phase_events.py:424 'Events: - cloud_boundary: R2 > rCloud (phase ending) - min_radius ... - velocity_runaway ...' (no max_radius); phase_events.py:135 'Event function for solve_ivp with terminal=True, direction=1'; phase_events.py:157 '# Only trigger when R2 crosses max_r from below'; phase_events.py:493/537/575 '# Only add max_radius event if stop_r is set'.",
    "expected": "Either the energy phase also registers max_radius, or the later phases' max_radius handles the already-above-threshold case (initial-value check rather than a from-below crossing), or the prose documents that stop_r is only honoured beyond rCloud.",
    "failure_scenario": "A user sets stop_r below rCloud expecting the run to stop there; the shell blows past it during the energy phase and the run continues to rCloud and beyond, terminating for an unrelated reason (or not at all) with no warning that stop_r was never armed.",
    "repro": "Run a config with stop_r set to ~0.5 * rCloud and check whether termination_reason is the max_radius/stop_r code and whether final R2 <= stop_r. Also check whether build_energy_phase_events registers any stop_r event.",
    "confidence": "medium"
  },
  {
    "id": "S11-B-04",
    "file": "trinity/main.py",
    "line": 42,
    "class": "divergence",
    "severity": "S3",
    "claim": "_check_stop_r_rCloud_interaction's docstring asserts that a stop_r smaller than rCloud 'silently disable[s] their stop_at_rCloud_nSnap termination' (i.e. stop_r wins), and main.py:35 says the two conditions 'race'. The event-set prose implies the opposite: stop_r is not armed in the energy phase at all, so rCloud is reached first by construction.",
    "evidence": "main.py:42 'a user setting stop_r in a .param file may accidentally pick a value smaller than rCloud and silently disable their stop_at_rCloud_nSnap termination'; main.py:35 'Below this multiple of rCloud, the two termination conditions race.'; phase_events.py:424 energy-phase event list omits max_radius.",
    "expected": "The warning text should describe the actual precedence between stop_r and stop_at_rCloud_nSnap, and the 'race' claim should be either demonstrated or removed.",
    "failure_scenario": "The user is warned about the wrong hazard: told stop_r will pre-empt their rCloud snapshot, when in fact their stop_r may be silently ignored.",
    "repro": "Run two configs, stop_r = 0.5*rCloud and stop_r = 0.9*rCloud, with stop_at_rCloud_nSnap set, and record which termination_reason each produces; compare with the emitted warning level ('warning'/'info'/None).",
    "confidence": "medium"
  },
  {
    "id": "S11-B-05",
    "file": "trinity/phase_general/phase_events.py",
    "line": 73,
    "class": "units",
    "severity": "S3",
    "claim": "Two asymmetric velocity-runaway constants are documented (inward '~490 km/s', outward '~978 km/s', i.e. roughly 500 and 1000 pc/Myr), but make_velocity_runaway_event exposes a single v_max ('Default 500 pc/Myr for collapse') and the 'both' mode is documented as the symmetric test |v2| > v_max.",
    "evidence": "phase_events.py:73 '# pc/Myr (~490 km/s) - extreme inward velocity'; phase_events.py:74 '# pc/Myr (~978 km/s) - extreme outward velocity'; phase_events.py:169 'v_max : float Maximum velocity magnitude (pc/Myr). Default 500 pc/Myr for collapse.'; phase_events.py:204 '# Triggers when |v2| > v_max'.",
    "expected": "Either 'both' mode applies the asymmetric pair (inward 500, outward 1000 pc/Myr), or the two constants are documented as belonging to the 'collapse' and 'expansion' modes respectively and 'both' is documented as using only one of them.",
    "failure_scenario": "Builders passing direction='both' with the collapse default clip outward velocities at 500 pc/Myr (~489 km/s) instead of the documented ~978 km/s outward limit, terminating fast but physical expansions as 'numerical instability'.",
    "repro": "Read the two constants' values and which one each of build_{energy,implicit,transition,momentum}_phase_events passes to make_velocity_runaway_event, plus the direction string used.",
    "confidence": "medium"
  },
  {
    "id": "S11-B-06",
    "file": "trinity/phase_general/phase_events.py",
    "line": 3,
    "class": "silent-failure",
    "severity": "S2",
    "claim": "velocity_runaway is documented as detecting 'numerical instability' yet is classified as a Simulation-Ending Event carrying a reason_code / SimulationEndReason / SimulationEndCode identical in kind to the physical fates. No prose anywhere in the slice documents how solver failure is distinguished from a physical outcome, and no end code for solver failure is named.",
    "evidence": "phase_events.py:3 '1. **Simulation-Ending Events** (EndSimulationDirectly=True): ... - velocity_runaway: |v2| > threshold (numerical instability)'; phase_events.py:169 'catches runaway dynamics before the solver becomes numerically unstable'; phase_events.py:90-92 '# Short code for termination_reason / # Human-readable message for SimulationEndReason / # Exit code for SimulationEndCode'; main.py:192 'Spell out the stopping fate ... this is the headline scientific result of the run.'",
    "expected": "A documented, distinct end-code class for numerical/solver terminations vs physical fates, and documented handling of solve_ivp failure (sol.status < 0, non-convergence), which the prose never mentions.",
    "failure_scenario": "A sweep aggregates end codes across configs; runs that hit the integrator guardrail are counted as physical outcomes, biasing the reported fate statistics with no flag distinguishing them.",
    "repro": "Enumerate the SimulationEndCode values and check whether any corresponds to solver failure; then check whether run_expansion inspects sol.status/sol.success at all after each solve_ivp call.",
    "confidence": "high"
  },
  {
    "id": "S11-B-07",
    "file": "trinity/phase_general/phase_events.py",
    "line": 3,
    "class": "citation",
    "severity": "S3",
    "claim": "The module docstring's usage example calls result.triggered and result.name, but the documented EventResult field comments describe an index ('-1 if none'), t, y, an end-simulation flag, reason_code, reason_message and an exit code — no 'triggered' and no 'name'.",
    "evidence": "phase_events.py:3 'if result.triggered: print(f\"Event \\'{result.name}\\' triggered at t={result.t}\")'; phase_events.py:86 '# Which event in the list triggered (-1 if none)'; phase_events.py:87-92 (t, y, end flag, reason_code, reason_message, exit code).",
    "expected": "The documented example should exercise attributes that exist, or the field list should include triggered/name.",
    "failure_scenario": "A user or a future caller copies the documented idiom and gets AttributeError, or worse, a truthiness test against a field that is 0/-1-valued and silently never fires.",
    "repro": "Import EventResult and check for 'triggered' and 'name' attributes; run the exact docstring snippet.",
    "confidence": "high"
  },
  {
    "id": "S11-B-08",
    "file": "trinity/phase_general/phase_events.py",
    "line": 3,
    "class": "citation",
    "severity": "S4",
    "claim": "The 'this event ends the run' concept is given three different names across the prose: 'EndSimulationDirectly=True' (module docstring), 'event.is_simulation_ending' (min_radius factory), and an unnamed EventResult field described as 'True if simulation should end'.",
    "evidence": "phase_events.py:3 '1. **Simulation-Ending Events** (EndSimulationDirectly=True)'; phase_events.py:100 'Has additional attributes: event.name, event.is_simulation_ending, event.reason_code, event.reason_message'; phase_events.py:89 '# True if simulation should end'.",
    "expected": "One name. 'EndSimulationDirectly' looks like a stale reference to an identifier from an earlier design.",
    "failure_scenario": "A maintainer greps for EndSimulationDirectly to find the termination gate and finds nothing, or sets the wrong attribute on a hand-rolled event so it silently never terminates.",
    "repro": "grep for 'EndSimulationDirectly' across trinity/; check the actual attribute name set by the factories and the EventResult field name.",
    "confidence": "high"
  },
  {
    "id": "S11-B-09",
    "file": "trinity/phase_general/phase_events.py",
    "line": 92,
    "class": "state",
    "severity": "S3",
    "claim": "EventResult is documented to carry an 'Exit code for SimulationEndCode', but the documented event attributes are only name, is_simulation_ending, reason_code and reason_message — no exit code. The prose never says where the exit code comes from.",
    "evidence": "phase_events.py:92 '# Exit code for SimulationEndCode'; phase_events.py:100 'Has additional attributes: event.name, event.is_simulation_ending, event.reason_code, event.reason_message'.",
    "expected": "Either events carry an exit code attribute, or the derivation (reason_code -> exit code mapping) is documented and total.",
    "failure_scenario": "An event whose reason_code has no mapping yields a default/placeholder exit code, so metadata.json records a stopping fate that does not match the event that actually fired.",
    "repro": "Check how EventResult's exit-code field is populated in check_event_termination and whether every factory-produced reason_code has a mapping.",
    "confidence": "medium"
  },
  {
    "id": "S11-B-10",
    "file": "trinity/phase_general/phase_events.py",
    "line": 590,
    "class": "state",
    "severity": "S2",
    "claim": "apply_event_result documents state_keys defaulting to ['R2', 'v2'] (two keys), while the integrated state vector is documented as y0 = [r0, v0, E0, T0] (four elements). On the default path, Eb and T0 at the event are not written back to params.",
    "evidence": "phase_events.py:590 'state_keys : list of str Keys for state variables in order matching y. Default [\\'R2\\', \\'v2\\'].' and '# Update state variables'; main.py:225 '# y0 = [r0, v0, E0, T0]'; phase_events.py:254 'y_index : int Index of Eb in state vector y. Default 2 for [R2, v2, Eb, ...]'.",
    "expected": "Either every caller passes the full 4-key list, or the default covers the full state, or the prose documents that Eb/T0 are intentionally not updated at event time.",
    "failure_scenario": "A run terminated by an event records the final R2 and v2 at the event time but stale Eb and T0 from the last accepted solver step, corrupting the final-state block in the end report and any downstream energy accounting.",
    "repro": "Find every apply_event_result call site and check the state_keys passed; then, for an event-terminated run, compare params['Eb']/params['T0'] against the values implied by result.y.",
    "confidence": "medium"
  },
  {
    "id": "S11-B-11",
    "file": "trinity/phase_general/phase_events.py",
    "line": 86,
    "class": "silent-failure",
    "severity": "S2",
    "claim": "EventResult records a single triggered event ('Which event in the list triggered (-1 if none)'), but the implicit phase is documented to register a non-terminal monitoring event (velocity_sign) alongside three terminal ones. The prose never states precedence when a non-terminal event and a terminal event both fire in the same solve_ivp call.",
    "evidence": "phase_events.py:86 '# Which event in the list triggered (-1 if none)'; phase_events.py:391 '# Check each event'; phase_events.py:459 'Events: - velocity_sign: v2 crosses zero (monitoring, non-terminal) - min_radius ... - max_radius ... - velocity_runaway ...'; phase_events.py:310 '# Non-terminal by default - just records the crossing'.",
    "expected": "check_event_termination should be documented to select the terminal event (or the earliest-in-time terminal event) and to report monitoring crossings separately, so a monitoring crossing can never be reported as the termination reason and can never mask one.",
    "failure_scenario": "velocity_sign is listed first in the implicit phase's event list; if check_event_termination scans in order and returns the first event with a non-empty t_events, a benign zero-velocity crossing is reported as the run's outcome while the actual terminal event (e.g. min_radius) is discarded — or the crossing is recorded with is_simulation_ending False and the run is treated as having ended for no reason.",
    "repro": "Force an implicit-phase run where v2 crosses zero and R2 subsequently falls below the min_radius threshold in the same segment; inspect EventResult.index/reason_code and the recorded termination_reason.",
    "confidence": "medium"
  },
  {
    "id": "S11-B-12",
    "file": "trinity/phase_general/phase_events.py",
    "line": 320,
    "class": "divergence",
    "severity": "S3",
    "claim": "cooling_balance is listed in the module's Phase-Ending Events taxonomy in a module whose docstring says 'These events are passed to scipy.integrate.solve_ivp', while its own NOTE says it 'requires segment-level checking since Lgain/Lloss are computed during segment setup, not available to solve_ivp'. The only documented 1b->1c transition is therefore not enforced by the integrator.",
    "evidence": "phase_events.py:3 'These events are passed to scipy.integrate.solve_ivp to enable safe termination during integration.' and '- cooling_balance: L_cool ~ L_gain (implicit -> transition)'; phase_events.py:320 'NOTE: This event requires segment-level checking since Lgain/Lloss are computed during segment setup, not available to solve_ivp.'; phase_events.py:459 'Also returns cooling_balance factory for segment-level checking.'",
    "expected": "The taxonomy should mark cooling_balance as segment-level, and the implicit phase's segment length should bound the overshoot past the true cooling-balance point.",
    "failure_scenario": "The implicit->transition switch happens up to one full segment after (Lgain-Lloss)/Lgain first drops below threshold, making the 1b->1c boundary segment-length-dependent rather than physics-dependent — an integration-step-size sensitivity in a documented phase boundary.",
    "repro": "Run an implicit-phase config at two segment lengths and compare the t and R2 at the 1b->1c boundary; a shift with segment length confirms the overshoot.",
    "confidence": "high"
  },
  {
    "id": "S11-B-13",
    "file": "trinity/phase_general/phase_events.py",
    "line": 344,
    "class": "silent-failure",
    "severity": "S3",
    "claim": "The cooling-balance criterion divides by Lgain and the guard is documented as 'No event if no gain'. With Lgain == 0 the only documented implicit->transition transition can never fire, and no fallback path out of the implicit phase is documented.",
    "evidence": "phase_events.py:320 'Event triggers when (Lgain - Lloss) / Lgain < threshold'; phase_events.py:344 '# No event if no gain'; phase_events.py:459 (implicit phase's other events are all simulation-ending except velocity_sign, which is non-terminal).",
    "expected": "A documented fallback for Lgain == 0 (or a statement that Lgain > 0 is invariant in the implicit phase).",
    "failure_scenario": "In a regime where mechanical luminosity gain drops to zero, the run cannot leave the implicit phase via the documented route and can only end via min_radius / max_radius / velocity_runaway — i.e. the momentum phase is never reached and the fate is reported as a limit trip rather than the physical energy->momentum transition.",
    "repro": "Construct or find a late-time config where Lgain -> 0 during phase 1b and check whether the run ever reaches phase 1c/2, and what termination_reason it reports.",
    "confidence": "medium"
  },
  {
    "id": "S11-B-14",
    "file": "trinity/main.py",
    "line": 329,
    "class": "state",
    "severity": "S3",
    "claim": "At the transition->momentum boundary, Eb is documented to be inherited 'near ENERGY_FLOOR = 1e3' and then set to exactly 0 by the momentum phase runner at initialization — a documented discontinuous drop of ~1e3 Msun*pc^2/Myr^2 in the state across a phase boundary.",
    "evidence": "main.py:329 'Eb is inherited from the transition phase (near ENERGY_FLOOR = 1e3). The momentum phase runner sets Eb = 0 at its initialization. Do NOT set Eb=1 here — it was dead code that caused confusion.'; main.py:335 '# --- Transition -> Momentum boundary diagnostic ---'.",
    "expected": "Either the discontinuity is documented as physically negligible with a stated comparison scale, or the inherited Eb is carried into the momentum phase.",
    "failure_scenario": "Any output series or downstream analysis that integrates or differences Eb across the 1c->2 boundary sees a step to zero; an energy-budget audit reports a spurious ~1e3-unit loss at the phase change.",
    "repro": "Plot Eb vs t across the 1c->2 boundary from dictionary.jsonl for param/simple_cluster.param; check for the step and compare 1e3 against Eb at the start of phase 1a.",
    "confidence": "high"
  },
  {
    "id": "S11-B-15",
    "file": "trinity/main.py",
    "line": 314,
    "class": "state",
    "severity": "S3",
    "claim": "Cooling-related parameters are documented to be reset only after phase 1c ('Since cooling is not needed anymore after this phase, we reset values. COOLING_PHASE_KEYS contains all cooling-related parameters that can be cleared.'). Runs that end in phase 1a or 1b never reach the reset.",
    "evidence": "main.py:314-315 as quoted; main.py:260 documents an early exit after phase 1a ('we explicitly do NOT want phases 1b/1c/2 to advance past it'); phase_events.py:424/459 show simulation-ending events in phases 1a and 1b.",
    "expected": "Either a documented statement that the output schema legitimately differs by end path, or a reset that runs on every termination path.",
    "failure_scenario": "Two runs of the same sweep end in different phases and produce different key sets in the output/metadata; a reader or comparison tool that assumes a fixed schema (or treats a stale cooling key as current) mis-parses or silently uses last-computed cooling values as final.",
    "repro": "Run one config that terminates in phase 1a (stop_at_rCloud_nSnap == 0) and one that reaches phase 2; diff the key sets of the final params/metadata for the COOLING_PHASE_KEYS members.",
    "confidence": "medium"
  },
  {
    "id": "S11-B-16",
    "file": "trinity/main.py",
    "line": 149,
    "class": "regime",
    "severity": "S2",
    "claim": "A TODO states that a non-zero star-formation time is not handled: 'if tSF != 0.: we would actually need to shift the feedback parameters by tSF'. No guard, warning or error for tSF != 0 is documented anywhere in the slice.",
    "evidence": "main.py:149-151 '# TODO: / # if tSF != 0.: we would actually need to shift the feedback parameters by tSF / # update'.",
    "expected": "Either the shift is applied, or tSF != 0 is rejected/warned at parameter-validation time.",
    "failure_scenario": "A user sets tSF != 0 in a .param file; the SPS feedback parameters are evaluated on an unshifted time axis, so the entire feedback history — and therefore every phase boundary and the final fate — is wrong, silently and with no warning.",
    "repro": "grep tSF in trinity/_input schema/defaults and in the SPS loader call path; run a config with tSF != 0 and check whether any warning is emitted or any time shift applied.",
    "confidence": "high"
  },
  {
    "id": "S11-B-17",
    "file": "trinity/main.py",
    "line": 217,
    "class": "deadcode",
    "severity": "S3",
    "claim": "run_expansion's docstring claims evolution proceeds 'until next recollapse or (if no re-collapse) until end of simulation', implying multiple starburst generations, while the corresponding block is an unimplemented TODO ('add loop so that this simulation starts over with old generation of parameter to simulate new starburst environment').",
    "evidence": "main.py:217-219 as quoted; main.py:208 '# ########### STEP 2: In case of recollapse, prepare next expansion ##########################'; main.py:209 '# TODO: add loop so that this simulation starts over ...'.",
    "expected": "The docstring should describe single-generation evolution, or the loop should exist.",
    "failure_scenario": "A reader assumes recollapse restarts the simulation with a new generation; results are interpreted as multi-generation when only the first expansion was ever integrated.",
    "repro": "Check whether any loop or re-entry into run_expansion exists after the recollapse block in start_expansion.",
    "confidence": "high"
  },
  {
    "id": "S11-B-18",
    "file": "trinity/main.py",
    "line": 142,
    "class": "regime",
    "severity": "S3",
    "claim": "The cluster-mass scaling factor is documented as possibly valid only for high-mass clusters: 'this might only be accurate for high mass clusters (~>1e5) in which the IMF is fully sampled'. No validity guard or warning is documented.",
    "evidence": "main.py:142-143 '# Scaling factor for cluster masses. Though this might only be accurate for / # high mass clusters (~>1e5) in which the IMF is fully sampled.'",
    "expected": "A documented warning (or at least a recorded flag in metadata) when the scaled cluster mass falls below the stated validity floor.",
    "failure_scenario": "A sweep over low cluster masses silently applies an SPS scaling outside its documented validity range; the feedback luminosities — and hence every phase boundary — are wrong for the low-mass end with no indication in the output.",
    "repro": "Find the scaling application and check for any mass-threshold warning; run a config with a cluster mass well below 1e5 and inspect the log for any validity notice.",
    "confidence": "medium"
  },
  {
    "id": "S11-B-19",
    "file": "trinity/main.py",
    "line": 217,
    "class": "other",
    "severity": "S3",
    "claim": "No time-based stopping condition is documented anywhere in the slice. The only reference is 'until end of simulation'; the momentum phase's documented event list contains only min_radius, max_radius and velocity_runaway, and no phase-ending or time-limit event.",
    "evidence": "main.py:217-219 'until next recollapse or (if no re-collapse) until end of simulation'; phase_events.py:547 'Events: - min_radius ... - max_radius ... - velocity_runaway ...'; phase_events.py:3 lists no time event in any category.",
    "expected": "The stopping time (tStop/tEnd), its unit, and its end code should be documented alongside the three event-based end paths, since it is presumably the most common way a run ends.",
    "failure_scenario": "A reader cannot tell whether a run that ended without an event code hit a time limit or exited some other way; a run whose stop time is misconfigured has no documented guard.",
    "repro": "grep for the stopping-time parameter in trinity/_input defaults and for where the momentum-phase loop's time bound is enforced; check which SimulationEndCode a time-limited run reports.",
    "confidence": "medium"
  },
  {
    "id": "S11-B-20",
    "file": "trinity/main.py",
    "line": 224,
    "class": "units",
    "severity": "S4",
    "claim": "No time quantity in the slice carries a stated unit. 't0 = start time for Weaver phase' and 'Time of event (NaN if not triggered)' give no unit, while neighbouring state variables are explicitly annotated (pc, pc/Myr, K, Msun*pc^2/Myr^2). Myr is only implied transitively.",
    "evidence": "main.py:224 '# t0 = start time for Weaver phase'; main.py:226-229 annotate '(pc)', '(pc/Myr)', '(K)' but line 228 'Eb = initial energy' has none; phase_events.py:87 '# Time of event (NaN if not triggered)'.",
    "expected": "Time annotated as Myr (or whatever the convention is) wherever the neighbouring state variables are annotated, per the project's stated units-are-a-recurring-bug-class convention.",
    "failure_scenario": "A future edit or a downstream reader mixes Myr and s (or yr) at a phase boundary; the project's own CLAUDE.md names units as a recurring bug class here.",
    "repro": "Check trinity/_functions/unit_conversions.py for the time convention and confirm t0 / EventResult.t use it.",
    "confidence": "high"
  },
  {
    "id": "S11-B-21",
    "file": "trinity/phase_general/phase_events.py",
    "line": 459,
    "class": "regime",
    "severity": "S3",
    "claim": "velocity_sign — the sole documented collapse-onset detector — is registered only in the implicit phase. The energy, transition and momentum phase event lists do not include it.",
    "evidence": "phase_events.py:459 'Events: - velocity_sign: v2 crosses zero (monitoring, non-terminal) ...'; phase_events.py:424, :505, :547 list no velocity_sign; phase_events.py:3 '- velocity_sign: v2 crosses zero (collapse onset detection)'.",
    "expected": "Either collapse onset is documented as physically impossible outside the implicit phase, or the monitor is registered in every phase where v2 can change sign.",
    "failure_scenario": "A shell that turns around during the transition or momentum phase has no collapse-onset crossing recorded; the run is later classified by end code alone (min_radius) with no recorded turnaround time, and 'Mark collapse if it's a collapse-related event' (line 626) never fires for that path.",
    "repro": "Find a config where v2 changes sign in phase 2 and check whether any collapse-onset time is recorded in the output.",
    "confidence": "medium"
  },
  {
    "id": "S11-B-22",
    "file": "trinity/phase_general/phase_events.py",
    "line": 310,
    "class": "citation",
    "severity": "S4",
    "claim": "'Non-terminal by default' implies a way to make velocity_sign terminal, but the documented factory signature exposes only y_index and name — no terminal override.",
    "evidence": "phase_events.py:310 '# Non-terminal by default - just records the crossing'; phase_events.py:288 'Parameters ---------- y_index : int ... name : str ...' and 'Event function for solve_ivp with terminal=False (monitoring only)'.",
    "expected": "Either drop 'by default', or document the override parameter.",
    "failure_scenario": "A caller assumes an override exists and sets .terminal on the returned function directly, changing termination behaviour in a way no docstring describes.",
    "repro": "Check make_velocity_sign_event's signature for any terminal parameter, and grep for any site that mutates a returned event's .terminal.",
    "confidence": "high"
  },
  {
    "id": "S11-B-23",
    "file": "trinity/main.py",
    "line": 260,
    "class": "state",
    "severity": "S3",
    "claim": "The stop_at_rCloud_nSnap == 0 early exit asserts a pre-existing snapshot: 'The energy-phase reconciliation snapshot already captured R2 = rCloud, and we explicitly do NOT want phases 1b/1c/2 to advance past it.' This claims (a) a snapshot at exactly R2 = rCloud exists before the check, and (b) no further phase advances. Only the == 0 case is documented; the meaning of non-zero values is stated nowhere.",
    "evidence": "main.py:260-262 as quoted; main.py:126 'rCloud is now known; warn the user if stop_r will starve stop_at_rCloud_nSnap of any chance to fire (or race with it).'; main.py:42 references stop_at_rCloud_nSnap without defining its non-zero semantics.",
    "expected": "The semantics of stop_at_rCloud_nSnap for every value (0 and non-zero) documented in one place, plus verification that the reconciliation snapshot is at R2 == rCloud (not the last pre-crossing step).",
    "failure_scenario": "If the reconciliation snapshot is at the last accepted step rather than exactly at rCloud, the run terminated 'at the cloud edge' actually reports R2 < rCloud; and a user setting a non-zero nSnap gets undocumented behaviour.",
    "repro": "Run with stop_at_rCloud_nSnap = 0 and check the final recorded R2 against rCloud to machine precision; then run with a non-zero value and record what changes.",
    "confidence": "medium"
  },
  {
    "id": "S11-B-24",
    "file": "trinity/phase_general/phase_events.py",
    "line": 626,
    "class": "other",
    "severity": "S3",
    "claim": "'Mark collapse if it's a collapse-related event' does not enumerate which events are collapse-related. Two candidates are documented: velocity_sign ('collapse onset detection', non-terminal) and min_radius ('shell collapse', terminal); velocity_runaway's 'collapse' direction mode is a third possible match.",
    "evidence": "phase_events.py:626 '# Mark collapse if it\\'s a collapse-related event'; phase_events.py:3 '- min_radius: R2 < coll_r (shell collapse)' and '- velocity_sign: v2 crosses zero (collapse onset detection)'; phase_events.py:169 'direction : str \"collapse\" for inward (v2 < -v_max)'.",
    "expected": "An explicit, enumerated set of collapse-marking events, since the collapse flag feeds the run's reported fate.",
    "failure_scenario": "The collapse flag is set by a name-substring or reason_code heuristic; a velocity_runaway event created with direction='collapse' marks a numerically-terminated run as a physical collapse, or conversely a genuine min_radius collapse fails the heuristic and is not marked.",
    "repro": "Read the predicate used at apply_event_result's collapse-marking branch and evaluate it against every factory's reason_code/name.",
    "confidence": "medium"
  },
  {
    "id": "S11-B-25",
    "file": "trinity/main.py",
    "line": 82,
    "class": "citation",
    "severity": "S4",
    "claim": "start_expansion documents its parameter as 'params : Object — An object describing TRINITY parameters', while all four phase-event builders and apply_event_result document 'params : dict'.",
    "evidence": "main.py:82-95 'params : Object An object describing TRINITY parameters.'; phase_events.py:424/459/505/547 'params : dict Parameter dictionary with ...'; phase_events.py:590 'params : dict Parameter dictionary to update.'",
    "expected": "One documented type for the parameter container across the orchestration path, or an explicit note that an attribute-style object is converted to a dict before reaching the event builders.",
    "failure_scenario": "A caller following the __init__.py quickstart passes what start_expansion documents as an Object into a helper that does params['rCloud'], raising TypeError, or an attribute-style access silently returns a default.",
    "repro": "Check what read_param.read_param returns and whether the same container reaches build_energy_phase_events unchanged.",
    "confidence": "medium"
  }
]
```
