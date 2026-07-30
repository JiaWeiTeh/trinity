# S11 — Orchestration & phase events: reconciled

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

Inputs: `S11_orchestration_lensA.md` (behaviour, comment-stripped source), `S11_orchestration_lensB.md`
(prose only), `S11_orchestration_lensC.md` (physics spec + signatures only). No source was read by the
reconciler. Files in scope: `trinity/main.py`, `trinity/phase_general/phase_events.py`,
`trinity/phase_general/__init__.py`, `trinity/__init__.py`.

---

## 1. Verdict

The *event primitives* in this slice are in better shape than any single lens suggests: every one of
the seven `solve_ivp` residuals has the residual form, sign convention and `direction` that lens C's
physics derivation demands, and the two phase-boundary events (`cloud_boundary`, `energy_floor`) are
correctly marked non-simulation-ending — so lens C's worst-case trap ("every run terminates at the
energy→transition boundary") is *not* what this code does. The damage is concentrated one layer up,
in the **classification and reporting layer**: `check_event_termination` decides which event ended a
phase by list index rather than by root time or by terminality, and `apply_event_result` decides
whether a run collapsed by substring-matching `'radius'` against the reason code. The single most
important finding is that this substring test (`phase_events.py:627`) returns **True for
`large_radius_event`** — a run terminated by *expanding* past `stop_r` is recorded as a collapsed
shell — and **False for `velocity_runaway_event`**, so the most violent infall the code can detect
(v2 < −500 pc/Myr) is not recorded as a collapse. That is pure deterministic string logic, verified
against lens B's admission that the "collapse-related" event set is nowhere enumerated, and against
lens C's requirement that the fate census be filterable without parsing free text.

Second in importance, and structurally the same disease: the slice has **no channel that
distinguishes a solver failure from a physical fate**. `sol.status`/`sol.success` are never read, the
not-triggered `EventResult` is byte-identical between "ran cleanly to `t_end`" and "integrator
aborted", `start_expansion` returns 0 unconditionally while discarding the `99` failure code, and
three broad `except Exception` blocks — including the one around `params.flush()` — downgrade output
loss to a warning. Lens B confirms the prose is *silent* on this, so it is an outright gap rather
than doc-drift.

**One caveat governs the severity of roughly a third of these findings and must not be lost**: no
lens read the four phase runners (`run_energy`, `run_phase_energy`, `run_phase_transition`,
`run_phase_momentum`). Every claim about what `check_event_termination`'s verdict *does*, what `t`/`y`
and `state_keys` reach `apply_event_result`, and whether the cooling factory is spliced into
`solve_ivp` or evaluated per segment, is an assumption all three lenses share. Those are marked OPEN
with the exact grep that settles each.

---

## 2. Findings table

| id | sev | type | file:line | claim | status | conf |
|---|---|---|---|---|---|---|
| S11-R-01 | S1 | A≠C | `phase_events.py:392-405`, `:488` | Phase-end attribution returns the **lowest-index** event with any root, not the earliest or the terminal one; the only `terminal=False` event sits at index 0 of the implicit phase | CONFIRMED (mechanism) / OPEN (blast radius) | high mech / med impact |
| S11-R-02 | S1 | A≠C | `phase_events.py:627` | `isCollapse` set by substring: `'radius' in 'large_radius_event'` → expansion recorded as collapse; `'velocity_runaway_event'` matches nothing → runaway infall not recorded as collapse | CONFIRMED | high |
| S11-R-03 | S1 | A≠C | `main.py:211`, `:196`, `:204`, `:358`; `phase_events.py:379-416` | No solver-failure channel anywhere in the slice: `sol.status` never read, identical not-triggered result for abort and clean finish, `return 0` unconditional, `flush()` failure swallowed | CONFIRMED in-slice / OPEN (runners, exit-code propagation) | high / med |
| S11-R-04 | S2 | A≠B | `main.py:60-67`; `phase_events.py:447-451`, `:157` | `stop_r ≤ rCloud` makes `stop_r` permanently unfireable — the exact opposite of the warning the code emits | CONFIRMED (mechanism) | med-high |
| S11-R-05 | S2 | A≠B / B≠C | `phase_events.py:3` vs `:445,485,529,568` | Docstring taxonomy: "min_radius: R2 < coll_r"; code fires at `max(1.5·coll_r, 0.01)` — a 50 % margin on the published collapse radius, under the same `SHELL_COLLAPSED` code as the numerical guard | CONFIRMED | high |
| S11-R-06 | S2 | A≠C | `phase_events.py:341-346` | `cooling_balance` residual reads neither `t` nor `y` — constant per call, no bracketable root; it carries `terminal`/`direction` attributes that only mean anything to `solve_ivp` | OPEN (which path the 1b runner uses) | high on constancy / open on use |
| S11-R-07 | S2 | A≠C | `phase_events.py:343-346` | The guard is `if Lgain <= 0` only; for small **positive** `Lgain` the residual → −∞, so the criterion can trip on the SPS wind/SN gap rather than on cooling | OPEN (reachability of `Lgain → 0⁺` in 1b) | med |
| S11-R-08 | S2 | A≠C | `main.py:264-265` | `R2 >= rCloud` re-tested *after* a root-solved `cloud_boundary` event; the root state straddles `rCloud` at ~1e-12 relative, so `stop_at_rCloud_nSnap=0` termination is a coin flip | CONFIRMED (mechanism) | med-high |
| S11-R-09 | S2 | A≠C | `phase_events.py:589`, `:612`, `:615-617` | `apply_event_result` writes caller-supplied `t`/`y` (never `result.t`/`result.y`) and only `len(state_keys)` components — default 2, so `Eb` (`y[2]`) is droppable at the 1c hand-off | OPEN (call sites) | med |
| S11-R-10 | S2 | A≠C | `main.py:283`, `:303`, `:343` | All three phase gates are `== False` on a key `run_expansion` never initialises; an unset/`None` `EndSimulationDirectly` silently skips phases 1b/1c/2, and a reused `params` inherits a prior run's `True` | OPEN (schema default + type) | med |
| S11-R-11 | S3 | A≠B | `main.py:278`, `:301`, `:327` | `current_phase` advanced and the "Entering …" banner printed **before** the gate, contradicting `main.py:260`'s "we explicitly do NOT want phases 1b/1c/2 to advance past it" | CONFIRMED | high |
| S11-R-12 | S3 | B≠C | `phase_events.py:3`, `:100`, `:169` | Docs give `min_radius` and `velocity_runaway` two meanings each (physical fate *and* solver protection) under one end-code channel; spec requires disjoint outcome vocabularies | CONFIRMED | high |
| S11-R-13 | S3 | A≠B | `main.py:217` vs `:209`, `:366` | `run_expansion` docstring claims evolution "until next recollapse"; the recollapse loop is a TODO and `expansion_next` is a 7-argument no-op stub | CONFIRMED | high |
| S11-R-14 | S3 | A≠B | `phase_events.py:74`, `:195-205` | `MAX_VELOCITY_EXPANSION` documented as half of an asymmetric pair; never referenced, and the `"expansion"`/`else` branches are unreachable from all four builders | CONFIRMED | high |
| S11-R-15 | S3 | A≠B | `main.py:317`, `:336-341` | Boundary `pRam` computed unconditionally, logged as "(momentum phase will use this)", never stored or passed — and computed *after* `reset_keys(COOLING_PHASE_KEYS)` | CONFIRMED (main.py) / OPEN (key membership) | high / med |
| S11-R-16 | S3 | A≠C | `phase_events.py:239-246` | `cloud_boundary` is `direction=+1` only; the RHS density-branch discontinuity is not restarted on an inward re-crossing | CONFIRMED (code) / OPEN (reachability) | med |
| S11-R-17 | S3 | scope-creep | `phase_events.py:504` | Absolute `energy_floor = 1e3` code units as the 1c→2 criterion; spec argues the criterion must be relative across the shipped ~4.7-dex `mCloud` grid | CONFIRMED as design divergence | med |
| S11-R-18 | S4 | A≠C | `phase_events.py:401` | `getattr(event, 'is_simulation_ending', True)` — fail-dangerous default; `end_code` two lines later defaults to the safe `None` | CONFIRMED | high |
| S11-R-19 | S4 | A=B=C | `phase_events.py:589` | Mutable default argument `state_keys: List[str] = ['R2','v2']`, one list object for the process lifetime | CONFIRMED | high |
| S11-R-20 | S4 | A≠B | `phase_events.py:71` | `MIN_RADIUS_SAFETY = 0.01` pc is dead whenever `1.5·coll_r > 0.01`, i.e. for any `coll_r > 0.0067` pc | OPEN (`coll_r` default) | med |
| S11-R-21 | S4 | A-only | `main.py:144` | `f_mass = params['mCluster'] / params['sps_refmass']` divides `DescribedItem` objects, not `.value`s | OPEN (dunder support) | low |
| S11-R-22 | S4 | A-only | `phase_events.py:394`, `:501` | `events[i]` is re-indexed positionally against `sol.t_events` with no length assertion, while the implicit builder alone returns a `(events, factory)` tuple | OPEN (caller) | low |

---

## 3. Per-finding detail

### S11-R-01 — event attribution by list index, with a non-terminal event at index 0 — S1

**A** (`S11-A-01`, `S11-A-02`, `S11-A-03`): `check_event_termination` is
`for i, (t_ev, y_ev) in enumerate(zip(sol.t_events, sol.y_events)): if len(t_ev) > 0: … return`. The
token `terminal` appears in the file only as an assignment (`:124, :156, :208, :243, :278, :310, :348`)
and is never read. `make_velocity_sign_event` is the only `terminal=False` event and
`build_implicit_phase_events` puts it at index 0.

**B** (§4.4, `S11-B-11`): the docstring documents `EventResult` as recording *one* index
("Which event in the list triggered (-1 if none)") and **"Precedence when more than one event fires is
not documented."** B predicted this exact scenario from prose alone.

**C** (`S11-C-20`, `S11-C-04`, trap I): "When more than one event roots within a step, the orchestrator
must select the earliest root, not the first entry in the event list… the recorded phase sequence
becomes an artefact of list construction order." And separately: `EventResult` must carry
`ends_segment` and `ends_run` as two independent fields.

**Diagnosis — A≠C (physics/logic defect), with B corroborating that the docs never claimed otherwise.**
Two independent bugs compound. (a) *Index-order, not time-order*: in the transition phase the list is
`[energy_floor, min_radius, velocity_runaway, max_radius]`; if `min_radius` roots at t=1 and
`energy_floor` at t=5, `energy_floor` is reported with the wrong time and `is_simulation_ending=False`
instead of `True`. (b) *Non-terminal shadowing*: scipy populates `t_events[i]` for **every** detected
crossing, terminal or not. In phase 1b, `velocity_sign` (index 0, `terminal=False`, `direction=-1`)
records the ordinary expansion→collapse turnover; the classifier returns
`triggered=True, name='velocity_sign', is_simulation_ending=False, t=<turnover>` and never examines
`min_radius` (1), `velocity_runaway` (2) or `max_radius` (3), even though one of those is what
actually stopped the integration.

**Failure scenario, concrete.** A high-`mCloud`/low-`sfe` config whose shell turns around inside the
implicit phase and then collapses to `min_r` in the same `solve_ivp` call: scipy stops at
`min_radius` (terminal), but `t_events[0]` already holds the earlier `v2 = 0` root.
`check_event_termination` reports `velocity_sign`, `is_simulation_ending=False`. `apply_event_result`
then takes the `if result.is_simulation_ending:` branch **not at all** —
`EndSimulationDirectly` is never set, `SimulationEndCode` is never `SHELL_COLLAPSED`, and phases 1c
and 2 run from the post-collapse state. This is precisely lens C's §3.3 corruption: the recollapse
census under-counts, and the miss is correlated with exactly the regime the grid is studying.

**Status.** CONFIRMED for the mechanism (deterministic control flow read directly by A, corroborated
by B's independent observation that precedence is undocumented). OPEN for how bad it gets, because no
lens read the 1b runner: it may re-check `result.is_simulation_ending` itself, or loop per segment so
that only one event can root per call.

**Lookup that settles the blast radius**:
`grep -rn "check_event_termination\|apply_event_result" trinity/phase_general/` then read the loop
around the phase-1b call — specifically whether it `break`s on `result.triggered` and whether it
re-inspects `sol.t_events` itself.

**Repro if confirmed**: construct `sol` with `t_events = [array([1.0]), array([5.0])]`, pass
`events=[non_terminal, terminal]`, assert `check_event_termination(...).name` is the terminal one.

---

### S11-R-02 — `isCollapse` decided by substring match — S1

**A** (`S11-A-05`, `S11-A-06`): `if 'radius' in result.reason_code.lower() or 'collapse' in
result.reason_code.lower(): params['isCollapse'].value = True` at `:627`, evaluated against the four
ending reason codes: `small_radius_event` → True (correct), `large_radius_event` → **True (wrong —
the run terminated by expanding through `stop_r`)**, `velocity_runaway_event` → **False even for the
`direction="collapse"` variant**, which is by construction `v2 < −500 pc/Myr`.

**B** (`S11-B-24`, §7.5): the only prose is `# Mark collapse if it's a collapse-related event`
(`:626`); **"the set of collapse-related events is never enumerated"**. B independently predicted:
"The collapse flag is set by a name-substring or reason_code heuristic; a velocity_runaway event
created with direction='collapse' marks a numerically-terminated run as a physical collapse, or
conversely a genuine min_radius collapse fails the heuristic."

**C** (`S11-C-11`, `S11-C-14`, `S11-C-21`): the outcome must carry a **kind**
(`physical_fate` / `numerical_cutoff` / `solver_failure`), `stop_r` is explicitly a numerical cutoff
"never escape", and re-collapse is a physical fate that feeds `expansion_next`.

**Diagnosis — A≠C, confirmed by triangulation; B shows the docs are too vague to have caught it.**
This is not an inference: `'radius' in 'large_radius_event'` is `True` in Python by inspection. Two
inverted classifications, in opposite directions, from one line.

**Failure scenario.** A sweep cell that expands to `stop_r` writes `isCollapse=True`. A sweep cell
that implodes at 489 km/s writes nothing. Any downstream census of the dispersal/recollapse boundary
built on `isCollapse` is wrong on both sides.

**Severity coupling.** The `large_radius_event` half is only reachable if `max_radius` can fire, which
S11-R-04 shows requires `stop_r > rCloud`. The `velocity_runaway` half is reachable whenever a
collapse runs away, independent of `stop_r`. Rate S1 if lookup Q5 shows `stop_r > rCloud` in shipped
configs; S2 (latent) otherwise, but the runaway half stays S1/S2 regardless.

**Repro**: set `stop_r` above `rCloud` in a `.param`, run, inspect `params['isCollapse']` in the
output; then force `v2 < −500 pc/Myr` and inspect it again.

---

### S11-R-03 — no solver-failure channel — S1

**A** (`S11-A-08`, `S11-A-09`, `S11-A-21`): `check_event_termination` reads only `sol.t_events` /
`sol.y_events`; the "no events attribute" result (`:379-389`) and the "ran to `t_end`" result
(`:407-416`) are literally the same construction, so a `status == -1` abort with empty `t_events`
produces a byte-identical `EventResult` to a clean finish. `start_expansion` computes
`exit_code = write_simulation_end(params)`, sets it to `99` on exception, logs it at DEBUG, and then
`return 0` unconditionally at `:211`. Three broad `except Exception` blocks (`:196`, `:204`, `:358`)
downgrade end-report, termination-report and **`params.flush()`** failures to `logger.warning`, after
which `:361` logs "All expansion phases complete".

**B** (§3.4): **"No prose in this slice states how a solver failure is distinguished from a physical
outcome."** There is no documented `SimulationEndCode` meaning "solver failed", no documented handling
of `sol.status < 0`, and no documented distinction in the end report between "the bubble collapsed"
and "the integrator was about to blow up".

**C** (`S11-C-31`, `S11-C-21`, §3.3, §5): SPEC-105 already mandates a `termination` block with
`{exit_code, outcome, detail}`, a `final_state` and a `termination_debug`; SPEC-100 separates physical
fates from numerical cutoffs. C's §3.3 argument is the load-bearing one: *"if solver aborts in that
corner are labelled `recollapse`, the paper's conclusion 'clouds denser than X re-collapse' is a
restatement of 'the integrator fails above density X'. The correlation is perfect and undetectable
from the outputs alone."*

**Diagnosis — A≠C, with B confirming an outright documentation gap rather than drift.** All three
agree; none of them is inferring.

**Failure scenario.** A stiff `f1edge_hidens` run aborts at `t = 2 Myr` of a 15 Myr budget with
`status=-1`. `check_event_termination` returns not-triggered; `apply_event_result` returns
immediately and writes nothing; `run_expansion` proceeds to the next phase from the last successful
state; `start_expansion` returns 0. The run is filed as a complete 100 %-energy-phase evolution.
Independently: a disk-full during `params.flush()` loses the run's entire output while the process
reports success.

**Status.** CONFIRMED in-slice for the `return 0` and the three swallowing handlers (A read them
directly). OPEN for the claim "no code anywhere checks solver status" — the phase runners are outside
every lens's scope, and one of them may inspect `sol.success`.

**Lookup**: `grep -rn "\.status\b\|\.success\b\|sol\.message" trinity/phase_general/ trinity/_output/`
and `grep -n "start_expansion" run.py` (does `run.py` propagate a return value to `sys.exit`?).

---

### S11-R-04 — `stop_r ≤ rCloud` disables `stop_r`, opposite to the emitted warning — S2

**A** (`S11-A-07`, `S11-A-20`): `build_energy_phase_events` (`:447-451`) builds exactly
`[cloud_boundary, min_radius, velocity_runaway]` and never reads `stop_r` — the only builder that
omits it. Phase 1a's `cloud_boundary` is `terminal=True`, so phase 1a stops at `R2 ≈ rCloud`. Phases
1b/1c/2 then build `make_max_radius_event(stop_r)` with `g = R2 − stop_r`, `direction=+1`; at entry
`R2 ≈ rCloud ≥ stop_r`, so `g ≥ 0` already and only a negative→positive crossing is caught.
Meanwhile `main.py:60-67` warns that *"stop_r will terminate the run before stop_at_rCloud_nSnap can
fire"*.

**B** (`S11-B-03`, `S11-B-04`): derived the *same conclusion from prose alone* — the energy-phase
event list omits `max_radius`, and `max_radius` is documented "Only trigger when R2 crosses max_r from
below". B then flags that `_check_stop_r_rCloud_interaction`'s docstring asserts the opposite
causality (small `stop_r` "silently disable[s] their `stop_at_rCloud_nSnap` termination") and that
`main.py:35` says the two "race".

**C** (`S11-C-29`, trap J): the check must be a fail-fast startup validation against `rCloud` *plus
the radial extent the requested snapshots span*, not a fixed multiplicative factor, and using the
`rCloud` the solver will actually use (SPEC-005: 11 % difference at ε = 0.3 depending on
normalisation).

**Diagnosis — A≠B (doc-drift on the warning text) layered on a real A≠C behaviour gap.** A and B
reached the mechanism independently from disjoint inputs (stripped code vs prose only), which is
genuine corroboration rather than a shared assumption. The warning is not merely vague — it names the
wrong hazard, which is exactly the S3 that licenses a future wrong "fix".

**Premise to check.** The chain requires phase 1a to end at `cloud_boundary`. If phase 1a instead runs
to `t_end` below `rCloud`, then `R2 < stop_r` at 1b entry and `max_radius` *can* fire. So the defect
is "whenever the shell reaches the cloud edge", i.e. the common case, not universal.

**Repro**: set `stop_r < rCloud` in a `.param`, run, observe the warning at `main.py:60` and that no
`large_radius_event` is ever recorded.

---

### S11-R-05 — documented collapse radius is `coll_r`; actual is `1.5 · coll_r` — S2

**A**: `min_r = max(coll_r * MIN_RADIUS_FACTOR, MIN_RADIUS_SAFETY) = max(1.5·coll_r, 0.01)` pc,
computed identically in all four builders (`:445, :485, :529, :568`); the event's `end_code` is
`SHELL_COLLAPSED`.

**B** (`S11-B-01`, `S11-B-02`): the module taxonomy says "min_radius: R2 < **coll_r** (shell
collapse)"; the factory docstring says "Typically coll_r * factor or MIN_RADIUS_SAFETY"; the four
builders say only "R2 < safety threshold"; and the same factory also says the event "prevents LSODA
from crashing when R2 approaches zero" — physical fate and numerical guard sharing one end code.

**C** (`S11-C-11`): `coll_r` is the physical re-collapse radius that feeds the next SF episode; a
numerical floor protecting `1/R2²` and `1/(R2³−R1³)` is a different thing with a different outcome
kind. "Same residual shape, same direction, completely different meaning."

**Diagnosis — A≠B on the number (doc-drift, S3-in-kind but with a results consequence), B≠C on the
taxonomy (one end code for two meanings, S2).** A run reported as `SHELL_COLLAPSED` stopped at 1.5×
the documented collapse radius; a downstream reader taking the taxonomy at face value will believe the
final `R2 < coll_r`.

**Note on a lens-C hypothesis I decline to promote.** `S11-C-13` predicts "every compact-cluster run
terminates at t ≈ 0 reporting 'collapsed'" because `r0 < coll_r`. A's direction reading refutes this:
`direction = -1` means the initial `−→+` crossing (as the shell expands out through `min_r`) is
**ignored**, and only the later `+→−` infall crossing fires. The event is therefore correctly latched
by construction for any expand-then-collapse trajectory. C's structural point (absolute constant vs
state-dependent scale) stands; its predicted symptom does not.

**Repro**: run a collapsing config and compare final `R2` against `coll_r` and against `1.5·coll_r`.

---

### S11-R-06 — `cooling_balance` is a constant function wearing `solve_ivp` clothing — S2, OPEN

**A** (`S11-A-04`): the returned `event(t, y)` at `:342-346` reads neither `t` nor `y`; it closes over
scalar `Lgain`/`Lloss` bound at factory-call time. Its value is a constant for the whole of any one
`solve_ivp` call, so no root can exist regardless of `terminal=True` (`:348`) / `direction=-1`
(`:349`). A explicitly caveats: "It can only 'work' if the caller rebuilds it per step and evaluates
its **sign** by hand outside the integrator."

**B** (`S11-B-12`, §4.6): the code *knows this*. `:320` — **"NOTE: This event requires segment-level
checking since Lgain/Lloss are computed during segment setup, not available to solve_ivp"** — and
`build_implicit_phase_events` returns the factory **separately from the event list**. But the module
docstring simultaneously says "These events are passed to scipy.integrate.solve_ivp" and lists
`cooling_balance` in the phase-ending taxonomy.

**C** (`S11-C-01`, trap A): flags `def factory(Lgain: float, Lloss: float) -> event(t, y)` from the
*signature alone* as "precisely the shape that produces this degeneracy" and rates it S1.

**Diagnosis — the disagreement type here is decisive and I will not collapse it.** A and C see a dead
event; B shows an explicit architectural note saying it is deliberately *not* a `solve_ivp` event, and
a builder API that keeps it out of the event list. If B's note describes reality, the design is sound
and the finding degrades to:
 - **S3**: the factory sets `terminal` and `direction`, attributes that have meaning only to
   `solve_ivp`, on an object documented as never reaching `solve_ivp` — a trap for the next reader.
 - **S3**: the module docstring's blanket "these events are passed to solve_ivp" plus the taxonomy
   listing is drift.
 - **S3 (C's residual concern)**: segment-level evaluation means the 1b→1c boundary is quantised to
   the segment length, so the transition time is step-size dependent, not physics dependent
   (`S11-B-12`'s failure scenario, which B rated high confidence).

If instead the runner splices this into `solve_ivp`'s event list, C is right and it is S1: the
documented 1b→1c physics transition never fires and the phase ends on whatever guard trips first.

**Status: OPEN.** This is the finding where two lenses agreeing (A and C) is *not* proof — they share
the assumption that the factory reaches the integrator, and B has direct evidence against it.

**Lookup that settles it**: find the 1b runner
(`grep -rln "run_phase_energy" trinity/phase_general/`), then in that file
`grep -n "build_implicit_phase_events\|cooling\|solve_ivp"` — check whether the second tuple element
is appended to the `events=` argument of `solve_ivp`, or called per segment with its return value
sign-tested in Python.

---

### S11-R-07 — the `Lgain` guard covers zero but not near-zero — S2, OPEN

**A**: `if Lgain <= 0: return 1.0` else `return (Lgain - Lloss)/Lgain - threshold`.

**B** (`S11-B-13`): documents the guard as "No event if no gain" and flags that with `Lgain == 0` the
only documented 1b→1c route can never fire, with no documented fallback.

**C** (`S11-C-03`): "L_gain passes through zero or near-zero in the wind/SN gap of a low-mass cluster's
SPS track. `g = (L_gain − L_loss)/L_gain` then diverges to −∞ and, worse, changes sign spuriously as
L_gain crosses zero. The residual must be formed as `L_gain(1−θ) − L_loss` (multiply through — same
root, same sign, no pole)."

**Diagnosis — A≠C.** The code implements *half* of C's prescription: the pole is excluded exactly at
`Lgain ≤ 0` (and returns a positive "not yet" value, which is the right sense), but for
`0 < Lgain ≪ Lloss` the residual is a large negative number, so the criterion reads "transition now"
on the strength of a near-vanishing denominator rather than on cooling physics. C's algebraic fix
(`Lgain*(1−θ) − Lloss`) is the same root with no pole and would close this.

**Status: OPEN** — reachability depends on whether `Lgain` in phase 1b can approach zero from above
in shipped configs (the wind/SN gap is a low-mass-cluster feature).

**Lookup**: `grep -rn "Lgain\|L_gain" trinity/phase_general/ trinity/bubble_structure/` to find where
`Lgain` is computed per segment, and check whether it is floored or can pass through the SPS
wind→SN minimum.

---

### S11-R-08 — float-fragile `R2 >= rCloud` after a root-solved event — S2

**A** (`S11-A-19`): `if (nSnap_rCloud is not None and nSnap_rCloud == 0 and params['R2'].value >=
params['rCloud'].value):` at `main.py:264-265`, immediately after phase 1a terminated on a
`cloud_boundary` event whose root was located by scipy's bracketed root finder on `R2 − rCloud`.

**B** (`S11-B-23`): the comment at `main.py:260` asserts *"The energy-phase reconciliation snapshot
already captured R2 = rCloud"* — an exact-equality claim — and B flags that if the snapshot is at the
last accepted step rather than exactly at `rCloud`, the assertion fails.

**C** (§2.4): treats `R2 = rCloud` as the RHS-discontinuity restart point and requires the bookkeeping
crossing to be keyed off the event, not re-derived.

**Diagnosis — A≠C (numerics) with an A≠B overtone (the comment asserts an equality the arithmetic
cannot deliver).** scipy locates the root by `brentq` on the dense output to `xtol ≈ 4e-12` in `t`;
the state at that `t` satisfies `R2 − rCloud ≈ 0` with **arbitrary sign**. `>=` is therefore a coin
flip at the ULP level. When it lands low, the run does not stop, silently continues into 1b/1c/2, and
produces a completely different trajectory from an otherwise identical run. This is exactly the
"float equality on event triggers" hazard.

**Fix shape**: key off the `cloud_boundary` event having fired (the `EventResult` already carries the
name and reason code) rather than re-testing the radius; or compare with a relative tolerance.

**Reachability**: only with `stop_at_rCloud_nSnap = 0` configured.

**Repro**: run the same config twice with slightly different `rtol` so the root lands on either side;
observe whether `EndSimulationDirectly` is set.

---

### S11-R-09 — `apply_event_result` ignores the event root and can drop `Eb` — S2, OPEN

**A** (`S11-A-12`, `S11-A-14`): `:612` writes `params['t_now'].value = t` (the *argument*), `:615-617`
writes `params[state_keys[i]].value = float(y[i])` (the *argument*) — `result.t` (from
`float(t_ev[0])`) and `result.y` (from `y_ev[0].copy()`) are extracted and then read nowhere. The
write loop is bounded by `len(state_keys)`, default 2, while the transition phase's state is
`[R2, v2, Eb]` (`make_energy_floor_event(..., y_index=2)`).

**B** (`S11-B-10`): documents `state_keys` default `['R2','v2']` against `main.py:225`'s
`# y0 = [r0, v0, E0, T0]` — a documented **four**-component state versus a two-key writer.

**C** (`S11-C-22`, S1): must write the dense-output state at `t_root`, not `sol.y[:, -1]`, and
`state_keys` must be derived from the target phase's declared layout; a stale `Eb` is "an unaccounted
energy injection or removal whose magnitude grows with segment length… silently, because R2 and v2
remain continuous and the trajectory looks smooth."

**Diagnosis — A=B=C on the hazard shape; entirely caller-dependent on whether it bites.** If every
call site passes `state_keys=['R2','v2','Eb']` in energy-type phases and passes the root state as
`t`/`y`, there is no defect and the unused `result.t`/`result.y` fields are S4 hygiene. If any call
site relies on the default, or passes `sol.t[-1]`/`sol.y[:, -1]`, it is S1 per C.

**Note the internal inconsistency worth recording separately (S3)**: B's `y0 = [r0, v0, E0, T0]`
(4 components) versus A's and C's inference of a 3-component `[R2, v2, Eb]` integrated state. Both
are consistent with `y_index=2 → Eb`, but they cannot both describe the ODE state vector.

**Status: OPEN.**

**Lookup**: `grep -rn "apply_event_result" trinity/` and read the `t`, `y` and `state_keys` arguments
at every call site. Secondarily: `grep -rn "solve_ivp" trinity/phase_general/` and check whether
`dense_output=True` and `sol.sol(t_root)` are used, or `sol.y[:, -1]`.

---

### S11-R-10 — `== False` gate on a key `run_expansion` never initialises — S2, OPEN

**A** (`S11-A-11`, §6.1 literal table): `run_expansion` initialises only `t_now, R2, v2, Eb, T0`
(`:232-238`) and `current_phase` (`:244`). `EndSimulationDirectly` is first **read** at `:283`. All
three phase gates are written as `if params['EndSimulationDirectly'].value == False:` — an
identity-style comparison, not `if not …`.

**B**: silent — no prose covers initialisation of the termination flags.

**C** (`S11-C-30`, §6): process- and order-independence is a hard requirement, and the project's own
`CLAUDE.md` records that trinity leaks module-level global state in-process.

**Diagnosis — two coupled hazards, both OPEN on the same lookup.**
1. *Type sensitivity*: `== False` is `True` for `False`, `0`, `0.0`, `np.False_`; it is **`False` for
   `None`**. If the schema default for `EndSimulationDirectly` is `None` (or the key is absent and
   `.value` returns `None`), phases 1b, 1c and 2 are **skipped on every run**, silently, while all
   three "Entering …" banners print (see S11-R-11). That would be an S1 across the board — and
   plausibly visible as runs that end suspiciously early.
2. *Cross-run leakage*: nothing resets `EndSimulationDirectly`, `SimulationEndReason`,
   `SimulationEndCode`, `isCollapse` or `current_phase` at the top of `run_expansion`. A second
   `run_expansion(params)` on the same object inherits the first run's `True` and skips 1b/1c/2.

Hazard 1 is a reconciler-originated observation from A's literal transcription, not a claim any lens
made. It is a hypothesis until the schema default is read. Hazard 2 is A's, corroborated structurally
by C's §6 and by `CLAUDE.md`, but its reachability depends on whether any driver reuses a `params`
object.

**Lookup**: `grep -rn "EndSimulationDirectly" trinity/_input/` — read the declared default and its
type. Then `grep -n "Pool\|Process\|start_expansion\|run_expansion" run.py` — does any path call
`start_expansion` more than once per process, or reuse a `params` object?

---

### S11-R-11 — phase label and banner advance before the gate — S3

**A** (`S11-A-10`): `params['current_phase'].value = 'implicit'` at `:278`, banner at `:280-281`,
**then** `if params['EndSimulationDirectly'].value == False:` at `:283`. Identical ordering at
`:298-303` and `:324-343`. A run that ends inside phase 1a therefore finishes with
`current_phase == 'momentum'` and all three later banners printed.

**B**: `main.py:260` — *"we explicitly do NOT want phases 1b/1c/2 to advance past it."*

**C** (`S11-C-27`, §3.3): the phase label is published science — the timeline figure's phase fractions
are built from it, and `implicit` is merged into `energy` for display.

**Diagnosis — A≠B, clean doc-drift with a metadata consequence.** The comment states the intent
("do NOT advance"); the code advances the label anyway. Any reader or downstream tool keyed on
`current_phase` mis-classifies a phase-1a termination as having reached the momentum phase.

**Severity is S3 or S2 depending on one lookup**: if `current_phase` is only a log label and a
metadata field, S3; if it is written per-snapshot into `dictionary.jsonl` and consumed by
`paper/methods/make_figures.py` for phase durations, the mis-labelled tail biases published phase
fractions and it is S2.

**Lookup**: `grep -rn "current_phase" trinity/_output/ paper/ tools/`.

---

### S11-R-12 — one end-code channel for physical fates and solver guards — S3

**B** is the primary source here: `min_radius` is documented both as "R2 < coll_r (**shell
collapse**)" and as "prevents **LSODA from crashing**"; `velocity_runaway` is documented as
"(**numerical instability**)" yet classified under "Simulation-Ending Events" alongside physical fates
and issued with a first-class `reason_code`/`SimulationEndCode`.

**A** confirms the mechanism: `SHELL_COLLAPSED` and `VELOCITY_RUNAWAY` are ordinary `SimulationEndCode`
values written through the same `apply_event_result` path as everything else; there is no `kind` field
on `EventResult`.

**C** (`S11-C-11`, `S11-C-14`, `S11-C-17`, `S11-C-21`, §3.3) requires
`{physical_fate, model_domain, numerical_cutoff, solver_failure}` as a structured field with a
non-zero exit code for the last two, on the argument that the published dispersal/recollapse boundary
otherwise becomes a map of solver stiffness.

**Diagnosis — B≠C (the docs claim a taxonomy the spec says is the wrong taxonomy), with A confirming
the code implements the docs' version.** This is the structural root of S11-R-02 and S11-R-03; the
substring heuristic and the missing failure channel are both symptoms of there being no `kind`.

---

### S11-R-13 — `run_expansion` docstring claims multi-generation evolution that does not exist — S3

**A** (`S11-A-17`): `expansion_next(tStart, ODEpar, sps_data_old, sps_f_old, mypath, cloudypath,
ii_coll)` at `main.py:366` takes seven parameters, uses none, and returns `None`. `run_expansion` is a
straight-line sequence with no loop and no back edge.

**B** (`S11-B-17`): docstring says "until **next recollapse** or (if no re-collapse) until end of
simulation"; `main.py:209` is `# TODO: add loop so that this simulation starts over with old
generation of parameter to simulate new starburst environment`.

**C** (`S11-C-28`, §1.2): `expansion_next(..., ii_coll)` *should* reset the phase machine, reset the
SPS clock to a new cluster age zero, clear the bubble state, carry forward remaining gas mass and
cumulative stellar mass, and cap the episode count with its own outcome reason.

**Diagnosis — A≠B (docstring claims behaviour that does not exist) and A≠C (the capability the
signature advertises is absent).** Not S1: nothing computes a wrong number, the feature simply is not
there. But a reader who trusts the docstring will interpret single-generation results as
multi-generation. Per project rule 3, `expansion_next` should be `git mv`'d to
`docs/dev/to-be-removed/` rather than deleted, and the docstring corrected to describe
single-generation evolution.

---

### S11-R-14 — `MAX_VELOCITY_EXPANSION` documented as live, is dead — S3

**A** (`S11-A-16`): `MAX_VELOCITY_EXPANSION = 1000.0` appears nowhere except its definition; all four
builders call `make_velocity_runaway_event(MAX_VELOCITY_COLLAPSE, direction='collapse')`, so the
`elif "expansion"` and the `else` branches never execute. The `else` also silently accepts any
misspelled `direction` string instead of raising.

**B** (`S11-B-05`): documents the two constants as an "asymmetric pair" (inward ~490 km/s, outward
~978 km/s) and worries that `"both"` mode clips outward velocities at the collapse cap.

**C** (`S11-C-16`, `S11-C-17`): both branches should exist with branch-specific residual forms; the
caps should be referred to physical bounds (wind terminal speed ~1–3×10³ km/s; free fall
`v_ff = sqrt(2G(M_*+M_sh)/R2)`).

**Diagnosis — A≠B (docs describe a pair that is half dead) plus S4 hygiene.** B's fear that `"both"`
mode clips outward expansion at 500 pc/Myr does **not** materialise, because no builder uses `"both"`.
The practical consequence is only that outward runaway is not caught by velocity — though `max_radius`
covers it by radius when armed (see S11-R-04).

**Silent-typo hazard (S4, real)**: `direction='colapse'` falls through to the `else` branch and
silently yields the `|v2| > v_max` semantics with no error.

---

### S11-R-15 — boundary `pRam` computed, logged as consumed, discarded — S3

**A** (`S11-A-18`, `S11-A-24`): `main.py:336-341` computes `P_ram_bnd = pRam(R2_bnd, Lmech_bnd,
v_mech_bnd)` and logs "(momentum phase will use this)"; the value is never assigned into `params` and
never referenced again. The block runs unconditionally — the `EndSimulationDirectly` gate is at
`:343` — and **after** `params.reset_keys(COOLING_PHASE_KEYS)` at `:317`, which is itself
unconditional.

**B** (§4.3): quotes only the neutral banner `# --- Transition -> Momentum boundary diagnostic ---`
(`:335`); the "will use this" claim lives in a log string, not a comment, which is why B did not
record it.

**C** (§4.2): `P_ram` is explicitly on the "must be recomputed, never carried" list — so *not* passing
it to the momentum phase is correct. C's concern is the opposite: recomputation must happen at the new
state.

**Diagnosis — A≠C on the *log text*, not on the behaviour.** Discarding `P_ram_bnd` is right per C §4.2;
the log line claiming the momentum phase will use it is the defect (S3), plus S4 dead computation.

**One genuine open sub-question**: if `Lmech_total` / `v_mech_total` are members of
`COOLING_PHASE_KEYS`, the logged diagnostic is computed from freshly-reset values and is meaningless.
**Lookup**: `grep -n "COOLING_PHASE_KEYS" -A 40 trinity/_input/dictionary.py` and check for
`Lmech_total` and `v_mech_total`.

---

### S11-R-16 — `cloud_boundary` is one-sided across an RHS discontinuity — S3

**A**: `g = y[0] - rCloud`, `direction = +1`, `terminal = True`, `is_simulation_ending = False`.

**B**: "Only trigger when R2 crosses rCloud from below" (`:244`); "Phase ending, not simulation
ending" (`:246`).

**C** (`S11-C-19`, §2.4): `R2 = rCloud` is where `ρ_amb` switches from the cloud profile to `n_ISM`
(SPEC-021, SPEC-060) — a **discontinuity in the ODE right-hand side**. The restart job needs
`direction = 0` because a shell that crossed outward can turn around and re-enter; the bookkeeping job
(`stop_at_rCloud_nSnap`) needs `+1`. One event cannot do both.

**Diagnosis — A≠C.** Code and docs agree (`+1`); the spec says one direction is insufficient. Concretely:
after phase 1a hands off at `R2 ≈ rCloud`, phases 1b/1c/2 do not arm `cloud_boundary` at all (per A's
builder table), so an inward re-crossing in the momentum phase steps across the density jump with no
restart and no valid error estimate — precisely where the trajectory is most sensitive.

**Status**: CONFIRMED that the code is one-sided and that later phases do not re-arm it. OPEN whether
inward re-crossing occurs in shipped configs (it does in any recollapsing run that got past `rCloud`).

**Lookup**: `grep -rn "rCloud" trinity/cloud_properties/` to confirm the ambient-density branch really
switches at `rCloud` in the RHS, rather than being smoothed.

---

### S11-R-17 — absolute `energy_floor` as the 1c→2 criterion — S3, scope/design divergence

**A = B** exactly: `energy_floor: float = 1e3` code units (≈ 1.9×10⁴⁶ erg), `g = Eb − energy_floor`,
`direction = -1`, `is_simulation_ending = False`.

**C** (`S11-C-06`, `S11-C-25`, §2.3): the physical statement is `P_b ≪ P_HII + P_ram`, so the criterion
should be relative; one absolute number cannot express the same smallness across the shipped
`mCloud` range (~10⁵ to 5×10⁹ M⊙). At the massive end the floor is reached far too late; at the low
end it may already be satisfied at phase entry, giving a zero-length transition phase. Both silent.
Separately, `B` (`S11-B-14`) records the documented `Eb ≈ 1e3 → 0` step at the 1c→2 boundary and `C`
(`S11-C-25`) requires that discarded residual to enter an explicit energy ledger.

**Diagnosis — A = B (code matches its docs) but C says the design itself is wrong.** This is a design
divergence, not a coding error; the actionable part is (a) record `energy_floor` and the discarded
`Eb` per run, (b) assert `Eb(entry) > energy_floor` in `build_transition_phase_events` so the
zero-length-transition case is loud rather than silent.

---

### S11-R-18 / S11-R-19 / S11-R-20 / S11-R-21 / S11-R-22 — S4 hygiene

- **R-18**: `is_simulation_ending=getattr(event, 'is_simulation_ending', True)` (`:401`) defaults a
  missing attribute to **True** — an event that forgets the flag ends the whole simulation. Two lines
  later `end_code` defaults to the safe `None`. All nine current events set the flag, so this is
  latent. Fail-safe is `False`, or raise.
- **R-19**: `state_keys: List[str] = ['R2','v2']` (`:589`) is one list object evaluated at import and
  shared by every call for the process lifetime. C names mutable defaults explicitly in its
  global-state class list (§6 item 5).
- **R-20**: `min_r = max(1.5·coll_r, 0.01)` makes `MIN_RADIUS_SAFETY = 0.01` pc unreachable for any
  `coll_r > 0.0067` pc. If `coll_r` defaults to ~1 pc (C cites SPEC-101), the constant is dead and the
  documented "or MIN_RADIUS_SAFETY" alternative never applies. **Lookup**:
  `grep -rn "coll_r" trinity/_input/default.param trinity/_input/*.py`.
- **R-21**: `f_mass = params['mCluster'] / params['sps_refmass']` (`main.py:144`) omits `.value` on both
  sides, unlike every neighbouring access. A rated this low confidence and could not verify.
  **A deduction that bounds it**: `python run.py param/simple_cluster.param` is the documented
  quickstart, and this line is on the unconditional startup path — so it must evaluate, meaning
  `DescribedItem` implements `__truediv__` and `__format__`. That makes it at most a consistency wart
  (S4), not a crash. **Lookup**: `grep -n "__truediv__\|__format__" trinity/_input/dictionary.py`.
- **R-22**: `events[i]` is re-indexed positionally against `sol.t_events` with no
  `len(events) == len(sol.t_events)` assertion, while `build_implicit_phase_events` alone returns
  `(events, factory)` rather than a bare list. A rated this low confidence; B documents the tuple
  return, so it is at least deliberate. Downgraded to S4/OPEN pending the same runner lookup as R-06.

---

## 4. Affirmative clearances

All three lenses agree and the spec sanctions the following. These are deliverables, not filler —
lens C's trap catalogue predicted several of these as likely defects and the code is clean on each.

| # | Cleared item | Why it clears |
|---|---|---|
| C1 | **Collapse-runaway residual form.** `g = v2 + v_max`, `direction = -1`. | C's trap C(i) is the single most damaging sign error available here — writing the collapse branch in the expansion form (`v_max − v2 → +∞` as `v2 → −∞`) makes infall runaway invisible and produces `R2 < 0` then NaNs. A reads the correct form; B documents "Triggers when v2 < −v_max". |
| C2 | **min_radius residual/direction.** `g = R2 − min_r`, `direction = -1`, terminal. | Matches `S11-C-12` exactly. B: "Only trigger when R2 crosses min_r from above." |
| C3 | **max_radius residual/direction agree.** `g = R2 − max_r` with `direction = +1`. | `S11-C-15` warns that `stop_r − R2` with `+1` would silently never fire. The convention is internally consistent. (The *arming* problem, S11-R-04, is a separate defect and does not touch the sign convention.) |
| C4 | **energy_floor residual/direction.** `g = Eb − energy_floor`, `direction = -1`. | Matches `S11-C-07`'s required sense. |
| C5 | **velocity_sign direction is −1 (turnaround), not +1 (bounce).** | `S11-C-09` / trap E predicts the opposite-direction error, in which a re-expanding shell is recorded as "collapse detected". The code detects the physical `+→−` crossing. |
| C6 | **velocity_sign is `terminal = False` and `is_simulation_ending = False`.** | `S11-C-08` (S1) requires the integration to continue through `v2 = 0` so re-collapse is reachable. It does. B documents this in four consistent places. |
| C7 | **`cloud_boundary` and `energy_floor` are `is_simulation_ending = False`.** | C's trap D — "cooling_balance/energy_floor run-terminal ⇒ every run reports an outcome at a phase boundary and no run ever reaches a physical fate" — does **not** apply. Phase boundaries are correctly solver-terminal but not run-terminal. B's four explicit "Phase ending, not simulation ending" declarations match the code. |
| C8 | **All seven `solve_ivp` residuals are continuous `quantity − threshold` distances.** | C's trap F (boolean/indicator residuals whose root time depends on step sequence) does not apply to any of them. The sole exception is `cooling_balance`, tracked as S11-R-06. |
| C9 | **Phase-label monotonicity within an episode.** | `S11-C-26` requires monotone non-decreasing phase labels or explicit hysteresis, warning about chatter producing zero-length phases. `run_expansion` is a straight-line, non-looping, non-re-entrant sequence with no back edge — monotonicity holds by construction and chatter is structurally impossible. |
| C10 | **`current_phase` enum excludes `'collapse'`.** | `S11-C-27`: collapse must be a post-processing label on the final momentum interval, not an integration phase. The enum is exactly `{energy, implicit, transition, momentum}`. |
| C11 | **`Lgain ≤ 0` is guarded and returns the "not yet" sign** (`+1.0`). | C's requested disarm-on-no-gain exists. (Only the near-zero-positive case is unguarded — S11-R-07.) |
| C12 | **De-aliasing is deliberate where it matters.** `y_ev[0].copy()` at `:400`; `float(y[i])` at `:617`. | `EventResult.y` does not alias the solver buffer, and scalars written into `params` do not alias `y`. |
| C13 | **`min_r` is latched by construction against the start-of-run false positive.** | `S11-C-13` predicts every compact-cluster run terminating at t≈0 as "collapsed". `direction = -1` means the initial outward `−→+` crossing is ignored; only the later infall crossing fires. C's structural point stands, its predicted symptom does not. |

---

## 5. Open questions for the orchestrator

Each is a claim I declined to promote to a fact, with the single lookup that resolves it.

| # | Question | Decisive lookup | Which findings it moves |
|---|---|---|---|
| Q1 | Is the `cooling_balance` factory spliced into `solve_ivp(events=…)` or evaluated per segment in Python? | `grep -rln "run_phase_energy" trinity/phase_general/`, then in that file `grep -n "build_implicit_phase_events\|cooling\|solve_ivp"` | **S11-R-06**: S1 if spliced, S3 if per-segment. Also R-22. |
| Q2 | What `t`, `y` and `state_keys` does each `apply_event_result` call site pass? Is `dense_output`/`sol.sol(t_root)` used, or `sol.y[:, -1]`? | `grep -rn "apply_event_result" trinity/` and `grep -rn "solve_ivp" trinity/phase_general/` | **S11-R-09**: S1 if defaults/last-step are used, S4 otherwise. |
| Q3 | What is the schema default **and type** of `EndSimulationDirectly`? | `grep -rn "EndSimulationDirectly" trinity/_input/` | **S11-R-10** hazard 1: S1 if the default is `None` (all runs skip 1b/1c/2), S4 if `False`. |
| Q4 | Does any code outside this slice inspect `sol.status` / `sol.success`? Does `run.py` propagate `start_expansion`'s return value? | `grep -rn "\.status\b\|\.success\b\|sol\.message" trinity/` ; `grep -n "start_expansion\|sys.exit" run.py` | **S11-R-03**: confirms whether the failure channel is absent everywhere or only here. |
| Q5 | What is the default `stop_r`, and in shipped `.param` files is it above or below `rCloud`? | `grep -rn "stop_r" trinity/_input/default.param param/*.param docs/dev/performance/f1edge_*.param` | **S11-R-04** and the `large_radius_event` half of **S11-R-02**: reachable (S1) vs latent (S2). |
| Q6 | Is `current_phase` written per-snapshot and consumed by the figure scripts? | `grep -rn "current_phase" trinity/_output/ paper/ tools/` | **S11-R-11**: S2 if it feeds published phase fractions, S3 if log/metadata only. |
| Q7 | Is `isCollapse` read anywhere downstream? | `grep -rn "isCollapse" trinity/ paper/ tools/ test/` | **S11-R-02**: sets whether the mis-classification propagates into published figures or stops at metadata. |
| Q8 | Are `Lmech_total` / `v_mech_total` members of `COOLING_PHASE_KEYS`? | `grep -n "COOLING_PHASE_KEYS" -A 40 trinity/_input/dictionary.py` | **S11-R-15**: whether the boundary diagnostic is computed from reset values. |
| Q9 | What is the default `coll_r`? | `grep -rn "coll_r" trinity/_input/default.param trinity/_input/*.py` | **S11-R-05** (size of the 1.5× offset in pc) and **S11-R-20** (is `MIN_RADIUS_SAFETY` dead?). |
| Q10 | Does any execution path call `start_expansion`/`run_expansion` more than once per process, or reuse a `params` object? | `grep -n "Pool\|Process\|Thread\|start_expansion\|run_expansion" run.py` | **S11-R-10** hazard 2: whether cross-run flag leakage is reachable in sweeps. |
| Q11 | Is `stop_at_rCloud_nSnap >= 1` implemented anywhere outside `main.py`? | `grep -rn "stop_at_rCloud_nSnap\|nSnap_rCloud" trinity/ run.py` | Whether the non-zero semantics B and C both describe exist at all (currently undocumented and, in `main.py`, unhandled). |
| Q12 | Does `Lgain` in phase 1b pass near zero from above (SPS wind→SN gap)? | `grep -rn "Lgain\|L_gain" trinity/phase_general/ trinity/bubble_structure/` and inspect whether it is floored | **S11-R-07**: reachable (S2) vs unreachable (S4). |
| Q13 | Does `DescribedItem` implement `__truediv__` / `__format__`? | `grep -n "__truediv__\|__format__" trinity/_input/dictionary.py` | **S11-R-21**: closes it either way (the quickstart running implies yes). |
| Q14 | Is the integrated ODE state 3-component `[R2, v2, Eb]` or 4-component `[R2, v2, Eb, T0]`? | `grep -rn "solve_ivp" trinity/phase_general/` and read the `y0=` argument | Resolves the A/C vs B disagreement noted under **S11-R-09**; also validates `y_index=2`. |

**Standing caveat for the orchestrator.** Findings R-01, R-06, R-09 and R-22 all rest on the same
unread code — the four phase runners. All three lenses share the assumption that
`check_event_termination` and `apply_event_result` are called with the builders' lists and with the
event root. Q1 + Q2 together clear or confirm that whole cluster, and are the highest-value pair of
lookups in this slice.
