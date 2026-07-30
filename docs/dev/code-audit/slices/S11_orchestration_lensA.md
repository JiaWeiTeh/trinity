# S11 orchestration — Lens A (what the code does)

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

**Slice files read (only these):**
`trinity/main.py` · `trinity/phase_general/phase_events.py` · `trinity/phase_general/__init__.py` · `trinity/__init__.py`
**Shared read-only exception used:** `trinity/_functions/unit_conversions.py` (S1 copy) — consulted only to pin the astro-unit system (Msun / pc / Myr) and the meaning of `E_au2cgs`, `ndens_au2cgs`, `Pb_au2_KcmInv`, `v_au2kms`.
**Not read:** the slice's `prose.md` / `signatures.md`, the real `trinity/` tree, `docs/dev/`, any other lens report. All comments and docstrings were blanked in my copies; I make no claims about intent, only about behaviour.

Callers of the event machinery (`phase1_energy`, `phase1b_energy_implicit`, `phase1c_transition`, `phase2_momentum`) are **outside my slice**. Where a conclusion depends on them I say so explicitly and mark confidence accordingly. `phase_general/__init__.py` is empty (1 blank line); `trinity/__init__.py` contains only `__version__ = "1.0.0"` and `__author__` (lines 22–23) — no re-exports, no side effects.

---

## 1. The phase state machine

### 1.1 The graph as written

`run_expansion` (`trinity/main.py:216`) is a **straight-line, non-branching, non-looping sequence**. There is no dispatch on which event fired, no re-entry, no back edge.

```
start_expansion (main.py:81)
  ├─ logging.basicConfig  (only if root logger has no handlers)   :104
  ├─ get_InitCloudProp                                            :122
  ├─ _check_stop_r_rCloud_interaction  -> warn/info only          :128
  ├─ read_sps + get_interpolation -> params['sps_data','sps_f']   :147-153
  ├─ np.loadtxt(cooling) + interp1d -> 3 cStruc_* params          :165-171
  ├─ run_expansion(params)                                        :180
  ├─ write_simulation_end(params) -> exit_code   [except -> 99]   :191/198
  ├─ params.write_termination_report(reason)     [except -> warn] :203
  └─ return 0            <-- ALWAYS 0, exit_code discarded        :211

run_expansion (main.py:216)
  get_y0 -> (t0,r0,v0,E0,T0) -> params t_now,R2,v2,Eb,T0          :232-238
        │
        ▼
  current_phase='energy'                                          :244
  PHASE 1a  run_energy(params)                UNCONDITIONAL       :251
        │
        ▼
  if stop_at_rCloud_nSnap is not None and ==0 and R2>=rCloud:     :264
        EndSimulationDirectly=True
        SimulationEndReason="Reached cloud edge (stop_at_rCloud_nSnap=0)"
        SimulationEndCode=SimulationEndCode.RCLOUD_BOUNDARY.code
        │
        ▼
  current_phase='implicit'   <-- set BEFORE the gate              :278
  banner + terminal_prints.phase(...)  <-- printed BEFORE gate    :280-281
  if EndSimulationDirectly == False:  PHASE 1b run_phase_energy   :283-286
  else: log "skipping implicit phase"
        │
        ▼
  current_phase='transition' <-- set BEFORE the gate              :301
  banner + terminal_prints.phase(...)  <-- printed BEFORE gate    :298-299
  if EndSimulationDirectly == False:  PHASE 1c run_phase_transition :303-306
  else: log "skipping transition phase"
        │
        ▼
  params.reset_keys(COOLING_PHASE_KEYS)   UNCONDITIONAL           :317
        │
        ▼
  current_phase='momentum'   <-- set BEFORE the gate              :327
  banner + terminal_prints.phase(...)  <-- printed BEFORE gate    :325
  P_ram_bnd = pRam(R2, Lmech_total, v_mech_total)  UNCONDITIONAL  :336-341
        (log-only; never stored, never passed on)
  if EndSimulationDirectly == False:  PHASE 2 run_phase_momentum  :343-346
  else: log "skipping momentum phase"
        │
        ▼
  params.flush()  [except -> warn]                                :356
  return None                                                     :363
```

### 1.2 Phase enumeration and transition conditions

| # | `current_phase` value | set at | entry condition | runner |
|---|---|---|---|---|
| 1a | `'energy'` | main.py:244 | **unconditional** — always runs | `run_energy_phase.run_energy` |
| 1b | `'implicit'` | main.py:278 | `EndSimulationDirectly == False` | `run_energy_implicit_phase.run_phase_energy` |
| 1c | `'transition'` | main.py:301 | `EndSimulationDirectly == False` | `run_transition_phase.run_phase_transition` |
| 2 | `'momentum'` | main.py:327 | `EndSimulationDirectly == False` | `run_momentum_phase.run_phase_momentum` |

Only **one** transition condition exists in this file, and it is the same boolean for all three gates: `params['EndSimulationDirectly'].value == False`. There is no condition that selects *between* phases; there is no condition that can send the run from 1a straight to 2, or from 1c back to 1b. `current_phase` is a **label written unconditionally**, never a dispatch key.

**Reachability.** All four phases are reachable. Nothing here is unreachable at phase granularity. But:

* Phase 1a is **not** gated — an `EndSimulationDirectly` that is already `True` on entry (set by a previous run against the same `params`, see §4) does not stop phase 1a from running.
* Because `current_phase` is assigned *before* each gate, a run that ends inside phase 1a finishes with `params['current_phase'].value == 'momentum'` and with the terminal banners for 1b, 1c and 2 all printed to the user. The recorded phase never tells you where the run actually stopped. (`S11-A-10`)
* Only `stop_at_rCloud_nSnap == 0` is handled in this file. `nSnap >= 1` produces no code path here at all; the `_check_stop_r_rCloud_interaction` messages (main.py:41–78) are advisory strings and change no behaviour.
* `expansion_next` (main.py:366) takes seven parameters, uses none, and returns `None`. Pure dead stub. (`S11-A-17`)

---

## 2. Event functions — exhaustive

All events have the scipy `solve_ivp` signature `event(t, y) -> float`; `y[0]` is `R2` (pc), `y[1]` is `v2` (pc/Myr), `y[2]` (where present) is `Eb` (astro-unit energy). scipy `direction = -1` catches only **positive→negative** crossings; `direction = +1` only **negative→positive**.

### 2.1 Table of every event factory

| factory (line) | expression `g(t,y)` | zero when | `direction` | crossing actually caught | `terminal` | `is_simulation_ending` | `reason_code` | `end_code` |
|---|---|---|---|---|---|---|---|---|
| `make_min_radius_event` :99 | `y[0] - min_r` :122 | `R2 == min_r` | `-1` :125 | R2 **falling** through `min_r` | **True** :124 | **True** :127 | `small_radius_event` | `SHELL_COLLAPSED` :130 |
| `make_max_radius_event` :134 | `y[0] - max_r` :154 | `R2 == max_r` | `+1` :157 | R2 **rising** through `max_r` | **True** :156 | **True** :159 | `large_radius_event` | `LARGE_RADIUS` :162 |
| `make_velocity_runaway_event(direction="collapse")` :189 | `y[1] + v_max` :192 | `v2 == -v_max` | `-1` :193 | v2 **falling** through `-500` | **True** :208 | **True** :210 | `velocity_runaway_event` | `VELOCITY_RUNAWAY` :212 |
| … `direction="expansion"` :195 | `v_max - y[1]` :198 | `v2 == +v_max` | `-1` :199 | v2 **rising** through `+500` | True | True | same | same |
| … any other `direction` :201 | `v_max - abs(y[1])` :204 | `abs(v2) == v_max` | `-1` :205 | `abs(v2)` **rising** through 500 | True | True | same | same |
| `make_cloud_boundary_event` :220 | `y[0] - rCloud` :241 | `R2 == rCloud` | `+1` :244 | R2 **rising** through `rCloud` | **True** :243 | **False** :246 | `cloud_boundary` | *(none set → `None`)* |
| `make_energy_floor_event` :252 | `y[2] - energy_floor` :276 | `Eb == floor` | `-1` :279 | Eb **falling** through floor | **True** :278 | **False** :281 | `energy_floor` | *(none)* |
| `make_velocity_sign_event` :287 | `y[1]` :308 | `v2 == 0` | `-1` :311 | v2 **falling** through 0 (expansion→collapse only) | **False** :310 | **False** :313 | `velocity_sign_change` | *(none)* |
| `make_cooling_balance_event(...)(Lgain,Lloss)` :319/341 | `1.0` if `Lgain<=0` else `(Lgain-Lloss)/Lgain - 0.05` :343–346 | ratio `== 0.05` | `-1` :349 | *(see below — none)* | **True** :348 | **False** :351 | `cooling_balance` | *(none)* |

Every sign convention in the table above is **self-consistent** with its stated `direction` — I found no event whose sign flips the crossing out of reach *by construction*. The problems are elsewhere (initial-condition side, and the caller).

### 2.2 The cooling-balance event cannot fire under the solver

`make_cooling_balance_event` (:319) returns a **factory** (:341), not an event. The factory closes over scalar `Lgain` and `Lloss` (:341) and the returned `event(t, y)` (:342) **reads neither `t` nor `y`** (:343–346). Its value is therefore a **constant** for the whole of any one `solve_ivp` call — either `1.0` (when `Lgain <= 0`) or the fixed number `(Lgain-Lloss)/Lgain - 0.05`. A constant function has no zero crossing, so scipy can never report a root for it regardless of `terminal=True` / `direction=-1`. If the caller in `phase1b` genuinely hands this to `solve_ivp`, it is a dead event. It can only "work" if the caller rebuilds it per step and evaluates its **sign** by hand outside the integrator. Caller is outside my slice → medium confidence on which of the two it is; high confidence on the constancy. (`S11-A-04`)

Also on this factory: the `Lgain <= 0` guard is loop-invariant (re-tested on every call), and line 353's message is an f-string with no placeholders.

### 2.3 Which events exist in which phase — and their index order

Index order matters enormously; see §2.4.

| builder | idx 0 | idx 1 | idx 2 | idx 3 (conditional) | extras |
|---|---|---|---|---|---|
| `build_energy_phase_events` :423 | `cloud_boundary(rCloud)` | `min_radius(min_r)` | `velocity_runaway(collapse)` | **— no `stop_r` event —** | |
| `build_implicit_phase_events` :458 | **`velocity_sign` (terminal=False)** | `min_radius` | `velocity_runaway` | `max_radius(stop_r)` iff `stop_r is not None and stop_r > 0` :494 | also returns `cooling_factory` :497/501 |
| `build_transition_phase_events` :504 | `energy_floor(1e3, y_index=2)` | `min_radius` | `velocity_runaway` | `max_radius(stop_r)` :538 | |
| `build_momentum_phase_events` :546 | `min_radius` | `velocity_runaway` | `max_radius(stop_r)` :576 | | |

`min_r = max(coll_r * MIN_RADIUS_FACTOR, MIN_RADIUS_SAFETY) = max(1.5 * coll_r, 0.01)` pc, computed identically in all four builders (:445, :485, :529, :568). All builders capture `rCloud`/`coll_r`/`stop_r` **by value at build time** — if any of them changes mid-phase, the event keeps the stale bound.

Note the structural asymmetry: **`stop_r` has no event in phase 1a.** It is the only phase where the user's stop radius cannot stop anything. (`S11-A-20`)

`build_implicit_phase_events` returns a **2-tuple** `(events, factory)` while the other three return a bare list — a caller that treats the four builders uniformly gets a tuple where it expects a list.

### 2.4 How the calling code decides an event ended a phase — it does **not** check `terminal`

`check_event_termination(sol, events)` (:363) is the decision function. Verbatim logic:

```python
if sol.t_events is None: -> EventResult(triggered=False, ...)          :379-389
for i, (t_ev, y_ev) in enumerate(zip(sol.t_events, sol.y_events)):     :392
    if len(t_ev) > 0:                                                  :393
        event = events[i]                                              :394
        return EventResult(triggered=True, ..., t=float(t_ev[0]), y=y_ev[0].copy(), ...)
return EventResult(triggered=False, ...)                               :407-416
```

Stated plainly, as requested:

1. **`event.terminal` is never read anywhere in this file outside the factories that set it.** `grep`-equivalent: `terminal` appears only at :124, :156, :208, :243, :278, :310, :348 — all *assignments*. It is never inspected. (`S11-A-01`)
2. Termination is inferred from **`len(sol.t_events[i]) > 0`** — the mere presence of a recorded root (:393). In scipy, `t_events[i]` is populated for **every** event with a detected crossing, terminal or not; a `terminal=False` event records all of its roots while integration continues to `t_end`.
3. The winner is the **lowest-index** event with any root, **not the earliest in time** (:392 iterates in list order and returns on the first hit). If `min_radius` (idx 1) and `velocity_runaway` (idx 2) both have roots, idx 1 is reported whichever happened first. (`S11-A-03`)
4. The reported time/state is **`t_ev[0]` / `y_ev[0]`** (:399–400) — the *first* root of that event, which for a non-terminal event may be arbitrarily far before the actual end of integration.
5. `sol.status`, `sol.success`, `sol.message` are **never consulted**. (§3)

**Non-terminal events that can be read as a termination.** Exactly one event in the codebase has `terminal = False`:

* **`make_velocity_sign_event` (:287, `terminal = False` at :310)** — and it is placed at **index 0** of `build_implicit_phase_events` (:488). It is therefore the first thing `check_event_termination` looks at in phase 1b. A single downward `v2` zero-crossing — the ordinary, expected expansion→collapse turnover — makes `check_event_termination` return `triggered=True, name='velocity_sign', t=<first crossing>` and **shadow every genuinely terminal event in the same solve**: `min_radius` (idx 1), `velocity_runaway` (idx 2) and `max_radius(stop_r)` (idx 3) are never even examined. The phase is reported as ending at the first sign change, at that earlier `t`, with that earlier `y`. (`S11-A-02`)

The `cooling_balance` event, were it ever spliced into the list, would be the second affected case — but by §2.2 it cannot produce a root at all, so it is inert rather than misleading.

### 2.5 Initial-condition blind spots (events that silently cannot fire)

Because every event is one-sided, an event whose `g` already has the "post-crossing" sign at the phase's first step can never fire:

* `max_radius(stop_r)`, `direction=+1`: if `R2 >= stop_r` at phase entry, it needs `R2` to first drop below `stop_r` and climb back. See `S11-A-07` for the concrete case where this is guaranteed.
* `min_radius`, `direction=-1`: if `R2 < min_r` at entry (i.e. `r0 < max(1.5*coll_r, 0.01)`), the shell is already "collapsed" but nothing fires.
* `cloud_boundary`, `direction=+1`: if `R2 >= rCloud` at entry, phase 1a runs to `t_end` with no boundary event.
* `energy_floor`, `direction=-1`: if `Eb <= 1e3` (AU) at the start of phase 1c, no event.

(`S11-A-22`, one finding covering the class.)

### 2.6 The `stop_r <= rCloud` trap

`_check_stop_r_rCloud_interaction` (main.py:41) emits a **warning** (main.py:60–67) when `stop_r <= rCloud`, whose text is *"stop_r will terminate the run before stop_at_rCloud_nSnap can fire"*. Trace what actually happens:

1. Phase 1a has **no** `stop_r` event (:447–451). `R2` sails past `stop_r` unimpeded.
2. Phase 1a's `cloud_boundary` event is `terminal=True` (:243) → phase 1a stops at `R2 ≈ rCloud`.
3. Phase 1b builds `max_radius(stop_r)` with `direction=+1` (:495, :157). At phase-1b entry `R2 ≈ rCloud >= stop_r`, so `g = R2 - stop_r >= 0` already. Only a positive-going crossing is caught, which now requires `R2` to fall below `stop_r` first.
4. Same for phases 1c and 2.

So `stop_r <= rCloud` makes `stop_r` **inert for the whole run** — the exact opposite of the warning. The `info` branch (main.py:69–76, `stop_r <= 1.5 * rCloud`) covers `rCloud < stop_r <= 1.5 rCloud`, where `stop_r` does work. (`S11-A-07`)

---

## 3. Termination and end-code handling

### 3.1 Every way a run or phase can end (as visible in this slice)

| ending | where | `EndSimulationDirectly` | `SimulationEndReason` | `SimulationEndCode` | `isCollapse` |
|---|---|---|---|---|---|
| `min_radius` event | `apply_event_result` :620–629 | `True` | `"Small radius reached (event)"` | `SHELL_COLLAPSED.code` | **True** ('radius' matches) |
| `max_radius`(=`stop_r`) event | same | `True` | `"Large radius reached (event)"` | `LARGE_RADIUS.code` | **True** ← wrong, 'radius' matches |
| `velocity_runaway` event | same | `True` | `"Collapse velocity runaway (event)"` | `VELOCITY_RUNAWAY.code` | **False** ← neither 'radius' nor 'collapse' in `velocity_runaway_event` |
| `cloud_boundary` event | `apply_event_result` early-outs of the ending block (`is_simulation_ending=False` :246) | *unchanged* | *unchanged* | *unchanged* | unchanged |
| `energy_floor` event | same (`False` :281) | unchanged | unchanged | unchanged | unchanged |
| `velocity_sign` event | same (`False` :313) | unchanged | unchanged | unchanged | unchanged |
| `stop_at_rCloud_nSnap == 0` and `R2 >= rCloud` after phase 1a | main.py:264–272 | `True` | `"Reached cloud edge (stop_at_rCloud_nSnap=0)"` | `RCLOUD_BOUNDARY.code` | not set |
| integration reaches `t_end` with no event | `check_event_termination` :407 → `triggered=False` → `apply_event_result` returns at :608–609 | untouched | untouched | untouched | untouched |
| **solver failure** (`sol.status == -1`) | *not detected anywhere in this slice* | untouched | untouched | untouched | untouched |

`SimulationEndReason` is read back at main.py:202 with an `'in params'` guard defaulting to `"Unknown"`, and passed to `params.write_termination_report`.

### 3.2 The `isCollapse` substring heuristic

```python
if 'radius' in result.reason_code.lower() or 'collapse' in result.reason_code.lower():   :627
```
matched against the four ending `reason_code`s:

* `small_radius_event` → matches 'radius' → `isCollapse = True` (correct).
* `large_radius_event` → **also matches 'radius'** → `isCollapse = True` on a run that terminated by **expanding** past `stop_r`. Flatly wrong. (`S11-A-05`)
* `velocity_runaway_event` → matches neither, even for the `direction="collapse"` variant, which is by construction `v2 < -500 pc/Myr` — i.e. the most violent collapse the code can detect is *not* recorded as a collapse. (`S11-A-06`)
* `cloud_boundary` / `energy_floor` / `velocity_sign_change` are non-ending, so the block never runs for them (it is nested inside `if result.is_simulation_ending:` :620).

### 3.3 Is a solver failure distinguishable from a normal completion? **No.**

`check_event_termination` inspects only `sol.t_events` / `sol.y_events`. A `solve_ivp` return with `status == -1` (step-size underflow, RHS blow-up, `LSODA`/`BDF` convergence failure) and empty `t_events` produces **byte-for-byte the same `EventResult`** as a clean run to `t_end`: `triggered=False, name="", index=-1, t=nan, y=array([]), is_simulation_ending=False, reason_code="", reason_message=""` (:379–389 and :407–416 are literally the same construction). `apply_event_result` then returns immediately (:608) and writes nothing. From this slice the caller has no signal. (`S11-A-08`)

At the process level the same collapse happens:

* `run_expansion` returns bare `None` (main.py:363); the return is not captured at main.py:180.
* `write_simulation_end`'s `exit_code` is assigned (main.py:191), logged at DEBUG (main.py:195), set to `99` on exception (main.py:198) — and then **`start_expansion` unconditionally `return 0`** (main.py:211). The `99` failure code is unreachable to any caller. (`S11-A-09`)
* Three broad `except Exception` blocks (main.py:196, 204, 358) downgrade end-report failure, termination-report failure and **`params.flush()` failure** (i.e. output never written to disk) to `logger.warning`, after which the run still reports success. (`S11-A-21`)

### 3.4 Fail-dangerous getattr default

`is_simulation_ending=getattr(event, 'is_simulation_ending', True)` (:401) defaults a missing attribute to **True** — an event object that forgets the flag terminates the whole simulation. Contrast :404, where `end_code` defaults to `None` (safe). All nine events in this file set the flag, so this is latent. (`S11-A-15`)

---

## 4. State handed between phases

### 4.1 What is written and read, in order

**Before phase 1a** (`run_expansion`, main.py:232–238): `get_y0` returns `(t0, r0, v0, E0, T0)`, written in this order into `params['t_now'], ['R2'], ['v2'], ['Eb'], ['T0']`. Nothing else is initialised here — in particular `EndSimulationDirectly`, `isCollapse`, `SimulationEndReason`, `SimulationEndCode` are **not** initialised by `run_expansion`; it *reads* `EndSimulationDirectly` at :283 assuming someone else set it.

**During each phase** the phase runner (outside slice) calls `apply_event_result` (:588). What that writes, in order:

1. `params['t_now'].value = t` — the **caller-supplied `t`**, not `result.t` (:612).
2. `params[state_keys[i]].value = float(y[i])` for `i < len(y)` and `key in params` (:615–617) — again the **caller-supplied `y`**, not `result.y`.
3. Only if `result.is_simulation_ending`: `SimulationEndReason`, `SimulationEndCode`, `EndSimulationDirectly=True`, plus the `isCollapse` heuristic (:620–629).

So `EventResult.t` and `EventResult.y` — carefully extracted as `t_ev[0]` and `y_ev[0].copy()` at :399–400 — are **computed and then never used** by the only consumer in this slice. If the caller passes `sol.t[-1]`/`sol.y[:,-1]`, the recorded state is the *end of integration*, not the *event*; combined with the non-terminal `velocity_sign` shadow (§2.4) the two can be far apart. (`S11-A-12`)

`state_keys` defaults to `['R2', 'v2']` (:589). In phase 1c the state vector is `[R2, v2, Eb]` (the energy-floor event reads `y[2]`, :532/:275). Unless the caller passes a three-element `state_keys`, **`Eb` from the event state is silently dropped** — the loop guard `i < len(y)` (:616) only protects the other direction. (`S11-A-14`)

**Between phases** (main.py): `current_phase` is written at :244/278/301/327 and read nowhere in this file. `EndSimulationDirectly` is the only value read across phase boundaries (:283, :303, :343). `params.reset_keys(COOLING_PHASE_KEYS)` at :317 is the only explicit reset — it fires **unconditionally**, including when the run already ended in phase 1a, and it fires **before** main.py:337–338 read `Lmech_total` and `v_mech_total` for the log-only `pRam`. Whether those two keys are in `COOLING_PHASE_KEYS` I cannot tell (`_input/dictionary.py` is outside my slice), so I flag only the ordering. (`S11-A-24`)

### 4.2 Aliasing / staleness

* Event closures capture `rCloud`, `coll_r`, `stop_r`, `min_r`, `energy_floor`, `v_max`, `Lgain`, `Lloss` **by value at build time** (:442–451, :482–497, :526–539, :565–577, :341). Any later change to those `params` entries is invisible to an already-built event.
* `y_ev[0].copy()` (:400) is the one deliberate de-aliasing in the file — `EventResult.y` does not alias the solver's buffer. Ironically it is the value nobody uses.
* `float(y[i])` (:617) de-aliases the scalars written into `params`.
* `EventResult` is a plain (non-frozen) dataclass; `y: np.ndarray` is a mutable field, but no code mutates it.

### 4.3 Module-level / global mutable state persisting across phases

| object | line | mutable? | leaks across phases? | leaks across runs in one process? |
|---|---|---|---|---|
| `params` (the `DescribedDict`) | threaded everywhere | **yes** | **yes — it is the entire carrier** | yes if the same object is reused |
| `state_keys=['R2','v2']` default arg | phase_events.py:589 | **yes** (shared list literal, evaluated once at import) | yes | yes — a caller that mutates it poisons every later call |
| `logger` | main.py:32, phase_events.py:63 | no (module logger) | — | — |
| root logging config | main.py:104–108 | global process state | yes | second call skips `basicConfig` because handlers now exist |
| `MIN_RADIUS_*`, `MAX_VELOCITY_*` | phase_events.py:71–74 | immutable floats | — | — |

**Second run in the same process.** `run_expansion` never resets `EndSimulationDirectly`, `SimulationEndReason`, `SimulationEndCode`, `isCollapse` or `current_phase`. If the same `params` object is handed to `run_expansion` twice — or if `params` is rebuilt but any of these keys carries a default derived from a prior run — the second run executes phase 1a and then, at main.py:283, sees the **first run's** `EndSimulationDirectly=True` and silently skips phases 1b, 1c and 2 while still printing all three "Entering …" banners. Likewise a stale `isCollapse=True` is never cleared. (`S11-A-11`)

---

## 5. Dimensions

Astro units throughout ("AU" in this codebase = **Msun · pc · Myr**, per `unit_conversions.py`).

| quantity | symbol / slot | unit |
|---|---|---|
| integrator independent variable | `t`, `params['t_now']`, `t0`, `EventResult.t` | **Myr** (astro time unit; `s2Myr` is the base time conversion) |
| shell radius | `y[0]`, `R2`, `r0`, `rCloud`, `coll_r`, `stop_r`, `min_r`, `max_r` | **pc** (asserted by main.py:63–64, :123, phase_events.py:454, :542) |
| shell velocity | `y[1]`, `v2`, `v0` | **pc/Myr** (main.py:258); 1 pc/Myr ≈ 0.978 km/s |
| bubble energy | `y[2]`, `Eb`, `E0`, `energy_floor` | **Msun·pc²/Myr²**; ×`cvt.E_au2cgs` → erg (main.py:332) |
| temperature | `T0` | K (no conversion applied) |
| number density | `nCore` | AU number density; ×`cvt.ndens_au2cgs` → cm⁻³ (main.py:124) |
| ram pressure | `P_ram_bnd` | AU pressure; ×`cvt.Pb_au2_KcmInv` → K cm⁻³ i.e. P/k_B (main.py:340) |
| mechanical luminosity / velocity | `Lmech_total`, `v_mech_total` | AU (Msun·pc²/Myr³ and pc/Myr) |
| SPS mass scaling | `f_mass = mCluster / sps_refmass` | dimensionless ratio |
| cooling table | `logT`, `logLambda` | log10 K, log10 (erg cm³ s⁻¹) — CGS, **not** converted; `interp1d` built with `kind='linear'` and default `bounds_error=True` (main.py:167) |
| durations | `phase1a_elapsed` … | `datetime.timedelta` (wall clock — unrelated to sim time) |

Unit sanity of the two velocity constants: `MAX_VELOCITY_COLLAPSE = 500.0` pc/Myr ≈ **489 km/s**; `MAX_VELOCITY_EXPANSION = 1000.0` pc/Myr ≈ **978 km/s**. `energy_floor = 1e3` AU ≈ **1.9 × 10⁴⁶ erg** (1 Msun·pc²/Myr² ≈ 1.9 × 10⁴³ erg) — roughly 2 × 10⁻⁵ of a 10⁵¹ erg bubble.

One dimensional oddity in main.py:144: `f_mass = params['mCluster'] / params['sps_refmass']` divides two **`DescribedItem` objects**, not their `.value`s — every other access in the file uses `.value`. It is then `f"{f_mass:.4f}"`-formatted (:145) and passed to `read_sps.read_sps` (:147). This only works if `DescribedItem` implements `__truediv__` and `__format__`; I cannot verify that from my slice. (`S11-A-23`, low confidence.)

---

## 6. Numeric literals

### 6.1 `trinity/main.py`

| line | literal | expression it sits in |
|---|---|---|
| 38 | `1.5` | `_STOP_R_RCLOUD_RACE_FACTOR = 1.5` |
| 60 | *(none)* | `if stop_r <= rCloud:` |
| 69 | via `1.5` | `if stop_r <= _STOP_R_RCLOUD_RACE_FACTOR * rCloud:` |
| 106 | `logging.DEBUG` | `level=logging.DEBUG` (forces DEBUG on the whole process) |
| 112 | `16`, `15` | `"=" * 16 + " TRINITY … " + "=" * 15` |
| 178 | `5` | `"=" * 5` |
| 185 | `16`, `15` | as :112 |
| 198 | `99` | `exit_code = 99` (failure code — never returned) |
| 211 | `0` | `return 0` (unconditional success) |
| 246, 280, 298, 324 | `5` | `"-" * 5` banners |
| 264 | `0` | `nSnap_rCloud == 0` |
| 265 | *(none)* | `params['R2'].value >= params['rCloud'].value` |
| 283, 303, 343 | `False` | `if params['EndSimulationDirectly'].value == False:` (identity-style compare, not `not …`) |
| 123,145 / 124,332,340 / 257,258 / 173 | `.4f` / `.4e` / `.6e` / `.1f` | format precisions only |

### 6.2 `trinity/phase_general/phase_events.py`

| line | literal | expression it sits in |
|---|---|---|
| 71 | `0.01` | `MIN_RADIUS_SAFETY = 0.01` → floor of `min_r` (pc) |
| 72 | `1.5` | `MIN_RADIUS_FACTOR = 1.5` → `min_r = max(coll_r * 1.5, 0.01)` |
| 73 | `500.0` | `MAX_VELOCITY_COLLAPSE = 500.0` (pc/Myr) |
| 74 | `1000.0` | `MAX_VELOCITY_EXPANSION = 1000.0` — **never referenced** anywhere |
| 121–122 | `y[0]` | `return y[0] - min_r` |
| 124–125 | `True`, `-1` | `terminal`, `direction` |
| 152–157 | `y[0]`, `True`, `1` | `return y[0] - max_r`; `direction = 1` |
| 166 | `MAX_VELOCITY_COLLAPSE` | `v_max: float = MAX_VELOCITY_COLLAPSE` default |
| 191–193 | `y[1]`, `-1` | `return v2 + v_max` |
| 196–199 | `y[1]`, `-1` | `return v_max - v2` |
| 202–205 | `y[1]`, `-1` | `return v_max - abs(v2)` |
| 208–210 | `True`, `True` | `terminal`, `is_simulation_ending` |
| 239–246 | `y[0]`, `True`, `1`, `False` | cloud boundary |
| 252 | `2` | `y_index: int = 2` (energy slot) |
| 274–281 | `y[y_index]`, `True`, `-1`, `False` | `return Eb - energy_floor` |
| 287 | `1` | `y_index: int = 1` (velocity slot) |
| 306–313 | `False`, `-1`, `False` | `terminal = False`, `direction = -1` |
| 319 | `0.05` | `threshold: float = 0.05` |
| 343–344 | `0`, `1.0` | `if Lgain <= 0: return 1.0` |
| 345–346 | — | `ratio = (Lgain - Lloss) / Lgain`; `return ratio - threshold` |
| 348–351 | `True`, `-1`, `False` | attributes |
| 386, 410 | `-1` | `index=-1` sentinel in the not-triggered results |
| 387, 411 | `np.nan` | `t=np.nan` |
| 393 | `0` | `if len(t_ev) > 0:` ← **the termination test** |
| 399–400 | `[0]`, `[0]` | `float(t_ev[0])`, `y_ev[0].copy()` |
| 401 | `True` | `getattr(event, 'is_simulation_ending', True)` |
| 445, 485, 529, 568 | via `1.5`, `0.01` | `min_r = max(coll_r * MIN_RADIUS_FACTOR, MIN_RADIUS_SAFETY)` |
| 494, 538, 576 | `0` | `if stop_r is not None and stop_r > 0:` |
| 497 | `0.05` | `make_cooling_balance_event(threshold=0.05)` |
| 504 | `1e3` | `energy_floor: float = 1e3` (AU energy) |
| 532 | `2` | `make_energy_floor_event(energy_floor, y_index=2)` |
| 589 | `['R2','v2']` | mutable default argument |
| 616–617 | — | `if i < len(y) and key in params:` / `float(y[i])` |
| 627 | `'radius'`, `'collapse'` | `if 'radius' in reason_code.lower() or 'collapse' in reason_code.lower():` |

---

## 7. Summary of what to flag, by the categories asked for

* **An event that cannot fire:** `cooling_balance` (constant in `t` and `y`) — `S11-A-04`. Plus the whole initial-condition class in §2.5 — `S11-A-22`. Plus `max_radius(stop_r)` guaranteed inert whenever `stop_r <= rCloud` — `S11-A-07`.
* **An event whose sign convention makes the crossing undetectable:** none. Every `g`/`direction` pair is internally consistent.
* **Termination inferred without checking terminality:** `check_event_termination` :393 — `S11-A-01`; concretely harmful via `velocity_sign` at index 0 of the implicit phase — `S11-A-02`; compounded by lowest-index-wins rather than earliest-time-wins — `S11-A-03`.
* **A failure path indistinguishable from success:** `check_event_termination` ignores `sol.status` — `S11-A-08`; `start_expansion` always `return 0` and discards `exit_code`/`99` — `S11-A-09`; three swallowing `except Exception` blocks incl. `params.flush()` — `S11-A-21`.
* **Global state leaking between phases:** `params` itself plus the never-reset end flags — `S11-A-11`; mutable default `state_keys` — `S11-A-13`.
* **A transition condition never satisfiable / satisfiable at t=0:** the `EndSimulationDirectly` gate is satisfiable at entry (phase 1a is ungated but 1b/1c/2 would be skipped) — folded into `S11-A-11`. `stop_at_rCloud_nSnap == 0` needs `R2 >= rCloud` immediately after a root-solved boundary event — float-fragile, `S11-A-19`.
* **Unreachable branches:** `make_velocity_runaway_event` `"expansion"` and `else` branches, and `MAX_VELOCITY_EXPANSION` — `S11-A-16`; `expansion_next` — `S11-A-17`; `P_ram_bnd` computed and discarded — `S11-A-18`.
* **Return values ignored by their caller:** `write_simulation_end` → `exit_code` — `S11-A-09`; `run_expansion`'s return — same; `EventResult.t` / `EventResult.y` never used by `apply_event_result` — `S11-A-12`; `EventResult.index` / `.name` used only for logging.

Severity key I used: **S1** = silently wrong scientific output on an ordinary run; **S2** = major logic defect / silent failure; **S3** = conditional or moderate; **S4** = minor, cosmetic, dead code.

```json
[
  {
    "id": "S11-A-01",
    "file": "trinity/phase_general/phase_events.py",
    "line": 393,
    "class": "state",
    "severity": "S2",
    "claim": "check_event_termination decides an event ended a phase purely from `len(t_ev) > 0` and never inspects the event's `terminal` attribute.",
    "evidence": "Line 392-405: `for i, (t_ev, y_ev) in enumerate(zip(sol.t_events, sol.y_events)): if len(t_ev) > 0: ... return EventResult(triggered=True, ...)`. The token `terminal` appears in this file only as an assignment (lines 124, 156, 208, 243, 278, 310, 348) and is never read. scipy populates t_events[i] for any detected crossing, terminal or not.",
    "expected": "The decision should consult `getattr(events[i], 'terminal', False)` (and/or `sol.status == 1`) before treating a recorded root as the cause of phase termination.",
    "failure_scenario": "Any non-terminal event with a root is reported as the phase-ending event, at that root's time, while integration in fact continued past it to t_end or to a different terminal event.",
    "repro": "Build a phase with events=[non_terminal_event, terminal_event]; run solve_ivp so both record roots; check_event_termination returns the non-terminal one.",
    "confidence": "high"
  },
  {
    "id": "S11-A-02",
    "file": "trinity/phase_general/phase_events.py",
    "line": 488,
    "class": "state",
    "severity": "S1",
    "claim": "The only terminal=False event (velocity_sign) is placed at index 0 of build_implicit_phase_events, so an ordinary v2 zero-crossing masks every genuinely terminal event in phase 1b and reports the phase as ending at the first sign change.",
    "evidence": "make_velocity_sign_event sets `event.terminal = False` (line 310) and `direction = -1` (line 311). build_implicit_phase_events puts it first: `events = [make_velocity_sign_event(), make_min_radius_event(min_r), make_velocity_runaway_event(...)]` (lines 487-491), with max_radius(stop_r) appended at index 3 (line 495). check_event_termination returns on the first index with a root (line 392-405), so index 0 always wins.",
    "expected": "Either velocity_sign should not be consulted as a phase-ending event, or check_event_termination should skip events with terminal=False, or velocity_sign should be ordered last.",
    "failure_scenario": "Phase 1b that actually ran to stop_r (terminal max_radius at index 3) is instead recorded as ending at the earlier v2=0 turnover: t_now and R2/v2 are rewound to that earlier crossing, is_simulation_ending is False so no end code/reason is written, and the run silently continues from a stale state.",
    "repro": "Run param/simple_cluster.param with stop_r set just above rCloud so phase 1b hits max_radius after the v2 turnover; inspect the EventResult name returned by check_event_termination.",
    "confidence": "high"
  },
  {
    "id": "S11-A-03",
    "file": "trinity/phase_general/phase_events.py",
    "line": 392,
    "class": "state",
    "severity": "S2",
    "claim": "When several events record roots, check_event_termination returns the lowest-index event, not the earliest-in-time one.",
    "evidence": "`for i, (t_ev, y_ev) in enumerate(zip(sol.t_events, sol.y_events)): if len(t_ev) > 0: ... return ...` (lines 392-405) iterates in list order and returns on the first hit; t_ev[0] of that event is reported as the phase end time with no comparison against other events' times.",
    "expected": "Select the triggered event with the smallest root time (argmin over t_ev[0] across events with roots).",
    "failure_scenario": "In the transition phase, events are [energy_floor, min_radius, velocity_runaway, max_radius]; if min_radius fires physically before energy_floor records a root, energy_floor is still reported, with the wrong end time, wrong end reason, and is_simulation_ending=False instead of True.",
    "repro": "Construct a solve_ivp result where events[0] has a root at t=5 and events[1] at t=1; check_event_termination reports events[0].",
    "confidence": "high"
  },
  {
    "id": "S11-A-04",
    "file": "trinity/phase_general/phase_events.py",
    "line": 342,
    "class": "deadcode",
    "severity": "S2",
    "claim": "The cooling-balance event value is constant in both t and y, so it can never produce a zero crossing and can never fire under solve_ivp.",
    "evidence": "`def event(t, y): if Lgain <= 0: return 1.0; ratio = (Lgain - Lloss) / Lgain; return ratio - threshold` (lines 342-346) reads neither t nor y; Lgain and Lloss are scalars closed over at factory-call time (line 341). It nevertheless declares terminal=True (348) and direction=-1 (349).",
    "expected": "The event should be a function of the integrated state (e.g. recompute Lgain/Lloss from y), or the cooling-balance test should not be expressed as an ODE event at all.",
    "failure_scenario": "Phase 1b never terminates on cooling balance; the intended energy->transition handoff criterion is silently never evaluated by the integrator and the phase runs to some other event or t_end.",
    "repro": "factory = make_cooling_balance_event()(Lgain=1.0, Lloss=0.5); evaluate the returned event at many (t,y) - the value is identical everywhere.",
    "confidence": "medium"
  },
  {
    "id": "S11-A-05",
    "file": "trinity/phase_general/phase_events.py",
    "line": 627,
    "class": "sign",
    "severity": "S2",
    "claim": "apply_event_result sets isCollapse=True when a run terminates by EXPANDING past stop_r, because the substring test matches 'radius' in 'large_radius_event'.",
    "evidence": "Line 627: `if 'radius' in result.reason_code.lower() or 'collapse' in result.reason_code.lower(): params['isCollapse'].value = True`. make_max_radius_event sets reason_code = 'large_radius_event' (line 160) with end_code LARGE_RADIUS (162) and direction=+1 (157), i.e. R2 rising through max_r.",
    "expected": "isCollapse should be driven by the event identity or end_code (SHELL_COLLAPSED / VELOCITY_RUNAWAY-collapse), not by a substring of the reason_code string.",
    "failure_scenario": "Every run that terminates at stop_r is recorded and downstream-classified as a collapsed shell, inverting the physical outcome in the output metadata.",
    "repro": "Set stop_r above rCloud in a param file so max_radius fires; inspect params['isCollapse'] after the run.",
    "confidence": "high"
  },
  {
    "id": "S11-A-06",
    "file": "trinity/phase_general/phase_events.py",
    "line": 211,
    "class": "sign",
    "severity": "S3",
    "claim": "A collapse-direction velocity runaway does NOT set isCollapse, because its reason_code contains neither 'radius' nor 'collapse'.",
    "evidence": "make_velocity_runaway_event sets `event.reason_code = 'velocity_runaway_event'` (line 211) for all three direction variants, including the collapse variant whose event is `v2 + v_max` with direction=-1 (lines 190-193), i.e. v2 falling below -500 pc/Myr. The test at line 627 therefore fails to match.",
    "expected": "The collapse variant should carry a reason_code that the isCollapse test recognises (or the test should key off end_code / direction).",
    "failure_scenario": "A run that terminates on runaway inward collapse is not flagged as a collapse in the output metadata, while a run that terminates by expansion (S11-A-05) is.",
    "repro": "Force a run into v2 < -500 pc/Myr; inspect params['isCollapse'].",
    "confidence": "high"
  },
  {
    "id": "S11-A-07",
    "file": "trinity/main.py",
    "line": 60,
    "class": "regime",
    "severity": "S2",
    "claim": "When stop_r <= rCloud the stop_r event becomes permanently unfireable, which is the opposite of the warning the code emits.",
    "evidence": "build_energy_phase_events (phase_events.py:447-451) contains no max_radius event, so phase 1a ignores stop_r; phase 1a's cloud_boundary event is terminal=True (phase_events.py:243) so phase 1a stops at R2 ~= rCloud; phases 1b/1c/2 then build make_max_radius_event(stop_r) with direction=+1 (phase_events.py:157), which requires a negative-to-positive crossing of R2-stop_r, but R2 >= rCloud >= stop_r already at entry. main.py:60-67 nevertheless warns 'stop_r will terminate the run before stop_at_rCloud_nSnap can fire'.",
    "expected": "Either add a stop_r event to the energy phase, or make max_radius fire on an already-exceeded threshold (direction=0 / an explicit pre-check), or correct the diagnostic to say stop_r <= rCloud disables stop_r.",
    "failure_scenario": "A user sets stop_r below rCloud expecting an early stop, sees a warning confirming an early stop will happen, and instead gets a full-length run with stop_r having no effect at all.",
    "repro": "Set stop_r < rCloud in a .param and run; observe the warning at main.py:60 and that no large_radius_event ever fires.",
    "confidence": "medium"
  },
  {
    "id": "S11-A-08",
    "file": "trinity/phase_general/phase_events.py",
    "line": 407,
    "class": "silent-failure",
    "severity": "S2",
    "claim": "A solve_ivp failure is indistinguishable from a clean completion: check_event_termination never inspects sol.status / sol.success / sol.message and returns an identical not-triggered EventResult in both cases.",
    "evidence": "The function reads only sol.t_events and sol.y_events (lines 379, 392). The 'no events' result (lines 379-389) and the 'ran to t_end' result (lines 407-416) are the same construction: triggered=False, name='', index=-1, t=np.nan, y=np.array([]), is_simulation_ending=False, reason_code='', reason_message=''. apply_event_result then returns immediately at line 608-609 and writes nothing.",
    "expected": "Propagate sol.status/sol.success into EventResult (or raise) so the caller can distinguish integration failure from normal completion.",
    "failure_scenario": "A stiff configuration where the integrator aborts with status=-1 mid-phase is treated as a phase that simply ran to its end; the next phase starts from the last successful state and the run reports success with no diagnostic.",
    "repro": "Pass a solve_ivp result with status=-1 and empty t_events to check_event_termination; compare the returned EventResult to one from a successful run with no events.",
    "confidence": "high"
  },
  {
    "id": "S11-A-09",
    "file": "trinity/main.py",
    "line": 211,
    "class": "silent-failure",
    "severity": "S2",
    "claim": "start_expansion always returns 0; the exit_code from write_simulation_end — including the 99 assigned on exception — is computed, logged at DEBUG, and then discarded.",
    "evidence": "Line 191 `exit_code = write_simulation_end(params)`; line 195 logs it at DEBUG; line 198 `exit_code = 99` in the except branch; line 211 `return 0` unconditionally. run_expansion likewise returns bare None (line 363) and its return is not captured at line 180.",
    "expected": "`return exit_code` (or otherwise surface a non-zero status) so a caller / shell / sweep driver can detect the failure.",
    "failure_scenario": "In a parameter sweep, runs whose end report failed to write are indistinguishable from successful runs at the process-exit level; the sweep records them all as successes.",
    "repro": "Make write_simulation_end raise; observe the warning at main.py:197 and that start_expansion still returns 0.",
    "confidence": "high"
  },
  {
    "id": "S11-A-10",
    "file": "trinity/main.py",
    "line": 278,
    "class": "state",
    "severity": "S3",
    "claim": "current_phase is advanced and the phase banner printed BEFORE the EndSimulationDirectly gate, so a run that ended in phase 1a still reports current_phase='momentum' and prints the 1b/1c/2 'Entering ...' banners.",
    "evidence": "Lines 278-283: `params['current_phase'].value = 'implicit'` then the logger/terminal_prints.phase banner, and only then `if params['EndSimulationDirectly'].value == False:`. Identical ordering at 298-303 ('transition') and 324-343 ('momentum').",
    "expected": "Set current_phase and emit the banner inside the gated branch, so the recorded phase reflects where the run actually stopped.",
    "failure_scenario": "Output metadata and terminal transcript claim the run reached the momentum phase when it terminated during the energy phase; any downstream analysis keyed on current_phase mis-classifies the run.",
    "repro": "Run with stop_at_rCloud_nSnap=0 and a cloud small enough that phase 1a reaches rCloud; inspect params['current_phase'] afterwards.",
    "confidence": "high"
  },
  {
    "id": "S11-A-11",
    "file": "trinity/main.py",
    "line": 216,
    "class": "state",
    "severity": "S3",
    "claim": "run_expansion never resets EndSimulationDirectly, SimulationEndReason, SimulationEndCode, isCollapse or current_phase; a second run against the same params object in the same process inherits the first run's termination flags.",
    "evidence": "Lines 232-244 initialise only t_now, R2, v2, Eb, T0 and current_phase. EndSimulationDirectly is first READ at line 283 without ever being written in this function except inside the nSnap==0 branch (line 266) and by apply_event_result (phase_events.py:624). params.reset_keys(COOLING_PHASE_KEYS) at line 317 is the only reset and does not cover these keys by name.",
    "expected": "Reset the termination/classification flags at the top of run_expansion alongside the state initialisation.",
    "failure_scenario": "A driver that reuses a params object (e.g. an in-process sweep) gets a second run that executes phase 1a, then skips phases 1b/1c/2 outright while printing all their banners, and reports the first run's SimulationEndReason.",
    "repro": "Call run_expansion(params) twice on the same params where the first run sets EndSimulationDirectly=True; the second skips 1b/1c/2.",
    "confidence": "high"
  },
  {
    "id": "S11-A-12",
    "file": "trinity/phase_general/phase_events.py",
    "line": 612,
    "class": "state",
    "severity": "S3",
    "claim": "apply_event_result writes the caller-supplied t and y into params and ignores result.t / result.y, so the carefully extracted event root time and state are never used.",
    "evidence": "Line 612 `params['t_now'].value = t` and lines 615-617 `params[key].value = float(y[i])` use the function's t/y arguments; result.t (set from float(t_ev[0]) at line 399) and result.y (from y_ev[0].copy() at line 400) are read nowhere in the function.",
    "expected": "Use result.t / result.y when result.triggered, or drop those fields from EventResult.",
    "failure_scenario": "If the caller passes sol.t[-1]/sol.y[:,-1] rather than the event root, the recorded phase-boundary state is the end of integration, not the event; combined with S11-A-02 the two can differ by the whole span from the v2 turnover to t_end.",
    "repro": "Call apply_event_result(params, result, t=999.0, y=[1,2]) with a result whose .t is 1.0; params['t_now'] becomes 999.0.",
    "confidence": "high"
  },
  {
    "id": "S11-A-13",
    "file": "trinity/phase_general/phase_events.py",
    "line": 589,
    "class": "state",
    "severity": "S4",
    "claim": "apply_event_result uses a mutable default argument `state_keys: List[str] = ['R2', 'v2']`, a single list object shared by every call for the process lifetime.",
    "evidence": "`def apply_event_result(params, result: EventResult, t: float, y: np.ndarray, state_keys: List[str] = ['R2', 'v2']) -> None:` at lines 588-589. The list is evaluated once at import.",
    "expected": "`state_keys: Optional[List[str]] = None` with `state_keys = state_keys or ['R2','v2']` inside, or a tuple default.",
    "failure_scenario": "Any caller that mutates the passed-in list (append/sort) permanently changes the default for every subsequent call in the process, including in later phases and later runs.",
    "repro": "Call apply_event_result once with the default, then `apply_event_result.__defaults__[-1].append('Eb')`; all later default calls now write Eb.",
    "confidence": "high"
  },
  {
    "id": "S11-A-14",
    "file": "trinity/phase_general/phase_events.py",
    "line": 589,
    "class": "state",
    "severity": "S3",
    "claim": "With the default state_keys=['R2','v2'], the third state component Eb (y[2]) is silently dropped when applying an event result in the transition phase.",
    "evidence": "Default state_keys has two entries (line 589) while build_transition_phase_events uses make_energy_floor_event(energy_floor, y_index=2) (line 532), i.e. a 3-component state [R2, v2, Eb]. The write loop `for i, key in enumerate(state_keys): if i < len(y) and key in params:` (lines 615-617) is bounded by len(state_keys), so y[2] is never written.",
    "expected": "Pass state_keys=['R2','v2','Eb'] for the transition phase, or derive the key list from the phase's state vector.",
    "failure_scenario": "The transition phase terminates on the energy_floor event but params['Eb'] retains its pre-event value; the momentum phase then starts from an inconsistent (R2, v2, Eb) triple.",
    "repro": "Call apply_event_result with y = np.array([r, v, E]) and the default state_keys; params['Eb'] is unchanged.",
    "confidence": "medium"
  },
  {
    "id": "S11-A-15",
    "file": "trinity/phase_general/phase_events.py",
    "line": 401,
    "class": "silent-failure",
    "severity": "S4",
    "claim": "check_event_termination defaults a missing `is_simulation_ending` attribute to True, so an event that forgets the flag terminates the whole simulation.",
    "evidence": "Line 401: `is_simulation_ending=getattr(event, 'is_simulation_ending', True)`. Contrast line 404, where end_code defaults to the safe None.",
    "expected": "Default to False (fail-safe) or raise on a missing attribute.",
    "failure_scenario": "A future event factory that omits the flag silently ends every run at its first crossing rather than merely ending the phase.",
    "repro": "Pass a plain lambda (no attributes) as an event with a root; the returned EventResult has is_simulation_ending=True.",
    "confidence": "high"
  },
  {
    "id": "S11-A-16",
    "file": "trinity/phase_general/phase_events.py",
    "line": 74,
    "class": "deadcode",
    "severity": "S4",
    "claim": "MAX_VELOCITY_EXPANSION is never referenced, and neither the 'expansion' nor the fallback branch of make_velocity_runaway_event is reachable from any builder.",
    "evidence": "`MAX_VELOCITY_EXPANSION = 1000.0` (line 74) appears nowhere else. All four builders call make_velocity_runaway_event(MAX_VELOCITY_COLLAPSE, direction='collapse') (lines 450, 490, 534, 572), so the elif at line 195 and the else at line 201 never execute. The else branch also silently accepts any misspelled direction string instead of raising.",
    "expected": "Remove the unused constant and branches, or wire them up; validate the `direction` argument against the two supported values.",
    "failure_scenario": "A caller that typos direction='colapse' silently gets the abs()-based event with different semantics and no error.",
    "repro": "grep MAX_VELOCITY_EXPANSION across the package; only the definition is found.",
    "confidence": "high"
  },
  {
    "id": "S11-A-17",
    "file": "trinity/main.py",
    "line": 366,
    "class": "deadcode",
    "severity": "S4",
    "claim": "expansion_next is a dead stub: it takes seven parameters, uses none, and immediately returns None.",
    "evidence": "Lines 366-368: `def expansion_next(tStart, ODEpar, sps_data_old, sps_f_old, mypath, cloudypath, ii_coll): return`.",
    "expected": "Delete it (per project rules, git mv to docs/dev/to-be-removed/) or implement it.",
    "failure_scenario": "Any caller believing it advances the simulation gets a silent no-op.",
    "repro": "",
    "confidence": "high"
  },
  {
    "id": "S11-A-18",
    "file": "trinity/main.py",
    "line": 339,
    "class": "deadcode",
    "severity": "S4",
    "claim": "P_ram_bnd is computed unconditionally at the transition->momentum boundary, logged, and then discarded; the log text claims the momentum phase will use it, but nothing stores or passes it.",
    "evidence": "Lines 336-341 compute R2_bnd, Lmech_bnd, v_mech_bnd and `P_ram_bnd = get_bubbleParams.pRam(R2_bnd, Lmech_bnd, v_mech_bnd)`, then log `'(momentum phase will use this)'`. P_ram_bnd is never assigned into params and never referenced again. The block runs even when EndSimulationDirectly is True (the gate is at line 343) and after params.reset_keys(COOLING_PHASE_KEYS) at line 317.",
    "expected": "Either store the value into params for the momentum phase, or move the diagnostic inside the gated branch and correct the message.",
    "failure_scenario": "A reader trusts the log line and assumes the boundary ram pressure was handed to the momentum solver; also, if Lmech_total/v_mech_total are among COOLING_PHASE_KEYS, the logged number is computed from freshly reset values.",
    "repro": "",
    "confidence": "high"
  },
  {
    "id": "S11-A-19",
    "file": "trinity/main.py",
    "line": 265,
    "class": "numerical",
    "severity": "S3",
    "claim": "The stop_at_rCloud_nSnap==0 check uses a bare `R2 >= rCloud` immediately after a root-solved terminal cloud_boundary event, where R2 lands on either side of rCloud within solver tolerance.",
    "evidence": "main.py:264-265 `if (nSnap_rCloud is not None and nSnap_rCloud == 0 and params['R2'].value >= params['rCloud'].value):`. The value of R2 comes from make_cloud_boundary_event's root (phase_events.py:239-244), located by scipy's bracketed root finder, whose returned R2 satisfies R2 - rCloud ~= 0 with an arbitrary sign.",
    "expected": "Compare with a tolerance, e.g. R2 >= rCloud * (1 - 1e-10), or key off the cloud_boundary event having fired rather than re-testing the radius.",
    "failure_scenario": "With stop_at_rCloud_nSnap=0, the run intermittently fails to stop at the cloud edge (R2 comes back a few ULP below rCloud), continues into phases 1b/1c/2, and produces a completely different trajectory from an otherwise identical run.",
    "repro": "Run the same config twice with slightly different rtol so the root lands on either side of rCloud; observe whether EndSimulationDirectly is set.",
    "confidence": "medium"
  },
  {
    "id": "S11-A-20",
    "file": "trinity/phase_general/phase_events.py",
    "line": 447,
    "class": "regime",
    "severity": "S3",
    "claim": "build_energy_phase_events is the only builder that omits the stop_r max-radius event, so stop_r cannot terminate phase 1a.",
    "evidence": "Lines 447-451 build exactly [cloud_boundary, min_radius, velocity_runaway] and never read params['stop_r']; the other three builders all append make_max_radius_event(stop_r) under `if stop_r is not None and stop_r > 0` (lines 494-495, 538-539, 576-577).",
    "expected": "Either add the stop_r event to the energy phase for consistency, or document that stop_r is only honoured from phase 1b onward.",
    "failure_scenario": "A stop_r that the shell would cross during phase 1a is ignored; combined with S11-A-07 it is then unreachable for the rest of the run.",
    "repro": "Set stop_r to a value the shell crosses during phase 1a and observe that no large_radius_event fires.",
    "confidence": "high"
  },
  {
    "id": "S11-A-21",
    "file": "trinity/main.py",
    "line": 358,
    "class": "silent-failure",
    "severity": "S3",
    "claim": "Three broad `except Exception` blocks downgrade end-report, termination-report and params.flush() failures to warnings, after which the run still reports success.",
    "evidence": "main.py:196-198 (write_simulation_end), 204-205 (write_termination_report), 358-359 (`except Exception as e: logger.warning(f\"Could not flush parameters: {e}\")`). Line 361 then logs 'All expansion phases complete' and line 211 returns 0.",
    "expected": "At minimum, record the failure in the run's end state / exit code; a failed flush means simulation output was not persisted.",
    "failure_scenario": "A disk-full or permissions error during params.flush() loses the entire run's output while the process exits 0 and the log says the run completed.",
    "repro": "Make params.flush() raise; observe the warning and the 0 exit.",
    "confidence": "high"
  },
  {
    "id": "S11-A-22",
    "file": "trinity/phase_general/phase_events.py",
    "line": 157,
    "class": "regime",
    "severity": "S3",
    "claim": "Every event is one-sided (direction=+1 or -1), so an event whose guard function already has its post-crossing sign at the phase's first step can never fire, with no warning.",
    "evidence": "direction=+1 at lines 157 (max_radius) and 244 (cloud_boundary); direction=-1 at 125 (min_radius), 193/199/205 (velocity_runaway), 279 (energy_floor), 311 (velocity_sign), 349 (cooling_balance). No builder checks the sign of g at the initial state before installing the event.",
    "expected": "Check g(t0, y0) at build time and either raise, log, or terminate immediately when the threshold is already breached.",
    "failure_scenario": "min_radius: r0 < max(1.5*coll_r, 0.01) pc means the shell starts inside the collapse radius but never triggers SHELL_COLLAPSED. cloud_boundary: r0 >= rCloud means phase 1a runs to t_end with no boundary handoff. energy_floor: Eb <= 1e3 AU at the start of phase 1c means the transition phase never ends on energy.",
    "repro": "Set stop_r below the phase-entry R2 and confirm max_radius never fires; likewise set coll_r so that 1.5*coll_r > r0 and confirm min_radius never fires.",
    "confidence": "medium"
  },
  {
    "id": "S11-A-23",
    "file": "trinity/main.py",
    "line": 144,
    "class": "units",
    "severity": "S4",
    "claim": "f_mass is computed by dividing two DescribedItem objects rather than their .value fields, unlike every other parameter access in the file.",
    "evidence": "Line 144: `f_mass = params['mCluster'] / params['sps_refmass']` (no `.value` on either side), then formatted as a float at line 145 (`f\"{f_mass:.4f}\"`) and passed to read_sps.read_sps at line 147. Every neighbouring access uses `.value` (lines 123, 124, 129-131, 162, ...).",
    "expected": "`params['mCluster'].value / params['sps_refmass'].value`, unless DescribedItem deliberately implements __truediv__ and __format__.",
    "failure_scenario": "If DescribedItem does not implement __truediv__/__format__ this raises TypeError at startup; if it implements __truediv__ returning a DescribedItem, read_sps receives a wrapper rather than a float and any unit metadata attached to it is silently carried into the SPS scaling.",
    "repro": "Check whether trinity/_input/dictionary.py's DescribedItem defines __truediv__ and __format__.",
    "confidence": "low"
  },
  {
    "id": "S11-A-24",
    "file": "trinity/main.py",
    "line": 317,
    "class": "state",
    "severity": "S4",
    "claim": "params.reset_keys(COOLING_PHASE_KEYS) runs unconditionally, including when the run already terminated in phase 1a, and it runs before the Lmech_total / v_mech_total reads used for the boundary pRam log.",
    "evidence": "Line 317 `params.reset_keys(COOLING_PHASE_KEYS)` sits outside any EndSimulationDirectly gate (the gates are at 283, 303, 343). Lines 337-338 subsequently read params['Lmech_total'].value and params['v_mech_total'].value.",
    "expected": "Gate the reset on the run still being alive, and/or perform the boundary diagnostic before the reset.",
    "failure_scenario": "Cooling-phase diagnostics from the phase where the run actually ended are wiped before the end report is written; and if Lmech_total/v_mech_total are members of COOLING_PHASE_KEYS the logged boundary ram pressure is computed from reset values.",
    "repro": "Inspect COOLING_PHASE_KEYS in trinity/_input/dictionary.py for Lmech_total / v_mech_total membership.",
    "confidence": "medium"
  },
  {
    "id": "S11-A-25",
    "file": "trinity/phase_general/phase_events.py",
    "line": 394,
    "class": "state",
    "severity": "S3",
    "claim": "check_event_termination indexes `events[i]` positionally against sol.t_events, so any caller that adds or reorders events between building the list and solving reports the wrong event, or raises IndexError.",
    "evidence": "Line 394 `event = events[i]` where i enumerates sol.t_events (line 392). build_implicit_phase_events returns a 2-tuple `(events, cooling_factory)` (line 501) rather than a bare list like the other three builders, so the phase-1b caller must splice the cooling event in itself; nothing in this file records or checks the resulting index alignment.",
    "expected": "Pass the exact list handed to solve_ivp, or attach the event objects to the result rather than re-indexing, or assert len(events) == len(sol.t_events).",
    "failure_scenario": "Phase 1b appends the cooling event to the solver's event list but passes the original 3- or 4-element list to check_event_termination; a root in the cooling slot either raises IndexError or is attributed to the wrong event's name, reason_code and end_code.",
    "repro": "Call check_event_termination(sol, events) where sol has one more event slot than len(events) and only the last slot has a root.",
    "confidence": "low"
  }
]
```
