# S4 phase1 energy — Lens B (what the code claims)

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

Prose-only transcription of `trinity/phase1_energy/energy_phase_ODEs.py` and
`trinity/phase1_energy/run_energy_phase.py`. Every statement below is a *claim made by the
prose*, not a statement about the code. I have not seen the implementation and cannot say
whether any claim holds.

**Headline for the other lens:** this slice's prose contains **no equations, no equation
numbers, no Weaver+77 equation references, and no Rahner-thesis reference at all.** The
entire equation of motion is documented by four one-line section headers ("Gravity force
(self + cluster)", "Radiation force", "Time derivatives", "Energy derivative"). There is
exactly one equation-like citation in the whole slice — `(Eq. leak)` — and it names no
source. Sign, coefficient and exponent errors in the EOM are therefore **unfalsifiable from
the documentation**; Lens A's reading of the code is the only check that exists.

---

## 1. Contracts

### 1.1 State vector and derivatives

`trinity/phase1_energy/energy_phase_ODEs.py:169` (docstring `get_ODE_Edot_pure`):

- Input `t : float` — "Time in Myr".
- Input `y : list` — "State vector `[R2, v2, Eb]` — radius, velocity, energy".
- Returns `list` — "Derivatives `[rd, vd, Ed]`".

Transcribed ordering claim:

```
y    = [ R2 , v2 , Eb ]          (index 0,1,2)
dy/dt= [ rd , vd , Ed ]          (index 0,1,2)
```

so `rd = dR2/dt`, `vd = dv2/dt`, `Ed = dEb/dt`. This ordering is stated **once** and never
restated; no other comment in the slice contradicts it, and no comment confirms it either.
**No units are given for `R2`, `v2` or `Eb`** — see §3.

Additional inputs claimed at the same site:
- `snapshot : ODESnapshot` — "Frozen snapshot of parameters".
- `params_for_feedback : DescribedDict` — "Original params dict, used ONLY for
  `get_current_sps_feedback` (which needs interpolation tables that are too large to copy)".

### 1.2 `ODESnapshot` — documented fields (grouped as the prose groups them)

`trinity/phase1_energy/energy_phase_ODEs.py:60`: "Frozen snapshot of parameters needed for
ODE evaluation. This captures all values needed by the ODE function at the start of an
integration segment, **ensuring the ODE function never reads from or writes to the params
dictionary during integration**."

Field groups, with the prose's own labels:

| line | claimed group / field |
|---|---|
| `:67` | Shell properties |
| `:73` | "Density at ionization front (from shell structure ODE)" — i.e. `n_IF` originates in the shell-structure ODE, not here |
| `:74` | "Gate all HII pressure" — a gate that switches **all** HII pressure off/on |
| `:75` | "Radius of ionization front (pc)" |
| `:77` | Cluster/bubble properties |
| `:84` | Physical constants |
| `:90` | ISM properties |
| `:95` | Timing |
| `:98` | Phase info |
| `:102` | Cloud properties |
| `:105` | "HII pressure from Strömgren ionization balance in shell (`n_IF_Str`)" |
| `:108` | "Covering-fraction leak: `Cf=1` -> sealed bubble (no leak); `c_sound` is the hot-bubble sound speed (from `bubble_Tavg`), **frozen per segment**." |

### 1.3 `create_ODE_snapshot` contract

`trinity/phase1_energy/energy_phase_ODEs.py:115`: "Create a frozen snapshot of all
parameters needed for ODE evaluation. **This should be called once at the start of each
integration segment, not during ODE evaluation.**" Inputs: `params : DescribedDict`,
`shell_props : ShellProperties` — "Shell properties from the current `shell_structure_pure()`
call. **Used to compute the radiation pressure force inline.**"

`trinity/phase1_energy/energy_phase_ODEs.py:129`: "Radiation pressure force (direct +
IR-trapped)" — so `F_rad` is claimed to be assembled from **two** components, direct and
IR-trapped, and computed **inside snapshot creation** (⇒ frozen for the whole segment).

### 1.4 `ODEResult` contract

`trinity/phase1_energy/energy_phase_ODEs.py:290`: "Result from ODE evaluation, containing
values to update params with. This is returned after a successful integration segment and
contains all the values that should be written to params." Field groups:
- `:296` "State variables"
- `:302` "Derived quantities (optional, **computed during last ODE evaluation**)"
- `:313` "Pressure quantities"
- `:321` "Covering-fraction energy leak (registered output; **0 when `Cf=1`**)"

### 1.5 `compute_derived_quantities` contract

`trinity/phase1_energy/energy_phase_ODEs.py:326`: "Compute all derived quantities after a
successful integration step. **This is called ONCE after integration completes, not during
ODE evaluation.** Returns an `ODEResult` dataclass that can be used with `updateDict`."

### 1.6 `get_press_ion` contract

`trinity/phase1_energy/energy_phase_ODEs.py:37`: "Pressure from photoionized part of cloud at
radius `r`." — `r : float` "Radius in **pc**"; `params : DescribedDict`; returns `float`
"Pressure of ionized gas in **code units**."

### 1.7 `_scalar` contract

`trinity/phase1_energy/energy_phase_ODEs.py:31`: "Convert len-1 arrays / 0-d arrays to Python
scalars; otherwise return `x`." (Claim: identity for anything not len-1/0-d.)

### 1.8 Ordering / sequencing contracts in the runner

- `trinity/phase1_energy/run_energy_phase.py:133`: "Follows the same **compute → save → ODE**
  pattern as phases 1b/1c/2." (cross-phase parity claim)
- `trinity/phase1_energy/run_energy_phase.py:161`: "3. Compute bubble structure (**always,
  not conditional on `loop_count`**)".
- `trinity/phase1_energy/run_energy_phase.py:196`: "3b. Compute shell mass **BEFORE** shell
  structure so that the shell termination condition uses the **current** `R2`'s swept-up mass
  rather than the previous iteration's **stale** value."
- `trinity/phase1_energy/run_energy_phase.py:259`: "6. Save snapshot **BEFORE ODE** — all
  values consistent at `t_now`".
- `trinity/phase1_energy/run_energy_phase.py:383`: "Phase-boundary reconciliation snapshot.
  Recompute derived properties (`Pb`, shell structure) with the **post-ODE** state so the
  snapshot is fully consistent. A bare `save_snapshot()` would save **stale** derived values
  AND **block the next phase's correct first snapshot via the duplicate guard**."
  (⇒ documented side effect: a duplicate-guard exists and can suppress the next phase's first
  snapshot.)
- Loop step order as numbered by the prose: 1 (update params with current state, `:145`) → 2
  (get feedback, `:154`) → 3 (bubble structure, `:160`) → 3b (shell mass, `:196`) → 3c (shell
  structure, `:204`; `P_HII` computed at `:211`; sound speed at `:221`) → **[no step 4]** → 5
  (forces and diagnostics, `:225`) → 6 (save snapshot, `:259`) → 6b (cooling-balance transition
  trigger, `:264`) → 7 (create ODE snapshot and integrate, `:289`) → 8 (extract new state,
  `:333`; early-phase-approximation switch `:341`; update params with new state `:353`).

### 1.9 Purity contract (stated three times)

`trinity/phase1_energy/energy_phase_ODEs.py:3` (module docstring):
- "This module provides ODE functions that **do NOT mutate** the params dictionary."
- Rationale: "essential for using adaptive ODE solvers like `scipy.integrate.solve_ivp`, which
  take trial steps that can be **rejected**. If ODE functions mutate state during rejected trial
  steps, the params dictionary becomes **corrupted**."
- "`get_ODE_Edot_pure()` returns only derivatives, never writes to params"
- "**All parameters are read at the start** and passed as a frozen snapshot"
- "Dictionary updates happen **only after successful integration segments**"

`trinity/phase1_energy/energy_phase_ODEs.py:60`: "…ensuring the ODE function **never reads
from or writes to** the params dictionary during integration."

`trinity/phase1_energy/energy_phase_ODEs.py:169`: "This function does NOT write to any
dictionary. It only reads from the frozen snapshot and returns derivatives."

Same-file qualifiers that weaken it (see finding S4-B-01):
- `:194` "Get feedback values (**this reads from params** but doesn't write)"
- `:199` "Calculate shell mass using existing `mass_profile` module (**only reads from
  params**, safe for ODE evaluation)"
- `:234` "Uses existing `get_press_ion()` which **only reads from params**"

`trinity/phase1_energy/run_energy_phase.py:3` restates the same rationale: "The dictionary
mutation problem: Original ODE functions write to params during evaluation; Adaptive solvers
take trial steps that can be rejected; Rejected trial steps leave params in corrupted state;
Solution: Pure ODE functions + update params only after successful segments."

---

## 2. Equations of motion — what the prose says is in, and what is out

The prose gives **no formula, no coefficient and no explicit sign for any term.** The
following is the complete inventory of terms it names.

### 2.1 Momentum equation (`vd`) — terms claimed **INCLUDED**

| term | prose | site | stated sign / direction |
|---|---|---|---|
| Gravity, shell self-gravity **+** cluster | "Gravity force (self + cluster)" | `energy_phase_ODEs.py:219` | none stated (physically inward; not written) |
| Bubble pressure `Pb` | "Bubble pressure calculation (uses shared helper for ODE/diagnostics consistency)" | `energy_phase_ODEs.py:225` | none stated |
| Photoionized-gas pressure outside shell | "**Inward** pressure from photoionized gas outside shell" | `energy_phase_ODEs.py:233` | **inward** (explicit) |
| ISM pressure | "Add ISM pressure **if shell beyond cloud**" | `energy_phase_ODEs.py:242` | none stated; conditional on `R2 > rCloud` |
| Warm-ionized-gas pressure `P_HII` | "P_HII from Strömgren ionization balance in shell (`n_IF_Str`). **Pre-computed in phase runner and stored in snapshot.**" | `energy_phase_ODEs.py:246` | none stated |
| Radiation force | "Radiation force" | `energy_phase_ODEs.py:260` | none stated |
| Early-phase approximation | "Early phase approximation" | `energy_phase_ODEs.py:268` | none stated |

Driving-pressure combination rule, stated **verbatim twice**:
```
P_drive = max(Pb, P_HII)          # "energy / implicit phases: max(Pb, P_HII)"
```
`energy_phase_ODEs.py:257` (ODE right-hand side) and `energy_phase_ODEs.py:393`
(`compute_derived_quantities`). No citation for choosing `max()` over a sum.

Shared-helper claim: `energy_phase_ODEs.py:360` "Bubble pressure (uses shared helper for
ODE/diagnostics consistency)"; `:361` "In momentum phase, this returns `pRam`; in energy
phase, returns `bubble_E2P`."

### 2.2 Terms claimed **NEGLECTED / zero**

| term | prose | site |
|---|---|---|
| Ram pressure `P_ram` | "P_ram: only relevant in transition; **0 in energy/implicit**" | `energy_phase_ODEs.py:398` |
| ISM pressure while shell is inside the cloud | implied by "**if** shell beyond cloud" | `energy_phase_ODEs.py:242` |
| Covering-fraction leak when `Cf=1` | "`Cf=1` -> **0 exactly** (sealed Weaver bubble)" / "0 when `Cf=1`" | `energy_phase_ODEs.py:276`, `:321` |
| All HII pressure, when gated | "Gate all HII pressure" | `energy_phase_ODEs.py:74` |

### 2.3 Energy equation (`Ed`)

`energy_phase_ODEs.py:272` "Energy derivative", then `:274`:

> "Geometry-set covering-fraction leak (**Eq. leak**): computed live from the **same
> instantaneous `Pb` and `R2` used by the `P dV` term**; `cs` and `Cf` are **frozen per
> segment** in the snapshot. `Cf=1` -> **0 exactly** (sealed Weaver bubble)."

Transcribed as far as the prose licenses:
```
Ed = dEb/dt = (mechanical energy input)  -  (P dV term, using instantaneous Pb, R2)
                                          -  L_leak(Pb, R2, c_s, Cf)
L_leak is SUBTRACTED from Edot          ["the same value the RHS subtracts from Edot", :404]
L_leak(Cf = 1) = 0 exactly
L_leak uses c_s = hot-bubble sound speed from bubble_Tavg, frozen per segment   [:108]
L_leak uses Cf frozen per segment                                              [:275]
```
The **functional form of `L_leak` is not documented anywhere** — only that it is
"geometry-set", depends on `(Pb, R2, c_s, Cf)`, and vanishes at `Cf = 1`. Whether cooling
`L_cool` appears in `Ed` is **not stated** in this file's prose (it appears only in the
runner's transition-trigger comment, §6).

### 2.4 Shell mass rule (stated **twice, verbatim**)

`energy_phase_ODEs.py:201` (inside the ODE RHS) and `energy_phase_ODEs.py:339` (inside
`compute_derived_quantities`), identical text:

> "Two conditions for freezing shell mass: 1. During collapse (`isCollapse=True`): shell mass
> is frozen 2. Shell mass can **NEVER** decrease — once mass is swept up, it stays in shell"

plus `:211` and `:349`, both "Ensure shell mass never decreases". Transcribed:
```
if isCollapse:  M_sh(t) = M_sh(frozen)
else:           M_sh(t) = max( M_sh(previous) , M_swept(R2) )      # monotone non-decreasing
```
Claim: `M_sh` comes from "the existing `mass_profile` module" (`:199`).

### 2.5 `R1`

`energy_phase_ODEs.py:222` "Calculate `R1` (inner bubble radius)"; also `:357` "R1". No
formula, no citation. Related failure note: `run_energy_phase.py:362` says an energy-driven
collapse "would drive `R1 -> R2` and **divide-by-zero -> Eb=nan**", and
`run_energy_phase.py:164` says "`solve_R1` cannot bracket" in the degenerate limit.

---

## 3. Units and conventions

| claim | site |
|---|---|
| `t` (ODE independent variable) is in **Myr** | `energy_phase_ODEs.py:169` |
| `r` argument of `get_press_ion` is in **pc** | `energy_phase_ODEs.py:37` |
| `get_press_ion` returns pressure in "**code units**" (never defined in this slice) | `energy_phase_ODEs.py:37` |
| Radius of ionization front is in **pc** | `energy_phase_ODEs.py:75` |
| `MAX_DURATION` constant is in **Myr**, "~3000 years" | `run_energy_phase.py:54` |
| segment length constant is in **Myr**, "~30 years" | `run_energy_phase.py:55` |
| tfinal-proximity exit constant is in **Myr** | `run_energy_phase.py:56` |
| cooling-recalculation interval is in **Myr**, "every 50k years" | `run_energy_phase.py:57` |

**No units are documented for any state-vector component** (`R2`, `v2`, `Eb`), for `Pb`,
`P_HII`, `n_IF`, `c_sound`, `M_sh`, or for any force. **No conversion site is named anywhere
in the slice** — there is not a single comment in this prose saying where a unit conversion is
applied. Given the project's stated unit-bug history, this is a documentation gap, not a
proven defect.

---

## 4. Citations

Complete list. There is **no Weaver+77 equation number, no Rahner thesis reference, and no
textbook reference** in this slice.

| citation, verbatim | what is attributed to it | site |
|---|---|---|
| "the **Weaver+77** bubble expansion model" | the whole energy-driven phase model — no equation number, no year-qualified section | `run_energy_phase.py:63` |
| "sealed **Weaver** bubble" | the `Cf=1` limit of the leak term | `energy_phase_ODEs.py:276` |
| "**Eq. leak**" | the covering-fraction leak term — **no document, no number, no source** | `energy_phase_ODEs.py:274` |
| "**Strömgren** ionization balance" (×3) | the origin of `P_HII` / `n_IF_Str` | `energy_phase_ODEs.py:105`, `:248`, `:384`; `run_energy_phase.py:211` |
| "`docs/dev/transition/pdv-trigger/HIMASS_HANDOFF_PLAN.md`" (×2) | the deferred 1a→momentum collapse routing | `run_energy_phase.py:168`, `:367` |
| "**SAME formula as `run_energy_implicit_phase.py`** (`Lgain=Lmech_total`, `Lloss=effective_Lloss(Lcool=bubble_LTotal, leak)`)" | the step-6b cooling-balance transition trigger | `run_energy_phase.py:270` |
| "`run_energy_implicit_phase.classify_energy_collapse`" | phase 1b's routing of a clean `Eb<=0` collapse to the momentum phase | `run_energy_phase.py:364` |
| "`scipy.integrate.solve_ivp`" / "**RK45** adaptive solver" | the integrator, "instead of manual Euler" | `run_energy_phase.py:3`, `:63` |
| "existing `mass_profile` module" | shell-mass computation | `energy_phase_ODEs.py:199`, `:338` |
| "`bubble_Tavg`" | source of the hot-bubble sound speed `c_sound` | `energy_phase_ODEs.py:109` |
| "dataclass returns from `bubble_luminosity`" | integration approach item 4 | `run_energy_phase.py:3` |
| "`shell_structure_pure()`" | source of `shell_props` for the snapshot | `energy_phase_ODEs.py:115` |
| "phases 1b/1c/2" | source of the compute→save→ODE pattern | `run_energy_phase.py:133` |

Note the two `docs/dev/…` pointers: per the project's own rules those documents are
point-in-time and unverified, so two live code comments defer their justification to material
that is explicitly not maintained.

---

## 5. Regimes, assumptions, entry/exit conditions

Claimed **exit / termination paths** for this phase (all of them, as documented):

1. **Max duration** — `run_energy_phase.py:54`, "max duration (~3000 years)".
2. **Proximity to `tfinal`** — `run_energy_phase.py:56`, "exit when this close to `tfinal`".
3. **Solver events** — `run_energy_phase.py:114` "Build events for safe termination";
   `:41` "Import centralized event functions"; `:323` "Check if an event terminated the
   integration". **Which events, whether terminal, and in which direction: not documented.**
4. **Bubble-structure failure** (`run_energy_phase.py:163`): "In the energy-driven `Eb -> 0`
   collapse the bubble degenerates: the cooling table goes out of bounds, `solve_R1` cannot
   bracket, etc. **Any such failure here** means the energy-driven model has broken down —
   **stop the run cleanly rather than crash with the bare exception.**"
5. **Post-ODE `Eb <= 0` collapse** (`run_energy_phase.py:359`): "a massive/dense cloud can
   lose the bubble's thermal energy (PdV work on a heavy shell, or radiative cooling) faster
   than the wind resupplies it, so `Eb` falls through zero. The energy-driven model is then
   invalid (it would drive `R1->R2` and divide-by-zero -> `Eb=nan`). Phase 1b now ROUTES such
   a collapse to the momentum phase (`run_energy_implicit_phase.classify_energy_collapse`);
   **routing it from 1a too is deferred (rare: collapse within the fixed ~3000-yr early
   window). Until then 1a stops cleanly here.**"
6. **Cooling-balance transition trigger (step 6b)** (`run_energy_phase.py:265`):
   "Transition-trigger parity with phase 1b (`cooling_balance`). A violently cooling cloud can
   reach the energy→momentum cooling balance WITHIN this fixed ~3000-yr early phase; without a
   check here it would either wait for the 1a→1b boundary or, if cooling drives `Eb<=0` first,
   hit the collapse routing below. **Evaluated at the consistent pre-ODE snapshot** with the
   SAME formula as `run_energy_implicit_phase.py` (`Lgain=Lmech_total`,
   `Lloss=effective_Lloss(Lcool=bubble_LTotal, leak)`). **No-op for healthy bubbles (early
   cooling is negligible, ratio ~1 >> threshold) -> byte-identical (G0).**"

Claimed **assumptions / approximations**:
- "**Early phase approximation**" — `energy_phase_ODEs.py:268`, with a switch handled at
  `run_energy_phase.py:341` ("Handle early phase approximation switch"). **What is
  approximated, the switch criterion, and the validity range are all undocumented.**
- `Cf = 1` ⇒ sealed Weaver bubble, leak exactly 0 (`energy_phase_ODEs.py:108`, `:276`, `:321`).
- Shell mass monotone non-decreasing, frozen during collapse (§2.4) — a modelling assumption
  ("once mass is swept up, it stays in shell"), not a numerical guard.
- Radiation force is "direct + IR-trapped" and is computed at snapshot time
  (`energy_phase_ODEs.py:115`, `:129`) ⇒ **held constant over each ~30-yr segment**.
- `c_sound` and `Cf` "frozen per segment" (`energy_phase_ODEs.py:109`, `:275`), while `Pb` and
  `R2` in the same leak term are "computed live".
- `P_ram = 0` in energy/implicit phases (`energy_phase_ODEs.py:398`).
- Naming: the module is "the energy-driven phase (**Phase 1**)" (`run_energy_phase.py:63`) but
  the comments call it "**1a**" (`:359`, `:364`) alongside "phases 1b/1c/2" (`:133`).

---

## 6. Numerical claims

- Integrator: `scipy.integrate.solve_ivp`, **RK45**, adaptive stepping, explicitly "rather
  than manual Euler integration" (`run_energy_phase.py:3`, `:63`).
- "**Segment-based integration**: short segments with params updates only after success"
  (`run_energy_phase.py:3`).
- Named tolerance constants: "Relative tolerance for `solve_ivp`" (`run_energy_phase.py:58`),
  "Absolute tolerance for `solve_ivp`" (`:59`). (Values are code, not prose; not judged here.)
- Timing constants: max duration ~3000 yr (`:54`); segment length ~30 yr (`:55`); tfinal
  proximity exit (`:56`); cooling recalculated "every 50k years" (`:57`).
- Cooling structure is "computed **periodically**" (`run_energy_phase.py:121`) — contrast
  bubble structure, computed "**always**, not conditional on `loop_count`" (`:161`).
- Event handling: events are built for "safe termination"; the code checks "if an event
  terminated the integration". **No direction (`+1`/`-1`/`0`) and no terminal flag is
  documented for any event.**
- Fallback behaviour: broad failure in step 3 ⇒ clean run stop (§5 item 4); post-ODE
  `Eb <= 0` ⇒ clean run stop (§5 item 5). Neither path is documented as raising.
- Byte-identity claim: step 6b is asserted "**byte-identical (G0)**" for healthy bubbles
  (`run_energy_phase.py:272`).

---

## 7. Admissions of debt (verbatim)

| admission | site |
|---|---|
| "routing it from 1a too **is deferred** — see `docs/dev/transition/pdv-trigger/HIMASS_HANDOFF_PLAN.md`" | `run_energy_phase.py:168` |
| "routing it from 1a too **is deferred** (rare: collapse within the fixed ~3000-yr early window). **Until then 1a stops cleanly here.**" | `run_energy_phase.py:365` |
| "**stop the run cleanly rather than crash with the bare exception**" (⇒ a bare/broad exception handler exists) | `run_energy_phase.py:166` |
| "Early phase **approximation**" / "Handle early phase **approximation** switch" | `energy_phase_ODEs.py:268`; `run_energy_phase.py:341` |
| "Derived quantities (**optional**, …)" | `energy_phase_ODEs.py:302` |
| "rather than the previous iteration's **stale** value" | `run_energy_phase.py:199` |
| "would save **stale** derived values AND **block** the next phase's correct first snapshot via the duplicate guard" | `run_energy_phase.py:387` |
| "`Eb=nan`" as the documented consequence of the un-handled path | `run_energy_phase.py:366` |

No `TODO`, `FIXME`, `XXX`, `hack`, `not physical`, `temporary`, `unclear` or `I think` markers
appear anywhere in this slice's prose.

---

## 8. Flags — prose contradicting prose, and claims to test

### S4-B-01 (S2) — the module's central purity invariant contradicts itself
`energy_phase_ODEs.py:60` claims the snapshot ensures the ODE function "**never reads from or
writes to** the params dictionary during integration", and `:3` claims "**All** parameters are
read at the start and passed as a frozen snapshot". But the same file documents **three live
reads of `params` inside the RHS**: `params_for_feedback` "used ONLY for
`get_current_sps_feedback`" (`:169`), `:194` "this reads from params but doesn't write", `:199`
`mass_profile` "only reads from params", `:234` `get_press_ion` "only reads from params". The
"never reads" claim is false on the prose's own testimony; the load-bearing claim reduces to
"never *writes*", asserted for three helpers by comment only. If any of the three writes to
`params`, caches into it, or touches module-level global state, the rejected-trial-step
corruption the module exists to prevent recurs **silently**.

### S4-B-02 (S3) — when are derived quantities computed?
`energy_phase_ODEs.py:302` says derived quantities are "computed **during last ODE
evaluation**"; `energy_phase_ODEs.py:326` says `compute_derived_quantities` "is called ONCE
after integration completes, **not during ODE evaluation**". Direct contradiction. If the
former is true, the derived values correspond to whatever trial step the solver last
evaluated — which may be a **rejected** step at a time ≠ the segment end.

### S4-B-03 (S3) — the periodic cooling recomputation cannot fire inside this phase
`run_energy_phase.py:57` sets the cooling recalculation interval to "every **50k years**";
`run_energy_phase.py:54` caps the phase at "~**3000 years**". 50 000 / 3000 ≈ 16.7, so the
"Cooling structure (**computed periodically**)" block (`:121`) can only ever run its initial
computation within Phase 1. Either the label/constant is misleading dead configuration, or the
constant is shared with a longer phase and the cooling structure is silently stale for the
whole of Phase 1.

### S4-B-04 (S2) — broad exception handler re-labels any bug as physical breakdown
`run_energy_phase.py:163`: "**Any such failure here** means the energy-driven model has broken
down — stop the run cleanly rather than crash with the bare exception." A `TypeError`,
`AttributeError`, `KeyError` or an unrelated regression inside bubble-structure computation is
indistinguishable from a genuine `Eb→0` degeneracy and is reported as a clean physical stop.

### S4-B-05 (S3) — `(Eq. leak)` is an unresolvable citation
`energy_phase_ODEs.py:274` cites "Eq. leak" with no paper, thesis, section or number. It is the
only equation reference in the slice, and it labels the one term whose functional form is
otherwise entirely undocumented.

### S4-B-06 (S3) — the equation of motion has no written form anywhere
Gravity (`:219`), radiation (`:260`), time derivatives (`:263`), energy derivative (`:272`) are
one-line headers. No sign is stated for any term except "**Inward** pressure from photoionized
gas" (`:233`). No coefficient, no exponent, no Weaver+77 equation number, no Rahner reference.
Nothing in the documentation can catch a sign flip or a factor error.

### S4-B-07 (S3) — state-vector units undocumented while neighbouring units are specified
`t` is documented as Myr and `get_press_ion`'s `r` as pc, but `R2`, `v2`, `Eb` in
`y = [R2, v2, Eb]` (`:169`) carry no units, and "code units" (`:37`) is never defined in the
slice. No comment names a conversion site anywhere in either file.

### S4-B-08 (S3) — shell-mass invariant implemented in two places
Identical four-line rule at `energy_phase_ODEs.py:201`/`:211` (ODE RHS) and
`energy_phase_ODEs.py:339`/`:349` (`compute_derived_quantities`). Two copies of a stateful
"never decreases" rule that must agree; a drift between them makes the recorded shell mass
differ from the one the RHS integrated against.

### S4-B-09 (S3) — leak diagnostic asserted equal to the RHS term
`energy_phase_ODEs.py:404`: "Covering-fraction leak diagnostic (**the same value the RHS
subtracts from `Edot`**)", plus `:321` "0 when `Cf=1`" and `:276` "`Cf=1` -> 0 exactly". Two
independent evaluations claimed identical, with `Pb`/`R2` "computed live" but `cs`/`Cf` frozen
— the diagnostic is computed from the post-ODE state, the RHS from the in-step state.

### S4-B-10 (S3) — mixed frozen/live terms within one segment, never flagged as an approximation
`F_rad` (direct + IR-trapped) is computed inside `create_ODE_snapshot` (`:115`, `:129`) ⇒
constant over each ~30-yr segment; `c_sound` and `Cf` are "frozen per segment" (`:108`, `:275`);
but `Pb` and `R2` are "computed live" (`:274`). The prose flags the freeze for `cs`/`Cf` and
not for `F_rad`, and never states the resulting accuracy limit.

### S4-B-11 (S3) — "Early phase approximation" with no stated content or validity range
`energy_phase_ODEs.py:268` and `run_energy_phase.py:341` ("Handle early phase approximation
switch"). What is approximated, when the switch flips, and where the approximation is valid
are all undocumented, in a term that sits directly in the momentum equation.

### S4-B-12 (S3) — event semantics undocumented
`run_energy_phase.py:41`, `:114`, `:323` reference "centralized event functions" built for
"safe termination", but no comment states which events are `terminal`, what `direction` each
uses, or what happens when several fire in one segment.

### S4-B-13 (S2) — phase-dependent outcome for the same physical collapse, admitted
`run_energy_phase.py:359`: an `Eb<=0` collapse occurring in 1a **stops the run**, whereas the
identical collapse in 1b is **routed to the momentum phase**. The justification given is
rarity, not physics, and it defers to an unverified `docs/dev/` plan. A massive/dense cloud
that collapses inside the fixed ~3000-yr window silently loses its momentum-phase evolution.

### S4-B-14 (S3) — `max(Pb, P_HII)` is an uncited modelling choice
Stated verbatim at `energy_phase_ODEs.py:257` and `:393`. Taking the maximum rather than the
sum of two physically co-existing pressures has no citation. Paired with `P_ram = 0` in
energy/implicit (`:398`) and ISM pressure only "if shell beyond cloud" (`:242`), this is the
complete documented pressure budget.

### S4-B-15 (S3) — the HII gate is documented once but must apply in two paths
"Gate all HII pressure" appears only as a snapshot field (`energy_phase_ODEs.py:74`). The
`P_HII` banner is duplicated verbatim in the RHS (`:246`) and in `compute_derived_quantities`
(`:382`); neither restates the gate. If the gate is honoured in one path and not the other,
the diagnostics disagree with the integrated dynamics.

### S4-B-16 (S3) — step-6b parity and "byte-identical" claim
`run_energy_phase.py:265` claims the 1a trigger uses the "SAME formula as
`run_energy_implicit_phase.py` (`Lgain=Lmech_total`, `Lloss=effective_Lloss(Lcool=
bubble_LTotal, leak)`)" and is a "**No-op for healthy bubbles … -> byte-identical (G0)**".
Two testable claims: formula parity across modules, and no output change for healthy runs.
The parenthetical "ratio ~1 >> threshold" also implies the threshold is ≪ 1, which the prose
never states.

### S4-B-17 (S3) — ordering requirements stated but unenforced by documentation
Four sequencing contracts (`run_energy_phase.py:133`, `:196`, `:259`, `:383`) plus
"call `create_ODE_snapshot` once at the start of each segment"
(`energy_phase_ODEs.py:115`). Violating any of them yields stale-but-plausible output rather
than an error — notably the shell-termination condition using the previous iteration's mass
(`:199`) and the duplicate guard suppressing the next phase's first snapshot (`:387`).

### S4-B-18 (S4) — main-loop step numbering skips 4
Steps run 1, 2, 3, 3b, 3c, [nothing], 5, 6, 6b, 7, 8; "Calculate sound speed"
(`run_energy_phase.py:221`) sits unnumbered exactly where step 4 would be. Suggests a removed
or renamed step and a comment left un-renumbered.

### S4-B-19 (S4) — "Phase 1" vs "1a"
`run_energy_phase.py:63` calls this "the energy-driven phase (**Phase 1**)" while `:359` and
`:364` call it "**1a**" and `:133` names "phases 1b/1c/2" as siblings. Two names for one phase
in one file, in an area where the routing rules differ **between** 1a and 1b.

---

```json
[
  {
    "id": "S4-B-01",
    "file": "trinity/phase1_energy/energy_phase_ODEs.py",
    "line": 60,
    "class": "state",
    "severity": "S2",
    "claim": "ODESnapshot 'ensur[es] the ODE function never reads from or writes to the params dictionary during integration' and the module docstring claims 'All parameters are read at the start and passed as a frozen snapshot'.",
    "evidence": "Same file contradicts it three times: get_ODE_Edot_pure takes 'params_for_feedback : DescribedDict Original params dict, used ONLY for get_current_sps_feedback' (energy_phase_ODEs.py:169); ':194 Get feedback values (this reads from params but doesn't write)'; ':199 Calculate shell mass using existing mass_profile module (only reads from params, safe for ODE evaluation)'; ':234 Uses existing get_press_ion() which only reads from params'.",
    "expected": "Either the snapshot is complete and the RHS touches no dict, or the docstrings say params is read (read-only) during evaluation for feedback, shell mass and get_press_ion. The read-only-ness of those three helpers should be enforced, not merely asserted in comments.",
    "failure_scenario": "If get_current_sps_feedback, mass_profile or get_press_ion writes to params or to module-level global state (trinity is documented as leaking module-level globals), a rejected RK45 trial step corrupts params exactly as the module docstring says it must not — silently, with no error, producing a wrong trajectory.",
    "repro": "deepcopy(params) before a solve_ivp segment in run_energy; run the segment; compare dict contents after. Assert no key changed during integration (not just after).",
    "confidence": "high"
  },
  {
    "id": "S4-B-02",
    "file": "trinity/phase1_energy/energy_phase_ODEs.py",
    "line": 302,
    "class": "state",
    "severity": "S3",
    "claim": "ODEResult field group is labelled 'Derived quantities (optional, computed during last ODE evaluation)'.",
    "evidence": "compute_derived_quantities docstring at energy_phase_ODEs.py:326 states the opposite: 'This is called ONCE after integration completes, not during ODE evaluation.'",
    "expected": "One consistent statement of when derived quantities are produced.",
    "failure_scenario": "If they really are captured during the last ODE evaluation, that evaluation may be a REJECTED trial step at a time other than the segment end, so recorded diagnostics (Pb, forces, leak) correspond to a state the solver discarded.",
    "repro": "Log t at each derived-quantity computation and compare with the segment end time returned by solve_ivp; they must be equal.",
    "confidence": "high"
  },
  {
    "id": "S4-B-03",
    "file": "trinity/phase1_energy/run_energy_phase.py",
    "line": 57,
    "class": "numerical",
    "severity": "S3",
    "claim": "Cooling structure is 'computed periodically' (run_energy_phase.py:121) on a constant documented as 'Myr - recalculate cooling every 50k years'.",
    "evidence": "run_energy_phase.py:54 documents the phase max duration as 'Myr - max duration (~3000 years)'. 50000 yr > 3000 yr by a factor ~16.7, so the periodic branch can never re-fire within Phase 1.",
    "expected": "Either an interval shorter than the phase duration, or a comment saying the cooling structure is deliberately computed once for this phase.",
    "failure_scenario": "Cooling structure is silently stale for the whole of Phase 1 while the code and comment imply it is refreshed; alternatively the recompute branch is unreachable dead code that a reader trusts.",
    "repro": "Instrument the cooling-recompute branch with a counter; run param/simple_cluster.param; assert count > 1 if 'periodically' is intended.",
    "confidence": "high"
  },
  {
    "id": "S4-B-04",
    "file": "trinity/phase1_energy/run_energy_phase.py",
    "line": 163,
    "class": "silent-failure",
    "severity": "S2",
    "claim": "'In the energy-driven Eb -> 0 collapse the bubble degenerates: the cooling table goes out of bounds, solve_R1 cannot bracket, etc. Any such failure here means the energy-driven model has broken down -- stop the run cleanly rather than crash with the bare exception.'",
    "evidence": "The comment explicitly scopes the handler to 'Any such failure here' and contrasts it with 'the bare exception', i.e. a broad catch around the whole step-3 bubble-structure computation.",
    "expected": "Catch only the specific degeneracy exceptions (table out-of-bounds, bracketing failure) and let programming errors propagate.",
    "failure_scenario": "A TypeError/AttributeError/KeyError introduced by an unrelated refactor inside bubble structure is reported as 'energy-driven model has broken down', the run stops cleanly, and a regression looks like physics. No test would fail loudly.",
    "repro": "Inject `raise TypeError('canary')` at the top of the step-3 bubble-structure call; run param/simple_cluster.param; observe whether the run reports a clean physical stop instead of an error.",
    "confidence": "medium"
  },
  {
    "id": "S4-B-05",
    "file": "trinity/phase1_energy/energy_phase_ODEs.py",
    "line": 274,
    "class": "citation",
    "severity": "S3",
    "claim": "The covering-fraction leak term is attributed to '(Eq. leak)'.",
    "evidence": "No paper, thesis, section or equation number accompanies it; it is the only equation-style citation in the entire slice.",
    "expected": "A resolvable reference (e.g. Weaver+77 Eq. N, or Rahner thesis Eq. N, or a docs/dev derivation) for the leak formula, which is otherwise undocumented apart from 'geometry-set', dependence on (Pb, R2, c_s, Cf), and L_leak(Cf=1)=0.",
    "failure_scenario": "The leak term's functional form, coefficient and exponents cannot be checked against any source; an error in it is invisible to review and only manifests as a wrong Eb trajectory for Cf<1 runs.",
    "repro": "",
    "confidence": "high"
  },
  {
    "id": "S4-B-06",
    "file": "trinity/phase1_energy/energy_phase_ODEs.py",
    "line": 219,
    "class": "citation",
    "severity": "S3",
    "claim": "The equations of motion are documented only by section headers: 'Gravity force (self + cluster)' (:219), 'Radiation force' (:260), 'Time derivatives' (:263), 'Energy derivative' (:272).",
    "evidence": "No formula, coefficient, exponent or sign appears in prose for any term except 'Inward pressure from photoionized gas outside shell' (:233). The only model citation is 'the Weaver+77 bubble expansion model' (run_energy_phase.py:63) with no equation number; the Rahner thesis is never cited in this slice.",
    "expected": "Each EOM term carries its source equation (Weaver+77 Eq. N / Rahner Eq. N) and its sign convention, per the project's own unit/sign bug-class warning.",
    "failure_scenario": "A sign flip or missing factor in the gravity, radiation or PdV term cannot be detected by reading the code against its documentation; it would show only as a quantitatively wrong but plausible expansion history.",
    "repro": "",
    "confidence": "high"
  },
  {
    "id": "S4-B-07",
    "file": "trinity/phase1_energy/energy_phase_ODEs.py",
    "line": 169,
    "class": "units",
    "severity": "S3",
    "claim": "'t : float Time in Myr; y : list State vector [R2, v2, Eb] - radius, velocity, energy' — t is given units, the state components are not.",
    "evidence": "get_press_ion documents 'Radius in pc' and returns pressure 'in code units' (energy_phase_ODEs.py:37); ionization-front radius is documented as pc (:75); the four runner constants are documented as Myr (:54-:57). No units are given for R2, v2, Eb, Pb, P_HII, n_IF, c_sound or any force, 'code units' is never defined, and no comment in either file names a unit-conversion site.",
    "expected": "State-vector components documented with units consistent with t in Myr (e.g. R2 in pc, v2 in pc/Myr or km/s — these differ by ~1.022), and 'code units' defined or cross-referenced to trinity/_functions/unit_conversions.py.",
    "failure_scenario": "A km/s vs pc/Myr mix-up in v2, or a code-units vs cgs mix-up between get_press_ion's return and the Pb used in max(Pb, P_HII), gives a silently wrong force budget of order unity-to-1e-something with no documentation to catch it.",
    "repro": "Check the units of v2 as consumed by rd = dR2/dt against t in Myr and R2 in pc; check get_press_ion's return against the units of bubble_E2P at the max(Pb, P_HII) site.",
    "confidence": "medium"
  },
  {
    "id": "S4-B-08",
    "file": "trinity/phase1_energy/energy_phase_ODEs.py",
    "line": 201,
    "class": "divergence",
    "severity": "S3",
    "claim": "The shell-mass rule is stated verbatim in two places: 'Two conditions for freezing shell mass: 1. During collapse (isCollapse=True): shell mass is frozen 2. Shell mass can NEVER decrease - once mass is swept up, it stays in shell' at energy_phase_ODEs.py:201 (ODE RHS) and :339 (compute_derived_quantities), each followed by 'Ensure shell mass never decreases' (:211, :349).",
    "evidence": "Duplicated stateful invariant across the integrated path and the diagnostics path.",
    "expected": "One shared helper, or a check that both copies implement max(M_prev, M_swept(R2)) with the same M_prev source and the same isCollapse gate.",
    "failure_scenario": "If the two copies read a different 'previous mass' (e.g. RHS uses the segment-start value, diagnostics use the last-written params value), the recorded shell mass differs from the one actually integrated against, and the saved force budget will not reproduce the trajectory.",
    "repro": "At each saved step compare the ODEResult shell mass with the shell mass the RHS used at the segment end; assert equality.",
    "confidence": "high"
  },
  {
    "id": "S4-B-09",
    "file": "trinity/phase1_energy/energy_phase_ODEs.py",
    "line": 404,
    "class": "divergence",
    "severity": "S3",
    "claim": "'Covering-fraction leak diagnostic (same value the RHS subtracts from Edot)'; and it is 0 when Cf=1 (:321, :276 'Cf=1 -> 0 exactly').",
    "evidence": "The RHS leak is 'computed live from the same instantaneous Pb and R2 used by the P dV term' with cs and Cf frozen per segment (:274-:276); the diagnostic is computed in compute_derived_quantities, i.e. from the post-integration state.",
    "expected": "The diagnostic equals the RHS-subtracted value at the same t, and is exactly 0.0 (not ~1e-30) when Cf=1.",
    "failure_scenario": "Diagnostics report an energy leak that does not match the one integrated, so an energy-budget audit of the output closes only approximately; or a nonzero leak appears for a sealed Cf=1 bubble, contradicting the documented Weaver limit.",
    "repro": "Run with Cf=1 and assert the registered leak output is identically 0.0 at every step; run with Cf<1 and compare the registered leak against the Edot leak term recomputed from the same (Pb, R2, cs, Cf).",
    "confidence": "medium"
  },
  {
    "id": "S4-B-10",
    "file": "trinity/phase1_energy/energy_phase_ODEs.py",
    "line": 129,
    "class": "numerical",
    "severity": "S3",
    "claim": "'Radiation pressure force (direct + IR-trapped)' is computed inside create_ODE_snapshot, whose docstring says shell_props is 'Used to compute the radiation pressure force inline' and that the snapshot is created 'once at the start of each integration segment' (:115).",
    "evidence": "The RHS then only has 'Radiation force' (:260). Meanwhile the leak comment explicitly distinguishes live vs frozen quantities ('computed live from the same instantaneous Pb and R2 … cs and Cf are frozen per segment', :274-:276), showing the author tracks this distinction — but it is never stated for F_rad.",
    "expected": "Either F_rad recomputed live in the RHS, or a comment stating that F_rad is held constant over each ~30-yr segment and why that is acceptable.",
    "failure_scenario": "Radiation force is piecewise-constant over each segment while pressure and gravity vary continuously, introducing a first-order-in-segment-length error in v2 that is invisible to the RK45 adaptive error estimate (the error is in the RHS definition, not in the step control).",
    "repro": "Halve the segment-length constant (run_energy_phase.py:55) and check whether the Phase 1 end state changes by more than rtol; a segment-length-dependent answer confirms the frozen-term error.",
    "confidence": "medium"
  },
  {
    "id": "S4-B-11",
    "file": "trinity/phase1_energy/energy_phase_ODEs.py",
    "line": 268,
    "class": "regime",
    "severity": "S3",
    "claim": "'Early phase approximation' is applied inside the ODE right-hand side, with a corresponding 'Handle early phase approximation switch' in the runner (run_energy_phase.py:341).",
    "evidence": "Neither site states what is approximated, what the switch criterion is, or over what regime the approximation is valid.",
    "expected": "A named approximation with its criterion and validity range, and a citation, since it modifies the momentum equation directly.",
    "failure_scenario": "The approximation is applied outside its regime (e.g. a dense/massive cloud where the early phase is not self-similar), silently biasing R2(t) and v2(t) for the whole ~3000-yr window that all later phases inherit as initial conditions.",
    "repro": "Compare the Phase 1 end state with the approximation forced on vs off across param/simple_cluster.param and the low/high-density f1 edge configs.",
    "confidence": "medium"
  },
  {
    "id": "S4-B-12",
    "file": "trinity/phase1_energy/run_energy_phase.py",
    "line": 114,
    "class": "numerical",
    "severity": "S3",
    "claim": "'Build events for safe termination' (:114), events come from 'centralized event functions' (:41), and after integration the code will 'Check if an event terminated the integration' (:323).",
    "evidence": "No comment anywhere in the slice states which events are terminal, what direction each uses, or the precedence when more than one fires in a segment.",
    "expected": "Per-event documentation of terminal flag and direction, since a non-terminal or wrong-direction event silently fails to stop the phase.",
    "failure_scenario": "An event intended to end the phase is registered with terminal=False or the wrong sign, so the phase runs past its physical validity to MAX_DURATION and hands a wrong state to the next phase — no error raised.",
    "repro": "Enumerate the centralized event callables and assert each has the intended .terminal and .direction attributes; add a test that each fires exactly once in a run where it should.",
    "confidence": "high"
  },
  {
    "id": "S4-B-13",
    "file": "trinity/phase1_energy/run_energy_phase.py",
    "line": 359,
    "class": "regime",
    "severity": "S2",
    "claim": "'Phase 1b now ROUTES such a collapse to the momentum phase (run_energy_implicit_phase.classify_energy_collapse); routing it from 1a too is deferred (rare: collapse within the fixed ~3000-yr early window). Until then 1a stops cleanly here.' Same admission at :167.",
    "evidence": "Self-declared deferred work, justified by rarity and deferred to docs/dev/transition/pdv-trigger/HIMASS_HANDOFF_PLAN.md, which the project's own rules classify as unverified and possibly stale.",
    "expected": "The same physical event (Eb falls through zero) produces the same routing regardless of which sub-phase detects it.",
    "failure_scenario": "A massive/dense GMC whose bubble collapses inside the first ~3000 yr has its run terminated instead of continuing into the momentum phase — the outcome depends on the arbitrary 1a/1b boundary rather than on physics, and a sweep silently loses those parameter combinations.",
    "repro": "Construct a high-mass/high-density config that collapses before MAX_DURATION and compare its fate with the same physics reaching collapse just after the 1a->1b boundary.",
    "confidence": "high"
  },
  {
    "id": "S4-B-14",
    "file": "trinity/phase1_energy/energy_phase_ODEs.py",
    "line": 257,
    "class": "regime",
    "severity": "S3",
    "claim": "'energy / implicit phases: max(Pb, P_HII)' — the driving pressure is the maximum of the bubble pressure and the warm-ionized-gas pressure, stated verbatim at :257 (ODE RHS) and :393 (compute_derived_quantities). Companion neglects: 'P_ram: only relevant in transition; 0 in energy/implicit' (:398); ISM pressure added only 'if shell beyond cloud' (:242).",
    "evidence": "No citation is given for taking a maximum rather than a sum of two co-existing pressures, in a slice whose only model citation is 'the Weaver+77 bubble expansion model' with no equation number.",
    "expected": "A source for the max() combination rule, or a comment explaining why the two pressures are not additive.",
    "failure_scenario": "If the correct treatment is additive (or if P_HII should act on a different surface), the shell is under-accelerated whenever Pb and P_HII are comparable — a systematic, physically plausible-looking error.",
    "repro": "",
    "confidence": "medium"
  },
  {
    "id": "S4-B-15",
    "file": "trinity/phase1_energy/energy_phase_ODEs.py",
    "line": 74,
    "class": "divergence",
    "severity": "S3",
    "claim": "A snapshot field is documented as 'Gate all HII pressure' (:74); the P_HII banner block ('P_HII from Strömgren ionization balance in shell (n_IF_Str). Pre-computed in phase runner and stored in snapshot.') is duplicated verbatim in the ODE RHS (:246-:250) and in compute_derived_quantities (:382-:386).",
    "evidence": "Neither duplicated banner mentions the gate; the gate is documented only at the field definition.",
    "expected": "The gate is honoured identically in both the integrated path and the diagnostics path.",
    "failure_scenario": "The gate suppresses P_HII in the RHS but not in the diagnostics (or vice versa), so the saved pressure budget does not explain the integrated trajectory, and a gated run appears to have HII pressure it never felt.",
    "repro": "Run with the HII gate off and assert the registered P_HII / P_drive diagnostics are consistent with the dynamics (recompute vd from saved forces and compare to the saved v2 derivative).",
    "confidence": "medium"
  },
  {
    "id": "S4-B-16",
    "file": "trinity/phase1_energy/run_energy_phase.py",
    "line": 265,
    "class": "divergence",
    "severity": "S3",
    "claim": "Step 6b is 'Evaluated at the consistent pre-ODE snapshot with the SAME formula as run_energy_implicit_phase.py (Lgain=Lmech_total, Lloss=effective_Lloss(Lcool=bubble_LTotal, leak))' and is a 'No-op for healthy bubbles (early cooling is negligible, ratio ~1 >> threshold) -> byte-identical (G0)'.",
    "evidence": "Two duplicated-formula claims across modules plus an equivalence claim; the parenthetical 'ratio ~1 >> threshold' also implies a threshold much smaller than 1, which is never stated.",
    "expected": "The 1a trigger and the 1b trigger compute the same Lgain/Lloss from the same inputs, and enabling 6b leaves healthy-bubble output byte-identical.",
    "failure_scenario": "If the two formulas drift (different Lcool source, or leak included in one and not the other), a cloud transitions at a different time depending on which sub-phase evaluated the balance; if the no-op claim is false, existing baselines silently change.",
    "repro": "Per project rule 5: run param/simple_cluster.param plus the f1edge low/high-density configs in separate processes with 6b present vs removed and diff dictionary.jsonl byte-for-byte at matched t.",
    "confidence": "medium"
  },
  {
    "id": "S4-B-17",
    "file": "trinity/phase1_energy/run_energy_phase.py",
    "line": 196,
    "class": "state",
    "severity": "S3",
    "claim": "Four ordering contracts: shell mass must be computed BEFORE shell structure 'so that the shell termination condition uses the current R2's swept-up mass rather than the previous iteration's stale value' (:196); the snapshot must be saved BEFORE the ODE 'all values consistent at t_now' (:259); the loop must follow 'the same compute -> save -> ODE pattern as phases 1b/1c/2' (:133); the phase-boundary snapshot must recompute Pb and shell structure post-ODE because 'A bare save_snapshot() would save stale derived values AND block the next phase's correct first snapshot via the duplicate guard' (:383). Plus create_ODE_snapshot 'should be called once at the start of each integration segment' (energy_phase_ODEs.py:115).",
    "evidence": "All five are stated as requirements in comments only; none is described as enforced by an assertion or test.",
    "expected": "Each ordering requirement backed by a check, since violating any of them produces stale-but-plausible numbers rather than an error.",
    "failure_scenario": "A reordering during future maintenance makes the shell termination condition use the previous iteration's mass, or lets the duplicate guard swallow the next phase's first snapshot — both produce a complete, plausible-looking run with a shifted phase boundary.",
    "repro": "Assert in the loop that the shell-mass timestamp equals R2's timestamp before shell_structure is called; assert the first snapshot of the next phase is present in dictionary.jsonl.",
    "confidence": "medium"
  },
  {
    "id": "S4-B-18",
    "file": "trinity/phase1_energy/run_energy_phase.py",
    "line": 221,
    "class": "deadcode",
    "severity": "S4",
    "claim": "The main loop's numbered steps are 1, 2, 3, 3b, 3c, 5, 6, 6b, 7, 8 — step 4 is missing, and 'Calculate sound speed' (:221) sits unnumbered exactly where step 4 would be.",
    "evidence": "run_energy_phase.py:145 (1), :154 (2), :160 (3), :196 (3b), :204 (3c), :221 (sound speed, unnumbered), :225 (5), :259 (6), :264 (6b), :289 (7), :333 (8).",
    "expected": "Contiguous numbering, or a comment noting why 4 is absent.",
    "failure_scenario": "Suggests a step was removed or merged without renumbering; a reader looking for 'step 4' in a cross-referencing comment or doc finds nothing.",
    "repro": "",
    "confidence": "high"
  },
  {
    "id": "S4-B-19",
    "file": "trinity/phase1_energy/run_energy_phase.py",
    "line": 63,
    "class": "other",
    "severity": "S4",
    "claim": "The module docstring calls this 'the energy-driven phase (Phase 1)' while comments in the same file call it '1a' (:359, :364) and name 'phases 1b/1c/2' as siblings (:133).",
    "evidence": "run_energy_phase.py:63 vs :133, :359, :364.",
    "expected": "One consistent name, since the collapse-routing rules differ specifically between 1a and 1b.",
    "failure_scenario": "A reader applies a statement about 'Phase 1' (which includes 1b) to 1a, or vice versa — directly relevant given that Eb<=0 routing exists in 1b but not 1a.",
    "repro": "",
    "confidence": "medium"
  }
]
```
