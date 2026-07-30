# S11 orchestration — Lens C (what it should be)

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

**Scope.** Derived from physics + the interface only. Inputs read: the S11 signature list
(names, values redacted) and `docs/dev/code-audit/reference/PHYSICS_SPEC.md`. No `trinity/` source,
no comments, no docstrings, no other lens report. Literature access blocked; Rahner+17/19 and
WARPFIELD claims below are from internal knowledge and are tagged accordingly.

**Interface facts I am allowed to use** (from the signature list only, values redacted):
`trinity/main.py` exposes `start_expansion`, `run_expansion`, `expansion_next(..., ii_coll)` and a
config check `_check_stop_r_rCloud_interaction(nSnap_rCloud, stop_r, rCloud)` with a constant
`_STOP_R_RCLOUD_RACE_FACTOR`. `trinity/phase_general/phase_events.py` exposes event *factories*
(`make_min_radius_event`, `make_max_radius_event`, `make_velocity_runaway_event`,
`make_cloud_boundary_event`, `make_energy_floor_event`, `make_velocity_sign_event`,
`make_cooling_balance_event`), four per-phase builders (`build_energy_phase_events`,
`build_implicit_phase_events`, `build_transition_phase_events`, `build_momentum_phase_events`), a
classifier `check_event_termination(sol, events) -> EventResult`, and a state write-back
`apply_event_result(params, result, t, y, state_keys=['R2','v2'])`. Default keyword arguments
visible in the signatures: `energy_floor: float = 1000.0`, `threshold: float = 0.05`,
`y_index: int = 2` (energy floor), `y_index: int = 1` (velocity sign), `direction: str = 'collapse'`.
From `y_index=1 → v2` and `y_index=2 → E_b` the integrated state in the energy-type phases is
`y = [R2, v2, E_b]`; `state_keys=['R2','v2']` implies the momentum-type phase integrates `y = [R2, v2]`.

---

## 1. The physically correct phase sequence

### 1.1 The phases

A feedback bubble driven by a coeval cluster inside a spherical GMC passes through the following
**dynamical regimes**. I distinguish *physical* phases (a different equation is being solved) from
*numerical* phases (the same equation, a different solution strategy) — the distinction is the first
correctness question, because only physical phases may be reported as science.

| # | Phase | Physics | State vector | Driver |
|---|---|---|---|---|
| P0 | **Free expansion** | wind free-streams, `M_sw ≪ M_ej`, `R ∝ t` at `~v_w` | — | wind ram |
| P1 | **Energy-driven** (Weaver) | hot shocked-wind reservoir, `t_cool,b ≫ t_dyn`, near-isobaric `P_b = E_b/[2π(R2³−R1³)]` | `[R2, v2, E_b]` | `4πR2² max(P_b, P_HII)` |
| P1′ | **Implicit** | *identical physics to P1*; a numerical continuation (SPEC-010) | `[R2, v2, E_b]` | same as P1 |
| P2 | **Transition** | bubble radiating away its reservoir; driver migrates continuously (SPEC-016) | `[R2, v2, E_b]` | `4πR2² max(P_b, P_HII + P_ram)` |
| P3 | **Momentum-driven** | reservoir gone; only momentum + `P_HII` + radiation act | `[R2, v2]` | `4πR2² (P_HII + P_ram)` |

P0 is normally *not* integrated: the energy-driven similarity solution is an attractor (SPEC-056),
so a code may start on the P1 branch at small `R0, v0`. That is legitimate **only if** the start
state is inside the basin and no event is armed that the start state already satisfies (§7, trap B).

P1′ is a **numerical** phase. Consequences that must hold:
- it must carry the *same physical termination criteria* as P1 (SPEC-010: it is the energy phase);
- crossing P1 ↔ P1′ must change no physical quantity — the boundary is invisible in `R2`, `v2`,
  `E_b`, `dv2/dt`, `P_drive`;
- it must be merged with `energy` for any published phase-duration statistic (the shipped figure
  scripts already map `implicit → energy`, SPEC-010), i.e. the split must never leak into science.

**Re-collapse is a fate, not a phase** (SPEC-017). The integrator's `current_phase` enum must be
`{energy, implicit, transition, momentum}` only; `collapse` is a post-processing label applied to the
part of the final `momentum` interval after the interpolated `v2 = 0` crossing.

### 1.2 The graph

```
                 ┌──────────────── (numerical strategy switch, physics-neutral) ───┐
                 ↓                                                                 │
 [start P1: energy] ⇄ [P1′: implicit] ──(cooling balance | ebpeak | blowout)──→ [P2: transition]
                 │                                                                 │
                 │                                                    (E_b → 0, i.e. P_b ≪ P_HII+P_ram)
                 │                                                                 ↓
                 └──────────── (any phase) ───────────────────────────────→ [P3: momentum]
                                    │
                                    ↓
   terminal fates (reachable from ANY phase):
     dispersal · escape · stall · feedback exhausted · re-collapse
                                    │
                          (re-collapse only)
                                    ↓
     new star-formation episode: ii_coll → ii_coll+1, phase machine RESET to P1
```

**Legal transitions (within one feedback episode, i.e. one cluster / one `ii_coll`):**
`P1 → P1′`, `P1′ → P1`, `P1|P1′ → P2`, `P2 → P3`, and `P1|P1′ → P3` directly (a degenerate
zero-length transition phase is legal if the transition criterion and the energy-floor criterion are
met at the same instant — e.g. a blowout that instantaneously depressurises the bubble).

**Illegal within an episode (under the standard WARPFIELD-lineage simplification):**
`P3 → P2`, `P3 → P1`, `P2 → P1`. Rationale: energy → momentum is *thermodynamically irreversible*
under the modelling assumptions. Once the shocked wind has radiated, re-establishing an adiabatic
reservoir requires either a jump in `L_mech` or a drop in bubble density; in a 1-D thin-shell model
with monotonically declining `L_w` and a bubble that only gets denser as the shell decelerates,
neither is available.

**Physically, re-entry is not impossible** and this is the one place where the standard
simplification is genuinely questionable [medium confidence, internal knowledge]: the onset of
core-collapse SNe at `t ≈ 3–4 Myr` raises `L_mech` by 1–2 dex over the wind-only value, and the
freshly shocked ejecta can have `t_cool > t_dyn` again — the "reborn superbubble". A code may
therefore legitimately allow `P3 → P1`. What it must **not** do is allow re-entry *by accident*:

> **Requirement.** Whichever choice is made, the phase label must be **monotone non-decreasing
> within an episode**, or re-entry must be an explicitly modelled, recorded decision guarded by
> hysteresis. The energy↔momentum criterion is a threshold on a ratio (SPEC-014) evaluated on a
> non-differentiable `max()` RHS (SPEC-023); an un-hystereticised threshold on a chattering quantity
> will flip phases repeatedly, producing many zero-length phases, a phase-duration histogram that is
> pure numerics, and (if each flip re-initialises the bubble structure) a slow, non-reproducible run.

**Across episodes** (`expansion_next(..., ii_coll)`): re-collapse to `coll_r` ends the episode and,
in the WARPFIELD lineage, triggers a *new* star-formation event on the recollapsed gas. That new
episode must **reset** the phase machine to P1, reset the SPS clock to a new cluster age zero, reset
the bubble state (`E_b`, `R1`, interior profile), and carry forward only: the remaining gas mass, the
accumulated stellar mass from previous generations (it still gravitates and still radiates), and the
absolute simulation time. Carrying a stale bubble structure or a stale phase label into episode `n+1`
is a silent-corruption bug of exactly the kind §6 describes.

---

## 2. Transition criteria — the quantity, the sign, the direction, and terminal-vs-diagnostic

### 2.0 The governing numerics (this is what makes the rest concrete)

An event-driven ODE phase machine detects a phase change as a **root of a scalar residual**
`g(t, y)`. Three properties are non-negotiable:

1. **`g` must be continuous and must actually change sign** across the event. A residual that only
   *touches* zero (tangential) or that is piecewise constant (a boolean indicator, a value bound at
   phase start) is **undetectable** — a bracketing root-finder needs `g(t_k)·g(t_{k+1}) < 0`.
2. **`direction` must match the physical crossing.** `direction = +1` detects only `−→+`,
   `−1` only `+→−`, `0` both. A direction opposite to the physical crossing makes the event
   *silently never fire*: no error, no warning, the run simply continues to some other stopping
   condition and reports **that** as the outcome.
3. **`terminal` in the solver sense means "stop this `solve_ivp` call"**, which is *not* the same as
   "stop the simulation". A phase machine needs **two** independent booleans: `ends_segment` and
   `ends_run`. Collapsing them into one is the single most damaging structural error available here
   (§7, trap D/G).

Also: `g` must be evaluable from `(t, y)` plus the *current* auxiliary solve, and must be finite for
every reachable state (no division by a quantity that can vanish).

### 2.1 Table of required events

`✔seg` = must stop the current integration segment. `✔run` = must stop the whole simulation.
`✘run` = **must not** stop the simulation — a diagnostic/monitor.

| Event | Residual `g(t,y)` | Sign while "not yet" | Crossing | `seg` | `run` | Kind |
|---|---|---|---|---|---|---|
| **cooling balance** (P1→P2) | `(L_gain − L_loss)/L_gain − θ`, θ≈0.05 | `> 0` | ↓ `direction = −1` | ✔ | ✘run | phase transition |
| **ebpeak** (P1→P2) | `Ė_b = L_gain − L_loss` | `> 0` | ↓ `−1` | ✔ | ✘run | phase transition |
| **blowout** (P1→P2 option) | `R2 − r_cloud` | `< 0` | ↑ `+1` | ✔ | ✘run | phase transition |
| **energy floor** (P2→P3) | `E_b − E_floor` | `> 0` | ↓ `−1` | ✔ | ✘run | phase transition |
| **cloud boundary** (RHS switch) | `R2 − r_cloud` | either | **both, `0`** | ✔ | ✘run | numerical restart + diagnostic |
| **velocity sign** (turnaround) | `v2` | `> 0` | ↓ `−1` | optional | **✘run** | diagnostic |
| **velocity sign** (bounce) | `v2` | `< 0` | ↑ `+1` | optional | ✘run | diagnostic |
| **min radius** — *physical* | `R2 − coll_r` | `> 0` | ↓ `−1` | ✔ | ✔run | **physical fate: re-collapse** |
| **min radius** — *guard* | `R2 − r_floor,num` | `> 0` | ↓ `−1` | ✔ | ✔run | **numerical failure** |
| **max radius** | `R2 − stop_r` | `< 0` | ↑ `+1` | ✔ | ✔run | **numerical cutoff** |
| **velocity runaway (expansion)** | `v_max,exp − v2` | `> 0` | ↓ `−1` | ✔ | ✔run | **numerical failure** |
| **velocity runaway (collapse)** | `v2 + v_max,coll` | `> 0` | ↓ `−1` | ✔ | ✔run | **numerical failure** |
| **dissolution** | `n_sh,max − n_ISM` held `< 0` for `stop_t_diss` | `> 0` | ↓ + dwell | ✔ | ✔run | physical fate |
| **stall** | *not an event* — see §2.6 | — | — | — | ✔run | physical fate |
| **feedback exhausted** | `t_cluster − t_SPS,max` | `< 0` | ↑ `+1` | ✔ | ✔run | model-domain limit |
| **t > stop_t** | solver `t_span` end | — | — | ✔ | ✔run | **numerical cutoff** |

### 2.2 Energy → transition

**Physical statement (SPEC-013).** The hot bubble ceases to be an energy reservoir:
`L_loss/L_gain → 1`, equivalently `(L_gain − L_loss)/L_gain → 0`, equivalently `t_cool,b < t_dyn`.
`L_gain = η_w L_mech,w + η_SN L_mech,SN`; `L_loss = L_cool + L_conduction-front + L_leak`
(SPEC-035, SPEC-036).

Required properties of the residual:

- **State-dependence (critical).** `L_gain` and `L_loss` are functionals of the *current* bubble
  structure — they depend on `E_b(t)`, `R2(t)`, `T_b`, the cooling table, and the cluster age. The
  residual must therefore be recomputed at every solver step from `(t, y)`. A residual whose
  `L_gain`/`L_loss` are **scalars bound once at phase entry** is a *constant function of `t` and
  `y`* — it can never change sign, so the transition it is supposed to detect is **invisible to the
  detector**. The signature `def factory(Lgain: float, Lloss: float)` returning `def event(t, y)`
  is precisely the shape that produces this degeneracy, so this is the highest-priority check in
  the slice. Symptom if violated: the energy phase never ends on its own and always terminates on
  some *other* event (`stop_t`, `max_radius`, an energy floor), while the reported transition time
  becomes an artefact of whichever guard fired.
- **Sign and direction.** `g = 1 − L_loss/L_gain − θ` is positive in the adiabatic regime and
  decreases as cooling catches up ⇒ `direction = −1`. Using `direction = +1` or `0` risks firing on
  the *recovery* branch (e.g. at SN onset, when `L_gain` jumps and `g` crosses upward), i.e.
  transitioning to the momentum phase exactly when the bubble becomes *more* energy-driven.
- **Finiteness.** `L_gain` passes through zero or near-zero in the wind/SN gap of a low-mass
  cluster's SPS track. `g = (L_gain − L_loss)/L_gain` then diverges to `−∞` and, worse, changes sign
  spuriously as `L_gain` crosses zero. The residual must be formed as `L_gain(1−θ) − L_loss`
  (multiply through — same root, same sign, no pole) with an explicit `L_gain > 0` guard, or the
  event must be disarmed while `L_gain ≤ 0`.
- **Threshold semantics.** `θ = 0.05` is a *numerical regularisation of a "→0" statement*
  (SPEC-014); the transition **time** is threshold-dependent. This must be recorded per run, and any
  published transition time carries a systematic from `θ`.
- **`ebpeak` is the same residual at `θ = 0`** and therefore fires strictly *later* whenever
  `(L_gain − L_loss)/L_gain` declines monotonically. A "whichever fires first" composite of
  `{cooling_balance(0.05), ebpeak}` is therefore always won by `cooling_balance`; the `ebpeak`
  branch only matters if it is used alone.
- **`blowout` is a different mechanism** (geometric, SPEC-014 Reading C), and it is `R2 − r_cloud`
  with `direction = +1`. It must not be conflated with the thermal criterion in the recorded reason.

**Terminal?** Segment-terminal **yes**; run-terminal **no**. A cooling-balance event that ends the
run is the archetypal "monitor treated as a termination" bug: every run would report an outcome at
the energy→transition time and no run would ever reach a physical fate.

### 2.3 Transition → momentum

**Physical statement.** The bubble no longer contributes to the shell's driving pressure:
`P_b ≪ P_HII + P_ram`. Since `P_b = E_b/[2π(R2³−R1³)]` (SPEC-024), the natural residual is on `E_b`,
but the *correct* criterion is **dimensionless and relative**, e.g.

```
    g = (P_HII + P_ram) · ε_P  −  P_b          (direction +1 as P_b falls below the branch)
or  g = E_b − ε_E · E_b,peak                   (direction −1)
```

An **absolute** floor (`energy_floor: float = 1000.0` in code units, i.e. ≈ 1.9×10⁴⁶ erg using
SPEC-091's `1 M⊙ pc² Myr⁻² = 1.90148e43 erg`) is *scale-dependent*: TRINITY's shipped grids span
`mCloud` from ~10⁵ to 5×10⁹ M⊙ (SPEC-103), i.e. ~4.7 dex in `L_mech` and hence in `E_b`. One
absolute number cannot be the right relative smallness across that span — at the top of the grid the
floor is reached far too late (the code integrates a `P_b` that is already dynamically irrelevant),
and at the bottom it may be satisfied *at phase entry*, giving a zero-length transition phase.
Both failure modes are silent.

- **Sign/direction.** `g = E_b − E_floor` is positive while the reservoir survives ⇒ `direction = −1`.
- **`y_index` correctness.** `y_index = 2` selects `E_b` **only if the phase that arms this event
  integrates `[R2, v2, E_b]`**. Arming the same factory in a phase whose state is `[R2, v2]` reads
  out of range or (worse, if the state is longer for a different reason) reads an unrelated
  component. The builder must assert the state layout it assumes.
- **Not satisfiable at entry.** The builder must assert `E_b(t_entry) > E_floor`; otherwise the
  event roots at the first step and the transition phase has zero duration (§7, trap B).
- **Terminal?** Segment-terminal yes; run-terminal **no**.

### 2.4 The cloud boundary — two distinct jobs, two different directions

`R2 = r_cloud` is where `ρ_amb` switches from the cloud profile to `n_ISM` (SPEC-021, SPEC-060).
This is a **discontinuity in the ODE right-hand side**, so the integrator must be *stopped and
restarted* there regardless of any physics: an adaptive stiff solver that steps across an RHS jump
silently loses its order estimate and its error control is meaningless over that step. For that job
the direction must be **`0` (both)**, because a shell that has crossed outward can turn around and
re-enter the cloud, hitting the same discontinuity from the other side.

For the *bookkeeping* job (`stop_at_rCloud_nSnap`: terminate `N` snapshots after the crossing) the
relevant crossing is the **first outward** one, `direction = +1`.

**Crossing `r_cloud` is not escape** (SPEC-104): a shell can cross with `v2 < v_esc(R2)` and fall
back. Therefore:
- `cloud_boundary` must be **run-terminal only when the user explicitly asks** (`stop_at_rCloud_nSnap`
  set); its default must be non-terminal-for-run;
- when it *is* used to stop, the recorded outcome must be `"user cutoff at cloud edge"`, never
  `"escape"` or `"dispersal"`.

### 2.5 Velocity sign — a diagnostic that must not stop the run

`v2 = 0` is the turnaround. Per SPEC-017 the `collapse` label is *post-processing* on the momentum
interval, which means the integration must **continue through** `v2 = 0` into the infall. Hence:

- run-terminal: **NO**. If `velocity_sign` is run-terminal, the code can never reach the re-collapse
  fate, can never reach `coll_r`, and can never trigger `expansion_next`'s next episode
  (`ii_coll+1`). Every gravity-dominated run would report "stopped at turnaround" — and a grid built
  from those runs would show **zero re-collapses**, i.e. the headline dispersal/recollapse boundary
  would be a pure artefact.
- segment-terminal: optional and harmless (restarting at the turnaround is numerically clean), but
  then `apply_event_result` must hand off `v2 = 0` exactly and the next segment must not re-fire the
  same root at the same `t` (§7, trap H).
- `direction = −1` for turnaround, `+1` for bounce. If a single event is used with `direction = +1`
  to mark "collapse begins", the physical crossing (positive → negative) is invisible.
- `v2 = 0` may also be the **initial condition** (a shell started from rest). Then the event's
  residual is exactly zero at `t0` and fires on the first step. The builder must either start with
  `v0 > 0` strictly or disarm this event for the first step.

### 2.6 Stall — provably not detectable by a sign-change event

A stall is `v2 → 0` **with `dv2/dt → 0`** (SPEC-032/SPEC-100): a pressure-supported equilibrium. The
residual `g = v2` then approaches zero **tangentially and never changes sign**. No bracketing root
finder can detect it — `g(t_k)·g(t_{k+1}) > 0` for every step. This is the "event function degenerate
at the very condition it detects" trap in its purest form.

A stall must therefore be detected by a **dwell condition**, not an event:

```
    |v2| < v_tol   AND   |dv2/dt| < a_tol   sustained for Δt ≥ t_dwell
```

with `v_tol` referred to a physical scale (e.g. `c_s,ISM`, or the ambient sound speed — SPEC-102
Reading B makes exactly this point for dissolution) rather than an absolute pc/Myr number. Without a
dwell test, a stalled run does not stop: it grinds along at `v2 ≈ 0` until `stop_t` and is then
reported as a **time cutoff**, i.e. a genuine physical fate is filed as a numerical one. The same
structure applies to dissolution, which SPEC-101 already specifies with a dwell (`stop_t_diss`).

### 2.7 Numerical guards (must be terminal, must never be reported as fates)

- **min radius.** Two *different* thresholds live here and they must not share an outcome label:
  (i) `coll_r` — the physical "cloud has re-collapsed" radius; (ii) a numerical floor
  (`MIN_RADIUS_FACTOR·something + MIN_RADIUS_SAFETY`) protecting the `1/R2²` terms in `F_grav`
  (SPEC-031) and `P_b ∝ 1/(R2³−R1³)` (SPEC-024) from divergence. Both are `direction = −1`. Both are
  run-terminal. Their **outcomes are different in kind**: (i) is a physical fate that feeds
  `expansion_next(ii_coll+1)`; (ii) is a solver-protection abort. Conflating them makes a grid's
  "recollapse fraction" contaminated by wherever the ODE happened to become stiff.
- **max radius (`stop_r`).** `direction` must match the residual's sign convention: `R2 − stop_r`
  needs `+1`, `stop_r − R2` needs `−1`. A mismatch means the event never fires and the run ends on
  `stop_t` instead — the outcome silently changes from "radius cutoff" to "time cutoff", and every
  downstream duration statistic changes with it. `stop_r` is a **numerical cutoff** (SPEC-100), never
  escape.
- **velocity runaway.** Two branches with **different residual forms**:
  expansion `g = v_max,exp − v2` (fires when `v2` exceeds the cap, `direction = −1`);
  collapse `g = v2 + v_max,coll` (fires when `v2 < −v_max,coll`, `direction = −1`).
  Using the *expansion* form for the collapse branch gives `g = v_max − v2 → +∞` as `v2 → −∞` — the
  runaway is **invisible**, and the run instead produces `R2 < 0`, then NaNs, then whatever the NaN
  handling reports. Physical sanity anchors for the caps: the shell can never exceed the wind
  terminal speed (`~1–3 × 10³ km/s`), and infall can never exceed free fall,
  `v_ff = sqrt(2G(M_*+M_sh)/R2)` — 293 km/s for `10⁷ M⊙` inside 1 pc using
  `G = 4.30091e-3 pc M⊙⁻¹ (km/s)²` (SPEC-091). A cap tighter than the physical bound turns a real
  regime into a spurious failure; a cap looser than it lets NaNs form first.

---

## 3. Terminal fates

### 3.1 Physical outcomes and their criteria

| Fate | Criterion | Why it is physical |
|---|---|---|
| **Dispersal / dissolution** | `n_sh,max < n_ISM` sustained for `stop_t_diss` (SPEC-101/102); or, better posed, the shell becomes subsonic/pressure-confined `\|v2\| < c_s,ISM` and fragments | the shell is no longer distinguishable from ambient |
| **Re-collapse** | `v2 < 0` sustained **and** bound: `½v2² < G(M_* + M_sh)/R2`, integrated down to `R2 ≤ coll_r` | gravity has won; feeds the next SF episode |
| **Escape / blowout** | `R2 > r_cloud` **and** `v2 > v_esc(R2) = sqrt(2G(M_*+M_sh)/R2)` (SPEC-032/104) | the shell will not return |
| **Stall** | `\|v2\| < v_tol` and `\|dv2/dt\| < a_tol` sustained (§2.6) | force balance at finite radius |
| **Feedback exhausted** | cluster age exceeds the SPS table span | the driver is gone — *but see below* |

`Feedback exhausted` is a **model-domain limit** rather than a clean physical fate: past the end of
the SPS table the code has no driver data, and any extrapolation is not physics. It must be reported
as a distinct third category ("model domain exceeded"), because a downstream reader will otherwise
interpret it as "the cluster stopped pushing", which is a physical claim the run did not test.

### 3.2 Numerical outcomes (never fates)

`t > stop_t`; `R2 > stop_r`; `stop_at_rCloud_nSnap` cutoff; min-radius numerical floor; velocity
runaway; non-finite state (`NaN`/`Inf` in `y` or in the RHS); solver non-convergence
(`solve_ivp` `status < 0`, root-finder failure in the implicit bubble-structure solve, maximum
step-count or maximum-segment-count exceeded); the monotonic-guard rejection the project already
knows about (`CLAUDE.md`: the bubble-structure integrator's monotonic guard rejects certain
floating-point output).

### 3.3 Why the distinction is load-bearing

The published science *is* the outcome classification: the dispersal-vs-recollapse boundary in the
`(M_cl, n_core, ε)` grid, and the phase durations in the timeline figure. Two concrete corruptions
if numerical outcomes are filed as physical ones:

1. **The boundary becomes a stiffness map.** The solver gets stiffest exactly where the physics is
   most interesting — high `n_core` (fast cooling, short `t_cool`), high `ε` (violent early
   expansion), small `R2` (the `1/R2²` and `1/(R2³−R1³)` terms). If solver aborts in that corner are
   labelled `recollapse`, the paper's conclusion "clouds denser than X re-collapse" is a restatement
   of "the integrator fails above density X". The correlation is perfect and undetectable from the
   outputs alone.
2. **Phase durations become truncation artefacts.** A run aborted at `t = 2 Myr` of a 15 Myr budget
   reports a momentum phase of length 0 and an energy fraction of 100%. Averaged over a grid, the
   mean energy-phase fraction rises with whatever makes runs fail early. This is the same hazard the
   project's own working rule 5 names: runs truncate at different `t`, so any cross-run comparison
   must be at **matched simulation time**, and a truncation must be *visible* for that matching to
   be possible at all.

**Requirement.** The recorded outcome must carry a **kind** field with at least
`{physical_fate, model_domain, numerical_cutoff, solver_failure}`, and the exit code must be
non-zero for the last two. A downstream script must be able to filter to `physical_fate` without
parsing free text.

---

## 4. Required continuity across a phase boundary

Let `t*` be the event root. The hand-off must satisfy:

### 4.1 Must be continuous (bit-level carry, not recomputation)

| Quantity | Why |
|---|---|
| `t` | The next segment starts at exactly `t*`, taken from the **dense-output root**, not from the last accepted step `sol.t[-1]`. Using the last step leaves a gap `[t*, t_last]` integrated with the wrong phase's RHS (or re-integrates it, duplicating snapshots). |
| `R2` | Geometry cannot jump. A jump changes `M_sh(R2)` and hence every force. |
| `v2` | Momentum cannot jump: no impulsive force exists in the model. A `v2` discontinuity is an injection/removal of kinetic energy `½M_sh Δ(v2²)` with no source. |
| `E_b` across P1→P1′ and P1→P2 | The reservoir is a physical state variable. |
| absolute time / cluster age / SPS clock | The driver `L_mech(t)`, `Q_i(t)`, `ṗ(t)` must be continuous. Restarting the clock at 0 at a phase boundary re-runs the cluster's youth. |
| swept mass `M_sh` and its integral history | Must be *the same value*, whether carried or recomputed from `R2`. |

### 4.2 Must be recomputed, never carried

| Quantity | Why |
|---|---|
| `R1 = sqrt(ṗ_w/(4πP_b))` (SPEC-025) | It is an algebraic function of the new phase's `P_b`; carrying it makes `V_b` and hence `P_b` self-inconsistent. |
| `P_b = E_b/[2π(R2³−R1³)]` (SPEC-024) | Algebraic in `(E_b, R2, R1)`. |
| bubble interior profile, `T0`, `(α, β, δ)` | These are outputs of the implicit structure solve at the *current* state; a stale profile silently sets the first steps of the new phase to the previous phase's structure. `(α, β, δ)` in particular are *logarithmic time derivatives* (SPEC-041) and are meaningless if evaluated with the old segment's `t`. |
| `P_HII`, `P_ram`, `f_abs`, `τ_IR` | Algebraic in the current state. |
| the ambient-density branch (cloud profile vs `n_ISM`) | Must be re-evaluated at `R2(t*)` (see §2.4). |

### 4.3 May legitimately jump — but must be recorded

- **`P_drive` and hence `dv2/dt`.** SPEC-022's phase-aware prescription changes the driver from
  `max(P_b, P_HII)` (energy) to `max(P_b, P_HII + P_ram)` (transition) to `P_HII + P_ram`
  (momentum). If the non-bubble branch is active on both sides of the P1→P2 boundary, `P_drive`
  jumps by exactly `+P_ram` — a *built-in* discontinuity in the acceleration. SPEC-016 requires
  continuity in `dv2/dt` (audit test T13); the honest position is that a jump is permissible only if
  it is **documented and quantified per run**. The magnitude of the jump at each boundary should be
  a recorded diagnostic.
- **`E_b` at P2→P3.** The momentum-phase state is `[R2, v2]` — `E_b` is dropped. Whatever residual
  energy `E_b(t*)` remained is removed from the universe at that instant. That is defensible as a
  modelling statement ("it has been radiated"), but the amount must go into an explicit ledger, not
  vanish, or the total energy budget (`injected = radiated + kinetic + thermal + gravitational`)
  will not close and SPEC-051's `E_b/(L_mech t) → 5/11` test cannot be applied across the boundary.

### 4.4 Invariants that must hold at `t*` evaluated from both sides

1. **Force closure** (SPEC-007): `M_sh dv2/dt = F_drive + F_rad − F_grav − F_ram,amb − 4πR2²P_ISM`,
   each side using its own `P_drive`.
2. **Monotone time**: `t_start(new) ≥ t_end(old)`, with no snapshot at a `t` already written and no
   `t` rewind in `dictionary.jsonl`.
3. **Single-valued state**: at most one snapshot per `t`; if both the old segment's last point and
   the new segment's first point are written, they must be *identical* in `R2`, `v2`, `E_b`.
4. **Phase-label monotonicity** within an episode (§1.2).
5. **Ionizing-photon closure** (SPEC-028): `f_gas + f_dust + f_esc = 1` must still hold at `t*`.

### 4.5 `apply_event_result`

Its job is exactly this hand-off. Requirements implied by its signature:

- it must write **the state at the event root**, obtained from dense output at `t*` — not
  `sol.y[:, -1]`;
- `state_keys` must cover **every** component the next phase needs. A default of `['R2','v2']`
  is correct only for a hand-off into the momentum phase; a hand-off into or within an energy-type
  phase must also write `E_b`. If `E_b` is silently not written, the next segment restarts from a
  stale (phase-entry) `E_b` — a hidden energy injection whose size grows with the segment length;
- it must be idempotent and must not partially write: if it raises after writing `R2` but before
  `v2`, the shared params object is left inconsistent and the next phase integrates a state that
  never existed.

---

## 5. Failure-reporting requirements

When a segment ends because the **solver** failed rather than because physics said so, a trustworthy
run must record, in machine-readable form (SPEC-105 already mandates a `termination` block with
`{exit_code, outcome, detail, timestamp, model_name}`, a `final_state`, and a `termination_debug`):

1. **Kind** — `solver_failure`, distinct from `physical_fate` / `numerical_cutoff` /
   `model_domain` (§3.3).
2. **Which detector fired, or that none did** — the event **name** (the factories all take a `name`,
   so the name must survive into the record), the residual value at the root, the crossing
   direction, and `t_root`. If no event fired, record the integrator's own `status`/`message`/
   `success`, the last accepted `t` and `y`, the last step size, and the step count.
3. **The failing sub-solve** — for the implicit bubble-structure path, the residual at abort, the
   iteration count, the bracket, and which guard rejected the state (the project already has a
   monotonic guard that rejects certain floating-point output).
4. **A non-finite inventory** — which fields are `NaN`/`Inf` at abort and at the previous snapshot
   (SPEC-105 names this `termination_debug`), plus the last-two-snapshot diff so a reader can see
   what was diverging.
5. **A closure check at the last state** — force closure (SPEC-007) and photon closure (SPEC-028)
   residuals, which distinguish "the integrator lost accuracy" from "the model state was already
   unphysical".
6. **A non-zero process exit code** and a `success: false` flag, so a sweep driver cannot count the
   cell as complete.
7. **The truncation time relative to the requested span** (`t_last / stop_t`), which is what a
   matched-time comparison needs.

**What a downstream analysis wrongly concludes if these are conflated** (concretely, for this code):

- The phase-timeline figure splits the final `momentum` interval at the interpolated `v2 = 0` and
  labels the remainder `collapse` (SPEC-017). A run that *aborted* while `v2` happened to be
  negative therefore acquires a `collapse` bar. The figure asserts a physical outcome that the run
  never demonstrated.
- A grid's dispersal-vs-recollapse map becomes a map of solver stiffness (§3.3, item 1).
- Phase-duration statistics inherit early-truncation bias (§3.3, item 2), and the reported
  energy→momentum transition time — the quantity most sensitive to `θ` and to the cooling-boost
  knobs (SPEC-014/015) — becomes a mixture of a physics threshold and an abort time.
- Force-closure and photon-closure tests (SPEC-007, SPEC-028) applied to a corpus of runs will show
  spurious violations clustered in the aborted runs, and if the aborts are invisible, the tests
  themselves will be blamed and loosened.

---

## 6. Global-state requirements

**Independence requirement.** For any two configurations A and B, the outputs of
`run(A); run(B)` in one process must be **byte-identical** to `run(A)` and `run(B)` each in a fresh
process — and to `run(B); run(A)`. Order-independence and process-independence are the whole test.
(The project already asserts this is *not* currently true: `CLAUDE.md` states trinity "leaks
module-level global state in-process" and mandates separate-process comparison. That makes this
section a live requirement, not a hypothetical.)

**Classes of module-level state that silently couple runs:**

1. **A module-level mutable params/ODEpar singleton** written by the orchestration and by
   `apply_event_result`. If the phase machine's state lives on a module global rather than on an
   object owned by the run, run B inherits run A's `current_phase`, `ii_coll`, `rCloud`, or last
   `y`. This is the highest-risk class here because `apply_event_result(params, ...)` is explicitly
   a *mutation* of a shared object.
2. **Caches keyed incompletely.** An SPS-table or cooling-table cache keyed on file path but not on
   `M_cluster`/`Z`/age-grid, or a memoised `rCloud`, returns run A's values to run B. A `lru_cache`
   on a function taking a params object hashes by identity, so mutating the object in place gives
   wrong cache hits within a single run too.
3. **Accumulating logger handlers.** Adding a file handler per run without removing it makes run B
   write into run A's log and duplicates every message — and, if a handler holds an open file, the
   first run's log never closes.
4. **Module-level counters and flags** — `ii_coll`, "warning already emitted", "table already
   loaded", a step counter, a segment counter. Any of these carried across runs changes behaviour
   (e.g. a suppressed warning, an episode cap already consumed).
5. **Mutable default arguments** (`def f(x, cache={})`) — the classic.
6. **Global RNG state** (`np.random.seed` at module import or in a run) — makes any stochastic
   element order-dependent.
7. **Global numeric/plotting configuration** — `np.seterr`, `warnings.filterwarnings`,
   `matplotlib.rcParams` set at import and mutated per run.

**Sweep consequence.** `run.py --workers N` makes this a correctness issue for published grids: if
workers are threads, or a process pool that reuses workers for multiple cells, leaked globals couple
cells and the grid result depends on scheduling. The requirement is either strict per-run state
ownership, or one fresh process per cell (which the `--emit-jobs`/SLURM path gives for free —
another reason the two execution paths must be verified to agree).

---

## 7. Known traps (each one, instantiated for this interface)

**A. Degenerate event — never changes sign at the very condition it detects.**
Two instances here. (i) `make_cooling_balance_event` via `factory(Lgain: float, Lloss: float)`: if
the two luminosities are *bound as scalars* when the event is built, the returned `event(t, y)` is
constant, so the transition is undetectable — the phase never ends on physics (§2.2). (ii) **Stall**
as a root of `v2`: `v2 → 0` with `dv2/dt → 0` is tangential, never brackets, and is undetectable in
principle (§2.6). *Detection test:* evaluate each event residual over the phase's actual trajectory
and assert it takes both signs.

**B. Criterion already satisfied at `t = 0` (or at phase entry).**
Instances: (i) `min_radius` with `coll_r = 1 pc` (SPEC-101) versus a start radius that for a compact
cluster is typically `≪ 1 pc` — the run would report "collapsed" at `t ≈ 0`; (ii) `energy_floor`
armed in the energy phase, where `E_b(t0) ≈ 0` by construction; (iii) `velocity_sign` when the shell
starts from rest (`v0 = 0` is a root); (iv) `cooling_balance` at early times, when the bubble is
small, dense, and genuinely radiative, so `(L_gain−L_loss)/L_gain` can already be `≤ θ`. Required:
each builder asserts its residual is strictly the "not yet" sign at phase entry, or **latches** the
event (arm it only after the residual has first been comfortably on the correct side —
`MIN_RADIUS_FACTOR`/`MIN_RADIUS_SAFETY` are the right shape for exactly this, applied to the
*achieved maximum* radius rather than to an absolute constant).

**C. Sign convention makes the physical crossing invisible to a direction-sensitive detector.**
Instances: (i) collapse runaway written in the expansion form (`v_max − v2 → +∞` as `v2 → −∞`,
never fires — §2.7); (ii) `max_radius` residual written as `stop_r − R2` while `direction = +1`
(never fires; run silently ends on `stop_t` instead); (iii) `cooling_balance` with `direction = +1`,
which fires on the SN-onset *recovery* rather than the decline. All three are silent — nothing
errors, the outcome label just changes.

**D. A diagnostic monitor treated as a termination.**
Instances: `velocity_sign` run-terminal (no run can ever reach re-collapse or the `ii_coll+1`
episode; the grid shows zero recollapses — §2.5); `cloud_boundary` run-terminal by default (crossing
`r_cloud` is not escape, SPEC-104, and terminating there forecloses the very question the code
exists to answer); `cooling_balance`/`energy_floor` run-terminal (every run reports an outcome at a
phase boundary and no run ever reaches a physical fate). This is the failure the slice brief calls
out: *a code that cannot tell a monitor from a terminal event ends runs early on a monitor.*

**E. A terminal event firing on the wrong crossing direction.**
`velocity_sign` with `direction = +1` used as the collapse marker fires on the **bounce**
(`−→+`), i.e. on re-expansion, so a shell that turns around and then re-accelerates outward is
recorded as "collapse detected" at the moment it started expanding again.

**F. Boolean / piecewise-constant residuals.**
An event that returns `1.0` or `0.0` (or `float(condition)`) has no bracketable root; the root
finder either fails or returns a step boundary. The reported event time is then a function of the
solver's step sequence, so it changes with `rtol`/`atol` and is not reproducible. Every residual must
be a **continuous real-valued distance to the condition**.

**G. Conflating `terminal` (segment) with "run over".**
`EventResult` / `check_event_termination` must carry both `ends_segment` and `ends_run`, plus the
event `name` and `kind`. A single boolean cannot express "the energy phase ended, start the
transition phase".

**H. Repeated root / zero-length segments.**
If a segment is restarted at `t*` with a state that still satisfies the event, the same event roots
immediately and the orchestrator loops. Required: advance `t` by at least the root tolerance,
disarm the just-fired event for the first step of the new segment, and cap the segment count with a
**distinct** failure reason (`too many phase segments`) rather than silently producing thousands of
zero-length phases.

**I. Simultaneous events resolved by list order rather than by earliest root.**
When a step brackets two roots, the orchestrator must select the **smallest `t_root`**. Selecting the
first item of the event list means a guard listed early can mask a physical transition that occurred
earlier, and the recorded phase sequence becomes an artefact of list construction order.

**J. `stop_r` racing `r_cloud`.**
If `stop_r` is not comfortably larger than `r_cloud`, a run that is asked to record `N` snapshots
after crossing the cloud edge (`stop_at_rCloud_nSnap`) will hit the radius cutoff first and produce
fewer than `N` — silently, with the outcome recorded as a radius cutoff. `_check_stop_r_rCloud_
interaction(nSnap_rCloud, stop_r, rCloud)` is the right place to catch this, and it must be a
**fail-fast configuration validation at startup** (before hours of compute), comparing `stop_r`
against `r_cloud` *plus the radial extent the requested snapshots will span* — not merely against a
fixed multiplicative factor, since the post-crossing expansion rate varies by orders of magnitude
across the shipped grid. `rCloud` itself must be the value the solver will use (SPEC-005: post-SFE
vs total mass changes it by 11% at `ε = 0.3`), or the check validates the wrong number.

**K. Event root tolerance vs published precision.**
Phase durations are a published quantity. The root-finder tolerance on `t*` must be tighter than the
precision at which transition times are reported, and the transition time must additionally carry
the `θ = 0.05` systematic (SPEC-014) — the threshold is a regularisation, not physics.

---

## 8. Confidence notes

- **High confidence** (first-principles or numerics-of-event-detection): §2.0, §2.6 (tangential
  stall), §3 (fate vs failure), §4.1–4.2 (continuity/recompute), §7 A/C/D/E/F/G/I.
- **Medium**: the specific WARPFIELD-lineage claim that energy→momentum is one-way within an
  episode, and that re-collapse triggers a new SF episode with `ii_coll` counting generations
  (internal knowledge of Rahner+17/19; literature unreachable).
- **Medium/low**: the numeric interpretation of `energy_floor = 1000.0` as ≈1.9×10⁴⁶ erg — it
  depends on the code's internal energy unit being `M⊙ pc² Myr⁻²` (SPEC-090/091). The *structural*
  objection (an absolute floor cannot be scale-correct across 4.7 dex in cloud mass) holds
  regardless of the unit.

```json
[
  {
    "id": "S11-C-01",
    "file": "trinity/phase_general/phase_events.py",
    "line": 341,
    "class": "numerical",
    "severity": "S1",
    "claim": "The cooling-balance residual must be recomputed from the current state at every solver step; L_gain and L_loss must not be bound as constants when the event is constructed.",
    "evidence": "An ODE event is detected only by a sign change of g(t,y) between successive steps. L_gain and L_loss are functionals of the instantaneous bubble structure (E_b, R2, T_b, cluster age; SPEC-013, SPEC-035). If they are captured as scalars at phase entry, g is identically constant in t and y and can never change sign, so the transition it exists to detect is invisible to the detector. The signature `def factory(Lgain: float, Lloss: float)` returning `def event(t, y)` is exactly the shape that produces this degeneracy.",
    "expected": "event(t, y) evaluates L_gain(t,y) and L_loss(t,y) (or a closure over a live state object updated every step) and returns a residual that provably takes both signs along the phase trajectory.",
    "failure_scenario": "The energy phase never ends on physics. It ends instead on whatever guard fires first (stop_t, max_radius, energy floor), so the published energy->momentum transition time and the energy-phase duration are artefacts of a numerical cutoff, not of SPEC-013 physics.",
    "repro": "Evaluate the cooling-balance residual along a stored energy-phase trajectory (param/simple_cluster.param) and assert it takes both signs; assert it is not constant in t.",
    "confidence": "high"
  },
  {
    "id": "S11-C-02",
    "file": "trinity/phase_general/phase_events.py",
    "line": 319,
    "class": "sign",
    "severity": "S2",
    "claim": "The cooling-balance residual must be positive while the bubble is energy-driven and cross zero downward; direction must be -1.",
    "evidence": "g = (L_gain - L_loss)/L_gain - theta is ~1 in the adiabatic limit (L_loss << L_gain) and declines toward theta as cooling catches up (SPEC-013/014). The physical event is the decreasing crossing.",
    "expected": "direction = -1 (decreasing) on a residual whose 'still energy-driven' sign is positive.",
    "failure_scenario": "With direction = +1 the event fires on the recovery branch instead - e.g. at SN onset, when L_gain jumps by 1-2 dex and the residual crosses upward - so the code declares the end of the energy phase exactly when the bubble becomes more energy-driven. With direction = 0 both crossings fire and the phase machine chatters.",
    "repro": "",
    "confidence": "high"
  },
  {
    "id": "S11-C-03",
    "file": "trinity/phase_general/phase_events.py",
    "line": 342,
    "class": "divergence",
    "severity": "S2",
    "claim": "The cooling-balance residual must be finite for every reachable state, including L_gain -> 0.",
    "evidence": "L_gain = eta_w L_mech,w + eta_SN L_mech,SN (SPEC-035) passes through a deep minimum between the wind era and SN onset for a low-mass cluster. Dividing by L_gain then gives a pole and a spurious sign change at the pole rather than at the physical root.",
    "expected": "Form the residual as L_gain*(1-theta) - L_loss (same root, sign-preserving for L_gain>0, no pole), with an explicit guard that disarms the event while L_gain <= 0.",
    "failure_scenario": "A spurious transition fires at the L_gain zero-crossing, ending the energy phase at a time set by the SPS table's wind/SN gap rather than by cooling. Alternatively an inf/NaN residual makes the root-finder fail and the run aborts with no physical reason.",
    "repro": "Evaluate the residual at a state with L_gain = 0 and assert it is finite.",
    "confidence": "high"
  },
  {
    "id": "S11-C-04",
    "file": "trinity/phase_general/phase_events.py",
    "line": 363,
    "class": "state",
    "severity": "S1",
    "claim": "Phase-transition events (cooling_balance, ebpeak, blowout, energy_floor) must end the current integration segment but must NOT end the simulation; EventResult must express 'ends segment' and 'ends run' as two independent fields.",
    "evidence": "solve_ivp's terminal flag means 'stop this integration call'. A phase machine needs to stop the call and then start the next phase. The physically meaningful end states (SPEC-100) are dispersal, re-collapse, escape, stall, feedback exhausted - none of which is a phase boundary.",
    "expected": "check_event_termination returns an EventResult carrying at least {name, kind in {phase_transition, physical_fate, numerical_cutoff, solver_failure}, t_root, residual value, direction, ends_segment, ends_run}; ends_run is False for every phase-transition event.",
    "failure_scenario": "Every run terminates at the energy->transition boundary and reports that as its outcome. No run reaches the momentum phase, no run reaches a physical fate, the phase-timeline figure has only one bar, and the dispersal/recollapse grid is empty of both.",
    "repro": "",
    "confidence": "high"
  },
  {
    "id": "S11-C-05",
    "file": "trinity/phase_general/phase_events.py",
    "line": 252,
    "class": "state",
    "severity": "S1",
    "claim": "make_energy_floor_event's y_index must match the state layout of the phase that arms it, and the builder must assert that layout.",
    "evidence": "y_index=2 selects E_b only if the arming phase integrates [R2, v2, E_b]. apply_event_result's default state_keys=['R2','v2'] shows at least one phase integrates a 2-component state. Indexing y[2] of a 2-component state raises; indexing y[2] of a longer state with a different layout silently reads an unrelated variable.",
    "expected": "Each build_*_phase_events asserts len(y) and the index-to-name mapping it assumes before constructing index-based events.",
    "failure_scenario": "The energy-floor event compares an unrelated state component against an energy threshold. It either never fires (transition phase runs to another cutoff) or fires immediately (zero-length transition phase), and in both cases the recorded phase boundary is meaningless while the run appears to succeed.",
    "repro": "",
    "confidence": "high"
  },
  {
    "id": "S11-C-06",
    "file": "trinity/phase_general/phase_events.py",
    "line": 504,
    "class": "regime",
    "severity": "S2",
    "claim": "The transition->momentum criterion must be relative/dimensionless (P_b <= eps*(P_HII+P_ram), or E_b <= eps*E_b,peak), not an absolute energy floor.",
    "evidence": "The physical statement is 'the bubble no longer contributes to P_drive' (SPEC-022). E_b scales with L_mech*t, i.e. with cluster mass; the shipped grids span mCloud ~1e5 to 5e9 Msun (SPEC-103), about 4.7 dex. One absolute constant (build_transition_phase_events default energy_floor = 1000.0, i.e. ~1.9e46 erg using SPEC-091's 1 Msun pc^2 Myr^-2 = 1.90148e43 erg) cannot represent the same physical smallness across that span.",
    "expected": "A relative criterion, with the absolute floor retained only as a numerical backstop reported under a different outcome label.",
    "failure_scenario": "At the massive end the code integrates a dynamically irrelevant P_b for a long time, inflating the transition-phase duration; at the low-mass end the floor is met at phase entry and the transition phase has zero length. Both bias the published phase-duration statistics systematically WITH cloud mass - i.e. exactly along the axis the grid is varying.",
    "repro": "",
    "confidence": "medium"
  },
  {
    "id": "S11-C-07",
    "file": "trinity/phase_general/phase_events.py",
    "line": 274,
    "class": "numerical",
    "severity": "S2",
    "claim": "The energy-floor residual must be strictly positive at transition-phase entry, and the event must not be armed in a phase where E_b starts near zero.",
    "evidence": "g = E_b - E_floor with direction -1. In the energy phase E_b starts at ~0 by construction (the bubble has not yet been inflated), so the residual is already negative at t0 and the first step registers the crossing.",
    "expected": "build_transition_phase_events asserts E_b(t_entry) > energy_floor; the energy-phase builder does not arm this event at all.",
    "failure_scenario": "A zero-length transition phase (or an immediate energy->momentum jump at t~0), reported as a physical phase transition. The run then integrates the whole evolution in the momentum phase with P_drive = P_HII + P_ram, silently discarding TRINITY's energy-driven physics while producing a complete, plausible-looking output file.",
    "repro": "",
    "confidence": "high"
  },
  {
    "id": "S11-C-08",
    "file": "trinity/phase_general/phase_events.py",
    "line": 287,
    "class": "silent-failure",
    "severity": "S1",
    "claim": "The velocity-sign (v2 = 0) event is a diagnostic; it must not terminate the run.",
    "evidence": "SPEC-017: 'collapse' is constructed in post-processing as the part of the final momentum interval after the interpolated v2=0 crossing - which requires the integration to continue THROUGH v2 = 0. Re-collapse as a fate (SPEC-100) requires integrating the infall down to coll_r, and expansion_next(..., ii_coll) requires that episode to complete.",
    "expected": "ends_run = False for velocity_sign in every phase builder. Segment-terminal is acceptable (a clean restart at turnaround) provided the hand-off is exact and the root does not immediately re-fire.",
    "failure_scenario": "Every gravity-dominated run stops at turnaround and is recorded as ending there. The re-collapse fate becomes unreachable, the dispersal/re-collapse boundary in the published grid shows zero re-collapses, the 'collapse' bar in the phase-timeline figure is always empty, and expansion_next's multi-generation path is dead code that never executes.",
    "repro": "Run a configuration that must re-collapse (high mCloud, low sfe) and assert the output contains snapshots with v2 < 0.",
    "confidence": "high"
  },
  {
    "id": "S11-C-09",
    "file": "trinity/phase_general/phase_events.py",
    "line": 306,
    "class": "sign",
    "severity": "S2",
    "claim": "If velocity_sign marks the onset of collapse, its direction must be -1 (positive to negative); +1 detects the bounce, not the turnaround.",
    "evidence": "Turnaround is v2 crossing zero from + to -. Re-expansion after a bounce is the - to + crossing. These are physically opposite events sharing one root.",
    "expected": "direction = -1 for a turnaround marker; direction = +1 only if a bounce is what is wanted; direction = 0 only if both are wanted and the event is non-terminal.",
    "failure_scenario": "A shell that turns around and later re-accelerates outward is recorded as 'collapse detected' at the instant it resumed expanding - the recorded collapse onset time is wrong by the whole duration of the infall, and a run that never collapses at all can be labelled as collapsing.",
    "repro": "",
    "confidence": "high"
  },
  {
    "id": "S11-C-10",
    "file": "trinity/phase_general/phase_events.py",
    "line": 287,
    "class": "numerical",
    "severity": "S2",
    "claim": "Stall must be detected by a dwell condition, not by a sign-change event; a stall is a tangential zero of v2 and is undetectable by any bracketing root finder.",
    "evidence": "A stall is v2 -> 0 WITH dv2/dt -> 0 (SPEC-032, SPEC-100): the residual g = v2 approaches zero without changing sign, so g(t_k)*g(t_k+1) > 0 for every step and no root is ever bracketed. Contrast dissolution, for which SPEC-101 already prescribes a dwell (stop_t_diss).",
    "expected": "A separate check: |v2| < v_tol AND |dv2/dt| < a_tol sustained for a dwell time, with v_tol referred to a physical scale (e.g. the ambient sound speed), reported as outcome 'stall'.",
    "failure_scenario": "A stalled run does not stop; it grinds at v2 ~ 0 until stop_t and is filed as a TIME CUTOFF. A genuine physical outcome (pressure-supported equilibrium at finite radius) is recorded as a numerical limit, so the outcome census under-counts stalls to zero and over-counts 'ran out of time'.",
    "repro": "",
    "confidence": "high"
  },
  {
    "id": "S11-C-11",
    "file": "trinity/phase_general/phase_events.py",
    "line": 99,
    "class": "silent-failure",
    "severity": "S1",
    "claim": "The physical collapse threshold (coll_r) and the numerical small-radius guard (MIN_RADIUS_FACTOR / MIN_RADIUS_SAFETY) must produce different recorded outcomes.",
    "evidence": "coll_r = 1 pc is declared as 'radius below which the cloud is considered completely collapsed' (SPEC-101/103) - a physical fate that feeds the next star-formation episode. A numerical floor exists for a different reason: F_grav ~ 1/R2^2 (SPEC-031) and P_b ~ 1/(R2^3 - R1^3) (SPEC-024) both diverge as R2 -> 0. Same residual shape, same direction, completely different meaning.",
    "expected": "Two events with distinct names and distinct outcome kinds: 'recollapse' (physical_fate) and 'min-radius guard' (solver_failure / numerical_cutoff).",
    "failure_scenario": "A grid's re-collapse fraction is contaminated by cells where the ODE merely became singular. Because the guard trips preferentially in the stiffest corner of parameter space (small R2, high n_core), the contamination is perfectly correlated with the physical axis being studied, so the paper's dispersal/re-collapse boundary is partly a map of where the integrator breaks - and this is undetectable from the output files alone.",
    "repro": "",
    "confidence": "high"
  },
  {
    "id": "S11-C-12",
    "file": "trinity/phase_general/phase_events.py",
    "line": 120,
    "class": "sign",
    "severity": "S2",
    "claim": "The min-radius residual must be positive above the threshold and its direction must be -1.",
    "evidence": "g = R2 - min_r is positive while the shell is outside the threshold and crosses zero downward as the shell collapses inward. direction = +1 would detect only the outward re-crossing.",
    "expected": "g = R2 - min_r, direction = -1, terminal.",
    "failure_scenario": "The inward crossing is invisible: the integrator continues past R2 = min_r toward R2 -> 0, where 1/R2^2 and 1/(R2^3-R1^3) overflow. The run ends on NaN/Inf or on a velocity runaway instead of on a clean collapse, and the physical re-collapse outcome is lost.",
    "repro": "",
    "confidence": "high"
  },
  {
    "id": "S11-C-13",
    "file": "trinity/phase_general/phase_events.py",
    "line": 99,
    "class": "numerical",
    "severity": "S1",
    "claim": "The min-radius event must not be satisfiable at the start of the run; it must be latched or referred to the achieved maximum radius rather than to an absolute constant.",
    "evidence": "coll_r defaults to 1 pc (SPEC-101) while a compact cluster's initial shell radius is typically far below 1 pc. g = R2 - coll_r is then negative at t0 and roots on the first step. MIN_RADIUS_FACTOR / MIN_RADIUS_SAFETY have the right shape for a latch but only if referred to a state-dependent scale.",
    "expected": "Arm the event only after R2 has first exceeded the threshold by a margin (a latch), or define min_r as a fraction of the peak radius achieved so far.",
    "failure_scenario": "Every compact-cluster run terminates at t ~ 0 reporting 'collapsed'. The output is a complete, well-formed file with a physical-sounding outcome and essentially no evolution in it; a grid built from these reports 100% re-collapse in exactly the regime where the initial radius is smallest.",
    "repro": "Assert R2(t0) > min_r at every build_*_phase_events call site.",
    "confidence": "high"
  },
  {
    "id": "S11-C-14",
    "file": "trinity/phase_general/phase_events.py",
    "line": 134,
    "class": "silent-failure",
    "severity": "S2",
    "claim": "max_radius (stop_r) is a numerical run limit and must be recorded as such, never as escape, blowout or dispersal.",
    "evidence": "SPEC-100 lists 'R2 > stop_r' explicitly under 'Numerical cutoff - not physics'. Escape requires R2 > r_cloud AND v2 > v_esc(R2) (SPEC-032/104); stop_r knows nothing about either.",
    "expected": "outcome kind = numerical_cutoff, with a non-physical outcome label and a note of stop_r's value.",
    "failure_scenario": "Runs that were merely truncated at 500 pc are counted as 'cloud dispersed / shell escaped'. Because stop_r is hit first by the fastest-expanding (highest-feedback) cells, the escape fraction acquires a spurious dependence on sfe and mCloud, which is precisely the published result.",
    "repro": "",
    "confidence": "high"
  },
  {
    "id": "S11-C-15",
    "file": "trinity/phase_general/phase_events.py",
    "line": 152,
    "class": "sign",
    "severity": "S3",
    "claim": "The max-radius residual's sign convention and its direction must agree: R2 - max_r needs direction +1; max_r - R2 needs direction -1.",
    "evidence": "A direction-sensitive detector ignores crossings of the wrong sense. Nothing errors when they disagree - the event simply never fires.",
    "expected": "Residual and direction consistent, with a unit test that drives R2 past max_r and asserts the event fires.",
    "failure_scenario": "The radius cutoff silently never triggers; the run continues to stop_t and is recorded as a TIME cutoff. Every duration and final-state statistic changes, and the stop_r parameter appears to have no effect - which a user would interpret as 'the shell never got that big'.",
    "repro": "",
    "confidence": "high"
  },
  {
    "id": "S11-C-16",
    "file": "trinity/phase_general/phase_events.py",
    "line": 166,
    "class": "sign",
    "severity": "S2",
    "claim": "The collapse and expansion branches of the velocity-runaway event need different residual forms: collapse g = v2 + v_max_collapse, expansion g = v_max_expansion - v2, both with direction -1.",
    "evidence": "In collapse v2 is negative and runs to -inf. The expansion-form residual v_max - v2 then runs to +inf and never crosses zero, so the runaway is invisible. Only v2 + v_max crosses zero downward as v2 falls below -v_max.",
    "expected": "Branch-specific residuals with matching direction; a test per branch that drives v2 past each cap.",
    "failure_scenario": "An infall runaway is never caught. R2 goes negative within one step, then P_b ~ 1/(R2^3-R1^3), F_grav ~ 1/R2^2 and any sqrt() of a now-negative quantity produce NaN. The run ends on a NaN with no named event, and the failure is attributed to whatever generic handler catches it rather than to the runaway.",
    "repro": "",
    "confidence": "high"
  },
  {
    "id": "S11-C-17",
    "file": "trinity/phase_general/phase_events.py",
    "line": 166,
    "class": "silent-failure",
    "severity": "S2",
    "claim": "Velocity runaway is a numerical-failure detector, not a physical fate; the caps must be referred to physical bounds and the outcome must carry a failure exit code.",
    "evidence": "Physical bounds: the shell cannot exceed the wind terminal speed (~1e3-3e3 km/s), and infall cannot exceed free fall v_ff = sqrt(2G(M_*+M_sh)/R2) = 293 km/s for 1e7 Msun inside 1 pc with G = 4.30091e-3 pc Msun^-1 (km/s)^2 (SPEC-091). A cap tighter than the physical bound converts a real regime into a spurious failure; looser, and NaNs form first.",
    "expected": "kind = solver_failure, non-zero exit code, and caps documented against the physical bounds above.",
    "failure_scenario": "If reported as a fate, a numerically diverging run enters the outcome census as a physical result. If the cap is too tight, physically valid high-velocity regimes (high sfe, low mCloud) are systematically excluded from the grid without any record that they were excluded.",
    "repro": "",
    "confidence": "medium"
  },
  {
    "id": "S11-C-18",
    "file": "trinity/phase_general/phase_events.py",
    "line": 220,
    "class": "silent-failure",
    "severity": "S2",
    "claim": "The cloud-boundary event must not end the run by default, and when stop_at_rCloud_nSnap does end it the outcome must be labelled a user cutoff, not escape.",
    "evidence": "SPEC-104: crossing r_cloud means the shell entered the ambient ISM; it does not mean escape - a shell can cross with v2 < v_esc and turn around. stop_at_rCloud_nSnap defaults to None (do not stop), and terminating at the crossing forecloses the dispersal-vs-recollapse question the code exists to answer.",
    "expected": "ends_run = False by default; when the user sets stop_at_rCloud_nSnap, outcome kind = numerical_cutoff with detail naming the parameter.",
    "failure_scenario": "Every run stops at the cloud edge, so no run ever demonstrates escape or return. The published escape fraction becomes 'fraction of clouds whose shell reached r_cloud', which is a much weaker and different statement, presented as the stronger one.",
    "repro": "",
    "confidence": "high"
  },
  {
    "id": "S11-C-19",
    "file": "trinity/phase_general/phase_events.py",
    "line": 239,
    "class": "numerical",
    "severity": "S3",
    "claim": "R2 = rCloud is a discontinuity in the ODE right-hand side and must stop and restart the integration, with direction = 0 so inward re-crossings are caught too.",
    "evidence": "rho_amb switches from the cloud profile to n_ISM at r_cloud (SPEC-021, SPEC-060) - a jump in M_sh' and in the ram term. An adaptive stiff solver stepping across an RHS jump has no valid error estimate over that step. A shell that has crossed outward can turn around and re-enter, hitting the same jump from the other side, so a +1-only detector misses half the crossings.",
    "expected": "A segment-terminal, run-non-terminal event with direction = 0 for the RHS restart, plus a separate +1 first-crossing marker for stop_at_rCloud_nSnap bookkeeping.",
    "failure_scenario": "Silent loss of accuracy at the cloud edge - no error, no warning, just a step whose error estimate is meaningless. Since the ambient density drops by orders of magnitude there, this is where the trajectory is most sensitive, so the post-crossing radius law (the quantity compared against t^{3/5} and t^{1/2} in the published figure) inherits an uncontrolled error.",
    "repro": "",
    "confidence": "high"
  },
  {
    "id": "S11-C-20",
    "file": "trinity/phase_general/phase_events.py",
    "line": 363,
    "class": "numerical",
    "severity": "S2",
    "claim": "When more than one event roots within a step, the orchestrator must select the earliest root, not the first entry in the event list.",
    "evidence": "A solver returns all events that crossed in a step, each with its own root time. Physical ordering is by t_root; list order is an artefact of how the builder assembled the list.",
    "expected": "Select min(t_root) across fired events; on an exact tie, prefer the physical-fate/phase-transition event over a numerical guard and record that a tie occurred.",
    "failure_scenario": "A guard listed early (max_radius, min_radius) masks a phase transition or a physical fate that actually occurred earlier in the same step. The recorded phase sequence and outcome then depend on the order in which build_*_phase_events appended events - a property no one reviews and that changes whenever the builder is edited.",
    "repro": "",
    "confidence": "high"
  },
  {
    "id": "S11-C-21",
    "file": "trinity/phase_general/phase_events.py",
    "line": 82,
    "class": "state",
    "severity": "S1",
    "claim": "EventResult must classify the outcome by kind - {phase_transition, physical_fate, model_domain, numerical_cutoff, solver_failure} - and must carry the event name, the root time, the residual value and the crossing direction.",
    "evidence": "SPEC-100 separates physical end states from 'Numerical cutoff - not physics'. SPEC-105 requires metadata.json to record termination = {exit_code, outcome, detail, ...}. A downstream census must be able to filter to physical outcomes without parsing free text.",
    "expected": "A structured result with a kind enum and an exit code that is non-zero for numerical_cutoff and solver_failure.",
    "failure_scenario": "Outcome classification collapses into a single free-text string. A sweep-level analysis cannot distinguish 'the cloud re-collapsed' from 'the integrator gave up while v2 happened to be negative', so the paper's dispersal/re-collapse map silently mixes the two and its boundary tracks solver stiffness rather than physics.",
    "repro": "",
    "confidence": "high"
  },
  {
    "id": "S11-C-22",
    "file": "trinity/phase_general/phase_events.py",
    "line": 588,
    "class": "state",
    "severity": "S1",
    "claim": "apply_event_result must write the dense-output state evaluated at the event root, and must write every state component the next phase integrates - including E_b when the next phase is energy-type.",
    "evidence": "The last accepted solver step lies beyond the root; using sol.y[:, -1] hands over a state at the wrong t, leaving [t_root, t_last] either unintegrated or double-integrated with the wrong phase's RHS. state_keys defaulting to ['R2','v2'] is complete only for a hand-off into the momentum phase; energy-type phases integrate [R2, v2, E_b] (implied by energy-floor y_index = 2).",
    "expected": "y taken from dense output at t_root; state_keys derived from the target phase's declared state layout, not from a default constant.",
    "failure_scenario": "If E_b is not written, the next segment restarts from a stale E_b (typically the previous phase's entry value). That is an unaccounted energy injection or removal whose magnitude grows with segment length; it breaks SPEC-035's energy equation and SPEC-051's E_b/(L_mech t) -> 5/11 test, and it does so silently because R2 and v2 remain continuous and the trajectory looks smooth.",
    "repro": "Assert E_b is continuous across every phase boundary in dictionary.jsonl to solver tolerance.",
    "confidence": "high"
  },
  {
    "id": "S11-C-23",
    "file": "trinity/main.py",
    "line": 366,
    "class": "state",
    "severity": "S2",
    "claim": "Across a phase hand-off, t, R2, v2 and E_b must be carried continuously, while R1, P_b, T0, (alpha,beta,delta), P_HII, P_ram and the ambient-density branch must be recomputed from the new state.",
    "evidence": "R2 and v2 cannot jump (no impulsive force). R1 = sqrt(pdot_w/(4 pi P_b)) (SPEC-025) and P_b = E_b/[2 pi (R2^3 - R1^3)] (SPEC-024) are algebraic in the current state, so carrying them makes V_b and P_b mutually inconsistent. (alpha, beta, delta) are logarithmic time derivatives (SPEC-041) and are meaningless if evaluated against the previous segment's clock.",
    "expected": "Explicit recomputation of all algebraic/structural quantities at phase entry; a continuity assertion on t, R2, v2, E_b.",
    "failure_scenario": "A stale bubble structure sets the first steps of the new phase, so P_b (hence P_drive, hence the whole trajectory) is wrong at the boundary by an amount no diagnostic reports. Because the error is injected exactly at the phase transition, it contaminates the quantity the paper measures - the transition time and the post-transition radius law.",
    "repro": "Sample R1, P_b, T0 immediately either side of each boundary and check they satisfy their algebraic definitions at the boundary state.",
    "confidence": "high"
  },
  {
    "id": "S11-C-24",
    "file": "trinity/main.py",
    "line": 366,
    "class": "divergence",
    "severity": "S3",
    "claim": "Any jump in dv2/dt across a phase boundary must be quantified and recorded, not assumed absent.",
    "evidence": "SPEC-022's phase-aware driver changes from max(P_b, P_HII) to max(P_b, P_HII + P_ram) to P_HII + P_ram. If the non-bubble branch is active on both sides of the energy->transition boundary, P_drive jumps by exactly +P_ram - a genuine discontinuity in acceleration. SPEC-016 and audit test T13 require continuity in dv2/dt to integrator tolerance.",
    "expected": "Record the two-sided dv2/dt at every boundary as a diagnostic; if the jump exceeds integrator tolerance, document it as a modelling discontinuity rather than letting T13 fail silently.",
    "failure_scenario": "An unrecorded acceleration jump means the trajectory has a kink whose size depends on which max() branch was active - so two runs differing infinitesimally in parameters can differ finitely in post-transition radius, and the parameter dependence of the published radius law acquires a discontinuity that looks like physics.",
    "repro": "",
    "confidence": "medium"
  },
  {
    "id": "S11-C-25",
    "file": "trinity/main.py",
    "line": 366,
    "class": "state",
    "severity": "S3",
    "claim": "The residual E_b discarded when the state contracts from [R2,v2,E_b] to [R2,v2] at transition->momentum must be recorded in an energy ledger.",
    "evidence": "The momentum phase does not carry E_b (implied by apply_event_result's state_keys default). Whatever E_b remained at the boundary leaves the budget at that instant. SPEC-035's energy equation and SPEC-051's partition test both require the budget to close.",
    "expected": "A recorded 'E_b discarded at transition' quantity in metadata/final_state, and a check that it is small relative to the integrated injected energy.",
    "failure_scenario": "The total energy budget silently fails to close across the boundary. A reader applying the SPEC-051 test (E_b/(L_mech t) -> 5/11) or auditing energy conservation finds a deficit and cannot tell whether it is a modelling choice or a bug - and if the discarded amount is large, the momentum phase begins from a state the energy equation never sanctioned.",
    "repro": "",
    "confidence": "medium"
  },
  {
    "id": "S11-C-26",
    "file": "trinity/main.py",
    "line": 216,
    "class": "state",
    "severity": "S2",
    "claim": "Within one feedback episode the phase label must be monotone non-decreasing (energy/implicit -> transition -> momentum), or any re-entry into an energy-driven phase must be an explicit, hysteretic, recorded decision.",
    "evidence": "Energy->momentum is thermodynamically irreversible under the model's assumptions; the WARPFIELD lineage treats it as one-way [medium confidence, literature unreachable]. Physically re-entry is possible at SN onset, when L_mech jumps 1-2 dex and t_cool can again exceed t_dyn - so a code may allow it, but the criterion is a threshold (SPEC-014, theta = 0.05) evaluated on a non-differentiable max() RHS (SPEC-023), which will chatter without hysteresis.",
    "expected": "Either a strictly monotone phase machine, or re-entry gated on the criterion being satisfied with a margin and sustained for a dwell time, with each re-entry recorded.",
    "failure_scenario": "The phase label oscillates at the threshold, producing dozens of zero-length phases. Phase-duration statistics become a measure of solver step size; if each flip re-initialises the bubble structure, the run also becomes slow and non-reproducible across tolerance settings.",
    "repro": "Assert the phase sequence in dictionary.jsonl is monotone within an episode and that no phase has zero duration.",
    "confidence": "medium"
  },
  {
    "id": "S11-C-27",
    "file": "trinity/main.py",
    "line": 216,
    "class": "other",
    "severity": "S4",
    "claim": "current_phase must take only the integrator phase values {energy, implicit, transition, momentum}; 'collapse' is a post-processing label.",
    "evidence": "SPEC-017: the collapse bar in the published timeline is constructed by splitting the final momentum interval at the interpolated v2 = 0 crossing. It is a fate, not an integration phase.",
    "expected": "The phase enum excludes 'collapse'; the post-processing derives it.",
    "failure_scenario": "If the solver also emits a 'collapse' phase, the post-processing split double-counts or mis-slices the final interval, and the published phase-fraction figure disagrees with the raw output for reasons no one can trace.",
    "repro": "",
    "confidence": "high"
  },
  {
    "id": "S11-C-28",
    "file": "trinity/main.py",
    "line": 366,
    "class": "state",
    "severity": "S2",
    "claim": "expansion_next must fully reset the phase machine for each new episode (ii_coll) - phase label to energy, cluster/SPS clock to a new zero, bubble state (E_b, R1, interior profile) cleared - while carrying forward remaining gas mass, cumulative stellar mass and absolute time; and the episode count must be capped with its own outcome reason.",
    "evidence": "A new star-formation event after re-collapse is a new cluster: its SPS drivers start at age zero, and the previous bubble no longer exists. Previous generations still gravitate (they enter M_cluster in SPEC-031 and v_esc in SPEC-032) and still radiate. Absolute simulation time must not rewind or stop_t becomes meaningless.",
    "expected": "An explicit per-episode state object; a max-episodes cap reported as its own outcome rather than as a physical fate.",
    "failure_scenario": "A stale phase label or stale bubble state carried into episode n+1 makes the second generation start in the momentum phase with the first generation's E_b, so second-generation feedback is silently weaker. Cumulative stellar mass omitted from M_cluster makes gravity too weak and every later episode too easy to disperse. An uncapped episode loop turns a marginally-bound cloud into an unbounded run.",
    "repro": "",
    "confidence": "medium"
  },
  {
    "id": "S11-C-29",
    "file": "trinity/main.py",
    "line": 41,
    "class": "other",
    "severity": "S3",
    "claim": "_check_stop_r_rCloud_interaction must be a fail-fast startup validation comparing stop_r against rCloud plus the radial extent the requested post-crossing snapshots will span, using the same rCloud the solver will use.",
    "evidence": "stop_at_rCloud_nSnap asks for N snapshots after R2 > r_cloud (SPEC-101). If stop_r is not comfortably larger than r_cloud the radius cutoff wins the race and fewer than N snapshots are produced - silently. A single multiplicative factor cannot bound this because the post-crossing expansion rate varies by orders of magnitude across the shipped grid. SPEC-005 additionally warns that r_cloud differs by 11% at eps = 0.3 depending on whether it is normalised by total or post-SFE mass.",
    "expected": "A startup error or explicit warning naming both parameters, evaluated against the rCloud actually used by the solver.",
    "failure_scenario": "A sweep silently produces cells with incomplete post-crossing sampling, all of them in the same corner of parameter space (small clouds, where r_cloud approaches stop_r). Any statistic computed over the post-crossing window is then biased in a parameter-correlated way, and the run is recorded as a clean radius cutoff with no indication that the requested data are missing. Discovered only after hours of compute if not checked at startup.",
    "repro": "",
    "confidence": "medium"
  },
  {
    "id": "S11-C-30",
    "file": "trinity/main.py",
    "line": 81,
    "class": "state",
    "severity": "S1",
    "claim": "Two runs executed in the same process must produce byte-identical output to the same two runs executed in separate processes, and in either order.",
    "evidence": "The project's own CLAUDE.md records that trinity leaks module-level global state in-process and therefore mandates separate-process comparison. Coupling classes: a module-level mutable params/ODEpar singleton (apply_event_result mutates a shared params object), caches keyed incompletely (SPS/cooling tables, memoised rCloud), accumulating logger handlers, module-level counters (ii_coll, 'warning already emitted'), mutable default arguments, global RNG seed, and global numpy/matplotlib configuration.",
    "expected": "All run state owned by an explicit per-run object; module-level state either absent or reset at run start; a test that runs A then B in one process and compares against separate-process baselines.",
    "failure_scenario": "run.py --workers N with a reusing pool couples sweep cells, so a published grid depends on scheduling order and is not reproducible. Worse, the coupling is state-dependent (a cached rCloud or a leaked phase label), so a cell's result can change depending on which cell ran before it - and nothing in the output records which that was.",
    "repro": "python run.py A; python run.py B (separate processes) vs one process running A then B; diff dictionary.jsonl byte-for-byte.",
    "confidence": "high"
  },
  {
    "id": "S11-C-31",
    "file": "trinity/main.py",
    "line": 216,
    "class": "silent-failure",
    "severity": "S1",
    "claim": "A phase that ends because the solver failed must record kind = solver_failure with the solver status/message, the last accepted t and y, the step count and last step size, the failing sub-solve's residual, a NaN/Inf inventory, the closure residuals, a non-zero exit code, and t_last/stop_t - and must never emit a physical outcome label.",
    "evidence": "SPEC-105 already requires termination = {exit_code, outcome, detail, ...} plus final_state and termination_debug with a last-two-snapshot diff and a NaN/Inf inventory. SPEC-100 separates physical fates from numerical cutoffs. The closure invariants SPEC-007 (forces) and SPEC-028 (photon budget) distinguish 'the integrator lost accuracy' from 'the state was already unphysical'.",
    "expected": "All of the above recorded; exit_code != 0; outcome drawn from a failure vocabulary disjoint from the physical-fate vocabulary.",
    "failure_scenario": "The post-processing that splits the final momentum interval at v2 = 0 (SPEC-017) attaches a 'collapse' bar to any run that aborted while v2 happened to be negative - asserting a physical outcome the run never demonstrated. Grid-level dispersal/re-collapse fractions then track solver stiffness, and phase-duration means inherit early-truncation bias, both invisibly. Because runs truncate at different t, cross-run comparison at 'final state' also compares different physical times.",
    "repro": "",
    "confidence": "high"
  },
  {
    "id": "S11-C-32",
    "file": "trinity/phase_general/phase_events.py",
    "line": 82,
    "class": "numerical",
    "severity": "S2",
    "claim": "Every event residual must be a continuous real-valued distance to its condition, never a boolean or an indicator.",
    "evidence": "A bracketing root finder needs g(t_k)*g(t_k+1) < 0 and a continuous g to interpolate the root. A residual returning 0.0/1.0 (or float(condition)) is piecewise constant: either no bracket exists, or the 'root' returned is a step boundary.",
    "expected": "Residuals of the form (quantity - threshold), continuous in t and y over the phase's domain.",
    "failure_scenario": "The reported event time becomes a function of the solver's step sequence, so it changes with rtol/atol and is not reproducible. Since the energy->momentum transition time is a published quantity, its value would depend on integrator settings rather than on physics - and a tolerance change intended as a convergence check would move the science result.",
    "repro": "",
    "confidence": "high"
  },
  {
    "id": "S11-C-33",
    "file": "trinity/main.py",
    "line": 216,
    "class": "numerical",
    "severity": "S3",
    "claim": "A restarted segment must not immediately re-fire the event that ended the previous one; repeated roots at the same t must be detected and the segment count capped with its own failure reason.",
    "evidence": "The hand-off state satisfies the event that just fired (it is the root). Re-arming the same event without advancing past the root reproduces it on the first step of the new segment.",
    "expected": "Advance t past the root by at least the root tolerance, or disarm the just-fired event for the first step; cap the number of segments and report 'too many phase segments' as a solver_failure.",
    "failure_scenario": "An infinite loop of zero-length segments, or thousands of duplicate snapshots at the same t in dictionary.jsonl. Downstream time-series analysis sees repeated timestamps, interpolation becomes ill-posed, and phase-duration accounting reports many phases of length zero.",
    "repro": "Assert strictly increasing t and no zero-length phase intervals in dictionary.jsonl.",
    "confidence": "high"
  },
  {
    "id": "S11-C-34",
    "file": "trinity/phase_general/phase_events.py",
    "line": 458,
    "class": "regime",
    "severity": "S2",
    "claim": "The implicit phase must arm the same physical transition criteria as the energy phase; it is a numerical continuation, not a distinct physical regime.",
    "evidence": "SPEC-010 documents implicit as 'a numerical continuation of the energy phase', merged into energy for display by both shipped figure scripts. Same physics implies the same exit criteria (cooling balance / ebpeak / blowout) plus its own numerical-failure detectors, which must be reported under a different kind.",
    "expected": "build_implicit_phase_events includes the transition criteria that build_energy_phase_events includes; the extra callable it returns is a solver-strategy hook, not a physics gate.",
    "failure_scenario": "If the implicit phase omits the cooling-balance criterion, a run that enters implicit before the transition condition is met can never transition on physics and instead ends on a numerical guard. Since implicit is merged into energy for display, the resulting over-long energy phase is invisible in the published timeline - it just looks like the bubble stayed energy-driven longer, which is exactly the quantity SPEC-015 says a 1-D code already over-predicts.",
    "repro": "",
    "confidence": "medium"
  },
  {
    "id": "S11-C-35",
    "file": "trinity/phase_general/phase_events.py",
    "line": 363,
    "class": "numerical",
    "severity": "S4",
    "claim": "The event-root tolerance must be tighter than the precision at which phase-transition times are reported, and reported transition times must carry the threshold systematic.",
    "evidence": "Phase durations are a published quantity. The root is located by interpolation on dense output to a finite tolerance, and separately the criterion itself is a numerical regularisation of a 'to zero' statement (SPEC-014, theta = 0.05, any value in 0.01-0.2 equally defensible).",
    "expected": "Root tolerance recorded and tighter than reporting precision; theta recorded per run alongside every transition time.",
    "failure_scenario": "A transition time is quoted to a precision the root finder does not deliver, and its dominant uncertainty (the arbitrary 5% threshold) is not propagated - so a threshold-sensitivity difference between two studies reads as a physics disagreement.",
    "repro": "",
    "confidence": "medium"
  }
]
```
