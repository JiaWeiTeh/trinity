# S6 transition + momentum — Lens B (what the code claims)

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

Prose-only transcription. I have not seen one line of implementation. Everything below is a
**claim the code makes about itself**, recorded so a code-reading lens can test it. Where I write
"contradiction", I mean *prose contradicts prose* — never "the code is wrong", which I cannot know.

Files in slice:
- A = `trinity/phase1c_transition/run_transition_phase.py`
- B = `trinity/phase2_momentum/run_momentum_phase.py`

These two are near-twins. A large fraction of B's comments are word-for-word copies of A's. That
makes the *divergences* the most informative signal in the slice, so §9 is a full twin diff.

---

## 1. Formulas stated in prose

| # | Formula as stated | Where | Notes |
|---|---|---|---|
| F1 | `dE/dt = min(Ed_energy_balance, Ed_soundcrossing)` | `trinity/phase1c_transition/run_transition_phase.py:3` (module docstring, twice: Overview + Key Features), restated at `:200` | The central modelling choice of the whole phase. |
| F2 | `Ed_soundcrossing = -Eb / (R2 / c_sound)` | `trinity/phase1c_transition/run_transition_phase.py:236`; also written inline in the module docstring at `:3` as "the sound-crossing rate `-Eb/(R2/c_sound)`" | Equivalently `-Eb·c_sound/R2`. |
| F3 | `Ed_energy_balance = Lmech - Lcool - PdV` | `trinity/phase1c_transition/run_transition_phase.py:234` ("from energy balance (Lmech - Lcool - PdV)") | Sign convention: this is `dEb/dt`, so `PdV` enters with a minus. |
| F4 | `dR2/dt = v2` | `trinity/phase1c_transition/run_transition_phase.py:232` (comment "# = v2") | |
| F5 | `min()` selects "whichever gives faster energy loss (more negative Ed)" | `trinity/phase1c_transition/run_transition_phase.py:242` | Self-consistent: `min` of two signed rates does pick the more negative. |
| F6 | Transition phase-exit criterion: `P_ram / (P_b + P_ram) > threshold` | `trinity/phase1c_transition/run_transition_phase.py:749` ("exit when P_ram/(P_b + P_ram) > this"), with `:746` "Phase transition criterion: ram pressure dominates bubble pressure" | Ratio form; no guard documented for `P_b + P_ram = 0`. |
| F7 | Shell inner-edge density `nShell0 = Pb / (k_B · T_ion)` | `trinity/phase2_momentum/run_momentum_phase.py:581` | Stated as the *reason* for setting `Pb := P_ram` in momentum phase. |
| F8 | In momentum phase, `Pb = P_ram` (bubble pressure *is* ram pressure) | `trinity/phase2_momentum/run_momentum_phase.py:580`, `:665` ("Store Pb (ram pressure)"), `:3` module docstring ("Pb (ram pressure only, since Eb = 0)") | |
| F9 | In momentum phase, `R1 = R2` | `trinity/phase2_momentum/run_momentum_phase.py:587` ("no inner shock in momentum phase") | |
| F10 | In momentum phase, `Eb = 0`, `Ed = 0`, `Td = 0` | `trinity/phase2_momentum/run_momentum_phase.py:3` (Key Features 1 & 2), `:506` | See finding S6-B-14 on `= 0` vs `≈ 0`. |
| F11 | Momentum equation of motion terms: ram pressure (drive) + `P_HII` + `F_rad` + gravity | `trinity/phase2_momentum/run_momentum_phase.py:3`; body breakdown at `:417` (gravity), `:420` (ram), `:423` (HII at rShell), `:436` (ambient), `:447` ("Net pressure force using P_drive"), `:450` (derivatives) | `P_drive` is named but never defined in prose. |
| F12 | Adaptive-step trigger: `max dex (log10) change across monitored parameters` vs a dex threshold | `trinity/phase1c_transition/run_transition_phase.py:144`, `:101`; `trinity/phase2_momentum/run_momentum_phase.py:136`, `:93` | |
| F13 | dex threshold `0.1` ⇒ `10^0.1 ≈ 1.26×` | `trinity/phase1c_transition/run_transition_phase.py:101`; `trinity/phase2_momentum/run_momentum_phase.py:93` | Arithmetic checks out (10^0.1 = 1.2589). |
| F14 | Segment-duration growth/shrink factor `≈ 1.26` | `trinity/phase1c_transition/run_transition_phase.py:102`; `trinity/phase2_momentum/run_momentum_phase.py:94` | Same number as F13 — presumably `10^0.1` reused as the step factor. |
| F15 | `DT_COLLAPSE = 0.5 kyr = 5e-4 Myr` | `trinity/phase1c_transition/run_transition_phase.py:108`; `trinity/phase2_momentum/run_momentum_phase.py:100` | Unit conversion asserted in the comment itself; testable. |
| F16 | `max_step = 2e-5 Myr`, justified as "ensures >= 5 steps per segment" | `trinity/phase1c_transition/run_transition_phase.py:135`; `trinity/phase2_momentum/run_momentum_phase.py:127` | Implies smallest segment ≥ `1e-4` Myr. See S6-B-11. |
| F17 | Dissolution condition: `shell_nMax < nISM`, sustained (a "persistent timer") | `trinity/phase1c_transition/run_transition_phase.py:448`, `:805`; `trinity/phase2_momentum/run_momentum_phase.py:530`, `:858` | Duration threshold never stated. |
| F18 | Collapse detection: `v2 < 0` **AND** `R2` decreasing | `trinity/phase1c_transition/run_transition_phase.py:771`; `trinity/phase2_momentum/run_momentum_phase.py:824` | Contrast with F19. |
| F19 | Fine-timestep trigger: `|v2| >` threshold (constants), but applied "only during collapse (negative velocity)" | constants `trinity/phase1c_transition/run_transition_phase.py:105-107` / `trinity/phase2_momentum/run_momentum_phase.py:97-99`; usage `trinity/phase1c_transition/run_transition_phase.py:724` / `trinity/phase2_momentum/run_momentum_phase.py:803` | Direct prose-vs-prose mismatch. See S6-B-05. |
| F20 | Shell-mass invariant: `mShell(t+dt) >= mShell(t)` always; and frozen entirely while `isCollapse` | `trinity/phase1c_transition/run_transition_phase.py:532-534`, `:541`, `:546`; `trinity/phase2_momentum/run_momentum_phase.py:593-595`, `:602`, `:615` | Word-identical in both files. |
| F21 | `P_HII` comes from "Strömgren ionization balance in shell", via `n_IF_Str` | `trinity/phase1c_transition/run_transition_phase.py:318`, `:561`; `trinity/phase2_momentum/run_momentum_phase.py:258`, `:317`, `:442`, `:631` | No equation, no reference. |
| F22 | Radiation pressure force = "direct + IR-trapped" | `trinity/phase1c_transition/run_transition_phase.py:340`; `trinity/phase2_momentum/run_momentum_phase.py:274`, `:339` | Two-component claim; no formula, no `f_trap`/`τ_IR` named. |
| F23 | `Ed_energy_balance` at the phase boundary "matches the implicit phase's **beta-derived** Ed" | `trinity/phase1c_transition/run_transition_phase.py:200` | Only reference to `β` in the slice; β itself never defined here. |

## 2. Units stated in prose

| Quantity | Stated unit | Where |
|---|---|---|
| `t` (ODE independent var) | Myr | `trinity/phase1c_transition/run_transition_phase.py:200`; `trinity/phase2_momentum/run_momentum_phase.py:375` |
| `c_sound` | pc/Myr | `trinity/phase1c_transition/run_transition_phase.py:200` |
| velocity thresholds for timestep control | pc/Myr | `trinity/phase1c_transition/run_transition_phase.py:106-107`; `trinity/phase2_momentum/run_momentum_phase.py:98-99` |
| all segment durations (`DT_SEGMENT*`, `DT_COLLAPSE`) | Myr | `trinity/phase1c_transition/run_transition_phase.py:93-95`, `:108`; `trinity/phase2_momentum/run_momentum_phase.py:86-88`, `:100` |
| `min_step`, `max_step` | Myr | `trinity/phase1c_transition/run_transition_phase.py:134-135`; `trinity/phase2_momentum/run_momentum_phase.py:126-127` |
| adaptive threshold / step factor | dex (log10) | `trinity/phase1c_transition/run_transition_phase.py:101`; `trinity/phase2_momentum/run_momentum_phase.py:93` |
| `T_ion` (ionized shell temperature) | K | `trinity/phase2_momentum/run_momentum_phase.py:314` |

**Dimensional cross-checks I can do from prose alone (all pass):**
- F2 with `R2` in pc and `c_sound` in pc/Myr gives `Eb/Myr` — consistent with `t` in Myr. ✔
- F7 `Pb/(k_B T_ion)` → pressure/energy = number density. ✔
- F15 `0.5 kyr = 5e-4 Myr`. ✔
- F13 `10^0.1 = 1.2589 ≈ 1.26`. ✔

**No unit is stated anywhere for `Eb`, `Pb`, `P_ram`, `P_HII`, any force (`F_grav`, `F_ram`,
`F_ion`, `F_rad`), `Lmech`, `pdot`, `mShell`, `nISM`, or `shell_nMax`** — in a codebase whose own
CLAUDE.md calls units "a recurring bug class here". The energy floor constant
(`trinity/phase1c_transition/run_transition_phase.py:97`) is a dimensional threshold with no unit
given.

## 3. Citations

**There are none.** Across ~640 prose entries spanning two physics phase runners, the slice contains
**zero references to any paper, equation number, thesis, or textbook.** The only attributions of any
kind are:

- "Strömgren ionization balance" — named 6× (`trinity/phase1c_transition/run_transition_phase.py:318`,
  `:561`; `trinity/phase2_momentum/run_momentum_phase.py:258`, `:317`, `:442`, `:631`) with no
  reference and no equation.
- "the implicit phase's **beta**-derived Ed" (`trinity/phase1c_transition/run_transition_phase.py:200`)
  — a named formalism with no source.
- "IR-trapped" radiation pressure (`trinity/phase1c_transition/run_transition_phase.py:340`;
  `trinity/phase2_momentum/run_momentum_phase.py:274`) — no source, no trapping factor named.
- "Based on analysis of the top 30 most variable parameters"
  (`trinity/phase1c_transition/run_transition_phase.py:111`;
  `trinity/phase2_momentum/run_momentum_phase.py:103`) — an appeal to an analysis that is not
  identified, dated, or linked.
- "Snapshot Consistency (**January 2026**)" — a dated section header in both module docstrings
  (`trinity/phase1c_transition/run_transition_phase.py:3`; `trinity/phase2_momentum/run_momentum_phase.py:3`).
  The only provenance marker in the slice.
- "(same as in energy_implicit)" (`trinity/phase1c_transition/run_transition_phase.py:251`) — an
  intra-repo equivalence claim, unpinned to any commit or test.

The `min(Ed_energy_balance, Ed_soundcrossing)` decay law (F1/F2) is the defining physics of phase 1c
and is presented with **no derivation and no citation** — only an operational rationale
(`:242-244`). Recorded as S6-B-12.

## 4. Ranges, regimes, assumptions

- **A** applies "when bubble thermal energy (Eb) becomes negligible as cooling dominates"
  (`trinity/phase1c_transition/run_transition_phase.py:3`).
- **B** applies when "bubble thermal pressure is negligible" / "Eb ≈ 0"
  (`trinity/phase2_momentum/run_momentum_phase.py:3`). Stated as the *final* expansion phase.
- **Assumed ordering of the two decay rates.** The docstring asserts `min()` is "continuous with the
  implicit phase early on" and "falls back to the sound-crossing rate once cooling becomes
  inefficient" (`trinity/phase1c_transition/run_transition_phase.py:3`, restated `:242-244`). This is
  only true if `Ed_energy_balance < Ed_soundcrossing` at phase entry. Nothing in the prose asserts or
  enforces that ordering. If the sound-crossing branch wins on the *first* step, the advertised
  continuity property silently does not hold.
- **B assumes no inner shock** (`R1 = R2`, `:587`) and **no thermal pressure** (`:214`, `:224`, `:420`).
- **B's fine-timestep and collapse machinery assumes inward motion only** (`:803`) — see S6-B-05.
- **Both assume the ambient/ISM pressure term only engages beyond the cloud edge**
  (`trinity/phase1c_transition/run_transition_phase.py:312`;
  `trinity/phase2_momentum/run_momentum_phase.py:252`, `:436`).
- **`Pb := P_ram` in B is an explicitly stated workaround**, not a physical identity: "Without this,
  Pb = 0 in momentum phase would give n_IF → 0" (`trinity/phase2_momentum/run_momentum_phase.py:582`).
  The justification is that it makes `nShell0` "physically meaningful" — i.e. the substitution exists
  to keep a downstream formula well-posed.

## 5. Contracts

**Inputs / outputs**
- `run_phase_transition(params: ParameterDict) -> TransitionPhaseResults`; results "Contains t, R2,
  v2, Eb arrays and termination info" (`trinity/phase1c_transition/run_transition_phase.py:3`, `:368`, `:185`).
- `run_phase_momentum(params: ParameterDict) -> MomentumPhaseResults`; "Contains t, R2, v2 arrays and
  termination info" (`trinity/phase2_momentum/run_momentum_phase.py:3`, `:462`, `:177`).
- A's ODE state vector: `y = [R2, v2, Eb]`, returns `[dR2/dt, dv2/dt, dEb/dt]`
  (`trinity/phase1c_transition/run_transition_phase.py:200`); confirmed "no T0 in state vector for
  transition" (`:426`).
- B's ODE state vector: `y = [R2, v2]`, returns `[dR2/dt, dv2/dt]`
  (`trinity/phase2_momentum/run_momentum_phase.py:375`).

**Purity**
- A: "Pure ODE functions: No dictionary mutations during integration"
  (`trinity/phase1c_transition/run_transition_phase.py:3`); "Pure ODE function for transition phase"
  (`:200`); "Compute all force components **without mutating params**" (`:278`).
- B: same key feature (`:3`); "Reads snapshot but does **NOT** mutate during integration" (`:375`);
  "without mutating params" (`:214`).
- Tension: B `:405` says the ODE performs a **live feedback lookup** against `params` during
  integration. Reading is compatible with purity only if the feedback/SPS lookup does not memoise
  into `params`. Nothing asserts that.

**Ordering requirements (explicit)**
1. A `:499-502`: "Get R1 and Pb **BEFORE** shell structure so Pb is current when
   `shell_structure_pure` reads it."
2. A `:530-531` / B `:591-592` (word-identical): "Compute shell mass **BEFORE** shell structure so
   that the shell termination condition uses the current R2's swept-up mass."
3. A `:588` / B `:670`: "Save snapshot **BEFORE** ODE — all values are consistent at `t_now`."
   A enumerates the consistent set as `t_now, R2, v2, Eb, feedback, shell_props, R1, Pb, forces,
   mShell` (`:590-591`); B as `t_now, R2, v2, feedback, shell_props, mShell, forces, Pb` (`:672-673`).
4. A `:323` / B `:263`, `:443`: `P_HII` is "pre-computed in phase runner from `n_IF_Str`" — the force
   routine and the ODE both *consume* a value the runner must have already produced (`:561` / `:631`).
5. A `:746-748`: the exit check must "recompute Pb and P_ram at the **post-ODE** state so the check
   uses current values, not stale pre-ODE ones."
6. A `:618` / B `:700`: monitor values captured **before** integration for the dex comparison.

**Invariants**
- `mShell` never decreases; frozen during collapse (F20). Stated twice per file — once in the primary
  block and once in the adaptive-stepping block, with B `:770-771` / A `:693-694` explicitly saying
  "Apply the **same** collapse-freeze and never-decrease guards as the primary shell mass block above."
  That is a documented code-duplication contract: two implementations that must stay in lockstep.
- Snapshot timestamp consistency ("all values correspond to the same timestamp (t_now)"), both `:3`.

**Termination / phase-transition conditions**
- A, per docstring `:368`: "Energy decays on the sound-crossing timescale until it reaches a **floor
  value**, then momentum phase begins."
- A, per event list `:456`: "Transition phase events: `energy_floor` (**phase ending**), `min_radius`,
  `velocity_runaway`."
- A, per constant `:97`: "Minimum energy before transition to momentum phase."
- A, per in-loop check `:746-749`: "**Phase transition criterion**: ram pressure dominates bubble
  pressure … exit when `P_ram/(P_b + P_ram) > this`."
- A, per `:765`: "**Safety fallback**: absolute energy floor."
- A also: stop_t reached (`:603`, `:778`), stop_r reached (`:796`), dissolution (`:805`),
  collapse (`:771`), `stop_at_rCloud_nSnap` (`:466`), max_segments (`:869`).
- B: stop_t (`:685`, `:831`), stop_r (`:849`), dissolution (`:858`), collapse (`:824`),
  `stop_at_rCloud_nSnap` (`:547`), max_segments (`:915`), plus unenumerated centralized events (`:537`).
- Both: "unknown" termination "means we fell through every known exit path — a **real bug surface**,
  not a routine completion. Surface it loudly." (A `:872-873`, B `:918-919`).
- Both: zero-work guard — if the prior phase already passed `stop_t`, surface it explicitly "instead
  of silently looping zero times and reporting `termination_reason='unknown'`" (A `:404-406`,
  B `:485-487`, word-identical).

**Side effects**
- A `:398`: "Update `cool_alpha` to match ODE-evolved v2 (preserves ODE continuity)" — mutates
  `params` at phase entry.
- B `:580`: sets `params['Pb'] = P_ram`; B `:587`: sets `R1 = R2`.
- Both `:668` / `:748`: "Apply event result to params" (B adds "(sets `SimulationEndReason`, etc.)").
- Both write snapshots inside the loop, plus a post-loop "Phase-boundary reconciliation snapshot"
  (A `:828-832`, B `:881-885`).
- Both `:595-596` / `:677-678`: the past-rCloud counter increments "only when the save actually wrote
  (duplicate guard in `save_snapshot` can skip)" — i.e. `save_snapshot` is documented as
  *silently skipping* duplicates.

## 6. Numerical claims

- Integrator: `scipy.integrate.solve_ivp` with **LSODA**, described as "Adaptive integration for
  accuracy" and "Auto-switches stiff/non-stiff" (both `:3`; A `:136`, B `:128`).
- `rtol`, `atol`, `min_step` (Myr), `max_step` configured (A `:132-135`, B `:124-127`).
- "`min_step` only supported by LSODA" (A `:626`, B `:708`) — a solver-capability claim gating the
  kwargs construction.
- `max_step = 2e-5 Myr`, justified "(ensures >= 5 steps per segment)" (A `:135`, B `:127`).
- Segment-based integration with adaptive `dt_segment`, bounded by min/max constants, scaled by
  `≈1.26` per the dex criterion (A `:100-102`, `:681-683`; B `:92-94`, `:759-761`).
- Event semantics: terminal events end the segment; the runner then "Update state from event" and
  "Add final state to results" and "Apply event result to params" (A `:651-668`, B `:733-748`).
- Velocity-based *proactive* step control with two tiers: "Extreme collapse velocity: use **minimum**
  segment duration" / "Moderate collapse velocity: use **intermediate** segment duration"
  (A `:726`, `:731`; B `:805`, `:810`).
- Dissolution uses a *persistent* timer rather than an instantaneous test (A `:448`, B `:530`) —
  the anti-chatter intent is stated; the threshold is not.
- B `:896-898`: the final reconciliation snapshot is wrapped so that a failure produces a **warning**
  carrying "exception class and deepest traceback frame … which step (SPS lookup, pRam,
  shell_structure, save_snapshot) actually failed". A documents no equivalent.
- B `:609`: "Handle array returns" — a defensive shape-coercion on the shell-mass result.

## 7. Admissions (known debt, verbatim)

| Admission | Where | Reading |
|---|---|---|
| "Ed diagnostic at first segment (**quantify the original discontinuity**)" | `trinity/phase1c_transition/run_transition_phase.py:520` | Direct admission that the implicit→transition boundary had (has?) a discontinuity, and that instrumentation for it is still shipped in the hot path. |
| "--- PHASE BOUNDARY DIAGNOSTIC ---" … "--- END DIAGNOSTIC ---" | `trinity/phase1c_transition/run_transition_phase.py:389`, `:396` | Debug scaffolding left in production initialization. |
| "**Safety fallback**: absolute energy floor" | `trinity/phase1c_transition/run_transition_phase.py:765` | The energy floor is demoted to a backstop, contradicting three other comments that call it the phase-ending criterion. |
| "Without this, Pb = 0 in momentum phase would give n_IF → 0" | `trinity/phase2_momentum/run_momentum_phase.py:582` | Explicit statement that `Pb := P_ram` exists to avoid a degenerate downstream value. |
| "'unknown' means we fell through every known exit path — **a real bug surface**, not a routine completion." | A `:872-873`, B `:918-919` | Acknowledged reachable bug state. |
| "instead of **silently** looping zero times and reporting termination_reason='unknown'" | A `:405-406`, B `:486-487` | Describes a silent-failure mode that was patched — implies the pattern existed. |
| "duplicate guard in `save_snapshot` **can skip**" | A `:596`, B `:678` | A documented silent no-op in the output path. |
| "Apply the **same** … guards as the primary shell mass block above" | A `:693-694`, B `:770-771` | Admitted duplicated logic with a lockstep requirement and no mechanism enforcing it. |
| "Based on analysis of the top 30 most variable parameters" | A `:111`, B `:103` | Unsourced, undated tuning provenance. |
| "Throttled progress heartbeat (outer loop only — **never inside solvers**)" | A `:689`, B `:766` | A performance landmine flagged for future maintainers. |
| "(same as in energy_implicit)" | `trinity/phase1c_transition/run_transition_phase.py:251` | Unpinned copy of another module's logic. |

No literal `TODO`/`FIXME`/`XXX`/`hack` tokens appear in the slice. The debt is phrased in prose.

## 8. Formula ↔ formula and formula ↔ contract tensions

- **F1 vs the function docstring.** `:3` and `:200` say `dE/dt = min(Ed_energy_balance,
  Ed_soundcrossing)`. `:368` says "Energy decays **on the sound-crossing timescale** until it reaches
  a floor value." The second describes the *pre-`min()`* model. → S6-B-02.
- **F6 vs the energy-floor contract.** F6 is labelled "Phase transition criterion"; the floor is
  labelled "Safety fallback" (`:765`) — yet `:97`, `:368` and `:456` all present the floor as the
  thing that ends the phase. → S6-B-01.
- **F19 vs F18 vs the constants.** Three different notions of "when we're collapsing" coexist:
  `|v2| >` threshold (constants), `v2 < 0` (timestep control), `v2 < 0 AND R2 decreasing` (detector).
  → S6-B-05, S6-B-16.
- **F8 vs A's force naming.** In B, `Pb` *is* ram pressure. In A, `F_ram` is documented as "Ram
  pressure force (**from bubble pressure**)" (`:260`) while the exit criterion (`:749`) treats `P_b`
  and `P_ram` as two distinct quantities in the same expression. → S6-B-10.
- **B's live feedback vs B's frozen snapshot.** `:405-406` "Use live feedback so SN turn-on events
  mid-segment are visible"; but `:339` computes the radiation-pressure force *inside*
  `create_momentum_snapshot`, i.e. once per segment. → S6-B-04.
- **B's ODE docstring vs B's ODE body.** Docstring: `params` is "Original params for **density
  profile lookup**" (`:375`). Body: `params` drives a **live feedback** lookup (`:405`). A's twin
  docstring says "Original params dict for **feedback interpolation**" (`:200`). → S6-B-03.

## 9. Twin diff — A vs B (the core of this report)

Word-identical in both files (copy-propagated correctly): the adaptive-stepping constants block and
its comments; the six monitor-group headings; the whole ODE-solver settings block including the
`2e-5 Myr` justification; `compute_max_dex_change` and `get_monitor_values` docstrings; the zero-work
guard; the `stop_at_rCloud_nSnap` comments (both sites); the entire shell-mass block (primary and
adaptive-stepping copies); the velocity-timestep-control block; the collapse detector; the
dissolution timer; the heartbeat note; the "unknown = real bug surface" note.

Divergences — each is a candidate signal:

| Topic | A (transition) | B (momentum) | Verdict |
|---|---|---|---|
| Initial segment duration | `:93` "Myr - initial segment duration" | `:86` "Myr - initial segment duration (**larger OK in momentum phase**)" | B copy-edited; B's segments explicitly larger, yet `max_step` comment is byte-identical → S6-B-11 |
| Energy floor constant | `:97` present | absent | Expected (Eb=0) |
| Monitor groups | `:117` Cooling, `:119` Bubble | `:109` Cooling, `:111` Bubble — **retained** | B evolves neither → S6-B-13 |
| ODE `params` arg description | `:200` "for feedback interpolation", typed `DescribedDict` | `:375` "for density profile lookup", typed `dict` | **Disagree**, and B's body contradicts B's docstring → S6-B-03 |
| `ForceProperties.F_ram` | `:260` "Ram pressure force (**from bubble pressure**)" | `:195` "Ram pressure force" | A's parenthetical is the odd one → S6-B-10 |
| Force-fn docstring | `:278` bare | `:214` bare **+** "In momentum phase, pressure is ram pressure only (no thermal pressure)." | B extended |
| Force-fn ordering | grav → ionization → outside-shell → ISM → warm-ionized → **ram contribution** → Forces → F_ram → F_rad | grav → **ram** → ionization → outside-shell → ISM → warm-ionized → Forces → F_ram → F_rad | Ram moved to front in B |
| ODE snapshot dataclass | referenced only, fields undocumented | `:303-317` documented, incl. `:316` "**Gate all HII pressure**" | A has no documented HII gate → S6-B-07 |
| Duplicate force path | `compute_forces_pure` (`:278-340`) vs the energy-balance ODE it delegates to (`:230`) | `compute_forces_momentum_pure` (`:214-274`) vs `get_ODE_momentum_pure` (`:417-450`) — both compute grav/ram/HII/ambient | Two force implementations per file, no stated equivalence → S6-B-06 |
| Array-return guard | absent at `:541-546` | `:609` "Handle array returns" | **Disagree** → S6-B-08 |
| Array guard in the *secondary* shell-mass block | absent (`:692-705`) | absent (`:769-784`) | Guard present in exactly 1 of 4 sites |
| Event enumeration | `:456` names all three events | `:537` names none | B under-documented |
| Event-apply comment | `:668` "Apply event result to params" | `:748` "… (**sets SimulationEndReason, etc.**)" | B copy-edited |
| ISM vs ambient wording | `:312` "ISM pressure" | `:252` "ISM pressure", `:436` "**ambient** pressure" | B uses both words in one file for (apparently) one term |
| Reconciliation snapshot | `:828-832` "Recompute derived properties (**Pb, shell structure, forces**)" | `:881-885` "Recompute derived properties" | B dropped the enumeration |
| Reconciliation error handling | none documented | `:896-898` catch → **warning** with traceback frame | **Disagree** → S6-B-09 |
| Phase-entry fixups | `:398` `cool_alpha` update | `:580` `Pb := P_ram`; `:587` `R1 := R2` | Phase-appropriate |
| Diagnostics | `:389-396`, `:520` present | none | A carries shipped debug scaffolding |
| Overview prose | one statement of the regime | `:3` states "thermal pressure negligible / drive from ram" **three times** (header, bullet 1, bullet 3) | Edit added a sentence without removing the old one |
| "Eb" statement | n/a | `:3` "Eb **≈** 0" (Overview, run docstring) vs "**Eb = 0**" (Key Feature 1) vs `:506` "Eb = 0" | → S6-B-14 |
| Snapshot field list | includes `R1, Pb` | includes `Pb` only (correct, `R1=R2`) | Consistent |

**Reading of the twin diff:** B is the later copy-edit in the *doc-comment* layer (added
justifications at `:86`, `:214`, `:748`; documented the snapshot dataclass; added the array guard and
the exception-detail comment) but the *stale* copy in the ODE-docstring layer (`:375` still describes
a pre-live-feedback design that A's twin at `:200` already describes correctly). Edits have flowed in
both directions and neither file is uniformly ahead. Everything word-identical between them should be
treated as un-reviewed since the split.

## 10. Claims too vague to check as written

- "Based on analysis of the top 30 most variable parameters" (A `:111`, B `:103`) — no artifact named.
- "persistent dissolution timer" (A `:448`, B `:530`) — "persistent" is never quantified.
- "(same as in energy_implicit)" (A `:251`) — no revision, no test pinning the equivalence.
- "Net pressure force using `P_drive`" (B `:447`) — `P_drive` is never defined in prose.
- "Radiation pressure force (direct + IR-trapped)" (A `:340`, B `:274`, `:339`) — no trapping model.
- "Extreme" vs "Moderate" collapse velocity (A `:726`/`:731`, B `:805`/`:810`) — the mapping from
  those words to the two named constants is left implicit, and `DT_COLLAPSE`'s own comment says
  "segment duration **during collapse**" (A `:108`, B `:100`) without saying it is the *moderate*
  tier only.
- "Auto-switches stiff/non-stiff" (A `:136`, B `:128`) — a property of LSODA, asserted without the
  tolerance regime in which it is expected to hold for this problem.

---

## Findings

```json
[
  {
    "id": "S6-B-01",
    "file": "trinity/phase1c_transition/run_transition_phase.py",
    "line": 765,
    "class": "regime",
    "severity": "S2",
    "claim": "The transition phase has two mutually exclusive documented phase-ending criteria. Three comments say the energy floor ends the phase: ':97' \"Minimum energy before transition to momentum phase\", ':368' \"Energy decays on the sound-crossing timescale until it reaches a floor value, then momentum phase begins\", ':456' \"Transition phase events: energy_floor (phase ending), min_radius, velocity_runaway\". Two comments say the criterion is ram-pressure dominance and demote the floor: ':746' \"Phase transition criterion: ram pressure dominates bubble pressure\" / ':749' \"exit when P_ram/(P_b + P_ram) > this\", and ':765' \"Safety fallback: absolute energy floor\".",
    "evidence": "trinity/phase1c_transition/run_transition_phase.py:97, :368, :456 vs :746-749, :765",
    "expected": "One documented exit criterion, with the other explicitly described as subordinate in every place it is mentioned. If ram-pressure dominance is primary, the module and function docstrings and the energy-floor constant comment must say so.",
    "failure_scenario": "If the run exits on P_ram/(P_b+P_ram) > threshold while Eb is still well above the floor, the momentum phase is entered with non-negligible thermal energy, which it then discards (Eb = 0, trinity/phase2_momentum/run_momentum_phase.py:506). The bubble's remaining thermal pressure vanishes at the handoff, producing a step in the driving pressure and a kink in v2(t) that is a modelling artefact, not physics. The stiffer the cooling (high-density GMC), the larger the residual Eb at the ratio crossing and the larger the artefact.",
    "repro": "Instrument both exit paths in trinity/phase1c_transition/run_transition_phase.py and record which fires, plus Eb at exit relative to the floor constant, for param/simple_cluster.param and docs/dev/performance/f1edge_hidens*.param. Then compare Eb at transition exit against Eb at momentum entry in dictionary.jsonl.",
    "confidence": "high"
  },
  {
    "id": "S6-B-02",
    "file": "trinity/phase1c_transition/run_transition_phase.py",
    "line": 368,
    "class": "other",
    "severity": "S3",
    "claim": "run_phase_transition's docstring states the pre-min() energy model: \"Energy decays on the sound-crossing timescale until it reaches a floor value, then momentum phase begins.\" The module docstring (:3) and the ODE docstring (:200) both state the current model, dE/dt = min(Ed_energy_balance, Ed_soundcrossing). The function docstring was not updated when the min() model landed.",
    "evidence": "trinity/phase1c_transition/run_transition_phase.py:368 vs :3 and :200",
    "expected": "The public function docstring should state dE/dt = min(Ed_energy_balance, Ed_soundcrossing), matching :3 and :200.",
    "failure_scenario": "A maintainer reading only the entry-point docstring believes energy decay is purely -Eb*c_sound/R2 and mis-diagnoses any Ed behaviour driven by the energy-balance branch (e.g. attributing a slow decay to a wrong c_sound rather than to Lcool).",
    "repro": "Read the three docstrings side by side; no run needed.",
    "confidence": "high"
  },
  {
    "id": "S6-B-03",
    "file": "trinity/phase2_momentum/run_momentum_phase.py",
    "line": 375,
    "class": "state",
    "severity": "S3",
    "claim": "get_ODE_momentum_pure's docstring documents its params argument as \"Original params for density profile lookup\", but the function body comment at :405-406 says \"Use live feedback so SN turn-on events mid-segment are visible (consistent with energy/implicit/transition ODEs)\" — i.e. params is also used for feedback interpolation during integration. The twin in the transition runner documents the same argument correctly as \"Original params dict for feedback interpolation\" (trinity/phase1c_transition/run_transition_phase.py:200) and types it DescribedDict rather than dict.",
    "evidence": "trinity/phase2_momentum/run_momentum_phase.py:375 vs :405-406; twin at trinity/phase1c_transition/run_transition_phase.py:200",
    "expected": "The momentum ODE docstring should list both uses (density profile lookup AND live feedback interpolation) and match the twin's type annotation.",
    "failure_scenario": "A caller trusting the docstring passes a params object carrying only the density profile (e.g. a trimmed dict in a test harness or a replay rig) and the live feedback lookup either raises or silently falls back, changing the ODE's drive term without any error.",
    "repro": "Check whether get_ODE_momentum_pure calls a feedback/SPS interpolation on params; confirm whether the transition and momentum ODEs use the same lookup entry point, as :406 asserts.",
    "confidence": "high"
  },
  {
    "id": "S6-B-04",
    "file": "trinity/phase2_momentum/run_momentum_phase.py",
    "line": 339,
    "class": "state",
    "severity": "S3",
    "claim": "The radiation pressure force is computed inside create_momentum_snapshot (\":339 # Radiation pressure force (direct + IR-trapped)\"), i.e. frozen once per segment, while the ODE deliberately re-reads feedback live: \":405-406 Use live feedback so SN turn-on events mid-segment are visible\". The two comments describe opposite freshness policies for quantities that share the same feedback source.",
    "evidence": "trinity/phase2_momentum/run_momentum_phase.py:339 vs :405-406; snapshot fields documented at :303-317",
    "expected": "Either F_rad is recomputed live alongside the other feedback-derived terms, or the live-feedback comment should state which terms are live and which are frozen.",
    "failure_scenario": "A supernova turning on mid-segment updates Lmech/pdot (live) but not the radiation force (frozen from segment start). The momentum equation then mixes post-SN mechanical drive with pre-SN radiative drive for the remainder of the segment. Worst at large DT_SEGMENT, which :86 says is deliberately larger in the momentum phase.",
    "repro": "Set up a run whose SN turn-on time falls strictly inside a momentum segment; compare v2 at segment end against the same run with DT_SEGMENT forced small enough that the turn-on lands on a segment boundary. Separate processes, matched t.",
    "confidence": "medium"
  },
  {
    "id": "S6-B-05",
    "file": "trinity/phase1c_transition/run_transition_phase.py",
    "line": 105,
    "class": "regime",
    "severity": "S3",
    "claim": "The velocity-based timestep control is documented on |v2| but applied only to negative v2. Constants: \":105 When |v2| exceeds threshold, reduce dt_segment to ensure fine temporal resolution\", \":106 pc/Myr - proactively reduce step when |v2| > this\", \":107 pc/Myr - use minimum step when |v2| > this\". Usage: \":724 Only during collapse (negative velocity = inward motion)\". Identical mismatch in the twin (trinity/phase2_momentum/run_momentum_phase.py:97-99 vs :803).",
    "evidence": "trinity/phase1c_transition/run_transition_phase.py:105-107 vs :724; trinity/phase2_momentum/run_momentum_phase.py:97-99 vs :803",
    "expected": "Either the constants' comments say \"when v2 < -threshold\", or the guard tests |v2| as the constants describe.",
    "failure_scenario": "A rapid OUTWARD acceleration gets no timestep refinement: shell breakout at the cloud edge into low-density ISM, or an SN turn-on, both produce large positive v2 excursions. Those are integrated at the full (in momentum phase, deliberately larger) segment length, coarsening the trajectory precisely where it changes fastest, while the snapshot cadence is also segment-driven so the output loses the feature too.",
    "repro": "Log dt_segment against v2 through the cloud-edge crossing for docs/dev/performance/f1edge_lowdens*.param and check whether dt_segment ever shrinks for positive v2.",
    "confidence": "high"
  },
  {
    "id": "S6-B-06",
    "file": "trinity/phase2_momentum/run_momentum_phase.py",
    "line": 417,
    "class": "state",
    "severity": "S3",
    "claim": "The momentum file documents two independent computations of the same force set, ~150 lines apart, with no comment asserting they agree. compute_forces_momentum_pure (:214-274) computes gravity (:219), ram (:224), ionization (:227), outside-shell inward pressure (:239), ISM pressure beyond cloud (:252), P_HII (:256-263), F_ram (:271), F_rad (:274). get_ODE_momentum_pure independently computes gravity (:417), ram (:420), HII at rShell (:423), ambient beyond cloud (:436), net pressure force via P_drive (:447). The first path feeds the saved snapshot/forces; the second drives the integration. The transition file has the same split (compute_forces_pure at :278-340 for output vs the energy-balance ODE at :230 for integration).",
    "evidence": "trinity/phase2_momentum/run_momentum_phase.py:214-274 vs :417-450; trinity/phase1c_transition/run_transition_phase.py:278-340 vs :230",
    "expected": "One force implementation, or a stated and tested equivalence between the reporting path and the integration path.",
    "failure_scenario": "The forces written to output (F_grav, F_ram, F_ion, F_rad, and the force-budget diagnostics that are a headline product of this code) are not the forces that actually produced the trajectory. Any drift between the two — a term added to one, a gate applied to one — makes the published force budget silently inconsistent with R2(t), v2(t) while every run still completes normally.",
    "repro": "For one momentum segment, evaluate both paths at identical state and compare each force term; assert equality in a pytest case.",
    "confidence": "medium"
  },
  {
    "id": "S6-B-07",
    "file": "trinity/phase2_momentum/run_momentum_phase.py",
    "line": 316,
    "class": "state",
    "severity": "S3",
    "claim": "MomentumODESnapshot documents a field whose sole purpose is to \"Gate all HII pressure\" (:316). No gate is mentioned anywhere in compute_forces_momentum_pure's HII block (:227-263), in its warm-ionized-gas section (:256-259), or in the transition file's corresponding blocks (trinity/phase1c_transition/run_transition_phase.py:286-323). The gate is documented on the integration path only.",
    "evidence": "trinity/phase2_momentum/run_momentum_phase.py:316 vs :227-263; no counterpart in trinity/phase1c_transition/run_transition_phase.py:286-323",
    "expected": "If HII pressure can be gated off, the same gate must apply to the reported F_ion / P_HII, and the transition phase needs the same switch or a comment saying why not.",
    "failure_scenario": "With the gate off, the ODE sees zero HII pressure while the saved snapshot reports a non-zero F_ion and P_HII, so the output claims a force contribution that never acted. Compounds S6-B-06. Also: if the gate exists only in phase 2, HII pressure switches on/off discontinuously at the phase-1c boundary.",
    "repro": "Find the config key behind the gate; run with it off and check whether the saved F_ion/P_HII are zero.",
    "confidence": "medium"
  },
  {
    "id": "S6-B-08",
    "file": "trinity/phase2_momentum/run_momentum_phase.py",
    "line": 609,
    "class": "silent-failure",
    "severity": "S3",
    "claim": "\"# Handle array returns\" appears at exactly one of the four shell-mass update sites in the slice: the momentum primary block (:609). It is absent from the transition primary block (trinity/phase1c_transition/run_transition_phase.py:541-546) and from BOTH adaptive-stepping copies (trinity/phase1c_transition/run_transition_phase.py:692-705 and trinity/phase2_momentum/run_momentum_phase.py:769-784) — even though those copies are documented as applying \"the same collapse-freeze and never-decrease guards as the primary shell mass block above\" (:693-694 / :770-771).",
    "evidence": "trinity/phase2_momentum/run_momentum_phase.py:609; absent at trinity/phase1c_transition/run_transition_phase.py:541-546, :692-705, trinity/phase2_momentum/run_momentum_phase.py:769-784",
    "expected": "If the shell-mass helper can return an array, all four sites coerce it identically; the 'same guards as above' comment is otherwise false.",
    "failure_scenario": "An array-valued shell mass reaches an unguarded site: the never-decrease comparison becomes an elementwise array whose truth value is ambiguous (raises) or, if a length-1 array, silently stores an array into params['shell_mass']. That then flows into the dex monitor (shell params are monitored, :115) and into the saved snapshot, corrupting adaptive stepping and output without an error.",
    "repro": "Determine under what inputs the shell-mass call returns an array; force that condition in the transition phase and in the adaptive-stepping block.",
    "confidence": "medium"
  },
  {
    "id": "S6-B-09",
    "file": "trinity/phase2_momentum/run_momentum_phase.py",
    "line": 896,
    "class": "silent-failure",
    "severity": "S3",
    "claim": "The momentum runner's final phase-boundary reconciliation snapshot (:881-885) is wrapped in exception handling that downgrades any failure to a warning: \":896-898 Include exception class and deepest traceback frame so the warning tells us which step (SPS lookup, pRam, shell_structure, save_snapshot) actually failed, instead of just the bare message.\" The transition runner's equivalent reconciliation block (trinity/phase1c_transition/run_transition_phase.py:828-832) documents no such handling.",
    "evidence": "trinity/phase2_momentum/run_momentum_phase.py:896-898 vs trinity/phase1c_transition/run_transition_phase.py:828-832",
    "expected": "Consistent policy across the twins, and a failure of the final snapshot should be visible in the run's termination status, not only in a log line.",
    "failure_scenario": "The final reconciliation snapshot fails (the comment names four plausible failure points, including shell_structure and the SPS lookup). The run reports success; the last record in the output is the pre-ODE in-loop snapshot at an earlier t_now, so the final state of the simulation is missing or stale while nothing marks the output as incomplete. Batch sweeps would carry this silently across many runs.",
    "repro": "Force shell_structure to raise in the last momentum segment; confirm the run still exits with a success termination_reason and inspect the last dictionary.jsonl record's t.",
    "confidence": "high"
  },
  {
    "id": "S6-B-10",
    "file": "trinity/phase1c_transition/run_transition_phase.py",
    "line": 260,
    "class": "state",
    "severity": "S3",
    "claim": "The name 'ram' denotes two different physical quantities across the slice, and both meanings appear in the transition file. Transition: \":260 Ram pressure force (from bubble pressure)\" — F_ram is derived from the bubble's THERMAL pressure — while the exit criterion at :749 evaluates P_ram/(P_b + P_ram), treating P_ram and P_b as distinct. Momentum: \":580 Set Pb to ram pressure\", \":665 Store Pb (ram pressure)\", module docstring \"Pb (ram pressure only, since Eb = 0)\", \":224/:420 Ram pressure (momentum phase - no thermal pressure)\".",
    "evidence": "trinity/phase1c_transition/run_transition_phase.py:260 vs :749; trinity/phase2_momentum/run_momentum_phase.py:3, :224, :420, :580, :665",
    "expected": "Distinct names for bubble thermal pressure and wind ram pressure, kept consistent across phases; and within the transition file, F_ram and P_ram should refer to the same quantity.",
    "failure_scenario": "Two consequences. (1) Output: the columns Pb and F_ram change physical meaning at the 1c->2 boundary, so any plot or analysis of Pb(t) or F_ram(t) across that boundary shows a definitional discontinuity indistinguishable from physics. (2) The exit criterion at :749 may be comparing the ram pressure against a Pb that the very same file's force code has already relabelled as 'ram' — if the two P_ram in that expression are not the same quantity the ratio, and hence the phase-transition time, is wrong.",
    "repro": "Trace which variable feeds ForceProperties.F_ram in the transition phase and which two variables enter the ratio at :749; check the units and identity of Pb in the saved output on either side of the phase boundary.",
    "confidence": "medium"
  },
  {
    "id": "S6-B-11",
    "file": "trinity/phase2_momentum/run_momentum_phase.py",
    "line": 127,
    "class": "numerical",
    "severity": "S3",
    "claim": "Both files carry the byte-identical justification \"# Max step = 2e-5 Myr (ensures >=5 steps per segment)\" (trinity/phase2_momentum/run_momentum_phase.py:127, trinity/phase1c_transition/run_transition_phase.py:135), yet the momentum file separately states its segments are deliberately longer: \":86 Myr - initial segment duration (larger OK in momentum phase)\". A fixed 2e-5 Myr cap justified by a >=5-steps-per-segment argument implies the smallest segment is ~1e-4 Myr; DT_COLLAPSE is 5e-4 Myr (:100) and DT_SEGMENT_MIN is a separate, unstated constant.",
    "evidence": "trinity/phase2_momentum/run_momentum_phase.py:127 vs :86, :87, :100; trinity/phase1c_transition/run_transition_phase.py:135 vs :94, :108",
    "expected": "max_step derived from dt_segment (e.g. dt_segment/5) if the intent is 'at least 5 steps per segment'; or a comment that states the actual invariant a fixed cap provides. And DT_SEGMENT_MIN >= 5 * max_step must hold in both files for the stated guarantee to be true.",
    "failure_scenario": "Two directions. If DT_SEGMENT_MIN < 1e-4 Myr, the documented >=5-steps guarantee is false at exactly the moment it matters (extreme collapse, where :805 forces the minimum segment). If DT_SEGMENT_MAX is large in the momentum phase, a hard 2e-5 Myr cap forces >=5e4 LSODA steps per Myr of a phase that can run for tens of Myr — a large fixed cost that defeats LSODA's adaptivity and is not what the comment claims to be buying.",
    "repro": "Read the four constants in each file and check DT_SEGMENT_MIN >= 5*MAX_STEP; count actual solver steps per segment in a momentum-phase run.",
    "confidence": "medium"
  },
  {
    "id": "S6-B-12",
    "file": "trinity/phase1c_transition/run_transition_phase.py",
    "line": 3,
    "class": "citation",
    "severity": "S3",
    "claim": "The slice contains zero literature citations. The defining physics of phase 1c — dE/dt = min(Ed_energy_balance, Ed_soundcrossing) with Ed_soundcrossing = -Eb/(R2/c_sound) (:3, :200, :236) — is presented with no derivation and no reference, justified only operationally at :242-244. 'Strömgren ionization balance' is invoked six times (trinity/phase1c_transition/run_transition_phase.py:318, :561; trinity/phase2_momentum/run_momentum_phase.py:258, :317, :442, :631) with no reference or equation. 'IR-trapped' radiation pressure (:340 / :274) has no trapping model named. The 'implicit phase's beta-derived Ed' (:200) names a formalism with no source.",
    "evidence": "trinity/phase1c_transition/run_transition_phase.py:3, :200, :236, :242-244, :318, :340; trinity/phase2_momentum/run_momentum_phase.py:258, :274, :442",
    "expected": "The energy-decay law, the Strömgren balance expression, and the IR-trapping model each carry a reference with an equation number, or an in-repo derivation link.",
    "failure_scenario": "The min() decay law cannot be validated against any published model, so a coefficient or sign error in either branch is undetectable by review. Specifically, the docstring's claim that the energy-balance branch dominates 'early on' and the sound-crossing branch 'once cooling becomes inefficient' is an assertion about the ordering of two rates that nothing derives or checks — if the sound-crossing branch wins at the first step, the advertised continuity with the implicit phase silently does not hold.",
    "repro": "Log which branch of the min() is selected at each step of the transition phase; check whether Ed_energy_balance is in fact the selected branch at t_transition_start for param/simple_cluster.param and for docs/dev/performance/f1edge_hidens*.param.",
    "confidence": "high"
  },
  {
    "id": "S6-B-13",
    "file": "trinity/phase2_momentum/run_momentum_phase.py",
    "line": 109,
    "class": "deadcode",
    "severity": "S4",
    "claim": "The momentum runner's adaptive-stepping monitor list retains the group headings \"# Cooling parameters\" (:109) and \"# Bubble properties\" (:111), copied verbatim from the transition runner (:117, :119), even though the momentum module docstring states Eb = 0, Ed = Td = 0, no energy/temperature evolution, and no thermal pressure. Both files also carry the unsourced provenance note \"Based on analysis of the top 30 most variable parameters\" (:103 / :111).",
    "evidence": "trinity/phase2_momentum/run_momentum_phase.py:103, :109, :111 vs :3; trinity/phase1c_transition/run_transition_phase.py:111, :117, :119",
    "expected": "The momentum monitor list should contain only quantities the momentum phase actually updates; and the '30 most variable parameters' claim should name the analysis or be removed.",
    "failure_scenario": "Monitored keys that never change in the momentum phase contribute a constant 0 dex to the max-dex reduction, so they can never trigger a step reduction but do dilute nothing (max, not mean). Harm is low unless the reduction is a mean or unless a stale cooling/bubble key holds a value from the transition phase and is compared against itself forever. The real cost is that a reader cannot tell which monitored keys are live.",
    "repro": "Count the monitor list entries (is it 30?) and log which of them ever change during a momentum-phase run.",
    "confidence": "medium"
  },
  {
    "id": "S6-B-14",
    "file": "trinity/phase2_momentum/run_momentum_phase.py",
    "line": 506,
    "class": "state",
    "severity": "S3",
    "claim": "The momentum phase describes its energy state three different ways in one file: \"Eb ~= 0 (thermal pressure negligible)\" (:3 Overview), \"**Eb = 0**: Energy-driven terms negligible\" (:3 Key Features 1), \"thermal pressure is negligible (Eb ~= 0)\" (:462 docstring), and \"# Initialize state (Eb = 0 in momentum phase)\" (:506). The initialization comment states an exact assignment, not an approximation — whatever Eb remained at the end of the transition phase is discarded rather than carried.",
    "evidence": "trinity/phase2_momentum/run_momentum_phase.py:3, :462, :506; transition handoff described at trinity/phase1c_transition/run_transition_phase.py:368",
    "expected": "State whether Eb is set to exactly 0 or carried forward, and record the discarded energy if it is set to 0. The docstrings should not alternate between '=' and '~='.",
    "failure_scenario": "Combined with S6-B-01: if the transition phase exits on ram-pressure dominance rather than on the energy floor, the discarded Eb is not small, and the total energy budget of the run has an unreported loss at the phase boundary. Any energy-conservation diagnostic across the full run would show a jump with no accounting entry.",
    "repro": "Record Eb at the last transition snapshot and at the first momentum snapshot in dictionary.jsonl for param/simple_cluster.param and docs/dev/performance/f1edge_hidens*.param; compare against the energy floor constant.",
    "confidence": "high"
  },
  {
    "id": "S6-B-15",
    "file": "trinity/phase1c_transition/run_transition_phase.py",
    "line": 251,
    "class": "other",
    "severity": "S4",
    "claim": "\"# Force Computation (same as in energy_implicit)\" asserts that this file's force block is a duplicate of another module's, with no revision pin and no test named. The momentum twin's force block carries no such claim (trinity/phase2_momentum/run_momentum_phase.py:186-187), so there are at least three force implementations in play (energy_implicit, transition, momentum) with one unenforced equivalence assertion between two of them.",
    "evidence": "trinity/phase1c_transition/run_transition_phase.py:251; no counterpart at trinity/phase2_momentum/run_momentum_phase.py:186",
    "expected": "Either import the shared implementation, or pin the claim with a test that asserts the two produce identical forces for identical state.",
    "failure_scenario": "A fix applied to energy_implicit's force computation is not propagated here; forces become phase-dependent for identical physical state, producing a discontinuity in the force budget at the energy->transition boundary that no test catches.",
    "repro": "Diff compute_forces_pure in trinity/phase1c_transition/run_transition_phase.py against the corresponding function in the energy_implicit module.",
    "confidence": "medium"
  },
  {
    "id": "S6-B-16",
    "file": "trinity/phase1c_transition/run_transition_phase.py",
    "line": 724,
    "class": "state",
    "severity": "S4",
    "claim": "Each file defines 'collapse' two ways. The timestep control says \"# Only during collapse (negative velocity = inward motion)\" (:724 / trinity/phase2_momentum/run_momentum_phase.py:803) — v2 < 0 alone. The collapse detector says \"# Collapse detection: velocity negative AND radius decreasing\" (:771 / :824) — a conjunction. Additionally, by line order the detector (:771) runs AFTER the timestep control (:719-731) and after the results store (:736), so the isCollapse flag consumed by the next segment's shell-mass freeze (:541 / :602) is one segment old.",
    "evidence": "trinity/phase1c_transition/run_transition_phase.py:724 vs :771, and the ordering :719-731 -> :736 -> :743 -> :771; twin at trinity/phase2_momentum/run_momentum_phase.py:803 vs :824",
    "expected": "One definition of collapse, or two clearly named conditions; and a stated policy on the one-segment lag in the shell-mass freeze.",
    "failure_scenario": "At collapse onset the shell mass continues to accrete for one extra segment before freezing, and if the shell re-expands the freeze persists one segment too long. Magnitude is bounded by dt_segment, which is exactly the quantity being shrunk during collapse, so the error is self-limiting but non-zero at onset — the moment the mass matters most.",
    "repro": "Log isCollapse, v2, R2-R2_prev and shell_mass per segment through a collapse onset.",
    "confidence": "medium"
  },
  {
    "id": "S6-B-17",
    "file": "trinity/phase1c_transition/run_transition_phase.py",
    "line": 398,
    "class": "regime",
    "severity": "S4",
    "claim": "\"# Update cool_alpha to match ODE-evolved v2 (preserves ODE continuity)\" sits in the phase Initialization block (:385-448), i.e. it runs once at phase entry, not per segment. The comment asserts a dependence of cool_alpha on v2 and that matching it preserves continuity. Meanwhile 'Cooling parameters' are in the per-segment adaptive-stepping monitor list (:117), implying cooling parameters are expected to vary during the phase.",
    "evidence": "trinity/phase1c_transition/run_transition_phase.py:398 within the Initialization block :385-448, vs :117",
    "expected": "State whether cool_alpha is intended to be a one-shot entry fixup or a per-segment quantity. If it depends on v2 and v2 evolves, a single update makes the continuity claim true only at t = t_entry.",
    "failure_scenario": "cool_alpha goes stale as v2 evolves within the transition phase, so Ed_energy_balance (which feeds the min() at :242) is computed with an entry-time cooling coefficient. Since the min() branch selection depends on Ed_energy_balance's magnitude, a stale cool_alpha can change WHICH branch is selected, not just its value.",
    "repro": "Check whether cool_alpha is recomputed anywhere inside the segment loop; log cool_alpha and v2 per segment through the transition phase.",
    "confidence": "medium"
  },
  {
    "id": "S6-B-18",
    "file": "trinity/phase1c_transition/run_transition_phase.py",
    "line": 456,
    "class": "numerical",
    "severity": "S4",
    "claim": "The energy floor is checked in two places with two different characterisations: as a terminal solve_ivp event, \":456 Transition phase events: energy_floor (phase ending), min_radius, velocity_runaway\", and as a post-segment scalar test, \":765 Safety fallback: absolute energy floor\". The word 'absolute' at :765 suggests it may not be the same threshold as the event's.",
    "evidence": "trinity/phase1c_transition/run_transition_phase.py:456, :765, and the constant at :97",
    "expected": "Both checks should reference the single constant documented at :97, or the comments should say why two thresholds exist.",
    "failure_scenario": "If the event threshold and the fallback threshold differ, the phase end time depends on which mechanism fires first, and the two can disagree by up to one segment. If they are the same, the fallback is unreachable dead code unless events are disabled.",
    "repro": "Check whether both sites read the same module constant; instrument which of the two actually fires.",
    "confidence": "medium"
  },
  {
    "id": "S6-B-19",
    "file": "trinity/phase1c_transition/run_transition_phase.py",
    "line": 448,
    "class": "other",
    "severity": "S4",
    "claim": "The dissolution criterion is documented as \"Persistent dissolution timer: track how long shell_nMax < nISM\" (:448 / trinity/phase2_momentum/run_momentum_phase.py:530) and \"Dissolution check: persistent timer based on shell_nMax < nISM\" (:805 / :858), but the required persistence duration is never stated in either file, and no unit is given for shell_nMax or nISM.",
    "evidence": "trinity/phase1c_transition/run_transition_phase.py:448, :805; trinity/phase2_momentum/run_momentum_phase.py:530, :858",
    "expected": "State the persistence threshold with its unit (Myr? number of segments?) and the units of shell_nMax/nISM.",
    "failure_scenario": "If the timer accumulates in segments rather than Myr, the dissolution time becomes a function of the adaptive stepping (which varies by ~1.26x per segment and collapses to 5e-4 Myr during collapse), making the stopping fate step-size dependent rather than physical.",
    "repro": "Find the accumulator's unit; run the same config with DT_SEGMENT_INIT halved and compare the dissolution time.",
    "confidence": "medium"
  },
  {
    "id": "S6-B-20",
    "file": "trinity/phase1c_transition/run_transition_phase.py",
    "line": 749,
    "class": "divergence",
    "severity": "S4",
    "claim": "The phase-transition criterion is stated as the ratio P_ram/(P_b + P_ram) > threshold (:749). No comment documents a guard for the denominator, and the surrounding prose establishes that P_b tends to zero by construction in this phase (the whole phase exists because \"bubble thermal energy (Eb) becomes negligible\", :3) while P_ram would also vanish if feedback shut off.",
    "evidence": "trinity/phase1c_transition/run_transition_phase.py:749, with :3 and :765 establishing P_b -> 0",
    "expected": "A documented guard for P_b + P_ram == 0, or a note that P_ram is bounded away from zero while this phase runs.",
    "failure_scenario": "If both pressures reach zero (feedback exhausted and Eb at the floor at the same segment), the ratio is 0/0 -> nan, the '> threshold' comparison is False, and the phase silently does NOT transition — falling through to the :765 fallback or, if that also mis-compares, to max_segments/'unknown'.",
    "repro": "Construct a state with Pb and P_ram both ~0 at a segment boundary and evaluate the criterion; check for a nan guard.",
    "confidence": "low"
  },
  {
    "id": "S6-B-21",
    "file": "trinity/phase1c_transition/run_transition_phase.py",
    "line": 520,
    "class": "deadcode",
    "severity": "S4",
    "claim": "The transition runner ships two diagnostic blocks with no counterpart in the momentum twin: \":389 --- PHASE BOUNDARY DIAGNOSTIC ---\" ... \":396 --- END DIAGNOSTIC ---\" in the initialization path, and \":520 --- Ed diagnostic at first segment (quantify **the original discontinuity**) ---\" inside the segment loop. The second is an explicit admission that a discontinuity existed at the implicit->transition boundary and that instrumentation for it remains in production.",
    "evidence": "trinity/phase1c_transition/run_transition_phase.py:389-396, :520; no counterpart in trinity/phase2_momentum/run_momentum_phase.py",
    "expected": "Either the discontinuity is resolved and the diagnostics are removed, or a comment records the current residual magnitude and why the instrumentation stays.",
    "failure_scenario": "Diagnostics in the segment loop run every first segment of every run in every sweep member; if any of them touches the logging or SPS path they carry a per-run cost. More importantly the phrase 'the original discontinuity' leaves unstated whether the discontinuity is fixed, which is precisely the property the min() model at :3/:200 claims to deliver.",
    "repro": "Read what the :520 block computes and whether its output is still nonzero for param/simple_cluster.param.",
    "confidence": "medium"
  }
]
```
