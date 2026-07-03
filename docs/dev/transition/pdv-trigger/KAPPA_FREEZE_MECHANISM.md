# The mid-implicit freeze — mechanism, physics identity, instrumentation, and how other codes avoid it

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

**Status (2026-07-02):** mechanism identified from committed sweep data + code audit; live
single-run reproduction in §4 (see status note there). This doc answers the maintainer's
challenge: *"are we sure f_κ is a no-go? I'm worried 'breaks non-monotonically' is a false
inference — check vigorously, maybe assumptions/approximations/caps avoid the bugs."*
**Verdict: the maintainer was right to be suspicious.** The "dead windows" are not physics
bands. They are a *silent solver failure with a physical root cause* — and the root cause is
a known, nameable piece of astrophysics the current solver refuses to represent.

---

## 1. The symptom (what "froze mid-implicit" means)

A run at certain `cooling_boost_kappa` values ends prematurely (t_final ≈ 0.4–1.4 Myr, far
short of stop_t) while still in the energy-driven implicit phase, without firing any trigger,
with θ = L_loss/L_mech frozen at its last value — and **exit code 0** (`main.py:211` returns 0
unconditionally; per-run health must be judged from `t_final`/`phase_final`, FINDINGS §9a.3).

Autopsy of the 819-run sweep (`data/make_kappa_freeze_autopsy.py` →
`data/kappa_freeze_autopsy.csv`, committed 2026-07-02):

- **38/819 runs froze mid-implicit.** The freeze rate *rises* with f_κ (~1/63 at f=1–2 up to
  ~5–7/63 at f=12–48) but **one run froze at f_κ = 1.0** (unboosted baseline, cell
  1e5/sfe0.1/n1e5, θ_max 0.9516): the failure mode pre-exists the knob; kappa only aggravates it.
- **34/38 froze at θ_max ≥ 0.8** (most 0.85–0.95), vs median θ_max 0.636 for healthy no-fire
  runs: freezes concentrate **on approach to the θ = 0.95 crossing**. They are would-fire runs
  that died at the door, not cold dead windows.
- The 23 "non-monotonic" arms (17 cells) of FINDINGS §9a decompose into: 12 froze-on-approach
  + 8 healthy-at-2-Myr with θ_max 0.87–0.93 (near-threshold at the stop_t=2 horizon, which the
  standing rules outlaw for θ quotes; precedent: the diffuse multiplier arm fires at t≈5.04 Myr)
  + 3 froze-early (θ 0.52–0.59 — the §8e mode, 2 cells, 3/819 runs). **None** ran healthy to a
  rule-compliant horizon and stayed cold.

So FINDINGS §9a's "interleaved firing bands and breakdown windows" over-read the data. Corrected
statement: *the kappa knob's fire set has holes because the solver crashes near the
cooling-balance crossing (and occasionally early), with a crash rate that rises with f_κ —
a numerical failure with a physical trigger, not band-structured physics.*

## 2. The code mechanism (file:line chain, audited 2026-07-02)

`cooling_boost_kappa` enters at three sites in `trinity/bubble_structure/bubble_luminosity.py`,
all as `f_κ · C_thermal`:

1. `bubble_luminosity.py:297` — the Weaver Eq. 33 evaporative-flux seed: `dMdt ∝ f_κ^{2/7}`.
2. `bubble_luminosity.py:377-388` — the near-front IC (Weaver Eq. 44): conduction-anchor
   thickness `dR2 ∝ f_κ/dMdt`, boundary gradient `dTdr ∝ dMdt/f_κ`.
3. `bubble_luminosity.py:413-416` — the T″ term of the structure ODE:
   `Pb/(f_κ·C_thermal·T^{5/2})·(…)`.

The freeze chain:

- The bubble-structure solve finds `dMdt` as an **eigenvalue** of the boundary-value problem
  `v(R1) = 0` (fsolve at `bubble_luminosity.py:254-260`). Boosted conduction raises the solved
  `dMdt` (measured ×1.08–1.17 at f_κ=2, KAPPA_EFF_SCOPING §6a).
- The β–δ hybr solver **gates acceptance on `dMdt > 0`**: `get_betadelta.py:861-869` raises
  `_NoPhysicalRoot` for any trial (β, δ) whose structure solve returns `dMdt ≤ 0` (or fails).
- On `no_physical_root` the implicit-phase runner does **not** break: it logs a warning, holds
  the last physical `bubble_dMdt`/`bubble_Lloss`, and continues
  (`run_energy_implicit_phase.py:835-845`). θ is frozen from this point on.
- With dt pinned at `DT_SEGMENT_MIN = 1e-4` and `MAX_SEGMENTS = 5000`
  (`run_energy_implicit_phase.py:113-115`), the phase grinds forward ≈ 0.5 Myr of simulation
  time and exits with `termination_reason = "max_segments"` — **no `SimulationEndCode`, no
  `EndSimulationDirectly`** (`run_energy_implicit_phase.py:1364-1372`) — which matches the
  observed freeze horizon (§8e froze at t≈0.44 after seizing at segment ~6 ≈ t 0.04–0.1).
- Secondary effects that shape the *bands*: each sweep arm cold-starts (no continuation across
  f_κ), the hybr options are tuned to the f_κ=1 noise floor (`get_betadelta.py:74`,
  `eps=3e-4` — "the residual noise floor measured in the f_κ=1 transect probe"), and warm-starts
  go stale when f_κ moves the solution branch → the root-finder lands on different
  branches/plateaus at different f_κ → interleaved fire/dead windows.
- The θ_max = 4.55 spike at f_κ=16 is the call-level observer artifact (⛔ CONTAMINATION #3),
  not physics.

**What it is NOT:** a thinning-boundary-layer resolution failure. `dR2 ∝ f_κ/dMdt` with
`dMdt ∝ f_κ^{2/7}` means the conduction anchor layer *thickens* like `f_κ^{5/7}`, and
`test/test_dR2min_magic_number.py` shows the integrator handles layers down to `dR2/R2 ~ 1e-11`.

## 3. The physics identity: evaporation → condensation reversal (McKee & Cowie 1977)

The `dMdt > 0` acceptance gate hardcodes **Weaver's evaporation-only regime**. But the classical
conduction-front result (McKee & Cowie 1977) is that the interface mass flux has **two regimes**:
when radiative losses in the interface are small compared to conductive heating, the shell
*evaporates* into the bubble (dMdt > 0); when cooling overcomes conductive heating, evaporation
stops and the hot medium **condenses onto the shell** (dMdt < 0). El-Badry+2019 (the workstream's
own reference model!) exhibits exactly this: conduction evaporates the shell early, and when
interface cooling dominates the flux reverses and the hot gas condenses.

Now note what the cooling-balance trigger *is*: θ = L_loss/L_mech → 0.95 means interface cooling
approaching the total heating supply. **The regime the transition trigger must pass through is
precisely the regime where the physical dMdt crosses zero.** That is why 34/38 freezes sit at
θ ≥ 0.8: the solver walks the bubble toward cooling balance, the physical eigenvalue dMdt heads
for zero and goes negative (condensation), the gate declares "no physical root", and the runner
freezes — *at the exact moment the physics is trying to hand the run to the momentum phase.*

So the freeze is a **model-domain boundary**, not a bug in the integrator: the current
bubble-structure formulation (Weaver Eq. 33/44 anchoring, evaporation-only) has no condensation
branch, and the code's only response to reaching that boundary is to stop updating.

Two corollaries:

- **The 3 froze-early runs** (θ 0.52–0.59, incl. §8e's f_κ=8 simple_cluster signature) are the
  same gate tripping *before* global cooling balance — plausibly a local/branch effect at high
  f_κ (boosted conduction can push the *front* into the condensing regime while the bubble as a
  whole is still gaining), or a stale-warm-start branch loss. The live repro (§4) decides.
- **The multiplier knob dodges this by construction**: `cooling_boost_fmix` scales L_cool *after*
  the structure solve (`get_betadelta.py:346,354` region), so it never changes the dMdt
  eigenvalue problem — which is why theta5/theta5b arms cross θ = 0.95 and hand off cleanly
  (fire-vs-drain race aside, FINDINGS §11) while kappa arms crash at the same crossing.

## 4. Live single-run reproduction

**Status: IN FLIGHT (2026-07-02)** — a local reproduction (simple_cluster, f_κ ∈ {1, 4, 8},
stop_t=2, mechanism diagnosis only — θ from these runs is NOT calibration data) is running; this
section gets its numbers when it lands. Expected signature per §2: warnings
`beta-delta: no physical (dMdt>0) root at segment N …` starting at segment ~6, θ frozen ~0.53,
`Implicit phase completed: max_segments` at t_final ≈ 0.44, exit 0.

## 5. Instrumentation (added 2026-07-02, log-only, behavior-neutral)

To make the freeze identifiable at a glance (and gone when fixed), the implicit runner now
carries a **no-root streak tracker** and a **dMdt approach trace** (all logging-only; physics,
outputs, and the `betadelta_phase_summary` signature untouched):

- `freeze-watch` DEBUG line per accepted segment: `t`, `dMdt`, θ — grep `freeze-watch` to watch
  dMdt walk toward zero as θ climbs.
- First `no_physical_root` in a streak logs WARNING (as before); repeats within a streak are
  demoted to DEBUG with a WARNING heartbeat every 500 segments (a frozen phase used to emit
  ~5000 identical warnings).
- At streak = 50 a one-time loud diagnosis names the state: frozen-implicit signature, dMdt
  gate, condensation-regime pointer to this doc.
- The end-of-phase completion line appends the freeze state when the phase dies frozen:
  `frozen: N-segment no-root streak since t=…`.

Exact lines: see `run_energy_implicit_phase.py` around the `no_physical_root` block (§2 refs).

## 6. How other codes/papers avoid this (searched 2026-07-02)

- **El-Badry+2019** (MNRAS 490, 1961): 1D **Lagrangian hydro** with conduction as an explicit
  diffusion operator — mass flux through the interface is an *outcome*, free to change sign;
  there is no eigenvalue gate to trip. Their bubbles evaporate early and condense late; the code
  follows the reversal smoothly. This is the "just solve the PDE" escape.
- **McKee & Cowie 1977** (ApJ 215, 213): the analytic criterion — evaporation vs condensation set
  by the saturation/cooling parameter (critical radius where radiative losses balance conductive
  heating). A semi-analytic code can *classify the regime first* and use the matching branch
  instead of assuming evaporation.
- **Cowie & McKee 1977** (ApJ 211, 135): **saturated conduction** — the heat flux is capped at
  the free-streaming value `q_sat ≈ 5φ_s ρ c_s³` (φ_s ≈ 0.3). TRINITY currently applies Spitzer
  κ **unbounded** (audit §2: no saturation cap anywhere in the bubble solve. NOTE
  F_KAPPA_FUNCTIONAL_FORM §11 *derived* the physical f_κ window from exactly this cap but it was
  never implemented as a solver bound). A saturation cap bounds dMdt and tames the high-f_κ limit.
- **Vieser & Hensler 2007** (A&A 472, 141): self-consistent spherical cloud models **with
  saturated conduction AND cooling together** — they solve through both evaporation and
  condensation states; the standard reference for doing this stably in 1D.
- **Lancaster+2021a/b, +2024**: sidestep conduction entirely — the interface is a turbulent
  fractal mixing layer in the **efficiently-cooled limit** (most input energy radiated by
  construction); the momentum-driven solution is imposed, so there is no conduction eigenvalue
  to lose. TRINITY's `multiplier` knob + `cooling_balance` trigger is structurally the same
  move: prescribe the loss level, let the phase switch, never solve a condensing front.
- **Semi-analytic superbubble models with catastrophic cooling** (Silich et al. 2003/2004;
  Tenorio-Tagle et al. 2005/2007; Mac Low & McCray 1988 lineage): treat strong-cooling onset as
  a **regime switch of the bulk solution** (energy- → momentum-driven, or a catastrophically
  cooled wind branch) rather than pushing one solution branch through it — i.e. exactly what a
  `no_physical_root → hand off to momentum` semantics would do.

## 7. The fix ladder (ranked; from the audit + §6)

1. **Semantic fix (LOW effort, near-zero risk): `no_physical_root` persistence ⇒ transition.**
   dMdt ≤ 0 at every reachable (β, δ) is not "no answer" — it is the physics saying the
   energy-driven conductive solution has ended (condensation onset ≈ cooling balance). Route a
   *persistent* no-root streak to the same handoff as `energy_to_momentum`
   (`run_energy_implicit_phase.py:1120-1131`) instead of grinding to `max_segments` with a
   misleading clean exit. This alone likely closes most "dead windows" (34/38 were en route to
   firing) *without touching the physics*. Guard: require a streak (e.g. 50 segments) so a
   transient rejection still recovers, and record a distinct termination_reason
   (`no_physical_root_handoff`) so it is auditable.
2. **Continuation in f_κ across sweep arms (LOW-MED):** warm-start (β, δ, dMdt) from the previous
   f_κ arm — kills the branch-selection component of the bands. Sweep-harness change only.
3. **Saturated-conduction cap (MED-HIGH, PHYSICAL):** `min(κ_S|∇T|, q_sat)` in the ODE + a
   numerical near-front IC (the closed-form Eq. 44 IC assumes pure T^{5/2}). Physically the
   right ceiling (Cowie & McKee 1977; Vieser & Hensler 2007) and also what F_KAPPA §11 assumed;
   bounds dMdt growth with f_κ. This is the real cure if kappa is ever to be a production knob.
4. **Condensation branch (HIGH, the faithful model):** allow dMdt < 0 (El-Badry-style interface),
   i.e. replace the evaporation-only Weaver anchoring. Research-grade; only worth it if the
   El-Badry coupling (evaporation suppression) gets ported (KMIX_SELFCONSISTENT ladder).
5. **Solver-hygiene items (LOW, incremental):** fallback ladder on no-root (cold-restart the dMdt
   fsolve before giving up); bound single-point spike depth in the tolerant monotonic guard;
   restrict θ observers to accepted segments (already the standing rule for harvesting).

## 8. Consequences for the knob decision (kept honest)

- The **multiplier remains the right production knob** — not because kappa's physics is banded,
  but because the multiplier structurally avoids the condensing-front domain boundary that the
  current solver cannot cross, and its calibration (window [4, 4.5], law rms 0.064 dex) is
  measured under the standing rules.
- FINDINGS §9a's conclusion ("knob-choice argument gets stronger") **survives**, but its
  *mechanism claim* ("interleaved firing bands and breakdown windows" as a property of the knob)
  is **superseded** by this doc: the windows are crash artifacts + an outlawed stop_t=2 horizon.
  §9a has been annotated (dated) rather than rewritten.
- `'auto'` interpolation risk (§9a.2) is *downgraded but not cleared*: interpolated f_κ values
  no longer face physics dead bands, but they still face a crash-prone crossing until fix #1
  lands.
- If fix #1 (streak ⇒ handoff) is implemented, a **rule-compliant kappa re-validation** (5 Myr,
  θ_max, the 8 configs — "theta5k") becomes cheap and would test whether kappa fires
  monotonically once the solver is allowed to leave the energy phase at the physical boundary.

## 9. Reproduce

- Autopsy: `python docs/dev/transition/pdv-trigger/data/make_kappa_freeze_autopsy.py`
  (reads the committed `data/summary.csv`; writes `data/kappa_freeze_autopsy.csv`).
- Live repro: simple_cluster + `cooling_boost_kappa 8`, `stop_t 2` (§4; params under the session
  scratchpad, mechanism-only — not calibration data).
- Literature anchors (searched 2026-07-02): McKee & Cowie 1977; Cowie & McKee 1977;
  El-Badry+2019 (arXiv:1902.09547); Vieser & Hensler 2007 (A&A 472, 141); Lancaster+2021a/b
  (arXiv:2104.07691/2104.07722), +2024 (arXiv:2405.02396); Silich+2003/2004,
  Tenorio-Tagle+2005/2007 (catastrophic cooling); Mac Low & McCray 1988.
