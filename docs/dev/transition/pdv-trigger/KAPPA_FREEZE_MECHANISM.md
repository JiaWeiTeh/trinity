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

**Status (2026-07-02):** mechanism identified from committed sweep data + code audit, and
**confirmed by live local reproduction** (§4: the solver converges to a *negative* dMdt root —
condensation — and the gate refuses it). This doc answers the maintainer's
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

**Primary-source recheck (maintainer, 2026-07-03) — three sharpenings:**

- **Weaver himself already had 40% interface cooling.** Weaver et al. 1977 Paper II (§V) state
  that ~40% of the conductive heat flux is radiated in the interface (O VI-type resonance
  lines) and only ~60% drives evaporation — Paper I's neglect of front radiative losses was an
  approximation they removed for the ionization structure (not the dynamics); saturation is
  negligible at the interface for their fiducial bubble. **The classical benchmark front budget
  is already 60/40**: a κ boost that raises supply *and* interior density (with n² cooling
  responding) tipping it past 100% is a modest push, not exotica. This is the quantitative
  anchor for why the reversal is close by.
- **TRINITY's closure literally has no condensation branch.** The TRINITY method paper's
  Eq. 15 family gives T ∝ Ṁ^{2/5} (visible in the code: the Eq. 44 IC at
  `bubble_luminosity.py:377-388` computes `T = (const · dMdt · dR2 / 4πR2²)^{2/5}`) — the
  Weaver similarity profile does not exist for Ṁ < 0. The dMdt>0 gate is not a bolted-on
  sanity check; it marks the edge of the profile family the whole β–δ structure is built from.
  This confirms fix #4's ranking: a condensation branch needs a *different profile family*,
  not a relaxed gate.
- **The planar analogue's eigenvalue is unique** (Tan, Oh & Gronke 2021 §2.2, citing Kim & Kim
  2013; sign set by pressure vs P_crit, Zel'dovich & Pikel'ner 1969). See §5 for what this does
  to the branch-multiplicity caveat.

Two corollaries:

- **The 3 froze-early runs** (θ 0.52–0.59, incl. §8e's f_κ=8 simple_cluster signature) are the
  same gate tripping *before* global cooling balance — plausibly a local/branch effect at high
  f_κ (boosted conduction can push the *front* into the condensing regime while the bubble as a
  whole is still gaining), or a stale-warm-start branch loss. The live repro (§4) decides.
- **The multiplier knob dodges this by construction**: `cooling_boost_fmix` scales L_cool *after*
  the structure solve (`get_betadelta.py:346,354` region), so it never changes the dMdt
  eigenvalue problem — which is why theta5/theta5b arms cross θ = 0.95 and hand off cleanly
  (fire-vs-drain race aside, FINDINGS §11) while kappa arms crash at the same crossing.

## 4. Live single-run reproduction (ran 2026-07-02, local container; simple_cluster + `cooling_boost_kappa`, mechanism-only — θ from these runs is NOT calibration data)

**The smoking gun, verbatim from the f_κ=8 log:**

```
WARNING | beta-delta: no physical (dMdt>0) root at segment 2 (t=3.410339e-03 Myr):
non-physical dMdt=-84.76 at (beta=1…
```

The structure solve did not fail — it **converged to a negative-dMdt root** (−84.76 Msun/Myr;
f_κ=7.5 gives −85.22, f_κ=16 gives −53.09 at the same epoch) and the acceptance gate refused it.
The condensation branch exists and the solver finds it; the model just declines to walk it.

| arm | no-root events | behavior |
|---|---:|---|
| f_κ=1 | 0 | healthy (wall-limited at t≈0.08 in 900 s — local runs are slow, not frozen) |
| f_κ=4 | 0 | healthy, converged (t≈0.05 at timeout); long variant handed off Eb→0 → momentum → clean recollapse fate at t≈0.31 |
| f_κ=7.5 | 6 | rejection burst from segment 2 (t=3.4e-3) |
| f_κ=8 | 125 | bursts from segment 2; recovers between bursts; still implicit, θ≈0.52–0.53 through t≈0.35 (container died mid-run) — on course for the sweep's t_final≈0.44 grind exit |
| f_κ=8 (legacy β-δ solver) | 0 | the legacy grid solver has no dMdt gate — the freeze is a hybr-gate semantic, θ 0.71 at t=0.014 |
| f_κ=16 | 18 | same early-burst signature |
| f_κ=8 + `MAX_SEGMENTS=40` monkeypatch | — | implicit exits early via max_segments → continues to momentum → **completes cleanly** (Eb→0, shell recollapse, proper end code). Proof-of-concept for fix #1: hand off instead of grinding and the run gets a well-formed fate. |

Three readings that sharpen §3:

- **The rejections are intermittent bursts, not a one-way seize** (f_κ=8: segments 2–6 at
  t≈0.0034–0.0044, later consecutive clusters at t≈0.34–0.35). Between bursts the warm-start
  recovers a dMdt>0 root and the run advances. The HPC "freeze" is the burst that never ends.
- **The early-freeze mode and the on-approach mode are one mechanism.** The dMdt sign is set by
  the *front's local* heating/cooling budget, not by global θ. Boosted κ at f≳7.5 pushes the
  front into the condensing regime almost immediately (t≈0.003, θ still ~0.5 — the §8e
  θ≈0.53 signature); unboosted or mildly boosted runs only get there when global θ→0.95 drags
  the whole bubble to cooling balance (the 34/38 on-approach freezes). Same reversal, two roads.
- **θ matches**: local f_κ=8 holds θ≈0.52–0.53 — the sweep's frozen θ_max=0.5331 on the
  matching cell, reproduced a third time, on a third machine.

Artifacts: session scratchpad `kappa_repro/` (params, logs, `interim_table.jsonl`;
`drive_fk8_maxseg.py` for the monkeypatch driver). Scratchpad is local-only — the numbers above
are the durable record, per the 💾 banner.

## 5. Instrumentation (LANDED 2026-07-02, log-only, behavior-neutral; betadelta tests 28/28 pass)

**Demo (f_κ=8, `stop_t 0.006`, `log_level DEBUG`, exit 0):** segment 1 solves cleanly at
dMdt=+1.121e3 with Lloss/Lgain=0.5316 — **the frozen θ≈0.53 is literally segment 1's ratio,
held from the first rejection onward**. From segment 2 the streak counter tracks the gate
refusals, each reporting the *negative root the solver actually found*: −84.76, −99.82, −75.99,
−56.05, −40.6, −29.24, −20.75, −279.7 … — the front hovers around the reversal (roots decaying
toward 0 = drifting back toward evaporation, then a deeper condensing excursion). The abrupt
+1121 → −85 swing between consecutive segments: the planar analogue's mass-flux eigenvalue is
**unique** for given boundary conditions (Tan, Oh & Gronke 2021 §2.2; maintainer recheck
2026-07-03), so genuine two-branch multiplicity is now the *disfavored* explanation — the
prior favors **fast-moving control parameters** (interior n, P between segments) or
root-finder bracket behavior. TRINITY's spherical v(R1)=0 problem is not literally the planar
one, so multiplicity stays a caveat, not a hypothesis. The freeze-watch trace on a
condense-vs-fire pair (dense k6 vs k8) is the discriminating experiment.

**Trace verdict (RAN 2026-07-03, dense nCore=1e6, k6=condense-arm vs k8=fire-arm, identical
early dt sequences — a controlled pair):**

- **The rejected root evolves smoothly — no bracket chaos.** k8's negative eigenvalue walks a
  continuous arc: −17.7 → −36 → decaying steadily to −14.8 over segments 2–21, a wobble, a dip
  to −5.3, then at segment 28 the root **recovers through zero to +65.3**, the structure solve
  is accepted, and cooling_balance fires within that segment (matches the HPC arm: n_impl=28,
  θ_first=0.617). k6 walks the same early arc but nearly recovers *early* (−16 → −4.0 by
  segment 8), then takes a **second dive** (−37.9 at segment 9) and decays only to ~−24 by
  segment 23 (local timeout) — on HPC it never recovers within the 50-segment window → handoff.
- **Verdict on the maintainer's bug question:** the solver is finding a well-defined,
  continuously-evolving eigenvalue every segment and the gate refuses it — consistent with the
  Tan–Oh–Gronke uniqueness prior; no erratic sign-flipping, no branch-hopping signature. The
  fire-vs-condense outcome is decided by whether the front's budget *recovers to evaporation*
  within the streak window — real trajectory physics, not solver noise.
- **One honest numerics caveat:** each trace has one discontinuous jump in the rejected-root
  sequence (k6: −4.0→−37.9 at segment 9; k8: −15.3→−5.3→−20.5 around segments 25–27). These
  correlate with discrete events in the segment loop (the `COOLING_UPDATE_INTERVAL`=5e-3 Myr
  table refresh is the prime suspect, not pinned). So: **the race is physical, but the exact
  f_κ location of its edge is sensitive to loop discretization** — per-config f_κ_fire values
  are razor-edge quantities; don't over-interpret them (unlike the multiplier's f_fire, which
  sits on the smooth θ₁-collapse law).

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
   *persistent* no-root streak to the same handoff as `cooling_balance` (break without
   `EndSimulationDirectly` → main runs 1c → momentum) instead of grinding to `max_segments` with
   a misleading clean exit. This alone likely closes most "dead windows" (34/38 were en route to
   firing) *without touching the physics*. Guard: require a streak (50 segments;
   observed healthy bursts are ≤ 8) so a transient rejection still recovers, and record a
   distinct termination_reason (`no_physical_root_handoff`) so it is auditable.

   **LANDED 2026-07-03** (`NO_ROOT_HANDOFF_STREAK = 50`, `run_energy_implicit_phase.py`).
   Verification: (a) handoff demo — `runs/drive_noroot_handoff_check.py` (threshold
   monkeypatched to 3, simple_cluster f_κ=8): streak 3 → condensation diagnosis →
   `Implicit phase completed: no_physical_root_handoff` → transition phase → momentum entry →
   clean `STOPPING_TIME` end code. The run that used to freeze flows through the phase
   machinery. (b) inertness — structural: the branch cannot execute below a 50-streak (observed
   healthy bursts ≤ 8); full pytest 614/614 green. A cross-process byte-identity gate proved
   **unattainable in the local container**: two runs of IDENTICAL code differ from dictionary
   row 1 at the same byte offset (`Lmech_SN` ~1e-18 — cubic-spline ringing on the SN noise
   floor — plus single-ULP wobbles in bubble-structure arrays; measured 2026-07-03, control
   experiment `fk8_identity` vs `fk8_identity2`). That is *pre-existing run-to-run FP
   nondeterminism* (prime suspect: unpinned OpenBLAS/OMP threading — the HPC sbatch pins
   `OMP_NUM_THREADS=1`, local runs don't; PYTHONHASHSEED ruled out by direct SPS-interpolator
   probe), and it exonerates the fix by direct A/A comparison. ⚠️ Consequence for the
   workstream's rule-5 "byte-identical `dictionary.jsonl`" gates: pin BLAS/OMP threads to 1
   (as on HPC) or expect ULP-level drift — a same-code A/A control run is now the mandatory
   companion to any local byte-identity claim. (c) harvest semantics — a handoff reaches
   momentum with θ<0.95, so `harvest_theta_max.py` classifies it like DRAIN, never as a θ
   transition. Note the handoff does NOT count as `cooling_fired` — it is a fate, not a
   trigger.
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
- ~~If fix #1 (streak ⇒ handoff) is implemented, a rule-compliant kappa re-validation becomes
  cheap~~ **DONE — theta5k RAN (2026-07-03, Helix, FINDINGS §12)**: 56/56 proper fates, **zero
  freezes** (5 arms exit via the condensation handoff — exactly the old "dead window" cells,
  validating both the fix and this doc's mechanism at scale). Answer to the monotonicity
  question: the fire set is *still* non-monotonic, but honestly so — the front condenses (or
  the shell drains/dissolves) before global θ crosses; θ_max itself rises ~monotonically. **No
  single f_κ fires the whole band** (best: f_κ=12, 5/6) vs the multiplier's [4, 4.5] at 6/6 —
  the production-knob choice is now measured on crash-free, rule-compliant, like-for-like data.
  The old sweep's simple_cluster "fires at f_κ=16" was a solver artifact (rule-compliant it
  CONDENSES at θ=0.624), which also hardens the `'auto'` demotion (its lookup grid embeds
  pre-fix artifacts).

## 9. Reproduce

- Autopsy: `python docs/dev/transition/pdv-trigger/data/make_kappa_freeze_autopsy.py`
  (reads the committed `data/summary.csv`; writes `data/kappa_freeze_autopsy.csv`).
- Live repro: simple_cluster + `cooling_boost_kappa 8`, `stop_t 2` (§4; params under the session
  scratchpad, mechanism-only — not calibration data).
- Literature anchors (searched 2026-07-02): McKee & Cowie 1977; Cowie & McKee 1977;
  El-Badry+2019 (arXiv:1902.09547); Vieser & Hensler 2007 (A&A 472, 141); Lancaster+2021a/b
  (arXiv:2104.07691/2104.07722), +2024 (arXiv:2405.02396); Silich+2003/2004,
  Tenorio-Tagle+2005/2007 (catastrophic cooling); Mac Low & McCray 1988.
