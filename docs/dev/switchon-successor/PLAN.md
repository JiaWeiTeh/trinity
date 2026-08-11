# A physically-motivated successor to `dt_switchon` — pre-registered plan

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

**Status (2026-08-06):** 🔵 actionable — **Batch 1 done; nothing implemented, no `trinity/` line
touched.** D1 (§3, end): the drain is **PdV work**, not cooling (0.1-0.8% of gain in both arms), and
the seeded state satisfies Weaver's *energy* exactly while violating its *work partition* by
**4.85×** — so the ramp is a relaxation device for an inconsistent initial condition, which
promotes **S4** (consistent seed) over the clock-shaped candidates. Next: **Batch 2** (run S1
anyway, cheaply, to convert an expectation into a measurement).

The workstream asks whether the *fixed 1e-3 Myr clock* in `dt_switchon` can be replaced by a
scale-free, physically-derived criterion. **This is not a re-run of "can the ramp be deleted" —
that was measured and answered NO** (`docs/dev/phase1a-stiffness/PLAN.md` §2 D6). Removal and
replacement are different questions; §1 says why the distinction matters. "Keep the constant and
document it better" is a pre-registered outcome (§4).

## 0. Why this is being opened when #2 was closed

Magic-number audit #2 was closed as *document-and-pin* on the strength of two results: the ramp is
load-bearing (deleting it is fatal), and a scale-relative switch-off "is no longer the recommended
successor shape" (`docs/dev/magic-numbers/SWEEP2_PLAN.md` §5). The first is measured, repeatedly,
and stands. **The second was an argument, not a measurement** — the reasoning was that a
scale-relative switch-off delivers full pressure *earlier* at the stiff edge, which is the
direction the collapse lives in. Plausible, and never run.

Two things changed that make the argument worth testing:

1. **The constant matters far more widely than the record assumed.** Ablation flips the stopping
   fate on **3 of 5** configs, `simple_cluster` — the default published config — among them
   (`phase1a-stiffness/data/dt_switchon_removability.csv`). An uncalibrated constant is deciding
   whether the energy-driven phase survives on the flagship configuration. That raises the value
   of getting its *form* right, even though its presence is settled.
2. **There is now a physics yardstick, not just an equivalence bar.** Weaver+77 Eq. 20 —
   `Eb = (5/11)·L_w·t`, the same relation TRINITY uses to seed `E0`
   (`phase0_init/get_InitPhaseParam.py`) — predicts `Eb/t = (5/11)L_w`. Measured on
   `simple_cluster` over the first six segments: **ramp on, `Eb/t` holds within ~12% of the
   analytic value (2.77e8 → 2.45e8); ramp off, it falls 154× below (→1.8e6)**. So the ramp is not
   only preventing a stall — it is keeping the early solution near the analytic attractor its own
   initial conditions are seeded from. **§0.3 bounds how far that argument can be pushed:** Weaver
   is wind-only, and radiation supplies a third to three-fifths of the drive here — so the
   reference bounds a successor's plausibility rather than defining its target. Even so it gives
   a candidate something to be *wrong* against, which pure equivalence testing cannot.

### 0.3 Caveat: Weaver is **wind-only**, and TRINITY is not (maintainer, 2026-08-06)

Weaver+77 solves a bubble driven by winds alone. TRINITY additionally carries **radiation
pressure, gravity, ionized-gas pressure and cooling**, so its solution is not Weaver's and has no
obligation to reproduce Eq. 20 or the 6/11 partition. Measured on `simple_cluster` in the early
window (ramp on, from the Batch 4 run):

| quantity | measured over the first six snapshots |
|---|---|
| `F_rad / (4πR2²·Pb)` | **0.39, 0.37, 0.35, 0.34, 0.32, 0.60** — radiation is a third to three-fifths of the drive |
| `F_grav / (4πR2²·Pb)` | 0.006-0.009 — negligible |
| `P_HII / Pb` | **1.0000** exactly — the known `n_IF_Str` min-cap (`phase1a-init` FINDINGS, Extra findings #1), so the ionized-gas term adds nothing independent here |

Four consequences, and they change how the rest of this plan must be read:

1. **Weaver is a limiting-case reference, not ground truth.** Departures of tens of percent are
   expected and *physical*. Any claim of the form "the solution should equal Weaver" is wrong.
2. **N1 is therefore comparative by design** — "no worse than the shipped ramp" — and must never
   be tightened into "must match Weaver". Its wording in §4 is unchanged; this is how to read it.
3. **The direction is still informative.** Extra radiative push means more expansion work and so
   *less* retained `Eb`: a deficit against wind-only Weaver is exactly what one expects, and the
   ramp-on arm shows 0.88-0.94. Likewise D1's `PdV/L_w → 0.563` against the wind-only 6/11 = 0.545
   should be read as "slightly above the wind-only partition, in the direction radiation pushes
   it" — not as proof of correctness.
4. **The ablated arm's 154× deficit is not attributable to the extra physics.** No plausible
   radiation/gravity/cooling contribution moves `Eb/t` by two orders of magnitude, so that
   collapse remains a genuine failure rather than a modelling difference.

A stricter reference — the wind + radiation similarity solution — would be a better yardstick and
is *not* attempted here; it would be its own derivation, and N1 does not need it.

## 1. Removal vs replacement — keep these separate

| question | status |
|---|---|
| Can the ramp be **deleted**? | **Answered NO**, measured across five configs, twice (`phase1a-stiffness` D6; `magic-numbers` SWEEP2 §4). Do not re-run. |
| Can the fixed **1e-3 Myr clock** be replaced by a scale-free physical criterion that preserves the protection? | **Open — this workstream.** |
| Is the ramp compensating for an inconsistent initial state rather than modelling a real effect? | **Open — §3 S4, and the most interesting possibility.** |

## 2. What the constant is, physically (verified 2026-08-06)

`get_effective_bubble_pressure` feeds a linearly-ramped `R1` into
`Pb = (γ−1)Eb / [(4π/3)(R2³ − R1³)]` for `t ≤ tSF + 1e-3` Myr. `R1` is the wind termination shock;
suppressing it enlarges the shocked-wind volume and lowers early `Pb`.

The *idea* has a real referent: in the Weaver four-zone structure `R1` does not exist at `t = 0`.
The wind expands freely until the swept-up mass equals the ejected mass; only then does the
termination shock form (the standard free-expansion → energy-driven transition, Koo & McKee 1992).
**TRINITY already computes exactly that moment** as `dt_phase0` (`phase0_init/`, verified in
`docs/dev/phase1a-init/FINDINGS.md` Q1: `M_swept/M_ejected = 1.000`).

The *implementation* is not that physics:

- `dt_phase0` spans **0.0115 yr** (compact probe) to **1.96 yr** (GMC control). The ramp runs for
  **1000 yr** — **500× to ~87,000× longer** than the establishment time it superficially models.
- The linear-from-zero shape has no derivation.
- No literature was found for a fixed-duration ramp on the inner shock radius; WARPFIELD's papers
  describe `R1` as evolving with the bubble's thermal pressure. Provenance beyond "inherited" is
  unestablished — settling it means reading WARPFIELD's own source.

## 3. Candidates (to be measured, not chosen up front)

- **S1 — physical clock.** `tmin = k·dt_phase0` (k ≈ 1), i.e. the ramp closes when the termination
  shock physically establishes. Scale-free by construction. *Expected to be hard:* it closes the
  ramp 500-87,000× sooner, delivering full `Pb` early — the direction the collapse lives in. Its
  value is that it converts SWEEP2 §5's argument into a measurement, cheaply.
- **S2 — state-based geometric trigger.** Ramp until the solution reaches its self-similar
  structure, judged by the dimensionless `R1/R2` (the ramp's leverage is `(R1/R2)³`), then hand
  over. No clock at all; scale-free; ends exactly when the physical structure exists.
- **S3 — analytic early segment.** Integrate the first window on the Weaver self-similar solution
  itself and hand the numerical integrator a state already on the attractor. Standard practice for
  a singular early transient, and it makes the `Eb/t` bar true by construction — which is also its
  risk (it can hide a genuine inconsistency rather than fix it).
- **S4 — fix the initial condition instead (root-cause candidate).** Hypothesis: the
  pressure-balance `R1` is inconsistent with the seeded `E0`/`r0`, and the ramp compensates. If so
  the honest fix is a self-consistent seed (`E0`, `r0`, `R1` from one solution) and **no ramp at
  all**. Highest value, highest blast radius — it touches phase 0, which everything depends on.
- **S0 — keep the ramp, document it better.** Registered as an acceptable outcome. If no candidate
  clears §4, this workstream still delivers: the Weaver-tracking evidence in §0.2 is a much
  stronger justification for the constant than anything on the page today.

### D1 — Batch 1 result: the drain is **PdV work**, and the seed is off the Weaver partition
### (2026-08-06; `data/drain_budget.csv`, harness `harness/drain_budget.py`)

Decomposed `dEb/dt = (Lmech − L_cool) − 4πR2²·Pb·v2 − L_leak` per snapshot on `simple_cluster`,
ramp-on vs ramp-off, from runs already held — **no new simulations**. Weaver+77 partitions the wind
input as 5/11 retained as thermal `Eb`, so the work term should tend to `PdV/L_w → 6/11 = 0.545`.

| | `t` (Myr) | 3.39e-7 (seed) | 3.73e-7 | 4.51e-7 | 5.46e-7 | 6.00e-7 |
|---|---|---|---|---|---|---|
| **ramp on** | `PdV/L_w` | 0.909 | 0.782 | 0.639 | 0.571 | **0.563** |
| | ÷ (6/11) | 1.67 | 1.43 | 1.17 | 1.05 | **1.03** |
| | `Eb/t` ÷ Weaver | 1.000 | 0.940 | 0.889 | 0.885 | 0.890 |
| **ramp off** | `PdV/L_w` | 2.647 | 2.186 | 1.592 | 1.255 | — |
| | ÷ (6/11) | 4.85 | 4.01 | 2.92 | 2.30 | — |
| | `Eb/t` ÷ Weaver | 1.000 | 0.629 | 0.195 | **0.006** | — |
| | `(R1/R2)³` | 0.657 | 0.761 | 0.914 | **0.997** | — |

**Cooling is not the mechanism.** `L_cool/L_w` is **0.1-0.8%** in *both* arms. The candidate risk
this batch existed to catch — that geometric successors would be tuned to mask a cooling problem —
**is cleared**: the drain is PdV work, the term the ramp acts on, so S1/S2 address the right
physics.

**With the ramp, the solution relaxes onto the Weaver partition** (`PdV/L_w` 0.909 → 0.563, i.e.
1.67× → 1.03× of 6/11) and `Eb/t` holds within ~12% of the analytic value. **Without it, the excess
work compounds**: `Eb/t` collapses to 0.006 of Weaver while `(R1/R2)³ → 0.997`, i.e. `R1 → R2` — the
shocked-wind volume is crushed toward zero, which is the degenerate case `bubble_E2P` has an
explicit floor for.

**The finding that matters for the design: the seeded state is itself inconsistent.** At the seed
both arms are identical, `Eb/t` is exactly 1.000 × Weaver **by construction** (`E0` is seeded from
Eq. 20) — yet `PdV/L_w = 2.647`, **4.85× the partition that same solution implies**. So phase 0
hands phase 1a a state that satisfies Weaver's *energy* and violates Weaver's *work rate*.

Arithmetic that narrows the cause: `PdV ∝ 1/(R2³ − R1³)`, so even **complete** suppression
(`R1 = 0`, the strongest the ramp can ever be) only reaches `PdV/L_w = 0.909 = 1.67 × 6/11`.
**Geometry alone cannot make the seed consistent** — a residual ~1.7× remains, and it is the right
size to be `v0`: the seed hands over the free-streaming wind speed, measured at **1.89× the Weaver
shell velocity** at `t0` (`docs/dev/phase1a-init/FINDINGS.md` Q2), and `PdV ∝ v2`.

**Consequences for the candidates** (§3), which Batch 2 onward must respect:

- **S4 is promoted to the leading candidate.** The ramp is a relaxation device for an inconsistent
  seed, not a model of the termination shock forming. The root-cause fix is a seed whose `E0`,
  `r0`, `R1` *and* `v0` come from one solution.
- **S1 is now expected to fail for a reason we can state**, not just suspect: the ramp is still
  ~99.94% suppressing at the point where the solution reaches the partition (t ≈ 6e-7 Myr, ramp
  fraction 6e-4). Closing it at `k·dt_phase0` releases `R1` abruptly while the state still needs
  the suppression, and `PdV/L_w` jumps straight back toward 2.6. Batch 2 should still run it —
  cheaply — because that converts the expectation into a measurement.
- **S2 gains a concrete trigger variable:** hand over when `PdV/L_w` reaches 6/11, which is
  dimensionless, computed from state the RHS already has, and satisfies N3 by construction.

**Method note.** The first run of this decomposition was **wrong** and said so loudly: it predicted
`dEb/dt = −1.0e9` where the run's own snapshots show **+9.4e7**. Cause: the snapshot's `Pb` is the
**unramped** pressure from the bubble-structure solve, ~3× the value the energy equation actually
used inside the ramp window — diagnostics do not show the pressure that drove the trajectory, the
same masking class `phase1a-init` found for the old `vd` override. The harness now reconstructs the
ramped `Pb` from the shell-volume ratio, and the budget reproduces the observed `dEb/dt` to ~4%.
Validating a decomposition against the trajectory it claims to explain is what caught it.

## 4. PRE-REGISTERED BARS (registered before any measurement or edit)

- **N0 — fate preservation (checked first, decisive).** All five screen configs keep the stopping
  fate they have on current `HEAD`. One flip kills the candidate. This is the bar the ramp itself
  fails when deleted, so it is the minimum a successor must clear.
- **N1 — the physics bar (new, and the point of this workstream; read it with §0.3 — Weaver is a
  wind-only *reference*, not ground truth, because TRINITY also has radiation, gravity, P_HII and
  cooling).** Over the early window, the candidate's `|Eb/t − (5/11)L_w| / ((5/11)L_w)` must be **no worse than the current ramp's**
  (measured: within ~12% on `simple_cluster`). A successor that preserves fates while drifting
  further off the analytic attractor is not an improvement — it is a differently-shaped fudge.
- **N2 — trajectory.** `|ΔR2| ≤ 0.5%` at every matched grid time and at end of run vs current
  `HEAD`, all five configs, separate processes (`docs/dev/screen/screen.py`).
- **N3 — no new magic numbers (the theme).** Any constant a candidate introduces must be either
  (a) dimensionless **and** derived from a stated physical criterion, or (b) a quantity the code
  already computes from the run's own state (`dt_phase0`, `R1/R2`, …). **A new absolute time,
  energy or length constant disqualifies the candidate outright**, however well it performs.
  Trading `1e-3 Myr` for `3e-4 Myr` is not progress.
- **N4 — suite & style.** Full `pytest` green; `pre-commit`; `mypy` no new errors vs baseline
  (see the D5 note in `phase1a-stiffness/PLAN.md` for how that clause has been read).
- **Decision rule:** a candidate lands only if N0-N4 all pass **and** it is simpler to explain than
  the constant it replaces. If none qualifies ⇒ **S0**: keep the ramp, write the physics
  justification into the source, and close #2 permanently with the Weaver evidence attached.
- **Maintainer gate:** any candidate that changes published trajectories (i.e. anything failing N2
  but passing N0/N1) is *recorded, not self-approved* — same precedent as
  `docs/dev/phase1a-init/PLAN.md` §4.

## 5. Batches

| # | name | deliverable | exit | cost |
|---|---|---|---|---|
| **0** | Pre-registration | this doc | committed before any measurement | done |
| **1** | ✅ **DONE 2026-08-06 — Diagnose the drain** | `harness/drain_budget.py` → `data/drain_budget.csv` (131 rows, no new sims) | **D1 (§3, end): the drain is PdV work, cooling is 0.1-0.8% in both arms.** Geometric candidates address the right term. And the seed violates the Weaver work partition by 4.85× while satisfying its energy exactly ⇒ **S4 promoted, S1 expected to fail for a stateable reason, S2 gains a dimensionless trigger** | ran 10 min |
| **2** | **S1, the physical clock** — `tmin = k·dt_phase0`, all five configs | `data/s1_physical_clock.csv` | N0/N1; converts SWEEP2 §5's argument into a measurement either way | ~40 min |
| **3** | **S2, the state-based trigger** (only if Batch 1 says geometry, not cooling) | `data/s2_state_trigger.csv` | N0/N1 | ~40 min |
| **4** | **S4, the consistent IC** (only if Batch 1 points at the seed) | `data/s4_consistent_ic.csv` | N0/N1, plus an explicit check that phase 0's published behaviour is unchanged | ~1 h |
| **5** | **Gate & land, or write S0** | screen ledger + failing-first test, or the S0 write-up | N2/N4; docs reconciled | ~40 min |

**Batch 1 first, and no candidate is written before it reports.** The measurements in §0.2 say the
unramped run leaves the Weaver attractor; they do not say *why*. If the energy leaves through
cooling, then S1/S2 (both geometric) are addressing the wrong term and would be tuned to mask a
cooling problem — the exact failure mode this session's audit exists to prevent.

## 6. What NOT to do

- Do not re-test deletion of the ramp (§1: answered, twice).
- Do not replace one absolute constant with another (N3).
- Do not accept a candidate on fates alone — N1 exists because "it runs" and "it is right" are
  different claims, and the current constant already satisfies the first.
- Do not fold this into `phase1a-stiffness/` or re-open `magic-numbers/SWITCHON_BRIEF.md`; both are
  resolved on their own terms and are cited here as inputs.

## 7. Inputs (all committed, none need re-running)

| what | where |
|---|---|
| the ramp is not deletable — fate flips on 3/5 | `docs/dev/phase1a-stiffness/data/dt_switchon_removability.csv` |
| the stall it protects against, instrumented | `docs/dev/magic-numbers/data/switchon_stall_probe.csv`, `switchon_stall_stacks.txt` |
| ramp-active reference trajectories, 5 configs | `docs/dev/phase1a-stiffness/data/equivalence_screen.csv` + the Batch 4 run dirs |
| `dt_phase0` derivation + verification (`M_swept/M_ejected = 1.000`) | `docs/dev/phase1a-init/FINDINGS.md` Q1 |
| the constant, with its current in-source rationale | `trinity/bubble_structure/get_bubbleParams.py`; pinned by `test/test_dt_switchon_ramp.py` |
| multi-config screen | `docs/dev/screen/screen.py` |
