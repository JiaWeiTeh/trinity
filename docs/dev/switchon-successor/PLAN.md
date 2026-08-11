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

**Status (2026-08-06):** 🔵 actionable — **pre-registered, nothing implemented, no `trinity/` line
touched.** Asks whether the *fixed 1e-3 Myr clock* in `dt_switchon` can be replaced by a
scale-free, physically-derived criterion. **This is not a re-run of "can the ramp be deleted" —
that was measured and answered NO** (`docs/dev/phase1a-stiffness/PLAN.md` §2 D6). Removal and
replacement are different questions; §1 says why the distinction matters. "Keep the constant and
document it better" is a pre-registered outcome (§5).

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
   only preventing a stall — it is keeping the early solution on the analytic attractor its own
   initial conditions assume. That gives a successor something to be *right* against, rather than
   merely equivalent to.

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

## 4. PRE-REGISTERED BARS (registered before any measurement or edit)

- **N0 — fate preservation (checked first, decisive).** All five screen configs keep the stopping
  fate they have on current `HEAD`. One flip kills the candidate. This is the bar the ramp itself
  fails when deleted, so it is the minimum a successor must clear.
- **N1 — the physics bar (new, and the point of this workstream).** Over the early window, the
  candidate's `|Eb/t − (5/11)L_w| / ((5/11)L_w)` must be **no worse than the current ramp's**
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
| **1** | **Diagnose the drain** — decompose `dEb/dt` into wind gain, cooling loss and `PdV` work per segment, ramp-on vs ramp-off on `simple_cluster`; track `R1/R2` and `(R1/R2)³` in both arms against the Weaver self-similar value | `data/drain_budget.csv` + one paragraph | **names what removes the energy without the ramp** (cooling? PdV? both?) — this decides which of S1-S4 is even addressing the cause | ~30 min |
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
