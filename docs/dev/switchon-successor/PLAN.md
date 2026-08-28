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

**Status (2026-08-14):** ✅ **CONCLUDED — outcome S0: the constant stays** — ⚠️ **but every batch
below was measured in a regime `main` no longer has.** All five batches done. No `trinity/`
behaviour changed by *this* workstream; the only source edit is the rationale block it produced, at
the constant in `trinity/bubble_structure/get_bubbleParams.py`, plus a correction to the stale
mechanism claim in `test/test_dt_switchon_ramp.py`'s docstring.

⚠️ **Pre-C3c provenance (added 2026-08-14).** Batches 1-5 ran against
`P_drive = max(press_bubble, P_HII)` where `P_HII` was `params['Pb']` relabelled — the **un-ramped**
pressure, frozen per segment (`docs/dev/phii-identity/PLAN.md` §3 item 3, "D-ramp"). The `max`
therefore selected the un-ramped floor throughout the switch-on window, so **`dt_switchon` never
reached the shell momentum equation**: it acted only through `Ed` and `L_leak`. `c43a50e` (PR #738,
merged the same day as this workstream) zeroes `P_HII` in the energy phase, so the ramp now throttles
`vd` too. Split of what that costs:

- **Survives — the algebra.** D1 and D4's `PdV/Lmech = 2(v2/v_wind)/(R1/R2)^2` with `E0` absent, the
  4.85× work-partition violation, and the six-digit seed universality all live in the energy
  equation, which C3c does not touch.
- **Does not survive as measurement — the ablations.** The full-ablation fate flips (3 of 5;
  `docs/dev/phase1a-stiffness/PLAN.md` §2 D6), the N1 Weaver Eq. 20 distances (~12% with the ramp vs
  154× below without), the 3.6-6.0× seed-variant figures, and the `|ΔR2| ≤ 0.006-0.017%` cost bound
  were all taken with the ramp half-connected. **Do not quote them as current.** The S0 *conclusion*
  is not in doubt — post-C3c, ablating the ramp restores the un-ramped pressure to **both** channels,
  so the runaway it protects against can only be stronger — but that is an argument, not a re-run.
  Re-running the ablation and the N1 bar on post-C3c `main` is this workstream's one open item.

**D1:** the drain is **PdV work**, not cooling (0.1-0.8% of gain in both arms), and
the seed satisfies Weaver's *energy* exactly while violating its *work partition* by **4.85×** —
the ramp is a relaxation device for an inconsistent initial condition. **D2:** the physical-clock
candidate S1 **fails N0 on 3 of 5 configs**, and the failures are *not* ordered by how much the
window shrinks (87,055× survives, 7× dies), so **no value of `k` rescues it and the entire
"better clock" family is retired** — window length is not the controlling variable. **D3:** S2 (a sustainability cap with no free constant) is the **first candidate to clear the fate
bar on all five configs** — and it self-selects a release at ≈3× each run's own `dt_phase0`,
which retro-explains why S1's 1× behaved like no ramp. But it **fails N1 on all five**: capping at
"no net energy loss" pins `dEb/dt≈0`, so `Eb` plateaus and `Eb/t` decays by construction, landing
about twice as far from the physics reference as the shipped ramp. **S2 is out, and the limiter
family with it** — a correct criterion needs a target *growth rate*, which needs a reference
solution TRINITY does not have (Weaver is wind-only; §0.3). **D4:** the handover work rate is
algebraic — `PdV/Lmech = 2(v2/v_wind)/(R1/R2)²`, with **`E0` absent** — so "reseed the energy" was
ruled out before running, and the seed is **identical to six digits on all five configs**. Both
measured seed-velocity variants **rescue 2 of the 3 fates full ablation destroys** (so most of the
ramp's protection is *velocity*, not geometry — the pre-registered prediction was half wrong) but
still **fail N0 on `f1edge_hidens`, N1 on all five (3.6-6.0× worse) and N2 everywhere**: starting
marginal only delays the runaway, because `R1/R2 → 1` as `Eb` dips. **All four candidate families
are now measured dead** (clock, limiter, seed-energy, seed-velocity). **Outcome: S0 — keep the
constant and write D1-D4 into the source.** **Batch 5 did exactly that** — the identity, the
per-config seed numbers, the four failed successors and the missing decelerating phase are now
written at the constant itself, so the next person to call it "uncalibrated" finds out why it is
still there before deleting it.

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
| `P_HII / Pb` | **1.0000** exactly *as measured 2026-08-06* — then the known `n_IF_Str` min-cap made `P_HII` an algebraic relabelling of the confining pressure. ⚠️ **SUPERSEDED 2026-08-14 by the C3c regime switch** (`get_bubbleParams.get_phii_c3c`, merged in `c43a50e`): the confined branch now returns **exactly 0.0**, and a fresh `simple_cluster` run on merged `main` measures `P_HII/Pb = 0.0000` through phase 1a. The *conclusion* is unchanged and in fact now explicit — the ionised term contributes nothing independent in the energy phase — but see the box below, because the change moves what the ramp controls. |

> ### ⚠️ 2026-08-14, post-merge: C3c changed what `dt_switchon` controls
>
> Every measured number in this workstream (D1–D4, the screen references, the fate tables) was
> taken **before** the C3c photoionised regime switch landed in `main` (`c43a50e`). That change is
> not cosmetic for phase 1a, and it cuts in the direction that makes the ramp *more* load-bearing,
> not less:
>
> * Phase 1a drives the shell with `P_drive = max(press_bubble, P_HII)`
>   (`energy_phase_ODEs.py:256`), where `press_bubble` is the **ramped** pressure.
> * **Before C3c**, `P_HII` equalled the unramped `Pb` exactly, so `P_drive` came out as the
>   *unramped* pressure — **the ramp did not affect the shell drive at all**. It acted only on the
>   energy equation's `PdV` drain (`:274`, which uses `press_bubble` directly).
> * **After C3c**, `P_HII = 0` in the energy phase, so `P_drive = press_bubble` — **the ramp now
>   governs the drive as well as the drain.**
>
> **What survives untouched:** everything algebraic. `PdV/Lmech = 2(v2/v_wind)/(R1/R2)²` is derived
> from the energy equation and the `solve_R1` balance, neither of which involves `P_HII`; the seed
> anatomy was re-run on merged `main` on 2026-08-14 and reproduces `R1/R2 = 0.869167` and
> `PdV/Lmech = 2.647425` on all five configs, unchanged to six digits. So D4's core result, and the
> conclusion that no seed *energy* can fix the handover, stand.
>
> **What is now dated and must be re-measured before being quoted:** the *trajectory and fate*
> results — D1's budget table, D2/D3/D4's N0/N1/N2 numbers, and
> `phase1a-stiffness/data/dt_switchon_removability.csv`. They remain the best evidence on record
> and their qualitative verdicts are expected to hold a fortiori (the ramp gained influence), but
> the specific percentages were measured under a different phase-1a drive.
>
> **What this does to S0:** it strengthens it. The constant is kept, and it is now doing strictly
> more work than the write-up at the constant claims.

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

### D2 — Batch 2 result: S1 is dead, and so is the whole "better clock" family
### (2026-08-06; `data/s1_physical_clock.csv`, harness `harness/s1_physical_clock.py`)

S1 sets the ramp window to `k·dt_phase0` (k = 1) — the free-expansion time the code already
computes — instead of the fixed 1e-3 Myr. All five configs, `stop_t = 0.02`, against the
ramp-active arms.

| config | `dt_phase0` | window shrinks by | fate |
|---|---|---|---|
| `simple_cluster` | 0.339 yr | 2951× | **`energy_collapsed`** — FLIP |
| `f1edge_lowdens` | 138.3 yr | **7×** | **`energy_collapsed`** — FLIP |
| `f1edge_hidens` | 0.196 yr | 5112× | **`energy_collapsed`** — FLIP |
| `gmc_control` | 1.956 yr | 511× | `stopping_time` — survives |
| compact probe | 0.0115 yr | **87055×** | `stopping_time` — survives |

**N0 fails on 3 of 5 ⇒ S1 is out.** SWEEP2 §5's argument is now a measurement, which was this
batch's stated purpose whichever way it went.

**The stronger result is *why*, and it retires more than one candidate.** The failures are **not
ordered by how much the window shrinks**: the config that shrinks the *most* (compact probe,
87,055×) survives, and the one that shrinks the *least* (`f1edge_lowdens`, 7×) dies. Survivors
span 511-87,055× and failures span 7-5112× — the ranges **overlap**, so no threshold in window
length separates them, and therefore **no single value of `k` rescues S1.** Window length is not
the controlling variable; the config's own early state is. That kills the entire
"replace the clock with a better clock" family, not merely `k = 1`.

Corroborating detail: the three configs S1 kills are **exactly** the three that die under full
ablation (`phase1a-stiffness/data/dt_switchon_removability.csv`). S1 does not behave like a
shorter ramp — it behaves like *no* ramp, because by the time these configs need the suppression
the physical clock has long since expired.

**Consistent with D1, and it sharpens the remaining candidates.** D1 found the seed violates the
Weaver work partition by 4.85× and that geometry alone cannot fix it. If the problem were the
clock, a shorter window would hurt in proportion; it does not. What decides survival is whether a
config can tolerate the full `Pb` at the moment the ramp lets go.

- **S2 survives, and is now the pragmatic front-runner:** a state-based release (hand over when
  the solution can take it — e.g. `PdV/L_w` reaching its partition) releases *when the state is
  ready* rather than when a clock says so, which is precisely the failure mode measured here.
- **S4 remains the root-cause candidate** (a seed that does not need rescuing at all).
- **S0 remains registered.** If S2 and S4 both fail their bars, the honest outcome is to keep the
  constant with the D1/D2 evidence written into the source — which would now be a far better
  justification than the "no derivation" the constant carries today.

### D3 — Batch 3 result: S2 clears the fates and fails the physics
### (2026-08-06; `data/s2_state_trigger.csv`, harness `harness/s2_state_trigger.py`)

S2 caps the pressure at what the wind can sustain and latches open permanently once the cap stops
binding — `Pb_sustain = (Lmech − L_cool − L_leak) / (4πR2²·v2)`, no free constant.

**N0: PASS on all five — the first candidate to clear the fate bar.** Every config keeps
`stopping_time`, including the three that die under both full ablation and S1.

**An emergent number worth recording.** The latch releases at, in units of each run's own
`dt_phase0`: **3.14×, 3.80×, 3.14×, 2.85×, 2.85×** — across configs whose `dt_phase0` spans four
decades (1.15e-8 to 1.38e-4 Myr). The criterion was never told about `dt_phase0`; it self-selects
a release at ≈3× it. That retro-explains D2 precisely: S1 forced release at 1×, roughly three
times too early, which is why it behaved like no ramp at all.

**N1: FAILS on all five, and the reason is structural, not tuning.** Mean `|1 − Weaver ratio|`
over the early snapshots: HEAD **0.084-0.123**, S2 **0.212-0.228** — about twice as far from the
reference. The signed values say why (`simple_cluster`): HEAD runs 1.000 → 0.885, S2 runs
1.000 → **0.658**, i.e. S2 falls further *below*. That is inherent to the criterion as posed:
capping at `Pb_sustain` sets `PdV = net gain` exactly, so `dEb/dt ≈ 0` while the cap binds — `Eb`
**plateaus** while `t` keeps running, and `Eb/t` decays as 1/t by construction. The shipped ramp
suppresses less and lets `Eb` keep growing.

**N2: passes at end of run everywhere; fails as literally written on 2 of 5.** `|ΔR2|` vs HEAD:

| config | 1e-5 | 1e-4 | 1e-3 | 3e-3 | 1e-2 | **2e-2 (end)** |
|---|---|---|---|---|---|---|
| `simple_cluster` | −3.520 | −1.912 | −0.669 | −0.281 | −0.085 | **−0.041** |
| `f1edge_hidens` | −2.844 | −1.591 | −0.540 | −0.220 | −0.062 | **−0.026** |
| `f1edge_lowdens` | — | — | **−2.780** | **−2.649** | −0.947 | **−0.464** |
| `gmc_control` | −2.242 | **−4.281** | −1.750 | −0.744 | −0.215 | **−0.115** |
| compact probe | −1.408 | −0.416 | −0.107 | −0.041 | −0.012 | **−0.006** |

Every deviation is negative and **monotonically converging**; all five are inside 0.5% at
end-of-run. The breaches are at intermediate times, inside the window S2 deliberately changes.
**This is the same shape of question `phase1a-init` §4 faced** — a bar written "at every compared
time" against a change whose whole purpose is to alter the early window — and it was re-sited
there by maintainer sign-off, with both versions left on the page. **Not re-sited here, and not
self-approved:** N2 stands as written, so it is recorded as failing on `f1edge_lowdens` and
`gmc_control`.

**Verdict: S2 does not land.** N1 fails outright on every config, and that is not a bar-siting
technicality — the candidate is measurably further from the physics reference than the constant it
would replace. Under §4's rule (all of N0-N4, *and* simpler to explain), S2 is out.

**What it teaches, which is the useful part.** The criterion needs a *target growth rate*, not a
*non-negativity floor*: "do not lose energy" pins `Eb` flat, whereas the solution should be
gaining. But specifying the right growth requires a reference solution TRINITY does not have —
Weaver's is wind-only, and §0.3 measured radiation at 32-60% of the drive, so importing 5/11 as a
target would be exactly the wind-only borrowing that section rules out. That is a real dead end
for the whole *limiter* family, and it points the remaining work at:

- **S4 (consistent seed)** — the root cause. If the seed did not violate the work partition by
  4.85× (D1), nothing would need rescuing and no criterion would be needed at all.
- **S0 (keep and justify)** — now with D1-D3 as the justification, which is far stronger than the
  nothing the constant carries today.

### Batch 4 pre-registration — S4's algebra, and the two variants it leaves
### (written and committed 2026-08-06 *before* any S4 run; `data/s4_seed_anatomy.csv`)

**The handover work rate is algebraic, and `Eb` is not in it.** `solve_R1` places `R1` where the
free wind's ram pressure balances the bubble pressure, i.e. `Pb = Lmech/(2π v_wind R1²)`. Substitute
that into the phase-1a energy equation's work term and everything else cancels:

> **`PdV / Lmech = 4π R2² Pb v2 / Lmech = 2 (v2/v_wind) / (R1/R2)²`**

`Eb` appears nowhere. It re-enters only through the balance root itself — a bigger `Eb` pushes
`R1/R2` *down*, which makes `PdV/Lmech` *worse*. Since `R1/R2 ≤ 1` by construction, that gives a
floor no seed energy can get under:

> **`PdV / Lmech ≥ 2 (v2 / v_wind)`, for any `E0` whatsoever.**

**Verified, not asserted.** Against Batch 1's committed `data/drain_budget.csv` at the seed
snapshot, the identity reproduces the measured `PdV/Lmech = 2.647425` to all six recorded digits
(`2/x²` with the run's own `x = 0.869167`), and the ramped arm reproduces `0.909091 = 10/11` to the
same precision. Evaluated fresh across all five configs with no simulation
(`harness/s4_seed_anatomy.py`), the seed is **the same to six digits everywhere**:

| | all five configs |
|---|---|
| `R1/R2` at the seed | **0.869167** |
| `v0 / v_wind` | **1.000000** |
| `PdV/Lmech`, unramped | **2.647425** |
| `PdV/Lmech`, with the ramp (`R1 → 0`) | **0.909091** = 10/11 |
| floor `2 v0/v_wind` — the best any `E0` can do | **2.000000** |
| `v2/v_wind` that would give `PdV = Lmech` | **0.377726** |

Four decades of density and mass, one number. **The seed's inconsistency is a property of the
seeding scheme, not of any config** — which is why every config needs the ramp and why they all
need the same one.

**This kills the obvious reading of S4 before it is written.** "Reseed the energy consistently"
cannot work: `E0` is absent from the bound, and moving it in the helpful-looking direction makes
things worse. The only lever at the handover is **`v2/v_wind`**, which the seed fixes at exactly
**1** because `v0` is the free-streaming wind terminal speed — and `PdV ≥ 2 Lmech` follows.

**Two variants, both ramp-OFF, both changing only the returned `v0`.** `r0`, `E0`, `T0` and
`dt_phase0` are untouched, so **phase 0's own published behaviour is unchanged** (that is how the
Batch-4 exit clause is read here: `dt_phase0`, whose `M_swept/M_ejected = 1.000` derivation is
verified in `phase1a-init/FINDINGS.md` Q1, must come out bit-identical; only the handover state
moves). Keeping `r0 = v_wind·dt_phase0` is deliberate — that radius *is* what `dt_phase0`'s
derivation assumes the wind front reached.

| variant | `v0` | `PdV/Lmech` at the seed | N3 | pre-registered prediction |
|---|---|---|---|---|
| **S4a** similarity velocity | `(3/5)·v_wind` | **1.588** | dimensionless, but 3/5 is Weaver's *wind-only* exponent (§0.3) | **N0 FAILS** — still above 1, so `Eb` still drains; expect the same 3 of 5 that die under ablation |
| **S4b** sustainable velocity | `(x²/2)·v_wind = 0.3777·v_wind` | **1.000** by construction | dimensionless **and** derived from TRINITY's own identity, `x` read from the run's own `solve_R1` | **uncertain, and that is the point** |

**S4a is run even though it is predicted to fail**, for the same reason D2 ran S1: it tests the
identity itself. If a run with `PdV/Lmech = 1.588` survives, the frame above is wrong and
everything built on it has to come down.

**Why S4b's prediction is honestly uncertain.** It starts the handover exactly marginal. Two
effects then compete and the sign is not obvious on paper: as `Eb` dips, `x` rises, which *lowers*
`PdV/Lmech`; but the same balance pressure is pushing on a shell that is now moving at 0.38 `v_wind`,
so `v2` will accelerate back up, which *raises* it. If `v2` relaxes to `v_wind` within a few steps,
S4b buys nothing and the whole S4 family is dead. **S4b is also the most favourable seed velocity
that is still derived rather than chosen** — anything slower is a tuned number and fails N3 — so a
failure here closes S4, not just this variant.

**A distinction from S2 worth stating in advance.** S2 also used a sustainability criterion, and
failed N1 because it held `dEb/dt ≈ 0` *continuously*, flattening `Eb`. S4b applies the same
equality **once, at the seed**, then lets the physics run free. So S2's specific N1 failure mode
does not automatically carry over — N1 has to be measured, not inferred.

### D4 — Batch 4 result: S4 rescues most of the fates, and wrecks the physics doing it
### (2026-08-06; `data/s4_consistent_seed.csv`, harnesses `harness/s4_consistent_seed.py`, `harness/s4_compare.py`)

Both pre-registered variants ran on all five configs, ramp OFF, separate processes, matched `t`.
N1 is scored over snapshots 1-5 — the window D3 used — and the reference column reproduces D3's
recorded values **exactly** on all five, so D3 and D4 sit on identical footing.

| config | N0 `sustain` | N0 `similarity` | N1 HEAD | N1 `sustain` | N1 `similarity` | N2 worst `sustain` |
|---|---|---|---|---|---|---|
| `simple_cluster` | ✅ `stopping_time` | ✅ | 0.0992 | **0.4230** | **0.5510** | 4.752% |
| `f1edge_hidens` | ❌ **`energy_collapsed`** | ❌ **`energy_collapsed`** | 0.1234 | **0.5212** | **0.6368** | 8.160% |
| `f1edge_lowdens` | ✅ | ✅ | 0.0938 | **0.3956** | **0.5277** | 6.741% |
| `gmc_control` | ✅ | ✅ | 0.0864 | **0.3696** | **0.5058** | 7.706% |
| compact probe | ✅ | ✅ | 0.0839 | **0.3607** | **0.4983** | 1.448% |

**N0 fails** (one flip is fatal), **N1 fails 5/5** at 3.6-6.0× the shipped ramp's distance from the
reference, and **N2 fails everywhere** (1.4-8.2% vs the 0.5% bar). **S4 does not land.**

**The pre-registered prediction for S4a was half wrong, and the half that was wrong is the
interesting half.** It said S4a would kill "the same 3 of 5 that die under ablation". Only **one**
dies. Both variants **rescue `simple_cluster` and `f1edge_lowdens`** — two of the three fates that
full ablation destroys (`phase1a-stiffness` D6). So **most of the ramp's fate protection comes from
the handover velocity, not from the geometry it actually manipulates.** That is a genuinely new
fact about what the constant is doing, and it was not visible from any previous batch.

**The identity ordered the variants correctly before either ran.** `sustain` (`PdV/Lmech = 1.000`)
beats `similarity` (1.588) on N1 for **every single config**, by 0.11-0.14. The algebra of §3's
Batch-4 pre-registration predicted the ranking of two candidates it had never seen executed.

**Why it still fails, in one line: the runaway is delayed, not stopped.** Starting at
`PdV/Lmech = 1.000` only holds for an instant — `Eb` dips, the balance root `R1/R2` climbs toward 1,
and `2(v2/v_wind)/(R1/R2)²` rises straight back above 1. The signed Weaver ratio shows it plainly on
`simple_cluster`: HEAD sits flat near 0.885 while `sustain` slides 0.876 → **0.306** and
`similarity` 0.781 → **0.187**, monotonically *diverging* (unlike S2's N2, which converged). The
pre-registration named this exact failure mode as the uncertainty; the measurement resolved it
against the candidate. `data/s4_identity_check.csv` records the same race in the fully ablated run:
`v2/v_wind` falls 1.000 → 0.626 over six snapshots while `R1/R2` climbs 0.869 → 0.999, so
`PdV/Lmech` only eases 2.65 → 1.26 and never reaches 1. **The shell decelerates and the bubble
empties, and the bubble wins.**

**All four candidate families are now measured dead** — clock (D2), limiter (D3), seed-energy (D4,
analytically, before running), seed-velocity (D4, measured). **S0 is the outcome.**

**What the workstream found that S0 should carry.** The ramp is not a model of the termination
shock forming. It is a crutch for handing over to the energy-driven description at the one moment
that description is marginal: at `t = dt_phase0` the shell is still moving at the wind speed by
construction, and `PdV/Lmech = 2(v2/v_wind)/(R1/R2)² ≥ 2` follows with no freedom left in it. The
honest fix is not a different seed but **a decelerating phase between free expansion and the
energy-driven solution — a phase TRINITY does not have.** That is a real physics gap, well outside
a magic-number audit, and it is recorded here as a pointer rather than started.

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
| **2** | ✅ **DONE 2026-08-06 — S1, the physical clock** (`tmin = k·dt_phase0`, k=1, all five configs) | `harness/s1_physical_clock.py` → `data/s1_physical_clock.csv` | **D2: N0 FAILS 3/5 ⇒ S1 out — and the failures are not ordered by window shortening (87,055× survives, 7× dies), so no `k` rescues it and the whole "better clock" family is retired** | ran 25 min |
| **3** | ✅ **DONE 2026-08-06 — S2, the state-based trigger** (sustainability cap + one-way latch, no free constant) | `harness/s2_state_trigger.py` → `data/s2_state_trigger.csv` | **D3: N0 PASSES 5/5 (a first) but N1 FAILS 5/5** — the cap pins `dEb/dt≈0` so `Eb` plateaus and `Eb/t` decays by construction; N2 passes at end-of-run everywhere, fails as written at intermediate times on 2/5. **S2 out**; the limiter family with it | ran 35 min |
| **4** | ✅ **DONE 2026-08-06 — S4, the consistent IC** (two ramp-off seed-velocity variants; `r0`/`E0`/`T0`/`dt_phase0` untouched, so phase 0's published behaviour is unchanged) | `harness/s4_seed_anatomy.py`, `harness/s4_consistent_seed.py`, `harness/s4_compare.py` → `data/s4_seed_anatomy.csv`, `data/s4_identity_check.csv`, `data/s4_consistent_seed.csv` | **D4: `E0` is absent from the handover work rate (so seed-energy is dead analytically), and both velocity variants FAIL N0 on `f1edge_hidens`, N1 5/5 and N2 everywhere — though they do rescue 2 of the 3 fates full ablation destroys.** S4 out; all four families dead ⇒ **S0** | ran 40 min |
| **5** | ✅ **DONE 2026-08-06 — Write S0** (the only remaining outcome — no candidate cleared §4) | the D1-D4 rationale written at the constant in `trinity/bubble_structure/get_bubbleParams.py`; the stale "bubble-structure solve stalls" claim corrected in `test/test_dt_switchon_ramp.py`'s docstring; `magic-numbers/AUDIT.md` #2, `SWITCHON_BRIEF.md` and `SWEEP2_PLAN.md` closed on the form question | N4 green; docs reconciled | ran 15 min |

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
