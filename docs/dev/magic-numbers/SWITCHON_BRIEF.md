# `dt_switchon = 1e-3` — brief for a scale-relative successor (magic-number #2)

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

**Status (2026-08-06):** ✅ resolved as **document-and-pin** — both load-bearing measurements
reproduced on HEAD, the stall mechanism instrumented and **corrected** (it is the phase-1a RK45
*segment integrator* that stalls, not the bubble-structure solve — see §4), and the pre-registered
decision rule in `docs/dev/magic-numbers/SWEEP2_PLAN.md` §5 landed on: keep the constant, document
it in-source, pin it with `test/test_dt_switchon_ramp.py`. A successor remains possible but is
phase-1a integrator work (stiff/switching segment solver or a terminal in-segment `Eb`-floor
event), not a re-shaping of this constant — **pre-registered as
`docs/dev/phase1a-stiffness/PLAN.md`, where re-opening this constant is Batch 6.** Owned by
`AUDIT.md` finding **#2**.

## 0. What you are being asked to do

> **SUPERSEDED 2026-08-06** — this design work was run and the pre-registered decision landed on
> **document-and-pin, no successor** (see the Status line and `SWEEP2_PLAN.md` §5). The sections
> below stand as the record; §4's mechanism is corrected in place.

Design and gate a **scale-relative replacement** for an absolute-time constant in the bubble
pressure path. You are *not* being asked to delete it — that was tried and measured, and it is
fatal on the stiffest configuration. Read §4 before proposing anything.

This is the sibling of a defect that was just fixed and landed: magic-number **#4**
(`vd = -1e8`), fixed on branch `hotfix/early-approximations` by the `docs/dev/phase1a-init/`
workstream. That fix is the model for the *diagnosis* here and the counter-example for the
*remedy* — read `docs/dev/phase1a-init/FINDINGS.md` §Q3 for why absolute times are the wrong
shape, and this brief's §4 for why the same deletion does not work twice.

## 1. The constant, exactly as it stands

`trinity/bubble_structure/get_bubbleParams.py`, inside
`get_effective_bubble_pressure(...)` (def at `:311`), in the `else:` branch that serves the
**energy and implicit phases** (verified against source 2026-08-05):

```python
        dt_switchon = 1e-3           # :368
        tmin = dt_switchon

        if t is not None and tSF is not None:
            if t <= (tmin + tSF):
                R1_tmp = (t - tSF) / tmin * R1
                return bubble_E2P(Eb, R2, R1_tmp, gamma)

        return bubble_E2P(Eb, R2, R1, gamma)
```

So for the first `1e-3` Myr (1000 yr) after star formation, the **inner discontinuity radius
`R1` is linearly ramped from 0 to its computed value** before being handed to `bubble_E2P`.
Everywhere else `R1` goes in unmodified.

`bubble_E2P` (`:198`) converts bubble energy to pressure using the shell volume between `R1` and
`R2`, so suppressing `R1` **inflates the volume and lowers `Pb`** — the ramp's leverage on the
answer scales as `(R1/R2)³`.

**Call sites** (only two, both in phase 1a): `trinity/phase1_energy/energy_phase_ODEs.py:224`
(the RHS) and `:356` (`compute_derived_quantities`), both forwarding `t`/`tSF`.
**CORRECTED 2026-08-06** (re-verified @ `731ac50`): an earlier version of this paragraph said
phases 1b/1c/2 "reach `get_effective_bubble_pressure` through other branches, or pass `t=None`" —
they do not reach it at all; they compute pressure via `bubble_E2P`/`pRam` directly
(`get_betadelta.py:329`, `bubble_luminosity.py:228`, momentum's own snapshot path). The
conclusion stands — the ramp is a phase-1a-window effect — but by absence of callers, not by
branch selection. Corollary worth knowing: if phase 1a exits early via `cooling_balance`
(`run_energy_phase.py:286-297`) before `tSF + 1e-3`, the 1b handoff drops the ramp mid-window,
a Pb discontinuity at the boundary (recorded for the transition workstream, not worked here).
Verify all of this before relying on it — it is the kind of claim that rots.

## 2. Why it is on the suspect list

`TFINAL_ENERGY_PHASE = 3e-3` Myr (`trinity/phase1_energy/run_energy_phase.py:54`). The ramp
therefore shapes the driving pressure across **the first third of phase 1a**.

The smell is exactly the one that made #4 a defect: **an absolute time compared against physics
whose timescale is not absolute.** The bubble's expansion timescale is `R/Ṙ = (5/3)t`, seeded at
`t0 = dt_phase0`, which spans 0.0115 yr at sub-GMC scale to 1.96 yr at GMC scale — a factor ~170 —
because it scales as `sqrt(M*/ρ)/v_w^{3/2}`. A fixed 1000-yr window is a different fraction of
the physics at every object scale, so:

- at **GMC scale** 1e-3 Myr is a genuinely early time — the ramp does something plausible;
- at **sub-GMC scale** (relaxation complete by ~160 yr = 1.6e-4 Myr) the ramp is still suppressing
  `R1` long after the bubble has physically established itself.

There is also no physics reference and no sensitivity note anywhere for the value `1e-3`. It was
flagged independently by two audit agents (A and D).

## 3. What has already been measured — do not repeat this

Experiment **E8b**, run 2026-08-05 on top of the phase-1a fix (never against stock — see §5).
Ablated by forwarding `t=None` so the ramp branch is skipped and *nothing else* changes; harness
`docs/dev/phase1a-init/harness/e8b_runner.py`. All comparisons at matched `t`, separate processes.

| config | ablation effect on R2 | verdict |
|---|---|---|
| compact probe (`mCloud=300, sfe=0.01, nCore=8.7e3`) | −1.43% @10 yr → −0.108% @1e3 yr → **−0.0059% @2.1e4 yr** | decays to nothing |
| GMC control (`mCloud=1e6, sfe=0.01, nCore=1e3`) | −4.71% @100 yr → −1.79% @1e3 yr → −0.079% @3e4 yr → **−0.017% @8e4 yr** | decays to nothing |
| `f1edge_hidens` (`mCloud=1e7, sfe=0.01, nCore=1e6`) | **run STALLS** — 4 rows to 0.26 yr in 90 min wall, vs 127 rows to 2e4 yr in minutes with the ramp | ramp is load-bearing |

> **SCOPE CORRECTION (2026-08-06).** Result 1 below is measured on the compact probe and
> `gmc_control` only. Ablating all five screen configs
> (`docs/dev/phase1a-stiffness/data/dt_switchon_removability.csv`) shows those are **the only two
> of five that survive ablation**: `simple_cluster` — the default published config — and
> `f1edge_lowdens` both collapse to `ENERGY_COLLAPSED`, as `f1edge_hidens` does. So "bounded and
> small" describes the two recovering configs, not the ramp in general; on the majority of the
> config span the ramp is the difference between a bubble that grows and one that dies. Result 2
> stands and is strengthened.

Two results, and the second is the important one:

1. **Its trajectory consequence is bounded and small** *(on the two configs that survive ablation
   — see the correction above).* At the compact probe's observed age the ramp is worth
   0.006% in R2 — an order of magnitude below the `phase1a_segFrac` 0.1→0.03 convergence step the
   shipped schedule itself carries. Note the effect is *weaker* in the compact regime, the
   opposite scaling to `SEGMENT_DURATION`, because the compact probe's `t0` is ~170× earlier so by the time the
   1e-3 Myr window closes `(R1/R2)³` has already fallen to 4e-3. **So it is not a second
   discretisation artifact.** Whatever it is costing in accuracy, it is not costing much.
2. **It cannot simply be deleted.** At `nCore=1e6` ablation raises `Pb` (the suppressed `R1` was
   inflating the shell volume), ~~the bubble-structure ODE stiffens, and the solve stops
   converging three segments in~~ — **mechanism corrected 2026-08-06 (R3, instrumented):** the
   bubble-structure solve stays healthy (every `get_bubbleproperties_pure` call returns in
   ~1.3 s); the raised `Pb` drains `Eb` 180 → 29 au across segments 1-4 and then **phase 1a's
   segment integrator (`solve_ivp`, hard-coded `method='RK45'`, `run_energy_phase.py:309`)
   stalls in micro-steps** on the stiffened energy ODE in segment 5. Evidence:
   `docs/dev/magic-numbers/data/switchon_stall_probe.csv` + `switchon_stall_stacks.txt`.

Raw numbers: `docs/dev/phase1a-init/data/gate_results.csv`, rows tagged `E8b`. Trajectories:
`docs/dev/phase1a-init/data/e8b_{m43,gmc}_noramp.csv` and
`e8b_hidens_noramp_STALLED.csv` (that file *is* the finding — 4 rows).
Write-up: `docs/dev/phase1a-init/PLAN.md` §8.

## 4. The conclusion that shapes the work

**`vd = -1e8` papered over a discretisation error; `dt_switchon` papers over genuine stiffness.**
They look alike from the outside — both absolute early-time constants with no derivation — and
they need opposite remedies. #4 was safe to delete once the segment schedule was fixed. #2 is not
safe to delete at all, at any schedule, because what it is protecting against is ~~the bubble
structure ODE becoming unsolvable when `Pb` is high early~~ **(corrected 2026-08-06, R3):**
the early-`Pb`-driven collapse of `Eb`, which stiffens the phase-1a *segment energy ODE* until
its hard-coded RK45 integrator stalls in micro-steps — the bubble-structure solve itself stays
healthy throughout. Same conclusion, different protected component: the honest successor is
phase-1a stiffness handling, not a better clock on this ramp.

So the shape of the successor is:

> Keep the numerical protection. Make **when it switches off** scale-relative instead of a fixed
> 1e-3 Myr.

And the ordering constraint that follows:

> A **stiffness gate comes first**. `f1edge_hidens` must *complete at all* before any trajectory
> bar is even meaningful. A candidate that improves compact-probe accuracy and stalls the stiff edge is not
> a candidate.

**AUDIT.md's own recommendation #2 is wrong and has been struck through** (2026-08-05). It said
"bit-diff a run with the ramp vs without on a healthy config; if inert, delete". The ramp *is*
inert by that test, and deleting it is still fatal — because a healthy config never exercises what
the constant protects. Do not let that phrasing back into the plan.

## 5. Rules the plan must respect

- **Measure on top of the phase-1a fix, never against stock.** The two constants interact through
  the early `Pb`; a stock baseline mixes in the artifact that was just removed.
- **CLAUDE.md rule 5.** This is an iterative/integrated path, so a per-call equivalence is
  necessary but *not sufficient*: clear it with a full-run equivalence on the stiffest regimes, in
  **separate processes** (trinity leaks module-level global state in-process), at **matched
  simulation `t`** (arms truncate at different `t` — interpolate, do not compare nearest
  snapshots). `docs/dev/phase1a-init/harness/matched_t.py` does the interpolation.
- **Define the bar before editing**, and keep the pre-registered version in the doc even if it is
  later re-sited. Model: `docs/dev/phase1a-init/PLAN.md` §4, where the original bar was missed,
  re-sited by maintainer sign-off, and both are still on the page.
- **Do not bump numpy past 2** (see repo `CLAUDE.md`) — the bubble integrator's monotonic guard is
  sensitive to floating-point output changes, which is *directly* relevant here since you are
  changing what goes into that integrator.
- **Physically plausible configs only.** Unphysical inputs exercise regimes the code never runs in
  and hide real regressions.

## 6. Configs to design against

| config | where | why it matters here |
|---|---|---|
| `f1edge_hidens` | `docs/dev/performance/f1edge_hidens*.param` | `nCore=1e6`; the stiffness gate. Ablation stalls here — this is the config that decides the design |
| compact probe | `docs/dev/phase1a-init/harness/params/probe.param` | sub-GMC scale, where an absolute 1e-3 Myr is most wrong relative to the physics |
| GMC control | `docs/dev/phase1a-init/harness/params/gmc_control.param` | the published regime; must not move |
| `simple_cluster` | `param/simple_cluster.param` | the default everything else is tested at |
| `f1edge_lowdens` | `docs/dev/performance/f1edge_lowdens*.param` | the other feedback extreme |

## 7. Open questions the plan should answer

1. **What sets the right switch-off time?** Candidates worth deriving rather than guessing: the
   free-streaming→Weaver relaxation time (`dt_phase0`, already computed in
   `phase0_init/get_InitPhaseParam.py`), some multiple of `t0`, or a state-based condition on
   `(R1/R2)³` rather than a time at all. A state-based trigger may be the honest answer, since
   what the ramp is really protecting against is a geometric ratio, not a clock.
2. **Is a ramp the right shape?** Linear-in-time from zero is arbitrary. If the real function is
   "keep `Pb` below what the ODE can integrate", a ramp is a proxy for a stiffness limiter and
   should perhaps be written as one.
3. **What actually stiffens at `nCore=1e6`?** E8b established *that* it stalls, not *why*.
   **ANSWERED 2026-08-06 (R3, `SWEEP2_PLAN.md` §4):** the phase-1a segment energy ODE, as `Eb`
   collapses under the unramped pressure; the segment `solve_ivp` (hard-coded RK45) stalls in
   micro-steps while the bubble-structure solve stays fast. The answer ruled *both* §7.1 shapes
   out for now: neither a better clock nor a small state guard fixes an integrator failure mode —
   the successor is phase-1a stiffness handling (stiff/switching segment solver, or a terminal
   in-segment `Eb`-floor event), its own workstream.
4. **Does the phase-1a segment schedule change the answer?** The fix landed
   `phase1a_segFrac = 0.1`, so segments now scale with the bubble age. It is plausible — not
   measured — that a scale-relative schedule makes a scale-relative switch-off easier to satisfy.
5. **Is the effect large enough to be worth the risk at all?** 0.006% at the compact probe's observed age is
   small. A legitimate outcome of this work is "document the constant, add the stiffness rationale
   and a test that pins it, change nothing" — that is a *result*, not a failure, and it should be
   on the table from the start.

## 8. Neighbouring constants — already checked, do not re-investigate

The bubble-pressure / `dMdt` chain is full of bare numbers and it is easy to lose a day
re-deriving ones that are already settled. Each of these was checked during the `phase1a-init`
work; line references re-verified against source 2026-08-05.

- **`dR2` — the conduction-layer thickness. CLOSED, no action.** This is the one most likely to
  catch your eye, because it is a hand-sized number sitting right next to the physics you are
  changing. `dR2` is the thickness of the thin conduction layer just inside `R2` where the backward
  Weaver temperature ODE is anchored (`r2_prime = R2 - dR2`), at
  `bubble_luminosity.py:402`:

  ```python
  dR2 = T_init**(5/2) / (constant * dMdt / (4 * np.pi * R2**2))
  ```

  **WARPFIELD floored it** — `dR2min = 1.0e-7`, with a `if Mclus > 1.0e7: dR2min = 1.0e-14*Mclus +
  1.0e-7` branch and a comment saying *"this number might have to be higher… TO DO"*. Since
  `dR2 ∝ 1/dMdt`, a bigger cluster means a thinner layer, and WARPFIELD clamped it. **Trinity uses
  the exact analytic value with no floor and no mass branch**, and
  `test/test_dR2min_magic_number.py` pins the pure `1/dMdt` scaling (a floor would flatten it),
  the conditioning of `R2 - dR2`, and cross-solver agreement on the unfloored layer. The floor
  would have inflated bubble luminosity ~8×. The whole `docs/dev/magic-numbers/` sweep is *named*
  for this story. Nothing to do here — but know it exists, because `dR2` responds to anything that
  moves `dMdt`, and `dt_switchon` moves `Pb`, which moves `dMdt`.
- **`_T_INIT_BOUNDARY = 3e4`** (`bubble_luminosity.py:52`) — **de-flagged, do not re-raise.** It
  looks leveraged, because `dR2 ∝ T_init^(5/2)` puts it at the 5/2 power of the layer thickness.
  It is a documented conduction/ionization boundary whose penalty is a known no-op (≈0.999994),
  and it is separately studied in `docs/dev/misc/tinit-sensitivity.md`, which concluded 3e4 is
  conservative. Its only open tail is that doc's recommendation #3 (drop the linear L3 patch),
  already owned there. I raised this as a candidate before checking the siblings; the check
  retired it. Do not repeat that.
- **`dMdt_factor = 1.646`** (`bubble_luminosity.py:299`, in `_get_init_dMdt`) — not a magic number.
  It is the Weaver+77 Eq. 33 similarity coefficient, and it only seeds an initial guess for the
  `dMdt` root-find.
- **Audit #3, `dt = 1e-9` Myr** (`sps/update_feedback.py:184`) — genuinely open, but a different
  mechanism (a central-difference step on the SPS spline, ~10⁶× below the table grid, so it can
  sample spline noise across a knot). Unrelated to the early-`t` bubble pressure. Leave it to its
  own work.
- **Audit #5, `0.05` / `0.9`** — transition-phase gating thresholds, owned by the transition
  workstream, and `AUDIT.md` says explicitly not to re-open the trigger choice from here.

### The one coupling that does matter

`cool_alpha = t·v2/R2` is set from the phase-1a exit state
(`run_energy_implicit_phase.py:662`, `:798`) and consumed **inside** the bubble solve — both the
ODE initial condition (`bubble_luminosity.py:405`) and the ODE itself (`:439`). So the phase-1a
exit state, the bubble solve, and `dt_switchon`'s effect on `Pb` all reach each other. This is not
a defect; it is the reason a change here can show up somewhere that looks unrelated, and the
reason to attribute effects by measurement rather than by reading the call graph.

## 9. Suggested first move

Do not start by writing a candidate. Start with Q7.3: instrument the stalling `f1edge_hidens`
ablation and find out what diverges — `Pb`, the ODE residual, the monotonic guard, the beta-delta
solve. That single measurement determines whether the successor is a better clock or a different
mechanism entirely, and everything else is speculation until it is answered.

## 10. Evidence index

| what | where |
|---|---|
| E8b numbers | `docs/dev/phase1a-init/data/gate_results.csv` (`E8b,*` rows) |
| E8b trajectories | `docs/dev/phase1a-init/data/e8b_m43_noramp.csv`, `e8b_gmc_noramp.csv`, `e8b_hidens_noramp_STALLED.csv` |
| E8b ablation harness | `docs/dev/phase1a-init/harness/e8b_runner.py` |
| E8b write-up + revised recommendation | `docs/dev/phase1a-init/PLAN.md` §8 |
| the audit row that owns this | `docs/dev/magic-numbers/AUDIT.md` finding #2 |
| the sibling fix (model for diagnosis, counter-example for remedy) | `docs/dev/phase1a-init/FINDINGS.md`, `PLAN.md` |
| matched-t comparison tool | `docs/dev/phase1a-init/harness/matched_t.py` |
| 2026-08-06 reproduction of both E8b claims (to the digit / bit-for-bit) | `docs/dev/magic-numbers/data/switchon_repro_ledger.csv` + `switchon_repro_*.csv` |
| the stall mechanism (R3): RK45 segment integrator, not the bubble solve | `docs/dev/magic-numbers/data/switchon_stall_probe.csv`, `switchon_stall_stacks.txt`; harness `docs/dev/magic-numbers/harness/switchon_probe_runner.py` |
| the in-source pin | `trinity/bubble_structure/get_bubbleParams.py` comment at the constant; `test/test_dt_switchon_ramp.py` |
| multi-config screen (2 refs x N configs, matched t, ledger + pass/fail) | `docs/dev/screen/` — built for exactly this kind of change; use it rather than hand-rolling a sweep |
| the `dR2` / `dR2min` story | `test/test_dR2min_magic_number.py`, `docs/dev/performance/BUBBLE_CONDUCTION_STIFFNESS.md` |
| `_T_INIT_BOUNDARY` study | `docs/dev/misc/tinit-sensitivity.md` |
