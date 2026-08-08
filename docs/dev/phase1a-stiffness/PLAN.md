# Phase-1a segment-integrator stiffness — pre-registered plan

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

**Status (2026-08-06):** 🟡 partial — **Batches 1-5 done; C2 is implemented, gated and pushed, with
one bar clause open.** Cleared: P0, P1, P1-free (byte-identical `dictionary.jsonl` on **all five**
configs), P2 (worst 1.007×), P4 (behavioural test, verified failing-first), full suite 1057/0,
`pre-commit`. **Open: P3's mypy clause fails as written** — 144 vs a 137 baseline, all +7 of one
`attr-defined` class the file already carries 49 of (§2 D5). Recorded for a maintainer ruling
rather than reinterpreted; three options and a recommendation are in D5. Batch 6 (re-open
magic-number #2 now that collapse terminates in-band) is unblocked but not started.
§1 is source-verified against `adfc23f`; §3 bars and §5 decision rule were registered *before* any
edit and are unchanged. **D1: production is ≥4.3e4× away from the stall** (whole phase-1a
integrator costs 0.2-0.6 s/run), so the solver swap C1 is ruled out on economics. **D2: the stall
is stiffness, not a singularity** — `Eb` pinned at 1.6e-6 au on a slow manifold, dominant
λ ≈ −1e13, ~7 days to finish one segment — and, newly, **`Eb` never reaches 0, so the existing
`Eb ≤ 0` guard is mis-thresholded as well as out-of-band.** **D3/Batch 3: C2 built** — a
per-segment event at `ENERGY_COLLAPSE_FRAC = 1e-3` of the segment's starting `Eb`, a threshold
bounded on both sides by measurement — and **P0 passes**: the stalling control now ends in 22 s
with the pre-existing `ENERGY_COLLAPSED` fate. **D4/Batch 4: the equivalence screen is clean** —
`dictionary.jsonl` **byte-identical on all five configs** (the bar asked only for
`simple_cluster`), fates unchanged, worst cost 1.007×. C0 is therefore no longer the likely
outcome: inertness is proven, not argued. Next: **Batch 5** (P4 behavioural test, P3 gates, land).
Spun out of
`docs/dev/magic-numbers/SWEEP2_PLAN.md` §4 R3, which measured the stall and named this as the
honest follow-up.

## 0. What this is, and what it is not

The `dt_switchon` ramp (magic-number audit #2) was closed 2026-08-06 as *document-and-pin*
because it protects against a stall that instrumentation traced to **phase 1a's own segment
integrator**, not to the bubble-structure solve as previously recorded. This workstream asks the
question that closure deferred: **is phase 1a's integrator configuration a latent defect worth
fixing, and if so what is the smallest safe fix?**

It is **not** a commitment to change the solver. Phase 1a's RK45 configuration integrates every
config in `test/` and every config in the screen set without complaint; the only measured stall is
in an *ablated* configuration that production never runs. A solver swap moves every published
trajectory, so the burden of proof sits on the change, not on the status quo. Batch 1 exists to
find out whether that burden can be met at all.

## 1. Source-verified observations (2026-08-06 @ `adfc23f`)

Each line below was read from current source, not inherited from a sibling doc.

1. **Phase 1a is the only phase on an explicit non-stiff integrator, with no step bounds.**
   `run_energy_phase.py:309-318` calls `solve_ivp(..., method='RK45', rtol=1e-6, atol=1e-9,
   dense_output=True)` — no `max_step`, no `min_step`. Phases **1b** (`ODE_METHOD = 'LSODA'`,
   `run_energy_implicit_phase.py:176`), **1c** (`:136`) and **2** all use LSODA — which
   auto-switches between stiff (BDF) and non-stiff (Adams) — *and* pass both `max_step` and
   `min_step` (`:1070-1077`, `:631-638`). So the asymmetry is real and one-sided: the phase with
   the least step control is the one integrating the earliest, fastest-changing state.
2. **The existing retry is also non-stiff, and is unreachable in the stall mode.**
   `run_energy_phase.py:320-331`: on `not solution.success` it retries over `dt_segment / 10` with
   **RK23** at 10× relaxed tolerances. RK23 is a lower-order *explicit* method — if the problem is
   stiffness, the fallback shares the failure mode. More importantly the branch is gated on
   `solve_ivp` **returning**: during the measured stall the call simply does not return in any
   practical wall time, so this protection never runs. There is no step budget and no wall guard.
3. **The `Eb ≤ 0` collapse guard is out-of-band with respect to the integrator.** Phase 1a's event
   list (`phase_events.py: build_energy_phase_events`) is `cloud_boundary`, `min_radius`,
   `velocity_runaway` — **no energy-floor event**. `Eb ≤ 0` is checked at
   `run_energy_phase.py:373` and the degenerate-bubble `except` at `:179-195`, both *between*
   segments. A segment that drives `Eb → 0` **inside** its own span therefore grinds in-solver;
   the clean `ENERGY_COLLAPSED` exit that exists for exactly this state can never be reached,
   because reaching it requires the segment to finish first. This is the sharpest statement of
   the mechanism, and it points at a remedy that is not a solver swap.
4. **The measured stall** (`docs/dev/magic-numbers/data/switchon_stall_probe.csv`,
   `switchon_stall_stacks.txt`): with the `dt_switchon` ramp ablated on `f1edge_hidens`
   (`mCloud=1e7, sfe=0.01, nCore=1e6`), `Eb` drains 180 → 121 → 71 → 29 au across segments 1-4
   (every bubble solve healthy, 1.1-1.3 s), then segment 5 runs > 44 min in `rk_step` — four
   SIGUSR1 stack samples identical in every outer frame. The RHS itself is fine; the step *count*
   diverges.

## 2. The load-bearing unknown (this is what Batch 1 answers)

**Does this bite in production — i.e. with the ramp active — anywhere?**

Known: **no**, on the five screen configs at `stop_t = 0.02` Myr (`simple_cluster`,
`f1edge_lowdens`, `f1edge_hidens`, `m43_probe`, `gmc_control`), all of which completed in minutes
during the magic-numbers round-2 screen (`docs/dev/magic-numbers/data/pdotdot_screen_results.csv`).
Unknown: whether any config approaches the cliff — a segment whose step count is orders of
magnitude above its neighbours is a near-miss, and near-misses are what decide whether this is a
latent defect or a curiosity of an ablated configuration.

The answer changes the acceptable remedy, which is why it comes first:

| Batch-1 finding | what becomes justified |
|---|---|
| Production configs stall or near-miss | a real fix, up to and including the solver swap (C1) |
| Production is far from the cliff, but the failure is one line from being clean | the in-band guard only (C2) — cheap, provably inert on healthy configs |
| Production is far from the cliff and nothing is cheap | **change nothing**; document the asymmetry + the unreachable retry (C0) |

### D1 — ANSWERED 2026-08-06 (Batch 1 run; `data/seg_stepcount.csv`, `..._summary.csv`)

Five production configs (ramp active) + the ablated `f1edge_hidens` positive control, `stop_t =
0.003` Myr (which covers all of phase 1a, `TFINAL_ENERGY_PHASE = 3e-3`), instrumented with
`harness/seg_stepcount_runner.py`. 443 recorded `solve_ivp` calls.

| run | 1a segments | median steps/seg | max steps | max nfev | max wall | total 1a solver wall |
|---|---|---|---|---|---|---|
| `simple_cluster` | 96 | 1 | 2 | 20 | 0.010 s | 0.43 s |
| `f1edge_lowdens` | 32 | 1 | **4** | 44 | **0.021 s** | 0.20 s |
| `f1edge_hidens` | 101 | 1 | 2 | 20 | 0.011 s | 0.45 s |
| compact probe | 131 | 1 | 2 | 20 | 0.012 s | 0.59 s |
| `gmc_control` | 77 | 1 | 2 | 20 | 0.014 s | 0.39 s |
| **`hidens_ablated`** (control) | 4 | 2 | 2 | 20 | 0.013 s | **1 call STALLED** |

**Production is nowhere near the cliff: worst case 4 accepted steps and 0.021 s in a single
segment, against a control call that ran >900 s before the wall cap killed it (exit 124) — a
lower bound of ~4.3e4× in wall, and unbounded in steps.** Two further facts the numbers make
plain:

1. **The whole phase-1a segment integrator costs 0.2-0.6 s per run.** RK45 clears a typical
   segment in *one accepted step*, while the segment's bubble solve costs ~1.3 s. So there is no
   performance case for a stiff solver here — C1 could only ever buy robustness, never speed, and
   it would move every published trajectory to do it.
2. **The failure is binary, not gradual — there is no near-miss gradient to measure.** The
   control's first three calls are indistinguishable from production (2 steps, 0.013 s); the
   fourth never returns. So "distance from the cliff" cannot be read off a step count, and a
   margin-based argument for safety would be false comfort. What decides reachability is whether
   a config drives `Eb → 0` *inside* a segment — and §1.3 notes phase 1a already carries a
   between-segment `Eb ≤ 0` handler, i.e. the code anticipates that collapse; the stall is what
   happens when it lands mid-segment instead of on a boundary. None of the five configs does
   this with the ramp active, which is why none of them stalls.

**Cross-check with the parent workstream:** the stalling call's entry state is `Eb = 29.2417 au`
at `t = 2.6037e-7` Myr — the same state at which `docs/dev/magic-numbers/data/switchon_stall_probe.csv`
recorded its last completed bubble solve (`Eb` drained 180 → 121 → 71 → 29 across four segments).
Two independent instrumentations, taken a day apart by different harnesses, agree on where the
run dies; this one adds *which call* dies, which the earlier probe could not see.

**Verdict: row 2 of the table above.** C1 (LSODA swap) is ruled out by §5.3 — Batch 1 does not
show production configs reaching the stall, and consistency alone is explicitly not a gate. C2
(in-band `Eb`-floor terminal event) stays eligible and is now the only candidate worth building,
with C0 the fallback if it cannot clear P1-free byte-identity. Batch 2 still has to run first:
if the cause is a singularity at `Eb → 0` rather than stiffness, that confirms C2 *and* would
have made C1 useless anyway (§5's trap).

### D2 — ANSWERED 2026-08-06 (Batch 2; `data/stall_anatomy.csv`, `harness/stall_anatomy_runner.py`)

938 samples taken *inside* the stalling call (segment 4 of the ramp-ablated `f1edge_hidens`
control), one per 500 RHS evaluations — 469,000 evaluations over 233 s — each carrying the state,
the RHS vector, and the eigenvalues of a finite-difference Jacobian of the 3-state RHS.

| quantity | measured | reads as |
|---|---|---|
| `t_now` | advances **monotonically**, +1.476e-12 Myr over 233 s | **not** a singularity: no asymptote to a finite `t*` |
| remaining span | 3.85e-9 Myr at 6.3e-15 Myr/s | **~7 days to finish this one segment** |
| `Eb` | pinned at **1.64e-6 au** (quartile medians 1.639/1.642/1.639/1.640; entered at 29.24) | collapsed 7 decades, then quasi-steady — bounded, never negative |
| \|RHS\| | median 8.33e9, bounded | no blow-up |
| dominant eigenvalue | negative in 660/938 samples, median **−1.03e13**; \|λ\| median 3.4e13 | fast **decaying** mode — the textbook stiffness signature |
| step size | median h = 2.8e-17, i.e. h/span = 1.1e-9 | ~3000× below even the RK45 stability limit 2.8/\|λ\| ≈ 8e-14, so *accuracy* control is pinning it too |
| cost split | 93% of wall inside the RHS, 2010 evals/s (~0.5 ms/eval, same as production) | millions of **cheap** evals, not a few expensive ones — the `solve_R1` root-find did not become the bottleneck |

**D2 = stiffness, not a singularity.** `Eb` collapses onto a slow manifold where huge gain and
loss terms nearly cancel (|dEb/dt| ≈ 8.3e9 while `Eb` holds at 1.6e-6), leaving a fast decaying
mode of |λ| ≈ 3e13 that an explicit method must resolve step by step. So §5's trap resolves in
favour of the stiff reading — LSODA/BDF *would* step over this.

**That does not revive C1**, and the pre-registered rule is why: §5.3 requires genuine stiffness
**and** Batch 1 showing production configs reaching it. Batch 1 showed the opposite (≥4.3e4× away,
0.2-0.6 s total for the whole phase-1a integrator). Buying robustness for an unreachable state by
moving every published trajectory remains a bad trade; the mechanism changes the *diagnosis*, not
the *economics*.

**New finding — the existing guard is mis-thresholded, not merely out-of-band.** §1.3 recorded
that the `Eb ≤ 0` check sits between segments. Batch 2 adds the sharper problem: **`Eb` never goes
≤ 0.** It stabilises at 1.6e-6 au and stays positive (sample minimum 3.8e-7). So even if the
segment did complete, `run_energy_phase.py:373` would not fire and the run would carry on
integrating a bubble whose energy has collapsed 7 decades. Consequences for C2, which this makes
the clearly right remedy:

- the floor must be a **positive threshold**, not zero;
- it must be **derived, not guessed** (§7) and **scale-relative** — a healthy segment entry `Eb`
  spans 90-235,000 au across the Batch 1 configs, so any absolute constant would be a new magic
  number of exactly the kind this workstream's parent audit exists to remove. Phase 1c's
  `energy_floor = 1e3` (`phase_events.py: build_transition_phase_events`) is an absolute value
  and is **not** a template to copy here;
- C2 stays byte-identity-testable: on a healthy config the threshold is never approached.

### D3 — Batch 3 result: the threshold, derived (2026-08-06; `data/candidate_gate.csv`)

C2 is built and clears P0. The part worth recording is **how the threshold was fixed**, since
"do not fix a magic number with a new one" (§7) is this workstream's own rule:

| bound | measured | source |
|---|---|---|
| healthy segments never lose energy — worst per-segment ratio **1.0268** over 437 segments, 5 configs, 4 decades of mass/density | ⇒ any `frac < 1` is unreachable in a healthy run, so the event is inert there | `data/seg_stepcount.csv` |
| the deepest per-segment drop that **still integrates** is **0.412** (the control clears segments at 0.67, 0.58, 0.41) | ⇒ `frac` must sit well *below* 0.41 or it would end runs that can continue | `data/seg_stepcount.csv` |
| by **5.47e-8** of segment-start the solver is already dead | ⇒ `frac` must sit well *above* that to fire in time | `data/stall_anatomy.csv` |

`ENERGY_COLLAPSE_FRAC = 1e-3` sits 2.6 decades below the survivable band and 4.2 decades above
the dead zone — near the geometric middle of the measured window, with the margins themselves
pinned by `test_threshold_keeps_its_measured_margins`. Being *relative to the segment's starting
`Eb`* is what makes it scale-free, which matters because segment-start `Eb` spans 90 → 8.9e7 au
across the screen set; an absolute floor could not have worked.

Result on the positive control: the run that ground for >900 s (≈7 days extrapolated for one
segment) now ends in **22 s** with `ENERGY_COLLAPSED` (code 51) — the fate phase 1a *already*
uses for a dead bubble, so no new stopping outcome enters the model. It fires in segment 4,
exactly the segment Batch 2 anatomised.

**Not yet cleared:** P1/P1-free (Batch 4) — the inertness argument above is measured on segment
*entry* values, and byte-identity on a healthy config is what actually proves it. `test_run_smoke`
(the quickstart golden triple) already passes unchanged, which is a necessary but not sufficient
sign.

### D4 — Batch 4 result: C2 clears every bar (2026-08-06; `data/equivalence_screen.csv`)

`docs/dev/screen/screen.py`, `--before 85b347a` (pre-candidate) `--after WORKTREE`, `stop_t = 0.02`
Myr, all five configs, separate processes, matched `t`, fate from `metadata.json[termination]`.

| config | ΔR2 at every compared time | fate | `dictionary.jsonl` | wall ratio |
|---|---|---|---|---|
| `simple_cluster` | +0.000% | unchanged | **byte-identical** (125 snapshots, 2.66 MB) | 0.975 |
| `f1edge_lowdens` | +0.000% | unchanged | **byte-identical** (61) | 0.969 |
| `f1edge_hidens` | +0.000% | unchanged | **byte-identical** (127) | 1.007 |
| compact probe | +0.000% | unchanged | **byte-identical** (161) | 0.932 |
| `gmc_control` | +0.000% | unchanged | **byte-identical** (107) | 0.992 |

- **P1-free: PASS, and stronger than the bar asked for.** It required byte-identity on
  `simple_cluster`; the candidate delivers it on **all five** configs, including the stiff edge and
  the sub-GMC scale. This is the proof the §2/D3 inertness argument needed — that argument rested
  on segment-*entry* ratios, which cannot by themselves rule out a mid-segment crossing. Now it is
  not an argument at all: the bytes are the same, so the event demonstrably never fires in a
  healthy run.
- **P1: PASS** trivially (a byte-identical run has ΔR2 ≡ 0 and an unchanged fate).
- **P2: PASS**, worst ratio **1.007×** against a 1.5× ceiling; three of five configs are slightly
  faster, which is run-to-run noise on a contended container, not a speedup to claim.

**Decision rule §5.2 fires: land C2 alone**, and explicitly do not also swap the solver — with
collapse now terminating in-band, C1's remaining benefit is unmeasured and its cost (moving every
published trajectory) is unchanged. Remaining for Batch 5: P4's behavioural failing-first test and
P3 (full suite / pre-commit / mypy), then the status flip.

### D5 — Batch 5 result: P4 passes, P3 passes except one clause (2026-08-06)

- **P4 PASS, verified failing-first *properly*.** `test/test_phase1a_collapse_terminates.py` runs
  the collapse configuration end-to-end in a subprocess and asserts it **terminates** with the
  pre-existing `energy_collapsed` fate — behaviour, not solver identity, as the bar required.
  With the guard: passes in ~9 s. With the two production files reverted to `85b347a`: **times
  out and fails** (the run never finishes). *Method note worth keeping:* the first attempt at this
  check used `git stash` on files that were already committed, so it stashed nothing and the test
  "passed" against the unchanged candidate — a vacuous check that looked like a real one. Redone
  with `git checkout 85b347a -- <files>`. A failing-first check that cannot be seen to fail is
  worth nothing.
- **P3 partially met.** Full suite **1057 passed / 0 failed** (up from 1018: +38 unit tests on the
  threshold's margins, +1 behavioural). `pre-commit run --all-files` passes.
- **P3's mypy clause FAILS AS WRITTEN, and is recorded rather than reinterpreted.** The bar says
  "no new errors vs a baseline worktree"; measured **144 vs 137**, i.e. **+7**. All seven are a
  single class — `"Callable[[Any, Any], Any]" has no attribute "terminal" / "direction" / "name" /
  "is_simulation_ending" / "reason_code" / "reason_message" / "end_code"` — produced by attaching
  metadata to the event closure, which is exactly what **every** event factory in
  `phase_events.py` does: the same file already carries **49** identical `attr-defined` errors
  inside the 137 baseline. So this is the module's established idiom repeated once more, not a new
  kind of type debt.

  **Maintainer decision, not self-approved** (precedent: `docs/dev/phase1a-init/PLAN.md` §4, where
  a bar was re-sited by sign-off with both versions left on the page). The options:
  1. **Accept** — read the bar's intent as "no new *kind* of error" and record the +7 as idiom
     conformance. Costs nothing, leaves `mypy trinity` at 144.
  2. **Silence** — add `# type: ignore[attr-defined]` to the seven new lines. Makes the number
     match, but the 49 sibling lines carry no such comment, so the new factory would be the only
     annotated one in the file — conformance to a metric at the cost of consistency with the code
     around it.
  3. **Revert** the candidate over a cosmetic typing count — not proportionate to the defect it
     fixes, and it would leave the stall in place.

  Recommendation: **option 1**, because option 2 games the number without improving type safety
  (the attributes are still dynamic) and would have to be undone the day someone types the whole
  module properly. Not applied unilaterally — the bar stays marked FAILED-AS-WRITTEN until ruled
  on.

## 3. PRE-REGISTERED BARS (registered 2026-08-06, before any `trinity/` edit)

Per `docs/dev/phase1a-init/PLAN.md` §4 precedent, these stay on this page verbatim even if later
re-sited; a re-site is recorded *next to* the original, never over it.

- **P0 — stiffness gate (positive control), checked BEFORE any trajectory bar.** The ramp-ablated
  `f1edge_hidens` run (`stop_t = 0.02`, the exact configuration that stalls today) must
  **complete**, with its stopping fate recorded, in **≤ 3× the ramp-active arm's wall time**
  (measured baseline 5m37s → ceiling ~17 min). A candidate that improves anything else and does
  not clear P0 is not a candidate.
- **P1 — do-no-harm equivalence (the real gate).** `docs/dev/screen/screen.py`, all five configs,
  `--before <pre-change HEAD> --after WORKTREE --stop-t 0.02`, separate processes, matched `t`:
  **|ΔR2| ≤ 0.5% at every compared time and at end of run, AND every stopping fate unchanged.**
  *Expectation stated up front:* a candidate that cannot fire on healthy configs should measure
  ~0; anything above 0.05% means it is firing where it was not meant to — investigate before
  landing, do not widen the bar.
- **P1-free — the free-win variant.** If the candidate is inert by construction on healthy configs
  (e.g. a threshold never crossed), P1 is replaced by the stronger **byte-identical
  `dictionary.jsonl`** on `param/simple_cluster.param` vs the pre-change ref, per the `phase1a-init`
  G1a standard. Claiming "inert" without byte-identity is not accepted.
- **P2 — cost.** No config more than **1.5×** slower end-to-end than the pre-change arm on the
  screen set. (Phase 1a is a small share of run time; a stiff solver that doubles it is not free.)
- **P3 — suite & style.** Full `pytest` green (current baseline **1015 passed / 0 failed**);
  `pre-commit run --all-files`; `mypy trinity` no new errors vs a baseline worktree (**137**
  pre-existing at `adfc23f`).
- **P4 — failing-first.** Whatever lands carries a test that **fails before the change and passes
  after**, asserting behaviour (a segment that drives `Eb → 0` terminates rather than grinding),
  not the identity of the solver.

## 4. Candidates (risk-ordered; the plan picks between them, Batch 3)

- **C0 — change nothing, document.** Record in-source that 1a is deliberately RK45-without-bounds
  while 1b/1c/2 are LSODA-with-bounds, and that the RK23 retry is unreachable during a grind.
  Zero risk, zero benefit beyond the next reader's time. **A legitimate outcome, not a failure.**
- **C2 — in-band energy-floor terminal event** (smallest real change). Add an `Eb`-floor event to
  `build_energy_phase_events` so the collapse the code *already* handles at `:373` is detected
  *inside* the segment and terminates it cleanly via the existing `ENERGY_COLLAPSED` path.
  Attractive because on any healthy bubble the threshold is never crossed ⇒ **P1-free
  byte-identity is achievable**, and it fixes the precise gap in §1.3. Risk: choosing a threshold
  — it must be a value the run only reaches when already collapsing, and it must be derived, not
  guessed (this workstream is a magic-number spin-off; do not fix a magic number with a new one).
  **Sharpened by Batch 2 (§2 D2):** the threshold must be **positive** — `Eb` stalls at 1.6e-6 au
  and never reaches 0, so the existing `Eb ≤ 0` test would miss this state even if the segment
  completed — and **scale-relative**, since healthy segment-entry `Eb` spans 90-235,000 au across
  the Batch 1 configs.
- **C1 — LSODA + `min_step`/`max_step`, matching 1b/1c/2.** The principled fix if Batch 2 shows
  genuine stiffness. Highest risk: it changes every trajectory in phase 1a, including the
  published regime, so it must clear P1 on its own merits and cannot be justified by consistency
  alone. Note `min_step` gives the run a *failure* instead of a grind, which makes the existing
  RK23 retry reachable again.
- **C3 — explicit step/wall budget** on the 1a segment solve. A crude backstop that converts a
  hang into a diagnosable failure. Only worth landing if C1/C2 are ruled out and Batch 1 says
  production can reach the cliff.

## 5. PRE-REGISTERED DECISION RULE

Applied in this order; each condition is checked against measurements, not judgement:

1. **If Batch 1 finds no production config within 2 orders of magnitude of the stall's step count
   AND Batch 3 produces no candidate that clears P1-free byte-identity → C0.** Write the result,
   change nothing, close the workstream. This is an expected outcome and is worth as much as a
   fix.
2. **If C2 clears P0 and P1-free byte-identity → land C2 alone.** Do not also swap the solver
   "while we are here": once collapse terminates cleanly in-band, the grind mode is gone and C1's
   remaining benefit is unmeasured.
3. **C1 is landed only if** Batch 2 shows *genuine stiffness* (not a singularity at `Eb → 0` —
   see the trap below) **and** Batch 1 shows production configs reaching it **and** C1 clears P0,
   P1 and P2. Consistency with 1b/1c/2 is a motivation, never a gate.
4. **If a candidate improves the compact regime and stalls or slows the stiff edge, it is not a
   candidate** — P0 first, always.

**The trap this rule exists to avoid.** "RK45 grinds" has two possible causes with *opposite*
remedies: genuine stiffness (large negative eigenvalue; LSODA switches to BDF and sails through)
versus a **singularity** as `Eb → 0` (the pressure `Pb ∝ Eb/(R2³−R1³)` and the bubble state
degenerate; *every* adaptive solver grinds into a singularity, LSODA included). If Batch 2 shows
the latter, C1 is the wrong fix for the right symptom and would burn a full-run gate to learn it.
Distinguishing them is Batch 2's entire job, and no candidate is chosen before it reports.

> **RESOLVED 2026-08-06 — it is stiffness** (§2 D2: `t` advances monotonically, `Eb` and the RHS
> stay bounded, dominant λ ≈ −1e13). So the trap did *not* fire in the direction feared: LSODA
> would genuinely step over this. C1 is nonetheless still out, by rule 3's *other* conjunct —
> Batch 1 measured production ≥4.3e4× away from the stall. Recording this explicitly because it
> is the case the rule was written for: the mechanism now argues *for* the bigger change, and the
> pre-registered economics still argue against it. The rule wins; that is what pre-registering is
> for.

## 6. Batches — the runnable unit ("run batch 1")

Each batch is independently runnable and restartable, states its own exit criteria, and ends by
committing artifacts + writing its result back into this doc (🔄/💾). Costs are wall-clock on a
4-core container, from this session's measurements.

| # | name | entry | deliverable | exit / decision | cost |
|---|---|---|---|---|---|
| **0** | Pre-registration | — | this doc + workstream registration | committed before any `trinity/` edit | done (this commit) |
| **1** | ✅ **DONE 2026-08-06 — Reconnaissance: does it bite in production?** | Batch 0 | `harness/seg_stepcount_runner.py` → `data/seg_stepcount.csv` (443 calls) + `data/seg_stepcount_summary.csv`, over the 5 screen configs (ramp active) + the ablated `f1edge_hidens` **positive control** | **D1 answered (§2):** production worst = 4 steps / 0.021 s per segment vs a control call that never returned in 900 s (≥4.3e4× wall). **C1 ruled out; C2 the only live candidate, C0 the fallback** | ran ~55 min |
| **2** | ✅ **DONE 2026-08-06 — Mechanism: stiffness or singularity?** | D1 | `harness/stall_anatomy_runner.py` → `data/stall_anatomy.csv` (938 samples / 469k RHS evals inside the stalling call, with Jacobian eigenvalues) | **D2 answered (§2):** **stiffness** — `Eb` pinned at 1.6e-6 au on a slow manifold, dominant λ ≈ −1e13, `t` advancing monotonically at ~7 days/segment. C1 still ruled out (Batch 1 economics); **C2 confirmed as the remedy, and its threshold must be positive and scale-relative** because `Eb` never reaches 0 | ran 4 min (cut short by a container restart; regime stationary) |
| **3** | ✅ **DONE 2026-08-06 — Candidate bake-off on the positive control** | D2 | C2 implemented (`make_energy_collapse_event` + per-segment wiring in `run_energy_phase.py`) and run against the ablated `f1edge_hidens`; ledger `data/candidate_gate.csv`; unit tests `test/test_energy_collapse_event.py` | **P0 PASS** — the run that used to grind (~7 days/segment) now completes in **22 s** with fate `ENERGY_COLLAPSED`, 46× inside the 1011 s ceiling and 15× faster than the ramp-active arm. Only C2 was built; C1 was already out | ran 5 min |
| **4** | ✅ **DONE 2026-08-06 — Equivalence on production configs** | C2 passed P0 | `docs/dev/screen/screen.py` over all 5 configs (before `85b347a` vs WORKTREE) → `data/equivalence_screen.csv` | **P1 PASS · P1-free PASS on all five configs, not just `simple_cluster` · P2 PASS** (worst 1.007× vs the 1.5× ceiling). §5.2 fires: **land C2 alone** | ran 35 min |
| **5** | 🟡 **DONE 2026-08-06 except one clause — Land** | Batch 4 verdict | C2's smallest diff (16 lines in `run_energy_phase.py` + one event factory) + P4 behavioural test `test/test_phase1a_collapse_terminates.py`, verified failing-first; then P3 | **P4 PASS · suite 1057/0 · pre-commit PASS · mypy 144 vs 137 baseline = FAILED AS WRITTEN** (+7, all one pre-existing idiom class — §2 D5). Awaiting a ruling on that clause | ran 25 min |
| **6** | **(conditional) Revisit magic-number #2** | a candidate landed AND P0 holds *without* the ramp | re-run the §4 R1/R2 protocol from `magic-numbers/SWEEP2_PLAN.md` with the new protection in place: can `dt_switchon` now be deleted or made scale-relative? | its own pre-registered bars, its own commit — do **not** fold into Batch 5 | ~1 h |

**Batch 6 is the prize and is deliberately last.** If phase 1a can survive an `Eb` collapse
cleanly, the ramp's stated justification weakens and #2 can be re-opened *with evidence*. Nothing
before Batch 6 should touch `get_bubbleParams.py`.

## 7. What NOT to do

- Do not re-tune or delete `dt_switchon` inside Batches 1-5. It is pinned
  (`test/test_dt_switchon_ramp.py`) and its removal has its own gate (Batch 6).
- Do not gate on the ablated config alone. It is the positive control, not the target; P1 is where
  the risk actually lives.
- Do not swap the solver for consistency. §5.3 makes this explicit because "1b does it this way"
  is exactly the kind of argument that moves published trajectories for free.
- Do not fix a magic number with a new one: any threshold introduced (C2's floor, C3's budget)
  must be derived from a measured quantity and carry its rationale in-source, per the standard
  this workstream's parent audit sets.
- Do not reformat or refactor the phase runners beyond the chosen candidate's lines.
- Do not bump `numpy` past 2 (repo `CLAUDE.md`) — the bubble integrator's monotonic guard is
  sensitive to FP output changes, and P1-free byte-identity would catch it late.

## 8. Provenance

Spun out of `docs/dev/magic-numbers/SWEEP2_PLAN.md` §4 R3 (the instrumented reproduction that
corrected the stall mechanism) and §5 (the decision that closed #2 as document-and-pin). Sibling
context: `docs/dev/magic-numbers/SWITCHON_BRIEF.md` §4, `docs/dev/phase1a-init/PLAN.md` §8 (the
E8b write-up, mechanism corrected in place 2026-08-06).

## 9. Artifact index (filled in as batches run)

| what | where |
|---|---|
| the stall, instrumented (input evidence) | `docs/dev/magic-numbers/data/switchon_stall_probe.csv`, `switchon_stall_stacks.txt` |
| ramp-active baseline for `f1edge_hidens` (P0 reference, 5m37s / 127 rows) | `docs/dev/magic-numbers/data/switchon_repro_hidens_active.csv` |
| Batch 1 per-segment step counts (443 calls, 6 runs) | `data/seg_stepcount.csv` (harness: `harness/seg_stepcount_runner.py`) |
| Batch 1 aggregates (the §2 D1 table, regenerated by `--reduce`) | `data/seg_stepcount_summary.csv` |
| Batch 2 stall anatomy (938 samples, Jacobian eigenvalues) | `data/stall_anatomy.csv` (harness: `harness/stall_anatomy_runner.py`) |
| Batch 3 candidate gate (P0) + threshold derivation | `data/candidate_gate.csv`; tests `test/test_energy_collapse_event.py` |
| Batch 4 equivalence screen (P1 / P1-free byte-identity / P2) | `data/equivalence_screen.csv` |
| multi-config screen harness | `docs/dev/screen/screen.py` |
