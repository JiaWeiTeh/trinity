# Phase-1a early-time discretisation fix — implementation & verification plan

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

**Status (2026-08-14):** ✅ **SHIPPED — merged to `main` 2026-08-06** (was branch `hotfix/early-approximations`); §3 settled, fix shipped (0df441f + a944727), **every gate G1–G7 passed at the time**. ⚠️ **The goldens this workstream re-baselined moved again on 2026-08-14** — `test_run_smoke.py` (`R2`/`v2`/`Eb`) and the `(cool_beta, cool_delta)` pair, ~1%. That is **not** a regression here: `c43a50e` (PR #738, `phii-identity` C3c) removes the `P_HII` channel that was carrying un-ramped pressure past `dt_switchon`, which moves the same phase-1a exit state these goldens pin. They were re-baselined the same day; the mechanism and the before/after table live in `docs/dev/phii-identity/PLAN.md` and `docs/dev/phii-identity/data/g34_golden_rebaseline.csv`. The G2 bar was re-sited by maintainer sign-off (§4: `|ΔR2| < 5%` at 1 Myr or end of run, *and* fate unchanged) and all four configs pass it, worst +0.44% (11x inside); both long configs are measured to their true natural end and the arms **converge** (GMC −0.001% at 2 Myr). The fix is also **16% faster** end-to-end. The three goldens on the stock phase-1a exit state are re-baselined (0ffa994): default suite **987 passed / 0 failed**, `pre-commit` passes, `mypy` shows no new errors vs the stock worktree. Magic-number audit finding #4 (`vd = -1e8`) is marked fixed in `docs/dev/magic-numbers/AUDIT.md` (§8 E8a), and #2's recommendation there corrected with the E8b result; nothing outstanding before merge. §§8-9 record adjacent follow-ups.

## 0. Mission (read this first)

You are implementing and gating a fix for a **confirmed S1 discretisation
defect** in phase 1a. The diagnosis is done — do not re-litigate it, but do
re-verify the load-bearing numbers against the committed CSVs before building
on them. Every baseline you need is committed under `data/` (manifest:
`data/README.md`); the prototype that produced the converged trajectories is
`harness/patched_runner.py` (commands: `harness/README.md`). **You should not
need to re-run any diagnostic to start.** The deliverable is a production
implementation in `trinity/`, gated per CLAUDE.md rule 5, with failing-first
tests, and the gate artifacts committed back into this workstream.

## 1. The defect (two halves, one root cause)

1. **Absolute segment duration vs scale-dependent physics.** Phase 1a
   integrates in fixed segments of `SEGMENT_DURATION = 3e-5` Myr (30 yr) with
   driving terms frozen per segment. The free-streaming → Weaver relaxation
   time scales with object size; at sub-GMC scale (`mCloud=300`, `sfe=0.01`,
   `nCore=8.7e3`) the entire relaxation fits inside segment 1, so the frozen
   state integrates a snowplow on unphysical values for 30 yr.
2. **The `vd = -1e8` override** (`energy_phase_ODEs.py`, segment 0 only,
   `EarlyPhaseApproximation` flag): a constant dv/dt in pc/Myr², giving the
   closed form `v_exit = v0 − 1e8·SEGMENT_DURATION = 722.82 km/s` for **every
   run on the bundled SB99 tables** regardless of mass/SFE/density (measured:
   `data/segment1_exit.csv`, mass and nCore sweeps). It is ~10²x too weak
   against the true early RHS, tolerance-independent (no rtol study surfaces
   it), and *partially cancels* half 1 — ablating it alone makes things worse
   (`data/*_noapprox.csv`: exit at 2429 km/s). The flag also **leaks**: default
   `True`, not in `default.param`, cleared on only one exit path — a
   documented config can carry `vd=-1e8` into phases 1b/1c and misdiagnose
   `VELOCITY_RUNAWAY` (see FINDINGS §"Independent corroboration", adopted from
   the `bugfix/code-audit` branch's Cluster C).

Consequences at sub-GMC scale: observed radius crossed at 620 yr instead of
1.35e4 yr (~22x), v ~12x observed at crossing. At GMC scale: −10% R2 after
segment 0, decaying transient, asymptote preserved. Full evidence:
`FINDINGS.md` §Verdicts, §Numerics-vs-physics.

## 2. The fix candidate (prototyped, measured — not yet production)

**Log-spaced segments + no override:** `dt_seg = eps·(t_now − tSF)` with
`eps = 0.1`, **uncapped** (see §3.1 — an earlier draft of this doc said
"capped at `SEGMENT_DURATION`"; that was wrong and is corrected below), and the
`vd = -1e8` branch removed. Prototype: `harness/patched_runner.py`
(`TRIN_LOGSEG=0.1 TRIN_NO_EARLY_APPROX=1`). Measured results to reproduce:

| Check | Result | Baseline file |
|---|---|---|
| compact probe at observed age (2.1e4 yr) | R2 = 0.196 pc, v2 = 5.1 km/s (obs: 0.153±0.011 pc, 5.0 km/s) | `data/m43_logseg.csv` |
| compact-probe free-streaming → Weaver | attractor ratio 1.07 by 1.4 yr, 1.00 by 160 yr; **zero solve_ivp failures** | same |
| GMC equivalence at matched t | −3.8% @1e3 yr, −0.95% @3e3 yr, −0.04% @8e4 yr vs stock | `data/gmc_logseg.csv` vs `data/gmc_control.csv` |
| No manufactured momentum | p = 0.28 vs stock's 283 at 410 yr | `data/m43_logseg.csv` vs `data/m43_probe.csv` |

### Is the override needed at all? (asked 2026-08-04 — answer: no)

Worth stating plainly, because it is the whole point of the fix: **nothing here
tunes `vd = -1e8` to a better value.** The three possible levels of
approximation in the phase-1a RHS are:

| Level | What `vd` is | Status |
|---|---|---|
| 1 | the constant `-1e8` for all of segment 0 | status quo — **deleted** by §3.3 |
| 2 | the force budget, with driving terms frozen per segment | what the fix ships; measured (`data/*_noapprox.csv`, `data/*_logseg.csv`) |
| 3 | the force budget with *nothing* frozen (bubble/shell/feedback recomputed inside the RHS) | not attempted — a bubble solve per solver step, likely 10²-10³x cost |

Level 2 was never reachable from a `.param`: `read_param.py:217` validates user
keys against `default.param`, and `EarlyPhaseApproximation` is absent from it
(verified 2026-08-04), so the flag was `True` for every run ever made. The
prototype had to monkeypatch it.

Was the override load-bearing for stability? **No** — measured, not assumed.
Deleting it alone (`data/*_noapprox.csv`) completes with no solver failures,
and the log-spaced prototype ran compact-probe and GMC scale with **zero `solve_ivp`
failures**. It was papering over the fixed 30-yr segment, not holding the
integrator up. The one honest caveat: deleting it *without* fixing the segment
schedule makes trajectories worse (2429 km/s snowplow — §6), which is the
likely reason it was added in the first place.

After the fix, level 2 with the stock fixed segment remains reachable as
`phase1a_segFrac = 0` — so "no approximation, stock segments" stays available as a
config, and is exactly what gate G1b measures. Level 3 stays open as a possible
follow-up: the log schedule is the cheap approximation to it, bounding the
staleness of the frozen terms to a fixed fraction `eps` of the expansion time
instead of an absolute 30 yr, and the eps convergence study (§3.4) measures how
close to level 3 that gets.

## 3. Design decisions — SETTLED (Batch 0, 2026-08-04)

Decided against current source; each line reference below was re-verified on
2026-08-04 (they all held). Batch 2 implements exactly this — no further
design latitude.

1. **Schedule form: `dt_seg = eps·(t_now − tSF)`, UNCAPPED.** `eps` is a new
   registry param `phase1a_segFrac` (default 0.1) exposed in `default.param`;
   `phase1a_segFrac = 0` falls back to the fixed `SEGMENT_DURATION`, and that is
   the exact-revert path the G1 byte-identity gate uses.
   **Correction to this doc's earlier draft**, which recommended
   `min(eps·(t−tSF), SEGMENT_DURATION)`: the cap is *wrong*. Every validated
   baseline was produced uncapped — `data/m43_logseg.csv` and
   `data/gmc_logseg.csv` both hold `dt/t = 0.100` to the last snapshot — and a
   cap at `SEGMENT_DURATION = 3e-5` Myr binds from t > 3e-4 Myr (300 yr)
   onward, which would revert essentially the whole trajectory to stock 30-yr
   segments and would NOT reproduce the 0.196 pc / −0.04% results this plan
   gates against. The cap is also unnecessary: phase 1a ends at
   `TFINAL_ENERGY_PHASE = 3e-3` Myr (`run_energy_phase.py:54`), so uncapped
   eps=0.1 peaks at `dt = 3e-4` Myr — bounded at 10x stock by construction.
   Do NOT special-case by cloud mass; the schedule is scale-free.
2. **Segment-0 seeding: nothing special needed.** `t0 = tSF +
   free-streaming duration` (`phase0_init/get_InitPhaseParam.py:63-64`), so
   the *age* `t_now − tSF` is strictly positive at loop entry — measured
   1.15e-8 Myr (compact probe) and 1.96e-6 Myr (GMC control). `tSF` defaults to 0
   (`registry.py:425`), so the age-based form is identical to the prototype's
   `eps·t_now` on every committed baseline; age-based is kept because it is
   the correct form if `tSF` is ever non-zero. Guard the degenerate case in
   the same expression that handles `phase1a_segFrac = 0` — one branch, not two:

   ```python
   dt_seg = phase1a_segFrac * (t_now - tSF)
   if dt_seg <= 0:          # phase1a_segFrac=0 (stock schedule) or degenerate age
       dt_seg = SEGMENT_DURATION
   t_segment_end = min(t_now + dt_seg, TFINAL_ENERGY_PHASE)
   ```

   The solver-retry path (`run_energy_phase.py:312`, currently
   `t_now + SEGMENT_DURATION / 10`) must use `dt_seg / 10` for consistency.
3. **Override and flag: both deleted.** Remove the `vd = -1e8` branch
   (`energy_phase_ODEs.py:269-270`) *and* the now-orphaned
   `EarlyPhaseApproximation` flag it exists for — the snapshot field
   (`:100`), its assignment (`:159`), the clear site
   (`run_energy_phase.py:342-344`), the `ParamSpec` (`registry.py:423`), and
   the two reader entries (`_output/trinity_reader.py:155` label and `:990`
   'State' display group). Rationale: deleting the flag is what removes the
   leak pathway permanently — patching exit paths leaves a live flag and the
   same class of bug. **Consumer check (2026-08-04): there are none outside
   `trinity/`** — `paper/`, `tools/`, and `test/` never reference it, so the
   only consequence is one fewer column in new `dictionary.jsonl` output.
   Update the category comment at `_input/param_spec.py:56`, which names the
   flag as an example of `runtime_control`.
   *Leak precision, for the Batch 1 test:* the clear at `:342-343` is
   `loop_count == 0`-guarded and sits after the event check, so **four in-loop
   exits skip it** (`:183` bubble collapse, `:287` cooling_balance, `:330`
   simulation-ending event `return`, `:331` event `break`) — `:379` (Eb<=0)
   is after the clear and does not. The fifth path is the loop never being
   entered at all (`while` false at entry, `:138`), which also leaves the flag
   `True`. Assert on *behaviour*, not on the flag, so the test survives its
   deletion.
4. **eps convergence.** Run eps ∈ {0.3, 0.1, 0.03} on the compact probe + GMC
   control; accept when halving eps moves R2 at the observed age by <1%.
   (eps=0.1 gave ~162 snapshots to 2.4e4 yr at the compact probe — cost is negligible.)
5. **Do not touch** TFINAL, the 1b DT floors, rtol/atol, or the `-1e8`
   constant's value — all measured second-order or irrelevant (E1).

## 4. Verification ladder (CLAUDE.md rule 5 — this is NOT a free win)

Run in order; do not skip a rung because an earlier one passed. Trinity leaks
module-level global state — **all A/B comparisons in separate processes, at
matched simulation t** (runs truncate at different t).

- **G0 — baselines are already captured.** Committed CSVs above + `git show
  HEAD` values. Re-extract with `harness/extract_csv.py` only for new runs.
- **G1 — revert-equivalence, in two sub-gates.** The two halves of the change
  (schedule plumbing, override deletion) must be isolated from each other, so
  **Batch 2 lands as two commits** and G1 straddles them:
  - **G1a (plumbing — bit-identical; this sub-claim IS a free win).** After
    commit 2a (schedule plumbing only, `vd=-1e8` and the flag still present),
    a `param/simple_cluster.param` run with `phase1a_segFrac = 0` must produce a
    **byte-identical `dictionary.jsonl`** vs stock HEAD. Nothing about the
    schedule can change behaviour when it is switched off; if this fails, the
    plumbing is wrong — stop.
  - **G1b (deletion is faithful — free measurement).** After commit 2b
    (override + flag deleted), a run with `phase1a_segFrac = 0` is *by
    construction* the already-measured "hack ablated" configuration, so it
    must reproduce the committed ablation baselines: segment-1 exit
    **2429.4 km/s** on the GMC control (`data/gmc_noapprox.csv`) and
    **2428.6 km/s** on the compact probe (`data/m43_noapprox.csv`), per
    `data/segment1_exit.csv`. This needs no new baseline and proves the
    deletion changed exactly what it should. Note the raw byte-diff vs HEAD
    is meaningless here — the `EarlyPhaseApproximation` column is gone and
    the trajectory legitimately differs.
- **G2 — full-run equivalence on the stiff edges (the real gate).** Configs:
  `param/simple_cluster.param`,
  `docs/dev/performance/f1edge_lowdens*.param`,
  `docs/dev/performance/f1edge_hidens*.param`, the compact probe
  (`harness/params/probe.param`), and the GMC control
  (`harness/params/gmc_control.param`). Bars, at matched t in separate
  processes: GMC-scale configs |ΔR2| < 1% for t ≥ 3e3 yr vs stock (the early
  transient is *supposed* to change — that is the fix); compact probe within 1%
  of `data/m43_logseg.csv` throughout; all runs reach their stock stopping
  fate (no new `VELOCITY_RUNAWAY`/collapse flips); zero solver failures.

  > **ADOPTED 2026-08-05 (maintainer sign-off) — this replaces the
  > stock-comparison half of the bar above.** The original `t >= 3e3 yr, 1%`
  > form is kept in place rather than deleted, so the record shows what was
  > pre-registered, what replaced it, and that the replacement was chosen with
  > the measured numbers for both bars already in hand.
  >
  > **Threshold revised 10% → 5% (maintainer, same day, after adoption.)** The
  > form of the bar — judge at 1 Myr / end of run, plus the fate clause — was
  > never in question; only how much slack it leaves a *future* change. At 10%
  > the bar sat ~23x above the worst measured config, loose enough that a real
  > regression could pass it unnoticed; 5% keeps every config passing with an
  > 11x margin and makes the bar useful as a standing gate rather than a
  > one-time verdict. Note this tightening does **not** re-open the decision:
  > it was made after the measurements, and no measured value moves.
  >
  > The `t ≥ 3e3 yr, 1%` form asks the wrong question. It treats the stock
  > trajectory as ground truth, when the finding of this workstream is that
  > stock's early phase is an *artifact* — its similarity slope is 0.31 where
  > theory says 0.6, and its segment-1 velocity is a constant independent of
  > the cloud. Demanding 1% agreement with a wrong reference, at the very
  > instant phase 1a hands off, is close to demanding the fix do nothing.
  >
  > The bar's real job is narrower: **confirm the change did not break the
  > long-term evolution.** So:
  >
  > **|ΔR2| < 5% at 1 Myr — or at the end of the run if it terminates
  > earlier — AND the stopping fate unchanged.**
  >
  > The fate clause is load-bearing: a loose radius threshold alone could hide
  > a run that collapses when it should not, by comparing at its own truncated
  > endpoint. Note also that "1 Myr" is unreachable for some configs *by
  > physics*, not by budget — `f1edge_hidens` collapses at 0.037-0.047 Myr —
  > which is why the bar has to be written as "or end of run".
  >
  > **CORRECTION (2026-08-05).** An earlier version of this paragraph also
  > listed the GMC control as ending naturally at 0.082 Myr. It does not.
  > Both fixed-arm GMC runs behind that claim (`data/gmc_logseg.csv`,
  > `data/g2_gmc_prod.csv`) were killed by an external SIGTERM — their
  > `trinity.log`s end with `Received SIGTERM, flushing pending snapshots...`
  > and carry no `Simulation ended` line — and the identical endpoint that was
  > read as corroboration is just the fixed 9-snapshot pending flush. Run
  > properly, the fixed arm reaches `stop_t = 2` Myr exactly as stock does
  > (`data/g2_gmc_fixed_full.csv`, 24m41s).
  >
  > **Measured, both configs now at their true natural end** (rows `G2long,*`
  > in `data/gate_results.csv`; separate processes, matched t via
  > `harness/matched_t.py`):
  >
  > | config | ΔR2 @1 Myr | at end of run | fate |
  > |---|---|---|---|
  > | `simple_cluster` (`stop_t=1`) | **−0.078%** | −0.078% @1 Myr | STOPPING_TIME, unchanged; phases energy→implicit on both |
  > | GMC control (`stop_t=2`) | **−0.002%** | −0.001% @2 Myr | STOPPING_TIME, unchanged |
  > | `f1edge_lowdens` | n/a | +0.26% @0.020 Myr | unchanged |
  > | `f1edge_hidens` | n/a | +0.44% @0.037 Myr | SHELL_COLLAPSED, unchanged |
  >
  > So the adopted 5% bar passes by 11x on the worst config (`f1edge_hidens`,
  > +0.44%), 64x on `simple_cluster` and ~5000x on the GMC control — every
  > config clears it on the numbers already measured, with none re-run.
  > The long baselines also say something the truncated ones could
  > not: the two trajectories do not merely stay within a tolerance, they
  > **converge** — GMC ΔR2 runs −28.8% @100 yr → −0.95% @3e3 → −0.28% @1e4 →
  > −0.037% @8e4 → −0.002% @1 Myr, and Δv2 is +0.014% at 2 Myr. The
  > disagreement is confined to the early transient, which is the part this
  > workstream argues stock gets wrong.
  >
  > Note for whoever rules on this: the GMC control **passes the ORIGINAL bar
  > too** (−0.949% at 3e3 yr, just inside 1%). The original bar's failures are
  > `simple_cluster` (−10.4%), `f1edge_lowdens` (+1.7%) and `f1edge_hidens`
  > (−22.8%) — i.e. it is satisfied only by the one config whose scale the
  > `vd = -1e8` constant was tuned for.
- **G3 — asymptotics.** Energy-phase slope `dlnR/dlnt → 3/5` on the
  uniform-density control (harness overlay in `harness/make_figures.py`).
- **G4 — leakage regression.** New failing-first pytest: the
  `cooling_boost_mode theta_target` + `cooling_boost_theta 0.96` config must
  not carry `EarlyPhaseApproximation: true` (or the override's effect) past
  phase 1a, and must not misdiagnose `VELOCITY_RUNAWAY`.
- **G5 — artifact-gone regression.** New failing-first pytest: segment-1 exit
  velocity must be mass-dependent (the 722.8 km/s invariant is the bug's
  fingerprint — assert it is gone across two masses). Use physically
  plausible params per CLAUDE.md, not round numbers.
- **G6 — suite + style.** Full `pytest` green; `pre-commit run --all-files`;
  `mypy trinity` no new errors.
  **Measured 2026-08-05.** Before the re-baseline: 2 failed, 973 passed, 10
  deselected. After (`0ffa994`): **987 passed, 0 failed, 10 deselected** — the
  +12 are the `phase1a_segFrac` validator cases (`17d2ed1`), the +2 are these
  goldens. `pre-commit run --all-files` passes. `mypy trinity` reports 150
  errors in 23 files on **both** this branch and the 99fa204 stock worktree —
  identical count, so no new errors; the 150 are pre-existing and out of scope.
  Both failures were goldens capturing the *stock* phase-1a exit state, which
  this change moves by design — they were the re-baseline decision, not
  regressions:
  - `test_run_smoke.py::test_quickstart_completes_cleanly` — `_FINAL_GOLDENS`
    R2 = 0.2857315 (captured 2026-07-10); the fix gives 0.2595598. That triple
    (R2, v2, Eb) *is* the 1a exit state — `stop_t = 1e-4` Myr is below
    `TFINAL_ENERGY_PHASE`, so the run stops at the end of phase 1a.
  - `test_phase_boundary.py::test_default_run_crosses_energy_to_implicit_boundary`
    — `_GOLDEN` (cool_beta, cool_delta) = (0.759260, −0.035387), and
    `cool_alpha = t·v2/R2` is set *from* the 1a exit state (§8, last block).
  - A third, `test_betadelta_hybr_stress.py::_GOLDEN`, carries the same
    (beta, delta) pair and is `-m stress` (deselected by default) — it needs
    the same re-baseline and will not show up in a default run.
    Re-baselined and **verified green** (`test_hybr_implicit_converges_and_matches_golden`,
    1 passed in 5m33s). Its companion `test_hybr_endtoend_no_crashes` also
    passes at `TRINITY_STRESS_N=2` (1 passed in 10m26s) — so the whole stress
    file is green, run as two separate invocations rather than one.
    Note for whoever runs this next: a first attempt at the
    whole stress file was killed at ~40 min with no summary line, and a
    `pgrep -f "pytest -q -m stress"` liveness check reported it as still running
    because the pattern matched its own shell wrapper. Wait on the **PID**, and
    treat a log with no summary line as a kill, not a pass.
  Earlier drafts of this Status line said "two golden tests"; it is three, two
  of them in the default suite.
- **G7 — persist.** Extract per-gate CSVs into `data/`, regenerate figures,
  update `data/README.md` with exact config + command per artifact, update
  FINDINGS §"What should change" status and this doc's Status line.

## 5. Risks / open questions (carry these, don't rediscover them)

- **1b handoff:** 1b derives `cool_alpha = t·v2/R2` from the 1a exit state;
  the fix changes that exponent (0.456 → ~0.33 at GMC scale). Expected and
  correct, but verify 1b starts cleanly (not at its DT floor) on all G2
  configs. The TFINAL=3e-4 experiment (`data/m43_tfinal3e-4.csv`) shows what
  a poisoned handoff looks like: DT-floor grind + beta-delta warnings.
- **Event arming:** with the override gone, segment-0 v2 stays near v0
  (~3700 km/s) longer; confirm `velocity_runaway` (threshold −500) and the
  other armed events cannot fire spuriously during the relaxation.
- **Sweep runner / SLURM emission** (`run.py --emit-jobs`) must be untouched
  by the new param (schema default only).
- **numpy pin:** stay `<2` (CLAUDE.md) — the bubble integrator's monotonic
  guard is sensitive to FP output changes; G1a's byte-identity gate will catch
  accidental sensitivity.
- **Coarser late-1a segments:** uncapped eps=0.1 makes the *last* 1a segments
  up to 10x longer than stock (3e-4 vs 3e-5 Myr), and 1a freezes driving terms
  per segment — so this coarsens the very end of 1a for published configs too.
  The committed GMC equivalence (−0.04% at 8e4 yr) says it is harmless there;
  G2 must confirm it on both `f1edge` configs, and the eps convergence study
  (§3.4) is the direct test — if eps=0.03 disagrees with eps=0.1, the coarse
  tail is why.
- Full-cloud compact-probe physics (rCloud plausibility at mCloud=300) was validated
  for the probe param; if you build new configs, keep `rCloud_max` checks
  passing.

## 6. What NOT to do

- Do not remove the `vd` hack without the schedule change (measured worse:
  2429 km/s snowplow, 1299x wind impulse).
- Do not "fix" by tuning the `-1e8` constant — any constant is wrong at some
  scale; the closed form guarantees it.
- Do not gate with tolerance sweeps (invariant to 2e-12) or in-process A/Bs
  (global-state leakage), or judge equivalence at unmatched t.
- Do not reformat/refactor the phase runners beyond the schedule + flag-clear
  + override removal; every changed line traces to §3.

## 7. Committed evidence inventory

`data/` (manifest with configs+commands: `data/README.md`): `m43_probe`,
`m43_seg1e-5`, `m43_seg3e-6`, `m43_noapprox`, `m43_tol1e-8`, `m43_logseg`,
`m43_tfinal3e-4`, `gmc_control`, `gmc_noapprox`, `gmc_logseg`, `mass_3e3/4/5/6`,
`ncore_3.7e3`, `ncore_2.6e4`, `segment1_exit.csv` (per-run segment-1 exit
table). `harness/`: `patched_runner.py` (env-var switches), `extract_csv.py`,
`build_segment1_table.py`, `make_figures.py`, `params/`. `figures/`:
convergence, momentum-budget, mass-sweep, Weaver/Spitzer overlays.
Independent corroboration and the leakage details: FINDINGS
§"Independent corroboration — code-audit Cluster C".

## 8. Adjacent numerics in the dMdt chain — checked 2026-08-04

Raised as "there is an artificially imposed dR with a comment like *might change
if the mass is large*". **The memory is exact, the number is real, and it is
already gone from trinity.** Recording the check, and what the cross-check
turned up, because two of the findings matter more than the original question.

**What it was.** WARPFIELD floored the bubble-structure integration offset:

```python
dR2min = 1.0e-7                       # "this number might have to be higher... TO DO"
if Mclus > 1.0e7:
    dR2min = 1.0e-14*Mclus + 1.0e-7
```

`dR2` is the thickness of the thin conduction layer just inside `R2` where the
backward Weaver temperature ODE is anchored (`r2Prime = R2 - dR2`). Since
`dR2 ∝ 1/dMdt`, a bigger cluster ⇒ thinner layer, and WARPFIELD clamped it.

**Status: removed, pinned, and owned elsewhere. No action.**
`bubble_luminosity.py:402` uses the exact analytic value with no floor and no
mass branch. `test/test_dR2min_magic_number.py` pins the pure `1/dMdt` scaling
(a floor would flatten it), the conditioning of `R2 - dR2`, and cross-solver
agreement on the unfloored layer. The whole `docs/dev/magic-numbers/`
workstream is *named* for this sweep, and records that the floor would have
inflated bubble luminosity ~8x.

**Two things the cross-check turned up that DO matter:**

- **E8a — this workstream closes magic-number #4. DONE 2026-08-05.**
  `magic-numbers/AUDIT.md` listed `vd = -1e8` (`energy_phase_ODEs.py:270`) as
  open finding **#4** of #2–#5; commit `a944727` deletes it, and the audit row
  is now marked fixed, citing `data/gate_results.csv`.
  An earlier version of this bullet said to do it "when the branch lands, not
  before" — written while the fix might still have been abandoned. Once the bar
  was signed off, *in* the branch became the right place: the audit entry
  becomes true at exactly the moment the deletion merges, instead of depending
  on a separate commit someone has to remember to make. Recorded there beyond
  "fixed": the audit's own question (*what does `-1e8` represent?*) has no
  answer — the value is an artifact, so it was deleted rather than documented —
  and its **MED, bounded to the 1st segment** severity was under-called, since
  at sub-GMC scale that one segment sets a trajectory that coasts for ~3000 yr.
  The same pass corrected #2's row and recommendation with the E8b result
  below, because that recommendation ("if inert, delete") would have led its
  next reader straight into the stall E8b measured.
- **E8b — magic-number #2 is the same defect class as ours, inside our window.**
  `get_bubbleParams.py:368`: `dt_switchon = 1e-3` Myr, an **absolute** early-time
  constant that linearly ramps the inner radius, `R1_tmp = (t - tSF)/1e-3 · R1`,
  into `bubble_E2P` for all `t <= tSF + 1e-3`. Phase 1a runs to
  `TFINAL_ENERGY_PHASE = 3e-3` Myr, so **this ramp shapes the bubble pressure
  across the first third of phase 1a**, and like `SEGMENT_DURATION` it is an
  absolute time compared against physics whose timescale is not. At sub-GMC scale
  (relaxation complete by ~160 yr = 1.6e-4 Myr) the ramp is still suppressing
  `R1` long after the bubble has physically established itself; at GMC scale
  1e-3 Myr is a genuinely early time. That is exactly the pathology this
  workstream just fixed one instance of.
  **Experiment (E8b): RUN 2026-08-05. Result: the ramp is not a second
  discretisation artifact, and it is NOT removable.** Ablated via
  `harness/e8b_runner.py` (forwards `t=None`, so the ramp branch is skipped and
  nothing else changes), on top of the phase-1a fix, at matched t. Full numbers
  in `data/gate_results.csv`; trajectories in `data/e8b_*.csv`.

  | config | ablation effect | verdict |
  |---|---|---|
  | compact probe | −1.43% @10 yr → **−0.0059% @2.1e4 yr** | inside the 0.1% bar → not an artifact |
  | GMC control | −4.71% @100 yr → −0.017% @8e4 yr | decays away |
  | `f1edge_hidens` | **run STALLS** — 4 rows to 0.26 yr in 90 min wall, vs 127 rows to 2e4 yr in minutes with the ramp | ramp is load-bearing |

  Two conclusions, and the second is the important one:
  1. The original bar is **passed**: at the compact probe's observed age the ramp is worth
     0.006%, ~17x inside the bar and an order of magnitude below the
     `eps` 0.1→0.03 convergence step the shipped schedule itself carries. The
     effect is also *weaker* in the compact regime, not stronger — the opposite
     scaling to `SEGMENT_DURATION` — because the compact probe's `t0` is ~170x earlier, so by
     the time the 1e-3 Myr window closes `(R1/R2)³` has fallen to 4e-3.
  2. **The ramp cannot simply be deleted.** At `nCore=1e6` ablation raises `Pb`
     (the suppressed `R1` was inflating the shell volume), ~~the bubble-structure
     ODE stiffens, and the solve stops converging three segments in~~ —
     **mechanism corrected 2026-08-06** by the magic-numbers round-2
     reproduction (`docs/dev/magic-numbers/SWEEP2_PLAN.md` §4 R3, instrumented):
     the bubble-structure solve stays healthy (~1.3 s/call); the raised `Pb`
     drains `Eb` 180 → 29 au across segments 1-4 and **phase 1a's own segment
     integrator (`solve_ivp`, hard-coded RK45, `run_energy_phase.py:309`)
     stalls in micro-steps** on the stiffened energy ODE. So unlike
     `vd = -1e8` — which papered over a *discretisation* error and was measured
     safe to delete — this constant papers over genuine *stiffness*, and
     removing it is fatal on the stiffest edge.

  **Revised recommendation:** do NOT pursue this as "delete the ramp". ~~Any
  successor must keep the numerical protection while making the switch-on
  scale-relative rather than an absolute 1e-3 Myr~~, and must carry a stiffness
  gate on `f1edge_hidens` (run completes at all) *before* any trajectory bar.
  ~~The constant stays open as magic-number audit finding #2.~~
  **CLOSED 2026-08-06** by the magic-numbers round 2
  (`docs/dev/magic-numbers/SWEEP2_PLAN.md` §5): both E8b claims reproduced on
  HEAD (compact-probe numbers to the digit; the hidens stall bit-for-bit), and with the
  mechanism now known to be the segment *integrator*, the pre-registered
  decision rule landed on **document-and-pin** — constant kept, rationale
  in-source at `get_bubbleParams.py`, pinned by `test/test_dt_switchon_ramp.py`.
  A scale-relative switch-off is no longer the recommended successor shape
  (it delivers full pressure *earlier* at the stiff edge); the honest follow-up
  is phase-1a segment-integrator stiffness handling, its own workstream.

**Explicitly NOT a lead:** `_T_INIT_BOUNDARY = 3e4` (`bubble_luminosity.py:52`),
despite `dR2 ∝ T_init^(5/2)` making it look leveraged. It is de-flagged as
justified by the same audit ("documented conduction/ionization boundary … its
penalty is a known no-op, ≈0.999994") and separately studied in
`misc/tinit-sensitivity.md`, which concluded 3e4 is conservative. Its only open
tail is recommendation #3 (drop the linear L3 patch), already owned by `misc/`.
I raised this as a candidate before checking the siblings; the check retired it.

**One real coupling between this fix and the dMdt chain**, worth knowing but not
an action: `cool_alpha = t·v2/R2` is set from the phase-1a exit state
(`run_energy_implicit_phase.py:662`, `:798`) and consumed *inside* the bubble
solve — the ODE initial condition (`bubble_luminosity.py:405`) and the ODE
itself (`:439`). This fix moves that exponent ~39% at the handoff, so it does
perturb `dMdt` and hence `dR2`. That is the most likely mechanism behind the
phase-boundary golden moving (`cool_beta` 0.759 → 0.888), and it is a
*consequence* of a better 1a exit state, not a new defect.

## 9. Missing infrastructure — there is no multi-config scheme screen

Raised as "is there a test that checks whether a new scheme idea works on normal
runs, like the pdv-trigger configs?". **Checked: no, and this is a real gap.**

- Every end-to-end test in the suite runs the **same single config**
  (`mCloud=1e5, sfe=0.3`): `test_run_smoke.py`, `test_phase_boundary.py`,
  `test_betadelta_hybr_stress.py`, `test_bubble_solver_stress.py`. The only
  outlier is `test_energy_collapse_snapshot.py` (a 5e9 heavy cloud, for the
  collapse handoff specifically).
- The genuinely multi-config coverage — `cal_compact`, `cal_dense`,
  `cal_diffuse`, `cal_mid`, `f1edge_hidens`, `f1edge_lowdens` in
  `docs/dev/transition/pdv-trigger/runs/params/` — exists only as `.param` files
  driven by **HPC sbatch campaigns**. `test/test_bench7_params.py` validates the
  *contents of the param files*, not that a run over them behaves.
- Consequence, felt directly in this workstream: Batch 4 had to hand-roll the
  stock-vs-fixed sweep (worktree + config matrix + matched-t interpolation +
  ledger). Every future scheme change will hand-roll it again, differently, and
  the first attempt will probably repeat the nearest-snapshot-instead-of-
  matched-t error corrected in commit 8457f6e.

**Proposal — `docs/dev/screen/` (a shared harness, NOT part of this fix).**
One runnable that takes two git refs and a config list, runs both arms in
separate processes, interpolates both to a common time grid, and emits a
ledger CSV plus a pass/fail table against a stated bar. The pieces already
exist in this workstream and only need lifting out of it:
`harness/extract_csv.py` (snapshot → CSV), the matched-t interpolation from
`harness/g3_slopes.py`, and `data/gate_results.csv`'s ledger schema.

Suggested screen set (spans the axes that actually broke things here — density
over four decades, and both feedback extremes): `simple_cluster`,
`f1edge_lowdens`, `f1edge_hidens`, `cal_compact`, `cal_diffuse`, plus the compact probe
probe as the sub-GMC scale that no existing config covers.

A **fast tier** belongs in `pytest` (one short-`stop_t` run per config,
asserting only structural invariants — phases visited, stopping fate, no solver
failures, finite state), with the expensive matched-t trajectory comparison
staying opt-in (`-m stress`) or manual. Sizing note from this batch: a
`stop_t=0.02` arm is ~5 min per config on a 4-core container, so a 6-config
two-arm screen is ~1 hour — too slow for the default suite, fine as a gate you
run before landing a scheme change.

This is scoped as a follow-up ticket. It should not gate the phase-1a fix, and
the phase-1a fix should not quietly grow into building it.
