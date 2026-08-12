# Magic-numbers round 2 — plan & gates for findings #2, #3, #5

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

**Status (2026-08-06):** ✅ shipped — bars were registered before any `trinity/` edit (`ba35d77`)
and every gate passed. **#2 closed as document-and-pin** (§4 reproduction + mechanism corrected to
the phase-1a RK45 segment integrator; §5 decision applied; `58acfb6`); **#3 fixed & gated**
(`db05694`; §3 results — B3 worst |ΔR2| 1.77e-8 across all 5 configs, fates unchanged);
**#5 recorded** (§6). Owned by `docs/dev/magic-numbers/AUDIT.md` findings **#2, #3, #5**.

## 0. Scope and ordering discipline

Round 2 of the magic-number sweep: work the three findings the audit still lists open. All bars in
this document were written **before** any `trinity/` source was edited on this branch, and per the
`docs/dev/phase1a-init/PLAN.md` §4 precedent they stay on this page verbatim even if later
re-sited — a re-site is recorded *next to* the original, never over it.

- **#3** (`dt = 1e-9` FD step) — measured; a surgical fix candidate is specified in §3 with gates.
- **#2** (`dt_switchon = 1e-3` ramp) — reproduce the two load-bearing E8b claims, instrument the
  stall, then apply the §5 decision rule. "Document and change nothing" is a first-class outcome.
- **#5** (`0.05`/`0.9`) — record current state (it drifted); owned by the transition workstream;
  no code change from here.

## 1. Source re-verification (2026-08-06, HEAD = 731ac50) — what drifted

Re-checked every claim/line reference the round builds on:

- **#2 holds.** The ramp is at `get_bubbleParams.py:368-376` @ `731ac50`, exactly as
  `SWITCHON_BRIEF.md` §1 quotes it. Call sites: only `energy_phase_ODEs.py:224` (RHS) and `:356`
  (`compute_derived_quantities`), both forwarding `t` and `tSF`. **One brief claim corrected:**
  phases 1b/1c/2 do *not* "reach `get_effective_bubble_pressure` through other branches, or pass
  `t=None`" — they never call it at all (`bubble_E2P`/`pRam` directly:
  `get_betadelta.py:329`, `bubble_luminosity.py:228`, momentum via its own snapshot). The
  conclusion (ramp is phase-1a-window-only) survives; the mechanism wording didn't. Corrected in
  the brief, dated. Corollary worth recording: if 1a exits early via `cooling_balance`
  (`run_energy_phase.py:286-297`) before `tSF + 1e-3`, the handoff to 1b drops the ramp
  mid-window — a Pb discontinuity at the boundary. Not worked here; noted for the transition
  workstream.
- **#3 holds.** `update_feedback.py:184-185` @ `731ac50`; `fpdot_total` is
  `scipy.interpolate.interp1d(t, pdot_total, kind='cubic')` (`read_sps.py:354`, built at
  `main.py:148` with the default `ftype='cubic'`). `pdotdot_total`'s only physics consumers are
  the phase-1b beta-delta chain — `a_coeff = 1.5·pdotdot/pdot` (`get_betadelta.py:251`,
  `get_bubbleParams.py:123`, `:177`) and `Ed` (`run_energy_implicit_phase.py:992`) — plus
  snapshot/reader columns.
- **#5 drifted — the audit row is stale.** There is now a registry param
  `phaseSwitch_LlossLgain` (default `0.05`, `registry.py:407`) and both **live** check sites honor
  it: phase 1a (`run_energy_phase.py:290-291`) and phase 1b
  (`run_energy_implicit_phase.py:1249-1254`), each with a hardcoded `0.05` fallback. The
  `phase_events.py` path (`make_cooling_balance_event(threshold=0.05)`, `:319` default + `:497`
  call inside `build_implicit_phase_events`) hardcodes it — but that factory, while returned and
  unpacked at `run_energy_implicit_phase.py:752`, is **never invoked**: vestigial (flag, don't
  delete, CLAUDE.md rule 3). The `0.9` is now the named local `RAM_DOMINANCE_THRESHOLD`
  (`run_transition_phase.py:749`). Disposition: recorded in the audit row; still owned by the
  transition workstream; nothing changed from here.

## 2. Finding #3 — the per-call measurement (done before this plan; no source touched)

Harness `harness/pdotdot_study.py`, artifact `data/pdotdot_percall.csv`
(command in both). On the bundled SB99 table via `param/simple_cluster.param`; every relative
number is `f_mass`-independent because `f_mass` scales `pdot_total` linearly.

| measurement | value | meaning |
|---|---|---|
| `interp1d(kind='cubic')` vs `make_interp_spline(k=3)` | max abs diff **0.0** | the B-spline's `.derivative()` IS the exact derivative of the production interpolant |
| FD `h=1e-9` rel. error vs exact | max **2.8e-2**, p99 8.1e-4, median 6.4e-6 | up to ~3% noise where \|pdotdot\| is small |
| FD abs. error / max\|pdotdot\| | 5.0e-7 | matches the roundoff prediction `eps·|pdot|/2h` — pure noise, no bias |
| h-sweep (max rel err) | 2.0e-5 @ `h=1e-6` → 2.8e-2 @ `1e-9` → 2.7e+1 @ `1e-12` | `1e-9` sits **three decades** below the truncation/roundoff optimum |
| edge windows | **CRASH** at both | `t` within `1e-9` of either table edge passes the function's own range check, then `t±h` raises `interp1d`'s bounds `ValueError` (latent: real runs start at `t0 ≳ 1e-8` and stop before the table end) |
| lever | `a_coeff` abs noise ≤ 1.5e-2 /Myr vs span ±4.4e3 | downstream lever is small — consistent with expecting a near-inert swap |

**Verdict: the smell is real and quantified.** `1e-9` is an uncalibrated *bad* FD step (deep in
the roundoff regime), and the FD construction itself carries a latent edge crash. The audit's
"can sample spline noise across a knot" wording is imprecise — the interpolant is C² so knots are
not the mechanism; float roundoff is.

## 3. Finding #3 — fix candidate and PRE-REGISTERED bars

**Candidate (smallest diff):** in `read_sps.get_interpolation`, additionally build
`fpdotdot_total = scipy.interpolate.make_interp_spline(t, pdot_total, k=3).derivative()`; in
`get_current_sps_feedback`, replace the two-sided FD (and the `dt = 1e-9` constant) with
`sps_f['fpdotdot_total'](t)[()]`. Nothing else changes: all value interpolators stay `interp1d`,
and §2 row 1 shows the derivative source is the *same* cubic. This deletes the magic number
outright instead of re-tuning it, and removes the edge-window crash as a side effect.

**Bars — registered 2026-08-06, before any `trinity/` edit:**

- **B1 (per-call equivalence):** on the bundled table, new `pdotdot_total` equals the analytic
  spline derivative exactly (it *is* it); every other `SPSFeedback` field bit-identical at the
  same `t`.
- **B2 (failing-first tests):** two tests written and confirmed failing on HEAD before the fix:
  (i) `pdotdot_total` at a noise-exposed `t` matches the exact derivative of the table interpolant
  to ≤1e-9 relative (FD fails by §2); (ii) `get_current_sps_feedback` at
  `t = t_min + 5e-10` and `t = t_max − 5e-10` does not raise (FD crashes by §2).
- **B3 (full-run screen — the real gate; CLAUDE.md rule 5):**
  `python docs/dev/screen/screen.py --before <HEAD sha> --after WORKTREE --stop-t 0.02` over the
  standard 5-config set (`simple_cluster`, `f1edge_lowdens`, `f1edge_hidens`, `m43_probe`,
  `gmc_control`), separate processes, matched `t`. Pass requires **(a)** every config completes
  with its stopping fate unchanged — `f1edge_hidens` completing at all is the stiffness gate and
  is checked before any radius bar; **(b)** worst `|ΔR2|` at any compared time ≤ **0.5%** (hard
  bar). *Expectation stated up front:* ≤0.05%; a result in (0.05%, 0.5%] still passes but must be
  investigated and written up before landing (it would mean trajectory-level amplification of a
  ≤5e-7 pdotdot perturbation, worth knowing about given the integrator's FP sensitivity).
- **B4 (suite):** full `pytest` green; `pre-commit run --all-files`; `mypy trinity` no new errors
  vs a HEAD baseline worktree (~150 pre-existing).

**Decision rule:** all four pass → land. B3 fate flip or radius fail → do **not** land the swap;
record the numbers and fall back to document-and-pin (the FD stays, its noise documented).

**RESULTS (2026-08-06) — all four bars PASS; landed as `db05694`:**

- **B1:** new `pdotdot_total` *is* the exact derivative (pinned ≤1e-9 rel across 200 times +
  both edges); all 12 other `SPSFeedback` fields **bit-identical** (hex-exact) between the stock
  tree and the fix across 60 log-spaced times.
- **B2:** both tests written first and confirmed failing on HEAD (`ValueError` at the edge;
  FD noise ≫ 1e-9), green after.
- **B3:** all 5 configs PASS (`data/pdotdot_screen_results.csv`): worst `|ΔR2|` rel diff
  **1.77e-8** (`f1edge_hidens` @2e4 yr) — five decades inside even the 0.05% expectation tier —
  every stopping fate `1 stopping_time` unchanged, `f1edge_hidens` completes. Run as five
  per-config screen invocations after container restarts twice killed the monolithic run; the
  first screen use in anger also surfaced and fixed a vacuous fate check in the harness
  (`eb959c4`, see `docs/dev/screen/README.md`). Arms verified genuinely different: the
  before-arm's FD `pdotdot` is frozen across adjacent snapshots (roundoff quantisation) while
  the after-arm varies smoothly.
- **B4:** full default suite 1009 passed / 0 failed; `pre-commit run --all-files` passes;
  `mypy trinity` 137 errors on both this tree and the `731ac50` baseline worktree — no new.

## 4. Finding #2 — reproduction protocol (the two load-bearing E8b claims)

Both claims re-derived on HEAD (which contains the phase-1a fix — the "on top of the fix, never
against stock" rule is satisfied by construction), ablation via the committed
`docs/dev/phase1a-init/harness/e8b_runner.py` mechanism (`t=None` forwarded; production source
untouched), separate processes, matched `t` by interpolation
(`docs/dev/phase1a-init/harness/matched_t.py`).

- **R1 (compact-probe decay-to-nothing):** compact probe (`harness/params/probe.param`, `stop_t=0.03`), both
  arms. *Reproduction bar:* ablation effect on R2 at the observed age 2.1e4 yr within a factor 3
  of the recorded **−0.0059%** (i.e. in [−0.018%, −0.002%]); same sign, decaying magnitude vs the
  earlier grid times.
- **R2 (hidens stall):** `f1edge_hidens_himass_losfe.param`, `stop_t=0.02`, both arms, ablated arm
  wall-capped at 20 min. *Reproduction bar:* the ablated arm's max simulation `t` reached inside
  its wall cap is < 1% of the ramp-active arm's `t` reached in the same wall time (the recorded
  ratio was ~1e-5: 0.26 yr vs 2e4 yr).
- **R3 (instrument the stall — SWITCHON_BRIEF §7 Q3/§9):** rerun the ablated hidens arm under
  `harness/switchon_probe_runner.py` (ablation + wall-time/state log around each
  `bubble_luminosity.get_bubbleproperties_pure` call + DEBUG logging for the bubble modules) to
  name **what** grinds: the `dMdt` root-find, the backward T-profile ODE + monotonic guard, the
  beta-delta solve, or the segment ODE itself. Deliverable: a committed per-call CSV and one
  paragraph naming the mechanism.

If R1 or R2 fails to reproduce, stop: update the brief/audit with the discrepancy, re-plan from
the measured truth (the docs are then wrong, per their own banner), and say so in the final
report.

**RESULTS (2026-08-06, `data/switchon_repro_ledger.csv`):**

> **SCOPE CORRECTION (2026-08-06, from `docs/dev/phase1a-stiffness/PLAN.md` §2 D6).** R1/R2 below
> ablate the ramp on three configs — the compact probe, `gmc_control` and `f1edge_hidens` — and the
> §5 write-up generalises from them ("worth 0.006%… decays to nothing", with the stiff edge as the
> lone exception). Ablating all **five** screen configs shows that generalisation is drawn from an
> unrepresentative sample: **`simple_cluster` (the default published config) and `f1edge_lowdens`
> also collapse without the ramp** (fate flips to `ENERGY_COLLAPSED` at t = 5.5e-7 and 2.2e-4 Myr).
> The compact probe and `gmc_control` are the only two of five that recover — i.e. exactly the two
> measured here. The numbers below are correct for the configs they were taken on; the conclusion
> "the ramp is nearly inert" is not general, and the ramp is load-bearing far more widely than this
> section implies. Evidence: `docs/dev/phase1a-stiffness/data/dt_switchon_removability.csv`.

- **R1 REPRODUCED, to the digit:** −1.433% @10 yr → −0.108% @1e3 → −0.041% @3e3 →
  **−0.006% @2.1e4 yr** (record: −1.43 → −0.108 → −0.041 → −0.0059), and dv2 +0.006% @2.1e4 yr
  (record +0.0056). Trajectories: `data/switchon_repro_m43_{active,ablated}.csv`.
- **R2 REPRODUCED, bit-for-bit:** ramp-active arm completes 2e4 yr in 5m37s with 127 rows
  (`data/switchon_repro_hidens_active.csv`); the ablated arm produced 4 rows to **0.26 yr** at a
  20-min wall cap — and those 4 rows are **numerically identical to the committed
  `e8b_hidens_noramp_STALLED.csv`** (t, R2, Eb, Pb to every digit). Ratio of t reached,
  1.3e-5 — five decades inside the <1% bar.
- **R3 ANSWERED — and it corrects the recorded mechanism.** The stall is **NOT** the
  bubble-structure solve: every `get_bubbleproperties_pure` call in the ablated run returned in
  1.1-1.3 s (`data/switchon_stall_probe.csv`). What actually happens: with the ramp ablated the
  full early pressure drives PdV/cooling losses that drain `Eb` 180 → 121 → 71 → 29 au across
  segments 1-4, and in segment 5 **phase 1a's segment integrator — `solve_ivp` with hard-coded
  `method='RK45'` at `run_energy_phase.py:309` — stalls taking micro-steps** on the stiffened
  energy ODE. Four grinding-stack samples taken 44 min into that one segment are identical in
  every outer frame (`data/switchon_stall_stacks.txt`); the RHS itself is healthy — the step
  *count* is what diverges. The E8b wording "the bubble-structure ODE stiffens, and the solve
  stops converging" (PLAN.md §8, SWITCHON_BRIEF §3/§4, AUDIT #2 row) is corrected in place,
  dated. The load-bearing conclusion **stands**: the ramp is genuine numerical protection —
  it protects the phase-1a segment integrator, not the bubble solve.

## 5. Finding #2 — PRE-REGISTERED decision rule

Registered before R1-R3 results were known:

- A **successor** to the ramp is pursued only if ALL of:
  - **C1 (stiffness gate, checked first):** candidate `f1edge_hidens` completes `stop_t=0.02` with
    unchanged fate, wall time ≤ 3× the ramp-active arm.
  - **C2 (do-no-harm):** on every §3-B3 config, candidate `|ΔR2|` vs HEAD ≤ **0.1%** (the
    phase-1a eps-convergence noise floor) at every matched grid time ≥ 3e3 yr *and* at the compact probe
    observed age.
  - **C3 (worth it):** R3 identifies a state-based trigger that is *small* (a guard on a measured
    quantity, not a second clock), and the candidate is measurably better than the ramp somewhere
    — earlier switch-off at compact scale with the early-window Pb closer to the unramped
    (physical) value, without violating C1/C2.
- Otherwise → **document-and-pin** (a *result*, per SWITCHON_BRIEF §7 Q5): (i) a comment block at
  the constant recording what it protects, its measured worth (≤0.006-0.017% beyond the early
  window), and the E8b/R2 stall evidence path; (ii) a new pinning test
  (`test/test_dt_switchon_ramp.py`): ramp active exactly on `[tSF, tSF+1e-3]` in the
  energy/implicit branch when `t`/`tSF` are supplied, `Pb` continuous at window close,
  `Pb_ramped ≤ Pb_unramped` inside the window (the leverage direction), and the `t=None` ablation
  contract e8b-style harnesses rely on; (iii) audit row and brief updated with the reproduced
  numbers. The pin is the guard against the next "if inert, delete" reading.
- **Bars for the document-and-pin path:** the new test fails if the ramp branch is deleted
  (verified by ablation in-test), and the full suite stays green; no behavioural diff (the change
  is comments + tests only — a `git diff` over `trinity/` touching nothing but comments, plus B4).

**DECISION (2026-08-06, applying the rule above to the R1-R3 results): document-and-pin.**
C3 fails: R3 found no small state-based trigger — the ramp protects against an *integrator*
failure mode (RK45 grinding as `Eb` collapses), so the honest successor is phase-1a stiffness
handling (a stiff/switching segment integrator, or a terminal in-segment `Eb`-floor event so the
segment ends cleanly instead of grinding), which is integrator work for its own workstream, not a
re-shaping of this constant. **That workstream now exists and is pre-registered:
`docs/dev/phase1a-stiffness/PLAN.md` (2026-08-06) — re-opening #2 is its Batch 6, gated on phase 1a
first surviving an `Eb` collapse on its own.** A merely scale-relative switch-off would deliver the full pressure
earlier at the stiff edge — the direction the stall lives in — to buy ≤0.006-0.017% accuracy on
healthy configs.

> **RE-OPENED FOR REPLACEMENT, THEN CLOSED (2026-08-06) — `docs/dev/switchon-successor/PLAN.md`.**
> The replacement search ran to completion: four pre-registered successors were measured on all
> five configs and **all four failed**, so the constant stays and the argument flagged below was
> ultimately vindicated — but by measurement this time, not by assertion. Two flags on
> the sentence immediately above. It is an **argument, not a measurement**: no scale-relative
> switch-off was ever run. And the "≤0.006-0.017%" it weighs against was measured on the only two
> configs that survive ablation (see the scope correction in §4) — the real stake is that ablation
> flips the stopping fate on **3 of 5** configs, `simple_cluster` included. The **deletion**
> verdict is unchanged and is not being re-litigated; what is under test is the constant's *form*,
> against a physics bar that did not exist when this was written (Weaver Eq. 20,
> `Eb/t = (5/11)L_w`: the ramp holds within ~12% of it, ablation falls 154× below).

Shipped: the in-source rationale block at the constant
(`get_bubbleParams.py`, commit `58acfb6`), the pinning tests (`test/test_dt_switchon_ramp.py`:
ramp shape, window continuity, `t=None` ablation contract, and a deletion guard), the
reproduction + stall evidence in `data/`, and the sibling-doc corrections. The audit row stays
at "measured, documented & pinned — intentionally not changed"; re-open it only alongside
phase-1a integrator work.

## 6. Finding #5 — disposition (record only)

§1 third bullet is the record: param exists and is honored at both live sites; hardcoded `0.05`s
remain as two fallbacks, one factory default, and one never-invoked factory call; `0.9` is a
named constant. The AUDIT.md row is updated to current line refs + this state, keeping ownership
with the transition workstream (`docs/dev/archive/transition/TRIGGER_PLAN.md` lineage). The F0-F5
trigger choice is **not** re-opened, per the audit's own instruction.

## 7. Artifact index (updated as work lands)

| what | where |
|---|---|
| #3 per-call study | `harness/pdotdot_study.py` → `data/pdotdot_percall.csv` |
| #3 screen ledger (B3) | `data/pdotdot_screen_results.csv` (from `docs/dev/screen/screen.py`) |
| #2 reproduction trajectories (R1/R2) | `data/switchon_repro_{m43_active,m43_ablated}.csv`, `data/switchon_repro_hidens_*.csv` |
| #2 matched-t ledger | `data/switchon_repro_ledger.csv` |
| #2 stall instrumentation (R3) | `harness/switchon_probe_runner.py` → `data/switchon_stall_probe.csv` + `data/switchon_stall_stacks.txt` |
| #2 in-source rationale + pins | `trinity/bubble_structure/get_bubbleParams.py` (comment at the constant), `test/test_dt_switchon_ramp.py` |
| E8b originals being reproduced | `docs/dev/phase1a-init/data/` (`e8b_*.csv`, `gate_results.csv` rows `E8b,*`) |
