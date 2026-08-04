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

**Status (2026-08-04):** 🔵 ready to implement — handoff plan for the log-segment fix; evidence complete in `FINDINGS.md`, nothing implemented in `trinity/` yet.

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
   time scales with object size; at M43 scale (`mCloud=300`, `sfe=0.01`,
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

Consequences at M43 scale: observed radius crossed at 620 yr instead of
1.35e4 yr (~22x), v ~12x observed at crossing. At GMC scale: −10% R2 after
segment 0, decaying transient, asymptote preserved. Full evidence:
`FINDINGS.md` §Verdicts, §Numerics-vs-physics.

## 2. The fix candidate (prototyped, measured — not yet production)

**Log-spaced segments + no override:** `dt_seg = eps·(t_now − tSF)` with
`eps = 0.1`, capped at the stock `SEGMENT_DURATION` (so late-time behaviour
reverts to stock exactly), and the `vd = -1e8` branch removed. Prototype:
`harness/patched_runner.py` (`TRIN_LOGSEG=0.1 TRIN_NO_EARLY_APPROX=1`).
Measured results to reproduce:

| Check | Result | Baseline file |
|---|---|---|
| M43 at observed age (2.1e4 yr) | R2 = 0.196 pc, v2 = 5.1 km/s (obs: 0.153±0.011 pc, 5.0 km/s) | `data/m43_logseg.csv` |
| M43 free-streaming → Weaver | attractor ratio 1.07 by 1.4 yr, 1.00 by 160 yr; **zero solve_ivp failures** | same |
| GMC equivalence at matched t | −3.8% @1e3 yr, −0.95% @3e3 yr, −0.04% @8e4 yr vs stock | `data/gmc_logseg.csv` vs `data/gmc_control.csv` |
| No manufactured momentum | p = 0.28 vs stock's 283 at 410 yr | `data/m43_logseg.csv` vs `data/m43_probe.csv` |

## 3. Design decisions you must make (recommendation first)

1. **Where the schedule lives.** Recommend: a schedule function in the
   phase-1a runner — `dt_seg = min(eps·(t_now − tSF), SEGMENT_DURATION)` —
   with `eps` a new registry param (suggest `SEGMENT_EPS`, default 0.1),
   exposed in `default.param`. Do NOT special-case by cloud mass; the schedule
   is scale-free by construction.
2. **First segment seeding.** `t_now − tSF = 0` at t0 gives dt=0. The
   prototype seeds from t0 itself (dt₀ = eps·t0); re-derive and verify against
   `get_y0`'s t0.
3. **Fate of the override.** Recommend: delete the `vd = -1e8` branch
   entirely (it exists to paper over the segment problem the schedule now
   solves). Independently of that choice, **fix the flag leak**: clear
   `EarlyPhaseApproximation` on every phase-1a exit path, and add it to
   `default.param` if it survives at all.
4. **eps convergence.** Run eps ∈ {0.3, 0.1, 0.03} on the M43 probe + GMC
   control; accept when halving eps moves R2 at the observed age by <1%.
   (eps=0.1 gave ~162 snapshots to 2.4e4 yr at M43 — cost is negligible.)
5. **Do not touch** TFINAL, the 1b DT floors, rtol/atol, or the `-1e8`
   constant's value — all measured second-order or irrelevant (E1).

## 4. Verification ladder (CLAUDE.md rule 5 — this is NOT a free win)

Run in order; do not skip a rung because an earlier one passed. Trinity leaks
module-level global state — **all A/B comparisons in separate processes, at
matched simulation t** (runs truncate at different t).

- **G0 — baselines are already captured.** Committed CSVs above + `git show
  HEAD` values. Re-extract with `harness/extract_csv.py` only for new runs.
- **G1 — revert-equivalence (bit-identical; this sub-claim IS a free win).**
  With the cap making the schedule degenerate (eps large or `SEGMENT_EPS`
  disabled) and the override retained, a `param/simple_cluster.param` run must
  produce a **byte-identical `dictionary.jsonl`** vs stock HEAD. If it
  doesn't, the plumbing changed behaviour — stop and find out why.
- **G2 — full-run equivalence on the stiff edges (the real gate).** Configs:
  `param/simple_cluster.param`,
  `docs/dev/performance/f1edge_lowdens*.param`,
  `docs/dev/performance/f1edge_hidens*.param`, the M43 probe
  (`harness/params/probe.param`), and the GMC control
  (`harness/params/gmc_control.param`). Bars, at matched t in separate
  processes: GMC-scale configs |ΔR2| < 1% for t ≥ 3e3 yr vs stock (the early
  transient is *supposed* to change — that is the fix); M43 probe within 1%
  of `data/m43_logseg.csv` throughout; all runs reach their stock stopping
  fate (no new `VELOCITY_RUNAWAY`/collapse flips); zero solver failures.
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
  guard is sensitive to FP output changes; G1's byte-identity gate will catch
  accidental sensitivity.
- Full-cloud M43 physics (rCloud plausibility at mCloud=300) was validated
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
