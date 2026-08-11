# docs/dev — workstream status ledger

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

**Status (2026-07-06):** 📘 rebuilt at **workstream level** by the docs/dev housekeeping pass.
Per-doc verdicts live only in each doc's own dated Status line (see `CONVENTIONS.md`); the old
per-doc evidence ledger (verified 2026-06-16/22) was parked in `to-be-removed/`, reviewed, and
deleted by the maintainer (`9459234a`, 2026-07-06).

### Legend

✅ SHIPPED · ⛔ SUPERSEDED · 🔵 ACTIVE/actionable · 🟡 PARTIAL · 📘 REFERENCE · 🧊 FROZEN (archive)

## Workstreams

| Workstream | Verdict | Entry point | Verified |
|---|---|---|:---:|
| `roadmap/` | 🔵 ACTIVE — repo-wide execution queue + solver audit (F1 ✅ fixed; B/C lanes open) + REORG hand-off spec | `README.md` | 2026-07-06 |
| `test-suite/` | 🔵 ACTIVE — test-suite remediation plan (2026-07-06 four-slice audit @ `70f07532`) | `PLAN.md` | 2026-07-06 |
| `transition/kappa-3way/` | 🔵 ACTIVE — the three-way f_κ/f_A/f_mix band-entry calibration, measured fresh (cutoff 2026-07-29); 294 arms designed + gated, **not run** | `report.html` | 2026-07-29 |
| `transition/pdv-trigger/` | ⚠️ DEMOTED 2026-07-29 — "could be true, verify before use". History, physics reasoning, literature imprints and HPC tooling stay live; measured values are VERIFY until re-run | `INDEX.md` | 2026-07-29 |
| `transition/cleanroom/` | ✅ concluded — "transition is geometric, not thermal" (live evidence for pdv-trigger) | `FINDINGS.md` | 2026-07-06 |
| `transition/pt4/` | ✅ concluded audits (H1–H5 + R1 shadow) — feed pdv-trigger | `README.md` | 2026-07-06 |
| `rosette-cf/` | ✅ COMPLETE — 72/72 Cf-scan arms ran in-container (exit 0); 72 gzipped raw dicts committed under `data/` for offline reduction; fallback match PROVISIONAL (§11) | `README.md` | 2026-07-14 |
| `phase1a-init/` | 🔵 ACTIONABLE — sub-GMC-scale early-phase artifact fixed and gated on branch `hotfix/early-approximations` (age-scaled phase-1a segments via `phase1a_segFrac` + `vd=-1e8` override deleted); all gates PASS against the G2 bar adopted 2026-08-05 (`|ΔR2| < 5%` at 1 Myr or end of run + fate unchanged), arms converge to −0.001% at 2 Myr, and the fix runs 16% faster; three goldens on the stock 1a exit state re-baselined | `PLAN.md`, `data/gate_results.csv` | 2026-08-05 |
| `phase1a-stiffness/` | 🔵 ACTIONABLE — **Batches 1-2 done, no `trinity/` line touched.** Is 1a's `RK45`-without-step-bounds segment integrator a latent defect (1b/1c/2 all use LSODA with bounds)? Measured: production is ≥4.3e4× from the stall (worst segment 4 steps / 0.021 s; whole 1a integrator 0.2-0.6 s/run) → **LSODA swap ruled out on economics by the pre-registered rule**; the grind is **stiffness** (`Eb` pinned at 1.6e-6 au on a slow manifold, λ ≈ −1e13, ~7 days/segment), and **`Eb` never reaches 0 — so 1a's `Eb ≤ 0` guard is mis-thresholded as well as out-of-band**. **Batches 3-5: the fix is in and gated, one bar clause open.** A per-segment energy-collapse event at `ENERGY_COLLAPSE_FRAC = 1e-3` of the segment's starting `Eb` (threshold bounded both sides by measurement): the stalling control ends in 22 s with the pre-existing `ENERGY_COLLAPSED` fate; the equivalence screen is **byte-identical on all five configs** (bar required only `simple_cluster`), fates unchanged, worst cost 1.007×; P4's behavioural test verified failing-first; suite 1057/0, `pre-commit` green. **P3's mypy clause fails as written** (144 vs 137 baseline; all +7 of an `attr-defined` idiom the file already has 49 of) — recorded for a maintainer ruling in `PLAN.md` §2 D5, not reinterpreted. **Batch 6 answers the motivating question: `dt_switchon` is NOT removable** — ablation still flips the fate on 3 of 5 configs incl. `simple_cluster`, so magic-number #2 stays document-and-pinned, and the "nearly inert" figure is corrected as having come from the only two configs that survive ablation (§2 D6) | `PLAN.md`, `data/dt_switchon_removability.csv` | 2026-08-06 |
| `switchon-successor/` | 🔵 ACTIONABLE — pre-registered only, no `trinity/` line touched. **Deletion of `dt_switchon` is settled (NO); this asks about its FORM** — the fixed 1e-3 Myr clock runs 500-87,000× longer than `dt_phase0`, the establishment time the code itself computes. Four candidates incl. a root-cause one (inconsistent initial condition); bar **N3 forbids replacing it with another absolute constant**, and bar **N1 judges successors against Weaver Eq. 20** (`Eb/t = (5/11)L_w`: ramp holds ~12%, ablation 154× below). "Keep it and justify it better" is a registered outcome; Batch 1 (diagnose the drain) not run | `PLAN.md` | 2026-08-06 |
| `screen/` | 🔵 ACTIVE — multi-config scheme screen (2 refs x N configs, separate processes, matched-t ledger + pass/fail); first run in anger 2026-08-06 (finding-#3 gate) — found and fixed a vacuous fate check (end record lives in `metadata.json[termination]`, not the jsonl tail) | `README.md` | 2026-08-06 |
| `cooling/` | 🟡 PARTIAL — two side items shipped; loader refactor PR-1–4 pending | `refactor-audit.md` | 2026-06-22 |
| `performance/` | 📘 reference (perf history A→D + F1) · 🟡 HOTPATH §F1-cousin/§F5 open | `BUBBLE_LUMINOSITY_PERFORMANCE.md` | 2026-06-22 |
| `shell-solver/` | 🟡 MIXED — overflow fix ✅ shipped; MIGRATION doc is a 🟠 correction (mxstep diagnosis retracted) | `OVERFLOW_FIX_PLAN.md` | 2026-07-06 |
| `magic-numbers/` | ✅ AUDIT CLOSED (round 2, `SWEEP2_PLAN.md`) — #1/#4 fixed & gated earlier; #3 fixed & gated 2026-08-06 (exact spline derivative replaces the `h=1e-9` FD; screen worst \|ΔR2\| 1.77e-8, fates unchanged); #2 closed as document-and-pin (load-bearing, mechanism corrected to the phase-1a RK45 segment integrator, pinned by `test_dt_switchon_ramp.py`); #5 re-verified, tail owned by the transition workstream | `AUDIT.md` | 2026-08-06 |
| `failed-large-clouds/` | ✅ SHIPPED (2026-06-19) — 1b fate routing superseded 2026-07-01 (now → momentum) | `PLAN.md` | 2026-07-06 |
| `misc/` | 🟡 MIXED — backward-compat ~95% open · tinit rec #3 open · leak D/F/G open · TERMINATION_EVENTS 📘 | per-doc Status lines | 2026-06-22 |
| `cluster/` | 📘 operational guide (on-cluster plotting) | `PLOTTING_WORKFLOW.md` | 2026-06-19 |
| `html-insights/` | 📘 storyline books + verification ledgers (fix-list partially open) | `README.md` | 2026-06-22 |
| `codebase_review/` | 📘 concluded point-in-time audit (52 findings, 2026-06-16) | `../CODEBASE_REVIEW.md` | 2026-06-16 |
| `archive/` | 🧊 FROZEN — betadelta ✅ · bubble ✅/⛔ · n-consistency ✅ · transition trio ⛔ · older audits | `archive/README.md` | 2026-07-06 |

## Open items carried forward

One bullet per open tail, pointing at the doc that owns it — details live there, not here.
(These tails are also sequenced — with gates and execution-tier tags — in `roadmap/PLAN.md`
lane C; keep the two lists reconciled.)

- **β–δ Phase-5 root fix** (mixing-layer cooling/leakage + regime-spanning Eb-peak handoff) —
  now owned by the active `transition/pdv-trigger/` program (`PLAN.md`); historical context in
  `archive/betadelta/HYBR_PLAN.md` Phase 5.
- **Backward-compat cleanup** ~95% un-executed → `misc/backward-compat-audit.md`.
- **Magic numbers — audit closed 2026-08-06** → `magic-numbers/AUDIT.md`. Remaining tails live
  elsewhere: #5's fallback/vestigial-factory cleanup with the transition workstream, and #2's
  only re-open path is now its own workstream (next bullet).
- **Phase-1a segment-integrator stiffness** → `phase1a-stiffness/PLAN.md` (🟡 all six batches done —
  the in-band energy-floor event is landed on the branch and byte-identical on all five configs;
  **open: a ruling on P3's mypy clause**, §2 D5). Batch 6 closed the magic-number #2 re-open path:
  `dt_switchon` is measured **not removable**, so #2 stays document-and-pinned.
- **`dt_switchon`'s form (not its existence)** → `switchon-successor/PLAN.md` (🔵 pre-registered,
  Batch 1 pending): can the fixed 1e-3 Myr clock become a scale-free physical criterion, judged
  against Weaver Eq. 20 rather than mere equivalence?
- **HOTPATH §F1-cousin + §F5** → `performance/HOTPATH_PLAN.md`.
- **Leaking luminosities Phase D/F/G + findings #7/#8** → `misc/LEAKING_LUMINOSITIES_SKELETON.md`.
- **Cooling loader refactor PR-1–4** → `cooling/refactor-audit.md`.
- **T_init recommendation #3** (drop the linear L3 patch over `[1e4, T_init]`) → `misc/tinit-sensitivity.md`.
- **`caseB_alpha` stored in AU** (mixed-unit conditioning/correctness item, ownership unclear) →
  `shell-solver/OVERFLOW_FIX_PLAN.md`.
