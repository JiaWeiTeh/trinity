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
| `dictionary-robustness/` | 🟡 PARTIAL — snapshot-machinery audit @ `030b658`, **batteries A–G executed and landed** as 60 green characterization tests (`test_dictionary_stress{,_process}.py`; they pin *current* behavior, defects included). 13 findings probe-verified (F1 boundary-disarmed duplicate guard **confirmed**; F6 flush-retry id shift; F7 load-time clobber of the termination reason; F5 four `save_snapshot` crash modes) + 5 inherited (F14–F18) from the off-trunk 17-finding audit on `fix/audit-dictionary-system` @ `e554316f`, reconciled in `PLAN.md` §1b (unlanded, old-layout, numbers not quotable) + **3 from execution (§1c)**: F19 F5 is reachable from a real `read_param` dict (production survives only because phase 0 fills the arrays before the first save), F20 the crash is in the R² *diagnostic* not the downsampler (cheapest fix, diagnostics-only), F21 F1×F4 ⇒ ~1 phase boundary in 10 duplicates instead of suppressing. O5/O6 resolved (signal reason **is** clobbered by atexit; off-main-thread construction raises). Field scan: fast config clean on I1/I3/I4 but **97/97 lines carry `NaN`**, and F4's 5.0e-4 Myr boundary signature reproduces the old audit exactly. **Owed:** battery H on the `f1edge_*` configs + an all-four-phase run. No `dictionary.py` edit — fixes queued at `PLAN.md` §6 | `README.md`, `PLAN.md` | 2026-08-17 |
| `transition/kappa-3way/` | 🔵 ACTIVE — the three-way f_κ/f_A/f_mix band-entry calibration, measured fresh (cutoff 2026-07-29); 294 arms designed + gated, **not run** | `report.html` | 2026-07-29 |
| `transition/pdv-trigger/` | ⚠️ DEMOTED 2026-07-29 — "could be true, verify before use". History, physics reasoning, literature imprints and HPC tooling stay live; measured values are VERIFY until re-run | `INDEX.md` | 2026-07-29 |
| `transition/cleanroom/` | ✅ concluded — "transition is geometric, not thermal" (live evidence for pdv-trigger) | `FINDINGS.md` | 2026-07-06 |
| `transition/pt4/` | ✅ concluded audits (H1–H5 + R1 shadow) — feed pdv-trigger | `README.md` | 2026-07-06 |
| `rosette-cf/` | ✅ COMPLETE — 72/72 Cf-scan arms ran in-container (exit 0); 72 gzipped raw dicts committed under `data/` for offline reduction; fallback match PROVISIONAL (§11) | `README.md` | 2026-07-14 |
| `phase1a-init/` | ✅ SHIPPED — merged to `main` 2026-08-06 (was branch `hotfix/early-approximations`); sub-GMC-scale early-phase artifact fixed and gated (age-scaled phase-1a segments via `phase1a_segFrac` + `vd=-1e8` override deleted); all gates PASS against the G2 bar adopted 2026-08-05 (`|ΔR2| < 5%` at 1 Myr or end of run + fate unchanged), arms converge to −0.001% at 2 Myr, and the fix runs 16% faster; three goldens on the stock 1a exit state re-baselined. ⚠️ **Two of those three goldens went red again 2026-08-14** when `phii-identity`'s C3c landed — that is a *different* change to the same 1a exit state (C3c removes the `P_HII` channel that was carrying un-ramped pressure past `dt_switchon`), not a regression in this workstream's fix; see the `phii-identity/` row | `PLAN.md`, `data/gate_results.csv` | 2026-08-14 |
| `phase1a-stiffness/` | ✅ SHIPPED — merged to `main` 2026-08-14 (PR #737); **all six batches done, every bar closed.** Is 1a's `RK45`-without-step-bounds segment integrator a latent defect (1b/1c/2 all use LSODA with bounds)? Measured: production is ≥4.3e4× from the stall (worst segment 4 steps / 0.021 s; whole 1a integrator 0.2-0.6 s/run) → **LSODA swap ruled out on economics by the pre-registered rule**; the grind is **stiffness** (`Eb` pinned at 1.6e-6 au on a slow manifold, λ ≈ −1e13, ~7 days/segment), and **`Eb` never reaches 0 — so 1a's `Eb ≤ 0` guard is mis-thresholded as well as out-of-band**. **Batches 3-5: the fix is in and gated, one bar clause open.** A per-segment energy-collapse event at `ENERGY_COLLAPSE_FRAC = 1e-3` of the segment's starting `Eb` (threshold bounded both sides by measurement): the stalling control ends in 22 s with the pre-existing `ENERGY_COLLAPSED` fate; the equivalence screen is **byte-identical on all five configs** (bar required only `simple_cluster`), fates unchanged, worst cost 1.007×; P4's behavioural test verified failing-first; suite 1057/0, `pre-commit` green. P3's mypy clause failed as written (144 vs 137; all +7 of an `attr-defined` idiom the file already has 49 of) and was **ruled ACCEPT** by the maintainer — 144 is the new baseline (`PLAN.md` §2 D5). **Batch 6 answers the motivating question: `dt_switchon` is NOT removable** — ablation still flips the fate on 3 of 5 configs incl. `simple_cluster`, so magic-number #2 stays document-and-pinned, and the "nearly inert" figure is corrected as having come from the only two configs that survive ablation (§2 D6). ⚠️ **Every measurement here predates C3c** (`c43a50e`, landed the same day from PR #738): the byte-identical screen and the D6 ablation both ran while `P_HII` was still injecting the **un-ramped** `Pb` into `P_drive`, so `dt_switchon` reached only the energy equation and never `vd`. The *direction* of D6 survives by argument (post-C3c, ablating the ramp restores the un-ramped pressure to **both** channels, so the runaway can only get stronger) — but the figures are not quotable until re-measured | `PLAN.md`, `data/dt_switchon_removability.csv` | 2026-08-14 |
| `switchon-successor/` | ✅ **CONCLUDED — outcome S0, the constant stays; no `trinity/` behaviour changed** (the only source edit is the D1-D4 rationale block now carried at the constant, plus a stale-claim correction in the pinning test's docstring). **Deletion of `dt_switchon` is settled (NO); this asks about its FORM** — the fixed 1e-3 Myr clock runs 500-87,000× longer than `dt_phase0`, the establishment time the code itself computes. Four candidates incl. a root-cause one (inconsistent initial condition); bar **N3 forbids replacing it with another absolute constant**, and bar **N1 judges successors against Weaver Eq. 20** as a *wind-only limiting reference* (§0.3: radiation supplies 32-60% of the early drive, so TRINITY's solution is not Weaver's — the bar is comparative, not "must match"). **Batches 1-4 done.** The drain is **PdV work** (cooling 0.1-0.8%) and the seed violates Weaver's work partition by 4.85× while satisfying its energy exactly (D1); the physical-clock candidate **failed N0 on 3 of 5**, and since the failures are not ordered by window shortening (87,055× survives, 7× dies) **the whole "better clock" family is retired** (D2); the sustainability cap S2 — a limiter with no free constant — became the **first candidate to clear the fate bar on all five** and self-selected release at ≈3× each run's own `dt_phase0`, but **failed N1 on all five** because capping at "no net energy loss" pins `dEb/dt≈0`, so `Eb` plateaus and `Eb/t` decays by construction, landing ~2× further from the reference than the shipped ramp — **S2 out, and the limiter family with it** (D3). **Batch 4 done (D4):** the handover work rate is algebraic — `PdV/Lmech = 2(v2/v_wind)/(R1/R2)^2`, **`E0` absent** — so reseeding the energy was ruled out before running, and the seed is **identical to six digits on all five configs** (`R1/R2 = 0.869167`, `PdV/Lmech = 2.647425`; the identity is exact to 1e-12 along a whole run). Both measured seed-velocity variants **rescue 2 of the 3 fates full ablation destroys** — so most of the ramp's protection is *velocity*, not geometry, and the pre-registered prediction was half wrong — but still **fail N0 on `f1edge_hidens`, N1 on all five (3.6-6.0x worse) and N2 everywhere**, because starting marginal only delays the runaway (`R1/R2 → 1` as `Eb` dips). **All four candidate families are measured dead** (clock, limiter, seed-energy, seed-velocity) ⇒ **outcome S0: keep the constant, write D1-D4 into the source (Batch 5)**. ⚠️ **Re-opened as a measurement question 2026-08-14 by C3c** (`c43a50e`, PR #738, merged the same day as this workstream): every batch here ran against `P_drive = max(Pb_ramped, P_HII)` with `P_HII == Pb` **un-ramped**, so the ramp was inert in the momentum equation and acted only through `Ed`/`L_leak`. C3c zeroes `P_HII` in the energy phase, so the ramp now throttles `vd` for the first time. **D1 and D4's algebra survives** (`PdV/Lmech = 2(v2/v_wind)/(R1/R2)^2` lives in `Ed`, untouched); **the fate/ablation figures and the N1 Weaver comparisons do not** — S0's conclusion is untouched, its evidence base needs a re-run. `phii-identity/PLAN.md` §3 item 3 named this collision (**D-ramp**) before either branch landed | `PLAN.md`, `data/s4_consistent_seed.csv` | 2026-08-14 |
| `screen/` | 🔵 ACTIVE — multi-config scheme screen (2 refs x N configs, separate processes, matched-t ledger + pass/fail); first run in anger 2026-08-06 (finding-#3 gate) — found and fixed a vacuous fate check (end record lives in `metadata.json[termination]`, not the jsonl tail) | `README.md` | 2026-08-06 |
| `phii-identity/` | 🟡 **C3c SHIPPED 2026-08-14 (`c43a50e`, PR #738) — one branch still open.** Batches 0/1/3/4a + 5 stages 1/1b/2/3 done; D1-D4 answered; independent audit corrections in PLAN.md §9. Identity holds to <=2.9e-16, cap binds on 100% of rows; the coupling is the ionised volume, not the cap. Double-count measured (momentum 2x, transition 1.82x); C1 priced it at <=4.0% dR2 but is wrong-target. C3b rejected (no `Qi` dependence). **C3c — a regime switch that transmits while `P_C3a <= P_conf` and drives at `P_C3a` above — passed its run arm: 5/5 configs, zero numerical distress, NO fate changes, dR2 12.8-20.5% (pre-registered), null passed exactly (`P_HII`=0 on 0/330 implicit rows). The offline screen predicted the self-consistent regime structure to the printed digit on 3/5 configs.** It now runs in production at all six `P_HII` call sites across the four phase runners. **Stage 3 split the verdict: transition passes** the pre-registered wind ladder (`ratio@entry` 0.7144/0.1227/0.0553/0.0235 vs predicted 0.68/0.12/0.054/0.022, 2-7% per rung) while **momentum stays open** — 100% HII-dominated on all four rungs, `P_C3a/P_ram ∝ Lw^−0.33` ⇒ inversion only at an unphysical `Lw ≈ 260`; the registered dichotomy is recorded **mis-specified**, and the suspect is the `R2^−3/2` cavity geometry, not the prefactor. **Landing consequence (2026-08-14):** C3c fixes D-ramp as a side effect (predicted at PLAN.md §3c; the defect is defined at §3 item 3, whose separate "removing the cap drops the drive" sentence stays retracted — that was about C1, not C3c), so the energy phase now drives on the ramped `Pb` alone and two trajectory goldens are red pending re-baseline (`test_run_smoke` R2 0.259560 → 0.256722; `test_phase_boundary` `cool_beta` 0.888197 → 0.878395, both −1.1%). D4 already granted re-baseline authority for `test_phase_boundary` (and `test_betadelta_hybr_stress` / `test_scheme_screen` fixtures) conditional on G3.4's before/after table; **`test_run_smoke` is not on that list and needs its own sign-off**. Red too, for an unrelated and purely structural reason: the site count in `test_mu_audit_drift` (11 → 5 inline; 6 sites consolidated into `get_phii_c3c`) | `README.md`, `PLAN.md` | 2026-08-14 |
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
  the in-band energy-floor event is landed on the branch, byte-identical on all five configs, and
  the mypy clause was ruled ACCEPT — nothing outstanding). Batch 6 closed the magic-number #2 re-open path:
  `dt_switchon` is measured **not removable**, so #2 stays document-and-pinned.
- **`dt_switchon`'s form (not its existence)** → `switchon-successor/PLAN.md` (🔵 Batches 1-4 done,
  ✅ concluded — outcome S0, the constant stays with D1-D4 written at it): can the fixed 1e-3 Myr clock become a scale-free
  physical criterion? Two candidate families are already measured dead (clock, limiter); what is
  left is fixing the seed that made the ramp necessary, or keeping the constant with D1-D3 as its
  justification.
- **HOTPATH §F1-cousin + §F5** → `performance/HOTPATH_PLAN.md`.
- **Leaking luminosities Phase D/F/G + findings #7/#8** → `misc/LEAKING_LUMINOSITIES_SKELETON.md`.
- **Cooling loader refactor PR-1–4** → `cooling/refactor-audit.md`.
- **T_init recommendation #3** (drop the linear L3 patch over `[1e4, T_init]`) → `misc/tinit-sensitivity.md`.
- **`caseB_alpha` stored in AU** (mixed-unit conditioning/correctness item, ownership unclear) →
  `shell-solver/OVERFLOW_FIX_PLAN.md`.
