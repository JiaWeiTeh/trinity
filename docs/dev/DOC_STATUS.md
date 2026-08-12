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
| `phase1a-init/` | 🔵 ACTIONABLE — M43-scale early-phase artifact fixed and gated on branch `hotfix/early-approximations` (age-scaled phase-1a segments via `phase1a_segFrac` + `vd=-1e8` override deleted); all gates PASS against the G2 bar adopted 2026-08-05 (`|ΔR2| < 5%` at 1 Myr or end of run + fate unchanged), arms converge to −0.001% at 2 Myr, and the fix runs 16% faster; three goldens on the stock 1a exit state re-baselined | `PLAN.md`, `data/gate_results.csv` | 2026-08-05 |
| `phii-identity/` | 🔵 ACTIONABLE — consolidates five independent sightings (weak-winds, switchon-successor, phase1a-init, transition/cleanroom, momentum-pdrive; across `feature/low-winds-regime`, `hotfix/other-magic-numbers`, `feature/threeway-pt2`) of `P_HII` == the local confining pressure to 4-10 digits; proved an exact algebraic identity of the `n_IF_Str <= shell_n0` cap (float residual reproduced to <=2 ULP by `harness/roundtrip_ulp.py`). Phases 1a/1b absorb it in `max(Pb, P_HII)`; **transition's `max(Pb, P_HII + P_ram)` never binds (it reduces to `Pb + P_ram` on every step) and momentum's bare `P_HII + P_ram` gives `2*P_ram` — both are ODE right-hand sides**. No `trinity/` change made — the fix effort in `PLAN.md` on branch `bugfix/phii-pt1`: **Batches 0 and 1 PASS**. Cap binds on 100% of rows in every phase (5 configs, 4 decades of nCore); blow-up `raw/shell_n0` maxes at 3.33x so the pre-registered C2a kill bar (1e2) does NOT trip and cap removal is authorised to test. Double-count sized: transition median 1.82x, momentum exactly 2.000x. Third defect found (D-ramp): `P_HII` carries the un-ramped bubble pressure past the `dt_switchon` R1 ramp, up to 3.2x, so a cap fix will drop early driving pressure unless paired. Only `trinity/` change is the bit-identity-gated `n_IF_Str_raw` diagnostic | `README.md`, `PLAN.md` | 2026-08-12 |
| `screen/` | 🔵 ACTIVE — multi-config scheme screen (2 refs x N configs, separate processes, matched-t ledger + pass/fail); harness written and smoke-tested, no screen run in anger yet | `README.md` | 2026-08-05 |
| `cooling/` | 🟡 PARTIAL — two side items shipped; loader refactor PR-1–4 pending | `refactor-audit.md` | 2026-06-22 |
| `performance/` | 📘 reference (perf history A→D + F1) · 🟡 HOTPATH §F1-cousin/§F5 open | `BUBBLE_LUMINOSITY_PERFORMANCE.md` | 2026-06-22 |
| `shell-solver/` | 🟡 MIXED — overflow fix ✅ shipped; MIGRATION doc is a 🟠 correction (mxstep diagnosis retracted) | `OVERFLOW_FIX_PLAN.md` | 2026-07-06 |
| `magic-numbers/` | 🟡 PARTIAL — audit done; #1 and #4 fixed & gated (#4 = `vd=-1e8`, deleted by `phase1a-init` on `hotfix/early-approximations`); #2 measured and found load-bearing — not removable as its recommendation assumed; #3, #5 open | `AUDIT.md` | 2026-08-05 |
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
- **Magic numbers #2, #3, #5** → `magic-numbers/AUDIT.md` (#4 fixed 2026-08-05; #2 now carries
  measured bounds and a ruled-out fix, so its successor starts from evidence, not from scratch).
- **HOTPATH §F1-cousin + §F5** → `performance/HOTPATH_PLAN.md`.
- **Leaking luminosities Phase D/F/G + findings #7/#8** → `misc/LEAKING_LUMINOSITIES_SKELETON.md`.
- **Cooling loader refactor PR-1–4** → `cooling/refactor-audit.md`.
- **T_init recommendation #3** (drop the linear L3 patch over `[1e4, T_init]`) → `misc/tinit-sensitivity.md`.
- **`caseB_alpha` stored in AU** (mixed-unit conditioning/correctness item, ownership unclear) →
  `shell-solver/OVERFLOW_FIX_PLAN.md`.
