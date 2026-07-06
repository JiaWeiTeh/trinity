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
per-doc evidence ledger (verified 2026-06-16/22) is parked in
`to-be-removed/DOC_STATUS_per-doc-ledger_2026-06-22.md` pending the maintainer's review.

### Legend

✅ SHIPPED · ⛔ SUPERSEDED · 🔵 ACTIVE/actionable · 🟡 PARTIAL · 📘 REFERENCE · 🧊 FROZEN (archive)

## Workstreams

| Workstream | Verdict | Entry point | Verified |
|---|---|---|:---:|
| `transition/pdv-trigger/` | 🔵 ACTIVE — PdV/f_κ mechanism + θ calibration | `INDEX.md` | 2026-07-06 |
| `transition/cleanroom/` | ✅ concluded — "transition is geometric, not thermal" (live evidence for pdv-trigger) | `FINDINGS.md` | 2026-07-06 |
| `transition/pt4/` | ✅ concluded audits (H1–H5 + R1 shadow) — feed pdv-trigger | `README.md` | 2026-07-06 |
| `cooling/` | 🟡 PARTIAL — two side items shipped; loader refactor PR-1–4 pending | `refactor-audit.md` | 2026-06-22 |
| `performance/` | 📘 reference (perf history A→D + F1) · 🟡 HOTPATH §F1-cousin/§F5 open | `BUBBLE_LUMINOSITY_PERFORMANCE.md` | 2026-06-22 |
| `shell-solver/` | 🟡 MIXED — overflow fix ✅ shipped; MIGRATION doc is a 🟠 correction (mxstep diagnosis retracted) | `OVERFLOW_FIX_PLAN.md` | 2026-07-06 |
| `magic-numbers/` | 🟡 PARTIAL — audit done; #1 fixed & gated, #2–#5 open | `AUDIT.md` | 2026-06-22 |
| `failed-large-clouds/` | ✅ SHIPPED (2026-06-19) — 1b fate routing superseded 2026-07-01 (now → momentum) | `PLAN.md` | 2026-07-06 |
| `misc/` | 🟡 MIXED — backward-compat ~95% open · tinit rec #3 open · leak D/F/G open · TERMINATION_EVENTS 📘 | per-doc Status lines | 2026-06-22 |
| `cluster/` | 📘 operational guide (on-cluster plotting) | `PLOTTING_WORKFLOW.md` | 2026-06-19 |
| `html-insights/` | 📘 storyline books + verification ledgers (fix-list partially open) | `README.md` | 2026-06-22 |
| `codebase_review/` | 📘 concluded point-in-time audit (52 findings, 2026-06-16) | `../CODEBASE_REVIEW.md` | 2026-06-16 |
| `archive/` | 🧊 FROZEN — betadelta ✅ · bubble ✅/⛔ · n-consistency ✅ · transition trio ⛔ · older audits | `archive/README.md` | 2026-07-06 |

## Open items carried forward

One bullet per open tail, pointing at the doc that owns it — details live there, not here.

- **β–δ Phase-5 root fix** (mixing-layer cooling/leakage + regime-spanning Eb-peak handoff) —
  now owned by the active `transition/pdv-trigger/` program (`PLAN.md`); historical context in
  `archive/betadelta/HYBR_PLAN.md` Phase 5.
- **Backward-compat cleanup** ~95% un-executed → `misc/backward-compat-audit.md`.
- **Magic numbers #2–#5** → `magic-numbers/AUDIT.md`.
- **HOTPATH §F1-cousin + §F5** → `performance/HOTPATH_PLAN.md`.
- **Leaking luminosities Phase D/F/G + findings #7/#8** → `misc/LEAKING_LUMINOSITIES_SKELETON.md`.
- **Cooling loader refactor PR-1–4** → `cooling/refactor-audit.md`.
- **T_init recommendation #3** (drop the linear L3 patch over `[1e4, T_init]`) → `misc/tinit-sensitivity.md`.
- **`caseB_alpha` stored in AU** (mixed-unit conditioning/correctness item, ownership unclear) →
  `shell-solver/OVERFLOW_FIX_PLAN.md`.
