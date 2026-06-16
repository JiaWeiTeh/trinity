# docs/dev — internal development notes

> ⚠️ **This document may be out of date — verify before trusting it.** It is a
> point-in-time analysis/audit, not a maintained spec; the code moves faster
> than these notes (paths, line numbers, and "what shipped" status drift).
> **Any agent or person reading this: treat it as unverified. Flag that it may
> be stale and re-check each claim, snippet, and line reference against the
> current source before relying on it.**
>
> 🔄 **Living index — recheck and refine on every visit.** This map drifts as
> docs are added, superseded, or moved. Anyone who adds/removes a doc under
> `docs/dev/` should update this index (and move retired docs to `archive/`).

## What this is

Internal **plan / audit / diagnostic** write-ups for TRINITY development, grouped
one folder per **workstream**. This tree is **not** part of the rendered
documentation (Sphinx builds only `docs/source/`) and is **not** packaged
(`pyproject` excludes `docs*`) — it is a working record of investigations, not a
maintained spec. The user-facing docs live in `docs/source/` and the project site.

**Status-driven layout.** **Active** workstreams (work still pending) sit at the
top; workstreams whose work has **shipped or been superseded** are moved under
`archive/`. The per-doc verdicts (with `trinity/…:line` evidence) live in
[`DOC_STATUS.md`](DOC_STATUS.md) — start there to see what's done vs. live.

**Banner convention.** Every plan/analysis doc carries **three** banner paragraphs
right under its H1 — ⚠️ *verify* (staleness), 🔄 *living* (recheck & refine on every
visit), 💾 *persist diagnostics* (commit results under `docs/dev/`, don't re-run).
Canonical banner text is in `CLAUDE.md`. (This index carries the first two.) Each
doc's "About this document" block also carries a verified **Status** line.

**Naming.** Each workstream folder is self-contained (writeups **+** its harnesses
and figures); the folder name gives the context, so filenames inside drop the
workstream prefix (`transition/TRIGGER_PLAN.md`, not `TRANSITION_TRIGGER_PLAN.md`).
`SCREAMING_SNAKE.md` = a major plan/spec/overview, `kebab-case.md` = an audit/results note.

## Layout

```
docs/dev/
├── CODEBASE_REVIEW.md + codebase_review/   fresh-clone consistency review (this audit)
├── DOC_STATUS.md                           per-doc verified status (shipped / actionable / superseded)
├── data/                                   committed diagnostic CSVs (provenance for writeups)
├── transition/   implicit→momentum transition trigger   (🔵 ACTIVE)
├── cooling/      cooling-table refactor                 (🔵 ACTIVE)
├── misc/         standalone audits / notes              (🟡 mixed)
└── archive/      shipped / superseded / historical:
    ├── betadelta/        β–δ implicit solver / hybr      (✅ shipped)
    ├── bubble/           bubble solver / integrator      (✅/⛔ done)
    ├── n-consistency/    n ≡ n_H & pressure terms        (✅/⛔ done)
    └── restructure-audit.md, sb99-refactor-audit.md, …   (older history)
```

The top-level `scratch/` (repo root) is separate, git-ignored, local-only.

## Active workstreams

### `transition/` — implicit→momentum transition trigger
- `TRIGGER_PLAN.md` — **plan** (🔵 actionable): characterize the transition trigger, then decide.
- `P0.md` — **results** (✅): P0 harvest (both clocks + candidate divergence).
- `pshadow-design.md` — **design** (🟡 partial): two-criterion (F0 ∨ F4) trigger; P-shadow (log-only) shipped, P-promote pending sign-off.
- `harness/` — offline harvest / P-sensitivity harnesses + README.
- Data: `data/transition_*.csv`.

### `cooling/` — cooling-table refactor
- `refactor-audit.md` — **plan** (🔵 actionable): decouple the loaders from hardcoded SB99/OPIATE/CLOUDY. Nothing shipped yet.

### `misc/` — standalone audits / notes
- `backward-compat-audit.md` — (🔵 actionable) backward-compat / stale-code cleanup; ~95% pending.
- `tinit-sensitivity.md` — (🟡 partial) is `T_init = 3e4` a relabel-only knob? Study done; one open rec.
- `TERMINATION_EVENTS.md` — (📘 reference) current per-phase termination-events reference.
- `LEAKING_LUMINOSITIES_SKELETON.md` — (🟡 partial) `coverFraction` leak; A–C shipped, D/F/G open.

## Archived (shipped / superseded — see `DOC_STATUS.md`)

Moved under `archive/` once their work landed; kept as historical record (harnesses + data move with them).

- `archive/betadelta/` — β–δ solver repair: `HYBR_PLAN`, `PHASE0_BASELINES`, `PHASE2_ARMS`, `stalling-energy-phase` + `diagnostics/`, `velstruct/`. ✅ shipped (one open tail: the Phase-4 default flip to `hybr`).
- `archive/bubble/` — `integrator-robustness` (⛔ superseded by the `solve_ivp` migration), `conduction-convergence` (✅ shipped).
- `archive/n-consistency/` — `audit`, `implementation-plan` (✅ shipped, pinned by `test_mu_audit_drift.py`), `pressure-terms-audit` (⛔ superseded).
- `archive/restructure-audit.md`, `archive/sb99-refactor-audit.md` — older completed restructures.

---

*This index lives alongside the docs it maps; keep it in sync when docs change.*
