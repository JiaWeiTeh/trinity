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

**Banner convention.** Every plan/analysis doc carries **three** banner paragraphs
right under its H1 — ⚠️ *verify* (staleness), 🔄 *living* (recheck & refine on every
visit), 💾 *persist diagnostics* (commit results under `docs/dev/`, don't re-run).
Canonical banner text is in `CLAUDE.md`. (This index carries the first two.)

**Naming.** Each workstream folder is self-contained (writeups **+** its harnesses
and figures); the folder name gives the context, so filenames inside drop the
workstream prefix (`betadelta/HYBR_PLAN.md`, not `BETADELTA_HYBR_PLAN.md`).
`SCREAMING_SNAKE.md` = a major plan/spec/overview, `kebab-case.md` = an audit/results note.

## Layout

```
docs/dev/
├── CODEBASE_REVIEW.md + codebase_review/   fresh-clone consistency review (this audit)
├── data/                                   committed diagnostic CSVs (provenance for writeups)
├── archive/                                superseded / historical docs
├── betadelta/        β–δ implicit solver / hybr
├── transition/       implicit→momentum transition trigger
├── bubble/           bubble solver / integrator
├── cooling/          cooling tables
├── n-consistency/    the meaning of `n` (n ≡ n_H) & pressure terms
└── misc/             standalone audits / notes
```

`data/` and the top-level `scratch/` (repo root, git-ignored, local-only) are **not**
inside the workstream folders; committed diagnostics live under each workstream.

## Workstreams

### `betadelta/` — β–δ implicit-phase solver & hybr
- `HYBR_PLAN.md` — **plan**: beta–delta solver repair (drift cap, metric, bounds, hybr).
- `PHASE0_BASELINES.md` — **results**: solver baselines across four configs.
- `PHASE2_ARMS.md` — **results**: Phase 2.3 four-arm shadow experiment + Gate G2.
- `stalling-energy-phase.md` — **study**: stalling energy-driven phase, rising Pb, negative β.
- `diagnostics/` — harnesses + figures (rootmaps, arms, cage, stalling, negvel) + their README.
- `velstruct/` — the velocity-structure ("Problem 2") hunt harness + README.
- Data: `data/stalling_*.csv`, `data/hunt_*.csv`.

### `transition/` — implicit→momentum transition trigger
- `TRIGGER_PLAN.md` — **plan**: characterize the transition trigger, then decide.
- `P0.md` — **results**: P0 harvest (both clocks + candidate divergence).
- `pshadow-design.md` — **design**: two-criterion (F0 ∨ F4) trigger.
- `harness/` — offline harvest / P-sensitivity harnesses + README.
- Data: `data/transition_*.csv`.

### `bubble/` — bubble solver / integrator
- `integrator-robustness.md` — the flaky `MonotonicError`, diagnosis + fix.
- `conduction-convergence.md` — conduction-zone luminosity convergence audit.

### `cooling/`
- `refactor-audit.md` — cooling-table refactor: audit + implementation plan.

### `n-consistency/` — the meaning of `n` & pressure
- `audit.md` — `n ≡ n_H` consistency audit against the paper.
- `implementation-plan.md` — line-by-line plan for the above.
- `pressure-terms-audit.md` — pressure terms & the meaning of `n` (`2nkT`, `mu_p`/`mu_n`).

### `misc/` — standalone audits / notes
- `backward-compat-audit.md` — backward-compatibility & stale-code audit.
- `tinit-sensitivity.md` — is `T_init = 3e4` a relabel-only knob?
- `TERMINATION_EVENTS.md` — overview of the simulation termination events.
- `LEAKING_LUMINOSITIES_SKELETON.md` — phase plan for the `coverFraction` geometry leak.

---

*This index lives alongside the docs it maps; keep it in sync when docs change.*
