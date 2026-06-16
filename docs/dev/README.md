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

Internal **plan / audit / diagnostic** write-ups for TRINITY development. This
tree is **not** part of the rendered documentation (Sphinx builds only
`docs/source/`) and is **not** packaged (`pyproject` excludes `docs*`). It is a
working scratchpad of investigations, not a maintained spec — the user-facing
docs live in `docs/source/` and at the project's docs site.

**Banner convention.** Every plan/analysis doc here carries **three** banner
paragraphs right under its H1 — ⚠️ *verify* (staleness), 🔄 *living* (recheck &
refine on every visit), and 💾 *persist diagnostics* (commit results, don't
re-run). The canonical banner text is in `CLAUDE.md`. (This index carries the
first two; the 💾 clause applies to the content docs that hold diagnostics.)

**Naming.** `SCREAMING_SNAKE_CASE.md` = a major **plan / spec / overview**;
`kebab-case.md` = an **audit note / results doc**.

## Layout

| Path | What's in it |
|------|--------------|
| `CODEBASE_REVIEW.md` + `codebase_review/` | The fresh-clone consistency review (this audit) + its 7 per-area section files. |
| `data/` | Committed diagnostic **CSVs** — so a future session reproduces/compares **without** re-running expensive sims. |
| `scratch/` | Diagnostic **harnesses + figures**, grouped by workstream (`betadelta-diagnostics/`, `betadelta-velstruct/`, `transition-trigger/`); each has its own README. The **top-level** `scratch/` (repo root) is separate and git-ignored (local-only). |
| `archive/` | Superseded / historical docs, kept for reference (see `archive/README.md`). |

## Documents by workstream

### β–δ (beta–delta) implicit solver / hybr
- `BETADELTA_HYBR_PLAN.md` — **plan**: beta–delta solver repair (drift cap, metric, bounds, hybr).
- `PHASE0_BETADELTA_BASELINES.md` — **results**: solver baselines across four configs.
- `BETADELTA_PHASE2_ARMS.md` — **results**: Phase 2.3 four-arm shadow experiment + Gate G2.
- `stalling-energy-phase.md` — **study**: stalling energy-driven phase, rising Pb, negative β.
- Data: `data/stalling_*.csv`, `data/hunt_*.csv` · Harnesses/figs: `scratch/betadelta-diagnostics/`, `scratch/betadelta-velstruct/`.

### implicit→momentum transition trigger
- `TRANSITION_TRIGGER_PLAN.md` — **plan**: characterize the transition trigger, then decide.
- `transition-trigger-P0.md` — **results**: P0 harvest (both clocks + candidate divergence).
- `transition-trigger-pshadow-design.md` — **design**: two-criterion (F0 ∨ F4) trigger.
- Data: `data/transition_*.csv` · Harnesses: `scratch/transition-trigger/`.

### bubble solver / integrator
- `bubble-integrator-robustness.md` — the flaky `MonotonicError`, diagnosis + fix.
- `bubble-conduction-convergence.md` — conduction-zone luminosity convergence audit.

### cooling
- `cooling-refactor-audit.md` — cooling-table refactor: audit + implementation plan.

### `n` / pressure consistency
- `n-consistency-audit.md` — `n ≡ n_H` consistency audit against the paper.
- `n-consistency-implementation-plan.md` — line-by-line implementation plan for the above.
- `pressure-terms-audit.md` — pressure terms & the meaning of `n` (`2nkT`, `mu_p`/`mu_n`).

### other audits / notes
- `backward-compat-audit.md` — backward-compatibility & stale-code audit.
- `tinit-sensitivity.md` — is `T_init = 3e4` a relabel-only knob?
- `TERMINATION_EVENTS.md` — overview of the simulation termination events.
- `LEAKING_LUMINOSITIES_SKELETON.md` — phase plan for the `coverFraction` geometry leak.

---

*This index lives alongside the docs it maps; keep it in sync when docs change.*
