# docs/dev conventions — how to add and maintain docs here

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

**Status (2026-07-06):** 📘 current — created by the docs/dev housekeeping pass.

One page: everything a new doc or workstream under `docs/dev/` must follow. Canonical banner
text lives in `CLAUDE.md` §"`docs/dev/` plan & audit docs are unverified" — copy it verbatim,
never fork it. `test/test_docs_dev_conventions.py` enforces the mechanical parts.

## Banners

- **Active** plan/audit/write-up docs: all **four** banners (⚠️ 🔄 💾 🔗), verbatim from
  CLAUDE.md, right under the H1.
- **Archived** docs (`archive/`): keep ⚠️, replace 🔄/💾/🔗 with the single 🧊 *frozen* banner
  (text in CLAUDE.md, same section).
- **Exempt** (need only ⚠️): pure how-to-run harness READMEs, data-manifest notes, and
  machine-generated files (e.g. `MANIFEST.md`) — the list is pinned as `EXEMPT` in
  `test/test_docs_dev_conventions.py`. Anything else missing a banner fails that test.
- `to-be-removed/` is a staging area for deletion candidates and is not checked.

## Status line

Directly under the banners: `**Status (YYYY-MM-DD):** <emoji> <one-sentence verdict>`.
Emojis: ✅ shipped · ⛔ superseded · 🔵 actionable · 🟡 partial · 📘 reference · 🧊 frozen.
The doc's own Status line is the **only** per-doc source of truth. `DOC_STATUS.md` tracks whole
workstreams (one row each); `README.md` is navigation only. Never record a doc's status in a
third place.

## Workstream folder template

`<workstream>/README.md` (entry point) + writeups + `harness/` (with a README giving the exact
command, run from the repo root, and where output lands) + `data/` (committed artifacts with
provenance headers) + `figures/`. A workstream that grows past ~5 docs gets a hub/INDEX doc
(model: `transition/pdv-trigger/INDEX.md`).

## Naming

`ALL_CAPS_SNAKE.md` = major plan/spec/overview · `kebab-case.md` = audit/results note.
The folder gives the context, so filenames drop the workstream prefix
(`shell-solver/MIGRATION_PLAN.md`, not `shell-solver/SHELL_SOLVER_MIGRATION_PLAN.md`).

## Citations

- Another doc: cite by **repo-relative path** (`docs/dev/transition/cleanroom/FINDINGS.md`), never a bare basename.
- Code: pin an **absolute anchor** — a commit SHA (e.g. `get_betadelta.py:583` @ `4d8a2631`) —
  line numbers alone drift within days.

## Provenance (any data-bearing workstream)

The standard is `docs/dev/transition/PROVENANCE_PROTOCOL.md`: every run stamped with commit +
command + param hash; harvested CSVs carry a provenance header. Templates worth copying, all in
`docs/dev/transition/pdv-trigger/`:

- `REPRODUCE.md` — claim → param file → command → committed artifact table (with cost tags);
- `CONTAMINATION.md` — per-artifact quotability register (⛔/⚠️/✅ before quoting any number);
- `runs/README.md` — the standard config matrix + measurement protocol for knob claims;
- `MANIFEST.md` + `make_manifest.py` — artifact recency ledger, regenerated with each artifact change.

## New-workstream checklist

1. `<ws>/README.md` with banners + Status line;
2. one line in `docs/dev/README.md` §Layout;
3. one row in `docs/dev/DOC_STATUS.md`;
4. `pytest test/test_docs_dev_conventions.py` passes.

## Retiring work

Shipped/superseded → move the docs to `archive/<ws>/`, swap 🔄/💾/🔗 → 🧊, update the Status
line, fix inbound links (grep for the old path). Deletion candidates → move to
`docs/dev/to-be-removed/` with a line in its README saying why; the maintainer reviews and
deletes personally — nothing is deleted directly.
