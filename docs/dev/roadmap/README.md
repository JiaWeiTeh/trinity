# roadmap — the master execution queue (START HERE)

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

**Status (2026-07-06):** 🔵 ACTIVE — created by the 2026-07-06 solver-audit session.

## What this workstream is

The repo-wide **execution queue**: every open work item, sequenced, each with a **why**, a
**pass/fail gate**, and an **execution-tier tag** so any future session — regardless of which
model is driving — picks the top item and executes against checks instead of re-deriving the
thinking. Per-topic detail stays in the owning workstream doc; this folder holds only the queue,
the code critique that seeded it, and the reorganization spec.

## Read in this order

1. **`PLAN.md`** — the sequenced queue + gate table (the hub).
2. **`solver-audit.md`** — the 2026-07-06 deep critique of the solver core: ranked findings,
   each with a failure scenario and its frozen check.
3. **`REORG.md`** — the code-reorganization spec: mechanical items packaged for hand-off,
   each with its equivalence gate.

Related, not siblings: `docs/dev/DOC_STATUS.md` (workstream ledger),
`docs/dev/CODEBASE_REVIEW.md` (the 2026-06-16 52-finding audit this builds on),
`docs/dev/to-be-removed/README.md` (deletion candidates awaiting maintainer review).
