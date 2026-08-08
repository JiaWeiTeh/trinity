# weak-winds — how much do stellar winds matter?

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

**Status (2026-08-08):** 🔵 actionable — harness + tests shipped and green; 15-run science sweep
designed, not yet executed.

The `FB_thermCoeffWind` sensitivity study: step the wind-thermalization knob down a
log ladder (1.0 → 0.01) across three cloud regimes and measure what winds actually
do to shell trajectories, force budgets, phase chronology, and fates. Motivated by a
collaborator's "can we switch winds off?" — a strict off switch is not runnable today
(0/0 in `v_mech_total` pre-SN; wind-built initial conditions), so this study brackets
the answer from the runnable side. See `PLAN.md` §1 for that analysis.

- **`PLAN.md`** — background, the knob's exact scalings, hypotheses H0–H4, study
  design, execution order, risks, implications. Start here.
- **`FINDINGS.md`** — smoke results so far; sweep results land here.
- **`harness/`** — param files + commands (`harness/README.md`).
- `test/test_weak_winds.py` (repo suite) — the loader scaling contract, feedback
  finiteness on the ladder, free-streaming IC scalings, end-to-end boot stress test.
