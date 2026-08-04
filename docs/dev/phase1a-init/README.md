# phase1a-init — early (phase-1a) initialisation at sub-GMC scale

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

**Status (2026-08-04):** 🔵 actionable — investigation complete; handoff plan for the fix in PLAN.md, not implemented.

Why a TRINITY run at M43 scale (0.15 pc, 2.1e4 yr compact H II region;
`mCloud=300`, `sfe=0.01`, `nCore=8.7e3`) crosses the observed radius ~30x too
early at ~12x the observed velocity — and what that means for the phase-1a
initialisation (`get_y0`, `SEGMENT_DURATION`, the `vd=-1e8` branch).

- `FINDINGS.md` — verdicts per question, numerics-vs-physics split, proposed fix.
- `PLAN.md` — implementation & verification handoff plan (gates, design
  decisions, test matrix) for the agent landing the fix.
- `harness/` — patched runner + param files + CSV extractor + figure script
  (`harness/README.md` has the exact commands).
- `data/` — committed per-run CSVs (`data/README.md` is the manifest).
- `figures/` — trajectory-convergence, momentum-budget, and mass-sweep figures.
