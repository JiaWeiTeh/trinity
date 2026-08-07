# phase1a-stiffness — is phase 1a's segment integrator a latent defect?

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

**Status (2026-08-06):** 🔵 actionable — **Batch 1 (reconnaissance) is done; no `trinity/` line has
been touched, and "change nothing" remains a pre-registered outcome.** Measured: production is
≥4.3e4× away from the stall in wall time (worst segment 4 steps / 0.021 s; the whole phase-1a
segment integrator costs 0.2-0.6 s per run) while the ablated control never returns from one
call. That rules out the LSODA swap by the pre-registered rule and leaves the in-band `Eb`-floor
event as the only live candidate. Next: Batch 2 (stiffness vs singularity). `PLAN.md` §2 holds
the numbers.

## The question

Phase 1a integrates its segments with `solve_ivp(method='RK45')` and **no `min_step`/`max_step`**
(`run_energy_phase.py:309-318` @ `adfc23f`), while phases 1b, 1c and 2 all use **LSODA** —
stiff/non-stiff auto-switching — *with* both step bounds. When the `dt_switchon` ramp is ablated
at `nCore = 1e6`, `Eb` collapses inside a segment and that RK45 call grinds for > 44 minutes on
one segment; the `Eb ≤ 0` guard that would end the run cleanly lives *between* segments, so it
never gets the chance. Measured: `docs/dev/magic-numbers/data/switchon_stall_probe.csv` and
`switchon_stall_stacks.txt`.

So: latent defect, or a curiosity of a configuration production never runs? That is what this
workstream measures before proposing anything.

## Where to start

1. `PLAN.md` §1 — what is actually in the source (verified, with the asymmetry vs 1b/1c/2).
2. `PLAN.md` §2 — the load-bearing unknown: does this bite with the ramp *active*?
3. `PLAN.md` §3 §5 — the pre-registered bars and the decision rule, including the
   stiffness-vs-singularity trap that decides which remedy is even correct.
4. `PLAN.md` §6 — the batches. Batch 1 is done (its D1 answer is in §2); **Batch 2** is next.

## Why it exists

Spun out of `docs/dev/magic-numbers/SWEEP2_PLAN.md` §4-5, which closed magic-number audit #2
(`dt_switchon = 1e-3`) as document-and-pin: the ramp is load-bearing, but what it protects turned
out to be this integrator rather than the bubble-structure solve as the earlier write-ups
recorded. Removing that constant is only conceivable once phase 1a can survive an `Eb` collapse
on its own — which is `PLAN.md` Batch 6, deliberately last.
