# Agent cost ledger — what the audit actually spent

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

**Status (2026-07-30):** 🔵 ACTIVE — measured spend, kept so the next batch can be sized instead of guessed.

## Measured, 2026-07-30 session

| Batch | Agents | Tokens | Mean | Max |
|---|---:|---:|---:|---:|
| Phase 2 lenses | 10 | 1.02M | 102k | 161k |
| Phase 2 reconcilers | 5 | 0.70M | 139k | 181k |
| Phase 3 sweeps | 4 | 1.09M | **272k** | **332k** |
| Phase 4 test-suite audit | 1 | 0.21M | 210k | 210k |
| Phase 5 skeptics wave 1 | 9 | 1.13M | 125k | 195k |
| Phase 5 skeptics wave 2 | 7 | 0.85M | 121k | 170k |
| Phase 5 calibration control | 3 | 0.27M | 89k | 92k |
| **TOTAL** | **39** | **5.27M** | 135k | 332k |

Plus one long baseline sim that ran ~40 min of CPU without reaching 1 % of
`stop_t`, and never finished.

## What the numbers say

**Scope discipline is the cost lever, not agent count.** The cheapest agents
(lenses, 102k) got an explicit file list of 2-8 files. The most expensive
(sweeps, 272k mean) got "read `trinity/**`, ~26k lines". Same model, same
instructions, 2.7x the spend — the difference is entirely how bounded the
mandate was.

**The calibration control was the best value in the session** (89k mean, the
cheapest batch) and it caught a real orchestrator error. Cheap checks on the
*method* beat more coverage of the *subject*.

**Plan for ~150k per agent.** Multiply by the batch size before launching; if the
answer is over ~600k, the batch is too big.
