# dictionary-robustness — snapshot-machinery edge cases & stress campaign

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

**Status (2026-08-17):** 🔵 actionable — 13 edge-case findings verified against `030b658`
(probe harness committed, seconds to re-run); the stress-test batteries in `PLAN.md` are
specified but **not yet executed**.

Motivating question (maintainer, 2026-08-17): *"Is it true that the duplicate guard is skipped
entirely at every 10-snapshot boundary?"* — **Yes** (finding F1, probe P1): `flush()` clears the
pending buffer, the guard requires a non-empty buffer, so the first `save_snapshot()` after any
flush saves unconditionally — a same-`(t_now, R2)` state straddling the boundary lands as
adjacent duplicate lines in `dictionary.jsonl`. And the guard is load-bearing at phase handoffs
(`run_energy_phase.py:400-419` engineers around it), so record content depends on
`save_count % 10` alignment.

Contents:

- **`PLAN.md`** — the deliverable: verified findings F1–F13 with severities, robustness
  invariants I1–I9, test batteries A–H for a follow-up session to execute, ground rules
  (characterize, don't fix), and the queued maintainer decisions.
- **`harness/`** — `probe_dictionary.py`, the self-contained reproduction of every finding
  (no simulation, ~seconds); see `harness/README.md` for the command.

Highest-severity findings (details + probe output in `PLAN.md` §1): **F7** — merely *loading* a
snapshot rewrites the loaded run's `metadata.json` at interpreter exit, clobbering a recorded
crash reason with `'Normal exit / atexit'`; **F6** — a non-serializable value makes `flush()`
fail mid-append, and a retry duplicates already-written lines, silently shifting every later
snapshot id; **F5** — four `save_snapshot()` crash modes on profile-array states the code itself
can produce (including `reset_keys`' NaN default).
