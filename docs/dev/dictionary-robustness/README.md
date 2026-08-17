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

**Status (2026-08-17):** 🟡 partial — 13 findings probe-verified against `030b658`, 5 inherited
from an earlier off-trunk audit (`PLAN.md` §1b), and 3 more found by executing the plan (F19–F21,
§1c). **Batteries A–G are landed as green characterization tests.** **Three fixes have shipped**, each
gated bit-identical on `dictionary.jsonl`: F20's empty-curve guard (`PLAN.md` §1d), resolving the
reachable phase-0 crash; F7's loader handler-skip (§1e), making a load side-effect-free; and F6's
transactional flush append (§1f), after re-verification showed that finding was **understated in
consequence and wrong about its trigger**. Everything else stays characterized-not-fixed and queued
for the maintainer (`PLAN.md` §6).
Battery H is scanned on the fast config only — the stiff/edge configs are owed.

Motivating question (maintainer, 2026-08-17): *"Is it true that the duplicate guard is skipped
entirely at every 10-snapshot boundary?"* — **Yes** (finding F1, probe P1): `flush()` clears the
pending buffer, the guard requires a non-empty buffer, so the first `save_snapshot()` after any
flush saves unconditionally — a same-`(t_now, R2)` state straddling the boundary lands as
adjacent duplicate lines in `dictionary.jsonl`. And the guard is load-bearing at phase handoffs
(`run_energy_phase.py:400-419` engineers around it), so record content depends on
`save_count % 10` alignment.

Contents:

- **`PLAN.md`** — the deliverable: verified findings F1–F13 with severities (§1), reconciliation
  with the earlier off-trunk audit incl. 5 inherited findings F14–F18 (§1b — **read first**),
  robustness invariants I1–I9, test batteries A–H for a follow-up session to execute, ground
  rules (characterize, don't fix), and the queued maintainer decisions.
- **`harness/`** — `probe_dictionary.py`, the self-contained reproduction of every §1 finding (no
  simulation, ~seconds), and `scan_field_record.py`, the battery-H invariant scanner over a real
  run's `dictionary.jsonl` (also imported by the test suite, so scanner and artifact never drift);
  see `harness/README.md` for the commands.
- **`data/field_scan.csv`** — committed battery-H results, one row per (config, commit).
- **`data/f20_equivalence.csv`**, **`data/f7_equivalence.csv`**, **`data/f6_equivalence.csv`** —
  one gate-evidence file per shipped fix: the identical `dictionary.jsonl` hash pre- and post-fix,
  the behaviour arms, and the exact config and commands. F6's also records the two corrections that
  re-verification forced on its original write-up.

The tests that pin all of this: `test/test_dictionary_stress.py` (51, in-process) and
`test/test_dictionary_stress_process.py` (14 default + 2 stress, real interpreters). They pin
**current** behavior, defects included — a red test there after a deliberate fix means
"re-baseline the pin", not "regression".

Prior art: an earlier 17-finding audit of the same file sits **off-trunk** on
`fix/audit-dictionary-system` (`git show e554316f:analysis/dictionary-system-audit.md`), written
against the old `src/` layout. It was found only after this workstream's probes ran; `PLAN.md` §1b
reconciles the two ID sets, inherits its five findings that the probes did not cover, and inherits
its field measurements (phase-boundary Δt signature, 112 NaN-bearing lines in one real run) so
battery H compares instead of rediscovering. Its fix set is a proposal to evaluate, never landed.

Highest-severity findings (details + probe output in `PLAN.md` §1): **F7** ✅ *fixed* — merely
*loading* a snapshot used to rewrite the loaded run's `metadata.json` at interpreter exit,
clobbering a recorded crash reason with `'Normal exit / atexit'`; **F6** ✅ *fixed* — a failed
`flush()` used to leave a partial file that production's four swallowed retries then re-appended
once each (re-verification corrected both the scale and the trigger: see `PLAN.md` §1f); **F5** — four `save_snapshot()` crash modes on profile-array states the
code itself can produce; the two empty-array ones are ✅ *fixed* (F20), while the missing-companion
`KeyError` and `reset_keys`' NaN `IndexError` remain open.

Still open and worth knowing before you trust a run's record: **F1/F21**'s flush-alignment lottery
at phase boundaries, **F11**'s bare `NaN` literals (every line of a real run), **O1** — an explicit
save on a *loaded* dict still deletes the source files — **F13**'s loader id shift on a corrupt or
blank line, and the half of F6 that is not a file-consistency problem: a failed flush still *loses*
its buffered snapshots while `main.py` only logs it, so a run can report success with a window of
snapshots missing.
