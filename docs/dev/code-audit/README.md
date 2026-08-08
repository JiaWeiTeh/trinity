# code-audit — full correctness audit of `trinity/`

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

**Status (2026-08-08):** 🔵 ACTIVE — **all seven phases pass the completeness checker.**
Findings-only: no source has been fixed by this audit (`git diff origin/main HEAD --
trinity/ test/ run.py param/` is empty). 690 findings, 16 S1 after revision. **The four
open Phase-6 probes are closed** (`data/dynamic_verification.md` §4, §8, §9, §10);
**eight** S1-rated candidates remain untested — see [`HANDOFF.md`](HANDOFF.md).

**Ask the checker, not this line:** `python docs/dev/code-audit/harness/check_completeness.py`
prints per-phase completion and exits non-zero while anything is missing. It is the only
authority on what is done — status prose in these docs goes stale, and once did so in the
direction that matters (Phase 3 was called complete at 5/9 sweeps).

Entry point for the full correctness audit of the `trinity/` package (72 files, 26,359 lines)
requested on `bugfix/code-audit`: no sloppiness, wrong physics, wrong logic, misinterpreted
docstrings, stale results. The package has had substantial AI assistance, so the audit is built
specifically around the defect classes that introduces — see `PLAN.md` §"Why this audit exists".

## Read in this order

| File | What it is |
|---|---|
| [`HANDOFF.md`](HANDOFF.md) | **Start here in a new session** — state, ground rules, next steps, traps |
| [`PLAN.md`](PLAN.md) | Method, slice partition, severity rubric, gates, batch cap, revision protocol |
| [`FINDINGS.md`](FINDINGS.md) | The deliverable — verified findings ranked S1→S4, each with repro + fix outline |
| [`UNVERIFIED.md`](UNVERIFIED.md) | Removed, demoted, or never-tested candidates. **Do not act on these** |
| [`reference/PHYSICS_SPEC.md`](reference/PHYSICS_SPEC.md) | What the code is *supposed* to compute — built without reading the implementation |
| [`reference/STRUCTURE_MAP.md`](reference/STRUCTURE_MAP.md) | Call graph, state-key write/read table, solver inventory — descriptive, no judgment |
| `slices/*.md` | Raw per-agent reports (provenance for every finding) |

## Harness

Both scripts are deterministic and re-runnable from the repo root.

```bash
# Lens A input: every comment and docstring blanked, line numbers preserved exactly
python docs/dev/code-audit/harness/strip_comments.py trinity <outdir>

# The closed checklist: every prose claim, numeric literal, guard, and param
python docs/dev/code-audit/harness/extract_claims.py trinity docs/dev/code-audit/data

# Per-slice Lens A/B/C inputs (stripped code, prose-only, signatures-only)
python docs/dev/code-audit/harness/slices.py <outdir>

# Which phases are actually complete — exits 0 only when all are
python docs/dev/code-audit/harness/check_completeness.py
```

## Data

| File | Rows | What |
|---|---:|---|
| `data/claims_prose.csv` | 4560 | Every comment + docstring, flagged for citation / units / formula content (1426 assert a formula or cite a source) |
| `data/claims_literals.csv` | 1644 | Every numeric literal, flagged when it sits inside arithmetic |
| `data/claims_guards.csv` | 344 | 131 `except` handlers (51 bare or `Exception`-broad) + 213 numeric clamps — every place a physics failure can be swallowed |
| `data/claims_params.csv` | 80 | Schema keys vs actual references in the package |
| `data/baseline.md` | — | Pre-audit `pytest` / lint / reference-run state |
| `data/calibration.md` | — | Phase 0e: detection rate against synthetic injected defects |
