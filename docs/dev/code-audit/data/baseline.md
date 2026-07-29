# Pre-audit baseline

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

**Status (2026-07-29):** 📘 reference — the "before" state every audit finding is measured against.

**Commit:** `c5b2c01` (merge of PR #732) · **Branch:** `bugfix/code-audit` · **Date:** 2026-07-29

## Test suite

```
python -m pytest -q -p no:randomly --deselect test/test_run_smoke.py
→ 768 passed, 10 deselected, 153 warnings in 196.44s
```

One additional failure, `test_docs_dev_conventions.py::test_readme_lists_every_workstream`,
appeared only *because* this audit created `docs/dev/code-audit/` before registering it in
`docs/dev/README.md`. It is self-inflicted, not pre-existing, and was fixed in the same commit
that created the workstream. **Baseline is green.**

`test_run_smoke.py` (~2.5 min end-to-end) was deselected for time and 10 `stress`-marked tests
were deselected by the default marker expression.

## Lint

| Tool | Result |
|---|---|
| `ruff check trinity --select F821,F811,F823,E9` | **clean** — all checks passed |
| `mypy trinity` | **150 errors in 23 files** (72 files checked) — full output in `mypy_baseline.txt` |

mypy is not gated in CI, so these are pre-existing. They are audit *input*, not findings: some
are typing noise (`no-any-return`, 22), but `misc` (12) and `index` (10) include things like
`run_energy_implicit_phase.py:962: "float" object is not iterable`, which is the shape of a real
defect. Distribution by error code and by file:

```
attr-defined 49 · assignment 27 · no-any-return 22 · arg-type 19 · misc 12
index 10 · operator 5 · union-attr 2 · var-annotated 2 · return-value 2

phase_general/phase_events.py            49
_input/sweep_parser.py                   19
_output/trinity_reader.py                14
phase1_energy/energy_phase_ODEs.py       10
phase2_momentum/run_momentum_phase.py     7
_output/simulation_end.py                 6
cloud_properties/mass_profile.py          5
bubble_structure/bubble_luminosity.py     5
```

Note: `pyproject.toml` sets `python_version = 3.9`, which the installed mypy rejects
(“must be 3.10 or higher”) and silently ignores. That is itself a claim-vs-reality mismatch
worth a verdict, given the package advertises ≥3.9 support.

## Claims ledger

Extracted mechanically by `harness/extract_claims.py` at this commit:

| Ledger | Rows | Notes |
|---|---:|---|
| `claims_prose.csv` | 4560 | 1426 assert a formula or cite a source (82 carry a literature citation, 617 assert units) |
| `claims_literals.csv` | 1644 | numeric literals, flagged when inside arithmetic |
| `claims_guards.csv` | 344 | 131 `except` handlers — **51 of them bare or `Exception`-broad** — plus 213 numeric clamps |
| `claims_params.csv` | 80 | schema keys vs references; 10 `sps_col_*` keys show ≤1 reference (dynamic lookup suspected — needs a verdict, not an assumption) |

## Reference runs

Run in **separate processes** per CLAUDE.md rule 5. Configs span feedback strength × cloud density.

| Config | Purpose | Result |
|---|---|---|
| `param/simple_cluster.param` | energy-driven baseline | see `runs/simple_cluster.stdout` |
| `docs/dev/performance/f1edge_lowdens_himass_hisfe.param` | stiff edge — low density, high SFE | pending |
| `docs/dev/performance/f1edge_hidens_himass_losfe.param` | stiff edge — high density, low SFE | pending |
