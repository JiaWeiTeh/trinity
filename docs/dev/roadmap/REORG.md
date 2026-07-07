# REORG — code reorganization spec, packaged for mechanical execution

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

**Status (2026-07-06):** 🔵 actionable — R1–R6 ready for a mechanical session; R2 blocked on
PLAN B5 (classification).

## How to execute an item (rules for the executing session)

1. Read the item's gate BEFORE editing; if the gate no longer matches the code, **stop and
   flag** — don't improvise a new gate.
2. Smallest diff that passes. No drive-by refactors, no style sweeps (CLAUDE.md rules 2–3).
3. "Free win" claims (R1) meet the CLAUDE.md rule-5 bar: **bit-identical** `dictionary.jsonl`
   vs `git show HEAD`, full runs, **separate processes**, matched t, on
   `param/simple_cluster.param` + `docs/dev/performance/f1edge_{lowdens,hidens}*.param`.
4. Full `pytest` + ruff F-rules after; update this doc's item status + PLAN §3 ledger.

## Items

### R1 — consolidate `compute_max_dex_change` (3 verbatim copies → 1)

- **Now:** logic-identical copies in `phase1b_energy_implicit/run_energy_implicit_phase.py`,
  `phase1c_transition/run_transition_phase.py`, `phase2_momentum/run_momentum_phase.py`
  (guarded by `test/test_phase_helper_sync.py`). Per-phase `ADAPTIVE_THRESHOLD_DEX` constants
  (0.05 vs 0.1) are call-site config, NOT part of the function — leave them where they are.
- **Do:** move one copy to `trinity/_functions/operations.py` (matches its numeric-helper
  role); import at the three sites; delete the copies.
- **Gate:** bit-identical bar (rule 3 above) + replace the sync test's AST comparison with an
  import-identity assertion (`phase1b.compute_max_dex_change is operations.compute_max_dex_change`
  for all three) so the guard survives, + full pytest.

### R2 — forces trio: classify, then (maybe) consolidate — ⛔ BLOCKED on PLAN B5

- Do not start until `solver-audit.md` F5 classification exists. If classification says the
  differences are intentional physics: add the why-comments at each divergent hunk and extend
  `test_phase_helper_sync.py` to the documented-common core only. If it finds missed fixes:
  sync those hunks FIRST (each with its own full-run gate on the affected phase), then
  consolidate the common core as in R1.

### R3 — print → logger sweep in `trinity/`

- **Now:** ~32 `print()` calls in phase/solver modules bypass `trinity_*.log`; a crashed run's
  stdout is lost.
- **Do:** replace with `logger.info/debug` (module-local `logger = logging.getLogger(__name__)`
  already exists in most files). Leave CLI `__main__` blocks and `trinity/_output/terminal_prints.py`
  (deliberate terminal UI) alone.
- **Gate:** `grep -rn "^\s*print(" trinity/ --include="*.py"` returns only the deliberate
  terminal-UI/CLI sites (list them in the PR); full pytest; one quickstart run's log file
  contains the formerly-printed lines.

### R4 — production provenance in `metadata.json`

- **Now:** production runs record no commit hash, param hash, or command line;
  `docs/dev/transition/harness/run_stamped.py` proves the pattern but is opt-in research
  tooling. Every debugging session starts by asking "which code produced this run?" —
  unanswerable today.
- **Do:** at run start, write into the metadata run-constants block: git commit (short),
  dirty flag, sha256 of the resolved param file, `sys.argv`, python version. Degrade
  gracefully (`"unknown"`) when git is absent — never fail a run over provenance.
- **Gate:** metadata schema test (fields present; run in a git-less temp dir still completes);
  `show_run` displays them; full pytest.

### R5 — `--debug` / `--log-level` CLI override

- **Now:** verbosity requires editing the `.param` (`log_level`, `log_console`); no fast "re-run
  loud" loop when a run misbehaves.
- **Do:** `run.py` argparse flag that overrides the param's `log_level`/`log_console` for that
  invocation only (schema default untouched — configuration stays in `.param` per CLAUDE.md).
- **Gate:** unit test on the override precedence; quickstart with `--debug` emits DEBUG lines.

### R6 — make the run-diff workflow one obvious command

- **Now:** `tools/compare_outputs.py` diffs two runs' `dictionary.jsonl` into PDF grids, but
  nothing points to it — it's undiscoverable at the moment of need.
- **Do (smallest):** a "Debugging a run" section in the top-level `README.md`/docs listing the
  three commands: `python -m trinity._output.show_run <dir>`, `python tools/compare_outputs.py
  <a> <b>`, and (post-R5) `--debug`. No new code.
- **Gate:** the doc section exists and the commands in it run as written.

## Explicitly rejected (so nobody re-proposes them)

- **Repackaging the phase directories** (`phase1b_energy_implicit/` etc. → `phases/`): pure
  churn — every import, doc, and muscle memory breaks for zero behavioral gain. Layout is fine.
- **A cache/abstraction layer for params**: `DescribedDict` works; PLAN B8 (find the leak)
  must land before anyone touches state handling.
- **Widening lint rules / style sweeps**: standing CLAUDE.md prohibition.
