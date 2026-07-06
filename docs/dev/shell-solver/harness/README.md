# docs/dev/shell-solver/harness — shell-ODE migration & overflow-fix harness

> ⚠️ **This document may be out of date — verify before trusting it.** It is a
> point-in-time analysis/audit, not a maintained spec; the code moves faster
> than these notes (paths, line numbers, and "what shipped" status drift).
> **Any agent or person reading this: treat it as unverified. Flag that it may
> be stale and re-check each claim, snippet, and line reference against the
> current source before relying on it.**

Offline harness for the shell-structure solver workstream (`../MIGRATION_PLAN.md`,
`../OVERFLOW_FIX_PLAN.md`). Production is never touched — everything monkeypatches. All
commands run **from the repo root**.

- **Capture/replay matrix** (odeint → solve_ivp candidates): `capture_replay.py` (first probe),
  `capture_replay_variants.py` (per-variant timing + φ-event), driven resumably by
  `bash docs/dev/shell-solver/harness/run_matrix_sweep.sh` → `../data/replay_variants_matrix_<config>.csv`,
  aggregated by `python docs/dev/shell-solver/harness/aggregate_matrix.py` → `../data/master_table.csv`.
- **End-to-end science gate** (full sims, `get_shellODE` variant from `get_shellODE_variants.py`):
  `python docs/dev/shell-solver/harness/run_endtoend.py <param> {baseline|phiguard|clip|cgs} [stop_t]`
  → `outputs/<model>__<idea>/dictionary.jsonl` + a final `ENDTOEND_METRICS` JSON line; matrix driver
  `run_endtoend_matrix.sh` (NB: hardcodes a container path `cd /home/user/trinity`; logs to `/tmp/eteo_*.log`);
  compare/aggregate via `compare_endtoend.py` + `aggregate_endtoend.py` → `../data/eval_endtoend.csv`.
- **One-shot diagnostics**: `verify_overflow.py` (proves the nShell² overflow lives in the discarded
  tail, ~30 s), `diagnose_first_call.py`, `phase_probe.py`, `eval_phi_guard.py`, `eval_terminate.py`
  (→ `../data/eval_*.csv`).

Committed artifacts regenerated from here: the CSVs under `../data/` backing the two plan docs and
`../insights.html` (rebuilt by `python docs/dev/shell-solver/make_insights_html.py`).
