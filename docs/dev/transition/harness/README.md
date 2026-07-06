# docs/dev/transition/harness — implicit→momentum transition-trigger harness

> ⚠️ **This document may be out of date — verify before trusting it.** It is a
> point-in-time analysis/audit, not a maintained spec; the code moves faster
> than these notes (paths, line numbers, and "what shipped" status drift).
> **Any agent or person reading this: treat it as unverified. Flag that it may
> be stale and re-check each claim, snippet, and line reference against the
> current source before relying on it.**

Offline harness/tooling for the **transition-trigger** investigation: when (and
on what criterion) should the implicit energy phase hand off to the momentum
phase? **Not source — regenerable.** Canonical writeups:
`docs/dev/archive/transition/TRIGGER_PLAN.md` (plan), `docs/dev/archive/transition/P0.md`
(P0 results), `docs/dev/archive/transition/pshadow-design.md` (design).

## Files

- `harvest.py` — **P0 harvest**: reads a finished run's `dictionary.jsonl`
  (+ `metadata.json`) and evaluates every candidate trigger (plan F0–F4) on the
  same trajectory, per implicit-phase segment → `docs/dev/data/transition_*.csv`.
  Pure offline read — no production change.
- `psens.py` — **P-sensitivity** on the committed `transition_*.csv` (no new
  runs): ε-sensitivity of the F0 firing epoch, sustained-vs-instantaneous rule,
  and the "does the transition fire before the first WR/SN surge?" structural test.
- `heartbeat.py` — one-shot health check (RUN/DONE/CRASH + last `t_now`) of the
  background hybr runs; used by the polling monitor.
- `{mock_hybr,dense_flat,steep,steep_long}.param` — the four harvested configs.

Data produced: `docs/dev/data/transition_{mock_hybr,dense_flat,steep,steep_long}.csv`.
See `docs/dev/archive/betadelta/diagnostics/README.md` for the run-name glossary and the β–δ phases.
