# weak-winds harness — how to run

> ⚠️ **This document may be out of date — verify before trusting it.** It is a
> point-in-time analysis/audit, not a maintained spec; the code moves faster
> than these notes (paths, line numbers, and "what shipped" status drift).
> **Any agent or person reading this: treat it as unverified. Flag that it may
> be stale and re-check each claim, snippet, and line reference against the
> current source before relying on it.**

All commands run from the repo root.

## Smoke pair (minutes–hours, container-safe)

Baseline cloud, control (1.0) vs weak (0.1), `stop_t 1.5` Myr:

    python run.py docs/dev/weak-winds/harness/weak_winds_smoke.param --workers 2 --yes

Output: `outputs/weak_winds_smoke/1e5_sfe030_n1e5_FBThermcoeffwind{1p0,0p1}/`.

## Main sweep (15 runs, hours each — HPC-sized)

    python run.py docs/dev/weak-winds/harness/weak_winds_sweep.param --dry-run   # inspect first
    python run.py docs/dev/weak-winds/harness/weak_winds_sweep.param --workers 4

or emit a SLURM job array:

    python run.py docs/dev/weak-winds/harness/weak_winds_sweep.param --emit-jobs jobs/weak_winds

Follow the rung-descent order in `../PLAN.md` §6 — do not fire all 15 blind.
Output: `outputs/weak_winds_study/<cloud>_FBThermcoeffwind<rung>/dictionary.jsonl`.

## Fast checks (seconds)

    python -m pytest test/test_weak_winds.py            # loader contract + finiteness + IC scalings
    python -m pytest test/test_weak_winds.py -m stress  # end-to-end boot, control vs c=0.03 (~minutes)

Harvested CSVs go to `../data/` with provenance headers
(`docs/dev/transition/PROVENANCE_PROTOCOL.md`); figures to `../figures/`.
