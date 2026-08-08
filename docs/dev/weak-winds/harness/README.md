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

## Main study — run it in batches

`batches/` holds one param file per ladder rung (3 clouds each), plus the two
short batch-0 plumbing arms. **`../RUNBOOK.md` is the sequence to follow** — it
gives each batch's command, its pass/fail gate, and the descent rules. Example:

    python run.py docs/dev/weak-winds/harness/batches/batch1_c1p0.param --workers 3 --yes
    python docs/dev/weak-winds/harness/check_batch.py outputs/weak_winds_study/c1p0

Each batch writes to its own `outputs/weak_winds_study/<batch>/` subdirectory: a
fixed (non-swept) knob does not appear in the run-folder name, so batches sharing
one directory would overwrite each other.

`weak_winds_sweep.param` describes the same 15 runs as a single sweep — the design
of record, kept for reference — but running it all at once forfeits the gates:

    python run.py docs/dev/weak-winds/harness/weak_winds_sweep.param --dry-run

## Gate / harvest / figures

    python docs/dev/weak-winds/harness/check_batch.py outputs/weak_winds_study/c0p3
    python docs/dev/weak-winds/harness/check_batch.py --compare RUN_A RUN_B
    python docs/dev/weak-winds/harness/harvest.py outputs/weak_winds_study \
        --out ../data/ladder.csv

`check_batch.py` exits non-zero when a run is missing, non-finite, or stopped for
no recorded reason — that exit code is the gate. `harvest.py` recurses, so one
call collects every batch that has run so far.

## Fast checks (seconds)

    python -m pytest test/test_weak_winds.py            # loader contract + finiteness + IC scalings
    python -m pytest test/test_weak_winds.py -m stress  # end-to-end boot, control vs c=0.03 (~minutes)

Harvested CSVs go to `../data/` with provenance headers
(`docs/dev/transition/PROVENANCE_PROTOCOL.md`); figures to `../figures/`.
