# phase1a-stiffness harness — how to reproduce the runs

> ⚠️ **This document may be out of date — verify before trusting it.** It is a
> point-in-time analysis/audit, not a maintained spec; the code moves faster
> than these notes (paths, line numbers, and "what shipped" status drift).
> **Any agent or person reading this: treat it as unverified. Flag that it may
> be stale and re-check each claim, snippet, and line reference against the
> current source before relying on it.**

All commands run from the repo root. Nothing here modifies `trinity/` — the
instrumentation is installed at runtime, so a measurement can never be mistaken
for a change.

## `seg_stepcount_runner.py` — Batch 1 (per-segment integrator cost)

Wraps `scipy.integrate.solve_ivp` and records one row per **phase-runner** call
(1a's RK45 segment solve; 1b/1c/2's LSODA solves for contrast). Calls from the
bubble solver are ignored — this measures the segment integrator. Configs are
read from `docs/dev/screen/screen.py`'s `CONFIGS`, so the screen set and this
harness cannot drift apart.

```bash
# one production config, ramp active. stop_t = 0.003 Myr covers ALL of phase 1a
# (TFINAL_ENERGY_PHASE = 3e-3), so the recon costs minutes, not the ~5 min/arm a
# stop_t = 0.02 screen arm costs.
python docs/dev/phase1a-stiffness/harness/seg_stepcount_runner.py \
    --config simple_cluster --stop-t 0.003 --workdir <dir>

# the positive control: same config with the dt_switchon R1 ramp ablated, which
# reproduces the stall. ALWAYS wall-cap it from outside — it does not finish.
timeout 900 python docs/dev/phase1a-stiffness/harness/seg_stepcount_runner.py \
    --config f1edge_hidens --stop-t 0.003 --ablate-ramp --workdir <dir>

# merge finished run dirs into the committed ledger (data/seg_stepcount.csv)
python docs/dev/phase1a-stiffness/harness/seg_stepcount_runner.py --reduce <dir> [<dir> ...]
```

Each run writes `seg_calls.csv` into its `--workdir`, plus the usual TRINITY
output under `<workdir>/outputs/screen/`.

**Reading the output.** Two rows per call, `enter` then `exit`. A call that
stalls never writes its `exit` row, and the reducer records that as `STALLED`
rather than dropping it — an `enter` with no `exit` is the finding, and the
line-buffered file survives the external `timeout` kill that ends such a run.
