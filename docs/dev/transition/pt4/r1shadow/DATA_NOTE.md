# R1 transition shadow — data-collection note (pt4)

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
> a committed artifact (a CSV/table under `docs/dev/data/`, or a force-added
> harness/figure in the relevant `docs/dev/<workstream>/` folder) — never left in
> `/tmp` or an untracked `outputs/`. A future visit must be able to reproduce or
> compare against the numbers **without re-running**; record the exact config +
> command that produced each artifact.

**Date:** 2026-06-22. Branch `fix/transition-trigger-problem-pt4` (worktree off the committed
shadow commit `b71cca6`). The R1 shadow is the already-committed, inert (byte-identical) 1b
instrumentation in `trinity/phase1b_energy_implicit/run_energy_implicit_phase.py`
(`evaluate_r1_shadow` + the terminator-site shadow block + sideline `shadow_R1_1b.csv` writer).
**No production code was modified for this data collection** — every run used `--variant V0`
(= NO monkeypatch = the plain shadow-instrumented production code).

## What this collects

The in-code shadow evaluates two R1 transition criteria **every 1b segment** at the live
`cooling_balance` terminator site, and logs the first firing of each (never drives the phase
switch — production still ends only on `cooling_balance`/`reached_tmax`/etc.):

- **blowout**: `R2 > rCloud`
- **Eb-peak**: `betadelta_result.Edot_from_balance <= 0` (the PdV-inclusive net-energy turnover)

Each run writes `dictionary.jsonl` (the production snapshot stream) + the sideline
`shadow_R1_1b.csv` with columns
`t_now,R2,rCloud,R2_over_rCloud,Eb,v2,Pb,Lgain,Lloss,cooling_ratio,edot_balance,blowout_fired,ebpeak_fired`.

## Exact commands

Driver: `docs/dev/transition/pt4/h3_run_variant.py` (V0 = no patch). One sim per process,
`OMP_NUM_THREADS=1`, `timeout 900`, up to 4 in parallel across the 4 cores. Launcher:
`docs/dev/transition/pt4/r1shadow/run_all.sh` (or the inline equivalent used this session).

Per-config invocation (run from the repo/worktree root):

```bash
OMP_NUM_THREADS=1 timeout 900 python docs/dev/transition/pt4/h3_run_variant.py \
    --variant V0 --param <PARAM> --stop_t <STOP_T> \
    --out  docs/dev/transition/pt4/r1shadow/runs/<config> \
    --csv  docs/dev/transition/pt4/r1shadow/runs/<config>_row.csv
# then: cp runs/<config>/shadow_R1_1b.csv  shadow_<config>.csv
```

**stop_t / timeout actually used (final).** First-pass `stop_t` (the task's initial values) made
3 cleanroom runs hit the wall-clock `timeout` *before* the 1b loop terminated — and because the
sideline CSV is written only **after** the 1b loop exits (`run_energy_implicit_phase.py:1262`, after
the break), a timed-out run leaves **no** `shadow_R1_1b.csv` at all (not even a partial one). So for
those 3 the `stop_t` was lowered to just past the (offline-estimated) blowout epoch, so the run
terminates on `reached_tmax` shortly after blowout and writes a complete CSV that captures the
blowout firing. The table below is the **final** (CSV-producing) configuration; the failed first-pass
values are noted in the last column.

| config                | param                                                           | stop_t (final) | timeout | note |
|-----------------------|-----------------------------------------------------------------|--------|---------|------|
| simple_cluster        | docs/dev/transition/cleanroom/configs/simple_cluster.param      | 0.2    | 900 s   | completed first pass (893 s) |
| small_dense_highsfe   | docs/dev/transition/cleanroom/configs/small_dense_highsfe.param | 0.05   | 900 s   | completed first pass (619 s) |
| midrange_pl0          | docs/dev/transition/cleanroom/configs/midrange_pl0.param        | 0.6    | 900 s   | completed first pass (792 s) |
| pl2_steep             | docs/dev/transition/cleanroom/configs/pl2_steep.param           | 0.95   | 900 s   | first pass stop_t=1.2 timed out (no CSV); re-ran at 0.95 (818 s) |
| be_sphere             | docs/dev/transition/cleanroom/configs/be_sphere.param           | 0.95   | 900 s   | first pass stop_t=1.2 timed out (no CSV); re-ran at 0.95 (777 s) |
| large_diffuse_lowsfe  | docs/dev/transition/cleanroom/configs/large_diffuse_lowsfe.param| 3.9    | 2400 s  | first pass stop_t=4.0 timed out; 3.6 finished but pre-blowout (max R2/rCloud=0.994); 3.9 captures blowout (1551 s) |
| fail_repro            | docs/dev/failed-large-clouds/harness/params/fail_repro.param    | 0.02   | 900 s   | completed first pass (87 s); empty 1b shadow (see below) |
| fail_helix            | docs/dev/failed-large-clouds/harness/params/fail_helix.param    | 0.02   | 900 s   | completed first pass (61 s); empty 1b shadow (collapses in 1a) |

`stop_t` is set just past the known blowout epoch so the run reaches blowout *and* terminates (the
shadow CSV is only written after the 1b loop exits). `stop_t` lives in the schema defaults
(`trinity/_input/default.param: stop_t 15`), so the driver's `--stop_t` override applies cleanly
(none of these configs set `stop_t` themselves). Runs were one-per-process, `OMP_NUM_THREADS=1`, up
to 4 in parallel across the 4 cores.

## Files (all committed under docs/dev/transition/pt4/r1shadow/)

- `shadow_<config>.csv` — the in-code 1b shadow sideline, copied from each run's
  `runs/<config>/shadow_R1_1b.csv`.
- `r1_shadow_summary.csv` — one row per config (firing epochs + final state + runtime/status);
  built by `build_summary.py`.
- `run_all.sh` — the parallel run launcher.
- `build_summary.py` — derives `r1_shadow_summary.csv` from the shadow CSVs + `runs/<config>_status.txt`.
- `cross_validate.py` — offline blowout epoch from `dictionary.jsonl` (R2 > rCloud, rCloud from
  `../h2_rcloud_edge.csv`) vs in-code `shadow_<config>.csv` blowout_t.
- `runs/<config>/` — per-run `dictionary.jsonl` (the production snapshot stream; the
  cross-validation source), `shadow_R1_1b.csv`, and `metadata.json`, plus
  `runs/<config>_status.txt` (stop_t/timeout/runtime + completed/timeout) and `runs/<config>.log`
  (driver stdout). The verbose per-run `trinity.log` was removed (regenerable, not needed for any
  committed check).
- `cross_validate_result.txt` — saved output of `cross_validate.py`.
- `GATE_RESULT.txt` + `gate_simple_cluster_shadow_R1_1b.csv` — the pre-existing byte-identical G1
  gate evidence (committed with the shadow code).

## Results — firing epochs

See `r1_shadow_summary.csv`. Summary (filled after the runs completed):

| config               | n_seg | blowout_t (Myr) | blowout R2/rCloud | ebpeak_t | which first | min cooling_ratio | final t / R2 / v2          | status   |
|----------------------|-------|-----------------|-------------------|----------|-------------|-------------------|----------------------------|----------|
| simple_cluster       |   57  | 0.09018         | 1.0224            | —        | blowout     | 0.3242            | 0.20 / 25.83 / 25.83       | completed|
| small_dense_highsfe  |   23  | 0.01166         | 1.0154            | —        | blowout     | 0.2832            | 0.05 / 0.905 / 22.48       | completed|
| midrange_pl0         |   78  | 0.39216         | 1.0054            | —        | blowout     | 0.3689            | 0.60 / 11.39 / 16.78       | completed|
| pl2_steep            |   72  | 0.83979         | 1.0500            | —        | blowout     | 0.4892            | 0.95 / 25.83 / 33.01       | completed|
| be_sphere            |   83  | 0.85612         | 1.0067            | —        | blowout     | 0.4750            | 0.95 / 17.06 / 16.43       | completed|
| large_diffuse_lowsfe |  150  | 3.66000         | 1.0026            | —        | blowout     | 0.4728            | 3.90 / 91.45 / 13.76       | completed|
| fail_repro           |    0  | — (empty 1b)    | —                 | —        | none        | n/a               | — (collapsed)              | completed|
| fail_helix           |    0  | — (empty 1b)    | —                 | —        | none        | n/a               | — (collapsed)              | completed|

**Blowout fires (and fires first) for all 6 normal/cleanroom configs; Eb-peak never fires in-cloud
for any normal config** (their net energy stays positive — monotonic-Eb, consistent with H1/H4).
The full table with all columns is `r1_shadow_summary.csv`.

**Heavy clouds (fail_repro, fail_helix) have EMPTY 1b shadow CSVs (n_seg = 0) — expected:**
- `fail_helix` collapses entirely in **1a** (never enters 1b; 57 snapshots all `energy`-phase).
- `fail_repro` runs 52 segments in 1a — its `Eb` **peaks at t≈0.00153 (1a)** then crashes through
  zero by t≈0.00341, which is *exactly* the 1a→1b boundary. It enters 1b with `Eb<=0` already, so the
  production catastrophic-collapse guard (`Eb<=0` → `energy_collapsed` break at
  `run_energy_implicit_phase.py:1041`) fires **before** the shadow evaluation site (:1117). No shadow
  row is appended; no CSV is written. The Eb-peak for fail_repro is therefore a **1a event** (covered
  separately, per the plan), not a 1b-shadow firing. This is a placement property of the committed
  shadow (after the collapse guard), not a missed detection.

## Cross-validation (in-code 1b vs offline from dictionary.jsonl)

`python docs/dev/transition/pt4/r1shadow/cross_validate.py` (output saved to
`cross_validate_result.txt`) — recompute the blowout epoch offline (first snapshot `R2 > rCloud`,
rCloud from `../h2_rcloud_edge.csv` since it is run-const / blank in snapshots) and compare to the
in-code `shadow_<config>.csv` `blowout_t`. Done for `simple_cluster` + `pl2_steep` (required) and
`be_sphere` + `large_diffuse_lowsfe` (extra):

```
config               rCloud_pc  offline_blowout_t   incode_blowout_t       |dt|     seg_dt  within_1seg
simple_cluster          1.6900   0.09018283765523    0.09018283765523   0.000e+00  1.581e-02  True
pl2_steep              21.3547   0.83978544105555    0.83978544105555   0.000e+00  5.000e-02  True
be_sphere              15.5007   0.85611571458403    0.85611571458403   0.000e+00  5.000e-02  True
large_diffuse_lowsfe   88.0530   3.66000210961084    3.66000210961084   0.000e+00  5.000e-02  True
```

**Exact match (|dt| = 0) for all four** — well within one 1b segment. This is expected: the in-code
shadow reads the same per-segment `R2` and `rCloud` that the offline check reads from the snapshot
stream, so the blowout epochs are identical to the digit (not merely close). No mismatch.

## Sanity checks

1. **Does blowout fire for every normal config?** (expect yes) — **YES.** All 6 cleanroom configs
   fire blowout (`R2 > rCloud`), each at R2/rCloud just above 1.0 (1.003–1.050).
2. **Does the Eb-peak fire in-cloud for any normal config?** (expect NO) — **NO.** `ebpeak_fired` is
   `False` in every row of all 6 cleanroom shadow CSVs; their `edot_balance` stays positive (the net
   PdV-inclusive energy never turns over inside the cloud → monotonic-Eb).
3. **For fail_repro does Eb-peak fire early in 1b?** (expect yes) — the Eb-peak/turnover *is* real and
   *early*, but it is a **1a→boundary event**: Eb peaks in 1a (t≈0.00153) and is already `<=0` by the
   first 1b segment, so the production `Eb<=0` collapse break (:1041) precedes the shadow site (:1117)
   and the 1b shadow CSV is **empty** (n_seg=0). The 1b-only shadow does not — and structurally
   cannot — log it. (Consistent with the task note that the heavy-cloud Eb-peak is a 1a event.)
4. **cooling_ratio never below 0.05?** (consistent with H1) — **PASS.** Minimum cooling_ratio across
   *all* shadow rows of *all* configs is **0.2832** (small_dense_highsfe); per-config minima range
   0.2832–0.4892. None approaches 0.05 — the cooling-balance trigger (`(Lgain−Lloss)/Lgain < 0.05`)
   never fires, which is exactly why production never transitions and R1/blowout is needed.

## Byte-identical / inert confirmation

The committed `GATE_RESULT.txt` already proves G1 (byte-identical `dictionary.jsonl` for
simple_cluster). Re-confirmed this session: (a) `dictionary.jsonl` from these runs carries **no**
shadow-specific keys (`blowout_fired`/`ebpeak_fired`/`R2_over_rCloud`/`edot_balance` absent) — the
shadow output is confined to the sideline CSV; (b) `pytest test/test_r1_shadow.py` → 7 passed. No
production code (`trinity/`, `run.py`) was modified.
