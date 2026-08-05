# phase1a-init harness — how to reproduce the runs

> ⚠️ **This document may be out of date — verify before trusting it.** It is a
> point-in-time analysis/audit, not a maintained spec; the code moves faster
> than these notes (paths, line numbers, and "what shipped" status drift).
> **Any agent or person reading this: treat it as unverified. Flag that it may
> be stale and re-check each claim, snippet, and line reference against the
> current source before relying on it.**

All commands run from the repo root. Each run takes ~5-15 min on one core and
writes `<path2output>/dictionary.jsonl` (~1-3 MB). Turn a finished run into a
committed CSV with:

    python docs/dev/phase1a-init/harness/extract_csv.py <run_dir> docs/dev/phase1a-init/data/<name>.csv "<provenance>"

Figures (read only the committed CSVs):

    python docs/dev/phase1a-init/harness/make_figures.py

Compare two finished arms at **matched simulation time** — the only valid A/B
form (CLAUDE.md rule 5); both arms are interpolated onto the requested times,
and `--last` adds the latest time the two arms share:

    python docs/dev/phase1a-init/harness/matched_t.py data/g2_longsimple_stock.csv data/g2_longsimple_fixed.csv --last

## Runs

Stock-constant baselines (plain entry point):

    python run.py docs/dev/phase1a-init/harness/params/probe.param          # -> m43_probe
    python run.py docs/dev/phase1a-init/harness/params/gmc_control.param    # -> gmc_control

Everything else via the patched runner (`patched_runner.py`), which reads env
vars to override phase-1a/1b module constants before starting the simulation
(edit `path2output` in a copy of the param file so runs don't collide):

| data/ CSV | command (env + param) |
|---|---|
| `m43_seg1e-5` | `TRIN_SEG_DUR=1e-5 python .../patched_runner.py .../probe.param` |
| `m43_seg3e-6` | `TRIN_SEG_DUR=3e-6 ...` |
| `m43_seg1e-4` | `TRIN_SEG_DUR=1e-4 ...` (terminates at ~72 yr — that is the finding) |
| `m43_tol1e-8` | `TRIN_RTOL=1e-8 TRIN_ATOL=1e-11 ...` |
| `m43_noapprox` | `TRIN_NO_EARLY_APPROX=1 ...` |
| `gmc_noapprox` | `TRIN_NO_EARLY_APPROX=1 ... gmc_control.param` |
| `m43_logseg` | `TRIN_LOGSEG=0.1 TRIN_NO_EARLY_APPROX=1 ...` (log-spaced-segment prototype) |
| `gmc_logseg` | `TRIN_LOGSEG=0.1 TRIN_NO_EARLY_APPROX=1 ... gmc_control.param` |
| `m43_tfinal3e-4` | `TRIN_TFINAL=3e-4 TRIN_DT_EXIT=1e-5 ...` |
| `mass_3e3..3e6` | stock constants, `probe.param` with `mCloud` swapped and `stop_t 0.004` |
| `ncore_3.7e3, 2.6e4` | stock constants, `probe.param` with `nCore` swapped and `stop_t 0.004` |

`TRIN_LOGSEG=eps` replaces `SEGMENT_DURATION` with an object whose `__radd__`
makes each phase-1a segment `dt = eps * t_now` (log-spaced) — a prototype of
adaptive segmenting with zero production-code changes. Pair it with
`TRIN_NO_EARLY_APPROX=1` (the `vd=-1e8` branch assumes the fixed 30-yr segment).
