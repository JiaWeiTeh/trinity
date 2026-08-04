# phase1a-init data manifest

> ⚠️ **This document may be out of date — verify before trusting it.** It is a
> point-in-time analysis/audit, not a maintained spec; the code moves faster
> than these notes (paths, line numbers, and "what shipped" status drift).
> **Any agent or person reading this: treat it as unverified. Flag that it may
> be stale and re-check each claim, snippet, and line reference against the
> current source before relying on it.**

One CSV per run, produced by `../harness/extract_csv.py` from the run's
`dictionary.jsonl` (one row per snapshot; TRINITY AU units except `v2_kms`;
provenance header in each file records the exact config + command). All runs
at commit bb94c78, numpy 1.26.4 / scipy 1.17.1, single core.

| file | what varies vs the M43 probe baseline |
|---|---|
| `m43_probe.csv` | nothing — stock constants (the brief's §2.1 run) |
| `gmc_control.csv` | GMC-scale control (brief §2.2), stock constants |
| `m43_seg1e-5.csv` / `m43_seg3e-6.csv` | SEGMENT_DURATION 3x / 10x shorter |
| `m43_seg1e-4.csv` | SEGMENT_DURATION 3.3x longer (run dies at ~72 yr) |
| `m43_tol1e-8.csv` | solve_ivp rtol/atol 100x tighter |
| `m43_noapprox.csv` | `vd = -1e8` branch ablated |
| `gmc_noapprox.csv` | same ablation on the GMC control |
| `m43_logseg.csv` / `gmc_logseg.csv` | log-spaced segments dt=0.1*t, no hack (fix prototype) |
| `m43_tfinal3e-4.csv` | TFINAL_ENERGY_PHASE 0.1x (earlier 1a->1b handoff) |
| `mass_3e3.csv` ... `mass_3e6.csv` | mCloud sweep at sfe=0.01, nCore=8.7e3, stop_t=0.004 |
| `ncore_3.7e3.csv` / `ncore_2.6e4.csv` | nCore sweep at mCloud=300, stop_t=0.004 |
| `segment1_exit.csv` | derived table: segment-1 exit state per run + analytic prediction |
