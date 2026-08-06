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

## Gate-era artifacts (2026-08-04/05)

Produced while gating the production fix (`0df441f` + `a944727`) against stock
(`99fa204`, checked out as a second worktree). Every A/B below ran in a
**separate process** and is judged at **matched t** via
`../harness/matched_t.py`; each file's own header line carries the exact config
and command. `stock` = the 99fa204 worktree, `fixed`/`prod` = the branch head at
its `phase1a_segFrac = 0.1` default.

| file | what it is |
|---|---|
| `gate_results.csv` | **the ledger** — one row per gate check, with reference, source, measured value and verdict. Start here. |
| `g1b_m43_prod_segfrac0.csv` / `g1b_gmc_prod_segfrac0.csv` | G1b: production code at `phase1a_segFrac=0`, reproducing the committed `*_noapprox` ablation baselines |
| `g2_m43_prod.csv` / `g2_gmc_prod.csv` | G2: production vs the `*_logseg` prototype at the same eps (worst rel diff 2.3e-8 in R2). `g2_gmc_prod.csv` is SIGTERM-truncated at 8.2e4 yr — fine for this comparison, not for long-baseline work |
| `g2_simple_{stock,fixed}.csv`, `g2_lowdens_{stock,fixed}.csv`, `g2_hidens_{stock,fixed}.csv` | G2 stock-vs-fixed arms on `simple_cluster` and the two `f1edge` configs |
| `g2_longsimple_{stock,fixed}.csv`, `g2_longhidens_{stock,fixed}.csv` | the same pairs with `stop_t` extended (0.15 Myr) — `longhidens` is where the collapse-time and phase-at-collapse difference shows |
| `g2_1myr_simple_{stock,fixed}.csv` | **G2long**: `simple_cluster` to `stop_t=1`, both arms to STOPPING_TIME at 1.0 Myr. ΔR2 −0.078% at 1 Myr |
| `g2_gmc_fixed_full.csv` | **G2long**: the GMC control on the fixed arm run to its real end, `stop_t=2` → STOPPING_TIME at 2.0 Myr. Compare vs `gmc_control.csv`; ΔR2 −0.001%. Supersedes the SIGTERM-truncated `gmc_logseg.csv` / `g2_gmc_prod.csv` past 8.2e4 yr |
| `eps0.3_m43.csv` / `eps0.03_m43.csv` | eps convergence either side of the shipped 0.1 (pair with `g2_m43_prod.csv`) |
| `g3_slopes.csv` | G3: local `dlnR/dlnt` vs the Weaver 3/5 attractor, fixed and stock |
| `e8b_m43_noramp.csv`, `e8b_gmc_noramp.csv`, `e8b_hidens_noramp_STALLED.csv` | E8b (PLAN §8): ablating the `dt_switchon` R1 ramp *on top of* the fix. The `_STALLED` file is the finding — 4 rows in 90 min |
