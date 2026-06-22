# Data — single source of truth

Many CSVs accumulated here. Use the **canonical** one for each question; do not mix.
All `*.log` are gitignored (run logs). `bubble_Lgain == Lmech_total` exactly (verified,
0 rel-diff) — they are the same quantity; the production trigger uses `Lmech_total`
(`run_energy_implicit_phase.py:1075`).

| file glob | what it is | canonical for | key columns | NOT in it |
|---|---|---|---|---|
| **`c0_*_h0.csv`** | hybr full run, stop_t=6 | **triggers + cooling + forces** — F0–F4 harvest, WARPFIELD/leakage ratio, dip, surge, β–δ, blowout | `Lmech_total`(=`bubble_Lgain`), `bubble_Lloss`, `Lmech_W/SN`, `F_ram/F_rad/F_grav/F_ISM`, `P_*`, `R1/R2/v2`, `rCloud`, `cool_beta/delta`, `T0`, `Pb`, `Eb` | — |
| **`c0_*_st6.csv`** | hybr full run, stop_t=6 | **C0 certification + f_ret** | `f_ret`, `res_beta`, `res_T0_struct`, `cool_beta/delta`, `Eb`, `Pb`, `T0`, `Lmech_total`, `R2/v2` | ⚠️ **no `bubble_Lloss`, no forces** — never harvest triggers from st6 (silently gives `Lloss=0`) |
| `c0_*_legacy.csv` | **legacy** solver, stop_t=2.5 | BEFORE side of the BEFORE/AFTER (legacy crosses 0.05) | same columns as h0 | — |
| `leaktest/c0_sc_cf0XX.csv` | hybr + `coverFraction<1` | leakage test (does L_leak trip the cooling trigger?) | same as h0 | — |
| `c0_be_sphere_refine4*.csv` | hybr, 4× timestep refinement | C0.2 `res_beta ∝ Δt` truncation check (be_sphere only) | `res_beta` | — |
| `surge_coincidence.csv` | derived table | regenerable output of `plot_surge.py` | — | — |

## Stale / scratch — do NOT use
- `c0_*_st0p05.csv` — early short cert runs (stop_t=0.05), **superseded by `c0_*_st6.csv`**.
- `c0_small_dense_st0p5.csv`, `smoke_mock_provenance_unknown.csv` — old/mock, provenance not certified.

## Who reads what
- `harvest_h0.py`, `plot_{f0path,g0,blowout,surge,phaseportrait,dipdrivers,dipmechanism}.py` → `c0_*_h0.csv`
- `c0_consistency.py` (analysis), `plot_{fret,beta,cert}.py` → `c0_*_st6.csv`
- `plot_beforeafter.py` → `c0_*_legacy.csv` (BEFORE) + `c0_*_h0.csv` (AFTER)
