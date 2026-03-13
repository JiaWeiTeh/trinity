# Plan: Refactor `_calc/` Monolithic Files (Issue 3.4)

## Problem Summary

All 10 Python files in `src/_calc/` (16,765 lines total) follow the same monolithic pattern: each file combines data extraction, physics computation, curve fitting, plotting, file output, and CLI handling into a single file. Eight exceed 1,000 lines. This makes the computation logic hard to reuse, test, or maintain independently from visualization.

## Key Observations

1. **Duplicated helpers** across 5+ files: `_cloud_radius_pc()`, `_surface_density()`, `_freefall_time_Myr()`, `_cumtrapz_1d()`, `_ols_sigma_clip()`
2. **Every `plot_*()` function** mixes data filtering/interpolation with matplotlib calls — computation is not separable from rendering
3. **All files share the same pipeline**: extract → collect → fit → plot → output
4. **Two inference files** (`infer_cluster_mass.py`, `infer_cluster_age.py`) share pre-stored observable systems and grid helpers
5. **`run_all.py`** orchestrates via subprocess — no in-process imports of computation functions

## Refactoring Strategy

Decompose each monolithic file into **three layers**:

```
src/_calc/
├── _common/                    # NEW: shared utilities
│   ├── __init__.py
│   ├── cloud_physics.py        # cloud_radius_pc, surface_density, freefall_time
│   ├── fitting.py              # ols_sigma_clip, power-law fitters
│   ├── integration.py          # cumtrapz_1d
│   └── observables.py          # pre-stored systems (RCW120, Orion, etc.)
│
├── velocity_radius/            # REFACTORED from velocity_radius.py
│   ├── __init__.py
│   ├── compute.py              # extract_run, compute_alpha_local, compute_eta, fit_*
│   ├── plot.py                 # all plot_* functions
│   └── __main__.py             # CLI entry point (argparse + dispatch)
│
├── terminal_momentum/          # same pattern
│   ├── __init__.py
│   ├── compute.py
│   ├── plot.py
│   └── __main__.py
│
├── ... (same for all 8 analysis files)
│
├── run_all.py                  # keep as-is (orchestration)
└── __init__.py
```

## Step-by-Step Plan

### Step 1: Extract shared utilities into `_common/`

Create `src/_calc/_common/` with:

- **`cloud_physics.py`**: Deduplicate `_cloud_radius_pc()`, `_surface_density()`, `_freefall_time_Myr()` (currently copy-pasted in 5 files)
- **`fitting.py`**: Extract `_ols_sigma_clip()` (duplicated in 6 files) and shared power-law fitting utilities
- **`integration.py`**: Extract `_cumtrapz_1d()` (duplicated in 2 files)
- **`observables.py`**: Extract pre-stored systems dict (RCW120, Orion_Veil, Carina, Rosette, N49) shared by `diagnostic_diagrams`, `infer_cluster_mass`, `infer_cluster_age`

### Step 2: Refactor the smallest file first as a template — `diagnostic_diagrams.py` (903 lines)

This file is the simplest (no fitting, pure visualization + grid loading). Use it to establish the pattern:

1. Create `src/_calc/diagnostic_diagrams/` directory
2. Move grid-loading and interpolation helpers into `compute.py`
3. Move `plot_fig1()` through `plot_fig4()` into `plot.py`, making them accept pre-computed data as arguments instead of loading internally
4. Move CLI (`main()`, argparse) into `__main__.py`
5. Update `run_all.py` subprocess call if needed (use `python -m src._calc.diagnostic_diagrams`)
6. Delete original `diagnostic_diagrams.py`
7. Verify: run the script, confirm identical output

### Step 3: Refactor `scaling_phases.py` (1,537 lines)

1. Create `src/_calc/scaling_phases/` directory
2. `compute.py`: `extract_timescales()`, `collect_data()`, `fit_scaling()`, `fit_scaling_piecewise()`
3. `plot.py`: `plot_parity()`, `plot_parity_diagnostic()`, `plot_parity_piecewise()` — accept fitted data as args
4. `__main__.py`: CLI + dispatch
5. Replace inline `_cloud_radius_pc()` etc. with imports from `_common`
6. Verify output

### Step 4: Refactor `collapse_criterion.py` (1,336 lines)

Same pattern:
- `compute.py`: `classify_outcome()`, `find_epsilon_min()`, `fit_eps_min_nM()`, `fit_eps_min_sigma()`
- `plot.py`: `plot_phase_diagram()`, `plot_eps_vs_sigma()`, `plot_parity()`
- `__main__.py`: CLI

### Step 5: Refactor `dispersal_timescale.py` (1,343 lines)

- `compute.py`: `extract_run()`, `collect_data()`, `fit_scaling()`, helpers
- `plot.py`: `plot_t_disp_vs_sfe()`, `plot_t_disp_normalized()`, `plot_feedback_velocity()`, `plot_epsilon_ff()`
- `__main__.py`: CLI

### Step 6: Refactor `energy_retention.py` (1,432 lines)

- `compute.py`: `extract_run()`, energy budget integration, `fit_scaling()`, `_analytic_tcool_Myr()`
- `plot.py`: `plot_xi_evolution()`, `plot_energy_budget()`, `plot_xi_vs_params()`, `plot_thalf_vs_tcool()`
- `__main__.py`: CLI

### Step 7: Refactor `terminal_momentum.py` (1,864 lines)

- `compute.py`: `extract_run()`, `fit_p_mstar()`, `fit_p_mstar_quad()`, `fit_p_mstar_piecewise()`
- `plot.py`: 7 plot functions
- `__main__.py`: CLI

### Step 8: Refactor `velocity_radius.py` (2,348 lines)

Largest file. May need `compute.py` split further:
- `compute.py`: `extract_run()`, `compute_alpha_local()`, `compute_eta()`, `fit_velocity_scaling()`, `fit_radius_scaling()`
- `plot.py`: 9 plot functions
- `__main__.py`: CLI

### Step 9: Refactor `bubble_distribution.py` (2,186 lines)

- `compute.py`: `extract_bubble_data()`, sampling functions, `_build_2d_grid()`, `run_population_synthesis_2d()`, `run_sensitivity()`
- `plot.py`: 5 plot functions
- `__main__.py`: CLI

### Step 10: Refactor inference files

**`infer_cluster_mass.py` (2,167 lines):**
- `compute.py`: grid loading, density interpolation, `compute_posterior_grid()`, `run_progressive_inference()`
- `plot.py`: `plot_posterior()`, `plot_2d_scatter()`, `plot_Rt_tracks()`
- `__main__.py`: CLI

**`infer_cluster_age.py` (1,123 lines):**
- `compute.py`: `compute_age_posterior()`, `run_progressive_inference()`
- `plot.py`: `plot_posterior()`, `plot_Rt_tracks()`
- `__main__.py`: CLI
- Import shared grid helpers from `infer_cluster_mass.compute` instead of duplicating

### Step 11: Update `run_all.py`

- Update script registry paths from `src._calc.velocity_radius` → `src._calc.velocity_radius.__main__` (or use `-m` flag)
- Verify all 7 dispatched scripts still run correctly

### Step 12: Clean up

- Remove any remaining duplicated helpers replaced by `_common` imports
- Ensure `__init__.py` files export the right public API for each subpackage
- Run full `run_all.py` end-to-end to verify no regressions

## Principles

- **No behavior changes** — this is a pure refactoring, all outputs (CSVs, PDFs, JSON equations) must remain identical
- **Plot functions receive data, not file paths** — computation functions return dataframes/dicts, plot functions consume them
- **One responsibility per module** — `compute.py` has zero matplotlib imports, `plot.py` has no fitting logic
- **Incremental & verifiable** — each step produces a working state that can be tested before proceeding
- **Preserve CLI interface** — `python -m src._calc.velocity_radius` must work identically to the old `python src/_calc/velocity_radius.py`

## Risk Assessment

- **Low risk**: All changes are internal restructuring with no physics changes
- **Main risk**: `run_all.py` uses subprocess dispatch — paths must be updated carefully
- **Mitigation**: Refactor one file at a time, verify output after each step
