# Changelog

All notable changes to TRINITY will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] — Unreleased

First public release.

### Features

- Feedback-driven HII-region evolution code with phase transitions
  (energy-driven → transition → momentum-driven) and stopping fates
  (stall, dissolution, escape).
- Cloud density profiles: power-law, Bonnor-Ebert, homogeneous.
- Parameter sweep mode (Cartesian product and explicit-tuple modes).
- CLOUDY input-deck generation from TRINITY snapshots.
- Bundled minimal defaults (SB99 SPS table + cooling tables under
  `lib/default/`) so the README quickstart runs out of the box.

### Added

- Covering-fraction leak (`coverFraction`, Cf): geometry-set energy/mass leak
  where hot gas vents through the open `(1-Cf)*4*pi*R2^2` area at the interior
  sound speed. `Cf=1` recovers the sealed (Weaver) bubble exactly.
- `rCloud_max` parameter: user-tunable GMC-size validation limit (previously a
  hard-coded 200 pc cap).
- Cluster / SLURM-aware execution: `detect_allocated_cpus()` respects the SLURM
  allocation, `--workers` is validated and refuses over-requests, single-run
  `--dry-run`, and advisory nudges toward job arrays / off login nodes.
- Stricter `.param` validation: declaring `dens_profile=densPL` / `densBE`
  without its companion (`densPL_alpha` / `densBE_Omega`) now errors instead of
  silently inheriting a default, and `sps_refmass` must be declared explicitly
  when `sps_path` is user-set (fixes silently-wrong `f_mass = mCluster/sps_refmass`
  scaling for non-bundled SPS tables).

### Changed

- Bubble-structure integration migrated from `odeint` to
  `solve_ivp(method='LSODA', dense_output=True)`, decoupling integration accuracy
  from output sampling so near-duplicate radii in the legacy grid no longer trip
  dense-output interpolation.
- Package restructured: `src/` → `trinity/` (import paths change); paper figures
  funneled into `paper/plots/`.
- User-facing units / readability: densities are reported in cm^-3 (the
  parameter-file unit) in GMC suggestions and error messages, the GMC suggestion
  search is quieter, and the run-termination reason now reports the true cause.
- `simplify` monotonic-stack pass roughly 2× faster on large profiles
  (byte-identical output).

#### Changed — number-density / mean-molecular-weight audit

The whole code now uses a single convention: **every number density `n` is a
hydrogen-nuclei density** `n_H`, with mass density `rho = mu_convert * n_H`. Gas
composition is set by `x_He` and the ionisation states `Z_He` (hot bubble) and
`Z_He_shell` (~1e4 K shell); all mean molecular weights and the electron factors
`chi_e` / `chi_e_shell` are derived from them at load.

- Ionised-gas pressure now uses the He-aware factor `mu_H/mu_p` instead of the
  pure-hydrogen `2` (P_HII, P_ext across all phases).
- Bubble interior: density `n_H = (mu_p/mu_H) Pb/(k_B T)` and `rho = mu_H n_H`
  (fixes a factor-of-2 mass/self-gravity deficit); CIE cooling carries the
  electron factor `n_e n_H = chi_e n_H^2`.
- Shell structure rewritten on `n_H`: pressure-gradient prefactors `mu_p/mu_H`
  (ionised) and `mu_n/mu_H` (neutral), recombination / Strömgren balance carry
  `chi_e`, and the IR column uses `mu_H`.
- Helium is doubly ionised in the hot bubble but **singly ionised in the ~1e4 K
  shell/HII region** (`mu_ion_shell`, `chi_e_shell`), which is physical there.
- Bonnor-Ebert `densBE_Teff` clarified as an *effective* (turbulent) temperature
  and the support velocity dispersion exposed as `densBE_sigma` [km/s];
  `get_soundspeed` docstring corrected (adiabatic; pc/Myr).
- Removed dead `get_shellParams.py`. Added `test/test_mu_audit_drift.py` pinning
  every refined operation against its pre-fix value to prevent silent drift.

### Removed

- Dropped unused/experimental parameters: `adiabaticOnlyInCore`, `immediate_leak`,
  `stop_v`, `use_adaptive_solver`. (Old `.param` files referencing these should
  drop them.)

### Fixed

- Nondeterministic bubble-solver crash: detect LSODA `odeint` failure
  (`istate != 2`) instead of consuming uninitialised memory; return a
  deterministic penalty residual or raise `BubbleSolverError`. Fixes intermittent
  `MonotonicError` and cooling out-of-bounds errors under scipy ≥ 1.15.
- IR optical depth: carry the dimensionless `tau_IR` directly and fix a
  `dust_KappaIR` unit error (~4800× too large in the fallback path).
