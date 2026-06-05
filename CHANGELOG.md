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

### Changed — number-density / mean-molecular-weight audit

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
