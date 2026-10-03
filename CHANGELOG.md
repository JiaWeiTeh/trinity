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

- **`P_HII` is now a real photoionised pressure, not a relabelling of the
  confining pressure.** It was computed from the Strömgren density *after* that
  density was capped at the shell's inner value `shell_n0` — and `shell_n0` is
  itself defined by pressure balance against `Pb`, so converting it back to a
  pressure returned `Pb` term for term. The cap bound on 100% of rows, making
  `P_HII` carry no dependence on `Qi`, the escape fraction, or the ionised
  volume. In the momentum phase, where `Pb` *is* the wind ram pressure, this
  made `P_drive = P_HII + P_ram` come out at exactly `2 × P_ram` — the same
  pressure counted twice, half of it attributed to photoionisation.
  `P_HII` is now a confinement regime switch on the **cavity** Strömgren density
  (`get_bubbleParams.get_phii_c3c`): while the ionised gas is confined it is a
  thin skin that transmits the confining pressure and contributes exactly `0.0`;
  once confinement fails it drives at its own pressure. Photoionisation now
  responds to `Qi` and `R2`. Fates are unchanged on all 13 test configurations;
  shell radii move by 7.6–20.5%, and one configuration (a small dense cloud at
  high SFE) stays energy-driven where it previously handed over to the
  momentum phase.
- Fixes a related defect: the old `P_HII` read the **un-ramped** `Pb`, so it
  smuggled un-ramped pressure past the `dt_switchon` ramp and the ramp never
  throttled the drive. The confined branch returning exactly `0.0` restores it.
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
- Developer tooling now matches what the project actually runs: the `[dev]`
  extra and `CONTRIBUTING.md` use ruff + pre-commit (replacing the unused
  `flake8`), with `pip install -e ".[dev]"` as the documented dev setup.
- Citation updated to the method paper Teh et al. (2026), `arXiv:2605.27517`,
  consistently across the README and the Sphinx docs.

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
- A turnaround now ends the energy-driven phase: in phases 1a, 1b and 1c a downward
  zero crossing of `v2` drops `Eb` to the energy floor and goes straight to the
  momentum phase (`transition_channel` `velocity_sign_change`; phase 1c is skipped
  after any forced hand-off, i.e. `Eb_handoff` set and `Eb` at the floor, which also
  removes the one-segment 1c pass, and its occasional `solver_error`, after an `Eb <= 0`
  hand-off). This is an
  approximation (ruled 2026-10-01): a bubble that turns around while still at the
  ISM pressure loses its pressure support at once. The energy dropped is recorded in
  `Eb_handoff`.
- The collapse radius is `collapse_radius = min(coll_r, coll_r_frac * R2_max)` (new
  parameter `coll_r_frac`, default 0.1; `R2_max` is the largest radius reached), so
  a shell that never grew past a few pc is not stopped at `coll_r` on its first
  inward swing, floored at 0.01 pc. The `min_radius` event now fires at
  `collapse_radius` itself (it was `1.5 * coll_r`). The radius used and the term that set it are in `collapse_radius`
  and `collapse_rule` (`final_state`, not in snapshots), and the end reason reads e.g.
  "Small radius reached: R2 < 0.08 pc (0.1 x R2_max 0.8 pc; coll_r 1 pc not used)".
  0.1 from pilot_v6: 1 of 111 declared collapses premature, against 8 of 115 with 0.25.
- Anything not normal about a run is recorded in `metadata.json` `final_state`:
  `transition_channel` (how the energy phase ended, previously only in
  `trinity.log`), `solver_flags` (comma-separated: `1a_structure_failure`,
  `no_physical_root_handoff`, `cost_cap`; empty for a normal run),
  `n_unsolvable_segments`, `n_cost_capped_segments` (failed segments where the cap
  skipped the rescue ladder; they count as unsolvable) and `Eb_handoff`.
  `show_run` prints them.

### Removed

- Dropped unused/experimental parameters: `adiabaticOnlyInCore`, `immediate_leak`,
  `stop_v`, `use_adaptive_solver`. (Old `.param` files referencing these should
  drop them.)

### Fixed

- Old free-streaming seeds no longer lose the energy-driven phase. When the
  phase-0 seed time `dt_phase0` exceeds 1/3 kyr (weak winds, large M*/n), the R1
  switch-on ramp now spans `3·dt_phase0` instead of a fixed 1 kyr, and phase 1a
  runs to 3× that window (capped at `stop_t`, with the non-CIE cooling table
  refreshed on phase 1b's 5 kyr interval). Younger seeds keep the 1 kyr / 3 kyr values
  exactly. New runtime flag `dt_switchon` (not in snapshots).
- A spent bubble in phase 1a (the `energy_collapse` event, or a finite `Eb <= 0`)
  now continues in the momentum phase via 1c, skipping 1b, as phase 1b already
  did for `Eb <= 0`. It used to end the run as `ENERGY_COLLAPSED`. Non-finite `Eb`
  still ends the run; a phase-1a bubble-solve failure ends phase 1a early and phase
  1b continues (`solver_flags` `1a_structure_failure`, see below). New runtime flag
  `energy_handoff_1a` (not in snapshots); `metadata.json` `final_state` carries
  both new flags. Runs that neither have an old seed nor collapse in 1a are
  byte-identical in `dictionary.jsonl`.
- The event checker returned the non-terminal `velocity_sign` monitoring event, so
  from 2026-01-22 phase 1b handed over to 1c the moment `v2` first went negative,
  and a terminal event later in the same segment was replaced by a rewind to the
  turnaround. Monitoring events are now ignored by the checker. The turnaround is
  instead a deliberate phase end (see Changed).
- Phase 1a ended the run as `ENERGY_COLLAPSED` "Eb -> 0" on any bubble-solve
  failure, but it fired on growing bubbles: LSODA's dense output raised
  "`ts` must be strictly increasing or decreasing" (two equal step points) on
  5e9 Msun / n 1e5 clouds at 0.5–1.7 kyr with `Eb` ~6e53 erg and rising. The
  structure solve now retries once with Radau on that error only, and a phase-1a solve failure ends
  phase 1a early (phase 1b, which has a rescue ladder, continues) and is flagged
  `1a_structure_failure` in `solver_flags`.
- Phase 1b could grind for hours without handing off when the beta-delta structure
  had no physical root: the hand-off counted 50 *consecutive* no-root segments and
  an occasional successful rescue reset it. It now hands off when 50 of the last 60
  segments had no physical root or a non-finite residual (channel still
  `no_physical_root_handoff`), and once the root has been unreachable for more than
  10 segments the rescue ladder runs only every 10th segment (cost cap).
- Nondeterministic bubble-solver crash: detect LSODA `odeint` failure
  (`istate != 2`) instead of consuming uninitialised memory; return a
  deterministic penalty residual or raise `BubbleSolverError`. Fixes intermittent
  `MonotonicError` and cooling out-of-bounds errors under scipy ≥ 1.15.
- IR optical depth: carry the dimensionless `tau_IR` directly and fix a
  `dust_KappaIR` unit error (~4800× too large in the fallback path).
- Unsupported `ZCloud` (anything other than the bundled 1.0 / 0.15) now raises a
  clear `ValueError` naming the available metallicities, instead of a cryptic
  `UnboundLocalError` from deep inside the non-CIE cooling loader.
- Stale docstrings, comments, and dead doc pointers corrected: the reader docs'
  nonexistent `example_scripts/` reference, the run startup-banner documentation
  link (now the maintained `jiaweiteh.github.io/trinity-web`), and several wrong
  unit labels (`Eb`, `get_dudt`, the shell ODE, Bonnor-Ebert debug logs).

### Documentation

- Added a fresh-clone consistency review (`docs/dev/CODEBASE_REVIEW.md`) auditing
  the repo for code/docstring/comment inconsistencies and "shouldn't-ship" cruft.
- Reorganised internal development notes under `docs/dev/` into self-contained
  per-workstream folders (`betadelta/`, `transition/`, `bubble/`, `cooling/`,
  `n-consistency/`, `misc/`), each holding its writeups plus its harnesses/figures;
  the former top-level `analysis/` tree and the `scratch/` diagnostics were folded
  in, and a `docs/dev/README.md` index maps everything. (`scratch/` at the repo
  root remains git-ignored / local-only.)
