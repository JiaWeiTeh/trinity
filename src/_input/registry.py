"""Module-level registry of ParamSpec entries — the single source of truth.

Phase 2 populates ``SPECS`` with one ``ParamSpec`` per parameter that
TRINITY produces (187 total: 72 declared in ``default.param`` + 115
runtime/derived created in ``read_param`` Steps 6/8/10).  Nothing in
production imports this module yet; ``read_param.py`` and
``run_constants.py`` are untouched until Phases 5–10 wire it in.

``default`` field convention
----------------------------
* **Input specs** (``category`` starts with ``input_``): ``default`` is
  the *raw source string* exactly as it would appear in
  ``default.param`` (e.g. ``"1e7"``, ``"5/3"``, ``"None"``, ``"True"``).
  Phase 10's builder parses it via the same ``parse_value`` path
  ``read_param`` uses for file content.  Fraction-encoded constants
  (``mu_atom`` etc.) are stored as ``"14/11"`` strings per the agreed
  representation; they round-trip to the identical float.
* **Runtime / derived specs**: ``default`` is the literal initial
  *value* assigned in Step 6/8/10 (``0``, ``np.nan``, ``np.array([])``,
  ``False``, ``None`` …) — used directly, no parsing.

Derivation helpers
------------------
``run_const_keys()`` and ``metadata_exclude_keys()`` reproduce
``src._output.run_constants.RUN_CONST_KEYS`` and ``METADATA_EXCLUDE``
exactly from the per-spec ``run_const`` / ``metadata_exclude`` booleans.
The reconciliation tests in ``test/test_registry.py`` pin the equality
(modulo the four known-stale legacy entries documented below).  Phase 5
swaps ``run_constants.py`` to call these helpers and deletes the stale
entries.
"""
from __future__ import annotations

from collections import OrderedDict
from typing import Iterable

import numpy as np

from src._input.param_spec import Category, ParamSpec


# ---------------------------------------------------------------------------
# Conditional-schema predicates (consumed by Phase 8; dormant until then).
# ``read_param`` Step 8 keeps densBE_* / pops densPL_alpha when the cloud
# uses a Bonnor-Ebert profile, and vice-versa.
# ---------------------------------------------------------------------------
def _profile_value(params) -> object:
    item = params.get("dens_profile")
    return getattr(item, "value", item)


def _active_densBE(params) -> bool:
    return _profile_value(params) == "densBE"


def _active_densPL(params) -> bool:
    return _profile_value(params) == "densPL"


# Legacy run_constants entries that refer to keys no longer produced by
# any code path (cleaned up when Phase 5 swaps the derivation).  Named
# here so the reconciliation tests can carve them out explicitly.
KNOWN_STALE_RUN_CONST: frozenset[str] = frozenset({
    "expansionBeyondCloud",          # never created in params
})
KNOWN_STALE_METADATA_EXCLUDE: frozenset[str] = frozenset({
    "SB99_data", "SB99f", "path_sps",  # removed in PR #627, never cleaned
})


SPECS: tuple[ParamSpec, ...] = (
    ParamSpec(name='model_name', default='default', info='Specifies the model name, which serves as the prefix for all output filenames.', category='input_admin', unit=None, run_const=True),
    ParamSpec(name='path2output', default='def_dir', info='Defines the output directory where all generated files will be stored.', category='input_admin', unit=None, exclude_from_snapshot=True, metadata_exclude=True),
    ParamSpec(name='output_format', default='JSON', info='Specifies the output format.', category='input_admin', unit=None, exclude_from_snapshot=True, run_const=True),
    ParamSpec(name='simplify_npoints', default='100', info='Target number of points retained for simplified profile arrays in saved snapshots (bubble_T_arr, bubble_n_arr, bubble_dTdr_arr, bubble_v_arr, shell_grav_force_m, shell_n_arr). Default 100. Larger values give higher-fidelity snapshots at the cost of larger output files. Clamped to >= 20 (matches the coverage-skeleton chunk count). The first two simplify calls per implicit-phase snapshot log their reconstruction R² at INFO level so you can verify the chosen budget is faithful.', category='input_admin', unit=None, exclude_from_snapshot=True),
    ParamSpec(name='log_level', default='DEBUG', info='Logging level for terminal and file output. Controls how much detail is logged.', category='input_admin', unit=None, exclude_from_snapshot=True, run_const=True),
    ParamSpec(name='log_console', default='False', info='Enable console (terminal) output for logging. If True, log messages are printed to terminal.', category='input_admin', unit=None, exclude_from_snapshot=True, run_const=True),
    ParamSpec(name='log_file', default='True', info='Enable file output for logging. If True, log messages are written to a .log file in the output directory.', category='input_admin', unit=None, exclude_from_snapshot=True, run_const=True),
    ParamSpec(name='log_colors', default='True', info='Use colored output in terminal. If True, log messages are color-coded by severity (DEBUG=cyan, INFO=green, WARNING=yellow, ERROR=red, CRITICAL=magenta).', category='input_admin', unit=None, exclude_from_snapshot=True, run_const=True),
    ParamSpec(name='mCloud', default='1e7', info='The mass of the molecular cloud.', category='input_physical', unit='Msun', run_const=True),
    ParamSpec(name='sfe', default='0.01', info='Star formation efficiency.', category='input_physical', unit=None, exclude_from_snapshot=True, run_const=True),
    ParamSpec(name='ZCloud', default='1', info='Cloud metallicity', category='input_physical', unit='Zsun', exclude_from_snapshot=True, run_const=True),
    ParamSpec(name='include_PHII', default='True', info='Include HII pressure (from Strömgren ionization balance in shell) in P_drive. When False, P_HII is set to zero.', category='input_physical', unit=None, exclude_from_snapshot=True, run_const=True),
    ParamSpec(name='dens_profile', default='densPL', info='Specifies how the cloud density scales with radius.', category='input_profile', unit=None, run_const=True),
    ParamSpec(name='densBE_Omega', default='14.1', info='if `densBE` is selected, then the ratio `Omega = nCore/nCloudEdge` must be specified.', category='input_profile', unit=None, exclude_from_snapshot=True, run_const=True, active_when=_active_densBE),
    ParamSpec(name='densPL_alpha', default='0', info='if `densPL` is selected, then the power-law coefficient `nCore*(r/rCore)^alpha` (0 = homogeneous, -2 = isothermal) must be specified.', category='input_profile', unit=None, run_const=True, active_when=_active_densPL),
    ParamSpec(name='nCore', default='1e5', info='Hydrogen nuclei number density of cloud core (n_H). Standard GMC/ISM convention. Mass density: rho = nCore * mu_convert * m_H. If `densPL` AND densPL_alpha = 0, this is the average cloud density.', category='input_physical', unit='cm**-3', run_const=True),
    ParamSpec(name='nISM', default='1', info='Hydrogen nuclei number density of the ambient ISM (n_H). Mass density: rho = nISM * mu_convert * m_H.', category='input_physical', unit='cm**-3', run_const=True),
    ParamSpec(name='rCore', default='0.01', info='Core radius of the molecular cloud.', category='input_physical', unit='pc', run_const=True),
    ParamSpec(name='allowShellDissolution', default='True', info='Allow shell dissolution to terminate simulation. If False, shell dissolution check is disabled.', category='input_termination', unit=None, exclude_from_snapshot=True, run_const=True),
    ParamSpec(name='stop_t_diss', default='1', info='Duration (in Myr) that shell_nMax must remain below nISM before dissolution is triggered.', category='input_termination', unit='Myr', exclude_from_snapshot=True, run_const=True),
    ParamSpec(name='stop_r', default='500', info='Maximum radial extent permitted for shell expansion. Set to None to disable this termination condition.', category='input_termination', unit='pc', exclude_from_snapshot=True, run_const=True),
    ParamSpec(name='stop_v', default='-1e4', info='', category='deprecated', unit='km * s**-1', exclude_from_snapshot=True, run_const=True, deprecated_note='Parsed for backward compatibility with existing .param files but NOT consumed by any current code path. Changing this has no effect.'),
    ParamSpec(name='stop_t', default='15', info='Maximum duration of the simulation. Set to None to disable this termination condition.', category='input_termination', unit='Myr', exclude_from_snapshot=True, run_const=True),
    ParamSpec(name='stop_at_rCloud_nSnap', default='None', info='Terminate simulation after the shell crosses the cloud edge (R2 > rCloud).', category='input_termination', unit=None, exclude_from_snapshot=True),
    ParamSpec(name='coll_r', default='1', info='Radius below which the cloud is considered completely collapsed.', category='input_termination', unit='pc', exclude_from_snapshot=True, run_const=True),
    ParamSpec(name='SB99_rotation', default='1', info='Stellar-rotation flag. Selects rot vs norot non-CIE cooling tables', category='input_sps', unit=None, exclude_from_snapshot=True, run_const=True),
    ParamSpec(name='sps_refmass', default='def_value', info='Reference cluster mass used in f_mass = mCluster / sps_refmass.', category='input_sps', unit='Msun', exclude_from_snapshot=True),
    ParamSpec(name='FB_mColdWindFrac', default='0', info='Fraction of cold mass entrained in stellar winds (increases Mdot_wind, reduces velocity).', category='input_sps', unit=None, exclude_from_snapshot=True, run_const=True),
    ParamSpec(name='FB_mColdSNFrac', default='0', info='Fraction of cold mass entrained in supernova ejecta (increases Mdot_SN, reduces velocity).', category='input_sps', unit=None, exclude_from_snapshot=True, run_const=True),
    ParamSpec(name='FB_thermCoeffWind', default='1', info='Defines the thermalization efficiency for colliding stellar winds.', category='input_sps', unit=None, exclude_from_snapshot=True, run_const=True),
    ParamSpec(name='FB_thermCoeffSN', default='1', info='Defines the thermalization efficiency for supernova ejecta.', category='input_sps', unit=None, exclude_from_snapshot=True, run_const=True),
    ParamSpec(name='FB_vSN', default='1e4', info='Specifies the velocity of supernova ejecta.', category='input_sps', unit='km * s**-1', exclude_from_snapshot=True, run_const=True),
    ParamSpec(name='mu_atom', default='14/11', info='Mean molecular weight for neutral atomic gas (HI + He). Dimensionless, in units of m_H. Composition: He:H = 1:10 by number. Formula: (10×1 + 1×4) / 11 = 14/11.', category='input_constants', unit='m_H', exclude_from_snapshot=True, run_const=True),
    ParamSpec(name='mu_ion', default='14/23', info='Mean molecular weight for fully ionized gas (H+ + He++). Dimensionless, in units of m_H. Composition: He:H = 1:10 by number. Formula: (10×1 + 1×4) / 23 = 14/23.', category='input_constants', unit='m_H', exclude_from_snapshot=True, run_const=True),
    ParamSpec(name='mu_mol', default='14/6', info='Mean molecular weight for molecular gas (H2 + He). Dimensionless, in units of m_H. Composition: He:H = 1:10 by number. Formula: (10×1 + 1×4) / 6 = 14/6.', category='input_constants', unit='m_H', exclude_from_snapshot=True, run_const=True),
    ParamSpec(name='mu_convert', default='1.4', info='Conversion factor for mass density calculation (mass integration). Use this for n→rho conversion. Independent of ionization state - ionization changes particle counts but NOT total mass.', category='input_constants', unit='m_H', exclude_from_snapshot=True, run_const=True),
    ParamSpec(name='TShell_neu', default='1e2', info='Temperature of the neutral shell region.', category='input_constants', unit=None, exclude_from_snapshot=True, run_const=True),
    ParamSpec(name='TShell_ion', default='1e4', info='Temperature of the ionised shell region.', category='input_constants', unit=None, exclude_from_snapshot=True, run_const=True),
    ParamSpec(name='dust_sigma', default='1.5e-21', info='Dust cross-section at solar metallicity.', category='input_constants', unit='cm**2', exclude_from_snapshot=True, run_const=True),
    ParamSpec(name='dust_noZ', default='0.05', info='Metallicity below which there is effectively no dust', category='input_constants', unit='Zsun', exclude_from_snapshot=True, run_const=True),
    ParamSpec(name='dust_KappaIR', default='4', info='The Rosseland mean dust opacity kappa_IR.', category='input_constants', unit='cm**2 * g**-1', exclude_from_snapshot=True, run_const=True),
    ParamSpec(name='caseB_alpha', default='2.59e-13', info='The case B recombination coefficient', category='input_constants', unit='cm**3 * s**-1', exclude_from_snapshot=True, run_const=True),
    ParamSpec(name='gamma_adia', default='5/3', info='the adiabatic index', category='input_constants', unit=None, exclude_from_snapshot=True, run_const=True),
    ParamSpec(name='C_thermal', default='6e-7', info='The thermal conduction coefficient C', category='input_constants', unit='erg * s**-1 * cm**-1 * K**(-7/2)', exclude_from_snapshot=True, run_const=True),
    ParamSpec(name='c_light', default='29979245800', info='speed of light', category='input_constants', unit='cm * s**-1', exclude_from_snapshot=True, run_const=True),
    ParamSpec(name='G', default='6.6743e-08', info='Gravitational constant', category='input_constants', unit='cm**3 * g**-1 * s**-2', exclude_from_snapshot=True, run_const=True),
    ParamSpec(name='k_B', default='1.380649e-16', info='Boltzmann constant', category='input_constants', unit='erg * K**-1', exclude_from_snapshot=True, run_const=True),
    ParamSpec(name='PISM', default='0', info='ISM Pressure, P/k', category='input_constants', unit='K * cm**-3', exclude_from_snapshot=True, run_const=True),
    ParamSpec(name='adiabaticOnlyInCore', default='False', info='', category='deprecated', unit=None, exclude_from_snapshot=True, run_const=True, deprecated_note='Parsed for backward compatibility with existing .param files but NOT consumed by any current code path. Changing this has no effect.'),
    ParamSpec(name='immediate_leak', default='True', info='', category='deprecated', unit=None, exclude_from_snapshot=True, run_const=True, deprecated_note='Parsed for backward compatibility with existing .param files but NOT consumed by any current code path. Changing this has no effect.'),
    ParamSpec(name='phaseSwitch_LlossLgain', default='0.05', info='When (Lgain-Lloss)/Lgain approaches this value, begin momentum-driving phase.', category='input_solver', unit=None, exclude_from_snapshot=True, run_const=True),
    ParamSpec(name='use_adaptive_solver', default='True', info='', category='deprecated', unit=None, exclude_from_snapshot=True, run_const=True, deprecated_note='Parsed for backward compatibility with existing .param files but NOT consumed by any current code path. The modified adaptive solver is now the only implementation; changing this flag has no effect.'),
    ParamSpec(name='cool_alpha', default='0.6', info='Cooling related values. alpha = v2*t_now/R2', category='input_solver', unit=None),
    ParamSpec(name='cool_beta', default='0.8', info='Cooling related values. beta = - dPb/dt.', category='input_solver', unit=None),
    ParamSpec(name='cool_delta', default='-6/35', info='Cooling related values. delta = dT/dt.', category='input_solver', unit=None),
    ParamSpec(name='path_cooling_CIE', default='3', info='Selects the CIE (T > 10^5.5 K) cooling curve.', category='input_cooling', unit=None, exclude_from_snapshot=True, metadata_exclude=True),
    ParamSpec(name='path_cooling_nonCIE', default='def_dir', info='Folder containing the non-CIE (T < 10^5.5 K) OPIATE/CLOUDY cubes.', category='input_cooling', unit=None, exclude_from_snapshot=True, metadata_exclude=True),
    ParamSpec(name='sps_path', default='def_path', info='Full path to an SPS data file. If def_path (default), TRINITY uses', category='input_sps', unit=None, exclude_from_snapshot=True),
    ParamSpec(name='sps_col_t', default='def_unset', info='', category='input_sps', unit=None, exclude_from_snapshot=True),
    ParamSpec(name='sps_col_Qi', default='def_unset', info='', category='input_sps', unit=None, exclude_from_snapshot=True),
    ParamSpec(name='sps_col_fi', default='def_unset', info='', category='input_sps', unit=None, exclude_from_snapshot=True),
    ParamSpec(name='sps_col_Lbol', default='def_unset', info='', category='input_sps', unit=None, exclude_from_snapshot=True),
    ParamSpec(name='sps_col_Lmech_W', default='def_unset', info='', category='input_sps', unit=None, exclude_from_snapshot=True),
    ParamSpec(name='sps_col_pdot_W', default='def_unset', info='', category='input_sps', unit=None, exclude_from_snapshot=True),
    ParamSpec(name='sps_col_Lmech_total', default='def_unset', info='', category='input_sps', unit=None, exclude_from_snapshot=True),
    ParamSpec(name='sps_col_Lmech_SN', default='def_unset', info='', category='input_sps', unit=None, exclude_from_snapshot=True),
    ParamSpec(name='sps_col_pdot_SN', default='def_unset', info='', category='input_sps', unit=None, exclude_from_snapshot=True),
    ParamSpec(name='sps_col_Mdot_SN', default='def_unset', info='', category='input_sps', unit=None, exclude_from_snapshot=True),
    ParamSpec(name='sps_col_v_SN', default='def_unset', info='', category='input_sps', unit=None, exclude_from_snapshot=True),
    ParamSpec(name='sps_col_Li', default='def_unset', info='', category='input_sps', unit=None, exclude_from_snapshot=True),
    ParamSpec(name='sps_col_Ln', default='def_unset', info='', category='input_sps', unit=None, exclude_from_snapshot=True),
    ParamSpec(name='bubble_xi_Tb', default='0.98', info='The relative radius xi = r/R2, at which we measure the bubble temperature.', category='input_solver', unit=None, exclude_from_snapshot=True, run_const=True),
    ParamSpec(name='mCloud_input', default=1000000.0, info='Pre-SFE input cloud mass (= mCloud + mCluster). Matches the .param file and the sweep folder-name tag.', category='derived_init', unit='Msun', exclude_from_snapshot=True, run_const=True),
    ParamSpec(name='mCluster', default=10000.0, info='Cluster mass (mCloud_input * sfe)', category='derived_init', unit='Msun', exclude_from_snapshot=True, run_const=True),
    ParamSpec(name='sps_column_map', default=None, info='SPS column mapping (canonical -> ColumnSpec)', category='runtime_loaded', unit='N/A', exclude_from_snapshot=True),
    ParamSpec(name='current_phase', default='', info='Current simulation phase: energy/implicit/transition/momentum', category='runtime_state', unit='N/A'),
    ParamSpec(name='EndSimulationDirectly', default=False, info='Flag to immediately end simulation', category='runtime_state', unit='N/A'),
    ParamSpec(name='SimulationEndReason', default='', info='Reason for simulation completion', category='runtime_state', unit='N/A'),
    ParamSpec(name='SimulationEndCode', default=None, info='Exit code (SimulationEndCode enum) for simulation completion', category='runtime_state', unit='N/A'),
    ParamSpec(name='EarlyPhaseApproximation', default=True, info='Using approximations for early phase?', category='runtime_state', unit='N/A'),
    ParamSpec(name='_snapshots_after_rCloud', default=0, info='Snapshots saved with R2 > rCloud (used by stop_at_rCloud_nSnap)', category='runtime_state', unit='N/A', exclude_from_snapshot=True),
    ParamSpec(name='tSF', default=0, info='Time of star formation', category='derived_init', unit='Myr', run_const=True),
    ParamSpec(name='t_now', default=0, info='Current simulation time', category='runtime_state', unit='Myr'),
    ParamSpec(name='v2', default=0, info='Velocity at R2 (outer bubble radius = inner shell edge)', category='runtime_state', unit='pc/Myr'),
    ParamSpec(name='R2', default=0, info='Outer bubble radius (= inner shell edge)', category='runtime_state', unit='pc'),
    ParamSpec(name='T0', default=0, info='Characteristic bubble temperature (at xi_Tb fraction of bubble thickness)', category='runtime_state', unit='K'),
    ParamSpec(name='Eb', default=0, info='Bubble energy', category='runtime_state', unit='Msun*pc**2/Myr**2'),
    ParamSpec(name='R1', default=0, info='Inner bubble radius', category='runtime_state', unit='pc'),
    ParamSpec(name='Pb', default=0, info='Bubble pressure', category='runtime_state', unit='Msun/Myr**2/pc'),
    ParamSpec(name='c_sound', default=0, info='Sound speed', category='runtime_state', unit='pc/Myr'),
    ParamSpec(name='t_next', default=0, info='Next time for mShell interpolation', category='runtime_state', unit='Myr'),
    ParamSpec(name='rCloud', default=0, info='Cloud radius', category='derived_init', unit='pc', run_const=True),
    ParamSpec(name='rShell', default=0, info='Shell outer radius', category='runtime_state', unit='pc'),
    ParamSpec(name='nEdge', default=0, info='Number density at cloud edge', category='derived_init', unit='1/pc**3', run_const=True),
    ParamSpec(name='initial_cloud_r_arr', default=np.array([]), info='Initial cloud radius array', category='runtime_state', unit='pc'),
    ParamSpec(name='initial_cloud_n_arr', default=np.array([]), info='Initial cloud density array', category='runtime_state', unit='1/cm**3'),
    ParamSpec(name='initial_cloud_m_arr', default=np.array([]), info='Initial cloud enclosed mass array', category='runtime_state', unit='Msun'),
    ParamSpec(name='sps_data', default=0, info='SPS raw 11-array datacube', category='runtime_loaded', unit='N/A', exclude_from_snapshot=True),
    ParamSpec(name='sps_f', default=0, info='SPS interpolators (dict of scipy interp1d)', category='runtime_loaded', unit='N/A', exclude_from_snapshot=True),
    ParamSpec(name='Lmech_W', default=0, info='Wind mechanical luminosity', category='runtime_state', unit='Msun*pc**2/Myr**3'),
    ParamSpec(name='Lmech_SN', default=0, info='SN mechanical luminosity', category='runtime_state', unit='Msun*pc**2/Myr**3'),
    ParamSpec(name='Lmech_total', default=0, info='Total mechanical luminosity', category='runtime_state', unit='Msun*pc**2/Myr**3'),
    ParamSpec(name='v_mech_total', default=0, info='mechanical velocity (winds+SN)', category='runtime_state', unit='pc/Myr'),
    ParamSpec(name='pdot_W', default=0, info='Wind momentum rate', category='runtime_state', unit='Msun*pc/Myr**2'),
    ParamSpec(name='pdot_SN', default=0, info='Supernova momentum rate', category='runtime_state', unit='Msun*pc/Myr**2'),
    ParamSpec(name='pdot_total', default=0, info='Total momentum rate', category='runtime_state', unit='Msun*pc/Myr**2'),
    ParamSpec(name='pdotdot_total', default=0, info='Rate of wind momentum rate', category='runtime_state', unit='Msun*pc/Myr**3'),
    ParamSpec(name='Qi', default=0, info='Ionizing photon rate', category='runtime_state', unit='1/Myr'),
    ParamSpec(name='Lbol', default=0, info='Bolometric luminosity', category='runtime_state', unit='Msun*pc**2/Myr**3'),
    ParamSpec(name='Ln', default=0, info='Non-ionizing luminosity', category='runtime_state', unit='Msun*pc**2/Myr**3'),
    ParamSpec(name='Li', default=0, info='Ionizing luminosity', category='runtime_state', unit='Msun*pc**2/Myr**3'),
    ParamSpec(name='t_previousCoolingUpdate', default=1e+30, info='Time of previous cooling update', category='runtime_state', unit='Myr'),
    ParamSpec(name='cStruc_cooling_nonCIE', default=0, info='Non-CIE cooling cube', category='runtime_loaded', unit='N/A', exclude_from_snapshot=True, metadata_exclude=True),
    ParamSpec(name='cStruc_heating_nonCIE', default=0, info='Non-CIE heating cube', category='runtime_loaded', unit='N/A', exclude_from_snapshot=True, metadata_exclude=True),
    ParamSpec(name='cStruc_net_nonCIE_interpolation', default=0, info='Non-CIE net cooling interpolation', category='runtime_loaded', unit='N/A', exclude_from_snapshot=True, metadata_exclude=True),
    ParamSpec(name='cStruc_cooling_CIE_logT', default=0, info='CIE log temperature array', category='runtime_loaded', unit='N/A', exclude_from_snapshot=True, metadata_exclude=True),
    ParamSpec(name='cStruc_cooling_CIE_logLambda', default=0, info='CIE log lambda array', category='runtime_loaded', unit='N/A', exclude_from_snapshot=True, metadata_exclude=True),
    ParamSpec(name='cStruc_cooling_CIE_interpolation', default=0, info='CIE cooling interpolation', category='runtime_loaded', unit='N/A', exclude_from_snapshot=True, metadata_exclude=True),
    ParamSpec(name='shell_fAbsorbedIon', default=1, info='Fraction of absorbed ionizing radiation', category='runtime_state', unit='dimensionless'),
    ParamSpec(name='shell_fAbsorbedNeu', default=0, info='Fraction of absorbed non-ionizing radiation', category='runtime_state', unit='dimensionless'),
    ParamSpec(name='shell_fAbsorbedWeightedTotal', default=0, info='Total absorption fraction (luminosity weighted)', category='runtime_state', unit='dimensionless'),
    ParamSpec(name='shell_fIonisedDust', default=0, info='Ionized dust fraction', category='runtime_state', unit='dimensionless'),
    ParamSpec(name='shell_thickness', default=0, info='Shell thickness', category='runtime_state', unit='pc'),
    ParamSpec(name='shell_nMax', default=0, info='Maximum shell density', category='runtime_state', unit='1/pc**3'),
    ParamSpec(name='shell_tauKappaRatio', default=0, info='tau_IR / kappa_IR ratio', category='runtime_state', unit='Msun/pc**2'),
    ParamSpec(name='shell_grav_r', default=np.array([]), info='Radius array for gravitational calculations', category='runtime_state', unit='pc'),
    ParamSpec(name='shell_grav_phi', default=np.array([]), info='Gravitational potential', category='runtime_state', unit='pc**2/Myr**2'),
    ParamSpec(name='shell_grav_force_m', default=np.array([]), info='Gravitational force per unit mass', category='runtime_state', unit='pc/Myr**2'),
    ParamSpec(name='shell_r_arr', default=np.array([]), info='Radial grid through ionized+neutral shell', category='runtime_state', unit='pc'),
    ParamSpec(name='shell_n_arr', default=np.array([]), info='Number density through ionized+neutral shell', category='runtime_state', unit='1/pc**3', metadata_exclude=True),
    ParamSpec(name='shell_ion_idx', default=-1, info='Last index of ionized region in shell_r/n_arr (-1 if empty)', category='runtime_state', unit='N/A'),
    ParamSpec(name='shell_mass', default=0, info='Shell mass', category='runtime_state', unit='Msun'),
    ParamSpec(name='shell_massDot', default=0, info='Shell mass accretion rate', category='runtime_state', unit='Msun/Myr'),
    ParamSpec(name='shell_interpolate_massDot', default=False, info='Use shell mass interpolation?', category='runtime_state', unit='N/A'),
    ParamSpec(name='shell_n0', default=0, info='Shell inner density (pressure balance)', category='runtime_state', unit='1/pc**3'),
    ParamSpec(name='F_grav', default=0, info='Gravitational force', category='runtime_state', unit='Msun*pc/Myr**2'),
    ParamSpec(name='F_ram', default=0, info='Ram pressure force (from Pb-Eb relation)', category='runtime_state', unit='Msun*pc/Myr**2'),
    ParamSpec(name='F_ram_wind', default=0, info='Wind ram pressure force (from SPS interpolators)', category='runtime_state', unit='Msun*pc/Myr**2'),
    ParamSpec(name='F_ram_SN', default=0, info='SN ram pressure force (from SPS interpolators)', category='runtime_state', unit='Msun*pc/Myr**2'),
    ParamSpec(name='F_ion_in', default=0, info='Inward photoionization pressure', category='runtime_state', unit='Msun*pc/Myr**2'),
    ParamSpec(name='F_HII', default=0, info='Outward HII pressure force (= P_HII * 4piR2^2)', category='runtime_state', unit='Msun*pc/Myr**2'),
    ParamSpec(name='F_rad', default=0, info='Radiation pressure', category='runtime_state', unit='Msun*pc/Myr**2'),
    ParamSpec(name='F_ISM', default=0, info='ISM pressure force (placeholder, never computed — always 0)', category='runtime_state', unit='Msun*pc/Myr**2'),
    ParamSpec(name='n_IF', default=0.0, info='Density at ionization front from shell ODE', category='runtime_state', unit='1/pc**3'),
    ParamSpec(name='n_IF_ODE', default=0.0, info='Raw ODE-derived n_IF (same as n_IF, kept for diagnostics)', category='runtime_state', unit='1/pc**3'),
    ParamSpec(name='R_IF', default=0.0, info='Radius of ionization front', category='runtime_state', unit='pc'),
    ParamSpec(name='n_IF_Str', default=0.0, info='Stroemgren ionization balance density (Lancaster+2025), sole source of P_HII', category='runtime_state', unit='1/pc**3'),
    ParamSpec(name='zeta', default=1.0, info='WBB vs PIR dominance ratio (Lancaster+2025)', category='runtime_state', unit=None),
    ParamSpec(name='P_HII', default=0.0, info='HII pressure from Stroemgren ionization balance in shell (n_IF_Str)', category='runtime_state', unit='Msun/Myr**2/pc'),
    ParamSpec(name='P_drive', default=0.0, info='Total driving pressure', category='runtime_state', unit='Msun/Myr**2/pc'),
    ParamSpec(name='P_ram', default=0.0, info='Ram pressure from freely-streaming wind', category='runtime_state', unit='Msun/Myr**2/pc'),
    ParamSpec(name='press_HII_in', default=0.0, info='Inward HII pressure at shell (confining)', category='runtime_state', unit='Msun/Myr**2/pc'),
    ParamSpec(name='bubble_LTotal', default=0, info='Total luminosity lost to cooling', category='runtime_state', unit='Msun*pc**2/Myr**3'),
    ParamSpec(name='bubble_L1Bubble', default=0, info='Cooling in bubble zone', category='runtime_state', unit='Msun*pc**2/Myr**3'),
    ParamSpec(name='bubble_L2Conduction', default=0, info='Cooling in conduction zone', category='runtime_state', unit='Msun*pc**2/Myr**3'),
    ParamSpec(name='bubble_L3Intermediate', default=0, info='Cooling in intermediate zone', category='runtime_state', unit='Msun*pc**2/Myr**3'),
    ParamSpec(name='bubble_Tavg', default=0, info='Average bubble temperature', category='runtime_state', unit='K'),
    ParamSpec(name='bubble_mass', default=0, info='Bubble mass', category='runtime_state', unit='Msun'),
    ParamSpec(name='bubble_r_Tb', default=0, info='Radius at bubble_xi_Tb * R2', category='runtime_state', unit='pc'),
    ParamSpec(name='bubble_T_r_Tb', default=0, info='Temperature at r_Tb', category='runtime_state', unit='K'),
    ParamSpec(name='bubble_r_arr', default=np.array([]), info='Bubble radius structure', category='runtime_state', unit='pc', metadata_exclude=True),
    ParamSpec(name='bubble_v_arr', default=np.array([]), info='Bubble velocity structure', category='runtime_state', unit='pc/Myr'),
    ParamSpec(name='bubble_T_arr', default=np.array([]), info='Bubble temperature structure', category='runtime_state', unit='K', metadata_exclude=True),
    ParamSpec(name='bubble_dTdr_arr', default=np.array([]), info='Bubble temperature gradient', category='runtime_state', unit='K/pc', metadata_exclude=True),
    ParamSpec(name='bubble_n_arr', default=np.array([]), info='Bubble density structure', category='runtime_state', unit='1/pc**3', metadata_exclude=True),
    ParamSpec(name='bubble_dMdtGuess', default=0, info='Bubble dM/dt guess', category='runtime_state', unit='Msun/Myr'),
    ParamSpec(name='bubble_dMdt', default=np.nan, info='Bubble mass loss rate (thermal conduction)', category='runtime_state', unit='Msun/Myr'),
    ParamSpec(name='bubble_Lgain', default=np.nan, info='Luminosity gain from winds', category='runtime_state', unit='Msun*pc**2/Myr**3'),
    ParamSpec(name='bubble_Lloss', default=np.nan, info='Luminosity loss from cooling/leaking', category='runtime_state', unit='Msun*pc**2/Myr**3'),
    ParamSpec(name='bubble_Leak', default=0, info='Leaking luminosity', category='runtime_state', unit='Msun*pc**2/Myr**3'),
    ParamSpec(name='isCollapse', default=False, info='Is cloud collapsing?', category='runtime_state', unit='N/A'),
    ParamSpec(name='isDissolved', default=False, info='Has shell dissolved?', category='runtime_state', unit='N/A'),
    ParamSpec(name='is_phiDepleted', default=False, info='Are ionising photons exhausted inside shell (phi->0)?', category='runtime_state', unit='N/A'),
    ParamSpec(name='residual_deltaT', default=0, info='Temperature residual (T1-T2)/T2', category='runtime_state', unit='dimensionless'),
    ParamSpec(name='residual_betaEdot', default=0, info='Energy rate residual', category='runtime_state', unit='dimensionless'),
    ParamSpec(name='residual_Edot1_guess', default=np.nan, info='Edot from beta', category='runtime_state', unit='Msun*pc**2/Myr**3'),
    ParamSpec(name='residual_Edot2_guess', default=np.nan, info='Edot from energy balance', category='runtime_state', unit='Msun*pc**2/Myr**3'),
    ParamSpec(name='residual_T1_guess', default=np.nan, info='T from bubble_Trgoal', category='runtime_state', unit='K'),
    ParamSpec(name='residual_T2_guess', default=np.nan, info='T from T0', category='runtime_state', unit='K'),
    ParamSpec(name='densBE_Teff', default=0, info='Effective temperature of BE sphere', category='derived_init', unit='K', exclude_from_snapshot=True, run_const=True, active_when=_active_densBE),
    ParamSpec(name='densBE_xi_arr', default=[], info='Lane-Emden xi array', category='runtime_state', unit='dimensionless', exclude_from_snapshot=True, active_when=_active_densBE),
    ParamSpec(name='densBE_u_arr', default=[], info='Lane-Emden u array', category='runtime_state', unit='dimensionless', exclude_from_snapshot=True, active_when=_active_densBE),
    ParamSpec(name='densBE_dudxi_arr', default=[], info='Lane-Emden du/dxi array', category='runtime_state', unit='dimensionless', exclude_from_snapshot=True, active_when=_active_densBE),
    ParamSpec(name='densBE_rho_rhoc_arr', default=[], info='Density contrast array', category='runtime_state', unit='dimensionless', exclude_from_snapshot=True, active_when=_active_densBE),
    ParamSpec(name='densBE_f_rho_rhoc', default=0, info='Interpolation function for density contrast', category='runtime_loaded', unit='dimensionless', exclude_from_snapshot=True, metadata_exclude=True, active_when=_active_densBE),
    ParamSpec(name='densBE_f_m', default=None, info='Lane-Emden mass interpolation function', category='runtime_loaded', unit='N/A', exclude_from_snapshot=True, metadata_exclude=True, active_when=_active_densBE),
    ParamSpec(name='densBE_xi_out', default=0, info='Dimensionless outer radius at cloud edge', category='runtime_loaded', unit='dimensionless', exclude_from_snapshot=True, metadata_exclude=True, active_when=_active_densBE),
)

REGISTRY: "OrderedDict[str, ParamSpec]" = OrderedDict(
    (s.name, s) for s in SPECS
)


def specs_by_category(*categories: Category) -> Iterable[ParamSpec]:
    cat_set = set(categories)
    return (s for s in SPECS if s.category in cat_set)


def run_const_keys() -> tuple[str, ...]:
    """Keys written once to ``metadata.json`` (constant after phase 0).

    Phase-5 drop-in replacement for
    ``src._output.run_constants.RUN_CONST_KEYS``.
    """
    return tuple(s.name for s in SPECS if s.run_const)


def metadata_exclude_keys() -> frozenset[str]:
    """Keys explicitly blocked from ``metadata.json`` (paths / loaded
    tables / empty array placeholders) even though they look constant.

    Phase-5 drop-in replacement for
    ``src._output.run_constants.METADATA_EXCLUDE``.
    """
    return frozenset(s.name for s in SPECS if s.metadata_exclude)
