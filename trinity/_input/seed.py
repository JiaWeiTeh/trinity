#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Seeded inputs: a run described by what is observed instead of by (mCloud, sfe, nCore).

docs/dev/qh0-seed/PLAN.md. Two independent groups of .param keys, both optional:

  the cluster   seed_Qi | seed_Lbol | seed_LHa | seed_mCluster    at most one; replaces sfe
                seed_age    the cluster age [Myr] at which seed_Qi / seed_Lbol / seed_LHa hold (default 0)
                seed_fesc   with seed_LHa: the share of the ionising photons that make no H-alpha in the region
  the gas       seed_gas    cloud (default) | ism
                seed_rCloud | seed_Sigma    cloud mode, uniform clouds: the cloud's size instead of its density

No physics module reads a seed key. apply_seed() rewrites sfe, mCloud, nCore, TCloud and rCloud_max in place once
read_param has built the parameter set, with the arithmetic read_param itself uses (Step 4 unit conversion, Step 6
mass split). A seeded run is therefore bit-identical to the plain .param recorded in params['seed_canonical']
(written to metadata.json), and a .param without a seed_* key is not touched at all.
"""

import logging
import math

import numpy as np

import trinity._functions.unit_conversions as cvt
from trinity._input.errors import ParameterFileError

logger = logging.getLogger(__name__)

CLUSTER_KEYS = ('seed_Qi', 'seed_Lbol', 'seed_LHa', 'seed_mCluster')
CLOSURE_KEYS = ('seed_rCloud', 'seed_Sigma')
SEED_KEYS = CLUSTER_KEYS + CLOSURE_KEYS + ('seed_age', 'seed_fesc', 'seed_gas')

# Ionising photons per erg of H-alpha (case B, 1e4 K). Kennicutt (1998, ARA&A 36, 189, eq. 2) gives
# SFR = 7.9e-42 L_Ha [erg/s] = 1.08e-53 Q(H0) [1/s], so Q / L_Ha = 7.9e-42 / 1.08e-53 = 7.31e11 per erg.
# ponytail: one temperature; make it a key if a TShell_ion-consistent H-alpha emissivity is ever needed.
QH0_PER_ERG_HA = 7.9e-42 / 1.08e-53

# ism mode: the uniform medium is a uniform cloud whose edge lies at this multiple of stop_r, so the shell never
# reaches it. The density is the same on both sides of that edge; the ambient pressure is carried by TCloud inside
# it and by PISM outside it (density_profile.get_press_cloud, swept.p_ext). Inside, as for any cloud, that thermal
# term acts while the shell absorbs every ionising photon; once photons leak, the gas ahead is taken as
# photoionised (get_press_ion) and the neutral medium's PISM no longer enters.
ISM_EDGE_FACTOR = 2.0

# Below this cluster mass [Msun] a fully sampled stellar table misrepresents the few massive stars present
# (da Silva, Fumagalli & Krumholz 2012, ApJ 745, 145). Paper II masks its grid at the same value.
MCLUSTER_WARN = 1e3


def _fail(msg):
    raise ParameterFileError(f"seeded input: {msg}")


def _positive(params, key):
    v = params[key].value
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v <= 0.0:
        _fail(f"{key} must be a positive number, got {v!r}.")
    return v


def _mass_from_table(params, key):
    """Cluster mass [Msun] for which the run's own stellar table gives the seeded Q(H0) or L_bol at seed_age.

    The table is read exactly as main.py reads it, for a cluster of sps_refmass (f_mass = 1); the run scales
    every mass-scaled column by mCluster / sps_refmass, so the seeded value is met at seed_age by construction.
    """
    from trinity.sps import read_sps      # local import: main.py loads the table again with the run's f_mass

    age = params['seed_age'].value
    sps = read_sps.read_sps(1.0, params)
    f = read_sps.get_interpolation(sps, params['sps_interp'].value)
    want = params[key].value
    if key == 'seed_LHa':
        # code luminosity -> erg/Myr -> ionising photons/Myr that made H-alpha; the photons that made none
        # (escaped or absorbed by dust) are added back with seed_fesc
        want = QH0_PER_ERG_HA * want / cvt.E_cgs2au / (1.0 - params['seed_fesc'].value)
    try:
        per_ref = float(f['fLbol' if key == 'seed_Lbol' else 'fQi'](age))
    except ValueError:                    # sps_interp cubic raises outside the table; pchip gives NaN
        per_ref = float('nan')
    if not (math.isfinite(per_ref) and per_ref > 0.0):
        _fail(f"the stellar table {params['sps_path'].value} has no positive output at seed_age = {age:g} Myr "
              f"(it covers 0 to {float(np.max(sps[0])):g} Myr).")
    return params['sps_refmass'].value * want / per_ref


def apply_seed(params, user_dict, merged_dict):
    """Rewrite the seeded .param keys into TRINITY's own inputs. No-op without a seed_* key in the .param.

    params       the DescribedDict read_param has built (all steps done; values in code units)
    user_dict    the keys the user's .param sets (raw values)
    merged_dict  key -> (info, unit, raw value in input units) for every default.param key
    """
    if not any(k in user_dict for k in SEED_KEYS):
        return

    canonical = {}

    def put(key, raw):
        """Store the input-unit value raw exactly as read_param Step 4 would, and record it."""
        params[key].value = raw * cvt.convert2au(merged_dict[key][1])
        canonical[key] = raw

    def val(key):
        return params[key].value

    # ---------------------------------------------------------------- what was asked for
    gas = val('seed_gas')
    if gas not in ('cloud', 'ism'):
        _fail(f"seed_gas must be 'cloud' or 'ism', got {gas!r}.")
    cluster = [k for k in CLUSTER_KEYS if val(k) is not None]
    closures = [k for k in CLOSURE_KEYS if val(k) is not None]
    if len(cluster) > 1:
        _fail(f"set at most one of {', '.join(CLUSTER_KEYS)}; got {', '.join(cluster)}.")
    if len(closures) > 1:
        _fail("set seed_rCloud or seed_Sigma, not both.")
    for k in cluster + closures:
        _positive(params, k)
    age, fesc = val('seed_age'), val('seed_fesc')
    if isinstance(age, bool) or not isinstance(age, (int, float)) or not math.isfinite(age) or age < 0.0:
        _fail(f"seed_age must be a number >= 0 [Myr], got {age!r}.")
    if isinstance(fesc, bool) or not isinstance(fesc, (int, float)) or not 0.0 <= fesc < 1.0:
        _fail(f"seed_fesc must lie in [0, 1), got {fesc!r}.")
    if 'seed_age' in user_dict and cluster not in (['seed_Qi'], ['seed_Lbol'], ['seed_LHa']):
        _fail("seed_age needs seed_Qi, seed_Lbol or seed_LHa (it is the age at which that value holds).")
    if 'seed_fesc' in user_dict and cluster != ['seed_LHa']:
        _fail("seed_fesc needs seed_LHa (it corrects the H-alpha photon count).")
    if cluster and 'sfe' in user_dict:
        _fail(f"{cluster[0]} fixes the cluster mass, so sfe is derived; remove sfe from the .param.")
    if str(user_dict.get('cooling_boost_kappa', '')).strip().lower() == 'auto':
        _fail("cooling_boost_kappa 'auto' is resolved from (mCloud, sfe, nCore) before the seed is applied; "
              "give it a number.")
    uniform = (val('dens_profile') == 'densPL' and 'densPL_alpha' in params and val('densPL_alpha') == 0)
    if closures:
        if gas == 'ism':
            _fail(f"{closures[0]} describes a cloud; it cannot be combined with seed_gas ism.")
        if 'nCore' in user_dict:
            _fail(f"{closures[0]} fixes the cloud's density, so nCore is derived; remove nCore from the .param.")
        if not uniform:
            _fail(f"{closures[0]} is implemented for uniform clouds only (dens_profile densPL, densPL_alpha 0); "
                  "for other profiles give nCore.")
    if gas == 'ism':
        if not cluster:
            _fail("seed_gas ism needs a seeded cluster (seed_Qi, seed_Lbol, seed_LHa or seed_mCluster).")
        clash = [k for k in ('mCloud', 'nCore', 'TCloud') if k in user_dict]
        if clash:
            _fail(f"seed_gas ism derives {', '.join(clash)} from nISM, PISM and stop_r; remove "
                  f"{', '.join(clash)} from the .param.")
        if not uniform:
            _fail("seed_gas ism is a uniform medium; leave dens_profile and densPL_alpha at their defaults.")
        if val('stop_r') is None:
            _fail("seed_gas ism needs a finite stop_r (the stand-in cloud's edge is placed beyond it).")
        for k, low in (('nISM', 'nISM > 0'), ('PISM', 'PISM >= 0')):
            v = merged_dict[k][2]
            if isinstance(v, bool) or not isinstance(v, (int, float)) or v < 0.0 or (k == 'nISM' and v == 0.0):
                _fail(f"seed_gas ism needs {low} (the medium's density and pressure), got {k} = {v!r}.")

    # ---------------------------------------------------------------- the cluster
    m_cl = None
    if cluster:
        m_cl = val('seed_mCluster') if cluster == ['seed_mCluster'] else _mass_from_table(params, cluster[0])

    # ---------------------------------------------------------------- the gas
    mu = val('mu_convert')                      # Msun per hydrogen nucleus
    m_in = val('mCloud_input')                  # the .param's mCloud: cloud mass before star formation [Msun]
    if gas == 'ism':
        r_edge = ISM_EDGE_FACTOR * val('stop_r')
        n_raw = merged_dict['nISM'][2]          # cm^-3 as typed (or the default): the same number goes to nCore
        put('nCore', n_raw)
        m_in = m_cl + 4.0 / 3.0 * math.pi * r_edge**3 * val('nCore') * mu
        # the medium's own pressure P/k = PISM, inside the stand-in cloud:
        # (mu_convert / mu_mol) n k_B TCloud = PISM k_B   (density_profile.get_press_cloud)
        put('TCloud', merged_dict['PISM'][2] / n_raw * val('mu_mol') / mu)
        if val('rCloud_max') < 1.01 * r_edge:
            put('rCloud_max', 1.01 * r_edge)
    if cluster and not m_cl < m_in:
        _fail(f"the seeded cluster ({m_cl:.4g} Msun) is not lighter than the cloud it formed from "
              f"(mCloud = {m_in:.4g} Msun).")
    if closures:
        m_gas = m_in - m_cl if cluster else val('mCloud')     # the gas left after star formation sets rCloud
        if closures == ['seed_rCloud']:
            r_cl = val('seed_rCloud')
        else:
            r_cl = math.sqrt(m_gas / (math.pi * val('seed_Sigma')))
        put('nCore', 3.0 * m_gas / (4.0 * math.pi * r_cl**3 * mu) * cvt.ndens_au2cgs)
        if val('nCore') < val('nISM'):
            _fail(f"a {m_gas:.4g} Msun cloud of radius {r_cl:.4g} pc has n_H = {canonical['nCore']:.4g} cm^-3, "
                  f"below the ambient nISM; lower nISM or give a smaller cloud.")

    # ---------------------------------------------------------------- the mass split (read_param Step 6)
    if cluster:
        sfe = m_cl / m_in
        params['sfe'].value = sfe
        params['mCloud_input'].value = m_in
        params['mCluster'].value = m_in * sfe
        params['mCloud'].value = m_in - params['mCluster'].value
        canonical['mCloud'] = m_in
        canonical['sfe'] = sfe

    params['seed_canonical'].value = ' '.join(f"{k}={v!r}" for k, v in canonical.items())
    logger.info(f"seeded input ({', '.join(k for k in SEED_KEYS if k in user_dict)}): cluster "
                f"{params['mCluster'].value:.6g} Msun; the plain keys this run uses: "
                f"{params['seed_canonical'].value or '(none changed)'}")
    if cluster and params['mCluster'].value < MCLUSTER_WARN:
        logger.warning(
            f"seeded cluster of {params['mCluster'].value:.4g} Msun (< {MCLUSTER_WARN:g} Msun): one or two massive "
            f"stars carry its output, and a fully sampled stellar table gives them the wrong wind-to-photon "
            f"ratio. Supply a single-star or stochastically sampled table through sps_path for such regions "
            f"(docs/dev/qh0-seed/SCIENCE.md Sect. 2.3).")
