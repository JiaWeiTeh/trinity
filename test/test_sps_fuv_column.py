"""The stellar table's far-UV column (docs/dev/sps-tables/PLAN.md; P6 of PDR_DEFAULT_PLAN.md): read when declared
(sps_col_L_FUV, header-name mapping), NaN otherwise, refused when unphysical, and the PDR layer's G0 uses it in place of
fFUV_Ln x Ln. The band integrator that made the column is checked on the Draine 1978 field."""
import math
from pathlib import Path

import numpy as np
import pytest

import trinity._functions.unit_conversions as cvt
from trinity._input.read_param import read_param
from trinity._input.registry import _resolve_sps_bundle
from trinity.shell_structure.shell_structure import shell_structure_pure
from trinity.sps import read_sps
from trinity.sps.update_feedback import get_current_sps_feedback

REPO = Path(__file__).resolve().parents[1]
DEFAULT_CSV = REPO / 'lib' / 'default' / 'sps' / 'starburst99' / '1e6cluster_default.csv'
HYBRID_CSV = REPO / 'lib' / 'default' / 'sps' / 'starburst99' / '1e6cluster_default_fuv.csv'
USER_COLS = {'sps_col_t': '0 yr linear', 'sps_col_Qi': '1 1/s log', 'sps_col_fi': '2 dimensionless log',
             'sps_col_Lbol': '3 erg/s log', 'sps_col_Lmech_total': '4 erg/s log', 'sps_col_pdot_W': '5 g*cm/s^2 log',
             'sps_col_Lmech_W': '6 erg/s log'}
N0_CGS, H_NU_ERG, T_MOL = 1e3, 13.6 * 1.602176634e-12, 20.0


class _Item:
    def __init__(self, value):
        self.value = value


def _params(**over):
    p = {'ZCloud': _Item(1.0), 'SB99_rotation': _Item(1), 'sps_refmass': _Item('def_value'), 'FB_mColdWindFrac': _Item(0.0),
         'FB_mColdSNFrac': _Item(0.0), 'FB_thermCoeffWind': _Item(1.0), 'FB_thermCoeffSN': _Item(1.0),
         'FB_vSN': _Item(1.0e4 * cvt.v_kms2au)}
    for k, v in over.items():
        p[k] = _Item(v)
    return p


def _small_table(tmp_path, share, n=5):
    """The default table's first n rows with an L_FUV column = share x Ln (log10 erg/s), behind a '#' line."""
    rows = DEFAULT_CSV.read_text().splitlines()
    out = ['# a test table', rows[0] + ',L_FUV']
    for ln in rows[1:1 + n]:
        _t, _qi, fi, lbol, *_ = ln.split(',')
        ln_cgs = 10 ** float(lbol) * (1 - 10 ** float(fi))
        out.append(ln + f',{math.log10(share * ln_cgs):.6f}')
    f = tmp_path / 'fuv.csv'
    f.write_text('\n'.join(out) + '\n')
    return f


def test_default_table_has_no_far_uv_column():
    p = _params()
    p['sps_path'] = _Item(_resolve_sps_bundle('def_path', p))
    sps = read_sps.read_sps(1.0, p)
    assert len(sps) == 12 and np.all(np.isnan(sps[11]))
    for kind in ('pchip', 'cubic'):
        f = read_sps.get_interpolation(sps, kind=kind)
        assert np.isnan(f['fL_FUV'](1.0))
    p['sps_f'] = _Item(read_sps.get_interpolation(sps))
    fb = get_current_sps_feedback(1.0, p)
    assert math.isnan(fb.L_FUV) and len(fb) == 14 and math.isnan(list(fb)[13])


def test_declared_far_uv_column_is_read_scaled_and_bounded(tmp_path):
    f = _small_table(tmp_path, 0.6)
    p = _params(sps_refmass=1e6, sps_col_L_FUV='L_FUV erg/s log', **USER_COLS)
    p['sps_path'] = _Item(_resolve_sps_bundle(str(f), p))
    sps = read_sps.read_sps(0.5, p)                       # f_mass 0.5: L_FUV is mass-scaled like Ln
    Ln, L_FUV = sps[3], sps[11]
    assert np.all(np.isfinite(L_FUV)) and L_FUV[1:] == pytest.approx(0.6 * Ln[1:], rel=1e-5)
    assert L_FUV[0] == L_FUV[1]                           # the t = 0 prepend
    p0 = _params()
    p0['sps_path'] = _Item(_resolve_sps_bundle('def_path', p0))
    ref = read_sps.read_sps(0.5, p0)
    for a, b in zip(sps[:11], ref[:11]):                  # the seven old columns through the declared map = the default loader
        assert np.array_equal(a[:6], b[:6])
    bad = _small_table(tmp_path, 1.2)
    pb = _params(sps_refmass=1e6, sps_col_L_FUV='L_FUV erg/s log', **USER_COLS)
    pb['sps_path'] = _Item(_resolve_sps_bundle(str(bad), pb))
    with pytest.raises(ValueError, match='L_FUV'):
        read_sps.read_sps(1.0, pb)


@pytest.mark.skipif(not HYBRID_CSV.exists(), reason='the hybrid table is not bundled yet (PLAN Phase 3)')
def test_bundled_hybrid_table_loads_with_the_old_columns_unchanged():
    p = _params(sps_refmass=1e6, sps_col_L_FUV='L_FUV erg/s log', **USER_COLS)
    p['sps_path'] = _Item(_resolve_sps_bundle(str(HYBRID_CSV), p))
    sps = read_sps.read_sps(1.0, p)
    p0 = _params()
    p0['sps_path'] = _Item(_resolve_sps_bundle('def_path', p0))
    ref = read_sps.read_sps(1.0, p0)
    assert all(np.array_equal(a, b) for a, b in zip(sps[:11], ref[:11]))
    f = sps[11] / sps[3]
    assert np.all(np.isfinite(f)) and np.all((0.0 < f) & (f < 1.0))


def _auto_slab(m_over_m_ion, g0):
    """The step-(b) slab of test_shell_two_zone.py (_slab + _auto_slab, copied): TShell_neu 'auto', dust a millionth of the
    default, Ln set so that the front's field is about g0 Habing at R_IF ~ 1 pc with fFUV_Ln 0.67."""
    p = read_param(str(REPO / 'param' / 'simple_cluster.param'))
    g = lambda k: float(p[k].value)  # noqa: E731
    p['TShell_neu'].value, p['TShell_mol'].value, p['tauShell_mol'].value = 'auto', T_MOL, 3e-6
    p['dust_sigma'].value = 1.5e-27 * cvt.cm2pc**2
    n0 = N0_CGS * cvt.ndens_cgs2au
    Pb = n0 * g('k_B') * g('TShell_ion') * g('mu_convert') / g('mu_ion_shell')
    Qi = 4 * math.pi / 3 * g('chi_e_shell') * g('caseB_alpha') * n0**2 * 1.0**3
    m_ion = g('mu_convert') * n0 * 4 * math.pi / 3 * 1.0**3
    p['Qi'].value = Qi
    p['Li'].value = 1e-20 * Qi * H_NU_ERG * cvt.E_cgs2au
    p['Pb'].value, p['R2'].value, p['rShell'].value = Pb, 1e-3, 1e-3
    p['shell_mass'].value, p['bubble_mass'].value, p['isDissolved'].value = m_over_m_ion * m_ion, 0.0, False
    p['Ln'].value = g0 * 4 * math.pi * cvt.pc2cm**2 * 1.6e-3 / 0.67 * cvt.L_cgs2au
    return p


def test_pdr_field_uses_the_table_far_uv_when_present():
    p = _auto_slab(10.0, 1e2)
    assert math.isnan(float(p['L_FUV'].value))            # the registry default: no column
    a = shell_structure_pure(p)
    assert a.has_neutral and 50.0 < float(a.shell_G0_IF) < 200.0
    p['L_FUV'].value = 2.0 * float(p['fFUV_Ln'].value) * float(p['Ln'].value)
    b = shell_structure_pure(p)
    assert float(b.shell_G0_IF) == pytest.approx(2.0 * float(a.shell_G0_IF), rel=1e-9)


def test_band_integrator_reproduces_the_draine_field():
    from tools.sps_tables.make_sb99_table import draine_field_check
    lw_photons, habing = draine_field_check()
    assert lw_photons == pytest.approx(2.07e7, rel=0.01)  # Sternberg et al. 2014 eq. 4
    assert 1.6 < habing < 1.8                             # the Draine field is 1.7 Habing in 6-13.6 eV
