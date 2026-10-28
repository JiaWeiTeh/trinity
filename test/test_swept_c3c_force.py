"""PLAN.md Step 7c-4 Stage B (S-F_body): c3c's P_HII closure evaluated on swept's shell body r_s."""
from types import SimpleNamespace

import pytest

from trinity._input.errors import ParameterFileError
from trinity._input.registry import _validate_swept_c3c_force
from trinity.bubble_structure import get_bubbleParams
from trinity.phase2_momentum.swept import c3c_phii_at_body


class _P(dict):
    """params stand-in: p[key].value and `key in p`."""

    def __init__(self, **kw):
        super().__init__({k: SimpleNamespace(value=v) for k, v in kw.items()})


def _params(**over):
    base = dict(R2=1.5, Pb=7.0, Qi=3.2e56, include_PHII=True, chi_e_shell=1.1, caseB_alpha=1.0e-3, mu_convert=1.0,
                mu_ion_shell=0.5, k_B=1.0, TShell_ion=1.0)
    base.update(over)
    return _P(**base)


def test_matches_c3c_on_the_body_and_restores_params():
    p, sp = _params(), SimpleNamespace(n_IF_Str=1.0, shell_fAbsorbedIonGas=0.8)
    fb = SimpleNamespace(Lmech_total=2.0e3, v_mech_total=50.0)
    rs = 4.0
    got = c3c_phii_at_body(p, sp, rs, fb)
    q = _params(R2=rs, Pb=get_bubbleParams.pRam(rs, fb.Lmech_total, fb.v_mech_total))
    assert got == get_bubbleParams.get_phii_c3c(q, sp)
    assert (p['R2'].value, p['Pb'].value) == (1.5, 7.0)          # borrowed state given back


def test_gates_as_c3c():
    sp = SimpleNamespace(n_IF_Str=1.0, shell_fAbsorbedIonGas=0.8)
    fb = SimpleNamespace(Lmech_total=2.0e3, v_mech_total=50.0)
    assert c3c_phii_at_body(_params(include_PHII=False), sp, 4.0, fb) == 0.0
    assert c3c_phii_at_body(_params(), SimpleNamespace(n_IF_Str=0.0, shell_fAbsorbedIonGas=0.8), 4.0, fb) == 0.0


@pytest.mark.parametrize("bad", [1, 'yes', None])
def test_validator(bad):
    with pytest.raises(ParameterFileError):
        _validate_swept_c3c_force(bad, None)
    _validate_swept_c3c_force(True, None)
    _validate_swept_c3c_force(False, None)
