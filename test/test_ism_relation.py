"""The ambient ISM's sigma_eff relation (sigma_eff_ISM; docs/dev/ism-sigma-eff/PLAN.md): read_param.derive_ism, its
place in read_param before the unit conversion and before seed.py, the sweep pre-flight check, and the held log
records that let read_param's lines reach trinity.log (logging_setup.HeldRecords, used by run.py).

Rulings: D-a (PISM 0 given alone keeps nISM), D-b (neither given: the defaults), D-c (one key, default 'fit'), D-e (the
origin in metadata.json), 2026-11-24; D-d (an nISM alone keeps PISM 0 unless the .param sets sigma_eff_ISM), the range
flag and the three-figure constants 8.74 / 0.123, 2026-11-26.
"""
import pytest

import trinity._functions.unit_conversions as cvt
from trinity._input.errors import ParameterFileError
from trinity._input.read_param import SIGMA_EFF_FIT_BETA, SIGMA_EFF_FIT_KMS, derive_ism, read_param
from trinity._input.registry import SPECS
from trinity._input.sweep_runner import _validate_sweep_combination

MU_H, K_B = 1.4, 1.380649e-16
T_KMS = MU_H * cvt.M_H_CGS * 1.0e10 / K_B          # mu_H m_H (1 km/s)^2 / k_B [K], as derive_ism
N_AU = cvt.convert2au('cm**-3')


def _read(tmp_path, name, **keys):
    base = {"model_name": name, "log_console": "False", "path2output": str(tmp_path / name)}
    base.update({k: str(v) for k, v in keys.items()})
    par = tmp_path / f"{name}.param"
    par.write_text("\n".join(f"{k} {v}" for k, v in base.items()) + "\n")
    return read_param(str(par))


def test_fit_sets_nISM_from_PISM_and_back():
    for P in (1e3, 1e4, 1e6, 5e9):
        P1, n, origin, _ = derive_ism({'PISM'}, P, 1.0, 'fit', MU_H, K_B)
        sigma = SIGMA_EFF_FIT_KMS * (P / 1e4) ** SIGMA_EFF_FIT_BETA
        assert P1 == P and n == pytest.approx(P / (T_KMS * sigma ** 2), rel=1e-14)
        assert origin.startswith('nISM from PISM: ')
        P2, n2, origin2, _ = derive_ism({'nISM', 'sigma_eff_ISM'}, 0.0, n, 'fit', MU_H, K_B)
        assert n2 == n and P2 == pytest.approx(P, rel=1e-12) and origin2.startswith('PISM from nISM: ')
    # 8.74 / 0.123 (fit C at three figures, PLAN_BR_breakout.md section 13): 0.771 cm^-3 at 1e4, 1.53e4 at 5e9 K cm^-3
    assert derive_ism({'PISM'}, 1e4, 1.0, 'fit', MU_H, K_B)[1] == pytest.approx(0.77142, rel=1e-4)
    assert derive_ism({'PISM'}, 5e9, 1.0, 'fit', MU_H, K_B)[1] == pytest.approx(1.5287e4, rel=1e-4)


def test_cases_that_leave_both_keys_as_they_are():
    assert derive_ism({'PISM', 'nISM'}, 1e5, 1.0, 'fit', MU_H, K_B) == (1e5, 1.0, 'given', False)
    assert derive_ism(set(), 0.0, 1.0, 'fit', MU_H, K_B) == (0.0, 1.0, 'default', False)
    assert derive_ism({'nISM'}, 0.0, 10.0, 'fit', MU_H, K_B) == (0.0, 10.0, 'nISM only', False)            # D-d
    assert derive_ism({'PISM', 'sigma_eff_ISM'}, 1e5, 1.0, 'off', MU_H, K_B) == (1e5, 1.0, 'off', False)
    assert derive_ism({'PISM'}, 0.0, 1.0, 'fit', MU_H, K_B) == (0.0, 1.0, 'PISM 0', False)                  # D-a
    assert derive_ism({'nISM', 'sigma_eff_ISM'}, 0.0, 0.0, 'fit', MU_H, K_B) == (0.0, 0.0, 'nISM 0', False)


def test_a_constant_sigma_eff():
    _, n, origin, flag = derive_ism({'PISM'}, 1e6, 1.0, 10.0, MU_H, K_B)
    assert n == 1e6 / (T_KMS * 100.0) and flag is False and origin.endswith('km/s (constant)')
    P2, n2, _, _ = derive_ism({'nISM', 'sigma_eff_ISM'}, 0.0, n, 10.0, MU_H, K_B)
    assert n2 == n and P2 == pytest.approx(1e6, rel=1e-14)


def test_the_fit_outside_its_anchors_is_flagged(tmp_path, caplog):
    for P, out in ((7.7e2, True), (7.8e2, False), (1e6, False), (6.7e7, False), (1e8, True), (5e9, True)):
        _, _, origin, flag = derive_ism({'PISM'}, P, 1.0, 'fit', MU_H, K_B)
        assert flag is out and origin.endswith("anchors 7.8e2-6.7e7 K cm^-3") is out
    assert derive_ism({'PISM'}, 5e9, 1.0, 20.0, MU_H, K_B)[3] is False      # a constant sigma_eff has no anchors
    # read_param: the flag in ISM_relation (metadata.json) and one WARNING at load
    with caplog.at_level('WARNING', logger='trinity._input.read_param'):
        P = _read(tmp_path, "x", mCloud="1e5", sfe="0.3", PISM="1e8")
    assert P['ISM_relation'].value.endswith("anchors 7.8e2-6.7e7 K cm^-3")
    assert sum('outside its anchors' in r.getMessage() for r in caplog.records if r.levelname == 'WARNING') == 1


def test_a_bad_sigma_eff_ISM_is_refused():
    for bad in ('fits', -1.0, 0.0, True, float('nan'), None):
        with pytest.raises(ParameterFileError, match='sigma_eff_ISM must be'):
            derive_ism({'PISM'}, 1e5, 1.0, bad, MU_H, K_B)


def test_a_bad_value_to_derive_from_is_refused():
    for typed, P, n in (({'PISM'}, -1.0, 1.0), ({'PISM'}, float('inf'), 1.0), ({'PISM'}, None, 1.0),
                        ({'nISM', 'sigma_eff_ISM'}, 0.0, -2.0)):
        with pytest.raises(ParameterFileError, match='finite number >= 0'):
            derive_ism(typed, P, n, 'fit', MU_H, K_B)


def test_read_param_pism_alone_equals_nISM_typed_by_hand(tmp_path):
    A = _read(tmp_path, "a", mCloud="1e5", sfe="0.3", PISM="1e5")
    origin = A['ISM_relation'].value
    assert origin.startswith('nISM from PISM: nISM=')
    n_typed = origin.split('nISM=')[1].split(' ')[0]                     # repr: the float round-trips exactly
    B = _read(tmp_path, "b", mCloud="1e5", sfe="0.3", PISM="1e5", nISM=n_typed)
    assert B['ISM_relation'].value == 'given'
    assert A['nISM'].value == B['nISM'].value == float(n_typed) * N_AU
    for k in sorted(set(A) - {'ISM_relation'}):
        a, b = A[k].value, B[k].value
        if isinstance(a, (bool, int, float)) or a is None:
            assert a == b or (a != a and b != b), k


def test_read_param_nISM_alone_keeps_PISM_unless_asked(tmp_path):
    D = _read(tmp_path, "d", mCloud="1e5", sfe="0.3")
    assert D['ISM_relation'].value == 'default' and D['PISM'].value == 0.0 and D['nISM'].value == 1.0 * N_AU
    A = _read(tmp_path, "n", mCloud="1e5", sfe="0.3", nISM="10")
    assert A['ISM_relation'].value == 'nISM only' and A['PISM'].value == 0.0
    B = _read(tmp_path, "m", mCloud="1e5", sfe="0.3", nISM="10", sigma_eff_ISM="fit")
    assert B['ISM_relation'].value.startswith('PISM from nISM: ')
    assert B['PISM'].value == pytest.approx(2.9905e5 * cvt.convert2au('K * cm**-3'), rel=1e-4)


def test_seed_ism_takes_the_derived_density(tmp_path):
    P = _read(tmp_path, "s", seed_Qi="1e50", seed_gas="ism", PISM="1e5", stop_r="100")
    n = derive_ism({'PISM'}, 1e5, 1.0, 'fit', MU_H, K_B)[1]
    assert P['nISM'].value == n * N_AU
    assert P['nCore'].value == P['nISM'].value               # seed.py: the stand-in cloud is the medium itself


def _ism_errors(result):
    return [e for e in result.errors if 'ISM density' in e]


def test_sweep_preflight_sees_the_derived_density():
    cell = {'mCloud': 1e5, 'sfe': 0.1, 'nCore': 300.0, 'dens_profile': 'densPL', 'densPL_alpha': 0.0}
    errs = _ism_errors
    assert errs(_validate_sweep_combination(dict(cell, PISM=1e5))) == []               # nISM 4.4 < 300
    assert len(errs(_validate_sweep_combination(dict(cell, PISM=1e8)))) == 1           # nISM 800 > 300
    assert errs(_validate_sweep_combination(dict(cell, PISM=1e8, nISM=1.0))) == []     # both given: as given


def test_read_param_log_lines_are_held_and_replayed(tmp_path):
    """run.py holds read_param's records (before trinity.log exists) and replays them after setup_logging."""
    import logging
    from trinity._functions.logging_setup import HeldRecords
    seen = []

    class Grab(logging.Handler):
        def emit(self, record):
            seen.append((record.levelname, record.getMessage()))

    root, grab = logging.getLogger(), Grab(logging.INFO)
    level = root.level
    root.addHandler(grab)
    root.setLevel(logging.INFO)
    try:
        held = HeldRecords()
        with held:
            _read(tmp_path, "h", mCloud="1e5", sfe="0.3", PISM="1e5")
        assert not any('sigma_eff_ISM' in m for _, m in seen)                    # held, nothing emitted yet
        held.replay()
        assert any(lv == 'INFO' and m.startswith('sigma_eff_ISM: ISM_relation = nISM from PISM') for lv, m in seen)
        with pytest.raises(RuntimeError):                                        # an error replays at once
            with HeldRecords():
                logging.getLogger('trinity._input.read_param').warning('held until the error')
                raise RuntimeError
        assert ('WARNING', 'held until the error') in seen
    finally:
        root.removeHandler(grab)
        root.setLevel(level)


def test_the_keys():
    spec = {s.name: s for s in SPECS}
    assert spec['sigma_eff_ISM'].default == 'fit' and spec['sigma_eff_ISM'].run_const
    assert spec['ISM_relation'].category == 'derived_init' and spec['ISM_relation'].run_const
