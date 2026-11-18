"""Seeded inputs (trinity/_input/seed.py; docs/dev/qh0-seed/PLAN.md).

A cluster given by Q(H0), L_bol, L_Halpha or mass, and gas given as a cloud with a stated size or as a uniform medium,
are rewritten into TRINITY's own inputs at load. Checked here: the cluster mass against independent arithmetic on the
bundled Starburst99 table, the seeded value met by the run's own table, the rewritten gas, the refusals, and the
property everything else rests on: a seeded .param loads to exactly the parameters of the plain .param it records.
"""
import math

import pytest

import trinity._functions.unit_conversions as cvt
from trinity._input.errors import ParameterFileError
from trinity._input.read_param import read_param
from trinity._input.sweep_runner import _validate_sweep_combination
from trinity.cloud_properties import density_profile
from trinity.cloud_properties.validate_gmc import validate_gmc_from_params
from trinity.phase0_init.get_InitCloudProp import get_InitCloudProp
from trinity.sps import read_sps

# lib/default/sps/starburst99/1e6cluster_default.csv, first row (1e4 yr; the loader holds it down to t = 0):
# log10 Q(H0) [1/s] and log10 L_bol [erg/s] of a 1e6 Msun cluster
LOGQ0, LOGL0, REFMASS = 52.732, 42.538, 1e6
QH0_PER_ERG_HA = 7.9e-42 / 1.08e-53        # Kennicutt (1998), eq. 2

CLOUD = {"mCloud": "1e5", "nCore": "1e3"}   # a uniform 1e5 Msun cloud at 1e3 cm^-3 (rCloud about 8.8 pc)


def _read(tmp_path, name="s", **keys):
    base = {"model_name": name, "log_console": "False", "path2output": str(tmp_path / name)}
    base.update({k: str(v) for k, v in keys.items()})
    par = tmp_path / f"{name}.param"
    par.write_text("\n".join(f"{k} {v}" for k, v in base.items()) + "\n")
    return read_param(str(par))


def _table_value(P, column, t):
    """The run's own feedback table at time t [Myr], scaled as main.py scales it; cgs."""
    f = read_sps.get_interpolation(read_sps.read_sps(P["mCluster"].value / P["sps_refmass"].value, P),
                                   P["sps_interp"].value)
    v = float(f[column](t))
    return v * cvt.s2Myr if column == "fQi" else v * cvt.L_au2cgs


def test_param_without_seed_keys_is_untouched(tmp_path):
    P = _read(tmp_path, sfe="0.3", **CLOUD)
    assert P["seed_canonical"].value == ""
    assert P["mCloud_input"].value == 1e5 and P["sfe"].value == 0.3
    assert P["mCluster"].value == 1e5 * 0.3 and P["mCloud"].value == 1e5 - 1e5 * 0.3


def test_Qi_at_zero_age(tmp_path):
    P = _read(tmp_path, seed_Qi="1e50", **CLOUD)
    want = REFMASS * 1e50 / 10**LOGQ0                       # 1853.5 Msun
    assert P["mCluster"].value == pytest.approx(want, rel=1e-9)
    assert P["sfe"].value == pytest.approx(want / 1e5, rel=1e-9)
    assert P["mCloud"].value == pytest.approx(1e5 - want, rel=1e-9)
    assert P["mCloud_input"].value == 1e5
    assert _table_value(P, "fQi", 0.0) == pytest.approx(1e50, rel=1e-9)


def test_Qi_at_an_age_is_met_at_that_age(tmp_path):
    young = _read(tmp_path, "young", seed_Qi="1e50", **CLOUD)
    old = _read(tmp_path, "old", seed_Qi="1e50", seed_age="5", **CLOUD)
    assert _table_value(old, "fQi", 5.0) == pytest.approx(1e50, rel=1e-9)
    # the table's Q per unit mass has halved by 5 Myr (log Q 52.732 -> 52.427 at 5.01 Myr): about twice the mass
    assert 1.8 < old["mCluster"].value / young["mCluster"].value < 2.3


def test_Lbol(tmp_path):
    P = _read(tmp_path, seed_Lbol="1e40", **CLOUD)
    assert P["mCluster"].value == pytest.approx(REFMASS * 1e40 / 10**LOGL0, rel=1e-9)
    assert _table_value(P, "fLbol", 0.0) == pytest.approx(1e40, rel=1e-9)


def test_LHa_is_Qi_through_case_B_and_the_escape_fraction(tmp_path):
    ha = _read(tmp_path, "ha", seed_LHa="1e38", seed_fesc="0.5", **CLOUD)
    qi = _read(tmp_path, "qi", seed_Qi=repr(QH0_PER_ERG_HA * 1e38 / 0.5), **CLOUD)
    assert ha["mCluster"].value == pytest.approx(qi["mCluster"].value, rel=1e-12)


def test_cluster_mass_given_directly(tmp_path):
    P = _read(tmp_path, seed_mCluster="500", **CLOUD)
    assert P["sfe"].value == 500.0 / 1e5
    assert P["mCluster"].value == pytest.approx(500.0, rel=1e-15)


def test_uniform_medium(tmp_path):
    P = _read(tmp_path, seed_Qi="1e50", seed_gas="ism", nISM="100", PISM="1e5", stop_r="100")
    assert P["nCore"].value == P["nISM"].value                               # no density step at the stand-in edge
    assert P["TCloud"].value == pytest.approx(1e5 / 100 * P["mu_mol"].value / P["mu_convert"].value, rel=1e-14)
    assert P["rCloud_max"].value == pytest.approx(202.0, rel=1e-12)
    assert validate_gmc_from_params(P).valid
    get_InitCloudProp(P)
    assert P["rCloud"].value == pytest.approx(200.0, rel=1e-6)               # 2 x stop_r: never reached
    # the medium pushes back with PISM inside the stand-in cloud, as it does outside it
    inside = density_profile.get_press_cloud(50.0, 1.0, P)
    assert inside == pytest.approx(P["PISM"].value * P["k_B"].value, rel=1e-12)
    assert P["mCluster"].value == pytest.approx(REFMASS * 1e50 / 10**LOGQ0, rel=1e-9)


@pytest.mark.parametrize("closure, r_want", [
    ({"seed_rCloud": "20"}, 20.0),
    ({"seed_Sigma": "100"}, math.sqrt(9e4 / (math.pi * 100.0))),             # 16.93 pc for the 9e4 Msun left
])
def test_cloud_size_instead_of_density(tmp_path, closure, r_want):
    P = _read(tmp_path, mCloud="1e5", sfe="0.1", **closure)
    assert validate_gmc_from_params(P).valid
    get_InitCloudProp(P)
    assert P["rCloud"].value == pytest.approx(r_want, rel=1e-9)


SEEDED = [
    dict(seed_Qi="1e50", **CLOUD),
    dict(seed_Lbol="3e40", seed_age="2", **CLOUD),
    dict(seed_LHa="1e38", seed_fesc="0.4", seed_age="3", **CLOUD),
    dict(seed_mCluster="2000", mCloud="1e5", seed_rCloud="15"),
    dict(seed_Qi="3e49", mCloud="1e5", seed_Sigma="80"),
    dict(mCloud="1e5", sfe="0.05", seed_rCloud="12"),
    dict(seed_Qi="1e50", seed_gas="ism", nISM="100", PISM="1e5", stop_r="100"),
    dict(seed_Qi="1e49", seed_age="4", seed_gas="ism", nISM="3", stop_r="300"),
]


@pytest.mark.parametrize("keys", SEEDED)
def test_seeded_param_loads_as_the_plain_param_it_records(tmp_path, keys):
    """The equivalence gate at the input boundary: every parameter the run reads is bit-identical."""
    seeded = _read(tmp_path, "seeded", **keys)
    record = dict(tok.split("=") for tok in seeded["seed_canonical"].value.split())
    assert record, "a seeded .param must record the plain keys it was rewritten to"
    plain_keys = {k: v for k, v in keys.items() if not k.startswith("seed_")}
    plain_keys.update(record)
    plain = _read(tmp_path, "plain", **plain_keys)
    assert plain["seed_canonical"].value == ""
    for k in seeded:
        if k.startswith("seed_") or k in ("model_name", "path2output"):
            continue
        a, b = seeded[k].value, plain[k].value
        if isinstance(a, float) and math.isnan(a):
            assert isinstance(b, float) and math.isnan(b), k
        elif a is None or isinstance(a, (bool, int, float, str)):
            assert a == b, f"{k}: seeded {a!r} != plain {b!r}"


@pytest.mark.parametrize("keys, why", [
    (dict(seed_Qi="1e50", seed_Lbol="1e40", **CLOUD), "at most one"),
    (dict(seed_Qi="1e50", sfe="0.1", **CLOUD), "sfe is derived"),
    (dict(seed_age="2", sfe="0.1", **CLOUD), "seed_age needs"),
    (dict(seed_mCluster="500", seed_age="2", **CLOUD), "seed_age needs"),
    (dict(seed_Qi="1e50", seed_fesc="0.5", **CLOUD), "seed_fesc needs"),
    (dict(seed_LHa="1e38", seed_fesc="1.0", **CLOUD), "seed_fesc must"),
    (dict(seed_Qi="-1e50", **CLOUD), "positive number"),
    (dict(seed_Qi="1e50", seed_age="500", **CLOUD), "no positive output"),
    (dict(seed_mCluster="2e5", **CLOUD), "not lighter than the cloud"),
    (dict(seed_Qi="1e50", seed_gas="disc", **CLOUD), "'cloud' or 'ism'"),
    (dict(seed_gas="ism", nISM="100"), "needs a seeded cluster"),
    (dict(seed_Qi="1e50", seed_gas="ism", mCloud="1e5"), "remove mCloud"),
    (dict(seed_Qi="1e50", seed_gas="ism", stop_r="None"), "finite stop_r"),
    (dict(seed_Qi="1e50", seed_gas="ism", seed_rCloud="20"), "describes a cloud"),
    (dict(mCloud="1e5", sfe="0.1", nCore="1e3", seed_rCloud="20"), "nCore is derived"),
    (dict(mCloud="1e5", sfe="0.1", seed_rCloud="20", seed_Sigma="100"), "not both"),
    (dict(mCloud="1e5", sfe="0.1", seed_rCloud="20", dens_profile="densPL", densPL_alpha="-2"), "uniform clouds"),
    (dict(mCloud="1e5", sfe="0.1", seed_rCloud="2000", nISM="1"), "below the ambient nISM"),
])
def test_refusals(tmp_path, keys, why):
    with pytest.raises(ParameterFileError, match="seeded input") as err:
        _read(tmp_path, **keys)
    assert why in str(err.value)


def test_sweep_preflight_skips_gas_it_cannot_see():
    raw = {"dens_profile": "densPL", "densPL_alpha": 0.0, "mCloud": 1e7, "nCore": 1e5}
    assert _validate_sweep_combination(raw) is not None
    assert _validate_sweep_combination({**raw, "seed_gas": "ism"}) is None
    assert _validate_sweep_combination({**raw, "seed_rCloud": 20.0}) is None
    assert _validate_sweep_combination({**raw, "seed_Sigma": 100.0}) is None
