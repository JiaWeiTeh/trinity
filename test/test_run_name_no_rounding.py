"""Run names must not round an axis value (MUST-FIX ruled 2026-11-14), and a sweep must refuse
two combinations that get one run name.

Before the fix ``generate_run_name`` used ``int(round(sfe * 100))``: sfe 0.001 / 0.002 / 0.005
all became ``sfe000`` and 0.015 / 0.02 both ``sfe002``. The density tags did the same:
``int(densPL_alpha)`` sent -1.8 to ``PL-1`` and ``round(densBE_Omega)`` sent 14.1 to ``BE14``.
Runs that shared a name silently overwrote one output folder.
"""
import pytest

from trinity._input.sweep_parser import (
    SweepConfig,
    generate_combinations_from_config,
    generate_run_name,
)
from trinity._output.trinity_reader import parse_simulation_params, sfe_from_token

BASE = {"mCloud": 1e5, "nCore": 1e3}  # a plausible GMC; only the sfe token is under test


def _name(sfe):
    return generate_run_name({**BASE, "sfe": sfe})


@pytest.mark.parametrize(
    "sfe, token",
    [
        (0.01, "001"), (0.02, "002"), (0.05, "005"), (0.07, "007"), (0.1, "010"), (0.15, "015"),
        (0.29, "029"), (0.3, "030"), (0.57, "057"), (0.9, "090"), (1.0, "100"),
    ],
)  # fmt: skip
def test_whole_percent_names_unchanged(sfe, token):
    # every existing grid uses whole percents: their folder names must not move
    assert _name(sfe) == f"1e5_sfe{token}_n1e3"


@pytest.mark.parametrize(
    "sfe, token", [(0.005, "000p5"), (0.015, "001p5"), (0.0025, "000p25"), (0.001, "000p1")]
)
def test_non_whole_percent_keeps_its_decimals(sfe, token):
    assert _name(sfe) == f"1e5_sfe{token}_n1e3"


def test_former_collisions_now_distinct():
    assert len({_name(s) for s in (0.001, 0.002, 0.005)}) == 3
    assert _name(0.015) != _name(0.02)


@pytest.mark.parametrize("sfe", [0.0025, 0.005, 0.01, 0.015, 0.07, 0.1, 0.29, 0.9])
def test_reader_round_trip(sfe):
    tok = parse_simulation_params(_name(sfe) + "_PL0")["sfe"]
    assert sfe_from_token(tok) == pytest.approx(sfe, rel=1e-12)


def test_duplicate_combination_is_refused():
    cfg = SweepConfig(
        base_params={},
        sweep_params={},
        tuple_params=["mCloud", "nCore", "sfe"],
        tuple_values=[[1e5, 1e3, 0.01], [1e6, 1e3, 0.01], [1e5, 1e3, 0.01]],
    )
    with pytest.raises(ValueError, match="Run-name collision"):
        list(generate_combinations_from_config(cfg))


def test_distinct_combinations_pass():
    cfg = SweepConfig(base_params=dict(BASE), sweep_params={"sfe": [0.01, 0.015, 0.02]})
    names = [n for _, n in generate_combinations_from_config(cfg)]
    assert names == ["1e5_sfe001_n1e3", "1e5_sfe001p5_n1e3", "1e5_sfe002_n1e3"]


def _profile_name(**kw):
    return generate_run_name({**BASE, "sfe": 0.01, **kw})


@pytest.mark.parametrize(
    "alpha, tag",
    [(0, "PL0"), (-1, "PL-1"), (-2.0, "PL-2"), (-0.0, "PL0"), (-1.8, "PL-1p8"), (-1.5, "PL-1p5")],
)
def test_powerlaw_tag(alpha, tag):
    assert _profile_name(dens_profile="densPL", densPL_alpha=alpha) == f"1e5_sfe001_n1e3_{tag}"


@pytest.mark.parametrize(
    "omega, tag", [(14, "BE14"), (7.0, "BE7"), (14.1, "BE14p1"), (14.4, "BE14p4")]
)
def test_bonnor_ebert_tag(omega, tag):
    assert _profile_name(dens_profile="densBE", densBE_Omega=omega) == f"1e5_sfe001_n1e3_{tag}"


def test_profile_sweep_that_collided_now_runs():
    # alpha -1 and -1.8 both used to be named PL-1 (8 runs, 6 folders)
    cfg = SweepConfig(
        base_params=dict(BASE),
        sweep_params={"sfe": [0.01, 0.1]},
        tuple_params=["dens_profile", "densPL_alpha", "densBE_Omega"],
        tuple_values=[
            ["densPL", 0, 0], ["densPL", -1, 0], ["densPL", -1.8, 0], ["densBE", 0, 14.1],
        ],
    )
    names = [n for _, n in generate_combinations_from_config(cfg)]
    assert len(names) == len(set(names)) == 8
