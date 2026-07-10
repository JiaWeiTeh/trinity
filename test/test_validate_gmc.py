import math

import pytest

from trinity._functions import unit_conversions as cvt
from trinity.cloud_properties.bonnorEbertSphere import solve_lane_emden
from trinity.cloud_properties.validate_gmc import R_CLOUD_MAX, validate_gmc_params


M_CLOUD = 1.0e5
N_CORE = 1.0e3 * cvt.ndens_cgs2au
N_ISM = 1.0 * cvt.ndens_cgs2au
MU_CONVERT = 1.4 * cvt.convert2au("m_H")


@pytest.fixture(scope="module")
def lane_emden_solution():
    return solve_lane_emden()


def _params_for(profile, lane_emden_solution):
    params = {
        "mCloud": M_CLOUD,
        "nCore": N_CORE,
        "mu": MU_CONVERT,
        "nISM": N_ISM,
        "r_max": R_CLOUD_MAX,
    }
    if profile == "densPL":
        params.update({"dens_profile": "densPL", "alpha": -1.0, "rCore": 1.0})
    elif profile == "densBE":
        params.update(
            {
                "dens_profile": "densBE",
                "Omega": 4.0,
                "gamma": 5.0 / 3.0,
                "lane_emden_solution": lane_emden_solution,
            }
        )
    else:
        raise ValueError(profile)
    return params


@pytest.mark.parametrize("profile", ["densPL", "densBE"])
def test_validate_gmc_accepts_plausible_cloud(profile, lane_emden_solution):
    result = validate_gmc_params(**_params_for(profile, lane_emden_solution))

    assert result.valid
    assert result.errors == []
    assert math.isfinite(result.rCloud)
    assert 0.0 < result.rCloud < R_CLOUD_MAX
    assert result.nEdge >= N_ISM
    assert result.mass_error == pytest.approx(0.0, abs=1.0e-12)


@pytest.mark.parametrize("profile", ["densPL", "densBE"])
def test_validate_gmc_rejects_cloud_larger_than_rmax(profile, lane_emden_solution):
    params = _params_for(profile, lane_emden_solution)
    accepted = validate_gmc_params(**params)
    assert accepted.valid

    params["r_max"] = accepted.rCloud * 0.9
    rejected = validate_gmc_params(**params)

    assert not rejected.valid
    assert rejected.rCloud == pytest.approx(accepted.rCloud)
    assert rejected.nEdge == pytest.approx(accepted.nEdge)
    assert rejected.mass_error == pytest.approx(0.0, abs=1.0e-12)
    assert any("exceeds maximum GMC size" in error for error in rejected.errors)
