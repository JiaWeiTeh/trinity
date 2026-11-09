"""path_cooling_CIE: integer presets unchanged; any other value is a file path (docs/dev/cooling-tables/PLAN.md, D-CT4).

Copied by the plan to test/test_cie_path.py. ZCloud other than 1 is refused by its validator today, so only ZCloud 1 is
exercised; the explicit-path branch does not look at ZCloud.
"""
import os
import shutil

import pytest

from trinity._input.errors import ParameterFileError
from trinity._input.read_param import read_param

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_BASE = os.path.join(_REPO, "param", "simple_cluster.param")
_CIE = os.path.join(_REPO, "lib", "default", "CIE")
_PRESETS = {
    1: "coolingCIE_1_Cloudy.dat",
    2: "coolingCIE_2_Cloudy_grains.dat",
    3: "coolingCIE_3_Gnat-Ferland2012.dat",
}


def _read(tmp_path, value=None):
    f = tmp_path / "cie.param"
    extra = "" if value is None else f"\npath_cooling_CIE    {value}\n"
    f.write_text(open(_BASE).read() + extra)
    return read_param(str(f))["path_cooling_CIE"].value


def test_default_is_curve_3(tmp_path):
    assert _read(tmp_path) == os.path.join(_CIE, _PRESETS[3])


@pytest.mark.parametrize("k", sorted(_PRESETS))
def test_integer_presets_unchanged(tmp_path, k):
    assert _read(tmp_path, k) == os.path.join(_CIE, _PRESETS[k])


def test_relative_path_from_repo_root(tmp_path):
    assert _read(tmp_path, "lib/default/CIE/coolingCIE_1_Cloudy.dat") == os.path.join(
        _CIE, _PRESETS[1]
    )


def test_absolute_path(tmp_path):
    own = tmp_path / "my_cie.dat"
    shutil.copy(os.path.join(_CIE, _PRESETS[3]), own)
    assert _read(tmp_path, str(own)) == str(own)


def test_missing_file_refused(tmp_path):
    with pytest.raises(ParameterFileError):
        _read(tmp_path, "lib/default/CIE/no_such_curve.dat")
