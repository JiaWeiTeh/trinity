"""Phase-7 resolver contract tests.

Pins the three sentinel resolvers wired into the registry
(``path2output``, ``path_cooling_nonCIE``, ``sps_path``) and the
``resolve_all`` driver against the behavior the pre-Phase-7 inline
Step-7 block produced.  These catch drift if a resolver is rewritten
or a spec loses its ``resolver=`` / ``consumed_by=`` attachment.
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest

from src._input.errors import ParameterFileError
from src._input.registry import (
    REGISTRY,
    _REPO_ROOT,
    _resolve_path2output,
    _resolve_path_cooling_nonCIE,
    _resolve_sps_bundle,
    resolve_all,
)
from src.sps import sps_columns


class _Item:
    """Minimal stand-in for DescribedItem so resolvers can read/write
    ``params[key].value`` without a real run."""
    def __init__(self, value):
        self.value = value


def _params(**kwargs) -> dict:
    return {k: _Item(v) for k, v in kwargs.items()}


# ---------------------------------------------------------------------------
# Spec attachment
# ---------------------------------------------------------------------------
def test_path2output_spec_has_resolver() -> None:
    assert REGISTRY["path2output"].resolver is _resolve_path2output


def test_path_cooling_nonCIE_spec_has_resolver() -> None:
    assert REGISTRY["path_cooling_nonCIE"].resolver is _resolve_path_cooling_nonCIE


def test_sps_path_spec_has_resolver() -> None:
    assert REGISTRY["sps_path"].resolver is _resolve_sps_bundle


def test_sps_refmass_delegates_to_sps_path() -> None:
    """Option A: sps_refmass is owned by sps_path's bundle resolver."""
    assert REGISTRY["sps_refmass"].resolver is None
    assert REGISTRY["sps_refmass"].consumed_by == "sps_path"


# ---------------------------------------------------------------------------
# path2output
# ---------------------------------------------------------------------------
def test_path2output_default_uses_cwd_outputs_modelname(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    params = _params(model_name="mymodel")
    out = _resolve_path2output("def_dir", params)
    assert out == os.path.join(str(tmp_path), "outputs", "mymodel")
    assert Path(out).is_dir()


def test_path2output_user_path_taken_as_is_and_created(tmp_path) -> None:
    target = tmp_path / "custom_out"
    out = _resolve_path2output(str(target), _params(model_name="x"))
    assert out == str(target)
    assert target.is_dir()


# ---------------------------------------------------------------------------
# path_cooling_nonCIE
# ---------------------------------------------------------------------------
def test_path_cooling_nonCIE_default_resolves_to_opiate_folder() -> None:
    out = _resolve_path_cooling_nonCIE("def_dir", {})
    assert out == str(_REPO_ROOT / "lib" / "default" / "opiate") + os.sep


def test_path_cooling_nonCIE_user_path_taken_as_is_and_created(tmp_path) -> None:
    target = tmp_path / "cool"
    out = _resolve_path_cooling_nonCIE(str(target), {})
    assert out == str(target)
    assert target.is_dir()


# ---------------------------------------------------------------------------
# sps_path bundle resolver (default branch)
# ---------------------------------------------------------------------------
def _default_sps_params(**overrides) -> dict:
    base = dict(ZCloud=1.0, SB99_rotation=1, sps_refmass="def_value")
    base.update(overrides)
    return _params(**base)


def test_sps_bundle_default_resolves_path_refmass_and_columnmap() -> None:
    params = _default_sps_params()
    out = _resolve_sps_bundle("def_path", params)
    assert out.endswith("1e6cluster_default.csv")
    assert params["sps_refmass"].value == 1e6
    assert params["sps_column_map"].value == sps_columns.DEFAULT_SPS_COLUMN_MAP


def test_sps_bundle_default_rejects_nonsolar_ZCloud() -> None:
    params = _default_sps_params(ZCloud=0.5)
    with pytest.raises(ValueError, match="ZCloud=0.5 is not supported"):
        _resolve_sps_bundle("def_path", params)


def test_sps_bundle_default_rejects_norot() -> None:
    params = _default_sps_params(SB99_rotation=0)
    with pytest.raises(ValueError, match="SB99_rotation=0 is not supported"):
        _resolve_sps_bundle("def_path", params)


# ---------------------------------------------------------------------------
# sps_path bundle resolver (user branch)
# ---------------------------------------------------------------------------
def _user_sps_params(csv: Path, *, refmass="def_value") -> dict:
    """A params dict declaring the minimum required sps_col_* so the
    column map validates; lets us reach the sps_refmass check."""
    cols = {
        "sps_col_t": "0 Myr linear",
        "sps_col_Qi": "1 1/Myr linear",
        "sps_col_fi": "2 dimensionless linear",
        "sps_col_Lbol": "3 erg/s linear",
        "sps_col_Lmech_W": "4 erg/s linear",
        "sps_col_pdot_W": "5 g*cm/s^2 linear",
        "sps_col_Lmech_total": "6 erg/s linear",
    }
    return _params(ZCloud=1.0, SB99_rotation=1, sps_refmass=refmass, **cols)


def _write_csv(tmp_path: Path) -> Path:
    csv = tmp_path / "my.csv"
    csv.write_text(
        "t Qi fi Lbol Lmech_W pdot_W Lmech_total\n"
        "1 2 3 4 5 6 7\n2 3 4 5 6 7 8\n",
        encoding="utf-8",
    )
    return csv


def test_sps_bundle_user_path_without_refmass_raises(tmp_path) -> None:
    csv = _write_csv(tmp_path)
    params = _user_sps_params(csv, refmass="def_value")
    with pytest.raises(ParameterFileError, match="sps_refmass is required"):
        _resolve_sps_bundle(str(csv), params)


def test_sps_bundle_user_path_with_refmass_builds_user_map(tmp_path) -> None:
    csv = _write_csv(tmp_path)
    params = _user_sps_params(csv, refmass=5e5)
    out = _resolve_sps_bundle(str(csv), params)
    assert out == str(csv)
    assert params["sps_refmass"].value == 5e5
    assert set(params["sps_column_map"].value.keys()) == {
        "t", "Qi", "fi", "Lbol", "Lmech_W", "pdot_W", "Lmech_total"
    }


# ---------------------------------------------------------------------------
# resolve_all driver
# ---------------------------------------------------------------------------
def test_resolve_all_skips_missing_keys() -> None:
    """A spec with a resolver whose key isn't in params is a no-op,
    not a KeyError."""
    resolve_all({})  # empty params, must not raise


def test_resolve_all_skips_resolverless_specs(tmp_path, monkeypatch) -> None:
    """Keys without a resolver (e.g. model_name) pass through untouched."""
    monkeypatch.chdir(tmp_path)
    params = _params(model_name="keepme", path2output="def_dir")
    resolve_all(params)
    assert params["model_name"].value == "keepme"  # untouched
    assert params["path2output"].value.endswith(os.path.join("outputs", "keepme"))
