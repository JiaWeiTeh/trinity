"""Phase-8 active_when contract tests.

Pins ``apply_active_when`` against the behavior the pre-Phase-8
inline Step-8 block produced for the densBE / densPL profile
families.  The reconciliation tests in ``test_registry.py``
(``test_registry_covers_all_param_keys`` etc.) already prove the
post-``read_param`` param sets are byte-identical; these tests cover
the driver directly so future regressions surface as targeted
failures rather than as opaque live-run drift.
"""
from __future__ import annotations

import pytest

from src._input.dictionary import DescribedItem
from src._input.registry import (
    REGISTRY,
    _active_densBE,
    _active_densPL,
    apply_active_when,
)


def _item(v):
    return DescribedItem(v)


# Subset of registry names this phase manages.
_BE_RUNTIME_ADDS = (
    "densBE_Teff", "densBE_xi_arr", "densBE_u_arr", "densBE_dudxi_arr",
    "densBE_rho_rhoc_arr", "densBE_f_rho_rhoc", "densBE_f_m", "densBE_xi_out",
)


# ---------------------------------------------------------------------------
# Predicate sanity (the registry attaches these to the conditional specs)
# ---------------------------------------------------------------------------
def test_active_densBE_true_only_for_densBE() -> None:
    assert _active_densBE({"dens_profile": _item("densBE")}) is True
    assert _active_densBE({"dens_profile": _item("densPL")}) is False
    assert _active_densBE({}) is False  # missing key → not active


def test_active_densPL_true_only_for_densPL() -> None:
    assert _active_densPL({"dens_profile": _item("densPL")}) is True
    assert _active_densPL({"dens_profile": _item("densBE")}) is False
    assert _active_densPL({}) is False


# ---------------------------------------------------------------------------
# densBE branch: prune densPL_alpha, add the 8 densBE_* runtime
# ---------------------------------------------------------------------------
def test_densBE_branch_pops_densPL_alpha() -> None:
    params = {
        "dens_profile": _item("densBE"),
        "densBE_Omega": _item(14.1),
        "densPL_alpha": _item(0),
    }
    apply_active_when(params)
    assert "densPL_alpha" not in params
    assert "densBE_Omega" in params  # active+present is a no-op


def test_densBE_branch_adds_all_runtime_with_spec_metadata() -> None:
    params = {
        "dens_profile": _item("densBE"),
        "densBE_Omega": _item(14.1),
        "densPL_alpha": _item(0),
    }
    apply_active_when(params)
    for name in _BE_RUNTIME_ADDS:
        assert name in params, f"{name} should have been added"
        item = params[name]
        spec = REGISTRY[name]
        # Metadata comes from the spec, not from a hand-written DescribedItem.
        assert item.info == spec.info
        assert item.ori_units == (spec.unit if spec.unit is not None else "N/A")
        # The driver sets exclude_from_snapshot at creation; Step 9 would set
        # it for any non-time-varying key anyway, but the spec already says
        # True for all 8 so end-state and creation-state agree here.
        assert item.exclude_from_snapshot is spec.exclude_from_snapshot
        assert item.value == spec.default or (
            # np.nan / [] etc. — equality is enough for the current defaults
            item.value is spec.default
        )


# ---------------------------------------------------------------------------
# densPL branch: prune densBE_Omega, no adds
# ---------------------------------------------------------------------------
def test_densPL_branch_pops_densBE_Omega_only() -> None:
    params = {
        "dens_profile": _item("densPL"),
        "densBE_Omega": _item(14.1),
        "densPL_alpha": _item(0),
    }
    apply_active_when(params)
    assert "densBE_Omega" not in params
    assert "densPL_alpha" in params
    for name in _BE_RUNTIME_ADDS:
        assert name not in params


# ---------------------------------------------------------------------------
# Mutable-default isolation: two calls produce independent objects
# ---------------------------------------------------------------------------
def test_mutable_defaults_are_deep_copied_between_runs() -> None:
    p1 = {
        "dens_profile": _item("densBE"),
        "densBE_Omega": _item(14.1),
        "densPL_alpha": _item(0),
    }
    p2 = {
        "dens_profile": _item("densBE"),
        "densBE_Omega": _item(14.1),
        "densPL_alpha": _item(0),
    }
    apply_active_when(p1)
    apply_active_when(p2)

    # Each [] default is a distinct list object — mutating one must not
    # affect the other or the underlying registry spec.
    p1["densBE_xi_arr"].value.append("sentinel")
    assert p2["densBE_xi_arr"].value == []
    assert REGISTRY["densBE_xi_arr"].default == []


# ---------------------------------------------------------------------------
# Driver skips specs without active_when (no false adds)
# ---------------------------------------------------------------------------
def test_driver_only_touches_active_when_specs() -> None:
    """A spec with no active_when must not be added, popped, or otherwise
    touched — even if its name happens to be absent from ``params``."""
    params = {
        "dens_profile": _item("densPL"),
        "densPL_alpha": _item(-1),
    }
    snapshot = dict(params)
    apply_active_when(params)
    # densPL_alpha unchanged (active+present); no other registry key got added
    # except those gated by active_when (which here is densPL → no adds).
    assert params["densPL_alpha"] is snapshot["densPL_alpha"]
    # mCloud / mu_atom / etc. (no active_when) are never spontaneously created
    for name in ("mCloud", "mu_atom", "Lbol", "rCloud"):
        assert name not in params


# ---------------------------------------------------------------------------
# Idempotence: a second call is a no-op
# ---------------------------------------------------------------------------
def test_apply_active_when_is_idempotent() -> None:
    params = {
        "dens_profile": _item("densBE"),
        "densBE_Omega": _item(14.1),
        "densPL_alpha": _item(0),
    }
    apply_active_when(params)
    first_pass_items = {k: params[k] for k in params}
    apply_active_when(params)
    # Same key set, same DescribedItem instances (no replacement on no-op).
    assert set(params) == set(first_pass_items)
    for k, v in first_pass_items.items():
        assert params[k] is v
