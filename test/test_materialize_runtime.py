"""Phase-9 materialize_runtime contract tests.

Pins ``materialize_runtime`` against the behavior the pre-Phase-9
inline Step-10 block produced for the 103 runtime/derived-init
adds.  The reconciliation tests in ``test_registry.py``
(``test_registry_covers_all_param_keys``, ``test_runtime_units_match_live``,
``test_exclude_from_snapshot_matches_live``) already prove the
post-``read_param`` param sets are byte-identical on both densBE
and densPL fixtures; these tests cover the driver directly so future
regressions surface as targeted failures rather than as opaque
live-run drift.
"""
from __future__ import annotations

import numpy as np
import pytest

from trinity._input.dictionary import DescribedItem
from trinity._input.registry import (
    REGISTRY,
    SPECS,
    materialize_runtime,
)


def _item(v):
    return DescribedItem(v)


# ---------------------------------------------------------------------------
# Skip conditions
# ---------------------------------------------------------------------------
def test_skip_already_present() -> None:
    """A spec already in params is not replaced."""
    existing = DescribedItem(42, info="user-set", ori_units="K")
    params = {"v2": existing}
    materialize_runtime(params)
    assert params["v2"] is existing  # exact same instance, untouched


def test_skip_consumed_by_specs() -> None:
    """consumed_by specs (sps_refmass + 13 sps_col_*) are owned by
    sps_path's bundle resolver and must not be materialized here."""
    params: dict = {}
    materialize_runtime(params)
    for name in ("sps_refmass", "sps_col_t", "sps_col_Qi", "sps_col_fi",
                 "sps_col_Lbol", "sps_col_Lmech_W", "sps_col_pdot_W"):
        assert name not in params, f"{name} (consumed_by) should be skipped"


def test_skip_active_when_specs() -> None:
    """active_when specs are owned by Phase 8 and must not be
    materialized here, even when their predicate would be False."""
    params: dict = {}
    materialize_runtime(params)
    for name in ("densBE_Omega", "densPL_alpha",
                 "densBE_Teff", "densBE_xi_arr", "densBE_u_arr",
                 "densBE_dudxi_arr", "densBE_rho_rhoc_arr",
                 "densBE_f_rho_rhoc", "densBE_f_m", "densBE_xi_out"):
        assert name not in params, f"{name} (active_when) should be skipped"


# ---------------------------------------------------------------------------
# Add count and metadata correctness
# ---------------------------------------------------------------------------
def _step10_entry_state(profile: str) -> dict:
    """Build the params state as it would look at Step 10 entry on a
    real read_param run -- all input/deprecated keys present, plus
    Step 6 / Step 7 / Step 8 additions / pops.  Lets us measure the
    driver's add count against the live flow."""
    inputs = [s.name for s in SPECS
              if s.category.startswith("input_") or s.category == "deprecated"]
    state = {k: _item(None) for k in inputs}
    state["mCloud_input"] = _item(0.0)                  # Step 6
    state["mCluster"] = _item(0.0)                      # Step 6
    state["chi_e"] = _item(1.2)                         # Step 6 (composition)
    state["sps_column_map"] = _item(None)               # Step 7 (sps bundle)
    if profile == "densPL":
        del state["densBE_Omega"]                       # Step 8 pop
    elif profile == "densBE":
        del state["densPL_alpha"]                       # Step 8 pop
        for k in ("densBE_Teff", "densBE_sigma", "densBE_xi_arr", "densBE_u_arr",
                  "densBE_dudxi_arr", "densBE_rho_rhoc_arr",
                  "densBE_f_rho_rhoc", "densBE_f_m", "densBE_xi_out"):
            state[k] = _item(None)                      # Step 8 adds
    return state


@pytest.mark.parametrize("profile", ["densPL", "densBE"])
def test_live_flow_adds_exactly_103(profile: str) -> None:
    """At Step 10 entry on a real read_param run (densPL or densBE),
    materialize_runtime adds exactly 103 items.  Both branches converge
    because the 8 densBE_* runtime are owned by Phase 8 either way --
    materialize_runtime sees them and skips."""
    params = _step10_entry_state(profile)
    pre = set(params)
    materialize_runtime(params)
    added = set(params) - pre
    assert len(added) == 103, f"{profile}: expected 103 adds, got {len(added)}"


def test_live_flow_add_excl_split_is_9_true_94_false() -> None:
    """Of the 103 live-flow adds, 9 carry exclude_from_snapshot=True
    (cooling cubes, sps_data/sps_f, the rcloud counter) and 94 carry
    False (time-varying simulation state).  Locks in the exact split
    from the fidelity audit."""
    params = _step10_entry_state("densPL")
    pre = set(params)
    materialize_runtime(params)
    added = set(params) - pre
    n_true = sum(1 for k in added if params[k].exclude_from_snapshot)
    n_false = sum(1 for k in added if not params[k].exclude_from_snapshot)
    assert n_true == 9
    assert n_false == 94


def test_added_items_metadata_comes_from_spec() -> None:
    """Each materialized item's info / ori_units / exclude_from_snapshot
    mirrors its registry spec exactly."""
    params: dict = {}
    materialize_runtime(params)
    for name, item in params.items():
        spec = REGISTRY[name]
        assert item.info == spec.info, f"{name}: info drift"
        assert item.ori_units == (
            spec.unit if spec.unit is not None else "N/A"
        ), f"{name}: ori_units drift"
        assert item.exclude_from_snapshot is spec.exclude_from_snapshot, (
            f"{name}: exclude_from_snapshot drift"
        )


def test_added_items_value_equals_spec_default() -> None:
    """Materialized values equal their spec defaults (nan-safe; arrays
    compared by content)."""
    params: dict = {}
    materialize_runtime(params)
    for name, item in params.items():
        spec = REGISTRY[name]
        v_got, v_exp = item.value, spec.default
        if isinstance(v_got, float) and isinstance(v_exp, float):
            if np.isnan(v_got) and np.isnan(v_exp):
                continue
        if isinstance(v_got, np.ndarray) or isinstance(v_exp, np.ndarray):
            assert np.array_equal(np.asarray(v_got), np.asarray(v_exp)), name
            continue
        assert v_got == v_exp, f"{name}: value {v_got!r} != spec {v_exp!r}"


# ---------------------------------------------------------------------------
# Mutable-default isolation
# ---------------------------------------------------------------------------
def test_mutable_defaults_are_deep_copied_between_runs() -> None:
    """np.array([]) defaults are distinct objects per run; mutating
    one must not bleed into the other or the underlying spec."""
    p1: dict = {}
    p2: dict = {}
    materialize_runtime(p1)
    materialize_runtime(p2)

    a1 = p1["initial_cloud_r_arr"].value
    a2 = p2["initial_cloud_r_arr"].value
    assert a1 is not a2  # distinct objects

    # Mutate p1's array in-place; p2 and the spec stay clean
    a1.resize(5, refcheck=False)
    a1[:] = 1.0
    assert p2["initial_cloud_r_arr"].value.shape == (0,)
    assert REGISTRY["initial_cloud_r_arr"].default.shape == (0,)


# ---------------------------------------------------------------------------
# Idempotence
# ---------------------------------------------------------------------------
def test_materialize_runtime_is_idempotent() -> None:
    """A second call is a no-op: every key the first call added is
    now present, so the skip-if-present branch fires for all."""
    params: dict = {}
    materialize_runtime(params)
    first = {k: params[k] for k in params}
    materialize_runtime(params)
    assert set(params) == set(first)
    for k, v in first.items():
        assert params[k] is v, f"{k} was replaced on second call"


# ---------------------------------------------------------------------------
# Cross-phase interaction
# ---------------------------------------------------------------------------
def test_does_not_replace_active_when_added_items() -> None:
    """If Phase 8 has already added densBE_xi_out (active_when),
    Phase 9 must not re-add or replace it."""
    pre_existing = DescribedItem(0, info="from-Phase-8", ori_units="dimensionless")
    params = {"densBE_xi_out": pre_existing}
    materialize_runtime(params)
    assert params["densBE_xi_out"] is pre_existing
