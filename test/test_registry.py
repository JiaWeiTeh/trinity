"""Guardrails on the ParamSpec registry.

Phase 1 added the construction-time invariants (frozen, identifier
names, deprecated-needs-note); Phase 2 populated ``SPECS`` with all
187 parameters; Phase 5 swapped ``run_constants`` to derive its lists
from the registry; Phases 6/7/8 wired the ``validator`` / ``resolver``
+ ``consumed_by`` / ``active_when`` axes into ``read_param`` Steps
5/7/8.  The reconciliation tests below pin the registry against a
live ``read_param`` run (densBE and densPL fixtures) and against the
``run_constants`` module that now derives from it — they are the
byte-identical guard for every phase that touched ``params``.
"""
from __future__ import annotations

import dataclasses
from fractions import Fraction
from pathlib import Path

import pytest

from trinity._input.dictionary import DescribedDict
from trinity._input.param_spec import SENTINEL_PREFIX, ParamSpec
from trinity._input.read_param import read_param
from trinity._input.registry import (
    REGISTRY,
    SPECS,
    metadata_exclude_keys,
    run_const_keys,
)


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------
@pytest.fixture(autouse=True)
def _disable_crash_handlers(monkeypatch):
    """Stop ``DescribedDict.__init__`` registering atexit/signal hooks
    so repeated ``read_param`` calls don't leak handlers."""
    monkeypatch.setattr(
        DescribedDict, "_register_crash_handlers", lambda self: None
    )


def _parse_value(val_str: str):
    """Mirror of ``read_param``'s nested ``parse_value`` (None → bool →
    float → fraction → str).  Duplicated here because the original is
    still a closure inside ``read_param`` — Phase 10's builder will
    need a module-scope version too, at which point this can import it.
    """
    s = val_str.strip()
    if s.lower() == "none":
        return None
    if s.lower() == "true":
        return True
    if s.lower() == "false":
        return False
    try:
        return float(s)
    except ValueError:
        pass
    try:
        return float(Fraction(s))
    except (ValueError, ZeroDivisionError):
        pass
    return s


def _parse_default_param() -> dict:
    """Return {key: (raw_value, unit_or_None)} parsed from default.param,
    using the same INFO/UNIT grammar as read_param."""
    path = Path(__file__).resolve().parents[1] / "trinity" / "_input" / "default.param"
    out: dict[str, tuple] = {}
    unit = None
    for line in path.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if s.startswith("# UNIT:"):
            unit = s[len("# UNIT:"):].strip().strip("[]").strip()
            continue
        if s.startswith("#"):
            continue
        if not s:
            unit = None
            continue
        parts = s.split(None, 1)
        if len(parts) == 2:
            key, val = parts[0], parts[1].split("#")[0].strip()
            out[key] = (val, unit)
            unit = None
    return out


def _write_param(tmp_path: Path, profile: str) -> Path:
    """Write a minimal user .param selecting the given density profile,
    routing output into tmp_path so no real outputs/ dir is created.

    The companion (densPL_alpha for densPL, densBE_Omega for densBE) is
    declared explicitly because ``validate_companions`` rejects
    bare-trigger files -- exactly the silent-default trap the rule was
    added to catch."""
    body = (
        f"model_name    reg_test_{profile}\n"
        f"path2output    {tmp_path / profile}\n"
        f"dens_profile    {profile}\n"
    )
    if profile == "densBE":
        body += "densBE_Omega    14.1\n"
    elif profile == "densPL":
        body += "densPL_alpha    0\n"
    p = tmp_path / f"{profile}.param"
    p.write_text(body, encoding="utf-8")
    return p


@pytest.fixture(scope="function")
def live_keys(tmp_path) -> dict:
    """Union of params produced by a densPL run and a densBE run, plus
    the per-key live DescribedItem for value/unit/exclude assertions."""
    pl = read_param(_write_param(tmp_path, "densPL"))
    be = read_param(_write_param(tmp_path, "densBE"))
    merged = dict(pl)
    for k, v in be.items():
        merged.setdefault(k, v)
    return merged


# ---------------------------------------------------------------------------
# Structural invariants (Phase 1, still hold)
# ---------------------------------------------------------------------------
def test_registry_has_unique_names() -> None:
    names = [s.name for s in SPECS]
    assert len(names) == len(set(names)), "Duplicate spec names in SPECS"


def test_registry_dict_matches_spec_tuple() -> None:
    assert list(REGISTRY.keys()) == [s.name for s in SPECS]
    assert all(REGISTRY[s.name] is s for s in SPECS)


def test_run_const_keys_returns_tuple_of_strings() -> None:
    result = run_const_keys()
    assert isinstance(result, tuple)
    assert all(isinstance(k, str) for k in result)


def test_metadata_exclude_keys_returns_frozenset_of_strings() -> None:
    result = metadata_exclude_keys()
    assert isinstance(result, frozenset)
    assert all(isinstance(k, str) for k in result)


def test_run_const_and_metadata_exclude_disjoint() -> None:
    assert set(run_const_keys()) & metadata_exclude_keys() == set()


# ---------------------------------------------------------------------------
# Construction-time guards
# ---------------------------------------------------------------------------
def test_deprecated_category_requires_note() -> None:
    with pytest.raises(ValueError, match="deprecated_note"):
        ParamSpec(name="probe_dep", default=False, info="x", category="deprecated")


def test_run_const_and_metadata_exclude_mutually_exclusive() -> None:
    with pytest.raises(ValueError, match="mutually"):
        ParamSpec(
            name="probe_both", default=0, info="x", category="input_physical",
            run_const=True, metadata_exclude=True,
        )


def test_paramspec_name_must_be_identifier() -> None:
    with pytest.raises(ValueError, match="identifier"):
        ParamSpec(name="not valid", default=0, info="x", category="input_physical")


def test_paramspec_is_frozen() -> None:
    spec = ParamSpec(name="probe_frozen", default=0, info="x", category="input_physical")
    with pytest.raises(dataclasses.FrozenInstanceError):
        spec.name = "other"  # type: ignore[misc]


def test_sentinel_prefix_constant() -> None:
    assert SENTINEL_PREFIX == "def_"


def test_every_sentinel_default_has_resolver_or_pointer() -> None:
    """Every ``def_*`` sentinel default must either carry its own
    ``resolver`` (standalone case — ``path2output``,
    ``path_cooling_nonCIE``, ``sps_path``) or declare ``consumed_by``
    pointing at another spec that does (bulk-consumed case — ``sps_refmass``
    and the 13 ``sps_col_*`` specs delegate to ``sps_path``).

    Wired in Phase 7: the three standalone resolvers land in the registry
    and ``read_param`` Step 7 drives them via ``resolve_all``.
    """
    offenders = [
        s.name for s in SPECS
        if isinstance(s.default, str)
        and s.default.startswith(SENTINEL_PREFIX)
        and s.resolver is None
        and s.consumed_by is None
    ]
    assert not offenders, f"sentinel specs missing resolver/consumed_by: {offenders}"


def test_consumed_by_targets_exist() -> None:
    """Every ``consumed_by`` value names a real spec.  Catches typos and
    stale pointers left behind by renames."""
    bad = [
        (s.name, s.consumed_by) for s in SPECS
        if s.consumed_by is not None and s.consumed_by not in REGISTRY
    ]
    assert not bad, f"consumed_by points at unknown specs: {bad}"


def test_consumed_by_only_on_sentinel_defaults() -> None:
    """``consumed_by`` is meaningful only for sentinel (``def_*``)
    defaults — it tells the resolver-wiring step which other spec owns
    the resolution.  Catches accidental annotations on non-sentinel
    specs.
    """
    misplaced = [
        s.name for s in SPECS
        if s.consumed_by is not None
        and not (isinstance(s.default, str)
                 and s.default.startswith(SENTINEL_PREFIX))
    ]
    assert not misplaced, (
        f"consumed_by set on non-sentinel specs: {misplaced}"
    )


# ---------------------------------------------------------------------------
# Reconciliation against a live read_param run
# ---------------------------------------------------------------------------
def test_registry_covers_all_param_keys(live_keys) -> None:
    """Every key produced by a real run has a spec, and no spec names a
    key the loader never creates."""
    registry_names = set(REGISTRY)
    live = set(live_keys)
    assert live - registry_names == set(), f"params missing a spec: {sorted(live - registry_names)}"
    assert registry_names - live == set(), f"specs with no live param: {sorted(registry_names - live)}"


def test_input_defaults_match_default_param() -> None:
    """Parsed value of each input spec's default equals the parsed value
    of default.param's raw entry (tolerates 14/11 vs 1.2727… encoding)."""
    dp = _parse_default_param()
    for s in SPECS:
        if not s.category.startswith("input_") and s.category != "deprecated":
            continue
        assert s.name in dp, f"{s.name} not in default.param"
        raw, _ = dp[s.name]
        assert _parse_value(str(s.default)) == _parse_value(raw), (
            f"{s.name}: registry default {s.default!r} != default.param {raw!r}"
        )


def test_input_units_match_default_param() -> None:
    """Input spec units match default.param's # UNIT: annotation."""
    dp = _parse_default_param()
    for s in SPECS:
        if not s.category.startswith("input_") and s.category != "deprecated":
            continue
        _, unit = dp[s.name]
        assert s.unit == unit, f"{s.name}: unit {s.unit!r} != annotation {unit!r}"


def test_runtime_units_match_live(live_keys) -> None:
    """Runtime/derived spec units match the live DescribedItem.ori_units."""
    for s in SPECS:
        if s.category.startswith("input_") or s.category == "deprecated":
            continue
        assert s.unit == live_keys[s.name].ori_units, (
            f"{s.name}: unit {s.unit!r} != live {live_keys[s.name].ori_units!r}"
        )


def test_exclude_from_snapshot_matches_live(live_keys) -> None:
    for s in SPECS:
        assert s.exclude_from_snapshot == live_keys[s.name].exclude_from_snapshot, (
            f"{s.name}: exclude_from_snapshot {s.exclude_from_snapshot} != live "
            f"{live_keys[s.name].exclude_from_snapshot}"
        )


# ---------------------------------------------------------------------------
# run_constants now DERIVES from the registry (Phase 5)
# ---------------------------------------------------------------------------
def test_run_constants_module_derives_from_registry() -> None:
    """trinity._output.run_constants re-exports the registry derivation
    verbatim (identity), not a parallel hand-list."""
    from trinity._output.run_constants import METADATA_EXCLUDE, RUN_CONST_KEYS

    assert tuple(RUN_CONST_KEYS) == run_const_keys()
    assert METADATA_EXCLUDE == metadata_exclude_keys()


def test_no_stale_keys_in_run_constants(live_keys) -> None:
    """The four legacy stale entries are gone now that the lists derive
    from the registry, and none names a live param (proving their
    removal didn't drop a real key)."""
    from trinity._output.run_constants import METADATA_EXCLUDE, RUN_CONST_KEYS

    stale = {"expansionBeyondCloud", "SB99_data", "SB99f", "path_sps"}
    assert stale.isdisjoint(RUN_CONST_KEYS)
    assert stale.isdisjoint(METADATA_EXCLUDE)
    assert stale.isdisjoint(live_keys)


# ---------------------------------------------------------------------------
# Per-field consistency
# ---------------------------------------------------------------------------


def test_active_when_only_on_conditional_specs() -> None:
    """Only the densBE/densPL profile-conditional keys carry active_when."""
    with_active = {s.name for s in SPECS if s.active_when is not None}
    expected = {
        "densBE_Omega", "densPL_alpha",
        "densBE_Teff", "densBE_sigma",
        "densBE_xi_arr", "densBE_u_arr", "densBE_dudxi_arr",
        "densBE_rho_rhoc_arr", "densBE_f_rho_rhoc", "densBE_f_m", "densBE_xi_out",
    }
    assert with_active == expected


def test_runtime_categories_match_reader_table() -> None:
    """Pins the runtime-bucket split against the trinity_reader Snapshot
    grouping (Table E.2): a wrong move between buckets — or a regression
    that collapses runtime_state back — fails loudly here."""
    by_cat: dict[str, set[str]] = {}
    for s in SPECS:
        by_cat.setdefault(s.category, set()).add(s.name)

    assert "runtime_state" not in by_cat, (
        "runtime_state was removed in Phase 2; specs should now sit in "
        "a physical-role bucket (runtime_bubble, runtime_shell, …)"
    )
    # Spot-check the four answers locked during the recategorisation
    # review (c_sound→shell, R_IF→radii kept split from n_IF*,
    # bubble_Leak→bubble_cooling, zeta removed).
    assert "c_sound" in by_cat["runtime_shell"]
    assert "R_IF" in by_cat["runtime_radii"]
    assert {"n_IF", "n_IF_ODE", "n_IF_Str"} <= by_cat["runtime_shell"]
    assert "bubble_Leak" in by_cat["runtime_bubble_cooling"]
    assert "zeta" not in {s.name for s in SPECS}
