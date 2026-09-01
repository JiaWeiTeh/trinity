#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Checks for the bubble end-fate SSOT (``tools/bubble_fate.py``).

The point of these is that the fate table is TOTAL and UNAMBIGUOUS: every
combination of the four axes lands on exactly one report, no combination
falls through, and the specific mislabellings that motivated the rewrite
(the ``isCollapse`` substring bug, ``energy_collapsed`` read as a shell
collapse, "still expanding" read as "dispersed") cannot come back.
"""

import itertools
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

import bubble_fate as BF  # noqa: E402

# --------------------------------------------------------------------------
# The table is total and unambiguous
# --------------------------------------------------------------------------


def _axis_states():
    return itertools.product(BF.MOTIONS, BF.CLEARED, BF.SHELL, BF.STOPS)


def test_every_axis_combination_yields_exactly_one_report():
    for motion, cleared, shell, stop in _axis_states():
        hits = [n for n, _w, p in BF.REPORT_TABLE if p(motion, cleared, shell, stop)]
        assert hits, "no rule fires for %s" % ((motion, cleared, shell, stop),)
        assert hits[0] in BF.REPORT_BY_NAME


def test_every_report_name_is_documented_ordered_and_coloured():
    names = {r.name for r in BF.REPORTS}
    assert names == set(BF.REPORT_ORDER), "REPORT_ORDER and REPORTS disagree"
    assert names == set(BF.REPORT_COLORS), "REPORT_COLORS and REPORTS disagree"
    # S3 of V2_AUDIT: 'unresolved' vanishing from the vocabulary is what let
    # 18.3% of the grid leak into all-run denominators.
    assert "unresolved" in BF.REPORT_ORDER
    assert "unresolved" in BF.REPORT_COLORS


def test_every_rule_in_the_table_is_reachable():
    reached = set()
    for state in _axis_states():
        for name, _w, pred in BF.REPORT_TABLE:
            if pred(*state):
                reached.add(name)
                break
    assert reached == set(BF.REPORT_ORDER), "unreachable report(s): %s" % (
        set(BF.REPORT_ORDER) - reached,
    )


def test_every_cause_maps_to_a_known_stop_class():
    for c in BF.CAUSES:
        assert c.stop in BF.STOPS, "%s has stop=%r" % (c.outcome, c.stop)
        assert c.code == BF.CAUSE_BY_CODE[c.code].code
        assert c.sites, "%s records no file:line" % c.outcome


# --------------------------------------------------------------------------
# The specific mislabellings this module exists to prevent
# --------------------------------------------------------------------------


def _row(**kw):
    base = dict(
        v2_final=10.0,
        R2_final=50.0,
        R2_max=50.0,
        t_final=5.0,
        rCloud=20.0,
        isCollapse=False,
        isDissolved=False,
        broke_out=False,
        end_reason="stopping_time",
    )
    base.update(kw)
    return base


def test_large_radius_with_isCollapse_is_not_a_collapse():
    """The 13,993-run defect: max_radius_event sets isCollapse=True because
    its name contains 'radius'. A maximally expanded, outward-moving shell
    must never read as collapsed."""
    v = BF.classify(
        _row(
            isCollapse=True, end_reason="large_radius", v2_final=100.0, R2_final=500.0, R2_max=500.0
        )
    )
    assert v["motion"] == "expanding"
    assert v["report"] == "radius_capped"
    assert v["report"] not in ("recollapsed", "recaptured", "collapsing")
    assert v["censored"] is True


def test_energy_collapsed_is_unresolved_not_recollapsed():
    """The 7,968-run defect: 'energy_collapsed' is about Eb, not the shell.
    99.6% were still moving outward when it fired."""
    v = BF.classify(
        _row(
            isCollapse=True,
            end_reason="energy_collapsed",
            v2_final=654.0,
            t_final=0.0015,
            R2_final=0.4,
            R2_max=0.4,
        )
    )
    assert v["stop"] == "numerical"
    assert v["report"] == "unresolved"
    assert v["cloud_outcome"] == "undetermined"


def test_velocity_runaway_is_unresolved():
    """A solver death is never a fate -- but when the shell died moving inward we
    know the direction, and `unresolved_infall` records that without promoting it.

    Both branches must stay censored and both must leave the cloud's outcome
    undetermined: knowing which way it was going is not knowing where it stopped.
    """
    inward = BF.classify(_row(end_reason="velocity_runaway", v2_final=-500.0))
    assert inward["report"] == "unresolved_infall"
    assert inward["stop"] == "numerical"
    assert inward["censored"] is True
    assert inward["cloud_outcome"] == "undetermined"

    # 224 velocity_runaway runs on the v2 grid ended with v2_final > 0 despite the
    # event being inward-only, so the split keys on motion, not on the cause token.
    outward = BF.classify(_row(end_reason="velocity_runaway", v2_final=373.9))
    assert outward["report"] == "unresolved"
    assert outward["cloud_outcome"] == "undetermined"


def test_still_expanding_is_not_cleared():
    """'expanding' is a censoring; 'breakout' is a result. They must not
    share a word."""
    inside = BF.classify(_row(v2_final=20.0, R2_max=10.0, rCloud=20.0))
    assert inside["report"] == "expanding"
    assert inside["cloud_outcome"] == "retained"
    assert inside["censored"] is True

    outside = BF.classify(_row(v2_final=20.0, R2_max=90.0, R2_final=90.0, rCloud=20.0))
    assert outside["report"] == "breakout"
    assert outside["cloud_outcome"] == "cleared"


def test_collapsed_and_cleared_is_recaptured_not_either_one():
    """The case the whole rewrite is for: both 'collapsing' and 'dispersed'
    are true, in that order."""
    v = BF.classify(
        _row(
            end_reason="shell_collapsed",
            isCollapse=True,
            v2_final=-30.0,
            R2_final=1.0,
            R2_max=90.0,
            rCloud=20.0,
        )
    )
    assert v["cleared"] == "yes"
    assert v["motion"] == "contracting"
    assert v["report"] == "recaptured"
    assert v["cloud_outcome"] == "cleared_then_recaptured"

    never = BF.classify(
        _row(
            end_reason="shell_collapsed",
            isCollapse=True,
            v2_final=-30.0,
            R2_final=1.0,
            R2_max=10.0,
            rCloud=20.0,
        )
    )
    assert never["report"] == "recollapsed"
    assert never["cloud_outcome"] == "retained"


def test_stall_is_scale_free():
    """Same dimensionless state at two very different scales must classify
    the same way -- that is the whole reason for not using a km/s cut."""
    small = BF.classify(_row(R2_final=5.0, R2_max=5.0, t_final=10.0, v2_final=0.01, rCloud=20.0))
    big = BF.classify(_row(R2_final=500.0, R2_max=500.0, t_final=10.0, v2_final=1.0, rCloud=20.0))
    assert small["motion"] == "stalled"
    assert big["motion"] == "stalled"
    # ... and a bubble moving 10x faster at the same scale is NOT stalled
    moving = BF.classify(_row(R2_final=5.0, R2_max=5.0, t_final=10.0, v2_final=1.0, rCloud=20.0))
    assert moving["motion"] == "expanding"


def test_stall_inside_and_outside_the_cloud_are_different_reports():
    inside = BF.classify(_row(R2_final=5.0, R2_max=5.0, t_final=10.0, v2_final=0.001, rCloud=20.0))
    outside = BF.classify(
        _row(R2_final=50.0, R2_max=50.0, t_final=10.0, v2_final=0.01, rCloud=20.0)
    )
    assert inside["report"] == "stalled"
    assert inside["cloud_outcome"] == "retained"
    assert outside["report"] == "stalled_beyond"
    assert outside["cloud_outcome"] == "cleared"


def test_dissolved_wins_over_motion_but_keeps_the_cleared_axis():
    """A shell can dissolve inside the cloud or outside it; the report is the
    same but cloud_outcome is not."""
    inside = BF.classify(_row(isDissolved=True, R2_max=10.0, rCloud=20.0))
    outside = BF.classify(_row(isDissolved=True, R2_max=90.0, R2_final=90.0, rCloud=20.0))
    assert inside["report"] == outside["report"] == "shell_dissolved"
    assert inside["cloud_outcome"] == "retained"
    assert outside["cloud_outcome"] == "cleared"


def test_missing_columns_do_not_invent_a_fate():
    v = BF.classify({})
    assert v["motion"] == "undetermined"
    assert v["cleared"] == "undetermined"
    assert v["report"] == "unresolved"


def test_csv_string_values_are_handled():
    """summary.csv gives everything as strings; 'False' must not be truthy."""
    v = BF.classify(
        {
            "v2_final": "10.0",
            "R2_final": "50",
            "R2_max": "50",
            "t_final": "5",
            "rCloud": "20",
            "isCollapse": "False",
            "isDissolved": "False",
            "broke_out": "False",
            "end_reason": "stopping_time",
        }
    )
    assert v["report"] == "breakout"
    assert BF._truthy("False") is False
    assert BF._truthy("nan") is False
    assert BF._num("nan") is None


# --------------------------------------------------------------------------
# The reference table itself must stay honest
# --------------------------------------------------------------------------


def test_every_report_states_what_it_is_not():
    for r in BF.REPORTS:
        assert r.not_.strip(), "%s has no NOT clause" % r.name
        assert r.means.strip()


def test_the_word_dispersed_is_not_in_either_vocabulary():
    """It was doing three jobs at once. It is retired on purpose."""
    assert not any("dispersed" == n for n in BF.REPORT_ORDER)
    assert not any("dispersed" == k for k in BF.CLOUD_OUTCOMES)


def test_upstream_defects_are_recorded_with_locations():
    assert len(BF.UPSTREAM_DEFECTS) >= 5
    for title, where, code, why in BF.UPSTREAM_DEFECTS:
        assert title and where and code and why


@pytest.mark.parametrize("flag", ["--table", "--reports", "--defects"])
def test_cli_renders(flag):
    mod = Path(BF.__file__)
    out = subprocess.run([sys.executable, str(mod), flag], capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    assert len(out.stdout) > 500


def test_explain_mentions_the_rule_that_fired():
    txt = BF.explain(
        _row(
            end_reason="shell_collapsed",
            isCollapse=True,
            v2_final=-30.0,
            R2_final=1.0,
            R2_max=90.0,
            rCloud=20.0,
        )
    )
    assert "recaptured" in txt
    assert "cloud" in txt
