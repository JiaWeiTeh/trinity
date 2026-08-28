#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Bubble end-fate: the single source of truth
===========================================

What a TRINITY run's ending MEANS, in one place.

The problem this file exists to solve
-------------------------------------
"Dispersed", "collapsed" and "dissolved" were being used as if they were
mutually exclusive labels on one axis.  They are not:

* A bubble can clear the cloud AND then fall back.  "Collapsing" and
  "dispersed" are both true of it.
* A bubble can collapse without ever clearing anything.
* "Disperse" was doing double duty for *the cloud was cleared* and for
  *the shell was still expanding when we stopped looking*.  Those are
  opposite epistemic situations: one is a result, the other is a censoring.
* A bubble can sit at ~zero velocity for the rest of the run, neither
  expanding nor collapsing.  Nothing in ``trinity/`` has any concept of
  this, so it was being absorbed into whichever neighbouring label the
  priority ladder happened to reach first.

So this module does NOT start from a list of words.  It records four
INDEPENDENT facts about a run, and derives the headline word from them
through a table you can print (``--reports``).  Any two of the four facts
can be true at once, because in the physics they can.

    motion    what the shell was doing at the last integrated instant
    cleared   whether the shell EVER got past the cloud edge
    shell     whether the swept shell is still a distinct overdensity
    stop      why the integration ended, and whether to trust it

    report        = f(motion, cleared, shell, stop)      # REPORT_TABLE
    cloud_outcome = g(cleared, motion, R2_final/rCloud)  # the CLOUD's fate,
                                                        # never the bubble's

``report`` is the bubble's fate.  ``cloud_outcome`` is the cloud's.  They
are different questions and this module refuses to answer them with the
same word.

Provenance
----------
Every ``cause`` row below was read out of the phase runners at
``f39e7c41`` (2026-08-21).  ``python bubble_fate.py --table`` prints them
with their file:line so a future session can re-check the code instead of
trusting this docstring.

Two upstream defects are encoded here as data, not worked around silently:

* ``isCollapse`` IS NOT "the bubble is collapsing".
  ``phase_general/phase_events.py:694`` sets it from a *substring* test on
  the terminating event's NAME::

      if 'radius' in reason_code.lower() or 'collapse' in reason_code.lower():
          params['isCollapse'].value = True

  so ``large_radius_event`` — the ``stop_r`` wall, i.e. a bubble that got
  too BIG — sets "is the cloud collapsing?" to True because its name
  contains "radius".  Never read the flag without the gate in
  :func:`motion_of`.  See ``UPSTREAM_DEFECTS``.

* ``allowShellDissolution`` does not disable dissolution.  Its only read
  (``bubble_structure/shell_structure.py:447``) feeds
  ``diss_condition_met``, which is returned and then consumed nowhere; the
  live terminal checks test ``shell_nMax < nISM`` directly and ungated.

Usage
-----
    python tools/bubble_fate.py --reports     # the fate vocabulary
    python tools/bubble_fate.py --table       # every cause -> report, with file:line
    python tools/bubble_fate.py --matrix      # all 144 axis combinations -> report
    python tools/bubble_fate.py --defects     # the upstream traps
    python tools/bubble_fate.py               # all four
    python tools/bubble_fate.py --check paper/II-survey/plots_v2/summary.csv

    from tools.bubble_fate import classify, explain
    classify(row)   # -> dict(motion=, cleared=, shell=, stop=, report=, ...)
    explain(row)    # -> str, the reasoning chain for one run
"""

from __future__ import annotations

import argparse
import sys
from typing import Any, Dict, Mapping, NamedTuple, Optional, Sequence

__all__ = [
    "STALL_DISPLACEMENT_FRAC",
    "CAUSES",
    "CAUSE_BY_CODE",
    "CAUSE_BY_OUTCOME",
    "REPORTS",
    "REPORT_TABLE",
    "CLOUD_OUTCOMES",
    "UPSTREAM_DEFECTS",
    "motion_of",
    "cleared_of",
    "shell_of",
    "stop_of",
    "classify",
    "explain",
]


# ---------------------------------------------------------------------------
# The stall criterion.  There is NO stall concept anywhere in trinity/ --
# this is the first one, so it is defined here and only here.
# ---------------------------------------------------------------------------

STALL_DISPLACEMENT_FRAC = 0.05
"""A bubble is STALLED if, at its terminal speed, it would move less than
this fraction of its own radius over a time equal to its own age::

    |v2_final| * t_final  <  STALL_DISPLACEMENT_FRAC * R2_final

Why ``t_final`` and not ``stop_t - t_final``.  The natural phrasing is "it
would not move appreciably in the time left before the clock runs out" --
but the runs where stall is the interesting question are exactly the ones
that RAN OUT the clock, for which ``stop_t - t_final == 0`` and the test
fires for every run including the fast ones.  ``t_final`` is the bubble's
own dynamical reference time: always positive, never degenerate, and for
the clock-truncated population (the one that matters) ``t_final ~ stop_t``,
so the two readings coincide numerically where both are defined.

Scale-free by construction: no km/s threshold to defend, and it means the
same thing for a 2 pc bubble and a 300 pc one.  At stop_t = 10 Myr it
corresponds to |v| < 0.24 km/s for a 50 pc bubble and < 2.4 km/s at the
500 pc wall.
"""


# ---------------------------------------------------------------------------
# Axis 1-4 vocabularies
# ---------------------------------------------------------------------------

MOTIONS = {
    "expanding": "v2_final > 0 and not stalled -- shell moving outward",
    "contracting": "v2_final < 0 and not stalled -- shell moving inward",
    "stalled": "|v2_final| * t_final < %g * R2_final -- effectively stopped"
    % STALL_DISPLACEMENT_FRAC,
    "undetermined": "no usable v2_final / R2_final / t_final on the row",
}

CLEARED = {
    "yes": "R2_max > rCloud at any point (or broke_out flag) -- the shell EVER left the cloud",
    "no": "the shell never reached the cloud edge",
    "undetermined": "no usable R2_max / rCloud",
}

SHELL = {
    "intact": "the swept shell is still a distinct overdensity",
    "dissolved": "shell_nMax fell below nISM for stop_t_diss (default 1 Myr) continuously",
}

STOPS = {
    "physical": "the run reached a genuine physical endpoint; the fate is a RESULT",
    "edge": "we chose to stop watching at the cloud edge (stop_at_rCloud_nSnap); "
    "an OPERATOR truncation, not a physical endpoint",
    "clock": "stop_t truncated it; the fate is RIGHT-CENSORED IN TIME",
    "wall": "stop_r truncated it; the fate is RIGHT-CENSORED IN RADIUS",
    "numerical": "the integration bailed out; there is NO fate, only a death",
    "unknown": "no end code was ever set; treat as numerical",
}


# ---------------------------------------------------------------------------
# CAUSES -- every way a run can end, read out of the phase runners.
# ---------------------------------------------------------------------------


class Cause(NamedTuple):
    """One terminating site in trinity, and what it licenses us to say."""

    code: int  # SimulationEndCode numeric
    outcome: str  # metadata.json[termination].outcome token
    details: Sequence[str]  # the literal SimulationEndReason strings
    stop: str  # -> STOPS
    condition: str  # the guard, as it appears in the source
    phases: str  # where it can fire
    sites: Sequence[str]  # file:line
    sets_flags: str  # side effects on isCollapse / isDissolved
    means: str  # what it licenses us to conclude
    trap: str  # what it does NOT license (empty if none)


CAUSES: Sequence[Cause] = (
    Cause(
        code=4,
        outcome="shell_collapsed",
        details=("Small radius reached", "Small radius reached (event)"),
        stop="physical",
        condition="isCollapse AND R2 < coll_r (default 1 pc); or the min_radius event at "
        "max(coll_r*1.5, 0.01) pc crossing downward",
        phases="implicit (1b), transition (1c), momentum (2); event also in energy (1a)",
        sites=(
            "phase1b_energy_implicit/run_energy_implicit_phase.py:1318-1326",
            "phase1c_transition/run_transition_phase.py:788-796",
            "phase2_momentum/run_momentum_phase.py:841-849",
            "phase_general/phase_events.py:134 (make_min_radius_event)",
        ),
        sets_flags="isCollapse=True (event path); EndSimulationDirectly=True",
        means="The shell came back to the collapse radius. This is a completed collapse.",
        trap="Says NOTHING about whether the cloud was cleared first -- 27.1% of these "
        "runs carry broke_out=True. Those are 'recaptured', not 'recollapsed'.",
    ),
    Cause(
        code=0,
        outcome="shell_dissolved",
        details=("Shell dissolved",),
        stop="physical",
        condition="shell_nMax < nISM held CONTINUOUSLY for stop_t_diss (default 1 Myr); "
        "the timer resets to inf the moment shell_nMax >= nISM again",
        phases="transition (1c), momentum (2)",
        sites=(
            "phase1c_transition/run_transition_phase.py:807-823",
            "phase2_momentum/run_momentum_phase.py:860-876",
        ),
        sets_flags="isDissolved=True; EndSimulationDirectly=True",
        means="The SHELL stopped being a shell -- its peak density merged into ambient. "
        "The bubble ceased to exist as a swept structure.",
        trap="This is the SHELL dissolving, not the CLOUD dispersing. The threshold is "
        "hard-wired to nISM (there is no stop_n_diss parameter), and "
        "allowShellDissolution does NOT switch it off (see --defects).",
    ),
    Cause(
        code=3,
        outcome="rcloud_boundary",
        details=(
            "Reached cloud edge (stop_at_rCloud_nSnap=0)",
            "Reached N segment(s) past rCloud (stop_at_rCloud_nSnap)",
        ),
        stop="edge",
        condition="stop_at_rCloud_nSnap is not None AND R2 > rCloud AND "
        "_snapshots_after_rCloud >= that value",
        phases="main.py after 1a (nSnap==0); top of loop in 1b / 1c / 2 (nSnap>0)",
        sites=(
            "main.py:264-272",
            "phase1b_energy_implicit/run_energy_implicit_phase.py:764-775",
            "phase1c_transition/run_transition_phase.py:468-479",
            "phase2_momentum/run_momentum_phase.py:549-560",
        ),
        sets_flags="EndSimulationDirectly=True",
        means="We DELIBERATELY stopped watching at the cloud edge. The bubble cleared "
        "the cloud; what it did afterwards was never integrated.",
        trap="An operator choice, not a physical endpoint. Default is None (disabled), so "
        "on the survey grid crossing rCloud is a PHASE TRANSITION that records nothing "
        "-- 'broke out' has to be reconstructed from R2_max > rCloud.",
    ),
    Cause(
        code=2,
        outcome="large_radius",
        details=("Large radius reached", "Large radius reached (event)"),
        stop="wall",
        condition="stop_r is not None AND R2 > stop_r (default 500 pc)",
        phases="implicit (1b), transition (1c), momentum (2)",
        sites=(
            "phase1b_energy_implicit/run_energy_implicit_phase.py:1329-1335",
            "phase1c_transition/run_transition_phase.py:799-805",
            "phase2_momentum/run_momentum_phase.py:852-858",
            "phase_general/phase_events.py:166 (make_max_radius_event)",
        ),
        sets_flags="isCollapse=True ON THE EVENT PATH ONLY (substring bug) -- the three "
        "inline checks set nothing. Same physical fate, two different flag sets.",
        means="The bubble is still growing and we stopped measuring. RIGHT-CENSORED in "
        "radius: R2_max is a lower bound, not a size.",
        trap="Every one of these carries isCollapse=True if it fired via the event path, "
        "and 100% of them are moving OUTWARD. Reading the flag raw calls a maximally "
        "expanded bubble 'collapsed'. This is the single largest fate defect in the "
        "pipeline (13,993 runs on the v2 grid).",
    ),
    Cause(
        code=1,
        outcome="stopping_time",
        details=("Stopping time reached", "Reached stop_t=<...> Myr during prior phase"),
        stop="clock",
        condition="t_now >= stop_t (default 15 Myr; the survey grid runs 10)",
        phases="implicit (1b), transition (1c), momentum (2) -- 9 sites",
        sites=(
            "phase1b_energy_implicit/run_energy_implicit_phase.py:670-690,1042-1048,1311-1316",
            "phase1c_transition/run_transition_phase.py:407-424,607-613,781-786",
            "phase2_momentum/run_momentum_phase.py:488-504,689-695,834-839",
        ),
        sets_flags="EndSimulationDirectly=True",
        means="The clock ran out. The bubble's fate is UNDECIDED -- whatever it was doing "
        "at t=stop_t, it was still doing.",
        trap="This is where motion has to do the work: a clock-truncated run is "
        "'expanding', 'stalled' or 'collapsing' depending on v2_final, and NONE of "
        "those three is a completed fate. Never fold these into a 'dispersed' count.",
    ),
    Cause(
        code=50,
        outcome="velocity_runaway",
        details=("Collapse velocity runaway (event)",),
        stop="numerical",
        condition="v2 < -MAX_VELOCITY_COLLAPSE (500 pc/Myr). Inward only -- the expansion "
        "and 'both' variants exist in the factory but are never built.",
        phases="all four (the event is in every phase's event list)",
        sites=("phase_general/phase_events.py:170-216 (make_velocity_runaway_event)",),
        sets_flags="EndSimulationDirectly=True; isCollapse NOT set (reason_code has "
        "neither 'radius' nor 'collapse' in it)",
        means="Nothing physical. The integrator hit a stiff inward manifold and bailed.",
        trap="v2_final is pinned at exactly -500.0000 by construction. Quoting these in "
        "any velocity statistic quotes the trigger threshold back at yourself.",
    ),
    Cause(
        code=51,
        outcome="energy_collapsed",
        details=(
            "Energy-driven bubble collapsed (Eb fell to a fraction of segment start)",
            "Energy-driven bubble collapsed: Eb fell to <= 0 (...)",
            "Energy-driven bubble collapsed: Eb non-finite (...)",
            "Energy-driven bubble collapsed: bubble solve degenerate as Eb -> 0 (...)",
        ),
        stop="numerical",
        condition="Eb < 1e-3 * Eb_segment_start (the event); or Eb <= 0 / non-finite "
        "between segments; or the bubble solver raising as Eb -> 0",
        phases="energy (1a), implicit (1b)",
        sites=(
            "phase_general/phase_events.py:323-382 (ENERGY_COLLAPSE_FRAC = 1e-3)",
            "phase1_energy/run_energy_phase.py:180-194,385-396",
            "phase1b_energy_implicit/run_energy_implicit_phase.py:1150-1165",
        ),
        sets_flags="isCollapse=True on the event path (reason_code contains 'collapse'); "
        "NOT set on the two inline paths",
        means="The energy-driven solve went degenerate. A stiffness bail-out.",
        trap="The word 'collapsed' in the token is about the BUBBLE ENERGY, not the "
        "shell. On the v2 grid 99.6% of these runs were still moving OUTWARD when it "
        "fired (median v2 = +654 pc/Myr, R2_final/R2_max = 1.0000, median t_final "
        "1.5 kyr). A substring net on 'collaps' turns 7,968 expanding runs into "
        "recollapsed clouds.",
    ),
    Cause(
        code=99,
        outcome="unknown",
        details=("unknown",),
        stop="unknown",
        condition="no site ever set SimulationEndCode -- the run fell out of every phase "
        "through a no-code exit (solver_error, solver_failed, max_segments, "
        "ram_dominated, energy_floor, cooling_balance, no_physical_root_handoff)",
        phases="any",
        sites=("_output/simulation_end.py:192-196 (the fallback)",),
        sets_flags="none",
        means="We do not know why this run ended. Treat exactly as 'numerical'.",
        trap="These are silent: the phase hand-off paths do not log a code, so a run that "
        "died in the solver looks identical on the way out to one that simply "
        "finished. Only the absence of a code distinguishes it.",
    ),
    Cause(
        code=10,
        outcome="error_*",
        details=("(never emitted by production code)",),
        stop="numerical",
        condition="codes 10-29 (invalid params / mass inconsistency / edge density / "
        "radius too large / numerical / velocity / solver / negative values)",
        phases="none -- no production site sets any of them",
        sites=("_output/simulation_end.py:77-86 (declared, unused)",),
        sets_flags="none",
        means="Reserved. If one ever appears in a grid, something changed upstream.",
        trap="Their absence is not proof of health: real solver failures exit with NO "
        "code and land in 'unknown' (99) instead of ERROR_SOLVER (22).",
    ),
)

CAUSE_BY_CODE: Mapping[int, Cause] = {c.code: c for c in CAUSES}
CAUSE_BY_OUTCOME: Mapping[str, Cause] = {c.outcome: c for c in CAUSES}


# ---------------------------------------------------------------------------
# REPORTS -- the headline vocabulary.  Ten words, each with a definition and
# an explicit statement of what it must NOT be read as.
# ---------------------------------------------------------------------------


class Report(NamedTuple):
    name: str
    when: str  # the axis combination
    means: str
    not_: str  # what this word does NOT mean
    censored: bool  # is the true fate unknown beyond this point?


REPORTS: Sequence[Report] = (
    Report(
        "recollapsed",
        "stop=physical(shell_collapsed), cleared=no",
        "The shell went out, turned round, and came back to coll_r without ever "
        "reaching the cloud edge. A completed collapse; the cloud survives intact.",
        "Not 'the feedback failed' -- it may have cleared a large fraction of the "
        "cloud mass on the way. Only the SHELL returned.",
        False,
    ),
    Report(
        "recaptured",
        "stop=physical(shell_collapsed), cleared=yes",
        "The shell DID clear the cloud edge, and then fell back to coll_r. This is "
        "the case that made 'collapsed' and 'dispersed' look contradictory: both are "
        "true, in that order. 27.1% of shell_collapsed runs on the v2 grid.",
        "Not 'recollapsed' -- the cloud WAS opened. Not 'dispersed' either -- it "
        "closed again. Counting these as either one loses the mechanism, which is "
        "ambient recapture after clearing (P_ISM is gated on breakout).",
        False,
    ),
    Report(
        "collapsing",
        "stop=clock or wall, motion=contracting",
        "Moving inward when we stopped watching, but it never reached coll_r. The "
        "collapse is UNDERWAY, not finished.",
        "Not 'recollapsed'. A run that would have turned around at t=11 Myr is in "
        "here. Do not merge it into a completed-collapse count.",
        True,
    ),
    Report(
        "stalled",
        "motion=stalled, cleared=no, shell=intact",
        "Neither expanding nor collapsing: at its terminal speed it would move less "
        "than 5% of its radius over its own lifetime. Parked inside the cloud.",
        "Not 'collapsed' and not 'still expanding'. This is its own outcome and it "
        "was previously being swallowed by whichever neighbour the ladder hit first.",
        True,
    ),
    Report(
        "stalled_beyond",
        "motion=stalled, cleared=yes, shell=intact",
        "Stalled, but outside the cloud -- it cleared the cloud and then stopped "
        "growing. The CLOUD's fate and the BUBBLE's fate differ here.",
        "Not 'stalled' -- for the cloud census this one counts as cleared, and "
        "lumping it with the inside-the-cloud stalls double-counts a failure.",
        True,
    ),
    Report(
        "expanding",
        "stop=clock, motion=expanding, cleared=no",
        "Still growing inside the cloud when the clock ran out. The outcome is simply "
        "NOT KNOWN within stop_t.",
        "NOT 'dispersed'. This is the single worst word-collision in the old scheme: "
        "'still expanding' is a censoring, 'cleared the cloud' is a result. They were "
        "sharing a label.",
        True,
    ),
    Report(
        "breakout",
        "cleared=yes, motion=expanding, stop=clock or edge",
        "Cleared the cloud edge and still moving outward at the end. THIS is what "
        "'the cloud was dispersed by the bubble' means.",
        "Not a size measurement -- R2_max is where we stopped looking, not where it "
        "stopped. Check `stop` before quoting a radius.",
        True,
    ),
    Report(
        "shell_dissolved",
        "shell=dissolved (any motion, any cleared)",
        "The swept shell's peak density fell to ambient and stayed there for "
        "stop_t_diss. The bubble stopped being a bubble. Report `cleared` alongside "
        "it -- a shell can dissolve inside the cloud or outside it.",
        "Not 'the cloud dispersed'. The criterion is about the SHELL's density "
        "contrast against nISM, nothing else. Once dissolved, F_rad is forced to 0, "
        "so any force budget after this point is a different physical problem.",
        False,
    ),
    Report(
        "radius_capped",
        "stop=wall, motion != contracting",
        "Hit the stop_r = 500 pc wall. Physically a `breakout`, held separate so a "
        "right-censored radius can never be quoted as a size.",
        "Not a fate at all, strictly -- it is a measurement limit. 22.9% of the v2 "
        "grid sits on this wall.",
        True,
    ),
    Report(
        "unresolved",
        "stop=numerical or unknown",
        "The integration died: velocity runaway, energy-collapse bail-out, a silent "
        "solver failure, or no code at all. 18.3% of the v2 grid (25.0% at "
        "etaw=0.01).",
        "NOT a physical outcome and never guessable into one. It must appear in "
        "every legend and every denominator, or the percentages stop summing to the "
        "sample with nothing saying why. Excluding it silently is how a solver death "
        "at 1.5 kyr ends up counted as a cloud that dispersed.",
        True,
    ),
)

REPORT_BY_NAME: Mapping[str, Report] = {r.name: r for r in REPORTS}

# Plot ordering + a colour per report.  Ordered worst-for-the-cloud to
# best-for-the-cloud, with the two non-fates last so they read as a
# separate block in every legend.
REPORT_ORDER: Sequence[str] = (
    "recollapsed",
    "recaptured",
    "collapsing",
    "stalled",
    "stalled_beyond",
    "expanding",
    "breakout",
    "shell_dissolved",
    "radius_capped",
    "unresolved",
)

REPORT_COLORS: Mapping[str, str] = {
    "recollapsed": "#8c2d04",
    "recaptured": "#d95f0e",
    "collapsing": "#fe9929",
    "stalled": "#fec44f",
    "stalled_beyond": "#fee391",
    "expanding": "#a6bddb",
    "breakout": "#2c7fb8",
    "shell_dissolved": "#7fcdbb",
    "radius_capped": "#74c476",
    "unresolved": "#9e9e9e",
}

CLOUD_OUTCOMES = {
    "cleared": "the shell reached the cloud edge and was still outside it at the end",
    "cleared_then_recaptured": "the shell got past the cloud edge and came back inside",
    "retained": "the shell never reached the cloud edge",
    "undetermined": "the run died before the question could be answered (unresolved)",
}

UPSTREAM_DEFECTS = (
    (
        "isCollapse is a substring test on an event NAME",
        "phase_general/phase_events.py:694",
        "if 'radius' in reason_code.lower() or 'collapse' in reason_code.lower():",
        "large_radius_event contains 'radius', so a bubble that blew past stop_r is "
        "flagged as collapsing; energy_collapse_event contains 'collapse', so a "
        "stiffness bail-out at 1.5 kyr is too. Together 21,961 runs, 34.8% of the v2 "
        "grid. FIX UPSTREAM: dispatch on result.end_code against the SimulationEndCode "
        "enum, not on the name. Until then, motion_of() gates the flag.",
    ),
    (
        "isCollapse means 'contracting', not 'collapsed', and never resets",
        "registry.py:540 (default False); set at 1b:1305, 1c:775, 2:828 by "
        "`if v2 < 0 and R2 < R2_prev`",
        "no `isCollapse = False` assignment exists anywhere in trinity/",
        "One inward step latches it for the rest of the run, and it then changes the "
        "physics (shell mass is frozen) and arms the R2 < coll_r terminal check. A "
        "bubble that dips and recovers carries the flag to the end.",
    ),
    (
        "allowShellDissolution does not disable dissolution",
        "bubble_structure/shell_structure.py:447-450",
        "diss_condition_met = bool(allow_dissolution and nShell_max < nISM)",
        "diss_condition_met is returned on ShellProperties and consumed NOWHERE. The "
        "live terminal checks (1c:807, 2:860) test shell_nMax < nISM directly and "
        "ungated, so setting the parameter False changes nothing about termination.",
    ),
    (
        "the same fate produces two different flag sets",
        "event path via apply_event_result vs the inline between-segment checks",
        "phase_events.py:688-696 vs 1b:1329 / 1c:799 / 2:852",
        "Crossing stop_r INSIDE a segment sets isCollapse=True; crossing it BETWEEN "
        "segments sets nothing. Identical physics, different metadata, depending only "
        "on where the sample landed.",
    ),
    (
        "check_event_termination returns the lowest-index event, not the earliest",
        "phase_general/phase_events.py:430-483",
        "returns the first event in list order that has any recorded crossing",
        "In phase 1b the non-terminal velocity_sign event is index 0, so it can "
        "pre-empt a same-segment min_radius / max_radius / velocity_runaway and turn a "
        "terminal outcome into a silent hand-off to 1c with no end code.",
    ),
    (
        "there is no stall concept in trinity at all",
        "searched: stall|stagn|quasi-static|standstill, and any |v2| < eps test",
        "the only velocity-magnitude logic (VELOCITY_THRESHOLD_COLLAPSE = 50, "
        "_EXTREME = 150 pc/Myr) only shrinks dt_segment; it never terminates",
        "A bubble coasting at v ~ 0 runs to stop_t and reports stopping_time, "
        "indistinguishable from a fast one that also ran out of clock. "
        "STALL_DISPLACEMENT_FRAC in this file is the first definition of the concept.",
    ),
)


# ---------------------------------------------------------------------------
# Row helpers.  A "row" is any Mapping: a dict, a pandas Series, a csv
# DictReader row.  Values may be strings (CSV) or numbers.
# ---------------------------------------------------------------------------


def _num(v: Any) -> Optional[float]:
    """Float or None. NaN, '', 'nan', None and non-numeric all become None."""
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return None if f != f else f


def _truthy(v: Any) -> bool:
    """Bool from CSV-ish input: 'False', '0', 'nan', '' and None are all False."""
    return bool(v) and str(v).strip().lower() not in ("false", "0", "nan", "none", "")


def _outcome_of(row: Mapping[str, Any]) -> str:
    """The end-reason TOKEN, lowercased. Prefers the clean outcome token; falls
    back to the free-text detail, matched only by exact token so the broad
    substring nets that caused the original mislabelling cannot come back."""
    for key in ("end_outcome", "outcome", "end_reason", "SimulationEndReason"):
        raw = row.get(key)
        if isinstance(raw, str) and raw.strip():
            tok = raw.strip().lower()
            if tok in CAUSE_BY_OUTCOME:
                return tok
            # free-text detail -> match against the literal reason strings
            for cause in CAUSES:
                if any(tok == d.lower() for d in cause.details):
                    return cause.outcome
            # 'Reached stop_t=... Myr during prior phase'
            if tok.startswith("reached stop_t"):
                return "stopping_time"
            if tok.startswith("reached ") and "rcloud" in tok:
                return "rcloud_boundary"
    code = _num(row.get("end_code") or row.get("SimulationEndCode"))
    if code is not None and int(code) in CAUSE_BY_CODE:
        return CAUSE_BY_CODE[int(code)].outcome
    return "unknown"


# ---------------------------------------------------------------------------
# The four axes
# ---------------------------------------------------------------------------


def motion_of(row: Mapping[str, Any], stall_frac: float = STALL_DISPLACEMENT_FRAC) -> str:
    """Axis 1: what the shell was doing at the last integrated instant.

    Reads v2_final, and NOT isCollapse -- see UPSTREAM_DEFECTS. The flag is
    consulted only as a tie-break when v2_final is missing, and even then it
    is gated on the shell not sitting at its own peak radius.
    """
    v2 = _num(row.get("v2_final"))
    r_fin = _num(row.get("R2_final"))
    t_fin = _num(row.get("t_final"))

    if v2 is not None and r_fin is not None and t_fin is not None and r_fin > 0:
        if abs(v2) * t_fin < stall_frac * r_fin:
            return "stalled"
    if v2 is not None:
        return "expanding" if v2 > 0 else "contracting" if v2 < 0 else "stalled"

    # No velocity. Fall back to the flag, gated: a shell parked at its own
    # maximum radius has not begun to come back, whatever the flag says.
    r_max = _num(row.get("R2_max"))
    if _truthy(row.get("isCollapse")):
        if r_fin is not None and r_max and r_fin >= 0.999 * r_max:
            return "undetermined"
        return "contracting"
    return "undetermined"


def cleared_of(row: Mapping[str, Any]) -> str:
    """Axis 2: did the shell EVER get past the cloud edge.

    'Ever', not 'at the end' -- that distinction is the whole point. A shell
    that broke out and fell back is cleared=yes AND contracting.
    """
    if _truthy(row.get("broke_out")):
        return "yes"
    r_max, r_cloud = _num(row.get("R2_max")), _num(row.get("rCloud"))
    if r_max is None or not r_cloud:
        return "undetermined"
    return "yes" if r_max > r_cloud else "no"


def shell_of(row: Mapping[str, Any]) -> str:
    """Axis 3: is the swept shell still a distinct overdensity."""
    if _truthy(row.get("isDissolved")):
        return "dissolved"
    return "dissolved" if _outcome_of(row) == "shell_dissolved" else "intact"


def stop_of(row: Mapping[str, Any]) -> str:
    """Axis 4: why the integration ended, and whether to trust the answer."""
    cause = CAUSE_BY_OUTCOME.get(_outcome_of(row))
    return cause.stop if cause else "unknown"


# ---------------------------------------------------------------------------
# The derived headline
# ---------------------------------------------------------------------------


# REPORT_TABLE is the literal mapping, in evaluation order. Each entry is
# (predicate-description, callable) so --table can print the rules that
# produce each word rather than asking you to read the function.
def _report_rules():
    return (
        (
            "unresolved",
            "stop in {numerical, unknown}",
            lambda m, c, s, st: st in ("numerical", "unknown"),
        ),
        (
            "shell_dissolved",
            "shell == dissolved",
            lambda m, c, s, st: s == "dissolved",
        ),
        (
            "recaptured",
            "stop == physical (i.e. shell_collapsed; dissolved was caught above) "
            "and cleared == yes",
            lambda m, c, s, st: st == "physical" and c == "yes",
        ),
        (
            "recollapsed",
            "stop == physical (shell_collapsed) and cleared != yes",
            lambda m, c, s, st: st == "physical",
        ),
        (
            "collapsing",
            "motion == contracting (clock- or wall-truncated)",
            lambda m, c, s, st: m == "contracting",
        ),
        (
            "stalled_beyond",
            "motion == stalled and cleared == yes",
            lambda m, c, s, st: m == "stalled" and c == "yes",
        ),
        (
            "stalled",
            "motion == stalled and cleared != yes",
            lambda m, c, s, st: m == "stalled",
        ),
        (
            "radius_capped",
            "stop == wall (and not contracting -- that was caught above)",
            lambda m, c, s, st: st == "wall",
        ),
        (
            "breakout",
            "cleared == yes and motion == expanding",
            lambda m, c, s, st: c == "yes" and m == "expanding",
        ),
        (
            "expanding",
            "motion == expanding, still inside the cloud",
            lambda m, c, s, st: m == "expanding",
        ),
        (
            "unresolved",
            "fallback: no axis carried usable information",
            lambda m, c, s, st: True,
        ),
    )


REPORT_TABLE = _report_rules()


def _cloud_outcome(row: Mapping[str, Any], cleared: str, report: str) -> str:
    if report == "unresolved":
        return "undetermined"
    if cleared != "yes":
        return "retained"
    r_fin, r_cloud = _num(row.get("R2_final")), _num(row.get("rCloud"))
    if r_fin is not None and r_cloud and r_fin < r_cloud:
        return "cleared_then_recaptured"
    return "cleared"


def classify(row: Mapping[str, Any], stall_frac: float = STALL_DISPLACEMENT_FRAC) -> Dict[str, Any]:
    """Four orthogonal facts about one run, plus the two derived labels.

    Returns a dict with keys:
        motion, cleared, shell, stop   -- the independent axes
        cause                          -- the end-reason token
        report                         -- the bubble's headline fate
        cloud_outcome                  -- the CLOUD's fate (a different question)
        censored                       -- True if the true fate is unknown beyond here
    """
    motion = motion_of(row, stall_frac)
    cleared = cleared_of(row)
    shell = shell_of(row)
    stop = stop_of(row)
    for name, why, pred in REPORT_TABLE:
        if pred(motion, cleared, shell, stop):
            report, rule = name, why
            break
    return {
        "motion": motion,
        "cleared": cleared,
        "shell": shell,
        "stop": stop,
        "cause": _outcome_of(row),
        "report": report,
        "rule": rule,
        "cloud_outcome": _cloud_outcome(row, cleared, report),
        "censored": REPORT_BY_NAME[report].censored,
    }


def explain(row: Mapping[str, Any], stall_frac: float = STALL_DISPLACEMENT_FRAC) -> str:
    """The reasoning chain for one run, as text. For arguing with a figure."""
    v = classify(row, stall_frac)
    cause = CAUSE_BY_OUTCOME.get(v["cause"])
    rep = REPORT_BY_NAME[v["report"]]
    why = v["rule"]
    lines = [
        "cause    %-18s %s" % (v["cause"], cause.means if cause else ""),
        "motion   %-18s %s" % (v["motion"], MOTIONS[v["motion"]]),
        "cleared  %-18s %s" % (v["cleared"], CLEARED[v["cleared"]]),
        "shell    %-18s %s" % (v["shell"], SHELL[v["shell"]]),
        "stop     %-18s %s" % (v["stop"], STOPS[v["stop"]]),
        "",
        "REPORT   %-18s (rule: %s)" % (v["report"], why),
        "         %s" % rep.means,
        "  NOT    %s" % rep.not_,
        "cloud    %-18s %s" % (v["cloud_outcome"], CLOUD_OUTCOMES[v["cloud_outcome"]]),
        "censored %s" % v["censored"],
    ]
    if cause and cause.trap:
        lines.insert(1, "  trap   %s" % cause.trap)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _wrap(text: str, width: int = 76, indent: str = " " * 4) -> str:
    import textwrap

    return "\n".join(textwrap.wrap(text, width, initial_indent=indent, subsequent_indent=indent))


def _print_causes() -> None:
    print("=" * 79)
    print("CAUSE -> REPORT.  Every way a TRINITY run can end.")
    print("Read out of the phase runners at f39e7c41; file:line given so you can re-check.")
    print("=" * 79)
    for c in CAUSES:
        print("\n[%2d] %-18s  stop=%s" % (c.code, c.outcome, c.stop))
        print(_wrap("condition: " + c.condition))
        print(_wrap("phases:    " + c.phases))
        for s in c.sites:
            print("      %s" % s)
        print(_wrap("flags:     " + c.sets_flags))
        print(_wrap("MEANS:     " + c.means))
        if c.trap:
            print(_wrap("TRAP:      " + c.trap))
        reachable = sorted({n for n, _w, _p in REPORT_TABLE if _plausible(c, n)})
        print(_wrap("reports:   " + ", ".join(reachable)))


def _plausible(cause: Cause, report: str) -> bool:
    """Which headline reports a given cause can produce, over all axis states."""
    for motion in MOTIONS:
        for cleared in CLEARED:
            for shell in SHELL:
                if cause.outcome == "shell_dissolved" and shell != "dissolved":
                    continue
                if cause.outcome != "shell_dissolved" and shell == "dissolved":
                    continue
                for name, _w, pred in REPORT_TABLE:
                    if pred(motion, cleared, shell, cause.stop):
                        if name == report:
                            return True
                        break
    return False


def _print_reports() -> None:
    print("=" * 79)
    print("THE FATE VOCABULARY.  Ten words. Each says what it is NOT.")
    print("=" * 79)
    for name in REPORT_ORDER:
        r = REPORT_BY_NAME[name]
        print(
            "\n%-16s %s"
            % (r.name, "[CENSORED -- true fate unknown]" if r.censored else "[completed]")
        )
        print(_wrap("when:  " + r.when))
        print(_wrap("means: " + r.means))
        print(_wrap("NOT:   " + r.not_))
    print("\n" + "-" * 79)
    print("THE CLOUD'S OUTCOME is a separate question with its own vocabulary:")
    for k, v in CLOUD_OUTCOMES.items():
        print(_wrap("%-24s %s" % (k, v), indent=""))
    print("\nThe word 'dispersed' appears in NEITHER vocabulary. It was doing three")
    print("jobs at once (cloud cleared / shell still expanding / shell dissolved).")
    print("Use 'breakout' for the bubble, 'cleared' for the cloud, and say which.")


def _print_matrix() -> None:
    """The full enumeration: every state of the four axes -> its report.

    This is the literal answer to "what are all the combinations and what do
    they report".  144 rows; the impossible ones (shell=dissolved without the
    dissolution cause) are marked rather than hidden, because the classifier
    would still answer if it saw one.
    """
    print("=" * 79)
    print("THE FULL MATRIX.  4 axes -> 1 report. Every combination, no exceptions.")
    print("=" * 79)
    print(
        "%-10s %-13s %-12s %-10s %-16s %s"
        % ("stop", "motion", "cleared", "shell", "REPORT", "censored")
    )
    print("-" * 79)
    for stop in STOPS:
        for motion in MOTIONS:
            for cleared in CLEARED:
                for shell in SHELL:
                    for name, _w, pred in REPORT_TABLE:
                        if pred(motion, cleared, shell, stop):
                            break
                    print(
                        "%-10s %-13s %-12s %-10s %-16s %s"
                        % (
                            stop,
                            motion,
                            cleared,
                            shell,
                            name,
                            "yes" if REPORT_BY_NAME[name].censored else "-",
                        )
                    )
        print("-" * 79)
    print("\nThe stop axis comes from the CAUSE (--table); the other three are")
    print("measured off the run. `cloud_outcome` is derived separately and is")
    print("NOT a function of these four alone -- it also needs R2_final/rCloud,")
    print("which is how 'cleared_then_recaptured' is told from 'cleared'.")


def _print_defects() -> None:
    print("=" * 79)
    print("UPSTREAM DEFECTS.  These are in trinity/, not in the analysis.")
    print("=" * 79)
    for title, where, code, why in UPSTREAM_DEFECTS:
        print("\n* %s" % title)
        print("  %s" % where)
        print("      %s" % code)
        print(_wrap(why, indent=" " * 2))


def _check(path: str, limit: int = 0) -> int:
    import collections
    import csv

    counts: "collections.Counter[str]" = collections.Counter()
    clouds: "collections.Counter[str]" = collections.Counter()
    axes = {k: collections.Counter() for k in ("motion", "cleared", "shell", "stop")}
    n = 0
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh):
            v = classify(row)
            counts[v["report"]] += 1
            clouds[v["cloud_outcome"]] += 1
            for k in axes:
                axes[k][v[k]] += 1
            n += 1
            if limit and n >= limit:
                break
    if not n:
        print("no rows in %s" % path)
        return 1
    print("%d runs from %s\n" % (n, path))
    for k, c in axes.items():
        print(
            "%-9s %s" % (k, "  ".join("%s=%.1f%%" % (a, 100 * v / n) for a, v in c.most_common()))
        )
    print("\n%-16s %8s %7s" % ("report", "runs", "%"))
    unaccounted = n
    for name in REPORT_ORDER:
        v = counts.get(name, 0)
        unaccounted -= v
        flag = " (censored)" if REPORT_BY_NAME[name].censored else ""
        print("%-16s %8d %6.2f%%%s" % (name, v, 100 * v / n, flag))
    print("%-16s %8d %6.2f%%" % ("--- TOTAL", n - unaccounted, 100 * (n - unaccounted) / n))
    print("\n%-24s %8s %7s" % ("cloud_outcome", "runs", "%"))
    for k in CLOUD_OUTCOMES:
        v = clouds.get(k, 0)
        print("%-24s %8d %6.2f%%" % (k, v, 100 * v / n))
    if unaccounted:
        print("\nERROR: %d runs produced a report outside REPORT_ORDER" % unaccounted)
        return 1
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__.split("Usage")[0], formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--table", action="store_true", help="print every cause -> report")
    ap.add_argument("--reports", action="store_true", help="print the fate vocabulary")
    ap.add_argument("--matrix", action="store_true", help="print all 144 axis combinations")
    ap.add_argument("--defects", action="store_true", help="print the upstream traps")
    ap.add_argument("--check", metavar="SUMMARY_CSV", help="classify a real grid and census it")
    ap.add_argument("--limit", type=int, default=0, help="with --check, stop after N rows")
    args = ap.parse_args(argv)

    if args.check:
        return _check(args.check, args.limit)
    if not (args.table or args.reports or args.matrix or args.defects):
        args.table = args.reports = args.matrix = args.defects = True
    if args.reports:
        _print_reports()
    if args.table:
        print()
        _print_causes()
    if args.matrix:
        print()
        _print_matrix()
    if args.defects:
        print()
        _print_defects()
    return 0


if __name__ == "__main__":
    sys.exit(main())
