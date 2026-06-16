#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Implicit->momentum transition-trigger criteria (F0 / F4) and shadow logging.

This module is the single source of truth for the two transition criteria:
  F0 instantaneous rate-ratio (cooling balance): ``(Lgain - Lloss)/Lgain < eps``
  F4 blowout (geometric):                        ``R2 > rCloud``

The ``*_fires`` predicates and ``implicit_termination_reason`` are used by **both**
the live implicit-phase terminator (``run_energy_implicit_phase.py``) and the
``ShadowTransitionLog`` below, so the shadow F0 epoch equals the live break epoch
*by construction*. Selection is via the ``transition_trigger`` param:
  ``'instantaneous'`` (default): terminate on F0 only; F4 is logged (shadow) but
                                 not acted on -> byte-identical snapshots.
  ``'cooling_or_blowout'`` (P-promote): terminate on F0 **or** F4, whichever fires
                                 first. Both route 1b->1c->2 like ``cooling_balance``.

``ShadowTransitionLog`` records the first segment where each criterion *would*
fire and writes a sideline ``transition_shadow.jsonl`` — it never touches
snapshots, result arrays, or ``params`` state. See
``docs/dev/transition/pshadow-design.md`` and ``TRIGGER_PLAN.md``.
"""
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

SHADOW_FILENAME = "transition_shadow.jsonl"

VALID_TRANSITION_TRIGGERS = ("instantaneous", "cooling_or_blowout")


def validate_transition_trigger(value):
    """Return ``value`` if it is a known ``transition_trigger``, else raise.

    Fails loudly so a misconfigured run does not silently fall back to the
    default behaviour.
    """
    if value not in VALID_TRANSITION_TRIGGERS:
        raise ValueError(
            f"transition_trigger={value!r} is invalid; "
            f"expected one of {VALID_TRANSITION_TRIGGERS}."
        )
    return value


def cooling_balance_fires(Lgain, Lloss, threshold):
    """F0 (cooling balance): the instantaneous energy-retention ratio
    ``(Lgain - Lloss)/Lgain`` has dropped below ``threshold``.

    This is the live implicit-phase terminator in every mode. ``Lgain <= 0``
    never fires (no injected power for cooling to overtake).
    """
    return Lgain > 0 and (Lgain - Lloss) / Lgain < threshold


def blowout_fires(R2, rCloud):
    """F4 (blowout, geometric): the shell radius ``R2`` has escaped the cloud
    (``R2 > rCloud``).

    Independent of the energy budget — a blown-out shell may still be formally
    energy-driven (the steep-profile fate; see pshadow-design.md §6b).
    """
    return rCloud is not None and R2 > rCloud


def implicit_termination_reason(transition_trigger, Lgain, Lloss, threshold, R2, rCloud):
    """Transition reason for this implicit-phase segment, or ``None`` to continue.

    F0 (``"cooling_balance"``) terminates the implicit phase in every mode. F4
    (``"blowout"``) terminates it only under ``'cooling_or_blowout'`` (P-promote).
    F0 takes precedence when both hold in the same segment, so the cooling path
    is byte-identical to the pre-promote behaviour. Both reasons route
    1b->1c->2 (neither sets ``EndSimulationDirectly``); the caller owns the
    break, logging, and any param state.
    """
    if cooling_balance_fires(Lgain, Lloss, threshold):
        return "cooling_balance"
    if transition_trigger == "cooling_or_blowout" and blowout_fires(R2, rCloud):
        return "blowout"
    return None


class ShadowTransitionLog:
    """Accumulate the first-fire epoch of each shadow transition criterion.

    ``update`` is idempotent per criterion: only the *first* segment where a
    criterion holds is kept, matching the design's "first epoch where each
    criterion would fire".
    """

    def __init__(self):
        self.F0 = None  # cooling balance — the current live trigger
        self.F4 = None  # blowout — R2 > rCloud (shadow only, never acted on in 1b)

    def update(self, t, R2, rCloud, Lgain, Lloss, threshold):
        """Record the first segment where F0 / F4 would fire.

        F0 uses the *same* expression as the production terminator
        (``run_energy_implicit_phase.py``), so the shadow F0 epoch equals the
        live ``cooling_balance`` break epoch.
        """
        ratio_F0 = (Lgain - Lloss) / Lgain if Lgain > 0 else float("nan")
        if self.F0 is None and cooling_balance_fires(Lgain, Lloss, threshold):
            self.F0 = {"which": "F0", "t": t, "R2": R2, "rCloud": rCloud,
                       "ratio_F0": ratio_F0}
        if self.F4 is None and blowout_fires(R2, rCloud):
            self.F4 = {"which": "F4", "t": t, "R2": R2, "rCloud": rCloud,
                       "ratio_F0": ratio_F0}

    def records(self):
        """The recorded first-fire epochs (F0 before F4), skipping criteria that never fired."""
        return [r for r in (self.F0, self.F4) if r is not None]

    def write(self, out_dir):
        """Write recorded first-fire epochs to ``<out_dir>/transition_shadow.jsonl``.

        Writes nothing if no criterion fired (absence == "no shadow transition
        in 1b"). Failures are logged, never raised — a diagnostic must not break
        a production run.
        """
        recs = self.records()
        if not recs:
            return
        try:
            path = Path(out_dir) / SHADOW_FILENAME
            with open(path, "w", encoding="utf-8") as f:
                for r in recs:
                    f.write(json.dumps(r) + "\n")
            logger.debug(f"Wrote {len(recs)} shadow transition epoch(s) to {path}")
        except OSError as e:
            logger.warning(f"Could not write {SHADOW_FILENAME}: {e}")
