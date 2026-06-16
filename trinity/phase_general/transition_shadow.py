#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Shadow (log-only) transition-trigger diagnostics for the implicit phase (1b).

Records the first segment where each candidate implicit->momentum transition
criterion *would* fire, **without acting on it**. Production still terminates the
implicit phase on F0 (``cooling_balance``) only; F4 (blowout, ``R2 > rCloud``) is
logged here for the P-promote follow-up.

Zero production impact: this writes a sideline ``transition_shadow.jsonl`` and
never touches snapshots, result arrays, or ``params`` state, so snapshot output
stays byte-identical. See ``docs/dev/transition/pshadow-design.md`` (P-shadow).

Criteria (plan F0 / F4; see ``docs/dev/transition/TRIGGER_PLAN.md``):
  F0 instantaneous rate-ratio (current live trigger): ``(Lgain - Lloss)/Lgain < eps``
  F4 blowout (geometric):                              ``R2 > rCloud``
"""
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

SHADOW_FILENAME = "transition_shadow.jsonl"


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
        if self.F0 is None and Lgain > 0 and ratio_F0 < threshold:
            self.F0 = {"which": "F0", "t": t, "R2": R2, "rCloud": rCloud,
                       "ratio_F0": ratio_F0}
        if self.F4 is None and rCloud is not None and R2 > rCloud:
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
