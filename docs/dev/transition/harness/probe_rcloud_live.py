#!/usr/bin/env python3
"""Probe: is ``params['rCloud'].value`` finite at runtime (so F4 blowout can fire)?

Why this exists — a gotcha that cost a debugging session (2026-06-16): every
per-segment record in ``dictionary.jsonl`` shows ``rCloud: null``, which looks
like F4 (``R2 > rCloud``) can never fire. That ``null`` is a **serialization
artifact** — ``rCloud`` is a run-constant (``registry.py``: ``run_const=True``),
stored once in ``metadata.json`` and nulled in per-snapshot lines. The **live**
``params['rCloud'].value`` is finite (computed in phase0 init, before phase 1b),
so the F4 terminator in ``run_energy_implicit_phase.py`` reads a real number.

This runs only ``read_param`` + ``get_InitCloudProp`` (no phases, ~1 s) and
prints the live value + the F4 decision, so the claim is reproducible without a
full run. See ``docs/dev/transition/pshadow-design.md``.

    python docs/dev/transition/harness/probe_rcloud_live.py [path.param]

Default config = the tt_steep blowout case (rCloud ~= 23.42 pc).
"""
import logging
import sys
import tempfile

logging.disable(logging.CRITICAL)

from trinity._input import read_param  # noqa: E402
from trinity.phase0_init import get_InitCloudProp  # noqa: E402
from trinity.phase_general.transition_shadow import (  # noqa: E402
    blowout_fires,
    implicit_termination_reason,
)

STEEP_PARAM = """\
model_name    probe_rcloud
mCloud    1e6
sfe    0.01
dens_profile    densPL
densPL_alpha    -2
nCore    1e5
rCore    1
nISM    10
stop_t    4.0
log_level    ERROR
betadelta_solver    legacy
transition_trigger    cooling_or_blowout
"""


def main(path):
    params = read_param.read_param(path)
    get_InitCloudProp.get_InitCloudProp(params)  # phase0 — sets rCloud, runs before 1b
    rCloud = params["rCloud"].value
    print(f"LIVE params['rCloud'].value = {rCloud!r}  (type {type(rCloud).__name__})")
    assert isinstance(rCloud, float) and rCloud > 0, "rCloud not a positive float at runtime!"

    inside, escaped = 0.5 * rCloud, 1.1 * rCloud
    print(f"blowout_fires(R2={inside:.2f}, rCloud={rCloud:.2f}) -> {blowout_fires(inside, rCloud)}  (inside)")
    print(f"blowout_fires(R2={escaped:.2f}, rCloud={rCloud:.2f}) -> {blowout_fires(escaped, rCloud)}  (escaped)")
    # cooling not firing (ratio high): F4 must drive the decision past rCloud
    print("decision escaped:", implicit_termination_reason(
        "cooling_or_blowout", 100.0, 1.0, 0.05, escaped, rCloud))
    print("decision inside :", implicit_termination_reason(
        "cooling_or_blowout", 100.0, 1.0, 0.05, inside, rCloud))
    print("OK — live rCloud is finite; F4 fires only once R2 escapes the cloud.")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        main(sys.argv[1])
    else:
        # Write + close (flush) the temp param BEFORE reading it — calling main()
        # inside the `with` block would read a partially-flushed file.
        with tempfile.NamedTemporaryFile("w", suffix=".param", delete=False) as fh:
            fh.write(STEEP_PARAM)
            path = fh.name
        main(path)
