#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""The bubble-structure LSODA solve is wrapped in ``_quiet_lsoda_fortran()`` to
suppress LSODA's Fortran "t + h = t ... solver will continue anyway" chatter, which
floods when the integrator crosses the bubble's ultra-thin conduction layer in
machine-precision sub-steps. That regime is stiff-but-finite and the solve is
verified correct (see docs/dev/performance/BUBBLE_CONDUCTION_STIFFNESS.md), so the
warning is pure noise. This pins that the context manager actually suppresses
C-level fd-1/fd-2 output and restores it afterward -- without it the "fix" is a no-op.
"""
import os

from trinity.bubble_structure.bubble_luminosity import _quiet_lsoda_fortran


def test_quiet_lsoda_suppresses_and_restores_clevel_output(tmp_path):
    # Point fd 1 at a temp file acting as "the terminal", so the test is independent
    # of pytest's own stdout capture; the Fortran prints go to this fd, not Python's.
    f = tmp_path / "fd1.txt"
    fd = os.open(str(f), os.O_WRONLY | os.O_CREAT | os.O_TRUNC)
    saved = os.dup(1)
    os.dup2(fd, 1)
    try:
        os.write(1, b"BEFORE\n")
        with _quiet_lsoda_fortran():
            os.write(1, b"INSIDE_SHOULD_VANISH\n")  # mimics the Fortran t+h=t print
        os.write(1, b"AFTER\n")
    finally:
        os.dup2(saved, 1)
        os.close(saved)
        os.close(fd)

    txt = f.read_text()
    assert "BEFORE" in txt       # normal output before the context
    assert "AFTER" in txt        # fd restored after the context
    assert "INSIDE" not in txt   # suppressed inside the context


def test_quiet_lsoda_restores_on_exception(tmp_path):
    """fds must be restored even if the wrapped solve raises (e.g. BubbleSolverError)."""
    f = tmp_path / "fd1.txt"
    fd = os.open(str(f), os.O_WRONLY | os.O_CREAT | os.O_TRUNC)
    saved = os.dup(1)
    os.dup2(fd, 1)
    try:
        try:
            with _quiet_lsoda_fortran():
                raise RuntimeError("boom")
        except RuntimeError:
            pass
        os.write(1, b"RESTORED\n")
    finally:
        os.dup2(saved, 1)
        os.close(saved)
        os.close(fd)

    assert "RESTORED" in f.read_text()
