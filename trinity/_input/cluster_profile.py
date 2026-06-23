#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Per-user HPC *site profile* for ``run.py`` job submission.

Separates the *how/where to run* (scheduler settings, environment activation)
from the *what to compute* (the ``.param``).  This is the place the user sets
the cluster-specific bits ONCE — partition, walltime, memory, the array
throttle, and the ``module``/``conda`` activation snippet — instead of
hand-editing every emitted sbatch.

Discovery order:
  1. an explicit path argument,
  2. ``$TRINITY_CLUSTER_PROFILE``,
  3. ``~/.config/trinity/cluster.ini`` (XDG-style default).

A missing file yields an **empty** profile, which reproduces TRINITY's generic
sbatch defaults — so the profile is purely opt-in.  Format is stdlib INI
(``configparser``) — no new dependency.  Example::

    [sbatch]
    partition = cpu-single
    time      = 02:00:00
    mem       = 2G
    export    = NONE
    # account = ...        ; optional (also honors $SBATCH_ACCOUNT)

    [submit]
    throttle  = 150        ; %N concurrent array tasks
    chunk     = 880        ; max array tasks per submission ("auto" -> let caller decide)

    [env]
    prologue_file = ~/.config/trinity/helix_prologue.sh
    # or inline (configparser continuation lines):
    # prologue = module load devel/miniforge
    #            conda activate trinity
"""
from __future__ import annotations

import configparser
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


def default_profile_path() -> Path:
    """``~/.config/trinity/cluster.ini`` — the default discovery location."""
    return Path.home() / ".config" / "trinity" / "cluster.ini"


@dataclass
class ClusterProfile:
    """Resolved site settings.  Every field is optional; ``None``/empty means
    "use TRINITY's default", so an absent profile is a no-op."""

    partition: Optional[str] = None
    time: Optional[str] = None
    mem: Optional[str] = None
    account: Optional[str] = None
    export: Optional[str] = None       # e.g. "NONE" -> "#SBATCH --export=NONE"
    throttle: Optional[int] = None     # %N array parallelism (concurrent tasks)
    chunk: Optional[int] = None        # max array tasks per submission; None/"auto" -> caller
    prologue: str = ""                 # verbatim shell injected before `python run.py`
    source: Optional[str] = None       # the file it was loaded from (diagnostics)


def _clean(value: Optional[str]) -> Optional[str]:
    """Trim and collapse an empty string to None."""
    if value is None:
        return None
    value = value.strip()
    return value or None


def _int_or_none(value: Optional[str]) -> Optional[int]:
    """Parse a positive int; ``None``/empty/``"auto"``/garbage -> ``None``."""
    value = _clean(value)
    if value is None or value.lower() == "auto":
        return None
    try:
        n = int(value)
    except ValueError:
        return None
    return n if n > 0 else None


def _resolve_prologue(env_section, profile_dir: Path) -> str:
    """Return the verbatim env-activation snippet: ``prologue_file`` (path,
    ``~``-expanded, relative paths anchored at the profile's directory) wins
    over an inline ``prologue``; absent/unreadable -> empty string."""
    if env_section is None:
        return ""
    pf = _clean(env_section.get("prologue_file"))
    if pf:
        p = Path(os.path.expanduser(pf))
        if not p.is_absolute():
            p = profile_dir / p
        try:
            return p.read_text(encoding="utf-8").strip("\n")
        except OSError:
            return ""
    inline = env_section.get("prologue")
    return inline.strip("\n") if inline else ""


def load_profile(path=None) -> ClusterProfile:
    """Load the site profile (see module docstring for discovery + format).
    A missing file yields an empty :class:`ClusterProfile`."""
    if path is None:
        path = os.environ.get("TRINITY_CLUSTER_PROFILE") or default_profile_path()
    path = Path(path)
    if not path.exists():
        return ClusterProfile()

    # interpolation=None: shell prologues / values may contain a literal '%'
    # (e.g. SLURM patterns), which BasicInterpolation would choke on.
    cp = configparser.ConfigParser(interpolation=None)
    cp.read(path, encoding="utf-8")

    sb = cp["sbatch"] if cp.has_section("sbatch") else None
    sub = cp["submit"] if cp.has_section("submit") else None
    env = cp["env"] if cp.has_section("env") else None

    def g(section, key):
        return _clean(section.get(key)) if section is not None else None

    return ClusterProfile(
        partition=g(sb, "partition"),
        time=g(sb, "time"),
        mem=g(sb, "mem"),
        account=g(sb, "account"),
        export=g(sb, "export"),
        throttle=_int_or_none(sub.get("throttle") if sub is not None else None),
        chunk=_int_or_none(sub.get("chunk") if sub is not None else None),
        prologue=_resolve_prologue(env, path.parent),
        source=str(path),
    )
