#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Cluster / CPU-allocation detection for TRINITY sweeps.

These helpers let the sweep runner choose a sensible default number of
parallel workers on both a personal laptop and inside a batch-scheduler
allocation (SLURM on bwForCluster Helix / bwUniCluster), without
oversubscribing shared compute nodes.

Why not just ``os.cpu_count()``?
    ``os.cpu_count()`` (and ``multiprocessing.cpu_count()``) report the
    physical core count of the whole node, *not* the cores the scheduler
    actually granted this job. On a 64-core Helix node a 4-core job would
    otherwise spin up ~31 workers and either thrash or get killed for
    exceeding its cgroup. We therefore prefer, in order:

      1. ``SLURM_CPUS_PER_TASK``  - cores granted to this task
      2. ``SLURM_CPUS_ON_NODE``   - cores granted on this node
      3. ``os.sched_getaffinity`` - cores this process may run on
                                    (Linux; respects cgroup/cpuset pinning)
      4. ``os.cpu_count()``       - last-resort physical count
"""

import os


def detect_allocated_cpus():
    """Return ``(n_cpus, source)`` for the cores actually available here.

    ``source`` is a short label naming which signal was used, so callers
    can show provenance (e.g. ``Workers: 4 (SLURM_CPUS_PER_TASK)``).
    """
    for var in ('SLURM_CPUS_PER_TASK', 'SLURM_CPUS_ON_NODE'):
        val = os.environ.get(var)
        if val and val.isdigit() and int(val) > 0:
            return int(val), var

    # Linux-only; respects the cgroup/cpuset confinement the scheduler applies.
    if hasattr(os, 'sched_getaffinity'):
        n = len(os.sched_getaffinity(0))
        if n > 0:
            return n, 'sched_getaffinity'

    return (os.cpu_count() or 1), 'cpu_count'


def get_optimal_workers():
    """Default worker count when ``--workers`` is not given.

    Inside a SLURM job (``SLURM_JOB_ID`` set) we use the *full* allocation:
    the cores are reserved for us, so there is nothing else to be polite to.

    On a laptop/workstation we stay conservative -- ``max(1, cpu//2 - 1)`` --
    leaving cores free for the editor/browser and avoiding thermal
    throttling (each worker spawns a full simulation subprocess; see the
    ``OMP_NUM_THREADS=1`` pinning in sweep_runner).
    """
    if os.environ.get('SLURM_JOB_ID'):
        return max(1, detect_allocated_cpus()[0])
    return max(1, (os.cpu_count() or 1) // 2 - 1)
