#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""One-shot HPC submission orchestration for ``run.py --submit``.

Turns an already-emitted job bundle into submitted SLURM array(s) + an
auto-collect job, doing for the user what they previously hand-rolled:

  * **chunk** a grid larger than the site cap into multiple arrays, computing
    the ``OFFSET`` of each chunk automatically (no hand-written ``0 880 1760``);
  * **submit** each chunk with the throttle ``%N`` (from the profile);
  * **feed** past a full queue: on ``QOSMaxSubmitJobPerUserLimit`` wait + retry
    the same chunk (the cap is on pending+running jobs, so a feeder is
    intrinsic — we just stop the user from babysitting one in tmux);
  * **auto-collect**: chain ``sbatch --dependency=afterany:<last> --wrap
    "python run.py --collect <bundle>"`` so the report writes itself.

All scheduler interaction goes through an injectable ``runner`` (and the wait
through ``sleep``), so the whole flow is unit-testable with a fake ``sbatch`` —
no real SLURM required. ``run.py`` supplies the real ``subprocess`` runner and
``time.sleep``.
"""
from __future__ import annotations

import re
import subprocess
from typing import Callable, List, Optional, Tuple

# ``Submitted batch job 12345`` — SLURM's success line on stdout.
_JOBID_RE = re.compile(r"Submitted batch job (\d+)")

# Substrings SLURM emits when the per-user submit cap is hit; treated as
# "wait and retry" rather than a hard error.
_QUEUE_FULL_TOKENS = (
    "QOSMaxSubmitJobPerUserLimit",
    "QOSMaxSubmitJob",
    "AssocMaxSubmitJobLimit",
    "MaxSubmitJobs",
)


def compute_chunks(n: int, chunk: Optional[int]) -> List[Tuple[int, int]]:
    """Split ``n`` array tasks into ``(offset, size)`` chunks of at most
    ``chunk`` each. ``chunk`` falsy or ``>= n`` -> a single ``[(0, n)]``
    (no chunking). Offsets are the ``OFFSET`` the emitted sbatch adds to
    ``SLURM_ARRAY_TASK_ID``, so chunk ``(off, sz)`` covers runs.tsv lines
    ``off+1 .. off+sz``."""
    if n <= 0:
        return []
    if not chunk or chunk >= n:
        return [(0, n)]
    chunks = []
    offset = 0
    while offset < n:
        chunks.append((offset, min(chunk, n - offset)))
        offset += chunk
    return chunks


def parse_job_id(text: str) -> Optional[str]:
    """Extract the SLURM job id from ``sbatch`` output, or None."""
    m = _JOBID_RE.search(text or "")
    return m.group(1) if m else None


def is_queue_full(text: str) -> bool:
    """True when ``sbatch`` output indicates the per-user submit cap (retryable)."""
    return any(tok in (text or "") for tok in _QUEUE_FULL_TOKENS)


def _default_runner(cmd: List[str]):
    """Real scheduler call: run ``cmd`` capturing combined output."""
    return subprocess.run(cmd, capture_output=True, text=True)


def _combined(proc) -> str:
    return (getattr(proc, "stdout", "") or "") + (getattr(proc, "stderr", "") or "")


def build_array_command(sbatch_path, offset: int, size: int,
                        throttle: Optional[int]) -> List[str]:
    """``sbatch --export=OFFSET=<offset> --array=1-<size>[%throttle] <sbatch>``.
    The CLI ``--array``/``--export`` override the script's own directives."""
    array = f"1-{size}" + (f"%{throttle}" if throttle else "")
    return ["sbatch", f"--export=OFFSET={offset}", f"--array={array}", str(sbatch_path)]


def build_collect_command(last_job_id: str, collect_cmd: str) -> List[str]:
    """``sbatch --dependency=afterany:<last> --wrap "<collect_cmd>"`` — runs the
    report once the final chunk finishes (``afterany`` = even on partial fail)."""
    return ["sbatch", f"--dependency=afterany:{last_job_id}", "--wrap", collect_cmd]


def feed_and_collect(
    *,
    sbatch_path,
    n_jobs: int,
    throttle: Optional[int],
    chunk: Optional[int],
    collect_cmd: Optional[str],
    skip_offsets=(),
    runner: Callable = _default_runner,
    sleep: Callable[[float], None] = None,
    retry_wait: float = 300.0,
    max_retries: int = 288,
    log: Callable[[str], None] = print,
):
    """Submit all chunks (retrying past a full queue) then chain auto-collect.

    Returns ``(submitted, collect_job_id)`` where ``submitted`` is a list of
    ``(offset, size, job_id)``. ``skip_offsets`` lets ``--resume`` skip chunks
    already submitted. ``max_retries`` caps per-chunk queue-full retries
    (default 288 ≈ 24 h at 300 s) so a permanently-stuck queue eventually errors
    rather than spinning forever. A non-queue-full ``sbatch`` failure raises.
    """
    if sleep is None:
        import time
        sleep = time.sleep

    chunks = compute_chunks(n_jobs, chunk)
    skip = set(skip_offsets)
    submitted: List[Tuple[int, int, str]] = []
    last_job_id: Optional[str] = None

    for i, (offset, size) in enumerate(chunks, 1):
        if offset in skip:
            log(f"chunk {i}/{len(chunks)} offset {offset}: already submitted, skipping")
            continue
        retries = 0
        while True:
            cmd = build_array_command(sbatch_path, offset, size, throttle)
            proc = runner(cmd)
            text = _combined(proc)
            job_id = parse_job_id(text)
            if job_id:
                log(f"chunk {i}/{len(chunks)} offset {offset} (1-{size}) -> job {job_id}")
                submitted.append((offset, size, job_id))
                last_job_id = job_id
                break
            if is_queue_full(text):
                retries += 1
                if retries > max_retries:
                    raise RuntimeError(
                        f"chunk {i} (offset {offset}): queue still full after "
                        f"{max_retries} retries; giving up. Resume later with "
                        f"--resume."
                    )
                log(f"chunk {i}/{len(chunks)} offset {offset}: queue full; "
                    f"waiting {retry_wait:.0f}s (retry {retries}/{max_retries})")
                sleep(retry_wait)
                continue
            raise RuntimeError(
                f"chunk {i} (offset {offset}): sbatch failed (rc="
                f"{getattr(proc, 'returncode', '?')}):\n{text.strip()}"
            )

    collect_job_id = None
    if collect_cmd and last_job_id is not None:
        proc = runner(build_collect_command(last_job_id, collect_cmd))
        collect_job_id = parse_job_id(_combined(proc))
        log(f"auto-collect (afterany:{last_job_id}) -> job {collect_job_id}")

    return submitted, collect_job_id
