#!/usr/bin/env python3
"""END-TO-END science gate for the shell-ODE overflow fix (de-risk matrix).

Drives a FULL TRINITY sim with `get_shellODE` either left at production (baseline)
or monkeypatched to a candidate variant {phiguard, clip, cgs} from
get_shellODE_variants.py. Measures, for the WHOLE run:

  * wall-clock seconds,
  * total LSODA `t+h=t` / overflow-driver chatter lines (fd-level stdout/stderr),
  * total numpy overflow Python-warnings (the nShell**2 pole; np.seterr over=warn),
  * n_timesteps and the final/trajectory science (via dictionary.jsonl, compared
    separately by compare_endtoend.py).

Each run writes outputs/<model>__<idea>/dictionary.jsonl (unique model_name so
runs never clobber). Production code is untouched; the monkeypatch is restored in
`finally`. Authored 2026-06-18.

Usage (one fresh process per run — recommended; keeps state clean):
    python docs/dev/shell-solver/harness/run_endtoend.py <param> <idea>
        <idea> in {baseline, phiguard, clip, cgs}
Prints a JSON line of metrics to stdout's LAST line (rest is sim chatter).
"""
import os
import sys
import json
import time
import tempfile
import warnings
import contextlib
from pathlib import Path

sys.path.insert(0, os.getcwd())

import numpy as np

TRINITY_ROOT = Path(__file__).resolve().parents[4]
if str(TRINITY_ROOT) not in sys.path:
    sys.path.insert(0, str(TRINITY_ROOT))

HARNESS = TRINITY_ROOT / "docs" / "dev" / "shell-solver" / "harness"
sys.path.insert(0, str(HARNESS))

_LSODA_MARKERS = ("lsoda", "t + h = t", "t+h=t", "excess work", "intdy")


def _count_lsoda_lines(text):
    if not text:
        return 0
    low = text.lower()
    return sum(1 for line in low.splitlines()
               if any(m in line for m in _LSODA_MARKERS))


@contextlib.contextmanager
def _fd_capture():
    """Capture fd-level stdout+stderr (the FORTRAN LSODA chatter) to a temp file."""
    sys.stdout.flush()
    sys.stderr.flush()
    saved_out, saved_err = os.dup(1), os.dup(2)
    tmp = tempfile.TemporaryFile(mode="w+b")
    captured = {"text": ""}
    try:
        os.dup2(tmp.fileno(), 1)
        os.dup2(tmp.fileno(), 2)
        yield lambda: captured["text"]
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        os.dup2(saved_out, 1)
        os.dup2(saved_err, 2)
        os.close(saved_out)
        os.close(saved_err)
        try:
            tmp.flush()
            tmp.seek(0)
            captured["text"] = tmp.read().decode("utf-8", errors="replace")
        finally:
            tmp.close()


def _install_variant(idea):
    """Monkeypatch trinity.shell_structure.get_shellODE.get_shellODE for `idea`.
    Returns the original callable so the caller can restore it."""
    import trinity.shell_structure.get_shellODE as mod
    orig = mod.get_shellODE
    if idea == "baseline":
        return orig
    import get_shellODE_variants as V
    fn = {
        "phiguard": V.get_shellODE_phiguard,
        "clip": V.get_shellODE_clip,
        "cgs": V.get_shellODE_cgs,
    }[idea]
    mod.get_shellODE = fn
    # shell_structure.py imports the *module* and calls get_shellODE.get_shellODE,
    # so patching the module attribute suffices for the real solver call site.
    return orig


def main():
    param_path = sys.argv[1]
    idea = sys.argv[2]
    # Optional 3rd arg: stop_t override (Myr) -> identical bounded simulated
    # evolution for every variant, so trajectories are the SAME length and the
    # final/trajectory diffs are an apples-to-apples science comparison (and the
    # run stops NATURALLY, not via an external timeout that truncates mid-flush).
    stop_t = float(sys.argv[3]) if len(sys.argv) > 3 else None
    assert idea in ("baseline", "phiguard", "clip", "cgs"), idea

    from trinity._input import read_param
    from trinity.cloud_properties.validate_gmc import validate_gmc_from_params
    from trinity import main as trinity_main

    params = read_param.read_param(param_path)
    if stop_t is not None:
        params["stop_t"].value = stop_t
    # Unique output dir per (config, idea): override model_name so runs never clobber.
    base_model = params["model_name"].value
    model = f"{base_model}__{idea}"
    params["model_name"].value = model
    params["path2output"].value = str(TRINITY_ROOT / "outputs" / model)
    # Quiet the file/console logging noise but DO keep the sim running normally.
    import logging
    logging.disable(logging.CRITICAL)

    gmc = validate_gmc_from_params(params)
    if not gmc.valid:
        raise RuntimeError("GMC validation failed: " + "; ".join(gmc.errors))

    orig = _install_variant(idea)
    n_overflow = 0
    lsoda_lines = 0
    t0 = time.perf_counter()
    err = ""
    try:
        with warnings.catch_warnings(record=True) as wlist:
            warnings.simplefilter("always")
            old_seterr = np.seterr(over="warn")
            with _fd_capture() as chatter:
                try:
                    trinity_main.start_expansion(params)
                except SystemExit:
                    pass
            np.seterr(**old_seterr)
        lsoda_lines = _count_lsoda_lines(chatter())
        n_overflow = sum("overflow" in str(w.message).lower() for w in wlist)
    except Exception as exc:  # noqa: BLE001
        err = f"{type(exc).__name__}: {exc}"
    finally:
        import trinity.shell_structure.get_shellODE as mod
        mod.get_shellODE = orig
    wall = time.perf_counter() - t0

    # n_timesteps + terminal state from the jsonl we just wrote
    jsonl = TRINITY_ROOT / "outputs" / model / "dictionary.jsonl"
    n_ts = 0
    final_t = None
    end_reason = None
    if jsonl.exists():
        last = None
        with open(jsonl) as f:
            for line in f:
                if line.strip():
                    n_ts += 1
                    last = line
        if last is not None:
            d = json.loads(last)
            final_t = d.get("t_now")
            end_reason = d.get("SimulationEndReason")

    metrics = {
        "param": param_path, "idea": idea, "model": model,
        "stop_t": stop_t, "wall_s": round(wall, 3), "lsoda_lines": lsoda_lines,
        "overflow_warns": n_overflow, "n_timesteps": n_ts,
        "final_t_now": final_t, "end_reason": end_reason,
        "jsonl": str(jsonl), "error": err,
    }
    # Final line on real stdout = the JSON metrics (everything above was captured).
    print("ENDTOEND_METRICS " + json.dumps(metrics), flush=True)


if __name__ == "__main__":
    main()
