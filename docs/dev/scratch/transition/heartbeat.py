#!/usr/bin/env python3
"""One-shot health check of the transition-trigger background hybr runs.

Prints one compact status line covering each run's state (RUN/DONE/CRASH) and
its last logged t_now. Exit 3 when all runs have finished (none alive) so a
polling monitor can stop. Used by the 10-min heartbeat Monitor.
"""
import datetime
import json
import os
import subprocess
import sys

NAMES = ["steep_long"]
BASE = "/home/user/trinity"


def last_t(model):
    jf = os.path.join(BASE, "outputs", model, "dictionary.jsonl")
    if not os.path.exists(jf):
        return None
    last = None
    with open(jf) as f:
        for ln in f:
            ln = ln.strip()
            if ln:
                last = ln
    if last:
        try:
            return json.loads(last).get("t_now")
        except Exception:
            return None
    return None


def alive(name):
    r = subprocess.run(["pgrep", "-f", f"transition/{name}.param"], capture_output=True)
    return r.stdout.strip() != b""


def crashed(name):
    lf = os.path.join(BASE, "docs/dev/scratch/transition", name + ".log")
    if not os.path.exists(lf):
        return False
    txt = open(lf, encoding="utf-8", errors="ignore").read()
    return ("Traceback (most recent" in txt) or ("ParameterFileError" in txt)


parts = []
any_alive = False
for n in NAMES:
    t = last_t("tt_" + n)
    ts = f"{t:.4f}" if isinstance(t, (int, float)) else "?"
    if alive(n):
        any_alive = True
        parts.append(f"{n}:RUN t={ts}")
    elif crashed(n):
        parts.append(f"{n}:CRASH")
    else:
        parts.append(f"{n}:DONE t={ts}")

print(f"HEARTBEAT {datetime.datetime.now():%H:%M} | " + " | ".join(parts))
sys.exit(0 if any_alive else 3)
