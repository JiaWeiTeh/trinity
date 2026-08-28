#!/usr/bin/env python3
"""Edge-case probes for ``trinity/_input/dictionary.py`` (DescribedDict).

Provenance: written 2026-08-17 against commit 030b658; each probe prints one
``PROBE-<id> [F<n>]: <verdict>`` line mapping to the findings table in
``docs/dev/dictionary-robustness/PLAN.md``.  Pure in-process probes plus one
subprocess probe (P14, real atexit); total runtime a few seconds, no simulation.

Run from the repo root:

    python docs/dev/dictionary-robustness/harness/probe_dictionary.py

Output goes to stdout; scratch dirs live in a tempdir and are removed at the
end.  Crash handlers are disabled in-process (same trick as the
``disable_crash_handlers`` fixture in ``test/test_metadata.py``) so the harness
itself neither hijacks SIGINT nor writes termination reports at exit; the
subprocess probe (P14) runs real interpreters and therefore real handlers.
"""

import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT))

from trinity._input.dictionary import DescribedDict, DescribedItem  # noqa: E402

# Keep the harness process free of atexit/signal side effects (probes create
# many dicts); subprocess probes get real handlers in their own interpreters.
DescribedDict._register_crash_handlers = lambda self: None

SCRATCH = Path(tempfile.mkdtemp(prefix="dict_probes_"))


def fresh_dir(name: str) -> Path:
    d = SCRATCH / name
    d.mkdir(parents=True)
    return d


def make_params(out: Path) -> DescribedDict:
    p = DescribedDict()
    p["path2output"] = DescribedItem(str(out))
    p["t_now"] = DescribedItem(0.0)
    p["R2"] = DescribedItem(1.0)
    return p


# ---------------------------------------------------------------- P1 / F1
# Duplicate guard at the 10-snapshot flush boundary.
out = fresh_dir("p1")
p = make_params(out)
for i in range(10):
    p["t_now"].value = 0.1 * i
    p["R2"].value = 1.0 + i
    p.save_snapshot()  # flush fires on the 10th save, buffer cleared
p.save_snapshot()  # identical (t_now, R2) to snapshot 9
dup_at_boundary = p.save_count == 11
p.save_snapshot()  # buffer now holds snap 10 -> in-window guard
dup_in_window = p.save_count == 12
p.flush()
lines = [json.loads(l) for l in (out / "dictionary.jsonl").read_text().splitlines()]
print(
    f"PROBE-1 [F1]: boundary duplicate saved={dup_at_boundary}, "
    f"in-window duplicate saved={dup_in_window}, lines={len(lines)}, "
    f"line9==line10 on (t,R2)="
    f"{(lines[9]['t_now'], lines[9]['R2']) == (lines[10]['t_now'], lines[10]['R2'])}"
)

# ---------------------------------------------------------------- P2 / F2
# Guard is also skipped after ANY flush (manual / emergency / termination).
p = make_params(fresh_dir("p2"))
for i in range(3):
    p["t_now"].value = 0.1 * i
    p.save_snapshot()
p.flush()
p.save_snapshot()  # identical to snapshot 2
print(f"PROBE-2 [F2]: duplicate saved after manual mid-window flush={p.save_count == 4}")

# ---------------------------------------------------------------- P3 / F5
# Profile-array special-case crash matrix.
p = make_params(fresh_dir("p3a"))
p["bubble_r_arr"] = DescribedItem(np.array([]))
p["bubble_T_arr"] = DescribedItem(np.array([]))
try:
    p.save_snapshot()
    print("PROBE-3a [F5a]: empty bubble arrays -> snapshot OK")
except Exception as e:
    print(f"PROBE-3a [F5a]: empty bubble arrays -> {type(e).__name__}: {e}")

p = make_params(fresh_dir("p3b"))
p["bubble_T_arr"] = DescribedItem(np.linspace(1e6, 1e4, 50))  # companion absent
try:
    p.save_snapshot()
    print("PROBE-3b [F5b]: missing bubble_r_arr -> snapshot OK")
except Exception as e:
    print(f"PROBE-3b [F5b]: missing bubble_r_arr -> {type(e).__name__}: {e}")

p = make_params(fresh_dir("p3c"))
p["bubble_r_arr"] = DescribedItem(np.nan)  # reset_keys() default
p["bubble_T_arr"] = DescribedItem(np.nan)
try:
    p.save_snapshot()
    print("PROBE-3c [F5c]: scalar-NaN bubble arrays -> snapshot OK")
except Exception as e:
    print(f"PROBE-3c [F5c]: scalar-NaN bubble arrays -> {type(e).__name__}: {e}")

p = make_params(fresh_dir("p3d"))
p["shell_grav_r"] = DescribedItem(np.array([]))
p["shell_grav_force_m"] = DescribedItem(np.array([]))
try:
    p.save_snapshot()
    print("PROBE-3d [F5d]: empty shell_grav arrays -> snapshot OK")
except Exception as e:
    print(f"PROBE-3d [F5d]: empty shell_grav arrays -> {type(e).__name__}: {e}")

# ---------------------------------------------------------------- P4 / F6
# Non-serializable value poisons flush mid-append: partial write, buffer kept.
out = fresh_dir("p4")
p = make_params(out)
p["t_now"].value = 0.5
p.save_snapshot()
p["bad"] = DescribedItem(object())
p["t_now"].value = 0.6
p.save_snapshot()
try:
    p.flush()
    print("PROBE-4 [F6]: flush with non-serializable value -> no exception (?)")
except TypeError:
    jl = out / "dictionary.jsonl"
    n_lines = len(jl.read_text().splitlines()) if jl.exists() else 0
    print(
        f"PROBE-4 [F6]: flush raised TypeError; lines on disk={n_lines}, "
        f"pending buffer size={len(p.previous_snapshot)}"
    )
    p.previous_snapshot = {}

# ---------------------------------------------------------------- P11 / F6
# First-flush retry self-heals: flush_count stays 0 -> fresh-run delete rewrites.
out = fresh_dir("p11")
p = make_params(out)
p["t_now"].value = 0.5
p.save_snapshot()
p["bad"] = DescribedItem(object())
p["t_now"].value = 0.6
p.save_snapshot()
try:
    p.flush()
except TypeError:
    pass
del p["bad"]
p.previous_snapshot["1"] = {"t_now": 0.6, "R2": 1.0}  # caller "fixes" snap 1, retries
p.flush()
lines = [json.loads(l) for l in (out / "dictionary.jsonl").read_text().splitlines()]
print(
    f"PROBE-11 [F6]: first-flush retry -> lines={len(lines)}, "
    f"t sequence={[l['t_now'] for l in lines]}"
)

# ---------------------------------------------------------------- P13 / F6
# Later-flush retry DUPLICATES the already-written line -> id/line shift.
out = fresh_dir("p13")
p = make_params(out)
p.save_snapshot()
p.flush()  # flush_count -> 1
p["t_now"].value = 1.0
p.save_snapshot()  # snap 1 (clean)
p["bad"] = DescribedItem(object())
p["t_now"].value = 2.0
p.save_snapshot()  # snap 2 (poisoned)
try:
    p.flush()
except TypeError:
    pass
del p["bad"]
p.previous_snapshot["2"] = {"t_now": 2.0, "R2": 1.0}
p.flush()
lines = [json.loads(l) for l in (out / "dictionary.jsonl").read_text().splitlines()]
print(
    f"PROBE-13 [F6]: later-flush retry -> lines={len(lines)}, "
    f"t sequence={[l['t_now'] for l in lines]}"
)

# ---------------------------------------------------------------- P5 / F10
# _excluded_keys is sticky: re-inserting a non-excluded item stays excluded.
p = make_params(fresh_dir("p5"))
p["secret"] = DescribedItem(42.0, exclude_from_snapshot=True)
p["secret"] = DescribedItem(43.0, exclude_from_snapshot=False)
p.save_snapshot()
print(
    f"PROBE-5 [F10]: replaced non-excluded key present in snapshot="
    f"{'secret' in p.previous_snapshot['0']}"
)
p.previous_snapshot = {}

# ---------------------------------------------------------------- P6 / F9
# __str__ with a 0-d numpy array value.
p = make_params(fresh_dir("p6"))
p["zero_d"] = DescribedItem(np.array(3.0))
try:
    _ = str(p)
    print("PROBE-6 [F9]: print(params) with 0-d array -> OK")
except Exception as e:
    print(f"PROBE-6 [F9]: print(params) with 0-d array -> {type(e).__name__}: {e}")

# ---------------------------------------------------------------- P7 / F8
# t_now=None survives the guard but crashes the debug-log f-string.
p = make_params(fresh_dir("p7"))
p["t_now"].value = None
try:
    p.save_snapshot()
    print("PROBE-7 [F8]: t_now=None -> save OK")
except Exception as e:
    print(f"PROBE-7 [F8]: t_now=None -> {type(e).__name__}: {e}")

# ---------------------------------------------------------------- P8 / F3+F11
# NaN t_now defeats the guard; NaN/Infinity land as non-strict JSON literals.
out = fresh_dir("p8")
p = make_params(out)
p["t_now"].value = float("nan")
p["weird"] = DescribedItem(np.array([1.0, np.inf, -np.inf, np.nan]))
p.save_snapshot()
p.flush()
raw = (out / "dictionary.jsonl").read_text().splitlines()[0]
p.save_snapshot()  # identical NaN state -> guard cannot match NaN
print(
    f"PROBE-8 [F3,F11]: NaN/Infinity literals in jsonl="
    f"{('NaN' in raw) or ('Infinity' in raw)}; "
    f"NaN t_now defeats duplicate guard={p.save_count == 2}"
)
p.previous_snapshot = {}

# ---------------------------------------------------------------- P9 / F12
# Round-trip type morphing.
out = fresh_dir("p9")
p = make_params(out)
p["names"] = DescribedItem(["alpha", "beta"])
p["pair"] = DescribedItem((1, 2))
p["flag"] = DescribedItem(True)
p["count"] = DescribedItem(7)
p.save_snapshot()
p.flush()
loaded = DescribedDict.load_snapshot(out, 0)
nv = loaded["names"].value
print(
    f"PROBE-9 [F12]: names -> {type(nv).__name__}"
    f"({nv.dtype if isinstance(nv, np.ndarray) else ''}), "
    f"pair -> {type(loaded['pair'].value).__name__}, "
    f"flag -> {type(loaded['flag'].value).__name__}, "
    f"count -> {type(loaded['count'].value).__name__}"
)

# ---------------------------------------------------------------- P10 / F13
# Blank line mid-file shifts snapshot ids on load.
out = fresh_dir("p10")
p = make_params(out)
for i in range(3):
    p["t_now"].value = float(i)
    p.save_snapshot()
p.flush()
txt = (out / "dictionary.jsonl").read_text().splitlines()
(out / "dictionary.jsonl").write_text("\n".join([txt[0], "", txt[1], txt[2]]) + "\n")
snaps = DescribedDict.load_snapshots(out)
print(
    f"PROBE-10 [F13]: ids after mid-file blank line={sorted(snaps.keys())} "
    f"(t of id '2'={snaps.get('2', {}).get('t_now')})"
)

# ---------------------------------------------------------------- P12+P14 / F7
# Merely LOADING a snapshot rewrites the run's metadata.json at interpreter
# exit, clobbering a recorded crash reason with "Normal exit / atexit".
out = fresh_dir("p14")
writer = f"""
import sys
sys.path.insert(0, {str(REPO_ROOT)!r})
from trinity._input.dictionary import DescribedDict, DescribedItem
p = DescribedDict()
p["path2output"] = DescribedItem({str(out)!r})
p["t_now"] = DescribedItem(0.0)
p["R2"] = DescribedItem(1.0)
p.save_snapshot()
p.flush()
p.set_termination_reason("ODE solver failed")   # simulate a crashed run
"""
subprocess.run([sys.executable, "-c", writer], check=True, capture_output=True)
meta_path = out / "metadata.json"
before = json.loads(meta_path.read_text())["termination_debug"]["reason"]
reader = f"""
import sys
sys.path.insert(0, {str(REPO_ROOT)!r})
from trinity._input.dictionary import DescribedDict
loaded = DescribedDict.load_snapshot({str(out)!r}, 0)
"""
subprocess.run([sys.executable, "-c", reader], check=True, capture_output=True)
after = json.loads(meta_path.read_text())["termination_debug"]["reason"]
print(
    f"PROBE-14 [F7]: termination reason before load={before!r}, "
    f"after load={after!r}, clobbered={before != after}"
)

shutil.rmtree(SCRATCH)
print("ALL PROBES DONE")
