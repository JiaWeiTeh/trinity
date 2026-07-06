#!/usr/bin/env python3
"""Emit the matrix config .param files into harness/params/.

Fixed block matches param/paperII_grid_sweep.param (the Paper II grid that
fails on Helix). Run: python harness/make_params.py
"""
import os

HERE = os.path.dirname(os.path.abspath(__file__))
PARAMS = os.path.join(HERE, "params")

# Where the runs write their output. Default keeps the historical /tmp/flc
# location (ephemeral by design; the durable artifacts are the committed CSVs).
# ponytail: one env knob, no CLI flag -- override with TRINITY_FLC_RUNROOT.
RUNROOT = os.environ.get("TRINITY_FLC_RUNROOT", "/tmp/flc")

TEMPLATE = """\
# {comment}
model_name      {name}
path2output     {runroot}/{name}

sfe             {sfe}
mCloud          {mCloud}
nCore           {nCore}
PISM            {PISM}
nISM            {nISM}

# fixed block (paperII_grid_sweep.param)
dens_profile    densPL
densPL_alpha    0
ZCloud          1
coverFraction   1.0
rCloud_max      1e9

allowShellDissolution    True
stop_t_diss     1
stop_r          500
stop_t          10
coll_r          1
stop_at_rCloud_nSnap    None

log_level       INFO
log_console     False
log_file        True
include_PHII    True
"""

# (name, comment, sfe, mCloud, nCore, PISM, nISM)
CONFIGS = [
    # --- failing band (must stop crashing) ---
    ("fail_helix",  "real Helix failing point", "0.05", "5e9", "1e2", "0",   "0.1"),
    ("fail_repro",  "local repro point",        "0.1",  "5e9", "1e2", "1e4", "0.1"),
    ("fail_pism6",  "failing band, high PISM",  "0.1",  "5e9", "1e2", "1e6", "1"),
    # --- mass threshold scan (nCore=1e2, sfe=0.1) ---
    ("mass_1e8",    "mass scan 1e8",            "0.1",  "1e8", "1e2", "1e4", "0.1"),
    ("mass_5e8",    "mass scan 5e8",            "0.1",  "5e8", "1e2", "1e4", "0.1"),
    ("mass_1e9",    "mass scan 1e9",            "0.1",  "1e9", "1e2", "1e4", "0.1"),
    # --- density scan (mCloud=5e9, sfe=0.1) ---
    ("ncore_1e3",   "density scan n=1e3",       "0.1",  "5e9", "1e3", "1e4", "0.1"),
    ("ncore_1e4",   "density scan n=1e4",       "0.1",  "5e9", "1e4", "1e4", "0.1"),
    # --- healthy controls (fix MUST be a no-op) ---
    ("small_1e5",   "healthy control 1e5",      "0.1",  "1e5", "1e2", "1e4", "0.1"),
    ("small_1e6",   "healthy control 1e6",      "0.1",  "1e6", "1e2", "1e4", "0.1"),
    ("small_1e7",   "healthy control 1e7",      "0.1",  "1e7", "1e2", "1e4", "0.1"),
]


def main():
    os.makedirs(PARAMS, exist_ok=True)
    for name, comment, sfe, mCloud, nCore, PISM, nISM in CONFIGS:
        body = TEMPLATE.format(name=name, comment=comment, sfe=sfe, runroot=RUNROOT,
                               mCloud=mCloud, nCore=nCore, PISM=PISM, nISM=nISM)
        with open(os.path.join(PARAMS, name + ".param"), "w") as f:
            f.write(body)
    print(f"wrote {len(CONFIGS)} param files to {os.path.relpath(PARAMS)}")


if __name__ == "__main__":
    main()
