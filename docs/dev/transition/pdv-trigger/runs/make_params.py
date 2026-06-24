#!/usr/bin/env python3
"""Generate the cooling-boost LIVE-run param matrix.

PLAN.md §Task B "Open next step": the matched-t edge-config LIVE runs (boosted vs
`none`, SEPARATE processes) that replace the frozen-trajectory screen and settle
constant-`f_mix` vs coupled `theta_target(Da)`.

Every emitted .param is self-contained and sets a UNIQUE `path2output` under the
gitignored `outputs/` tree, so parallel runs (and parallel git worktrees) never
collide. `cooling_boost_mode` is the opt-in knob wired in Task B (default `none`
=> byte-identical). Regenerate: python docs/dev/transition/pdv-trigger/runs/make_params.py
"""
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "params"

# base physics per config -- only the keys each overrides; the rest fall back to
# the schema defaults in trinity/_input/default.param (same as how these are
# normally launched).
CONFIGS = {
    # energy-driven baseline (normal, compact)
    "simple_cluster": [("mCloud", "1e5"), ("sfe", "0.3")],
    # diffuse massive, strong feedback -> fast/strong bubble (normal, diffuse end)
    "f1edge_lowdens": [("mCloud", "1e7"), ("sfe", "0.5"), ("nCore", "1e2")],
    # dense massive, weak feedback -> struggles vs dense gas (normal, dense end)
    "f1edge_hidens": [("mCloud", "1e7"), ("sfe", "0.01"), ("nCore", "1e6")],
    # heavy 5e9 super-critical (PdV-dominated; cooling boost should NOT rescue it)
    "fail_repro": [
        ("mCloud", "5e9"), ("sfe", "0.1"), ("nCore", "1e2"),
        ("PISM", "1e4"), ("nISM", "0.1"),
        ("dens_profile", "densPL"), ("densPL_alpha", "0"), ("ZCloud", "1"),
        ("coverFraction", "1.0"), ("rCloud_max", "1e9"),
        ("allowShellDissolution", "True"), ("stop_t_diss", "1"), ("stop_r", "500"),
        ("stop_t", "10"), ("coll_r", "1"), ("stop_at_rCloud_nSnap", "None"),
        ("include_PHII", "True"),
    ],
}

# boost mode -> extra cooling_boost lines (the Task B opt-in knob)
MODES = {
    "none": [("cooling_boost_mode", "none")],
    "mult2": [("cooling_boost_mode", "multiplier"), ("cooling_boost_fmix", "2.0")],
    "mult3": [("cooling_boost_mode", "multiplier"), ("cooling_boost_fmix", "3.0")],
}

# which modes to run per config. lowdens (frozen screen wanted f~3.8) gets the
# f=2,3 spread to read the constant-vs-coupled question live.
MATRIX = {
    "simple_cluster": ["none", "mult2"],
    "f1edge_lowdens": ["none", "mult2", "mult3"],
    "f1edge_hidens": ["none", "mult2"],
    "fail_repro": ["none", "mult2"],
}

COMMON = [("log_console", "False"), ("log_file", "True")]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    written = []
    for cfg, modes in MATRIX.items():
        for mode in modes:
            name = f"{cfg}__{mode}"
            lines = [
                f"# LIVE cooling-boost run: {cfg} / {mode}",
                "# matched-t boosted-vs-none edge test (PLAN.md Task B).",
                "# Run ONLY via run_stamped.py (separate process + provenance).",
                f"model_name      {name}",
                f"path2output     outputs/pdvlive/{name}",
            ]
            for k, v in CONFIGS[cfg] + COMMON + MODES[mode]:
                lines.append(f"{k}    {v}")
            (OUT / f"{name}.param").write_text("\n".join(lines) + "\n")
            written.append(name)
    print(f"wrote {len(written)} param files to {OUT}:")
    for n in written:
        print(f"  {n}.param")


if __name__ == "__main__":
    main()
