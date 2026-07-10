#!/usr/bin/env python3
"""Generate the theta5s matrix — the f_A (`cooling_boost_fA`) all-9-config live validation.

theta5s is the source-term-boost analogue of theta5k (kappa): the same 📏 standard protocol
(stop_t=5, theta_max from dictionary.jsonl accepted rows), the same config band, single-knob by
construction (cooling_boost_mode=none, cooling_boost_kappa=1 on every arm — only cooling_boost_fA
varies). It asks the Phase-4 question of SOURCE_TERM_DESIGN.md §3:

    does a single f_A fire the cooling_balance trigger across the 7 FIREABLE configs (the
    multiplier's gold standard was window [4, 4.5] at 7/7), while the two controls
    (small_1e6 route-a, fail_repro PdV) pass UNCHANGED?  "works on ALL configs" is per-CLASS,
    not "all fire" — never tune f_A to make the controls fire.

Matrix: 9 configs x f_A {1, 2, 4, 6, 8, 12, 16, 24, 32} = 81 arms. The f=1 arm is `__none`
(default, byte-identical — Phase 3 proved literal byte-identity — and measures native theta0).
Grid rationale (SOURCE_TERM_DESIGN §3 Phase 4): the lit inversion predicts the whole-band f_A in
[8, 13]; the screen's laggards (midrange/pl2 at theta_max 0.89/0.85 by fA=16, effective exponent
~0.19-0.30) extrapolate their crossings to ~20-32; the grid brackets both. Bracket rule
(pre-committed): if any FIREABLE config is NOFIRE at fA=32, submit {48, 64} for that config
before reading the Phase-6 decision tree — do NOT widen the grid to force a control to fire.

The 9 configs = the canonical 8 from make_theta5_params.CONFIGS (reused byte-for-byte) + the 9th
standard config normal_n1e3 (mCloud 1e6, nCore 1e3, sfe 0.01, flat; deliberately kept out of
CONFIGS so theta5/theta5b regenerate byte-identically — same rule theta5n follows).

Expected outcome classes per arm (post fix #1, KAPPA_FREEZE_MECHANISM): FIRED (theta>=0.95),
DRAIN (momentum with theta<0.95 — includes Eb<=0 and no_physical_root_handoff exits; NOT a theta
transition), CONDENSE-handoff, NOFIRE. Phase 1 established f_A has NO reachable condensation edge
(no dMdt<=0 even at fA=512), so a live CONDENSE via f_A is not expected — a DRAIN or stay-energy
is the non-fire fate. Any freeze (theta frozen, exit 0) would falsify Phase-2/3 stability -> STOP
and write it up (do not tune around it).

Harvest into the standard namespace:
    python runs/harvest_theta_max.py "$WS"/outputs/theta5s/* --csv runs/data/theta5s_summary.csv

Regenerate:  python docs/dev/transition/pdv-trigger/runs/make_theta5s_params.py
Run on HPC:  sbatch docs/dev/transition/pdv-trigger/runs/run_theta5s.sbatch   (array 1-81)
"""

from pathlib import Path

from make_theta5_params import CONFIGS as _BAND8

HERE = Path(__file__).resolve().parent
OUT = HERE / "params" / "theta5s"

# The 9th standard config, matching make_theta5n_params.BASE exactly (kept out of the shared
# CONFIGS on purpose; see that builder). Appended so theta5s spans all nine.
NORMAL_N1E3 = [
    ("mCloud", "1e6"),
    ("nCore", "1e3"),
    ("rCore", "1"),
    ("sfe", "0.01"),
    ("dens_profile", "densPL"),
    ("densPL_alpha", "0"),
]
CONFIGS = {**_BAND8, "normal_n1e3": NORMAL_N1E3}

F_A = ["1", "2", "4", "6", "8", "12", "16", "24", "32"]
STOP_T = 5


def emit(name, base, fa):
    lines = [
        f"model_name             {name}",
        *(f"{k:<22} {v}" for k, v in base),
        f"{'stop_t':<22} {STOP_T}",
    ]
    if fa != "1":
        lines += [f"{'cooling_boost_fA':<22} {fa}"]
    lines += [
        "log_console            False",
        "log_file               True",
        f"path2output            outputs/theta5s/{name}",
    ]
    (OUT / f"{name}.param").write_text("\n".join(lines) + "\n")
    return name


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    names = []
    for cfg, base in CONFIGS.items():
        for f in F_A:
            mode = "none" if f == "1" else f"fa{f.replace('.', 'p')}"
            names.append(emit(f"{cfg}__{mode}", base, f))
    print(f"wrote {len(names)} params to {OUT}  ({len(CONFIGS)} configs x {len(F_A)} f_A)")


if __name__ == "__main__":
    main()
