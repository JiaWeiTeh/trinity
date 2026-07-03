#!/usr/bin/env python3
"""Generate the theta5n matrix — the maintainer's 9th standard config, both knobs.

Maintainer request (2026-07-03): add a "normal cloud" — mCloud 1e6, nCore 1e3, sfe 0.01,
flat profile (densPL_alpha 0) — to the standard check band. M_cluster = 1e4 Msun: a weaker
driver than any current band member at an intermediate density; it sits in the n gap between
large_diffuse (1e2) and midrange_pl0 (1e4). NB it is deliberately NOT added to
make_theta5_params.CONFIGS — the committed theta5/theta5b param sets must keep regenerating
byte-identically from their builders; from theta5n on, the band is NINE configs (PLAN rules).

Arms (15, stop_t=5, standard rules):
  multiplier f_mix in {1(none), 2, 2.5, 3, 3.5, 4, 4.5, 5, 8}  — matches the theta5+theta5b
      coverage for one config: measures theta0, the fine f_fire bracket, and whether the
      adopted f_mix=4 / window [4, 4.5] still fires the (now nine-config) band. The
      theta1-collapse law makes an out-of-sample prediction from theta0 BEFORE the fine arms
      are read — record theta0 first, predict, then check.
  kappa f_kappa in {2, 4, 6, 8, 12, 16}  — extends the theta5k matrix to the new config
      (post fix #1; expect FIRED/CONDENSE/DRAIN/NOFIRE, never a freeze).

Harvest into the standard namespace:
    python runs/harvest_theta_max.py "$WS"/outputs/theta5n/* --csv runs/data/theta5n_summary.csv

Regenerate:  python docs/dev/transition/pdv-trigger/runs/make_theta5n_params.py
Run on HPC:  sbatch docs/dev/transition/pdv-trigger/runs/run_theta5n.sbatch   (array 1-15)
"""

from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "params" / "theta5n"

CONFIG = "normal_n1e3"
BASE = [
    ("mCloud", "1e6"),
    ("nCore", "1e3"),
    ("rCore", "1"),
    ("sfe", "0.01"),
    ("dens_profile", "densPL"),
    ("densPL_alpha", "0"),
]
FMIX = ["2", "2.5", "3", "3.5", "4", "4.5", "5", "8"]
FKAPPA = ["2", "4", "6", "8", "12", "16"]
STOP_T = 5


def emit(name, extra):
    lines = [
        f"model_name             {name}",
        *(f"{k:<22} {v}" for k, v in BASE),
        f"{'stop_t':<22} {STOP_T}",
        *(f"{k:<22} {v}" for k, v in extra),
        "log_console            False",
        "log_file               True",
        f"path2output            outputs/theta5n/{name}",
    ]
    (OUT / f"{name}.param").write_text("\n".join(lines) + "\n")
    return name


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    names = [emit(f"{CONFIG}__none", [])]
    for f in FMIX:
        mode = f"mult{f.replace('.', 'p')}"
        names.append(
            emit(
                f"{CONFIG}__{mode}",
                [("cooling_boost_mode", "multiplier"), ("cooling_boost_fmix", f)],
            )
        )
    for f in FKAPPA:
        names.append(emit(f"{CONFIG}__kappa{f}", [("cooling_boost_kappa", f)]))
    print(f"wrote {len(names)} params to {OUT}")


if __name__ == "__main__":
    main()
