#!/usr/bin/env python3
"""Generate the theta5k matrix — the first RULE-COMPLIANT `cooling_boost_kappa` validation.

Every prior kappa dataset is contaminated for θ quotes (stop_t=2, CONTAMINATION rule a) and,
worse, pre-dates the no-physical-root => momentum handoff (KAPPA_FREEZE_MECHANISM fix #1,
2026-07-03): runs used to FREEZE at the evaporation->condensation domain boundary instead of
handing off, which is what manufactured FINDINGS §9a's "non-monotonic dead windows"
(re-diagnosed in §9b). theta5k asks the corrected question:

    with the handoff in place, does the kappa knob fire the cooling_balance trigger
    MONOTONICALLY in f_kappa across the 8-config band, under the standing rules
    (stop_t=5, theta_max from dictionary.jsonl accepted rows)?

Matrix: 8 configs x f_kappa {1, 2, 4, 6, 8, 12, 16} = 56 runs. f=1 arms double as the
native-theta0 control (and as full-run equivalence evidence for fix #1 on healthy paths —
the handoff branch is inert unless a 50-segment no-root streak occurs). Expected outcome
classes per arm: FIRED (theta>=0.95), DRAIN (momentum with theta<0.95 — includes both Eb<=0
and no_physical_root_handoff exits; do NOT count either as a theta transition), NOFIRE.

Same protocol as theta5/theta5b (CONFIGS reused from make_theta5_params.py). Harvest:
    python runs/harvest_theta_max.py "$WS"/outputs/theta5k/* --csv runs/data/theta5k_summary.csv

Regenerate:  python docs/dev/transition/pdv-trigger/runs/make_theta5k_params.py
Run on HPC:  sbatch docs/dev/transition/pdv-trigger/runs/run_theta5k.sbatch   (array 1-56)
"""

from pathlib import Path

from make_theta5_params import CONFIGS

HERE = Path(__file__).resolve().parent
OUT = HERE / "params" / "theta5k"

F_KAPPA = ["1", "2", "4", "6", "8", "12", "16"]
STOP_T = 5


def emit(name, base, fkappa):
    lines = [
        f"model_name             {name}",
        *(f"{k:<22} {v}" for k, v in base),
        f"{'stop_t':<22} {STOP_T}",
    ]
    if fkappa != "1":
        lines += [f"{'cooling_boost_kappa':<22} {fkappa}"]
    lines += [
        "log_console            False",
        "log_file               True",
        f"path2output            outputs/theta5k/{name}",
    ]
    (OUT / f"{name}.param").write_text("\n".join(lines) + "\n")
    return name


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    names = []
    for cfg, base in CONFIGS.items():
        for f in F_KAPPA:
            mode = "none" if f == "1" else f"kappa{f.replace('.', 'p')}"
            names.append(emit(f"{cfg}__{mode}", base, f))
    print(f"wrote {len(names)} params to {OUT}")


if __name__ == "__main__":
    main()
