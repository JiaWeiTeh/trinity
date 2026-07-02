#!/usr/bin/env python3
"""Generate the theta5b REFEREE matrix — fine f_mix bracket + the long diffuse arm.

Answers the two referee questions left open by theta5 (FINDINGS §10; PLAN "REFEREE DEFENSE"):

  Q1 "why exactly f_mix=4 — are 2.5/3.4/4.7 workable?"  → fine bracket f ∈ {2.5, 3, 3.5, 4.5, 5}
     on all 8 configs: pins each config's f_fire to ±0.25–0.5, measures the WORKABLE WINDOW
     (smallest f that fires the whole normal-GMC band → the over-boost ceiling), and turns the
     θ₁-collapse-law prediction (pl2 4.3, be 4.0, diffuse 3.9) into a measured sensitivity table.
  Q2 "does the diffuse cloud fire at lower f given more time?" → large_diffuse stop_t=8 arms at
     f ∈ {1, 2, 2.5} (it grazed θ=0.9552 at exactly t=5 under f=2; its native peak is t≈4.9 Myr).
     stop_t=8 EXCEEDS the ≥5 Myr rule — rule-compliant by construction.

Same protocol as theta5 (see make_theta5_params.py — CONFIGS reused from there): θ = θ_max from
dictionary.jsonl accepted rows via harvest_theta_max.py; separate processes; unique outdirs.
Harvest into the SAME summary namespace, then re-fit:
    python runs/harvest_theta_max.py "$WS"/outputs/theta5b/* --csv runs/data/theta5b_summary.csv
    python runs/make_theta5_calibration.py --csv runs/data/theta5b_summary.csv
(the fitter accepts any f via mode names below; theta5+theta5b rows can be concatenated for the
combined fit once both exist — keep the stamps of both files).

Regenerate:  python docs/dev/transition/pdv-trigger/runs/make_theta5b_params.py
Run on HPC:  sbatch docs/dev/transition/pdv-trigger/runs/run_theta5b.sbatch   (array 1-43)
"""

from pathlib import Path

from make_theta5_params import CONFIGS

HERE = Path(__file__).resolve().parent
OUT = HERE / "params" / "theta5b"

FINE_F = ["2.5", "3", "3.5", "4.5", "5"]
DIFFUSE_LONG_F = ["1", "2", "2.5"]  # stop_t=8 arms, large_diffuse only


def emit(name, base, stop_t, fmix):
    lines = [
        f"model_name             {name}",
        *(f"{k:<22} {v}" for k, v in base),
        f"{'stop_t':<22} {stop_t}",
    ]
    if fmix != "1":
        lines += [
            "cooling_boost_mode     multiplier",
            f"{'cooling_boost_fmix':<22} {fmix}",
        ]
    lines += [
        "log_console            False",
        "log_file               True",
        f"path2output            outputs/theta5b/{name}",
    ]
    (OUT / f"{name}.param").write_text("\n".join(lines) + "\n")
    return name


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    names = []
    for cfg, base in CONFIGS.items():
        for f in FINE_F:
            names.append(emit(f"{cfg}__mult{f.replace('.', 'p')}", base, "5", f))
    for f in DIFFUSE_LONG_F:
        tag = "none" if f == "1" else f"mult{f.replace('.', 'p')}"
        names.append(
            emit(f"large_diffuse_lowsfe_t8__{tag}", CONFIGS["large_diffuse_lowsfe"], "8", f)
        )
    print(f"wrote {len(names)} params to {OUT}")


if __name__ == "__main__":
    main()
