#!/usr/bin/env python3
"""Generate the 📏 STANDARD-PROTOCOL param matrix: 8 configs x multiplier f in {none,2,4,8}.

The maintainer's standing rules (PLAN.md 📏 boxes, 2026-06-30/07-01, re-affirmed 2026-07-01):
  1. every test run goes to >=5 Myr (stop_t 5) — or its natural physics end (recollapse,
     large_radius), never a cheapness truncation;
  2. theta is reported as THETA_MAX over the whole run (harvest_theta_max.py), NEVER at blowout;
  3. theta comes from dictionary.jsonl accepted rows (bubble_Lloss/Lmech_total), never a
     call-level observer (retraction R6).

The 8 configs are the canonical guardrail set (INDEX.md §3): the cleanroom 6
(docs/dev/transition/cleanroom/configs/) + fail_repro (heavy 5e9 PdV-dominated) + small_1e6
(healthy control, docs/dev/failed-large-clouds). The knob is `multiplier`
(cooling_boost_fmix) — the tentative production mechanism (FINDINGS §8e) whose calibration
was never re-derived after the KNOB CORRECTION (SESSION_HANDOFF next-step #1).

Every emitted .param sets a UNIQUE `path2output` under gitignored `outputs/theta5/`, so
parallel runs never collide. Regenerate:
    python docs/dev/transition/pdv-trigger/runs/make_theta5_params.py
Run (per file, separate processes; HPC: run_theta5.sbatch):
    python run.py docs/dev/transition/pdv-trigger/runs/params/theta5/<name>.param
Harvest:
    python docs/dev/transition/pdv-trigger/runs/harvest_theta_max.py outputs/theta5/* \
        --csv docs/dev/transition/pdv-trigger/runs/data/theta5_summary.csv
"""

from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "params" / "theta5"

STOP_T = "5"  # standing rule 1

# The canonical 8 (only the keys each overrides; the rest fall back to the schema
# defaults). Sources: docs/dev/transition/cleanroom/configs/*.param (first 6),
# runs/make_params.py fail_repro, docs/dev/failed-large-clouds/harness small_1e6.
CONFIGS = {
    "simple_cluster": [("mCloud", "1e5"), ("sfe", "0.3")],
    "small_dense_highsfe": [
        ("mCloud", "1e4"),
        ("nCore", "1e6"),
        ("rCore", "0.1"),
        ("sfe", "0.5"),
        ("dens_profile", "densPL"),
        ("densPL_alpha", "0"),
    ],
    "pl2_steep": [
        ("mCloud", "1e6"),
        ("nCore", "1e5"),
        ("rCore", "1"),
        ("sfe", "0.1"),
        ("dens_profile", "densPL"),
        ("densPL_alpha", "-2"),
    ],
    "midrange_pl0": [
        ("mCloud", "1e6"),
        ("nCore", "1e4"),
        ("sfe", "0.1"),
        ("dens_profile", "densPL"),
        ("densPL_alpha", "0"),
    ],
    "be_sphere": [
        ("mCloud", "1e6"),
        ("nCore", "1e4"),
        ("rCore", "1"),
        ("sfe", "0.05"),
        ("dens_profile", "densBE"),
        ("densBE_Omega", "14.0"),
    ],
    "large_diffuse_lowsfe": [
        ("mCloud", "1e7"),
        ("nCore", "1e2"),
        ("rCore", "1"),
        ("sfe", "0.01"),
        ("dens_profile", "densPL"),
        ("densPL_alpha", "0"),
    ],
    "fail_repro": [
        ("mCloud", "5e9"),
        ("sfe", "0.1"),
        ("nCore", "1e2"),
        ("PISM", "1e4"),
        ("nISM", "0.1"),
        ("dens_profile", "densPL"),
        ("densPL_alpha", "0"),
        ("ZCloud", "1"),
        ("coverFraction", "1.0"),
        ("rCloud_max", "1e9"),
        ("allowShellDissolution", "True"),
        ("stop_t_diss", "1"),
        ("stop_r", "500"),
        ("coll_r", "1"),
        ("stop_at_rCloud_nSnap", "None"),
        ("include_PHII", "True"),
    ],
    "small_1e6": [
        ("mCloud", "1e6"),
        ("sfe", "0.1"),
        ("nCore", "1e2"),
        ("PISM", "1e4"),
        ("nISM", "0.1"),
    ],
}

# arm -> the cooling-boost lines (none = production default, byte-identical; it
# measures the config's native theta_1, the "starting deficit" of FINDINGS §9).
MODES = {
    "none": [],
    "mult2": [("cooling_boost_mode", "multiplier"), ("cooling_boost_fmix", "2")],
    "mult4": [("cooling_boost_mode", "multiplier"), ("cooling_boost_fmix", "4")],
    "mult8": [("cooling_boost_mode", "multiplier"), ("cooling_boost_fmix", "8")],
}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    names = []
    for cfg, base in CONFIGS.items():
        for mode, extra in MODES.items():
            name = f"{cfg}__{mode}"
            lines = [
                f"model_name             {name}",
                *(f"{k:<22} {v}" for k, v in base),
                f"{'stop_t':<22} {STOP_T}",
                *(f"{k:<22} {v}" for k, v in extra),
                "log_console            False",
                "log_file               True",
                f"path2output            outputs/theta5/{name}",
            ]
            (OUT / f"{name}.param").write_text("\n".join(lines) + "\n")
            names.append(name)
    print(f"wrote {len(names)} params to {OUT}")


if __name__ == "__main__":
    main()
