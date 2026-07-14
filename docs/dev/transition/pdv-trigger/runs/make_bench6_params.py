#!/usr/bin/env python3
"""Generate the bench6 matrix — the Phase-6 DECISION inputs (SOURCE_TERM_DESIGN section 3 Phase 6).

Two questions the Phase-5 result (FINDINGS section 15h) left open, one matrix to answer both,
designed for Helix (HPC restored 2026-07-13):

  A. f_A DOSE EXTENSION — where (if ever) do the diffuse benches enter the L21b band?
     Phase 5 measured bench2/bench1 Theta_cum max 0.54/0.40 at f_A=16 (band NOT reached).
     Extend the dose: bench1, bench2 x f_A {24, 32, 64, 128}; bench3 x {24, 32} (overshoot
     check above its band entry at ~16). Settles which pre-committed Phase-6 tree row applies
     ("no whole-band f_A within grid ∪ bracket" vs "band-fire and benchmark-match want
     different f_A").

  B. f_mix HEAD-TO-HEAD — same benches, same metrics, the OTHER knob:
     all 5 benches x cooling_boost_mode=multiplier, cooling_boost_fmix {2, 3, 4, 8}.
     Theta under the multiplier is ~linear in f_mix (vs f_A^~0.3), so this grid brackets the
     band for every bench. The decision metric is band-entry-dose UNIFORMITY across density:
     the knob whose calibrated dose varies less across the suite is the better single-constant.
     (The physical asymmetry is already established without sims: f_A acts inside the
     bubble-structure ODE — the structure responds and evaporation dMdt FALLS, the El-Badry
     Eq-47 sign, measured in theta5s; f_mix multiplies the resolved loss integral after the
     structure solve — the structure is frozen and dMdt is untouched by construction.)

Per-bench arms as in bench5: production (cooling_balance live -> fire map) + diagnostic
(transition_trigger=blowout -> uncensored theta(t) to blowout = the L21b window; FINDINGS 15h).
Single-knob per arm by construction (the fa arms keep mode=none; the fm arms keep fA=1).

= (2x4 + 1x2) f_A-extension x 2 arms + 5x4 f_mix x 2 arms = 20 + 40 = 60 params.

Benches, mapping, and emit gates are IDENTICAL to make_bench5_params.py (L21b Table-1 [V],
LANCASTER_REFERENCE section 7b; exact mapping mCloud=M_cl(1+eps), sfe=eps/(1+eps)).

Harvest (theta only from dictionary.jsonl accepted rows):
    python runs/harvest_bench5.py "$WS"/outputs/bench6/* \
        --csv runs/data/bench6_summary.csv --traj-dir runs/data/bench6_traj
Analyse:  python data/make_bench6_analysis.py

Regenerate:  python docs/dev/transition/pdv-trigger/runs/make_bench6_params.py
"""

import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "params" / "bench6"
REPO = HERE.parents[3]

# (name, M_cl [Msun], R_cl [pc], n_H [cm^-3], eps_*) — L21b Table 1, [V] 2026-07-12.
BENCHES = [
    ("bench1_m5e4_r20", 5e4, 20.0, 43.1, 0.1),
    ("bench2_m1e5_r10", 1e5, 10.0, 690.0, 0.1),
    ("bench3_m1e5_r5", 1e5, 5.0, 5520.0, 0.1),
    ("bench4_m1e5_r2p5", 1e5, 2.5, 44200.0, 0.1),
    ("bench5_m5e5_r2p5", 5e5, 2.5, 228000.0, 0.01),
]

# A: f_A dose extension per bench (bench4/bench5 excluded: dense, fire at <=4, collapse-window
# only — no L21b band metric to extend; high-f_A dense diag is also the known stiffness/freeze
# case, bench5_fa16_diag in 15h).
FA_EXT = {
    "bench1_m5e4_r20": ["24", "32", "64", "128"],
    "bench2_m1e5_r10": ["24", "32", "64", "128"],
    "bench3_m1e5_r5": ["24", "32"],
}
# B: f_mix head-to-head, all benches (fm=1 baseline == bench5's __none arms; not re-run).
F_MIX = ["2", "3", "4", "8"]
STOP_T = 5

_MH_G, _MSUN_G, _PC_CM = 1.6726e-24, 1.989e33, 3.086e18


def _gas_rcloud_pc(m_gas_msun, n_h):
    """Homogeneous-cloud radius from gas mass at density n_H (mu_H=1.4), in pc."""
    rho = 1.4 * _MH_G * n_h
    r_cm = (3 * m_gas_msun * _MSUN_G / (4 * math.pi * rho)) ** (1 / 3)
    return r_cm / _PC_CM


def emit(name, mcloud, ncore, sfe, knob_lines, diag):
    lines = [
        f"model_name             {name}",
        f"{'mCloud':<22} {mcloud:.6g}",
        f"{'nCore':<22} {ncore:.6g}",
        f"{'rCore':<22} 1",
        f"{'sfe':<22} {sfe:.16g}",
        f"{'dens_profile':<22} densPL",
        f"{'densPL_alpha':<22} 0",
        f"{'stop_t':<22} {STOP_T}",
    ]
    lines += knob_lines
    if diag:
        lines += [f"{'transition_trigger':<22} blowout"]
    lines += [
        "log_console            False",
        "log_file               True",
        f"path2output            outputs/bench6/{name}",
    ]
    (OUT / f"{name}.param").write_text("\n".join(lines) + "\n")
    return name


def main():
    sys.path.insert(0, str(REPO))
    from trinity._input.sweep_runner import _validate_sweep_combination

    OUT.mkdir(parents=True, exist_ok=True)
    names = []
    for bench, m_cl, r_cl, n_h, eps in BENCHES:
        mcloud = m_cl * (1 + eps)
        sfe = eps / (1 + eps)

        # Gate 2: exact mapping (2% tolerance for Table-1's own 2.5-pc rounding; see bench5).
        r_derived = _gas_rcloud_pc((1 - sfe) * mcloud, n_h)
        if abs(r_derived - r_cl) / r_cl >= 0.02:
            sys.exit(f"ABORT {bench}: derived rCloud {r_derived:.3f} pc != R_cl {r_cl} pc")

        # Gate 1: TRINITY's own pre-run GMC plausibility validation.
        res = _validate_sweep_combination(
            {
                "dens_profile": "densPL",
                "mCloud": mcloud,
                "nCore": n_h,
                "densPL_alpha": 0.0,
                "rCore": 1.0,
            }
        )
        if res is None or not res.valid:
            sys.exit(f"ABORT {bench}: GMC validation failed: {getattr(res, 'errors', 'n/a')}")

        arms = [(f"fa{v}", [f"{'cooling_boost_fA':<22} {v}"]) for v in FA_EXT.get(bench, [])]
        arms += [
            (
                f"fm{v}",
                [f"{'cooling_boost_mode':<22} multiplier", f"{'cooling_boost_fmix':<22} {v}"],
            )
            for v in F_MIX
        ]
        for tag, knob_lines in arms:
            for diag in (False, True):
                arm = f"{bench}__{tag}" + ("_diag" if diag else "")
                names.append(emit(arm, mcloud, n_h, sfe, knob_lines, diag))
        print(
            f"{bench}: rCloud(gas)={r_derived:.2f} pc (target {r_cl}), "
            f"valid={res.valid}, arms={len(arms) * 2}"
        )
    print(f"wrote {len(names)} params to {OUT}  (f_A extension 20 + f_mix head-to-head 40)")


if __name__ == "__main__":
    main()
