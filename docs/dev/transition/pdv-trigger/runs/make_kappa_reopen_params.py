#!/usr/bin/env python3
"""Generate the bench7 matrix — the `cooling_boost_kappa` RE-OPEN campaign (KAPPA_REOPEN_PLAN.md).

`§23` falsified the "f_kappa pushes evaporation the wrong El-Badry way" claim, and K0 (`§24`) re-read
the surviving f_kappa evidence without it. The gap that re-read exposed is the whole reason this
campaign exists: **f_kappa has never been through the L21b Theta_cum band-entry calibration that
decided f_A vs f_mix**, so the published head-to-head is two-way where it should be three-way.
Predictions P1-P5 and gates G0-G6 are pre-registered in KAPPA_REOPEN_PLAN.md sections 3 and 5 — read
those before reading a number out of this campaign.

ALL FIVE K-PHASES EMIT INTO THE ONE DIRECTORY `params/bench7/`, distinguished only by a filename
prefix, so the campaign is one sbatch array, one reduce and one download (run_bench7.sbatch;
`./sync_bench.sh bench7 submit` auto-sizes --array from the committed param count):

  k1_   bench1/2/3 x f_kappa {2,3,4,6,8,12,16,24,32} x {prod, diag}      54   the missing third leg
  k1b_  bench4/bench5 x f_kappa {2,4,8} x {prod, diag}                   12   dense fire-map only
  k2_   the 6 band configs x f_kappa {5,7,9} x prod                      18   the squeeze fine grid
  k3_   5 fate-flip arms, each emitted TWICE (_a/_b) x prod              10   determinism (P4)
  k4_   bench1/bench2 x f_mix {2,3,4,8,12,16} x {prod, diag}             24   the f_mix ladder REDO
                                                                       ----
                                                                        118

Every arm: stop_t = 5, one process per arm, theta from dictionary.jsonl accepted rows, and the
bench5/bench6 two-arm protocol — production (live `cooling_balance` -> fire map) + diagnostic
(`transition_trigger blowout` -> uncensored theta(t) to blowout = the L21b window). Single-knob by
construction: the f_kappa arms keep cooling_boost_mode=none and cooling_boost_fA=1, the f_mix arms
keep cooling_boost_kappa=1 and cooling_boost_fA=1.

⚠️ K4 IS AN ASSUMPTION, NOT A RULING — FLIP IT BEFORE SUBMITTING IF IT IS WRONG. The section-6.0(c)
maintainer ruling reads "no, redo if possible": no to the originally-planned 8-arm ride-along
(bench1/bench2 x f_mix {12,16}, which would have stitched new points onto the 2026-07-19 bench6
harvest), and "redo" is read here as re-measuring the WHOLE f_mix ladder inside bench7 so f_mix band
entry comes from one campaign, one code state, one reduce — the same in-grid bar G4 sets for f_kappa,
and the fix for exactly the flaw `§18` had to flag. That reading was not confirmed. To drop K4
instead, set F_MIX_K4 = [] (campaign falls to 94 arms and P5 is recorded NOT RUN, not missed); to
run the literal ride-along, set F_MIX_K4 = ["12", "16"] (102 arms). Nothing else changes.

f_kappa = 1 IS NOT RE-RUN. The bench5 `__none` arms already are it (`cooling_boost_kappa` is gated
x1.0 exact), and they are the K1/G2 equivalence baseline. Likewise the f_mix ladder's fm=1 point.

Benches, the exact L21b Table-1 mapping and the emit gates are IDENTICAL to make_bench5_params.py /
make_bench6_params.py (L21b Table 1 [V], LANCASTER_REFERENCE section 7b; mCloud = M_cl*(1+eps),
sfe = eps/(1+eps)). K2/K3 instead reuse the canonical theta5 config band, so the new fine-grid doses
sit on exactly the configs `data/theta5k_fire_map.csv` measured.

Emit gates (G1 — the builder self-checks; a failing gate aborts the emit BEFORE anything is written):
  1. TRINITY's own pre-run GMC plausibility validation passes for every arm (bench AND theta5
     configs) — the same `_validate_sweep_combination` path run.py sweeps use;
  2. the exact L21b mapping holds for every bench: rCloud from the post-SF gas mass matches R_cl to
     <2% (2%, not 1%, for Table-1's own 2.5-pc rounding — see make_bench5_params.py);
  3. end-to-end `read_param` load-check on every emitted file (parses, resolves, no validator
     raises — this is what catches a cross-knob/double-boost combination);
  4. the emitted count matches the per-phase arithmetic above.

Submit / reduce / download (tooling already committed; the reduce is ONE-SHOT — it declares
--extra-cols Pb,bubble_dMdt,bubble_L2Conduction,bubble_L3Intermediate because P2 and the K0.Q1b
back-reaction read them and the six default trajectory columns do not carry them):
    ./sync_bench.sh bench7 up | submit | watch | reduce | down
Analyse:  python data/make_bench7_analysis.py

Regenerate:  python docs/dev/transition/pdv-trigger/runs/make_kappa_reopen_params.py
"""

import math
import sys
from pathlib import Path

from make_theta5_params import CONFIGS
from make_theta5n_params import BASE as NORMAL_N1E3_BASE, CONFIG as NORMAL_N1E3

HERE = Path(__file__).resolve().parent
OUT = HERE / "params" / "bench7"
REPO = HERE.parents[4]  # repo root (parents[3] is docs/)

# (name, M_cl [Msun], R_cl [pc], n_H [cm^-3], eps_*) — L21b Table 1, [V] 2026-07-12.
BENCHES = {
    "bench1_m5e4_r20": (5e4, 20.0, 43.1, 0.1),
    "bench2_m1e5_r10": (1e5, 10.0, 690.0, 0.1),
    "bench3_m1e5_r5": (1e5, 5.0, 5520.0, 0.1),
    "bench4_m1e5_r2p5": (1e5, 2.5, 44200.0, 0.1),
    "bench5_m5e5_r2p5": (5e5, 2.5, 228000.0, 0.01),
}

# K1 — the decision metric. 9 doses, not 4: the deliverable is the dose-response exponent q (P1
# predicts 0.55-0.70), not just the crossing point, and a wide grid also exposes the saturation
# that made the f_mix extrapolation untrustworthy in the first place (§18).
K1_BENCHES = ["bench3_m1e5_r5", "bench2_m1e5_r10", "bench1_m5e4_r20"]
F_KAPPA_K1 = ["2", "3", "4", "6", "8", "12", "16", "24", "32"]

# K1b — fire-map completeness at the dense end only. bench4/bench5 fire at low dose into a collapse
# window, so they have no clean L21b breakout window and are excluded from the decision metric (the
# same exclusion bench6 applied to f_A).
K1B_BENCHES = ["bench4_m1e5_r2p5", "bench5_m5e5_r2p5"]
F_KAPPA_K1B = ["2", "4", "8"]

# K2 — is the K0.Q2 squeeze real, or coarse sampling? `pl2_steep` needs f_kappa >= 8 while
# `simple_cluster` condenses from f_kappa = 8 up, and theta5k never sampled between them. The 6 BAND
# configs only: the two controls (fail_repro, small_1e6) and normal_n1e3 (fires unmodified at f = 1,
# so it never tests a knob) cannot change a whole-band verdict. f_kappa 6 and 8 are reused from
# theta5k rather than re-run; only 5, 7, 9 are new.
K2_CONFIGS = [
    "simple_cluster",
    "small_dense_highsfe",
    "pl2_steep",
    "midrange_pl0",
    "be_sphere",
    "large_diffuse_lowsfe",
]
F_KAPPA_K2 = ["5", "7", "9"]

# K3 — are the non-monotonic fates physical or nondeterministic (P4)? Selection rule, applied to
# `data/theta5k_fire_map.csv`: every cell whose fate reverses against its dose neighbours. Two are
# isolated single-cell reversals, three are the grid-edge/onset reversals the squeeze rests on:
#   be_sphere            FIRED@6  -> DRAIN@8     -> FIRED@12   isolated
#   small_dense_highsfe  FIRED@4  -> CONDENSE@6  -> FIRED@8    isolated
#   pl2_steep            FIRED@12 -> CONDENSE@16               grid edge
#   normal_n1e3          FIRED@12 -> DRAIN@16                  grid edge
#   simple_cluster       FIRED@6  -> CONDENSE@8                the squeeze's upper limit
# Each is emitted twice (_a/_b) as an identical-physics pair; the reduce hashes the trajectory CSVs
# (physics columns only, no run_name), so P4 is a diff of the paired rows in bench7_hashes.csv.
K3_FLIPS = [
    ("be_sphere", "8"),
    ("small_dense_highsfe", "6"),
    ("pl2_steep", "16"),
    ("normal_n1e3", "16"),
    ("simple_cluster", "8"),
]

# K4 — see the ASSUMPTION banner in the docstring. [] drops the phase; ["12","16"] is the original
# ride-along. fm = 1 is the shared mode-none baseline and is not re-run.
K4_BENCHES = ["bench1_m5e4_r20", "bench2_m1e5_r10"]
F_MIX_K4 = ["2", "3", "4", "8", "12", "16"]

STOP_T = 5

# Verified constants for the emit-time exact-mapping check (mu_H = 1.4, LANCASTER_REFERENCE 7b).
_MH_G, _MSUN_G, _PC_CM = 1.6726e-24, 1.989e33, 3.086e18

# Keys _validate_sweep_combination reads, in INPUT units. gamma_adia is deliberately absent: its
# default.param value is the string "5/3", which the validator float()s, and its own fallback
# (5.0/3.0) is the same number.
_GMC_KEYS = (
    "dens_profile",
    "mCloud",
    "nCore",
    "rCore",
    "densPL_alpha",
    "densBE_Omega",
    "nISM",
    "mu_convert",
    "rCloud_max",
)


def _gas_rcloud_pc(m_gas_msun, n_h):
    """Homogeneous-cloud radius from gas mass at density n_H (mu_H=1.4), in pc."""
    rho = 1.4 * _MH_G * n_h
    r_cm = (3 * m_gas_msun * _MSUN_G / (4 * math.pi * rho)) ** (1 / 3)
    return r_cm / _PC_CM


def _schema_defaults():
    """The `_GMC_KEYS` defaults, parsed straight out of trinity/_input/default.param."""
    out = {}
    for line in (REPO / "trinity" / "_input" / "default.param").read_text().splitlines():
        line = line.split("#", 1)[0].strip()
        if line:
            key, _, val = line.partition(" ")
            if key in _GMC_KEYS:
                out[key] = val.strip()
    return out


def gmc_gate(arm, overrides, validate):
    """Gate 1 — GMC plausibility, on the arm's own keys over the schema defaults."""
    res = validate({**_schema_defaults(), **{k: v for k, v in overrides if k in _GMC_KEYS}})
    if res is None or not res.valid:
        sys.exit(f"ABORT {arm}: GMC validation failed: {getattr(res, 'errors', 'n/a')}")


def emit(name, base, knob_lines, diag):
    lines = [
        f"model_name             {name}",
        *(f"{k:<22} {v}" for k, v in base),
        f"{'stop_t':<22} {STOP_T}",
        *knob_lines,
    ]
    if diag:
        lines += [f"{'transition_trigger':<22} blowout"]
    lines += [
        "log_console            False",
        "log_file               True",
        f"path2output            outputs/bench7/{name}",
    ]
    (OUT / f"{name}.param").write_text("\n".join(lines) + "\n")
    return name


def kappa(f):
    return [f"{'cooling_boost_kappa':<22} {f}"]


def fmix(f):
    return [f"{'cooling_boost_mode':<22} multiplier", f"{'cooling_boost_fmix':<22} {f}"]


def bench_base(bench, validate):
    """The .param key/value base for an L21b bench, gated (G1 gates 1 + 2) before it is used."""
    m_cl, r_cl, n_h, eps = BENCHES[bench]
    mcloud, sfe = m_cl * (1 + eps), eps / (1 + eps)

    # Gate 2: the exact mapping — post-SF gas (1-sfe)*mCloud = M_cl must sit at R_cl.
    r_derived = _gas_rcloud_pc((1 - sfe) * mcloud, n_h)
    if abs(r_derived - r_cl) / r_cl >= 0.02:
        sys.exit(f"ABORT {bench}: derived rCloud {r_derived:.3f} pc != R_cl {r_cl} pc")

    base = [
        ("mCloud", f"{mcloud:.6g}"),
        ("nCore", f"{n_h:.6g}"),
        ("rCore", "1"),
        ("sfe", f"{sfe:.16g}"),
        ("dens_profile", "densPL"),
        ("densPL_alpha", "0"),
    ]
    gmc_gate(bench, base, validate)
    print(f"  {bench}: rCloud(gas)={r_derived:.2f} pc (target {r_cl})")
    return base


def main():
    sys.path.insert(0, str(REPO))
    from trinity._input.read_param import read_param
    from trinity._input.sweep_runner import _validate_sweep_combination as validate

    configs = {**CONFIGS, NORMAL_N1E3: NORMAL_N1E3_BASE}
    OUT.mkdir(parents=True, exist_ok=True)
    names = []

    print("K1 + K1b — f_kappa on the L21b benches:")
    for benches, doses, prefix in (
        (K1_BENCHES, F_KAPPA_K1, "k1"),
        (K1B_BENCHES, F_KAPPA_K1B, "k1b"),
    ):
        for bench in benches:
            base = bench_base(bench, validate)
            for f in doses:
                for diag in (False, True):
                    arm = f"{prefix}_{bench}__fk{f}" + ("_diag" if diag else "")
                    names.append(emit(arm, base, kappa(f), diag))

    print("K2 — the condensation-squeeze fine grid (prod only):")
    for cfg in K2_CONFIGS:
        gmc_gate(cfg, configs[cfg], validate)
        for f in F_KAPPA_K2:
            names.append(emit(f"k2_{cfg}__fk{f}", configs[cfg], kappa(f), False))

    print("K3 — the fate-flip determinism pairs (prod only):")
    for cfg, f in K3_FLIPS:
        gmc_gate(cfg, configs[cfg], validate)
        for rep in ("a", "b"):
            names.append(emit(f"k3_{cfg}__fk{f}_{rep}", configs[cfg], kappa(f), False))

    if F_MIX_K4:
        print("K4 — the f_mix ladder redo:")
        for bench in K4_BENCHES:
            base = bench_base(bench, validate)
            for f in F_MIX_K4:
                for diag in (False, True):
                    arm = f"k4_{bench}__fm{f}" + ("_diag" if diag else "")
                    names.append(emit(arm, base, fmix(f), diag))

    # Gate 4: the count matches the per-phase arithmetic (a silent duplicate name would overwrite).
    expected = (
        len(K1_BENCHES) * len(F_KAPPA_K1) * 2
        + len(K1B_BENCHES) * len(F_KAPPA_K1B) * 2
        + len(K2_CONFIGS) * len(F_KAPPA_K2)
        + len(K3_FLIPS) * 2
        + len(K4_BENCHES) * len(F_MIX_K4) * 2
    )
    if len(names) != expected or len(set(names)) != len(names):
        sys.exit(f"ABORT: emitted {len(names)} ({len(set(names))} unique), expected {expected}")

    # Gate 3: end-to-end load-check — every file parses, resolves and passes its validators.
    for name in names:
        try:
            read_param(OUT / f"{name}.param")
        except Exception as exc:  # noqa: BLE001 — any load failure is an abort
            sys.exit(f"ABORT {name}: read_param load-check failed: {exc}")

    print(
        f"wrote {len(names)} params to {OUT}  (K1 {len(K1_BENCHES) * len(F_KAPPA_K1) * 2}"
        f" + K1b {len(K1B_BENCHES) * len(F_KAPPA_K1B) * 2}"
        f" + K2 {len(K2_CONFIGS) * len(F_KAPPA_K2)}"
        f" + K3 {len(K3_FLIPS) * 2}"
        f" + K4 {len(K4_BENCHES) * len(F_MIX_K4) * 2}) — all 4 emit gates PASS"
    )


if __name__ == "__main__":
    main()
