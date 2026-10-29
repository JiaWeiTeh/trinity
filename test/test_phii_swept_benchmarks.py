"""Fast single-state checks for PLAN.md §0.0.11 Step 2, the swept-formulation benchmarks (2026-10-20).

The benchmarks themselves (docs/dev/phii-identity/harness/option3_bench.py --all) are full integrations and run
on Helix; these are the single-state pieces the laptop can afford, each against an independent route:
  * Lancaster et al. 2025 I (arXiv:2505.22730) switch state: the dimensional definitions in cem_start
    (eqs. 21-24, 32, 37, 41) against the paper's dimensionless forms (C31, C39; D47-D49, C37);
  * the swept drive against eq. 28's force (B4) and the Hosokawa-Inutsuka force (B1) at one state;
  * the zero-light W47 branch (B3) giving a finite, radiation-free root;
  * option3_offline.py --summary failing closed (an absent, VOID or incomplete arm is never a pass).
The harness lives under the gitignored docs/dev/, so this module skips where it is absent.
"""

import math
import sys
from pathlib import Path

import pytest

HARNESS = Path(__file__).resolve().parents[1] / "docs" / "dev" / "phii-identity" / "harness"
if not all((HARNESS / f).exists() for f in ("option3_bench.py", "option3_offline.py", "_stamp.py")):
    pytest.skip("phii-identity harness not present (docs/dev is gitignored)", allow_module_level=True)
sys.path.insert(0, str(HARNESS))
import option3_bench as Bn  # noqa: E402
import option3_offline as O  # noqa: E402
import trinity.phase2_momentum.swept as S  # noqa: E402


def _trinity_drive(syn, rs, Ms, t):
    """Step 4 (PLAN 0.0.11.6): the root and the drive through trinity/'s swept code, not the harness's."""
    fb, P = syn.fb(t), syn.P
    P["Qi"].value, P["Li"].value, P["Ln"].value, P["Lbol"].value = fb.Qi, fb.Li, fb.Ln, fb.Lbol
    r = S.root_R2(rs, Ms, fb, P)
    assert r["ok"], r.get("why")
    F = (Bn.FOUR_PI * r["R2"] ** 2 * r["Pc"] + r["K"]
         - Bn.FOUR_PI * rs ** 2 * S.p_ext(rs, r["sp"].shell_fAbsorbedIon, P) + S.f_rad(r["sp"], P))
    return r, F


def _cem(zeta, kind):
    """Lancaster's set-up in TRINITY's own units: n = 100 cm^-3, Q = 4e50 s^-1, pure H, a 1 K rind."""
    syn = Bn.Synth("t", 100.0, 4e50, T_ion=1e4, T_neu=1.0, no_r1=(kind == "ED"))
    syn.P["TCloud"].value = 0.0        # Lancaster's medium has no pressure ahead (the default is 10 K since 2026-10-29)
    source, R_eq, R_i, t_eq, u0 = Bn.cem_start(syn, zeta, kind)
    if kind == "MD":
        syn.pdot = source
        syn._fb.Lmech_total = 0.5 * source * syn._fb.v_mech_total
    return syn, R_eq, R_i, t_eq, u0


@pytest.mark.parametrize("zeta", [0.70, 0.99])
def test_md_switch_state_matches_the_dimensionless_form(zeta):
    syn, R_eq, R_i, t_eq, u0 = _cem(zeta, "MD")
    R_St, c_i = syn.R_St(), syn.c_i()
    t_d = math.sqrt(3) / 2 * R_St / c_i                                              # eq. 17
    assert (Bn.R_ch(syn) / R_St) ** 0.25 == pytest.approx(zeta, rel=1e-12)          # eqs. 25, 33, 34
    assert R_i / R_eq == pytest.approx((1 + R_eq / Bn.R_ch(syn)) ** (1 / 3), rel=1e-12)   # eq. 32 at xi_w = 1
    assert t_eq / t_d == pytest.approx(math.sqrt(2) / 3 * zeta, rel=1e-12)          # C31
    a = 1 + 7 * math.sqrt(2) / 12 * zeta
    dxi = (R_i / R_eq) ** -3 * (3 / (2 * math.sqrt(2)) / zeta + zeta**-4 * a ** (9 / 7) * (1 - a ** (-6 / 7)))  # C39
    assert u0 == pytest.approx(dxi * R_eq / t_d, rel=1e-12)


def test_ed_switch_state_matches_the_dimensionless_form():
    zeta = 0.5
    syn, R_eq, R_i, t_eq, u0 = _cem(zeta, "ED")
    R_St, c_i = syn.R_St(), syn.c_i()
    t_d = math.sqrt(3) / 2 * R_St / c_i
    chi = 2 / 5 * math.sqrt(7 / 3) * zeta                                            # D47
    assert t_eq / t_d == pytest.approx(chi, rel=1e-12)
    k = 5 / 2 * math.sqrt(3 / 7)                                                     # D48's constant
    x = 1 + 7 / 4 * chi
    p_over_p0 = 3 / 5 * k ** (12 / 5) * zeta ** (-12 / 5) * chi ** (7 / 5) + zeta**-4 * x ** (9 / 7) * (1 - x ** (-6 / 7))
    assert u0 == pytest.approx(p_over_p0 / (R_i / R_eq) ** 3 * R_eq / t_d, rel=1e-12)   # D49 + C37, p = p0 xi^3 xi'


@pytest.mark.parametrize("zeta", [0.70, 0.99])
def test_swept_drive_is_lancaster_eq28_at_the_switch_state(zeta):
    syn, R_eq, R_i, t_eq, u0 = _cem(zeta, "MD")
    rs, Ms, _ = Bn.swept_state(syn, "momentum", R_eq, 0.0, t_eq)
    sol, F = _trinity_drive(syn, rs, Ms, t_eq)
    R_IF = float(sol["sp"].R_IF)
    assert R_IF / R_i == pytest.approx(1.0, abs=1e-6)            # measured 2026-10-20: 6e-8
    assert sol["R2"] / R_eq == pytest.approx(1.0, abs=1e-10)
    # eq. 28's force pdot (R_i/R_w)^2; the residual is the 1 K rind's share of K (measured 1e-5; trinity's drive
    # equals the harness's to the last bit here, 2026-10-24)
    assert F / (syn.pdot * (R_IF / sol["R2"]) ** 2) == pytest.approx(1.0, abs=1e-4)
    assert abs(sol["L8"]) < 1e-10


def test_swept_drive_is_the_hosokawa_inutsuka_force_at_the_b1_start():
    syn = Bn.Synth("t", Bn.SB_RHO / Bn.M_H, Bn.SB_Q, T_ion=1e4, T_neu=1.0)
    syn.P["TCloud"].value = 0.0        # Hosokawa & Inutsuka's medium has no pressure ahead (default 10 K since 2026-10-29)
    R_St = syn.R_St()
    assert R_St == pytest.approx(0.314, rel=2e-3)                # Bisbas et al. 2015 §4.1
    assert syn.c_i() / Bn.cvt.v_kms2au == pytest.approx(12.85, rel=1e-3)     # their c_i (km/s)
    syn.pdot = Bn.FOUR_PI * syn.p_ion * syn.g("nCore") * (1e-3 * R_St) ** 2       # D16
    syn._fb.Lmech_total = 0.5 * syn.pdot * syn._fb.v_mech_total
    R0 = 1.1 * R_St
    rs, Ms, sp = Bn.rs_from_RIF(syn, "momentum", R0, 0.0, 0.0)
    assert float(sp.R_IF) / R0 == pytest.approx(1.0, abs=1e-10)
    sol, F = _trinity_drive(syn, rs, Ms, 0.0)     # trinity's own bracket: no D18 needed at this state (14 solves)
    F_HI = Bn.FOUR_PI * R0**2 * syn.rho * syn.c_i() ** 2 * (R_St / R0) ** 1.5      # Bisbas eq. 6 at R_IF
    assert F / F_HI == pytest.approx(1.0, abs=1e-4)              # measured 2026-10-20: 5e-6; trinity 2026-10-24: 4.9e-6


def test_zero_light_w47_root_is_finite_and_radiation_free():
    syn = Bn.Synth("t", 100.0, 0.0, T_ion=1e4, T_neu=100.0, pdot_cgs=2e37 / 2e8, vw_kms=2000.0)
    L, t0 = syn._fb.Lmech_total, 1e-3
    R0 = (125 / (154 * math.pi)) ** 0.2 * (L * t0**3 / syn.rho) ** 0.2              # Weaver at t0
    E0 = 5 / 11 * L * t0
    rs, Ms, _ = Bn.swept_state(syn, "energy", R0, E0, t0)
    sol = O.Swept(syn, "energy", 0).root(rs, Ms, t0, E0, syn.fb(t0), dict(bubble_mass=0.0), scan=False)
    assert sol["ok"]
    assert sol["Frad"] == 0.0 and sol["Frad_sh"] == 0.0 and sol["Lbol_ratio"] == 0.0
    assert math.isfinite(sol["L8"]) and math.isfinite(sol["K"])


def _rows(cfg, dRs, complete=True, ctrl_err=0.0, phase="implicit"):
    """Two states of one config and phase: R2 = 1 pc, rShell = 1.1 pc on the run."""
    base = dict(cfg=cfg, phase=phase, mode="segment", n_rows=2, run_R2=1.0, run_R_IF=1.1, run_rShell=1.1,
                gate_replay_rShell=0.0)
    first = dict(base, k=0, t=0.1, ctrl_R2=1.0, sw_rs=1.1, sw_R2=1.0, sw_root_ok=1, sw_nsign=1, sw_scan_nan=0)
    last = dict(base, k=1, t=0.2, ctrl_R2=1.0 * (1 + ctrl_err))
    if complete:
        last.update(sw_rs=1.1 * (1 + dRs), sw_R2=1.0, sw_root_ok=1, sw_nsign=1, sw_scan_nan=0)
    return [first, last]


def _g644(tmp_path, rows):
    p = tmp_path / "b64.csv"
    O.write(rows, p, [])
    text = O.summary([p])
    return next(ln for ln in text.splitlines() if ln.strip().startswith("segment"))


def test_summary_fails_closed(tmp_path):
    # one determined hit above 5 % meets the rule even with the other two undetermined
    assert "condition MET" in _g644(tmp_path, _rows("M43B", -0.10) + _rows("ROS", 0, complete=False))
    # no hit, but ROS incomplete and B3MW001 absent: never "not met"
    line = _g644(tmp_path, _rows("M43B", -0.01) + _rows("ROS", 0, complete=False))
    assert "condition UNDETERMINED" in line and "ROS UNDETERMINED (arm incomplete)" in line
    assert "B3MW001 UNDETERMINED (arm absent)" in line
    # a control off by more than 2 % voids the phase, so its large Delta r_s is not a hit
    assert "condition UNDETERMINED" in _g644(tmp_path, _rows("M43B", -0.10, ctrl_err=0.03) + _rows("ROS", -0.01)
                                                       + _rows("B3MW001", -0.01))
    # all three determined and small: "not met"
    assert "condition not met" in _g644(tmp_path, _rows("M43B", -0.01) + _rows("ROS", -0.01) + _rows("B3MW001", 0.0))


def test_summary_fails_closed_on_g641_g645_g646(tmp_path):
    rows = _rows("M43B", -0.10) + _rows("ROS", 0, complete=False)
    mom = _rows("ROS", -0.01, phase="momentum")
    mom[0]["var_rs"] = 1.2                                       # the v2-hand-off arm stops after its first state
    nan_l8 = _rows("SSC6", 0.0)
    nan_l8[0]["sw_L8"], nan_l8[1]["sw_L8"] = float("nan"), 2e-7  # a NaN residual must not vanish from G64.6
    p = tmp_path / "b64.csv"
    O.write(rows + mom + nan_l8, p, [])
    text = O.summary([p])
    g641 = next(ln for ln in text.splitlines() if ln.startswith("G64.1"))
    assert "unreached states counted against it: 1" in g641 and "FAIL" in g641
    assert "ROS: UNDETERMINED (v2 arm incomplete)" in text
    assert "inf" in next(ln for ln in text.splitlines() if ln.startswith("G64.6")) and "FAIL" in text.split("G64.6")[1]


def test_bench_gates_fail_closed():
    def b4(t, t_end, why=""):
        return dict(bench="B4_zeta0.70", t=t, t_end=t_end, sw_rs=1.0, sw_R_IF=1.0, sw_R2=0.5, ref_R=1.0, ref_Rw=0.5,
                    why=why)
    # an arm that stopped short of its registered end, or stopped with a reason, is never a PASS
    assert "UNDETERMINED" in Bn.gates([b4(0.1, 1.0), b4(0.5, 1.0)])[0]
    assert "UNDETERMINED" in Bn.gates([b4(0.1, 1.0), b4(1.0, 1.0), dict(bench="B4_zeta0.70", t=1.0, t_end=1.0,
                                                                         why="no bracket")])[0]
    assert "PASS" in Bn.gates([b4(0.1, 1.0), b4(1.0, 1.0)])[0]
    # B5 with no energy residual at all (a two-state arm) reports, it does not crash
    b5 = [dict(bench="B5", t=t, t_end=1.0, sw_R_IF=1.0, sw_R2=0.5, sw_P=1.0, ref_R=1.0, ref_Rw=0.5, ref_P=1.0)
          for t in (0.1, 1.0)]
    assert Bn.gates(b5)[0].startswith("B5")
