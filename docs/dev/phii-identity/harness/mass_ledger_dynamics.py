#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""B11.C — does the 56% mass double-book have a supply, and does it matter dynamically?

Gates pre-registered in PLAN.md (§Batch 11 → "Pre-registered gates for B11.A–D").

G11.C1 — SUPPLY. A Strömgren-filled cavity has to be refilled as it grows. Compare the
photoevaporative supply off the shell's ionised face against what the shipped
trajectory demands:

    Mdot_supply  = 4 pi R2**2 * n_C3a * mu_convert * c_i,  c_i = sqrt(k_B T_ion / mu_ion_shell)
    Mdot_required = d/dt [ (4/3) pi R2**3 * n_C3a * mu_convert ]      (central difference)

`c_i` is the ISOTHERMAL ionised sound speed, fixed in the pre-registration so it could
not be chosen after seeing the answer.

G11.C2 — DYNAMICS. Re-integrate the momentum-phase equation of motion offline from the
first momentum snapshot, once with the shipped inertia and once with it debited by the
cavity mass the C3a premise claims. The EOM is `run_momentum_phase.get_ODE_momentum_pure`:

    dR2/dt = v2
    dv2/dt = (4 pi R2**2 (P_HII + P_ram - P_ext) - mShell_dot v2 - F_grav + F_rad) / mShell

Everything that can be made a function of the state is, so the debited run responds to
its own trajectory rather than replaying the shipped one:
  * `mShell(R2)` is analytic here — B11.0 measured `shell_mass` = cloud gas + swept ambient
    to 3e-6 on every driving row — hence `mShell_dot = 4 pi R2**2 nISM mu_convert v2`.
  * `P_ram(R2, t)` and `P_HII(R2, t)` use their shipped closed forms at the current R2,
    with `Qi(t)`, `Lmech(t)`, `v_mech(t)` and `f_abs(t)` interpolated from the run.
  * `F_rad(t)` and `P_ext(t)` are interpolated in t only. Neither has an explicit R2
    dependence in the shipped code; they enter through the shell's optical depth, which is
    not reconstructible offline. Together they are ~17% of the driving force, so this is
    the main approximation — and it is exactly what G11.C2a's control gate is for.

G11.C2a is BLOCKING: if the control (undebited) integration cannot reproduce the run's own
R2(t=1.5) to 2%, the debited result is VOID, not a small effect.

Two debit variants bracket the gravity treatment, because moving gas from the shell to the
cavity lightens the shell but does not remove it from the interior:
  inertia  — only the inertia and `mShell_dot` are debited; the gravitating mass is unchanged
  full     — `F_grav` uses the debited shell as well

    python docs/dev/phii-identity/harness/mass_ledger_dynamics.py <run_dir> \
        --out docs/dev/phii-identity/data/b11_mass_dynamics.csv
"""

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np
import scipy.integrate
from scipy.interpolate import interp1d

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from trinity._input.read_param import read_param  # noqa: E402

from _stamp import stamp  # noqa: E402

FOUR_PI = 4.0 * math.pi


def load(run_dir):
    run_dir = Path(run_dir)
    pfile = next(run_dir.glob("*.param"), None)
    if pfile is None:
        sys.exit(f"no .param in {run_dir}")
    params = read_param(str(pfile))
    meta = json.loads((run_dir / "metadata.json").read_text())
    rows = [json.loads(l) for l in (run_dir / "dictionary.jsonl").open() if l.strip()]
    return params, float(meta["rCloud"]), rows


class Model:
    """The shipped momentum EOM, rebuilt on the run's own feedback so R2 can move."""

    def __init__(self, params, rCloud, mom):
        self.G = params["G"].value
        self.mCluster = params["mCluster"].value
        self.mCloud = params["mCloud"].value
        self.rCloud = rCloud
        self.mu_c = params["mu_convert"].value
        self.rho_ism = params["nISM"].value * self.mu_c
        self.chi_e = params["chi_e_shell"].value
        self.alpha_B = params["caseB_alpha"].value
        self.pref = (self.mu_c / params["mu_ion_shell"].value) * params["k_B"].value * params[
            "TShell_ion"
        ].value
        self.c_i = math.sqrt(params["k_B"].value * params["TShell_ion"].value
                             / params["mu_ion_shell"].value)

        t = np.array([r["t_now"] for r in mom])
        def f(key):
            return interp1d(t, np.array([r.get(key) or 0.0 for r in mom]),
                            kind="linear", bounds_error=False,
                            fill_value=(mom[0].get(key) or 0.0, mom[-1].get(key) or 0.0))
        self.Qi, self.Lmech, self.vmech = f("Qi"), f("Lmech_total"), f("v_mech_total")
        self.F_rad, self.P_ext = f("F_rad"), f("press_HII_in")
        # f_abs is not persisted; recover it from the shipped P_HII, which is exactly
        # pref * sqrt(3 Qi f_abs / (4 pi chi alpha R2^3)) on every driving row (B11.0 route P == Q).
        fab = []
        for r in mom:
            n = (r.get("P_HII") or 0.0) / self.pref
            q = r["Qi"]
            fab.append(min(1.0, FOUR_PI * self.chi_e * self.alpha_B * r["R2"] ** 3 * n * n / (3.0 * q))
                       if (n > 0 and q > 0) else 1.0)
        self.f_abs = interp1d(t, np.array(fab), kind="linear", bounds_error=False,
                              fill_value=(fab[0], fab[-1]))

    def m_shell(self, R2):
        if R2 <= self.rCloud:
            return self.mCloud
        return self.mCloud + FOUR_PI / 3.0 * (R2**3 - self.rCloud**3) * self.rho_ism

    def m_shell_dot(self, R2, v2):
        return FOUR_PI * R2**2 * self.rho_ism * v2 if R2 > self.rCloud else 0.0

    def n_c3a(self, R2, t):
        q = float(self.Qi(t)) * float(self.f_abs(t))
        d = FOUR_PI * self.chi_e * self.alpha_B * R2**3
        return math.sqrt(3.0 * q / d) if (q > 0 and d > 0) else 0.0

    def m_cav(self, R2, t):
        return FOUR_PI / 3.0 * R2**3 * self.n_c3a(R2, t) * self.mu_c

    def rhs(self, t, y, debit):
        R2, v2 = max(y[0], 1e-10), y[1]
        m_sw = self.m_shell(R2)
        m_dot = self.m_shell_dot(R2, v2)
        P_ram = float(self.Lmech(t)) / (2.0 * math.pi * R2**2 * float(self.vmech(t)))
        P_HII = self.pref * self.n_c3a(R2, t)

        m_in = m_sw            # inertia
        m_gr = m_sw            # gravitating shell mass
        m_int = self.mCluster  # interior mass the shell feels
        if debit:
            m_cav = min(self.m_cav(R2, t), 0.999 * m_sw)
            m_in = m_sw - m_cav
            m_dot *= m_in / m_sw
            if debit == "full":
                m_gr = m_in
                m_int = self.mCluster + m_cav
        F_grav = self.G * m_gr / R2**2 * (m_int + 0.5 * m_gr)
        F_press = FOUR_PI * R2**2 * (P_HII + P_ram - float(self.P_ext(t)))
        return np.array([v2, (F_press - m_dot * v2 - F_grav + float(self.F_rad(t))) / max(m_in, 1e-10)])

    def integrate(self, t0, t1, R0, v0, debit=None):
        sol = scipy.integrate.solve_ivp(
            lambda t, y: self.rhs(t, y, debit), (t0, t1), [R0, v0],
            method="LSODA", rtol=1e-8, atol=1e-10, dense_output=True,
        )
        if not sol.success:
            sys.exit(f"offline integration failed ({debit}): {sol.message}")
        return sol


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run")
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()

    params, rCloud, rows = load(args.run)
    mom = [r for r in rows if r.get("current_phase") == "momentum"]
    # The phase-boundary reconciliation snapshot repeats t, which makes dM_cav/dt singular.
    # Keep the last row at each t — that is the one the next segment integrates from.
    seen = {}
    for r in mom:
        seen[r["t_now"]] = r
    mom = [seen[k] for k in sorted(seen)]
    if len(mom) < 3:
        sys.exit(f"only {len(mom)} momentum rows — B11.C2 is VOID, not a null")
    m = Model(params, rCloud, mom)

    # ---------------- G11.C1 supply ----------------
    t = np.array([r["t_now"] for r in mom])
    R2 = np.array([r["R2"] for r in mom])
    M_cav = np.array([m.m_cav(r["R2"], r["t_now"]) for r in mom])
    n_c3a = np.array([m.n_c3a(r["R2"], r["t_now"]) for r in mom])
    need = np.gradient(M_cav, t)
    supply = FOUR_PI * R2**2 * n_c3a * m.mu_c * m.c_i
    ratio = np.divide(supply, need, out=np.full_like(supply, np.nan), where=need > 0)
    good = np.isfinite(ratio)
    frac_ok = float(np.mean(ratio[good] >= 1.0)) if good.any() else float("nan")
    print(f"c_i = {m.c_i:.4f} pc/Myr  ({m.c_i * 0.9778:.2f} km/s), isothermal, pre-registered\n")
    print("=== G11.C1 supply ===")
    print(f"  Mdot_required (dM_cav/dt) {need.min():.3e}..{need.max():.3e} Msun/Myr")
    print(f"  Mdot_supply   (photoevap) {supply.min():.3e}..{supply.max():.3e} Msun/Myr")
    print(f"  ratio supply/required     {np.nanmin(ratio):.2f}..{np.nanmax(ratio):.2f}"
          f"   frac >= 1: {frac_ok:.4f}   (ADEQUATE if >= 0.95)")
    print(f"  VERDICT: {'supply adequate' if frac_ok >= 0.95 else 'SUPPLY-LIMITED'}"
          f"  — but note B11.0: shell_mass already holds 100% of the run's gas, so any"
          f" real supply must debit it")

    # ---------------- G11.C2 dynamics ----------------
    t0, t1 = mom[0]["t_now"], mom[-1]["t_now"]
    R0, v0 = mom[0]["R2"], mom[0]["v2"]
    ctrl = m.integrate(t0, t1, R0, v0, debit=None)
    R_ctrl = float(ctrl.sol(t1)[0])
    R_run = mom[-1]["R2"]
    err = abs(R_ctrl - R_run) / R_run
    print("\n=== G11.C2a control (BLOCKING) ===")
    print(f"  run R2(t={t1:.4f}) = {R_run:.4f} pc, offline control = {R_ctrl:.4f} pc"
          f"  -> |err| = {err * 100:.3f}%   (VOID if > 2%)")
    if err > 0.02:
        print("  *** G11.C2a FAILED — G11.C2b is VOID, not a null. Reported and stopped. ***")

    print("\n=== G11.C2b debited vs control ===")
    out_rows = []
    variants = {}
    for name in ("inertia", "full"):
        sol = m.integrate(t0, t1, R0, v0, debit=name)
        variants[name] = sol
        R_d = float(sol.sol(t1)[0])
        print(f"  debit={name:8s} R2(t={t1:.4f}) = {R_d:7.4f} pc"
              f"   dR2 vs control = {(R_d - R_ctrl) / R_ctrl * 100:+7.3f}%"
              f"   vs run = {(R_d - R_run) / R_run * 100:+7.3f}%"
              + ("   [VOID: control failed]" if err > 0.02 else ""))

    for i, r in enumerate(mom):
        ti = r["t_now"]
        rec = dict(
            run=Path(args.run).name, t=ti, R2_run=r["R2"], v2_run=r["v2"],
            M_cav=M_cav[i], M_shell_run=r["shell_mass"],
            M_cav_over_shell=M_cav[i] / r["shell_mass"] if r["shell_mass"] else None,
            n_c3a=n_c3a[i], Mdot_required=need[i], Mdot_supply=supply[i],
            supply_over_required=ratio[i] if np.isfinite(ratio[i]) else None,
            R2_offline_control=float(ctrl.sol(ti)[0]),
            R2_offline_debit_inertia=float(variants["inertia"].sol(ti)[0]),
            R2_offline_debit_full=float(variants["full"].sol(ti)[0]),
        )
        rec["control_rel_err"] = (rec["R2_offline_control"] - r["R2"]) / r["R2"]
        out_rows.append(rec)

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("w", newline="") as fh:
            fh.write(stamp(__file__) + "\n")
            fh.write(f"# run {args.run} (momentum rows only)\n")
            fh.write(f"# c_i = {m.c_i!r} pc/Myr (isothermal, pre-registered)\n")
            fh.write(f"# G11.C2a control error at t={t1}: {err!r}\n")
            w = csv.DictWriter(fh, fieldnames=list(out_rows[0]))
            w.writeheader()
            w.writerows(out_rows)
        print(f"\nwrote {args.out} ({len(out_rows)} rows)")


if __name__ == "__main__":
    main()
