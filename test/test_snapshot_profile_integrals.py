"""The stored profile arrays against the solver's own numbers (docs/dev/phii-identity/weak-wind/neutral-shell/
output-profiles/PLAN.md, E10): the class of check that would have caught the step bug of 2026-11-03 (chapter4.tex,
sec:simplify_steps), in which a GMC shell's stored 100-point profile carried 70 % of its neutral mass.

Three routes, none of them a long run:
  1. a realistic shell grid at the march's resolution, pushed through the writer's own path (_clean_for_snapshot of a
     DescribedDict) and integrated back: synthetic (always available) and, once committed, a real GMC row (c3c,
     row 219, 2.69 Myr, on 367b7c0f: the row where the unpatched simplifier keeps the least neutral mass, 74 %;
     test/data/profile_row_gmc_row219.npz, written by output-profiles/harness/make_fixture.py from a
     profile_dump_every run; skipped while the file is absent);
  2. a bubble-like grid (60 000 log-spaced samples, T ~ (1 - r/R2)^(2/5), n ~ 1/T) through the same path;
  3. the quickstart smoke run (as test_run_smoke.py) read row by row: endpoints, the ionised mass against
     shell_mass_ion, the bubble mass against bubble_mass (marked stress: it spawns run.py).
The bars (BAR) are the ones of PLAN.md §7; the integrals use the solver's own rules (shell: right-point mass, left-point
column, shell_structure.py:496-499 and :942-948 @ 367b7c0f; bubble: trapezoid mass, bubble_luminosity.py:957-961).
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from trinity._input.dictionary import DescribedDict, DescribedItem

REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURE = REPO_ROOT / "test" / "data" / "profile_row_gmc_row219.npz"
BAR = 0.05           # stored-profile trapezoid against the solver's number (PLAN.md section 7, provisional until A0)
BAR_SYN = 0.01       # a synthetic shell grid (piecewise linear n) integrated back with trapezoids
BAR_BUB = 0.02       # a bubble-like grid: n convex in r, every stored interval <= 0.2 dex (E4) -> trapezoid bias <= 1.8 %
MU = 1.4             # mu_convert [Msun per (pc^-3 pc^3)], the default

# The solver's own rules (the full grid): right-point mass (shell_structure.py:496-499), left-point column (:942-948),
# trapezoid bubble mass (bubble_luminosity.py:957-961). The stored (coarse) profile is always integrated with
# trapezoids: a right-point sum over a 100-point subsample is first-order and would fail on its own.


@pytest.fixture
def no_handlers(monkeypatch):
    """No atexit/signal hooks from the constructed dicts (as test_dictionary_stress.py)."""
    monkeypatch.setattr(DescribedDict, "_register_crash_handlers", lambda self: None)


def _snapshot(**arrays) -> dict:
    """The writer's JSON-ready row for these raw profile arrays (plus the scalars the writer needs)."""
    d = DescribedDict()
    d["t_now"] = DescribedItem(1.0)
    d["R2"] = DescribedItem(1.0)
    for k, v in arrays.items():
        d[k] = DescribedItem(np.asarray(v))
    return d._clean_for_snapshot(0)


def _right_point_mass(r, n):
    """The march's rule: sum over n[1:] mu 4 pi r[1:]^2 diff(r)."""
    return float(np.sum(n[1:] * MU * 4 * np.pi * r[1:] ** 2 * np.diff(r)))


def _trap(f, x):
    return float(np.sum(0.5 * (f[1:] + f[:-1]) * np.diff(x)))


def _gmc_shell():
    """A shell-first GMC shell at 3 Myr (main 32523375's ladder): ionised from R2 to R_IF, then the PDR layer's atomic
    zone, the H2 step (x 11/6) and warm H2 to the cold switch, the cold step (x 101/20) and flat cold gas to the edge.
    Every zone boundary is two samples at one radius, as the march stores them; 1000 (ionised) / 5000 (neutral) samples
    per zone; n in pc^-3 (cm^-3 x pc^3) so the writer's log10 sees the stored unit."""
    pc3 = 3.0856775814913673e18 ** 3
    r2, r_if = 17.583, 30.224

    def zone(x0, x1, n0, n1, k):
        x = np.linspace(x0, x1, k)
        return x, (n0 + (n1 - n0) * (x - x0) / (x1 - x0)) * pc3

    parts = [zone(r2, r_if, 2.0, 16.0, 1000), zone(r_if, r_if + 0.0097, 3308.0, 3334.5, 5000),
             zone(r_if + 0.0097, r_if + 0.1069, 3334.5 * 11 / 6, 6440.8, 5000),
             zone(r_if + 0.1069, r_if + 0.1586, 6440.8 * 101 / 20, 32612.0, 5000)]
    r = np.concatenate([p[0] for p in parts])
    n = np.concatenate([p[1] for p in parts])
    return r, n, r_if


class TestShellIntegrals:
    def test_synthetic_gmc_shell_masses_and_column(self, no_handlers):
        r, n, r_if = _gmc_shell()
        snap = _snapshot(shell_r_arr=r, shell_n_arr=n)
        rs = np.asarray(snap["shell_r_arr"], float)
        ns = 10.0 ** np.asarray(snap["log_shell_n_arr"], float)
        assert rs[0] == r[0] and rs[-1] == r[-1]
        for sel_f, sel_s in ((r <= r_if, rs <= r_if), (r >= r_if, rs >= r_if)):
            full = _right_point_mass(r[sel_f], n[sel_f])
            kept = _trap(4 * np.pi * rs[sel_s] ** 2 * ns[sel_s] * MU, rs[sel_s])
            assert abs(kept / full - 1) < BAR_SYN, (kept / full)
        col_full = float(np.sum(n[:-1] * np.diff(r)))
        col_kept = _trap(ns, rs)
        assert abs(col_kept / col_full - 1) < BAR_SYN, (col_kept / col_full)

    @pytest.mark.skipif(not FIXTURE.is_file(), reason="test/data/profile_row_gmc_row219.npz not committed yet")
    def test_real_gmc_row_masses_against_the_solver(self, no_handlers):
        with np.load(FIXTURE) as z:
            r, n = z["shell_r_arr"], z["shell_n_arr"]
            r_if, m_ion, m_neu = float(z["R_IF"]), float(z["shell_mass_ion"]), float(z["shell_mass_neutral"])
            col, mu = float(z["shell_tauKappaRatio"]), float(z["mu_convert"])
        snap = _snapshot(shell_r_arr=r, shell_n_arr=n)
        rs = np.asarray(snap["shell_r_arr"], float)
        ns = 10.0 ** np.asarray(snap["log_shell_n_arr"], float)
        assert rs[0] == r[0] and rs[-1] == r[-1]
        assert int(np.sum(rs == r_if)) == 2, "both samples at the ionisation front must survive"
        ion, neu = rs <= r_if, rs >= r_if
        m_ion_s = _trap(4 * np.pi * rs[ion] ** 2 * ns[ion] * mu, rs[ion])
        m_neu_s = _trap(4 * np.pi * rs[neu] ** 2 * ns[neu] * mu, rs[neu])
        col_s = mu * _trap(ns, rs)
        assert abs(m_ion_s / m_ion - 1) < BAR, m_ion_s / m_ion
        assert abs(m_neu_s / m_neu - 1) < BAR, m_neu_s / m_neu       # 0.74 before the step fix
        assert abs(col_s / col - 1) < BAR, col_s / col


class TestBubbleIntegrals:
    def test_bubble_like_grid_mass_and_temperature(self, no_handlers):
        R1, R2 = 0.02, 5.0
        r = np.sort((R2 + R1) - np.logspace(np.log10(R1), np.log10(R2 * 0.999), 60000))[::-1]   # descending, dense at R2
        T = 1.0e7 * np.clip(1 - r / R2, 1e-4, None) ** 0.4
        n = 1.0e7 / T * 3.0856775814913673e18 ** 3 * 1e-3                                       # n ~ 1/T, pc^-3
        snap = _snapshot(bubble_r_arr=r, bubble_T_arr=T, bubble_n_arr=n, bubble_dTdr_arr=np.gradient(T, r),
                         bubble_v_arr=np.full(r.size, 1.0))
        rn = np.asarray(snap["bubble_n_arr_r_arr"], float)
        ns = 10.0 ** np.asarray(snap["log_bubble_n_arr"], float)
        ra, na = r[::-1], n[::-1]
        o = np.argsort(rn)
        full = _trap(4 * np.pi * ra ** 2 * na * MU, ra)
        kept = _trap(4 * np.pi * rn[o] ** 2 * ns[o] * MU, rn[o])
        assert abs(kept / full - 1) < BAR_BUB, kept / full
        rt = np.asarray(snap["bubble_T_arr_r_arr"], float)
        Ts = 10.0 ** np.asarray(snap["log_bubble_T_arr"], float)
        o = np.argsort(rt)
        tavg_full = 3 * _trap(T[::-1] * ra ** 2, ra) / (ra[-1] ** 3 - ra[0] ** 3)
        tavg_kept = 3 * _trap(Ts[o] * rt[o] ** 2, rt[o]) / (rt[o][-1] ** 3 - rt[o][0] ** 3)
        assert abs(tavg_kept / tavg_full - 1) < BAR_BUB, tavg_kept / tavg_full


@pytest.mark.stress
def test_smoke_run_rows_agree_with_their_scalars(tmp_path):
    """Every row of a short real run: shell endpoints, the ionised mass against shell_mass_ion and the bubble mass
    against bubble_mass (the neutral layer does not form within stop_t; the GMC fixture covers it)."""
    param = tmp_path / "smoke.param"
    param.write_text("mCloud      1e5\n" "sfe         0.3\n" "stop_t      1e-4\n" "model_name  smoke\n" "TCloud      0\n")
    res = subprocess.run([sys.executable, str(REPO_ROOT / "run.py"), str(param)], cwd=tmp_path,
                         capture_output=True, text=True, timeout=600)
    assert res.returncode == 0, res.stderr[-2000:]
    run_dir = tmp_path / "outputs" / "smoke"
    mu = float(json.loads((run_dir / "metadata.json").read_text())["mu_convert"])
    rows = [json.loads(ln) for ln in (run_dir / "dictionary.jsonl").read_text().splitlines() if ln.strip()]
    checked = 0
    for r in rows:
        if r.get("shell_r_arr") and r.get("shell_mass_ion", 0) > 0:
            x = np.asarray(r["shell_r_arr"], float)
            n = 10.0 ** np.asarray(r["log_shell_n_arr"], float)
            assert np.isclose(x[0], r["R2"], rtol=1e-6) and np.isclose(x[-1], r["rShell"], rtol=1e-6)
            ion = x <= r["R_IF"]
            if ion.sum() > 1:
                m = _trap(4 * np.pi * x[ion] ** 2 * n[ion] * mu, x[ion])
                assert abs(m / r["shell_mass_ion"] - 1) < BAR, (r["t_now"], m / r["shell_mass_ion"])
                checked += 1
        if r.get("bubble_n_arr_r_arr") and r.get("bubble_mass", 0) > 0:
            x = np.asarray(r["bubble_n_arr_r_arr"], float)
            n = 10.0 ** np.asarray(r["log_bubble_n_arr"], float)
            o = np.argsort(x)
            m = _trap(4 * np.pi * x[o] ** 2 * n[o] * mu, x[o])
            assert abs(m / r["bubble_mass"] - 1) < BAR, (r["t_now"], m / r["bubble_mass"])
            checked += 1
    assert checked > 0, "no row carried a shell or bubble profile with a positive mass"
