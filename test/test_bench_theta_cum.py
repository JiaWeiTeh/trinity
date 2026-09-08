"""The corrected bench5/bench6 Θ_cum metric stays effective-loss-correct.

Covers docs/dev/transition/pdv-trigger/data/make_bench5_analysis.py::theta_cum_prefire and
::slope_1mtheta. FINDINGS §17 found that the old numerator integrated the traj ``Lcool`` column
(the RAW ``bubble_LTotal``), which silently drops the f_mix boost under
``cooling_boost_mode='multiplier'``; §18 fixed it to integrate ``θ·L_mech`` (θ = the effective
``bubble_Lloss``/``Lmech``, boost-correct under every mode). That bug published a whole
"f_mix eliminated by measurement" conclusion, so a regression here is expensive: pin both the
no-boost identity (the f_A-side regression gate) and the boost-carrying behaviour.
"""

import importlib.util
from pathlib import Path

import pytest

DATA = Path(__file__).resolve().parent.parent / "docs/dev/transition/pdv-trigger/data"

# docs/dev is untracked (local-only, see .gitignore) as of `a32b098`: absent in a
# fresh clone and in CI. Every test here reads that tree, so skip the module.
if not (DATA / "make_bench5_analysis.py").is_file():
    pytest.skip(
        "docs/dev is untracked (local-only); pdv-trigger data unavailable",
        allow_module_level=True,
    )


def _load(name):
    spec = importlib.util.spec_from_file_location(name, DATA / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def mod():
    return _load("make_bench5_analysis")


def _rows(ts, thetas, lcool, lmech, lleak=0.0):
    return [
        {
            "t_now": repr(t),
            "theta": repr(th),
            "Lcool": repr(lc),
            "Lleak": repr(lleak),
            "Lmech": repr(lm),
            "R2": "1.0",
        }
        for t, th, lc, lm in zip(ts, thetas, lcool, lmech)
    ]


def test_no_boost_numerator_matches_the_raw_construction(mod):
    """mode 'none'/f_A: θ == (Lcool+Lleak)/Lmech, so effective and raw must agree to ~ULP.

    This is FINDINGS §18's gate (i) — the whole f_A side of the record depends on it.
    """
    lmech = [100.0, 110.0, 120.0, 130.0]
    lcool = [20.0, 33.0, 48.0, 65.0]
    thetas = [c / m for c, m in zip(lcool, lmech)]
    eff, raw, t_end, leak = mod.theta_cum_prefire(_rows([0.1, 0.2, 0.4, 0.8], thetas, lcool, lmech))
    assert eff == pytest.approx(raw, rel=1e-12)
    assert t_end == pytest.approx(0.8)
    assert leak == 0.0


def test_multiplier_boost_is_carried_by_theta_not_by_lcool(mod):
    """mode 'multiplier': Lcool stays RAW while θ carries the boost -> eff == fmix * raw.

    The exact regression FINDINGS §17 caught: the old numerator returned `raw` here and so
    reported a *falling* cooling fraction for a run that was draining fmix x more energy.
    """
    fmix = 8.0
    lmech = [100.0, 110.0, 120.0]
    lcool = [20.0, 30.0, 42.0]
    thetas = [fmix * c / m for c, m in zip(lcool, lmech)]  # bubble_Lloss = fmix*Lcool
    eff, raw, _, _ = mod.theta_cum_prefire(_rows([0.1, 0.2, 0.4], thetas, lcool, lmech))
    assert eff == pytest.approx(fmix * raw, rel=1e-12)


def test_leak_fraction_uses_the_channel_split(mod):
    """Lleak is committed separately so the Rogers & Pittard channel check runs offline."""
    _, _, _, leak = mod.theta_cum_prefire(
        _rows([0.1, 0.2], [0.5, 0.5], [30.0, 30.0], [100.0, 100.0], lleak=10.0)
    )
    assert leak == pytest.approx(10.0 * 2 / (10.0 * 2 + 30.0 * 2))


def test_too_few_usable_rows_returns_all_none(mod):
    assert mod.theta_cum_prefire(_rows([0.1], [0.5], [30.0], [100.0])) == (None, None, None, None)


def test_slope_1mtheta_recovers_a_known_power_law(mod):
    """Phase-5 metric 2's slope half: 1−θ ∝ t^(−1/2) must fit to −0.5 (the L21b expectation)."""
    ts = [0.1, 0.2, 0.4, 0.8, 1.6]
    thetas = [1 - 0.1 * t**-0.5 for t in ts]
    rows = _rows(ts, thetas, [1.0] * 5, [100.0] * 5)
    assert mod.slope_1mtheta(rows) == pytest.approx(-0.5, abs=1e-9)


def test_slope_skips_rows_where_1_minus_theta_is_undefined(mod):
    """θ ≥ 1 (the high-dose / frozen-no-root rows of §18) has no log(1−θ) — must be dropped."""
    ts = [0.1, 0.2, 0.4, 0.8, 1.6, 3.2]
    thetas = [1 - 0.1 * t**-0.5 for t in ts] + []
    rows = _rows(ts, thetas, [1.0] * 6, [100.0] * 6)
    rows.append(
        {
            "t_now": "6.4",
            "theta": "4.6",
            "Lcool": "1.0",
            "Lleak": "0.0",
            "Lmech": "100.0",
            "R2": "1.0",
        }
    )
    assert mod.slope_1mtheta(rows) == pytest.approx(-0.5, abs=1e-9)
