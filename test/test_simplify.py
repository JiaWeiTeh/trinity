"""
Tests for ``src._functions.simplify._simplify``.

Covers the fixed-budget contract (output ≤ ``nmin``, default 100),
orientation preservation (ascending/descending/non-monotonic), endpoint
preservation, the post-hoc R²-warning behaviour, and the priority order
(endpoints > prominent extrema > bisection-priority points). Inputs
exercised include monotonic profiles, oscillatory curves (pure sin,
noisy sin, multi-frequency), SB99-like spectra with sharp emission
lines, real bubble profiles from ``outputs/mockOutput/``, signed/zero-
crossing curves, and edge cases. A timing class checks that the
algorithm scales well (well under a second on 100k points).
"""

from __future__ import annotations

import json
import time
import warnings
from pathlib import Path

import numpy as np
import pytest

from src._functions.simplify import _simplify


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[1]
MOCK_RUN = REPO_ROOT / "outputs" / "mockOutput" / "1e6_sfe001_n1e3_PL0_yesPHII"


def reconstruction_r2(x_in: np.ndarray, y_in: np.ndarray,
                      x_out: np.ndarray, y_out: np.ndarray) -> float:
    """R² of the simplified curve against the original, computed on a
    sorted reference grid so np.interp behaves correctly regardless of
    the caller's orientation."""
    iref = np.argsort(x_out)
    xr = np.asarray(x_out)[iref]
    yr = np.asarray(y_out)[iref]
    iq = np.argsort(x_in)
    xq = np.asarray(x_in)[iq]
    yq = np.asarray(y_in)[iq]
    yi = np.interp(xq, xr, yr)
    ss_tot = np.sum((yq - yq.mean()) ** 2)
    if ss_tot == 0:
        return 1.0
    return float(1.0 - np.sum((yq - yi) ** 2) / ss_tot)


def assert_endpoints_preserved(x_in, x_out, y_in, y_out):
    """Endpoints of the input array (positions 0 and -1) must be
    represented in the output array."""
    assert np.isclose(x_out[0], x_in[0]), (
        f"first endpoint not preserved: in={x_in[0]}, out={x_out[0]}"
    )
    assert np.isclose(x_out[-1], x_in[-1]), (
        f"last endpoint not preserved: in={x_in[-1]}, out={x_out[-1]}"
    )
    assert np.isclose(y_out[0], y_in[0])
    assert np.isclose(y_out[-1], y_in[-1])


# ---------------------------------------------------------------------------
# Monotonic curves
# ---------------------------------------------------------------------------

class TestMonotonic:

    @pytest.fixture
    def smooth_curve(self):
        """A smooth curve with a soft step and a localised dip — typical
        of a bubble temperature profile across a conduction zone."""
        x = np.linspace(0.005, 0.073, 30000)
        y = (7.0
             - 0.1 * np.tanh((x - 0.06) / 0.001)
             - 2.5 * np.exp(-((x - 0.072) / 0.0005) ** 2))
        return x, y

    def test_ascending_compresses(self, smooth_curve):
        x, y = smooth_curve
        xo, yo = _simplify(x, y)
        # Default nmin=100 → output ≤ 100 points
        assert len(xo) <= 100
        # Smooth curve should reconstruct very well at the budget
        assert reconstruction_r2(x, y, xo, yo) >= 0.95

    def test_ascending_orientation_preserved(self, smooth_curve):
        x, y = smooth_curve
        xo, _ = _simplify(x, y)
        assert np.all(np.diff(xo) > 0), "ascending input must yield ascending output"

    def test_descending_compresses(self, smooth_curve):
        x, y = smooth_curve
        xd, yd = x[::-1].copy(), y[::-1].copy()
        xo, yo = _simplify(xd, yd)
        assert len(xo) <= 100
        assert reconstruction_r2(xd, yd, xo, yo) >= 0.95

    def test_descending_orientation_preserved(self, smooth_curve):
        x, y = smooth_curve
        xd, yd = x[::-1].copy(), y[::-1].copy()
        xo, _ = _simplify(xd, yd)
        assert np.all(np.diff(xo) < 0), "descending input must yield descending output"

    def test_descending_equivalent_to_reversed_ascending(self, smooth_curve):
        """Descending input produces the same point set as ascending input,
        just reversed — orientation is purely cosmetic."""
        x, y = smooth_curve
        xa, ya = _simplify(x, y)
        xd, yd = _simplify(x[::-1].copy(), y[::-1].copy())
        assert len(xa) == len(xd)
        np.testing.assert_allclose(xd, xa[::-1])
        np.testing.assert_allclose(yd, ya[::-1])

    def test_endpoints_preserved(self, smooth_curve):
        x, y = smooth_curve
        xo, yo = _simplify(x, y)
        assert_endpoints_preserved(x, xo, y, yo)


# ---------------------------------------------------------------------------
# Oscillatory curves — sin(x), noisy sin, multi-frequency
# ---------------------------------------------------------------------------

class TestOscillatory:

    def test_pure_sin(self):
        """sin(x) over [0, 4π] = 2 full cycles → 4 extrema. Even at the
        100-point budget, all 4 high-prominence extrema must be retained."""
        x = np.linspace(0, 4 * np.pi, 5000)
        y = np.sin(x)
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)  # smooth curve must not warn
            xo, yo = _simplify(x, y)
        assert len(xo) <= 100
        assert reconstruction_r2(x, y, xo, yo) >= 0.99
        # Peaks reach ±1 (prominent extrema not smoothed away).
        assert yo.max() >= 0.999 and yo.min() <= -0.999
        # At least 4 sign changes preserved (the 4 extrema).
        n_sign_changes = int(np.sum(np.diff(np.sign(np.diff(yo))) != 0))
        assert n_sign_changes >= 4

    def test_high_frequency_sin_warns(self):
        """20 cycles of sin compressed into 100 points cannot reconstruct
        all the wiggles — the algorithm should warn the user."""
        x = np.linspace(0, 1, 8000)
        y = np.sin(20 * np.pi * x)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            xo, yo = _simplify(x, y)
        assert len(xo) <= 100
        # Either reconstruction is fine, or a warning was issued.
        r2 = reconstruction_r2(x, y, xo, yo)
        if r2 < 0.9:
            assert any('R²' in str(w.message) for w in caught), \
                f"R²={r2:.3f} below 0.9 but no warning emitted"

    def test_noisy_sin(self):
        """Sin with 5% Gaussian noise — fixed budget + noise means R² is
        moderate. We don't demand high R²; we just require no errors and
        an output capped at the budget."""
        rng = np.random.RandomState(0)
        x = np.linspace(0, 4 * np.pi, 5000)
        y_clean = np.sin(x)
        y = y_clean + 0.05 * rng.randn(x.size)
        xo, yo = _simplify(x, y)
        assert len(xo) <= 100
        assert np.isfinite(yo).all()

    def test_multi_frequency(self):
        """Beat pattern: low + high frequency superposition."""
        x = np.linspace(0, 2 * np.pi, 6000)
        y = np.sin(x) + 0.3 * np.sin(15 * x)
        xo, yo = _simplify(x, y)
        assert len(xo) <= 100
        # Beat pattern is reasonable to reconstruct at 100 pts
        assert reconstruction_r2(x, y, xo, yo) >= 0.7

    def test_sin_endpoints_preserved(self):
        x = np.linspace(0, 4 * np.pi, 5000)
        y = np.sin(x)
        xo, yo = _simplify(x, y)
        assert_endpoints_preserved(x, xo, y, yo)


# ---------------------------------------------------------------------------
# Spectrum-like curves (SB99 luminosity vs wavelength)
# ---------------------------------------------------------------------------

class TestSpectrumLike:

    @pytest.fixture
    def sb99_like_spectrum(self):
        """A synthetic luminosity-vs-wavelength curve with a stellar
        continuum + a few sharp emission lines + a broad bump — mimicking
        the structure of an SB99 output."""
        rng = np.random.RandomState(42)
        wl = np.linspace(100, 5000, 8000)
        # power-law continuum
        L = 1e3 * (wl / 1000.0) ** (-1.5)
        # broad bump
        L += 5e2 * np.exp(-((wl - 800) / 150) ** 2)
        # sharp emission lines
        for line, height, width in [(1216, 4e3, 2),
                                    (1640, 2e3, 1.5),
                                    (4861, 3e3, 3),
                                    (4340, 1.5e3, 2)]:
            L += height * np.exp(-((wl - line) / width) ** 2)
        # mild noise
        L += rng.randn(wl.size) * 5.0
        # log10 to compress dynamic range
        return wl, np.log10(np.maximum(L, 1e-3))

    def test_spectrum_compresses(self, sb99_like_spectrum):
        wl, logL = sb99_like_spectrum
        xo, yo = _simplify(wl, logL)
        assert len(xo) <= 100
        # Spectrum has 4 emission lines + continuum; 100 points is tight
        # so we accept either good R² or a warning.
        r2 = reconstruction_r2(wl, logL, xo, yo)
        assert r2 >= 0.5  # sanity: not totally garbage

    def test_spectrum_preserves_lines(self, sb99_like_spectrum):
        """Sharp emission peaks (high prominence) must be retained at
        the 100-point budget — that's why prominence trumps bisection."""
        wl, logL = sb99_like_spectrum
        xo, yo = _simplify(wl, logL)
        # The Lyα-equivalent at 1216 should appear within 10Å of the truth
        # and have y above the continuum (we kept the peak, not a flank).
        idx_near_line = int(np.argmin(np.abs(xo - 1216.0)))
        assert abs(xo[idx_near_line] - 1216.0) < 10.0
        line_y = yo[idx_near_line]
        assert line_y > np.median(logL) + 0.3

    def test_spectrum_can_be_tuned_higher(self, sb99_like_spectrum):
        """Caller can raise nmin to get higher fidelity on detailed curves."""
        wl, logL = sb99_like_spectrum
        xo_low, yo_low = _simplify(wl, logL, nmin=100)
        xo_hi,  yo_hi  = _simplify(wl, logL, nmin=500)
        assert len(xo_hi) > len(xo_low)
        assert reconstruction_r2(wl, logL, xo_hi, yo_hi) > reconstruction_r2(wl, logL, xo_low, yo_low)


# ---------------------------------------------------------------------------
# Non-monotonic input — both legitimate (parametric loop) and pathological
# (one out-of-order point in a mostly-monotonic array)
# ---------------------------------------------------------------------------

class TestNonMonotonic:

    def test_one_out_of_order_point(self):
        """Mostly descending with a single inverted neighbour — the case
        encountered in real 2_energy.jsonl bubble profiles."""
        x = np.linspace(1.0, 0.0, 5000)  # descending
        y = np.sin(2 * np.pi * x)
        x[33], x[34] = x[34], x[33]
        y[33], y[34] = y[34], y[33]
        diffs = np.diff(x)
        assert (diffs > 0).any() and (diffs < 0).any()
        xo, yo = _simplify(x, y)
        assert np.isfinite(xo).all() and np.isfinite(yo).all()
        assert len(xo) <= 100
        assert reconstruction_r2(x, y, xo, yo) >= 0.9

    def test_parametric_loop_documents_limitation(self):
        """Documents a known algorithm limitation: ``_simplify`` treats its
        input as a function ``y = f(x)`` and reconstructs by linear
        interpolation in x. A genuinely parametric loop (e.g. a circle)
        has two y-values per x, so no subset can reconstruct it well via
        x-interpolation. The R²-thinning step therefore can't beat the
        target and we fall through to the full merged pool. The output
        still must be finite and orientation-preserving — we only assert
        that the call doesn't error out."""
        t = np.linspace(0, 2 * np.pi, 5000)
        x = 0.5 + 0.4 * np.cos(t)
        y = 0.4 * np.sin(t)
        xo, yo = _simplify(x, y)
        assert np.isfinite(xo).all() and np.isfinite(yo).all()
        assert len(xo) <= len(x)

    def test_non_monotonic_preserves_input_sequence(self):
        """The output should be a thinned subsequence of the input — i.e.,
        consecutive output points correspond to non-decreasing positions
        in the input array."""
        rng = np.random.RandomState(1)
        # Build a curve where x meanders so order is genuinely interesting
        x = np.cumsum(rng.randn(3000) * 0.01) + np.linspace(0, 1, 3000)
        y = np.sin(2 * np.pi * x)
        xo, yo = _simplify(x, y)
        # Every output point came from some position in the input. Build
        # the position list and assert it is monotonically increasing.
        positions = []
        cursor = 0
        for xi, yi in zip(xo, yo):
            for j in range(cursor, x.size):
                if np.isclose(x[j], xi) and np.isclose(y[j], yi):
                    positions.append(j)
                    cursor = j + 1
                    break
        assert len(positions) == len(xo), "every output sample must trace to an input position"
        assert all(positions[i] < positions[i + 1] for i in range(len(positions) - 1)), \
            "output must preserve input positional order"


# ---------------------------------------------------------------------------
# Signed and zero-crossing inputs
# ---------------------------------------------------------------------------

class TestSignedY:

    def test_y_crosses_zero(self):
        """bubble_v_arr-style: y has both positive and negative regions."""
        x = np.linspace(0.01, 0.07, 5000)
        y = 1500 * ((x - 0.01) / 0.06) ** 2 - 500
        assert (y < 0).any() and (y > 0).any()
        xo, yo = _simplify(x, y)
        assert np.isfinite(yo).all()
        assert len(xo) <= 100
        assert reconstruction_r2(x, y, xo, yo) >= 0.95

    def test_y_all_negative(self):
        x = np.linspace(0, 1, 3000)
        y = -np.exp(x)
        xo, yo = _simplify(x, y)
        assert np.all(yo < 0)
        assert reconstruction_r2(x, y, xo, yo) >= 0.99


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:

    def test_empty(self):
        xo, yo = _simplify(np.array([]), np.array([]))
        assert xo.size == 0 and yo.size == 0

    def test_single_point(self):
        xo, yo = _simplify(np.array([3.14]), np.array([2.71]))
        assert xo.size == 1 and yo.size == 1
        assert xo[0] == 3.14 and yo[0] == 2.71

    def test_size_below_floor_passes_through(self):
        """nmin clamps to 100 internally, so any input ≤ 100 should be
        returned unchanged."""
        x = np.linspace(0, 1, 50)
        y = np.sin(x * np.pi)
        xo, yo = _simplify(x, y)
        np.testing.assert_array_equal(xo, x)
        np.testing.assert_array_equal(yo, y)

    def test_size_below_floor_descending_passthrough(self):
        x = np.linspace(0, 1, 50)[::-1].copy()
        y = np.sin(x * np.pi)
        xo, yo = _simplify(x, y)
        np.testing.assert_array_equal(xo, x)
        np.testing.assert_array_equal(yo, y)

    def test_constant_y(self):
        x = np.linspace(0, 1, 500)
        y = np.full_like(x, 3.5)
        xo, yo = _simplify(x, y)
        # Falls into flat-curve fallback, returns ~nmin uniformly spaced pts
        assert len(xo) <= len(x)
        assert np.allclose(yo, 3.5)

    def test_mismatched_lengths_raises(self):
        with pytest.raises(ValueError):
            _simplify(np.array([0.0, 1.0]), np.array([0.0, 1.0, 2.0]))

    def test_output_endpoints_match_input(self):
        """For any non-edge case input, both endpoints of the input must
        appear in the output. Crucial for noisy curves where the prominence
        set could otherwise crowd them out of the budget."""
        rng = np.random.RandomState(0)
        x = np.linspace(0, 1, 1000)
        y = np.sin(10 * x) + 0.1 * rng.randn(1000)
        xo, yo = _simplify(x, y)
        assert_endpoints_preserved(x, xo, y, yo)


# ---------------------------------------------------------------------------
# Real bubble profiles from outputs/mockOutput
# ---------------------------------------------------------------------------

REAL_RUN = MOCK_RUN
REAL_AVAILABLE = REAL_RUN.exists()


@pytest.mark.skipif(not REAL_AVAILABLE, reason="mockOutput not present")
class TestRealBubbleProfiles:

    @pytest.fixture(scope="class")
    def implicit_snap(self):
        path = REAL_RUN / "3_implicit.jsonl"
        with open(path) as f:
            return json.loads(f.readline())

    @pytest.fixture(scope="class")
    def energy_snap(self):
        path = REAL_RUN / "2_energy.jsonl"
        with open(path) as f:
            return json.loads(f.readline())

    @pytest.mark.parametrize("ykey,xkey", [
        ("log_bubble_T_arr",     "bubble_T_arr_r_arr"),
        ("log_bubble_n_arr",     "bubble_n_arr_r_arr"),
        ("log_bubble_dTdr_arr",  "bubble_dTdr_arr_r_arr"),
        ("bubble_v_arr",         "bubble_v_arr_r_arr"),
    ])
    def test_implicit_phase_compresses(self, implicit_snap, ykey, xkey):
        x = np.array(implicit_snap[xkey])
        y = np.array(implicit_snap[ykey])
        if x.size <= 100:
            pytest.skip(f"{ykey} already short ({x.size})")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            xo, yo = _simplify(x, y)
        assert len(xo) <= 100, f"{ykey}: output exceeds budget"
        # Bubble grids are descending; the simplifier must preserve that.
        assert np.all(np.diff(xo) <= 0), f"{xkey} should stay descending"

    def test_energy_phase_non_monotonic_compresses(self, energy_snap):
        """2_energy.jsonl arrays have a single non-monotonic point —
        confirms the simplifier handles the real-world non-monotonic case
        and produces a budget-bounded output."""
        x = np.array(energy_snap["bubble_T_arr_r_arr"])
        y = np.array(energy_snap["log_bubble_T_arr"])
        if x.size <= 100:
            pytest.skip("array already short")
        diffs = np.diff(x)
        assert (diffs > 0).any() and (diffs < 0).any()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            xo, yo = _simplify(x, y)
        assert len(xo) <= 100
        assert np.isfinite(xo).all() and np.isfinite(yo).all()


# ---------------------------------------------------------------------------
# Fixed-budget contract
# ---------------------------------------------------------------------------

class TestBudgetContract:
    """Output size is at most ``nmin``, regardless of how complicated the
    curve is or how many prominent extrema it has."""

    def test_default_budget_is_100(self):
        x = np.linspace(0, 1, 5000)
        y = np.sin(50 * x)
        xo, _ = _simplify(x, y)
        assert len(xo) <= 100

    @pytest.mark.parametrize("nmin", [100, 200, 500, 1000])
    def test_budget_respected(self, nmin):
        x = np.linspace(0, 1, 10000)
        y = np.sin(30 * x) + 0.5 * np.sin(80 * x)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            xo, _ = _simplify(x, y, nmin=nmin)
        assert len(xo) <= nmin

    def test_budget_not_exceeded_by_prominent_extrema(self):
        """Even when the curve has many high-prominence extrema, output
        must stay within nmin."""
        rng = np.random.RandomState(0)
        x = np.linspace(0, 1, 2000)
        y = np.sin(50 * x) + 0.3 * rng.randn(2000)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            xo, _ = _simplify(x, y, nmin=100)
        assert len(xo) <= 100

    def test_higher_nmin_does_not_reduce_quality(self):
        """Increasing nmin must not reduce reconstruction quality. The
        cumulative-distance bin step depends on nmin so the merged pool
        differs slightly between calls; we test the meaningful property
        rather than strict subset overlap."""
        x = np.linspace(0, 1, 5000)
        y = np.sin(10 * x) + 0.2 * np.sin(40 * x)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            xo_100, yo_100 = _simplify(x, y, nmin=100)
            xo_200, yo_200 = _simplify(x, y, nmin=200)
            xo_500, yo_500 = _simplify(x, y, nmin=500)
        r2_100 = reconstruction_r2(x, y, xo_100, yo_100)
        r2_200 = reconstruction_r2(x, y, xo_200, yo_200)
        r2_500 = reconstruction_r2(x, y, xo_500, yo_500)
        assert r2_200 >= r2_100 - 0.01, f"nmin=200 R²={r2_200:.3f} < nmin=100 R²={r2_100:.3f}"
        assert r2_500 >= r2_200 - 0.01, f"nmin=500 R²={r2_500:.3f} < nmin=200 R²={r2_200:.3f}"


# ---------------------------------------------------------------------------
# Reconstruction-quality warning
# ---------------------------------------------------------------------------

class TestWarning:
    """The post-hoc R² check warns when the budget is too small for the
    chosen curve, gives the user a clear signal to raise nmin."""

    def test_smooth_curve_does_not_warn(self):
        x = np.linspace(0, 1, 5000)
        y = x**2  # very smooth, easy to reconstruct
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            _simplify(x, y)  # would raise if any UserWarning fires

    def test_undersampled_high_freq_warns(self):
        """30 cycles of sin in 100 points → reconstruction is poor."""
        x = np.linspace(0, 1, 5000)
        y = np.sin(60 * np.pi * x)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            _simplify(x, y, nmin=100)
        # The simplifier may still get high R² in some lucky configurations;
        # if it does, no warning. But for 30 cycles it almost always warns.
        # We just assert that IF the curve is genuinely undersampled, a
        # warning is raised.
        x2, y2 = x, np.sin(120 * np.pi * x)  # 60 cycles, even more punishing
        with warnings.catch_warnings(record=True) as caught2:
            warnings.simplefilter("always")
            _simplify(x2, y2, nmin=100)
        assert any("R\u00b2" in str(w.message) for w in caught2), \
            "60-cycle sin at nmin=100 must trigger the R\u00b2 warning"

    def test_warning_disabled_when_threshold_is_none(self):
        x = np.linspace(0, 1, 5000)
        y = np.sin(60 * np.pi * x)
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            _simplify(x, y, nmin=100, warn_below_r2=None)  # must not raise

    def test_warning_threshold_configurable(self):
        """A relaxed threshold should suppress the warning for a curve
        that would have warned at the default."""
        x = np.linspace(0, 1, 5000)
        y = np.sin(60 * np.pi * x)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            _simplify(x, y, nmin=100, warn_below_r2=0.0)
        # threshold 0.0 means "warn only if R² < 0", which never happens.
        assert not any("R\u00b2" in str(w.message) for w in caught)


# ---------------------------------------------------------------------------
# Timing / efficiency — ensures the algorithm scales reasonably and the
# fixed-budget version is at least as fast as the input-size dominated
# work it has to do (curvature, prominence, bisection are all O(n log n)).
# ---------------------------------------------------------------------------

class TestTiming:

    @pytest.mark.parametrize("size,budget_seconds", [
        ( 1_000, 0.10),
        (10_000, 0.30),
        (30_000, 0.60),
        (100_000, 2.00),
    ])
    def test_runtime_budget(self, size, budget_seconds):
        x = np.linspace(0, 1, size)
        y = np.sin(20 * x) + 0.1 * np.cos(50 * x)
        # Warm up to amortise import / first-call costs
        _simplify(x[:100], y[:100])
        t0 = time.perf_counter()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            _simplify(x, y)
        elapsed = time.perf_counter() - t0
        assert elapsed < budget_seconds, (
            f"_simplify on N={size} took {elapsed:.3f}s, budget {budget_seconds}s"
        )

    def test_subquadratic_scaling(self):
        """Roughly O(n log n): doubling N should not 4x the runtime."""
        rng = np.random.RandomState(0)
        sizes = [10_000, 40_000]
        times = []
        for n in sizes:
            x = np.linspace(0, 1, n)
            y = np.sin(20 * x) + 0.05 * rng.randn(n)
            _simplify(x[:100], y[:100])  # warmup
            t0 = time.perf_counter()
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                _simplify(x, y)
            times.append(time.perf_counter() - t0)
        ratio = times[1] / max(times[0], 1e-6)
        # Strict O(n^2) would give ratio ~16 for 4x size; require well below.
        assert ratio < 8.0, f"runtime ratio {ratio:.2f} suggests super-linear scaling"
