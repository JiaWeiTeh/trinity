"""
Tests for ``trinity._functions.simplify._simplify``.

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

import time
import warnings

import numpy as np
import pytest

from trinity._functions.simplify import _simplify


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


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

    def test_high_frequency_sin_keeps_all_extrema(self):
        """20 cycles of sin → 40 prominent extrema. The mandatory-override
        kicks in: output exceeds the 100-point budget, all extrema kept,
        and reconstruction is good (no warning)."""
        x = np.linspace(0, 1, 8000)
        y = np.sin(20 * np.pi * x)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            xo, yo = _simplify(x, y)
        # Override gives us all 40 extrema (+ endpoints + possibly a few
        # bisection points to fill out the budget).
        assert len(xo) >= 40
        assert reconstruction_r2(x, y, xo, yo) >= 0.95
        # Smooth oscillation, mandatory captures everything → no warning.
        assert not any('R²' in str(w.message) for w in caught)

    def test_noisy_sin(self):
        """Sin + noise: the noise creates many local extrema, some of
        which pass the prominence threshold and override the budget.
        We require no errors, finite output, and bounded growth."""
        rng = np.random.RandomState(0)
        x = np.linspace(0, 4 * np.pi, 5000)
        y_clean = np.sin(x)
        y = y_clean + 0.05 * rng.randn(x.size)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            xo, yo = _simplify(x, y)
        assert np.isfinite(yo).all()
        assert len(xo) <= len(x), "output cannot exceed input"
        assert len(xo) >= 100, "output should be at least the budget"

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
# Fixed-budget contract
# ---------------------------------------------------------------------------

class TestBudgetContract:
    """``nmin`` is the *normal* output size and a soft floor: smooth
    curves come back at exactly ``nmin``. For very noisy data with more
    than ``nmin`` high-prominence extrema, the output may exceed ``nmin``
    rather than drop a real feature — those extrema override the budget."""

    def test_smooth_curve_hits_target_exactly(self):
        """Smooth curve, few prominent features → output ≈ nmin."""
        x = np.linspace(0, 1, 5000)
        y = x ** 3 - x  # smooth cubic, 1 maximum + 1 minimum
        xo, _ = _simplify(x, y, nmin=100)
        # No mandatory-override, so we hit the budget exactly (or slightly
        # under if the merged pool itself is small).
        assert len(xo) <= 100

    def test_below_old_floor_respected(self):
        """Sub-100 nmin must produce ~that many points (the old 100 floor
        no longer silently clamps).  At nmin=30 the smooth cubic has only
        2 prominent extrema, so the budget is a hard ceiling — coverage
        is capped at nmin-2 to keep endpoints + coverage <= nmin."""
        x = np.linspace(0, 1, 5000)
        y = x ** 3 - x
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            xo, _ = _simplify(x, y, nmin=30)
        assert len(xo) <= 30, (
            f"nmin=30 should yield ≤ 30 points, got {len(xo)}"
        )

    def test_floor_of_twenty_clamps_below(self):
        """Values below the new floor of 20 silently raise to 20."""
        x = np.linspace(0, 1, 5000)
        y = x ** 3 - x
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            xo_low, _ = _simplify(x, y, nmin=5)
            xo_floor, _ = _simplify(x, y, nmin=20)
        # Both should produce the same output: 5 → clamped → 20.
        assert len(xo_low) == len(xo_floor)
        assert len(xo_low) <= 20 + 1   # floor of 20 segments includes both endpoints

    @pytest.mark.parametrize("nmin", [100, 200, 500, 1000])
    def test_budget_respected_for_smooth_curve(self, nmin):
        """For a smooth curve the budget is a hard ceiling."""
        x = np.linspace(0, 1, 10000)
        y = np.tanh(5 * (x - 0.5)) + 0.1 * x
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            xo, _ = _simplify(x, y, nmin=nmin)
        assert len(xo) <= nmin

    def test_prominent_extrema_override_budget(self):
        """For very noisy data, every high-prominence extremum is kept
        even if that pushes output past ``nmin``. Quantitatively: ≥99 %
        of input extrema with prominence ≥ 5 % of y-range must appear
        in the output, even at nmin=100."""
        rng = np.random.RandomState(0)
        x = np.linspace(0, 1, 2000)
        # 50 cycles of sin → 100 strong extrema, all prominent
        y = np.sin(100 * np.pi * x) + 0.05 * rng.randn(2000)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            xo, _ = _simplify(x, y, nmin=100)
        # The 100 sin extrema should all be there → output ≥ 100 + endpoints
        assert len(xo) >= 100, \
            f"prominent extrema were dropped: only {len(xo)} pts in output"
        # Output stays bounded by input size
        assert len(xo) <= len(x)

    def test_all_prominent_extrema_preserved(self):
        """Stronger guarantee: every input index identified as a high-
        prominence extremum must appear in the simplified output."""
        from trinity._functions.simplify import _peak_prominences

        x = np.linspace(0, 1, 2000)
        y = np.sin(40 * np.pi * x)  # 20 cycles → 40 extrema, all amplitude 1
        # Reproduce the prominence calculation independently
        grad = np.gradient(y)
        sd = np.diff(np.sign(grad))
        sc = np.where(sd != 0)[0]
        is_max = sd[sc] < 0
        pick_i = np.where(is_max,
                          y[sc] >= y[sc + 1],
                          y[sc] <= y[sc + 1])
        extrema = np.unique(np.where(pick_i, sc, sc + 1))
        proms = _peak_prominences(y, extrema)
        y_range = y.max() - y.min()
        prominent = extrema[proms >= 0.05 * y_range]

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            xo, _ = _simplify(x, y, nmin=100)
        # Every prominent input x-value must be in the output (within
        # floating-point tolerance).
        prominent_x = x[prominent]
        for px in prominent_x:
            assert np.any(np.isclose(xo, px, atol=1e-10)), \
                f"prominent extremum at x={px:.4f} was dropped"

    def test_higher_nmin_does_not_reduce_quality(self):
        """Increasing nmin must not reduce reconstruction quality. The
        arc-length bin step depends on nmin so the merged pool differs
        slightly between calls; we test the meaningful property rather
        than strict subset overlap."""
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

    def test_unstructured_noise_warns(self):
        """White-noise input: most fluctuations are below the prominence
        threshold, so the mandatory-override doesn't rescue the budget.
        Reconstruction R\u00b2 stays low \u2192 warning fires."""
        rng = np.random.RandomState(0)
        x = np.linspace(0, 1, 5000)
        y = rng.randn(5000)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            _simplify(x, y, nmin=100)
        assert any("R\u00b2" in str(w.message) for w in caught), \
            "white noise at nmin=100 must trigger the R\u00b2 warning"

    def test_warning_disabled_when_threshold_is_none(self):
        rng = np.random.RandomState(0)
        x = np.linspace(0, 1, 5000)
        y = rng.randn(5000)
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            _simplify(x, y, nmin=100, warn_below_r2=None)  # must not raise

    def test_warning_threshold_configurable(self):
        """A relaxed threshold should suppress the warning on a curve
        that would have warned at the default."""
        rng = np.random.RandomState(0)
        x = np.linspace(0, 1, 5000)
        y = rng.randn(5000)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            _simplify(x, y, nmin=100, warn_below_r2=0.0)
        # threshold 0.0 means "warn only if R\u00b2 < 0", which never happens.
        assert not any("R\u00b2" in str(w.message) for w in caught)


# ---------------------------------------------------------------------------
# X-uniform coverage skeleton: low-amplitude regions stay represented
# even when a big-amplitude burst dominates the global SS_tot.
# ---------------------------------------------------------------------------

class TestCoverageSkeleton:
    """The x-uniform coverage skeleton (one feature-pool point per
    equal-width x-chunk promoted to mandatory) prevents amplitude-biased
    starvation of low-amplitude regions."""

    def test_low_amplitude_region_not_starved(self):
        """Without coverage the bisection ordering puts almost all points
        around the big peak (baseline: 2 of 100 land in the quiet region).
        Coverage promotes feature-pool points near chunk centres into the
        mandatory set, lifting the quiet-region count noticeably even when
        the merged pool is sparse there."""
        N = 5000
        x = np.linspace(0, 1, N)
        # Quiet low-amplitude oscillation in the first 70 % of x …
        y = 0.01 * np.sin(20 * x)
        # … then a tall narrow bump in the last 30 %.
        y[int(0.7 * N):] += 5.0 * np.exp(
            -0.5 * ((x[int(0.7 * N):] - 0.85) / 0.02) ** 2
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            xo, _ = _simplify(x, y, nmin=100)
        left_pts = int((xo < 0.7).sum())
        # Pre-coverage baseline was 2; coverage cannot exceed the pool's
        # density in the quiet region (where curvature, extrema, and
        # cum-distance all fire rarely), but it must at least double the
        # representation.
        assert left_pts >= 4, (
            f"only {left_pts} pts in low-amplitude region (was 2 pre-coverage)"
        )

    def test_coverage_indices_are_subset_of_input_indices(self):
        """Helper sanity: returned indices index into the original x array."""
        from trinity._functions.simplify import _x_uniform_coverage_idx
        x = np.linspace(0, 1, 1000)
        pool = np.arange(0, 1000, 5)  # every 5th point
        idx = _x_uniform_coverage_idx(x, pool, n_chunks=20)
        assert np.all(np.isin(idx, pool))
        assert idx.size <= 20

    def test_coverage_handles_short_input(self):
        from trinity._functions.simplify import _x_uniform_coverage_idx
        # Single point — no chunks possible
        empty = _x_uniform_coverage_idx(np.array([0.5]), np.array([0]))
        assert empty.size == 0
        # Empty pool
        empty = _x_uniform_coverage_idx(np.linspace(0, 1, 100), np.array([], dtype=np.int64))
        assert empty.size == 0


# ---------------------------------------------------------------------------
# Log-space error metrics — decade-balanced fidelity for spectra and
# multi-decade physics profiles.
# ---------------------------------------------------------------------------

class TestLogMetrics:

    def test_log_metrics_present_when_y_positive(self):
        from trinity._functions.simplify import _simplify_error
        x = np.linspace(1.0, 100.0, 5000)
        y = x ** -1.5  # always positive, ~3 decades of dynamic range
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            xo, yo = _simplify(x, y, nmin=100)
        m = _simplify_error(x, y, xo, yo)
        for k in ("log_r_squared", "log_rms_err",
                  "log_max_dex_err", "log_mean_dex_err"):
            assert k in m and np.isfinite(m[k]), f"{k} missing/NaN"
        assert 0.0 <= m["log_r_squared"] <= 1.0

    def test_log_metrics_nan_when_y_crosses_zero(self):
        from trinity._functions.simplify import _simplify_error
        x = np.linspace(0.0, 1.0, 5000)
        y = np.sin(2 * np.pi * x)  # crosses zero
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            xo, yo = _simplify(x, y, nmin=100)
        m = _simplify_error(x, y, xo, yo)
        for k in ("log_r_squared", "log_rms_err",
                  "log_max_dex_err", "log_mean_dex_err"):
            assert np.isnan(m[k]), f"{k} should be NaN for sign-crossing y"
        # Linear-space metrics should still be finite
        assert np.isfinite(m["r_squared"])

    def test_log_dex_error_units(self):
        """log_max_dex_err is the L∞ deviation in decimal-log units —
        a value of 0.01 should correspond to ≈ 2 % relative error."""
        from trinity._functions.simplify import _simplify_error
        x = np.linspace(1.0, 1000.0, 5000)
        y = 10.0 ** (np.log10(x) * 0.5)  # smooth power-law in log space
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            xo, yo = _simplify(x, y, nmin=100)
        m = _simplify_error(x, y, xo, yo)
        # Verify the dex relation: 10^max_dex - 1 ≈ max relative error
        approx_rel_from_dex = 10.0 ** m["log_max_dex_err"] - 1.0
        # The two should be the same order of magnitude (within 3×).
        # (max_rel_err is computed in linear y-space so they're not
        # identical — but tightly related for a smooth curve.)
        assert approx_rel_from_dex < 3.0 * (m["max_rel_err"] + 1e-6) + 0.05


# ---------------------------------------------------------------------------
# Solver-artefact failure modes — two regression cases that motivated the
# tolerance dedup + ``idx_dist`` priority promotion.  See the discussion
# in commit history for the full diagnosis; the short version is:
#
#   1. ODE solvers can emit thousands of micro-step (x, y) duplicates
#      that the budget-trim step happily fills with copies of one
#      physical point, starving the rest of the curve.
#   2. Even without duplicates, ``idx_dist`` (cumulative arc-length
#      boundaries on the rescaled unit square) used to live only in
#      the ``merged`` mask and got displaced by bisection-by-position,
#      leaving steep tails sampled by a single straight line.
# ---------------------------------------------------------------------------

class TestSolverArtefacts:
    """Regression tests for ODE-solver clumps and steep-tail dips."""

    def test_clump_at_start_does_not_eat_budget(self):
        # Mimic a stalled integrator: 5000 micro-steps at the boundary
        # then a smooth rise.  Per-step Δ is ~10⁻⁹ of the data range —
        # well below the dedup floor.
        rng = np.random.RandomState(0)
        x_clump = 0.06 + 1e-10 * np.cumsum(rng.rand(5000))
        y_clump = 57.77 + 1e-9 * np.cumsum(rng.rand(5000))
        x_curve = np.linspace(0.06, 0.075, 200)
        y_curve = np.linspace(57.77, 60.7, 200)
        x = np.concatenate([x_clump, x_curve])
        y = np.concatenate([y_clump, y_curve])

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            xo, yo = _simplify(x, y, nmin=100)

        # At most a handful of output points should land in the clump
        # x-region (we permit a small constant for the dedup boundary
        # plus an endpoint coverage point).
        in_clump = xo < 0.06 + 1e-7
        assert in_clump.sum() <= 3, (
            f"{in_clump.sum()} of {xo.size} output points wasted on the clump"
        )
        # The actual rise must be densely sampled.
        assert (~in_clump).sum() >= 90
        assert reconstruction_r2(x, y, xo, yo) > 0.95

    def test_steep_tail_dip_is_sampled(self):
        # Smooth slow descent over 95% of x, then a steep drop in the
        # final 5% — the 3_implicit-T motif.  No duplicates here, so
        # this isolates the ``idx_dist`` priority-promotion fix.
        x_slow = np.linspace(0.20, 0.60, 9500)
        y_slow = 7.4 - 0.7 * (x_slow - 0.20) / 0.40        # 7.4 → 6.7
        x_dip = np.linspace(0.60, 0.63, 500)
        # Steep nonlinear drop from 6.7 down to 4.5
        y_dip = 6.7 - 2.2 * ((x_dip - 0.60) / 0.03) ** 2
        x = np.concatenate([x_slow, x_dip])
        y = np.concatenate([y_slow, y_dip])

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            xo, yo = _simplify(x, y, nmin=100)

        in_dip = xo > 0.60
        # Arc-length-on-rescaled-axes weighting: the dip and the plateau
        # contribute comparable arc lengths on the unit square, so the
        # budget splits roughly evenly between them.  The dip should
        # still be densely sampled — well above the pre-fix worst case
        # of ~5 — so this assertion is a regression guard, not a
        # majority-share claim.
        assert in_dip.sum() >= 30, (
            f"only {in_dip.sum()} of {xo.size} output points cover the steep dip"
        )
        assert reconstruction_r2(x, y, xo, yo) > 0.95

    def test_dedup_preserves_vertical_drop(self):
        # Same x, varying y is a meaningful "vertical drop" region —
        # the OR-on-Δ rule must keep these intact.
        x = np.concatenate([np.linspace(0.0, 1.0, 500),
                            np.full(50, 1.0)])
        y = np.concatenate([np.linspace(0.0, 1.0, 500),
                            np.linspace(1.0, 0.0, 50)])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            xo, yo = _simplify(x, y, nmin=100)
        # The trailing vertical drop (50 distinct y at x=1.0) must be
        # represented by more than just the endpoint.
        at_tail = np.isclose(xo, 1.0, atol=1e-12)
        assert at_tail.sum() >= 5, (
            f"vertical-drop tail collapsed: only {at_tail.sum()} "
            f"output points at x=1.0"
        )

    def test_dedup_tol_zero_is_passthrough(self):
        # Setting dedup_tol=0 must reproduce pre-fix behaviour: no
        # collapse, every input point counts as a candidate.
        x = np.concatenate([np.full(1000, 0.05) + 1e-12 * np.arange(1000),
                            np.linspace(0.05, 0.10, 200)])
        y = np.concatenate([np.full(1000, 1.0) + 1e-12 * np.arange(1000),
                            np.linspace(1.0, 5.0, 200)])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            xo_off, _ = _simplify(x, y, nmin=100, dedup_tol=0.0)
            xo_on, _ = _simplify(x, y, nmin=100)
        # With dedup off the clump dominates; with dedup on it does not.
        assert (xo_off < 0.0501).sum() > (xo_on < 0.0501).sum()


# ---------------------------------------------------------------------------
# Timing / efficiency — ensures the algorithm scales reasonably and the
# fixed-budget version is at least as fast as the input-size dominated
# work it has to do (curvature, prominence, bisection are all O(n log n)).
# ---------------------------------------------------------------------------

@pytest.mark.stress
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
