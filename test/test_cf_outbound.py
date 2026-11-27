"""cover_law sigma under hole_forces on: the outbound column (docs/dev/leaky-reservoir/CF_LAW_PLAN.md §11, ruled
2026-11-27). An open direction sweeps no gas, so C_f at outer radius r is cf_law at the smallest column the shell had on
its way out up to r: it never rises while the shell reaches new radii, a shell falling back closes holes again inside
the radius where they opened, one moving out again over the same radii reopens them on the same curve, and C_f cannot
ratchet. Under hole_forces off the law reads the column now, as before."""
import math
from types import SimpleNamespace

import numpy as np
import pytest

import trinity.bubble_structure.get_bubbleParams as G

H = 2.0


def _ns(**kw):
    """Hand-built params: {key: object with .value}, as the helpers read them."""
    return {k: SimpleNamespace(value=v) for k, v in kw.items()}


def _vented(forces="on", **kw):
    """A vented leaky reservoir under cover_law sigma in the momentum phase (no evaporated mass in the column) at its
    vent row, r_s 10 pc and swept mass 1e4 Msun; the deficit's and the table's keys at their defaults."""
    base = dict(hole_forces=forces, vent_model="reservoir", transition_channel="blowout", coverFraction=1.0,
                cover_law="sigma", cf_half_radius=H, _cf_Sigma_vent=0.0, rs_dyn=10.0, shell_mass=1e4,
                bubble_mass=0.0, current_phase="momentum", _hole_D=0.0, _hole_Ms_seg=0.0, _hole_Cf_seg=1.0,
                _cf_tab_r=None, _cf_tab_S=None)
    base.update(kw)
    return _ns(**base)


def _vent(p):
    """The vent row as swept builds it: _forces_to_params ends with hole_deficit, then the hook stores the column."""
    G.hole_deficit(p, 1e4)
    G.store_vent_column(p)
    return p["_cf_Sigma_vent"].value


def _row(p, r, Ms):
    """One later row: rs_dyn and shell_mass written, then hole_deficit; returns the row's C_f."""
    p["rs_dyn"].value, p["shell_mass"].value = r, Ms
    G.hole_deficit(p, Ms)
    return G.cover_fraction(p)


def _uniform(chi):
    """Swept mass of a shell that vented at the edge of a uniform cloud (r_v 10 pc, M_v 1e4 Msun) and then sweeps a
    uniform ambient medium chi times less dense: M(r) = M_v [1 + ((r/r_v)^3 - 1)/chi] (Paper II, fig_shell_column)."""
    return lambda r: 1e4 * (1.0 + ((r / 10.0) ** 3 - 1.0) / chi)


def _col(m, r):
    return m / (4.0 * math.pi * r**2)


def test_the_hook_starts_the_table_at_the_vent():
    p = _vented()
    Sv = _vent(p)
    assert Sv == _col(1e4, 10.0) and G.cover_fraction(p) == 1.0
    assert (p["_cf_tab_r"].value, p["_cf_tab_S"].value) == ([10.0], [Sv])
    p["rs_dyn"].value, p["shell_mass"].value = 20.0, 3e4
    G.store_vent_column(p)                                               # once only: the table is not restarted
    assert (p["_cf_tab_r"].value, p["_cf_tab_S"].value) == ([10.0], [Sv])
    q = _vented(cover_law="constant")
    G.store_vent_column(q)
    assert q["_cf_tab_r"].value is None                                  # no law, no table


def test_cf_never_rises_while_the_shell_reaches_new_radii():
    p, M = _vented(), _uniform(10.0)
    Sv = _vent(p)
    R = [float(r) for r in np.linspace(10.0, 60.0, 201)[1:]]
    cf = [_row(p, r, M(r)) for r in R]
    assert all(b <= a for a, b in zip(cf, cf[1:]))                       # never rises while moving out
    col = [_col(M(r), r) for r in R]
    runmin = np.minimum.accumulate(col)
    assert cf == pytest.approx([G.cf_law(Sv / s, H) for s in runmin], rel=1e-12)   # the law at the running minimum
    i = int(np.argmin(col))
    assert R[i] == pytest.approx(10.0 * 18.0 ** (1.0 / 3.0), abs=0.25)  # the column's minimum, [2(chi - 1)]^(1/3) r_v
    assert cf[-1] == cf[i]                                               # held from there on
    assert G.cf_law(Sv / col[-1], H) > cf[-1] + 0.1                      # where the law alone would have re-closed
    assert len(p["_cf_tab_r"].value) == len(R) + 1                       # one table row per new radius


def test_a_shell_falling_back_closes_holes_inside_their_opening_radius_and_never_ratchets():
    p, M = _vented(), _uniform(10.0)
    _vent(p)
    R = [float(r) for r in np.linspace(10.0, 40.0, 121)[1:]]             # steps of 0.25 pc, exact in floating point
    out = {r: _row(p, r, M(r)) for r in R}
    Mfar = M(40.0)                                                       # falling back: no new mass
    back = [_row(p, r, Mfar) for r in (35.0, 26.25, 20.0, 15.0, 12.5)]
    assert back == pytest.approx([out[r] for r in (35.0, 26.25, 20.0, 15.0, 12.5)], rel=1e-12)   # the outbound curve
    assert back[-1] > back[0]                                            # holes close again on the way in
    assert _row(p, 8.0, Mfar) == 1.0                                     # inside the vent radius: closed
    again = [_row(p, r, Mfar) for r in (12.5, 20.0, 35.0, 40.0)]         # out again over the same radii
    assert again == pytest.approx([out[r] for r in (12.5, 20.0, 35.0, 40.0)], rel=1e-12)
    assert len(p["_cf_tab_r"].value) == len(R) + 1                       # old ground adds no table row
    for _ in range(3):                                                   # cycles: no ratchet
        _row(p, 30.0, Mfar)
        assert _row(p, 40.0, Mfar) == out[40.0]


def test_a_row_repeating_the_furthest_radius_lowers_the_table():
    """A row at the furthest row radius with a column below the table's last entry (a column at the vent set above the
    row's, as two runner tests do; or a reconciliation row) lowers that entry, so the next row further out cannot read
    a higher column than this row did: C_f never rises on the way out (smoke 16216221, 2026-11-27)."""
    S0 = _col(1e4, 10.0)
    p = _vented(_cf_Sigma_vent=3.0 * S0, _cf_tab_r=[10.0], _cf_tab_S=[3.0 * S0])
    c0 = _row(p, 10.0, 1e4)                                              # same radius, a third of the table's column
    assert c0 == G.cf_law(3.0, H) and p["_cf_tab_S"].value == [S0]
    assert _row(p, 11.0, 1.5e4) == c0                                    # further out, the column grows: held
    assert p["_cf_tab_S"].value == [S0, S0]


def test_calls_beyond_the_furthest_radius_take_the_running_minimum_and_inside_it_ignore_the_mass():
    p, M = _vented(), _uniform(10.0)
    Sv = _vent(p)
    for r in np.linspace(10.0, 20.0, 41)[1:]:
        _row(p, float(r), M(float(r)))
    rt, St = p["_cf_tab_r"].value, p["_cf_tab_S"].value
    for r, m in ((25.0, 1e4), (25.0, 1e7), (30.0, M(30.0)), (20.0, 1e3)):    # the RHS's own (r, m), r >= 20
        assert G.cover_fraction(p, r, m) == G.cf_law(Sv / min(St[-1], _col(m, r)), H)
    for m in (1e3, 1e4, 1e9):                                            # old ground: the table, whatever m
        assert G.cover_fraction(p, 15.1, m) == pytest.approx(G.cf_law(Sv / float(np.interp(15.1, rt, St)), H),
                                                             rel=1e-12)


def test_hole_forces_off_keeps_the_law_bit_for_bit():
    on, off, M = _vented(), _vented("off"), _uniform(10.0)
    Sv = _vent(on)
    assert _vent(off) == Sv
    for r in [float(r) for r in np.linspace(10.0, 60.0, 101)[1:]]:
        c_off, c_on = _row(off, r, M(r)), _row(on, r, M(r))
        assert c_off == G.cf_law(Sv / _col(M(r), r), H)                  # the column now: re-closes while moving out
        assert c_on <= c_off
    assert len(off["_cf_tab_r"].value) == 1                              # hole_deficit is a no-op under off
    assert c_off > c_on + 0.1
    # the hole_forces gate in cover_fraction: inside the vent radius with a thin column the table (on) keeps the sky
    # closed while the law alone (off, whose hook also started a one-row table) opens it
    assert G.cover_fraction(off, 8.0, 1e3) == G.cf_law(Sv / _col(1e3, 8.0), H) < 1.0
    assert G.cover_fraction(on, 8.0, 1e3) == 1.0


def test_without_a_table_the_column_is_returned():
    """Params whose column at the vent is set by hand (older tests) carry no table: the law reads the column now."""
    p = _vented(_cf_Sigma_vent=_col(1e4, 10.0))
    assert G.outbound_column(p, 30.0, 1.5) == 1.5
    assert G.cover_fraction(p, 30.0, 2e4) == G.cf_law(_col(1e4, 10.0) / _col(2e4, 30.0), H)
