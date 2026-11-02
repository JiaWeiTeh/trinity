"""betadelta_wide_search: opt-in; off -> unchanged.

docs/dev/phii-identity/weak-wind/early-1b/F1_ADOPTION.md
"""
from trinity._input.registry import SPECS
from trinity.phase1b_energy_implicit import get_betadelta as GB


class _V:
    def __init__(self, v):
        self.value = v


ON = {'betadelta_wide_search': _V(True), 'betadelta_solver': _V('hybr')}


def _result(beta, delta, converged=True, no_root=False, reason=None):
    return GB.BetaDeltaResult(beta=beta, delta=delta, Edot_residual=0.0, T_residual=0.0,
                              total_residual=0.0, converged=converged, iterations=1,
                              no_physical_root=no_root, no_root_reason=reason)


def _island(b0, d0):
    """A physical structure only near (b0, d0)."""
    def g(beta, delta, params, seed):
        if abs(beta - b0) < 0.26 and abs(delta - d0) < 0.26:
            return beta - b0, delta - d0, None
        raise GB._NoPhysicalRoot("structure solve failed")
    return g


def test_registry_default_is_off():
    spec = next(s for s in SPECS if s.name == 'betadelta_wide_search')
    assert spec.default == 'False' and spec.run_const is True


def test_off_and_string_false_return_the_original():
    sentinel = object()
    key = 'betadelta_wide_search'
    for params in ({}, {key: _V(False)}, {key: _V('False')}):
        assert GB._wide_search(sentinel, params, 'grid') is sentinel


def test_on_seeds_hybr_at_an_interior_island(monkeypatch):
    seen = {}
    monkeypatch.setattr(GB, "_hybr_g_residual", _island(-0.5, -0.5))

    def hybr(b, d, params, method):
        seen['seed'] = (b, d)
        return _result(-0.5, -0.5)
    monkeypatch.setattr(GB, "_solve_betadelta_hybr", hybr)
    r = GB._wide_search(object(), ON, 'grid')
    assert seen['seed'] == (-0.5, -0.5) and r.converged and r.beta == -0.5


def test_edge_best_point_extends_the_box(monkeypatch):
    """The first box's best point is on the beta = 4 edge; the island's centre is at beta = 5."""
    def g(beta, delta, params, seed):
        if abs(beta - 5.0) < 1.01 and abs(delta + 1.0) < 0.26:
            return beta - 5.0, delta + 1.0, None
        raise GB._NoPhysicalRoot("structure solve failed")
    seen = {}
    monkeypatch.setattr(GB, "_hybr_g_residual", g)
    monkeypatch.setattr(GB, "_solve_betadelta_hybr",
                        lambda b, d, p, m: seen.setdefault('seed', (b, d)) and _result(b, d))
    GB._wide_search(object(), ON, 'grid')
    assert seen['seed'] == (5.0, -1.0)


def test_unconverged_or_no_point_returns_the_original(monkeypatch):
    sentinel = object()
    monkeypatch.setattr(GB, "_hybr_g_residual", _island(-0.5, -0.5))
    monkeypatch.setattr(GB, "_solve_betadelta_hybr",
                        lambda b, d, p, m: _result(b, d, converged=False))
    assert GB._wide_search(sentinel, ON, 'grid') is sentinel
    monkeypatch.setattr(GB, "_hybr_g_residual",
                        lambda b, d, p, s: (_ for _ in ()).throw(GB._NoPhysicalRoot("x")))
    assert GB._wide_search(sentinel, ON, 'grid') is sentinel


def test_wired_through_solve_betadelta_pure(monkeypatch):
    """A guess without structure, a rescue grid returning the guess, then the wide root."""
    fail = _result(0.8, -6 / 35, converged=False, no_root=True,
                   reason="structure solve failed at (0.8, -0.17)")
    calls = []

    def hybr(b, d, params, method):
        calls.append((b, d))
        return fail if (b, d) == (0.8, -6 / 35) else _result(b, d)
    monkeypatch.setattr(GB, "_solve_betadelta_hybr", hybr)
    monkeypatch.setattr(GB, "_solve_betadelta_legacy",
                        lambda b, d, p, m: _result(b, d, converged=False))
    monkeypatch.setattr(GB, "_hybr_g_residual", _island(-0.5, -0.5))
    r = GB.solve_betadelta_pure(0.8, -6 / 35, ON, rescue=True)
    assert r.beta == -0.5 and r.converged and calls[-1] == (-0.5, -0.5)
    off = {'betadelta_wide_search': _V(False), 'betadelta_solver': _V('hybr')}
    assert GB.solve_betadelta_pure(0.8, -6 / 35, off, rescue=True) is fail


# The first-segment restriction (F1_ADOPTION.md §8): the phase-1b runner passes wide=False after its first
# WIDE_SEARCH_MAX_SOLVES solves; wide=False must keep the search out, wide=True (the default) must not change anything.

def _structure_failure_everywhere(monkeypatch, scans):
    monkeypatch.setattr(GB, "_solve_betadelta_hybr",
                        lambda b, d, params, method: _result(b, d, converged=False, no_root=True,
                                                             reason="structure solve failed at the guess"))
    monkeypatch.setattr(GB, "_solve_betadelta_legacy",
                        lambda b, d, params, method="grid": _result(b, d, converged=False))   # the grid finds nothing better

    def scan(params, betas, deltas, seen):
        scans.append(len(betas) * len(deltas))
        return None                                                         # no structure anywhere: the search gives up
    monkeypatch.setattr(GB, "_wide_scan", scan)


def test_restriction_constant_is_three():
    assert GB.WIDE_SEARCH_MAX_SOLVES == 3


def test_wide_false_keeps_the_search_out(monkeypatch):
    scans = []
    _structure_failure_everywhere(monkeypatch, scans)
    r = GB.solve_betadelta_pure(0.8, -6 / 35, ON, rescue=True, wide=False)
    assert r.no_physical_root and "structure" in r.no_root_reason
    assert scans == []


def test_wide_true_is_the_default_and_scans(monkeypatch):
    scans = []
    _structure_failure_everywhere(monkeypatch, scans)
    r = GB.solve_betadelta_pure(0.8, -6 / 35, ON, rescue=True)
    assert r.no_physical_root
    assert scans == [77]                                                    # the 11 x 7 map, once


def test_wide_false_also_holds_inside_the_balance_retry(monkeypatch):
    # the default-on balance retry re-runs hybr + rescue: wide=False must keep the search out there too
    scans = []
    _structure_failure_everywhere(monkeypatch, scans)
    monkeypatch.setattr(GB, "_beta_from_balance", lambda params: -9.0)
    on_retry = dict(ON, betadelta_balance_retry=_V(True))
    r = GB.solve_betadelta_pure(0.8, -6 / 35, on_retry, rescue=True, wide=False)
    assert r.no_physical_root and scans == []
    r = GB.solve_betadelta_pure(0.8, -6 / 35, on_retry, rescue=True)
    assert r.no_physical_root and scans == [77, 77]                         # the first rescue's search and the retry's


def test_wide_false_at_the_rescue_second_return_point(monkeypatch):
    # the legacy grid proposes a new seed and hybr fails there too: the rescue's second hand-over to the search
    scans = []
    _structure_failure_everywhere(monkeypatch, scans)
    monkeypatch.setattr(GB, "_solve_betadelta_legacy",
                        lambda b, d, params, method="grid": _result(b + 0.02, d, converged=False))
    r = GB.solve_betadelta_pure(0.8, -6 / 35, ON, rescue=True, wide=False)
    assert r.no_physical_root and scans == []
    r = GB.solve_betadelta_pure(0.8, -6 / 35, ON, rescue=True)
    assert r.no_physical_root and scans == [77]
