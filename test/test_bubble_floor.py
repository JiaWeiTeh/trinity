"""The bubble-structure RHS fails a trial at once when T falls below _T_FLOOR_FAIL (1e3 K).

A conductively heated bubble interior is hotter than its boundary (Weaver et al. 1977): integrated inward from the
_T_INIT_BOUNDARY anchor (3e4 K), a valid profile rises. A trial that cools below 1e3 K is infeasible. Before the floor
the integration ran on towards T = 0, where the T**-5/2 terms push LSODA's step below float resolution, and only the
RHS work cap (1e7 calls, ~3.5 min) ended it (docs/dev/bubble-solve-hang/PLAN_S1_T_FLOOR.md).
"""
import numpy as np
import pytest

import trinity.bubble_structure.bubble_luminosity as BL


def test_floor_constant_sits_below_the_anchor():
    assert BL._T_FLOOR_FAIL == 1.0e3
    assert BL._T_FLOOR_FAIL < BL._T_INIT_BOUNDARY


@pytest.mark.parametrize("T", [999.0, 1.0, 1.0e-6, 0.0, -5.0, np.nan, np.inf, -np.inf])
def test_rhs_raises_below_the_floor_or_nonfinite(T):
    # the guard is the RHS's first statement, so params is never read for these T
    with pytest.raises(BL.BubbleSolverError, match="floor"):
        BL._get_bubble_ODE(1.0, [1.0, T, -1.0], None, 1.0)
