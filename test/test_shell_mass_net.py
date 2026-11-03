"""shell_mass_net (docs/dev/phii-identity/mass-budget/PLAN.md): the shell holds the swept mass minus the bubble's.

Conduction evaporates shell gas into the hot bubble; the structure solve integrates what the bubble holds into
bubble_mass. Under the key (on by default) that mass is taken off the swept mass wherever the shell's mass acts -- the shell
solve's mass condition and the inertia and weight of the equation of motion -- in phases 1b and 1c only (1a unchanged,
ruling D-MB1; back in the shell in the momentum phase, D-MB2). shell_mass itself stays the swept mass.
"""

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import trinity.bubble_structure.get_bubbleParams as get_bubbleParams
import trinity.phase1_energy.energy_phase_ODEs as ODEs
from trinity._input.read_param import read_param
from trinity._input.registry import SPECS
from trinity.shell_structure.shell_structure import shell_structure_pure

REPO = Path(__file__).resolve().parents[1]

# An archived state whose LyC runs out inside the shell, so a neutral layer grows (test_cover_fraction_totals.py's
# 'depleted_front'); its bubble holds 22 % of the swept mass.
FRONT = dict(Qi=5.10769707029876e64, Pb=568523787.5685359, mShell=0.017255627982153413, R2=0.010596828851730349,
             Li=96433084786.95534, Ln=75447216220.5714, bubble_mass=0.003878283636525603)


def _params(phase, on):
    p = read_param(str(REPO / "param" / "simple_cluster.param"))
    p["current_phase"].value = phase
    p["shell_mass_net"].value = on
    p["Qi"].value, p["Li"].value, p["Ln"].value = FRONT["Qi"], FRONT["Li"], FRONT["Ln"]
    p["Pb"].value = FRONT["Pb"]
    p["R2"].value = p["rShell"].value = FRONT["R2"]
    p["shell_mass"].value = FRONT["mShell"]
    p["bubble_mass"].value = FRONT["bubble_mass"]
    return p


def test_registry_default_is_on():
    spec = next(s for s in SPECS if s.name == "shell_mass_net")
    assert spec.default == "True" and spec.run_const is True and spec.exclude_from_snapshot is True


@pytest.mark.parametrize("phase", ["energy", "implicit", "transition", "momentum"])
def test_evaporated_mass_by_phase(phase):
    assert get_bubbleParams.evaporated_mass(_params(phase, False)) == 0.0
    want = FRONT["bubble_mass"] if phase in ("implicit", "transition") else 0.0
    assert get_bubbleParams.evaporated_mass(_params(phase, True)) == want


def test_hand_built_params_without_the_key_are_off():
    assert get_bubbleParams.evaporated_mass({}) == 0.0
    assert get_bubbleParams.evaporated_mass({"shell_mass_net": SimpleNamespace(value="False")}) == 0.0


@pytest.mark.parametrize("on", [False, True])
def test_shell_profile_holds_the_net_mass(on):
    """The profile's own mass (ionised + neutral) is the swept mass less the bubble's with the key, the swept mass without.

    The two are independent quadratures of the same gas (agreement ~1e-7 on archived rows); a forgotten subtraction is 22 %.
    """
    s = shell_structure_pure(_params("implicit", on))
    want = FRONT["mShell"] - (FRONT["bubble_mass"] if on else 0.0)
    assert s.shell_mass_ion + s.shell_mass_neutral == pytest.approx(want, rel=1e-4)


def test_no_shell_left_is_refused():
    """A bubble holding the whole swept mass leaves no shell: the solve refuses it (D-MB10) instead of a profile of
    zero or negative mass. With the key off the same state solves."""
    p = _params("implicit", True)
    p["_shell_mass_net_on"].value = True       # the subtraction has started (D-MB18)
    p["bubble_mass"].value = FRONT["mShell"]
    with pytest.raises(ValueError, match="shell_mass_net"):
        shell_structure_pure(p)
    p["shell_mass_net"].value = False
    shell_structure_pure(p)


def test_subtraction_waits_until_the_bubble_holds_less():
    """D-MB18: a weak wind whose 1a lasts ~1 yr enters 1b with the bubble holding ~3x the swept mass (AU006W); the shell keeps
    it until the bubble holds less, the latch then stays on, and a later rise to the swept mass is refused (D-MB10)."""
    p = _params("implicit", True)
    p["bubble_mass"].value = 3.1 * FRONT["mShell"]
    assert get_bubbleParams.evaporated_mass(p) == 0.0 and p["_shell_mass_net_on"].value is False
    s = shell_structure_pure(p)                # solves with the whole swept mass, as in 1a
    assert s.shell_mass_ion + s.shell_mass_neutral == pytest.approx(FRONT["mShell"], rel=1e-4)
    p["bubble_mass"].value = FRONT["bubble_mass"]
    assert get_bubbleParams.evaporated_mass(p) == FRONT["bubble_mass"] and p["_shell_mass_net_on"].value is True
    p["bubble_mass"].value = FRONT["mShell"]
    with pytest.raises(ValueError, match="shell_mass_net"):
        shell_structure_pure(p)


# A physically plausible segment-0 snapshot, test_early_phase_override.py's (phase1a-init probe.param), set in phase 1b.
# shell_fAbsorbedIon = 1 and rShell < rCloud keep P_ext at zero, so the RHS needs no params lookups but the stubbed one.
SNAP = ODEs.ODESnapshot(
    shell_fAbsorbedIon=1.0, F_rad=7.39e4, rShell=4.2953e-5, shell_mass=9.9976e-5, isCollapse=True, n_IF=8.7e3,
    include_PHII=True, R_IF=4.2953e-5, mCluster=3.0, bubble_LTotal=1.31e6, Qi=1.6e47, Lmech_total=6.0845e8,
    v_mech_total=3739.24, G=4.4986e-3, k_B=1.3807e-16, gamma_adia=5.0 / 3.0, caseB_alpha=2.59e-13, PISM=2.3e5,
    nISM=1e2, TShell_ion=1e4, tSF=0.0, current_phase="implicit", rCloud=0.617, P_HII=1.858e9, phii_scheme="c3c",
    coverFraction=1.0, c_sound=1.4e3,
)


def test_rhs_moves_and_weighs_the_net_mass(monkeypatch):
    """isCollapse holds the swept mass at the snapshot's, so the key's RHS must equal, bit for bit, the legacy RHS given
    the net mass as the shell's and the evaporated gas as mass inside R2 (added to the cluster's)."""
    monkeypatch.setattr(ODEs, "get_current_sps_feedback",
                        lambda t, params: SimpleNamespace(Lmech_total=6.0845e8, v_mech_total=3739.24))
    t, y = 1.1487e-6, [4.2953e-5, 3739.2407, 3.1770e2]
    M, Mb = SNAP.shell_mass, 0.22 * SNAP.shell_mass
    keyed = ODEs.get_ODE_Edot_pure(t, y, replace(SNAP, evaporated_mass=Mb), None)
    by_hand = ODEs.get_ODE_Edot_pure(t, y, replace(SNAP, shell_mass=M - Mb, mCluster=SNAP.mCluster + Mb), None)
    legacy = ODEs.get_ODE_Edot_pure(t, y, SNAP, None)
    assert keyed == by_hand
    assert keyed[1] != legacy[1] and keyed[0] == legacy[0] and keyed[2] == legacy[2]


def test_swept_1b_rhs_moves_and_weighs_the_net_mass(monkeypatch):
    """Swept's 1b RHS (ImplicitSwept.rhs) with the key equals, bit for bit, the key-off RHS given the net mass as the swept
    mass and the evaporated gas added to the cluster's mass; a bubble holding the whole swept mass is refused (D-MB10)."""
    import trinity.phase1b_energy_implicit.get_betadelta as GB
    import trinity.phase2_momentum.swept as SW
    # a ROS state in 1b (docs/dev/phii-identity/data/hpc/c_20261026_215721Z/swept/ROS, the 11th 1b row)
    rs, u, Ms, Mb, Eb = 0.36503641553015964, 41.81423186153189, 7.047364647790844, 0.13901869145008655, 11337.116525750755
    R2, R1, Pb, Mdot = 0.35505674869183823, 0.1416187661149576, 43042.84780371602, 2562.2073557856893
    fb = SimpleNamespace(Lmech_total=20282258.66281576, v_mech_total=3739.330388145636)
    monkeypatch.setattr(SW, "get_current_sps_feedback", lambda t, params: fb)
    monkeypatch.setattr(SW, "edge_speed", lambda v, r, fabs_ion, params: v)
    monkeypatch.setattr(SW, "p_ext", lambda r, fabs_ion, params, turbulent=True: 0.0)
    monkeypatch.setattr(SW, "_p_conf", lambda R2t, E, fb_t, params, phase, t: (Pb, R1))
    monkeypatch.setattr(GB, "compute_R1_Pb", lambda *a: (R1, Pb))
    swept = {}
    monkeypatch.setattr(SW, "swept_mass", lambda r, U, M_floor, frozen, params: (swept["m"], Mdot))

    def rhs(on, m, dm_cluster, M_floor=Ms, started=False):
        p = _params("implicit", on)
        p["bubble_mass"].value = Mb
        p["shell_mass"].value = M_floor             # as the 1b runner writes it (sw.Ms) before building the RHS
        p["_shell_mass_net_on"].value = started     # D-MB18's latch: rhs() sets it here when Mb < shell_mass
        p["mCluster"].value = p["mCluster"].value + dm_cluster
        p["F_rad"].value = 12079.826019091612
        swept["m"] = m
        sw = object.__new__(SW.ImplicitSwept)
        sw.root, sw.Ms, sw.fabs_ion = {"K": 4244.1644641410785, "R2": R2}, M_floor, 1.0
        sw._closure, sw.tan, sw.zeta, sw.Eb0, sw.Rdot2 = (lambda r, E, M, t: R2), None, 1.0, Eb, 41.2
        return list(sw.rhs(p, 0.0, 0.0)(0.004594518710842068, np.array([rs, u, Eb, 1.0e6])))

    keyed, by_hand, legacy = rhs(True, Ms, 0.0), rhs(False, Ms - Mb, Mb), rhs(False, Ms, 0.0)
    assert keyed == by_hand
    assert keyed[1] != legacy[1] and keyed[0] == legacy[0]
    with pytest.raises(ValueError, match="shell_mass_net"):
        rhs(True, Ms, 0.0, M_floor=Mb, started=True)   # after the subtraction has started (D-MB18)


@pytest.mark.parametrize("on", [False, True])
def test_snapshot_carries_the_evaporated_mass(on):
    """create_ODE_snapshot wires the evaporated mass into c3c's RHS (bubble_mass in 1b with the key, 0.0 without)."""
    s = ODEs.create_ODE_snapshot(_params("implicit", on), SimpleNamespace(isDissolved=True))
    assert s.evaporated_mass == (FRONT["bubble_mass"] if on else 0.0)


def test_swept_root_refuses_before_its_bracket_ladder():
    """Swept's root raises the refusal itself: its bracket ladder would otherwise swallow the shell solve's ValueError as a
    rejected trial and end the run as a root failure (D-MB10)."""
    import trinity.phase2_momentum.swept as SW
    p = _params("implicit", True)
    with pytest.raises(ValueError, match="shell_mass_net"):
        SW.root_R2(1.2 * FRONT["R2"], FRONT["bubble_mass"], None, p)


def test_validator_refuses_a_non_bool():
    """'shell_mass_net 1' or 'yes' would otherwise read as off; True and False pass."""
    from trinity._input.errors import ParameterFileError
    from trinity._input.registry import _validate_shell_mass_net
    for ok in (True, False):
        _validate_shell_mass_net(ok, None)
    for bad in (1.0, 0.0, "yes", "on"):
        with pytest.raises(ParameterFileError, match="shell_mass_net"):
            _validate_shell_mass_net(bad, None)
