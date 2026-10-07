"""shell_bc: the inner-boundary pressure of the shell solve ('Pb' default, 'drive' = max(Pb, P_HII_prev)).

PLAN.md row 63 option (1), 2026-10-06. 'drive' is a consistency choice under the c3c closure
(one pressure at the surface R2), not a physics repair: the registry entry carries the
double-count caveat and the measured trajectory effect. These tests pin

  * the default is 'Pb' and reproduces eq. 14 exactly (bit-identity of the default path is
    the full-run gate, test_shell_validity_report's docstring);
  * 'drive' uses max(Pb, P_HII) as the boundary, thins the layer, and reduces to 'Pb' where
    P_HII <= Pb;
  * the stored shell_P_bc is the pressure actually used, and shell_bc_mismatch compares
    P_HII against it;
  * the validator rejects unknown values and 'drive' under any closure other than c3c.

Full-run gate for the commit that added the switch (2026-10-07, separate processes):
  default 'Pb'   cloud_example_BE (285 snapshots) and cloud_example_homogeneous (503) identical
                 on every pre-existing key of every row to the runs made before the switch
                 existed; shell_P_bc == Pb on every row.
  'drive'        the homogeneous run reproduces, on every pre-existing key of all 507 rows, a
                 run made by wrapping shell_structure_pure from OUTSIDE the package with the
                 same rule (two implementations, one answer).
  Pb vs drive    energy/implicit rows identical on both configs. Homogeneous: momentum R2
                 +4.05 % at 15 Myr, turnaround 14.02 -> 14.52 Myr, f_abs = 1 throughout,
                 R_IF/R2 1.20-1.26 instead of 1.4-2.6. BE: recollapse 3.981 -> 3.989 Myr,
                 R2 within 1 % until the last 0.2 Myr of the collapse, f_abs = 1 throughout,
                 R_IF/R2 1.17-1.23 instead of 1.2-1.74. Neither run dissolves by its end.
"""
from pathlib import Path

import numpy as np
import pytest

from trinity._input.dictionary import DescribedDict, DescribedItem
from trinity._input.errors import ParameterFileError
from trinity._input.read_param import read_param
from trinity.shell_structure.shell_structure import record_validity, shell_structure_pure

REPO = Path(__file__).resolve().parents[1]
EXAMPLE = REPO / "examples" / "runs" / "homogeneous"


@pytest.fixture(autouse=True)
def _no_crash_handlers(monkeypatch):
    monkeypatch.setattr(DescribedDict, "_register_crash_handlers", lambda self: None)


def test_default_is_Pb():
    p = read_param(str(REPO / "param" / "simple_cluster.param"))
    assert p["shell_bc"].value == "Pb"


@pytest.fixture
def fired_state():
    """A stored momentum-phase state with C3c fired (P_HII > Pb), from the committed example run."""
    if not (EXAMPLE / "dictionary.jsonl").is_file():
        pytest.skip("examples/runs/homogeneous not present")
    snaps = DescribedDict.load_snapshots(EXAMPLE)
    sid = next(k for k in sorted(snaps, key=int)
               if snaps[k]["current_phase"] == "momentum" and snaps[k]["P_HII"] > snaps[k]["Pb"])
    p = DescribedDict.load_snapshot(EXAMPLE, sid)
    for k, v in {"allowShellDissolution": True, "coverFraction": 1.0, "phii_scheme": "c3c"}.items():
        if k not in p:
            p[k] = DescribedItem(v)
    return p


def _n0(p, P):
    return p["mu_ion_shell"].value / p["mu_convert"].value / (p["k_B"].value * p["TShell_ion"].value) * P


def test_Pb_mode_is_eq14_and_records_the_pressure_used(fired_state):
    p = fired_state
    p["shell_bc"] = DescribedItem("Pb")
    sp = shell_structure_pure(p)
    assert sp.shell_P_bc == p["Pb"].value
    assert sp.shell_n0 == _n0(p, p["Pb"].value)
    # the example run was written by an earlier engine; the live path's bit-identity is the
    # full-run gate, this only says it is the same layer
    assert sp.R_IF / p["R2"].value == pytest.approx(p["R_IF"].value / p["R2"].value, rel=1e-3)


def test_drive_mode_uses_the_larger_pressure_and_thins_the_layer(fired_state):
    p = fired_state
    Pb, PH = p["Pb"].value, p["P_HII"].value
    assert PH > Pb
    p["shell_bc"] = DescribedItem("Pb")
    ref = shell_structure_pure(p)
    p["shell_bc"] = DescribedItem("drive")
    sp = shell_structure_pure(p)
    assert sp.shell_P_bc == PH
    assert sp.shell_n0 == _n0(p, PH)
    assert sp.R_IF < ref.R_IF
    # a uniform layer at the C3c pressure is the cavity volume again: R_IF/R2 -> 2**(1/3); the
    # hydrostatic density gradient keeps it below that
    assert 1.0 < sp.R_IF / p["R2"].value <= 2 ** (1 / 3) + 1e-9


def test_drive_mode_reduces_to_Pb_where_confined(fired_state):
    p = fired_state
    p["P_HII"].value = 0.0                         # C3c's confined branch
    p["shell_bc"] = DescribedItem("Pb")
    ref = shell_structure_pure(p)
    p["shell_bc"] = DescribedItem("drive")
    sp = shell_structure_pure(p)
    assert sp.shell_P_bc == p["Pb"].value
    assert sp.R_IF == ref.R_IF and sp.rShell == ref.rShell and sp.shell_fAbsorbedIon == ref.shell_fAbsorbedIon


def test_mismatch_flag_compares_against_the_pressure_used(fired_state):
    p = fired_state
    for k in ("shell_bc_mismatch", "shell_vt_ci", "shell_vt_t_prev", "shell_vt_thickness_prev", "shell_P_bc"):
        if k not in p:
            p[k] = DescribedItem(np.nan if k != "shell_bc_mismatch" else False)
    p["shell_P_bc"].value = p["Pb"].value          # what 'Pb' mode stores
    record_validity(p)
    assert p["shell_bc_mismatch"].value is True
    p["shell_P_bc"].value = p["P_HII"].value       # what 'drive' mode stores once P_HII has settled
    record_validity(p)
    assert p["shell_bc_mismatch"].value is False


def _param_file(tmp_path, extra):
    f = tmp_path / "t.param"
    f.write_text("mCloud    1e6\nsfe    0.01\ndens_profile    densPL\ndensPL_alpha    0\nnCore    1e3\n" + extra)
    return str(f)


def test_validator_rejects_unknown_value(tmp_path):
    with pytest.raises(ParameterFileError, match="shell_bc"):
        read_param(_param_file(tmp_path, "shell_bc    maxPb\n"))


@pytest.mark.parametrize("scheme", ["front", "o1", "k11"])
def test_drive_is_c3c_only(tmp_path, scheme):
    with pytest.raises(ParameterFileError, match="c3c"):
        read_param(_param_file(tmp_path, f"shell_bc    drive\nphii_scheme    {scheme}\n"))


def test_drive_accepted_with_c3c(tmp_path):
    p = read_param(_param_file(tmp_path, "shell_bc    drive\nphii_scheme    c3c\n"))
    assert p["shell_bc"].value == "drive"
