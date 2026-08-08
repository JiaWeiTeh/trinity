"""The mu_* family is derived, so setting it in a .param must fail loudly.

`read_param` recomputes all four of ``mu_convert``/``mu_atom``/``mu_ion``/``mu_mol``
from ``x_He`` and ``Z_He`` and assigns straight onto the existing ``DescribedItem``.
A user value was therefore silently discarded: the run proceeded on the derived
numbers while the user believed their composition had been applied, and nothing in
the output said otherwise.

The anti-stomp guard in the same function cannot catch this. It tests
``params[k] is not v_before`` -- object *identity* -- and the derivation mutates
``.value`` in place, so identity is preserved and the guard is structurally blind
to the whole class (code audit S12a-R-01).

Refusing is the fix rather than honouring: the four are not independent (each is a
function of ``x_He`` and ``Z_He``, as is ``chi_e``), so accepting one while the rest
stayed derived would produce a silently inconsistent composition.
"""

from __future__ import annotations

import pytest

from trinity._input.errors import ParameterFileError
from trinity._input.read_param import read_param

DERIVED = ("mu_convert", "mu_atom", "mu_ion", "mu_mol")


def _write(tmp_path, body):
    p = tmp_path / "probe.param"
    p.write_text("mCloud    1e5\nsfe    0.3\n" + body, encoding="utf-8")
    return str(p)


@pytest.mark.parametrize("key", DERIVED)
def test_setting_a_derived_mu_raises(tmp_path, key):
    path = _write(tmp_path, f"{key}    1.23\n")

    with pytest.raises(ParameterFileError) as excinfo:
        read_param(path)

    message = str(excinfo.value)
    assert key in message
    # The message has to point somewhere useful, or the user just deletes the key
    # and keeps the wrong mental model of what controls composition.
    assert "x_He" in message and "Z_He" in message


def test_all_four_are_reported_together(tmp_path):
    path = _write(tmp_path, "".join(f"{k}    1.23\n" for k in DERIVED))

    with pytest.raises(ParameterFileError) as excinfo:
        read_param(path)

    for key in DERIVED:
        assert key in str(excinfo.value)


def test_a_param_that_does_not_set_them_is_unaffected(tmp_path):
    """The guard must be invisible to every config that inherits the defaults."""
    params = read_param(_write(tmp_path, "x_He    0.1\nZ_He    2\n"))

    # Derived values still land, and still match the historical encodings.
    assert params["mu_convert"].value == pytest.approx(
        params["mu_atom"].value * (1 + 0.1) / 1.0, rel=1e-12
    )
    for key in DERIVED:
        assert params[key].value > 0
