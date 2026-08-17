"""The shipped example run must stay loadable by the current reader.

`examples/lifecycle_run/` is committed so `examples/quickstart.ipynb` works on a
fresh clone, with no simulation to run first. That makes it a frozen artefact of
whatever output schema was current when it was generated. If the metadata schema
or the reader moves on, this test fails loudly here rather than silently breaking
the notebook for everyone who clones the repository.

When it fails: regenerate the example (see `examples/README.md`), don't patch the
assertions.
"""
from pathlib import Path

import pytest

from trinity._output.run_constants import METADATA_VERSION
from trinity._output.trinity_reader import TrinityOutput

EXAMPLE = Path(__file__).parent.parent / 'examples' / 'lifecycle_run'

pytestmark = pytest.mark.skipif(
    not (EXAMPLE / 'dictionary.jsonl').exists(),
    reason='example run not present (examples/ is optional in some checkouts)',
)


@pytest.fixture(scope='module')
def output():
    return TrinityOutput.open(EXAMPLE / 'dictionary.jsonl')


def test_example_loads_with_current_reader(output):
    """The notebook's first cell, in test form."""
    assert len(output) > 0
    assert output.model_name


def test_metadata_schema_matches_current_version(output):
    """The frozen example must not fall behind the reader's schema."""
    written = output.metadata.get('_metadata_version')
    assert written == METADATA_VERSION, (
        f'example run was written with metadata schema v{written}, but the code now '
        f'expects v{METADATA_VERSION} — regenerate examples/lifecycle_run/'
    )


def test_keys_the_notebook_plots_are_present(output):
    """Guard the specific keys examples/quickstart.ipynb reads."""
    for key in ('t_now', 'R2', 'v2', 'current_phase'):
        assert key in output.keys(), f'{key} missing from the example run'
    for key in ('F_grav', 'F_ram', 'F_HII', 'F_rad'):
        assert key in output.keys(), f'force budget key {key} missing from the example run'


def test_run_reached_a_terminal_state(output):
    """A run that never finished would be a poor advertisement."""
    assert output.termination, 'example run has no termination block — it did not finish'


def test_at_least_one_snapshot_carries_a_usable_profile(output):
    """The notebook's profile plot needs a snapshot with a real array in it."""
    longest = max(len(snap.get('shell_r_arr', []) or []) for snap in output)
    assert longest >= 10, f'no snapshot has a usable shell profile (longest was {longest})'
