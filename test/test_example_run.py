"""The shipped example runs must stay loadable by the current reader.

`examples/runs/` is committed so `examples/quickstart.ipynb` works on a fresh clone,
with no simulation to run first. That makes those runs frozen artefacts of whatever
output schema was current when they were generated. If the metadata schema or the
reader moves on, this fails loudly here rather than silently breaking the notebook
for everyone who clones the repository.

When it fails: regenerate the examples (see `examples/README.md`), don't patch the
assertions.
"""
from pathlib import Path

import pytest

from trinity._output.run_constants import METADATA_VERSION
from trinity._output.trinity_reader import TrinityOutput

RUNS = Path(__file__).parent.parent / 'examples' / 'runs'
NAMES = ('homogeneous', 'powerlaw', 'bonnor_ebert')

pytestmark = pytest.mark.skipif(
    not RUNS.is_dir(),
    reason='example runs not present (examples/runs/ is optional in some checkouts)',
)


@pytest.fixture(scope='module')
def outputs():
    return {n: TrinityOutput.open(RUNS / n / 'dictionary.jsonl') for n in NAMES}


@pytest.mark.parametrize('name', NAMES)
def test_example_loads_with_current_reader(outputs, name):
    """The notebook's first cell, in test form."""
    out = outputs[name]
    assert len(out) > 0
    assert out.model_name


@pytest.mark.parametrize('name', NAMES)
def test_metadata_schema_matches_current_version(outputs, name):
    """A frozen example must not fall behind the reader's schema."""
    written = outputs[name].metadata.get('_metadata_version')
    assert written == METADATA_VERSION, (
        f'{name} was written with metadata schema v{written}, but the code now expects '
        f'v{METADATA_VERSION} — regenerate examples/runs/{name}/'
    )


@pytest.mark.parametrize('name', NAMES)
def test_keys_the_notebook_plots_are_present(outputs, name):
    """Guard the specific keys examples/quickstart.ipynb reads."""
    keys = outputs[name].keys   # a property, not a method
    for key in ('t_now', 'R2', 'v2', 'current_phase',
                'F_grav', 'F_ram', 'F_HII', 'F_rad'):
        assert key in keys, f'{key} missing from examples/runs/{name}/'


@pytest.mark.parametrize('name', NAMES)
def test_run_reached_a_terminal_state(outputs, name):
    """A run that never finished would be a poor advertisement."""
    assert outputs[name].termination, f'{name} has no termination block — it did not finish'


@pytest.mark.parametrize('name', NAMES)
def test_thinning_preserved_every_phase(outputs, name):
    """Thinning drops snapshots, so check no phase was thinned out of existence.

    The notebook shades plots by phase and the whole point of these runs is that they
    show the full lifecycle; losing one to an over-aggressive --every would be silent.
    """
    phases = set(outputs[name].get('current_phase', as_array=False))
    missing = {'energy', 'implicit', 'transition', 'momentum'} - phases
    assert not missing, (
        f'{name} is missing {sorted(missing)} — it was thinned too aggressively, '
        f'or did not run the full lifecycle'
    )


@pytest.mark.parametrize('name', NAMES)
def test_snapshots_are_chronological_and_unique(outputs, name):
    """The shipped runs must be sorted by time and free of duplicate snapshots.

    Raw TRINITY output is neither: snapshots are written in buffer-flush order,
    and a long run can repeat close to half its lines. `examples/thin_run.py`
    de-duplicates and sorts before shipping, so anything plotting these runs in
    file order gets a real trajectory rather than a zig-zag. This guards that.
    """
    t = outputs[name].get('t_now')
    assert all(b > a for a, b in zip(t, t[1:])), (
        f'{name} is not strictly increasing in t_now — re-run examples/thin_run.py'
    )


@pytest.mark.parametrize('name', NAMES)
def test_phase_sequence_is_forward_only(outputs, name):
    """energy -> implicit -> transition -> momentum, never backwards.

    run_expansion() runs the phases in a fixed order, so a shipped run that
    appears to go backwards means the snapshots are mis-ordered, not that the
    solver re-entered a phase.
    """
    order = {'energy': 0, 'implicit': 1, 'transition': 2, 'momentum': 3}
    ranks = [order[p] for p in outputs[name].get('current_phase', as_array=False)]
    assert all(b >= a for a, b in zip(ranks, ranks[1:])), (
        f'{name} has a backwards phase step — snapshots are out of order'
    )


def test_profile_arrays_survived_for_the_profile_plot(outputs):
    """The notebook's profile plot needs a snapshot with a real array in it."""
    out = outputs['homogeneous']
    longest = max(len(s.get('shell_r_arr', []) or []) for s in out)
    assert longest >= 10, f'no snapshot has a usable shell profile (longest was {longest})'
