"""Suite-wide pytest hooks: the stored goldens of the full-run tests.

A golden pins a number a run produced, so it is meant to turn red once, in the PR that changes the
result, and to be updated in that PR with the reason. ``pytest --regen-goldens <test>`` rewrites the
test's ``test/goldens/<name>.json`` from the run instead of comparing; review the JSON diff, then
write the reason next to the test's history comment and in the commit message.
"""

import json
from pathlib import Path

import pytest

GOLDEN_DIR = Path(__file__).parent / "goldens"


def pytest_addoption(parser):
    parser.addoption(
        "--regen-goldens",
        action="store_true",
        help="rewrite test/goldens/*.json from this run instead of comparing (review the diff)",
    )


@pytest.fixture
def golden(request):
    """``golden(name, actual)`` returns the stored values of ``test/goldens/<name>.json``.

    With ``--regen-goldens`` it writes ``actual`` there first and returns it, so the test's own
    assertions pass trivially and the JSON diff shows the old and new numbers.
    """
    regen = request.config.getoption("--regen-goldens")

    def _golden(name, actual):
        path = GOLDEN_DIR / f"{name}.json"
        if regen:
            path.write_text(json.dumps(actual, indent=2) + "\n")
        return json.loads(path.read_text())

    return _golden
