"""Mechanical checks for the docs/dev conventions (docs/dev/CONVENTIONS.md).

Active docs carry the four banner paragraphs (warn/living/persist/cross-check),
archived docs are frozen (warn + frozen), exempt how-to/manifest notes carry at
least the warn banner, and every top-level workstream dir is reachable from the
README index. Canonical banner text lives in docs/dev/CLAUDE.md; here we only check the
marker emojis, not the wording.
"""

from pathlib import Path

import pytest

DOCS_DEV = Path(__file__).resolve().parents[1] / "docs" / "dev"

WARN, LIVING, PERSIST, CROSSCHECK, FROZEN = "⚠️", "\U0001f504", "\U0001f4be", "\U0001f517", "\U0001f9ca"

# ⚠️-only docs: pure how-to-run harness READMEs, data-manifest notes, and
# machine-generated files (see CONVENTIONS.md §Banners). CLAUDE.md holds the
# canonical banner templates (agent context, not a plan doc). Paths relative to docs/dev.
EXEMPT = {
    "CLAUDE.md",
    "data/README.md",
    "failed-large-clouds/harness/README.md",
    "magic-numbers/harness/README.md",
    "performance/harness/README.md",
    "shell-solver/harness/README.md",
    "transition/cleanroom/data/README.md",
    "transition/harness/README.md",
    "transition/pdv-trigger/MANIFEST.md",
    "transition/pdv-trigger/SESSION_HANDOFF_2026-07-01.md",
    "transition/pt4/r1shadow/runs/README.md",
}

# ponytail: banners must sit near the top; 6 kB of head is plenty and avoids
# false hits from emoji mentioned deep in body prose.
HEAD = 6000


def _head(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")[:HEAD]


def _docs():
    return sorted(
        p.relative_to(DOCS_DEV).as_posix()
        for p in DOCS_DEV.rglob("*.md")
        if "to-be-removed" not in p.parts
    )


@pytest.mark.parametrize("rel", _docs())
def test_banners(rel):
    head = _head(DOCS_DEV / rel)
    assert WARN in head, f"{rel}: missing the warn banner"
    if rel in EXEMPT:
        return
    if rel.startswith("archive/"):
        assert FROZEN in head, f"{rel}: archived doc missing the frozen banner"
        # frozen replaces living: any later 'living' emoji must be body prose,
        # i.e. appear only after the frozen banner.
        if LIVING in head:
            assert head.index(FROZEN) < head.index(LIVING), (
                f"{rel}: archived doc still opens with the living-plan banner"
            )
    else:
        for emoji, name in [(LIVING, "living"), (PERSIST, "persist"), (CROSSCHECK, "cross-check")]:
            assert emoji in head, f"{rel}: active doc missing the {name} banner"


def test_readme_lists_every_workstream():
    readme = (DOCS_DEV / "README.md").read_text(encoding="utf-8")
    dirs = [p.name for p in DOCS_DEV.iterdir() if p.is_dir() and p.name != "to-be-removed"]
    missing = [d for d in dirs if f"{d}/" not in readme]
    assert not missing, f"docs/dev/README.md layout omits workstream dirs: {missing}"
