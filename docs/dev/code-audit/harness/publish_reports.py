"""Copy raw agent reports into docs/dev/code-audit/slices/ with the banners on.

Agents write plain markdown to a scratch dir; asking each of ~40 of them to
reproduce four banner paragraphs verbatim is a reliability problem, so the
banner is stamped here instead (test/test_docs_dev_conventions.py requires it).

    python docs/dev/code-audit/harness/publish_reports.py <raw_dir>
"""

import pathlib
import re
import sys

BANNERS = pathlib.Path(__file__).resolve().parents[2] / "CLAUDE.md"
DEST = pathlib.Path(__file__).resolve().parents[1] / "slices"


def banner_text():
    """The four canonical active-doc banners, lifted from docs/dev/CLAUDE.md."""
    block = re.search(r"```markdown\n(> ⚠️.*?)```", BANNERS.read_text(), re.S)
    return block.group(1).rstrip() + "\n"


def main(raw_dir):
    banners = banner_text()
    DEST.mkdir(parents=True, exist_ok=True)
    for src in sorted(pathlib.Path(raw_dir).glob("*.md")):
        body = src.read_text().lstrip()
        title, _, rest = body.partition("\n")
        if not title.startswith("# "):
            title, rest = f"# {src.stem}", body
        (DEST / src.name).write_text(
            f"{title}\n\n{banners}\n"
            f"**Status (2026-07-29):** 📘 raw agent report — provenance for "
            f"`FINDINGS.md`; unreconciled and unverified on its own.\n\n"
            f"{rest.lstrip()}"
        )
        print(f"published {src.name}")


if __name__ == "__main__":
    main(sys.argv[1])
