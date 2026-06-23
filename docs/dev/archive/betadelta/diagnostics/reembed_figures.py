#!/usr/bin/env python3
"""Re-embed the diagnostics figures into insights_betadelta_illustrated.html.

The illustrated β–δ report (../insights_betadelta_illustrated.html) is a hand-authored
static page whose figures are inline base64. The figure GENERATORS live here in
diagnostics/ (analyze_arms.py, analyze_negvel.py, analyze_probe.py, plot_hunt.py,
cage_compare.py, make_rootmap_gif.py). After re-running those, this script swaps each
figure's base64 blob in the HTML for the freshly generated file on disk, matched by the
<img> alt text (which carries the figure's filename). Idempotent; run it after
regenerating the figures so the static report — and storyline_s0, which stitches it —
stay in sync.

    python docs/dev/archive/betadelta/diagnostics/reembed_figures.py
"""
import base64
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent  # diagnostics/
HTML = HERE.parent / "insights_betadelta_illustrated.html"

# Order-independent: process each whole <img ...> tag (the base64 blob has no '>'),
# read its alt to find the figure file, and swap just the base64 blob in its src.
_TAG = re.compile(r"<img\b[^>]*>", re.I)
_ALT = re.compile(r'alt="([^"]*)"', re.I)
_SRC = re.compile(r'(src="data:image/(?:png|gif);base64,)[A-Za-z0-9+/=]*(")', re.I)


def main() -> None:
    html = HTML.read_text(encoding="utf-8")
    updated, missing = [], []

    def fix_tag(m: re.Match) -> str:
        tag = m.group(0)
        am = _ALT.search(tag)
        if not am:
            return tag
        fname = am.group(1).split()[0].strip()  # leading token of alt, e.g. rootmap_cage.gif
        fpath = HERE / fname
        if not fpath.exists():
            missing.append(fname)
            return tag
        b64 = base64.b64encode(fpath.read_bytes()).decode()
        updated.append(fname)
        return _SRC.sub(lambda s: s.group(1) + b64 + s.group(2), tag, count=1)

    HTML.write_text(_TAG.sub(fix_tag, html), encoding="utf-8")
    print(f"re-embedded {len(updated)} figures: {updated}")
    if missing:
        print(f"MISSING (left unchanged): {missing}")


if __name__ == "__main__":
    main()
