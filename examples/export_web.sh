#!/usr/bin/env bash
# Build everything trinity-web needs from the example notebook, into one folder.
#
#   ./examples/export_web.sh
#   cp -R examples/web/. ../trinity-web/public/notebook/
#
# Run it from the repository root, after the notebook has been executed and
# saved with its outputs. Everything it writes is derived — safe to delete and
# rebuild at any time, which is why examples/web/ is gitignored here: the
# published copy lives in trinity-web, not twice.
set -euo pipefail

cd "$(dirname "$0")/.."
OUT=examples/web

python3 -c "import nbconvert" 2>/dev/null || {
    echo "nbconvert not found — pip install nbconvert"; exit 1; }

rm -rf "$OUT"
mkdir -p "$OUT"

# The rendered page: one self-contained file, images inlined, no input prompts
# or execution counts, so it reads as a document rather than as an editor.
python3 -m jupyter nbconvert \
    --to html \
    --embed-images \
    --no-prompt \
    --output-dir "$OUT" \
    --output quickstart.html \
    examples/quickstart.ipynb

# Restyle it to match trinity-web: the site's serif for prose, Inter for UI,
# a readable measure, and its ink colours. Without this the page arrives in
# Jupyter's own theme and reads as a foreign object embedded in the site.
python3 - "$OUT/quickstart.html" <<'PY'
import sys
from pathlib import Path

page = Path(sys.argv[1])
html = page.read_text()

style = """
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500&family=Source+Serif+4:opsz,wght@8..60,400;8..60,600&display=swap" rel="stylesheet">
<style>
  /* Match trinity-web's typography so the notebook reads as part of the site. */
  :root {
    --jp-content-font-family: 'Source Serif 4', Georgia, serif;
    --jp-ui-font-family: 'Inter', sans-serif;
    --jp-content-font-size1: 16px;
    --jp-content-line-height: 1.65;
  }
  body { background: #FFFEFA; color: #1E2430; }
  .jp-Notebook { max-width: 820px; margin: 0 auto; padding: 2.5rem 1.25rem 4rem; }
  .jp-RenderedHTMLCommon { font-family: 'Source Serif 4', Georgia, serif; color: #1E2430; }
  .jp-RenderedHTMLCommon h1,
  .jp-RenderedHTMLCommon h2,
  .jp-RenderedHTMLCommon h3 { font-weight: 600; letter-spacing: -0.01em; margin-top: 2.2em; }
  .jp-RenderedHTMLCommon h1 { font-size: 2rem; margin-top: 0; }
  .jp-RenderedHTMLCommon h2 { font-size: 1.35rem; }
  .jp-RenderedHTMLCommon a { color: #0EA5C8; }
  .jp-RenderedHTMLCommon table { font-family: 'Inter', sans-serif; font-size: 0.85rem; }
  .jp-RenderedHTMLCommon blockquote {
    border-left: 3px solid #D8D2C6; color: #5E6776; padding-left: 1rem; font-style: normal;
  }
  /* Code: quieter chrome than Jupyter's default, closer to the site's blocks. */
  .jp-CodeCell .jp-Editor, .jp-InputArea-editor {
    background: #F7F4EE; border: 1px solid #E7E1D7; border-radius: 6px;
  }
  .jp-OutputArea-output pre { font-size: 0.82rem; line-height: 1.5; }
  .jp-OutputArea-output img { max-width: 100%; height: auto; }
</style>
"""

if '</head>' in html:
    html = html.replace('</head>', style + '</head>', 1)
    page.write_text(html)
    print('  restyled to match trinity-web')
else:
    print('  WARNING: no </head> found, page left in the default theme')
PY

# The notebook itself, so the site can offer a download alongside the render.
cp examples/quickstart.ipynb "$OUT/quickstart.ipynb"

printf '\n%s\n' "$OUT contains:"
ls -lh "$OUT" | tail -n +2 | awk '{printf "  %-22s %s\n", $9, $5}'
printf '\nMove it across with:\n  cp -R %s/. ../trinity-web/public/notebook/\n' "$OUT"
