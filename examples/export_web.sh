#!/usr/bin/env bash
# Render the example notebook into the website, as a native page.
#
#   ./examples/export_web.sh
#   cp -R examples/web/. ../trinity-web/
#
# The output mirrors trinity-web's own layout, so the copy above drops each file
# where it belongs in one go. Everything it writes is derived — safe to delete
# and rebuild — which is why examples/web/ is gitignored here.
#
# Markdown rather than HTML on purpose: rendered by the site's own Markdown
# component, the notebook becomes a real docs page. Headings feed the "on this
# page" rail, code blocks inherit the site's styling and copy button, and the
# maths goes through the KaTeX already in that pipeline. An exported HTML page
# needed a stylesheet and a copy-button script injected into it just to look
# like it belonged; none of that is needed now.
set -euo pipefail

cd "$(dirname "$0")/.."
OUT=examples/web
SITE_PATH=/trinity-web/notebook          # vite `base` + where the figures land

python3 -c "import nbconvert" 2>/dev/null || {
    echo "nbconvert not found — pip install nbconvert"; exit 1; }

rm -rf "$OUT"
mkdir -p "$OUT/src/docs" "$OUT/public/notebook"

python3 -m jupyter nbconvert \
    --to markdown \
    --output-dir "$OUT/public/notebook" \
    --output quickstart \
    examples/quickstart.ipynb

# The notebook itself, so the page can offer a download.
cp examples/quickstart.ipynb "$OUT/public/notebook/quickstart.ipynb"

python3 - "$OUT" "$SITE_PATH" <<'PY'
import re
import shutil
import sys
from pathlib import Path

out, site_path = Path(sys.argv[1]), sys.argv[2]
generated = out / 'public' / 'notebook' / 'quickstart.md'
body = generated.read_text()

# nbconvert writes figures beside the markdown and links them relatively. The
# page is served by a single-page app from a different route, so the links have
# to be absolute.
body = body.replace('](quickstart_files/', f']({site_path}/quickstart_files/')

# Drop the notebook's own H1: the page supplies its own title, and two would
# read as a mistake.
body = re.sub(r'\A#\s+[^\n]*\n+', '', body)

header = f"""# Tutorial notebook

[Download this notebook]({site_path}/quickstart.ipynb) to run it yourself, or read it
here. It works on a fresh clone — the runs it opens ship with the repository.

"""

(out / 'src' / 'docs' / '03-notebook.md').write_text(header + body)
generated.unlink()          # the markdown belongs in src/docs, not in public/

figures = out / 'public' / 'notebook' / 'quickstart_files'
n_figures = len(list(figures.glob('*'))) if figures.is_dir() else 0
print(f"\n  page: src/docs/03-notebook.md ({len(header + body) // 1024} KB)")
print(f"  assets: public/notebook/ ({n_figures} figures + the .ipynb)")
PY

printf '\nMove it across with:\n  cp -R %s/. ../trinity-web/\n' "$OUT"
