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

command -v jupyter >/dev/null || { echo "jupyter not found — pip install jupyter"; exit 1; }

rm -rf "$OUT"
mkdir -p "$OUT"

# The rendered showcase: one self-contained file, images inlined, no input
# prompts or toolbar so it reads as a page rather than as an editor.
jupyter nbconvert \
    --to html \
    --embed-images \
    --no-prompt \
    --output-dir "$OUT" \
    --output quickstart.html \
    examples/quickstart.ipynb

# The notebook itself, so the site can offer a download alongside the render.
cp examples/quickstart.ipynb "$OUT/quickstart.ipynb"

printf '\n%s\n' "$OUT contains:"
ls -lh "$OUT" | tail -n +2 | awk '{printf "  %-22s %s\n", $9, $5}'
printf '\nMove it across with:\n  cp -R %s/. ../trinity-web/public/notebook/\n' "$OUT"
