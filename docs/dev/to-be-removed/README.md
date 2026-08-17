
## sphinx/ (moved 2026-08-17)

The whole Read the Docs / Sphinx tree — `source/`, `Makefile`, `make.bat`,
`requirements.txt` — retired in one move. It documented an older TRINITY, the RTD site it
fed is being deleted, and `jiaweiteh.github.io/trinity-web` is the documentation now. If
Sphinx is ever wanted again it should be regenerated from current source rather than
resurrected from here.

Nothing depended on it: no package code, no test. The references that did point at it were
updated in the same change — `MANIFEST.in` (no longer packages it), `pyproject.toml` (the
now-dead `[docs]` extra removed), `param/README.md` and `tools/gen_default_param.py` (now
link to the website's parameter reference), and the layout blocks plus the `make html`
command in `README.md`, `CLAUDE.md` and `AGENTS.md`.

`trinity_reader.rst` and `visualization.rst` were the two pages the website never had.
Their content is the source material for the site's "reading outputs" page; harvest from
here before deleting the folder.
