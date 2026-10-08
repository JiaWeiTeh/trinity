# Contributing to TRINITY

Thanks for your interest! Bug reports, fixes, and feature ideas are
welcome.

## Repository layout

```
run.py         single entry point for individual runs and parameter sweeps
trinity/       the package: solver, evolution phases, bubble/shell/cloud physics, I/O
param/         .param config files (the tracked ones are worked examples)
lib/default/   bundled defaults — SB99 SPS table + cooling tables (quickstart runs out of the box)
paper/         scripts that regenerate published figures (see "Reproducing the figures")
docs/dev/      internal plan & audit write-ups (not user documentation)
test/          pytest test suite
tools/         small CLI utilities (param generation, audits, output comparisons)
```

## Dev environment

```bash
git clone https://github.com/JiaWeiTeh/trinity
cd trinity
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pre-commit install
```

## Running tests and lint

```bash
pytest test/
pre-commit run --all-files
```

`pre-commit` only enables bug-class checks (undefined names, syntax
errors, redefinitions). Pure-style rules are intentionally out of scope
so existing code does not need mass reformatting to unblock commits.

## Filing issues

When reporting a bug, please include:

- Python version + OS
- The `.param` file you used (or relevant excerpt)
- The traceback or unexpected output

## Pull requests

- Branch from `main`.
- Keep PRs focused — one logical change per PR.
- Tests for new behaviour are appreciated but not required for small fixes.
