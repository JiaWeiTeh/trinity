# Contributing to TRINITY

Thanks for your interest! Bug reports, fixes, and feature ideas are
welcome.

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
