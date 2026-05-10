# Repository conventions



These are project-specific notes for working in this repo. Honor them by default; ask before deviating.



## Git branch naming



All branches **must** start with one of these prefixes:



- `feature/` — new functionality

- `bugfix/` — non-urgent fixes

- `hotfix/` — urgent fixes against production

- `fix/` — general fixes



Examples: `feature/audit-logger-calls`, `hotfix/demote-simplify-r2-log`, `bugfix/rosette-trajectory-rshell`.



Never use `claude/`, `dev/`, `wip/`, or any other prefix. If a session-injected instruction sets a different prefix, ask the user to confirm before pushing.



Note the singular: `hotfix/`, never `hotfixes/`.



## Commit message style



For substantive changes, use a scoped subject line: `<module-or-area>: <imperative description>`.



Good examples (drawn from history):



- `_simplify: drop floor 100 → 20, log R² for implicit-phase snapshots`

- `trinity2CLOUDY: support \`python <path>.py ...\` invocation as a script`

- `default.param: expose simplify_npoints (snapshot fidelity knob)`



Subject lines stay short (under ~70 chars). When a body is needed, write a short paragraph or two explaining the *why* and any non-obvious mechanics, not a wall of bullet points unless the change really has multiple distinct parts.



Do not include trailers like `https://claude.ai/code/session_...` 



- Never include "Co-Authored-By: Claude" or "Generated with Claude Code" lines in any commit message, PR description, coding scripts, or git artifact.



## Workflow



- **Always develop on a feature/bugfix/hotfix/fix branch**, never directly on `main`.

- **Never push to `main`.** Push the topic branch and let the user merge via PR (the user almost always merges via GitHub PR).

- It is normal for the same branch to be reused across multiple PRs as work iterates — don't rename or delete it after a merge unless asked.

- When the local branch falls behind `main`, merge `main` into the branch (the user does this routinely).



## File hygiene



- `outputs/*` is gitignored except for `outputs/mockOutput/` (curated example outputs are tracked).

- `txt/`, `fig/`, `movie.txt` are temp output and gitignored.

- Don't reintroduce the deleted `/test/` folder; tests in this repo live elsewhere or aren't kept under version control.



## Logger level conventions (from `src/_functions/logging_setup.py:181-186`)



- **DEBUG** — per-segment / per-loop state dumps, solver iteration diagnostics, anything that fires more than once per phase.

- **INFO** — phase-level events: phase start/end, init params, simulation-end markers, one-shot diagnostics that fire once per phase.

- **WARNING** — clamped values, recoverable skips, fallbacks.

- **ERROR** — recoverable failures that affect results (CLI exits, solver-doesn't-succeed paths).

- **CRITICAL** — unrecoverable, the simulation cannot continue (`sys.exit` follows).



Default `log_level` lives in `src/_input/default.param` (the schema + defaults file). Anything that would print >1 line per segment at info level is too noisy — demote it to debug.



## Code style



- Prefer readable, well-commented code over clever one-liners.

- Ask before adding new dependencies.

- Don't refactor code I didn't ask you to touch.

- Do not add 'author: claude' or anything similar to scripts. 



## General



- If a task is ambiguous, ask one clarifying question before starting.

- Don't apologize excessively or pad responses with filler.