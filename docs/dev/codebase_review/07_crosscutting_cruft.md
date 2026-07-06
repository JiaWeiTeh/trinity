> ⚠️ **This document may be out of date — verify before trusting it.** It is a
> point-in-time analysis/audit, not a maintained spec; the code moves faster
> than these notes (paths, line numbers, and "what shipped" status drift).
> **Any agent or person reading this: treat it as unverified. Flag that it may
> be stale and re-check each claim, snippet, and line reference against the
> current source before relying on it.**
>
> 🔄 **Living document — recheck and refine on every visit.** This is an
> evolving audit, not a frozen record. Any agent or person who opens this file
> must, as part of the visit: (1) re-verify the claims and line references above
> against current source; (2) update anything that has drifted; (3) reconsider
> the findings themselves — if a finding is stale, mis-scoped, or a better fix
> has landed, revise it and note what changed and why (date it). Leave it better
> than you found it. **Keep all banner paragraphs at the top of every plan and
> analysis doc.**
>
> 💾 **Persist diagnostics — commit, don't re-run.** The container is ephemeral
> and full runs cost minutes-to-hours, so any diagnostic worth keeping must be
> saved as a committed artifact (a CSV/table under `docs/dev/data/`, or a
> force-added harness/figure under `scratch/`) — never left in `/tmp` or an
> untracked `outputs/`. A future visit must be able to reproduce or compare
> against the numbers **without re-running**; record the exact config + command
> that produced each artifact.
>
> 🔗 **Cross-check the sibling docs — keep the workstream self-consistent.** This file is one of
> several living docs for its workstream (its `PLAN.md`, `FINDINGS.md`, `runs/README.md`, `NOTE_PATCHES.md`,
> and any other notes in the same folder). They drift out of sync *with each other* as fast as they drift
> from the code. Any agent or person editing one MUST, as part of the visit, circle back through the
> siblings and reconcile: if a number, status, claim, or line reference here contradicts a sibling — or a
> sibling has gone stale — fix it (or flag it, dated) so no two docs in the workstream disagree. Never
> update one in isolation.

# 07 — Cross-cutting sweep & cruft

**Scope.** A static, repo-wide hygiene/privacy audit from the perspective of a stranger who `git clone`s this public repo. I swept all tracked files (`git ls-files`, `git grep`, `git check-ignore`) for hardcoded machine/personal paths, secrets, committed generated artifacts, debug leftovers, TODO/FIXME markers, `.gitignore`-vs-tracked inconsistencies, the mandated `docs/dev/` staleness banner, and confusing top-level dirs. The dominant real finding is a **65-file `scratch/` tree that `.gitignore` explicitly says is "kept locally, not tracked" yet is fully committed (~6.4 MB, mostly PNG/GIF/jsonl artifacts)**. Privacy is clean: the only email and personal name are the intentional author/contact attribution; there are no secrets, tokens, debug breakpoints, or `__pycache__`/`.DS_Store` junk tracked. The three files `.gitignore` flags as "Private" (`paper/barnes26/`, `test/test_barnes_population.py`, `movie.txt`) are confirmed **not** tracked — no leak. `outputs/mockOutput/` is confirmed by-design test fixtures (referenced by `test/test_cloudy_*.py` and `test/test_metadata.py`), not an accidental output dump.

> 🔄 **Update (post-audit):** the `scratch/` tree discussed below was subsequently
> moved into `docs/dev/` and the top-level `.gitignore` anchored to `/scratch/`, so
> the committed diagnostics are now *normally* tracked (the ignored-but-tracked
> contradiction is resolved) and top-level `scratch/` is local-only. They have since
> been reorganised into per-workstream folders: the old `scratch/phase2`/`phase6`
> are now `docs/dev/archive/betadelta/diagnostics/` and `docs/dev/archive/betadelta/velstruct/`, and
> the transition harness is `docs/dev/transition/harness/`. The H1 finding below
> records the **original** state; see `CODEBASE_REVIEW.md` → Status (Rounds 3–4).

---

## 🔴 High

### [🔴] `scratch/` is `.gitignore`d as "kept locally, not tracked" but 65 files are committed (6.4 MB of scratch artifacts)
- **Where:** `scratch/phase2/` (55 files) + `scratch/phase6/` (10 files) = 65 tracked files; ignore rule at `.gitignore:30` (`scratch/`, under header `# Personal scratch/test config — kept locally, not tracked`). Confirmed via `git check-ignore --no-index --verbose` matching every scratch path to `.gitignore:30`. Total ≈ 6.4 MB.
- **Issue:** The `.gitignore` header (`.gitignore:29-30`) declares scratch private/untracked, but the tree was committed before the rule (or `git add -f`'d). Contents are explicitly non-source, self-described as regenerable: `scratch/phase2/README.md:1` "**Not source** — regenerable"; `scratch/phase6/README.md` "**Not source.**". Bulk is binary/generated diagnostics: 18 `.png`, 3 `.gif`, 14 `.jsonl`, 13 `.param`, 15 `.py`, 2 README. Largest: `scratch/phase2/rootmap_cage.gif` (1.6 MB), `scratch/phase2/arms_rootmap_simple1e5.gif` (424 KB), `scratch/phase2/arms_rootmap_mock4e3.gif` (376 KB), `scratch/phase2/cage_compare.png` (220 KB).
- **Impact (git-puller):** A cloner downloads 6.4 MB of one developer's private diagnostic plots/GIFs/run logs that CLAUDE.md itself says are "do not treat as ground truth" (CLAUDE.md, generated/scratch list). The ignored-but-tracked state is confusing: a contributor editing these sees them staged-by-default yet `.gitignore` claims they're local-only, so new scratch files silently won't be tracked while old ones are. It bloats the repo with non-reproducible binaries and blurs what is canonical (the real writeups live in `docs/dev/*.md` and `docs/dev/*.md`, which the scratch READMEs point to).
- **Fix:** Decide intent. If scratch is genuinely meant to be local: `git rm -r --cached scratch/` and commit (keeps files on disk, honors the existing `.gitignore:30` rule). If these diagnostics are meant to be shared, move the canonical CSVs that plot scripts actually read (already at `docs/dev/data/`) and drop the binaries, then remove or scope the `scratch/` ignore rule so the state is self-consistent. Either way the current "ignored but tracked" contradiction should not ship.

---

## 🟠 Medium

### [🟠] Committed binary/generated artifacts under `scratch/` (PNG/GIF/jsonl run dumps)
- **Where:** `scratch/phase2/` — 18 PNG, 3 GIF, 14 jsonl. E.g. `scratch/phase2/rootmap_cage.gif` (1.6 MB), `scratch/phase2/probe_cloud1e6.jsonl` (168 KB), `scratch/phase2/arms_rootmap.png` (184 KB).
- **Issue:** These are generated diagnostic figures and raw `.jsonl` simulation outputs, not source or fixtures used by the test suite (no `test/*` references them; tests use `outputs/mockOutput/` and `docs/dev/data/`). They are the byte-bulk inside the High finding above and are independently bad source-repo hygiene.
- **Impact (git-puller):** Repo carries megabytes of non-reproducible, non-fixture binaries that git cannot meaningfully diff; every clone/fetch pays for them forever in history.
- **Fix:** Remove from tracking together with the `scratch/` cleanup (`git rm -r --cached scratch/`). If any single CSV is genuinely an input to a committed plot script, relocate that CSV to `docs/dev/data/` (where the canonical ones already live per `scratch/phase2/README.md:75`) and drop the rest.

### [🟠] `docs/dev/archive/README.md` is missing the mandatory staleness banner
- **Where:** `docs/dev/archive/README.md:1-9` (no `> ⚠️ This document may be out of date` block under the H1).
- **Issue:** CLAUDE.md mandates every `docs/dev/*.md` and `docs/dev/*.md` doc carry the staleness banner directly under the H1. I checked all 17 such docs (excluding this review's own files): **16 have it, 1 does not** — `docs/dev/archive/README.md`. Its body does carry a bespoke "everything here has shipped / verify against source" caution, so the spirit is met, but the exact mandated banner is absent.
- **Impact (git-puller):** Minor inconsistency with the project's own documented rule; a reader auditing doc hygiene will flag it as the one offender. Low real-world risk because the file already warns it is historical.
- **Fix:** Add the standard banner block under the `# docs/dev/archive` H1 (the README can keep its extra historical-context note below it).

### [🟠] `.gitignore` contains wrapped/garbled comment lines (copy-paste formatting damage)
- **Where:** `.gitignore` — comment lines wrapped mid-sentence so the continuation is no longer a `#` comment, e.g. `:77` `into it.`, `:134` `the code is`, `:140` `Pipfile.lock in version control.`, `:155` a bare `https://python-poetry.org/...` URL, `:216` a bare `https://github.com/github/gitignore/...` URL.
- **Issue:** Long `#` comments from the upstream GitHub Python template were hard-wrapped and the wrapped halves lost their leading `#`, leaving bare non-comment lines in `.gitignore`. They are harmless as patterns (they only match literal files named e.g. `into it.`, which don't exist) but are clearly broken text, not intended ignore rules.
- **Impact (git-puller):** Cosmetic; signals a hand-mangled `.gitignore` and could in principle create a junk ignore pattern. No functional file is wrongly ignored.
- **Fix:** Re-prefix the wrapped continuation lines with `#` (or delete the boilerplate template comments entirely — the active rules at the top are what matter).

---

## 🟡 Low

### [🟡] Open TODO markers in shipped source (19 in `*.py`)
- **Where:** 19 TODO/FIXME/XXX/HACK hits in tracked `*.py` (full table below); 0 FIXME/XXX/HACK — all are `TODO`. 3 of the 19 are test assertions about a deliberate `TODO` line in cloudy output, not leftovers.
- **Issue:** Genuine "not yet implemented / revisit" notes remain in physics modules (cooling, shell structure, init). None are secrets or broken code; they are normal in-progress markers.
- **Impact (git-puller):** A reader sees unfinished-feature notes (e.g. `# TODO: add for non-solar metallicity`); expected for research code, low concern.
- **Fix:** Optional — none required. Convert long-lived ones to issues if desired.

### [🟡] `termination_debug.txt` fixture embeds a wall-clock timestamp
- **Where:** `outputs/mockOutput/mockFullrun/termination_debug.txt:4` — `Timestamp: 2026-04-30 00:30:37`.
- **Issue:** A committed test fixture (by-design under `mockOutput/`) carries a concrete generation timestamp. Not personal info, but it dates the fixture and would change on regeneration.
- **Impact (git-puller):** Negligible; only relevant if the fixture is regenerated and the diff churns on the timestamp.
- **Fix:** Optional — none required; if the fixture is ever regenerated, consider stripping the timestamp line for stable diffs.

---

## TODO/FIXME catalog

All tracked `*.py` (19 hits; no FIXME/XXX/HACK found). The 3 `test_cloudy_cli.py` entries are assertions about an intentional `TODO` line emitted in cloudy decks, not leftover work.

| file:line | tag | text |
|---|---|---|
| `test/test_cloudy_cli.py:306` | TODO | `assert "TODO" in captured.out` (test asserts the deck TODO line) |
| `test/test_cloudy_cli.py:348` | TODO | docstring: "drops NAME into the deck and suppresses the TODO line" |
| `test/test_cloudy_cli.py:360` | TODO | `assert "TODO" not in captured.out` |
| `trinity/_input/input_warnings.py:29` | TODO | `# TODO. E.g., if metalicity is >0, dens profile str is correct, etc.` |
| `trinity/_output/cloudy/trinity_to_cloudy.py:456` | TODO | docstring: "Closing-summary TODO printed only when the SB99 sentinel is in the deck." |
| `trinity/_output/cloudy/trinity_to_cloudy.py:459` | TODO | `f"TODO:  edit {DEFAULT_SB99_SENTINEL} in the \`table star\` line — "` (intentional user-facing reminder) |
| `trinity/cooling/CIE/read_coolingcurve.py:20` | TODO | `# TODO: add for non-solar metallicity` |
| `trinity/cooling/CIE/read_coolingcurve.py:23` | TODO | `# TODO: add file saving for quicker computation time.` |
| `trinity/cooling/net_coolingcurve.py:78` | TODO | `# TODO: this in the future has to depend on the file. It should` |
| `trinity/cooling/non_CIE/read_cloudy.py:61` | TODO | `# TODO: add option to immediately get saved cubes.` |
| `trinity/cooling/non_CIE/read_cloudy.py:254` | TODO | `# Future TODO: If it fails, i.e., if it returns NaN ...` |
| `trinity/main.py:101` | TODO | `# TODO: put this in read_param, and make it depend on param file.` |
| `trinity/main.py:149` | TODO | `# TODO:` (empty) |
| `trinity/main.py:206` | TODO | `# TODO: add loop so that this simulation starts over with old generation of parameter ...` |
| `trinity/phase0_init/get_InitCloudyDens.py:36` | TODO | `# TODO: shouldn't this be +dx_small then?` |
| `trinity/phase0_init/get_InitCloudyDens.py:59` | TODO | `# TODO: make this sound better, and also check if the logr is in correct unit.` |
| `trinity/phase1b_energy_implicit/run_energy_implicit_phase.py:106` | TODO | `# TODO: very fine grid in this phase. Only in transition phase it goes coarse.` |
| `trinity/shell_structure/get_shellODE.py:19` | TODO | `# TODO: add cover fraction cf (f_cover)` |
| `trinity/shell_structure/shell_structure.py:105` | TODO | `# TODO: Add f_cover from fragmentation mechanics` |

(Markdown docs also contain "TODO" in prose — `docs/dev/misc/backward-compat-audit.md:15,96,105`, `trinity/_output/cloudy/README.md:142` — these are documentation about TODOs, not actionable code markers.)

---

## .gitignore-vs-tracked

Cross-referenced every tracked file (`git ls-files`) against the ignore rules with `git check-ignore --no-index`.

| Pattern | Header / intent | Tracked files matching | Verdict |
|---|---|---|---|
| `scratch/` (`.gitignore:30`) | "Personal scratch/test config — kept locally, not tracked" | **65** (`scratch/phase2/` 55, `scratch/phase6/` 10) | 🔴 **Contradiction** — ignored-but-tracked, see High finding |
| `paper/barnes26/` (`.gitignore:19`) | "Private: kept local, not tracked" | 0 | ✅ Clean — not tracked |
| `test/test_barnes_population.py` (`.gitignore:20`) | Private | 0 | ✅ Clean — not tracked |
| `movie.txt` (`.gitignore:21`) | Private | 0 | ✅ Clean — not tracked |
| `outputs/*` w/ `!outputs/mockOutput/` (`.gitignore:13-15`) | mock outputs whitelisted | 19, **all** under `outputs/mockOutput/` | ✅ By-design — whitelist works; fixtures used by `test/test_cloudy_*.py`, `test/test_metadata.py` |
| `param/*` w/ `!param/cloud_*`, `!param/*_example.param`, `!param/simple_cluster.param` (`.gitignore:26-29`) | whitelist worked examples | 7, all whitelisted (verified each negation matches) | ✅ By-design — `sweep_hybrid_example`/`sweep_tuple_example` caught by `!param/*_example.param` |
| `lib/*` w/ `!lib/default/**` (`.gitignore:22-24`) | bundled tables whitelisted | tracked files only under `lib/default/` | ✅ By-design |
| `__pycache__/`, `*.pyc`, `.DS_Store`, `*.swp`, `.ipynb_checkpoints` | junk | 0 | ✅ Clean — no junk tracked |

Only **one** real ignored-but-tracked inconsistency: `scratch/`. The whitelist-based negations (`outputs`, `param`, `lib`) all behave as intended.

---

## Top-level / confusing dirs

Tracked-file counts (`git ls-files`): `scratch/` = 65 (🔴), `outputs/` = 19 (by-design fixtures, all `mockOutput/`), `docs/dev/` = 22 docs+data. **Not tracked / absent** (gitignored or empty, so no clone confusion): `fig/`, `tbd/`, `old_doNotRead/`, `paper/plots/`, `txt/`, `jobs/` — all 0 tracked files. CLAUDE.md lists `outputs/ fig/ scratch/ tbd/ old_doNotRead/` as generated/scratch; of these only `scratch/` actually ships content into a clone, which is exactly the contradiction flagged above.

## Privacy / secrets summary (all clear)

- **Emails:** 1 only — `README.md:99` `<jiaweiteh.astro@gmail.com>` (intentional contact for "available on request" data). `pyproject.toml` author block (`:12-14`) lists name only, no email. ✅ By-design.
- **Names:** all author attribution (`@author: Jia Wei Teh` headers, `docs/source/conf.py:19-20`, `trinity/__init__.py:23`, `trinity/_output/header.py:54` `© J.W. Teh, R.S. Klessen, S.C.O. Glover, K. Kreckel`) or cited-paper authors (Rahner, Teh et al. 2026). ✅ By-design.
- **Secrets/tokens:** none. Every `git grep` hit for `token`/`key`/`secret` is tokenizer/parser code (`sps/sps_columns.py`, `sweep_parser.py`) or dict keys. No API keys, passwords, SSH/PEM material.
- **Machine/HPC paths:** no real personal absolute paths in *source*. `/home/user/trinity/...` appears only inside `docs/dev/*.md` audit docs (not code). HPC references (`bwForCluster Helix`, `bwUniCluster`, `sbatch`) are documented SLURM support with placeholder `--account=YOUR_ACCOUNT` (`trinity/_input/sweep_jobs.py:58-60`, `README.md:70-76`, `run.py:22`) — ✅ by-design, no real account/host leaked.
- **Debug leftovers:** none — 0 `breakpoint()`/`pdb`/`set_trace` in tracked `*.py`.

## Counts: 1 high / 3 medium / 2 low
