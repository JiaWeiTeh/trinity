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

# 01 — trinity/ I/O + infra

**Scope:** `trinity/main.py`, `trinity/__init__.py`, `trinity/_input/` (param
parsing, schema/defaults, sweep expansion), `trinity/_output/` (run output,
readers, terminal/metadata I/O, cloudy export), `trinity/_functions/` (units,
cluster, logging, operations), `trinity/_analysis/`. Audited as a stranger who
`git clone`s the public repo and tries to run a simulation. This is a static,
read-only review; no code was run (the env lacks numpy/scipy). Every claim is
cited to `path:line` against current source. **Good news first:** the
reproducibility-critical paths are sound — `default.param`'s `tools.gen_default_param`
generator exists, every bundled-asset reference (`lib/default/CIE/*`,
`lib/default/opiate/*`, `lib/default/sps/starburst99/1e6cluster_default.csv`)
resolves, the SLURM `sbatch` template uses a clean `YOUR_ACCOUNT` placeholder
(`sweep_jobs.py:59`), and no hardcoded personal/machine paths or secrets exist
in scope (the one `C:\foo` hit is a test-reject fixture in `sweep_parser.py:983`;
the bwForCluster names in `cluster.py` are generic SLURM docs, by-design). The
real issues are dead code, broken doc cross-refs, and stale self-reported counts.

---

### [🔴] `input_warnings.py` is dead code that references a removed parameter schema
- **Where:** `trinity/_input/input_warnings.py:14-74` (whole module)
- **Issue:** The module's `input_warnings()` iterates `trueFalseParams = ['is_mCloud_beforeSF', 'rand_input', 'stochastic_sampling', 'mult_exp', 'frag_enabled', 'frag_grav', 'frag_RTinstab', 'frag_densInhom', 'frag_enable_timescale', 'inc_grav']` and does `if params_dict[pars] not in [0,1]: sys.exit(...)`. None of those keys exist in the current schema (`grep` for them in `registry.py`/`default.param` returns 0 hits). The function is never imported or called anywhere (`grep "input_warnings"` outside the file itself returns nothing). It also carries a large commented-out `params_dict = {...}` block (lines 81-140) full of dead legacy param names (`out_dir`, `log_mCloud`, `imf`, `dens_g_bE`, `mu_n`/`mu_p` as raw floats).
- **Impact (git-puller):** If anyone wires this orphan in (it *looks* like a validation entry point), it `KeyError`s before it can even `sys.exit`, on the very first parameter. As-is it is pure confusion: a reader trying to understand input validation finds a file describing a WARPFIELD-era schema that no longer exists, suggesting the real validation lives somewhere it doesn't.
- **Fix:** Delete the module (validation now lives in `registry.py` validators + `read_param` Steps 3/5). If kept for history, gut the body and add a one-line "superseded by registry validators" note. (Flagging as pre-existing dead code per CLAUDE.md rule 3 — recommend removal, not silently doing it.)

### [🟠] `_output/README.md` references a nonexistent `example_scripts/` directory (also in `trinity_reader.py` docstring)
- **Where:** `trinity/_output/README.md:121-127`; `trinity/_output/trinity_reader.py:120-121`
- **Issue:** README says *"See `example_scripts/` for comprehensive examples: `example_reader_overview.py` … `example_plot_radius_vs_time.py`"*, and `trinity_reader.py`'s module docstring's "See Also" lists the same two files. No `example_scripts/` directory exists anywhere in the repo (`git ls-files` + filesystem both empty; only `paper/methods/figures/` exists, which IS referenced correctly elsewhere).
- **Impact (git-puller):** A new user following the reader docs to find runnable examples hits a dead end — the headline "comprehensive examples" pointer goes nowhere.
- **Fix:** Either add the example scripts, or repoint both references to the real examples under `paper/methods/figures/` (and drop the `example_scripts/` names).

### [🟠] `_output/README.md` documents `Eb` as `[erg]`, contradicting the reader's own unit docs (it's internal units)
- **Where:** `trinity/_output/README.md:25` (`Eb = output.get('Eb')  # Bubble energy [erg]`) and `:69` (`` `Eb`: Bubble thermal energy [erg] ``)
- **Issue:** `output.get('Eb')` returns the stored internal value in `Msun*pc^2/Myr^2`, NOT erg. The reader's own `PARAM_DOCS` is explicit: `trinity_reader.py:163` says *"Eb: Bubble thermal energy [Msun\*pc^2/Myr^2] (× INV_CONV.E_au2cgs → erg)"*, and the registry unit is `Msun*pc**2/Myr**2` (`registry.py:388`). The README's `[erg]` label is wrong by a factor of `INV_CONV.E_au2cgs ≈ 1.9e43`.
- **Impact (git-puller):** Units are explicitly flagged as a recurring bug class in this repo. A user copying the README's "Quick Start" snippet and treating `Eb` as erg is off by 43 orders of magnitude — a silent, plot-ruining error.
- **Fix:** Change both README lines to `[Msun*pc^2/Myr^2] (× INV_CONV.E_au2cgs → erg)` to match `PARAM_DOCS`.

### [🟠] `output_format` is a documented parameter that nothing consumes (output is always JSONL)
- **Where:** `trinity/_input/default.param:31` (`output_format    JSON`); `registry.py:295`
- **Issue:** The spec is declared and defaults to `JSON`, info=*"Specifies the output format."* But no code reads `params['output_format']` (`grep` across `trinity/` + `run.py` finds only the spec definition and a category comment in `param_spec.py:35`). The writer (`dictionary.py:flush`) unconditionally emits `dictionary.jsonl` regardless of this value. The default value `JSON` is itself misleading — the actual file is JSONL, not JSON.
- **Impact (git-puller):** A user who sets `output_format ASCII` (a value the legacy WARPFIELD code accepted — see the commented dead block in `input_warnings.py:84`) gets silently ignored JSONL output with no warning. The knob looks functional but is inert.
- **Fix:** Either implement the format switch, or remove the spec; if kept as reserved, change the info to say it is currently unused and the format is always JSONL.

### [🟡] Stale self-reported spec counts in registry/param_spec/read_param docstrings (186/72/114/187/103 vs actual 195/74)
- **Where:** `registry.py:4-5` ("186 total: 72 declared in `default.param` + 114 runtime/derived"); `registry.py:601` ("Today this adds 103 items"); `param_spec.py:15` ("fully populated (187 specs)"); `read_param.py:465` ("Default 103 adds today")
- **Issue:** Actual counts (verified): `SPECS` has **195** `ParamSpec` entries; **74** are `input_*` category (matching `default.param`'s 74 key lines). Every quoted number is stale: 186→195 total, 72→74 declared, 114→121 runtime, 187→195, and the "103 items materialized" figure is inconsistent with itself across two files and no longer matches.
- **Impact (git-puller):** Cosmetic — a contributor reading these to sanity-check the registry will be misled about its size, but nothing breaks. Symptom of docstrings drifting behind the spec set.
- **Fix:** Update the numbers, or (better) drop the hard counts from prose and let `test_registry.py` pin them.

### [🟡] `header.display()` docstring documents a `params` parameter the function doesn't take
- **Where:** `trinity/_output/header.py:17-29`
- **Issue:** `def display():` takes no arguments, but its docstring has a `Parameters: params : DescribedDict` section. The actual param-summary function is the separate `show_param(params)` (`header.py:85`). The docstring was likely copied/left behind when the two were split.
- **Impact (git-puller):** Minor confusion; a reader might pass `params` to `display()` and get a `TypeError`.
- **Fix:** Remove the Parameters block from `display()`'s docstring.

### [🟡] Termination-debug comparison row keyed on `F_ion`, but the real param is `F_ion_in` — row is always blank
- **Where:** `trinity/_output/simulation_end.py:426` (`('F_ion', 'Ion force', 'code', 1.0)` in `CRITICAL_PARAMS`)
- **Issue:** The registry/reader name is `F_ion_in` (`registry.py:441`, `trinity_reader.py:195`). No parameter named exactly `F_ion` exists. In `write_termination_debug_report`, `snap_old.get('F_ion')` / `snap_new.get('F_ion')` always return `None`, so the "Ion force" comparison row renders as `—` for every run.
- **Impact (git-puller):** Debug-only; the `metadata.json[termination_debug]` comparison table silently drops the ionization-force row. No effect on physics or normal output, but a confusing dead row for anyone inspecting termination diagnostics.
- **Fix:** Change `'F_ion'` to `'F_ion_in'` in `CRITICAL_PARAMS`.

### [🟡] Leftover `# debug` print() statements in `operations.py` fire on any negative-valued array
- **Where:** `trinity/_functions/operations.py:35-36` and `:156-157`
- **Issue:** Both `find_nearest_lower` and `find_nearest_higher` open with `# debug` / `if any(array < 0): print(array)` — an unconditional raw-array dump to stdout whenever an input array contains a negative value. This is leftover debugging cruft (explicitly marked `# debug`), bypassing the logging system the rest of the codebase uses.
- **Impact (git-puller):** On a normal run that passes any array with negatives through these helpers, the user gets unexplained array spew on stdout (not the log file), polluting terminal output.
- **Fix:** Remove the two `# debug` / `print(array)` lines (or route through `logger.debug` if the signal is wanted).

### [🟡] `read_mist_models.py` is an unused MIST-isochrone reader with a hard matplotlib import
- **Where:** `trinity/_functions/read_mist_models.py:1-3` (whole module)
- **Issue:** TRINITY's SPS path is Starburst99 (`sps/`, `lib/default/sps/starburst99/`), not MIST. This module reads MIST `.iso` files and is never imported anywhere (`grep "read_mist_models"/"MIST"/"isochrone"` across `trinity/` + `run.py` returns nothing but the file itself). It also `import matplotlib.pyplot as plt` at module top (line 3) and uses bare `print()` — heavier than its non-existent callers warrant.
- **Impact (git-puller):** Confusion — a newcomer sees a stellar-isochrone reader and wonders whether MIST is a supported SPS backend (it isn't). Pure dead weight.
- **Fix:** Remove the module, or move it to a clearly-labeled `tools/`/`scratch` location if it's a kept utility. (Pre-existing dead code — flag, don't silently delete.)

### [🟡] `SB99_rotation` info comments reference a known-fragile name; cosmetic naming drift (by-design, noted for completeness)
- **Where:** `default.param:135` / `registry.py:320`
- **Issue:** The `SB99_rotation` info text itself documents that the name is SB99-flavored and "May rename to sps_rotation in a future PR." Not a bug — the spec is honest about the drift — but a first-time reader pairing `SB99_rotation` with the more general `sps_path`/`sps_col_*` family may be briefly confused about why one knob is SB99-named and the rest aren't.
- **Impact (git-puller):** None functional; the default fallback correctly rejects `SB99_rotation=0` with a clear message (`registry.py:247-253`). Listed only so it isn't re-flagged as a bug by a future audit.
- **Fix:** None required; the planned `sps_rotation` rename would resolve it.

---

## Counts: 1 high / 3 medium / 6 low
