# Backward-compatibility & stale-code audit

> ⚠️ **This document may be out of date — verify before trusting it.** It is a
> point-in-time analysis/audit, not a maintained spec; the code moves faster
> than these notes (paths, line numbers, and "what shipped" status drift).
> **Any agent or person reading this: treat it as unverified. Flag that it may
> be stale and re-check each claim, snippet, and line reference against the
> current source before relying on it.**
>
> 🔄 **Living plan — recheck and refine on every visit.** This is an evolving
> strategy doc, not a frozen record. Any agent or person who opens this file
> must, as part of the visit: (1) re-verify the claims and line references above
> against current source; (2) update anything that has drifted; (3) **rethink the
> strategy itself** — if a better ordering, gate, candidate, or experiment
> exists, revise the doc and note what changed and why (date it). Leave it better
> than you found it. **Keep all banner paragraphs at the top of every plan and
> analysis doc.**
>
> 💾 **Persist diagnostics — commit, don't re-run.** The container is ephemeral
> and full/hybr runs cost hours, so any diagnostic worth keeping must be saved as
> a committed artifact (a CSV/table under `docs/dev/data/`, or a force-added
> harness/figure in the relevant `docs/dev/<workstream>/` folder as the hybr work did) — never left in `/tmp` or
> an untracked `outputs/`. A future visit must be able to reproduce or compare
> against the numbers **without re-running**; record the exact config + command
> that produced each artifact.

**About this document**
- **Status (verified 2026-06-16):** 🔵 **ACTIONABLE** (verified 2026-06-16) — only the Tier-4 `unit_conversions` relabel shipped; ~95% of the cleanup is still pending. Cited line numbers have drifted.
- **Type:** audit — a tiered inventory of backward-compat shims, dead code, stale comments, and misnamed-but-live paths across the repo, with a suggested cleanup-PR sequence.
- **Workstream:** `misc/` — standalone (backward-compat & stale-code cleanup).
- **Where it sits:** standalone — feeds the future `feature/remove-backward-compat-codeblocks` cleanup branch; audit-only, no code changed (these have no before/after within misc/).
- **Code it concerns:** cross-cutting — `_output/` (reader, cloudy loader, `simulation_end`, `show_run`), `_input/` (registry, read_param, param_spec), bubble-structure (`bubble_luminosity`, `get_bubbleParams`), cooling, plus `test/`, `paper/`, and `docs/`.
- **Linked files & data:** code `trinity/bubble_structure/bubble_luminosity.py`, `trinity/_output/simulation_end.py`, `trinity/_output/cloudy/run_loader.py`, `trinity/_output/trinity_reader.py`, `trinity/_input/registry.py`, `trinity/_functions/unit_conversions.py`; tests `test/test_phase5_text_drop.py`, `test/test_phase4_consumer_migration.py`; related docs `docs/dev/bubble/integrator-robustness.md`.

Audited at commit `b7c5db6` (2026-06-10). Feeds the future cleanup branch
`feature/remove-backward-compat-codeblocks`. **Audit only — no code was changed.**

Method: four parallel sweeps (`_output/`, `_input/`+`tools/`, physics modules,
`test/`+`docs/`+`paper/`) over patterns like *backward/compat/legacy/deprecat/
old/previously/fallback/alias/shim/TODO*, followed by manual caller-graph
verification (grep) of every load-bearing claim. Line numbers below were
re-checked against source at the audit commit.

---

## Tier 1 — Dead now, zero callers (remove any time, LOW risk)

| What | Where | Notes |
|---|---|---|
| `_create_adaptive_radius_grid()` | `trinity/bubble_structure/bubble_luminosity.py:865-989` | Explicitly disabled (see note at `:477-491`); no callers anywhere. ~125 lines. |
| `_solve_bubble_ode_with_ivp()` | `trinity/bubble_structure/bubble_luminosity.py:991-1040` | "Kept for future experimentation"; no callers. Referenced only by `docs/dev/bubble/integrator-robustness.md` (a stale-by-design doc). |
| `get_beta_delta_wrapper_pure()` | `trinity/phase1b_energy_implicit/get_betadelta.py:781` | Docstring says it "matches the interface of the original `get_beta_delta_wrapper`" — **that original no longer exists**. Zero callers. |
| Commented-out "depreciated" cooling blocks | `trinity/cooling/net_coolingcurve.py:70-75, 107-110, 136-139` | Old non-CIE netcooling-grid path, commented out (note misspelling "depreciated"). |
| `plot_folder_grid = plot_grid` alias | `paper/methods/figures/paper_feedback.py:572` | Zero callers. |
| `cpr.SAVE` alias | `trinity/_output/terminal_prints.py:77` | `SAVE = BOLD`; zero callers (`FILE` is the name actually used — `header.py:103`, `get_InitCloudyDens.py:60`). Keep `FILE`. |
| `ParameterFileError` re-export | `trinity/_input/read_param.py:502-505` (`__all__`) and the back-compat note in `trinity/_input/errors.py:6` | Verified: **no internal code** imports it from `read_param` — everything imports from `trinity._input.errors`. Only out-of-repo user scripts could break. |

## Tier 2 — The Phase-6 removal set (coordinated: shims + tests + docs together)

The metadata migration (Phases 1–5) left text-parse fallbacks that the code
itself promises to remove "in Phase 6". This is the single biggest cluster and
should be one PR, removing each shim together with the tests that pin it:

**Production shims (emit `DeprecationWarning`):**
- `trinity/_output/cloudy/run_loader.py:154-185` — `_parse_summary_txt()` (legacy `<model>_summary.txt`).
- `trinity/_output/cloudy/run_loader.py:188-282` — `_parse_simulation_end()` incl. the legacy `Status`/`End Reason`/`Raw Reason` key mapping (`:227-280`) and the fallback call sites at `:114` and `:138`.
- `trinity/_output/simulation_end.py:309-401` — `read_simulation_end()` text-parse fallback branch (~`:348-400`).
- `trinity/_output/show_run.py:246-266` — last-resort text-parse leg of `_resolve_run_status()`.

**Tests that exist only to pin the above (delete with the shims):**
- `test/test_phase5_text_drop.py` — entire file (pins "text files not written" + the deprecation warnings).
- `test/test_phase4_consumer_migration.py` — entire file (pins the one-time consumer migration; legacy-text test at `:108-121`).
- `test/test_cloudy_run_loader.py:224-235` (`test_parse_simulation_end_legacy_back_compat`) and the related warn/fallback tests (`:135-152`, `:287-296`).
- `test/test_show_run.py:85-120` (`_write_legacy_v1_run` fixture) + `:176-186` (`test_legacy_v1_run_falls_back_to_text`).
- `test/test_metadata.py` — `TestLegacy` class (`:283-361`), `test_legacy_v1_inline_arrays_used` (`:523-587`), `test_falls_back_to_text_for_legacy_runs` (`:799-825`), legacy-array fixtures (`:33-36`, `:48-62`).

**Consumer to simplify:** `paper/methods/figures/paper_rcloud_smoothing.py:184` —
`output.termination or read_simulation_end(...)`; after Phase 6 the call stays
valid (it reads metadata.json) but the legacy-fallback comment at `:176-177`
goes stale.

**Decide separately:** v1-inline-array support in
`trinity/_output/trinity_reader.py:574-642` (`initial_cloud_profile()` dual
path) — this serves old *data on disk*, not old code. Keep it if pre-v2 run
directories still matter to you; otherwise it joins this PR.

**Caveat before merging this tier:** any run directory produced before Phase 5
becomes unreadable for termination info. If old published datasets must stay
loadable, re-run them or keep `read_simulation_end`'s fallback only.

## Tier 3 — Live compat aliases (removal = mechanical rename of callers first)

- **`load_output = read`** — `trinity/_output/trinity_reader.py:1096`. Verified callers: `trinity/_analysis/check_yesno.py:51,91`, `paper/barnes26/_barnes_lib.py:38,152`, `paper/methods/figures/paper_feedback.py:28,130`, `paper_radiusComparison.py:29,84`, `paper_teaser.py:49,147`, `paper_densityProfile.py:40,310`, `paper_rcloud_smoothing.py:44,158`, `test/test_barnes_population.py` (5 uses), plus `trinity/_output/README.md:54-57`, docstring at `trinity_reader.py:1366`, and `docs/source/trinity_reader.rst:40,201,213`. ~10 files to touch; trivially scriptable.
- **`SB99_rotation` parameter** — `trinity/_input/registry.py:306` carries "name retained for stability. May rename to sps_rotation in a future PR". A rename is a user-facing `.param` break — needs the `deprecated` category machinery (below) or a release note. Out of scope for a pure cleanup PR.

## Tier 4 — The label is wrong, not the code (fix comments/names, keep behavior)

These are the "wrongly documented" items the cleanup should *relabel*, because
future sessions keep misreading them as removable:

- **✅ FIXED on `bugfix/stale-audit`** — **`unit_conversions.py:213-264` "Backward compatibility aliases (deprecated…)"** — the comment was false. `import trinity._functions.unit_conversions as cvt` is the standard pattern in ~20 production modules, so `cvt.pc2cm`, `cvt.cm2pc`, `cvt.s2Myr`, … resolve through *exactly these module-level names*; they are the primary interface, not a deprecated one. Also, `Pb_au2_KcmInv` (`:259`) and `Mdot_au2Msunyr` (`:261`) inside the block are **original definitions, not aliases** — a blanket delete breaks `simulation_end.py`, `show_run.py`, `_barnes_lib.py`, `paper_PradSources.py`. Action: rewrite the section header to say these are the canonical module-level API (or, if you ever want them gone, that's a repo-wide `cvt.X` → `cvt.CONV.X` refactor — separate effort, HIGH cost).
- **`_bubble_luminosity_legacy()` / `_create_legacy_radius_grid()`** — `bubble_luminosity.py:494`, `:827`. Despite "legacy" in the name, this **is the unconditional production path** (called at `:490`; grid also used by `tools/bubble_audit/audit.py:85` and mirrored in `tools/bubble_conduction_convergence.py`). The planned solve_ivp primary path was never landed (its scaffolding is the Tier-1 dead code). Action: either land the solve_ivp path or rename these to drop "legacy".
- **"old code:" name-archaeology comments** — e.g. `get_bubbleParams.py:69` (`beta_to_Edot()`), `:140` (`Edot_to_beta()`), `:420` (`R1_zero()`); `net_coolingcurve.py:10-11` (`coolnoeq.cool_interp_master()`); `non_CIE/read_cloudy.py:10-11,93`; `get_InitCloudyDens.py:31` (`__cloudy__.create_dlaw()`); `get_betadelta.py:68-69,140-141`. These reference WARPFIELD-era names that no longer exist anywhere. Pure comment deletions, zero risk.
- **NumPy shim** — `bubble_luminosity.py:39` `_trapezoid = getattr(np, 'trapezoid', None) or np.trapz`. NumPy is pinned `<2` (`pyproject.toml:38`, `requirements.txt:10`), so the `trapezoid` branch can never fire today; it's a *forward*-compat shim for a version the project deliberately excludes. Harmless — keep it (it becomes load-bearing the day the pin is lifted), but the situation deserves a one-line comment pointing at the pin.

## Tier 5 — Looks like compat cruft, is not (keep; listed to stop future "cleanup")

- `trinity/_input/param_spec.py:60-61,140,147-150` + `tools/_param_text.py` `# DEPRECATED:` parser + `tools/gen_default_param.py:74,80` + `test/test_registry.py:158-160` — the `deprecated` **parameter category framework**. Currently zero specs use it, but it is exactly the machinery a future `SB99_rotation` → `sps_rotation` rename needs. Keep.
- `trinity/_functions/operations.py:76-94` — tolerant-monotonicity guard explicitly marked "RETAINED FALLBACK … do not remove as dead code". Keep.
- `trinity/_input/dictionary.py:175-177` `__array__` — NumPy *interop* protocol, not back-compat.
- Defensive `.get()` defaults in `sweep_runner.py` / `sweep_parser.py` — ordinary optional-key handling, not legacy shims.
- Phase-5 migration *narrative* docstrings (`simulation_end.py:24-37,152-157`, `run_constants.py:35-87`, `read_param.py:494-497`, `dictionary.py:332,753,871`) — currently accurate history. They only become removable when Tier 2 lands (then sweep them in the same PR).

## Docs to update when Tier 2/3 land

- `docs/source/trinity_reader.rst:12,40,201-213,233-236` — `load_output` usage and the "legacy `.json` (pre-2026)" format note (the `.json` fallback at `trinity_reader.py:1379` is still real today).
- `docs/source/running.rst:303-304` — legacy `.json` mention.
- `trinity/_output/README.md:54-57` — `load_output` example.
- Note: `docs/source/conf.py:59,109` says the whole Sphinx site is a frozen, deprecated mirror — decide whether stale-doc fixes there are worth it at all.

## Leftover TODO inventory (judgment calls, not compat)

Genuine open questions, listed so the cleanup doesn't blindly delete them:
`net_coolingcurve.py:78-83` (cooling-floor temperature mystery),
`get_InitCloudyDens.py:36` ("shouldn't this be +dx_small?"),
`read_cloudy.py:61`, `read_coolingcurve.py:20,23` (caching/metallicity),
`get_shellODE.py:19` + `shell_structure.py:105` (cover fraction),
`run_energy_implicit_phase.py:106` (grid resolution note),
`bubble_luminosity.py:482-484` (re-enabling adaptive grid — moot if Tier 1
deletes the grid; delete the TODO with it).

## Suggested PR sequence for `feature/remove-backward-compat-codeblocks`

1. **PR 1 (zero risk):** Tier 1 deletions + Tier 4 comment fixes/renames. ~400 lines gone, no behavior change; full test suite must stay green.
2. **PR 2 (`load_output` retirement):** rename call sites in `trinity/_analysis`, `paper/`, `test/`, docs; then delete the alias.
3. **PR 3 (Phase 6):** Tier 2 — text-parse fallbacks + their pinning tests + stale migration narratives, after deciding the old-run-directory question above.
4. **Not in this branch:** `cvt` module-alias refactor and `SB99_rotation` rename — both are API-shaping decisions, not cleanup.
