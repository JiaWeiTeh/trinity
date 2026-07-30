# S13a output core — Lens B (what the code claims)

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
> a committed artifact under `docs/dev/` (a CSV/table in `docs/dev/data/`, or a
> harness/figure in the relevant `docs/dev/<workstream>/` folder) — never left in
> `/tmp`, the local-only `scratch/`, or an untracked `outputs/`. A future visit must be able to reproduce or compare
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

**Status (2026-07-29):** 📘 raw agent report — provenance for `FINDINGS.md`; unreconciled and unverified on its own.

This is a **prose-only transcription**. My entire input was
`scratchpad/lens/S13a_output_core/prose.md`, which contains every comment and docstring
extracted from the slice, tagged with source file and line range. **I have seen no code** —
no function bodies, no signatures, no constants, no literals. Every statement below is a
*claim made by the prose*, never an assertion about behaviour.

Slice files (8):

- `trinity/_output/trinity_reader.py`
- `trinity/_output/show_run.py`
- `trinity/_output/simulation_end.py`
- `trinity/_output/terminal_prints.py`
- `trinity/_output/header.py`
- `trinity/_output/run_constants.py`
- `trinity/_output/_metadata_io.py`
- `trinity/_output/__init__.py`

**Extraction caveat that shapes everything below.** `trinity_reader.py:136-278` is prose-tagged
only as a run of section-header comments — `# Model info`, `# Simulation state`,
`# Termination flags`, `# Main dynamical variables`, `# Cooling parameters`,
`# Feedback luminosities`, `# Momentum injection`, `# Forces`, `# Shell properties`,
`# Cloud profile parameters (constants, saved for radial profile reconstruction)`,
`# Initial cloud arrays`, `# Shell absorption`, `# Bubble luminosities`, `# Bubble structure`,
`# Bubble arrays (radial profiles)`, `# Residuals (beta-delta solver diagnostics)`,
`# Gravitational potential arrays`, `# Cooling update`. The block is introduced by the banner
`# Parameter Documentation` (`trinity_reader.py:136`) and `_print_parameters` is documented as
"Print all parameters with documentation" (`trinity_reader.py:981`). So the authoritative
per-key documentation table almost certainly lives in a **code literal**, which by construction
is invisible to this lens. The schema table below is therefore what the *narrative prose* claims,
not the full documented schema. Lens A will see far more keys than I can. Treat every
"not stated" cell as "not stated *in prose*", not "undocumented in the file".

---

## 1. The output contract as documented

### 1.1 Artifacts a run directory is claimed to contain

`run_constants.py:3` (v4 entry of the version history):

> "Output directory shrinks from 7 files to 4 (``.param`` + ``trinity_*.log`` +
> ``dictionary.jsonl`` + ``metadata.json``)."

`simulation_end.py:3` claims the three removed artefacts were `simulationEnd.txt`,
`termination_debug.txt`, and `<run>_summary.txt` (7 − 3 = 4; the arithmetic is internally
consistent). `run_constants.py:94` calls `metadata.json` "the per-run metadata sidecar, sibling to
``dictionary.jsonl`` in each run output directory".

Legacy artefacts the prose still claims to read: `dictionary.json` (old single-object format,
`trinity_reader.py:395`), `simulationEnd.txt` (`simulation_end.py:311`, `show_run.py:3`),
inline `initial_cloud_*_arr` arrays in v1 metadata (`trinity_reader.py:574`).

### 1.2 `metadata.json` top level

| Key | Claimed content | Claimed schema version | Rehydrated into snapshots? | Cited at |
|---|---|---|---|---|
| `_metadata_version` | "Schema version of ``metadata.json``. Increment whenever the layout changes in a backwards-incompatible way." | v1+ | **No** — reserved; "consumed and discarded by the reader before rehydrate" | `run_constants.py:98`, `run_constants.py:3` |
| *run-constant scalars* | "input parameters or set-once derived values that do not change after phase 0"; "written exactly once per run"; "stripped from every per-snapshot dictionary in ``dictionary.jsonl``" | v1 (13 keys) → v2 ("~57 scalars/strings/bools") | **Yes** — "via ``setdefault`` (so any per-snapshot value, when present, takes precedence)" | `run_constants.py:3` |
| `termination` | `{exit_code, outcome, detail, timestamp, model_name}` | v3 (Phase 2) | **No** — reserved | `simulation_end.py:3`, `trinity_reader.py:480` |
| `final_state` | "every scalar/bool/string on ``params`` at run end, in INTERNAL units (pc/Myr, pc⁻³, …)… arrays excluded" | v3 (Phase 2) | **No** — reserved ("Rehydrating them into each snapshot would smear the run-end state into every timestep, which is misleading") | `simulation_end.py:3`, `run_constants.py:102` |
| `termination_debug` | keys `timestamp`, `reason`, `snapshot_count`, `time`, `comparison`, `warnings`, `invalid_values`, `sanity_checks` | v4 (Phase 5) | **No** — reserved | `trinity_reader.py:511`, `run_constants.py:102` |
| `initial_cloud_*_arr` (r, n, m) | v1-only inline arrays; "dropped" in v2, "readers reconstruct on demand" | v1 only | n/a | `run_constants.py:85`, `trinity_reader.py:215` |

Three writers are claimed to touch the file (`_metadata_io.py:3`): `DescribedDict.flush()`
(run-constants, "typically at run start"), `write_simulation_end()` (`termination` +
`final_state`), `write_termination_debug_report()` (`termination_debug`, "at emergency-flush
time"). All three are claimed to go "through the shared atomic helper in
:mod:`trinity._output._metadata_io`, so a partial write can never leave a corrupt file"
(`simulation_end.py:3`) — a *never* guarantee.

### 1.3 Per-snapshot keys named in prose (`dictionary.jsonl`)

Only `trinity_reader.py:3` ("Key Parameters") names individual keys with units. Everything else
in this table comes from scattered comments/usage examples.

| Key | Claimed meaning | Claimed unit | Claimed dtype/shape | Group / cited at |
|---|---|---|---|---|
| `t_now` | "Current simulation time" | **Myr** | scalar (numeric) | Dynamical, `trinity_reader.py:3` |
| `R2` | "Outer bubble radius (= inner shell edge)" | **pc** | scalar | Dynamical, `trinity_reader.py:3` |
| `v2` | "Velocity at R2 (outer bubble / inner shell edge)" | **pc/Myr** (internal); displayed in km/s | scalar | Dynamical, `trinity_reader.py:3`; `show_run.py:181` |
| `Eb` | "Bubble thermal energy" | **Msun·pc²/Myr²** internal; "× INV_CONV.E_au2cgs → erg" | scalar | Dynamical, `trinity_reader.py:3`; `simulation_end.py:418` |
| `T0` | "Characteristic bubble temperature" | **K** | scalar | Dynamical, `trinity_reader.py:3` |
| `R1` | "Inner bubble radius (wind termination shock)" | **pc** | scalar | Dynamical, `trinity_reader.py:3` |
| `Pb` | "Bubble pressure" | **Msun/pc/Myr²** (internal); displayed as P/k_B in K cm⁻³ | scalar | Dynamical, `trinity_reader.py:3`; `show_run.py:197` |
| `cool_beta` | "Pressure evolution parameter β = -(t/Pb)(dPb/dt)" | dimensionless (implied) | scalar; may be NaN in momentum phase | Cooling, `trinity_reader.py:3`; `simulation_end.py:244` |
| `cool_delta` | "Temperature evolution parameter δ" | **not stated** (no formula given) | scalar | Cooling, `trinity_reader.py:3` |
| `F_grav` | "Gravitational force" | **not stated** | scalar | Forces, `trinity_reader.py:3` |
| `F_ram` | "Ram pressure force (total)" | **not stated** | scalar | Forces, `trinity_reader.py:3` |
| `F_HII` | "HII pressure force (outward)" | **not stated** | scalar | Forces, `trinity_reader.py:3` |
| `F_rad` | "Radiation pressure force" | **not stated** | scalar | Forces, `trinity_reader.py:3` |
| `residual_Edot1_guess` | "Edot from beta" | **Msun·pc²/Myr³** internal; "× INV_CONV.L_au2cgs → erg/s" | scalar | Residuals, `trinity_reader.py:3` |
| `residual_Edot2_guess` | "Edot from energy balance" | same as above | scalar | Residuals, `trinity_reader.py:3` |
| `residual_T1_guess` | "Bubble temperature T_bubble" | **K** | scalar | Residuals, `trinity_reader.py:3` |
| `residual_T2_guess` | "Target temperature T0" | **K** | scalar | Residuals, `trinity_reader.py:3` |
| `current_phase` | phase label | n/a | **non-numeric** ("For non-numeric data, disable array conversion") | `trinity_reader.py:3` |
| `snap_id` | snapshot ordinal | n/a | numeric; "Sort by snap_id"; "-1 indicates interpolated" | `trinity_reader.py:410`, `:908` |
| `is_interpolated` | set on synthesised snapshots | n/a | bool ("Interpolated snapshot with is_interpolated=True") | `trinity_reader.py:746` |
| `model_name` | "Model name from first snapshot" | n/a | str | `trinity_reader.py:645` |
| `shell_nMax` | shell max density | internal **pc⁻³**; displayed cm⁻³ | scalar | `show_run.py:191`, `simulation_end.py:131` |
| `shell_mass` | shell mass | **Msun** | scalar | `show_run.py:187` |
| `bubble_Tavg` | bubble mean temperature | "(internal units)" | scalar | `show_run.py:209` |
| `isCollapse` | "the shell was *contracting* (``v2 < 0`` and ``R2`` falling) at exit — NOT that it reached the collapse radius ``coll_r``" | bool | scalar | `show_run.py:66`; `simulation_end.py:699` |
| `isDissolved` | dissolution flag (meaning never documented) | n/a | scalar | `show_run.py:221`, `run_constants.py:3` |
| `coll_r` | collapse radius | pc (implied) | scalar | `show_run.py:66` |

Whole groups are named but no member key of them appears in prose: **Feedback luminosities**
(`:173`, "all luminosities below are stored in code units [Msun*pc^2/Myr^3]. Multiply by
INV_CONV.L_au2cgs to obtain erg/s"), **Momentum injection** (`:184`), **Shell absorption**
(`:224`), **Bubble luminosities** (`:230`, same unit note repeated verbatim), **Bubble structure**
(`:241`), **Bubble arrays (radial profiles)** (`:249`), **Gravitational potential arrays**
(`:267`), **Cooling update** (`:272`), **Termination flags** (`:152`), **Model info** (`:141`),
**Simulation state** (`:146`).

### 1.4 Run-constant scalars named in prose

`mCloud`, `nCore`, `nISM`, `rCore`, `rCloud`, `dens_profile` (string-valued —
`show_run.py:136`: "string rows (dens_profile) keep conv=1.0"), `densPL_alpha`, `mu_convert`,
`densBE_Omega`, `gamma_adia` ("for BE"), `mu_atom`, `sfe` (`show_run.py:131`: "mCluster is derived
from mCloud * sfe"). `nCore`/`nISM` claimed "stored internally in pc⁻³; show them in cm⁻³ (the
input unit) via ndens_au2cgs" (`show_run.py:136`). Cited at `trinity_reader.py:215`,
`trinity_reader.py:574`, `trinity_reader.py:549`, `show_run.py:129`, `simulation_end.py:131`.

`run_constants.py:3` names keys that are **deliberately not** run-constants: "``EndSimulationDirectly``,
``isCollapse``, ``isDissolved``, ``is_phiDepleted``, ``bubble_dMdtGuess``, ``t_next``,
``shell_interpolate_massDot``, ``F_ISM``, …" — "they represent runtime state that varies across
runs even when constant within one run." `run_constants.py:67` names four keys claimed **removed**
by the registry-derivation: "``expansionBeyondCloud`` in run-consts; ``SB99_data`` / ``SB99f`` /
``path_sps`` in the exclude set… they have no spec, so the derivation cannot emit them."

`METADATA_EXCLUDE` is claimed to hold "absolute paths, loaded function tables/interpolators, and
empty-array placeholders whose real data lives in the per-snapshot stream. The writer also skips
them defensively." (`run_constants.py:79`). `path2output` is separately excluded from
`final_state` as "the absolute path of the run dir itself; redundant and a privacy concern"
(`simulation_end.py:272`).

### 1.5 `final_state` — three non-identical descriptions of one block

| Source | Claimed contents |
|---|---|
| `simulation_end.py:3` (module) | "every scalar/bool/string on ``params`` at run end… arrays excluded (their last values live in the dictionary.jsonl tail)" |
| `simulation_end.py:131` (`write_simulation_end`) | "every non-array non-run-constant scalar from ``params`` at run end" |
| `simulation_end.py:244` (`_build_final_state_block`) | "every scalar/string/bool key on ``params`` EXCEPT: run-constants; keys in ``METADATA_EXCLUDE``; long arrays listed in ``FINAL_STATE_EXCLUDE_ARRAYS``; the ``SimulationEndCode`` proxy" — plus `SimulationEndReason` and `path2output` per `simulation_end.py:272` |
| `run_constants.py:120` | "Anything scalar/string/bool flows through to ``final_state``" |

All four agree on **units**: internal (pc/Myr, pc⁻³), "same convention as ``Snapshot.get(key)``"
(`simulation_end.py:3`, `:131`, `:244`; `trinity_reader.py:496`). This is the most consistently
repeated invariant in the slice. See finding **08** for the exclusion-list drift.

---

## 2. Claimed write → read round-trip

| Writer claim | Reader claim | Agreement |
|---|---|---|
| `write_simulation_end` writes `{exit_code, outcome, detail, timestamp, model_name}` (`simulation_end.py:131`) | `TrinityOutput.termination` returns the same five keys (`trinity_reader.py:480`) | ✅ agree |
| same | `read_simulation_end` returns "``exit_code``, ``outcome``, ``detail``, ``timestamp``, ``model``" (`simulation_end.py:311`) | ❌ **disagree on the fifth key** — finding **01** |
| run-constants written once, stripped from snapshots (`run_constants.py:3`) | `_rehydrate_metadata` merges them back "via ``setdefault`` semantics (per-snapshot value wins when both are present)" (`trinity_reader.py:429`) | ✅ agree; `trinity_reader.py:549` adds "both return the same value thanks to the rehydrate step" |
| reserved top-level keys not rehydrated: `_metadata_version`, `termination`, `final_state`, `termination_debug` (`run_constants.py:102`) | `metadata_keys_to_rehydrate` docstring lists only three (`run_constants.py:136`) | ❌ finding **06** |
| aborted run ⇒ minimal file with only `_metadata_version` + blocks; "readers will then return ``None`` for run-constants" (`_metadata_io.py:97`) | `metadata` property "Returns ``{}`` if the file is absent or malformed" (`trinity_reader.py:549`); `initial_cloud_profile` "Raises KeyError" (`trinity_reader.py:574`) | ❌ finding **05** |
| unserializable values "logged at WARNING and silently dropped" (`_metadata_io.py:97`); "only include keys whose final value is JSON-friendly" (`simulation_end.py:301`) | no reader-side claim that a key may be missing for this reason | ❌ finding **10** |
| NaN/Inf written as bare `NaN` tokens (`simulation_end.py:244`, `:680`) | reader claims `.jsonl` / JSON Lines compatibility (`trinity_reader.py:3`) | ⚠ finding **11** |
| `final_state` in internal units | `show_run` "re-applies km/s and cm⁻³ conversions for human reading" (`trinity_reader.py:496`, `show_run.py:162`) | ✅ agree at docstring level; ❌ the comments list two further conversions — finding **20** |

---

## 3. Termination / exit-code documentation

### 3.1 Ranges (`simulation_end.py:56`, `SimulationEndCode`)

> "Exit code ranges: - 0-9: Clean physical or intentional terminations (auto-trust) - 10-19:
> Parameter/configuration errors - 20-29: Numerical/runtime errors - 50-59: Inspection required
> (completed, but warrants a human look) - 99: Unknown/unhandled termination (fallback safety net)"

> "Each member carries (code, outcome_token). The outcome token is mirrored into
> ``metadata.json[termination].outcome``."

Predicates: `is_clean` — "True if the run finished with a clean physical/intentional outcome
(0-9)" (`:110`); `is_error` — "True if the run failed with a parameter or numerical error (10-29)"
(`:114`); `is_inspection_required` — "True if the run completed but warrants a human look (50-59
or 99)" (`:118`); `from_code` — "Look up the enum member by numeric code, or UNKNOWN if no match"
(`:123`). Group comments confirm the same four bands plus `# Unknown — also treated as
inspection-required` (`:92`).

**Ranges 30-49 and 60-98 are named nowhere.** No individual enum member, code value, or reason
string appears in prose (finding **15**).

### 3.2 The only named outcome token

`shell_collapsed` — `show_run.py:66`: "``\"yes\"`` is keyed strictly on the ``shell_collapsed``
outcome — the one unambiguous signal that the collapse-radius event actually fired";
`show_run.py:97`: "A terminal collapse shows up in ``outcome`` itself ('shell_collapsed' → state
\"yes\"), so it is never repeated."

Fates referred to only in prose, never as tokens: "stopping time reached", "dissolution",
"a mid-run numerical crash", "an early dissolution" (`show_run.py:66`, `:97`).

### 3.3 Provenance of the code

`simulation_end.py:131`: "The exit code and outcome category are read directly from
``params['SimulationEndCode']`` (set at the source by the site that decided to terminate); the
verbatim ``SimulationEndReason`` message becomes ``termination.detail``."
`simulation_end.py:189`: "End code is set at the source (phase runners, main.py, phase_events) as
the integer .code so it survives JSON serialization. If a site forgot to set it, fall back to
UNKNOWN (an inspection-required state)."
`simulation_end.py:232`: "The exit code is the contract of this function; the metadata write
failing should not bring the run down."
`terminal_prints.py:206`: `format_end_report` "Reads the numeric SimulationEndCode + verbatim
SimulationEndReason that the phase runners set, so the actual fate (why the bubble stopped) is
visible in trinity.log rather than only in metadata.json."

### 3.4 Reader-side success flag

`trinity_reader.py:3`: "output.is_successful_run # True | False | None — three-valued / True iff
exit_code in [0, 9]".
`trinity_reader.py:526`: "* ``True`` — exit code in [0, 9] (clean termination per
``SimulationEndCode.is_clean()``); * ``False`` — exit code outside that range; * ``None`` — no
``termination`` block (legacy run, crash before ``write_simulation_end`` fired)."
See findings **02** and **03**.

### 3.5 `show_run` process exit codes (`show_run.py:3`)

> "* 0 — pretty-print succeeded (or, with ``--quiet``, the run is successful per
> ``output.is_successful_run``). * 1 — run directory not found, or both ``metadata.json`` and
> ``simulationEnd.txt`` missing. * 1..9 — with ``--quiet``, the run's own ``exit_code`` from the
> termination block (floored at 1, capped at 9 so it fits in POSIX)."

`show_run.py:484`: "Failure path: propagate the run's exit_code if we have one, capped to [1, 9]
so the value fits in POSIX." See findings **13** and **14**.

### 3.6 Three-state collapse descriptor (`show_run.py:66`)

> "``\"no\"`` — not contracting at exit / ``\"collapsing\"`` — contracting at exit, but the run
> ended for some other reason (stopping time, dissolution, error, …) / ``\"yes\"`` — terminal
> collapse: the run stopped *because* the shell reached ``coll_r``"

Negative claim: "Using a bare ``R2 <= coll_r`` test instead would mislabel runs that merely
happened to be small-and-contracting when they ended for an unrelated reason". Returns `None`
"when ``isCollapse`` is absent (legacy runs)".

### 3.7 Termination-debug thresholds (`simulation_end.py:436`)

Documented defaults, values mostly not given in prose: "50% change flagged by default"; "Time can
jump (don't flag)"; "Phase changes are **always** flagged if different"; "Collapse status changes
**always** flagged"; "Velocity can change sign, be more lenient"; "Energy can change rapidly";
"Pressure can change rapidly". Tracked-parameter groups: time and radii, velocities
("pc/Myr -> km/s"), energies ("Msun*pc^2/Myr^2 -> erg"), "P/k_B", shell properties, forces,
temperatures, phase (`:410`–`:431`).

Sanity checks are documented as **phase-aware** (`simulation_end.py:699`): "In the momentum phase
the bubble energy is zeroed (``Eb=0``) and the inner discontinuity is collapsed onto the shell
(``R1=R2``) by design, so the ``Eb > 0`` and ``R1 < R2`` checks are skipped there instead of
reported as spurious failures. Shell collapse is a physical outcome… not a health check, so it is
not asserted here." This is corroborated by `terminal_prints.py:144` ("Eb=0 (momentum phase) is
finite and renders normally") — a rare cross-file agreement on a physical regime.

---

## 4. Documented defaults, invariants, and absolute statements

| Claim | Cited at |
|---|---|
| "They are written **exactly once** per run, in ``<run_dir>/metadata.json``, and stripped from **every** per-snapshot dictionary" | `run_constants.py:3` |
| "a partial write can **never** leave a corrupt file" (atomic temp-file + rename) | `simulation_end.py:3`, `_metadata_io.py:79` |
| "if the process dies mid-write, the existing file (if any) survives" | `_metadata_io.py:79` |
| "The two encoders **MUST** stay in sync." | `_metadata_io.py:41` |
| "Output is pretty-printed (``indent=2``)… keys are emitted in insertion order"; "Caller is responsible for ensuring ``run_dir`` exists." | `_metadata_io.py:79` |
| "Merge ``termination_debug`` into metadata.json; **never** raise." | `simulation_end.py:737` |
| "Importing this module **must not** pull in TRINITY's runtime container." | `_metadata_io.py:3` |
| "snapshots are saved BEFORE ODE integration, **ensuring** all values in a snapshot correspond to the same timestamp (t_now)" | `trinity_reader.py:3` |
| "per-snapshot value wins when both are present" (setdefault rehydrate) | `trinity_reader.py:429` |
| "A terminal collapse shows up in ``outcome`` itself… so it is **never** repeated." | `show_run.py:97` |
| "it logs only every ``HEARTBEAT_EVERY``-th segment, so it **never** floods the log" | `terminal_prints.py:188` |
| default `get_at_time(mode='interpolate')`, `n_neighbors=5` ("uses 2-3 on each side") | `trinity_reader.py:695` |
| `info(verbose=False)` default; `iter_progress(label="Processing")` default | `trinity_reader.py:950`, `:1099` |
| `resolve_data_input(output_dir=...)` "Defaults to 'outputs' or TRINITY_OUTPUT_DIR env var" | `trinity_reader.py:1260` |
| file-search precedence: "_modified folder (new runs)" before "{run_name} folder"; "dictionary.jsonl > dictionary.json" | `trinity_reader.py:1134` |
| `find_all_simulations` "Returns empty list if base_dir doesn't exist or contains no simulations" | `trinity_reader.py:1344` |
| `read_metadata` / `metadata` return `{}` on absent-or-malformed; "Callers that need to distinguish absent-vs-corrupt should check existence themselves." | `_metadata_io.py:60`, `trinity_reader.py:549` |
| `write_termination_debug_report` "Returns ``None`` for backwards compatibility with the old callers (they only logged the return; the path was never consumed programmatically)." | `simulation_end.py:559` |
| `format_run_summary` "Pure function — no I/O side effects beyond the file reads." | `show_run.py:372` |
| `_phys` "Returns 'n/a' if the key is absent or its value is None/non-numeric, and the literal 'nan'/'inf' if the value is non-finite." | `terminal_prints.py:144` |
| `to_dataframe` "scalar values only" | `trinity_reader.py:1036` |

---

## 5. Citations and external-format references

- **JSON Lines / `.jsonl`** — the advertised output format (`trinity_reader.py:3`, `:417`), with
  a self-declared violation: "technically non-standard JSON" NaN tokens (`simulation_end.py:244`).
- **`astropy.io.fits`** — API analogy: "Similar to astropy.io.fits, provides easy access to
  simulation data with a clean, Pythonic API" (`trinity_reader.py:3`).
- **pandas** `DataFrame` (`trinity_reader.py:3`, `:1036`); **numpy** arrays throughout.
- **`paper/methods/figures/paper_*.py`** — "worked examples that use TrinityOutput"
  (`trinity_reader.py:3`). Only external-path citation in the slice; unverifiable from prose.
- **`INV_CONV.E_au2cgs`, `INV_CONV.L_au2cgs`, `ndens_au2cgs`** (`trinity_reader.py:3`, `:174`,
  `:231`; `show_run.py:136`).
- **`trinity._input.registry` ParamSpec** — "the single source of truth" for `RUN_CONST_KEYS` /
  `METADATA_EXCLUDE` (`run_constants.py:3`).
- **`trinity._input.dictionary`** — `DescribedDict.flush()` (`_metadata_io.py:3`) and
  `NpEncoder` (`_metadata_io.py:41`); the writer side of the schema (`run_constants.py:3`).
- **`read_param.write_summary`** — "no longer writes ``<run>_summary.txt``" (`run_constants.py:3`).
- **"the cloudy run-loader"** / "the cloudy ambient extension" — a named downstream consumer
  (`run_constants.py:3`, `trinity_reader.py:600`).
- **`main.py`, phase runners, `phase_events`** — the sites that set `SimulationEndCode`
  (`simulation_end.py:131`, `:189`).
- **PR / phase names**: "PR2" (v1), "Phase 1" (v2), "Phase 2" (v3), "Phase 5" (v4), "Phase 6"
  (planned removal of text-parse fallbacks), PR "``metadata-expand-constants``"
  (`run_constants.py:3`, `trinity_reader.py:574`, `simulation_end.py:3`).
- **POSIX exit-code range** (`show_run.py:3`, `:484`) — see finding **14**.
- **ANSI escape codes** and **OSC 8** clickable hyperlinks (`terminal_prints.py:51`,
  `header.py:56`, `:74`).
- **Dates**: "As of January 2026" snapshot-consistency change (`trinity_reader.py:3`); file
  headers "Created on Wed Aug 16 15:39:35 2023" (`terminal_prints.py:3`) and "Wed Jul 12 13:37:22
  2023" (`header.py:3`), author "Jia Wei Teh" / "TRINITY Team".

---

## 6. Orphan / stale-doc signals (documented in exactly one place)

- **`_modified` run folders** — a whole run-naming convention ("`{run_name}_modified` folder (new
  runs)", `trinity_reader.py:1134`; "_modified suffix or profile tags like _PL0, _BE14",
  `trinity_reader.py:1389`) that no other prose in the slice acknowledges, including the v4
  "4 files per run directory" description.
- **`TRINITY_OUTPUT_DIR`** — sole mention `trinity_reader.py:1260` (finding **25**).
- **`load_output`** — used in an example, never introduced (finding **24**).
- **`F_ISM`, `EndSimulationDirectly`, `is_phiDepleted`, `bubble_dMdtGuess`, `t_next`,
  `shell_interpolate_massDot`** — named only in `run_constants.py:3`'s exclusion rationale
  (finding **19** covers `F_ISM`).
- **`DROPPED_IN_V2`** — exported constant named at `run_constants.py:3` and described at `:85`,
  but no reader-side prose says anything consumes it.
- **`isDissolved`** — flagged as a rendered row (`show_run.py:221`) and a non-run-const
  (`run_constants.py:3`), but its meaning, its outcome token, and its exit code are never
  documented anywhere in the slice.
- **`trinity/_output/__init__.py`** — contributes **zero** prose (finding **29**).

---

## 7. Findings

```json
[
  {
    "id": "S13a-B-01",
    "file": "trinity/_output/trinity_reader.py",
    "line": 480,
    "class": "state",
    "severity": "S3",
    "claim": "The reader's `termination` property claims to mirror `read_simulation_end()`'s return shape, but the two docstrings name a different fifth key: `model_name` vs `model`. A consumer migrating per the docstring's promise ('consumer migrations are one-line') would read a key that the other source never returns.",
    "evidence": "trinity_reader.py:480 — \"Mirrors :func:`trinity._output.simulation_end.read_simulation_end`'s return shape: ``{exit_code, outcome, detail, timestamp, model_name}``.\"  |  simulation_end.py:311 — \"Keys: ``exit_code``, ``outcome``, ``detail``, ``timestamp``, ``model``.\"  |  simulation_end.py:342 (comment) — \"# Legacy callers expect 'model' (not 'model_name')\"  |  simulation_end.py:131 — \"mirrors ``read_simulation_end()``'s return shape so consumer migrations are one-line\"",
    "expected": "One key name, or an explicit statement that `read_simulation_end` renames `model_name`→`model` for legacy callers and that the two shapes are therefore NOT identical.",
    "failure_scenario": "A plotter follows trinity_reader.py:480 and switches from `read_simulation_end(run_dir)['model']` to `output.termination['model']` (or vice versa) and gets a KeyError, or silently falls into a `.get()` default and labels every figure with a blank/wrong model name.",
    "repro": "Diff trinity_reader.py:480 against simulation_end.py:311 and :342; then check which key the JSON block actually carries and what read_simulation_end actually returns.",
    "confidence": "high"
  },
  {
    "id": "S13a-B-02",
    "file": "trinity/_output/trinity_reader.py",
    "line": 526,
    "class": "other",
    "severity": "S3",
    "claim": "`is_successful_run` is documented with the notation `exit_code in [0, 9]`, which in Python reads as membership of the two-element list {0, 9}, while the parenthetical ties it to `SimulationEndCode.is_clean()` which the enum docstring defines as the closed range 0-9. The doc is ambiguous about which of two materially different predicates holds.",
    "evidence": "trinity_reader.py:3 — \"output.is_successful_run # True | False | None — three-valued # True iff exit_code in [0, 9]\"  |  trinity_reader.py:526 — \"* ``True`` — exit code in [0, 9] (clean termination per ``SimulationEndCode.is_clean()``); * ``False`` — exit code outside that range\"  |  simulation_end.py:110 — \"True if the run finished with a clean physical/intentional outcome (0-9).\"  |  simulation_end.py:56 — \"- 0-9: Clean physical or intentional terminations (auto-trust)\"",
    "expected": "`0 <= exit_code <= 9` written unambiguously, in both the module docstring and the property docstring.",
    "failure_scenario": "A downstream sweep-triage script reimplements the documented predicate literally as `code in [0, 9]`; every clean run that terminated with code 1-8 (e.g. stopping-time reached) is reported as a failure and silently dropped from the analysis sample.",
    "repro": "Compare the notation at trinity_reader.py:3 and :526 with the range wording at simulation_end.py:56/:110; then check whether the implementation uses is_clean() or a literal membership test.",
    "confidence": "high"
  },
  {
    "id": "S13a-B-03",
    "file": "trinity/_output/trinity_reader.py",
    "line": 526,
    "class": "silent-failure",
    "severity": "S3",
    "claim": "`is_successful_run == None` is documented as meaning 'legacy run, crash before write_simulation_end fired', but simulation_end explicitly documents that a *successful* call to write_simulation_end can leave no termination block, because a failing metadata write is swallowed to protect the run's exit code. The documented interpretation of None is therefore incomplete in exactly the case an operator most needs to know about.",
    "evidence": "trinity_reader.py:526 — \"* ``None`` — no ``termination`` block (legacy run, crash before ``write_simulation_end`` fired).\"  |  simulation_end.py:232 (comment) — \"# The exit code is the contract of this function; the # metadata write failing should not bring the run down.\"  |  simulation_end.py:131 — \"Returns ------- int Numeric exit code from ``SimulationEndCode``.\"",
    "expected": "The None branch should also list 'write_simulation_end ran but the metadata write failed (logged, not raised)', or the writer should record the failure somewhere the reader can see.",
    "failure_scenario": "A batch triage over a sweep sees `is_successful_run is None`, concludes 'old run, re-run not needed / legacy format', and never notices that a modern run's metadata write failed — the run is silently excluded from or wrongly included in the published sample.",
    "repro": "Read trinity_reader.py:526 next to simulation_end.py:232; then check whether the write path swallows exceptions and whether any marker survives for the reader.",
    "confidence": "high"
  },
  {
    "id": "S13a-B-04",
    "file": "trinity/_output/trinity_reader.py",
    "line": 574,
    "class": "silent-failure",
    "severity": "S2",
    "claim": "`initial_cloud_profile` documents its third return as 'enclosed mass [Msun]', but a comment on the legacy fast path admits the array may be filled with zeros, and justifies this with a claim that cannot hold — that a consumer needing m 'will fall through to the v2 scalar path', when the fast path has already returned.",
    "evidence": "trinity_reader.py:574 — \"Returns ------- tuple[np.ndarray, np.ndarray, np.ndarray] ``(r_arr, n_arr, m_arr)`` — radius [pc], density [internal pc⁻³], enclosed mass [Msun].\"  |  trinity_reader.py:600 (comment) — \"# Fast path: legacy v1 inline arrays. Real v1 metadata.json files # carry all three (r, n, m); some synthetic test fixtures provide # only (r, n), in which case we fill m with zeros — consumers that # don't need the enclosed-mass array (e.g. the cloudy ambient # extension) can discard it transparently, and any future # consumer that *does* need m will fall through to the v2 scalar # path because zeros would be obviously wrong.\"",
    "expected": "Either raise/return None for the missing array, or document in the Returns block that `m_arr` may be all zeros on legacy v1 input. 'Zeros would be obviously wrong' is an unfalsifiable safety argument — nothing in the described control flow routes a zeros result to the v2 path.",
    "failure_scenario": "An analysis reconstructs the initial cloud profile from a v1 file lacking the m array, gets M(<r) ≡ 0, and computes a gravitational binding energy / escape velocity / free-fall time of zero or infinity without any warning.",
    "repro": "Read trinity_reader.py:574's Returns block against the comment at :600; then check whether the fast path can return before the v2 reconstruction and whether any zeros-detection exists.",
    "confidence": "high"
  },
  {
    "id": "S13a-B-05",
    "file": "trinity/_output/_metadata_io.py",
    "line": 97,
    "class": "state",
    "severity": "S3",
    "claim": "`update_metadata_atomic` claims that after an aborted run 'readers will then return None for run-constants', but every reader-side docstring in the slice documents a different failure mode: `{}` from the metadata accessors and a KeyError from the reconstruction helper. No documented reader returns None for a missing run-constant.",
    "evidence": "_metadata_io.py:97 — \"If ``metadata.json`` does not exist (the run terminated before any flush wrote it), a minimal file is created containing only ``_metadata_version`` and the supplied blocks — readers will then return ``None`` for run-constants that never made it to disk, which is the correct semantics for an aborted run.\"  |  trinity_reader.py:549 — \"Returns ``{}`` if the file is absent or malformed.\"  |  _metadata_io.py:60 — \"Returns ``{}`` if the file is absent or malformed\"  |  trinity_reader.py:574 — \"Raises ------ KeyError If ``metadata.json`` is missing the scalars required to reconstruct\"",
    "expected": "The writer docstring should say readers see the key as absent (KeyError on subscript, `{}` from the accessors), not None.",
    "failure_scenario": "A consumer writes `if output.metadata.get('mCloud') is None:` per this docstring — which happens to work — but another writes `output.metadata['mCloud']` per the recommendation at trinity_reader.py:549 and crashes with KeyError on every aborted run in a sweep.",
    "repro": "Read _metadata_io.py:97 against trinity_reader.py:549 and :574.",
    "confidence": "high"
  },
  {
    "id": "S13a-B-06",
    "file": "trinity/_output/run_constants.py",
    "line": 136,
    "class": "state",
    "severity": "S3",
    "claim": "`metadata_keys_to_rehydrate`'s docstring enumerates three reserved top-level keys; the module comment defining RESERVED_TOP_LEVEL_KEYS enumerates four. `termination_debug` (the v4/Phase-5 addition) was added to the constant's comment but not to the function docstring — a visible doc drift on the one function that defines what gets merged into every snapshot.",
    "evidence": "run_constants.py:136 — \"Reserved entries (``_metadata_version``, ``termination``, ``final_state``) are surfaced via dedicated ``TrinityOutput`` properties instead.\"  |  run_constants.py:102 (comment) — \"# Top-level keys in ``metadata.json`` that are NOT rehydrated into # every snapshot's data dict. Three reasons a key lives up here: … # * ``termination_debug`` — Phase-5 block; last-2-snapshot # comparison written by ``write_termination_debug_report``.\"  |  trinity_reader.py:511 — \"``termination_debug`` block from ``metadata.json`` (Phase 5, v4+ schema)\"",
    "expected": "The docstring should list all four reserved keys, or defer to the constant rather than restating its membership.",
    "failure_scenario": "A maintainer reading only the function docstring concludes `termination_debug` is rehydrated and writes a plotter that reads `snap['termination_debug']`, or adds a fifth reserved block by pattern-matching on a docstring that is already one behind.",
    "repro": "Read run_constants.py:136 against run_constants.py:102.",
    "confidence": "high"
  },
  {
    "id": "S13a-B-07",
    "file": "trinity/_output/run_constants.py",
    "line": 136,
    "class": "other",
    "severity": "S4",
    "claim": "The same docstring says all reserved entries — including `_metadata_version` — 'are surfaced via dedicated TrinityOutput properties instead', but the module docstring says the version field is consumed and discarded by the reader, and no `_metadata_version` property is documented anywhere in the slice (the reader documents only `termination`, `final_state`, `termination_debug`, `metadata`).",
    "evidence": "run_constants.py:136 — \"Reserved entries (``_metadata_version``, ``termination``, ``final_state``) are surfaced via dedicated ``TrinityOutput`` properties instead.\"  |  run_constants.py:3 — \"The version field is consumed and discarded by the reader before rehydrate.\"  |  run_constants.py:102 (comment) — \"# * ``_metadata_version`` — describes the metadata file itself.\"",
    "expected": "Say that `_metadata_version` is stripped and discarded, and that only the three data blocks have properties.",
    "failure_scenario": "A consumer looks for `output.metadata_version`, finds nothing, and hand-rolls a schema-version check by re-reading metadata.json — or assumes the version is unavailable and skips version-gating entirely.",
    "repro": "Read run_constants.py:136 against run_constants.py:3 and :102, and against the property list in trinity_reader.py:480/:496/:511/:549.",
    "confidence": "high"
  },
  {
    "id": "S13a-B-08",
    "file": "trinity/_output/simulation_end.py",
    "line": 131,
    "class": "state",
    "severity": "S3",
    "claim": "The `final_state` block is described three times with three different exclusion sets: 'every scalar/bool/string on params… arrays excluded' (module), 'every non-array non-run-constant scalar' (write_simulation_end), and a four-clause exclusion list plus two more exclusions in comments (the builder). The two higher-level docstrings promise keys the builder documents itself as dropping.",
    "evidence": "simulation_end.py:3 — \"``metadata.json[final_state]`` — every scalar/bool/string on ``params`` at run end, in INTERNAL units\"  |  simulation_end.py:131 — \"``metadata.json[final_state]`` — every non-array non-run-constant scalar from ``params`` at run end\"  |  simulation_end.py:244 — \"Includes every scalar/string/bool key on ``params`` EXCEPT: * run-constants…; * keys in ``METADATA_EXCLUDE`` (paths, function tables, …); * long arrays listed in ``FINAL_STATE_EXCLUDE_ARRAYS``…; * the ``SimulationEndCode`` proxy\"  |  simulation_end.py:272 (comment) — \"``SimulationEndReason`` is the source string for ``termination.detail``… ``path2output`` is the absolute path of the run dir itself; redundant and a privacy concern.\"  |  run_constants.py:120 (comment) — \"Anything scalar/string/bool flows through to ``final_state``.\"",
    "expected": "One canonical statement of the final_state membership rule, with the higher-level docstrings deferring to the builder rather than restating a looser version.",
    "failure_scenario": "A consumer written against simulation_end.py:3 expects `final_state['dens_profile']` or `final_state['SimulationEndReason']`, gets a KeyError on every run, and cannot tell from the docs whether the key was dropped by policy or lost by a serialization failure (see S13a-B-10).",
    "repro": "Diff the three descriptions at simulation_end.py:3, :131, :244 plus the comment at :272 and run_constants.py:120.",
    "confidence": "high"
  },
  {
    "id": "S13a-B-09",
    "file": "trinity/_output/simulation_end.py",
    "line": 288,
    "class": "deadcode",
    "severity": "S4",
    "claim": "`FINAL_STATE_EXCLUDE_ARRAYS` is documented as the mechanism that keeps long arrays out of final_state, but the builder separately documents skipping arrays/lists of ANY length. If both claims hold, the named constant is a no-op and its ~10-50 KB rationale is moot.",
    "evidence": "simulation_end.py:244 — \"* long arrays listed in ``FINAL_STATE_EXCLUDE_ARRAYS`` (their last-snapshot values are still available in the dictionary.jsonl stream's final line);\"  |  simulation_end.py:288 (comment) — \"# Skip arrays/lists with any length — final_state is scalars only.\"  |  run_constants.py:120 (comment) — \"# Keys excluded from the ``final_state`` block. Long per-snapshot # arrays already live in ``dictionary.jsonl``… duplicating them in metadata.json would # bloat the file by ~10-50 KB with no information gain.\"",
    "expected": "Either the blanket array skip or the named exclusion list, not both — and the surviving doc should say which one is load-bearing.",
    "failure_scenario": "A maintainer adds a new long array to `FINAL_STATE_EXCLUDE_ARRAYS` believing it is required, when the blanket skip already covers it; or removes the blanket skip believing the list is exhaustive, and metadata.json balloons.",
    "repro": "Read simulation_end.py:244 against the comment at :288 and run_constants.py:120; check whether FINAL_STATE_EXCLUDE_ARRAYS is ever consulted on a path the blanket skip does not already cover.",
    "confidence": "medium"
  },
  {
    "id": "S13a-B-10",
    "file": "trinity/_output/_metadata_io.py",
    "line": 97,
    "class": "silent-failure",
    "severity": "S3",
    "claim": "Two independent layers are documented to silently drop values that fail JSON serialization — the final_state builder and the atomic merge helper — with only a WARNING log. No reader-side prose tells a consumer that a documented final_state key may simply be absent for this reason, so absent-by-policy and absent-by-serialization-failure are indistinguishable downstream.",
    "evidence": "_metadata_io.py:97 — \"Defensive serialization: any value that fails ``json.dumps`` is logged at WARNING and silently dropped from the merged payload rather than crashing the write.\"  |  simulation_end.py:301 (comment) — \"# Defensive: only include keys whose final value is JSON-friendly # (json.dumps tolerates None/str/int/float/bool/NaN).\"  |  simulation_end.py:232 (comment) — \"# The exit code is the contract of this function; the # metadata write failing should not bring the run down.\"",
    "expected": "Reader-side documentation of the failure mode, or a recorded list of dropped keys inside the metadata (e.g. a `_dropped` field) so absence is distinguishable from policy.",
    "failure_scenario": "A key whose final value is a numpy object array or an unserializable enum vanishes from final_state on every run of a sweep; a downstream aggregation quietly emits NaN columns for that quantity and the omission is discovered only if someone reads the log.",
    "repro": "Read _metadata_io.py:97 and simulation_end.py:301; check whether any dropped-key record is emitted into the payload.",
    "confidence": "high"
  },
  {
    "id": "S13a-B-11",
    "file": "trinity/_output/simulation_end.py",
    "line": 244,
    "class": "citation",
    "severity": "S3",
    "claim": "The prose advertises the output as JSON / JSON Lines while simultaneously documenting that non-finite values are written as bare `NaN` / `Inf` tokens, which it concedes are 'technically non-standard JSON'. The format citation and the write contract are in direct tension: the files are claimed to round-trip only through Python's json module.",
    "evidence": "simulation_end.py:244 — \"NaN / non-finite values are kept as-is — ``json.dump`` emits them as ``NaN``, which Python's ``json.load`` reads back faithfully (technically non-standard JSON; this is the same compromise the snapshot writer makes for fields like ``cool_beta`` in the momentum phase).\"  |  simulation_end.py:680 — \"``json.dump(..., allow_nan=True)`` accepts NaN/Inf — same compromise the snapshot writer makes.\"  |  trinity_reader.py:3 — \"reading and processing TRINITY simulation output files (.jsonl)\"  |  trinity_reader.py:417 — \"Load new-style .jsonl format (line-delimited JSON).\"",
    "expected": "The reader/module docs should state that the emitted files are Python-JSON, not RFC-8259 JSON, so non-Python consumers (jq, R jsonlite, JS JSON.parse) are known to fail on runs that produce non-finite values.",
    "failure_scenario": "A collaborator pipes `dictionary.jsonl` or `metadata.json` through `jq` or loads it in R/JS; the parse aborts on the first `NaN` token — most likely on exactly the pathological runs someone is trying to debug.",
    "repro": "Read simulation_end.py:244 and :680 against the format claims at trinity_reader.py:3/:417; then attempt a strict-JSON parse of a run with a momentum-phase `cool_beta`.",
    "confidence": "high"
  },
  {
    "id": "S13a-B-12",
    "file": "trinity/_output/_metadata_io.py",
    "line": 41,
    "class": "other",
    "severity": "S3",
    "claim": "The metadata encoder is documented as a deliberate duplicate of the snapshot writer's encoder, held together only by a docstring 'MUST stay in sync' with no stated enforcement. Since the two encoders govern the on-disk representation of the same values in metadata.json and dictionary.jsonl, drift between them is a silent output-schema divergence.",
    "evidence": "_metadata_io.py:41 — \"JSON encoder that coerces numpy scalars / arrays to plain Python. Duplicates ``trinity._input.dictionary.NpEncoder`` to avoid a circular import (this module is imported by ``simulation_end``, which is imported by ``dictionary``). The two encoders MUST stay in sync.\"  |  simulation_end.py:680 — \"numpy scalars are unboxed so they don't leak into the metadata file as ``{\\\"__numpy__\\\": ...}``-style escapes.\"",
    "expected": "A test asserting the two encoders produce identical output for a representative value set, or a comment pointing at one if it exists — otherwise the MUST is unenforceable by construction.",
    "failure_scenario": "Someone fixes a numpy-dtype coercion in `trinity._input.dictionary.NpEncoder` only; metadata.json's final_state starts serializing a numpy scalar differently from the same quantity in dictionary.jsonl, and a cross-source comparison silently reads two different types for one field.",
    "repro": "Read _metadata_io.py:41; then check whether any test compares the two encoders.",
    "confidence": "high"
  },
  {
    "id": "S13a-B-13",
    "file": "trinity/_output/show_run.py",
    "line": 3,
    "class": "other",
    "severity": "S3",
    "claim": "The documented `show_run` exit-code table is not a partition: code 1 means both 'run directory not found / no metadata' and 'the run's own exit_code floored to 1', and the table has no entry for the documented third state of `is_successful_run` (None). The very shell loop the docstring proposes cannot distinguish a missing run from a clean-ish run, nor classify a legacy run at all.",
    "evidence": "show_run.py:3 — \"* 0 — pretty-print succeeded (or, with ``--quiet``, the run is successful per ``output.is_successful_run``). * 1 — run directory not found, or both ``metadata.json`` and ``simulationEnd.txt`` missing. * 1..9 — with ``--quiet``, the run's own ``exit_code`` from the termination block (floored at 1, capped at 9 so it fits in POSIX).\"  |  show_run.py:3 — \"for d in outputs/sweep_*/*/; do python -m trinity._output.show_run --quiet \\\"$d\\\" || echo \\\"BAD: $d\\\"; done\"  |  trinity_reader.py:526 — \"* ``None`` — no ``termination`` block (legacy run, crash before ``write_simulation_end`` fired).\"",
    "expected": "Distinct codes for infrastructure failure vs run outcome, and an explicit documented code for `is_successful_run is None`.",
    "failure_scenario": "A sweep-triage shell loop reports 'BAD' identically for a run directory that does not exist, a run that terminated with code 1, and (undocumented behaviour) a legacy run with no termination block — so the operator cannot tell a missing simulation from a failed one.",
    "repro": "Read the exit-code table at show_run.py:3 alongside the three-valued flag at trinity_reader.py:526 and the failure path comment at show_run.py:484.",
    "confidence": "high"
  },
  {
    "id": "S13a-B-14",
    "file": "trinity/_output/show_run.py",
    "line": 3,
    "class": "other",
    "severity": "S3",
    "claim": "`--quiet` is documented to cap the propagated exit code at 9 'so it fits in POSIX'. The stated rationale does not hold as written (process exit statuses run 0-255), and the cap collapses the enum's three distinct non-clean bands — parameter errors (10-19), numerical errors (20-29), inspection-required (50-59), and UNKNOWN (99) — onto the single value 9, which the same enum defines as clean.",
    "evidence": "show_run.py:3 — \"* 1..9 — with ``--quiet``, the run's own ``exit_code`` from the termination block (floored at 1, capped at 9 so it fits in POSIX).\"  |  show_run.py:484 (comment) — \"# Failure path: propagate the run's exit_code if we have one, # capped to [1, 9] so the value fits in POSIX.\"  |  simulation_end.py:56 — \"- 0-9: Clean physical or intentional terminations (auto-trust) - 10-19: Parameter/configuration errors - 20-29: Numerical/runtime errors - 50-59: Inspection required… - 99: Unknown/unhandled termination\"",
    "expected": "Either propagate the real code (10-99 all fit in a single-byte exit status) or document that the shell-visible code is a lossy severity indicator only, and that the real code must be read from `metadata.json[termination].exit_code`.",
    "failure_scenario": "Batch triage over a sweep sees exit status 9 for a parameter error, a numerical blow-up, an inspection-required run, and an UNKNOWN termination alike — and 9 is also a valid *clean* code — so the operator cannot separate configuration mistakes from integrator failures without re-reading every metadata.json.",
    "repro": "Read show_run.py:3 and :484 against the enum ranges at simulation_end.py:56.",
    "confidence": "high"
  },
  {
    "id": "S13a-B-15",
    "file": "trinity/_output/simulation_end.py",
    "line": 56,
    "class": "other",
    "severity": "S3",
    "claim": "The termination contract is documented only as numeric bands. No prose in the slice names a single enum member, a single code→fate mapping, or a single reason string — the sole outcome token that appears anywhere is `shell_collapsed`, and it appears only in show_run's rendering logic. Bands 30-49 and 60-98 are never described at all.",
    "evidence": "simulation_end.py:56 — \"Exit code ranges: - 0-9… - 10-19… - 20-29… - 50-59… - 99: Unknown/unhandled termination (fallback safety net). Each member carries (code, outcome_token).\"  |  simulation_end.py:69/76/82/88/92 (comments) — \"# Clean (0-9)\", \"# Parameter errors (10-19)\", \"# Numerical errors (20-29)\", \"# Inspection required (50-59)\", \"# Unknown — also treated as inspection-required\"  |  show_run.py:66 — \"``\\\"yes\\\"`` is keyed strictly on the ``shell_collapsed`` outcome\"",
    "expected": "The docstring that defines the exit-code contract should enumerate the outcome tokens, since `termination.outcome` is a documented public field that downstream automation is explicitly told to key on.",
    "failure_scenario": "A downstream tool wanting to select, say, dissolved runs has no documented token to match on; it string-matches free-form `detail` text instead, and breaks the first time a phase runner rewords its message.",
    "repro": "Search the whole slice's prose for outcome tokens: only `shell_collapsed` appears (show_run.py:66, :97). Compare with the members Lens A can see in the enum body.",
    "confidence": "high"
  },
  {
    "id": "S13a-B-16",
    "file": "trinity/_output/trinity_reader.py",
    "line": 3,
    "class": "other",
    "severity": "S3",
    "claim": "The module docstring's own usage block documents `get_at_time` two ways in four consecutive lines: first as returning the closest snapshot, then as interpolating by default. One of the two comments must be wrong.",
    "evidence": "trinity_reader.py:3 — \"# Get snapshot closest to a specific time snap_at_1myr = output.get_at_time(1.0) # Get snapshot at a specific time (interpolated by default) snap = output.get_at_time(0.5) # Returns interpolated snapshot snap = output.get_at_time(0.5, mode='closest') # Returns closest actual snapshot\"  |  trinity_reader.py:695 — \"mode : str - 'interpolate' (default): Interpolate values from neighboring snapshots. Returns an interpolated snapshot with a warning message. - 'closest': Return the actual snapshot closest to requested time.\"",
    "expected": "The first usage comment should read 'Get snapshot at a specific time (interpolated by default)' or pass `mode='closest'` explicitly.",
    "failure_scenario": "A user copies the first idiom believing they are sampling a real integrator output, and instead plots a synthesised point — a fabricated data point in a published figure, indistinguishable unless they check `is_interpolated`.",
    "repro": "Read the Basic Usage block at trinity_reader.py:3 against the `mode` default documented at :695.",
    "confidence": "high"
  },
  {
    "id": "S13a-B-17",
    "file": "trinity/_output/trinity_reader.py",
    "line": 746,
    "class": "silent-failure",
    "severity": "S2",
    "claim": "`_interpolate_snapshot`'s docstring promises 'Interpolated snapshot with is_interpolated=True', but the comment trail documents at least five paths where a field falls back to the *closest* value instead — strings/phases, booleans, NaN values, too-few points, mismatched array lengths, a catch-all default, and a blanket 'if interpolation fails for any reason'. The returned object is therefore a mixture, and the single documented marker cannot tell a consumer which fields are which.",
    "evidence": "trinity_reader.py:746 — \"Returns ------- Snapshot Interpolated snapshot with is_interpolated=True\"  |  trinity_reader.py:820/826/836/843/889/894/899 (comments) — \"# Handle strings/phases - use closest\", \"# Handle booleans - use closest\", \"# Handle NaN values\", \"# Not enough points to interpolate\", \"# Different lengths or empty - use closest\", \"# Default: use closest value\", \"# If interpolation fails for any reason, use closest value\"  |  trinity_reader.py:695 — \"'interpolate' (default): Interpolate values from neighboring snapshots.\"",
    "expected": "Document that interpolated snapshots mix interpolated and nearest-neighbour values, and which categories fall back — ideally per-field, since this is the default mode of a public accessor.",
    "failure_scenario": "A figure samples every run at a fixed t via the default interpolate mode; for one field the interpolation silently degrades to the nearest snapshot (NaN neighbour, array length change at a phase boundary), so the plotted value belongs to a different time than the axis label claims — and `is_interpolated=True` gives no hint.",
    "repro": "Read the Returns block at trinity_reader.py:746 against the fallback comments at :820-:899.",
    "confidence": "high"
  },
  {
    "id": "S13a-B-18",
    "file": "trinity/_output/trinity_reader.py",
    "line": 3,
    "class": "units",
    "severity": "S3",
    "claim": "In the 'Key Parameters' contract every documented quantity carries a bracketed unit except the four forces, which carry none — in a codebase whose own conventions call units 'a recurring bug class'.",
    "evidence": "trinity_reader.py:3 — \"**Forces:** - F_grav: Gravitational force - F_ram: Ram pressure force (total) - F_HII: HII pressure force (outward) - F_rad: Radiation pressure force\"  |  same docstring, for contrast — \"- Eb: Bubble thermal energy [Msun*pc^2/Myr^2] (internal; × INV_CONV.E_au2cgs → erg)\", \"- Pb: Bubble pressure [Msun/pc/Myr^2] (internal units)\"",
    "expected": "An internal-unit annotation for the force keys (the surrounding convention implies Msun·pc/Myr², i.e. Eb/pc), plus the CGS conversion factor as given for Eb and the luminosities.",
    "failure_scenario": "A force-budget figure mixes F_* read raw from snapshots with an externally computed force in dyne or in Msun·pc/Myr²; the ratio is off by the unit factor and the plotted force budget does not close, with no unit annotation to catch it.",
    "repro": "Read the Forces group at trinity_reader.py:3; compare with the explicit units given for every other group and with the per-key table in the code block at :191 that this lens cannot see.",
    "confidence": "high"
  },
  {
    "id": "S13a-B-19",
    "file": "trinity/_output/trinity_reader.py",
    "line": 3,
    "class": "other",
    "severity": "S4",
    "claim": "`F_ISM` is named by run_constants as a real runtime force key, but the reader's documented Forces list has only F_grav, F_ram, F_HII, F_rad. Either the user-facing key list is incomplete or F_ISM is stale.",
    "evidence": "run_constants.py:3 — \"State-machine flags that *happen* to be constant in a particular run (``EndSimulationDirectly``, ``isCollapse``, ``isDissolved``, ``is_phiDepleted``, ``bubble_dMdtGuess``, ``t_next``, ``shell_interpolate_massDot``, ``F_ISM``, …) are deliberately NOT listed here\"  |  trinity_reader.py:3 — \"**Forces:** - F_grav… - F_ram… - F_HII… - F_rad…\"",
    "expected": "F_ISM either documented in the Forces group with its unit, or removed from the run_constants rationale if it no longer exists.",
    "failure_scenario": "A force-budget plot built from the documented Forces list omits the ISM term and the budget does not sum to the net acceleration.",
    "repro": "Compare the key list at run_constants.py:3 with the Forces group at trinity_reader.py:3 and with the `# Forces` block at trinity_reader.py:191.",
    "confidence": "medium"
  },
  {
    "id": "S13a-B-20",
    "file": "trinity/_output/show_run.py",
    "line": 162,
    "class": "units",
    "severity": "S4",
    "claim": "`_final_state_section`'s docstring names two display conversions (v2→km/s, shell_nMax→cm⁻³); the comments inside the same function document four, adding Pb→P/k_B in K cm⁻³ and Eb→erg. The docstring understates the transformation applied to the human-facing output.",
    "evidence": "show_run.py:162 — \"Applies unit conversions for human reading (km/s for ``v2``, cm⁻³ for ``shell_nMax``) — same convention the legacy ``simulationEnd.txt`` used. The internal value is shown in parentheses for traceability.\"  |  show_run.py:197 (comment) — \"# Pb as P/k_B in K cm⁻³ with internal in parens (like v2 / shell_nMax)\"  |  show_run.py:203 (comment) — \"# Eb in erg with internal in parens (like Pb / shell_nMax / v2)\"",
    "expected": "The docstring should list all converted rows, or state the rule (every convertible row shows display units with the internal value in parentheses).",
    "failure_scenario": "Someone reading the docstring assumes the Pb and Eb rows are internal units and transcribes an erg value into a pipeline expecting Msun·pc²/Myr² — the parenthesised internal value is the only thing that saves them.",
    "repro": "Read show_run.py:162 against the comments at :177, :181, :187, :191, :197, :203, :209.",
    "confidence": "high"
  },
  {
    "id": "S13a-B-21",
    "file": "trinity/_output/show_run.py",
    "line": 66,
    "class": "state",
    "severity": "S3",
    "claim": "The collapse descriptor is documented as reading 'from the final snapshot', while the sanity-check prose says `isCollapse` is 'recorded in final_state' — two different sources (the dictionary.jsonl tail vs the metadata.json block). The slice separately documents that the two can disagree, because at least one per-snapshot end-of-run field is set AFTER save_snapshot ran.",
    "evidence": "show_run.py:66 — \"Three-state collapse status from the final snapshot.\"  |  simulation_end.py:699 — \"Shell collapse is a physical outcome (recorded in ``final_state`` as ``isCollapse`` and in the comparison table), not a health check\"  |  simulation_end.py:272 (comment) — \"Including either here would leak duplicated (and possibly inconsistent — the per-snapshot value is set AFTER save_snapshot ran) info into final_state.\"  |  show_run.py:239 — \"Returns a dict with keys ``metadata``, ``termination``, ``final_state``, ``termination_debug``…\"",
    "expected": "Name the authoritative source for `isCollapse` explicitly, given that the slice documents a known last-snapshot vs final_state skew.",
    "failure_scenario": "A run whose collapse flag flips in the final step is rendered 'no' by one consumer (reading the last snapshot) and 'collapsing' by another (reading final_state); two analyses of the same sweep disagree on the collapsed fraction.",
    "repro": "Read show_run.py:66 against simulation_end.py:699 and the skew note at simulation_end.py:272; check which dict `_collapse_descriptor` is actually handed by `_resolve_run_status`.",
    "confidence": "medium"
  },
  {
    "id": "S13a-B-22",
    "file": "trinity/_output/trinity_reader.py",
    "line": 3,
    "class": "state",
    "severity": "S4",
    "claim": "The snapshot-consistency guarantee is stated as an absolute ('ensuring all values in a snapshot correspond to the same timestamp'), but the slice documents at least one snapshot field written after the snapshot was saved, and the enumerated 'this includes' list quietly scopes the guarantee to a subset.",
    "evidence": "trinity_reader.py:3 — \"As of January 2026, TRINITY snapshots are saved BEFORE ODE integration, ensuring all values in a snapshot correspond to the same timestamp (t_now). This includes: t_now, R2, v2, Eb, T0, feedback properties, shell properties, bubble properties, forces, and beta-delta residuals.\"  |  simulation_end.py:272 (comment) — \"the per-snapshot value is set AFTER save_snapshot ran\"",
    "expected": "State the guarantee as scoped ('the fields listed below'), and name the known exceptions (the end-code/end-reason fields).",
    "failure_scenario": "A consumer trusts that every field of the last snapshot, including the termination flags, reflects the same instant, and reads a stale end-code from the final snapshot instead of from `termination`.",
    "repro": "Read trinity_reader.py:3's Snapshot Consistency section against simulation_end.py:272.",
    "confidence": "medium"
  },
  {
    "id": "S13a-B-23",
    "file": "trinity/_output/trinity_reader.py",
    "line": 429,
    "class": "silent-failure",
    "severity": "S3",
    "claim": "`_rehydrate_metadata` justifies its silent skip on missing metadata.json by asserting the only such case is legacy files 'where every snapshot already contains the run-constants'. The writer prose documents a second case that assertion does not cover: a v2+ run that aborted before the first flush, whose snapshots have already been stripped of run-constants.",
    "evidence": "trinity_reader.py:429 — \"Silently skips when ``metadata.json`` is absent — that's the normal case for files written before this feature landed, where every snapshot already contains the run-constants.\"  |  _metadata_io.py:3 — \"``DescribedDict.flush()`` writes the run-constants on the first flush (typically at run start, when the first batch of snapshots is saved).\"  |  _metadata_io.py:97 — \"If ``metadata.json`` does not exist (the run terminated before any flush wrote it)…\"  |  run_constants.py:3 — \"They are written exactly once per run, in ``<run_dir>/metadata.json``, and stripped from every per-snapshot dictionary in ``dictionary.jsonl``.\"",
    "expected": "Acknowledge the aborted-modern-run case, in which the run-constants exist in neither source and every snapshot silently lacks mCloud/nCore/rCloud.",
    "failure_scenario": "An early-abort run in a sweep loads without error; `output[0].get('mCloud')` returns None and `output.metadata` is `{}`; a plotting script labels or normalises the curve with a missing/zero cloud mass rather than skipping the run.",
    "repro": "Read trinity_reader.py:429 against _metadata_io.py:3/:97 and run_constants.py:3.",
    "confidence": "medium"
  },
  {
    "id": "S13a-B-24",
    "file": "trinity/_output/trinity_reader.py",
    "line": 1344,
    "class": "other",
    "severity": "S4",
    "claim": "A docstring example calls `load_output(data_path)`, a symbol introduced nowhere else in the slice's prose. The documented entry points are `TrinityOutput.open` and `read`; a bare comment mentions 'Alias for backwards compatibility' without naming the alias.",
    "evidence": "trinity_reader.py:1344 — \">>> sim_files = find_all_simulations('/path/to/outputs') >>> for data_path in sim_files: ... output = load_output(data_path)\"  |  trinity_reader.py:1094 (comment) — \"# Alias for backwards compatibility\"  |  trinity_reader.py:1069 — \"Open a TRINITY output file (convenience function).\" … \">>> output = trinity.read('simulation.jsonl')\"",
    "expected": "Use the documented API in the example, or document `load_output` alongside `read` in the module docstring.",
    "failure_scenario": "A user copies the example, `load_output` is not importable under the name they guessed, and the recommended discovery workflow fails at the first line.",
    "repro": "Read trinity_reader.py:1344 against the module docstring's Basic Usage (:3) and the alias comment at :1094.",
    "confidence": "medium"
  },
  {
    "id": "S13a-B-25",
    "file": "trinity/_output/trinity_reader.py",
    "line": 1260,
    "class": "other",
    "severity": "S3",
    "claim": "`resolve_data_input`'s default output directory is documented as \"'outputs' or TRINITY_OUTPUT_DIR env var\" — the precedence is unstated, and the phrasing lists the hardcoded default first. This is the slice's only mention of the environment variable.",
    "evidence": "trinity_reader.py:1260 — \"output_dir : str or Path, optional Base directory for output folders. Defaults to 'outputs' or TRINITY_OUTPUT_DIR env var.\"  |  trinity_reader.py:1289 (comment) — \"# Default output directory\"",
    "expected": "State precedence explicitly, e.g. 'TRINITY_OUTPUT_DIR if set, else \"outputs\"'.",
    "failure_scenario": "A user on an HPC node sets TRINITY_OUTPUT_DIR to a scratch mount, passes a bare run name, and the resolver looks under a relative 'outputs' instead — 'no data file found' for runs that exist.",
    "repro": "Read trinity_reader.py:1260; check which of the two the code consults first.",
    "confidence": "medium"
  },
  {
    "id": "S13a-B-26",
    "file": "trinity/_output/trinity_reader.py",
    "line": 1260,
    "class": "other",
    "severity": "S4",
    "claim": "`resolve_data_input`'s docstring documents three accepted input forms; the inline comments enumerate six resolution cases, including two forms the docstring does not mention (a path with an extension that does not exist yet, and a base path without an extension).",
    "evidence": "trinity_reader.py:1260 — \"Accepts: 1. Output folder name… 2. Folder path… 3. File path (e.g., \\\"/path/to/dictionary.jsonl\\\") - uses directly\"  |  trinity_reader.py:1295/1299/1309/1317/1325/1331 (comments) — \"# Case 1: It's a file that exists\", \"# Case 2: It's a directory…\", \"# Case 3: Check if it's a path with extension that doesn't exist yet\", \"# Case 4: It might be a folder name - check in output_dir\", \"# Case 5: Try as a base path (no extension) with find_data_path\", \"# Case 6: Try in output_dir as base path\"",
    "expected": "The docstring should list all accepted forms, or say the resolution is best-effort over several fallbacks.",
    "failure_scenario": "A caller relies on the documented three forms and hand-rolls extension handling that the resolver already does, or is surprised when an extensionless path resolves to a different file than expected.",
    "repro": "Read the Accepts list at trinity_reader.py:1260 against the case comments at :1295-:1331.",
    "confidence": "high"
  },
  {
    "id": "S13a-B-27",
    "file": "trinity/_output/trinity_reader.py",
    "line": 574,
    "class": "units",
    "severity": "S4",
    "claim": "Two distinct mean-molecular-weight run-constants are named across the slice — `mu_convert` (required for cloud-profile reconstruction) and `mu_atom` (given as an example run-constant) — with neither defined, no unit stated, and no prose relating them.",
    "evidence": "trinity_reader.py:574 — \"the arrays are rebuilt deterministically from ``nCore``, ``nISM``, ``rCore``, ``rCloud``, ``dens_profile``, ``densPL_alpha``, ``mu_convert`` (and ``densBE_Omega``, ``gamma_adia`` for BE).\"  |  trinity_reader.py:215 (comment) — \"# from the run-constant scalars (nCore, nISM, rCore, rCloud, # dens_profile, densPL_alpha, mu_convert).\"  |  trinity_reader.py:549 — \"every constant-through-run parameter (``mCloud``, ``nCore``, ``mu_atom``, …)\"",
    "expected": "Define both keys and their relation where they are first named — mean molecular weight conventions (per particle vs per hydrogen nucleus, with vs without helium) are a standard source of factor-~1.4 errors and this repo flags units as a recurring bug class.",
    "failure_scenario": "An external reconstruction of n(r) or M(<r) uses `mu_atom` where the reconstruction path uses `mu_convert`, producing an enclosed-mass profile wrong by a constant factor that looks physically plausible.",
    "repro": "Read trinity_reader.py:574/:215 against :549; check whether both keys exist in metadata.json and what each means.",
    "confidence": "medium"
  },
  {
    "id": "S13a-B-28",
    "file": "trinity/_output/simulation_end.py",
    "line": 3,
    "class": "other",
    "severity": "S4",
    "claim": "The simulation_end module docstring labels all three metadata blocks '(v4+ schema)', while the version history and the reader's properties date `termination` and `final_state` to v3 (Phase 2) and only `termination_debug` to v4 (Phase 5). A reader gating on the version would exclude v3 runs that do carry the first two blocks.",
    "evidence": "simulation_end.py:3 — \"All run-end data lands in ``metadata.json`` (v4+ schema) as three structured blocks\"  |  run_constants.py:3 — \"* v3 — Phase 2: adds top-level ``termination`` and ``final_state`` blocks… * v4 — Phase 5: adds top-level ``termination_debug`` block\"  |  trinity_reader.py:480 — \"(Phase 2, v3+ schema)\"  |  trinity_reader.py:511 — \"(Phase 5, v4+ schema)\"  |  simulation_end.py:311 — \"Prefers the ``termination`` block in ``metadata.json`` (v3+ schema…)\"",
    "expected": "Per-block version labels in the module docstring, matching the version history.",
    "failure_scenario": "A consumer gates `termination` reads on `_metadata_version >= 4` per the module docstring and treats every v3 run as having no termination data.",
    "repro": "Read simulation_end.py:3 against run_constants.py:3's version history and trinity_reader.py:480/:511.",
    "confidence": "medium"
  },
  {
    "id": "S13a-B-29",
    "file": "trinity/_output/__init__.py",
    "line": 1,
    "class": "other",
    "severity": "S4",
    "claim": "The package `__init__.py` for the output subpackage contributes zero comments and zero docstrings — no module docstring, no statement of the public output API. Every documented import path in the slice reaches past it into the submodule.",
    "evidence": "prose.md contains sections for only 7 of the 8 slice files; `trinity/_output/__init__.py` has no entry at all.  |  trinity_reader.py:3 — \"from trinity._output.trinity_reader import TrinityOutput, find_data_file, find_data_path\"",
    "expected": "A short module docstring naming the supported entry points (`TrinityOutput`, `read`, `read_simulation_end`, `show_run`), since this slice is described as 'the output contract every downstream analysis depends on'.",
    "failure_scenario": "Consumers import deep module paths; any future reorganisation of `_output` breaks every plotter and paper script, with no re-export layer to absorb it.",
    "repro": "Confirm the file contains no docstring/comment (it produced no prose section); compare with the documented import paths at trinity_reader.py:3.",
    "confidence": "high"
  }
]
```
