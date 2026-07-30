# S13a output core — Lens A (what the code does)

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

## Scope declaration

**Read (stripped copy, line numbers preserved):** all 8 files of the slice under
`.../lens/S13a_output_core/code/_output/` — `trinity_reader.py` (1561 L), `simulation_end.py` (745 L),
`show_run.py` (500 L), `terminal_prints.py` (232 L), `run_constants.py` (146 L), `_metadata_io.py` (132 L),
`header.py` (105 L), `__init__.py` (1 L, empty).

**Shared exception used:** yes. I read the real `/home/user/trinity/trinity/_functions/unit_conversions.py`
(definitions only: lines 58–100, 150–190, and a grep of the re-export block at 249–303) to pin the unit
system and to settle whether `cvt.E_au2cgs` and `INV_CONV.E_au2cgs` are the same number. They are — the
module re-exports the frozen-dataclass fields (`E_au2cgs = INV_CONV.E_au2cgs` at :266, `ndens_au2cgs =
INV_CONV.ndens_au2cgs` at :262, `v_au2kms = INV_CONV.v_au2kms` at :276, and `Pb_au2_KcmInv = Pb_au2cgs /
K_B_CGS` at :287 is module-level only). Numerically `ndens_au2cgs = 1/2.937998946096347e+55 ≈ 3.404e-56`
(internal pc⁻³ → cm⁻³) and `E_au2cgs = 1/5.260183968837699e-44`.

**Not read:** the real `trinity/` tree (apart from the declared exception), `docs/dev/`, `test/`, `param/`,
`outputs/`, this slice's `prose.md` / `signatures.md`, and any other agent's report. Nothing was written to
the repo.

**Structural blind spot, stated up front:** the *writer* of `dictionary.jsonl` is **not in this slice**, and
neither is the writer of the top-level metadata run-constants, `trinity._input.dictionary.DescribedItem`,
or `trinity._input.registry` (source of `RUN_CONST_KEYS` / `METADATA_EXCLUDE`). I can therefore only walk
**half** the round trip directly: metadata write (`_metadata_io` + `simulation_end`) → read
(`trinity_reader`, `show_run`). For the snapshot half I can only characterise what the reader *assumes*
and what it does when the assumption breaks. Every conclusion that depends on the snapshot writer is marked
and its confidence lowered.

---

## 1. The write→read round trip, as the code actually implements it

There are **two** persisted artefacts and they are produced by different machinery with different
serialisation contracts.

**`dictionary.jsonl`** — one JSON object per line, appended during the run (inferred: `_load_last_snapshots`
in `trinity/_output/simulation_end.py:464-485` opens it at termination and takes the last lines; the writer
itself is out of slice). Read back by `TrinityOutput._load_jsonl_format`
(`trinity/_output/trinity_reader.py:416-425`), which is four lines of `json.loads(line)` with no
error tolerance whatsoever, no ordering check, and no schema check.

**`metadata.json`** — a single JSON object, written by `_metadata_io.write_metadata_atomic`
(`trinity/_output/_metadata_io.py:78-93`) through a `.tmp` + `os.replace`. It holds run constants at top
level plus four reserved blocks, enumerated in `trinity/_output/run_constants.py:113-118`:
`_metadata_version`, `termination`, `final_state`, `termination_debug`.

The two are stitched together at read time by `_rehydrate_metadata`
(`trinity/_output/trinity_reader.py:427-461`): every top-level metadata key that is *not* reserved is
pushed into *every* snapshot dict via `snap.setdefault(k, v)`. So the reader's notion of a "snapshot" is
a merge of two files, and `TrinityOutput.keys` (`:349-352`, `:649-651`) is the union of per-snapshot
time-series keys and run constants with no marker distinguishing them.

Three things about that merge are worth naming precisely.

First, `setdefault` inserts **the same object** for every snapshot — a metadata list such as
`initial_cloud_r_arr` becomes one list aliased into N snapshot dicts. Any consumer that mutates
`snap['initial_cloud_r_arr'][0]` silently mutates it in all N.

Second, the reserved-key filter is a denylist (`run_constants.py:145-146`,
`return {k: v for k, v in metadata.items() if k not in RESERVED_TOP_LEVEL_KEYS}`). Any future top-level
block is silently rehydrated into every snapshot as if it were a physical quantity.

Third, `_load_json_format` and `_load_jsonl_format` are not equivalent readers of the same data.
The JSON path **mutates** what it loads — `snap['snap_id'] = int(key)` at `trinity_reader.py:405` — and
then **sorts** by that key (`:411`, `snapshots.sort(key=lambda s: s.get('snap_id', 0))`). The JSONL path
does neither. So the key set and the ordering guarantee of `TrinityOutput` depend on which of the two
accepted extensions you handed it, and `find_data_path` / `resolve_data_input` will happily hand you either
(`trinity_reader.py:1223-1250`, `:1300-1335`).

### 1.1 Serialisation fidelity of `metadata.json`

Precision is fine: `json.dump` uses `repr` for floats, so full round-trip precision is preserved
(`_metadata_io.py:92`).

Non-finite values are **not** fine. Nothing on the write path ever passes `allow_nan=False`. The guard in
`update_metadata_atomic` (`_metadata_io.py:123`) is `json.dumps(value, cls=_NpEncoder)` — default
`allow_nan=True`. The actual write (`_metadata_io.py:92`) is `json.dump(payload, f, cls=_NpEncoder,
indent=2, sort_keys=False)` — default `allow_nan=True`. And the `final_state` guard in
`simulation_end.py:303` is explicitly `json.dumps(val, allow_nan=True)`. Python emits the bare tokens
`NaN`, `Infinity`, `-Infinity`, which Python's own `json.load` accepts on the way back in — so the
in-Python round trip looks clean — but which are **not** JSON, and which `jq`, JavaScript `JSON.parse`,
Go, R `jsonlite`, and most other consumers reject outright. `show_run --json`
(`show_run.py:466-472`) prints the file verbatim, so the invalid token propagates.

This is not hypothetical. `_compute_change` in `simulation_end.py` returns `float('inf')` on four distinct
paths — `:501` (`"NEW"`), `:503` (`"GONE"`), `:508` (a string or bool changed), `:540` (`old_f == 0`) —
and the caller writes it straight into the persisted block at `simulation_end.py:643`,
`"rel_change": _jsonable(rel_change),`. `_jsonable` (`:679-695`) passes floats through untouched. A phase
change between the last two snapshots, or `R1`/`Eb` going absent when the run ends in the momentum phase
(which `_build_sanity_checks` at `:714` and `:723` treats as expected), or any of `F_grav`/`F_ram`/`F_rad`
being exactly 0 in the penultimate snapshot, is enough. Termination is precisely when those things happen.

### 1.2 Three different serialisation policies for the same problem

Within this one slice, an unserialisable value is handled three ways:

- `_build_final_state_block` (`simulation_end.py:294-305`) **drops it**, using a bare `json.dumps` with no
  encoder — so a `np.ndarray` is rejected here…
- …even though `_NpEncoder` (`_metadata_io.py:47-56`) would have serialised it via `obj.tolist()`, and
  `update_metadata_atomic`'s own guard (`:123`) does use that encoder. The `final_state` guard is
  therefore **stricter than the writer that consumes its output**.
- `_jsonable` (`simulation_end.py:679-695`) **stringifies it**: `return str(val)`. A list value reaching
  the `old`/`new` fields of the comparison table is persisted as the string `"[1.0, 2.0]"`.

And within `_build_final_state_block` the empty-container cases split: `if isinstance(val, (list, tuple))
and len(val) > 0: continue` (`:289`) means an **empty list survives** into `final_state`, while
`if isinstance(val, np.ndarray) and val.size > 0: continue` (`:291`) lets an empty ndarray through to
`json.dumps`, which raises `TypeError` and drops it (verified: `json.dumps(np.array([]))` → `TypeError`;
`json.dumps([])` → `"[]"`). Same semantic case, opposite outcome.

The consequence for a downstream consumer is that **the `final_state` key set is a function of the values,
not of the schema**. A key whose terminal value happens to be an array is absent; the same key with a
scalar terminal value is present. No two runs are guaranteed to have the same `final_state` keys, and
`show_run._final_state_section` (`show_run.py:169-235`) is written entirely as `if X is not None` guards —
it renders a shorter report rather than reporting that anything is missing.

### 1.3 What the reader does with key drift and shape drift

`TrinityOutput.get` (`trinity_reader.py:684-689`) is `values = [s.get(key) for s in self._snapshots]`
followed by `np.array(values)`. A key present in only some snapshots yields an object-dtype array with
embedded `None`s — silently, and with no way for the caller to tell that from a genuine `None` column.
Ragged list values raise `ValueError` inside `np.array` and the `except (ValueError, TypeError)` at `:688`
returns the raw Python list instead, so the return type of `get(..., as_array=True)` is not stable.

`Snapshot.__getitem__` (`:291-293`) is `return self.data.get(key)` — a **misspelled key returns `None`
rather than raising `KeyError`**. `Snapshot.t_now` (`:303-306`) defaults to `0.0`, and `t_min`/`t_max`
(`:658-666`) use `s.get('t_now', 0)`, so a snapshot missing `t_now` silently drags the reported time range
to zero.

`to_dataframe` (`:1035-1057`) decides which keys are scalar by sampling **snapshot 0 only** (`:1051-1054`),
as does `_print_parameters`'s type column (`:1014-1019`). A key that is scalar early and array-valued later
is admitted to the DataFrame and becomes an object column.

---

## 2. `get_at_time` and the interpolator

`get_at_time` (`trinity_reader.py:692-743`) first tries an exact match:

```
721	        exact_idx = np.where(np.isclose(times, t, rtol=1e-10))[0]
```

`np.isclose` has `atol=1e-8` by default and it is **not overridden**. The effective test is
`|t_i - t| <= 1e-8 + 1e-10·|t|`, so the absolute term dominates for every `t < 0.1 Myr` — i.e. for the
whole early/energy-driven phase. I verified the consequence directly:
`np.isclose(1.005e-6, 1e-6, rtol=1e-10)` → `True`, and `np.isclose(5e-9, 1e-9, rtol=1e-10)` → `True`
(a factor-of-five error reported as an exact hit). The tightened `rtol` is inert; the tolerance the code
actually enforces is a fixed 1e-8 Myr window.

`_interpolate_snapshot` (`:745-911`) has four separable behaviours worth recording.

**Key set from one neighbour.** `all_keys = self._snapshots[neighbor_indices[0]].keys()` (`:806`) — the
first (earliest) neighbour only. Any key that appears later in the window is absent from the interpolated
snapshot entirely.

**Type inference from one neighbour.** `first_val = values[0]` (`:814`), and if it is `None` the code
short-circuits: `interpolated_data[key] = None; continue` (`:816-818`). A key that is `None` in the
earliest neighbour and float-valued in the other four returns `None`.

**Silent degradation to nearest-neighbour.** The whole per-key body is wrapped in
`except Exception:` at `:898-901`, whose handler is the closest-snapshot lookup. There is no logging.
Meanwhile the banner printed at `:796-800` says *"Interpolating from N neighbors … NOTE: These are
interpolated values"*, and the returned object carries `is_interpolated=True` (`:906-911`). A snapshot in
which every key fell through to the closest-value path is indistinguishable, to the caller, from a genuinely
interpolated one — and it is labelled as interpolated in `Snapshot.__repr__` (`:314-315`).

**The scalar and array branches disagree on the degenerate case.** With fewer than two finite points the
scalar branch takes the *valid* one — `y_vals[valid_mask][0]` (`:844`) — while the array branch takes
element zero regardless of validity — `elem_values[0]` (`:875`). For an array element that is NaN in the
first neighbour and finite in exactly one other, the scalar policy returns the finite value and the array
policy returns NaN.

Separately, `isinstance(first_val, bool)` is special-cased to nearest-neighbour at `:827-830`, but plain
`int` falls into the numeric branch at `:833` and is linearly interpolated. Integer-coded quantities —
`snap_id` injected by `_load_json_format`, counters, any 0/1 flag stored as `int` rather than `bool` —
come back as fractional floats.

Finally the two modes of `get_at_time` disagree on out-of-range input: `'interpolate'` raises `ValueError`
(`:766-770`) while `'closest'` silently returns the nearest endpoint (`:729-737`).

---

## 3. Termination and exit reporting

### 3.1 The taxonomy

`SimulationEndCode` (`simulation_end.py:55-127`) defines four bands: clean fates 0–4
(`shell_dissolved`, `stopping_time`, `large_radius`, `rcloud_boundary`, `shell_collapsed`), input/validation
errors 10–13, numerical/solver errors 20–23, "inspection required" 50–51 (`velocity_runaway`,
`energy_collapsed`), and `UNKNOWN = 99`. The three predicates are `is_clean()` → `0 <= code <= 9`,
`is_error()` → `10 <= code <= 29`, `is_inspection_required()` → `(50 <= code <= 59) or code == 99`.

`from_code` (`:121-127`) returns `UNKNOWN` for any unmatched integer — including 5–9, which are inside the
"clean" band but unassigned.

### 3.2 The classification is narrowed to a boolean on the read path

`TrinityOutput.is_successful_run` (`trinity_reader.py:524-545`) collapses the taxonomy:

```
543	            return 0 <= int(ec) <= 9
```

Everything ≥ 10 is `False`. `show_run._status_line` (`show_run.py:108-115`) then renders `False` as
`"✗ ERROR"`. So a run that ended with `VELOCITY_RUNAWAY` (50) or `ENERGY_COLLAPSED` (51) — physical fates
the enum deliberately separates from errors — is printed as `ERROR` by `show_run`, while
`terminal_prints.format_end_report` (`terminal_prints.py:222-227`) prints *"Simulation ended (inspection
required)"* for the same code. **The terminal transcript and the file-derived summary disagree about
whether the run failed.** A consumer reading only `is_successful_run` cannot distinguish a solver blow-up
(22) from a physically interesting runaway (50) from a code that could not be parsed (99).

### 3.3 `--quiet` clamps the exit code away

```
489	                return min(max(int(t["exit_code"]), 1), 9)
```
(`show_run.py:489`, reached only when `is_successful` is not `True`, i.e. when the code is ≥ 10 or absent.)

Every failure mode is mapped onto the single shell status 9. The argparse help two lines above advertises
`"exit with the run's exit_code (0=success, non-zero=failure)"` (`show_run.py:455-456`). Worse, 9 lies
*inside* the clean band `0 <= code <= 9` that `is_clean()` and `is_successful_run` both use, and
`SimulationEndCode.from_code(9)` returns `UNKNOWN`. A shell loop that re-interprets the status with the
project's own rules gets a self-contradictory answer.

### 3.4 The code can be silently downgraded to UNKNOWN at write time

```
193	    if 'SimulationEndCode' in params:
194	        raw = params['SimulationEndCode'].value
195	        if isinstance(raw, int):
196	            end_code = SimulationEndCode.from_code(raw)
```
(`simulation_end.py:192-196`; `end_code` was initialised to `SimulationEndCode.UNKNOWN` at `:192`.)

If `.value` is anything other than a Python `int` — a `np.int64` (verified: `isinstance(np.int64(4), int)`
is `False`), a float `4.0`, a string `"4"`, or the enum member itself — the branch is skipped **with no
warning**, and the run is persisted as `{"exit_code": 99, "outcome": "unknown"}`. `is_successful_run` then
reports `False` and `show_run` prints `✗ ERROR` for a run that terminated cleanly. Whether this fires
depends on the producer of `SimulationEndCode`, which is outside my slice — hence medium confidence on the
trigger, high on the mechanism. Note also that `reason_str` at `:184-185` is
`params['SimulationEndReason'].value or 'unknown'`, so an empty-string reason also becomes `'unknown'`, and
the human-readable `detail` survives even when the code does not — the only surviving evidence of the real
fate is a free-text string.

### 3.5 A solver failure and a physical fate *are* separable in principle, and are not in practice

The enum bands do separate them. But three things erase the separation on the way out: the boolean
narrowing in §3.2, the exit-status clamp in §3.3, and the `isinstance(raw, int)` gate in §3.4. What a
downstream consumer can actually rely on is `termination["outcome"]` (a string) and
`termination["detail"]` (free text). `termination["exit_code"]` is unreliable in the two failure modes
above; the process exit status is unreliable always.

---

## 4. Metadata I/O

`update_metadata_atomic` (`_metadata_io.py:96-132`) is read-modify-write:

```
117	    existing = read_metadata(run_dir)
118	    if not existing:
119	        existing = {"_metadata_version": METADATA_VERSION}
```

and `read_metadata` (`:59-75`) returns `{}` on **any** `OSError` or `JSONDecodeError`, with only a
`logger.warning`. Composed, these two mean: **if `metadata.json` is unreadable or corrupt at termination
time, every run constant it contained is silently discarded and the file is replaced by a stub containing
only the termination blocks.** The `if not existing` test cannot tell "file absent" from "file corrupt"
from "file legitimately `{}`". After that, `TrinityOutput.initial_cloud_profile` raises `KeyError`
(`trinity_reader.py:623-628`), `_rehydrate_metadata` injects nothing, and `show_run._cloud_section` renders
an empty Cloud block — all without any indication that data was destroyed rather than never written.

`METADATA_VERSION = 4` (`run_constants.py:100`) is stamped **only in the `not existing` branch** (`:119`).
Updating a pre-existing v1/v2/v3 file leaves the old stamp in place while adding v4-shaped blocks, and
nothing anywhere in this slice ever reads `_metadata_version` back or validates it — `read_metadata`,
`TrinityOutput.metadata` (`trinity_reader.py:547-571`) and `metadata_keys_to_rehydrate`
(`run_constants.py:135-146`) all ignore it. The version field is write-only and can be stale.

**When it is written relative to the run.** Both writers fire at termination:
`write_simulation_end` (`simulation_end.py:130-240`) writes `termination` + `final_state`;
`write_termination_debug_report` (`:558-676`) writes `termination_debug`. Both call
`update_metadata_atomic`, each doing its own read-modify-write, so they must be sequential — which they are
within one process. Across processes there is no lock, and the temp path is a fixed name,
`tmp = path.with_suffix(path.suffix + ".tmp")` (`:90`) → `metadata.json.tmp`; two processes writing the
same run directory collide on it and one block is lost. There is no `f.flush()`/`os.fsync()` before
`os.replace` (`:91-93`), so the rename is atomic with respect to the directory entry but the contents are
not made durable.

**Partial/interrupted write.** If `json.dump` raises mid-write the `.tmp` is left on disk and `os.replace`
never runs, so the previous `metadata.json` survives — that part is correct. The failure that is *not*
handled is at the caller:

```
231	    except Exception as e:
...
235	        logging.getLogger(__name__).warning(
236	            "Failed to mirror termination/final_state into metadata.json: %s",
237	            e,
238	        )
239
240	    return end_code.code
```
(`simulation_end.py:225-240`.) The exit code is returned to the caller as if everything succeeded. The run
then presents to `show_run` exactly as an aborted run does:
`"Status   : ? UNKNOWN  (no termination block — legacy or aborted run)"` (`show_run.py:109`).
`_merge_termination_debug` (`simulation_end.py:736-745`) swallows identically.

One more silent redirection: if `path2output` is absent from `params`, `write_simulation_end` writes into
the current working directory — `output_dir = '.'` at `simulation_end.py:181`, followed by
`os.makedirs(output_dir, exist_ok=True)` at `:207`.

---

## 5. Robustness of the read path against an interrupted run

`_load_jsonl_format` (`trinity_reader.py:416-425`) calls `json.loads(line)` with **no per-line guard**. A
run killed mid-append leaves a truncated final line, and the read raises `json.JSONDecodeError`. The
suffix dispatch in `TrinityOutput.open` catches that only in the *unknown-extension* branch
(`:381-386`); for an explicit `.jsonl` — which is what `find_data_path` returns for a run directory
(`:1246-1250`) — the exception propagates. And `show_run._resolve_run_status` guards only
`except FileNotFoundError:` (`show_run.py:271`).

So `show_run`, the tool whose entire purpose is to inspect a run's outcome, **tracebacks on the output of
a run that was killed mid-write** — the exact case it exists for. The contrast is inside this same slice:
`_load_last_snapshots` (`simulation_end.py:479-483`) *does* wrap `json.loads` in
`except json.JSONDecodeError: continue`. Two readers of the same file, opposite policies.

(The tolerant one has its own cost: it takes the last two lines with `lines[-n:]` after
`f.readlines()` — the whole trajectory into memory to read two records — and when the last line is
truncated it silently skips it, so `debug_block["snapshot_count"]` reports 1 and the entire comparison
table is omitted, indistinguishable from a run that produced exactly one snapshot.)

---

## 6. The termination diagnostic can report "no change" for a value that became NaN or Inf

`_compute_change` (`simulation_end.py:488-555`) is the change detector behind
`debug_block["comparison"]` and `debug_block["warnings"]`.

Array branch:
```
524	        if not np.isfinite(max_rel):
525	            max_rel = 0.0
526	        return f"max Δ={max_rel:.1%}", max_rel, max_rel > 0.5
```
The clamp happens **before** the string is formatted, so an array whose relative change is `inf` (a finite
value going to `±inf`) or all-NaN (`np.nanmax` of an all-NaN array returns NaN, verified) is persisted as
`"max Δ=0.0%"` with `flagged: false`.

Scalar branch: `if old_f == new_f:` (`:535`) is a float equality, and NaN fails it, so NaN values fall
through to `diff = nan`, `rel_change = nan`; every subsequent comparison against NaN is `False`, giving
`change_str = "nan (nan%)"` and, back in the caller at `simulation_end.py:634`,
`flagged = rel_change > threshold` → `False`. **A quantity that went finite → NaN is never flagged by the
comparison table.**

It is partially caught elsewhere — the independent scan at `:653-670` populates
`debug_block["invalid_values"]`, and `show_run._termination_debug_section` (`show_run.py:356-363`) does
surface those. But that scan covers only `snap_new`, so a value that was NaN in the penultimate snapshot
and recovered is invisible, and the "warnings" list that the summary leads with is wrong.

Two further hygiene notes on the same function: the significance threshold `0.5` is hard-coded at `:526`
while `CHANGE_THRESHOLDS['default'] = 0.5` lives at `:439` — the same magic number in two places, only one
of which is configurable; and `is_sig` (the third return value) is consumed by the caller only when
`threshold == 0` (`simulation_end.py:631-634`), so the array branch's own verdict is dead in every other
case.

---

## 7. Divergences between the three formatters of the same quantities

There are three independent tables of "the interesting state variables plus their unit conversions":
`terminal_prints._STATE_FIELDS` (`terminal_prints.py:131-140`), `simulation_end.CRITICAL_PARAMS`
(`simulation_end.py:409-434`), and the hand-written rows in `show_run._final_state_section`
(`show_run.py:176-216`).

**The conversion factors agree.** I verified via the declared exception that `cvt.v_au2kms is
INV_CONV.v_au2kms`, `cvt.E_au2cgs is INV_CONV.E_au2cgs`, `cvt.ndens_au2cgs is INV_CONV.ndens_au2cgs`
(re-exports at `unit_conversions.py:262,266,276`), and that `Pb_au2_KcmInv` is a single module-level
definition (`:287`) imported by all three. **No unit divergence between the terminal and the file summary.**

**The display precisions do not agree.** `t_now`: `.6f` in the terminal (`terminal_prints.py:132`), `.3f`
in `show_run` (`show_run.py:173`), `.4e` in the reader (`trinity_reader.py:964`). `R2`: `.4f` vs `.3f`.
`Eb`: `.4e` vs `.3e`. `.3f` on a sub-milliparsec radius or a sub-millisecond-of-Myr time prints `0.000`.

**`PARAM_DOCS` mislabels the density family by a factor of ~2.9e55.** `trinity_reader.py:203-205` and
`:209-210` document `shell_n0`, `shell_nMax`, `nEdge`, `nCore`, `nISM` as `[cm^-3]`. But
`simulation_end.py:422` reads `shell_nMax` **out of the snapshots** and multiplies it by
`INV_CONV.ndens_au2cgs` to reach `cm⁻³`; `show_run.py:194` does the same for the metadata copy and labels
the unconverted value `pc⁻³` at `:196`; `show_run.py:141,146` do it for `nCore`/`nISM`; and
`header.py:94` does `params['nCore'].value*cvt.ndens_au2cgs`. The stored values are internal pc⁻³. This
matters because `PARAM_DOCS` is not a comment — it is printed to the user by `info(verbose=True)`
(`trinity_reader.py:1013`, `:1020`) as the file's unit contract, and every *other* entry in the table is
scrupulous about it (`'Eb': '… (× INV_CONV.E_au2cgs → erg)'` at `:162`, `'Qi': '[1/Myr] (× s2Myr →
photons/s)'` at `:182` — which I checked and is dimensionally correct). The density entries are the sole
outliers.

**`mCloud` is interpreted two different ways.** `show_run._cloud_section` derives the cluster mass as
```
135	        mCluster = md["mCloud"] * md["sfe"]
```
(`show_run.py:134-136`), i.e. `mCloud` is the total. `header.show_param` prints
```
91	    print(f"\tlog_mCloud: {np.log10(params['mCloud']/(1-params['sfe']))} Msun")
```
(`header.py:91`), i.e. the quantity labelled `log_mCloud` is `log10(mCloud/(1-sfe))`, which is only the
cloud mass if the stored `mCloud` is the *post-star-formation gas* mass. The two cannot both be right.
Note also that line 91 is the only line in `show_param` that omits `.value` — `:90`, `:93`, `:94`, `:95`
all use it. Whether that line even executes depends on `DescribedItem`'s numeric protocol, which is outside
my slice; I report the inconsistency, not a verdict on which convention is correct.

---

## 8. Dead code and unreachable branches (flagged only, per project rule)

- `trinity_reader.py:132` imports pandas at module scope, which makes the `try: import pandas as pd / except
  ImportError: raise ImportError("pandas is required for to_dataframe()")` guard at `:1044-1047`
  **unreachable** — the module could not have imported at all. Same shape for
  `from scipy import interpolate as scipy_interp` at `:133`: reading a JSONL file has a hard scipy
  dependency.
- `trinity_reader.py:1018-1019`, `elif isinstance(sample, float) and not np.isnan(sample): stype = f'float'`
  — `type(sample).__name__` on line `:1015` already yields `'float'`, so the branch changes nothing.
- `run_constants.py:88-92`, `DROPPED_IN_V2` — defined, never referenced anywhere in the slice.
- `trinity_reader.py:1133`, `find_data_file` — zero callers in the slice; superseded in function by
  `find_data_path` (`:1192`) and `resolve_data_input` (`:1258`), both of which *are* used.
- `Snapshot.interpolation_time` (`trinity_reader.py:289`) is written at `:910` and never read.
- `SimulationEndCode` codes 5–9 are unassigned yet inside the band that `is_clean()` and
  `is_successful_run` accept.
- `trinity_reader.py:1095`, `load_output = read` — alias with no in-slice user.
- `_output/__init__.py` is empty, so nothing is re-exported at package level; every import in the slice is
  fully qualified. (Not a defect — noted because it means `from trinity._output import X` fails.)

## 9. Ordering, module state, and miscellany

There is **no module-level mutable run state** in this slice — `PARAM_DOCS`, `CRITICAL_PARAMS`,
`CHANGE_THRESHOLDS`, `_STATE_FIELDS`, `HEARTBEAT_EVERY` are all read-only, and `TrinityOutput._metadata_cache`
is per-instance. So nothing here leaks across runs in one process. Two side effects are worth naming
anyway:

- `show_run.py:41-43` mutates `sys.path` at **import** time (`_sys.path.insert(0, str(_PROJECT_ROOT))`),
  affecting the whole interpreter for anyone who merely imports `format_run_summary`.
- `terminal_prints.log_file_saved` / `log_warning` / `log_error` (`:106-118`) embed ANSI escapes
  unconditionally into `logger.*` calls, with no `isatty()` test — so redirected logs and log files carry
  control codes. `header.py:28-33` and the OSC-8 hyperlink escape in `header.link` (`:75`) do the same to
  stdout.
- `TrinityOutput.phases` (`:653-656`) returns `list(set(...))`; with string hash randomisation the order
  varies between processes. `info()` sorts it (`:970`) but the public property does not.
- `TrinityOutput.__getitem__` is annotated `-> Snapshot` but returns a **list** for a slice index (`:469-470`).
- `heartbeat` (`terminal_prints.py:198`) defaults a missing `t_now` to `tmax`, i.e. reports 100% progress.
- `_phys` (`terminal_prints.py:143-160`) returns the string `"n/a"` both for a missing key and for a value
  that fails `float()` — the two are indistinguishable in the transcript.
- `_status_line` (`show_run.py:108-109`) prints *"no termination block — legacy or aborted run"* whenever
  `is_successful is None`, which also happens when the block **is** present but its `exit_code` is missing
  or unparsable (`trinity_reader.py:539-545`).
- `organize_simulations_for_grid` (`trinity_reader.py:1506`) keys the grid on `(mCloud, sfe)` only, so with
  no `ndens_filter` two runs at different densities silently overwrite each other while `ndens_list` still
  reports both. The keys are the **strings** captured by the regex, so `'1e6'` and `'1.0e6'` are distinct
  cells for the same mass.
- `parse_simulation_params` (`:1406-1410`) requires an exponent-form mass (`e\d+`, no sign) and an
  **integer** SFE (`sfe(\d+)`); a folder named `…_sfe0.05_n…` returns `None`.
  `organize_simulations_for_grid` prints a warning on that (`:1484`) while `info_simulations` (`:1548-1554`)
  and `get_unique_ndens` (`:1435-1439`) skip silently — same failure, two policies.

---

```json
[
  {
    "id": "S13a-A-01",
    "file": "trinity/_output/_metadata_io.py",
    "line": 92,
    "class": "numerical",
    "severity": "S2",
    "claim": "metadata.json is written with json's default allow_nan=True on every path, so NaN/Infinity/-Infinity are emitted as bare tokens. The file is not valid JSON and is rejected by every non-Python consumer. This is reliably triggered at termination, not hypothetically.",
    "evidence": "_metadata_io.py:92 `json.dump(payload, f, cls=_NpEncoder, indent=2, sort_keys=False)`; guard at _metadata_io.py:123 `json.dumps(value, cls=_NpEncoder)`; simulation_end.py:303 `json.dumps(val, allow_nan=True)`. Producers of inf: simulation_end.py:501 `return \"NEW\", float('inf'), True`, :503 `return \"GONE\", float('inf'), True`, :508 `return f\"{old_val} → {new_val}\", float('inf'), True`, :540 `rel_change = float('inf') if diff != 0 else 0.0`; persisted at :643 `\"rel_change\": _jsonable(rel_change),` and _jsonable passes floats through (:693-694 `if isinstance(val, (int, float)): return val`).",
    "expected": "allow_nan=False with an explicit sentinel (null, or a string \"Infinity\"), or a normalisation pass before write, so metadata.json is parseable by jq/JS/R.",
    "failure_scenario": "A run ends in the momentum phase where R1/Eb are absent from the last snapshot but present in the penultimate one -> _compute_change returns 'GONE' with rel_change=inf -> metadata.json contains `\"rel_change\": Infinity`. `jq . metadata.json` fails; any JS/Go/R analysis pipeline over a sweep aborts on that run. Python readers are unaffected, so the corruption is invisible in-project.",
    "repro": "python3 -c \"import json;print(json.dumps({'rel_change':float('inf')}))\"  ->  {\"rel_change\": Infinity}; then `echo '{\"a\": Infinity}' | jq .` -> parse error.",
    "confidence": "high"
  },
  {
    "id": "S13a-A-02",
    "file": "trinity/_output/_metadata_io.py",
    "line": 117,
    "class": "silent-failure",
    "severity": "S2",
    "claim": "update_metadata_atomic destroys all existing metadata when read_metadata fails. read_metadata returns {} on any OSError/JSONDecodeError with only a warning; `if not existing` then replaces the whole file with a stub containing only the new blocks. A corrupt or transiently-unreadable metadata.json is overwritten rather than preserved.",
    "evidence": "_metadata_io.py:73-75 `except (OSError, json.JSONDecodeError) as e: logger.warning(\"Failed to read %s: %s\", path, e); return {}`; _metadata_io.py:117-119 `existing = read_metadata(run_dir)` / `if not existing:` / `existing = {\"_metadata_version\": METADATA_VERSION}`; then :132 `write_metadata_atomic(run_dir, existing)`.",
    "expected": "Distinguish 'absent' from 'unreadable'. On a read error, refuse to overwrite (or side-write metadata.json.recovered) rather than replacing the run's constants with a stub.",
    "failure_scenario": "metadata.json is truncated by a full disk or a killed writer. At termination write_simulation_end calls update_metadata_atomic, read_metadata warns and returns {}, and the file is rewritten containing only _metadata_version/termination/final_state. Every run constant (mCloud, nCore, rCloud, dens_profile, mu_convert, initial_cloud_*) is gone. TrinityOutput.initial_cloud_profile then raises KeyError (trinity_reader.py:623-628), _rehydrate_metadata injects nothing, show_run's Cloud section renders empty — with no indication that data was destroyed rather than never written.",
    "repro": "printf '{\"mCloud\": 1e6, ' > <run>/metadata.json ; then run write_simulation_end(params, output_dir='<run>') and diff the file.",
    "confidence": "high"
  },
  {
    "id": "S13a-A-03",
    "file": "trinity/_output/trinity_reader.py",
    "line": 423,
    "class": "silent-failure",
    "severity": "S2",
    "claim": "_load_jsonl_format has no per-line error tolerance, so a truncated final line (a run killed mid-append) makes TrinityOutput.open raise JSONDecodeError. show_run guards only FileNotFoundError, so the tool for inspecting a run's outcome tracebacks on exactly the interrupted-run case it exists for. The sibling reader in the same slice does tolerate it.",
    "evidence": "trinity_reader.py:419-423 `with open(filepath, 'r') as f:` / `for line in f:` / `line = line.strip()` / `if line:` / `snapshots.append(json.loads(line))` — no try. trinity_reader.py:379-380 dispatches `.jsonl` straight there, bypassing the `except json.JSONDecodeError` that only guards the unknown-suffix branch at :383-386. show_run.py:271 `except FileNotFoundError:`. Contrast simulation_end.py:479-483 `try: snap = json.loads(line); snapshots.append(snap) except json.JSONDecodeError: continue`.",
    "expected": "Either tolerate a trailing partial line (matching _load_last_snapshots) and report how many lines were skipped, or catch JSONDecodeError in _resolve_run_status and fall back to the metadata-only path that is already implemented at show_run.py:273-292.",
    "failure_scenario": "SLURM sends SIGKILL at the wall-clock limit while a snapshot line is half-written. `python -m trinity._output.show_run <rundir>` raises json.decoder.JSONDecodeError instead of printing the termination block that is sitting intact in metadata.json. Sweep post-processing that loops show_run over runs dies on the first killed run.",
    "repro": "cp a good dictionary.jsonl, `truncate -s -20 dictionary.jsonl`, then `python -m trinity._output.show_run <rundir>`.",
    "confidence": "high"
  },
  {
    "id": "S13a-A-04",
    "file": "trinity/_output/simulation_end.py",
    "line": 195,
    "class": "silent-failure",
    "severity": "S2",
    "claim": "The end-code is only honoured if params['SimulationEndCode'].value is a Python int. A numpy integer, a float, or a string silently leaves end_code at UNKNOWN(99) with no warning, so a cleanly terminated run is persisted and reported as a failure.",
    "evidence": "simulation_end.py:192-196 `end_code = SimulationEndCode.UNKNOWN` / `if 'SimulationEndCode' in params:` / `raw = params['SimulationEndCode'].value` / `if isinstance(raw, int):` / `end_code = SimulationEndCode.from_code(raw)`. Then :217-218 `\"exit_code\": int(end_code.code), \"outcome\": str(end_code.outcome)`. Verified: isinstance(np.int64(4), int) is False.",
    "expected": "Coerce via int(raw) inside a try, and log a warning (not silence) when the code cannot be interpreted, so 'unparsable' is distinguishable from a genuine UNKNOWN fate.",
    "failure_scenario": "Any producer that stores the code as np.int64 (e.g. from a numpy-typed comparison) yields metadata {\"exit_code\": 99, \"outcome\": \"unknown\"} for a run that ended with SHELL_COLLAPSED. is_successful_run -> False, show_run prints '✗ ERROR', sweep triage discards a valid run. Only the free-text `detail` field still carries the truth.",
    "repro": "In a Python shell: `import numpy as np; isinstance(np.int64(4), int)` -> False; feed a DescribedItem whose .value is np.int64(4) to write_simulation_end and inspect metadata.json['termination'].",
    "confidence": "medium"
  },
  {
    "id": "S13a-A-05",
    "file": "trinity/_output/show_run.py",
    "line": 489,
    "class": "divergence",
    "severity": "S2",
    "claim": "show_run --quiet clamps the run's exit_code into 1..9 before returning it as the process status, collapsing every distinct failure mode onto 9 — a value that lies inside the project's own 'clean' band and that from_code() resolves to UNKNOWN. The argparse help two lines earlier promises the run's exit_code.",
    "evidence": "show_run.py:489 `return min(max(int(t[\"exit_code\"]), 1), 9)`; help text at show_run.py:455-456 `\"Print only the status line; exit with the run's exit_code (0=success, non-zero=failure).\"`; band definition simulation_end.py:111 `return 0 <= self._code <= 9`; simulation_end.py:124-127 `for member in cls: if member._code == code: return member` / `return cls.UNKNOWN` (no member has code 9).",
    "expected": "Either return a small fixed set of documented statuses (0 clean / 1 error / 2 inspection) or return the real code, but not a clamp that lands inside the success band and maps to no enum member.",
    "failure_scenario": "A shell triage loop `for d in outputs/*; do show_run --quiet $d || echo $?; done` reports 9 for an ERROR_SOLVER(22), an ERROR_INVALID_PARAMS(10) and a VELOCITY_RUNAWAY(50) alike; a script that then does SimulationEndCode.from_code(9) gets UNKNOWN, and one that applies the project's own 0<=code<=9 rule to the status calls it clean.",
    "repro": "Write a metadata.json with {\"termination\":{\"exit_code\":22,\"outcome\":\"error_solver\"}}, run `python -m trinity._output.show_run --quiet <dir>; echo $?` -> 9.",
    "confidence": "high"
  },
  {
    "id": "S13a-A-06",
    "file": "trinity/_output/trinity_reader.py",
    "line": 543,
    "class": "divergence",
    "severity": "S2",
    "claim": "Two classifiers of the same quantity disagree. is_successful_run narrows the 4-band enum to a boolean at 0..9, so codes 50/51 (VELOCITY_RUNAWAY / ENERGY_COLLAPSED) and 99 are rendered '✗ ERROR' by show_run while terminal_prints prints 'Simulation ended (inspection required)' for the identical code. A consumer of is_successful_run cannot tell a solver failure from a physical fate.",
    "evidence": "trinity_reader.py:542-545 `try: return 0 <= int(ec) <= 9 except (TypeError, ValueError): return None`; show_run.py:110-111 `glyph = \"✓\" if is_successful else \"✗\"` / `label = \"SUCCESS\" if is_successful else \"ERROR\"`; vs simulation_end.py:117-119 `def is_inspection_required(self) -> bool: return (50 <= self._code <= 59) or self._code == 99` and terminal_prints.py:222-227 `if end.is_clean(): headline = \"Simulation ended\"` / `elif end.is_error(): headline = \"Simulation FAILED\"` / `else: headline = \"Simulation ended (inspection required)\"`.",
    "expected": "One classifier. Expose the three-way verdict on the reader (e.g. TrinityOutput.outcome_class returning clean/error/inspection) so the terminal transcript and the file-derived summary cannot disagree.",
    "failure_scenario": "A run terminates with VELOCITY_RUNAWAY(50). The console says 'Simulation ended (inspection required)'; `show_run <dir>` on the same directory says '✗ ERROR (velocity_runaway)'. A sweep filter `[r for r in runs if r.is_successful_run]` discards every runaway and every energy-collapse run as if they were solver crashes.",
    "repro": "metadata.json with {\"termination\":{\"exit_code\":50,\"outcome\":\"velocity_runaway\"}} -> compare `show_run <dir>` output with `SimulationEndCode.from_code(50).is_error()` (False) and `.is_inspection_required()` (True).",
    "confidence": "high"
  },
  {
    "id": "S13a-A-07",
    "file": "trinity/_output/trinity_reader.py",
    "line": 204,
    "class": "units",
    "severity": "S3",
    "claim": "PARAM_DOCS labels the whole density family as [cm^-3] while the stored values are internal pc^-3 — a factor of 2.938e55. Every other entry in the same table is explicit about internal units and the needed conversion; the density entries are the sole outliers, and this table is printed to users as the file's unit contract.",
    "evidence": "trinity_reader.py:203-205 `'shell_n0': 'Shell number density at inner edge [cm^-3]',` / `'shell_nMax': 'Maximum shell number density [cm^-3]',` / `'nEdge': 'Number density at shell edge [cm^-3]',` and :209-210 `'nCore': 'Core number density [cm^-3]', 'nISM': 'ISM number density [cm^-3]',`. Contradicted by simulation_end.py:422 `('shell_nMax', 'Shell peak density', 'cm⁻³', INV_CONV.ndens_au2cgs),` (reading the same snapshot key), show_run.py:194-196 `snm_cgs = snm * INV_CONV.ndens_au2cgs` … `f\"({_fmt_or_na(snm, '.3e')} pc⁻³)\"`, show_run.py:141,146 for nCore/nISM, header.py:94 `params['nCore'].value*cvt.ndens_au2cgs`. Contrast the correct style at trinity_reader.py:162 `'Eb': 'Bubble thermal energy [Msun*pc^2/Myr^2] (× INV_CONV.E_au2cgs → erg)'`. Factor from unit_conversions.py:88 `ndens_cgs2au: float = 2.937998946096347e+55`.",
    "expected": "'[pc^-3] (× INV_CONV.ndens_au2cgs → cm^-3)', matching the Eb/Qi/Edot convention already used in the same dict.",
    "failure_scenario": "An analyst runs `TrinityOutput.info(verbose=True)`, reads 'Maximum shell number density [cm^-3]', and plots output.get('shell_nMax') as cm^-3. The plotted densities are wrong by 2.9e55; because the reader is the only unit documentation shipped with the file, nothing downstream contradicts them.",
    "repro": "python -c \"from trinity._output.trinity_reader import read; o=read('<run>/dictionary.jsonl'); o.info(verbose=True)\" and compare the printed unit for shell_nMax with show_run's two-column output for the same run.",
    "confidence": "high"
  },
  {
    "id": "S13a-A-08",
    "file": "trinity/_output/trinity_reader.py",
    "line": 721,
    "class": "numerical",
    "severity": "S2",
    "claim": "The exact-match test passes rtol=1e-10 but leaves np.isclose's default atol=1e-8 in force, so the effective tolerance is a fixed 1e-8 Myr for every t below 0.1 Myr. The intended relative tolerance is inert across the entire early phase.",
    "evidence": "trinity_reader.py:721 `exact_idx = np.where(np.isclose(times, t, rtol=1e-10))[0]`; np.isclose evaluates |a-b| <= atol + rtol*|b| with atol=1e-8 by default. Verified: np.isclose(1.005e-6, 1e-6, rtol=1e-10) -> True; np.isclose(5e-9, 1e-9, rtol=1e-10) -> True.",
    "expected": "np.isclose(times, t, rtol=1e-10, atol=0.0) — the absolute floor must be set explicitly, or the comparison should be on a relative basis only.",
    "failure_scenario": "get_at_time(1e-9) during the earliest bubble-initialisation snapshots returns the snapshot at t=5e-9 as an 'exact' hit — no interpolation, no warning, wrong snapshot by a factor of five in time. Any diagnostic that samples the early phase at specific times gets whichever snapshot happens to be within 1e-8 Myr.",
    "repro": "python -c \"import numpy as np; print(np.isclose(5e-9, 1e-9, rtol=1e-10))\" -> True",
    "confidence": "high"
  },
  {
    "id": "S13a-A-09",
    "file": "trinity/_output/simulation_end.py",
    "line": 524,
    "class": "numerical",
    "severity": "S2",
    "claim": "_compute_change reports 'no change' for values that became NaN or Inf. The array branch clamps a non-finite relative change to 0.0 *before* formatting the string, and the scalar branch's float-equality guard plus NaN-poisoned comparisons make flagged always False for NaN. The comparison table in the termination diagnostic therefore cannot flag the failure mode it exists to catch.",
    "evidence": "simulation_end.py:520-526 `denom = np.maximum(np.abs(old_arr), 1e-30)` / `rel = np.abs(new_arr - old_arr) / denom` / `max_rel = np.nanmax(rel)` / `if not np.isfinite(max_rel): max_rel = 0.0` / `return f\"max Δ={max_rel:.1%}\", max_rel, max_rel > 0.5`. Scalar path: :535 `if old_f == new_f: return \"—\", 0.0, False` (NaN != NaN so it falls through), then :545-553 every comparison against NaN is False, and the caller at :634 does `flagged = rel_change > threshold` -> False. Verified: np.nanmax(np.array([np.nan, np.nan])) -> nan.",
    "expected": "Treat non-finite as maximally significant (flag it) rather than clamping to zero; and test NaN transitions explicitly rather than relying on float equality.",
    "failure_scenario": "The final integration step drives bubble_v_arr to inf. write_termination_debug_report persists `\"change\": \"max Δ=0.0%\", \"flagged\": false` and show_run's diagnostics section prints '(no flagged changes...)'. The separate invalid_values scan (:653-670) covers only snap_new, so a value that was NaN in the penultimate snapshot and recovered leaves no trace at all.",
    "repro": "python -c \"from trinity._output.simulation_end import _compute_change; print(_compute_change([1.0,2.0],[float('inf'),2.0])); print(_compute_change(1.0,float('nan')))\"",
    "confidence": "high"
  },
  {
    "id": "S13a-A-10",
    "file": "trinity/_output/trinity_reader.py",
    "line": 806,
    "class": "state",
    "severity": "S2",
    "claim": "_interpolate_snapshot takes both its key set and its per-key type decision from the single earliest neighbour. Keys absent from that one snapshot are dropped from the result entirely, and a key that is None there returns None even when the other four neighbours hold floats.",
    "evidence": "trinity_reader.py:806 `all_keys = self._snapshots[neighbor_indices[0]].keys()`; :811 `values = [self._snapshots[idx].get(key) for idx in neighbor_indices]`; :814 `first_val = values[0]`; :816-818 `if first_val is None:` / `interpolated_data[key] = None` / `continue`.",
    "expected": "Union the keys across the neighbour window, and infer type from the first non-None value rather than from position 0.",
    "failure_scenario": "A key that only starts being emitted at the transition into the implicit phase is silently missing from every interpolated snapshot requested near that transition, so a time series assembled from get_at_time() has holes that a series assembled from raw snapshots does not.",
    "repro": "Build a TrinityOutput from snapshots [{t:0}, {t:1,'X':5.0}, {t:2,'X':6.0}] and call get_at_time(0.5); 'X' is absent from the returned Snapshot.keys().",
    "confidence": "high"
  },
  {
    "id": "S13a-A-11",
    "file": "trinity/_output/trinity_reader.py",
    "line": 898,
    "class": "silent-failure",
    "severity": "S2",
    "claim": "A blanket `except Exception` silently degrades any failed per-key interpolation to nearest-neighbour, with no logging, while the banner printed a few lines earlier explicitly promises interpolated values and the returned Snapshot is stamped is_interpolated=True.",
    "evidence": "trinity_reader.py:898-901 `except Exception:` / `closest_idx = neighbor_indices[np.argmin(np.abs(neighbor_times - t))]` / `interpolated_data[key] = self._snapshots[closest_idx].get(key)`; banner at :796-800 `print(f\"[TrinityOutput] Time t={t:.6e} Myr not found in snapshots. Interpolating from {len(neighbor_indices)} neighbors ... NOTE: These are interpolated values, not actual simulation output.\")`; stamp at :906-911 `is_interpolated=True`.",
    "expected": "Record per-key provenance (interpolated vs nearest) on the Snapshot, or at minimum log which keys fell back, so a caller can tell the two apart.",
    "failure_scenario": "One snapshot in the window has None for R2 -> `np.array(values, dtype=float)` at :834 raises TypeError -> the handler returns the raw nearest value. The caller sees Snapshot(INTERPOLATED, t=...) and treats R2 as interpolated to the requested time when it is actually the value at a different time. Since t is used as the x-axis, a plot silently mixes provenances.",
    "repro": "Snapshots [{t:0,'R2':1.0},{t:1,'R2':None},{t:2,'R2':3.0}] -> get_at_time(0.5)['R2'] returns a raw snapshot value while __repr__ says INTERPOLATED.",
    "confidence": "high"
  },
  {
    "id": "S13a-A-12",
    "file": "trinity/_output/trinity_reader.py",
    "line": 875,
    "class": "divergence",
    "severity": "S4",
    "claim": "The scalar and array interpolation branches use opposite degenerate-case fallbacks: the scalar branch picks the first *valid* value, the array branch picks element 0 regardless of validity, so the array path can return NaN where the scalar path returns the finite value.",
    "evidence": "scalar branch trinity_reader.py:842-845 `if np.sum(valid_mask) < 2:` / `interpolated_data[key] = y_vals[valid_mask][0] if np.any(valid_mask) else np.nan` / `continue`; array branch :874-875 `if np.sum(valid_mask) < 2:` / `result.append(elem_values[0] if len(elem_values) > 0 else np.nan)`.",
    "expected": "Array branch should mirror the scalar branch: `elem_values[valid_mask][0] if np.any(valid_mask) else np.nan`.",
    "failure_scenario": "A bubble profile array element is NaN in the earliest neighbour and finite in exactly one other. The interpolated profile carries NaN at that index while the equivalent scalar quantity would have carried the finite value — one NaN poisons a subsequent np.trapz or np.max over the profile.",
    "repro": "Snapshots with key 'p' = [nan, 1.0], [1.0, 2.0] at t=0,1 where only one neighbour is finite for index 0; compare against the same data stored as a scalar key.",
    "confidence": "high"
  },
  {
    "id": "S13a-A-13",
    "file": "trinity/_output/trinity_reader.py",
    "line": 833,
    "class": "numerical",
    "severity": "S4",
    "claim": "bool is special-cased to nearest-neighbour but plain int is not, so integer-coded quantities (snap_id, counters, 0/1 flags stored as int) are linearly interpolated to fractional values.",
    "evidence": "trinity_reader.py:827-830 `if isinstance(first_val, bool):` / `closest_idx = ...` / `interpolated_data[key] = self._snapshots[closest_idx].get(key)` / `continue`; then :833 `if isinstance(first_val, (int, float)):` -> :857 `interpolated_data[key] = float(interp_func(t))`. snap_id is injected as an int by _load_json_format at :405 `snap['snap_id'] = int(key)`.",
    "expected": "Route int-typed keys to nearest-neighbour alongside bool, or maintain an explicit list of non-interpolable keys.",
    "failure_scenario": "get_at_time on a .json-format run returns snap_id=17.4, which then fails to index anything and reads as a corrupted identifier in any downstream join.",
    "repro": "Open a .json-format run, call get_at_time(t_between_two_snapshots), inspect snap['snap_id'].",
    "confidence": "medium"
  },
  {
    "id": "S13a-A-14",
    "file": "trinity/_output/simulation_end.py",
    "line": 289,
    "class": "state",
    "severity": "S2",
    "claim": "The final_state key set is a function of the run's terminal values rather than of a schema: non-empty sequences are dropped, empty lists/tuples are kept, and empty ndarrays are dropped by a different mechanism. Two runs of the same model can produce final_state blocks with different keys, and consumers have no way to distinguish 'not applicable' from 'dropped'.",
    "evidence": "simulation_end.py:289-292 `if isinstance(val, (list, tuple)) and len(val) > 0:` / `continue` / `if isinstance(val, np.ndarray) and val.size > 0:` / `continue`; then :303 `json.dumps(val, allow_nan=True)` inside `try` with :304-305 `except (TypeError, ValueError): continue`. Verified: json.dumps([]) succeeds, json.dumps(np.array([])) raises TypeError.",
    "expected": "A fixed key list (or an explicit null for a dropped array), so downstream code can rely on presence and distinguish absent-because-array from absent-because-not-computed.",
    "failure_scenario": "show_run._final_state_section is a chain of `if X is not None` guards (show_run.py:178-231): a run whose terminal shell_nMax happened to be array-valued simply renders a shorter report, indistinguishable from one where the shell was never computed. A sweep table built by reading final_state keys has ragged columns.",
    "repro": "Call _build_final_state_block with a params dict holding one DescribedItem whose value is [] and one whose value is [1.0]; the first survives, the second is dropped.",
    "confidence": "high"
  },
  {
    "id": "S13a-A-15",
    "file": "trinity/_output/simulation_end.py",
    "line": 303,
    "class": "divergence",
    "severity": "S4",
    "claim": "Three different serialisation policies coexist for the same problem in one write path: the final_state guard drops with a bare json.dumps, _NpEncoder converts ndarray->list, and _jsonable stringifies. The final_state guard is stricter than the writer it feeds, so values the writer could have serialised are discarded.",
    "evidence": "simulation_end.py:303 `json.dumps(val, allow_nan=True)` (no encoder) vs _metadata_io.py:54-55 `if isinstance(obj, np.ndarray): return obj.tolist()` used by both the guard at :123 `json.dumps(value, cls=_NpEncoder)` and the write at :92; vs simulation_end.py:695 `return str(val)`.",
    "expected": "One encoder used everywhere, so what passes the guard is exactly what the writer emits.",
    "failure_scenario": "A scalar-shaped 0-d or empty np.ndarray value is dropped from final_state by the guard, while the same value inside termination_debug is written fine — the same quantity is present in one block and absent from the other for no reason a consumer can infer.",
    "repro": "python -c \"import json,numpy as np; print(json.dumps(np.array([])))\" -> TypeError; then the same with cls=_NpEncoder -> '[]'.",
    "confidence": "high"
  },
  {
    "id": "S13a-A-16",
    "file": "trinity/_output/_metadata_io.py",
    "line": 119,
    "class": "state",
    "severity": "S3",
    "claim": "_metadata_version is stamped only when the metadata file is empty or unreadable, so updating a pre-existing v1/v2/v3 file leaves the stale version stamp while adding v4-shaped blocks. Nothing in the slice ever reads the version back or validates it.",
    "evidence": "_metadata_io.py:117-119 `existing = read_metadata(run_dir)` / `if not existing:` / `existing = {\"_metadata_version\": METADATA_VERSION}` — the stamp is inside the branch. run_constants.py:100 `METADATA_VERSION: int = 4`. No read of '_metadata_version' anywhere: run_constants.py:145-146 only excludes it from rehydration, TrinityOutput.metadata (trinity_reader.py:563-571) and read_metadata never inspect it.",
    "expected": "Set existing['_metadata_version'] = METADATA_VERSION unconditionally on every update, and have at least one reader check it and warn on a mismatch.",
    "failure_scenario": "An old run directory is re-terminated (or a v3 metadata.json is updated by a v4 build). The file claims _metadata_version: 3 while carrying v4 blocks; a future migration keyed on the stamp skips or mis-migrates it.",
    "repro": "Write {\"_metadata_version\": 1, \"mCloud\": 1e6} to metadata.json, call update_metadata_atomic(dir, termination={...}), and re-read the file — the stamp is still 1.",
    "confidence": "high"
  },
  {
    "id": "S13a-A-17",
    "file": "trinity/_output/trinity_reader.py",
    "line": 461,
    "class": "state",
    "severity": "S4",
    "claim": "_rehydrate_metadata inserts the *same* object into every snapshot via setdefault, so a metadata list is aliased across all N snapshots; and it uses a denylist, so any non-reserved top-level metadata key becomes a per-snapshot key indistinguishable from a genuine time series.",
    "evidence": "trinity_reader.py:458-461 `run_consts = metadata_keys_to_rehydrate(metadata)` / `for snap in snapshots:` / `for k, v in run_consts.items():` / `snap.setdefault(k, v)`; run_constants.py:145-146 `return {k: v for k, v in metadata.items() if k not in RESERVED_TOP_LEVEL_KEYS}` with RESERVED_TOP_LEVEL_KEYS = {_metadata_version, termination, final_state, termination_debug} at :113-118.",
    "expected": "Deep-copy (or freeze) the injected values, and mark rehydrated keys so TrinityOutput.keys can distinguish run constants from time series.",
    "failure_scenario": "A consumer normalises the initial cloud profile in place — `snap['initial_cloud_n_arr'][:] = ...` on snapshot 0 — and silently changes it in all N snapshots. Separately, any newly added top-level metadata block (not in the reserved set) is broadcast into every snapshot and appears in output.keys as if it were a per-timestep quantity.",
    "repro": "o = read('<run>/dictionary.jsonl'); a = o[0]['initial_cloud_r_arr']; b = o[5]['initial_cloud_r_arr']; a is b -> True.",
    "confidence": "high"
  },
  {
    "id": "S13a-A-18",
    "file": "trinity/_output/simulation_end.py",
    "line": 231,
    "class": "silent-failure",
    "severity": "S2",
    "claim": "write_simulation_end swallows a failure to persist the termination and final_state blocks with a log warning and still returns the exit code, so a completed run whose metadata write failed is byte-for-byte indistinguishable from an aborted run in every downstream reader.",
    "evidence": "simulation_end.py:225-240 `try:` / `update_metadata_atomic(Path(output_dir), termination=termination_block, final_state=final_state_block,)` / `except Exception as e:` / `logging.getLogger(__name__).warning(\"Failed to mirror termination/final_state into metadata.json: %s\", e,)` / `return end_code.code`. The same pattern in _merge_termination_debug at :738-745. Consumer side: show_run.py:108-109 `if termination is None or is_successful is None:` / `return \"Status   : ? UNKNOWN  (no termination block — legacy or aborted run)\"`.",
    "expected": "Re-raise, or return a distinct sentinel, so the caller can fail the run loudly. An output layer that swallows produces a file that looks complete.",
    "failure_scenario": "Disk quota is hit at the moment of termination. The run prints its normal end report, run.py exits 0, and the directory looks like a run that was killed. In a 500-point sweep, the affected runs are silently reclassified as aborted and either re-run or dropped from the analysis.",
    "repro": "chmod a-w on the run directory (so os.replace fails), call write_simulation_end, observe it returns the code and only warns; then run show_run on the directory.",
    "confidence": "high"
  },
  {
    "id": "S13a-A-19",
    "file": "trinity/_output/simulation_end.py",
    "line": 181,
    "class": "silent-failure",
    "severity": "S4",
    "claim": "When params has no path2output, write_simulation_end silently writes metadata.json into the current working directory.",
    "evidence": "simulation_end.py:177-181 `if output_dir is None:` / `if 'path2output' in params:` / `output_dir = params['path2output'].value` / `else:` / `output_dir = '.'`; then :207 `os.makedirs(output_dir, exist_ok=True)`.",
    "expected": "Raise, or at minimum log an error naming the fallback location — a run's audit trail landing in CWD is never intentional.",
    "failure_scenario": "A programmatic invocation that builds params without path2output drops metadata.json into the repository root and, on the next such run, overwrites it — while the actual run directory has no termination block at all.",
    "repro": "write_simulation_end({'SimulationEndCode': item(0)}) with no 'path2output' key, then ls ./metadata.json.",
    "confidence": "high"
  },
  {
    "id": "S13a-A-20",
    "file": "trinity/_output/trinity_reader.py",
    "line": 405,
    "class": "divergence",
    "severity": "S4",
    "claim": "The two accepted input formats are read by non-equivalent readers: the .json path mutates the loaded snapshots by injecting snap_id and then sorts by it; the .jsonl path does neither. Key set and ordering guarantees depend on the file extension.",
    "evidence": "trinity_reader.py:401-411 `for key, snap in data.items():` / `if isinstance(snap, dict):` / `if 'snap_id' not in snap:` / `try: snap['snap_id'] = int(key)` / `except ValueError: pass` … `snapshots.sort(key=lambda s: s.get('snap_id', 0))`; vs :418-423 for jsonl, which appends in file order with no snap_id and no sort. Both are reachable through the same public entry points (find_data_path :1223-1250, resolve_data_input :1300-1335).",
    "expected": "One normalisation step applied to both, or an explicit statement of which ordering guarantee the reader provides.",
    "failure_scenario": "A comparison harness that diffs output.keys() or output.get('snap_id') between a legacy .json run and a current .jsonl run reports a spurious schema difference; and a .json run whose object keys are non-numeric collapses every unparsable snapshot to sort position 0.",
    "repro": "read('a.json').keys vs read('a.jsonl').keys on the same data -> 'snap_id' present only in the first.",
    "confidence": "high"
  },
  {
    "id": "S13a-A-21",
    "file": "trinity/_output/trinity_reader.py",
    "line": 293,
    "class": "silent-failure",
    "severity": "S4",
    "claim": "The reader turns every missing key into None or 0.0 rather than an error: Snapshot.__getitem__ uses .get with no default sentinel, t_now defaults to 0.0, t_min/t_max default t_now to 0, and TrinityOutput.get fills absent keys with None inside an object array.",
    "evidence": "trinity_reader.py:293 `return self.data.get(key)`; :306 `return self.data.get('t_now', 0.0)`; :661 `return min(s.get('t_now', 0) for s in self._snapshots)`; :666 `return max(s.get('t_now', 0) for s in self._snapshots)`; :684 `values = [s.get(key) for s in self._snapshots]`.",
    "expected": "__getitem__ should raise KeyError (that is what [] means; .get already exists at :295-297 for the tolerant case), and t_min/t_max should not manufacture a zero.",
    "failure_scenario": "A typo — snap['shell_nmax'] instead of snap['shell_nMax'] — returns None, is multiplied by a conversion factor, raises TypeError far from the typo, or is plotted as an empty series. One snapshot missing t_now silently reports the run's time range as starting at 0.",
    "repro": "o[0]['no_such_key'] -> None instead of KeyError.",
    "confidence": "high"
  },
  {
    "id": "S13a-A-22",
    "file": "trinity/_output/header.py",
    "line": 91,
    "class": "divergence",
    "severity": "S3",
    "claim": "The startup banner computes the quantity it labels 'log_mCloud' as log10(mCloud/(1-sfe)), while show_run derives the cluster mass as mCloud*sfe. The two encode incompatible interpretations of what the stored mCloud is. Line 91 is also the only line in show_param that omits .value.",
    "evidence": "header.py:91 `print(f\"\\tlog_mCloud: {np.log10(params['mCloud']/(1-params['sfe']))} Msun\")` — compare the surrounding lines :90 `params['model_name'].value`, :92 `params['sfe'].value`, :93 `params['ZCloud'].value`, :94 `params['nCore'].value*cvt.ndens_au2cgs`, :95 `params['dens_profile'].value`. vs show_run.py:134-136 `mCluster = md.get(\"mCluster\")` / `if mCluster is None and \"mCloud\" in md and \"sfe\" in md:` / `mCluster = md[\"mCloud\"] * md[\"sfe\"]`. PARAM_DOCS calls it 'Initial cloud mass [Msun]' (trinity_reader.py:143).",
    "expected": "One convention for mCloud across the banner, the metadata, and show_run, with .value used consistently.",
    "failure_scenario": "The value printed at run start as 'log_mCloud' is not log10 of the mCloud written to metadata.json and read back by show_run/TrinityOutput; a user comparing the console banner against the run summary sees two different cloud masses for the same run. Whichever of the two formulas is wrong, the derived mCluster in show_run is off by 1/(1-sfe).",
    "repro": "Run any model and compare the banner's log_mCloud with log10(metadata['mCloud']) and with show_run's mCluster row.",
    "confidence": "medium"
  },
  {
    "id": "S13a-A-23",
    "file": "trinity/_output/trinity_reader.py",
    "line": 1506,
    "class": "state",
    "severity": "S2",
    "claim": "organize_simulations_for_grid keys its grid on (mCloud, sfe) only, so when ndens_filter is None runs at different ambient densities silently overwrite one another while the returned ndens_list still advertises all of them.",
    "evidence": "trinity_reader.py:1506 `grid[(mCloud, sfe)] = sim_file` inside the loop that has just added ndens to ndens_set at :1505; the return at :1513-1519 includes `'ndens': ndens_list[0] if len(ndens_list) == 1 else None,` and `'ndens_list': ndens_list`. The keys are the raw regex strings from :1412-1416, so '1e6' and '1.0e6' are distinct cells for the same mass.",
    "expected": "Key on (mCloud, sfe, ndens), or refuse to build a grid when more than one ndens survives the filters.",
    "failure_scenario": "A sweep over mCloud x sfe x ndens is passed to organize_simulations_for_grid without ndens_filter. Each grid cell silently keeps whichever density came last in sorted(sim_files) order, so a published parameter-grid figure mixes densities cell by cell while its caption reports the full ndens_list.",
    "repro": "Point it at a directory with m1e6_sfe10_n100 and m1e6_sfe10_n1000; len(result['grid']) == 1 while result['ndens_list'] has two entries.",
    "confidence": "medium"
  },
  {
    "id": "S13a-A-24",
    "file": "trinity/_output/trinity_reader.py",
    "line": 1406,
    "class": "divergence",
    "severity": "S4",
    "claim": "parse_simulation_params requires an exponent-form mass and an integer SFE; unparsable folder names are warned about in one consumer and silently skipped in two others.",
    "evidence": "trinity_reader.py:1406-1410 `match = re.search(r'm?(\\d+\\.?\\d*e\\d+)_sfe(\\d+)_n(\\d+\\.?\\d*(?:e\\d+)?)', folder_name, re.IGNORECASE)` — `e\\d+` admits no sign, `sfe(\\d+)` admits no decimal point. Consumers: :1483-1485 `if params is None:` / `print(f\"Warning: Could not parse folder name: {folder_name}\")` / `continue` vs :1550-1554 in info_simulations and :1437-1439 in get_unique_ndens, both of which just skip.",
    "expected": "One shared failure policy — either all warn or the parser returns a diagnosable result — and a regex that accepts the decimal SFE and signed-exponent forms actually produced by the sweep namer.",
    "failure_scenario": "A sweep directory named with decimal SFEs (…_sfe0.05_n…) makes info_simulations return count>0 but empty mCloud/sfe/ndens lists, with nothing printed; the user reads that as 'no simulations found'.",
    "repro": "python -c \"from trinity._output.trinity_reader import parse_simulation_params as p; print(p('m1e6_sfe0.05_n100'))\" -> None",
    "confidence": "medium"
  },
  {
    "id": "S13a-A-25",
    "file": "trinity/_output/trinity_reader.py",
    "line": 1044,
    "class": "deadcode",
    "severity": "S4",
    "claim": "Unreachable and unused code (flagged only, not proposed for deletion): the pandas ImportError guard in to_dataframe cannot fire because pandas is imported at module scope; the float/NaN branch in _print_parameters is a no-op; DROPPED_IN_V2 has no reader; find_data_file has no in-slice caller; Snapshot.interpolation_time is written and never read.",
    "evidence": "trinity_reader.py:132 `import pandas as pd` (module scope) vs :1044-1047 `try:` / `import pandas as pd` / `except ImportError:` / `raise ImportError(\"pandas is required for to_dataframe()\")`. trinity_reader.py:1015 `stype = type(sample).__name__ if sample is not None else '?'` followed by :1018-1019 `elif isinstance(sample, float) and not np.isnan(sample):` / `stype = f'float'` — identical result. run_constants.py:88-92 `DROPPED_IN_V2: frozenset[str] = frozenset({...})` with no reference. trinity_reader.py:1133 `def find_data_file(...)` with no caller. trinity_reader.py:289 `interpolation_time: Optional[float] = None`, set at :910, never read.",
    "expected": "n/a — flagged per project rule. Note the module-scope scipy import at :133 likewise makes scipy a hard dependency of merely reading a file.",
    "failure_scenario": "No runtime failure. The pandas guard gives a false impression that the reader degrades gracefully without pandas; in fact `import trinity._output.trinity_reader` fails outright on a pandas-less install.",
    "repro": "grep -n 'import pandas' trinity/_output/trinity_reader.py -> lines 132 and 1045.",
    "confidence": "high"
  },
  {
    "id": "S13a-A-26",
    "file": "trinity/_output/simulation_end.py",
    "line": 471,
    "class": "silent-failure",
    "severity": "S4",
    "claim": "_load_last_snapshots reads the entire trajectory into memory to obtain two records, and silently discards a truncated final line so that snapshot_count under-reports and the comparison table is omitted — indistinguishable from a run that genuinely produced one snapshot.",
    "evidence": "simulation_end.py:471-472 `with open(jsonl_path, 'r', encoding='utf-8') as f:` / `lines = f.readlines()`; :475-483 `for line in lines[-n:]:` … `except json.JSONDecodeError:` / `continue`; consumer at :597 `\"snapshot_count\": len(snapshots),` and :605 `snap_old = snapshots[-2] if len(snapshots) >= 2 else None` which suppresses the whole comparison block at :620.",
    "expected": "Seek from the end for the last two complete lines, and record in the debug block that a partial line was discarded.",
    "failure_scenario": "A run terminated after a partial line was flushed produces termination_debug with snapshot_count 1, no comparison rows and no warnings; show_run prints '(no flagged changes, no NaN/Inf, all sanity checks passed)' for a run whose last write was interrupted. On a long run, the readlines() also pulls the full multi-hundred-MB trajectory into memory at the worst possible moment.",
    "repro": "truncate -s -20 dictionary.jsonl, then call write_termination_debug_report(dir) and inspect metadata.json['termination_debug'].",
    "confidence": "high"
  },
  {
    "id": "S13a-A-27",
    "file": "trinity/_output/_metadata_io.py",
    "line": 90,
    "class": "other",
    "severity": "S4",
    "claim": "The 'atomic' metadata write uses a fixed temp filename and no fsync, and update_metadata_atomic's read-modify-write is not atomic across processes — two writers to the same run directory collide on metadata.json.tmp and one block is lost.",
    "evidence": "_metadata_io.py:89-93 `path = Path(run_dir) / METADATA_FILENAME` / `tmp = path.with_suffix(path.suffix + \".tmp\")` / `with open(tmp, \"w\", encoding=\"utf-8\") as f:` / `json.dump(payload, f, cls=_NpEncoder, indent=2, sort_keys=False)` / `os.replace(tmp, path)` — no f.flush()/os.fsync(), no unique suffix, no lock. update_metadata_atomic reads at :117 and writes at :132 with no interlock.",
    "expected": "A unique temp name (tempfile.mkstemp in the same directory), fsync before replace, and a lock or single-writer discipline if concurrent updates are possible.",
    "failure_scenario": "Two processes finishing in the same run directory (or a debug-report writer racing a termination writer) both read the old metadata and both write; the later os.replace wins and one block vanishes with no error. On a power loss after os.replace, metadata.json can be zero-length because the contents were never fsynced.",
    "repro": "Concurrently call update_metadata_atomic(dir, termination={...}) and update_metadata_atomic(dir, termination_debug={...}) from two processes; one block is missing from the result.",
    "confidence": "medium"
  },
  {
    "id": "S13a-A-28",
    "file": "trinity/_output/show_run.py",
    "line": 108,
    "class": "divergence",
    "severity": "S4",
    "claim": "Miscellaneous reporting divergences: the status line blames a 'missing termination block' when the block exists but its exit_code is unparsable; the three formatters of the same state quantities use different display precisions; get_at_time raises out-of-range in one mode and silently clamps in the other; and TrinityOutput.phases returns a set-ordered list.",
    "evidence": "show_run.py:108-109 `if termination is None or is_successful is None:` / `return \"Status   : ? UNKNOWN  (no termination block — legacy or aborted run)\"` while is_successful is also None when int(ec) fails (trinity_reader.py:544-545). Precision: terminal_prints.py:132 `(\"t\", \"t_now\", 1.0, \".6f\", \"Myr\")` vs show_run.py:173 `t_label = f\"  [t = {_fmt_or_na(t_now, '.3f')} Myr]\"` vs trinity_reader.py:964 `f\"  Time range:    [{self.t_min:.4e}, {self.t_max:.4e}] Myr\"`; R2 `.4f` (terminal_prints.py:137) vs `.3f` (show_run.py:180). Mode divergence: trinity_reader.py:766-770 raises ValueError vs :729-737 returning the endpoint. trinity_reader.py:656 `return list(set(s.get('current_phase', 'unknown') for s in self._snapshots))`.",
    "expected": "Distinguish 'no block' from 'unparsable block' in the message; one display-precision table; consistent out-of-range policy; a sorted phases property (info() already sorts at :970).",
    "failure_scenario": "A run whose termination block is present but malformed is reported as legacy/aborted, sending the user looking for the wrong problem. A final time of 0.0004 Myr prints as '0.000 Myr' in show_run while the console showed '0.000400 Myr'. output.phases iterated in a report gives a different order on each invocation because of string hash randomisation.",
    "repro": "metadata.json with {\"termination\":{\"exit_code\":\"four\"}} -> show_run prints 'no termination block'. python -c \"import numpy as np\" not needed; PYTHONHASHSEED=1 vs 2 changes list(set(['a','b','c'])) order.",
    "confidence": "high"
  }
]
```
