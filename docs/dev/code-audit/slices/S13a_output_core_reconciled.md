# S13a output core — reconciled (A vs B)

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

## What this is, and what it is not

I read exactly two files:

- `scratchpad/raw/S13a_output_core_lensA.md` — **Lens A**, written from a comment- and
  docstring-stripped copy of the eight slice files. It reports what the code *does* and never saw a
  comment.
- `scratchpad/raw/S13a_output_core_lensB.md` — **Lens B**, written from the extracted comments and
  docstrings only. It reports what the code *claims* and never saw a line of code.

**I did not read any source.** No file under `trinity/`, `test/`, `docs/dev/`, `param/`, no stripped
`code/` copy, no `prose.md`, no `signatures.md`, no other slice's report. Every verdict below is a
diff of two accounts, not an inspection. Where the two accounts do not settle a question, I say so
and put it in **Open questions** as a named one-step lookup rather than guessing.

**Limits this imposes.** I cannot confirm a line number, cannot check whether a fix already exists
elsewhere in the tree, and cannot adjudicate any claim that depends on code neither lens saw. Lens A
declared one blind spot that propagates into this report: the *writer* of `dictionary.jsonl`, the
`DescribedItem`/`DescribedDict` container, and `trinity._input.registry` (source of `RUN_CONST_KEYS`
/ `METADATA_EXCLUDE`) are all outside the slice. Anything whose trigger lives in a producer — most
importantly R-04 — carries lowered confidence for that reason, in both lenses and here.

**The declared asymmetry.** Lens B flagged that `trinity_reader.py:136-278` — the per-key
`PARAM_DOCS` table — is a **code literal**, so prose extraction captured only its 18 group-header
comments (`# Model info`, `# Forces`, `# Bubble arrays (radial profiles)`, …). Lens A saw the whole
table. Therefore **"A names a key B never mentions" is expected across the entire per-key schema and
is not evidence of anything.** I have manufactured no A-only or B-only finding out of that gap. It
applies specifically to: the per-key documentation of every snapshot quantity, the enum member names
and outcome tokens (B could see the band comments but no member), and the `# Forces` group's unit
annotations. Each place where the asymmetry bites is marked **[asym]** in the correspondence table.

One consequence worth stating positively: the asymmetry does **not** neutralise R-14 (density units).
There, B is not silent — B affirmatively transcribed *four separate doc sites* stating the opposite
unit for the same keys. That is a doc-vs-doc conflict the method can adjudicate, not an extraction gap.

**Infra tier — no Lens C.** Nothing here was checked against a third reading.

---

## Correspondence table

Axis: **A≡B** corroborated (ranked highest) · **A≠B** doc-drift · **A-only** undocumented behaviour ·
**B-only** unimplemented or stale claim.

| # | Claim | Lens | Axis | Verdict |
|---|---|---|---|---|
| R-01 | `is_successful_run` narrows the 4-band enum to a 0–9 boolean; `show_run` renders the result "✗ ERROR" while `terminal_prints` renders the same code "inspection required" | A (code); B (doc §3.4 documents the narrowing) | A≡B on the narrowing; **A-only** on the render divergence | Confirmed. The narrowing is *documented*; the cross-consumer contradiction is not, and no doc acknowledges the other renderer |
| R-02 | `--quiet` clamps every failure to shell status 9; 9 is inside the documented clean band and `from_code(9)` → UNKNOWN; status 1 also means "directory not found"; no status for `is_successful_run is None` | A (`show_run.py:489`); B (B-13, B-14 — doc states the clamp *and* a false POSIX rationale) | **A≡B** | Strongest corroboration in the slice. Code does exactly what the doc says; the doc's own rationale is wrong; both lenses independently spotted that 9 is a clean code |
| R-03 | `write_simulation_end` swallows a metadata-write failure and still returns the exit code, so a completed run is indistinguishable from an aborted one | A (`simulation_end.py:225-240`); B (B-03 — comment: "the metadata write failing should not bring the run down"; reader doc's `None` branch names only "legacy run / crash before write fired") | **A≡B** | Confirmed from both directions. The swallow is deliberate and documented at the writer; the reader's documented interpretation of `None` omits it |
| R-04 | `isinstance(raw, int)` gate downgrades a non-`int` end code to UNKNOWN(99) with no warning | A (code); B transcribes `simulation_end.py:189` "set at the source … as the integer `.code` so it survives JSON serialization. If a site forgot to set it, fall back to UNKNOWN" | **A≠B** on trigger set | Doc names one trigger (site forgot); code has two (forgot, *or* set it as `np.int64`/float/str). Trigger probability depends on out-of-slice producers — medium confidence |
| R-05 | `update_metadata_atomic` replaces a corrupt/unreadable `metadata.json` with a stub, destroying every run constant | A (`_metadata_io.py:117-119` + `read_metadata` returning `{}` on `OSError`/`JSONDecodeError`); B (`_metadata_io.py:97` documents the stub branch **only** for "does not exist"; `_metadata_io.py:60` warns "Callers that need to distinguish absent-vs-corrupt should check existence themselves") | **A≡B**, and the doc convicts itself | Confirmed. The module documents the absent-vs-corrupt hazard and tells callers to handle it; its own internal caller does not |
| R-06 | `Infinity`/`NaN` written as bare tokens; file is not RFC-8259 JSON | A (four `float('inf')` producers reaching `rel_change`); B (B-11 — "technically non-standard JSON", `allow_nan=True`, "same compromise the snapshot writer makes") | **A≡B** | Confirmed and *escalated*: B shows it is a knowing, documented compromise; A shows `Infinity` (not just `NaN`) is produced **at termination, on ordinary paths** |
| R-07 | `_compute_change` reports "no change / not flagged" for a value that went finite→NaN or finite→±Inf | A (clamp to `0.0` before formatting; float-equality guard fails on NaN); B (§3.7 documents "50% change flagged by default", "Phase changes are **always** flagged", "Collapse status changes **always** flagged"; B-11 documents NaN as *expected* in `cool_beta`) | **A≡B** (doc contract vs code) | Confirmed. A documented change detector is blind to the one class of change it most needs to catch, on quantities the docs say are routinely NaN |
| R-08 | `final_state` membership is a function of the run's terminal *values*, not of a schema | A (empty list survives, empty ndarray dropped, non-empty sequences skipped, bare `json.dumps` guard); B (B-08 — three non-identical descriptions; B-10 — two layers silently drop, WARNING only, no reader-side doc) | **A≡B** | Confirmed. Doc side admits inconsistent membership rules and silent drops; code side shows the key set varies run to run |
| R-09 | Interpolation silently degrades to nearest-neighbour with no provenance, while the docstring promises interpolated values and the object is stamped `is_interpolated=True` | A (`except Exception` → closest, no logging; banner promises interpolation); B (B-17 — Returns block says "Interpolated snapshot with is_interpolated=True", comments document ≥7 fallback paths incl. "# If interpolation fails for any reason, use closest value") | **A≡B** | Confirmed. B found the comments describing exactly the handler A found. The fallbacks are documented *in comments only*; the public docstring is unqualified |
| R-10 | `_interpolate_snapshot` takes its key set **and** its per-key type decision from the earliest neighbour only | A | **A-only** (not [asym] — this is control flow, not a table) | Undocumented behaviour. B's fallback list does not include it |
| R-11 | `np.isclose(times, t, rtol=1e-10)` leaves `atol=1e-8` in force, so the tightened tolerance is inert below 0.1 Myr | A | **A-only** | No documented tolerance contract exists for the exact-match path — B transcribed none. Undocumented behaviour with a verified numeric consequence |
| R-12 | `_load_jsonl_format` has no per-line guard; `show_run` tracebacks on a run killed mid-write | A (contrast: `_load_last_snapshots` in the same slice *does* tolerate it) | **A-only** | Undocumented. B's `show_run` exit table has no code for "data file unreadable" (B-13), corroborating that the case was never contemplated |
| R-13 | `organize_simulations_for_grid` keys on `(mCloud, sfe)` only; runs at different densities overwrite each other while `ndens_list` still reports both | A | **A-only** | Undocumented behaviour, medium confidence (regex/keying details) |
| R-14 | `PARAM_DOCS` labels `nCore`/`nISM`/`shell_n0`/`shell_nMax`/`nEdge` as `[cm^-3]`; stored values are internal pc⁻³ (factor 2.938e55) | A (the literal); B (four *other* sites: `show_run.py:136`, `show_run.py:191`, `simulation_end.py:131`, and the `final_state` "INTERNAL units (pc/Myr, pc⁻³)" invariant repeated 4×) | **A≠B (doc vs doc)** | Confirmed against `PARAM_DOCS`. It is 1-vs-N against the slice's most repeated documented invariant; the code (per A) converts correctly everywhere. See **Units** |
| R-15 | The four force keys carry no unit at all in the module docstring's "Key Parameters" contract | B | **B-only** | Stale/incomplete claim. A's silence is **[asym]**-adjacent: A saw `PARAM_DOCS`' `# Forces` group but reported only the density rows as outliers. Needs lookup (OQ-2) |
| R-16 | `initial_cloud_profile` may return an all-zeros `m_arr` on legacy v1 input; the justifying comment ("any future consumer that *does* need m will fall through to the v2 scalar path") cannot hold if the fast path already returned | B | **B-only** | A saw only that the v2 path raises `KeyError`; it did not describe the v1 fast path. Unresolved — OQ-6 |
| R-17 | Reader docstring promises `model_name`; `read_simulation_end` docstring promises `model`; comment says "Legacy callers expect 'model'" | B | **B-only** | **A does not settle it** — A never names either key. Unresolved — OQ-1 |
| R-18 | "A partial write can **never** leave a corrupt file" vs no `fsync`, a fixed `.tmp` name, and a cross-process read-modify-write with no lock | A (mechanism); B (the absolute "never" guarantee, stated twice) | **A≡B** | Confirmed. B supplies the violated promise A did not know it was violating |
| R-19 | `_metadata_version` is stamped only when the file is absent/unreadable, is never read back, and the module docstring mislabels which blocks belong to which version | A (stamp inside the `if not existing` branch; no reader anywhere); B (B-07 — "surfaced via dedicated `TrinityOutput` properties" but no such property; B-28 — module doc says all three blocks are "v4+" while the history says v3/v3/v4) | **A≡B** | Confirmed. The schema-version field is decorative in both accounts |
| R-20 | "snapshots are saved BEFORE ODE integration, **ensuring** all values correspond to the same timestamp" vs a comment admitting a per-snapshot value "is set AFTER save_snapshot ran"; and `isCollapse` is documented as coming from the final snapshot in one place and from `final_state` in another | B (B-21, B-22) | **B-only** (a self-contradiction *within* the prose) | A could not see it — both halves are comments. Real doc defect with a physical consequence if the two sources disagree. Needs OQ-9 |
| R-21 | `_rehydrate_metadata`'s silent skip is justified as "the normal case for legacy files, where every snapshot already contains the run-constants" — which does not cover a modern run whose `metadata.json` is absent or was stubbed | B (B-23); A (R-05 supplies a *second* uncovered case: corrupt→stubbed) | **A≡B** (B found the gap, A found a new way to fall into it) | Confirmed and widened |
| R-22 | Status line prints "no termination block — legacy or aborted run" when the block exists but its `exit_code` is unparsable | A | **A-only** | Undocumented; B's transcription of the `None` semantics (B-03) shows the doc believes only two causes exist |
| R-23 | No outcome token is documented anywhere except `shell_collapsed`; bands 30–49 and 60–98 are described nowhere | B | **B-only** **[asym]-adjacent** | Genuine doc gap. A had to read the enum body to learn the tokens — which is exactly the point: the *contract* docstring enumerates none of them |
| R-24 | `header.show_param` prints `log_mCloud` as `log10(mCloud/(1-sfe))` while `show_run` derives `mCluster = mCloud*sfe` | A; B corroborates only the `show_run` side ("mCluster is derived from mCloud * sfe") | **A-only**, contradicting a documented convention | The two formulas cannot both be right. Also: that line is the only one in `show_param` omitting `.value`. Medium confidence — OQ-8 |
| R-25 | The module docstring documents `get_at_time` two ways four lines apart — "# Get snapshot closest to a specific time" and "(interpolated by default)" | B | **B-only** | Stale doc. A confirms the default is `'interpolate'` by describing the interpolate path as the one that raises on out-of-range |
| R-26 | Four distinct serialisation policies for one problem; the `final_state` guard is stricter than the writer that consumes its output; `FINAL_STATE_EXCLUDE_ARRAYS` is probably inert; the two encoders are held together by a "MUST stay in sync" comment | A (three policies + the guard/writer mismatch); B (B-09 — named list vs blanket skip; B-12 — duplicated encoder, no enforcement) | **A≡B** | Confirmed. B predicted the list is a no-op if a blanket skip exists; A saw the blanket skip. OQ-5, OQ-11 |
| R-27 | `_rehydrate_metadata` inserts the *same object* into all N snapshots (`setdefault`), and filters by denylist so any future top-level block is broadcast as a per-snapshot quantity | A; B documents the `setdefault` mechanism and its precedence rule but not the aliasing or the denylist | **A-only** on the consequences | Undocumented behaviour |
| R-28 | Every miss becomes `None` or `0.0`: `Snapshot.__getitem__` uses `.get`, `t_now` defaults `0.0`, `t_min`/`t_max` default to `0` | A | **A-only** | Undocumented. B documents the *opposite* discipline for `_phys` in `terminal_prints` ("returns 'n/a' if the key is absent"), so the slice has two policies |
| R-29 | `.json` and `.jsonl` are read by non-equivalent readers — the JSON path injects `snap_id` and sorts, the JSONL path does neither | A; B documents "Sort by snap_id" at the JSON site only | **A-only** on the divergence | Undocumented |
| R-30 | Dead/unreachable: module-scope `pandas` import makes the `to_dataframe` ImportError guard unreachable; the float/NaN branch in `_print_parameters` is a no-op; `DROPPED_IN_V2` unused; `find_data_file` uncalled; `interpolation_time` write-only; `load_output` alias unused in slice | A; B corroborates `DROPPED_IN_V2` ("no reader-side prose says anything consumes it") and `load_output` (B-24, used in an example, introduced nowhere) | **A≡B** on `DROPPED_IN_V2` and `load_output`; A-only on the rest | Flagged only, per project rule 3. **A corrects B-24's failure scenario**: the alias does exist, so the documented example runs; the defect is documentation, not breakage |
| R-31 | `_load_last_snapshots` reads the whole trajectory into memory for two records and silently drops a truncated final line, so `snapshot_count` reports 1 and the entire comparison table is omitted | A | **A-only** | Undocumented; indistinguishable from a genuine one-snapshot run |
| R-32 | Missing `path2output` → `metadata.json` written to the current working directory | A | **A-only** | Undocumented. B transcribes only "Caller is responsible for ensuring `run_dir` exists" |
| R-33 | `bool` is routed to nearest-neighbour but plain `int` is linearly interpolated; and the scalar/array branches use opposite degenerate-case fallbacks | A | **A-only**, **partially contradicted by B** | B transcribes `trinity_reader.py:908` "-1 indicates interpolated" for `snap_id`. If the code stamps `snap_id = -1` after the loop, A's `snap_id=17.4` example is wrong even though the general int point stands. **Demoted** — OQ-4 |
| R-34 | Display precision for the same quantity differs across the three formatters (`t_now` `.6f`/`.3f`/`.4e`; `R2` `.4f`/`.3f`; `Eb` `.4e`/`.3e`); `get_at_time` raises out-of-range in one mode and clamps in the other; `phases` returns a set-ordered list | A | **A-only** | Undocumented hygiene |
| R-35 | `parse_simulation_params` requires exponent-form mass and integer SFE; three consumers apply three different failure policies (warn / silent / silent) | A; B independently documents folder conventions the regex may not admit (`_modified`, `_PL0`, `_BE14`) | **A≡B** (weak) | Both lenses point at folder-name handling as under-specified |
| R-36 | `mu_convert` and `mu_atom` both named as run constants; neither defined, no units, no stated relation | B | **B-only** | Doc gap in the declared bug-class area. See **Units** |
| R-37 | `metadata_keys_to_rehydrate`'s docstring lists three reserved keys; the constant has four (`termination_debug` missing from the docstring) | B; **A settles it** — A read `RESERVED_TOP_LEVEL_KEYS` and confirms four members | **B-only, resolved by A** | Docstring is the stale side; code is correct |
| R-38 | `_final_state_section`'s docstring names two display conversions; comments document four (adds `Pb`→P/k_B, `Eb`→erg) | B; A independently enumerates four converted rows in the code | **A≡B** | Confirmed docstring understatement |
| R-39 | `resolve_data_input` documents three accepted input forms; comments enumerate six cases. `TRINITY_OUTPUT_DIR` precedence unstated | B | **B-only** | A confirms the resolver spans a long multi-case body but did not enumerate. OQ-7 |
| R-40 | `trinity/_output/__init__.py` contributes nothing — no docstring, no re-exports | A ("empty, so nothing is re-exported; every import in the slice is fully qualified"); B ("contributes zero prose") | **A≡B** — exact | Confirmed |
| — | **Negative result:** the three formatters' unit *conversion factors* agree; `cvt.X` and `INV_CONV.X` are the same objects | A (verified via a declared read of `unit_conversions.py`); B corroborates the pc⁻³→cm⁻³ display convention | **A≡B** | **No unit divergence between the terminal transcript and the file summary.** Recorded so a future session need not re-check |

---

## 1. Termination encoding — how a physical fate stops being one

This is the slice's spine, and it is the place where the method paid off best: the two lenses walked
the same chain from opposite ends and met in the middle. B transcribed a contract that is internally
inconsistent *before any code runs*; A found that the code then narrows it three more times on the
way to a consumer.

### 1.1 What the contract says (Lens B)

> "Exit code ranges: - 0-9: Clean physical or intentional terminations (auto-trust) - 10-19:
> Parameter/configuration errors - 20-29: Numerical/runtime errors - 50-59: Inspection required
> (completed, but warrants a human look) - 99: Unknown/unhandled termination (fallback safety net)"
> — B, quoting `simulation_end.py:56`

> "`output.is_successful_run # True | False | None — three-valued # True iff exit_code in [0, 9]`"
> — B, quoting `trinity_reader.py:3`

> "* 1..9 — with `--quiet`, the run's own `exit_code` from the termination block (floored at 1,
> capped at 9 so it fits in POSIX)." — B, quoting `show_run.py:3`

B's own reading of those three, before seeing any code: the band table has holes (30–49, 60–98 named
nowhere); `exit_code in [0, 9]` is Python notation for a two-element membership test, not a range;
the `show_run` exit codes are not a partition (1 means both "directory not found" and "the run's
code, floored"); and the cap sends every non-clean band onto 9, *a value the same enum calls clean*.
B also notes the stated rationale is false — POSIX statuses run 0–255, so 10–99 all fit.

### 1.2 What the code does (Lens A)

> "`return 0 <= int(ec) <= 9`" — A, quoting `trinity_reader.py:543`

> "`return min(max(int(t[\"exit_code\"]), 1), 9)`" — A, quoting `show_run.py:489`

> "`if isinstance(raw, int): end_code = SimulationEndCode.from_code(raw)`" — A, quoting
> `simulation_end.py:192-196`, where `end_code` was initialised to `UNKNOWN`

A settles B's ambiguity in the code's favour on one point: `is_successful_run` implements the
**range**, not the membership test, so B-02's worst case (every code 1–8 reported as failure) does
not happen. The docstring notation is misleading; the code is right. That is a clean S4/S3 resolution
and it is the kind of thing only the paired method produces.

Everything else goes the other way.

### 1.3 The joint picture: six places a fate is destroyed

The orchestrator asked how many distinct ways a *physical fate* becomes indistinguishable from a
*solver failure* or from *success*. Merging both accounts, the answer is **six**, plus one way the
report never prints at all.

| # | Where | Direction of loss | Lens | Documented? |
|---|---|---|---|---|
| 1 | Write — `simulation_end.py:195` | any fate → `{"exit_code": 99, "outcome": "unknown"}` if `.value` is not a Python `int` | A | doc names only "a site forgot to set it" |
| 2 | Write — `simulation_end.py:225-240` | any fate → **no `termination` block at all** → reads as an aborted run | A + B | swallow deliberate & documented; reader's `None` semantics omit it |
| 3 | Read — `trinity_reader.py:543` | four bands → two values: 50, 51, 99 and 22 all become `False` | A | documented as-is |
| 4 | Render — `show_run.py:110` vs `terminal_prints.py:222-227` | the same code prints "✗ ERROR" in the file summary and "Simulation ended (inspection required)" in the transcript | A | neither doc acknowledges the other |
| 5 | Exit — `show_run.py:489` | every non-clean code → shell status **9**, inside the clean band, matching no enum member | A + B | documented, with a false rationale |
| 6 | Render — `show_run.py:108` | a *malformed* `exit_code` → "no termination block — legacy or aborted run" | A | doc says `None` means legacy/crash only |
| 0 | Read — `trinity_reader.py:423` | a truncated final JSONL line → `JSONDecodeError` propagates; `show_run` prints nothing | A | not contemplated (B's exit table has no code for it) |

Read as a pipeline: **#1 and #2 erase the fate at write time**, so nothing downstream can recover it;
**#3 and #5 map physical fates onto the same value as solver failures**; **#5 additionally maps
failures onto a value the project's own `is_clean()` predicate accepts**; **#4 and #6 make two
artefacts of the same run disagree about whether it failed.**

What actually survives all six is exactly two fields: `termination["outcome"]` (a string) and
`termination["detail"]` (free text). Both are degradable — `outcome` becomes `"unknown"` under #1,
and A notes `reason_str = params['SimulationEndReason'].value or 'unknown'`, so an empty reason
string erases `detail` too. And B's finding R-23 is the sting: **no outcome token is documented
anywhere in the slice except `shell_collapsed`.** The only channel that survives the encoding is the
one the contract never enumerates, so a downstream tool that wants "all dissolved runs" has nothing
to match on but free-form text.

### 1.4 The orchestrator's guard-rail, respected

S11 established that `sol.status`/`sol.success` **is** checked by every phase runner. Nothing here
contradicts that, and I make no claim that failure is undetectable at the solver. B's transcription
independently supports it: `simulation_end.py:189` says the code is "set at the source (phase
runners, `main.py`, `phase_events`)", and `terminal_prints.py:206` says `format_end_report` reads
"the numeric `SimulationEndCode` + verbatim `SimulationEndReason` that the phase runners set, so the
actual fate … is visible in `trinity.log` rather than only in `metadata.json`." The failure is
detected correctly and named correctly at the source. It is destroyed **downstream of detection**,
in the output encoding — which is precisely the narrower question this slice was asked.

That also identifies the cheapest mitigation: the transcript (`trinity.log`) is the *only* artefact
that carries the three-way verdict intact, because `terminal_prints` is the only consumer that calls
`is_clean()`/`is_error()`/`is_inspection_required()` rather than the boolean. Everything file-derived
has already been narrowed.

---

## 2. `metadata.json` — validity and destruction

Two separate defects, and the pairing changes the verdict on both.

### 2.1 Destruction outranks sloppiness, because a documented promise is violated

A found the mechanism:

> "`existing = read_metadata(run_dir)` / `if not existing:` / `existing = {\"_metadata_version\":
> METADATA_VERSION}`" — A, `_metadata_io.py:117-119`, with `read_metadata` returning `{}` on **any**
> `OSError` or `JSONDecodeError`

B found the promise, and — decisively — found that the module already knows:

> "`read_metadata` / `metadata` return `{}` on absent-or-malformed; **'Callers that need to
> distinguish absent-vs-corrupt should check existence themselves.'**" — B, `_metadata_io.py:60`

> "If `metadata.json` does not exist (the run terminated before any flush wrote it), a minimal file
> is created … which is the correct semantics for an aborted run." — B, `_metadata_io.py:97`

The documented rationale for the stub branch is **"does not exist."** The code's trigger is
`if not existing`, which fires equally on *absent*, *corrupt*, *unreadable* and *legitimately `{}`*.
So the module states the hazard in one docstring, instructs callers to guard against it, and then its
own internal caller — the one function whose entire job is preserving prior content across a
read-modify-write — does not guard against it. A corrupt `metadata.json` is not repaired, not
side-saved, not left alone: it is **replaced by a stub**, and every run constant it held is gone.

This is the finding I would fix first. It is the only defect in the slice that destroys data that
cannot be reconstructed, and it fires at exactly the moment a run is most likely to have a damaged
file — termination after an unclean event.

Layered on top, B supplies the absolute that A's R-18 violates:

> "a partial write can **never** leave a corrupt file" — B, `simulation_end.py:3` and
> `_metadata_io.py:79`

A's account shows two ways it can: a fixed temp name `metadata.json.tmp` (two writers in one run
directory collide and one block is lost), and no `f.flush()`/`os.fsync()` before `os.replace` (the
rename is atomic for the directory entry; the *contents* are not durable, so a power loss can leave a
zero-length file). Note the honest limit: A rated this S4 not knowing a "never" had been promised. I
have promoted it to S3 on the strength of B's transcription — that is a reconciliation the single
lens could not have made.

### 2.2 `Infinity` in JSON: documented, deliberate, and worse than the doc admits

Here the pairing runs the other way and **de-escalates the framing while keeping the severity**.

A's account reads as a discovery: nothing passes `allow_nan=False`, the file is not valid JSON, and
`jq`/JS/R/Go reject it. B shows it is a knowing compromise, conceded in the source:

> "NaN / non-finite values are kept as-is — `json.dump` emits them as `NaN`, which Python's
> `json.load` reads back faithfully (**technically non-standard JSON**; this is the same compromise
> the snapshot writer makes for fields like `cool_beta` in the momentum phase)." — B,
> `simulation_end.py:244`

> "`json.dump(..., allow_nan=True)` accepts NaN/Inf — same compromise the snapshot writer makes."
> — B, `simulation_end.py:680`

So this is **not** code violating a doc. It is a documented trade-off. What A adds, and what changes
the cost, is *how routine the `Infinity` case is*: `_compute_change` returns `float('inf')` on four
distinct paths — `"NEW"`, `"GONE"`, a string/bool change, and `old_f == 0` — and the caller writes it
straight into the persisted block as `"rel_change"`. A phase change between the last two snapshots, or
`R1`/`Eb` going absent when a run ends in the momentum phase (which the sanity checks *expect*), or
any force being exactly zero in the penultimate snapshot, produces `Infinity` in `metadata.json`.
Termination is precisely when those happen.

The doc's concession is scoped to `NaN` in a physically-motivated field (`cool_beta` in the momentum
phase). The reality is `Infinity` in a *diagnostic* field on ordinary terminations. The correct
reading: the compromise was made for one reason and is being paid for by a different, avoidable one.
`rel_change` has no physical need to be non-finite — a sentinel string, or `null`, costs nothing.

**Is there a schema-version guarantee this violates?** No — and that is itself a finding. B
transcribed `_metadata_version` as "Schema version of `metadata.json`. Increment whenever the layout
changes in a backwards-incompatible way", and B-07 shows no `TrinityOutput` property surfaces it. A
independently confirms nothing in the slice ever *reads* it, and that it is stamped only in the
`if not existing` branch, so updating a v1/v2/v3 file leaves a stale stamp beside v4-shaped blocks.
There is a version field, it is never checked, it can lie, and B-28 shows the module docstring
mislabels which blocks belong to which version anyway. **There is no enforceable schema contract for
either lens's finding to violate** — R-19.

---

## 3. Units

Units are this repo's declared recurring bug class, and this slice produced three separate items plus
one important clean result.

### 3.1 `PARAM_DOCS` density rows — the one that matters

Lens A:

> "`'shell_n0': 'Shell number density at inner edge [cm^-3]'`, `'shell_nMax': 'Maximum shell number
> density [cm^-3]'`, `'nEdge': 'Number density at shell edge [cm^-3]'` … `'nCore': 'Core number
> density [cm^-3]'`, `'nISM': 'ISM number density [cm^-3]'`"

Lens B, from four other locations, none of which is `PARAM_DOCS`:

> `shell_nMax` — "internal **pc⁻³**; displayed cm⁻³" (`show_run.py:191`, `simulation_end.py:131`)
>
> `nCore`/`nISM` — "stored internally in pc⁻³; show them in cm⁻³ (the input unit) via
> `ndens_au2cgs`" (`show_run.py:136`)
>
> `final_state` — "in INTERNAL units (pc/Myr, **pc⁻³**, …)" … "**the most consistently repeated
> invariant in the slice**", stated at `simulation_end.py:3`, `:131`, `:244` and
> `trinity_reader.py:496`

**This is the strongest units verdict the method can produce, and the code-literal asymmetry does not
weaken it.** B is not silent here — B affirmatively documents the opposite unit for the same keys,
from four independent sites, and identifies the internal-units rule as the slice's most-repeated
invariant. `PARAM_DOCS`' density rows stand 1-vs-N against it. Meanwhile A verified the *code*
converts correctly everywhere it displays these keys (`simulation_end.py:422`, `show_run.py:141,146`,
`show_run.py:194-196`, `header.py:94` all multiply by `ndens_au2cgs`). So: the code is right, the
other docs are right, and five rows of one table are wrong by **2.938e55**.

Two aggravating facts. First, `PARAM_DOCS` is not an internal comment — A reports it is printed to the
user by `info(verbose=True)` as the file's unit contract, which makes it the *only unit documentation
shipped with the data*. Second, A checked the neighbours and every other entry is scrupulous:
`'Eb': '… [Msun*pc^2/Myr^2] (× INV_CONV.E_au2cgs → erg)'`, `'Qi': '[1/Myr] (× s2Myr → photons/s)'`.
The density family is the sole outlier in an otherwise careful table — which is exactly the shape that
gets trusted.

By the rubric this is S3: the code is correct, the contract is wrong. Its true cost is not S3-shaped;
see §5.

### 3.2 Forces carry no units at all

Lens B:

> "**Forces:** - F_grav: Gravitational force - F_ram: Ram pressure force (total) - F_HII: HII
> pressure force (outward) - F_rad: Radiation pressure force" — `trinity_reader.py:3`, against, in
> the same docstring, "- Eb: Bubble thermal energy [Msun*pc^2/Myr^2] (internal; × INV_CONV.E_au2cgs
> → erg)"

This is the module docstring's "Key Parameters" block — a *different* table from `PARAM_DOCS`. A did
not report the force rows as unit-less, but A also only claimed that "every other entry **in the same
dict**" (i.e. `PARAM_DOCS`) is scrupulous. The two lenses are describing two tables, so there is no
contradiction and no resolution either. **Open question OQ-2**: does `PARAM_DOCS`' `# Forces` group
(around `trinity_reader.py:191-200`) carry bracketed units? If yes, this is a docstring-only gap
(S4); if no, the force family is undocumented everywhere and it is S3 in the declared bug class.

Related, B-only: **`F_ISM`** is named in `run_constants.py`'s exclusion rationale as a real runtime
key but appears in no documented force list. **[asym]** — A saw `PARAM_DOCS` and did not report it,
but A's silence about an individual key is exactly what the asymmetry says to ignore. OQ-3.

### 3.3 `mu_convert` vs `mu_atom`

B-only, and it belongs in this section rather than buried at S4. Two mean-molecular-weight constants
are named across the slice — `mu_convert` (required for cloud-profile reconstruction) and `mu_atom`
(given as an example run-constant) — with neither defined, no unit stated, and no prose relating
them. Mean-molecular-weight conventions (per particle vs per hydrogen nucleus, with or without
helium) are a standard source of factor-~1.4 errors, and an external reconstruction of `n(r)` or
`M(<r)` that picks the wrong one produces a profile that looks entirely plausible.

### 3.4 The clean result — record it so nobody re-checks

A verified, via a declared read of `trinity/_functions/unit_conversions.py`, that
`cvt.v_au2kms is INV_CONV.v_au2kms`, `cvt.E_au2cgs is INV_CONV.E_au2cgs`,
`cvt.ndens_au2cgs is INV_CONV.ndens_au2cgs`, and that `Pb_au2_KcmInv` has a single module-level
definition imported by all three formatters. **There is no unit divergence between the terminal
transcript, the file summary, and the reader.** B's independent transcription of the display
conventions (`v2`→km/s, `shell_nMax`→cm⁻³, `Pb`→P/k_B, `Eb`→erg) matches. The only unit defects in
this slice are in *labels*, not in *arithmetic*. That is worth stating plainly: the three-formatter
duplication is a maintenance hazard (R-34, R-38) but not currently a numerical one.

---

## 4. `model` vs `model_name` — unresolved, and A does not settle it

B found a three-way inconsistency inside the prose alone:

> `trinity_reader.py:480` — "Mirrors `read_simulation_end`'s return shape:
> `{exit_code, outcome, detail, timestamp, model_name}`"
>
> `simulation_end.py:311` — "Keys: `exit_code`, `outcome`, `detail`, `timestamp`, `model`."
>
> `simulation_end.py:342` (comment) — "# Legacy callers expect 'model' (not 'model_name')"
>
> `simulation_end.py:131` — "mirrors `read_simulation_end()`'s return shape so consumer migrations
> are one-line"

**Lens A never names either key.** A quotes the termination block's construction only as far as
`"exit_code": int(end_code.code), "outcome": str(end_code.outcome)` and does not enumerate the fifth
field anywhere in 835 lines. So the code account is silent and this is genuinely open.

I will state the hypothesis and label it as one. The comment at `:342` sits inside
`read_simulation_end`, which suggests the *file* carries `model_name` and the *function* renames it
to `model` on the way out for legacy callers. If so, both docstrings are locally accurate and the
false claim is the third one — that the two shapes "mirror" each other and migrations are one-line.
That would make this S3 doc-drift with a real trap: a consumer switching from
`read_simulation_end(d)['model']` to `output.termination['model']` on the strength of the "mirrors"
promise gets a `KeyError`, or worse, a silent `.get()` default that labels every figure with a blank
model name.

I cannot confirm the rename direction without reading source. **OQ-1.**

---

## 5. Severity — where the rubric under-serves this slice

Neither lens rated anything S1, and by a strict reading of the rubric that is correct: `S1` is
"results-wrong on configs run today", and this slice integrates nothing. I do not think that strict
reading serves the maintainer, and I want to say why in specific terms rather than as a general
complaint.

**The rubric's S1 is implicitly physics-centric.** It asks whether the *simulation* produced wrong
numbers. This slice is the sole audit trail: every downstream analysis, every figure, and every paper
number is read back through `trinity_reader`, `show_run` and `metadata.json`. A defect that silently
corrupts or mislabels the recorded trajectory yields a wrong published result on a config run today,
with a correct simulation underneath it. That is S1 by *intent* and S2/S3 by *letter*. Three findings
sit in that gap, and I would put them in front of the maintainer as S1-equivalent for triage:

**R-05 (metadata destruction).** The only irreversible defect in the slice. A corrupt
`metadata.json` — a full disk, a killed writer — is silently replaced by a stub at termination, and
`mCloud`, `nCore`, `rCloud`, `dens_profile`, `mu_convert` and the initial cloud arrays are gone. Not
degraded: *gone*, with no marker distinguishing "destroyed" from "never written". Every subsequent
consumer then behaves exactly as it does for a run that never had constants: `_rehydrate_metadata`
injects nothing (R-21), `initial_cloud_profile` raises `KeyError`, `show_run` renders an empty Cloud
block. The simulation was fine; its record is unrecoverable. Rubric says S2 (latent — it needs a
corrupt file first). I would say the latency is illusory, because the corrupting event and the
destroying event are the *same* event: an unclean termination.

**R-14 (density units in `PARAM_DOCS`).** Rubric S3 — code correct, doc wrong. True cost: an analyst
runs `info(verbose=True)`, reads "Maximum shell number density [cm^-3]", and plots
`output.get('shell_nMax')` as cm⁻³. The result is wrong by 2.9e55, the reader is the only unit
documentation shipped with the file, and nothing downstream contradicts it. A doc defect that
reliably converts a correct number into a wrong figure is not "misleading"; it is results-wrong with
an extra step. The 1-vs-N corroboration in §3.1 means there is no ambiguity about which side is wrong.

**R-13 (grid keying).** Rubric S2 — latent, and it needs `ndens_filter=None`. But the failure mode is
a *published parameter-grid figure* in which each cell silently keeps whichever density sorted last,
while the caption's `ndens_list` faithfully reports every density in the sweep. There is no error, no
warning, and the figure looks exactly like the intended one. Medium confidence (A rated it so), but
if it is real it is the single most direct path in this slice from an output-layer bug to a wrong
number in a paper.

Two more where the rubric reads low for the wrong reason:

**R-07 (the NaN detector that cannot see NaN)** is rated S2, but it is the *health check on the audit
trail*. B transcribed the documented promise — "50% change flagged by default", "Phase changes are
**always** flagged", "Collapse status changes **always** flagged" — and A showed that a value going
finite→`inf` is persisted as `"max Δ=0.0%"`, `flagged: false`, because the non-finite clamp runs
*before* the string is formatted; and that a finite→NaN transition never flags at all, because
`old_f == new_f` is false for NaN and every subsequent comparison against NaN is also false. B
separately established that NaN is *expected* in tracked quantities (`cool_beta` in the momentum
phase). So the detector is blind on exactly the data the docs say it will see. `show_run` then prints
"(no flagged changes, no NaN/Inf, all sanity checks passed)" over a run whose last integration step
diverged. A clean bill of health issued to a broken run is worse than no bill of health.

**R-12 (JSONL intolerance)** is rated S2 and reads like robustness hygiene. It is not: `show_run` is
the tool whose entire purpose is inspecting a run's outcome, and it tracebacks on the output of a run
killed mid-write — the exact case it exists for — while an intact `termination` block sits unread in
`metadata.json` two lines away. A points out the contrast is *inside this one slice*:
`_load_last_snapshots` wraps the identical `json.loads` in `except json.JSONDecodeError: continue`.
Two readers of the same file, opposite policies, one of them in the diagnostic tool.

**Where I agree the rubric is right.** The large S4 tail (R-27 through R-40) really is hygiene:
aliasing, precision drift, dead code, set-ordered lists, doc counts off by one. Those should be
batched, not prioritised. And I have resisted inflating the doc-drift findings: R-25, R-37, R-38 and
R-39 are genuine, cheap, and low-stakes.

---

## 6. Demotions and disagreements

Recording these explicitly, because "these two disagree and I cannot tell who is right" is the
honest answer in three places.

- **R-33 demoted (A-13).** A claims integer keys are linearly interpolated and offers
  `snap_id = 17.4` as the example. B transcribes `trinity_reader.py:908` as documenting "-1 indicates
  interpolated" for `snap_id`. If the code stamps `snap_id = -1` on the returned snapshot after the
  per-key loop, A's specific example is wrong even though the general point (plain `int` falls into
  the numeric branch while `bool` is special-cased) stands. Demoted to S4/low confidence pending OQ-4.
- **B-24 corrected by A.** B's failure scenario for `load_output` — "not importable under the name
  they guessed, and the recommended discovery workflow fails at the first line" — is wrong. A
  observed `load_output = read` as a real alias. The documented example runs; the defect is that the
  alias is never introduced in prose. Kept as part of R-30 at S4.
- **B-02 resolved in the code's favour.** The `exit_code in [0, 9]` notation is misleading, but A
  confirms the implementation is `0 <= int(ec) <= 9`. B's failure scenario (a reimplementation
  dropping every code 1–8) is a risk to a *downstream reimplementer*, not a live defect. Folded into
  R-01 at S4 weight.
- **R-37 resolved in the code's favour.** B found the docstring listing three reserved keys; A read
  `RESERVED_TOP_LEVEL_KEYS` and confirms four. Docstring is stale; code is correct. B rated S3, I
  reconcile to S4 — the sole consequence is a maintainer misreading one docstring, and the constant
  next to it is unambiguous.
- **R-04 confidence held at medium.** A's mechanism is solid (`isinstance(np.int64(4), int)` is
  `False`, verified) but the trigger lives in producers neither lens could see, and B's transcription
  of `simulation_end.py:189` — the code is set "as the integer `.code` so it survives JSON
  serialization" — actively suggests the producers *do* cast. That lowers the probability this fires
  today. It does not lower the severity if it ever does, because the failure is silent.
- **R-16 and R-20 stand on B alone.** Both are comment-only findings A structurally could not see.
  Both are specific and internally coherent, but neither is corroborated. Medium confidence.

---

## 7. Open questions — each a one-step lookup

Every item names the file and exactly what to check. None requires running a simulation.

- **OQ-1 — settles R-17.** `trinity/_output/simulation_end.py`: read the `termination_block` dict
  literal (near `:210-220`) and the return statement of `read_simulation_end` (near `:311-345`). Does
  the block carry `model_name` and the function rename it to `model`? If yes, both docstrings are
  locally correct and only the "mirrors … one-line migrations" claim at `:131`/`:480` is false.
- **OQ-2 — settles R-15.** `trinity/_output/trinity_reader.py` `:191-200`, the `# Forces` group of
  `PARAM_DOCS`. Do `F_grav`/`F_ram`/`F_HII`/`F_rad` carry bracketed internal units and a conversion
  factor, in the style of the `Eb` row at `:162`? If no → S3 in the declared bug class.
- **OQ-3 — settles R-15's second half.** Same file, grep `F_ISM` across `:136-278`. Present in
  `PARAM_DOCS` → the module docstring's Forces list is incomplete. Absent → `run_constants.py:3`'s
  rationale names a stale key.
- **OQ-4 — settles R-33.** `trinity/_output/trinity_reader.py` `:902-911`. Is `snap_id` overwritten
  to `-1` after the per-key interpolation loop? If yes, A's `snap_id=17.4` example is void.
- **OQ-5 — settles R-26.** `grep -rn FINAL_STATE_EXCLUDE_ARRAYS trinity/`. Is it consulted on any
  path the blanket `isinstance(val, (list, tuple)) and len(val) > 0: continue` skip at
  `simulation_end.py:289` does not already cover? If not, the constant and its ~10-50 KB rationale
  are dead.
- **OQ-6 — settles R-16.** `trinity/_output/trinity_reader.py` `:596-628`. Does the legacy v1 fast
  path `return` before the v2 scalar reconstruction? If yes, the comment's justification ("any future
  consumer that *does* need m will fall through to the v2 scalar path") is unreachable and an
  all-zeros `m_arr` escapes.
- **OQ-7 — settles R-39.** `trinity/_output/trinity_reader.py` `:1285-1295`. Does `resolve_data_input`
  consult `TRINITY_OUTPUT_DIR` before or after the hardcoded `'outputs'`?
- **OQ-8 — settles R-24.** `trinity/_output/header.py:91` uses `params['mCloud']` with no `.value`,
  unlike every neighbouring line. Check `trinity/_input/dictionary.py` for `DescribedItem.__truediv__`
  / `__rtruediv__` / `__float__`. If absent, that line raises and the banner never printed what it
  claims; if present, the `log10(mCloud/(1-sfe))` convention genuinely conflicts with `show_run`'s
  `mCloud*sfe`.
- **OQ-9 — settles R-20.** `trinity/_output/show_run.py`: at the `_collapse_descriptor` call site
  inside `_resolve_run_status`, is it handed the final snapshot from `dictionary.jsonl` or the
  `final_state` block? The two are documented to skew ("the per-snapshot value is set AFTER
  save_snapshot ran").
- **OQ-10 — sets R-04's real probability.** `grep -rn "SimulationEndCode" trinity/ --include=*.py`
  outside `_output/`. Does any producer assign `.value` from a numpy comparison result or a float?
  This is the only lookup that decides whether R-04 fires today or is purely latent.
- **OQ-11 — settles R-26's second half.** `grep -rn "NpEncoder" test/`. Is there any test asserting
  `trinity._output._metadata_io._NpEncoder` and `trinity._input.dictionary.NpEncoder` agree? The
  comment says they "MUST stay in sync" with no stated enforcement.
- **OQ-12 — settles R-19's scope.** `grep -rn "_metadata_version" trinity/ test/ tools/`. Does
  anything *outside* this slice read the schema version? A established nothing inside it does.

---

```json
[
  {
    "id": "S13a-R-01",
    "file": "trinity/_output/trinity_reader.py",
    "line": 543,
    "class": "state",
    "severity": "S2",
    "claim": "The 4-band termination taxonomy is narrowed to a boolean on the read path (0<=code<=9), so VELOCITY_RUNAWAY(50), ENERGY_COLLAPSED(51) and UNKNOWN(99) are indistinguishable from ERROR_SOLVER(22). show_run renders that boolean as '✗ ERROR' while terminal_prints renders the identical code as 'Simulation ended (inspection required)' — two artefacts of one run disagree about whether it failed. The narrowing is documented; the render divergence is not.",
    "evidence": "A: trinity_reader.py:542-545 `try: return 0 <= int(ec) <= 9 except (TypeError, ValueError): return None`; show_run.py:110-111 `glyph = \"✓\" if is_successful else \"✗\"` / `label = \"SUCCESS\" if is_successful else \"ERROR\"`; vs simulation_end.py:117-119 `is_inspection_required(): return (50 <= self._code <= 59) or self._code == 99` and terminal_prints.py:222-227 three-way headline. B (doc): trinity_reader.py:526 '* True — exit code in [0, 9] (clean termination per SimulationEndCode.is_clean()); * False — exit code outside that range'; simulation_end.py:56 four documented bands. A also settles B's `in [0, 9]` notation ambiguity: the code implements the range, not a membership test.",
    "expected": "One classifier. Expose the three-way verdict on the reader (e.g. TrinityOutput.outcome_class -> clean/error/inspection) so the transcript and the file-derived summary cannot disagree; write the docstring as `0 <= exit_code <= 9`.",
    "failure_scenario": "A run terminates with VELOCITY_RUNAWAY(50). trinity.log says 'Simulation ended (inspection required)'; show_run on the same directory says '✗ ERROR'. A sweep filter `[r for r in runs if r.is_successful_run]` discards every runaway and every energy-collapse run as if they were solver crashes, and the discarded set is exactly the physically interesting one.",
    "repro": "Write metadata.json with {\"termination\":{\"exit_code\":50,\"outcome\":\"velocity_runaway\"}}; compare `show_run <dir>` output against SimulationEndCode.from_code(50).is_error() (False) and .is_inspection_required() (True).",
    "confidence": "high"
  },
  {
    "id": "S13a-R-02",
    "file": "trinity/_output/show_run.py",
    "line": 489,
    "class": "divergence",
    "severity": "S2",
    "claim": "The --quiet process exit status is a lossy, self-contradictory channel, and both lenses reached that independently. Every non-clean code is clamped into [1,9], so parameter errors (10-19), numerical errors (20-29), inspection-required (50-59) and UNKNOWN(99) all arrive as 9 — a value inside the project's own clean band, which from_code() resolves to UNKNOWN and which no enum member owns. Status 1 additionally means 'run directory not found', and there is no status for the documented third state is_successful_run is None. The docstring's stated rationale ('so it fits in POSIX') is false: statuses run 0-255.",
    "evidence": "A: show_run.py:489 `return min(max(int(t[\"exit_code\"]), 1), 9)`; argparse help at :455-456 promises 'exit with the run's exit_code (0=success, non-zero=failure)'; is_clean() at simulation_end.py:111 `return 0 <= self._code <= 9`; from_code at :124-127 returns UNKNOWN for 9 (no member holds it). B (doc): show_run.py:3 '* 1 — run directory not found…  * 1..9 — with --quiet, the run's own exit_code (floored at 1, capped at 9 so it fits in POSIX)'; comment at show_run.py:484 repeats the rationale; simulation_end.py:56 band table. B also notes codes 5-9 are inside the clean band; A confirms no enum member is assigned there.",
    "expected": "Either propagate the real code (10-99 all fit in a single-byte status) or return a small documented partition (0 clean / 2 error / 3 inspection / 1 infrastructure), and drop the false POSIX rationale.",
    "failure_scenario": "`for d in outputs/sweep_*/*/; do show_run --quiet \"$d\" || echo \"BAD: $d\"; done` — the documented triage idiom — reports 9 for a config error, a solver blow-up, a velocity runaway and an unparsable code alike, reports 1 for both a missing directory and a code-1 run, and a script that re-applies the project's own `0 <= code <= 9` rule to the status calls all of them clean.",
    "repro": "metadata.json with {\"termination\":{\"exit_code\":22,\"outcome\":\"error_solver\"}}; `python -m trinity._output.show_run --quiet <dir>; echo $?` -> 9; then `SimulationEndCode.from_code(9)` -> UNKNOWN.",
    "confidence": "high"
  },
  {
    "id": "S13a-R-03",
    "file": "trinity/_output/simulation_end.py",
    "line": 231,
    "class": "silent-failure",
    "severity": "S2",
    "claim": "write_simulation_end swallows a failure to persist the termination and final_state blocks (log warning only) and still returns the exit code, so a completed run whose metadata write failed is byte-for-byte indistinguishable from an aborted run. The writer documents the swallow as deliberate; the reader's documented meaning of is_successful_run is None names only 'legacy run, crash before write_simulation_end fired' and omits this case entirely.",
    "evidence": "A: simulation_end.py:225-240 `try: update_metadata_atomic(...) except Exception as e: logging.getLogger(__name__).warning(\"Failed to mirror termination/final_state into metadata.json: %s\", e,)` / `return end_code.code`; identical pattern in _merge_termination_debug at :736-745; consumer at show_run.py:108-109 prints 'Status : ? UNKNOWN (no termination block — legacy or aborted run)'. B (doc): simulation_end.py:232 comment 'The exit code is the contract of this function; the metadata write failing should not bring the run down'; trinity_reader.py:526 '* None — no termination block (legacy run, crash before write_simulation_end fired)'; simulation_end.py:737 'Merge termination_debug into metadata.json; never raise.'",
    "expected": "Either re-raise / return a distinct sentinel so the caller can fail loudly, or record the write failure somewhere the reader can see it — and list this case in the None branch of the is_successful_run docstring.",
    "failure_scenario": "Disk quota is exhausted at termination in a 500-point sweep. Each affected run prints its normal end report, run.py exits 0, and the directory looks like a run that was killed. Triage reclassifies completed runs as aborted and either re-runs them at full cost or drops them from the published sample.",
    "repro": "chmod a-w the run directory so os.replace fails, call write_simulation_end, observe it returns the code and only warns; then run show_run on the directory.",
    "confidence": "high"
  },
  {
    "id": "S13a-R-04",
    "file": "trinity/_output/simulation_end.py",
    "line": 195,
    "class": "silent-failure",
    "severity": "S2",
    "claim": "The end code is honoured only if params['SimulationEndCode'].value is a Python int. A numpy integer, a float or a string silently leaves end_code at UNKNOWN(99) with no warning, so a cleanly terminated run is persisted and reported as a failure. The docstring names a narrower trigger for the UNKNOWN fallback than the code implements.",
    "evidence": "A: simulation_end.py:192-196 `end_code = SimulationEndCode.UNKNOWN` / `if 'SimulationEndCode' in params:` / `raw = params['SimulationEndCode'].value` / `if isinstance(raw, int):` / `end_code = SimulationEndCode.from_code(raw)`; persisted at :217-218 as {\"exit_code\": 99, \"outcome\": \"unknown\"}; verified isinstance(np.int64(4), int) is False. Also :184-185 `params['SimulationEndReason'].value or 'unknown'` erases an empty reason string. B (doc): simulation_end.py:189 'End code is set at the source (phase runners, main.py, phase_events) as the integer .code so it survives JSON serialization. If a site forgot to set it, fall back to UNKNOWN' — one trigger documented, two implemented.",
    "expected": "Coerce via int(raw) inside a try and log a warning when the code cannot be interpreted, so 'unparsable' is distinguishable from a genuine UNKNOWN fate.",
    "failure_scenario": "A producer stores the code as np.int64 (e.g. from a numpy-typed comparison). metadata.json records {\"exit_code\": 99, \"outcome\": \"unknown\"} for a run that ended SHELL_COLLAPSED; is_successful_run -> False, show_run prints '✗ ERROR', and the only surviving evidence of the real fate is the free-text detail string.",
    "repro": "`python -c \"import numpy as np; print(isinstance(np.int64(4), int))\"` -> False; feed a DescribedItem whose .value is np.int64(4) to write_simulation_end and inspect metadata.json['termination']. Trigger probability: see OQ-10.",
    "confidence": "medium"
  },
  {
    "id": "S13a-R-05",
    "file": "trinity/_output/_metadata_io.py",
    "line": 117,
    "class": "silent-failure",
    "severity": "S2",
    "claim": "update_metadata_atomic destroys all existing metadata when read_metadata fails. read_metadata returns {} on any OSError/JSONDecodeError with only a warning; `if not existing` then replaces the whole file with a stub containing only _metadata_version and the new blocks. The docstring justifies that branch solely for the 'file does not exist' case, and the module's own read_metadata docstring warns that callers needing absent-vs-corrupt must check existence themselves — which this caller does not do.",
    "evidence": "A: _metadata_io.py:73-75 `except (OSError, json.JSONDecodeError) as e: logger.warning(...); return {}`; :117-119 `existing = read_metadata(run_dir)` / `if not existing:` / `existing = {\"_metadata_version\": METADATA_VERSION}`; :132 write_metadata_atomic(run_dir, existing). Downstream: initial_cloud_profile raises KeyError (trinity_reader.py:623-628), _rehydrate_metadata injects nothing, show_run's Cloud block renders empty. B (doc): _metadata_io.py:97 'If metadata.json does not exist (the run terminated before any flush wrote it), a minimal file is created… which is the correct semantics for an aborted run'; _metadata_io.py:60 'Returns {} if the file is absent or malformed… Callers that need to distinguish absent-vs-corrupt should check existence themselves.'",
    "expected": "Distinguish absent from unreadable. On a read error, refuse to overwrite (or side-write metadata.json.corrupt) rather than replacing the run's constants with a stub.",
    "failure_scenario": "metadata.json is truncated by a full disk or a killed writer. At termination, write_simulation_end calls update_metadata_atomic, read_metadata warns and returns {}, and the file is rewritten with only _metadata_version/termination/final_state. mCloud, nCore, rCloud, dens_profile, mu_convert and the initial cloud arrays are permanently gone, with no indication that data was destroyed rather than never written. The corrupting event and the destroying event are the same event.",
    "repro": "`printf '{\"mCloud\": 1e6, ' > <run>/metadata.json`; call write_simulation_end(params, output_dir='<run>'); diff the file against the original.",
    "confidence": "high"
  },
  {
    "id": "S13a-R-06",
    "file": "trinity/_output/_metadata_io.py",
    "line": 92,
    "class": "numerical",
    "severity": "S2",
    "claim": "metadata.json is written with json's default allow_nan=True on every path, so NaN/Infinity/-Infinity are emitted as bare tokens and the file is not RFC-8259 JSON. Both lenses agree this is a knowing, documented compromise — but the docs scope the concession to NaN in a physically-motivated field (cool_beta in the momentum phase), while the code produces Infinity in a purely diagnostic field on ordinary terminations.",
    "evidence": "A: _metadata_io.py:92 `json.dump(payload, f, cls=_NpEncoder, indent=2, sort_keys=False)`; guard at :123 `json.dumps(value, cls=_NpEncoder)`; simulation_end.py:303 `json.dumps(val, allow_nan=True)`. Producers of inf: simulation_end.py:501 ('NEW'), :503 ('GONE'), :508 (string/bool changed), :540 (old_f == 0); persisted at :643 `\"rel_change\": _jsonable(rel_change)` and _jsonable passes floats through. show_run --json (:466-472) prints the file verbatim. B (doc): simulation_end.py:244 'NaN / non-finite values are kept as-is… technically non-standard JSON; this is the same compromise the snapshot writer makes for fields like cool_beta in the momentum phase'; :680 'json.dump(..., allow_nan=True) accepts NaN/Inf'; contrasted with trinity_reader.py:3/:417 advertising .jsonl / JSON Lines.",
    "expected": "Keep the NaN compromise if it is load-bearing for physical fields, but normalise rel_change (a diagnostic with no physical need to be non-finite) to null or a sentinel string; and state in the reader docs that emitted files are Python-JSON, not RFC-8259 JSON.",
    "failure_scenario": "A run ends in the momentum phase where R1/Eb are absent from the last snapshot but present in the penultimate one -> _compute_change returns 'GONE' with rel_change=inf -> metadata.json contains `\"rel_change\": Infinity`. `jq . metadata.json` fails; any JS/Go/R pipeline over the sweep aborts on that run — most likely the pathological run someone is trying to debug. Python readers are unaffected, so the breakage is invisible in-project.",
    "repro": "`python3 -c \"import json;print(json.dumps({'rel_change':float('inf')}))\"` -> {\"rel_change\": Infinity}; then `echo '{\"a\": Infinity}' | jq .` -> parse error.",
    "confidence": "high"
  },
  {
    "id": "S13a-R-07",
    "file": "trinity/_output/simulation_end.py",
    "line": 524,
    "class": "numerical",
    "severity": "S2",
    "claim": "_compute_change reports 'no change, not flagged' for values that became NaN or Inf, contradicting the documented change-detection contract. The array branch clamps a non-finite relative change to 0.0 BEFORE formatting the string; the scalar branch's float-equality guard fails on NaN and every subsequent NaN comparison is False, so flagged is always False. The comparison table cannot flag the failure mode it exists to catch — on quantities the docs say are routinely NaN.",
    "evidence": "A: simulation_end.py:520-526 `max_rel = np.nanmax(rel)` / `if not np.isfinite(max_rel): max_rel = 0.0` / `return f\"max Δ={max_rel:.1%}\", max_rel, max_rel > 0.5`; scalar path :535 `if old_f == new_f: return \"—\", 0.0, False` (NaN falls through), caller at :634 `flagged = rel_change > threshold` -> False; verified np.nanmax(np.array([nan,nan])) -> nan. Partial mitigation at :653-670 (invalid_values) covers only snap_new. B (doc): simulation_end.py:436 '50% change flagged by default', 'Phase changes are always flagged if different', 'Collapse status changes always flagged'; simulation_end.py:244 documents NaN as expected for cool_beta in the momentum phase; terminal_prints.py:144 documents the opposite policy elsewhere ('the literal nan/inf if the value is non-finite').",
    "expected": "Treat non-finite as maximally significant (flag it) rather than clamping to zero; test NaN transitions explicitly instead of relying on float equality; and scan both snapshots for invalid values, not just snap_new.",
    "failure_scenario": "The final integration step drives bubble_v_arr to inf. write_termination_debug_report persists `\"change\": \"max Δ=0.0%\", \"flagged\": false`, and show_run's diagnostics section prints '(no flagged changes, no NaN/Inf, all sanity checks passed)'. A clean bill of health is issued over a diverged run, and a value that was NaN in the penultimate snapshot and recovered leaves no trace at all.",
    "repro": "`python -c \"from trinity._output.simulation_end import _compute_change; print(_compute_change([1.0,2.0],[float('inf'),2.0])); print(_compute_change(1.0,float('nan')))\"`",
    "confidence": "high"
  },
  {
    "id": "S13a-R-08",
    "file": "trinity/_output/simulation_end.py",
    "line": 289,
    "class": "state",
    "severity": "S2",
    "claim": "The final_state key set is a function of the run's terminal values rather than of a schema, and two independent layers silently drop values with only a WARNING. Non-empty sequences are skipped, an empty list survives, an empty ndarray is dropped by a different mechanism (bare json.dumps raises TypeError). Two runs of the same model can produce final_state blocks with different keys, and no reader-side documentation tells a consumer a key may be absent.",
    "evidence": "A: simulation_end.py:289-292 `if isinstance(val, (list, tuple)) and len(val) > 0: continue` / `if isinstance(val, np.ndarray) and val.size > 0: continue`, then :303 `json.dumps(val, allow_nan=True)` inside try with :304-305 `except (TypeError, ValueError): continue`; verified json.dumps([]) succeeds and json.dumps(np.array([])) raises. Consumer show_run._final_state_section (:169-235) is a chain of `if X is not None` guards. B (doc): three non-identical membership descriptions at simulation_end.py:3, :131, :244 (plus :272 and run_constants.py:120); _metadata_io.py:97 'any value that fails json.dumps is logged at WARNING and silently dropped'; simulation_end.py:301 'only include keys whose final value is JSON-friendly'.",
    "expected": "A fixed key list (explicit null for a dropped value), one canonical statement of the membership rule with the higher-level docstrings deferring to the builder, and a recorded _dropped list so absent-by-policy is distinguishable from absent-by-serialization-failure.",
    "failure_scenario": "A run whose terminal shell_nMax happened to be array-valued simply renders a shorter show_run report, indistinguishable from a run where the shell was never computed. A sweep table built by reading final_state keys has ragged columns, and a downstream aggregation emits NaN for a quantity that was silently dropped — discoverable only by reading the log.",
    "repro": "Call _build_final_state_block with a params dict holding one DescribedItem whose value is [] and one whose value is [1.0]; the first survives, the second is dropped.",
    "confidence": "high"
  },
  {
    "id": "S13a-R-09",
    "file": "trinity/_output/trinity_reader.py",
    "line": 898,
    "class": "silent-failure",
    "severity": "S2",
    "claim": "A blanket `except Exception` silently degrades any failed per-key interpolation to nearest-neighbour with no logging, while the banner printed a few lines earlier promises interpolated values and the returned Snapshot is stamped is_interpolated=True. Both lenses found this from opposite sides: A found the handler, B found the comment trail documenting at least seven fallback paths that the public Returns block does not mention.",
    "evidence": "A: trinity_reader.py:898-901 `except Exception:` / `closest_idx = neighbor_indices[np.argmin(np.abs(neighbor_times - t))]` / `interpolated_data[key] = self._snapshots[closest_idx].get(key)`; banner at :796-800 'Interpolating from N neighbors … NOTE: These are interpolated values, not actual simulation output.'; stamp at :906-911. B (doc): trinity_reader.py:746 Returns 'Snapshot — Interpolated snapshot with is_interpolated=True'; comments at :820/:826/:836/:843/:889/:894/:899 — '# Handle strings/phases - use closest', '# Handle booleans - use closest', '# Handle NaN values', '# Not enough points to interpolate', '# Different lengths or empty - use closest', '# Default: use closest value', '# If interpolation fails for any reason, use closest value'.",
    "expected": "Record per-key provenance on the Snapshot (interpolated vs nearest), or at minimum log which keys fell back; and document in the public docstring that an interpolated snapshot is a mixture.",
    "failure_scenario": "One snapshot in the window has None for R2 -> np.array(values, dtype=float) raises TypeError -> the handler returns the raw nearest value. The caller sees Snapshot(INTERPOLATED, t=…) and treats R2 as interpolated to the requested time when it belongs to a different time. A figure sampling every run at a fixed t silently mixes provenances along its own x-axis.",
    "repro": "Snapshots [{t:0,'R2':1.0},{t:1,'R2':None},{t:2,'R2':3.0}] -> get_at_time(0.5)['R2'] returns a raw snapshot value while __repr__ says INTERPOLATED.",
    "confidence": "high"
  },
  {
    "id": "S13a-R-10",
    "file": "trinity/_output/trinity_reader.py",
    "line": 806,
    "class": "state",
    "severity": "S2",
    "claim": "_interpolate_snapshot takes both its key set and its per-key type decision from the single earliest neighbour. Keys absent from that one snapshot are dropped from the result entirely, and a key that is None there returns None even when the other four neighbours hold floats.",
    "evidence": "A: trinity_reader.py:806 `all_keys = self._snapshots[neighbor_indices[0]].keys()`; :811 `values = [self._snapshots[idx].get(key) for idx in neighbor_indices]`; :814 `first_val = values[0]`; :816-818 `if first_val is None:` / `interpolated_data[key] = None` / `continue`. B: not documented — B's transcription of the fallback comments (see R-09) does not include key-set or type selection, so this is undocumented behaviour rather than an extraction gap.",
    "expected": "Union the keys across the neighbour window, and infer type from the first non-None value rather than from position 0.",
    "failure_scenario": "A key that only starts being emitted at the transition into the implicit phase is silently missing from every interpolated snapshot requested near that transition, so a time series assembled via get_at_time has holes that a series assembled from raw snapshots does not.",
    "repro": "Build a TrinityOutput from snapshots [{t:0}, {t:1,'X':5.0}, {t:2,'X':6.0}] and call get_at_time(0.5); 'X' is absent from the returned Snapshot.keys().",
    "confidence": "high"
  },
  {
    "id": "S13a-R-11",
    "file": "trinity/_output/trinity_reader.py",
    "line": 721,
    "class": "numerical",
    "severity": "S2",
    "claim": "The exact-match test passes rtol=1e-10 but leaves np.isclose's default atol=1e-8 in force, so the effective tolerance is a fixed 1e-8 Myr window for every t below 0.1 Myr — the whole early/energy-driven phase. The tightened relative tolerance is inert. No documented tolerance contract exists for this path in either lens.",
    "evidence": "A: trinity_reader.py:721 `exact_idx = np.where(np.isclose(times, t, rtol=1e-10))[0]`; np.isclose evaluates |a-b| <= atol + rtol*|b|. Verified: np.isclose(1.005e-6, 1e-6, rtol=1e-10) -> True; np.isclose(5e-9, 1e-9, rtol=1e-10) -> True (a factor-of-five error reported as an exact hit). B: transcribed no exact-match tolerance anywhere — the only get_at_time documentation concerns mode and n_neighbors.",
    "expected": "np.isclose(times, t, rtol=1e-10, atol=0.0) — the absolute floor must be set explicitly.",
    "failure_scenario": "get_at_time(1e-9) during the earliest bubble-initialisation snapshots returns the snapshot at t=5e-9 as an 'exact' hit — no interpolation, no warning, wrong snapshot by a factor of five in time. Any diagnostic sampling the early phase at specific times silently gets whichever snapshot happens to lie within 1e-8 Myr.",
    "repro": "`python -c \"import numpy as np; print(np.isclose(5e-9, 1e-9, rtol=1e-10))\"` -> True",
    "confidence": "high"
  },
  {
    "id": "S13a-R-12",
    "file": "trinity/_output/trinity_reader.py",
    "line": 423,
    "class": "silent-failure",
    "severity": "S2",
    "claim": "_load_jsonl_format has no per-line error tolerance, so a truncated final line (a run killed mid-append) makes TrinityOutput.open raise JSONDecodeError. show_run guards only FileNotFoundError, so the tool for inspecting a run's outcome tracebacks on exactly the interrupted-run case it exists for — while an intact termination block sits unread in metadata.json. The sibling reader in the same slice does tolerate it.",
    "evidence": "A: trinity_reader.py:419-423 open/for-line/json.loads with no try; :379-380 dispatches .jsonl straight there, bypassing the `except json.JSONDecodeError` that guards only the unknown-suffix branch at :383-386; find_data_path returns .jsonl for a run directory (:1246-1250); show_run.py:271 `except FileNotFoundError:`. Contrast simulation_end.py:479-483 `try: snap = json.loads(line) … except json.JSONDecodeError: continue`. B: show_run.py:3's exit-code table has no entry for an unreadable data file, corroborating that the case was never contemplated.",
    "expected": "Tolerate a trailing partial line (matching _load_last_snapshots) and report how many lines were skipped, or catch JSONDecodeError in _resolve_run_status and fall back to the metadata-only path already implemented at show_run.py:273-292.",
    "failure_scenario": "SLURM sends SIGKILL at the wall-clock limit while a snapshot line is half-written. `python -m trinity._output.show_run <rundir>` raises JSONDecodeError instead of printing the termination block sitting intact in metadata.json. Sweep post-processing that loops show_run over runs dies on the first killed run.",
    "repro": "Copy a good dictionary.jsonl, `truncate -s -20 dictionary.jsonl`, then `python -m trinity._output.show_run <rundir>`.",
    "confidence": "high"
  },
  {
    "id": "S13a-R-13",
    "file": "trinity/_output/trinity_reader.py",
    "line": 1506,
    "class": "state",
    "severity": "S2",
    "claim": "organize_simulations_for_grid keys its grid on (mCloud, sfe) only, so when ndens_filter is None runs at different ambient densities silently overwrite one another while the returned ndens_list still advertises all of them. The keys are the raw regex strings, so '1e6' and '1.0e6' are distinct cells for the same mass.",
    "evidence": "A: trinity_reader.py:1506 `grid[(mCloud, sfe)] = sim_file` inside the loop that has just added ndens to ndens_set at :1505; return at :1513-1519 includes `'ndens': ndens_list[0] if len(ndens_list) == 1 else None` and `'ndens_list': ndens_list`; keys come from the regex at :1412-1416. B: not documented.",
    "expected": "Key on (mCloud, sfe, ndens), or refuse to build a grid when more than one ndens survives the filters.",
    "failure_scenario": "A sweep over mCloud x sfe x ndens is passed to organize_simulations_for_grid without ndens_filter. Each grid cell silently keeps whichever density came last in sorted(sim_files) order, so a published parameter-grid figure mixes densities cell by cell while its caption faithfully reports the full ndens_list. No error, no warning, and the figure looks exactly like the intended one.",
    "repro": "Point it at a directory containing m1e6_sfe10_n100 and m1e6_sfe10_n1000; len(result['grid']) == 1 while result['ndens_list'] has two entries.",
    "confidence": "medium"
  },
  {
    "id": "S13a-R-14",
    "file": "trinity/_output/trinity_reader.py",
    "line": 204,
    "class": "units",
    "severity": "S3",
    "claim": "PARAM_DOCS labels the whole density family (shell_n0, shell_nMax, nEdge, nCore, nISM) as [cm^-3] while the stored values are internal pc^-3 — a factor of 2.938e55. This is not an extraction gap: Lens B, which could not see the table, independently transcribed four OTHER doc sites stating pc^-3 for the same keys, including the slice's most-repeated invariant ('final_state … in INTERNAL units (pc/Myr, pc^-3)'). PARAM_DOCS is 1-vs-N against the slice's own contract, and it is the table printed to users as the file's unit documentation.",
    "evidence": "A: trinity_reader.py:203-205 `'shell_n0': 'Shell number density at inner edge [cm^-3]'`, `'shell_nMax': 'Maximum shell number density [cm^-3]'`, `'nEdge': 'Number density at shell edge [cm^-3]'`; :209-210 `'nCore': 'Core number density [cm^-3]'`, `'nISM': 'ISM number density [cm^-3]'`; printed by info(verbose=True) at :1013/:1020. Code converts correctly: simulation_end.py:422 `('shell_nMax','Shell peak density','cm⁻³',INV_CONV.ndens_au2cgs)`, show_run.py:194-196, show_run.py:141,146, header.py:94. Contrast the correct style at :162 `'Eb': '… (× INV_CONV.E_au2cgs → erg)'`. Factor from unit_conversions.py:88 ndens_cgs2au = 2.937998946096347e+55. B (doc, four independent sites): show_run.py:136 'nCore/nISM stored internally in pc⁻³; show them in cm⁻³ (the input unit) via ndens_au2cgs'; show_run.py:191 and simulation_end.py:131 for shell_nMax; simulation_end.py:3/:131/:244 and trinity_reader.py:496 for the internal-units invariant.",
    "expected": "'[pc^-3] (× INV_CONV.ndens_au2cgs → cm^-3)', matching the Eb/Qi/Edot convention already used in the same dict.",
    "failure_scenario": "An analyst runs TrinityOutput.info(verbose=True), reads 'Maximum shell number density [cm^-3]', and plots output.get('shell_nMax') as cm^-3. The plotted densities are wrong by 2.9e55; because PARAM_DOCS is the only unit documentation shipped with the file, nothing downstream contradicts them.",
    "repro": "`python -c \"from trinity._output.trinity_reader import read; o=read('<run>/dictionary.jsonl'); o.info(verbose=True)\"` and compare the printed unit for shell_nMax with show_run's two-column output for the same run.",
    "confidence": "high"
  },
  {
    "id": "S13a-R-15",
    "file": "trinity/_output/trinity_reader.py",
    "line": 3,
    "class": "units",
    "severity": "S3",
    "claim": "In the module docstring's 'Key Parameters' contract every documented quantity carries a bracketed unit except the four forces, which carry none — in a codebase whose own conventions name units a recurring bug class. Separately, F_ISM is named by run_constants as a real runtime force key but appears in no documented force list.",
    "evidence": "B: trinity_reader.py:3 '**Forces:** - F_grav: Gravitational force - F_ram: Ram pressure force (total) - F_HII: HII pressure force (outward) - F_rad: Radiation pressure force', against the same docstring's '- Eb: Bubble thermal energy [Msun*pc^2/Myr^2] (internal; × INV_CONV.E_au2cgs → erg)' and '- Pb: Bubble pressure [Msun/pc/Myr^2] (internal units)'; run_constants.py:3 names F_ISM among runtime-state keys. A: reported the density rows as the sole outliers within PARAM_DOCS but did not characterise the force rows of either table — this is a different table from PARAM_DOCS, so the two lenses are not in conflict and the question is open (OQ-2, OQ-3).",
    "expected": "An internal-unit annotation for the force keys (the surrounding convention implies Msun·pc/Myr²) plus the CGS conversion factor, in both the module docstring and PARAM_DOCS; and F_ISM either documented or removed from the run_constants rationale.",
    "failure_scenario": "A force-budget figure mixes F_* read raw from snapshots with an externally computed force in dyne or in Msun·pc/Myr²; the ratio is off by the unit factor and the budget does not close, with no unit annotation anywhere to catch it — and the ISM term is missing from the documented list entirely.",
    "repro": "Read the Forces group at trinity_reader.py:3; then check PARAM_DOCS' `# Forces` block near :191-200 for bracketed units and for an F_ISM row.",
    "confidence": "medium"
  },
  {
    "id": "S13a-R-16",
    "file": "trinity/_output/trinity_reader.py",
    "line": 600,
    "class": "silent-failure",
    "severity": "S2",
    "claim": "initial_cloud_profile documents its third return as 'enclosed mass [Msun]', but a comment on the legacy v1 fast path admits m_arr may be filled with zeros, justified by a claim that cannot hold if the fast path has already returned — 'any future consumer that does need m will fall through to the v2 scalar path'.",
    "evidence": "B: trinity_reader.py:574 Returns '(r_arr, n_arr, m_arr) — radius [pc], density [internal pc⁻³], enclosed mass [Msun]'; trinity_reader.py:600 comment 'Fast path: legacy v1 inline arrays… some synthetic test fixtures provide only (r, n), in which case we fill m with zeros — consumers that don't need the enclosed-mass array (e.g. the cloudy ambient extension) can discard it transparently, and any future consumer that *does* need m will fall through to the v2 scalar path because zeros would be obviously wrong.' A: saw only that the v2 path raises KeyError (:623-628) and did not describe the v1 fast path — uncorroborated.",
    "expected": "Either raise / return None for the missing array, or document in the Returns block that m_arr may be all zeros on legacy v1 input. 'Zeros would be obviously wrong' is an unfalsifiable safety argument if nothing routes a zeros result anywhere.",
    "failure_scenario": "An analysis reconstructs the initial cloud profile from a v1 file lacking the m array, gets M(<r) == 0, and computes a gravitational binding energy, escape velocity or free-fall time of zero or infinity without any warning.",
    "repro": "Read trinity_reader.py:574's Returns block against the comment at :600 and check whether the fast path returns before the v2 reconstruction (OQ-6).",
    "confidence": "medium"
  },
  {
    "id": "S13a-R-17",
    "file": "trinity/_output/trinity_reader.py",
    "line": 480,
    "class": "state",
    "severity": "S3",
    "claim": "The reader's termination property claims to mirror read_simulation_end's return shape, but the two docstrings name a different fifth key: model_name vs model, with a comment confirming 'Legacy callers expect model (not model_name)'. UNRESOLVED — Lens A never names either key, so the code account does not settle which name is produced.",
    "evidence": "B: trinity_reader.py:480 'Mirrors read_simulation_end's return shape: {exit_code, outcome, detail, timestamp, model_name}'; simulation_end.py:311 'Keys: exit_code, outcome, detail, timestamp, model.'; simulation_end.py:342 comment '# Legacy callers expect \"model\" (not \"model_name\")'; simulation_end.py:131 'mirrors read_simulation_end()'s return shape so consumer migrations are one-line'. A: quotes the termination block construction only as far as simulation_end.py:217-218 ({\"exit_code\":…, \"outcome\":…}) and names neither key anywhere in the report.",
    "expected": "One key name, or an explicit statement that read_simulation_end renames model_name -> model for legacy callers and that the two shapes are therefore NOT identical — which would make the 'one-line migration' claim the false one.",
    "failure_scenario": "A plotter follows trinity_reader.py:480 and switches from read_simulation_end(run_dir)['model'] to output.termination['model'] (or vice versa) and gets a KeyError, or silently falls into a .get() default and labels every figure with a blank model name.",
    "repro": "OQ-1: read the termination_block dict literal near simulation_end.py:210-220 and the return of read_simulation_end near :311-345.",
    "confidence": "medium"
  },
  {
    "id": "S13a-R-18",
    "file": "trinity/_output/_metadata_io.py",
    "line": 90,
    "class": "other",
    "severity": "S3",
    "claim": "The docs promise twice that 'a partial write can never leave a corrupt file'. The implementation uses a fixed temp filename and no fsync, and update_metadata_atomic's read-modify-write is not atomic across processes — so two writers to one run directory collide on metadata.json.tmp and lose a block, and a power loss after os.replace can leave a zero-length file. Lens A rated this S4 without knowing an absolute had been promised; the promise is what makes it S3.",
    "evidence": "A: _metadata_io.py:89-93 `tmp = path.with_suffix(path.suffix + \".tmp\")` / open / `json.dump(...)` / `os.replace(tmp, path)` — no f.flush(), no os.fsync(), no unique suffix, no lock; update_metadata_atomic reads at :117 and writes at :132 with no interlock; both write_simulation_end and write_termination_debug_report call it, each doing its own read-modify-write. B (doc): simulation_end.py:3 and _metadata_io.py:79 'a partial write can never leave a corrupt file'; _metadata_io.py:79 'if the process dies mid-write, the existing file (if any) survives'.",
    "expected": "A unique temp name (tempfile.mkstemp in the same directory), fsync before replace, and a lock or a documented single-writer discipline — or soften the 'never' to what the mechanism actually guarantees.",
    "failure_scenario": "A debug-report writer races a termination writer in the same run directory; both read the old metadata, both write, the later os.replace wins, and one block vanishes with no error. Separately, on power loss after os.replace the directory entry points at contents that were never fsynced, so metadata.json can be zero-length — precisely the 'corrupt file' the docstring says can never happen, and precisely the input that then triggers R-05.",
    "repro": "Concurrently call update_metadata_atomic(dir, termination={...}) and update_metadata_atomic(dir, termination_debug={...}) from two processes; one block is missing from the result.",
    "confidence": "medium"
  },
  {
    "id": "S13a-R-19",
    "file": "trinity/_output/_metadata_io.py",
    "line": 119,
    "class": "state",
    "severity": "S3",
    "claim": "The schema-version field is decorative in both accounts. _metadata_version is stamped only when the file is empty or unreadable, so updating a pre-existing v1/v2/v3 file leaves a stale stamp beside v4-shaped blocks; nothing in the slice ever reads it back; no TrinityOutput property surfaces it despite a docstring saying reserved entries are 'surfaced via dedicated properties'; and the simulation_end module docstring labels all three blocks '(v4+ schema)' while the version history dates termination and final_state to v3.",
    "evidence": "A: _metadata_io.py:117-119 — the stamp is inside the `if not existing` branch; run_constants.py:100 METADATA_VERSION = 4; no read of '_metadata_version' anywhere (run_constants.py:145-146 only excludes it from rehydration; TrinityOutput.metadata and read_metadata never inspect it). B: run_constants.py:98 'Schema version of metadata.json. Increment whenever the layout changes in a backwards-incompatible way'; run_constants.py:3 'The version field is consumed and discarded by the reader before rehydrate'; run_constants.py:136 'Reserved entries (_metadata_version, termination, final_state) are surfaced via dedicated TrinityOutput properties instead' — no such property is documented; simulation_end.py:3 'All run-end data lands in metadata.json (v4+ schema) as three structured blocks' vs run_constants.py:3's v3/v3/v4 history and trinity_reader.py:480 '(Phase 2, v3+ schema)'.",
    "expected": "Set existing['_metadata_version'] = METADATA_VERSION unconditionally on every update; have at least one reader check it and warn on mismatch; per-block version labels in the module docstring; and drop the claim that _metadata_version has a property.",
    "failure_scenario": "An old run directory is re-terminated, or a v3 metadata.json is updated by a v4 build. The file claims _metadata_version: 3 while carrying v4 blocks. A future migration keyed on the stamp skips or mis-migrates it — and a consumer gating termination reads on version >= 4 per the module docstring treats every v3 run as having no termination data.",
    "repro": "Write {\"_metadata_version\": 1, \"mCloud\": 1e6} to metadata.json, call update_metadata_atomic(dir, termination={...}), re-read — the stamp is still 1.",
    "confidence": "high"
  },
  {
    "id": "S13a-R-20",
    "file": "trinity/_output/trinity_reader.py",
    "line": 3,
    "class": "state",
    "severity": "S3",
    "claim": "The snapshot-consistency guarantee is stated as an absolute — 'snapshots are saved BEFORE ODE integration, ensuring all values in a snapshot correspond to the same timestamp' — while a comment elsewhere in the slice admits at least one per-snapshot value 'is set AFTER save_snapshot ran'. Compounding it, isCollapse is documented as coming from the final snapshot in one place and from final_state in another, and the same comment says the two can be inconsistent.",
    "evidence": "B: trinity_reader.py:3 'As of January 2026, TRINITY snapshots are saved BEFORE ODE integration, ensuring all values in a snapshot correspond to the same timestamp (t_now). This includes: t_now, R2, v2, Eb, T0, feedback properties, shell properties, bubble properties, forces, and beta-delta residuals.'; simulation_end.py:272 comment 'Including either here would leak duplicated (and possibly inconsistent — the per-snapshot value is set AFTER save_snapshot ran) info into final_state.'; show_run.py:66 'Three-state collapse status from the final snapshot.'; simulation_end.py:699 'Shell collapse is a physical outcome (recorded in final_state as isCollapse and in the comparison table)'. A: comment-only material, structurally invisible to Lens A.",
    "expected": "Scope the guarantee to the enumerated field list and name the known exceptions; and name the authoritative source for isCollapse given the documented skew.",
    "failure_scenario": "A run whose collapse flag flips in the final step is rendered 'no' by a consumer reading the last snapshot and 'collapsing' by one reading final_state; two analyses of the same sweep disagree on the collapsed fraction. More broadly, a consumer trusts that every field of the last snapshot — including termination flags — reflects one instant, and reads a stale end-code from the snapshot instead of from the termination block.",
    "repro": "OQ-9: at the _collapse_descriptor call site inside _resolve_run_status, check whether it is handed the final snapshot or the final_state block.",
    "confidence": "medium"
  },
  {
    "id": "S13a-R-21",
    "file": "trinity/_output/trinity_reader.py",
    "line": 429,
    "class": "silent-failure",
    "severity": "S3",
    "claim": "_rehydrate_metadata justifies its silent skip on a missing metadata.json by asserting the only such case is legacy files 'where every snapshot already contains the run-constants'. Two further cases are uncovered: a modern run that aborted before the first flush (whose snapshots have already been stripped of run-constants), and — per R-05 — a modern run whose metadata.json was corrupt and got stubbed. In both, the run constants exist in neither source and the reader reports nothing wrong.",
    "evidence": "B: trinity_reader.py:429 'Silently skips when metadata.json is absent — that's the normal case for files written before this feature landed, where every snapshot already contains the run-constants'; _metadata_io.py:3 'DescribedDict.flush() writes the run-constants on the first flush (typically at run start)'; _metadata_io.py:97 'If metadata.json does not exist (the run terminated before any flush wrote it)…'; run_constants.py:3 'written exactly once per run… and stripped from every per-snapshot dictionary'. A: _rehydrate_metadata at trinity_reader.py:427-461 injects nothing when metadata is empty; R-05 shows a corrupt file becomes an empty-of-constants stub.",
    "expected": "Acknowledge the aborted-modern-run and stubbed-file cases, and warn (once) when a .jsonl whose snapshots lack run-constants is loaded with no metadata to rehydrate from.",
    "failure_scenario": "An early-abort run in a sweep loads without error; output[0].get('mCloud') returns None and output.metadata is {}. A plotting script labels or normalises the curve with a missing or zero cloud mass rather than skipping the run — and because Snapshot.__getitem__ returns None rather than raising (R-28), nothing surfaces the absence.",
    "repro": "Read trinity_reader.py:429 against _metadata_io.py:3/:97 and run_constants.py:3; then load a run directory whose metadata.json has been removed but whose dictionary.jsonl was written by a v2+ build.",
    "confidence": "medium"
  },
  {
    "id": "S13a-R-22",
    "file": "trinity/_output/show_run.py",
    "line": 108,
    "class": "silent-failure",
    "severity": "S3",
    "claim": "The status line prints 'no termination block — legacy or aborted run' whenever is_successful_run is None, which also happens when the block IS present but its exit_code is missing or unparsable. The operator is sent to look for the wrong problem, and the documented meaning of None names only the legacy/crash cause.",
    "evidence": "A: show_run.py:108-109 `if termination is None or is_successful is None:` / `return \"Status   : ? UNKNOWN  (no termination block — legacy or aborted run)\"`; trinity_reader.py:539-545 returns None from the `except (TypeError, ValueError)` around `int(ec)`. B (doc): trinity_reader.py:526 '* None — no termination block (legacy run, crash before write_simulation_end fired)' — the documented cause list omits the malformed-block case.",
    "expected": "Distinguish 'no block' from 'block present, exit_code unparsable' in both the message and the docstring; print the raw value when it cannot be parsed.",
    "failure_scenario": "A run whose termination block was written with a non-numeric exit_code is reported as a legacy or aborted run. The operator searches for a crashed job that does not exist, while the actual defect — a bad code at the write site — is one field away in a file the tool already read.",
    "repro": "metadata.json with {\"termination\":{\"exit_code\":\"four\"}} -> show_run prints 'no termination block'.",
    "confidence": "high"
  },
  {
    "id": "S13a-R-23",
    "file": "trinity/_output/simulation_end.py",
    "line": 56,
    "class": "other",
    "severity": "S3",
    "claim": "The termination contract is documented only as numeric bands. No docstring names a single enum member, a single code->fate mapping, or a single reason string; the sole outcome token appearing anywhere in prose is shell_collapsed, in show_run's rendering logic. Bands 30-49 and 60-98 are described nowhere, and codes 5-9 are documented as clean yet assigned to no member (from_code(9) returns UNKNOWN). This matters because outcome is the only field that survives the encoding chain intact (R-01..R-05).",
    "evidence": "B: simulation_end.py:56 band list plus 'Each member carries (code, outcome_token). The outcome token is mirrored into metadata.json[termination].outcome'; group comments '# Clean (0-9)', '# Parameter errors (10-19)', '# Numerical errors (20-29)', '# Inspection required (50-59)', '# Unknown — also treated as inspection-required'; show_run.py:66 'yes is keyed strictly on the shell_collapsed outcome'. A (code, via the enum body — an [asym] complement): members are shell_dissolved, stopping_time, large_radius, rcloud_boundary, shell_collapsed (0-4), errors 10-13 and 20-23, velocity_runaway/energy_collapsed (50-51), UNKNOWN 99; from_code returns UNKNOWN for any unmatched integer including 5-9.",
    "expected": "The docstring that defines the exit-code contract should enumerate the outcome tokens, since termination.outcome is the public field downstream automation is explicitly told to key on — and should not advertise 0-9 as clean when only 0-4 exist.",
    "failure_scenario": "A tool wanting to select dissolved runs has no documented token to match on, so it string-matches free-form detail text and breaks the first time a phase runner rewords its message — while the token it actually needed (shell_dissolved) was sitting in the enum, undocumented.",
    "repro": "Search the slice's docstrings for outcome tokens: only shell_collapsed appears. Compare against the enum members at simulation_end.py:55-127.",
    "confidence": "high"
  },
  {
    "id": "S13a-R-24",
    "file": "trinity/_output/header.py",
    "line": 91,
    "class": "divergence",
    "severity": "S3",
    "claim": "The startup banner computes the quantity it labels 'log_mCloud' as log10(mCloud/(1-sfe)), while show_run derives the cluster mass as mCloud*sfe — the documented convention. The two encode incompatible interpretations of what the stored mCloud is. That line is also the only one in show_param that omits .value.",
    "evidence": "A: header.py:91 `print(f\"\\tlog_mCloud: {np.log10(params['mCloud']/(1-params['sfe']))} Msun\")` against neighbours :90/:92/:93/:94/:95 which all use .value; show_run.py:134-136 `mCluster = md.get(\"mCluster\")` / `if mCluster is None and \"mCloud\" in md and \"sfe\" in md:` / `mCluster = md[\"mCloud\"] * md[\"sfe\"]`; PARAM_DOCS calls it 'Initial cloud mass [Msun]' (trinity_reader.py:143). B (doc, corroborating only the show_run side): show_run.py:131 'mCluster is derived from mCloud * sfe'.",
    "expected": "One convention for mCloud across the banner, the metadata and show_run, with .value used consistently.",
    "failure_scenario": "The value printed at run start as 'log_mCloud' is not log10 of the mCloud written to metadata.json and read back by show_run; a user comparing the console banner against the run summary sees two different cloud masses for one run. Whichever formula is wrong, one of the two reported masses is off by 1/(1-sfe) or by sfe.",
    "repro": "Run any model and compare the banner's log_mCloud against log10(metadata['mCloud']) and against show_run's mCluster row. OQ-8 decides whether line 91 evaluates at all.",
    "confidence": "medium"
  },
  {
    "id": "S13a-R-25",
    "file": "trinity/_output/trinity_reader.py",
    "line": 3,
    "class": "other",
    "severity": "S3",
    "claim": "The module docstring's usage block documents get_at_time two ways in four consecutive lines — first as returning the closest snapshot, then as interpolating by default. Combined with R-11 (the exact-match tolerance is a fixed 1e-8 Myr window) and R-09 (interpolated snapshots are silently a mixture), the slice's most-used public accessor is both mis-specified and under-specified.",
    "evidence": "B: trinity_reader.py:3 '# Get snapshot closest to a specific time  snap_at_1myr = output.get_at_time(1.0)  # Get snapshot at a specific time (interpolated by default)  snap = output.get_at_time(0.5)  # Returns interpolated snapshot  snap = output.get_at_time(0.5, mode=\"closest\")'; trinity_reader.py:695 documents mode 'interpolate' as the default. A: confirms 'interpolate' is the default path (it is the branch that raises ValueError on out-of-range at :766-770, while 'closest' clamps at :729-737).",
    "expected": "Fix the first usage comment to say interpolated-by-default, or pass mode='closest' explicitly in that example.",
    "failure_scenario": "A user copies the first idiom believing they are sampling a real integrator output, and instead plots a synthesised point — a fabricated data point in a published figure, indistinguishable unless they check is_interpolated, which per R-09 is set even when the value is a nearest-neighbour fallback.",
    "repro": "Read the Basic Usage block at trinity_reader.py:3 against the mode default at :695.",
    "confidence": "high"
  },
  {
    "id": "S13a-R-26",
    "file": "trinity/_output/simulation_end.py",
    "line": 303,
    "class": "other",
    "severity": "S4",
    "claim": "Four different serialisation policies coexist for one problem: the final_state guard drops with a bare json.dumps (no encoder), _NpEncoder converts ndarray->list, update_metadata_atomic's guard uses that encoder, and _jsonable stringifies. The final_state guard is therefore STRICTER than the writer it feeds. Separately, FINAL_STATE_EXCLUDE_ARRAYS is documented as the mechanism keeping long arrays out while a blanket any-length array skip already exists, and the two duplicated encoders are held together only by a 'MUST stay in sync' comment with no stated enforcement.",
    "evidence": "A: simulation_end.py:303 `json.dumps(val, allow_nan=True)` with no encoder vs _metadata_io.py:54-55 `if isinstance(obj, np.ndarray): return obj.tolist()` used by the guard at :123 and the write at :92; vs simulation_end.py:695 `return str(val)` (a list becomes the string '[1.0, 2.0]'); blanket skip at :289-292. B: simulation_end.py:244 'long arrays listed in FINAL_STATE_EXCLUDE_ARRAYS' vs :288 comment '# Skip arrays/lists with any length — final_state is scalars only'; _metadata_io.py:41 'Duplicates trinity._input.dictionary.NpEncoder to avoid a circular import… The two encoders MUST stay in sync.'",
    "expected": "One encoder used everywhere so what passes the guard is exactly what the writer emits; keep either the blanket skip or the named list and say which is load-bearing; add a test asserting the two encoders agree on a representative value set.",
    "failure_scenario": "A 0-d or empty np.ndarray value is dropped from final_state by the strict guard while the same value inside termination_debug is written fine — the same quantity is present in one block and absent from the other for no reason a consumer can infer. Separately, a numpy-dtype fix applied to only one encoder makes metadata.json and dictionary.jsonl serialise the same quantity as different types.",
    "repro": "`python -c \"import json,numpy as np; print(json.dumps(np.array([])))\"` -> TypeError; same with cls=_NpEncoder -> '[]'. Then OQ-5 (`grep -rn FINAL_STATE_EXCLUDE_ARRAYS trinity/`) and OQ-11 (`grep -rn NpEncoder test/`).",
    "confidence": "high"
  },
  {
    "id": "S13a-R-27",
    "file": "trinity/_output/trinity_reader.py",
    "line": 461,
    "class": "state",
    "severity": "S4",
    "claim": "_rehydrate_metadata inserts the same object into every snapshot via setdefault, so a metadata list is aliased across all N snapshots; and it filters by denylist, so any future non-reserved top-level metadata block is silently broadcast into every snapshot and appears in output.keys as if it were a per-timestep quantity. The setdefault mechanism and its precedence rule are documented; these two consequences are not.",
    "evidence": "A: trinity_reader.py:458-461 `run_consts = metadata_keys_to_rehydrate(metadata)` / `for snap in snapshots:` / `for k, v in run_consts.items():` / `snap.setdefault(k, v)`; run_constants.py:145-146 `return {k: v for k, v in metadata.items() if k not in RESERVED_TOP_LEVEL_KEYS}` with the reserved set at :113-118. B (doc): run_constants.py:3 'merged back via setdefault (so any per-snapshot value, when present, takes precedence)'; trinity_reader.py:429 'per-snapshot value wins when both are present' — mechanism documented, aliasing and denylist consequences not.",
    "expected": "Deep-copy (or freeze) the injected values, and mark rehydrated keys so TrinityOutput.keys can distinguish run constants from time series.",
    "failure_scenario": "A consumer normalises the initial cloud profile in place — snap['initial_cloud_n_arr'][:] = … on snapshot 0 — and silently changes it in all N snapshots. Separately, any newly added top-level block is broadcast into every snapshot with no marker.",
    "repro": "`o = read('<run>/dictionary.jsonl'); a = o[0]['initial_cloud_r_arr']; b = o[5]['initial_cloud_r_arr']; a is b` -> True.",
    "confidence": "high"
  },
  {
    "id": "S13a-R-28",
    "file": "trinity/_output/trinity_reader.py",
    "line": 293,
    "class": "silent-failure",
    "severity": "S4",
    "claim": "The reader turns every missing key into None or 0.0 rather than an error: Snapshot.__getitem__ uses .get with no sentinel, t_now defaults to 0.0, t_min/t_max default t_now to 0, and TrinityOutput.get fills absent keys with None inside an object-dtype array. The slice documents the opposite discipline elsewhere (_phys returns the literal 'n/a'), so there are two tolerance policies.",
    "evidence": "A: trinity_reader.py:293 `return self.data.get(key)`; :306 `return self.data.get('t_now', 0.0)`; :661/:666 `s.get('t_now', 0)`; :684 `values = [s.get(key) for s in self._snapshots]` then np.array with `except (ValueError, TypeError)` at :688 returning the raw list, so the return type of get(as_array=True) is not stable. B: terminal_prints.py:144 '_phys returns n/a if the key is absent or its value is None/non-numeric' — a documented tolerant policy in the other direction, and A notes 'n/a' is also returned for a value that fails float(), making the two cases indistinguishable in the transcript.",
    "expected": "__getitem__ should raise KeyError (that is what [] means; .get already exists for the tolerant case), and t_min/t_max should not manufacture a zero.",
    "failure_scenario": "A typo — snap['shell_nmax'] instead of snap['shell_nMax'] — returns None, is multiplied by a conversion factor, and raises TypeError far from the typo or is plotted as an empty series. One snapshot missing t_now silently drags the reported time range to start at 0.",
    "repro": "`o[0]['no_such_key']` -> None instead of KeyError.",
    "confidence": "high"
  },
  {
    "id": "S13a-R-29",
    "file": "trinity/_output/trinity_reader.py",
    "line": 405,
    "class": "divergence",
    "severity": "S4",
    "claim": "The two accepted input formats are read by non-equivalent readers: the .json path mutates the loaded snapshots by injecting snap_id and then sorts by it; the .jsonl path does neither. The key set and the ordering guarantee of TrinityOutput therefore depend on which extension the user handed it, and both are reachable through the same public entry points.",
    "evidence": "A: trinity_reader.py:401-411 `snap['snap_id'] = int(key)` then `snapshots.sort(key=lambda s: s.get('snap_id', 0))`; vs :418-423 for jsonl, which appends in file order with no snap_id and no sort; both reachable via find_data_path (:1223-1250) and resolve_data_input (:1300-1335). B: documents 'Sort by snap_id' at the JSON site (trinity_reader.py:410) and the search precedence 'dictionary.jsonl > dictionary.json' (:1134), but never states that the guarantees differ.",
    "expected": "One normalisation step applied to both paths, or an explicit statement of which ordering guarantee the reader provides for which format.",
    "failure_scenario": "A comparison harness that diffs output.keys() between a legacy .json run and a current .jsonl run reports a spurious schema difference; and a .json run whose object keys are non-numeric collapses every unparsable snapshot to sort position 0.",
    "repro": "`read('a.json').keys` vs `read('a.jsonl').keys` on the same data -> 'snap_id' present only in the first.",
    "confidence": "high"
  },
  {
    "id": "S13a-R-30",
    "file": "trinity/_output/trinity_reader.py",
    "line": 1044,
    "class": "deadcode",
    "severity": "S4",
    "claim": "Dead, unreachable and undocumented-API code (flagged only, per project rule 3): the pandas ImportError guard in to_dataframe cannot fire because pandas is imported at module scope; the float/NaN branch in _print_parameters is a no-op; DROPPED_IN_V2 has no reader; find_data_file has no in-slice caller; Snapshot.interpolation_time is written and never read; load_output is an alias with no in-slice user that nevertheless appears in a docstring example without ever being introduced.",
    "evidence": "A: trinity_reader.py:132 module-scope `import pandas as pd` vs :1044-1047 `try: import pandas as pd except ImportError: raise ImportError(\"pandas is required for to_dataframe()\")`; :1015 `stype = type(sample).__name__…` followed by :1018-1019 `elif isinstance(sample, float) and not np.isnan(sample): stype = f'float'` (identical result); run_constants.py:88-92 DROPPED_IN_V2 unreferenced; trinity_reader.py:1133 find_data_file uncalled; :289 interpolation_time set at :910 never read; :1095 `load_output = read`. Module-scope `from scipy import interpolate` at :133 likewise makes scipy a hard dependency of merely reading a file. B: corroborates DROPPED_IN_V2 ('no reader-side prose says anything consumes it') and load_output (B-24 — used in the find_all_simulations example at :1344, introduced nowhere; the alias comment at :1094 does not name it).",
    "expected": "n/a — flagged, not proposed for deletion. Note that A's account corrects B-24's failure scenario: the alias exists, so the documented example runs; the defect is documentation only.",
    "failure_scenario": "No runtime failure. The pandas guard gives a false impression that the reader degrades gracefully without pandas; in fact `import trinity._output.trinity_reader` fails outright on a pandas-less or scipy-less install.",
    "repro": "`grep -n 'import pandas' trinity/_output/trinity_reader.py` -> lines 132 and 1045.",
    "confidence": "high"
  },
  {
    "id": "S13a-R-31",
    "file": "trinity/_output/simulation_end.py",
    "line": 471,
    "class": "silent-failure",
    "severity": "S4",
    "claim": "_load_last_snapshots reads the entire trajectory into memory to obtain two records, and silently discards a truncated final line so that snapshot_count under-reports and the whole comparison table is omitted — indistinguishable from a run that genuinely produced one snapshot.",
    "evidence": "A: simulation_end.py:471-472 `with open(jsonl_path, 'r', encoding='utf-8') as f: lines = f.readlines()`; :475-483 `for line in lines[-n:]` with `except json.JSONDecodeError: continue`; consumer at :597 `\"snapshot_count\": len(snapshots)` and :605 `snap_old = snapshots[-2] if len(snapshots) >= 2 else None`, which suppresses the comparison block at :620. B: documents snapshot_count as a termination_debug key but no tolerance semantics.",
    "expected": "Seek from the end for the last two complete lines, and record in the debug block that a partial line was discarded.",
    "failure_scenario": "A run terminated after a partial line was flushed produces termination_debug with snapshot_count 1, no comparison rows and no warnings; show_run prints '(no flagged changes, no NaN/Inf, all sanity checks passed)' for a run whose last write was interrupted. On a long run the readlines() also pulls the full multi-hundred-MB trajectory into memory at the worst possible moment.",
    "repro": "`truncate -s -20 dictionary.jsonl`, then call write_termination_debug_report(dir) and inspect metadata.json['termination_debug'].",
    "confidence": "high"
  },
  {
    "id": "S13a-R-32",
    "file": "trinity/_output/simulation_end.py",
    "line": 181,
    "class": "silent-failure",
    "severity": "S4",
    "claim": "When params has no path2output, write_simulation_end silently writes metadata.json into the current working directory.",
    "evidence": "A: simulation_end.py:177-181 `if output_dir is None:` / `if 'path2output' in params:` / `output_dir = params['path2output'].value` / `else:` / `output_dir = '.'`; then :207 `os.makedirs(output_dir, exist_ok=True)`. B: documents only '_metadata_io.py:79 — Caller is responsible for ensuring run_dir exists', with no fallback semantics.",
    "expected": "Raise, or at minimum log an error naming the fallback location — a run's audit trail landing in CWD is never intentional.",
    "failure_scenario": "A programmatic invocation that builds params without path2output drops metadata.json into the repository root and, on the next such run, overwrites it — while the actual run directory has no termination block at all.",
    "repro": "`write_simulation_end({'SimulationEndCode': item(0)})` with no 'path2output' key, then `ls ./metadata.json`.",
    "confidence": "high"
  },
  {
    "id": "S13a-R-33",
    "file": "trinity/_output/trinity_reader.py",
    "line": 833,
    "class": "numerical",
    "severity": "S4",
    "claim": "Interpolation type handling has two edge defects: bool is special-cased to nearest-neighbour but plain int falls into the numeric branch and is linearly interpolated; and the scalar and array branches use opposite degenerate-case fallbacks (scalar takes the first VALID value, array takes element 0 regardless of validity). DEMOTED: A's headline example (snap_id -> 17.4) may be void, because B transcribes a doc claim that interpolated snapshots carry snap_id = -1.",
    "evidence": "A: trinity_reader.py:827-830 bool -> closest, then :833 `if isinstance(first_val, (int, float)):` -> :857 `interpolated_data[key] = float(interp_func(t))`; snap_id injected as int by _load_json_format at :405. Degenerate case: scalar :842-845 `y_vals[valid_mask][0] if np.any(valid_mask) else np.nan` vs array :874-875 `result.append(elem_values[0] if len(elem_values) > 0 else np.nan)`. B: trinity_reader.py:908 documents snap_id '-1 indicates interpolated' — if the code stamps -1 after the per-key loop, the snap_id instance of A's claim does not occur.",
    "expected": "Route int-typed keys to nearest-neighbour alongside bool (or keep an explicit non-interpolable key list), and make the array branch mirror the scalar branch: `elem_values[valid_mask][0] if np.any(valid_mask) else np.nan`.",
    "failure_scenario": "An integer-coded counter or 0/1 flag stored as int (not bool) comes back fractional from get_at_time. Separately, a bubble profile element that is NaN in the earliest neighbour and finite in exactly one other carries NaN in the interpolated profile while the equivalent scalar quantity would carry the finite value — one NaN then poisons a downstream np.trapz or np.max.",
    "repro": "OQ-4 first: check trinity_reader.py:902-911 for a post-loop `snap_id = -1` stamp. Then compare a scalar key and an array key with the same NaN pattern across neighbours.",
    "confidence": "low"
  },
  {
    "id": "S13a-R-34",
    "file": "trinity/_output/show_run.py",
    "line": 173,
    "class": "divergence",
    "severity": "S4",
    "claim": "Reporting divergences across the three formatters of the same quantities: display precision differs (t_now .6f / .3f / .4e; R2 .4f vs .3f; Eb .4e vs .3e), get_at_time raises on out-of-range in 'interpolate' mode but silently clamps in 'closest' mode, TrinityOutput.phases returns a set-ordered list whose order varies with string hash randomisation, __getitem__ is annotated -> Snapshot but returns a list for a slice index, and heartbeat defaults a missing t_now to tmax (reporting 100% progress). The show_run docstring also names two display conversions where the code applies four.",
    "evidence": "A: terminal_prints.py:132 ('t','t_now',1.0,'.6f','Myr') vs show_run.py:173 '.3f' vs trinity_reader.py:964 '.4e'; R2 .4f (terminal_prints.py:137) vs .3f (show_run.py:180); trinity_reader.py:766-770 raises vs :729-737 clamps; :656 `return list(set(...))` (info() sorts at :970, the property does not); :469-470 slice returns a list; terminal_prints.py:198 heartbeat defaults t_now to tmax. B: show_run.py:162 names only km/s for v2 and cm⁻³ for shell_nMax, while comments at :197 and :203 add Pb -> P/k_B and Eb -> erg.",
    "expected": "One display-precision table shared by the three formatters, a consistent out-of-range policy, a sorted phases property, and a docstring that states the conversion rule rather than enumerating two of four rows.",
    "failure_scenario": "A final time of 0.0004 Myr prints as '0.000 Myr' in show_run while the console showed '0.000400 Myr', and a sub-milliparsec radius prints as 0.000 — a user comparing the two artefacts sees a value that looks like zero in one and not the other. output.phases iterated in a report gives a different order on each invocation.",
    "repro": "Compare terminal_prints._STATE_FIELDS, simulation_end.CRITICAL_PARAMS and show_run._final_state_section for the same run; `PYTHONHASHSEED=1` vs `2` changes `list(set(['a','b','c']))` order.",
    "confidence": "high"
  },
  {
    "id": "S13a-R-35",
    "file": "trinity/_output/trinity_reader.py",
    "line": 1406,
    "class": "other",
    "severity": "S4",
    "claim": "parse_simulation_params requires an exponent-form mass (no sign) and an INTEGER sfe, so a folder named …_sfe0.05_n… returns None; and three consumers apply three different failure policies (one prints a warning, two skip silently). The documented folder-naming conventions include forms the regex may not admit.",
    "evidence": "A: trinity_reader.py:1406-1410 `re.search(r'm?(\\d+\\.?\\d*e\\d+)_sfe(\\d+)_n(\\d+\\.?\\d*(?:e\\d+)?)', folder_name, re.IGNORECASE)`; consumers :1483-1485 (warn) vs :1550-1554 in info_simulations and :1437-1439 in get_unique_ndens (silent). B: documents folder conventions '{run_name}_modified folder (new runs)' (trinity_reader.py:1134) and '_modified suffix or profile tags like _PL0, _BE14' (:1389) — a naming convention no other prose in the slice acknowledges, including the v4 four-files-per-directory description.",
    "expected": "One shared failure policy, and a regex that accepts the decimal-SFE and signed-exponent forms the sweep namer actually produces.",
    "failure_scenario": "A sweep directory named with decimal SFEs makes info_simulations return count>0 but empty mCloud/sfe/ndens lists, with nothing printed; the user reads that as 'no simulations found'.",
    "repro": "`python -c \"from trinity._output.trinity_reader import parse_simulation_params as p; print(p('m1e6_sfe0.05_n100'))\"` -> None",
    "confidence": "medium"
  },
  {
    "id": "S13a-R-36",
    "file": "trinity/_output/trinity_reader.py",
    "line": 574,
    "class": "units",
    "severity": "S4",
    "claim": "Two distinct mean-molecular-weight run-constants are named across the slice — mu_convert (required for cloud-profile reconstruction) and mu_atom (given as an example run-constant) — with neither defined, no unit stated, and no prose relating them.",
    "evidence": "B: trinity_reader.py:574 'the arrays are rebuilt deterministically from nCore, nISM, rCore, rCloud, dens_profile, densPL_alpha, mu_convert (and densBE_Omega, gamma_adia for BE)'; trinity_reader.py:215 comment repeats the mu_convert list; trinity_reader.py:549 'every constant-through-run parameter (mCloud, nCore, mu_atom, …)'. A: does not distinguish them.",
    "expected": "Define both keys and their relation where they are first named — per-particle vs per-hydrogen-nucleus conventions, with or without helium, are a standard factor-~1.4 trap and this repo names units a recurring bug class.",
    "failure_scenario": "An external reconstruction of n(r) or M(<r) uses mu_atom where the reconstruction path uses mu_convert, producing an enclosed-mass profile wrong by a constant factor that looks physically plausible.",
    "repro": "Read trinity_reader.py:574/:215 against :549; check whether both keys exist in metadata.json and what each means.",
    "confidence": "medium"
  },
  {
    "id": "S13a-R-37",
    "file": "trinity/_output/run_constants.py",
    "line": 136,
    "class": "state",
    "severity": "S4",
    "claim": "metadata_keys_to_rehydrate's docstring enumerates three reserved top-level keys while the constant it describes has four — termination_debug (the v4/Phase-5 addition) was added to the constant's comment but not to the function docstring. RESOLVED: Lens A read RESERVED_TOP_LEVEL_KEYS and confirms four members, so the docstring is the stale side and the code is correct.",
    "evidence": "B: run_constants.py:136 'Reserved entries (_metadata_version, termination, final_state) are surfaced via dedicated TrinityOutput properties instead'; run_constants.py:102 comment lists four including '* termination_debug — Phase-5 block'. A: run_constants.py:113-118 defines RESERVED_TOP_LEVEL_KEYS = {_metadata_version, termination, final_state, termination_debug}, used by the denylist filter at :145-146.",
    "expected": "The docstring should list all four reserved keys, or defer to the constant rather than restating its membership. (Note B rated this S3; reconciled to S4 because A confirms the code is correct and the impact is one maintainer misreading one docstring next to an unambiguous constant.)",
    "failure_scenario": "A maintainer reading only the function docstring concludes termination_debug is rehydrated and writes a plotter that reads snap['termination_debug'], or adds a fifth reserved block by pattern-matching on a docstring that is already one behind.",
    "repro": "Read run_constants.py:136 against run_constants.py:102 and :113-118.",
    "confidence": "high"
  },
  {
    "id": "S13a-R-38",
    "file": "trinity/_output/trinity_reader.py",
    "line": 1260,
    "class": "other",
    "severity": "S4",
    "claim": "resolve_data_input's docstring documents three accepted input forms while the inline comments enumerate six resolution cases, and its default output directory is documented as \"'outputs' or TRINITY_OUTPUT_DIR env var\" with the precedence unstated and the hardcoded default listed first. This is the slice's only mention of the environment variable.",
    "evidence": "B: trinity_reader.py:1260 'Accepts: 1. Output folder name… 2. Folder path… 3. File path… - uses directly' and 'Defaults to \"outputs\" or TRINITY_OUTPUT_DIR env var'; comments at :1295/:1299/:1309/:1317/:1325/:1331 enumerate six cases including 'a path with extension that doesn't exist yet' and 'a base path (no extension)'. A: confirms resolve_data_input spans :1258-1335 and is a public entry point that can hand back either file format (see R-29), but does not enumerate the cases or the env precedence.",
    "expected": "State the precedence explicitly ('TRINITY_OUTPUT_DIR if set, else \"outputs\"') and list all accepted forms, or say resolution is best-effort over several fallbacks.",
    "failure_scenario": "A user on an HPC node sets TRINITY_OUTPUT_DIR to a scratch mount, passes a bare run name, and the resolver looks under a relative 'outputs' instead — reporting 'no data file found' for runs that exist.",
    "repro": "OQ-7: read trinity_reader.py:1285-1295 and check which of the two the code consults first.",
    "confidence": "medium"
  },
  {
    "id": "S13a-R-39",
    "file": "trinity/_output/__init__.py",
    "line": 1,
    "class": "other",
    "severity": "S4",
    "claim": "The output subpackage's __init__.py is empty — no module docstring, no re-exports, no statement of the public output API. Both lenses reached this independently and identically. Every documented import path in the slice reaches past it into a submodule.",
    "evidence": "A: '_output/__init__.py is empty, so nothing is re-exported at package level; every import in the slice is fully qualified… it means `from trinity._output import X` fails.' B: 'prose.md contains sections for only 7 of the 8 slice files; trinity/_output/__init__.py has no entry at all', against the documented import path at trinity_reader.py:3 'from trinity._output.trinity_reader import TrinityOutput, find_data_file, find_data_path'.",
    "expected": "A short module docstring naming the supported entry points (TrinityOutput, read, read_simulation_end, show_run), since this slice is the output contract every downstream analysis depends on.",
    "failure_scenario": "Consumers import deep module paths; any future reorganisation of _output breaks every plotter and paper script, with no re-export layer to absorb it.",
    "repro": "Confirm the file contains no docstring or comment; compare with the documented import paths at trinity_reader.py:3.",
    "confidence": "high"
  }
]
```
