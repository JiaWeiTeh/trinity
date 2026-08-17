# Stress-test plan: `dictionary.py` snapshot machinery

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

**Status (2026-08-17):** 🔵 actionable — 13 findings verified by probe against `030b658`;
batteries A–H specified below, **none executed yet**. The executing session writes
`test/test_dictionary_stress.py` (characterization) and runs battery H, then updates this doc.

## 0. Scope and object under test

Object: `trinity/_input/dictionary.py` @ `030b658` — `DescribedItem`, `DescribedDict`
(snapshot buffer, duplicate guard, `flush()`, loaders, crash handlers), `save_debug_snapshot`,
`updateDict`. Sibling contracts: `trinity/_output/run_constants.py` (strip/rehydrate schema),
`trinity/_functions/simplify.py` (profile downsampling), `trinity/_output/simulation_end.py`
(termination blocks).

The question this workstream answers: **are `dictionary.jsonl` / `metadata.json` robust — does
the on-disk record faithfully reflect the simulation under edge conditions** (flush boundaries,
crashes, poisoned values, phase handoffs, loading), and where it is not, is each deviation a bug
to fix or a limitation to document?

Existing coverage to build on, not duplicate: `test/test_metadata.py` (metadata split/rehydrate,
corrupt-metadata tolerance, non-serializable *metadata* keys, `disable_crash_handlers` fixture at
line 39), `test/test_simplify.py` (the downsampler in isolation, incl. empty/single-point input),
`test/test_show_run.py`, `test/test_run_smoke.py` (subprocess run pattern with `stop_t=1e-4`).

## 1. Verified findings (probes run 2026-08-17 @ `030b658`)

Every row below was reproduced by the committed harness
(`harness/probe_dictionary.py` — seconds, no simulation). Probe numbers refer to its output lines.

| ID | Probe | Finding (current behavior) | Severity |
|----|-------|----------------------------|----------|
| F1 | P1 | **Duplicate guard is skipped at every flush boundary.** `flush()` clears `previous_snapshot`; the guard (dictionary.py:721) requires a non-empty buffer, so the first `save_snapshot()` after any flush saves unconditionally. A same-`(t_now, R2)` state straddling the 10-snapshot boundary produces adjacent duplicate lines (verified: lines 9 and 10 identical). In-window dedup works. | Medium |
| F2 | P2 | Same skip after **any** flush: manual `flush()`, `write_termination_report()`, emergency `_safe_flush()` all clear the buffer and disarm the guard for the next save. | Medium |
| F3 | P8 | **NaN `t_now` defeats the guard** (`NaN != NaN`): consecutive identical NaN-time states are all saved. | Medium |
| F4 | — | Guard compares **only** `(t_now, R2)`: an in-window snapshot differing in any other key (phase label, energy, forces) is silently dropped. Phase code *relies* on this — `run_energy_phase.py:400-419` builds a reconciliation snapshot precisely so the guard blocks the next phase's stale first snapshot. F1 ⇒ whether a phase-handoff snapshot is deduped **depends on `save_count % 10` alignment**: record content is not a pure function of the trajectory. Design-level; battery A pins it. | Medium |
| F5 | P3a–d | **Profile-array special cases crash `save_snapshot()`**: (a) empty `bubble_r_arr`+`bubble_T_arr` → `ValueError`; (b) `bubble_T_arr` present but companion `bubble_r_arr` missing → `KeyError`; (c) scalar-NaN arrays — exactly what `reset_keys()` writes by default — → `IndexError`; (d) empty `shell_grav_r`+`shell_grav_force_m` → `ValueError`. Only `shell_n_arr` has an empty-guard (dictionary.py:696) — and when it trips, the keys are silently absent from that line (per-line schema varies). The commented-out bubble entries in `COOLING_PHASE_KEYS` (dictionary.py:1217-1222) are the fossil of (c). | High (latent) |
| F6 | P4, P11, P13 | **A non-serializable value poisons `flush()` mid-append**: snapshots before the poisoned one are already written, the exception propagates, the buffer stays intact. On the **first** flush a retry self-heals by accident (`flush_count` still 0 → fresh-run delete rewrites). On any **later** flush, a retry appends the already-written lines again: verified file `t = [0.0, 1.0, 1.0, 2.0]` — every subsequent snapshot id shifts by one line. Contrast: the metadata path (dictionary.py:836-849) does a defensive per-key `json.dumps`; the snapshot path does not. | **High** |
| F7 | P14 | **Merely loading a snapshot rewrites the loaded run's `metadata.json`.** `load_snapshot()` constructs `cls()` → registers atexit → at interpreter exit `_safe_flush()` writes a fresh `termination_debug` block + `metadata_humanreadable.txt` into the *loaded* run's directory. Verified: a recorded crash reason `'ODE solver failed'` is clobbered to `'Normal exit / atexit'` by an analysis script that only reads. | **High** |
| F8 | P7 | `t_now=None` passes the guard but crashes `save_snapshot()` with `TypeError` in the debug-log f-string (`:.6e` on None, dictionary.py:759) — only `KeyError` is caught, and the f-string is evaluated regardless of log level. | Low |
| F9 | P6 | `print(params)` raises `TypeError` when any value is a 0-d `np.ndarray` (`shorten_display` calls `len()`; the `hasattr(__len__)` check passes because the attribute exists). | Low |
| F10 | P5 | `_excluded_keys` is **sticky**: replacing an excluded key with a fresh `DescribedItem(exclude_from_snapshot=False)` does not un-exclude — the refresh loop (dictionary.py:614-617) only ever adds. The key silently vanishes from all snapshots forever. | Medium |
| F11 | P8 | NaN/Infinity are serialized as bare `NaN`/`Infinity` literals — `dictionary.jsonl`/`metadata.json` are **not strict RFC-8259 JSON**. Python's `json` reads them back; `jq` and strict parsers reject the file. | Low |
| F12 | P9 | Round-trip type morphing in `load_snapshot`: every JSON list becomes `np.asarray` — `["alpha","beta"]` → `ndarray('<U5')`, `(1, 2)` → `ndarray`. Scalars (int/bool/float/None) survive. | Low |
| F13 | P10 | Loader id-shift: `load_snapshots` keys snapshots by **file line number** (`enumerate`), and skipped lines (blank/corrupt) still consume an index — a mid-file blank line yields ids `{0, 2, 3}` with id 1 missing while its data lives under id 2. Combined with F6's duplicated line, every downstream `snap_id` is silently off by one. | Medium |

Unprobed design observations (each gets a battery case before it may be quoted as fact):

- **O1** — fresh-run semantics: first `flush()` of a process deletes existing
  `dictionary.jsonl` **and** `metadata.json` (dictionary.py:810-816). Resume-in-place is
  impossible by design; a second `DescribedDict` writing to the same dir clobbers the first; a
  manual `flush()` with an *empty* buffer still deletes and writes metadata.
- **O2** — no fsync; a `kill -9` can tear the final line. The loader tolerates it (warn + skip).
- **O3** — the guard reads only the in-memory buffer, never disk — the structural cause of F1/F2.
- **O4** — `snapshot_interval` is a plain attribute: `0` → `ZeroDivisionError` on first save;
  `1` → buffer cleared every save → guard permanently disarmed (F1 at every step).
- **O5** — every `DescribedDict` registers its own atexit hook (they accumulate) and overwrites
  the process-wide SIGINT/SIGTERM handlers (last dict wins). After a signal, `_signal_handler`
  flushes with reason `"Signal SIGINT"`, then `sys.exit` runs the atexit hooks, which rewrite the
  termination report with the generic reason — the true signal cause is likely clobbered (same
  mechanism as F7). To verify in battery C.
- **O6** — constructing a `DescribedDict` in a non-main thread should raise `ValueError`
  (`signal.signal` restriction). To verify in battery C.
- **O7** — downstream effect of duplicate rows (F1): `np.diff(t) == 0` in any reader-side rate or
  interpolation. What `trinity_reader` actually does with duplicate `t` is unmeasured. Battery H.

## 2. Robustness invariants (what "outputs are robust" means)

The batteries gate against these. Each is currently TRUE, FALSE, or UNKNOWN — the campaign's job
is to make every cell KNOWN and pinned by a test.

| ID | Invariant | Status @ 030b658 |
|----|-----------|------------------|
| I1 | Every line of `dictionary.jsonl` parses as JSON (Python `json`; strictness caveat F11) | UNKNOWN (battery H) |
| I2 | Loader ids are contiguous `0..N-1` and equal the writer's `snap_id`s | FALSE on corrupt/blank lines (F13) and after F6 retry |
| I3 | `t_now` is non-decreasing across lines; adjacent duplicate `(t_now, R2)` pairs occur **only** at line indices `≡ 0 (mod snapshot_interval)` | UNKNOWN in the field (battery H — any duplicate elsewhere is a *new* finding) |
| I4 | Per-line key-set is stable across a run (modulo documented phase-dependent keys) | UNKNOWN (battery H; F5's shell guard drops keys silently) |
| I5 | `save_snapshot()` never raises for states the code itself produces (incl. `reset_keys` output, phase-0 placeholders) | FALSE (F5, F8) |
| I6 | Loading is side-effect-free on the run directory | **FALSE** (F7) |
| I7 | A failed `flush()` retried after remediation neither loses nor duplicates lines | **FALSE** for flush ≥ 2 (F6) |
| I8 | save→load→save round-trip is value-stable (types per F12's documented morphing table) | UNKNOWN (battery E) |
| I9 | The recorded termination reason survives signals, atexit ordering, and later loads | FALSE for loads (F7); UNKNOWN for signals (O5, battery C) |

## 3. Test batteries

Batteries A–G are fast unit/characterization tests (milliseconds each, default `pytest` set).
Battery H is integration on real runs (`@pytest.mark.stress` for anything beyond the smoke
config). Each case states **expected current behavior** — the executing session pins behavior
as-is (`pytest.raises`, equality on the probed values) and tags candidate bugs; it does NOT fix
`dictionary.py` (see §4 ground rules).

### A. Duplicate-guard semantics (F1–F4, O3, O4)

1. Boundary skip: 10 distinct saves → 11th with identical `(t_now, R2)` **is saved**; jsonl gets
   adjacent duplicate lines 9/10 (pin of F1, the motivating question — answer: **yes, true**).
2. In-window dedup: same state twice inside a window → second dropped, `save_count` unchanged.
3. Guard disarmed after manual `flush()` / `write_termination_report()` mid-window (F2).
4. NaN `t_now`: consecutive identical NaN states all saved (F3).
5. Lossy key coverage: two in-window saves, same `(t_now, R2)`, different `current_phase` (or any
   scalar) → second silently dropped (F4). Pin it and flag: this is the load-bearing behavior the
   phase-boundary reconciliation comment (`run_energy_phase.py:403`) documents.
6. Alignment dependence: replay the *same* save sequence twice, once offset so the identical pair
   straddles a flush boundary — assert the two runs produce different line counts (the crisp
   demonstration that record content depends on `save_count % 10`, not just the trajectory).
7. `snapshot_interval=1` → every save flushes → guard never active (O4); `snapshot_interval=0` →
   `ZeroDivisionError` (pin; flag as missing validation).
8. Guard exception path: `t_now` present but `R2` missing → `KeyError` caught → saves (pin).

### B. Flush atomicity, retry, fresh-run semantics (F6, O1, O2)

1. Poisoned first flush → `TypeError`, partial file, intact buffer; retry after removing the
   poison → clean 2-line file (the accidental self-heal, P11).
2. Poisoned second flush → retry duplicates the already-written line (P13). Pin the corrupted
   `t = [0.0, 1.0, 1.0, 2.0]` shape; flag `flush()`'s per-line `json.dumps` inside the write loop
   as the cause (serialize-then-write would be atomic-ish).
3. Empty-buffer `flush()` at `flush_count == 0` deletes a pre-existing `dictionary.jsonl` +
   `metadata.json` from a previous run and writes fresh metadata (O1). Pin; flag the data-loss
   trap for any pre-run validation path that calls `flush()`.
4. Two `DescribedDict`s, same output dir: second's first flush deletes the first's file (O1).
5. Torn final line: truncate the file mid-line, `load_snapshots` → warn + skip, earlier ids
   intact (O2 tolerance pin).
6. `_safe_flush` swallow-behavior: poisoned buffer + `_safe_flush()` → error logged, no raise,
   buffer still intact (it's cleared only on success) — pin that repeated atexit handlers don't
   raise.

### C. Crash handlers and process lifecycle (F7, O5, O6)

Subprocess-based (real atexit/signals), following the harness's P14 pattern; each case < 5 s.

1. Load-only mutation (F7): writer proc records reason X; reader proc only loads; assert
   `metadata.json` unchanged **fails** today — pin the clobber to `'Normal exit / atexit'`. This
   is the highest-priority fix candidate: every analysis script rewrites history.
2. Signal reason integrity (O5): run a writer proc that saves a few snapshots then sleeps; send
   SIGINT; assert exit code 130, pending snapshots flushed, and inspect whether
   `termination_debug.reason` ends as `"Signal SIGINT"` or the atexit generic — resolve O5's
   "likely" into a fact.
3. SIGTERM same, exit code 143.
4. Handler stacking: create two dicts with different output dirs in one proc, exit — both dirs
   get termination reports; signal goes to the last-created only. Pin.
5. Non-main-thread construction (O6): `threading.Thread(target=DescribedDict)` → expect
   `ValueError` from `signal.signal`. Relevant for any future threaded driver; `run.py --workers`
   uses processes and is safe — verify by reading `sweep_runner.py`, note the answer here.
6. `kill -9` mid-run (O2): SIGKILL a writer between flushes; assert loss is at most the pending
   window (≤ `snapshot_interval-1` snapshots) plus possibly one torn line, and the file still
   loads.

### D. Profile-array special cases (F5)

1. Crash matrix pins: empty-pair `ValueError` (a), missing-companion `KeyError` (b), scalar-NaN
   `IndexError` (c), empty shell-grav `ValueError` (d).
2. `shell_n_arr` guard asymmetry: empty `shell_r_arr` → snapshot succeeds but `log_shell_n_arr` /
   `shell_r_arr` keys absent from the line (schema drift, feeds I4).
3. `reset_keys(['bubble_r_arr', ...])` then `save_snapshot()` → today IndexError (c). Pin, and
   cross-reference the commented-out entries in `COOLING_PHASE_KEYS` as the reason they must stay
   commented until fixed.
4. Value edge-cases through the `log10(max(·, eps))` clamp: zero and negative densities silently
   become `-300`; `bubble_T_arr` uses no `abs()` (negative T → -300) while `bubble_dTdr_arr` does.
   Pin the clamp values; flag silent masking.
5. `inf` in a profile array: `_simplify_error` R² becomes NaN → `NaN < 0.9` is False → no warning
   (silent degradation path). Verify and pin.
6. Mismatched lengths `bubble_r_arr` vs `bubble_T_arr` → `ValueError` with the keyname message
   (dictionary.py:510-514). Pin the message contract.
7. Phase-0 reality check: construct params exactly as `read_param`/registry does for phase 0
   (placeholder profile arrays) and `save_snapshot()` — establish which of (a)–(d) production
   actually dodges today and by what mechanism (exclusion flags? non-empty placeholders?). This
   turns F5 from "latent" into either "unreachable today" or "reachable via path X".

### E. Serialization round-trip (F11, F12, I8)

1. Type-morphing table: str/int/float/bool/None/list/tuple/np scalar/np array in → what comes
   back. Commit the table into this doc when measured.
2. Ragged nested list (`[[1,2],[3]]`) → `np.asarray` in `load_snapshot` — expect `ValueError` on
   numpy ≥ 1.24 (loader crash on a file the writer happily produced). Verify.
3. NaN/Inf literals: `json.loads` round-trips; `jq . dictionary.jsonl` fails (document F11; a
   one-line `jq`-check in battery H tells us whether real runs emit NaN lines at all).
4. save→load→save: write a state, load it, save again, byte-compare the two lines (after keying
   out rehydrated run-consts). Establishes I8.
5. Deep structures: dict-valued params (`{'a': np.float64(1)}`) — `_to_json_ready_value` falls
   through to raw `val`, `NpEncoder` rescues nested numpy on dump; verify load returns plain dict
   (not ndarray) — pin.

### F. Loader robustness (F13, plus existing metadata coverage)

1. Blank line mid-file → ids `{0, 2, 3}` (pin F13).
2. Corrupt (non-JSON) line mid-file → same shift, warning printed on stdout (note: `print`, not
   `logging` — flag inconsistency).
3. Duplicate line (F6 aftermath) → ids fine but content shifted vs. writer intent —
   characterize with a doctored file.
4. `load_snapshot` unknown id → `KeyError` listing available ids (pin message).
5. `load_latest_snapshot` on empty/absent file → `ValueError` / `FileNotFoundError` (pin).
6. Rehydration precedence (per-snapshot wins over metadata) — already covered by
   `test_metadata.py::test_per_snapshot_value_wins_over_metadata`; do not duplicate, cite.

### G. API misc (F8, F9, F10)

1. `t_now=None` → `TypeError` from the log f-string (pin F8; the guard's `except KeyError` is the
   wrong net).
2. `print(params)` with 0-d array → `TypeError` (pin F9).
3. Sticky exclusion (pin F10) + the positive control: `exclude_from_snapshot=True` insert-time
   exclusion works.
4. `DescribedDict.copy()` returns a plain `dict` (machinery lost) — verify and document.
5. `DescribedItem.__eq__` on array values returns an array → `if params['x'] == y:` raises
   ambiguity `ValueError`; also `__eq__` without `__hash__` makes items unhashable. Pin as API
   footguns.
6. `updateDict` contract: mismatched lengths → `ValueError`; dataclass mode skips missing keys
   silently (pin both).
7. `save_debug_snapshot` with callables/interpolators → skipped-keys metadata; overwrite-always;
   works without `path2output`. Pin the contract (it has zero tests today).

### H. Full-run field checks (integration; the "is the record robust in anger" battery)

Reuses the separate-process discipline from CLAUDE.md rule 5 and the run pattern of
`test_run_smoke.py` (subprocess `run.py`, `stop_t=1e-4`, ~1 min). Configs, in cost order:

- smoke config (`test_run_smoke.py`'s inline param) — default suite;
- `param/simple_cluster.param` — `@pytest.mark.stress`;
- `docs/dev/performance/f1edge_lowdens_himass_hisfe.param` and
  `docs/dev/performance/f1edge_hidens_himass_losfe.param` — `@pytest.mark.stress` (these span
  feedback strength × density; they are the canonical stiff/edge regimes).

For each completed run, a single invariant-scanner (write it once, in the test file) checks
`dictionary.jsonl` + `metadata.json` for:

1. I1: every line parses; count of NaN/Infinity literals per line (F11 field incidence).
2. I2: loader ids contiguous.
3. I3: `t_now` non-decreasing; **catalog every adjacent duplicate `(t_now, R2)` pair with its
   line index mod 10** — F1 predicts duplicates only at `≡ 0`; any elsewhere is a new finding.
   Also catalog near-duplicates (`|Δt| < 1e-12`) that the exact-equality guard misses.
4. I4: per-line key-set; report lines missing profile keys (F5 shell-guard incidence in anger).
5. Phase-handoff audit (F4): locate phase-transition lines (via `current_phase` changes), check
   whether the handoff produced 1 or 2 records and whether that correlates with
   `line_index % 10` across the config set.
6. O7: feed the file to `trinity_reader` / `TrinityOutput`; compute `np.diff(t)`; check the
   reader's derived quantities don't divide by zero on duplicate-t rows.
7. F7 in anger: after the run, load the latest snapshot in a fresh subprocess, exit, and diff
   `metadata.json` bytes — quantifies the mutation on a real run artifact.

Persist the scanner's per-run results as a CSV under `data/` (one row per config × invariant,
with the exact command and commit in a provenance header), so the next session compares without
re-running (💾).

## 4. Ground rules for the executing session

1. **Characterize, don't fix.** All batteries pin **current** behavior (via `pytest.raises` where
   it crashes) with a `# CANDIDATE-BUG F<n> — see docs/dev/dictionary-robustness/PLAN.md`
   comment. No edits to `trinity/_input/dictionary.py` in the stress pass. Any fix is a follow-up
   under the CLAUDE.md risky-change ladder: the snapshot writer is on every run's output path, so
   a fix to the guard/flush changes `dictionary.jsonl` **content** — that is *never* a "free win";
   it needs a pre-registered gate stating the expected diff (e.g. "exactly the boundary-duplicate
   lines disappear; everything else byte-identical") on the battery-H config set, in separate
   processes, at matched `t`.
2. **Where tests go**: one new file, `test/test_dictionary_stress.py`. Batteries A–G in the
   default set (fast); battery H beyond the smoke config marked `@pytest.mark.stress`. Copy the
   `disable_crash_handlers` fixture from `test/test_metadata.py:39` for every in-process test;
   battery C uses subprocesses instead (real handlers). Suite invariant: 0 failed, before and
   after (`test/CLAUDE.md`).
3. **Plausible values** per `test/CLAUDE.md`: where a test needs physical params (battery D.7,
   battery H), use realistic GMC values, not round numbers. Pure machinery tests (guard, flush)
   may use minimal dicts — they never touch physics.
4. **Sandbox**: some tempdir tests flake when `/tmp` is unwritable — set `TMPDIR` to a writable
   dir before judging failures (`test/CLAUDE.md`).
5. **Do not deduplicate this work**: the probe set (P1–P14; no P12, it was merged into P14) is
   already committed as the harness; the
   test file supersedes the harness as the *enforced* record, but keep the harness runnable (it
   is the fast, self-contained reproduction).
6. **Update this doc** (and `README.md` + `DOC_STATUS.md` row) with: battery results, resolved
   UNKNOWNs in §2, any new findings (F14+), and the battery-H CSV path. Date every change (🔄/🔗).

## 5. Suggested execution order

1. Batteries A + B (the headline mechanics; ~an hour of work, all fast tests).
2. Battery C (subprocess; resolves O5/O6 and hardens F7's evidence).
3. Batteries D–G (breadth; D.7 is the one requiring real registry/param plumbing).
4. Battery H smoke config; then stress configs if time/budget allows.
5. Doc + DOC_STATUS updates; commit CSV.

## 6. Maintainer decisions queued (fix vs. document — do not resolve unilaterally)

1. F1/F2: should the guard compare against the *last written* snapshot (persist `(t_now, R2)` of
   the last save across flushes — a 2-tuple attribute, no disk read) so dedup is
   alignment-independent? Changes jsonl content ⇒ risky-ladder.
2. F4: is `(t_now, R2)` the right duplicate key, given phase handoffs deliberately exploit it?
   Any change must re-read `run_energy_phase.py:400-419` first.
3. F6: make `flush()` serialize all lines *before* opening the file (atomic-ish append), or
   at least clear the buffer only for lines actually written?
4. F7: should `load_snapshot`-created dicts skip `_register_crash_handlers` (e.g. a
   `register_handlers=False` classmethod path)? This is the cheapest high-value fix and does not
   touch run output — but it changes atexit behavior of analysis scripts.
5. F5c: `reset_keys`' NaN default vs. the profile-array branches — guard the branches, or forbid
   resetting profile arrays?
6. F11: `allow_nan=False` with a sanitize step would make files strict JSON — worth the content
   change?
