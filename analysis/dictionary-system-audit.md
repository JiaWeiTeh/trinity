# Dictionary system audit — `src/_input/dictionary.py`

Scope: `DescribedDict.save_snapshot`, `flush`, `_clean_for_snapshot`,
`_safe_flush`, `reset_keys`, the crash/exit handlers, and the call sites in
`main.py` and the four phase runners (`phase1_energy`, `phase1b_energy_implicit`,
`phase1c_transition`, `phase2_momentum`).

Verified against the real run output at
`outputs/mockOutput/mockFullrun/dictionary.jsonl` (178 snapshots, four phases).


## 1. Mock simulation walkthrough

```
run.py / start_expansion
└─ run_expansion(params)             save_count flush_count buffer  jsonl   metadata
   ├─ phase0 init (t0,R2…)             0          0          {}      —      —
   ├─ phase 1a (energy):
   │    seg 0: save_snapshot           1          0          {0}     —      —
   │    seg 9: save_snapshot          10          1          {}      10 ln  written
   │    reconciliation save_snap      11          1          {10}    10 ln  ok
   ├─ phase 1b (implicit):
   │    seg 0: save_snapshot          DUPLICATE-SUPPRESSED (same t_now,R2 as recon)
   │    seg 1: save_snapshot          12          1          {10,11} 10 ln  ok
   │    …reconciliation save_snap     N           ?          {…}     …      ok
   ├─ phase 1c (transition):
   │    seg 0: save_snapshot          DUPLICATE-SUPPRESSED
   │    …reconciliation save_snap
   ├─ main.py:241  reset_keys(COOLING_PHASE_KEYS) → 25 keys set to np.nan
   ├─ phase 2 (momentum):
   │    seg 0: save_snapshot          DUPLICATE-SUPPRESSED
   │    …reconciliation save_snap
   └─ params.flush()                                                  appended
write_simulation_end (separate file: simulationEnd.txt)
write_termination_report  → flush() (noop) + termination_debug.txt
process exit → atexit → _safe_flush(reason="Normal exit / atexit")
                              └─ overwrites termination_debug.txt
```

Empirical confirmation from the real run:

| Boundary             | Last snap (old phase)            | First snap (new phase)            | Δt              |
|----------------------|----------------------------------|-----------------------------------|-----------------|
| energy → implicit    | snap[97]  t=2.910161e-03         | snap[98]  t=3.410161e-03          | 5.0e-4 Myr      |
| implicit → transition| snap[146] t=9.352877e-02         | snap[147] t=9.552877e-02          | 2.0e-3 Myr      |
| transition → momentum| snap[163] t=1.171814e-01         | snap[164] t=1.191814e-01          | 2.0e-3 Myr      |

The Δt at every boundary equals exactly one segment of the *next* phase. That is
consistent with the iter-0 save of every new phase being silently dropped by the
duplicate guard, and the snapshot we see being iter-1 (one ODE segment in).


## 2. Findings

| #  | Severity | Where                                              | Finding                                                                                                                                                                                                                                                                                                                                                                              | Verified                                                                                  |
|----|----------|----------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------|
| 1  | High     | `dictionary.py:647-657`                            | Duplicate guard checks **only** `t_now` and `R2`. Every new phase enters at the same `(t_now, R2)` as the previous phase's reconciliation snapshot, so iter-0 is silently dropped. Earliest record of every new phase is one ODE segment in, not at the boundary.                                                                                                                    | confirmed by Δt = one segment at every boundary in the real jsonl                          |
| 2  | High     | `dictionary.py:647 + 778`                          | `if … and self.previous_snapshot:` short-circuits immediately after every flush (buffer just emptied). Duplicate suppression silently disables itself once per `snapshot_interval`.                                                                                                                                                                                                  | latent (no duplicates surfaced in this run)                                                |
| 3  | High     | `dictionary.py:262-290, 252-260`                   | atexit always calls `write_termination_debug_report`; on normal exit it overwrites the descriptive reason written by `main.py:140`. SIGINT path is worse: `sys.exit(128+sig)` re-fires atexit, which writes the report a second time.                                                                                                                                                | confirmed: `termination_debug.txt` says "Normal exit / atexit"; `simulationEnd.txt` has the real reason |
| 4  | High     | `dictionary.py:565-627`                            | Asymmetric pair handling. `bubble_r_arr / shell_grav_r / shell_r_arr` blocks `continue` unconditionally, so if the partner array is missing the r-array is silently dropped. The reverse (partner without r-array) raises `KeyError` mid-snapshot.                                                                                                                                   | latent (pairs always present in this run)                                                  |
| 5  | High     | `dictionary.py:569-611`                            | Only `shell_n_arr` has `if x_arr.size > 0`. `bubble_*` and `shell_grav_force_m` will pass empty arrays into `simplify()` → `_simplify` blows up.                                                                                                                                                                                                                                     | latent (non-empty in this run)                                                             |
| 6  | High     | `main.py:241` + `_clean_for_snapshot:540-544` + `flush():767-772` | `reset_keys(COOLING_PHASE_KEYS)` zeros ~25 keys to `NaN` but leaves them in the dict, so they keep getting serialised. `json.dumps(NaN)` emits the literal token `NaN` (non-RFC); strict parsers reject the file.                                                                                                                                                  | confirmed: 112 lines contain `NaN`; 8+ cooling keys present as NaN in 14/14 momentum snapshots |
| 7  | Medium   | `dictionary.py:540-544, 217-218`                   | `_excluded_keys` is append-only. Setting `item.exclude_from_snapshot = False` after insert never clears the entry.                                                                                                                                                                                                                                                                   | code review                                                                                |
| 8  | Medium   | `dictionary.py:833-874`                            | `load_snapshot()` returns a dict with `flush_count=0` and `path2output` set to the source directory. An accidental `save_snapshot` / `flush` triggers the fresh-run branch (#10) and **deletes the original `dictionary.jsonl` and `metadata.json` before writing back**.                                                                                                            | code review (destructive footgun)                                                          |
| 9  | Medium   | `dictionary.py:767-772 + 800-811`                  | `dictionary.jsonl` is plain append — no `fsync`, no temp+rename (asymmetric to `metadata.json:757-760`). Crash mid-write yields a partial trailing line; `load_snapshots` swallows it as a `print` warning and silently skips.                                                                                                                                                       | code review                                                                                |
| 10 | Medium   | `dictionary.py:736-742`                            | "Fresh run" branch fires whenever `flush_count == 0`, deleting both files. Combined with #8, any first flush of a loaded dict wipes the source files.                                                                                                                                                                                                                                | code review                                                                                |
| 11 | Low      | `dictionary.py:252-260`                            | `sys.exit(128+signum)` re-fires atexit, so the flush+report path runs twice. Compounds #3.                                                                                                                                                                                                                                                                                           | follows from #3                                                                            |
| 12 | Low      | `run_constants.py:67`, `metadata_keys_to_rehydrate`| `METADATA_VERSION` is written but never compared. The "forward-compat" affordance is aspirational.                                                                                                                                                                                                                                                                                   | code review                                                                                |
| 13 | Low      | `dictionary.py:1170-1175`                          | `updateDict(dataclass)` silently skips fields whose name isn't in the dict. Typos / forgotten registrations → silent data loss.                                                                                                                                                                                                                                                      | code review                                                                                |
| 14 | Low      | phase runners (1a/1b/1c/2) reconciliation blocks   | Broad `try/except Exception` only logs a warning. If reconciliation raises, no save_snapshot fires, then #1 also drops the next phase's iter-0 → entire phase boundary state is missing.                                                                                                                                                                                              | latent                                                                                     |
| 15 | Low      | `dictionary.py:670-673`                            | Dead branch: `if until_flush == self.snapshot_interval: until_flush = 0` is unreachable because the divisible-by-interval case is fully handled by the if/else above.                                                                                                                                                                                                                | code review                                                                                |
| 16 | Low      | `dictionary.py:190`                                | `snapshot_interval = 10` is hardcoded and absent from `param/default.param`.                                                                                                                                                                                                                                                                                                         | tuning                                                                                     |
| 17 | Info     | `dictionary.py:932-1035` vs `494-516`              | `save_debug_snapshot` re-implements JSON-readying instead of reusing `_to_json_ready_value`. Drift risk if the snapshot schema evolves.                                                                                                                                                                                                                                              | code review                                                                                |


## 3. Suggested fixes (post-consistency-check)

Each fix has been checked against the rest of the codebase to ensure it is
legitimate, internally consistent, and does not introduce a new bug. See §4 for
what changed between the first draft of these fixes and the version below, and
§5 for cross-fix interactions.

| #  | Fix                                                                                                                                                                                                                                                                                                                                                              |
|----|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| 1  | Add `current_phase` to the dedupe key. Replace lines 647-657 with: read `cur_phase = self["current_phase"].value if "current_phase" in self else None`; build the key as `(t_now, r2, cur_phase)`; suppress only on full-tuple match. `current_phase` is registered at `read_param.py:405` and set at `main.py:183, 203, 225, 251` before each phase runner is invoked, so it is always populated when `save_snapshot` fires. |
| 2  | Move dedupe state out of the flush buffer. Add `self._last_save_key: tuple | None = None` in `__init__`; compare/update it instead of looking up `previous_snapshot[str(save_count - 1)]`. This persists across flushes (#2) and is naturally reset by `cls()` in `load_snapshot*`.                                                                                |
| 3  | Idempotency for the termination report. Inside `_safe_flush`, gate **only** the report-write block on `getattr(self, "_termination_report_written", False)`; leave the buffer-flush block at lines 276-282 unconditional. In `write_termination_report`, set `self._termination_report_written = True` *after* a successful write. The buffer flush itself is already idempotent (empty buffer is a no-op). |
| 4  | Pair-handling guards. In each derived block (`bubble_T_arr`, `bubble_n_arr`, `bubble_v_arr`, `bubble_dTdr_arr`, `shell_grav_force_m`, `shell_n_arr`), early-return to default serialisation with a `logger.warning` if the x-key (`bubble_r_arr` / `shell_grav_r` / `shell_r_arr`) is missing. Do **not** restructure into a `PAIRS` table or a two-pass consumption tracker — that would change the on-disk schema and break readers that don't expect the r-array to reappear. The current "drop-r-on-continue" behaviour stays for the nominal case where the partner exists. |
| 5  | Lift the empty-array guard. Add `if x_arr.size == 0: new_dict[<out_y>] = []; new_dict[<out_x>] = []; continue` in every paired block, mirroring what `shell_n_arr` already does.                                                                                                                                                                                  |
| 6  | Add an `exclude` parameter to `reset_keys`: `reset_keys(self, keys, value=np.nan, exclude=False)`. When `exclude=True`, also flip `item.exclude_from_snapshot = True` for each key. Update `main.py:241` to pass `exclude=True`. As a belt-and-braces measure, also pass `allow_nan=False` to `json.dumps` and use a custom encoder that emits `null` for `NaN`/`Inf`. |
| 7  | Drop the `self._excluded_keys` cache. Recompute the excluded set inline at the top of `_clean_for_snapshot` from the live `item.exclude_from_snapshot` flags. Remove the eager add in `__setitem__` (lines 217-218). The cache currently can only grow; recomputing makes the flag the single source of truth and lets users un-exclude a key. Cost is O(n) per snapshot, n ≈ few hundred items — negligible. |
| 8  | Mark loaded dicts read-only. In `load_snapshot`, `load_latest_snapshot`, `load_snapshots` (when returning a `DescribedDict`), set `params._readonly = True`. In `save_snapshot` and explicit `flush()`, raise `RuntimeError("Loaded snapshot is read-only; set params._readonly = False to override")` if the flag is set. In `_safe_flush`, **early-return silently** if `_readonly` so atexit on analysis scripts doesn't log error noise. |
| 9  | Crash safety on jsonl writes. After the `with open(... "a")` block in `flush()`, call `os.fsync(f.fileno())` before exit. Promote `print(f"Warning: Could not parse line {idx}: {e}")` at line 810 to `logger.warning(...)` so the corruption signal lands in the structured log. Leave the metadata.json print at line 829 as-is (its fallback is benign).        |
| 10 | Gate the fresh-run delete. Add `self._fresh_run: bool = True` in `__init__`; replace `if self.flush_count == 0:` at lines 736-742 and 750 with `if self.flush_count == 0 and self._fresh_run:`. In `load_snapshot*`, set `_fresh_run = False`. Combined with #8, this is defense in depth.                                                                       |
| 11 | Subsumed by #3. The `_termination_report_written` idempotency flag also covers the signal-handler→atexit re-entrance.                                                                                                                                                                                                                                            |
| 12 | Honour `_metadata_version`. In `metadata_keys_to_rehydrate` (or before): `v = metadata.get("_metadata_version", 0); if v > METADATA_VERSION: logger.warning(f"metadata.json version {v} is newer than reader {METADATA_VERSION}; some keys may be silently ignored")`. Don't auto-downgrade older files.                                                          |
| 13 | `updateDict` debug log + `strict=False` kwarg. Default behaviour unchanged. In dataclass mode, when any fields are missing, emit `logger.debug(f"updateDict: skipped {missing} missing keys for {type(dc).__name__}")`. With `strict=True`, raise `KeyError` on the missing list.                                                                                |
| 14 | Reconciliation hygiene. Keep the broad `except Exception` (narrowing per-step would mean rewriting four runners), but log via `logger.exception(...)` so the full traceback lands. Add `finally: params.save_snapshot()` so the post-ODE state is always recorded. Apply **after** #1, otherwise the rescue snapshot is itself dropped by the duplicate guard.    |
| 15 | Delete dead lines 671-673. The divisible case is fully handled at line 676.                                                                                                                                                                                                                                                                                       |
| 16 | Expose `snapshot_interval`. Add `snapshot_interval 10` to `param/default.param` near `simplify_npoints` (line 42). Read it in `__init__` or once at `start_expansion` via `self.get("snapshot_interval")`, mirroring how `simplify_npoints` is handled at `dictionary.py:447-448`.                                                                                |
| 17 | Promote `_to_json_ready_value` to a module-level function `_to_json_ready(val)`. Have both `_clean_for_snapshot` and `save_debug_snapshot` call it. Keep the "skip non-serialisable" branch as a thin wrapper in `save_debug_snapshot` that catches `TypeError`/`ValueError` from the shared helper.                                                              |


## 4. What changed between the first draft and these fixes

Three refinements were made after re-reading the codebase.

1. Fix #3 originally proposed two idempotency flags (`_termination_report_written`
   and `_safe_flush_done`). The second is redundant: the buffer-flush block is
   already a no-op when the buffer is empty, and gating it on a flag would risk
   suppressing a legitimate flush of late-arriving snapshots. Drop the second
   flag; gate only the report write.

2. Fix #4 originally proposed a `PAIRS` dict + helper that did a two-pass
   consumption tracker so an x-array would survive if its derived partner was
   missing. That would change the on-disk schema (e.g. `bubble_r_arr` would
   start reappearing in snapshots after years of absence) and break readers
   that don't expect it. Drop the two-pass idea. Keep the smaller fix:
   guard against a missing x-key inline and fall back to default serialisation.

3. Fix #14 originally proposed narrowing each phase runner's broad
   `try/except Exception` to specific exception types. That would mean
   rewriting four files for marginal benefit. Compromise: keep the broad
   except, but upgrade the log call to `logger.exception(...)` (full traceback)
   and add `finally: params.save_snapshot()` to guarantee the boundary state
   is recorded. This depends on #1 — without it, the rescue snapshot would be
   silently de-duplicated.


## 5. Cross-fix interactions

| Pair    | Interaction                                                                                                                                                              |
|---------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| #1 + #2 | Both touch the dedupe path. Land them in one patch.                                                                                                                       |
| #3 + #11| #3's idempotency flag makes #11 (signal-handler→atexit re-entrance) a non-issue. One patch.                                                                                |
| #6 + #7 | #6 sets the `exclude_from_snapshot` flag; #7 makes the snapshot loop honour the flag at write time without a stale cache. Apply together.                                  |
| #8 + #10| Defense in depth against accidental overwrite of source files from a loaded dict. Apply together.                                                                          |
| #1 + #14| #14's `finally: save_snapshot()` only makes sense after #1. Order-dependent.                                                                                                |
| #4 + #5 | Both touch the same paired blocks. One patch.                                                                                                                              |


## 6. Risks introduced by the proposed fixes

| Risk                                                                                                  | Source fix | Mitigation                                                                                                                                                                                                                                                                                          |
|-------------------------------------------------------------------------------------------------------|------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Phase boundaries will produce two snapshots at the **same** `t_now` (one per side of the boundary)    | #1         | Spot-checked the readers (`paper_bubblePhase.py`, `velocity_radius.py`, `paper_v2R2.py`, `trinity_reader._snapshots`); none assume strict t-monotonicity, all key off `current_phase`. Recommend a one-line note in the `save_snapshot` docstring + a CHANGELOG entry. Did not read every reader.    |
| `os.fsync` on every flush could be slow on networked filesystems                                       | #9         | Flushes happen every `snapshot_interval` saves; even ~50 ms per fsync is negligible at that cadence.                                                                                                                                                                                                |
| `_readonly` raising from `save_snapshot` could surprise scripts that load + mutate + flush             | #8         | The example in the dictionary docstring (line 29) is load-only. No internal code does load + mutate + flush. Provide an explicit `params._readonly = False` opt-out for users who need it.                                                                                                          |
| `strict=True` on `updateDict` could fail if a dataclass adds a field that wasn't registered           | #13        | Default stays `False`. Only opt-in users hit this — and they'd want to fail.                                                                                                                                                                                                                        |
| Promoted `logger.warning` for a partial trailing jsonl line could become noisy if crashes are frequent | #9         | The partial-line case is rare (only on hard crash mid-flush). One log line per affected file load is appropriate.                                                                                                                                                                                   |


## 7. Suggested implementation order

1. **#3** — gate only the report write on `_termination_report_written`. Smallest patch, immediately verifiable on the next run by inspecting `termination_debug.txt`.
2. **#1 + #2** — single patch on the dedupe key. Highest user-visible impact (phase boundaries land in the data).
3. **#6 + #7** — `reset_keys(..., exclude=True)` + recompute excluded set on the fly. Removes NaN noise from momentum-phase snapshots and produces RFC-valid JSON.
4. **#8 + #10** — `_readonly` and `_fresh_run` flags. Closes the destructive footgun on loaded dicts.
5. **#4 + #5** — pair-handling guards + empty-array guard.
6. **#9** — `os.fsync` + log promotion.
7. **#14** — `logger.exception` + `finally: save_snapshot()` (after #1).
8. **#15, #12, #13, #16, #17** — cleanup tier; do whenever already in the file.


## 8. Verdict

15 of 17 proposed fixes pass the consistency check unchanged. **#3** drops the
redundant `_safe_flush_done` flag. **#4** drops the two-pass consumption-tracking
restructure (would change schema and break readers); the simpler missing-x-key
guard survives. **#14** stays broad-except but upgrades to `logger.exception` +
`finally`. The set is internally consistent and respects the codebase's
existing `exclude_from_snapshot` mechanism (used at `read_param.py:390-398,
454-459` and elsewhere).

The single behaviour change to flag is fix #1 producing same-`t_now` snapshots
at phase boundaries. Empirically that's *more* information, not less; readers
that key on `current_phase` will handle it correctly, but a CHANGELOG note is
warranted in case any downstream tooling outside this repo assumes strict
t-monotonicity.
