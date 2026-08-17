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

**Status (2026-08-17):** 🟡 partial — **batteries A–G executed and landed** as 65 characterization
tests (`test/test_dictionary_stress.py` 49 + `test/test_dictionary_stress_process.py` 16, of which
2 are `stress`), all green, pinning current behavior including the defects — a red one after a
deliberate fix means re-baseline, not regression. 13 findings probe-verified (§1) + 5
inherited (§1b) + 3 from execution (§1c). **Four fixes landed:** F20's empty-curve guard (§1d, resolves the reachable phase-0 crash
F19/F5a/F5d), F7's loader handler-skip (§1e, I6 now holds), and F6's transactional flush append
(§1f) — whose re-verification **corrected §1's F6 row twice**: the consequence is worse than
written (production retries 4× and compounds) and the trigger it names is unreachable, while an
unnamed one (torn I/O write) is live; and F1/F2's persisted dedupe key (§1h), which keeps the guard
armed across flushes so the record no longer depends on `save_count % 10` alignment (F21 resolved
with it), while F3's NaN behaviour and F4's `(t_now, R2)` key are deliberately untouched. The first
three are bit-identical by construction and were also verified **together** against a pre-everything
tree on a config that crosses into the implicit phase (§1g); F1/F2 is *not* neutral by construction
but measured output-neutral on both configs in matched sessions (§1h). ⚠️ **§1h also carries the
workstream's most reusable lesson: a recorded hash is session-local** — this container's
`DYNAMIC_ARCH` OpenBLAS shifts the bubble solver's last digits across restarts, which briefly looked
exactly like the F1/F2 fix changing output. **Battery H partially executed**: the fast and multi-phase configs are
scanned and committed (`data/field_scan.csv`); the `f1edge_*` stiff configs are still owed.

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
| F1 | P1 | **Duplicate guard is skipped at every flush boundary.** `flush()` clears `previous_snapshot`; the guard (dictionary.py:721) requires a non-empty buffer, so the first `save_snapshot()` after any flush saves unconditionally. A same-`(t_now, R2)` state straddling the 10-snapshot boundary produces adjacent duplicate lines (verified: lines 9 and 10 identical). In-window dedup works. | ✅ **FIXED** 2026-08-17 — §1h |
| F2 | P2 | Same skip after **any** flush: manual `flush()`, `write_termination_report()`, emergency `_safe_flush()` all clear the buffer and disarm the guard for the next save. | ✅ **FIXED** 2026-08-17 — §1h |
| F3 | P8 | **NaN `t_now` defeats the guard** (`NaN != NaN`): consecutive identical NaN-time states are all saved. | Medium |
| F4 | — | Guard compares **only** `(t_now, R2)`: an in-window snapshot differing in any other key (phase label, energy, forces) is silently dropped. Phase code *relies* on this — `run_energy_phase.py:400-419` builds a reconciliation snapshot precisely so the guard blocks the next phase's stale first snapshot. F1 ⇒ whether a phase-handoff snapshot is deduped **depends on `save_count % 10` alignment**: record content is not a pure function of the trajectory. Design-level; battery A pins it. | Medium |
| F5 | P3a–d | **Profile-array special cases crash `save_snapshot()`**: (a) empty `bubble_r_arr`+`bubble_T_arr` → `ValueError`; (b) `bubble_T_arr` present but companion `bubble_r_arr` missing → `KeyError`; (c) scalar-NaN arrays — exactly what `reset_keys()` writes by default — → `IndexError`; (d) empty `shell_grav_r`+`shell_grav_force_m` → `ValueError`. Only `shell_n_arr` has an empty-guard (dictionary.py:696) — and when it trips, the keys are silently absent from that line (per-line schema varies). The commented-out bubble entries in `COOLING_PHASE_KEYS` (dictionary.py:1217-1222) are the fossil of (c). **Refined by execution**: F19 shows (a)/(d) are reachable from a real `read_param` dict, and F20 locates the crash in the R² diagnostic rather than the downsampler. | **High — reachable (F19)** |
| F6 | P4, P11, P13 | **A non-serializable value poisons `flush()` mid-append**: snapshots before the poisoned one are already written, the exception propagates, the buffer stays intact. On the **first** flush a retry self-heals by accident (`flush_count` still 0 → fresh-run delete rewrites). On any **later** flush, a retry appends the already-written lines again: verified file `t = [0.0, 1.0, 1.0, 2.0]` — every subsequent snapshot id shifts by one line. Contrast: the metadata path (dictionary.py:836-849) does a defensive per-key `json.dumps`; the snapshot path did not. ⚠️ **This row was corrected on re-verification — see §1f:** the consequence is worse (production retries 4× and compounds; the failing snapshot never lands) and the *trigger* named here is **not reachable** (all interpolator keys are `exclude_from_snapshot`); the live trigger is an I/O failure mid-write. | ✅ **FIXED** 2026-08-17 — §1f |
| F7 | P14 | **Merely loading a snapshot rewrites the loaded run's `metadata.json`.** `load_snapshot()` constructs `cls()` → registers atexit → at interpreter exit `_safe_flush()` writes a fresh `termination_debug` block + `metadata_humanreadable.txt` into the *loaded* run's directory. Verified: a recorded crash reason `'ODE solver failed'` is clobbered to `'Normal exit / atexit'` by an analysis script that only reads. | ✅ **FIXED** 2026-08-17 — §1e |
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

## 1b. Prior art — the earlier dictionary-system audit (read before executing)

An earlier audit of this same file exists **off-trunk** and was found only after §1's probes were
run: `analysis/dictionary-system-audit.md` (179 lines) on branch `fix/audit-dictionary-system`,
pinned at `e554316f`. Read it with:

```bash
git show e554316f:analysis/dictionary-system-audit.md
```

It is written against the **old layout** (`src/_input/dictionary.py`), so **none of its line
numbers map** to `trinity/_input/dictionary.py`; its *findings* do. Per the docs/dev rule, treat
its claims as unverified — three were re-checked here (see "corrections" below). It carries three
things this plan does not: **field measurements from a real 178-snapshot four-phase run**, a
**17-item fix set** consistency-checked against the codebase (its §3), and a **risk + ordering
analysis** for those fixes (its §5–§7). Do not re-derive any of that.

Reconciliation (this plan's IDs ↔ audit's `#`):

| Here | Audit | Relationship |
|------|-------|--------------|
| F1, F2 | #2 | Same mechanism. The audit rated it **latent** ("no duplicates surfaced in this run"); probe P1 **constructs** the duplicate, upgrading latent → demonstrated. |
| F4 | #1 | Same finding, and the audit **measured it in the field**: Δt at all three phase boundaries equals exactly one segment of the *next* phase, i.e. every new phase's iter-0 save is silently dropped. Rated High. |
| F5a, F5d | #5 | Identical (empty arrays into `simplify`). |
| F5b | #4 | The audit adds the **reverse direction** I did not probe: the `continue` in the r-array branches is unconditional, so if the *derived* partner is missing the r-array is silently dropped (no exception, data just absent). |
| F7 | #3, #8, #10 | Overlapping but **not** the same. The audit's #3 is the run-path clobber (atexit overwrites `main.py`'s descriptive reason) and #11 the signal re-entrance; my F7 is the **load-only** path — an analysis script that merely reads rewrites the run's `metadata.json`. #8/#10 sharpen it further: a loaded dict carries `flush_count == 0` **and** `path2output` pointing at the source, so any accidental `save_snapshot()`/`flush()` **deletes the source `dictionary.jsonl` + `metadata.json`** (my O1, framed as a destructive footgun). |
| F10 | #7 | Identical (append-only `_excluded_keys`). |
| F11 | #6 | The audit adds the **cause and the field incidence**: `reset_keys(COOLING_PHASE_KEYS)` (`main.py`) NaNs ~25 keys that stay in the dict and keep serializing — **112 lines contained `NaN`**, with 8+ cooling keys NaN in 14/14 momentum snapshots. |
| O1 | #10 | Same fresh-run delete branch. |
| O2 | #9 | Same (no fsync, no temp+rename; asymmetric to `metadata.json`). Audit also flags the `print`-not-`logger` corruption warning (my battery F.2 note). |
| O5 | #3, #11 | Same signal→atexit re-entrance. |
| G.6 | #13 | Same `updateDict` silent-skip. |
| — | #12, #14, #15, #16, #17 | **Inherited, not probed here** — added below as F14–F18. |

Inherited findings (from the audit; **unverified against current source** except where noted):

| ID | Audit | Finding |
|----|-------|---------|
| F14 | #14 | Phase runners' reconciliation blocks catch broad `Exception` and only `logger.warning`. If reconciliation raises, no snapshot fires **and** F4 then drops the next phase's iter-0 → the entire boundary state is missing from the record. Verified present in shape at `run_energy_phase.py:410-421`. |
| F15 | #15 | Dead code: the `until_flush = 0` reset. **Re-verified present** at `dictionary.py:746-747` — its only reader is the `else` branch at :757, which is reached only when `save_count % interval != 0`, where the reset condition is false by construction. |
| F16 | #16 | `snapshot_interval = 10` is hardcoded (`dictionary.py:220`) and not a `.param` key. **Re-verified**: no hit in `param/` or `trinity/_input/registry.py`. Contrast `simplify_npoints`, which *is* read from the dict (`dictionary.py:505-507`). Relevant to O4 and to F1 (the boundary effect's period is this constant). |
| F17 | #12 | `_metadata_version` written but not honoured on read. **Partially stale** — `trinity/_output/cloudy/run_loader.py:106` does gate on `metadata.get("_metadata_version", 1) >= 2`. The gap is narrower than the audit states: no *newer-than-reader* check anywhere. |
| F18 | #17 | `save_debug_snapshot` re-implements JSON-readying instead of reusing `_to_json_ready_value` (drift risk). Still true by inspection (`dictionary.py:1090-1109` vs `553-575`). |

Consequences for this plan:

1. **Battery H is partly pre-answered.** The audit's boundary Δt table (H.5) and NaN incidence
   (H.1) are exactly two of the field measurements H asks for. Its numbers predate the current
   tree by many merges (incl. C3c) so they are **not quotable**, but H should be run as a
   *comparison* against them, not a from-scratch discovery — and the executing session should
   report whether the boundary Δt signature still holds.
2. **Battery D gains a case** (F5b-reverse, audit #4): derived partner missing → r-array silently
   dropped, no exception. Add it to §3 D.1.
3. **§6's decisions should start from the audit's fix set**, not a blank page — with the standing
   caveat that its fixes were never landed, never gated, and were written against the old layout.

## 1c. Execution results (2026-08-17, batteries A–G + partial H)

Landed as 60 tests, all green: `test/test_dictionary_stress.py` (48, in-process, batteries
A/B/D/E/F/G) and `test/test_dictionary_stress_process.py` (12, real interpreters, battery C + the
field scanner). Every §1/§1b finding that a test can express is now pinned, so a future
"fix" flips a red test rather than passing silently. Run them with:

```bash
pytest test/test_dictionary_stress.py test/test_dictionary_stress_process.py   # default set
pytest -m stress test/test_dictionary_stress_process.py                        # real-run scans
```

**Three new findings from execution:**

| ID | Finding | Severity |
|----|---------|----------|
| F19 | **F5 is reachable from a real production dict, not merely latent.** `read_param.read_param()` on a minimal valid `.param` returns all five profile arrays as `np.array([])` with `exclude_from_snapshot=False`, and `save_snapshot()` on that dict raises `ValueError`. Production survives *only* because phase-0 init populates the arrays before the first save fires (confirmed: line 0 of a real run already carries `log_bubble_T_arr`). Consequence: **any `save_snapshot()` placed before phase-0 completes crashes the run**, which is a live constraint on anyone adding an initial-condition snapshot, an early-termination snapshot, or a pre-run validation record. | ✅ **FIXED** (§1d) |
| F20 | **The empty-array crash is in the R² diagnostic, not the downsampler.** `_simplify` handles empty input fine (`test_simplify.py::test_empty`); the traceback ends in `_simplify_error` → `np.interp` ("array of sample points is empty"), called unconditionally by `DescribedDict.simplify()` (dictionary.py:524) for the R² log line — which is computed regardless of log level. So F5a/F5d are a *diagnostics* bug and the cheapest fix is guarding that one call, independent of the F5b/F5c pair-handling questions. | ✅ **FIXED** (§1d) — gated bit-identical |
| F21 | **F1 and F4 interact: ~1 phase boundary in 10 will duplicate instead of suppress.** F4's suppression only happens while the guard is armed. A boundary landing at a line index ≡ 0 (mod `snapshot_interval`) has the guard disarmed (F1), so it emits **two** records at the same `t_now` instead of one. Observed boundary in the multi-phase run sat at index 97 (mod 10 = 7 ⇒ suppressed). Neither outcome is wrong-by-design, but which one occurs is decided by flush alignment, not physics — the concrete cost of F1. | ✅ **FIXED** 2026-08-17 via F1/F2 — §1h. The guard is now armed at every boundary, so the suppress branch is taken every time; never observed in the field before the fix |

**Resolved open questions:**

- **O5 — resolved, the signal reason is clobbered.** `_signal_handler` writes `"Signal SIGINT"`,
  then `sys.exit(128+signum)` runs the atexit handler, which overwrites it with
  `"Normal exit / atexit"`. Exit codes are 130/143 as expected and pending snapshots *are*
  flushed. So F7's mechanism destroys signal provenance too — the "likely" in §1 is now a fact,
  pinned by battery C.
- **O6 — resolved.** Constructing a `DescribedDict` off the main thread raises `ValueError`
  (`signal.signal` restriction). `run.py --workers` uses processes, so this is a latent
  constraint on any future threaded driver, not a live bug.
- **O2 — bounded.** SIGKILL between flushes loses exactly the pending window and leaves the
  flushed prefix complete and loadable (10/10 in the pinned case); no torn line when the kill
  lands on a clean boundary.

**Field scan (battery H, partial) — `data/field_scan.csv`:**

| Config | Result |
|--------|--------|
| smoke (`mCloud 1e5`, `sfe 0.3`, `stop_t 1e-4`) — 97 snapshots, phase 1a only | I1 ✅ (0 unparsable) · I3 ✅ (t non-decreasing, **0 duplicates**) · I4 ✅ (1 distinct key-set) · F11 **97/97 lines carry `NaN`**, 0 carry `Infinity` |
| `simple_cluster` (partial run, crossed into 1b) | **F4 field-confirmed on the current tree**: energy→implicit boundary Δt = **5.0e-4 Myr** against a typical energy segment of 1.008e-4 (≈5×), i.e. the implicit phase's iter-0 save was dropped and its first record is one segment late. This reproduces the old audit's 5.0e-4 figure exactly (§1b), so that signature has survived every merge since. |

Two refinements to inherited claims:

- **F11's NaN source is broader than the audit's #6.** The audit attributed the NaN literals to
  `reset_keys(COOLING_PHASE_KEYS)` in the momentum phase; in the fast config *every* line carries
  NaN while still in phase 1a, from keys the energy phase never populates (`v_neg_frac_thick`,
  `bubble_Lgain`, `bubble_Lloss`, `residual_*`). Both sources are real; `reset_keys` is not the
  only one, so a fix aimed only at it would not produce strict JSON.
- **F1 has no observed field incidence in the fast config** — 0 duplicates in 97 snapshots. The
  mechanism is real (battery A constructs it), but its practical consequence in a real run is
  F21's boundary lottery rather than spurious duplicate rows.

**Still owed** (for the next session): battery H on the stiff/edge configs
(`docs/dev/performance/f1edge_{lowdens,hidens}*.param`) and on a run that completes all four
phases, adding a row per config to `data/field_scan.csv`; the multi-phase row above came from a
truncated run and records only the 1a→1b boundary.

## 1d. F20 fix — landed and gated (2026-08-17)

The first and only `trinity/` change from this workstream. **Scope: one guard, diagnostics-only.**

```python
# trinity/_input/dictionary.py, in DescribedDict.simplify(), after _simplify()
if x_out.size == 0:
    return x_out, y_out
```

Why this is the right first fix: it is the *only* one of the queued decisions (§6) that removes a
**High-severity, reachable** crash (F19) without touching a semantic the rest of the code depends
on. The dedupe key (F1/F4), the pair-handling (F5b/F5c) and the NaN policy (F11) all change what
lands on disk for healthy runs; this one cannot, because the guarded branch is a path that
currently *raises*.

**Pre-registered gate, and the result:**

| Gate | Bar | Result |
|------|-----|--------|
| Per-call equivalence | `simplify()` returns bit-identical output for non-empty curves | ✅ identical on 5 sizes (5 / 21 / 100 / 137 / 999 — spanning below-floor, at-floor and large), each compared against the pre-fix version (today's `HEAD~1`) by stashing the then-uncommitted diff and re-running in a **separate process** |
| Full-run byte equivalence | `sha256(dictionary.jsonl)` unchanged on the fast config, separate processes | ✅ `17370033d8e16ac9147291720b2d6425ae4efd7828e3251d23d09db30b65e006`, 97 snapshots, **identical pre- and post-fix** |
| Behavior change | phase-0 `read_param` dict can snapshot | ✅ was `ValueError`, now succeeds (`test_freshly_read_params_can_snapshot`, written failing-first) |
| Suite | no new failures | ✅ 3 failed / 1149 passed — the same three known-red post-C3c goldens; `test_run_smoke`'s R2 is `0.25672223355034657` on both sides (§1c) |
| mypy · ruff F-rules · black | no new complaints | ✅ 4 errors on the file before and after; lint clean |

Evidence artifact: `data/f20_equivalence.csv` (arm, source ref, config, hash, snapshot count + the
exact commands). Because the hash is unchanged, this qualifies as the "free win" case in
CLAUDE.md rule 5 — proven bit-identical rather than merely equivalent.

**Method note, worth reusing.** The pre-fix arm of a full-run comparison must materialize the old
source explicitly — `git show HEAD~1:trinity/_input/dictionary.py > trinity/_input/dictionary.py`,
with the guard's absence confirmed by `grep` *before* the run, and the file restored by
`git checkout --` after. A first attempt used `git stash push -- <file>`, which is a **silent
no-op once the fix is committed** (nothing to stash ⇒ exit 0 ⇒ the "baseline" arm silently runs
post-fix code and the comparison is vacuous). The per-call arm was run while the fix was still
uncommitted, where `git stash` does work.

The bit-identical result also holds **by construction**: the guard fires only when
`x_out.size == 0`, which is exactly the input on which the old code raised — so no run that
completes on `HEAD~1` can reach it. The measurement confirms the reasoning rather than carrying
it alone.

The `simplify()` docstring's input/output contract gained the empty-input case, which is now
defined behavior rather than a crash.

**What the fix does and does not resolve:**

- **Resolved**: F5a, F5d and therefore F19 — the empty-array crashes, which were the reachable
  ones. An empty pair now records `[]` for both keys.
- **Not touched**: F5b (missing companion → `KeyError`) and F5c (scalar-NaN → `IndexError`, what
  `reset_keys` produces) — different exceptions from different code, still open at §6.5.
- **New inconsistency, deliberately left**: an empty bubble/shell-grav pair now writes `[]`, while
  `shell_n_arr`'s older guard omits its keys entirely. Both are non-crashing; unifying them is the
  maintainer's §6.5 call, not part of F20.
- Two pins were **re-baselined** as predicted by §4.1 (`test_empty_bubble_pair_crashes` →
  `..._records_empty_arrays`, same for shell-grav) — the intended workflow when a fix lands, not a
  regression.

## 1e. F7 fix — landed and gated (2026-08-17)

The second `trinity/` change. **Scope: loading a run must not modify it** (invariant I6).

```python
# DescribedDict.__init__ gains a keyword-only opt-out …
def __init__(self, *args, register_handlers: bool = True, **kwargs):
    ...
    if register_handlers:
        self._register_crash_handlers()

# … and load_snapshot() uses it, because loading is a read:
params = cls(register_handlers=False)
```

Why it was safe to take: **no production code calls the loaders.** `load_snapshot` /
`load_latest_snapshot` are analysis-only API (grep: callers are tests and user scripts), and the
only production construction of a `DescribedDict` is `read_param.py:253`, which keeps the default
and therefore keeps its handlers. So the blast radius is exactly the code path whose behavior was
wrong.

**Pre-registered gate, and the result** (evidence: `data/f7_equivalence.csv`):

| Gate | Bar | Result |
|------|-----|--------|
| Failing tests first | a read-only load mutates the run dir | ✅ three expectations written and confirmed red on the pre-fix tree (dir bytes, humanreadable file, signal handlers) |
| Behavior change | after a load, **every file** in the run dir is byte-identical and the recorded reason survives | ✅ `'ODE solver failed'` now survives a load (was clobbered to `'Normal exit / atexit'`) |
| No signal hijack | `signal.getsignal(SIGINT/SIGTERM)` unchanged across a load | ✅ `UNCHANGED` (was `HIJACKED`) |
| Positive control — writers keep handlers | a real `run.py` still writes `termination`, `termination_debug`, `metadata_humanreadable.txt`; a pending snapshot is still flushed at exit; SIGINT/SIGTERM still exit 130/143 | ✅ all preserved |
| Writer path untouched | `sha256(dictionary.jsonl)` unchanged on the fast config | ✅ `17370033…`, 97 snapshots — the same hash as both F20 arms |
| Suite · mypy · lint | no new failures | ✅ suite unchanged bar the three known-red goldens; mypy 4 → 4; ruff/black clean |

**What this fix does and does not resolve:**

- **Resolved**: F7 — loading is now side-effect-free, so I6 holds. Also removes the load-side
  signal-handler takeover, which was the part of O5 that could surprise a *tool* embedding a load.
- **NOT resolved — I9 is still false.** The signal-reason clobber inside a *writer* process is
  untouched: `_signal_handler` writes `"Signal SIGINT"`, then `sys.exit` runs atexit, which
  overwrites it with the generic reason. That needs the audit's `_termination_report_written`
  idempotency flag (its #3/#11), still open at §6.
- **NOT resolved — O1 stays open by design.** A loaded dict still carries `flush_count == 0` and a
  `path2output` aimed at the source, so an *explicit* `save_snapshot()` + `flush()` on it still
  deletes the source files. That is the audit's #8/#10 (`_readonly` / `_fresh_run` flags) and a
  separate decision at §6.4; `test_explicit_save_on_a_loaded_dict_still_works` pins the current
  behavior so the choice stays visible rather than drifting.
- Two pins were **re-baselined** (`test_loading_a_snapshot_rewrites_the_source_run` →
  `..._leaves_the_source_run_untouched`; `test_read_only_load_also_writes_the_humanreadable_file` →
  `..._writes_no_humanreadable_file`), and two tests were added (signal-handler check, scope guard).

## 1f. F6 fix — landed and gated (2026-08-17), and two corrections to §1's write-up

The third `trinity/` change. **Scope: make the `flush()` append transactional.**

**Re-verified before fixing — and §1's F6 row was wrong in two ways.** Both corrections matter
more than the fix itself, because they change what the finding *means*:

1. **The consequence is worse than described.** §1 recorded a retry producing
   `t = [0.0, 1.0, 1.0, 2.0]` — one duplicated line, from one manual retry. But a failing flush is
   retried **up to four times per run with the same buffer, every exception swallowed**: the
   periodic flush inside `save_snapshot()`, `main.py`'s explicit `params.flush()` (in a
   `try/except` that only logs), `write_termination_report()`, and the atexit `_safe_flush()`.
   Measured pre-fix, that chain yields **`t = [0.0, 1.0, 1.0, 1.0, 1.0]`** — the clean prefix
   duplicated once *per retry*, and the snapshot that caused the failure **never written at all**.
2. **The described trigger is NOT reachable; a different one is.** §1 blames "a non-serializable
   value", which a registry audit says cannot happen today: every interpolator/table key carries
   `exclude_from_snapshot=True`, and the one un-excluded candidate (`shell_interpolate_massDot`) is
   a plain `bool` (`False`) in a real run's snapshot. What *is* reachable is any **I/O failure
   mid-write** — a full disk above all, which this environment documents as a real occurrence —
   and it lands on the identical code path with the identical consequence. So F6's severity stands,
   but for a reason §1 did not name.

**The change**: build the whole payload before opening the file, write it in one call, and roll the
file back to its pre-flush length if the write tears.

```python
payload = "".join(json.dumps(self.previous_snapshot[str(i)], cls=NpEncoder) + "\n"
                  for i in snap_ids)          # serialize first: a bad value writes nothing
mode = "a" if path2jsonl.exists() else "w"
rollback_to = path2jsonl.stat().st_size if mode == "a" else 0
try:
    with open(path2jsonl, mode, encoding="utf-8") as f:
        f.write(payload)
except Exception:
    os.truncate(path2jsonl, rollback_to)      # (guarded) so a retry cannot duplicate
    raise
```

Covering both classes matters precisely because of correction 2: serialize-first alone would fix
only the unreachable trigger, leaving the live one (a torn append on a full disk) still duplicating.
Truncation is worth attempting even out of space — deletes and truncations still succeed there.

**Pre-registered gate, and the result** (evidence: `data/f6_equivalence.csv`):

| Gate | Bar | Result |
|------|-----|--------|
| Failing tests first | four expectations red pre-fix | ✅ poisoned flush writes nothing · later-flush retry does not duplicate · the 4-deep production chain appends nothing · a torn append rolls back |
| Behaviour | retries are safe | ✅ `[0.0, 1.0, 2.0]` (was `[0.0, 1.0, 1.0, 2.0]`); production chain now leaves `[0.0]` (was five lines) |
| Torn-append rollback | file bytes unchanged after a half-written append | ✅ byte-equal to pre-flush |
| Writer path unchanged | `sha256(dictionary.jsonl)` unchanged on the fast config | ✅ `17370033…`, 97 snapshots — same hash as the F20 and F7 arms, across ~10 flushes through the rewritten path |
| Suite · mypy · lint | no new failures | ✅ suite unchanged bar the three known-red goldens; mypy 4 → 4; ruff/black clean |

**Not resolved here:** the retry chain itself. Production still calls `flush()` up to four times
with a poisoned buffer and swallows every failure, so the *snapshots in that buffer are still lost*
— the fix makes the file consistent, not the data complete. Surfacing a failed flush to the user
(rather than a `logger.warning` inside `main.py`) is a separate call, and it belongs with the
audit's #3 idempotency work at §6. `test_production_retry_chain_does_not_compound` pins the
current, now-harmless behaviour of that chain.

## 1g. Cumulative equivalence — all three fixes at once, on a phase-crossing config

§1d/§1e/§1f each gated their own fix against the tree as it stood. That leaves two gaps a
maintainer would rightly worry about: the fixes were never compared **together** against a
pre-everything tree, and the only byte-compared config (`stop_t = 1e-4`) never leaves phase 1a — so
it barely exercises `simplify()`, the function F20 actually changes. Both are now closed
(evidence: `data/cumulative_equivalence.csv`).

**Method.** A pristine `git worktree` at `e3b4692d` (the pre-all-fixes commit), with each fix's
marker confirmed absent by `grep -c` → `0/0/0` *before* running; each arm run from its own tree in
its own process on the same `.param`. Deliberately **not** `git stash` — see §1f's method note.

| Config | Snapshots / phases | `sha256(dictionary.jsonl)` | Verdict |
|--------|--------------------|----------------------------|---------|
| A — `stop_t 1e-4` | 97, phase 1a only | `17370033…` both arms | bit-identical |
| B — `stop_t 0.01` | 114, **crosses 1a→1b** | `7d6a0136…` both arms | bit-identical |

Config B is the one that matters: `log_bubble_T_arr` is populated on **114/114** lines at up to 100
points, so `simplify()` — and therefore F20's guarded call site — runs on every snapshot.

`metadata.json` was compared too, as a recursive leaf diff with `NaN == NaN`: **exactly 3 leaves
differ, 0 of them real** — two wall-clock timestamps and `final_state.sps_path`, which differs only
because the pre arm ran from the worktree and so resolved its bundled SPS table to a different
absolute path. Every physical value in `final_state` and every row of `termination_debug.comparison`
is equal.

Two by-products worth keeping:

- **Battery H gained its multi-phase row** (`data/field_scan.csv`, `multiphase_stop0.01`): 114
  snapshots, I1 ✅, I3 ✅ (0 duplicates), I4 ✅, phases `[energy, implicit]`.
- **F21 remains unobserved but is now bounded.** The 1a→1b boundary landed at line 97 (`97 % 10 =
  7`), so the guard was armed and suppressed the new phase's iter-0 save — the *suppress* branch of
  F21's lottery, exactly as at `stop_t 0.01`'s predecessor run. Catching the *duplicate* branch
  needs a run whose boundary happens to land on a multiple of 10; still owed, along with the
  `f1edge_*` configs.

## 1h. F1/F2 fix — the dedupe guard now survives a flush (2026-08-17)

The fourth `trinity/` change, and **the first one that is not output-neutral by construction** — read
the coupling subsection before trusting any equivalence claim about it. Taken on the maintainer's
instruction, with the explicit scope "fix F1/F2, keep F4 as is".

**The change.** The guard read its comparison value back out of `previous_snapshot`, which every
flush empties — so the first save after any flush was unconditional. The key is now held on the
instance:

```python
self._last_save_key: Optional[Tuple[Any, Any]] = None   # __init__
...
if self._last_save_key is not None:                      # armed across flushes now
    last_t_now, last_r2 = self._last_save_key
    if t_now == last_t_now and r2 == last_r2:
        return
...
self._last_save_key = (self["t_now"].value, self["R2"].value)   # after a real save
```

**What is deliberately NOT changed:**

- **F4 stays.** The key is still exactly `(t_now, R2)`, so the phase-handoff suppression that
  `run_energy_phase.py:400-419` deliberately relies on behaves as before. Any snapshot differing
  only in `current_phase`, `Eb` or the force budget is still dropped in-window.
- **F3 stays**, and avoiding an accidental fix took care: the key is compared **element-wise, not as
  a tuple**. Tuple equality short-circuits on element *identity*, so a repeated NaN `t_now` (the same
  float object) would have compared equal and been suppressed — silently changing F3. Scalar `==`
  does not do this. `test_nan_t_now_defeats_the_guard` documents the trap.
- **F16 stays**: `snapshot_interval` is still a hardcoded plain attribute. The fix does remove its
  role as the *period* of F1's effect, so a wrong value is now merely a buffering choice rather than
  a correctness one — at `snapshot_interval = 1` the guard used to be permanently dead and now works.

### The coupling — why this one can change a run

Three phase runners gate a counter on **whether the guard let a save through**, e.g.
`run_energy_implicit_phase.py:1018-1029` (the comment there already names the duplicate guard):

```python
_save_count_before = params.save_count
params.save_snapshot()
if (params['stop_at_rCloud_nSnap'].value is not None
        and params.save_count > _save_count_before      # only when the save really wrote
        and R2 > params['rCloud'].value):
    params['_snapshots_after_rCloud'].value += 1
```

`_snapshots_after_rCloud` drives `stop_at_rCloud_nSnap` termination. So a save that previously wrote
(because a flush had just disarmed the guard) and is now suppressed withholds one increment, and the
run can stop one snapshot later. Conditions for that to bite, all three required:

1. `stop_at_rCloud_nSnap` is not `None` — it **defaults to `None`**, and every tracked `.param`
   leaves it `None` (checked: `param/*.param`, `docs/dev/performance/*.param`);
2. the run is past the cloud edge (`R2 > rCloud`);
3. a duplicate-eligible save lands immediately after a flush.

Both gated configs below had `stop_at_rCloud_nSnap = None` and `_snapshots_after_rCloud = 0`, so they
cannot exercise it. **A user who sets that parameter is the population at risk**, and
`test_suppressed_save_leaves_save_count_untouched` pins the mechanism the runners depend on so it
cannot drift silently.

### Pre-registered gate, and the result

Evidence: `data/f1f2_equivalence.csv`. **The expected diff was declared before editing**, and it was
*not* bit-identity: this fix removes a line whenever a duplicate-eligible save lands immediately
after a flush. Neither gated config contains one — the field scan found 0 adjacent duplicates, and
the 1a→1b boundary sits at line 97 (`97 % 10 = 7`), so the guard was already armed and suppressing
there. So "no change on these two configs" was the prediction, and it is what was measured.

| Gate | Bar | Result |
|------|-----|--------|
| Failing tests first | the F1/F2 expectations red pre-fix | ✅ 5 red (boundary, manual flush, alignment-independence, `interval=1`, save_count delta) |
| F4 untouched | its pin stays green without edit | ✅ `test_guard_key_ignores_every_other_field` unchanged |
| F3 untouched | its pin stays green without edit | ✅ and the tuple-identity trap avoided (element-wise compare) |
| Matched-session A/B, config A | expected: no change | ✅ pre `b0685d4c…` == post `b0685d4c…` |
| Matched-session A/B, config B (crosses 1a→1b) | expected: no change | ✅ pre `211734c8…` == post `211734c8…` |
| `save_count` coupling | the runners' delta contract pinned | ✅ `test_suppressed_save_leaves_save_count_untouched` |
| Suite · lint · mypy | no new failures | ✅ 66/66 in the two stress files; ruff/black clean; mypy 4 → 4 |

### ⚠️ Methodology finding — a recorded hash is SESSION-LOCAL

Found the hard way while gating this fix, and it invalidates a habit the rest of this workstream
(and `CLAUDE.md` rule 5) encourages: comparing a fresh run against a hash written down earlier.

**What happened.** Config A had hashed `17370033…` four times across four different source versions.
After the F1/F2 change it hashed `b0685d4c…`, which looked exactly like the fix altering output. It
was not. Running the **pre-fix** source in the same session also produced `b0685d4c…` — identical to
post-fix. The container had restarted in between, and `17370033…` is unreachable from *any* source
version in the new container.

**Mechanism.** This image's OpenBLAS is built `DYNAMIC_ARCH=1` (runtime CPU-kernel selection) with
no thread pinning and `MAX_THREADS=2`, so the kernel follows whichever host CPU the container lands
on. The FP reduction order changes, `v2` moves by 1 ulp, and the iterative bubble-structure solve
amplifies that to ~1e-7 relative in `bubble_Tavg` / `bubble_L1Bubble`. Byte reproducibility holds
**within** a container instance and not across restarts.

**Three tells that separated this from a real behaviour change** — worth reusing, because they are
cheap and decisive:

1. **Line counts and the `t_now` sequence were identical** (97 → 97, all 97 values equal). The only
   thing suppressing a duplicate can do is *remove a line*; nothing was removed.
2. **Line 0 already differed** — before the guard is ever consulted (the first save has no
   predecessor to compare against, in either version).
3. **The deltas were last-digit**, not structural: `v2` `…97178` → `…97173`.

**Protocol consequence.** Both arms of an equivalence comparison must run in the **same session**;
a hash from an earlier session proves only what it proved then. The committed
`data/*_equivalence.csv` files now carry this caveat at the top so a future visit does not
"discover" a phantom regression the way this one did. The §1g cumulative comparison is unaffected —
both of its arms ran back-to-back in one container — but its stored hashes are equally session-local.

This also means a **no-change gate cannot be discharged by a stored hash**. For a fix whose expected
diff is "nothing", re-measure both arms; for a fix with an expected diff, compare *shape* (line
counts, the `t_now` sequence, which lines vanished) rather than only the digest, since FP drift will
otherwise masquerade as the change you were looking for — or hide it.

## 2. Robustness invariants (what "outputs are robust" means)

The batteries gate against these. Each is currently TRUE, FALSE, or UNKNOWN — the campaign's job
is to make every cell KNOWN and pinned by a test.

Statuses below are as measured on 2026-08-17 (§1c); "field" means checked against a real run's
record, "pinned" names the battery whose test holds the behavior in place.

| ID | Invariant | Status @ `030b658` |
|----|-----------|--------------------|
| I1 | Every line of `dictionary.jsonl` parses as JSON (Python `json`; strictness caveat F11) | **TRUE** in the field (0 unparsable / 97) — but only under a *permissive* parser; strict RFC-8259 parsers reject 97/97 lines (F11). Pinned: E, H |
| I2 | Loader ids are contiguous `0..N-1` and equal the writer's `snap_id`s | **FALSE** on corrupt/blank lines (F13) and after an F6 retry; TRUE in the field for an uncorrupted run. Pinned: B, F |
| I3 | `t_now` is non-decreasing across lines; adjacent duplicate `(t_now, R2)` pairs occur **only** at line indices `≡ 0 (mod snapshot_interval)` | **RESTATED and now TRUE unconditionally.** The mod-`snapshot_interval` escape clause existed only because of F1; with the guard armed across flushes (§1h) adjacent duplicates cannot occur **anywhere**, so the invariant is simply "no adjacent duplicate `(t_now, R2)` pairs". Field: t non-decreasing, 0 duplicates on both configs. The scanner still reports `adjacent_dups_off_boundary` separately — post-fix, *any* duplicate at all is a finding. Pinned: A, H |
| I4 | Per-line key-set is stable across a run (modulo documented phase-dependent keys) | **TRUE** in the field (1 distinct key-set / 97 lines). Still breakable via F5's silent shell-guard path. Pinned: D, H |
| I5 | `save_snapshot()` never raises for states the code itself produces (incl. `reset_keys` output, phase-0 placeholders) | **STILL FALSE, but narrowed.** The phase-0 case (F19, via empty arrays F5a/F5d) is **fixed** (§1d) and now pinned green by `test_freshly_read_params_can_snapshot`. Remaining: F5b (missing companion → `KeyError`), F5c (`reset_keys`' NaN → `IndexError`), F8 (`t_now=None` → `TypeError`). Pinned: D, G |
| I6 | Loading is side-effect-free on the run directory | ✅ **NOW TRUE** (F7 fixed, §1e) — pinned by a subprocess test that byte-compares *every* file in the run dir across a load, plus one asserting the process's signal handlers survive. Caveat: an **explicit** `save_snapshot()`/`flush()` on a loaded dict still writes, and still deletes the target as a "fresh run" (O1, open at §6.4). Pinned: C |
| I7 | A failed `flush()` retried after remediation neither loses nor duplicates lines | **DUPLICATES: ✅ fixed** (F6, §1f) — the append is now transactional, so no retry (including production's 4-deep swallowed chain) can duplicate a line, and a torn write rolls back. **LOSES: ❌ still true** — the snapshots in a poisoned buffer are never written, and `main.py` only logs the failure, so a run can finish "successfully" having silently dropped a window of snapshots. Pinned: B |
| I8 | save→load→save round-trip is value-stable (types per F12's morphing table) | **TRUE** for values (a reloaded state re-serializes to an identical line); **FALSE** for types — lists become ndarrays, so `str`/`tuple` change class (F12). Pinned: E |
| I9 | The recorded termination reason survives signals, atexit ordering, and later loads | **STILL FALSE, but half-fixed.** Surviving a later load: ✅ fixed (F7, §1e). Surviving a signal: ❌ still clobbered — `_signal_handler` writes `"Signal SIGINT"`, then `sys.exit` runs atexit, which overwrites it with the generic reason (O5). Needs the audit's `_termination_report_written` idempotency flag (#3/#11), open at §6. Pinned: C |

## 3. Test batteries

> **Executed 2026-08-17 — A–G complete, H partial (§1c).** The case lists below are kept as the
> design record and as the map from finding → test. Two deliberate deviations from the plan as
> written: (i) battery H's real-run scans are **all** `@pytest.mark.stress`, not just the
> non-smoke configs — a ~40 s simulation per default-suite run buys little when
> `test_run_smoke.py` already runs the same config, so the *scanner* is covered in the default set
> against synthetic records instead; (ii) the scanner itself lives in
> `harness/scan_field_record.py` and is imported by path (the `test_rosette_cf_harness.py`
> pattern) so the committed CSV and the tests share one implementation.

Batteries A–G are fast unit/characterization tests (milliseconds each, default `pytest` set).
Battery H is integration on real runs. Each case states **expected current behavior** — the
executing session pins behavior as-is (`pytest.raises`, equality on the probed values) and tags
candidate bugs; it does NOT fix `dictionary.py` (see §4 ground rules).

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
   `IndexError` (c), empty shell-grav `ValueError` (d). Plus the **reverse asymmetry** (audit #4,
   §1b): `bubble_r_arr` present with **no** derived partner → the unconditional `continue`
   (dictionary.py:639-641) drops the r-array silently, no exception, key simply absent. Assert the
   silence explicitly — it is the one branch of F5 that fails without a traceback.
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
2. **Where tests go** (as landed): `test/test_dictionary_stress.py` for the in-process batteries
   and `test/test_dictionary_stress_process.py` for battery C + H — split because C's whole point
   is the atexit/signal handlers that the in-process `no_handlers` fixture (copied from
   `test/test_metadata.py:39`) disables. Real-run scans are `@pytest.mark.stress`. Suite
   invariant: 0 *new* failures, before and after — the tree carries 3 known-red goldens pending
   the post-C3c re-baseline (`test_run_smoke`, `test_phase_boundary`, `test_mu_audit_drift`; see
   the `phii-identity/` row in `docs/dev/DOC_STATUS.md`), unrelated to this workstream.
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

## 5. Execution order — done, and what remains

1. ~~Batteries A + B~~ — done (16 tests).
2. ~~Battery C~~ — done (8 tests); resolved O5 and O6, hardened F7 to an end-to-end subprocess pin.
3. ~~Batteries D–G~~ — done (36 tests). D.7 (the real registry/param path) produced F19/F20, the
   most consequential result of the pass.
4. **Battery H — partial.** Fast config scanned and committed; **owed**: the two
   `f1edge_*` configs and one run that completes all four phases, each adding a
   `data/field_scan.csv` row via `harness/scan_field_record.py`.
5. ~~Doc + DOC_STATUS updates; commit CSV~~ — done for this pass; redo after the owed H rows.

Next session's shortest path: run the owed H configs, append the CSV rows, then check whether any
boundary landed at an index ≡ 0 (mod 10) — that is the one prediction (F21) still unobserved in
the field, and it is the concrete cost of F1.

## 6. Maintainer decisions queued (fix vs. document — do not resolve unilaterally)

**Start from the prior audit's fix set, not a blank page.** Its §3 proposes a fix for each of its
17 findings, §4 records three fixes it walked back after re-reading the code, §5 maps which fixes
must land in one patch, §6 tabulates the risks each fix introduces, and §7 gives an implementation
order. It was never landed or gated, and it is written against the old layout — so it is a
*proposal to evaluate*, not a plan of record. Its most load-bearing judgement, worth surfacing to
the maintainer verbatim: fixing the duplicate key (its #1) makes phase boundaries emit **two
snapshots at the same `t_now`**, and it spot-checked four readers plus `trinity_reader._snapshots`
for t-monotonicity assumptions (found none) — but did not read every reader. That is precisely the
kind of content change that needs the risky-change ladder here.

The decisions below are this plan's own framing; where the audit already proposes a fix, its
number is cited so the two are compared rather than re-derived.

1. ~~F1/F2 (audit #2): persist the last-written key so dedup is alignment-independent?~~
   **DONE 2026-08-17 (§1h)** — implemented as the audit's `_last_save_key`, with F4's key and F3's
   NaN behaviour deliberately untouched. Measured output-neutral on both gated configs, but *not*
   neutral by construction: see §1h's coupling subsection for the `stop_at_rCloud_nSnap` case,
   which is the one place a user can see a different stop point.
2. F4 (audit #1): is `(t_now, R2)` the right duplicate key, given phase handoffs deliberately
   exploit it? The audit proposes adding `current_phase` to the key, which by its own §6 makes
   boundaries emit two same-`t_now` records. Any change must re-read
   `run_energy_phase.py:400-419` first.
3. ~~F6: make `flush()` serialize all lines *before* opening the file (atomic-ish append)?~~
   **DONE 2026-08-17 (§1f)** — serialize-first *plus* a truncate-back rollback, because
   re-verification showed the reachable trigger is a torn I/O write, which serialize-first alone
   would not have covered. Not in the audit — new here. **The remaining half is a real decision:**
   a failed flush still *loses* that buffer's snapshots, and `main.py` only logs it
   (`logger.warning("Could not flush parameters: …")`), so a run can report success while a window
   of snapshots is missing. Should a failed flush be fatal, retried after dropping the offending
   key, or at minimum recorded in `metadata.json[termination]`?
4. ~~F7: should `load_snapshot`-created dicts skip `_register_crash_handlers`?~~ **DONE 2026-08-17
   (§1e)** — taken as `register_handlers=False` on the loader path; loading no longer writes to the
   run directory or hijacks the process's signal handlers. **The write-side half is still open**:
   the audit's `_readonly` + `_fresh_run` flags (its #8/#10) guard against an *explicit*
   `save_snapshot()`/`flush()` on a loaded dict, which still deletes the source files as a "fresh
   run" (O1). Those two flags are complementary to what landed, not superseded by it — the
   decision on whether to take them is unchanged, and
   `test_explicit_save_on_a_loaded_dict_still_works` keeps the current behavior visible.
5. F5c (audit #5/#6): `reset_keys`' NaN default vs. the profile-array branches — guard the
   branches (audit #5: emit `[]`), or stop serializing reset keys at all (audit #6:
   `reset_keys(..., exclude=True)`)?
5b. ~~**F19/F20**: guard the `_simplify_error` call so an empty curve skips the R² metric.~~
   ✅ **DONE and landed** — see §1d. Gated bit-identical (hash unchanged) on the fast config plus
   per-call identical on five curve sizes vs `HEAD`. The remaining decision it *raises*: an empty
   pair now writes `[]` while `shell_n_arr` still omits its keys — fold into item 5's pair-handling
   choice. Still owed for full ladder compliance: the same byte-comparison on the two `f1edge_*`
   configs (the fast config is the only one measured).
6. F11 (audit #6): `allow_nan=False` with a sanitize step would make files strict JSON — worth the
   content change? Note the audit measured 112 NaN-bearing lines in one run, so this is a
   *large* content diff, not a cosmetic one.
7. F15/F16/F17/F18 (audit #15/#16/#12/#17): the cleanup tier — dead `until_flush` reset, exposing
   `snapshot_interval` as a `.param` key, a newer-than-reader metadata-version warning, and
   de-duplicating the JSON-ready helper. Cheap and independent; the only question is whether they
   ride along with a substantive fix or land separately.
