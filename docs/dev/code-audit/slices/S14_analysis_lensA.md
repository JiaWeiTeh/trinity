# S14 analysis helpers — Lens A (what the code does)

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

## Provenance

**Read (whole world for this report):** the stripped copy at
`.../lens/S14_analysis/code/_analysis/check_yesno.py` (297 lines) and
`.../lens/S14_analysis/code/_analysis/__init__.py` (1 line). Citations below use the real-tree paths
`trinity/_analysis/check_yesno.py:<line>` and `trinity/_analysis/__init__.py:<line>`; line numbers are
preserved by the stripping process.

**Documented exception used:** yes. I read the real `trinity/_functions/unit_conversions.py` (grep only,
no edits) to pin the unit system, because two of my questions — whether the hardcoded `1e-20` pressure
floor is physically meaningful, and whether the hardcoded `Myr` label on the time axis is right — cannot
be answered from a doc-blind slice. Confirmed: the "AU" system is `Msun, pc, Myr`
(`unit_conversions.py:18-21`), pressure converts `g/cm/s² → Msun/pc/Myr²` with
`Pb_cgs2au = 1545441495671.806` (`unit_conversions.py:115-116`), and the AU time unit is Myr
(`s2Myr`, line 78). Nothing else from that file informs this report.

**Not read:** the real `trinity/` tree (including `trinity/_output/trinity_reader.py`), `docs/dev/`,
`test/`, `param/`, `outputs/`, `run.py`, `tools/`, `paper/`, this slice's `prose.md` / `signatures.md`,
and any other lens's report. I also ran a scratch numpy snippet (no repo imports) to confirm library
semantics I rely on — `object == None` elementwise, `np.nanmax` on all-NaN vs empty, `np.diff` on
length ≤ 1, `np.mean` of an empty bool array, and NaN comparison outcomes — under numpy 1.26.4.

I made no changes to the repository.

---

## What this module is

`trinity/_analysis/check_yesno.py` is a standalone argparse CLI, not a library. It scans an output
folder for pairs of simulation directories whose names end in `_yesPHII` / `_noPHII`
(`check_yesno.py:54-55`), loads a fixed set of six columns from each, compares the two shell-radius
trajectories `R2(t)` on a common time grid, measures how often the HII-region pressure beats the bubble
pressure in the "yes" run, and prints one of six verdicts per pair. It exits non-zero only when it
believes it has found a bug.

Its whole purpose is to answer one question about code elsewhere: *does the `include_PHII` flag actually
change anything, and if not, is that correct?* The classification logic it uses to answer that is where
the interesting problems are.

`trinity/_analysis/__init__.py` is empty after stripping (its 1 line is blank), i.e. the original file
contained at most a module docstring. It re-exports nothing, defines nothing, and imports nothing. It
exists only to make `trinity._analysis` a regular package.

## Reachability (question 1)

Within the slice the call graph is a single tree rooted at the `__main__` guard:

- `main()` — called **only** at `check_yesno.py:297`, under `if __name__ == "__main__":` (line 296).
- `pair_yes_no()` — called once, `check_yesno.py:269`, by `main`.
- `diagnose_pair()` — called once, `check_yesno.py:280`, by `main`.
- `load_run()` — called twice, `check_yesno.py:169-170`, by `diagnose_pair`.
- `_get_field()` — called six times, `check_yesno.py:92-97`, by `load_run`.
- `compare_trajectories()` — called once, `check_yesno.py:181`, by `diagnose_pair`.
- `pressure_dominance()` — called once, `check_yesno.py:185`, by `diagnose_pair`.
- Module constants: `_HERE` used at line 49; `YES_SUFFIX` / `NO_SUFFIX` used at lines 67-70.

So **no function in this slice has any in-slice caller other than through `main`, and `main` has no
caller at all except direct script execution.** The empty `__init__.py` means nothing is re-exported
from `trinity._analysis`, so a `from trinity._analysis import ...` cannot reach any of this without
naming the submodule explicitly. Two further signals point the same way: the `sys.path` bootstrap at
`check_yesno.py:48-49` prepends the repo root so that `python trinity/_analysis/check_yesno.py` can
resolve `import trinity` (line 51) — that hack is only needed for direct execution — and the module
takes its configuration from `argparse` (lines 242-262) rather than from a function signature.

**This is a dead-code candidate, and I cannot settle it.** The true caller set lives in `run.py`,
`tools/`, `paper/`, notebooks, CI config, or a maintainer's shell history, none of which I am allowed to
read. Everything I can see is consistent with "hand-run diagnostic, invoked from a terminal when
someone suspects the PHII gate is broken." Flagged only; per project rule I propose no deletion.

## Line-by-line accounting

### Module prologue (lines 1-55)

Lines 1-39 are the stripped module docstring. That docstring is **functional**, not decorative: it is
passed as the argparse `epilog` at `check_yesno.py:246`, so its text is user-facing CLI help. Under
`python -OO` it becomes `None`, which argparse accepts silently, so `-OO` quietly shrinks `--help`. I
cannot read its content, so I cannot check whether it agrees with the code — that is a job for the other
lenses.

`check_yesno.py:48-49` is the module's only global side effect:

```python
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent.parent))
```

`_HERE` is `<repo>/trinity/_analysis`, so `_HERE.parent.parent` is the repo root. This runs at **import**
time, not just under the `__main__` guard, so merely importing `trinity._analysis.check_yesno` from
anywhere mutates the interpreter's global `sys.path` — and inserts at position **0**, ahead of stdlib and
site-packages. Any top-level directory at the repo root that shares a name with an importable module
(the layout in this project includes `test/`, `tools/`, `lib/`, `docs/`, `paper/`, `param/`) would win
that name for the rest of the process. When run as a script the insert is redundant-but-harmless
(Python already put the script's own dir on the path — though that is `trinity/_analysis`, not the root,
which is exactly why the line exists). Finding S14-A-08.

`YES_SUFFIX = "_yesPHII"` / `NO_SUFFIX = "_noPHII"` (lines 54-55) are hardcoded and not overridable from
the CLI. The pairing contract is therefore a directory-naming convention produced by some sweep tool
outside my slice.

### `pair_yes_no(folder)` — lines 61-73

Iterates `find_all_simulations(folder)` (out of slice; I assume it yields a path per simulation whose
**parent directory** name carries the suffix, because line 66 takes `p.parent.name`). For each name it
strips the suffix by slice — `name[: -len(YES_SUFFIX)]` (lines 68, 70) — and buckets the path by base
name. Returns `sorted(set(yes) | set(no))` zipped with `.get(b)`, so the `None` in a returned tuple is a
**meaningful sentinel** for "partner run missing", consumed at line 164.

Edge behaviour:

- A simulation directory ending in neither suffix is **silently discarded** — no count, no warning. The
  header at line 275 reports only the number of *paired* bases, so a folder of 100 runs where 98 lack
  the suffix prints "Found 1 base name(s)" and looks healthy.
- Base names collide on `p.parent.name` only. Two runs in different subtrees whose parent dirs happen to
  share a name (`sweepA/run7_yesPHII/…` and `sweepB/run7_yesPHII/…`) map to the same base `run7`, and the
  second **silently overwrites** the first (plain dict assignment, lines 68/70). Which one survives
  depends on `find_all_simulations` iteration order. Finding S14-A-11.
- A directory named exactly `_yesPHII` yields base `""`, which pairs normally and prints an empty header.
  Harmless.
- Empty folder → empty list → handled at lines 270-272.

### `_get_field(output, name, default=np.nan)` — lines 79-87

Three behaviours worth naming.

1. **Missing column → fabricated array** (line 83): `np.full(len(output), default, dtype=float)`. This
   assumes `len(output)` is the **snapshot count**. That holds for a pandas DataFrame (`len(df)` = rows,
   `df.get(col)` = column). It is wrong for a dict-of-arrays, where `len()` is the number of *keys*, and
   the fabricated column would silently have the wrong length — which then either mismatches under the
   fancy-index sort at lines 101-103 (`IndexError`, caught as ERROR) or, worse, matches by coincidence.
   `load_output` is out of slice, so I cannot resolve which. Finding S14-A-06, medium confidence.
2. **Object columns**: `np.where(arr == None, default, arr)` (line 86). I verified that on a numpy object
   array `arr == None` is a genuine elementwise comparison returning a bool array (numpy 1.26.4), so this
   works as written; it would not on a non-object dtype, hence the guard at line 85. Note this is the
   `== None` spelling that `E711` would flag, but the project's ruff set is `F821/F811/F823/E9`, so it is
   not enforced — mentioning it only so the reconciler knows it is deliberate-looking, not accidental.
3. **No input mutation.** `np.asarray` may alias, but `np.where` allocates and `astype(float)` copies by
   default, so the returned arrays never share storage with the loaded output. Nothing in this module
   mutates its inputs in place. Anything `astype(float)` cannot parse (a string cell, say) raises
   `ValueError`, which propagates into the caller's `try` at lines 168-173 and becomes an `ERROR` verdict
   with only `str(e)` printed.

### `load_run(path)` — lines 90-105

Loads exactly six fields. **The defaults are not uniform** (lines 92-97): `t_now`, `R2`, `Pb`, `P_drive`
default to `np.nan` when absent, while `P_HII` and `P_ram` default to `0.0`. That asymmetry means a
missing `P_HII` column is indistinguishable downstream from a column of genuine zeros — see
S14-A-05 for why that matters for both verdict directions.

Conditional re-sort (lines 99-103): the sort fires only if `np.any(np.diff(t) < 0)`. Verified edge cases:
`np.diff` of a length-0 or length-1 array is empty and `np.any([])` is `False`, so short runs skip the
sort correctly. But **a NaN in `t` never triggers the sort** (`nan < 0` is `False`), and `np.interp` with
a NaN in `xp` returns NaN silently — I confirmed both. The tuple-unpacking of a generator expression at
lines 101-103 is valid Python and evaluates all six before rebinding.

`P_drive` is loaded (line 97), carried through the sort (line 102), and stored in the returned dict
(line 105) — and then **never read anywhere in the module**. It is the one field on disk that would
empirically settle the `P_drive = max(Pb, P_HII)` claim the tool prints at line 218, and the tool
discards it in favour of recomputing a proxy from `Pb`/`P_HII`/`P_ram`. Finding S14-A-10. (Similarly,
the "no" run's `Pb`, `P_ram` and `P_drive` are loaded but unused; only `no["t"]`, `no["R2"]` and
`no["P_HII"]` are ever consumed.)

### `compare_trajectories(yes, no)` — lines 111-124

Overlap window is `t_lo = max(mins)`, `t_hi = min(maxes)` (lines 113-114). On an **empty** array `.min()`
raises `ValueError` — and this call site (line 181) is *outside* the `try` block, so an empty run kills
the whole scan (S14-A-07).

The guard at line 115, `if not np.isfinite(t_lo) or not np.isfinite(t_hi) or t_hi <= t_lo`, returns the
sentinel `(nan, nan, (nan, nan))`. Note `t_hi <= t_lo` is inclusive, so a degenerate single-point overlap
is (correctly) rejected rather than fed to `linspace`. The problem is what the caller does with the
sentinel — nothing distinguishes it from a real answer; see S14-A-01.

`t_grid = np.linspace(t_lo, t_hi, 512)` (line 118) is a **hardcoded 512-point uniform-in-time grid**. Two
consequences. (a) If a run writes more than 512 snapshots the comparison is a subsample, so a short-lived
excursion in `R2` between grid points is invisible to `rel.max()`. (b) Uniform-in-`t` spends its
resolution where the trajectory is slow. A feedback bubble's interesting dynamics — the energy-driven
phase where `P_HII` could plausibly compete — are early and fast; if a run spans tens of Myr, the first
few hundred kyr get a handful of the 512 points. The grid density is also independent of cloud mass and
density, which set the phase timescales, so the effective resolution of the early phase silently degrades
for configs with longer total runtime. Finding S14-A-14.

`denom = np.maximum(np.abs(R_yes), 1e-30)` (line 122) clamps only the denominator. In AU units `R2` is in
pc, so `1e-30` pc is ~1e-12 cm — the clamp bites only at an exactly-zero radius. The normalisation is
**asymmetric**: the relative error is referenced to the "yes" run alone, so `R_yes → 0` with `R_no`
finite gives `rel ~ 1e30` (verdict DIVERGES), while the mirror-image case gives `rel ≈ 1`. For a
same-physics comparison this asymmetry is unlikely to matter in practice; I record it rather than
elevate it. `rel.mean()` (line 124) is a uniform-in-time average over the overlap, so it is
duration-weighted and dominated by the late, slowly-varying part of the trajectory — it is reported but
never used in a decision.

### `pressure_dominance(yes)` — lines 127-153

`valid = np.isfinite(Pb) & np.isfinite(P_HII) & (Pb > 0)` (line 143). `Pb > 0` is strict, so `Pb == 0`
snapshots are dropped. `P_ram` is *not* required finite here; instead line 149 replaces non-finite ram
values with `0.0` — a silent substitution that turns "ram pressure unknown" into "ram pressure zero" with
no notice to the caller.

`frac = float(np.mean((phii + pram) > pb))` (line 151) — fraction of *valid* snapshots where
`P_HII + P_ram` strictly exceeds `Pb`. Ties count as not-winning.

`max_ratio = float(np.max(phii / pb))` (line 152) — **no ram term**. Division is safe because the mask
guarantees `pb > 0`.

So the function returns two dominance measures computed against **different** numerators, and the caller
prints them adjacently (lines 194 and 195-196) and classifies on the ram-inclusive one while narrating
the ram-exclusive physics. That mismatch is finding S14-A-02.

`if not np.any(valid): return np.nan, np.nan` (lines 144-145) is the second NaN sentinel, and like the
first it is not distinguishable downstream.

### `diagnose_pair(...)` — lines 159-235

Prints the header before validating (lines 160-162), so a missing partner prints `noPHII : None` and then
`[SKIP]` (lines 164-166) → `"MISSING"`.

The `try` at lines 168-173 wraps **only** the two `load_run` calls. `except Exception as e` is broad but
correctly excludes `KeyboardInterrupt`/`SystemExit` (both `BaseException`); it prints `{e}` with no
exception type and no traceback, so a bare `KeyError('R2')` surfaces to the user as `load failed: 'R2'`.
Everything after line 175 — the two `np.nanmax` calls, `compare_trajectories`, `pressure_dominance` — is
**unprotected**, and each can raise on an empty array. Finding S14-A-07.

The two "did the flag work" measurements are deliberately asymmetric (lines 176-178): the "no" run is
checked on `np.abs(P_HII)` (any-magnitude leak), the "yes" run on the raw signed max (must be positive to
count as produced). That reads as intentional and I do not fault it. What I do fault is that **an
all-NaN `P_HII` column passes both gates**: `np.nanmax` of all-NaN returns NaN (with a RuntimeWarning
that goes nowhere), and both `nan > phii_tol` (line 200) and `nan <= phii_tol` (line 204) evaluate
`False` — I verified all four comparisons. So the most alarming possible state of the field, "every
value is NaN", produces zero complaints and the verdict is then decided by whatever the trajectory
comparison says. Finding S14-A-04.

`trajectories_match = np.isfinite(rel_max) and rel_max < r2_tol` (line 182) — strict `<`, so a value
exactly at tolerance counts as *not* matching, and a non-finite `rel_max` (including `inf` from the
`1e-30` clamp path) counts as not matching.

The verdict cascade at lines 209-234:

| condition | verdict |
|---|---|
| `bugs` non-empty (lines 200-207) | `BUG` |
| `trajectories_match and frac_phii_wins == 0` (214) | `EXPECTED` |
| `trajectories_match and frac_phii_wins > 0` (221) | `UNEXPECTED` |
| anything else (229) | `DIVERGES` |

`frac_phii_wins == 0` is a float equality — and here it is **safe**: `np.mean` of an all-`False` bool
array is exactly `0.0`, and every attainable value is an exact multiple of `1/n`. This is the one float
comparison in the module that needs no tolerance and correctly has none.

The hole is that lines 214 and 221 look exhaustive over `trajectories_match == True` but are not: if
`frac_phii_wins` is NaN, **both are False** and control reaches the `else`. The tool then declares
`DIVERGES` and prints "R2 differs by up to 0.00% — P_HII is materially changing the dynamics"
(lines 231-233) for a pair whose trajectories it just measured as *identical*. The same sink swallows
every degenerate input: no time overlap, a missing `t_now` column (all-NaN `t` → NaN `t_lo`), a missing
`R2` column (all-NaN interp → NaN `rel_max`), a missing `Pb` column (no valid snapshots → NaN `frac`).
All of them print a confident physical conclusion the code has no evidence for, with `nan%` or `0.00%`
in the number slot. There is no `INDETERMINATE` bucket. Finding S14-A-01.

Line 220 prints a hardcoded cross-file source citation, `"energy_phase_ODEs.py:253-256"`, into
user-facing output. I cannot verify it from this slice, and by construction it rots the moment that file
is edited. Finding S14-A-12.

Every path returns one of exactly six strings, all six of which are keys of the tally dict at lines
277-278, so `tally[status] += 1` (line 282) cannot `KeyError`.

### `main()` — lines 241-293

Two CLI knobs. `--tol` default `1e-3` (lines 252-255) is a relative radius tolerance — dimensionless,
so it is regime-independent, which is the right choice. `--phii-tol` default `1e-20` (lines 257-261) is
**absolute**, in AU pressure units (`Msun/pc/Myr²`). With `Pb_cgs2au ≈ 1.545e12`, an HII-region pressure
of `P/k ~ 10⁶–10⁷ K cm⁻³` (i.e. `~1e-10–1e-9` cgs) is `~10²–10³` in these units. So the floor sits ~22
orders of magnitude below any physical value, and the constant does double duty in **opposite senses**:
as a leak floor (line 200) it is effectively an exact-zero test that would call a `1e-15` numerical
residue "the `include_PHII=False` gate is leaking", and as a production floor (line 204) it is satisfied
by a physically meaningless `1e-19`. Rescaling the cloud mass or density moves physical pressures by
orders of magnitude but never near `1e-20`, so the constant does not *break* in another regime — it is
simply not measuring what its two use sites need, which a ratio against `Pb` would. Finding S14-A-09.

Folder validation (lines 264-267) is correct and reports to stderr. No pairs found → message + `exit(1)`
(lines 270-272), i.e. "nothing to check" returns the same code as "bug found".

The summary loop (lines 287-289) prints only non-zero tallies, so `ERROR: 3` does show up on screen. But
the exit code at line 293 is `sys.exit(1 if tally["BUG"] else 0)` — **`ERROR`, `MISSING`, `DIVERGES` and
`UNEXPECTED` all exit 0**. A run in which every single pair failed to load returns success. So does a run
where the tool's own text says "either the flag is being ignored despite the folder suffix, or the P_HII
field written to disk doesn't reflect what the ODE actually saw" (lines 226-228) — a description of a bug
that is nevertheless classified as not-a-bug for exit purposes. Any CI or shell `&&` chain consuming this
tool learns nothing. Finding S14-A-03.

## Summary judgement

The numerics are unremarkable and mostly careful — the one float equality in the file is the one place
where exact equality is provably correct, nothing mutates its inputs, the conditional re-sort handles
short arrays correctly, and the `Pb > 0` mask makes the division at line 152 safe. The weaknesses are all
in *verdict plumbing*: five distinct degenerate conditions (no overlap, missing `t_now`, missing `R2`,
missing `Pb`, all-NaN `P_HII`) each collapse into a confident, wrong, physically-worded conclusion, and
the exit code reports success for all of them. For a tool whose entire output is a judgement about
whether other code is broken, the failure mode "says DIVERGES with 0.00% divergence" is the expensive one.

Whether any of it runs at all is a question this slice cannot answer.

```json
[
  {
    "id": "S14-A-01",
    "file": "trinity/_analysis/check_yesno.py",
    "line": 229,
    "class": "silent-failure",
    "severity": "S2",
    "claim": "The verdict cascade has no indeterminate bucket: every NaN sentinel produced upstream falls through to the final `else` and is reported as DIVERGES with the physical conclusion 'P_HII is materially changing the dynamics', even when the tool has measured no divergence at all or has no data.",
    "evidence": "L115-116 `if not np.isfinite(t_lo) or not np.isfinite(t_hi) or t_hi <= t_lo:` / `return np.nan, np.nan, (np.nan, np.nan)`; L144-145 `if not np.any(valid):` / `return np.nan, np.nan`; L214 `elif trajectories_match and frac_phii_wins == 0:`; L221 `elif trajectories_match and frac_phii_wins > 0:`; L229-233 `else:` / `status = \"DIVERGES\"` / `print(f\"       R2 differs by up to {rel_max*100:.2f}% — \" f\"P_HII is materially changing the dynamics.\")`. Verified under numpy 1.26.4 that `nan == 0`, `nan > 0` are both False, so a NaN `frac_phii_wins` satisfies neither elif.",
    "expected": "A NaN from compare_trajectories or pressure_dominance means 'could not measure', which is a distinct outcome from 'measured a large difference'. Lines 214/221 look exhaustive over trajectories_match==True but are not.",
    "failure_scenario": "A pair whose Pb column is entirely non-positive or non-finite (so `valid` is empty and frac is NaN) but whose R2 trajectories are bit-identical prints '>> DIAGNOSIS: DIVERGES / R2 differs by up to 0.00% — P_HII is materially changing the dynamics.' The same sink also swallows: no time overlap between the two runs, a missing t_now column (all-NaN t -> NaN t_lo), and a missing R2 column (all-NaN interp -> NaN rel_max), each printing 'nan%' or '0.00%' beside a confident physics claim.",
    "repro": "Point --folder at a yes/no pair whose outputs have disjoint t ranges (or drop the Pb column from both), and observe DIVERGES with a nan/0.00% delta.",
    "confidence": "high"
  },
  {
    "id": "S14-A-02",
    "file": "trinity/_analysis/check_yesno.py",
    "line": 151,
    "class": "divergence",
    "severity": "S2",
    "claim": "The dominance predicate includes ram pressure, but the classification narrative it drives asserts the ram-free coupling P_drive = max(Pb, P_HII) and states 'P_HII exceeds Pb'. A snapshot where P_HII < Pb but P_HII + P_ram > Pb flips the verdict from EXPECTED to UNEXPECTED and produces a printed statement that is literally false.",
    "evidence": "L151 `frac = float(np.mean((phii + pram) > pb))` vs L152 `max_ratio = float(np.max(phii / pb))` (no ram term); L217-218 `print(f\"       Pb dominates P_HII at every snapshot, so \" f\"P_drive=max(Pb,P_HII)=Pb identically in both runs.\")`; L224-225 `print(f\"       P_HII exceeds Pb in {frac_phii_wins*100:.2f}% of \" f\"snapshots, yet R2(t) matches to {rel_max:.1e}.\")`.",
    "expected": "The statistic used to decide whether identical trajectories are expected should test the same quantity the coupling uses. Either the predicate should be `phii > pb`, or the narrative and the printed max ratio (L152, L194) should include P_ram.",
    "failure_scenario": "In a momentum-dominated or high-ram regime with P_HII comfortably below Pb, frac_phii_wins > 0 purely from the ram term, so a correct run is labelled UNEXPECTED and the operator is told 'either the flag is being ignored despite the folder suffix, or the P_HII field written to disk doesn't reflect what the ODE actually saw' (L226-228) — sending them to hunt a nonexistent bug. Note the loaded P_drive column, which would settle this empirically, is discarded (see S14-A-10). Whether the real ODE's P_drive includes ram is out of slice.",
    "repro": "A pair where max(P_HII/Pb) < 1 (printed at L194) yet the frac line at L195-196 is non-zero — the two printed numbers are then mutually inconsistent by construction.",
    "confidence": "high"
  },
  {
    "id": "S14-A-03",
    "file": "trinity/_analysis/check_yesno.py",
    "line": 293,
    "class": "silent-failure",
    "severity": "S2",
    "claim": "Only the BUG tally affects the exit code. ERROR, MISSING, UNEXPECTED and DIVERGES all exit 0, so a scan in which every pair failed to load reports success to any automated caller.",
    "evidence": "L293 `sys.exit(1 if tally[\"BUG\"] else 0)`; L171-173 `except Exception as e:` / `print(f\"  [ERROR] load failed: {e}\")` / `return \"ERROR\"`; L164-166 `[SKIP] missing partner run` / `return \"MISSING\"`. Contrast L272 `sys.exit(1)` for the 'no pairs found' case.",
    "expected": "A diagnostic gate should fail on states it cannot vouch for. UNEXPECTED in particular is described by the tool's own text (L226-228) as a bug indicator, and ERROR means it never measured anything.",
    "failure_scenario": "Wired into CI or a `&&` chain after a sweep, the tool returns 0 when the output format changed and every load raised, or when half the partner runs never completed — the pipeline reports green. Additionally, 'no pairs found' (L272) exits 1, the same code as 'bug found', so the two most different outcomes are indistinguishable by exit status.",
    "repro": "Run against a folder of _yesPHII/_noPHII pairs with corrupted or renamed output files; observe ERROR lines then `echo $?` -> 0.",
    "confidence": "high"
  },
  {
    "id": "S14-A-04",
    "file": "trinity/_analysis/check_yesno.py",
    "line": 200,
    "class": "numerical",
    "severity": "S2",
    "claim": "An all-NaN P_HII column passes both bug gates, because np.nanmax returns NaN and every comparison against NaN is False.",
    "evidence": "L176 `no_phii_max = float(np.nanmax(np.abs(no[\"P_HII\"])))`; L178 `yes_phii_max = float(np.nanmax(yes[\"P_HII\"]))`; L200 `if no_phii_max > phii_tol:`; L204 `if yes_phii_max <= phii_tol:`. Verified: `np.nanmax([nan, nan])` -> nan with a RuntimeWarning, and `nan > 1e-20` and `nan <= 1e-20` are both False.",
    "expected": "An entirely-NaN pressure field is the strongest possible signal that something is wrong and should be reported, not skipped by both directional checks.",
    "failure_scenario": "A run whose P_HII solver produced NaN throughout gets no BUG flag; the verdict falls through to the trajectory branches and, since frac is also NaN (no valid mask entries once isfinite(P_HII) fails at L143), lands in the DIVERGES sink of S14-A-01. The RuntimeWarning numpy emits is not captured or surfaced.",
    "repro": "Feed a yes/no pair whose P_HII column is all NaN; no BUG is reported.",
    "confidence": "high"
  },
  {
    "id": "S14-A-05",
    "file": "trinity/_analysis/check_yesno.py",
    "line": 95,
    "class": "silent-failure",
    "severity": "S2",
    "claim": "P_HII and P_ram default to 0.0 when the column is absent, while t_now/R2/Pb/P_drive default to NaN. The 0.0 default makes 'field not written to disk' indistinguishable from 'field written and zero', in both verdict directions.",
    "evidence": "L95-96 `P_HII = _get_field(out, \"P_HII\", default=0.0)` / `P_ram = _get_field(out, \"P_ram\", default=0.0)` against L92-94,97 which pass no default (np.nan); L81-83 `arr = output.get(name)` / `if arr is None:` / `return np.full(len(output), default, dtype=float)`; L149 `pram = np.where(np.isfinite(P_ram[valid]), P_ram[valid], 0.0)` applies the same substitution to non-finite ram values.",
    "expected": "The tool's entire job is to decide whether P_HII is present and non-zero; silently synthesising a zero column for a missing field removes the evidence the decision needs.",
    "failure_scenario": "If an output format simply omits P_HII for noPHII runs, the leak check at L200 passes trivially and contributes to an EXPECTED verdict. In the other direction, if both runs omit it, L204-207 fires and blames a specific physics cause — 'P_HII is not being computed or n_IF_Str=0 always' — when the actual cause is a missing column, misdirecting the investigation. Non-finite P_ram is likewise silently read as zero ram pressure at L149.",
    "repro": "Load an output that lacks the P_HII column; _get_field returns zeros with no warning.",
    "confidence": "medium"
  },
  {
    "id": "S14-A-06",
    "file": "trinity/_analysis/check_yesno.py",
    "line": 83,
    "class": "other",
    "severity": "S2",
    "claim": "The missing-column fallback sizes its array with len(output), which is the snapshot count only if the loader returns a row-oriented object (e.g. a DataFrame). For a dict-of-arrays it is the number of keys.",
    "evidence": "L81-83 `arr = output.get(name)` / `if arr is None:` / `return np.full(len(output), default, dtype=float)`.",
    "expected": "The synthesised column must have the same length as the real columns it will be indexed and interpolated alongside.",
    "failure_scenario": "If load_output returns a mapping of column-name -> array, a missing P_HII yields an array of length == number-of-columns. That either raises IndexError at the fancy-index sort (L101-103, caught as ERROR) or, if lengths coincide, silently produces a wrong-length field used in pressure_dominance. I cannot resolve this: trinity/_output/trinity_reader.py is outside my slice, and the `.get()`/`len()` pair is consistent with a pandas DataFrame, in which case the code is correct as written.",
    "repro": "Inspect the return type of trinity._output.trinity_reader.load_output; if it is not row-length-addressable by len(), this line is wrong.",
    "confidence": "medium"
  },
  {
    "id": "S14-A-07",
    "file": "trinity/_analysis/check_yesno.py",
    "line": 176,
    "class": "other",
    "severity": "S2",
    "claim": "The try/except covers only the two load_run calls; every analysis call after it is unprotected, so one empty run raises an uncaught ValueError and aborts the whole scan, discarding results for all remaining pairs.",
    "evidence": "L168-173 `try:` / `yes = load_run(yes_path)` / `no = load_run(no_path)` / `except Exception as e:` — the block ends there; L176 `no_phii_max = float(np.nanmax(np.abs(no[\"P_HII\"])))` and L113-114 `t_lo = max(yes[\"t\"].min(), no[\"t\"].min())` sit outside it. Verified: `np.nanmax(np.array([]))` and `np.array([]).min()` both raise ValueError ('zero-size array to reduction operation').",
    "expected": "A per-pair diagnostic loop should isolate per-pair failures; the ERROR status and tally slot exist precisely for that.",
    "failure_scenario": "A simulation that crashed early and wrote a zero-row output file causes a traceback out of main; pairs alphabetically after it are never examined, and the summary/exit code never print. The failure is loud, but the loop's fault isolation is defeated for exactly the class of broken run this tool is meant to survey.",
    "repro": "Place a zero-row output in one _yesPHII folder and run; the scan dies at that pair.",
    "confidence": "high"
  },
  {
    "id": "S14-A-08",
    "file": "trinity/_analysis/check_yesno.py",
    "line": 49,
    "class": "state",
    "severity": "S4",
    "claim": "The module mutates global interpreter state at import time, prepending the repo root to sys.path at position 0 — ahead of stdlib and site-packages — even when it is imported as part of the package rather than run as a script.",
    "evidence": "L48-49 `_HERE = Path(__file__).resolve().parent` / `sys.path.insert(0, str(_HERE.parent.parent))`, executed at module scope, before `from trinity._output.trinity_reader import ...` at L51.",
    "expected": "A path bootstrap needed only for `python trinity/_analysis/check_yesno.py` should not fire for `import trinity._analysis.check_yesno`, and appending would be safer than inserting at index 0.",
    "failure_scenario": "Any process that imports this module thereafter resolves top-level names against the repo root first. Repo-root directories sharing a name with an importable module (the project layout lists test/, tools/, lib/, docs/, paper/, param/) would shadow it for the rest of the process. I cannot check which of those are importable packages, so I rate this hygiene rather than latent breakage.",
    "repro": "`python -c \"import sys, trinity._analysis.check_yesno as m; print(sys.path[0])\"` prints the repo root.",
    "confidence": "medium"
  },
  {
    "id": "S14-A-09",
    "file": "trinity/_analysis/check_yesno.py",
    "line": 258,
    "class": "regime",
    "severity": "S2",
    "claim": "A single absolute constant, phii_tol = 1e-20 AU pressure units, serves two opposite-sense tests. As a leak floor it is effectively an exact-zero test; as a production floor it is satisfied by physically meaningless values. A ratio against Pb would be regime-independent.",
    "evidence": "L257-261 `--phii-tol`, `type=float, default=1e-20`, help 'Absolute P_HII floor below which values count as zero (default: 1e-20, in TRINITY AU pressure units)'; used in opposite senses at L200 `if no_phii_max > phii_tol:` ('gate is leaking') and L204 `if yes_phii_max <= phii_tol:` ('never produced P_HII'). Unit system pinned from trinity/_functions/unit_conversions.py:18-21 (Msun, pc, Myr) and :115-116 (Pb_cgs2au = 1545441495671.806), so an HII-region pressure of ~1e-10 cgs is ~1e2 in these units — the floor sits ~22 orders below anything physical.",
    "expected": "The 'did the gate leak' test wants 'negligible compared to Pb'; the 'did it produce P_HII' test wants 'dynamically relevant compared to Pb'. Neither is an absolute 1e-20.",
    "failure_scenario": "Any numerical residue left by the include_PHII=False path — 1e-15 in AU units, i.e. ~1e-27 cgs, utterly irrelevant dynamically — is reported as 'include_PHII=False gate is leaking'. Conversely a yes-run producing 1e-19 counts as having produced P_HII. Changing cloud mass or density rescales physical pressures by orders of magnitude but never brings them near 1e-20, so the constant does not break across regimes; it simply never measures relevance. Contrast --tol at L253, which is correctly dimensionless.",
    "repro": "Compare the printed 'max P_HII' (L192) against the printed Pb-relative ratio (L194) for any run; the pass/fail at L200/L204 keys off the former, which has no relation to the latter.",
    "confidence": "high"
  },
  {
    "id": "S14-A-10",
    "file": "trinity/_analysis/check_yesno.py",
    "line": 97,
    "class": "deadcode",
    "severity": "S4",
    "claim": "P_drive is loaded, re-sorted and returned but never read anywhere in the module — despite being the one on-disk field that would directly verify the P_drive = max(Pb, P_HII) claim the tool prints as its EXPECTED justification.",
    "evidence": "L97 `P_drive = _get_field(out, \"P_drive\")`; L101-103 include it in the re-sort tuple; L105 `return dict(t=t, R2=R2, Pb=Pb, P_HII=P_HII, P_ram=P_ram, P_drive=P_drive)` — and no other reference to P_drive exists in the file. Compare L218 `P_drive=max(Pb,P_HII)=Pb identically in both runs.`, which is asserted rather than measured.",
    "expected": "Either use the field to check the coupling directly, or do not load it.",
    "failure_scenario": "No incorrect result; wasted I/O and a missed check. Related: the 'no' run's Pb, P_ram and P_drive are all loaded and never used (only no['t'], no['R2'] and no['P_HII'] are consumed). Flagged only, not proposed for deletion.",
    "repro": "grep P_drive in the file: lines 97, 102, 105 only.",
    "confidence": "high"
  },
  {
    "id": "S14-A-11",
    "file": "trinity/_analysis/check_yesno.py",
    "line": 68,
    "class": "silent-failure",
    "severity": "S4",
    "claim": "pair_yes_no silently discards simulations whose parent directory matches neither suffix, and silently overwrites on base-name collisions between different subtrees, because the base key is derived from p.parent.name alone.",
    "evidence": "L65-70 `for p in find_all_simulations(folder):` / `name = p.parent.name` / `if name.endswith(YES_SUFFIX): yes_by_base[name[: -len(YES_SUFFIX)]] = p` / `elif name.endswith(NO_SUFFIX): no_by_base[...] = p` — no else branch, plain dict assignment.",
    "expected": "A scan tool should report what it ignored, and should key on something unique (the full relative path) rather than a single directory component.",
    "failure_scenario": "sweepA/run7_yesPHII and sweepB/run7_yesPHII both map to base 'run7'; whichever find_all_simulations yields second wins, and the pair reported may cross sweeps — comparing sweepB's yes run against sweepA's no run, which would produce a spurious DIVERGES. Separately, a folder where most runs lack the suffix prints only the paired count (L275) and looks complete.",
    "repro": "Create two subtrees with identically-named _yesPHII dirs; only one appears in the output.",
    "confidence": "high"
  },
  {
    "id": "S14-A-12",
    "file": "trinity/_analysis/check_yesno.py",
    "line": 220,
    "class": "citation",
    "severity": "S3",
    "claim": "A hardcoded cross-file line-number citation is printed into user-facing diagnostic output, where it will silently rot as the referenced file is edited.",
    "evidence": "L219-220 `print(f\"       Identical R2(t) is the correct consequence of \" f\"max()-coupling in energy_phase_ODEs.py:253-256.\")`.",
    "expected": "A file-level reference (or a function name) survives edits; a line range does not.",
    "failure_scenario": "Any insertion above line 253 of energy_phase_ODEs.py makes the tool print a citation to unrelated code as the justification for its EXPECTED verdict. I cannot verify the current accuracy of the range — that file is outside my slice — so the claim here is about the fragility of the reference, not its correctness today.",
    "repro": "Compare the printed range against the current contents of the referenced file.",
    "confidence": "high"
  },
  {
    "id": "S14-A-13",
    "file": "trinity/_analysis/__init__.py",
    "line": 1,
    "class": "deadcode",
    "severity": "S4",
    "claim": "Nothing in this slice imports check_yesno; the package __init__ is empty and re-exports nothing; the module's only entry point is its __main__ guard. The slice is a dead-code candidate on the evidence available to me.",
    "evidence": "trinity/_analysis/__init__.py is a single blank line after stripping — no imports, no __all__, no definitions. In check_yesno.py the sole non-nested call to main() is L296-297 `if __name__ == \"__main__\":` / `main()`; the sys.path bootstrap at L48-49 exists to make direct script execution resolve `import trinity` at L51; all configuration arrives via argparse (L242-262) rather than a callable API.",
    "expected": "n/a — flag only, per project rule. Not proposed for deletion.",
    "failure_scenario": "No runtime failure. The true caller set can only be settled outside this slice: run.py, tools/, paper/, docs, notebooks, CI config, or a maintainer's own script could invoke `python trinity/_analysis/check_yesno.py` or `python -m trinity._analysis.check_yesno`. I read none of those and make no claim that it is unused — only that it has no in-slice caller and no package-level export.",
    "repro": "grep -rn 'check_yesno' across run.py, tools/, paper/, test/, docs/ and any CI config to settle it.",
    "confidence": "medium"
  },
  {
    "id": "S14-A-14",
    "file": "trinity/_analysis/check_yesno.py",
    "line": 118,
    "class": "numerical",
    "severity": "S2",
    "claim": "The comparison grid is a hardcoded 512 points uniform in t over the overlap window, which subsamples long or densely-written runs and under-resolves the early fast-expansion phase where P_HII is most likely to compete.",
    "evidence": "L118 `t_grid = np.linspace(t_lo, t_hi, 512)`; L119-120 `R_yes = np.interp(t_grid, yes[\"t\"], yes[\"R2\"])` / `R_no = np.interp(t_grid, no[\"t\"], no[\"R2\"])`; L124 `return float(rel.max()), float(rel.mean()), ...`.",
    "expected": "Either compare at the union of the runs' own snapshot times, or scale the grid to the data; and consider log spacing in t given the phase structure.",
    "failure_scenario": "Two directions. (a) A run with more than 512 snapshots is compared on a subsample, so a transient excursion in R2 between grid points never reaches rel.max() and a real divergence is reported as EXPECTED. (b) The grid is uniform in t while the interesting dynamics are early and fast, so for a config with a long total runtime the energy-driven phase receives a handful of the 512 points; the effective early-phase resolution therefore depends on cloud mass and density through the phase timescales, with no compensation. Separately, if the two runs write snapshots at different cadences, linear interpolation error between them is a floor on rel_max that is not obviously below the 1e-3 default tolerance, risking false DIVERGES.",
    "repro": "Re-run compare_trajectories with 512 replaced by len(yes['t']) + len(no['t']) and compare rel_max on a densely-sampled pair.",
    "confidence": "medium"
  }
]
```
