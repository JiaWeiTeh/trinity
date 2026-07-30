# S14 analysis helpers — reconciled (A vs B)

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

## What I read, and what that constrains

I read exactly two files:

- `raw/S14_analysis_lensA.md` — Lens A, working from a comment- and docstring-stripped copy of
  `trinity/_analysis/check_yesno.py` (297 lines) and `trinity/_analysis/__init__.py` (1 line). A never
  saw a comment. A declares one documented exception: it grepped the real
  `trinity/_functions/unit_conversions.py` to pin the AU unit system (Msun, pc, Myr;
  `Pb_cgs2au ≈ 1.545e12`), and it ran scratch numpy snippets to confirm library semantics.
- `raw/S14_analysis_lensB.md` — Lens B, working from a comment/docstring dump for the same two files.
  B never saw a line of code, a signature, a test, or a `.param`.

**I did not read any source.** Not `trinity/`, not `test/`, not `docs/dev/`, not the stripped `code/`
copies, not `prose.md` or `signatures.md`, not any other slice. Everything below is a diff of two
blind accounts.

What that means concretely:

- Where A and B agree, I can state a corroborated fact with high confidence, because two
  non-communicating readers of *disjoint halves of the same file* landed in the same place.
- Where they disagree, I can localise the disagreement precisely but **cannot adjudicate it from
  source**. Every such case is marked for orchestrator lookup with a named file and a specific check.
- Where only one lens speaks, I distinguish two very different situations: (i) the other lens was
  *structurally blind* to that evidence — B cannot see argparse `help=` strings or the 512 in
  `linspace`, A cannot see a `# Check B:` label — from (ii) the other lens *looked and reported
  otherwise*. Only (ii) is a real conflict. A's per-function line-by-line accounting is detailed
  enough that I can usually tell which applies; where I cannot, I say so.
- **This is an infra-tier slice: there is no Lens C.** Nothing here has been executed, so no finding
  below carries runtime confirmation. "High confidence" means "two blind readings converge", not
  "observed".

One structural note that shapes the whole reconciliation: the two lenses saw **complementary
halves** of this file, and the file's documentary content is unusually concentrated (B: the module
docstring at L3–38 carries the entire verdict taxonomy, the usage examples, the physics hypothesis,
and the only external citation). So B's account is essentially "what the module docstring promises"
and A's is "what the 297 lines do". The interesting findings are exactly where those two do not meet.

---

## Correspondence table

Axis key: **A≡B** corroborated · **A≠B** doc-drift · **A-only** undocumented behaviour ·
**B-only** unimplemented/stale/unverifiable claim. "A-only (B silent-corroborated)" means A found a
behaviour and B independently flagged the *absence of documentation* for exactly that input class —
the two lenses converging on the same gap from opposite sides.

| # | Claim | Lens(es) | Axis | Verdict |
|---|---|---|---|---|
| R-01 | `frac_phii_wins` counts `P_HII + P_ram > Pb`; `max_ratio` is `P_HII/Pb` with no ram; the verdict narrative asserts ram-free `max(Pb,P_HII)` | A-02 ≡ B-01,B-02,B-03 | **A≡B** | **Corroborated defect, rank 1.** Code and docstring agree with each other; both lenses independently conclude the metric does not test the physics the verdict claims. |
| R-02 | Verdict cascade has no indeterminate bucket: every NaN sentinel lands in `else:` → `DIVERGES` with the printed line "R2 differs by up to 0.00% — P_HII is materially changing the dynamics" | A-01 (B-08 silent-corroborated) | **A-only** | **Upheld.** Highest true cost in the slice. B flagged the same degenerate inputs as undocumented but guessed the wrong mechanism (crash/drop, not confident lie). |
| R-03 | All-NaN `P_HII` passes both bug gates (`nan > tol` and `nan <= tol` are both False) | A-04 | **A-only** | Upheld. B's prose has no NaN handling anywhere (B §6: "There are **no** documented preconditions"). |
| R-04 | Exit non-zero only on `BUG`; `ERROR`/`MISSING`/`UNEXPECTED`/`DIVERGES` all exit 0; "no pairs found" exits 1, same as BUG | A-03 ≡ B-04 | **A≡B** | **Corroborated defect, rank 2.** Code matches the documented contract exactly; both lenses call the contract wrong for `UNEXPECTED`. A adds two statuses B could not know about. |
| R-05 | `_get_field` defaults are asymmetric: `P_HII`/`P_ram` → `0.0`, `t_now`/`R2`/`Pb`/`P_drive` → NaN; missing column indistinguishable from genuine zero | A-05 ≡ B-07 | **A≡B** | **Corroborated defect, rank 3.** B predicted the failure blind ("if that default is 0.0 …"); A confirms it *is* 0.0 for exactly the two fields that matter. |
| R-06 | `--phii-tol` default `1e-20` (absolute, AU pressure units) does double duty in opposite senses: leak floor and production floor | A-09 (B-05, B-07 silent-corroborated) | **A-only** | Upheld. A supplies the number B said was missing; A's judgement is that the number is the wrong *kind* of threshold. B independently rated "≈ 0" and "dominate" as verdict-critical and unquantified. |
| R-07 | `try/except` wraps only the two `load_run` calls; `nanmax`, `compare_trajectories`, `pressure_dominance` are unprotected → one empty run aborts the whole scan | A-07 (B-08 silent-corroborated) | **A-only** | Upheld. Defeats the per-pair `ERROR` isolation the tally slot exists for. |
| R-08 | Hardcoded 512-point grid, uniform in `t`, over the overlap window | A-14 (B §3.3 silent-corroborated: "Silent on … how many sample points") | **A-only** | Upheld, medium. B named the exact silence; A filled it in and judged it under-resolves the early fast phase. |
| R-09 | Missing-column fallback sizes with `len(output)` — snapshot count only for a row-oriented object | A-06 | **A-only** | **Unresolved.** Needs `trinity/_output/trinity_reader.py`. Open question 1. |
| R-10 | `pair_yes_no` silently discards non-suffixed dirs and silently overwrites on `p.parent.name` collisions across subtrees | A-11 (B §3.1 silent-corroborated: docstring "silent on … how `base_name` is derived") | **A-only** | Upheld. Discard half is hygiene; collision half can mis-pair two sweeps. |
| R-11 | Line-pinned citation `energy_phase_ODEs.py:253-256` | A-12 ≡ B-10 | **A≡B** | **Corroborated, and reconciliation adds a fact neither lens had:** the citation appears **twice** — in the module docstring (B, full path) *and* in a printed f-string at L219-220 (A, bare filename). Both lenses independently called it the most drift-prone reference form. |
| R-12 | The documented verdict taxonomy has **four** entries; the code returns **six** statuses | B (4: EXPECTED/BUG/UNEXPECTED/DIVERGES) vs A (6: + MISSING, ERROR) | **A≠B** | **Doc-drift, reconciler-derived.** The docstring is incomplete, not wrong. `MISSING` and `ERROR` are undocumented *and* exit 0 (R-04). |
| R-13 | `compare_trajectories` returns one value (B: "return max rel diff") vs three (A: `rel_max, rel_mean, (t_lo, t_hi)`) | B §3.3 vs A L124 | **A≠B** | **Doc-drift, reconciler-derived.** Docstring documents 1 of 3 returns; the undocumented `rel_mean` is computed and printed but never used in a decision (A). |
| R-14 | Documented usage block shows `--tol 1e-4` only; actual `--tol` default is `1e-3` and `--phii-tol` is never mentioned in any docstring | B-05 vs A L253/L258 | **A≠B** | **Doc-drift, reconciler-derived**, medium. The flag that gates Checks B and C is absent from the only user-facing summary. Caveat: B could not see argparse `help=` strings. |
| R-15 | "Not one unit is stated anywhere in the slice" (B-06) vs `--phii-tol` help text "in TRINITY AU pressure units" (A) | B-06 vs A L257-261 | **A≠B** | **B overstated; demoted.** Resolution is a lens boundary: units *are* stated once, in an argparse help string, which is code and so absent from B's dump. B's claim holds for docstrings/comments only. |
| R-16 | Verdict taxonomy has no docstring at the point of decision; no documented precedence | B-09 (A supplies the answer) | **B-only, resolved** | Contract gap upheld, but A settles the substance: precedence is `BUG` → `EXPECTED` → `UNEXPECTED` → `else DIVERGES`, i.e. BUG-first, which is sensible. Demoted to S4. Also: 4 of 7 functions have docstrings (`load_run`, `diagnose_pair`, `main` have none). |
| R-17 | Bare `# noqa: E711` with no rationale (B-13) | B-13 (A supplies the rationale) | **B-only, resolved** | Upheld as doc hygiene only. A verified the `== None` spelling is *required* for elementwise comparison on a numpy object array, and notes E711 is not in this project's ruff set (`F821/F811/F823/E9`) — so the suppression is inert as well as unexplained. |
| R-18 | `P_drive` is loaded, re-sorted, returned, and never read | A-10 | **A-only** | Upheld. Sharpened by B: the docstring's central claim *is* `P_drive = max(Pb, P_HII)`, and the tool discards the one on-disk field that would measure it. |
| R-19 | `sys.path.insert(0, repo_root)` executes at import time, not only under `__main__` | A-08 | **A-only** | Upheld, hygiene. Cannot rate higher without knowing which repo-root dirs are importable. Open question 8. |
| R-20 | No in-slice caller; no documented programmatic caller; script-only one-off scaffolding | A-13 ≡ B-12 | **A≡B** | **Corroborated.** A: only entry is the `__main__` guard; empty `__init__` re-exports nothing; config arrives via argparse. B: only documented consumer is a human at a shell against a specific past experiment's output dir. Neither lens can settle it — Open question 4. |
| R-21 | `trinity/_analysis/__init__.py` is empty | A-13 ≡ B-11 | **A≡B** | **Corroborated from both sides:** A — one blank line, no imports/`__all__`/definitions; B — zero prose entries, the file does not even appear as a section in the dump. The package makes no claim about itself in code *or* prose. |
| R-22 | "Fiducial massive cluster … dominates by orders of magnitude" is unquantified; no `.param` named; only concrete artefact is an untracked `outputs/` path | B-14 | **B-only** | Upheld (A is docstring-blind and structurally could not corroborate). Open question 7. |
| R-23 | Checks labelled A–D appear in source order B, C, A, D | B-15 | **B-only** | Upheld, low. A was structurally blind — the labels exist only in comments. Not "A looked and it isn't there". |
| R-24 | The module docstring is the argparse `epilog`, so under `python -OO` it becomes `None` | A (prose, unfiled) + B (taxonomy lives *only* there) | **A≠B-derived** | **Reconciler-derived.** Combining the two: under `-OO` the sole documentation of the four verdicts silently vanishes from `--help`. Neither lens could see this alone. |
| — | `pair_yes_no` docstring "missing partner → None" — does None replace the path or the tuple? (B could not tell) | B §3.1, A L71-73 | **A≡B, no defect** | Resolved: `None` occupies the path slot, consumed at L164 → `MISSING`. Docs accurate. |
| — | `frac_phii_wins == 0` float equality | A only | **A-only, no defect** | A verified `np.mean` of an all-False bool array is exactly `0.0` and every attainable value is a multiple of `1/n`. Correctly needs no tolerance. Recording it so nobody re-flags it. |
| — | Nothing in the module mutates its inputs | A only | **A-only, no defect** | Recorded. |

---

## Corroborated defects, ranked

These are the entries where two lenses, blind to each other and reading disjoint halves of the file,
independently concluded the same thing is broken. **That agreement is the strongest signal this
method produces**, and it is why these rank above findings with higher raw cost.

### Rank 1 — R-01: the dominance metric does not test the physics the verdict claims

Both lenses arrived here from opposite directions and neither could have seen the other's evidence.

> **Lens A** (code only): "`frac = float(np.mean((phii + pram) > pb))` (line 151) — fraction of
> *valid* snapshots where `P_HII + P_ram` strictly exceeds `Pb`. … `max_ratio = float(np.max(phii /
> pb))` (line 152) — **no ram term**. … the caller prints them adjacently … and classifies on the
> ram-inclusive one while narrating the ram-exclusive physics."

> **Lens B** (prose only): "`frac_phii_wins` is claimed to compare **`P_HII + P_ram`** against `Pb`;
> `max_ratio` is claimed to be `max(P_HII / Pb)` — **without `P_ram`**. By the docstring's own
> definitions a caller can be handed `frac_phii_wins > 0` together with `max_ratio < 1`, i.e. 'P_HII
> wins in some snapshots' alongside 'P_HII never reached Pb'."

The code and the docstring **agree with each other** — the docstring even flags the choice ("we use
the more permissive P_HII+P_ram comparison everywhere, which is a superset"). So this is not
doc-drift. It is a design defect that both readings expose independently, and each lens contributes a
piece the other lacks:

- **B supplies the physics** A could not see: the module docstring claims *two phase-dependent* drive
  laws — `max(Pb, P_HII)` in the energy/implicit phases, `max(Pb, P_HII + P_ram)` in transition only.
  The metric applies the transition-phase superset **in every phase**.
- **A supplies the printed consequence** B could not see: line 224-225 prints
  `"P_HII exceeds Pb in {frac*100:.2f}% of snapshots, yet R2(t) matches to {rel_max:.1e}."` — a
  sentence that is *literally false* whenever the ram term drove the count. Line 217-218 prints the
  opposite narrative, `"P_drive=max(Pb,P_HII)=Pb identically in both runs."`
- **A also supplies the trigger threshold**, which answers B's complaint that Check D's "dominate" is
  never quantified: the classification keys on `frac_phii_wins == 0` vs `> 0`. There is no threshold.
  **A single snapshot** where `P_HII + P_ram` beats `Pb` — even in a phase whose documented drive law
  ignores `P_ram` entirely — flips the verdict from `EXPECTED` to `UNEXPECTED`.

Both lenses independently wrote the same failure scenario: an operator is told to hunt a
`max()`-coupling bug that does not exist. B: "sending an investigator after a non-existent
max()-coupling failure — the exact anomaly-chasing this script was written to prevent." A: "sending
them to hunt a nonexistent bug."

The clean tell is that the tool prints both numbers side by side, so a `max P_HII/Pb` below 1 next to
a non-zero `P_HII exceeds Pb in X%` is self-contradicting output the tool emits without noticing.

### Rank 2 — R-04: the exit code cannot express the states the tool exists to find

> **Lens A**: "`sys.exit(1 if tally[\"BUG\"] else 0)` — **`ERROR`, `MISSING`, `DIVERGES` and
> `UNEXPECTED` all exit 0**. A run in which every single pair failed to load returns success."

> **Lens B**: "'# Non-zero exit only on BUG (data inconsistency). EXPECTED is a # physics conclusion,
> not a failure.' … `UNEXPECTED` — which the module docstring describes as a state that contradicts
> the coupling model — therefore exits zero, silently, by the documented contract. Nothing says this
> is intended for `UNEXPECTED` specifically; the rationale given covers only `EXPECTED`."

The code does exactly what the comment says, so there is no drift. Both lenses instead fault the
contract, and B's reading of the comment is the sharper indictment: the *stated rationale* covers
only `EXPECTED`, and the author appears never to have decided what `UNEXPECTED` should exit with.
A extends it to two statuses B could not know exist (`ERROR`, `MISSING` — see R-12), and adds the
inversion that "no pairs found" exits **1**, the same code as "bug found" — so the two most different
outcomes are indistinguishable to any caller.

### Rank 3 — R-05: a missing column is silently synthesised as zero, in the one field the tool judges

B predicted this blind, from a docstring that does not even name the default:

> **Lens B**: "`_get_field`'s docstring promises a None-to-default substitution without naming the
> default … A yesPHII run whose P_HII field is missing or all-None is substituted to the default; **if
> that default is 0.0** it is indistinguishable from a genuinely absent P_HII and the tool reports BUG
> for an I/O gap."

> **Lens A**: "**The defaults are not uniform** (lines 92-97): `t_now`, `R2`, `Pb`, `P_drive` default
> to `np.nan` when absent, while `P_HII` and `P_ram` default to `0.0`. That asymmetry means a missing
> `P_HII` column is indistinguishable downstream from a column of genuine zeros."

It is 0.0, for exactly the two fields whose presence the tool is built to adjudicate. A confirms both
directions of B's prediction: with the column absent in the "no" run the leak check at L200 passes
trivially and feeds an `EXPECTED` verdict; with it absent in both, L204-207 fires and prints a
*specific physics cause* — "P_HII is not being computed or n_IF_Str=0 always" — for what is actually
a missing column. A adds a third instance of the same substitution: L149 replaces non-finite `P_ram`
with `0.0` inside `pressure_dominance`, turning "ram pressure unknown" into "ram pressure zero".

### Rank 4 — R-11: a line-pinned cross-file citation, in two places

> **Lens A**: "Line 220 prints a hardcoded cross-file source citation, `\"energy_phase_ODEs.py:253-256\"`,
> into user-facing output. … by construction it rots the moment that file is edited."

> **Lens B**: "'(see ``trinity/phase1_energy/energy_phase_ODEs.py:253-256``)' … The **entire hypothesis
> under test** rests on this reference. It cites a **line range in another module**, which cannot be
> validated from prose and is the most drift-prone form of cross-reference."

Reconciling these produces a fact neither lens had: the same line range is hardcoded **twice** — once
in the module docstring (B's copy carries the full package path) and once inside a printed f-string
that the tool emits as the *justification for its `EXPECTED` verdict* (A's copy is the bare
filename). Two copies of a line-pinned reference drift independently. Whether the range is *already*
stale is Open question 2.

### Rank 5 — R-20 / R-21: a package that documents nothing, called by nothing visible

> **Lens A**: "no function in this slice has any in-slice caller other than through `main`, and `main`
> has no caller at all except direct script execution. … `trinity/_analysis/__init__.py` is empty
> after stripping … It re-exports nothing, defines nothing, and imports nothing."

> **Lens B**: "**No prose anywhere in the slice names a programmatic caller, an importing module, or a
> test.** `__init__.py` says nothing, so there is no documented re-export. The prose's own framing
> ('Hypothesis under test', a hard-coded example output path) reads as a one-off investigation tool
> that was committed into the package tree."

Both lenses stop short of calling it dead, correctly — the caller set lives in files neither was
allowed to read. This is a one-step grep for the orchestrator (Open question 4). The `__init__.py`
result is stronger: it is empty of *both* code and prose, which is as complete a corroboration as
this method can produce for a file.

---

## Where the two lenses disagree about the same line or quantity

These are the highest-value entries, because a disagreement localises the drift to a specific line.
**I cannot adjudicate any of them from source**; each is listed with the one-step check.

| Quantity | Lens A says | Lens B says | My reading |
|---|---|---|---|
| **Number of verdict statuses** | Six: `BUG`, `EXPECTED`, `UNEXPECTED`, `DIVERGES`, `MISSING`, `ERROR` — "all six of which are keys of the tally dict at lines 277-278" | Four: "EXPECTED … BUG … UNEXPECTED … DIVERGES", stated "**only here**, in the module docstring" | Not a contradiction — an **incomplete docstring**. `MISSING` and `ERROR` are real, user-visible, tallied, printed in the summary, and exit 0, and appear in no documentation. Filed as R-12. |
| **`compare_trajectories` return arity** | Three: `float(rel.max()), float(rel.mean()), (t_lo, t_hi)`; sentinel is `(nan, nan, (nan, nan))` | One: "Claimed return: the maximum **relative** difference" | Docstring documents 1 of 3. The undocumented `rel_mean` is computed, printed, and never used in a decision (A). Filed as R-13. |
| **Units in this file** | Quotes the `--phii-tol` help text: "Absolute P_HII floor … **in TRINITY AU pressure units**" | "**not one unit is stated** anywhere in the slice … No units appear in any of the 30 prose entries" | **B is overstated and I demote it.** The disagreement is a lens boundary, not drift: argparse `help=` strings are code literals, invisible to a comment/docstring extractor. B's claim is true of docstrings and comments; false of the file. R-15, demoted to S4. |
| **The trajectory tolerance** | `--tol` default is `1e-3` | Usage example shows `--tol 1e-4`, "value shown **only as an example** — no default documented" | Compatible (example ≠ default) but the docstring never states the default, and never mentions `--phii-tol` at all — the flag that gates two of the four checks. Filed as R-14. |
| **Where the ODE citation lives** | A printed f-string at L219-220, bare filename | The module docstring at L3-38, full package path | Two independent copies of the same line-pinned range. Filed as R-11. |
| **What happens on no time overlap** | Returns the NaN sentinel, which falls through to `DIVERGES` and prints a confident physics conclusion | "the tool either **crashes mid-report or silently drops the pair**" | B was guessing from silence and guessed the two *benign* options. The actual behaviour (A) is the third and worst one. This near-miss is why R-02 ranks where it does. |

None of these six requires reading the whole file to settle; each is a single-line check named in Open
questions.

---

## Undocumented behaviour (A-only) — the contract gaps that matter

R-02 is the most expensive finding in the slice and is single-lens on *behaviour*, so it does not
qualify for the corroborated ranking above. It belongs here, at the top.

**R-02 — the verdict cascade has no indeterminate bucket.** A traced five distinct degenerate inputs
— no time overlap, missing `t_now`, missing `R2`, missing `Pb`, all-NaN `P_HII` — into a single sink:

> "if `frac_phii_wins` is NaN, **both are False** and control reaches the `else`. The tool then
> declares `DIVERGES` and prints 'R2 differs by up to 0.00% — P_HII is materially changing the
> dynamics' … for a pair whose trajectories it just measured as *identical*. … All of them print a
> confident physical conclusion the code has no evidence for, with `nan%` or `0.00%` in the number
> slot. There is no `INDETERMINATE` bucket."

B did not see this, but B independently listed the exact same input classes as undocumented (§9.5:
"zero snapshots (`frac_phii_wins` denominator); `Pb == 0` (the `max_ratio` divisor); two runs with no
overlapping time window") and §6: "There are **no documented preconditions**." So both lenses
converge on "these inputs are unhandled"; only A can say how they fail.

Also A-only, in descending order of cost: **R-03** (all-NaN `P_HII` passes both bug gates, because
`nan > tol` and `nan <= tol` are both False — the most alarming possible field state produces zero
complaints); **R-06** (`--phii-tol = 1e-20` is ~22 orders of magnitude below any physical AU
pressure, and serves as both a leak floor and a production floor, i.e. it is an exact-zero test in
one direction and satisfied by meaningless values in the other — note A contrasts this with `--tol`,
which is correctly dimensionless); **R-07** (only the two `load_run` calls are inside the `try`, so
one zero-row output aborts the entire scan and the summary never prints); **R-08** (hardcoded
512-point uniform-in-`t` grid, which subsamples runs with more snapshots and under-resolves the early
fast phase where `P_HII` could compete — and whose interpolation error between mismatched cadences is
a floor on `rel_max` that is not obviously below the `1e-3` default); **R-10** (base names key on
`p.parent.name` alone, so identically-named run dirs in two sweeps silently overwrite and the tool
may compare sweep B's yes-run against sweep A's no-run); **R-09** (`len(output)` sizing, unresolvable
without the loader); **R-18** and **R-19** (hygiene).

---

## Where the true cost exceeds the rubric severity

Both lenses rated this slice 0×S1 and I uphold that, but not by inheritance — by the following
argument, which I want on the record because it is the judgement most likely to be wrong.

`trinity/_analysis/check_yesno.py` produces **no physical output**. It writes no `dictionary.jsonl`,
feeds no ODE, and sets no parameter. Nothing it does can make a published number wrong. By the letter
of the rubric — "S1 = results-wrong on configs run today" — nothing in this file can be S1, and I
have not inflated anything to pretend otherwise.

But the rubric's notion of "results" does not fit a diagnostic. This module's entire product *is* a
verdict about whether the rest of the code is broken, and the verdict is delivered in confident
physical prose to a human who is, by construction, already suspicious. Three findings have a true
cost well above their rubric severity:

1. **R-02 at S2 is the expensive one.** The tool prints `DIVERGES` with "R2 differs by up to 0.00% —
   P_HII is materially changing the dynamics" for a pair whose trajectories it measured as identical.
   An investigator who trusts that sentence goes and edits the `max()`-coupling in
   `energy_phase_ODEs.py` — a real, S1-grade change to real physics, one step removed from a finding
   the rubric caps at S2. The rubric measures the blast radius of the file; the damage here happens in
   another file, done by a person.
2. **R-05 + R-02 together form the worst path in the slice**, and neither lens composed them because
   each held only half. If the output writer omits `P_HII` when `include_PHII=False` (Open question
   6), the "no" run's leak check passes trivially, and the tool certifies `EXPECTED` — "Pb dominates
   P_HII at every snapshot" — for a pair where the field was never written at all. **The audit tool
   blesses the exact failure it exists to catch**, and exits 0 (R-04). That composite is S2 by the
   rubric and catastrophic by consequence, because it is silent in the direction that produces no
   further investigation.
3. **R-01 at S2 costs an investigator's week, not a number.** Both lenses wrote the same scenario
   independently: `UNEXPECTED` fires, the tool prints "either the flag is being ignored despite the
   folder suffix, or the P_HII field written to disk doesn't reflect what the ODE actually saw", and
   someone goes looking for a bug that ram pressure invented.

The inverse also holds, and argues *against* inflating: every one of these costs is paid only if
someone runs the tool, and R-20 says nobody demonstrably does. Severity here is genuinely bimodal on
Open question 4. If the grep finds no caller, this whole slice is S4 hygiene on an uncalled script.
If it finds one — a CI job, a `tools/` wrapper, a documented workflow — then R-04's "always exits 0"
and R-02's confident lie are being consumed by something, and the S2 ratings are floors rather than
ceilings. **I have rated everything below on the assumption that the tool is run by hand,
occasionally, by a maintainer. Settle Open question 4 before acting on any of it.**

---

## Open questions — each a single decidable lookup

Ordered by how much of the reconciliation they unblock.

1. **Does the folder flag `-f` actually exist?** *Neither lens confirmed it.* B transcribes both
   documented invocations as `python -m trinity._analysis.check_yesno -f outputs/trinity_fiducial_yesno`;
   A describes argparse setup at `check_yesno.py:242-262` and quotes `--tol` and `--phii-tol` by name
   but **never states the folder argument's flag spelling**, referring only to "folder validation
   (lines 264-267)". **Check:** in `trinity/_analysis/check_yesno.py:242-262`, does the
   `add_argument` for the folder declare a short option `-f`? If it declares only `--folder`, then
   *every documented invocation in the module docstring fails to parse* — which would be a live S3
   (the tool's only usage instructions do not run) and should be filed.
2. **Is the citation already stale?** **Check:** does `trinity/phase1_energy/energy_phase_ODEs.py`
   lines 253-256 currently contain the `max(Pb, P_HII)` drive-pressure coupling? Both the module
   docstring and the printed f-string at `check_yesno.py:219-220` pin that exact range. If it has
   drifted, R-11 upgrades from "fragile" to "wrong today".
3. **Does the loader return rows or keys?** **Check:** the return type of `load_output` in
   `trinity/_output/trinity_reader.py`. Specifically: is `len(obj)` the snapshot count (pandas
   DataFrame) or the column count (dict of arrays)? This decides R-09 outright — DataFrame ⇒ the code
   at `check_yesno.py:83` is correct as written and R-09 should be dropped; dict-of-arrays ⇒ the
   fabricated column is length-6 and R-09 is live.
4. **Is this script called by anything?** **Check:**
   `grep -rn 'check_yesno' run.py tools/ test/ paper/ docs/ param/ .github/ .pre-commit-config.yaml
   pyproject.toml`. Decides R-20, and per the paragraph above, recalibrates the severity of the whole
   slice. A hit in CI or a `Makefile` promotes R-04 (always exits 0) substantially.
5. **Does the suffix matching work at all?** **Check:** in `trinity/_output/` (the module imported at
   `check_yesno.py:51`), does `find_all_simulations` yield paths whose `.parent.name` is the run
   directory name — or the run directory itself? A explicitly assumed the former from `p.parent.name`
   at L66 and flagged the assumption. If it yields the directory, every `.parent.name` is the *sweep*
   directory, no suffix ever matches, and the tool always reports "Found 0 base name(s)" — which
   would be S1-for-this-tool and would also explain why nothing calls it.
6. **Is the missing-column path reachable?** **Check:** in the output writer under `trinity/_output/`,
   are `P_HII`, `P_ram`, and `P_drive` always written, or omitted when `include_PHII=False`? If they
   are omitted, R-05 is live rather than hypothetical, and the composite failure in §"true cost"
   item 2 is a real path.
7. **Is the documented experiment reproducible?** **Check:** does any tracked file under `param/`
   define a `yesno` sweep producing `outputs/trinity_fiducial_yesno`, and does any `.param` set
   `include_PHII`? Decides R-22 and tells you whether "fiducial" has a tracked definition at all.
   (Note `outputs/` is untracked scratch per project conventions, so the documented invocation is not
   reproducible from a clean checkout regardless.)
8. **Is the `sys.path` insert dangerous or merely untidy?** **Check:** do any of the repo-root
   directories `test/`, `tools/`, `lib/`, `docs/`, `paper/`, `param/` contain an `__init__.py` or a
   top-level module name that could shadow an installed package? All-no ⇒ R-19 stays S4; any-yes ⇒
   it becomes latent breakage for any process that imports this module.
9. **Does the `include_PHII` key exist under that name?** **Check:** grep
   `trinity/_input/default.param` and the schema in `trinity/_input/` for `include_PHII`. B names it
   as "the only named configuration key in the slice"; A never saw it (it appears only in prose). If
   the key has been renamed, the docstring's whole hypothesis is stale.
10. **Is the transition-phase drive law real?** **Check:** grep for `P_ram` in the phase modules —
    does any phase use `max(Pb, P_HII + P_ram)`? This is the premise B transcribed and A could not
    see. If no phase uses the ram-inclusive form, R-01 is worse than stated: the metric's "superset"
    is a superset of nothing and every ram-driven `UNEXPECTED` is a pure false positive.

---

```json
[
  {
    "id": "S14-R-01",
    "file": "trinity/_analysis/check_yesno.py",
    "line": 151,
    "class": "divergence",
    "severity": "S2",
    "claim": "The statistic that drives the EXPECTED/UNEXPECTED verdict compares P_HII + P_ram against Pb in every phase, while both the printed narrative and the second returned metric use the ram-free coupling P_drive = max(Pb, P_HII). A single snapshot in which ram pressure alone carries the sum above Pb flips the verdict and produces a printed sentence that is literally false.",
    "evidence": "CORROBORATED, both lenses blind to each other. Lens A (code only): 'frac = float(np.mean((phii + pram) > pb))' (L151) vs 'max_ratio = float(np.max(phii / pb))' (L152) — 'no ram term'; and the printed verdict text at L217-218 'P_drive=max(Pb,P_HII)=Pb identically in both runs.' and L224-225 'P_HII exceeds Pb in {frac*100:.2f}% of snapshots'. A: 'the caller ... classifies on the ram-inclusive one while narrating the ram-exclusive physics.' Lens B (prose only), reading the same thing from the docstring at L128-138: 'frac_phii_wins is claimed to compare P_HII + P_ram against Pb; max_ratio is claimed to be max(P_HII / Pb) — without P_ram ... a caller can be handed frac_phii_wins > 0 together with max_ratio < 1.' B additionally supplies the phase structure A could not see (module docstring L3-38): 'In the energy / implicit phases TRINITY uses P_drive = max(Pb, P_HII) ... and in the transition phase P_drive = max(Pb, P_HII + P_ram)', while the metric 'we use the more permissive P_HII+P_ram comparison everywhere, which is a superset'. A supplies the trigger threshold B said was missing: classification keys on 'frac_phii_wins == 0' vs '> 0', so there is no dominance threshold at all — one snapshot suffices.",
    "expected": "The dominance predicate should test the same quantity as the coupling whose failure the verdict alleges — either compare phii > pb, or apply the ram term only in the phase whose documented drive law includes it, or restate the UNEXPECTED text in terms of the superset criterion and acknowledge the false-positive rate.",
    "failure_scenario": "In any regime where ram pressure is non-negligible while P_HII sits comfortably below Pb, frac_phii_wins becomes non-zero purely from the ram term. A correct run is labelled UNEXPECTED and the operator is told 'either the flag is being ignored despite the folder suffix, or the P_HII field written to disk doesn't reflect what the ODE actually saw' (L226-228) — sent to hunt a max()-coupling bug that does not exist. Both lenses wrote this scenario independently.",
    "repro": "Any pair where the printed max P_HII/Pb (L194) is below 1 while the printed frac line (L195-196) is non-zero: the two numbers the tool prints side by side are then mutually inconsistent by construction.",
    "confidence": "high"
  },
  {
    "id": "S14-R-02",
    "file": "trinity/_analysis/check_yesno.py",
    "line": 229,
    "class": "silent-failure",
    "severity": "S2",
    "claim": "The verdict cascade has no indeterminate bucket. Every NaN sentinel produced upstream — no time overlap, missing t_now, missing R2, missing Pb, all-NaN P_HII — falls through both elif arms into the final else and is reported as DIVERGES together with the physical conclusion 'P_HII is materially changing the dynamics', even when the tool measured no divergence at all or measured nothing.",
    "evidence": "Lens A (code only), verified against numpy 1.26.4: 'L115-116 if not np.isfinite(t_lo) or not np.isfinite(t_hi) or t_hi <= t_lo: return np.nan, np.nan, (np.nan, np.nan)'; 'L144-145 if not np.any(valid): return np.nan, np.nan'; 'L214 elif trajectories_match and frac_phii_wins == 0'; 'L221 elif trajectories_match and frac_phii_wins > 0'; 'L229-233 else: status = \"DIVERGES\" ... print(f\"R2 differs by up to {rel_max*100:.2f}% — P_HII is materially changing the dynamics.\")'. A: 'lines 214 and 221 look exhaustive over trajectories_match == True but are not ... There is no INDETERMINATE bucket.' Lens B could not see the code but independently listed the same inputs as undocumented (§9.5: 'zero snapshots (frac_phii_wins denominator); Pb == 0 (the max_ratio divisor); two runs with no overlapping time window') and §6: 'There are no documented preconditions.' NOTE the near-miss: B guessed the degenerate path would 'either crash mid-report or silently drop the pair' — the actual behaviour is the third and worst option.",
    "expected": "A NaN from compare_trajectories or pressure_dominance means 'could not measure', which is a distinct outcome from 'measured a large difference'. An INDETERMINATE status, tallied and reported, with no physical narration.",
    "failure_scenario": "A pair whose Pb column is entirely non-positive or non-finite (valid mask empty, frac NaN) but whose R2 trajectories are bit-identical prints '>> DIAGNOSIS: DIVERGES / R2 differs by up to 0.00% — P_HII is materially changing the dynamics.' An investigator who trusts that sentence edits the max()-coupling in energy_phase_ODEs.py — a real physics change driven by a diagnostic that had no data.",
    "repro": "Point the folder flag at a yes/no pair whose outputs have disjoint t ranges, or drop the Pb column from both, and observe DIVERGES beside a nan% or 0.00% delta.",
    "confidence": "high"
  },
  {
    "id": "S14-R-03",
    "file": "trinity/_analysis/check_yesno.py",
    "line": 200,
    "class": "numerical",
    "severity": "S2",
    "claim": "An entirely-NaN P_HII column passes both bug gates, because np.nanmax of all-NaN returns NaN and every comparison against NaN is False in both directions.",
    "evidence": "Lens A only (Lens B's prose contains no NaN handling anywhere, and B §6 records 'There are no documented preconditions'). A, verified under numpy 1.26.4: 'L176 no_phii_max = float(np.nanmax(np.abs(no[\"P_HII\"])))'; 'L178 yes_phii_max = float(np.nanmax(yes[\"P_HII\"]))'; 'L200 if no_phii_max > phii_tol:'; 'L204 if yes_phii_max <= phii_tol:'; 'np.nanmax([nan, nan]) -> nan with a RuntimeWarning, and nan > 1e-20 and nan <= 1e-20 are both False.' A: 'the most alarming possible state of the field, \"every value is NaN\", produces zero complaints.'",
    "expected": "An all-NaN pressure field is the strongest available signal that something upstream is broken and should raise a BUG or an explicit data-quality status, not slip past both directional checks.",
    "failure_scenario": "A run whose P_HII solver produced NaN throughout gets no BUG flag; frac is also NaN (the isfinite mask at L143 empties), so the verdict lands in the DIVERGES sink of S14-R-02 and reports a confident physics conclusion. The RuntimeWarning numpy emits is neither captured nor surfaced.",
    "repro": "Feed a yes/no pair whose P_HII column is all NaN; no BUG is reported and the pair is diagnosed DIVERGES.",
    "confidence": "high"
  },
  {
    "id": "S14-R-04",
    "file": "trinity/_analysis/check_yesno.py",
    "line": 293,
    "class": "silent-failure",
    "severity": "S2",
    "claim": "Only the BUG tally affects the exit code. ERROR, MISSING, UNEXPECTED and DIVERGES all exit 0, so a scan in which every pair failed to load reports success; and 'no pairs found' exits 1, the same code as 'bug found'.",
    "evidence": "CORROBORATED. Lens A (code only): 'L293 sys.exit(1 if tally[\"BUG\"] else 0)'; 'L171-173 except Exception as e: print(f\"  [ERROR] load failed: {e}\") return \"ERROR\"'; 'L164-166 [SKIP] missing partner run -> return \"MISSING\"'; 'Contrast L272 sys.exit(1) for the no-pairs-found case.' A: 'A run in which every single pair failed to load returns success.' Lens B (prose only), quoting the comment at L291-292: '# Non-zero exit only on BUG (data inconsistency). EXPECTED is a # physics conclusion, not a failure.' B: 'UNEXPECTED — which the module docstring describes as a state that contradicts the coupling model — therefore exits zero, silently, by the documented contract. Nothing says this is intended for UNEXPECTED specifically; the rationale given covers only EXPECTED.' Code and comment agree; both lenses fault the contract.",
    "expected": "An exit-status mapping decided for all statuses, with a distinct non-zero code for 'could not measure' (ERROR/MISSING) and an explicit, reasoned choice for UNEXPECTED; 'no pairs found' should not share a code with 'bug found'.",
    "failure_scenario": "Wired into CI or a && chain after a sweep, the tool returns 0 when the output format changed and every load raised, or when half the partner runs never completed — the pipeline reports green. The most scientifically interesting outcome the tool can produce, UNEXPECTED, is visible only in stdout nobody reads.",
    "repro": "Run against a folder of _yesPHII/_noPHII pairs with corrupted or renamed output files; observe ERROR lines, then echo $? -> 0.",
    "confidence": "high"
  },
  {
    "id": "S14-R-05",
    "file": "trinity/_analysis/check_yesno.py",
    "line": 95,
    "class": "silent-failure",
    "severity": "S2",
    "claim": "P_HII and P_ram default to 0.0 when the column is absent, while t_now/R2/Pb/P_drive default to NaN. The 0.0 default makes 'field never written to disk' indistinguishable from 'field written and genuinely zero' — in the one quantity the tool exists to adjudicate, and in both verdict directions.",
    "evidence": "CORROBORATED, and B predicted it blind from a docstring that never names the default. Lens B: '_get_field's docstring promises a None-to-default substitution without naming the default ... A yesPHII run whose P_HII field is missing or all-None is substituted to the default; if that default is 0.0 it is indistinguishable from a genuinely absent P_HII and the tool reports BUG for an I/O gap.' Lens A (code only) confirms the value and the asymmetry: 'L95-96 P_HII = _get_field(out, \"P_HII\", default=0.0) / P_ram = _get_field(out, \"P_ram\", default=0.0) against L92-94,97 which pass no default (np.nan)'; 'L81-83 arr = output.get(name) / if arr is None: return np.full(len(output), default, dtype=float)'. A adds a third instance: 'L149 pram = np.where(np.isfinite(P_ram[valid]), P_ram[valid], 0.0)' silently reads unknown ram pressure as zero.",
    "expected": "The absence of a column the tool's verdict depends on should be an explicit, reported condition, not a synthesised value indistinguishable from a physical measurement.",
    "failure_scenario": "If the output writer omits P_HII for noPHII runs, the leak check at L200 passes trivially and feeds an EXPECTED verdict — the audit tool certifies the exact failure it exists to catch. In the other direction, if both runs omit it, L204-207 fires and blames a specific physics cause, 'P_HII is not being computed or n_IF_Str=0 always', when the actual cause is a missing column.",
    "repro": "Load an output that lacks the P_HII column; _get_field returns zeros with no warning and no status change.",
    "confidence": "high"
  },
  {
    "id": "S14-R-06",
    "file": "trinity/_analysis/check_yesno.py",
    "line": 258,
    "class": "regime",
    "severity": "S2",
    "claim": "A single absolute constant, phii_tol = 1e-20 in AU pressure units, serves two opposite-sense tests: as a leak floor it is effectively an exact-zero test, and as a production floor it is satisfied by physically meaningless values. A ratio against Pb would be regime-independent.",
    "evidence": "Lens A (code only, using its declared unit-system exception): 'L257-261 --phii-tol, type=float, default=1e-20 ... used in opposite senses at L200 if no_phii_max > phii_tol: (gate is leaking) and L204 if yes_phii_max <= phii_tol: (never produced P_HII). Unit system pinned from trinity/_functions/unit_conversions.py:18-21 (Msun, pc, Myr) and :115-116 (Pb_cgs2au = 1545441495671.806), so an HII-region pressure of ~1e-10 cgs is ~1e2 in these units — the floor sits ~22 orders below anything physical.' Lens B, blind to the number, independently rated exactly these two checks as verdict-critical and unquantified: 'Check B's P_HII ~ 0 everywhere ... \"~\" is never given a tolerance' and 'Check D's does Pb dominate ... \"dominate\" is never thresholded', calling them 'the two verdict-critical criteria and neither is quantified'. A also notes the contrast with --tol (L253), which is correctly dimensionless.",
    "expected": "The leak test wants 'negligible compared to Pb'; the production test wants 'dynamically relevant compared to Pb'. Neither is an absolute 1e-20; both should be ratios against Pb, and both should be documented with their values.",
    "failure_scenario": "Any numerical residue left by the include_PHII=False path — 1e-15 in AU units, i.e. ~1e-27 cgs and utterly irrelevant dynamically — is reported as 'include_PHII=False gate is leaking' and exits 1. Conversely a yes-run producing 1e-19 counts as having produced P_HII. Rescaling cloud mass or density never brings physical pressures near 1e-20, so the constant does not break across regimes — it simply never measures relevance.",
    "repro": "Compare the printed max P_HII (L192) against the printed Pb-relative ratio (L194) for any run; the pass/fail at L200/L204 keys off the former, which has no relation to the latter.",
    "confidence": "high"
  },
  {
    "id": "S14-R-07",
    "file": "trinity/_analysis/check_yesno.py",
    "line": 176,
    "class": "other",
    "severity": "S2",
    "claim": "The try/except covers only the two load_run calls; every analysis call after it is unprotected, so one empty run raises an uncaught ValueError and aborts the whole scan, discarding results for all remaining pairs and never printing the summary or exit status.",
    "evidence": "Lens A (code only), verified: 'L168-173 try: yes = load_run(yes_path) / no = load_run(no_path) / except Exception as e: — the block ends there; L176 no_phii_max = float(np.nanmax(np.abs(no[\"P_HII\"]))) and L113-114 t_lo = max(yes[\"t\"].min(), no[\"t\"].min()) sit outside it. np.nanmax(np.array([])) and np.array([]).min() both raise ValueError.' A: 'the loop's fault isolation is defeated for exactly the class of broken run this tool is meant to survey.' Lens B corroborates only the silence: 'No prose covers ... zero snapshots (frac_phii_wins denominator)' and records no documented preconditions.",
    "expected": "A per-pair diagnostic loop should isolate per-pair failures; the ERROR status and its tally slot exist precisely for that, so the try should span the whole per-pair body.",
    "failure_scenario": "A simulation that crashed early and wrote a zero-row output file causes a traceback out of main; pairs alphabetically after it are never examined, and the summary and exit code never print. The failure is loud, but a survey tool aborting on the first broken run is a survey tool that cannot survey broken runs.",
    "repro": "Place a zero-row output in one _yesPHII folder and run; the scan dies at that pair.",
    "confidence": "high"
  },
  {
    "id": "S14-R-08",
    "file": "trinity/_analysis/check_yesno.py",
    "line": 118,
    "class": "numerical",
    "severity": "S2",
    "claim": "The trajectory comparison uses a hardcoded 512-point grid, uniform in t over the overlap window, which subsamples runs with more snapshots and under-resolves the early fast-expansion phase where P_HII is most likely to compete.",
    "evidence": "Lens A (code only): 'L118 t_grid = np.linspace(t_lo, t_hi, 512)'; 'L119-120 R_yes = np.interp(t_grid, yes[\"t\"], yes[\"R2\"]) / R_no = np.interp(...)'. A: 'Uniform-in-t spends its resolution where the trajectory is slow ... the effective resolution of the early phase silently degrades for configs with longer total runtime.' Lens B, blind to the code, named the exact silence: the compare_trajectories docstring is 'Silent on: which run is interpolated onto which grid; how many sample points; what the denominator of \"rel\" is; units of R2 and t; what happens when the two runs have no overlapping window.' A answers all five.",
    "expected": "Compare at the union of the runs' own snapshot times, or scale the grid to the data; consider log spacing in t given the phase structure. Whatever is chosen should be stated in the docstring.",
    "failure_scenario": "(a) A run with more than 512 snapshots is compared on a subsample, so a transient excursion in R2 between grid points never reaches rel.max() and a real divergence is reported as EXPECTED. (b) If the two runs write at different cadences, linear interpolation error is a floor on rel_max that is not obviously below the 1e-3 default tolerance, risking a false DIVERGES.",
    "repro": "Re-run compare_trajectories with 512 replaced by len(yes['t']) + len(no['t']) and compare rel_max on a densely-sampled pair.",
    "confidence": "medium"
  },
  {
    "id": "S14-R-09",
    "file": "trinity/_analysis/check_yesno.py",
    "line": 83,
    "class": "other",
    "severity": "S2",
    "claim": "The missing-column fallback sizes its fabricated array with len(output), which equals the snapshot count only if the loader returns a row-oriented object such as a pandas DataFrame. For a dict-of-arrays it is the number of columns.",
    "evidence": "Lens A only: 'L81-83 arr = output.get(name) / if arr is None: / return np.full(len(output), default, dtype=float)'. A explicitly declines to resolve it: 'load_output is out of slice ... the .get()/len() pair is consistent with a pandas DataFrame, in which case the code is correct as written.' Lens B could not see this line; B's related observation is that _get_field's docstring is 'Silent on ... behaviour when the field is entirely absent versus present-but-None'. UNRESOLVED FROM THE LENSES — see open question 3.",
    "expected": "The synthesised column must have the same length as the real columns it will be fancy-indexed and interpolated alongside.",
    "failure_scenario": "If load_output returns a mapping of column-name -> array, a missing P_HII yields an array of length == number-of-columns (six here). That either raises IndexError at the re-sort (L101-103, caught as ERROR) or, if the lengths coincide by accident, silently produces a wrong-length field consumed by pressure_dominance.",
    "repro": "Inspect the return type of load_output in trinity/_output/trinity_reader.py; if len() is not the row count, this line is wrong.",
    "confidence": "medium"
  },
  {
    "id": "S14-R-10",
    "file": "trinity/_analysis/check_yesno.py",
    "line": 68,
    "class": "silent-failure",
    "severity": "S2",
    "claim": "pair_yes_no keys runs on p.parent.name alone, so identically-named run directories in different subtrees silently overwrite one another and the reported pair may cross sweeps; simulations whose parent matches neither suffix are silently discarded with no count or warning.",
    "evidence": "Lens A (code only): 'L65-70 for p in find_all_simulations(folder): / name = p.parent.name / if name.endswith(YES_SUFFIX): yes_by_base[name[: -len(YES_SUFFIX)]] = p / elif name.endswith(NO_SUFFIX): no_by_base[...] = p — no else branch, plain dict assignment.' A: 'a folder of 100 runs where 98 lack the suffix prints \"Found 1 base name(s)\" and looks healthy.' Lens B corroborates the contract gap from the docstring: pair_yes_no's contract is 'Silent on: what it takes as input; how a run directory is recognised as yes vs no ...; how base_name is derived; ordering'. NOTE: I rate this S2 rather than A's S4, because the collision half produces a wrong verdict (a mis-paired comparison), not merely a reporting gap; it is masked by requiring duplicate parent-directory names across subtrees, which is a layout neither lens evidenced. The discard half alone would be S4.",
    "expected": "Key on something unique — the full relative path — and report the count of ignored directories alongside the count of pairs.",
    "failure_scenario": "sweepA/run7_yesPHII and sweepB/run7_yesPHII both map to base 'run7'; whichever find_all_simulations yields second wins, so the tool may compare sweep B's yes-run against sweep A's no-run and report a spurious DIVERGES that reflects nothing but two different configurations.",
    "repro": "Create two subtrees with identically-named _yesPHII dirs; only one appears in the output, and which one depends on iteration order.",
    "confidence": "medium"
  },
  {
    "id": "S14-R-11",
    "file": "trinity/_analysis/check_yesno.py",
    "line": 220,
    "class": "citation",
    "severity": "S3",
    "claim": "The same line-pinned cross-file citation, energy_phase_ODEs.py:253-256, is hardcoded in two independent places — the module docstring and a printed f-string that the tool emits as the justification for its EXPECTED verdict. Both copies rot on any edit above line 253 of the referenced file.",
    "evidence": "CORROBORATED, and reconciling the two lenses reveals a fact neither had: there are two copies. Lens A (code only) found the printed one: 'L219-220 print(f\"Identical R2(t) is the correct consequence of max()-coupling in energy_phase_ODEs.py:253-256.\")' — bare filename. Lens B (prose only) found the docstring one: '(see ``trinity/phase1_energy/energy_phase_ODEs.py:253-256``)' — full package path. B: 'The entire hypothesis under test rests on this reference ... the most drift-prone form of cross-reference.' A: 'by construction it rots the moment that file is edited.' Neither lens could verify the range's current accuracy.",
    "expected": "Reference by symbol name (the function or the variable computing P_drive) rather than by line range, in both places — and ideally in one place, with the printed message deriving from the docstring.",
    "failure_scenario": "Lines shift in energy_phase_ODEs.py; the tool prints a citation to unrelated code as the justification for its verdict, and a reader who follows it cannot confirm the max(Pb, P_HII) claim on which the entire diagnosis rests.",
    "repro": "Open trinity/phase1_energy/energy_phase_ODEs.py at lines 253-256 and check whether the max(Pb, P_HII) coupling is there.",
    "confidence": "high"
  },
  {
    "id": "S14-R-12",
    "file": "trinity/_analysis/check_yesno.py",
    "line": 3,
    "class": "other",
    "severity": "S3",
    "claim": "The documented verdict taxonomy lists four outcomes; the code emits six. MISSING and ERROR are user-visible, tallied, printed in the summary, and documented nowhere — and both exit 0.",
    "evidence": "RECONCILER-DERIVED from an A-vs-B mismatch about the same quantity. Lens B transcribed the taxonomy verbatim from the module docstring and stated it exists nowhere else: 'EXPECTED ... BUG ... UNEXPECTED ... DIVERGES'; 'The taxonomy is stated only here, in the module docstring.' Lens A, code-only, enumerated the return paths: 'Every path returns one of exactly six strings, all six of which are keys of the tally dict at lines 277-278', naming MISSING (L164-166, missing partner) and ERROR (L171-173, load failure) in addition to B's four. Neither lens flagged the mismatch, because neither could see both halves.",
    "expected": "The module docstring's taxonomy should enumerate all six statuses, with each one's exit-code consequence stated (see S14-R-04).",
    "failure_scenario": "A user reads the --help epilog, sees four verdicts, and reads a report containing '[SKIP] missing partner run' and '[ERROR] load failed'. Neither appears in the documentation, so the user cannot tell whether these are failures, warnings, or expected states — and the exit code says they are fine.",
    "repro": "Compare the verdict list in the module docstring (check_yesno.py:3-38) with the keys of the tally dict at lines 277-278.",
    "confidence": "high"
  },
  {
    "id": "S14-R-13",
    "file": "trinity/_analysis/check_yesno.py",
    "line": 112,
    "class": "other",
    "severity": "S3",
    "claim": "compare_trajectories' docstring documents one return value; the function returns three. The undocumented rel_mean is computed and printed but never used in any decision, and the undocumented (t_lo, t_hi) overlap window is the only way a caller could tell a real answer from the NaN sentinel.",
    "evidence": "RECONCILER-DERIVED from an A-vs-B mismatch about the same line. Lens B (prose only), on the L112 docstring: 'Interpolate R2 onto the overlapping time window, return max rel diff.' — 'Claimed return: the maximum relative difference.' Lens A (code only): 'L124 return float(rel.max()), float(rel.mean()), ...' and the guard at L115-116 'return np.nan, np.nan, (np.nan, np.nan)' — a 3-tuple whose third element is the overlap window. A on the second element: 'rel.mean() ... is reported but never used in a decision.'",
    "expected": "A Returns block naming all three values, or a narrower return. The overlap window in particular is the datum that would let the caller distinguish 'no overlap' from 'measured' — the distinction S14-R-02 shows is missing.",
    "failure_scenario": "A maintainer reusing the function unpacks one value and gets a tuple, or (worse) reads the docstring, assumes the sentinel is a scalar NaN, and reproduces the S14-R-02 fall-through in new code.",
    "repro": "Compare the docstring at check_yesno.py:112 with the return statements at 115-116 and 124.",
    "confidence": "high"
  },
  {
    "id": "S14-R-14",
    "file": "trinity/_analysis/check_yesno.py",
    "line": 3,
    "class": "other",
    "severity": "S3",
    "claim": "The module docstring's usage block — the only user-facing summary, and the argparse epilog — documents --tol only, with an example value (1e-4) that differs from the real default (1e-3), and never mentions --phii-tol at all. --phii-tol is the flag that gates both of the checks the docstring states as invariants.",
    "evidence": "RECONCILER-DERIVED. Lens B (prose only): 'python -m trinity._analysis.check_yesno -f outputs/trinity_fiducial_yesno --tol 1e-4' — '--tol 1e-4 ... value shown only as an example — no default documented, and what it gates is never stated'; B's inventory of documented options lists --tol and nothing else. Lens A (code only): '--tol default 1e-3 (lines 252-255)' and '--phii-tol default 1e-20 (lines 257-261)', the latter gating L200 (Check B, 'noPHII must have P_HII ~ 0 everywhere') and L204 (Check C, 'yesPHII should have P_HII > 0'). CAVEAT: B could not see argparse help= strings (code literals), so B's silence about --phii-tol is a lens boundary; the drift is that the module *docstring* — which B did see in full — omits it.",
    "expected": "The usage block should list every flag that can change a verdict, with its default, and should not show an example value that differs from the default without saying so.",
    "failure_scenario": "Two users run the tool on the same pair with different --tol and get different verdicts; neither can tell from the documentation whether the disagreement is physics or a flag default. A user tuning 'what counts as P_HII ~ 0' never learns the knob exists.",
    "repro": "Compare the usage block in check_yesno.py:3-38 with the add_argument calls at 242-262.",
    "confidence": "medium"
  },
  {
    "id": "S14-R-15",
    "file": "trinity/_analysis/check_yesno.py",
    "line": 128,
    "class": "units",
    "severity": "S4",
    "claim": "No unit is stated in any docstring or comment in the slice, though Pb, P_HII, P_ram, R2 and t are compared, divided and thresholded throughout. One unit statement exists in the file, in an argparse help string.",
    "evidence": "A-vs-B DISAGREEMENT ABOUT THE SAME FILE, resolved as a lens boundary; B is demoted accordingly. Lens B (prose only): 'not one unit is stated anywhere in the slice ... No units appear in any of the 30 prose entries', citing L3-38, L112 and L128-138. Lens A (code only) quotes the --phii-tol help text: 'Absolute P_HII floor below which values count as zero (default: 1e-20, in TRINITY AU pressure units)'. Resolution: argparse help= strings are code literals and so absent from B's comment/docstring dump — B's claim is correct for docstrings and comments and overstated for the file. A separately pinned the system as Msun/pc/Myr from trinity/_functions/unit_conversions.py:18-21.",
    "expected": "Units named for each loaded field in the loader docstring, given the project's own conventions single out units as a recurring bug class. The one place units do appear is the flag least likely to be read.",
    "failure_scenario": "A field is loaded in a different pressure convention than Pb; every P_HII > Pb comparison and the P_HII/Pb ratio is then off by a fixed factor, silently shifting frac_phii_wins and flipping EXPECTED to UNEXPECTED with no visible symptom. Note this compounds with S14-R-06, whose absolute threshold is only meaningful in one specific unit system.",
    "repro": "Read the docstrings at check_yesno.py:80, 112 and 128-138; then read the --phii-tol help string at 257-261 — the only unit statement in the file.",
    "confidence": "high"
  },
  {
    "id": "S14-R-16",
    "file": "trinity/_analysis/check_yesno.py",
    "line": 198,
    "class": "other",
    "severity": "S4",
    "claim": "The verdict logic carries no docstring — the code that applies the taxonomy is marked only '# Diagnosis' — and three of the file's seven functions (load_run, diagnose_pair, main), together comprising the bulk of it, have no docstring at all. No prose states the precedence when several verdict conditions co-occur.",
    "evidence": "Lens B (prose only): 'L198 comment: \"# Diagnosis\"; L3-38 module docstring carries the entire taxonomy ... The per-pair report block (between L158 and L237) and the CLI block (from L240) carry no docstring at all'; B counts four function docstrings (L62, L80, L112, L128-138). Lens A's call graph supplies the denominator — seven functions: main, pair_yes_no, diagnose_pair, load_run, _get_field, compare_trajectories, pressure_dominance — so the three undocumented ones are load_run, diagnose_pair and main. A also settles B's precedence question: the cascade is bugs-non-empty -> BUG, then trajectories_match and frac == 0 -> EXPECTED, then trajectories_match and frac > 0 -> UNEXPECTED, else DIVERGES. BUG-first is sensible, so B's concern is a documentation gap rather than a defect; demoted to S4 on that basis.",
    "expected": "A docstring on diagnose_pair restating each verdict's computable condition, its threshold, and the precedence order — currently recoverable only by reading the cascade.",
    "failure_scenario": "A maintainer reorders the checks to 'clean up' the cascade and silently changes the tool's conclusions, because nothing states that BUG must be evaluated first.",
    "repro": "Read check_yesno.py:198 and note the absence of any docstring after L138.",
    "confidence": "high"
  },
  {
    "id": "S14-R-17",
    "file": "trinity/_analysis/check_yesno.py",
    "line": 86,
    "class": "other",
    "severity": "S4",
    "claim": "A bare '# noqa: E711' sits inside the loader whose one documented job is None-handling, with no rationale. The suppression is both necessary-looking and inert: the comparison form is required for elementwise object-array semantics, and E711 is not in this project's enforced ruff set.",
    "evidence": "B FLAGGED THE SILENCE, A SUPPLIED THE MISSING RATIONALE — a clean complementary pair. Lens B (prose only): 'L86 comment: \"# noqa: E711\"' — 'a bare lint suppression ... with no rationale comment explaining why the suppressed comparison form is required', inside the function whose docstring is 'Load a field as float array, replacing None with default.' Lens A (code only), verified under numpy 1.26.4: 'np.where(arr == None, default, arr) (line 86). I verified that on a numpy object array arr == None is a genuine elementwise comparison returning a bool array, so this works as written; it would not on a non-object dtype, hence the guard at line 85.' A adds: 'the project's ruff set is F821/F811/F823/E9, so it is not enforced.'",
    "expected": "One line of rationale beside the suppression: that `is None` would not broadcast elementwise over an object array. Optionally drop the noqa, since E711 is not enforced here.",
    "failure_scenario": "Exactly what B predicted: a future cleanup rewrites the comparison into the lint-preferred `is None` form, changing elementwise semantics on the object array and silently altering which entries receive the default — which shifts the two checks the verdict depends on.",
    "repro": "Read check_yesno.py:85-86 in the context of the docstring at 80, and check the ruff rule selection in the project config.",
    "confidence": "high"
  },
  {
    "id": "S14-R-18",
    "file": "trinity/_analysis/check_yesno.py",
    "line": 97,
    "class": "deadcode",
    "severity": "S4",
    "claim": "P_drive is loaded, carried through the re-sort, and returned, but never read anywhere in the module — despite being the one on-disk field that would directly verify the P_drive = max(Pb, P_HII) claim the tool prints as its EXPECTED justification. The 'no' run's Pb, P_ram and P_drive are likewise loaded and unused.",
    "evidence": "Lens A (code only): 'L97 P_drive = _get_field(out, \"P_drive\"); L101-103 include it in the re-sort tuple; L105 return dict(t=t, R2=R2, Pb=Pb, P_HII=P_HII, P_ram=P_ram, P_drive=P_drive) — and no other reference to P_drive exists in the file. Compare L218 P_drive=max(Pb,P_HII)=Pb identically in both runs., which is asserted rather than measured.' Lens B independently establishes why that matters: the module docstring makes the P_drive coupling the entire hypothesis under test ('In the energy / implicit phases TRINITY uses P_drive = max(Pb, P_HII)'), and B rates the supporting citation as the slice's only external reference.",
    "expected": "Either compare the on-disk P_drive against max(Pb, P_HII) directly — which would turn the docstring's central assertion into a measurement and make the line-pinned citation of S14-R-11 unnecessary — or stop loading the field.",
    "failure_scenario": "No incorrect result; wasted I/O and, more to the point, a missed check that would settle S14-R-01 empirically (does the real P_drive include the ram term or not?). Per project rule, flagged only, not proposed for deletion.",
    "repro": "grep P_drive in trinity/_analysis/check_yesno.py: lines 97, 102, 105 only.",
    "confidence": "high"
  },
  {
    "id": "S14-R-19",
    "file": "trinity/_analysis/check_yesno.py",
    "line": 49,
    "class": "state",
    "severity": "S4",
    "claim": "The module mutates global interpreter state at import time, prepending the repo root to sys.path at position 0 — ahead of stdlib and site-packages — even when imported as part of the package rather than run as a script.",
    "evidence": "Lens A only (invisible to a prose lens; no comment explains it, which B's census implicitly confirms — B lists eight substantive inline comments and none is at L48-49). A: 'L48-49 _HERE = Path(__file__).resolve().parent / sys.path.insert(0, str(_HERE.parent.parent)), executed at module scope, before from trinity._output.trinity_reader import ... at L51.' A also reads it as evidence of script-only intent, supporting S14-R-20.",
    "expected": "Guard the bootstrap under the __main__ branch, and append rather than insert at index 0.",
    "failure_scenario": "Any process that imports this module thereafter resolves top-level names against the repo root first. Repo-root directories sharing a name with an importable module (test/, tools/, lib/, docs/, paper/, param/) would shadow it for the rest of the process. Rated hygiene because neither lens could check which of those are importable — see open question 8.",
    "repro": "python -c \"import sys, trinity._analysis.check_yesno as m; print(sys.path[0])\" prints the repo root.",
    "confidence": "medium"
  },
  {
    "id": "S14-R-20",
    "file": "trinity/_analysis/check_yesno.py",
    "line": 296,
    "class": "deadcode",
    "severity": "S4",
    "claim": "The module has no caller visible to either lens and no documented programmatic consumer: its sole entry point is the __main__ guard, the package __init__ re-exports nothing, all configuration arrives via argparse, and the only documented invocation is a human at a shell pointing at a specific past experiment's untracked output directory.",
    "evidence": "CORROBORATED from both halves. Lens A (code only): 'no function in this slice has any in-slice caller other than through main, and main has no caller at all except direct script execution ... the sys.path bootstrap at L48-49 exists to make direct script execution resolve import trinity at L51; all configuration arrives via argparse (L242-262) rather than a callable API.' Lens B (prose only): 'No prose anywhere in the slice names a programmatic caller, an importing module, or a test. __init__.py says nothing, so there is no documented re-export. The prose's own framing (\"Hypothesis under test\", a hard-coded example output path) reads as a one-off investigation tool that was committed into the package tree.' Both lenses explicitly decline to call it dead, because the caller set lives in files neither may read.",
    "expected": "Either a stated in-package consumer or a test that keeps the helper honest, or an explicit note that it is a standalone one-off diagnostic retained for reference. Per project rule, no deletion is proposed.",
    "failure_scenario": "The tool silently rots against changes to the run-output format (see S14-R-09) or to the P_drive coupling it asserts (S14-R-11); nothing imports it and no test exercises it, so the next investigator trusts a stale verdict. NOTE: severity across this whole slice is bimodal on this question — if a CI job or wrapper does call it, S14-R-04 (always exits 0) and S14-R-02 (confident false verdict) are floors rather than ceilings.",
    "repro": "grep -rn 'check_yesno' run.py tools/ test/ paper/ docs/ param/ .github/ .pre-commit-config.yaml pyproject.toml",
    "confidence": "medium"
  },
  {
    "id": "S14-R-21",
    "file": "trinity/_analysis/__init__.py",
    "line": 1,
    "class": "other",
    "severity": "S4",
    "claim": "trinity/_analysis/__init__.py is empty of both code and prose: no imports, no __all__, no definitions, and no docstring. The sub-package makes no claim about what it is, what it exports, or what belongs in it.",
    "evidence": "CORROBORATED from both sides, which for a file this small is as complete as this method gets. Lens A (code only): 'trinity/_analysis/__init__.py is a single blank line after stripping — no imports, no __all__, no definitions ... It exists only to make trinity._analysis a regular package.' Lens B (prose only): 'trinity/_analysis/__init__.py contributes zero prose entries. It does not even appear as a section in the dump. The package carrying the name _analysis makes no claim at all about what it is, what it exports, or what belongs in it.'",
    "expected": "A one-line module docstring stating what _analysis is for and who its intended consumers are, given it is a private sub-package inside the shipped package.",
    "failure_scenario": "A contributor adding a new analysis helper has no stated convention for what qualifies, and a reader cannot tell whether _analysis is a supported surface or a scratch drawer inside the shipped package.",
    "repro": "Open trinity/_analysis/__init__.py.",
    "confidence": "high"
  },
  {
    "id": "S14-R-22",
    "file": "trinity/_analysis/check_yesno.py",
    "line": 3,
    "class": "regime",
    "severity": "S4",
    "claim": "The regime in which the EXPECTED verdict is the correct null hypothesis is described only qualitatively — 'a fiducial massive cluster' where bubble pressure dominates 'by orders of magnitude' — with no quantification and no .param file named. The only concrete artefact named anywhere is an output directory under untracked outputs/.",
    "evidence": "Lens B only; Lens A is docstring-blind and structurally could not corroborate this. B: 'For a fiducial massive cluster the mechanical bubble pressure dominates by orders of magnitude, so the yes/no runs integrate the same effective ODE and R2(t) is identical.' and 'python -m trinity._analysis.check_yesno -f outputs/trinity_fiducial_yesno'. B: 'Neither is quantified, and no .param file is named anywhere in the slice's prose.' A's independent contribution is consistent: the tool's thresholds are absolute constants with no regime scaling (S14-R-06), so nothing in the code adapts the null hypothesis to the configuration either.",
    "expected": "Name the tracked .param that defines 'fiducial' and quantify the dominance margin, so a reader can tell whether a new run falls inside the regime where EXPECTED is the right null hypothesis.",
    "failure_scenario": "The tool is run on a low-mass or high-density cloud where the ordering of Pb and P_HII genuinely differs; the operator reads EXPECTED or UNEXPECTED against a null hypothesis that was never valid for that configuration. Compounding: outputs/ is untracked scratch, so the documented invocation is not reproducible from a clean checkout.",
    "repro": "Read check_yesno.py:3-38; no .param file is named. Then grep param/ for a yes/no sweep and for include_PHII.",
    "confidence": "medium"
  },
  {
    "id": "S14-R-23",
    "file": "trinity/_analysis/check_yesno.py",
    "line": 175,
    "class": "other",
    "severity": "S4",
    "claim": "The four labelled checks appear in source order B, C, A, D.",
    "evidence": "Lens B only, and Lens A was STRUCTURALLY BLIND to it — the labels exist only in comments, so A's silence is not a contradiction. B: 'L175 \"# Check B: noPHII must have P_HII ~ 0 everywhere\"; L177 \"# Check C: yesPHII should have P_HII > 0 at some point\"; L180 \"# Check A: trajectory identity\"; L184 \"# Check D: does Pb dominate in the yesPHII run?\"'. A's independent line accounting is consistent with the ordering (A places the P_HII gates at L176/L178, before the trajectory comparison at L181 and pressure_dominance at L185), which corroborates the source order without seeing the labels.",
    "expected": "Labels ordered as they execute, or a note explaining the discrepancy.",
    "failure_scenario": "Cosmetic. B's own reading is the useful part: 'it is the kind of drift that indicates the labels were assigned before the code was reordered' — which is a weak hint that the check ordering changed at some point without the comments following.",
    "repro": "Read check_yesno.py:175-184.",
    "confidence": "low"
  },
  {
    "id": "S14-R-24",
    "file": "trinity/_analysis/check_yesno.py",
    "line": 246,
    "class": "other",
    "severity": "S4",
    "claim": "The module docstring is passed to argparse as the epilog, so it is user-facing CLI help — and it is also the sole home of the verdict taxonomy. Under python -OO the docstring becomes None, argparse accepts that silently, and the only documentation of what EXPECTED/BUG/UNEXPECTED/DIVERGES mean disappears from --help.",
    "evidence": "RECONCILER-DERIVED; neither lens could reach this alone. Lens A (code only): 'Lines 1-39 are the stripped module docstring. That docstring is functional, not decorative: it is passed as the argparse epilog at check_yesno.py:246, so its text is user-facing CLI help. Under python -OO it becomes None, which argparse accepts silently, so -OO quietly shrinks --help.' A explicitly could not read its content. Lens B (prose only) supplies the content and its uniqueness: 'The taxonomy is stated only here, in the module docstring. The comment that marks the code producing it says merely \"# Diagnosis\" (L198) — no function docstring restates the criteria at the point of decision.'",
    "expected": "Documentation that a user needs at the point of reading a verdict should not live only in a string that -OO strips; restate the taxonomy in the diagnosis routine (see S14-R-16) or in a module-level constant.",
    "failure_scenario": "A user running under -OO (or any wrapper that does) gets a --help with no epilog and no explanation of the six statuses the tool prints. Low likelihood, zero-cost fix, recorded because it is a pure product of combining the two lenses.",
    "repro": "python -OO -m trinity._analysis.check_yesno --help and compare with the un-optimised output.",
    "confidence": "medium"
  }
]
```
