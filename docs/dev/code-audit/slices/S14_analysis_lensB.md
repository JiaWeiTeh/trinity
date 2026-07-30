# S14 analysis helpers — Lens B (what the code claims)

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

This is a **prose-only transcription**. My entire input was the extracted comment/docstring dump for
the slice; I have **seen no code, no signatures, no tests, no `docs/dev/`, and no other lens's
report**. Slice files:

- `trinity/_analysis/check_yesno.py`
- `trinity/_analysis/__init__.py`

Every statement below is a **claim made by the prose**, never an assertion about behaviour. Where I
write "the prose claims X", I have no way to know whether the code does X — that is precisely the
point of the blind split. Citations are `source-file:line`, taken from the line tags in the prose
dump (which quotes source line ranges, e.g. `L128-138`).

---

## 1. Prose census — the finding is the thinness

The slice's whole documentary surface is **30 prose entries**, all from one file. Breakdown:

| Kind | Count | Lines |
|---|---|---|
| Module docstring | 1 | `check_yesno.py:3` (spans L3–38) |
| Function docstrings | 4 | L62, L80, L112, L128–138 |
| Substantive inline comments | 8 | L86, L175, L177, L180, L184, L187, L198, L291–292 |
| ASCII section-divider banners | 15 | L58–60, L76–78, L108–110, L156–158, L238–240 |
| File-header boilerplate (shebang, coding) | 2 | L1, L2 |

Two structural observations follow directly from the census:

1. **`trinity/_analysis/__init__.py` contributes zero prose entries.** It does not even appear as a
   section in the dump. The package carrying the name `_analysis` makes **no claim at all** about
   what it is, what it exports, or what belongs in it.
2. **Half the surviving prose (15/30) is decorative rule-lines.** The five banner blocks name the
   file's sections — `# Pairing` (L59), `# Loading` (L77), `# Diagnostic checks` (L109),
   `# Per-pair report` (L157), `# CLI` (L239) — and nothing more. The real documentary content is
   one module docstring plus four one-to-eleven-line function docstrings.
3. The last comment sits at **L291–292**, so the file is ≥292 lines. Docstring coverage stops at
   L128. Everything after L138 — the per-pair report block (~L158–237) and the CLI block
   (~L240–292), together the bulk of the file — carries **no docstring**, only the eight inline
   comments.

---

## 2. Module-level claims (`check_yesno.py:3`)

### 2.1 Stated purpose

> "check_yesno.py — diagnose why ``_yesPHII`` / ``_noPHII`` runs produce identical R2(t)
> trajectories."

The prose claims this is a **diagnostic for an already-observed anomaly** — the trajectories were
found identical, and this script exists to explain that. It is framed as investigation scaffolding
("Hypothesis under test"), not as a general-purpose library helper.

### 2.2 The claimed physics contract it tests

> "In the energy / implicit phases TRINITY uses P_drive = max(Pb, P_HII) (see
> ``trinity/phase1_energy/energy_phase_ODEs.py:253-256``) and in the transition phase
> P_drive = max(Pb, P_HII + P_ram)."

Two distinct claimed drive-pressure laws, keyed to phase:

| Claimed regime | Claimed `P_drive` |
|---|---|
| energy / implicit phases | `max(Pb, P_HII)` |
| transition phase | `max(Pb, P_HII + P_ram)` |

> "Because P_HII enters through a ``max``, toggling ``include_PHII`` only changes the trajectory when
> ``P_HII`` (or ``P_HII + P_ram`` during transition) exceeds ``Pb``."

This is the prose's central **claimed invariant**: the `include_PHII` switch is a no-op on the
trajectory unless the ionised-gas pressure wins the `max`. Note the switch is named
`include_PHII` — the only named configuration key in the slice.

> "For a fiducial massive cluster the mechanical bubble pressure dominates by orders of magnitude,
> so the yes/no runs integrate the same effective ODE and R2(t) is identical."

A **regime claim**: "fiducial massive cluster", "orders of magnitude". Neither is quantified, and no
`.param` file is named anywhere in the slice's prose — the only concrete artefact named is an output
directory (§5).

### 2.3 The claimed verdict taxonomy (four outcomes)

> "EXPECTED — trajectories match AND Pb dominates P_HII throughout
> BUG — noPHII run has non-zero P_HII, or yesPHII run has no P_HII despite include_PHII=True
> UNEXPECTED — trajectories match but P_HII > Pb somewhere (the max()-coupling should have driven
> divergence there)
> DIVERGES — trajectories differ as physically expected"

Transcribed as a decision table of *claimed* criteria:

| Verdict | Claimed trajectory condition | Claimed pressure condition |
|---|---|---|
| `EXPECTED` | match | `Pb` dominates `P_HII` throughout |
| `BUG` | (not constrained) | noPHII has non-zero `P_HII`, **or** yesPHII has no `P_HII` while `include_PHII=True` |
| `UNEXPECTED` | match | `P_HII > Pb` somewhere |
| `DIVERGES` | differ | (not constrained) |

The taxonomy is stated **only here**, in the module docstring. The comment that marks the code
producing it says merely `# Diagnosis` (L198) — no function docstring restates the criteria at the
point of decision.

---

## 3. Documented function contracts

Only four functions carry docstrings. Transcribed verbatim, with what each does and does **not**
promise.

### 3.1 `pair_yes_no` — `check_yesno.py:62`

> "Return list of (base_name, yes_path, no_path); missing partner → None."

- **Claimed return**: a list of 3-tuples `(base_name, yes_path, no_path)`.
- **Claimed error/degenerate behaviour**: an absent partner run yields `None` (presumably in the
  corresponding tuple slot — the prose does not say whether `None` replaces the path or the tuple).
- **Silent on**: what it takes as input; how a run directory is recognised as "yes" vs "no" (the
  `_yesPHII` / `_noPHII` suffixes appear only in the module docstring, never in this contract); how
  `base_name` is derived; ordering; and what the caller is expected to do with a `None` partner.

### 3.2 `_get_field` — `check_yesno.py:80`

> "Load a field as float array, replacing None with default."

- **Claimed return**: a float array.
- **Claimed transformation**: `None` entries are replaced by a `default`.
- **Silent on**: what the `default` is (or whether it has one); the source/format of the "field";
  units; behaviour when the field is entirely absent versus present-but-`None`; whether the
  replacement is elementwise.
- Adjacent, at `check_yesno.py:86`, is a bare `# noqa: E711` with **no accompanying rationale**.
  This is the only prose inside the loader body.

### 3.3 `compare_trajectories` — `check_yesno.py:112`

> "Interpolate R2 onto the overlapping time window, return max rel diff."

- **Claimed operation**: interpolation of `R2` onto the **overlapping** time window of the two runs.
- **Claimed return**: the maximum **relative** difference.
- **Silent on**: which run is interpolated onto which grid; how many sample points; what the
  denominator of "rel" is; units of `R2` and `t`; what happens when the two runs have **no**
  overlapping window; whether the return is signed or absolute.

### 3.4 `pressure_dominance` — `check_yesno.py:128`

The slice's only multi-line function docstring, and its densest contract:

> "In the yesPHII run, when does P_HII actually matter?
> Returns
> -------
> frac_phii_wins : float
>     Fraction of snapshots where P_HII > Pb (or P_HII+P_ram > Pb during the transition phase —
>     we use the more permissive P_HII+P_ram comparison everywhere, which is a superset).
> max_ratio : float
>     max over snapshots of P_HII / Pb."

- **Claimed scope**: the yesPHII run only.
- **Claimed return 1** — `frac_phii_wins : float`: a fraction over "snapshots". Its criterion is
  stated **twice and differently** (see §7.1).
- **Claimed return 2** — `max_ratio : float`: `max(P_HII / Pb)` over snapshots — note **no
  `P_ram`**.
- **Claimed simplification, explicitly flagged**: "we use the more permissive P_HII+P_ram comparison
  everywhere, which is a superset". The prose asserts the superset property; it does not document the
  consequence (over-counting outside the transition phase).
- **Silent on**: what a "snapshot" is; the denominator when snapshot count is zero; the precondition
  `Pb != 0` for the ratio; units of any pressure.

---

## 4. Documented thresholds, criteria, and magic values

Everything the prose offers as a numeric or comparative criterion, and the regime it is claimed for:

| Claimed criterion | Location | Regime it is claimed to apply to | Quantified? |
|---|---|---|---|
| `P_HII > Pb` | L3 (`UNEXPECTED`), L128 (`frac_phii_wins` headline) | "somewhere" / per-snapshot | yes (strict `>`) |
| `P_HII + P_ram > Pb` | L3 (transition phase), L128 (applied **everywhere**) | transition phase by physics; all phases by the metric | yes (strict `>`) |
| `--tol 1e-4` | L3 (usage example) | trajectory comparison, presumably | value shown **only as an example** — no default documented, and what it gates is never stated |
| "P_HII ≈ 0 everywhere" | L175 (`# Check B: noPHII must have P_HII ≈ 0 everywhere`) | noPHII runs | **no** — "≈" is never given a tolerance |
| "P_HII > 0 at some point" | L177 (`# Check C: yesPHII should have P_HII > 0 at some point`) | yesPHII runs | strict `> 0`, no tolerance |
| "trajectory identity" | L180 (`# Check A: trajectory identity`) | run pairs | **no** — no criterion given at the check site |
| "does Pb dominate" | L184 (`# Check D: does Pb dominate in the yesPHII run?`) | yesPHII runs | **no** — "dominate" is never thresholded |
| "dominates by orders of magnitude" | L3 | fiducial massive cluster | **no** |

The four checks are labelled **A–D but appear in source order B (L175), C (L177), A (L180),
D (L184)**.

Note also a wording tension: the module docstring says R2(t) "is identical" and the verdicts speak of
trajectories that "match", while the machinery is a **tolerance** comparison (`--tol`, "max rel
diff"). "Identical" is never reconciled with a tolerance.

---

## 5. Claims about who calls this and why

This is the slice's most complete documentation, and it names **only a human at a shell**:

> "Usage
> -----
> python -m trinity._analysis.check_yesno -f outputs/trinity_fiducial_yesno
> python -m trinity._analysis.check_yesno -f outputs/trinity_fiducial_yesno --tol 1e-4"
> — `check_yesno.py:3`

- **Claimed invocation**: `python -m trinity._analysis.check_yesno`, i.e. module-as-script.
- **Claimed input**: `-f <directory>`, exemplified twice as `outputs/trinity_fiducial_yesno` — a
  specific, named output directory from a specific past experiment.
- **Claimed option**: `--tol`.
- **Claimed consumer role**: "This script verifies that hypothesis per yes/no pair and prints a
  diagnosis" (L3) — the output is a **printed report** for a reader.
- A second, implied consumer is anything reading the **exit status**:
  > "# Non-zero exit only on BUG (data inconsistency). EXPECTED is a
  > # physics conclusion, not a failure." — `check_yesno.py:291`

  This is the only prose suggesting automated (CI-like) consumption.

**No prose anywhere in the slice names a programmatic caller, an importing module, or a test.**
`__init__.py` says nothing, so there is no documented re-export. The prose's own framing
("Hypothesis under test", a hard-coded example output path) reads as a one-off investigation tool
that was committed into the package tree.

---

## 6. Invariants, preconditions, and modal statements

Every "must"/"should"/"only"/"always"-class statement in the slice:

| Statement | Location | Modality |
|---|---|---|
| "noPHII **must** have P_HII ≈ 0 everywhere" | L175 | hard invariant, untoleranced |
| "yesPHII **should** have P_HII > 0 at some point" | L177 | soft expectation |
| "toggling `include_PHII` **only** changes the trajectory when P_HII … exceeds Pb" | L3 | claimed causal invariant |
| "the max()-coupling **should** have driven divergence there" | L3 | claimed implication, defines `UNEXPECTED` |
| "we use the more permissive P_HII+P_ram comparison **everywhere**" | L128 | stated implementation choice |
| "which **is a superset**" | L128 | asserted set relation, unproven in prose |
| "Non-zero exit **only** on BUG" | L291 | exit-status contract |
| "EXPECTED is a physics conclusion, **not** a failure" | L291 | exit-status rationale |
| "missing partner → None" | L62 | degenerate-input postcondition |

There are **no documented preconditions** — nothing states that `Pb > 0`, that snapshot arrays are
non-empty, that the two runs share a time window, or that the `-f` directory contains matched pairs.

---

## 7. Internal contradictions within the prose

### 7.1 `frac_phii_wins` is defined twice, incompatibly, in one docstring

> "frac_phii_wins : float
>     Fraction of snapshots where **P_HII > Pb** (or P_HII+P_ram > Pb during the transition phase —
>     we use the more permissive **P_HII+P_ram** comparison **everywhere**, which is a superset)."
> — `check_yesno.py:128`

The headline sentence and the parenthetical state different criteria for the same returned number.
The parenthetical wins by its own wording ("everywhere"), which means the value's name and its
first-stated definition are both wrong for any snapshot where `P_ram` is non-negligible.

### 7.2 The two returned metrics are not computed from the same quantity

`frac_phii_wins` is claimed to compare **`P_HII + P_ram`** against `Pb`; `max_ratio` is claimed to be
`max(P_HII / Pb)` — **without `P_ram`**. By the docstring's own definitions a caller can be handed
`frac_phii_wins > 0` together with `max_ratio < 1`, i.e. "P_HII wins in some snapshots" alongside
"P_HII never reached Pb". Nothing in the prose warns of this or tells the reader which to trust.

### 7.3 The `UNEXPECTED` verdict's criterion does not match the metric that feeds it

`UNEXPECTED` is defined by **`P_HII > Pb` somewhere**, justified by "the max()-coupling should have
driven divergence there" (L3). But the energy/implicit-phase coupling is claimed to be
`max(Pb, P_HII)` — `P_ram` does **not** enter it there. The metric available to the diagnosis
(`frac_phii_wins`) is claimed to use `P_HII + P_ram > Pb` **in all phases**. So, on the prose's own
terms, the diagnosis can be driven to `UNEXPECTED` by snapshots where only `P_HII + P_ram` exceeds
`Pb` in a phase whose drive law ignores `P_ram` — exactly the case where the `max()`-coupling would
**not** have driven divergence. The "superset" remark acknowledges the widening but not this
consequence.

### 7.4 "Identical" versus a tolerance

The stated observation is that runs "produce **identical** R2(t) trajectories" and verdicts turn on
trajectories that "match", yet the comparison is "max rel diff" against a `--tol` whose default is
never documented (§4).

---

## 8. Citations and external references

Exactly one, and it is line-pinned:

> "(see ``trinity/phase1_energy/energy_phase_ODEs.py:253-256``)" — `check_yesno.py:3`

The entire hypothesis under test rests on this reference. It cites a **line range in another
module**, which cannot be validated from prose and is the most drift-prone form of cross-reference.
There are no literature citations, no equation references, and no links to a `docs/dev/` write-up.

---

## 9. Undocumented surface (what a caller would have to know and cannot learn here)

Ordered roughly by how load-bearing the silence is.

1. **Units — total silence.** `Pb`, `P_HII`, `P_ram`, `R2`, `t`, and "snapshots" appear throughout;
   **not one unit is stated** anywhere in the slice. `max_ratio` and "max rel diff" are implicitly
   dimensionless, but even that is not said. Comparisons like `P_HII > Pb` are only meaningful if
   the two arrive in matched units, and no prose asserts that they do.
2. **What `--tol` gates, and its default.** `1e-4` appears once, in a usage example. Nothing says
   whether it applies to Check A only, to "P_HII ≈ 0" as well, or what the value is when omitted.
3. **The `≈ 0` and "dominate" thresholds** (Checks B and D) are the two verdict-critical criteria and
   neither is quantified.
4. **The verdict-selection logic has no docstring.** The four-way taxonomy lives only in the module
   docstring; the code that applies it is marked `# Diagnosis` (L198) and nothing maps a verdict to a
   computed quantity, a tolerance, or a precedence order when several conditions hold at once
   (e.g. `BUG` conditions co-occurring with `DIVERGES`).
5. **Empty and degenerate inputs.** No prose covers: zero pairs found; a `None` partner reaching the
   comparison; zero snapshots (`frac_phii_wins` denominator); `Pb == 0` (the `max_ratio` divisor);
   two runs with no overlapping time window (`compare_trajectories`).
6. **`_get_field`'s `default`.** Its value is unstated, yet the prose's own Checks B and C are pure
   threshold tests on the fields it produces — a `None`→`0.0` substitution and a genuinely-zero field
   would be indistinguishable to Check C's "yesPHII should have P_HII > 0", which the taxonomy maps
   to `BUG`.
7. **Run identification.** `_yesPHII` / `_noPHII` are named once (L3) as the thing being diagnosed;
   `pair_yes_no`'s contract (L62) never states the naming convention it pairs on.
8. **Exit statuses other than BUG.** The comment asserts "Non-zero exit only on BUG"; `UNEXPECTED` —
   which the module docstring describes as a state that contradicts the coupling model — therefore
   exits zero, silently, by the documented contract. Nothing says this is intended for `UNEXPECTED`
   specifically; the rationale given covers only `EXPECTED`.
9. **The `# noqa: E711` at L86** suppresses a lint check inside the very function whose one claimed
   job is `None`-handling, with no comment explaining why the suppressed comparison form is
   necessary.
10. **`trinity/_analysis/__init__.py`** — no module docstring, no statement of the sub-package's
    purpose or public surface.

---

## 10. Summary judgement (prose-only)

The slice documents **one script** with an unusually good *narrative* header — it states a
hypothesis, cites the code that motivates it, enumerates four named verdicts, and gives runnable
invocations. That is more context than most helpers preserve, and §5 is genuinely the surviving
record of intent.

Against that, the *contract* layer is thin and internally inconsistent: the one substantive Returns
block contradicts itself (§7.1), its two returned metrics are computed from different physical
quantities (§7.2), the criterion that fires the most interesting verdict does not match the metric
that feeds it (§7.3), three of the four checks have unquantified thresholds, no unit is stated
anywhere in a codebase whose own conventions call units "a recurring bug class", and the
sub-package's `__init__.py` is documentarily empty.

---

```json
[
  {
    "id": "S14-B-01",
    "file": "trinity/_analysis/check_yesno.py",
    "line": 128,
    "class": "regime",
    "severity": "S3",
    "claim": "The docstring for pressure_dominance defines its first return value, frac_phii_wins, twice with two different criteria in the same sentence: the headline says the fraction counts snapshots where P_HII > Pb, the parenthetical says the more permissive P_HII+P_ram comparison is used everywhere. The returned number cannot satisfy both, and the variable's name matches the definition the parenthetical overrides.",
    "evidence": "L128-138 docstring pressure_dominance: \"In the yesPHII run, when does P_HII actually matter? Returns ------- frac_phii_wins : float Fraction of snapshots where P_HII > Pb (or P_HII+P_ram > Pb during the transition phase — we use the more permissive P_HII+P_ram comparison everywhere, which is a superset).\"",
    "expected": "One stated criterion for one returned value, with the phase-dependence handled explicitly (either compare per-phase against the phase's own P_drive law, or rename the value to reflect the permissive criterion and drop the P_HII > Pb headline).",
    "failure_scenario": "A reader who stops at the headline sentence interprets frac_phii_wins as a pure P_HII-vs-Pb statistic and concludes the ionised-gas pressure wins the max() in some fraction of snapshots, when by the parenthetical the fraction may be driven entirely by ram pressure.",
    "repro": "Read trinity/_analysis/check_yesno.py:128-138; the two criteria are three lines apart.",
    "confidence": "high"
  },
  {
    "id": "S14-B-02",
    "file": "trinity/_analysis/check_yesno.py",
    "line": 128,
    "class": "other",
    "severity": "S3",
    "claim": "The two documented return values of pressure_dominance are claimed to be built from different physical quantities: frac_phii_wins from (P_HII + P_ram) vs Pb, max_ratio from P_HII / Pb with no P_ram. As documented, the pair can be mutually contradictory.",
    "evidence": "L128-138: \"frac_phii_wins : float Fraction of snapshots where P_HII > Pb (or P_HII+P_ram > Pb during the transition phase — we use the more permissive P_HII+P_ram comparison everywhere, which is a superset). max_ratio : float max over snapshots of P_HII / Pb.\"",
    "expected": "Both metrics computed from, and documented against, the same comparison quantity — or an explicit note that max_ratio deliberately excludes P_ram and why.",
    "failure_scenario": "The printed report shows frac_phii_wins > 0 together with max_ratio < 1 — 'P_HII wins somewhere' next to 'P_HII never reached Pb'. A reader cannot tell which metric governs the verdict and may declare a bug in the pressure fields that does not exist.",
    "repro": "Read trinity/_analysis/check_yesno.py:128-138 and compare the two Returns entries.",
    "confidence": "medium"
  },
  {
    "id": "S14-B-03",
    "file": "trinity/_analysis/check_yesno.py",
    "line": 3,
    "class": "regime",
    "severity": "S3",
    "claim": "The UNEXPECTED verdict is defined on the strict criterion P_HII > Pb and justified by the energy-phase coupling max(Pb, P_HII), but the metric documented to supply that evidence applies the transition-phase superset P_HII+P_ram > Pb in all phases. On the prose's own terms the verdict can fire in phases whose drive law is documented to ignore P_ram.",
    "evidence": "L3-38 module docstring: \"In the energy / implicit phases TRINITY uses P_drive = max(Pb, P_HII) ... and in the transition phase P_drive = max(Pb, P_HII + P_ram).\" and \"UNEXPECTED — trajectories match but P_HII > Pb somewhere (the max()-coupling should have driven divergence there)\"; L128-138: \"we use the more permissive P_HII+P_ram comparison everywhere, which is a superset\".",
    "expected": "The dominance criterion applied per phase, matching the P_drive law the module docstring attributes to that phase; or the UNEXPECTED verdict text restated in terms of the superset criterion actually used, with the resulting false-positive rate acknowledged.",
    "failure_scenario": "A fiducial run in which ram pressure alone pushes P_HII+P_ram above Pb during the energy phase is diagnosed UNEXPECTED, sending an investigator after a non-existent max()-coupling failure — the exact anomaly-chasing this script was written to prevent.",
    "repro": "Read trinity/_analysis/check_yesno.py:3-38 (verdict definitions and phase-dependent P_drive) against 128-138 (metric definition).",
    "confidence": "medium"
  },
  {
    "id": "S14-B-04",
    "file": "trinity/_analysis/check_yesno.py",
    "line": 291,
    "class": "silent-failure",
    "severity": "S3",
    "claim": "The exit-status contract is documented as non-zero only for BUG, with a rationale that covers EXPECTED only. UNEXPECTED — described in the module docstring as a state where divergence should have occurred but did not — therefore exits zero by the documented contract, with no prose acknowledging that choice.",
    "evidence": "L291-292 comment: \"# Non-zero exit only on BUG (data inconsistency). EXPECTED is a\" / \"# physics conclusion, not a failure.\"; L3-38: \"UNEXPECTED — trajectories match but P_HII > Pb somewhere (the max()-coupling should have driven divergence there)\".",
    "expected": "The exit-status mapping stated for all four verdicts, with an explicit decision (and reason) for UNEXPECTED and DIVERGES rather than silence.",
    "failure_scenario": "The script is wired into a CI job or a batch sweep over many yes/no pairs; an UNEXPECTED pair — the most scientifically interesting outcome the tool can produce — is reported only in stdout that nobody reads, and the job goes green.",
    "repro": "Read trinity/_analysis/check_yesno.py:291-292 next to the verdict taxonomy at 3-38.",
    "confidence": "medium"
  },
  {
    "id": "S14-B-05",
    "file": "trinity/_analysis/check_yesno.py",
    "line": 175,
    "class": "numerical",
    "severity": "S4",
    "claim": "Three of the four labelled checks turn on criteria the prose never quantifies: Check B's 'P_HII ~ 0 everywhere', Check D's 'does Pb dominate', and Check A's bare 'trajectory identity'. The only numeric tolerance anywhere in the slice is --tol 1e-4, and it appears solely as a usage example, with no stated default and no statement of which check it gates.",
    "evidence": "L175 \"# Check B: noPHII must have P_HII ≈ 0 everywhere\"; L180 \"# Check A: trajectory identity\"; L184 \"# Check D: does Pb dominate in the yesPHII run?\"; L3-38 usage: \"python -m trinity._analysis.check_yesno -f outputs/trinity_fiducial_yesno --tol 1e-4\". The module docstring also asserts trajectories are \"identical\" while the comparison is \"max rel diff\" against a tolerance (L112).",
    "expected": "Each check's threshold stated with its value and default (what counts as ~0, what ratio counts as dominance, what tolerance defines identity), so a reader can reproduce a verdict from the printed metrics.",
    "failure_scenario": "Two users run the tool on the same output pair with different --tol and get different verdicts; neither can tell from the documentation whether the disagreement is physics or a flag default.",
    "repro": "Read trinity/_analysis/check_yesno.py:175, 180, 184 and the usage block at 3-38.",
    "confidence": "high"
  },
  {
    "id": "S14-B-06",
    "file": "trinity/_analysis/check_yesno.py",
    "line": 3,
    "class": "units",
    "severity": "S4",
    "claim": "No unit is stated anywhere in the slice. Pb, P_HII, P_ram, R2, t and 'snapshots' are all named and compared, and no prose asserts what any of them is measured in or that compared quantities share a unit system.",
    "evidence": "L3-38 \"P_drive = max(Pb, P_HII)\", \"R2(t) is identical\"; L112 \"Interpolate R2 onto the overlapping time window, return max rel diff.\"; L128-138 \"Fraction of snapshots where P_HII > Pb\", \"max over snapshots of P_HII / Pb\". No units appear in any of the 30 prose entries.",
    "expected": "Units named for each loaded field, or an explicit statement that all pressures arrive in the same convention as the module they are read from — the project's own conventions single out units as a recurring bug class.",
    "failure_scenario": "A field is loaded in a different pressure convention than Pb; every comparison P_HII > Pb and the ratio P_HII/Pb is then off by a fixed factor, silently shifting frac_phii_wins and flipping EXPECTED to UNEXPECTED (or masking a real UNEXPECTED) with no visible symptom.",
    "repro": "Scan all prose in trinity/_analysis/check_yesno.py for a unit string; there is none.",
    "confidence": "high"
  },
  {
    "id": "S14-B-07",
    "file": "trinity/_analysis/check_yesno.py",
    "line": 80,
    "class": "numerical",
    "severity": "S4",
    "claim": "_get_field's docstring promises a None-to-default substitution without naming the default, and pressure_dominance documents a P_HII / Pb ratio without any precondition on Pb. Both silences sit directly under threshold tests the verdict taxonomy depends on.",
    "evidence": "L80 docstring _get_field: \"Load a field as float array, replacing None with default.\"; L128-138: \"max_ratio : float max over snapshots of P_HII / Pb.\"; L177 \"# Check C: yesPHII should have P_HII > 0 at some point\"; L175 \"# Check B: noPHII must have P_HII ≈ 0 everywhere\".",
    "expected": "The default value stated (and its interaction with Checks B and C called out), plus a documented precondition or guard for Pb == 0 in the ratio.",
    "failure_scenario": "A yesPHII run whose P_HII field is missing or all-None is substituted to the default; if that default is 0.0 it is indistinguishable from a genuinely absent P_HII and the tool reports BUG for an I/O gap. Separately, a snapshot with Pb == 0 makes max_ratio non-finite with no documented behaviour.",
    "repro": "Read trinity/_analysis/check_yesno.py:80 and 128-138 alongside the checks at 175 and 177.",
    "confidence": "medium"
  },
  {
    "id": "S14-B-08",
    "file": "trinity/_analysis/check_yesno.py",
    "line": 62,
    "class": "silent-failure",
    "severity": "S4",
    "claim": "Degenerate-input behaviour is documented only as 'missing partner -> None' for pair_yes_no; no prose covers what a downstream check does with that None, nor the no-overlap case for compare_trajectories, nor a zero-snapshot denominator for frac_phii_wins.",
    "evidence": "L62 docstring pair_yes_no: \"Return list of (base_name, yes_path, no_path); missing partner → None.\"; L112 docstring compare_trajectories: \"Interpolate R2 onto the overlapping time window, return max rel diff.\"; L128-138: \"Fraction of snapshots where P_HII > Pb\".",
    "expected": "Stated behaviour for the unpaired run (skipped, reported, or fatal), for two runs whose time windows do not overlap, and for an empty snapshot set.",
    "failure_scenario": "A sweep directory contains an odd run with no partner, or two runs that truncate at non-overlapping t; the tool either crashes mid-report or silently drops the pair, and the operator reads a report that is quietly missing rows.",
    "repro": "Read trinity/_analysis/check_yesno.py:62 and 112; neither states the degenerate path beyond the None return.",
    "confidence": "medium"
  },
  {
    "id": "S14-B-09",
    "file": "trinity/_analysis/check_yesno.py",
    "line": 198,
    "class": "other",
    "severity": "S4",
    "claim": "The four-verdict taxonomy exists only in the module docstring; the code that applies it is documented by the single word 'Diagnosis'. No prose maps a verdict to the computed metrics, and no precedence is documented for inputs satisfying more than one verdict's condition.",
    "evidence": "L198 comment: \"# Diagnosis\"; L3-38 module docstring carries the entire taxonomy (\"EXPECTED ... BUG ... UNEXPECTED ... DIVERGES\"). The per-pair report block (between L158 and L237) and the CLI block (from L240) carry no docstring at all — only L175/L177/L180/L184/L187/L198/L291.",
    "expected": "A docstring on the diagnosis routine restating each verdict's computable condition (which metric, which threshold) and the precedence order when conditions co-occur.",
    "failure_scenario": "A pair triggers a BUG condition and also diverges; the printed verdict depends on undocumented ordering, and a maintainer changing the check order silently changes the tool's conclusions.",
    "repro": "Read trinity/_analysis/check_yesno.py:198 and note the absence of any docstring after L138.",
    "confidence": "high"
  },
  {
    "id": "S14-B-10",
    "file": "trinity/_analysis/check_yesno.py",
    "line": 3,
    "class": "citation",
    "severity": "S4",
    "claim": "The module's entire hypothesis rests on a single line-pinned cross-reference into another module, trinity/phase1_energy/energy_phase_ODEs.py:253-256, which cannot be validated from the docstring and drifts with any edit upstream.",
    "evidence": "L3-38: \"In the energy / implicit phases TRINITY uses P_drive = max(Pb, P_HII) (see ``trinity/phase1_energy/energy_phase_ODEs.py:253-256``)\".",
    "expected": "A reference by symbol name (function/variable) rather than by line range, so the pointer survives edits to the ODE module.",
    "failure_scenario": "Lines shift in energy_phase_ODEs.py; a future reader follows the citation to unrelated code, cannot confirm the max(Pb, P_HII) claim, and either distrusts the tool or re-derives the coupling from scratch.",
    "repro": "Read trinity/_analysis/check_yesno.py:3-38; the citation is the only external reference in the slice.",
    "confidence": "high"
  },
  {
    "id": "S14-B-11",
    "file": "trinity/_analysis/__init__.py",
    "line": 1,
    "class": "other",
    "severity": "S4",
    "claim": "The _analysis sub-package contributes zero prose: no module docstring, no statement of purpose, no description of what the package exports or what belongs in it.",
    "evidence": "The prose extract for this slice contains no section for trinity/_analysis/__init__.py at all; all 30 prose entries come from check_yesno.py.",
    "expected": "A one-line module docstring stating what _analysis is for and what its intended consumers are, given it is a private sub-package inside the shipped package.",
    "failure_scenario": "A contributor adding a new analysis helper has no stated convention for what qualifies, and a reader cannot tell whether _analysis is a supported surface or a scratch drawer inside the package.",
    "repro": "Open trinity/_analysis/__init__.py and look for any comment or docstring.",
    "confidence": "high"
  },
  {
    "id": "S14-B-12",
    "file": "trinity/_analysis/check_yesno.py",
    "line": 3,
    "class": "deadcode",
    "severity": "S4",
    "claim": "The prose documents no programmatic caller. The only documented consumer is a human running the module as a script against a specific named output directory from a past experiment, and the module frames itself as testing a hypothesis rather than providing a reusable helper — the profile of one-off investigation scaffolding committed into the package tree.",
    "evidence": "L3-38: \"check_yesno.py — diagnose why ``_yesPHII`` / ``_noPHII`` runs produce identical R2(t) trajectories. Hypothesis under test\" and \"Usage ----- python -m trinity._analysis.check_yesno -f outputs/trinity_fiducial_yesno\". No prose in the slice names an importing module, function, or test; __init__.py documents no re-export.",
    "expected": "Either a stated in-package consumer / test that keeps the helper honest, or an explicit note that it is a standalone one-off diagnostic retained for reference (and, per project convention, output paths under outputs/ are untracked scratch, so the documented invocation is not reproducible from a clean checkout).",
    "failure_scenario": "The tool silently rots against changes to the run-output format or to the P_drive coupling it asserts; nobody notices because nothing imports it and no test exercises it, and the next investigator trusts a stale verdict.",
    "repro": "Read trinity/_analysis/check_yesno.py:3-38; the usage examples are the only documented entry point.",
    "confidence": "medium"
  },
  {
    "id": "S14-B-13",
    "file": "trinity/_analysis/check_yesno.py",
    "line": 86,
    "class": "other",
    "severity": "S4",
    "claim": "A bare lint suppression sits inside the loader whose only documented job is None-handling, with no rationale comment explaining why the suppressed comparison form is required.",
    "evidence": "L86 comment: \"# noqa: E711\"; the enclosing function's docstring at L80 reads \"Load a field as float array, replacing None with default.\"",
    "expected": "A short reason accompanying the suppression (e.g. why an identity comparison to None cannot be used at this site), since the suppressed check guards exactly the None handling the docstring promises.",
    "failure_scenario": "A future cleanup rewrites the comparison into the lint-preferred form, changing elementwise semantics on an object array and silently altering which entries get the default — which in turn shifts Checks B and C.",
    "repro": "Read trinity/_analysis/check_yesno.py:86 in the context of the docstring at 80.",
    "confidence": "medium"
  },
  {
    "id": "S14-B-14",
    "file": "trinity/_analysis/check_yesno.py",
    "line": 3,
    "class": "regime",
    "severity": "S4",
    "claim": "The script's EXPECTED verdict is only meaningful in a regime the prose describes qualitatively — 'a fiducial massive cluster' where bubble pressure dominates 'by orders of magnitude' — with no quantification and no .param file named. The only concrete artefact named is an output directory.",
    "evidence": "L3-38: \"For a fiducial massive cluster the mechanical bubble pressure dominates by orders of magnitude, so the yes/no runs integrate the same effective ODE and R2(t) is identical.\" and \"python -m trinity._analysis.check_yesno -f outputs/trinity_fiducial_yesno\".",
    "expected": "The configuration that defines 'fiducial' named explicitly (a tracked .param file), and the dominance margin quantified, so a reader can tell whether a new run falls inside the regime where EXPECTED is the right null hypothesis.",
    "failure_scenario": "The tool is run on a low-mass or high-density cloud where the ordering of Pb and P_HII genuinely differs; the operator reads EXPECTED/UNEXPECTED against a null hypothesis that was never valid for that regime.",
    "repro": "Read trinity/_analysis/check_yesno.py:3-38; no .param file is named anywhere in the slice.",
    "confidence": "medium"
  },
  {
    "id": "S14-B-15",
    "file": "trinity/_analysis/check_yesno.py",
    "line": 175,
    "class": "other",
    "severity": "S4",
    "claim": "The four checks are labelled A-D but appear in source order B, C, A, D.",
    "evidence": "L175 \"# Check B: noPHII must have P_HII ≈ 0 everywhere\"; L177 \"# Check C: yesPHII should have P_HII > 0 at some point\"; L180 \"# Check A: trajectory identity\"; L184 \"# Check D: does Pb dominate in the yesPHII run?\".",
    "expected": "Labels ordered as they execute, or the out-of-order labelling explained (e.g. the checks are printed in a different order than computed).",
    "failure_scenario": "A reader cross-referencing a printed report line 'Check A failed' against the source scans past it; minor, but it is the kind of drift that indicates the labels were assigned before the code was reordered.",
    "repro": "Read trinity/_analysis/check_yesno.py:175-184.",
    "confidence": "low"
  }
]
```
