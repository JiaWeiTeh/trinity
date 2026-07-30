# PROVENANCE — the freshness rule, and what you may quote

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

**Status (2026-07-30):** 🔵 in force — the fresh artifacts now exist (294 arms, stamped 2026-07-30).
§7's "if a fresh number contradicts a VERIFY number" clause **fired**: G0 failed 2/11, and the
reconciliation is `FINDINGS.md` §1.

---

## 1. The rule

> **CUTOFF = 2026-07-29.** A number is quotable in this workstream only if it comes from an artifact
> whose own first-line provenance stamp reads `# generated <ISO8601> …` with a date **on or after the
> cutoff**. Anything earlier is **VERIFY**: possibly true, not citable until re-measured.

That is the whole rule. It replaces a five-week register of per-artifact grades with one date comparison,
which is the point — a rule nobody can apply is not a rule.

`python docs/dev/transition/pdv-trigger/data/make_freshness_audit.py` applies it mechanically to every
committed CSV under the parent tree and writes `freshness_audit.csv`. Run it before believing anything.

## 2. Why a date, when the parent had a whole register

The parent workstream's `CONTAMINATION.md` grades artifacts individually (⛔ / ⚠️ / ✅) against rules
(a)–(e). It is a good document and it *worked* — it is how the E3/E4 contamination was contained. But it
requires a reader to hold five rules and ~50 rows in their head, and its failure mode is silent: a number
whose grade you did not check reads exactly like a number whose grade you did.

The events that produced this workstream are the argument for a coarser, louder rule:

- **`§17`/`§18` (2026-07-27/28).** The bench6 Θ_cum numerator integrated the raw `Lcool` column, which
  silently drops the boost under `cooling_boost_mode='multiplier'`. The published conclusion — *"f_mix
  eliminated as a calibration knob"* — was a **metric artifact**. When it was fixed, the f_mix dose–response
  went from wrong-sign to monotone, and the head-to-head **inverted**. That conclusion had been quoted in
  four documents for eight days.
- **`§23` (2026-07-29).** *"f_κ pushes evaporation the wrong way vs El-Badry"* — used as a physics
  discriminator to retire f_κ — was **false**. El-Badry Eq 47 carries `(C/6e-7)^{2/7}`: mass loss *rises*
  with conduction, and TRINITY reproduces that to 0.34–1.63%. Five sites, including a shipped `registry.py`
  info string, had to be corrected.
- **`§24` (2026-07-29).** `§12`'s stated *cause* for "no whole-band f_κ" (insufficient reach) was wrong; the
  result (5/6 at f_κ=12) was right. Cause re-attributed to condensation fallout.

None of these were contamination in the register's sense. Every one passed the grades. They were **correct
data with a wrong reading**, and the only thing that caught them was somebody re-deriving from scratch.
A date cutoff does not catch bad readings either — but it makes "when was this measured, and by which code"
the first question rather than the last, and it forces the re-derivation that does catch them.

## 3. The quotability ladder

| tier | what it means | may I quote it? |
|---|---|---|
| ✅ **FRESH** | stamp ≥ CUTOFF, from this campaign, gate-cleared | **yes** — cite the artifact path |
| 🟡 **PENDING** | a pre-registered prediction, no measurement yet | **only as a prediction**, never as a result. `verdict=PENDING` in the CSV |
| ⚠️ **VERIFY** | stamp < CUTOFF (the parent workstream's evidence) | **no** — re-measure, or quote it explicitly as *"pdv-trigger `§N`, 2026-07-DD, unverified"* |
| ⛔ **VOID** | superseded or withdrawn upstream | **never**, in any framing |

**UNSTAMPED is not FRESH.** A file with no `# generated` line falls back to its git *commit* date, which
only upper-bounds its age (bulk commits routinely lag generation by days). Treat unstamped as VERIFY.

## 4. What this workstream inherits vs re-measures

Inheritance is where a "fresh start" quietly leaks old numbers, so it is enumerated.

| input | status | why |
|---|---|---|
| **Lancaster 2021b Table 1** (M_cl, R_cl, n_H, ε_*) | ✅ **inherited, and that is correct** | Published literature, `[V]`-verified 2026-07-12 (`LANCASTER_REFERENCE.md §7b`). A re-run cannot refresh a paper. |
| **The L21b band [0.90, 0.99]** and **λδv ≈ 3** | ✅ inherited | Same — external calibration targets. |
| **El-Badry+2019 Eq 47** and the θ closed form | ✅ inherited | Same. |
| **The `.param` definitions** (5 benches, 6 band configs) | ✅ inherited | Inputs, not measurements; emit-gated and pinned by `test/test_bench7_params.py`. |
| **Θ₀ = 0.462 / 0.341 / 0.221** | ⚠️ **VERIFY → re-measured by `bench5r`** | 2026-07-19. It is also the **G0 target**: the fresh run must reproduce it. |
| **f_A band entry 13.9 / 53.5 / 74.8, spread 5.39×** | ⚠️ **VERIFY → re-measured by `bench5r`+`bench6r`** | 2026-07-19, post-`§18` correction. Also a G0 target. |
| **f_mix band entry 4 / 8.16\* / 11.9\*, spread 2.96×** | ⚠️ **VERIFY → re-measured by `bench6r`+K4** | 2026-07-19. \* = **extrapolated past the grid**, never measured — the flaw K4 exists to close. |
| **theta5k f_κ fire map** (2026-07-03) | ⚠️ **VERIFY → re-measured by K2** | K2's grid was widened to `{1,…,16}` precisely so this stops being an input. |
| **`§24` K0 Q1/Q1b/Q2** (Eq-47 match, back-reaction, squeeze) | ⚠️ VERIFY | Offline re-reads of pre-cutoff CSVs. Regenerates byte-identically, which shows *stability*, not freshness. Q1b is re-measured on full runs in §5.4 of `PLAN.md`. |
| **P1's predicted f_κ entry doses** | 🟡 PENDING | Frozen in `bench7_gate_g0.csv`, `verdict=PENDING`. Derived from a VERIFY Θ₀, so it will be recomputed from the fresh Θ₀ and **both** recorded. |

## 5. The stamping contract

Every generated artifact carries, as its first line:

```
# generated 2026-07-29T18:57:48Z | builder make_bench7_gate_g0.py | code 1e82a37+dirty
```

Written by `docs/dev/transition/pdv-trigger/_stamp.py`. Readers skip leading `#` lines
(`csv.DictReader` does **not** do this for you). As of the cutoff the stamp covers:

- summary CSVs (`<campaign>_summary.csv`) — already did;
- **per-arm trajectory CSVs** — new; these are what the Θ_cum metric actually reads, and they were the
  largest unstamped surface in the tree (262 files);
- **`<campaign>_hashes.csv`** — new;
- the analysis outputs, which additionally carry a **`# SOURCES READ:`** line naming their exact inputs.

Two traps, both live, both worth knowing before you build on this:

1. **Stamping nearly broke the K3 determinism check.** P4 is decided by hashing the reduced trajectory
   CSVs — and two identical runs stamped a second apart hash differently, which would have turned P4 into a
   guaranteed FAIL that looked like physics. The hash is therefore taken over **non-comment lines only**
   (`grep -v '^#' | sha256sum`). Any future header change is safe by construction.
2. **`+dirty` is not a reproducibility signal.** `_stamp.py` reads `git status` from inside the already-open
   output file, so *every* in-place regeneration records `+dirty`, even from a spotless checkout. Likewise
   an artifact always names the commit *before* the one that commits it, so "is it at HEAD" is red for
   correct work. Both were implemented as verdicts, observed firing on all four correctly-built CSVs, and
   removed. The real question — *did the builder change after its output?* — is `MANIFEST.md`'s ⚠️
   STALE-RISK flag.

## 6. Rules carried over from the parent, unchanged

These are the parent's hard-won measurement rules (`pdv-trigger/runs/README.md` 📏, `CLAUDE.md` rule 5).
They are not relaxed here; a fresh start is not a reason to re-learn them.

1. **Every arm runs to `stop_t = 5 Myr`** or its natural physics end — never a wall-clock truncation. No θ
   from a truncated run is quotable.
2. **θ is `theta_max` over the whole run**, from ACCEPTED `dictionary.jsonl` implicit rows. **Blowout-θ is
   retired** (it under-read diffuse θ by ~2×). Call-level observers are banned.
3. **Separate processes per run** — trinity leaks module-level globals in-process — with a unique
   `path2output`.
4. **Per-call equivalence is necessary but not sufficient** for anything on an iterative path. A full-run
   check at matched `t`, in separate processes, on the stiffest regime, is what clears it. `§24` Q1b is the
   worked example: an exact fixed-state match decayed to −11.3% once the state evolved.
5. **A prediction that misses is recorded as a miss**, not re-negotiated (the SC-0 pattern, `§15k`).
6. **The reduce is ONE-SHOT.** gpfs workspaces are cleaned and raw arms do not come back. Every column the
   analysis could need must be declared *before* the first reduce.

## 7. If a fresh number contradicts a VERIFY number

That is a **result**, not a problem, and it is handled by gate **G0**: the fresh baselines are checked
against the pre-registered 2026-07-19 targets, at the same tolerances, with no relaxation in either
direction.

- **G0 PASS** → the older result reproduced. The VERIFY tier for those specific quantities can be lifted,
  citing both artifacts.
- **G0 FAIL** → the older result did not reproduce. Record it in this workstream's findings with both
  numbers and the diff, **before** reading anything downstream of it. Do not quietly adopt either value.

Never silently merge a fresh and a pre-cutoff measurement into one fit or one table. If both are shown,
they are shown as two rows with two dates.
