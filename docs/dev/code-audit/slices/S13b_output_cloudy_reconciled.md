# S13b output CLOUDY export — reconciled (A vs B)

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

## What this is, and what it cannot be

I read exactly two files:

- `scratchpad/raw/S13b_output_cloudy_lensA.md` — Lens A, working from a comment- and
  docstring-stripped copy of the slice. A saw code and never saw a comment.
- `scratchpad/raw/S13b_output_cloudy_lensB.md` — Lens B, working from `prose.md`, the extracted
  comment/docstring blocks. B saw prose and never saw a line of code.

**I did not read the source.** Not `trinity/`, not `test/`, not `docs/dev/`, not the stripped
`code/` copies, not `prose.md`, not any other slice's report. That is the method: my value is the
diff between two independent accounts, and a third opinion of my own would destroy it.

What that limits: I cannot break a tie by looking. Where A and B contradict each other on a matter
of fact, I say so and demote, rather than guessing which is right. Where both are silent, I have
nothing. Every line number below is one lens's transcription, not my verification — A's numbers
come from a stripped copy whose line offsets may differ from the real file, and B's come from
`prose.md` block starts. Treat them as pointers, not coordinates.

This is an infra-tier slice: **there is no Lens C.** The strongest signal available here is A ≡ B —
two lenses reaching the same conclusion from disjoint inputs — and those are ranked first.

One framing note that colours every severity below. This slice's output is not simulation state; it
is an **export consumed by a third-party code**. A defect here does not crash TRINITY and does not
show up in `dictionary.jsonl`. It produces a CLOUDY deck that runs to completion and yields a
plausible, wrong spectrum. Several S2s below have a true cost well above their rubric rating for
exactly that reason, and I say so in prose rather than inflating the number.

---

## Correspondence table

Axis: **A≡B** corroborated · **A≠B** doc-drift · **A-only** undocumented behaviour · **B-only**
unimplemented/stale claim (annotated "A didn't look" vs "A looked, not there").

### The unit boundary

| # | Claim | Lens(es) | Axis | Verdict |
|---|---|---|---|---|
| 1 | Radius column: `log10(r_pc) + 18.4893505` → `log10(r/cm)`, applied exactly once | A (`dlaw.py:174,176`), B (`dlaw.py:1,173-174`) | **A≡B** | **Correct.** Code does what prose says. |
| 2 | Shell density arrives already `log10`, gets `+(-55.4680516)`, no second `log10` | A (`dlaw.py:175,177`), B (`dlaw.py:64,175`) | **A≡B** | **Correct.** No doubled log, none omitted. |
| 3 | Ambient density arrives linear, caller takes `log10`, dlaw adds the same offset | A (`snapshot_to_deck.py:244`), B (`snapshot_to_deck.py:59,240`) | **A≡B** | **Correct.** Two routes, one offset each, same destination unit. |
| 4 | `3·log10(pc2cm) + log10(ndens_au2cgs) == 0.0` bit-exactly in float64 | A only (measured) | A-only | **Clearance.** No cm/pc skew between the two columns. B could not have seen this. |
| 5 | `LOG_RIN`/`LOG_ROUT` get the same `+18.4893505` | A (`snapshot_to_deck.py:182,198`) | A-only | **Correct in code, undocumented.** Resolves B-01's feared 18.49-dex offset — it does not exist. |
| 6 | `LOG_RIN`/`LOG_ROUT` carry no documented unit or log convention anywhere | B only (A didn't look — A had no comments) | B-only | **Real gap.** S3, see R-14. |
| 7 | `Qi`: `log10(Qi) − log10(Myr2s)`, i.e. ph/Myr → ph/s with log applied | A (arithmetic + magnitude check), B (`snapshot_to_deck.py:181` comment `# ph/Myr -> ph/s`) | **A≡B** | **Correct**, on two independent weak confirmations. Log convention undocumented (B-09). |
| 8 | Age: exactly one `×1e6`; TITLE and filename reuse `age_myr`, not `age_yr` | A only | A-only | **Correct.** No doubled 1e6. B's prose never states the factor. |
| 9 | Density column is written as `n_H` but no composition factor exists | A (grep: none anywhere), B (`dlaw.py:1` says both `log10 n [cm^-3]` and `log10(n_H/cm^-3)`) | **A≡B** | **Open unit risk — R-01, ranked first.** |
| 10 | `ZREL`: zero conversion applied; no normalisation documented | A (`snapshot_to_deck.py:212,274` — straight through), B (`snapshot_to_deck.py:59,200` — no unit, no normalisation claimed) | **A≡B** | **Open unit risk — R-02.** |
| 11 | `+18.4894` / `-55.4681` comments are not exactly 3× each other | B only ("A looked, and the *literals* aren't there") | B-only, **partly contradicted** | Both comments round correctly; the code derives both from `INV_CONV`, not from literals. Maintenance-only — R-17. |
| 12 | pc⁻³ arrays vs cm⁻³ legacy scalars coexist with no documented bridge | B only (A didn't look at naming; A's dataflow shows no mixing path in-slice) | B-only | **Doc hazard only** — R-24, S4. |
| 13 | `run_loader` suffix rule "where they differ" contradicted by its own key list | B only (A didn't look) | B-only | Doc hygiene — R-24. |

### `dlaw` construction

| # | Claim | Lens(es) | Axis | Verdict |
|---|---|---|---|---|
| 14 | `dlaw table radius` / `continue ` / `end of dlaw`, `.6f`/`.4f`, no trailing newline | A (`dlaw.py:33-35,198-202`), B (`dlaw.py:1` format block) | **A≡B** | Emitted grammar matches the documented example exactly. |
| 15 | Grid is the shell's own sample points, no re-gridding; radius not depth | A, B | **A≡B** | Corroborated. |
| 16 | Sort + dedup keeping **last** | A (`dlaw.py:125-131`), B (`dlaw.py:124,152`) | **A≡B** | Mechanism corroborated. |
| 17 | …and it is **silent**: a duplicated radius silently loses its inner density; a non-monotonic input is silently straightened | A only | A-only | R-25, S2. |
| 18 | Bracket check in pc, before the cm conversion, rel_tol 1e-12, no extrapolation | A (`dlaw.py:162-171`), B (`dlaw.py:161,64`) | **A≡B** | Correct and loud on failure. Prose leaves the tolerance unquantified (R-20). |
| 19 | Densification only in "smooth" spans; edges preserved verbatim; all-edge ⇒ warn and return unchanged | A (`dlaw.py:225-236`), B (`dlaw.py:212`) | **A≡B** | Mechanism corroborated. |
| 20 | `edge_threshold=50` "separates PL O(1) from IF O(1e5) with margin" | B (comment `dlaw.py:39`) vs A (measured 7197/4743/2033 on a *smooth* 0.1 %-thick shell) | **A≠B** | **Doc-drift with a measurement — R-03, S2.** |
| 21 | Densification runs after the cm conversion, interpolating in (log r/cm, log n/cm⁻³) | B (pipeline comment order), A (same order) | **A≡B** | Corroborated; physically a no-op given CLOUDY's own log–log interpolation (A). |
| 22 | `dens_profile` accepted but unused | A (never read in body), B (docstring says "Currently unused") | **A≡B** | Honest doc, dead parameter — R-18. |
| 23 | `min_rows` default 10, `edge_threshold` default 50 | A (values), B (only 50 is in prose; "see module-level defaults" points nowhere) | A-only / B-gap | Doc gap folded into R-13. |
| 24 | Float monotonicity guard passes tables whose *printed* radii are equal | A only | A-only | **R-09, S2** — and it breaks B's "rows ordered by increasing radius". |
| 25 | Ambient eps clamp is a defensive safety guard | B (`snapshot_to_deck.py:240` comment) vs A (measured: emits `continue <r>  -363.1207`, passes every guard) | **A≠B** | **R-07, S2** — the guard is the defect. |

### CLI, selection, I/O

| # | Claim | Lens(es) | Axis | Verdict |
|---|---|---|---|---|
| 26 | `--all` writes "one deck per snapshot, plus manifest.json" | B (`trinity_to_cloudy.py:6`) vs A (`--prefix` short-circuits the per-index path; N−1 decks overwritten) | **A≠B** | **Violated documented contract — R-04, S2 (S1 for that flag pair).** |
| 27 | Exactly one picker enforced by extra code, not argparse | A (`:160-171`), B (`:159`) | **A≡B** | Corroborated. |
| 28 | `--age` "picks closest snapshot" | B (no metric, no tie-break stated), A (`mode="closest", quiet=True`, unbounded) | **A≡B** on the gap | **R-11, S2.** |
| 29 | Filename embeds the *requested* age, hiding divergence | B-21 only | B-only, **contradicted** | A: filename uses `age_myr = t_now − tSF` of the *selected* snapshot. Demoted. |
| 30 | `--phase` round-trips through `get_at_time` to recover the original index | B (`:221` comment states the purpose), A (the re-resolved snapshot is then *used*, not just its index) | **A≠B in effect** | **R-12, S2.** |
| 31 | Status gate: refuse outside 0–9; 10–29 / 50–59 / 99 force-overridable; 30–49, 60–98, ≥100 unspecified | B (doc gap) vs A (`isinstance(exit_code, int) and 0 <= exit_code <= 9`, else refuse-unless-force) | **B-only, resolved by A** | Code fails closed. B-06's "permissive fall-through" fear is **contradicted**. Doc incompleteness only — R-15, S3. |
| 32 | `<<<EDIT_ME>>>` survives rendering and must be replaced by hand | A (`PLACEHOLDER_RE` only matches `{{…}}`), B (docstring says exactly this) | **A≡B** | Documented by design. Residual: the stated *reason* (word boundary) is wrong (B-17), and the default deck on disk is unrunnable — R-19, S4. |
| 33 | `key in snap` never terminates; hence the `_MISSING` sentinel | B only (`snapshot_to_deck.py:34`) — A confirms the idiom exists but cannot verify the hazard (type is out of slice) | B-only, corroborating-adjacent | Idiom corroborated; hazard unverified. A adds the hole: a key present as `None` passes — R-16. |
| 34 | Legacy `_parse_simulation_end` strips unit suffixes | A (`:264` `value.split()[0]`), B (`:263` comment says exactly that) | **A≡B** | Correct. |
| 35 | …and `_parse_summary_txt` does **not**, so `"0.02 Zsun"` reaches `float()` | A only (B's transcribed coercion list has no unit step — a corroborating silence) | A-only | **R-08, S2.** |
| 36 | `VALID_DENS_PROFILES` is a hand-maintained mirror of `_input/registry.py` | B (`:35` comment), A (`frozenset({"densBE","densPL"})`, hard reject) | **A≡B** | Merged into R-18. |
| 37 | `main()` returns 0 unconditionally on the batch path | A only | A-only | R-21, S4. |
| 38 | `--abundances` parsed and never read; `extend_with_ambient=False` unreachable | A only (B's CLI transcription never mentions `--abundances`) | A-only | R-22, S4. |
| 39 | `DLAW_ROWS` extracted by stripping first/last line, while open/close are advertised configurable | B (`:256` vs `dlaw.py:64`), A (constants are single-line and never overridden) | B-only, latent | R-23, S4 — A clears it for today. |
| 40 | CLOUDY syntax is "best-guess for C17/C22"; no manual/Hazy citation anywhere | B only (A had no comments to see it in) | B-only | **R-13, S3** — cost far above the rating; see prose. |
| 41 | "Step 0/4/5", "Option B", "Phase 5/6" point at an unnamed document | B only | B-only | R-26, S4. |
| 42 | `_coerce_scalar` type list: docstring omits dicts, comment includes them | B only (A didn't enumerate the branches) | B-only, unresolved | R-27, S4. |
| 43 | The bundled `trinity2cloudy.in_template` is outside the slice | A (§blind spot), B (§2a) | **A≡B** | **Scope boundary, not a finding.** See below. |

---

## Ranked findings

### 1. `n` vs `n_H` — the one open unit risk, reached from both directions (R-01)

This is the finding the method exists to produce. Neither lens could see the other's input, and both
arrived at the same hole.

> **Lens B:** "`dlaw.py:1` prose sentence says the output column is 'log10 n [cm^-3]'; the format
> block three lines later says `{log10(n_H/cm^-3):.4f}`. … No composition, mean-molecular-weight, or
> He correction is claimed anywhere — the conversion is presented as purely geometric (`-55.4681`).
> Either the two descriptions mean the same thing (and the `_H` is loose), or a species conversion is
> missing and undocumented."

> **Lens A:** "The density column undergoes a pure unit shift and nothing else: no
> mean-molecular-weight, hydrogen-mass-fraction or ionisation-state factor is applied anywhere in the
> slice. … `grep -rn 'mu\|m_H\|n_H\|hden' trinity/_output/cloudy/` — no composition factor appears
> anywhere in the slice."

B found the ambiguity in the words. A found the absence in the code. The two together establish, with
as much force as this method can generate, that **the deck's density column is TRINITY's number
density shifted by a pure geometric factor and nothing else**, and that **the documentation does not
say which number density that is.**

Whether this is a defect turns entirely on one fact neither lens could reach, because the producer of
`log_shell_n_arr` is outside the slice:

- If `log_shell_n_arr` is **hydrogen** number density → the deck is correct and the finding collapses
  to a doc fix (delete the ambiguity at `dlaw.py:1`).
- If it is **total particle** density → every deck ever exported overstates n(H) by **~0.35 dex**
  (A's figure, for fully-ionised solar composition, `n_H = n_tot/(1 + y_He + x_e) ≈ 0.45 n_tot`).
  Uniform, so it does not distort the profile *shape*; it shifts the ionisation parameter and every
  line ratio. B's estimate of 0.05–0.15 dex is the H/He-only figure and is the low end; A's is the
  fully-ionised figure. Either way the deck runs and the spectrum looks fine.

**One lookup settles it.** Find the producer of `log_shell_n_arr` (`grep -rn "log_shell_n_arr"
trinity/ --exclude-dir=_output`), and read whether it is built from `ρ/(μ m_H)` (total) or
`ρ X_H/m_H` (hydrogen). Then confirm the consuming side: read
`trinity/_output/cloudy/trinity2cloudy.in_template` and check whether the dlaw table feeds `hden`
(CLOUDY's `dlaw` second column is hydrogen density by definition, so the template cannot rescue a
total-density input).

Severity **S2** as written — the defect is conditional and currently unresolvable in-repo, which is
the definition of latent. If the lookup returns "total particle density" it becomes **S1
retroactively for every deck already published**, and no in-repo test would ever have caught it.

### 2. `ZREL` normalisation — the second open unit risk, also corroborated (R-02)

Less prominent in the brief but structurally identical, and both lenses flagged it independently.

> **Lens A:** "`ZCloud` → `ZREL` | Zsun, linear (**assumed**) | `snapshot_to_deck.py:212,274` |
> transform applied: **none** | linear relative-to-solar (**assumed**)". And: "`ZCloud`: zero
> conversion applied. The value from `bundle.summary["ZCloud"]` is formatted straight through."

> **Lens B:** "ZREL has no documented unit, normalisation, or log convention. The prose says only
> where the value comes from and which override wins. … If ZCloud is an absolute metal mass fraction
> (e.g. 0.014) and the deck's command expects a solar-relative factor, every exported model runs at
> ~1.4 percent of the intended metallicity."

A's word is "assumed" — it flagged its own inference. B's word is "not stated". Same hole, opposite
approach. The error magnitude if wrong (~1.85 dex) is **five times larger** than the n/n_H risk, and
the failure mode is the same: a deck that runs. I rank it second only because the brief's evidence
for it is thinner — but if I were prioritising the lookups, this one and R-01 are the same errand.

**One lookup:** read `ZCloud`'s definition in the run metadata schema (`trinity/_input/`, wherever
`ZCloud` is declared/defaulted) and the metallicity command in
`trinity/_output/cloudy/trinity2cloudy.in_template`. If the template says `metals {{ZREL}}` (CLOUDY's
solar-relative scale factor) and `ZCloud` is an absolute mass fraction, every deck is wrong.

Compounded by **R-06**: whatever `ZREL` means, `%.4f` destroys it below ~10⁻² (A measured
`1e-5 → "0.0000"`, i.e. a *metal-free* CLOUDY model).

### 3. `edge_threshold = 50` — a doc claim contradicted by measurement (R-03)

The comment makes a quantitative regime claim. A, blind to that comment, measured the regime and got
the opposite answer.

> **Lens B (transcribing the comment):** "`# |Δlog n / Δlog r| above this counts as an IF-like
> discontinuity. PL profiles are O(1); transition-phase IFs in TRINITY snapshots are O(1e5). 50
> separates them with margin.`" — and flagged it: "no measurement, dataset, or regime is cited; 'with
> margin' is unquantified."

> **Lens A (measuring):** "for a 0.1 %-thick shell (`r = 5.0 … 5.005 pc`) carrying a 2-dex density
> ramp over 4 points, the three slopes are `7197, 4743, 2033` — all ≫ 50, so **every** interval is an
> 'edge' and densification is skipped entirely."

The comment's claim is not merely uncited; it is **false in the regime the code actually runs in**.
The threshold is applied to `|Δlog n / Δlog r|`, and for a geometrically thin shell `Δlog r` is
~`d/ln10`, so any density contrast of order 1 dex produces slopes of 10³–10⁴ **for a perfectly
smooth profile**. The discriminator does not discriminate: it labels smooth thin-shell spans as
ionisation fronts. Consequence: `--min-rows` is silently unmet, and CLOUDY receives the shell as a
handful of log–log line segments — the *sampling* the user asked for is not delivered, and only a
`UserWarning` says so.

B guessed the failure direction wrong ("a regime sits between 1 and 50 and is silently treated as
smooth"); the real failure is the mirror image (everything is treated as an edge). Note that in
A's account this direction is the *safe* one physically — densification is a near-no-op anyway — so
the cost is a thin table, not a smeared front. **S2.** Fixing the threshold means making it
scale-aware (normalise by shell thickness), not raising 50.

### 4. `--all --prefix` violates a documented contract (R-04)

The only finding where B supplies a stated contract and A shows the code breaking it.

> **Lens B:** "`--all` one deck per snapshot, plus manifest.json" — `trinity_to_cloudy.py:6`.

> **Lens A:** "`if args.prefix is not None: return _UNSAFE_PREFIX_RE.sub('_', args.prefix)` — returns
> before the auto path that appends `pick.index`. … `trinity_to_cloudy -F run --all --prefix
> mymodel` writes `mymodel.in` once per snapshot; only the last survives. `manifest.json` lists every
> snapshot with `"deck": "mymodel.in"`, and `_print_summary` reports 'Converted N snapshots'."

A violated documented contract outranks undocumented sloppiness. The manifest actively lies, and a
downstream batch runner reading it runs the same deck N times believing it has N snapshots. Rubric
says **S2** (needs two flags together); the true cost for anyone who uses that pair is S1, and
nothing in the CLI or the docs warns them off it.

### 5. Precision defects — no documented contract to violate, except once (R-05, R-06, R-09)

The brief asks whether B transcribed any precision or tolerance contract that A's formatting defects
break. The answer is mostly **no, and that is itself the point**:

| A's defect | Documented contract in B's transcription? | Verdict |
|---|---|---|
| `LOG_RIN`/`LOG_ROUT` at `%.4f` vs dlaw radii at `%.6f` | **None.** B-01: these keys have no documented unit, no log base, no format anywhere. | Undocumented sloppiness. There is no contract because there is no documentation. |
| `ZREL` at `%.4f` | **None.** B-10: no unit, no normalisation, no format. | Same. |
| Duplicate printed abscissae | **Weakly, yes.** B §3: "`dlaw table radius` — a *radius* table …, rows ordered by **increasing** radius", and the `.6f` format is documented at `dlaw.py:1`. | The emitted file has rows with **equal** printed radii. The documented ordering claim is broken in the artefact, though not in memory. Ranks above the other two. |
| `rel_tol=1e-12` endpoint check | **Yes, and honoured** — B step 6 states 1e-12, A confirms `math.isclose(..., rel_tol=1e-12)`. | No violation. Prose's "exact-equality drift" phrase is the drift (R-20). |
| dlaw bracket "tiny float tolerance" | Stated but **unquantified**; A reports 1e-12. | Consistent, under-documented. |

The worked numbers, both A's:

- `R2 = 0.5 pc`: true `log_rin = 18.188320549558156`. The deck gets `18.1883` (= 0.4999763 pc); the
  first dlaw row gets `18.188321` (= 0.5000000 pc). **CLOUDY's declared inner radius sits 4.7e-5
  relative below the first tabulated point.** And for a shell thinner than ~2.3e-5 relative,
  `LOG_RIN == LOG_ROUT` as strings — a zero-thickness model.
- `r = [5.0, 5.0(1+1e-6), 5.0(1+2e-6)]` passes `np.all(np.diff(log_r_cm) > 0)` and writes three rows
  all reading `continue 19.188321` with three different densities.
- `f"{1e-5:.4f}" == "0.0000"`; `f"{3.2e-4:.4f}" == "0.0003"` (6.25 % error).

All three are **S2**: they need a specific geometry or a specific `Z` to bite, and when they bite,
CLOUDY still runs.

### 6. The eps clamp that is documented as a safety guard (R-07)

> **Lens B (the comment):** "`# Convert linear pc^-3 → log10 pc^-3 (eps guard for safety; TRINITY
> writes positive values, but defends against future regressions).`"

> **Lens A (the consequence):** "that clamp produces a dlaw row of `continue <logr>  -363.1207` — a
> finite value that passes the `isfinite` guard at `dlaw.py:194` and is written out."

The comment's own stated purpose — defending against a future regression — is exactly what the code
fails to do. A regression that zeroes the ambient density is converted into a finite, plausible-typed
number that survives every downstream check and reaches CLOUDY as `hden = 10^-363`. The module
raises `SnapshotInvalid` for every other non-physical input (A lists a dozen). This one it launders.
**S2.**

### 7. The two legacy parsers that disagree in the same file (R-08)

> **Lens B:** `_parse_simulation_end` — "`# value is '<number> <unit>' — take the first
> whitespace-split token`". For `_parse_summary_txt`, B's transcribed coercion list is "bool, None,
> nan/inf, int, float, Python-literal, else string" — **no unit-stripping step at all**.

> **Lens A:** "`_parse_simulation_end` explicitly takes `value.split()[0]` before `float()`
> (`:264`), while `_parse_summary_txt` hands the whole remainder to `_coerce_scalar` (`:179-184`),
> which for `"0.02 Zsun"` falls through every branch and returns the raw string."

B's silence corroborates A's finding: one parser documents unit-stripping, the other's documented
type ladder has no place for it. The string then reaches `float(bundle.summary["ZCloud"])` and raises
`ValueError` — which A notes is **not** in the `--all` skip tuple, so the whole batch aborts after
partial output. Loud, at least. **S2.**

### 8–13. Remaining S2s, in brief

- **R-25 silent sort/dedup.** B documents "sort and dedup adjacent duplicates (keep last)"; A adds
  that both are silent, so a contact discontinuity represented as two points at one radius silently
  loses its inner density, and a non-monotonic input array is silently straightened into a
  plausible-looking, physically different table that then exports cleanly.
- **R-10 asymmetric `Z` validation.** `z_override` is checked finite and positive; `ZCloud` is not.
  `f"{float('nan'):.4f}" == "nan"` goes straight into the deck. Both feed the same field.
- **R-11 unbounded quiet time-matching.** All three time pickers use `mode="closest", quiet=True`;
  only `--age` reports how far off it landed. `--t-now 12.0` on a run that ended at 2.3 Myr exports
  the 2.3 Myr snapshot and says nothing. (Also: `--age` reads `bundle.metadata["tSF"]` unguarded,
  raising a bare `KeyError` where the sibling path raises a clean `SnapshotInvalid`.) B independently
  flagged the unstated metric and tie-break — but B's claim that the *filename* hides the divergence
  is **contradicted** by A, which reports the filename carries the selected snapshot's own age.
- **R-12 `--phase` round-trip.** B transcribes the comment explaining *why* the round-trip exists
  (recovering the unfiltered index). A shows the code then uses the *re-resolved snapshot* rather
  than the filtered one, so with duplicate `t_now` values the deck can be built from a different
  snapshot — possibly the other side of a phase transition — while TITLE and filename advertise the
  requested phase.

---

## Units verdict — the astro→cgs boundary

The joint A+B answer, which is neither A's clearance alone nor B's alarm alone.

### Cleared

Lens A traced the arithmetic and Lens B traced the claims, and on the mechanics they agree
completely. Every conversion in this slice is applied **exactly once**, in the **correct direction**,
with **no log₁₀ doubled and none omitted**:

| Conversion | Factor | Applied at | Corroboration |
|---|---|---|---|
| pc → cm (dlaw radii) | `+18.489350545222138` | `dlaw.py:174,176` | A arithmetic ≡ B comment |
| pc → cm (`LOG_RIN`/`LOG_ROUT`) | same | `snapshot_to_deck.py:180,182,198` | A only (B: undocumented) |
| pc⁻³ → cm⁻³ (shell, already log) | `−55.46805163566641` added, no re-log | `dlaw.py:175,177` | A ≡ B |
| pc⁻³ → cm⁻³ (ambient, linear) | `log10` in caller, then same offset | `snapshot_to_deck.py:244` + `dlaw.py:177` | A ≡ B |
| Myr⁻¹ → s⁻¹ (`Qi`) | `−13.499103967085228`, log applied | `snapshot_to_deck.py:181` | A arithmetic ≡ B comment |
| Myr → yr (age) | `×1e6`, exactly once | `snapshot_to_deck.py:167-168` | A only |

Two clearances worth stating explicitly, because "nothing is wrong here" is a result:

1. **No cm/pc skew between the dlaw columns.** A verified `3·log10(pc2cm) + log10(ndens_au2cgs) ==
   0.0` **exactly** in float64. The parsec implied by the radius column and the parsec implied by the
   density column are the same parsec to the last bit.
2. **No double-conversion on the two density arms.** The shell arm arrives log and is offset; the
   ambient arm arrives linear, is logged by the caller, and is offset by the same code. They converge
   on `log10(n/cm⁻³)` by different routes with exactly one offset each — the asymmetry is in *where*
   the log is taken, not in *how many times*.

Two of B's units findings are **contradicted by A and are hereby demoted**:

- **B-01's failure scenario** — that `LOG_RIN`/`LOG_ROUT` might be `log10(pc)` while the table is
  `log10(cm)`, an 18.49-dex offset — **does not occur**. The offset is applied at
  `snapshot_to_deck.py:182,198`.
- **B-04's failure scenario** — that the code hardcodes two independently-rounded literals — **does
  not occur** in this module. Both offsets are computed at runtime from `INV_CONV`.

### Open

| Risk | Magnitude if wrong | Status |
|---|---|---|
| **`n` vs `n_H`** — is `log_shell_n_arr` total or hydrogen density? | ~0.35 dex, uniform | **Open. R-01.** Both lenses, opposite directions. |
| **`ZREL`** — solar-relative factor or absolute mass fraction? | up to ~1.85 dex | **Open. R-02.** Both lenses, opposite directions. |
| **`Qi`** — is the snapshot field really photons Myr⁻¹? | 13.5 dex | **Near-closed.** B has the comment; A has a magnitude check (`log_qh = 53.0` for a 10⁶ M☉ cluster, self-consistent). A also notes the wrong assumption would be *obvious* (≈39.5), not subtle. Confirm at the producer. |

### The documentation of the boundary — S3, not a units bug

B's headline is correct and stands: **`LOG_RIN`/`LOG_ROUT` carry no documented unit and no documented
log convention anywhere in the slice.** The `+18.4894` step is documented only inside `dlaw.py`, for
a *different* pair of values (`r_in_pc`/`r_out_pc`, used for the bracket check). The same is true of
`LOG_QH`'s log (asserted only by the key name) and of `ZREL` entirely.

But B's alarm must not inflate into A's territory. The correct joint verdict:

> **The conversion is right and nobody wrote it down.** That is a real finding — in a repo whose own
> `CLAUDE.md` names units as the recurring bug class, an undocumented unit boundary is exactly how
> the next regression gets in — but it is **S3 (code correct, contract missing)**, not S1 or S2. It
> would be intellectually dishonest to rate an undocumented-but-correct conversion the same as a
> wrong one. It would be equally dishonest to let A's clearance erase the gap: A could only clear it
> by *reading the arithmetic*, which is precisely what a maintainer without a stripped-code audit
> cannot do.

The sharp version: **the only reason this slice's unit boundary is known to be correct is that an
auditor computed it by hand. The code says nothing, and there is no test comparing `LOG_RIN` against
the dlaw table's first row.** That absent test is the concrete remedy for R-14 and R-17 together.

---

## Scope boundary (recorded, not a finding)

Both lenses independently hit the same wall, and neither could pass it:

> **Lens A:** "The deck template itself — `trinity2cloudy.in_template`, referenced at
> `trinity_to_cloudy.py:67` — is **not in my slice**. … Everything else — which CLOUDY command
> consumes `LOG_RIN`/`LOG_ROUT`/`LOG_QH`/`AGE_YR`/`ZREL`, in what order, and whether `DLAW_BLOCK`
> and `DLAW_ROWS` are both interpolated (which would duplicate the table) — lives in the template."

> **Lens B:** "**No command order is claimed anywhere.** The prose never enumerates the deck's
> commands … Those live in 'the bundled .in template', which is not in this slice, so the deck
> contract as claimed is limited to the eight substitution keys."

Consequence: **command ordering, unit expectations of the consuming commands, and the possibility
that the dlaw table is emitted twice are all unauditable from S13b.** This is a slicing artefact, not
a defect, and I record it as such.

**Named follow-up:** a slice or task covering `trinity/_output/cloudy/trinity2cloudy.in_template` and
`trinity_linelist.dat`, answering (a) which command consumes each of the eight keys and in what
units, (b) whether both `DLAW_BLOCK` and `DLAW_ROWS` appear (double table), (c) whether the metals
command is solar-relative (settles R-02), (d) whether `AGE_YR`'s `%.4e` form with a lowercase signed
two-digit exponent is accepted by the command it lands in, and (e) whether the density column feeds
`hden` (settles R-01's consuming side).

---

## Open questions — each a single lookup

1. **`grep -rn "log_shell_n_arr" trinity/ --exclude-dir=_output`** → read the producing expression.
   Is it `ρ/(μ m_H)` (total particle density) or `ρ X_H/m_H` (hydrogen)? **Settles R-01.**
2. **`trinity/_output/cloudy/trinity2cloudy.in_template`** → does the metallicity line read
   `metals {{ZREL}}` (CLOUDY solar-relative) or an absolute-abundance form? Cross-read `ZCloud`'s
   declaration in `trinity/_input/` (schema/defaults). **Settles R-02.**
3. **Same template** → is `{{DLAW_BLOCK}}` present *and* `{{DLAW_ROWS}}` present? If both, the table
   is emitted twice. **Settles the A-18/B-19 double-emission risk.**
4. **`grep -rn "Qi" trinity/` at the producer** → confirm the snapshot field is photons Myr⁻¹, not
   photons s⁻¹. **Closes the last units item.**
5. **`trinity/_functions/unit_conversions.py`** → are `pc2cm` and `ndens_au2cgs` defined from one
   another, or independently? A verified they agree bit-exactly *today*; the question is whether
   anything enforces it. **Sizes R-17.**
6. **`test/` (any file)** → is there any test that compares `LOG_RIN` against the first dlaw row, or
   that round-trips a generated deck? A's account implies no. **Sizes R-14/R-05.**
7. **`trinity/_input/registry.py::_validate_dens_profile`** → diff its accepted set against
   `run_loader.py`'s `frozenset({"densBE","densPL"})`. **Settles R-18's drift half.**
8. **`grep -rn "in snap" trinity/ ; grep -n "__contains__\|__iter__" <TrinityOutput.Snapshot>`** →
   does the documented never-terminating membership hazard exist, and does any other caller hit it?
   **Settles R-16's hazard half** (B's strongest safety claim, unverifiable from either lens).
9. **`docs/dev/`** → find the plan document defining "Step 0/4/5", "Option B", "Phase 5/6"; check
   whether the "Step 5 smoke test" was ever run and against which CLOUDY build. **Settles R-13 and
   R-26.** (Per project convention, treat whatever is found there as unverified.)
10. **`_coerce_scalar` in `run_loader.py`** → feed it `'{"a": 1}'`; does it return a dict or a str?
    **Settles R-27.**

---

```json
[
  {
    "id": "S13b-R-01",
    "file": "trinity/_output/cloudy/dlaw.py",
    "line": 177,
    "class": "units",
    "severity": "S2",
    "claim": "The dlaw density column is a pure geometric unit shift with no composition factor anywhere in the slice, while the module docstring names the emitted column both 'log10 n [cm^-3]' (prose) and 'log10(n_H/cm^-3)' (format block). CLOUDY's dlaw table second column is hydrogen density by definition, so deck correctness depends on an undocumented property of the source array.",
    "evidence": "CORROBORATED, opposite directions. Lens A (code): dlaw.py:175,177 `log_ndens_offset = math.log10(INV_CONV.ndens_au2cgs)` / `log_n_cm3 = log_n_pc3 + log_ndens_offset` is the ONLY transformation on the density column; grep for mu/m_H/n_H/hden across trinity/_output/cloudy/ returns no composition factor. Lens B (prose): dlaw.py:1 'converts (r [pc], log10 n [pc^-3]) pairs into the (log10 r [cm], log10 n [cm^-3]) form CLOUDY expects' vs the format block three lines later 'continue {log10(r/cm):.6f} {log10(n_H/cm^-3):.4f}'; no composition, mean-molecular-weight or He correction claimed anywhere.",
    "expected": "Either log_shell_n_arr is hydrogen number density (then fix the docstring's ambiguity), or a composition factor n_H = n_tot/(1 + y_He + x_e) must be applied before writing the column, and the choice must be stated at the boundary.",
    "failure_scenario": "If log_shell_n_arr is total particle density, every exported deck overstates n(H) by ~0.35 dex uniformly (~0.05-0.15 dex if only H/He and neutral). The CLOUDY model runs to completion and produces a plausible spectrum with a systematically wrong ionisation parameter and wrong line ratios. No TRINITY test can catch it; the error lives entirely in the third-party code's output.",
    "repro": "grep -rn 'log_shell_n_arr' trinity/ --exclude-dir=_output ; read the producing expression and determine whether it divides by mu*m_H (total) or X_H*m_H (hydrogen). Then check trinity/_output/cloudy/trinity2cloudy.in_template for whether the dlaw table feeds hden.",
    "confidence": "high"
  },
  {
    "id": "S13b-R-02",
    "file": "trinity/_output/cloudy/snapshot_to_deck.py",
    "line": 212,
    "class": "units",
    "severity": "S2",
    "claim": "ZREL is bundle.summary['ZCloud'] formatted straight through with zero conversion, and no prose anywhere states whether it is a solar-relative scale factor, an absolute metal mass fraction, or a log10 value.",
    "evidence": "CORROBORATED, opposite directions. Lens A (code): snapshot_to_deck.py:212 `zrel = float(bundle.summary['ZCloud'])`; :274 `'ZREL': f'{zrel:.4f}'` -- transform applied: NONE; A's own per-quantity table marks the CLOUDY-side unit 'linear relative-to-solar (assumed)'. Lens B (prose): snapshot_to_deck.py:59 'Z handling -- bundle.summary[\"ZCloud\"] by default; z_override (>0, finite) wins'; :200 '# Z scale' -- no unit, no normalisation, no log convention claimed.",
    "expected": "State whether ZREL is solar-relative linear, absolute mass fraction, or log10, and which CLOUDY command consumes it; convert if the two differ.",
    "failure_scenario": "If ZCloud is an absolute metal mass fraction (e.g. 0.014) and the template's command expects a solar-relative factor, every exported model runs at ~1.4 percent of the intended metallicity -- a ~1.85 dex error, five times larger than the n/n_H risk. Cooling and every line ratio change; no error is raised.",
    "repro": "Read ZCloud's declaration/default in trinity/_input/ (schema or default.param) and the metallicity command in trinity/_output/cloudy/trinity2cloudy.in_template; compare conventions.",
    "confidence": "high"
  },
  {
    "id": "S13b-R-03",
    "file": "trinity/_output/cloudy/dlaw.py",
    "line": 39,
    "class": "regime",
    "severity": "S2",
    "claim": "The comment justifying edge_threshold=50 makes a quantitative regime claim that is false for the geometry the code actually runs in: |dlog n / dlog r| for a SMOOTH thin shell is O(1e3-1e4), not O(1), so every interval is classified an ionisation front, densification is skipped, and --min-rows is silently unmet.",
    "evidence": "DOC-DRIFT with a measurement. Lens B (comment, dlaw.py:39): '# |delta log n / delta log r| above this counts as an IF-like discontinuity. PL profiles are O(1); transition-phase IFs in TRINITY snapshots are O(1e5). 50 separates them with margin.' Lens A (measured): shell r = [5.0, 5.0016, 5.0033, 5.005] pc (0.1% thick) with log n = [57.0, 58.0, 58.7, 59.0] pc^-3 gives slopes [7196.7, 4742.9, 2033.4], all > 50; dlaw.py:229-236 then warns and returns the table unchanged, and the only hard floor is dlaw.py:188 (2 rows).",
    "expected": "A threshold on a log-log slope must be scale-aware. For a shell of relative thickness d, dlog_r across the whole shell is ~d/ln10, so any O(1 dex) density contrast produces slopes of 1e3-1e4 regardless of smoothness. Normalise by shell thickness, or classify on curvature rather than slope.",
    "failure_scenario": "A user passes --min-rows 10 on a thin-shell snapshot and receives a 4-row dlaw table with only a UserWarning. CLOUDY interpolates the shell as three log-log line segments instead of the requested sampling. The comment tells a maintainer the threshold is well-separated with margin, so the symptom is not attributed to the threshold.",
    "repro": "python3 -c \"import math,numpy as np; L=math.log10(3.0856775814913674e18); r=np.array([5.0,5.0016,5.0033,5.005]); n=np.array([57.,58.,58.7,59.]); lr=np.log10(r)+L; print(np.abs(np.diff(n)/np.diff(lr)))\"",
    "confidence": "high"
  },
  {
    "id": "S13b-R-04",
    "file": "trinity/_output/cloudy/trinity_to_cloudy.py",
    "line": 277,
    "class": "state",
    "severity": "S2",
    "claim": "--all is documented as writing one deck per snapshot, but a user-supplied --prefix short-circuits the per-index filename path, so all N snapshots write to the same two filenames while manifest.json reports N distinct outputs.",
    "evidence": "VIOLATED DOCUMENTED CONTRACT. Lens B (docstring, trinity_to_cloudy.py:6): '--all one deck per snapshot, plus manifest.json'; :273 'Auto-build a filename-safe prefix: <model>_<idx>_<phase>_t<age>myr'. Lens A (code): :277-281 `if args.prefix is not None: return _UNSAFE_PREFIX_RE.sub('_', args.prefix)` returns before the auto path that appends pick.index; :429-430 `deck_path = out_dir / f'{prefix}.in'`; records appended per snapshot at :378-392 with `'deck': f'{prefix}.in'`.",
    "expected": "--all must produce one file per snapshot: treat a user prefix as a stem the per-snapshot discriminator is appended to, or reject the flag combination at parse time as --all + --dry-run already is.",
    "failure_scenario": "`trinity_to_cloudy -F run --all --prefix mymodel` writes mymodel.in once per snapshot; only the last survives. manifest.json lists every snapshot with the same deck filename, _print_summary reports 'Converted N snapshots', exit code 0. A downstream batch runner reads the manifest and runs the same deck N times, producing N identical spectra labelled as N different epochs.",
    "repro": "Inspect trinity_to_cloudy.py:277-281 against :348-392: _build_prefix is called once per pick but returns a pick-independent value when args.prefix is set.",
    "confidence": "high"
  },
  {
    "id": "S13b-R-05",
    "file": "trinity/_output/cloudy/snapshot_to_deck.py",
    "line": 272,
    "class": "numerical",
    "severity": "S2",
    "claim": "The same physical radius is emitted at two precisions -- LOG_RIN/LOG_ROUT at %.4f in log10(cm) (1.15e-4 relative) and the dlaw radius column at %.6f (2.3e-6 relative) -- so the deck's declared integration range can fall outside the tabulated table, and for a shell thinner than ~2.3e-5 relative the two radii collapse to the same string.",
    "evidence": "Lens A (code + measurement): snapshot_to_deck.py:182 `log_rin = math.log10(R2) + log_pc_per_cm`; :272-273 `'LOG_RIN': f'{log_rin:.4f}', 'LOG_ROUT': f'{log_rout:.4f}'` vs dlaw.py:200 `lines.append(f'{dlaw_row_prefix}{lr:.6f}  {ln:.4f}')`. Lens B corroborates the ABSENCE of any contract: LOG_RIN/LOG_ROUT have no documented unit, no log base and no format anywhere in the slice (snapshot_to_deck.py:59 lists the key names only); the .6f/.4f dlaw format IS documented at dlaw.py:1.",
    "expected": "The deck radii and the dlaw endpoints describe the same two numbers; emit them at matching or finer precision than the table so the model range is provably inside the tabulated range, and document the invariant.",
    "failure_scenario": "R2 = 0.5 pc: true log_rin = 18.188320549558156. Deck gets '18.1883' (= 0.4999763 pc); first dlaw row gets '18.188321' (= 0.5000000 pc). CLOUDY's starting radius is 4.7e-5 relative BELOW the table's first entry, so the first zone is off the end of the table. Separately, with R2 = 5.0 pc and relative shell thickness 2e-5 or 5e-5, LOG_RIN and LOG_ROUT both render '19.1883' -- a zero-thickness model.",
    "repro": "python3 -c \"import math; L=math.log10(3.0856775814913674e18); a=math.log10(0.5)+L; print(f'{a:.4f}', f'{a:.6f}'); b=math.log10(5.0)+L; c=math.log10(5.0*(1+5e-5))+L; print(f'{b:.4f}', f'{c:.4f}')\"",
    "confidence": "high"
  },
  {
    "id": "S13b-R-06",
    "file": "trinity/_output/cloudy/snapshot_to_deck.py",
    "line": 274,
    "class": "numerical",
    "severity": "S2",
    "claim": "Metallicity is emitted with fixed 4-decimal formatting, destroying any sub-solar value below ~1e-2 and zeroing anything below 5e-5.",
    "evidence": "Lens A: snapshot_to_deck.py:274 `'ZREL': f'{zrel:.4f}'` (no log, no scaling); measured f'{1e-5:.4f}' == '0.0000', f'{3.2e-4:.4f}' == '0.0003' (6.25% error), f'{1e-4:.4f}' == '0.0001' (one significant figure). Lens B: no format contract exists for ZREL anywhere in the prose (snapshot_to_deck.py:59, :200) -- nothing documented is violated because nothing is documented.",
    "expected": "A quantity spanning orders of magnitude should be emitted in %.4e or a log form, not fixed-point. Compounds with R-02: the format is wrong regardless of which normalisation ZREL turns out to use.",
    "failure_scenario": "Z = 1e-5 Zsun renders as '0.0000', so CLOUDY runs a metal-free model: cooling, thermal structure and the entire emission-line spectrum are qualitatively wrong, with no error raised. Low-metallicity GMC studies are precisely the regime where this matters.",
    "repro": "python3 -c \"print(f'{1e-5:.4f}', f'{3.2e-4:.4f}', f'{1e-4:.4f}')\"",
    "confidence": "high"
  },
  {
    "id": "S13b-R-07",
    "file": "trinity/_output/cloudy/snapshot_to_deck.py",
    "line": 244,
    "class": "silent-failure",
    "severity": "S2",
    "claim": "A zero or negative ambient density is clamped to the smallest positive double rather than rejected, producing a finite but meaningless dlaw row that passes every downstream guard -- and the comment describes this clamp as a safety guard against future regressions, which is exactly what it fails to be.",
    "evidence": "DOC-DRIFT. Lens B (comment, snapshot_to_deck.py:240): '# Convert linear pc^-3 -> log10 pc^-3 (eps guard for safety; TRINITY writes positive values, but defends against future regressions).' Lens A (code + measurement): :242-244 `eps = np.finfo(float).tiny` / `ambient_log_n_pc3 = np.log10(np.maximum(amb_n, eps))`; the result survives dlaw.py:194's isfinite check and is written as `continue <logr>  -363.1207` (log10(2.225e-308) = -307.6527 plus the -55.4681 offset).",
    "expected": "Reject a non-positive ambient density (the module raises SnapshotInvalid for every other non-physical input) or drop those points. A guard whose stated purpose is catching an upstream regression must not convert that regression into a finite number.",
    "failure_scenario": "A run with nISM = 0 plus --radius-out beyond rShell writes 'continue <logr> -363.1207' for every spliced ambient row. No error, no warning, exit 0. CLOUDY receives hden = 10^-363 in the outer zones -- a vacuum where an ambient medium was intended -- and the deck runs.",
    "repro": "python3 -c \"import math,numpy as np; print(f'continue {math.log10(np.finfo(float).tiny)+math.log10(1/2.937998946096347e+55):.4f}')\"",
    "confidence": "high"
  },
  {
    "id": "S13b-R-08",
    "file": "trinity/_output/cloudy/run_loader.py",
    "line": 184,
    "class": "divergence",
    "severity": "S2",
    "claim": "The two legacy text parsers in the same module disagree on unit-suffixed values: _parse_simulation_end strips the suffix, _parse_summary_txt returns the raw string, which then reaches float() and aborts an --all batch mid-write.",
    "evidence": "Lens A (code): run_loader.py:264 `tok = value.split()[0] if value else ''` then `float(tok)` versus :179-184 `parts = line.split(None, 1)` ... `_coerce_scalar(value_str)` whose terminal `return s` at :324 keeps anything float() rejects. Lens B (prose) corroborates the asymmetry from the docs alone: :263 documents '# value is \"<number> <unit>\" -- take the first whitespace-split token' for one parser, while the other's documented coercion ladder (:155 'bool, None, nan/inf, int, float, Python-literal, else string') contains no unit-stripping step at all.",
    "expected": "Both legacy parsers read the same run's scalars and should coerce numbers-with-units identically.",
    "failure_scenario": "A legacy summary line 'ZCloud 0.02 Zsun' yields the string '0.02 Zsun'. snapshot_to_deck.py:212 executes float('0.02 Zsun') -> ValueError, which is NOT in the --all skip tuple at trinity_to_cloudy.py:393, so the batch aborts with a traceback after some decks are already on disk and before manifest.json is written. The identical line in simulationEnd.txt would have parsed fine.",
    "repro": "python3 -c \"s='0.02 Zsun'; print(float(s.split()[0])); print(float(s))\"  # second call raises ValueError",
    "confidence": "medium"
  },
  {
    "id": "S13b-R-09",
    "file": "trinity/_output/cloudy/dlaw.py",
    "line": 200,
    "class": "numerical",
    "severity": "S2",
    "claim": "Dedup and the strict-monotonicity guard both operate on float radii while the file is written at %.6f, so a table that passes every check can contain rows with identical printed abscissae and different densities -- breaking the documented 'ordered by increasing radius' property in the artefact CLOUDY actually reads.",
    "evidence": "Lens A (code + measurement): dlaw.py:129 `keep[:-1] = r_pc[:-1] != r_pc[1:]` (exact float dedup); :192 `if not np.all(np.diff(log_r_cm) > 0): raise DlawError(...)`; :200 `f'{dlaw_row_prefix}{lr:.6f}  {ln:.4f}'`. Measured with r = [5.0, 5.0*(1+1e-6), 5.0*(1+2e-6)]: the guard returns True and three rows all read 'continue 19.188321' with densities 1.5319 / 2.0319 / 2.5319. Lens B (prose): dlaw.py:1 documents the .6f format and section :124 documents sort-and-dedup; B's transcription describes a radius table with 'rows ordered by increasing radius'.",
    "expected": "Enforce uniqueness on the emitted representation, not only on the in-memory floats: %.6f in log10 r resolves only 2.30e-6 in relative radius.",
    "failure_scenario": "Any shell sampled with consecutive spacing dr/r < 2.3e-6 (e.g. a shell of relative thickness 1e-4 sampled with ~100 points) emits repeated abscissae. CLOUDY's dlaw reader sees a zero-width interval with two densities; behaviour is undefined and version-dependent -- it may interpolate to infinity, silently keep one, or error.",
    "repro": "python3 -c \"import math,numpy as np; L=math.log10(3.0856775814913674e18); r=np.array([5.0,5.0*(1+1e-6),5.0*(1+2e-6)]); lr=np.log10(r)+L; print(np.all(np.diff(lr)>0)); [print(f'continue {x:.6f}') for x in lr]\"",
    "confidence": "high"
  },
  {
    "id": "S13b-R-10",
    "file": "trinity/_output/cloudy/snapshot_to_deck.py",
    "line": 212,
    "class": "divergence",
    "severity": "S2",
    "claim": "The two metallicity sources feeding the same deck field are validated asymmetrically: the --z-override path checks finiteness and positivity, the bundle.summary['ZCloud'] path checks nothing, so NaN/inf/negative is formatted straight into the deck.",
    "evidence": "Lens A (code): snapshot_to_deck.py:201-206 `if z_override is not None: if not (math.isfinite(z_override) and z_override > 0): raise SnapshotInvalid(...)` versus :208-212 `if 'ZCloud' not in bundle.summary: raise ...; zrel = float(bundle.summary['ZCloud'])`; f'{float(\"nan\"):.4f}' == 'nan'. Lens B (prose) documents only 'z_override (>0, finite) wins' -- the asymmetry is faithfully reflected in the docs, so this is code sloppiness the doc mirrors rather than doc-drift.",
    "expected": "Both sources feed the same field and should pass the same validity gate.",
    "failure_scenario": "A run whose summary carries ZCloud = NaN (or a legacy summary.txt value that failed to coerce) yields ZREL = 'nan' in the deck. The file is written, the CLI reports WROTE and exits 0, and CLOUDY sees a non-numeric token on the abundance line.",
    "repro": "python3 -c \"print(repr(f'{float(\\\"nan\\\"):.4f}'), repr(f'{float(\\\"inf\\\"):.4f}'))\"",
    "confidence": "high"
  },
  {
    "id": "S13b-R-11",
    "file": "trinity/_output/cloudy/trinity_to_cloudy.py",
    "line": 200,
    "class": "silent-failure",
    "severity": "S2",
    "claim": "All time-based pickers use mode='closest' with quiet=True and no tolerance, so a requested time far outside the sampled range silently yields the nearest snapshot; only the --age path reports the miss distance. --age additionally reads metadata['tSF'] unguarded.",
    "evidence": "CORROBORATED. Lens A (code): :196 and :200 `bundle.output.get_at_time(target_t, mode='closest', quiet=True)`, same at :224-226; the delta report at :481-487 is guarded by `if args.age is not None`; :195 `target_t = args.age + float(bundle.metadata['tSF'])` raises a bare KeyError where snapshot_to_deck.py:120-121 raises a clean SnapshotInvalid. Lens B (prose): '--age MYR ... picks closest snapshot' with no closeness metric, no tie-break and no default for --pick documented (trinity_to_cloudy.py:6). NOTE: B's additional claim that the filename embeds the REQUESTED age is contradicted by A, which reports the filename uses the selected snapshot's own t_now - tSF; that sub-claim is demoted.",
    "expected": "A tolerance, or at least an unconditional proximity report, on every time-based pick, so that asking for a time the run never reached is visible.",
    "failure_scenario": "`--t-now 12.0` on a run that terminated at t = 2.3 Myr emits a deck built from the 2.3 Myr snapshot and prints only 'Picked snapshot: index=..., t_now=2.3000 Myr'. Nothing states the request was missed by 9.7 Myr. In a scripted sweep over requested epochs, many requests silently collapse onto the same final snapshot.",
    "repro": "Read trinity_to_cloudy.py:194-227 and :473-492; the --t-now branch has no corresponding delta print.",
    "confidence": "medium"
  },
  {
    "id": "S13b-R-12",
    "file": "trinity/_output/cloudy/trinity_to_cloudy.py",
    "line": 223,
    "class": "other",
    "severity": "S2",
    "claim": "The --phase picker discards the snapshot it selected: it reads that snapshot's t_now and re-resolves by closest time, so with duplicate t_now values the deck can be built from a different snapshot, possibly in a different phase from the one requested.",
    "evidence": "Lens A (code): :214-227 `filtered = bundle.output.filter(phase=args.phase)` ... `target_t_now = filtered[which]['t_now']` / `snap = bundle.output.get_at_time(target_t_now, mode='closest', quiet=True)` / `return [PickedSnapshot(index=snap.index, snap=snap)]`. Lens B (comment, :221): '# filter() re-indexes from 0; map back to the original index by round-tripping through get_at_time on the unfiltered output.' -- the documented purpose is index recovery only; the code also swaps the snapshot object.",
    "expected": "Use filtered[which] directly for the snapshot and the round-trip only for the index, as the comment describes.",
    "failure_scenario": "Two snapshots sharing a t_now (a phase transition written twice, or a restart) make the lookup return whichever the reader prefers -- possibly the one on the other side of the transition. The deck's TITLE and filename then advertise a phase the deck was not built from, and a phase-comparison study silently compares the wrong pair.",
    "repro": "Read trinity_to_cloudy.py:213-227; the filtered snapshot is discarded and only its t_now survives.",
    "confidence": "medium"
  },
  {
    "id": "S13b-R-13",
    "file": "trinity/_output/cloudy/dlaw.py",
    "line": 1,
    "class": "citation",
    "severity": "S3",
    "claim": "The CLOUDY input syntax this module emits is self-declared a best guess, no CLOUDY manual/Hazy section is cited anywhere in the slice, the referenced verification ('Step 5 smoke test') is never named or its outcome reported, and the DEFAULT_* values the docs point readers at are never printed in the prose.",
    "evidence": "Lens B (prose): dlaw.py:1 'Output format (defaults -- best-guess for CLOUDY C17/C22; see Step 5 smoke test)'; :31 '# Best-guess CLOUDY syntax (Step 0 / Option B). Override at call site if a live smoke test reveals a different working form.'; __init__.py:1 points at 'the DEFAULT_* constants' and dlaw.py:64 at 'module-level defaults', neither of which prints a value; only edge_threshold's 50 appears. Lens A supplies the actual values the prose withholds: DEFAULT_DLAW_OPEN = 'dlaw table radius', DEFAULT_DLAW_ROW_PREFIX = 'continue ', DEFAULT_DLAW_CLOSE = 'end of dlaw' (dlaw.py:33-35), min_rows default 10 (:37), and confirms no caller ever overrides them.",
    "expected": "Cite the Hazy section defining dlaw table radius / continue / end of dlaw, record whether the smoke test ran and against which CLOUDY build, and state the default values in the docstring since they define every emitted deck.",
    "failure_scenario": "If the guessed grammar is wrong in any detail (first data row must not carry 'continue', 'end of dlaw' not accepted in C22, column separator significance), every deck ever produced is unrunnable or -- worse -- misparsed into a different density law. Nothing in-repo would detect it; the failure surfaces only in the user's CLOUDY run, and only if CLOUDY happens to be strict. Rubric severity S3 (doc/verification), but the true cost is the correctness premise of the whole export.",
    "repro": "Run a generated .in through CLOUDY C17 and C22 and confirm the dlaw block parses; compare the emitted grammar against the Hazy dlaw section. Then read dlaw.py:33-42 for the DEFAULT_* values the prose omits.",
    "confidence": "high"
  },
  {
    "id": "S13b-R-14",
    "file": "trinity/_output/cloudy/snapshot_to_deck.py",
    "line": 59,
    "class": "units",
    "severity": "S3",
    "claim": "The package's astro-to-cgs boundary is entirely undocumented on the template-key side: LOG_RIN, LOG_ROUT, LOG_QH and ZREL carry no stated unit and no stated log convention anywhere in the slice. The conversions themselves are correct.",
    "evidence": "JOINT VERDICT. Lens B: 'LOG_RIN / LOG_ROUT are the radii CLOUDY integrates over, and the prose in this slice never once states their unit or their log base'; the only radius-unit claim is dlaw.py:64's r_in_pc/r_out_pc, a DIFFERENT pair used for a bracket check; LOG_QH's log is asserted only by the key name (:181 says only '# ph/Myr -> ph/s'); ZREL has no unit at all. Lens A verifies the arithmetic IS correct: snapshot_to_deck.py:180,182,198 apply +log10(INV_CONV.pc2cm) = 18.489350545222138 to both radii; :181 applies log10(Qi) - log10(Myr2s); and 3*log10(pc2cm) + log10(ndens_au2cgs) == 0.0 exactly in float64, so the radius and density columns share a bit-identical parsec. B-01's feared 18.49-dex pc/cm offset does NOT occur and is demoted.",
    "expected": "Each substitution key documents its unit and log convention at the boundary, e.g. 'LOG_RIN/LOG_ROUT: log10 of the inner/outer radius in cm', and a test asserts LOG_RIN equals the dlaw table's first radius to the emitted precision.",
    "failure_scenario": "No wrong number today. The cost is that the only reason this boundary is known correct is that an auditor recomputed it by hand; a maintainer has no documented invariant to preserve and no test guarding it, in a repo whose own CLAUDE.md names units as the recurring bug class. The next edit to either conversion site has nothing to fail against.",
    "repro": "Read snapshot_to_deck.py:59 (docstring key list) and :180-198 (the conversions) side by side; then grep test/ for any assertion relating LOG_RIN to the dlaw table's first row.",
    "confidence": "high"
  },
  {
    "id": "S13b-R-15",
    "file": "trinity/_output/cloudy/trinity_to_cloudy.py",
    "line": 238,
    "class": "other",
    "severity": "S3",
    "claim": "The status-gate docstring states a blanket refusal rule then enumerates only three force-overridable bands, leaving 30-49, 60-98 and >=100 unspecified. The code fails closed, so the doc gap is the only defect -- but isinstance(True, int) means a JSON `true` exit code is read as clean.",
    "evidence": "Lens B (prose, :238): 'Refuse to convert runs whose termination exit code is not in the clean range (0-9). Inspection-required (50-59 or 99) and error (10-29) outcomes both require --force.' -- 30-49, 60-98, >=100 absent. Lens A (code, :237-256) RESOLVES it: `isinstance(exit_code, int) and 0 <= exit_code <= 9`, else refuse unless --force; a missing or non-int code refuses. B's feared permissive fall-through does not exist and is demoted. A adds: `outcome = ... or 'unknown'` (:246) swallows ''/0/None alike (message-only), and isinstance(True, int) is True in Python.",
    "expected": "State the behaviour for every code, including missing/None and out-of-enumeration values. Reject bool explicitly if exit codes are read from JSON.",
    "failure_scenario": "Doc-only for the code bands. For the bool quirk: a metadata.json whose termination block carries `\"exit_code\": true` is accepted as a clean exit code 1... in fact as int 1, which is inside 0-9, so a run marked with a boolean instead of a code is silently treated as clean and exported without --force.",
    "repro": "python3 -c \"print(isinstance(True, int), 0 <= True <= 9)\"  # True True",
    "confidence": "medium"
  },
  {
    "id": "S13b-R-16",
    "file": "trinity/_output/cloudy/snapshot_to_deck.py",
    "line": 99,
    "class": "silent-failure",
    "severity": "S4",
    "claim": "The required-key check uses a _MISSING sentinel, so a key PRESENT with value None passes validation and then raises TypeError at float(None) -- and TypeError is not in the --all batch skip tuple, so one malformed snapshot aborts the whole batch after partial output. The sentinel idiom itself exists to dodge a documented never-terminating membership test.",
    "evidence": "Lens A (code): :37 `_MISSING = object()`; :97-100 `missing = [k for k in REQUIRED_SNAPSHOT_KEYS if snap.get(k, _MISSING) is _MISSING]`; :105 `t_now = float(snap['t_now'])`; trinity_to_cloudy.py:393 `except (SnapshotInvalid, DlawError, UnsubstitutedPlaceholder) as e:`; manifest and linelist are written after the loop (:413-416). Lens B (comment, :34): 'TrinityOutput.Snapshot has __getitem__ but no __contains__/__iter__, so `key in snap` falls back to integer-indexed iteration that never terminates. Use snap.get(k, _MISSING).' -- B calls this the strongest safety claim in the slice; A confirms the idiom is used but cannot verify the hazard (the type is outside the slice).",
    "expected": "Treat None as missing, and widen the batch handler so one malformed snapshot is skipped rather than aborting. Separately, the never-terminating-`in` hazard belongs on TrinityOutput.Snapshot itself (define __contains__), not only in a comment inside the CLOUDY exporter.",
    "failure_scenario": "One snapshot in an --all export has t_now: null. float(None) raises TypeError out of main(); the output directory holds the decks written so far, no manifest.json and no linelist. Separately, any caller elsewhere in the codebase writing `if 'key' in snapshot` hangs forever with no traceback -- a hung sweep worker or SLURM job rather than a failure.",
    "repro": "python3 -c \"snap={'t_now':None}; M=object(); print([k for k in ('t_now',) if snap.get(k,M) is M]); float(snap['t_now'])\" ; then grep -rn \"in snap\" trinity/ and check whether TrinityOutput.Snapshot defines __contains__/__iter__.",
    "confidence": "high"
  },
  {
    "id": "S13b-R-17",
    "file": "trinity/_output/cloudy/dlaw.py",
    "line": 174,
    "class": "coefficient",
    "severity": "S4",
    "claim": "The pc->cm log offset is recomputed independently at two sites, and the radius and density offsets derive from two separately-defined constants in unit_conversions that must remain exactly 3x consistent, with no test asserting it. The +18.4894 / -55.4681 comments are each correctly rounded; the residual is 4.8e-5 dex and physically irrelevant.",
    "evidence": "MAINTENANCE AXIS ONLY. Lens B flagged that 3 x 18.4894 = 55.4682 != 55.4681 and worried the code hardcodes two rounded literals -- Lens A CONTRADICTS that: dlaw.py:174 `log_pc_per_cm = math.log10(INV_CONV.pc2cm)` and :175 `log_ndens_offset = math.log10(INV_CONV.ndens_au2cgs)` compute both at runtime, and A verified 3*log10(pc2cm) + log10(ndens_au2cgs) == 0.0 EXACTLY in float64. True values: log10(pc/cm) = 18.4893505, -3x = -55.4680516; both comments round correctly. The real duplication is that `math.log10(INV_CONV.pc2cm)` is computed at dlaw.py:174 AND snapshot_to_deck.py:180, and the local name is inverted (pc2cm is cm-per-pc, so log_pc_per_cm holds log10(cm/pc)).",
    "expected": "One definition, named for what it holds, plus a test asserting 3*log10(pc2cm) + log10(ndens_au2cgs) == 0 so the two constants cannot drift apart.",
    "failure_scenario": "No numerical error today. A future edit to one site (switching to a different length constant, or to a cm2pc with a sign flip) leaves the deck's LOG_RIN/LOG_ROUT on one length scale and the dlaw table on another, with nothing comparing them -- and the resulting deck still runs. The inverted name makes exactly that edit more likely.",
    "repro": "python3 -c \"import math; L=math.log10(3.0856775814913674e18); print(repr(L), repr(3*L + math.log10(1/2.937998946096347e+55)))\"  # 18.489350545222138 0.0",
    "confidence": "high"
  },
  {
    "id": "S13b-R-18",
    "file": "trinity/_output/cloudy/dlaw.py",
    "line": 58,
    "class": "deadcode",
    "severity": "S4",
    "claim": "build_dlaw_block takes a dens_profile argument its own docstring calls unused and its body never reads, yet the whole chain plumbs it and run_loader hard-refuses to load any run whose dens_profile is outside a hand-maintained mirror of _input/registry.py.",
    "evidence": "CORROBORATED. Lens A (code): dlaw.py:58 `dens_profile: str = 'densPL',` with no other occurrence in the file; snapshot_to_deck.py:254 computes and passes it; run_loader.py:36 `VALID_DENS_PROFILES = frozenset({'densBE','densPL'})` and :94-99 raise RunLoadError otherwise. Lens B (prose): dlaw.py:64 'dens_profile -- TRINITY profile shape; reserved for future PCHIP-on-densBE support. Currently unused; densification is linear-in-(log r, log n).'; run_loader.py:35 '# Canonical TRINITY density-profile enum (mirrors _validate_dens_profile in trinity/_input/registry.py).'",
    "expected": "Per the repo's simplicity rule, no speculative parameters. Flagged only, no deletion proposed (pre-existing). If the gate stays, import the canonical set rather than mirroring it, or add a test asserting the two sets are equal.",
    "failure_scenario": "Adding a density profile in _input/registry.py without updating the mirror makes every run using it unloadable by the exporter -- RunLoadError on a valid run, discovered only at export time, with a message implying the profile matters to the deck when inside this slice it does not.",
    "repro": "grep -n dens_profile trinity/_output/cloudy/dlaw.py  # only the signature line ; then diff run_loader.py's frozenset against trinity/_input/registry.py::_validate_dens_profile",
    "confidence": "high"
  },
  {
    "id": "S13b-R-19",
    "file": "trinity/_output/cloudy/trinity_to_cloudy.py",
    "line": 304,
    "class": "other",
    "severity": "S4",
    "claim": "The default invocation writes a deck CLOUDY cannot parse: the <<<EDIT_ME>>> SB99 sentinel is substituted successfully, the unsubstituted-placeholder guard only recognises {{NAME}}, and only a printed TODO stands between the user and a failing run. The comment's stated reason for the sentinel's survival is also wrong.",
    "evidence": "DOCUMENTED BY DESIGN, with a wrong rationale. Lens B (prose): :6 'The <<<EDIT_ME>>> sentinel in the deck's table star line MUST be replaced by hand'; :293 'Sentinels not matching the {{KEY}} pattern (notably <<<EDIT_ME>>>) pass through unchanged'; :456 'Closing-summary TODO printed only when the SB99 sentinel is in the deck'; but :76 attributes the survival to 'Word-boundary match', which cannot be the reason since <<<EDIT_ME>>> contains no braces at all. Lens A (code): :74 `DEFAULT_SB99_SENTINEL = '<<<EDIT_ME>>>'`; :78 `PLACEHOLDER_RE = re.compile(r'\\{\\{(\\w+)\\}\\}')`; :304-308 the leftover scan finds nothing; A also notes render_template re-scans its OWN output, so a substituted value containing {{...}} would raise a spurious error.",
    "expected": "If the guard exists to prevent shipping an unrunnable deck, it should recognise the sentinel the tool itself inserts. Fix the comment to say the sentinel survives because it has no braces, not because of word boundaries.",
    "failure_scenario": "Default invocation writes <prefix>.in containing '<<<EDIT_ME>>>' on the table star line, prints WROTE plus a TODO, exits 0. A maintainer loosening the placeholder pattern on the strength of the word-boundary comment could preserve the wrong property and cause the sentinel to be consumed, shipping a deck with a bogus atmosphere grid name.",
    "repro": "python3 -c \"import re; P=re.compile(r'\\\\{\\\\{(\\\\w+)\\\\}\\\\}'); print(P.findall('table star <<<EDIT_ME>>>'))\"  # -> []",
    "confidence": "high"
  },
  {
    "id": "S13b-R-20",
    "file": "trinity/_output/cloudy/snapshot_to_deck.py",
    "line": 154,
    "class": "numerical",
    "severity": "S4",
    "claim": "Validation step 6 describes one comparison two incompatible ways in a single sentence (rel_tol=1e-12 AND 'exact-equality drift'), and the dlaw bracket check's tolerance is documented only as 'tiny float tolerance' with no value. The code is consistent: both are rel_tol 1e-12.",
    "evidence": "Lens B (prose): snapshot_to_deck.py:59 step 6 'shell_r_arr endpoints match R2 / rShell (rel_tol=1e-12) -- simplify preserves them by contract; an exact-equality drift would indicate upstream regression'; dlaw.py:161 '# --- 4. Bracket check (with tiny float tolerance)'. Lens A (code) resolves the value: snapshot_to_deck.py:155,160 `math.isclose(..., rel_tol=1e-12)` with no abs_tol (safe only because R2 > 0 and rShell > R2 are checked first); dlaw.py:162-171 uses rel_tol = 1e-12 for the bracket, and A notes the default r_out_pc = rShell case passes exactly. B's fear that the tolerance might be too tight for the common path is therefore not borne out.",
    "expected": "State which test is performed and print the bracket tolerance value. Note that the default case sits exactly on the boundary, so the tolerance is load-bearing rather than merely defensive.",
    "failure_scenario": "A maintainer tightening the check to exact equality on the strength of the second clause starts rejecting valid snapshots; one loosening it on the strength of the first lets a real upstream regression through. Also, math.isclose with rel_tol and no abs_tol is unsafe near zero -- currently masked by the positivity checks that run first.",
    "repro": "Read snapshot_to_deck.py:155,160 and dlaw.py:162-171; confirm both tolerances are 1e-12 and that no abs_tol is supplied.",
    "confidence": "medium"
  },
  {
    "id": "S13b-R-21",
    "file": "trinity/_output/cloudy/trinity_to_cloudy.py",
    "line": 419,
    "class": "silent-failure",
    "severity": "S4",
    "claim": "main() returns 0 on the batch path regardless of outcome, including when every snapshot was skipped and when no snapshot was picked at all.",
    "evidence": "Lens A (code): :396-410 appends a 'skipped' record and continues; :415-416 writes the manifest; :419 `return 0` with no inspection of records; :496-500 prints 'Converted {ok} snapshots ({skipped} skipped).' An empty picks list yields records == [], a manifest.json of [], and 'Converted 0 snapshots (0 skipped).' Lens B: no prose addresses the exit code at all.",
    "expected": "Non-zero exit when nothing was successfully converted, so shell/SLURM wrappers can detect it.",
    "failure_scenario": "An automated sweep runs the exporter over many runs; a run where every snapshot fails validation still exits 0, writes a manifest full of skipped records and a linelist, and the wrapper records success. The missing decks are noticed only when the CLOUDY stage finds no input.",
    "repro": "Read trinity_to_cloudy.py:347-419: `records` is never consulted for the return value.",
    "confidence": "high"
  },
  {
    "id": "S13b-R-22",
    "file": "trinity/_output/cloudy/trinity_to_cloudy.py",
    "line": 125,
    "class": "deadcode",
    "severity": "S4",
    "claim": "Three contracts in this slice have no consumer: --abundances is parsed and never read, snapshot_to_values(extend_with_ambient=False) is unreachable from the CLI, and DLAW_ROWS has no in-slice consumer.",
    "evidence": "Lens A (code): :125-128 `parser.add_argument('--abundances', ...)` with args.abundances appearing nowhere else; snapshot_to_deck.py:57 `extend_with_ambient: bool = True` never passed by main() (:350-358), making the raise at :217-224 unreachable from the CLI; snapshot_to_deck.py:258,276 `'DLAW_ROWS': dlaw_rows_only` with no in-slice reader. Lens B never mentions --abundances at all in its CLI transcription -- an undocumented option as well as an unimplemented one.",
    "expected": "n/a -- flagged per project rule, no deletion proposed.",
    "failure_scenario": "A user passes --abundances and believes the deck's abundance directive changed; it did not, and no warning is issued. DLAW_ROWS's status is unresolvable from this slice: if the out-of-slice template interpolates both DLAW_BLOCK and DLAW_ROWS, the dlaw table is emitted twice.",
    "repro": "grep -rn 'abundances\\|extend_with_ambient\\|DLAW_ROWS' trinity/_output/cloudy/ ; then grep DLAW_ROWS and DLAW_BLOCK in trinity/_output/cloudy/trinity2cloudy.in_template",
    "confidence": "high"
  },
  {
    "id": "S13b-R-23",
    "file": "trinity/_output/cloudy/snapshot_to_deck.py",
    "line": 256,
    "class": "other",
    "severity": "S4",
    "claim": "DLAW_ROWS is built by stripping exactly one leading and one trailing line, an invariant that contradicts build_dlaw_block's documented caller-overridable open/close strings. Latent only: no in-slice caller overrides them and both defaults are single-line.",
    "evidence": "Lens B (prose): snapshot_to_deck.py:256 '# Rows-only view: strip the first (header) and last (footer) lines.' versus dlaw.py:64 'dlaw_open, dlaw_row_prefix, dlaw_close -- CLOUDY syntax knobs' and dlaw.py:31 'Override at call site if a live smoke test reveals a different working form.' Lens A (code) clears it for today: the defaults at dlaw.py:33-35 are single-line strings and the parameters at :59-61 are never overridden by any caller in the slice.",
    "expected": "Either document 'dlaw_open and dlaw_close must be single-line' on build_dlaw_block, or have it return the rows separately instead of reconstructing them by line-slicing.",
    "failure_scenario": "The module explicitly anticipates overriding the syntax after a smoke test. A two-line open would leave a stray dlaw command inside DLAW_ROWS and silently drop the first density row from any template using DLAW_ROWS -- the exact scenario the configurability exists to support.",
    "repro": "Set dlaw_open to a two-line string and inspect DLAW_ROWS.",
    "confidence": "medium"
  },
  {
    "id": "S13b-R-24",
    "file": "trinity/_output/cloudy/run_loader.py",
    "line": 189,
    "class": "units",
    "severity": "S4",
    "claim": "run_loader's unit documentation is self-inconsistent in two ways: the stated suffix rule ('units in the key name where they differ from AU = Msun, pc, Myr') is contradicted by its own key list, and the slice carries two density conventions (pc^-3 arrays, cm^-3 legacy scalars) with no documented bridge.",
    "evidence": "Lens B only (Lens A had no comments and did not examine key naming). run_loader.py:189 lists t_now_myr, R2_pc, mCloud_msun, rCloud_pc, rCore_pc -- all CONFORMING to the declared AU convention yet suffixed -- alongside shell_nMax_cm3, shell_v_kms, nCore_cm3, nISM_cm3 which genuinely differ. Versus dlaw.py:1 '(r [pc], log10 n [pc^-3])' and snapshot_to_deck.py:59 'initial_cloud_{r,n}_arr (linear, in pc / pc^-3)'. MITIGATION from Lens A: A's dataflow trace shows the ambient arrays come from bundle.output.initial_cloud_profile(), not from any *_cm3 scalar, so no in-slice path mixes the two conventions today.",
    "expected": "Drop the 'where they differ' qualifier (all physical keys are suffixed), and add one note stating that scalar diagnostics are cgs-flavoured while profile arrays are pc-based.",
    "failure_scenario": "Documentation-only today. The hazard is future: a key added without a suffix would be read as AU by the stated rule, and the rule is demonstrably not followed; and anyone splicing an nISM_cm3 scalar into the pc^-3 ambient array would misplace the density by 55.468 dex, which the dlaw finite/NaN checks would not catch.",
    "repro": "Read run_loader.py:189's docstring key list against its own stated rule; grep for any code path feeding a *_cm3 scalar into the dlaw ambient arrays.",
    "confidence": "high"
  },
  {
    "id": "S13b-R-25",
    "file": "trinity/_output/cloudy/dlaw.py",
    "line": 125,
    "class": "silent-failure",
    "severity": "S2",
    "claim": "The shell profile is sorted and de-duplicated with no warning: a non-monotonic input is silently reordered, and an exactly-duplicated radius silently loses its inner density. The mechanism is documented; the silence is not.",
    "evidence": "Lens A (code): dlaw.py:125-131 `order = np.argsort(r_pc, kind='stable')` / `keep[:-1] = r_pc[:-1] != r_pc[1:]` / `r_pc = r_pc[keep]` / `log_n_pc3 = log_n_pc3[keep]` -- no warning on either operation; A also notes the caller's endpoint guarantee (snapshot_to_deck.py:155-164) is checked on the PRE-sort array, so a reordered profile still passes it. Lens B (prose) documents the mechanism but not the silence: dlaw.py:124 '# --- 2. Sort and dedup adjacent duplicates (keep last)' and :152 '# dedup ambient (keep last value at each unique r), same recipe as shell'.",
    "expected": "Warn when reordering or discarding sampled physics points; the endpoint guarantee should be evaluated against the array actually written.",
    "failure_scenario": "A shell profile representing a contact discontinuity as two points at one radius with different densities silently keeps only the outer density -- the jump vanishes from the exported deck. A profile arriving non-monotonic from an upstream bug is silently straightened into a plausible-looking, physically different table that passes every check and exports cleanly.",
    "repro": "python3 -c \"import numpy as np; r=np.array([1.,2.,2.,3.]); n=np.array([5.,6.,7.,8.]); k=np.ones(4,bool); k[:-1]=r[:-1]!=r[1:]; print(r[k], n[k])\"  # -> [1. 2. 3.] [5. 7. 8.]: n=6 dropped, no warning",
    "confidence": "high"
  },
  {
    "id": "S13b-R-26",
    "file": "trinity/_output/cloudy/dlaw.py",
    "line": 1,
    "class": "other",
    "severity": "S4",
    "claim": "Multiple documented behaviours and two scheduled removals are pinned to an unnamed external plan document ('Step 0', 'Step 4', 'Step 5 smoke test', 'Option B', 'Phase 5', 'Phase 6', 'pre-Phase-2'), so none can be checked or scheduled from the code alone.",
    "evidence": "Lens B only (A had no comments to see these in). dlaw.py:1 'best-guess for CLOUDY C17/C22; see Step 5 smoke test'; dlaw.py:31 '(Step 0 / Option B)'; snapshot_to_deck.py:1 'The CLI (Step 4) calls this'; run_loader.py:1 'Legacy runs (pre-Phase-5) ... they will be removed in Phase 6' (repeated at :155 and :189); run_loader.py:124 'this fallback is scheduled for removal once existing runs are re-processed'.",
    "expected": "Name the document (a path under docs/dev/ or an issue number) or restate the content inline. Per the repo's own convention, docs/dev/ write-ups are unverified point-in-time analyses, so an unnamed pointer is doubly unresolvable.",
    "failure_scenario": "The deprecation removals and the CLOUDY-syntax verification both become permanently unactionable: nobody can tell whether Phase 6 happened or whether the smoke test was ever run, so the legacy text parsers (which R-08 shows are actively divergent) and the guessed syntax persist indefinitely.",
    "repro": "grep -rn 'Step 5\\|Phase 6\\|Option B' docs/dev/ and look for a CLOUDY-export plan; check whether Phase 5/6 shipped.",
    "confidence": "high"
  },
  {
    "id": "S13b-R-27",
    "file": "trinity/_output/cloudy/run_loader.py",
    "line": 155,
    "class": "other",
    "severity": "S4",
    "claim": "The documented scalar-coercion type list disagrees with the inline comment describing the same step: the docstring omits dicts, the comment includes them. Unresolved -- Lens A did not enumerate the branches.",
    "evidence": "Lens B only. run_loader.py:155 'Values are coerced (in order): bool, None, nan/inf, int, float, Python-literal (lists, tuples), else string.' versus :317 '# Python literal (lists, tuples, dicts) -- must start with a recognisable literal opener, otherwise we'd accept arbitrary expressions.' Lens A describes _coerce_scalar's terminal `return s` at :324 for anything unparseable but does not say which literal openers are accepted -- so this is 'A didn't look', not 'A looked and it isn't there'.",
    "expected": "One list, in one place.",
    "failure_scenario": "A legacy summary value written as a dict literal either parses (contradicting the docstring) or falls through to str (contradicting the comment); a caller relying on the documented type set gets a str where it expected a mapping. Combined with R-08, the string fall-through is the mechanism that turns a parse ambiguity into a downstream float() crash.",
    "repro": "python3 -c \"import run_loader; print(type(run_loader._coerce_scalar('{\\\"a\\\": 1}')))\"  # or read the literal-opener check at run_loader.py:317",
    "confidence": "medium"
  }
]
```
