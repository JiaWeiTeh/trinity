# S12a input config — reconciled (A vs B)

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

## What I read, and what that limits

I read exactly two files:

- `scratchpad/raw/S12a_input_config_lensA.md` — **Lens A**, 40 findings, written from a
  comment- and docstring-stripped copy of the seven slice files. A saw only code. A declares one
  exception: it read the real `trinity/_functions/unit_conversions.py` to pin `convert2au()` and
  `ndens_au2cgs`.
- `scratchpad/raw/S12a_input_config_lensB.md` — **Lens B**, 32 findings, written from a prose-only
  extract (comments and docstrings, tagged with file and line). B saw no code at all: no bodies, no
  signatures, no literals except those the prose itself spells out.

**I did not read any source.** Not `trinity/_input/*`, not the stripped `code/` copies, not
`prose.md`, not `signatures.md`, not `test/`, not `param/`, not `docs/dev/`. Every file path and line
number below is inherited from a lens, not verified. That is deliberate: the value of this pass is
the diff between two independent accounts, and a third opinion from me would collapse it.

What that limits, concretely:

1. **No finding here is verified.** Where the two lenses agree I report high confidence *in the
   agreement*, which is strong evidence but not proof — both lenses could be reading the same line
   wrongly, and both were told which files to read, so their blind spots overlap.
2. **A cannot report intent; B cannot report behaviour.** When A calls something a defect, it may be
   documented as deliberate somewhere A could not see. When B calls something a promise, the code may
   already honour it. Every "doc-drift" verdict below is a claim about a mismatch between two
   *reports*, one of which was blind to intent and one to behaviour.
3. **`trinity/_input/default.param` is outside both lenses.** A flags this explicitly as its
   structural blind spot; B never had it either. This is the largest hole in the slice, because
   `default.param` is the file that actually supplies every input parameter's value, unit and info
   string. Several findings below — including the S1 — are conditional on what that file contains.
4. **`trinity._output.run_constants` (`RUN_CONST_KEYS`, `METADATA_EXCLUDE`), `_metadata_io`,
   `simulation_end`, `show_run`, `_functions.simplify` and `sps.sps_columns` are all outside the
   slice.** Findings that depend on them carry medium confidence and say so.
5. **Line numbers may be stale relative to HEAD.** Both lenses worked from a snapshot; the project's
   own CLAUDE.md warns that paths and line numbers drift fast.

Axes used: **A ≡ B** corroborated (ranked highest — the strongest signal this method produces) ·
**A ≠ B** doc-drift · **A-only** undocumented behaviour · **B-only** unimplemented or stale claim ·
**A refutes B** (B asserted something A's code account contradicts).

---

## Correspondence table

Every substantive claim from either lens, merged. 72 raw findings → 46 reconciled.

| ID | Reconciled claim | Lenses | Axis | Verdict |
|---|---|---|---|---|
| R-01 | `mu_convert`/`mu_atom`/`mu_ion`/`mu_mol` are schema-accepted input keys whose user values are silently overwritten in place | A-04 + B-02 | **A ≡ B** | **Upheld, S1.** B predicted the exact scenario from docs alone; A observed the four assignments. Conditional on mu_* being in `default.param`. |
| R-02 | Anti-stomp guard compares object identity, so it is blind to in-place `.value` writes and to `pop` | A-03 + B-25 + B §1 | **A ≡ B** | Upheld, S2. B supplies the guard's documented purpose and the shipped bug that motivated it; A shows it covers only half the failure modes. |
| R-03 | `apply_active_when` pops a user-set key with no warning | A-05 + B §5 | A-only consequence on a B-documented mechanism | Upheld, S2. The pop is documented as intended; the silent discard of *user* input is not. |
| R-04 | Every input key is declared twice (`default.param` vs `registry.SPECS`); `ParamSpec.info/.unit/.default` are dead for declared keys; nothing cross-checks | A-01 + B-19 + B-02 + B §6 | **A ≡ B** | Upheld, S3. Both lenses independently found duplicated declaration authority from opposite ends. |
| R-05 | `materialize_runtime` runs last, so injected keys skip validate/resolve/convert; registry input defaults are raw source strings | A-02 + B-19 | **A ≡ B** | Upheld, S2. B: the parser is future work ("Phase 10's builder parses it"). A: injection already deepcopies the unparsed string. |
| R-06 | Docs claim exactly three resolvers; code has at least four | B-01 + B-17 + A-27 | **A settles B** | Upheld, S3. A's evidence puts `resolver=resolve_fkappa_auto` on `registry.py:387`. The docs are stale, not the module. |
| R-07 | `mCloud/(1-sfe)` back-out warning is algebraically false given the code | B-11 + A §5.7 | **A settles B** | Upheld, S3. A's `read_param.py:386-389` implements exactly the two relations B quoted; the identity is exact. |
| R-08 | f_kappa censoring sentinel and measured `64.0` are the same double; a right-censored bound is interpolated as a point estimate | A-25 + B-15 (+B-13,14,16) | **A ≡ B** | Upheld, S2. Reached from opposite directions. Raises A's own medium → high. |
| R-09 | `shell_grav_force_m` written as log10\|·\| under an unprefixed key; sign discarded | A-15 | A-only | Upheld, S2. B transcribed no `log_` naming convention — there is no documented rule, which makes the naming pattern the only signal, and it is broken. |
| R-10 | `DescribedDict.__init__` arms atexit + SIGINT/SIGTERM; `load_snapshot` constructs one, so read-only analysis writes into the analysed run | A-18 | A-only | Upheld, S2. B documents the handlers' *coverage limits* and "must never break the run" — scoped to the writing process, silent on the reading one. |
| R-11 | Dedup keyed on exact `(t_now, R2)`, skipped if either absent, **and** disabled at every flush boundary | A-17 + B-09 | **A ≡ B**, A stronger | Upheld, S2. A adds the `previous_snapshot = {}` reset that voids the guard once per interval. |
| R-12 | Metadata path skips non-serializable values with a warning; snapshot path is unguarded and `_safe_flush` swallows the failure | A-20 + B-08 | **A ≡ B**, A stronger | Upheld, S2. A adds the asymmetry and the data loss. |
| R-13 | `METADATA_EXCLUDE` profile-array carve-out is a hand-maintained per-key chain; docs record a shipped regression of exactly this kind | B-07 + A §8.4 | **A ≡ B** (mechanism) | Upheld, S2, medium — depends on `run_constants`, outside slice. |
| R-14 | `bubble_T_arr` and `bubble_n_arr` independently decimated onto different r-grids | A-16 | A-only | Upheld, S2. Recoverable via companion `_r_arr` keys, so ranked below R-09. |
| R-15 | Log transform floors at `eps=1e-300`, so negative n/T is silently written as −300 | A-39 | A-only | Upheld, S2. |
| R-16 | `_validate_ZCloud` pins Z=1, making the *documented* ZCloud=0.15 Sutherland–Dopita path and two other branches dead | A-06 + B-23 + B §4 | **A refutes B** | Upheld, S3. B's worry ("ZCloud=0.5 silently gets solar cooling") is refuted — it raises. The real defect is a documented capability the validator forbids. B's validator table has no ZCloud row: the pin is undocumented. |
| R-17 | Docs state a closed `{1,2,3}` CIE preset; code has no validator, no `else`, and `int()`s a path into a bare `ValueError` | A-07 + B §3 | **A ≡ B** | Upheld, **demoted S2 → S3** (see severity section). |
| R-18 | `_resolve_path_cooling_nonCIE` mkdirs an input directory (documented) and appends `os.sep` only on the default branch (undocumented) | A-11 + B §3 | mixed | Upheld, S2 — driven by the separator asymmetry, not the mkdir. |
| R-19 | `cooling_boost_kappa` has a resolver but no validator; any non-`auto` string passes through verbatim | A-27 + B §4 | **A ≡ B** | Upheld, S2. B documents sibling `cooling_boost_fA`'s strict contract; A shows kappa has none. |
| R-20 | `fkappa_fire` log10s three inputs unguarded: `sfe=0` → −inf → clipped to 0.03 with a warning; negative → nan → opaque bounds error | A-26 | A-only | Upheld, S2, medium. |
| R-21 | `sfe` and `mCloud` carry no validator, while `coverFraction` and `rCloud_max` carry exactly this shape | A-28 + B §4 | **A ≡ B** (inventory gap) | Upheld, S2. B's validator inventory is the control: `sfe` is absent from it. |
| R-22 | `TShell_ion` out-of-band is warn-only; `caseB_alpha` pinned and not adjusted | A-40 + B-24 | **A ≡ B** | Upheld, S2. A supplies the 8000–11000 K gate B says is unstated. |
| R-23 | `parse_value` accepts `nan`/`inf`/`1e400`→inf and turns `1/0` into the string `'1/0'` | A-08 | A-only | Upheld, S2. The documented precedence chain itself is accurate (B §2 ≡ A §2.1) — a corroborated *correctness*. |
| R-24 | Malformed line silently skipped in `default.param`, hard error in the user file, against a docstring promising robust line-numbered errors | A-10 + B-12 | **A ≡ B** | Upheld, S3. A supplies the asymmetry; B supplies the contradicted promise. |
| R-25 | Duplicate keys silently last-wins in both files | A-09 | A-only | Upheld, S3. |
| R-26 | No quoting/escaping: `#` truncates values in both parsers | A-36 | A-only | Upheld, S3. B documents "inline comments are stripped" without the consequence for path-valued keys. |
| R-27 | `time_varying_keys` contradicts `run_const`; `COOLING_PHASE_KEYS` is a third grouping of the same triple | A-14 + B §8 + B-26 | **A ≡ B** | Upheld, S3, medium (depends on `run_constants`). |
| R-28 | Cooling-table paths, CIE selection, SPS file and refmass appear in neither snapshot nor `metadata.json` | A-32 + B §8 | A-only consequence of a B-documented routing model | Upheld, S3, medium. |
| R-29 | `updateDict`'s two branches have opposite unknown-key semantics; both bulk helpers document silent skipping | A-24 + B-10 | **A ≡ B** | Upheld, S3. |
| R-30 | `simplify()` masks every `_simplify` ValueError as a length mismatch; a degenerate array yields nan R², which passes the documented `<0.9` guard | A-21 + B-05 | A-only + **A ≡ B** | Upheld, S3. |
| R-31 | Stale-artifact window: deletion and metadata write happen only on first flush, which a run dying early never reaches | A-19 + B §7 | **A ≡ B** (mechanism), A-only (window) | Upheld, S2, medium. |
| R-32 | `ParamSpec.unit` mixes machine-parseable and free-text dialects; inconsistent spellings of one dimension | A-30 | A-only | Upheld, S4. |
| R-33 | `SENTINEL_PREFIX` is documented as the shared prefix and has zero readers; five literal comparisons instead | A-13 + B §3 | **A ≠ B** | Upheld, S4. |
| R-34 | `sps_col_*` count: **12** in code (A), **13** in docs (B), against a "canonical 7-column" layout (B) | A §9 + B-18 | **A ≠ B** | Upheld, S4, medium. A genuine three-way numeric drift. |
| R-35 | `_resolve_sps_bundle` raises two exception types for one error class and writes two keys it does not own | A-12 | A-only | Upheld, S4. |
| R-36 | `_validate_stop_at_rCloud_nSnap` mutates during validation; docs describe it as "Validate AND coerce" | A-29 + B §4 | **A ≡ B** | Upheld, S4 — documented, so hygiene only. |
| R-37 | `_excluded_keys` only ever grows; exclusion is permanently sticky | A-31 | A-only | Upheld, S4. |
| R-38 | `DescribedItem` defines `__eq__` without `__hash__` (unhashable); `__eq__` on arrays returns an array | A-33 | A-only | Upheld, S4. |
| R-39 | `dictionary.py`'s module description is not a docstring (follows `from __future__`) | A-34 | A-only | Upheld, S4. Explains why B saw it and stripping did not remove it. |
| R-40 | `load_snapshot` round-trip drops `info`/`ori_units`/`exclude_from_snapshot` and coerces lists to ndarray | B-27 | B-only | Upheld as documented, S4. A confirms `load_snapshot` constructs a bare dict; only B saw the metadata loss, only A saw the handler arming. |
| R-41 | `run_const ∩ metadata_exclude is always empty` is asserted with no named test, unlike its neighbours | B-30 | B-only | Upheld, S4 — A found no counterexample, so currently satisfied. |
| R-42 | Doc-drift bundle: phase labels (9 / 8-9 / 7-10 / "not yet wired"), `O(1)` vs `O(pending)`, `UserWarning` vs logger, "200 specs", "106 items" | B-28, B-06, B-05, B §6 | B-only | Upheld, S4. |
| R-43 | Declared-but-unused bundle: `deprecated` category, empty back-compat category, `specs_by_category`, duplicated `_REPO_ROOT`, unused `snap_id`, bare `except:`, empty `__init__.py` | A-35, A-38, A-37, A-22, A-23 + B-29, B-32 | mixed | Upheld, S4. `__init__.py` is the cleanest A ≡ B in the slice: A says "empty", B says "no prose extracted". |
| R-44 | Undocumented/unreachable tunables: `grad_inc`, `nmin` fallback 100, "output size is *normally* nmin" | B-22, B-20, B-21 | B-only | Upheld, S4. A confirms `grad_inc` is a real kwarg passed to `_simplify`; neither lens found a spec for it. |
| R-45 | f_A's cross-knob warning defines "kappa active" as any number, and kappa's default is the number 1.0 | B-04 | B-only, **unadjudicated** | **Demoted to low confidence.** A's account of `_validate_cooling_boost_fA` (registry.py:128-136) shows only the float+positivity check, not the cross-knob warning. Named as an open question. |
| R-46 | Prose records a shipped bug: a `default.param` key replaced by a runtime item, so every run used `include_PHII=True` regardless of the `.param` | B-25 | B-only | Upheld, S3. This is the documentary evidence for the S1 argument below. |

Claims I could not adjudicate and am **not** carrying as findings: B's `SOURCE_TERM_DESIGN.md` bare-filename citation (rolled into R-42); B-31's case-sensitivity half — **refuted by A**, which reports `val_str.lower() == 'none'`, so `'none'` does parse to `None`; B-03's category-listing omission — absorbed into R-04 as supporting evidence.

---

## 1. Silently discarded user input — the slice's worst shape

This is one defect with four moving parts, and the two lenses caught it from opposite ends. Rank it
first.

**What B says the pipeline promises.** B's transcription of the merge block is unambiguous:

> "**Merge/precedence** (`read_param.py:214`, `:233`, `:237`, `:241`, `:244`): user keys must exist in
> `default.param` or it is an error; **user values override defaults**; unset keys take the default;
> **overridden parameters are reported**." — Lens B §2

That is a two-part documented promise: the user's value wins, and the override is *announced*. B also
transcribes the guard that exists to enforce it, and — critically — the bug that motivated it:

> "A key from `default.param` that has been replaced (not just mutated) with a fresh `DescribedItem`
> has lost the user's value — the most recent offender was `include_PHII`, which meant **every run
> integrated with `include_PHII=True` regardless of what the `.param` file said**. Fail loudly so
> this never ships silently again." — `read_param.py:477`, via Lens B (B-25)

**What A says the code does.** A reports four in-place overwrites, 170 lines before the guard:

> ```
> 316	    params['mu_convert'].value = float(_muH)    * _mH_au
> 317	    params['mu_atom'].value    = float(_mu_n)   * _mH_au
> 318	    params['mu_ion'].value     = float(_mu_p)   * _mH_au
> 319	    params['mu_mol'].value     = float(_mu_mol) * _mH_au
> ```
> "`mu_atom`, `mu_ion`, `mu_mol`, `mu_convert` are all declared as settable input parameters
> (`registry.py:363-366`, with `default='14/11'` etc.), so they pass the unknown-key check at
> `:215-225` — and are then discarded. A user who writes `mu_convert 1.5` in their `.param` gets
> `1.4·m_H` and no message of any kind." — Lens A §4.1

And the guard cannot see it, because it compares identity, not value:

> ```
> 482	    _stomped = [
> 483	        k for k, v_before in _default_items_before.items()
> 484	        if k in params and params[k] is not v_before
> 485	    ]
> ```
> "The error text is explicit that the intent is to catch 'runtime init silently overwrote
> user-facing default.param key(s)'. … But four keys are overwritten **by value**, 170 lines
> earlier, and are invisible to it. … `apply_active_when`'s `params.pop` is similarly invisible: the
> guard's `if k in params` clause explicitly skips deleted keys." — Lens A §4.1

**The joint verdict.** The remediation for `include_PHII` was scoped to the flavour of the bug that
had just been found — *replacement* of the item — and not to the flavour that was already present in
the same function, *mutation* of the item's value. The guard's own error message ("remove the
conflicting assignment(s) from Step 6/8/10", per A-03) names Step 6 as a place conflicting
assignments live; Step 6 is exactly where lines 316-319 sit, and the guard is structurally incapable
of seeing them.

The independent arrival is what makes this strong. B never saw line 316. B reasoned purely from a
contradiction between two comments — `param_spec.py:41` lists `mu_*` under "Input-side (declared in
`default.param`)", `read_param.py:302` says `x_He`/`Z_He` are "the **single source of truth** for the
gas composition" — and wrote, as its own failure scenario:

> "A user sets `mu_ion` in their `.param` expecting it to take effect; Step 6 silently overwrites it
> from `x_He`/`Z_He` — **the same class of bug the `include_PHII` guard at `read_param.py:474` was
> added to catch.**" — Lens B, B-02

That is a prediction from documentation alone that A confirmed from code alone. Nothing else in this
slice reaches that standard of evidence.

**A cleared the arithmetic, which sharpens rather than softens the finding.** A checked every
composition formula against the declared defaults and found all seven exact — `mu_H = 7/5`,
`mu_atom = 14/11`, `mu_ion = 14/23`, `mu_mol = 14/6`, `chi_e = 1.2`, `mu_ion_shell = 7/11`,
`chi_e_shell = 1.1` — and praised `Fraction(...).limit_denominator(10**6)` as the right defensive
move. B independently transcribes the same claim as *verified*: "Exact-rational (Fraction) arithmetic
keeps the `mu_*` values byte-identical to the historical 14/11, 14/23, 14/6, 1.4 encodings … (verified)"
(`read_param.py:305`). So the derived values are right. The defect is not a wrong number in the
default configuration; it is that the derivation is unconditional, and the keys it overwrites are
advertised to users as settable.

**One aggravating detail, unverified.** B transcribes a documented override *report* at
`read_param.py:241`/`:244`. A's account of the merge block (`:234-242`) does not mention one. If that
report fires, a user who sets `mu_convert 1.2` is affirmatively told the override was applied, and
then it is discarded — a false confirmation, strictly worse than silence. This is open question Q3.

---

## 2. The two-declaration-site problem

Both lenses found duplicated declaration authority, from opposite ends, and neither could see the
file at the centre of it.

**A, from the code:**

> "Every input parameter is declared **twice**: in `default.param` — value, `# INFO:`, `# UNIT:` —
> which is what `read_param` actually consumes; in `registry.SPECS` — `default=`, `info=`, `unit=` —
> which `read_param` **never consults for a key that exists in `default.param`**. … `ParamSpec.info`
> is dead text. … `ParamSpec.unit` is decorative. … `ParamSpec.default` is dead **unless** the key is
> dropped from `default.param`, at which point it becomes live — and it is the wrong type." — Lens A §3

**B, from the comments:**

> "**Input specs** (`category` starts with `input_`): `default` is the *raw source string* exactly as
> it would appear in `default.param` … **Phase 10's builder parses it** via the same `parse_value`
> path `read_param` uses for file content." — `registry.py:1`, via B-19

Put those side by side and the drift mechanism is fully specified without anyone reading
`default.param`:

1. `read_param` takes value, `info` **and** `unit` from `default.param`'s comment blocks
   (`read_param.py:167` `default_dict[key] = (info, unit, value)`; `:262` `convert2au(unit)`;
   `:270-274` builds the `DescribedItem` from all three). Per A.
2. The registry's copies are consulted only by `materialize_runtime` and `apply_active_when`, both
   guarded by "only if the key is not already present" (`registry.py:654-655`, `:612`). Per A.
3. The registry's copy is a *source string* awaiting a parser that the docs put in the future tense.
   Per B.
4. `tools/gen_default_param.py` is documented as regenerating `default.param` **from** the registry
   ("Phase 3+", `param_spec.py:1`, per B) — so the registry is the nominal upstream — but the
   generated file is hand-editable and A reports **no cross-check in either direction**: "Nothing
   enumerates 'keys in `REGISTRY` but not in `default.param`' or the converse."

The drift is therefore one-way undetectable. An edit to `default.param` changes what runs and leaves
the registry — the source of `run_const`/`metadata_exclude` membership, per B's `registry.py:1` —
describing a different parameter. A's worked case is the unit field: if `default.param` said
`# UNIT: [g]` for `mCloud` while `registry.py:337` says `unit='Msun'`, the value is converted from
grams, the registry is never consulted, and nothing warns. That is ~34 orders of magnitude from a
plausible-looking file.

B's own defaults table (§6) is corroborating evidence at a second level: B flagged, without being
asked to, that `sps_refmass = 1e6` is restated in three places, the R² threshold `0.9` in two by two
different mechanisms, the ceiling `64` in two, the materialization count "106 items" in two, and
`mu_*` in two with *conflicting provenance*. Restating a constant in prose is the same disease as
restating it in a second declaration site, and B counted eight instances of it.

Severity S3, not higher: as long as `default.param` is complete and correct, every registry copy is
inert. The mechanism is a maintenance hazard and the reason R-05 is latent rather than live.

---

## 3. f_kappa censoring — both lenses, opposite directions

Rank this second among corroborated defects. It is the only finding where a physics-bearing number is
built on a documented data-quality shortcut, and both lenses reached it independently.

**B, from the module docstring:**

> "**Censored cells** (the diffuse/high-SFE corner where nothing up to `f_kappa=64` fired) are filled
> with the sweep ceiling 64; a resolved value at that ceiling means the calibration could **NOT**
> demonstrate firing, and the resolver warns accordingly." — `fkappa_auto.py:1`, via B-15
>
> "A right-censored measurement is a lower bound, not a value; trilinear interpolation across a
> censored cell mixes bounds with measurements." — B-15, `expected`

**A, from the table literals, having never seen that docstring:**

> "`_C = F_KAPPA_CEILING = 64.0` is used in six cells (`:55` ×2, `:61`, `:66`, `:67` ×2) while the
> literal `64.0` appears in five others (`:54`, `:60`, `:61`, `:65`, `:66`). … `_C` reads as a
> *censored* cell (the sweep never fired) and the literal `64.0` as a *measured* value. After
> `np.array` construction both are the double 64.0, so (a) the ceiling warning fires identically for
> censored and measured cells, and (b) the interpolant treats a censored cell as a hard datum
> `f_kappa = 64` when interpolating its neighbours — a right-censored bound used as a point estimate.
> **This is an inference about intent from the warning string, so medium confidence.**" — Lens A §7

B removes A's uncertainty. A's inference — that `_C` marks a censored cell and the bare `64.0` a
measured one — is precisely what the docstring states. **A-25 moves from medium to high confidence.**
And A adds what B could not know: the two encodings are *the same double*, so the distinction the
docstring draws does not survive into the array. The doc describes a design that maintains the
bound/datum distinction; the implementation discards it at construction time.

The consequences compound with B's two other f_kappa findings:

- **The hull is 3×3×7** (`mCloud {1e5,1e6,1e7}` × `sfe {0.03,0.1,0.3}` × seven densities), and
  outside it, coordinates are "clamped to it, with a warning" (B-16). B's sharpest observation is
  self-referential: the module's own negative result is that the sweep "refuted a single-variable
  `f_kappa(n_H)` law (spread up to **32x** across mCloud/sfe at fixed density)" — so the two axes it
  clamps are exactly the two axes with the largest measured spread. A sweep at `mCloud=1e8` inherits
  the `1e7` edge value on an axis the module says cannot be extrapolated.
- **The calibration regime is narrow**: "measured on flat power-law clouds (densPL, alpha=0),
  nISM=0.1, hybr solver. Other profiles resolve on the same table with no measured guarantee (a
  warning is logged)" (B-14). densBE is a first-class profile elsewhere in the code.
- **Censoring sits in the corner most likely to be clamped into.** B locates the censored cells at
  `sfe = 0.3` (2 censored) and the high-density/high-SFE edges — the hull boundary. So a run clamped
  to the `sfe = 0.3` edge lands on or beside censored cells: two approximations stacked, one warning.

**What A cleared, and it is a lot.** This is the part worth stating plainly, because it bounds the
finding:

- The **grid and dimensions are consistent**: `_F_FIRE` shape `(3,3,7)` matches `(M, sfe, n)`.
- The **interpolation choice is right**: linear in `log10(f_kappa)` over log-spaced axes, i.e. a
  piecewise power law — "the natural choice for a quantity spanning 1→64".
- The **unit conversion at the call site is correct**: `params["nCore"].value * cvt.ndens_au2cgs` at
  `:121` returns AU (pc⁻³) to cm⁻³ for the lookup. A: "This is the one place in the slice where a
  unit conversion is applied by hand, and it is right." This directly answers B-13, which flagged the
  contract as asserted in a one-line comment only — the assertion is true. B-13's residual hazard (a
  second caller passing code units would silently clamp to the hull edge) stands, demoted to S4 doc
  hygiene, because `fkappa_fire` is documented as a "pure lookup (no params dict)".
- The **axis semantics are right**: using `mCloud_input` rather than the SFE-reduced `mCloud`
  matches the grid axis label. B independently confirms the intent: "mCloud axis is the PRE-star-formation
  input mass (`mCloud_input`) … not the post-SFE `mCloud`" (`fkappa_auto.py:1`). **A ≡ B on a
  correctness**, which is worth as much as an agreement on a defect.
- **`max(1.0, …)` at `:94` is not dead code**: A verified that `10.0 ** _INTERP(...)` at an exact node
  returns e.g. `32.00000000000001`, so the clamp is a genuine round-trip repair at the f=1 nodes. B
  independently transcribes the contract it enforces: "Returns a float **>= 1**" (`fkappa_auto.py:76`).
- The **ceiling test has adequate margin**: `f_kappa >= 0.999 * F_KAPPA_CEILING` is comfortably
  outside the round-trip error.
- **No spurious clamp warnings**: A reimplemented the interpolator and confirmed all 63 exact grid
  nodes compare equal after `np.clip`.

So the module is arithmetically and dimensionally clean. The defect is entirely in the *encoding of
uncertainty*: a lower bound stored as a value, in a table read by an interpolator that cannot
distinguish them, warned about by a test that fires on both.

The one hazard A found that B could not: `fkappa_fire` takes `log10` of all three inputs unguarded
(R-20). `sfe = 0` — a plausible spelling of a no-star-formation control run — gives `-inf`, is clipped
to `sfe = 0.03`, and the run proceeds on the 3%-efficiency calibration with a numpy RuntimeWarning
and a clamp warning. A negative value gives `nan`, which survives `np.clip`, still fires the
"clamping" warning, and then raises an opaque `RegularGridInterpolator` bounds error naming no
parameter.

---

## 4. The resolver inventory — A settles it

B found the contradiction and could not resolve it:

> "`registry.py:565` and `param_spec.py:120` both say exactly three resolvers exist today, naming
> them; `fkappa_auto.py:98` declares itself 'Registry resolver for `cooling_boost_kappa` (read_param
> Step 7)'. `read_param.py:405` also lists only the three. Whichever is right, one of them is
> stale — and the consequence is not cosmetic: `resolve_all` leans on the three-resolver inventory to
> assert 'no inter-dependencies in that order', a claim that a fourth resolver reading
> `mCloud_input`/`sfe`/`nCore` would invalidate." — Lens B §14

**A settles it.** A-27's evidence reads: "`registry.py:387` declares `cooling_boost_kappa` with
`resolver=resolve_fkappa_auto` and no `validator=`". A was not looking for the inventory question — it
was documenting the missing validator — and reported the `resolver=` kwarg in passing. So the spec
carries a resolver, `resolve_all` iterates `SPECS` (A-29's evidence contrasts
`validate_all (:556-561)` with `resolve_all (:580-585)` `params[spec.name].value = spec.resolver(...)`),
and the fourth resolver runs. **The three-resolver inventory in `registry.py:565`, `param_spec.py:120`
and `read_param.py:405` is stale; `fkappa_auto.py:98` is correct.** S3 doc-drift, high confidence.

**Does the ordering guarantee actually break?** No, not today, and A's account is enough to show why.
`SPECS` is a single literal list spanning `registry.py:329-533` (A-35's evidence pins the range), so
declaration order is iteration order, and A's line numbers place `cooling_boost_kappa` at `:387`
*before* `path_cooling_nonCIE` (`:393`) and `sps_path` (`:394`). Its three inputs —
`mCloud_input`, `sfe`, `nCore` — are all set in Step 6 (`read_param.py:386-400`), which runs before
`resolve_all` at `:410`. And A's inventory of cross-key writes inside resolvers names only
`_resolve_sps_bundle` touching `sps_refmass` and `sps_column_map` — nothing mutates `mCloud_input`,
`sfe` or `nCore`. So the fourth resolver reads values that are already final, and the "no
inter-dependencies" claim holds by accident of ordering rather than by design.

That is B-17's point exactly, and it survives: the guarantee is stated over an inventory that omits
the only resolver with a cross-key *read* dependency, so nothing documents the constraint that keeps
it true. I do not have A's line number for the `path2output` spec, so I cannot print the full ordered
resolver list — open question Q4.

---

## 5. `mCloud_input` algebra — A settles it against the doc

B found two documented statements that cannot both stand:

> "`read_param.py:382` asserts `mCloud_input == mCloud + mCluster` … 'Downstream analysis that wants
> the input value should read `mCloud_input`, **not back out `mCloud / (1 - sfe)`**.' … `registry.py:409`
> gives '`mCluster` = `mCloud_input * sfe`'. Those two make `mCloud / (1 - sfe)` *exactly*
> `mCloud_input`. Yet the same comment warns readers not to back it out that way." — Lens B §14

**A's account of the code implements both relations, literally:**

> ```
> 387	    mCluster = mCloud_input_value * params['sfe'].value
> 388	    mCloud_after_SF = mCloud_input_value - mCluster
> 389	    params['mCloud'].value = mCloud_after_SF
> ```
> — Lens A §5.7, with `mCloud_input_value = params['mCloud'].value` (the pre-SFE value) at `:386`

Substituting: `mCloud = mCloud_input − mCloud_input·sfe = mCloud_input(1 − sfe)`. Therefore
`mCloud/(1−sfe) = mCloud_input` **exactly**, in floating point up to one multiply-and-subtract, and
the stated invariant `mCloud_input == mCloud + mCluster` is exact by construction (it is literally
line 388 rearranged).

**Verdict: the warning at `read_param.py:382` is a stale or over-cautious claim, not a statement about
this code.** S3, misleading. Two honest caveats:

1. `sfe = 1` makes the back-out a division by zero, and `sfe` has no validator (R-21). So the warning
   is *operationally* good advice at one point in the domain, for a reason it does not give.
2. Neither lens can see whether `mCloud` is re-mutated after `read_param` returns. If some later
   module rebinds it, the back-out would be wrong at analysis time even though it is exact at load
   time — and that would make the warning substantively correct and merely unexplained. That is a
   one-step lookup outside the slice: open question Q5.

A also flags what the algebra does not protect: `mCloud` is value-stomped at `:389` and so is
invisible to the identity guard (R-02). A notes "`mCloud` is intended (SFE), but the guard cannot tell
intended from accidental" — the same blindness that lets `mu_*` through is what makes the sanctioned
rebinding possible in the first place.

---

## 6. `shell_grav_force_m` — a broken convention with no documented convention behind it

A's finding, verbatim:

> "**`shell_grav_force_m` → `shell_grav_force_m`** (`:683`) — but the value written is
> `np.log10(np.maximum(np.abs(np.asarray(val)), eps))` (`:680`). Log-transformed and absolute-valued,
> **under the unmodified name**, in a file where every other log-transformed array carries a `log_`
> prefix. A reader that follows the prefix convention will treat log₁₀|F| as a linear force."
> — Lens A §8.4

The four siblings, per A: `bubble_T_arr`/`bubble_n_arr` → `log_*` (`:648`), `bubble_dTdr_arr` →
`log_*` (`:658`), `shell_n_arr` → `log_*` (`:699`); and the genuine control case
`bubble_v_arr` → `bubble_v_arr` (`:667`), which really is linear.

**I checked B for a documented naming convention. There is none.** B transcribes the `dictionary.jsonl`
format in detail — "line-delimited JSON", "Line 0: snapshot \"0\" as JSON object", "All arrays are
inlined as lists (no HDF5)", the three-axis snapshot routing model, the "must NEVER strip" carve-out
for profile arrays — and nowhere states that a `log_` prefix denotes log space, nor that any key is
written log-transformed, nor that `shell_grav_force_m` in particular is. B's inventory of
"must/always/never" statements (§13) contains fourteen entries and not one concerns key naming or
the log transform.

So the honest verdict is **not** "a violated documented convention". It is worse in a specific way:
there is no documentation of the transform at all, for any of the five keys. The `log_` prefix is the
*only* signal a consumer of `dictionary.jsonl` has about which arrays are in log space, it is a
convention learnable only by pattern-matching four keys against a fifth, and the fifth breaks it. A
reader who learns the pattern is actively misled; a reader who does not learn it has no way to know
any of the arrays are logged.

Severity **S2**, and I keep A's rating rather than demoting it to "inconsistency", for three reasons.
The snapshot file is a run's product, not an internal detail — it is what analysis and paper figures
read. The error is silent and large: A's worked case, a true value of `1e-3 pc/Myr²` read as
`-3 pc/Myr²`, is wrong in magnitude and in sign. And the sign is discarded by `np.abs`, so unlike the
mismatched-grid problem (R-14, where companion `_r_arr` keys make the data recoverable) this loss is
unrecoverable from the archive. It does not change numbers *inside* a run, which is why it is not S1.

Related and from the same block, A-39/R-15: the `eps = 1e-300` floor means a negative density or
temperature is written as `log10 = -300` rather than surfacing. A physics failure in the bubble solve
becomes a sharp-but-finite dip in every plot, and the archived snapshot — the only place a post-hoc
analysis could have found it — records nothing anomalous.

---

## 7. Crash handlers in a constructor a read path uses

A's finding (A-18/R-10), condensed:

> "`dictionary.py:240 self._register_crash_handlers()` in `__init__`; `:284 atexit.register(atexit_handler)`;
> `:287-288 signal.signal(SIGINT/SIGTERM, ...)`. … `DescribedDict.load_snapshot` constructs one
> (`:951 params = cls()`) and immediately gives it a `path2output` (`:954`). So a purely read-only
> analysis script that loads a snapshot will, at interpreter exit, run `_safe_flush("Normal exit / atexit")`
> → `write_termination_debug_report(output_dir, …)` and rewrite `metadata_humanreadable.txt`
> (`:325-342`) **into the directory of the run it was analysing**." — Lens A §8.2

Three consequences A draws: `signal.signal` off the main thread raises `ValueError`, so a
thread-based worker cannot construct one; the previous SIGINT handler is replaced rather than chained,
so Ctrl-C exits `128+signum` instead of raising `KeyboardInterrupt`; and every construction adds
another atexit hook, so ten loaded runs means ten hooks writing into ten finished runs' directories.

**Does any B-transcribed claim document this?** B documents the *mechanism* and its coverage limits,
and nothing about the read path:

- `dictionary.py:263` — "Does **NOT** cover: kill -9 (SIGKILL) … `os._exit()`". So the crash-handler
  design is documented, and documented precisely enough to enumerate what it misses. It does not say
  where it is registered, or that construction is the trigger.
- `dictionary.py:334` — "this convenience artefact **must never break the run**". B lists this among
  the slice's "never" statements. Read against A's `:325-342`, the artefact is
  `metadata_humanreadable.txt` — the file A says gets rewritten. The safety property is asserted for
  *the run that writes it*. There is no corresponding property for the run that is merely being read.
  The claim is scoped to the wrong process.
- `dictionary.py:929` — B-27's load-path contract: `load_snapshot` "reconstructs scalars directly into
  `DescribedItem(value)`" and "list values back into numpy arrays". A documented reconstruction
  contract that says nothing about arming signal handlers or setting `path2output`.

So: **A-only, undocumented.** The reading side effect is invisible in the prose, and the one safety
claim in the neighbourhood ("must never break the run") is true of the writer and false of the reader.

Severity **S2**, and I want to be precise about why not higher. It changes no simulation number. What
it does is mutate the artefacts of an already-finished run — overwriting a real termination record
with "Normal exit / atexit" — which makes the *archive* wrong, silently, as a side effect of reading
it. Reading data should not mutate it; among 46 findings this is the only one that destroys existing
data rather than producing or recording a wrong value. If `metadata_humanreadable.txt` and the
termination debug report are load-bearing records (they are what `python -m trinity._output.show_run`
formats, per B), this is S1 for those artefacts. I hold it at S2 only because how destructive the two
writers are cannot be determined from either lens — `write_termination_debug_report` is outside the
slice, and A says so explicitly. Open question Q7.

---

## 8. Severity call: is silently-discarded user input S1 or S2?

Lens A rated **no S1** in the entire slice, on this stated reasoning: it found nothing that changes
numbers "on a nominal successful run of a tracked config". I think that reasoning is sound as a rule
and wrong as applied here, and this is the most consequential call in the slice, so I will argue it
rather than assert it.

**Why the rule fails on this particular slice.** The rule takes the tracked `.param` files as the unit
of concern. For a solver or a residual that is right: the tracked configs exercise the integrator, and
a defect they do not reach is genuinely latent. But S12a *is the parameter reader* — the component
whose entire purpose is to make untracked configurations work. Scoring it against tracked configs is
circular. The tracked configs are, by construction, the subset of inputs that already works; a defect
in the input layer is invisible to them almost by definition. Applied consistently, the rule would
make it impossible for any input-validation defect to be rated S1, no matter how wrong the numbers it
produces, which is the wrong answer for the gate every user parameter passes through.

**Why I am not simply inflating everything to S1.** The rule is protecting something real: an S1 must
be a defect that produces wrong numbers *now*, not one that would if someone did something unusual.
So I need a discriminator sharper than "a user could set this". Mine is:

> **S1** — a user-supplied value is *accepted* by the schema, produces no error and no warning, and a
> *different* value silently drives the physics.
> **S2** — the input is rejected, crashes, warns, requires an exotic value, or is merely dropped
> without a wrong number being computed from it.

That draws the line at whether the run *silently computes with a number the user did not supply*, and
it puts exactly one finding above it.

**The one S1: R-01, the `mu_*` keys.** Every clause holds. The key is declared in the registry as
`category='input_constants'` with a real default (A: `registry.py:363-366`, defaults `'14/11'`,
`'14/23'`, `'14/6'`, `'1.4'`). It passes the unknown-key gate, so the file loads clean (A:
`read_param.py:215-225`). The documented contract says the user's value wins and the override is
reported (B §2). The code then overwrites `.value` in place with no guard on whether the key was
user-set (A: `:316-319`). The guard that exists for this class of bug cannot see it (A-03). And the
overwritten quantity is not peripheral: `mu_convert` is the mean mass per H nucleus in
`rho = mu_convert · n_H` (A §6), so every density-to-mass conversion in the run uses a value the user
explicitly replaced. There is no error, no warning, no output artefact recording the discrepancy, and
— if B's documented override report fires — an affirmative message telling the user it worked.

That is wrong numbers, today, for that user's run, while every tracked config stays green. The S2
escape hatches do not apply: nothing masks it, it is not unreachable (a one-line edit the schema
explicitly accepts), and it does not cancel.

**The precedent settles it.** The repo's own comment records this exact class shipping:
`include_PHII` "meant every run integrated with `include_PHII=True` regardless of what the `.param`
file said" (B-25/R-46). Nobody would call that S2 in retrospect. It was found late, the fix was
"Fail loudly so this never ships silently again", and A shows the fix covers replacement but not
mutation — while four mutations sat in the same function. "No evidence a user has hit it" is not the
same as "masked by a guard"; the include_PHII history is direct evidence that this class ships,
survives, and is discovered after results exist.

**Confidence, honestly stated.** I rate R-01 S1 with **medium** confidence, and the uncertainty is not
about severity but about one factual premise: whether `mu_*` actually appears in `default.param`. Both
lenses say input-side — A from `category='input_constants'` in the registry, B from the
`param_spec.py:41` comment listing `mu_*` under "Input-side (declared in `default.param`)" — but
*neither lens read `default.param`*. A's evidence is a registry category; B's is a comment asserting a
fact about a third file. If `mu_*` is absent from `default.param`, then `mu_convert 1.2` is rejected by
the unknown-key gate with a clear error, and R-01 collapses to an S3 doc-drift about a registry
category that lies. That is a single grep (open question Q1) and it flips the finding. I would rather
publish a conditional S1 with the condition named than a confident S2 that buries it.

**Where the rest of the cluster lands, and why not S1:**

- **R-02, the identity-based guard — S2.** It computes nothing itself. It is the enabling mechanism:
  without it, R-01 would have been caught at load. Latent-but-enabling is exactly S2.
- **R-03, `apply_active_when` popping a user-set key — S2.** The user's value is *deleted*, not
  used-wrongly. On a `densPL` run, `densBE_Omega` is genuinely inapplicable and no number is computed
  from it; the profile the user selected is the one that runs. The defect is the absence of any
  signal that input was dropped — sharpened by B's note that `validate_companions` runs pre-merge
  *specifically* so it "fires only when the user explicitly set the trigger", proving the codebase
  knows how to distinguish user-set from defaulted and does not do it on the pop path. The
  wrong-number tail (a user who sets `densBE_Omega` and forgets `dens_profile densBE`, and silently
  gets the default profile) is real, but there the code did what the file said.
- **R-05, `materialize_runtime` last with unparsed string defaults — S2.** Fully masked by
  `default.param` being complete. Textbook latent. Note the failure mode if the mask lifts:
  `log_console` becomes the *string* `'False'`, which is truthy; `stop_at_rCloud_nSnap` becomes the
  string `'None'`, also truthy, with its validator never run because the key was absent at
  `validate_all` time.
- **R-19, `cooling_boost_kappa atuo` — S2, with a caveat.** A typo'd string becomes the conduction
  multiplier verbatim. Whether that is S1 or S2 depends on what the bubble solver does with a `str`:
  A says "best case it raises a `TypeError` deep in the bubble-structure solve; worst case a
  truthiness or string-formatting path treats it as 'boost enabled'". A crash is S2; a silent
  truthiness path is S1. Unresolvable from this slice — Q8.
- **R-21, unvalidated `sfe` — S2.** `sfe = 1.0` at a sweep grid edge gives `mCloud = 0`, which
  degenerates loudly. `sfe > 1` gives a negative cloud mass that could propagate as a finite
  meaningless number — S1-shaped, but nobody sweeps `sfe > 1`. The corroborated part is the
  *inventory gap*: B's transcription shows `coverFraction` ("must be a number in (0, 1]") and
  `rCloud_max` ("must be a positive number") carry exactly this validator shape, and `sfe` — which
  divides into the two most fundamental masses in the run — does not.
- **R-22, `TShell_ion` — S2.** Here the user's value *is* honoured; what is stale is a coupled
  coefficient (`caseB_alpha`, ~1.6× too large at 2e4 K) that the docs explicitly tell the user to
  adjust, and a warning does fire. Documented + warned = S2. A's caveat is worth keeping: `log_console`
  defaults to `False`, so the warning may only reach the `.log` file.
- **R-08, f_kappa censoring — S2.** Reached from opposite directions and ranked high, but the code
  does what the docs say (fill = 64 is the documented design). The defect is that the design conflates
  a bound with a datum. For anyone publishing `cooling_boost_kappa auto` results in the
  diffuse/high-SFE corner it is effectively S1; as a code/doc mismatch it is not one.

**One demotion, for symmetry.** I moved **R-17** (`path_cooling_CIE`) from A's S2 to **S3**. Both
failure modes are eventually loud: `path_cooling_CIE 4` leaves the value as the float `4.0`, which
cannot be opened as a path, and a path-valued setting dies in `int()` with a bare `ValueError`. That
is a diagnostics defect — an error far from its cause, and the wrong exception type — not a silent
wrong number. It would be S2 if the cooling loader has a silent fallback for an unopenable table,
which is outside both lenses (Q6). Interrogating severity has to cut both ways or the ratings mean
nothing.

---

## Open questions

Each is one lookup. File and exactly what to check.

**Q1 — (blocks the S1) Is `mu_*` actually declared in `trinity/_input/default.param`?**
Grep `default.param` for `mu_convert`, `mu_atom`, `mu_ion`, `mu_mol`. If present → R-01 is S1 as
rated: the keys pass the unknown-key gate and are then overwritten at `read_param.py:316-319`. If
absent → `mu_convert 1.2` is rejected with a clear error, R-01 becomes S3 doc-drift (the registry
category `input_constants` and the `param_spec.py:41` "Input-side (declared in default.param)"
comment both lie), and A-02/R-05 gains a live consequence instead.

**Q2 — Do the `# UNIT:` strings in `default.param` agree with `ParamSpec.unit`?**
For `mCloud`, `nCore`, `nISM`, `rCore`, `caseB_alpha`, `dust_sigma`: compare each `# UNIT:` block in
`trinity/_input/default.param` against the `unit=` kwarg at `registry.py:337`, `:346`, `:347`, `:348`,
`:375`. This is the only way to turn R-04 from a described mechanism into an exhibited mismatch — and
A notes the registry field is not even internally consistent (`'cm**-3'` at `:346` vs `'1/cm**3'` at
`:439`).

**Q3 — Does `read_param` log the override list, and does `mu_convert` appear in it?**
Read `read_param.py:234-248`. B transcribes "overridden parameters are reported" at `:241`/`:244`; A's
account of the same block does not mention a report. If a report fires for `mu_*`, R-01 is worse than
silence — the user gets affirmative confirmation of an override that is then discarded.

**Q4 — Enumerate every spec in `registry.py:329-533` carrying `resolver=`.**
A confirms four (`path2output`, `path_cooling_nonCIE`, `sps_path`, `cooling_boost_kappa` at `:387`) but
never gives the `path2output` spec's line, so the ordered inventory is unpinned. Needed to state
whether `resolve_all`'s "no inter-dependencies in that order" is true of the real list (R-06).

**Q5 — Is `params['mCloud'].value` reassigned anywhere outside `trinity/_input/`?**
Grep `trinity/` for `\['mCloud'\]\.value\s*=`. If nothing outside `read_param.py:389` writes it, the
`mCloud/(1-sfe)` back-out is exact at analysis time and the `read_param.py:382` warning is simply
stale (R-07). If something does, the warning is substantively right and merely unexplained.

**Q6 — What does the cooling-table loader do with a non-path `path_cooling_CIE`?**
Find the consumer of `params['path_cooling_CIE']` outside `trinity/_input/` and check whether an
unopenable value raises or falls back to a default table. Raises → R-17 stays S3. Silent fallback →
R-17 is S2 and the demotion above is wrong.

**Q7 — How destructive is the atexit write?**
Read `trinity._output._metadata_io.write_termination_debug_report` (and the
`metadata_humanreadable.txt` writer at `dictionary.py:335-342`): does it truncate/overwrite an existing
report, or append/skip? Overwrite → R-10 destroys a completed run's termination record and is S1 for
that artefact.

**Q8 — What happens to a `str` `cooling_boost_kappa` in the bubble solver?**
Grep for the consumer of `cooling_boost_kappa` outside `trinity/_input/` and check whether it is used
in arithmetic (→ `TypeError`, R-19 stays S2) or in a truthiness/format context (→ silent wrong
multiplier, R-19 is S1).

**Q9 — Does `_simplify` emit a `warnings.warn`, a `logger.warning`, or both?**
Read `trinity/_functions/simplify.py` for `warnings.warn`. B quotes the docstring promising a
`UserWarning` below R²=0.9; A observed a logger path in the `dictionary.py` wrapper. Both may fire, or
the docstring may be stale (R-42/R-30). Also settle whether the wrapper's `nan` R² bypass (A: `nan < 0.9`
is `False`) means a degenerate array silently takes the good-fit branch.

**Q10 — Count `sps_col_*` specs in `registry.py:395-407`.**
A says twelve, the docs say thirteen, and the bundled layout is described as seven columns (R-34).
Three numbers, one grep.

**Q11 — Does `_clean_for_snapshot` consult `METADATA_EXCLUDE` at all?**
Read `dictionary.py:600-706` for a `METADATA_EXCLUDE` membership test. B quotes `dictionary.py:588`
saying those keys are "stripped defensively"; A's account of the same function reports only
`run_const_keys` (`:631`) and `_excluded_keys` (`:624`) as strip conditions. If A missed it, R-13's
carve-out is live; if the strip is gone, the comment is stale and the regression risk is different.

**Q12 — Does any spec carry both `run_const=True` and `metadata_exclude=True`?**
Grep `registry.py:329-533`. A found no counterexample but did not enumerate; B-30 notes the invariant
has no named test, unlike its three neighbours (R-41).

**Q13 — Where does `grad_inc` get its value?**
Read the `simplify` signature at `dictionary.py:~500` and check for a default or a `params` lookup. B
reports it is named once in prose with no spec, no default, no unit; A confirms it is a real kwarg
passed to `_simplify` (R-44).

**Q14 — Does `_validate_cooling_boost_fA` contain a cross-knob warning, and what is
`cooling_boost_kappa`'s registry default?**
Read `registry.py:118-150`. B-04/R-45 claims the warning treats any number as "kappa active" while the
default is `1.0`, so it would fire on every f_A run; A's account of the same validator shows only the
`float()` + positivity check. This is the one B finding I could neither confirm nor refute.

---

```json
[
  {
    "id": "S12a-R-01",
    "file": "trinity/_input/read_param.py",
    "line": 316,
    "class": "silent-failure",
    "severity": "S1",
    "claim": "mu_convert, mu_atom, mu_ion and mu_mol are schema-accepted input keys whose user-supplied values are unconditionally overwritten in place by values derived from x_He/Z_He, with no error, no warning, and no record in any output artefact — contradicting the documented merge contract that user values override defaults and that overrides are reported.",
    "evidence": "CORROBORATED. Lens A (code): registry.py:363-366 declare mu_atom/mu_ion/mu_mol/mu_convert as category='input_constants' with defaults '14/11','14/23','14/6','1.4', so they pass the unknown-key gate at read_param.py:215-225; read_param.py:316-319 `params['mu_convert'].value = float(_muH) * _mH_au` (and mu_atom/mu_ion/mu_mol) with no guard on whether the key was user-set. Lens B (prose, saw no code): read_param.py:214/:233/:237/:241/:244 document 'user values override defaults' and 'overridden parameters are reported'; param_spec.py:41 lists mu_* under '---- Input-side (declared in default.param) ----' while read_param.py:302 says x_He/Z_He are 'the single source of truth for the gas composition'. B-02's failure scenario, written from comments alone, predicts this exact bug and names it 'the same class of bug the include_PHII guard at read_param.py:474 was added to catch'.",
    "expected": "Either reject mu_* in a user .param (they are derived, not free), or honour the override, or at minimum log that the supplied value was superseded. A guard that compares values rather than object identity (see S12a-R-02) would have caught it.",
    "failure_scenario": "A user studying a helium-poor composition sets `mu_convert 1.2`. The file loads clean, the override may even be reported as applied, and mu_convert is silently reset to 1.4*m_H. Every density-to-mass conversion in the run (rho = mu_convert * n_H) uses a value the user explicitly replaced. Every tracked config stays green.",
    "repro": "Add `mu_convert 1.2` to param/simple_cluster.param and print params['mu_convert'].value after read_param. CONDITIONAL: first grep trinity/_input/default.param for mu_convert — if the key is absent there, the unknown-key gate rejects the line and this finding demotes to S3 doc-drift.",
    "confidence": "medium"
  },
  {
    "id": "S12a-R-02",
    "file": "trinity/_input/read_param.py",
    "line": 482,
    "class": "state",
    "severity": "S2",
    "claim": "The anti-stomp guard compares DescribedItem object identity, so it is structurally blind to in-place `.value` overwrites and to deletions — the two mechanisms by which user input is actually lost in this function. It catches only the item-replacement flavour of the bug that motivated it.",
    "evidence": "CORROBORATED. Lens A (code): read_param.py:286 `_default_items_before = {k: params[k] for k in default_dict if k in params}`; :482-485 `_stomped = [k for k, v_before in _default_items_before.items() if k in params and params[k] is not v_before]` — identity comparison, and the `if k in params` clause skips popped keys. Missed overwrites: :316-319 (mu_*), :367/:369 dust_sigma, :373 model_name, :389 mCloud, :425 path_cooling_CIE, plus registry.py:621 params.pop. Lens B (prose): read_param.py:279 'Later steps (6/8/10) ... must NOT silently replace any of these', and read_param.py:477 records the motivating regression verbatim — include_PHII 'meant every run integrated with include_PHII=True regardless of what the .param file said. Fail loudly so this never ships silently again.' The guard's own error text (per A) says 'remove the conflicting assignment(s) from Step 6/8/10' — Step 6 is where lines 316-319 live.",
    "expected": "Compare values (or a hash) rather than identity, or document in-place `.value` writes as the sanctioned mechanism and enumerate which keys use it.",
    "failure_scenario": "A new derived quantity is added as `params['x_He'].value = ...` instead of as a new key. The guard passes, the user's x_He is discarded, and the run proceeds on the derived value — precisely the scenario the RuntimeError text describes preventing.",
    "repro": "Add `params['nCore'].value = 1.0` anywhere between read_param.py:286 and :482 and observe the guard does not fire.",
    "confidence": "high"
  },
  {
    "id": "S12a-R-03",
    "file": "trinity/_input/registry.py",
    "line": 621,
    "class": "silent-failure",
    "severity": "S2",
    "claim": "apply_active_when deletes a parameter the user explicitly set whenever it belongs to the non-selected density profile, with no warning — even though the codebase demonstrably knows how to distinguish user-set from defaulted keys elsewhere.",
    "evidence": "Lens A (code): registry.py:612-621 `present = spec.name in params` / `elif present and not active: params.pop(spec.name)`; :344 densBE_Omega active_when=_active_densBE, :345 densPL_alpha active_when=_active_densPL; called from read_param.py:441. Since both keys must be in default.param (or users could not set them past the :215-225 gate), every run pops one. Lens B (prose) documents the pop as intended — registry.py:589 asserts the invariant 'the spec is in params iff active_when(params) returns True', present+inactive -> pop — and separately records that validate_companions runs pre-merge so it 'fires only when the user explicitly set the trigger, not when the trigger came from default.param' (registry.py:716). So the user-set/defaulted distinction is available and is not used on the pop path. B also notes validate_companions checks only the opposite direction (trigger -> companion required).",
    "expected": "Warn (or raise) when popping a key that was present in the raw user dict, rather than deleting a user-supplied value silently.",
    "failure_scenario": "A user sweeping both profiles from one template leaves `densBE_Omega 20` in a densPL run: the key is silently dropped. Worse, a user who sets `densBE_Omega 20` and forgets `dens_profile densBE` gets the default profile with no signal that their only profile-specific input was discarded.",
    "repro": "Set both `dens_profile densPL` and `densBE_Omega 20` in a .param and check `'densBE_Omega' in params` after read_param.",
    "confidence": "high"
  },
  {
    "id": "S12a-R-04",
    "file": "trinity/_input/registry.py",
    "line": 337,
    "class": "divergence",
    "severity": "S3",
    "claim": "Every input parameter is declared twice — in default.param (value + '# INFO:' + '# UNIT:') and in registry.SPECS (default/info/unit) — and read_param consumes ONLY the default.param copy for any key default.param declares. ParamSpec.info and .unit are dead for those keys, .default is dead until the key is dropped, and nothing cross-checks the two sites in either direction.",
    "evidence": "CORROBORATED from opposite ends. Lens A (code): read_param.py:167 `default_dict[key] = (info, unit, value)`; :262 `convert2au(unit)`; :270-274 builds DescribedItem from all three fields — all from default.param. registry.py:657 (materialize_runtime) and :615 (apply_active_when) are the only readers of spec.default/info/unit and both are guarded by 'only if not already present' (:654-655, :612). A: 'There is no cross-check in either direction. Nothing enumerates keys in REGISTRY but not in default.param or the converse.' Lens B (prose): registry.py:1 'default is the *raw source string* exactly as it would appear in default.param ... Phase 10's builder parses it'; param_spec.py:1 cites tools/gen_default_param.py as regenerating default.param from the registry (Phase 3+) with no round-trip check; B's defaults table independently flags eight constants restated in two or three places, including mu_* with conflicting provenance.",
    "expected": "One source of truth per key, or an enforced consistency check (names, units, defaults) between SPECS and default.param at load or in test/test_registry.py.",
    "failure_scenario": "default.param's '# UNIT:' for mCloud drifts to [g] while registry.py:337 still says unit='Msun'. read_param converts from grams, the registry is never consulted, nothing warns, and the run is silently off by ~34 orders of magnitude from a plausible-looking .param file. Conversely, a default.param key with no spec gets no validator, no resolver, no run_const/metadata_exclude flag, is marked exclude_from_snapshot by read_param.py:455-457, and vanishes from every output artefact while still steering the run.",
    "repro": "Change a '# UNIT:' line in trinity/_input/default.param and observe params[key].value changes while registry.SPECS is untouched and nothing warns. Then diff the key sets of default.param and registry.SPECS.",
    "confidence": "high"
  },
  {
    "id": "S12a-R-05",
    "file": "trinity/_input/read_param.py",
    "line": 472,
    "class": "state",
    "severity": "S2",
    "claim": "materialize_runtime runs last — after unit conversion, validate_all and resolve_all — so any key injected from the registry is never validated, never resolved and never unit-converted; and the registry's input-category defaults are unparsed source strings whose parser the docs place in future work, including the truthy 'False' and 'None'.",
    "evidence": "CORROBORATED. Lens A (code): conversion at read_param.py:255-274, validate_all at :295, resolve_all at :410, materialize_runtime at :472; registry.py:559-560 and :583-584 both `if spec.name not in params: continue`; registry.py:657 `copy.deepcopy(spec.default)` injects verbatim. Registry input defaults are strings: :334 log_console default='False', :337 mCloud '1e7', :354 stop_at_rCloud_nSnap 'None', :376 gamma_adia '5/3' — while derived/runtime specs use native types (:413 0.0, :415 1.2, :438 np.array([])). Lens B (prose): registry.py:1 'default is the *raw source string* ... Phase 10's builder parses it via the same parse_value path' — i.e. the parse is stated as not yet wired; registry.py:625 and read_param.py:469 both document 'Must run AFTER Step 9', so the ordering is deliberate.",
    "expected": "Either store registry defaults as native post-parse values, or materialize before validation/resolution/conversion so an injected default traverses the same pipeline as a default.param value.",
    "failure_scenario": "A key is dropped from default.param during an edit. materialize_runtime injects the raw string: log_console becomes 'False' (truthy — console logging silently turns ON), stop_at_rCloud_nSnap becomes 'None' (truthy, and its validator at registry.py:164-186 never ran because the key was absent at validate_all time), mCloud becomes '1e7' and the first arithmetic raises a TypeError far from the cause.",
    "repro": "Comment out one input key in trinity/_input/default.param, run `python run.py param/simple_cluster.param`, and inspect type(params[key].value).",
    "confidence": "high"
  },
  {
    "id": "S12a-R-06",
    "file": "trinity/_input/registry.py",
    "line": 565,
    "class": "divergence",
    "severity": "S3",
    "claim": "Three separate docstrings assert that exactly three resolvers exist today and name them; the code carries at least four. resolve_all's ordering guarantee is stated over the stale inventory, omitting the only resolver with a cross-key read dependency.",
    "evidence": "ADJUDICATED — A settles B. Lens B (prose) found the contradiction: registry.py:565 'the three current resolvers (path2output, path_cooling_nonCIE, sps_path) carry no inter-dependencies in that order'; param_spec.py:120 'Three specs carry resolvers today'; read_param.py:405 lists the same three; while fkappa_auto.py:98 declares 'Registry resolver for cooling_boost_kappa (read_param Step 7)' resolving 'against mCloud_input / sfe / nCore'. Lens A (code), reporting in passing while documenting a missing validator: 'registry.py:387 declares cooling_boost_kappa with resolver=resolve_fkappa_auto and no validator='. So the fourth resolver is real and the three-resolver inventory is stale. Ordering is currently safe: SPECS is one literal list spanning registry.py:329-533 (A-35 evidence), :387 precedes :393/:394, and mCloud_input/sfe/nCore are all set in Step 6 (read_param.py:386-400) before resolve_all at :410; A's inventory of in-resolver cross-key writes names only sps_refmass and sps_column_map.",
    "expected": "One consistent resolver inventory naming cooling_boost_kappa, and the ordering guarantee restated over the real list — including that the fkappa resolver reads three params set in Step 6.",
    "failure_scenario": "A maintainer trusting 'three resolvers, no inter-dependencies' reorders SPECS or adds a resolver that mutates nCore or sfe. The fkappa resolver silently resolves 'auto' against pre-resolution values; nothing in the docs signals the dependency, because the resolver is not in the inventory the guarantee is stated over.",
    "repro": "grep -n 'three current resolvers\\|Three specs carry resolvers' trinity/_input/registry.py trinity/_input/param_spec.py; then grep -n 'resolver=' trinity/_input/registry.py and count.",
    "confidence": "high"
  },
  {
    "id": "S12a-R-07",
    "file": "trinity/_input/read_param.py",
    "line": 382,
    "class": "state",
    "severity": "S3",
    "claim": "The comment warning downstream analysis not to back out mCloud_input as mCloud/(1-sfe) is algebraically false for the code as written: the two documented relations make that identity exact.",
    "evidence": "ADJUDICATED — A settles B. Lens B (prose): read_param.py:382 'invariant: mCloud_input == mCloud + mCluster. Downstream analysis that wants the input value should read mCloud_input, not back out mCloud / (1 - sfe).' and registry.py:409 'mCloud_input = input mCloud; mCluster = mCloud_input * sfe'. Lens A (code) implements exactly both: read_param.py:386-389 `mCloud_input_value = params['mCloud'].value` / `mCluster = mCloud_input_value * params['sfe'].value` / `mCloud_after_SF = mCloud_input_value - mCluster` / `params['mCloud'].value = mCloud_after_SF`. Hence mCloud = mCloud_input*(1-sfe), the stated invariant is line 388 rearranged, and mCloud/(1-sfe) == mCloud_input exactly.",
    "expected": "Either drop the warning, or state the real reason for it (sfe=1 divides by zero — and sfe has no validator, see S12a-R-21 — or mCloud is rebound again downstream, which neither lens can see).",
    "failure_scenario": "A reader concludes the invariant is only approximate and drops it from a consistency check; or, reading the warning as evidence that sfe is mutated later, writes defensive code for a condition that does not exist. Either way the doc misdirects rather than protects.",
    "repro": "grep -n 'not back out\\|mCloud_input ==' trinity/_input/read_param.py; grep -n 'mCluster = mCloud_input' trinity/_input/registry.py; then grep trinity/ for other writes to params['mCloud'].value.",
    "confidence": "high"
  },
  {
    "id": "S12a-R-08",
    "file": "trinity/_input/fkappa_auto.py",
    "line": 45,
    "class": "coefficient",
    "severity": "S2",
    "claim": "Right-censored cells of the f_kappa calibration table are filled with the sweep ceiling and stored as the same double as genuinely measured 64.0 entries, so the trilinear interpolant uses a lower bound as a point estimate and the ceiling warning cannot distinguish the two cases. The hull is 3x3x7 and clamps exactly the two axes the module's own result says carry a 32x spread.",
    "evidence": "CORROBORATED from opposite directions. Lens A (code, medium confidence, inferring intent from a warning string): fkappa_auto.py:45-46 `F_KAPPA_CEILING = 64.0` / `_C = F_KAPPA_CEILING`; sentinel cells at :55 x2, :61, :66, :67 x2; literal 64.0 in the same table at :54, :60, :61, :65, :66; ceiling test at :123 `if f_kappa >= 0.999 * F_KAPPA_CEILING`. Lens B (prose, never saw the table) confirms A's inference exactly: fkappa_auto.py:1 'Censored cells (the diffuse/high-SFE corner where nothing up to f_kappa=64 fired) are filled with the sweep ceiling 64; a resolved value at that ceiling means the calibration could NOT demonstrate firing'; :44 'Largest f_kappa the sweep tested; censored cells (never fired) carry it.' B adds the hull extent (mCloud {1e5,1e6,1e7}, sfe {0.03,0.1,0.3}, censored counts at :56/:62/:66/:68), the clamping policy, the 32x-spread negative result, and the regime limit (densPL alpha=0, nISM=0.1, hybr solver only). B's confirmation raises A's medium to high.",
    "expected": "Encode censored cells distinctly (np.nan or a mask array) so the interpolator does not treat a lower bound as a datum and the ceiling warning fires only for censored support. State the unit contract on fkappa_fire's docstring, not only at the call site.",
    "failure_scenario": "A run near the diffuse/high-SFE corner interpolates between a censored cell (true f_kappa unknown, >= 64) and a measured 48. The result is below the 0.999*ceiling test, so no warning fires, and a conduction multiplier with no calibration behind it scales C_thermal for the whole run. Conversely a run landing on a genuinely measured 64.0 gets a warning claiming no tested f_kappa fired.",
    "repro": "grep -n 'censored' trinity/_input/fkappa_auto.py; then resolve_fkappa_auto at mCloud_input=1e5, sfe=0.1, nCore=1e2 (literal 64.0) and at sfe=0.3 (a _C cell) and compare the warnings.",
    "confidence": "high"
  },
  {
    "id": "S12a-R-09",
    "file": "trinity/_input/dictionary.py",
    "line": 683,
    "class": "units",
    "severity": "S2",
    "claim": "shell_grav_force_m is written to dictionary.jsonl as log10(|value|) under its unmodified key name, while all four other log-transformed arrays in the same function are renamed with a 'log_' prefix. The sign is discarded irrecoverably, and no documentation anywhere states that any array is log-transformed.",
    "evidence": "Lens A (code): dictionary.py:678-684 `if key == 'shell_grav_force_m':` / `y_arr = np.log10(np.maximum(np.abs(np.asarray(val)), eps))` / `new_dict[key] = ...`. Contrast :648 `new_dict['log_' + key]` for bubble_T_arr/bubble_n_arr, :658 for bubble_dTdr_arr, :699 for shell_n_arr; and the genuine linear control :667 `new_dict[key]` for bubble_v_arr. Lens B: NO documented naming convention exists. B transcribes the dictionary.jsonl format in detail (line-delimited JSON, line N = snapshot N, arrays inlined as lists), the three-axis snapshot routing model, and a fourteen-item inventory of must/always/never statements — none of which mentions key naming, a 'log_' prefix, or any log transform. So the pattern is the only signal a consumer has, and it is undocumented as well as broken.",
    "expected": "`new_dict['log_' + key]` for a log-transformed array, matching its four siblings — or a linear write if the name is to stay. Document the transform for all five keys.",
    "failure_scenario": "Any reader following the file's own naming pattern treats shell_grav_force_m as a linear force per unit mass. A true value of 1e-3 pc/Myr^2 is read as -3 pc/Myr^2: wrong magnitude, wrong sign, and the sign is unrecoverable from the archive because np.abs was applied before writing.",
    "repro": "Compare the magnitudes of 'shell_grav_force_m' and 'log_bubble_T_arr' in one dictionary.jsonl line.",
    "confidence": "high"
  },
  {
    "id": "S12a-R-10",
    "file": "trinity/_input/dictionary.py",
    "line": 240,
    "class": "state",
    "severity": "S2",
    "claim": "DescribedDict.__init__ registers a process-global atexit hook and replaces the SIGINT/SIGTERM handlers. Because load_snapshot constructs one and immediately sets its path2output, a read-only analysis session writes a termination debug report and rewrites metadata_humanreadable.txt into the ANALYSED run's directory at interpreter exit.",
    "evidence": "Lens A (code): dictionary.py:240 `self._register_crash_handlers()` in __init__; :281-288 `atexit.register(atexit_handler)` / `signal.signal(signal.SIGINT, self._signal_handler)` / same for SIGTERM; :325-342 _safe_flush unconditionally calls write_termination_debug_report(str(output_dir), ...) and writes (Path(output_dir)/'metadata_humanreadable.txt').write_text(...); :951 `params = cls()` and :954 `params['path2output'] = DescribedItem(str(path2output), ...)` inside load_snapshot; :300 the signal handler calls sys.exit(128+signum). Lens B documents the mechanism but never the read-path side effect: dictionary.py:263 enumerates coverage limits ('Does NOT cover: kill -9 (SIGKILL) ... os._exit()'), and dictionary.py:334 asserts 'this convenience artefact must never break the run' — a safety property scoped to the writing process, silent about the reading one. B-27's load_snapshot contract (dictionary.py:929) describes reconstruction only, with no mention of handler registration or path2output.",
    "expected": "Register crash handlers via an explicit opt-in for the run that owns the output directory, not in the constructor that read paths also use; chain or restore the previous signal handlers.",
    "failure_scenario": "An analysis script loads snapshots from ten completed runs to build a figure. At exit, ten atexit hooks fire and each rewrites metadata_humanreadable.txt and a termination debug report reading 'Normal exit / atexit' into a finished run's directory, overwriting its real termination record. Additionally Ctrl-C during analysis exits 130 instead of raising KeyboardInterrupt, and constructing a DescribedDict off the main thread raises ValueError from signal.signal.",
    "repro": "python -c \"from trinity._input.dictionary import DescribedDict; DescribedDict.load_snapshot('outputs/<run>', 0)\" then check the mtime and contents of outputs/<run>/metadata_humanreadable.txt.",
    "confidence": "high"
  },
  {
    "id": "S12a-R-11",
    "file": "trinity/_input/dictionary.py",
    "line": 721,
    "class": "state",
    "severity": "S2",
    "claim": "The duplicate-snapshot guard compares only exact-equal (t_now, R2), is skipped entirely when either key is absent, AND is disabled at every flush boundary because flush() clears previous_snapshot — so a duplicate is written once per snapshot_interval.",
    "evidence": "CORROBORATED, A stronger. Lens B (prose): dictionary.py:712 'Duplicate guard: - If the last saved snapshot has the same t_now and R2, it will not save again.'; :730 '# If t_now/R2 not present, skip duplicate detection'. Lens A (code) adds the defeating mechanism: :721-728 `if self.save_count >= 1 and self.previous_snapshot:` / `last = self.previous_snapshot.get(str(self.save_count - 1), {})` / `if ('t_now' in last and t_now == last['t_now']) and ('R2' in last and r2 == last['R2'])`; :868 `self.previous_snapshot = {}` at the end of flush(); :750-752 flush fires when `save_count % snapshot_interval == 0` with snapshot_interval = 10 (:219). Plus `except KeyError: pass` at :729-731 silently disables dedup for any snapshot missing either key.",
    "expected": "Keep the last emitted (t_now, R2) in a field that survives flush, compare with a relative tolerance, and document why those two quantities uniquely identify simulation state.",
    "failure_scenario": "The solver re-emits an identical state at the same t. Nine times in ten it is suppressed; on the tenth — immediately after a flush — the duplicate line is written. Downstream time-series analysis sees an intermittent zero-length step and a dt-based derivative divides by zero.",
    "repro": "Call save_snapshot() 11 times without changing t_now/R2 and count lines in dictionary.jsonl.",
    "confidence": "high"
  },
  {
    "id": "S12a-R-12",
    "file": "trinity/_input/dictionary.py",
    "line": 861,
    "class": "silent-failure",
    "severity": "S2",
    "claim": "The metadata writer probes each value for JSON-serializability and skips-with-warning, but the snapshot writer has no such guard; a non-serializable value raises inside flush(), and when flush was reached from _safe_flush the exception is swallowed and every pending snapshot is discarded while the process exits 0.",
    "evidence": "CORROBORATED, A stronger. Lens B (prose): dictionary.py:836 'Defensive serialization: if the value can't be JSON-encoded ... log a warning and skip the key rather than crashing the whole flush.' Lens A (code) supplies the asymmetry and the data loss: :840-848 the metadata probe `try: ready = self._to_json_ready_value(item.value); json.dumps(ready, cls=NpEncoder) except (TypeError, ValueError) as e: logger.warning(...); continue`; snapshot path :575 `return val` (unrecognised type passed through) and :861 `json_line = json.dumps(snap_data, cls=NpEncoder)` with no try; swallowed at :317-322 `except Exception as e: logger.error(f'Failed to flush snapshots on exit: {e}')`.",
    "expected": "The same per-key probe on the snapshot path, or re-raise after logging so the process does not exit successfully having dropped data. A missing RUN_CONST key in metadata.json should also be loud, since B (dictionary.py:578) documents run-constants as living ONLY there.",
    "failure_scenario": "A new runtime key holds a scipy interpolator and is not marked exclude_from_snapshot. Every flush raises; during the run it propagates from save_snapshot, but on the atexit path it is caught, up to snapshot_interval snapshots are dropped with one ERROR line, and the process exits 0 leaving a truncated dictionary.jsonl that looks complete.",
    "repro": "params['x'] = DescribedItem(object()); params.save_snapshot(); params.flush().",
    "confidence": "high"
  },
  {
    "id": "S12a-R-13",
    "file": "trinity/_input/dictionary.py",
    "line": 588,
    "class": "silent-failure",
    "severity": "S2",
    "claim": "The prose asserts both that metadata_exclude-flagged profile source keys must NEVER be stripped by the snapshot writer and that METADATA_EXCLUDE keys ARE stripped defensively there; the two are reconciled only by a hand-maintained per-key carve-out, which is the exact failure mode a cited regression already shipped once.",
    "evidence": "CORROBORATED in mechanism. Lens B (prose): dictionary.py:61 'Several of them carry metadata_exclude in the ParamSpec registry ... the snapshot writer must NEVER strip these, or their data is silently lost from dictionary.jsonl (regression fixed in hotfix/metadata-excluding)'; :588 'METADATA_EXCLUDE keys (paths, function tables) are stripped defensively ... but the profile arrays in that set must survive'; :594 'the profile arrays in that set must survive'. Lens A (code) shows the carve-out is a literal per-key if-chain inside _clean_for_snapshot (:577-706): explicit branches at :643 (bubble_T_arr/bubble_n_arr), :655 (bubble_dTdr_arr), :667 (bubble_v_arr), :678 (shell_grav_force_m), :697 (shell_n_arr). A did not report a METADATA_EXCLUDE membership test in that function (only run_const_keys at :631 and _excluded_keys at :624), so whether the defensive strip is still present is unconfirmed.",
    "expected": "Derive the carve-out from the registry (e.g. a 'snapshot_required' axis) rather than from a second literal list that must track METADATA_EXCLUDE by hand.",
    "failure_scenario": "A new metadata_exclude'd profile array is added to the registry but not to the per-key chain in _clean_for_snapshot; its data vanishes from dictionary.jsonl silently — exactly the regression dictionary.py:61 records having already shipped.",
    "repro": "grep -n 'must NEVER strip\\|stripped defensively\\|must survive' trinity/_input/dictionary.py; then read dictionary.py:600-706 for a METADATA_EXCLUDE membership test.",
    "confidence": "medium"
  },
  {
    "id": "S12a-R-14",
    "file": "trinity/_input/dictionary.py",
    "line": 643,
    "class": "numerical",
    "severity": "S2",
    "claim": "bubble_T_arr and bubble_n_arr are decimated by two independent simplify() calls, so within one snapshot they land on different radius grids and possibly different lengths; bubble_r_arr itself is dropped and reappears as four separately-simplified copies.",
    "evidence": "Lens A (code): dictionary.py:643-650 `if key in ('bubble_T_arr','bubble_n_arr'):` / `x_arr = np.asarray(self['bubble_r_arr'].value)` / `new_r, new_y = self.simplify(x_arr, y_arr, keyname=key)` / `new_dict['log_'+key] = ...` / `new_dict[key+'_r_arr'] = ...` — simplify is y-dependent, so the retained subset differs per array; bubble_r_arr dropped at :639-641. Lens B: silent on this; B's simplify contract (dictionary.py:457) documents feature-preserving downsampling of y(x) with no statement about consistency across sibling arrays sharing an x-grid.",
    "expected": "One shared decimation of the r-grid applied to all bubble profile arrays, or an explicit statement that the companion _r_arr keys are mandatory for interpretation.",
    "failure_scenario": "An analysis zips log_bubble_T_arr with log_bubble_n_arr to build an (n, T) phase diagram of the bubble interior. The two are sampled at different radii and may differ in length, so every pair is mismatched — silently, since both arrays look well-formed. Recoverable via the companion _r_arr keys, which is why this ranks below S12a-R-09.",
    "repro": "Load a snapshot and compare bubble_T_arr_r_arr with bubble_n_arr_r_arr element-wise.",
    "confidence": "high"
  },
  {
    "id": "S12a-R-15",
    "file": "trinity/_input/dictionary.py",
    "line": 645,
    "class": "numerical",
    "severity": "S2",
    "claim": "The snapshot log-transform floors values at eps = 1e-300 via np.maximum, so a negative density or temperature is silently written as log10 = -300 instead of surfacing as an error or NaN.",
    "evidence": "Lens A (code): dictionary.py:620 `eps = 1e-300`; :645 `y_arr = np.log10(np.maximum(np.asarray(val), eps))` for bubble_T_arr/bubble_n_arr; :697 the same for shell_n_arr; :655 and :680 `np.log10(np.maximum(np.abs(...), eps))`, which additionally discard the sign. Lens B: silent — no documented statement about non-positive guards in the snapshot writer.",
    "expected": "np.maximum is the right guard for underflow to exactly zero, but a negative temperature or density is a physics failure and should be detected, not floored.",
    "failure_scenario": "A bubble-structure solve returns a small negative density in one zone. The snapshot records log10 n = -300 there; every plot shows a sharp but finite dip; the sign error is invisible in the archived output, which is the only place a post-hoc analysis could have found it.",
    "repro": "params['bubble_n_arr'].value = np.array([-1.0, 1.0]); save_snapshot(); inspect log_bubble_n_arr.",
    "confidence": "high"
  },
  {
    "id": "S12a-R-16",
    "file": "trinity/_input/registry.py",
    "line": 99,
    "class": "deadcode",
    "severity": "S3",
    "claim": "_validate_ZCloud hard-pins ZCloud == 1, which makes the DOCUMENTED ZCloud=0.15 Sutherland-Dopita CIE path unreachable, orphans two further metallicity branches, and turns the dust_sigma metallicity scaling into a guaranteed multiply by 1.0. The pin itself is documented nowhere.",
    "evidence": "A REFUTES B, and the refutation is the finding. Lens B (prose) documents the 0.15 behaviour as live: read_param.py:412 'Integer-index preset {1, 2, 3} (under ZCloud == 1) selects between the bundled CIE tables; ZCloud == 0.15 auto-pins to the Sutherland-Dopita file'; B-23 worried that ZCloud=0.5 might 'silently get solar-metallicity cooling'. B's validator inventory (§4: cooling_boost_fA, cooling_boost_kappa, cooling_boost_mode, betadelta solver, stop_at_rCloud_nSnap, coverFraction, rCloud_max) contains NO ZCloud entry. Lens A (code): registry.py:99-105 `def _validate_ZCloud(value, params): if value != 1: raise ParameterFileError(...)`, wired at :339 and executed at read_param.py:295 — so ZCloud=0.5 raises, not silently mis-cools, and ZCloud=0.15 cannot be reached. Dead as a result: read_param.py:426-429 the 0.15 elif; registry.py:277-282 `if params['ZCloud'].value != 1.0: raise ValueError(...)` inside _resolve_sps_bundle; read_param.py:367 `dust_sigma * ZCloud` is always *1.0, with the else at :368-369 reachable only via dust_noZ > 1 (default 0.05).",
    "expected": "Either the Z != 1 support these branches assume, or documentation that the metallicity paths are staged for a future capability — and a comment on the validator itself, which is currently the least documented and most restrictive gate in the slice.",
    "failure_scenario": "A user reads read_param.py:412 (or generated docs from it), sets ZCloud 0.15 expecting the Sutherland-Dopita table, and gets a ParameterFileError. Conversely someone relaxing _validate_ZCloud assumes the branch chain works: read_param.py:417 `if == 1:` / :426 `elif == 0.15:` has no else, so ZCloud=0.2 leaves path_cooling_CIE as a raw float selector with no error.",
    "repro": "Set `ZCloud 0.15` in a .param — it raises at validate_all, proving the elif at read_param.py:426 is unreachable.",
    "confidence": "high"
  },
  {
    "id": "S12a-R-17",
    "file": "trinity/_input/read_param.py",
    "line": 423,
    "class": "silent-failure",
    "severity": "S3",
    "claim": "The docs state a closed integer preset {1,2,3} for path_cooling_CIE, but the code has no validator, no resolver and no else branch: an out-of-range selector is silently left as a bare float, and a path-valued setting raises a bare ValueError from int() rather than a ParameterFileError.",
    "evidence": "CORROBORATED. Lens B (prose): read_param.py:412 'NOT a def_* sentinel: an integer-index preset keyed on ZCloud, so it stays inline ... Integer-index preset {1, 2, 3}'. Lens A (code): read_param.py:418-425 `cie_files = {1: ..., 2: ..., 3: ...}` / `cie_choice = int(params['path_cooling_CIE'].value)` / `if cie_choice in cie_files: params['path_cooling_CIE'].value = str(_REPO_ROOT / cie_files[cie_choice])` — no else; registry.py:392 declares the spec with neither validator nor resolver, unlike sibling path_cooling_nonCIE (:393, resolver=_resolve_path_cooling_nonCIE).",
    "expected": "A validator restricting the value to {1,2,3}, or a resolver mirroring path_cooling_nonCIE, raising ParameterFileError otherwise.",
    "failure_scenario": "`path_cooling_CIE 4` loads without complaint and the value stays 4.0; the cooling loader is handed a number where it expects a path. `path_cooling_CIE /my/tables/cie.dat` dies with 'invalid literal for int() with base 10' naming no parameter. DEMOTED from Lens A's S2 to S3 because both modes are eventually loud — a float cannot be opened as a path. Becomes S2 if the loader has a silent fallback for an unopenable table (open question Q6).",
    "repro": "Set `path_cooling_CIE 4` (silent) or `path_cooling_CIE /tmp/x.dat` (raw ValueError) in a .param.",
    "confidence": "high"
  },
  {
    "id": "S12a-R-18",
    "file": "trinity/_input/registry.py",
    "line": 245,
    "class": "other",
    "severity": "S2",
    "claim": "_resolve_path_cooling_nonCIE creates the directory it is meant to READ cooling cubes from, masking a typo'd path, and appends os.sep only on the default branch so user-set and default values carry different trailing-separator conventions.",
    "evidence": "Lens A (code): registry.py:245-249 `if value == 'def_dir': return str(_REPO_ROOT / 'lib' / 'default' / 'opiate') + os.sep` / `path_cooling = str(value)` / `Path(path_cooling).mkdir(parents=True, exist_ok=True)` / `return path_cooling`. Lens B (prose) documents the mkdir as intended for both resolvers: registry.py:242 for nonCIE ('the shipped OPIATE cube folder under lib/default/opiate/'; 'user path as-is and created') and registry.py:230 for path2output ('Either way the directory is created'). B says nothing about the separator asymmetry — 'user path as-is' documents it without flagging it. So the mkdir is a documented design choice; the separator asymmetry is A-only and undocumented, and it is what drives this severity.",
    "expected": "For an input directory: check existence and raise ParameterFileError if missing. Return the same trailing-separator form on both branches.",
    "failure_scenario": "`path_cooling_nonCIE /data/opiat` (typo) creates an empty /data/opiat and the run fails much later with a confusing 'no cooling cubes' error. Separately, downstream string concatenation `path + filename` works for the default (trailing sep present) and silently builds '/data/opiatefilename' for a user-set path — a wrong path with no error at the point of construction.",
    "repro": "Set path_cooling_nonCIE to a non-existent directory and observe it is created rather than rejected; then compare the returned string's last character for the default and user branches.",
    "confidence": "high"
  },
  {
    "id": "S12a-R-19",
    "file": "trinity/_input/fkappa_auto.py",
    "line": 104,
    "class": "silent-failure",
    "severity": "S2",
    "claim": "cooling_boost_kappa has a resolver but no validator, so any string other than 'auto' is returned verbatim and becomes the conduction multiplier — while its sibling knob cooling_boost_fA is strictly validated for type and positivity.",
    "evidence": "CORROBORATED. Lens A (code): fkappa_auto.py:104-105 `if not (isinstance(value, str) and value.strip().lower() == 'auto'): return value`; registry.py:387 declares cooling_boost_kappa with resolver=resolve_fkappa_auto and no validator=; registry.py:388 declares cooling_boost_fA with validator=_validate_cooling_boost_fA, which at :128-136 does `try: fA = float(value) except (TypeError, ValueError): raise ParameterFileError(...)` and `if not (fA > 0): raise ParameterFileError(...)`. Lens B (prose) confirms the asymmetry from the other side: registry.py:118 documents f_A's contract in detail ('f_A > 0 required', cross-knob warning) and describes cooling_boost_kappa only as 'still its raw value here -- a number or the string auto'; fkappa_auto.py:98 'Numeric values pass through UNTOUCHED'.",
    "expected": "A validator on cooling_boost_kappa accepting either the literal 'auto' or a float >= 1, mirroring cooling_boost_fA.",
    "failure_scenario": "`cooling_boost_kappa atuo` (typo) loads without error and the string 'atuo' becomes the conduction multiplier. If the consumer does arithmetic it raises a TypeError deep in the bubble-structure solve (S2); if a truthiness or string-formatting path treats it as 'boost enabled', the run silently uses an undefined multiplier (S1-shaped). Which one applies is outside both lenses — open question Q8.",
    "repro": "Set `cooling_boost_kappa atuo` in a .param and print params['cooling_boost_kappa'].value after read_param.",
    "confidence": "high"
  },
  {
    "id": "S12a-R-20",
    "file": "trinity/_input/fkappa_auto.py",
    "line": 83,
    "class": "silent-failure",
    "severity": "S2",
    "claim": "fkappa_fire takes log10 of its three inputs without a positivity check: zero or negative mCloud_input/sfe/nCore produce -inf or nan, which np.clip silently folds onto the grid hull (warning only) or turns into an opaque RegularGridInterpolator bounds error naming no parameter.",
    "evidence": "Lens A (code, verified by standalone reimplementation): fkappa_auto.py:83-85 `coords = np.log10([mCloud_input, sfe, nCore])` / `clamped = np.clip(coords, lo, hi)` / `if not np.array_equal(coords, clamped): logger.warning(...)`; :72 constructs the interpolator with default bounds_error=True and no fill_value; :94 `return max(1.0, float(10.0 ** _INTERP(clamped)[0]))`. A confirmed all 63 exact grid nodes compare equal after the clip, so there are no spurious clamp warnings at nominal points. Lens B documents the clamping policy ('Coordinates outside the calibrated hull are clamped to it, with a warning', fkappa_auto.py:1) but nothing about non-positive inputs.",
    "expected": "Validate that all three inputs are finite and > 0 before the log, raising ParameterFileError otherwise.",
    "failure_scenario": "sfe = 0 — a legitimate spelling of a no-star-formation control run — becomes log10(0) = -inf, is clipped to sfe = 0.03, and the run silently uses the f_kappa calibrated for 3% efficiency while a numpy RuntimeWarning scrolls past. A negative value yields nan, which survives np.clip, still fires the 'clamping to the hull' warning, then raises 'One of the requested xi is out of bounds' with no mention of which parameter.",
    "repro": "resolve_fkappa_auto with sfe=0 and cooling_boost_kappa auto.",
    "confidence": "medium"
  },
  {
    "id": "S12a-R-21",
    "file": "trinity/_input/registry.py",
    "line": 338,
    "class": "silent-failure",
    "severity": "S2",
    "claim": "sfe has no validator, so sfe >= 1 silently produces a zero or negative post-star-formation cloud mass; mCloud is likewise unchecked for positivity or finiteness. Two sibling parameters carry exactly this shape of validator, so the gap is an inventory inconsistency, not a missing feature.",
    "evidence": "CORROBORATED as an inventory gap. Lens A (code): registry.py:338 `ParamSpec(name='sfe', default='0.01', info='Star formation efficiency.', category='input_physical', unit=None, exclude_from_snapshot=True, run_const=True)` — no validator; read_param.py:386-389 `mCluster = mCloud_input_value * params['sfe'].value` / `mCloud_after_SF = mCloud_input_value - mCluster` / `params['mCloud'].value = mCloud_after_SF`. Lens B (prose) supplies the control: registry.py:190 coverFraction 'must be a number in (0, 1]' and registry.py:205 rCloud_max 'must be a positive number [pc]' — the same bounded-scalar check, documented, for two less fundamental quantities. B's validator inventory has no sfe or mCloud entry.",
    "expected": "A validator enforcing 0 < sfe < 1 and mCloud > 0 and finite, matching the shape already used for coverFraction and rCloud_max.",
    "failure_scenario": "A sweep generates sfe = 1.0 at a grid edge. mCloud becomes exactly 0 and the cloud-radius solve divides by zero or returns nan, surfacing far from the parameter that caused it; sfe = 1.0 also makes the mCloud/(1-sfe) back-out divide by zero (S12a-R-07). sfe > 1 gives a negative cloud mass that may propagate as a physically meaningless but finite number.",
    "repro": "Set `sfe 1.0` in a .param and print params['mCloud'].value after read_param.",
    "confidence": "high"
  },
  {
    "id": "S12a-R-22",
    "file": "trinity/_input/read_param.py",
    "line": 356,
    "class": "coefficient",
    "severity": "S2",
    "claim": "An out-of-band TShell_ion produces only a log warning: caseB_alpha stays pinned at its ~1e4 K value, no caller receives a programmatic signal, and the Stroemgren balance and HII pressure/force are left internally inconsistent by the code's own account.",
    "evidence": "CORROBORATED, each lens supplying what the other lacked. Lens B (prose): read_param.py:350 'caseB_alpha (default 2.59e-13 cm^3/s) is fixed at its ~1e4 K value and is NOT recomputed from TShell_ion. Since alpha_B(T) ~ T^-0.7, moving the ionised-shell temperature far from ~1e4 K leaves the Stroemgren balance (n_IF_Str) and P_HII/F_HII internally inconsistent unless caseB_alpha is adjusted to match. Warn once at load.' B flagged 'far from ~1e4 K' as having no stated trigger. Lens A (code) supplies the trigger: read_param.py:355-363 `if not (8000.0 <= _T_shell_ion <= 1.1e4): logger.warning(...)`, and registry.py:375 caseB_alpha default='2.59e-13' with no validator and no coupling to TShell_ion.",
    "expected": "Either derive caseB_alpha from TShell_ion via the T^-0.7 scaling the warning message already states, or make the mismatch a hard error unless the user also sets caseB_alpha explicitly. Document the 8000-11000 K band.",
    "failure_scenario": "A user sets TShell_ion 2e4 for a hotter HII region. alpha_B stays ~1.6x too large for that temperature, so n_IF_Str, P_HII and F_HII are all biased with no trace in any output artefact. The warning may only reach the .log file, since log_console defaults to False (registry.py:334).",
    "repro": "Set `TShell_ion 2e4` and confirm read_param returns normally with caseB_alpha unchanged.",
    "confidence": "high"
  },
  {
    "id": "S12a-R-23",
    "file": "trinity/_input/read_param.py",
    "line": 91,
    "class": "silent-failure",
    "severity": "S2",
    "claim": "parse_value silently accepts nan, inf and overflow-to-inf as numeric parameter values, and silently converts a divide-by-zero fraction into a string. The documented precedence chain itself is accurate.",
    "evidence": "Lens A (code, verified by standalone reimplementation): read_param.py:91-100 `try: return float(val_str) except ValueError: pass` then `try: return float(Fraction(val_str)) except (ValueError, ZeroDivisionError): pass`, falling through to :103 `return val_str`. Observed: 'nan'->nan, 'inf'->inf, '1e400'->inf, '1_000'->1000.0, '3'->3.0, '1/0'->the string '1/0'. Lens B (prose) documents precedence 'None -> boolean -> number -> fraction -> string' (read_param.py:73) with per-branch comments at :80/:84/:90/:96/:102 — which matches A's observed order exactly, a corroborated correctness. B says nothing about non-finite rejection. B-31 speculated the None branch might be case-sensitive, inverting cooling_boost_mode's 'none' default; A REFUTES this — the check is `val_str.lower() == 'none'`, so sentinel words are case-insensitive.",
    "expected": "Reject non-finite numeric literals at the trust boundary, and treat '1/0' as malformed rather than as a string parameter.",
    "failure_scenario": "`mCloud 1e400` becomes inf; mCluster = inf*sfe = inf; mCloud_after_SF = inf-inf = nan at read_param.py:388, and the run proceeds with a NaN cloud mass. `gamma_adia 1/0` becomes the string '1/0' and the first arithmetic use raises a TypeError with no reference to the parameter file.",
    "repro": "Set `mCloud 1e400` in a .param; separately set `gamma_adia 1/0` and print type(params['gamma_adia'].value).",
    "confidence": "high"
  },
  {
    "id": "S12a-R-24",
    "file": "trinity/_input/read_param.py",
    "line": 158,
    "class": "divergence",
    "severity": "S3",
    "claim": "The same malformed line is silently skipped in default.param but is a hard ParameterFileError in the user file — two grammars for identical input — and the silent branch contradicts the module docstring's promise of robust, line-numbered error handling.",
    "evidence": "CORROBORATED. Lens A (code): read_param.py:156-159 (default file) `parts = line.split(None, 1)` / `if len(parts) != 2: continue`; :196-202 (user file) same split / `raise ParameterFileError(f\"{Path(path2file).name}, line {line_num}: Expected format 'key value', got: '{line}'\")`. Lens B (prose): read_param.py:159 '# Skip malformed lines in default.param' against read_param.py:3 'Key features: ... - Robust error handling with line numbers and helpful messages'.",
    "expected": "One shared line-parsing routine, or at least the same diagnosis for the same defect. A valueless key in default.param is a shipped-file bug and should be loud.",
    "failure_scenario": "An edit leaves a bare key in default.param. The key silently vanishes from default_dict, so any user .param that sets it now fails the unknown-key check at :215-225 with 'Invalid parameter(s)' — a message pointing at the user's file instead of at the shipped schema. Alternatively the key falls through to materialize_runtime and is injected as an unparsed string (S12a-R-05).",
    "repro": "Delete the value (keeping the key) from any line in trinity/_input/default.param, then set that key in a user .param and read the resulting error.",
    "confidence": "high"
  },
  {
    "id": "S12a-R-25",
    "file": "trinity/_input/read_param.py",
    "line": 206,
    "class": "silent-failure",
    "severity": "S3",
    "claim": "Duplicate keys are silently last-wins in both the user file and default.param; there is no duplicate detection anywhere, despite line numbers being in scope.",
    "evidence": "Lens A (code): read_param.py:206 `user_dict[key] = value` inside the per-line loop and :167 `default_dict[key] = (info, unit, value)` — plain dict assignment, no membership check, with line_num already in scope at :184. Lens B: silent — no documented statement about duplicate keys in either parser.",
    "expected": "Raise ParameterFileError (or warn) on a repeated key, naming both line numbers.",
    "failure_scenario": "A user appends a corrected `sfe 0.1` at the bottom of a file that already sets `sfe 0.01` near the top, or the same happens after a merge conflict. The file loads cleanly, the run uses whichever line came last, and nothing indicates two values were supplied.",
    "repro": "Put `sfe 0.01` and `sfe 0.1` in the same .param and observe params['sfe'].value == 0.1 with no message.",
    "confidence": "high"
  },
  {
    "id": "S12a-R-26",
    "file": "trinity/_input/read_param.py",
    "line": 187,
    "class": "silent-failure",
    "severity": "S3",
    "claim": "The .param grammar has no quoting or escaping, so a '#' inside a value silently truncates it in both parsers — including for path- and label-valued parameters.",
    "evidence": "Lens A (code): read_param.py:186-187 (user file) `if '#' in line: line = line[:line.find('#')]`; :131-133/:147 (default file) `comment_pos = line.find('#')` / `before_comment = line[:comment_pos].strip()`. Both parsers use `line.split(None, 1)`, so values may contain spaces but not '#'. Lens B (prose) documents the stripping as a feature — 'Inline comments are stripped', with the module docstring example 'mCloud 1e6 # cloud mass' (read_param.py:3) — and never states that '#' is therefore illegal in a value.",
    "expected": "Either document that '#' cannot appear in a value, or support quoting. path2output, sps_path, model_name and transition_trigger are all plausible carriers.",
    "failure_scenario": "`path2output /scratch/run#3` silently resolves to /scratch/run, and _resolve_path2output creates that directory. Two sweep members writing to run#3 and run#4 both land in /scratch/run and overwrite each other's dictionary.jsonl.",
    "repro": "Set `model_name run#3` in a .param and print params['model_name'].value.",
    "confidence": "high"
  },
  {
    "id": "S12a-R-27",
    "file": "trinity/_input/read_param.py",
    "line": 449,
    "class": "divergence",
    "severity": "S3",
    "claim": "Three independent mechanisms group the same keys inconsistently: the time_varying_keys allow-list, the registry's run_const flags, and COOLING_PHASE_KEYS. Seven of time_varying_keys' ten entries are run_const=True and are stripped from every snapshot line anyway, leaving only cool_alpha/cool_beta/cool_delta with any effect — and COOLING_PHASE_KEYS lists two of those three.",
    "evidence": "CORROBORATED. Lens A (code): read_param.py:449-457 `time_varying_keys = ['model_name','mCloud','cool_alpha','cool_beta','cool_delta','nCore','nISM','rCore','dens_profile','densPL_alpha']` / `for key, val in params.items(): if key not in time_varying_keys: val.exclude_from_snapshot = True`; run_const=True on registry.py:329 model_name, :337 mCloud, :342 dens_profile, :345 densPL_alpha, :346 nCore, :347 nISM, :348 rCore; dictionary.py:631 `if key in run_const_keys: continue`. A also notes COOLING_PHASE_KEYS (dictionary.py:1180-1226) lists cool_beta and cool_delta but not cool_alpha. Lens B (prose) supplies the intended model: param_spec.py:99 'The three axes are independent' and read_param.py:447 'Only track time-varying quantities in snapshots / Exclude initial conditions and constants to save memory' with a carve-out for 'Cloud profile constants — needed for radial profile reconstruction' (:451) — which is precisely the seven run_const keys, so the carve-out is documented but ineffective.",
    "expected": "One mechanism deciding what lands in a snapshot line. If mCloud/nCore genuinely vary in time they must not be run_const; if they are run constants they do not belong in a 'time varying' list.",
    "failure_scenario": "Someone adds time-dependence to mCloud (mass loading) and relies on time_varying_keys to record it per snapshot. It is silently absent from dictionary.jsonl because run_const=True routes it to a single metadata.json entry, and the analysis reads a constant.",
    "repro": "Inspect a snapshot line in dictionary.jsonl for 'nCore' and 'mCloud'. Depends on trinity._output.run_constants.RUN_CONST_KEYS mirroring registry.run_const_keys(), which is outside this slice.",
    "confidence": "medium"
  },
  {
    "id": "S12a-R-28",
    "file": "trinity/_input/registry.py",
    "line": 392,
    "class": "other",
    "severity": "S3",
    "claim": "The resolved cooling-table paths, the CIE curve selection, the SPS file and its reference mass are excluded from BOTH the snapshot lines and metadata.json, so a run's outputs do not record which physics tables produced them.",
    "evidence": "Lens A (code): registry.py:392 path_cooling_CIE `exclude_from_snapshot=True, metadata_exclude=True` with run_const unset; :393 path_cooling_nonCIE the same; :394 sps_path exclude_from_snapshot=True and not run_const; :357 sps_refmass exclude_from_snapshot=True, not run_const. dictionary.py:828 `for k in RUN_CONST_KEYS:` is the only source of metadata.json entries; :624 `if key in self._excluded_keys: continue` drops them from snapshots. Lens B (prose) documents the routing model that produces this: param_spec.py:99 run_const -> metadata.json, metadata_exclude -> blocked from metadata ('paths, loaded tables, empty array placeholders'), exclude_from_snapshot -> omitted from the jsonl stream, 'The three axes are independent'. So exclusion from both artefacts is a documented consequence of two independent flags, not an oversight anyone declared.",
    "expected": "Record resolved table and SPS provenance in metadata.json — that is what a run-constant is for — even if the raw sentinel values are excluded.",
    "failure_scenario": "Two runs are compared six months apart. One used coolingCIE_3_Gnat-Ferland2012.dat and the bundled 1e6cluster SPS file, the other a user-supplied SPS file with a different sps_refmass. Neither output records which, so the discrepancy is unattributable without the original .param files.",
    "repro": "grep for 'sps_path' or 'path_cooling_CIE' in outputs/<run>/metadata.json and dictionary.jsonl. Depends on trinity._output.run_constants, outside this slice.",
    "confidence": "medium"
  },
  {
    "id": "S12a-R-29",
    "file": "trinity/_input/dictionary.py",
    "line": 1268,
    "class": "divergence",
    "severity": "S3",
    "claim": "updateDict has two branches with opposite failure semantics for the same mistake — the dataclass branch silently skips a key not in the dict, the sequence branch raises KeyError — and the silent behaviour is documented as intended in two separate helpers.",
    "evidence": "CORROBORATED. Lens A (code): dictionary.py:1264-1269 `for field in dataclasses.fields(keys_or_dataclass): key = field.name; val = getattr(...); if key in dictionary: dictionary[key].value = val`; :1277-1278 `for key, val in zip(keys, values): dictionary[key].value = val`. Lens B (prose): dictionary.py:1233 'When using dataclass mode, only fields that exist in the dictionary are updated. Missing keys are silently skipped.'; dictionary.py:982 (reset_keys) 'Keys that don't exist in the dictionary are silently skipped.'",
    "expected": "One policy for an unknown key across both branches, plus at minimum a debug log naming skipped keys.",
    "failure_scenario": "A dataclass field is renamed (e.g. an SPS feedback field) without a matching registry spec. The dataclass path silently stops writing it and params keeps a stale value from the previous timestep for the rest of the run; the same rename through the sequence path would have raised immediately. A typo in a COOLING_PHASE_KEYS entry is likewise a no-op.",
    "repro": "Pass a dataclass with a field name not present in params and observe no error and no update.",
    "confidence": "high"
  },
  {
    "id": "S12a-R-30",
    "file": "trinity/_input/dictionary.py",
    "line": 510,
    "class": "silent-failure",
    "severity": "S3",
    "claim": "simplify() converts every ValueError from _simplify into a fixed 'x and y must have same length' message without chaining, discarding the real cause; and a degenerate (constant) array yields nan R2, which passes the documented R2 < 0.9 quality guard because nan < 0.9 is False.",
    "evidence": "Lens A (code): dictionary.py:508-514 `try: x_out, y_out = _simplify(x_arr, y_arr, nmin=nmin, grad_inc=grad_inc) except ValueError: raise ValueError(f'simplify(): x and y must have same length for {keyname}. Instead got {len(x_arr)} and {len(y_arr)}')` — no `from err`. A also notes _simplify_error is computed on every call (:524) though logged only when r_squared < 0.9 or during the first two implicit-phase calls (:528-536), on a hot snapshot path, and that a constant array gives r_squared = nan which takes the good-fit branch. Lens B (prose) supplies the guard being bypassed: dictionary.py:457 'computes the linear-interpolation R2 of the simplified curve against the original grid and emits a UserWarning if it falls below 0.9' and :516 'If the simplified curve diverges from the original (R2 < 0.9) that's a real signal that simplify_npoints is too small — log it as a warning regardless of phase or snapshot count.' B-05 notes these name two different emission channels.",
    "expected": "Check the length precondition explicitly before the call and let other ValueErrors propagate (or re-raise `from err`); treat a nan R2 as a failure rather than a pass.",
    "failure_scenario": "_simplify raises ValueError for an unrelated reason (non-monotonic x, nmin larger than the input, a NaN). The user is told the arrays have mismatched lengths and is shown two equal numbers, sending debugging in the wrong direction. Separately, a degenerate profile silently reports a good fit and a too-small simplify_npoints ships unnoticed.",
    "repro": "Call params.simplify(x, y, nmin=<larger than len(x)>) with equal-length arrays and read the message; then simplify a constant y array and check whether any warning is emitted.",
    "confidence": "high"
  },
  {
    "id": "S12a-R-31",
    "file": "trinity/_input/dictionary.py",
    "line": 810,
    "class": "state",
    "severity": "S2",
    "claim": "Stale outputs are deleted and metadata.json written only inside flush(), and flush() is skipped when nothing was ever saved — so a run that dies before its first snapshot leaves the previous run's dictionary.jsonl and metadata.json in place, and the atexit path then regenerates a fresh-looking human-readable summary from them.",
    "evidence": "CORROBORATED mechanism, A-only window. Lens B (prose): dictionary.py:807 'If flush_count == 0 and file exists: overwrite (fresh run) - Else: append new snapshots', with the fresh-run branch deleting 'existing files (jsonl AND metadata) so we never end up with a stale metadata.json next to a new simulation's snapshots'; :821 metadata.json written 'on the very first flush of the run'. Lens A (code) supplies the hole: :810-816 the unlink block is guarded by `if self.flush_count == 0`, :825 the metadata write likewise, and :316 `if self.previous_snapshot:` guards the flush() call inside _safe_flush — so a run with zero saved snapshots never enters flush() at all, while :335-342 writes metadata_humanreadable.txt regardless. Output directories are reused: registry.py:234 `os.path.join(os.getcwd(), 'outputs', params['model_name'].value)` with model_name defaulting to the .param stem (read_param.py:373).",
    "expected": "Clear stale run artefacts when the output directory is resolved (registry.py:237 already mkdirs there), not lazily on first flush.",
    "failure_scenario": "A re-run of param/simple_cluster.param crashes during initialisation. outputs/simple_cluster/ still holds the previous run's dictionary.jsonl and metadata.json, and the atexit path regenerates metadata_humanreadable.txt from them — so the directory looks like a completed run and the user analyses last week's results believing they are today's.",
    "repro": "Run a config to completion, then re-run it with an input that fails before the first save_snapshot, and inspect outputs/<model>/.",
    "confidence": "medium"
  },
  {
    "id": "S12a-R-32",
    "file": "trinity/_input/registry.py",
    "line": 439,
    "class": "units",
    "severity": "S4",
    "claim": "ParamSpec.unit mixes two incompatible dialects in one field — machine-parseable conversion strings for input keys and free-text labels for runtime keys — with no marker distinguishing them, and spells the same dimension two ways.",
    "evidence": "Lens A (code, having read the real convert2au): parseable — registry.py:346 `unit='cm**-3'`, :375 `unit='cm**3 * s**-1'`, :377 `unit='erg * s**-1 * cm**-1 * K**(-7/2)'`. Not parseable by convert2au, whose token regex requires a leading letter or underscore — :437 `unit='1/pc**3'`, :439 `unit='1/cm**3'`, :451 `unit='1/Myr'`, :415 `unit='dimensionless'`, :418 `unit='N/A'`. Same dimension, two spellings: 'cm**-3' (:346) vs '1/cm**3' (:439). These never reach convert2au today because runtime keys skip conversion. Lens B corroborates the dual purpose from the docs: materialize_runtime applies `ori_units=spec.unit` '(or \"N/A\" when unitless)' (registry.py:625), i.e. the field is explicitly a display label on the runtime path.",
    "expected": "One parseable dialect with an explicit sentinel for dimensionless, or a separate display-label field.",
    "failure_scenario": "A runtime key is promoted to an input key (moved into default.param) and its registry unit string is copied into the '# UNIT:' line. convert2au raises UnitConversionError on '1/pc**3' at read_param.py:262 — the loud case. The quiet case is a reviewer treating 'dimensionless' and 'N/A' as equivalent to a real unit annotation when auditing S12a-R-04.",
    "repro": "python -c \"import trinity._functions.unit_conversions as c; c.convert2au('1/pc**3')\"",
    "confidence": "high"
  },
  {
    "id": "S12a-R-33",
    "file": "trinity/_input/param_spec.py",
    "line": 80,
    "class": "deadcode",
    "severity": "S4",
    "claim": "SENTINEL_PREFIX is documented as the shared prefix of the sentinel vocabulary but has zero readers; all five sentinel checks hard-code the literal string.",
    "evidence": "A != B. Lens B (prose): param_spec.py:64 enumerates 'def_dir, def_path, def_value, def_unset, all sharing a SENTINEL_PREFIX', and names a test guard test_every_sentinel_default_has_resolver_or_pointer (:78). Lens A (code): param_spec.py:80 `SENTINEL_PREFIX = \"def_\"` with no reader anywhere in the slice; the literals are registry.py:233 `if value == 'def_dir':`, :245 same, :275 `sps_path_is_default = value == 'def_path'`, :307 `if params['sps_refmass'].value == 'def_value':`, and :395-407 `default='def_unset'` (x12).",
    "expected": "Either use the constant (e.g. a shared is_sentinel() helper) or note that it is aspirational.",
    "failure_scenario": "The sentinel convention is renamed (say to 'auto_'); SENTINEL_PREFIX is updated, the five literal comparisons are not, and every resolver silently stops recognising its own default.",
    "repro": "grep -rn SENTINEL_PREFIX trinity/ — one definition, zero uses.",
    "confidence": "high"
  },
  {
    "id": "S12a-R-34",
    "file": "trinity/_input/registry.py",
    "line": 395,
    "class": "other",
    "severity": "S4",
    "claim": "Three different counts describe the SPS column specs: the code has twelve, two docstrings say thirteen, and the same docstring block describes the bundled file as a canonical 7-column layout — with no stated mechanism keeping any of them current.",
    "evidence": "A != B, a direct numeric disagreement. Lens A (code): 'The twelve sps_col_* specs (registry.py:395-407)' and A-13's evidence '`default='def_unset'` (x12)'. Lens B (prose): registry.py:253 'sps_refmass and the 13 sps_col_* specs declare consumed_by=\"sps_path\"' and param_spec.py:70 'the 13 sps_col_* specs are owned by sps_path's bundle resolver', in the same block that describes 'the canonical 7-column SB99 layout (DEFAULT_SPS_COLUMN_MAP)'. B also flags two further unverifiable counts: param_spec.py:1 '200 specs' and registry.py:625 '106 items: 9 ... True and 97 ... False' (restated at read_param.py:465).",
    "expected": "State whether 13 declarable columns map onto a 7-column canonical set (e.g. 6 optional), and pin the count in test/test_registry.py rather than in prose.",
    "failure_scenario": "A user supplying a custom sps_path declares 7 columns while the resolver expects 12 or 13 (or vice versa), producing a column-map error the docs do not explain. No runtime effect from the count drift itself.",
    "repro": "grep -c 'sps_col_' trinity/_input/registry.py and compare against the '13' in registry.py:253 and param_spec.py:70.",
    "confidence": "medium"
  },
  {
    "id": "S12a-R-35",
    "file": "trinity/_input/registry.py",
    "line": 278,
    "class": "divergence",
    "severity": "S4",
    "claim": "_resolve_sps_bundle raises plain ValueError for two user-configuration errors and ParameterFileError for a third in the same function, and mutates two keys other than the one it resolves.",
    "evidence": "Lens A (code): registry.py:278 `raise ValueError(f\"ZCloud={...} is not supported with the default SPS fallback ...\")`; :284 `raise ValueError(\"SB99_rotation=0 is not supported ...\")`; :311 `raise ParameterFileError(f\"sps_refmass is required when sps_path is user-set ...\")`. Cross-key writes: :309 `params['sps_refmass'].value = 1e6`; :319 `params['sps_column_map'] = DescribedItem(...)`, while resolve_all assigns only the resolver's own key (:585). A also notes sps_refmass (:357) carries consumed_by='sps_path' and so is excluded from materialize_runtime (:652-653), making :307's read a bare KeyError if default.param ever drops it. Lens B documents the bundle ownership as intended — registry.py:253 'owns sps_path + sps_refmass + sps_column_map' — and errors.py:1 explains ParameterFileError exists to avoid an import cycle, i.e. it is the designated user-error type.",
    "expected": "One exception type for user-config errors (ParameterFileError, which errors.py exists for), and cross-key effects declared rather than performed as a side effect.",
    "failure_scenario": "A caller catching ParameterFileError to print a friendly 'bad parameter file' message lets the ValueError from :278/:284 escape as an unhandled traceback for the same class of user mistake.",
    "repro": "Set SB99_rotation 0 with the default sps_path and compare the exception type to the sps_refmass error.",
    "confidence": "high"
  },
  {
    "id": "S12a-R-36",
    "file": "trinity/_input/registry.py",
    "line": 186,
    "class": "divergence",
    "severity": "S4",
    "claim": "_validate_stop_at_rCloud_nSnap mutates the parameter it validates, making it the only validator with a side effect; the resolver mechanism exists for value transformation and the two roles are enforced as distinct elsewhere.",
    "evidence": "CORROBORATED and documented, so hygiene only. Lens A (code): registry.py:180-186 `coerced = int(value)` / `if coerced < 0: raise ParameterFileError(...)` / `params['stop_at_rCloud_nSnap'].value = coerced`; validate_all (:556-561) passes `params[spec.name].value` positionally and ignores the return, whereas resolve_all (:580-585) does `params[spec.name].value = spec.resolver(...)`; param_spec.py:157-162 enforces resolver and consumed_by as mutually exclusive. Lens B (prose) documents the mutation as intended: registry.py:165 'Validate AND coerce: whole-number floats (e.g. 5.0 from \"5\") become ints; fractional floats / negatives / non-numerics raise', and registry.py:91/:547 'may raise ParameterFileError or normalize in place'.",
    "expected": "Perform the int coercion in a resolver, leaving validators read-only — or keep it and accept that 'validators may normalize in place' is the contract, which the docs already say.",
    "failure_scenario": "A future refactor makes validate_all idempotent/reorderable or runs validators on a copy for a dry-run; the int coercion is silently lost and stop_at_rCloud_nSnap stays a float, so an exact-type check downstream behaves differently.",
    "repro": "Read registry.py:164-186 against registry.py:556-561.",
    "confidence": "high"
  },
  {
    "id": "S12a-R-37",
    "file": "trinity/_input/dictionary.py",
    "line": 225,
    "class": "state",
    "severity": "S4",
    "claim": "_excluded_keys only ever grows — nothing removes a key — so snapshot exclusion is permanently sticky once set, and it is populated from two places that can disagree with the per-item flag.",
    "evidence": "Lens A (code): dictionary.py:225 `self._excluded_keys: set[str] = set()`; :254-255 `if value.exclude_from_snapshot: self._excluded_keys.add(key)` in __setitem__; :614-617 the same in _clean_for_snapshot; :624 `if key in self._excluded_keys: continue`. No discard or remove anywhere. read_param.py:457 `val.exclude_from_snapshot = True` sets the flag directly, bypassing __setitem__ — benign only because :614-617 re-syncs. Lens B: silent on the cache; B documents exclude_from_snapshot as 'the live DescribedItem flag' (param_spec.py:99), which is precisely the semantics a sticky cache breaks.",
    "expected": "Rebuild the set from the items each snapshot (the loop at :614-617 already walks them), or drop the cache and test item.exclude_from_snapshot directly.",
    "failure_scenario": "Code re-assigns a key with a fresh DescribedItem(exclude_from_snapshot=False) to start recording it mid-run. The key stays in _excluded_keys and is silently dropped from every snapshot; the diagnostic the developer added never appears, and its absence looks like the physics never fired.",
    "repro": "params['k'] = DescribedItem(1, exclude_from_snapshot=True); params['k'] = DescribedItem(2); then inspect a snapshot for 'k'.",
    "confidence": "high"
  },
  {
    "id": "S12a-R-38",
    "file": "trinity/_input/dictionary.py",
    "line": 186,
    "class": "other",
    "severity": "S4",
    "claim": "DescribedItem defines __eq__ without __hash__, so Python sets __hash__ = None and the class is unhashable despite an otherwise complete value-semantics protocol; __eq__ on an array-valued item returns an array.",
    "evidence": "Lens A (code): dictionary.py:118 `__slots__ = ('_value','info','ori_units','exclude_from_snapshot')`; :174-190 a full arithmetic/comparison protocol; :186 `def __eq__(self, other): return self.value == self._unwrap(other)`; :193 `__array__`; no __hash__ anywhere in :98-194. Lens B: silent — B's DescribedItem transcription (dictionary.py:99) covers value plus info, ori_units and exclude_from_snapshot, with no statement about hashability or comparison semantics.",
    "expected": "Define __hash__ explicitly (or __hash__ = None deliberately) so the intent is stated rather than inherited from a language rule.",
    "failure_scenario": "Code that builds `set(params.values())` or uses items as dict keys raises TypeError: unhashable type. Separately `if item == 0:` on an array-valued item raises 'truth value of an array is ambiguous'.",
    "repro": "python -c \"from trinity._input.dictionary import DescribedItem; hash(DescribedItem(1))\"",
    "confidence": "high"
  },
  {
    "id": "S12a-R-39",
    "file": "trinity/_input/dictionary.py",
    "line": 4,
    "class": "other",
    "severity": "S4",
    "claim": "dictionary.py has no module __doc__: the future-import at line 3 precedes the triple-quoted block at lines 4-44, so that block is a plain expression statement rather than a docstring — and it is the block documenting the on-disk output format.",
    "evidence": "Lens A (code): dictionary.py:3 `from __future__ import annotations` followed by :4 `\"\"\"` opening the module description that runs to :44; a module docstring must be the first statement. A notes this is also why the block survived the docstring-stripping pass. Corroborated indirectly by Lens B, which received that same block as extracted prose (B §7's dictionary.jsonl and metadata.json contracts are drawn from dictionary.py:61 and the surrounding range) — so the text exists and is substantive, it simply is not reachable as __doc__.",
    "expected": "Move the string above the future import so help() and Sphinx automodule pick it up.",
    "failure_scenario": "Sphinx automodule and help() show an empty module description for the file that documents the dictionary.jsonl / metadata.json layout — the single most useful docstring in the package for anyone reading outputs.",
    "repro": "python -c \"import trinity._input.dictionary as d; print(d.__doc__)\" -> None",
    "confidence": "high"
  },
  {
    "id": "S12a-R-40",
    "file": "trinity/_input/dictionary.py",
    "line": 929,
    "class": "other",
    "severity": "S4",
    "claim": "The snapshot round-trip is lossy for DescribedItem metadata and type-coercing for any list-valued key: info, ori_units and exclude_from_snapshot are not restored, and a genuinely list-typed parameter returns as an ndarray.",
    "evidence": "Lens B (prose): dictionary.py:929 'This reconstructs: - scalars directly into DescribedItem(value) - list values back into numpy arrays'; :958 '# Lists are converted back to numpy arrays'; contrast dictionary.py:99 which defines DescribedItem as value plus info, ori_units and exclude_from_snapshot. B notes the prose states the behaviour but never flags the asymmetry. Lens A confirms load_snapshot constructs a bare dict (:951 `params = cls()`) and sets only path2output (:954) — consistent with B, and the reason A found the crash-handler side effect (S12a-R-10) in the same function while B found the metadata loss.",
    "expected": "State explicitly that info/ori_units/exclude_from_snapshot are not restored and that a list-typed parameter returns as ndarray.",
    "failure_scenario": "Code resuming from a loaded snapshot reads params[k].ori_units for a unit-sensitive conversion and gets None; or a list-valued config key returns as an ndarray and changes truthiness and equality semantics.",
    "repro": "grep -n 'Lists are converted back' trinity/_input/dictionary.py and compare with the DescribedItem docstring at dictionary.py:99.",
    "confidence": "medium"
  },
  {
    "id": "S12a-R-41",
    "file": "trinity/_input/param_spec.py",
    "line": 109,
    "class": "other",
    "severity": "S4",
    "claim": "The invariant 'run_const intersect metadata_exclude is always empty' is asserted with no named enforcement, unlike the three neighbouring invariants which each cite a test. No counterexample was found.",
    "evidence": "Lens B (prose): param_spec.py:109 '# run_const ∩ metadata_exclude is always empty (a key is written to metadata or blocked from it, never both).' against param_spec.py:78 test_every_sentinel_default_has_resolver_or_pointer, :128 test_consumed_by_targets_exist, :137 test_active_when_only_on_conditional_specs. B notes registry.py:665/:674 make these two projections the sole source of truth for RUN_CONST_KEYS and METADATA_EXCLUDE. Lens A found no spec carrying both flags — A-32's inventory shows path_cooling_CIE and path_cooling_nonCIE as metadata_exclude=True with run_const unset — but did not enumerate exhaustively, so the invariant appears satisfied today.",
    "expected": "A named guard in test/test_registry.py, matching its three neighbours.",
    "failure_scenario": "A spec is flagged both run_const and metadata_exclude; depending on which projection is consulted first the key is either written to metadata.json or blocked, making metadata.json contents order-dependent.",
    "repro": "grep -n 'run_const=True' trinity/_input/registry.py and cross-check each hit for metadata_exclude=True.",
    "confidence": "medium"
  },
  {
    "id": "S12a-R-42",
    "file": "trinity/_input/registry.py",
    "line": 625,
    "class": "citation",
    "severity": "S4",
    "claim": "Doc-drift bundle with no runtime effect: phase labels disagree within one file, flush() is given two complexities, the simplify R2 diagnostic is attributed to two different emission channels, and several hard counts are stated in prose with no mechanism keeping them current.",
    "evidence": "All Lens B (prose), none visible to A. Phase labels: registry.py:1 'Step 10 calls materialize_runtime (Phase 9). Phase 10 will wire Step 6's derived-init resolvers' vs registry.py:625 'Phase-8/9 entry point for read_param Step 10' vs registry.py:409 'materialised by the derived-init resolver in Phase 7/10' — while Phase 10 is stated as unwired. Complexity: dictionary.py:201 'Append-only writes ensure O(1) flush performance' vs :765 'Performance: O(pending_snapshots)'. Channel: dictionary.py:457 'emits a UserWarning if it falls below 0.9' vs :516 'log it as a warning regardless of phase or snapshot count'. Counts: param_spec.py:1 '200 specs'; registry.py:625 '106 items: 9 True and 97 False' (restated read_param.py:465). Also registry.py:118 cites SOURCE_TERM_DESIGN.md by bare filename with no repo-relative path.",
    "expected": "One phase label per entry point; state per-flush cost (O(pending)) and per-snapshot cost (amortised O(1)) separately; name one emission channel; pin counts in test/test_registry.py rather than prose; give the design doc a repo-relative path.",
    "failure_scenario": "No runtime effect. A reader tracking migration status cannot tell which phases shipped — which the project's own CLAUDE.md names as a recurring doc-drift problem — and a sweep harness filtering UserWarnings to catch degraded simplifications may find the signal only in the log (see S12a-R-30 and open question Q9).",
    "repro": "grep -n 'Phase' trinity/_input/registry.py | head -40; grep -n 'O(1)\\|O(pending_snapshots)' trinity/_input/dictionary.py; grep -rn 'SOURCE_TERM_DESIGN' trinity/ docs/.",
    "confidence": "high"
  },
  {
    "id": "S12a-R-43",
    "file": "trinity/_input/param_spec.py",
    "line": 61,
    "class": "deadcode",
    "severity": "S4",
    "claim": "Declared-but-unused bundle, flagged not proposed for removal per project rule 3: the 'deprecated' category and its enforced deprecated_note contract have no members; a whole back-compat category is documented as empty; specs_by_category has no in-slice caller; _REPO_ROOT is computed identically in two files; _clean_for_snapshot takes an unused snap_id; save_debug_snapshot uses a bare except:; and __init__.py is empty.",
    "evidence": "Lens A (code): param_spec.py:61 'deprecated' in the Category Literal, :140 deprecated_note, :147-150 the __post_init__ enforcement — no spec in SPECS (:329-533) uses it; registry.py:541-543 specs_by_category with no reference in the slice (low confidence, callers may exist in tools/ or docs/); read_param.py:37 and registry.py:68 both `_REPO_ROOT = Path(__file__).resolve().parents[2]`; dictionary.py:577 `def _clean_for_snapshot(self, snap_id: int)` with snap_id unused in :600-706; dictionary.py:1063 a bare `except:` that also catches KeyboardInterrupt and SystemExit (ruff's configured F-rules do not include E722); read_param.py:21 `import sys` unused (F401 also not in the configured rule set). Lens B (prose) corroborates two of these independently: param_spec.py:60-61 '---- Parsed for back-compat only, never consumed ---- # back-compat retired specs (currently none; kept for future use)' (B-29), and the absence of any prose section for __init__.py (B-32) — which matches A's report that the file is empty. That is the cleanest A ≡ B agreement in the slice.",
    "expected": "Flag only. If the package re-exports read_param / DescribedDict / ParameterFileError, a one-line __init__ docstring would name the surface; the bare except should be `except (KeyError, TypeError, AttributeError):`.",
    "failure_scenario": "Ctrl-C pressed while save_debug_snapshot resolves the output path is discarded and the snapshot is silently written to the cwd instead of the run directory. If the package is re-nested (src/trinity/_input/) and only one parents[2] index is updated, the CIE paths built in read_param.py:425 and the SPS/opiate paths in registry.py:246/:291 point at different roots and only one fails loudly.",
    "repro": "grep -rn 'specs_by_category\\|SENTINEL_PREFIX\\|_REPO_ROOT =' trinity/ tools/ docs/; grep -n 'deprecated' trinity/_input/registry.py; read dictionary.py:1057-1066.",
    "confidence": "high"
  },
  {
    "id": "S12a-R-44",
    "file": "trinity/_input/dictionary.py",
    "line": 457,
    "class": "other",
    "severity": "S4",
    "claim": "simplify's tuning surface is undocumented or unfalsifiable: grad_inc is named once with no default, spec or unit; the nmin fallback of 100 silently duplicates the simplify_npoints schema default; and the output-size contract is 'normally nmin' with two independent rules that can each force extra mandatory points.",
    "evidence": "Lens B (prose): dictionary.py:457 'sharp bends (points where the Menger curvature exceeds grad_inc on rescaled [0, 1] axes — unit-free threshold)' with grad_inc appearing nowhere else in the slice's prose; 'nmin defaults to the simplify_npoints parameter on the dict (loaded from default.param), or to 100 if absent'; 'Output size is normally nmin. Endpoints and every high-prominence extremum are always retained — for very noisy curves with more than nmin such extrema, the output may exceed nmin', plus a second mandatory rule ('an x-uniform coverage skeleton — one feature-pool point per equal-width x-chunk is promoted to mandatory'). Lens A confirms grad_inc is a real keyword argument: dictionary.py:509 `_simplify(x_arr, y_arr, nmin=nmin, grad_inc=grad_inc)`. Neither lens found a registry spec for grad_inc.",
    "expected": "If grad_inc is a tunable it needs a spec and default like simplify_npoints; if it is a constant of the standalone simplify module, say so. State an upper bound on output size, and make the nmin fallback reference the schema default rather than restating 100.",
    "failure_scenario": "default.param's simplify_npoints is changed to 50; any path where the key is absent (a bare DescribedDict in a test, a debug-snapshot rehydration) silently uses 100 and produces snapshot arrays at a different resolution than production. Separately, a noisy bubble profile produces arrays much larger than simplify_npoints and no test can assert a size contract, because 'normally' admits any counterexample.",
    "repro": "grep -rn 'grad_inc' trinity/_input/ trinity/_functions/; grep -n simplify_npoints trinity/_input/default.param.",
    "confidence": "medium"
  },
  {
    "id": "S12a-R-45",
    "file": "trinity/_input/registry.py",
    "line": 118,
    "class": "other",
    "severity": "S4",
    "claim": "UNADJUDICATED — the f_A validator's documented definition of 'kappa active' includes any number, while cooling_boost_kappa's documented default is the number 1.0 (an explicit no-op), which read literally would fire the double-boost warning on every run that sets f_A.",
    "evidence": "Lens B only (prose): registry.py:118 'combining it with cooling_boost_mode != none or an active cooling_boost_kappa double-counts interface cooling. Validators run BEFORE resolvers (read_param Steps 5 vs 7), so cooling_boost_kappa is still its raw value here -- a number or the string \"auto\"; both count as \"kappa active\"'; fkappa_auto.py:98 'Numeric values pass through UNTOUCHED (the default 1.0 path stays byte-identical)'. Lens A's account of _validate_cooling_boost_fA (registry.py:128-136) reports only `try: fA = float(value) ... if not (fA > 0): raise ParameterFileError(...)` and does not mention a cross-knob warning, so A neither confirms nor refutes the warning's existence or its condition. A also does not state cooling_boost_kappa's registry default. DEMOTED to low confidence for that reason.",
    "expected": "'kappa active' should be defined as kappa != 1.0 or 'auto', otherwise the warning is unconditional noise for every f_A user.",
    "failure_scenario": "Users see a double-boost warning on a configuration with only one knob set, learn to ignore the warning, and miss the genuine double-boost case. If instead the warning does not exist in code, the comment describes a check that was never written.",
    "repro": "Read registry.py:118-150 for the warning condition, and read the default= kwarg on the cooling_boost_kappa spec at registry.py:387.",
    "confidence": "low"
  },
  {
    "id": "S12a-R-46",
    "file": "trinity/_input/read_param.py",
    "line": 477,
    "class": "state",
    "severity": "S3",
    "claim": "A comment records a shipped bug of exactly the class this slice's worst finding belongs to: a default.param key replaced by a runtime DescribedItem, so every run integrated with include_PHII=True regardless of the .param file. Results predating the guard carry that override.",
    "evidence": "Lens B (prose), verbatim: read_param.py:477 'A key from default.param that has been replaced (not just mutated) with a fresh DescribedItem has lost the user's value — the most recent offender was include_PHII, which meant every run integrated with include_PHII=True regardless of what the .param file said. Fail loudly so this never ships silently again.' Lens A (code) shows the resulting guard covers only replacement, not mutation, and that four mutations sit in the same function 170 lines earlier (see S12a-R-01 and S12a-R-02). The comment's own parenthetical — 'not just mutated' — names the gap.",
    "expected": "Documented as fixed for the replacement path. The finding is that (a) the mutation path was left open, and (b) any archived output predating the guard has include_PHII=True baked in.",
    "failure_scenario": "Historical outputs and paper figures generated before the guard integrated with include_PHII=True even where the .param said otherwise; comparisons against them are invalid, and nothing in those runs' metadata records which value was actually used.",
    "repro": "grep -n 'include_PHII' trinity/_input/read_param.py; then git log -S include_PHII to date the guard against any archived outputs/ or paper/ data.",
    "confidence": "high"
  }
]
```
