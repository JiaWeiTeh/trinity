# S13b output CLOUDY export — Lens B (what the code claims)

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

**This is a prose-only transcription.** Everything below is a statement *made by the comments and
docstrings* of the slice, not a statement about what the code does. I have seen **no code** — no
implementation, no signatures, no tests, no parameter files. My entire input was
`scratchpad/lens/S13b_output_cloudy/prose.md`, which contains only the comment and docstring blocks
extracted from five files:

- `trinity/_output/cloudy/trinity_to_cloudy.py`
- `trinity/_output/cloudy/snapshot_to_deck.py`
- `trinity/_output/cloudy/run_loader.py`
- `trinity/_output/cloudy/dlaw.py`
- `trinity/_output/cloudy/__init__.py`

Where I write "the prose claims X", that is deliberate: I cannot know whether X holds. Where the
prose is silent, I say so — silence at a unit boundary is itself the finding.

---

## 1. Per-quantity claimed-units table

The slice straddles TRINITY astro units and CLOUDY cgs/log10. Below, every quantity the prose
mentions, with the claimed unit on each side, the claimed conversion, and whether log10 is claimed.
"**not stated**" means the prose never says — it is not an inference that no conversion happens.

### 1a. Quantities flowing into / out of `dlaw.py`

| Quantity | TRINITY-side unit (claimed) | Conversion (claimed) | CLOUDY-side unit (claimed) | log10 applied? | Prose |
|---|---|---|---|---|---|
| Shell radius array (`shell_r_pc`) | pc, linear | "Convert pc → cm … in log space"; offset `+18.4894` | cm | **yes** — output col is `log10(r/cm)`, `.6f` | `dlaw.py:1`, `dlaw.py:64`, `dlaw.py:165`, `dlaw.py:174` |
| Shell density array (`shell_log_n_pc3`) | **already log10**, base unit pc⁻³ | "pc^-3 → cm^-3 in log space"; offset `-55.4681` | cm⁻³ | **yes**, on input *and* output | `dlaw.py:64`, `dlaw.py:165`, `dlaw.py:175` |
| Ambient radius array (`ambient_r_pc`, from `metadata.initial_cloud_r_arr`) | pc, linear | "Same units as the shell arrays" | cm | yes (same path as shell) | `dlaw.py:64`, `snapshot_to_deck.py:59` |
| Ambient density array (from `metadata.initial_cloud_n_arr`) | pc⁻³, **linear** | two-stage: `snapshot_to_deck` does linear pc⁻³ → log10 pc⁻³, then `dlaw` adds `-55.4681` | cm⁻³ | **yes**, added by the caller then offset by dlaw | `snapshot_to_deck.py:59`, `snapshot_to_deck.py:240` |
| `r_in_pc`, `r_out_pc` (dlaw params) | pc | none claimed — used for the bracket check | n/a | no | `dlaw.py:64` |
| `edge_threshold` | dimensionless `\|Δlog n / Δlog r\|` | none | n/a | operates on log-ratios | `dlaw.py:39`, `dlaw.py:64` |

Literal quotes:

> "The block converts (r [pc], log10 n [pc^-3]) pairs into the (log10 r [cm], log10 n [cm^-3]) form
> CLOUDY expects" — `dlaw.py:1`

> "shell_r_pc, shell_log_n_pc3 — Shell radius (pc) and log10 number density (1/pc^3). ≥2 points."
> — `dlaw.py:64`

> "ambient_r_pc, ambient_log_n_pc3 — Optional ambient ISM profile … Same units as the shell arrays."
> — `dlaw.py:64`

> "`# --- 5. Convert pc → cm and pc^-3 → cm^-3 in log space ------------------`" — `dlaw.py:173`
> followed by "`# +18.4894`" (`dlaw.py:174`) and "`# -55.4681`" (`dlaw.py:175`)

> "Pulls metadata.initial_cloud_{r,n}_arr (linear, in pc / pc^-3) and converts the density to log10
> before handing to dlaw." — `snapshot_to_deck.py:59`

> "`# Convert linear pc^-3 → log10 pc^-3 (eps guard for safety; TRINITY writes positive values, but
> defends against future regressions).`" — `snapshot_to_deck.py:240`

**Asymmetry claimed:** the shell density is claimed to arrive **already in log10 pc⁻³**, the ambient
density **linear pc⁻³**. That asymmetry is stated only on the *receiving* side (`dlaw.py:64`'s
parameter name and doc). `snapshot_to_deck.py` never states the log convention of the shell array it
forwards — it documents the log conversion only for the ambient arm.

### 1b. Template substitution keys produced by `snapshot_to_deck.snapshot_to_values`

Declared key set: "Substitution keys (TITLE, AGE_YR, LOG_QH, LOG_RIN, LOG_ROUT, ZREL, DLAW_BLOCK,
DLAW_ROWS) plus a `_diagnostics` sub-dict" — `snapshot_to_deck.py:59`.

| Key | TRINITY-side unit (claimed) | Conversion (claimed) | CLOUDY-side unit (claimed) | log10? | Prose |
|---|---|---|---|---|---|
| `TITLE` | — | — | — | — | no prose at all |
| `AGE_YR` | cluster age; CLI flag is `--age MYR`; validation band is `[age_min_yr, age_max_yr]` | Myr → yr factor **not stated** | years | no | `trinity_to_cloudy.py:6`, `snapshot_to_deck.py:166` |
| `LOG_QH` | `Qi`, photons **per Myr** | "`# ph/Myr → ph/s`" — numeric factor not stated | photons s⁻¹ | **not stated** — only the `LOG_` prefix implies it | `snapshot_to_deck.py:181` |
| `LOG_RIN` | **not stated** (presumably `R2`) | **not stated** — no pc→cm claim anywhere in `snapshot_to_deck.py` | **not stated** | **not stated** — key name only | `snapshot_to_deck.py:59` |
| `LOG_ROUT` | **not stated**; "Outer radius: rShell unless user requested extension" | **not stated** | **not stated** | **not stated** — key name only | `snapshot_to_deck.py:59`, `snapshot_to_deck.py:184` |
| `ZREL` | `bundle.summary["ZCloud"]`; "z_override (>0, finite) wins" | **not stated** (no solar normalisation claimed) | **not stated** | **not stated** | `snapshot_to_deck.py:59`, `snapshot_to_deck.py:200` |
| `DLAW_BLOCK` | — | — | "the full dlaw block (header + rows + footer)" | rows are log/log per §1a | `snapshot_to_deck.py:59` |
| `DLAW_ROWS` | — | — | "rows only" | as above | `snapshot_to_deck.py:59`, `snapshot_to_deck.py:256` |

> "DLAW_BLOCK is the full dlaw block (header + rows + footer); DLAW_ROWS is rows only. Templates can
> use either." — `snapshot_to_deck.py:59`

**The single highest-value gap:** `LOG_RIN` / `LOG_ROUT` are the radii CLOUDY integrates over, and
the prose in this slice never once states their unit or their log base. The only radius unit claim in
the whole slice is dlaw's `r_in_pc, r_out_pc … (pc)` (`dlaw.py:64`) — a *different* pair of values
consumed by a *different* function for a bracket check. The pc→cm step (`+18.4894`) is documented
only inside `dlaw.py`; nothing claims it is applied to `LOG_RIN`/`LOG_ROUT`.

### 1c. Legacy scalar quantities (`run_loader._parse_simulation_end`)

Stated convention: "units in the key name where they differ from the summary's AU = (Msun, pc, Myr)
convention" — `run_loader.py:189`.

| Key | Unit claimed by the key name |
|---|---|
| `t_now_myr` | Myr |
| `R2_pc` | pc |
| `shell_nMax_cm3` | cm⁻³ |
| `shell_v_kms` | km s⁻¹ |
| `mCloud_msun` | M☉ |
| `nCore_cm3` | cm⁻³ |
| `rCloud_pc` | pc |
| `rCore_pc` | pc |
| `alpha` | dimensionless (no suffix) |
| `nISM_cm3` | cm⁻³ |
| `model_name`, `outcome`, `detail`, `exit_code` | non-physical |

> "`# Final-state numeric fields with units stripped from the value`" — `run_loader.py:233`;
> "`# value is "<number> <unit>" — take the first whitespace-split token`" — `run_loader.py:263`

Note the package-internal collision: densities here are claimed **cm⁻³**, while the array densities
of §1a are claimed **pc⁻³**, and velocity here is **km/s** while the declared house convention is
`(Msun, pc, Myr)` (which would make velocity pc/Myr). Both live in the same five files.

---

## 2. Claimed CLOUDY deck contract

### 2a. What is claimed to be emitted

The prose names only **three** CLOUDY-syntax artefacts:

1. The dlaw block, quoted verbatim as an "Output format" example — `dlaw.py:1`:

```
dlaw table radius
continue {log10(r/cm):.6f} {log10(n_H/cm^-3):.4f}
continue ...
end of dlaw
```

2. A `table star` line carrying the SB99 atmosphere grid name —
   "The `<<<EDIT_ME>>>` sentinel in the deck's `table star` line MUST be replaced by hand with the
   user's CLOUDY-compiled SB99 atmosphere grid name before `cloudy -r ...`." (`trinity_to_cloudy.py:6`)

3. A bundled `trinity_linelist.dat` copied next to the decks —
   "`# Copy bundled linelist next to the decks (once per run)`" (`trinity_to_cloudy.py:412`). The
   prose never says which CLOUDY command consumes it.

**No command order is claimed anywhere.** The prose never enumerates the deck's commands (`radius`,
`Q(H)`/`ionization parameter`, `metals`/`abundances`, `age`, `title`, stopping criteria, save
commands). Those live in "the bundled .in template" (`snapshot_to_deck.py:1`), which is not in this
slice, so the deck contract as claimed is limited to the eight substitution keys of §1b plus the
sentinel above.

### 2b. Argument conventions claimed

- dlaw rows: `continue` prefix on every row (including, per the example, the first), two columns,
  `.6f` for log10 r/cm and `.4f` for log10 n/cm⁻³, closed by `end of dlaw`.
- The open/row-prefix/close strings are documented as *configurable*: "dlaw_open, dlaw_row_prefix,
  dlaw_close — CLOUDY syntax knobs. See module-level defaults." (`dlaw.py:64`). The default values
  are never printed in the prose.
- Substitution grammar: "`# A {{KEY}} placeholder.`" (`trinity_to_cloudy.py:76`) — the deck template
  uses `{{KEY}}`; `<<<EDIT_ME>>>` is a *non*-placeholder sentinel that survives rendering.

### 2c. Claimed CLOUDY version / syntax reference

There is exactly one version reference in the slice, and it is an admission of uncertainty:

> "Output format (defaults — **best-guess for CLOUDY C17/C22**; see Step 5 smoke test)" — `dlaw.py:1`

> "`# Best-guess CLOUDY syntax (Step 0 / Option B). Override at call site if a live smoke test
> reveals a different working form.`" — `dlaw.py:31`

**There is no citation to the CLOUDY manual, to Hazy, to any section number, to any paper, or to any
file-format spec anywhere in the slice.** The only external artefact named is
`trinity/_input/registry.py` (`run_loader.py:35`), an internal file. "Step 5 smoke test" is a pointer
to an unnamed plan document whose outcome is never reported in the prose.

One further uncited behavioural claim about CLOUDY itself:

> "The user must replace this in the deck before running CLOUDY (CLOUDY will fail loudly on an
> unknown atmosphere grid name, which is the safe failure)." — `trinity_to_cloudy.py:71`

### 2d. Output files claimed

> "Writes `<run_dir>/cloudy/<model>_<index>_<phase>_t<age>myr.in` plus a sidecar `.dlaw.txt` with
> just the dlaw block, plus a copy of the bundled `trinity_linelist.dat`." — `trinity_to_cloudy.py:6`

> "Auto-build a filename-safe prefix: `<model>_<idx>_<phase>_t<age>myr`. Floats use "p" instead of
> "." so the prefix is shell- and CLOUDY-safe." — `trinity_to_cloudy.py:273`

> "`# Only copy if missing or stale (avoids needless re-writes on --all)`" — `trinity_to_cloudy.py:440`

`--all` additionally writes `manifest.json` (`trinity_to_cloudy.py:6`).

---

## 3. Claimed `dlaw` conventions

**Grid convention.** `dlaw table radius` — a *radius* table (as opposed to a depth table), rows
ordered by increasing radius. Ordering is claimed by the pipeline step:
"`# --- 2. Sort and dedup adjacent duplicates (keep last) ------------------`" (`dlaw.py:124`), and
for the ambient arm "`# dedup ambient (keep last value at each unique r), same recipe as shell`"
(`dlaw.py:152`). Dedup tie-break is claimed to be **keep last**.

**Claimed pipeline order** (transcribed from the numbered section comments in `dlaw.py`):

0. Validate scalar parameters (`dlaw.py:93`)
1. Validate and arrayify inputs (`dlaw.py:109`)
2. Sort and dedup adjacent duplicates, keep last (`dlaw.py:124`)
3. Optionally splice ambient past the shell tail (`dlaw.py:133`)
4. Bracket check, "with tiny float tolerance" (`dlaw.py:161`)
5. pc → cm and pc⁻³ → cm⁻³ in log space (`dlaw.py:173`)
6. IF-preserving densification, "only if too sparse" (`dlaw.py:179`)
7. Final validation (`dlaw.py:187`)
8. Format (`dlaw.py:197`)

Note the claimed ordering: the **bracket check happens in pc, before the cm conversion**, and
**densification happens after the cm conversion**, i.e. inserted points are interpolated in
`(log r/cm, log n/cm⁻³)`.

**Spacing / densification.**

> "min_rows — If the post-splice profile has fewer rows, densify by inserting points in non-edge
> spans. Edge (IF-like) pairs are preserved verbatim." — `dlaw.py:64`

> "Insert linearly-interpolated points into non-edge spans until `len >= target_rows`. Edge (steep)
> pairs are preserved verbatim. If every pair is an edge, the input is returned unchanged with a
> warning." — `dlaw.py:212`

> "dens_profile — TRINITY profile shape; reserved for future PCHIP-on-densBE support. Currently
> unused; densification is linear-in-(log r, log n)." — `dlaw.py:64`

> "`# Distribute extra_needed slots across smooth pairs proportional to dlog_r`" — `dlaw.py:238`
> "`# Hand out the leftover slots to pairs with the largest fractional part`" — `dlaw.py:245`
> "`# Stable order ensures determinism for ties`" — `dlaw.py:247`
> "`# Build output: original rows, plus inserted inner points in smooth pairs`" — `dlaw.py:251`

**Edge / IF detection.**

> "`# |Δlog n / Δlog r| above this counts as an IF-like discontinuity. PL profiles are O(1);
> transition-phase IFs in TRINITY snapshots are O(1e5). 50 separates them with margin.`" — `dlaw.py:39`

(The default `edge_threshold` is therefore claimed to be **50**. "IF", "PL" and "densBE" are never
expanded in the slice.)

**Endpoints and extrapolation.**

> "r_in_pc, r_out_pc — Inner / outer radius CLOUDY will integrate over (pc). **Must lie within the
> union of shell + ambient r-range.**" — `dlaw.py:64`

> "`# --- 4. Bracket check (with tiny float tolerance) -----------------------`" — `dlaw.py:161`

No extrapolation path is claimed anywhere. Out-of-range is claimed to be an error, raised as
`DlawError` ("Raised when dlaw construction fails validation." — `dlaw.py:46`), and the caller
pre-empts one specific case:

> "`# Without ambient data we cannot satisfy the dlaw bracket check past rShell — surface this here
> rather than as a deeper "past dlaw range end" error from build_dlaw_block.`" — `snapshot_to_deck.py:218`

**Splice condition.**

> "Ambient splice — Only when radius_out_pc > rShell AND extend_with_ambient is True." —
> `snapshot_to_deck.py:59`; "`# Ambient splice from metadata, only when actually extending past
> rShell`" — `snapshot_to_deck.py:214`; "`# Outer radius: rShell unless user requested extension`" —
> `snapshot_to_deck.py:184`

**Postcondition.** "Returns — str: Multi-line dlaw block, **no trailing newline**." (`dlaw.py:64`)

**Rows-only extraction.** "`# Rows-only view: strip the first (header) and last (footer) lines.`"
(`snapshot_to_deck.py:256`) and "`# excluding open/close`" (`snapshot_to_deck.py:259`) — i.e. the
caller asserts open and close are exactly one line each.

---

## 4. Claimed snapshot-selection rule

Transcribed verbatim from `trinity_to_cloudy.py:6`:

> "Snapshot picker (mutually exclusive, exactly one required):
> `--age MYR` cluster age (Myr since tSF) — picks closest snapshot;
> `--t-now MYR` raw simulation time (advanced);
> `--index N` Nth snapshot, -1 = last;
> `--phase NAME [--pick first|last]`;
> `--all` one deck per snapshot, plus manifest.json"

Supporting claims:

- "`# Exactly-one-picker enforcement (argparse mutex only enforces "at most one")`" —
  `trinity_to_cloudy.py:159`. So *exactly one* is claimed to be enforced by extra code, not by argparse.
- "Resolve the picker flags into a list of (index, snapshot) tuples." — `trinity_to_cloudy.py:190`
- "`# filter() re-indexes from 0; map back to the original index by round-tripping through
  get_at_time on the unfiltered output.`" — `trinity_to_cloudy.py:221`
- "`# _parse_args ensures we never reach here`" — `trinity_to_cloudy.py:229` (claimed-unreachable branch)
- "`# Single-snapshot dry run is enforced at parse time; just print.`" — `trinity_to_cloudy.py:371`

**Unspecified:** the metric for "closest" under `--age` (closest in cluster age? absolute or
relative?), the tie-break when two snapshots are equidistant, and the default for `--pick` when
`--phase` is given without it.

---

## 5. Claimed defaults, invariants, preconditions, postconditions

### 5a. `snapshot_to_deck.snapshot_to_values` — the validation contract

Verbatim (`snapshot_to_deck.py:59`), with the mirrored inline section comments:

1. "Required keys present." (`snapshot_to_deck.py:96`)
2. "No NaN / Inf in used scalars / arrays." (`snapshot_to_deck.py:104`)
3. "**t_now > tSF (strict)**." (`snapshot_to_deck.py:126`)
4. "**R2 > 0, rShell > R2, Qi > 0**." (`snapshot_to_deck.py:133`)
5. "Shell array lengths match and are **>= 2**." (`snapshot_to_deck.py:143`)
6. "shell_r_arr endpoints match R2 / rShell (**rel_tol=1e-12**) — simplify preserves them **by
   contract**; an exact-equality drift would indicate upstream regression." (`snapshot_to_deck.py:154`)
7. "Cluster age in [age_min_yr, age_max_yr]: **warn** unless hard_age_bounds." (`snapshot_to_deck.py:166`)

> "**All steps except 7 raise `SnapshotInvalid`.**" — `snapshot_to_deck.py:59`

Defaults claimed: Z comes from `bundle.summary["ZCloud"]` "by default; z_override (>0, finite) wins"
(`snapshot_to_deck.py:59`); outer radius defaults to `rShell` (`snapshot_to_deck.py:184`).

Module-level postcondition: "**Pure transformation** … **No file I/O, no template rendering.**"
(`snapshot_to_deck.py:1`)

### 5b. Hard hazard claim (`snapshot_to_deck.py:34`)

> "`# Sentinel for "key absent" — needed because TrinityOutput.Snapshot has __getitem__ but no
> __contains__/__iter__, so `key in snap` falls back to integer-indexed iteration that **never
> terminates**. Use snap.get(k, _MISSING).`"

This is the strongest safety claim in the slice: a specific, named idiom is claimed to hang forever.

### 5c. `trinity_to_cloudy` — status gate and rendering

> "**Refuse** to convert runs whose termination exit code is not in the clean range (0–9).
> Inspection-required (50–59 or 99) and error (10–29) outcomes both require `--force`. Source:
> `bundle.end_state` — `metadata.json[termination]` for v3+, legacy `simulationEnd.txt` otherwise."
> — `trinity_to_cloudy.py:238`

> "Substitute `{{KEY}}` placeholders. **Raise `UnsubstitutedPlaceholder` on any `{{KEY}}` left after
> substitution.** Sentinels not matching the `{{KEY}}` pattern (notably `<<<EDIT_ME>>>`) pass through
> unchanged." — `trinity_to_cloudy.py:293`; "Raised when the rendered deck still contains `{{KEY}}`
> placeholders." — `trinity_to_cloudy.py:82`

> "`# A {{KEY}} placeholder. Word-boundary match means <<<EDIT_ME>>> is invisible to the renderer
> (passes through unchanged).`" — `trinity_to_cloudy.py:76`

> "The `<<<EDIT_ME>>>` sentinel … **MUST** be replaced by hand … before `cloudy -r ...`" —
> `trinity_to_cloudy.py:6`; "Closing-summary TODO printed **only when** the SB99 sentinel is in the
> deck." — `trinity_to_cloudy.py:456`

Bootstrap invariant: "`# … prepend the repo root (three levels up: cloudy/ → _output/ → trinity/ →
repo root) so the package imports resolve. The -m path is unaffected (__package__ is set when
launched as a module).`" — `trinity_to_cloudy.py:35`

### 5d. `run_loader` — layout and error contract

> "A v4+ run directory has this shape:: `<run_dir>/ ├── <model>.param # raw input config (not parsed
> here) ├── dictionary.jsonl # snapshot stream (via TrinityOutput) ├── metadata.json # run-invariant
> data + termination └── trinity_*.log # logs (ignored)`" — `run_loader.py:1`

> "Legacy runs (pre-Phase-5) additionally carried `<model>_summary.txt` and `simulationEnd.txt`. The
> text-parse fallbacks below still load those, with a `DeprecationWarning`; **they will be removed in
> Phase 6.**" — `run_loader.py:1` (repeated at `run_loader.py:155` and `run_loader.py:189`)

> "Raises — `RunLoadError`: If any expected file is missing, malformed, or carries an unknown
> `dens_profile`. `FileNotFoundError`: If `run_dir` itself does not exist." — `run_loader.py:58`

> "`# v2+ metadata.json carries every run-constant scalar as a top-level key, so it IS the summary.
> We strip the reserved blocks (termination/final_state/_metadata_version) so the returned mapping
> has the same shape as the legacy <model>_summary.txt parse.`" — `run_loader.py:102`

> "`# Prefer the structured termination block in metadata.json (v3+ schema, written by
> :func:write_simulation_end). Fall back to text-parsing simulationEnd.txt for legacy runs …`" —
> `run_loader.py:124`

> "`# Canonical TRINITY density-profile enum (mirrors _validate_dens_profile in
> trinity/_input/registry.py).`" — `run_loader.py:35`

Legacy `_parse_summary_txt` format contract: "Format: `<key><whitespace><value>` per line; comments
start with `#`; blank lines ignored. Values are coerced (in order): bool, `None`, `nan`/`inf`, int,
float, Python-literal (lists, tuples), else string." — `run_loader.py:155`; and the coercion notes
"`# nan / inf are NOT Python literals — float() handles them but ast doesn't.`" (`run_loader.py:302`),
"`# int (only if it looks like one — avoid "1.0" → ValueError fallthrough)`" (`run_loader.py:306`),
"`# Python literal (lists, tuples, dicts) — must start with a recognisable literal opener, otherwise
we'd accept arbitrary expressions.`" (`run_loader.py:317`).

`_parse_simulation_end` tolerance claim: "Pre-fix runs that wrote `Status`/`End Reason`/`Raw Reason`
lines are tolerated: those values are accepted as fallbacks for `outcome`/`detail` when the new keys
are absent." — `run_loader.py:189`; "`# Old "Status: SUCCESS/FAILED/ERROR" maps to a coarse outcome
bucket; callers should prefer exit_code for fine-grained gating.`" — `run_loader.py:274`.

### 5e. Public API surface claimed

`__init__.py:1` re-exports `build_dlaw_block, DlawError, load_run, RunBundle, RunLoadError,
snapshot_to_values, SnapshotInvalid`, and adds: "Sub-modules can also be imported directly when
finer-grained access is needed (e.g. the `DEFAULT_*` constants)." `snapshot_to_deck.py:1` declares its
public API as `snapshot_to_values, SnapshotInvalid`; `dlaw.py:1` as `build_dlaw_block, DlawError`;
`run_loader.py:1` as `load_run, RunBundle, RunLoadError`. These four are mutually consistent.

---

## 6. Citations and external references

| Reference | Kind | Where | Resolvable from prose? |
|---|---|---|---|
| "CLOUDY C17/C22" | software version | `dlaw.py:1` | version only, no manual/section |
| "Step 5 smoke test", "Step 0 / Option B", "Step 4" | plan-document steps | `dlaw.py:1`, `dlaw.py:31`, `snapshot_to_deck.py:1` | **no** — document never named |
| "Phase 5", "Phase 6", "pre-Phase-2" | roadmap phases | `run_loader.py:1`, `:124`, `:155`, `:189` | **no** |
| `trinity/_input/registry.py::_validate_dens_profile` | internal source | `run_loader.py:35` | yes (internal) |
| `write_simulation_end` | internal function | `run_loader.py:124` | partially (module not named) |
| `TrinityOutput`, `TrinityOutput.Snapshot`, `get_at_time`, `find_data_path`, `output.initial_cloud_profile()` | internal API | `run_loader.py:1`, `:54`; `trinity_to_cloudy.py:186`, `:221`; `snapshot_to_deck.py:226` | partially |
| `trinity_linelist.dat` | bundled data file | `trinity_to_cloudy.py:6` | named, format undocumented |

**No astrophysics literature is cited anywhere in the slice. No CLOUDY documentation (Hazy or
otherwise) is cited anywhere in the slice.** Given that this is the module that writes CLOUDY input
syntax, that absence is itself the headline citation finding.

---

## 7. Internal contradictions *within the prose*

1. **`n` vs `n_H`.** `dlaw.py:1` prose sentence says the output column is "log10 n [cm^-3]"; the
   format block three lines later says `{log10(n_H/cm^-3):.4f}`. The inputs are described only as
   "number density" / "density profile". No composition, mean-molecular-weight, or He correction is
   claimed anywhere — the conversion is presented as purely geometric (`-55.4681`). Either the two
   descriptions mean the same thing (and the `_H` is loose), or a species conversion is missing and
   undocumented.

2. **`+18.4894` vs `-55.4681`.** `dlaw.py:174`/`:175`. If the density offset were exactly `-3 ×` the
   radius offset it would be `-55.4682`; the true value `-3 × log10(3.0857e18) = -55.46806…` rounds
   to `-55.4681`. So the two rounded constants are each individually right but are **not** exactly
   3× each other, which the prose presents them as being ("pc → cm **and** pc^-3 → cm^-3").

3. **Unit-suffix rule contradicted by its own list.** `run_loader.py:189` says "units in the key name
   **where they differ** from the summary's AU = (Msun, pc, Myr) convention", then lists
   `t_now_myr`, `R2_pc`, `mCloud_msun`, `rCloud_pc`, `rCore_pc` — all of which *conform* to that
   convention and yet carry suffixes. Only `shell_nMax_cm3`, `shell_v_kms`, `nCore_cm3`, `nISM_cm3`
   satisfy the stated rule.

4. **Two density conventions in five files.** Arrays: pc⁻³ (`dlaw.py:1`, `:64`;
   `snapshot_to_deck.py:59`, `:240`). Legacy scalars: cm⁻³ (`run_loader.py:189`). Nothing in the
   prose flags the collision or names a conversion between them.

5. **Exit-code gate is incomplete.** `trinity_to_cloudy.py:238` first states a blanket rule ("not in
   the clean range (0–9)" ⇒ refuse), then enumerates only 10–29, 50–59, 99 as `--force`-able. Codes
   30–49, 60–98 and ≥100 fall under the blanket sentence but are absent from the enumeration; the
   prose does not say whether they are refusable-with-force or hard-refused.

6. **Tolerance language.** `snapshot_to_deck.py:59` step 6 states `rel_tol=1e-12` and, in the same
   sentence, "an **exact-equality** drift would indicate upstream regression". A `rel_tol=1e-12`
   comparison is by definition not exact equality; the sentence describes two different tests.

7. **Literal-coercion set.** `run_loader.py:155` docstring says "Python-literal (**lists, tuples**)";
   `run_loader.py:317` comment says "Python literal (**lists, tuples, dicts**)".

8. **`<<<EDIT_ME>>>` rationale.** `trinity_to_cloudy.py:76` attributes the sentinel's survival to
   "Word-boundary match". `<<<EDIT_ME>>>` contains no `{{`/`}}` at all, so under the stated `{{KEY}}`
   grammar (`trinity_to_cloudy.py:293`) it could not match regardless of word boundaries. The stated
   reason does not follow from the stated grammar — one of the two descriptions is imprecise.

9. **Sidecar content.** `trinity_to_cloudy.py:6` says the sidecar holds "just the dlaw block", while
   `snapshot_to_deck.py:59` distinguishes `DLAW_BLOCK` (header+rows+footer) from `DLAW_ROWS` (rows).
   "just the dlaw block" does not disambiguate which is written.

---

## 8. Claims that are unfalsifiable or too vague to check as written

- "**best-guess** for CLOUDY C17/C22; see Step 5 smoke test" (`dlaw.py:1`) and "Override at call site
  if a live smoke test reveals a different working form" (`dlaw.py:31`) — the deck's core syntax is
  self-declared unverified, and the referenced verification is neither named nor its outcome
  reported. Nothing in the prose lets a reader determine whether the smoke test ever ran.
- "PL profiles are O(1); transition-phase IFs in TRINITY snapshots are O(1e5). 50 separates them with
  margin." (`dlaw.py:39`) — no measurement, dataset, or regime is cited; "with margin" is
  unquantified.
- "CLOUDY will fail loudly on an unknown atmosphere grid name, which is the safe failure."
  (`trinity_to_cloudy.py:71`) — a claim about third-party behaviour with no version or citation.
- "simplify preserves them **by contract**" (`snapshot_to_deck.py:59`) — the contract is not named or
  located; `simplify` is not identified.
- "will be removed in Phase 6" ×3 (`run_loader.py:1`, `:155`, `:189`) — no date, no owner, no
  definition of Phase 6.
- "reserved for future PCHIP-on-densBE support. Currently unused" (`dlaw.py:64`) — a parameter that
  the prose itself says does nothing.
- "the `DEFAULT_*` constants" (`__init__.py:1`) and "See module-level defaults" (`dlaw.py:64`) —
  advertised as the way to see the CLOUDY syntax defaults, but no default value is printed anywhere
  in the prose except `edge_threshold`'s 50 and the format example.
- "`# Only copy if missing or stale`" (`trinity_to_cloudy.py:440`) — "stale" is never defined
  (mtime? hash? size?).
- "`--age MYR` … picks closest snapshot" (`trinity_to_cloudy.py:6`) — closeness metric and tie-break
  unstated; `--phase NAME [--pick first|last]` gives no default for `--pick`.
- "`# _parse_args ensures we never reach here`" (`trinity_to_cloudy.py:229`) — an unreachability
  claim with no stated mechanism beyond the mutex note at `:159`.

---

```json
[
  {
    "id": "S13b-B-01",
    "file": "trinity/_output/cloudy/snapshot_to_deck.py",
    "line": 59,
    "class": "units",
    "severity": "S2",
    "claim": "The template keys LOG_RIN and LOG_ROUT are the radii CLOUDY integrates over, but the prose never states their unit (pc vs cm), never states that log10 is applied (only the key name implies it), and never mentions the pc-to-cm conversion for them. The +18.4894 offset is documented only inside dlaw.py, for the dlaw table columns, not for these keys.",
    "evidence": "L59-95 snapshot_to_deck.py: 'Substitution keys (TITLE, AGE_YR, LOG_QH, LOG_RIN, LOG_ROUT, ZREL, DLAW_BLOCK, DLAW_ROWS)'; L184: '# Outer radius: rShell unless user requested extension'. The only radius-unit claim in the slice is dlaw.py L64-92: 'r_in_pc, r_out_pc — Inner / outer radius CLOUDY will integrate over (pc).' and dlaw.py L173-174: '# --- 5. Convert pc -> cm and pc^-3 -> cm^-3 in log space' / '# +18.4894'.",
    "expected": "At the declared unit boundary of the package, each substitution key should carry an explicit unit and log convention in the docstring, e.g. 'LOG_RIN/LOG_ROUT: log10 of the inner/outer radius in cm'.",
    "failure_scenario": "If LOG_RIN/LOG_ROUT were emitted as log10(pc) while the dlaw table is in log10(cm), CLOUDY's radius command and the dlaw grid would be offset by 18.4894 dex, placing the integration volume entirely outside the density table. Conversely a reader maintaining the code has no documented invariant to preserve.",
    "repro": "Diff the prose against the implementation of snapshot_to_values: check whether LOG_RIN/LOG_ROUT apply +18.4894 (or log10 of a cm value) and whether the emitted radius command's units match the dlaw table's radius column.",
    "confidence": "high"
  },
  {
    "id": "S13b-B-02",
    "file": "trinity/_output/cloudy/dlaw.py",
    "line": 1,
    "class": "units",
    "severity": "S2",
    "claim": "The module docstring describes the output density column two different ways in the same docstring: as 'log10 n [cm^-3]' in prose and as 'log10(n_H/cm^-3)' in the format block. No composition / mean-molecular-weight / helium correction is claimed anywhere, so the conversion is presented as purely geometric (pc^-3 -> cm^-3).",
    "evidence": "L1-18 dlaw.py: 'The block converts (r [pc], log10 n [pc^-3]) pairs into the (log10 r [cm], log10 n [cm^-3]) form CLOUDY expects' ... 'continue {log10(r/cm):.6f} {log10(n_H/cm^-3):.4f}'. L173-175: '# --- 5. Convert pc -> cm and pc^-3 -> cm^-3 in log space', '# +18.4894', '# -55.4681'.",
    "expected": "The prose should state unambiguously whether TRINITY's shell number density is a total particle density or a hydrogen density, and whether any species factor is applied before writing the CLOUDY dlaw column (which is a hydrogen-density table).",
    "failure_scenario": "If TRINITY n is a total or electron number density and CLOUDY's dlaw table radius expects hydrogen density, every emitted density is wrong by the composition factor (order 0.05-0.15 dex for typical H/He mixes), silently biasing every derived line ratio in the exported deck.",
    "repro": "Compare the quantity fed into build_dlaw_block's shell_log_n_pc3 with TRINITY's own definition of the shell density array, and check for any multiplicative species factor between them.",
    "confidence": "medium"
  },
  {
    "id": "S13b-B-03",
    "file": "trinity/_output/cloudy/snapshot_to_deck.py",
    "line": 59,
    "class": "units",
    "severity": "S2",
    "claim": "The two density arrays entering dlaw are claimed to arrive in different log conventions (shell already log10 pc^-3, ambient linear pc^-3), and this asymmetry is documented only on the receiving side. snapshot_to_deck.py's prose states the log10 conversion for the ambient arm only and never states the shell array's convention.",
    "evidence": "dlaw.py L64-92: 'shell_r_pc, shell_log_n_pc3 — Shell radius (pc) and log10 number density (1/pc^3).' vs snapshot_to_deck.py L59-95: 'Pulls metadata.initial_cloud_{r,n}_arr (linear, in pc / pc^-3) and converts the density to log10 before handing to dlaw.' and L240: '# Convert linear pc^-3 -> log10 pc^-3 (eps guard for safety; TRINITY writes positive values, but defends against future regressions).'",
    "expected": "The caller should document the log convention of BOTH arrays it forwards, since the two arms differ and the splice concatenates them into one table.",
    "failure_scenario": "If the snapshot shell array is in fact linear pc^-3 (or is ever changed to be), it would be concatenated with log10 ambient values and written straight to the deck: a linear density of e.g. 1e6 pc^-3 would be interpreted as 10^(1e6) cm^-3 after the offset. The splice makes this a silent, position-dependent corruption rather than an outright error.",
    "repro": "Inspect what snapshot_to_values passes as shell_log_n_pc3 and whether the snapshot field is stored linear or log10; then check the spliced array for a convention mismatch at the shell/ambient junction.",
    "confidence": "high"
  },
  {
    "id": "S13b-B-04",
    "file": "trinity/_output/cloudy/dlaw.py",
    "line": 174,
    "class": "coefficient",
    "severity": "S4",
    "claim": "The two unit-conversion constants are given as independently rounded values that are not exactly 3x each other, although the prose presents them as the same pc->cm conversion applied to a length and to a volume density.",
    "evidence": "L173-175 dlaw.py: '# --- 5. Convert pc -> cm and pc^-3 -> cm^-3 in log space ------------------' / '# +18.4894' / '# -55.4681'. Three times 18.4894 is 55.4682, not 55.4681.",
    "expected": "Either derive the density offset as -3 * the radius offset from a single constant, or note in the comment that both are independently rounded from log10(3.0857e18) = 18.48935 (3x = 55.46806).",
    "failure_scenario": "If the implementation hardcodes the two rounded literals rather than deriving one from the other, the radius and density columns are converted with mutually inconsistent constants; the residual is ~1e-4 dex (negligible physically) but it defeats any bit-identical-equivalence gate and hides which constant is authoritative.",
    "repro": "Check whether the source defines a single PC_IN_CM-derived constant or two independent float literals; if two, confirm the 3x relation holds to machine precision.",
    "confidence": "medium"
  },
  {
    "id": "S13b-B-05",
    "file": "trinity/_output/cloudy/dlaw.py",
    "line": 1,
    "class": "citation",
    "severity": "S2",
    "claim": "The CLOUDY input syntax this module emits is self-declared a guess, and the slice contains no citation to the CLOUDY manual, Hazy, any section number, or any format specification anywhere.",
    "evidence": "L1-18 dlaw.py: 'Output format (defaults — best-guess for CLOUDY C17/C22; see Step 5 smoke test)'. L31-32: '# Best-guess CLOUDY syntax (Step 0 / Option B). Override at call site if a live smoke test reveals a different working form.' No other external reference appears in any of the five files; the only version string in the slice is 'C17/C22'.",
    "expected": "A module whose entire purpose is emitting third-party input syntax should cite the manual section defining 'dlaw table radius' / 'continue' / 'end of dlaw', and record whether the smoke test was run and against which CLOUDY build.",
    "failure_scenario": "Every emitted deck is unrunnable or silently misparsed if the guessed row grammar is wrong (e.g. if the first data row must not carry 'continue', or if 'end of dlaw' is not the accepted terminator in C22). Because the guess is never validated in-repo, the failure surfaces only in the user's CLOUDY run.",
    "repro": "Run a generated .in through CLOUDY C17 and C22 and confirm the dlaw block parses; compare the emitted grammar against the Hazy dlaw section.",
    "confidence": "high"
  },
  {
    "id": "S13b-B-06",
    "file": "trinity/_output/cloudy/trinity_to_cloudy.py",
    "line": 238,
    "class": "silent-failure",
    "severity": "S3",
    "claim": "The status-gate docstring states a blanket refusal rule and then enumerates only three exit-code bands as force-overridable, leaving 30-49, 60-98 and >=100 unspecified.",
    "evidence": "L238-244: 'Refuse to convert runs whose termination exit code is not in the clean range (0-9). Inspection-required (50-59 or 99) and error (10-29) outcomes both require --force. Source: bundle.end_state — metadata.json[termination] for v3+, legacy simulationEnd.txt otherwise.'",
    "expected": "The docstring should say what happens for every possible exit code, in particular whether an unrecognised code (30-49, 60-98, >=100, or a missing/None code) is treated as clean, as force-overridable, or as a hard refusal.",
    "failure_scenario": "If an unrecognised code falls through to the permissive branch, a run that terminated in an undocumented failure mode is exported to a CLOUDY deck without warning, and the resulting line predictions are published as if the run were clean.",
    "repro": "Enumerate the exit codes TRINITY can emit and trace each through _check_status; look for a fall-through else branch or a missing/None end_state path.",
    "confidence": "medium"
  },
  {
    "id": "S13b-B-07",
    "file": "trinity/_output/cloudy/run_loader.py",
    "line": 189,
    "class": "units",
    "severity": "S4",
    "claim": "The unit-suffix rule stated in the docstring is contradicted by the key list in the same docstring: five of the listed keys carry unit suffixes that match the declared AU convention rather than differing from it.",
    "evidence": "L189-209: 'Returned keys (units in the key name where they differ from the summary's AU = (Msun, pc, Myr) convention):: model_name, outcome, detail, exit_code, t_now_myr, R2_pc, shell_nMax_cm3, shell_v_kms, mCloud_msun, nCore_cm3, rCloud_pc, rCore_pc, alpha, nISM_cm3'.",
    "expected": "Either drop the 'where they differ' qualifier (all physical keys are suffixed), or drop the conforming suffixes. As written a reader cannot use the rule to infer the unit of an unsuffixed key.",
    "failure_scenario": "A future key added without a suffix would be read as AU by the stated rule, but the rule is demonstrably not followed, so the reader has no reliable way to infer its unit — the classic entry point for this repo's declared recurring unit bug class.",
    "repro": "Compare the actual returned dict keys and the units of the values parsed out of simulationEnd.txt against this list.",
    "confidence": "high"
  },
  {
    "id": "S13b-B-08",
    "file": "trinity/_output/cloudy/run_loader.py",
    "line": 189,
    "class": "units",
    "severity": "S3",
    "claim": "The slice carries two mutually inconsistent density conventions with no documented bridge: array densities are pc^-3, legacy scalar densities are cm^-3, and the declared house convention (Msun, pc, Myr) implies neither cm^-3 nor km/s.",
    "evidence": "run_loader.py L189-209: 'shell_nMax_cm3, shell_v_kms, ... nCore_cm3, ... nISM_cm3' under the header 'the summary's AU = (Msun, pc, Myr) convention'. Versus dlaw.py L1-18: '(r [pc], log10 n [pc^-3])' and snapshot_to_deck.py L59-95: 'metadata.initial_cloud_{r,n}_arr (linear, in pc / pc^-3)'.",
    "expected": "A single note in the package stating that scalar diagnostics are cgs-flavoured while profile arrays are pc-based, so that no one mixes an nISM_cm3 into a pc^-3 array.",
    "failure_scenario": "Splicing an ambient value taken from nISM_cm3 into the pc^-3 ambient array (or vice versa) misplaces the density by 55.468 dex with no validation able to catch it — the dlaw finite/NaN checks would pass.",
    "repro": "Check whether any code path feeds a *_cm3 scalar into the dlaw ambient arrays or compares it against initial_cloud_n_arr.",
    "confidence": "high"
  },
  {
    "id": "S13b-B-09",
    "file": "trinity/_output/cloudy/snapshot_to_deck.py",
    "line": 181,
    "class": "units",
    "severity": "S3",
    "claim": "LOG_QH's log convention is asserted only by the key name. The single comment on its computation states a rate conversion and says nothing about taking a logarithm, and the numeric Myr->s factor is never given.",
    "evidence": "L181: '# ph/Myr -> ph/s'. The key is listed as 'LOG_QH' at L59-95. No other prose in the slice mentions Q(H), ionizing photon rates, or a log10 of them.",
    "expected": "'LOG_QH: log10 of the ionizing photon rate in photons per second, converted from the snapshot's Qi in photons per Myr (divide by 3.156e13).'",
    "failure_scenario": "If log10 were omitted, CLOUDY's Q(H) command (which takes a log by default in most forms) would receive ~1e50 read as an exponent, or a linear rate read as a log — either way the ionizing budget is wrong by tens of dex and the deck fails or produces nonsense.",
    "repro": "Check whether snapshot_to_values applies log10 to the ph/s value and what the bundled .in template's Q(H)/ionization line expects.",
    "confidence": "high"
  },
  {
    "id": "S13b-B-10",
    "file": "trinity/_output/cloudy/snapshot_to_deck.py",
    "line": 59,
    "class": "units",
    "severity": "S3",
    "claim": "ZREL has no documented unit, normalisation, or log convention. The prose says only where the value comes from and which override wins.",
    "evidence": "L59-95: 'Z handling ---------- bundle.summary[\"ZCloud\"] by default; z_override (>0, finite) wins.' L200: '# Z scale'.",
    "expected": "State whether ZREL is a solar-relative linear scale factor, an absolute mass fraction, or a log10 value, and which CLOUDY command consumes it (metals / abundances / metals log).",
    "failure_scenario": "If ZCloud is an absolute metal mass fraction (e.g. 0.014) and the deck's command expects a solar-relative factor, every exported model runs at ~1.4 percent of the intended metallicity, changing cooling and every emitted line ratio, with no error raised.",
    "repro": "Compare ZCloud's definition in the run metadata with the CLOUDY command the template substitutes ZREL into.",
    "confidence": "high"
  },
  {
    "id": "S13b-B-11",
    "file": "trinity/_output/cloudy/snapshot_to_deck.py",
    "line": 256,
    "class": "other",
    "severity": "S3",
    "claim": "The DLAW_ROWS extraction is documented as stripping exactly one leading and one trailing line, while dlaw's open/close strings are documented as caller-overridable knobs with no single-line requirement.",
    "evidence": "snapshot_to_deck.py L256: '# Rows-only view: strip the first (header) and last (footer) lines.' and L259: '# excluding open/close'. Versus dlaw.py L64-92: 'dlaw_open, dlaw_row_prefix, dlaw_close — CLOUDY syntax knobs. See module-level defaults.' and dlaw.py L31-32: 'Override at call site if a live smoke test reveals a different working form.'",
    "expected": "Either document the invariant 'dlaw_open and dlaw_close must be single-line' on build_dlaw_block, or have build_dlaw_block return the rows separately instead of reconstructing them by line-slicing.",
    "failure_scenario": "The module explicitly anticipates overriding the syntax after a smoke test. A two-line open (plausible for CLOUDY variants) would leave a stray 'dlaw ...' command inside DLAW_ROWS and silently drop the first density row from any template using DLAW_ROWS.",
    "repro": "Set dlaw_open to a two-line string and inspect DLAW_ROWS.",
    "confidence": "medium"
  },
  {
    "id": "S13b-B-12",
    "file": "trinity/_output/cloudy/snapshot_to_deck.py",
    "line": 34,
    "class": "state",
    "severity": "S2",
    "claim": "The prose asserts a non-terminating-loop hazard in the snapshot type: membership testing with 'in' on TrinityOutput.Snapshot never terminates, and the sentinel/.get idiom exists solely to avoid it.",
    "evidence": "L34-36: '# Sentinel for \"key absent\" — needed because TrinityOutput.Snapshot has __getitem__ but no __contains__/__iter__, so `key in snap` falls back to integer-indexed iteration that never terminates. Use snap.get(k, _MISSING).'",
    "expected": "This is a documented landmine in a shared type, not a local quirk. It belongs on TrinityOutput.Snapshot itself (or the type should define __contains__), not only in a comment inside the CLOUDY exporter.",
    "failure_scenario": "Any other caller anywhere in the codebase writing `if 'key' in snapshot` hangs forever with no traceback — a hung sweep worker or SLURM job rather than a failure.",
    "repro": "Grep for `in snap` / `in snapshot` membership tests outside this module; check whether TrinityOutput.Snapshot defines __contains__ or __iter__.",
    "confidence": "high"
  },
  {
    "id": "S13b-B-13",
    "file": "trinity/_output/cloudy/snapshot_to_deck.py",
    "line": 154,
    "class": "numerical",
    "severity": "S3",
    "claim": "Validation step 6 describes one comparison two incompatible ways in a single sentence: a relative tolerance of 1e-12 and an exact-equality test.",
    "evidence": "L59-95: '6. shell_r_arr endpoints match R2 / rShell (rel_tol=1e-12) — simplify preserves them by contract; an exact-equality drift would indicate upstream regression.'",
    "expected": "State which test is performed. If the endpoints are preserved bit-exactly by contract, say the check is exact; if a tolerance is used, do not describe the failure mode as exact-equality drift.",
    "failure_scenario": "A maintainer tightening the check to exact equality on the strength of the second clause would start rejecting valid snapshots; one loosening it on the strength of the first would let a real upstream regression through. The bracket check downstream is described only as 'with tiny float tolerance' (dlaw.py L161), so the two tolerances are also unrelated in the prose.",
    "repro": "Read the endpoint assertion and the dlaw bracket tolerance; check whether r_in/r_out equal the array endpoints exactly or within a tolerance, and whether the two tolerances are compatible.",
    "confidence": "medium"
  },
  {
    "id": "S13b-B-14",
    "file": "trinity/_output/cloudy/dlaw.py",
    "line": 161,
    "class": "regime",
    "severity": "S3",
    "claim": "The bracket precondition is a hard 'must', and the default configuration is claimed to place r_out exactly at the array endpoint, so the check is guarded only by an unquantified 'tiny float tolerance'.",
    "evidence": "dlaw.py L64-92: 'r_in_pc, r_out_pc — Inner / outer radius CLOUDY will integrate over (pc). Must lie within the union of shell + ambient r-range.' dlaw.py L161: '# --- 4. Bracket check (with tiny float tolerance) -----------------------'. snapshot_to_deck.py L184: '# Outer radius: rShell unless user requested extension' and L59-95 step 6: 'shell_r_arr endpoints match R2 / rShell (rel_tol=1e-12)'.",
    "expected": "The tolerance value should be stated, and the docstring should note that the default case sits exactly on the boundary so the tolerance is load-bearing rather than defensive.",
    "failure_scenario": "In the default (non-extended) case r_in == shell_r_arr[0] and r_out == shell_r_arr[-1] to within 1e-12 relative; a tolerance smaller than that drift makes ordinary runs raise DlawError, i.e. the common path fails on a rounding artefact rather than a physics problem.",
    "repro": "Run the default no-ambient path and check whether the bracket comparison is <=, <= with epsilon, or math.isclose, and what epsilon it uses relative to 1e-12.",
    "confidence": "medium"
  },
  {
    "id": "S13b-B-15",
    "file": "trinity/_output/cloudy/dlaw.py",
    "line": 64,
    "class": "deadcode",
    "severity": "S4",
    "claim": "build_dlaw_block accepts a dens_profile parameter that the docstring says is unused.",
    "evidence": "L64-92: 'dens_profile — TRINITY profile shape; reserved for future PCHIP-on-densBE support. Currently unused; densification is linear-in-(log r, log n).'",
    "expected": "Per the repo's stated simplicity rule (no speculative features), either the parameter is removed or its presence is justified. Flagging only, per the rule that pre-existing dead code is reported rather than deleted.",
    "failure_scenario": "A caller passes dens_profile expecting PCHIP interpolation on a Bonnor-Ebert profile and silently gets linear-in-log densification instead, with no warning that the argument was ignored.",
    "repro": "Check whether dens_profile is referenced anywhere in the function body.",
    "confidence": "high"
  },
  {
    "id": "S13b-B-16",
    "file": "trinity/_output/cloudy/run_loader.py",
    "line": 35,
    "class": "divergence",
    "severity": "S3",
    "claim": "The density-profile enum is a hand-maintained mirror of a validator that lives in a different package, and load_run is documented to hard-fail on any value not in the mirror.",
    "evidence": "L35: '# Canonical TRINITY density-profile enum (mirrors _validate_dens_profile in trinity/_input/registry.py).' L58-77: 'Raises ------ RunLoadError If any expected file is missing, malformed, or carries an unknown dens_profile.'",
    "expected": "Import the canonical set from trinity/_input/registry.py rather than duplicating it, or add a test asserting the two sets are equal.",
    "failure_scenario": "Adding a new density profile in _input/registry.py without updating this mirror makes every run using it unloadable by the CLOUDY exporter — RunLoadError on a perfectly valid run, discovered only at export time.",
    "repro": "Diff the literal set in run_loader.py against the accepted values in trinity/_input/registry.py::_validate_dens_profile.",
    "confidence": "high"
  },
  {
    "id": "S13b-B-17",
    "file": "trinity/_output/cloudy/trinity_to_cloudy.py",
    "line": 76,
    "class": "other",
    "severity": "S4",
    "claim": "The stated reason the <<<EDIT_ME>>> sentinel survives rendering does not follow from the stated placeholder grammar.",
    "evidence": "L76-77: '# A {{KEY}} placeholder. Word-boundary match means <<<EDIT_ME>>> is invisible to the renderer (passes through unchanged).' L293-297: 'Substitute {{KEY}} placeholders. Raise UnsubstitutedPlaceholder on any {{KEY}} left after substitution. Sentinels not matching the {{KEY}} pattern (notably <<<EDIT_ME>>>) pass through unchanged.'",
    "expected": "The sentinel survives because it has no braces, not because of word boundaries. The comment should say what the word-boundary component is actually for (presumably restricting KEY to word characters).",
    "failure_scenario": "A maintainer loosening the pattern to catch other placeholder styles could reasonably conclude the word-boundary is what protects <<<EDIT_ME>>> and preserve the wrong property, causing the sentinel to be consumed and a deck to be shipped with a bogus atmosphere grid name that CLOUDY may or may not reject.",
    "repro": "Read the placeholder regex and confirm what the word-boundary assertion constrains.",
    "confidence": "medium"
  },
  {
    "id": "S13b-B-18",
    "file": "trinity/_output/cloudy/dlaw.py",
    "line": 39,
    "class": "numerical",
    "severity": "S4",
    "claim": "The IF/edge detection threshold of 50 is justified purely by unquantified order-of-magnitude assertions with no cited measurement, and the log-space in which the ratio is evaluated is not stated.",
    "evidence": "L39-41: '# |delta log n / delta log r| above this counts as an IF-like discontinuity. PL profiles are O(1); transition-phase IFs in TRINITY snapshots are O(1e5). 50 separates them with margin.'",
    "expected": "Cite the run/config the O(1) and O(1e5) figures came from, and state whether the ratio is computed before or after the pc->cm conversion (the pipeline comments place densification after step 5, so presumably after).",
    "failure_scenario": "A regime the author did not sample (e.g. a steep power-law index, or a partially-resolved front) sits between 1 and 50 and is silently treated as smooth, so densification interpolates across a real discontinuity and smears the ionization front in the exported density law.",
    "repro": "Compute |dlog n/dlog r| across snapshots from param/simple_cluster.param and the f1edge configs; histogram it and check the 1-to-50 gap actually holds.",
    "confidence": "medium"
  },
  {
    "id": "S13b-B-19",
    "file": "trinity/_output/cloudy/trinity_to_cloudy.py",
    "line": 6,
    "class": "other",
    "severity": "S3",
    "claim": "The CLOUDY command set and command order of the emitted deck are documented nowhere in the slice; only the eight substitution keys and one 'table star' line are named. The actual contract lives in a bundled .in template not covered by any prose here.",
    "evidence": "trinity_to_cloudy.py L1-30 names only the output filenames and the table star sentinel; snapshot_to_deck.py L1-14: 'The CLI (Step 4) calls this, then substitutes the returned dict into the bundled .in template and writes the deck and sidecar dlaw .txt.' No file in the slice enumerates radius / Q(H) / metals / age / stop / save commands or their order.",
    "expected": "The deck contract (which commands, in what order, with which units) should be documented next to the code that fills it in, since that is what determines whether the substitution keys' units are right.",
    "failure_scenario": "Because the consuming commands are undocumented, a unit or log-convention error in LOG_RIN/LOG_ROUT/LOG_QH/ZREL cannot be caught by reading the exporter — the only place the mismatch is visible is an untracked template file plus the CLOUDY manual.",
    "repro": "Open the bundled .in template and list its commands and the units each expects; compare against the substitution keys' claimed units.",
    "confidence": "high"
  },
  {
    "id": "S13b-B-20",
    "file": "trinity/_output/cloudy/run_loader.py",
    "line": 155,
    "class": "other",
    "severity": "S4",
    "claim": "The documented scalar-coercion type list disagrees with the inline comment describing the same step: the docstring omits dicts, the comment includes them.",
    "evidence": "L155-165: 'Values are coerced (in order): bool, None, nan/inf, int, float, Python-literal (lists, tuples), else string.' versus L317-318: '# Python literal (lists, tuples, dicts) — must start with a recognisable literal opener, otherwise we'd accept arbitrary expressions.'",
    "expected": "One list, in one place. If dicts are accepted, the docstring should say so, since this function parses a config-shaped file.",
    "failure_scenario": "A legacy summary value written as a dict literal either parses (contradicting the docstring) or falls through to str (contradicting the comment); a caller relying on the documented type set gets a str where it expected a mapping.",
    "repro": "Feed '{\"a\": 1}' to _coerce_scalar and observe the returned type.",
    "confidence": "high"
  },
  {
    "id": "S13b-B-21",
    "file": "trinity/_output/cloudy/trinity_to_cloudy.py",
    "line": 6,
    "class": "other",
    "severity": "S4",
    "claim": "The snapshot-selection semantics are underspecified: --age is documented as picking the 'closest' snapshot with no metric or tie-break, and --phase's --pick is shown as optional with no stated default.",
    "evidence": "L1-30: '--age MYR cluster age (Myr since tSF) — picks closest snapshot ... --phase NAME [--pick first|last]'. L190: 'Resolve the picker flags into a list of (index, snapshot) tuples.' L221-222: '# filter() re-indexes from 0; map back to the original index by round-tripping through get_at_time on the unfiltered output.'",
    "expected": "State the closeness metric (absolute difference in cluster age), the tie-break rule, and the default for --pick.",
    "failure_scenario": "Two users invoking the same --age on the same run at different times, or after a re-run with slightly different output cadence, silently export different snapshots; the deck filename embeds the requested age, not the selected snapshot's actual age, so the divergence is invisible in the filename.",
    "repro": "Request an --age exactly midway between two snapshots and check which index is selected; omit --pick with --phase and check the default.",
    "confidence": "high"
  },
  {
    "id": "S13b-B-22",
    "file": "trinity/_output/cloudy/dlaw.py",
    "line": 1,
    "class": "other",
    "severity": "S4",
    "claim": "Multiple documented behaviours are pinned to an unnamed external plan document ('Step 0', 'Step 4', 'Step 5 smoke test', 'Option B', 'Phase 5', 'Phase 6', 'pre-Phase-2') that is never identified, so none of them can be checked or scheduled from the code alone.",
    "evidence": "dlaw.py L1-18: 'best-guess for CLOUDY C17/C22; see Step 5 smoke test'; dlaw.py L31-32: '(Step 0 / Option B)'; snapshot_to_deck.py L1-14: 'The CLI (Step 4) calls this'; run_loader.py L1-20: 'Legacy runs (pre-Phase-5) ... they will be removed in Phase 6'; run_loader.py L124-128: 'this fallback is scheduled for removal once existing runs are re-processed'.",
    "expected": "Name the document (path under docs/dev/ or an issue number) or restate the content inline. Per the repo's own note, docs/dev/ write-ups are unverified point-in-time analyses, so an unnamed pointer is doubly unresolvable.",
    "failure_scenario": "The deprecation removals and the CLOUDY-syntax verification both become permanently unactionable: nobody can tell whether Phase 6 happened or whether the smoke test was ever run, so the legacy text parsers and the guessed syntax persist indefinitely.",
    "repro": "Grep docs/dev/ for a CLOUDY-export plan naming Steps 0-5 and Phases 5-6; check whether Phase 5/6 have shipped.",
    "confidence": "high"
  },
  {
    "id": "S13b-B-23",
    "file": "trinity/_output/cloudy/__init__.py",
    "line": 1,
    "class": "other",
    "severity": "S4",
    "claim": "The package docstring directs users to the DEFAULT_* constants for finer-grained access, but no DEFAULT_* value is documented anywhere in the slice except the edge threshold, and dlaw's parameter docs defer to 'module-level defaults' that are never printed.",
    "evidence": "__init__.py L1-14: 'Sub-modules can also be imported directly when finer-grained access is needed (e.g. the DEFAULT_* constants).' dlaw.py L64-92: 'dlaw_open, dlaw_row_prefix, dlaw_close — CLOUDY syntax knobs. See module-level defaults.' and 'min_rows — If the post-splice profile has fewer rows, densify ...' (no default given). Only dlaw.py L39-41 states a value ('50').",
    "expected": "State the default values (min_rows, dlaw_open/row_prefix/close) in the prose, since they define the emitted CLOUDY syntax and the row count of every deck.",
    "failure_scenario": "A user overriding dlaw syntax after a failed CLOUDY run cannot tell what they are overriding, and cannot reproduce the shipped default; the min_rows default silently determines how finely the density law is sampled, which affects CLOUDY's zoning.",
    "repro": "Read the DEFAULT_* assignments in dlaw.py and compare against the documented format example.",
    "confidence": "high"
  }
]
```
