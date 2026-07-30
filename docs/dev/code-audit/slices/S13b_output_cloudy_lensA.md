# S13b output CLOUDY export — Lens A (what the code does)

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

## What I read

Stripped slice only, at `.../lens/S13b_output_cloudy/code/_output/cloudy/`:

- `trinity/_output/cloudy/trinity_to_cloudy.py` (511 lines)
- `trinity/_output/cloudy/snapshot_to_deck.py` (301 lines)
- `trinity/_output/cloudy/run_loader.py` (346 lines)
- `trinity/_output/cloudy/dlaw.py` (270 lines)
- `trinity/_output/cloudy/__init__.py` (36 lines)

**Shared exception used:** yes. I read the real `/home/user/trinity/trinity/_functions/unit_conversions.py` to pin the numeric values of `INV_CONV.pc2cm`, `INV_CONV.Myr2s`, `INV_CONV.ndens_au2cgs`, and the declared astro-unit system (Msun / pc / Myr). Nothing else from the real tree was read.

**Not read:** the real `trinity/` tree (other than the one file above), `docs/dev/`, `test/`, `param/`, `outputs/`, this slice's `prose.md` and `signatures.md`, any other agent's report.

**Structural blind spot you must weigh in every deck-syntax claim below.** The deck template itself — `trinity2cloudy.in_template`, referenced at `trinity/_output/cloudy/trinity_to_cloudy.py:67` — is **not in my slice**. Neither is `trinity_linelist.dat` (`:68`). So the *only* CLOUDY command syntax actually generated inside this slice is the `dlaw` block (`trinity/_output/cloudy/dlaw.py:198-202`). Everything else — which CLOUDY command consumes `LOG_RIN`/`LOG_ROUT`/`LOG_QH`/`AGE_YR`/`ZREL`, in what order, and whether `DLAW_BLOCK` and `DLAW_ROWS` are both interpolated (which would duplicate the table) — lives in the template. I audit the *values and their string forms*; I cannot audit command ordering. Every claim that depends on the template is marked medium or low confidence.

Also outside the slice and therefore assumed, not verified: `TrinityOutput` / `find_data_path` (`run_loader.py:31`), `metadata_keys_to_rehydrate` (`:32`), `bundle.output.initial_cloud_profile()` (`snapshot_to_deck.py:230`), `get_at_time(..., quiet=True)` (`trinity_to_cloudy.py:196,200,225`), and the physical meaning/unit of the snapshot fields `log_shell_n_arr`, `Qi`, `ZCloud`.

---

## 1. Pipeline shape

`main()` (`trinity_to_cloudy.py:330-419`) is: parse args → `load_run` → `_check_status` → `_pick_snapshots` → load template + linelist → for each pick `snapshot_to_values` → `render_template` → `_write_outputs` → `_copy_linelist` → optional `_write_manifest` → `_print_summary` → `return 0`.

`snapshot_to_values` (`snapshot_to_deck.py:47-291`) is the whole numeric boundary. It returns a flat `dict[str, str]` of template substitution values plus a `_diagnostics` sub-dict. `build_dlaw_block` (`dlaw.py:49-202`) is the only function that emits CLOUDY text.

**No in-place mutation anywhere.** `end_state = dict(output.termination)` copies (`run_loader.py:130`). `np.asarray(...)` on the snapshot arrays (`snapshot_to_deck.py:113-114`) plus fancy indexing in `dlaw.py:126-131` produce copies; the ambient arrays from `initial_cloud_profile()` are bound (`snapshot_to_deck.py:243`) but only read and re-indexed (`dlaw.py:150-155`), never written. `values["PREFIX"]` / `values["SB99_MOD"]` (`trinity_to_cloudy.py:365-366`) mutate a dict the callee just built and no one else holds. Module-level state is limited to constants and compiled regexes (`trinity_to_cloudy.py:66-78,263`; `dlaw.py:33-42`); nothing accumulates across snapshots except the local `records` list.

---

## 2. Dimensional trace

The two conversion constants used, evaluated:

```
INV_CONV.pc2cm         = 3.0856775814913674e+18   log10 = 18.489350545222138
INV_CONV.Myr2s         = 3.15576e+13              log10 = 13.499103967085228
INV_CONV.ndens_au2cgs  = 1/2.937998946096347e+55  log10 = -55.46805163566641
```

**Positive result worth recording: the length unit is bit-consistent between the two dlaw columns.** `3 * log10(pc2cm) + log10(ndens_au2cgs) == 0.0` **exactly** in float64 (`3*18.489350545222138 = 55.46805163566641`, and `log10(ndens_au2cgs) = -55.46805163566641`). So the parsec implied by the radius column (`dlaw.py:176`) and the parsec implied by the density column (`dlaw.py:177`) are the same parsec to the last bit. There is no hidden cm/pc skew between the columns.

### Per-quantity table

| Quantity | TRINITY-side unit | Site | Transform applied | CLOUDY-side unit | Emitted format |
|---|---|---|---|---|---|
| shell radii `shell_r_arr` | pc | `dlaw.py:174,176` | `log10(r_pc) + 18.489350545222138` | `log10(r / cm)` | `%.6f` (`:200`) |
| shell density `log_shell_n_arr` | `log10(n / pc⁻³)` (assumed) | `dlaw.py:175,177` | `+ (-55.46805163566641)` — **no** log10 re-application, input already log | `log10(n / cm⁻³)` | `%.4f` (`:200`) |
| ambient radii | pc (assumed) | `snapshot_to_deck.py:243` → `dlaw.py:176` | pass-through, then same `+18.4894` | `log10(r / cm)` | `%.6f` |
| ambient density | **linear** `n / pc⁻³` (assumed) | `snapshot_to_deck.py:244` | `log10(max(n, 2.2251e-308))`, then `+(-55.468)` in dlaw | `log10(n / cm⁻³)` | `%.4f` |
| `R2` → `LOG_RIN` | pc | `snapshot_to_deck.py:182` | `log10(R2) + 18.489350545222138` | `log10(r_in / cm)` | `%.4f` (`:272`) |
| `r_out_pc` → `LOG_ROUT` | pc | `snapshot_to_deck.py:198` | `log10(r_out) + 18.489350545222138` | `log10(r_out / cm)` | `%.4f` (`:273`) |
| `Qi` → `LOG_QH` | photons Myr⁻¹ (assumed) | `snapshot_to_deck.py:181` | `log10(Qi) − 13.499103967085228` | `log10(photons s⁻¹)` | `%.4f` (`:271`) |
| `t_now − tSF` → `AGE_YR` | Myr | `snapshot_to_deck.py:167-168` | `× 1.0e6` | yr | `%.4e` (`:270`) |
| `ZCloud` → `ZREL` | Zsun, linear (assumed) | `snapshot_to_deck.py:212,274` | **none** | linear relative-to-solar (assumed) | `%.4f` |
| age in `TITLE` | Myr | `snapshot_to_deck.py:264` | none | free text | `%.4f` |
| age in filename prefix | Myr | `trinity_to_cloudy.py:279` | none, `"."→"p"` | filename token | `%.4f` |

### Arithmetic checks

- **Radius, dlaw:** `log10(r_cm) = log10(r_pc · 3.0857e18) = log10(r_pc) + 18.48935…`. `dlaw.py:176` does exactly this. Correct, single conversion, no double-apply.
- **Density, dlaw:** input is already log10. `dlaw.py:177` adds the log offset instead of calling `log10` again — i.e. `log10(n_cm⁻³) = log10(n_pc⁻³) + log10(3.4036e-56)`. Correct. **No log10 is applied to an already-log quantity, and none is omitted.**
- **Ambient density:** input is linear (`snapshot_to_deck.py:244` calls `np.log10` on it), then the *same* offset is added inside `build_dlaw_block`. So the two density paths reach the same CLOUDY unit by different routes, and — importantly — the offset is applied exactly once on each. No double conversion. The asymmetry is only in *where* the log is taken.
- **`Qi`:** `log_qh = log10(Qi) − log10(Myr2s)` = `log10(Qi/3.15576e13)`. This is right **iff** `Qi` is a rate per Myr. Sanity magnitude: a 1e6 Msun cluster has Q ≈ 1e53 s⁻¹ ⇒ 3.16e66 Myr⁻¹ ⇒ `log_qh = 66.5 − 13.5 = 53.0`. Self-consistent. If `Qi` were already s⁻¹ the deck would emit `log Q(H) ≈ 39.5`, an obvious 13.5-dex failure rather than a subtle one.
- **Age:** the only ×1e6. `age_myr = t_now − tSF` (`:167`); the *same* `age_myr` (not `age_yr`) is reused for the TITLE and the filename, so those are in Myr and `AGE_YR` is in yr — consistent, no doubled 1e6.
- **`ZCloud`:** zero conversion applied. The value from `bundle.summary["ZCloud"]` is formatted straight through.

**No quantity is converted on one path and not its twin**, with one exception in kind rather than value: `log_pc_per_cm = math.log10(INV_CONV.pc2cm)` is computed independently at `dlaw.py:174` and `snapshot_to_deck.py:180`. I checked: both are the identical expression and therefore the identical float. It is a duplicated constant (the brief asks about exactly this) but **not** a numeric divergence today. The name is backwards — `pc2cm` is *cm per pc*, so `log_pc_per_cm` holds log10(cm/pc) (finding A-14).

---

## 3. `dlaw` table construction (`dlaw.py`)

**Emitted syntax** (`:198-202`):

```
dlaw table radius
continue 19.188321  3.5319
...
end of dlaw
```

Open/row-prefix/close come from module constants `DEFAULT_DLAW_OPEN = "dlaw table radius"`, `DEFAULT_DLAW_ROW_PREFIX = "continue "`, `DEFAULT_DLAW_CLOSE = "end of dlaw"` (`:33-35`), overridable per call (`:59-61`) but never overridden by any caller in the slice. Rows are `f"{dlaw_row_prefix}{lr:.6f}  {ln:.4f}"` — two spaces between columns, no trailing newline on the last line (the caller adds one, `trinity_to_cloudy.py:434`).

**Grid convention:** the table is *radius*, not depth, and it is the shell's own sample points — there is no re-gridding, no linear/log resampling to a fixed grid. The abscissa is whatever `shell_r_arr` contains, sorted ascending. The table therefore spans `[R2, rShell]` (endpoint-checked, see below), optionally extended to `r_out_pc` with ambient points.

**Endpoint handling / coverage guarantee** (`:162-171`): the table must cover the model's radius range, checked with `rel_tol = 1e-12`: error if `r_pc[0] > r_in_pc*(1+tol)` or `r_pc[-1] < r_out_pc*(1-tol)`. Because `snapshot_to_deck.py:155-164` separately asserts `shell_r[0] == R2` and `shell_r[-1] == rShell` to `rel_tol=1e-12`, the default (`r_out_pc = rShell`) always passes exactly. **There is no extrapolation and no clamping beyond the tabulated range** — the code raises `DlawError` instead. That is the right shape.

**Monotonicity** (`:192`): `np.all(np.diff(log_r_cm) > 0)` after construction, else `DlawError`. Note this is checked on the *float* values, not on the strings that are actually written. See finding A-02.

**Sort + dedup** (`:125-131`): `np.argsort(kind="stable")`, then `keep[:-1] = r_pc[:-1] != r_pc[1:]` — exact float inequality. Duplicate radii are dropped keeping the **last** (outermost) of each run. Both operations are silent: a non-monotonic input array is silently reordered, and a duplicated radius with a different density silently loses one of the two densities. No warning (finding A-11).

**Ambient splice** (`:134-159`): only entered if either ambient array is non-`None`; both-or-neither is enforced (`:135-138`). The splice itself is further gated on `r_out_pc > r_pc[-1] and a_r.size` (`:149`) — so passing ambient arrays that aren't needed is silently a no-op. The mask is `(a_r > r_pc[-1]) & (a_r <= r_out_pc)` (`:156`), i.e. strictly outside the shell and not past `r_out`. If the ambient grid has no point at or beyond `r_out_pc`, the coverage check at `:167` then fires with a message telling you to supply longer ambient arrays — loud, correct.

**Densification** (`:180-185`, `_densify_preserving_edges` `:205-269`): triggered only when the table has fewer than `min_rows` (default 10, `:37`). It classifies each interval by `|Δlog n / Δlog r| <= edge_threshold` (default 50.0, `:42`, `:226`) and inserts points **only inside "smooth" intervals**, allocating the deficit proportionally to smooth interval length with a largest-remainder rule (`:239-249`). Inserted points are `np.linspace` in log r with `np.interp` in log n (`:260-265`) — i.e. exactly on the log–log straight line between the bracketing rows. Since a CLOUDY `dlaw table` is itself interpolated log–log between entries, the inserted rows are (to rounding) physically a no-op; densification changes row count, not the profile.

If **no** interval is smooth, it warns and returns the original table (`:229-236`), and `min_rows` is silently not met — the only hard floor is 2 rows (`:188-191`). I measured when that happens: for a 0.1 %-thick shell (`r = 5.0 … 5.005 pc`) carrying a 2-dex density ramp over 4 points, the three slopes are `7197, 4743, 2033` — all ≫ 50, so **every** interval is an "edge" and densification is skipped entirely (finding A-05).

---

## 4. Snapshot selection & loading

`load_run` (`run_loader.py:57-147`) is strict and loud: missing dir → `FileNotFoundError`; missing/malformed `metadata.json` → `RunLoadError`; missing `model_name` → `RunLoadError`; `dens_profile` not in `{densBE, densPL}` → `RunLoadError` (`:94-99`).

Two schema branches:
- `metadata["_metadata_version"] >= 2` (default 1, `:106`) → `summary = metadata_keys_to_rehydrate(metadata)`;
- else → parse `<model>_summary.txt` with a `DeprecationWarning` (`:109-114`, `:166-172`).

Same pattern for the end state: `output.termination` if present, else parse `simulationEnd.txt` (`:129-138`, `:210-217`). Note `metadata.get("_metadata_version", 1) >= 2` compares a JSON value to an int — a string version would raise `TypeError`, not route to the legacy path.

The two legacy parsers do **not** agree on unit-suffixed values: `_parse_simulation_end` explicitly takes `value.split()[0]` before `float()` (`:264`), while `_parse_summary_txt` hands the whole remainder to `_coerce_scalar` (`:179-184`), which for `"0.02 Zsun"` falls through every branch and returns the raw string (`:313-324`). That string then reaches `float(bundle.summary["ZCloud"])` at `snapshot_to_deck.py:212` (finding A-09).

`_check_status` (`trinity_to_cloudy.py:237-256`) requires `isinstance(exit_code, int) and 0 <= exit_code <= 9`, else refuses unless `--force`. Fails closed (a missing or non-int exit code refuses). Two curiosities: `outcome = ... or "unknown"` (`:246`) swallows `""`/`0`/`None` alike but is message-only; and `isinstance(True, int)` is `True` in Python, so a JSON `true` exit code would be read as "clean".

**Snapshot pickers** (`_pick_snapshots`, `:189-230`): exactly one of `--age` / `--t-now` / `--index` / `--phase` / `--all`, enforced at `:160-171`.

- `--age`: `target_t = args.age + float(bundle.metadata["tSF"])` (`:195`) — unguarded `KeyError` if `tSF` is absent, whereas `snapshot_to_deck.py:120-121` raises a clean `SnapshotInvalid` for the same missing key.
- `--age` / `--t-now`: `get_at_time(target_t, mode="closest", quiet=True)`. `mode="closest"` never fails — it returns the nearest snapshot however far away it is — and `quiet=True` suppresses whatever the reader would have said about that. Only the `--age` path prints a Δ afterwards (`:481-487`); `--t-now` prints nothing (finding A-10).
- `--index`: manual negative-index arithmetic (`:205`) with a range check.
- `--phase`: filters by phase, takes `filtered[-1]` or `filtered[0]`, reads its `["t_now"]`, and then **re-resolves it through `get_at_time(..., mode="closest")`** rather than using the filtered snapshot object directly (`:214-227`). If two snapshots share a `t_now`, the round-trip can return the other one — possibly in a different phase — and the deck's TITLE and filename would then carry that other phase (finding A-13).
- `--all`: every snapshot.

**Malformed snapshot handling** (`snapshot_to_deck.py:97-102`): presence is tested with a `_MISSING` sentinel via `snap.get(k, _MISSING) is _MISSING`, so a key present with value `None` counts as present and then hits `float(None)` → `TypeError` at `:105-108`. `TypeError` is not in the `--all` skip handler's tuple (`trinity_to_cloudy.py:393`), so one such snapshot aborts the whole batch after some decks are already on disk (finding A-17).

---

## 5. Validation, clamps and swallowed failures

Loud and correct (no complaint):
- non-finite `t_now/R2/rShell/Qi` (`snapshot_to_deck.py:109-111`), non-finite shell arrays (`:115-118`), `t_now <= tSF` (`:127`), `R2 <= 0`, `rShell <= R2`, `Qi <= 0` (`:134-141`), array length/size (`:144-152`), endpoint identity to 1e-12 (`:155-164`), `radius_out_pc < rShell` (`:192`), all of `dlaw.py:94-122`.

Swallowed or softened:
1. **Age band** (`:169-177`): outside `[1e5, 1e8]` yr it is a `warnings.warn`, not an error, unless `--hard-age-bounds`. Nothing is clamped — the deck is emitted with the out-of-band age and CLOUDY is left to extrapolate the SB99 grid. This is explicit and surfaced (CLI flag + printed warning), so I treat it as designed behaviour, not a defect — but with `--all` on a run whose early snapshots are at ~1e3 yr, every one of those decks is emitted.
2. **Ambient zero density** clamped to `np.finfo(float).tiny` (`:242-244`). Measured: that clamp produces a dlaw row of `continue <logr>  -363.1207` — a finite value that passes the `isfinite` guard at `dlaw.py:194` and is written out (finding A-06).
3. **`--all` per-snapshot skip** (`trinity_to_cloudy.py:393-410`) catches only `SnapshotInvalid | DlawError | UnsubstitutedPlaceholder`. `ValueError`/`TypeError`/`KeyError`/`OSError` propagate. (Both custom errors subclass `ValueError` — `dlaw.py:45`, `snapshot_to_deck.py:43` — but the tuple names them explicitly, so a bare `ValueError` is not caught.)
4. **`_parse_simulation_end` numeric fallback** `out[out_key] = None` on `ValueError|IndexError` (`run_loader.py:266-268`) — an unparseable `Exit Code` becomes `None`, which `_check_status` then treats as unclean, i.e. it degrades safely.
5. **`_coerce_scalar`'s final `return s`** (`run_loader.py:324`) — anything unparseable silently stays a string, which is how the unit-suffix divergence above becomes a crash later rather than there.
6. **`main()` returns 0** unconditionally on the non-dry path (`:419`), including when every `--all` snapshot was skipped, and including when `picks` is empty (an empty stream yields `records == []`, a `manifest.json` of `[]`, and `Converted 0 snapshots (0 skipped).`).
7. **`render_template`'s leftover check** (`:304-308`) only detects `{{...}}`. The default `--sb99` value is the sentinel `<<<EDIT_ME>>>` (`:74`), which is substituted *successfully*, so the check passes and a deck is written that CLOUDY cannot run. A TODO is printed (`:455-463`) — surfaced, but the file on disk is broken by default.

---

## 6. Numerical hygiene and emitted-string precision

This is where I think the real risk of this slice sits, and it is measurable.

- **Two precisions for the same physical radius.** The dlaw radius column is `%.6f` in log10 cm (`dlaw.py:200`) ⇒ resolution `10^1e-6 − 1 = 2.30e-6` relative. `LOG_RIN`/`LOG_ROUT` are `%.4f` (`snapshot_to_deck.py:272-273`) ⇒ half-bucket error up to `10^5e-5 − 1 = 1.151e-4` relative — **100× coarser**, and in an unpredictable direction. Worked example, `R2 = 0.5 pc`: true `log_rin = 18.188320549558156`; the deck gets `18.1883` (= 0.4999763 pc) while the first dlaw row gets `18.188321` (= 0.5000000 pc). The declared inner radius is 4.7e-5 *below* the first tabulated radius (finding A-01).
- **Inner and outer radius can collapse to the same string.** Measured with `R2 = 5.0 pc`: relative shell thickness `2e-5` → `LOG_RIN = LOG_ROUT = 19.1883`; `5e-5` → still equal; `1e-4` → `19.1883` vs `19.1884`. Collapse is guaranteed below ~2.3e-5 relative thickness and possible up to 2.3e-4 (finding A-01).
- **Duplicate radius rows in the table.** The dedup at `dlaw.py:129` is exact float `!=` and the monotonicity guard at `:192` is on floats, but the file gets `%.6f`. Measured: `r = [5.0, 5.0(1+1e-6), 5.0(1+2e-6)]` passes `np.all(np.diff(log_r_cm) > 0)` and emits three rows all reading `continue 19.188321` with three different densities (finding A-02).
- **Metallicity at `%.4f`.** `f"{1e-5:.4f}" == "0.0000"`, `f"{3.2e-4:.4f}" == "0.0003"` (6.25 % error), `f"{1e-4:.4f}" == "0.0001"` (1 significant figure) (finding A-03).
- **Non-finite values format as words.** `f"{float('nan'):.4f}" == "nan"` and `f"{float('inf'):.4f}" == "inf"`. The `z_override` path validates finiteness and positivity (`:201-205`); the `bundle.summary["ZCloud"]` path validates nothing (`:208-212`) (finding A-04).
- **Negative zero:** `f"{-0.0:.6f}" == "-0.000000"` and small negatives print `-0.0000`. Radii are ~18.5 so this cannot hit the radius column; the density column can print `-0.0000` for `n ≈ 1 cm⁻³`. Benign for a numeric parser.
- **Exponent format:** `AGE_YR` is the only `%e` value, `f"{3.14159e6:.4e}" == "3.1416e+06"` — a signed two-digit exponent with a lowercase `e`. Whether CLOUDY's number reader accepts that form depends on the command it lands in, which is in the template I cannot see; I flag it only as an observation.
- **`math.isclose(..., rel_tol=1e-12)` with no `abs_tol`** (`:155,160`) — safe here only because `R2 > 0` and `rShell > R2` are checked first.
- **`slopes = np.abs(dlog_n / dlog_r)`** (`dlaw.py:225`) divides without guarding `dlog_r == 0`. Two radii differing by 1 ULP can share a `log10` double ⇒ `inf`/`nan` slope ⇒ classified "not smooth" ⇒ densification skipped for that interval ⇒ the post-check at `:192` then raises. It degrades to a loud error, but emits a numpy divide warning on the way.

---

## 7. Dead code (flagged only, per project rule — no deletion proposed)

- `build_dlaw_block`'s `dens_profile` parameter (`dlaw.py:58`) is **never read in the body**. `snapshot_to_deck.py:254` computes and passes it; `run_loader.py:94-99` hard-rejects any run whose `dens_profile` is outside `{densBE, densPL}` (`:36`) even though nothing in this slice consumes the value.
- `--abundances` (`trinity_to_cloudy.py:125-128`) is parsed into `args.abundances` and never read.
- `snapshot_to_values(extend_with_ambient=...)` (`:57`) is never passed by `main()` (`:350-358`), so it is always `True` and the `raise` at `:217-224` is unreachable from the CLI.
- `DLAW_ROWS` (`snapshot_to_deck.py:258,276`) has no consumer inside the slice; only the template could use it.
- `VALID_DENS_PROFILES` is exported (`run_loader.py:343`) but used only for the gate above.
- `PickedSnapshot.snap` is used; `dlaw_open`/`dlaw_row_prefix`/`dlaw_close`/`edge_threshold` are parameterised but never overridden by any in-slice caller.

---

```json
[
  {
    "id": "S13b-A-01",
    "file": "trinity/_output/cloudy/snapshot_to_deck.py",
    "line": 272,
    "class": "divergence",
    "severity": "S2",
    "claim": "The same physical radius is emitted at two different precisions: the deck's LOG_RIN/LOG_ROUT at %.4f in log10(cm) (1.15e-4 relative) and the dlaw table's radius column at %.6f (1.15e-6 relative). The deck's declared inner/outer radius can therefore fall outside the tabulated dlaw range, and for a shell thinner than ~2.3e-5 relative the two radii collapse to the identical string.",
    "evidence": "snapshot_to_deck.py:182 `log_rin = math.log10(R2) + log_pc_per_cm`; :272-273 `\"LOG_RIN\": f\"{log_rin:.4f}\", \"LOG_ROUT\": f\"{log_rout:.4f}\"` vs dlaw.py:200 `lines.append(f\"{dlaw_row_prefix}{lr:.6f}  {ln:.4f}\")`",
    "expected": "The deck radii and the dlaw endpoints describe the same two numbers and should be emitted at matching or finer precision than the table, so the model range is provably inside the tabulated range.",
    "failure_scenario": "R2=0.5 pc: true log_rin = 18.188320549558156. Deck gets '18.1883' = 0.4999763 pc; first dlaw row gets '18.188321' = 0.5000000 pc. CLOUDY's starting radius is 4.7e-5 relative BELOW the table's first entry, so the first zone is off the end of the table. Separately, with R2=5.0 pc and relative shell thickness 2e-5 or 5e-5, LOG_RIN and LOG_ROUT both render as '19.1883' -> zero-thickness model.",
    "repro": "python3 -c \"import math; L=math.log10(3.0856775814913674e18); a=math.log10(0.5)+L; print(f'{a:.4f}', f'{a:.6f}'); b=math.log10(5.0)+L; c=math.log10(5.0*(1+5e-5))+L; print(f'{b:.4f}', f'{c:.4f}')\"",
    "confidence": "high"
  },
  {
    "id": "S13b-A-02",
    "file": "trinity/_output/cloudy/dlaw.py",
    "line": 200,
    "class": "numerical",
    "severity": "S2",
    "claim": "The strict-monotonicity guard is applied to the float radii, but the file is written with %.6f, so a table that passes the guard can contain rows with identical printed radii and different densities.",
    "evidence": "dlaw.py:129 `keep[:-1] = r_pc[:-1] != r_pc[1:]` (exact float dedup); :192 `if not np.all(np.diff(log_r_cm) > 0): raise DlawError(...)`; :200 `lines.append(f\"{dlaw_row_prefix}{lr:.6f}  {ln:.4f}\")`",
    "expected": "Uniqueness should be enforced on the emitted representation, not only on the in-memory floats; %.6f in log10 r resolves only 2.30e-6 in relative radius.",
    "failure_scenario": "Any shell sampled with consecutive spacing dr/r < 2.3e-6 (e.g. a shell of relative thickness 1e-4 sampled with ~100 points) emits repeated abscissae into the dlaw table. Measured with r=[5.0, 5.0*(1+1e-6), 5.0*(1+2e-6)]: three rows all read 'continue 19.188321' with densities 1.5319 / 2.0319 / 2.5319, and np.all(np.diff(log_r_cm)>0) is True.",
    "repro": "python3 -c \"import math,numpy as np; L=math.log10(3.0856775814913674e18); r=np.array([5.0,5.0*(1+1e-6),5.0*(1+2e-6)]); lr=np.log10(r)+L; print(np.all(np.diff(lr)>0)); [print(f'continue {x:.6f}') for x in lr]\"",
    "confidence": "high"
  },
  {
    "id": "S13b-A-03",
    "file": "trinity/_output/cloudy/snapshot_to_deck.py",
    "line": 274,
    "class": "numerical",
    "severity": "S2",
    "claim": "Metallicity is emitted with fixed 4-decimal formatting, which destroys any sub-solar metallicity below ~1e-2 Zsun and zeroes anything below 5e-5 Zsun.",
    "evidence": "snapshot_to_deck.py:212 `zrel = float(bundle.summary[\"ZCloud\"])`; :274 `\"ZREL\": f\"{zrel:.4f}\"` (no log, no scaling)",
    "expected": "A relative abundance scale factor spanning orders of magnitude should be emitted in a format that preserves significant figures (e.g. %.4e or a log form), not fixed-point.",
    "failure_scenario": "Z = 1e-5 Zsun renders as '0.0000' -> CLOUDY runs a metal-free model and the emission-line spectrum is qualitatively wrong. Z = 3.2e-4 renders as '0.0003', a 6.25% error. Z = 1e-4 renders as '0.0001', one significant figure.",
    "repro": "python3 -c \"print(f'{1e-5:.4f}', f'{3.2e-4:.4f}', f'{1e-4:.4f}')\"",
    "confidence": "high"
  },
  {
    "id": "S13b-A-04",
    "file": "trinity/_output/cloudy/snapshot_to_deck.py",
    "line": 212,
    "class": "divergence",
    "severity": "S2",
    "claim": "The two metallicity sources are validated asymmetrically: the CLI --z-override path checks finiteness and positivity, the bundle.summary['ZCloud'] path checks nothing. A NaN/inf/negative ZCloud is formatted straight into the deck.",
    "evidence": "snapshot_to_deck.py:201-206 `if z_override is not None: if not (math.isfinite(z_override) and z_override > 0): raise SnapshotInvalid(...)` versus :208-212 `if \"ZCloud\" not in bundle.summary: raise ...; zrel = float(bundle.summary[\"ZCloud\"])`",
    "expected": "Both sources feed the same deck field and should pass the same validity gate.",
    "failure_scenario": "A run whose summary carries ZCloud = NaN (or a legacy summary.txt whose value failed to coerce) yields ZREL = 'nan' in the deck: f\"{float('nan'):.4f}\" == 'nan'. The deck is written, the CLI reports WROTE, and CLOUDY sees a non-numeric token on the abundance line.",
    "repro": "python3 -c \"print(repr(f'{float(chr(110)+chr(97)+chr(110)):.4f}'), repr(f'{float(\\\"inf\\\"):.4f}'))\"",
    "confidence": "high"
  },
  {
    "id": "S13b-A-05",
    "file": "trinity/_output/cloudy/dlaw.py",
    "line": 226,
    "class": "silent-failure",
    "severity": "S2",
    "claim": "The min_rows floor (default 10, exposed as --min-rows) is silently unmet for thin shells: the edge_threshold=50 test on |dlog n / dlog r| classifies every interval of a geometrically thin shell as an 'edge', so densification is skipped entirely and only a UserWarning is issued.",
    "evidence": "dlaw.py:42 `DEFAULT_EDGE_THRESHOLD = 50.0`; :225-227 `slopes = np.abs(dlog_n / dlog_r); is_smooth = slopes <= edge_threshold; smooth_idx = np.where(is_smooth)[0]`; :229-236 `if smooth_idx.size == 0: warnings.warn(...); return log_r, log_n`; the only hard floor is :188 `if log_r_cm.size < 2`",
    "expected": "A threshold on a log-log slope must be scale-aware: for a shell of relative thickness d, dlog_r over the whole shell is ~d/ln10, so any density contrast of order 1 dex gives slopes of order 1e3-1e4 and the threshold can never be satisfied.",
    "failure_scenario": "Measured: shell r = [5.0, 5.0016, 5.0033, 5.005] pc (0.1% thick) with log n = [57.0, 58.0, 58.7, 59.0] pc^-3 gives slopes [7196.7, 4742.9, 2033.4] -- all > 50. A 4-row table is emitted despite --min-rows 10, and the shell is handed to CLOUDY as three log-log line segments.",
    "repro": "python3 -c \"import math,numpy as np; L=math.log10(3.0856775814913674e18); r=np.array([5.0,5.0016,5.0033,5.005]); n=np.array([57.,58.,58.7,59.]); lr=np.log10(r)+L; print(np.abs(np.diff(n)/np.diff(lr)))\"",
    "confidence": "high"
  },
  {
    "id": "S13b-A-06",
    "file": "trinity/_output/cloudy/snapshot_to_deck.py",
    "line": 244,
    "class": "silent-failure",
    "severity": "S2",
    "claim": "A zero (or negative) ambient density is clamped to the smallest positive double rather than rejected, producing a finite but physically meaningless dlaw row that passes every downstream guard.",
    "evidence": "snapshot_to_deck.py:242-244 `eps = np.finfo(float).tiny` / `ambient_log_n_pc3 = np.log10(np.maximum(amb_n, eps))`; the resulting value survives dlaw.py:194 `if not np.all(np.isfinite(log_r_cm)) or not np.all(np.isfinite(log_n_cm3))`",
    "expected": "Either reject a non-positive ambient density (the module raises SnapshotInvalid for every other non-physical input) or drop those points; do not emit them.",
    "failure_scenario": "A run with nISM = 0 (no ambient medium) plus --radius-out beyond rShell: every spliced ambient row is written as 'continue <logr>  -363.1207' (log10(2.2250738585072014e-308) = -307.6527, plus the -55.4681 dex unit offset). No error, no warning; CLOUDY receives hden = 10^-363 in the outer zones.",
    "repro": "python3 -c \"import math,numpy as np; print(f'continue {math.log10(np.finfo(float).tiny)+math.log10(1/2.937998946096347e+55):.4f}')\"",
    "confidence": "high"
  },
  {
    "id": "S13b-A-07",
    "file": "trinity/_output/cloudy/trinity_to_cloudy.py",
    "line": 277,
    "class": "state",
    "severity": "S2",
    "claim": "--all combined with --prefix makes every snapshot write to the same two filenames, so N-1 decks are silently overwritten while the manifest claims N distinct outputs.",
    "evidence": "trinity_to_cloudy.py:277-281 `if args.prefix is not None: return _UNSAFE_PREFIX_RE.sub(\"_\", args.prefix)` -- returns before the auto path that appends `pick.index`; _write_outputs :429-430 `deck_path = out_dir / f\"{prefix}.in\"` / `sidecar_path = out_dir / f\"{prefix}.dlaw.txt\"`; records appended per snapshot at :378-392 with `\"deck\": f\"{prefix}.in\"`",
    "expected": "--all must produce one file per snapshot; a user-supplied prefix should be a stem that the per-snapshot discriminator is appended to, or the combination should be rejected at parse time (as --all + --dry-run already is at :173-174).",
    "failure_scenario": "`trinity_to_cloudy -F run --all --prefix mymodel` writes mymodel.in once per snapshot; only the last survives. manifest.json lists every snapshot with \"deck\": \"mymodel.in\", and _print_summary reports 'Converted N snapshots'. A downstream batch runner reads the manifest and runs the same deck N times.",
    "repro": "Inspect trinity_to_cloudy.py:277-281 against :348-392: the loop body calls _build_prefix once per pick and _write_outputs with that prefix; with args.prefix set the prefix is pick-independent.",
    "confidence": "high"
  },
  {
    "id": "S13b-A-08",
    "file": "trinity/_output/cloudy/dlaw.py",
    "line": 58,
    "class": "deadcode",
    "severity": "S4",
    "claim": "build_dlaw_block accepts a dens_profile argument that is never read anywhere in its body, yet the whole chain plumbs it and run_loader hard-refuses to load any run whose dens_profile is outside {densBE, densPL}.",
    "evidence": "dlaw.py:58 `dens_profile: str = \"densPL\",` -- no other occurrence in dlaw.py (grep over the file returns only line 58). snapshot_to_deck.py:254 `dens_profile=str(bundle.metadata.get(\"dens_profile\", \"densPL\")),`. run_loader.py:36 `VALID_DENS_PROFILES = frozenset({\"densBE\", \"densPL\"})` and :94-99 raise RunLoadError for anything else.",
    "expected": "Either the density profile shapes the dlaw construction (e.g. picks a splice or interpolation rule) or the parameter and the load-time gate are not needed.",
    "failure_scenario": "A run using any third density-profile identifier cannot be exported at all, and the rejection message implies the profile matters to the deck, which — inside this slice — it does not.",
    "repro": "grep -n dens_profile trinity/_output/cloudy/dlaw.py  # only line 58, the signature",
    "confidence": "high"
  },
  {
    "id": "S13b-A-09",
    "file": "trinity/_output/cloudy/run_loader.py",
    "line": 184,
    "class": "divergence",
    "severity": "S2",
    "claim": "The two legacy text parsers in the same module disagree on how to handle a value carrying a unit suffix: _parse_simulation_end strips it, _parse_summary_txt does not.",
    "evidence": "run_loader.py:264 `tok = value.split()[0] if value else \"\"` then `out[out_key] = float(tok)` versus :179-184 `parts = line.split(None, 1)` ... `out[key] = _coerce_scalar(value_str)`, and _coerce_scalar's terminal `return s` at :324 for anything float() rejects.",
    "expected": "Both legacy parsers read the same run's scalars and should coerce numbers-with-units identically.",
    "failure_scenario": "A legacy summary line `ZCloud 0.02 Zsun` yields the Python string '0.02 Zsun'. snapshot_to_deck.py:212 then executes float('0.02 Zsun') -> ValueError, which is not in the --all skip tuple at trinity_to_cloudy.py:393, so the whole batch aborts with a traceback after partial output. The same line in simulationEnd.txt would have parsed fine.",
    "repro": "python3 -c \"s='0.02 Zsun'; print(float(s.split()[0]));  print(float(s))\"  # second call raises ValueError",
    "confidence": "medium"
  },
  {
    "id": "S13b-A-10",
    "file": "trinity/_output/cloudy/trinity_to_cloudy.py",
    "line": 200,
    "class": "silent-failure",
    "severity": "S2",
    "claim": "All three time-based pickers request mode='closest' with quiet=True, so a requested time far outside the sampled range silently yields the nearest snapshot; only the --age path prints how far off it landed.",
    "evidence": "trinity_to_cloudy.py:196 `snap = bundle.output.get_at_time(target_t, mode=\"closest\", quiet=True)`; :200 same for --t-now; :224-226 same for --phase; the delta report at :481-487 is guarded by `if args.age is not None`",
    "expected": "A tolerance (or at least an unconditional proximity report) on every time-based pick, so that asking for a time the run never reached is visible.",
    "failure_scenario": "`--t-now 12.0` on a run that terminated at t = 2.3 Myr emits a deck built from the t = 2.3 Myr snapshot and prints only 'Picked snapshot: index=..., t_now=2.3000 Myr'. Nothing states that the request was missed by 9.7 Myr.",
    "repro": "Read trinity_to_cloudy.py:194-227 and :473-492; the --t-now branch has no corresponding delta print.",
    "confidence": "medium"
  },
  {
    "id": "S13b-A-11",
    "file": "trinity/_output/cloudy/dlaw.py",
    "line": 125,
    "class": "silent-failure",
    "severity": "S2",
    "claim": "The shell profile is silently sorted and de-duplicated with no warning: a non-monotonic input is reordered, and an exactly-duplicated radius loses one of its two densities (the inner one).",
    "evidence": "dlaw.py:125-131 `order = np.argsort(r_pc, kind=\"stable\")` / `keep = np.ones(r_pc.size, dtype=bool)` / `keep[:-1] = r_pc[:-1] != r_pc[1:]` / `r_pc = r_pc[keep]` / `log_n_pc3 = log_n_pc3[keep]`",
    "expected": "Reordering or discarding sampled physics points should at least warn; the caller's own endpoint guarantee (snapshot_to_deck.py:155-164) is checked on the pre-sort array, so a reordered profile still passes it.",
    "failure_scenario": "A shell profile that represents a contact discontinuity as two points at the same radius with different densities silently keeps only the outer density. A profile that arrives non-monotonic (upstream bug) is silently straightened into a plausible-looking, physically different table, and the emitted deck runs.",
    "repro": "python3 -c \"import numpy as np; r=np.array([1.,2.,2.,3.]); n=np.array([5.,6.,7.,8.]); k=np.ones(4,bool); k[:-1]=r[:-1]!=r[1:]; print(r[k], n[k])\"  # -> [1. 2. 3.] [5. 7. 8.]: n=6 dropped, no warning",
    "confidence": "high"
  },
  {
    "id": "S13b-A-12",
    "file": "trinity/_output/cloudy/trinity_to_cloudy.py",
    "line": 223,
    "class": "other",
    "severity": "S2",
    "claim": "The --phase picker does not use the snapshot it selected: it reads that snapshot's t_now and re-resolves by closest time, so the returned snapshot can be a different one, potentially in a different phase from the one requested.",
    "evidence": "trinity_to_cloudy.py:214-227 `filtered = bundle.output.filter(phase=args.phase)` ... `target_t_now = filtered[which][\"t_now\"]` / `snap = bundle.output.get_at_time(target_t_now, mode=\"closest\", quiet=True)` / `return [PickedSnapshot(index=snap.index, snap=snap)]`",
    "expected": "Use filtered[which] directly; it is already the snapshot object the user asked for.",
    "failure_scenario": "If two snapshots share the same t_now (a phase transition written twice, or a restart), the closest-time lookup returns whichever the reader prefers -- possibly the one on the other side of the transition. The deck's TITLE (snapshot_to_deck.py:261-265) and filename (:279-281) then advertise a phase the deck was not built from.",
    "repro": "Read trinity_to_cloudy.py:213-227; the filtered snapshot is discarded and only its t_now survives.",
    "confidence": "medium"
  },
  {
    "id": "S13b-A-13",
    "file": "trinity/_output/cloudy/dlaw.py",
    "line": 177,
    "class": "units",
    "severity": "S2",
    "claim": "The density column undergoes a pure unit shift and nothing else: no mean-molecular-weight, hydrogen-mass-fraction or ionisation-state factor is applied anywhere in the slice. Whatever TRINITY means by log_shell_n_arr is handed to CLOUDY as hydrogen density.",
    "evidence": "dlaw.py:175,177 `log_ndens_offset = math.log10(INV_CONV.ndens_au2cgs)` / `log_n_cm3 = log_n_pc3 + log_ndens_offset` -- the only transformation on the density column; dlaw.py:33 `DEFAULT_DLAW_OPEN = \"dlaw table radius\"`; snapshot_to_deck.py:244 applies only log10 to the ambient column.",
    "expected": "A CLOUDY dlaw table's second column is hydrogen density n(H). If the TRINITY array is a total particle density, a composition factor (n_H = n_tot / (1 + y_He + x_e), ~0.45 for fully-ionised solar-composition gas) is required, i.e. a ~0.35 dex offset.",
    "failure_scenario": "If log_shell_n_arr is total particle density, every deck overstates n(H) by ~0.35 dex uniformly. The model still runs and produces a plausible spectrum with systematically wrong ionisation parameter and line ratios -- exactly the silent-wrong-answer mode. I cannot resolve which it is from this slice; the producing code is outside it.",
    "repro": "grep -rn 'mu\\|m_H\\|n_H\\|hden' trinity/_output/cloudy/  # no composition factor appears anywhere in the slice",
    "confidence": "low"
  },
  {
    "id": "S13b-A-14",
    "file": "trinity/_output/cloudy/dlaw.py",
    "line": 174,
    "class": "other",
    "severity": "S4",
    "claim": "The pc->cm log offset is recomputed at two sites and the local name is inverted relative to the quantity it holds (pc2cm is cm-per-pc, so log_pc_per_cm is log10(cm/pc)).",
    "evidence": "dlaw.py:174 `log_pc_per_cm = math.log10(INV_CONV.pc2cm)` and snapshot_to_deck.py:180 `log_pc_per_cm = math.log10(INV_CONV.pc2cm)`",
    "expected": "One definition, named for what it holds. I verified there is no numeric divergence today: both expressions are identical and evaluate to 18.489350545222138, and 3*that + log10(INV_CONV.ndens_au2cgs) == 0.0 exactly, so the radius and density columns share a bit-identical parsec.",
    "failure_scenario": "A future edit to one site (e.g. switching to a different length constant, or to the CGS-side cm2pc with a sign flip) leaves the deck's LOG_RIN/LOG_ROUT on one length scale and the dlaw table on another, with no test comparing them.",
    "repro": "python3 -c \"import math; L=math.log10(3.0856775814913674e18); print(repr(L), repr(3*L + math.log10(1/2.937998946096347e+55)))\"",
    "confidence": "high"
  },
  {
    "id": "S13b-A-15",
    "file": "trinity/_output/cloudy/snapshot_to_deck.py",
    "line": 99,
    "class": "silent-failure",
    "severity": "S4",
    "claim": "The required-key check uses a sentinel default, so a key present with value None passes validation and then raises TypeError, which the --all batch handler does not catch.",
    "evidence": "snapshot_to_deck.py:37 `_MISSING = object()`; :97-100 `missing = [k for k in REQUIRED_SNAPSHOT_KEYS if snap.get(k, _MISSING) is _MISSING]`; :105 `t_now = float(snap[\"t_now\"])`; trinity_to_cloudy.py:393 `except (SnapshotInvalid, DlawError, UnsubstitutedPlaceholder) as e:`",
    "expected": "Either treat None as missing, or widen the batch handler so a single malformed snapshot is skipped rather than aborting the run mid-way.",
    "failure_scenario": "One snapshot in a --all export has t_now: null in the jsonl. float(None) raises TypeError, which propagates out of main(); the output directory holds the decks written so far, no manifest.json, and no linelist (both are written after the loop, :413-416).",
    "repro": "python3 -c \"snap={'t_now':None}; M=object(); print([k for k in ('t_now',) if snap.get(k,M) is M]); float(snap['t_now'])\"",
    "confidence": "high"
  },
  {
    "id": "S13b-A-16",
    "file": "trinity/_output/cloudy/trinity_to_cloudy.py",
    "line": 419,
    "class": "silent-failure",
    "severity": "S4",
    "claim": "main() returns exit code 0 on the batch path regardless of outcome, including when every snapshot was skipped and when no snapshot was picked at all.",
    "evidence": "trinity_to_cloudy.py:396-410 appends a `\"status\": \"skipped\"` record and continues; :415-416 `if args.all: _write_manifest(out_dir, records)`; :419 `return 0` with no inspection of records; :496-500 `ok = sum(1 for r in records if r[\"status\"] == \"ok\") ... print(f\"Converted {ok} snapshots ({skipped} skipped).\")`",
    "expected": "A non-zero exit when nothing was successfully converted, so shell/SLURM wrappers can detect it.",
    "failure_scenario": "An automated sweep runs `trinity_to_cloudy -F run --all` over many runs; a run where every snapshot fails validation still exits 0, writes manifest.json full of skipped records and a linelist, and the wrapper records success.",
    "repro": "Read trinity_to_cloudy.py:347-419: `records` is never consulted for the return value.",
    "confidence": "high"
  },
  {
    "id": "S13b-A-17",
    "file": "trinity/_output/cloudy/trinity_to_cloudy.py",
    "line": 304,
    "class": "other",
    "severity": "S4",
    "claim": "The unsubstituted-placeholder guard only recognises the {{NAME}} form, so the default SB99 sentinel <<<EDIT_ME>>> is substituted successfully and written into a deck that CLOUDY cannot parse.",
    "evidence": "trinity_to_cloudy.py:74 `DEFAULT_SB99_SENTINEL = \"<<<EDIT_ME>>>\"`; :78 `PLACEHOLDER_RE = re.compile(r\"\\{\\{(\\w+)\\}\\}\")`; :366 `values[\"SB99_MOD\"] = args.sb99_mod`; :304-308 `leftovers = sorted(set(PLACEHOLDER_RE.findall(out)))` / `raise UnsubstitutedPlaceholder(...)`",
    "expected": "If the guard exists to prevent shipping an unrunnable deck, it should recognise the sentinel the tool itself inserts.",
    "failure_scenario": "The default invocation writes <prefix>.in containing '<<<EDIT_ME>>>' on the table-star line, prints 'WROTE: ...' plus a TODO, and exits 0. The TODO is the only thing standing between the user and `cloudy -r <prefix>` failing to parse. Also note render_template re-scans its own output (:303-304), so a substituted value that itself contained {{...}} would raise a spurious error.",
    "repro": "python3 -c \"import re; P=re.compile(r'\\\\{\\\\{(\\\\w+)\\\\}\\\\}'); print(P.findall('table star <<<EDIT_ME>>>'))\"  # -> []",
    "confidence": "high"
  },
  {
    "id": "S13b-A-18",
    "file": "trinity/_output/cloudy/trinity_to_cloudy.py",
    "line": 125,
    "class": "deadcode",
    "severity": "S4",
    "claim": "Two contracts in this slice have no consumer: the --abundances CLI option is parsed and never read, and snapshot_to_values(extend_with_ambient=False) is never reachable from the CLI.",
    "evidence": "trinity_to_cloudy.py:125-128 `parser.add_argument(\"--abundances\", type=str, default=None, dest=\"abundances\", ...)` -- `args.abundances` appears nowhere else. snapshot_to_deck.py:57 `extend_with_ambient: bool = True` with trinity_to_cloudy.py:350-358 never passing it, making the raise at snapshot_to_deck.py:217-224 unreachable from main(). snapshot_to_deck.py:276 `\"DLAW_ROWS\": dlaw_rows_only` has no in-slice consumer.",
    "expected": "n/a -- flagged per project rule, no deletion proposed.",
    "failure_scenario": "A user passes --abundances and believes the deck's abundance directive changed; it did not. Note the DLAW_ROWS/DLAW_BLOCK pair is a live risk I cannot evaluate: if the template interpolates both, the dlaw table is emitted twice.",
    "repro": "grep -rn 'abundances\\|extend_with_ambient\\|DLAW_ROWS' trinity/_output/cloudy/",
    "confidence": "high"
  }
]
```
