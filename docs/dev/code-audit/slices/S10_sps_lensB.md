# S10 SPS feedback — Lens B (what the code claims)

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

Prose-only transcription. Three files in slice: `trinity/sps/read_sps.py`,
`trinity/sps/sps_columns.py`, `trinity/sps/update_feedback.py`.
I have not seen any code. Every statement below is a *claim made by the prose*, not a
verified fact. Line citations are the first line of the comment/docstring block that
carries the claim.

---

## 1. Column documentation (highest-value transcription)

### 1.1 The default ("legacy SB99") 7-column positional preset

`trinity/sps/sps_columns.py:152` — "Legacy SB99 7-column positional preset. Injected as
the column map for the bundled default file (sps_path = def_path →
`lib/default/sps/starburst99/1e6cluster_default.csv`) so users do not need to declare
`sps_col_*` lines." … "Column order matches the canonical SB99 export layout:"

| file col (0-based) | canonical | declared unit | log? | verbatim prose |
|---|---|---|---|---|
| 0 | `t` | `yr` | **linear** | `col 0: time [yr] (linear)` (`sps_columns.py:158`) |
| 1 | `Qi` | `1/s` | **log10** | `col 1: log10 Qi [1/s] (log)` (`sps_columns.py:159`) |
| 2 | `fi` | dimensionless `[-]` | **log10** | `col 2: log10 fi [-] (log) ← yes, log-space; the legacy loader does 10**file[:,2]` (`sps_columns.py:160`) |
| 3 | `Lbol` | `erg/s` | **log10** | `col 3: log10 Lbol [erg/s] (log)` (`sps_columns.py:162`) |
| 4 | `Lmech_total` | `erg/s` | **log10** | `col 4: log10 Lmech_total [erg/s] (log)` (`sps_columns.py:163`) |
| 5 | `pdot_W` | `g*cm/s^2` | **log10** | `col 5: log10 pdot_W [g*cm/s^2] (log)` (`sps_columns.py:164`) |
| 6 | `Lmech_W` | `erg/s` | **log10** | `col 6: log10 Lmech_W [erg/s] (log)` (`sps_columns.py:165`) |

Consequences the preset implies (not stated, but forced by the table): the bundled file
supplies **no** `Li`, `Ln`, `Lmech_SN`, `pdot_SN`, `Mdot_SN`, `v_SN` columns — all six must
come from the documented fallback derivations in §2.2.

The `fi` column being log-space is called out defensively ("yes, log-space"), i.e. the
author expected a reader to doubt it. `fi` is a fraction, so `log10 fi ≤ 0` is the
expected sign of the raw column.

### 1.2 Canonical target units the loader always produces

`trinity/sps/sps_columns.py:97` — "Per-canonical map: declared unit string ->
multiplicative factor that takes a **LINEAR (already-exponentiated)** value into the
canonical AU unit." `sps_columns.py:63` — "Canonical AU units the loader produces,
**regardless of what the file declares**."

| group | canonical(s) | target unit (`sps_columns.py:100`–`106`) |
|---|---|---|
| Time | `t` | `Myr` |
| Photon | `Qi` | `1/Myr` |
| Luminosity | `Lbol`, `Lmech_*`, `Li`, `Ln` | `Msun*pc^2/Myr^3` |
| Momentum rate | `pdot_*` | `Msun*pc/Myr^2` |
| Mass-loss | `Mdot_SN` | `Msun/Myr` |
| Velocity | `v_SN` | `pc/Myr` |
| Fraction | `fi` | dimensionless |

### 1.3 Accepted declared-unit strings / aliases

`trinity/sps/sps_columns.py:108` — "Each per-canonical sub-dict also includes a convenience
alias `'cgs'` which maps to the canonical's default cgs unit (e.g. erg/s for luminosities,
g*cm/s^2 for momentum rates, etc.). For dimensionless quantities `'cgs'` is a synonym for
`'dimensionless'`." Documented example: `sps_col_Qi 0 cgs log` "instead of having to
remember that Qi's cgs unit is 1/s."

Per-canonical alias comments:
- `sps_columns.py:119` — `t`: "alias for `'s'`" (so `t` declared `cgs` ⇒ seconds).
- `sps_columns.py:124` — `Qi`: "alias for `'1/s'`".
- `sps_columns.py:128` — `fi`: "alias for `'dimensionless'`".
- `sps_columns.py:139` — `Mdot_SN`: "`g/s -> Msun/Myr`"; `:141` "alias for `'g/s'`".
- `sps_columns.py:144` — `v_SN`: "`cm/s -> pc/Myr`"; `:145` "`km/s -> pc/Myr`"; `:147`
  "alias for `'cm/s'`".

### 1.4 `ColumnSpec` field contract

`trinity/sps/sps_columns.py:38` —
- `file_column : str or int` — "A string (header-row name) for user-defined sps_path.
  An int (positional index) for the default SPS preset."
- `units : str` — "Must be a key in `UNIT_CONVERSIONS[canonical]`."
- `log : bool` — "True if file values are in log10 space; False if linear."

### 1.5 `.param` declaration syntax

`trinity/sps/sps_columns.py:214` — `sps_col_<canonical>` value is
`"<file_column> <units> <log|linear>"`, whitespace-separated, **exactly 3 fields**. First
field is "a non-negative integer (0-based column index; **works on any file**, with or
without a header)" OR "a string column name matching the file's header row (the file must
have a header for this to resolve)". `sps_columns.py:234` — "Auto-detect: all-digits -> int."

---

## 2. Formulas

### 2.1 Stated as maths

| quantity | formula | citation |
|---|---|---|
| mass fraction | `f_mass = M_cluster / sps_refmass` (also written `mCluster / sps_refmass`) | `read_sps.py:39`, `sps_columns.py:3` |
| wind velocity | `v_wind = 2 * Lmech_W / pdot_W` (**after** corrections) | `read_sps.py:39` |
| SN velocity | "from `params['FB_vSN']` (after corrections)" | `read_sps.py:39` |
| effective mechanical velocity | `v_mech_total = 2 * Lmech_total / pdot_total` | `update_feedback.py:99`, `:180`; `read_sps.py:286` |
| ram pressure identity | `pRam = L/(2*pi*r^2*v)` with the above `v` "yields the correct total ram pressure: `pdot_total / (4*pi*r^2)`" | `update_feedback.py:99` |
| unit conversion order | 1) if `log`, `10**arr`; 2) multiply by `UNIT_CONVERSIONS[canonical][declared_units]`; mass scaling applied **separately by the caller** via `CANONICALS[canonical].mass_scaled` | `sps_columns.py:181` |

The ram-pressure identity is internally consistent: `L/(2πr²·(2L/pdot)) = pdot/(4πr²)`.

### 2.2 Fallback derivations for missing optional canonicals

`trinity/sps/read_sps.py:135` — "Missing optional canonicals fall back to the existing
derivations", listed in this order:

1. `Li, Ln  <- Lbol * fi,  Lbol * (1 - fi)`   [if not supplied]
2. `Lmech_SN_raw <- Lmech_total - Lmech_W`     [if `Lmech_SN` absent]
3. `Mdot_SN <- 2 * Lmech_SN_raw / v_SN^2`      [if `Mdot_SN` absent]
4. `v_SN <- params['FB_vSN'].value`            [if `v_SN` absent]
5. `pdot_SN <- Mdot_SN_modified * v_SN_mod`    [if `pdot_SN` absent]

"User-supplied columns plug into the pipeline at the points indicated above;
`FB_mColdSNFrac` / `FB_thermCoeffSN` **still apply on top**."

Note the listed order: (3) consumes `v_SN`, whose default is only established at (4).

### 2.3 Scaling with cluster mass

The **only** stated cluster-mass scaling is linear via `f_mass`: `sps_columns.py:3` — the
registry records "which are mass-scaled (`f_mass = mCluster / sps_refmass`)", and
`sps_columns.py:181` — "Mass scaling (multiply by `f_mass`) is applied separately by the
caller." The prose never enumerates *which* canonicals carry `mass_scaled=True`.

### 2.4 Correction coefficients — named but never given a formula

`read_sps.py:39` names `FB_mColdWindFrac`, `FB_thermCoeffWind` ("Wind corrections") and
`FB_mColdSNFrac`, `FB_thermCoeffSN`, `FB_vSN` ("SN corrections"); `read_sps.py:39` Notes:
"Thermal efficiency and cold mass corrections are applied to winds and SN." Section
markers `read_sps.py:210` "=== WIND corrections (same math as the legacy path) ===" and
`read_sps.py:224` "=== SN corrections (with user-override pluggability) ===". **No prose
anywhere states the algebraic form of these corrections.**

---

## 3. Units and log/linear notes

- All loader outputs are converted "to astronomical units (Msun, pc, Myr)"
  (`read_sps.py:39`).
- `read_sps.py:39` return units: `t` [Myr]; `Qi` [1/Myr] "(AU; × s2Myr → 1/s)";
  `Li` "(>13.6 eV)" [Msun*pc^2/Myr^3]; `Ln` "(<13.6 eV)" [Msun*pc^2/Myr^3];
  `Lbol`, `Lmech_W`, `Lmech_SN`, `Lmech_total` [Msun*pc^2/Myr^3];
  `pdot_W`, `pdot_SN`, `pdot_total` [Msun·pc/Myr²].
- `update_feedback.py:22` restates: "All luminosities are in code units [Msun·pc²/Myr³]
  (multiply by `INV_CONV.L_au2cgs` to get erg/s); raw cgs values are converted to AU at
  load time in `read_sps.py`." `v_mech_total` [pc/Myr]. `pdotdot_total` — "Time derivative
  of total momentum rate", **no unit given** (expected [Msun·pc/Myr³]).
- `update_feedback.py:161` — "The interpolators were built in read_sps.py from arrays
  already converted to code units (AU); luminosities here are [Msun*pc^2/Myr^3], not erg/s."
- Log-space is declared per column via `ColumnSpec.log` (`sps_columns.py:38`) and
  exponentiated first (`sps_columns.py:181`). In the bundled preset **six of seven columns
  are log10**; only `t` is linear (`sps_columns.py:158`–`165`).
- `sps_columns.py:32` — "Solar luminosity in erg/s (no `L_sun` constant currently in `cvt`)."

---

## 4. Table provenance

Everything the prose says about where the tables come from:

- "Legacy **SB99** 7-column positional preset" / "Column order matches the canonical
  **SB99 export layout**" (`sps_columns.py:152`, `:157`).
- Bundled default path: `lib/default/sps/starburst99/1e6cluster_default.csv`
  (`sps_columns.py:152`).
- `sps_path` is "resolved by read_param.py — either the user's `sps_path` or the bundled
  default file"; the bundled file "is used when the user hasn't overridden `sps_path`"
  (`read_sps.py:3`, `:39`).
- Background pointer: `docs/dev/archive/sb99-refactor-audit.md` §9 (legacy as permanent
  fallback) and §10 PR-2 (column-mapping design) — cited at `sps_columns.py:3`, `:152`,
  `:279`, `read_sps.py:39`.

**Not stated anywhere in this slice:** Starburst99 version number; IMF (form, slope, mass
limits); metallicity or any metallicity grid; the time grid (spacing, start, end); the
number of rows; whether the table is single-burst or continuous SF. The only quantitative
provenance hint is the filename `1e6cluster` (implying a 10⁶ M☉ reference cluster, i.e. the
expected default of `sps_refmass` — never stated).

---

## 5. Ranges, regimes, and behaviour past the grid

- **Nothing in the slice's prose says what happens for `t` beyond the last table row.** No
  mention of extrapolation, clamping, `fill_value`, or `bounds_error`. The only
  interpolation contract is "scipy cubic interpolators" (`read_sps.py:3`) /
  "`scipy.interpolate.interp1d` … Options: 'linear', 'cubic' (default), 'quadratic', etc.
  Cubic is recommended for small-value interpolations." (`read_sps.py:286`).
- `t` must be **strictly increasing** — enforced at load time (`sps_columns.py:335`,
  `read_sps.py:180`). Documented failure mode: "the file's time column was written with too
  few significant figures (e.g. `'%.2e'` format collapses 1.001e7, 1.002e7, 1.003e7 all to
  the same string `"1.00e+07"`)". (`read_sps.py:185` uses `'%.2E'` / `'1.00E+007'` for the
  same example.)
- `f_mass` validity: "Raises ValueError If `f_mass <= 0` or is NaN/inf, or if the file shape
  is invalid" (`read_sps.py:39`). `FileNotFoundError` if `sps_path` does not exist.
- `file_column` integer must be "non-negative" (`sps_columns.py:214`).

---

## 6. Contracts (inputs, outputs, state, side effects, ordering)

### `read_sps(f_mass, params)` — `read_sps.py:39`
- **Reads from params:** `sps_path`, `FB_mColdWindFrac`, `FB_thermCoeffWind`,
  `FB_mColdSNFrac`, `FB_thermCoeffSN`, `FB_vSN`. Body prose also names
  `params['sps_column_map']` (`read_sps.py:39` opening line) though it is absent from the
  Parameters list.
- **`f_mass` is computed by `main.py` before this function is called** (ordering
  requirement).
- **Returns** a list of exactly 11 arrays in this order:
  `[t, Qi, Li, Ln, Lbol, Lmech_W, Lmech_SN, Lmech_total, pdot_W, pdot_SN, pdot_total]`
  ("the 11-array tuple", `read_sps.py:3`).
- "All arrays have t=0 prepended with initial values for interpolation" (`read_sps.py:39`)
  — but `read_sps.py:262`: "t=0 prepend (**idempotent — skip if the file already starts at
  t=0**)".
- No documented mutation of `params`.

### `_read_sps_user` — `read_sps.py:135`
- Loads "any .txt or .csv file (delimiter auto-sniffed, header auto-detected,
  '#'-comment lines skipped)", applies per-column unit conversion + mass scaling, then
  "runs the FB_* correction pipeline".
- Documented internal order (by line): monotonic-`t` check (`:180`) → derive `Li/Ln`
  (`:188`) → derive `Lmech_SN_raw` (`:196`) → wind corrections (`:210`) → SN corrections
  (`:224`) → totals (`:248`) → convenience aliases (`:252`) → t=0 prepend (`:262`).
- `read_sps.py:188` — "Derive Li, Ln if not supplied (matches legacy 13.6 eV behaviour when
  only `fi` is given; **bypassed entirely when both Li and Ln are present** — this is what
  closes audit hot-spot #5)."
- `read_sps.py:196` — "Derive `Lmech_SN_raw` if not supplied. **Validation in read_param.py
  ensures at least one of (`Lmech_SN`, `Lmech_total`) is present.**" (cross-module contract)

### `get_interpolation(sps, ftype='cubic')` — `read_sps.py:286`
- Input: the 11-array list from `read_sps`. Output: dict `sps_f` with exactly **10** keys:
  `fQi, fLi, fLn, fLbol, fLmech_W, fLmech_SN, fLmech_total, fpdot_W, fpdot_SN, fpdot_total`.
- Module docstring (`read_sps.py:3`) instead says it "Wraps the 11-array tuple returned by
  `read_sps()` in scipy cubic interpolators **on `params['sps_f']`**" — i.e. describes the
  dict as an input location rather than a return value.
- Naming contract: `_W` = wind, `_SN` = supernova, `_total` = sum (`read_sps.py:286`,
  `update_feedback.py:99`).

### `sps_columns` helpers
- `convert_to_canonical_au` (`sps_columns.py:181`) — raises `KeyError` for unknown
  canonical, `ValueError` for a unit string not recognised for that canonical.
- `build_user_column_map` (`sps_columns.py:255`) — walks all `sps_col_<canonical>` params;
  "Entries still holding the `def_unset` sentinel are skipped"; `sps_columns.py:268`
  "Should never happen if default.param declares them all, but be defensive: missing
  entries are treated as unset." Returns `dict[canonical -> ColumnSpec]`.
- `validate_user_column_map` (`sps_columns.py:279`) — rules, verbatim:
  1. "Every canonical in `_REQUIRED_ALWAYS` must be present."
  2. "Either `fi` present, OR both `Li` AND `Ln` present."
  3. "`Lmech_total` OR `Lmech_SN` must be present (loader needs at least one to drive the
     SN pipeline; `Mdot_SN` alone is **not yet** a supported entry point here)."
  4. "`Li` XOR `Ln` is forbidden (must be both or neither)."
  Raises `ValueError` with "a fillable template" (`_format_missing_template`,
  `sps_columns.py:317`: "One-line error: what's expected, what's missing, what was
  declared").
- `validate_t_monotonic` (`sps_columns.py:335`) — "Applied by `_read_sps_user` in
  `trinity/sps/read_sps.py`"; raises a clearer `ValueError` pointing at the file and the
  **first offending row**.
- `_scan_layout` (`sps_columns.py:386`) — returns `(data_start, header_names, delimiter)`.
  Header definition, verbatim: "the **non-blank non-#** row **immediately above**
  `data_start` that has the **same token count** as the data row and contains **at least
  one non-numeric token**". `sps_columns.py:440` — "only the immediate predecessor counts".
  Delimiter: "',' if the first data line contains a comma; else None (whitespace, the
  `np.loadtxt` default)".
- `_can_parse_float` (`sps_columns.py:376`) — "True iff s parses as a float (covers
  integers, decimals, scientific notation, **inf/nan**). Used to distinguish data rows from
  header rows."
- `load_user_columns` (`sps_columns.py:447`) — returns dict keyed by canonical, "raw
  values, **no unit conversion yet**". Resolution: `int (>= 0) -> data[:, file_column]`;
  `str -> data[:, header_names.index(file_column)]` "(**raises** if no header detected or
  name not in header)". Supports "'#'-prefixed comment lines and blank lines anywhere
  **above** the data".

### `update_feedback.py`
- Module docstring (`update_feedback.py:3`): "Evaluate SPS feedback values at a given time
  **and update the params dictionary**."
- `get_current_sps_feedback(t, params)` (`update_feedback.py:99`) — needs
  `params['sps_f']`; returns `SPSFeedback`; documents no `params` mutation.
- `SPSFeedback` (`update_feedback.py:22`) — 13 fields, in this order: `t, Qi, Li, Ln, Lbol,
  Lmech_W, Lmech_SN, Lmech_total, pdot_W, pdot_SN, pdot_total, pdotdot_total,
  v_mech_total`. Supports attribute access, unpacking (`__iter__`, `:81`), indexing
  (`__getitem__`, `:90`, "`feedback[0] == feedback.t`"), and `__len__` (`:94`, "Return
  number of fields"). Unpacking order is claimed to be
  "`(t, Qi, Li, Ln, Lbol, ...) = get_current_sps_feedback(t, params)`" — "**Old style
  (still works)**".
- `update_feedback.py:183` — "Numerical derivative of total momentum rate for time
  evolution"; `:184` — "Myr (small timestep for derivative)".
- No caching claim appears anywhere in the slice.

---

## 7. Admissions of debt (verbatim)

| citation | admission |
|---|---|
| `sps_columns.py:279` | "`Mdot_SN` alone is **not yet** a supported entry point here" |
| `sps_columns.py:32` | "no `L_sun` constant **currently** in `cvt`" |
| `sps_columns.py:268` | "**Should never happen** if default.param declares them all, but be defensive" |
| `sps_columns.py:160` | "← **yes**, log-space; the legacy loader does `10**file[:,2]`" (defensive against a suspected error) |
| `read_sps.py:188` | "this is what **closes audit hot-spot #5**" (references an external audit finding) |
| `read_sps.py:180` / `sps_columns.py:335` | scipy's "native error is **cryptic**" — workaround wrapper |
| `read_sps.py:286` | "Cubic is **recommended** for small-value interpolations" (unsupported recommendation) |
| `read_sps.py:210` | "same math as the **legacy** path" (legacy parity as the correctness bar) |
| `update_feedback.py:22` | "Supports both attribute access and unpacking for **backward compatibility**" / "Old style (still works)" |

## 8. Prose-vs-prose contradictions found

1. `ColumnSpec.file_column` "A string … for user-defined sps_path. An int … for the default
   SPS preset" (`sps_columns.py:38`) **vs** "a non-negative integer (0-based column index;
   **works on any file**)" (`sps_columns.py:214`) and `load_user_columns` accepting ints
   (`sps_columns.py:447`).
2. "all with t=0 prepended" / "All arrays have t=0 prepended" (`read_sps.py:39`) **vs**
   "idempotent — skip if the file already starts at t=0" (`read_sps.py:262`).
3. "update the params dictionary" (`update_feedback.py:3`) **vs** a documented pure return
   with no side effect (`update_feedback.py:99`).
4. `get_interpolation` "returns `sps_f : dict`" (`read_sps.py:286`) **vs** "wraps … in
   scipy cubic interpolators **on** `params['sps_f']`" (`read_sps.py:3`).
5. Doc example `sps_col_Qi 0 cgs log` (`sps_columns.py:112`) puts `Qi` at index 0, while the
   canonical SB99 layout puts `t` at 0 and `Qi` at 1 (`sps_columns.py:158`–`159`).

---

```json
[
  {
    "id": "S10-B-01",
    "file": "trinity/sps/sps_columns.py",
    "line": 152,
    "class": "units",
    "severity": "S2",
    "claim": "The bundled default SPS file is read with a hard-coded 7-column positional preset: col 0 = time [yr] LINEAR; col 1 = log10 Qi [1/s]; col 2 = log10 fi [-]; col 3 = log10 Lbol [erg/s]; col 4 = log10 Lmech_total [erg/s]; col 5 = log10 pdot_W [g*cm/s^2]; col 6 = log10 Lmech_W [erg/s]. Every column except col 0 is log10.",
    "evidence": "sps_columns.py:152 'Legacy SB99 7-column positional preset. Injected as the column map for the bundled default file (sps_path = def_path -> lib/default/sps/starburst99/1e6cluster_default.csv)'; sps_columns.py:157-165 lists 'col 0: time [yr] (linear)' through 'col 6: log10 Lmech_W [erg/s] (log)'.",
    "expected": "The literal preset in code maps index->canonical->unit->log flag exactly as documented, and the first data row of lib/default/sps/starburst99/1e6cluster_default.csv is consistent with it (col 0 ~1e4-1e7 in yr; cols 1,3,4,6 in the 30-55 range as log10 of cgs; col 2 negative-or-zero as log10 of a fraction; col 5 ~30-34).",
    "failure_scenario": "Any index/log/unit mismatch silently rescales a feedback channel by orders of magnitude; e.g. swapping cols 4 and 6 makes Lmech_SN = Lmech_W - Lmech_total < 0 for the whole run.",
    "repro": "Compare the preset literal against this table; then check column count and per-column magnitude of the bundled csv.",
    "confidence": "high"
  },
  {
    "id": "S10-B-02",
    "file": "trinity/sps/sps_columns.py",
    "line": 160,
    "class": "exponent",
    "severity": "S2",
    "claim": "Column 2 (fi, the ionising fraction) is in log10 space and must be exponentiated: 'yes, log-space; the legacy loader does 10**file[:,2]'.",
    "evidence": "sps_columns.py:160 'col 2: log10 fi [-] (log) <- yes, log-space; the legacy loader does 10**file[:,2]'. Cross-ref read_sps.py:135 derivation 'Li, Ln <- Lbol * fi, Lbol * (1 - fi)'.",
    "expected": "The preset ColumnSpec for fi has log=True, and the loaded fi lies in (0, 1]. If any loaded fi > 1, Ln = Lbol*(1-fi) becomes negative.",
    "failure_scenario": "If the file actually stores linear fi (typical value ~0.2-0.6), exponentiating gives 10^0.2..10^0.6 = 1.6..4, so Li > Lbol and Ln < 0 for the whole run - a negative non-ionising luminosity fed straight into the radiation force budget.",
    "repro": "Assert 0 < fi <= 1 and Ln >= 0 for every row after loading the bundled default.",
    "confidence": "high"
  },
  {
    "id": "S10-B-03",
    "file": "trinity/sps/read_sps.py",
    "line": 39,
    "class": "units",
    "severity": "S3",
    "claim": "Qi is stored internally in [1/Myr] (photons per Myr), converted from the file's 1/s, and is converted back with 'x s2Myr -> 1/s'.",
    "evidence": "read_sps.py:39 'Qi : ndarray Ionizing photon rate [1/Myr] (AU; x s2Myr -> 1/s)'; sps_columns.py:101 'Photon: target = 1/Myr'; sps_columns.py:124 Qi cgs 'alias for 1/s'; update_feedback.py:22 repeats '[1/Myr] (internal; x s2Myr -> 1/s)'.",
    "expected": "UNIT_CONVERSIONS['Qi']['1/s'] equals seconds-per-Myr (~3.156e13), and every downstream consumer of Qi (ionisation-front / Stromgren radius, recombination balance) treats it as 1/Myr rather than 1/s.",
    "failure_scenario": "A downstream consumer using Qi as 1/s under-counts ionising photons by ~3.16e13, collapsing the ionised region.",
    "repro": "Check the Qi conversion factor's numeric value and grep every use of fQi/Qi for an implicit 1/s assumption.",
    "confidence": "medium"
  },
  {
    "id": "S10-B-04",
    "file": "trinity/sps/sps_columns.py",
    "line": 3,
    "class": "coefficient",
    "severity": "S2",
    "claim": "All mass-scaled SPS columns are multiplied by f_mass = mCluster / sps_refmass (strictly linear scaling with cluster mass), and the bundled default table is a 1e6 Msun cluster (filename 1e6cluster_default.csv).",
    "evidence": "sps_columns.py:3 'which are mass-scaled (f_mass = mCluster / sps_refmass)'; sps_columns.py:181 'Mass scaling (multiply by f_mass) is applied separately by the caller, using CANONICALS[canonical].mass_scaled'; read_sps.py:39 'f_mass : float Cluster mass fraction (f_mass = M_cluster / sps_refmass). Computed by main.py'; sps_columns.py:152 names the bundled file '1e6cluster_default.csv'.",
    "expected": "The default value of sps_refmass in the schema is 1e6 Msun, matching the bundled table's own cluster mass; and f_mass is applied exactly once per column.",
    "failure_scenario": "If the schema default of sps_refmass differs from the bundled table's cluster mass, every luminosity, photon rate and momentum rate is off by that constant ratio for every default run - a silent global normalisation error that no test on ratios would catch.",
    "repro": "Read the sps_refmass default from trinity/_input schema/default.param and compare with the bundled table's stated cluster mass.",
    "confidence": "high"
  },
  {
    "id": "S10-B-05",
    "file": "trinity/sps/read_sps.py",
    "line": 135,
    "class": "sign",
    "severity": "S2",
    "claim": "When Lmech_SN is not supplied (the default-preset case), it is derived as a difference of two independently log10-stored columns: Lmech_SN_raw <- Lmech_total - Lmech_W. The prose states no clamp, floor, or non-negativity guard.",
    "evidence": "read_sps.py:135 'Lmech_SN_raw <- Lmech_total - Lmech_W [if Lmech_SN absent]'; read_sps.py:196 'Derive Lmech_SN_raw if not supplied'; sps_columns.py:163,165 show cols 4 and 6 are separate log10 erg/s columns.",
    "expected": "Either a documented/implemented clamp to >= 0, or a demonstration that Lmech_total >= Lmech_W for every row of every shipped table.",
    "failure_scenario": "Before the first SN (t < ~3 Myr) Lmech_total should equal Lmech_W; round-off in the two log10 columns makes the difference a small number of either sign. A negative Lmech_SN_raw then propagates through Mdot_SN = 2*Lmech_SN_raw/v_SN^2 (negative mass loss) and pdot_SN = Mdot_SN*v_SN (negative momentum injection), and cubic interpolation across the sign flip amplifies it.",
    "repro": "Load the bundled default and assert min(Lmech_total - Lmech_W) >= 0 and min(Lmech_SN) >= 0 across all rows.",
    "confidence": "medium"
  },
  {
    "id": "S10-B-06",
    "file": "trinity/sps/read_sps.py",
    "line": 135,
    "class": "state",
    "severity": "S3",
    "claim": "The documented fallback list orders the Mdot_SN derivation (which consumes v_SN) BEFORE the v_SN default assignment: item 3 is 'Mdot_SN <- 2 * Lmech_SN_raw / v_SN^2' and item 4 is 'v_SN <- params[FB_vSN].value'.",
    "evidence": "read_sps.py:135 lists, in order: 'Lmech_SN_raw <- Lmech_total - Lmech_W', 'Mdot_SN <- 2 * Lmech_SN_raw / v_SN^2 [if Mdot_SN absent]', 'v_SN <- params['FB_vSN'].value [if v_SN absent]', 'pdot_SN <- Mdot_SN_modified * v_SN_mod'.",
    "expected": "In code, v_SN must be resolved (from column or FB_vSN) before it is used in the Mdot_SN derivation.",
    "failure_scenario": "If the code follows the documented order literally, the default path (no v_SN column) divides by an unset/None/sentinel v_SN, raising or producing garbage; if the code is right, the docstring ordering is misleading to the next maintainer.",
    "repro": "Check the statement order in _read_sps_user around read_sps.py:224 for where v_SN is bound relative to the Mdot_SN expression.",
    "confidence": "medium"
  },
  {
    "id": "S10-B-07",
    "file": "trinity/sps/read_sps.py",
    "line": 135,
    "class": "units",
    "severity": "S3",
    "claim": "The v_SN fallback takes params['FB_vSN'].value with no documented unit conversion, while the canonical target unit for v_SN is pc/Myr and the registry offers cm/s and km/s converters for file-supplied v_SN columns.",
    "evidence": "read_sps.py:135 'v_SN <- params['FB_vSN'].value [if v_SN absent]'; sps_columns.py:105 'Velocity: target = pc/Myr (v_SN)'; sps_columns.py:144-147 'cm/s -> pc/Myr', 'km/s -> pc/Myr', 'alias for cm/s'.",
    "expected": "params['FB_vSN'] is already stored in pc/Myr (AU) by read_param.py, OR the fallback converts it. Note 1 km/s = 1.0227 pc/Myr, so an omitted km/s->pc/Myr conversion is a deceptively small 2.3% error.",
    "failure_scenario": "FB_vSN declared in km/s and used raw as pc/Myr gives v_SN 2.3% low, Mdot_SN = 2L/v^2 4.6% high, pdot_SN 2.3% high - too small to notice, too large to be right; a cm/s value used raw would be off by 3e13.",
    "repro": "Check the declared unit of FB_vSN in the .param schema and whether read_param converts it to AU before read_sps reads .value.",
    "confidence": "medium"
  },
  {
    "id": "S10-B-08",
    "file": "trinity/sps/update_feedback.py",
    "line": 3,
    "class": "state",
    "severity": "S3",
    "claim": "The module docstring claims the module updates the params dictionary; the function docstring documents a pure interpolate-and-return with no params mutation.",
    "evidence": "update_feedback.py:3 'Evaluate SPS feedback values at a given time and update the params dictionary.' vs update_feedback.py:99 'Returns ------- SPSFeedback Dataclass containing all feedback parameters.' with params documented only as an input 'containing the sps_f interpolators'.",
    "expected": "Either the function does write feedback values back into params (in which case the key names and write ordering are an undocumented contract callers may depend on), or it does not and the module docstring is stale.",
    "failure_scenario": "If callers rely on params being refreshed as a side effect and it is not (or vice versa), the ODE right-hand side evaluates feedback at a stale time.",
    "repro": "Check get_current_sps_feedback for any params[...] = assignment.",
    "confidence": "high"
  },
  {
    "id": "S10-B-09",
    "file": "trinity/sps/read_sps.py",
    "line": 39,
    "class": "state",
    "severity": "S3",
    "claim": "read_sps documents producing a wind velocity (v_wind = 2*Lmech_W/pdot_W, after corrections) and an SN velocity, but neither appears in the documented 11-array return list nor among the 13 SPSFeedback fields.",
    "evidence": "read_sps.py:39 Notes: 'Wind velocity: v_wind = 2 * Lmech_W / pdot_W (after corrections)' and 'SN velocity: from params['FB_vSN'] (after corrections)'; the documented Returns list is exactly [t, Qi, Li, Ln, Lbol, Lmech_W, Lmech_SN, Lmech_total, pdot_W, pdot_SN, pdot_total]; update_feedback.py:22 lists no v_wind field.",
    "expected": "Either v_wind is an internal intermediate only (docstring should not present it as an output), or it is stashed in params / used elsewhere via an undocumented channel that consumers depend on.",
    "failure_scenario": "A consumer needing the wind velocity recomputes 2*Lmech_W/pdot_W from post-correction arrays and gets a different value than the loader's internal one if the corrections are applied in a different order.",
    "repro": "Grep for v_wind / vWind assignment targets in read_sps.py and for readers of that name elsewhere.",
    "confidence": "medium"
  },
  {
    "id": "S10-B-10",
    "file": "trinity/sps/sps_columns.py",
    "line": 38,
    "class": "citation",
    "severity": "S4",
    "claim": "ColumnSpec's docstring restricts file_column to str for user files and int for the default preset; two other docstrings say integer indices work on any file.",
    "evidence": "sps_columns.py:38 'A string (header-row name) for user-defined sps_path. An int (positional index) for the default SPS preset.' vs sps_columns.py:214 'a non-negative integer (0-based column index; works on any file, with or without a header)' and sps_columns.py:447 'int (>= 0) -> data[:, file_column] (0-based positional)'.",
    "expected": "One of the two statements is authoritative; the ColumnSpec docstring should describe the union type as parse_sps_col_value/load_user_columns implement it.",
    "failure_scenario": "A user reading only the ColumnSpec docstring believes a headerless user file cannot be used, or a validator written against ColumnSpec's wording rejects legal integer specs for user files.",
    "repro": "",
    "confidence": "high"
  },
  {
    "id": "S10-B-11",
    "file": "trinity/sps/read_sps.py",
    "line": 39,
    "class": "other",
    "severity": "S4",
    "claim": "The return contract states unconditionally that all arrays have t=0 prepended; the implementation comment says the prepend is skipped when the file already starts at t=0.",
    "evidence": "read_sps.py:39 'Time series arrays (all with t=0 prepended)' and Notes 'All arrays have t=0 prepended with initial values for interpolation' vs read_sps.py:262 '=== t=0 prepend (idempotent - skip if the file already starts at t=0) ==='.",
    "expected": "The post-condition that holds in both branches is 't[0] == 0', not 'a row was prepended'. Array length therefore depends on the input file (N or N+1) - relevant to any code indexing by row.",
    "failure_scenario": "Code assuming index 0 is a synthetic duplicate of index 1 (or assuming len == nrows+1) misbehaves for a file that already starts at t=0.",
    "repro": "",
    "confidence": "high"
  },
  {
    "id": "S10-B-12",
    "file": "trinity/sps/sps_columns.py",
    "line": 152,
    "class": "citation",
    "severity": "S3",
    "claim": "The only provenance recorded for the bundled tables is 'Legacy SB99 ... canonical SB99 export layout' plus the path lib/default/sps/starburst99/1e6cluster_default.csv. No Starburst99 version, IMF, metallicity, time grid, or grid range is documented anywhere in the slice.",
    "evidence": "sps_columns.py:152 and :157 are the only provenance statements; searching the whole slice prose finds no mention of IMF, Kroupa/Chabrier/Salpeter, Z, metallicity, BPASS, or any grid range. read_sps.py:3 only says the file is 'the bundled default file'.",
    "expected": "A published feedback model must state the SPS code version, IMF and metallicity, since Lmech and Qi vary by factors of a few across those choices. At minimum the bundled file header or a docs entry should record them.",
    "failure_scenario": "Results are not reproducible and cannot be compared against other feedback codes; a user swapping in a table computed with a different IMF gets silently different answers with no warning.",
    "repro": "",
    "confidence": "high"
  },
  {
    "id": "S10-B-13",
    "file": "trinity/sps/read_sps.py",
    "line": 286,
    "class": "regime",
    "severity": "S2",
    "claim": "No prose in the slice states what happens when feedback is requested at a time beyond the last row of the SPS table (or before the first): no mention of bounds_error, fill_value, extrapolation, or clamping. The only related claim is that interpolators are scipy.interpolate.interp1d with ftype defaulting to 'cubic'.",
    "evidence": "read_sps.py:286 'Interpolation type for scipy.interpolate.interp1d. Options: linear, cubic (default), quadratic'; read_sps.py:3 'Wraps the 11-array tuple ... in scipy cubic interpolators'; update_feedback.py:99 documents t as 'Current time [Myr]' with no stated validity range.",
    "expected": "An explicit, documented out-of-range policy. interp1d defaults to bounds_error=True (raises); if fill_value='extrapolate' were used, a cubic extrapolation of a log-scale quantity past the table end diverges rapidly and can go negative.",
    "failure_scenario": "A bubble integrated past the table's end time either aborts mid-run with a raw scipy ValueError, or silently receives cubically-extrapolated (possibly negative or exponentially large) luminosities and momentum rates that drive the remainder of the evolution.",
    "repro": "Run a simulation whose stop time exceeds the last time in the bundled table and observe whether it raises or returns extrapolated values; inspect the interp1d construction for bounds_error/fill_value.",
    "confidence": "high"
  },
  {
    "id": "S10-B-14",
    "file": "trinity/sps/update_feedback.py",
    "line": 183,
    "class": "numerical",
    "severity": "S3",
    "claim": "pdotdot_total is computed as a numerical derivative of pdot_total using 'a small timestep' in Myr; the field carries no documented unit and the step size is not stated in prose.",
    "evidence": "update_feedback.py:183 'Numerical derivative of total momentum rate for time evolution'; :184 'Myr (small timestep for derivative)'; update_feedback.py:22 'pdotdot_total : float Time derivative of total momentum rate' with no unit, while every neighbouring field has one.",
    "expected": "Unit [Msun*pc/Myr^3]; a documented step size; and a scheme that cannot evaluate the interpolator outside [t_first, t_last] when t is at either end of the grid.",
    "failure_scenario": "At t = 0 (the prepended row) a backward/centred difference samples t < 0; at the final time it samples t > t_max - either raises under interp1d's default bounds_error or returns extrapolated garbage. A fixed absolute step also gives a noisy derivative across the abrupt SN turn-on.",
    "repro": "Call get_current_sps_feedback at t=0.0 and at the exact last tabulated time and inspect pdotdot_total.",
    "confidence": "medium"
  },
  {
    "id": "S10-B-15",
    "file": "trinity/sps/read_sps.py",
    "line": 286,
    "class": "numerical",
    "severity": "S3",
    "claim": "'Cubic is recommended for small-value interpolations' and cubic is the default ftype for all ten interpolators, including the SN channels.",
    "evidence": "read_sps.py:286 'ftype : str, optional Interpolation type ... cubic (default) ... Cubic is recommended for small-value interpolations.'; read_sps.py:3 'scipy cubic interpolators'.",
    "expected": "A justification, or a per-channel choice. The claim is unsupported: cubic splines through a quantity that switches on abruptly (Lmech_SN and pdot_SN are ~0 until the first SN at ~3-4 Myr, then jump by orders of magnitude) overshoot and ring, producing negative luminosities/momentum rates between knots.",
    "failure_scenario": "Negative interpolated Lmech_SN or pdot_total near the SN turn-on feeds a negative energy/momentum injection into the bubble ODE, and v_mech_total = 2*Lmech_total/pdot_total can blow up or flip sign where pdot_total crosses zero.",
    "repro": "Evaluate fLmech_SN and fpdot_total on a dense grid over the first 10 Myr and assert non-negativity.",
    "confidence": "medium"
  },
  {
    "id": "S10-B-16",
    "file": "trinity/sps/sps_columns.py",
    "line": 386,
    "class": "silent-failure",
    "severity": "S3",
    "claim": "A header row is only detected if it is 'non-blank non-#', is the immediate predecessor of the first data row, has the same token count, and has at least one non-numeric token. A '#'-commented header is therefore never detected.",
    "evidence": "sps_columns.py:386 'A header is \"the non-blank non-# row immediately above data_start that has the same token count as the data row and contains at least one non-numeric token\"'; sps_columns.py:425 'header is the *immediately-preceding* non-blank non-# row'; :440 'only the immediate predecessor counts'; sps_columns.py:447 'str -> data[:, header_names.index(file_column)] (raises if no header detected or name not in header)'.",
    "expected": "Documented and tested behaviour for the common SPS/SB99 convention of a '# time  Qi  ...' commented header: name-based sps_col_* declarations must either resolve or fail with the fillable-template error rather than an IndexError/ValueError from list.index.",
    "failure_scenario": "A user with a '#'-commented header writes sps_col_Qi Qi cgs log and gets a bare 'x is not in list' from header_names.index, with no pointer to the real cause. A blank line between header and data has the same effect if blank lines are skipped before the predecessor test.",
    "repro": "Point sps_path at a file whose only header line starts with '#' and use a name-based column declaration.",
    "confidence": "medium"
  },
  {
    "id": "S10-B-17",
    "file": "trinity/sps/sps_columns.py",
    "line": 447,
    "class": "silent-failure",
    "severity": "S4",
    "claim": "For string file_column the prose documents a raise ('raises if no header detected or name not in header'); for integer file_column it documents only 'int (>= 0) -> data[:, file_column]' with no bounds check against the file's actual column count.",
    "evidence": "sps_columns.py:447 resolution rules; sps_columns.py:214 only constrains the index to be 'a non-negative integer'; read_sps.py:39 promises ValueError 'if the file shape is invalid' without defining invalid.",
    "expected": "An out-of-range integer index should produce the same style of actionable error (file, declared index, actual column count) as the name-resolution path.",
    "failure_scenario": "sps_col_Lbol 9 cgs log against a 7-column file surfaces a raw numpy IndexError from deep in the loader instead of a message naming the .param key.",
    "repro": "Declare a column index >= ncols for a user sps_path file.",
    "confidence": "medium"
  },
  {
    "id": "S10-B-18",
    "file": "trinity/sps/read_sps.py",
    "line": 210,
    "class": "other",
    "severity": "S3",
    "claim": "The FB_* correction pipeline - the physics that turns raw SPS output into injected feedback - is documented only by name ('Thermal efficiency and cold mass corrections are applied to winds and SN', 'same math as the legacy path'). No prose states the algebraic form for FB_thermCoeffWind, FB_mColdWindFrac, FB_thermCoeffSN, FB_mColdSNFrac.",
    "evidence": "read_sps.py:39 params list 'FB_mColdWindFrac, FB_thermCoeffWind : Wind corrections' / 'FB_mColdSNFrac, FB_thermCoeffSN, FB_vSN : SN corrections'; Notes 'Thermal efficiency and cold mass corrections are applied to winds and SN'; read_sps.py:210 '=== WIND corrections (same math as the legacy path) ==='; read_sps.py:224 '=== SN corrections (with user-override pluggability) ==='.",
    "expected": "Each coefficient should be documented as an explicit expression (e.g. whether the cold-mass fraction multiplies as f or (1-f), and whether the thermal coefficient scales Lmech only or also pdot), since it determines v_wind = 2*Lmech_W/pdot_W.",
    "failure_scenario": "A f vs (1-f) inversion in either cold-mass fraction changes the injected mass and hence the wind/SN velocity, with no documentation to check the code against. The claim is unverifiable as written.",
    "repro": "",
    "confidence": "high"
  },
  {
    "id": "S10-B-19",
    "file": "trinity/sps/sps_columns.py",
    "line": 3,
    "class": "other",
    "severity": "S3",
    "claim": "The registry is declared the single source of truth for 'which are mass-scaled', but no prose enumerates which canonicals have mass_scaled=True.",
    "evidence": "sps_columns.py:3 'which are mass-scaled (f_mass = mCluster / sps_refmass)'; sps_columns.py:181 'Mass scaling (multiply by f_mass) is applied separately by the caller, using CANONICALS[canonical].mass_scaled'; nothing else in the slice names a mass-scaled column.",
    "expected": "Extensive quantities (Qi, Lbol, Li, Ln, Lmech_*, pdot_*, Mdot_SN) mass-scaled; intensive ones (t, fi, v_SN) NOT mass-scaled.",
    "failure_scenario": "If fi were mass-scaled, the ionising fraction would exceed 1 for clusters above sps_refmass and Ln would go negative; if v_SN were mass-scaled, SN velocity would scale with cluster mass; if a luminosity were NOT mass-scaled, that channel would be frozen at the reference cluster's value for every run.",
    "repro": "Load with f_mass=1 and f_mass=2 and assert the ratio is exactly 2 for extensive columns and exactly 1 for t, fi, v_SN.",
    "confidence": "high"
  },
  {
    "id": "S10-B-20",
    "file": "trinity/sps/sps_columns.py",
    "line": 279,
    "class": "regime",
    "severity": "S4",
    "claim": "Admission of a gap in the accepted-input space: 'Lmech_total OR Lmech_SN must be present (loader needs at least one to drive the SN pipeline; Mdot_SN alone is not yet a supported entry point here)'.",
    "evidence": "sps_columns.py:279 validate_user_column_map rule 3.",
    "expected": "Declaring only Mdot_SN (plus v_SN) is rejected by validation with the fillable template, not accepted and then silently mis-derived.",
    "failure_scenario": "A user with a mass-loss-rate table declares Mdot_SN and no Lmech, and either gets a confusing error or a zeroed SN channel.",
    "repro": "",
    "confidence": "high"
  },
  {
    "id": "S10-B-21",
    "file": "trinity/sps/sps_columns.py",
    "line": 32,
    "class": "other",
    "severity": "S4",
    "claim": "A local solar-luminosity constant in erg/s is defined here because 'no L_sun constant currently in cvt' - a duplicated physical constant outside the central unit module.",
    "evidence": "sps_columns.py:32 'Solar luminosity in erg/s (no L_sun constant currently in cvt).'",
    "expected": "The literal should match the value used elsewhere in trinity (astropy/cvt) to within round-off; note L_sun is variously quoted as 3.828e33 or 3.846e33 erg/s (0.5% apart).",
    "failure_scenario": "A user declaring a luminosity column in Lsun gets a normalisation that differs by ~0.5% from the same conversion done elsewhere in the code.",
    "repro": "Compare the literal against astropy.constants.L_sun.cgs and against any L_sun used in trinity/_functions/unit_conversions.py.",
    "confidence": "medium"
  },
  {
    "id": "S10-B-22",
    "file": "trinity/sps/sps_columns.py",
    "line": 279,
    "class": "silent-failure",
    "severity": "S2",
    "claim": "The stated validation rules never require Lmech_W or pdot_W, yet the documented pipeline needs both: Lmech_SN_raw <- Lmech_total - Lmech_W and v_wind = 2*Lmech_W/pdot_W.",
    "evidence": "sps_columns.py:279 rules are: all of _REQUIRED_ALWAYS; fi OR (Li AND Ln); Lmech_total OR Lmech_SN; not (Li XOR Ln). read_sps.py:135 'Lmech_SN_raw <- Lmech_total - Lmech_W'; read_sps.py:39 'Wind velocity: v_wind = 2 * Lmech_W / pdot_W'. The contents of _REQUIRED_ALWAYS are never stated in prose (sps_columns.py:92 only says 'Strictly-required canonicals (loader cannot run without them)').",
    "expected": "Lmech_W and pdot_W are members of _REQUIRED_ALWAYS; otherwise a user map with Lmech_total but no Lmech_W passes validation and then fails or silently zeroes the SN split.",
    "failure_scenario": "A user declares Lmech_total, Lbol, fi, Qi, t and omits Lmech_W: validation passes, then Lmech_SN = Lmech_total - (missing) either raises deep in the loader or, if Lmech_W defaults to zero, assigns 100% of mechanical luminosity to SNe from t=0 and makes v_wind a 0/0.",
    "repro": "Build a user column map omitting Lmech_W / pdot_W and run validate_user_column_map, then _read_sps_user.",
    "confidence": "medium"
  },
  {
    "id": "S10-B-23",
    "file": "trinity/sps/read_sps.py",
    "line": 39,
    "class": "other",
    "severity": "S4",
    "claim": "The documented params contract for read_sps omits keys the same docstring's prose says it uses: sps_column_map (named in the summary line) and sps_refmass (implied by f_mass) are absent from the 'params : DescribedDict ... containing:' list.",
    "evidence": "read_sps.py:39 summary 'using the column layout in params['sps_column_map']' vs the Parameters list which enumerates only sps_path, FB_mColdWindFrac, FB_thermCoeffWind, FB_mColdSNFrac, FB_thermCoeffSN, FB_vSN.",
    "expected": "The listed key set should cover every params key actually read, so a caller constructing a minimal params dict does not hit a KeyError.",
    "failure_scenario": "A test or tool builds a params dict from the documented list and fails on a missing sps_column_map.",
    "repro": "",
    "confidence": "high"
  },
  {
    "id": "S10-B-24",
    "file": "trinity/sps/update_feedback.py",
    "line": 22,
    "class": "state",
    "severity": "S4",
    "claim": "SPSFeedback documents 13 fields in a fixed order (t, Qi, Li, Ln, Lbol, Lmech_W, Lmech_SN, Lmech_total, pdot_W, pdot_SN, pdot_total, pdotdot_total, v_mech_total) and promises tuple-unpacking, integer indexing (feedback[0] == feedback.t) and len() over them - while read_sps returns 11 arrays in the same leading order.",
    "evidence": "update_feedback.py:22 attribute list and 'Old style (still works): (t, Qi, Li, Ln, Lbol, ...) = get_current_sps_feedback(t, params)'; :81 '__iter__ Allow unpacking'; :90 '__getitem__ Allow indexing: feedback[0] == feedback.t'; :94 '__len__ Return number of fields'; read_sps.py:39 return list of 11.",
    "expected": "__iter__ / __getitem__ / __len__ all follow the declared dataclass field order, len() == 13, and the first 11 positions match read_sps's return order exactly.",
    "failure_scenario": "Any positional-unpacking caller written against the old 11-tuple shape silently mis-binds if the field order or count changed; a mismatch between __len__ and __iter__ breaks unpacking with a non-obvious error.",
    "repro": "Assert len(feedback) == 13 and tuple(feedback)[0:11] matches the read_sps ordering.",
    "confidence": "high"
  },
  {
    "id": "S10-B-25",
    "file": "trinity/sps/sps_columns.py",
    "line": 376,
    "class": "silent-failure",
    "severity": "S4",
    "claim": "Header-vs-data discrimination uses float-parsability, and the docstring explicitly includes inf/nan as parsable, i.e. numeric.",
    "evidence": "sps_columns.py:376 'True iff s parses as a float (covers integers, decimals, scientific notation, inf/nan). Used to distinguish data rows from header rows.'; sps_columns.py:386 header requires 'at least one non-numeric token'.",
    "expected": "Documented behaviour when a header token is literally 'nan', 'inf', 'NA' or a bare number (e.g. a column named '1'); and when a data row legitimately contains nan.",
    "failure_scenario": "A header whose tokens are all numeric-looking (e.g. '0 1 2 3 4 5 6') is classified as data, shifting data_start up by one row and injecting a bogus first row into the time series; conversely a data row containing 'nan' still parses so pass 1 may accept it as the first data line.",
    "repro": "",
    "confidence": "medium"
  },
  {
    "id": "S10-B-26",
    "file": "trinity/sps/sps_columns.py",
    "line": 386,
    "class": "regime",
    "severity": "S4",
    "claim": "Delimiter detection is a two-way choice: \"',' if the first data line contains a comma; else None (whitespace, the np.loadtxt default)\". Supported formats are stated as '.txt (whitespace) and .csv (comma)'.",
    "evidence": "sps_columns.py:386 delimiter return description; sps_columns.py:447 'Supports: - .txt (whitespace) and .csv (comma) - delimiter auto-sniffed from the first data line.'; read_sps.py:135 'Loads any .txt or .csv file (delimiter auto-sniffed...)'.",
    "expected": "Tab- or semicolon-separated files, and comma-decimal locales, are out of scope and should fail with a clear message rather than mis-parse.",
    "failure_scenario": "A semicolon- or tab-with-comma-decimal file is sniffed as comma-delimited and columns shift, silently remapping every canonical.",
    "repro": "",
    "confidence": "medium"
  },
  {
    "id": "S10-B-27",
    "file": "trinity/sps/read_sps.py",
    "line": 3,
    "class": "other",
    "severity": "S4",
    "claim": "The module docstring describes get_interpolation as wrapping the arrays 'in scipy cubic interpolators on params['sps_f']', while the function's own docstring says it returns a new dict named sps_f and takes no params argument.",
    "evidence": "read_sps.py:3 'get_interpolation(sps, ftype='cubic') Wraps the 11-array tuple returned by read_sps() in scipy cubic interpolators on `params['sps_f']`.' vs read_sps.py:286 signature (sps, ftype) and 'Returns ------- sps_f : dict'.",
    "expected": "Consistent description of whether get_interpolation writes into params or returns a dict the caller stores at params['sps_f'].",
    "failure_scenario": "",
    "repro": "",
    "confidence": "high"
  },
  {
    "id": "S10-B-28",
    "file": "trinity/sps/sps_columns.py",
    "line": 108,
    "class": "citation",
    "severity": "S4",
    "claim": "The documented usage example for the 'cgs' unit alias is `sps_col_Qi 0 cgs log`, placing Qi at column index 0 - which in the documented canonical SB99 layout is the time column.",
    "evidence": "sps_columns.py:112 'sps_col_Qi 0 cgs log'; sps_columns.py:158-159 'col 0: time [yr] (linear)' / 'col 1: log10 Qi [1/s] (log)'.",
    "expected": "Either the example is for an arbitrary user file (harmless but confusing next to the preset table) or it should use index 1 to match the shipped layout.",
    "failure_scenario": "A user copies the example verbatim against an SB99-layout file and loads the time column as the ionising photon rate, exponentiated - producing 10**(1e4..1e7).",
    "repro": "",
    "confidence": "medium"
  }
]
```
