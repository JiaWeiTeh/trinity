# S12a input config — Lens A (what the code does)

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

**Files read (all from the stripped slice copy, comments/docstrings blanked, line numbers preserved):**
`trinity/_input/read_param.py` (519 ln) · `trinity/_input/registry.py` (738 ln) · `trinity/_input/dictionary.py` (1342 ln) ·
`trinity/_input/param_spec.py` (162 ln) · `trinity/_input/fkappa_auto.py` (133 ln) · `trinity/_input/errors.py` (12 ln) ·
`trinity/_input/__init__.py` (empty).

**Shared exception used: yes.** I read the real `trinity/_functions/unit_conversions.py` to pin the unit system —
specifically `convert2au()` (its unit map, its `UnitConversionError` on unknown tokens) and `ndens_au2cgs`. Nothing else
from the real tree was opened.

**Not read:** the real `trinity/` package (other than the one declared exception), `docs/`, `docs/dev/`, `test/`,
`param/`, this slice's `prose.md` and `signatures.md`, any other lens report, `old_doNotRead/`, `outputs/`, `scratch/`,
`tbd/`, `fig/`.

**Structural blind spot to keep in mind while reading this report:** `trinity/_input/default.param` is *not* in my slice,
and it is the file that actually supplies every input parameter's value, unit and info string
(`read_param.py:112`, `:123`, `:167`). Anything that requires comparing the registry's declared defaults/units against
`default.param` I can only describe as a *mechanism*, not exhibit as a concrete mismatch. Likewise
`trinity._output.run_constants` (`RUN_CONST_KEYS`, `METADATA_EXCLUDE`, `DROPPED_IN_V2`), `trinity._output._metadata_io`,
`trinity._output.simulation_end`, `trinity._output.show_run`, `trinity._functions.simplify` and `trinity.sps.sps_columns`
are all outside the slice; findings that depend on them are marked medium/low confidence and say so.

I ran no repo code. I did run a standalone reimplementation of `parse_value` and of the `fkappa_auto` interpolator in the
scratchpad to check behaviour I could not read off the page; those transcripts are quoted below as observations, not as
repo state.

---

## 1. `read_param()` — control flow, in order

`read_param(path2file)` (`read_param.py:43`) is the single entry point. It returns a `DescribedDict` and has substantial
side effects on the filesystem (directories created) and on process-global state (signal handlers, atexit hooks). The
steps, in execution order:

1. **Locate `default.param`** next to the module (`:111-118`); hard `FileNotFoundError` if absent.
2. **Parse `default.param`** into `default_dict[key] = (info, unit, value)` (`:121-171`).
3. **Parse the user file** into `user_dict[key] = value` (`:179-206`).
4. **Reject unknown user keys** against `default_dict` (`:215-225`), then `validate_companions(user_dict)` (`:231`).
5. **Merge** — user value wins, info/unit always come from `default.param` (`:234-242`).
6. **Unit-convert** each numeric value by `cvt.convert2au(unit)` and wrap in `DescribedItem` (`:253-274`).
7. **Snapshot the item identities** into `_default_items_before` (`:286`).
8. **`validate_all(params)`** (`:295`).
9. **Derive the composition constants** from `x_He`/`Z_He`/`Z_He_shell` (`:308-348`).
10. **Warn on `TShell_ion`** out of the caseB range (`:355-363`); **scale `dust_sigma`** by metallicity (`:366-369`);
    **default `model_name`** to the param filename stem (`:372-373`).
11. **Apply SFE**: `mCloud` is *redefined* as post-star-formation gas mass; `mCloud_input` and `mCluster` are added
    (`:386-400`).
12. **`resolve_all(params)`** (`:410`) — runs every spec resolver.
13. **Select the CIE cooling table** by metallicity + integer selector (`:417-429`).
14. **`apply_active_when(params)`** (`:441`) — adds or *removes* profile-conditional keys.
15. **Blanket-mark everything not in `time_varying_keys` as `exclude_from_snapshot`** (`:449-457`).
16. **`materialize_runtime(params)`** (`:472`) — inject every registry spec not yet present.
17. **Stomp check** (`:482-492`), then return (`:499`).

Two orderings in that list are load-bearing and both are, in my reading, wrong-way-round; see §4.

---

## 2. The `.param` grammar as actually implemented

### 2.1 Value coercion (`parse_value`, `read_param.py:72-103`)

```
78	        val_str = val_str.strip()
81	        if val_str.lower() == 'none':
85	        if val_str.lower() == 'true':
87	        elif val_str.lower() == 'false':
91	        try:
92	            return float(val_str)
97	        try:
98	            return float(Fraction(val_str))
99	        except (ValueError, ZeroDivisionError):
103	        return val_str
```

Order: `none` → `True`/`False` → `float()` → `float(Fraction())` → raw string. Sentinel words are **case-insensitive**
(`NONE`, `TRUE` both work); parameter *keys* are case-**sensitive** (plain dict lookup at `:217`, `:236`).

Consequences I verified with a standalone reimplementation:

| input | result |
|---|---|
| `1e7` | `10000000.0` |
| `5/3`, `-6/35` | `1.666…`, `-0.1714…` (via `Fraction`) |
| `nan`, `inf`, `-inf` | `nan`, `inf`, `-inf` — **accepted as numbers, no gate** |
| `1e400` | `inf` — **silent overflow** |
| `1_000` | `1000.0` (Python literal underscores) |
| `1/0` | the **string** `'1/0'` — `ZeroDivisionError` is caught at `:99` and the token falls through to `:103` |
| `3` | `3.0` — **every integer becomes a float** |

The `1/0` case is the sharpest: a divide-by-zero fraction does not error, it becomes a `str` that then flows into
numeric code. The `nan`/`inf` cases mean `mCloud nan` is a legal parameter file.

Because everything numeric becomes a float, downstream integer semantics must re-coerce — e.g.
`int(params['path_cooling_CIE'].value)` at `:423` and `int(item.value)` in `simplify` at `dictionary.py:507`.

### 2.2 Comments, blanks, whitespace

**Default file** (`:129-171`): if a `#` appears anywhere, the line is split at the first `#`. A stripped line starting
with `# INFO:` sets `current_info`; `# UNIT:` sets `current_unit` (brackets stripped, `:143`); anything else keeps only
the pre-`#` text (`:147`). Blank result → skip. `INFO`/`UNIT` are consumed by the **next** key line and reset
(`:170-171`), so a stray second `# INFO:` before a key silently overwrites the first.

**User file** (`:184-206`): `#` truncation only (`:186-187`), no `INFO`/`UNIT` support.

Both parsers use `line.split(None, 1)` — split on any run of whitespace, at most once — so values may contain spaces
(`transition_trigger cooling_balance, blowout` survives intact) but are then `.strip()`ed inside `parse_value`.

There is **no quoting or escaping**: a `#` inside a value truncates the value in both parsers. `path2output /data/run#1`
silently becomes `/data/run`.

### 2.3 Malformed / unknown / missing / duplicate keys

* **Malformed line — asymmetric.** In the default file a line that doesn't split into two fields is **silently skipped**
  (`:158-159 if len(parts) != 2: continue`). In the user file the same line is a hard `ParameterFileError`
  (`:198-202`). Same grammar, two different failure modes.
* **Unknown user key** → collected and raised together with a truncated "available parameters" list (`:215-225`). The
  check is against `default_dict`, i.e. against `default.param` — **not** against the registry.
* **Missing key** → falls back to the `default.param` value (`:240-242`). A key missing from *both* `default.param` and
  the user file is later injected raw by `materialize_runtime` (see §4.2).
* **Duplicate key** → **silently last-wins**, in both files (`:167`, `:206`). No detection, no warning. A user who edits
  `mCloud` by pasting a second line rather than editing the first gets the second value with no signal.

### 2.4 Precedence and the unit conversion

Merge is `read_param.py:234-242`: value from `user_dict` when present, otherwise the default; **`info` and `unit` are
always taken from `default.param`**, never from the user file and never from the registry.

Conversion (`:255-274`):

```
258	        if value is None:
260	            converted_value = None
261	        elif isinstance(value, (int, float)) and not isinstance(value, bool):
262	            conversion_factor = cvt.convert2au(unit)
263	            converted_value = value * conversion_factor
```

Bools are correctly excluded from scaling. `convert2au(None)` returns `1.0`, so an un-annotated key is passed through.
`convert2au` raises `UnitConversionError` on an unknown token, so a typo'd `# UNIT:` line fails loudly — good.

---

## 3. Two declaration sites, one parameter

This is the structural centre of the slice. Every input parameter is declared **twice**:

* in `default.param` — value, `# INFO:`, `# UNIT:` — which is what `read_param` actually consumes;
* in `registry.SPECS` — `default=`, `info=`, `unit=` — which `read_param` **never consults for a key that exists in
  `default.param`**.

`materialize_runtime` (`registry.py:649-661`) and `apply_active_when` (`:608-621`) are the only consumers of
`ParamSpec.default`, `ParamSpec.info` and `ParamSpec.unit`, and both are guarded by "only if the key is not already
present". So for the ~60 input keys that `default.param` supplies:

* `ParamSpec.info` is dead text. The multi-hundred-character prose on `cooling_boost_kappa` (`registry.py:387`) and
  `cooling_boost_fA` (`:388`) is duplicated into `default.param`'s `# INFO:` blocks and only the latter reaches the user.
* `ParamSpec.unit` is decorative. `convert2au` is fed the `default.param` `# UNIT:` string (`read_param.py:166, 262`).
  If `default.param` said `# UNIT: [g]` for `mCloud` while `registry.py:337` says `unit='Msun'`, the number would be
  converted from grams and nothing in the slice would notice.
* `ParamSpec.default` is dead **unless** the key is dropped from `default.param`, at which point it becomes live —
  and it is the wrong type (§4.2).

There is no cross-check in either direction. Nothing enumerates "keys in `REGISTRY` but not in `default.param`" or the
converse. A `default.param` key with no spec gets no validator, no resolver, no `run_const`/`metadata_exclude` flag —
and because of `read_param.py:455-457` it is marked `exclude_from_snapshot` and, not being in `RUN_CONST_KEYS`, is also
absent from `metadata.json`: it vanishes from every output artifact while still steering the run.

Evidence that the registry's *unit* field is not even self-consistent as a machine string: `nCore` uses `'cm**-3'`
(`:346`) while `initial_cloud_n_arr` uses `'1/cm**3'` (`:439`), `nEdge` uses `'1/pc**3'` (`:437`) and `Qi` uses `'1/Myr'`
(`:451`). `convert2au` cannot parse a leading `1` (its token regex is `^([a-zA-Z_][a-zA-Z0-9_]*)…`), so those three
strings would raise if they ever reached it; `'N/A'` and `'dimensionless'` (`:415`, `:418`) likewise. They never do,
because runtime keys skip conversion — but the field is being used as a parseable unit for input keys and as a free-text
label for runtime keys, in one column, with no marker distinguishing the two dialects.

---

## 4. Ordering defects

### 4.1 The stomp guard is identity-based; the actual stomping is value-based

`read_param.py:286` captures the *object identity* of every default-derived `DescribedItem`; `:482-492` raises if any of
them was replaced:

```
482	    _stomped = [
483	        k for k, v_before in _default_items_before.items()
484	        if k in params and params[k] is not v_before
485	    ]
```

The error text is explicit that the intent is to catch "runtime init silently overwrote user-facing default.param
key(s)". The guard is reachable — `read_param.py:320` (`chi_e`), `:333` (`mu_ion_shell`), `:341` (`chi_e_shell`),
`:390` (`mCloud_input`), `:396` (`mCluster`) and `registry.py:319` (`sps_column_map`) all assign whole new
`DescribedItem`s and would trip it if those names appeared in `default.param`.

But four keys are overwritten **by value**, 170 lines earlier, and are invisible to it:

```
316	    params['mu_convert'].value = float(_muH)    * _mH_au
317	    params['mu_atom'].value    = float(_mu_n)   * _mH_au
318	    params['mu_ion'].value     = float(_mu_p)   * _mH_au
319	    params['mu_mol'].value     = float(_mu_mol) * _mH_au
```

`mu_atom`, `mu_ion`, `mu_mol`, `mu_convert` are all declared as settable input parameters (`registry.py:363-366`, with
`default='14/11'` etc.), so they pass the unknown-key check at `:215-225` — and are then discarded. A user who writes
`mu_convert 1.5` in their `.param` gets `1.4·m_H` and no message of any kind. `apply_active_when`'s `params.pop`
(`registry.py:621`) is similarly invisible: the guard's `if k in params` clause explicitly skips deleted keys.

Also value-stomped and therefore unguarded: `dust_sigma` (`:367`/`:369`), `model_name` (`:373`), `mCloud` (`:389`),
`path_cooling_CIE` (`:425`). `mCloud` is intended (SFE), but the guard cannot tell intended from accidental.

### 4.2 `materialize_runtime` runs after validation, resolution and conversion

`validate_all` is at `:295`, `resolve_all` at `:410`, unit conversion at `:255-274` — and `materialize_runtime` is at
`:472`, last. `validate_all` and `resolve_all` both skip absent keys (`registry.py:559-560`, `:583-584`). Therefore any
key that materialises from the registry is:

* never validated,
* never resolved,
* never unit-converted,
* and injected verbatim by `copy.deepcopy(spec.default)` (`registry.py:657`).

Now look at what the registry's input-category defaults actually are — **strings**:

```
337	    ParamSpec(name='mCloud', default='1e7', …)
334	    ParamSpec(name='log_console', default='False', …)
354	    ParamSpec(name='stop_at_rCloud_nSnap', default='None', …)
376	    ParamSpec(name='gamma_adia', default='5/3', …)
```

If `mCloud` were dropped from `default.param`, `params['mCloud'].value` would be the string `'1e7'`. `log_console`
would be the string `'False'` — which is **truthy**. `stop_at_rCloud_nSnap` would be the string `'None'` — truthy, and
its validator (`registry.py:164-186`, which rejects non-numeric values) never runs because the key was absent when
`validate_all` executed. Derived and runtime specs, by contrast, use native types (`mCloud_input` `0.0` at `:413`,
`chi_e` `1.2` at `:415`, `np.array([])` at `:438`). So the registry mixes "string that `parse_value` would have
converted" with "already-converted native value" in one `default=` column, and only the first kind is protected by
`default.param` being complete.

`copy.deepcopy` at `registry.py:657` and `:615` does correctly prevent the mutable defaults (`np.array([])` at `:438-440`,
`[]` at `:527-530`) from being aliased across runs — that hazard is handled.

### 4.3 `apply_active_when` silently deletes user input

```
612	        present = spec.name in params
613	        if active and not present:
620	        elif present and not active:
621	            params.pop(spec.name)
```

`densBE_Omega` (`:344`) is `active_when=_active_densBE`; `densPL_alpha` (`:345`) is `active_when=_active_densPL`. Since
both live in `default.param` (they must, or users could not set them past the `:215-225` check), *every* run pops one of
them. A user who sets `dens_profile densPL` **and** `densBE_Omega 20` has the second line silently deleted at
`read_param.py:441`. `validate_companions` (`registry.py:715-738`) only checks the opposite direction — that declaring
the trigger forces you to declare its companion — so nothing catches the contradiction.

---

## 5. Validators and resolvers

### 5.1 `_validate_ZCloud` pins Z=1 and orphans three metallicity branches

```
99	def _validate_ZCloud(value, params) -> None:
101	    if value != 1:
102	        raise ParameterFileError(
```

`validate_all` runs at `read_param.py:295`, before everything downstream. So execution can only reach later code with
`ZCloud == 1.0`. That makes unreachable:

* `read_param.py:426-429` — the `elif params['ZCloud'].value == 0.15:` Sutherland–Dopita CIE table. Dead.
* `registry.py:277-282` — `if params['ZCloud'].value != 1.0: raise ValueError(…)` inside `_resolve_sps_bundle`. Dead.
* `read_param.py:367` — `dust_sigma = dust_sigma * ZCloud` is a guaranteed multiply by `1.0`. The `else: dust_sigma = 0`
  branch at `:368-369` is reachable only by setting `dust_noZ > 1` (default `0.05`, `registry.py:373`), i.e. never in
  practice.

Both `== 1` (`:417`) and `== 0.15` (`:426`) are exact float equality on a parsed value; `0.15` has no exact binary
representation, so even if the validator were relaxed, `parse_value('0.15')` → `float('0.15')` → the branch would only
match that one literal spelling and not, say, `3/20`, which `Fraction` would render as the same double — actually equal
here, but the pattern is fragile.

### 5.2 `path_cooling_CIE` — silent pass-through and a raw `ValueError`

```
423	        cie_choice = int(params['path_cooling_CIE'].value)
424	        if cie_choice in cie_files:
425	            params['path_cooling_CIE'].value = str(_REPO_ROOT / cie_files[cie_choice])
```

Selector `4` (or `0`, or `7`) is **silently left as the float `4.0`**; the run continues and whatever opens the table
later receives a number where it expects a path. And because `int()` is applied unconditionally, setting
`path_cooling_CIE /my/tables/foo.dat` — which the spec's own name suggests is a path — raises a bare
`ValueError: invalid literal for int()` rather than a `ParameterFileError`. Note this selection logic is inline in
`read_param`, not a resolver, while its sibling `path_cooling_nonCIE` *is* a resolver (`registry.py:393`).

### 5.3 `_resolve_path_cooling_nonCIE` creates the directory it means to read

```
245	    if value == 'def_dir':
246	        return str(_REPO_ROOT / 'lib' / 'default' / 'opiate') + os.sep
247	    path_cooling = str(value)
248	    Path(path_cooling).mkdir(parents=True, exist_ok=True)
249	    return path_cooling
```

This is an **input** directory (the OPIATE/CLOUDY cubes). A typo'd path is silently *created* rather than rejected, and
the failure surfaces much later as "no cooling tables found" at a point far from the cause. Separately: the default
branch appends `os.sep`, the user branch does not — so any downstream string concatenation `path + filename` works for
the default and silently produces a wrong path for a user-set directory.

`_resolve_path2output` (`:237`) also mkdirs during parameter resolution, so an output directory exists even for a run
that a later validator rejects.

### 5.4 `_resolve_sps_bundle` — mixed exception types and cross-key side effects

`registry.py:278` and `:284` raise plain `ValueError` for user-configuration errors while `:311`, three lines further
down the same function, raises `ParameterFileError` for the same class of error. The function also mutates
`params['sps_refmass'].value` (`:309`) and inserts a whole new key `params['sps_column_map']` (`:319`) — a "resolver"
declared as resolving one key that writes three. `sps_refmass` (`:357`) carries `consumed_by='sps_path'` and is
therefore excluded from `materialize_runtime` (`:652-653`), so `:307`'s `params['sps_refmass'].value` is a bare
`KeyError` if `default.param` ever drops it.

### 5.5 `cooling_boost_kappa` has a resolver but no validator; `cooling_boost_fA` has the reverse rigour

`fkappa_auto.py:104`:

```
104	    if not (isinstance(value, str) and value.strip().lower() == "auto"):
105	        return value
```

Any string that is not `auto` is returned **unchanged**. `cooling_boost_kappa atuo` (typo) is accepted as the literal
string `'atuo'` and becomes the f_kappa value for the run. The spec (`registry.py:387`) declares no `validator=`.
Compare `cooling_boost_fA` (`:388`), which has `_validate_cooling_boost_fA` doing a full `float()` + positivity check
(`:128-136`). Two knobs in the same family, opposite levels of input hardening.

### 5.6 A validator that mutates

`_validate_stop_at_rCloud_nSnap` ends with `params['stop_at_rCloud_nSnap'].value = coerced` (`registry.py:186`) — an
int coercion performed inside the *validation* pass, which every other validator treats as read-only. The
resolver mechanism (`resolver=`) exists precisely for value transformation.

### 5.7 Un-validated inputs

`sfe` (`registry.py:338`) has no validator. `read_param.py:386-389`:

```
387	    mCluster = mCloud_input_value * params['sfe'].value
388	    mCloud_after_SF = mCloud_input_value - mCluster
389	    params['mCloud'].value = mCloud_after_SF
```

`sfe = 1.0` gives `mCloud = 0`; `sfe = 1.5` gives a **negative cloud mass**; `sfe` is not checked for being in `(0,1)`
anywhere in the slice. `mCloud` itself is not checked positive or finite (and `parse_value` will happily hand it `nan`).

`TShell_ion` gets only a `logger.warning` (`read_param.py:356-363`) when it leaves the 8000–11000 K window that the
default `caseB_alpha = 2.59e-13` (`registry.py:375`) assumes; `caseB_alpha` is not adjusted and the caller receives no
programmatic signal.

---

## 6. Composition derivation (`read_param.py:308-348`) — arithmetic checks out

```
308	    _xHe = Fraction(params['x_He'].value).limit_denominator(10**6)
310	    _muH    = 1 + 4 * _xHe
311	    _mu_n   = _muH / (1 + _xHe)
312	    _mu_p   = _muH / (2 + _xHe * (1 + _ZHe))
313	    _mu_mol = _muH / (Fraction(1, 2) + _xHe)
314	    _chi_e  = 1 + _ZHe * _xHe
315	    _mH_au  = cvt.convert2au('m_H')
```

At the declared defaults `x_He = 0.1`, `Z_He = 2`, `Z_He_shell = 1`:

* `mu_H = 1 + 4(0.1) = 7/5` ✓ (`registry.py:366` says 1.4)
* `mu_atom = (7/5)/(11/10) = 14/11` ✓ (`:363`)
* `mu_ion = (7/5)/(2 + 0.1·3) = (7/5)/(23/10) = 14/23` ✓ (`:364`)
* `mu_mol = (7/5)/(1/2 + 1/10) = (7/5)/(3/5) = 7/3 = 14/6` ✓ (`:365`)
* `chi_e = 1 + 2(0.1) = 1.2` ✓ (`:415`)
* `mu_ion_shell = (7/5)/(2 + 0.1·2) = (7/5)/(11/5) = 7/11` — `:331` computes exactly this
* `chi_e_shell = 1 + 1(0.1) = 1.1` ✓ (`:417`)

The `Fraction(...).limit_denominator(10**6)` at `:308-309` is the right defensive move: `Fraction(0.1)` is the exact
binary `3602879701896397/36028797018963968`, and `limit_denominator` recovers `1/10` so the mu ratios come out as exact
rationals rather than accumulating float error. `convert2au('m_H')` is `m_H[g] × g2Msun`, so every `mu_*` value is
**Msun per particle** — dimensionally consistent with the `unit='m_H'` labels and with `rho = mu_convert · n_H`.

`chi_e`, `mu_ion_shell`, `chi_e_shell` are created as *new* keys (`:320`, `:333`, `:341`) rather than assigned in place,
which is what makes the `_stomped` guard meaningful for them; `mu_convert`/`mu_atom`/`mu_ion`/`mu_mol` are assigned in
place, which is what defeats it (§4.1).

---

## 7. `fkappa_auto.py`

**Grid and dimensions.** `_LOG_M = log10([1e5, 1e6, 1e7])` (Msun), `_LOG_SFE = log10([0.03, 0.1, 0.3])`
(dimensionless), `_LOG_N = log10([1e2 … 1e5])` (cm⁻³); `_F_FIRE` has shape `(3, 3, 7)` matching `(M, sfe, n)`.
`_INTERP` interpolates `log10(f_kappa)` linearly over log-spaced axes (`:72`), i.e. a piecewise power law in each
variable — the natural choice for a quantity spanning 1→64. f_kappa is a dimensionless multiplier on `C_thermal`
(`registry.py:377`, unit `erg s⁻¹ cm⁻¹ K^(-7/2)`), so no dimensional issue arises.

**The unit conversion at the call site is correct:**

```
121	        params["nCore"].value * cvt.ndens_au2cgs,
```

`nCore` is declared `unit='cm**-3'` (`registry.py:346`), so after `read_param.py:262` it is stored in AU (pc⁻³);
`ndens_au2cgs = 1 / ndens_cgs2au` returns it to cm⁻³ for the lookup against `_LOG_N`. `mCloud_input` (`:118`) is Msun
and matches `_LOG_M`; using `mCloud_input` rather than the SFE-reduced `mCloud` is the consistent choice given the grid
axis label. This is the one place in the slice where a unit conversion is applied by hand, and it is right.

**Clamping is silent-ish and log-of-nonpositive is unguarded.**

```
84	    clamped = np.clip(coords, lo, hi)
85	    if not np.array_equal(coords, clamped):
86	        logger.warning(
```

`sfe = 0` → `np.log10(0)` → `-inf` (plus a numpy RuntimeWarning) → clipped to `log10(0.03)` → the run proceeds with
f_kappa evaluated at sfe = 3%, having only logged a warning. A negative input gives `nan`, and `nan` survives
`np.clip`, fails `array_equal` (so it warns "clamping"), and then hits `RegularGridInterpolator` with
`bounds_error=True` → an opaque "out of bounds" `ValueError`. I confirmed by reimplementation that all 63 exact grid
nodes compare equal after the clip, so there are no spurious clamp warnings at nominal grid points.

**`_C` and a measured `64.0` are indistinguishable.** `_C = F_KAPPA_CEILING = 64.0` (`:45-46`) is used in six cells
(`:55` ×2, `:61`, `:66`, `:67` ×2) while the literal `64.0` appears in five others (`:54`, `:60`, `:61`, `:65`, `:66`).
Given the warning text at `:124-129` — "resolved to the calibration ceiling … no tested f_kappa fired the
cooling_balance trigger in this regime" — `_C` reads as a *censored* cell (the sweep never fired) and the literal
`64.0` as a *measured* value. After `np.array` construction both are the double 64.0, so (a) the ceiling warning fires
identically for censored and measured cells, and (b) the interpolant treats a censored cell as a hard datum
`f_kappa = 64` when interpolating its neighbours — a right-censored bound used as a point estimate. This is an
inference about intent from the warning string, so medium confidence.

**Numerical round-trip.** `10.0 ** _INTERP(...)` at an exact node returns e.g. `32.00000000000001`, not `32.0` —
a log/exp round trip of the tabulated value. `max(1.0, …)` at `:94` is *not* dead code: it repairs undershoot at the
f = 1 nodes. The ceiling test `f_kappa >= 0.999 * F_KAPPA_CEILING` (`:123`) is comfortably outside the round-trip
error, so it is safe.

---

## 8. `dictionary.py`

### 8.1 `DescribedItem`

`__slots__` (`:118`) plus a full arithmetic/comparison protocol (`:174-190`) plus `__array__` (`:193`). Defining
`__eq__` at `:186` without a `__hash__` makes the class **unhashable** — `set()`s and dict keys of `DescribedItem` raise
`TypeError`. Nothing in the slice needs it, but the class advertises value semantics everywhere else. `__eq__` on an
array-valued item returns an array, so `if item == x:` is a truth-value-ambiguity waiting to happen.

### 8.2 `DescribedDict.__init__` mutates process-global state

```
240	        self._register_crash_handlers()
…
284	        atexit.register(atexit_handler)
287	        signal.signal(signal.SIGINT, self._signal_handler)
288	        signal.signal(signal.SIGTERM, self._signal_handler)
```

Three consequences:

1. `signal.signal` can only be called from the main thread — constructing a `DescribedDict` in a worker **thread**
   raises `ValueError`. (Process-based sweep workers are fine.)
2. The previous SIGINT handler is replaced, not chained or restored. Ctrl-C no longer raises `KeyboardInterrupt`; the
   handler calls `sys.exit(128 + signum)` (`:300`).
3. **Every** construction registers another atexit hook, and `DescribedDict.load_snapshot` constructs one
   (`:951 params = cls()`) and immediately gives it a `path2output` (`:954`). So a purely read-only analysis script that
   loads a snapshot will, at interpreter exit, run `_safe_flush("Normal exit / atexit")` → `write_termination_debug_report(output_dir, …)`
   and rewrite `metadata_humanreadable.txt` (`:325-342`) **into the directory of the run it was analysing**. The two
   writers are outside my slice so I cannot say how destructive that is, but the call is unconditional and the target
   directory is the loaded run's.

### 8.3 Snapshot de-duplication misfires at every flush boundary

```
721	        if self.save_count >= 1 and self.previous_snapshot:
722	            last = self.previous_snapshot.get(str(self.save_count - 1), {})
726	                if ("t_now" in last and t_now == last["t_now"]) and ("R2" in last and r2 == last["R2"]):
```

`flush()` ends with `self.previous_snapshot = {}` (`:868`), and `save_snapshot` flushes every
`snapshot_interval = 10` saves (`:219`, `:750-752`). So the snapshot immediately after each flush sees an empty
`previous_snapshot`, the whole guard is skipped, and a duplicate is written. The dedup is also exact float equality on
`t_now` and `R2` — a re-entered step that differs in the last ulp is treated as distinct. The `except KeyError: pass`
at `:729-731` additionally means a snapshot missing `t_now`/`R2` is never deduped, silently.

### 8.4 The saved-array contract is inconsistent, and one key lies about its content

`_clean_for_snapshot` (`:577-706`) rewrites profile arrays. The convention it establishes:

* `bubble_T_arr`, `bubble_n_arr` → written as `log_bubble_T_arr`, `log_bubble_n_arr` (`:648`) — log-transformed, **renamed**.
* `bubble_dTdr_arr` → `log_bubble_dTdr_arr` (`:658`) — `log10(|·|)`, **renamed**, sign discarded (`:655`).
* `bubble_v_arr` → `bubble_v_arr` (`:667`) — linear, name unchanged.
* `shell_n_arr` → `log_shell_n_arr` (`:699`) — log, renamed.
* **`shell_grav_force_m` → `shell_grav_force_m`** (`:683`) — but the value written is
  `np.log10(np.maximum(np.abs(np.asarray(val)), eps))` (`:680`). Log-transformed and absolute-valued, **under the
  unmodified name**, in a file where every other log-transformed array carries a `log_` prefix. A reader that follows
  the prefix convention will treat log₁₀|F| as a linear force.

Two more properties of this block:

* `bubble_T_arr` and `bubble_n_arr` go through the same branch but call `simplify` **independently** (`:646` each), so
  they are decimated onto *different* radius grids of possibly different lengths. Each writes its own companion grid
  (`bubble_T_arr_r_arr`, `bubble_n_arr_r_arr`, `:649`), so the information is recoverable — but zipping the two arrays
  element-wise, the obvious thing to do, silently pairs mismatched radii. `bubble_r_arr` itself is dropped (`:639-641`)
  and reappears as four separately-simplified copies.
* `eps = 1e-300` with `np.maximum(val, eps)` (`:645`, `:697`) means a **negative** density or temperature is silently
  mapped to `log10 = -300` rather than surfacing as an error or NaN.

### 8.5 Serialization guards are asymmetric; a failure loses every pending snapshot

The metadata path wraps each value in a serializability probe and skips-with-warning on failure:

```
840	                try:
841	                    ready = self._to_json_ready_value(item.value)
842	                    json.dumps(ready, cls=NpEncoder)
843	                except (TypeError, ValueError) as e:
844	                    logger.warning(
```

The snapshot path has no such guard — `_to_json_ready_value` falls through to `return val` for any unrecognised type
(`:575`) and `json.dumps(snap_data, cls=NpEncoder)` at `:861` raises. That exception propagates out of `flush()`, and
when `flush()` was reached from `_safe_flush` it is swallowed:

```
321	            except Exception as e:
322	                logger.error(f"Failed to flush snapshots on exit: {e}")
```

— i.e. up to `snapshot_interval` snapshots are discarded with one ERROR line and the process still exits 0.

### 8.6 Stale-artifact window

`flush()` deletes a pre-existing `dictionary.jsonl` / `metadata.json` **only on the first flush** (`:810-816`), and
writes `metadata.json` **only on the first flush** (`:825`). `path2output` defaults to `outputs/<model_name>` and
`model_name` defaults to the param-file stem (`registry.py:330`, `read_param.py:373`), so re-running the same `.param`
reuses the same directory. A run that dies before it ever saves a snapshot never enters `flush()` (guarded by
`if self.previous_snapshot:` at `:316`), so the previous run's `dictionary.jsonl` and `metadata.json` survive intact —
and `_safe_flush` then regenerates `metadata_humanreadable.txt` from them (`:335-342`), producing a fresh-looking
summary of the *old* run.

### 8.7 `simplify()` swallows the real error

```
508	        try:
509	            x_out, y_out = _simplify(x_arr, y_arr, nmin=nmin, grad_inc=grad_inc)
510	        except ValueError:
511	            raise ValueError(
512	                f"simplify(): x and y must have same length for {keyname}. "
513	                f"Instead got {len(x_arr)} and {len(y_arr)}"
514	            )
```

*Every* `ValueError` from `_simplify` — whatever its cause — is reported as a length mismatch, and the original message
is discarded (no `from err`). If the two lengths happen to match, the message contradicts itself.

Also here: `_simplify_error` is computed on every call (`:524`) even though the message is logged only when
`r_squared < 0.9` or during the first two implicit-phase calls (`:528-536`), i.e. on a hot snapshot path. And a
degenerate (constant) array typically yields `r_squared = nan`; `nan < 0.9` is `False`, so it silently takes the
"good fit" branch.

### 8.8 Sticky exclusion, unused parameter, bare except, divergent `updateDict`

* `_excluded_keys` (`:225`) only ever grows — `__setitem__` adds (`:254-255`) and `_clean_for_snapshot` re-adds
  (`:614-617`), nothing removes. Once a key is excluded it can never be re-included, even if a later
  `DescribedItem` for the same name has `exclude_from_snapshot=False`. `read_param.py:457` sets the flag directly on the
  item (bypassing `__setitem__`), which is only benign because `_clean_for_snapshot` re-syncs.
* `_clean_for_snapshot(self, snap_id: int)` (`:577`) never uses `snap_id`.
* `save_debug_snapshot` has a bare `except:` at `:1063`, which also catches `KeyboardInterrupt` and `SystemExit`.
* `updateDict` (`:1232-1278`) has two branches with opposite failure semantics for the same mistake: the dataclass
  branch **silently skips** any field not in the dict (`:1268 if key in dictionary:`), the sequence branch raises
  `KeyError` (`:1278`).
* `load_snapshots` reports parse failures with `print()` (`:900`, `:919`) while the rest of the module uses `logging`.
* The module has **no `__doc__`**: `from __future__ import annotations` is at line 3 and the triple-quoted block runs
  from line 4, so it is a plain expression statement, not a docstring. (This is also why it survived the stripping pass
  that blanked every genuine docstring in the slice.)

### 8.9 `time_varying_keys` vs `run_const` — the two mechanisms contradict each other

```
read_param.py:449	    time_varying_keys = [
450	        'model_name', 'mCloud', 'cool_alpha', 'cool_beta', 'cool_delta',
452	        'nCore', 'nISM', 'rCore', 'dens_profile', 'densPL_alpha',
453	    ]
```

Of these ten, seven carry `run_const=True` in the registry: `model_name` (`:329`), `mCloud` (`:337`), `nCore` (`:346`),
`nISM` (`:347`), `rCore` (`:348`), `dens_profile` (`:342`), `densPL_alpha` (`:345`). `_clean_for_snapshot` drops every
key in `run_const_keys` from every snapshot line (`dictionary.py:631`). If `RUN_CONST_KEYS` is derived from
`registry.run_const_keys()` (`:664-670`) — which I cannot confirm, `trinity._output.run_constants` is outside my slice
— then exempting those seven from the blanket exclusion at `read_param.py:455-457` accomplishes nothing: only
`cool_alpha`, `cool_beta`, `cool_delta` actually differ per snapshot. One list says "these vary with time", another
flag says "these are constant for the run". Related: `COOLING_PHASE_KEYS` (`dictionary.py:1180-1226`) lists
`cool_beta` and `cool_delta` for per-phase reset but **not** `cool_alpha`, a third inconsistent grouping of the same
triple.

---

## 9. Dead code and unused contracts (flagged only — not proposing removal)

* `param_spec.py:80 SENTINEL_PREFIX = "def_"` — never referenced. The five sentinel checks are hard-coded literals:
  `'def_dir'` (`registry.py:233`, `:245`), `'def_path'` (`:275`), `'def_value'` (`:307`), `'def_unset'` (`:395-407`).
* `param_spec.py:33-62` `Category` includes `"deprecated"`, and `:140 deprecated_note` plus the `__post_init__` check at
  `:147-150` enforce it — no spec in `SPECS` uses that category.
* `registry.py:541-543 specs_by_category(...)` — no caller inside the slice (callers may exist outside; low confidence).
* `read_param.py:21 import sys` — unused in the visible code. (`ruff`'s configured rule set is F821/F811/F823/E9, which
  does not include F401, so this would not be caught by the lint gate.)
* `_REPO_ROOT = Path(__file__).resolve().parents[2]` is duplicated verbatim at `read_param.py:37` and `registry.py:68`.
* The twelve `sps_col_*` specs (`registry.py:395-407`) carry `info=''` and, on the default-SPS path, keep the literal
  string `'def_unset'` for the whole run: `consumed_by` excludes them from `materialize_runtime` (`:652-653`) and
  nothing clears them.

## 10. Output provenance gap

`path_cooling_CIE` (`registry.py:392`) and `path_cooling_nonCIE` (`:393`) are `exclude_from_snapshot=True,
metadata_exclude=True` with `run_const` unset; `sps_path` (`:394`) and `sps_refmass` (`:357`) are
`exclude_from_snapshot=True` and not `run_const`. `metadata.json` is written only from `RUN_CONST_KEYS`
(`dictionary.py:828`), and snapshots drop excluded keys (`:624`). So the resolved cooling-table path, the CIE curve
selection, the SPS file and its reference mass are recorded in **neither** output artifact — the run cannot be
reproduced from its own outputs. Medium confidence: `RUN_CONST_KEYS` lives outside my slice.

---

```json
[
  {
    "id": "S12a-A-01",
    "file": "trinity/_input/registry.py",
    "line": 337,
    "class": "divergence",
    "severity": "S3",
    "claim": "Every input parameter is declared twice — in default.param (value + '# INFO:' + '# UNIT:') and in registry.SPECS (default/info/unit) — and read_param consumes ONLY the default.param copy for any key default.param declares. ParamSpec.info and ParamSpec.unit are dead for those keys, and nothing cross-checks the two sites.",
    "evidence": "read_param.py:167 `default_dict[key] = (info, unit, value)`; :262 `conversion_factor = cvt.convert2au(unit)`; :270-274 `params[key] = DescribedItem(value=converted_value, info=info, ori_units=unit_str)` — all three fields come from default.param. registry.py:337 `ParamSpec(name='mCloud', default='1e7', info='The mass of the molecular cloud.', category='input_physical', unit='Msun', run_const=True)`. registry.py:657 (materialize_runtime) uses spec.default/info/unit only when `if spec.name in params: continue` (:654-655) does not fire.",
    "expected": "One source of truth per key, or an explicit consistency check between REGISTRY and default.param (names, units, defaults) at load or in a test.",
    "failure_scenario": "default.param's '# UNIT:' for mCloud drifts to [g] while registry.py:337 still says unit='Msun'. read_param converts the value from grams to Msun (a factor ~5e-34), the registry is never consulted, and no error is raised — the run is silently off by 34 orders of magnitude with a plausible-looking .param file.",
    "repro": "Change the '# UNIT:' line above any key in trinity/_input/default.param and observe that params[key].value changes while registry.SPECS is unchanged and nothing warns.",
    "confidence": "high"
  },
  {
    "id": "S12a-A-02",
    "file": "trinity/_input/read_param.py",
    "line": 472,
    "class": "divergence",
    "severity": "S2",
    "claim": "materialize_runtime() runs AFTER validate_all(), resolve_all() and the unit conversion, so any key injected from the registry is never validated, never resolved and never unit-converted; and the registry's input-category defaults are unparsed strings, including the truthy 'False' and 'None'.",
    "evidence": "read_param.py:255-274 (conversion), :295 `validate_all(params)`, :410 `resolve_all(params)`, :472 `materialize_runtime(params)`. registry.py:559-560 `if spec.name not in params: continue` and :583-584 same in resolve_all. registry.py:657 `copy.deepcopy(spec.default)`. registry.py:334 `ParamSpec(name='log_console', default='False', ...)`; :354 `default='None'`; :337 `default='1e7'`; :376 `default='5/3'`.",
    "expected": "Either registry defaults stored as native post-parse values, or materialization before validation/resolution/conversion so an injected default goes through the same pipeline as a default.param value.",
    "failure_scenario": "A key is dropped from default.param during an edit. materialize_runtime injects the string: log_console becomes the string 'False' (truthy — console logging silently turns ON), stop_at_rCloud_nSnap becomes the string 'None' (truthy, and its validator at registry.py:164-186 never ran because the key was absent at validate_all time), mCloud becomes the string '1e7' and the first arithmetic on it raises a TypeError far from the cause.",
    "repro": "Comment out one input key in trinity/_input/default.param, run `python run.py param/simple_cluster.param`, and inspect type(params[key].value).",
    "confidence": "high"
  },
  {
    "id": "S12a-A-03",
    "file": "trinity/_input/read_param.py",
    "line": 482,
    "class": "divergence",
    "severity": "S2",
    "claim": "The anti-stomp guard compares DescribedItem object IDENTITY, so it cannot see the four in-place value overwrites performed 170 lines earlier by the same function — exactly the class of overwrite its own error message says it exists to catch.",
    "evidence": "read_param.py:286 `_default_items_before = {k: params[k] for k in default_dict if k in params}`; :482-485 `_stomped = [k for k, v_before in _default_items_before.items() if k in params and params[k] is not v_before]`; :488-491 error text 'runtime init silently overwrote user-facing default.param key(s) ... remove the conflicting assignment(s) from Step 6/8/10'. The overwrites it misses: :316-319 `params['mu_convert'].value = float(_muH) * _mH_au` (and mu_atom/mu_ion/mu_mol), :367/:369 dust_sigma, :373 model_name, :389 mCloud, :425 path_cooling_CIE.",
    "expected": "Compare values (or a hash of them), not object identity, if the guard is meant to catch silent overwrites of user-facing keys; or document that in-place .value writes are the sanctioned mechanism.",
    "failure_scenario": "A new derived quantity is added as `params['x_He'].value = ...` instead of a new key. The guard passes, the user's x_He is discarded, and the run proceeds with the derived value — precisely the scenario the RuntimeError text describes.",
    "repro": "Add `params['nCore'].value = 1.0` anywhere between read_param.py:286 and :482 and observe that the guard does not fire.",
    "confidence": "high"
  },
  {
    "id": "S12a-A-04",
    "file": "trinity/_input/read_param.py",
    "line": 316,
    "class": "silent-failure",
    "severity": "S2",
    "claim": "mu_convert, mu_atom, mu_ion and mu_mol are settable input parameters (they pass the unknown-key check) whose user-supplied values are unconditionally discarded and replaced by values derived from x_He/Z_He. No warning is emitted.",
    "evidence": "registry.py:363-366 declare mu_atom/mu_ion/mu_mol/mu_convert as `category='input_constants'` with defaults '14/11','14/23','14/6','1.4'. read_param.py:316-319 `params['mu_convert'].value = float(_muH) * _mH_au` / `params['mu_atom'].value = float(_mu_n) * _mH_au` / `params['mu_ion'].value = float(_mu_p) * _mH_au` / `params['mu_mol'].value = float(_mu_mol) * _mH_au`, with no guard on whether the key was user-set.",
    "expected": "Either reject these keys in a user .param (they are derived, not free), or honour a user override, or at minimum log a warning that the supplied value was superseded.",
    "failure_scenario": "A user studying a helium-poor composition sets `mu_convert 1.2` in their .param. The file loads without error; mu_convert is silently reset to 1.4*m_H; every density-to-mass conversion in the run uses the wrong mean molecular weight and the user believes they changed it.",
    "repro": "Add `mu_convert 1.2` to param/simple_cluster.param and print params['mu_convert'].value after read_param.",
    "confidence": "high"
  },
  {
    "id": "S12a-A-05",
    "file": "trinity/_input/registry.py",
    "line": 621,
    "class": "silent-failure",
    "severity": "S2",
    "claim": "apply_active_when deletes a parameter the user explicitly set, with no warning, whenever it belongs to the non-selected density profile.",
    "evidence": "registry.py:612-621 `present = spec.name in params` / `elif present and not active: params.pop(spec.name)`; :344 `ParamSpec(name='densBE_Omega', ..., active_when=_active_densBE)`; :345 `ParamSpec(name='densPL_alpha', ..., active_when=_active_densPL)`. Called from read_param.py:441. The companion validator (registry.py:715-738) only checks the opposite direction.",
    "expected": "Warn (or raise) when popping a key that was present in user_dict, rather than deleting a user-supplied value silently.",
    "failure_scenario": "A user sweeps both profiles from one template and leaves `densBE_Omega 20` in a densPL run. The key is silently dropped; downstream code that does params['densBE_Omega'] raises a bare KeyError with no connection to the .param file, or a .get() fallback silently uses a different value.",
    "repro": "Set both `dens_profile densPL` and `densBE_Omega 20` in a .param and check `'densBE_Omega' in params` after read_param.",
    "confidence": "high"
  },
  {
    "id": "S12a-A-06",
    "file": "trinity/_input/read_param.py",
    "line": 426,
    "class": "deadcode",
    "severity": "S2",
    "claim": "_validate_ZCloud hard-pins ZCloud == 1, which makes three downstream metallicity branches unreachable and turns the dust_sigma metallicity scaling into a guaranteed multiply by 1.0.",
    "evidence": "registry.py:99-105 `def _validate_ZCloud(value, params): if value != 1: raise ParameterFileError(...)`, wired at :339 and executed at read_param.py:295. Unreachable as a result: read_param.py:426-429 `elif params['ZCloud'].value == 0.15:` (Sutherland-Dopita table); registry.py:277-282 `if params['ZCloud'].value != 1.0: raise ValueError(...)` inside _resolve_sps_bundle; read_param.py:367 `params['dust_sigma'].value = params['dust_sigma'].value * params['ZCloud'].value` is always *1.0, and :368-369 `else: params['dust_sigma'].value = 0` needs dust_noZ > 1 (default 0.05 at registry.py:373).",
    "expected": "Either the Z != 1 support that these branches assume, or an explicit note that they are staged for a future capability; as written, three separate code paths silently claim a capability the validator forbids.",
    "failure_scenario": "Someone relaxes _validate_ZCloud to enable Z=0.15 and assumes the branch chain is exercised. read_param.py:417 `if ... == 1:` / :426 `elif ... == 0.15:` leaves any other Z (0.2, 0.5) with path_cooling_CIE still set to the raw float selector, with no else and no error.",
    "repro": "Set `ZCloud 0.15` in a .param — it raises at validate_all, proving the elif at :426 cannot be reached.",
    "confidence": "high"
  },
  {
    "id": "S12a-A-07",
    "file": "trinity/_input/read_param.py",
    "line": 423,
    "class": "silent-failure",
    "severity": "S2",
    "claim": "An out-of-range path_cooling_CIE selector is silently left as a bare float, and a path-valued setting crashes with an unhandled ValueError from int() instead of a ParameterFileError.",
    "evidence": "read_param.py:418-425 `cie_files = {1: ..., 2: ..., 3: ...}` / `cie_choice = int(params['path_cooling_CIE'].value)` / `if cie_choice in cie_files: params['path_cooling_CIE'].value = str(_REPO_ROOT / cie_files[cie_choice])` — no else. registry.py:392 declares the spec with no validator and no resolver, unlike its sibling path_cooling_nonCIE (:393, resolver=_resolve_path_cooling_nonCIE).",
    "expected": "A validator on path_cooling_CIE restricting it to {1,2,3}, or a resolver mirroring path_cooling_nonCIE, raising ParameterFileError for anything else.",
    "failure_scenario": "`path_cooling_CIE 4` loads without complaint and the value stays 4.0; the cooling loader is later handed the number 4.0 where it expects a filesystem path. `path_cooling_CIE /my/tables/cie.dat` dies with `ValueError: invalid literal for int() with base 10` and no mention of which parameter caused it.",
    "repro": "Set `path_cooling_CIE 4` (silent) or `path_cooling_CIE /tmp/x.dat` (raw ValueError) in a .param.",
    "confidence": "high"
  },
  {
    "id": "S12a-A-08",
    "file": "trinity/_input/read_param.py",
    "line": 91,
    "class": "silent-failure",
    "severity": "S2",
    "claim": "parse_value silently accepts nan, inf and overflow-to-inf as numeric parameter values, and silently converts a divide-by-zero fraction into a string.",
    "evidence": "read_param.py:91-100 `try: return float(val_str) except ValueError: pass` / `try: return float(Fraction(val_str)) except (ValueError, ZeroDivisionError): pass`. Verified with a standalone reimplementation: 'nan'->nan, 'inf'->inf, '1e400'->inf, '1/0'->the string '1/0' (ZeroDivisionError caught at :99, falls to :103 `return val_str`).",
    "expected": "Reject non-finite numeric literals at the trust boundary, and treat '1/0' as a malformed value rather than as a string parameter.",
    "failure_scenario": "`mCloud 1e400` becomes inf; mCluster = inf*sfe = inf, mCloud_after_SF = inf-inf = nan at read_param.py:388, and the run proceeds with a NaN cloud mass. `gamma_adia 1/0` becomes the string '1/0' and the first arithmetic use raises a TypeError with no reference to the parameter file.",
    "repro": "python -c \"from fractions import Fraction; print(float('1e400'))\" and set `mCloud 1e400` in a .param.",
    "confidence": "high"
  },
  {
    "id": "S12a-A-09",
    "file": "trinity/_input/read_param.py",
    "line": 206,
    "class": "silent-failure",
    "severity": "S3",
    "claim": "Duplicate keys in a .param file are silently last-wins in both the user file and default.param; there is no duplicate detection anywhere.",
    "evidence": "read_param.py:206 `user_dict[key] = value` inside the per-line loop, and :167 `default_dict[key] = (info, unit, value)` — plain dict assignment, no membership check.",
    "expected": "Raise ParameterFileError (or warn) on a repeated key, naming both line numbers; line_num is already in scope at :184.",
    "failure_scenario": "A user appends a corrected `sfe 0.1` at the bottom of a file that already sets `sfe 0.01` near the top, or vice versa after a merge conflict. The file loads cleanly and the run uses whichever line came last, with no indication that two values were supplied.",
    "repro": "Put `sfe 0.01` and `sfe 0.1` in the same .param and observe params['sfe'].value == 0.1 with no message.",
    "confidence": "high"
  },
  {
    "id": "S12a-A-10",
    "file": "trinity/_input/read_param.py",
    "line": 158,
    "class": "divergence",
    "severity": "S3",
    "claim": "The same malformed line is silently skipped in default.param but is a hard error in the user file — the two parsers implement different grammars for identical input.",
    "evidence": "read_param.py:156-159 (default file) `parts = line.split(None, 1)` / `if len(parts) != 2: continue`; read_param.py:196-202 (user file) `parts = line.split(None, 1)` / `if len(parts) != 2: raise ParameterFileError(f\"{Path(path2file).name}, line {line_num}: Expected format 'key value', got: '{line}'\")`.",
    "expected": "One shared line-parsing routine, or at least the same diagnosis for the same defect; a valueless key in default.param is a shipped-file bug that should be loud, not silent.",
    "failure_scenario": "An edit leaves a bare key in default.param. That key silently vanishes from default_dict, so any user .param that sets it now fails the unknown-key check at :215-225 with 'Invalid parameter(s)' — a message that points at the user's file instead of at the shipped default.",
    "repro": "Delete the value (keeping the key) from any line in trinity/_input/default.param, then set that key in a user .param and read the resulting error.",
    "confidence": "high"
  },
  {
    "id": "S12a-A-11",
    "file": "trinity/_input/registry.py",
    "line": 248,
    "class": "silent-failure",
    "severity": "S2",
    "claim": "_resolve_path_cooling_nonCIE creates the directory it is supposed to READ cooling tables from, masking a typo'd path; and it appends os.sep only on the default branch, so user-set and default values have different trailing-separator conventions.",
    "evidence": "registry.py:245-249 `if value == 'def_dir': return str(_REPO_ROOT / 'lib' / 'default' / 'opiate') + os.sep` / `path_cooling = str(value)` / `Path(path_cooling).mkdir(parents=True, exist_ok=True)` / `return path_cooling`.",
    "expected": "For an input directory: check existence and raise ParameterFileError if missing. And return the same trailing-separator form on both branches.",
    "failure_scenario": "`path_cooling_nonCIE /data/opiat` (typo) creates an empty /data/opiat and the run fails much later with a confusing 'no cooling cubes' error. Separately, downstream code that concatenates `path + 'filename'` works for the default (trailing sep present) and silently builds '/data/opiatefilename' for a user path.",
    "repro": "Set path_cooling_nonCIE to a non-existent directory and observe it is created rather than rejected.",
    "confidence": "high"
  },
  {
    "id": "S12a-A-12",
    "file": "trinity/_input/registry.py",
    "line": 278,
    "class": "divergence",
    "severity": "S4",
    "claim": "_resolve_sps_bundle raises plain ValueError for two user-configuration errors and ParameterFileError for a third, in the same function; it also mutates two keys other than the one it resolves.",
    "evidence": "registry.py:278 `raise ValueError(f\"ZCloud={...} is not supported with the default SPS fallback ...\")`; :284 `raise ValueError(\"SB99_rotation=0 is not supported ...\")`; :311 `raise ParameterFileError(f\"sps_refmass is required when sps_path is user-set ...\")`. Cross-key writes: :309 `params['sps_refmass'].value = 1e6`; :319 `params['sps_column_map'] = DescribedItem(...)`, while resolve_all only assigns the resolver's own key (:585).",
    "expected": "One exception type for user-config errors (ParameterFileError, which errors.py exists for), and cross-key effects declared rather than performed as a side effect of resolving sps_path.",
    "failure_scenario": "A caller that catches ParameterFileError to print a friendly 'bad parameter file' message lets the ValueError from :278/:284 escape as an unhandled traceback for the same class of user mistake.",
    "repro": "Set SB99_rotation 0 with the default sps_path and compare the exception type to the sps_refmass error.",
    "confidence": "high"
  },
  {
    "id": "S12a-A-13",
    "file": "trinity/_input/param_spec.py",
    "line": 80,
    "class": "deadcode",
    "severity": "S4",
    "claim": "SENTINEL_PREFIX = 'def_' is declared and never used; all five sentinel checks hard-code the literal string instead.",
    "evidence": "param_spec.py:80 `SENTINEL_PREFIX = \"def_\"`. Grep across the slice finds no reader. The literals: registry.py:233 `if value == 'def_dir':`, :245 `if value == 'def_dir':`, :275 `sps_path_is_default = value == 'def_path'`, :307 `if params['sps_refmass'].value == 'def_value':`, :395-407 `default='def_unset'` (x12).",
    "expected": "Either use the constant (e.g. a shared is_sentinel() helper) or note that it is aspirational.",
    "failure_scenario": "The sentinel convention is renamed (say to 'auto_'); SENTINEL_PREFIX is updated, the five literal comparisons are not, and every resolver silently stops recognising its own default.",
    "repro": "grep -rn SENTINEL_PREFIX trinity/ — one definition, zero uses.",
    "confidence": "high"
  },
  {
    "id": "S12a-A-14",
    "file": "trinity/_input/read_param.py",
    "line": 449,
    "class": "divergence",
    "severity": "S3",
    "claim": "The time_varying_keys allow-list contradicts the run_const flags: seven of its ten entries are declared run_const=True and are therefore stripped from every snapshot line anyway, leaving only cool_alpha/cool_beta/cool_delta with any effect.",
    "evidence": "read_param.py:449-457 `time_varying_keys = ['model_name','mCloud','cool_alpha','cool_beta','cool_delta','nCore','nISM','rCore','dens_profile','densPL_alpha']` / `for key, val in params.items(): if key not in time_varying_keys: val.exclude_from_snapshot = True`. run_const=True on: registry.py:329 model_name, :337 mCloud, :342 dens_profile, :345 densPL_alpha, :346 nCore, :347 nISM, :348 rCore. dictionary.py:631 `if key in run_const_keys: continue` drops them from _clean_for_snapshot.",
    "expected": "One mechanism deciding what lands in a snapshot line. If mCloud/nCore genuinely vary in time, they must not be run_const; if they are constants, they do not belong in a 'time varying' list.",
    "failure_scenario": "Someone adds time-dependence to mCloud (mass loading) and relies on time_varying_keys to record it per snapshot. It is silently absent from dictionary.jsonl because run_const=True still routes it to a single metadata.json entry, and the analysis reads a constant.",
    "repro": "Inspect a snapshot line in dictionary.jsonl for 'nCore' / 'mCloud' keys. (Depends on trinity._output.run_constants.RUN_CONST_KEYS mirroring registry.run_const_keys(), which is outside this slice.)",
    "confidence": "medium"
  },
  {
    "id": "S12a-A-15",
    "file": "trinity/_input/dictionary.py",
    "line": 683,
    "class": "divergence",
    "severity": "S2",
    "claim": "shell_grav_force_m is written to the snapshot as log10(|value|) but keeps its unmodified key name, while every other log-transformed array in the same function is renamed with a 'log_' prefix. The sign is also discarded.",
    "evidence": "dictionary.py:678-684 `if key == 'shell_grav_force_m':` / `y_arr = np.log10(np.maximum(np.abs(np.asarray(val)), eps))` / `new_dict[key] = self._to_json_ready_value(np.asarray(new_y))`. Contrast :648 `new_dict['log_' + key] = ...` for bubble_T_arr/bubble_n_arr, :658 for bubble_dTdr_arr, :699 for shell_n_arr; and :667 `new_dict[key] = ...` for bubble_v_arr, which really is linear.",
    "expected": "`new_dict['log_' + key]` for a log-transformed array, matching the four siblings — or a linear write if the name is to stay.",
    "failure_scenario": "Any reader following the file's own naming convention ('log_' prefix means log-space) treats shell_grav_force_m as a linear force per unit mass. A true value of 1e-3 pc/Myr^2 is read as -3 pc/Myr^2 — wrong magnitude and wrong sign.",
    "repro": "Compare the magnitudes of 'shell_grav_force_m' and 'log_bubble_T_arr' in a dictionary.jsonl line.",
    "confidence": "high"
  },
  {
    "id": "S12a-A-16",
    "file": "trinity/_input/dictionary.py",
    "line": 643,
    "class": "other",
    "severity": "S2",
    "claim": "bubble_T_arr and bubble_n_arr are decimated by two independent simplify() calls, so they land on different radius grids (and possibly different lengths) within one snapshot.",
    "evidence": "dictionary.py:643-650 `if key in ('bubble_T_arr','bubble_n_arr'):` / `x_arr = np.asarray(self['bubble_r_arr'].value)` / `new_r, new_y = self.simplify(x_arr, y_arr, keyname=key)` / `new_dict['log_'+key] = ...` / `new_dict[key+'_r_arr'] = ...` — simplify is y-dependent, so the retained subset differs per array. bubble_r_arr itself is dropped at :639-641.",
    "expected": "One shared decimation of the r-grid applied to all bubble profile arrays, or an explicit warning that the companion _r_arr keys are mandatory for interpretation.",
    "failure_scenario": "An analysis zips log_bubble_T_arr with log_bubble_n_arr to build a (n, T) phase diagram of the bubble interior. The two arrays are sampled at different radii (and may differ in length), so every pair is mismatched — silently, since both arrays look well-formed.",
    "repro": "Load a snapshot and compare bubble_T_arr_r_arr with bubble_n_arr_r_arr element-wise.",
    "confidence": "high"
  },
  {
    "id": "S12a-A-17",
    "file": "trinity/_input/dictionary.py",
    "line": 721,
    "class": "state",
    "severity": "S2",
    "claim": "The duplicate-snapshot guard is disabled at every flush boundary because flush() clears previous_snapshot, so a duplicate is written once per snapshot_interval; the comparison is also exact float equality.",
    "evidence": "dictionary.py:721-728 `if self.save_count >= 1 and self.previous_snapshot:` / `last = self.previous_snapshot.get(str(self.save_count - 1), {})` / `if ('t_now' in last and t_now == last['t_now']) and ('R2' in last and r2 == last['R2']): ... return`. dictionary.py:868 `self.previous_snapshot = {}` at the end of flush(); :750-752 flush fires when `self.save_count % self.snapshot_interval == 0` with snapshot_interval = 10 (:219).",
    "expected": "Keep the last emitted (t_now, R2) in a dedicated field that survives flush, and compare with a relative tolerance rather than ==.",
    "failure_scenario": "The solver re-emits an identical state at the same t. Nine times out of ten it is suppressed; on the tenth (right after a flush) the duplicate line is written to dictionary.jsonl. Downstream time-series analysis sees an intermittent zero-length step, and a dt-based derivative divides by zero.",
    "repro": "Call save_snapshot() 11 times without changing t_now/R2 and count the lines in dictionary.jsonl.",
    "confidence": "high"
  },
  {
    "id": "S12a-A-18",
    "file": "trinity/_input/dictionary.py",
    "line": 240,
    "class": "state",
    "severity": "S2",
    "claim": "Constructing any DescribedDict registers a process-global atexit hook and replaces the SIGINT/SIGTERM handlers. Because load_snapshot() constructs one and immediately sets its path2output, a read-only analysis session writes a termination report and metadata_humanreadable.txt into the ANALYSED run's directory at interpreter exit.",
    "evidence": "dictionary.py:240 `self._register_crash_handlers()` in __init__; :281-288 `def atexit_handler(): reason = self._termination_reason or 'Normal exit / atexit'; self._safe_flush(termination_reason=reason)` / `atexit.register(atexit_handler)` / `signal.signal(signal.SIGINT, self._signal_handler)` / `signal.signal(signal.SIGTERM, ...)`; :325-342 _safe_flush unconditionally calls write_termination_debug_report(str(output_dir), ...) and writes `(Path(output_dir) / 'metadata_humanreadable.txt').write_text(...)`; :951 `params = cls()` and :954 `params['path2output'] = DescribedItem(str(path2output), ...)` inside load_snapshot.",
    "expected": "Register crash handlers explicitly for the run that owns the output directory (an opt-in method), not in the constructor that read paths also use. Restore/chain the previous signal handlers.",
    "failure_scenario": "An analysis script loads snapshots from ten completed runs to make a figure. At exit, ten atexit hooks fire and each rewrites metadata_humanreadable.txt and a termination debug report reading 'Normal exit / atexit' into ten finished runs' directories, overwriting their real termination records. Also: Ctrl-C during analysis exits 130 via the handler instead of raising KeyboardInterrupt, and constructing a DescribedDict off the main thread raises ValueError from signal.signal.",
    "repro": "python -c \"from trinity._input.dictionary import DescribedDict; DescribedDict.load_snapshot('outputs/<run>', 0)\" and check the mtime of outputs/<run>/metadata_humanreadable.txt afterwards.",
    "confidence": "high"
  },
  {
    "id": "S12a-A-19",
    "file": "trinity/_input/dictionary.py",
    "line": 810,
    "class": "state",
    "severity": "S2",
    "claim": "Stale outputs are deleted and metadata.json is written only inside flush(), and flush() is skipped when nothing was ever saved — so a run that dies before its first snapshot leaves the previous run's dictionary.jsonl and metadata.json in place and then summarises them as if they were its own.",
    "evidence": "dictionary.py:810-816 `if self.flush_count == 0: if path2jsonl.exists(): path2jsonl.unlink() ... if path2metadata.exists(): path2metadata.unlink()`; :825 `if self.flush_count == 0:` guards the metadata write; :316 `if self.previous_snapshot:` guards the flush() call inside _safe_flush; :335-342 metadata_humanreadable.txt is written regardless. The output directory is reused across runs: registry.py:234 `os.path.join(os.getcwd(), 'outputs', params['model_name'].value)` with model_name defaulting to the .param stem (read_param.py:373).",
    "expected": "Clear stale run artifacts when the output directory is resolved (registry.py:237 already mkdirs there), not lazily on first flush.",
    "failure_scenario": "A re-run of param/simple_cluster.param crashes during initialisation. outputs/simple_cluster/ still holds the previous run's dictionary.jsonl and metadata.json, and the atexit path regenerates metadata_humanreadable.txt from them — so the directory looks like a completed run and the user analyses last week's results believing they are today's.",
    "repro": "Run a config to completion, then re-run it with an input that fails before the first save_snapshot, and inspect outputs/<model>/.",
    "confidence": "medium"
  },
  {
    "id": "S12a-A-20",
    "file": "trinity/_input/dictionary.py",
    "line": 861,
    "class": "silent-failure",
    "severity": "S2",
    "claim": "The metadata writer probes each value for JSON-serializability and skips-with-warning, but the snapshot writer has no such guard; a non-serializable value raises inside flush(), and when flush was reached from _safe_flush the exception is swallowed and every pending snapshot is lost.",
    "evidence": "dictionary.py:840-848 (metadata) `try: ready = self._to_json_ready_value(item.value); json.dumps(ready, cls=NpEncoder) except (TypeError, ValueError) as e: logger.warning('metadata.json: skipping non-serializable key %r (%s)', k, e); continue`. Snapshot path: :575 `return val` (unrecognised type passed through unchanged) and :861 `json_line = json.dumps(snap_data, cls=NpEncoder)` with no try. Swallowed at :317-322 `try: ... self.flush() except Exception as e: logger.error(f'Failed to flush snapshots on exit: {e}')`.",
    "expected": "The same per-key probe on the snapshot path, or at minimum re-raise after logging so the process does not exit 0 having discarded data.",
    "failure_scenario": "A new runtime key holds a scipy interpolator and is not marked exclude_from_snapshot. Every flush raises; during the run the exception propagates from save_snapshot, but on the atexit path it is caught and logged as one ERROR line, up to 10 snapshots are dropped, and the process exits successfully.",
    "repro": "params['x'] = DescribedItem(object()); params.save_snapshot(); params.flush().",
    "confidence": "high"
  },
  {
    "id": "S12a-A-21",
    "file": "trinity/_input/dictionary.py",
    "line": 510,
    "class": "silent-failure",
    "severity": "S3",
    "claim": "simplify() converts every ValueError from _simplify into a fixed 'x and y must have same length' message, discarding the real cause and without exception chaining.",
    "evidence": "dictionary.py:508-514 `try: x_out, y_out = _simplify(x_arr, y_arr, nmin=nmin, grad_inc=grad_inc) except ValueError: raise ValueError(f'simplify(): x and y must have same length for {keyname}. Instead got {len(x_arr)} and {len(y_arr)}')`.",
    "expected": "Check the length precondition explicitly before the call and let any other ValueError propagate (or re-raise `from err`).",
    "failure_scenario": "_simplify raises ValueError for an unrelated reason (non-monotonic x, nmin larger than the input, a NaN). The user is told the arrays have mismatched lengths and is shown two numbers that are equal, sending the debugging in the wrong direction.",
    "repro": "Call params.simplify(x, y, nmin=<larger than len(x)>) with equal-length arrays and read the message.",
    "confidence": "high"
  },
  {
    "id": "S12a-A-22",
    "file": "trinity/_input/dictionary.py",
    "line": 577,
    "class": "deadcode",
    "severity": "S4",
    "claim": "_clean_for_snapshot takes a snap_id parameter it never uses.",
    "evidence": "dictionary.py:577 `def _clean_for_snapshot(self, snap_id: int) -> Dict[str, Any]:` — snap_id appears nowhere in the body (:600-706). Called at :737 `clean_dict = self._clean_for_snapshot(snap_id=snap_id)`.",
    "expected": "Drop the parameter or use it (e.g. stamping the snapshot id into the record).",
    "failure_scenario": "A caller assumes the snapshot id is embedded in the returned dict and relies on line order in dictionary.jsonl instead; snapshot identity is positional only (load_snapshots keys by enumerate index at :892-898), so any dropped or duplicated line silently renumbers every subsequent snapshot.",
    "repro": "Read dictionary.py:577-706.",
    "confidence": "high"
  },
  {
    "id": "S12a-A-23",
    "file": "trinity/_input/dictionary.py",
    "line": 1063,
    "class": "silent-failure",
    "severity": "S4",
    "claim": "save_debug_snapshot uses a bare except:, which also swallows KeyboardInterrupt and SystemExit.",
    "evidence": "dictionary.py:1059-1064 `elif hasattr(params, '__getitem__') and 'path2output' in params:` / `try: out_dir = Path(params['path2output'].value if hasattr(params['path2output'],'value') else params['path2output'])` / `except:` / `out_dir = Path('.')`.",
    "expected": "`except (KeyError, TypeError, AttributeError):`. This is the only bare except in the slice; ruff's configured F-rules do not catch E722.",
    "failure_scenario": "Ctrl-C pressed while save_debug_snapshot resolves the output path is discarded, and the debug snapshot is silently written to the current working directory instead of the run directory.",
    "repro": "Read dictionary.py:1057-1066.",
    "confidence": "high"
  },
  {
    "id": "S12a-A-24",
    "file": "trinity/_input/dictionary.py",
    "line": 1268,
    "class": "divergence",
    "severity": "S3",
    "claim": "updateDict has two branches with opposite failure semantics for the same mistake: the dataclass branch silently skips keys not in the dict, the sequence branch raises KeyError.",
    "evidence": "dictionary.py:1264-1269 `for field in dataclasses.fields(keys_or_dataclass): key = field.name; val = getattr(...); if key in dictionary: dictionary[key].value = val`; :1277-1278 `for key, val in zip(keys, values): dictionary[key].value = val`.",
    "expected": "One policy for an unknown key — either both silently skip or both raise.",
    "failure_scenario": "A dataclass field is renamed (or a new one added) without a matching registry spec. The dataclass path silently drops it and the value never reaches params; the run continues with the previous value and the omission is invisible. The same rename through the sequence path would have raised immediately.",
    "repro": "Pass a dataclass with a field name not present in params and observe no error and no update.",
    "confidence": "high"
  },
  {
    "id": "S12a-A-25",
    "file": "trinity/_input/fkappa_auto.py",
    "line": 55,
    "class": "coefficient",
    "severity": "S2",
    "claim": "The censoring sentinel _C and genuinely measured 64.0 entries in the f_kappa table are the same double, so the interpolant treats a right-censored cell as a hard datum f_kappa = 64 and the ceiling warning cannot distinguish the two.",
    "evidence": "fkappa_auto.py:45-46 `F_KAPPA_CEILING = 64.0` / `_C = F_KAPPA_CEILING`. Sentinel cells: :55 `[_C, _C, 48.0, ...]`, :61 `[_C, 64.0, 48.0, ...]`, :66 `[_C, 48.0, ...]`, :67 `[_C, _C, 48.0, ...]`. Literal-64 cells in the same table: :54/:60 `[64.0, 48.0, ...]` and `[64.0, 32.0, ...]`, :65 `[64.0, 32.0, 12.0, ...]`. Ceiling test at :123 `if f_kappa >= 0.999 * F_KAPPA_CEILING:` with the warning text at :124-129 'no tested f_kappa fired the cooling_balance trigger in this regime'.",
    "expected": "Encode censored cells distinctly (np.nan, or a separate mask array) so the interpolator does not use a lower bound as a point estimate and the ceiling warning fires only for censored regions.",
    "failure_scenario": "A run at (mCloud=3e5, sfe=0.3, nCore=2e2) interpolates between a censored cell (true f_kappa unknown, >= 64) and a measured 48. The result is reported as a calibrated value with no ceiling warning, when part of its support is a lower bound. Conversely a run landing on a genuinely measured 64.0 gets a warning claiming no tested f_kappa fired.",
    "repro": "resolve_fkappa_auto at mCloud_input=1e5, sfe=0.1, nCore=1e2 (literal 64.0) triggers the same ceiling warning as sfe=0.3 (a _C cell).",
    "confidence": "medium"
  },
  {
    "id": "S12a-A-26",
    "file": "trinity/_input/fkappa_auto.py",
    "line": 83,
    "class": "silent-failure",
    "severity": "S2",
    "claim": "fkappa_fire takes log10 of its three inputs without checking positivity: zero or negative mCloud/sfe/nCore produce -inf or nan, which the clip then silently folds onto the grid hull (warning only) or turns into an opaque RegularGridInterpolator bounds error.",
    "evidence": "fkappa_auto.py:83-85 `coords = np.log10([mCloud_input, sfe, nCore])` / `clamped = np.clip(coords, lo, hi)` / `if not np.array_equal(coords, clamped): logger.warning(...)`; :72 constructs the interpolator with default bounds_error=True and no fill_value; :94 `return max(1.0, float(10.0 ** _INTERP(clamped)[0]))`.",
    "expected": "Validate that all three inputs are finite and > 0 before the log, raising ParameterFileError otherwise.",
    "failure_scenario": "sfe = 0 (a legitimate typo for a no-star-formation control run) becomes log10(0) = -inf, is clipped to sfe = 0.03, and the run silently uses the f_kappa calibrated for a 3% efficiency while a numpy RuntimeWarning about divide-by-zero scrolls past. A negative value yields nan, which survives np.clip, fires the 'clamping to the hull' warning, and then raises 'One of the requested xi is out of bounds' with no mention of which parameter.",
    "repro": "resolve_fkappa_auto with sfe=0 and cooling_boost_kappa auto.",
    "confidence": "medium"
  },
  {
    "id": "S12a-A-27",
    "file": "trinity/_input/fkappa_auto.py",
    "line": 104,
    "class": "divergence",
    "severity": "S2",
    "claim": "cooling_boost_kappa has a resolver but no validator, so any string other than 'auto' is returned verbatim as the f_kappa value — while its sibling knob cooling_boost_fA is strictly validated.",
    "evidence": "fkappa_auto.py:104-105 `if not (isinstance(value, str) and value.strip().lower() == 'auto'): return value`. registry.py:387 declares cooling_boost_kappa with `resolver=resolve_fkappa_auto` and no `validator=`; registry.py:388 declares cooling_boost_fA with `validator=_validate_cooling_boost_fA`, which at :128-136 does `try: fA = float(value) except (TypeError, ValueError): raise ParameterFileError(...)` and `if not (fA > 0): raise ParameterFileError(...)`.",
    "expected": "A validator on cooling_boost_kappa accepting either the literal 'auto' or a float >= 1, mirroring cooling_boost_fA.",
    "failure_scenario": "`cooling_boost_kappa atuo` (typo) loads without error and the string 'atuo' becomes the conduction multiplier. Best case it raises a TypeError deep in the bubble-structure solve; worst case a truthiness or string-formatting path treats it as 'boost enabled' and the run silently uses an undefined multiplier.",
    "repro": "Set `cooling_boost_kappa atuo` in a .param and print params['cooling_boost_kappa'].value after read_param.",
    "confidence": "high"
  },
  {
    "id": "S12a-A-28",
    "file": "trinity/_input/registry.py",
    "line": 338,
    "class": "silent-failure",
    "severity": "S2",
    "claim": "sfe has no validator, so sfe >= 1 silently produces a zero or negative post-star-formation cloud mass.",
    "evidence": "registry.py:338 `ParamSpec(name='sfe', default='0.01', info='Star formation efficiency.', category='input_physical', unit=None, exclude_from_snapshot=True, run_const=True)` — no validator. read_param.py:386-389 `mCloud_input_value = params['mCloud'].value` / `mCluster = mCloud_input_value * params['sfe'].value` / `mCloud_after_SF = mCloud_input_value - mCluster` / `params['mCloud'].value = mCloud_after_SF`.",
    "expected": "A validator enforcing 0 < sfe < 1 (and mCloud > 0, finite), consistent with the existing validators for coverFraction (registry.py:189-201) and rCloud_max (:204-216) which do exactly this shape of check.",
    "failure_scenario": "A sweep generates sfe = 1.0 at a grid edge. mCloud becomes exactly 0; the cloud-radius solve divides by zero or returns 0/nan, and the failure surfaces as a numerical error far from the parameter that caused it. sfe > 1 gives a negative cloud mass that may propagate as a physically meaningless but finite number.",
    "repro": "Set `sfe 1.0` in a .param and print params['mCloud'].value after read_param.",
    "confidence": "high"
  },
  {
    "id": "S12a-A-29",
    "file": "trinity/_input/registry.py",
    "line": 186,
    "class": "divergence",
    "severity": "S4",
    "claim": "_validate_stop_at_rCloud_nSnap mutates the parameter it is validating, making it the only validator with a side effect; value transformation is what the resolver mechanism is for.",
    "evidence": "registry.py:180-186 `coerced = int(value)` / `if coerced < 0: raise ParameterFileError(...)` / `params['stop_at_rCloud_nSnap'].value = coerced`. Compare validate_all (:556-561) which passes `params[spec.name].value` positionally and ignores the return, versus resolve_all (:580-585) `params[spec.name].value = spec.resolver(...)`. param_spec.py:157-162 even enforces that resolver and consumed_by are mutually exclusive, showing the two roles are meant to be distinct.",
    "expected": "Perform the int coercion in a resolver, leaving validators read-only.",
    "failure_scenario": "A future refactor makes validate_all idempotent/reorderable or runs validators on a copy for a dry-run; the int coercion is silently lost and stop_at_rCloud_nSnap stays a float, so an `is` / exact-type check downstream behaves differently.",
    "repro": "Read registry.py:164-186 against registry.py:556-561.",
    "confidence": "high"
  },
  {
    "id": "S12a-A-30",
    "file": "trinity/_input/registry.py",
    "line": 439,
    "class": "units",
    "severity": "S4",
    "claim": "ParamSpec.unit mixes two incompatible dialects in one field: machine-parseable conversion strings for input keys and free-text labels for runtime keys, several of which convert2au cannot parse at all.",
    "evidence": "Parseable: registry.py:346 `unit='cm**-3'`, :375 `unit='cm**3 * s**-1'`, :377 `unit='erg * s**-1 * cm**-1 * K**(-7/2)'`. Not parseable by convert2au (whose token regex requires a leading letter/underscore and whose unit_map has no such keys): :437 `unit='1/pc**3'`, :439 `unit='1/cm**3'`, :451 `unit='1/Myr'`, :415 `unit='dimensionless'`, :418 `unit='N/A'`. Also inconsistent spellings of the same dimension: 'cm**-3' (:346) vs '1/cm**3' (:439).",
    "expected": "Either one parseable dialect throughout with an explicit sentinel for dimensionless, or a separate display-label field.",
    "failure_scenario": "A runtime key is promoted to an input key (moved into default.param) and its registry unit string is copied into the '# UNIT:' line. convert2au raises UnitConversionError on '1/pc**3' at read_param.py:262 — the loud case. The quiet case is a reviewer treating 'dimensionless' and 'N/A' as equivalent to a real unit annotation.",
    "repro": "python -c \"import trinity._functions.unit_conversions as c; c.convert2au('1/pc**3')\"",
    "confidence": "high"
  },
  {
    "id": "S12a-A-31",
    "file": "trinity/_input/dictionary.py",
    "line": 225,
    "class": "state",
    "severity": "S4",
    "claim": "_excluded_keys only ever grows: nothing removes a key, so snapshot exclusion is permanently sticky once set, and it is populated from two places that can disagree with the per-item flag.",
    "evidence": "dictionary.py:225 `self._excluded_keys: set[str] = set()`; :254-255 `if value.exclude_from_snapshot: self._excluded_keys.add(key)` in __setitem__; :614-617 `for k, item in self.items(): if isinstance(item, DescribedItem): if item.exclude_from_snapshot: self._excluded_keys.add(k)` in _clean_for_snapshot; :624 `if key in self._excluded_keys: continue`. No discard/remove anywhere. read_param.py:457 `val.exclude_from_snapshot = True` sets the flag directly, bypassing __setitem__.",
    "expected": "Rebuild the set from the items each snapshot (the loop at :614-617 already walks them), or drop the cache and test item.exclude_from_snapshot directly.",
    "failure_scenario": "Code re-assigns a key with a fresh DescribedItem(exclude_from_snapshot=False) to start recording it mid-run. The key stays in _excluded_keys and is silently dropped from every snapshot; the diagnostic the developer added never appears in the output and the absence looks like the physics never fired.",
    "repro": "params['k'] = DescribedItem(1, exclude_from_snapshot=True); params['k'] = DescribedItem(2); then inspect a snapshot for 'k'.",
    "confidence": "high"
  },
  {
    "id": "S12a-A-32",
    "file": "trinity/_input/registry.py",
    "line": 392,
    "class": "other",
    "severity": "S3",
    "claim": "The resolved cooling-table paths, the CIE curve selection, the SPS file path and its reference mass are excluded from BOTH the snapshot lines and metadata.json, so a run's outputs do not record which physics tables produced them.",
    "evidence": "registry.py:392 path_cooling_CIE `exclude_from_snapshot=True, metadata_exclude=True` (run_const unset -> False); :393 path_cooling_nonCIE same; :394 sps_path `exclude_from_snapshot=True` and not run_const; :357 sps_refmass `exclude_from_snapshot=True`, not run_const. dictionary.py:828 `for k in RUN_CONST_KEYS:` is the only source of metadata.json entries; :624 `if key in self._excluded_keys: continue` drops them from snapshots.",
    "expected": "Record the resolved table/SPS provenance in metadata.json (that is what a run-constant is for), even if the raw sentinel values are excluded.",
    "failure_scenario": "Two runs are compared six months apart. One used coolingCIE_3_Gnat-Ferland2012.dat and the bundled 1e6cluster SPS file, the other a user-supplied SPS file with a different sps_refmass. Neither output records which, so the discrepancy is unattributable without the original .param files.",
    "repro": "grep for 'sps_path' or 'path_cooling_CIE' in outputs/<run>/metadata.json and dictionary.jsonl. (Depends on trinity._output.run_constants, outside this slice.)",
    "confidence": "medium"
  },
  {
    "id": "S12a-A-33",
    "file": "trinity/_input/dictionary.py",
    "line": 186,
    "class": "other",
    "severity": "S4",
    "claim": "DescribedItem defines __eq__ without __hash__, so Python sets __hash__ = None and the class is unhashable despite its otherwise complete value-semantics protocol.",
    "evidence": "dictionary.py:118 `__slots__ = ('_value','info','ori_units','exclude_from_snapshot')`; :186 `def __eq__(self, other): return self.value == self._unwrap(other)`; no __hash__ anywhere in the class (:98-194).",
    "expected": "Define __hash__ explicitly (or __hash__ = None deliberately) so the intent is stated rather than inherited from a language rule.",
    "failure_scenario": "Code that builds `set(params.values())` or uses items as dict keys raises `TypeError: unhashable type: 'DescribedItem'`. Separately, __eq__ on an array-valued item returns an array, so `if item == 0:` raises 'truth value of an array is ambiguous'.",
    "repro": "python -c \"from trinity._input.dictionary import DescribedItem; hash(DescribedItem(1))\"",
    "confidence": "high"
  },
  {
    "id": "S12a-A-34",
    "file": "trinity/_input/dictionary.py",
    "line": 4,
    "class": "other",
    "severity": "S4",
    "claim": "dictionary.py has no module __doc__: the future-import at line 3 precedes the triple-quoted block at lines 4-44, so that block is a plain expression statement, not a docstring.",
    "evidence": "dictionary.py:3 `from __future__ import annotations` followed by :4 `\"\"\"` opening the module description block that runs to :44. A module docstring must be the first statement in the module. (This is also why that block survived the docstring-stripping pass applied to the rest of the slice.)",
    "expected": "Move the string above the future import so `help(trinity._input.dictionary)` and Sphinx pick it up.",
    "failure_scenario": "Sphinx automodule and help() show an empty module description for the file that documents the on-disk output format (dictionary.jsonl / metadata.json layout) — the single most useful docstring in the package for anyone reading outputs.",
    "repro": "python -c \"import trinity._input.dictionary as d; print(d.__doc__)\" -> None",
    "confidence": "high"
  },
  {
    "id": "S12a-A-35",
    "file": "trinity/_input/param_spec.py",
    "line": 61,
    "class": "deadcode",
    "severity": "S4",
    "claim": "The 'deprecated' category and its deprecated_note contract are declared and enforced but used by no spec.",
    "evidence": "param_spec.py:61 `\"deprecated\",` in the Category Literal; :140 `deprecated_note: Optional[str] = None`; :147-150 `if self.category == 'deprecated' and not self.deprecated_note: raise ValueError(f\"{self.name}: category='deprecated' requires deprecated_note\")`. No entry in registry.SPECS (:329-533) uses category='deprecated'.",
    "expected": "Flag only — an unused contract, not a defect.",
    "failure_scenario": "None today. The contract exists but nothing exercises the branch, so a bug in it would not surface until the first deprecation.",
    "repro": "grep -n \"deprecated\" trinity/_input/registry.py",
    "confidence": "high"
  },
  {
    "id": "S12a-A-36",
    "file": "trinity/_input/read_param.py",
    "line": 187,
    "class": "silent-failure",
    "severity": "S3",
    "claim": "There is no quoting or escaping in the .param grammar, so a '#' inside a value silently truncates it in both parsers.",
    "evidence": "read_param.py:186-187 (user file) `if '#' in line: line = line[:line.find('#')]`; read_param.py:131-133/:147 (default file) `comment_pos = line.find('#')` / `before_comment = line[:comment_pos].strip()` / `line = before_comment`.",
    "expected": "Either document that '#' cannot appear in a value, or support quoting; path- and label-valued parameters (path2output, sps_path, model_name, transition_trigger) are all plausible carriers of a '#'.",
    "failure_scenario": "`path2output /scratch/run#3` silently resolves to /scratch/run, and _resolve_path2output (registry.py:237) creates that directory. Two sweep members writing to run#3 and run#4 both land in /scratch/run and overwrite each other's dictionary.jsonl.",
    "repro": "Set `model_name run#3` in a .param and print params['model_name'].value.",
    "confidence": "high"
  },
  {
    "id": "S12a-A-37",
    "file": "trinity/_input/read_param.py",
    "line": 37,
    "class": "divergence",
    "severity": "S4",
    "claim": "_REPO_ROOT is computed independently in two files of the same package.",
    "evidence": "read_param.py:37 `_REPO_ROOT = Path(__file__).resolve().parents[2]` and registry.py:68 `_REPO_ROOT = Path(__file__).resolve().parents[2]` — identical expressions, both relying on the module living exactly two levels below the repo root.",
    "expected": "One definition imported by the other (or a package-level constant).",
    "failure_scenario": "The package is re-nested (e.g. src/trinity/_input/) and only one of the two parents[2] indices is updated. The CIE table paths built in read_param.py:425 and the SPS/opiate paths built in registry.py:246/:291 then point at different roots, and only one of them fails loudly.",
    "repro": "grep -rn '_REPO_ROOT =' trinity/_input/",
    "confidence": "high"
  },
  {
    "id": "S12a-A-38",
    "file": "trinity/_input/registry.py",
    "line": 541,
    "class": "deadcode",
    "severity": "S4",
    "claim": "specs_by_category() has no caller inside this slice.",
    "evidence": "registry.py:541-543 `def specs_by_category(*categories: Category) -> Iterable[ParamSpec]: cat_set = set(categories); return (s for s in SPECS if s.category in cat_set)`. No reference in read_param.py, dictionary.py, param_spec.py, fkappa_auto.py or elsewhere in registry.py.",
    "expected": "Flag only. Callers may exist in modules outside this slice (docs generation, _output, tools/), which is why confidence is low.",
    "failure_scenario": "None demonstrable from the slice.",
    "repro": "grep -rn 'specs_by_category' trinity/ tools/ docs/",
    "confidence": "low"
  },
  {
    "id": "S12a-A-39",
    "file": "trinity/_input/dictionary.py",
    "line": 645,
    "class": "numerical",
    "severity": "S2",
    "claim": "The snapshot log-transform floors values at eps = 1e-300 via np.maximum, so a negative density or temperature is silently written as log10 = -300 instead of surfacing as an error or NaN.",
    "evidence": "dictionary.py:620 `eps = 1e-300`; :645 `y_arr = np.log10(np.maximum(np.asarray(val), eps))` for bubble_T_arr/bubble_n_arr; :697 the same for shell_n_arr; :655 `y_arr = np.log10(np.maximum(np.abs(v), eps))` and :680 `np.log10(np.maximum(np.abs(np.asarray(val)), eps))`, which additionally discard the sign.",
    "expected": "np.maximum is the right guard for an underflow to exactly zero, but a negative temperature or density is a physics failure and should be detected, not floored.",
    "failure_scenario": "A bubble-structure solve returns a small negative density in one zone. The snapshot records log10 n = -300 there; every plot shows a sharp but finite dip, and the sign error is invisible in the archived output — the only place a post-hoc analysis could have found it.",
    "repro": "params['bubble_n_arr'].value = np.array([-1.0, 1.0]); save_snapshot(); inspect log_bubble_n_arr.",
    "confidence": "high"
  },
  {
    "id": "S12a-A-40",
    "file": "trinity/_input/read_param.py",
    "line": 356,
    "class": "silent-failure",
    "severity": "S3",
    "claim": "An out-of-range TShell_ion produces only a log warning; caseB_alpha is not adjusted and no caller receives a programmatic signal that the Stroemgren balance is internally inconsistent.",
    "evidence": "read_param.py:355-363 `_T_shell_ion = params['TShell_ion'].value` / `if not (8000.0 <= _T_shell_ion <= 1.1e4): logger.warning(f\"TShell_ion = {...:.4g} K is outside the ~8000-11000 K range that the default caseB_alpha ... assumes. alpha_B is temperature-dependent (~T^-0.7) and is NOT auto-adjusted ...\")`. registry.py:375 `ParamSpec(name='caseB_alpha', default='2.59e-13', ...)` with no validator and no coupling to TShell_ion.",
    "expected": "Either derive caseB_alpha from TShell_ion (the T^-0.7 scaling is stated right there in the message), or make the mismatch a hard error unless the user also sets caseB_alpha explicitly.",
    "failure_scenario": "A user sets TShell_ion 2e4 for a hotter HII region. The warning scrolls past in a long run log (log_console defaults to False per registry.py:334, so it may only reach the .log file). alpha_B stays at the 1e4 K value, roughly 1.6x too large for 2e4 K, and n_IF_Str, P_HII and F_HII are all biased with no trace in the output artifacts.",
    "repro": "Set `TShell_ion 2e4` and check that read_param returns normally.",
    "confidence": "high"
  }
]
```
