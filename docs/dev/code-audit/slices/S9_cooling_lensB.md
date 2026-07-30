# S9 cooling — Lens B (what the code claims)

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

Source: `/tmp/.../phase2/S9_cooling/prose.md` (comments + docstrings only, no code seen).
Files covered: `trinity/cooling/net_coolingcurve.py`, `trinity/cooling/CIE/read_coolingcurve.py`,
`trinity/cooling/non_CIE/read_cloudy.py`.

I cannot see any implementation. Everything below is a **claim the prose makes**, recorded so
another lens can test it. Where the prose is silent, I say so explicitly rather than guessing.

---

## 1. Formulas the prose states

| # | Claim | Citation |
|---|---|---|
| F1 | `get_dudt` produces a **net** cooling rate combining CIE and non-CIE conditions ("NET cooling rate (dudt) curve containing both CIE and non-CIE conditions"). | `trinity/cooling/net_coolingcurve.py:3` |
| F2 | Branch structure: `Lambda(T)_CIE` if T above threshold; `Lambda(n,T,phi)` if below; if between "max of nonCIE and min of CIE", **interpolate between the two Lambda values**. | `net_coolingcurve.py:88`–`91` |
| F3 | Non-CIE net cooling grid = `cooling_nonCIE.datacube - heating_nonCIE.datacube` (cooling **minus** heating). Marked "depreciated" but stated twice, once per branch. | `net_coolingcurve.py:144`, `net_coolingcurve.py:175` |
| F4 | The non-CIE lookup is a `scipy.interpolate.RegularGridInterpolator` over axes **in the order (ndens, temp, phi)**. | `net_coolingcurve.py:146`, `net_coolingcurve.py:177`; ordering restated as "go from ndens, then T, then phi" at `trinity/cooling/non_CIE/read_cloudy.py:225` and `:243` |
| F5 | Interpolator arguments must be **log10** values ("remember that these have to be logged!"); the cube axes are log10 arrays. | `net_coolingcurve.py:148`; `read_cloudy.py:23` |
| F6 | The returned rate carries a **negative sign by convention** "since the rate of change is negative due to net cooling". Stated **only inside the non-CIE branch**. | `net_coolingcurve.py:155` |
| F7 | Interpolation-band evaluation: take "the maximum point of non-CIE" (i.e. evaluate non-CIE at its table-edge T), compute the CIE side separately, then interpolate. The debug print shows the interpolation is over `np.log10(T)` against `[nonCIE_Tcutoff, CIE_Tcutoff]` with values `[dudt_nonCIE, dudt_CIE]` — i.e. **linear in dudt, linear in log10 T**. | `net_coolingcurve.py:171`–`172`, `:181`–`193` |
| F8 | CIE: "the simple case when CIE is achieved, so **Lambda depends only on T**". | `trinity/cooling/CIE/read_coolingcurve.py:19` |
| F9 | CIE interpolant signature: a callable mapping **log K → log Lambda**; `get_Lambda` returns **Lambda** (not log Lambda). Only step documented is "change temperature to log for interpolation". | `CIE/read_coolingcurve.py:26`–`54`, `:59` |
| F10 | Cutoff definitions: `nonCIE_Tcutoff` = max of the non-CIE grid `.temp`; `nonCIE_Tmin` = min of it; `CIE_Tcutoff` = `min(logT_CIE[logT_CIE > 5.5])`. All in log10 space. | `net_coolingcurve.py:33`–`39`, `:49`, `:115` |
| F11 | CLOUDY table columns: some are "derived quantities in CLOUDY output"; heating/cooling column **signs are forced positive**. | `read_cloudy.py:189`, `:192` |
| F12 | Cube axis ticks are built by: sort → take log10 ("because the original was created in log space") → **round** ("because it makes things easier"); cube filling then does an index lookup of each row against those ticks. | `read_cloudy.py:208`–`212`, `:232`, `:249` |

### The density factor — **the prose never states it**

This is the single most conspicuous gap in the slice. The prose asserts the two ends of the
conversion but never the multiplier between them:

- Λ is `erg*cm^3/s` in cgs (`net_coolingcurve.py:59`–`76`, `:163`; `CIE/read_coolingcurve.py:26`–`54`);
- `dudt` is `erg/cm^3/s` in cgs (`net_coolingcurve.py:59`–`76`, `:154`, `:179`, `:187`).

Nowhere is `n²`, `n_e·n_H`, `n_tot²`, or `n` written down as the factor that bridges them. The only
constraint the prose imposes is dimensional (erg cm³ s⁻¹ × cm⁻⁶ = erg cm⁻³ s⁻¹ ⇒ *something*
squared-density-like), which cannot distinguish `n_tot²` from `n_e·n_H` — a factor that differs by
tens of percent in ionized gas and by orders of magnitude in neutral gas. **A checker must read the
multiplier off the code; the documentation provides no cross-check.** See finding S9-B-01.

---

## 2. Units the prose states

| Quantity | Stated unit | Citation |
|---|---|---|
| `get_dudt` input `age` | Myr | `net_coolingcurve.py:59`–`76` |
| `get_dudt` input `ndens` | 1/pc³ (code units) | `net_coolingcurve.py:59`–`76` |
| `get_dudt` input `T` | K | `net_coolingcurve.py:59`–`76` |
| `get_dudt` input `phi` | 1/pc²/Myr | `net_coolingcurve.py:59`–`76` |
| `get_dudt` return | code units **Msun/pc/Myr³**, "computed in cgs as erg/cm³/s, then converted via `cvt.dudt_cgs2au`" | `net_coolingcurve.py:59`–`76` |
| Λ | `erg*cm^3/s` cgs ⇔ **Msun*pc^5/Myr³** code units | `net_coolingcurve.py:59`–`76` |
| Required conversions inside `get_dudt` | ndens "pc-3 to cm-3"; phi "1/pc2/Myr to 1/cm2/s" — but **both conversion lines are commented out**, so only the intent is documented | `net_coolingcurve.py:80`–`83` |
| Non-CIE cube values | "Cooling rate is in units of **[erg cm3 / s]**" | `read_cloudy.py:23`–`46` |
| Non-CIE branch result | annotated **`u.erg / u.cm**3 / u.s`** | `net_coolingcurve.py:154`, `:179` |
| CIE branch Λ | "# erg/s * cm3", then `.to(u.erg / u.cm**3 / u.s)` | `net_coolingcurve.py:163`, `:187` |
| Cube axes | `.ndens` log cm⁻³, `.temp` log K, `.phi` log cm⁻² s⁻¹ | `read_cloudy.py:23`–`46` |
| `create_cubes` axes | `log_ndens_arr [cm-3]`, `log_temp_arr **[T]**` (not "K"), `log_phi_arr [cm-2s-1]`, all "in log space" | `read_cloudy.py:143`–`167` |
| `get_coolingStructure` age | `params['t_now']` in **Myr**, converted to **yr** internally | `read_cloudy.py:23`–`46` |
| `get_filename` age | **yr** | `read_cloudy.py:271`–`284` |

**Internal consistency of the code-unit statement (checkable arithmetic, prose-only):** with energy
= Msun pc²/Myr², erg cm⁻³ s⁻¹ ↔ Msun pc⁻¹ Myr⁻³ ✓ and erg cm³ s⁻¹ ↔ Msun pc⁵ Myr⁻³ ✓. The docstring's
two code-unit statements are mutually consistent and consistent with a squared-density multiplier.

**Inconsistency:** the non-CIE table is documented as **erg cm³ s⁻¹** (`read_cloudy.py:23`) but the
value read out of it in the non-CIE branch is annotated **erg cm⁻³ s⁻¹** (`net_coolingcurve.py:154`),
with no stated multiplier in between. See S9-B-02.

---

## 3. Table claims

### Non-CIE (CLOUDY / "opiate") cubes
- Filename convention: `opiate_cooling_[rotation]_Z[metallicity]_age[age].dat`; "Right now, only solar
  metallicity and rotation is considered" (`read_cloudy.py:285`–`286`) — contradicted 7 lines later by
  the 0.15-solar branch.
- Metallicities: "solar, Z = 0.014" (`read_cloudy.py:295`); "0.15 solar, Z = 0.002" (`:298`).
- **Available ages: 1e6, 2e6, 3e6, 4e6, 5e6, 1e7 yr** (`read_cloudy.py:61`). Note the 5e6→1e7 gap.
- Age handling: "find the nearest available age" (`:62`) **vs** "if age is between files, we find the
  nearest higher age and lower age neighbour, and do interpolation … e.g., if age = 2.3, do
  interpolation from 2 and 3" (`:336`–`337`). Outside the range: "use the max/min instead" (`:307`–`308`).
- Filename parsing: ages parsed out of the filename as e.g. `'1.00e+06'` (`:82`, `:314`, `:352`);
  "if in array, use the file" (`:318`) — i.e. exact match against the parsed age list.
- Cube shape: `(n_ndens, n_temp, n_phi)`, "**e.g. (33, 21, 22) for the bundled Z1.00 tables**"
  (`read_cloudy.py:227`). **No axis ranges, no resolutions, no dex spacing are stated anywhere.**
- **NaNs are present in both cubes**: "Some are NaN, because they are not available in the cooling
  table (perhaps non-physical)" (`read_cloudy.py:143`–`167`), and handling them is an unimplemented
  "Future TODO" (`:258`–`259`).
- Cube persistence: "Does the cube already exist?" (`:169`) and "Final step: save into an array to
  save time in the future" (`:263`–`264`).
- Behaviour outside table bounds — **temperature only**: a sub-table T is **clamped up** to
  `nonCIE_Tmin` "so it degrades to the table edge via the non-CIE branch below instead of falling
  through to the raise" (`net_coolingcurve.py:124`–`129`). Nothing is stated about out-of-bounds
  **density** or **phi**. The same comment says this "replaces a hard-coded 1e4 floor that
  over-floored the whole valid `[10**nonCIE_Tmin, 1e4)` **decade**" — the word "decade" implies
  `10**nonCIE_Tmin == 1e3 K`, but the value is never written down.
- Non-CIE upper temperature bound: "non-CIE curve is only up to **10^5.5 K**" (`net_coolingcurve.py:88`),
  restated as "we take the cutoff at **10e5.5 K**" (`:114` — literally 10^6.5; a typo).

### CIE curves
Library menu, selected via `path_cooling_CIE` in the `.param`, bundled under `lib/default/CIE/`
(`CIE/read_coolingcurve.py:26`–`54`):
1. CLOUDY cooling curve for HII region, solar metallicity.
2. CLOUDY cooling curve for HII region, solar metallicity — "Includes the evaporative (sublimation)
   cooling of icy interstellar grains (occurs e.g., when heated by cosmic-ray particle)".
3. Gnat and Ferland 2012 ("slightly interpolated for values").
4. Sutherland and Dopita 1993, for [Fe/H] = −1. "**Auto-pinned when ZCloud == 0.15 regardless of
   `path_cooling_CIE`.**"

**No temperature range, resolution, or CLOUDY version is stated for any CIE table.** Behaviour
outside bounds: "Might be a problem here because this does not support extrapolation. If this
happens, implement a function that does that." (`CIE/read_coolingcurve.py:56`–`57`) — i.e. an
explicit admission that out-of-range T is unhandled.

---

## 4. Citations, and exactly what is attributed

| Citation | Attributed to | Where |
|---|---|---|
| CLOUDY (no version) | CIE library 1: "cooling curve for HII region, solar metallicity" | `CIE/read_coolingcurve.py:26`–`54` |
| CLOUDY (no version) | CIE library 2: same description **verbatim** as library 1, plus evaporative/sublimation cooling of icy interstellar grains | `CIE/read_coolingcurve.py:26`–`54` |
| Gnat & Ferland 2012 | CIE library 3, "slightly interpolated for values" | `CIE/read_coolingcurve.py:26`–`54` |
| Sutherland & Dopita 1993 | CIE library 4, "[Fe/H] = −1"; auto-pinned at `ZCloud == 0.15` | `CIE/read_coolingcurve.py:26`–`54` |
| CLOUDY / "Opiate" | the non-CIE time-dependent `opiate_cooling_*.dat` tables (no version, no paper) | `read_cloudy.py:189`, `:285`, `:346` |
| Old code lineage | `coolnoeq.cool_interp_master()` → `net_coolingcurve.py` (`:3`); `cool.py` → `CIE/read_coolingcurve.py` (`:3`); `coolnoeq.py` → `read_cloudy.py` (`:3`); `create_onlycoolheat()`, `Cool_Struc['Cfunc']`, `Cool_Struc['Hfunc']` → the interpolation functions (`read_cloudy.py:97`) | — |
| `docs/dev/magic-numbers/` | the assertion that the bubble ODE never sends T below 3e4 K | `net_coolingcurve.py:129` |
| `run_energy_implicit_phase.py` | the ~5e4 yr cooling-structure refresh cadence | `net_coolingcurve.py:96` |

---

## 5. Regimes — the stated switch, limits, and continuity

- **Switch rule** (three branches):
  - `T <= nonCIE_Tcutoff` → non-CIE, `Lambda(n, T, phi)` (`net_coolingcurve.py:137`);
  - `T >= CIE_Tcutoff` → CIE, `Lambda(T)` (`:158`);
  - in between → interpolate (`:167`, `:190`).
- **Thresholds**: `nonCIE_Tcutoff` = max of the non-CIE temp grid; `CIE_Tcutoff` =
  `min(logT_CIE[logT_CIE > 5.5])` — i.e. the non-CIE edge is **table-derived** while the CIE edge is
  anchored to a **hard-coded 5.5** (`:33`–`39`, `:49`).
- The prose asserts the two coincide: the non-CIE table tops out at 10^5.5 K "which is **exactly**
  what our threshold is" (`:88`) — yet `CIE_Tcutoff` is by construction *strictly greater* than 5.5,
  and a commented-out guard `if nonCIE_Tcutoff != CIE_Tcutoff:` (`:134`) presumes they may differ.
- **Naming description at `:116`–`117`** ("cutoff at which temperature *above* switches to CIE file
  (`nonCIE_Tcutoff`); cutoff at which temperature *below* switches to non-CIE file (`CIE_Tcutoff`)")
  describes a two-branch switch, contradicting the three-branch structure it sits above.
- **Continuity**: never claimed in words. It is only *implied* by F7 (a linear blend anchored at the
  two branch values at the two cutoffs). Nothing states that the two branches agree in magnitude at
  the seam, and nothing states that the two branches carry the same physics content — the non-CIE
  side is cooling **minus heating** (F3) while the CIE side is cooling **only** (F8, `:161`–`163`).
- **Validity limits stated**: temperature — non-CIE valid `[10**nonCIE_Tmin, 10^5.5]` K, CIE valid
  above (upper bound unstated); metallicity — solar (Z=0.014) and 0.15 solar (Z=0.002) for non-CIE,
  four discrete libraries for CIE; density and phi — **no limits stated at all**.
- **Time**: the cooling structure is only refreshed "every once in a while… E.g., lets say 5e4 years"
  (`:94`–`96`) / at "the COOLING_UPDATE_INTERVAL cadence" (`:33`–`39`), so the age used for the
  time-dependent table lags the true age by up to that interval — an explicitly accepted inaccuracy.

---

## 6. Contracts — inputs, outputs, state, caching, error handling

- `get_dudt(age, ndens, T, phi)` → `dudt` in code units (§2). "These value should not be logged!"
  (`:78`) — presumably meaning the *arguments* are linear, not log10, in contrast to `:148`.
- State keys: `params_dict['cStruc_heating_nonCIE'].value` (commented out, `:100`);
  `params['t_now']` in Myr (`read_cloudy.py:23`–`46`); `path_cooling_CIE` and `ZCloud` from the
  `.param` (`CIE/read_coolingcurve.py:26`–`54`).
- **Caching, three separate mechanisms:**
  1. `_cie_tcutoff` — "cached **by array id**", `id(logT_CIE)`; justification: "`logT_CIE` is built
     once at startup (main.py) and never replaced, so its id is stable for the whole run → no
     id-reuse hazard" (`:27`–`29`, `:49`).
  2. `_noncie_cutoffs` — "cached **on the cube object** (not by id) so it refreshes automatically
     when the cube is rebuilt at the COOLING_UPDATE_INTERVAL cadence — a fresh cube has no cached
     attr → recomputed" (`:33`–`39`).
  3. Cube files persisted to disk/array (`read_cloudy.py:169`, `:263`–`264`).
- **Performance-equivalence claims** (project rule 5 makes these directly testable):
  - HOTPATH F2.3: hoisting the max/min reductions is "**Bit-identical**: the SAME reduction over the
    SAME array" (`:22`–`26`, `:118`, `:33`–`39`).
  - HOTPATH F2.4: "`Lambda_CIE` is evaluated in the CIE branch below, not here, so it is not computed
    on the non-CIE / interpolation paths" (`:119`–`120`), restated as "the non-CIE and interpolation
    branches **never use it**" (`:161`–`162`).
- **Bounds checking / error handling:**
  - Lower-T gate: silent clamp to the table edge (`:124`–`129`) — no warning or log is mentioned.
  - A `raise` exists for "temperature lower than the available non-CIE curve" (`:199`).
  - The non-CIE *file-not-found* handler is **commented out** (`# try:` `:287`; `# except: # raise
    Exception("Opiate/CLOUDY file (non-CIE) for cooling curve not found…")` `:345`–`346`).
  - CIE: **no** extrapolation and no guard (`CIE/read_coolingcurve.py:56`–`57`).
  - NaN in the cubes: **no** handling, only a "Future TODO" (`read_cloudy.py:258`–`259`).
- `get_filename(age)` documented to return `filename : str` (singular, `read_cloudy.py:271`–`284`),
  but four comments say it may return two: "include brackets to check if there is one or two
  filenames" (`:321`, `:327`, `:332`), "return both" (`:342`).

---

## 7. Admissions (verbatim inventory)

| Marker | Text | Citation |
|---|---|---|
| TODO | "add for non-solar metallicity" | `CIE/read_coolingcurve.py:20` |
| TODO | "add file saving for quicker computation time" | `CIE/read_coolingcurve.py:23` |
| TODO | "add option to immediately get saved cubes" | `read_cloudy.py:65` |
| Future TODO | "If it fails, i.e., if it returns NaN because the values don't exist in the cooling table, we do further operations" | `read_cloudy.py:258`–`259` |
| "Might be a problem" | "this does not support extrapolation. If this happens, implement a function that does that" | `CIE/read_coolingcurve.py:56`–`57` |
| "perhaps non-physical" | NaNs in the cubes | `read_cloudy.py:143`–`167` |
| accuracy trade | "In order to improve speed, here we use dictionary. This means that the age will **not be as accurate**" | `net_coolingcurve.py:94`–`95` |
| "or better, … in the future?" | "if temperature is lower than the available non-CIE curve, error (or better, provide some interpolation in the future?)" | `net_coolingcurve.py:199` |
| "makes things easier" | rounding the log axis ticks | `read_cloudy.py:212` |
| "in reality this can easily be a one-liner" | cuboid-side helper kept only for readability | `read_cloudy.py:205`–`206` |
| "New idea" | the whole if/else switch design note | `net_coolingcurve.py:88` |
| "depreciated" ×4 | the `get_coolingStructure` call and both netcooling/interpolator constructions | `net_coolingcurve.py:108`, `:109`, `:143`–`146`, `:174`–`177` |
| "Right now, only …" | "only solar metallicity and rotation is considered" | `read_cloudy.py:286` |
| compat note | "`str.removesuffix` is Python 3.9+; use slice form so this also works on 3.8 anaconda installs **even though pyproject declares >=3.9**" | `read_cloudy.py:170`–`171` |

Typos worth noting because they can mask meaning: "craetes" (`net_coolingcurve.py:3`), "convension"
(`:155`), "depreciated" for *deprecated* (×4), "10e5.5" for 10^5.5 (`:114`), "tney" (`read_cloudy.py:80`),
"curent" (`:271`), "fil in" (`:247`), "interpolater" (`:135`), "i.e." used for "e.g." (`:82`, `:314`, `:352`).

---

## 8. Flags

**Prose contradicting prose**
- Non-CIE table units: erg cm³ s⁻¹ vs erg cm⁻³ s⁻¹ (S9-B-02).
- Age selection: "nearest available age" vs interpolate-between-neighbours (S9-B-22).
- Metallicity: "only solar … is considered" vs the 0.15-solar branch; "TODO: add for non-solar
  metallicity" vs a shipped [Fe/H]=−1 library (S9-B-15).
- `get_filename` returns one str vs one-or-two filenames (S9-B-23).
- F2.4 "the interpolation branch never uses Lambda_CIE" vs a CIE cooling rate computed inside the
  interpolation branch (S9-B-14).
- Two-branch description at `:116`–`117` vs the three-branch implementation (S9-B-12).
- "Interpolate between two **Lambda** values" vs interpolating **dudt** (S9-B-13).
- Stale TODO "add option to immediately get saved cubes" vs cube save/load present (S9-B-31).

**Stated unit inconsistent with a stated formula** — S9-B-02, S9-B-03 (a `.to(erg/cm³/s)` applied to
a quantity documented as erg·cm³/s is a dimensionally impossible astropy conversion unless a
squared-density multiplier has already been applied and left undocumented).

**Density factor documented two different ways** — worse than that: **documented zero times**
(S9-B-01), while the two branches' unit annotations imply *different* treatments (S9-B-02).

**Claimed table range conflicting with a claimed default** — non-CIE top at exactly 10^5.5 K (`:88`)
vs `CIE_Tcutoff = min(logT_CIE[logT_CIE > 5.5])` which is strictly above 5.5 (S9-B-10); "10e5.5"
vs "10^5.5" (S9-B-11); `nonCIE_Tmin` never given a number, only implied as 1e3 K by the word
"decade" (S9-B-29).

**Citation attached to two different tables** — CIE libraries 1 and 2 carry identical provenance text
(S9-B-20); no CLOUDY version anywhere.

**Claims too vague to check** — "slightly interpolated for values" (Gnat & Ferland); "round, because
it makes things easier" (no precision); "perhaps non-physical" NaNs; "metallicity … selects/validates
against the CIE library"; "every once in a while or so"; "Inert on every profiled regime".

---

```json
[
  {
    "id": "S9-B-01",
    "file": "trinity/cooling/net_coolingcurve.py",
    "line": 59,
    "class": "units",
    "severity": "S2",
    "claim": "The docstring fixes both ends of the cooling conversion — Lambda is 'erg*cm^3/s in cgs, i.e. Msun*pc^5/Myr^3 in code units' and dudt is 'Energy-density rate ... computed in cgs as erg/cm^3/s' — but NEVER states the density factor that bridges them. No comment or docstring in the whole slice writes n^2, n_e*n_H, n_tot^2, or n.",
    "evidence": "net_coolingcurve.py:59-76 (both unit statements); :163 '# erg/s * cm3'; :187 '#.to(u.erg / u.cm**3 / u.s)'; :154 and :179 '#* u.erg / u.cm**3 / u.s'. No line anywhere states the multiplier.",
    "expected": "The docstring should state the exact multiplier, e.g. 'dudt = -n^2 * Lambda with n the total hydrogen number density in cm^-3', so a reader can distinguish n_tot^2 from n_e*n_H (which differ by tens of percent in ionized gas and by orders of magnitude in partially neutral gas) and confirm it is applied identically on both branches.",
    "failure_scenario": "If the code multiplies by n rather than n^2, or mixes n_e*n_H on one branch with n_tot^2 on the other, the bubble energy-loss term is wrong by orders of magnitude across the density range and nothing in the documentation contradicts it. Dimensional consistency of the docstring alone cannot catch this.",
    "repro": "Read the multiplier applied to the interpolated non-CIE value and to Lambda_CIE in trinity/cooling/net_coolingcurve.py (non-CIE branch after :148, CIE branch after :163, interpolation branch :181-193) and confirm all three use the same, documented, squared-density factor with ndens already converted pc^-3 -> cm^-3.",
    "confidence": "high"
  },
  {
    "id": "S9-B-02",
    "file": "trinity/cooling/non_CIE/read_cloudy.py",
    "line": 23,
    "class": "units",
    "severity": "S1",
    "claim": "The non-CIE cube is documented as holding a cooling rate 'in units of [erg cm3 / s]' (a Lambda), but the value read out of that same cube in the non-CIE branch of get_dudt is annotated 'u.erg / u.cm**3 / u.s' (a volumetric rate), with no multiplier documented in between. The same quantity is given two different units two files apart.",
    "evidence": "read_cloudy.py:23-46 'Cooling rate is in units of [erg cm3 / s]' vs net_coolingcurve.py:154 '#* u.erg / u.cm**3 / u.s' and :179 (same annotation in the interpolation branch), both immediately after the RegularGridInterpolator call on cooling_nonCIE.datacube - heating_nonCIE.datacube (:144, :175).",
    "expected": "One documented unit for the CLOUDY cube. CLOUDY cooling/heating output columns are commonly volumetric (erg cm^-3 s^-1), in which case the non-CIE branch needs NO density factor while the CIE branch needs n^2 - an asymmetry that must be documented explicitly because it is invisible dimensionally once both are cast to erg/cm^3/s.",
    "failure_scenario": "If a squared-density factor is applied to the non-CIE cube when the cube is already volumetric (or omitted when it is a Lambda), the non-CIE branch is wrong by n^2 (up to ~1e8 over the plausible 1e-2..1e4 cm^-3 range), and the interpolation band at 10^5.5 K linearly blends two incommensurable quantities, producing a large artificial jump or a spurious smooth ramp exactly at the CIE/non-CIE seam.",
    "repro": "Compare the multiplier applied to the interpolator output in the non-CIE branch of net_coolingcurve.py against the multiplier applied to Lambda_CIE in the CIE branch; then check the header/columns of a bundled lib/default non-CIE opiate table to settle which unit the cube actually holds.",
    "confidence": "high"
  },
  {
    "id": "S9-B-03",
    "file": "trinity/cooling/net_coolingcurve.py",
    "line": 187,
    "class": "units",
    "severity": "S2",
    "claim": "The CIE side of the interpolation branch is annotated with an astropy conversion '.to(u.erg / u.cm**3 / u.s)' applied to a quantity documented three lines earlier as 'erg/s * cm3'. As written that conversion is dimensionally impossible (erg cm^3 s^-1 -> erg cm^-3 s^-1 differ by cm^6); it can only succeed if an undocumented squared-density factor is applied first.",
    "evidence": "net_coolingcurve.py:163 '# erg/s * cm3'; :185 '# # get CIE cooling rate'; :187 '#.to(u.erg / u.cm**3 / u.s)'.",
    "expected": "Either the annotation names the multiplier, or the comment is corrected. A commented-out .to() between incompatible units is a trace of a unit bug that was worked around rather than resolved.",
    "failure_scenario": "A future edit re-enabling that annotation raises astropy UnitConversionError; more importantly, the comment misleads a reader into believing no density factor is needed on the CIE path.",
    "repro": "Check whether the CIE branch multiplies Lambda_CIE by a density term before the erg/cm^3/s annotation at net_coolingcurve.py:163-187.",
    "confidence": "high"
  },
  {
    "id": "S9-B-04",
    "file": "trinity/cooling/net_coolingcurve.py",
    "line": 155,
    "class": "sign",
    "severity": "S2",
    "claim": "The negative-sign convention is documented in exactly one of three return paths: 'return in negative sign for convension (since the rate of change is negative due to net cooling)' sits in the non-CIE branch only. The CIE branch (:158-163) and the interpolation branch (:167-193) carry no sign statement.",
    "evidence": "net_coolingcurve.py:155 (non-CIE branch, after :137 'if temperature is lower than the non-CIE temperature'); no equivalent comment at :158-165 or :167-197.",
    "expected": "All three branches return dudt with the same sign convention (negative when the gas is net-cooling), and the convention is stated once at the function contract (:59-76) rather than in one branch.",
    "failure_scenario": "If the CIE branch omits the minus sign, gas above 10^5.5 K is heated rather than cooled by the cooling term - the bubble never loses energy in the hot interior, and the interpolation band blends a negative against a positive value, crossing zero somewhere inside 10^5.5 K < T < CIE_Tcutoff.",
    "repro": "Check the sign of the returned expression in each of the three branches of get_dudt, and that a positive net-cooling table value maps to a negative dudt in all three.",
    "confidence": "high"
  },
  {
    "id": "S9-B-05",
    "file": "trinity/cooling/net_coolingcurve.py",
    "line": 144,
    "class": "regime",
    "severity": "S3",
    "claim": "The two branches carry different physics content: non-CIE uses net = cooling - heating ('netcooling = cooling_nonCIE.datacube - heating_nonCIE.datacube'), while the CIE branch is documented as cooling only ('get CIE cooling rate', 'Lambda depends only on T'). No comment justifies dropping the heating term above the switch, and no continuity claim is made at the seam.",
    "evidence": "net_coolingcurve.py:144 and :175 (net = cooling - heating); :161-163 (CIE branch, cooling only); CIE/read_coolingcurve.py:19 'Lambda depends only on T'. Nothing in the slice claims the two agree at the cutoff.",
    "expected": "Either a documented statement that photoionization heating is negligible above 10^5.5 K, or a heating term on the CIE side. Also a stated (and tested) continuity condition at nonCIE_Tcutoff and CIE_Tcutoff.",
    "failure_scenario": "A step discontinuity in dudt at the switch temperature, sized by the dropped heating term. Since get_dudt is the RHS of the bubble-structure ODE (:22), a discontinuous RHS at a temperature the bubble profile crosses can stall or chatter the integrator.",
    "repro": "Evaluate get_dudt at T = 10^5.5 - eps and 10^5.5 + eps at fixed (ndens, phi) for a representative bubble state and compare; repeat at CIE_Tcutoff +/- eps.",
    "confidence": "medium"
  },
  {
    "id": "S9-B-06",
    "file": "trinity/cooling/net_coolingcurve.py",
    "line": 124,
    "class": "silent-failure",
    "severity": "S2",
    "claim": "Temperatures below the non-CIE table minimum are silently clamped up to nonCIE_Tmin: 'Gate: clamp a sub-table temperature up to the cooling file's minimum tabulated T (nonCIE_Tmin, log10 K) so it degrades to the table edge via the non-CIE branch below instead of falling through to the raise.' No warning, log, or counter is mentioned. Safety is argued from 'Inert on every profiled regime: the bubble ODE never sends T below the 3e4 boundary (see docs/dev/magic-numbers/)'.",
    "evidence": "net_coolingcurve.py:124-129. The cited justification lives in docs/dev/, which CLAUDE.md declares point-in-time and unverified.",
    "expected": "A clamp on a hot-loop physics input either warns once or is provably unreachable; 'inert on every profiled regime' is a statement about the regimes someone happened to profile, not an invariant.",
    "failure_scenario": "A configuration outside the profiled set (low feedback strength, high density, late momentum-driven phase, or a stiff transient) drives T below the table floor; the cooling rate is then evaluated at the wrong temperature with no diagnostic, and the run completes looking healthy.",
    "repro": "Instrument the gate with a counter and run param/simple_cluster.param plus the two f1edge configs named in CLAUDE.md; confirm the counter stays zero. Also confirm what nonCIE_Tmin actually is (the comment's '[10**nonCIE_Tmin, 1e4) decade' implies 1e3 K).",
    "confidence": "high"
  },
  {
    "id": "S9-B-07",
    "file": "trinity/cooling/net_coolingcurve.py",
    "line": 199,
    "class": "deadcode",
    "severity": "S3",
    "claim": "The final branch 'if temperature is lower than the available non-CIE curve, error' is documented as unreachable by the gate added above it, which exists precisely so a sub-table T 'degrades to the table edge via the non-CIE branch below instead of falling through to the raise'.",
    "evidence": "net_coolingcurve.py:199 (the raise) vs :124-127 (the gate that prevents reaching it).",
    "expected": "Either the gate is conditional (in which case the condition must be documented, since :124 says only 'a sub-table temperature'), or the raise is dead and should be flagged as such rather than left as apparent protection.",
    "failure_scenario": "A reader (or a future edit) trusts a guard that can never fire; if the gate is later narrowed, the error path has never been exercised.",
    "repro": "Check whether the clamp at :124-129 is unconditional for all T < 10**nonCIE_Tmin; if so the raise at :199 is unreachable.",
    "confidence": "medium"
  },
  {
    "id": "S9-B-08",
    "file": "trinity/cooling/non_CIE/read_cloudy.py",
    "line": 143,
    "class": "silent-failure",
    "severity": "S2",
    "claim": "Both cubes are documented to contain NaNs - 'Some are NaN, because they are not available in the cooling table (perhaps non-physical)' - and NaN handling is explicitly deferred: 'Future TODO: If it fails, i.e., if it returns NaN because the values don't exist in the cooling table, we do further operations.'",
    "evidence": "read_cloudy.py:143-167 (cool_cube/heat_cube NaN note); :258-259 (the deferred handling); :96 and :135 (interpolators built directly over these cubes).",
    "expected": "Either the NaN cells are filled/masked before the RegularGridInterpolator is constructed, or get_dudt checks its result and raises. A RegularGridInterpolator returns NaN for any query whose enclosing cell touches a NaN, so the NaN region is larger than the NaN cells themselves.",
    "failure_scenario": "dudt returns NaN, poisoning the bubble-structure ODE RHS; the integrator either fails opaquely far from the cause or (if a monotonic/finite guard swallows it) produces a silently truncated run.",
    "repro": "Count non-finite entries in the constructed cool_cube/heat_cube for a bundled table, then check whether get_dudt validates its return value for finiteness.",
    "confidence": "high"
  },
  {
    "id": "S9-B-09",
    "file": "trinity/cooling/net_coolingcurve.py",
    "line": 27,
    "class": "state",
    "severity": "S2",
    "claim": "Two sibling caches use contradictory invalidation strategies, each justified in prose. _cie_tcutoff is 'cached by array id' with the argument 'logT_CIE is built once at startup (main.py) and never replaced, so its id is stable for the whole run -> no id-reuse hazard'. _noncie_cutoffs explicitly rejects that approach: 'Cached on the cube object (not by id) so it refreshes automatically when the cube is rebuilt'.",
    "evidence": "net_coolingcurve.py:27-29 and :49 (id-keyed CIE cache) vs :33-39 (object-attribute non-CIE cache).",
    "expected": "The id-keying is only safe if logT_CIE is truly never rebuilt. Two documented mechanisms could rebuild it: the ZCloud == 0.15 auto-pin to a different CIE library (CIE/read_coolingcurve.py:26-54), and any per-run re-initialisation inside a single process. CLAUDE.md itself notes trinity leaks module-level global state in-process.",
    "failure_scenario": "If logT_CIE is ever replaced, the old array can be garbage-collected and a new array allocated at the same id - the cache then returns the previous grid's CIE_Tcutoff for the new curve, silently shifting the CIE/non-CIE switch temperature. Most likely to bite a sweep or any multi-run process.",
    "repro": "grep for every assignment to logT_CIE and for any code path that reloads the CIE curve (metallicity auto-pin, sweep worker re-init); then run two simulations with different path_cooling_CIE in one process and check the cached cutoff changes.",
    "confidence": "medium"
  },
  {
    "id": "S9-B-10",
    "file": "trinity/cooling/net_coolingcurve.py",
    "line": 49,
    "class": "numerical",
    "severity": "S3",
    "claim": "The CIE cutoff is defined as 'min(logT_CIE[logT_CIE > 5.5])' - a hard-coded 5.5 - while the non-CIE cutoff is table-derived (max of the cube's .temp). The prose simultaneously asserts the non-CIE table tops out at 10^5.5 K 'which is exactly what our threshold is'.",
    "evidence": "net_coolingcurve.py:49 (hard-coded 5.5); :33-39 (table-derived non-CIE cutoffs); :88 ('only up to 10^5.5K, which is exactly what our threshold is'); :134 (commented-out 'if nonCIE_Tcutoff != CIE_Tcutoff:' implying they can differ).",
    "expected": "Either both cutoffs are derived from their tables, or the 5.5 is documented as a magic constant tied to the bundled table set. Note also that min() over an empty boolean selection raises - if a CIE library has no tick above 5.5, this fails at startup.",
    "failure_scenario": "Swapping to a CIE library on a different temperature grid (libraries 1-4 come from four different sources) shifts or empties the selection; the interpolation band silently widens, narrows, or the code raises on an empty min().",
    "repro": "Print nonCIE_Tcutoff and CIE_Tcutoff for each of the four bundled CIE libraries and check the band width and that the selection is non-empty in every case.",
    "confidence": "high"
  },
  {
    "id": "S9-B-11",
    "file": "trinity/cooling/net_coolingcurve.py",
    "line": 114,
    "class": "other",
    "severity": "S4",
    "claim": "'we take the cutoff at 10e5.5 K' - read literally this is 10 x 10^5.5 = 10^6.5 K, an order of magnitude above the 10^5.5 K stated 26 lines earlier.",
    "evidence": "net_coolingcurve.py:114 ('10e5.5 K') vs :88 ('only up to 10^5.5K').",
    "expected": "'10^5.5 K'.",
    "failure_scenario": "A reader implementing against the comment picks the wrong decade for the switch.",
    "repro": "",
    "confidence": "high"
  },
  {
    "id": "S9-B-12",
    "file": "trinity/cooling/net_coolingcurve.py",
    "line": 116,
    "class": "other",
    "severity": "S4",
    "claim": "The cutoff definitions describe a two-branch switch - 'cutoff at which temperature above switches to CIE file (nonCIE_Tcutoff); cutoff at which temperature below switches to non-CIE file (CIE_Tcutoff)' - contradicting the three-branch structure documented immediately above and below, where temperatures between the two cutoffs are interpolated rather than sent to the CIE file.",
    "evidence": "net_coolingcurve.py:116-117 vs :91 ('If between max of nonCIE and min of CIE, take interpolation') and :167 ('if temperature is between, do interpolation').",
    "expected": "'above nonCIE_Tcutoff leaves the non-CIE table; at/above CIE_Tcutoff use CIE; between the two, interpolate'.",
    "failure_scenario": "",
    "repro": "",
    "confidence": "high"
  },
  {
    "id": "S9-B-13",
    "file": "trinity/cooling/net_coolingcurve.py",
    "line": 91,
    "class": "other",
    "severity": "S3",
    "claim": "The design note says the middle band interpolates 'between two Lambda values', but the branch's own debug print interpolates between dudt values: 'np.log10(T), [nonCIE_Tcutoff, CIE_Tcutoff], [dudt_nonCIE, dudt_CIE]' - i.e. linear in dudt against log10 T, not linear in Lambda.",
    "evidence": "net_coolingcurve.py:91 vs :193; supporting structure at :171-172 ('This part is just for non-CIE ... Get the maximum point of non-CIE') and :182-187 ('This part is just for CIE').",
    "expected": "One stated interpolation variable and one stated abscissa. Whether the blend is linear in Lambda or in dudt, and linear or logarithmic in the interpolated quantity, changes the result inside the band; the prose should say which, and the two statements should agree.",
    "failure_scenario": "Linear-in-dudt blending of two values that can differ by orders of magnitude produces a band whose shape is dominated by the larger value; a log-space blend would be materially different. Neither is documented as the intended physics.",
    "repro": "Read the interpolation call in the middle branch of get_dudt (net_coolingcurve.py:189-197) and record the abscissa and ordinate variables actually passed.",
    "confidence": "high"
  },
  {
    "id": "S9-B-14",
    "file": "trinity/cooling/net_coolingcurve.py",
    "line": 119,
    "class": "other",
    "severity": "S3",
    "claim": "The HOTPATH F2.4 optimisation claim states 'Lambda_CIE is evaluated in the CIE branch below, not here, so it is not computed on the non-CIE / interpolation paths', restated as 'the non-CIE and interpolation branches never use it'. But the interpolation branch contains its own CIE section ('This part is just for CIE', '# get CIE cooling rate') and produces dudt_CIE, so it must compute a CIE cooling rate.",
    "evidence": "net_coolingcurve.py:119-120 and :161-162 (the claim) vs :182-187 (a CIE cooling-rate section inside the interpolation branch) and :193 (dudt_CIE used there).",
    "expected": "The claim should be scoped to the hoisted variable ('the interpolation branch computes its own'), or the interpolation branch should reuse the CIE branch's value. As written, the stated performance invariant is either false or ambiguous.",
    "failure_scenario": "The claimed saving does not hold in the interpolation band; worse, if the two branches compute the CIE rate by different code paths they can diverge, breaking continuity at CIE_Tcutoff.",
    "repro": "Check whether get_Lambda is called once or twice in net_coolingcurve.py and whether the interpolation branch's CIE value is computed by the same expression as the CIE branch's.",
    "confidence": "medium"
  },
  {
    "id": "S9-B-15",
    "file": "trinity/cooling/CIE/read_coolingcurve.py",
    "line": 20,
    "class": "other",
    "severity": "S3",
    "claim": "'TODO: add for non-solar metallicity' sits six lines above a docstring that documents a metallicity parameter, a non-solar library (Sutherland & Dopita 1993 at [Fe/H] = -1), and an automatic metallicity-driven library pin. One of the two is stale.",
    "evidence": "CIE/read_coolingcurve.py:20 (the TODO) vs :26-54 ('metallicity : float Cloud metallicity (selects/validates against the CIE library)', library 4 '[Fe/H] = -1. Auto-pinned when ZCloud == 0.15 regardless of path_cooling_CIE').",
    "expected": "Remove the TODO if non-solar metallicity is supported, or state precisely what is still missing (e.g. only two discrete metallicities are available, arbitrary Z is not).",
    "failure_scenario": "A user assumes non-solar Z is unsupported and does not check which curve their run actually used - or assumes arbitrary Z is supported when only Z_solar and 0.15 Z_solar exist.",
    "repro": "",
    "confidence": "high"
  },
  {
    "id": "S9-B-16",
    "file": "trinity/cooling/CIE/read_coolingcurve.py",
    "line": 26,
    "class": "citation",
    "severity": "S3",
    "claim": "At the same ZCloud the CIE and non-CIE paths are documented to use tables of different metallicity. CIE auto-pins to Sutherland & Dopita 1993 at [Fe/H] = -1 (0.1 Z_solar) when ZCloud == 0.15, while the non-CIE path at '0.15 solar' uses Z = 0.002 against a stated solar Z = 0.014 (0.143 Z_solar).",
    "evidence": "CIE/read_coolingcurve.py:26-54 (library 4, '[Fe/H] = -1', 'Auto-pinned when ZCloud == 0.15') vs read_cloudy.py:295 ('solar, Z = 0.014') and :298 ('0.15 solar, Z = 0.002').",
    "expected": "Either a CIE table at 0.15 Z_solar, or a documented statement that the nearest available CIE metallicity is 0.1 Z_solar and that the two branches therefore differ by ~50% in metals.",
    "failure_scenario": "A low-metallicity run cools with 0.143 Z_solar below 10^5.5 K and 0.1 Z_solar above it, with a metallicity step exactly at the switch temperature - and the interpolation band blends the two.",
    "repro": "Run the same config at ZCloud = 0.15 and confirm which CIE file and which opiate file are loaded; compare their nominal metallicities.",
    "confidence": "high"
  },
  {
    "id": "S9-B-17",
    "file": "trinity/cooling/CIE/read_coolingcurve.py",
    "line": 26,
    "class": "numerical",
    "severity": "S3",
    "claim": "The library auto-pin is documented as triggering on exact equality: 'Auto-pinned when ZCloud == 0.15 regardless of path_cooling_CIE'.",
    "evidence": "CIE/read_coolingcurve.py:26-54.",
    "expected": "A tolerance-based or interval-based selection, or documentation that only the two exact literals 1.0 and 0.15 are supported and anything else falls back to a solar curve.",
    "failure_scenario": "A sweep that generates ZCloud values arithmetically (0.05*3, or a linspace endpoint) misses the equality by one ulp and silently gets the solar CIE curve while the non-CIE side may still select the Z=0.002 table - an undetected mixed-metallicity run. Sweep expansion from .param list syntax is exactly the path that produces such values.",
    "repro": "Set ZCloud to a value that is 0.15 up to floating-point error (e.g. 0.45/3) and check which CIE file is loaded.",
    "confidence": "medium"
  },
  {
    "id": "S9-B-18",
    "file": "trinity/cooling/CIE/read_coolingcurve.py",
    "line": 56,
    "class": "silent-failure",
    "severity": "S2",
    "claim": "The CIE lookup admits it has no out-of-range handling - 'Might be a problem here because this does not support extrapolation. If this happens, implement a function that does that' - and no CIE table temperature range is documented anywhere in the slice. Meanwhile the CIE branch is entered for every T >= CIE_Tcutoff with no documented upper clamp (contrast the explicit lower-T gate on the non-CIE side).",
    "evidence": "CIE/read_coolingcurve.py:56-57; net_coolingcurve.py:158 (CIE branch entered for all T above the cutoff); net_coolingcurve.py:124-129 (a gate exists for the non-CIE lower edge but no counterpart for the CIE upper edge).",
    "expected": "A documented CIE table temperature range plus a stated behaviour above the maximum tabulated T (raise, clamp, or extrapolate) - and consistency with how the non-CIE lower edge is handled.",
    "failure_scenario": "A hot early bubble interior above the CIE table maximum either raises out of the ODE RHS (run dies mid-integration) or returns NaN/garbage from the interpolator, depending on the interpolator's fill behaviour. The asymmetry (guarded low side, unguarded high side) means only one edge was ever exercised.",
    "repro": "Determine the max tabulated log T for each of the four bundled CIE libraries and call get_Lambda just above it; record whether it raises, returns NaN, or extrapolates.",
    "confidence": "high"
  },
  {
    "id": "S9-B-19",
    "file": "trinity/cooling/CIE/read_coolingcurve.py",
    "line": 26,
    "class": "exponent",
    "severity": "S2",
    "claim": "The CIE interpolant is documented as mapping 'log K -> log Lambda', while get_Lambda is documented to return 'Lambda [erg/s * cm3]' (linear). The only conversion step commented is on the input side: 'change temperature to log for interpolation'. No comment mentions exponentiating the output.",
    "evidence": "CIE/read_coolingcurve.py:26-54 ('cooling_CIE_interpolation : callable Interpolation function (log K -> log Lambda)'; 'Returns Lambda [erg/s * cm3]'); :59 ('change temperature to log for interpolation'); :61 ('find lambda').",
    "expected": "An explicit 10** on the interpolant's output, and a comment saying so, symmetric with the documented log10 on the input.",
    "failure_scenario": "If the 10** is missing, get_Lambda returns log10(Lambda) ~ -22 instead of Lambda ~ 1e-22 - cooling is then wrong by ~22 orders of magnitude and of the wrong sign structure, for every T above the CIE cutoff. If instead the interpolant is actually linear->linear, the docstring's 'log K -> log Lambda' misdocuments the contract for anyone constructing that callable.",
    "repro": "Check that get_Lambda applies 10** to the interpolant result, and cross-check one value against the tabulated file at a known log T.",
    "confidence": "high"
  },
  {
    "id": "S9-B-20",
    "file": "trinity/cooling/CIE/read_coolingcurve.py",
    "line": 26,
    "class": "citation",
    "severity": "S3",
    "claim": "CIE libraries 1 and 2 carry identical provenance text - both 'CLOUDY cooling curve for HII region, solar metallicity' - distinguished only by library 2's added grain-sublimation clause. No CLOUDY version is given for either, and no version or reference is given for the non-CIE 'opiate/CLOUDY' tables.",
    "evidence": "CIE/read_coolingcurve.py:26-54 (entries 1 and 2 verbatim identical up to the grain clause); read_cloudy.py:189 ('these are derived quantities in CLOUDY output'), :285 (filename convention), :346 ('Opiate/CLOUDY file (non-CIE)').",
    "expected": "A CLOUDY version (and ideally the input deck or a reference) per table, so a published result can be traced. Two tables cannot be distinguished by a description that is identical.",
    "failure_scenario": "A result cannot be reproduced or attributed; a user selecting path_cooling_CIE = 1 vs 2 cannot tell from the documentation what physically differs beyond grain sublimation.",
    "repro": "",
    "confidence": "high"
  },
  {
    "id": "S9-B-21",
    "file": "trinity/cooling/CIE/read_coolingcurve.py",
    "line": 26,
    "class": "citation",
    "severity": "S3",
    "claim": "CIE library 3 is attributed to 'Gnat and Ferland 2012 (slightly interpolated for values)' - an unquantified modification of published data attached to the citation.",
    "evidence": "CIE/read_coolingcurve.py:26-54.",
    "expected": "State what was interpolated (which axis, onto what grid, by what scheme), so the shipped table can be checked against the published one.",
    "failure_scenario": "The bundled table diverges from the cited paper in a way no one can audit; a result is attributed to Gnat & Ferland 2012 while using resampled values.",
    "repro": "Compare the bundled lib/default/CIE library-3 file against the published Gnat & Ferland 2012 tabulation at the paper's own grid points.",
    "confidence": "medium"
  },
  {
    "id": "S9-B-22",
    "file": "trinity/cooling/non_CIE/read_cloudy.py",
    "line": 61,
    "class": "regime",
    "severity": "S3",
    "claim": "Age handling is documented two ways in the same file: 'For given time (cluster age), find the nearest available age' versus 'If age is between files, we find the nearest higher age and lower age neighbour, and do interpolation'. Ages outside the tabulated set are silently clamped: 'If the given age is greater than the maximum or is lower than the minimum, then use the max/min instead.' Available ages are 1e6, 2e6, 3e6, 4e6, 5e6, 1e7 yr - note the 5-Myr-wide gap between the last two.",
    "evidence": "read_cloudy.py:61-62 (available ages, 'nearest available age'); :307-308 (clamping); :336-339 (two-neighbour interpolation); :67 and :71 (one file vs two).",
    "expected": "One documented rule (snap vs interpolate) and a documented, warned behaviour for ages outside [1e6, 1e7] yr.",
    "failure_scenario": "Runs younger than 1 Myr - i.e. the entire early energy-driven phase - silently use the 1 Myr cooling structure, and runs past 10 Myr silently use the 10 Myr one, with no diagnostic. Between 5 and 10 Myr a linear blend spans a factor-2 age range where the ionizing output changes rapidly.",
    "repro": "Log which cooling file(s) and weights are chosen at t_now = 0.3, 2.3, 7.0 and 15.0 Myr.",
    "confidence": "high"
  },
  {
    "id": "S9-B-23",
    "file": "trinity/cooling/non_CIE/read_cloudy.py",
    "line": 271,
    "class": "other",
    "severity": "S3",
    "claim": "get_filename's documented contract is a single string ('Returns ------- filename : str'), but four comments in its body say it may return two: 'include brackets to check if there is one or two filenames' (x3) and 'return both'.",
    "evidence": "read_cloudy.py:271-284 (docstring) vs :321, :327, :332, :342; caller-side at :67 ('if return only one file, no need interpolation. see get_filename()') and :71 ('if two files, then it means there is interpolation').",
    "expected": "The docstring should state the union return type (a list/tuple of one or two filenames) since the caller branches on its length.",
    "failure_scenario": "A caller written to the docstring treats the return as a string; len() of a one-element list vs len() of a filename string are both truthy but mean different things, so a length test can misfire silently.",
    "repro": "Check the return statements of get_filename and whether the single-file case returns a bare str or a one-element sequence.",
    "confidence": "high"
  },
  {
    "id": "S9-B-24",
    "file": "trinity/cooling/non_CIE/read_cloudy.py",
    "line": 345,
    "class": "silent-failure",
    "severity": "S3",
    "claim": "The only documented error handling for a missing non-CIE cooling table is commented out: '# try:' at :287 and '# except: # raise Exception(\"Opiate/CLOUDY file (non-CIE) for cooling curve not found. Make sure to double check parameters ...\")' at :345-346.",
    "evidence": "read_cloudy.py:287-288, :345-346.",
    "expected": "Either an active, actionable error at this trust boundary (file path from user .param), or a note explaining why the raw exception is preferred.",
    "failure_scenario": "A user with a wrong path_cooling / metallicity / rotation combination gets an unhandled IndexError or FileNotFoundError from deep inside the loader instead of the actionable message that was written for exactly this case.",
    "repro": "Point the non-CIE cooling path at a directory with no matching .dat and observe the error surfaced to the user.",
    "confidence": "high"
  },
  {
    "id": "S9-B-25",
    "file": "trinity/cooling/non_CIE/read_cloudy.py",
    "line": 210,
    "class": "numerical",
    "severity": "S3",
    "claim": "Cube axis ticks are produced by sort -> log10 ('log array, because the original was created in log space') -> round ('round, because it makes things easier'). No rounding precision is stated. The cube is then filled by looking up 'which index these belong to' for each data row.",
    "evidence": "read_cloudy.py:208-212 (sort/log/round), :230-236 (cooling cube fill: 'find which index these belong to'), :247-253 (heating cube fill).",
    "expected": "A stated rounding precision and a stated reason it is safe - specifically that the same rounding is applied to both the tick array and the per-row lookup key, otherwise a row can fail to match its tick.",
    "failure_scenario": "If the axis ticks are rounded but the row keys are not (or vice versa), rows land in the wrong cell or silently fail to land at all, leaving the pre-initialised value (NaN, per :143-167) in cells that do have table data - which then propagates through the interpolator. Also, rounded ticks are not the table's true coordinates, so every interpolation is against slightly displaced abscissae.",
    "repro": "Build the cube for a bundled table and assert every non-NaN row of the file maps to a filled cell, and that the tick arrays equal the file's own distinct log values to within the rounding precision.",
    "confidence": "medium"
  },
  {
    "id": "S9-B-26",
    "file": "trinity/cooling/non_CIE/read_cloudy.py",
    "line": 192,
    "class": "sign",
    "severity": "S3",
    "claim": "'make sure signs in heating/cooling column are positive!' - the loader is documented to force the sign of the heating and cooling columns, which the net computation then subtracts as net = cooling - heating.",
    "evidence": "read_cloudy.py:192; net_coolingcurve.py:144 and :175 ('netcooling = cooling_nonCIE.datacube - heating_nonCIE.datacube').",
    "expected": "Documentation of the sign convention in the CLOUDY output being corrected (which column is emitted negative and why), so that taking abs() is a stated normalisation rather than a blanket sign wipe.",
    "failure_scenario": "An unconditional abs() cannot distinguish a table-wide sign convention from a genuinely signed entry; if any column legitimately changes sign, that information is destroyed and net = cooling - heating gets the wrong magnitude with no trace.",
    "repro": "Check whether the sign fix is np.abs() on the whole column or a conditional negation, and inspect the raw sign distribution of both columns in a bundled table.",
    "confidence": "medium"
  },
  {
    "id": "S9-B-27",
    "file": "trinity/cooling/net_coolingcurve.py",
    "line": 59,
    "class": "deadcode",
    "severity": "S3",
    "claim": "get_dudt documents an 'age [Myr]' input parameter, but the call that consumed it is commented out as 'depreciated' ('# cooling_nonCIE, heating_nonCIE = non_CIE.get_coolingStructure(age)'), the structure is now read from a dictionary for speed, and get_coolingStructure independently reads the age from params['t_now'] internally.",
    "evidence": "net_coolingcurve.py:59-76 (documented parameter), :108-109 ('depreciated' call), :94-96 ('here we use dictionary'), :100 (dict access); read_cloudy.py:23-46 (\"the current age is read internally as params['t_now']\").",
    "expected": "If age is no longer used, remove it from the signature and docstring, or document that it is retained for compatibility. Two independent age sources for the same physics is a latent divergence.",
    "failure_scenario": "The caller's age and params['t_now'] drift apart (they are updated at different points in the loop) and the cooling structure corresponds to neither the documented nor the expected time, on top of the already-accepted staleness of up to one refresh interval.",
    "repro": "grep for uses of the age parameter inside get_dudt; if unused, confirm every cooling-structure consumer reads params['t_now'].",
    "confidence": "medium"
  },
  {
    "id": "S9-B-28",
    "file": "trinity/cooling/net_coolingcurve.py",
    "line": 94,
    "class": "other",
    "severity": "S3",
    "claim": "The cooling-structure refresh cadence is documented twice, imprecisely and under two names: 'the cooling structure only updates every once in a while or so. E.g., lets say 5e4 years according to run_energy_implicit_phase.py' and 'rebuilt at the COOLING_UPDATE_INTERVAL cadence'. The value is hedged ('lets say'), sourced from another module, and the accuracy cost is accepted without bound.",
    "evidence": "net_coolingcurve.py:94-96 and :33-39.",
    "expected": "One named constant with its value and units stated where it is used, plus a statement of the resulting maximum age error in the cooling table.",
    "failure_scenario": "The interval is changed in run_energy_implicit_phase.py and this comment silently becomes wrong; nobody can bound the age error introduced into the cooling rate.",
    "repro": "grep COOLING_UPDATE_INTERVAL and confirm its value is 5e4 yr and that run_energy_implicit_phase.py is where it is applied.",
    "confidence": "high"
  },
  {
    "id": "S9-B-29",
    "file": "trinity/cooling/non_CIE/read_cloudy.py",
    "line": 227,
    "class": "other",
    "severity": "S3",
    "claim": "The only quantitative statement about the non-CIE grid is its shape: 'size = (n_ndens, n_temp, n_phi), e.g. (33, 21, 22) for the bundled Z1.00 tables'. No axis ranges, no dex spacing, and no minimum/maximum for density, temperature or phi are documented anywhere in the slice. nonCIE_Tmin is referenced by name but never given a value - it is only implied to be 1e3 K by the phrase 'the whole valid [10**nonCIE_Tmin, 1e4) decade'.",
    "evidence": "read_cloudy.py:227 (shape); read_cloudy.py:143-167 (axes described only as 'ticks in log space'); net_coolingcurve.py:127-128 (the 'decade' phrasing that implies 1e3 K); net_coolingcurve.py:88 (upper bound 10^5.5 K, the only stated range endpoint).",
    "expected": "Documented axis ranges and resolutions for the bundled tables, so the clamp at net_coolingcurve.py:124-129, the '3e4 boundary' claim at :129, and the out-of-range behaviour in ndens and phi can all be checked against a stated grid.",
    "failure_scenario": "Out-of-range ndens or phi behaviour is entirely undocumented - unlike temperature, no gate is mentioned for either - so a dense shell or a bright cluster can leave the cube on an axis nobody guarded.",
    "repro": "Print min/max/len of log_ndens_arr, log_temp_arr, log_phi_arr for a bundled Z1.00 table and compare against the (33, 21, 22) claim; confirm 10**nonCIE_Tmin and the 10^5.5 K top.",
    "confidence": "high"
  },
  {
    "id": "S9-B-30",
    "file": "trinity/cooling/non_CIE/read_cloudy.py",
    "line": 120,
    "class": "units",
    "severity": "S4",
    "claim": "Six commented-out lines attach physical units to arrays that are explicitly logarithmic: 'cooling_data.ndens = log_ndens_arr / u.cm**3', '.temp = log_temp_arr * u.K', '.phi = log_phi_arr / u.cm**2 / u.s' (repeated for heating_data).",
    "evidence": "read_cloudy.py:120-122 and :128-130; the arrays are documented as log10 at :23-46 and :143-167.",
    "expected": "A logarithm of a dimensional quantity is dimensionless; these annotations are wrong even as commentary and would be wrong if re-enabled.",
    "failure_scenario": "Re-enabling them attaches units to log values that the interpolator then consumes as plain numbers, or silently changes downstream unit arithmetic.",
    "repro": "",
    "confidence": "high"
  },
  {
    "id": "S9-B-31",
    "file": "trinity/cooling/non_CIE/read_cloudy.py",
    "line": 65,
    "class": "deadcode",
    "severity": "S4",
    "claim": "'TODO: add option to immediately get saved cubes' contradicts the cube persistence that is documented as present in create_cubes: 'Does the cube already exist?' and 'Final step: save into an array to save time in the future.' The parallel CIE TODO 'add file saving for quicker computation time' may be in the same state.",
    "evidence": "read_cloudy.py:65 vs :169 and :263-264; CIE/read_coolingcurve.py:23.",
    "expected": "Remove satisfied TODOs, or state what remains (e.g. cubes are cached but the cache is not reused at the get_coolingStructure level).",
    "failure_scenario": "",
    "repro": "",
    "confidence": "medium"
  },
  {
    "id": "S9-B-32",
    "file": "trinity/cooling/non_CIE/read_cloudy.py",
    "line": 170,
    "class": "other",
    "severity": "S4",
    "claim": "A deliberate compatibility shim is documented against the project's declared floor: 'str.removesuffix is Python 3.9+; use slice form so this also works on 3.8 anaconda installs even though pyproject declares >=3.9.'",
    "evidence": "read_cloudy.py:170-171.",
    "expected": "Either lower the declared floor or drop the shim; an undeclared, untested 3.8 path is a support claim nothing verifies.",
    "failure_scenario": "The 3.8 path is exercised by users but never by CI; other 3.9+ syntax elsewhere breaks it anyway, so the shim buys nothing while implying support.",
    "repro": "",
    "confidence": "high"
  }
]
```
