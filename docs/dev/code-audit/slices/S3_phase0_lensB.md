# S3 phase0 init — Lens B (what the code claims)

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

Scope: prose only (comments + docstrings) from
`trinity/phase0_init/get_InitCloudProp.py` and `trinity/phase0_init/get_InitPhaseParam.py`.
I have not seen one line of implementation. Everything below is a **claim the prose makes**,
transcribed for a second lens to test against the code. Nothing here is a verdict on the code.

Line citations are the first line of the comment/docstring block as given in the prose extract.

---

## 1. Formulas — transcribed exactly as written

### `trinity/phase0_init/get_InitCloudProp.py`

| # | Line | Formula as written in prose | Notes on the transcription |
|---|------|------------------------------|-----------------------------|
| F1 | 3 (module docstring) | `densPL: n(r) = nCore * (r/rCore)^alpha` | power-law profile |
| F2 | 150 (`_init_powerlaw_cloud`) | `n(r) = nCore * (r/rCore)^alpha` — restated identically | `alpha = 0` homogeneous, `alpha = -1` "intermediate", `alpha = -2` "isothermal" |
| F3 | 186 | `nEdge = nCore * (rCloud / rCore)^α = nISM` | the equation being solved by "Option 1" |
| F4 | 187 | `rCore_min = rCloud * (nCore / nISM)^(1/α)` | exponent **1/α**, base **nCore/nISM** (not the reciprocal). Algebraically consistent with F3. |
| F5 | 217 | `nCore_min = nISM * (rCloud / rCore)^(-α)` | exponent **−α**. Algebraically consistent with F3. Prose calls it a "first-order estimate from current rCloud". |
| F6 | 304 (`_init_bonnor_ebert_cloud`) | `M(r)/M_cloud = m(xi)/m(xi_out)` | "analytical Lane-Emden mass formula"; claimed to give "EXACT results: M(rCloud) = mCloud guaranteed" |
| F7 | 486 (`verify_mass_at_rCloud`) | `rel_error = |M(rCloud) - mCloud| / mCloud` | |
| F8 | 442 | radius array extends "up to `1.5 * rCloud`" beyond the cloud | |
| F9 | 8 / 90 / 150 / 304 | `mu_convert = 1.4` used for mass density | value **1.4** quoted at four sites; see §3 for the dimensional problem |
| F10 | 513 | error threshold `> 1%` | in `verify_mass_at_rCloud` |

### `trinity/phase0_init/get_InitPhaseParam.py`

| # | Line | Formula as written in prose | Notes |
|---|------|------------------------------|-------|
| G1 | 26 | `E0 = (5/11) * Lw * dt` | coefficient **5/11**; time argument written **dt** |
| G2 | 166 | `E = (5/11) * L_w * t` | coefficient **5/11** (same); time argument written **t** (not `dt`) — same equation, two different time symbols |
| G3 | 31 | `T = 1.51e6 K * (L/10^36 erg/s)^(8/35) * (n/1 cm^-3)^(2/35) * t^(-6/35) * (1-xi)^0.4` | exponents **8/35**, **2/35**, **−6/35**, **0.4**; prefactor **1.51e6 K**; reference luminosity **10^36 erg/s** |
| G4 | 171 | `T = 1.51e6 * (L/10^36)^(8/35) * (n)^(2/35) * t^(-6/35) * (1-xi)^0.4` | identical numbers to G3; units dropped in the restatement |
| G5 | 129 | `Mdot = pdot^2 / (2 * L)` | derived in prose "From: `L = 0.5 * Mdot * v^2` and `pdot = Mdot * v`" |
| G6 | 128 | `L = 0.5 * Mdot * v^2` | |
| G7 | 128 | `pdot = Mdot * v` | |
| G8 | 133 | `v = 2 * L / pdot` | "wind-only quantities" |
| G9 | 150 | `dt = sqrt(3 * Mdot / (4 * pi * rho_a * v^3))` | free-streaming duration |
| G10 | 145 | `rho = n_H * mu_convert`, with `mu_convert (=1.4)` (also stated at line 76) | see §3 |
| G11 | 45 (`get_y0`) | `t0 = tSF + free-streaming duration` | |
| G12 | 45 (`get_y0`) | `r0 = terminal velocity * free-streaming duration` (= R2) | |
| G13 | 45 (`get_y0`) | `v0 = wind terminal velocity` | |

Internal algebra self-checks (prose vs prose, no code involved):
- F4 and F5 both follow correctly from F3.
- G5 follows correctly from G6+G7; G8 follows correctly from G6+G7.
- G9 is dimensionally consistent: `[M/T] / ([M/L^3][L^3/T^3]) = T^2`, sqrt → time. It is the
  algebraic solution of "swept ambient mass = ejected wind mass" (`(4/3)π(v t)^3 ρ_a = Mdot·t`),
  **but the prose never states that balance** — see §4.

---

## 2. Citations — verbatim, with what is attributed

| Line | Citation as written | Attributed to |
|------|---------------------|---------------|
| 27 (`get_InitPhaseParam.py`) | "From **Weaver+77, Eq. 20** - assumes adiabatic index gamma = 5/3" | the energy fraction `E0 = (5/11)*Lw*dt` (G1) |
| 166 | "From **Weaver+77, Eq. 20**: `E = (5/11) * L_w * t`" | same equation, restated (G2) |
| 30 | "Temperature coefficient in **Weaver+77, Eq. 37**" | the `1.51e6 K` prefactor |
| 170 | "From **Weaver+77, Eq. 37**" | the full temperature law (G4) |
| 45 (`get_y0` docstring) | "Bubble energy: **Weaver+77, Eq. 20**"; "Bubble temperature: **Weaver+77, Eq. 37**" | as above |
| 45 (`get_y0` docstring) | "Free-streaming phase duration: **Rahner thesis Eq. 1.15**, `https://www.imprs-hd.mpg.de/399417/thesis_Rahner.pdf` **pg 17**" | `dt` (G9) |
| 149 | "From **Rahner thesis Eq. 1.15**" | `dt` (G9), same |
| 3 / 8 (`get_InitCloudProp.py`) | "**Lane-Emden equation**", "Bonnor-Ebert sphere" | BE density/mass profile (F6) |

Citation hygiene observed at prose level:
- **No citation is attached to two different formulas.** Weaver Eq. 20 → energy only; Eq. 37 →
  temperature only; Rahner Eq. 1.15 → `dt` only. Consistent across both statement sites.
- Every coefficient appears identically at both of its two sites (5/11 twice; 1.51e6, 8/35, 2/35,
  −6/35, 0.4 twice each; 10^36 twice). **No coefficient is quoted two different ways.**
- The Lane–Emden / Bonnor–Ebert branch carries **no literature citation at all** — no Bonnor 1956,
  Ebert 1955, or Chandrasekhar reference, and no equation number for `M(r)/M_cloud = m(ξ)/m(ξ_out)`.
- `gamma_adia` is required by the BE branch (line 90) with no citation and no stated role.

---

## 3. Units and conventions claimed

| Line | Quantity | Claimed unit / convention |
|------|----------|---------------------------|
| 52 | `rCloud`, `rCore` | pc |
| 52 | `nEdge` | cm^-3 |
| 52 | `r_arr` | pc, "includes rCore and rCloud exactly" |
| 52 | `n_arr` | cm^-3 |
| 52 | `M_arr` | Msun |
| 52 | `T_eff` | K (BE sphere only) |
| 52 | `xi_out` | dimensionless (BE sphere only) |
| 90 | `mCloud` | Msun; `nCore`, `nISM` | cm^-3; `rCore` | pc |
| 90 | `mu_convert` | "mean molecular weight for mass (=1.4)" |
| 90 | `densBE_Omega` | "density contrast (rho_core/rho_edge)" — dimensionless |
| 90 | `gamma_adia` | "adiabatic index" — dimensionless |
| 165 | mu selection | "Use mu_convert, **NOT** mu_neu or mu_ion" (for mass density) |
| 418 / 437 | `_create_radius_array` args | pc; `n_inside`, `n_outside` are ints |
| 347 | `be_result.c_s` | **cm/s**; line 349 converts **cm/s → km/s** for the exposed `sigma` |
| 45 | `t0` | Myr |
| 45 | `r0` | pc |
| 45 | `v0` | pc/Myr |
| 45 | `E0` | "[au]" (astro units, presumably Msun·pc²/Myr²; the token is ambiguous with *astronomical unit*) |
| 45 | `T0` | K |
| 84 | `tSF` | Myr |
| 127 / 132 / 165 | Mdot, v, E | "AU units"; v annotated "[pc/Myr in AU units]" |
| 144 | ambient density `rho_a` | "AU units: Msun/pc^3" |
| 76 / 145 | `mu_convert` | "mass per H nucleus — for `rho = n_H * mu_convert`"; "nCore is hydrogen nuclei density n_H" |
| 32 / 34 | Weaver T prefactor / reference luminosity | K / erg/s |
| 31 | Weaver T inputs | `L` in erg/s (via `/10^36 erg/s`), `n` in cm^-3; **units of `t` not stated** |
| 185 | conventional (paper) output units | `log Q [1/s]`, `L [erg/s]`, `Mdot [Msun/yr]`, `E0 [erg]` |
| 187 | internal unit | `[1/Myr]` |
| 188 | internal unit | `[Msun*pc^2/Myr^3]` |

---

## 4. Ranges, regimes, assumptions, stated balances

- **line 27** — "assumes adiabatic index gamma = 5/3" for the 5/11 energy fraction.
- **line 179** — auto-correction fires "if nEdge < nISM", and this is claimed to be
  "**only possible for α ≠ 0**".
- **line 185 / 215** — two mutually exclusive correction strategies: Option 1 raises `rCore`,
  Option 2 raises `nCore` ("keeps original rCore").
- **line 191** — "recomputing rCloud may make rCore >= rCloud, so we must verify after
  recomputation" — stated failure mode of Option 1.
- **line 209** — "rCore ended up >= rCloud after recomputation — fall through" (to Option 2).
- **line 218** — monotonicity assumption: "Increasing nCore shrinks rCloud, which only helps
  (nEdge ↑)". No sign restriction on α is stated for this claim.
- **line 234** — "iteratively halve rCore until rCore < rCloud (recomputing each time)".
  No iteration cap, no convergence criterion, no floor on rCore stated.
- **line 442** — the radius array covers only `r ≤ 1.5 * rCloud`; line 439 covers "from small
  radius" (value unstated) to rCloud; line 447 adds a single "near-origin point for mass profile".
- **line 304 / 8** — BE branch: `M(rCloud) = mCloud` claimed **exact / guaranteed**.
  No corresponding exactness claim for the power-law branch (which instead gets a "forward mass
  consistency check", line 258).
- **line 107–110** — "**CRITICAL**: Use WIND-ONLY quantities for wind velocity calculation… the
  wind terminal velocity `v = 2L/pdot` is only physical when using wind-only L and pdot (not total
  which includes SNe)." Stated only for the *velocity*; not stated for E0 or T0.
- **line 144–145** — the free-streaming ambient density is built from **`nCore`** (the cloud *core*
  density). The implied assumption — that the bubble is still inside the flat core at `r0` — is
  never stated.
- **line 37–40** — three floors "to prevent division by zero": in the `Mdot` calculation, in the
  velocity calculation, and in the `dt_phase0` calculation. Which quantity each floor clamps is
  not stated in prose.
- **Unstated but implied balance**: `dt` (G9) is the free-expansion→snowplough transition,
  i.e. swept mass = ejected mass. Prose gives only the closed form, never the balance.
- **Unstated ranges**: no valid range for `bubble_xi_Tb` (the `(1-xi)^0.4` factor demands
  `xi < 1`); no stability range for `densBE_Omega` (BE spheres have a critical contrast);
  no sign restriction on `densPL_alpha`; no statement of whether `t` in G3/G4 is `dt` or `t0`.

---

## 5. Contracts — inputs, outputs, state, ordering

`get_InitCloudProp(params)` (line 90):
- **Required keys**: `dens_profile` ∈ {`densPL`, `densBE`}, `mCloud`, `nCore`, `nISM`,
  `mu_convert`, `rCore` ("user-specified"), `path2output` ("output directory for saving plots").
  For `densPL`: `densPL_alpha`. For `densBE`: `densBE_Omega`, `gamma_adia`.
- **Returns**: `CloudProperties` dataclass (`rCloud`, `rCore`, `nEdge`, `r_arr`, `n_arr`, `M_arr`,
  and for BE `T_eff`, `xi_out`).
- **Declared side effect** (docstring "Notes", line 90): updates `params` **in place** with
  `rCloud`, `rCore`, `nEdge`, `initial_cloud_r_arr`, `initial_cloud_n_arr`, `initial_cloud_m_arr`;
  for BE also `densBE_Teff`, `densBE_xi_out`, `densBE_f_rho_rhoc`, `densBE_f_m`.
- **Undeclared side effects visible in comments**: `nCore` can be increased (line 215);
  a `sigma` (velocity dispersion, km/s) is written (lines 347–349); "Store computed values back to
  params" (line 276) and "Store computed values in params" (line 338) are broader than the list.
- `_validate_params` (line 381): "Raises ValueError if required parameters are missing or invalid."
- `_ensure_be_params_exist` (line 459): BE params "should normally be created by `read_param.py`,
  but this provides a safety fallback for standalone usage" — a second, duplicate source of defaults.
- `verify_mass_at_rCloud` (line 486): returns relative error; `verify_key_radii_in_array`
  (line 522): returns bool "True if both radii are in array exactly".
- `_create_radius_array` (line 418): returns "sorted unique radius array including rCore and
  rCloud exactly".

`get_y0(params)` (line 45):
- **Required keys**: `tSF`, `sps_f` ("SPS interpolation functions"), `nCore`, `mu_convert`,
  `bubble_xi_Tb`.
- **Returns**: `t0` [Myr], `r0` [pc] ("Initial bubble outer radius R2"), `v0` [pc/Myr],
  `E0` [au], `T0` [K].
- **Side effect**: "One-time feedback summary at SF onset" logged in conventional units (line 185).
- **Ordering**: never stated. But `get_y0` consumes `nCore` and `mu_convert`, and
  `get_InitCloudProp` may *rewrite* `nCore` and `rCore`. An ordering requirement
  (cloud init strictly before phase-0 init) is implied and undocumented.

---

## 6. Admissions of debt (verbatim triggers)

- line 501 — "`should be` exact due to `_create_radius_array`" + line 508 "Fallback to
  interpolation" — the exactness guarantee is hedged, with a fallback path.
- line 459 — "`should normally` be created by `read_param.py`, but this provides a `safety
  fallback` for standalone usage."
- line 344 — "Ensure BE-specific params exist (`may not be` in `read_param.py`)."
- line 216 — "`First-order estimate` from current rCloud" — acknowledged non-self-consistency
  (rCloud itself depends on nCore).
- line 251 — "`Final safety check` — `warn` if `still not satisfied` after correction" — the
  constraint may be violated on exit, with only a warning.
- line 124 — section header "COMPUTE WIND PROPERTIES (WIND-ONLY - **BUG FIX**)" — historical
  admission that total (wind+SNe) quantities were previously used here.
- line 168 — "Compute rCloud from physics (`not hardcoded!`)" — reads as a fix marker.
- line 191 — "we `must verify` after recomputation" — acknowledged fragility of Option 1.
- line 209 — "fall through" — acknowledged failure path of Option 1.
- line 320 — "Solve Lane-Emden equation (`can be cached` for efficiency)" — acknowledged unrealised
  optimisation.
- line 546–661 — an in-module `# Test / Example usage` `__main__` block with "Test 1/2/3",
  "Mock value class for testing", "Check all tests passed" — a parallel test harness outside pytest.
- No `TODO`/`FIXME`/`XXX`/`hack`/`temporary` tokens appear anywhere in this slice's prose.

---

## 7. Flags (claims a second lens must test against the code)

Grouped by the categories the brief asks for.

**Prose contradicting other prose**
- `rCore` is documented as an input "(user-specified)" (line 90) yet is raised (185), halved (234),
  and stored back (276) — and it *is* in the in-place-update list, so the docstring contradicts
  itself within one block. → S3-B-03
- `nCore` is documented only as an input (line 90) but is raised by Option 2 (215) and is **absent**
  from the docstring's in-place-update list. → S3-B-02
- "EXACT results: M(rCloud) = mCloud guaranteed" (304) vs a verifier with a 1% tolerance and an
  interpolation fallback (486/508/513). → S3-B-09
- `E0 = (5/11)*Lw***dt**` (26) vs `E = (5/11)*L_w***t**` (166) — same equation, different time
  symbol, and `t0 = tSF + dt` also exists in scope. → S3-B-14

**Same quantity described two different ways / stated unit inconsistent with stated formula**
- `mu_convert` = "mean molecular weight for mass (=1.4)" (90) vs "mass per H nucleus — for
  `rho = n_H * mu_convert`" (76). As written, `rho = n_H * mu_convert` with `mu_convert = 1.4` is
  dimensionally impossible unless `m_H` is folded in somewhere unstated. → S3-B-01
- `T_eff` labelled "Effective temperature [K]" (52) but line 347 says it is a recast of the support
  velocity dispersion `sigma = c_s`; the mean molecular weight used in that σ↔T conversion is never
  stated, and line 165 forbids `mu_neu`/`mu_ion` only for *mass density*. → S3-B-12
- `[Msun*pc^2/Myr^3]` (188) is a **power**, not an energy; energy in these units is
  `Msun*pc^2/Myr^2`. Correct only if it annotates `L`, wrong if it annotates `E0`. → S3-B-22
- Weaver Eq. 37 demands `n` in cm^-3 (31) while the adjacent block converts density to Msun/pc^3
  (144); which one is fed to `T0` is unstated. → S3-B-16

**Claims too vague to check as written**
- `rCloud` is the central computed quantity ("Self-consistent rCloud computation from fundamental
  inputs", line 3) and **no formula for it appears anywhere in the prose**. → S3-B-27
- Which luminosity (wind-only vs wind+SNe) enters `E0` and `T0`: the WIND-ONLY mandate (107) is
  scoped to the velocity; line 166 writes `L_w`, lines 31/171 write bare `L`. → S3-B-19
- Which quantity each of the three "prevent division by zero" floors (37–40) actually clamps.
  → S3-B-20
- The units of `t` in the Weaver Eq. 37 transcription (31/171) — Weaver's form uses time in
  units of 10^6 yr; the prose omits it. → S3-B-15

**Regime / range claims with no stated guard**
- "nEdge < nISM (only possible for α ≠ 0)" (179): for α = 0 the profile is homogeneous so
  nEdge = nCore, and nCore < nISM makes the condition reachable. → S3-B-06
- Both correction formulas (187, 217) are singular / degenerate as α → 0 and change character for
  α > 0; the only stated guard is the α ≠ 0 remark. → S3-B-05
- "Increasing nCore shrinks rCloud, which only helps (nEdge ↑)" (218) holds for α < 0 but is not
  obviously monotone for α > 0. → S3-B-28
- `bubble_xi_Tb` range unstated; `(1-xi)^0.4` needs xi < 1. → S3-B-21
- `densBE_Omega` has no stated stability/validity ceiling. → S3-B-29
- `nCore` used as the free-streaming ambient density (144) with no stated `r0 ≤ rCore` condition.
  → S3-B-18
- Mixed solution branches: `r0 = v_w·dt` and `v0 = v_w` (free-streaming) combined with
  `E0 = (5/11)·L·dt` (Weaver energy-driven self-similar, which implies `v = (3/5)·r/t`), all at the
  same instant, with no reconciliation stated. → S3-B-17
- Radius array truncated at `1.5*rCloud` (442). → S3-B-23

**Silent-failure / numerical claims**
- Final safety check only *warns* (251) — execution continues with nEdge < nISM. → S3-B-08
- Corrections target the boundary `nEdge = nISM` exactly (185/217), so round-off can leave the
  constraint marginally violated. → S3-B-08 (same finding)
- Unbounded `rCore` halving loop (234). → S3-B-07
- Exact floating-point membership test for `rCloud` in `r_arr` (501/504), hedged by "should be".
  → S3-B-10
- `_ensure_be_params_exist` (459) duplicates schema defaults and can drift from `read_param.py`.
  → S3-B-26

**Undocumented state**
- `sigma` (km/s) written into params (347–349) but not in the docstring's update list, and its key
  name never appears in prose. → S3-B-11
- Implied but unstated ordering contract between `get_InitCloudProp` and `get_y0` via mutated
  `nCore`/`rCore`. → S3-B-04

**Other**
- `gamma_adia` required for a Bonnor–Ebert (isothermal Lane–Emden) sphere, role unexplained,
  uncited. → S3-B-13
- "BUG FIX" marker (124). → S3-B-24
- `__main__` self-test block (546–661) parallel to the pytest suite. → S3-B-25

---

```json
[
  {
    "id": "S3-B-01",
    "file": "trinity/phase0_init/get_InitPhaseParam.py",
    "line": 76,
    "class": "units",
    "severity": "S2",
    "claim": "mu_convert is described two incompatible ways: 'mass per H nucleus - for rho = n_H * mu_convert' (get_InitPhaseParam.py:76, repeated :145 as 'use mu_convert (=1.4) for mass density') versus 'mean molecular weight for mass (=1.4)' (get_InitCloudProp.py:90, echoed :150 and :304).",
    "evidence": "get_InitPhaseParam.py:76 '# mass per H nucleus - for rho = n_H * mu_convert'; :145 'nCore is hydrogen nuclei density n_H; use mu_convert (=1.4) for mass density'; get_InitCloudProp.py:90 '- mu_convert: mean molecular weight for mass (=1.4)'.",
    "expected": "One convention, stated once. If mu_convert is dimensionless (1.4), then rho = n_H * mu_convert * m_H and the :76 formula is missing m_H. If rho = n_H * mu_convert literally, mu_convert must carry mass units and cannot equal 1.4 in Msun/pc^3-based AU units (1.4 m_H ~ 1.2e-57 Msun).",
    "failure_scenario": "If either call site takes the prose at face value, the ambient mass density rho_a is wrong by a factor of m_H (~1e57 in AU units) or by the factor 1.4, which propagates into dt = sqrt(3*Mdot/(4*pi*rho_a*v^3)), hence into t0, r0 and E0.",
    "repro": "Compare the numeric mu_convert value with the expression that builds rho_a in get_InitPhaseParam.py near line 145, and with the mass-density construction in get_InitCloudProp.py's power-law and BE branches; check the same factor is used in both files.",
    "confidence": "high"
  },
  {
    "id": "S3-B-02",
    "file": "trinity/phase0_init/get_InitCloudProp.py",
    "line": 90,
    "class": "state",
    "severity": "S2",
    "claim": "The docstring's exhaustive 'updates params in-place with' list names rCloud, rCore, nEdge, the three initial_cloud_* arrays and the BE keys, but NOT nCore - yet the auto-correction explicitly raises nCore ('Option 2: increase nCore instead', :215) and values are 'stored back to params' (:276).",
    "evidence": "get_InitCloudProp.py:90 Notes block; :215 '# Option 2: increase nCore instead (keeps original rCore).'; :217 '# nCore_min = nISM * (rCloud / rCore)^(-alpha)'; :276 '# Store computed values back to params'.",
    "expected": "Either nCore is never mutated, or the docstring lists it as an in-place output alongside rCore.",
    "failure_scenario": "A caller that read the docstring assumes params['nCore'] is still its own input, and later stages (get_y0's ambient density, any output metadata, any sweep bookkeeping) silently run on a value the user never set.",
    "repro": "Run a densPL config with nEdge < nISM (small |alpha| or low nCore) and compare params['nCore'] before and after get_InitCloudProp.",
    "confidence": "high"
  },
  {
    "id": "S3-B-03",
    "file": "trinity/phase0_init/get_InitCloudProp.py",
    "line": 90,
    "class": "state",
    "severity": "S2",
    "claim": "rCore is documented as an input, '- rCore: core radius [pc] (user-specified)', while the same function raises it to rCore_min (:185), may iteratively halve it (:234) and stores it back (:276). The docstring is internally contradictory: the same key is labelled user-specified and listed as an in-place update.",
    "evidence": "get_InitCloudProp.py:90 required-keys list and Notes list; :185 '# Option 1: increase rCore to the minimum that gives nEdge = nISM'; :234 '# If nCore increase shrank rCloud below rCore, iteratively halve rCore until rCore < rCloud'.",
    "expected": "The prose should state that rCore is an input that MAY be overwritten by the nEdge>=nISM correction, and the correction should be logged loudly.",
    "failure_scenario": "A parameter sweep over rCore silently collapses: several rCore inputs get corrected to the same value, and the recorded output rCore no longer matches the .param file the run is labelled with.",
    "repro": "Sweep rCore across a range where nEdge < nISM is triggered; diff the input rCore against the rCore written to output metadata.",
    "confidence": "high"
  },
  {
    "id": "S3-B-04",
    "file": "trinity/phase0_init/get_InitPhaseParam.py",
    "line": 45,
    "class": "state",
    "severity": "S2",
    "claim": "get_y0 declares it 'Must contain: tSF, sps_f, nCore, mu_convert, bubble_xi_Tb' but states no ordering requirement relative to get_InitCloudProp, which can rewrite nCore (and rCore) in place.",
    "evidence": "get_InitPhaseParam.py:45 Parameters block; :144-145 ambient density built from nCore; get_InitCloudProp.py:215 raises nCore, :276 stores back.",
    "expected": "An explicit 'must be called after get_InitCloudProp' contract, or get_y0 taking the corrected density from the cloud-properties object rather than from a param key that another function mutates.",
    "failure_scenario": "If get_y0 runs before the cloud correction, dt = sqrt(3*Mdot/(4*pi*rho_a*v^3)) uses the uncorrected nCore; t0, r0 and E0 are then inconsistent with the cloud the bubble actually expands into.",
    "repro": "Instrument the call order in run.py / the driver for a config where the nCore correction fires, and check which nCore value reaches get_y0.",
    "confidence": "medium"
  },
  {
    "id": "S3-B-05",
    "file": "trinity/phase0_init/get_InitCloudProp.py",
    "line": 187,
    "class": "divergence",
    "severity": "S3",
    "claim": "Both correction formulas are singular or degenerate as alpha -> 0: 'rCore_min = rCloud * (nCore / nISM)^(1/alpha)' (:187) has 1/alpha in the exponent, and 'nCore_min = nISM * (rCloud / rCore)^(-alpha)' (:217) collapses to nCore_min = nISM. The only stated guard is the parenthetical 'only possible for alpha != 0' (:179), which is a claim about when the condition arises, not a numerical guard against small |alpha|.",
    "evidence": "get_InitCloudProp.py:179 '# ---- Auto-correct if nEdge < nISM (only possible for alpha != 0) ----'; :187; :217.",
    "expected": "An explicit |alpha| > eps guard (or a branch for alpha == 0) before evaluating either correction.",
    "failure_scenario": "alpha = -1e-6 (a nearly-homogeneous cloud, reachable in a sweep) makes (nCore/nISM)^(1/alpha) underflow to 0 or overflow to inf, producing rCore_min = 0 or inf and a downstream divide-by-zero or a nonsensical cloud.",
    "repro": "Call the power-law branch with alpha = 1e-8 and nEdge < nISM and inspect rCore_min / nCore_min.",
    "confidence": "medium"
  },
  {
    "id": "S3-B-06",
    "file": "trinity/phase0_init/get_InitCloudProp.py",
    "line": 179,
    "class": "regime",
    "severity": "S3",
    "claim": "The comment asserts nEdge < nISM is 'only possible for alpha != 0'. For alpha = 0 the profile is n(r) = nCore everywhere (:150 'alpha = 0: homogeneous'), so nEdge = nCore and the condition is reachable whenever nCore < nISM.",
    "evidence": "get_InitCloudProp.py:179; :150 'alpha = 0: homogeneous (constant density)'; :3 'n(r) = nCore * (r/rCore)^alpha'.",
    "expected": "Either the claim is corrected, or the alpha = 0 case is explicitly validated (nCore >= nISM) rather than assumed impossible.",
    "failure_scenario": "A homogeneous config with nCore < nISM skips the correction entirely (because the block is gated on alpha != 0) and builds a cloud less dense than its own ambient medium, with no warning.",
    "repro": "densPL with densPL_alpha = 0, nCore below nISM; check whether any warning fires and what nEdge/nISM comparison the code performs.",
    "confidence": "medium"
  },
  {
    "id": "S3-B-07",
    "file": "trinity/phase0_init/get_InitCloudProp.py",
    "line": 234,
    "class": "numerical",
    "severity": "S3",
    "claim": "'If nCore increase shrank rCloud below rCore, iteratively halve rCore until rCore < rCloud (recomputing each time)' - an unbounded loop with no stated iteration cap, no floor on rCore and no stated proof that rCloud does not shrink at least as fast as rCore.",
    "evidence": "get_InitCloudProp.py:234-235.",
    "expected": "A maximum iteration count and an explicit failure when the loop does not converge.",
    "failure_scenario": "If rCloud is recomputed from a mass integral that also decreases as rCore shrinks, the loop can run until rCore underflows to 0, then hang or emit a zero/NaN core radius that poisons n(r) = nCore*(r/rCore)^alpha.",
    "repro": "Construct a densPL config that enters Option 2 and lands with rCloud < rCore; count loop iterations and the final rCore.",
    "confidence": "medium"
  },
  {
    "id": "S3-B-08",
    "file": "trinity/phase0_init/get_InitCloudProp.py",
    "line": 251,
    "class": "silent-failure",
    "severity": "S3",
    "claim": "'Final safety check - warn if still not satisfied after correction': the nEdge >= nISM constraint may be violated on exit with only a warning. This is compounded by both corrections targeting the boundary exactly ('the minimum that gives nEdge = nISM', :185; nCore_min solving nEdge = nISM, :217), so floating-point round-off can leave nEdge marginally below nISM even on the success path.",
    "evidence": "get_InitCloudProp.py:251 '# Final safety check - warn if still not satisfied after correction'; :185; :217; :216 'First-order estimate'.",
    "expected": "Either target nEdge = nISM*(1+eps) with a small margin, or raise rather than warn when the constraint is still violated after both corrections.",
    "failure_scenario": "A run proceeds with a cloud edge less dense than the ISM; the warning scrolls past in a sweep and the physically invalid run is written to outputs indistinguishably from valid ones.",
    "repro": "Trigger the correction and assert nEdge >= nISM on exit for a grid of alpha, nCore, rCore; log how often the final check warns.",
    "confidence": "medium"
  },
  {
    "id": "S3-B-09",
    "file": "trinity/phase0_init/get_InitCloudProp.py",
    "line": 304,
    "class": "numerical",
    "severity": "S3",
    "claim": "The BE branch claims 'This gives EXACT results: M(rCloud) = mCloud guaranteed' via M(r)/M_cloud = m(xi)/m(xi_out), but verify_mass_at_rCloud exists with a '> 1% error' threshold (:513) and an interpolation fallback (:508) - an unconditional guarantee alongside machinery that assumes it can fail by percent-level amounts.",
    "evidence": "get_InitCloudProp.py:304-311; :8 '(exact M(rCloud) = mCloud)'; :486 rel_error definition; :508 '# Fallback to interpolation'; :513 '# > 1% error'.",
    "expected": "For the BE branch the verified rel_error should be at machine precision (~1e-12), not merely under 1%. A 1% tolerance is the right bar only for the power-law branch, whose docstring makes no exactness claim.",
    "failure_scenario": "A percent-level mass error in the BE branch passes the check silently, so the cloud carries a different mass than mCloud while the docstring promises exactness; all derived quantities (rCloud, nEdge, the free-streaming ambient density) inherit the error.",
    "repro": "Run the BE branch and print verify_mass_at_rCloud's rel_error; assert < 1e-10 rather than < 0.01.",
    "confidence": "medium"
  },
  {
    "id": "S3-B-10",
    "file": "trinity/phase0_init/get_InitCloudProp.py",
    "line": 501,
    "class": "numerical",
    "severity": "S3",
    "claim": "'Find index of rCloud in array (should be exact due to _create_radius_array)' plus '# Check if rCloud is exactly in array' (:504) - an exact floating-point equality lookup, hedged by 'should be', with an interpolation fallback (:508). verify_key_radii_in_array (:522) likewise returns 'True if both radii are in array exactly'.",
    "evidence": "get_InitCloudProp.py:501; :504; :508; :418 '_create_radius_array ... including rCore and rCloud exactly'; :452 '# Add key radii exactly and ensure unique sorted array'.",
    "expected": "Exact equality is only safe if the identical float object is inserted and never re-derived (np.unique/sort preserve values; any arithmetic on the array does not). If the array is rebuilt, rescaled or unit-converted after insertion, the equality test fails and the silent interpolation path takes over.",
    "failure_scenario": "A later unit conversion or resampling of r_arr breaks the exact match; the verifier reports failure or the mass check silently switches to interpolation, masking a genuine mass-profile error.",
    "repro": "Assert np.any(r_arr == rCloud) and np.any(r_arr == rCore) after get_InitCloudProp for both profiles; also check whether the interpolation fallback is ever taken.",
    "confidence": "medium"
  },
  {
    "id": "S3-B-11",
    "file": "trinity/phase0_init/get_InitCloudProp.py",
    "line": 347,
    "class": "state",
    "severity": "S3",
    "claim": "A support velocity dispersion sigma = c_s is written into params ('Expose the support velocity dispersion sigma = c_s ... be_result.c_s is cm/s', :347-348) and converted cm/s -> km/s (:349), but this key appears nowhere in the docstring's list of in-place updates (:90) and its name is never given in prose.",
    "evidence": "get_InitCloudProp.py:347-349; :90 Notes list (rCloud, rCore, nEdge, initial_cloud_*, densBE_Teff, densBE_xi_out, densBE_f_rho_rhoc, densBE_f_m).",
    "expected": "The key is documented in the Notes list with its unit (km/s), since a downstream consumer reading it needs to know the unit is km/s and not cm/s or AU units (pc/Myr).",
    "failure_scenario": "A downstream module reads sigma expecting AU units (pc/Myr) or cgs (cm/s); 1 km/s = 1.022 pc/Myr = 1e5 cm/s, so the error is either ~2% (silently plausible) or 1e5x (obvious).",
    "repro": "Diff params keys before/after get_InitCloudProp on a densBE config; grep the codebase for readers of that key and check their assumed unit.",
    "confidence": "high"
  },
  {
    "id": "S3-B-12",
    "file": "trinity/phase0_init/get_InitCloudProp.py",
    "line": 347,
    "class": "units",
    "severity": "S3",
    "claim": "T_eff is documented as 'Effective temperature [K] (BE sphere only)' (:52) and stored as densBE_Teff, but :347 states sigma = c_s is 'the transparent physical quantity behind the effective densBE_Teff' - i.e. T_eff encodes a support velocity dispersion, not a gas temperature. The mean molecular weight used in the sigma <-> T conversion is never stated, and :165 forbids mu_neu/mu_ion only for mass density.",
    "evidence": "get_InitCloudProp.py:52; :347-348; :165 '# Use mu_convert, NOT mu_neu or mu_ion'.",
    "expected": "c_s^2 = k_B*T_eff/(mu_particle*m_H) requires the per-particle mean molecular weight (~2.3 molecular, ~0.6 ionised), NOT the per-H-nucleus mass factor 1.4. Prose should say which mu the BE solver uses.",
    "failure_scenario": "Using mu_convert = 1.4 in the sound-speed relation misstates c_s by sqrt(2.3/1.4) ~ 1.28, which propagates into rCloud for the BE sphere (BE radius scales with c_s) and into the reported densBE_Teff.",
    "repro": "For a densBE config, check numerically whether densBE_Teff and the exposed sigma satisfy k_B*T/(mu*m_H) = sigma^2 for mu = 1.4, 2.3 and 0.6, and see which one holds.",
    "confidence": "medium"
  },
  {
    "id": "S3-B-13",
    "file": "trinity/phase0_init/get_InitCloudProp.py",
    "line": 90,
    "class": "regime",
    "severity": "S3",
    "claim": "gamma_adia ('adiabatic index') is listed as a REQUIRED key for the densBE branch, but a Bonnor-Ebert sphere is by construction the ISOTHERMAL Lane-Emden solution (:3 'Bonnor-Ebert sphere from Lane-Emden equation'; :320 'Solve Lane-Emden equation'). The prose never states where gamma_adia enters and gives no citation for the BE formulation at all.",
    "evidence": "get_InitCloudProp.py:90 'For densBE: - densBE_Omega: density contrast (rho_core/rho_edge) - gamma_adia: adiabatic index'; :3; :304; :320.",
    "expected": "Either gamma_adia is genuinely used (e.g. in a pressure or sound-speed relation, which should be documented and cited) or it is a vestigial required key that will raise on configs that legitimately omit it.",
    "failure_scenario": "A densBE .param that omits gamma_adia fails validation for a parameter the solver never reads; or gamma_adia silently perturbs an isothermal solution, making the BE sphere non-standard and uncomparable to published BE results.",
    "repro": "Vary gamma_adia (5/3 vs 1.0) on an otherwise identical densBE config and check whether rCloud, T_eff, xi_out change at all.",
    "confidence": "medium"
  },
  {
    "id": "S3-B-14",
    "file": "trinity/phase0_init/get_InitPhaseParam.py",
    "line": 26,
    "class": "regime",
    "severity": "S2",
    "claim": "The Weaver+77 Eq. 20 energy is written twice with different time arguments: 'E0 = (5/11) * Lw * dt' (:26) and 'E = (5/11) * L_w * t' (:166). Both cite Weaver+77 Eq. 20. In this scope dt (free-streaming duration) and t0 = tSF + dt (:159, :45) are both defined and generally differ.",
    "evidence": "get_InitPhaseParam.py:26-27; :166; :159 '# Start time for Weaver phase [Myr]'; :45 't0 : float [Myr] Start time for Weaver phase (= tSF + free-streaming duration)'.",
    "expected": "Weaver's t is time since bubble formation, i.e. dt here, not t0. If the code multiplies by t0 the initial energy is inflated by the factor (tSF+dt)/dt, which is large whenever tSF > 0.",
    "failure_scenario": "With tSF = 1 Myr and dt ~ 1e-3 Myr, using t0 instead of dt overstates E0 by ~1000x, giving a bubble that starts vastly over-energised and never matches the Weaver similarity solution.",
    "repro": "Run two configs identical except tSF (0 vs 1 Myr) and compare E0; under the correct reading E0 should be essentially unchanged (dt depends on Mdot, v, rho only), under the wrong reading it scales with tSF.",
    "confidence": "medium"
  },
  {
    "id": "S3-B-15",
    "file": "trinity/phase0_init/get_InitPhaseParam.py",
    "line": 31,
    "class": "units",
    "severity": "S2",
    "claim": "The Weaver+77 Eq. 37 transcription 'T = 1.51e6 K * (L/10^36 erg/s)^(8/35) * (n/1 cm^-3)^(2/35) * t^(-6/35) * (1-xi)^0.4' gives explicit units for L and n but NONE for t, and does not say whether t is dt or t0. The restatement at :171 drops the units entirely.",
    "evidence": "get_InitPhaseParam.py:31; :171; :30 'Temperature coefficient in Weaver+77, Eq. 37'.",
    "expected": "Weaver+77 Eq. 37 uses t6, time in units of 10^6 yr (Myr), measured from bubble formation. The prose should state '(t/1 Myr)^(-6/35)' and name the time variable, as it did for L and n.",
    "failure_scenario": "Feeding t in yr instead of Myr changes T0 by 10^6^(6/35) ~ 11x; feeding t0 instead of dt changes T0 by ((tSF+dt)/dt)^(6/35). The -6/35 exponent is weak enough that a wrong unit produces a wrong-but-plausible temperature rather than an obvious blow-up.",
    "repro": "Compute T0 by hand from the printed L, n, dt in Myr and compare with the code's T0; repeat with t = t0 to see which reproduces it.",
    "confidence": "medium"
  },
  {
    "id": "S3-B-16",
    "file": "trinity/phase0_init/get_InitPhaseParam.py",
    "line": 31,
    "class": "units",
    "severity": "S3",
    "claim": "Weaver Eq. 37 requires n in cm^-3 ('(n/1 cm^-3)^(2/35)', :31), but the immediately preceding block converts the same density to AU units ('Ambient density [AU units: Msun/pc^3]', :144). The prose never says which of the two is passed to the T0 expression.",
    "evidence": "get_InitPhaseParam.py:31; :144-145; :169-171.",
    "expected": "The number-density value in cm^-3 (nCore) must enter the temperature law, not rho_a in Msun/pc^3.",
    "failure_scenario": "Passing rho_a [Msun/pc^3] where n [cm^-3] is expected shifts T0 by (rho_a/n)^(2/35). Because the exponent is 2/35 ~ 0.057, even a 10^2 unit error moves T0 by only ~30% - large enough to matter for cooling, small enough to never look wrong.",
    "repro": "Print the argument actually raised to 2/35 in the T0 expression and compare it against params['nCore'].",
    "confidence": "medium"
  },
  {
    "id": "S3-B-17",
    "file": "trinity/phase0_init/get_InitPhaseParam.py",
    "line": 45,
    "class": "regime",
    "severity": "S2",
    "claim": "The three initial conditions come from two different solution branches evaluated at the same instant, with no reconciliation stated: r0 and v0 are free-streaming ('r0 = terminal velocity * free-streaming duration', 'v0 = wind terminal velocity'), while E0 = (5/11)*Lw*dt is the Weaver energy-driven self-similar result (:26, :166), whose own kinematics require v = (3/5)*R/t, i.e. v0 = 0.6*r0/dt, not r0/dt.",
    "evidence": "get_InitPhaseParam.py:45 (r0, v0 definitions); :26-27 Weaver Eq. 20 with gamma = 5/3; :162 '# Initial separation / bubble radius [pc]'; :166.",
    "expected": "Either the prose flags this as a deliberate approximate hand-off (the free-streaming state seeded with the Weaver energy budget), or the initial state is made self-consistent. As written, (r0, v0, E0) do not lie on the Weaver similarity solution the next phase integrates.",
    "failure_scenario": "The energy-driven integrator is started off the similarity manifold and spends early steps relaxing onto it, producing a stiff transient at t0 - exactly the regime where the bubble-structure integrator's monotonic guard is known to be fragile.",
    "repro": "At t0, check whether E0, r0, v0 satisfy the Weaver relations simultaneously (E = 5/11 L t, R ∝ t^(3/5), v = 3/5 R/t); quantify the mismatch for param/simple_cluster.param.",
    "confidence": "medium"
  },
  {
    "id": "S3-B-18",
    "file": "trinity/phase0_init/get_InitPhaseParam.py",
    "line": 144,
    "class": "regime",
    "severity": "S3",
    "claim": "The free-streaming ambient density is built from nCore, the cloud CORE density, with no stated validity condition. This is only correct if the bubble is still inside the flat core at r0, i.e. r0 <= rCore; the prose never states or checks that.",
    "evidence": "get_InitPhaseParam.py:144-145 '# Ambient density [AU units: Msun/pc^3] / # nCore is hydrogen nuclei density n_H'; get_InitCloudProp.py:150 'n(r) = nCore * (r/rCore)^alpha' (nCore is the density at rCore, constant inside it).",
    "expected": "An assertion or at least a documented assumption that r0 <= rCore, since r0 itself depends on dt which depends on rho_a - a mild self-consistency loop.",
    "failure_scenario": "For a strong wind and a small rCore (project convention says rCore ~ 1 pc), r0 = v_w*dt can exceed rCore; the free-streaming duration is then computed against a density the bubble has already left, biasing dt, r0, E0 and T0 in the same direction.",
    "repro": "Print r0 and rCore for param/simple_cluster.param and the two f1edge configs; check whether r0 <= rCore holds in all of them.",
    "confidence": "medium"
  },
  {
    "id": "S3-B-19",
    "file": "trinity/phase0_init/get_InitPhaseParam.py",
    "line": 107,
    "class": "other",
    "severity": "S3",
    "claim": "The 'CRITICAL: Use WIND-ONLY quantities' mandate is scoped explicitly to the velocity calculation ('The wind terminal velocity v = 2L/pdot is only physical when using wind-only L and pdot (not total which includes SNe)'), and the prose never says which luminosity enters E0 (:166 writes 'L_w') or T0 (:31 and :171 write bare 'L').",
    "evidence": "get_InitPhaseParam.py:107-110; :124 section header '(WIND-ONLY - BUG FIX)'; :166 'E = (5/11) * L_w * t'; :31 '(L/10^36 erg/s)^(8/35)'; :171 '(L/10^36)^(8/35)'.",
    "expected": "Each of Mdot, v, E0, T0 should state wind-only vs total. Weaver's Eq. 20 and Eq. 37 both refer to the mechanical luminosity driving the bubble.",
    "failure_scenario": "Mixing wind-only L for velocity with total L for energy at the same tSF makes E0 inconsistent with (r0, v0); at early tSF the SNe contribution is zero so the bug is invisible, and only appears for tSF past the first SNe.",
    "repro": "Log the L used in each of the four expressions for a tSF after the first supernovae and confirm they are the intended ones.",
    "confidence": "medium"
  },
  {
    "id": "S3-B-20",
    "file": "trinity/phase0_init/get_InitPhaseParam.py",
    "line": 37,
    "class": "silent-failure",
    "severity": "S3",
    "claim": "Three floors are declared under '# Minimum valid values to prevent division by zero': '# Prevent div by zero in Mdot calculation' (:38), '# Prevent div by zero in velocity calculation' (:39), '# Prevent div by zero in dt_phase0 calculation' (:40). The prose does not say which quantity each clamps, what the values are, or whether clamping is logged.",
    "evidence": "get_InitPhaseParam.py:37-40; the formulas at risk are Mdot = pdot^2/(2*L) (:129, L in denominator), v = 2*L/pdot (:133, pdot in denominator) and dt = sqrt(3*Mdot/(4*pi*rho_a*v^3)) (:150, rho_a and v^3 in denominator).",
    "expected": "Clamping should be reported, not silent: a clamped L, pdot or rho_a means the returned t0, r0, v0, E0, T0 are not the physics the user configured.",
    "failure_scenario": "A config with negligible wind feedback (very low mass cluster, or tSF past the wind era) hits a floor; the run continues with a fabricated terminal velocity or duration and produces an apparently normal but meaningless bubble trajectory.",
    "repro": "Set up a config with near-zero wind luminosity at tSF and check whether any clamp fires and whether anything is logged.",
    "confidence": "medium"
  },
  {
    "id": "S3-B-21",
    "file": "trinity/phase0_init/get_InitPhaseParam.py",
    "line": 45,
    "class": "regime",
    "severity": "S3",
    "claim": "bubble_xi_Tb is a required input (:45) feeding the factor (1-xi)^0.4 (:31, :171), but no valid range is stated. The factor requires xi < 1 (T -> 0 at xi = 1) and is complex/NaN for xi > 1.",
    "evidence": "get_InitPhaseParam.py:45 'Must contain: tSF, sps_f, nCore, mu_convert, bubble_xi_Tb'; :31 '* (1-xi)^0.4'; :171 same.",
    "expected": "A documented range 0 <= bubble_xi_Tb < 1 and validation in the INPUT VALIDATION block (:90-92), which the prose says validates params and SPS values but does not mention xi.",
    "failure_scenario": "bubble_xi_Tb = 1 gives T0 = 0 K (a zero initial bubble temperature silently entering the cooling tables); bubble_xi_Tb > 1 gives a NaN or a Python complex from a negative base raised to 0.4.",
    "repro": "Call get_y0 with bubble_xi_Tb = 1.0 and 1.1 and inspect T0.",
    "confidence": "high"
  },
  {
    "id": "S3-B-22",
    "file": "trinity/phase0_init/get_InitPhaseParam.py",
    "line": 188,
    "class": "units",
    "severity": "S3",
    "claim": "The internal-unit annotation '# internal [Msun*pc^2/Myr^3]' (:188) is a POWER (energy per time), not an energy. The adjacent summary line lists 'log Q [1/s], L [erg/s], Mdot [Msun/yr], E0 [erg]' (:186) and '# internal [1/Myr]' (:187). Energy in AU units is Msun*pc^2/Myr^2.",
    "evidence": "get_InitPhaseParam.py:185-188; the docstring's 'E0 : float [au]' (:45), itself an ambiguous token (astro units vs astronomical unit).",
    "expected": "Msun*pc^2/Myr^3 is correct only if it annotates L (luminosity). If it annotates E0, the exponent on Myr is wrong and the erg conversion factor applied to E0 will be off by a factor of one Myr in seconds (3.156e13).",
    "failure_scenario": "E0 reported in the feedback summary is wrong by ~3e13 (or 1/3e13), which looks like an implausible headline number in the run log and in any paper table built from it, while the simulation itself is unaffected.",
    "repro": "Check which variable the :188 annotation sits on and verify the E0 -> erg conversion constant used in the summary block.",
    "confidence": "medium"
  },
  {
    "id": "S3-B-23",
    "file": "trinity/phase0_init/get_InitCloudProp.py",
    "line": 442,
    "class": "regime",
    "severity": "S3",
    "claim": "The radius array is bounded: '# Beyond cloud: up to 1.5 * rCloud' (:442), inner bound only '# Inside cloud: logspace from small radius to rCloud' (:439) with 'small radius' unspecified, plus a single '# Near-origin point for mass profile' (:447). The docstring (:418) documents n_inside/n_outside but not the 1.5 factor or the inner cutoff.",
    "evidence": "get_InitCloudProp.py:418-436, :437, :439, :442, :445, :447.",
    "expected": "The 1.5*rCloud ceiling and the inner cutoff should be documented, since the shell expands past rCloud into the ISM and any consumer interpolating initial_cloud_n_arr / initial_cloud_m_arr beyond 1.5*rCloud is extrapolating.",
    "failure_scenario": "Once the shell exceeds 1.5*rCloud, a density or mass lookup extrapolates off the end of the array - either clamping to the last value or producing a spurious trend - without any error.",
    "repro": "Check r_arr[-1]/rCloud and r_arr[0] for both profiles, and find every consumer of initial_cloud_r_arr to see how out-of-range radii are handled.",
    "confidence": "medium"
  },
  {
    "id": "S3-B-24",
    "file": "trinity/phase0_init/get_InitPhaseParam.py",
    "line": 124,
    "class": "other",
    "severity": "S4",
    "claim": "Section header '# COMPUTE WIND PROPERTIES (WIND-ONLY - BUG FIX)' records a past defect in-band: the wind velocity was previously computed from total (wind+SNe) L and pdot.",
    "evidence": "get_InitPhaseParam.py:124; the supporting rationale at :107-110.",
    "expected": "A regression test pinning wind-only usage, so the marker can be removed from the source. As it stands the only record of the fix is a comment.",
    "failure_scenario": "The fix silently regresses if someone swaps the SPS accessor back to a total-feedback one; nothing fails.",
    "repro": "Check the pytest suite for a test asserting that v0 = 2*L_wind/pdot_wind rather than 2*L_total/pdot_total at a tSF where SNe contribute.",
    "confidence": "high"
  },
  {
    "id": "S3-B-25",
    "file": "trinity/phase0_init/get_InitCloudProp.py",
    "line": 546,
    "class": "deadcode",
    "severity": "S4",
    "claim": "A '# Test / Example usage' block (:546) with '# Mock value class for testing' (:557), 'Test 1: Power-law alpha = -2' (:563), 'Test 2: Homogeneous cloud (alpha = 0)' (:591), 'Test 3: Bonnor-Ebert sphere' (:618), '# Summary' (:649) and '# Check all tests passed' (:661) lives in the module - roughly 120 lines of test harness that pytest never runs.",
    "evidence": "get_InitCloudProp.py:545-661.",
    "expected": "Per project convention tests belong in test/test_*.py. Three named cases (alpha = -2, alpha = 0, BE) are exactly the coverage the pytest suite should hold.",
    "failure_scenario": "The in-module block rots (its mock params class drifts from the real DescribedDict schema), giving false confidence that these three profiles are covered while CI exercises none of them.",
    "repro": "Run the module directly and see whether the three self-tests still pass; grep test/ for equivalent cases.",
    "confidence": "high"
  },
  {
    "id": "S3-B-26",
    "file": "trinity/phase0_init/get_InitCloudProp.py",
    "line": 459,
    "class": "state",
    "severity": "S3",
    "claim": "_ensure_be_params_exist: 'These should normally be created by read_param.py, but this provides a safety fallback for standalone usage' (:459-464), echoed by '# Ensure BE-specific params exist (may not be in read_param.py)' (:344) - a second, duplicate source of BE defaults outside the schema.",
    "evidence": "get_InitCloudProp.py:344; :459-464.",
    "expected": "One source of defaults (the schema in trinity/_input/), per the project convention that .param files override schema defaults and values are not hardcoded elsewhere.",
    "failure_scenario": "The fallback defaults drift from the schema defaults; a BE run silently uses the fallback value for a key the schema also defines, and the two diverge without any test catching it.",
    "repro": "Compare every default this function supplies against the corresponding entry in trinity/_input/ (default.param / schema).",
    "confidence": "medium"
  },
  {
    "id": "S3-B-27",
    "file": "trinity/phase0_init/get_InitCloudProp.py",
    "line": 168,
    "class": "other",
    "severity": "S3",
    "claim": "rCloud is the central computed output ('Self-consistent rCloud computation from fundamental inputs', :3; '# Compute rCloud from physics (not hardcoded!)', :168) and NO formula, integral or defining condition for it appears anywhere in the slice's prose - unlike nEdge, rCore_min, nCore_min, the BE mass ratio and every phase-0 quantity, all of which are written out.",
    "evidence": "get_InitCloudProp.py:3 module docstring 'Key features'; :168; :90 docstring lists rCloud only as an output.",
    "expected": "The defining relation should be stated, presumably mCloud = integral of mu_convert*n(r)*4*pi*r^2 dr from 0 to rCloud with n(r) = nCore*(r/rCore)^alpha (with the flat core inside rCore). Without it, no reader or auditor can check the single most consequential number the module produces - and the whole nEdge/nISM correction dance depends on how rCloud responds to nCore and rCore.",
    "failure_scenario": "Any error in the mass integral (missing flat core inside rCore, wrong mu, wrong limits, alpha = -3 divergence) is undetectable from the documentation and only shows up as a wrong cloud size.",
    "repro": "Recompute rCloud independently from the documented profile and mCloud and compare; check the alpha <= -3 case where the mass integral diverges at the origin without a core.",
    "confidence": "high"
  },
  {
    "id": "S3-B-28",
    "file": "trinity/phase0_init/get_InitCloudProp.py",
    "line": 218,
    "class": "regime",
    "severity": "S3",
    "claim": "'Increasing nCore shrinks rCloud, which only helps (nEdge up)' (:218) is asserted without a sign condition on alpha, and is used to justify the single-shot 'first-order estimate' (:216) instead of iterating. With nEdge = nCore*(rCloud/rCore)^alpha, the argument holds for alpha < 0 (both factors push nEdge up) but not obviously for alpha > 0, where the shrinking rCloud pushes (rCloud/rCore)^alpha DOWN and partially cancels the nCore increase.",
    "evidence": "get_InitCloudProp.py:216-218; :186 nEdge definition; :187 rCore_min formula, which for alpha > 0 and nCore > nISM gives (nCore/nISM)^(1/alpha) > 1, i.e. rCore_min > rCloud - an impossible correction.",
    "expected": "Either alpha < 0 is enforced/documented as a precondition for the whole correction block, or both options handle alpha > 0.",
    "failure_scenario": "With alpha > 0 the Option 1 target is unreachable (rCore_min > rCloud, so the code 'falls through' per :209) and Option 2's single-shot estimate under-corrects, so the final safety check (:251) warns and the run continues with nEdge < nISM.",
    "repro": "Run densPL with a positive densPL_alpha and nEdge < nISM; observe which option is taken, whether it converges, and whether the final warning fires.",
    "confidence": "medium"
  },
  {
    "id": "S3-B-29",
    "file": "trinity/phase0_init/get_InitCloudProp.py",
    "line": 90,
    "class": "regime",
    "severity": "S3",
    "claim": "densBE_Omega is documented only as 'density contrast (rho_core/rho_edge)' with no stated valid range, and xi_out is exposed as 'Dimensionless outer radius' (:52) with no stated range either. Bonnor-Ebert spheres are gravitationally unstable above a critical contrast, and the Lane-Emden solve (:320) must be tabulated over a finite xi range.",
    "evidence": "get_InitCloudProp.py:90; :52; :320 '# Solve Lane-Emden equation (can be cached for efficiency)'; :304 'M(r)/M_cloud = m(xi)/m(xi_out)'.",
    "expected": "A documented (and validated) range for densBE_Omega, and a documented maximum xi the Lane-Emden table covers, since xi_out is inverted from Omega.",
    "failure_scenario": "A densBE_Omega beyond the tabulated xi range makes the xi_out inversion extrapolate or fail; m(xi_out) in the denominator of the mass ratio then poisons the whole mass profile, and the 'exact M(rCloud) = mCloud' guarantee (:304) quietly stops holding.",
    "repro": "Sweep densBE_Omega well above and below the usual range and check xi_out, m(xi_out) and verify_mass_at_rCloud's rel_error at each end.",
    "confidence": "medium"
  }
]
```
