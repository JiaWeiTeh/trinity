# S8 shell structure — Lens B (what the code claims)

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

Prose-only transcription. Two files in slice: `trinity/shell_structure/shell_structure.py`
and `trinity/shell_structure/get_shellODE.py`. I have not seen any implementation; every
line below is a *claim the code makes about itself*, recorded so a code-reading lens can
falsify it. Nothing here is a statement that the code does or does not do the thing.

---

## 1. Formulas stated in prose

### 1.1 Strömgren ionisation balance (the only fully-written equation in the slice)

`trinity/shell_structure/shell_structure.py:234`

```
n_IF_Str = sqrt( 3 (1 - f_esc_ion) Qi / (4 π χ_e α_B ΔV) )
```

i.e.

$$n_{\rm IF,Str} = \sqrt{\frac{3\,(1-f_{\rm esc,ion})\,Q_i}{4\pi\,\chi_e\,\alpha_B\,\Delta V}}$$

Attached regime substitutions (`shell_structure.py:236-238`):

| flag | ΔV | f_esc_ion |
|---|---|---|
| `is_phiDepleted = True`  | `R_IF**3  - R2**3` | `≈ 0` |
| `is_phiDepleted = False` | `R_sh**3  - R2**3` | `phi(R_sh)` |

Plus `shell_structure.py:241`: "R_IF = rShell_arr_ion[-1] in both regimes (I-front or shell edge)".

Cap claim `shell_structure.py:239`: `n_IF_Str ≤ shell_n0`, justified as "pressure
equilibrium for thin skins".

Continuity claim `shell_structure.py:236`: the expression is "Continuous across regimes".
Checkable: at the switch point `R_IF → R_sh` and `phi(R_sh) → 0`, so both branches must
coincide.

**Dimensional note (checkable).** With the explicit `3/(4π)` prefactor present, `ΔV` in this
formula must be a *radius-cubed difference* (`R³−R2³`, as the substitution table literally
says), **not** the physical volume `(4π/3)(R³−R2³)` — the standard balance is
`Qi = χ_e α_B n² V`. The symbol is named `ΔV` ("volume") but substituted as `R³−R2³`. If the
implementation computes `ΔV` as an actual volume *and* keeps the `3/(4π)` prefactor,
`n_IF_Str` is low by `sqrt(4π/3) ≈ 2.05`. See finding S8-B-01.

**Unit self-consistency of the formula as written (my check, consistent):** with code units
`Qi [1/Myr]`, `α_B [pc³/Myr]`, `ΔV [pc³]`, `χ_e` unitless → `n² [pc⁻⁶]` → `n [pc⁻³]`, which
matches the declared `nShell [1/pc3]`.

### 1.2 Infrared optical depth / column

`shell_structure.py:61`

```
tau_IR / kappa_IR = integral(rho dr)
```

i.e. `τ_IR = κ_IR ∫ ρ dr`. The stored dataclass field is the **mass column** `∫ρ dr`, not
`τ_IR` itself. No units given for `κ_IR` or `ρ` at this site (see S8-B-12).

### 1.3 Shell ODE terms — stated only by fragment, never written out

The slice never writes the shell ODE. Only these fragments exist:

- `get_shellODE.py:19` — "The ionised shell ODE has a `dn/dr ∝ +nShell**2` recombination
  term, which is a finite-radius pole: just past the ionisation front nShell runs away
  toward infinity." → **sign claim: the n² recombination term enters `dn/dr` with a `+` sign.**
- `get_shellODE.py:109` — "the `-n*sigma_d*phi` term" (a photon-loss term; from context it
  belongs to `dphi/dr`) → `dφ/dr ⊃ −n σ_d φ`.
- `get_shellODE.py:110` — "`Li*phi` from inverting the radiation pressure gradient" → `dn/dr`
  (or the pressure gradient feeding it) contains a term ∝ `Li · φ`.

No prose anywhere states the pressure-balance / momentum equation that produces `dn/dr`, no
coefficient, no temperature dependence, no reference. See S8-B-19.

### 1.4 Dissolution condition

`shell_structure.py:71`, `shell_structure.py:441` — condition is `shell_nMax < nISM`
(evaluated "instantaneously", "this timestep").

### 1.5 Neutral-region existence condition

`shell_structure.py:220` — "Neutral region exists only when photons are depleted AND mass
remains." Restated `shell_structure.py:301`: "Neutral region integration (if φ depleted and
mass remains)".

### 1.6 Fully-ionised-shell condition

`shell_structure.py:406-407` — "If `shell_ion_idx == len(shell_r_arr)-1`, the entire shell is
ionized (either `is_phiDepleted` with no neutral region, or all mass swept with photons
leaking out)."

### 1.7 Optically-thin / dissolved limit

`shell_structure.py:417` — "dissolved shell = no absorber; ionizing photons escape freely."
Implies `f_esc_ion = 1` and all absorbed fractions = 0 in the dissolved branch. No prose
states the *optically thick* limiting behaviour anywhere in the slice.

---

## 2. Units and conventions

| Claim | Site |
|---|---|
| "All quantities are in code units [Msun, pc, Myr]" | `get_shellODE.py:43` |
| `nShell [1/pc3]` | `get_shellODE.py:43` |
| `phi [unitless]` — "fraction of ionizing photons that reaches a surface with radius r" | `get_shellODE.py:43` |
| `tau [unitless]` — "optical depth of dust in the shell" | `get_shellODE.py:43` |
| `r [pc]` | `get_shellODE.py:43` |
| `f_cover: float, 0 < f_cover <= 1` | `get_shellODE.py:43` |
| return `dndr [1/pc4]` | `get_shellODE.py:43` |
| return `dphidr [1/pc]` (ionised region only) | `get_shellODE.py:43` |
| return `dtaudr [1/pc]` | `get_shellODE.py:43` |
| `α_B` "case-B recombination coeff [code units; physically cm^3/s]" — a conversion is asserted to exist | `get_shellODE.py:85` |
| "unravel, and make sure they are in the right units" — a conversion is asserted at unravel time, unspecified | `get_shellODE.py:95` |
| `phi` "Attenuation function for ionizing flux (unitless)" | `shell_structure.py:119` |
| `n_IF` "Density at ionization front from shell ODE (code units)" — weaker than `[1/pc^3]` | `shell_structure.py:74` |
| `R_IF` "Radius of ionization front (pc)" | `shell_structure.py:76` |
| shell radial grid "[pc]" | `shell_structure.py:80` |
| shell number-density array "[1/pc^3]" | `shell_structure.py:81` |
| `τ_IR/κ_IR = ∫ρ dr` — **no units on κ_IR or ρ** | `shell_structure.py:61` |

**Unit-arithmetic cross-check inside `get_shellODE.py:19-31` (my check, self-consistent):**
"the ionisation front peaks at ~1e65 in code units, i.e. ~1e10 cm^-3". `1 pc⁻³ = (3.086e18 cm)⁻³
= 3.4e-56 cm⁻³`, so `1e65 pc⁻³ = 3.4e9 cm⁻³ ≈ 1e10 cm⁻³`. Consistent. "a neutron star is
~1e38 cm^-3" is also plausible (nuclear density ≈1.4e38 cm⁻³). The unit convention `[1/pc³]`
therefore appears internally coherent with the guard comment.

---

## 3. Citations

Exactly **one** literature citation exists in the whole slice, appearing twice:

- `shell_structure.py:77` — "Strömgren ionization balance density (**Lancaster+2025**), sole
  source of P_HII"
- `shell_structure.py:232` — "Strömgren ionization balance density (**Lancaster+2025,
  generalised**)"

Attributed to it: the formula in §1.1, the two-regime `ΔV`/`f_esc_ion` substitution table,
and (by adjacency) the `n_IF_Str ≤ shell_n0` cap.

**No equation number, no journal, no arXiv identifier, no page.** The word "generalised" is
an admitted departure from the source with the nature of the generalisation unstated. See
S8-B-14.

Two internal-doc citations (both under `docs/dev/`, which project CLAUDE.md declares
unverified — I did not open them):

- `shell_structure.py:33` — "verified across 6 configs in `docs/dev/shell-solver`: the
  `odeint(mxstep=50k)` variant is 1.00x speed with `rel_n=0` in the realistic regimes"
- `get_shellODE.py:30` — "the consumed shell profile is bit-identical to the unguarded solve
  (verified end-to-end, `docs/dev/shell-solver/OVERFLOW_FIX_PLAN.md`)"

No citation exists for: the shell ODE itself, the dust cross-section σ_d, the helium
ionisation assumption, the electron factor χ_e, the temperature/density jump at the I-front,
the gravitational-potential integration, or the dissolution criterion.

---

## 4. Ranges, regimes, assumptions, boundary conditions

**Boundary condition.** `shell_structure.py:117` — "Initialize values at `r = rShell0` = inner
edge of shell." Integration therefore starts at the inner edge and proceeds outward.
`shell_structure.py:123` "Density at inner edge of shell" (`shell_n0`);
`shell_structure.py:120` "tau(r) at ionized region"; `shell_structure.py:143` "Maximum shell
radius (for integration bounds)".

**Two-region structure.** Ionised region integrated first (`shell_structure.py:155`), then a
neutral region *only* if φ depleted and mass remains (`shell_structure.py:220`, `:301`).
`shell_structure.py:306` asserts a "Temperature/density discontinuity at boundary" — the jump
factor is not stated.

**Neutral-branch physics switch.** `get_shellODE.py:43` — "`is_ionised`: Is this part of the
shell ionised? If not, then `phi = Li = 0`, where `r > R_ionised`." `get_shellODE.py:128` —
"If not, omit ionised paramters such as Li and phi." [sic, "paramters"]

**Ionisation state assumption.** `get_shellODE.py:80` — "shell HII is singly ionised
(Z_He_shell)"; `get_shellODE.py:82` — "shell electron factor (singly ionised)". So helium is
assumed **singly** ionised throughout the shell; no validity range is stated for that
assumption (it fails for hard spectra / He II regions).

**Cover fraction range.** `get_shellODE.py:43` — `0 < f_cover <= 1`, "The fraction of shell
that remained after fragmentation process. f_cover = 1: all remained." Contradicted by two
TODOs (§7).

**Guard validity claim.** `get_shellODE.py:26-29` — the `nShell` cap "is ~55 orders of
magnitude above any physical shell density … so it never bites in the used region". With the
stated front peak of `~1e65` code units this puts the cap at `~1e120`. Stated as "a NUMERICAL
safety rail, NOT a physics cutoff".

**Degenerate regime, named.** `shell_structure.py:29-30` — the "degenerate
code-unit-overflow regime" is identified as **`simple_cluster`**, i.e. the tracked baseline
example config. See S8-B-06.

**Optically-thin behaviour.** Only the dissolved case is documented
(`shell_structure.py:417`: photons escape freely). The optically *thick* limit is nowhere
documented.

---

## 5. Contracts (inputs, outputs, state vector, side effects, failure handling)

### 5.1 Purity contract

`shell_structure.py:3` — "This module provides shell structure calculations that **return a
dataclass instead of mutating the params dictionary**. This is essential for use with adaptive
ODE solvers. … `shell_structure_pure()` returns a `ShellProperties` dataclass … **No dictionary
mutations during calculation** … Use `updateDict(params, shell_data)` after call returns."

Reinforced at `shell_structure.py:86` ("does NOT mutate params", "params : DescribedDict
Parameter dictionary (**read-only access**)"), `shell_structure.py:102` ("Read input
parameters (no mutations)"), and `shell_structure.py:210` ("`shell_structure_pure` is
stateless").

Testable contract: call `shell_structure_pure(params)` on a deep-copied `params`, then assert
the original is unchanged, and assert no module-level global is written.

### 5.2 State vector of `get_shellODE`

Two prose statements of ordering, **both agreeing** on the ionised branch:

- `get_shellODE.py:3` (module) — "ODE of the ionised number density (**n**), fraction of
  ionizing photons … (**phi**), and the optical depth (**tau**)"
- `get_shellODE.py:43` (function) — "`y : list` … `# nShell [1/pc3]` … `# phi [unitless]` …
  `# tau [unitless]`"

→ **claimed order `y = [nShell, phi, tau]`; returns `(dndr, dphidr, dtaudr)`.**

**Undocumented second shape.** The docstring qualifies `dphidr` as "(only in ionised region)",
and the neutral branch comments run `unravel` (`:130`) → "number density" (`:139`) → "optical
depth" (`:143`) → "return" (`:146`) with **no φ step**. So the neutral branch evidently takes a
**2-component** `y` and returns a **2-tuple**. The docstring documents only one `y` layout and
never says the neutral branch's `y` is `[nShell, tau]`. See S8-B-04.

**Independent-variable type.** `get_shellODE.py:43` documents `r [pc]: list — An array of radii
where y is evaluated`. For a `odeint` right-hand side, `r` is the scalar independent variable
per call, not an array. See S8-B-18.

### 5.3 `ShellProperties` dataclass — documented meaning of each field

`shell_structure.py:40` — "Dataclass containing all shell structure properties. This can be
used with `updateDict(params, shell_properties)` to update the params dictionary after shell
calculation completes."

| Group | Field meaning (verbatim) | Site |
|---|---|---|
| Shell density | "Shell density" | `:46` |
| | "Density at inner edge of shell" | `:47` |
| Geometry | "Outer radius of shell" | `:50` |
| | "Thickness of shell" | `:51` |
| Absorption | "Fraction of ionizing radiation absorbed" | `:54` |
| | "Fraction of non-ionizing radiation absorbed" | `:55` |
| | "Luminosity-weighted total absorption" | `:56` |
| | "Fraction of ionizing radiation absorbed by dust" | `:57` |
| Shell props | "Maximum density in shell" | `:60` |
| | "tau_IR / kappa_IR = integral(rho dr)" | `:61` |
| Gravity | "Radius array for gravity" / "Gravitational potential" / "Gravitational force per unit mass" | `:64-66` |
| Flags | "Is the shell dissolved?" | `:69` |
| | "Are ionising photons exhausted inside the shell (φ→0)?" | `:70` |
| | "Is shell_nMax < nISM this timestep?" | `:71` |
| I-front | "Density at ionization front from shell ODE (code units)" | `:74` |
| | "**Same as n_IF** (raw ODE value, kept for diagnostics)" | `:75` |
| | "Radius of ionization front (pc)" | `:76` |
| | "Strömgren ionization balance density (Lancaster+2025), **sole source of P_HII**" | `:77` |
| Profile | "Radial grid through shell [pc]" | `:80` |
| | "Number density through shell [1/pc^3]" | `:81` |
| | "Last index of ionized region in shell_r/n_arr (**-1 if empty**)" | `:82` |

Dissolved-branch contract: `shell_structure.py:417` no absorber / free escape; `:428` "Keep
previous rShell value when dissolved (matches original behavior)"; `:430` "No ionization front
when dissolved". Feeder comment `:111`: "Capture previous rShell for dissolved case (original
doesn't update rShell when dissolved)".

### 5.4 Documented error / failure handling — **this is the weak spot**

The *only* prose in the slice about what happens when the integration fails is
`shell_structure.py:28-34`:

> "odeint's default internal step ceiling (mxstep=500) is exhausted in the degenerate
> code-unit-overflow regime (simple_cluster), where it emits 'Excess work done on this call'
> and **silently truncates the shell integration**. Raising the ceiling silences the warning
> and lets the solve complete; where the ceiling was never hit the result is bit-identical …
> Robustness fix only -- same LSODA solver."

Claims: (a) failure mode is *silent truncation*, not an exception; (b) the mitigation is a
larger `mxstep`, not detection; (c) no prose anywhere says the return code / `infodict` is
inspected, nothing says a warning is raised, and nothing says what the caller should do if the
raised ceiling is *also* exhausted. See S8-B-05.

Related soft-failure handling:
- `shell_structure.py:182` "small positive threshold" (φ termination)
- `shell_structure.py:204` "guard against sub-threshold negative phi"
- `get_shellODE.py:98-99` "cap nShell so the +nShell**2 pole in the discarded post-front tail
  cannot overflow float64"
- `get_shellODE.py:102`, `:133` "prevent underflow for very large tau values"
- `get_shellODE.py:108-110` "Clamp phi: negative values are unphysical (ionizing photons
  cannot be regenerated). This prevents the `-n*sigma_d*phi` term from acting as a photon
  source and `Li*phi` from inverting the radiation pressure gradient."

All five are *silent* corrections — no prose claims any of them is counted, logged, or
surfaced to the caller.

---

## 6. Numerical claims

| Claim | Site |
|---|---|
| Solver is `scipy.integrate.odeint` → LSODA; "same LSODA solver" after the fix | `shell_structure.py:28`, `:34` |
| `odeint` default `mxstep = 500` | `shell_structure.py:28` |
| Raised variant is `mxstep = 50k` | `shell_structure.py:33` |
| "1.00x speed with `rel_n=0` in the realistic regimes", "verified across 6 configs" | `shell_structure.py:33` |
| "where the ceiling was never hit the result is bit-identical" | `shell_structure.py:32` |
| φ termination threshold: "first `phi<=1e-9` / mass-limited row" | `get_shellODE.py:22` |
| Same threshold described only as "small positive threshold" | `shell_structure.py:182` |
| float64 ceiling quoted as `1.8e308` | `get_shellODE.py:23` |
| `_NSHELL_MAX` ≈ 55 orders above a `~1e65` code-unit peak ⇒ cap ≈ `1e120` | `get_shellODE.py:26-28` |
| "the `~1e55` dndr prefactor" — with the cap this gives `dndr ≈ 1e295` | `get_shellODE.py:31` |
| Symptom of overflow: "LSODA is driven to machine-precision steps and floods 't + h = t' warnings" | `get_shellODE.py:24` |
| "the consumed shell profile is bit-identical to the unguarded solve (verified end-to-end)" | `get_shellODE.py:29-30` |

No rtol/atol, no grid resolution, no convergence criterion, and no maximum step size is stated
anywhere in the slice.

---

## 7. Admissions of debt (verbatim)

1. `shell_structure.py:114` — "**TODO**: Add f_cover from fragmentation mechanics"
2. `get_shellODE.py:35` — "**TODO**: add cover fraction cf (f_cover)"
3. `get_shellODE.py:111` — "**`# <-- add this line`**" — a stray editing/diff instruction left
   in the source. Pure debris.
4. `shell_structure.py:30` — odeint "**silently truncates** the shell integration" in the
   `simple_cluster` regime (admits a past silent-wrong-answer bug in the flagship config).
5. `shell_structure.py:29` — "the **degenerate** code-unit-overflow regime (simple_cluster)".
6. `shell_structure.py:111` / `:428` — "(**original** doesn't update rShell when dissolved)" /
   "(**matches original behavior**)" — deliberate bug-compat with an unnamed predecessor, no
   rationale given.
7. `shell_structure.py:237` — "f_esc_ion **≈ 0**" — an approximation in a branch condition.
8. `shell_structure.py:232` — "Lancaster+2025, **generalised**" — undocumented departure from
   the cited source.
9. `get_shellODE.py:26` — "This is a **NUMERICAL safety rail, NOT a physics cutoff**" — an
   admission the RHS is deliberately modified away from the physical equation.
10. `shell_structure.py:75` — a field documented as "**Same as** n_IF … kept for diagnostics"
    — an admitted duplicate.
11. `get_shellODE.py:128` — typo "paramters" (cosmetic).

---

## 8. Contradictions and vagueness flags

**A. Cap target stated two different ways.** `shell_structure.py:239` says the cap is
`n_IF_Str ≤ shell_n0` ("pressure equilibrium for thin skins"); `shell_structure.py:250` says
"Cap: thin ionised skin → **P_HII cannot exceed P_b**". A density cap against `shell_n0` and a
pressure cap against `P_b` are only the same operation if `P_b ∝ shell_n0` at a fixed
temperature ratio. One of the two comments is describing something the code does not do.

**B. `f_cover` documented-but-TODO.** `get_shellODE.py:43` documents `f_cover` as a live
parameter with a validity range `0 < f_cover <= 1`; `get_shellODE.py:35` and
`shell_structure.py:114` both say it still needs adding. A documented parameter that the code
admits is unimplemented is either ignored, hardwired to 1, or a trap for callers.

**C. State-vector shape documented once, used two ways.** §5.2 — the neutral branch's
`y = [nShell, tau]` layout has no docstring.

**D. φ threshold stated precisely in one file, vaguely in the other.** `1e-9`
(`get_shellODE.py:22`) vs "small positive threshold" (`shell_structure.py:182`). The
authoritative value lives in the file that does *not* implement the test.

**E. `dissolved` consumed before `diss_condition_met` is computed.** The dissolved branch is
used at `shell_structure.py:417`, `:428`, `:430`; the dissolution condition is only evaluated
at `:441`, and `:209-210` explicitly says "Dissolution condition is now evaluated **after**
shell structure is computed". So the `dissolved` gate acting at `:417` must be an *input* from
a prior timestep — a one-step lag that no comment names.

**F. `-1 if empty` sentinel.** `shell_structure.py:82`. In Python `-1` is a *valid* index (the
last element). Any consumer doing `n_arr[shell_ion_idx]` on an empty ionised region silently
reads the outermost neutral cell instead of erroring.

**G. `ΔV` named as a volume, substituted as `R³−R2³`.** §1.1. Factor-of-`sqrt(4π/3)` risk.

**H. Vague / uncheckable as written:**
- `shell_structure.py:56` "Luminosity-weighted total absorption" — weighting formula unstated.
- `shell_structure.py:54` vs `:57` — "Fraction of ionizing radiation absorbed" vs "Fraction of
  ionizing radiation absorbed by dust": is the latter a fraction of the total ionizing
  luminosity or of the already-absorbed part? Not stated, and the two are used together.
- `shell_structure.py:306` "Temperature/density discontinuity at boundary" — no jump factor.
- `get_shellODE.py:95` "make sure they are in the right units" — no unit named, no conversion
  factor.
- `get_shellODE.py:102`, `:133` "prevent underflow for very large tau values" — no threshold.
- `shell_structure.py:61` `τ_IR/κ_IR = ∫ρ dr` — no units on `κ_IR` or `ρ`.

**I. Whole shell ODE undocumented.** The module exists to return `dn/dr`, `dφ/dr`, `dτ/dr`;
prose never states any of the three equations, only three isolated terms (`+n²`, `−n σ_d φ`,
`Li φ`) mentioned incidentally inside numerical-guard comments. There is no citation for the
shell ODE at all.

**J. Non-smooth RHS not acknowledged.** Both `get_shellODE.py:98` (cap `nShell`) and `:108`
(clamp `phi ≥ 0`) modify the *derivative* as a function of the state. That makes the RHS
piecewise and kinked at `n = _NSHELL_MAX` and `φ = 0` — exactly the discontinuity a stiff BDF
solver's internal Jacobian/Newton loop handles badly. The comments frame both purely as
overflow/sign hygiene and claim bit-identical output; neither acknowledges that the ODE being
solved is no longer the documented one.

**K. Arithmetic margin in the guard comment.** `get_shellODE.py:31` calls
`cap² × 1e55 ≈ 1e295` "well under float64's ceiling" of `1.8e308` — that is only ~13 decades
of headroom on a quantity the same comment says spans 55 decades of uncertainty.

---

```json
[
  {
    "id": "S8-B-01",
    "file": "trinity/shell_structure/shell_structure.py",
    "line": 234,
    "class": "units",
    "severity": "S2",
    "claim": "n_IF_Str = sqrt(3 (1 - f_esc_ion) Qi / (4 pi chi_e alphaB dV)), where the substitution table gives dV = R_IF^3 - R2^3 or R_sh^3 - R2^3.",
    "evidence": "shell_structure.py:234 states the formula with an explicit 3/(4pi) prefactor; shell_structure.py:237-238 substitute dV = R^3 - R2^3, i.e. a radius-cubed difference, not the physical volume (4pi/3)(R^3 - R2^3). The symbol is nonetheless named 'dV' (a volume).",
    "expected": "Standard Stromgren balance Qi(1-f_esc) = chi_e alphaB n^2 V with V = (4pi/3)(R^3-R2^3). The prefactor 3/(4pi) and dV = R^3-R2^3 are mutually consistent; a physical volume for dV plus the 3/(4pi) prefactor is not.",
    "failure_scenario": "If the implementation computes dV as an actual volume (4pi/3)(R^3-R2^3) while retaining the 3/(4pi) prefactor, n_IF_Str is low by sqrt(4pi/3) = 2.046, so P_HII is low by ~2.05x (or ~4.2x if pressure goes as n^2/T-weighted terms). Since line 77 calls n_IF_Str the 'sole source of P_HII', this propagates straight into the shell force budget and the expansion history.",
    "repro": "Assert n_IF_Str**2 * chi_e * alphaB * (4*pi/3) * (R_IF**3 - R2**3) == (1 - f_esc_ion) * Qi to within 1e-12 relative, for a state captured at a mid-evolution timestep.",
    "confidence": "medium"
  },
  {
    "id": "S8-B-02",
    "file": "trinity/shell_structure/shell_structure.py",
    "line": 239,
    "class": "other",
    "severity": "S3",
    "claim": "The n_IF_Str cap is described twice with two different targets: 'n_IF_Str <= shell_n0 (pressure equilibrium for thin skins)' at line 239, and 'thin ionised skin -> P_HII cannot exceed P_b' at line 250.",
    "evidence": "shell_structure.py:239 vs shell_structure.py:250. A density cap against shell_n0 and a pressure cap against P_b are the same operation only if P_b is proportional to shell_n0 at a fixed temperature.",
    "expected": "One cap, described consistently, with the temperature convention that links n to P made explicit.",
    "failure_scenario": "If the code caps on shell_n0 but the intended physics is P_HII <= P_b, the cap binds at the wrong point whenever T_HII differs from the shell/bubble temperature used to define P_b, either over- or under-limiting HII pressure in thin-skin timesteps.",
    "repro": "At a timestep where the cap is active, check whether the clipped quantity equals shell_n0 exactly or whether the resulting P_HII equals P_b exactly. Only one can hold.",
    "confidence": "medium"
  },
  {
    "id": "S8-B-03",
    "file": "trinity/shell_structure/get_shellODE.py",
    "line": 43,
    "class": "deadcode",
    "severity": "S2",
    "claim": "f_cover is documented as a live parameter with validity range 0 < f_cover <= 1 ('The fraction of shell that remained after fragmentation process'), yet two TODOs say the cover fraction still needs to be added.",
    "evidence": "get_shellODE.py:43 documents the parameter and its range; get_shellODE.py:35 says 'TODO: add cover fraction cf (f_cover)'; shell_structure.py:114 says 'TODO: Add f_cover from fragmentation mechanics'.",
    "expected": "Either f_cover is applied to the ODE terms it should scale (photon flux / column through the covered fraction), or it is not a parameter and the docstring should not advertise it with a validity range.",
    "failure_scenario": "A caller passes f_cover < 1 expecting a partially fragmented shell and it is silently ignored, so absorbed fractions, tau, and the radiation force are computed as if the shell were fully covering. Fragmentation physics is silently absent while the API says it is supported.",
    "repro": "Call get_shellODE with f_cover=1.0 and f_cover=0.5 on identical y and r; if the returned derivatives are identical the parameter is inert.",
    "confidence": "high"
  },
  {
    "id": "S8-B-04",
    "file": "trinity/shell_structure/get_shellODE.py",
    "line": 43,
    "class": "state",
    "severity": "S2",
    "claim": "The docstring documents exactly one state vector, y = [nShell, phi, tau] returning (dndr, dphidr, dtaudr); but it also says dphidr is returned 'only in ionised region', implying an undocumented second, 2-component layout for the neutral branch.",
    "evidence": "get_shellODE.py:3 and :43 both give the 3-component order (n, phi, tau). The return section qualifies 'dphidr [1/pc]: ODE (only in ionised region)'. The neutral branch comments run unravel (:130) -> number density (:139) -> optical depth (:143) -> return (:146), with no phi step, and :128 says 'If not, omit ionised paramters such as Li and phi.'",
    "expected": "The docstring should state both shapes explicitly: ionised y = [nShell, phi, tau], neutral y = [nShell, tau], with the matching return arities, since the caller must pack y differently per branch.",
    "failure_scenario": "A caller (or a future edit) packs the neutral initial condition as [n, phi, tau] or unpacks a 2-tuple as 3 values. The most dangerous silent variant: neutral y unpacked as (nShell, tau) while the caller passed (nShell, phi) -- phi is then used as the optical depth, corrupting the exp(-tau) attenuation with no exception raised.",
    "repro": "Inspect the arity of the y passed to odeint for the neutral integration in shell_structure and the arity returned by the neutral branch of get_shellODE; assert they match and match the docstring.",
    "confidence": "high"
  },
  {
    "id": "S8-B-05",
    "file": "trinity/shell_structure/shell_structure.py",
    "line": 30,
    "class": "silent-failure",
    "severity": "S2",
    "claim": "odeint 'emits \"Excess work done on this call\" and silently truncates the shell integration'; the documented mitigation is raising mxstep from 500 to 50k, and nothing in the prose claims the solver return status is ever inspected.",
    "evidence": "shell_structure.py:28-34. The comment describes the failure as silent truncation, describes the fix as 'Robustness fix only -- same LSODA solver', and scopes the equivalence claim to 'where the ceiling was never hit'.",
    "expected": "The integration should detect a non-successful odeint return (full_output infodict message / istate) and either raise, retry, or record a flag on ShellProperties, rather than consuming a truncated profile as if it were converged.",
    "failure_scenario": "In any regime stiffer than the ones tested, mxstep=50k is also exhausted; odeint returns a short/garbage profile with no exception. shell_structure then derives shell_nMax, f_absorbed, R_IF and n_IF_Str from a truncated profile, and the run continues to completion producing physically wrong but plausible-looking output. The comment itself confirms this already happened once, in the tracked baseline config.",
    "repro": "Call odeint with full_output=1 in the shell integration and assert infodict message is 'Integration successful.'; then run param/simple_cluster.param and the stiffest available edge config and check the assertion never fires.",
    "confidence": "high"
  },
  {
    "id": "S8-B-06",
    "file": "trinity/shell_structure/shell_structure.py",
    "line": 29,
    "class": "regime",
    "severity": "S2",
    "claim": "The 'degenerate code-unit-overflow regime' in which the shell integration overflows and was silently truncated is identified by name as simple_cluster.",
    "evidence": "shell_structure.py:29-30 names simple_cluster as the config exhibiting the failure. Separately, get_shellODE.py:19-31 says the ionisation front peaks at ~1e65 in code units and the discarded post-front tail overflows float64.",
    "expected": "The tracked quickstart/baseline config (param/simple_cluster.param, per project CLAUDE.md the documented single-run example) should not be the config that drives the shell solver into float64 overflow. Either the code-unit scaling of nShell needs revisiting or the config is mislabelled as the baseline.",
    "failure_scenario": "The example every new user and every regression comparison runs is the one operating in an admitted numerical-degeneracy regime. Any equivalence gate anchored on simple_cluster is measuring behaviour dominated by overflow guards rather than physics.",
    "repro": "Run param/simple_cluster.param and record max(nShell) reached inside get_shellODE (including the discarded tail) against _NSHELL_MAX and against float64 max.",
    "confidence": "high"
  },
  {
    "id": "S8-B-07",
    "file": "trinity/shell_structure/shell_structure.py",
    "line": 75,
    "class": "deadcode",
    "severity": "S3",
    "claim": "A ShellProperties field is documented as 'Same as n_IF (raw ODE value, kept for diagnostics)', i.e. two fields holding the same value.",
    "evidence": "shell_structure.py:74 'Density at ionization front from shell ODE (code units)'; shell_structure.py:75 'Same as n_IF (raw ODE value, kept for diagnostics)'. Assignment comments at :224 ('Density at ionization front from shell ODE') and :225 ('Preserve raw ODE value for diagnostics') mirror this.",
    "expected": "If the two are genuinely always equal, one is redundant state. If one is later clipped, overwritten or reset (e.g. the dissolved branch at :430 'No ionization front when dissolved'), then 'Same as n_IF' is false and the docstring misleads any consumer choosing between them.",
    "failure_scenario": "A consumer picks the 'raw' field believing it equals n_IF, but n_IF has since been clipped or zeroed (dissolved branch), so downstream pressure/diagnostics disagree between two supposedly identical outputs.",
    "repro": "Assert the two fields are equal on every returned ShellProperties across a full run, including dissolved and phi-depleted timesteps.",
    "confidence": "medium"
  },
  {
    "id": "S8-B-08",
    "file": "trinity/shell_structure/shell_structure.py",
    "line": 82,
    "class": "state",
    "severity": "S3",
    "claim": "shell_ion_idx is 'Last index of ionized region in shell_r/n_arr (-1 if empty)'.",
    "evidence": "shell_structure.py:82 defines the sentinel; shell_structure.py:405-407 reuse the field for the fully-ionised test 'If shell_ion_idx == len(shell_r_arr)-1, the entire shell is ionized'.",
    "expected": "A sentinel that cannot be confused with a valid index, or every consumer explicitly testing 'if shell_ion_idx < 0' before indexing.",
    "failure_scenario": "-1 is a legal Python index. Any consumer doing shell_n_arr[shell_ion_idx] or shell_r_arr[shell_ion_idx] on an empty ionised region silently reads the outermost neutral cell instead of raising, so 'no ionised region' is reported as 'ionisation front at the shell outer edge with the neutral outer density'. Also arr[:shell_ion_idx] with -1 silently drops the last element rather than yielding nothing.",
    "repro": "Construct a timestep with no ionised region and grep every consumer of shell_ion_idx for a guarded (< 0) read before indexing or slicing.",
    "confidence": "medium"
  },
  {
    "id": "S8-B-09",
    "file": "trinity/shell_structure/shell_structure.py",
    "line": 77,
    "class": "state",
    "severity": "S3",
    "claim": "n_IF_Str is the 'sole source of P_HII'.",
    "evidence": "shell_structure.py:77. Yet the dataclass also carries n_IF ('Density at ionization front from shell ODE', :74) and its documented duplicate (:75), which are the natural alternative inputs to an HII pressure.",
    "expected": "Exactly one code path computing P_HII, reading n_IF_Str and not n_IF or its duplicate.",
    "failure_scenario": "If any consumer computes an HII pressure from n_IF (the raw, uncapped ODE value that the guard comment says can reach ~1e65 code units near the pole), P_HII is unbounded and the cap at :239/:250 is bypassed, blowing up the shell force budget on a single stiff timestep.",
    "repro": "Grep all uses of n_IF, its diagnostic duplicate, and n_IF_Str; assert only n_IF_Str feeds any pressure calculation.",
    "confidence": "medium"
  },
  {
    "id": "S8-B-10",
    "file": "trinity/shell_structure/get_shellODE.py",
    "line": 111,
    "class": "deadcode",
    "severity": "S4",
    "claim": "The source contains the comment '# <-- add this line' immediately after the phi-clamp comment block.",
    "evidence": "get_shellODE.py:111, directly following get_shellODE.py:108-110 which explain the phi clamp.",
    "expected": "No editing/diff instructions committed into source. The marker suggests the phi clamp was pasted in from an external instruction and the annotation was never removed.",
    "failure_scenario": "Cosmetic on its own, but it is a marker that the adjacent clamp was applied without review; worth checking that the clamp landed on the intended line and that nothing else from the same paste is missing.",
    "repro": "",
    "confidence": "high"
  },
  {
    "id": "S8-B-11",
    "file": "trinity/shell_structure/get_shellODE.py",
    "line": 26,
    "class": "numerical",
    "severity": "S3",
    "claim": "The nShell cap is 'a NUMERICAL safety rail, NOT a physics cutoff' and 'the consumed shell profile is bit-identical to the unguarded solve'; separately phi is clamped non-negative inside the RHS.",
    "evidence": "get_shellODE.py:26-30 (cap, bit-identical claim, cites docs/dev/shell-solver/OVERFLOW_FIX_PLAN.md) and get_shellODE.py:108-110 (phi clamp). Both modify the derivative as a function of the state.",
    "expected": "Acknowledgement that clamping inside the RHS makes it piecewise and non-differentiable at n = _NSHELL_MAX and phi = 0, which is precisely what LSODA's internal Jacobian and step-size controller are sensitive to. The bit-identical claim rests on a docs/dev writeup that project CLAUDE.md declares unverified.",
    "failure_scenario": "The kink at phi = 0 sits exactly where the phi<=1e-9 termination test fires, i.e. in the region the profile is actually read from. If LSODA rejects steps or reduces order near that kink, the accepted output row at the ionisation front can shift, moving R_IF and n_IF, and hence P_HII, without any warning.",
    "repro": "Re-derive the bit-identical claim independently of docs/dev: run a config in a separate process with and without each guard, and diff the consumed shell profile arrays and dictionary.jsonl byte-for-byte at matched simulation time.",
    "confidence": "medium"
  },
  {
    "id": "S8-B-12",
    "file": "trinity/shell_structure/shell_structure.py",
    "line": 61,
    "class": "units",
    "severity": "S3",
    "claim": "A ShellProperties field is documented as 'tau_IR / kappa_IR = integral(rho dr)', i.e. the field stores the mass column density, not tau_IR.",
    "evidence": "shell_structure.py:61. No units are given for kappa_IR or rho at this site, and no unit annotation is given for the field, unlike the neighbouring shell_r_arr '[pc]' and n_arr '[1/pc^3]' at :80-81.",
    "expected": "An explicit unit for the stored column (code units Msun/pc^2 vs cgs g/cm^2) and for the kappa_IR the consumer must multiply by, since kappa_IR is conventionally quoted in cm^2/g while the rest of the module is in [Msun, pc, Myr].",
    "failure_scenario": "A consumer multiplies a code-unit column (Msun/pc^2) by a cgs opacity (cm^2/g) without conversion. The Msun/pc^2 -> g/cm^2 factor is ~0.0002, so tau_IR would be wrong by roughly four orders of magnitude, silently switching the IR-trapping term between negligible and dominant.",
    "repro": "Find where this field is multiplied by kappa_IR and check the unit of each factor at that site against trinity/_functions/unit_conversions.py.",
    "confidence": "medium"
  },
  {
    "id": "S8-B-13",
    "file": "trinity/shell_structure/get_shellODE.py",
    "line": 102,
    "class": "numerical",
    "severity": "S4",
    "claim": "'prevent underflow for very large tau values', stated identically in the ionised branch (:102) and the neutral branch (:133), with no threshold value given in either.",
    "evidence": "get_shellODE.py:102 and get_shellODE.py:133.",
    "expected": "A named threshold (or a documented clamp on tau / on exp(-tau)) so the value can be checked, and confirmation that the two branches use the same one.",
    "failure_scenario": "exp(-tau) underflowing to 0.0 is already graceful in IEEE754, so the guard is presumably clamping tau itself. If the two branches clamp at different values, the attenuation is discontinuous across the ionisation front, on top of the discontinuity already claimed at :306.",
    "repro": "Compare the clamp constant used at :102 and :133; assert they are the same symbol.",
    "confidence": "medium"
  },
  {
    "id": "S8-B-14",
    "file": "trinity/shell_structure/shell_structure.py",
    "line": 232,
    "class": "citation",
    "severity": "S3",
    "claim": "The only literature citation in the slice is 'Lancaster+2025' (at :77 and :232), attached to the Stromgren balance formula, the two-regime dV/f_esc_ion substitution table, and the adjacent cap. Line 232 qualifies it as 'generalised'.",
    "evidence": "shell_structure.py:77 'Stromgren ionization balance density (Lancaster+2025), sole source of P_HII'; shell_structure.py:232 'Stromgren ionization balance density (Lancaster+2025, generalised)'. No equation number, no journal, no arXiv id, no page anywhere in the slice.",
    "expected": "A resolvable reference plus the specific equation number, and an explicit statement of what 'generalised' changed relative to the source (the two-regime dV switch and the shell_n0 cap are the obvious candidates and are not in a plain Stromgren balance).",
    "failure_scenario": "The formula cannot be checked against its source. In particular the reader cannot tell whether the 3/(4pi) prefactor, the R^3-R2^3 shell-volume form, the (1-f_esc_ion) factor, or the shell_n0 cap come from the paper or are local inventions -- which is exactly the ambiguity behind S8-B-01.",
    "repro": "",
    "confidence": "high"
  },
  {
    "id": "S8-B-15",
    "file": "trinity/shell_structure/shell_structure.py",
    "line": 182,
    "class": "numerical",
    "severity": "S3",
    "claim": "The ionised-region termination threshold is described as 'small positive threshold' in the file that applies it, and as the specific value 'phi<=1e-9' only in the other file.",
    "evidence": "shell_structure.py:182 'small positive threshold' (at 'Find termination index', :180); get_shellODE.py:22 'shell_structure truncates the profile AT the front (first phi<=1e-9 / mass-limited row)'.",
    "expected": "The numeric threshold documented where it is used, and the get_shellODE comment kept in sync with it (or referring to the shared constant).",
    "failure_scenario": "The threshold is changed in shell_structure.py and get_shellODE.py:19-31 -- the entire justification for _NSHELL_MAX, which depends on how far past the front the discarded tail extends -- becomes stale. A larger threshold truncates earlier (guard over-conservative); a smaller one lets the n^2 pole run further before truncation, potentially past the cap's headroom.",
    "repro": "Read the constant used at shell_structure.py:182 and compare with the 1e-9 quoted at get_shellODE.py:22.",
    "confidence": "medium"
  },
  {
    "id": "S8-B-16",
    "file": "trinity/shell_structure/shell_structure.py",
    "line": 306,
    "class": "other",
    "severity": "S3",
    "claim": "'Temperature/density discontinuity at boundary' -- a jump is applied to the state handed from the ionised integration to the neutral integration, with no factor, no temperature values, and no justification stated.",
    "evidence": "shell_structure.py:306, inside the 'Neutral region integration (if phi depleted and mass remains)' block (:300-302).",
    "expected": "The jump condition written out (conventionally pressure continuity across the I-front, n_neutral = n_ion * (chi_e T_ion / T_neutral) or similar), with the temperatures named and their source in params identified.",
    "failure_scenario": "The jump sets the neutral region's initial density and hence the whole neutral column, shell_nMax, tau, and the absorbed fractions. An unstated factor is unauditable: a missing chi_e (electrons contributing to ionised pressure but not neutral) is a factor ~2 error in the neutral density and column, silently changing the non-ionising absorbed fraction.",
    "repro": "Extract the multiplicative factor applied at the boundary and check it against pressure continuity with the ionised and neutral temperatures actually used.",
    "confidence": "medium"
  },
  {
    "id": "S8-B-17",
    "file": "trinity/shell_structure/shell_structure.py",
    "line": 417,
    "class": "state",
    "severity": "S3",
    "claim": "The dissolved branch is consumed at :417 ('dissolved shell = no absorber; ionizing photons escape freely'), :428 and :430, while the dissolution condition itself is only evaluated at :441 -- and :209-210 states 'Dissolution condition is now evaluated after shell structure is computed (see diss_condition_met below); shell_structure_pure is stateless.'",
    "evidence": "shell_structure.py:209-210, :417, :428, :430, :441. Two separate flags exist for this: 'Is the shell dissolved?' (:69) and 'Is shell_nMax < nISM this timestep?' (:71).",
    "expected": "Explicit documentation that the 'dissolved' gate acting during this call is an input carried in from the previous timestep, while diss_condition_met is this timestep's fresh evaluation -- i.e. a deliberate one-step lag.",
    "failure_scenario": "If the lag is not intended, the shell is treated as an absorber for one extra timestep after it dissolves (f_esc_ion under-reported, radiation force over-reported), or as dissolved one step early. If it is intended, nothing records that, and a future edit collapsing the two flags into one silently changes the dissolution timing.",
    "repro": "Trace which flag gates :417/:428/:430 and whether it originates from params (previous step) or from the value computed at :441.",
    "confidence": "medium"
  },
  {
    "id": "S8-B-18",
    "file": "trinity/shell_structure/get_shellODE.py",
    "line": 43,
    "class": "other",
    "severity": "S4",
    "claim": "The docstring documents the independent variable as 'r [pc]: list -- An array of radii where y is evaluated.'",
    "evidence": "get_shellODE.py:43. The function is the right-hand side handed to odeint (shell_structure.py:28-34 confirms odeint/LSODA), which calls the RHS with a scalar independent variable per evaluation.",
    "expected": "'r [pc]: float -- the radius at which the derivative is evaluated', matching how odeint invokes an RHS. The array of radii is the output grid passed to odeint, not an argument of the RHS.",
    "failure_scenario": "Mostly documentation drift, but it misleads anyone adding an r-dependent term (e.g. a geometric 2/r factor or a radiation-dilution 1/r^2): writing it array-style would broadcast wrongly or raise, and writing it scalar-style contradicts the stated contract.",
    "repro": "",
    "confidence": "high"
  },
  {
    "id": "S8-B-19",
    "file": "trinity/shell_structure/get_shellODE.py",
    "line": 3,
    "class": "other",
    "severity": "S3",
    "claim": "The module's stated purpose is to return the ODEs for n, phi and tau, but none of the three equations is written anywhere in the slice; only three isolated terms appear, and only incidentally inside numerical-guard comments: dn/dr contains '+nShell**2' (:19), dphi/dr contains '-n*sigma_d*phi' (:109), and a 'Li*phi' term drives a radiation pressure gradient (:110).",
    "evidence": "get_shellODE.py:3, :19, :109-110; and the derivative-labelling comments 'number density' (:114, :139), 'ionising photons' (:119), 'optical depth' (:121, :143) which name the outputs without stating them.",
    "expected": "The three RHS expressions written out with their coefficients, plus a citation for the shell momentum/pressure-balance equation. There is no literature citation for the shell ODE anywhere in this slice.",
    "failure_scenario": "The core equation of the module is unauditable from prose. Every coefficient, temperature, mean-molecular-weight and sigma_d in dn/dr is unchecked, and a sign or factor error there cannot be caught by review -- only the '+' sign of the n^2 recombination term is documented, and only because it happens to cause an overflow.",
    "repro": "",
    "confidence": "high"
  },
  {
    "id": "S8-B-20",
    "file": "trinity/shell_structure/shell_structure.py",
    "line": 428,
    "class": "other",
    "severity": "S4",
    "claim": "'Keep previous rShell value when dissolved (matches original behavior)' (:428), set up by 'Capture previous rShell for dissolved case (original doesn't update rShell when dissolved)' (:111).",
    "evidence": "shell_structure.py:111 and :428.",
    "expected": "Either a physical justification for freezing rShell at dissolution, or an acknowledgement that this is bug-compatibility with a predecessor implementation that is being preserved deliberately. 'matches original behavior' is not a justification and 'the original' is never identified.",
    "failure_scenario": "Downstream consumers reading rShell after dissolution get a stale radius that no longer tracks the (now unresolved) shell, and there is no documented way to tell a frozen rShell from a live one apart from the dissolved flag.",
    "repro": "",
    "confidence": "high"
  },
  {
    "id": "S8-B-21",
    "file": "trinity/shell_structure/get_shellODE.py",
    "line": 31,
    "class": "numerical",
    "severity": "S4",
    "claim": "'The value keeps nShell**2 times the ~1e55 dndr prefactor well under float64's ceiling', where the cap is stated as ~55 orders of magnitude above a front peak of ~1e65 code units (:26-28), i.e. cap ~1e120.",
    "evidence": "get_shellODE.py:26-31 and the float64 ceiling of 1.8e308 quoted at :23. Working: (1e120)^2 * 1e55 = 1e295, versus 1.8e308.",
    "expected": "'Well under' should mean a margin large compared with the uncertainty in the quantities involved. Here it is ~13 decades of headroom on estimates the same comment gives only to one significant figure, against a quantity the comment says spans 55 decades.",
    "failure_scenario": "If the dndr prefactor is larger than ~1e55 in some configuration (it is stated as an approximation, and it presumably scales with cluster luminosity and shell parameters), or if the actual cap constant is above 1e120, the product overflows anyway and the guard does not prevent the inf/nan it was written to prevent.",
    "repro": "Read the literal _NSHELL_MAX constant and assert _NSHELL_MAX**2 * (max observed dndr prefactor) < 1e300 across the available edge configs.",
    "confidence": "medium"
  },
  {
    "id": "S8-B-22",
    "file": "trinity/shell_structure/shell_structure.py",
    "line": 237,
    "class": "other",
    "severity": "S4",
    "claim": "In the is_phiDepleted=True branch the escape fraction is documented as 'f_esc_ion ~= 0' (approximately zero), while the other branch uses the exact value phi(R_sh).",
    "evidence": "shell_structure.py:237-238. Line 236 claims the whole expression is 'Continuous across regimes'.",
    "expected": "Either f_esc_ion is set exactly to 0 in that branch (in which case the '~=' should say '='), or it retains a small residual, in which case the continuity claim at :236 needs the residual to match phi(R_sh) at the switch point.",
    "failure_scenario": "If the depleted branch hardwires 0 while the other branch reads phi(R_sh) at the termination threshold (documented as phi <= 1e-9), the (1-f_esc_ion) factor jumps by at most 1e-9 -- harmless. But if the branches disagree on which radius R_IF refers to, the claimed continuity fails by a finite amount at the flag flip, producing a step in P_HII.",
    "repro": "Force a timestep at the is_phiDepleted boundary and evaluate n_IF_Str with both branches; assert they agree to within the phi termination threshold.",
    "confidence": "medium"
  },
  {
    "id": "S8-B-23",
    "file": "trinity/shell_structure/shell_structure.py",
    "line": 54,
    "class": "other",
    "severity": "S4",
    "claim": "Four absorption-fraction fields are documented one line each with no formulas and no stated relationship: 'Fraction of ionizing radiation absorbed' (:54), 'Fraction of non-ionizing radiation absorbed' (:55), 'Luminosity-weighted total absorption' (:56), 'Fraction of ionizing radiation absorbed by dust' (:57).",
    "evidence": "shell_structure.py:53-57, plus :275 'Dust vs hydrogen absorption' and :397 'Absorption fractions (f_esc_ion computed above)'.",
    "expected": "The weighting formula for the luminosity-weighted total (presumably (f_ion*L_ion + f_nonion*L_nonion)/L_tot), and an explicit statement of whether the dust-absorbed ionizing fraction is normalised to the total ionizing luminosity or to the already-absorbed part.",
    "failure_scenario": "The dust fraction being a fraction-of-a-fraction versus a fraction-of-total differs by exactly the factor f_ion_absorbed. A consumer splitting the ionising budget into dust and hydrogen channels using the wrong normalisation double-counts or under-counts the dust-absorbed ionising photons, which feeds the radiation force and the IR-trapping term.",
    "repro": "Check whether the four fractions satisfy the identity a consumer would assume: f_ion_dust <= f_ion_absorbed <= 1, and f_total equal to the luminosity-weighted combination of f_ion and f_nonion.",
    "confidence": "medium"
  },
  {
    "id": "S8-B-24",
    "file": "trinity/shell_structure/shell_structure.py",
    "line": 3,
    "class": "state",
    "severity": "S3",
    "claim": "shell_structure_pure does NOT mutate params ('read-only access', 'No dictionary mutations during calculation', 'shell_structure_pure is stateless'), and this purity is 'essential for use with adaptive ODE solvers'; the caller is expected to apply updateDict(params, shell_data) after the call returns.",
    "evidence": "shell_structure.py:3-16 (module docstring), :40-45 (dataclass docstring), :86-101 (function docstring, 'params : DescribedDict Parameter dictionary (read-only access)'), :102 ('Read input parameters (no mutations)'), :210 ('shell_structure_pure is stateless').",
    "expected": "A regression test asserting the contract, since it is the module's stated reason for existing and the outer adaptive solver's correctness depends on it: a deep-copied params must compare equal before and after, and no module-level global may be written.",
    "failure_scenario": "An adaptive solver evaluates the shell structure at trial steps that are later rejected. If any write to params or to a module global leaks (project CLAUDE.md rule 5 explicitly warns 'trinity leaks module-level global state in-process'), a rejected trial step permanently contaminates the accepted trajectory -- a history-dependent, non-reproducible error that no single-step equivalence test would catch.",
    "repro": "Deep-copy params, call shell_structure_pure, assert the copy compares equal to the original; and call it twice with identical inputs asserting bit-identical ShellProperties on the second call.",
    "confidence": "medium"
  }
]
```
