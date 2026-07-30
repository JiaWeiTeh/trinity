# S7 bubble structure — reconciled

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

**Status (2026-07-30):** 📘 reconciled agent report for slice S7 (★ high-stiffness). Inputs were the
three raw lens reports only (`S7_bubble_lensA.md`, `S7_bubble_lensB.md`, `S7_bubble_lensC.md`).
**No source was read.** Every statement below is an alignment of the three lenses' claims, not a
verification against `trinity/`.

**Raw input volume:** A = 29 candidates, B = 23, C = 44 (96 total). **Reconciled output: 30 items,
0 × S1, 11 × S2, 12 × S3, 7 × S4.** 22 dropped or folded as refuted/moot, the rest merged.

---

## 0. How to read this

- **A** = what the code computes (comment-stripped source). **B** = what the prose claims. **C** =
  what the physics requires (spec + first-principles derivation, no code seen).
- `corroborated` = ≥2 lenses independently reached it. `single-lens` = only one lens saw it and the
  others were **silent** (silence is not agreement). `contested` = the lenses positively disagree
  and I cannot adjudicate from the reports alone.
- **Lens C refused to state Weaver+77 prefactors and equation numbers** (literature unreachable in
  its container). Every A≠C gap that lands on a number C declined to pin is filed as an **open
  question**, not a defect, and listed in §8.

---

## 1. Coverage table

Which lenses actually spoke to each quantity. "—" means the lens was silent (**not** that it agreed).

| # | Quantity | A | B | C | Status | Verdict |
|---|---|---|---|---|---|---|
| 1 | Luminosity **volume element** | `4πr²dr` in all 3 zones | — | `4πr²dr`, nothing else | corroborated (A,C) | **agree** |
| 2 | Luminosity **integrand densities** | `χ_e·n²·Λ`, `n = P_b/((μ_conv/μ_ion)k_B T)` | region tags only | `n_e n_H Λ`; `n_tot²` = ×4.4, `n_H²` = ÷1.2 | corroborated (A,C) | **structurally agree**, conditional on 2 constants |
| 3 | Luminosity **limits** | `[R1, R2 + (2/3)dR2]` | region 3 "tiny (or non-existent)" | `[R1, r2Prime]`, `r2Prime < R2` | contested (A≠B≠C) | **diverge — top finding** |
| 4 | **Bolometric vs band-limited** | 1-D CIE `Λ(log₁₀T)` + non-CIE net cubes | — | must be bolometric | single-lens (C) | **no evidence of band-limiting** |
| 5 | Interior ODE `dv/dr` | 4 terms | "Eqs 42-43 Weaver" | 4 terms, derived | corroborated (A,C) | **term-identical** |
| 6 | Interior ODE `d²T/dr²` incl. cooling **sign** | net-heating `−u̇/(κC T^{5/2})` | "Eqs 42-43" | `+Λ_vol/(C T^{5/2})` | corroborated (A,C) | **agree** (A's `u̇` = heat−cool ⇒ same sign) |
| 7 | Front asymptotic `25/4`, `2/5`, `μ_ion` | `K=(25/4)k_B/(μ_ion κC)`, `T^{2/5}`, `dT/dr=−(2/5)T/dR2` | "Eq 44 Weaver" | identical (medium on `25/4`) | corroborated (A,C) | **agree** |
| 8 | Isobaric density convention | `n·T = P_b/((μ_conv/μ_ion)k_B)` | — | `n_H T = P_b/(2.3 k_B)` if n_H stored | corroborated (A,C) | **agree if μ_conv/μ_ion = 2.3** |
| 9 | **Monotonicity guard** — what it tests | `operations.monotonic(T_array)`, search solve | `find_nearest_higher` → `MonotonicError`; 2 known triggers | guard T and n, **never v** | corroborated (A,C) | tests **T** ⇒ C's fear unrealised |
| 10 | Monotonicity guard — what it returns | constant `+1e2` | never documented at the function | must not manufacture a root | corroborated (A,B,C) | **defect** |
| 11 | Monotonicity re-checked on the used profile? | **no** (only `T<0`) | — | — | single-lens (A) | **gap** |
| 12 | **Region beyond R₂** | intermediate zone → `R2+(2/3)dR2`, `n = P_b/(2.3k_BT)` | "tiny (or non-existent)" | shell module owns `r>R2` | corroborated (A,C) + A≠B | **defect** |
| 13 | **Sign conventions in `L_total`** | `L_bub` = +cooling; `L_cond`,`L_int` = `abs(net)` | "L_eff = L1 + fA(L2+L3)" | CIE cooling above, net below (sanctioned); `L>0` | contested | `abs()` on signed net is the residual defect |
| 14 | **Convergence check never inspected** | `fsolve(...)[0]`, no `full_output`, no `ier` | "sentinel steers fsolve away" | sentinel can manufacture a root | corroborated (A,B,C) | **defect** |
| 15 | `infodict['ier']` | never set; read anyway → `-999` | ":1053 no ier key; success from status" | — | corroborated (A,B) | **defect (diagnostic only)** |
| 16 | `bubble_E2P` form | `3(γ−1)E_b/(4π(r2³−r1³))`, `r1` retained | "Rahner pg71 Eq 6" | identical | corroborated (A,C) | **agree** |
| 17 | γ hard-coding in A12 / `get_r1` | literal `2π` ⇒ γ=5/3 | same, from the prose formula | γ-only coefficients must move or be documented | corroborated (A,B,C) | **defect** |
| 18 | `get_r1` residual | `√(L/(v E_b)·(r2³−r1³)) − r1` | endpoints only | identical, unique root, bracket `[0,R2]` | corroborated (A,B,C) | **agree** |
| 19 | `solve_R1` fallbacks | `0.0` for `L≤0`, `R2≤0/NaN`; raises for non-finite E_b | "returns 0.0 … raises on root-finding failure" | fallback to `R1=0` breaks `P_b`, sits at the transition | corroborated (A,B,C) | **defect** |
| 20 | `pRam` | `L/(2πr²v)` | `L/(2πr²v)` | `L/(2πr²v)` | corroborated (A,B,C) | **agree** |
| 21 | `get_leak_luminosity` | `γ/(γ−1)(1−C_f)4πR2²P_b c_s`, `0.0` at `C_f≥1` | identical | identical; enthalpy is the correct choice | corroborated (A,B,C) | **agree** |
| 22 | `delta2dTdt` / `dTdt2delta` | `δT/t`, `t·Ṫ/T` | equation never transcribed | `δT/t`, `t·Ṫ/T` | corroborated (A,C) | **agree** |
| 23 | `_get_init_dMdt` prefactor | `16π/25 · μC/k_B · R2 T_c^{5/2}` | "Eq 33 Weaver" | `16π/25 · μ m_H C T^{5/2} R2/k_B` (medium) | corroborated (A,C) | **agree**; `1.646` unpinned |
| 24 | **Inner BC (the eigenvalue closure)** | `v(R1) = 0` | never defined at the function | `v(R1) ≈ v_w/4` (medium) | contested (A≠C) | **open — see §5** |
| 25 | Residual normalisation | `/(v[0] + 1e-4)` | "v[-1]/v[0]" | relative to the **target** | corroborated (A,C) | **defect** |
| 26 | Radius grid refinement | logspace reflected, clustered at the **outer** end, + 2 refinements | "three logspace chunks, 60k, decreasing" | must be log-refined toward R2 | corroborated (A,B,C) | **agree — C's fear refuted** |
| 27 | Grid cleaning | mask from original diffs, rel. `1e-12` | `1e-12` is below the `1e-8..1e-9` gaps it names | must not thin the front | corroborated (A,B) | near-no-op; docstring wrong |
| 28 | `t` vs `t − tSF` in similarity terms | `params['t_now'].value` (absolute) | tSF only in `get_effective_bubble_pressure` | must be `t − tSF` | single-lens (C), evidence from A | **open/defect** |
| 29 | Unit system inside the ODE | AU (M☉,pc,Myr,K), dimensionally balanced term-by-term | "au in / cgs out, except get_dudt" | "should be wholly cgs" | contested | **preference, not defect** |
| 30 | Mass integral domain | `[R1, r2Prime]`, `ρ = n·μ_conv` | "`M(r)=∫[0→r]`" | `[R1,r2Prime]`, matched `(μ,n)` pair | corroborated (A,C) | code right, comment stale |
| 31 | Gravity outputs | `None`, discarded by caller | "DISABLED"; block restorable "verbatim" | must be attractive | corroborated (A,B) | **moot (dead)** |
| 32 | Purity of `get_bubbleproperties_pure` | writes nothing to params | claim stated 3× | — | corroborated (A,B) | **agree** |
| 33 | `δ = (2/7)(2α−β−1)` closure | — | — | must hold in the energy phase | single-lens (C) | **untested invariant** |
| 34 | Model-validity ceilings (saturation, NEI, `w≠0`) | — | — | 9 named regimes | single-lens (C) | **undocumented ceilings** |

---

## 2. Divergence table

| Div | Item | A says | B says | C says | Class | Verdict |
|---|---|---|---|---|---|---|
| **ABC** | Intermediate zone's outer limit | ends at `R2 + (2/3)dR2`, *always* | "tiny (or non-existent)" | must stop at `r2Prime < R2` | regime | **S2 · code defect + doc-drift** |
| **ABC** | dMdt convergence | never checked | sentinel "steers fsolve away … instead of falsely converging" | sentinel manufactures roots | silent-failure | **S2 · code defect; prose overclaims** |
| **ABC** | γ = 5/3 baked into A12 / `get_r1` | literal `2π` | `2π = (3/2)(4π/3)`, no γ arg | must carry γ or be documented | coefficient | **S2 · code defect** |
| **ABC** | `solve_R1` → `0.0` fallback | 2 silent paths | documented as intentional | "a fallback R1=0 … the divergence sits exactly at the phase transition" | silent-failure | **S2 · C predicted it blind** |
| **ABC** | `max(P_thermal, P_ram)` in transition | max-clamp | "ensures **smooth** handoff" | max is a **no-op** under `P_b/P_ram=(R2/R1)²≥1`; a hard switch is unneeded | regime | **S2 · selecting P_ram is a symptom of R1=0** |
| **AB** | Monotonicity guard's return | constant `+1e2`, sign+magnitude destroyed | residual "never defined where it lives"; known trigger is a **benign** boundary transient | — | numerical | **S2 · code defect, prose gap** |
| **AC** | Residual normalisation | `/(v[0] + 1e-4)` | — | relative to target, scale-free | numerical | **S2 · code defect (pole + sign flip)** |
| **AC** | Inner BC target | `v(R1) = 0` | — | `v(R1) ≈ v_w/4` (C: medium) | regime | **contested — §5, needs Weaver+77** |
| **AC** | `t` in similarity terms | `t_now` absolute | — | `t − tSF` | regime | **S3 · open** |
| **AB** | `fA` band | ODE: `T < 10^5.5` only; sum: whole of `L_cond`+`L_int` | "scales … consistently with the in-ODE boost" | — | regime | **S3 · L3 has no ODE counterpart** |
| **AB** | region 2 lower bound | `3e4 → 10^5.5` | ":752 says `1e4 → 10^5.5`" | — | citation | **S3 · doc-drift; code correct** |
| **AB** | `infodict['ier']` | key never set, read anyway | ":1053 admits there is no ier key" | — | deadcode | **S4 · dead diagnostic** |
| **AB** | grid cleaning threshold | mask from original diffs | `1e-12` below the gaps it names | must preserve the front | numerical | **S4 · near-no-op; docstring wrong** |
| **BC** | R1 geometry in prose | — | ":381 'outer solar wind'" (backwards) | free wind **inside** R1 | citation | **S4 · C arbitrates for B** |
| **BC** | Weaver eq. numbers 33/42/43/44 | — | recorded verbatim | **refuses to assert any** | citation | **open — §8** |
| **AC** | Unit system in the ODE | AU throughout, balanced | — | "should be wholly cgs" | units | **preference — dropped as a defect** |
| **AB** | ξ / `r_Tb` definition | `R1 + ξ(R2−R1)` (thickness fraction) | "relative to bubble thickness" | assumed `0.98·R2` (radius fraction) | other | **spec correction — C's test T10 mis-specified** |
| **scope-creep** | Intermediate zone as a whole | 1000-pt linear extrapolation past the ODE domain | documented as a cooling-resolution device | not in the Weaver structure at all | regime | folded into the top finding |

---

## 3. THE LUMINOSITY INTEGRAL — direct comparison

Lens C stated the correct integral **before seeing any code**; Lens A transcribed what is actually
integrated. Comparing them line by line is the highest-value output of this slice, so I state each
axis plainly, including where they agree.

### 3.1 What C specified (blind)

```
L_cool = ∫_{R1}^{r2Prime}  n_e(r) · n_H(r) · Λ(T(r))  ·  4π r² dr        [erg s⁻¹]
```
with `Λ` bolometric, `r2Prime < R2`, and the warning that `2πr dr` (plane annulus) or a bare `dr`
(line element) both fail dimensionally.

### 3.2 What A transcribed

```
L_bubble       = |∫ χ_e · n² · Λ_CIE(T) · 4π r² dr|      over [r_CIE, R1],      T ≥ 10^5.5
L_conduction   = |∫ u̇_nonCIE(n,T,φ)   · 4π r² dr|       over [r2Prime, r_CIE], 3e4 ≤ T < 10^5.5
L_intermediate = |∫ u̇_nonCIE(n,T,φ)   · 4π r² dr|       over [r2Prime, R2+(2/3)dR2], 1e4 ≤ T ≤ 3e4
L_total        = L_bubble + fA·L_conduction + fA·L_intermediate
n(r)           = P_b / ((μ_convert/μ_ion) · k_B · T(r))
```

### 3.3 Axis-by-axis verdict

**(a) Volume element — THEY AGREE.**
Every one of the three integrands carries `4π r² dr`. There is no `2πr dr` and no bare `dr` anywhere
in A's transcription of the luminosity path. Lens C's `S7-C-01` (S1) is **refuted and dropped**.
Lens A independently dimension-checked it: `[1]·L⁻⁶·(M L⁵T⁻³)·L³ = M L²T⁻³` ✓ for the CIE arm and
`M L⁻¹T⁻³·L³ = M L²T⁻³` ✓ for the tabulated arms. The `abs()` is present because the radius arrays
are descending, not because of a geometry error. Lens A also confirms the `Tavg` weighting uses `3`
with no `4π` — correct, because the `4π` cancels between `4π∫r²T dr` and `(4π/3)Δr³`.
*Verdict: the single most-feared S1 in this slice does not exist.*

**(b) Density factors — THEY AGREE STRUCTURALLY; two constants are unverified.**
C quantified the spread: relative to the correct `n_e n_H`, using `n_tot²` over-counts by
`2.3²/1.2 = 4.4`, and `n_H²` under-counts by `1.2`. A's code computes
`n = P_b/((μ_convert/μ_ion) k_B T)`, i.e. the isobaric total density `P_b/(k_B T)` **divided by**
`μ_convert/μ_ion`. If `μ_convert` is mass-per-H-nucleus and `μ_ion` mass-per-particle, that ratio is
`n_tot/n_H = 2.3`, so `n ≡ n_H`, and `χ_e·n² = (n_e/n_H)·n_H² = n_e n_H` — **exactly C's expected
integrand**, sitting at the correct point of the 4.4× spread rather than at either end. A reached the
same conclusion from a different direction: pairing `ρ = n·μ_convert` in `_get_mass_and_grav` with
this `n` gives `ρ = μ_ion P_b/(k_B T)`, a self-consistent `(μ, n)` pair — which is precisely what
C's `S7-C-08` demanded, so that item is **refuted too**.
*What is NOT in evidence:* the numeric values of `chi_e` and `mu_convert/mu_ion` (both are `.param`
entries; neither lens saw them). The finding therefore converts from "wrong by up to 4.4×" to a
**two-line verification item** (`S7-R-25`): assert `chi_e ≈ 1.2` and `mu_convert/mu_ion ≈ 2.3` for
the declared composition. `S7-C-02` demoted S1 → S3.
*Second-order gap neither lens closed:* only the CIE arm uses the explicit `χ_e n²Λ` form. The
conduction and intermediate arms take a **volumetric** `u̇` straight from the non-CIE cubes, whose
internal density convention is a property of the table, not of this code. Nothing in any lens
establishes that the two halves of `L_total` use the same convention across the `10^5.5` seam
(see `S7-R-14`).

**(c) Integration limits — THEY DIVERGE. This is the finding.**
- Lower limit: `R1`. **A = C. Agree.**
- Upper limit: C says `r2Prime`, strictly **inside** `R2`, and warns that going past it "walks into
  the singularity and into gas the shell module also owns".
  A shows the total span is `[R1, R2_coolingswitch]` with
  `R2_coolingswitch = R2 + (2/3)·dR2` — **outside `R2`, deterministically, in every ordinary run.**
  A derives this algebraically rather than observing it: because `_T_INIT_BOUNDARY = 3e4 K` exceeds
  `_coolingswitch = 1e4 K`, `index_cooling_switch ≡ 0`, so `T_array[0] = 3e4` exactly (it is the IC)
  and `dTdR = −(2/5)·3e4/dR2` exactly (also the IC), giving
  `(1e4 − 3e4)/(−1.2e4/dR2) + (R2 − dR2) = (5/3)dR2 + R2 − dR2 = R2 + (2/3)dR2`. No solver noise is
  involved; the `2/3` is exact.
- Lens B independently contradicts the code here from the prose side: region 3 is documented as
  "tiny (or non-existent) when `R2_prime` is very close to `R2`". A shows it is never non-existent
  and its thickness is `(5/3)·dR2` — i.e. **1.67× the entire conduction front**, always.

  *Reconciler arithmetic, from C's own asymptotic `R2 − r ∝ T^{5/2}`:* the true `T = 1e4 K` isotherm
  sits at `R2 − r = dR2·(1/3)^{5/2} = 0.064·dR2`, i.e. **inside** `R2` and only `0.94·dR2` outward
  of `r2Prime`. The code's linear extrapolation matches the true gradient exactly at the `3e4 K`
  anchor (it uses the IC gradient) but diverges as `T` falls, over-stating `|dr/dT|` by ~16× at
  `1e4 K`. Since the emissivity there goes as `T⁻²Λ`, the low-`T` end is exactly where the weight
  is — so `L_intermediate` is over-counted by an order-few-to-ten factor, on a band that C's own
  `dL/dlnT ∝ Λ T^{1/2}` table gives ≈3% of the front total. Net expected bias on `L_total`: tens of
  percent, one-signed (too much cooling), plus a `(2/3)dR2` slab of "bubble" volume outside `R2` in
  the `Tavg` denominator, plus potential double-counting with whatever the shell module does with
  the same `1e4 K` gas. The zone's density is the **bubble** pressure continued isobarically past
  the contact discontinuity, reaching ~3× the boundary value.
- Asymmetry worth noting: this slab's **luminosity and volume** enter `L_total` and `Tavg`, but its
  **mass** does not (`_get_mass_and_grav` integrates only `[R1, r2Prime]`). The domains of the three
  reported bulk quantities are not the same.

*Verdict: the volume element and integrand are right; **the outer limit is wrong**, by a deterministic
`(2/3)·dR2` past the contact discontinuity, and both A and C reached it independently while B's prose
asserts the opposite.* → `S7-R-01`, S2, high confidence, ABC.

**(d) Bolometric vs band-limited — NO DIVERGENCE FOUND.**
C flagged (S1) the risk that Weaver's famous `0.5–4.5 keV` `L_X` is used where the energy equation
needs bolometric losses. A transcribes the CIE arm as a **1-D interpolation in `log₁₀T` only**
returning `log₁₀Λ` in cgs, and the non-CIE arms as `10^heat − 10^cool` from 3-D `(n,T,φ)` cubes.
Nothing in A or B mentions a band, an energy range, or an X-ray label. **`S7-C-06` is dropped as
unsubstantiated** — but note this is an argument from A's and B's *silence about a band*, which is
weaker than a positive refutation. It should be closed by one grep of the cooling-table loader
(outside this slice).

**(e) Grid resolution of the emitting layer — C's fear is REFUTED.**
C's `S7-C-04`/`S7-C-22` (S1/S2) predicted that a uniform radial grid would put the entire `10⁵ K`
emitting layer between the last two nodes. A shows the grid is
`r = (r2Prime + R1) − logspace(log10(R1), log10(r2Prime), 20000)` — a logspace **reflected about the
endpoint sum**, which clusters points at the **outer** (front) end, plus two further 20000-point
refinements of the first and last intervals, ~6×10⁴ points total. Moreover C's own geometry table
places the `10⁵ K` peak at `1 − r/R2 = 1.2e-3`, while `r2Prime` sits at `5.7e-5` — the peak is
**well inside** the integration domain, not at its edge. Both items dropped. C's related `S7-C-05`
(bulk/front junction double-counting) is also refuted: A confirms the three zones tile
`[R1, R2_coolingswitch]` contiguously with no overlap and no gap.

**(f) Positivity / sign of the sum — partially divergent, see §6.**

---

## 4. Interior temperature/density profile and exponents

**A and C independently derived the same system.** This is the cleanest part of the slice and both
lenses say so unprompted.

| Element | C (derived blind) | A (transcribed) | Verdict |
|---|---|---|---|
| `dv/dr` | `(β+δ)/t − 2v/r + (v − αr/t)T′/T` | `(β+δ)/t + (v − αr/t)T′/T − 2v/r` | **identical, term for term** |
| `d²T/dr²` bracket | `(β + 5δ/2)/t + (5/2)(v−αr/t)T′/T` | `(β + 2.5δ)/t + 2.5(v−αr/t)T′/T` | **identical** |
| Expansion terms | `− (5/2)T′²/T − (2/r)T′` | `− 2.5T′²/T − 2T′/r` | **identical** |
| **Cooling sign** | `+Λ_vol/(C T^{5/2})` | `−u̇/(κC T^{5/2})` with `u̇` = heating−cooling | **identical** (`u̇ = −Λ_vol`) |
| Conduction coefficient | `C T^{5/2}` | `κ·C_thermal·T^{5/2}` (`κ = cooling_boost_kappa`) | agree; boost scales conduction and rescales the radiative term relative to it, which is self-consistent |
| Front `T(r)` | `[(25/4)(k_B/(μ m_H C))(Ṁ/4πR2²)(R2−r)]^{2/5}` | `T = (K·Ṁ·dR2/(4πR2²))^{2/5}`, `K=(25/4)k_B/(μ_ion κC)` | **identical, incl. `25/4` and `2/5`** |
| Front `dT/dr` | `−(2/5)T/(R2−r)`, strictly negative | `−(2/5)T/dR2` | **identical** |
| Front `v` | `α r/t − (Ṁ/4πr²)(k_B T/(μ m_H P_b))` | `α R2/t − (Ṁ/4πR2²)(k_B T/(μ_ion P_b))` | agree; code evaluates at `R2` not `r2Prime` — an `O(dR2/R2) ≈ 6e-5` difference, noted not filed |
| `μ` choice | mass **per particle** (`μ_ion`), C's "trap #1" | `μ_ion` throughout the BC | **agree — the trap is avoided** |
| Isobaric density | `n_H T = P_b/(2.3 k_B)` if `n_H` is stored | `n T = P_b/((μ_conv/μ_ion) k_B)` | **agree if the ratio is 2.3** |
| `Ṁ` seed prefactor | `16π/25 · μ m_H C T^{5/2} R2/k_B` (C: medium) | `16π/25 · μC/k_B · R2 · T_c^{5/2}` (A re-derived `12/75·1.646^{5/2}·4π = 16π/25`) | **agree** — a medium-confidence C claim confirmed by A's independent algebra |

**Exponents: no divergence anywhere.** `2/5`, `−2/5`, `5/2`, `7/2`, `2/7`, `5/7` all match. C's
warning that "exponent-correct is not coefficient-correct" (enthalpy `5/2` vs internal energy `3/2`
⇒ `25/4` vs `15/4` ⇒ a silent 19% error in `T(r2Prime)`) is **discharged**: A shows the code uses
`25/4`, the enthalpy value.

**Two residual gaps:**
- The `1.646` prefactor in `T_c = 1.646(P_b R2²/(Ct))^{2/7}` is a Weaver similarity number C
  explicitly declined to pin. **Open question, not a defect** (§8).
- C's `S7-C-12` (the two `_rhs` at L337 and L490 must be term-identical) is **refuted**: A shows a
  single `_get_bubble_ODE` at L414 with two thin wrapper closures that both call it and both latch
  `rhs_error`. Dropped.

---

## 5. The inner boundary condition — the one contested physics claim

This is the item I would most want settled, and I cannot settle it from the reports.

- **A:** the residual is `(v[-1] − 0)/(v[0] + 1e-4)`; the enforced condition is **`v(R1) = 0`**.
- **C:** the closure must be the strong-shock jump, `v(R1) = Ṙ1 + (v_w − Ṙ1)/4 ≈ v_w/4` — hundreds
  of km/s, not zero — with `T(R1) = 3μm_H v_w²/(16k_B) = 1.38e7 K (v_w/1000 km/s)²`.
- **B:** silent. `_get_velocity_residuals`' docstring is one line; B explicitly files this as a gap
  ("the velocity residual is never defined where it lives"), so the prose cannot arbitrate.

**Why I will not call this a defect.** C rated its own claim **medium**, and named the alternative
in the same breath: "medium that a velocity residual at `R1` is *the* right closure rather than an
equivalent mass-flux statement". `v(R1) = 0` is the natural inner condition in the standard Weaver
limit where the wind mass flux is negligible against the evaporated mass and the inward evaporative
flow stagnates at the termination shock; `v(R1) = v_w/4` is the natural condition if the wind mass
flux is retained. Both are defensible from the reports alone, and the two lenses are looking at
different idealisations, not necessarily at a bug.

**Why it matters enough to rank second.** C's closed form gives `L_front ∝ P_b² R2⁴ C / Ṁ` — the
dominant luminosity term is *inversely* proportional to the evaporation eigenvalue. The inner BC is
what fixes `Ṁ`. A factor-2 error in the closure is a factor-2 error in the headline number that
drives the energy→momentum transition.

**Two things that are defects regardless of which target is right:**
1. The normalisation `/(v[0] + 1e-4)` (A) is not the target scale (C wants
   `(v_solved(R1) − v_target)/v_target`). A shows `v[0]` decreases monotonically with `Ṁ` and does
   cross zero, so `v[0] = −1e-4 pc/Myr` is a **pole with a sign flip** carrying no physical content.
   → `S7-R-09`.
2. Neither the target nor the normalisation is documented anywhere (B's flag L). → folded into
   `S7-R-04`.

---

## 6. Sign conventions where luminosity components are summed

- `L_bubble` integrates `χ_e n² Λ_CIE` — **strictly positive, pure cooling**, no heating term.
- `L_conduction` and the non-CIE arm of `L_intermediate` integrate the **net** `(heating − cooling)`
  from the non-CIE cubes and then take `np.abs()`.
- All three are summed at line 851 and fed to the energy budget as a loss.

**C partially sanctions the mixed convention:** `S7-C-06` expects "bolometric CIE (`T>10^5.5 K`) plus
the non-CIE net cooling-minus-heating table below". So *cooling-only above the seam and net below* is
the intended design, and A's framing of it as "sibling terms use opposite conventions" is too broad.
I have narrowed the finding accordingly.

**What survives is the `abs()`.** A net-heating sub-region cancels against cooling *inside* the
integral (correct), but if the *total* for a zone comes out net-heating, `abs()` re-signs an energy
**source** into an equally large energy **sink**. C's `S7-C-07` requires `L_cool > 0` as an *outcome*,
not as an enforcement. Neither B nor C corroborates the `abs()` observation, so it stays
`single-lens`, S3, medium. → `S7-R-12`.

**The failure-sentinel signs are a separate, corroborated problem.** A maps them: `+1e3` (three
solver-failure paths), `+1e2` (non-monotonic), `−1e3` (NaN `T`). Two adjacent failure regions of
**opposite sign** manufacture a spurious sign change. C predicted exactly this blind
(`S7-C-19`: "a large constant substituted for a failed evaluation creates an artificial sign change
adjacent to the failure region"). B's prose claims the opposite design intent ("large and non-zero
so fsolve is steered away … instead of falsely converging on a garbage (~0) residual") — which is
true of the *magnitude* and false of the *sign*. → folded into `S7-R-03`.

---

## 7. The monotonicity guard, and the convergence checks nobody inspects

### 7.1 The monotonicity guard

| Question | Answer | Source |
|---|---|---|
| What does it test? | `operations.monotonic(T_array)` — the **temperature** array | A |
| Which solution? | the **dMdt-search** solve: 500 points, `rtol = 1e-6` | A |
| What does it return? | the bare constant `+1e2` — sign and magnitude both discarded | A |
| Is the used solution tested? | **No.** The production profile (~6×10⁴ points, `rtol = 1e-8`) is checked only for `T_array < 0` — which does **not** catch NaN | A |
| Is it reached on every path? | **No.** The `min_T < 3e4` branch returns first, with a sign-**preserving** multiplicative penalty `(3e4/(min_T+0.1))²` — so the two guards have incompatible composition rules | A |
| Is any of this documented? | No. The residual is defined only as a parenthetical in a *constant's* comment | B (flag L) |
| Does C sanction guarding `T`? | Yes — and C warns specifically against guarding **`v`**, which "is NOT monotonic and must not be constrained" | C |

**Three conclusions.**
1. **C's `S7-C-28` is refuted and dropped.** The guard tests `T`, which C explicitly endorses. The
   feared "monotonicity guard on `v` rejects valid solutions" does not exist.
2. **A + B together make `S7-R-04` the strongest defect in the slice.** A shows the response is a
   hard constant `+1e2`. B independently documents that the *known* trigger is a
   `"boundary_transient"` — "a small smooth dip at the `T_init = 3e4` outer edge, confined to the
   first ~0.1% of points; the bulk is monotonic". So the guard's dominant real-world trigger is a
   **benign numerical artefact at the boundary**, and the response is to flatten the residual into a
   constant plateau that destroys the root-finder's gradient and its bracket. Neither lens could
   have concluded this alone.
3. The guard is applied to a solution that is **not** the one whose numbers are reported. C's
   `S7-C-20` guessed a related but different mechanism (a stale cached profile from the last
   residual evaluation) — **refuted**, because A shows the ICs are explicitly recomputed at the
   converged `dMdt`. The real gap is the tolerance/grid mismatch. → `S7-R-16`.

### 7.2 Convergence checks that are never inspected

| Check | Present? | Inspected? | Source |
|---|---|---|---|
| `fsolve` `ier` | not requested (`full_output` omitted) | **never** — `fsolve(...)[0]` | A |
| `fsolve` final residual magnitude | available | **never** compared to any bar | A |
| `solve_ivp` `sol.success` (search solve) | yes | yes → `+1e3` | A |
| `solve_ivp` `sol.success` (production solve) | yes | yes → `BubbleSolverError` | A |
| `infodict['ier']` | **key is never created** | read by the diagnostic → always `None` → stored as `-999` | A + B (`:1053` already admits it) |
| `v(R1) = 0` on the production profile | — | **never re-checked** | A |
| monotonicity on the production profile | — | **never checked** | A |
| `T_array` NaN on the production profile | `np.any(T < 0)` only | NaN passes | A |

Because the residual returns constants (`+1e3`, `+1e2`, `−1e3`) that can never be zero, a stalled
`fsolve` terminates on "not making good progress" and **the last iterate is silently accepted**.
B's prose asserts the sentinel design prevents exactly this. It does not: it prevents a *false zero*,
not a *silent stall*. → `S7-R-03`, ABC-corroborated, high confidence.

---

## 8. What the actual Weaver+77 paper or the Rahner thesis would settle

Lens C refused to assert Weaver prefactors and equation numbers. These are **open questions, not
defects**, and each needs one page of the literature:

1. **Weaver+77 Eq. 33** — is it the `Ṁ` estimate? Would settle the `1.646` prefactor in
   `T_c = 1.646(P_b R2²/(Ct))^{2/7}`, which nothing in this slice pins. (B records the citation; C
   refused it; A confirmed only the surrounding `16π/25` algebra.)
2. **Weaver+77 Eqs. 42–43** — are these the interior structure ODEs? A and C independently derived
   an identical system, so the physics is corroborated; only the *citation* is unverified.
3. **Weaver+77 Eq. 44** — are these the boundary conditions? Would confirm the `25/4` coefficient
   (C: medium) and, critically, **which inner BC Weaver uses** — `v(R1)=0` (A) or the strong-shock
   `v_w/4` (C). This single lookup resolves the top contested item, §5.
4. **Rahner thesis pg 80 Eq A12** — does it carry `γ` explicitly, or is the `2π` genuinely a γ=5/3
   substitution? Settles whether `S7-R-05` is a code defect or a faithful transcription of a
   γ=5/3-only equation that merely needs a documented assertion.
5. **Rahner thesis pg 71 Eq 6** and **eq 1.25** — B notes the thesis is cited in *three different
   numbering styles* ("pg 79 Eq A5", "pg 80 Eq A12", "pg71 Eq 6", "eq 1.25"); worth checking that is
   not a transcription slip.
6. **"the leakage spec, Eq. (leak)"** — B could locate no such document. A and C agree on the leak
   formula, so this is a provenance gap, not a physics gap.
7. **Weaver's hard-coded prefactors** `T_b = 1.51e6`, `n_b = 4.02e-3` — C's own arithmetic narrows
   the reported `4.2×` mismatch to `1.8×` (or `1.34×` with `T_b = 2.07e6`) once composition
   conventions are fixed, but it does not close. Neither A nor B saw these constants in this slice,
   so they may live elsewhere.

---

## 9. Dropped or demoted, with reasons

**Dropped as refuted (the other lens's account plainly explains why the concern is unfounded):**

| Dropped | Filed by | Why |
|---|---|---|
| `S7-C-01` volume element (S1) | C | A: `4πr²dr` in all three integrands, dimension-checked |
| `S7-C-04`, `S7-C-22` uniform grid discards the front (S1/S2) | C | A: grid is log-clustered at the **outer** end + 2 refinements; and C's own table puts the `10⁵ K` peak *inside* `r2Prime` |
| `S7-C-05` bulk/front junction double-count | C | A: three zones tile contiguously, no overlap, no gap |
| `S7-C-06` band-limited `Λ` (S1) | C | A: 1-D `Λ(log₁₀T)` + net `(n,T,φ)` cubes; no band anywhere in A or B |
| `S7-C-08` `μ`/`n` mismatch in the mass integral (S1) | C | A: `ρ = n·μ_conv` with `n = P/((μ_conv/μ_ion)kT)` ⇒ `ρ = μ_ion P/(kT)` — a matched pair. Folded into the constants check `S7-R-25` |
| `S7-C-09` gravity sign | C | A + B: gravity returns `None` and the caller discards it — moot |
| `S7-C-12` two `_rhs` must match | C | A: one `_get_bubble_ODE`; the two `_rhs` are wrappers |
| `S7-C-20` profile cached from the last residual eval | C | A: ICs are recomputed at the converged `dMdt`. The tolerance-mismatch residue kept as `S7-R-16` |
| `S7-C-23` cleaning thins the front | C | B's arithmetic: the `1e-12` relative cut is *below* the `1e-8..1e-9` gaps, so cleaning removes ~nothing |
| `S7-C-26` `ξ=0.98` enters the free wind | C | A **and** B: `r_Tb = R1 + ξ(R2−R1)` is a **thickness** fraction, not `0.98·R2`. Also corrects C's test T10 as specified |
| `S7-C-28` monotonic guard on `v` | C | A: the guard tests `T`, which C endorses |
| `S7-C-37` leak flux `5/2` vs `3/2` | C | A: the code uses `γ/(γ−1)` = enthalpy — the choice C itself calls physically correct |
| `S7-C-13`/`-29`/`-30` "must be cgs" | C | A: the ODE is AU throughout and dimensionally balanced term-by-term. A preference, not a defect. **Only the `t` vs `t−tSF` half survives** → `S7-R-13` |
| `S7-B-02` `abs()` missing on the `Tavg` numerator | B | A: `abs()` is applied to **each** numerator term individually |
| `S7-B-03` region 1 `[au]` vs region 2 `[cgs]` | B | A: both convert into AU (`Lambda_cgs2au`, `dudt_cgs2au`) before summation |
| `S7-B-19` `STATE_DT` additive vs ratio | B | A: it **is** a ratio test, so default `1.0` really is "no spacing". Wording only |
| `S7-B-22` purity contract violated | B | A: `get_bubbleproperties_pure` writes nothing into `params` |
| `S7-B-23` region 1 queries two cooling curves | B | A: only `cStruc_cooling_CIE_interpolation` is used in region 1. Copy-pasted comment |
| `S7-B-20` numpy-2 shims under a `numpy<2` pin | B | CLAUDE.md itself records numpy 2.0/2.3 passing; the shims are not obviously dead. Noise |

**Demoted:**

| Item | Was | Now | Why |
|---|---|---|---|
| `S7-C-02` density convention | S1 | S3 | A shows the code's `χ_e n²` **is** `n_e n_H` given the expected constants; converts to a value check |
| `S7-B-01` region 2 lower bound `1e4` | S2 | S3 | A: the code's region 2 really is `3e4 → 10^5.5`. Doc-drift, code correct |
| `S7-A-19` unguarded `/dTdR_coolingswitch` | S2 | S3 | A's own §2.9(e) shows the divisor is deterministically the negative IC gradient in every ordinary run; the trigger needs the empty-mask fallback |
| `S7-A-11` "sibling terms use opposite conventions" | S2 | S3 | C sanctions CIE-cooling-above / net-below; only the `abs()` survives |
| `S7-C-24` `T_init` must be ≤ table floor | S2 | S3 | C's own `dL/dlnT` table gives the `1e4–3e4 K` band ≈3%; and the code does attempt to recover it. Same fix as `S7-R-01` |
| `S7-B-06` fd redirect | S3 | S3 (narrowed) | A confirms restoration in `finally`, so B's "not exception-safe" half is refuted; the process-wide **scope** stands |

**Gap noted:** Lens A's JSON skips `S7-A-24` — no such candidate exists in its array. Nothing to
reconcile; recorded so a future reader does not hunt for it.

---

## 10. Merged ranked list

Ranked S1 → S4, then by corroboration and impact. **No item reached S1 after reconciliation** — the
three S1 candidates aimed at the luminosity integral (volume element, band-limited `Λ`, density
convention) were refuted or reduced to a constants check, and the remaining S1-scale candidate (the
inner boundary condition) is contested and turns on a page of Weaver+77.

```json
[
  {
    "id": "S7-R-01",
    "file": "trinity/bubble_structure/bubble_luminosity.py",
    "line": 801,
    "class": "regime",
    "severity": "S2",
    "claim": "The 'intermediate' luminosity zone is a linear extrapolation of the temperature profile that deterministically ends at R2 + (2/3)*dR2, i.e. OUTSIDE the contact discontinuity, and its luminosity and volume are both counted in L_total and bubble_Tavg with the bubble pressure continued isobarically past R2.",
    "evidence": "A derives R2_coolingswitch = (1e4 - T_array[ics])/dTdR + r_array[ics] with ics == 0 always (because _T_INIT_BOUNDARY = 3e4 > _coolingswitch = 1e4), T_array[0] = 3e4 exactly (it is the IC) and dTdR = -(2/5)*3e4/dR2 exactly (also the IC), giving (5/3)dR2 + R2 - dR2 = R2 + (2/3)dR2 with no solver noise. C specifies the upper limit must be r2Prime < R2 and warns that going past it enters 'gas the shell module also owns'. B's prose says the opposite of A: region 3 is 'tiny (or non-existent) when R2_prime is very close to R2'. n_interm = Pb/((mu_conv/mu_ion) k_B T_interm) reaches ~3x the boundary density.",
    "expected": "Upper limit r2Prime (or a clamp R2_coolingswitch <= R2). If the 3e4->1e4 K layer is wanted, place it with the front asymptotic R2 - r ∝ T^(5/2): the true 1e4 K isotherm sits at R2 - 0.064*dR2, INSIDE R2, not at R2 + 0.667*dR2.",
    "failure_scenario": "L_total carries an over-thick, over-dense 1e4-3e4 K slab whose emission is over-counted (the linear profile over-states |dr/dT| by ~16x at the 1e4 K end, where the T^-2 emissivity weight is largest) and whose volume inflates the Tavg denominator; the same gas may be counted again by the shell module. One-signed excess cooling shifts the energy->momentum transition earlier.",
    "repro": "Print R2_coolingswitch, params['R2'].value, r2Prime and dR2 each call; (R2_cs - R2)/dR2 should equal 2/3 to solver accuracy. Then compare L_intermediate against the same band placed with R2 - r = dR2*(T/3e4)**2.5.",
    "confidence": "high",
    "lenses": ["A", "B", "C"],
    "divergence": "ABC",
    "status": "corroborated",
    "source_ids": ["S7-A-13", "S7-B-01", "S7-C-03", "S7-C-24"]
  },
  {
    "id": "S7-R-02",
    "file": "trinity/bubble_structure/bubble_luminosity.py",
    "line": 368,
    "class": "regime",
    "severity": "S2",
    "claim": "CONTESTED: the dMdt eigenvalue is closed on v(R1) = 0, while the physics lens derives the inner condition as the strong-shock jump v(R1) = Rdot1 + (v_w - Rdot1)/4 ~ v_w/4.",
    "evidence": "A: residual = (v_array[-1] - 0)/(v_array[0] + 1e-4), i.e. the target is v(R1) = 0. C (own confidence: medium): Rankine-Hugoniot at gamma=5/3 gives v(R1) ~ v_w/4 with T(R1) = 3 mu m_H v_w^2/(16 k_B) = 1.38e7 K (v_w/1000 km/s)^2, and C explicitly hedges that 'a velocity residual at R1' may be replaced by 'an equivalent mass-flux statement'. B is silent: the residual is never defined at its own function.",
    "expected": "Whichever closure Weaver+77 Eq 44 uses, stated explicitly at _get_velocity_residuals. v(R1)=0 is defensible in the limit where the wind mass flux is negligible against evaporation; v_w/4 is required if the wind mass flux is retained.",
    "failure_scenario": "If the target is wrong, the converged Mdot is wrong, and C's closed form gives L_front ∝ 1/Mdot — so the dominant cooling term and hence the energy->momentum transition time move by the same factor.",
    "repro": "Log v(R1) and v_w/4 on a converged energy-phase step; check whether the solved profile's inner velocity is anywhere near v_w/4.",
    "confidence": "low",
    "lenses": ["A", "C"],
    "divergence": "AC",
    "status": "contested",
    "source_ids": ["S7-C-17", "S7-C-18"]
  },
  {
    "id": "S7-R-03",
    "file": "trinity/bubble_structure/bubble_luminosity.py",
    "line": 261,
    "class": "silent-failure",
    "severity": "S2",
    "claim": "The dMdt root-find never checks convergence, and the residual's failure sentinels have mixed signs (+1e3, +1e2, -1e3) that can manufacture a spurious sign change; a stalled or artefactual fsolve result is used to build every reported quantity.",
    "evidence": "A: fsolve(...)[0] is called without full_output and ier/mesg are never inspected; the residual returns constants +1e3 (three solver-failure paths), +1e2 (non-monotonic) and -1e3 (NaN T), none of which can be zero, so fsolve terminates on 'not making good progress' and the last iterate is accepted. C predicted this blind (S7-C-19): 'a large constant substituted for a failed evaluation creates an artificial sign change adjacent to the failure region'. B's prose asserts the opposite design intent: the sentinel is 'large and non-zero so fsolve is steered away from the infeasible dMdt instead of falsely converging' — true of the magnitude, false of the sign, and it does not prevent a silent stall.",
    "expected": "Request full_output, require ier == 1 and |residual| below a stated bar, raise BubbleSolverError otherwise; use one sign for all failure sentinels, or signal failure out of band.",
    "failure_scenario": "A stiff regime parks the residual on the +1e2 plateau; fsolve stalls at the initial guess and the run continues with an evaporation rate that does not satisfy its own boundary condition, with no warning. Or a +1e3/-1e3 adjacency gives a 'root' that is purely an artefact of two neighbouring failure modes.",
    "repro": "Wrap the residual to log (dMdt, return value) across an fsolve call on docs/dev/performance/f1edge_hidens*.param and count returns of exactly 100.0, 1000.0 and -1000.0.",
    "confidence": "high",
    "lenses": ["A", "B", "C"],
    "divergence": "ABC",
    "status": "corroborated",
    "source_ids": ["S7-A-06", "S7-A-08", "S7-C-19"]
  },
  {
    "id": "S7-R-04",
    "file": "trinity/bubble_structure/bubble_luminosity.py",
    "line": 380,
    "class": "numerical",
    "severity": "S2",
    "claim": "The monotonicity guard replaces the residual with the bare constant +1e2 — destroying both sign and magnitude — and its documented dominant trigger is a benign boundary transient, not a real failure.",
    "evidence": "A: `if not operations.monotonic(T_array): return 1e2`, where T_array comes from the 500-point, rtol=1e-6 search solve. Unlike the min_T branch two lines earlier (which returns a sign-PRESERVING multiplicative penalty residual*(3e4/(min_T+1e-1))**2), this one is a replacement. B independently documents the guard's two known triggers, one of which is 'boundary_transient — a small smooth dip at the T_init=3e4 outer edge, confined to the first ~0.1% of points; the bulk is monotonic'. B also flags that the residual and the composition of its penalties are never defined at the function. C sanctions guarding T (and warns only against guarding v).",
    "expected": "A sign-preserving penalty consistent with the min_T branch, so the root stays bracketed; and a tolerance on the boundary transient so a 0.1%-of-points dip at the anchor does not veto an otherwise-monotonic profile.",
    "failure_scenario": "A region of dMdt space whose true residual is negative returns +1e2; the root-finder sees a flat plateau with no sign change, stalls, and (per S7-R-03) the stalled value is accepted. Because the trigger is a boundary artefact, this fires on physically fine profiles.",
    "repro": "Instrument _get_velocity_residuals to record (dMdt, return value) over a sweep and plot; the plateau at exactly 100.0 is a flat band. Cross-check against the TRINITY_BUBBLE_DIAG mode label.",
    "confidence": "high",
    "lenses": ["A", "B", "C"],
    "divergence": "AB",
    "status": "corroborated",
    "source_ids": ["S7-A-07", "S7-B-12"]
  },
  {
    "id": "S7-R-05",
    "file": "trinity/bubble_structure/get_bubbleParams.py",
    "line": 130,
    "class": "coefficient",
    "severity": "S2",
    "claim": "cool_beta_to_Ebdot, Ebdot_to_cool_beta and the get_r1 endpoint algebra hard-code gamma = 5/3 through the literal 2*np.pi, while bubble_E2P, get_leak_luminosity and get_effective_bubble_pressure all take gamma as a live parameter.",
    "evidence": "A re-derived the A12 expression independently by differentiating Eb = 2*pi*Pb*d and eliminating dR1/dt via R1^2*Eb = (pdot/2)*d, matching the code term for term including a non-obvious cancellation — and shows 2*pi = 4*pi/(3(gamma-1)) only at gamma = 5/3. B reached the same conclusion from the prose alone: 'the leading 2*pi is exactly (3/2)*(4*pi/3)'. C's trap 4 requires gamma-only coefficients either to move with gamma or to be documented as gamma=5/3-only, and C-35 confirms bubble_E2P's (gamma-1) form is correct.",
    "expected": "2*np.pi -> 4*np.pi/(3*(gamma-1)) with gamma read from params, or an explicit assertion gamma_adia == 5/3 covering A12 and get_r1.",
    "failure_scenario": "A run with gamma_adia != 5/3 makes the Eb<->beta conversions and the R1 pressure balance inconsistent with the pressure used everywhere else, by the smooth factor 3(gamma-1)/2; nothing crashes and the trajectory is silently wrong.",
    "repro": "Set gamma_adia = 1.4 and round-trip Ebdot_to_cool_beta(bubble_E2P(Eb,R2,R1,1.4), R1, Edot, p) through cool_beta_to_Ebdot; the round trip closes only at gamma = 5/3.",
    "confidence": "high",
    "lenses": ["A", "B", "C"],
    "divergence": "ABC",
    "status": "corroborated",
    "source_ids": ["S7-A-02", "S7-B-08"]
  },
  {
    "id": "S7-R-06",
    "file": "trinity/bubble_structure/get_bubbleParams.py",
    "line": 437,
    "class": "silent-failure",
    "severity": "S2",
    "claim": "solve_R1 silently returns 0.0 for Lmech_total <= 0 and for R2 <= 0 or NaN (while raising for a non-finite Eb), and get_r1's Ebubble floor maps a collapsing or negative Eb to +1e-30 which drives R1 -> R2; the two fallbacks then feed bubble_E2P's 1e-13 shell-volume floor or np.log10(0) in the grid builder.",
    "evidence": "A: `if Lmech_total <= 0: return 0.0` / `if not (R2 > 0): return 0.0` (also swallows NaN) then `raise ValueError` for non-finite Eb; get_r1 clamps `if Ebubble < 1e-30: Ebubble = 1e-30`; bubble_E2P replaces shell_volume <= 0 with 1e-13*r2**3, inflating Pb by ~1e13 across an infinitesimal change in r2; _create_radius_grid takes np.log10(R1), and np.linspace from -inf yields an all-NaN grid. C predicted precisely this chain blind (S7-C-40): 'if it is caught and papered over with a fallback (e.g. R1=0 or R1=R2), P_b either loses the R1 correction entirely or diverges as V_b->0 — and the divergence sits exactly at the phase transition.' B documents the fallbacks as intentional and additionally claims the floor is 'bit-identical on every physical bubble', which cannot hold for 0 < shell_volume < floor.",
    "expected": "Consistent handling: either all invalid inputs raise, or R1 = 0 is validated before the log grid is built; and the E2P degenerate branch should signal rather than return a ~1e13-inflated pressure.",
    "failure_scenario": "A timestep after the last SN (Lmech_total <= 0) returns R1 = 0.0, the radius grid becomes all-NaN and the solve fails with a message unrelated to the cause. Or, as Eb collapses, R1 -> R2 crosses into r2 <= r1 and Pb jumps ~13 orders of magnitude into solve_R1, the ODE density and the whole luminosity integral with no diagnostic.",
    "repro": "solve_R1(R2=1.0, Eb=1e4, Lmech_total=0.0, v_mech_total=1e3) -> 0.0, then _create_radius_grid(0.0, 0.9). Separately bubble_E2P(1.0, 1.0, 1.0, 5/3) vs bubble_E2P(1.0, 1.0+1e-6, 1.0, 5/3).",
    "confidence": "high",
    "lenses": ["A", "B", "C"],
    "divergence": "ABC",
    "status": "corroborated",
    "source_ids": ["S7-A-01", "S7-A-23", "S7-B-07", "S7-C-40"]
  },
  {
    "id": "S7-R-07",
    "file": "trinity/bubble_structure/get_bubbleParams.py",
    "line": 353,
    "class": "regime",
    "severity": "S2",
    "claim": "The transition-phase max(P_thermal, P_ram) should be a no-op under the derived invariant P_b/P_ram(R2) = (R2/R1)^2 >= 1; it can only select P_ram when R1 has been zeroed, which two separate code paths do.",
    "evidence": "C derives (high confidence) that with the ram-balance R1 and V_b = (4pi/3)(R2^3 - R1^3), P_b = L_w/(2 pi R1^2 v_w) and P_ram(R2) = L_w/(2 pi R2^2 v_w), so max() is identically P_b — 'if such a max() is ever observed to select P_ram, then either R1 is not the ram-balance radius or V_b was computed with R1 dropped.' A confirms R1 IS the ram-balance root (get_r1) and V_b DOES retain r1 (bubble_E2P) — but also shows two paths that set R1 to 0 or near-0: solve_R1's silent 0.0 fallbacks, and the switch-on ramp R1_tmp = (t - tSF)/1e-3 * R1. B documents the transition branch only as a body comment and calls max() a 'smooth handoff'.",
    "expected": "Either drop the max() (the invariant makes it redundant) or instrument it: any selection of P_ram is a live assertion failure pointing at R1. max() is C0 but not C1, so 'smooth' should read 'continuous'.",
    "failure_scenario": "With R1 zeroed, P_b loses the (R2/R1)^2 boost, max() silently substitutes P_ram, and the driving pressure takes a derivative kink the physics does not require — an unphysical impulse in the shell EOM exactly at the phase boundary the code is built to predict.",
    "repro": "Log P_b/P_ram(R2) and which branch max() selected at every transition-phase step; the ratio must never drop below 1.",
    "confidence": "medium",
    "lenses": ["A", "B", "C"],
    "divergence": "ABC",
    "status": "corroborated",
    "source_ids": ["S7-C-39", "S7-B-14", "S7-A-27"]
  },
  {
    "id": "S7-R-08",
    "file": "trinity/bubble_structure/bubble_luminosity.py",
    "line": 409,
    "class": "divergence",
    "severity": "S2",
    "claim": "r2_prime = R2 - dR2 has no positivity or ordering guard: dMdt -> 0 sends dR2 -> inf and r2_prime -> -inf, dMdt < 0 sends r2_prime > R2, and the finiteness screen checks only [v, T, dTdr] — all of which stay finite in both limits.",
    "evidence": "A: dR2 = T_init**(5/2)/(K*dMdt/(4*pi*R2**2)) diverges as dMdt -> 0 while v -> cool_alpha*R2/t, T = 3e4 and dTdr -> -0 all remain finite; r2Prime then reaches solve_ivp's t_span, np.linspace and np.log10 in _create_radius_grid, producing an all-NaN grid or a non-BubbleSolverError exception that escapes get_bubbleproperties_pure. fsolve is unconstrained with epsfcn=1e-4, so it probes small and negative dMdt during the Jacobian step. C's S7-C-07 requires that a failed profile solve must not yield a NaN luminosity consumed by the energy equation; A separately notes np.any(T_array < 0) does not catch NaN.",
    "expected": "Reject or clamp dMdt so 0 < r2_prime and R1 < r2_prime < R2 before building the grid; include r2Prime in the finiteness screen; test for NaN as well as negative T on the production profile.",
    "failure_scenario": "The Jacobian probe alone can produce a t_span of (-inf, R1); either the run dies with a scipy error unrelated to the cause, or NaN propagates into L_total and Eb.",
    "repro": "_get_velocity_residuals(np.array([1e-30]), params, Pb, R1) from a normal simple_cluster state.",
    "confidence": "medium",
    "lenses": ["A", "C"],
    "divergence": "AC",
    "status": "single-lens",
    "source_ids": ["S7-A-05", "S7-C-07"]
  },
  {
    "id": "S7-R-09",
    "file": "trinity/bubble_structure/bubble_luminosity.py",
    "line": 368,
    "class": "numerical",
    "severity": "S2",
    "claim": "The velocity residual is normalised by (v_array[0] + 1e-4) rather than by the target scale, introducing a pole and a sign flip at v(r2Prime) = -1e-4 pc/Myr.",
    "evidence": "A: `residual = (v_array[-1] - 0)/(v_array[0] + 1e-4)`, where v_array[0] = cool_alpha*R2/t - dMdt*k_B*T/(4*pi*R2**2*mu_ion*Pb) decreases monotonically with dMdt and does cross zero. C independently requires the residual to be relative to the TARGET and scale-free: 'residual = (v_solved(R1) - v_target)/v_target', noting v(R1) ~ 250 km/s versus v(r2Prime) ~ 10 km/s so a tolerance tuned to one is meaningless for the other.",
    "expected": "Normalise by a strictly positive target scale, e.g. |v_target| or cool_alpha*R2/t_now — never by a quantity that passes through zero.",
    "failure_scenario": "For dMdt large enough that the evaporative inflow exceeds the co-moving term, v(r2Prime) passes through -1e-4, the residual diverges and flips sign, and the root-finder is attracted to the pole rather than to the boundary condition.",
    "repro": "Sweep dMdt so that v_array[0] crosses -1e-4 and plot the residual.",
    "confidence": "medium",
    "lenses": ["A", "C"],
    "divergence": "AC",
    "status": "corroborated",
    "source_ids": ["S7-A-09", "S7-C-18"]
  },
  {
    "id": "S7-R-10",
    "file": "trinity/bubble_structure/bubble_luminosity.py",
    "line": 339,
    "class": "numerical",
    "severity": "S2",
    "claim": "The rhs_error latch is write-once and never reset, so a single REJECTED LSODA trial evaluation that momentarily reaches |T| < 1e-5 permanently aborts an otherwise-healthy integration.",
    "evidence": "A: both _rhs wrappers (lines 337-345 and 490-498) do `if rhs_error is not None: return np.zeros_like(y)`; the flag is set by the BubbleSolverError raised whenever `np.abs(T - 0) < 1e-5` in _get_bubble_ODE. LSODA evaluates the RHS at predictor/trial states it may itself reject, and the latch does not distinguish accepted from rejected evaluations. B independently characterises this as 'a stiff but finite regime, NOT an overflow' where the solve 'still SUCCEEDS' — i.e. the prose expects trial excursions to be survivable. A also notes the T guard is two-sided on |T|, so a NEGATIVE T with |T| > 1e-5 passes through into T**(5/2) and yields NaN.",
    "expected": "Return a large/penalised derivative or NaN so the solver rejects the step, instead of latching; and test T > 0 rather than |T| > 1e-5.",
    "failure_scenario": "A stiff trial step near the cold boundary overshoots to T ~ 0 in an evaluation LSODA would have rejected; the RHS then returns zeros for the rest of the march, the state freezes, and the whole solve is declared failed.",
    "repro": "Log every RHS call in a stiff run and compare the r at which rhs_error latches against sol.t; a latch at an r not present in sol.t proves it was a rejected trial step.",
    "confidence": "medium",
    "lenses": ["A"],
    "divergence": "none",
    "status": "single-lens",
    "source_ids": ["S7-A-10"]
  },
  {
    "id": "S7-R-11",
    "file": "trinity/bubble_structure/bubble_luminosity.py",
    "line": 724,
    "class": "divergence",
    "severity": "S2",
    "claim": "The brentq that locates the 10^5.5 K crossing is called with no bracket check; a bubble whose interior never reaches 3.16e5 K raises a raw scipy ValueError out of the whole calculation.",
    "evidence": "A: fT_interp is built on T_array[:index_CIE_switch + 20] - _CIEswitch and passed to brentq(min(r_interp), max(r_interp), xtol=1e-8) with no try/except and no endpoint sign check. What find_nearest_higher returns when no element exceeds 10^5.5 is outside the slice. C independently establishes that this regime is reachable and is exactly where the model breaks down (S7-C-43: once the front's radiative loss approaches the conductive enthalpy flux the interior goes radiative, and L_front ∝ P_b^2 ∝ n0^(6/5) so it fails first at high ambient density).",
    "expected": "Check T_array[-1] >= _CIEswitch (or that fT_interp changes sign) before calling brentq, and fall back to the no-CIE-zone path; raise BubbleSolverError rather than letting scipy's ValueError escape.",
    "failure_scenario": "A young or heavily-cooled bubble aborts the timestep with an exception unrelated to the physical cause, in precisely the high-density regime where the Weaver solution is expected to fail.",
    "repro": "Construct a state with low Eb / high density so max(T_array) < 10**5.5 and call _bubble_luminosity. Confirming the exact trigger needs operations.find_nearest_higher's contract.",
    "confidence": "medium",
    "lenses": ["A", "C"],
    "divergence": "none",
    "status": "single-lens",
    "source_ids": ["S7-A-17", "S7-C-43"]
  },
  {
    "id": "S7-R-12",
    "file": "trinity/bubble_structure/bubble_luminosity.py",
    "line": 795,
    "class": "sign",
    "severity": "S3",
    "claim": "np.abs() is applied to the integral of a SIGNED net (heating - cooling) rate, so a zone that is net-heating is re-signed into an equally large energy sink before being summed into L_total.",
    "evidence": "A: L_bubble integrates chi_e*n**2*Lambda (Lambda > 0, pure loss) while L_conduction and the non-CIE arm of L_intermediate integrate (heat - cool)*dudt_cgs2au and then take np.abs(); all three are summed. C's S7-C-06 SANCTIONS the split convention (CIE cooling above 10^5.5, net cooling-minus-heating below), so the mixed convention itself is intended — what is not sanctioned is forcing the sign. C's S7-C-07 requires L_cool > 0 as an outcome, not as an enforcement.",
    "expected": "Let the signed net stand (the trapezoid over a descending grid can be sign-corrected by ordering, not by abs), and surface a net-heating zone rather than converting it to a loss.",
    "failure_scenario": "In a strongly photo-heated interface (large Qi, high phi) the conduction-zone integral comes out net positive; abs() turns an energy source into an equal-magnitude sink, draining the bubble via a term that physically deposits energy.",
    "repro": "Print the signed _trapezoid(integrand_cond, x=r_conduction) alongside L_conduction across a run with high ionizing luminosity.",
    "confidence": "medium",
    "lenses": ["A"],
    "divergence": "AC",
    "status": "single-lens",
    "source_ids": ["S7-A-11"]
  },
  {
    "id": "S7-R-13",
    "file": "trinity/bubble_structure/bubble_luminosity.py",
    "line": 430,
    "class": "regime",
    "severity": "S3",
    "claim": "The similarity terms use the absolute clock t_now rather than the elapsed bubble age t - tSF.",
    "evidence": "A transcribes the ODE's co-moving term as v_term = cool_alpha*r/params['t_now'].value and the boundary velocity as cool_alpha*R2/t_now, with no tSF subtraction anywhere in bubble_luminosity.py. C requires (S7-C-13) that alpha = v2*t/R2 presumes t measured from R2 = 0, so 'if the run has a non-zero tSF, using the absolute clock instead of t - tSF biases alpha, beta, delta and hence the whole interior structure', and notes the tSF argument on get_effective_bubble_pressure proves the distinction exists in the code's vocabulary. A confirms that argument exists (the R1 switch-on ramp uses t and tSF).",
    "expected": "One documented convention for the similarity clock, used identically in the ODE, the boundary conditions and wherever alpha/beta/delta are produced.",
    "failure_scenario": "With a non-zero tSF the co-moving term alpha*r/t is systematically wrong, biasing the interior velocity field and hence T(r), n(r) and L_cool, with no visible symptom.",
    "repro": "Run with tSF = 0 and tSF > 0 at otherwise matched state and compare the solved profile; also check the run's measured alpha = v2*t/R2 relaxes to 0.6 in the energy phase.",
    "confidence": "medium",
    "lenses": ["A", "C"],
    "divergence": "AC",
    "status": "single-lens",
    "source_ids": ["S7-C-13"]
  },
  {
    "id": "S7-R-14",
    "file": "trinity/bubble_structure/bubble_luminosity.py",
    "line": 430,
    "class": "other",
    "severity": "S3",
    "claim": "The cooling/heating rate that drives the structure ODE and the one integrated to report the luminosity come from two independent code paths with different unit conventions and different fA rules, and nothing enforces that they agree.",
    "evidence": "A: the ODE calls net_coolingcurve.get_dudt(t, n, T, phi, params) with AU-unit arguments, while the luminosity block hits params['cStruc_cooling_nonCIE'].interp / ['cStruc_heating_nonCIE'].interp directly with cgs arguments, and L_bubble uses cStruc_cooling_CIE_interpolation with no heating term at all. The fA rule also differs (ODE: only where T < 10^5.5; sum: whole of L_conduction and L_intermediate). B corroborates the two-convention split from the prose ('All cooling calculations take in au values, but the inner operations and outputs are cgs. The exception is get_dudt(), which takes in au and returns in au'). Separately, only the CIE arm makes its density convention explicit (chi_e * n**2); the non-CIE cubes carry theirs internally, so nothing in any lens establishes that the two halves of L_total share a density convention across the 10^5.5 seam.",
    "expected": "One function producing dudt for a given (n, T, phi, t), consumed by both the ODE and the luminosity integrals, with one fA rule.",
    "failure_scenario": "The energy the ODE removes from the structure differs from the L_total reported to the energy budget, so the bubble's thermal solution and its radiative loss are not mutually consistent — a discrepancy no test in this slice would catch.",
    "repro": "Evaluate net_coolingcurve.get_dudt and the direct table combination at the same (n, T, phi) and diff; separately check that the CIE and non-CIE arms agree in the limit T -> 10^5.5 from both sides.",
    "confidence": "medium",
    "lenses": ["A", "B"],
    "divergence": "none",
    "status": "single-lens",
    "source_ids": ["S7-A-30"]
  },
  {
    "id": "S7-R-15",
    "file": "trinity/bubble_structure/bubble_luminosity.py",
    "line": 848,
    "class": "regime",
    "severity": "S3",
    "claim": "cooling_boost_fA is applied to the whole of L_conduction and L_intermediate with no temperature condition, while the ODE applies it only where T < 10^5.5 — and L_intermediate is the 1e4-3e4 K band the ODE never integrates at all, so its boost has no in-ODE counterpart.",
    "evidence": "A: `if fA != 1.0: L_conduction = fA*L_conduction; L_intermediate = fA*L_intermediate` with no T test, versus the ODE's `if fA != 1.0 and T < _T_INTERFACE_BAND: dudt = fA*dudt`; L_intermediate's CIE (T >= 10^5.5) arm is covered by the unconditional boost. B reached the same gap from the prose: the in-ODE boost applies 'in the interface band ONLY' with the band's LOWER bound never stated, while the ODE domain bottoms out at T_init = 3e4 and L3 is by construction the 1e4-3e4 K extrapolated region outside that domain.",
    "expected": "Apply fA per-regime in the luminosity sum exactly as in the ODE, and state the interface band's lower bound; if the 1e4-3e4 K band is meant to be boosted, the ODE has no matching source term and the 'consistently' claim should be corrected.",
    "failure_scenario": "With fA != 1 the energy accounting applies the boost to a temperature range the ODE source term never saw, so L_eff and the structure actually integrated disagree.",
    "repro": "pytest test/test_fA_source_boost.py; then compare L_intermediate at fA=1 vs fA!=1 against a re-integrated structure. Note the CIE arm of L_intermediate is currently unreachable, so today only the band mismatch is live.",
    "confidence": "medium",
    "lenses": ["A", "B"],
    "divergence": "AB",
    "status": "corroborated",
    "source_ids": ["S7-A-12", "S7-B-05"]
  },
  {
    "id": "S7-R-16",
    "file": "trinity/bubble_structure/bubble_luminosity.py",
    "line": 645,
    "class": "numerical",
    "severity": "S3",
    "claim": "dMdt is converged against a 500-point, rtol=1e-6 integration, but every reported quantity comes from a separate ~6e4-point, rtol=1e-8 integration on which neither v(R1)=0 nor monotonicity is ever re-checked.",
    "evidence": "A: the search solve uses t_eval=linspace(...,500) with rtol=_RESIDUAL_RTOL=1e-6 while the production solve uses _create_radius_grid (3 x int(2e4) points) with rtol=_BUBBLE_RTOL=1e-8, dense_output=True and no t_eval; fsolve's xtol is 1e-4 with epsfcn=1e-4. The only post-hoc check on the production profile is np.any(T_array < 0), which does not catch NaN. B's numerical claim 2 asserts the tolerance gap is worth <= 0.3% in the converged dMdt (measured) — a claim about dMdt, not about the profile's residual. C's S7-C-20 flags an adjacent mechanism (a cached profile from the last residual evaluation), which A refutes: the ICs ARE recomputed at the converged dMdt.",
    "expected": "Assert |v(R1)|/|v(r2Prime)| on the production solution is within the intended bar, and re-run the monotonicity test there; or converge dMdt against the same tolerance and grid.",
    "failure_scenario": "The reported bubble structure violates its own inner boundary condition by more than the nominal xtol, with no diagnostic, and the discrepancy grows in stiff regimes where the two tolerances give visibly different profiles.",
    "repro": "After _solve_bubble_structure in _bubble_luminosity, log psoln[-1,0]/psoln[0,0] and compare against the residual fsolve converged to.",
    "confidence": "high",
    "lenses": ["A", "B"],
    "divergence": "none",
    "status": "single-lens",
    "source_ids": ["S7-A-28"]
  },
  {
    "id": "S7-R-17",
    "file": "trinity/bubble_structure/bubble_luminosity.py",
    "line": 801,
    "class": "divergence",
    "severity": "S3",
    "claim": "The division by dTdR_coolingswitch that sets the intermediate zone's outer radius has no zero or sign guard; a zero gradient gives an infinite radius (NaN grid) and a positive gradient inverts the zone so it runs inward and overlaps the conduction zone, hidden by the downstream abs().",
    "evidence": "A: R2_coolingswitch = (1e4 - T_array[ics])/dTdR_coolingswitch + r_array[ics], then r_interm = np.linspace(r_array[ics], R2_coolingswitch, 1000); every downstream integral wraps np.abs, so a reversed interval yields a positive contribution rather than an error. DEMOTED from A's S2: A's own analysis shows dTdR_coolingswitch is deterministically dTdr_cond[0] = -(2/5)*3e4/dR2 in every ordinary run, so the trigger requires the empty-mask fallback to dTdr_bubble[0].",
    "expected": "Require dTdR_coolingswitch < 0 and a bounded extrapolation length; otherwise skip the intermediate zone.",
    "failure_scenario": "A locally flat or inverted gradient at the outer boundary yields either an all-NaN r_interm propagating into L_total and Tavg, or a zone that double-counts radii already covered by the conduction zone.",
    "repro": "Force the conduction mask empty (all sampled T >= 10^5.5) and inspect dTdR_coolingswitch and R2_coolingswitch.",
    "confidence": "medium",
    "lenses": ["A"],
    "divergence": "none",
    "status": "single-lens",
    "source_ids": ["S7-A-19"]
  },
  {
    "id": "S7-R-18",
    "file": "trinity/bubble_structure/bubble_luminosity.py",
    "line": 863,
    "class": "state",
    "severity": "S3",
    "claim": "r_conduction[0] and r_conduction[-1] are indexed in the Tavg volume without the emptiness guard that the same post-mask array receives ~90 lines earlier.",
    "evidence": "A: line 776 uses `dTdR_coolingswitch = dTdr_cond[0] if len(dTdr_cond) > 0 else dTdr_bubble[0]`, guarding the empty-mask case; line 862-864 does `abs(r_conduction[0]**3 - r_conduction[-1]**3)` with no guard. Both consume the array after `mask = T_cond < _CIEswitch`.",
    "expected": "Guard both, or hoist a single early exit when the conduction mask is empty.",
    "failure_scenario": "If every sampled conduction-zone temperature is >= 10^5.5 the mask empties, line 776 falls back cleanly and line 863 then raises IndexError from inside the Tavg computation.",
    "repro": "",
    "confidence": "high",
    "lenses": ["A"],
    "divergence": "none",
    "status": "single-lens",
    "source_ids": ["S7-A-18"]
  },
  {
    "id": "S7-R-19",
    "file": "trinity/bubble_structure/bubble_luminosity.py",
    "line": 785,
    "class": "numerical",
    "severity": "S3",
    "claim": "The non-CIE table lookups take log10 of phi with no positivity guard, so Qi = 0 yields -inf as a table coordinate.",
    "evidence": "A: phi_cond = params['Qi'].value/(4*pi*r_conduction**2), then 10 ** interp(np.transpose(np.log10([n/ndens_cgs2au, T, phi/phi_cgs2au]))) in both the conduction and intermediate blocks. n and T are positive by construction; phi is proportional to Qi, which can legitimately reach zero.",
    "expected": "Floor phi at the table's lower edge before taking the logarithm.",
    "failure_scenario": "After the ionizing output has died away the interpolator is queried at -inf; behaviour depends on the external table object (extrapolation, NaN, or exception) and the resulting dudt silently contaminates L_conduction and L_intermediate.",
    "repro": "Set Qi = 0 in params and call _bubble_luminosity.",
    "confidence": "medium",
    "lenses": ["A"],
    "divergence": "none",
    "status": "single-lens",
    "source_ids": ["S7-A-26"]
  },
  {
    "id": "S7-R-20",
    "file": "trinity/bubble_structure/bubble_luminosity.py",
    "line": 881,
    "class": "deadcode",
    "severity": "S3",
    "claim": "Because _T_INIT_BOUNDARY = 3e4 exceeds _coolingswitch = 1e4, index_cooling_switch is 0 in every ordinary run, making four branches effectively or provably dead: the T_rgoal else, the CIE arm of L_intermediate, the Tavg else arm, and the fA CIE-band inconsistency.",
    "evidence": "A: no element of T_array is ever below 1e4 because the outer boundary IS 3e4, so index_cooling_switch != index_CIE_switch is essentially always True. The `else: T_rgoal = T_bubble[0]` branch is PROVABLY unreachable: it requires ics == iCIE, but then the elif test is textually the same expression as the if test that already evaluated False. L_intermediate's CIE mask (T_interm >= 10^5.5) is always empty over [1e4, 3e4] so that arm always `continue`s. B independently reports the region-2 lower bound documented as 1e4 (:752) versus the 3e4 anchor (:41-:45) — the same root cause seen from the prose side.",
    "expected": "Either remove the dead arms, or make the relationship between _T_INIT_BOUNDARY and _coolingswitch explicit so a future change to either is caught.",
    "failure_scenario": "Not active today. It is a latent trap: lowering _T_INIT_BOUNDARY below 1e4 (which C's S7-C-24 recommends) activates four untested branches simultaneously, including the fA inconsistency of S7-R-15.",
    "repro": "Add `raise AssertionError` at the T_rgoal else and run the suite — it never fires.",
    "confidence": "high",
    "lenses": ["A", "B"],
    "divergence": "AB",
    "status": "corroborated",
    "source_ids": ["S7-A-15", "S7-A-12", "S7-B-01"]
  },
  {
    "id": "S7-R-21",
    "file": "trinity/bubble_structure/bubble_luminosity.py",
    "line": 60,
    "class": "regime",
    "severity": "S3",
    "claim": "The 10^5.5 K threshold exists as at least two independent literals in this file plus a third cooling-table-derived value elsewhere, and the lockstep between them is defended only by a pinning test on the default bundle.",
    "evidence": "B (admitted in the prose): 'THREE places must stay in lockstep: this constant, the local _CIEswitch in _bubble_luminosity, and the cooling-table-derived nonCIE_Tcutoff in net_coolingcurve._noncie_cutoffs (they coincide on the default bundle; a table swap moves the third)'. A corroborates the first two independently: _T_INTERFACE_BAND = 10**5.5 at module level and _coolingswitch/_CIEswitch = 10**5.5 as locals in _bubble_luminosity. The same shape is admitted for _T_INIT_BOUNDARY ('THREE coupled roles, all of which must move together').",
    "expected": "Derive all three from one source, or assert their equality at load time rather than relying on a test pinned to the default bundle.",
    "failure_scenario": "A user swapping in a different cooling table moves nonCIE_Tcutoff; the fA band top and _CIEswitch stay put, so the non-CIE table is evaluated outside its documented validity range with no error.",
    "repro": "pytest test/test_fA_source_boost.py with a non-default cooling table bundle.",
    "confidence": "medium",
    "lenses": ["A", "B"],
    "divergence": "none",
    "status": "corroborated",
    "source_ids": ["S7-B-13"]
  },
  {
    "id": "S7-R-22",
    "file": "trinity/bubble_structure/bubble_luminosity.py",
    "line": 119,
    "class": "silent-failure",
    "severity": "S3",
    "claim": "_quiet_lsoda_fortran dup2-redirects file descriptors 1 and 2 to /dev/null for the duration of the solve, which is process-global and swallows any concurrent output, not just the LSODA banner the docstring names.",
    "evidence": "A confirms the mechanism and that restoration happens in a `finally` block — so B's 'if the redirect is not exception-safe' half is REFUTED and only the scope concern stands. B: the docstring claims it 'suppresses only that noise' and 'touches no numerics'.",
    "expected": "Scope the suppression to the LSODA banner, or state in the docstring that ALL fd-1/fd-2 output during the solve is discarded, including from other libraries, warnings and concurrent workers.",
    "failure_scenario": "Under `--workers N`, real error text emitted by any component during a bubble solve disappears, making a failing sweep much harder to diagnose.",
    "repro": "Write to fd 1 from inside the ODE RHS and confirm it is swallowed; confirm restoration after a BubbleSolverError propagates.",
    "confidence": "high",
    "lenses": ["A", "B"],
    "divergence": "none",
    "status": "corroborated",
    "source_ids": ["S7-B-06"]
  },
  {
    "id": "S7-R-23",
    "file": "trinity/bubble_structure/bubble_luminosity.py",
    "line": 1022,
    "class": "deadcode",
    "severity": "S4",
    "claim": "_capture_bubble_integration reads infodict['ier'], a key _solve_bubble_structure never creates, so every diagnostic log prints ier=None and every saved npz stores -999.",
    "evidence": "A: infodict is built with keys {'message','status','nfev','nst','hu'}; line 1022 does `_ier = infodict.get('ier')` and line 1068 stores `ier if ier is not None else -999`. 'ier' is an odeint-era key. B independently found the prose already admitting it at :1053: '(No ier key; success is read from status.)' — while :454-:476 documents success as sol.success, giving two documented sources of truth.",
    "expected": "Read 'status' (the solve_ivp equivalent), and document one source of truth for solver success.",
    "failure_scenario": "",
    "repro": "TRINITY_BUBBLE_DIAG=1 on any run that trips the diagnostic; every saved npz has ier == -999.",
    "confidence": "high",
    "lenses": ["A", "B"],
    "divergence": "AB",
    "status": "corroborated",
    "source_ids": ["S7-A-14", "S7-B-11"]
  },
  {
    "id": "S7-R-24",
    "file": "trinity/bubble_structure/bubble_luminosity.py",
    "line": 608,
    "class": "numerical",
    "severity": "S4",
    "claim": "_clean_radius_grid is close to a no-op and its docstring describes something it does not do: the threshold is a RELATIVE 1e-12 while the duplicates it names are absolute gaps of 1e-8 to 1e-9, and the keep-mask is computed from the ORIGINAL consecutive differences rather than the distance to the last kept point.",
    "evidence": "B's arithmetic: at radii of order 0.1-100 pc an absolute 1e-8..1e-9 gap is a relative 1e-8..1e-11, comfortably ABOVE the 1e-12 cut, so the documented threshold would keep the documented duplicates. A's mechanism: `keep_mask = concatenate([[True], relative_diff >= min_relative_spacing])` — dropping r[i+1] does not re-reference r[i+2] against r[i]. C's related fear (S7-C-23, that cleaning would thin the front's log-refined region) is REFUTED by the same arithmetic: a near-no-op cannot thin anything.",
    "expected": "Either state that the quoted 1e-8/1e-9 differences are relative, or fix the constant; and use a sequential filter comparing each candidate against the last accepted point.",
    "failure_scenario": "Harmless today ('grid hygiene only'), but the docstring misleads anyone who tunes the constant or re-enables a dense-output path — and raising MIN_SPACING to make it 'work' would delete exactly the near-CD points that carry the luminosity.",
    "repro": "_create_radius_grid(R1~1e-3, r2Prime~1) and compare len() before/after cleaning; also _clean_radius_grid(np.array([1.0, 1.0+5e-13, 1.0+1e-12, 1.0+1.5e-12])).",
    "confidence": "medium",
    "lenses": ["A", "B"],
    "divergence": "AB",
    "status": "corroborated",
    "source_ids": ["S7-A-21", "S7-B-04"]
  },
  {
    "id": "S7-R-25",
    "file": "trinity/bubble_structure/bubble_luminosity.py",
    "line": 746,
    "class": "coefficient",
    "severity": "S3",
    "claim": "VERIFICATION ITEM: the CIE emissivity chi_e*n**2*Lambda equals the required n_e*n_H*Lambda, and rho = n*mu_convert is the matched (mu, n) pair, if and only if chi_e = n_e/n_H ~ 1.2 and mu_convert/mu_ion = n_tot/n_H ~ 2.3 — two .param values no lens saw.",
    "evidence": "A: n = Pb/((mu_convert/mu_ion)*k_B*T), so n is the isobaric total density divided by mu_convert/mu_ion; if that ratio is 2.3 then n = n_H and chi_e*n**2 = (n_e/n_H)*n_H**2 = n_e*n_H, exactly C's expected integrand. A reached the same pairing from the mass side: rho = n*mu_convert gives rho = mu_ion*Pb/(k_B*T), which is C's S7-C-08 requirement satisfied. C quantifies the penalty for getting it wrong: n_tot**2 over-counts n_e n_H by 2.3**2/1.2 = 4.4, n_H**2 under-counts by 1.2, and the error lands directly on the phase-transition trigger. C's S7-C-01 (volume element) and S7-C-08 (mu/n mismatch) are refuted by A; this residual value check is what remains of C's S1 cluster.",
    "expected": "chi_e ~ 1.2 and mu_convert/mu_ion ~ 2.3 for the declared composition; assert n*T*(mu_convert/mu_ion)*k_B == Pb across the stored profile (the isobaricity test), and M_b/V_b reproducing n_tot = Pb/(k_B*T_b).",
    "failure_scenario": "If either constant carries the wrong composition convention, L_bubble is off by up to 4.4x and bubble_mass by 2.3x, shifting the energy-phase duration and every downstream grid result — with no dimensional or shape symptom.",
    "repro": "Assert max|n*T*k_B*(mu_convert/mu_ion)/Pb - 1| < tol on bubble_n_arr/bubble_T_arr; grep default.param for chi_e, mu_convert, mu_ion.",
    "confidence": "medium",
    "lenses": ["A", "C"],
    "divergence": "none",
    "status": "corroborated",
    "source_ids": ["S7-C-02", "S7-C-08", "S7-C-27"]
  },
  {
    "id": "S7-R-26",
    "file": "trinity/bubble_structure/get_bubbleParams.py",
    "line": 373,
    "class": "regime",
    "severity": "S4",
    "claim": "The early-phase R1 switch-on ramp R1_tmp = (t - tSF)/1e-3 * R1 is unclamped, so for t < tSF it goes negative, giving r1**3 < 0 and a shell volume LARGER than geometric.",
    "evidence": "A: `dt_switchon = 1e-3; if t <= tmin + tSF: R1_tmp = (t - tSF)/tmin * R1; return bubble_E2P(Eb, R2, R1_tmp, gamma)` with bubble_E2P's `shell_volume = r2**3 - r1**3`. B corroborates that the ramp is undocumented — it exists only as a body comment while the docstring enumerates two phases.",
    "expected": "Clamp the ramp factor to [0, 1]; document the branch in the docstring.",
    "failure_scenario": "Any evaluation at t < tSF (an ODE probe backwards, or a delayed cluster formation time) returns a pressure below the correct value rather than being rejected. The same ramp also drives R1 -> 0 for t just after tSF, which is the condition that breaks the max() invariant of S7-R-07.",
    "repro": "get_effective_bubble_pressure('energy', Eb, R2, R1, 5/3, t=0.9, tSF=1.0) vs t=1.0.",
    "confidence": "high",
    "lenses": ["A", "B"],
    "divergence": "none",
    "status": "corroborated",
    "source_ids": ["S7-A-27", "S7-B-14"]
  },
  {
    "id": "S7-R-27",
    "file": "trinity/bubble_structure/get_bubbleParams.py",
    "line": 282,
    "class": "regime",
    "severity": "S4",
    "claim": "get_leak_luminosity guards coverFraction >= 1.0 but has no lower bound, so a negative covering fraction gives a leak factor (1 - Cf) > 1 — an open area exceeding the whole sphere.",
    "evidence": "A: `if coverFraction >= 1.0 or Pb <= 0.0 or c_sound <= 0.0: return 0.0` then `* (1.0 - coverFraction) *`. B: the documented domain is Cf in (0, 1] and the 'self-limits and never injects energy' claim covers only the injection direction, not over-draining. C confirms the rest of the function is correct (form, the gamma/(gamma-1) enthalpy factor, and exact zero at Cf = 1), so this is the only residual.",
    "expected": "Clamp coverFraction to [0, 1], or validate it at the .param trust boundary.",
    "failure_scenario": "A negative or zero coverFraction from a sweep .param drains enthalpy faster than the geometric maximum, killing the bubble early with no error raised.",
    "repro": "pytest test/test_cf_leak.py with coverFraction <= 0.",
    "confidence": "high",
    "lenses": ["A", "B"],
    "divergence": "none",
    "status": "corroborated",
    "source_ids": ["S7-A-25", "S7-B-21"]
  },
  {
    "id": "S7-R-28",
    "file": "trinity/bubble_structure/get_bubbleParams.py",
    "line": 169,
    "class": "state",
    "severity": "S4",
    "claim": "cool_beta_to_Ebdot reads params entries as `.value` while its exact algebraic inverse Ebdot_to_cool_beta reads them as bare floats, so the two cannot be called with the same container.",
    "evidence": "A: `t_now = my_params['t_now']` etc. in Ebdot_to_cool_beta versus `params['t_now'].value` in cool_beta_to_Ebdot. B found the same asymmetry stated explicitly in both docstrings ('Must provide .value for' vs 'plain float values, not .value-wrapped'). C requires the pair to be an exact machine-precision round trip.",
    "expected": "One container convention, or an explicit adapter; and a round-trip test beta -> Ebdot -> beta.",
    "failure_scenario": "Passing the standard params object to Ebdot_to_cool_beta propagates wrapper objects into the arithmetic; at worst a float-like DescribedDict entry is consumed silently and produces a wrong beta.",
    "repro": "Round-trip test on a single state.",
    "confidence": "high",
    "lenses": ["A", "B"],
    "divergence": "none",
    "status": "corroborated",
    "source_ids": ["S7-A-29", "S7-B-15"]
  },
  {
    "id": "S7-R-29",
    "file": "trinity/bubble_structure/bubble_luminosity.py",
    "line": 752,
    "class": "citation",
    "severity": "S3",
    "claim": "Doc-drift cluster: seven prose statements that the code lens contradicts and the physics lens arbitrates against — the code is right and the comments are stale.",
    "evidence": "(1) :752 documents region 2 as 10**4 < T < 10**5.5; A shows it is 3e4 -> 10**5.5 because r2Prime IS the 3e4 anchor. (2) :934 states M(r) = INT[0 -> r]; A shows the grid spans [R1, r2Prime] and C's expected limits are also [R1, r2Prime]. (3) :381 'R1 = interface separating inner bubble radius and outer solar wind'; C's spec states the free (stellar, not solar) wind is INSIDE R1 and the shocked bubble outside, matching :199-:217 and :415. (4) :741 'import values from two cooling curves' in the CIE-only region; A shows only cStruc_cooling_CIE_interpolation is queried there. (5) :855-:859 explains abs() only for the volume denominator; A shows abs() is applied to each numerator term too. (6) the intermediate region is 'tiny (or non-existent)'; A shows it is always (5/3)*dR2 thick (see S7-R-01). (7) the transition max() is a 'smooth handoff'; max is C0 but not C1.",
    "expected": "Correct each comment to match the code. None of these is a runtime defect; together they are why several of this slice's real findings went unnoticed.",
    "failure_scenario": "A future edit trusts the comment over the code — e.g. lowering _T_INIT_BOUNDARY to make region 2 match its documented 1e4 bound, which activates the four dead branches of S7-R-20 at once.",
    "repro": "",
    "confidence": "high",
    "lenses": ["A", "B", "C"],
    "divergence": "AB",
    "status": "corroborated",
    "source_ids": ["S7-B-01", "S7-B-02", "S7-B-09", "S7-B-14", "S7-B-18", "S7-B-23"]
  },
  {
    "id": "S7-R-30",
    "file": "trinity/bubble_structure/bubble_luminosity.py",
    "line": 415,
    "class": "citation",
    "severity": "S4",
    "claim": "OPEN QUESTION, not a defect: four Weaver+77 equation numbers (33, 42, 43, 44), four Rahner-thesis references in three different numbering styles, and one unlocatable 'leakage spec, Eq. (leak)' are all unverifiable from this container.",
    "evidence": "B recorded them verbatim and notes the internal consistency (Eq 33 attached to the same quantity twice) but could not check any. C REFUSED to assert Weaver equation numbers or numerical prefactors on principle ('the paper is unreachable here ... any code comment of the form Weaver eq. N is unverifiable and must not be treated as evidence for a coefficient'). A and C independently derived identical physics for the ODE (Eqs 42-43), the boundary conditions (Eq 44) and the dMdt seed's 16pi/25 grouping (Eq 33), so the PHYSICS is corroborated even though the CITATIONS are not. The one number nothing pins is the 1.646 prefactor in T_c = 1.646*(Pb*R2**2/(C*t))**(2/7).",
    "expected": "Verify 33/42/43/44 against Weaver, Castor, McCray & Moore 1977 (ApJ 218, 377); verify the Rahner numbering ('pg 79 Eq A5', 'pg 80 Eq A12', 'pg71 Eq 6', 'eq 1.25') is not a transcription slip; give 'the leakage spec' a resolvable path.",
    "failure_scenario": "Not a runtime defect. It defeats every future attempt to verify the implemented physics against its source — and it is the reason S7-R-02 (the inner boundary condition) cannot be closed here.",
    "repro": "Fetch Weaver+77 Eq 44 and check whether the inner closure is v(R1)=0 or the strong-shock v_w/4; fetch Rahner A12 and check whether gamma appears explicitly.",
    "confidence": "medium",
    "lenses": ["B", "C"],
    "divergence": "BC",
    "status": "contested",
    "source_ids": ["S7-B-16", "S7-B-17", "S7-C-42"]
  },
  {
    "id": "S7-R-31",
    "file": "trinity/bubble_structure/bubble_luminosity.py",
    "line": 392,
    "class": "deadcode",
    "severity": "S4",
    "claim": "Vestigial-code cluster: _get_bubble_ODE_initial_conditions accepts R1 and never uses it; its line-404 temperature is an algebraically exact round-trip back to _T_INIT_BOUNDARY; _get_mass_and_grav always returns grav_phi = grav_force_m = None which the sole caller discards; `Optional` and `astropy.units as u` are imported and never used; and bubble_E2P mutates its arguments in place.",
    "evidence": "A: the ICs body references only k_B, mu_ion, cooling_boost_kappa, C_thermal, R2, dMdt, Pb, cool_alpha, t_now; T = (K*dMdt*dR2/(4*pi*R2**2))**(2/5) with dR2 = T_init**(5/2)/(K*dMdt/(4*pi*R2**2)) is identically T_init; grav returns None; `Optional` and `u.` never appear again; `r1 *= pc2cm`, `r2 *= pc2cm`, `Eb *= E_au2cgs`, `r2 += 1e-10` would mutate a caller's numpy array. B corroborates the gravity block as 'currently DISABLED' with None placeholders, and additionally notes that restoring it 'verbatim' would give a radius-independent SCALAR labelled as a potential, omit the -G*M(r)/r interior term, and soften the force with an epsilon chosen for an r=0 that cannot occur on a grid starting at R1 > 0.",
    "expected": "Drop the unused R1 parameter (or use it to guard r2_prime > R1); use T_init directly; return only the cumulative mass; remove the unused imports; bind bubble_E2P's conversions to new local names.",
    "failure_scenario": "None today. The in-place mutation becomes live the moment any call site passes a numpy array.",
    "repro": "assert _get_bubble_ODE_initial_conditions(dMdt, p, Pb, R1)[1] == 3e4 for any dMdt > 0. a = np.array([1.0]); bubble_E2P(np.array([1.0]), a, np.array([0.5]), 5/3); a is now in cm.",
    "confidence": "high",
    "lenses": ["A", "B"],
    "divergence": "none",
    "status": "corroborated",
    "source_ids": ["S7-A-03", "S7-A-04", "S7-A-16", "S7-A-20", "S7-A-22", "S7-B-10"]
  },
  {
    "id": "S7-R-32",
    "file": "trinity/bubble_structure/bubble_luminosity.py",
    "line": 199,
    "class": "other",
    "severity": "S4",
    "claim": "Model-validity ceilings and closure invariants that no code in this slice records or asserts: delta = (2/7)(2*alpha - beta - 1); E_b/(L_w*t) -> 5/11; classical unsaturated Spitzer conduction; CIE assumed in a non-equilibrium front; and the Weaver exponents alpha=3/5, beta=4/5, delta=-6/35 being w=0 values only.",
    "evidence": "C only; A and B are silent on all of them, which is itself the finding — none of these appears as an assertion, a comment or a diagnostic. C derived delta = (2/7)(2 alpha - beta - 1) independently from the conduction closure T_b^(7/2) ∝ P_b R2^2/(C t) and notes it links three otherwise independent default.param constants. C also notes that with rho ∝ r^-w, alpha = 3/(5-w), so a run with densPL_alpha != 0 must re-solve the exponents rather than assume them; that lambda_e ~ 1e4*T^2/n_e ~ 30 pc at T=1e7, n_e~1e-2 makes the hot interior formally saturated (so the code sets the MAXIMUM possible evaporation); and that the 1e4-1e5.5 K band carrying ~99% of L_cool is served cleanly by neither the CIE table nor the photoionised-shell cube.",
    "expected": "A cheap runtime or test-suite assertion on the delta closure and the 5/11 energy ratio during the energy phase; and a documented statement of the four physical ceilings and their bias directions.",
    "failure_scenario": "A systematic violation of the delta closure means the interior structure being integrated is not the one the (alpha, beta, delta) triple describes — and nothing would surface it. The saturation and NEI ceilings bias L_cool by factors of a few in a known direction that is currently undocumented.",
    "repro": "Extract alpha, beta, delta from dictionary.jsonl and plot delta against (2/7)(2*alpha - beta - 1); check alpha -> 0.6 and E_b/(L_w t) -> 5/11 in the energy phase. Note the alpha/beta/delta producers are outside this slice.",
    "confidence": "medium",
    "lenses": ["C"],
    "divergence": "none",
    "status": "single-lens",
    "source_ids": ["S7-C-41", "S7-C-25", "S7-C-43", "S7-C-44"]
  }
]
```
