# S9 cooling — Lens C (what it should be)

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

**Scope.** Derived-from-physics expectations for `trinity/cooling/net_coolingcurve.py`,
`trinity/cooling/CIE/read_coolingcurve.py`, `trinity/cooling/non_CIE/read_cloudy.py`.
I have read **only** the signature list and `PHYSICS_SPEC.md`. No implementation, no comments, no
tables. Literature egress is blocked, so every *tabulated* number below is flagged **[recalled]**
and every *derived* number is reproducible from the arithmetic shown.

Interface I am reasoning against:

```
net_coolingcurve.get_dudt(age, ndens, T, phi, params_dict)
net_coolingcurve._cie_tcutoff(logT_CIE)
net_coolingcurve._noncie_cutoffs(cooling_nonCIE)
CIE/read_coolingcurve.get_Lambda(T, cooling_CIE_interpolation, metallicity)
non_CIE/read_cloudy.{get_coolingStructure, cube_linear_interpolate, cube(age,datacube,interp,ndens,temp,phi),
                     create_cubes, create_limits, get_filename(age, metallicity, SB99_rotation, path), get_fileage}
```

---

## 1. The correct volumetric cooling rate, and the normalisation factor of **4.41**

### 1.1 Why it is a product of *two* densities

Every radiative cooling channel that matters between 10⁴ and 10⁸ K in optically thin gas is a
**two-body collisional** process: a free electron collides with an ion (or neutral) and the energy
goes into a photon that escapes. The rate per unit volume of a two-body process is
`∝ n_collider × n_target`. Hence

```
    (dE/dV/dt)_cool = n_e · Σ_X,i  n_{X,i} · Λ_{X,i}(T)          [erg cm⁻³ s⁻¹]
```

with `Λ_{X,i}` the per-ion cooling efficiency in **erg cm³ s⁻¹**. This is exactly the quantity
Gnat & Ferland 2012 tabulate ion by ion. Photo-*heating*, by contrast, is a **one-body** process
(`Γ ∝ n_H Φ σ`), which is the root cause of every convention problem below: the net rate is *not*
`n² × f(T)` once radiation is present, which is precisely why the non-CIE data must be a **cube**
in `(n, T, Φ)` and not a curve. **[derived, high]**

Because `n_e` and every `n_{X,i}` are proportional to `n_H` at fixed ionisation state, the sum
collapses to a single function of `T` (and `Z`) times a *chosen* density product. Which product is
chosen is pure convention — and the choice must be matched between the table and the code.

### 1.2 The three conventions and the exact conversion factors

Composition: cosmic/solar, `y ≡ n_He/n_H = 0.1` (i.e. `X≈0.71, Y≈0.28`), metals negligible in
number (<1 % of `n_e`). **Fully ionised**:

```
    n_e   = n_H (1 + 2y) = 1.2  n_H
    n_ion = n_H (1 + y)  = 1.1  n_H          (nuclei)
    n_tot = n_e + n_ion  = 2.3  n_H
    ρ     = 1.4 m_H n_H                       (mass per H nucleus, μ_H = 1.4)
    μ     = ρ/(n_tot m_H) = 1.4/2.3 = 0.609   (= 14/23, matches SPEC-092)
```

Density products, all in units of `n_H²`:

| product | value / `n_H²` |
|---|---|
| `n_e n_H` | **1.20** |
| `n_H²` | 1.00 |
| `n_e n_ion` | 1.32 |
| `n_tot²` | **5.29** |
| `(ρ/m_H)²` | 1.96 |

**THE NUMBER.** For fully ionised gas of cosmic abundance,

```
    n_tot² / (n_e n_H)  =  5.29 / 1.20  =  4.408   ≈  4.41
```

So a code that reads a table normalised **per `n_e n_H`** and multiplies it by **`n_tot²`**
over-cools by a factor **4.41**; the reverse pairing under-cools by the same 4.41. Robustness of
this number: with `y = 0.084` (X=0.7381, Y=0.2485) it is 4.34; with the round `μ = 0.6`
(`n_tot = 2.333 n_H`) it is 4.54. **The answer is 4.4 ± 0.15 for any defensible cosmic
composition.** **[derived, high]**

Secondary pairings, same derivation:

| mispairing | multiplicative error |
|---|---|
| table per `n_e n_H`, code uses `n_H²` | **0.833** (under-cools 17 %) |
| table per `n_H²`, code uses `n_tot²` | **5.29** |
| table per `n_e n_H`, code uses `(ρ/m_H)²` | **1.633** |
| table per `n_e n_ion`, code uses `n_e n_H` | 0.909 |
| table per `n_e n_H`, code uses `n_e n_tot` | 1.917 |

Important sub-trap: **`n_e = 1.2 n_H` only in fully ionised gas.** In the 10⁴ K photoionised shell
He is singly ionised at most, so `n_e ≈ 1.1 n_H` (SPEC-029's `chi_e_shell = 1.1`); in neutral gas
`n_e/n_H ~ 10⁻⁴`. A code that hard-codes one electron fraction across the whole 10⁴–10⁸ K range is
wrong at one end. In the *hot* branch (>10⁵·⁵ K) full ionisation is safe; in the non-CIE branch it
is not, which is another reason the non-CIE table should supply the net rate rather than a `Λ` the
caller must re-multiply. **[derived, high]**

### 1.3 Which convention each table family uses

- **Sutherland & Dopita 1993** — normalised `Λ_N` such that the rate is `n_e n_t Λ_N`, with `n_t`
  the total *ion* (nuclei) density; they also publish a `Λ/n_H²` variant. **[recalled, medium]**
- **Gnat & Sternberg 2007** — cooling efficiency per `n_H n_e`. **[recalled, medium]**
- **Gnat & Ferland 2012** (TRINITY's default CIE file per SPEC-081) — *ion-by-ion* efficiencies
  `Λ_ion` defined per **`n_e n_ion`**; a total curve assembled from them is naturally per
  **`n_e n_H`** once the ion fractions and abundances are folded in. **[recalled, medium]**
- **CLOUDY** — `save cooling` emits the *volumetric* rate in erg cm⁻³ s⁻¹ at the model's own
  density; the community normalisation applied afterwards is almost always `/n_H²`. A file named
  `coolingCIE_1_Cloudy.dat` is therefore most likely **per `n_H²`** while
  `coolingCIE_3_Gnat-Ferland2012.dat` is most likely **per `n_e n_H`** — the two bundled options
  plausibly differ by 1.2 and a single code-side multiplier cannot be right for both.
  **[recalled, medium — this is a concrete thing the reconciler should check]**

### 1.4 What `get_dudt` must therefore return

```
    du/dt  =  Γ(n_H, T, Φ, age, Z)  −  n_e n_H Λ(T, Z)          [erg cm⁻³ s⁻¹]
           =  − Λ_net · (matched density product)
    du/dt  <  0   ⇔   net cooling
```

with `u` the thermal energy density (`u = (3/2) n_tot k_B T` for a monatomic ideal gas), so that
the bubble/shell energy equation reads `∂u/∂t|_rad = du/dt` directly and
`L_cool = −∫ 4πr² (du/dt) dr`. **[derived, high]**

Sanity anchor for the reconciler (derived): `t_cool = (3/2) n_tot k_B T /(n_e n_H Λ)
= 3.97×10⁻¹⁶ · T/(n_H Λ)` s. At `n_H = 1`, `T = 10⁶`, `Λ = 3×10⁻²³` ⇒ 0.42 Myr (the textbook
"~1 Myr at n=1, T=10⁶"). At bubble conditions `n_H = 10⁻²`, `T = 10⁷`, `Λ = 10⁻²³` ⇒ **1.3 Gyr** —
i.e. the *bulk* of a Weaver bubble is effectively non-radiative and essentially all of `L_cool`
comes from the thin, dense, cool layer at the conduction front. **A factor-4.4 normalisation error
therefore does not just rescale a small term; it rescales the quantity that fires the
energy→momentum transition (SPEC-013).** **[derived, high]**

---

## 2. Structure of Λ(T), 10⁴ → 10⁸ K

Solar metallicity, CIE, optically thin, per `n_e n_H`, in erg cm³ s⁻¹:

1. **`T < 10⁴ K` — the cliff.** Hydrogen recombines; there is no electron reservoir and no
   accessible excitation. `Λ` falls by **4–6 decades** between 10⁴ and 10³·⁵ K, governed by the
   Lyα Boltzmann factor `exp(−1.18×10⁵ K/T)`. Local log–log slope at 10⁴ K:
   `d lnΛ/d lnT ≈ χ/kT = 11.8`. **This is by a wide margin the steepest feature in the whole
   curve.** Nearly all CIE tables simply stop at 10⁴ K. **[derived, high]**
2. **`10⁴ – 3×10⁴ K` — H Lyα peak.** Sharp local maximum near `log T ≈ 4.2–4.3`, `Λ ~ 10⁻²¹·⁵`.
   **[recalled, medium]**
3. **`~10⁴·⁷ K` — a dip**, where H is exhausted as a coolant and the metal ions have not yet
   turned on. **[recalled, medium]**
4. **`10⁵ – 10⁵·⁵ K` — the main line-cooling peak**, the dominant feature: collisionally excited
   resonance lines of C III/C IV, N V, O III–O VI, plus He II Lyα.
   **`Λ_peak ≈ (2–4)×10⁻²² erg cm³ s⁻¹` at `log T ≈ 5.0–5.3`.** **[recalled, medium — value
   ±0.3 dex; location high confidence]**
5. **`10⁶ – 10⁶·⁵ K` — Fe-L shoulder**, `Λ ~ 5×10⁻²³`. **[recalled, medium]**
6. **`≳ 2×10⁷ K` — bremsstrahlung tail, `Λ ∝ T^{1/2}`.** Derived, not recalled:
   `ε_ff = 1.4×10⁻²⁷ T^{1/2} n_e Σ n_i Z_i² ḡ`, with `Σ n_i Z_i² = n_H(1+4y) = 1.4 n_H`,
   `ḡ ≈ 1.2`:

```
    Λ_ff = 1.4e-27 × 1.4 × 1.2 × T^{1/2} = 2.35×10⁻²⁷ √T   erg cm³ s⁻¹   (per n_e n_H)
         ⇒ 2.35×10⁻²³ at 10⁸ K,  7.4×10⁻²⁴ at 10⁷ K
```

   **[derived, high]** — this is the correct high-T extrapolation law (see §5).

**Where the curve is steepest, hence where interpolation error is worst**, in rank order:

| rank | region | log–log slope | consequence |
|---|---|---|---|
| 1 | `T ≲ 10⁴ K` cliff | up to **+12** | any linear/clamped treatment is catastrophically wrong |
| 2 | `10⁴–10⁴·³` Lyα rise | +3 to +6 | sets the shell's thermal equilibrium |
| 3 | curvature at the `10⁵` peak | 0, curvature `≈ −3 dex⁻²` | linear interpolation **systematically undershoots** a concave peak |
| 4 | `10⁵·⁵–10⁷` falling side | ≈ −1 | mild |
| 5 | bremsstrahlung tail | +½ | exactly linear in log–log ⇒ error-free |

**Structural consequence for TRINITY specifically.** The declared split is `10^{5.5} K`
(SPEC-080), which sits **above** the main line-cooling peak. Therefore *the peak of Λ — the single
most important feature of the curve for bubble energetics — is evaluated on the **non-CIE**
branch*, not the CIE branch. The non-CIE cube must reproduce the CIE peak in its low-ionisation-
parameter limit or `L_cool` is wrong by the peak/off-peak contrast. **[derived, high]**

Counterpoint worth checking: with the Weaver profile `T(ξ) = T_b(1−ξ)^{2/5}` and the reported cut
`bubble_xi_Tb = 0.98` (SPEC-040), the coldest gas inside the integrated domain is
`0.02^{0.4} = 0.209 T_b ≈ 2×10⁶ K` for `T_b = 10⁷ K`; `T = 10^{5.5}` is reached only at
`1−ξ = (10^{−1.5})^{5/2} = 1.8×10⁻⁴`, i.e. `ξ = 0.99982`. So **for the bubble interior the CIE
branch should do essentially all the work**, and the non-CIE cube is exercised by shell/H II gas.
Also: the cooling integrand behaves as `(1−ξ)^{−4/5}·Λ(T(ξ)) ∝ (1−ξ)^{−6/5}` for `Λ ∝ T^{−1}`, so
`L_cool` is dominated by the last ~1 % in radius and shifts by **~15–20 % for a 0.98→0.99 change
in the cut**. **[derived, medium]**

---

## 3. Metallicity scaling

### 3.1 The correct law

In the optically thin, low-density, CIE limit each line's emissivity is *linear* in the emitting
ion's abundance, and in CIE the ionisation balance of element X is independent of the abundance of
every other element. Therefore the cooling function **separates additively**:

```
    Λ(T, Z) = Λ_HHe(T)  +  (Z/Z_⊙) · Λ_metals,⊙(T)          with Λ(T, Z_⊙) = Λ_⊙(T)
```

- **Scales with Z:** metal line cooling (C, N, O, Ne, Si, S, Fe), metal recombination continua,
  and (separately, and *not* with the same functional form) dust/gas grain cooling and
  photoelectric heating, both `∝` dust-to-gas `∝ Z`.
- **Does NOT scale with Z:** H and He line cooling, H/He recombination and two-photon continua,
  and **thermal bremsstrahlung** — which is the *entire* budget above ~10⁷·⁵ K.
- Second-order: metals contribute ~1 % of `n_e`, so even `n_e`-mediated terms are Z-independent to
  1 % for `Z ≤ few Z_⊙`. **[derived, high]**

### 3.2 The traps

1. **`Λ(T,Z) = (Z/Z_⊙)·Λ_⊙(T)` is wrong.** It sends `Λ → 0` as `Z → 0`, but the true floor is
   H/He + bremsstrahlung. At `Z = 0.1 Z_⊙` this under-predicts `Λ(10⁸ K)` by a **factor 10**
   (bremsstrahlung is Z-independent) and `Λ(2×10⁴ K)` by a similar factor (Lyα-dominated). The
   error is *largest at the two ends and small only near the 10⁵ K metal peak*, so a run that
   looks "roughly right" at the peak can be an order of magnitude wrong in the bubble.
2. **Double-scaling.** `coolingCIE_3_Gnat-Ferland2012.dat` is a table *at a specific metallicity*
   (solar). Multiplying it again by `Z/Z_⊙` double-counts. `get_Lambda(T, interp, metallicity)`
   must do exactly one of: (a) select a per-Z table/axis, (b) apply the two-component law of §3.1
   against a separately tabulated `Λ_HHe`, or (c) validate `Z == 1` and raise. A bare
   `Lambda *= metallicity` is only correct if the table is a **metals-only** table.
3. **Silent no-op at `Z = 1`.** Because SPEC-085 says only `Z = 1` is supported, *any* of the wrong
   behaviours above is invisible in every shipped run and only detonates the first time someone
   sweeps `ZCloud` — which `docs/source/running.rst` already advertises as an example
   (`ZCloud = [0.5, 1.0]`). This is the classic dormant-landmine class.
4. **Linear Z vs dex Z.** If `metallicity` is `[Z/H]` in dex and used as a linear multiplier,
   `Z_⊙ → 0` ⇒ `Λ ≡ 0` ⇒ **no cooling at all** (the bubble never leaves the energy phase). If it is
   linear and used as a dex offset, the error is 10^Z. At `Z = 1` the two disagree completely
   (`1` vs `0`), so this one *is* detectable at solar.
5. **Scaling a log-stored table.** CIE tables usually store `log₁₀ Λ`. Multiplying the *stored log*
   by `Z` gives `Λ^Z` — identity at `Z = 1`, nonsense elsewhere, and silently sign-flipping the
   cooling for `Λ < 1` (which is always, since `Λ ~ 10⁻²²`): `log Λ = −22`, `×0.5 ⇒ 10⁻¹¹`, a
   **10¹¹ over-cooling**. This is loud, but only in a non-solar run.
6. **Metallicity must also gate the *non-CIE* cube.** `get_filename(age, metallicity,
   SB99_rotation, path)` takes `metallicity` — so Z there is a *file selector*, not a multiplier.
   Applying a Z multiplier to the cube's output *and* selecting a Z-specific file is the same
   double-count as (2). **[all derived, high]**

---

## 4. CIE versus non-CIE

### 4.1 When CIE is valid

CIE requires all four of:

1. **Ionisation equilibrium**, i.e. `t_ion,rec ≪ t_cool`. Both scale as `1/n`, so the ratio is
   **density-independent** and purely a function of `T`:
   `t_rec/t_cool = (2/3) n_H Λ /(α n_tot k_B T)`. At `T = 10⁵ K` with `Λ = 3×10⁻²²`,
   `α ≈ 4×10⁻¹³`: **`t_rec/t_cool ≈ 16`** — recombination is an order of magnitude *slower* than
   cooling. **CIE therefore fails, by construction, in exactly the 10⁴·⁵–10⁵·⁵ K band where the
   cooling peak lives.** Rapidly cooling gas stays over-ionised ("frozen-in"), which *reduces* the
   cooling efficiency relative to CIE by up to a factor ~2 near 10⁵ K (Gnat & Sternberg 2007
   isochoric/isobaric results) **[derivation high; the factor ~2 recalled, medium]**.
   *This is the conduction front of a Weaver bubble.*
2. **Negligible photoionisation:** photoionisation rate ≪ collisional ionisation rate, i.e.
   ionisation parameter `U = Φ/(n_H c) ≲ 10⁻⁴`. Numerically, for `Q_i = 10⁵¹ s⁻¹`, `R₂ = 10 pc`:
   `Φ = Q_i/(4πR₂²) = 8×10¹⁰ cm⁻² s⁻¹`; at bubble density `n_H = 10⁻²` this is `U ≈ 3×10²`. The
   bubble interior is optically thin to LyC, so **the cluster's ionising field does stream through
   it and `U` there is enormous** — CIE for the metals is not automatic even at 10⁶ K, though at
   `T ≳ 10⁶` the relevant ions are collisionally stripped anyway. **[derived, high]**
3. **Optically thin** to its own line and continuum radiation (fails for the dense shell, where
   resonance-line trapping suppresses cooling).
4. **Low density**, `n_e ≪ n_crit`, so that level populations are collisionally-excited/
   radiatively-de-excited and `Λ` is density-*independent*. This **fails in the swept shell**: with
   `nCore = 10⁵ cm⁻³` and post-shock compression the shell reaches `n_H ≳ 10⁶–10⁸ cm⁻³`, far above
   `n_crit` for the forbidden lines that dominate at 10⁴ K (`[C II] 158 μm`, `n_crit ~ 3×10³`;
   `[O III] 5007`, `~7×10⁵`). Above `n_crit`, cooling per volume tends to `∝ n`, not `∝ n²`, so a
   low-density `Λ` over-predicts shell cooling by up to `n/n_crit`. **The CIE table's lack of an
   `n` axis is the physical reason it must not be used below ~10⁵·⁵ K, independently of the
   photoionisation argument.** **[derived, high]**

### 4.2 What the non-CIE cube must be, and its sign

The cube is a **net** quantity: `photoheating − cooling`, function of `(n_H, T, Φ, age, Z)`. It
**must be allowed to be positive** (net heating). The physically essential feature is the
**zero crossing**: `du/dt(T_eq) = 0` at `T_eq ≈ 10⁴ K` in photoionised gas, with `du/dt > 0` below
and `< 0` above — a *stable* thermal equilibrium. That zero crossing is what holds the ionised
shell layer at `TShell_ion = 10⁴ K` and hence what generates `P_HII`, TRINITY's headline term
(SPEC-029/030). A code that clips the net rate at `≤ 0` ("cooling only"), or takes `abs()`, or
takes a `log` of the net value, destroys `T_eq` and lets the ionised layer cool to ~10 K, killing
`P_HII`. **[derived, high]**

### 4.3 Continuity at the switch — the hard requirement

Let `T_s = 10^{5.5} K`. The two branches must satisfy

```
    lim_{T→T_s⁻}  du/dt |_nonCIE (n, T, Φ)   =   lim_{T→T_s⁺}  du/dt |_CIE (T)
```

for any `(n, Φ)` in the collision-dominated corner of the cube (`U → 0`, `n → 0`). Requirements a
correct implementation must meet:

1. **No gap and no overlap ambiguity.** `_cie_tcutoff(logT_CIE)` and `_noncie_cutoffs(...)` must
   produce a *single* partition of the T axis: the non-CIE upper bound and the CIE lower bound must
   coincide (or overlap with a documented, deterministic tie-break). A gap ⇒ some temperatures have
   no table; if that returns `0.0`, the code silently reports **zero cooling in a band**, which is
   the worst possible failure because it is smooth-looking and monotone.
2. **Amplitude agreement.** Independent tables (Gnat & Ferland CIE vs a CLOUDY/OPIATE cube) will
   not agree to better than ~20–30 % even when both are right; a *normalisation* mismatch (§1.2)
   shows up here as a factor 1.2 / 4.4 / 5.3 step. Testing `du/dt(T_s⁻)/du/dt(T_s⁺)` at low `Φ`
   is the cheapest possible detector for the whole of §1 and should land in `[0.7, 1.4]`.
3. **Why a jump is not cosmetic.** `T` runs continuously across `T_s` inside the bubble-structure
   integration and inside any shell temperature solve. A step in the ODE right-hand side (i) forces
   an adaptive stiff solver to shrink the step or chatter, (ii) can be straddled by a root-finder
   so that a bisection on `du/dt = 0` converges to the *discontinuity*, not to `T_eq`, and (iii)
   per `CLAUDE.md` there is a monotonic guard in the bubble-structure integrator that a step can
   trip. **[derived, high]**
4. **Continuity in age too.** `cube_linear_interpolate(x, ages, cubes)` implies the cubes *are*
   interpolated in age, which is correct and answers SPEC-083's concern — provided the weights are
   `w = (x − a₀)/(a₁ − a₀)`, sum to 1, and the same variable (linear age or log age) is used to
   form the bracket and the weight. Interpolating in log age while bracketing in linear age (or
   vice versa) is a smooth, silent O(10 %) error that no test will catch.

---

## 5. Interpolation and out-of-bounds

### 5.1 Variables

- **CIE:** interpolate **`log₁₀ Λ` linearly in `log₁₀ T`**. Justification: the physical asymptotes
  are power laws (`Λ_ff ∝ T^{1/2}`; the falling side `∝ T^{−1}`), which are *straight lines* in
  log–log — linear-in-log-log is then exact, not merely accurate. `Λ` spans ~5 decades over 4
  decades in `T`, so linear-in-`Λ` or linear-in-`T` is not a small correction.
- **Non-CIE cube:** node spacing in **`log n`, `log T`, `log Φ`** (all span many decades), but the
  *value* must be interpolated **linearly in the signed net rate** — you cannot take a log of a
  quantity that changes sign. This is an unavoidable asymmetry between the two branches and a place
  where a copy-pasted log-interpolation from the CIE path produces `NaN` on the heating side.
- **Age:** linear (or log-linear) between the two bracketing cube files.

### 5.2 Quantified interpolation error

Linear interpolation of `f = log₁₀Λ` on a node spacing `h = Δlog₁₀T` has max error
`|f''| h²/8`, so the relative error in `Λ` is `≈ ln10 · |f''| h²/8`. Near the 10⁵ K peak the
log–log curvature is `|f''| ≈ 3 dex⁻²`:

| `h` (dex) | max error in `Λ` | sign |
|---|---|---|
| 0.05 | 0.2 % | under |
| 0.10 | 0.9 % | under |
| 0.25 | 5.4 % | under |
| 0.50 | **22 %** | under |

The error near a concave peak is **one-signed (undershoot)**, so it does not average out over the
cooling integral — it biases `L_cool` systematically low, which biases the energy→momentum
transition systematically *late*. **[derived, high]**

**Interpolating linearly in *linear* `T` across a log-spaced table is quantitatively disastrous.**
On a decade-wide cell `[T₀, 10T₀]`, the geometric midpoint `10^{0.5}T₀` sits at fractional position
`(10^{0.5}−1)/9 = 0.240`. So a linear-in-T interpolant assigns 76 % of its weight to the *bottom*
node over the lower ~4/5 of the cell in log space: evaluating at `10⁵ K` between nodes at `10^{4.5}`
and `10^{5.5}` returns essentially `Λ(10^{4.5})`, missing the peak entirely. **[derived, high]**

**Cubic splines in log–log overshoot.** At the 10⁴ K cliff (slope +12 next to slope ~+3) a natural
cubic spline rings, and the overshoot on the low side can drive the interpolated `Λ` **negative** —
i.e. spurious *heating* from a pure-cooling table, with no bound on its magnitude. A monotone
(PCHIP) or plain linear-in-log-log interpolant is mandatory near the cliff and near the peak.
**[derived, high]**

### 5.3 What must happen outside the table bounds

Required behaviour, per axis, in order of preference: (1) extrapolate with the correct physical
asymptote; (2) clamp *and record/warn*, but only where the physics genuinely asymptotes; (3) raise.
**Never** silently clamp on an axis where the true function keeps moving steeply, and **never**
return `NaN` or `0.0` into the ODE right-hand side.

| axis / edge | correct behaviour | error if silently clamped |
|---|---|---|
| `T` **above** table max (10⁸ K) | extrapolate `Λ = Λ(T_max)·(T/T_max)^{1/2}` (bremsstrahlung) | mild: factor 3.2 low at 10⁹ K. **Clamping is tolerable here and only here on the T axis.** |
| `T` **below** table min (10⁴ K) | `Λ → 0` steeply (or hand off to a molecular/fine-structure coolant) | **catastrophic: up to 10⁴–10⁶ over-cooling.** Clamping returns the near-peak Lyα value for 100 K gas |
| `n` above cube max | true `Λ_eff` *decreases* per n² (LTE, `n > n_crit`); clamping the *rate coefficient* over-cools | over-cools the shell, potentially by orders of magnitude at `n ~ 10⁸` |
| `n` below cube min | the low-density limit is **exact** — clamping is physically correct | benign |
| `Φ` → 0 | the `U → 0` limit is CIE — clamping to the lowest-`Φ` slice is physically correct | benign, **and mandatory**: `log(0) = −inf` otherwise |
| `Φ` above cube max | heating grows ~linearly in `Φ`; clamping caps the heating | under-heats ⇒ the H II layer cools below 10⁴ K ⇒ `P_HII` collapses |
| `age` outside cube set | clamp to first/last cube (spectrum shape varies slowly) | benign, but must be recorded |

**Physical regimes in which an out-of-bounds query legitimately occurs in a bubble simulation** —
these are not pathological inputs, they are the normal operating envelope:

- **`T > 10⁸ K`:** the immediate post-wind-shock temperature is `T = 3μm_H v_w²/(16k_B)`
  = `5.5×10⁷ K` at `v_w = 2000 km/s` and **`1.2×10⁸ K` at `v_w = 3000 km/s`**. Young massive
  clusters and any SN-dominated interval reach this. Also `T_b ∝ t^{−6/35}` diverges as `t → 0`, so
  **the very first timesteps of every run** are the hottest.
- **`T < 10⁴ K`:** the outer neutral/molecular shell layer at 10–100 K (SPEC-002 zone 3), the
  undisturbed cloud, the shell after `Q_i` collapses (`t ≳ 5–10 Myr`), and the whole re-collapse
  fate (SPEC-017).
- **`n` above the cube's max:** `nCore = 10⁵ cm⁻³` (the *default*), and the swept shell is
  compressed above that — `n_H ~ 10⁶–10⁸ cm⁻³` is routine.
- **`n` below the cube's min:** the hot bubble interior, `n_H ~ 10⁻³–10⁻²`, and the free-streaming
  wind zone `n = Ṁ_w/(4πr²v_w μ m_H)`, lower still.
- **`Φ` above the max:** `Φ = Q_i/(4πR₂²) → ∞` as `R₂ → 0`; at `R₂ = 0.1 pc`, `Q_i = 10⁵¹`,
  `Φ = 8×10¹⁴ cm⁻² s⁻¹`. Every run starts here.
- **`Φ = 0` exactly:** an SPS table row with `Q_i = 0` at late age, or a fully absorbing shell.
- **`age` past the last cube:** runs integrate to tens of Myr; CLOUDY cube sets typically stop
  earlier.

Because every axis is legitimately exceeded, **a silent clamp is not a rare-corner concern — it is
the default behaviour of the code for a substantial fraction of all calls.** The only safe design
records that it happened (a counter/flag in the output) so the run can be judged. **[derived, high]**

---

## 6. Dimensions and the mandatory conversions

| quantity | cgs | AU (`M⊙, pc, Myr`) | cgs→AU factor |
|---|---|---|---|
| `Λ` (per two densities) | **erg cm³ s⁻¹** | `M⊙ pc⁵ Myr⁻³` | — |
| `n` | cm⁻³ | pc⁻³ | `1 cm⁻³ = 2.9380×10⁵⁵ pc⁻³` |
| `u` (energy density) = pressure | erg cm⁻³ = dyn cm⁻² | `M⊙ pc⁻¹ Myr⁻²` | `1 AU = 6.4721×10⁻¹³ erg cm⁻³` |
| **`du/dt`** | **erg cm⁻³ s⁻¹** | `M⊙ pc⁻¹ Myr⁻³` | **`1 AU = 2.0509×10⁻²⁶ erg cm⁻³ s⁻¹`; multiply cgs by `4.8760×10²⁵`** |
| `L_cool = −∫4πr²(du/dt)dr` | erg s⁻¹ | `M⊙ pc² Myr⁻³` | `1 AU = 6.0255×10²⁹ erg s⁻¹` |
| `T` | K | K | none |
| `Φ` | cm⁻² s⁻¹ | `pc⁻² Myr⁻¹` | `1 cm⁻² s⁻¹ = 2.9994×10⁵⁰ pc⁻² Myr⁻¹` |

(`u` and pressure share a unit — that identity is itself a check: the energy-density conversion
must equal the pressure conversion in SPEC-091, and it does: `1.90148e43/2.9380e55 = 6.4721e-13`.)

**Mandatory discipline.** The tables are cgs (`erg cm³ s⁻¹`, `K`, `cm⁻³`, `cm⁻² s⁻¹`); the dynamics
is AU. The only safe pattern is: **convert the caller's `(ndens, T, phi)` to cgs on entry, do the
whole cooling calculation in cgs, and convert the single scalar `du/dt` to AU exactly once on
exit.** Failure modes: (i) mixing `n` in cm⁻³ with `Λ` already converted to AU → error
`2.938e55² / 4.876e25` ≈ `10⁸⁵` (loud); (ii) converting once in `get_dudt` and again in the caller
→ `2.4×10⁵¹` (loud); (iii) **converting `n` but not `Φ`** → the `Φ` axis is off by `3×10⁵⁰`, which
lands off the top of the axis, gets clamped, and produces *maximal photoheating everywhere*
(**silent, and it inverts the sign of `du/dt`**). (iii) is the dangerous one. **[derived, high]**

Also dimension-adjacent: `Λ` in most files is stored as `log₁₀ Λ`, and `T` as `log₁₀ T`. Passing a
linear `T ~ 10⁶` into a `log₁₀T`-indexed interpolator queries `T = 10⁶` *in log units*, i.e. far
above the top of the axis ⇒ clamp ⇒ constant `Λ(10⁸ K)` returned for every temperature in the run.
Silent, and it flattens the cooling curve entirely. Same trap for `Φ`.

---

## 7. Known traps — consolidated checklist

| # | Trap | Signature to look for | Magnitude |
|---|---|---|---|
| T1 | `n_tot²` paired with an `n_e n_H` table | any `ndens**2` where `ndens` is total density | **×4.41** |
| T2 | `n_H²` paired with an `n_e n_H` table | missing `χ_e = 1.2` factor | ×0.833 |
| T3 | `(ρ/m_H)²` used as `n²` | `mu_convert = 1.4` leaking into the density product | ×1.633 |
| T4 | `ndens` is ambiguous (`n_H` vs `n_tot`) at the call site | `get_dudt(age, ndens, …)` — nothing in the name says which | ×5.29 per the pair |
| T5 | fully-ionised `χ_e = 1.2` used in the 10⁴ K branch where `χ_e ≈ 1.1`, or in neutral gas where `χ_e ~ 10⁻⁴` | one hard-coded electron fraction | ×1.09 → ×10⁴ |
| T6 | Z applied to a table that already has it | `Lambda * metallicity` on a solar table | ×Z (invisible at Z=1) |
| T7 | naive `Λ ∝ Z` instead of `Λ_HHe + Z Λ_metals` | no separate H/He term | ×10 at Z=0.1 in the brems and Lyα regimes |
| T8 | Z applied to `log₁₀Λ` rather than `Λ` | scaling the stored column | `Λ^Z`; 10¹¹ at Z=0.5 |
| T9 | linear-in-`T` (or linear-in-`Λ`) interpolation across the 10⁵ K peak | non-log axes | peak missed entirely; up to ×10 |
| T10 | cubic spline overshoot at the 10⁴ K cliff | spline instead of PCHIP/linear | `Λ < 0` ⇒ spurious heating |
| T11 | CIE curve used for photoionised 10⁴ K gas | switch temperature too low, or non-CIE branch unreachable | shell cools below `T_ion`; `P_HII` → 0 |
| T12 | **sign convention mismatch** between producer and consumer | CIE branch returns `+n²Λ` (a positive *cooling rate*) while the non-CIE cube returns `heating − cooling` (positive = heating) | `du/dt` has the **wrong sign above 10⁵·⁵ K**: the bubble is heated; `(L_gain−L_loss)/L_gain > 1` forever ⇒ **the energy→momentum transition never fires** |
| T13 | net rate clipped to `≤ 0` / `abs()` / `log()`-ed | any monotonicity assumption on a signed quantity | destroys `T_eq ≈ 10⁴ K` |
| T14 | silent clamp at the low-`T` edge | `np.clip` on the T axis | up to ×10⁶ over-cooling of cold gas |
| T15 | `Φ` not unit-converted / `log(0)` at `Q_i = 0` | clamp to max `Φ` ⇒ max heating | sign inversion of `du/dt` |
| T16 | discontinuity at the CIE/non-CIE handover | `_cie_tcutoff` and `_noncie_cutoffs` disagreeing | solver chatter; root-finder converges to the step |
| T17 | gap between the two tables returning `0.0` | a band with no coolant | silent, smooth, undetectable by eye |
| T18 | axis stored descending; `create_limits` uses `(a[0], a[-1])` not `(min, max)` | inverted bounds ⇒ everything "out of range" | total |
| T19 | age→cube weights formed in a different variable than the bracket | `cube_linear_interpolate` | smooth ~10 % |
| T20 | `get_fileage` mis-parsing the age unit (yr vs Myr vs `1e6 yr`) from the filename | wrong cube for the whole run | order-unity in the heating |
| T21 | cached `get_coolingStructure` keyed without `(Z, SB99_rotation, path)` | module-level global state (`CLAUDE.md` warns of this) | wrong table across a sweep |
| T22 | `cooling_boost_*` multiplier applied to the **signed net** rate | boosts photo*heating* by the same factor | inverts the intent of the knob |
| T23 | low-density `Λ` applied at shell densities `n > n_crit` | no `n` axis on the CIE branch | over-cools the shell by `n/n_crit` |

---

## 8. Cheapest decisive tests for the reconciler

1. **Normalisation ratio.** Read the CIE file header, take its stated normalisation, and compare
   with the code's density product. The answer is one of `{1.0, 0.833, 1.1, 1.2, 1.633, 4.41,
   5.29}`. Anything other than 1.0 is a finding.
2. **Switch continuity.** Evaluate `get_dudt` at `T = 10^{5.5}(1∓ε)` with a small `phi` and a
   low `ndens`. The ratio must be within ~1.4 of unity.
3. **Sign.** `get_dudt(age, n=1, T=1e6, phi=0)` must be **negative** and of order
   `−1.2 × (3×10⁻²³) = −3.6×10⁻²³ erg cm⁻³ s⁻¹` (`= −1.8×10⁻³ M⊙ pc⁻¹ Myr⁻³`).
4. **Bremsstrahlung tail.** `Λ(10⁸ K)` must be `≈ 2.4×10⁻²³ erg cm³ s⁻¹` per `n_e n_H`
   (`2.8×10⁻²³` per `n_H²`) and must scale as `T^{1/2}` above 3×10⁷ K.
5. **Equilibrium temperature.** With a photoionised-shell `(n, Φ)`, the root of
   `get_dudt = 0` must land near 10⁴ K.
6. **Z no-op.** `get_Lambda(T, interp, 1.0)` must equal the raw table value exactly; if it does
   not, the metallicity handling is already wrong at solar.
7. **Out-of-bounds.** Call at `T = 10², 10⁹`, `n = 10⁻⁶, 10⁹`, `Φ = 0, 10¹⁶`, `age` beyond the
   last cube. None may return `NaN`; the low-`T` and high-`Φ` cases must not return the edge value
   silently.

---

```json
[
  {
    "id": "S9-C-01",
    "file": "trinity/cooling/net_coolingcurve.py",
    "line": 58,
    "class": "coefficient",
    "severity": "S1",
    "claim": "The density product multiplying the CIE table must match the normalisation the table was published in. Pairing an n_e n_H table with n_tot^2 is a factor 4.41 over-cooling.",
    "evidence": "Two-body cooling gives dE/dV/dt = n_e * sum_i n_i * Lambda_i. Tables factor this as n_e n_H, n_e n_ion, n_H^2, or n_tot^2. For fully ionised cosmic gas with y = n_He/n_H = 0.1: n_e = 1.2 n_H, n_ion = 1.1 n_H, n_tot = 2.3 n_H. Hence n_tot^2/(n_e n_H) = 5.29/1.20 = 4.408; robust at 4.4 +/- 0.15 for y in [0.084, 0.1] or mu = 0.6. Other mispairings: n_H^2 vs n_e n_H = 0.833; (rho/m_H)^2 vs n_e n_H = 1.633; n_e n_ion vs n_e n_H = 1.1; n_H^2 vs n_tot^2 = 5.29.",
    "expected": "du/dt = -(n_e n_H) * Lambda_table if the table is per n_e n_H, with n_e = chi_e * n_H and chi_e appropriate to the ionisation state (1.2 hot, 1.1 in the 10^4 K H II layer). The code-side factor and the table's header normalisation must agree exactly.",
    "failure_scenario": "L_cool is wrong by 4.41x. L_cool enters the transition trigger (L_gain-L_loss)/L_gain <= 0.05 (SPEC-013) directly, so the energy->momentum transition time - the code's headline prediction - moves by a large factor, changing dispersal vs re-collapse outcomes across the whole published grid. Completely silent: every run still completes.",
    "repro": "Read the normalisation stated in the header of lib/default/.../coolingCIE_3_Gnat-Ferland2012.dat, then read the density factor applied in get_dudt. Ratio must be 1.0.",
    "confidence": "high"
  },
  {
    "id": "S9-C-02",
    "file": "trinity/cooling/net_coolingcurve.py",
    "line": 58,
    "class": "units",
    "severity": "S2",
    "claim": "The meaning of the `ndens` argument (hydrogen-nuclei density vs total particle density) must be fixed, documented, and consistent with the table normalisation and with every call site.",
    "evidence": "SPEC-003 declares nCore/nISM as hydrogen-nuclei densities and SPEC-092 puts n_H -> rho with mu_convert = 1.4 (mass per H nucleus) at the top of the error-prone list. But n_tot/n_H = 2.3 in the hot bubble, 2.2 in the ionised shell, 1.1 atomic, 0.6 molecular (SPEC-092). Passing n_tot where n_H is expected changes the density product by up to 5.29 and simultaneously shifts the query point on the cube's n axis by 0.36 dex.",
    "expected": "ndens is n_H (hydrogen nuclei per cm^3), cgs, everywhere; any n_e or n_tot is formed inside the cooling module from an explicit, regime-appropriate composition constant.",
    "failure_scenario": "A caller in the bubble-structure integrator passes n_tot while the shell path passes n_H (or vice versa); cooling is inconsistent between bubble and shell by up to 5.3x, with no error and no test failure.",
    "repro": "Compare what every call site of get_dudt passes as ndens against the composition constant used to form the density product inside.",
    "confidence": "high"
  },
  {
    "id": "S9-C-03",
    "file": "trinity/cooling/net_coolingcurve.py",
    "line": 58,
    "class": "sign",
    "severity": "S1",
    "claim": "The CIE branch and the non-CIE branch must return du/dt in the SAME sign convention: negative for net cooling, positive for net heating.",
    "evidence": "A CIE table is a pure cooling efficiency (always positive); a CLOUDY/OPIATE net cube is heating-minus-cooling and changes sign at T_eq ~ 10^4 K (SPEC-084). A producer/consumer mismatch is therefore structurally likely: the CIE branch naturally yields +n^2*Lambda (a cooling RATE, positive) while the cube naturally yields a signed net RATE OF CHANGE. Only one of these can be du/dt.",
    "expected": "get_dudt returns du/dt with u the thermal energy density, so du/dt < 0 whenever cooling dominates; the CIE branch must return -(n_e n_H)Lambda, i.e. an explicit minus sign, and the cube branch must return the table's net value with whatever sign flip the table's own convention requires.",
    "failure_scenario": "Above 10^5.5 K the bubble is HEATED rather than cooled. L_cool becomes negative, (L_gain - L_loss)/L_gain exceeds 1 permanently, the cooling_balance transition trigger never fires, every run stays energy-driven, and every cloud is predicted to disperse.",
    "repro": "get_dudt(age=1, ndens=1.0, T=1e6, phi=0) must be about -3.6e-23 erg/cm^3/s (-1.8e-3 M_sun/pc/Myr^3), i.e. NEGATIVE; and get_dudt at T=3e3 with a strong phi must be POSITIVE.",
    "confidence": "high"
  },
  {
    "id": "S9-C-04",
    "file": "trinity/cooling/net_coolingcurve.py",
    "line": 32,
    "class": "regime",
    "severity": "S2",
    "claim": "The CIE and non-CIE temperature ranges must partition the T axis with no gap; a temperature covered by neither table must not silently return zero.",
    "evidence": "_cie_tcutoff(logT_CIE) and _noncie_cutoffs(cooling_nonCIE) derive bounds from two independently produced datasets. SPEC-080 declares the split at 10^5.5 K, but the CIE file's own minimum log T and the cube's own maximum log T are whatever the file authors chose. If cube_max < cie_min there is an uncovered band.",
    "expected": "The two cutoffs coincide, or overlap with a deterministic documented tie-break. A query in an uncovered band raises, not returns 0.0.",
    "failure_scenario": "A band of temperatures returns du/dt = 0 (perfectly adiabatic gas). Because zero is smooth, monotone and physically plausible-looking, the bubble simply fails to cool in that band and the transition is delayed with no diagnostic whatsoever.",
    "repro": "Compare the value returned by _cie_tcutoff against the upper T bound from _noncie_cutoffs; then evaluate get_dudt on a fine T sweep across 10^5.3 to 10^5.7 and look for an exact-zero interval.",
    "confidence": "high"
  },
  {
    "id": "S9-C-05",
    "file": "trinity/cooling/net_coolingcurve.py",
    "line": 58,
    "class": "numerical",
    "severity": "S2",
    "claim": "du/dt must be continuous in T across the CIE/non-CIE switch to within table uncertainty (~30%) in the collision-dominated corner (low phi, low n).",
    "evidence": "T varies continuously across 10^5.5 K inside the bubble-structure integration (Weaver profile T(xi) = T_b (1-xi)^{2/5}) and inside any shell temperature solve. A step in the ODE RHS makes an adaptive stiff solver chatter, can make a bisection on du/dt = 0 converge to the discontinuity rather than to T_eq, and can trip the bubble-structure monotonic guard documented in CLAUDE.md. Physically, at low ionisation parameter the photoionised cube MUST reduce to the CIE curve, so any residual step is a code or normalisation artefact, not physics.",
    "expected": "ratio du/dt(T_s^-)/du/dt(T_s^+) in [0.7, 1.4] at low phi. A step of exactly 1.2, 4.41 or 5.29 is a direct fingerprint of S9-C-01.",
    "failure_scenario": "Solver chatter and step-size collapse near the conduction front; systematic error in L_cool; a monotonic-guard rejection that presents as an unrelated integrator failure.",
    "repro": "Evaluate get_dudt at T = 10^5.5*(1-1e-6) and 10^5.5*(1+1e-6) with phi=0, ndens=1e-2, and take the ratio.",
    "confidence": "high"
  },
  {
    "id": "S9-C-06",
    "file": "trinity/cooling/CIE/read_coolingcurve.py",
    "line": 25,
    "class": "coefficient",
    "severity": "S2",
    "claim": "Metallicity must enter as Lambda(T,Z) = Lambda_HHe(T) + (Z/Zsun) Lambda_metals,sun(T), not as a bare multiplier on a total solar curve.",
    "evidence": "In optically thin CIE each line's emissivity is linear in its ion's abundance and the ionisation balance of one element is independent of the others' abundances, so the cooling function separates additively into a Z-independent H/He + bremsstrahlung part and a Z-linear metal part. Bremsstrahlung (Lambda_ff = 2.35e-27 sqrt(T), derived from eps_ff = 1.4e-27 T^{1/2} n_e sum n_i Z_i^2 g with sum n_i Z_i^2 = 1.4 n_H, g = 1.2) carries the ENTIRE budget above ~3e7 K and does not scale with Z at all; H Lya dominates near 2e4 K and likewise does not.",
    "expected": "A separate H/He floor, or per-Z tables selected by the metallicity argument, or an explicit validation that Z == 1. A naive Lambda *= Z is only correct on a metals-only table.",
    "failure_scenario": "At Z = 0.1 Zsun the naive scaling under-predicts Lambda(1e8 K) and Lambda(2e4 K) by a factor 10 while being roughly right at the 1e5 K peak - so it looks calibrated where it is checked and is an order of magnitude wrong in the bubble. Invisible in every shipped run because SPEC-085 says only Z=1 is supported, yet docs/source/running.rst advertises sweeping ZCloud = [0.5, 1.0].",
    "repro": "get_Lambda(T, interp, 1.0) must equal the raw table value bit-for-bit; get_Lambda(1e8, interp, 0.1) must be within ~10% of get_Lambda(1e8, interp, 1.0), not 10x smaller.",
    "confidence": "high"
  },
  {
    "id": "S9-C-07",
    "file": "trinity/cooling/CIE/read_coolingcurve.py",
    "line": 25,
    "class": "coefficient",
    "severity": "S1",
    "claim": "A metallicity factor must not be applied to a table that already encodes that metallicity, and must not be applied to a log-stored column.",
    "evidence": "coolingCIE_3_Gnat-Ferland2012.dat is a table AT a metallicity (solar). Applying (Z/Zsun) on top double-counts. Separately, CIE tables conventionally store log10(Lambda); scaling the stored log gives Lambda^Z - identity at Z=1 and, since log10(Lambda) ~ -22, a factor 1e11 over-cooling at Z=0.5.",
    "expected": "Exactly one of: (a) metallicity selects a per-Z table or a Z axis; (b) metallicity multiplies a metals-only component in linear space; (c) metallicity is validated == 1.0 and raises otherwise. Never (a)+(b) together.",
    "failure_scenario": "Dormant landmine: every shipped run has Z=1 so the bug is a no-op; the first ZCloud sweep produces cooling wrong by Z or by 10^(22(1-Z)).",
    "repro": "Check whether get_filename() also selects on metallicity (it takes a metallicity argument) while get_Lambda applies a Z factor - that combination is the double-count.",
    "confidence": "high"
  },
  {
    "id": "S9-C-08",
    "file": "trinity/cooling/CIE/read_coolingcurve.py",
    "line": 25,
    "class": "units",
    "severity": "S2",
    "claim": "The metallicity argument must be linear Z/Zsun, not [Z/H] in dex, and the choice must match the parameter-file declaration.",
    "evidence": "ZCloud is declared in Zsun units with default 1 (SPEC-003). If a dex value were passed and used linearly, Zsun -> 0 and Lambda == 0 identically; if a linear value is used as a dex offset, the error is 10^Z. The two conventions disagree even at solar (1 vs 0), so this one is detectable in the default configuration.",
    "expected": "metallicity is dimensionless Z/Zsun with 1.0 == solar; get_Lambda(T, interp, 1.0) is the identity.",
    "failure_scenario": "Lambda identically zero: the bubble never cools, every run stays energy-driven, every cloud disperses.",
    "repro": "get_Lambda(1e6, interp, 1.0) vs the raw table value at log10 T = 6.",
    "confidence": "high"
  },
  {
    "id": "S9-C-09",
    "file": "trinity/cooling/CIE/read_coolingcurve.py",
    "line": 25,
    "class": "numerical",
    "severity": "S2",
    "claim": "Lambda must be interpolated as log10(Lambda) linear in log10(T), never linear in T or linear in Lambda.",
    "evidence": "Lambda spans ~5 decades over 4 decades in T; its asymptotes are power laws (Lambda_ff ~ T^{1/2}; the falling side ~ T^{-1}) which are exactly straight in log-log, so linear-in-log-log is exact there. Quantitatively, on a decade-wide cell [T0, 10 T0] the geometric midpoint sits at fractional position (10^0.5 - 1)/9 = 0.240, so a linear-in-T interpolant evaluated at 10^5 K between nodes at 10^4.5 and 10^5.5 returns essentially Lambda(10^4.5) and misses the 10^5 K peak entirely.",
    "expected": "log-log linear (or a monotone log-log interpolant). Both the abscissa passed in and the axis stored must be in the same space - passing linear T into a log10-T-indexed interpolator queries far above the axis top and, after clamping, returns Lambda(1e8 K) for every temperature in the run.",
    "failure_scenario": "The dominant cooling feature of the entire curve is skipped; L_cool is wrong by up to an order of magnitude, systematically low, and the transition fires late. If linear T is passed to a log axis, the curve is flattened to a constant - completely silent.",
    "repro": "Evaluate get_Lambda on a dense T grid and check the recovered log-log slope: it must be +0.5 above 3e7 K and must show a maximum near log10 T = 5.0-5.3.",
    "confidence": "high"
  },
  {
    "id": "S9-C-10",
    "file": "trinity/cooling/CIE/read_coolingcurve.py",
    "line": 25,
    "class": "numerical",
    "severity": "S3",
    "claim": "The interpolant must be shape-preserving near the 10^4 K cliff and the 10^5 K peak; an unconstrained cubic spline can return Lambda < 0.",
    "evidence": "At 10^4 K the local log-log slope is d lnLambda/d lnT = chi_Lya/kT = 1.18e5/1e4 = 11.8, adjacent to slopes of order +3. A natural cubic spline through nodes with that slope change rings; the undershoot on the low side is unbounded and can go negative. Negative Lambda from a pure-cooling table is spurious heating.",
    "expected": "linear-in-log-log or PCHIP; alternatively an explicit assertion Lambda > 0 on every returned value.",
    "failure_scenario": "Spurious heating of cold gas near the table's low-T edge; possible runaway in the shell temperature solve.",
    "repro": "Sample get_Lambda densely between the two lowest table nodes and check for negative or non-monotone values.",
    "confidence": "medium"
  },
  {
    "id": "S9-C-11",
    "file": "trinity/cooling/CIE/read_coolingcurve.py",
    "line": 25,
    "class": "silent-failure",
    "severity": "S1",
    "claim": "Clamping T to the LOW edge of the CIE table (10^4 K) is unsafe and must not happen silently; clamping to the HIGH edge is tolerable but must extrapolate as T^{1/2}.",
    "evidence": "Below 10^4 K, CIE Lambda collapses by 4-6 decades (governed by exp(-1.18e5/T)); Lambda(10^4 K) is near the Lya peak. Clamping therefore over-cools 100 K gas by up to 10^6. Above 10^8 K the true behaviour is bremsstrahlung Lambda = 2.35e-27 sqrt(T), so clamping under-predicts by only sqrt(T/T_max) - a factor 3.2 at 10^9 K. The asymmetry is 5-6 orders of magnitude.",
    "expected": "T > T_max: extrapolate Lambda(T_max)*(T/T_max)^{1/2}. T < T_min: hand off to the non-CIE branch, or return ~0, or raise - never return Lambda(T_min).",
    "failure_scenario": "The neutral/molecular outer shell layer (10-100 K, SPEC-002 zone 3) and the undisturbed cloud are assigned near-peak cooling efficiency; the shell's energy budget is drained and any shell temperature solve is driven to nonsense - silently, because the number returned is finite and positive.",
    "repro": "get_Lambda(100.0, interp, 1.0) and get_Lambda(1e9, interp, 1.0): the first must not equal get_Lambda(1e4,...); the second must be about 3.2x get_Lambda(1e8,...).",
    "confidence": "high"
  },
  {
    "id": "S9-C-12",
    "file": "trinity/cooling/net_coolingcurve.py",
    "line": 58,
    "class": "regime",
    "severity": "S2",
    "claim": "Out-of-bounds queries on every axis (T, n, phi, age) occur in the normal operating envelope of a bubble run and must be detected and recorded, not silently clamped.",
    "evidence": "T > 1e8 K: post-wind-shock T = 3 mu m_H v_w^2/(16 k_B) = 5.5e7 K at v_w = 2000 km/s and 1.2e8 K at 3000 km/s; also T_b ~ t^{-6/35} diverges as t->0, so the first timesteps of EVERY run are the hottest. T < 1e4 K: the neutral/molecular shell layer at 10-100 K, the undisturbed cloud, the shell after Q_i collapses (t >~ 5-10 Myr), and the whole re-collapse fate. n above the cube max: nCore = 1e5 cm^-3 is the DEFAULT and the shell is compressed above it (1e6-1e8). n below the cube min: the bubble interior at 1e-3 to 1e-2 and the free-wind zone below that. phi above the max: phi = Q_i/(4 pi R2^2) = 8e14 cm^-2 s^-1 at R2 = 0.1 pc, Q_i = 1e51 - i.e. every run's first steps. phi = 0 exactly: late ages where Q_i -> 0. age past the last cube: runs integrate to tens of Myr.",
    "expected": "Per-axis policy: extrapolate with the physical asymptote where one exists (T^{1/2} at high T; heating ~ linear in phi); clamp only where the physics genuinely asymptotes (low n = exact low-density limit; phi -> 0 = the CIE limit, and clamping there is MANDATORY to avoid log(0)); and record every clamp so the run can be judged.",
    "failure_scenario": "Because every axis is legitimately exceeded, silent clamping is not a corner case but the default behaviour for a large fraction of all calls; the run's cooling is quietly evaluated at the table edge over whole phases with nothing in the output to say so.",
    "repro": "Instrument get_dudt with a counter per axis per direction and run param/simple_cluster.param; the counts will be non-zero.",
    "confidence": "high"
  },
  {
    "id": "S9-C-13",
    "file": "trinity/cooling/non_CIE/read_cloudy.py",
    "line": 87,
    "class": "silent-failure",
    "severity": "S2",
    "claim": "phi = 0 must map to the lowest-phi slice of the cube (which is the physically correct CIE limit), never to log(0) = -inf or to a clamp at the TOP of the phi axis.",
    "evidence": "The cube's phi axis is almost certainly stored as log10(phi) because phi spans ~8 decades (8e6 to 8e14 cm^-2 s^-1 over a run). Q_i falls by several decades after ~10 Myr and can be exactly 0 in an SPS row. Separately, if phi is passed in AU (pc^-2 Myr^-1) into a cgs-indexed axis, the value is 3.0e50 times too large, lands off the top of the axis, and clamps to MAXIMUM photoheating everywhere - which inverts the sign of du/dt.",
    "expected": "phi converted to cgs (cm^-2 s^-1) on entry; phi <= phi_min maps to the phi_min slice; no log of a zero or negative argument reaches the interpolator.",
    "failure_scenario": "Either NaN propagating into the ODE RHS (loud), or maximal photoheating applied to gas that sees no photons (silent and sign-inverting: the shell is heated instead of cooled, T_eq is never reached, and P_HII is generated from an artefact).",
    "repro": "get_dudt(age, ndens=1.0, T=1e4, phi=0.0) must be finite and must equal the collisional (CIE-like) limit, i.e. negative.",
    "confidence": "high"
  },
  {
    "id": "S9-C-14",
    "file": "trinity/cooling/non_CIE/read_cloudy.py",
    "line": 104,
    "class": "state",
    "severity": "S2",
    "claim": "The non-CIE cube must be interpolated linearly in the SIGNED net value; log-space interpolation of the cube's values is impossible because the net rate changes sign at T_eq.",
    "evidence": "The cube is heating-minus-cooling (SPEC-084) and its zero crossing at T_eq ~ 1e4 K is the physically essential feature - it is what holds the ionised shell layer at TShell_ion = 1e4 K and therefore what generates P_HII, TRINITY's headline term (SPEC-029/030). log(x) of a signed quantity is NaN for x < 0; abs() destroys the crossing; clipping at <= 0 removes heating entirely.",
    "expected": "Node spacing in log n, log T, log phi (each spans many decades) but VALUE interpolation linear in the signed rate. The zero crossing must be resolved by the T grid - linear interpolation between a strongly heating node and a strongly cooling node places T_eq incorrectly in proportion to the node spacing.",
    "failure_scenario": "NaN on the heating side (loud) or a destroyed thermal equilibrium (silent): the photoionised layer cools to ~10 K, P_HII collapses, and the code's central claimed contribution over WARPFIELD vanishes.",
    "repro": "Solve get_dudt(age, n_shell, T, phi_shell) = 0 for T with a shell-like (n, phi); the root must land near 1e4 K.",
    "confidence": "high"
  },
  {
    "id": "S9-C-15",
    "file": "trinity/cooling/non_CIE/read_cloudy.py",
    "line": 87,
    "class": "numerical",
    "severity": "S3",
    "claim": "Age interpolation between cubes must form the bracket and the weights in the same variable, with weights summing to 1.",
    "evidence": "cube_linear_interpolate(x, ages, cubes) implies age IS interpolated (this answers SPEC-083's piecewise-constant concern favourably). Correctness requires w = (x - a0)/(a1 - a0) with the same 'age' variable used to locate a0, a1. Bracketing in linear age while weighting in log age (or vice versa) is a smooth O(10%) error with no discontinuity to reveal it.",
    "expected": "Consistent variable; w in [0,1]; w0 + w1 == 1; x outside [ages[0], ages[-1]] clamps to the first/last cube (defensible, since the ionising spectrum shape varies slowly) but is recorded.",
    "failure_scenario": "A smooth, ~10% systematic error in the photoheating term across the whole run, invisible to any continuity test.",
    "repro": "Call cube_linear_interpolate at x exactly equal to a tabulated age and check the result equals that cube exactly; then at the geometric mean of two ages and check it equals the arithmetic-mean weighting.",
    "confidence": "medium"
  },
  {
    "id": "S9-C-16",
    "file": "trinity/cooling/non_CIE/read_cloudy.py",
    "line": 204,
    "class": "silent-failure",
    "severity": "S3",
    "claim": "create_limits must return (min, max) of the axis and must assert the axis is strictly monotone increasing; taking (array[0], array[-1]) breaks on a descending axis.",
    "evidence": "CLOUDY output files commonly list temperature or density grids in descending order. If limits are formed as (first, last) they come out inverted, so every in-range test is inverted: either every query is flagged out of bounds, or none is. Independently, an interpolator built on a non-monotone axis returns arbitrary values.",
    "expected": "sorted, strictly-increasing axes (with the datacube reordered to match) and limits = (min, max); an explicit monotonicity assertion.",
    "failure_scenario": "Either total failure (loud) or a bounds check that never fires, in which case S9-C-12's clamping goes entirely unnoticed.",
    "repro": "Check the ordering of the ndens/temp/phi axes as read from the shipped cube files against how create_limits derives the bounds.",
    "confidence": "medium"
  },
  {
    "id": "S9-C-17",
    "file": "trinity/cooling/non_CIE/read_cloudy.py",
    "line": 270,
    "class": "state",
    "severity": "S3",
    "claim": "The cooling structure cache must be keyed on every input that selects a table: metallicity, SB99_rotation, path2cooling, and the age set.",
    "evidence": "get_coolingStructure(params) reads and parses many files and is called from the ODE RHS, so it is certainly cached. CLAUDE.md explicitly warns that trinity leaks module-level global state in-process, and run.py supports parameter sweeps that vary ZCloud and SB99_rotation within one worker process.",
    "expected": "Cache key includes (metallicity, SB99_rotation, path2cooling); or the cache is per-run and cleared between sweep combinations.",
    "failure_scenario": "In a sweep over ZCloud or SB99_rotation run in one process, the second and subsequent combinations silently reuse the first combination's cooling cubes. Results differ between --workers 1 and --workers N, and between a sweep and the same points run individually - the classic in-process-state bug this project already warns about.",
    "repro": "Run two sweep points with different ZCloud in a single process and compare the cube identity/ages actually used.",
    "confidence": "medium"
  },
  {
    "id": "S9-C-18",
    "file": "trinity/cooling/non_CIE/read_cloudy.py",
    "line": 349,
    "class": "units",
    "severity": "S3",
    "claim": "get_fileage must parse the filename age in a unit consistent with the `age` argument threaded through get_dudt -> get_filename -> cube_linear_interpolate.",
    "evidence": "TRINITY works internally in Myr (SPEC-090) while CLOUDY/OPIATE grids are commonly labelled in yr or in units of 1e6 yr. A factor 1e6 mismatch pushes every query past the end of the age list, where it clamps to a single cube for the entire run.",
    "expected": "One documented unit for age throughout the cooling module (Myr, matching the AU system), applied at the filename-parsing boundary.",
    "failure_scenario": "The whole run uses the youngest (or oldest) cube: the ionising spectrum's hardness is frozen, photoheating is systematically wrong, and the error is constant so nothing discontinuous ever appears.",
    "repro": "Compare the values returned by get_fileage over the shipped file set against the age values passed in from the solver.",
    "confidence": "medium"
  },
  {
    "id": "S9-C-19",
    "file": "trinity/cooling/net_coolingcurve.py",
    "line": 58,
    "class": "units",
    "severity": "S2",
    "claim": "The cgs<->AU conversion of du/dt must be applied exactly once, and the inputs (ndens, T, phi) must all be converted to cgs on entry - including phi.",
    "evidence": "Derived from SPEC-091: energy density unit M_sun pc^-1 Myr^-2 = 1.90148e43/2.9380e55 = 6.4721e-13 erg cm^-3 (identical to the pressure conversion, as it must be); dividing by Myr gives the rate unit M_sun pc^-1 Myr^-3 = 2.0509e-26 erg cm^-3 s^-1, so cgs -> AU is a multiply by 4.8760e25. Number density: 1 cm^-3 = 2.9380e55 pc^-3. Photon flux: 1 cm^-2 s^-1 = 2.9994e50 pc^-2 Myr^-1.",
    "expected": "Convert inputs to cgs on entry, compute entirely in cgs, convert the single scalar du/dt to AU once on exit by 4.8760e25.",
    "failure_scenario": "Double conversion (2.4e51) or mixed n/Lambda units (~1e85) fail loudly; but converting n while forgetting phi is SILENT: the phi query lands 50 decades above the axis, clamps to maximum photoheating, and inverts the sign of du/dt everywhere.",
    "repro": "Check the units of every argument at each call site of get_dudt against the units the table axes are stored in.",
    "confidence": "high"
  },
  {
    "id": "S9-C-20",
    "file": "trinity/cooling/net_coolingcurve.py",
    "line": 58,
    "class": "regime",
    "severity": "S3",
    "claim": "Any cooling_boost multiplier must be applied to the COOLING component only, not to the signed net rate.",
    "evidence": "SPEC-015: cooling_boost_mode/fmix/theta/kappa/fA exist to represent turbulent mixing across a fractal contact discontinuity enhancing bubble ENERGY LOSS (El-Badry+19, Lancaster+21). Applied to a signed net rate that includes photoheating, a multiplier fmix = 4 (the value shipped in param/paperII_grid_sweep.param) would amplify photoheating by 4 as well, which is the opposite of the knob's physical intent and can flip the sign of the net rate's zero crossing.",
    "expected": "The boost multiplies the cooling term; the heating term is untouched. If the boost is applied downstream on L_cool rather than inside get_dudt, that is fine and this expectation is vacuous - which is itself worth recording.",
    "failure_scenario": "With cooling_boost_fmix = 4, the photoionised layer's equilibrium temperature is shifted and P_HII is mis-set in exactly the runs used for Paper II.",
    "repro": "Trace where cooling_boost_* is consumed relative to the sign of the quantity it multiplies.",
    "confidence": "medium"
  },
  {
    "id": "S9-C-21",
    "file": "trinity/cooling/CIE/read_coolingcurve.py",
    "line": 25,
    "class": "regime",
    "severity": "S3",
    "claim": "A CIE curve has no density axis and is only valid in the low-density, optically-thin, ionisation-equilibrium limit; it must not be applied at swept-shell densities.",
    "evidence": "CIE Lambda is density-independent only for n_e << n_crit. nCore = 1e5 cm^-3 is the DEFAULT and post-shock compression pushes the shell to 1e6-1e8 cm^-3, far above n_crit for the forbidden lines that dominate at 1e4 K ([C II] 158 um ~ 3e3; [O III] 5007 ~ 7e5). Above n_crit, cooling per volume tends to ~n rather than ~n^2, so a low-density Lambda over-predicts by up to n/n_crit. Separately, t_rec/t_cool = (2/3) n_H Lambda/(alpha n_tot k T) is density-INDEPENDENT and equals ~16 at 1e5 K, so ionisation equilibrium itself fails in the 1e4.5-1e5.5 K band - exactly the conduction front of a Weaver bubble (Gnat & Sternberg 2007 find non-equilibrium Lambda suppressed by up to ~2x there).",
    "expected": "The CIE branch is restricted to T > 10^5.5 K (which SPEC-080 declares) AND to low-density gas; the dense shell is handled by the n-resolved non-CIE cube. Note the corollary: with bubble_xi_Tb = 0.98 the coldest gas in the integrated bubble domain is 0.02^0.4 = 0.209 T_b, i.e. >~ 2e6 K, so the CIE branch should carry essentially all bubble cooling and the cube should be exercised by shell/HII gas - a testable statement about which branch is hit where.",
    "failure_scenario": "The dense shell is over-cooled by up to n/n_crit; the bubble's conduction front is over-cooled by up to ~2x from the CIE assumption; both bias the transition early. Neither produces any error.",
    "repro": "Instrument which branch is taken vs (T, n) over a full run of param/simple_cluster.param and histogram the (T, n) of every CIE-branch call.",
    "confidence": "medium"
  },
  {
    "id": "S9-C-22",
    "file": "trinity/cooling/CIE/read_coolingcurve.py",
    "line": 25,
    "class": "coefficient",
    "severity": "S3",
    "claim": "The returned Lambda must reproduce the known shape of the solar CIE curve: a bremsstrahlung tail Lambda = 2.35e-27 sqrt(T) erg cm^3 s^-1 per n_e n_H above ~3e7 K, and a line-cooling maximum near log10 T = 5.0-5.3.",
    "evidence": "Derived: eps_ff = 1.4e-27 T^{1/2} n_e sum_i n_i Z_i^2 g_ff with sum n_i Z_i^2 = n_H(1 + 4y) = 1.4 n_H and g_ff = 1.2 gives Lambda_ff = 2.35e-27 sqrt(T) per n_e n_H (2.82e-27 sqrt(T) per n_H^2), i.e. 2.35e-23 at 1e8 K and 7.4e-24 at 1e7 K. Peak value (2-4)e-22 near 1e5 K is recalled, medium confidence; its LOCATION is high confidence.",
    "expected": "get_Lambda(1e8, interp, 1.0) ~ 2.4e-23 (or 2.8e-23 if the table is per n_H^2) and log-log slope +0.5 above 3e7 K; a maximum in log10 T between 5.0 and 5.3.",
    "failure_scenario": "If the returned tail is a factor 1.2 off, that alone identifies the table's normalisation and hence resolves S9-C-01. If the maximum is absent or displaced, the interpolation is at fault (S9-C-09).",
    "repro": "Dense T sweep of get_Lambda; check the absolute value at 1e8 K, the slope above 3e7 K, and the argmax.",
    "confidence": "medium"
  },
  {
    "id": "S9-C-23",
    "file": "trinity/cooling/net_coolingcurve.py",
    "line": 58,
    "class": "regime",
    "severity": "S3",
    "claim": "The 10^5.5 K split places the dominant cooling peak on the NON-CIE branch, so the cube's low-ionisation-parameter limit must reproduce the CIE peak.",
    "evidence": "The main metal-line peak of the CIE curve sits at log10 T = 5.0-5.3, BELOW the declared 10^5.5 K split (SPEC-080). Therefore the single most important feature of Lambda(T) for bubble energetics is evaluated from a photoionisation-equilibrium CLOUDY cube, under a radiation field that does not apply to shielded collisional gas. Physically the cube must reduce to CIE as U = phi/(n c) -> 0; if it does not (e.g. because its lowest phi slice still has appreciable U), the peak is mis-evaluated.",
    "expected": "cube net rate at (low n, low phi, T = 1e5) approximately equals -(n_e n_H) Lambda_CIE(1e5) from the same normalisation, within ~30%.",
    "failure_scenario": "The peak of the cooling curve - which dominates L_cool because the bubble's cooling integrand peaks where T sweeps through it - is systematically wrong, shifting the energy->momentum transition time.",
    "repro": "Compare the cube's lowest-phi slice at T = 1e5 K against an independently normalised CIE value; also record the minimum ionisation parameter actually spanned by the cube's phi and n axes.",
    "confidence": "medium"
  }
]
```
