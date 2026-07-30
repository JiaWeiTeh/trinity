# S10 SPS feedback — Lens C (what it should be)

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

**Slice:** SPS table ingestion + feedback update (`trinity/sps/read_sps.py`,
`trinity/sps/sps_columns.py`, `trinity/sps/update_feedback.py`).
**Method:** derived from first principles + `PHYSICS_SPEC.md` (SPEC-070…074, SPEC-090…092,
SPEC-025, SPEC-035, SPEC-026, SPEC-029) + recalled SB99 conventions. I have **not** read the
implementation, its comments, or the `lib/` tables. Literature access blocked; every SB99
*file-layout* statement below is flagged **[recalled]** and every *physical relation* is flagged
**[derived]**.

**Confidence discipline used here:**
- **[derived]** = follows from definitions/dimensional analysis; I stand behind it firmly.
- **[recalled]** = from memory of SB99 output conventions; treat as a hypothesis to check against
  the actual table header, not as ground truth.
- **[estimate]** = order-of-magnitude anchor, ±0.3–0.5 dex, usable only as a gross-error gate.

---

## A. The exact identities (the strongest test available on this layer)

### A.1 Definitions

A steady, spherically symmetric outflow that carries mass away from the cluster at rate `Ṁ` and
asymptotes to speed `v` has, through any sphere in the free-streaming region:

```
    mass flux      :  Ṁ                                    [M yr⁻¹]
    momentum flux  :  ṗ = Ṁ v                              [M L T⁻²]  (= a force)
    kinetic-energy :  L = ½ Ṁ v²                           [M L² T⁻³]  (= a power)
```

These are definitions, not models. **[derived, high]**

### A.2 The four quantities have only two degrees of freedom

`{L, ṗ, Ṁ, v}` are related by two independent equations, so **exactly two are free**. All four
closures:

```
    v  = 2 L / ṗ                    ṗ = 2 L / v
    Ṁ  = ṗ² / (2 L)                 L = ṗ v / 2  =  ½ Ṁ v²
    ṗ  = sqrt(2 L Ṁ)                v = sqrt(2 L / Ṁ)
```

**This is the single most valuable check on a table-reading layer.** If the loader admits more than
two of `{Lmech_W, pdot_W, Mdot_W, v_W}` (or the SN analogues `{Lmech_SN, pdot_SN, Mdot_SN, v_SN}` —
and SPEC-070 says *all four* SN canonicals are accepted) as **independent** columns, the redundancy
must be either (i) checked to table precision and rejected/warned on failure, or (ii) explicitly
resolved by declaring which pair is primary. Silently accepting an inconsistent quadruple means the
bubble's energy equation (SPEC-035) and its momentum equation (SPEC-020) are being driven by two
*different* physical winds. **[derived, high]**

### A.3 Dimensional verification in TRINITY's AU system

Using SPEC-091 (`M⊙, pc, Myr`):

| Identity | LHS units | RHS units | ✓ |
|---|---|---|---|
| `v = 2L/ṗ` | `pc Myr⁻¹` | `(M⊙ pc² Myr⁻³)/(M⊙ pc Myr⁻²)` = `pc Myr⁻¹` | ✓ |
| `Ṁ = ṗ²/(2L)` | `M⊙ Myr⁻¹` | `(M⊙ pc Myr⁻²)²/(M⊙ pc² Myr⁻³)` = `M⊙ Myr⁻¹` | ✓ |
| `ṗ = sqrt(2LṀ)` | `M⊙ pc Myr⁻²` | `sqrt(M⊙ pc² Myr⁻³ · M⊙ Myr⁻¹)` = `M⊙ pc Myr⁻²` | ✓ |

**Free corollary — a test on the conversion constants themselves.** With `C_X` ≡ (cgs value of one
AU unit of X), the identities force

```
    C_L / C_p  =  C_v        6.0255e29 / 6.1623e24 = 9.7781e4 cm/s   ✓ (= 0.977781 km/s)
    C_p² / C_L =  C_Ṁ        (6.1623e24)² / 6.0255e29 = 6.302e19 g/s ✓ (= 1 M⊙/Myr)
```

Both hold to the digits given in SPEC-091. Therefore **`v` and `Ṁ` derived in AU from AU-converted
`L, ṗ` must agree with the same quantities derived in cgs and then converted** — bit-for-bit up to
rounding. Any disagreement is a unit-constant bug, not a physics bug. **[derived, high]**

### A.4 How a reconciler tests this numerically

1. **Row-wise redundancy.** For every row of the ingested table, form
   `r₁ = 2·L_w / (ṗ_w · v_w) − 1` and `r₂ = ṗ_w/(Ṁ_w v_w) − 1`. If both `v_w`/`Ṁ_w` are *derived*
   from `(L,ṗ)`, `|r| < 1e-12` (floating point). If any is read **independently** from the table,
   `|r| ≲ 3e-3` is the most that log-column rounding can excuse (SB99 prints logs to 3 decimals ⇒
   0.115% per quantity ⇒ ~0.3% on a three-quantity ratio). `|r| > 1%` ⇒ genuine inconsistency;
   `|r| ~ 1` ⇒ a factor-of-2 (the `½`) or a swapped column. **[derived, high]**
2. **Scaling invariance.** Recompute the table with `f_mass` and `10·f_mass`. `v_w(t)` must be
   **identical**; `L, ṗ, Ṁ, Q, L_bol` must each be exactly `10×`. (Section C.)
3. **Range gate.** `v_w` must land in `10²–10⁴ km/s` = `10²–10⁴ pc/Myr` (1 km/s = 1.0227 pc/Myr).
   A hard guard `0 ≤ v_w < c = 3.0664e5 pc/Myr` must never be violated — violating it is the
   signature of `L_bol` substituted for `L_w` (Section F.4).
4. **Cauchy–Schwarz ordering (populations, not single stars).** For a *population* with
   `L = ½Σ Ṁᵢvᵢ²`, `ṗ = Σ Ṁᵢvᵢ`, `Ṁ_tot = Σ Ṁᵢ`, define
   `v_eff ≡ 2L/ṗ`, `v_rms ≡ sqrt(2L/Ṁ_tot)`, `v_mean ≡ ṗ/Ṁ_tot`. Cauchy–Schwarz gives
   `(Σ Ṁv)² ≤ (Σ Ṁ)(Σ Ṁv²)`, hence **`v_eff ≥ v_rms ≥ v_mean`** and
   **`Ṁ_eff ≡ ṗ²/(2L) ≤ Ṁ_tot`**, with equality iff the population is monokinetic.
   *Consequence:* if the code takes `Ṁ` from a table column (true total mass return) **and** `v`
   from `2L/ṗ`, then `L ≠ ½Ṁv²` — necessarily, by an amount that is a *diagnostic of the velocity
   spread*, not a bug per se. But it means `Ṁ`, `v` and `L` cannot all three be used in the same
   formula without saying which is being sacrificed. `ρ_w(r) = Ṁ/(4πr²v)` in particular is
   ambiguous by that same factor. **[derived, high]**
5. **The velocity-free formulas are the safe ones.** `R1 = sqrt(ṗ_w/(4πP_b))` (SPEC-025) and the
   free-wind ram pressure `ρ_w v_w² = ṗ_w/(4πr²)` depend on `ṗ` alone and are immune to the
   `Ṁ`/`v` ambiguity. Any place the code needs `ρ_w` or `v_w` *separately* is where the ambiguity
   bites. **[derived, high]**

### A.5 Combining winds and SNe

Momenta and powers are additive; velocities are **not**:

```
    L_tot = L_W + L_SN        ṗ_tot = ṗ_W + ṗ_SN        Ṁ_tot = Ṁ_W + Ṁ_SN
    v_tot = 2(L_W + L_SN)/(ṗ_W + ṗ_SN)     ≠  any weighted mean of (v_W, v_SN) in general
```

Requirement: a single effective injection velocity used downstream must be built from the **totals**,
consistently. Using `v_W` (winds only) together with `ṗ_tot` — or vice versa — mixes populations and
breaks `L = ½Ṁv²` by an O(1) factor once SNe switch on (`v_SN ≈ 3000–10⁴ km/s` vs `v_W ≈ 2000 km/s`).
**[derived, high]**

### A.6 Mass loading (SPEC-072) must be applied to the right pair

Entraining cold gas conserves energy, not momentum. With `Ṁ_tot = (1 + f)Ṁ_w`, at fixed `L`:

```
    v_eff = sqrt(2L/Ṁ_tot) = v_w /sqrt(1+f)        ↓ by (1+f)^(-1/2)
    ṗ_eff = sqrt(2 L Ṁ_tot) = ṗ_w · sqrt(1+f)      ↑ by (1+f)^(+1/2)
```

So the correct implementation holds `L` fixed, increases `Ṁ`, and **recomputes both `v` and `ṗ`**.
A implementation that reduces `v` but keeps the tabulated `ṗ` is implicitly changing `L`
(since `L = ṗv/2`) by the factor `(1+f)^(-1/2)` — an energy leak proportional to the loading. Test:
with `FB_mColdWindFrac = f`, `ṗ_out/ṗ_in` must equal `sqrt(1+f)` and `L_out/L_in` must equal `1`
exactly. Defaults are 0, so this path may be untested. **[derived, high]**

---

## B. Expected magnitudes for a young massive cluster

Reference population: instantaneous burst, Kroupa/Chabrier IMF `0.1–100 M⊙`, `Z = Z⊙`,
normalised to **`M_cl = 10⁶ M⊙`** (the SB99 default burst mass and TRINITY's stated
`sps_refmass`, SPEC-073). All entries **[estimate]**, ±0.3–0.5 dex — they are gross-error gates,
not validation targets.

| Quantity | `t < 3 Myr` | `3.5–10 Myr` | `10–40 Myr` | `t > 40 Myr` |
|---|---|---|---|---|
| `Q_i` [s⁻¹] | `10^52.5–53.0` | `10^51.5 → 10^50.5` | `10^50 → 10^48` | `≲10^47`, falling |
| `L_bol` [erg s⁻¹] | `10^42.7–43.0` | `10^42.5 → 10^42` | `10^42 → 10^41.5` | slow `∝t^-1.3` |
| `f_i = L_i/L_bol` | `0.1–0.3` | `0.03–0.1` | `<0.01` | `≈0` |
| `L_w` (winds) [erg s⁻¹] | `10^39.5–40.3` (peak at WR, 3–4 Myr) | `10^39.5` falling | `≲10^39` | `≪10^38` |
| `ṗ_w` [dyn] | `10^31.7–32.3` | `10^31.5` | `≲10^31` | `≪10^30` |
| `v_w = 2L_w/ṗ_w` | `1500–3000 km/s` | `1000–2500` | `~10³` | ill-defined (0/0) |
| `Ṁ_w` [M⊙ yr⁻¹] | `10^-2.5 – 10^-2` | `~10^-2.5` | `≲10^-3` | AGB only, `v~10 km/s` |
| `L_SN` [erg s⁻¹] | **0** | `10^40–10^41` | `~10^40` (roughly flat) | **0** |
| SN rate [yr⁻¹] | 0 | `~3×10^-4` | `~3×10^-4` | 0 |

**Per unit stellar mass** (divide by `10⁶`): `Q_i ≈ 10^{46.5}` s⁻¹ M⊙⁻¹; `L_bol ≈ 10^{36.9}` erg s⁻¹
M⊙⁻¹ (≈ `1000–1500 L⊙/M⊙`); `L_w ≈ 10^{34}` erg s⁻¹ M⊙⁻¹; `Ṁ_w ≈ 10^{-8}` M⊙ yr⁻¹ M⊙⁻¹.

**In AU units** (the numbers the reconciler will actually see downstream of the loader, for
`f_mass = 1`, i.e. a `10⁶ M⊙` cluster at `t ~ 1–3 Myr`):

```
    Q_i     ≈ 3e66   Myr⁻¹                (= 1e53 s⁻¹ × 3.15576e13)
    L_bol   ≈ 1.7e13 M⊙ pc² Myr⁻³         (= 1e43 / 6.0255e29)
    L_w     ≈ 1.7e10 M⊙ pc² Myr⁻³         (= 1e40 / 6.0255e29)
    ṗ_w     ≈ 1.6e7  M⊙ pc Myr⁻²          (= 1e32 / 6.1623e24)
    v_w     ≈ 2.0e3  pc Myr⁻¹             (= 2L/ṗ ⇒ 2000 km/s ✓ self-consistent)
    Ṁ_w     ≈ 1e4    M⊙ Myr⁻¹             (= 1e-2 M⊙/yr)
```

I computed `v_w` from the AU `L` and `ṗ` above and it lands on 2000 km/s — the anchors are mutually
consistent, which is itself a check that the SPEC-091 constants are right. **[derived from
estimates]**

### B.1 Time evolution — the qualitative shape a correct table must have

**[estimate/recalled, medium confidence on slopes]**

1. **`Q_i`** — nearly flat (falling ≲0.2 dex) to `t ≈ 3–3.5 Myr` while the `>40 M⊙` stars live, then
   a **steep cliff**: ~1 dex by 5 Myr, ~2 dex by 10 Myr, ~4–5 dex by 30 Myr. Effective power law
   after 4 Myr steeper than `t^-3` (commonly quoted `∝ t^-4…-5`). **This is the fastest-declining
   quantity in the table** and the one most damaged by a linear interpolant (Section D).
2. **`L_bol`** — flat to ~3 Myr, then a *shallow* decline `∝ t^-1.2…-1.4`, because intermediate-mass
   stars keep shining. **`L_bol` and `Q_i` must decouple after ~4 Myr** — if the code's `L_bol(t)`
   tracks `Q_i(t)` (e.g. because `L_i` is formed with a *constant* `f_i`), that is a bug.
   `f_i(t)` must fall by 1–2 dex over 3→10 Myr.
3. **`L_w`** — rises modestly (factor ~2–5) from `t=0` to a peak at the Wolf–Rayet phase
   (`t ≈ 3–4 Myr`), driven by the increase of `v_∞` and `Ṁ` in the WR stage, then declines.
4. **`L_SN`** — **exactly zero** before the first core collapse (`t_SN,on ≈ 3–4 Myr`, the lifetime of
   the most massive star; rotation and the IMF upper cutoff move it in the 3–4.5 Myr window), then a
   near-step to `10^40–10^41` erg/s, roughly flat until the last `8 M⊙` progenitor dies at
   `t_SN,off ≈ 37–45 Myr`, then **exactly zero again**. The onset is a *genuine discontinuity* in the
   underlying physics, smoothed only by the IMF's continuum of lifetimes.
5. **Winds vs SNe crossover:** the total mechanical luminosity is roughly *continuous* across
   `t_SN,on` in a `10⁶ M⊙` cluster (`L_w` peak `~10^40.2` meets `L_SN ~10^40`), so the total is
   `10^40–10^41` erg s⁻¹ throughout `0–40 Myr` and then falls off a cliff. **A table whose total
   mechanical luminosity is still `~10^40` at its last row has not yet reached the cliff.**

### B.2 Integral invariants (excellent end-to-end unit checks)

These bound the table *without* depending on any file-layout recollection:

```
    N_SN ≈ M_cluster / 100 M⊙                     (Kroupa; range 1/50 … 1/150 M⊙⁻¹)
    ∫ L_SN dt  ≈ N_SN × 1e51 erg ≈ 1e49 erg per M⊙  ⇒  1e55 erg per 1e6 M⊙
    ∫ L_w  dt  ≈ 1e54 … 1e55 erg per 1e6 M⊙       (winds ≈ 10–100% of SN energy)
    ∫ (Ṁ_w+Ṁ_SN) dt |_{40 Myr} ≈ 0.1–0.2 M_cluster      and  < M_cluster  ALWAYS (hard)
    ∫ L_bol dt |_{40 Myr} ≈ 1e-3 M_cluster c²     and  < 0.007 M_cluster c²  (hard: H→He efficiency)
```

The last two are **theorems**, not estimates: a population cannot return more mass than it contains,
nor radiate more than nuclear burning can supply. Either would be violated by a `10⁶` double-scaling
or a `L☉`→`erg/s` inversion, and both are cheap to evaluate by trapezoid over the ingested arrays.
`0.007 M_cluster c²` for `10⁶ M⊙` is `1.25e58 erg`; the expected `~10^58 erg` sits just below it —
comfortably diagnostic of a factor ≥10 error, not of a factor 1.5. **[derived, high for the bounds;
estimate for the expected values]**

### B.3 Cross-quantity bounds that need no file-layout knowledge

1. **`L_i / Q_i ≥ 13.6 eV = 2.179e-11 erg` — exact.** A photon counted in `Q_i` is by definition
   above the Lyman edge, so the mean energy of the ionizing photons cannot be below 13.6 eV.
   Realistically `⟨hν⟩ ≈ 15–25 eV` at all ages (softening with time). In AU, `L_i/Q_i` has units of
   energy (`M⊙ pc² Myr⁻²` = `1.90148e43 erg`), so the bound is `L_i/Q_i ≥ 1.146e-54` AU and the
   expected band is `1.3e-54 … 2.1e-54`. **This is an independent consistency check between two
   table columns that a units bug in either one will break.** **[derived, high]**
2. **`L_i + L_n = L_bol`** exactly (SPEC-074), and `0 ≤ f_i ≤ 1`. A negative `f_i` means a log column
   was read as linear; `f_i > 1` means a linear column was raised to a power of ten.
3. **`ṗ_w c / L_bol ≈ 0.1–0.5`** at `t < 4 Myr` (from the anchors: `1e32 × 3e10 / 1e43 = 0.3`). This
   is the well-known statement that **direct radiation pressure exceeds wind momentum by ~3×** in a
   young cluster (SPEC-071). A value `≫1` means `L_w` has been substituted for `L_bol`, or `ṗ` is
   over-scaled; a value `≪0.01` means the wind columns are being under-read.
4. **`L_w/L_bol ≈ 10^{-3}` (range `10^{-4}…10^{-2}`).** A ratio near unity is a swap (Section F.4).

---

## C. Scaling with cluster mass

### C.1 Extensive vs intensive — the complete classification

Every SPS quantity is either a **sum over stars** (extensive, exactly linear in `N_*`, hence in
`M_cluster` at fixed IMF shape) or a **ratio of two such sums** (intensive, invariant).

| Extensive (× `f_mass`) | Intensive (**must not** be scaled) |
|---|---|
| `Q_i`, `L_bol`, `L_i`, `L_n` | `t` (independent variable) |
| `Lmech_W`, `Lmech_SN`, `Lmech_total` | `f_i = L_i/L_bol` |
| `pdot_W`, `pdot_SN`, `pdot_total` | `v_W = 2L_W/ṗ_W`, `v_SN = 2L_SN/ṗ_SN` |
| `Mdot_W`, `Mdot_SN` | any ratio: `L_w/L_bol`, `ṗc/L_bol`, `L_i/Q_i`, `⟨hν⟩` |
| cumulative `E`, cumulative `M_ret` | `E_SN` per event, `N_SN/M_cluster` |

This exactly matches SPEC-073's "everything except `t`, `fi`, `v_SN`" **provided `v_W` is derived
rather than tabulated**. If `v_W` were ever a stored column it belongs in the right-hand list too.
**[derived, high]**

### C.2 The sharp test: `v` is scale-invariant

`v = 2L/ṗ` is a ratio of two extensive quantities ⇒ **`v(f_mass) = v(1)` identically**. This turns
into the cleanest possible detector of scaling bugs:

| Bug | Effect on `L`, `ṗ` | Effect on `v` | Detectable? |
|---|---|---|---|
| `f_mass` applied twice to everything | both `× f²` | **unchanged** | ✗ by `v`; ✓ by B.2 integrals |
| `f_mass` applied to `L` only | `L×f`, `ṗ×1` | `× f` | ✓ loudly |
| `f_mass` applied to `ṗ` only | `L×1`, `ṗ×f` | `÷ f` | ✓ loudly |
| `f_mass` applied to `v` (wrongly listed extensive) | — | `× f` | ✓ loudly |

Note the **uniform** double-scaling is *invisible* to `v` — it needs the absolute check
(`L(f_mass)/L(1) == f_mass` exactly, or the B.2 mass/energy bounds). Both tests are required.
**[derived, high]**

### C.3 The trap of scaling an already-normalised table

`f_mass = M_cluster / sps_refmass` (SPEC-073). The whole construction is only correct if
`sps_refmass` **is the normalisation the table was generated with**. Failure modes:

- SB99's **instantaneous-burst** mode normalises to the burst mass declared in the `.input` file —
  `10⁶ M⊙` is the shipped default but *not a law*. **[recalled]**
- SB99's **continuous-star-formation** mode normalises to a **star-formation rate of 1 M⊙ yr⁻¹**,
  not a mass. A continuous-SF table is dimensionally a *different object* (its `Q(t)` rises to a
  plateau rather than falling), and dividing it by any `sps_refmass` in M⊙ is a category error that
  will not raise anything — it will just produce a cluster whose ionizing output *increases* with
  age. **A loader that accepts a user table must at minimum record which mode produced it, and the
  monotonicity of `Q(t)` after 5 Myr is a free discriminator.** **[recalled layout, derived
  consequence]**
- A per-`M⊙`-normalised table with `sps_refmass` left at the bundled `10⁶` is a `10⁶` error.
  Direction: feedback `10⁶×` too weak ⇒ the bubble never expands ⇒ the run terminates as a
  stall/collapse. **Silent and plausible-looking** — exactly the class B.2's hard bounds catch.

**Equivalence test:** the same physical cluster expressed as (table `T`, `sps_refmass = 10⁶`) and as
(`T/10⁶`, `sps_refmass = 1`) must produce **identical** drivers. **[derived, high]**

### C.4 Where linearity actually breaks — stochastic IMF sampling

Linear scaling assumes the IMF is **fully sampled**, i.e. that the cluster contains enough massive
stars for the ensemble mean to describe an individual object.

- `N(>8 M⊙) ≈ M_cluster / 100 M⊙`; `N(>20 M⊙) ≈ M_cluster / 400 M⊙` (order).
- Poisson scatter alone is `1/√N`. But `Q_i` and `L_w` are dominated by the **top few** stars, so
  the effective `N` is far smaller than `N(>8)` and the scatter far larger than Poisson.
- Practical thresholds **[estimate]**:
  - `M_cluster ≳ 10⁵ M⊙` — winds, `Q_i`, WR phase all well sampled; linear scaling safe (≲few %).
  - `10⁴ … 10⁵ M⊙` — `Q_i` scatter ~10–30%; the WR phase (a handful of stars) is *not* sampled, so
    `L_w(t)` around 3–4 Myr is unreliable in an individual cluster.
  - `10³ … 10⁴ M⊙` — factor-of-few scatter in `Q_i`; the **median** cluster falls *below* the
    IMF-averaged table (the mean is preserved only because rare draws carry it).
  - `< 10³ M⊙` — `⟨N(>8 M⊙)⟩ ≲ 10`; distribution of `Q_i` spans orders of magnitude and is strongly
    skewed. The IMF-averaged table describes **no individual cloud**.
  - `= 10² M⊙` — `⟨N(>8 M⊙)⟩ ≈ 1`. The table asserts a *continuous* SN rate of `~10⁻⁶ SN/yr` and a
    smooth wind; reality is a coin flip between "one SN" and "none".
- SPEC-073 already flags that `param/paperII_grid_sweep.param` reaches `M_cluster = 100 M⊙`. **That
  cell is outside the model's validity by ~3 orders of magnitude in `N`, and a linear-scaling loader
  cannot know it.** The correct behaviour is a **warning at ingest** keyed on `M_cluster` (or
  `f_mass × sps_refmass`), with a documented threshold. Suppressing it entirely is a
  regime/validity failure, not a numerical one. **[derived, high]**

There is a second, unfixable-by-scaling non-linearity: the table is a **coeval, single-metallicity,
instantaneous burst**. Extended star formation, age spread and internal `Z` spread cannot be
recovered by any multiplicative factor.

---

## D. Time-grid behaviour

### D.1 Requirements on the grid itself

1. **Strictly increasing `t`.** Duplicate abscissae make any interpolant either singular or
   silently arbitrary; a non-monotone `t` makes bisection-based lookup return the wrong bracket
   *without erroring*. `validate_t_monotonic` must reject `t[i+1] ≤ t[i]`, not merely
   `t[i+1] < t[i]` — a repeated row is the more common corruption (a duplicated line in a
   hand-edited table) and `>=` vs `>` is the whole difference. **[derived, high]**
2. **Resolution at the SN onset.** The onset is a near-step at `t ≈ 3.5 Myr`. A grid coarser than
   ~0.1 Myr there aliases the step; SB99's default `10⁵ yr` linear grid is adequate,
   a `1 Myr` grid is not. **[recalled grid, derived consequence]**
3. **Coverage.** TRINITY integrates until dispersal or re-collapse, which for a massive/dense cloud
   is tens of Myr. The table must therefore span at least `0 → 40+ Myr`; anything shorter *will* be
   exceeded by real runs.

### D.2 What must happen beyond the last row — and why clamping is wrong

The physical late-time behaviour of each quantity, after the last core-collapse SN
(`t ≳ 40 Myr`) **[derived from stellar evolution, high on direction, estimate on rates]**:

| Quantity | Late-time truth | Clamping to last row gives |
|---|---|---|
| `L_mech,SN` | **exactly 0** (no progenitors left) | a perpetual `~10^40 erg/s` energy source |
| `ṗ_SN` | **exactly 0** | perpetual momentum injection |
| `L_mech,W` | collapses ≥2–3 dex; AGB winds only | perpetual wind |
| `ṗ_W` | collapses; AGB `v ~ 10 km/s`, so `ṗ` falls faster than `Ṁ` | ditto |
| `v_W` | drops from `~2000` to `~10 km/s` (hot-star → AGB wind) | frozen at `~2000 km/s` |
| `Q_i` | collapses ≥5 dex; a tiny post-AGB/WD floor only after `≳100 Myr` | perpetual HII region |
| `L_bol` | does **not** collapse — declines slowly `∝ t^-1.3` | mildly too high (least harmful) |

**Why clamping is wrong, stated precisely.** If the table ends at (or before) the end of the SN
epoch, the last row still carries `L_mech ~ 10^40 erg s⁻¹`. Holding it fixed converts a
finite-energy budget (`∫L dt ≈ 10^55 erg`) into an **unbounded** one: the injected energy grows
without limit, so a bubble that should stall and stop being driven instead keeps expanding forever.
The failure is not a small error — it changes the *fate*, which is TRINITY's headline output
(SPEC-100/101), and it is silent because every intermediate value stays finite and plausible. A run
that "never terminates" or that reports dispersal at `t ≫ t_table` is the signature. **[derived,
high]**

Acceptable behaviours, in decreasing order of honesty:
- **(i) Refuse.** Raise/terminate the run with an explicit end-reason when `t > t_max`. The model
  genuinely has no information there; stopping is the truthful answer and is trivially auditable.
- **(ii) Zero the injection terms** (`L_mech`, `ṗ`, `Ṁ`, `Q_i` → 0) and continue `L_bol` on a
  documented decay. Physically defensible if the table ends *after* the last SN.
- **(iii) Log–log power-law extrapolation** with the physical slope, clipped at zero.
- **(iv) Clamp to the last row** — defensible **only** if the last row's values are already
  negligible (`L_mech` ≲ 10⁻³ of its peak). Silently defensible-looking, and wrong otherwise.

**Test:** evaluate the feedback function at `t = 2 × t_max` and at `t = 10 × t_max`. If it returns
the `t_max` values unchanged *and* those values are a non-negligible fraction of the table's peak,
that is the bug. **[derived, high]**

### D.3 Behaviour *below* the first row

If the table's first row is `t > 0` (SB99 grids often start at `10⁴–10⁵ yr` **[recalled]**) and the
simulation starts at `t = 0`, the loader must extrapolate below `t_min`. For a burst all quantities
are essentially flat over `0 → 0.5 Myr` (no massive star has evolved), so **clamping to the first
row is the physically correct choice here** — the opposite of the `t_max` case. A cubic spline
extrapolating backwards, by contrast, can undershoot to negative values in the first timestep, which
is where the ODE solver is most fragile. **[derived, high]**

### D.4 Interpolation scheme

`get_interpolation(sps, ftype='cubic')` names a cubic default. Physics requirements:

1. **Non-negativity is mandatory.** `L`, `ṗ`, `Ṁ`, `Q` are non-negative by construction. Any
   interpolant that can return a negative value must be clipped. A cubic spline through the SN
   onset step **will** undershoot on the pre-onset side (Gibbs/overshoot); a negative `L_mech`
   injected into `dE_b/dt` (SPEC-035) removes energy from the bubble — a sign error with no
   physical meaning. **Test: `min` over a 10⁴-point uniform grid of every interpolated quantity
   must be `≥ 0`.** **[derived, high]**
2. **Interpolate in log space, or use a shape-preserving scheme.** `Q_i` spans ~5 dex over the table
   and falls steeply after 4 Myr. Linear-in-linear interpolation across a decade-wide gap
   systematically **over**-estimates a convex decaying function (chord above curve) — by up to tens
   of percent for a factor-10 drop across one interval. Log-linear (equivalently, power-law between
   knots) is exact for a power law and is the right default for `Q_i`, `L_bol`, `L_mech`.
3. **Log space and exact zeros are incompatible.** `L_SN ≡ 0` before onset ⇒ `log(0) = −∞` ⇒ NaN.
   Any log-space interpolation must floor the argument (this is a plausible role for `EPSILON`),
   and the floor must be far below any dynamically relevant value (e.g. `1e-30 ×` the peak), not a
   "convenient" `1e-6` which would be a real luminosity in AU units for some quantities.
   **[derived, high]**
4. **A monotone interpolant (PCHIP) beats a cubic spline here**: same order of accuracy on smooth
   stretches, no overshoot at the SN step, guaranteed non-negative given non-negative data.
5. **RHS smoothness.** The ODE solver sees `L(t)` in `dE_b/dt`. A `C⁰`-only (piecewise-linear)
   driver puts kinks in the RHS at every table node, forcing an adaptive solver to shorten steps at
   each node; a `C¹` interpolant avoids it. This is a performance/robustness argument for splines
   and it is in *tension* with (1)/(4) — the correct resolution is a monotone `C¹` scheme, not an
   unconstrained cubic.

---

## E. Expected dimensions, log columns, and the mandatory conversions

### E.1 SB99 output conventions **[recalled — verify against the actual table header]**

| File | Column content | Units | log₁₀? |
|---|---|---|---|
| `*.quanta` | `TIME` | **yr** (linear) | no |
| | `Q(H I)`, `Q(He I)`, `Q(He II)` | s⁻¹ | **yes** |
| | ionizing luminosity fractions | – | **yes** (⇒ negative values) |
| | `L_bol` | erg s⁻¹ | **yes** |
| `*.power` | `TIME` | yr | no |
| | power (winds / SN / total) | erg s⁻¹ | **yes** |
| | momentum flux (winds / SN / total) | dyn = g cm s⁻² | **yes** |
| | cumulative energy | erg | **yes** |
| `*.snr` | SN rate | yr⁻¹ | **yes** |
| | typical / lowest progenitor mass | M⊙ | **no** (linear — mixed file!) |
| `*.mass` | mass-loss rate | M⊙ yr⁻¹ | **yes** |

Confidence: **medium** that time is in **years and linear** (I am fairly firm on this one);
**medium** on which luminosity/momentum columns are logged (I believe essentially all rate columns
are); **low** on exact column ordering. **The mixed linear/log file (`*.snr`) is the trap**: a
per-file "everything is log" assumption is wrong. This is presumably why the interface carries a
per-column `log: bool` — which is the right design.

### E.2 Mandatory conversions into `[M⊙, pc, Myr]`

Derived from SPEC-091. `C` = cgs value of one AU unit; convert by **dividing** by `C`.

| Canonical | Source unit | Multiply by |
|---|---|---|
| `t` | yr | `1e-6` |
| `t` | s | `1/3.15576e13 = 3.1689e-14` |
| `Qi` | s⁻¹ | `3.15576e13` |
| `Lbol`,`Li`,`Ln`,`Lmech_*` | erg s⁻¹ | `1/6.0255e29 = 1.6596e-30` |
| `Lbol` etc. | L⊙ | `3.828e33/6.0255e29 = 6.353e3` |
| `pdot_*` | dyn | `1/6.1623e24 = 1.6228e-25` |
| `Mdot_*` | M⊙ yr⁻¹ | `1e6` |
| `Mdot_*` | g s⁻¹ | `3.15576e13/1.98892e33 = 1.5867e-20` |
| `v_SN` | cm s⁻¹ | `1.0227e-5` |
| `v_SN` | km s⁻¹ | `1.022712` |
| `fi` | – | `1` |

**`_L_SUN_ERG_S` must be `3.828e33` (IAU nominal `L⊙ = 3.828e26 W`)**; `3.839e33` / `3.846e33` are
older values, a 0.5% difference — negligible via `R ∝ L^{1/5}` (0.1%), so **S4**. Anything outside
`[3.8e33, 3.9e33]` is a real error. **[derived, high]**

### E.3 Order of operations — de-log first, then convert

```
    CORRECT:   X_AU = 10**arr / C            (equivalently  log10(X_AU) = arr − log10(C))
    WRONG:     X_AU = 10**(arr / C)          ← catastrophic, e.g. 10^(40/6e29) = 1.000…
    WRONG:     X_AU = 10**(arr * f)          ← same class
    WRONG:     X_AU = 10**(10**arr / C)      ← double de-log
```

The first wrong form is *not* loud: `10**(40/6.0255e29) ≈ 1 + 1.5e-28`, i.e. every luminosity
becomes `≈1.0` in AU — finite, positive, and utterly wrong. The bubble then receives
`~6e29 erg/s`⁻¹… no: `1 M⊙pc²Myr⁻³ = 6e29 erg/s`, so it receives `6e29 erg/s` ≈ `10^{-10}` of the
true value ⇒ a bubble that never expands ⇒ terminates as "no expansion / collapse". **Silent.**
Test: for a log column with `declared_units='cgs'`, `convert_to_canonical_au` must satisfy
`out == 10**arr / C` to machine precision. **[derived, high]**

---

## F. Known traps, each with its detector

| # | Trap | Signature | Detector | Severity if present |
|---|---|---|---|---|
| F.1 | **log column read as linear** | `Q = 53` instead of `10^53`; `L = 40` instead of `10^40` | absolute range gate (`Q > 1e40 s⁻¹`; `L_bol > 1e38 erg/s`) for any cluster with `M > 10³ M⊙` | S1 — silent, feedback vanishes, run "stalls" |
| F.2 | **linear column read as log** | `10**1e53` → `inf`/overflow, or `10**0.3 = 2` for `f_i` | `f_i ∈ [0,1]`; `isfinite` on everything | S1, but usually loud |
| F.3 | **per-unit-mass table used as absolute** (or vice versa) | `10⁶` error in every extensive quantity | `∫Ṁdt < M_cluster`; `∫L_bol dt < 0.007 M c²`; `Q/M ≈ 10^46.5 s⁻¹ M⊙⁻¹` | S1 — silent |
| F.4 | **`L_w` ↔ `L_bol` swap** | factor `~10³`; **`v = 2L_bol/ṗ_w` → `~10⁶ km/s > c`** | `v_w < c` guard; `L_w/L_bol ∈ [1e-4, 1e-2]` | S1 |
| F.5 | **mass scaling applied twice** | `f_mass²` | `L(f)/L(1) == f` exactly; integral bounds of B.2 | S1 |
| F.6 | **mass scaling applied to an intensive column** (`f_i`, `v_SN`) | `f_i` outside `[0,1]`; `v_SN` scales with cluster mass | invariance test of C.2 | S1 |
| F.7 | **ionizing photon *rate* vs cumulative *number*** | a monotonically **increasing** `Q(t)`; or a `~10^{13.5}` inflation if a cumulative-energy column is read as a power | *rate* columns must be non-monotone (peak then fall); *cumulative* columns must be non-decreasing — this is a shape test, immune to unit errors | S1 |
| F.8 | **linear interpolation of a multi-decade quantity** | systematic over-estimate of `Q_i` in the steep decline; negative values from spline overshoot at the SN step | `min(interp) ≥ 0` on a fine grid; compare linear vs log-linear interpolants on the steepest interval | S2 |
| F.9 | **SN energy as a smooth rate vs discrete events** | valid for `N_SN ≳ 10²` (`M_cluster ≳ 10⁴ M⊙`); qualitatively wrong for `N_SN ≲ 10` | warn on `M_cluster < 10³–10⁴ M⊙` | S2 (regime) |
| F.10 | **`L_SN = L_total − L_W` going slightly negative** | at `t < t_SN,on`, both logs round to 3 decimals; the difference of two ~equal rounded numbers has ~0.2% noise of **either sign** | `L_SN ≥ 0` clip, and `L_SN(t < 3 Myr) == 0` | S3 (S2 if unclipped and fed to `dE/dt`) |
| F.11 | **year vs Myr on the `t` column** | `10⁶` error in time; the run then sees only the `t=0` row for its whole life — no SNe, no `Q` decline | `t[-1]` must be `10–10³` in Myr; a `t[-1] ~ 10⁷` is years | S1 — silent |
| F.12 | **`log(0)` in a log-space interpolant** | NaN from `L_SN ≡ 0` rows | `isfinite` over the whole ingested + interpolated arrays | S1 (loud but fatal) |
| F.13 | **`0/0` in `v = 2L/ṗ` at late times** | both → 0 together; a bare `2L/(ṗ + ε)` gives an *enormous* `v` for small-but-finite `L` | `0 ≤ v < c`, `isfinite`, at every row **and** every interpolated `t` | S2 |
| F.14 | **header row parsed as data** | a spurious first row with garbage `t` | `_can_parse_float`/`_scan_layout` must skip it; `t[0] ≥ 0` and `t` monotone | S3 |
| F.15 | **continuous-SF table treated as a burst table** | `Q(t)` rises to a plateau instead of falling | `Q(20 Myr) < Q(1 Myr)` | S2 |
| F.16 | **bundled vs user path divergence** | the two loaders apply units/`f_mass` differently | round-trip: write the bundled table out as a user table with explicit column specs, re-ingest, require identical canonical arrays | S2 |

---

## G. The `SPSFeedback` record interface

`SPSFeedback` exposes `__iter__`/`__getitem__`/`__len__` — i.e. **positional unpacking over a set of
physically distinct quantities, several of which share units and magnitude** (`Lmech_W`, `Lmech_SN`,
`Lmech_total` are all powers of order `10^40 erg/s`; `pdot_W`, `pdot_SN`, `pdot_total` likewise).

Physics consequence: **a positional swap between two same-unit fields is undetectable by any range,
dimension, or `isfinite` check.** The only defences are (a) the algebraic identities of §A applied
per-field (`v_W = 2L_W/ṗ_W` must be `~2000 km/s` while `v_SN = 2L_SN/ṗ_SN` must be `~3000–10⁴ km/s`
— they differ enough to distinguish), and (b) a test asserting positional order matches named
access for every field. Additionally `Lmech_total == Lmech_W + Lmech_SN` and
`pdot_total == pdot_W + pdot_SN` must hold to machine precision (or to log-rounding precision if
`total` is tabulated independently), which pins the three-way grouping.

`get_current_sps_feedback(t, params)` must be evaluated at the **cluster age**, which for TRINITY's
single-coeval-cluster model (SPEC-001) equals simulation time only if the cluster forms at `t = 0`.
Any restart, `t_start > 0`, or re-collapse-and-reburst path must not silently re-zero the SPS clock.
**[derived, medium — the offset question depends on the run driver, not on this slice]**

---

## H. Summary of what I could not determine without the source

- Whether `Ṁ_w` and `v_w` are **derived** from `(L, ṗ)` or read independently (§A.2). This
  determines whether the identity check is a machine-precision test or a physics test.
- Whether `EPSILON` is a log-floor (correct use) or a denominator guard (dangerous use, §F.13).
- The bundled table's actual `t_max`, and hence whether §D.2 is a live risk or a theoretical one.
- Whether `_L_SUN_ERG_S` is ever exercised (does any shipped table declare `Lsun`?).

---

```json
[
  {
    "id": "S10-C-01",
    "file": "trinity/sps/read_sps.py",
    "line": 38,
    "class": "coefficient",
    "severity": "S1",
    "claim": "Wind terminal velocity must be v_w = 2 L_w / pdot_w and mass-loss rate Mdot_w = pdot_w^2 / (2 L_w); the factor 2 (from L = 1/2 M v^2) must be present exactly once.",
    "evidence": "For a steady wind, L_w = (1/2) Mdot v_w^2 and pdot_w = Mdot v_w by definition of kinetic-energy and momentum flux. Dividing: 2L/pdot = v. Substituting back: Mdot = pdot^2/(2L). Dimensions in AU: (M pc^2 Myr^-3)/(M pc Myr^-2) = pc/Myr (velocity) and (M pc Myr^-2)^2/(M pc^2 Myr^-3) = M/Myr (mass rate). SPEC-071.",
    "expected": "v_w = 2*L_w/pdot_w exactly; Mdot_w = pdot_w**2/(2*L_w) exactly. Omitting the 2 gives v low by 2x and Mdot high by 2x; using v = L/pdot (factor 1/2) is the classic slip.",
    "failure_scenario": "A factor-2 error in v_w propagates to rho_w = Mdot/(4 pi r^2 v) and to any place a wind velocity is used, and silently rescales the free-wind density by 4x while leaving the momentum flux pdot (which alone sets R1 and the ram pressure) correct - so the bug hides in exactly the terms that are hardest to validate.",
    "repro": "For every ingested row assert abs(2*L_w/(pdot_w*v_w) - 1) < 1e-12 when v_w is derived; assert v_w in [100, 10000] km/s = [102, 10227] pc/Myr for t < 5 Myr.",
    "confidence": "high"
  },
  {
    "id": "S10-C-02",
    "file": "trinity/sps/read_sps.py",
    "line": 38,
    "class": "state",
    "severity": "S1",
    "claim": "The set {L, pdot, Mdot, v} has only TWO degrees of freedom; if the loader accepts three or four of them (SPEC-070 lists Lmech_SN, pdot_SN, Mdot_SN and v_SN as all admissible), the redundancy must be checked or one pair declared primary.",
    "evidence": "Two definitions (L = 1/2 Mdot v^2, pdot = Mdot v) constrain four quantities. Any third independent input over-determines the wind. For a POPULATION the over-determination is not merely redundant: Cauchy-Schwarz gives (sum Mdot v)^2 <= (sum Mdot)(sum Mdot v^2), i.e. Mdot_eff = pdot^2/(2L) <= Mdot_true and v_eff = 2L/pdot >= v_rms >= v_mean, with equality only for a monokinetic wind.",
    "expected": "Either (a) exactly two canonicals per component are used and the rest derived, or (b) all supplied values are cross-checked to |residual| < 3e-3 (log-column rounding) and a mismatch raises/warns. Silently mixing a tabulated Mdot with a derived v means the energy equation and the momentum equation are driven by two different winds.",
    "failure_scenario": "Bubble energy input uses L directly while the ram-pressure/free-wind density uses an inconsistent (Mdot, v) pair; energy and momentum budgets disagree by the velocity-dispersion factor of the population, with no error raised.",
    "repro": "For each row where more than two of the quadruple are present, assert abs(2*L/(pdot*v) - 1) < 3e-3 and abs(pdot/(Mdot*v) - 1) < 3e-3; also assert pdot**2/(2*L) <= Mdot_tabulated*(1+3e-3).",
    "confidence": "high"
  },
  {
    "id": "S10-C-03",
    "file": "trinity/sps/read_sps.py",
    "line": 38,
    "class": "exponent",
    "severity": "S1",
    "claim": "f_mass must multiply every EXTENSIVE canonical exactly once and no INTENSIVE canonical at all; the invariance of v = 2L/pdot under f_mass is the sharpest detector.",
    "evidence": "Every SPS quantity is either a sum over stars (extensive, linear in N_* hence in M_cluster at fixed IMF) or a ratio of two such sums (intensive). Extensive: Qi, Lbol, Li, Ln, Lmech_*, pdot_*, Mdot_*, cumulative E and M. Intensive: t, fi = Li/Lbol, v_SN, v_W = 2L_W/pdot_W, and every ratio. SPEC-073's list ('everything except t, fi, v_SN') is correct provided v_W is derived, not stored.",
    "expected": "L(f_mass)/L(1) == f_mass exactly for each extensive array; v_w(f_mass) == v_w(1) bitwise; fi unchanged. Scaling L but not pdot makes v scale as f_mass; the uniform double-scaling (f_mass^2 on everything) leaves v invariant and needs the absolute test instead.",
    "failure_scenario": "Uniform double scaling is invisible to every ratio test and produces a cluster whose feedback is f_mass times too strong or weak - for the default M_cloud=1e7, sfe=0.01 (M_cluster=1e5, f_mass=0.1) that is a factor of 10, enough to flip dispersal to re-collapse.",
    "repro": "Ingest the same table with f_mass and 10*f_mass in separate processes; assert exact 10x on all extensive arrays and bitwise equality of 2*L/pdot; independently assert integral(Mdot dt) < M_cluster and integral(Lbol dt, 0..40 Myr) < 0.007*M_cluster*c^2.",
    "confidence": "high"
  },
  {
    "id": "S10-C-04",
    "file": "trinity/sps/read_sps.py",
    "line": 285,
    "class": "silent-failure",
    "severity": "S1",
    "claim": "Evaluating the SPS interpolant beyond the table's last time must not clamp to the final row's values when those values are still dynamically significant.",
    "evidence": "Physically, after the last core-collapse SN (t ~ 37-45 Myr for an 8 Msun progenitor) the SN rate, L_mech_SN and pdot_SN are EXACTLY zero, and the wind terms collapse by 2-3 dex (only AGB winds at v ~ 10 km/s remain). Qi collapses by >5 dex. Only L_bol persists, declining slowly as ~t^-1.3. If the table ends at or before ~40 Myr its last row still carries L_mech ~ 1e40 erg/s per 1e6 Msun; holding that constant converts the finite SN energy budget (~1e55 erg, i.e. ~1e49 erg per Msun) into an unbounded one.",
    "expected": "For t > t_max, either terminate the run with an explicit end-reason, or zero the injection terms (L_mech, pdot, Mdot, Qi) and continue L_bol on a documented decay, or extrapolate log-log with the physical slope clipped at zero. Clamping is acceptable ONLY if the last row is already negligible (L_mech < 1e-3 of its peak). Below t_min the opposite holds: clamping to the first row IS correct, because a burst is flat over 0-0.5 Myr.",
    "failure_scenario": "A long-running low-feedback cloud passes t_max, keeps receiving ~1e40 erg/s forever, and eventually reports 'dispersal' at a time the model has no information about - changing the headline fate (SPEC-100/101) with every intermediate value finite and plausible.",
    "repro": "Call the feedback function at t = 2*t_max and 10*t_max; if the returned L_mech equals the t_max value and that value exceeds 1e-3 of the table's peak L_mech, the clamp is live.",
    "confidence": "high"
  },
  {
    "id": "S10-C-05",
    "file": "trinity/sps/read_sps.py",
    "line": 285,
    "class": "numerical",
    "severity": "S1",
    "claim": "No interpolated SPS quantity may be negative; a cubic spline through the supernova-onset step will undershoot on the pre-onset side.",
    "evidence": "L_mech, pdot, Mdot and Qi are non-negative by construction (they are fluxes of positive quantities). L_SN is exactly zero before the first core collapse (~3.5 Myr) and jumps to ~1e40-1e41 erg/s per 1e6 Msun within a fraction of a Myr. An unconstrained cubic spline across a near-step overshoots on both sides (Gibbs-type ringing); on the zero side the overshoot is negative. Qi additionally falls ~5 dex over the table, so any linear-in-linear interpolant across a steep interval both over-estimates (chord above a convex decaying curve) and can ring.",
    "expected": "min over a fine (>=1e4 point) grid of every interpolated quantity must be >= 0. A monotone C1 scheme (PCHIP) or log-space interpolation with an explicit floor is the correct default; an unconstrained cubic is not.",
    "failure_scenario": "A negative L_mech entering dE_b/dt (SPEC-035) removes energy from the bubble for a few timesteps before the SN onset - a sign error with no physical meaning that a solver will happily integrate.",
    "repro": "Build the interpolant, evaluate on linspace(t_min, t_max, 100000), assert (values >= 0).all() for Qi, Lbol, Lmech_W, Lmech_SN, pdot_W, pdot_SN.",
    "confidence": "high"
  },
  {
    "id": "S10-C-06",
    "file": "trinity/sps/sps_columns.py",
    "line": 180,
    "class": "units",
    "severity": "S1",
    "claim": "De-logging must precede the linear unit conversion: X_AU = 10**arr / C, never 10**(arr/C) or 10**(arr*factor).",
    "evidence": "For a log10 column, log10(X_AU) = log10(X_cgs) - log10(C). Applying the conversion factor inside the exponent gives 10**(40/6.0255e29) = 1 + 1.5e-28, i.e. every luminosity becomes ~1.0 in AU units, which is 6.0e29 erg/s - about 1e-10 of the true 1e40 erg/s.",
    "expected": "convert_to_canonical_au(arr, canonical, declared_units, log=True) must return 10**arr * (linear factor), identical to 10**(arr - log10(C)) to machine precision. It must not de-log twice when declared_units itself names a log unit.",
    "failure_scenario": "Every feedback driver is ~1e-10 of truth; the bubble never expands and the run terminates as a stall or collapse. Finite, positive, plausible-looking, and completely wrong.",
    "repro": "For a synthetic column arr = [39.0, 40.0, 41.0] with declared_units='cgs', log=True, canonical='Lmech_W', assert output == 10**arr / 6.0255e29 to within 1e-12 relative.",
    "confidence": "high"
  },
  {
    "id": "S10-C-07",
    "file": "trinity/sps/sps_columns.py",
    "line": 180,
    "class": "units",
    "severity": "S1",
    "claim": "The AU conversion constants must satisfy C_L/C_p = C_v and C_p^2/C_L = C_Mdot, so that v and Mdot derived in AU equal the cgs-derived values converted.",
    "evidence": "Derived from the identities themselves. Using SPEC-091: 6.0255e29/6.1623e24 = 9.7781e4 cm/s = 1 pc/Myr (correct); (6.1623e24)^2/6.0255e29 = 6.302e19 g/s = 1 Msun/Myr (correct, since 1.98892e33/3.15576e13 = 6.302e19).",
    "expected": "Luminosity factor 6.0255e29 erg/s per (Msun pc^2 Myr^-3); force factor 6.1623e24 dyn per (Msun pc Myr^-2); velocity 0.977781 km/s per (pc/Myr); Qi: 1 s^-1 = 3.15576e13 Myr^-1; Mdot: 1 Msun/yr = 1e6 Msun/Myr. Any independently hardcoded constant must be consistent with the other two to <1e-5 relative, or v = 2L/pdot computed in AU will disagree with the cgs value.",
    "failure_scenario": "A slightly-off Myr (e.g. 3.15e13 instead of 3.15576e13, the 3.15e7 s/yr shortcut) is a 0.2% time error, which via the Weaver R ~ (L/rho)^(1/5) t^(3/5) scaling is a 1% error in inferred L - too small to notice, too large to be right (SPEC-092 item 5).",
    "repro": "Assert C_L/C_p == C_v and C_p**2/C_L == C_Mdot to 1e-9 relative using whatever constants the module actually defines.",
    "confidence": "high"
  },
  {
    "id": "S10-C-08",
    "file": "trinity/sps/sps_columns.py",
    "line": 334,
    "class": "silent-failure",
    "severity": "S2",
    "claim": "validate_t_monotonic must reject non-STRICTLY increasing time, i.e. t[i+1] <= t[i], not merely t[i+1] < t[i].",
    "evidence": "A duplicated abscissa (the commonest corruption of a hand-edited or concatenated table) makes any interpolant either singular (division by zero in a divided difference) or silently arbitrary (a spline solve with a repeated knot). A non-monotone t makes a bisection lookup return the wrong bracket without erroring at all.",
    "expected": "assert (numpy.diff(t) > 0).all(); a >= check is not sufficient. The message should name the offending row index and the filepath.",
    "failure_scenario": "A duplicated row produces either a NaN interpolant (loud) or, worse, an interpolant that is silently wrong over one interval - which for a table whose SN onset is a near-step could land exactly at the onset.",
    "repro": "Feed a t array with a repeated value and assert it raises.",
    "confidence": "high"
  },
  {
    "id": "S10-C-09",
    "file": "trinity/sps/read_sps.py",
    "line": 35,
    "class": "numerical",
    "severity": "S2",
    "claim": "EPSILON must not be used as a bare denominator guard in v = 2L/(pdot + eps); the physical 0/0 at late times requires a threshold test, not a nudge.",
    "evidence": "As the massive stars die, L_w -> 0 and pdot_w -> 0 together, so v = 2L/pdot is a genuine 0/0 limit. Adding eps only to the denominator gives v = 2L/eps, which for a small-but-nonzero L is an enormous velocity - potentially superluminal - while the correct answer is that BOTH the energy and momentum injection are zero and the velocity is dynamically irrelevant.",
    "expected": "Guard on the magnitude of L or pdot (e.g. if pdot < threshold: v = 0 and Mdot = 0), and enforce 0 <= v_w < c = 3.0664e5 pc/Myr at every row and every interpolated t. If EPSILON is instead a log-space floor (for log(L_SN = 0) -> -inf), it must be far below any dynamically relevant value - e.g. 1e-30 times the peak, not a convenient 1e-6, since 1e-6 in AU luminosity units is still 6e23 erg/s.",
    "failure_scenario": "A superluminal or NaN wind velocity enters rho_w = Mdot/(4 pi r^2 v) or any energy-partition expression; the run either crashes late or produces a vanishing wind density that silently zeroes a term.",
    "repro": "Evaluate v_w over the full table and a fine interpolated grid; assert isfinite and 0 <= v_w < 3.0664e5 pc/Myr everywhere.",
    "confidence": "medium"
  },
  {
    "id": "S10-C-10",
    "file": "trinity/sps/read_sps.py",
    "line": 38,
    "class": "sign",
    "severity": "S2",
    "claim": "A supernova term derived by subtraction (Lmech_SN = Lmech_total - Lmech_W, per SPEC-070's 'either/or' requirement) can be slightly NEGATIVE before the first supernova and must be clipped at zero.",
    "evidence": "SB99 prints log columns to three decimals [recalled], i.e. ~0.115% precision per value. Before the SN onset the total and the wind mechanical luminosity are the SAME number, so their independently-rounded de-logged values differ by O(0.2%) of L_W with random sign. Subtracting gives a residual that is negative roughly half the time, of order 2e37 erg/s per 1e6 Msun.",
    "expected": "L_SN = max(L_total - L_W, 0) and identically for pdot_SN; and L_SN must be exactly 0 for t < t_SN,onset (~3.5 Myr). The clip magnitude should be negligible (<1e-3 of L_W); if it ever isn't, the two columns are inconsistent and that is a different bug.",
    "failure_scenario": "A negative L_SN drains bubble energy for the first few Myr - a small but systematically-signed error in exactly the pre-SN window that TRINITY's science claim is about (SPEC-001: 'pre-supernova stellar feedback'). It also makes v_SN = 2L_SN/pdot_SN negative or NaN.",
    "repro": "After ingest, assert (Lmech_SN >= 0).all() and (pdot_SN >= 0).all(); assert Lmech_SN[t < 3.0] == 0.",
    "confidence": "medium"
  },
  {
    "id": "S10-C-11",
    "file": "trinity/sps/sps_columns.py",
    "line": 213,
    "class": "units",
    "severity": "S1",
    "claim": "A log10 column read as linear (or the reverse) must be caught by an absolute physical range gate, because it is otherwise silent in one direction.",
    "evidence": "Q_i for a young cluster is ~1e46.5 s^-1 per Msun; L_bol is ~1e36.9 erg/s per Msun (1000-1500 Lsun/Msun); L_mech,W is ~1e34 erg/s per Msun. Read as linear, a log column yields Q = 53 s^-1 and L = 40 erg/s - finite, positive, and ~1e50 times too small. Read as log, a linear column yields 10**1e53 = inf (loud) or, for a fraction like fi, 10**0.3 = 2 > 1.",
    "expected": "Post-conversion sanity gates: Qi/M_cluster in [1e45, 1e48] s^-1 Msun^-1 at t<3 Myr; Lbol/M_cluster in [1e36, 1e38] erg s^-1 Msun^-1; 0 <= fi <= 1; all arrays finite. Note SB99's *.snr mixes LOG rate columns with LINEAR progenitor-mass columns in one file [recalled], so a per-file 'everything is log' assumption is wrong - the per-column log flag in the ColumnSpec interface is the right design and must actually be honoured per column.",
    "failure_scenario": "The log-read-as-linear direction gives ~zero feedback; the bubble never expands; the run terminates with a plausible 'stalled/collapsed' verdict and no warning.",
    "repro": "Ingest, then assert the four range gates above for a 1e6 Msun cluster at the first table row.",
    "confidence": "high"
  },
  {
    "id": "S10-C-12",
    "file": "trinity/sps/sps_columns.py",
    "line": 278,
    "class": "regime",
    "severity": "S2",
    "claim": "The loader must warn when f_mass * sps_refmass = M_cluster falls below the IMF-sampling threshold (~1e4 Msun), because linear table scaling is invalid there.",
    "evidence": "N(>8 Msun) ~ M_cluster/100 Msun and N(>20 Msun) ~ M_cluster/400 Msun for a Kroupa IMF. Qi and L_w are dominated by the top few stars, so the effective sample is far smaller than N(>8) and the scatter far larger than Poisson. Below ~1e3 Msun the DISTRIBUTION of Qi spans orders of magnitude and is strongly skewed: the IMF-averaged table's mean is carried by rare draws, so the median cluster falls well below it. SPEC-073 notes param/paperII_grid_sweep.param reaches M_cluster = 100 Msun (mCloud=1e4, sfe=0.01), i.e. an expected massive-star count of ~1.",
    "expected": "A warning (not a hard error) at ingest keyed on M_cluster, with documented thresholds: safe >= 1e5 Msun; ~10-30% scatter 1e4-1e5 (and the WR phase unsampled); factor-of-few 1e3-1e4; invalid < 1e3. Nothing in the schema currently flags it.",
    "failure_scenario": "A published parameter grid contains cells the model cannot represent, presented with the same confidence as the valid cells.",
    "repro": "Run with M_cluster = 100 Msun and check whether any warning is emitted.",
    "confidence": "high"
  },
  {
    "id": "S10-C-13",
    "file": "trinity/sps/read_sps.py",
    "line": 134,
    "class": "units",
    "severity": "S1",
    "claim": "sps_refmass must be the normalisation the table was actually generated with, and a continuous-star-formation SB99 table (normalised to 1 Msun/yr, not to a mass) must not be accepted as a burst table.",
    "evidence": "f_mass = M_cluster/sps_refmass is only meaningful if sps_refmass names the table's own normalisation. SB99's instantaneous-burst mode normalises to the burst mass in the .input file (1e6 Msun is only the shipped default) while its continuous mode normalises to a star-formation RATE of 1 Msun/yr [recalled]. Dividing a continuous-SF table by a mass is a category error that raises nothing.",
    "expected": "A per-table declared sps_refmass, validated > 0, plus a shape discriminator: for a burst, Qi(20 Myr) must be far below Qi(1 Myr); for continuous SF, Qi rises to a plateau. Equivalence requirement: (table T, sps_refmass=1e6) and (T/1e6, sps_refmass=1) must produce identical drivers.",
    "failure_scenario": "A 1e6 error in every extensive driver, in either direction, with no diagnostic. Downward it stalls the bubble; upward it violates the mass- and energy-return bounds.",
    "repro": "Assert Qi[t~20 Myr] < 0.1*Qi[t~1 Myr] on ingest; and assert integral(Mdot_w + Mdot_SN, 0..40 Myr) < M_cluster and integral(Lbol, 0..40 Myr) < 0.007*M_cluster*c^2.",
    "confidence": "high"
  },
  {
    "id": "S10-C-14",
    "file": "trinity/sps/read_sps.py",
    "line": 38,
    "class": "coefficient",
    "severity": "S1",
    "claim": "Wind mechanical luminosity must never be substituted for bolometric luminosity or vice versa; they differ by ~1e3 for a young cluster.",
    "evidence": "Anchors: L_bol ~ 1e43 erg/s and L_w ~ 1e40 erg/s per 1e6 Msun, so L_w/L_bol ~ 1e-3 (range 1e-4 to 1e-2). The sharpest detector is the derived velocity: v = 2*L_bol/pdot_w = 2e43/1e32 = 2e11 cm/s, which is ~7x the speed of light. Independently, pdot_w * c / L_bol = 1e32*3e10/1e43 = 0.3, i.e. direct radiation pressure exceeds wind momentum by ~3x at early times (SPEC-071) - a well-known result any correct force-budget plot must show.",
    "expected": "L_w/L_bol in [1e-4, 1e-2] at t < 5 Myr; pdot_w*c/L_bol in [0.02, 2]; a hard guard v_w < c. A ratio near unity indicates a swap.",
    "failure_scenario": "L_bol used as the bubble energy source inflates dE_b/dt by ~1000x, producing a bubble that disperses every cloud regardless of parameters - which would look like a strong (and wrong) scientific result rather than a bug.",
    "repro": "Assert the three ratio gates on the ingested arrays at the earliest table rows.",
    "confidence": "high"
  },
  {
    "id": "S10-C-15",
    "file": "trinity/sps/sps_columns.py",
    "line": 213,
    "class": "units",
    "severity": "S1",
    "claim": "An ionising photon RATE column must not be confused with a cumulative photon NUMBER, nor a power column with a cumulative-energy column; SB99's *.power carries both power [erg/s] and cumulative energy [erg] side by side [recalled].",
    "evidence": "A cumulative column is monotone non-decreasing over the whole table; a rate column must PEAK and then FALL (Qi falls ~5 dex after 4 Myr; L_mech falls off a cliff after the last SN). This shape test is immune to every unit error. The magnitude signature is also distinctive: log(energy) - log(power) = log(t in seconds) ~ 13-14, so reading energy where power belongs inflates L by ~1e13.5.",
    "expected": "Rate canonicals (Qi, Lbol, Lmech_*, pdot_*, Mdot_*) must be non-monotone over a table spanning >5 Myr, specifically Qi[-1] < Qi[0]. Any canonical that is monotone non-decreasing across the full table is a cumulative column mis-mapped.",
    "failure_scenario": "A 1e13 luminosity inflation would be loud, but a cumulative Qi read as a rate rises with time and would give an HII region that grows without limit - plausible-looking in a plot and wrong.",
    "repro": "assert Qi[-1] < Qi[0] and assert not (numpy.diff(Qi) >= 0).all() for any table covering >5 Myr.",
    "confidence": "medium"
  },
  {
    "id": "S10-C-16",
    "file": "trinity/sps/sps_columns.py",
    "line": 94,
    "class": "other",
    "severity": "S2",
    "claim": "The ionising split must satisfy Li + Ln = Lbol exactly, 0 <= fi <= 1, and the exact photon-energy bound Li/Qi >= 13.6 eV.",
    "evidence": "Li and Ln partition Lbol by definition (SPEC-074). Any photon counted in Qi is by construction above the Lyman edge, so the mean ionising photon energy Li/Qi cannot be below 13.6 eV = 2.179e-11 erg - a theorem, not an estimate. Realistically <hv> ~ 15-25 eV, softening with age. In AU (Li in Msun pc^2 Myr^-3, Qi in Myr^-1) the ratio has units of energy = 1.90148e43 erg, so the hard bound is Li/Qi >= 1.146e-54 and the expected band is 1.3e-54 to 2.1e-54.",
    "expected": "Assert Li + Ln == Lbol to machine precision when both are supplied; assert 0 <= fi <= 1; assert 13.6 eV <= Li/Qi <= ~50 eV at every row. Also, fi must be strongly TIME-DEPENDENT (0.1-0.3 at t<3 Myr, <0.01 by 10 Myr) because Qi collapses ~5 dex while Lbol declines only as ~t^-1.3; a hard-coded constant ionising fraction is physically wrong after ~4 Myr.",
    "failure_scenario": "A constant fi makes the radiation-pressure and Stromgren terms track each other forever, suppressing the real post-4 Myr decoupling of ionising output from bolometric output - which is precisely the epoch when TRINITY decides dispersal vs re-collapse.",
    "repro": "Assert the three bounds row-wise; and assert fi[t~10 Myr] < 0.5*fi[t~1 Myr].",
    "confidence": "high"
  },
  {
    "id": "S10-C-17",
    "file": "trinity/sps/update_feedback.py",
    "line": 98,
    "class": "coefficient",
    "severity": "S2",
    "claim": "Totals must be formed by adding powers and momenta, never by averaging velocities: L_tot = L_W + L_SN, pdot_tot = pdot_W + pdot_SN, and any single effective injection velocity must be 2*L_tot/pdot_tot.",
    "evidence": "Energy and momentum fluxes are additive; velocities are not. v_tot = 2(L_W+L_SN)/(pdot_W+pdot_SN) is not any weighted mean of v_W and v_SN. The two components differ substantially: v_W ~ 1500-3000 km/s (hot-star winds) while SN ejecta at 1e51 erg per ~10 Msun give v ~ sqrt(2E/M) ~ 3200 km/s and higher for lower ejecta masses.",
    "expected": "Lmech_total == Lmech_W + Lmech_SN and pdot_total == pdot_W + pdot_SN to machine precision (or to log-rounding precision if 'total' is tabulated independently); any effective velocity used downstream built from the totals, consistently.",
    "failure_scenario": "Pairing v_W with pdot_total (or v_SN with pdot_W) mixes populations and breaks L = 1/2 Mdot v^2 by an O(1) factor from the moment SNe switch on - i.e. exactly at the phase transition the code is designed to resolve.",
    "repro": "assert allclose(Lmech_total, Lmech_W + Lmech_SN) and allclose(pdot_total, pdot_W + pdot_SN); check which velocity feeds rho_w downstream.",
    "confidence": "high"
  },
  {
    "id": "S10-C-18",
    "file": "trinity/sps/read_sps.py",
    "line": 38,
    "class": "coefficient",
    "severity": "S2",
    "claim": "Cold-gas mass loading (SPEC-072) must hold L fixed and recompute BOTH v and pdot: v -> v/sqrt(1+f), pdot -> pdot*sqrt(1+f).",
    "evidence": "Entraining cold mass conserves energy, not momentum. With Mdot_tot = (1+f) Mdot_w at fixed L: v_eff = sqrt(2L/Mdot_tot) = v_w/sqrt(1+f) and pdot_eff = sqrt(2 L Mdot_tot) = pdot_w*sqrt(1+f). Reducing v while keeping the tabulated pdot implicitly changes L by (1+f)^(-1/2), since L = pdot*v/2.",
    "expected": "With FB_mColdWindFrac = f: pdot_out/pdot_in == sqrt(1+f) exactly, v_out/v_in == 1/sqrt(1+f) exactly, L_out/L_in == 1 exactly. Both fractions default to 0, so this path may be entirely untested.",
    "failure_scenario": "A silent energy leak proportional to the mass loading, in a code path that is off by default and therefore never exercised by the default test suite - it will surface only in a paper run that enables it.",
    "repro": "Set FB_mColdWindFrac to 1.0 and assert pdot scales by exactly sqrt(2) and L is unchanged.",
    "confidence": "high"
  },
  {
    "id": "S10-C-19",
    "file": "trinity/sps/sps_columns.py",
    "line": 445,
    "class": "units",
    "severity": "S1",
    "claim": "The time column's unit must be resolved explicitly; SB99 writes TIME in YEARS, linear [recalled], and mistaking years for Myr is a 1e6 error that is completely silent.",
    "evidence": "If a years column is treated as Myr, the whole 40 Myr evolution is mapped to 4e7 Myr, so a simulation running to t = 30 Myr never leaves the first table row: it sees the t=0 feedback forever - no SN onset, no decline in Qi, no decline in Lbol. Every value is finite, positive and of the right magnitude; only the TIME DEPENDENCE is destroyed.",
    "expected": "t[-1] in Myr must land in [10, 1e3] for any usable table; a t[-1] ~ 1e7 is years and a t[-1] ~ 1e14 is seconds. The conversions are yr -> Myr x 1e-6 and s -> Myr x 3.1689e-14.",
    "failure_scenario": "A run with permanently-young feedback: no supernovae ever fire, Qi never falls, the bubble is driven at peak strength indefinitely. Indistinguishable from a physically strong-feedback result without checking the SPS time axis.",
    "repro": "After ingest assert 10 <= t[-1] <= 1e3 (Myr) and assert Qi[-1] < 0.1*Qi[0].",
    "confidence": "high"
  },
  {
    "id": "S10-C-20",
    "file": "trinity/sps/read_sps.py",
    "line": 285,
    "class": "numerical",
    "severity": "S2",
    "claim": "If the interpolation is done in log space, the exactly-zero pre-onset supernova rows must be floored, or log(0) = -inf produces NaN.",
    "evidence": "L_SN, pdot_SN and Mdot_SN are exactly zero before ~3.5 Myr and again after ~40 Myr. Log-space interpolation of an array containing exact zeros yields -inf at those knots and NaN in the interpolant. Conversely, SB99 sometimes writes a placeholder such as 0.000 or -99.000 in the log SN columns before onset [recalled, low confidence] - de-logged these give 1 erg/s or 1e-99 erg/s respectively, both dynamically negligible but only the second obviously so.",
    "expected": "isfinite over every ingested and interpolated array. Any log floor must be far below any dynamically relevant value (e.g. 1e-30 of the peak). L_SN before onset must be either exactly 0 or utterly negligible (<1e-9 of L_W), never comparable to L_W.",
    "failure_scenario": "NaN propagates into dE_b/dt and the solver either fails loudly (best case) or produces NaN outputs written to dictionary.jsonl.",
    "repro": "assert numpy.isfinite(...).all() on every canonical array and on a 1e4-point interpolated grid; assert Lmech_SN[t<3] < 1e-9 * Lmech_W[t<3].max().",
    "confidence": "medium"
  },
  {
    "id": "S10-C-21",
    "file": "trinity/sps/update_feedback.py",
    "line": 80,
    "class": "state",
    "severity": "S2",
    "claim": "SPSFeedback's positional protocol (__iter__/__getitem__/__len__) exposes several same-unit, same-magnitude fields, so a positional swap is undetectable by any dimensional or range check and needs an explicit ordering test.",
    "evidence": "Lmech_W, Lmech_SN and Lmech_total are all powers of order 1e40 erg/s; pdot_W, pdot_SN and pdot_total are all forces of order 1e32 dyn. No unit, sign, magnitude or finiteness check can distinguish them. The only physical discriminators are (a) the derived velocities: 2*L_W/pdot_W ~ 2000 km/s vs 2*L_SN/pdot_SN ~ 3000-10000 km/s, and (b) the sum rules L_total = L_W + L_SN, pdot_total = pdot_W + pdot_SN.",
    "expected": "A test asserting positional order matches named access for every field, plus the two sum rules, plus the two velocity bands. Any consumer that unpacks positionally is one field-reorder away from a silent physics error.",
    "failure_scenario": "Adding a field to SPSFeedback reorders the tuple; a downstream positional unpack now feeds the SN luminosity where the wind luminosity belongs. All tests that only check magnitudes still pass.",
    "repro": "assert list(fb) == [getattr(fb, name) for name in fb._field_order] and assert len(fb) equals the number of fields; assert the sum rules.",
    "confidence": "medium"
  },
  {
    "id": "S10-C-22",
    "file": "trinity/sps/read_sps.py",
    "line": 38,
    "class": "regime",
    "severity": "S3",
    "claim": "Supernova energy delivered as a smooth continuous rate is a valid approximation only for N_SN >> 1; below ~10 supernovae the discrete-event nature changes the bubble evolution qualitatively.",
    "evidence": "N_SN ~ M_cluster/100 Msun (Kroupa). For M_cluster = 1e6 Msun, N_SN ~ 1e4 over ~36 Myr and a smooth rate of ~3e-4 /yr is an excellent approximation (the injected power ~1e40 erg/s is the correct ensemble AND individual value). For M_cluster = 1e3 Msun, N_SN ~ 10 and the true injection is ten 1e51 erg blasts separated by ~3 Myr - long enough that a bubble can cool and stall between them, which a smooth rate can never reproduce.",
    "expected": "The smooth-rate treatment is correct for the 1-D model and should not be changed, but the validity limit belongs in the same warning as S10-C-12. An integral check pins the units end-to-end: integral(L_SN dt) / 1e51 must equal N_SN, and N_SN/M_cluster must be ~1/100 Msun^-1 (range 1/50 to 1/150).",
    "failure_scenario": "Low-mass grid cells report a smooth, sustained late-time expansion that no real cluster of that mass would produce.",
    "repro": "Compute trapz(Lmech_SN, t)/1e51 in cgs and compare with M_cluster/100.",
    "confidence": "high"
  },
  {
    "id": "S10-C-23",
    "file": "trinity/sps/sps_columns.py",
    "line": 33,
    "class": "units",
    "severity": "S4",
    "claim": "_L_SUN_ERG_S must be the IAU nominal solar luminosity 3.828e33 erg/s (= 3.828e26 W).",
    "evidence": "IAU 2015 Resolution B3 fixes the nominal solar luminosity at 3.828e26 W. Older values in circulation are 3.839e33 and 3.846e33 erg/s, a spread of 0.5%. Via the Weaver scaling R ~ (L/rho)^(1/5) that is a 0.1% radius error - genuinely negligible, but there is no reason to carry a wrong constant.",
    "expected": "3.828e33 erg/s; anything outside [3.8e33, 3.9e33] is an error of a different kind (e.g. the solar constant 1.361e6 erg/s/cm^2, or a W/erg-s confusion of 1e7).",
    "failure_scenario": "A 1e7 slip (W vs erg/s) would be caught by the range gates in S10-C-11; a 0.5% slip would not, and is harmless.",
    "repro": "assert 3.8e33 < _L_SUN_ERG_S < 3.9e33.",
    "confidence": "high"
  },
  {
    "id": "S10-C-24",
    "file": "trinity/sps/read_sps.py",
    "line": 134,
    "class": "other",
    "severity": "S2",
    "claim": "The bundled-table path and the user-table path must produce byte-identical canonical arrays for the same physical table.",
    "evidence": "Two loaders for the same physical object is the classic site of a units or f_mass divergence: e.g. the bundled table pre-converted to AU and scaled while the user path converts from cgs and scales separately, or f_mass applied in one path and not the other. Nothing in the physics distinguishes them, so only a round-trip test can.",
    "expected": "Write the bundled table out as a user table with explicit column specs (units and log flags), re-ingest through _read_sps_user with the same f_mass, and require identical canonical arrays to machine precision.",
    "failure_scenario": "Users' custom SPS tables are silently scaled differently from the bundled one, so published bundled-table results are not reproducible with an equivalent user table.",
    "repro": "Round-trip the bundled table through the user-column path and assert array equality on every canonical.",
    "confidence": "medium"
  },
  {
    "id": "S10-C-25",
    "file": "trinity/sps/read_sps.py",
    "line": 285,
    "class": "divergence",
    "severity": "S3",
    "claim": "Below the table's first time the interpolant must clamp (not extrapolate), because a burst's feedback is flat over 0-0.5 Myr and a spline extrapolating backwards can undershoot to negative values on the very first timestep.",
    "evidence": "No star more massive than ~100 Msun has evolved off the main sequence in the first ~0.5 Myr, so Qi, Lbol, L_w and pdot_w are essentially constant there. SB99 grids commonly start at 1e4-1e5 yr rather than exactly 0 [recalled]. Backward cubic extrapolation from a rising WR-era trend can therefore produce a negative or near-zero driver precisely at t=0, where the ODE solver has no established step size and is least robust.",
    "expected": "For t < t_min, return the first row's values. This is the OPPOSITE of the correct t > t_max behaviour (S10-C-04), and the asymmetry is physical, not arbitrary: the pre-table region is flat, the post-table region is a cliff.",
    "failure_scenario": "The very first solver step receives a negative or vanishing driving luminosity, giving a spurious initial contraction or a step-size collapse that shows up as a slow/failed run rather than as an obvious error.",
    "repro": "Evaluate the interpolant at t = 0 and at t = 0.5*t_min; assert equality with the first row and non-negativity.",
    "confidence": "medium"
  }
]
```
