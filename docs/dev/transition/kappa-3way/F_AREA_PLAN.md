# F_AREA — the area-faithful knob: physics, calibration, wiring, and the pre-registered campaign

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

**Status (2026-08-02):** 🔵 actionable — physics derived, code path verified (zero `trinity/`
changes needed for the experiment), predictions pre-registered from fresh data, campaign designed
(**A0 offline free + A1 one bench8 submission, 514 arms**). **Nothing has been run.** Every number
in §6 is a prediction. *(2026-08-02 revision: the maintainer's HPC can hold <1000 arms in one
array, so the original 48-arm coarse grid was redesigned — fine dose ladder, in-campaign
single-knob margins, band-config fire map promoted from "optional A2" to a first-class tier,
determinism pairs. The predictions in §6 are unchanged: nothing had been run, and they are
grid-independent — a finer grid only tightens the brackets they are judged against.)*

---

## 0. TL;DR

TRINITY's three cooling knobs each move one channel of the wind/shell interface and each fails a
different test: f_κ (conduction) never reaches the trigger on the diffuse bench; f_A (radiation)
suppresses evaporation — the **wrong sign** for a wrinkled interface; f_mix is a scalar on the
answer with no structure at all, and its calibration lead just evaporated under stale-row exclusion
(`FINDINGS §12`). The physical motivation for *all* of them is the same: turbulent mixing wrinkles
the contact discontinuity, so its true area exceeds 4πR², raising **every** interface flux together.

**f_area applies f_κ = f_A = f — one shared constant on both channels — and this is not a
heuristic: in the thin-layer limit it is the *exact* 1-D representation of multiplying interface
area by f** (§2, an identity: the layer's T-profile is invariant while every flux scales by
exactly f; verified symbolically-trivially, numerically to machine precision, and at the code
level, where the dTdrr radiative term carries f_A/f_κ and cancels at equal doses).

From fresh measured inputs only (θ₀ and the interface share per bench), the zeroth-order model
predicts band entries **≈ 3.6 / 5.0 / 7.3** with **spread ≈ 2.1×** — better than every measured
single knob, via a real cancellation (diffuse clouds have lower θ₀ but a *higher* interface
share). The construction also predicts **Ṁ rising with dose** — the unique area signature no
shipped knob shows. Both are falsifiable, and the campaign is **one 514-arm submission** (the HPC
holds <1000 arms per array): a 24-dose fine ladder on the three clean benches with matched-dose
single-knob margins in the same code state, the six band configs, and determinism pairs.

---

## 1. Charter and constraints from the record

- **The charter is `pdv-trigger/PLAN.md` item 9** ("the area-faithful successor — the one
  experiment that would settle it"): *"a genuine fractal-area increase should move the whole
  interface budget — conduction, evaporation and radiation together — predicting **ṁ RISES with
  dose**."* Registered 2026-07-29, un-actioned. This plan is that design note.
- **The kappa-3way campaign supplies the motivation** (`FINDINGS §10`): at the operating doses,
  f_A's Ṁ ratio is 0.988→0.855 (wrong sign), f_κ's crosses below 1 at f≈7 (right sign only
  briefly), f_mix's is ≡1 (no response). *No shipped knob raises mass loading.* The mechanism
  ranking (f_κ > f_A > f_mix) is the reverse of the calibration ranking — and after `§12`, the
  calibration ranking itself now favours f_A (2.71× vs f_mix 3.70× on solved-row trigger entries).
- **SC-0 compatibility** (`pdv-trigger/FA_STATE_COUPLED.md:39-46, :255-261`): the state-coupled
  f_A ladder is TERMINAL; reopening requires *"a NEW candidate law … entering at SC-0, not SC-1"*
  and *"a new physical idea for the truncation scale / interface area, not another fit"*
  (`FINDINGS §15k`). **f_area complies:** it is a new *construction* (a derived two-channel
  identity, §2) with a *predicted* spread (§6), entering through an SC-0-style offline screen
  (Phase A0) before any wiring; and it directly addresses the named gap — the interface-area
  question. What it does **not** yet supply is a derived *value* (§3.4); the value stays measured,
  which is why §8's ship path is explicitly gated on the maintainer.
- **Bounds inherited:** any area-motivated κ enhancement is capped **≲30 with a smooth limiter,
  never a hard min** (`SOURCE_TERM_DESIGN.md:270-272` — the κ_mix hard-max lesson);
  `cooling_boost_kappa='auto'` is **forbidden** in combination (`registry.py:387`); α_A must not be
  tuned to rescue a fit (`FA_STATE_COUPLED.md:289-291`).

## 2. The physics — why f_κ = f_A = f IS the area knob (an identity, not an analogy)

### 2.1 The wrinkle picture and its unambiguous signature

If mixing wrinkles the contact discontinuity so its true area is A = f·4πR², and the wrinkle
radius of curvature is large compared to the interface-layer thickness (the thin-layer limit),
then each surface patch runs the same local 1-D conduction–radiation problem and there are simply
f× more patches. Every interface flux scales together: conductive heat flux ∝ A, radiated power
∝ A (layer volume at fixed thickness), **evaporative mass flux ∝ A**. The signature is therefore
not "more cooling" — any knob does that — but **all three fluxes rising with one factor, Ṁ
included.**

### 2.2 The invariance identity

The quasi-static interface layer obeys

```
d/dz ( C T^{5/2} dT/dz ) = n² Λ(T)          (conduction balances radiation)
```

Apply **f_κ = f_A = f**, i.e. C → fC and Λ → fΛ. The constant f factors out of both sides
*identically*, so:

- **T(z) through the layer is exactly invariant** (same equation, same boundary temperatures);
- conductive flux q = fC T^{5/2} dT/dz = **f × baseline**;
- radiated power per unit area = ∫ f n²Λ dz = **f × baseline**;
- evaporative mass flux per area (enthalpy balance across the front: (5/2)(ṁ/A)k_BT/μ = q_in −
  q_rad, both sides ×f) = **f × baseline**.

Same local structure per patch, f× the throughput: **exactly** area multiplication. Verified
numerically to machine precision on an arbitrary stiff cooling curve (max|ΔT/T| = 0.0 at f = 2
and f = 8; flux ratios exactly 2.000000 and 8.000000).

Contrast the single knobs, which *cannot* represent area: under f_κ alone the layer *restructures*
(thickness ∝ √(κ/Λ) grows as √f, per-area flux ∝ √(κΛ) grows only as √f); under f_A alone the
layer thins and the evaporative flux *falls* — the measured wrong sign.

### 2.3 The identity holds inside TRINITY's actual ODE

`trinity/bubble_structure/bubble_luminosity.py:441` (verified at source, 2026-07-30):

```python
dTdrr = ( Pb / (f_κ · C_thermal · T^{5/2}) · [ (β + 2.5δ)/t + 2.5(v−v_t)·T'/T  −  f_A·dudt/Pb ]
          − 2.5 T'²/T − 2 T'/r )
```

The radiative source term carries the combined factor **f_A/f_κ — which cancels exactly at equal
doses** in the interface band (`T < 10^5.5 K`, `_T_INTERFACE_BAND`, `:60-65`). What deviates from
strict invariance, precisely enumerated:

1. the advective/time-dependent terms `(β + 2.5δ)/t` and `2.5(v−v_t)T'/T` are **divided by f_κ** —
   the quasi-static-layer correction, suppressed as 1/f;
2. above the band (T > 10^5.5), only 1/f_κ acts (no f_A) — but there radiation is subdominant by
   construction;
3. the boundary anchor `dR2 ∝ f_κ/dMdt` (`:398-412`, Weaver Eq 44) and the fsolve seed
   `∝ f_κ^{2/7}` (`:304-308`, Weaver Eq 33). Note the self-consistency: **if the converged dMdt
   scales ≈ f (the identity's prediction), then dR2 ∝ f/f is invariant too** — the anchor sits
   still. That is a directly measurable A0 observable.

The identity also **dissolves the record's standing objection** to f_A as an area factor
(`registry.py:388`: the L2+L3 band is *"NOT the interface surface itself"*; `FINDINGS §22`): a
volume-integrated emission × f equals area × f **exactly when the layer thickness is invariant** —
which it is under the combined knob, and is not under f_A alone.

### 2.4 What f_area does NOT capture (stated before anyone asks)

1. **Mass loading of the hot phase by mixed-in cold gas** — κ_mix Rung-B territory, shelved
   (born-saturated at 10⁵–10⁸× Spitzer, `KMIX_SELFCONSISTENT.md`). f_area moves fluxes through the
   interface; it does not add a cold-gas entrainment term.
2. **Time-dependent area**: Lancaster's A ∝ R_b^{2+d}/ℓ^d *grows* as the bubble grows; f_area is
   a constant. (A dose-ramp is a conceivable follow-up, not this campaign.)
3. **The L1 interior** (60–77% of L_cool; fresh interface share s = 25–34%, §6.1) is untouched —
   correct for an area knob (no interface there), but it caps the θ response slope at s per unit f.
4. **Saturated conduction** (Cowie–McKee): unmodelled, inherited from the base code.
5. **The thin-layer assumption itself**: at large f the implied wrinkle scale approaches the layer
   thickness and the picture breaks — one more reason for the dose cap of 24 (§5).

## 3. Literature calibration — what the papers actually support

*(Web-verified 2026-07-30 against arXiv/ADS search records plus the repo's [V]-graded paper
transcriptions; full citations at bottom.)*

### 3.1 The area law exists and is measured

- **Lancaster+2021a (ApJ 914, 89 — fractal theory):** interface area
  `A_b(R_b; ℓ) = 4π α_A R_b² (R_b/ℓ)^d` with **adopted d = 1/2** and α_A ~ 1 (verbatim: an
  order-unity fudge, not a measured constant). The effective enthalpy flux uses
  `v_equiv(ℓ) = v_t(ℓ)(R_b/ℓ)^d` (Eq 12); with the cascade index p = 1/2, v_equiv is
  scale-independent — the fractal cancellation that makes their Θ → 0.9–0.99 robust.
- **Lancaster+2021b (ApJ 914, 90 — sims):** **measured d ≈ 0.4–0.7** (D ≈ 2.4–2.7) across the
  same 12-cloud suite our benches map onto; Θ = 0.9–0.99 for all models.
- **Fielding+2020 (ApJL 894, L24):** independent TRML measurement, **D = 5/2 exactly** (d = 1/2),
  `A_λ/A_L = (λ/L)^{−1/2}`; cooling resolution-independent because area × local speed cancels —
  the same logic as §2.2's identity.
- **Lancaster+2024 (ApJ 970, 18):** couples the momentum enhancement directly to the area excess:
  `α_p = (3/4)·(V_w/4)/⟨v_out⟩ · (4πR_w²/A_w)` — published precedent for treating the area ratio
  as *the* dynamical quantity.
- **El-Badry+2019 (MNRAS 490, 1961):** the 1-D counterpart. κ_mix = λδv·ρk_B/μm_p (Eq 21, a
  *diffusivity*, applied as max() against Spitzer — not a Spitzer multiplier); θ(λδv, n) closed
  form (Eqs 37–38, A_mix = 3.5 fit); **Eq 47**: ṁ ∝ (C/6×10⁻⁷)^{2/7} · (1−θ)^{37/35}/θ^{2/7} —
  the C-channel and θ-channel that f_κ and f_A separately reproduce (`FINDINGS §23`, 0.34–1.63%).

### 3.2 What magnitude of f is *plausible* from the literature

No paper publishes "A_eff = N × 4πR²" — the area is scale-dependent and the papers quote d. But
with measured d ≈ 0.5 and the truncation scale ℓ anywhere near the L21b grid scales
(Δx = 0.01–0.15 pc) at R_b ~ 2.5–20 pc, the implied enhancement is **A_eff/4πR² ~ 10–30**
(⚠️ that arithmetic is ours, not theirs). Our predicted operating range — 3.6–7.3 zeroth-order,
inflating toward ~10–25 with back-reaction (§6) — **sits inside the physically motivated bracket**,
and comfortably under the inherited ≲30 cap. That is the honest calibration statement: the
literature bounds the *plausible range*; it does not hand us the value.

### 3.3 ⛔ Why the value cannot be derived from Eq 11 today

Already falsified in-repo (`LANCASTER_REFERENCE.md:281-294`, `FA_STATE_COUPLED.md:297-299`):
evaluating Lancaster's own truncation closure (`v_t(ℓ_cool)·t_cool = ℓ_cool`) with TRINITY's
cooling table gives **ℓ_cool ~ 10⁻¹⁵ pc** — below every physical and numerical scale — implying
f ~ 10⁹–10²⁴. *"C2 survives only if someone supplies an independent physical truncation scale."*
This plan does **not** claim to have solved that. Candidate closures worth testing *after* A1
(recorded here so a future visit starts somewhere): (a) the **conduction-layer (Field) length**
δ ~ √(κT/n²Λ) as the smoothing scale — conduction erases wrinkles below it; first estimate gives
f ~ 300–3000, still high, so it needs (b) **Lancaster's own saturation cap** (Eq 12's v_equiv may
not exceed Eq 15's v_hot ≈ V_w/(6α_p−2)) — the flux-limited regime where Θ saturates near 1
regardless of ℓ. Either way: **in this campaign the value of f_area is a measured quantity**, with
the literature bracket as the plausibility check.

### 3.4 Is that "another fit" (the thing SC-0 forbids)?

No — with one honest caveat. C3 (SC-0's rejected fitted constant) was a *free scalar with no
structural content* and it failed because the required doses spread 56×. f_area is a **derived
two-channel construction** whose *spread* is predicted in advance (≈2.1×, §6.2) from an identity
plus two measured inputs; the campaign tests that prediction. If the measured spread blows up to
C3-like values, f_area dies by the same pre-registered sword (§7 TERMINAL). The caveat: the
*absolute* value is still calibrated, not derived — which is why §8 keeps any production ship
behind a maintainer gate and the truncation-scale question (§3.3) open.

## 4. Where in the code — the complete interaction surface (verified 2026-07-30)

**The experiment requires ZERO changes to `trinity/`.** Empirically verified: a `.param` setting
`cooling_boost_kappa 8` **and** `cooling_boost_fA 8` loads cleanly through `read_param` at full
strength. The only guard anywhere is `_validate_cooling_boost_fA` (`registry.py:117-149`), which
**warns** (never raises) on cross-knob combination. For bench8 that warning is *desirable
provenance* — one line per run, captured by `log_file True`, confirming both knobs were live.

The complete runtime read surface (nothing else in `trinity/` reads either knob):

| site | knob | role |
|---|---|---|
| `bubble_luminosity.py:304` | f_κ | dMdt fsolve **seed** (Weaver Eq 33) — `∝ f_κ^{2/7}` |
| `bubble_luminosity.py:398` | f_κ | R2 boundary **anchor** (Weaver Eq 44) — `dR2 ∝ f_κ/dMdt` |
| `bubble_luminosity.py:441` | f_κ | `dTdrr` conduction prefactor |
| `bubble_luminosity.py:435` | f_A | `dudt` source boost, band-gated `T < 10^5.5` |
| `bubble_luminosity.py:845` | f_A | L2/L3 loss-integral scaling (post-solve) |

dMdt itself is solved by fsolve on the **v(R1) = 0** boundary condition (`:236-266`, residual
`:311-388`) — f_κ enters it analytically (seed + anchor), f_A only implicitly through the solved
structure. Neither knob touches `L_leak`, the shell, the momentum phase, or the cooling tables.
`cooling_boost_mode`/`fmix` stay at their defaults in every arm — the §16 double-boost path is
f_mix-specific and f_A is the identity on it (`FINDINGS §16`, verified).

**Known bug to fix in passing (independent of bench8):** `make_kappa_reopen_params.py:60-61`
claims Gate 3's `read_param` load-check "catches a cross-knob/double-boost combination" — it
cannot (warnings aren't exceptions). The single-knob guarantee always lived in
`test_bench7_params.py:82`. One-line docstring fix.

**Option B (ship path only, NOT this experiment):** a first-class `cooling_boost_farea` param whose
resolver sets both knobs — cleaner semantics, enforces equality by construction, and retires the
warning. Deliberately deferred: writing production code before the measurement is the SC-1 mistake
SC-0 exists to prevent.

## 5. The campaign — A0 (free) → A1 (one bench8 submission, 514 arms)

*(2026-08-02: redesigned from the original 48+24 two-phase layout. The maintainer's HPC runs
<1000 simulations in one array with the standing `sync_bench.sh up → submit → reduce → down`
workflow, so arm scarcity is no longer the binding constraint — the binding constraints are the
1:30:00 partition cap per arm and the ONE-SHOT reduce. The redesign spends the capacity on
resolution and same-code-state controls, not on new physics questions.)*

### Phase A0 — offline frozen-state screen (in-container, ~minutes, run BEFORE any HPC)

New builder `pdv-trigger/data/make_farea_screen.py`, modelled on `make_fkappa_leverage.py` (which
already runs the full production solve at the two committed captured states with the real gated
knobs — both states carry the full registry params, so setting both knobs bypasses nothing).

Grid: f ∈ {1, 2, 4, 8, 16} × three modes {κ-only, fA-only, **combined**} × 2 captured states.
Measured per call: `bubble_dMdt`, `bubble_LTotal`, L2/L3, the full `T(r)` array, `r2_prime`,
solver health. Deliverable: `data/farea_screen.csv` + figure.

**Scope per `FINDINGS §12a`: falsifier and sign-check ONLY — no A0 number is a full-run
calibration.** The one principled exception: the invariance identity (§2.2) is itself a per-call
statement, so A0 tests it in its own regime.

**A0 pre-registered checks (PASS/FAIL each):**

| # | check | PASS bar | FAIL ⇒ |
|---|---|---|---|
| A0.1 | T-profile invariance ranking | max\|ΔT/T\|(combined) **<** max\|ΔT/T\|(κ-only) at every f, both states | identity fails in the real solver — STOP, campaign not submitted |
| A0.2 | per-call Ṁ superadditivity | Ṁ-ratio(combined, f) **>** Ṁ-ratio(κ-only, f) = ~f^{0.28} at every f; layer-invariance limit f^{1.0} is the upper reference | channels don't compose — STOP |
| A0.3 | L_total linearity | ratio ≈ 1 + s·(f−1) within ±30%, s measured from the same state | bookkeeping surprise — investigate before submitting |
| A0.4 | viability ceiling | combined solve healthy (fsolve converges, dMdt > 0, T monotonic) to **f ≥ 8** at both states | grid cap lowered accordingly; if ceiling < 4, STOP |
| A0.5 | anchor invariance | \|Δr2_prime\|(combined) < \|Δr2_prime\|(κ-only) at every f | §2.3's self-consistency fails — flag, proceed with caution |

### Phase A1 — the HPC campaign: **bench8**, 514 arms, ONE submission

**The fine dose ladder F24** (24 values, geometric-ish, densest where the zeroth-order entries
sit at 3.5–7.3 so every band entry is bracketed to ≲14% instead of the old ladder's ~50%):

```
F24 = {1.5, 1.75, 2, 2.25, 2.5, 2.75, 3, 3.25, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7,
       8, 9, 10, 12, 14, 16, 20, 24}
```

Cap stays 24 per the inherited ≲30 bound and the stiffness corner (§9). Non-integer doses get
zero-padded tags (`f01p50`, `f12p00`, …) so the sbatch dispatcher's **lexical `ls | sed` ordering
stays deterministic** — a builder emit-gate, not a convention.

| tier | grid | arms | what it buys |
|---|---|---|---|
| **A1-core** | 3 clean benches (bench3/2/1, same committed clouds as bench5, G2-pinned) × F24 × {prod, diag} | **144** | the f_area entries (PA1), spread (PA2), Ṁ ladder (PA3), exponent q from 24 points not 8 (PA4), ΔL_cool–ΔṀ correlation (PA5) |
| **A1-margin** | same 3 benches × F24 × {κ-only, fA-only} × {prod, diag} | **288** | matched-dose single-knob margins **in the same code state and the same reduce** — every (bench, f) is a {combined, κ-only, fA-only} triplet, so PA1's "beats the best single knob" and PA4's superadditivity are same-campaign comparisons, not cross-campaign ones. The coarse bench5r/6r/7 anchors remain the *pre-registration* inputs (§6.1); the margins refine, and must reproduce, them (GA2b) |
| **A1-band** | 6 band configs × f_area ∈ {1.5, 2, 2.5, 3, 3.5, 4, 5, 6, 8, 10, 12, 16} × prod | **72** | the fire map + CONDENSE boundary (PA6) — formerly "optional A2", now first-class: does the combined knob shift the CONDENSE onset that broke f_κ's whole-band coverage? |
| **A1-determ** | 5 flip-adjacent (bench, f, mode) arms × 2 identical repeats | **10** | K3-style fate determinism at the boundaries the fine grid will expose |
| | | **514** | one `--array=1-514` submission, auto-sized by `submit` |

- Combined arms set `cooling_boost_kappa f` **and** `cooling_boost_fA f` with **equal values**;
  margin arms set exactly one knob. **Mode-correctness is enforced by the builder and by the
  test** (`assert set(active) == expected_for_mode` and, for combined, `kv[kappa] == kv[fA]`: the
  "one shared constant" is a mechanical guarantee, not a convention). `stop_t 5`, prod/diag
  two-arm protocol, single process each — all standing 📏 rules.
- **f = 1 baselines: NOT re-run.** f = 1 is byte-identical to the bench5r `__none` arms in every
  mode (all knobs gated); GA6 spot-checks the load, no arm is spent on it.
- **Tooling:** `runs/make_farea_params.py` (emit gates G1-style: GMC plausibility, exact L21b
  mapping ≤2%, `read_param` load-check *plus a positive check that the cross-knob warning fires
  exactly once per combined file and never in a margin file*, count/uniqueness, per-mode knob
  checks, tag-ordering check); `runs/run_bench8.sbatch` (copy of bench7's: 1:30:00 partition cap,
  `.exit_code`/`.duration` markers); `sync_bench.sh` gains `bench8) SRC=bench8; … EXTRA=$COLS ;;`
  — the existing `--extra-cols Pb,bubble_dMdt,bubble_L2Conduction,bubble_L3Intermediate` is
  exactly what the analysis needs, and **the reduce stays ONE-SHOT**: any additional column must
  be declared before the first reduce (514 arms make a re-run even more expensive than before —
  double-check `$COLS` *before* `submit`). `test/test_bench8_params.py` mirrors bench7's
  (byte-identical regeneration, counts, protocol, per-mode knob rules). bench7's own test is
  **not relaxed**.
- **What the capacity is deliberately NOT spent on:** doses > 24 (thin-layer breakdown + the ≲30
  cap, §2.4/§1), re-running f = 1, extra stochastic replicates beyond A1-determ (the code is
  deterministic — K3 measured 5/5 bit-identical), or new physics questions (dose ramps,
  state-coupled laws) that belong behind the §8 decision tree, not inside the measurement.

## 6. Pre-registered predictions — frozen HERE, before any run

### 6.1 Measured inputs (all FRESH, 2026-07-30)

| bench | θ₀ = θ_max at f=1 (prod) | interface share s = (L2+L3)/L_cool (median, diag) |
|---|---|---|
| bench3 (n̄=5520) | 0.580 | 0.250 |
| bench2 (n̄=690) | 0.447 | 0.285 |
| bench1 (n̄=43) | 0.301 | 0.342 |

Single-knob anchors: trigger entries f_κ {6.53, 13.8, never}, f_A {11.3, 23.5, 30.8},
f_mix {3.00, 6.41, 11.11 solved}; Ṁ ratios f_κ {1.07@2 … 0.29@32, crossing ≈7}, f_A
{0.988@2 … 0.855@32}; Θ_cum exponents q ≈ 0.25–0.32 (f_κ, f_A).

### 6.2 The zeroth-order model (layer invariance, structure frozen)

L_cool → L1 + f·(L2+L3) ⟹ instantaneous θ-ratio = **1 + s·(f−1)**. Solving θ₀·[1+s(f−1)] = 0.95:

> **entry(bench3) = 3.55 · entry(bench2) = 4.95 · entry(bench1) = 7.30 → spread 2.06×**

The cancellation driving the low spread is physical and *specific to this construction*: diffuse
clouds have lower θ₀ (need more help) but a **higher interface share** (get more help per unit f).
No single knob has this property.

### 6.3 The predictions (a miss is recorded as a miss — SC-0 pattern)

| # | prediction | bar | decided by |
|---|---|---|---|
| **PA1** | trigger-metric (θ_max, **solved rows** per `§12`) band entries at 0.95: bench3 ∈ [3.5, 8], bench2 ∈ [5, 12], bench1 ∈ [7, 25] — zeroth-order × back-reaction inflation ∈ [1, ~3]. Hard floor of the construction: **every entry < the same bench's best single-knob entry** (6.53 / 13.8 / 30.8), and **bench1 fires in-grid** (f_κ never did) | all three sub-bars | A1 prod arms |
| **PA2** | **spread ∈ [1.7, 2.8]×, central 2.1×** — strictly better than f_A's 2.71× (solved) | spread < 2.71× | A1 |
| **PA3** | **full-run Ṁ ratio > 1 at every f ≤ 8, all three benches** — the area signature; any crossing pushed above 8 (f_κ alone crossed at ≈7) | sign | A1 diag `bubble_dMdt` vs bench5r baseline |
| **PA4** | Θ_cum dose exponent q(f_area) **> max(q_κ, q_A) per bench** (> 0.28/0.27/0.32); central ≈ 0.5 | superadditivity | A1 diag |
| **PA5** | across doses, ΔL_cool and ΔṀ are **positively correlated** — unique to the area construction (f_A anti-correlates, f_mix is flat in Ṁ) | correlation sign | A1 diag |
| **PA6** | the CONDENSE onset for the band configs sits at **lower f than f_κ-alone's** (more cooling per unit dose) — a *risk* prediction, stated so it can't be spun as a surprise | fire map | A1-band |

### 6.4 Metrics ruling (locked by `§11`/`§12`)

Primary: **θ_max on solved rows** (the trigger criterion, stale-immune per `§12`). Secondary:
θ_max all-rows (what the code would literally do) and **Θ_cum with its stale share printed
beside it** (the L21b energy-budget comparison only — never the knob verdict). All three from one
builder (`make_bench8_analysis.py`, to be written after the reduce shape is known — the bench7
lesson).

## 7. Gates and the TERMINAL clause

| gate | bar | fail ⇒ |
|---|---|---|
| **GA0** | A0 checks A0.1–A0.5 | STOP before HPC (A0.1/A0.2/A0.4-severe) or shrink grid |
| **GA1** | emit gates incl. per-mode knob rules (combined: fk==fA + warning-fires-once; margins: exactly one knob, no warning) + deterministic tag ordering | fix builder, don't submit |
| **GA2** | each arm's cloud == its bench5 `__none` sibling (test-pinned) | not a controlled comparison — stop |
| **GA2b** | A1-margin arms at the doses bench5r/6r/7 already ran reproduce those fresh anchors (θ_max within numerical noise) | code state drifted between campaigns — cross-campaign comparisons in §6.1 are void; re-derive anchors from bench8 alone |
| **GA3** | truncation accounting: every no-`outcome` arm listed; entries carry truncated-bracket counts; 1:30:00 is the cap — a truncated bracket ⇒ **bound, not value** (`§1a` method) | — |
| **GA4** | entries **measured in-grid**; extrapolation only with "ESTIMATE" label | no headline from an extrapolated leg |
| **GA5** | report all three metrics (§6.4) + stale share + Ṁ table together | no ranking published |
| **GA6** | the f = 1 identity: spot-check one bench8-emitted f=1 config loads byte-identically to `__none` (both knobs gated) | wiring differs from assumption — stop |

**Pre-registered TERMINAL:** if **PA3 fails** (Ṁ sign flips at f ≤ 4 on any clean bench) **and**
**PA1's bench1 in-grid fire fails**, then no combination of the existing knobs represents a
wrinkled interface in this 1-D machinery — the area program closes *within current TRINITY*, the
parent single-constant TERMINAL stop applies, and the recorded next step is the truncation-scale
physics (§3.3), not another knob. Partial failures (PA2 misses but PA3 holds, etc.) are results,
not stops — the construction would then be area-faithful but not density-uniform, which is itself
the answer to `PLAN` item 9.

## 8. Decision tree after A1

- **PA1–PA3 pass** → f_area is the first knob that is both calibration-viable *and*
  mechanism-faithful. Then, and only then, Option B (a first-class `cooling_boost_farea` param,
  SC-1-style wiring, maintainer nod required) + revisit the truncation-scale derivation (§3.3) so
  the shipped value has a physics story, not just a fit.
- **PA3 passes, PA2 fails** → area-faithful but density-dependent: publish as the measured
  refutation of "one geometric constant", feeding the same conclusion as SC-0.
- **TERMINAL fires** → §7. The Θ→1 saturated-flux limit (Lancaster Eq 12 ≤ Eq 15) becomes the
  only remaining 1-D-portable idea on the table.

## 9. Cost and risk

514 arms ≈ 1.75× the kappa-3way campaign, one array, same 1:30:00 partition cap per arm — the cap
is per-arm, so a bigger array costs queue time, not truncation risk. Expected stiffness corner:
**bench1 × f ≥ 12 diag** (f_κ enters the structure ODE; the combined knob adds the f_A
source on top — A0.4 exists to catch a lowered viability ceiling before submission). Warm-start
hazard: the fsolve seed scales f^{2/7} while the layer-invariant root scales ≈ f — the seed will
undershoot at high dose; if A0 shows stranded solves, note the seed correction (multiply the Eq-33
seed by f^{5/7} when both knobs are active) as the *one* candidate `trinity/` edit, gated behind
A0 evidence. Truncated arms: handled per GA3; no "more walltime" option exists.

## 9a. What comes DOWN — the size budget (measured 2026-08-02)

A 514-arm campaign only works if the cluster keeps the bulk and git gets the distillate. Measured
from the campaigns already on disk, not estimated:

| stage | per arm | × 514 | travels? |
|---|---|---|---|
| raw `dictionary.jsonl` (gpfs) | 3–4.5 MB | **≈ 1.8–2.3 GB** | **never** — stays on gpfs, reduced in place |
| reduced trajectory (bench7, 10 cols) | 18 KB mean / 30 KB max | ≈ 9–15 MB | yes, but as ONE bundled file |
| summary + derived scalars | ~160 B | **≈ 90 KB** | yes — **this is the analysis surface** |
| hashes (K3 determinism) | ~90 B | ≈ 45 KB | yes |

So the fear that `down` pulls "many GB" does not apply to this workflow: the raw arms were never
downloadable, `reduce` has always run on the cluster, and the reduced total is ~10 MB. The real
problems at 514 arms were **file count** (514 per-arm CSVs is not a reviewable commit, and no
analysis wants 514 opens) and **where the arithmetic ran** — Θ_cum and the solved/stale split were
being recomputed laptop-side on every visit. Both are fixed for bench8:

- **`harvest_bench5.py --derived`** computes the distilled per-arm scalars ON the cluster and writes
  them into the summary: `n_rows, n_stale, stale_time_frac, theta_cum, theta_cum_raw,
  theta_cum_solved, theta_cum_stale, t_window_end, leak_frac, theta_max_solved, theta_max_is_stale`.
  `theta_max_solved` is the §12 stale-corrected trigger metric — computed once, at reduce time,
  never re-derived. **The headline reads (PA1–PA6, the fire map, the band entries) are all
  answerable from `bench8_summary.csv` alone — ~90 KB, one file, one `git diff`.**
- **`--traj-bundle`** writes every arm's θ(t) into ONE `bench8_traj.csv` keyed by a leading
  `run_name` column — same rows, same float precision, ~10 MB, one file. Read it back with
  `data/read_bundle.py::load` → `{run_name: [row dicts]}`, which yields exactly the dicts the
  per-arm readers already consume, so `theta_cum_prefire`, `decompose` and the track plots work
  unchanged. Per-arm CSVs are still written on gpfs (the K3 hashes are taken over them) but for
  bench8 `down` does **not** fetch them.
- Flags are **opt-in**: bench5–bench7 reduce exactly as before, so the frozen record's column sets
  and file layouts do not move.
- Side benefit: `pdv-trigger/MANIFEST.md` carries one provenance row per committed data file, so the
  bundle keeps bench8 to ~3 manifest rows instead of ~514.

**Equivalence gate (met).** The on-cluster `derived` cannot import the canonical laptop-side
implementations — they live in `data/` modules that import matplotlib at module level, and reduce
must stay dependency-light — so the arithmetic is duplicated and pinned by
`test/test_bench_derived.py`: against `make_bench5_analysis.theta_cum_prefire` and
`make_bench_stale_segments.decompose` on synthetic rows (~ULP), **and replayed over all 173
committed bench7 trajectories** — `n_rows`/`n_stale` exact on every arm, every float within
5.0e-05, which is the half-ulp of the `%.4f` the record is stored at. The bundle is pinned to be
string-identical to the per-arm files it replaces.

## 10. Reproduce (in dependency order)

```bash
# A0 (in-container, free) — write + run the screen, commit CSV + figure
python docs/dev/transition/pdv-trigger/data/make_farea_screen.py

# A1 params + tests (after GA0)
python docs/dev/transition/pdv-trigger/runs/make_farea_params.py     # -> runs/params/bench8/ (514)
pytest test/test_bench8_params.py

# HPC (maintainer; needs ssh helix)
./docs/dev/transition/pdv-trigger/runs/sync_bench.sh bench8 up
./docs/dev/transition/pdv-trigger/runs/sync_bench.sh bench8 submit   # auto-sized --array=1-514
./docs/dev/transition/pdv-trigger/runs/sync_bench.sh bench8 reduce   # ⚠️ ONE-SHOT; --derived + bundle
./docs/dev/transition/pdv-trigger/runs/sync_bench.sh bench8 down     # 3 files, ~10 MB (§9a)
#   -> runs/data/bench8_summary.csv   fire map + the distilled scalars  <- the analysis surface
#      runs/data/bench8_traj.csv      all 514 trajectories in one file  <- evidence/backup
#      runs/data/bench8_hashes.csv    K3 determinism hashes
python docs/dev/transition/pdv-trigger/data/read_bundle.py \
    docs/dev/transition/pdv-trigger/runs/data/bench8_traj.csv        # sanity: arms + row count

# analysis + the source of truth
python docs/dev/transition/pdv-trigger/data/make_bench8_analysis.py  # (written post-reduce)
python docs/dev/transition/pdv-trigger/data/make_freshness_audit.py
python docs/dev/transition/kappa-3way/make_report.py
```

## 11. Open questions for the maintainer (none block A0)

1. **Approve the A0 → A1 sequence and the 514-arm layout** (§5) — in particular whether the
   288-arm A1-margin tier is worth the queue time (it can be halved to prod-only, 144, at the
   cost of same-campaign Ṁ margins; the coarse bench5r/6r/7 anchors would then carry PA3/PA4's
   single-knob side alone).
2. **The cross-knob warning**: for bench8 it stays and is asserted-on (provenance). If f_area ever
   ships as Option B, the warning text needs a carve-out — maintainer wording preferred.
3. **The stale-row convention** (`§12` resolved it for analysis; blessing it as the standing rule
   retires open question Q3).

---

**Citations.** Lancaster, Ostriker, Kim & Kim 2021a, ApJ 914, 89 (fractal theory; Eq 11–13) ·
Lancaster et al. 2021b, ApJ 914, 90 (d ≈ 0.4–0.7 measured; Θ = 0.9–0.99) · Lancaster et al. 2024,
ApJ 970, 18 (α_p–area coupling) · Fielding, Ostriker, Bryan & Jermyn 2020, ApJL 894, L24
(D = 5/2) · El-Badry, Ostriker, Kim, Quataert & Weisz 2019, MNRAS 490, 1961 (Eq 21, 37–38, 47) ·
Tan, Oh & Gronke 2021, MNRAS 502, 3179 (front convergence) · Gronke, Oh, Ji & Norman 2022, MNRAS
511, 859 (A ∝ m^{5/6}). Repo imprints: `pdv-trigger/LANCASTER_REFERENCE.md` §7b/§7c,
`ELBADRY_REFERENCE.md`, `FINDINGS.md §23`.
