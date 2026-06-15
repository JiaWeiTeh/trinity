# The stalling energy-driven phase, rising Pb, and negative β

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
> than you found it. **Keep both banner paragraphs at the top of every plan and
> analysis doc.**

Investigation (2026-06-13) of two things the self-consistent hybr runs surfaced
that legacy (β clamped to [0,1]) could never show:

1. **Steep / low-mass clouds *stall*** — the cooling ratio `(Lgain−Lloss)/Lgain`
   plateaus well above the 0.05 transition threshold and never crosses it, so
   the bubble stays energy-driven for many Myr (see
   `analysis/BETADELTA_PHASE2_ARMS.md`, Phase-3 section).
2. **β goes *negative*** (down to −2.4) in places — i.e. **Pb is *rising***
   (β = −(t/Pb)(dPb/dt), so β<0 ⇔ dPb/dt>0).

These turn out to be the **same story**, and it is a feedback-history effect.

## The finding: feedback luminosity surges re-pressurise the bubble

Negative β is **not** noise and **not** a numerical artifact. It tracks
**jumps in the mechanical luminosity** `Lmech_total` (= `bubble_Lgain`): when
feedback power surges, the bubble gains energy faster than it expands, so
`Eb` rises, `Pb` rises (β<0), and the conductive evaporation `dMdt` rises with
it. Two surges drive it, both visible in `sweep_steep` (1e6, α=−2):

| t [Myr] | β | Lmech_W | Lmech_SN | Lmech_total | dMdt | Pb | ratio |
|---|---|---|---|---|---|---|---|
| 2.83 | +2.15 | 1.95e8 | ~0 | 1.95e8 | 387 | 792 | 0.38 |
| 3.03 | +1.65 | 2.06e8 | ~0 | 2.06e8 | 419 | 677 | 0.44 |
| **3.08** | **−0.29** | **2.52e8** ↑ | ~0 | 2.52e8 | 596 | 659 | — |
| **3.18** | **−2.44** | **3.34e8** ↑ | ~0 | 3.34e8 | 1359 | 679 | — |
| 3.43 | +1.05 | 3.43e8 | ~0 | 3.40e8 | 1380 | 733 | 0.55 |
| **3.63** | **−0.59** | 3.39e8 | **1.24e8** ↑ | **4.63e8** | 1382 | 687 | 0.64 |
| **3.68** | **−0.95** | 3.33e8 | **1.70e8** ↑ | 5.02e8 | 1707 | 692 | — |
| 4.00 | +3.43 | 2.49e8 | 1.94e8 | 4.42e8 | 1379 | 627 | 0.52 |

Two distinct re-energisation episodes:
- **~3.0–3.4 Myr — a wind-luminosity surge** (`Lmech_W` climbs 2.0e8 → 3.5e8;
  the WR phase). β dives to −2.44, dMdt jumps 4× (419 → 1685).
- **~3.5–3.8 Myr — the supernova onset** (`Lmech_SN` jumps from ~1e7 to
  >1e8). β dives again, dMdt spikes to ~2100, `Lmech_total` nearly doubles.

The low-mass mock (4e3) shows the **identical two-episode structure** (wind
surge ~3.1 Myr → β=−0.23; SN onset ~3.6 Myr → β=−0.90), scaled down by its tiny
cluster. So this is generic, not config-specific.

## Answers to the specific questions

- **Why is Pb rising?** A feedback power surge (WR winds, then SNe) injects
  energy faster than the bubble expands → `dEb/dt > 0` outpaces the volume
  growth → `Pb ∝ Eb/V` rises → β<0. It is the bubble **re-pressurising**.
- **Is dMdt also rising?** **Yes, strongly** — dMdt rises in lockstep with the
  luminosity (419 → ~2100 across the surge). More injected power ⇒ more
  conductive mass evaporation across the contact discontinuity. (`Lmech_SN`
  itself oscillates ±, an artifact of the discrete-SN SB99 table differencing;
  `Lmech_total` and the trend are what matter.)
- **What else comes along?** `Eb` jumps (7.6e7 → 2.0e8 over the surge),
  the cooling **ratio jumps back up** (0.44 → 0.67 — re-energised, *further*
  from transition), `R2`/`v2` keep climbing. The cooling `Lloss` lags the
  Lmech jump, which is exactly why the ratio rises.
- **Is this unusual?** **Physically, no — it is expected** for time-variable
  feedback (the classic constant-luminosity Weaver bubble has β>0 always
  because Pb only ever declines; realistic SB99 winds→WR→SN make Lmech
  non-monotonic, so Pb can rise). **Numerically, it is new to TRINITY**: the
  legacy solver clamps β∈[0,1], so it **cannot represent re-pressurisation at
  all** — it would pin β=0 through every SN/WR surge, silently mis-integrating
  the energy injection. hybr capturing β<0 is a genuine physical gain.

## The velocity structure goes unphysical at the deepest β<0 (WARPFIELD "Problem 2")

A converged `dMdt>0` does **not** guarantee a physical velocity profile. The
structure solve enforces only the inner-edge BC (`v(R1)≈0`) and `min_T` /
monotonic-T (`_get_velocity_residuals`) — it **never checks the sign of the
interior velocity**. At the deepest, fastest β dive (the WR wind surge,
t≈3.18–3.33 Myr) the bubble develops substantial **negative (inflow)**
velocities:

| t | β | δ | β+δ | negative-v points (of 100) |
|---|---|---|---|---|
| 3.178 | −2.44 | +1.82 | −0.63 | 25 |
| 3.228 | −2.10 | +0.99 | −1.11 | **50** |
| 3.278 | −1.26 | +0.29 | −0.97 | 44 |
| 3.328 | −0.39 | −0.10 | −0.49 | 10 |

The driver is **(β+δ)/t**, the source term of `dv/dr` in the bubble ODE
(`bubble_luminosity.py` `_get_bubble_ODE`): when β+δ goes strongly negative the
velocity falls steeply and crosses zero over a chunk of the profile — *inflow*,
which the Weaver self-similar (outflow) structure does not admit. So these few
segments are **converged but partially unphysical** in their velocity field.
(Note δ goes *positive* here — T rising — concurrent with β<0; both flag the
surge.) This is exactly WARPFIELD's "Problem 2," and TRINITY's gate (like
legacy's) has **no guard against it**.

Distinctions that matter:
- **Transient and localized** — 4 segments of 133, only during the fastest
  surge; the bubble recovers (β, v back to normal) by t≈3.4 Myr.
- Most other "negative-v" flags in the CSV are a single **innermost grid point**
  near the `v(R1)≈0` BC dipping slightly negative (`v_struct_nneg=1`) — a
  numerical boundary artifact, not a real inflow region. The real cases have
  `v_struct_nneg ≈ 10–50` (plot `v_struct_nneg` vs `beta_plus_delta`).
- **Open question (Phase 5):** are these deepest β<0 excursions physically real
  (genuine transient inflow during a violent re-pressurisation) or a
  structure-model breakdown? A future option (outside the solver-repair scope)
  is a velocity-sign guard analogous to `min_T` — reject/penalise interior-v<0
  structures so the gate covers Problem 2, not just `dMdt>0` (Problem 1).

### Diagnosis from the captured profiles (2026-06-14)

Reading the full `bubble_v_arr` / `bubble_v_arr_r_arr` at the affected segments
(steep run) pins it down:

- **The trigger is β+δ, not β.** `dv/dr`'s source term is `(β+δ)/t`, so the
  inflow appears only when **β+δ ≲ −0.5**, regardless of β alone. Proof from
  the two runs: `sweep_mock` reaches β=−1.04 but δ compensates so its
  **(β+δ)_min = +0.25 → zero real inflow segments**; `sweep_steep` hits
  **(β+δ)_min = −1.11 → 4 inflow segments** (their β+δ ∈ [−1.11, −0.49]). So
  negative β is *necessary-ish but not sufficient* — you need β+δ negative.
- **It is an inner-bubble band.** v runs 0 at R1 (inner, the BC) up to ≈v₂ at
  R2 (outer). During the surge the dip is in the **inner ~2–70 % of the bubble
  thickness** (radial fraction from R1), recovering to positive toward R2:

  | t | β | β+δ | v_min [pc/Myr] | r(v_min)/R₂ | negative band (frac. of thickness) |
  |---|---|---|---|---|---|
  | 3.178 | −2.44 | −0.63 | −0.11 | 0.21 | 0.02–0.42 |
  | 3.228 | −2.10 | −1.11 | **−0.62** | 0.40 | **0.02–0.73** |
  | 3.278 | −1.26 | −0.97 | −0.45 | 0.37 | 0.02–0.68 |
  | 3.328 | −0.39 | −0.49 | −0.01 | 0.08 | 0.02–0.16 |

- **Magnitude is small but real:** v_min ≈ −0.1…−0.6 pc/Myr against a shell
  v₂ ≈ 10 — a genuine reversal (inflow) in the inner band, ~1–6 % of v₂. The
  band grows to the inner ~¾ of the bubble at peak (β+δ=−1.11) then collapses.
- **Reading:** the quasi-steady Weaver structure is being pushed outside its
  validity by a *violent transient* re-pressurisation (β+δ swinging strongly
  negative over ~0.15 Myr); the inner gas momentarily flows inward, which the
  self-similar outflow ansatz cannot represent. Whether to treat that as real
  transient inflow or to guard against it (velocity-sign penalty) is the
  Phase-5 call.

### Is the inflow physical? A physical reading (2026-06-14) — INTERPRETATION, NOT established

> **Epistemic status: mostly conjecture.** The "measured" bullets are read
> straight from `analysis/data/hunt_h1_steep_base.csv`; everything labelled
> *interpretation* or *guess* is a physical story fitted to those numbers — **not**
> verified against a time-dependent hydro solve, an independent structure code, or
> the literature. Treat every causal claim below as a hypothesis to check, not a
> result. Do not cite any of it as settled.

**Measured at the deepest dip (h1, t=3.228 Myr):**
- interior T ≈ 4.9e6 K → `c_sound` ≈ 338 pc/Myr (~330 km/s).
- `v_struct_min` = −0.62 pc/Myr ⇒ **inflow Mach |v_min|/c_sound ≈ 0.002**;
  |v_min|/v2 ≈ 6 %; kinetic-vs-thermal energy of that gas ~ (v/c_s)² ~ **1e-6**.
- across the band Eb rises smoothly 7.6e7→1.2e8, T rises 4.3→4.9e6 K, R2 keeps
  expanding 26→29.6 pc, v2 keeps rising 9.2→10.5 pc/Myr; Pb dips then bumps up.
- β<0 and δ>0 occur *together* (Pb rising **and** T rising — both reinforce the
  negative `(β+δ)/t` source).

**Interpretation (plausible, unverified) — what the dip *might* be:** β<0 ⇔ Pb
rising, δ>0 ⇔ T rising; both at once ⇒ the bubble is being **re-pressurised and
re-heated faster than expansion relieves it**. The timing (~3 Myr, before the
SN-onset marker) is *consistent with* the **Wolf–Rayet wind luminosity surge**
("WR bump" in SB99-type `Lmech`) of a coeval cluster — but we have **not**
confirmed the surge is WR-specific vs another `Lmech` feature; check the SB99
input. The plot-1 velocity shape (v=0 at R1, a subsonic negative trough over the
inner ~40–74 % of the thickness, steep rise to +v2 at R2) *reads as* a
**stagnation radius** — re-energised inner gas back-drifting while the massive
outer shell coasts out on its inertia. That is a cartoon, not a derived flow.

**Why it is *probably* an artefact more than a real inflow (conjecture):** the
Weaver/WARPFIELD interior structure is derived on the **self-similar expanding
attractor** (β≈0.5–0.6, monotonic outflow, T∝(1−r/R2)^{2/5}); the `(β+δ)/t` term
is that steady structure's time-dependence book-keeping. Feeding it β+δ≈−1
(transiently *anti*-self-similar) extrapolates the ansatz outside its domain, so
the returned profile is mathematically consistent but **not guaranteed to be the
true transient flow**. A time-dependent hydro solve would *probably* show a weak
compressive / sound-wave re-adjustment instead — **we have not run one; this is a
guess.**

**Why it "settles itself" (reasoning, plausible):**
1. the driver is transient — β<0 only while `Lmech` is steeply *rising* (~0.15 Myr);
2. the interior is Mach ~1e-3 with sound-crossing ~R2/c_s ≈ 0.08 Myr, so any
   imbalance is ironed out almost instantly — no reservoir to sustain a flow;
3. no positive feedback — `v` is absent from the radiative cooling integrals
   (`bubble_luminosity.py:612/659/677`) so it cannot change `Lloss`, and it
   carries ~1e-6 of the thermal energy so it cannot materially move `Eb`/`Pb`
   either. ⇒ a **forced, damped, bounded excursion, not an instability** — which
   matches the observed ~4-segment spike-and-recover. *(Caveat: "no positive
   feedback" is an inference from where v does/doesn't appear in the code, not a
   proof of dynamical stability.)*

**Why steep dips but the mock does not (interpretation):** the inflow needs
β+δ≲−0.5, i.e. the pressure surge (β<0) and temperature surge (δ) must
*reinforce*. In steep-1e6 they do (β+δ→−1.1); in the 4e3 mock δ compensates
(β+δ floor +0.25) → no inflow. So it is *not* a generic feedback feature — it
seems to need the dense / strongly-cooling / weakly-driven corner where P and T
surge together. The δ sign-behaviour driving this wants its own check.

**Net (still a guess):** most likely a *physically-motivated flag* of a real
re-pressurisation event, rendered as a *quantitatively unreliable* inflow profile
by the quasi-steady ansatz, and **energetically negligible** (~1e-6 of thermal)
so it does not touch the macro budget. Whether a *real* bubble has a small genuine
transient inflow here is **open** and would need a time-dependent interior solve
to settle — do **not** record it as "resolved".

## Phase 6.0 contamination hunt: is the inflow ever non-cosmetic? (2026-06-14)

The Problem-2 open question above — *real transient or structure breakdown, and
does it corrupt anything?* — got a dedicated **gate**
(`docs/dev/BETADELTA_HYBR_PLAN.md` Phase 6.0). Six hybr runs were instrumented
(harness `scratch/phase6/hunt.py`, which wraps `solve_betadelta_pure` and dumps
one row per accepted energy-implicit segment, reading the full `bubble_v_arr`
for the velocity diagnostics) to hunt a regime where the inflow stops being
cosmetic — non-convergence, a kink in `Lloss`/`dMdt`/`Eb` across the band, or
the band dominating the bubble.

**Configs** (all steep r⁻² except the flat control; `betadelta_solver=hybr`):

| config | mCloud | sfe | cluster | profile | nCore | stop_t | probe |
|--------|--------|-----|---------|---------|-------|--------|-------|
| h1 base | 1e6 | 0.01 | 1e4 | α=−2 | 1e5 | 4 | reproduce baseline |
| h2 sfe10 | 1e6 | 0.10 | 1e5 | α=−2 | 1e5 | 6 | 10× stronger SN |
| h3 sfe30 | 1e6 | 0.30 | 3e5 | α=−2 | 1e5 | 6 | strongest SN |
| h4 dense | 1e6 | 0.10 | 1e5 | α=−2 | 1e6 | 6 | sustain through SN |
| h5 long | 1e6 | 0.03 | 3e4 | α=−2 | 1e5 | 8 | full WR→SN→decline |
| h6 flat | 1e6 | 0.30 | 3e5 | α=0 | 1e3 | 6 | flat control |

**Result — 909 segments, 100% converged. Gate G6 marginally OPEN on one bounded
channel; cosmetic in 5/6.**

| config | rows | β+δ min | real inflow | t band [Myr] | v_min | max frac | verdict |
|--------|------|---------|-------------|--------------|-------|----------|---------|
| h1 base | 134 | −1.11 | 4 | 3.18–3.33 | −0.62 | 0.74 | cosmetic |
| h2 sfe10 | 174 | −0.42 | 3 | 3.74–3.84 | −0.16 | 0.31 | cosmetic |
| h3 sfe30 | 172 | −0.35 | 3 | 3.75–3.85 | −0.17 | 0.30 | cosmetic |
| h4 dense | 25 | −0.27 | 1 | 0.003 (handoff) | −1.33 | 0.72 | cosmetic* |
| h5 long | 215 | +0.14 | 0 | — | — | — | cosmetic |
| h6 flat | 189 | −0.37 | 3 | 3.76–3.86 | −0.22 | 0.34 | **flags dMdt** |

(*h4's deep band is the explicit→implicit handoff transient — excluded; it
transitions to momentum at t=0.037 Myr. **Grid note:** the hunt harness reads
the full ~6e4-point `bubble_v_arr`, so its raw `v_struct_nneg` count is *not*
comparable to the old "of 100" tables above — use `v_neg_frac_thick` (the
thickness fraction, which *does* match: h1 peak 0.74 ≈ old 0.73) or
`v_struct_nneg / v_struct_npts`.)

**Three findings:**

1. **"Stronger surge → worse inflow" is FALSIFIED.** The deepest dip/inflow is
   in the *weakest*-feedback baseline (h1, sfe 0.01: β+δ→−1.11, frac 0.74);
   stronger feedback keeps β+δ shallow (h2/h3/h6: −0.35…−0.42, frac ~0.30) or
   positive (h5: +0.14, zero inflow). A highly-pressurised bubble sees the WR/SN
   surge as a *small relative* perturbation, so Pb rises less → β+δ stays up.
   (Plot: `min(beta_plus_delta)` and `max(v_neg_frac_thick)` vs cluster mass.)

2. **The inflow is energy-budget-immune.** `v` is **absent from all three
   cooling integrals** (`bubble_luminosity.py:612` bubble, `:659` conduction,
   `:677` intermediate — they use `n²Λ(T)` / `dudt(n,T,φ)` only), so a deep
   inflow band cannot corrupt `Lloss` or `Eb`. The only v-coupled output is
   `dMdt` (the structure solve matches the velocity BC).

3. **The dMdt "kink" is the feedback surge, not the inflow — it LEADS the
   inflow.** Walking h1's WR surge (driver = `Lmech_W` 2.06e8→3.54e8; SN still
   noise at ~1e4–1e6):

   | t | β+δ | Lmech_W | dMdt | %dMdt step | v_min |
   |---|-----|---------|------|-----------|-------|
   | 3.078 | +1.47 | 2.52e8 | 596 | +42% | 0 (no inflow) |
   | 3.128 | +0.41 | 3.00e8 | 963 | **+62%** | 0 (no inflow) |
   | 3.178 | −0.63 | 3.34e8 | 1359 | +41% | −0.11 (inflow starts) |
   | 3.228 | −1.11 | 3.51e8 | 1616 | +19% | −0.62 (deepest) |
   | 3.278 | −0.97 | 3.54e8 | 1684 | +4% | −0.45 |

   The biggest dMdt jumps (+42%, +62%) are **before** β+δ goes negative — driven
   by the Lmech surge; by the time inflow appears the jump is already shrinking,
   and `Lloss` rises smoothly straight through. So the inflow adds **no**
   roughness. Deconfounding each config's band step against its own surge ramp
   (lead/trail windows): h1 dMdt ×0.7, h2/h3 ×0.9 (clean); h4 excluded (handoff).
   **Only h6** keeps a dMdt step (10.9%, ×1.9) at its inflow onset while Lmech is
   flat — but that looks like a *lagged* response to the SN surge (dMdt
   under-shot the rise then caught up), not a clean inflow signature.

**Gate-G6 verdict: marginally OPEN, on one bounded, ambiguous dMdt channel
(h6).** The inflow is real, sometimes deep (74 % of thickness), always
converges, and is provably energy-budget-immune; the only thing it can touch is
`dMdt`, and even that is mostly the surge. The honest screen cannot certify the
dMdt channel as exactly zero-impact, so the principled next step is a **narrow
Phase 6.1 counterfactual**: clip v≥0 / reject-and-hold on the inflow segments,
measure ΔdMdt and the macro deltas (R2, v2, terminal momentum, transition time).
Expected low/no macro impact given the energy immunity and bounded dMdt response
— but that is a measurement, not an assumption.

### Phase 6.1 — counterfactual: the inflow IS immaterial (measured, 2026-06-14)

The narrow 6.1 was run (harness `--hold-inflow`, classifier
`scratch/phase6/compare_hold.py`): for the four configs with real inflow, every
inflow segment was **rejected and held** (flagged `no_physical_root` so the
runner holds the last physical structure — arm C, via the production hold path),
and the held trajectory diffed against the accepted baseline.

| config | segs held | dMdt kicked (local) | final ΔR2 | final Δv2 | final ΔEb | max ΔEb (transient) |
|--------|-----------|---------------------|-----------|-----------|-----------|----------------------|
| h1 (deepest, frac 0.74) | 4 | 42.8 % | +0.043 % | +0.038 % | +0.022 % | 0.63 % |
| h2 (sfe10) | 3 | 11.7 % | −0.0001 % | +0.0001 % | −0.0002 % | 0.026 % |
| h3 (sfe30) | 3 | 9.6 % | +0.0000 % | −0.0000 % | +0.0000 % | 0.0002 % |
| h6 (flat, the dMdt flag) | 3 | 12.5 % | +0.0000 % | +0.0000 % | +0.0000 % | 0.0000 % |

**Deleting the inflow entirely — a 9.6–42.8 % local kick to dMdt — moves R2, v2,
Eb (hence terminal momentum) by ≤0.04 % at the end** across all four; h1 (the
smallest, most sensitive bubble) is the only nonzero final effect, the large
bubbles ~0 (h6 differs by ~1e-9). So the inflow is not only energy-immune in
principle, it is **empirically immaterial** to every reported quantity —
including the dMdt channel that kept G6 "marginally open."

**On "why so small" (checked, not hand-waved):** it is *not* a units or
propagation artefact. Deltas are relative (dimensionless) so units cancel; and
the held outputs genuinely respond (h1's Eb deviates 0.63 % *during* the band,
nonzero, then re-equilibrates), so the held structure reaches the integrated
state via the segment-start snapshot `solve_ivp` uses — not zero by
construction. The smallness is physical: the band is brief (~0.15 Myr of a 4 Myr
run), the bubble recovers, and dMdt (conductive evaporation) is a small term next
to the cumulative Lmech injection and the swept shell mass. *(A sustained-freeze
positive control to bound the channel gain was attempted but the ephemeral
container reclaimed the long runs before they finished; the nonzero held-run
responses already establish propagation, so the conclusion does not rest on it.)*

**Net: Problem 2 is closed** — the WR-re-pressurisation inflow is real but
cosmetic for the energy/momentum budget. `v_neg_frac_thick` ships as a snapshot
diagnostic (registry + `COOLING_PHASE_KEYS`) so any future config that *does*
drive a deep inflow is flagged automatically; no treatment is applied.

## Why this matters for the transition criterion (Phase 5)

The "stall" is **feedback-sustained**, not just halo-fed. The steep bubble
drifts toward cooling balance (ratio falling), but every feedback episode
(WR surge, SN onset) **resets it upward** — at 3 Myr the ratio jumps 0.44→0.67,
pushing the bubble *back* to strongly energy-driven. So:

- The implicit→momentum transition is **not a monotonic cooling-balance
  crossing**; it is modulated by the feedback *history*. A bubble "about to
  transition" can be re-energised by SNe and stay energy-driven for Myr more.
- A fixed `(Lgain−Lloss)/Lgain < 0.05` trigger may **never fire** for clouds
  whose feedback keeps Lmech high (steep halos, and any cloud still inside its
  SN epoch). This is the core Phase-5 question: the transition criterion likely
  needs to be feedback/dynamics-aware (e.g. force-ratio or blowout), not a pure
  energy-ratio threshold. See `docs/dev/BETADELTA_HYBR_PLAN.md` Phase 5.

## Data for plotting

Full per-segment time series (committed, plottable). One row = one accepted
energy-implicit (β,δ) segment.

*Original two (2026-06-13), 100-point velocity grid:*
- `analysis/data/stalling_steep_1e6_alpha-2.csv` — `sweep_steep`, 133 rows.
- `analysis/data/stalling_mock_4e3.csv` — `sweep_mock`, 144 rows.

*Phase 6.0 hunt (2026-06-14), full ~6e4-point velocity grid, six configs:*
- `analysis/data/hunt_h1_steep_base.csv` … `hunt_h6_flat_sfe30.csv` (see the
  config table above; 909 rows total).

**Column dictionary** (units: t [Myr]; v, c_sound [pc/Myr]; R2 [pc]; T [K];
luminosities [M⊙ pc² Myr⁻³]; dMdt [M⊙ Myr⁻¹]; Pb, Eb in code/au units):

| column | meaning |
|--------|---------|
| `t_now` | segment time |
| `cool_beta`, `cool_delta` | β = −(t/Pb)dPb/dt, δ = (t/T)dT/dt |
| `beta_plus_delta` | β+δ — the `dv/dr` source `(β+δ)/t`; **inflow driver** |
| `Pb` | bubble pressure |
| `bubble_dMdt` | conductive mass flux shell→bubble (the v-coupled output) |
| `Lmech_total`/`_W`/`_SN` | mechanical luminosity: total / winds(+WR) / SNe |
| `bubble_Lgain` | = `Lmech_total` (energy gain) |
| `bubble_Lloss` | radiative cooling loss (uses n²Λ(T); **no v**) |
| `cooling_ratio` | (Lgain−Lloss)/Lgain — the transition diagnostic |
| `v_struct_min` | min of `bubble_v_arr` (most negative interior velocity) |
| `v_struct_nneg` | count of v<0 grid points (**hunt: of ~6e4; old: of 100**) |
| `v_struct_npts` | grid size (hunt CSVs only) — use `nneg/npts` for fraction |
| `v_neg_frac_thick` | radial-thickness fraction with v<0 (hunt CSVs only) |
| `R2`, `v2` | shell radius, velocity |
| `Eb` | bubble energy |
| `bubble_Tavg` | volume-avg bubble temperature |
| `c_sound` | bubble sound speed |
| `no_physical_root` | gate fired (dMdt≤0 / solve failed) — hunt CSVs only |
| `betadelta_converged` | (β,δ) root converged |

`v_struct_min`/`v_neg_frac_thick` are the **Problem-2** diagnostics; for
cross-grid comparison prefer `v_neg_frac_thick` (or `v_struct_nneg/v_struct_npts`)
over the raw count.

Suggested plots / things to investigate:
- `β` and `Pb` vs `t` with `Lmech_W`, `Lmech_SN` overlaid — does every β<0
  episode line up with an Lmech rise? (Yes here; check across more configs.)
- `cooling_ratio` vs `t` with feedback episodes marked — quantify how much each
  SN/WR surge resets the ratio, and whether it ever recovers downward to 0.05.
- `dMdt` vs `Lmech_total` — is the conductive evaporation a clean function of
  injected power? Slope/lag?
- `Eb`, `Pb` vs `t` — re-pressurisation amplitude vs surge strength.
- Sweep SB99 age of WR/SN onset vs cluster mass: does the β<0 timing track the
  SN clock (and is it stochastic for tiny clusters like the 4e3 mock)?

## Reproduce

```bash
# steep (stalls + re-pressurises): 1e6 M_sun, alpha_rho=-2, nCore=1e5, rCore=1
python run.py <param: mCloud=1e6 sfe=0.01 densPL_alpha=-2 nCore=1e5 rCore=1 \
    betadelta_solver=hybr stop_t=4.0>
# then dump implicit-phase rows (betadelta_total_residual non-nan) from
# outputs/<model>/dictionary.jsonl to CSV (see the columns above).

# Phase 6.0 hunt: per-segment velocity diagnostics straight to CSV, plus the
# Gate-G6 classifier (run single-thread to avoid BLAS oversubscription):
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    python scratch/phase6/hunt.py scratch/phase6/h1_steep_base.param \
    --out analysis/data/hunt_h1_steep_base.csv
python scratch/phase6/analyze_hunt.py analysis/data/hunt_h*.csv   # G6 verdict
```

Phase-6-specific plots worth making from the hunt CSVs:
- `v_struct_min` (and `v_neg_frac_thick`) vs `beta_plus_delta` — the inflow law:
  the band opens once β+δ ≲ −0.5 and deepens roughly with |β+δ|.
- `min(beta_plus_delta)` and `max(v_neg_frac_thick)` vs cluster mass across the
  six configs — the "stronger feedback suppresses the dip" trend.
- `dMdt`, `Lmech_W`, `Lmech_SN` vs `t` zoomed on each inflow band — confirm the
  dMdt step *leads* β+δ<0 (surge-driven), and inspect h6's lagged onset step.

Note: the original two CSVs were captured from `/tmp` scratch; the hunt harness
+ configs live in `scratch/phase6/` (gitignored scratch — present locally).
Re-run to extend `stop_t` (does the steep bubble *ever* transition once the SN
epoch ends, ~40 Myr?) — that is the open endpoint.
question.
