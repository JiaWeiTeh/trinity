# Negative interior velocity (WARPFIELD "Problem 2"): the causal chain

> ⚠️ **Point-in-time analysis (2026-06-14) — verify before trusting it.** Scratch
> copy on the diag plotting branch. The canonical home is
> `analysis/stalling-energy-phase.md` (on `bugfix/beta-delta-solver-pt2`,
> Phase 5/6); this lives here to avoid a merge conflict with that file. Re-check
> every claim, line reference, and number against current source before relying
> on it.

What drives the transient inner-bubble inflow, separating **cause** from
**consequence**. Figures: `scratch/phase2/negvel_*.png` + `hunt_*.png`
(`analyze_negvel.py`, `plot_hunt.py`, `reconstruct_vprofile.py`).

## Definitions (verified in source)

- `beta  = -(t/Pb)·dPb/dt`   (`get_bubbleParams.py:167`)  →  **β < 0 ⇔ Pb rising**
- `delta = +(t/T)·dT/dt`     (`get_bubbleParams.py:63`)   →  **δ > 0 ⇔ T rising**
- bubble structure ODE (`bubble_luminosity.py:1150`):
  `dv/dr = (β+δ)/t + (v − α·r/t)·dTdr/T − 2v/r`. The **forcing/source term is `(β+δ)/t`**.

## The clean identity (why β+δ, not β, is the trigger)

With the two definitions above (and `Pb ∝ n·T`):

```
(β+δ)/t = −Ṗb/Pb + Ṫ/T = d/dt·ln(T/Pb) = −d·ln(n)/dt
```

So **β+δ < 0 ⇔ the inner gas is being compressed (n rising)**. That is the
physical trigger: a negative source term in `dv/dr` pulls the inner velocity
below zero (inflow), held against the `v(R1)≈0` boundary. β (Pb-rate) and δ
(T-rate) carry opposite sign conventions, so δ’s positive swing *partly cancels*
β’s negative one — inflow happens only when the Pb-rise out-paces the T-rise.

## Causal chain (cause → consequence)

| step | what | direction |
|---|---|---|
| **① CAUSE** | time-variable SB99 feedback → `Lmech_total` non-monotonic → a surge (WR-wind ramp ~3 Myr, then SN onset) | `Lmech` **spikes up** |
| **② MECHANISM** | surge injects energy faster than the bubble expands → `Eb` climbs → `Pb` rises (re-pressurise); gas heats, `T` rises | `Eb↑`, `Pb↑`, `T↑` |
| **③ TRIGGER** | β dives (Pb-rate) and **outweighs** the δ rise (T-rate) → **β+δ < ~−0.4** = net compression | β→−2.4, δ→+1.8, net **−1.1** |
| **④ CONSEQUENCE** | the `(β+δ)/t` source goes negative → inner velocity reverses → **inflow band** | `v_min` → −0.6 pc/Myr |

(steep 1e6, α=−2; deepest WR-surge segment, t≈3.23 Myr. See `negvel_causal.png`.)

## It is the RATE *relative to the current pressure*, not an absolute spike

`β = −(t/Pb)·Ṗb` is large-negative when `Ṗb` is large **relative to** `Pb`. A
weakly-pressurised bubble gets a big β swing from a modest surge. This is why the
Phase-6 hunt found the **deepest** inflow in the **weakest-feedback** config
(h1, cluster 1e4: β+δ→−1.11, band 74% of thickness), while stronger feedback
stays shallow (h2/h3/h6: −0.35…−0.42) or never goes negative (h5: +0.14, zero
inflow). **"Stronger surge → worse inflow" is falsified** (`hunt_massdep.png`).

## Causation vs consequence

- **Upstream cause:** the `Lmech` spike (relative to current `Pb`).
- **Downstream consequences (both terminal):** the inflow band **and** the `dMdt`
  jump. Neither feeds back — `v` is absent from all three cooling integrals
  (`bubble_luminosity.py:612/659/677`), so the inflow is **energy-budget-immune**,
  and it is **transient** (recovers when the surge passes). The `dMdt` jump
  **leads** the inflow (both driven by the same `Lmech` cause; `dMdt` responds
  first) — so the `dMdt` "kink" is the *feedback surge*, not the inflow band
  (`hunt_dmdt_leads.png`).

## Thoroughness of the parameter detection

Phase-6.0 hunt: 6 configs (cluster mass 1e4→3e5, sfe 0.01→0.30, flat+steep+dense;
909 segments, 100% converged). Established: β+δ is the trigger (not β alone); the
`Lmech`/`dMdt` jump leads the inflow; weakest feedback = deepest; energy-immune.
Gate G6 marginally open only on h6's `dMdt` channel (ambiguous, surge-confounded).

## Figures

| figure | shows |
|---|---|
| `negvel_causal.png` | the 4-panel cause→consequence ladder + β/δ decomposition (this analysis) |
| `negvel_timeline.png` | steep vs mock — chain aligned in time; mock never crosses −0.4 |
| `negvel_dmdt_lmech.png` | `dMdt` tracks `Lmech` (Pearson r≈0.95) with a transient hysteresis lag |
| `negvel_feedback.png` | `Eb↑`, `Pb` bump, cooling-ratio reset (re-pressurisation + stall) |
| `negvel_profile.png` | the real reconstructed `v(r)` — the inner inflow band |
| `negvel_trigger.png` / `hunt_*.png` | the β+δ trigger and the 6-config demographics |

## Open gaps

- Density `n(t)` is not logged; the `−d·ln(n)/dt = compression` reading is
  *inferred* from β, δ. A direct `n(t)` would need the structure re-solve
  (cf. `reconstruct_vprofile.py`).
- A quantitative **β vs (Ṗb/Pb)** / relative-`Lmech`-jump scatter across the hunt
  configs would *prove* (not infer) the relative-spike driver.
