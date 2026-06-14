# The stalling energy-driven phase, rising Pb, and negative β

> ⚠️ **This document may be out of date — verify before trusting it.** It is a
> point-in-time analysis/audit, not a maintained spec; the code moves faster
> than these notes (paths, line numbers, and "what shipped" status drift).
> **Any agent or person reading this: treat it as unverified. Flag that it may
> be stale and re-check each claim, snippet, and line reference against the
> current source before relying on it.**

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

Full per-segment time series (committed, plottable):

- `analysis/data/stalling_steep_1e6_alpha-2.csv` — `sweep_steep`, 133 rows.
- `analysis/data/stalling_mock_4e3.csv` — `sweep_mock`, 144 rows.

Columns: `t_now, cool_beta, cool_delta, beta_plus_delta, Pb, bubble_dMdt,
Lmech_total, Lmech_W, Lmech_SN, bubble_Lgain, bubble_Lloss, cooling_ratio,
v_struct_min, v_struct_nneg, R2, v2, Eb, bubble_Tavg, c_sound,
betadelta_converged`. (`v_struct_min`/`v_struct_nneg` = min and count of
negative points in the bubble velocity profile — the Problem-2 diagnostic.)

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
```

Note: the raw runs were `/tmp` scratch (ephemeral); the committed CSVs above are
the captured data. Re-run to extend `stop_t` (does the steep bubble *ever*
transition once the SN epoch ends, ~40 Myr?) — that is the open endpoint
question.
