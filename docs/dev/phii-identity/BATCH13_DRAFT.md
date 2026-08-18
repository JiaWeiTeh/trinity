# Batch 13 — the first `data-new/` grid — Status: ⬜ **DRAFT, pre-registered, unrun**

**Why this grid and not a bigger one.** Every Batch 11 diagnostic is `B3M` alone; the replay
family is `Lw ∈ {1,3,10}`; the only paired stock-vs-C3c full trajectory is `B3M`. And Batch 12
taught the lesson that decides the axes: **seam C tracks bubble size, not the degree of HII
dominance** (`M_cav ∝ R2^{3/2}·sqrt(Qi f_abs)`; the low-wind rung reached only 7.733 pc and the
ratio fell to 0.1296, firing its own registered falsifier). So the grid must vary *what sets
R2* independently of *what sets HII dominance*, which a one-dimensional wind ladder cannot do.

**Axes.**

| axis | rungs | what it isolates |
|---|---|---|
| `Lw` scale | 0.01, 0.1, 1, 3, 10 | confined-vs-driving branch; HII dominance |
| `nCore` | 5 rungs, ~1 dex apart | R2 reached at matched `t` — the seam-C controller |
| `Qi` (via `mCluster`) | 5 rungs | `R_St`, hence `R_ch ∝ α_p² ṗ²/Q₀` |
| `densPL_alpha` | 0, 2 | straddles Geen 2022's `eqn:overflowcondition` ω = 5/4 threshold |
| arm | stock (`fca7d88e`), C3c (current) | old-vs-new |

5 × 5 × 5 × 2 × 2 = **1000 tasks**, which is one array at `%200` throttle and roughly
150 core-hours at B3M's measured 496 s. Headroom for failures and one re-run is the reason
not to spend the whole budget on the first arm.

⚠️ **Pilot first, and this is not optional.** Submit **16 tasks** spanning the corners
(`Lw` extremes × `nCore` extremes × both ω, C3c arm only), run `make_manifest.py` and
`alphap_screen.py` on them, and confirm the manifest's `ok` fraction and `reached_momentum`
before releasing the remaining 984. A reducer bug found at 16 tasks costs minutes; found at
1000 it costs the arm. (`PLAN.md` B11.0 S1 is the precedent: the layer-volume bug was found
*after* the numbers were committed.)

---

## Pre-registered gates — written before any task is submitted

Each names its falsifier. **A run that does not reach the phase a gate needs is VOID, never a
confirming null** — `make_manifest.py` flags these automatically.

**G13.1 — the old identity, grid-wide out-of-sample.** Stock arm: `|P_HII/Pb − 1| ≤ 1e-12` on
≥99% of rows in every phase, every config. Batch 0 established this on 6 configs and B12 gave
it one out-of-sample rung; this is the first test across four decades of `nCore` and two
density profiles simultaneously.
*Falsifier:* any stock row off unity outside the documented stale-`Pb` 1a→1b handoff rows.

**G13.2 — seam C is bubble-size-scoped, not dominance-scoped.** Report `M_cav/M_shell` at
matched `t` against both `R2^{3/2}` and `frac_HII_dominated`.
*Prediction:* `M_cav/M_shell` collapses onto `R2^{3/2}·sqrt(Qi f_abs)` across the whole grid,
and its partial correlation with `frac_HII_dominated` at fixed `R2` is consistent with zero.
*Falsifier of B12's reading:* a config with large `R2` and small `M_cav/M_shell`, or the
converse. That would put the seam back on the dominance axis and reopen the regime-scoping.

**G13.3 — is the α_p handover value physical or numerical?** From `alphap_screen.py`, the
handover `(R2/R1)²` across all configs.
*Prediction:* it clusters tightly (spread < 2×), because it is set by
`ENERGY_FLOOR = 1e3` (`run_transition_phase.py:97`), not by cloud or cluster properties.
*Falsifier:* spread > 2× across the grid, correlated with a physical axis — which would mean
the handover value is a real feature of the trajectory and the collapse to α_p = 1 is
config-dependent, not a numerical artefact of the floor.
*Depends on:* the B3M screen result; if that already shows a discontinuity, this gate becomes
the measurement of how it varies rather than a check on whether it exists.

**G13.4 — Geen's overflow sign, tested.** `densPL_alpha` 0 and 2 straddle
ω = 5/4. Geen 2022 `eqn:overflowcondition`: for ω < 5/4 overflow becomes *less* likely with
increasing radius (trapping wins); for ω > 5/4, *more* likely.
*Prediction:* `f_esc_ion` onset, as a function of `R2` at matched `t`, moves in **opposite**
directions between the two ω rungs.
*Falsifier:* the same sign on both. That would mean TRINITY's overflow criterion does not
inherit the analytic result it is a numerical version of — and §2.3's reference-mass
difference ($M(<r_w)$ vs $M(<r_i)$, a factor up to ~2 at ω = 2) is the first place to look.

**G13.5 — the thin-shell validity boundary, mapped.** Report where in (`Lw`, `nCore`) the
trajectory crosses `dR_ion/R2 = 1/3`, and in which phase.
*Prediction:* crossed inside the **transition** phase on most of the grid (B11.D measured
0.658–1.308 in B3M momentum; B12 measured it *worse* at low wind, 1.171–1.438).
*No pass/fail* — the deliverable is a stated validity limit with a boundary on it, which is
what the methods section needs.

**G13.6 — cost and completeness.** Total core-hours, manifest `ok` fraction, and
`reached_momentum` fraction.
*Bar:* ≥95% `ok`. Below that, fix the harness before trusting any reduced CSV from the arm.

---

## What this batch deliberately does NOT do

- **No `trinity/` changes.** Both arms are existing code refs; the arm is selected by git
  worktree, not by a patch (C-2).
- **It does not test the balance volume (K5/K6).** That is a code change and needs its own
  pre-registered ablation. This grid is the *baseline* that ablation will be measured against,
  which is the reason to run it first.
- **It does not settle D5.** D5 is a physics-intent call and cannot be settled by measurement
  — `PLAN.md` §7 says so and this batch does not pretend otherwise.
