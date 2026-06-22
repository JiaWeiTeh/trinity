# H1 audit: is the non-firing transition a bug in the `Lcool` (`bubble_LTotal`) computation, or real physics?

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
> a committed artifact (a CSV/table under `docs/dev/data/`, or a force-added
> harness/figure in the relevant `docs/dev/<workstream>/` folder) — never left in
> `/tmp` or an untracked `outputs/`. A future visit must be able to reproduce or
> compare against the numbers **without re-running**; record the exact config +
> command that produced each artifact.

**Date:** 2026-06-22. **Branch:** `fix/transition-trigger-problem-pt4`.
**Scope:** read-only audit of production `Lcool` = `bubble_LTotal` against the
committed cleanroom CSVs. No production code changed; no sims re-run.

---

## TL;DR verdict

1. **The `Lcool` computation is CORRECT. No bug or regression was found.** Every
   integrand, cooling-table call, unit factor, and the `np.abs()` trapezoids are
   **character-for-character identical** across the entire recent refactor chain
   (`7f08e58`, `24c6914`, `4996060`, `60fb362`), and identical in *physics* to the
   original `src/` implementation. The two real changes in history are both
   defensible and far too small to matter (see §3).
2. **`Lcool` does NOT "keep surging up". It surges UP early, then COLLAPSES.** The
   maintainer's "surging up" intuition is the **first ~0.1 Myr** (Lloss roughly
   doubles); after the bubble grows it **falls by up to ~10×** as `n²V` dilutes.
   The prior FINDINGS' "collapse" story is **correct** for the regime that
   determines the stall.
3. **The non-firing is a PHYSICS result, not a computational artifact.** Under
   hybr the cooling-balance ratio `(Lgain−Lloss)/Lgain` plateaus at **0.28–0.49
   (min)** and recovers to **0.56–0.83**, never reaching 0.05. The divergence from
   legacy is entirely the **β-clamp**, not the cooling integral (§4).

---

## 1. What `Lcool` (`bubble_LTotal`) is — precise map (current source)

Entry: `trinity/bubble_structure/bubble_luminosity.py`,
`get_bubbleproperties_pure` (:186) → `_bubble_luminosity` (:575). The structure
ODE (Weaver Eq 42–43) is integrated backward (`_solve_bubble_structure` :417,
LSODA dense output) on the 60k-point grid (`_create_radius_grid` :481), giving
`T_array`, `dTdr_array`, `v_array`; density is the **hot-bubble** relation
`n = Pb / ((mu_convert/mu_ion)·k_B·T)` (:623). `Pb` from `bubble_E2P`, `R1` from
`solve_R1`. The profile is split at `_CIEswitch = 10**5.5 K` (:656) and
`_coolingswitch = 1e4 K` (:653) into three zones, each trapezoid-integrated:

- **L_bubble** (CIE, T>10^5.5), :694–698:
  `Lambda_bubble = 10**cooling_CIE(log10 T) · cvt.Lambda_cgs2au`;
  `integrand = chi_e · n² · Λ · 4πr²`; `L_bubble = |trapz(integrand, r_bubble)|`.
  `chi_e` (≈1.2 = n_e/n_H) makes it `n_e·n_H·Λ`. **Positive-definite cooling.**
- **L_conduction** (non-CIE, 1e4<T<10^5.5), :716–745: structure sampled from the
  dense solution `_sol.sol` over the conduction band (`_CONDUCTION_NPTS=2000`);
  `dudt_cond = (heat_cond − cool_cond) · cvt.dudt_cgs2au` from the 3-D
  `(n,T,φ)` non-CIE tables; `integrand = dudt_cond · 4πr²`;
  `L_conduction = |trapz(...)|`. **This is a NET (heat−cool) integrand** — the
  tables already carry their own electron factor (no `chi_e` here, correctly).
- **L_intermediate** (1e4 → T[index_cooling_switch], :749–785): a 1000-point
  linspace, split by regime — non-CIE branch uses the net `(heat−cool)` table,
  CIE branch uses `chi_e · n² · Λ`. `L_intermediate += |trapz(...)|` per regime.
- **L_total = L_bubble + L_conduction + L_intermediate** (:790) → returned as
  `bubble_LTotal` (:835).

**`Lloss` in the transition is purely this radiative integral** — no PdV, no
velocity term. `run_energy_implicit_phase.py:1080` sets `Lloss =
bubble_props.bubble_LTotal` (+ `bubble_Leak`, which is 0 at Cf=1), and the trigger
at `:1095` is `Lgain>0 and (Lgain−Lloss)/Lgain < 0.05` with `Lgain = Lmech_total`
(:1075). The PdV work term `4πR2²·v2·Pb` appears only in the **solver's**
`Edot_from_balance` (`get_betadelta.py:434`), NOT in the trigger. So this audit's
`Lloss` is exactly the cooling integral above.

### On the `np.abs()` trapezoids (:698, :745, :785)
For **L_bubble** and the **CIE** intermediate branch the integrand is
`chi_e·n²·Λ·4πr² ≥ 0`; `np.abs` only flips the sign that comes from `r` being a
*descending* slice (backward integration), so it is harmless and correct. For the
**conduction / non-CIE intermediate** branches the integrand is the *net*
`(heat−cool)`, which could in principle be negative (net heating). `np.abs` would
then make a net-heating cell *add* to the loss. **However** the bubble interior
runs at T0 ≈ 3–8×10⁶ K (cooling-dominated, see data), so over the integrated band
the net term is cooling-signed and `|trapz|` returns the true loss. This is the
**original** behaviour (present verbatim in `afb239c`), not a regression, and the
sign cannot explain a 0.05↔0.3 gap. **Not a masked bug here.**

## 2. How the hybr β/δ solve feeds `Lcool`

`get_betadelta.py`: `BubbleParamsView` (:107) overrides only `cool_beta`,
`cool_delta` (and optionally the dMdt seed) and passes everything else to params;
`get_residual_pure` (:353) calls `get_bubbleproperties_pure(params_view)`. β and δ
enter the **structure ODE RHS** `_get_bubble_ODE` (:386) through `cool_beta`,
`cool_delta` in `dTdrr`/`dvdr` (:406–412): they set the interior `T(r)`/`v(r)`
profile, hence `n(r)=Pb/(…k_B T)`, hence every cooling integrand. **Pb itself does
not depend on β/δ** (it is `bubble_E2P(Eb,R2,R1)`), so β/δ act on `Lcool` purely
through the *shape* of `T(r)` (and the dMdt root that anchors the boundary).

Legacy clamps β∈[0,1], δ∈[−1,0] (`BETA_MIN/MAX`, :41–44). hybr
(`_solve_betadelta_hybr` :874) is the **unbounded** scipy `root(method='hybr')` on
the pole-free `g` residual, gated only on `dMdt>0` + a valid structure
(`_NoPhysicalRoot`). So hybr can reach β≈+3.5…+4.2 (seen in data). A large-β
profile is steeper/hotter, which **suppresses** the CIE `n²Λ` emission (Λ falls
above ~2×10⁵ K and the hot interior dilutes), so `Lcool` collapses — this is a
*physical* response of the (more correct) unbounded root, not a unit/normalization
slip. The integrals are evaluated on the same grid with the same factors
regardless of β/δ magnitude; nothing integrates over a region it shouldn't.

## 3. Git-history regression check — what actually changed, and whether it matters

`git log --follow` chain for the cooling integral (newest→oldest):
`4996060` (regroup "bit-identical") · `24c6914` (F1 residual perf) · `7f08e58`
(drop `_legacy`) · … · `5f4f229` (conduction from dense output) · `9222a96`
(#660, composition) · `4109c2e` (#636 src→trinity rename).

Diffing the integrand / `L_*` / `np.abs(trapz)` / `Lambda_cgs2au` /
`dudt_cgs2au` / `chi_e` lines across `7f08e58~1, 7f08e58, 4996060~1, 4996060,
HEAD` — **all five are byte-identical**. The recent refactors moved blocks; they
did **not** touch the cooling math. `git blame` attributing lines to `49960608`
is the *move*, not a content change (verified by `git show` of each revision).

Two genuine content changes exist, both upstream of this branch and both benign:

- **`9222a96` (#660) added `chi_e` to the CIE integrands** (`integrand_bubble`,
  CIE `integrand_int`). Original `afb239c` had bare `n²·Λ`. This is a **physics
  fix** (CIE cooling is `n_e·n_H·Λ`, and `n_array` is `n_H`), applied
  *consistently* with `get_dudt`'s CIE branch (`net_coolingcurve.py:164,187`).
  Effect: scales CIE `Lloss` by `chi_e = 1 + Z_He·x_He = 1.2`. A **+20%** bump —
  it would *help* the trigger fire, and is ~15× too small to bridge 0.3→0.05.
  No double-count: the conduction/non-CIE branches use the net table, not `chi_e`.
- **`5f4f229` was explicitly NOT bit-identical** (its own message says so): it
  replaced the ~100-point conduction re-solve with `_sol.sol` dense-output
  sampling, moving `bubble_L2Conduction` by ≤0.9% and `bubble_LTotal` by ≤0.18%
  toward the *converged* value (the old re-solve was ~0.9% **low**). The integrand
  was unchanged. ≤0.2% on LTotal — negligible vs the 0.05↔0.3 gap.

**No sign flip, dropped/added factor (beyond the deliberate `chi_e`), changed
integration bound, or au↔cgs slip was found.** The `Lambda_cgs2au`,
`dudt_cgs2au`, `ndens_cgs2au`, `phi_cgs2au` factors match `get_dudt`'s conventions
and the unit-conversion module.

## 4. Empirical direction — from committed data (no re-run)

Harness: `analyze_lcool_direction.py` + `trajectory_probe.py` (this folder);
summary `H1_lcool_direction_summary.csv`. Source: `../cleanroom/data/c0_*_{h0,legacy}.csv`.
`ratio = (bubble_Lgain − bubble_Lloss)/bubble_Lgain`, finite `Lgain>0` rows only.

| config | solver | ratio min | @t (Myr) | ratio final | crosses 0.05? |
|---|---|---|---|---|---|
| simple_cluster | **hybr** | 0.324 | 0.098 | 0.764 | **NO** |
| large_diffuse_lowsfe | **hybr** | 0.465 | 4.86 | 0.561 | **NO** |
| small_dense_highsfe | **hybr** | 0.283 | 0.015 | 0.695 | **NO** |
| midrange_pl0 | **hybr** | 0.364 | 0.432 | 0.833 | **NO** |
| pl2_steep | **hybr** | 0.489 | 0.037 | 0.831 | **NO** |
| be_sphere | **hybr** | 0.471 | 0.556 | 0.829 | **NO** |
| simple_cluster | legacy | −0.007 | 0.178 | — | **YES @0.178** |
| large_diffuse_lowsfe | legacy | 0.514 | 2.50 | — | NO (truncated) |
| small_dense_highsfe | legacy | 0.024 | 0.024 | — | **YES @0.024** |
| midrange_pl0 | legacy | −0.009 | 0.82 | — | **YES @0.822** |
| pl2_steep | legacy | −0.001 | 0.128 | — | **YES @0.128** |
| be_sphere | legacy | −0.020 | 1.04 | — | **YES @1.04** |

**hybr: 0/6 cross 0.05. legacy: 5/6 cross** (large_diffuse never crosses; min
0.514). This reproduces the prior claim exactly.

### `Lloss` (=`Lcool`) direction — surge THEN collapse
The coarse first-third→last-third trend is mixed because runs truncate at
different `t`, so the trajectory shape is what matters (`trajectory_probe.py`):

- **simple_cluster hybr:** Lloss rises 1.90e8 → **peak 4.10e8 @ t=0.098**, then
  **collapses to 5.68e7 @ t=0.84** (−7.2×), recovering only to 2.33e8 by the t=6
  cap. Ratio: 0.69 → min 0.32 → back to 0.92 → 0.76.
- **pl2_steep hybr:** Lloss 5.5e8 → **peak 1.04e9 @ t=0.037** → **2.38e8 @ t=2.99**
  (−4.3×). Ratio min 0.49 → 0.93.
- **small_dense hybr:** Lloss 4.3e7 → **peak 7.27e7 @ t=0.015** → 8.4e6 @ t=0.7
  (−8.6×). Ratio min 0.28 → 0.92.

So **`Lcool` does surge up early (≈2×) then collapse (≈4–9×)**. The surge is the
emission-measure rise (volume growth beats dilution while R2 is tiny); the
collapse is `n²V ∝ (Pb/T0)²R2³` falling as R2 grows and the interior stays
3–8×10⁶ K (too hot to radiate efficiently — T0 in the data never enters the
10⁵–10⁶ K Λ-peak band). **`Lgain` (=Lmech) also rises** (feedback surges at
~3 Myr: simple_cluster 6.1e8→1.3e9), which pushes the ratio further from 0.05. So
the non-firing is driven by **both**: Lloss collapses (dominant) and Lgain surges.

### Legacy vs hybr — the divergence is the β-clamp, not the integral
Both solvers track **identically** to t≈0.08 (ratio≈0.34, β pinned near the upper
bound ~0.85–0.92, Lloss rising in lockstep). At t≈0.12 they split
(`simple_cluster` implicit-phase rows):

- **legacy** β clamped to [0,1] → forced to 0.5, δ→−0.6; this **keeps Lloss
  climbing** to 6.16e8 and the ratio crosses to −0.007 @ t=0.178 → transition.
- **hybr** β free → **jumps to +3.5 (t=0.22), +4.2 (t=0.46)**; the hotter/steeper
  profile makes **Lloss collapse** to 5.3e7 and the ratio recovers to 0.92.

Identical `Pb`, identical `T0` trajectory pre-split; the only difference is the
β value the solver is *allowed* to reach. Legacy's "crossing" is a **constrained
edge-root artifact of the clamp**, consistent with the C0 certification that hybr
finds the true (unbounded) root. The cooling integral is the same code in both.

## 5. Conclusion

- **(a) Is `Lcool` correct?** YES. The integrand, cooling-table calls, unit
  factors, zone splits, and `np.abs(trapz)` are byte-identical through every
  recent refactor and physics-identical to the original. The only deliberate
  change (`chi_e`, +20%, consistent with `get_dudt`) is correct and would *help*
  the trigger. The conduction `_sol.sol` change (≤0.18% on LTotal) is a documented
  convergence improvement. **No surging-up bug; no regression.**
- **(b) Does `Lcool` surge or collapse?** BOTH, in sequence: a ~2× early surge
  (t≲0.1 Myr, the maintainer's observation) **then a 4–9× collapse** as `n²V`
  dilutes (the FINDINGS' observation). The two are reconciled — different epochs.
- **(c) Physics or artifact?** PHYSICS. Under the (more-correct) unbounded hybr
  root the interior is too hot to radiate; no cooling-balance event exists to
  trigger on (ratio floors 0.28–0.49, recovers to 0.56–0.83). Legacy's crossing
  is the β-clamp pinning Lloss high, not extra/real cooling.

**Recommendation (unchanged from prior FINDINGS, now independently re-verified):**
do not "fix" by tuning the 0.05 threshold or by hunting a `Lcool` bug — there is
no bug and no cooling event in the hybr regime. The principled levers are
structure-level mixing-layer cooling or leakage (Cf<1), or a geometric/Eb-peak
transition. The maintainer's H1 ("a bug we accidentally introduced into
bubble_luminosity such that Lcool keeps surging up") is **not supported**: Lcool
is computed correctly and does not keep surging — it collapses after an early bump.

## 6. Artifacts (committed, this folder)
- `analyze_lcool_direction.py` — per-config Lloss/Lgain direction, ratio min/final,
  0.05-crossing, hybr vs legacy. Run: `python docs/dev/transition/pt4/analyze_lcool_direction.py`.
- `trajectory_probe.py` — sampled Lloss(t)/Lgain(t)/ratio(t) trajectories showing
  the surge-then-collapse and the legacy/hybr split.
- `H1_lcool_direction_summary.csv` — machine-readable summary table.
- Source data (not produced here): `../cleanroom/data/c0_*_{h0,legacy}.csv`.
