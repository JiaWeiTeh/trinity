# κ_mix offline prototype — step 1 of the Rung-B implementation (de-risk before production)

> ⚠️ **This document may be out of date — verify before trusting it.** It is a
> point-in-time analysis/audit, not a maintained spec; the code moves faster
> than these notes. **Any agent or person reading this: treat it as unverified.
> Re-check each claim, snippet, and line reference against current source.**
>
> 🔄 **Living plan — recheck and refine on every visit.** Re-verify the claims and
> line references against current source; update drift; rethink the strategy and
> note what changed (date it). Leave it better than you found it. **Keep all banner
> paragraphs at the top.**
>
> 💾 **Persist diagnostics — commit, don't re-run.** Diagnostics worth keeping are
> committed under `docs/dev/transition/pdv-trigger/{data,}` (CSV + figure + builder),
> reproducible without re-running; record the exact command.
>
> 🔗 **Cross-check the sibling docs — keep the workstream self-consistent.** Siblings:
> `INDEX.md`, `PLAN.md`, `FINDINGS.md`, `F_KAPPA_FUNCTIONAL_FORM.md`, `KMIX_DIFFUSIVITY.md`,
> `RUNGB_SCOPING.md`. Reconcile any number/claim that disagrees; never update one in isolation.

---

## 0. What this is, and the guardrail it respects

The physics work (`F_KAPPA_FUNCTIONAL_FORM.md` §13, `KMIX_DIFFUSIVITY.md`) concluded the faithful fix is the
**structural κ_mix term (Rung B)**, not the scalar f_κ. Before *any* production wiring, the maintainer's rule is:
**de-risk offline, test all 8 configs, mind the units.** This doc records **step 1** — an offline scoping harness
that touches **no solver and changes no production code**. Builder: `data/make_kmix_prototype.py` →
`data/kmix_prototype.csv` + `kmix_prototype.png` (reads committed `runs/data/harvest_*.csv`, no sims).

**Scope (honest).** It answers *"does κ_mix matter in TRINITY's regime, where, and for what λδv?"* — the
go/no-go for the bigger step. It does **not** give the self-consistent θ (that needs *re-solving* the structure
with κ_mix, which is the gated in-solver step that follows and is tested on all 8 configs).

## 1. The units (the recurring bug class — handled explicitly)

At the conduction front the gas sits at bubble pressure Pb and a layer temperature T (pressure equilibrium
`n = Pb/k_B T`), so the dominance ratio reduces to (k_B cancels):

```
   κ_mix / κ_Spitzer = (λδv)·Pb / (C_th · T^(7/2))      [dimensionless]
   T_cross = ((λδv)·Pb / C_th)^(2/7)   — κ_mix dominates for T < T_cross
```

Unit conversions made explicit in the builder (dimensional self-check printed):
- **`Pb` is in TRINITY AU units `Msun/Myr²/pc`**, *not* cgs → `Pb_cgs = Pb_au / 1.5454414956718×10¹²`
  (the `Pb_cgs2au` factor in `trinity/_functions/unit_conversions.py`).
- **`λδv` in pc·km/s → cm²/s** via `× 3.086×10²³`.
- `C_th = 6×10⁻⁷`, `k_B = 1.380649×10⁻¹⁶` are cgs (registry); `T` in K. All combined in cgs.

## 2. Result (4 clean density anchors, run in-container 2026-06-30; units-correct)

The 4 `cal_*` f_κ=1 baselines were **run in the container** (each ~12 min, all `STOPPING_TIME` at t=0.3 Myr;
harvested to `runs/data/harvest_cal_*.csv`), spanning the canonical nCore 1e2–1e6. Median Pb over the implicit
phase, converted to cgs:

| config (regime) | Pb [cgs] | κ_mix/κ_Spitzer @2e4 K | @2e5 K | T_cross (λδv=1) |
|---|---:|---:|---:|---:|
| diffuse (n~1e2) | 4.4×10⁻⁸ | 2.0×10⁷ | 6.3×10³ | 2.4×10⁶ K |
| mid (n~1e4) | 1.5×10⁻⁷ | 6.9×10⁷ | 2.2×10⁴ | 3.5×10⁶ K |
| compact (n~1e5) | 5.4×10⁻⁷ | 2.5×10⁸ | 7.8×10⁴ | 5.0×10⁶ K |
| dense (n~1e6) | 6.2×10⁻⁸ | 2.8×10⁷ | 8.9×10³ | 2.7×10⁶ K |

**Reading.** In the cool mixing layer (2×10⁴–2×10⁵ K, where n²Λ peaks) **κ_mix dominates Spitzer by 10³–10⁸ even
at λδv=1**, and T_cross (2.4–5.0×10⁶ K) sits **far above** the layer in *every* regime. So κ_mix would
**substantially restructure the conduction front** — the go decision: wiring it is warranted, it is *not* a
negligible correction. The dominance is **fairly uniform across density** (Pb varies little at matched epoch),
so κ_mix matters generically, not just at one end. Equally important, even **λδv ≪ 1** already dominates (the
"λδv to dominate 2×10⁵ K" is ≈0 for all), so **λδv is the sensitive magnitude knob** — at λδv=1 the ratio is
already 10³–10⁸, so the value must be **calibrated to Lancaster θ~0.9–0.99, not imported/cranked** (this is the
RUNGB_SCOPING "κ_mix swamps Spitzer" concern made quantitative; same family as that doc's 10²⁴ for
`D_turb=R2 v2`, but tamed once λδv is pinned).

*(Cross-checks: the earlier f1edge/simple_cluster harvests, and the heavy `fail_repro`, gave consistent
dominance. The heavy 5e9 is **excluded** — it `ENERGY_COLLAPSED` in the energy phase with negative Pb; no
implicit/cooling structure ever forms, so κ_mix is moot for it. That is itself a finding: the pathological heavy
cloud has no mixing layer to enhance.)*

> **Diagnosis — the heavy-cloud negative Pb (investigated 2026-06-30; NOT a results bug).** `fail_repro` shows
> `Pb = −1.6×10¹⁸` at its *terminal* row only. Root cause: `Pb = (γ−1)·Eb/V` (`get_bubbleParams.py:236`), so Pb is
> linear in Eb; the heavy bubble's `Eb` crosses to **negative** at the collapse (energy exhausted), and the extreme
> magnitude is that negative Eb divided by a tiny collapsing shell volume `V→0`. The collapse **is** correctly
> caught (`Eb <= 0` → `ENERGY_COLLAPSED`, code 51, `run_energy_implicit_phase.py:1074`). It appears **only** in the
> heavy run, **only** in the last row (it does not propagate into the integration; the run stops), and the
> **4 healthy cal runs have zero negative Pb/Eb** across 600+ rows. **Source of the bad row (re-traced 06-30 —
> corrects the earlier guess here):** it is **not** the line-1074-vs-865 ordering. The line-865/868 Pb is the
> last *healthy* value; the garbage row comes from the **phase-boundary reconciliation snapshot** (lines
> 1269–1297) that runs *after* the collapse `break` and recomputes `Pb_f = compute_R1_Pb(R2, Eb<0, …)` (line
> 1273) from the now-negative collapse `Eb`, then `save_snapshot()` (line 1297). Full diagnosis + a one-line fix
> (skip reconciliation when `termination_reason == "energy_collapsed"`) + its test plan: **`PB_COLLAPSE_GUARD_FIX.md`**.
> Low priority (correctness/stop-fate already right; downstream analysis excludes collapsed runs) — queued behind
> the guardrail, not yet applied.

## 3. Coverage + the next step

- **The full density span (nCore 1e2–1e6) is covered** by the 4 `cal_*` anchors, run in-container 2026-06-30
  (~12 min each; the runs fit comfortably in <60 min — no HPC needed, contrary to the earlier assumption). The
  named closure-8 labels (`midrange_pl0`, `be_sphere`, `pl2_steep`, `small_dense_highsfe`, `small_1e6`) are
  *upstream analysis* configs whose density range is already spanned here; the heavy `fail_repro` is excluded
  (energy-collapse, no implicit phase). So the **GO conclusion is firm across the density range.**
- **Next (still pre-production):** the self-consistent test — re-solve the structure with `κ = max(κ_mix,
  κ_Spitzer)` (a harness that *calls* the `bubble_luminosity.py` solver functions with κ_mix injected, still
  off the production path), on all 8 configs, byte-identical-off proven, then the gated production mode
  (`RUNGB_SCOPING.md` §8). Only after all 8 pass does anything reach production.

*Written 2026-06-29 on `feature/PdV-trigger-term-pt2`. No production code touched; no sims.*
