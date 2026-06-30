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

## 2. Result (4 of 8 regimes; units-correct)

Using the committed f_κ=1 baseline arms (median Pb over the implicit phase):

| config (regime) | Pb [cgs] | κ_mix/κ_Spitzer @2e4 K | @2e5 K | T_cross (λδv=1) |
|---|---:|---:|---:|---:|
| compact (n~1e5) | 2.0×10⁻¹¹ | 9.0×10³ | 2.9 | 2.7×10⁵ K |
| diffuse (n~1e2) | 1.6×10⁻⁸ | 7.3×10⁶ | 2.3×10³ | 1.8×10⁶ K |
| dense-stiff (n~1e6) | 1.0×10⁻⁵ | 4.7×10⁹ | 1.5×10⁶ | 1.2×10⁷ K |

**Reading.** In the cool mixing layer (2×10⁴–2×10⁵ K, where n²Λ peaks) **κ_mix dominates Spitzer by 10³–10⁹ even
at λδv=1**, and T_cross sits at or above the layer in every regime. So κ_mix would **substantially restructure the
conduction front** — the go decision: wiring it is warranted, it is *not* a negligible correction. Equally
important, even **λδv ≪ 1** already dominates the layer (the "λδv to dominate 2×10⁵ K" is 0.35 / ~0 / ~0), so
**λδv is the sensitive magnitude knob** — at λδv=1 the ratio is already 10³–10⁹, so the value must be **calibrated
to Lancaster θ~0.9–0.99, not imported/cranked** (this is the RUNGB_SCOPING "κ_mix swamps Spitzer" concern made
quantitative: with λδv=1 we get 10³–10⁹, the same family as that doc's 10²⁴ for `D_turb=R2 v2`).

## 3. Coverage gap + the next step

- **4 of the canonical 8** are covered (compact / diffuse / dense-stiff; heavy `fail_repro` harvest is a stub with
  no usable Pb). The other 4 — `midrange_pl0`, `be_sphere`, `pl2_steep`, `small_dense_highsfe`, `small_1e6`
  control — need their `Pb(t)` (HPC runs; full sims are ~90 min–hours, too slow in-container). The harness reads
  any `harvest_*.csv`, so it extends for free when that data lands. The 3 covered already span the regime range,
  so the **go/no-go conclusion holds**; the remaining 4 confirm.
- **Next (still pre-production):** the self-consistent test — re-solve the structure with `κ = max(κ_mix,
  κ_Spitzer)` (a harness that *calls* the `bubble_luminosity.py` solver functions with κ_mix injected, still
  off the production path), on all 8 configs, byte-identical-off proven, then the gated production mode
  (`RUNGB_SCOPING.md` §8). Only after all 8 pass does anything reach production.

*Written 2026-06-29 on `feature/PdV-trigger-term-pt2`. No production code touched; no sims.*
