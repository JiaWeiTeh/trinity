# HANDOFF — `pdv-trigger` workstream (energy→momentum transition / interface cooling)

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
> several living docs for its workstream (its `PLAN.md`, `FINDINGS.md`, `REPRODUCE.md`, `runs/README.md`,
> `NOTE_PATCHES.md`, and any other notes in the same folder). They drift out of sync *with each other* as fast
> as they drift from the code. Any agent or person editing one MUST, as part of the visit, circle back through the
> siblings and reconcile: if a number, status, claim, or line reference here contradicts a sibling — or a
> sibling has gone stale — fix it (or flag it, dated) so no two docs in the workstream disagree. Never
> update one in isolation.

---

## 0. To continue this conversation in a new room — read these, in order

| # | Path | What it gives you |
|---|------|-------------------|
| 1 | **this file** (`docs/dev/transition/pdv-trigger/HANDOFF.md`) | one-screen orientation + open tasks |
| 2 | `docs/dev/transition/pdv-trigger/FINDINGS.md` | the *settled, verified* results (what we know) |
| 3 | `docs/dev/transition/pdv-trigger/PLAN.md` | the dated ledger of every step + the argument/strategy |
| 4 | `docs/dev/transition/pdv-trigger/REPRODUCE.md` | result → `.param` → command → artifact (paper reproducibility) |
| 5 | `docs/dev/transition/pdv-trigger/make_pdvtrigger_report.py` → `pdvtrigger_report.html` | the rendered storyline (run the script to rebuild the HTML) |
| 6 | root `CLAUDE.md` | the project rules (rule 5 = equivalence-depth gate; dev-only constraint) |

**Branch:** `feature/PdV-trigger-term` (this is the working branch — develop here, push here).
**HEAD at handoff:** `3e68143` (El-Badry verification + overlay).
**Relation to `main`:** 4 ahead, 0 behind. `main` is branch-protected (direct push → 403); it
is updated via PR only. The earlier f_κ↔f_mix docs (`b09a307`, `cd0a51d`) are on this branch and
**not yet on main** — land them via PR if/when wanted.

---

## 1. What this workstream is about (one paragraph)

TRINITY transitions a feedback bubble from **energy-driven** to **momentum-driven** when the hot
interior stops doing useful PdV work — physically, when interface cooling/leakage drains the
mechanical luminosity. The shipped trigger is **`cooling_balance` at a 0.95 loss fraction**
(θ = L_loss/L_mech ≥ 0.95). This workstream asks: *what sets θ, and what knobs could move the
transition?* It catalogs three independent axes and tests them as **dev-only diagnostics** — nothing
here changes production behavior by default.

### The taxonomy (3 orthogonal axes — keep these straight)

- **(A) Outcome-side** — *how the loss is composed.*
  `cooling_boost_mode=multiplier`: L_loss = L_leak + f_mix·L_cool (f_mix scales the cooling term).
  `theta_target`: L_loss = max(L_cool+L_leak, θ·L_mech) — the **Lancaster-θ floor**.
- **(B) Mechanism-side** — *El-Badry conduction/mixing.* `cooling_boost_kappa` = **f_κ** multiplies
  the Spitzer conductivity C_th in κ_eff = f_κ·C_th·T^(5/2). θ then **emerges** from the structure
  (f_κ does NOT multiply L_cool directly). "El-Badry-κ" and "modify the conduction front k_f" are
  the **same knob**, not two.
- **(C) Trigger-side** — *what condition fires.* `transition_trigger=ebpeak` (fire at the Ė-balance
  peak) vs the default loss-fraction trigger.

---

## 2. What is SETTLED (verified, in FINDINGS.md)

1. **ebpeak does not fire at f_κ=1** for any normal config. PdV is the dominant energy sink but is
   **not a substitute** for κ_eff: the PdV-inclusive ratio peaks *below* 1.0 (compact 0.912, diffuse
   0.862) then declines — a cooling↔PdV trade-off keeps it flat. PdV alone can't trigger the
   transition. *(make_ebpeak_trigger_test.py, make_ebpeak_8config_xcheck.py)*
2. **8-config cross-check:** 6 normal configs peak PdV-incl 0.85–0.92 and never fire; only the
   pathological heavy-5e9/control fire. Live `simple_cluster` (0.911) and `midrange_pl0` (0.901)
   match the frozen screen to the digit.
3. **f_κ is grounded in code** at 3 sites in `bubble_luminosity.py` (dMdt seed Eq33 ⇒ dMdt∝f_κ^(2/7);
   ICs Eq44; T-curvature ODE Eq42–43). Seed scaling verified: dMdt(f_κ=2)/dMdt(1)=1.2175 vs
   2^(2/7)=1.219. *(make_fkappa_definition.py)*
4. **f_κ calibration (3 anchors):** θ(f_κ=1) = 0.67 / 0.61 / 0.17 (compact/mid/diffuse); f_κ to fire
   ≈ 4 (measured) / ~5–6 / ~60 (extrap). *(make_kappa_blowout_calibration.py)*
5. **f_κ ↔ f_mix:** NOT interchangeable. f_mix is a pure rescale of L_cool (definitional); f_κ is
   structural. f_mix_equiv = L_cool(f_κ)/L_cool(f_κ=1); measured f_κ=2 → 1.23–1.50. Power law
   f_mix≈f_κ^q with q ~0.30 (developed) → 0.58 (seed), bracketed below by **2/7** (Weaver
   evaporation). Framed as heuristic, not derived. *(report §14, FINDINGS taxonomy table)*
6. **El-Badry+2019 (1902.09547) §5.2 VERIFIED from the PDF** *(newest, 3e68143)*: Eq37
   ψ=A_mix·(λδv)^½·n_H^½ with **A_mix=3.5**; Eq38 θ=ψ/(11/5+ψ); domain n_H=0.1–10 cm⁻³,
   λδv=0.1–10 pc·km/s; θ time-independent. The √n form and saturation are **genuine** (earlier
   "confabulation" skepticism was retracted). Our resolved θ_1D points sit far **below** the
   El-Badry target in the GMC regime (n=1e2–1e6, i.e. 1–5 decades **extrapolated**) — that gap is
   the cooling deficit mixing must supply. *(make_elbadry_overlay.py → elbadry_overlay.png)*
7. **Dense-edge stiffness** (nCore 1e6) is a solver/density problem, NOT f_κ-driven — slow for
   hybr AND legacy at f_κ=1 baseline. *(data/dense_stiffness_diag.csv)*

---

## 3. What is OPEN / next (pending tasks)

1. **Controlled f_κ(n_H) sweep — built, NOT yet run.** `runs/params/sweep_fkappa_nH.param` =
   7 nCore × 13 f_κ × 3 mCloud × 3 sfe = **819 combos** (under the 1000 ceiling). De-conflation
   test: does the f_κ-to-fire collapse onto one n_H curve, or also depend on mCloud/sfe? **HPC only**
   (do not run 819 combos in the container). Pipeline is reduce-then-plot: `sync.sh submit/watch/
   collect/reduce/down` → `data/reduce_fkappa_sweep.py` (stdlib streaming reducer, θ bit-identical
   to harvest()) → `data/make_fkappa_nH_sweep.py` (fit + de-conflation figure). See REPRODUCE Block C.
   **→ User runs this on Helix; the new room analyzes the resulting summary.csv.**
2. **(B) gated f_κ(n_H) mode wiring** — deferred by design until the sweep gives the actual curve.
   Argued for a design spec first, not a premature implementation.
3. **Land f_κ↔f_mix + El-Badry docs on main** — `b09a307`, `cd0a51d`, `3e68143` are feature-only.
   main is protected → needs a PR (not yet opened; user hasn't asked for one).
4. **(Offered, unconfirmed)** Wire `elbadry_overlay.png` into the report §7 as the literature anchor,
   replacing the old schematic `theta_vs_density.png`.

---

## 4. Hard constraints (do not violate)

- **Dev-only.** θ/El-Badry/Lancaster/trigger knobs are **paper diagnostics, not shipped defaults**.
  Production must stay byte-identical when the new modes are off. The shipped default is
  `cooling_balance` @ 0.95. (The *good* production changes already on main: hybr solver, bugfixes,
  default trigger = 0.95 loss.)
- **Push only to `feature/PdV-trigger-term`.** Never push to main directly (protected; use a PR).
- **Commits:** no Claude session links, no "Generated by Claude", no co-author trailers, no model ID
  in any committed artifact.
- **numpy<2** pinned (monotonic-guard FP sensitivity); the whole sci stack is version-capped.
- Equivalence-depth gate (CLAUDE.md rule 5) applies to any solver/residual/hot-loop edit:
  full-run equivalence on stiff edge regimes, separate processes, matched `t`.

---

## 5. Reproduction quick-reference

```
# rebuild every figure from committed CSVs (no sims):
for f in docs/dev/transition/pdv-trigger/data/make_*.py; do python "$f"; done
# rebuild the storyline HTML:
python docs/dev/transition/pdv-trigger/make_pdvtrigger_report.py   # -> pdvtrigger_report.html
# inspect the 819-combo sweep without running it:
python run.py docs/dev/transition/pdv-trigger/runs/params/sweep_fkappa_nH.param --dry-run
```
Every figure's exact `.param` + command is tabulated in **REPRODUCE.md**.

---
*Handoff written 2026-06-29 at HEAD 3e68143. Update §2/§3 and the HEAD line when this branch moves.*
