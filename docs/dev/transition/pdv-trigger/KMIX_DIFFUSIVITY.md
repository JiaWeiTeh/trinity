# Where the mixing diffusivity comes from — manuscript draft, verified

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
> a committed artifact under `docs/dev/` (a CSV/table in `docs/dev/<workstream>/data/`, or a
> harness/figure in the relevant `docs/dev/<workstream>/` folder) — never left in
> `/tmp`, the local-only `scratch/`, or an untracked `outputs/`. A future visit must be able to reproduce or compare
> against the numbers **without re-running**; record the exact config + command
> that produced each artifact.
>
> 🔗 **Cross-check the sibling docs — keep the workstream self-consistent.** This file is one of
> several living docs for its workstream (`PLAN.md`, `FINDINGS.md`, `REPRODUCE.md`, `F_KAPPA_FUNCTIONAL_FORM.md`,
> `RUNGB_SCOPING.md`, `KAPPA_EFF_SCOPING.md`). They drift out of sync *with each other* as fast as they drift from
> the code. Any agent or person editing one MUST circle back through the siblings and reconcile any number, status,
> claim, or line reference that disagrees. Never update one in isolation.

---

## 0. What this is

The maintainer supplied a **LaTeX manuscript draft** (2026-06-29) of two paper sections — *"A functional form for
the conduction multiplier"* and *"Where the mixing diffusivity comes from"* — plus a claims/evidence table. This
file records a **line-by-line verification** of that draft against our committed results (do **not** assume the
draft is correct), and folds the parts that are new or that refine our docs. Most of the draft restates work already
in `F_KAPPA_FUNCTIONAL_FORM.md` §0–§13; the genuinely **new** content is the *origin of λδv* (§3 here).

## 1. Verification verdict (claim → status)

| draft claim | our check | status |
|---|---|---|
| f_mix scales θ linearly; f_κ acts in-structure, sublinearly; `f_mix = f_κ^q`, q≈0.3–0.4 | matches our measured raw leverage (p=0.21–0.42); f_mix=f_κ^q with q=ln1.3/ln2≈0.4 | ✓ **match** |
| q < 1/2 (El-Badry transport exponent), because f_κ stiffens Spitzer T^(5/2), not the T-independent diffusivity | our measured q∈[0.21,0.42] < 0.5; physically sound framing | ✓ **match** (good framing) |
| `f_κ(n)=[θ_target/θ_0]^(1/q)=f_mix^(1/q)`; ≈4 compact, ≈60 diffuse; "analytic skeleton of the calibration" | this is exactly our composed form (§0–§8); the anchors match `kappa_blowout_calibration` | ✓ **match** — ⚠️ the ≈60 anchor is DEAD per `FINDINGS.md §10` (2026-07-02: blowout-metric artifact; measured multiplier f_fire = 4) — do not carry it into the manuscript |
| `θ/(1−θ)=A n^(1/2)`, A≈1.6; θ_target saturates 0.94→0.999 across 1e2–1e6 | A=(5/11)·A_mix=(5/11)·3.5=**1.59** ✓; equivalent to our ψ=3.5√ form; θ values reproduce | ✓ **match** (algebra correct) |
| a flat θ_target≈0.95 floored = restates the trigger (degeneracy); density survives via θ_0(n_H) | matches FINDINGS §2a degeneracy + our §9 (density from baseline) | ✓ **match** |
| f_κ≈60 is not a conduction coefficient → faithful κ_mix is the honest object | matches §13 | ✓ **match** — ⚠️ the ≈60 anchor itself is DEAD per `FINDINGS.md §10` (measured multiplier f_fire = 4); the "not a conduction coefficient" point survives |
| **single-variable sweep design** (step n at fixed M_cl/sfe, repeat at 2nd/3rd) to separate q, A from cloud-scale confounds; "until that sweep is run, Eq is a derived expectation" | **WE RAN IT** (the 819-combo sweep). Result: **FAN-OUT** — the M_cl/sfe series do *not* collapse (×2–32 spread), so f_κ answers to more than n_H. | ⚠️ **STALE** — the draft predates the sweep; its open question is **answered** (multi-dimensional). Update before submission. |
| κ_mix=(λδv)ρk_B/(μm_p) (El-Badry Eq 21), mixing-length theory, κ_mix∝ρ T-independent | matches our verified §13 | ✓ **match** |
| KH growth ω~k v_rel (ρ_hi/ρ_lo)^(−1/2); mode matching 1/Δt_SNe ⇒ λ~Δt_SNe v_rel(contrast)^(−1/2), λδv~1 pc·km/s, λ~0.3 pc | our KH estimate lands λδv≈0.1–2 pc·km/s (v_rel 3–10 km/s); order-of-magnitude OK | ✓ **plausible** (bracket only, as the draft says) |
| **eddy-turnover replaces the SN cadence** for continuous wind driving (no Δt_SNe): set ω=δv/λ instead of 1/Δt_SNe | setting ω=δv/λ with δv~v_rel pins the **contrast** (=(2π)²≈39), **not** λ → the closure is circular/heuristic | ★ **NEW idea, but the step is hand-wavy** — the *conclusion* (λ is not cadence-set; calibrate it) is right; the specific closure does not fix λ |
| **don't import El-Badry's 1–10 pc·km/s** (doubly off-regime: discrete-SN vs continuous-wind; ISM n~0.1–10 vs GMC cores). Use El-Badry for *mechanism/structure*; take δv from v_rel; pin λ by calibrating κ_mix so resolved θ matches **Lancaster 2021b** (θ~0.9–0.99, continuous driving = the magnitude anchor) | sound and **refines** our §13 (which said "λδv∈[1,10]"); Lancaster (no cadence) is the better magnitude anchor for TRINITY | ★ **NEW refinement — adopt** |
| f_κ is bounded by the physical diffusivity (v_rel, λ), so it cannot be cranked to the diffuse end's demand → a cloud with modest (v_rel, λ) stays energy-driven | unifies the functional-form §11 and the κ_mix §13 from opposite ends | ★ **nice synthesis** — but see the tension in §4 |
| ebpeak fires for heavy + dense control, not the 6 normal (peaks 0.85–0.92); PdV/Lmech≈0.45; Da degenerate/refuted; fractal d≈0.5 (Fielding 2020) | all match our committed FINDINGS/figures | ✓ **match** |

## 2. The new content worth keeping — the origin of λδv

The draft's §"Where the mixing diffusivity comes from" is the part not already in our docs, and it is a useful
mechanism story (verified above as plausible):
- **λδv is a turbulent eddy diffusivity** (length × velocity, hence pc·km/s); κ_mix = diffusivity × heat-capacity-
  per-volume (~n k_B), El-Badry Eq 21. T-independent because eddies move heat by bulk motion (vs Spitzer's T^(5/2)).
- **Its scale comes from a KH-instability argument:** the shell/bubble interface is Kelvin–Helmholtz unstable; the
  dominant mode's growth rate matches the disruption rate. In El-Badry (discrete SNe) that rate is 1/Δt_SNe ⇒
  λ~0.3 pc, λδv~1 pc·km/s.
- **Three ingredients, carried over to TRINITY:** δv (= interface shear v_rel, which the code tracks) ✓; the
  *structure* (κ_mix∝ρ, θ∝√(λδv·n)) ✓; the *length* λ ✗ — it is **not** Δt_SNe·v_rel (no cadence under continuous
  winds) and **not** R_b (El-Badry's 0.3 pc is ~2 dex below R_b; setting λ=R_b is the refuted Damköhler over-estimate
  that gave RUNGB_SCOPING's absurd κ_mix/κ_S≈10²⁴).
- **Practical recipe (adopt):** use El-Badry for the *mechanism* + the √(λδv·n) scaling; take δv from v_rel;
  **pin the remaining scale by calibrating κ_mix (≡ the front conduction) so the resolved θ matches Lancaster
  2021b (0.9–0.99)** — Lancaster's continuous, cadence-free 3D bubbles are the right **magnitude anchor**, El-Badry
  the **mechanism anchor**. λ is the one quantity El-Badry themselves calibrate, so the division of labour is clean.

This **refines** `F_KAPPA_FUNCTIONAL_FORM.md` §13 ("the derived number is λδv∈[1,10]"): the [1,10] range is
off-regime for TRINITY and should be **re-calibrated to Lancaster's θ magnitude**, not imported.

## 3. Flags — what NOT to take at face value

1. **The single-variable-sweep paragraph is stale.** It frames the 819-combo sweep as not-yet-run; it **ran**
   (Helix, 2026-06-29) and the answer is **fan-out** (multi-dimensional f_κ; the second axis is roughly cloud
   size/column, §9–§10). The manuscript must be updated to report the measured outcome before submission.
2. **The eddy-turnover closure is heuristic.** Replacing 1/Δt_SNe with ω=δv/λ does not fix λ — it pins the density
   contrast to ≈40. The right statement is the draft's own conclusion: for continuous driving λ is the
   energy-containing scale of the sustained mixing layer, **sub-parsec and to be calibrated, not computed**. Keep
   the conclusion; soften the derivation.
3. **Route-a vs route-b tension (genuine open question).** The draft leans toward **diffuse clouds blow out
   energy-driven** (because the *physical* diffusivity, set by v_rel and a sub-pc λ, is bounded and may not reach
   θ=0.95). Our `F_KAPPA_FUNCTIONAL_FORM.md` §13 leans the other way — El-Badry's *verified* θ_target is flat-high
   even at diffuse (0.94 at n=1e2), so a non-firing 1D cloud is likely **under-cooled** and κ_mix *should* get it
   there. **Both are hypotheses; the κ_mix implementation calibrated to Lancaster decides it per cloud:** if the
   physically-calibrated κ_mix reaches the target → route b (cools, transitions); if the bounded (v_rel, λ) fall
   short → route a (energy-driven). Do not assert either until κ_mix is wired and tested (all 8 configs, units).
4. **References unverified here:** `2026arXiv260527517T` (the TRINITY paper), `Lancaster2025`, `TanOhGronke2021`,
   `Fielding2020` d≈0.5 — bibliographically plausible, not checked against PDFs.

## 4. What was folded where (2026-06-29)

- `F_KAPPA_FUNCTIONAL_FORM.md` §13: added the **Lancaster-magnitude-anchor refinement** + the **λδv-origin / λ-is-
  calibrated-not-imported** note + a pointer to this doc and the route-a/b tension.
- `PLAN.md`: ledger entry (this verification) + the refinement that the Rung-B knob's magnitude is calibrated to
  Lancaster, not imported from El-Badry's off-regime [1,10].
- This doc is the durable record of the manuscript draft and its verification.

*Written 2026-06-29 on `feature/PdV-trigger-term-pt2`. Numerical checks in the commit's analysis; no new sims.*
