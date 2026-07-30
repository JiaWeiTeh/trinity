# S5a beta/delta solve — reconciled

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

**Status (2026-07-29):** 📘 raw agent report — provenance for `FINDINGS.md`; unreconciled and unverified on its own.

**Status (2026-07-30):** 📙 reconciled slice report — merged from three blind lens reports
(`S5a_betadelta_lensA.md`, `_lensB.md`, `_lensC.md`). No source was read while writing this.
Findings below are candidates for `FINDINGS.md`; none has been verified against source.

**Method note.** I saw only the three lens reports. Lens A read comment/docstring-stripped source,
Lens B read only prose, Lens C read only signatures plus a physics spec, and Lens C had **no
literature access** (arXiv/ADS blocked), so its coefficients are derived-from-memory and it explicitly
refuses to assert Weaver+77 equation numbers. Weighting follows that: an A≠C divergence where C is
self-declared uncertain is logged as a *question to settle*, not a confirmed defect.

---

## 0. Headline: the core equation triangulates clean

Before the divergences, the strongest positive result. The central object of this slice —
`cool_beta_to_Ebdot_pure`'s map from a trial β to `Ėb` — was reached three independent ways and the
three agree:

* **A** transcribed the computed expression from stripped source (lines 248–269).
* **B** transcribed the docstring formula (`get_betadelta.py:193`, cited "Rahner thesis pg 80, Eq A12").
* **C** derived it from scratch from `Eb = (3/2)Pb Vb`, `Vb = (4π/3)(R2³−R1³)` and ram-pressure balance,
  never having seen either.

A's and B's expressions are literally the same formula. C's expression is written in a different
algebraic form; substituting `Pb = Eb/(2π(R2³−R1³))` and `ṗ = 4πR1²Pb` into C's
`Ėb = 6πPb R2²v2 − 3πPb R1³(p̈/ṗ) − (β/t)(Eb + 3πPb R1³)` reproduces A's boxed expression **term by
term** (I checked all three terms; the `0.75 ṗ R1` in A equals C's `1.5 Eb R1³/d` exactly under
ram-pressure balance). So:

* **the formula is right, the β sign is right, the dimensions balance** — at γ = 5/3, and *conditional
  on* `Pb` being formed with `(R2³−R1³)` and `pdot_total` being the same ṗ that defines R1;
* the code does what its docstring says (no doc-drift in F1) and what the physics requires;
* C's S1 findings **C-01** (formula) and **C-02** (β sign) are therefore **satisfied, not defects** —
  dropped below with reason.

That agreement is what makes the residual divergences credible: the lenses are not systematically
mis-reading each other's subject matter.

Two further clean passes: `delta2dTdt_pure` returns `+δT/t` with no minus sign (A), exactly as C
requires (**C-05 satisfied**); and `effective_Lloss`'s three branches are identical across A (code),
B (docstring) and C (physics spec), including C's preferred `max(Lcool+Lleak, θ·Lmech)` for the theta
mode (**C-10, C-12 satisfied**).

---

## 1. Coverage table

`✔` = addressed substantively · `~` = touched but not resolved · `—` = silent (**not** corroboration).

| Quantity / behaviour | A (code) | B (prose) | C (physics) | Status |
|---|---|---|---|---|
| `Ėb(β)` closed form | ✔ full transcription | ✔ docstring verbatim + own reduction | ✔ derived independently | **corroborated (agree)** |
| β sign convention `Ṗb = −βPb/t` | ✔ | ✔ stated twice | ✔ required | **corroborated (agree)** |
| δ sign convention `dT/dt = +δT/t` | ✔ | — **never documented** | ✔ required | corroborated A+C; doc gap |
| γ threading (2π, 0.75, 1.5 literals) | ✔ γ=5/3 only | ✔ same conclusion from prose | ✔ same requirement | **corroborated (defect)** |
| `compute_R1_Pb` internals (`solve_R1`, `bubble_E2P`) | ~ delegates, bodies out of slice | ~ no formula documented | ✔ full derivation | **open, corroborated concern** |
| `Vb` convention (`R2³` vs `R2³−R1³`) | ~ unverifiable in slice | ~ undocumented | ✔ requires `R2³−R1³` | open |
| `ṗ` provenance (`pdot_total` vs `2L/v`) | ✔ two different param sources | — units not even stated | ✔ must be one quantity | corroborated A+C |
| Energy-balance arm (`L−Lloss−4πR2²v2Pb`) | ✔ | ~ "gain − loss − work", work undefined | ✔ PdV exactly once | corroborated A+C (agree) |
| Inner-face work `+4πPbR1²Ṙ1` | ✔ absent | — | ~ listed parenthetically | contested |
| Residual normalisation (f vs g) | ✔ both forms | ✔ both + the pole rationale | ✔ must be commensurate | **corroborated (defect)** |
| Residual fallback branches (NaN→0, dimensional) | ✔ | — | ~ general warning | corroborated A+C(partial) |
| `RESIDUAL_THRESHOLD` semantics | ✔ 1e-4 on **sum of squares** | ✔ "the 1e-4 acceptance threshold" | ✔ needs relative 1e-4…1e-3 | **corroborated (defect)** |
| Bounds values `[0,1]`,`[−1,0]` | ✔ values | ~ "bounds exist", no values | ✔ needs β≳2 (up to ~5) | **corroborated (defect)** |
| Bounds *enforcement* per solver | ✔ grid/lbfgsb yes, hybr no | ✔ "hybr unbounded", "legacy bounded" | ✔ clipping ≠ convergence | corroborated ABC |
| Solver default | ✔ `legacy` when key missing/falsy | ✔ schema/production `hybr` | ✔ must be recorded+equivalent | **contested (two defaults)** |
| `no_physical_root` set only by hybr | ✔ | ✔ "legacy never sets it" | ✔ end-of-phase must be detected | corroborated A+B |
| `effective_Lloss` branches | ✔ | ✔ | ✔ | **corroborated (agree)** |
| Cooling-boost reaching the E→p trigger | — (out of slice) | ✔ claims 3 identical call sites | ✔ highest-value cross-check | corroborated B+C, unverified |
| `dMdt` seed carry-forward | ✔ mechanism | ✔ + a determinism claim that contradicts it | ✔ seed must not move the root | **corroborated ABC** |
| Plateau sentinel value | ✔ `100.0, 100.0` | ~ "a plateau value", unnamed | ✔ failures must be classified | corroborated A+B |
| L-BFGS-B acceptance re-test | ✔ re-evaluated and re-ranked | ~ one-line docstring | ✔ must be re-tested as a root | **C refuted by A** |
| Grid geometry (5×5, ±0.02, spacing 0.01) | ✔ values | ✔ shape, no values | ✔ requirements on spacing | corroborated A+C |
| hybr numerics (`xtol` 1e-8, `eps` 3e-4, `maxfev` 30) | ✔ values | ✔ rationale, **no values** | ✔ noise-floor requirement | corroborated ABC |
| `T0` value / measurement point ξ | ~ read from params | ✔ flagged as never documented | ✔ ξ must be fixed (cancellation) | corroborated B+C |
| δ closure `δ=(2/7)(2α−β−1)` | — | — | ✔ C only | **single-lens** |
| Power-law Weaver asymptotes `β=(4+w)/(5−w)` | — | — | ✔ C only | **single-lens** |
| Clock origin of `t_now` | ~ read from params | — | ✔ must be the SPS clock | **single-lens** |
| Convergence status persisted to outputs | ~ fields exist on the result | ✔ "for saving to dictionary" | ✔ must be in jsonl/metadata | contested |
| Citations (Rahner A12/A5, El-Badry, Lancaster) | — (stripped) | ✔ recorded verbatim | — (no literature access) | **single-lens B, unverifiable** |
| Caller behaviour on `converged=False` | ~ explicitly out of slice | ~ explicitly out of slice | ✔ caller must act | corroborated as an *unknown* |

**Silence to note explicitly:** B never gives a numeric constant that the prose does not state, so B
cannot corroborate any of A's values; C never saw a line of code, so no C finding is evidence *about*
the code, only about what the code must satisfy; A never saw a citation, so the Rahner A12/A5
attributions are **single-lens B and completely unverified** — nothing in this slice tests whether
the cited equations say what the docstring claims. That is the single largest coverage hole here, and
only fetching the thesis closes it.

---

## 2. Divergence table

| # | Quantity | A says | B says | C says | Class | Verdict |
|---|---|---|---|---|---|---|
| D1 | β admissible range | `BETA_MAX = 1.0`; enforced by grid/L-BFGS-B, **not** by hybr | bounds exist; hybr "unbounded", legacy "bounded" | asymptote `β=(4+w)/(5−w)`; **>1 for any `densPL_alpha` steeper than −0.5**, `=2` at −2, "well above 2" pre-transition | **AC** | **Code is the defect.** The declared box is physically too narrow. Corollary: A-06's proposed fix (clip hybr to the box) would *introduce* C's clipping bug — the right fix is to widen the box, then enforce it consistently. → R-01 |
| D2 | γ in the `Ėb` coefficients | `2π` and `0.75` are γ=5/3 literals; `gamma_adia` goes only to `bubble_E2P` | docstring's A12 has a bare `2π`, no γ dependence; `compute_R1_Pb` takes `gamma_adia` | `6π→4π/(γ−1)`, `3π→2π/(γ−1)`; a γ honoured in only some places is a fake knob | none (all agree) | Agreed defect, not a divergence: `gamma_adia` is a partially-honoured parameter. Escalates only if a `.param` can set γ≠5/3. → R-02 |
| D3 | `Vb`/`Pb` volume convention | unverifiable (delegates out of slice) | undocumented; flags the `R1` naming drift ("inner bubble radius" vs "termination shock radius") | must be `(R2³−R1³)` in both places; `O(1)` error exactly at the transition | AC (open) | **Unresolved by construction** — needs `get_bubbleParams.solve_R1`/`bubble_E2P`. All three lenses independently flagged it. → R-03 |
| D4 | Meaning of `1e-4` | sum of squares of two **relative** residuals ⇒ ~1e-2 per component | "the 1e-4 acceptance threshold" | relative 1e-4…1e-3 needed; **≥1e-2 aliases into the 0.05 transition trigger** | **ABC** | Prose overstates the tightness by 2 decades; C's bar says 1e-2 is the "too loose" regime. A stated the arithmetic in its narrative but did **not** raise it as a finding — this row is a pure reconciliation product. → R-04 |
| D5 | Residual determinism | seed threaded via `bubble_dMdt` override; residual depends on evaluation history | ":520 get_bubbleproperties_pure is **deterministic in (params, beta, delta)**" | seed must change the path, never the root | **AB** + C | **Docstring is false** (stale/wrong), and the underlying behaviour is a real reproducibility defect. → R-05 |
| D6 | Reported vs scored evaluation | `converged`/`total_residual` from the ranking pass; every other field from a second, differently-seeded re-solve that can fail | re-solve skip is "equivalent to re-solving" | `detailed` must equal `pure` | AB + C | Code defect; C-23's "different route" form is refuted (A: same arithmetic), the *seed* is what differs. → R-06 |
| D7 | Residual at the `Ėb→0` pole | `|X|≤1e-300` ⇒ `f_E := B` (raw luminosity), and `abs(NaN)>0` is False ⇒ **NaN→0.0** | the f denominator "crosses zero near the Eb peak — the pole" (the stated reason g exists) | residuals must be dimensionless and commensurate | AB(+C) | **A supplies the mechanism, B proves the regime is real and traversed.** C-14's general form is refuted for the primary path (both branches *are* relative) and survives only here. → R-07 |
| D8 | Solver default | `'legacy'` when the key is missing/falsy | production default `'hybr'`; key-less unit-test fixtures fall back to `legacy` | at most one of two implementations is the published model | AB (contested) | Two defaults for one key; the production path may be untested. One grep of `trinity/_input/` settles it. → R-08 |
| D9 | "legacy is bounded / domain-respecting" | grid endpoints are clamped **independently**, so a guess >GRID_EPSILON outside the box yields an inverted `linspace` with every node out of bounds | ":650 the legacy grid is bounded and penalty-guarded, so its optimum is a domain-respecting seed" | clipping must never be reported as convergence | **AB** | **Prose claim is false in exactly the case the rescue exists for.** Chain: unbounded hybr (A-06) → out-of-box guess → inverted grid (A-09) → rescue returns an out-of-domain seed (B-17) → failure repeats. B-17 as *stated* is refuted (A shows the exact-guess equality check), but it survives through A-09. → R-09 |
| D10 | `R1 → R2` / `d ≤ 0` | `|d(1−cf)| < 1e-300` ⇒ **`return 0.0`**; `d < 0` not caught at all | — | `R1 ≥ R2` is *guaranteed* eventually and is the physical end of the energy phase; must be trapped, never allowed to produce negative volume | **AC** | Code turns a physically meaningful end-of-regime signal into `Ėb = 0`, which then trips D7's fallback. → R-10 |
| D11 | End-of-regime detection | `no_physical_root` never set on the legacy path | "legacy never sets it"; the runner hands off on the flag | the disappearance of a root is physics and must end the phase | none (A=B) | Agreed: under `legacy` there is no physical end-of-phase criterion. → R-11 |
| D12 | δ closure | δ enters **only** through the structure solve; residual 2 is `(T_b−T0)/T0`; `delta2dTdt_pure` is **not called in this module** | T0 and the measurement ξ never documented | expected `r_δ = (dT/dt)^struct − delta2dTdt_pure(...)`; δ must satisfy `(2/7)(2α−β−1)` | **AC** | **Contested.** C's framing is self-rated *medium* and it may simply be a different-but-equivalent closure. Two cheap experiments settle it (sweep δ at fixed β; check the closure relation). → R-13 |
| D13 | L-BFGS-B acceptance | its candidate's residual is **re-evaluated** and re-ranked against the same 1e-4 root test | `LBFGSB_THRESHOLD=5.0` is a *gate on whether to run*, not a tolerance | "a local minimum of ‖g‖ must not be accepted as a root" | AC | **C-19 refuted** by A and B jointly. Dropped. |
| D14 | Plateau sentinel size | `100.0, 100.0` ⇒ total 2e4 | value never stated; "if it is modest, a failed point could win the sort" | failures must be classified and counted | AB | **B-18's strong form refuted** by A's value (2e4 ≫ 5.0). Survives only as exception-masking. → R-22 |
| D15 | Solver cascade itself | grid → L-BFGS-B, hybr, rescue, legacy — five code paths | each path individually justified | the Jacobian is diagonally dominant; a seeded Newton should converge in <10 iterations, so the cascade is a **symptom** of a noisy/non-smooth residual | **scope-creep** | A and B document and justify machinery that C says the physics does not require. Not a bug per se; it is the reason a noise-floor measurement is the highest-value experiment here. → R-23 |
| D16 | `dMdt ≤ 0` | rejected by `_usable_dMdt`; raises `_NoPhysicalRoot` | a condensation root is "real physics" that must trigger the momentum handoff, not be retried | condensation is a **real late-energy-phase regime** that should arguably be modelled, not treated as the end | **BC** | Modelling choice, documented, questioned by the physics lens. C self-rates *medium* and is literature-blocked. → R-26 |
| D17 | `[au]` as a pressure unit | independently establishes the code system = M⊙/pc/Myr | flags "[au]" as reading like *astronomical unit* | pressure is `M⊙ pc⁻¹ Myr⁻²` | none | **B-07 demoted**: "au" is this project's own name for its code-unit system (A confirms from `unit_conversions.py`). Still uninformative, now cosmetic. → R-31 |
| D18 | Grid centre-skip near a bound | clamps applied to each endpoint *before* `linspace`, so the centre shifts off the guess | describes both "clamped" and "shifted" and flags the tension | — | none | **B-27 resolved by A**: the "shifted" reading is what the code does. Dropped as a finding; the prose could be clearer. |

---

## 3. Findings dropped or demoted (be a filter, not an amplifier)

| Lens finding | Was | Now | Why |
|---|---|---|---|
| C-01 `Ėb(β)` expression | S1 | **dropped** | A's transcription reproduces C's derived expression term-for-term at γ=5/3. Requirement satisfied. |
| C-02 β sign | S1 | **dropped** | A: `Ṗb = −βPb/t`, B: docstring states it twice. Requirement satisfied. |
| C-05 δ sign | S1 | **dropped** | A: `dT/dt = (T/t)·δ`, no minus. Requirement satisfied. (The *doc* gap survives as R-29.) |
| C-03 `p̈/ṗ` guard | S2 | **narrowed → R-17** | The guard exists (`a := 0` if `ṗ ≤ 0`). Only C's tiny-positive-ṗ sub-case and the asymmetry with `c_coeff` survive. |
| C-10 / C-12 `effective_Lloss` | S2/S3 | **dropped** | A=B=C on all three branches, including the `max()` that C-12 asked for. |
| C-13 PdV work | S1 | **mostly satisfied → R-25 (S3)** | The outer work term appears exactly once with the right sign and coefficient. Only C's parenthesised inner-face term is absent. |
| C-14 non-dimensionalisation | S1 | **refuted for the primary path → folded into R-07** | Both primary residuals are relative and dimensionless. Only the fallback branches are dimensional. |
| C-17 clipping/failure reported as convergence | S1 | **mostly refuted → R-06/R-07** | A's explicit audit: every non-converged path sets `converged=False`. Two specific mechanisms survive. |
| C-18 persistence of convergence status | S2 | **contested → R-27 (S4)** | B's contract lists diagnostic fields "for saving to dictionary"; the strong form ("no trace anywhere") is not supported. |
| C-19 minimiser ≠ root-finder | S1 | **dropped** | The L-BFGS-B candidate is re-evaluated and re-ranked against the same root test; `LBFGSB_THRESHOLD` is a run-gate, not a tolerance. |
| C-21 branch continuity | S2 | **dropped** | The solve is seeded from the previous accepted point and the grid box is ±0.02, i.e. the "prefer the nearest root" behaviour C asks for is already the design. |
| C-23 `detailed` vs `pure` | S3 | **narrowed → R-06** | A: "same arithmetic". The divergence is the *seed*, not the route. |
| C-24 view purity | S2 | **narrowed → R-33** | A: the view writes nothing and required keys are read as `params['X'].value` (which raises). Only `bubble_Leak`'s `getattr` default matches C's concern. |
| B-07 `[au]` | S4 | **kept at S4, demoted rationale** | "au" is the project's code-unit system, not astronomical units (A). |
| B-17 rescue returns the failing seed | S2 | **refuted as stated → survives inside R-09** | A shows the exact-guess equality check. It re-emerges only through the inverted-grid path. |
| B-18 plateau too small | S2 | **refuted → R-22 (S3)** | A gives the value: `100.0, 100.0` ⇒ total 2e4, far above `LBFGSB_THRESHOLD = 5.0`. |
| B-27 clamped vs shifted grid | S4 | **dropped** | A resolves it: endpoints are clamped before the `linspace`, so the centre genuinely shifts. |
| A-06 "hybr should be bounded" | S2 | **reframed → R-01** | Enforcing the *current* box would clip the physical root for steep density profiles. The defect is the box, plus the inconsistency between paths. |

Input: 19 (A) + 27 (B) + 28 (C) = **74 raw findings → 34 reconciled candidates** (65 of the 74 fold
into a merged entry; 9 were dropped outright per the table above). Of the 34: **25 corroborated**
(≥2 lenses independently), **5 single-lens**, **4 contested**.

---

## 4. What the missing literature would settle

Lens C worked without arXiv/ADS/OUP. Fetching **Rahner's thesis (imprs-hd.mpg.de/399417), pp. 79–80,
Eqs A5 and A12** would resolve, in one read:

1. **R-03** — whether A12's `d` is `R_b³ − R_ts³` and whether Rahner's `P_b` uses the same volume.
   This is the only way to close the `Vb`-convention question without also reading
   `get_bubbleParams` (which the next slice can do independently).
2. **R-02** — whether A12 is stated for general γ or specialised to 5/3 in the source. C derived
   γ=5/3 independently and B derived it from the prose, so the *fact* is settled; what the thesis
   settles is whether the docstring's citation is faithful (S3 citation) or whether the code inherited
   a γ-specialisation the thesis flags (S1 if the thesis generalises and the code silently did not).
3. **R-13 / R-29** — A5's δ convention and, critically, whether A5's residual is a `dT/dt` match (C's
   reading) or a `T` match at a reference ξ (what the code does). This is the largest open physics
   question in the slice.
4. **R-26** — whether WARPFIELD/Rahner admit `Ṁ < 0` (condensation) inside the energy phase or treat
   it as the end of the regime.

Everything else in this slice is settleable from source alone.

---

## 5. Severity rubric used

* **S1** — produces wrong numbers on configurations run today. *Nothing in this slice reaches S1 on
  the available evidence*; **R-01** and **R-02** carry explicit escalation conditions to S1, each
  resolvable by one grep of `trinity/_input/`. Lens C assigned 11 S1s; every one of them is either a
  requirement the code already satisfies (§3) or a claim no lens could verify from its own view — the
  S1 count falling to zero is the reconciliation working, not the findings being softened.
* **S2** — wrong output conditionally (a reachable regime, a settable parameter, one of two solver
  paths), or an unverifiable correctness claim in a load-bearing place.
* **S3** — doc-drift, fragility, honesty-of-reporting, numerical hygiene.
* **S4** — cosmetic, dead code, naming, unsourced prose.

---

## 6. Verify these first

1. **R-01 — `BETA_MAX = 1.0` vs `β_Weaver = (4+w)/(5−w)`.** The only divergence in the slice with a
   number on both sides: A read the constant, C derived the asymptote, and the arithmetic says the
   cap is below the physical root for every `densPL_alpha` steeper than −0.5 (and is exactly half the
   asymptote at −2). Two cheap checks settle it — read the schema default for `betadelta_solver`
   (does the bounded path ever run in production?) and solve a `densPL_alpha = −2` config against the
   analytic 2.0. Note the trap: "fixing" A-06 by clipping hybr to the current box makes this worse.
2. **R-04 — what `1e-4` actually means.** A's own narrative says the test is on the *sum of squares*
   of two relative residuals, i.e. ~1 % per component; B's prose calls it "the 1e-4 acceptance
   threshold"; C's propagation of the 0.05 transition trigger says ≥1e-2 relative is the regime where
   the closure error aliases into the headline result. No single lens raised this. One instrumented
   run recording `fE, fT, gE, gT` at acceptance answers it.
3. **R-03 — the `Vb` and `ṗ` conventions across `compute_R1_Pb` ↔ `cool_beta_to_Ebdot_pure`.** All
   three lenses flagged it independently and none could resolve it, because the bodies live in
   `get_bubbleParams`. It is the one place where the otherwise-clean core equation can be silently
   wrong, and it is wrong by `O(1)` precisely at the transition. Resolved by the next slice, or by
   Rahner pp. 79–80.

Runner-up: **R-13** (is δ actually constrained?) — the `sweep δ at fixed β` experiment is minutes of
work and would either close a contested S2 or expose the most consequential defect in the slice.

---

```json
[
  {
    "id": "S5a-R-01",
    "file": "trinity/phase1b_energy_implicit/get_betadelta.py",
    "line": 41,
    "class": "regime",
    "severity": "S2",
    "claim": "The declared admissible box BETA_MIN/BETA_MAX = 0.0/1.0, DELTA_MIN/DELTA_MAX = -1.0/0.0 is narrower than the physical solution space, and the two solver paths disagree about enforcing it: the grid (1043-1046) and L-BFGS-B (1118-1119, 1127) clamp to it, the hybr path never applies it and returns unclipped.",
    "evidence": "Lens A read the constants and the enforcement sites; Lens B read prose calling hybr 'unbounded' and legacy 'bounded', with hybr the stated production default; Lens C derived the power-law-generalised Weaver asymptote beta = (4+w)/(5-w), delta = -6/(7(5-w)) for rho ~ r^-w. Reconciler arithmetic on C's formula: beta exceeds 1.0 for any w > 0.5, i.e. for every densPL_alpha steeper than -0.5, and equals 2.0 at densPL_alpha = -2; C adds that beta must run well above 2 approaching the transition. The default guess cool_beta 0.8 leaves only 0.2 of headroom to BETA_MAX, and the grid moves at most 0.02 per call.",
    "expected": "A box that contains the supported physics (C suggests beta in [~0, ~5], delta in [~-1, ~+0.5]), enforced identically by every solver path, with a bound-hit reported as non-converged rather than as a root. Note the fix ordering: widening must precede enforcing — clipping hybr to the present box would introduce the clipping defect rather than remove it.",
    "failure_scenario": "A densPL_alpha <= -1 run on the legacy/grid path (or any run whose rescue path routes through the grid): the true beta is at or above the cap, every grid node is a boundary node, the returned beta is pinned at 1.0, Pb is held too high, the shell is over-driven and the energy->momentum transition is delayed — reported with converged True whenever the pinned residual happens to fall below threshold.",
    "repro": "Assert BETA_MAX > 2 and that the same bounds are applied on the hybr path; then run a densPL_alpha = -2 config and compare the solved beta against the analytic asymptote 2.0 and against BETA_MAX. Escalates to S1 on confirming that any published/steep-profile run used a bounds-enforcing path.",
    "confidence": "high",
    "lenses": ["A", "B", "C"],
    "divergence": "AC",
    "status": "corroborated",
    "source_ids": ["S5a-A-06", "S5a-B-13", "S5a-C-16"]
  },
  {
    "id": "S5a-R-02",
    "file": "trinity/phase1b_energy_implicit/get_betadelta.py",
    "line": 260,
    "class": "coefficient",
    "severity": "S2",
    "claim": "cool_beta_to_Ebdot_pure hard-codes gamma = 5/3 in the literals 2*np.pi (line 260), 0.75 (line 252) and 1.5 (line 251), and takes no gamma argument, while compute_R1_Pb passes gamma_adia to bubble_E2P (line 329) and not to solve_R1 (line 327). gamma_adia is therefore a partially honoured parameter.",
    "evidence": "Three independent routes to the same conclusion. A: the literals reproduce the analytic derivative of Eb = 4*pi*Pb*d/(3*(gamma-1)) exactly and only at gamma=5/3, with 2*pi = 4*pi/(3*(gamma-1)) and 0.75 = 1/(2*(gamma-1)). B: the docstring's A12 states a bare 2*pi with no gamma dependence, and B's own reduction shows the equation collapses to d/dt[2*pi*Pb*d] only if Pb = Eb/(2*pi*d), i.e. gamma=5/3. C: requires 6*pi -> 4*pi/(gamma-1) and 3*pi -> 2*pi/(gamma-1) for general gamma and names a partially-threaded gamma a 'fake knob'.",
    "expected": "Either thread gamma_adia into cool_beta_to_Ebdot_pure (2*np.pi -> 4*np.pi/(3*(gamma-1)), 0.75 -> 1/(2*(gamma-1))), or assert gamma_adia == 5/3 and document A12 as gamma=5/3-only.",
    "failure_scenario": "A .param setting gamma_adia != 5/3 (e.g. 1.4): Pb is computed with the new gamma but Edot_from_beta keeps the 5/3 coefficients, so the beta residual is driven to zero against the wrong target and beta is biased by roughly (2/3)/(gamma-1), with no warning.",
    "repro": "Check whether gamma_adia is user-settable in trinity/_input/ schema defaults. Then: pick R1, R2, Eb, pdot satisfying pdot*(R2**3-R1**3) == 3*(g-1)*Eb*R1**2 and Pb = 3*(g-1)*Eb/(4*pi*(R2**3-R1**3)); assert cool_beta_to_Ebdot_pure equals the analytic 4*pi/(3*(g-1))*d/dt[Pb*d]. Passes at g=5/3, fails at g=1.4. Escalates to S1 if any shipped .param sets gamma_adia != 5/3.",
    "confidence": "high",
    "lenses": ["A", "B", "C"],
    "divergence": "none",
    "status": "corroborated",
    "source_ids": ["S5a-A-01", "S5a-B-05", "S5a-C-09"]
  },
  {
    "id": "S5a-R-03",
    "file": "trinity/phase1b_energy_implicit/get_betadelta.py",
    "line": 297,
    "class": "coefficient",
    "severity": "S2",
    "claim": "The Ebdot identity is exact only if (a) Pb is formed with the bubble volume (4*pi/3)*(R2**3 - R1**3), not the Weaver R2**3 shortcut, and (b) the pdot_total passed to cool_beta_to_Ebdot_pure is the same momentum rate that defines R1 (pdot = 4*pi*R1**2*Pb = 2*Lmech_total/v_mech_total). Neither is verified anywhere in this slice, and the two quantities demonstrably come from different sources at the call site.",
    "evidence": "A: compute_R1_Pb delegates to get_bubbleParams.solve_R1(R2, Eb, Lmech_total, v_mech_total) and bubble_E2P(Eb, R2, R1, gamma_adia) — bodies outside the slice; and get_residual_pure reads pdot_total/pdotdot_total from params while R1 is built from Lmech_total/v_mech_total, i.e. two different provenances for the same physical quantity. B: compute_R1_Pb documents no formula at all, tags Pb only as '[au] (code units)', and calls R1 'Inner bubble radius' where the A12 docstring calls it 'Termination shock radius R_ts'. C: derives Pb = 3(gamma-1)Eb/(4*pi*(R2**3-R1**3)) and R1 = sqrt(Lmech/(2*pi*v_mech*Pb)), notes the pair is coupled, and flags passing both Pb and Eb into the Ebdot map as a consistency hazard that hides a convention mismatch at the call site.",
    "expected": "One bubble-volume convention shared by compute_R1_Pb, the Ebdot map and the PdV work term; one pdot definition shared by R1 and cool_beta_to_Ebdot_pure. An assertion abs(4*pi*R1**2*Pb/pdot_total - 1) < 1e-10 at the call site would pin both.",
    "failure_scenario": "If Pb is formed with R2**3 while the Ebdot algebra assumes R2**3 - R1**3, the kinematic identity is violated by O((R1/R2)**3): negligible early, order unity exactly when Eb collapses and R1 -> R2, i.e. at the energy->momentum transition, corrupting the headline transition time. If the two pdot values differ, R1dot is the derivative of a different R1 than the one in Vb, with the error growing once SNe dominate the momentum budget.",
    "repro": "Read get_bubbleParams.solve_R1 and bubble_E2P (next slice) and check the denominator (R2**3 vs R2**3-R1**3) and whether R1/Pb are solved self-consistently; separately assert 4*pi*R1**2*Pb == pdot_total at every call in get_residual_pure. Finite-difference check: advance the state by eps using the returned Edot_b, recompute (Pb, R1), assert d(ln Pb)/dt == -beta/t to O(eps).",
    "confidence": "medium",
    "lenses": ["A", "B", "C"],
    "divergence": "AC",
    "status": "corroborated",
    "source_ids": ["S5a-A-01", "S5a-B-06", "S5a-C-04", "S5a-C-07", "S5a-C-27"]
  },
  {
    "id": "S5a-R-04",
    "file": "trinity/phase1b_energy_implicit/get_betadelta.py",
    "line": 47,
    "class": "numerical",
    "severity": "S2",
    "claim": "RESIDUAL_THRESHOLD = 1e-4 is compared against the SUM OF SQUARES of two residuals, so 'converged' means each relative residual is only ~1e-2 (a 1 % tolerance) — two decades looser than the prose implies and inside the range the physics lens names as 'too loose'. The same constant is additionally applied to two differently normalised energy residuals (legacy f_E = (X-B)/X, hybr g_E = (X-B)/Lmech_total).",
    "evidence": "A (narrative, not raised as a finding): the convergence test at 712/751/792/827/966/1002 is on fE**2 + fT**2 < 1e-4, so each component is ~1e-2; the grid early-exit bar 1e-5 is ~0.3 % each. B: prose calls 1e-4 'the acceptance threshold' with no mention of the squaring, and separately documents that gE is Lmech-normalised while gT is relative — a mixed-scale vector tested by one constant, whose value and form the prose never states. C: propagating the transition trigger (L_gain-L_loss)/L_gain <= 0.05 to ~1 % accuracy in the transition time needs a relative residual of ~1e-3 or tighter, and names >= 1e-2 relative as the regime where the closure error aliases straight into the trigger.",
    "expected": "Either compare sqrt(fE**2 + fT**2) to the threshold, or set the constant to the square of the intended per-component tolerance and say so; and give the f-metric and g-metric separately named, separately calibrated thresholds since 1e-4 means a different physical tightness under each.",
    "failure_scenario": "Accepted points routinely carry ~1 % relative error in the bubble energy balance; that error is 20 % of the 0.05 loss-fraction trigger, so the reported energy->momentum transition time is partly a numerical artefact. Separately, a state with |Edot_from_beta| ~ 1e-3 * Lmech_total passes the hybr test with ~100 % error in Edot while failing the legacy test at the same point.",
    "repro": "Instrument accepted points across param/simple_cluster.param and docs/dev/performance/f1edge_{lowdens,hidens}*.param: record fE, fT, gE, gT separately at acceptance and histogram the per-component magnitudes. Persist as a committed CSV.",
    "confidence": "medium",
    "lenses": ["A", "B", "C"],
    "divergence": "ABC",
    "status": "corroborated",
    "source_ids": ["S5a-A-07", "S5a-B-15", "S5a-C-15"]
  },
  {
    "id": "S5a-R-05",
    "file": "trinity/phase1b_energy_implicit/get_betadelta.py",
    "line": 978,
    "class": "state",
    "severity": "S2",
    "claim": "The dMdt seed is carried forward between residual evaluations (grid points at 1081/1086, hybr iterations at 978) and reaches the structure solver through the bubble_dMdt override, so the residual is a function of evaluation history rather than of (beta, delta) alone — while the docstring at :520 asserts get_bubbleproperties_pure is 'deterministic in (params, beta, delta)' and uses that assertion to justify skipping a re-solve.",
    "evidence": "A: the seed mechanism and its three propagation sites; hybr's finite-difference Jacobian columns are therefore built from differently seeded evaluations. B: the determinism docstring, contradicted in the same file by the warm-start prose at :108/:1017; plus B-21's observation that the warm-start justification ('adjacent grid points differ by at most GRID_EPSILON') does not match the centre-out scan order, whose consecutive evaluated points can be a full grid diagonal apart; plus B-22's point that 'the previous segment's accepted dMdt carried in params' requires someone to write params, contradicting the module's purity claim. C: a seed may change the iteration path but must never change the accepted root, on pain of non-reproducibility.",
    "expected": "Either seed deterministically from a fixed reference state per call, or correct the docstring to 'equal to fsolve tolerance, not bit-identical' and document bubble_dMdt as inter-segment mutable state. A Newton solver with a numerical Jacobian needs a deterministic residual.",
    "failure_scenario": "Two runs of the same config whose grid scans early-exit at different points carry different seeds and converge to different dMdt at the same accepted (beta, delta) — a reproducibility break that propagates into bubble_properties and defeats the project's own bit-identical equivalence gate for any change that alters seeding order.",
    "repro": "Re-solve a captured step with dMdt_guess perturbed by +/-20 % and +/-2x and assert the returned (beta, delta) agree to solver tolerance; also call get_residual_pure twice at the same (beta, delta) with dMdt_guess=None and with a neighbour's value and diff the residuals. Then grep for writes to params['bubble_dMdt'] across the phase-1b runner.",
    "confidence": "high",
    "lenses": ["A", "B", "C"],
    "divergence": "AB",
    "status": "corroborated",
    "source_ids": ["S5a-A-13", "S5a-B-04", "S5a-B-21", "S5a-B-22", "S5a-C-25"]
  },
  {
    "id": "S5a-R-06",
    "file": "trinity/phase1b_energy_implicit/get_betadelta.py",
    "line": 841,
    "class": "state",
    "severity": "S2",
    "claim": "On the legacy path total_residual and converged come from the ranking evaluation, but every other returned field (Edot_residual, T_residual, Edot_from_beta/balance, T_bubble, L_gain, L_loss, bubble_properties) comes from a second, differently-seeded re-evaluation at line 841 that can fail outright and return Edot_residual=100.0, NaN physical fields, bubble_props=None and dataclass-default L_gain=L_loss=0.0.",
    "evidence": "A: line 841 calls get_residual_detailed(best_beta, best_delta, params, bubble_props=best_props); the L-BFGS-B candidate is appended with props=None (788), so for that candidate the call rebuilds BubbleParamsView WITHOUT dMdt_guess (534) and re-solves; the failure path (541-549) supplies the sentinel values, which are copied into the result at 852-865 while total_residual and converged stay from the earlier pass. B: the re-solve skip is justified by a determinism claim that R-05 shows is false. C: get_residual_detailed must agree with get_residual_pure exactly, or every diagnostic figure describes a model that was never integrated.",
    "expected": "Carry the residuals and props from the winning evaluation, or recompute total_residual from the same details object, so that no result can report converged=True alongside Edot_residual=100 and bubble_properties=None.",
    "failure_scenario": "grid_residual > 5.0, the L-BFGS-B branch runs and wins, the re-solve at 841 lands on a different dMdt branch or fails: the caller receives converged=True with total_residual=1e-5 alongside Edot_residual=100, L_gain=L_loss=0 and bubble_properties=None, and any bubble_properties that do propagate downstream come from a different structure solve than the one that scored the point.",
    "repro": "Monkeypatch get_bubbleproperties_pure to raise on the second call, force the lbfgsb candidate to win, and assert result.converged implies isfinite(result.Edot_from_beta) and result.bubble_properties is not None. Separately, property-test get_residual_detailed against get_residual_pure at random (beta, delta) with a pinned seed.",
    "confidence": "high",
    "lenses": ["A", "B", "C"],
    "divergence": "AB",
    "status": "corroborated",
    "source_ids": ["S5a-A-05", "S5a-B-04", "S5a-C-23"]
  },
  {
    "id": "S5a-R-07",
    "file": "trinity/phase1b_energy_implicit/get_betadelta.py",
    "line": 481,
    "class": "silent-failure",
    "severity": "S2",
    "claim": "The residual fallback branches (481, 493 and their duplicates at 585, 593) are both dimensional AND NaN-swallowing: when |Edot_from_beta| <= 1e-300 the code assigns f_E := Edot_from_balance (a raw luminosity) and when |T0| <= 1e-300 it assigns f_T := T_bubble (raw K); and because 'abs(x) > 0' is False for NaN, a NaN residual is mapped to 0.0, which passes the convergence test.",
    "evidence": "A: the branch structure, and the fact that cool_beta_to_Ebdot_pure returns exactly 0.0 by design when |denominator| < 1e-300 (line 266-267), so the |X| <= 1e-300 branch is reachable from inside the same module. B (independently, from prose): the legacy f denominator 'crosses zero near the E_b peak — the pole' — this is the documented rationale for introducing g, i.e. the prose confirms real runs traverse the regime where this branch is live. C: residuals must be dimensionless and commensurate before being combined (C's general form is refuted for the primary branches, which are relative, and survives only here).",
    "expected": "Return a large finite sentinel or raise on a non-finite residual — never map NaN to zero — and normalise the fallback by a fixed scale (Lmech_total for Edot, a reference temperature for T) as the hybr gE already does.",
    "failure_scenario": "A step where R2 has effectively caught up with R1 so the denominator collapses -> Edot_from_beta = 0.0 -> f_E takes the balance branch; if the structure solve also produced NaN luminosities, total_res_input = 0.0 passes the < 1e-4 test at line 712 and the solver returns converged=True with iterations=0 at the untouched input guess. Alternatively T0 = 0 makes f_T ~ 1e6-1e7, its square dominates the objective and every reported residual is meaningless.",
    "repro": "get_residual_pure with params such that R2 == R1 and bubble_LTotal = nan; assert the returned Edot_residual is not 0.0 and that converged cannot be True. Separately set params['T0'].value = 0.0 and assert |T_residual| is O(1).",
    "confidence": "medium",
    "lenses": ["A", "B", "C"],
    "divergence": "AC",
    "status": "corroborated",
    "source_ids": ["S5a-A-03", "S5a-A-04", "S5a-C-14"]
  },
  {
    "id": "S5a-R-08",
    "file": "trinity/phase1b_energy_implicit/get_betadelta.py",
    "line": 613,
    "class": "divergence",
    "severity": "S2",
    "claim": "Two full, non-equivalent solver implementations coexist (legacy grid+L-BFGS-B, hybr) with two different defaults for one key — the docstring says the production default is 'hybr' while the code falls back to 'legacy' when betadelta_solver is missing or falsy — no documented equivalence gate between them, and (per the docstring itself) the unit-test fixtures are the key-less ones.",
    "evidence": "A: _get_betadelta_solver returns 'legacy' on None/missing/falsy; the two paths differ in residual normalisation, bounds enforcement, failure flagging and convergence bookkeeping. B: ':613 robust to params that predate the betadelta_solver key (e.g. the unit-test fixtures), which fall back to the legacy path' against ':631 hybr (production default)'; also the module header still advertises grid-first as the default. C: two implementations of one closure in a stiff iterative path require a documented full-run equivalence gate, the selection must be recorded, and at most one of them is the published model.",
    "expected": "One default; the selected solver recorded in metadata.json; an equivalence gate run on the stiffest regimes per CLAUDE.md rule 5; and test fixtures that exercise the production path.",
    "failure_scenario": "The production path (hybr: unbounded root-find, _NoPhysicalRoot as BaseException, g residual, no_physical_root handoff, rescue) is never exercised by the unit tests, so regressions land green; and published results depend on which path a config happens to select, with two runs of nominally the same physics diverging in transition time and final fate.",
    "repro": "Read the schema default for betadelta_solver in trinity/_input/; grep test/ for 'betadelta_solver' and count tests reaching _solve_betadelta_hybr vs _solve_betadelta_legacy; then run param/simple_cluster.param and the f1edge configs under each setting, in separate processes, and compare R2, v2, Eb, Pb at matched simulation time.",
    "confidence": "high",
    "lenses": ["A", "B", "C"],
    "divergence": "AB",
    "status": "corroborated",
    "source_ids": ["S5a-B-01", "S5a-B-12", "S5a-C-22"]
  },
  {
    "id": "S5a-R-09",
    "file": "trinity/phase1b_energy_implicit/get_betadelta.py",
    "line": 1043,
    "class": "numerical",
    "severity": "S2",
    "claim": "The grid box clamps are applied independently to each endpoint (beta_min = max(0, b0-0.02), beta_max = min(1, b0+0.02)), so a guess more than GRID_EPSILON outside the declared bounds produces an inverted, entirely out-of-bounds linspace — and that guess is reachable, because the hybr path returns unclipped (beta, delta) that seed both the next step and the rescue call.",
    "evidence": "A: for b0 = 1.05, beta_min = 1.03 > beta_max = 1.00 and np.linspace(1.03, 1.00, 5) silently returns a descending sequence with every element >= 1.0; same construction for delta. B (from the opposite direction): ':650 the legacy grid search is bounded and penalty-guarded, so its optimum is a domain-respecting seed' — the claim the rescue path relies on. Chain assembled across lenses: unbounded hybr (R-01) -> out-of-box guess -> inverted grid (here) -> the rescue hands back an out-of-domain point that is not float-equal to the guess, so _rescue_structure_failure retries hybr from it -> the failure repeats every segment, which is the all-NaN bubble_Lloss loop the rescue was written to break.",
    "expected": "np.clip the guess into the box before forming the endpoints so beta_min <= beta_max always holds; and state whether the rescue may return an out-of-bounds point.",
    "failure_scenario": "hybr returns beta = 1.05 on a stiff step; the grid then scans beta in [1.00, 1.03] only, evaluating the structure solve entirely outside the physical range and returning the best of those as the 'domain-respecting seed'.",
    "repro": "_solve_grid(beta_guess=1.05, delta_guess=-0.5, ...): assert beta_range.min() >= BETA_MIN and beta_range.max() <= BETA_MAX. Then force a structure failure at an out-of-box (beta, delta) and assert the rescued seed lies inside the declared bounds.",
    "confidence": "high",
    "lenses": ["A", "B"],
    "divergence": "AB",
    "status": "corroborated",
    "source_ids": ["S5a-A-09", "S5a-B-17"]
  },
  {
    "id": "S5a-R-10",
    "file": "trinity/phase1b_energy_implicit/get_betadelta.py",
    "line": 266,
    "class": "silent-failure",
    "severity": "S2",
    "claim": "The degenerate state R1 -> R2 is absorbed rather than detected: cool_beta_to_Ebdot_pure returns 0.0 when |denominator| < 1e-300, and R1 > R2 (negative d_coeff) is not trapped at all, so it silently flips the sign of the whole expression instead of ending the energy phase.",
    "evidence": "A: line 266 '|denominator| < 1e-300 -> return 0.0'; the returned 0.0 then trips the |Edot_from_beta| <= 1e-300 fallback in the residual (see R-07) and swaps a relative residual for a dimensional one. Nothing in the transcription guards d_coeff < 0. C: R1 ~ Pb^-1/2 ~ Eb^-1/2, so R1 crossing R2 is guaranteed if the energy phase runs long enough; it is the physical statement 'the bubble can no longer stand off the wind' and must be converted into an end-of-energy-phase / no-physical-root outcome, never allowed to produce a negative bubble volume or negative Pb.",
    "expected": "Detect R1 >= R2 (and Eb <= 0) explicitly and raise the no-physical-root / end-of-phase condition the runner already understands, instead of returning 0.0 or a sign-flipped value.",
    "failure_scenario": "Late in the energy phase the bubble volume goes to zero and then negative; the code reports Edot_from_beta = 0.0, the residual falls into its dimensional fallback, and the step is either accepted at the unchanged guess or scored with a meaningless number — the end of the energy-driven regime is missed and the trajectory continues on a negative-volume bubble.",
    "repro": "Sweep Eb downward at fixed R2 and assert the function raises or flags before R2**3 - R1**3 changes sign; scan a run's snapshots for R1 >= R2 or Pb <= 0.",
    "confidence": "medium",
    "lenses": ["A", "C"],
    "divergence": "AC",
    "status": "corroborated",
    "source_ids": ["S5a-C-08", "S5a-A-03"]
  },
  {
    "id": "S5a-R-11",
    "file": "trinity/phase1b_energy_implicit/get_betadelta.py",
    "line": 170,
    "class": "regime",
    "severity": "S2",
    "claim": "The no_physical_root flag — the signal the runner uses to end the implicit phase and hand off to the transition phase — is set only on the hybr path; the legacy path never sets it, so under betadelta_solver='legacy' (including the key-less unit-test fixtures) there is no physical end-of-regime detection at all.",
    "evidence": "A (code): 'no_physical_root is never set on this path — a total failure returns converged=False, no_physical_root=False, bubble_properties=None'. B (prose, independently): ':170 legacy never sets it' plus ':933 the runner hands off on the flag'. C: the disappearance of a root is physically meaningful (strong cooling, R1->R2) and must end the energy phase rather than be integrated through.",
    "expected": "Either legacy raises the same condition, or Phase 1b's termination criterion under legacy is named and documented.",
    "failure_scenario": "A comparison or reproduction run set to 'legacy' continues integrating the energy-driven implicit phase past the point where no physical root exists, converging on plateau-penalty or bounds-pinned beta/delta instead of handing off to the transition phase — so the two solver settings disagree about the phase-transition time, the code's headline output.",
    "repro": "Run the same stiff config under both solver settings to the same simulation time and compare the phase-transition time recorded in the output.",
    "confidence": "high",
    "lenses": ["A", "B", "C"],
    "divergence": "none",
    "status": "corroborated",
    "source_ids": ["S5a-B-14"]
  },
  {
    "id": "S5a-R-12",
    "file": "trinity/phase1b_energy_implicit/get_betadelta.py",
    "line": 360,
    "class": "divergence",
    "severity": "S2",
    "claim": "effective_Lloss_from_params is claimed to be the single point where the cooling boost is applied, feeding the beta-delta residual, the energy ODE's Edot_from_balance and the energy->momentum trigger identically — but only two of the three call sites are inside this slice, so the cross-module invariant is asserted from inside one module and is unverified.",
    "evidence": "B: ':335/:361 the single point where the opt-in unresolved-interface-cooling boost is applied ... thin wrapper so the three call sites stay one line and identical'; only :472 and :576 are visible. C: SPEC-013/014 define the transition on (L_gain - L_loss)/L_gain <= 0.05 and SPEC-015 records that the published Paper-II grid runs with cooling_boost_fmix = 4, so the boosted and unboosted paths genuinely differ in shipped configurations; C ranks this its highest-value cross-check for this function.",
    "expected": "Exactly three call sites of effective_Lloss_from_params and no other site composing Lcool + Lleak by hand.",
    "failure_scenario": "If the energy ODE or the E->p trigger sums Lcool + Lleak directly, then with cooling_boost enabled the bubble evolves with boosted cooling but transitions on unboosted cooling (or vice versa) — the transition time decouples from the dynamics that produced it and Paper-II's grid is internally inconsistent.",
    "repro": "Grep the package for 'effective_Lloss_from_params' and, separately, for every expression combining Lcool with Lleak; confirm the counts match. Then instrument a cooling_boost_mode=multiplier, fmix=4 run and assert the L_loss in the transition-trigger diagnostic equals the L_loss in the Edot residual to machine precision.",
    "confidence": "medium",
    "lenses": ["B", "C"],
    "divergence": "none",
    "status": "corroborated",
    "source_ids": ["S5a-B-23", "S5a-C-11"]
  },
  {
    "id": "S5a-R-13",
    "file": "trinity/phase1b_energy_implicit/get_betadelta.py",
    "line": 488,
    "class": "divergence",
    "severity": "S2",
    "claim": "The delta residual is a temperature MATCH, (T_bubble - T0)/T0, not the dT/dt-consistency residual the physics lens expected; delta2dTdt_pure is not called anywhere in this module; and neither T0's value nor the measurement point xi = r/R2 is documented. Whether delta is genuinely constrained by this residual is therefore open.",
    "evidence": "A: residual 2 is (T_b - T0)/T0 with T_b = bubble_props.bubble_T_r_Tb; delta enters ONLY through the structure solve; delta2dTdt_pure (272-294) is 'not called anywhere in this module'. B: ':400 difference between bubble temperature and target temperature T0', ':487 temperature at measurement point vs target temperature' — with neither T0 nor xi given a value anywhere in the slice. C: expected r_delta = (dT/dt)^structure - delta2dTdt_pure(t, T, delta), i.e. the pure function is the definition side of the residual — but rates C's own framing of the delta residual as only *medium* confidence, and separately warns that a delta whose residual is weak leaves delta effectively unconstrained.",
    "expected": "Either the two framings are shown equivalent, or the delta closure is documented for what it is. In either case T0 and xi must be traceable to a .param key rather than a literal, and xi must be held fixed (C's argument: a fixed xi cancels in d ln T/d ln t, an adaptive one does not).",
    "failure_scenario": "If the T-match residual does not actually pin delta, the solver reports convergence on a beta-dominated condition and returns whatever delta the seed carried; the interior temperature profile and hence L_cool are wrong while every diagnostic says converged.",
    "repro": "Hold beta at the solved value and sweep delta over its box; if the total residual varies by less than the convergence threshold across the sweep, delta is unconstrained. Separately, extract (alpha, beta, delta) from a run and test C's prefactor-free closure delta ~= (2/7)(2*alpha - beta - 1) with alpha = v2*t/R2. Reading Rahner Eq A5 (p. 79) settles the intended framing.",
    "confidence": "low",
    "lenses": ["A", "B", "C"],
    "divergence": "AC",
    "status": "contested",
    "source_ids": ["S5a-B-09", "S5a-C-14", "S5a-C-26"]
  },
  {
    "id": "S5a-R-14",
    "file": "trinity/phase1b_energy_implicit/get_betadelta.py",
    "line": 771,
    "class": "regime",
    "severity": "S3",
    "claim": "The L-BFGS-B fallback is gated on 'not grid_converged AND grid_residual > 5.0', so residuals in [1e-4, 5] get no second solver — while the _solve_betadelta_legacy docstring says the fallback runs 'if grid search fails or doesn't converge', without the conjunction.",
    "evidence": "A: line 771 is the conjunction; GRID_EPSILON = 0.02 caps per-call motion and _solve_grid's best_* is initialised to the input guess, so a failed grid returns the guess essentially unmoved. B: ':684 automatically falls back to lbfgsb if grid search fails or doesn't converge' against ':50-52/:765 only run L-BFGS-B if grid residual exceeds this ... AND grid residual is bad' — a prose self-contradiction that A resolves in favour of the conjunction.",
    "expected": "Fix the docstring to state the conjunction; and justify why a residual up to 5e4x the acceptance threshold is 'a reasonable result' not worth refining, or gate on 'not grid_converged'.",
    "failure_scenario": "A stiff step whose true root is 0.1 away in beta: the grid pins beta at guess +/- 0.02 with residual ~0.5, no fallback runs, and the caller receives converged=False with a value only marginally moved. Whether that changes physical output depends entirely on how the caller treats converged=False — which no lens could see.",
    "repro": "Construct a segment whose grid residual lands in (1e-4, 5.0) and confirm no L-BFGS-B run occurs; then check whether any caller of solve_betadelta_pure / get_beta_delta_wrapper_pure branches on result.converged.",
    "confidence": "high",
    "lenses": ["A", "B"],
    "divergence": "AB",
    "status": "corroborated",
    "source_ids": ["S5a-A-10", "S5a-B-02", "S5a-B-03"]
  },
  {
    "id": "S5a-R-15",
    "file": "trinity/phase1b_energy_implicit/get_betadelta.py",
    "line": 1010,
    "class": "numerical",
    "severity": "S3",
    "claim": "_solve_grid quantises the answer to a lattice of spacing 2*GRID_EPSILON/(GRID_SIZE-1) = 0.01 and caps per-call motion at 0.02 in each coordinate, so whenever the true per-step change in beta exceeds 0.02 the accepted node is always a boundary node — box clipping in disguise.",
    "evidence": "A supplies the numbers (GRID_SIZE=5, GRID_EPSILON=0.02, best_* seeded from the input guess so a failed grid returns the guess). C supplies the requirement: GRID_EPSILON must exceed the true per-step |d beta|, the lattice spacing must induce an L_cool error below tolerance, and a boundary-node result must be treated as non-converged and refined; beta changes fastest near the transition, exactly where the grid is most likely to be out-run. Reconciler note: with the default guess 0.8 and BETA_MAX 1.0, the grid needs at least ten consecutive calls just to traverse its own admissible range.",
    "expected": "Either a step-adaptive box, or a boundary-node result flagged as non-converged; and a demonstration that the lattice spacing's dL_cool is below the residual tolerance.",
    "failure_scenario": "beta(t) becomes a staircase, L_cool is piecewise-constant, the outer ODE right-hand side is discontinuous, and the transition time is quantised by the grid.",
    "repro": "Histogram the solved beta over a full run and look for lattice clustering at spacing 0.01; separately record how often the accepted node lies on the grid boundary. Applies to the legacy path only, so run it with betadelta_solver='legacy'.",
    "confidence": "medium",
    "lenses": ["A", "C"],
    "divergence": "AC",
    "status": "corroborated",
    "source_ids": ["S5a-A-10", "S5a-C-20"]
  },
  {
    "id": "S5a-R-16",
    "file": "trinity/phase1b_energy_implicit/get_betadelta.py",
    "line": 262,
    "class": "numerical",
    "severity": "S3",
    "claim": "Ebc = Eb + c_coeff is guarded before being used as a divisor at line 256 (c_frac) but not at line 262, so Ebc == 0 raises ZeroDivisionError (including the 0.0/0.0 case when a_coeff == 0) and Ebc < 0 evaluates a formula whose third numerator term and denominator disagree in sign.",
    "evidence": "A: line 256 'c_frac = c_coeff / Ebc if Ebc > 0 else 0.0' versus the unguarded '- a_coeff * R1**3 * Eb**2 / Ebc' at 262. When the guard at 256 fires the denominator at 264 becomes d_coeff (not d_coeff*Eb/Ebc) while the third term still carries a negative Ebc. Reachability is the open question: Ebc = Eb*(1 + 1.5*R1**3/d) is positive whenever Eb > 0 and R1 < R2, so this needs Eb <= 0 or pdot_total < 0 — states that C's asymptotics say do occur late in the energy phase.",
    "expected": "One guard covering both uses — return early or fall back consistently when Ebc <= 0, rather than zeroing c_frac while keeping 1/Ebc live.",
    "failure_scenario": "Bubble energy driven negative late in the energy-driven phase, or pdot_total < 0: exactly zero raises an unhandled ZeroDivisionError, slightly negative returns a value whose third term has the wrong sign relative to the suppressed c_frac normalisation.",
    "repro": "cool_beta_to_Ebdot_pure(beta=0.5, Pb=1.0, t_now=1.0, R1=1.0, R2=2.0, v2=1.0, Eb=-0.75, pdot_total=1.0, pdotdot_total=0.0) -> ZeroDivisionError instead of a finite value.",
    "confidence": "high",
    "lenses": ["A"],
    "divergence": "none",
    "status": "single-lens",
    "source_ids": ["S5a-A-02"]
  },
  {
    "id": "S5a-R-17",
    "file": "trinity/phase1b_energy_implicit/get_betadelta.py",
    "line": 251,
    "class": "numerical",
    "severity": "S3",
    "claim": "The pdotdot/pdot guard is on the SIGN, not the magnitude ('a_coeff = 1.5*pdotdot_total/pdot_total if pdot_total > 0 else 0.0'), so an arbitrarily small positive pdot_total still produces an unbounded a_coeff; and c_coeff = 0.75*pdot_total*R1 on the next line is not guarded at all, so for non-positive pdot_total the two coefficients derived from the same momentum-flux relation disagree.",
    "evidence": "A: lines 251-252, and the observation that the two coefficients are then derived from mutually inconsistent assumptions. C: independently demanded a guard here because SPS tables have a wind/SN gap where the total momentum injection rate passes through a deep minimum, and explicitly asked that a tiny positive pdot_total (1e-300) also return finite. C's requested guard exists in the pdot <= 0 case (so C-03's main form is satisfied) but not in the small-positive case.",
    "expected": "Guard on magnitude relative to a physical scale, and treat pdot_total <= 0 consistently in both coefficients (both derive from the same ram-pressure relation) or reject the state.",
    "failure_scenario": "In the wind-SN gap, pdot_total passes through a deep minimum: a_coeff blows up and Edot_from_beta becomes inf/nan (poisoning every residual), or pdot_total turns non-positive and the pdotdot term is silently discarded while a negative c_coeff still reshapes the denominator and can drive Ebc <= 0 (see R-16).",
    "repro": "Call with pdot_total = 1e-300 and pdotdot_total = 1.0 and assert the return is finite; call with pdot_total = -1.0 and check that a_coeff = 0 while c_coeff = -0.75*R1 — neither the full formula nor a clean fallback.",
    "confidence": "medium",
    "lenses": ["A", "C"],
    "divergence": "AC",
    "status": "corroborated",
    "source_ids": ["S5a-A-17", "S5a-C-03"]
  },
  {
    "id": "S5a-R-18",
    "file": "trinity/phase1b_energy_implicit/get_betadelta.py",
    "line": 74,
    "class": "numerical",
    "severity": "S3",
    "claim": "HYBR_OPTIONS = dict(xtol=1e-8, factor=0.1, maxfev=30, eps=3e-4) pairs a step tolerance four orders of magnitude below the finite-difference step, caps the budget at ~30 evaluations for a 2-D problem, and never acts on sol.status; and the eps value was measured as the noise floor of the OLD f metric, before the switch to the Lmech-normalised g metric changed the residual's scale.",
    "evidence": "A: the constant values, and that sol.success/sol.status appear only in the log message while convergence is decided solely by g_total < 1e-4. B: ':70-71 the finite-difference step eps is the residual noise floor measured in the Phase-2.1 transect probe (docs/dev/archive/betadelta/diagnostics), not the 1e-4 acceptance threshold' — a load-bearing justification whose numeric value is not in the code comment and which points into docs/dev material the project's own rules declare unverified. C: independently ranks 'measure the residual noise floor and compare it to RESIDUAL_THRESHOLD' as the single highest-value experiment in this slice.",
    "expected": "xtol commensurate with eps; the numeric eps stated next to the constant so the code is self-describing; and either act on sol.status or state that the residual test supersedes it.",
    "failure_scenario": "Termination is normally by evaluation budget rather than by convergence, and after the f->g switch (which rescaled the residual by ~1/Lmech) the Jacobian is differenced on a step that no longer matches the noise floor, so convergence degrades silently across the mass sweep.",
    "repro": "Log sol.status across a run (expect status 5, maxfev exceeded, on stiff steps rather than status 1); re-run the transect probe under the g metric and compare the measured noise floor against eps = 3e-4 and against RESIDUAL_THRESHOLD. Persist as a committed CSV.",
    "confidence": "medium",
    "lenses": ["A", "B", "C"],
    "divergence": "none",
    "status": "corroborated",
    "source_ids": ["S5a-A-14", "S5a-B-25", "S5a-C-28"]
  },
  {
    "id": "S5a-R-19",
    "file": "trinity/phase1b_energy_implicit/get_betadelta.py",
    "line": 904,
    "class": "silent-failure",
    "severity": "S3",
    "claim": "gE divides by Lmech_total with no zero or finiteness guard, and the resulting ZeroDivisionError has no handler anywhere on the hybr path — only _NoPhysicalRoot is caught (962, 986, 998), and it deliberately subclasses BaseException so that except-Exception blocks do not catch it.",
    "evidence": "A: lines 903-904 'Lmech_total = float(params[...].value); gE = (...) / Lmech_total'; no handler exists between there and solve_betadelta_pure's caller. B corroborates the exception-type design ':870 a BaseException, not Exception' but says nothing about the division.",
    "expected": "Guard Lmech_total (fall back to a nonzero scale, or raise _NoPhysicalRoot with a reason), as the other divisions in this file are guarded.",
    "failure_scenario": "betadelta_solver='hybr' at a time when the cluster mechanical luminosity has dropped to exactly zero (a truncated SB99 table returning 0, or feedback switched off): the run aborts with an unhandled ZeroDivisionError instead of reporting no_physical_root.",
    "repro": "_hybr_g_residual with params['Lmech_total'].value = 0.0 -> ZeroDivisionError.",
    "confidence": "medium",
    "lenses": ["A"],
    "divergence": "none",
    "status": "single-lens",
    "source_ids": ["S5a-A-08"]
  },
  {
    "id": "S5a-R-20",
    "file": "trinity/phase1b_energy_implicit/get_betadelta.py",
    "line": 357,
    "class": "silent-failure",
    "severity": "S3",
    "claim": "effective_Lloss falls through to the unboosted Lcool + Lleak for any unrecognised cooling_boost_mode, and the docstring frames this as a feature ('a typo cannot perturb a run') — the exact opposite of the policy the same file applies to an unknown betadelta_solver, which raises ValueError.",
    "evidence": "A (code): only 'multiplier' and 'theta_target' are matched; every other value, including misspellings and strings with trailing whitespace, reaches the bare return Lcool + Lleak, and '... or none' additionally maps ''/None/0 to 'none'. B (prose): the same behaviour documented verbatim as intentional, alongside ':645 the param validator guards user input; this guards programmatic misuse' for the solver key. Code and prose agree; the two config keys have opposite invalid-value policies.",
    "expected": "One policy for unknown enum values in this module — validate at load and raise, or at minimum log a warning on the fallback.",
    "failure_scenario": "A .param with cooling_boost_mode='mutliplier' and fmix=4 runs the entire sweep with no boost applied and no warning, and the published result is attributed to a boosted run.",
    "repro": "Check trinity/_input/ for a cooling_boost_mode enum/validator; if absent, run with a deliberately misspelled mode and confirm no warning is emitted. effective_Lloss('multiplyer', 2.0, 0.0, 1.0, 0.0, 10.0) should not silently return 1.0.",
    "confidence": "high",
    "lenses": ["A", "B"],
    "divergence": "none",
    "status": "corroborated",
    "source_ids": ["S5a-A-15", "S5a-B-10", "S5a-B-11"]
  },
  {
    "id": "S5a-R-21",
    "file": "trinity/phase1b_energy_implicit/get_betadelta.py",
    "line": 649,
    "class": "state",
    "severity": "S3",
    "claim": "The physically decisive distinction between a rescuable structure-solve failure and a genuine condensation root (dMdt <= 0, which must trigger the momentum handoff) is carried only in a human-readable reason string and matched by substring at the caller.",
    "evidence": "B: ':650 a found condensation root (\"non-physical dMdt=...\") is real physics and must NOT be retried away — the caller only routes \"structure solve failed\" reasons here', while ':880 _hybr_g_residual raises _NoPhysicalRoot if the structure solve fails OR the resulting dMdt is non-finite / <= 0' — one exception type for two causes. A confirms the mechanism from the code side: the rescue fires when no_physical_root is set and the reason CONTAINS 'structure solve failed'.",
    "expected": "Carry the cause as a typed field or enum on the exception/result rather than a message substring, or pin the exact strings with a test.",
    "failure_scenario": "A reworded message, or one that happens to contain both phrases, routes a genuine condensation root into the grid rescue — retrying away the physics that should trigger the momentum handoff — or conversely leaves a rescuable failure unrescued and re-creates the all-NaN bubble_Lloss columns the rescue was written to fix.",
    "repro": "Find every construction site of the two reason strings and the caller's comparison; add a test asserting each cause routes to the intended branch.",
    "confidence": "medium",
    "lenses": ["A", "B"],
    "divergence": "none",
    "status": "corroborated",
    "source_ids": ["S5a-B-16"]
  },
  {
    "id": "S5a-R-22",
    "file": "trinity/phase1b_energy_implicit/get_betadelta.py",
    "line": 439,
    "class": "silent-failure",
    "severity": "S3",
    "claim": "get_residual_pure wraps the structure solve in a bare 'except Exception -> return 100.0, 100.0, None', which converts ANY failure — including programming errors such as a callee using dict protocols BubbleParamsView does not implement (__setitem__, __contains__, keys, iteration, attribute access) — into a plateau score rather than a diagnosed failure.",
    "evidence": "A: the handler at 439, the sentinel values, and the observation that BubbleParamsView implements only __getitem__ and get, so any callee using another form raises and is swallowed here. B: ':870 get_residual_pure's except Exception plateau handler' and ':650 structure failures score a plateau value instead of aborting', with the value never stated in prose. B-18's worry that the plateau might be too small to dominate the candidate sort is REFUTED by A's value: 100**2 + 100**2 = 2e4, far above LBFGSB_THRESHOLD = 5.0. C: exceptions from the inner structure integration must be classified (integration failure vs non-finite vs R1 >= R2 vs table out-of-range) and counted.",
    "expected": "Catch the expected failure classes explicitly, log via _describe_exc (which exists for exactly this purpose), and count the swallowed failures per run.",
    "failure_scenario": "A refactor that introduces a genuine bug in the params-access contract shows up as a run in which every step scores the plateau and the solver quietly returns the input guess for the whole phase, indistinguishable from a hard physics regime.",
    "repro": "Assert the handler logs (via _describe_exc) and that a TypeError/AttributeError from the view protocol is not scored as a plateau; count plateau hits per run and record them in metadata.json.",
    "confidence": "medium",
    "lenses": ["A", "B", "C"],
    "divergence": "none",
    "status": "corroborated",
    "source_ids": ["S5a-B-18", "S5a-C-24"]
  },
  {
    "id": "S5a-R-23",
    "file": "trinity/phase1b_energy_implicit/get_betadelta.py",
    "line": 625,
    "class": "other",
    "severity": "S3",
    "claim": "The slice contains five solver code paths (grid, L-BFGS-B fallback, hybr, legacy wrapper, structure-failure rescue) for a 2-D root-find whose Jacobian the physics lens argues is diagonally dominant and whose root should be unique and easy from the previous step's seed — machinery the physics does not call for, and a symptom of a noisy or non-smooth residual rather than a fix for one.",
    "evidence": "A and B jointly document and justify all five paths (A from code structure, B from prose rationale for each). C: r_beta is dominated by -(Eb/t)(1 + 1.5*R1**3/(R2**3-R1**3)) and r_delta by -T/t, so a well-seeded Newton should converge in well under ten iterations; the cascade is evidence that either the inner ODE/shooting tolerance is looser than RESIDUAL_THRESHOLD (chasing integration noise) or the residual is non-smooth (piecewise-constant per-age cooling FILE selection; a max() branch). Both are fixable at the source; neither is fixed by adding another minimiser.",
    "expected": "Measure the residual noise floor and the per-step solver path counts before adding or keeping fallbacks; if the noise floor exceeds the threshold, fix the inner tolerance or the table interpolation instead.",
    "failure_scenario": "If the inner integration noise exceeds the outer threshold, no solver can converge, the cascade fires every step, and the accepted (beta, delta) is whatever the last fallback produced — robust-looking silent degradation.",
    "repro": "Evaluate get_residual_pure ~20x at fixed inputs with jittered dMdt seeds and measure the spread; instrument which solver path is taken per step over param/simple_cluster.param and the f1edge configs; commit both as CSVs.",
    "confidence": "medium",
    "lenses": ["A", "B", "C"],
    "divergence": "scope-creep",
    "status": "corroborated",
    "source_ids": ["S5a-C-28"]
  },
  {
    "id": "S5a-R-24",
    "file": "trinity/phase1b_energy_implicit/get_betadelta.py",
    "line": 442,
    "class": "regime",
    "severity": "S3",
    "claim": "The t_now used for beta (and for alpha = v2*t/R2 elsewhere) must be the cluster age measured from feedback onset, i.e. the SPS table's t = 0; nothing in this slice verifies the clock origin, and a phase-local clock would shift all three similarity exponents systematically with no visible error.",
    "evidence": "C only: the (alpha, beta, delta) triple are logarithmic derivatives about a common time origin, so Weaver similarity requires t measured from bubble birth while the SPS table's t=0 is cluster formation. A confirms only that t_now is read from params; B states the unit (Myr) but not the origin.",
    "expected": "t_now = cluster age in Myr, identical to the SPS-table clock, with a t <= 0 guard (which delta2dTdt_pure has and cool_beta_to_Ebdot_pure appears to lack).",
    "failure_scenario": "A phase-local clock shifts all three exponents; the closure delta = (2/7)(2*alpha - beta - 1) fails systematically and the bubble structure is built on wrong coefficients, with no error raised.",
    "repro": "Extract alpha = v2*t/R2 from an energy-phase run of param/simple_cluster.param and assert it relaxes to ~0.6; a systematic offset indicates a clock mismatch. Cheap and prefactor-free.",
    "confidence": "medium",
    "lenses": ["C"],
    "divergence": "none",
    "status": "single-lens",
    "source_ids": ["S5a-C-06"]
  },
  {
    "id": "S5a-R-25",
    "file": "trinity/phase1b_energy_implicit/get_betadelta.py",
    "line": 475,
    "class": "coefficient",
    "severity": "S3",
    "claim": "The energy-balance arm uses only the outer work term, Edot_from_balance = Lmech - Lloss - 4*pi*R2**2*v2*Pb, while the kinematic arm retains the full Vdot including R1 motion; the inner-face work term +4*pi*Pb*R1**2*R1dot is absent from the balance side.",
    "evidence": "A transcribes the balance at line 475. C's expected form is 'L_gain - L_loss - Pb*4*pi*R2**2*v2 (+ Pb*4*pi*R1**2*R1dot)' — the inner term in parentheses. Reconciler algebra: equating the code's two arms gives L - Lloss = (5/2)*Pb*4*pi*R2**2*v2 - 6*pi*Pb*R1**2*R1dot - beta*Eb/t, whereas the strict merged form (5/2)*Pb*Vdot - beta*Eb/t gives -10*pi*Pb*R1**2*R1dot — a difference of exactly 4*pi*Pb*R1**2*R1dot. NOTE: the standard Weaver/WARPFIELD statement of the bubble energy equation also omits this term, and C put it in parentheses, so this may be the intended convention rather than a defect.",
    "expected": "Either the inner-face term is included on the balance side, or the omission is documented as the adopted convention with its magnitude bounded (it scales as R1**2*R1dot / (R2**2*v2)).",
    "failure_scenario": "Negligible while R1 << R2; approaches order unity as R1 -> R2 and R1dot grows (R1 ~ Eb**-1/2), i.e. exactly at the energy->momentum transition, where a small systematic bias in beta translates into a shifted transition time.",
    "repro": "Evaluate 4*pi*Pb*R1**2*R1dot / (4*pi*Pb*R2**2*v2) along a full run and check where it exceeds the residual tolerance. Rahner Eq A12's surrounding text would settle the intended convention.",
    "confidence": "low",
    "lenses": ["A", "C"],
    "divergence": "AC",
    "status": "contested",
    "source_ids": ["S5a-C-13"]
  },
  {
    "id": "S5a-R-26",
    "file": "trinity/phase1b_energy_implicit/get_betadelta.py",
    "line": 374,
    "class": "regime",
    "severity": "S3",
    "claim": "Negative dMdt is treated as a non-physical root that ends the energy-driven regime (momentum handoff), whereas the physics lens argues condensation — the conduction front reversing so hot gas condenses onto the shell, dMdt < 0 — is a real late-energy-phase regime that this module is supposed to model.",
    "evidence": "A: _usable_dMdt returns None unless dMdt is finite and > 0, and _hybr_g_residual raises _NoPhysicalRoot on dMdt <= 0. B: documents the intent — a condensation root is 'real physics' and must NOT be retried away, but is routed to the KAPPA_FREEZE momentum-handoff semantics rather than modelled. C: rejecting negative dMdt 'forecloses a physically real regime ... if negative values are filtered out the late-phase structure is biased' — but rates this medium confidence and is literature-blocked ('I recall the sign reversal but cannot pin where WARPFIELD draws the line').",
    "expected": "Either an explicit, cited justification for treating dMdt <= 0 as the end of the regime, or an admitted condensation branch with a documented sign convention.",
    "failure_scenario": "If condensation is a genuine sub-regime of the energy phase, the code ends the phase early and hands off to momentum-driven evolution at the wrong time — again the headline output.",
    "repro": "Literature check first (Rahner thesis / WARPFIELD treatment of the evaporation-condensation reversal); only then consider a code change. Meanwhile, count how often runs terminate the energy phase on the dMdt <= 0 branch versus the cooling-balance trigger.",
    "confidence": "low",
    "lenses": ["B", "C"],
    "divergence": "BC",
    "status": "contested",
    "source_ids": ["S5a-C-25"]
  },
  {
    "id": "S5a-R-27",
    "file": "trinity/phase1b_energy_implicit/get_betadelta.py",
    "line": 500,
    "class": "other",
    "severity": "S4",
    "claim": "Per-step convergence status and achieved residual may not survive into the run outputs: the physics lens requires a per-snapshot convergence flag in dictionary.jsonl and aggregate counts (converged / rescued / fell-back / clipped) in metadata.json.",
    "evidence": "C states the requirement. B partially refutes the strong form: ':162-169/:505-511 residual_Edot1_guess, residual_Edot2_guess, residual_T1_guess, residual_T2_guess, bubble_Lgain, bubble_Lloss are carried on both BetaDeltaResult and ResidualDetails, for saving to dictionary'. A confirms those fields exist on the result object. What is unverified is whether `converged`, `total_residual` and `no_physical_root` are among the persisted columns and whether any aggregate count reaches metadata.json.",
    "expected": "A boolean/enum convergence status and a numeric residual per snapshot, plus aggregate counts in metadata.json's termination_debug block.",
    "failure_scenario": "A run whose closure failed on a large fraction of steps is indistinguishable in the published outputs from a clean run, so no reader can tell which grid cells are trustworthy.",
    "repro": "Grep a produced dictionary.jsonl and metadata.json for a beta-delta convergence field and for any non-convergence count.",
    "confidence": "medium",
    "lenses": ["B", "C"],
    "divergence": "BC",
    "status": "contested",
    "source_ids": ["S5a-C-18"]
  },
  {
    "id": "S5a-R-28",
    "file": "trinity/phase1b_energy_implicit/get_betadelta.py",
    "line": 629,
    "class": "deadcode",
    "severity": "S4",
    "claim": "Two dead identifiers: the 'method' parameter is threaded through solve_betadelta_pure, _solve_betadelta_legacy, _solve_betadelta_hybr and _rescue_structure_failure and read by none of them (and the sole public entry point does not forward it at all); and lbfgsb_result is assigned at line 790 and never read.",
    "evidence": "A: 'method' declared at 629/682/948/649 and passed at 639/641/643/664/667 with no function body referencing it; lbfgsb_result appears only at 768 and 790, unlike grid_result which is read at 769. B corroborates the first from prose: ':949 method is accepted for signature parity with the legacy solver and ignored' — so the dead parameter is documented rather than accidental, and a caller passing method='lbfgsb' under the production solver gets hybr with no warning.",
    "expected": "Remove the parameter at the dispatch boundary (or warn when a non-default method reaches hybr); delete lbfgsb_result or use it as grid_result is used.",
    "failure_scenario": "A diagnostic script forces method='lbfgsb' to compare optimisers, silently gets identical hybr results, and concludes the method has no effect.",
    "repro": "ruff F841 flags lbfgsb_result; the project's ruff set (F821/F811/F823/E9) does not include it, so it is not caught today. grep 'method' inside the four function bodies.",
    "confidence": "high",
    "lenses": ["A", "B"],
    "divergence": "none",
    "status": "corroborated",
    "source_ids": ["S5a-A-11", "S5a-A-12", "S5a-B-26"]
  },
  {
    "id": "S5a-R-29",
    "file": "trinity/phase1b_energy_implicit/get_betadelta.py",
    "line": 273,
    "class": "citation",
    "severity": "S4",
    "claim": "delta2dTdt_pure's docstring gives only a citation (Rahner pg 79, Eq A5) and parameter units — no formula and, critically, no sign convention — while beta's opposite-signed convention is stated explicitly and twice. The code itself is correct.",
    "evidence": "B: the docstring omission, contrasted with ':193/:247 beta = -(t/Pb)(dPb/dt)'. A: the body is 'return 0.0 if t <= 0 else (T/t)*delta', i.e. dT/dt = +delta*T/t. C: independently requires exactly +delta*T/t with no minus sign, because delta = +dlnT/dlnt carries the OPPOSITE convention to beta = -dlnPb/dlnt, and warns that any tidying that harmonises the two flips a first-order term. So the code is verified correct by two independent lenses; only the documentation is missing.",
    "expected": "State the delta convention inline as beta's is: 'delta = +(t/T)(dT/dt), negative while the bubble cools — note the opposite sign convention to beta'.",
    "failure_scenario": "A future editor 'harmonises' the two conventions, flipping a first-order term in the interior density scaling dln n/dln t = -(beta+delta) and mis-computing L_cool, with nothing in the docstring to stop them.",
    "repro": "Documentation change only. The regression guard is the existing behaviour: assert delta2dTdt_pure(t=1.0, T=1e7, delta=-6/35) is negative.",
    "confidence": "high",
    "lenses": ["A", "B", "C"],
    "divergence": "none",
    "status": "corroborated",
    "source_ids": ["S5a-B-08", "S5a-C-05"]
  },
  {
    "id": "S5a-R-30",
    "file": "trinity/phase1b_energy_implicit/get_betadelta.py",
    "line": 261,
    "class": "other",
    "severity": "S4",
    "claim": "The (1 - c_frac) factor on the second numerator term cancels exactly against the (1 - c_frac) in the denominator, so it is algebraically inert while appearing to modulate that term; the same redundancy is present in the docstring's transcription of the cited equation.",
    "evidence": "A: numerator term 2 = 3*Eb*v2*R2**2*(1-c_frac) over denominator d_coeff*(1-c_frac) gives 3*Eb*v2*R2**2/d_coeff independent of c_frac; only the first and third numerator terms actually see c_frac. B's verbatim docstring formula carries the identical structure, so the redundancy is inherited from the cited equation as printed, not introduced by the implementation.",
    "expected": "Leave the arithmetic alone (changing it is a bit-identity risk on an iterative path for no benefit) and add a one-line comment noting the cancellation, so a future editor does not read c_frac as scaling the v2 term.",
    "failure_scenario": "",
    "repro": "Evaluate with the (1-c_frac) factor removed from line 261 and c_frac forced to 0 in the denominator only; the difference isolates the redundancy.",
    "confidence": "high",
    "lenses": ["A", "B"],
    "divergence": "none",
    "status": "corroborated",
    "source_ids": ["S5a-A-18"]
  },
  {
    "id": "S5a-R-31",
    "file": "trinity/phase1b_energy_implicit/get_betadelta.py",
    "line": 193,
    "class": "units",
    "severity": "S4",
    "claim": "Pb is documented as 'Bubble pressure [au] (code units)' in two docstrings, an uninformative tag for a pressure whose code unit is Msun/(pc*Myr^2).",
    "evidence": "B raised it, reading '[au]' as potentially meaning astronomical unit (a length). A independently establishes from unit_conversions.py that 'au' is this project's own name for its code-unit system (mass Msun, length pc, time Myr, temperature K), which substantially defuses B's misreading scenario — the tag is project jargon, not a wrong unit. It remains uninformative: every neighbouring parameter in the same docstring is given an explicit pc/Myr/Msun unit.",
    "expected": "'Pb : bubble pressure [Msun/(pc*Myr^2)] (code units)'.",
    "failure_scenario": "",
    "repro": "Documentation change only.",
    "confidence": "high",
    "lenses": ["A", "B"],
    "divergence": "none",
    "status": "corroborated",
    "source_ids": ["S5a-B-07"]
  },
  {
    "id": "S5a-R-32",
    "file": "trinity/phase1b_energy_implicit/get_betadelta.py",
    "line": 856,
    "class": "other",
    "severity": "S4",
    "claim": "The 'iterations' field means four different things depending on which candidate wins: 0 for the input guess, a successful-evaluation count for the grid (failed points are not counted), scipy's nit for L-BFGS-B, and a total function-evaluation count for hybr.",
    "evidence": "A: iterations=best_iterations sourced from candidates appended at 733 (0), 747 (n_evals from _solve_grid) and 787 (result.nit); line 1007 passes state['n'], incremented once per residual evaluation.",
    "expected": "One definition, or separate fields for evaluations and iterations.",
    "failure_scenario": "",
    "repro": "Compare result.iterations across the three winning branches for the same config; the numbers are not on a common scale.",
    "confidence": "high",
    "lenses": ["A"],
    "divergence": "none",
    "status": "single-lens",
    "source_ids": ["S5a-A-19"]
  },
  {
    "id": "S5a-R-33",
    "file": "trinity/phase1b_energy_implicit/get_betadelta.py",
    "line": 469,
    "class": "state",
    "severity": "S4",
    "claim": "bubble_Leak is read via getattr(params.get('bubble_Leak', None), 'value', 0.0), so if the key holds a plain float rather than a .value-bearing object the leak luminosity is silently replaced by 0.0 — the only parameter in the block that degrades silently rather than raising.",
    "evidence": "A: lines 469-471 and the duplicate at 573-575, against params['X'].value for every neighbouring parameter (442-451). C independently requires that a view's get(key, default) must not silently supply a default for a required physics key, since a zeroed driver produces a smooth, plausible, wrong trajectory; A's audit shows that concern applies to exactly this one key.",
    "expected": "Read it the same way as the neighbouring parameters, or make the two failure modes (key absent vs wrong container type) explicit.",
    "failure_scenario": "A code path that stores bubble_Leak as a bare float drops the leak term from Edot_from_balance entirely, shifting the beta root with no diagnostic.",
    "repro": "get_residual_pure with params['bubble_Leak'] = 1.0e3 (a float): L_loss silently omits it.",
    "confidence": "low",
    "lenses": ["A", "C"],
    "divergence": "none",
    "status": "corroborated",
    "source_ids": ["S5a-A-16", "S5a-C-24"]
  },
  {
    "id": "S5a-R-34",
    "file": "trinity/phase1b_energy_implicit/get_betadelta.py",
    "line": 59,
    "class": "other",
    "severity": "S4",
    "claim": "Three quantitative empirical claims in the prose — '~25-100x faster per evaluation', 'residuals grow by roughly 3x per subsequent segment', '~50 expensive function evaluations' — carry no cited measurement, committed CSV/figure or named config, yet the tuning of GRID_EARLY_EXIT_RESIDUAL rests on the 3x figure. Separately, three different baselines are used for 'identical': 'byte-identical to the pre-switch behaviour', 'matching original get_betadelta.py', and 'identical to the original semantics' (applied only to the no-early-exit grid path).",
    "evidence": "B, from prose alone (':3', ':52', ':61', ':631', ':55', ':66', ':1017'). CLAUDE.md rule 5 requires exactly this class of measurement to be persisted as a committed diagnostic with the exact config and command; and the early-exit path is by construction not bit-identical to a full-grid global best.",
    "expected": "Persist the measurements (or cite the existing artifact), and name one baseline per equivalence claim with an explicit statement that the early-exit path is behaviourally equivalent but not bit-identical.",
    "failure_scenario": "The 3x growth rate is regime-dependent; where it is larger, GRID_EARLY_EXIT_RESIDUAL no longer guarantees the next segment short-circuits and the claimed saving inverts. Separately, a future session takes 'byte-identical' at face value and skips the full-run equivalence gate when touching the legacy path.",
    "repro": "Instrument input-guess residual per segment across param/simple_cluster.param and the f1edge configs and fit the growth factor; commit the CSV.",
    "confidence": "medium",
    "lenses": ["B"],
    "divergence": "none",
    "status": "single-lens",
    "source_ids": ["S5a-B-20", "S5a-B-24"]
  }
]
```
