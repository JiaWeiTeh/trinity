# PdV-in-the-transition-trigger — argument, evidence, and a plan to test it

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
> several living docs for its workstream (its `PLAN.md`, `FINDINGS.md`, `runs/README.md`, `NOTE_PATCHES.md`,
> and any other notes in the same folder). They drift out of sync *with each other* as fast as they drift
> from the code. Any agent or person editing one MUST, as part of the visit, circle back through the
> siblings and reconcile: if a number, status, claim, or line reference here contradicts a sibling — or a
> sibling has gone stale — fix it (or flag it, dated) so no two docs in the workstream disagree. Never
> update one in isolation.

## Re-entry ledger — open this FIRST (the 🔄 banner, operationalized)

The recheck list the banners demand. **Every visit:** re-verify the anchors below, update the ledger,
*then* read on. All findings here are **already persisted** (CSVs + figures under `data/` and this
folder) — do **not** re-run the hours-long sims to recover them; reproduce only to extend.

### ⭐ Current synthesis — the GOAL and "the merge" (read this first; 2026-06-26)

**The goal (north star, maintainer-stated):** modify the cooling so this 1D sim has **enhanced cooling
comparable to observations and 3D simulations**, and **somewhat dependent on cloud/cluster/bubble
properties** — i.e. raise the loss fraction `θ = L_cool/L_mech` from the 1D-resolved **0.25 (diffuse) → 0.70
(dense)** at blowout toward the obs/3D values (Lancaster ≈ **0.9–0.99**; El-Badry `θ(n_H, λδv)`), **density-
dependently**.

**The merge (current understanding — supersedes the earlier "κ_eff endgame / evaporation-decoupling"
framing):**
| role | what | status |
|---|---|---|
| **Mechanism** | **κ_eff** = `cooling_boost_kappa` (Rung A) — enhances conduction ⇒ more ~10⁵ K radiating gas ⇒ raises **emergent** cooling in-structure (θ comes out, not imposed) | **built, gated, byte-identical-off**; measured `bubble_LTotal` ×1.23–1.38 at f_κ=2 |
| **Target** | **θ(n_H)** from El-Badry (`λδv`=κ_eff, a *set* 1D knob) + Lancaster (3D, parameter-free ≈0.9–0.99) | the calibration data |
| **Knob** | **f_κ(properties)** tuned so emergent θ → target, density-dependently | the remaining work = **calibration** |

- **`θ_target` vs κ_eff was a FALSE dichotomy** — `θ(n_H)` is the *target*, κ_eff is the *mechanism* of the
  same knob. (`RUNGB_SCOPING.md` §2a is the canonical θ/`λδv`/`f_κ`/0.95 reconciliation.)
- **Evaporation-decoupling (the old "Rung B endgame") is DEMOTED to an optional high-fidelity bonus.** The
  1D `dMdt` is anchored at the 3×10⁴ K front, so it *resists* El-Badry-style evaporation suppression — but
  that suppression is **not in the goal**. *(Update 2026-06-29: this demotion is specifically of the
  **evaporation-suppression** aspect of Rung B. The **κ_mix conductivity term** — the other aspect — is
  **RE-PROMOTED** to the faithful cooling fix for the diffuse end; see the 06-29 ledger entry + §13. The scalar
  Rung-A `f_κ` cannot represent cool-layer mixing, so κ_mix(λδv) is the physical mechanism after all.)* `FM1`/`FM1b` (`data/fm1*_*.py`) are **useful negative results**
  that ruled out the wrong knobs (imposing `dMdt`; an interior loss-integrand term) and point **back to
  κ_eff** as the mechanism.
- **REFINED GOAL (2026-06-29): a *physically-bounded* f_κ(n) prescription, not f_κ tuned to force every cloud
  to fire.** The 819-sweep showed f_κ-to-fire ∝ n^−0.6 (falls) and a diffuse/high-sfe corner that never fires
  even at f_κ=64. But f_κ=64 is unphysical (it over-conducts the hot interior), and the *physical* enhancement
  (El-Badry κ_mix ∝ n) **rises** with density — opposite sign. So the honest target is: set f_κ(n) to a
  physically-motivated, capped value, and **accept that clouds which can't reach θ=0.95 stay energy-driven**
  ("not meant to be"). That predicts a falsifiable **critical column** for the energy→momentum split
  (N_crit≈1–4×10²³ at f_max≈2–8), to test against obs — vs the alternative of adding the structural κ_mix
  (Rung B) if Lancaster's 3D "diffuse clouds also cool" is the truth. Full treatment: `F_KAPPA_FUNCTIONAL_FORM.md`
  §11–§12. The calibration history below stands as the road to this reframing.
- **Remaining work = calibration of f_κ(properties) to obs/3D θ(n_H), reusing the existing knob — no new
  production code required for the calibration itself.** First cut **DONE** (`make_fkappa_leverage.py`): κ_eff
  has the leverage (`L_cool ∝ f_κ^0.63`, viable to f_κ=64). Calibration **estimate DONE**
  (`make_kappa_calibration_estimate.py`): `f_κ(n_H)` density-dependent — diffuse ≈8, dense ≈1.6 (for θ≈0.95).
  Two-anchor full-run grid **DONE** (`make_kappa_blowout_calibration.py`, 06-26 ledger): the estimate was
  **optimistic** — compact fires cooling at **f_κ≈4**, diffuse needs **≈60** (the developed-epoch leverage is
  weaker than the snapshot, exponent ~0.3–0.4). PdV-in-the-trigger probed (`make_ebpeak_trigger_test.py`,
  06-28 ledger): `ebpeak` is an assist, not a substitute — it does not remove the need for the boost. Next: a
  denser n_H full-run grid to pin `f_κ(n_H)`, then wire `cooling_boost_kappa` as an optional density-dependent
  `f_κ(n_H)` mode (gated, default-off byte-identical).

**Status ledger (newest first):**
- **2026-06-29 (maintainer MANUSCRIPT DRAFT verified + folded; new doc `KMIX_DIFFUSIVITY.md`).** Line-by-line
  checked a 2-section LaTeX draft ("A functional form for the conduction multiplier" + "Where the mixing
  diffusivity comes from") + claims table against our committed results. **~90% matches** (f_mix=f_κ^q with
  q≈0.3–0.4 < the El-Badry 1/2; `f_κ(n)=[θ_target/θ_0]^(1/q)` ≈4 compact/≈60 diffuse; θ/(1−θ)=1.6√n folding the
  11/5 = our 3.5 form; the degeneracy; f_κ=60⇒κ_mix). **Three flags:** (i) the draft's "single-variable sweep, not
  yet run" is **STALE** — we ran the 819-combo sweep and it **fanned out** (multi-dimensional), so its open question
  is answered; (ii) the eddy-turnover closure (ω=δv/λ replacing the SN cadence for continuous winds) is
  **heuristic** — it pins the contrast (≈40), not λ; the *conclusion* (λ sub-pc, calibrate not compute) is the
  keeper; (iii) **route a vs b unresolved** — draft leans diffuse→energy-driven (bounded physical diffusivity),
  §13 leans diffuse-under-cooled→κ_mix; the κ_mix implementation calibrated to Lancaster + tested on all 8 configs
  decides it. **Adopted refinement:** do **not** import El-Badry's λδv∈[1,10] (doubly off-regime: discrete-SN +
  ISM density); use El-Badry for *mechanism*, δv from v_rel, and **pin λ by calibrating κ_mix to Lancaster's
  θ~0.9–0.99** (the cadence-free magnitude anchor). Folded into `F_KAPPA_FUNCTIONAL_FORM.md` §13.
- **2026-06-29 (PHYSICAL PRESCRIPTION DERIVED → it's κ_mix(λδv), Rung B RE-PROMOTED).** Followed the
  "negative power isn't physical" thread to its end (`F_KAPPA_FUNCTIONAL_FORM.md` §13; builder
  `data/make_fkappa_physical_derivation.py` → `data/fkappa_physical_derivation.csv` + `fkappa_physical_derivation.png`).
  Three distinct f_κ(n): **mechanism** κ_mix/κ_Spitzer ∝ n (RISES, the physical enhancement); **target**
  θ*(n;λδv) Eq37/38 (flat-high 0.94–0.999); **boost** to reach the target ∝ n^−0.6 (FALLS — a boost factor, NOT a
  conductivity). Key results: (1) crossover κ_mix=κ_Spitzer at **n_crit=0.25 cm⁻³** (T=2e5, λδv=1) — matches
  El-Badry's "n≳0.2"; (2) a **scalar f_κ can't represent the mechanism** — in the cool layer (T~2e4)
  κ_mix/κ_Spitzer≈10³–10⁷ because Spitzer∝T^(5/2) vanishes → the faithful form is the **structural κ_mix term**,
  λδv∈[1,10] pc·km/s the single parameter, saturation-capped ∝n; (3) **course-correction on the "accept
  non-transition" idea** — El-Badry's verified θ* is flat-high *even at diffuse* (0.94 at n=1e2 vs 1D baseline
  0.29, gap 0.65), so the diffuse never-fire is most likely a **1D under-cooling artifact** → route (b) κ_mix, not
  route (a) accept. This **re-promotes Rung B** from "optional fidelity bonus" to the faithful fix, and
  reconciles RUNGB_SCOPING's κ_mix-magnitude absurdity (κ_mix/κ_S≈10²⁴ came from D_turb=R2·v2; use λδv∈[1,10]
  instead → sane n_crit≈0.2). The "derived number" asked for = **λδv ∈ [1,10] pc·km/s**, not an f_max/power law.
  Next concrete step: **wire the gated κ_mix mode** (RUNGB_SCOPING §8 front-conduction intervention), default-off
  byte-identical. Documented in §13 + storyline §15.7.
- **2026-06-29 (STRATEGY REFINEMENT — physically-bounded f_κ, accept non-transition; the "don't force it"
  reframing).** Prompted by the maintainer: searching f_κ up to 64 to force every cloud to fire assumes every
  cloud must be momentum-driven, which isn't physical. Two facts (new doc sections `F_KAPPA_FUNCTIONAL_FORM.md`
  §11–§12; builder `data/make_fkappa_physical_cap.py` → `data/fkappa_physical_cap.csv` + `fkappa_physical_cap.png`):
  **(1) the sign flip** — El-Badry's κ_mix=(λδv)·n·k_B/(μm_p) ∝ n while κ_Spitzer ∝ T^(5/2), so the *physical*
  f_κ **rises** with density (∝ n^+1), OPPOSITE to the empirical fire-threshold (∝ n^−0.6). Using the −0.6 as a
  prescription gives diffuse clouds the *most* boost = the forcing we want to avoid; the physical (rising)
  prescription gives diffuse the *least* → dense transition, diffuse stay energy-driven. **(2) the physical-cap
  experiment** (pure re-analysis of `summary.csv`, no sims): cap the enhancement at f_max; a cloud is
  momentum-driven iff f_κ_fire ≤ f_max, else energy-driven. f_max≈2–8 ⇒ a **falsifiable critical column
  N_crit≈1–4×10²³ cm⁻²** (soft boundary; 6/63 never fire under any cap). **Open tension kept:** Lancaster 3D says
  even diffuse clouds cool (so non-transition might be 1D under-cooling, not truth) — route (a) accept
  non-transition vs (b) add κ_mix (Rung B) settled against obs, not asserted. **Sweep design (answered):** most
  prescriptions are testable by re-analysis of the existing grid (free); a new sweep is only needed for f_κ<1
  (suppression) or to verify a chosen prescription as real runs (a 63-run generator that sets
  cooling_boost_kappa=clamp(A·nCore^q, fmin, f_max), vs the 819 of the free scan). This **refines** the
  ⭐ synthesis: the goal is a *physically-bounded* f_κ(n) prescription, not f_κ cranked to fire every cloud.
- **2026-06-29 (sweep follow-up — the fan-out's anatomy + the metric, documented).** Merged main's 3-panel
  faceted `fkappa_nH_sweep.png` (by sfe) + raw `summary.csv` into the branch. Then dissected the fan-out
  (`data/make_fkappa_cliff_metric.py` → `data/fkappa_cliff_metric.csv` + `fkappa_cliff_metric.png`):
  **(A) the 1e7 "broken power law" = a catastrophic-cooling CLIFF** — θ@f_κ=1 jumps past 0.95 (fires with no
  boost) at lower density for more massive clouds (1e7 at n≈3e3 vs 1e5 at n≈2e4), because a bigger cloud sweeps
  the same **column** at lower density; the cliff is ≈ a constant-column threshold (nCore spread ×11 → column
  spread ×5.7; median cliff column ≈8×10²³). **(B) the fan-out is genuinely multi-dimensional** — nCore is the
  best single predictor of θ@f_κ=1 (R²=0.73), column slightly worse globally (0.71) though it nails the cliff,
  2-var(nCore,rCloud)=0.75 (coef ratio 2:1, not pure column); f_κ_fire is **independent of cluster mass**
  M★=sfe·mCloud (R²=0.002, as expected since θ is L_mech-normalised). **(C) the metric is sound** — θ=
  L_cool/L_mech sampled at blowout (R2>rCloud), firing on theta_max≥0.95; snapshot-vs-peak barely matters
  (median theta_max−theta_blowout=0.004, >0.05 in 5/63); regimes split 403 cooled-before-escape / 416 escaped.
  One fixable imprecision: theta_max isn't capped at blowout_t (post-escape peak can falsely tag "fired
  in-cloud", ~5 cells; needs the jsonl to fix). Documented in `F_KAPPA_FUNCTIONAL_FORM.md` §9–§10 + the HTML
  storyline. NOT confirmed: PdV as the cliff driver (the firing metric is radiative θ; would need PdV-logged runs).
- **2026-06-29 (819-combo f_κ(n_H) SWEEP RAN ON HELIX — results in, predictions scored).** The controlled grid
  (7 nCore × 3 mCloud × 3 sfe = 63 cells × 13 f_κ) ran; reduced to `data/fkappa_nH_sweep.csv`. Scored against
  the form's pre-registered predictions (`data/make_fkappa_sweep_analysis.py` → `data/fkappa_sweep_scorecard.csv`,
  `fkappa_sweep_analysis.png`): **P2 de-conflation = FAN-OUT confirmed** (×2–32 spread across mCloud/sfe at fixed
  n ⇒ f_κ is multi-dimensional, NOT f(n_H) alone) ✅; **diffuse→κ_mix confirmed** (6/63 low-n high-sfe cells
  never fire even at f_κ=64) ✅; **P1 slope WRONG** — measured **f_κ_fire ≈ 1.0×10³·n_core^(−0.60)**, vs the
  predicted n^(−0.30) (2× too shallow) ❌; **P3 root cause** — the 6-anchor baseline θ₀(n) was undersampled
  (0.41/dex) vs the real grid (**1.13/dex**) ❌; **P4 leverage** median 0.21 vs predicted 0.31 ⚠. Net: the
  *qualitative* conclusions (steep decline, multi-dimensional, diffuse-corner-needs-κ_mix) held and are now
  measured; the *slope magnitude* was off because of the baseline. **Closes the §3.1 OPEN sweep task.** Next:
  (a) regress measured f_κ_fire on (n_core, mCloud, sfe) for the second axis; (b) spec the gated El-Badry κ_mix
  mode for the never-fire corner. Doc: `F_KAPPA_FUNCTIONAL_FORM.md` §0 banner + new §8 scorecard.
- **2026-06-29 (El-Badry+2019 §3.1/§5.2 VERIFIED from the maintainer-supplied PDF).** The PDF (pp.5–6,13,15)
  confirms line-by-line: Eq 16 (Spitzer C=6e-7·T^(5/2), density-weak), Eq 17 (Parker), Spitzer↔Parker cross at
  6.6×10⁴ K, Eq 19/20 (saturation, q_sat=(3/2)ρc_s,iso³ = 5φρc³ with φ=0.3), **Eq 21 κ_mix=(λδv)ρk_B/μm_p**
  (temperature-INDEPENDENT; κ=max(κ_mix,κ_Spitzer); κ_mix dominates T≲2×10⁵K, n_H≳0.2), **Eq 35 (11/5)θ/(1−θ)**,
  **Eq 37 ψ=A_mix√(λδv·n_H), A_mix≈1.7 analytic / 3.5 fit**, **Eq 38 θ=ψ/(11/5+ψ)** (θ time-independent, depends
  on ρ₀ not Δt_SNe). El-Badry **themselves** propose calibrating λδv to 3D cooling rates (p6) = this workstream's
  strategy. ⇒ The earlier in-container `[unverified]`/`[schematic/to-verify]` hedges on El-Badry's algebra are
  **RETRACTED** (a 403 access gap, not an error; the prior room's transcription was right). Folded the verified
  θ_EB(n_H,λδv) into `make_fkappa_functional_form.py` as a target cross-check: it saturates to 0.94–0.999 in the
  GMC range (matching Lancaster's plateau), giving f_κ≈46/11/3.6 (diffuse/mid/dense), within ~15% of the
  Lancaster-θ*=0.95 numbers — so the functional form is robust to the target choice. Updated FINDINGS §-pointer,
  this ledger, and `F_KAPPA_FUNCTIONAL_FORM.md` §0/§2.1/§3/§5/§7.
- **2026-06-29 (f_κ(n_H) FUNCTIONAL FORM — composed closed form, while the 819-sweep is pending; new doc
  `F_KAPPA_FUNCTIONAL_FORM.md`).** Answered the maintainer's "give me a usable f_κ(n_H) from literature/other
  quantities, don't wait for the sweep to fit one cold." Result: **f_κ(n_H) = (θ*/θ₀(n_H))^(1/p) ≈
  1.4×10²·n_H^(−0.30)** (θ*=0.90). Composed from three separable, independently-checkable pieces:
  **(1) target** θ*≈0.90 = the **Lancaster 2021 plateau** (verbatim: "generic over more than three orders of
  magnitude in density" ⇒ density-INDEPENDENT target — *not* a rising El-Badry √n curve, which is unverified
  and an n~0.1–10 SN-superbubble regime anyway); **(2) baseline** `logit θ₀ = −1.73 + 0.41 log₁₀ n_H` (fit, 6
  anchors; the density structure of f_κ comes from THIS rising baseline under a flat target — which dissolves
  the §2a "flat target == 0.95 trigger" worry, because that equivalence only holds for the *linear* f_mix knob,
  not the *structural* f_κ); **(3) leverage** p≈0.31 measured as the raw power-law exponent over the FULL range
  to firing. ⚠️ **Self-correction (same day, prompted by external review):** the first cut inverted leverage in
  **logit/odds space** (q≈0.55) and got f_κ≈291 (diffuse)…121 (compact) — **wrong by ~10–30×** at the one
  *measured* anchor (compact **fires at f_κ≈3.4**, not ~120). Cause: θ(f_κ) **accelerates toward firing**
  (convex: compact 0.667→0.739→1.024), because the bubble transitions *before* θ saturates, so a saturating
  logit extrapolated from f_κ∈{1,2} overshoots. Raw-power p (0.31/0.21/0.42, full-range) reproduces the measured
  anchor and matches the El-Badry-back-reaction estimate q=ln1.3/ln2≈0.4. Only the **amplitude** changed (~10×
  lower); the **slope (−0.30) is robust**. Corrected numbers: f_κ≈48(diffuse)/9(mid)/3(dense) for θ*=0.95.
  **Literature verdict: there is NO published `f_κ ∝ n_H^p` law** (10-subagent survey;
  all PDFs 403-blocked, so eq.#s unverified) — classical Spitzer is n_H⁰, the only effective-κ density power is
  the *saturated* branch κ_sat∝n_H¹ (the CEILING, rising), and Lancaster Θ is density-independent. The
  **physical bracket**: required f_κ falls (∝n_H^−0.30) while the saturation ceiling rises (∝n_H¹) → they cross;
  the **diffuse end (f_κ~tens) is likely unreachable by Spitzer boost** and needs El-Badry's temperature-
  independent κ_mix — quantifies the Rung-A/Rung-B boundary. Artifacts: `data/make_fkappa_functional_form.py`
  → `data/fkappa_functional_form.csv` + `fkappa_functional_form.png` (reads committed CSVs, no sims). The
  819-sweep now has a concrete job: **measure q(n_H[,mCloud,SFE]) and re-fit θ₀** to confirm/refine this curve.
  Also corrected the El-Badry citation everywhere it was wrong (MNRAS 490,1961 / Weisz / 1902.09547 — not ApJ
  879 / not Weinberg) and flagged that this branch lacks the external handoff's `3e68143` El-Badry-overlay work.
- **2026-06-29 (Helix run scaffolding for the 819-combo sweep — committed).** The sweep was "HPC-ready"
  only via a bare `--emit-jobs jobs/` + `sbatch jobs/submit_sweep.sbatch`, which would have **failed on
  Helix**: outputs resolve under the read-only `/home` repo checkout, and the emitted sbatch leaves
  account/partition/`--export=NONE`/conda unset (the maintainer's per-cluster `patch_sbatch` step, which
  `sweep_fkappa_nH.param` didn't mention). Added committed, pre-patched `runs/run_fkappa.sbatch` (array
  1-819, cpu-single/bw22J006/`--export=NONE`/conda, reads the emit `runs.tsv`) + `runs/sync.sh` driver,
  mirroring `paper/shellSSC6` but as an array; the driver emits the bundle from `/gpfs` so `path2output`
  lands on the writable workspace. **Adopted the II-survey reduce-then-plot split** (merging the parallel
  `2dcfc9e` work): stdlib-only `data/reduce_fkappa_sweep.py` walks the multi-GB jsonl once on the cluster →
  tiny `summary.csv`; `data/make_fkappa_nH_sweep.py` now reads ONLY that CSV (fit + de-conflation figure on
  the laptop). `sync.sh` = submit/watch/collect/**reduce**/down; reducer selftests streaming θ vs the proven
  `harvest()`, plotter selftests `fit_fire`. Reconciled the collision the two parallel efforts left in the
  plotter — removed an undefined-`_DEFAULT_SUMMARY` crash (no-arg run) and a dead jsonl/`FKAPPA_SWEEP_OUT`
  guard that aborted the laptop step even with a valid `summary.csv`. `REPRODUCE.md` #18/Block C +
  `sweep_fkappa_nH.param` header reconciled. No production code touched. **NEXT: run the grid** (`sync.sh submit`).
- **2026-06-28 (controlled f_κ(n_H) calibration sweep — built, HPC-ready, not yet run; broadened to 819).**
  The clean replacement for the conflated 3-anchor estimate (compact/mid/diffuse vary mCloud+sfe+nCore
  together). `runs/params/sweep_fkappa_nH.param` sweeps **nCore [1e2,3e2,1e3,3e3,1e4,3e4,1e5] (primary, fine) ×
  cooling_boost_kappa [1,1.5,2,3,4,6,8,12,16,24,32,48,64] (fine — brackets θ→0.95 firing precisely, not
  extrapolated) × mCloud [1e5,1e6,1e7] × sfe [0.03,0.1,0.3] = 819 combos** (HPC; under the maintainer's 1000
  ceiling). The mCloud/sfe axes are a **de-conflation test**: do the series collapse onto one n_H curve
  (⇒ clean f_κ(n_H)) or spread (⇒ multi-dimensional)? Verified: `--dry-run` expands to 819, `--emit-jobs`
  gives a working SLURM array (`--array=1-819`), and the whole grid stays < the 200 pc `rCloud_max` (max
  mCloud 1e7 × nCore 1e2 ≈ 70–85 pc; diffuse extreme nCore 1e2 = 39.6 pc). nCore capped at 1e5 (1e6 is the
  stiff corner, result #15). Harvest+fit harness `data/make_fkappa_nH_sweep.py` (reuses proven `harvest()`;
  4-axis run-name parser self-tested; groups by (mCloud,sfe,nCore) cell, fits θ∝f_κ^p → f_κ_fire, overlays the
  M_cl/sfe series; graceful "no data yet") → `fkappa_nH_sweep.csv` + `.png`. Registered in `REPRODUCE.md`
  (#18 / Block C). **NEXT: run the 819-combo grid on HPC** → first clean f_κ(n_H) + de-conflation verdict.
  No production code touched.
- **2026-06-28 (paper reproducibility manifest — `REPRODUCE.md`).** Created `REPRODUCE.md` at the maintainer's
  request: a single map from **every storyline result** (the figures/numbers in `pdvtrigger_report.html`) to
  **the exact `.param` + run command + derived artifact**, tagged 🟢 cheap (re-reads a committed CSV in seconds)
  / 🟡 a few full runs / 🔴 grid-HPC. Includes the two expensive blocks' exact commands, a "rebuild all figures
  with no sims" loop (every figure is a pure read of a committed CSV, since `outputs/` is git-ignored), and the
  gated-knob table. Verified every referenced harness + `.param` resolves. So a future paper write-up can
  re-run any piece and prove the storyline is reproducible. No production code touched.
- **2026-06-28 (dense-edge stiffness diagnosed — NOT f_κ; it's an extreme-density solver-stiffness cost).**
  Ran the #1 de-risk experiment: `small_dense_highsfe` (nCore 1e6) at **f_κ=1 BASELINE** (default everything),
  hybr vs legacy head-to-head (`runs/params/diag_dense_{hybr,legacy}.param`, `data/dense_stiffness_diag.csv`).
  **Answer to "is the dense-edge hang f_κ-driven?": NO** — f_κ=1 baseline is just as slow, so the cooling boost
  is NOT the cause; the f_κ(n_H) calibration is not blocked by κ_eff. **What it actually is:** at this extreme
  density the implicit bubble-structure solve is **pathologically slow** (Pb≈10¹⁰; minutes per stiff segment
  past cloud dispersal) for **both** solvers — hybr reached t=0.050 / legacy t=0.004 in ~11 min wall, neither
  finishing. **Honest correction:** mid-experiment I hypothesized "hybr HARD-stalls" — wrong; hybr broke through
  the t=0.0132 wall after ~4.5 min and was actually *ahead* of legacy. So it is **slowness, not a hard hang,
  and not clearly solver-specific.** Oddity: the committed cleanroom legacy data (2026-06-21) *completed* this
  config to t=6 Myr (265 rows) — far faster than live legacy now ⇒ **possible slowdown regression since then,
  UNVERIFIED.** **Actionable:** the calibration doesn't need the nCore 1e6 corner (extreme/borderline-unphysical,
  rCore 0.1 pc); hybr runs fine at nCore≤1e5 (compact 1e5, mid 1e4 both completed quickly), so pin f_κ(n_H)
  over the physical range and flag the extreme-density slowness as a **separate perf/solver item** (chase the
  possible regression only if that corner is ever needed). No production code touched.
- **2026-06-28 (taxonomy table — disambiguating the approaches).** Added a physics taxonomy (report §14 +
  `FINDINGS.md` "Taxonomy" section) after a maintainer asked what is what. Resolves a real conflation: the
  "three things" are really **2 cooling-magnitude approaches on opposite sides of the structure solve + 1
  trigger axis**. **A (outcome-side):** `cooling_boost_mode` = `multiplier` (L_loss=L_leak+f_mix·L_cool) or
  `theta_target` (L_loss=max(L_cool+L_leak, θ·L_mech) — the **Lancaster-θ floor**, imposed), at
  `get_betadelta.py:354/356`. **B (mechanism-side):** `cooling_boost_kappa`=f_κ (κ_eff=f_κ·C_th·T^(5/2), 3
  sites — the **El-Badry conduction/mixing** way, θ emerges). **Key:** "El-Badry-κ" and "modify the conduction
  front k_f" are the **SAME** knob (B), not two things. **C (trigger):** `transition_trigger=ebpeak`. A and B
  must not be stacked (the max() closure keeps it single-count). No production code touched.
- **2026-06-28 (what IS f_κ? — equation-grounded definition + mid live runs + a consistent plot style).**
  Wrote the precise, code-grounded definition of `f_κ` (no assumptions; report §13 + `make_fkappa_definition.py`
  → `fkappa_definition.png`). **f_κ = `cooling_boost_kappa`** is a dimensionless multiplier on the
  **Spitzer–Härm conduction coefficient** `C_thermal = 6e-7 erg s⁻¹ cm⁻¹ K⁻⁷ᐟ²` (`registry.py:341`):
  κ_eff(T) = f_κ·C_th·T^(5/2). It enters the 3 sites in `bubble_luminosity.py` — the dMdt seed (Eq 33,
  `:291`, ⇒ **dMdt ∝ f_κ^(2/7)**), the conduction-layer ICs (Eq 44, `:370`, ⇒ layer thickness **ΔR₂ ∝ f_κ at fixed dMdt**;
  folding in the seed dMdt∝f_κ^(2/7) ⇒ ΔR₂∝f_κ^(5/7)), and the T-curvature ODE (Eq 42-43, `:406`, term ∝ 1/(f_κ·C_th·T^(5/2))). **It does NOT multiply L_cool**:
  the local `get_dudt(t,n,T,φ)` is integrated over the (thicker) structure, so θ=L_cool/L_mech EMERGES.
  **Analytic seed scaling VERIFIED vs measurement:** dMdt(f_κ=2)/dMdt(f_κ=1) = 1.2175 at the seed vs
  2^(2/7)=1.219 (≈0.1%; softens later as Pb drains ~3%). **Side effect (why it's a probe):** dMdt rises too
  (El-Badry would suppress it). **mid live runs done** (`cal_mid__ek{1,2,4}`, midrange_pl0): θ_blowout
  0.610→0.711→0.814, ebpeak fires at f_κ=4 (peak 1.027) — a 3rd calibration config, and mid live PdV-incl
  peak 0.901 == frozen 0.901 (2nd digit-perfect validation). **Calibration now 3 configs** (compact/mid/diffuse,
  `make_kappa_blowout_calibration.py`): θ(f_κ=1)=0.17/0.61/0.67, f_κ-to-fire ≈ 4 (compact, bracketed — fires
  at f_κ=4) / ~5-6 (mid, extrap.) / ~60 (diffuse, extrap.) — answering the
  user's question: **at f_κ=1 the under-cooled clouds stay below ~0.9 and never fire; they need much higher
  f_κ, steeply density-dependent.** **Plot style:** added `data/_trinity_style.py` (loads `paper/_lib/trinity.mplstyle`,
  LaTeX-free fallback — container has no system LaTeX) and applied it to all recent storyline figures for
  consistency. No production code touched.
- **2026-06-28 (does the ebpeak finding hold on the 8 configs? — frozen-screen cross-check + live validation).**
  Honest coverage answer: the recent full-run κ_eff/ebpeak work ran on **2 density-edge configs**
  (compact=`simple_cluster`, diffuse=`f1edge_lowdens`), NOT all 8. But the **f_κ=1 ebpeak conclusion
  generalizes** to the full 8-config universe via the *earlier* frozen-trajectory screen
  (`data/pdv_combined_trigger.csv` + `pdv_regime_budget.csv`), reconciled here by
  `data/make_ebpeak_8config_xcheck.py` (→ `data/ebpeak_8config_xcheck.csv` + `ebpeak_8config_xcheck.png`).
  **All 6 "normal" configs peak at PdV-inclusive 0.85–0.92 and do NOT fire** (be_sphere 0.905, midrange_pl0
  0.901, pl2_steep 0.847, simple_cluster 0.911, small_dense_highsfe 0.919, large_diffuse_lowsfe 1.019 — the
  last barely, **post-blowout**); only the **heavy 5e9** (`fail_repro`, super-critical PdV/Lmech>1, peak 1.57)
  and the **small_1e6 control** (birth blip, 1.11) fire. **Live-vs-frozen validation:** my live full-run peak
  for `simple_cluster` (0.911) matches the frozen peak (0.911) **to the digit** ⇒ the frozen screen is
  trustworthy for the other configs. **Remaining gap (live-only, can't be frozen):** the f_κ-DEPENDENCE (the
  cooling↔PdV trade-off / calibration) — freezing the trajectory hides the Eb/Pb/PdV drainage that *is* the
  trade-off. Extending it live: `runs/params/cal_{mid,dense}__ek{1,2,4}.param` (midrange_pl0 + small_dense_highsfe
  × f_κ∈{1,2,4}, ebpeak-active). `mid` running; **`dense` (nCore 1e6) stalled — pathologically stiff
  integrator**, killed, relying on its frozen point. No production code touched.
- **2026-06-28 (does PdV ALONE trigger the transition? — MEASURED on the actual code path; corrects the
  2026-06-26 optimism).** Ran the concrete test the prior entry's caveat (ii) demanded: two dedicated runs
  with `transition_trigger=cooling_balance,ebpeak` ACTIVE at f_κ=1 (`runs/params/cal_{compact,diffuse}__ebpeak.param`,
  harvested by `data/make_ebpeak_trigger_test.py` → `data/ebpeak_trigger_test.csv` + `ebpeak_trigger_test.png`).
  **Result: ebpeak does NOT fire at f_κ=1 for EITHER config** — both ran to `stop_t` and ended on
  `STOPPING_TIME` with shadow `ebpeak_t=None`. The PdV-inclusive ratio `(Lloss+PdV)/Lgain` **peaks BELOW the
  1.0 threshold, then DECLINES:** compact peaks **0.912 @t=0.12** (just past dispersal); diffuse peaks **0.862
  @t=1.06** (well past dispersal) then falls as the bubble RE-ACCELERATES in the low-density ISM (the
  diffuse__ebpeak run reached t=1.5, R2=191 pc, v2=168 km/s, **Eb still growing** → net energy never turns
  over). **This REFUTES the linear extrapolation in the prior entry that diffuse would fire ~1.2–1.3 Myr** —
  the ratio is non-monotone (both sinks shrink vs Lmech in the deep ISM). **What survives:** PdV is the
  dominant sink (PdV/Lgain = 0.20 compact / 0.46 diffuse) and lifts the balance from radiative-only (0.66 /
  0.17) up to ~0.86–0.91 — it NARROWS the gap but does not close it; a cooling boost is still required to
  trigger. **New, sharper finding — the cooling↔PdV trade-off CAPS the PdV path:** the PdV-inclusive **peak**
  is nearly f_κ-INSENSITIVE for diffuse (**0.848→0.849→0.853** across f_κ 1,2,4 — flat) while the radiative
  ratio nearly doubles (0.165→0.297). ⇒ **for diffuse the only path to fire is radiative `cooling_balance`
  (f_κ~60), NOT `ebpeak`**; PdV helps the COMPACT case (peak 0.91 at f_κ=1; `ebpeak` fires by f_κ~2–4, where
  `cooling_balance` also fires — `ebpeak` ~5 ms earlier at f_κ=4: 0.0772 vs 0.082). **Net:** "include PdV"
  (`ebpeak`) is a real ASSIST for transition TIMING (raises the diffuse floor 0.17→0.85) but is **NOT a
  substitute** for `κ_eff`; the complementary split (PdV=timing, κ_eff=cooling magnitude) stands, downgraded
  from the optimistic "PdV alone fixes the f_κ~60 problem." Opt-in dev runs; **no production code touched**
  (default `transition_trigger=cooling_balance` unchanged).
- **2026-06-26 (include PdV in the trigger? — the founding question, with fresh data).** `data/make_pdv_trigger_compare.py`
  (→ `pdv_trigger_compare.csv/png`) measures, on the cal runs at cloud dispersal, the radiative-only ratio
  (`Lcool/Lmech`, the `cooling_balance` criterion) vs the **PdV-inclusive** ratio (`(Lcool+leak+PdV)/Lmech`, the
  `ebpeak` criterion `edot_balance≤0`). **PdV is the DOMINANT sink:** PdV/Lmech = 0.21 (compact) / **0.48
  (diffuse)** vs Lcool/Lmech 0.67 / 0.17. So the **PdV-inclusive ratio is 0.65–0.91 at f_κ=1**, vs radiative-only
  0.17–0.67 — i.e. **the diffuse cloud that needs f_κ~60 on the radiative-only trigger is already ~0.65–0.85 on
  the PdV-inclusive (ebpeak) trigger with NO boost.** Two honest caveats: (i) **cooling↔PdV trade-off** — boosting
  cooling drains Eb→lowers Pb→lowers PdV, so the PdV-incl ratio rises only slowly with f_κ (diffuse 0.65→0.71),
  you can't crank cooling to push it to 1.0; (ii) the capped runs reach ~0.85–0.91 max, not quite the 1.0 ebpeak
  threshold — need to continue the run / a small extra to confirm it fires. **Reframing:** including PdV addresses
  the TRANSITION-TIMING goal (bubble goes momentum naturally — the diffuse-f_κ~60 problem was an artifact of the
  radiative-only trigger); it does NOT make cooling efficient (θ stays 0.14–0.30 vs the literature 0.9), which is
  a SEPARATE goal κ_eff still owns. So **ebpeak (PdV) for the trigger + κ_eff for the cooling magnitude are
  COMPLEMENTARY** — a cleaner split than "boost cooling until it triggers at 0.95." (This is the workstream's
  founding `PdV-trigger` question, reopened for *normal* clouds with measured data.)
  **➤ RESOLVED 2026-06-28 (see top entry):** the "continue the run to confirm it fires" of caveat (ii) was run —
  `ebpeak` does **NOT** fire at f_κ=1 for either config; the PdV-inclusive ratio peaks **below** 1.0 (compact
  0.912, diffuse 0.862) and then declines. The optimistic "diffuse is already ~0.85, nearly triggers" reading
  here is **superseded**: PdV narrows but does not close the gap, and the trade-off keeps the diffuse PdV-incl
  peak ~flat across f_κ — so PdV is an assist, not a substitute for κ_eff.
- **2026-06-26 (f_κ calibration — MEASURED, full runs) + a trigger-framing CORRECTION.** Ran the 6-sim
  grid (compact `simple_cluster` + diffuse `f1edge_lowdens` × f_κ∈{1,2,4}, ~24 min parallel;
  `data/make_kappa_blowout_calibration.py` → `kappa_blowout_calibration.csv/png`). **Correctness ✓:**
  `θ_blowout(f_κ=1)` reproduces the baselines exactly (compact 0.667, diffuse 0.169). **Result:** compact
  `θ_blowout` 0.667→0.74→**1.024** (f_κ 1,2,4) — at **f_κ=4 it crosses the 0.95 `cooling_balance` trigger →
  the run enters the momentum phase via COOLING (no geometric blowout)**; diffuse stays 0.17→0.23→0.30
  (needs `f_κ≈60`, extrapolated, at the viability edge). The **snapshot estimate was optimistic** — the
  developed-epoch leverage is weaker (exponent ~0.3–0.4 vs the snapshot 0.63), so true f_κ is 2–8× higher
  (compact ~3–4 not 1.75; diffuse ~60 not 8). Metric fix: θ peaks at cloud dispersal then DROPS in the ISM,
  so the developed value is `θ_blowout`/`θ_max`, not the last row.
  **⚠️ FRAMING CORRECTION (verified in code, propagate to FINDINGS/report):** the DEFAULT energy→momentum
  trigger is **`cooling_balance`** (Lloss/Lgain>0.95, `run_energy_implicit_phase.py:1206`; `transition_trigger`
  default `cooling_balance`, `default.param:282`) — a **cooling-driven** transition, same intent as the
  literature. `blowout` (R2>rCloud) is **opt-in, default OFF**. So the earlier "blowout is the transition
  trigger for normal clouds" was a **mischaracterization**: blowout/cloud-dispersal is the *fallback symptom*
  when the 1D cooling is too weak for `cooling_balance` to fire. **The real job of κ_eff is to make the
  cooling-driven `cooling_balance` transition fire** (θ→0.95) for under-cooled clouds — exactly the
  Lancaster/El-Badry/Gronke "cooling creeps up → momentum naturally" picture.
- **2026-06-26 (f_κ(n_H) calibration — the estimate; full-run grid is HPC-only) — the merge's payoff curve.**
  Attempted the full-run blowout-θ grid but a single sim to blowout is **~90 min (compact) → ~hours (diffuse)**
  — the energy phase runs a fine time grid (smoke run reached only t=0.0027/0.109 Myr in 139 s). So the full
  grid is **HPC-only**; the params (`runs/params/cal_{compact,diffuse}__k{1,2,4}.param`) + harvester
  (`data/make_kappa_blowout_calibration.py`) are committed and ready for it. In-session, combined the two
  verified ingredients — the leverage `L_cool ∝ f_κ^0.63` (`fkappa_leverage.csv`) and the resolved baseline
  `θ(n_H)` at blowout (`fmix_table.csv` + `da_replay.csv` nCore) — into the calibration **estimate**
  `θ(f_κ,n_H) ≈ min(0.99, θ_base·f_κ^0.63)` ⇒ `f_κ_needed = (θ_target/θ_base)^{1/0.63}`. Result
  (`data/make_kappa_calibration_estimate.py`, `kappa_calibration_estimate.csv/png`): **f_κ(n_H) is
  density-dependent — diffuse (θ_base 0.25) needs `f_κ≈8`, dense (θ_base 0.70) needs `f_κ≈1.6`** to reach
  θ≈0.95 (well inside the viable range, ≤64). **Caveat (kept):** the leverage was measured on early snapshots
  (θ≈0.01) far from the θ→1 ceiling, so near the target it **saturates** — the true `f_κ` is ≥ this estimate
  (optimistic). The full-run grid would replace the estimate with a measurement. **This is the merge delivered:
  a density-dependent cooling enhancement via the existing gated knob, calibrated (estimated) to obs/3D.**
- **2026-06-26 (f_κ calibration — first cut) — κ_eff has the leverage AND stays viable; the merge path is
  feasible. No production edit (uses the gated knob).** `data/make_fkappa_leverage.py` sweeps the real
  `cooling_boost_kappa` (`f_κ ∈ {1..64}`) through the full `get_bubbleproperties_pure` on the two captured
  states (f_κ=1 recovers the converged `dMdt` — correctness check). Result (`data/fkappa_leverage.csv`,
  `fkappa_leverage.png`): **`L_cool` scales ∝ f_κ^0.6** (×1.5 at f_κ=2, ×2.3 at 4, ×3.4–4.0 at 8, ×11–16 at
  64), so the **target enhancement ×1.3–3.6** (lift blowout θ 0.25–0.70 → ~0.9) is reached at **f_κ ≈ 2–8**;
  the solve stays **healthy to f_κ=64** (no viability ceiling found) with `dMdt` rising only ∝ f_κ^0.28 (×3.3 at
  64) — so `L_cool` *outpaces* evaporation (the ratio improves with f_κ). Both states behave near-identically.
  **Caveat (honest):** this is the SNAPSHOT leverage on early bubbles (θ_snap ~0.01), not the absolute
  blowout-θ. **Next:** full-run blowout-θ calibration across a density grid (vary `cooling_boost_kappa`, measure
  θ at blowout) to pin `f_κ(n_H)` against the `θ(n_H)` target — the leverage shape + viability here say it is
  feasible.
- **2026-06-26 (the merge) — reframed around the GOAL; κ_eff recognized as the cooling MECHANISM, evaporation-
  decoupling demoted to a fidelity bonus.** Critical re-think (maintainer steer): the goal is *enhanced,
  density-dependent cooling matched to obs/3D*, **not** evaporation suppression. κ_eff (`cooling_boost_kappa`,
  Rung A, already built) **is** the in-structure cooling mechanism — it raised `bubble_LTotal` ×1.23–1.38. The
  `θ_target`-vs-κ_eff split was a false dichotomy: `θ(n_H)` (El-Badry `λδv`=κ_eff + Lancaster) is the *target*,
  κ_eff is the *mechanism*, `f_κ(properties)` is the knob to calibrate. `FM1`/`FM1b` are negative results that
  ruled out the wrong knobs and point back to κ_eff. **Remaining work = f_κ calibration** (reuses the existing
  knob; no new production code). All workstream docs + the storyline reframed to lead with this (see
  ⭐ synthesis block above). Next: offline `cooling_boost_kappa` sweep → emergent-θ response + viability.
- **2026-06-26 (FM1b) — second offline prototype: in-structure interface cooling lowers `dMdt` (El-Badry sign
  ✓) but negligibly. No code touched.** `data/make_fm1b_evapsign.py` monkeypatches `net_coolingcurve.get_dudt`
  to add localized ~10⁵ K cooling (`×(1+A·gaussian)`) and runs the **full** `get_bubbleproperties_pure` on the
  two captured stiff states (`A=0` recovers the converged `dMdt` — correctness check). Result
  (`data/fm1b_evapsign.csv`, `fm1b_evapsign.png`): `dMdt` decreases **monotonically** with injected cooling in
  both states (above the fsolve noise floor) — the **El-Badry sign**, *not* Rung-A re-coupling — **but the
  magnitude is negligible**: `−0.10%` (stiff) / `−0.03%` (mild) at a 5× cooling boost, vs the El-Badry target of
  `−67…−97%` (3–30×). Reason: `dMdt` is anchored at the 3×10⁴ K **front**, so 10⁵ K **interior** cooling barely
  couples to it (the FM1 lesson again). **Convergent narrowing:** FM1 (impose `dMdt`) and FM1b (interior cooling)
  both fail because `dMdt` is a *front* quantity ⇒ the next prototype must perturb the **front conduction**
  itself (a faithful `κ_eff`/`λδv` acting at the front, not an interior loss term). Risk #2 + any gated code come
  only after a path clears that. Details: `RUNGB_SCOPING.md` §8/§9 (FM1b ◐).
- **2026-06-26 (literature anchor) — θ / λδv / f_mix / 0.95 roles pinned vs El-Badry & Lancaster
  (`RUNGB_SCOPING.md` §2a, verified).** θ (loss fraction) is **measured** in both papers; **λδv** (= `κ_eff`
  mixing diffusivity = our `κ_mix`) is the **set** knob, only in 1D (El-Badry "arbitrary parameter, range
  explored"; Lancaster's 3D needs none); **0.95** is a **threshold on measured θ**. So the genuine El-Badry
  analog is the **structural `κ_mix` (Rung B)**, not the scalar `f_mix` (degenerate) — and the calibration
  target is **density-dependent θ(n_H)**, not a flat 0.9–0.99 (which would over-cool diffuse clouds). **Payoff:**
  El-Badry's own mechanism (interface cooling *reduces* the evaporative mass flux) is an **independent
  prediction that FM1b should show `ΔdMdt < 0`** — a clean falsification test for the next prototype. Sources:
  arXiv:1902.09547 (El-Badry 2019), arXiv:2104.07720/22 (Lancaster 2021).
- **2026-06-26 (latest) — Rung B risk #1 PROTOTYPED OFFLINE → §3a plan REFUTED, redirected. No code touched.**
  `data/make_fm1_rootcheck.py` replays the §3a closure (fix `dMdt`, shoot `v(R1)=0` on `dTdr_front`) on two
  **real captured stiff states**, sweeping `dTdr_front` over 6 decades for suppression `s ∈ {1,3,10,30}`.
  **Result** (`data/fm1_rootcheck.csv`, `fm1_rootcheck.png`): `s=1` finds the root (built-in correctness
  check — recovers Spitzer), **`s=3/10/30` find NO root in either state, anywhere** — so **FM1 fired**. Why:
  the recoil term is tiny (shifts `v_front` by ~0.5 of a ~2243 streaming velocity) but the stiff BVP
  **exponentially amplifies `v_front`** (that ~0.5 moves `v(R1)` by ~2000), while `dTdr_front` barely moves
  `v(R1)`. So **`v(R1)=0` is set by `dMdt` (the recoil), not the conduction gradient** — the sign argument was
  backwards, and `dMdt` is **not a free dial**. **Redirect:** keep `dMdt` as the Weaver eigenvalue, add
  mixing-layer `L_mix` only to the **in-structure loss integrand** (~10⁵ K band, κ unchanged), re-solve, and
  **measure ΔL_cool vs ΔdMdt** (the new make-or-break, FM1b) — next offline prototype, still before any code.
  The capture/replay discipline worked: a wrong design hypothesis cost a 2-fixture harness, not a regression.
- **2026-06-26 (later) — `κ_eff` Rung B scoped on paper (`RUNGB_SCOPING.md`), no code touched.** Two
  independent verifications (IC algebra + cooling/evaporation decoupling, both adversarially checked vs
  current source; the front-balance identity confirmed to machine precision). Headline finding: in the
  Weaver solve the conductive flux `q=κ·dT/dr` at the front is **one quantity read twice** (fixes `dMdt`
  via the enthalpy balance *and* sets the radiating profile) — which is *why* Rung A raised both together,
  and why a faithful `κ_eff` must **sever `dMdt` from the front balance** (entrainment-set, `>0` by
  construction) rather than swap `κ`. The mix-branch near-front IC is **numerical** (`κ_mix∝1/T ⇒ p=−1 ⇒
  q=−1` is not front-regular); `κ_mix`'s magnitude needs an entrainment efficiency `α_mix≪1` (literal
  `D_turb=R2·v2` ⇒ `T_cross~10¹²` K, absurd) — *that factor is the model*. `dMdt>0` safety threads the
  cleanroom §6.6 trap because `dMdt` becomes an input, not a root. `(β,δ)` solver untouched (no
  conduction-law dependence; its `dMdt>0` gate + `bubble_LTotal` use are the coupling surfaces).
  **Risk #1 now worked on paper (`RUNGB_SCOPING.md` §3a):** fixing `dMdt` over-determines the BCs, so
  **demote `dMdt` to an entrainment-set input and shoot `v(R1)=0` on the front gradient `dTdr_front`** — the
  conduction layer absorbs the boundary mismatch by radiating more/less instead of by changing evaporation
  (the decoupling, in the closure). Make-or-break is **FM1** (does that closure admit a `v(R1)=0` root? — a
  §9 failure-mode ledger records FM1–FM6), to prove OFFLINE on a captured state before any code. Also
  clarified the Rung-A figure (`kappa_backreaction.png`): added an absolute-`Lcool` panel (both runs rise;
  `f_κ=2` sits above `f_κ=1`) so the ratio panel's downward slope isn't misread as "cooling falling." Still
  no production edit.
- **2026-06-26 — `κ_eff` Rung A executed (back-reaction probe, gated/byte-identical-off).** Added
  `cooling_boost_kappa` (`f_κ`, default 1.0) multiplying the Spitzer coefficient `C_thermal` at all 3
  bubble-structure sites (`bubble_luminosity.py:291/:370/:406`). **Gate passed:** byte-identical when
  `f_κ=1` (sha `acbad31b`, 79 rows of `f1edge_hidens`), diverges when `f_κ=2`; full `pytest` 595 green
  (the `test_dR2min_magic_number.py::_scalar_params` minimal fixture patched to carry the neutral key),
  ruff F-rules clean. **Crux measured** (`data/make_kappa_backreaction.py`, `data/kappa_backreaction.csv`,
  `kappa_backreaction.png`): at matched `t`, `f_κ=2` raises `Lcool` ×1.23–1.38 (cooling rises *through the
  structure*, θ as an output) **but `dMdt` ×1.08–1.17 rides along** — the El-Badry coupling a faithful
  `κ_eff` must instead suppress. A `2×` κ buys only **+0.05–0.10** loss-ratio toward the 0.95 trigger ⇒
  brute-`f_κ` is non-viable *as a way to reach the trigger* ~~confirming Rung B is required, not optional~~
  **[superseded same-day by the merge: reaching the trigger is not the goal; Rung A already delivers the
  cooling magnitude, and Rung B is an optional bonus]**. Details: `KAPPA_EFF_SCOPING.md`
  §6a. **Production unchanged** — `cooling_boost_kappa` defaults to 1.0 (opt-in, byte-identical off).
- **2026-06-25 (late) — Cooling-boost program CONCLUDED; PLAN re-validated line-by-line.** Completed the
  diffuse arm (`f1edge_lowdens` ×2/×3 → **4/4 live configs**; `runs/data/live_compare.csv`) — no constant
  fires across density. Put the coupled `θ_target(Da)` on trial: offline Da-screen **NO-GO** + a
  **gate-validated real-Da replay** (`data/make_da_replay.py`; reproduces logged `bubble_Lloss` to ≤3.9e-5,
  interface L3 bit-identical) → **`θ_target(Da)` REFUTED** (T_int ~const ⇒ real Da ≈ proxy; Da≫1 everywhere ⇒
  `θmax·Da/(1+Da)` saturates to a constant; non-monotonic in nCore). Live `theta_target` validation: the
  literature θ (0.9–0.99) **straddles** the 0.95 trigger threshold ⇒ a scalar can't separate magnitude from
  triggering. **Pivot (§Outcome & pivot):** for normal clouds **blowout is the transition trigger**; the
  cooling boost corrects *magnitude*; `κ_eff` is the scoped endgame (`KAPPA_EFF_SCOPING.md`, feasible/bounded).
  **Re-validation:** all 8 offline screens re-ran **byte-identical**, the real-Da replay re-passed its gate,
  20/20 tests green; code line-refs corrected (`Edot_from_balance :434→:475`; trigger `:1200→:1206`;
  shadow/drive drifts) and the stale `f1edge_lowdens "NOT run"` reconciled. See `FINDINGS.md`.
- **2026-06-25 — LIVE matched-t edge runs (3/4 configs) DONE; a constant f=2 over/under-shoots by density.**
  Ran `none` vs `multiplier f=2` for hidens (dense), simple_cluster (compact), fail_repro (heavy) in
  separate processes (provenance clean, `commit=6642ff4, dirty=False, rc=0`; persisted `runs/data/live_compare.csv`
  + 6 harvest trajectories). Findings: hidens f=2 fires cooling **at birth** (t=0.0034, before blowout —
  over-boost); simple_cluster f=2 fires **just after** blowout (t=0.131 vs blowout 0.109) with a large live
  trajectory shift (Eb −47%, v2 −44%, R2 −15% → **frozen screen insufficient, confirmed**); fail_repro
  collapses identically with/without boost (cooling doesn't rescue heavy clouds — control confirmed). ⇒
  **no constant f_mix fits the density grid** → confirms the coupled `θ_target(n)=θ_lit(n)` direction
  (calibrate to the literature loss fraction, NOT to the 0.95 trigger threshold — the latter is circular).
  Diffuse `f1edge_lowdens` was not run *in this batch* (worktree mis-fork from `main` + a ~55–60 min env
  wall-cap on background runs) — **but was run later the same day (×2/×3; 4/4 configs total); see the top
  ledger entry and `runs/data/live_compare.csv`.** See `runs/README.md` §Live results.
- **2026-06-24 (pm) — Verified the maintainer's revised note line-by-line against source + screen data.**
  Code anchors all **confirmed** (Eq.1 ODE = `get_betadelta.py:475`; trigger = `(Lgain−Lloss)/Lgain<0.05`
  radiative-only `:1206`; no boost knob in `trinity/`). My screen numbers **reproduce exactly**. Found and
  fixed: (a) **trigger-convention bug** — the note's Table 2 headline `f_mix≈1.1–1.5` is the *with-PdV*
  screen, inconsistent with the note's own *no-PdV* recommended trigger; consistent value is
  **`f_mix≈1.4–2.8`** (`data/fmix_table.csv`, both conventions); (b) the **5×10⁵-draw** double-count claim
  had no committed script → added `data/make_doublecount_mc.py`+`doublecount_mc.csv` (0 draws enter the
  region); (c) Table 2 now script-emitted (`data/make_fmix_table.py`). Literature values farmed out to a
  web-verify pass (separate). **Then started Task B** — wiring opt-in `cooling_boost_mode` (gated,
  byte-identical when off) for the live test. See §"Task B".
- **2026-06-24** — Folded in the maintainer's Paper-II interface-cooling note (`f_mix` *multiplier* vs
  `θ_target` *fraction*; **boost the loss, not the trigger**; one `Lloss_eff` in three places; `κ_eff`
  endgame) — §Refined plan. Ran the **8-config staged shadow** (frozen trajectory) → §Stage results.
  **Verdict so far:** normal clouds want a *cooling boost* (`f_mix≈1.5–2` lands the ratio near the
  transition); heavy 5e9 wants the *PdV/`ebpeak`* handoff — a clean sub/super-critical split. A *constant*
  knob can't place the transition at blowout across the density grid (the firing f_mix spans 1.1→3.1) ⇒ points
  to the coupled `θ_target(Da)`/`κ_eff` form. **Production still unchanged** (grep-confirmed, anchor 3).
- **2026-06-23** — Scoped the maintainer's "PdV in the trigger" question. "PdV negligible" is false
  (`PdV/Lmech` median 0.43–0.55); the real fork is `PdV/Lmech ≷ 1`. Offline-tested **reading B**
  (`(Lmech−Lloss−PdV)/Lmech<0.05`) → fails as a usable trigger; recommended **reading A** (`ebpeak`).
  All offline from already-committed per-step CSVs.

**Decision RESOLVED (maintainer said go, 2026-06-24):** the opt-in `cooling_boost_mode ∈
{none, multiplier, theta_target}` wiring is **implemented and gated** — byte-identical when `none`
(confirmed through the active-cooling region), `multiplier f=2` diverges at the first active-cooling step.
See **§Task B**. **Open next step:** the matched-`t` edge-config **live** runs (boosted vs unboosted,
separate processes) that replace the frozen screen and settle constant-`f_mix` vs `θ_target(Da)`.

**Re-verify these load-bearing anchors on entry** (re-validated line-by-line 2026-06-25 — all 8 offline
screens reproduce byte-identical, real-Da replay re-passed its gate, 20/20 tests green; line-refs below corrected):
1. **PdV at 3 sites** (§Where PdV lives) — ODE `run_energy_implicit_phase.py:847-848`
   (`residual_Edot2_guess ← betadelta_result.Edot_from_balance`); `cooling_balance` trigger `:1206`
   (radiative, **no** PdV); `ebpeak` shadow `evaluate_r1_shadow():198-211` + drive `:1198-1204`.
2. **Opt-in is byte-identical** — `transition_trigger` default `cooling_balance` (`registry.py:347`,
   `default.param:282`); a non-default token only *drives* the R1 handoff, never perturbs a default run.
3. **Cooling boost knob has LANDED in production (2026-06-25, supersedes the 2026-06-24 "production
   untouched").** `grep -rn 'cooling_boost_mode' trinity/` is **no longer empty** (re-run 2026-06-25:
   `get_betadelta.py`, `run_energy_implicit_phase.py`, `registry.py`, `default.param`). Both the
   `multiplier` AND `theta_target` modes are implemented (`effective_Lloss`/`effective_Lloss_from_params`,
   `get_betadelta.py:334,360`: `multiplier` → `Lleak + fmix·Lcool`; `theta_target` → `max(Lcool+Lleak,
   θ·Lmech)`), declared as `cooling_boost_mode/_fmix/_theta` (`registry.py:348-350`, `default.param`), and
   fed **consistently** to the (β,δ) residual (`get_betadelta.py:473,577`), the `Edot_from_balance` energy
   ODE (`get_betadelta.py:475`), and the 0.95 trigger (≡ `(Lgain−Lloss)/Lgain<0.05`,
   `run_energy_implicit_phase.py:1153/1157`) — default `none` ⇒ byte-identical (§Task B). What remains
   **UNimplemented** is ONLY the density/Da-coupled target `θ_target(Da)` (constant `θ`/`f_mix` only so
   far) — see §Next deliverable. (docs/dev spirit: this anchor was stale within a day; re-verify the grep
   each visit.)
4. **The Stage numbers are a SCREEN, not a forecast** — `data/closure_test.csv` is a *frozen-trajectory*
   reconstruction; boosting cooling lowers `Pb`→`PdV`→**moves blowout itself**, so the fire-times need the
   Tier-2 **live** run (separate processes, matched `t`) before any verdict is trusted (§Hard caveat).

---

**Last updated:** 2026-06-26 (live status in the re-entry ledger above). **Branch:**
`feature/PdV-trigger-term`. This note answers the maintainer's question ("add a PdV term to the transition
trigger — what was the argument against it, and is it still valid for larger clusters?"), the **2026-06-23
redirect** (test reading B directly; what does the standalone `PdV/Lmech` diagnostic buy us), and the
**2026-06-24 interface-cooling direction** (boost the *loss*, not the trigger — Paper-II note). Sibling
priors (re-verify per banner): `../pt4/TRANSITION_FIX_SCOPING.md` (Route 1),
`../pt4/r1shadow/R1_FINDINGS.md`, `../../failed-large-clouds/PLAN.md` §6.

---

## The question (maintainer, 2026-06-23)

> "Perhaps it's time to add a PdV term into the transition trigger. What was the argument against it?
> If it was that PdV is always deemed too small and negligible, maybe that's not the case now — or not
> the case for larger clusters."

## TL;DR (answers, with evidence below)

1. **"PdV negligible" was never the real argument.** Measured, PdV is **order-unity** — `PdV/Lmech`
   median **0.43–0.55** for *every* normal cloud (table below), not "way too small." So the premise's
   *stated* reason is false: PdV is not negligible.
2. **The actual argument is two-fold:** (a) PdV is **already in the energy evolution** — `Eb(t)` is
   integrated from `Edot_from_balance = Lmech − Lloss − 4πR2²·v2·Pb` (`get_betadelta.py:475`), which
   *includes* the PdV work term; the `cooling_balance` *trigger* deliberately watches only the
   **radiative** ratio `(Lmech − Lloss)/Lmech` because the modelled transition was hypothesised to be
   cooling-driven. (b) Putting PdV *into the trigger* (= the `ebpeak` criterion `Edot_from_balance ≤ 0`)
   **fires nowhere new for normal clouds**, because they are **sub-critical**: even with PdV included,
   net energy keeps growing (Eb grows 1.5–14×10³ monotonically; shipped shadow fires `ebpeak` **0/6**).
3. **The premise is exactly right for large clusters.** At `mCloud=5e9` the cloud is **super-critical**
   — `PdV/Lmech` median **1.42** (PdV *exceeds* Lmech), net energy goes negative, `Eb` peaks and
   collapses (growth **1.014×**). That super-critical regime is precisely where the heavy-cloud crash /
   `ENERGY_COLLAPSED` lives. So a PdV-inclusive trigger (`ebpeak`) **is** the principled handoff *there*.
4. **The machinery already exists.** `ebpeak` shipped as an opt-in, default-off `transition_trigger`
   token (default `cooling_balance`, byte-identical). So this is **not** "add PdV" from scratch — it is
   **"validate the PdV-inclusive trigger toward becoming the default (or the heavy-cloud handoff), and
   measure where the sub→super-critical boundary sits across the science grid."**

## Where PdV lives today (3 sites — verified against current source 2026-06-23)

| site | formula | PdV included? | role |
|---|---|---|---|
| **Energy evolution** `get_betadelta.py:475` (`Edot_from_balance`), stored `residual_Edot2_guess` (`run_energy_implicit_phase.py:847-848`) | `Lmech − Lloss − 4πR2²·v2·Pb` | **yes** | how `Eb` actually evolves — PdV already drains the reservoir |
| **`cooling_balance` trigger** `run_energy_implicit_phase.py:1206` | `(Lmech − Lloss)/Lmech < 0.05`, `Lloss = bubble_LTotal (+leak)` | **no** | the default energy→momentum handoff; pure radiative |
| **`ebpeak` trigger** (opt-in) `evaluate_r1_shadow` `:208-210`, shadow `:1166-1190`, drive `:1192-1204` | `Edot_from_balance ≤ 0` | **yes** | "PdV in the trigger" — the net-energy turnover; default-off |

⇒ The maintainer's "add PdV to the trigger" **is** the existing `ebpeak` criterion. It is *not* the same
as "add PdV to the 0.05 cooling ratio" — see §Two readings.

## Evidence: PdV magnitude per regime

Persisted: `data/pdv_regime_budget.csv` (derived from the committed `../cleanroom/data/c0_*_h0.csv` and
`../../failed-large-clouds/data/budget_*.csv`; regenerate with `data/make_pdv_regime_table.py`). Startup
rows dropped; ratios in trinity code units (`PdV = 4πR2²·v2·Pb`, same convention as `Edot_from_balance`).

| config | regime | `PdV/Lmech` med | `PdV/Lmech` max | `Eb` growth | `Eb` monotonic | real in-cloud Eb-peak? |
|---|---|---|---|---|---|---|
| simple_cluster | normal | 0.460 | 0.646 | 2405× | yes | **no** |
| small_dense_highsfe | normal | 0.464 | 0.658 | 1951× | yes | **no** |
| midrange_pl0 | normal | 0.461 | 0.702 | 2009× | yes | **no** |
| pl2_steep | normal | 0.429 | 0.701 | 2171× | yes | **no** |
| be_sphere | normal | 0.453 | 0.666 | 1715× | yes | **no** |
| large_diffuse_lowsfe | normal | 0.443 | 0.550 | 1499× | yes | **no** (end-of-run blip only) |
| small_1e6 (ctrl) | normal | 0.554 | 1.102 | 13617× | yes | **no** (end-of-run blip only; max>1) |
| **fail_repro** | **heavy 5e9** | **1.423** | 1.561 | **1.014×** | **no** | **yes (row 5, t≈1.53e-3 Myr)** |

Authoritative cross-check (real shipped shadow, segment-wise, not a CSV reconstruction):
`../pt4/r1shadow/r1_shadow_summary.csv` — all 6 normal configs fire **blowout**, `ebpeak` column **blank
(0/6)**; `fail_repro`/`fail_helix` show `n_seg=0` because they collapse in **phase 1a**, *before* the 1b
shadow ever runs. ⚠️ My offline `net_Edot≤0` reconstruction has edge sensitivity (flags a few
end-of-run / startup blips as "fires"); where it disagrees with the shipped shadow, **trust the shadow**.

**Reading:** `PdV/Lmech` is a smooth control parameter. Normal clouds sit ~0.45 (sub-critical, ≪1
margin to the Eb-peak); the 5e9 cloud sits ~1.4 (super-critical). **The whole behavioural fork is which
side of `PdV/Lmech = 1` the cloud lands on.** "PdV negligible" is false everywhere; "PdV decisive" is
true only past the unity crossing.

## Two readings of "add PdV to the trigger" (pick deliberately)

- **(A) `ebpeak` — net-energy turnover `Lmech − Lloss − PdV ≤ 0`.** Threshold = 0 (energy stops
  growing). Principled (it is the physical Eb-peak), already implemented, fires only super-critically.
  Normal clouds: never in-cloud. Heavy clouds: at birth. **This is the recommended meaning.**
- **(B) PdV inside the 0.05 cooling ratio — `(Lmech − Lloss − PdV)/Lmech < 0.05`.** Keeps the legacy
  threshold but moves the operating point by ~`PdV/Lmech ≈ 0.45`. This is **not** physically grounded
  (0.05 was calibrated for a radiative-only ratio) and would fire for normal clouds at an arbitrary
  epoch set by an un-recalibrated constant. **Not recommended** unless re-derived from a model — record
  it only as the literal interpretation of the request, then steer to (A).

## Offline test of reading B — does `(Lmech−Lloss−PdV)/Lmech < 0.05` fire? (2026-06-23 redirect)

The maintainer asked to **test reading B directly** and questioned the point of the standalone `PdV/Lmech`
diagnostic. Both are answered **offline** from the already-committed per-step CSVs — no sims — by
`data/make_combined_trigger_table.py` (→ `data/pdv_combined_trigger.csv`, `pdv_combined_trigger.png`).
Numbers reproduced by an independent recompute on `large_diffuse_lowsfe`/`simple_cluster`/`small_dense_highsfe`.

**The identity that settles the `PdV/Lmech` question.** Write `cool = (Lmech−Lloss)/Lmech` (the shipped
radiative cooling ratio, *no* PdV) and `coolPdV = (Lmech−Lloss−PdV)/Lmech` (the same ratio *with* PdV =
reading B). The with-PdV ratio is *algebraically* the radiative one minus `PdV/Lmech`:

    coolPdV = (Lmech−Lloss−PdV)/Lmech = (Lmech−Lloss)/Lmech − PdV/Lmech = cool − PdV/Lmech

So **`PdV/Lmech` is exactly the offset between the shipped `cooling_balance` trigger and reading B** — its
only role is to quantify how much folding PdV into the ratio loosens the operating point. It is a
*decomposition* diagnostic, **not** a threshold variable; thresholding it against 1 (old Step 1) chases a
sufficient-but-not-necessary proxy (the real crossing is `(Lloss+PdV)/Lmech`, and `Lloss/Lmech` ≈ 0.17–0.29
is not negligible). Equivalently: **reading B = the shipped trigger run at threshold `0.05 + PdV/Lmech ≈ 0.5`**
— a ~10× looser, un-recalibrated constant.

**Result — first-fire of `coolPdV < 0.05` (sustained), vs the shipped `cool < 0.05`:**

| regime | configs | `cool<0.05` fires | `coolPdV<0.05` fires | where / note |
|---|---|---|---|---|
| normal | 5/6 cleanroom | 0 | **0** | min `coolPdV` only 0.08–0.15 — never reaches 0.05 |
| normal | large_diffuse_lowsfe | 0 (cool≈0.49 there) | **yes, sustained** | t≈4.76 Myr, **86% through** the run — arbitrary epoch |
| heavy 5e9 | fail_repro | 0 | at birth (row 3, t≈1.5e-3) | `coolPdV<0` immediately, stays `<0` for the physical run |
| ctrl | small_1e6 | 0 | row 0 startup blip (not real) | spurious — recovers to ~0.40 |

**Verdict on reading B (threshold 0.05): it does not behave as a usable trigger.** For 5/6 normal clouds it
is silent (the bubble never stops gaining energy — `coolPdV` bottoms at 0.08–0.15 and recovers); for the 6th
it fires at a late, arbitrary epoch fixed by the mis-set constant, where `cool` is still ≈0.49 (no physical
handoff). The only physically-grounded threshold for the PdV-inclusive ratio is **0** (= `ebpeak`/reading A,
net energy stops growing): normal clouds essentially never cross it in-cloud (`large_diffuse` only oscillates
across 0 at the very end, non-sustained), the 5e9 crosses at birth. **The data confirms reading A over B.**

**Corollary — the real handoff for normal clouds is not energy-budget at all.** Sub-critical clouds fire
*neither* `cool` nor `coolPdV`, so what drives their transition is **blowout** (geometric `R2 > rCloud`),
consistent with the shipped 1b shadow (6/6 blowout, `ebpeak` 0/6). An energy-balance trigger — radiative or
PdV-inclusive — is the wrong family for them; it is decisive only super-critically (the 5e9 pathology).
The figure now marks each config's **blowout point** (`R2 = rCloud`, recovered from `r1_shadow_summary.csv`
since the CSVs export `rCloud` as all-NaN; persisted as `blowout_t`/`cool_at_blowout`/`coolPdV_at_blowout`):
**at blowout the cooling ratio is still 0.30–0.75 (no PdV) / 0.12–0.58 (with PdV)** — i.e. the clouds hand
off to momentum while the energy budget is nowhere near the 0.05 band. That gap is the room a cooling-boost
`θ_cool` would have to close — see next section.

## A more promising direction: PdV **+** a cooling-boost `θ_cool` (2026-06-23 maintainer idea)

> "Maybe a combination of PdV in the cooling **and** a cooling-boost factor `θ_cool`, argued from El-Badry,
> Lancaster, Gronke: our 1D model has no turbulent mixing layers, so we under-count interface cooling.
> `θ_cool` could be a constant from those papers — but shouldn't it couple to the bubble physics?"

**Why it's promising (offline screening — frozen-trajectory, see caveat).** Solve for the constant boost that
makes the PdV-inclusive ratio reach the threshold, `(Lmech − θ_cool·Lloss − PdV)/Lmech = 0.05`, on the
*committed (unboosted)* trajectories:

| config | `f_mix` @blowout **(w/ PdV)** | `f_mix` anywhere (w/ PdV) | `f_mix` anywhere (no PdV) | **`f_mix` @blowout (no PdV) — consistent** |
|---|---|---|---|---|
| small_dense_highsfe | 1.10 | 1.04 | 1.33 | **1.36** |
| simple_cluster | 1.12 | 1.06 | 1.41 | **1.42** |
| midrange_pl0 | 1.20 | 1.08 | 1.49 | **1.56** |
| be_sphere | 1.26 | 1.18 | 1.80 | **1.86** |
| pl2_steep | 1.49 | 1.24 | 1.86 | **2.78** |
| large_diffuse_lowsfe | 3.13 | 0.87 (already <1) | 1.78 | **3.81** |

> ⚠️ **Convention fix (2026-06-24 verification).** The first three columns are the *original 2026-06-23*
> screen, which put **PdV inside the trigger ratio** (`(Lmech − f·Lcool − PdV)/Lmech = 0.05`). The
> Paper-II note's recommended trigger keeps **PdV out** (in the ODE only; reversible vs irreversible).
> The consistent screen is therefore the **last column** (`f = 0.95/(Lcool/Lmech)` at blowout) — and the
> note's Table 2 imported the *with-PdV* column (1.1–1.5) as its headline, which understates the boost by
> ~`PdV/Lmech`. The consistent headline is **`f_mix ≈ 1.4–2.8`** (compact five) — matching both my newer
> §Stage-results `cb` screen (1.5–2) and the literature target (lift `Lcool/Lmech≈0.25–0.7` to `θ≈0.95`).
> Reproducible now: `data/make_fmix_table.py` → `data/fmix_table.csv` (both conventions, from
> `pdv_combined_trigger.csv`).

So **a modest cooling boost `f_mix ≈ 1.4–2.8` (no-PdV trigger; 1.1–1.5 if PdV is folded in) would fire the
energy→momentum handoff right at blowout** for 5/6 normal
clouds — and that boost is *below* the enhancement the mixing-layer literature argues for (El-Badry+19
catastrophic cooling; Lancaster+21 near-complete wind-energy cooling; Gronke & Oh mixing layers). This is the
first candidate that makes a PdV-inclusive trigger physically *and* numerically land where the cloud actually
transitions, instead of never (reading B) or at an arbitrary epoch.

> ⚠️ **Caveat — these θ are a SCREENING estimate, not the answer (rule 5: per-call ≠ full-run).** `θ_cool` is
> "we're missing real cooling," so it belongs in the **energy evolution** (`Edot_from_balance = Lmech −
> θ_cool·Lloss − PdV`), not only in the trigger ratio. Boosting cooling lowers `Pb` → lowers `PdV` (`∝Pb`) →
> changes `Eb(t), R2(t), v2(t)` → **moves blowout itself**. The table freezes the unboosted trajectory, so it
> is necessary-but-not-sufficient. The honest test is a **full run with boosted cooling**, separate processes,
> matched `t`, on the edge configs.

**Constant vs. coupled (the maintainer's question).** Start **constant** — one opt-in float `theta_cool`
(default `1.0`, byte-identical; ponytail-simplest, calibratable, testable). But constant is physically a
placeholder: the mixing-layer luminosity is **not** constant — it scales with the contact-discontinuity area
(`∝R2²`), the shear/turbulent velocity (`∝v2` / hot-gas sound speed), and the mixing-layer cooling function
(Damköhler number; Tan/Oh/Gronke 21, Lancaster fractal-area scaling). The data already argues coupling is
needed: **the firing f_mix spans 1.1 → 3.1** (with-PdV) across configs, so no single constant fires them all at blowout.
Upgrade path: `θ_cool(R2, v2, T)` from the mixing-layer scalings — mark the constant version with a
`ponytail:` comment naming that ceiling.

**Where it plugs in (code map, verified 2026-06-23).** Cooling is computed in
`trinity/bubble_structure/bubble_luminosity.py::_bubble_luminosity()` (three-zone trapezoid integral →
`bubble_LTotal`); **no existing boost knob** (`cool_alpha/beta/delta` are Weaver evolution params, not
efficiency). Add `theta_cool` like `transition_trigger`: `ParamSpec` in `trinity/_input/registry.py` (~:350)
+ a line in `default.param`, then multiply the cooling integrand at the `_bubble_luminosity` site (R2, v[r],
T[r], T_avg, n[r], Pb are all in scope there for a coupled form). Default `1.0` ⇒ byte-identical.

**Recommended sequencing:** (1) opt-in constant `theta_cool` (default 1.0); (2) full-run screening on the edge
configs (`simple_cluster` + `f1edge_{lowdens,hidens}` + a 5e9) to see whether `θ_cool ≈ 1.5–3` makes the
PdV-inclusive trigger fire near blowout *self-consistently*; (3) only then a coupled `θ_cool(R2,v2,T)`.

## Refined plan — unresolved-interface-cooling closure (Paper-II note, 2026-06-24)

Supersedes/sharpens the `θ_cool` sketch above (where my "θ_cool" = the note's **`f_mix`**, a *multiplier*, not a
*fraction*). Driver: the maintainer methods note *"Adding unresolved interface cooling to TRINITY without
double-counting"* + an adversarial physics check (double-count algebra **verified**; `max()` closure is
**single-count by construction** — `Lloss_eff/Lmech = max(Lcool/Lmech, θ)`, never the forbidden
`Lcool/Lmech + θ` — confirmed empirically by `data/make_doublecount_mc.py` (5×10⁵ draws, **0** enter the
double-count region; result `data/doublecount_mc.csv`).

**Framework (note §2–6):**
- Distinguish loss **fraction** `θ ≡ Lloss/Lmech ∈ [0,1]` (a target/output) from loss **multiplier**
  `f_mix ≡ Lcool_mix/Lcool_smooth ≥ 1` (a knob on the resolved integral). One symbol must not name two operations.
- **Never double-count.** TRINITY already subtracts the explicit `Lcool`; adding a `(1−θ)Lmech` input-rescale on
  top removes `2θLmech` at consistency (net drive negative for θ>½). The correction must **add only the missing
  part**, never rescale `Lmech`.
- **Boost the LOSS, keep the trigger form.** Note's trigger is `(Lmech − Lloss_eff)/Lmech < 0.05`, with PdV in
  the **ODE only** (not the trigger). Physics: PdV is *reversible* (recoverable as shell momentum), cooling is
  *irreversible* — fire on the irreversible channel. ⇒ **This is distinct from reading B** (which put PdV in the
  trigger); the note instead fixes the cooling *magnitude*.

**Closures under test (default `none` ⇒ byte-identical):**
- `multiplier`: `Lloss_eff = Lleak + f_mix·Lcool` — sweep probe; does *not* change the T-profile or evaporation
  (its ceiling — a scalar can't back-react on the evaporative mass flux).
- `theta_target`: `Lloss_eff = max(Lcool+Lleak, θ_target·Lmech)` — double-count-free **iff** the two terms are
  estimators of the *same* sink (they are: `θ_target·Lmech` is a target on the resolved-cooling fraction). Tops
  up to the target, switches OFF where resolved cooling already exceeds it.
- `kappa_eff` (endgame, out of shadow scope): `κ_eff = max(κ_Spitzer, κ_mix)`, `κ_mix ~ ρ cp D_turb`,
  `D_turb ~ λ δv ~ R2 v2`. The only honest form — couples cooling↔evaporation and can reproduce El-Badry's 3–30×
  evaporation suppression; scalar closures cannot. The scalars are calibration probes that point here.

**Consistency contract (note §Code-level):** one helper feeds the β–δ residual, the energy ODE, *and* the
trigger — the same `Lloss_eff`. Shadow ⇒ reconstruct the trigger ratio only; production ⇒ this is the gate.

**Staged shadow / non-disruptive test — all 8 configs (6 normal + fail_repro + fail_helix):**
- **Stage 1 — Gate audit** (note's "check the gate first"): per-segment {active triggers, Lcool, Lleak, Lmech,
  PdV, β, δ, residual, baseline ratios}. Confirm cooling-balance is *active but never trips* (ratio stays high),
  not a gate bug. Plot: baseline ratio trajectories + blowout markers.
- **Stage 2 — Closure sweep (FROZEN trajectory):** both closures over `f_mix ∈ {1,1.5,2,3,5,10,30}` and
  `θ_target ∈ {0.3,0.5,0.7,0.8,0.9,0.95}` (ceiling **θ_max=0.95** at GMC-core n — the El-Badry density scaling is
  an *extrapolation* there). Per (config × value): does the note `cb` trigger fire? sustained? `t_fire/R2_fire`
  vs blowout? Plots: (a) per-config ratios under increasing boost; (b) fire-vs-blowout heatmap (config × value),
  multiplier and theta_target.
- **Stage 3 — Double-count / consistency check:** instantiate the note's Fig 1 with real per-config `Lcool/Lmech`;
  show the closures stay on the single-count line and never enter the `2θ` region. Plot: that diagram, 8 configs placed.
- **Stage 4 — Which is good:** rank by — fires near blowout for normal clouds (not birth, not never), preserves
  heavy-cloud collapse, double-count-safe, and whether the firing value is ~constant across configs (⇒ a constant
  knob suffices) or spreads (⇒ needs the Da/κ_eff coupling). Recommend a candidate + the gated **Tier-2 full run**
  (disruptive: apply `Lloss_eff` in residual+ODE+trigger, separate processes, matched `t`) as the NEXT step.

**Hard caveat (rule 5 + physics-check §5.1): the shadow only SCREENS.** Boosting cooling lowers Pb → lowers PdV →
moves blowout itself; the unboosted trajectory is *not* the state the boosted ODE visits. Shadow fire-times are a
screen, **not predictions** — the verdict needs Tier-2.

### (HISTORICAL, superseded by the merge) Next deliverable that *was* PRIMARY (2026-06-25) — the coupled `θ_target(Da)`

> **⭐ SUPERSEDED (2026-06-26):** `θ_target(Da)` was both **REFUTED** (below) *and* the framing is obsolete — the
> primary next deliverable is now **`f_κ(properties)` calibration** via the κ_eff mechanism (⭐ synthesis at
> top). Kept as the motivating analysis for *why* a constant fails and a density-dependent target is needed.

> **STATUS 2026-06-25: `θ_target(Da)` was TESTED and is REFUTED** — Step A (offline proxy) and Step A′ (the
> gate-validated real-Da replay) are **both NO-GO**. The rationale below is kept as the motivating argument;
> the revised forward plan is in **"Outcome & pivot"** at the end of this section.

This was previously filed as a "coupled upgrade to **record (not implement)**". As of 2026-06-25 it is
promoted to the **primary next deliverable**, because the analysis below shows a *constant* target is not a
real contribution — only a trajectory-varying `θ_target(Da)` is.

**The constant-θ / `fmix_no_pdv` calibration is DEGENERATE with the existing 0.95 trigger.** The
"consistent" screen solves `f_mix(n) = 0.95/(L_cool/L_mech)` at blowout (§Cooling-boost table last column)
— but the 0.95 there is *the trigger threshold itself* (the trigger is `(Lgain−Lloss)/Lgain<0.05` ⇒ fire
when `Lloss/Lmech` reaches 0.95). So `f_mix(n)=0.95/(L_cool/L_mech)` is **bit-identical to the `fmix_no_pdv`
column by construction** — it just restates "boost the resolved loss until it hits the threshold." A flat
literature `θ_lit≈0.95` therefore adds **nothing quantitative**: it lands exactly where the un-boosted
trigger already would if cooling reached 0.95. **A constant target is not a real contribution.**

**The only non-degenerate upgrade is a target that VARIES along the trajectory:** `θ_target(Da)`,
`Da = t_turb/t_cool` (Damköhler number) — density- AND time-dependent. Because it moves with the state, it
absorbs the density/SFE/stage confound that the edge configs cannot separate (recall the firing f_mix spans
1.1→3.1 across the grid — no constant fires them all). Functional form to validate:
`θ_target(state) = θ_max · Da/(1+Da)` — recovers El-Badry (high-Da, interface-dominated) and Weaver
(low-Da, energy-driven) limits from one dimensionless ratio.

**Honest prerequisite scoping (verified against source 2026-06-25).** Production computes **none** of the Da
ingredients yet: `grep -rn 't_turb\|Damk' trinity/` is **empty** (2026-06-25), and there is **no standalone
interface density `n_int`** (only `n_interm`, the intermediate-zone density already used in the cooling
integral, `bubble_luminosity.py:761`). The closest existing proxy is the **OFFLINE** `F2_tcool_tdyn =
(Eb/Lloss)/(R2/v2)` in `docs/dev/transition/harness/harvest.py:14,110-112` — diagnostic only, and
previously judged a **red herring** (it fires ~60× too early). So the deliverable scope is:
  1. **Build Da from LIVE solver state** — the interface `n,T` already used in the cooling integral
     (`bubble_luminosity.py`), with `R2/v2` as the turbulent-timescale proxy (`t_turb ~ λ/δv ~ R2/v2`).
  2. **Choose/validate the `θ_max·Da/(1+Da)` form** against the edge configs (does it fire near blowout
     self-consistently across the density grid where a constant cannot?).
  3. **Gate it byte-identical-when-off** exactly like the existing cooling-boost knob (§Task B): a new
     `theta_target` sub-mode/parameterisation that reduces to the current behaviour when disabled.

(This stays PLAN/scoping prose — it is the *next step*, not an implementation.)

#### Step A result (2026-06-25) — offline Da-screen: NO-GO for the `(R2/v2)·Pb` proxy → the real Da needs a replay

`data/make_da_screen.py` (+ `data/da_screen.csv`, `da_screen.png`) screened the **offline** Da target on the
6 cleanroom trajectories. Under a fixed characteristic interface T_int, `Da` collapses to
`Da_shape = (R2/v2)·Pb` (units absorbed by a swept normalization ⇒ a **unit-independent structural test**).
Result: **no single normalization fires the grid at blowout** — two failure modes:
- `Da_shape`@blowout is **non-monotonic in nCore and spans ~14×** (`pl2_steep` 1e5 = 4222, *below*
  `large_diffuse` 1e2 = 4601; `simple_cluster` 1e5 = 54690). The `θ_max·Da/(1+Da)=0.95` crossing is at one
  fixed Da, so it can coincide with blowout only if `Da_shape`@blowout were ~constant — it is not.
- `Da_shape` is large early (high Pb at small R2) → any C that pushes the diffuse configs to θ≈0.95 fires the
  dense configs at **birth** (fmb ≈ −0.85). 0/6 valid sustained fires anywhere on the C×θ_max grid.
The bulk `Da_bulk = 1/F2` baseline fires far before blowout (confirms the red herring). Empirical θ/(1−θ)@
blowout rises only ~6.9× over 4 decades (slope ~0.18) — **shallower than √n**; 6 points + confounded SFE
cannot decide √n (El-Badry) vs linear-n (Da).

**What it rules out / does NOT.** It rules out the *offline shortcut* (Da from frozen `(R2/v2)·Pb`), because
that combination collapses away the per-config/per-time `T_int` and `Λ` — the very quantities that could
separate the configs. It does **not** rule out `θ_target(Da)` itself; the proper Da is untested. So step 2
is revised:

  **2′. Compute the REAL Da by REPLAY (no full re-runs).** For each row of the committed cleanroom
  trajectories, re-invoke trinity's interface calc (`bubble_luminosity.py` → `T_int(r)`, `Λ(T_int)`,
  `n_int = Pb/(k_B T_int)` ⇒ `t_cool,int = (3/2)k_B T_int/(n_int Λ)`) to get `Da = (R2/v2)/t_cool,int`, then
  re-run `make_da_screen.py` on the real Da. **GO** ⇒ implement (step 3). **NO-GO on the real Da** ⇒
  `θ_target(Da)` is the wrong closure — revisit (the shallow θ(n) may mean the constant target / degeneracy
  is the honest end state, or a different functional form is needed). The replay reuses production code on
  frozen states (CLAUDE.md rule 5) — cheaper and more faithful than the proxy or a full re-run.

#### Step A′ result (2026-06-25) — real-Da replay: gate PASS, verdict **NO-GO** (`θ_target(Da)` refuted)

`data/make_da_replay.py` (+ `data/da_replay.csv`, `da_replay.png`) recomputed the REAL Da by replaying
trinity's own interface cooling on the 6 cleanroom trajectories. **Validation gate PASSES:** the replay
reproduces the logged `bubble_Lloss` to ≤3.9e-5 (tol 1e-3) and the interface zone `L3` is **bit-identical**
(reldiff 0) — so the real Da is trustworthy, not a proxy artifact. Verdict: **NO-GO** — 0/6 valid sustained
fires under any single `(C, θ_max)`. Three decisive reasons:
- **`T_int` is ~constant across all configs (~21.4–22.6 kK).** The radiative interface sits where Λ peaks,
  independent of cloud, so `Da ≈ (R2/v2)·Pb·Λ(T_int)/const ≈ proxy × const` — the offline proxy was a *good*
  approximation and its NO-GO carries over.
- **Real `Da`@blowout is still NON-monotonic in nCore** (pl2_steep 1e5 = 4.7e4, *below* large_diffuse 1e2 =
  5.6e4 and midrange 1e4 = 4.2e5; spread 14×). No monotonic `θ(Da)` can order the configs by density.
- **`Da ≫ 1` everywhere at blowout (4.7e4–6.6e5)**, so `θ_max·Da/(1+Da)` **saturates to ~θ_max for every
  config** → collapses to a *constant* target → exactly the degeneracy that adds nothing (density-law
  exponent p≈0, flat).

#### Outcome & pivot (2026-06-25)

A cooling-magnitude knob — constant **or** `Da`-coupled — is **not** what triggers the energy→momentum
transition for these clouds. At blowout the resolved loss ratio is only 0.25–0.70 (well short of 0.95), and
`Da` neither orders by density nor discriminates. Convergent, data-backed conclusion (matches the methods
note's closing nuance): **for normal clouds the operative handoff is geometric blowout (`R2=rCloud`), not
cooling balance.** Revised program:
  1. **Drop `θ_target(Da)` as a trigger mechanism** (refuted by a gate-validated replay).
  2. **Treat blowout as the transition trigger for normal clouds** — which TRINITY's default already does
     (cooling_balance rarely fires first; the momentum phase begins at blowout). The "runs never transition"
     symptom is the *cooling magnitude*, not the trigger.
  3. **Correct the cooling MAGNITUDE with the κ_eff mechanism, calibrated to a density-dependent target.**
     **Update (the merge, 2026-06-26):** κ_eff = `cooling_boost_kappa` (Rung A, **already built/gated**) is the
     in-structure mechanism that raises emergent cooling (`bubble_LTotal` ×1.23–1.38); the calibration *target*
     is `θ(n_H)` (El-Badry `λδv`=κ_eff + Lancaster ≈0.9–0.99), and the knob is `f_κ(properties)`. A *constant*
     `θ` via `theta_target` is the degenerate special case (≈0.95 = the trigger); the real upgrade is the
     **density-dependent f_κ calibration**, not a scalar floor. So `θ, Eb, Pb, R2, v2` come out right *through*
     the blowout handoff because the cooling fraction emerges per cloud. (The faithful evaporation-decoupling
     re-derivation in **`KAPPA_EFF_SCOPING.md`** / **`RUNGB_SCOPING.md`** is an *optional high-fidelity bonus*,
     not required for the goal — the 1D front-anchored `dMdt` resists it; see `FM1`/`FM1b`.)
  4. **Confirm with live matched-`t` runs** that the magnitude correction doesn't distort the trajectory.

**Data:** 7/8 offline-reconstructable (6 cleanroom h0 + `budget_fail_repro`); `fail_helix` has only logs (collapses
in phase 1a) → needs the in-solver shadow run. Artifacts: `data/make_closure_test.py`, `data/closure_test.csv`,
`closure_stage{1..4}*.png`.

### Stage results (2026-06-24 — FROZEN-TRAJECTORY SCREEN; bounds the knob, does not forecast)
1. **`cb` trigger (boost loss, no PdV) is the right family for normal clouds:** `f_mix ≈ 1.5–2` brings their cooling
   ratio into the band near the transition. Supersedes reading B (don't put PdV in the trigger; fix the cooling).
2. **A constant knob can't place the transition at blowout across the grid (Stage 2 heatmap).** At `f_mix≈2`,
   compact/dense fire *at* blowout (`simple_cluster −0.07`, `small_dense −0.01` Myr, at `f_mix=2`) but diffuse fire *well before*
   (`pl2_steep −0.81`, `large_diffuse −1.3…−3.65`). Density-ordered (dense already cool: `Lcool/Lmech≈0.7` at
   blowout; diffuse `≈0.25`) ⇒ **the data argues for the coupled `θ_target(Da)`/`κ_eff` form, not a constant.**
3. **`theta_target` constant is blunt:** fires nowhere below 0.95, ~at birth at 0.95 — use only via the
   density-dependent `θ_target(n)` model + ceiling. The **multiplier `f_mix` is the better probe.**
4. **Heavy clouds are complementary, not covered by cooling:** `fail_repro` never fires `cb` even at `f_mix=30`
   (`Lcool/Lmech≈0.01`, PdV-dominated) ⇒ heavy clouds need the **PdV/ebpeak handoff**, normal clouds the **cooling
   boost**. Clean sub-/super-critical regime split.
5. **Double-count check (Stage 3):** every config sits on the single-count line; the closures never enter `2θ`.

**Next (gated, disruptive — NOT in the shadow):** wire an opt-in `cooling_boost_mode ∈ {none,multiplier,theta_target}`
feeding the β–δ residual + ODE + trigger *consistently* (note §Code-level), run ≥2 edge configs **live** (separate
processes, matched `t`) to test self-consistency vs the frozen screen; add the in-solver 1a/1b shadow to cover
`fail_helix`. Then decide constant-vs-`θ_target(Da)` from the live spread.

## Task B — opt-in `cooling_boost_mode` wiring (2026-06-24, gated, byte-identical when off)

The maintainer authorised wiring the closure for a **live** test. Built exactly as the note's §Code-level
rule demands — **one helper, three sites, default off ⇒ byte-identical**.

**Implementation (production):**
- `effective_Lloss(mode, fmix, theta_target, Lcool, Lleak, Lmech)` + the params wrapper
  `effective_Lloss_from_params(...)` in `get_betadelta.py` (after `compute_R1_Pb`). Modes: `none` →
  `Lcool+Lleak` (byte-identical); `multiplier` → `Lleak + fmix·Lcool`; `theta_target` →
  `max(Lcool+Lleak, θ·Lmech)`. An unrecognised token falls back to the resolved loss (a typo can't
  perturb a run).
- Fed **consistently** to all three sites (the note's consistency contract): the β–δ residual
  (`get_residual_pure`), the `Edot_from_balance` ODE/detail path (`get_residual_detailed`), and the
  `cooling_balance` trigger (`run_energy_implicit_phase.py` ~:1147). Same `Lloss_eff` everywhere.
- 3 params (`cooling_boost_mode`/`_fmix`/`_theta`) in `registry.py` + `default.param`, mirroring
  `transition_trigger`'s `exclude_from_snapshot=True, run_const=True` — drops them from
  `dictionary.jsonl` (`dictionary.py:254/616`), routes them to `metadata.json` → default run byte-identical.

**Gate (rule 5 — real runs, separate processes; `simple_cluster` mCloud=1e5 sfe=0.3):**
- **`none` byte-identical to HEAD through the active-cooling region** (snapshots 1–128; resolved cooling
  activates at snap 98 — `bubble_Lloss` is NaN before that, so the test only bites past 98, and PASSES
  there). Provable too: the `none` branch is the identical `Lcool+Lleak` float op the original ran.
- **`multiplier f=2` diverges at snapshot 99** — the *first* active-cooling step — confirming the boost
  is genuinely live; `metadata.json` confirms the knobs load.
- ✅ ruff F-rules clean; ✅ 20/20 tests (`test_cooling_boost.py` 6 + `test_r1_shadow.py` 14).

**Still a SCREEN, not a forecast (anchor 4):** the gate proves the wiring is correct and *safe*; it does
**not** yet replace the frozen screen. NEXT: matched-`t` edge-config live runs (`simple_cluster` +
`f1edge_{lowdens,hidens}` + a 5e9), boosted vs unboosted in separate processes, to settle
constant-`f_mix` vs coupled `θ_target(Da)`.

## Plan & test design (rule-5 ladder — this is a risky/iterative/outward-facing change)

The change touches the solver's phase-handoff and the late-time **fate** outputs, and is a
**default-flip** candidate ⇒ full ladder, no rung skipped because an earlier passed.

### Step 0 — Gate first (define "equivalent" before any edit)
- **Hard gate:** any default change must be **byte-identical** (`dictionary.jsonl` sha256) on every
  config that *already* transitions via `cooling_balance`. (Under `hybr` that set is currently empty —
  0/6 fire — but legacy/clamped-β configs and any future-cooled model still use it, so the gate stands.)
- **Continuity gate:** at the handoff, `Eb / R2 / v2 / P_drive` must enter phase 1c no more
  discontinuously than the `cooling_balance` handoff does. **The heavy-cloud Eb-peak is the make-or-break
  case** (reservoir grew only 1.014× → 1c may reject a near-empty bubble).
- Pass/fail bars + `f_ret` targets written here *before* editing.

### Step 1 — Decisive new measurement: combined-ratio first-fire across the science grid
> **2026-06-23 redirect (supersedes the old "map `PdV/Lmech = 1`" framing).** The decision-relevant
> quantity is the **combined ratio** `coolPdV = cool − PdV/Lmech`, not `PdV/Lmech` alone (see §Offline test
> for why `PdV/Lmech=1` is a sufficient-but-not-necessary proxy). The offline first-cut is **done** above;
> the open question is the *in-process, authoritative* version. Still record max/median `PdV/Lmech` per cell,
> but only as the **offset diagnostic** that explains the `coolPdV`–`cool` gap — not as the boundary to map.

The open scientific question behind the maintainer's premise: **does any *realistic* cluster (not just
the 5e9 pathology) approach super-critical?** If the boundary sits far above the science range, the PdV
trigger is an edge-case guard; if real sweeps straddle it, it is a default-relevant correctness fix.
- Sweep `mCloud × sfe × density-profile` (reuse `../../failed-large-clouds/harness/params/` +
  `../cleanroom/configs/`), each run in a **separate process**, with the **shadow active** (default
  trigger ⇒ byte-identical), harvesting `shadow_R1_1b.csv` + per-segment `PdV/Lmech`.
- Record max/median `PdV/Lmech` and first `ebpeak`/`blowout` epoch per cell → a contour of the
  sub→super-critical boundary. Persist as `data/pdv_boundary_grid.csv` + a figure.
- **Note the phase-1a gap:** the 5e9 Eb-peak is a **phase-1a** event invisible to the 1b shadow
  (`r1_shadow_summary.csv` `n_seg=0`). To measure the heavy end, add a **read-only 1a shadow** of
  `Edot_from_balance` (mirror of the 1b shadow; logging only, no break) — itself a gated micro-change.

### Step 2 — Baseline capture
`git show HEAD` trajectories (Eb/R2/v2/P_drive/end-code) on the edge set: `simple_cluster` +
`../../performance/f1edge_{lowdens,hidens}*.param` + a 5e9 point. Saved here so "before" survives.

### Step 3 — Equivalence / behaviour gate
- **Per-call (cheap, necessary, not sufficient):** unit-test `evaluate_r1_shadow` / `r1_transition_decision`
  truth tables already exist (`test/test_r1_shadow.py`, 14/14) — extend with the 1a-shadow analogue.
- **Full-run, stiffest regimes, separate processes, matched `t`:** run `transition_trigger=ebpeak`,
  `blowout`, and `r1` (=both) on all 6 normal configs + ≥1 heavy 5e9, compared against the `cooling_balance`
  baseline at matched simulation time. Check: (i) run completes; (ii) **phase-1c continuity** of the four
  state vars; (iii) terminal fate (momentum, final R2, stop code) is physically defensible.

### Step 4 — Apply the smallest diff that passes
Likely candidates, smallest first: (a) add the **1a Eb-peak shadow** (read-only); (b) if Step 3 shows
the heavy handoff is clean, wire **`ebpeak` as the heavy-cloud handoff** replacing `ENERGY_COLLAPSED`
(opt-in first); (c) only if Step 1 shows science-grid relevance, propose a **default flip** to
`cooling_balance,blowout` (and/or `ebpeak`) — additive, so it never perturbs a run that already fires.

### Step 5 — Re-verify
Continuity + byte-identity gate again, full `pytest` (`-m "not stress"`), ruff F-rules.

### Step 6 — Persist
Boundary grid CSV + figure, baseline/edge trajectories, and the handoff-continuity comparison committed
under `docs/dev/transition/pdv-trigger/` with the exact config + command for each.

## Open questions / risks
- **Heavy handoff viability:** the 5e9 Eb-peak hands off a *stillborn* reservoir (1.014× growth) — does
  phase 1c accept it, or is `ENERGY_COLLAPSED` still the honest end? (pt4 H3/H4 lean toward "needs
  momentum continuation or added cooling," not just a trigger relabel.)
- **Does reading (B) ever make sense?** Only if a recalibrated, model-derived threshold replaces 0.05.
  Default to (A).
- **Boundary location:** if no realistic cluster reaches `PdV/Lmech > 1`, the PdV trigger is a guard for
  the pathological edge, not a science-sweep correctness fix — that changes the priority of a default flip.

### In-solver shadow insertion point (code map, verified 2026-06-23)
If/when the authoritative in-process confirmation of reading B is wanted (the offline reconstruction has the
end-of-run/startup edge sensitivity the §Evidence cross-check warns about), it is a ~5-line **read-only**
add at `trinity/phase1b_energy_implicit/run_energy_implicit_phase.py:1166`, right after the existing
`ebpeak` shadow eval — `Lgain` (=`Lmech_total`), `Lloss`, `R2`, `v2`, `params['Pb'].value` and
`betadelta_result.Edot_from_balance` are all in scope there. Add `combined_ratio = (Lgain − Lloss −
4πR2²·v2·Pb)/Lgain` (= `edot_balance/Lgain`) and a `combined_ratio` / `combined_ratio_fired` column to the
`shadow_rows` dict (the same block already logs `cooling_ratio` and `edot_balance` → `shadow_R1_1b.csv`).
Byte-identical (logging only); extend the `test/test_r1_shadow.py` truth table (14 tests). **Lower priority
given the offline verdict** — it confirms, it does not change, the reading-B finding.

## Artifacts
- `data/fmix_table.csv` (+ builder `data/make_fmix_table.py`) — the methods-note **Table 2**, now
  script-emitted from `pdv_combined_trigger.csv`. Both trigger conventions (with-PdV screen vs the
  consistent no-PdV recommended trigger); headline `f_mix ≈ 1.4–2.8`. Regenerate: `python
  docs/dev/transition/pdv-trigger/data/make_fmix_table.py`.
- `data/doublecount_mc.csv` (+ builder `data/make_doublecount_mc.py`) — the 5×10⁵-draw Monte-Carlo that
  backs the note's double-count-free claim (0 draws enter the `2θ` region; single-count by construction).
- `data/closure_test.csv` (+ builder `data/make_closure_test.py`, figures `data/make_closure_plots.py` →
  `closure_stage{1..4}*.png`) — the §Refined-plan **8-config staged shadow** (frozen-trajectory screen;
  §Stage results). Regenerate: `python docs/dev/transition/pdv-trigger/data/make_closure_test.py && python
  docs/dev/transition/pdv-trigger/data/make_closure_plots.py`.
- `data/pdv_combined_trigger.csv` (+ `data/make_combined_trigger_table.py`, figure `pdv_combined_trigger.png`)
  — the §Offline-test reading-B first-fire table. Regenerate: `python docs/dev/transition/pdv-trigger/data/make_combined_trigger_table.py`.
- `data/pdv_regime_budget.csv` (+ `data/make_pdv_regime_table.py`) — the §Evidence table.
- `data/da_screen.csv` / `data/da_replay.csv` (+ `make_da_screen.py` / `make_da_replay.py`, figs
  `da_screen.png` / `da_replay.png`) — the offline Da-shape screen + the gate-validated real-Da replay that
  **refuted `θ_target(Da)`**.
- **κ_eff / the merge:** `data/kappa_backreaction.csv` (+ `make_kappa_backreaction.py`, fig
  `kappa_backreaction.png`) — Rung A back-reaction (`bubble_LTotal` ×1.23–1.38, the **cooling mechanism** at
  work); `data/fkappa_leverage.csv` (+ `make_fkappa_leverage.py`, fig `fkappa_leverage.png`) — the **f_κ
  calibration first cut** (leverage `∝ f_κ^0.63`, viable to f_κ=64); `data/kappa_calibration_estimate.csv`
  (+ `make_kappa_calibration_estimate.py`, fig `kappa_calibration_estimate.png`) — the **f_κ(n_H) calibration
  estimate** (diffuse ≈8, dense ≈1.6); `fkappa_definition.png` (+ `data/make_fkappa_definition.py`) — the
  **equation-grounded f_κ definition** (Spitzer law κ_eff=f_κ·C_th·T^(5/2); seed dMdt∝f_κ^(2/7) verified
  1.2175 vs 1.219); `runs/params/cal_{compact,diffuse}__k{1,2,4}.param` + `runs/params/cal_mid__ek{1,2,4}.param`
  + `data/make_kappa_blowout_calibration.py` (→ `kappa_blowout_calibration.png`) — the **measured full-run
  calibration (3 configs)**: θ(f_κ=1)=0.67/0.61/0.17, f_κ-to-fire ≈4/~5-6/~60 (compact measured/mid & diffuse extrap.);
  `ideas_comparison.png` (+ `make_ideas_comparison.py`) — the all-ideas scoreboard; `data/_trinity_style.py` —
  the **shared TRINITY plot style** (loads `paper/_lib/trinity.mplstyle`, LaTeX-free) for storyline consistency.
- **PdV-in-the-trigger (the founding question, measured):** `data/pdv_trigger_compare.csv` (+
  `make_pdv_trigger_compare.py`, fig `pdv_trigger_compare.png`) — PdV is the dominant sink, PdV-inclusive ratio
  0.65–0.91 at f_κ=1; `runs/params/cal_{compact,diffuse}__ebpeak.param` + `data/make_ebpeak_trigger_test.py`
  (→ `data/ebpeak_trigger_test.csv`, fig `ebpeak_trigger_test.png`) — the **code-path test**: `ebpeak` does NOT
  fire at f_κ=1 (peaks 0.91/0.86 then declines); the cooling↔PdV trade-off keeps diffuse PdV-incl flat across f_κ.
  `data/make_ebpeak_8config_xcheck.py` (→ `data/ebpeak_8config_xcheck.csv`, fig `ebpeak_8config_xcheck.png`) —
  the **8-config coverage cross-check**: frozen-screen peak ratio per config + live overlay (simple_cluster
  live 0.911 == frozen 0.911); 6 normal configs peak 0.85–0.92 / no fire, only heavy-5e9 + control fire.
- **Rung-B negative results (offline, optional-bonus line):** `data/fm1_rootcheck.csv` (+ `make_fm1_rootcheck.py`,
  fig `fm1_rootcheck.png`) — FM1 (imposing `dMdt` refuted); `data/fm1b_evapsign.csv` (+ `make_fm1b_evapsign.py`,
  fig `fm1b_evapsign.png`) — FM1b (interior cooling: El-Badry sign, negligible magnitude).
- Storyline report: `make_pdvtrigger_report.py` → `pdvtrigger_report.html`.
- Upstream (committed): `../cleanroom/data/c0_*_h0.csv`, `../../failed-large-clouds/data/budget_*.csv`,
  `../pt4/r1shadow/r1_shadow_summary.csv`.
