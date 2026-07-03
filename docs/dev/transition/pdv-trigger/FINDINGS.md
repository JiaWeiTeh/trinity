# PdV-trigger workstream — findings (✅ direction corrected 2026-07-01: θ is an output, f_κ reinstated — see §8c)

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

> **Provenance of this write-up.** Verified rewrite — line-by-line checks against source caught real errors
> in an earlier draft (listed in §0). Every number was re-checked against the committed CSVs / config files
> / run logs on 2026-06-25; claims are tagged **[data]** (measured), **[interpretation]**, or
> **[schematic / to-verify]**.

## Taxonomy of the approaches (read first; 2026-06-28)

> ✅ **DIRECTION CORRECTED 2026-07-01 — read `PLAN.md` ⭐⭐ canonical synthesis + `FINDINGS.md §8c` FIRST.**
> The current direction is **Rung A / f_κ — boost the cooling MECHANISM and let θ EMERGE** (`cooling_boost_mode=
> 'multiplier'`), with El-Badry/Lancaster as the **calibration target** for the emergent θ, not an enforced
> value. The intermediate "impose El-Badry θ" (`theta_target`/θ_elbadry) avenue is **demoted to an opt-in
> override** because enforcing θ double-counts the PdV loss on massive clouds (§8b) — a symptom of enforcement
> that f_κ does not have (§8c). So the f_κ callouts below are **back on the critical path** (the power-law
> *exponents* were recalibrated for the `multiplier` knob on 2026-07-02 — **see §10**, the rule-compliant
> theta5 matrix, which supersedes the "still need recalibration" state this banner previously recorded); the
> θ_elbadry callouts at the bottom are the **opt-in option**, not the
> default. Historical caveat: earlier revisions of THIS banner (06-30) said the reverse — treat those as
> superseded.

> **→ Calibration target (2026-06-29) — 🛑 SUPERSEDED (kept as history):** the closed form below was falsified
> on both parameters (slope: sweep measured −0.60 vs −0.30, scorecard P1 ❌; baseline: the 6-anchor θ₀ slope
> 0.41/dex vs the grid's 1.13/dex, P3 ❌), rests on the retired blowout-θ metric, and the whole f_κ(n) framing
> was then superseded by the **single-physical-constant DECISION** (F_KAPPA §14 ✅) + the **θ₁-collapse law**
> (§9). Do not calibrate against it — `CONTAMINATION.md` ⛔ #3. Original text:
> the composed closed-form **f_κ(n_H) = (θ\*/θ₀(n_H))^(1/p) ≈
> 1.4×10²·n_H^(−0.30)** now lives in **`F_KAPPA_FUNCTIONAL_FORM.md`** (target = Lancaster flat θ*≈0.90 · baseline
> `logit θ₀ = −1.73+0.41 log₁₀ n_H` · raw full-range leverage p≈0.31). f_κ≈48(diffuse)/9(mid)/3(dense) for
> θ*=0.95 — matches the measured firing anchor (compact fires at f_κ≈3.4). It supersedes the §2-area schematic
> for *how to set f_κ* and confirms there is **no literature `f_κ ∝ n_H^p`** to borrow. (A logit/odds-space first
> cut overshot ~10–30× — θ fires before it saturates; see the doc's §0 🛠 correction.) **El-Badry §3.1/§5.2 now
> VERIFIED from the maintainer-supplied PDF** (Eq 16/19/20/21/35/37/38; A_mix≈3.5) — the earlier `[unverified]`
> hedge is retracted, and El-Badry's θ(n_H,λδv) target agrees with Lancaster to ~15% in f_κ. Citation: MNRAS
> 490,1961 / Weisz / arXiv:1902.09547 (*not* ApJ 879). This supersedes the §2/§2a "[schematic/to-verify]" flags
> for El-Badry's specific algebra.
>
> **→ SWEEP RESULTS (2026-06-29):** the 819-combo grid ran (Helix). Measured central trend **f_κ_fire ≈
> 1.0×10³·n_core^(−0.60)** (θ\*=0.95) — steeper than the predicted n^(−0.30). **De-conflation = fan-out** (×2–32
> spread across mCloud/sfe ⇒ f_κ is multi-dimensional, not f(n_H) alone), and **6/63 low-n high-sfe cells never
> fire even at f_κ=64** (the diffuse corner needs the structural κ_mix). The pre-registered scorecard (2 ✅
> qualitative, slope ❌ 2× too shallow from an undersampled 6-anchor baseline) is in `F_KAPPA_FUNCTIONAL_FORM.md`
> §8 (`data/fkappa_nH_sweep.csv`, `data/make_fkappa_sweep_analysis.py`).
>
> **→ Fan-out anatomy + metric (2026-06-29, §9–§10):** the 1e7 "broken power law" = a **catastrophic-cooling
> cliff** — θ@f_κ=1 fires with no boost above a ≈constant **column** N_H≈8×10²³ (massive clouds fire at lower
> density because they sweep that column at lower n). The fan-out is multi-dimensional: nCore primary (R²=0.73),
> + rCloud/cloud-size secondary (2-var 0.75), **independent of cluster mass** (f_κ_fire vs M★ R²=0.002). The
> metric (θ=L_cool/L_mech at blowout R2>rCloud; fire on theta_max≥0.95) is robust — snapshot-vs-peak median
> 0.004 — with one fixable imprecision (theta_max not capped at blowout_t). Builder `data/make_fkappa_cliff_metric.py`.
>
> **→ Don't-force-it reframing (2026-06-29, §11–§12):** the *physical* f_κ (El-Badry κ_mix∝n) **rises** with
> density, OPPOSITE the empirical fire-threshold (∝n^−0.6) — so a physically-bounded f_κ leaves the diffuse corner
> **energy-driven by choice** rather than cranking f_κ to 64 to force it. A physical cap f_max≈2–8 predicts a
> falsifiable **critical column** N_crit≈1–4×10²³ cm⁻² for the energy→momentum split (6/63 never fire under any
> cap). Tension: Lancaster 3D says diffuse clouds also cool → accept-non-transition vs add-κ_mix, settled vs obs.
> Builder `data/make_fkappa_physical_cap.py`; prescriptions are testable by re-analysis of `summary.csv` (no sims).
>
> **→ Physical prescription DERIVED (2026-06-29, §13):** three f_κ(n) — mechanism κ_mix/κ_Spitzer ∝ n (rises),
> target θ*(n;λδv) flat-high, boost ∝ n^−0.6 (falls; a boost factor, NOT a conductivity). Crossover n_crit=0.25
> (matches El-Badry); a **scalar f_κ can't represent κ_mix** (Spitzer ∝ T^(5/2) vanishes in the cool layer →
> ratio 10³–10⁷). The verified El-Badry θ* is flat-high even at diffuse (0.94 vs 1D 0.29) ⇒ the diffuse
> never-fire is likely a **1D under-cooling artifact** → faithful fix is the **structural κ_mix (Rung B,
> re-promoted)**, not a scalar f_κ power law. Builder `data/make_fkappa_physical_derivation.py`;
> reconciles RUNGB_SCOPING's κ_mix-magnitude. Next: wire the gated κ_mix mode (RUNGB §8).
>
> **→ Manuscript draft verified (2026-06-29, `KMIX_DIFFUSIVITY.md`):** a maintainer LaTeX draft, ~90% matching our
> results. Key **refinement adopted**: do *not* import El-Badry's λδv∈[1,10] (doubly off-regime — discrete-SN +
> ISM density); use El-Badry for the *mechanism*, take δv from v_rel, and **calibrate λ so resolved θ matches
> Lancaster 0.9–0.99** (the cadence-free magnitude anchor). Flags: the draft's "sweep not yet run" is stale (it
> fanned out); the eddy-turnover λ closure is heuristic; route a (diffuse energy-driven) vs b (κ_mix under-cooling)
> stays open until κ_mix is wired + tested on all 8 configs.
>
> **→ κ_mix offline prototype (2026-06-29, `KMIX_PROTOTYPE.md`):** step 1 of the Rung-B implementation, **GO**.
> Offline, units-correct (Pb AU→cgs /1.5454e12; λδv→cm²/s), no solver touched. At the front
> `κ_mix/κ_Spitzer = λδv·Pb/(C_th·T^{7/2})`: in the cool layer (2e4–2e5 K) **κ_mix dominates Spitzer 10³–10⁹ even
> at λδv=1** (compact/diffuse/dense; 4 of 8 configs) → wiring warranted, but λδv is the sensitive knob (calibrate
> to Lancaster). Master navigation now in **`INDEX.md`**.
>
> **→ κ_mix SELF-CONSISTENT in the real solver (2026-06-30, `KMIX_SELFCONSISTENT.md`):** injected `κ_eff =
> max(κ_mix, κ_Spitzer)` into the production structure solve (monkeypatch, no edit), 6 cleanroom + 2 fixtures.
> G1 bit-identical-off + G2 replay pass. **κ_mix raises θ and the solver is stable, BUT θ SATURATES by
> λδv≈0.01** (κ_mix is 10⁵–10⁸× Spitzer the instant it is on) ⇒ **λδv is NOT a tunable knob — the "calibrate λδv
> to Lancaster" step is RETIRED.** The saturated θ is density-**mismatched** (diffuse overshoots → fires; mid/dense
> plateau 0.23–0.35 ≪ Lancaster, 1/6 fires). **Reconciliation (§2a):** f_κ scaled because it is a *modest scalar
> on hot-interior Spitzer* (linear regime, but unphysical); κ_mix saturates because it is a *T-independent floor
> born deep in the cool layer* (physical, but past the dial). **The tunable knob (f_κ) isn't physical; the
> physical term (κ_mix) isn't tunable.** The low dense θ is the *same* ceiling as the sweep's "6/63 never fire",
> revealed not created. Caveat: the plateau is a single near-blowout row — a time-integrated metric could move
> it (open). Gated production **on hold** pending the strategy revision (`KMIX_SELFCONSISTENT.md` §3).
>
> **→ κ_mix TIME-RESOLVED θ — the blowout metric was the wrong epoch (2026-06-30, `KMIX_SELFCONSISTENT.md`
> §2b).** Re-solved κ_mix across ~14 rows/config of the implicit phase (not one row). **θ peaks EARLY (high
> Pb) and decays — blowout = the low-θ tail**, so the §2 single-row read *undersold* κ_mix badly (`be_sphere`
> 0.23 → trajectory-max **1.84**). So the earlier "only 1/6 fires / misses Lancaster for mid" is **walked
> back**: where it solves, the **mid (n~1e4) clouds exceed θ=0.95 and would fire**; only the **dense (n≥1e5)**
> end stays low (θ_max≲0.5) — *that* ceiling is robust. **BUT the decisive early high-Pb epochs FAIL to solve**
> with the hard-max injection (0/4 early rows; baseline solves there) → firing *plausible but unconfirmed*.
> Also caught a **faithfulness bug**: κ_mix ∝ n ∝ 1/T, so the κ_mix-regime kprime is **−1/T not 0** (harness +
> SPEC §3). Next: a **smooth-max + correct-kprime** injection that survives the early phase, then re-run.
> Builder `data/make_kmix_theta_trajectory.py`.

What looks like "three ways to boost cooling" is really **two cooling-magnitude approaches on opposite sides
of the structure solve, plus a separate trigger axis**. The key disambiguation: **"modify cooling like
El-Badry with κ" and "modify the conduction front k_f" are the *same* knob** (`cooling_boost_kappa`) — raising
the conduction coefficient *is* the 1D stand-in for more radiating surface / mixing. Every row is read from
source (knob `registry.py`, equation file:line); no assumptions.

| axis / approach | knob | what it changes (from source) | θ: imposed or **emergent**? | literature | status / verdict |
|---|---|---|---|---|---|
| **A. Outcome-side** — operate on `L_loss` *after* the structure solve (`effective_Lloss`, `get_betadelta.py:334`) | | | | | |
| · scalar multiplier | `cooling_boost_mode=multiplier`, f_mix | `L_loss = L_leak + f_mix·L_cool` (`:354`) | scaled (semi-imposed) | — | no single f_mix fires across density (1.4–3.8) → superseded by §10 (2026-07-02): under θ_max a single f_mix=4 DOES fire the band (the 1.4–3.8 was the blowout frozen screen) |
| · **θ-target floor** ("sum like Lancaster θ") | `cooling_boost_mode=theta_target`, θ | `L_loss = max(L_cool+L_leak, θ·L_mech)` (`:356`) | **imposed** (top-down) | **Lancaster** θ≈0.9 | degenerate: constant θ=0.95 *is* the 0.95 trigger; θ(Da) refuted |
| **B. Mechanism-side** — operate on the conduction *inside* the structure; θ comes out | | | | | |
| · **κ_eff conduction multiplier** ("El-Badry κ" **=** "modify k_f / conduction front" — same knob) | `cooling_boost_kappa`, f_κ | `κ_eff = f_κ·C_th·T^(5/2)` at 3 sites (`bubble_luminosity.py:291/370/406`) → thicker front → more 10⁵–10⁶ K gas (more surface/mixing) | **emergent** (bottom-up) | **El-Badry** mixing (λδv↔κ_eff) | built/gated; f_κ≈4 (compact)…~60 (diffuse) (⛔ #3; knob later broke — §9a; see §10); side-effect: dMdt↑ |
| **C. Trigger-side** — *when* to transition, not *how much* it cools | | | | | |
| · PdV-inclusive trigger | `transition_trigger=ebpeak` | fire when `L_gain−L_loss−PdV ≤ 0` (`run_energy_implicit_phase.py:198,1206`) | n/a (timing) | El-Badry/Lancaster "cooling creeps up" | doesn't fire alone at f_κ=1; assist not substitute |

**A** imposes the result (Lancaster's θ lives here); **B** changes what *produces* the cooling so θ emerges
(El-Badry lives here, and it is the *same* knob as "modify the conduction front"); **C** is a different axis
(the transition criterion). **A and B must never be stacked** — the `max(·)` closure (§2 of the report) keeps
the loss single-count. Current direction = **B** (κ_eff), calibrated to a density-dependent target, with **C**
(PdV) as an optional timing assist.

## 0. What the verification changed (errors caught and fixed)
1. **§2 gap numbers were wrong.** An earlier draft said "gap ~0.45 (diffuse) → ~0.25 (dense), shrinking."
   Recomputing the gap from the *actual* plotted band function gives a **non-monotonic** result and a
   **negative** gap at the diffuse end (TRINITY sits *above* the schematic band there). Only the
   dense-end ~0.25 was right. The committed figure `theta_vs_density.png` carried the same wrong
   annotations — **they have been removed and the figure regenerated (this commit)**.
2. **§1 spread mis-stated.** Earlier "1.1×→3.8×, 3.5× spread" mixed two columns. Corrected below.
3. **§3 lowdens** runs were truncated at the 1200 s ceiling (run.py `exit=124`), not a natural finish —
   so the claim is "had not fired *by blowout*," not "never fires."
4. **Edge configs vary SFE too** (hidens sfe 0.01, lowdens sfe 0.5) — not a clean density-only contrast.
5. **§7 (Provenance)** does not pin the committed live runs to a commit hash (no tracked provenance).
6. **Blowout time for lowdens was misread.** A first pass eyeballed the diffuse blowout at ~1.3 Myr from
   the harvest — that was a column miscount. The matched-t comparator (R2 vs rCloud=70.12 pc) gives
   blowout ≈ **0.61–0.64 Myr** (none 0.611, ×2 0.620, ×3 0.639). Corrected in §3.

**Thesis under test:** TRINITY's resolved cooling-loss fraction `L_cool/L_mech` rises with density but the
constant boost needed to ignite the energy→momentum transition also rises steeply with *decreasing*
density — so **no single constant `f_mix` works across the density range**; the boost should track a
density-dependent target. Support rests on **§1 (boost-to-trigger spread) and §3 (live firing behaviour)**,
both solid; the literature-`θ_lit(n)` comparison (§2) is currently **schematic** and not yet evidence.
**→ Superseded 2026-07-02: this thesis is REFUTED for the θ_max metric** — the single-constant DECISION
(F_KAPPA §14) plus the theta5 matrix (§10) showed a single f_mix=4 fires the whole normal-GMC band; the
spread above was a blowout-era artifact of the frozen screen.

> **Update (the merge, 2026-06-26):** the "density-dependent target" is now concrete — the **mechanism** is
> **κ_eff** (`cooling_boost_kappa`, **Rung A, built/gated**), which raises the *emergent* cooling in-structure
> (§6 — measured `bubble_LTotal` ×1.23–1.38); the **target** is `θ(n_H)` (El-Badry `λδv`=κ_eff + Lancaster);
> the **knob** is `f_κ(properties)`. So the goal is **enhanced, density-dependent cooling matched to obs/3D**,
> delivered by *calibrating* f_κ — not by a scalar floor and not by chasing evaporation suppression. See
> `PLAN.md` ⭐ synthesis and `RUNGB_SCOPING.md` §2a (the canonical θ/`λδv`/`f_κ`/0.95 reconciliation).

---

## 1. [data] Boost needed to reach the 0.95 trigger rises steeply as density falls

> **Provenance note (2026-06-30):** the `PdV/L_mech` and `L_cool/L_mech` magnitudes in this section come
> from **frozen trajectory CSVs** (post-processed by `make_pdv_regime_table.py`, which runs no sim) —
> stale-risk. A **live re-measurement against current code** is in `data/live_pdv_decomp.csv`
> (`data/make_live_pdv_decomp.py`) and is recorded in `HIMASS_HANDOFF_PLAN.md` §1. It confirms the
> decomposition is **density-dependent**: diffuse-massive (5e9, n=1e2) is **PdV-dominated** (PdV ≈
> 1.43·Lmech, radiative ≈ 0.009 — live ≈ the frozen 1.42), but dense-massive (1e7, n=1e6) has PdV and
> radiative **co-dominant** (≈0.29 each; radiative 0.45 > PdV 0.27 at the Eb-peak), i.e. radiative is NOT
> negligible for dense clouds. Trust the live numbers where they differ.

`data/fmix_table.csv` (per config, at blowout). nCore column independently confirmed from
`docs/dev/transition/cleanroom/configs/<config>.param` (simple_cluster unset → schema default 1e5):

| config | nCore [cm⁻³] | L_cool/L_mech | PdV/L_mech | f_mix (with PdV) | f_mix (no PdV) |
|---|---:|---:|---:|---:|---:|
| small_dense_highsfe  | 1e6 | 0.697 | 0.182 | 1.10 | 1.36 |
| simple_cluster       | 1e5 | 0.667 | 0.206 | 1.12 | 1.42 |
| midrange_pl0         | 1e4 | 0.610 | 0.219 | 1.20 | 1.56 |
| be_sphere            | 1e4 | 0.511 | 0.308 | 1.26 | 1.86 |
| pl2_steep            | 1e5 | 0.342 | 0.441 | 1.49 | 2.78 |
| large_diffuse_lowsfe | 1e2 | 0.250 | 0.169 | 3.13 | 3.81 |

- The constant boost needed roughly **triples** from dense to diffuse — f_mix(no PdV) **1.36 → 3.81
  (≈2.8×)**, f_mix(with PdV) **1.10 → 3.13 (≈2.85×)**. **No single `f_mix` fits all densities** — the core
  of the thesis, and this is real measured data.
- `pl2_steep` (nCore 1e5) sits low at 0.342 — density is the main driver but **not the only one**
  (profile shape / SFE scatter it). [interpretation]
- **Figure** `fmix_vs_density.png` (data-only scatter; `data/make_fmix_spread_plot.py`) visualizes this
  spread: no horizontal "constant f_mix" line crosses all six configs. It is scatter (not a curve) on
  purpose — `pl2_steep` and `simple_cluster` share nCore 1e5 yet need 2.78 vs 1.42, so there is no clean
  f_mix(n). The figure also states the degeneracy (§2a): `f_mix = 0.95/(L_cool/L_mech)` is exactly what a
  flat θ_lit≈0.95 would prescribe, because the trigger threshold *is* 0.95.

## 2. θ_lit(n) figure — `theta_vs_density.png` (schematic) → `elbadry_overlay.png` (VERIFIED, 2026-06-29)

TRINITY's resolved `L_cool/L_mech` (= 1 − `cool_at_blowout`) vs ambient nCore is **real [data]** and rises
**0.250 (1e2) → 0.697 (1e6)**. The original literature overlay (`theta_vs_density.png`) was **schematic** —
El-Badry+2019 / Lancaster+2021 PDFs 403'd, so the band was an arbitrary saturating stand-in.

**UPDATE 2026-06-29 — El-Badry PDF obtained, equations VERIFIED [data]:** `elbadry_overlay.png`
(+ `data/make_elbadry_overlay.py`) replaces the schematic band with the **real El-Badry §5.2 model**:
`θ = ψ/(11/5+ψ)`, `ψ = A_mix·(λδv)^½·n_H^½`, **A_mix=3.5** (Eqs 37–38, verified line-by-line). Our resolved
θ_1D points sit **far below** that target across the GMC range. **Crucial caveat:** El-Badry calibrated this
at **n_H,0 = 0.1–10 cm⁻³** (Figs 6–7); our clouds at n=1e2–1e6 are 1–5 decades beyond, where θ_target is
saturated to ≈0.94–0.999 by **extrapolation**, not measurement. (The earlier in-session doubt that those
equations were confabulated is **retracted** — they are genuine; only the GMC extrapolation is the open issue.)

**Recomputed gap (band_center − TRINITY) at each nCore — shows the schematic is not a usable comparator:**

| config | nCore | TRINITY | schematic band_c | gap |
|---|---:|---:|---:|---:|
| large_diffuse_lowsfe | 1e2 | 0.250 | 0.171 | **−0.079** (TRINITY above band) |
| be_sphere            | 1e4 | 0.511 | 0.833 | 0.323 |
| midrange_pl0         | 1e4 | 0.610 | 0.833 | 0.223 |
| pl2_steep            | 1e5 | 0.342 | 0.936 | 0.594 |
| simple_cluster       | 1e5 | 0.667 | 0.936 | 0.269 |
| small_dense_highsfe  | 1e6 | 0.697 | 0.949 | 0.251 |

- The gap is **non-monotonic** and **negative at the diffuse end** — so the earlier "0.45 diffuse → 0.25
  dense, shrinking" is **wrong** and is retracted. The figure's "gap ~0.45 / ~0.25" arrows **have been
  removed** (figure regenerated this commit); the script now documents why.
- The only defensible literature statement right now: at the **dense** end TRINITY (0.70) is below
  Lancaster's reported retained-cooling fraction (~0.9) — gap ~0.2. The **diffuse** end is **unknown**
  until real θ(n) is digitized (the schematic's 0.17 there is meaningless). [interpretation]
- **Open next step:** quote no gap until the El-Badry/Lancaster θ(n) is digitized (the 3 citations in
  `NOTE_PATCHES.md`); replacing the schematic band with a real one upgrades §2 from corroboration to
  evidence. The TRINITY trend itself stands.
- Caveat retained: x-axis is *ambient* nCore; θ_lit(n) tracks the *higher interface* density.

### 2a. Literature reconciliation (verified 2026-06-25, 3 subagents)

We ran an external literature report past three subagents (codebase / literature / reasoning). All
bibcodes resolve and the headline physics checks out — **but every arXiv/ADS/publisher endpoint returned
HTTP 403**, so El-Badry's *specific* numbers (Fig 7 @ 10 Myr; Eq 35 √ρ form; the θ-vs-n anchors) are
**UNVERIFIED (not refuted)** — they stay **[schematic / to-verify]**.

- **Correction to §2's comparator [interpretation]:** at GMC density (n ~ 1e2–1e6) the right anchor is the
  **Lancaster+2021a/b θ ≈ 0.90–0.99 plateau** — a derived, 3D-sim-validated result ("generic over >3 dex
  in density"), **NOT** an El-Badry √ρ extrapolation. El-Badry+2019 is a *supernova-superbubble* paper
  (ambient n ~ 0.1–10); its θ(n) **must not be pushed to GMC densities**. Best read on shape:
  **flat-and-high** — the plateau is well-supported, but the exact slope across 1e2–1e6 is **inferred**
  (no accessible source tabulates θ at 1e2/1e4/1e6). [schematic / to-verify]
  - **Reconciliation with `RUNGB_SCOPING.md` §2a (the merge):** "don't push El-Badry's √ρ *curve* to GMC
    density" stands — but El-Badry's *framework* (θ emerges from `κ_eff = λδv`, a set 1D knob) is exactly the
    mechanism TRINITY uses (`cooling_boost_kappa`). So El-Badry supplies the **mechanism/parametrization** and
    **Lancaster supplies the GMC magnitude** (θ ≈ 0.9–0.99); the calibration target is the two together, hit by
    tuning `f_κ`. The two docs are consistent under this reading.
- **If the band is redrawn flat at θ_lit ≈ 0.95**, the gap (θ_lit − TRINITY) is **positive everywhere** and
  **shrinks 0.70 (diffuse, 1e2) → 0.25 (dense, 1e6)** (0.95 − 0.250 = 0.70; 0.95 − 0.697 = 0.253), with
  `pl2_steep` an outlier (~0.61, its L/Lm anomalously low at 0.342). This **resolves the negative-gap
  artifact** the schematic produced and restores the *direction* of the retracted "gap shrinks
  diffuse→dense" intuition — for the right reason (TRINITY rising toward a flat ceiling). [interpretation;
  conditional on the plateau holding at the diffuse end — unverified]
- **Critical degeneracy [data/interpretation]:** if θ_lit ≈ 0.95 and flat, calibrating the boost to
  θ_lit(n) gives f_mix(n) = 0.95/(L_cool/L_mech) = our existing `fmix_no_pdv` column **bit-identically**,
  because TRINITY's trigger threshold *is* 0.95. So a **flat** "calibrate to θ_lit" is the **same arithmetic**
  as "calibrate to the 0.95 trigger" — it adds **no content** over §1. The escape is therefore a **non-flat,
  density-dependent target** that the cooling fraction is calibrated to. **Update (the merge, 2026-06-26):**
  the way to deliver that is the **κ_eff mechanism** — `cooling_boost_kappa` makes θ *emerge* per cloud (§6),
  and `f_κ(properties)` is calibrated so emergent θ tracks `θ(n_H)` (El-Badry `λδv`=κ_eff + Lancaster). This
  **supersedes** the earlier pointer to `θ_target(Da)` (now **refuted** — §5; Da≫1, non-monotonic, saturates):
  the density-dependence must come from `f_κ(n_H)`, not from a `Da`-coupled scalar floor.
- **Action:** still do **NOT** redraw the figure with a flat band (that is just another schematic); redraw
  only once Lancaster θ(n) is actually digitized. The TRINITY trend and the §1 boost spread are unaffected
  by any of this.

## 3. [data] LIVE matched-t edge runs — does the boost fire cooling before blowout?

`none` vs boosted, matched simulation time, separate processes. `fired_cooling_boost` = handed off via a
*cooling* trigger (True) vs blew out / never transitioned (False). Committed: `runs/data/live_compare.csv`
(+ per-arm `runs/data/harvest_f1edge_lowdens__*.csv`). Times in Myr. **The edge configs vary SFE as well as
density** (hidens 1e6/sfe0.01; lowdens 1e2/sfe0.5; simple_cluster 1e5/default) — a feedback×density edge
set, not a pure density sweep.

| config (boost) | nCore, sfe | t_trans none→boost | blowout (boost) | fired cooling? | reading |
|---|---|---|---|---|---|
| f1edge_hidens (×2)   | 1e6, 0.01 | 0.0314 → 0.0034 (1st step) | none (nan) | **True**  | dense fires cooling at birth, before any blowout |
| simple_cluster (×2)  | 1e5, dflt | 10.44 → 0.131 | 0.109 | **False** | blows out (0.109) *before* it transitions (0.131); ΔEb up to 47% |
| f1edge_lowdens (×2)  | 1e2, 0.50 | no transition (trunc.) | 0.620 | **False** | diffuse: doesn't fire by blowout; ΔEb 13%, ΔR2 5% at matched t; blowout +9 kyr vs none |
| f1edge_lowdens (×3)  | 1e2, 0.50 | no transition (trunc.) | 0.639 | **False** | doesn't fire even at ×3; ΔEb 24%, ΔR2 9%; blowout +28 kyr vs none |
| fail_repro (×2)      | heavy/path. | 0.0034 → 0.0034 | none (nan) | **False** | pathological config; boost has no effect |

lowdens baseline `none` blew out at **0.611 Myr** and never transitioned; **all three lowdens arms were
truncated at the 1200 s ceiling** (exit 124) at sim-time t≈3.0–3.3 Myr, so "fires *after* blowout, before
stop_t?" is unanswered — but "fires *before* blowout?" is a clean **No** for both ×2 and ×3.

**Live trend [interpretation]:** the boost needed to actually ignite cooling rises sharply as density
falls — dense fires at birth (×2), mid blows out before firing (×2), diffuse hasn't fired by blowout even
at ×3. Consistent with §1; confirms live that no constant `f_mix` fires cooling across the range
(blowout-era, truncated runs — see §10 for the θ_max result, which reverses this for f_mix=4).
(Note: density and SFE move together here, so "density" is shorthand for the dense-weak ↔ diffuse-strong
edge, not a clean one-variable result.)

## 4. [interpretation] Frozen-screen vs live discrepancy — worth scrutiny

Static table (§1) says simple_cluster needs only f_mix = 1.42 (no PdV) / 1.12 (with PdV) to fire; the live
boost is ×2.0 (confirmed in the param: `cooling_boost_fmix 2.0`), which exceeds both — yet the live ×2 run
does **not** fire cooling (it blows out at 0.109 first). The static "at-blowout" snapshot appears to
**over-predict firing** because blowout intervenes before the boosted cooling integral crosses 0.95 in a
sustained way. This frozen-vs-live gap is the main open interpretive question.

## 5. Caveats / open items
- lowdens ×2/×3/none all complete (all truncated at the 1200 s ceiling; blowout ~0.61–0.64 Myr). "Fires
  after blowout, before stop_t (15 Myr)?" is unanswered (runs cut at t≈3.0–3.3) — only "not before
  blowout" is established for the diffuse cloud.
- Edge configs confound density with SFE (§3) — keep that in any density-only claim.
- θ_lit(n) band schematic (§2) until PDFs digitized; figure gap annotations now removed.
- `fired_cooling_boost=False` + large t_trans shift (simple_cluster) = "transitioned, but via blowout, not
  cooling" — read the *mechanism*, not just the time.
- Diffuse table point (large_diffuse_lowsfe, cleanroom) ≠ live diffuse arm (f1edge_lowdens) — different
  mCloud/SFE; both nCore 1e2.
- **Da-screen (offline, 2026-06-25): NO-GO for the `(R2/v2)·Pb` proxy.** It can't separate the configs at
  blowout under any normalization (Da_shape@blowout non-monotonic, spans ~14×; dense configs fire at birth),
  so `θ_target(Da)` **can't be validated or refuted offline** — the proper Da needs the solver's interface
  `t_cool,int`. Next: compute the REAL Da by replaying trinity's interface calc on the frozen trajectories
  (no full re-runs), then re-screen (PLAN.md "Next deliverable" step 2′). θ/(1−θ)@blowout rises only ~6.9×
  over 4 decades — shallower than √n; can't decide √n vs linear. Artifacts: `data/make_da_screen.py`,
  `data/da_screen.csv`, `da_screen.png`. [data]
- **Da-screen — real-Da replay (gate-validated, 2026-06-25): also NO-GO → `θ_target(Da)` REFUTED.**
  `make_da_replay.py` re-ran trinity's own interface cooling on the frozen trajectories; the **gate PASSES**
  (`bubble_Lloss` reproduced to ≤3.9e-5, interface `L3` **bit-identical**), so the real Da is trustworthy.
  It is *still* non-monotonic in nCore (spread 14×), `T_int` is ~constant (~21–22.6 kK) so real Da ≈ proxy,
  and `Da≫1` everywhere → `θ_max·Da/(1+Da)` saturates to a constant → degenerate. **Pivot:** the cooling
  boost corrects cooling *magnitude*, not the trigger. **⚠️ FRAMING CORRECTED (06-26, verified in code):** the
  *default* energy→momentum trigger is the cooling-driven **`cooling_balance`** (`Lloss/Lgain>0.95`,
  `run_energy_implicit_phase.py:1206`; `transition_trigger` default `cooling_balance`, `default.param:282`);
  **geometric blowout (`R2>rCloud`) is opt-in, default OFF** and is only the *fallback symptom* when 1D cooling
  is too weak (resolved loss ratio only 0.25–0.70) for `cooling_balance` to fire. So the job of `κ_eff` is to
  make that cooling-driven trigger fire — the earlier "blowout is the trigger" was a mischaracterization. See
  PLAN.md ledger (06-26 + 06-28). Artifacts: `data/make_da_replay.py`, `data/da_replay.csv`, `da_replay.png`. [data]

## 6. [data] κ_eff IS the cooling mechanism — Rung A (the merge, 2026-06-26)

The pivot's "cooling boost corrects *magnitude*" (§5) now has a concrete, **in-structure** mechanism, and it
is **already built**: `cooling_boost_kappa` (`f_κ`, default 1.0, gated/byte-identical-off) multiplies the
Spitzer conduction coefficient `C_thermal` at all three sites in `bubble_luminosity.py` (`:291/:370/:406`).
Enhancing conduction puts **more gas in the ~10⁵ K radiating band**, so the cooling **emerges** higher (θ is an
*output*, El-Badry's approach — not a post-hoc floor).

- **What f_κ IS (equation-grounded, no assumptions; report §13) [data]:** `f_κ` = `cooling_boost_kappa` is a
  dimensionless multiplier on the **Spitzer–Härm conduction coefficient** `C_thermal = 6e-7 erg s⁻¹ cm⁻¹
  K⁻⁷ᐟ²` (`registry.py:341`): **κ_eff(T) = f_κ·C_th·T^(5/2)**. It enters the 3 sites in `bubble_luminosity.py`
  — dMdt seed (`:291`, ⇒ **dMdt ∝ f_κ^(2/7)**), conduction-layer ICs (`:370`, ⇒ layer thickness **ΔR₂ ∝ f_κ
  at fixed dMdt**; folding in the seed ⇒ f_κ^(5/7)), T-curvature ODE (`:406`, ∝ 1/(f_κ·C_th·T^(5/2))). It does
  **not** multiply `L_cool`: `get_dudt(t,n,T,φ)` is integrated over the (now thicker) structure, so **θ =
  L_cool/L_mech emerges**. The seed law is **verified vs measurement**: dMdt(f_κ=2)/dMdt(f_κ=1) = 1.2175 at
  the seed vs 2^(2/7)=1.219 (≈0.1%). Side effect: dMdt
  rises too (a faithful El-Badry κ_eff would *suppress* evaporation) ⇒ f_κ is a **structural probe**.
  Artifacts: `fkappa_definition.png` (+ `data/make_fkappa_definition.py`).
- **Measured back-reaction [data]:** at matched `t` on the stiff dense edge (`f1edge_hidens`), `f_κ=2` raises the
  resolved cooling `bubble_LTotal` **×1.23–1.38**, moving the loss-ratio proxy **+0.05–0.10** toward the trigger.
  Artifacts: `data/kappa_backreaction.csv` + `kappa_backreaction.png` (full table in `KAPPA_EFF_SCOPING.md` §6a).
- **Calibration — how much f_κ, measured on full runs (3 configs) [data]:** developed θ at cloud dispersal vs
  f_κ for compact (`simple_cluster`) / mid (`midrange_pl0`) / diffuse (`f1edge_lowdens`): **θ(f_κ=1) =
  0.67 / 0.61 / 0.17** (all **measured**), all below the obs/3D ~0.9 and the 0.95 `cooling_balance` trigger.
  **f_κ to fire (θ→0.95): ≈4 (compact — bracketed, it fires at f_κ=4) / ≈5–6 (mid, extrapolated) / ≈60
  (diffuse, extrapolated)** (⛔ #3 — the ≈60 is dead per §10, a blowout-metric artifact; measured multiplier
  f_fire = 4) — steeply density-dependent (only compact reaches 0.95 within the measured f_κ≤4
  grid). So **at f_κ=1 the under-cooled clouds stay below ~0.9 and never fire; they need much higher f_κ.** Artifacts:
  `data/kappa_blowout_calibration.csv` + `kappa_blowout_calibration.png`.
- **The merge:** κ_eff is the **mechanism**; `θ(n_H)` (El-Badry `λδv`=κ_eff + Lancaster ≈0.9–0.99) is the
  **target**; `f_κ(properties)` is the knob. The earlier "`θ_target` vs κ_eff" split was a false dichotomy
  (target vs mechanism). The remaining work is **calibrating f_κ(properties)** so emergent θ → target — *no new
  production code*, reusing this knob.
- **Negative results that confirm the mechanism [data]:** `FM1` (imposing `dMdt` — refuted; `dMdt` pinned by
  `v(R1)=0`) and `FM1b` (an interior loss-integrand term — El-Badry *sign* but negligible magnitude, because
  `dMdt` is front-anchored) ruled out the two *wrong* knobs and point back to κ_eff. They also show the
  full El-Badry **evaporation-suppression is an optional high-fidelity bonus** the 1D structure resists — not
  in the goal. Artifacts: `data/fm1_rootcheck.*`, `data/fm1b_evapsign.*`; design in `RUNGB_SCOPING.md`.

## 6a. [data] Does PdV ALONE trigger the transition? — `ebpeak` measured (2026-06-28)

The workstream's founding question, settled on the actual code path. Two runs with
`transition_trigger=cooling_balance,ebpeak` **active** at `f_κ=1`
(`runs/params/cal_{compact,diffuse}__ebpeak.param`) both ran to `stop_t` and ended on `STOPPING_TIME` with
shadow `ebpeak_t=None` — **`ebpeak` never fired**.

- **The PdV-inclusive ratio `(Lloss+PdV)/Lgain` peaks BELOW the 1.0 threshold, then declines:** compact peaks
  **0.912 @t=0.12** (just past dispersal); diffuse peaks **0.862 @t=1.06** then falls as the bubble
  **re-accelerates** in the low-density ISM (the diffuse run reached t=1.5, R2=191 pc, v2=168 km/s, Eb still
  *growing* — net energy never turns over). This **corrects** an earlier linear extrapolation that wrongly
  predicted diffuse would fire ~1.2–1.3 Myr (the ratio is non-monotone).
- **PdV is the dominant sink** (PdV/Lgain = 0.20 compact / 0.46 diffuse) and lifts the balance from
  radiative-only (0.66 / 0.17) to ~0.86–0.91 — it **narrows** the gap but does not close it; a cooling boost is
  still required to fire.
- **Cooling↔PdV trade-off caps the PdV path:** the PdV-inclusive peak is nearly `f_κ`-insensitive for diffuse
  (0.848→0.849→0.853 across f_κ 1,2,4 — flat) while the radiative ratio nearly doubles (0.165→0.297). ⇒ for
  diffuse, the only path to fire is radiative `cooling_balance` (f_κ~60), **not** `ebpeak`; PdV helps the
  *compact* case (fires by f_κ~2–4). (2026-07-02: the ~60 is dead per §10 — the diffuse GMC fires at
  f_mix=4; the ebpeak-vs-cooling conclusion survives.) **Net:** PdV (`ebpeak`) is an assist for transition
  *timing*, **not a substitute** for `κ_eff` (cooling *magnitude*) — complementary, downgraded from "PdV
  alone fixes f_κ~60."
  Artifacts: `data/ebpeak_trigger_test.csv` + `ebpeak_trigger_test.png` (+ `data/make_ebpeak_trigger_test.py`).
  No production code touched (default `transition_trigger=cooling_balance` unchanged).
- **8-config coverage [data]:** the f_κ=1 conclusion above is **2 live configs**, but it **generalizes to all 8**
  via the earlier frozen-trajectory screen (`make_ebpeak_8config_xcheck.py` → `ebpeak_8config_xcheck.csv/png`):
  all **6 normal** configs peak at PdV-inclusive **0.85–0.92** and never fire (only heavy-5e9 `fail_repro` 1.57
  and the `small_1e6` control 1.11 do; `large_diffuse_lowsfe` 1.02 barely, post-blowout). **Live-vs-frozen
  agrees to the digit** (simple_cluster live 0.911 == frozen 0.911). The f_κ-*dependence* (trade-off) is
  live-only and extended to `mid`=midrange_pl0 (running) + `dense`=small_dense_highsfe (stalled — nCore 1e6 is
  numerically stiff; frozen point used). HPC-deferred for the remaining configs.

## 8. [data] Stage-A shadow — El-Badry θ imposed end-to-end on 9 configs (2026-06-30)

> 📖 **Illustrated walkthrough:** `ELBADRY_THETA_STORY.html` (6 figures, `make_elbadry_story_figs.py` →
> `fig/elbadry_f{1..6}_*.png`) narrates the closed form, what §2/§3 impose & check, the physics, and the §8b
> reversal. Regenerate with `python docs/dev/transition/pdv-trigger/make_elbadry_story_figs.py` (reads only the
> committed CSVs).

First end-to-end test of the capstone (`THETA_ELBADRY_SPEC.md`). `data/_theta_elbadry_runner.py`
monkeypatches `effective_Lloss_from_params` (in BOTH `get_betadelta` and `run_energy_implicit_phase`) to
the El-Badry analytic θ — `θ = A_mix·√(λδv·n_amb)/(11/5 + A_mix·√(λδv·n_amb))`, A_mix=3.5, λδv=3,
θ_max=0.99 — **without touching production code**, then runs each config in a separate process to ≥5 Myr
with `transition_trigger=cooling_balance,ebpeak`. Harvest: `data/harvest_shadow.py` →
**`data/shadow_te_fate.csv`** (the table below). `n_amb` = local cloud density at the shell,
`get_density_profile(R2)·ndens_au2cgs`.

| config | n_core (cm⁻³) | mCloud | sfe | θ imposed | fire t (Myr) | fate | end (t, R2, v2) |
|---|---|---|---|---|---|---|---|
| simple_cluster | 1e5 | 1e5 | 0.30 | 0.990 | 0.009 | **SHELL_COLLAPSED** | 0.14 Myr, 0.99 pc, −0.1 |
| pl2_steep | 1e5 | 1e6 | 0.10 | 0.990 | 0.011 | **SHELL_COLLAPSED** | 0.06 Myr, 0.94 pc, −0.3 |
| be_sphere | 1e4 | 1e6 | 0.05 | 0.990 | 0.015 | **SHELL_COLLAPSED** | 2.30 Myr, 1.5 pc, −23 |
| midrange_pl0 | 1e4 | 1e6 | 0.10 | 0.990 | 0.017 | **SHELL_COLLAPSED** | 1.42 Myr, 1.5 pc, −28 |
| large_diffuse_lowsfe | 100 | 1e7 | 0.01 | 0.965 | 0.052 | **SHELL_COLLAPSED** | 14.3 Myr, 1.5 pc, −122 |
| small_1e6 | 100 | 9e5 | 0.10 | 0.965 | 0.052 | STOPPING_TIME | 10.0 Myr, 254 pc, +28 |
| diffuse_probe | 10 | 9e5 | 0.10 | 0.897 | 0.154 (ebpeak) | STOPPING_TIME | 6.0 Myr, 139 pc, +28 |
| fail_repro | 100 | 4.5e9 | 0.10 | 0.965 | — | energy_collapsed | 0.003 Myr (pre-existing heavy-cloud break) |
| small_dense_highsfe | 1e6 | 1e4 | 0.50 | 0.990 | — | CRASHED_EARLY | 0.004 Myr (pre-existing β-δ solver stiffness) |

**Findings:**

1. **§6 `max(resolved, target)` gate is SAFE — resolved-wins 0/N on all 9 configs.** El-Badry's θ is ≥
   TRINITY's native resolved θ in *every* call, so the `max()` never selects the resolved term: imposing
   θ_target here is operationally identical to direct θ assignment. This clears the SPEC §6 gate. **It also
   means the patch raises θ above baseline everywhere** — a key fact for the isolation read (point 5).
2. **θ(n) is monotone and the firing threshold behaves as designed.** n=10 → θ=0.897 (<0.95, so
   `cooling_balance` never trips — fires only later via `ebpeak`); n=100 → 0.965; n≥1e4 → 0.99 (capped).
   The n_fire≈48–50 cm⁻³ threshold is confirmed: the n=10 cloud stays energy-driven until ebpeak, n≥100
   trips cooling_balance promptly.
3. **Fate splits by cluster power vs cloud binding, NOT by θ alone:**
   - *Dense compact* (n≥1e4, θ=0.99): fire at t<0.02 Myr → **SHELL_COLLAPSED**. 99% cooling strips the
     thermal support and the shell stalls then **recollapses** — v2 goes negative, R2 decreases, `isCollapse`
     is set, and the run ends when R2 falls below `coll_r`=1 pc (`run_transition_phase.py:772/789`,
     `run_momentum_phase.py:842`). SHELL_COLLAPSED is **endcode 4 = a CLEAN physical fate** (range 0–9), the
     code's label for shell recollapse — not a numerical error. (Confirmed in the data: collapsed configs end
     with v2<0: be_sphere −23, midrange_pl0 −28, large_diffuse_lowsfe −122 km/s.)
   - *Diffuse, well-powered* (small_1e6: n=100, sfe=0.1, mCloud=9e5): STOPPING_TIME, expands to 254 pc.
   - *Diffuse, under-powered* (large_diffuse_lowsfe: n=100 but sfe=0.01 in a 1e7 cloud): fires early yet
     SHELL_COLLAPSES only at **t=14.3 Myr** — the cluster is too weak to hold a 1e7 cloud open; late
     recollapse is physically expected.
   - *Very diffuse* (diffuse_probe: n=10): θ<0.95, stays energy-driven until ebpeak fires at 0.15 Myr, then
     expands healthily (139 pc) to stop-time.
4. **The two non-results are at unphysical extremes and are pre-existing, NOT patch-induced.** fail_repro
   (mCloud 4.5e9 — absurd) energy-collapses at t=0.003 *before any transition*; small_dense_highsfe
   (nCore 1e6) hits the known β-δ `MonotonicError` solver stiffness at t=0.004, also before any transition.
   Neither reaches the patched code path long enough for θ to matter.
5. **The SHELL_COLLAPSE IS patch-induced — resolved from committed data, not new runs.** The intended
   isolation was a stock-trinity baseline (`data/_baseline_runner.py`), but dense baselines are hours-scale
   in the stiff early implicit solve and the container restarts repeatedly killed them. **We don't need
   them:** the committed frozen-trajectory cross-check (`§6a` + `data/ebpeak_8config_xcheck.csv`) already
   gives stock TRINITY's *native* radiative θ — the quantity `cooling_balance` actually tests (PdV-exclusive).
   Per §6a it **peaks at ~0.66 for compact/dense clouds** (0.17 diffuse), *far* below the 0.95 firing
   threshold. So stock TRINITY **never fires `cooling_balance` for these dense clouds and keeps them
   energy-driven** — consistent with the whole reason this workstream exists (TRINITY under-cools). Imposing
   El-Badry θ=0.99 pushes native 0.66 → 0.99, and *that* is what collapses them. resolved-wins=0 (point 1)
   is the same statement from the shadow side: θ_elbadry always exceeds native θ.
6. **So the real question is physical, and it belongs to the maintainer:** El-Badry/Lancaster say θ≈0.9–0.99
   *is* correct for dense clouds (high n → θ→1), so raising θ is faithful to the literature. But in
   El-Badry's own sims a θ→1 bubble is momentum-driven and **still expands** — it does not recollapse.
   TRINITY instead drives these to SHELL_COLLAPSED. Two readings remain, and they need a physics call:
   (a) **physical for these configs** — the clusters are modest (3e4–1e5 M⊙ vs dense cores), so genuine
   recollapse is plausible (Lancaster doesn't test this weak-cluster/dense corner); or (b) **artifact** —
   either θ_max=0.99 is too aggressive (a softer cap ~0.95 leaves ~2× more driving luminosity), *or*
   TRINITY's momentum phase mishandles a near-zero-thermal-energy bubble and collapses it when El-Badry
   would keep it expanding. Distinguishing (a) from (b) is the **one open Stage-A item**; a θ_max sweep
   (0.80/0.85/0.90/0.95/0.99) on the fast dense configs would separate "cap too high" from "momentum phase
   recollapses regardless," and is far cheaper than the baseline full-runs. **RUNNING (2026-06-30):** sweep on
   pl2_steep + simple_cluster, `outputs/sweep_tmax/tmax_*`; results → §8a below. NB θ<0.95 caps can't trip
   `cooling_balance` at all (firing needs θ≥0.95), so the sub-0.95 rows also test whether these clouds then
   stay energy-driven.

**Stage-A verdict:** the mechanism works end-to-end, the numerical gate (§6 max) is clean, θ(n) and the
firing threshold behave, and the dense-cloud SHELL_COLLAPSE is confirmed *patch-induced* (not stock, not a
solver bug). What remains before Stage B is a **physics decision** — is early collapse of θ≈0.99 dense
clouds correct, or does θ_max need softening / the momentum phase need scrutiny? Do **not** wire the
production `theta_elbadry` mode until that decision is made. Artifacts (committed):
`data/_theta_elbadry_runner.py`, `data/_baseline_runner.py`, `data/harvest_shadow.py`,
`data/shadow_te_fate.csv`.

## 8a. [data] θ_max sweep — the cap is NOT the lever; dense-cloud recollapse is intrinsic (2026-06-30)

The §8 point-6 discriminator, run. Swept **θ_max ∈ {0.80, 0.85, 0.90, 0.95, 0.99}** on the two fast dense
configs (pl2_steep n=1e5, simple_cluster n=1e5), same harness/trigger (`cooling_balance,ebpeak`), λδv=3, to
5 Myr. Harvest → **`data/sweep_tmax_fate.csv`**.

| θ_max | config | θ used | fire t (Myr) | trigger | end (t, R2, v2) | fate |
|---|---|---|---|---|---|---|
| 0.80 | pl2_steep | 0.80 | 0.012 | ebpeak (PdV) | 0.057, 0.96 pc, −0.3 | SHELL_COLLAPSED |
| 0.85 | pl2_steep | 0.85 | 0.013 | ebpeak (PdV) | 0.057, 0.97 pc, −0.3 | SHELL_COLLAPSED |
| 0.90 | pl2_steep | 0.90 | 0.010 | ebpeak (PdV) | 0.055, 0.90 pc, −0.1 | SHELL_COLLAPSED |
| 0.95 | pl2_steep | 0.95 | 0.010 | cooling_balance | 0.055, 0.90 pc, −0.1 | SHELL_COLLAPSED |
| 0.99 | pl2_steep | 0.99 | 0.011 | cooling_balance | 0.056, 0.94 pc, −0.3 | SHELL_COLLAPSED |
| 0.80 | simple_cluster | 0.80 | 0.010 | ebpeak (PdV) | 0.130, 0.99 pc, −0.1 | SHELL_COLLAPSED |
| 0.85 | simple_cluster | 0.85 | 0.009 | ebpeak (PdV) | 0.133, 0.98 pc, −0.1 | SHELL_COLLAPSED |
| 0.90 | simple_cluster | 0.90 | 0.010 | ebpeak (PdV) | 0.129, 1.00 pc, −0.1 | SHELL_COLLAPSED |
| 0.95 | simple_cluster | 0.95 | 0.019 | cooling_balance | 0.184, 0.97 pc, −4.3 | SHELL_COLLAPSED |
| 0.99 | simple_cluster | 0.99 | 0.009 | cooling_balance | 0.135, 0.99 pc, −0.1 | SHELL_COLLAPSED |

**Verdict — θ_max is not a useful knob here, and reading (b)"cap too aggressive" is refuted:**

1. **All 10 runs recollapse**, at essentially the *same* fire time (~0.01 Myr) and collapse time (~0.055 Myr
   pl2_steep / ~0.13 Myr simple_cluster), **independent of θ_max**. Lowering the cap from 0.99 to 0.80 does
   not save the cloud and barely shifts the timing.
2. **Below 0.95 the transition still fires — via `ebpeak` (PdV), not `cooling_balance`.** Imposing θ≥~0.80
   raises L_loss to θ·L_mech, which is already enough to drive `Edot_from_balance≤0` and trip ebpeak at
   t~0.01 Myr. (This is why it fires where stock doesn't: stock's native radiative θ~0.66 leaves the
   PdV-inclusive ratio at ~0.91<1, so stock ebpeak never fires — §6a.) So *any* physically-plausible imposed
   θ for a dense cloud (El-Badry says 0.9–0.99) transitions it, and it then recollapses.
3. **Therefore the recollapse is intrinsic to these dense compact clouds transitioning — not an artifact of
   the specific cap.** The remaining question from §8 point 6 collapses to a single fork, now cleanly posed:
   is the recollapse (a) **physical** — a dense compact core with a modest cluster (3e4–1e5 M⊙) genuinely
   recaptures its shell once the bubble stops being energy-supported (TRINITY has a *dedicated clean fate*,
   SHELL_COLLAPSED, for exactly this) — or (b′) a **momentum/transition-phase fidelity** issue, where TRINITY
   recollapses a bubble that El-Badry's θ→1 sims keep weakly expanding? **This is no longer a trigger-design
   question** (the trigger correctly decides *when* to leave energy-driven); it is a question about the
   momentum phase's treatment of a near-zero-thermal-energy shell, which is **outside this workstream's
   scope**.

**Bottom line for Stage B:** the `theta_elbadry` trigger works as designed and θ_max needs no tuning — pick
0.95 or 0.99 (they behave identically for dense clouds; 0.99 matters only for the *magnitude* of (1−θ) driving
on clouds that survive). Whether dense compact clouds *should* recollapse is a physics call for the maintainer
that is independent of wiring the trigger. If the maintainer accepts SHELL_COLLAPSED as the correct fate for
weak-cluster/dense-core configs (the likely reading), **Stage A is clean and Stage B can proceed.** Artifacts:
`data/sweep_tmax_fate.csv`, `outputs/sweep_tmax/tmax_*/`.

> 🛑 **§8/§8a partly SUPERSEDED by §8b (2026-07-01) — the code changed under us.** §8/§8a ran on **pre-PR#715**
> code. Main has since merged `bugfix/high-mass-cluster-transition-without-ebpeak` (PR #715), which **routes a
> finite `Eb≤0` collapse to the momentum phase** instead of dead-stopping. Re-running on the merged code flips
> the high-mass verdict — see §8b. The §8/§8a **firing/threshold/max-gate results still hold** (they're about
> the trigger algebra); the **fate conclusions for massive/dense clouds do NOT** (the momentum handoff, plus a
> regime error in applying El-Badry's θ, changes them).

## 8b. [data] Re-run on the merged high-mass handoff (PR #715) — imposing El-Badry θ REVERSES the fix (2026-07-01)

The maintainer merged the high-mass energy→momentum handoff to `main` (PR #715,
`bugfix/high-mass-cluster-transition-without-ebpeak`; `HIMASS_HANDOFF_PLAN.md`). Two code changes touch the
phases the Stage-A shadow exercises: (1) phase 1b now routes a **finite `Eb≤0`** collapse to the momentum phase
(`classify_energy_collapse` + `ENERGY_HANDOFF_FLOOR=1e3`) instead of the `ENERGY_COLLAPSED` dead-stop; (2) phase
1a gains a `cooling_balance` parity check. I **merged main into this branch** (code auto-merged clean; my Pb-fix
and the routing coexist — different regions) and re-ran on the merged code. Artifacts:
`data/newcode_default_vs_theta.csv`, `outputs/{baseline_v2,shadow_te_v2}/`.

**The decisive contrast (same configs, merged code, default trigger vs El-Badry θ imposed):**

| config (mass, n) | DEFAULT (stock trigger) | θ_elbadry imposed (λδv=3, θ_max=0.99) |
|---|---|---|
| fail_repro (5e9, n=1e2) | **large_radius (exit 2)** — energy→implicit→**momentum**, expands to the 500 pc stop radius, v2=+37 | **velocity_runaway (exit 50)** — collapses inward, v2=**−500** pc/Myr, R2=5 pc, dies in `transition` |
| pl2_steep (1e6, n=1e5) | **expanding** (v2=+23, R2=1.2 pc, healthy) when stopped in the stiff implicit solve | **velocity_runaway (exit 50)** — v2=**−500**, R2=0.07 pc |

**Findings:**

1. **The maintainer's fix works and is verified here:** `fail_repro` — the canonical diffuse-massive dead-stop
   (was `ENERGY_COLLAPSED` at t=0.003 Myr) — now runs cleanly to **large_radius (500 pc)** on the default path.
2. **Imposing El-Badry θ REVERSES it.** With `theta_elbadry` forcing `L_loss=θ·L_mech` (θ=0.965–0.99), the
   *same* clouds violently recollapse (v2 pinned at the −500 pc/Myr `MAX_VELOCITY_COLLAPSE` cap, R2→~0). What
   §8 recorded as `SHELL_COLLAPSED` on the old code is now `velocity_runaway` (or a near-zero-radius solver
   stall) on the new code — but the physics is the same and now it is **unambiguously the θ-imposition
   causing it, since the default path expands these very clouds.**
3. **Why it's a regime error, not just aggressive tuning (ties to `HIMASS_HANDOFF_PLAN.md` §1):** the
   maintainer verified the high-mass turnover is **PdV / inertial-loading driven, NOT radiative** — for
   `fail_repro`, radiative is ~1% of L_mech while PdV/L_mech≈1.4. El-Badry's θ is a **radiative** ratio
   (L_cool/L_mech) from SN-driven sims. Imposing θ=0.99 there injects a **fake radiative sink of 0.99·L_mech
   on top of the real PdV sink** — double-draining the bubble and crashing it inward. PdV already enters the
   energy budget separately (`Edot_from_balance = Lgain − Lloss − 4πR2²v2·Pb`), so the imposed radiative θ is
   double-counting the loss in exactly the regime where PdV dominates. **The θ_max sweep (§8a) missed this
   because it only ran on the old dead-stop code and read the terminal *label*, not the default contrast.**

**Consequence for the plan — the `theta_elbadry` SPEC needs a regime gate, and Stage B is NOT ready.** As
specified (impose θ=A_mix√(λδv·n) on *every* cloud via `effective_Lloss`), the mode **re-breaks precisely the
massive clouds PR #715 just fixed**. El-Badry's θ is only physical where **radiative cooling actually dominates**
(the dense/compact regime where `cooling_balance` engages natively) — it must **not** be applied to the
**PdV/inertia-dominated** massive/diffuse clouds, which the momentum handoff already carries. Options for the
revised spec: (a) gate `theta_elbadry` off when PdV/L_mech ≳ 1 (or when radiative ≪ L_mech), deferring those
clouds to the handoff; (b) restrict the imposed θ to the radiative channel only and let PdV + the handoff do the
rest; (c) drop θ-imposition for high-mass and keep it only as the diffuse-end cooling correction it was
originally scoped for. This must be resolved **before** any production wiring. Prior "Stage A clean → Stage B"
(end of §8a) is **retracted for the massive-cloud regime.** **→ RESOLVED in §8c (2026-07-01):** the answer is
not to *gate* enforcement (options a/b/c above) but to **stop enforcing** — boost the mechanism (f_κ) and let θ
emerge; enforcement (`theta_elbadry`) becomes an opt-in override.

## 8c. [data] Direction corrected — θ is an OUTPUT; f_κ reinstated, θ_elbadry demoted to opt-in (2026-07-01)

§8b framed the fix as "gate `theta_elbadry` off in the PdV regime." Prototyping that gate
(`data/_theta_elbadry_gated_runner.py`, `data/gate_prototype.csv`) worked — and in working, it revealed that
the whole *enforce-θ* framing is the wrong primitive. Maintainer steer: **θ should be an output of the solved
bubble, not an input you set.**

**The gate prototype (measured):**

| config | variant | trigger | gated calls | fate |
|---|---|---|---|---|
| fail_repro (5e9, n=1e2) | θ imposed (§8b) | cooling_balance,ebpeak | 0 | **velocity_runaway** (recollapse) |
| fail_repro | θ **gated** (PdV/L_mech>0.7) | cooling_balance | 69/84 | **large_radius** (expands to 500 pc) |
| fail_repro | θ **gated** | cooling_balance,**ebpeak** | 57/72 | **large_radius** (expands to 500 pc) |
| pl2_steep (1e6, n=1e5) | θ **gated** | cooling_balance | 1/46 | shell_collapsed (radiative regime → gate barely fires → θ kept) |

**Findings:**

1. **The gate fixes the reversal:** deferring the PdV-dominated cloud (`fail_repro`, PdV/L_mech peaks 2.65)
   makes it expand to 500 pc like the default path, instead of `velocity_runaway`. And the **θ-gate ALONE does
   it** — `ebpeak` on/off is irrelevant (both → large_radius). So the culprit was never `ebpeak`; it was the
   imposed θ over-cooling. The gate is **selective**: `pl2_steep` (radiative regime, PdV<0.7) is barely gated,
   so it is unchanged.
2. **But the gate is just re-deriving, by hand, what f_κ gives for free.** The double-counting in §8b is a
   *direct symptom of enforcing θ*: `L_loss=θ·L_mech` is blind to whether the loss is radiative or PdV, so on a
   PdV-dominated cloud it injects a fake radiative sink on top of the real PdV sink. The Rung-A **`multiplier`**
   mode (`L_loss = L_leak + f_κ·L_cool`) scales **only the radiative channel** — in a PdV-dominated bubble
   radiative is ~1% of L_mech, so `f_κ·L_cool` *physically cannot* over-drain it. **No regime error, no gate
   needed.** The gate is a symptom-patch; f_κ removes the disease.
3. **Corrected direction (see `PLAN.md` ⭐⭐ + top ledger):** Rung A (`multiplier`/f_κ, already shipped, gated
   default-off) is PRIMARY and θ **emerges**; El-Badry/Lancaster are the **calibration target** for that emergent
   θ (pick f_κ(n) so the *solved* θ lands in-band), **not** an enforced value. Set f_κ at a **physical** value
   and **accept diffuse route-a non-transition** (maintainer: "diffuse clouds may never enter momentum — the
   physics never allows it") — do NOT crank f_κ to ~60 to force it (`F_KAPPA_FUNCTIONAL_FORM.md` §11–13) —
   and §10 (2026-07-02) showed no cranking is needed: the diffuse GMC fires at the physical f_mix=4.
   Massive/PdV clouds ride the PR #715 `Eb≤0→momentum` handoff, untouched by θ. **`theta_elbadry`/`theta_target`
   remains as a documented opt-in override** (`THETA_ELBADRY_SPEC.md`) for users who explicitly want forced
   cooling — the gate/§8b caveat is why it is not the default.
4. **Rung A (scalar f_κ, reinstated) ≠ Rung B (structural κ_mix, still SHELVED).** The reinstatement is of the
   scalar multiplier only; the structural κ_mix injection remains numerically unstable (`KMIX_SELFCONSISTENT.md`).

Artifacts: `data/_theta_elbadry_gated_runner.py`, `data/gate_prototype.csv`,
`outputs/{shadow_gate,shadow_gate_ebpeak}/`.

## 8d. [data] The diffuse-config "hang" is a PERFORMANCE cliff, not a stall/bug — diagnosed (2026-07-01)

Validating the §14 route-a end (n=100, `multiplier` mode) hit what looked like a stall at **t≈0.003 Myr** (the
fixed 1a→1b handoff — as the maintainer noted, 1a ends ~3e-3 Myr). A DEBUG investigation with an f_κ sweep
(`data/_fkappa_validation_runner.py` with `LOG_LEVEL=DEBUG`, ≥6.5 min/run) **overturned three of my initial
claims** — logging them here because the retractions are the finding:

**Measured throughput (`large_diffuse_lowsfe`, 6.5 min wall each; the loop logs `[Implicit] t=` per segment):**

| f_κ | segments in 6.5 min | sim-t reached | bubble-solves/seg (first 5 seg) |
|---:|---:|---:|---:|
| 1 (default) | 47 | 0.059 Myr | 144 |
| 2 | 41 | 0.041 Myr | 175 |
| 8 | 14 | 0.0078 Myr | 214 |

**Findings (each correcting an earlier hypothesis):**

1. **It is NOT a stall / hang / infinite loop — it is slow forward progress.** Every f_κ advances (47/41/14
   segments); I first called it a "stall" because I checked while it was still crawling through the expensive
   early segments and the INFO log rounds to the t=0.003 entry.
2. **It is NOT a convergence failure.** The beta-delta `hybr` solver converges *perfectly* every segment
   (`beta-delta hybr result: g=1e-13…1e-17, converged=True, ier=1, evals≈20`). So "accept ~1e-4 and move on"
   does not apply — nothing marginal is being rejected; it lands at ~1e-15. **(Retracts the "implicit-solve
   non-convergence / no physical root" claim.)**
3. **It is NOT the `min_T` guard.** The 513 "Rejected. min T: 29999.99…" lines are boundary transients whose
   rejection penalty `residual·(3e4/min_T)²` = **0.999993 ≈ 1.0** — benign. (My "relax the guard" test was also
   invalid: lowering `_T_INIT_BOUNDARY` moves the IC *and* the guard together, so the transient just follows.)
   **Red herring.**
4. **It is NOT f_κ-specific — answering the maintainer's Q1.** f_κ=1 (default, no boost) hits the *same* t=0.003
   handoff and grinds too; it just clears the early segments faster. So there are **two** compounding effects:
   (a) **config-intrinsic slowness** — even f_κ=1 only reaches t=0.059 Myr in 6.5 min → **~11 h to reach 6 Myr**
   (the "failed-large-clouds" class, mCloud=1e7 diffuse); and (b) a **cooling-boost cost concentrated in the
   early implicit segments** at the small-R2 / fast-v2 (262 km/s) handoff corner — f_κ=8 does ~⅓ the segments
   and reaches ~7× less sim-time than f_κ=1. Past that corner the per-segment cost converges (f_κ=1 and f_κ=2
   both ~7 bubble-solves/seg overall). **(Retracts the "f_κ=8 stiffens the structure enough to trip it" framing
   — the config is slow at f_κ=1 too.)**

**Root cause:** a *performance* cliff — each implicit segment at this early, small-radius, fast-expanding,
diffuse state costs many (7–50) bubble-structure `dMdt` fsolves while `dt` is small (5e-4, shrinking), so
sim-time crawls; a cooling boost multiplies the early-segment cost. Not a correctness bug in the solver.

**Why the boost slows it (precise mechanism — NOT "stiffening the ODE"):** the `multiplier` knob **does not
enter the bubble-structure ODE at all** — it scales `L_cool = bubble_LTotal` *after* the structure is solved
(`get_betadelta.py:473`), feeding only the residual / energy-ODE / trigger; the structure ODE's conduction is
`cooling_boost_kappa` (`bubble_luminosity.py:291/370/406`), untouched here. So f_κ=8 cannot "stiffen" the
structure. It is slower because 8× larger `L_loss` → `dEb/dt` (=Edot) much more negative → Eb evolves fast →
the **adaptive stepper shrinks dt** (0.0012→0.00034 Myr/seg, ~3.5×) → more, smaller segments; plus a modest
(~1.5×) rise in structure-solve evals as the shifted beta-delta solution lands in a harder region.

> ⚠️ **KNOB ERROR flagged (2026-07-01): these runs used `multiplier`, but §14's leverage/θ₀ were fit with
> `cooling_boost_kappa` — different knobs (see `PLAN.md` ⭐⭐ KNOB CORRECTION).** So the §14 validation table
> above (`θ_max=1.33/1.01`) is for `multiplier` and does **not** validate the `kappa`-based §14 calibration;
> re-run the validation with `cooling_boost_kappa` at the calibrated f_κ. `kappa` *does* enter the structure ODE,
> so its throughput/robustness behaviour on the diffuse handoff may differ from the `multiplier` runs here.

> 🪵 **Logging nuisance (worth a one-line fix): the `"Rejected. min T: 29999.99…"` DEBUG line is misleading.**
> It fires for every benign boundary transient (penalty ≈1.0) and misled this very investigation. The **guard**
> is correct (it properly penalizes a *real* sub-floor min_T); only the **log** is noisy for the FP-undershoot
> case. Fix: only log when the penalty is actually significant (min_T meaningfully below `_T_INIT_BOUNDARY`), or
> reword/downgrade it. Production `bubble_luminosity.py:345` — a logging-quality change (behaviour-neutral),
> flagged for the maintainer.

**Relevance to the plan:** the emergent-θ mechanism itself is *correct* here (beta-delta converges); the issue
is that **boosted diffuse runs are computationally impractical to carry to ≥5 Myr** in this environment — which,
with the physics (§14 route-a) and the intrinsic mCloud=1e7 slowness, is a further reason to **cap f_κ low at
the diffuse end and accept route-a** (superseded on HPC, 2026-07-02: the boosted diffuse arms completed AND
fired — §10; the in-container impracticality stands). Making these runs fast is a bubble-structure/`dt`
performance item, out of this workstream's scope.

**Size-control (settles mass-vs-handoff):** `small_1e6` (n=100 but mCloud=**9e5**, 10× smaller than
large_diffuse's 1e7) at f_κ=1 hits the **identical handoff state** (R2=1.3819 pc, v2=262.47 km/s, t=0.00293 —
the outer cloud mass doesn't touch the inner bubble yet) and progresses at a **near-identical rate** (42 seg →
t=0.044 Myr vs large_diffuse's 47 seg → 0.059 Myr in 6.5 min). So the slowness is the **diffuse (n=100)
early-implicit handoff corner** (small R2, fast v2), **NOT the 1e7 cloud mass**. Both diffuse configs *do* finish
given enough wall-time (their `theta_target` shadows reached 10/14 Myr, §8) — they are slow, not stuck.

Artifacts: `data/_fkappa_validation_runner.py` (θ_max observer + `LOG_LEVEL`), `data/_minT_tol_confirm_runner.py`
(the retracted min_T test), `outputs/{fkappa_val,fkappa_debug,fk_compare_1,fk_compare_2,fk_compare_8}/`.

## 8e. [data] Correct-knob (`cooling_boost_kappa`) validation — it BREAKS DOWN at f_κ=8; reframes the knob choice (2026-07-01)

Re-running the §14 validation with the **correct** knob (`cooling_boost_kappa` — the structural conduction boost
the §14 leverage/θ₀ were fit on, not the `multiplier` I mistakenly used; `data/_kappa_validation_runner.py`,
`cooling_boost_mode='none'` so θ_emergent = bubble_Lloss/Lmech). Result: **the structural knob does NOT cleanly
validate — at the physical f_κ=8 it breaks down.**

- **`simple_cluster` (n=1e5), kappa=8:** from implicit segment ~6 the beta-delta solver hits **"no physical
  (dMdt>0) root"** — kappa's boosted conduction drives dMdt **negative** (the evaporative-flux side-effect the
  registry warns of: *"raises the evaporative mass flux … a structural probe, not the final model"*). The solver
  **holds the last physical dMdt and the state freezes**, so the *physical* emergent θ (bubble_Lloss/Lmech from
  `dictionary.jsonl`, the accepted state) sticks at **~0.53 — it does NOT fire**. Nothing like the `multiplier`
  run's θ_max≈1.33 (§14). So **the §14 multiplier validation does NOT transfer to the knob the calibration was
  actually fit on.**
- **`be_sphere` (n=1e4), kappa=8:** same story — emergent θ ≤ **0.48**, **does NOT fire** (vs `multiplier`
  θ_max=1.006, §14). So on *both* dense configs the structural knob gives a **much lower** emergent θ than the
  multiplier. Physical reason: boosting the conduction coefficient raises the **evaporative mass flux** (more
  cool gas mixed in), and that back-reaction *damps* the net radiated L_cool — it does not simply 8× it. This is
  exactly the coupling El-Badry says a faithful κ_eff must **suppress** (registry note); `cooling_boost_kappa`
  does the opposite, so θ stays moderate. The faithful version (κ_mix, Rung B) would suppress evaporation — but
  it is SHELVED (unstable).
- **`small_1e6` (n=100, diffuse), kappa=8:** physical θ≈**0.25**, no fire (also stuck in phase 1a). So the three
  configs give a clean **monotonic emergent θ(n) = 0.25 / 0.48 / 0.53 for n = 100 / 1e4 / 1e5** — all early-time
  (phase 1a / frozen, well before the blowout peak), all below the 0.95 threshold. The **density ordering is the
  encouraging part**: emergent θ rises with density exactly as the "let θ emerge, route-a falls out" picture
  predicts — but the *magnitudes* are far below `multiplier`'s and (because they're pre-peak + kappa breaks/stalls)
  we cannot say whether the dense ones would eventually fire.
- **kappa is also far SLOWER at ANY f_κ:** it enters the bubble-structure ODE, so every structure solve is
  costlier. **kappa=2 (stable — 0 non-physical dMdt, so f_κ=8 was simply too high) STILL timed out in phase 1a**
  (t=0.00291, 0 implicit segments in 6.5 min), physical θ≈0.49 — where the same config under `multiplier`
  reached the trigger and fired by t=0.009. So kappa is impractical here **regardless of f_κ**: fragile at 8,
  slow at all values.
- **⚠️ Methodology correction:** the runner's `theta_max` observer over-counts — it records *every*
  `effective_Lloss` call, incl. the solver's **non-physical trial (β,δ) points** (it reported a bogus
  θ_max=3.223). The trustworthy emergent θ is `bubble_Lloss/Lmech_total` at the **accepted** segments in
  `dictionary.jsonl` (the value that feeds the trigger) — that is the ~0.53 above. This caveat applies to the
  §14 `multiplier` θ_max numbers too (re-harvest from the dictionary before quoting them as physical).

**Reframes the knob decision (`PLAN.md` KNOB CORRECTION):** the "fully-emergent structural knob"
(`cooling_boost_kappa`) is **numerically impractical at a physical f_κ** — fragile (non-physical dMdt / raises
evaporation) *and* slow (enters the structure ODE). With the structural κ_mix (Rung B) already SHELVED for the
same class of reason, that leaves **`cooling_boost_mode='multiplier'`** (scalar on the resolved L_cool) as the
**pragmatic mechanism**: it is stable and fast (never touches the structure ODE) and still scales only the
radiative channel (so it keeps the §8c no-PdV-double-count property), at the cost of being "structural-L_cool ×
scalar" rather than fully emergent. **kappa=2 confirmed kappa is impractical at all physical f_κ** (stable but
still too slow to reach the θ peak). A definitive structural validation of §14 would need HPC. **Tentative
decision: adopt `multiplier` as the production mechanism; θ still emerges (from the structural L_cool), just
scaled** — with the caveat that `multiplier`'s emergent θ (8×L_cool, fires easily) is *cruder* than kappa's
back-reacted θ (~0.5, doesn't fire), so the **calibrated f_κ magnitude must be re-derived for `multiplier`**
(the §14 θ₀/p were fit on kappa and do not carry over) → **DONE, §10 (2026-07-02)**.

Artifacts: `data/_kappa_validation_runner.py`, `outputs/{kappa_val,kappa_val_fk2}/`, `KAPPA_VALIDATION_PLAN.md`.

## 7. Provenance
- Commits (`feature/PdV-trigger-term` — the pt1 branch; the line continued on `feature/PdV-trigger-term-pt2`,
  merged via PR #717, then reconciled with `feature/transition-trigger-pt3`, see `INDEX.md §5`):
  `6642ff4` matrix+comparator, `dc1c2fd` note patches, `17f9653`
  live 3/4 configs, `8bcc6b0` θ_lit plot, `b94689c` plot layout fix, plus this commit (4/4 + figure
  de-annotated). Branch is also mirrored to `claude/amazing-darwin-pl1kzl`.
- Data: `data/{fmix_table,pdv_combined_trigger}.csv`, `runs/data/live_compare.csv` (5 rows),
  `runs/data/harvest_*.csv` (4 configs), `theta_vs_density.png` (+ `data/make_theta_density_plot.py`),
  `fmix_vs_density.png` (+ `data/make_fmix_spread_plot.py`), `da_screen.png`
  (+ `data/make_da_screen.py`, `data/da_screen.csv`), `da_replay.png`
  (+ `data/make_da_replay.py`, `data/da_replay.csv`).
- Committed live runs hidens/simple_cluster/fail_repro: produced via `run_stamped` (clean-tree + per-run
  `provenance.json`), but the run dirs aren't tracked, so no commit hash is pinned here.
- Live lowdens (now committed under `runs/data/`): `harvest_f1edge_lowdens__{none,mult2,mult3}.csv` + the
  two `f1edge_lowdens_*` rows of `live_compare.csv`; produced via `run.py` under `timeout` in an isolated
  clean worktree at `17f9653` — these used `run.py` directly, not `run_stamped`, so no `provenance.json`.

## 9. [data] The 819-run sweep landed — de-conflation verdict + the θ₁-collapse law (2026-07-01, pt3)

> ✳️ **Merge note (2026-07-01):** this section arrived from the parallel `feature/transition-trigger-pt3`
> branch (commits `ca3b4c7`/`01b9616`), written the same day as §8b–§8e but **without knowledge of them**.
> It was numbered §8 there; renumbered §9 here to avoid the collision. Read it together with §8e — the two
> sections' `cooling_boost_kappa` results are in open tension (see the ⚠️ contamination note at the end).

The controlled f_κ(n_H) grid (REPRODUCE #18, Block C) **ran on Helix 2026-06-29** — 786/819 ok in
10h17m (`data/sweep_report.txt`; 33 array tasks died without a sentinel, all interior duplicates of
bracketed cells). Reduced to `data/summary.csv` (786 rows), fitted per (mCloud, sfe, nCore) cell in
`data/fkappa_nH_sweep.csv` (63 cells; `fkappa_nH_sweep.png`). Three results:

1. **De-conflation verdict: a single-variable f_κ(n_H) is REFUTED.** At fixed nCore the measured
   f_κ_fire spreads up to **32×** across (mCloud, sfe) (worst at nCore=3e3: 1→32). sfe is a strong
   secondary axis (higher sfe ⇒ more Lmech ⇒ lower θ ⇒ more boost), and mCloud dominates the dense
   end — 1e7 M☉ clouds fire at f_κ=1 for n≥3e3 while 1e5 M☉ still needs 3–4.
2. **What collapses it: the starting deficit.** Over the 41 fired-above-1 cells,
   `log10 f_κ_fire = 0.041 + 3.755·log10(0.95/θ₁)` (corr 0.968, rms 0.116 dex — vs 0.21 dex for the
   best 3-input fit), i.e. **f_κ_fire ≈ (0.95/θ₁)^3.76** with θ₁ the resolved loss fraction at
   f_κ=1. Equivalently a **universal leverage θ ∝ f_κ^0.266** — the pessimistic developed-epoch
   exponent of §6, not the optimistic 0.63 snapshot estimate (which is hereby retired for
   calibration use). `data/make_fkappa_theta1_collapse.py` → `fkappa_theta1_collapse.{csv,png}`.
3. **Firing ⇒ momentum, at Lancaster-band θ.** At each cell's measured f_κ_fire, 57/57 runs fire
   `cooling_balance` and 57/57 leave the energy phase (45 in `momentum`, 12 still in `transition`
   at stop_t=2); θ_max at fire spans 0.93–1.21 (median 1.02) — the trigger crosses at θ=0.95 and
   segment granularity overshoots the 0.99 band edge for ~half the cells.

**Production consequence (shipped, pt3):** `cooling_boost_kappa = 'auto'` — a load-time registry
resolver (`trinity/_input/fkappa_auto.py`) that trilinearly interpolates the measured 63-cell grid
in (log mCloud_input, log sfe, log nCore); hull-clamped with a warning; the censored diffuse/high-SFE
corner (6 cells, nothing ≤64 fired) resolves to the ceiling 64 with an explicit may-not-fire warning.
Numeric values pass through untouched, so the default 1.0 path stays byte-identical. Tests:
`test/test_fkappa_auto.py`. Acceptance run: `runs/params/fkauto_verify.param` (1e5 M☉, sfe 0.03,
nCore 1e3 — a Lancaster-like GMC; auto→12) reduced by `data/make_fkappa_auto_verify.py` →
`data/fkappa_auto_verify.csv` (REPRODUCE #26). Caveats: calibration is densPL α=0, nISM 0.1,
stop_t=2, hybr — other profiles/solvers resolve on the same table with no measured guarantee (a
warning is logged); f_κ remains the Rung-A structural probe (it still RAISES evaporative dMdt, §6),
so 'auto' inherits that caveat.

> ⚠️ **Post-merge contamination + tension flags (2026-07-01, added when pt3 was merged into the pt2 line;
> see `CONTAMINATION.md`):**
> 1. **Standing-rule violations (📏 PLAN rules 1+2):** the sweep behind the θ₁-collapse law and the `'auto'`
>    grid ran at **`stop_t=2` Myr** (rule 1 demands ≥5 Myr) and defines `f_κ_fire` as "**fired by 2 Myr**",
>    not as "θ_max over a ≥5 Myr run" (rule 2). A cell that would fire between 2 and 5 Myr at lower f_κ is
>    over-boosted by 'auto'; the six censored cells might fire by 5 Myr. **The 63-cell grid is therefore
>    PROVISIONAL until re-measured under the 8-config × 5 Myr × θ_max protocol** (`runs/README.md`).
> 2. **Open tension with §8e:** §8e (same day, pt2 line, no cross-knowledge) found `cooling_boost_kappa=8`
>    drives the β-δ solver to non-physical `dMdt` and a frozen state on `simple_cluster`/`be_sphere`/
>    `small_1e6`, while this sweep reports 57/57 cells firing cleanly at f_κ_fire up to 64 under
>    (`betadelta_solver=hybr`, α=0, nISM 0.1, stop_t 2). ~~Candidate explanations — solver choice, config
>    differences, or §8e's early-time truncation — are **unresolved**.~~ **→ RESOLVED same day from the
>    committed sweep data itself — see §9a** (both results are true; the knob's breakdown is
>    non-monotonic in f_κ).
> 3. **Direction tension:** 'auto' interpolates a per-cloud f_κ so that *every* cloud fires — this
>    chases the target, in tension with the same-day maintainer decisions "single physical f_κ constant,
>    NOT f_κ(n)" and route-a ("diffuse clouds may never enter momentum"). 'auto' therefore stands as a
>    **documented opt-in convenience mode** (like `theta_target`), not the production direction.

**Acceptance run EXECUTED (2026-07-01, in-container, ~14 min — REPRODUCE #26 ✅, 4/4 checks PASS):**
`fkauto_verify` (1e5 M☉, sfe 0.03, nCore 1e3 — the Lancaster-like GMC): `'auto'` resolved to the
sweep-measured **f_κ=12.0**, phase 1b exited via **`cooling_balance`** at t≈0.375 Myr, ran 1c → **momentum**
to stop_t=2; emergent **θ_max = 1.061** from `dictionary.jsonl` accepted rows (`data/fkappa_auto_verify.csv`).
So the 'auto' *mechanics* work end-to-end on this cell (two latent path bugs in the never-run reducer
`make_fkappa_auto_verify.py` were fixed in the process). This validates the plumbing only — the grid's
calibration contaminations (flag #1) and the dead-window interpolation risk (§9a) stand unchanged.

## 9a. [data] The §8e⇄§9 kappa tension RESOLVED — breakdown is NON-MONOTONIC in f_κ (2026-07-01, no new sims)

> **⚠️ Mechanism claim SUPERSEDED by §9b (2026-07-02).** The "interleaved firing bands /
> breakdown windows" reading below over-read the data: the windows are solver crashes at the
> evaporation→condensation domain boundary (+ an outlawed stop_t=2 horizon), not knob physics.
> The knob-choice conclusion survives on the new grounds given in §9b.

Read straight out of the committed `data/summary.csv` (the 819-run Helix sweep) — no new runs needed.
Builder: `data/make_kappa_stability_map.py` → `data/kappa_stability_map.csv`.

**The decisive cell.** The sweep's `simple_cluster` analog (mCloud 1e5, sfe 0.3, nCore 1e5 — same mass/SFE/
density; recipe differs only in nISM 0.1 vs 1 and stop_t) across its 13 f_κ values:

| f_κ | outcome |
|---:|---|
| 1, 1.5 | healthy to stop_t=2, θ_max 0.68/0.75, no fire |
| 2, 3 | **froze mid-implicit** (t_final 0.54/0.62), no fire |
| **4, 6** | **FIRES** → momentum (θ at fire 1.02/1.04) |
| **8, 12** | **froze mid-implicit** (t_final 0.44/0.37), **θ_max = 0.5331 / 0.588** — §8e's "θ stuck ~0.53 at f_κ=8", reproduced independently on Helix |
| 16 | fires, but θ_max=4.55 (non-physical solver spike en route) |
| 24 | broke (t_final 0.048) |
| 32, 48, 64 | fire violently (n_impl 3–6 rows) |

**Grid-wide stats:** 57 cells fired; **17/57 are non-monotonic** (at least one f_κ *above* the cell's
f_κ_fire fails to fire); **38/819 runs froze mid-implicit without firing** (the §8e signature: premature
end, still `implicit`, θ frozen sub-threshold). So:

- **§8e was right**: f_κ=8 breaks the solver on simple_cluster — the Helix sweep hit the identical freeze
  (θ 0.533) on the matching cell. Not a container artifact, not the solver choice (both lines ran the
  default `betadelta_solver=hybr`).
- **§9 was right**: every one of the 57 cells fires *at its own f_κ_fire* — the firing bands are real.
- **Neither refutes the other**: the kappa knob has **interleaved firing bands and breakdown windows**
  (here: fire 4–6, dead 8–12, fire 16+). §8e happened to sample inside a dead window.

**Consequences.**
1. **The knob-choice argument against `cooling_boost_kappa` gets stronger**: a production knob whose
   usable values form disconnected bands, with silent mid-run freezes between them, is not shippable —
   independent of the evaporation side-effect. The `multiplier` tentative choice (§8e) stands, now on
   firmer ground.
2. **New risk flagged for `'auto'`**: the resolver trilinearly *interpolates* f_κ between grid cells, but
   only the grid's f_κ_fire values are measured to fire — an interpolated value (e.g. 5.3 or 10 for this
   cell) can land inside a dead window. 'auto' remains opt-in/PROVISIONAL (flag #1 above) with this
   added caveat.
3. The "786/819 ok" sweep report over-reads: "ok" includes the 38 mid-implicit freezes (exit-0 runs that
   died early without firing). The fit's f_κ_fire values are unaffected (smallest *fired* f), but
   per-run health must be judged from `t_final`/`phase_final`, not the exit code.

## 9b. [data+repro] §9a's "non-monotonic breakdown" re-examined — the freeze is the evaporation→condensation domain boundary, NOT physics bands (2026-07-02)

The maintainer challenged §9a ("are we sure f_κ is a no-go? 'breaks non-monotonically' may be a
false inference — check vigorously"). The challenge was **right**. Full treatment:
`KAPPA_FREEZE_MECHANISM.md`; data: `data/make_kappa_freeze_autopsy.py` →
`data/kappa_freeze_autopsy.csv`; live repro logs summarized in that doc's §4.

1. **The freeze pre-exists the knob**: 1/819 sweep runs froze at f_κ=1.0 (unboosted). The rate
   rises with f_κ (~1/63 → ~5–7/63): kappa aggravates, does not create.
2. **34/38 freezes died at θ_max ≥ 0.8** (healthy no-fire median: 0.636) — crashes ON APPROACH
   to the 0.95 crossing, i.e. would-fire runs, not cold windows.
3. **All 23 "non-monotonic" arms decompose** into 12 froze-on-approach + 8 healthy-at-2-Myr with
   θ_max 0.87–0.93 (stop_t=2 is an outlawed horizon; diffuse-fires-at-5.04-Myr precedent) +
   3 froze-early. **Zero** arms ran healthy to a rule-compliant horizon and stayed cold.
4. **Live repro smoking gun** (local, simple_cluster): at f_κ=7.5/8/16 the β–δ structure solve
   *converges to a negative dMdt root* (−85.22/−84.76/−53.09 Msun/Myr at t≈3.4e-3 Myr) and the
   `dMdt>0` acceptance gate (`get_betadelta.py:861-869`) refuses it; the runner holds state
   (`run_energy_implicit_phase.py:835-845`) and, if the burst persists, grinds to `max_segments`
   with θ frozen (f_κ=8 holds θ≈0.52–0.53 — §8e/§9a's 0.5331, reproduced a third time). The
   legacy β–δ solver (no gate) shows zero events. A `MAX_SEGMENTS=40` monkeypatch run exits the
   phase early and completes cleanly through momentum — proof-of-concept that
   no-root ⇒ handoff semantics yields well-formed fates.
5. **Physics identity**: dMdt is the eigenvalue of the conduction-front budget; when interface
   radiative losses exceed conductive heating, evaporation physically reverses to condensation
   (McKee & Cowie 1977; El-Badry+19 bubbles do this). Cooling balance IS that reversal condition
   — the trigger's target regime is the gate's forbidden regime. Early-mode freezes are the same
   reversal reached locally (boosted κ) before global θ catches up.

**Supersessions (dated 2026-07-02):** §9a's *mechanism claim* ("interleaved firing bands and
breakdown windows" as a knob property) is superseded — the windows are crash artifacts + an
outlawed horizon. §9a's *practical conclusion* survives on new grounds: multiplier stays the
production knob because it never touches the dMdt eigenvalue (structurally immune), and kappa
stays un-shippable *until* the domain-boundary semantics are fixed (fix ladder in
KAPPA_FREEZE_MECHANISM §7: no-root-streak ⇒ momentum handoff; continuation; saturated-conduction
cap; condensation branch). Instrumentation landed (log-only): `freeze-watch` per-segment
dMdt/θ trace, streak-demoted warnings, frozen-state note on the completion line.

**Fix #1 LANDED (2026-07-03):** a 50-segment no-root streak now hands the phase off to momentum
(`termination_reason="no_physical_root_handoff"`, routed exactly like `cooling_balance`; the
handoff is a *fate*, not a trigger — harvest classifies it like DRAIN, θ<0.95). Verified by
`runs/drive_noroot_handoff_check.py` (threshold 3, f_κ=8: diagnosis → handoff → transition →
momentum → clean STOPPING_TIME end), full pytest 614/614, and structural inertness (the branch
cannot execute below a 50-streak; observed healthy bursts ≤ 8). NB a local byte-identity gate
proved unattainable: an A/A control (same code, two runs) differs from row 1 at the SN noise
floor + ULP level — pre-existing local FP nondeterminism (unpinned BLAS threads suspected; HPC
pins OMP_NUM_THREADS=1), which exonerates the fix and flags that future LOCAL byte-identity
claims need a same-code A/A companion run (KAPPA_FREEZE_MECHANISM §7.1b). The
rule-compliant re-validation matrix is committed and ready: **theta5k** = 8 configs ×
f_κ {1,2,4,6,8,12,16} at stop_t=5 (`runs/make_theta5k_params.py`, 56/56 validate through
`read_param`; `runs/run_theta5k.sbatch`, array 1-56). It asks the corrected question: does
kappa fire monotonically once the solver may leave the energy phase at the physical boundary?

## 10. [data] The theta5 matrix RAN — first fully rule-compliant `multiplier` calibration (2026-07-02, Helix)

The 📏 standard-protocol matrix (8 configs × f_mix {none,2,4,8} × stop_t 5) ran on Helix — **32/32 arms
compliant**: every run reached t=5 Myr or a genuine physics end (shell_collapsed); zero wall-clock
truncations. θ harvested as θ_max from `dictionary.jsonl` accepted rows (`runs/harvest_theta_max.py`),
same knob fit and validated. Artifacts: `runs/data/theta5_summary.csv` (32 rows) →
`runs/data/theta5_calibration.csv` (`runs/make_theta5_calibration.py`). This replaces every number in
F_KAPPA §14 (`CONTAMINATION.md` ⛔ #1–#3).

| config | nCore | θ₀ (θ_max @ f=1, 5 Myr) | f_fire | fate at fire |
|---|---:|---:|---:|---|
| simple_cluster | 1e5 | 0.676 | **2** | fires t≈0.12, **momentum to 5 Myr (healthy)**; f=4/8 fire instantly then shell_collapse ~0.14 |
| pl2_steep | 1e5 | 0.511 | 4 | fires instantly, shell_collapse t≈0.055 |
| midrange_pl0 | 1e4 | 0.636 | 4 | fires t≈0.30, shell_collapse t≈1.20 |
| be_sphere | 1e4 | 0.529 | 4 | fires t≈0.44, shell_collapse t≈2.48 |
| **large_diffuse_lowsfe** | **1e2** | **0.535** | **4** | fires t≈2.43, **survives to 5 Myr** (transition @4, momentum @8); f=2 grazes 0.9552 at exactly stop_t |
| small_1e6 (control) | 1e2 | 0.297 | >8 | never fires (0.835 @8) — **route-a**, healthy 5 Myr at all f |
| fail_repro (5e9) | 1e2 | 0.003 | n/a | radiative θ ≤0.025 at all f; rides the PR#715 Eb≤0→momentum handoff identically with/without boost — **§8b acceptance PASSES** |
| small_dense_highsfe | 1e6 | 0.717 | n/a | f=2: Eb≤0 handoff at t=0.045 WITHOUT firing; f=4/8: **NaN loss rows** then handoff+collapse — the known dense-edge (nCore 1e6) stiffness, now under boost |

**Headline results:**

1. **The blowout metric under-read diffuse θ by ~2× — 📏 rule 2 vindicated.** large_diffuse θ₀ = 0.535
   with the peak at t≈4.9 Myr (vs 0.17–0.25 at blowout). Consequence: **the diffuse GMC fires at
   f_mix=4**, not the f≈60 the contaminated blowout calibration demanded. The route-a boundary moves
   far down.
2. **θ₁-collapse law for `multiplier`:** over the 5 fired configs,
   **log₁₀ f_fire = 0.142 + 1.824·log₁₀(0.95/θ₀)**, i.e. **f_fire ≈ 1.4·(0.95/θ₀)^1.8** — a leverage
   θ ∝ f^0.55, ~2× kappa's 0.27 (no structural back-reaction to eat the boost). Same functional form
   as §9's law, different constants — knob-specific, as the KNOB CORRECTION predicted.
3. **A single physical f_mix ≈ 4 fires the whole normal-GMC band** (n=1e2–1e5, masses 1e5–1e7, incl.
   the diffuse cloud) with θ_max at fire 0.96–1.04 — at/just over the Lancaster 0.9–0.99 band edge
   (segment-granularity overshoot, cf. §9's median 1.02). f_mix=2 fires only simple_cluster.
4. **Route-a is real and de-conflated:** small_1e6 (same nCore=1e2 as the diffuse config!) stays
   energy-driven through f=8 — f_fire is set by θ₀ (mass/SFE/structure), not by density alone,
   confirming §9's de-conflation with the correct knob.
5. **NEW failure mode — fire-then-recollapse:** at f≥4 every dense-core config that fires promptly
   shell_collapses (0.05–2.5 Myr). Only simple_cluster@2 and the diffuse@4/8 fire AND survive to
   5 Myr. This is §8a's recollapse question resurfacing from EMERGENT cooling (not imposed θ).
   **→ ✅ RULED (maintainer, 2026-07-02): acceptable physics** — firing into the momentum phase and then
   recollapsing is "completely fine"; fire-then-recollapse is an outcome class, **not** a failure mode.
   (The f=8 Eb-drain-without-firing of point 6 and the dense-edge NaNs remain the actual pathologies.)
6. **NEW failure mode — over-boost Eb-drain:** midrange@8 reaches momentum via the Eb≤0 handoff
   WITHOUT the trigger firing (θ_max 0.923, `fired=False`); the dense edge does the same at every
   boosted f. `multiplier` has an over-boost ceiling — gentler than kappa's dead windows (§9a), but
   the same lesson: more boost ≠ more transition. **→ refined by §11:** this is the general
   fire-vs-drain race; the fine bracket shows per-config no-fire GAPS (e.g. simple_cluster at
   f=2.5–3) and pulls midrange's ceiling down to f=5.

**Figures (2026-07-02, `data/make_theta5_figures.py` → REPRODUCE #29):** `theta5_arms.png` (the full
matrix, outcome-classed), `theta5_collapse_law.png` (the law + kappa's for contrast),
`theta5_metric_correction.png` (blowout vs θ_max per config), `theta5_target_vs_emergent.png`
(El-Badry target vs native and f=4 points), `theta5_knob_choice.png` (kappa dead windows vs multiplier
monotonicity); quantitative margins in `runs/data/theta5_fmix_scorecard.csv` (per config: θ₀, measured
f_fire bracket, law-predicted f_fire, θ_max and fate at f=4).

**Open after this section:** (a) ~~pin f_mix~~ **✅ f_mix = 4 ADOPTED (2026-07-02 maintainer ruling —
momentum-then-recollapse is acceptable physics; PLAN ledger)**; the theta5b fine bracket
(f∈{2.5,3,3.5,4.5,5}) remains as the referee sensitivity refinement, not a gate; (b) the dense-edge
(nCore 1e6) stiffness under boost needs its own diagnosis (NaN loss rows on accepted steps);
(c) large_diffuse@2 grazing 0.9552 exactly at stop_t=5 suggests a t>5 Myr fire — the committed
theta5b stop_t=8 arms bracket the diffuse f_fire.

## 11. [data] theta5b — the fine bracket + long diffuse arms: window [4, 4.5], law validated, and the fire-vs-drain race (2026-07-02, Helix)

The 43-arm referee matrix ran (fine f_mix ∈ {2.5, 3, 3.5, 4.5, 5} × 8 configs + large_diffuse
stop_t=8 at f ∈ {1, 2, 2.5}); all arms reached stop_t or a physics end. Combined analysis with
theta5: `data/make_theta5b_analysis.py` → `data/theta5_fire_map.csv`, `data/theta5_law_check.csv`,
`theta5b_fire_map.png`, `theta5b_law_check.png`.

1. **Fine f_fire per config (5 Myr):** simple_cluster **2**, midrange_pl0 **2.5**,
   small_dense_highsfe **2.5**, be_sphere **3.5**, large_diffuse **3.5**, pl2_steep **4**;
   small_1e6 never (through 8); fail_repro n/a (PdV handoff at all f).
2. **The θ₁-collapse law survives out-of-sample: rms 0.064 dex (~16%) over 6 configs.** The
   theta5-fit law f_fire = 1.4·(0.95/θ₀)^1.82 predicted every fine-measured f_fire within one
   half-grid-step (worst: simple_cluster, +0.11 dex). One parameter (θ₀, which the solved bubble
   supplies) carries all the cloud-property dependence — the referee-grade argument for the
   constant-f prescription.
3. **The whole-band window is [4, 4.5] — measured, and narrower than assumed.** At f=3.5,
   pl2_steep does not fire (0.850, Eb-drains); at f=5, midrange_pl0 stops firing (0.927,
   Eb-drains — theta5 had put its ceiling at 8; the fine scan pulls it to 5). **f_mix=4 is not
   just the minimal band-firing constant — it sits inside a narrow measured window**, bounded
   below by pl2_steep's threshold and above by midrange's over-boost drain. Referee answer:
   2.5 and 3.4 measurably miss part of the band; 4.5 works; 5 already drops a config.
4. **NEW SYSTEMATIC — the fire-vs-drain race (supersedes §10's "multiplier has no dead
   windows" phrasing and the ch.16/F5 claim).** Below a config's f_fire, extra boost often
   *prevents* firing: the boosted Eb drain reaches Eb≤0 and hands off to momentum BEFORE θ
   crosses 0.95 (DRAIN cells in the fire map). simple_cluster fires at f=2, then does NOT at
   2.5–3 (θ_max 0.67–0.81, handoff at t≈0.13), then fires again at 3.5+ — a real per-config
   no-fire gap. Unlike kappa's §9a dead windows these are NOT solver pathologies: every run is
   healthy, completes, and still reaches momentum — just via Eb≤0 with θ_max < 0.95 instead of
   via the trigger in the Lancaster band. The corrected statement: *the multiplier knob has no
   solver breakdowns; its fire SET is non-monotonic in f because firing races Eb-drain.*
5. **Diffuse long arms:** (a) f=2 @ stop_t=8 **fires at t≈5.04 Myr** (θ_max 0.960) — the
   theta5 graze (0.9552 at exactly t=5) was a real near-fire; the diffuse f_fire is
   horizon-dependent: 3.5 within 5 Myr, 2 within ~5.1 Myr. The 5 Myr protocol horizon is an
   operational choice and should be stated as such (GMC lifetime scale), not physics.
   (b) f=1 @ stop_t=8: θ_max identical to the 5 Myr run (0.535, peak t≈4.86, no later growth) —
   **the 5 Myr window does capture the native diffuse peak**; 📏 rule 1 self-check passes.
   (c) f=2.5 @ stop_t=8 still DRAINs (θ_max 0.828 at t≈2.56) — the gap is not time-limited.
6. **The dense edge fires after all:** small_dense_highsfe fires at every fine arm
   (2.5→0.950, 3→1.009, 3.5→0.975, 4.5→0.991, 5→0.960; collapse t≈0.04–0.05). theta5's NaN
   arms at exactly f=4 and 8 were erratic stiffness draws, not a wall — the nCore=1e6 ticket
   stays open but is downgraded from "breaks under boost" to "intermittently NaNs".

**Consequences for the f_mix=4 adoption (ledger 07-02):** unchanged and strengthened — 4 is
measured to be in the window's interior-bottom; the paper sentence is now "any f in [4, 4.5]
gives identical conclusions; 4 adopted" with the fire map as evidence. The DRAIN phenomenology
should be mentioned wherever over-boost is discussed (it replaces "more boost = more
transition" intuition).

7. **No density/mass/SFE term survives at grid resolution (residual test, 2026-07-02):** the law
   residuals correlate with nothing — vs log n_core r=−0.39 (p=0.45, slope −0.013 dex/dex), vs
   log M_cloud r=+0.42 (p=0.40), vs log SFE r=−0.25 (p=0.63); all slopes are below the 0.079 dex
   grid step. Sharper: the **θ₀-matched trio** (diffuse n=1e2 / be n=1e4 / pl2 n=1e5, θ₀ all
   0.51–0.54) spans 3 dex in density yet their f_fire spread is 0.058 dex (3.5/3.5/4.0) →
   **|∂log f_fire/∂log n| ≲ 0.02 at fixed θ₀**. All cloud-property dependence flows through θ₀.
   Temperature/Λ(T) is the one *untested* axis (never varied independently); the theta5c design
   — swap `path_cooling_CIE ∈ {1,2,3}` at fixed config — is specced in PLAN → REFEREE DEFENSE Q2.

## 12. [data] theta5k RAN — first rule-compliant kappa matrix: ZERO freezes, but NO whole-band f_κ exists (2026-07-03, Helix)

56/56 arms (8 configs × f_κ ∈ {1,2,4,6,8,12,16}, stop_t=5, θ_max from dictionary rows), run on
the post-fix-#1 branch. Data `runs/data/theta5k_summary.csv`; analysis
`data/make_theta5k_analysis.py` → `data/theta5k_fire_map.csv`, `theta5k_fire_map.png`,
`theta5k_theta_rise.png`. Outcome counts: 21 FIRED / 18 NOFIRE / 12 DRAIN / 5 CONDENSE / **0
freezes** — the §9a freeze class is extinct; every arm ends in a proper fate.

1. **Fix #1 validated at scale.** The five CONDENSE arms (n_impl pinned at the 50-segment
   handoff cap) are exactly where §9a saw "dead windows": simple_cluster 8/12/16 (θ held
   0.533/0.587/0.624 — the §8e 0.5331, now a *fate* instead of a crash), dense 6, pl2 16.
   The old sweep's simple_cluster "fire at f_κ=16" is exposed as a solver artifact: rule-
   compliant, it CONDENSES at 0.624 (the old θ_max=4.55 spike was the ⛔ #3 observer).
2. **The fire set is still non-monotonic — but now it is honest physics of the knob, not a
   crash.** sc fires 4–6 then condenses 8+; dense fires 4, condenses 6, fires 8–16; pl2 drains
   4–6, fires 8–12, condenses 16; be fires 6, drains 8 (θ 0.833, shell dissolves t≈4.86),
   fires 12+. Structural boosting loses a *race*: the front goes condensing (or the shell
   dissolves/drains) before global θ crosses. diffuse and midrange are cleanly monotonic
   (fire 4+). θ_max itself rises ~monotonically with f_κ everywhere it can be measured
   (`theta5k_theta_rise.png`) — the race, not the knob's reach, decides.
3. **HEADLINE — no single f_κ fires the whole band.** Best column: f_κ=12 fires 5/6 (misses
   sc, which condenses). The multiplier's measured window [4, 4.5] fires 6/6. The production-
   knob choice (multiplier) now rests on rule-compliant, crash-free, like-for-like data — the
   strongest form of the §9a.1 argument, with the mechanism correctly attributed.
4. **`'auto'` demotion hardens.** The 819-grid f_κ_fire values that `'auto'` resolves from
   were measured pre-fix at stop_t=2; theta5k shows at least one cell's grid value (sc 16)
   doesn't fire at all under the rules. 'auto' stays opt-in/PROVISIONAL and should be
   re-derived from theta5k-class data if it is ever promoted.
5. Controls behave: fail_repro DRAINs immediately at all f_κ (PdV regime, θ≤0.011);
   small_1e6 climbs 0.297→0.680 without firing. Caveat: fired-arm θ_max up to 1.99 (dense
   4/16) — structural boosting distorts θ en route (known kappa pathology, ⛔ #3-adjacent);
   quote fire/no-fire from this matrix, not θ magnitudes above ~1.2.

6. **Maintainer primary-source recheck + non-monotonicity bug-hunt (2026-07-03).** (a) Weaver II
   §V: the classical front budget is already 60/40 (≈40% of conductive flux radiated in the
   interface) — the reversal is *close by* even unboosted; (b) TRINITY's closure T ∝ Ṁ^{2/5}
   has no Ṁ<0 profile family — the gate is the closure's domain edge (KAPPA_FREEZE_MECHANISM
   §3); (c) planar-analogue eigenvalue uniqueness (Tan–Oh–Gronke §2.2) demotes branch
   multiplicity below fast-moving boundary conditions as the +1121→−85 explanation. Bug-hunt on
   the flip arms: the feared stale-dictionary signature WAS found — small_1e6 ⇄ large_diffuse
   share bit-identical theta_first in all 7 arm pairs — and resolved as real physics (both have
   M_cluster=1e5, nCore=1e2, flat profile → identical early trajectories; register ⚡ note: they
   are ONE check, not two, for early-time claims). Everything monotone IS monotone in f_κ
   (theta_first per config; the CONDENSE-held θ: sc 0.533/0.587/0.624); only the discrete
   outcome flips, always decided within the first ~30–50 segments (dense t<0.06 Myr) — a
   photo-finish between the trigger clock and the condensation/dissolution clocks, both
   accelerated by f. be k8 is a different racer: shell DISSOLVED at t≈4.86 with θ=0.833.
   Discriminating trace (dense k6 vs k8 freeze-watch) queued; 9th config (normal_n1e3:
   mCloud 1e6, nCore 1e3, sfe 0.01, PL0) added via `runs/make_theta5n_params.py` (15 arms,
   both knobs — the law predicts its f_fire from θ₀ before the fine arms are read).

7. **The discriminating trace RAN (2026-07-03, dense k6 vs k8, controlled pair — identical
   early dt sequences).** The rejected eigenvalue evolves *smoothly* in both arms (no bracket
   chaos, no branch-hopping): k8's root decays −36→−15, dips, then RECOVERS through zero
   (+65.3 at segment 28) → structure accepted → fires (matches HPC n_impl=28, θ_first=0.617);
   k6 nearly recovers early (−4.0 at segment 8), second-dives to −37.9, and never recovers in
   the 50-segment window → handoff. Verdict: solver exonerated — fire-vs-condense is decided by
   whether the front budget recovers to evaporation, i.e. trajectory physics. Caveat kept: one
   discontinuous jump per trace correlates with segment-loop discrete events (cooling-table
   refresh suspected) → **the race is physical but its exact f_κ edge is
   discretization-sensitive; treat per-config f_κ_fire as razor-edge, not law-grade** (the
   multiplier's f_fire sits on the smooth θ₁-collapse law instead). Full trace excerpts:
   KAPPA_FREEZE_MECHANISM §5.
