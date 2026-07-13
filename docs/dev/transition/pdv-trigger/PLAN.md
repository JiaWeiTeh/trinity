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

> **Sibling (updated 2026-07-01):** `HIMASS_HANDOFF_PLAN.md` — **now PARTIALLY IMPLEMENTED, not just a plan.**
> **Shipped:** phase 1b routes a finite `Eb<=0` collapse to the momentum phase (via 1c) instead of the
> `ENERGY_COLLAPSED` dead-stop (`run_energy_implicit_phase.classify_energy_collapse`); phase 1a gained a
> `cooling_balance` check. Verified: `fail_repro` (5e9,n1e2) now reaches momentum out to 500 pc; G0
> bit-identical on `simple_cluster`; pytest 596 passed. **Deferred:** phase-1a collapse routing, and the
> smoother pressure-crossover terminal event (the 1b handoff currently fires on post-step `Eb<=0`).
> Records the verified correction that the high-mass turnover is **PdV/inertial-loading driven, not
> radiative** (`FINDINGS.md` §1/§6a + the `Pb` cancels `Eb` identity), and that `ebpeak` is opt-in/shadow,
> not the default path. Branch `bugfix/high-mass-cluster-transition-without-ebpeak`.

The recheck list the banners demand. **Every visit:** re-verify the anchors below, update the ledger,
*then* read on. All findings here are **already persisted** (CSVs + figures under `data/` and this
folder) — do **not** re-run the hours-long sims to recover them; reproduce only to extend.

### ⭐⭐ CANONICAL SYNTHESIS + VERDICT (read this first — supersedes all earlier synthesis blocks; corrected 2026-07-01)

> ✅ **DIRECTION CORRECTED 2026-07-01 (maintainer steer). θ is an OUTPUT, not an input.** The earlier plan to
> **impose** El-Badry's θ (`cooling_boost_mode='theta_target'`, the "θ_elbadry" SPEC) is **demoted to an opt-in
> second option.** Enforcing θ was a shortcut around a hard calibration, and it broke the causal structure:
> it sets `L_loss=θ·L_mech` regardless of what the loss actually is, so on **PdV/inertia-dominated** massive
> clouds it double-counts the loss (PdV is already in `Edot_from_balance`) and drives them to recollapse —
> **reversing PR #715's momentum handoff** (proven: `FINDINGS.md §8b`, `data/newcode_default_vs_theta.csv`).
> The corrected direction returns to **Rung A — boost the cooling MECHANISM (the `cooling_boost_kappa` /
> `multiplier` f_κ family) and let the solved bubble PRODUCE θ**, with El-Badry/Lancaster used to **calibrate**
> what θ should emerge to, not to enforce it. See the top status-ledger entry (2026-07-01) and `FINDINGS.md §8c`.
>
> ⚠️ **KNOB CORRECTION (2026-07-01, later): `cooling_boost_kappa` ≠ `multiplier` — they are different, and the
> §14 calibration + the f_κ-throughput validation used DIFFERENT ones.** `cooling_boost_kappa` scales the Spitzer
> conduction coefficient *inside* the bubble-structure ODE (`bubble_luminosity.py:291/370/406`) → L_cool changes
> *through* the physics, θ **fully emergent** — this is what the §14 leverage/θ₀ were measured with, and the true
> κ_eff "θ-as-output" mechanism (but it also raises the evaporative mass flux — registry note calls it "a
> structural probe, not the final model"). `cooling_boost_mode='multiplier'` (`cooling_boost_fmix`) instead scales
> the *already-computed* L_cool in `effective_Lloss` (`get_betadelta.py:473`) — it never enters the structure ODE
> (avoids the evaporation side-effect, but θ is only "structural L_cool × scalar"). **My 2026-07-01 validation +
> f_κ-sweep runs used `multiplier`, so they do NOT validate the `kappa`-based §14 numbers** (`FINDINGS §8d`). OPEN
> decision: which knob is the production mechanism. This does not change the §8c "θ-should-emerge, not be enforced"
> conclusion (both knobs are emergent-flavoured and neither double-counts PdV); it does mean the §14 calibration
> must be re-validated with the SAME knob it was fit on.
> **→ OUTCOME (`FINDINGS §8e`, 2026-07-01): re-validated with `kappa` — it BREAKS DOWN at the physical f_κ=8**
> (non-physical dMdt / evaporation side-effect → state freezes, physical θ~0.53, no fire) **and is far slower**
> (enters the structure ODE). With Rung-B κ_mix also SHELVED, the **tentative decision is `multiplier` as the
> production mechanism** — stable, fast, still radiative-only (no PdV double-count), θ emerges from the
> structural L_cool (just scaled). **Confirmed since (2026-07-02):** kappa=2 impractically slow (§8e), kappa
> dead windows (§9a — re-diagnosed §9b/§12: crashes at the evaporation→condensation boundary,
> fixed by the no-root handoff; even crash-free, no whole-band f_κ exists), and the 32/32-stable
> theta5 matrix on the multiplier knob (§10).

> ➕ **f_A BRANCH ADDENDUM (2026-07-06 → 07-11) — a THIRD knob now exists and is fully validated
> in-container; the mechanism decision below is therefore INCOMPLETE until Phase 6.** After this block
> was frozen (f_mix=4 adopted 2026-07-02), the maintainer commissioned the **in-ODE source-term knob
> `cooling_boost_fA`** (`SOURCE_TERM_DESIGN.md` — THE single plan doc for that branch): it multiplies the
> net radiative source *inside* the bubble-structure ODE + the resolved interface losses, only in the
> interface band (T<10^5.5 K) — the physically-correct in-structure boost that `kappa` wanted to be,
> WITHOUT the wrong-sign evaporation coupling (its dMdt *falls* with boost, El-Badry Eq-47 sign; kappa's
> rose → condensation crashes). Status: **Phases 0–4 ✅** — shipped gated default-1.0 LITERAL
> byte-identical; no condensation edge even at f_A=512; **81/81 theta5s matrix complete in-container
> (2026-07-11, PROVISIONAL pending HPC): collapse-law p=3.330 confirms the registered p_source≈3.3
> (vs multiplier's 1.82); both controls cold at every f_A; 3 classes (normal_n1e3 fires unmodified /
> small_dense 4, simple_cluster 4, midrange 6, large_diffuse 8, pl2_steep 12, be_sphere 12 / controls
> never); dMdt suppression <1 matrix-wide** (`FINDINGS.md §15–§15e`,
> `data/theta5s_{fire_map,collapse_law,dmdt_suppression}.csv`). **What stays true from the block above:**
> f_mix=4 remains the ADOPTED production working value today (default `none`). **What is now OPEN
> (Phase 6):** whether f_A supersedes `multiplier` as the production mechanism — decided AFTER Phase 5
> (bench5 Lancaster/El-Badry calibration) + the HPC confirmation of the in-container matrix. Do not read
> the 2026-07-01/02 verdict above as the final mechanism ruling.
>
> ↳ **Re-entry 2026-07-12 (`FINDINGS.md §15f`) — Phase 5 was BLOCKED on the L21b Table-1 maintainer
> input** (pre-step re-verified unreachable: proxy 403 on arXiv/IOP/NSF-PAR; WebSearch = snippets only);
> nothing frozen or run at that point, only the f_A wiring line-ref drift-fix
> (`bubble_luminosity.py:435/845/65`).
> **↳ RESOLVED same-day (`FINDINGS.md §15g`):** the maintainer supplied the Table-1/Eq-8–11/Fig-17
> excerpts in-chat. Verified + imprinted (`LANCASTER_REFERENCE.md §7b`): μ_H=1.4 ⇒ n̄_H is TRINITY's
> nCore convention; v_t = the α_vir=2 virial velocity (12/12 rows); **"M_*=5000 fixed" FALSIFIED**
> (ε_* ∈ {0.01,0.1,1} → the spec's sfe=0.05 benches matched no published model, corrected to ε0.1);
> §7's Eq-10 transcription corrected. **60 bench5 params frozen** (`runs/make_bench5_params.py`,
> exact mapping mCloud=M_cl(1+ε)/sfe=ε/(1+ε), emit gates + end-to-end read_param check).
> **↳ IN-CONTAINER RUN — COMPLETE (`FINDINGS.md §15h`, 2026-07-12):** maintainer ruled **all 60 bench5
> arms run IN-CONTAINER** ("just run the 60 in-container — it's definitely doable; anything within 2 h is
> runnable"). Campaign done (`runs/{run,harvest,checkpoint}_bench5*.py` + `autocommit_bench5.sh`) at
> `--per-arm-timeout 7200` (2 h/arm): **60/60 ran, 59 compliant** (1 dense diag wall-killed, non-critical).
> **FIRE MAP** (production arms): threshold **1 → 4 → 12 → >16 → >16** as n̄ falls 2.28e5→4.42e4→5520→690→43
> — bench5 fires UNMODIFIED, bench3 at f_A≥12, bench2/bench1 NOFIRE ≤16. **Θ_cum L21b calibration** (diagnostic
> arms, all complete; diffuse benches blow out cleanly = the L21b breakout window): bench3 enters the band
> [0.90,0.99] at **f_A≈16** (Θ_cum 0.965); bench2/bench1 do NOT reach it even at f_A=16 (max 0.54/0.40) →
> **f_A >16 / ≫16**. dex-vs-El-Badry never below 0.85. **Result: no single global f_A reproduces L21b across
> density; the required boost climbs steeply toward low density — the route-a boundary.** **Next open: Phase 6
> (ship decision) — see §15h for the three options.**
>
> ### ⚠️⚠️ WHAT REQUIRES HPC/HELIX (be explicit — maintainer directive 2026-07-12) ⚠️⚠️
> **Exactly ONE item needs HPC: the theta5s Phase-4 CONFIRMATION** (`FINDINGS.md §15e`). The 81-arm
> theta5s matrix was run in Claude's ephemeral container (HPC was down) and is **PROVISIONAL** — p=3.330,
> the controls, the 3-class result, and the dMdt trend must be reproduced on Helix (`sbatch
> runs/run_theta5s.sbatch`; params committed) before any theta5s number is authoritative/paper-ready.
> **bench5 (Phase 5) does NOT require HPC** — it is being run in-container per the ruling above. The
> maintainer-facing checklist with exact commands is the repo-root file **`temporary-HPC-runs.md`**
> (created 2026-07-12; the maintainer deletes it after running). Do not add bench5 to the HPC list.

*This single block (+ the dated f_A addendum) replaces the older layered ⭐/⚡/⚡⚡ synthesis. It reflects
the grand view across `ELBADRY_REFERENCE.md`, `LANCASTER_REFERENCE.md`, `F_KAPPA_FUNCTIONAL_FORM.md`, and
all the κ_mix work. Whenever a decision is made, update THIS block and the affected sibling docs together.*

**The goal (maintainer north star):** give TRINITY's 1D bubble cooling **comparable to 3D/obs (Lancaster
θ~0.9–0.99) and dependent on cloud properties**, so the energy→momentum transition fires physically — and
**let transition be "fate"** (clouds that can't reach the threshold stay energy-driven, by design).

**The corrected approach — θ EMERGES from a boosted cooling mechanism (Rung A / f_κ):** The one master
parameter is **`θ ≡ L_cool/L_mech`** — *identical* in TRINITY, El-Badry (`L_int/Ė_in`), and Lancaster
(`Ė_cool/Lw`); all SB dynamics follow from it via `(1−θ)`. **Physically θ is not a knob you get to set** — the
cooling (whatever its mechanism) decides how much energy is radiated, and θ is the *result*. So the only
legitimate lever is the cooling **mechanism**: `cooling_boost_mode='multiplier'` (Rung A, f_κ) scales the
**resolved radiative channel** `L_loss = L_leak + f_κ·L_cool`, then the bubble solver produces whatever θ
follows. El-Badry's closed form `θ = A_mix·√(λδv·n)/(11/5 + A_mix·√(λδv·n))` (A_mix=3.5) and Lancaster's
θ≈0.9–0.99 are the **calibration target** for that emergent θ — pick the **single physical f** via the
θ₁-collapse law (§10; the earlier "pick f_κ(n)" wording predates the single-constant DECISION).

| element | decision | anchor |
|---|---|---|
| **mechanism** | **boost the cooling channel** so θ **emerges** — via one of two knobs (see ⚠️ below): `cooling_boost_kappa` (structural conduction boost, θ fully emergent) or `cooling_boost_mode='multiplier'` (scalar on the resolved L_cool). Either scales only the radiative channel → **cannot over-drain a PdV-dominated bubble** (no regime error, no gate — `FINDINGS.md §8c`) | causally honest |
| **calibration** | pick the **single physical f** via the θ₁-collapse law (§10) so the **solved** θ matches El-Badry/Lancaster | `F_KAPPA_FUNCTIONAL_FORM.md` (El-Badry/Lancaster as the target, not the enforced value); measured law: `FINDINGS.md §10`, `runs/data/theta5_calibration.csv` |
| **f_κ form** | a **SINGLE physical f_κ constant** (~few–8), **not** a steep f_κ(n) formula — the physical enhancement κ_mix/κ_Spitzer ∝ n *rises* with density (opposite the chase-El-Badry f_κ(n)), so no physical f_κ(n) fires every cloud; density-dependence **emerges** as the route-a critical density | `F_KAPPA_FUNCTIONAL_FORM.md` §14 DECISION (2026-07-01) |
| **diffuse fate** | clouds whose physics never reaches θ≥0.95 **stay energy-driven, by design** (route-a). **§10 nuance (2026-07-02): the canonical diffuse GMC *fires* at f_mix=4** — route-a = `small_1e6` + `fail_repro` (a θ₀-set boundary, not density-set) | maintainer-endorsed 2026-07-01: "diffuse clouds may never enter momentum — the cooling/physics never allows it"; El-Badry √n, uncontradicted by Lancaster (GMC-only plateau); measured split: `FINDINGS.md §10` |
| **massive/PdV clouds** | **no θ-imposition;** the PR #715 `Eb≤0→momentum` handoff carries them | `HIMASS_HANDOFF_PLAN.md`; `FINDINGS.md §8b/§8c` |
| **El-Badry θ_elbadry** | **opt-in second option only** (forced cooling), fully documented incl. its regime caveat | `THETA_ELBADRY_SPEC.md` (now framed as the override, not the default) |

**κ_mix (Rung B) is SHELVED as a structural injection** — it saturates (10⁵–10⁸× Spitzer instantly) and is
numerically unstable in the Weaver ODE (`KMIX_SELFCONSISTENT.md`). It survives only as the **physical
justification** for *why* enhanced interface cooling scales ∝√(λδv·n). Note Rung A (scalar f_κ, WORKS, now
primary) ≠ Rung B (structural κ_mix, SHELVED) — they are different mechanisms.

**VERDICT: return to emergent θ via f_κ; El-Badry calibrates, does not enforce.** The θ_elbadry detour
established real value — a verified closed form, the n-mapping, the source-verified `(1−θ)` budget, and the
proof that *enforcing* θ double-counts PdV — but its conclusion is that enforcement is the wrong primitive.
Rung A (`multiplier`) is already shipped in production, gated default-off byte-identical. **The calibration ran
(§10, 2026-07-02) and the recollapse gate is CLEARED (maintainer ruling, same day: firing into the momentum
phase and then recollapsing is acceptable physics — an outcome, not a failure mode). f_mix = 4 is the ADOPTED
working value** (production default stays `cooling_boost_mode='none'`; 4 is the documented recommended
setting), subject only to the theta5b fine bracket (the referee sensitivity statement) and the diffuse
stop_t=8 arm — not a new mechanism.

**SETTLED (2026-07-01):** θ is the master parameter (3-way identical) & is an **output** · knob = cooling
**mechanism** (`multiplier`/f_κ), not enforced θ · El-Badry/Lancaster = **calibration target** for emergent θ ·
f_κ at a **physical** value; **accept diffuse route-a non-transition** · massive/PdV clouds handled by the
PR #715 handoff, **not** θ · `theta_elbadry`/`theta_target` = **opt-in override**, documented with its
double-counting caveat · ≥5 Myr per run.
**OPEN (re-written 2026-07-01 after the KNOB CORRECTION + §8e — the earlier "✅ f_κ(n) calibrated … one cheap
confirmation run" wording here was wrong and is retracted):**
(1) ✅ **DONE (2026-07-02 — `FINDINGS.md §10`, `runs/data/theta5_calibration.csv`): the calibration was
re-derived on the `multiplier` knob.** The 📏 standard-protocol matrix (8 configs × f_mix ∈ {none,2,4,8} ×
5 Myr) ran on Helix, 32/32 rule-compliant; θ₀/p fit per config from θ_max; the θ₁-collapse analogue re-fit for
`multiplier`: **f_fire ≈ 1.4·(0.95/θ₀)^1.8**; **f_mix=4 fires the whole normal-GMC band incl. the diffuse
cloud**; route-a = small_1e6 + fail_repro (θ₀-based, not a density threshold). (The §14 θ₀/p — fit on
`cooling_boost_kappa`, blowout snapshot, falsified slope, R6 observer — remain void, `CONTAMINATION.md`
⛔ #1–#2.) **Residue → ✅ gate CLEARED (2026-07-02, maintainer ruling): momentum-then-recollapse is
acceptable physics ("completely fine"), so f_mix = 4 is ADOPTED as the working value**; the theta5b fine
bracket + diffuse stop_t=8 arm remain as the referee sensitivity refinement (they may sharpen the window,
not the decision). Dense-edge (nCore 1e6) NaN stiffness ticket still open. (2) pin **f_max** — resolved with
the same ruling: 4 (the minimal band-firing constant, inside the physical 2–8 window); theta5b measures the
window edges; structural κ_mix that would justify more at the diffuse end stays SHELVED. (3) ✅ confirmed by theta5 — the
massive cloud hands off cleanly under `multiplier` at every boosted f (fail_repro handoff untouched by the
boost — §10; earlier shown for the gated/default path, `fail_repro` → 500 pc). (4) ✅ **RESOLVED (same day, from committed data — `FINDINGS.md §9a`)**: the §8e⇄§9 kappa tension was
non-monotonic breakdown windows in f_κ (fire 4–6, dead 8–12, fire 16+ on the simple_cluster cell; §8e's
θ≈0.53 freeze reproduced on Helix at 0.5331) — kappa is even less shippable; `'auto'` gains an
interpolation-into-a-dead-window caveat. (5) decide `'auto'` (pt3): revalidate its 63-cell grid under the
📏 protocol or keep it opt-in-provisional.

**BEST PATH FORWARD (updated 2026-07-02):** (i) **Rung A `multiplier` is already in production** (gated
default-off — `grep cooling_boost_mode trinity/`). (ii) ✅ **DONE (2026-07-02 — §10):** the 📏 theta5 matrix ran
on Helix (32/32 compliant) and re-derived the emergent-θ calibration on the *shippable* knob — θ measured only
as **θ_max over ≥5 Myr from `dictionary.jsonl`** (never blowout, never the observer). (iii) ✅ **PINNED
(2026-07-02): f_mix = 4** — the recollapse gate cleared by maintainer ruling (momentum-then-recollapse is
fine physics); theta5b refines the workable window for the referee statement. Publish the **MEASURED
θ₀-based route-a split**
(§10: route-a = small_1e6 + fail_repro; small_1e6 shares nCore=1e2 with the firing diffuse config, so the
boundary is NOT a clean density threshold — the earlier "n ≳ 48 cm⁻³" framing is dropped).
(iv) Massive clouds keep riding the PR #715 handoff untouched (§8b lesson). (v) Keep **θ_elbadry and `'auto'`
as documented opt-in overrides** (`THETA_ELBADRY_SPEC.md`; FINDINGS §9 flags). Evidence chain:
`ELBADRY_REFERENCE.md` (closed form, n-mapping) · `LANCASTER_REFERENCE.md` (θ magnitude, route-a) ·
`FINDINGS.md §8b/§8c` (why enforcement double-counts) · `FINDINGS.md §8e/§9` (knob evidence, both sides) ·
`HIMASS_HANDOFF_PLAN.md` (the PR #715 handoff that carries the massive clouds) · `CONTAMINATION.md` (what is
quotable).

**REFEREE DEFENSE (2026-07-02) — the two questions a referee will ask, and the tests that answer them.**
Committed, ready-to-run ammunition: the **theta5b matrix** (`runs/make_theta5b_params.py` →
`runs/params/theta5b/`, 43 params validated through `read_param`; `runs/run_theta5b.sbatch`, array 1-43,
~same cost as theta5). Storyline treatment: `pdvtrigger_report.html` §16.2.

*Q1 — "why exactly f_mix=4? are 2.5/3.4/4.7 workable?"* **ANSWERED BY MEASUREMENT (theta5b, 2026-07-02 —
FINDINGS §11, `data/theta5_fire_map.csv`):** 2.5 and 3.4 measurably miss part of the band (pl2_steep fires
only at 4; be/diffuse at 3.5); 4.5 works; **5 already drops midrange_pl0** (over-boost Eb-drain). **The
whole-band window is [4, 4.5]** and 4 sits inside it — a measured choice, no longer a grid-point argument.
The matrix as designed:

| theta5b arm | what it measures | referee answer it produces |
|---|---|---|
| f ∈ {2.5, 3, 3.5} × 8 configs | window lower edge + law accuracy | **MEASURED:** pl2 needs 4 (3.5 → drain); be/diffuse fire at 3.5; mid/dense at 2.5; law rms **0.064 dex** out-of-sample (`data/theta5_law_check.csv`) |
| f ∈ {4.5, 5} × 8 configs | window upper side | **MEASURED:** 4.5 keeps the full fire set; **5 drops midrange** (Eb-drain) → the paper sentence is "any f∈[4, 4.5] gives identical conclusions; 4 adopted" |
| large_diffuse stop_t=8 at f ∈ {1, 2, 2.5} | the t=5 graze + peak capture | **MEASURED:** f=2 fires at t≈5.04 Myr (horizon-dependent threshold — state 5 Myr as the operational GMC-lifetime horizon); f=1@8Myr: native peak t≈4.86, unchanged — **5 Myr captures it**; f=2.5 still drains |

*Q2 — "why a constant f and not f(cloud properties)?"* Four rows, each independently sufficient; the decisive
test is out-of-sample prediction (fit the θ₁-collapse law on a config subset, predict the held-out f_fire
brackets from θ₀ alone — a per-cloud f(properties) fit cannot beat it without over-fitting):

| prescription | free params | physical sign | measured behavior | falsifiable? | verdict |
|---|---|---|---|---|---|
| **constant f (+ θ₁-collapse law)** | 1 | n/a (physical cap 2–8, F_KAPPA §11) | fires the normal-GMC band at f=4; route-a emerges from θ₀ (§10) | yes: θ₀ > 0.95/f^0.55 ⇒ fires | **current choice** |
| f(n_H) power law | 2 | ❌ wrong sign (κ_mix/κ_Sp ∝ n RISES; the chase-target f(n) falls) | REFUTED by the 819-sweep: 32× spread at fixed n (§9) | weakly | dead |
| `'auto'` 3-axis lookup | 63 cells | ❌ chases the target (fires everything by construction) | grid embeds PRE-FIX artifacts (§9b/§12: old sc "fire at k16" is rule-compliant CONDENSE; freezes were crashes at the condensation boundary); calibrated at stop_t=2 (⛔ #4) | no (fits anything) | opt-in convenience only — re-derive from theta5k-class data before any promotion |
| local κ_mix(n,T) (Rung B) | 1 (λδv=3 pinned) | ✅ the physically faithful form | port SHELVED (saturation, solver failures — KMIX_SELFCONSISTENT); S0–S4 re-port ladder exists | yes | the *physics narrative*; revisit only if multiplier fails a gate |

The constant-f story the paper can defend: *all cloud-property dependence flows through θ₀, which the solved
bubble already computes; a single physical f then lets the physics select the fire set (route-a) instead of
forcing it — one parameter, one falsifiable prediction.*

**Q2 MEASURED (theta5b residual test, 2026-07-02).** Does f need a density (or temperature) term? Two
independent checks on the fine bracket say no at grid resolution:

1. **Residual correlations** (law residuals from `data/theta5_law_check.csv` vs config properties, 6 fired
   configs — indicative, not conclusive, at N=6): resid vs log n_core Pearson r=−0.39 (p=0.45), slope
   **−0.013 dex/dex**; vs log M_cloud r=+0.42 (p=0.40), slope +0.019; vs log SFE r=−0.25 (p=0.63), slope
   −0.020. Every slope is far below the grid step (0.079 dex) over the sampled range; no property survives
   as a predictor once θ₀ is used.
2. **The θ₀-matched trio** (the sharper argument — density varied at *fixed* θ₀): large_diffuse (n=1e2,
   θ₀=0.535), be_sphere (n=1e4, θ₀=0.529), pl2_steep (n=1e5, θ₀=0.511) share θ₀ to ±0.013 while spanning
   **3 dex in density**; their measured f_fire are 3.5 / 3.5 / 4.0 — a spread of 0.058 dex, *below* the
   0.079 dex grid step. Bound: **|∂log f_fire/∂log n| ≲ 0.02 at fixed θ₀** — a density term is unresolved.

**Temperature is the untested axis.** The 8-config suite never varies Λ(T) independently — the interface
temperature is emergent, so f_k's T-dependence is degenerate with the cooling table already inside L_cool.
Cheapest falsification test (**theta5c**, zero code changes): re-run 2–3 configs × `path_cooling_CIE ∈
{1,2,3}` (bundled Cloudy / Cloudy+grains / Gnat–Ferland 2012 tables — different Λ(T) shapes) at f ∈
{1, 3.5, 4}. If the swapped Λ(T) shifts θ₀ and the law still predicts the shifted f_fire from θ₀ alone,
all T-dependence flows through θ₀ and constant-f stands; if f_fire moves at *fixed* θ₀, f_k needs a
T-term. (A metallicity lever also exists — Z=0.15 auto-pins a Sutherland–Dopita table in
`read_param.py` — but `_validate_ZCloud` currently locks Z=1, so that arm needs a validator decision
first.)

---

*Historical context below (pre-2026-06-30 κ_eff/κ_mix framing) — superseded by the canonical block above; kept
for provenance.*

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
  **optimistic** — compact fires cooling at **f_κ≈4**, diffuse needs **≈60** (→ dead per §10, blowout-metric
  artifact, 2026-07-02) (the developed-epoch leverage is
  weaker than the snapshot, exponent ~0.3–0.4). PdV-in-the-trigger probed (`make_ebpeak_trigger_test.py`,
  06-28 ledger): `ebpeak` is an assist, not a substitute — it does not remove the need for the boost. The
  819-combo grid **ran (2026-06-29)** and settled the de-conflation question: f_κ_fire is **NOT** a clean
  function of n_H (spread up to 32× across mCloud/sfe at fixed density), but all fired cells collapse on the
  starting deficit, f_κ_fire ≈ (0.95/θ₁)^3.76 (universal leverage θ∝f_κ^0.27; `fkappa_theta1_collapse.png`).
  Production mode shipped 2026-07-01 (pt3): `cooling_boost_kappa='auto'` interpolates the full 3-axis
  measured grid at load time (`trinity/_input/fkappa_auto.py`; gated, default 1.0 byte-identical).

> **📏 STANDING RULE (maintainer, AUTHORITATIVE, 2026-06-30): run every run to AT LEAST 5 Myr.** Never cap
> `stop_t` short for cheapness — the energy→momentum transition epoch is θ's *peak*, which sits ~0.4–1 Myr for
> compact/mid configs **but up to ≈4.9 Myr for diffuse (§10, measured 2026-07-02)** — hence the ≥5 Myr floor,
> and the open stop_t=8 diffuse spot-check — and the bubble keeps evolving for several Myr; a run capped at
> 0.3 Myr stops *before* the transition-relevant physics. **Only exception:** a run the physics ends sooner (blowout / `ENERGY_COLLAPSED` / recollapse) — that
> is the run's fate, not a truncation. **Audit (2026-06-30):** ✅ the current κ_mix conclusions use the 6 Myr
> cleanroom runs; ❌ the `cal_*` anchors behind `KMIX_PROTOTYPE.md` were capped 0.3–1.0 Myr → re-derive from
> ≥5 Myr runs. f1edge_lowdens (3 Myr) is also short.
>
> **📏 STANDING RULE (maintainer, AUTHORITATIVE, 2026-07-01): θ is measured as its PEAK over the ≥5 Myr run
> (θ_max), NOT at blowout.** The trigger fires on the **first crossing** of θ≥0.95, so the physically-meaningful
> summary of a run is `θ_max = max_t θ(t)` over the whole ≥5 Myr evolution (or the natural end if the physics
> stops sooner). **Blowout-θ (θ at R2=rCloud) is retired as a metric** — it is a single arbitrary/late epoch, it
> is **undefined for clouds that recollapse before rCloud**, and it **under-reads the peak** (illustrated
> 2026-07-01: at f_κ=8, θ_max = 1.33 / 1.01 for n=1e5 / 1e4 vs the blowout-θ₀ calibration's 0.99 / 0.91 — so
> blowout-θ₀ is *conservative*; ⚠️ those two θ_max values were later found to come from the **contaminated
> call-level observer** (R6) *and* the wrong knob (R5) — the qualitative under-read stands, the numbers are
> not quotable, `CONTAMINATION.md` ⛔ #1). **All current and future θ measurement uses θ_max-over-≥5 Myr,
> harvested from `dictionary.jsonl`** (`runs/harvest_theta_max.py`). ⚠️ The §14
> f_κ calibration and `kappa_blowout_calibration.csv` still use the old **blowout-θ₀** baseline → they are
> **superseded by §10 / `runs/data/theta5_calibration.csv` (2026-07-02), which measured the blowout under-read
> (diffuse θ₀ 0.535 vs 0.25 — 2×)**. ⚠️ Do NOT use `data/_fkappa_validation_runner.py`'s
> observer for that (it is the R6-contaminated one) — the sanctioned harvester is `runs/harvest_theta_max.py`
> on `dictionary.jsonl`. `F_KAPPA_FUNCTIONAL_FORM.md` §10/§14 flagged accordingly.

## In-container long-run ops playbook (proven on the 81/81 theta5s run, 2026-07-11)

> The maintainer flagged this combination as a keeper: it let an ~11-hour, 81-simulation matrix run
> **unattended** in the ephemeral (restart-every-few-minutes) container with zero lost results and no
> stale silences. Reuse it for any long background run. The theta5s-specific scripts live in `runs/`
> (`run_theta5s_local.py`, `autocommit_theta5s.sh`, `checkpoint_theta5s.py`) — adapt, don't rewrite.

**The six pieces (all required — each covers a failure mode the others miss):**
1. **Resumable runner** (`run_theta5s_local.py` pattern): every work unit writes `.exit_code`/`.duration`
   markers; on relaunch it skips units already done *in the committed summary* (so /tmp wipes don't lose
   ground). Order work **completable-first** (here: high-fA fires fast; baselines last). Generous per-unit
   timeout — **≥1.5 h before any wall-kill** (maintainer floor; we used 2 h), because a killed unit's
   number is truncated garbage. **Never premature-stop a running unit** — no reorders/tweaks that kill
   in-flight work; apply changes on the next natural relaunch.
2. **Repo-side autocommitter** (`autocommit_theta5s.sh`): merges fresh results into the committed CSV and
   commits+pushes every ~2 min. Git remote is the ONLY durable store; a restart then costs ≤2 min. It is
   the **sole committer** (watchdogs only relaunch it) — no concurrent-commit races. Lives in the repo,
   not /tmp, so hard restarts can't erase it.
3. **Merge-checkpoint helper** (`checkpoint_theta5s.py`): union-by-name, prefer-compliant — accumulates
   across container generations instead of overwriting.
4. **~20-min `send_later` heartbeat that RE-ARMS itself**: each firing checks procs alive (relaunch if
   dead), verifies the runner is *progressing* (a running unit's `t_now`/CPU advances — not just alive),
   reports N/total + new results to the user, then re-arms with the SAME message + updated CONTEXT line.
   Carry the running context IN the message — the session may have compacted by the time it fires.
   Include a **TERMINAL clause** (when done: run the analysis, finalize docs, delete triggers, report).
5. **Hourly cron watchdog** (`create_trigger`, backs up the heartbeat): same relaunch+report duties; also
   carries the terminal clause. Two independent wake channels + the harness's restart notifications = no
   silent stall. **Delete both when the run completes** (terminal cleanup), or they fire into a dead task.
6. **Live notes + honest reporting**: report every tick even if unchanged ("no change, X/81, grinding
   <unit>"); record completions/corrections in FINDINGS/ledger as they land, not at the end.

**Measured environment facts (this container):** 4 cores (3 workers saturate); restarts range ~2 min to
~50+ min apart with no pattern; /tmp *sometimes* survives a soft restart (never rely on it); a unit needs
~40–60 min *uninterrupted* to finish — so completion is gated by stable-window length, not compute; the
implicit integrator accelerates near the end (early-`t_now` linear extrapolation over-predicts 5–30×).
`git push`/`send_later` failing with "stream closed" usually still landed — **always re-verify (`git log`,
sync counts) instead of retrying blind.**

### Habits that made this work (keep them)

- **Register predictions before running** (p_source≈3.3 was written down 5 days before the matrix) — that
  is what makes "CONFIRMED" a claim instead of a story.
- **Controls inside every matrix**, and "a control fires" = BUG to flag, never a pass.
- **Quote only the analysis artifact** (the CSV's FIRED/NOFIRE), never a raw intermediate (θ_max) — raw
  reads produced two wrong thresholds this run; the third read (complete data + analysis) was right.
- **Thresholds from incomplete data are upper bounds, not results** — say "not yet" instead of guessing.
- **Compliance gate before quoting anything** (`t_final ≥ 5` or a physics end; ≥1.5 h before deciding).
- **Measure, don't extrapolate** — the "67 h, needs HPC" call was wrong by 30×; the fix was reading
  `t_now` twice, 3 s apart.
- **Every correction gets recorded where the wrong number lived** (dated, with the reason), not just fixed
  silently — see the threshold-correction lesson in `FINDINGS §15e`.
- **Stale-doc sweep at every milestone**: grep the workstream for the OLD status words ("awaiting",
  "incomplete", "in progress", old counts) and reconcile every hit — this sweep found five.
- **After any tool error, verify state before acting** — half this run's "failures" had actually landed.

### Next-chat kickoff prompt (copy-paste to continue this workstream in a fresh session)

```
Continue the f_A (cooling_boost_fA) workstream in JiaWeiTeh/trinity. Work on the branch you are given
(previously pdv-trigger-pt4); never push elsewhere without permission.

READ FIRST, in order — the docs are the ground truth, chat memory does not exist:
1. docs/dev/transition/pdv-trigger/SOURCE_TERM_DESIGN.md — the "Next-chat handoff" block at top, then
   the Status line and §3 (THE single plan doc for f_A; maintainer directive: one workflow stream).
2. docs/dev/transition/pdv-trigger/FINDINGS.md §15–§15e (results; §15e = the 81/81 in-container matrix).
3. docs/dev/transition/pdv-trigger/PLAN.md — the 2026-07-11 ledger entry + the ops playbook (for any
   long background runs) + the f_A addendum in the CANONICAL SYNTHESIS block.
Honor the ⚠️/🔄/💾/🔗 banners: re-verify claims against current source before relying on them.

STATE: Phases 0–4 ✅. Phase 4 completed IN-CONTAINER (81/81, PROVISIONAL — not HPC-verified):
collapse-law p=3.330 confirmed the registered p_source≈3.3; both controls cold at every f_A; 3 classes
(normal_n1e3 fires unmodified / 6 configs need f_A, f_fire 4–12 / 2 controls); dMdt suppression < 1
matrix-wide (Eq-47 sign). Authoritative data: data/theta5s_{fire_map,collapse_law,dmdt_suppression}.csv.

YOUR TASK — the next open phase, in this priority order:
(a) If HPC (Helix) is available: run the AUTHORITATIVE matrix — ./sync_theta5s.sh {up,submit,watch,run,
    down} (sbatch runs/run_theta5s.sbatch) — then execute the §15e mandatory re-check of everything
    downstream (fire map, p=3.33, controls, dMdt) against the HPC summary. HPC wins any disagreement.
(b) Else Phase 5 (SOURCE_TERM_DESIGN §3): bench5 Lancaster/El-Badry calibration — an f_A value is good
    if the SOLVED θ matches the published bubble sims at similar time.
(c) Then Phase 6: the decision tree — does f_A supersede cooling_boost_mode='multiplier' (f_mix=4,
    adopted 2026-07-02) as the production mechanism? Feed it the 3-class result + p=3.33 + controls +
    dMdt fidelity + Phase 5.

STANDING RULES (unchanged): θ only as θ_max over ≥5 Myr from dictionary.jsonl accepted rows via
runs/harvest_theta_max.py (never blowout, never the R6 observer); FIRE = the trigger actually fired
(the analysis CSV), NOT θ_max≥0.95; every arm must clear the compliance gate before its number is
quoted; single-knob arms (mode=none, kappa=1 when varying fA); commit every artifact WITH its builder
script + exact command; every result gets a FINDINGS entry + REPRODUCE row and reconcile
INDEX/PLAN/SOURCE_TERM_DESIGN same-visit; run pytest + test/test_docs_dev_conventions.py before
declaring done; commits carry no AI attribution/links; if a registered prediction is falsified, STOP
and write it up before touching code. For long unattended runs, use the PLAN.md ops playbook
(resumable runner + repo autocommitter + re-arming heartbeat + hourly cron watchdog; ≥1.5 h per arm;
NEVER premature-stop a running arm; report every tick).
```

### Generic phase-runner prompt (paste after EVERY completed phase/item — self-renewing)

```
Run the NEXT open phase/item of the f_A workstream (JiaWeiTeh/trinity, assigned branch only).

RE-ENTRY (every time, no exceptions): read docs/dev/transition/pdv-trigger/SOURCE_TERM_DESIGN.md
(Status line + "Next-chat handoff" block + the relevant §3 phase spec), the newest FINDINGS.md §, and
the top PLAN.md ledger entry. Honor the ⚠️/🔄/💾/🔗 banners — re-verify claims, line refs, and status
words against current source; fix any drift you find AS PART of the visit (dated). Identify the single
next open item from the Status line; if it is ambiguous or maintainer-gated, say so and ask — do not
pick silently and do not fake a gated step.

EXECUTE per the phase spec, sized per CLAUDE.md's planning protocol (risky/iterative ⇒ gate first:
define pass/fail BEFORE editing; capture a baseline; full-run equivalence on stiff regimes in separate
processes at matched t; byte-identity for any "free win").

STANDING RULES: θ only as θ_max over ≥5 Myr from dictionary.jsonl accepted rows via
runs/harvest_theta_max.py (never blowout, never the R6 observer); FIRE = the trigger actually fired
(analysis CSV's FIRED/NOFIRE), NOT θ_max≥0.95; no number quoted before its compliance gate (t_final≥5
or physics end; ≥1.5 h/arm before any wall-kill verdict); single-knob arms; thresholds from incomplete
data are upper bounds, not results; a control that fires is a BUG to flag, never a pass; if a
registered prediction is falsified, STOP and write it up before touching code; measure, don't
extrapolate; after any tool error, verify state (git log / sync) before retrying. Long unattended
runs ⇒ the ops playbook above (resumable runner + repo autocommitter + re-arming heartbeat w/
context+terminal clause + hourly cron; NEVER premature-stop a running arm; report every tick; delete
triggers at terminal).

CLOSE-OUT (this is what makes this prompt reusable — leave the docs so the SAME prompt works again):
1. Commit every artifact WITH its builder script + the exact command (💾; nothing left in /tmp).
2. Dated FINDINGS.md entry + REPRODUCE.md row for the result.
3. Reconcile ALL siblings same-visit (🔗): SOURCE_TERM_DESIGN Status line + phase header + handoff
   block (advance it to the new state), INDEX.md row, PLAN.md ledger (newest-first entry; mark
   superseded claims).
4. Regenerate MANIFEST (python make_manifest.py) in the same commit as new artifacts.
5. Run pytest + test/test_docs_dev_conventions.py; fix before declaring done.
6. Milestone stale-doc sweep: grep the workstream for old status words ("awaiting", "in progress",
   "incomplete", old counts/thresholds) and reconcile every hit.
7. Push (no --force). Commits: no AI attribution, no session links, no co-author trailers.
8. Report: what ran, the numbers (with their provisional/authoritative status), what is now the next
   open item.
```

**Status ledger (newest first):**
- **2026-07-11 (🟢 f_A PHASE 4 in-container COMPLETE 81/81 — PROVISIONAL, HPC confirmation pending).**
  The in-container theta5s matrix reached 81/81 compliant over ~11 h across ~dozen container restarts
  (runner + repo autocommitter + send_later heartbeat + hourly cron; 2 h/arm; each arm got its full
  budget, only restarts cut arms short). Ran `data/make_theta5s_analysis.py`. **HEADLINE: collapse-law
  p=3.330 (rms 0.055 dex, n=6) CONFIRMS the registered prediction p_source≈3.3;** both controls
  (`fail_repro`, `small_1e6`) never fire at any fA; 3-class structure (fires-unmodified `normal_n1e3` /
  needs-f_A {small_dense 4, simple_cluster 4, midrange 6, large_diffuse 8, pl2_steep 12, be_sphere 12} /
  controls). Corrected two θ_max-vs-FIRED threshold errors (simple_cluster fa4 not fa6; large_diffuse fa8
  not fa6). Artifacts: `data/theta5s_{fire_map,collapse_law}.csv` + `theta5s_{fire_map,theta_rise}.png`.
  Still ASSUMED (not HPC) — §15e mandatory action stands: re-run on `run_theta5s.sbatch`, re-check
  downstream. Wrote the next-chat handoff into `SOURCE_TERM_DESIGN.md`. FINDINGS §15e (final), REPRODUCE
  row 41. **Phases 5 (bench5 Lancaster calibration) + 6 (decision) open.**
  **Late addition (same day): the dMdt fidelity read (read iii) salvaged from /tmp before wipe** —
  `data/theta5s_dmdt_suppression.csv`, 49/72 quotable, **all <1 falling with f_A (0.99→~0.85)** = Eq-47
  suppression sign matrix-wide, the measurement f_mix cannot produce. Also swept the workstream for stale
  claims post-completion: refreshed the summary-CSV banner (was "INCOMPLETE"), SOURCE_TERM_DESIGN Phase-4
  header/STATE ("awaiting submission" → complete-provisional), REPRODUCE row-41 typo, and added the f_A
  addendum to the CANONICAL SYNTHESIS block above (which still read as if multiplier/f_mix=4 were the
  final mechanism ruling — Phase 6 is the ruling).
- **2026-07-10 (⚠️ f_A PHASE 4 PROVISIONAL in-container fallback — NOT HPC, ASSUMED not
  authoritative; historical — superseded by the 2026-07-11 entry above: the matrix DID complete 81/81
  and the "fast-arm-biased/uncomputable" framing was corrected).** Maintainer had no HPC access and asked the session to run the theta5s matrix
  in-container instead of on Helix. Built `runs/run_theta5s_local.py` (resumable, high-fA-first
  ordering, ≥20-min/arm per maintainer ruling — set to 30 min) + `runs/checkpoint_theta5s.py`
  (merges each ephemeral container's completed arms into the committed
  `runs/data/theta5s_summary.csv`, which carries a PROVISIONAL header). The container is
  **restart-prone (~2–13 min) and compute-limited**, so the sample is **fast-arm-biased**: only the
  fastest-firing arms complete (fail_repro control never fires ✓, small_dense fires fA≥4,
  normal_n1e3 fA12/16), while `__none`/low-fA baselines and diffuse configs wall-kill and are
  absent/truncated. **This does NOT satisfy Phase 4 and its fire-map/collapse-law/dMdt numbers must
  NOT feed the Phase-6 decision as final.** ⛔ Once HPC is available: re-run the full 81-arm matrix
  via `run_theta5s.sbatch` and RE-CHECK every downstream read (analysis, both controls, Phases 5–6,
  paper numbers) against the HPC summary; §15e is then superseded. FINDINGS §15e, REPRODUCE row 41.
- **2026-07-06 (🟡 f_A PHASE 4 TOOLING READY — 81-arm theta5s matrix built & committed, awaiting
  maintainer HPC submission).** Executor session (Opus) built Phase-4 tooling per
  `SOURCE_TERM_DESIGN.md §3`, mirroring the theta5k conventions (studied `runs/` on maintainer
  request). Committed: `runs/make_theta5s_params.py` + **81 validated params** (`runs/params/theta5s/`;
  9 configs × f_A {1,2,4,6,8,12,16,24,32}, single-knob, `__none` = byte-identical baseline);
  `runs/run_theta5s.sbatch` (array 1-81, `--time=6:00:00` wall-armor); `runs/sync_theta5s.sh`;
  `data/make_theta5s_analysis.py` (fire map + θ-rise + collapse-law fit, registered p_source≈3.3,
  whole-band over 7 fireable + control-fire-is-a-bug check; smoke-tested on synthetic data);
  `runs/harvest_dmdt_suppression.py` (read iii; smoke-tested on the Phase-3 fA=1/fA=8 pair → median
  ratio 0.934). **The matrix is NOT run — no θ quotable.** Maintainer runs
  `./sync_theta5s.sh {up,submit,watch,run,down}` on Helix; a follow-up session analyzes the
  harvest. FINDINGS §15d. Phase 4 🟡 awaiting HPC; then Phase 5 (bench5) / Phase 6 (decision).
- **2026-07-06 (✅ f_A PHASE 3 RAN — all 4 gates pass; default LITERAL byte-identity; first LIVE
  El-Badry sign).** Executor session (Opus) ran Phase 3 gates of `SOURCE_TERM_DESIGN.md §3`.
  (1) pytest 742. (2) **Byte-identity is literal, not value-diff**: pre (git worktree @919feaec,
  pre-Phase-2) == postA == postB, identical sha256 of `dictionary.jsonl` (simple_cluster,
  stop_t 0.03, pinned OMP/OPENBLAS/MKL=1); the A/A control is itself bit-identical (thread pinning
  removed the §9b FP wobble) — the `fA != 1.0` guards proven inert to the byte. (3) Screen re-run
  reproduces §2 (6/6 gates, zero git diff). (4) Live smoke fA=8 (stop_t 0.03, DEBUG): clean, 0
  freeze/no-root events, dMdt(8)<dMdt(1) and θ(8)>θ(1) in 29/29 matched segments — the El-Badry
  ṁ-suppression sign surviving the FULL COUPLED run (all prior evidence was replayed states).
  FINDINGS §15c. Phase 3 ✅; **Phase 4 (theta5s HPC matrix) next — the first maintainer-gated
  phase (sbatch submission + sync).**
- **2026-07-06 (✅ f_A PHASE 2 RAN — production wiring landed, gated default-1.0 byte-identical).**
  Executor session (Opus) ran Phase 2 of `SOURCE_TERM_DESIGN.md §3`. `cooling_boost_fA` wired into
  production: two edit sites in `bubble_luminosity.py` behind `fA != 1.0` guards (RHS `dudt`
  band-multiply at :416; L₂/L₃ component scaling at :811; `_T_INTERFACE_BAND=10**5.5` at :59),
  registry ParamSpec + `_validate_cooling_boost_fA` (rejects ≤0, cross-knob double-boost warning),
  kappa-`'auto'` text note. New `test/test_fA_source_boost.py` (9 tests); full pytest **742 green**
  (was 733; +9). fA=1 reproduces the pre-patch solve exactly (2.3271e8/5630); fA=4 raises LTotal
  ×1.35, lowers dMdt ×0.934 (El-Badry sign). Two plan corrections recorded (FINDINGS §15b): (1)
  `default.param` is auto-generated — regenerated via `python -m tools.gen_default_param --write`,
  NOT hand-edited (byte gate `test_gen_default_param`); (2) the predicted string-pin collateral did
  NOT break (fixture params come from `read_param`; `_scalar_params` feeds only the untouched IC).
  The rigorous cross-process byte-identity gate is Phase 3 (next session). Phase 2 ✅; Phase 3 next.
- **2026-07-06 (✅ f_A PHASE 1 RAN — all-9 offline coverage + edge map; θ≈1 edge prediction
  FALSIFIED in the SAFE direction).** Executor session (Opus) ran Phase 1 of `SOURCE_TERM_DESIGN.md
  §3`. (a) Coverage closed: committed partial trajectories for the two never-screened configs
  (`data/traj_normal_n1e3.csv` 61 rows, `data/traj_small_1e6.csv` 56 rows — both early-epoch, the
  §8d cliff blocks ≥5 Myr in-container) + the 2 FM1 fixtures; both new configs reproduce the §15
  dial/sign/stability, both controls stay far below fire (small_1e6 θ_max 0.25 @ fA16; stiff-5e9
  θ≈0.02). (b) Condensation-edge map (`data/make_fA_edge_map.py` → `fA_edge_map.csv`/`.png`,
  `fA_coverage9.csv`): **0/50 states reach dMdt≤0 within f_A≤128; a probe to f_A=512 finds NO edge**
  (θ driven to 6–26, dMdt stays large-positive). Per the STOP rule this falsification was written up
  (`FINDINGS.md §15a`), not tuned around: the θ≈1 edge is a *conduction-knob* (f_κ/McKee–Cowie)
  phenomenon; the source knob never touches the evaporative eigenvalue, so it has no reachable
  condensation edge — a strengthening of the solver-safety case. Phase-6 note added: "dense
  condense-first" now reads as DRAIN/stay-energy-driven, not f_A-driven condensation. No production
  code touched. Phase 1 ✅; Phase 2 (wiring) is next.
- **2026-07-06 (🧩 PLAN CONSOLIDATED TO ONE DOC + LITERATURE-CALIBRATION PHASE; 2-agent review).**
  Maintainer directives: one source of truth (no parallel plan docs), comprehensive coverage of
  ALL 9 configs, and calibration against published sims ("similar θ at similar time").
  `FA_IMPLEMENTATION_SPEC.md` was folded into `SOURCE_TERM_DESIGN.md §3` (Phases 0–6) and
  DELETED. Two agents reviewed the plan: the coverage audit produced the per-config
  expected-outcome/acceptance table (7/7 *fireable* + 2 controls-unchanged replaces the
  unsatisfiable "9/9 fire"), the grid extension {…,24,32} + bracket rule (screen laggards
  midrange/pl2 extrapolate to ~20–32), offline fixtures for the never-screened small_1e6 +
  normal_n1e3, wall-time/compliance armor, the L_leak not-scaled statement, required interaction
  warnings (incl. kappa-'auto' grid invalidation), the band-edge↔cooling-table pin test, epoch
  force-includes, and found a pre-existing latent fallback double-boost (FINDINGS §16, flagged
  not fixed). The literature agent produced Phase 5: five bespoke Lancaster-2021b-matched
  bench5 configs (n̄ 43→2.2e5), the operationalized metric (cumulative Θ_cum ∈ [0.9,0.99] +
  instantaneous 1−θ tracks ≤0.5 dex at {0.5,1,2}·t_ff capped 3 Myr; censoring + leak-split
  rules; El-Badry overlay), the predicted whole-band f_A ∈ [8,13] (vs laggard 20–32 tension the
  matrix resolves), and the circularity honesty clause. Blocking pre-step registered: PDF-verify
  L21b Table 1 into `LANCASTER_REFERENCE.md` before freezing bench params.
- **2026-07-06 (📋 f_A L1/L2 EXECUTOR SPEC LANDED; direction endorsed — spec since folded into
  `SOURCE_TERM_DESIGN.md` and deleted, see the entry above).** Maintainer confirmed
  the goal is a back-reacting in-ODE factor (not the post-hoc f_mix) and asked for an
  execution-ready plan: `FA_IMPLEMENTATION_SPEC.md` pins the two production edit sites (RHS dudt
  band-multiply + L₂/L₃ component scaling), the `cooling_boost_fA` registry wiring (standalone
  param, default 1.0 byte-identical, NOT a mode), the gate ladder with pass bars (pytest →
  byte-identity + A/A control → screen re-run → live smoke → theta5s HPC matrix), two registered
  predictions (collapse-law p_source ≈ 3.3 vs multiplier 1.82; emergent dMdt suppression vs
  El-Badry Eq 47), the traps, and a pre-committed decision tree. SOURCE_TERM_DESIGN gains §2b
  (f_mix-vs-f_A FAQ + why small-λδv cannot rescue κ_mix). No production code touched.
- **2026-07-06 (🟢 THE PHYSICAL IN-ODE SUCCESSOR IDENTIFIED — f_A source-term screen passes 4/4
  predictions 6/6; design doc landed).** The maintainer's standing wish (an in-ODE knob more
  physical than the output multiplier) is answered by moving the boost from the CONDUCTIVITY to
  the interface COOLING SOURCE (`dudt`, T<10^5.5 band only): `SOURCE_TERM_DESIGN.md` (the knob
  2×2, the lit anchors incl. the Gupta+18 f_T precedent and the f_κ = f_mix^{7/5} inconsistency
  algebra, the generalized near-front IC (★) unlocking saturation-cap/κ_mix-boundary/condensation
  branch, and the L0–L4 ladder). Offline screen (FINDINGS §15, `data/make_fA_source_boost.py`):
  continuous dial (no κ_mix saturation, no dense ceiling), dMdt FALLS as cooling rises (the
  El-Badry sign f_κ provably violates), 300/300 stable to f_A=16, no domain-edge cliff.
  Production untouched; multiplier f_mix=4 stays the shipped knob. Next: L1 stiff-fixture screen,
  then the gated param + theta5-protocol live matrix (L2).
- **2026-07-03 (✅ RESCUE LADDER SHIPPED — the never-succeeds solve class fixed; F5 de-mixed).**
  A 'structure solve failed' no-root now re-seeds hybr from the bounded legacy grid optimum
  (found-dMdt<0 roots excluded — handoff semantics intact; healthy path byte-identical). Live
  verification: dense mult4's segment-1 failure (the all-NaN mechanism, FINDINGS §14) is rescued
  and the run fires. Full pytest 617 green. Also: theta5_knob_choice.png had colliding panel
  titles that read as f_κ/f_mix mixing — retitled, 'crash windows (pre-fix)', caption states the
  two panels are two different knobs; theta5k remains the like-for-like comparison.
- **2026-07-03 (✅ DENSE-EDGE NaN TICKET RESOLVED-AS-UNDERSTOOD — never-solved default + machine-flippable
  root at the domain edge).** DEBUG repro of theta5's small_dense mult4/mult8 NaN arms: the NaN is the
  registry DEFAULT (`bubble_Lloss=np.nan`) written verbatim because the β–δ solve never succeeds ("structure
  solve failed", hybr pushed to unintegrable (β,δ) by the boosted balance). Locally the SAME arms recover and
  FIRE (mult4 at segment 3; mult8 after 9 fails) — the f≈4–8 root sits on the integrable-domain edge and ULP
  nondeterminism picks the side. Dynamical fate unaffected (same collapse t either way — Eb ODE uses β-side
  Edot). Fire-map NAN legend corrected; quote the theta5b fine arms as the dense calibration evidence.
  FINDINGS §14.
- **2026-07-03 (✅ THETA5N RAN — the ninth config fires NATIVELY; law point 7; window [4, 4.5] now 7/7).**
  normal_n1e3 (1e6, n1e3, sfe 0.01 — M_cluster 1e4) crosses θ=0.95 unboosted at t≈2.5 Myr
  (θ₀=1.047): route-a demonstrated live. Law predicts f_fire=1.16, measured 1.0 (resid 0.065
  dex; combined rms stays 0.064 over SEVEN configs, θ₀ 0.51–1.05). Every multiplier arm fires
  (boost just moves the crossing earlier; recollapse fates). Kappa fires 2–12, DRAINs at 16 —
  the race again. FINDINGS §13; 9-row fire maps + 7-point law check regenerated; report §16.6 +
  the new "shipped model" section (equation, param block).
- **2026-07-03 (🔍 PRIMARY-SOURCE RECHECK + NON-MONOTONICITY BUG-HUNT + 9th CONFIG).** Maintainer
  verified the freeze doc against Weaver II (§V: classical front budget already 60/40 — the
  reversal is close by), the TRINITY method paper (T ∝ Ṁ^{2/5}: no Ṁ<0 profile family — the
  gate is the closure's domain edge; fix #4 = new profile family, confirmed research-grade), and
  Tan–Oh–Gronke §2.2 (unique planar eigenvalue → the +1121→−85 swing is more likely fast-moving
  BCs / bracket behavior than two branches). theta5k flip-arm audit: stale-dict hypothesis tested
  and REFUTED (the found bit-identical theta_first pairs are the small_1e6⇄large_diffuse
  M_cluster/nCore degeneracy — registered as ⚡ ONE-check note); all monotone quantities monotone
  in f_κ; outcome flips are photo-finish races decided in the first ~30–50 segments. Band
  extended to NINE configs: normal_n1e3 (1e6, n1e3, sfe 0.01, PL0) via `runs/make_theta5n_params.py`
  (15 arms, both knobs, REPRODUCE #34, ⬜ ready). Dense k6-vs-k8 freeze-watch trace queued as the
  bracket-vs-physics discriminator.
- **2026-07-03 (✅ THETA5K RAN — zero freezes; fix #1 validated at scale; NO whole-band f_κ).**
  First rule-compliant kappa matrix (56 arms, stop_t=5): 21 FIRED / 18 NOFIRE / 12 DRAIN /
  5 CONDENSE (the no-root handoff, landing exactly on §9a's old "dead window" cells — sc
  8/12/16 at θ 0.533/0.587/0.624, dense 6, pl2 16). Fire set non-monotonic for *physical*
  reasons (fire-vs-condensation race); θ_max rises ~monotonically. Best single f_κ fires 5/6
  (k12, misses sc); multiplier [4, 4.5] fires 6/6 → production-knob choice now measured
  like-for-like. Old sc "fires at k16" exposed as pre-fix artifact → 'auto' demotion hardened.
  FINDINGS §12; `data/theta5k_fire_map.csv`; `theta5k_{fire_map,theta_rise}.png`.
- **2026-07-03 (✅ FIX #1 LANDED — no-root streak ⇒ momentum handoff; theta5k READY).** A
  50-segment no-physical-root streak now ends the implicit phase like `cooling_balance` does
  (`termination_reason="no_physical_root_handoff"`; a *fate*, not a trigger — harvest counts it
  like DRAIN). Verified: handoff demo (`runs/drive_noroot_handoff_check.py`, threshold 3, f_κ=8 →
  diagnosis → transition → momentum → clean end) + structural inertness + full pytest 614/614.
  NB: local byte-identity gates are UNATTAINABLE without pinned BLAS threads — an A/A control
  (identical code, two runs) differs at the SN noise floor from row 1; any future local
  byte-identity claim needs a same-code A/A companion (KAPPA_FREEZE_MECHANISM §7.1b).
  The rule-compliant kappa re-validation is committed ready-to-run: **theta5k** (8 configs ×
  f_κ {1,2,4,6,8,12,16}, stop_t=5; `runs/make_theta5k_params.py` 56/56 validate,
  `runs/run_theta5k.sbatch` array 1-56). KAPPA_FREEZE_MECHANISM §7.1 has the details.
- **2026-07-02 (✅ KAPPA FREEZE MECHANISM IDENTIFIED — §9a's "non-monotonic breakdown" was a false
  inference, as the maintainer suspected).** The freeze is the solver hitting the McKee–Cowie
  evaporation→condensation domain boundary: the dMdt eigenvalue physically crosses zero exactly at
  cooling balance, the `dMdt>0` gate refuses the condensing root (live repro: hybr converges to
  dMdt=−84.76 at f_κ=8), and the runner freezes instead of handing off. 34/38 sweep freezes were at
  θ≥0.8 (would-fire); all 23 "non-monotonic" arms explained without physics bands; 1 freeze at f_κ=1
  proves the mode pre-exists the knob. Multiplier stays the production knob (structurally immune —
  it never touches the eigenvalue). Fix ladder + instrumentation: `KAPPA_FREEZE_MECHANISM.md`,
  FINDINGS §9b, `data/kappa_freeze_autopsy.csv`. A rule-compliant kappa re-validation ("theta5k")
  becomes worthwhile only after fix #1 (no-root-streak ⇒ momentum handoff) lands.
  **UPDATE 2026-07-03: theta5k RAN (FINDINGS §12)** — 56/56 proper fates, zero freezes (fix #1
  validated at scale; the 5 condensation handoffs are the old "dead window" cells). Kappa's
  fire set stays non-monotonic for physical reasons (fire-vs-condensation race) and **no single
  f_κ fires the whole band** (best 5/6 at f_κ=12) — the multiplier window [4, 4.5] (6/6) is now
  the measured, like-for-like production-knob argument. 'auto' demotion hardened (its grid
  embeds pre-fix artifacts, e.g. sc f_κ=16 "fire" → rule-compliant CONDENSE at θ=0.624).
- **2026-07-02 (✅ THETA5B RAN — window measured [4, 4.5]; law validated out-of-sample; the fire-vs-drain
  race → `FINDINGS.md §11`, `data/theta5_fire_map.csv`, `data/theta5_law_check.csv`).** The 43-arm referee
  matrix completed on Helix. Fine f_fire: sc 2 · mid 2.5 · dense-edge 2.5 · be 3.5 · diffuse 3.5 · pl2 4;
  small_1e6 never (≤8). **The whole-band window is [4, 4.5]** (3.5 misses pl2; 5 drops midrange via over-boost
  Eb-drain — its ceiling moved down from theta5's 8). The θ₁-collapse law predicted every fine threshold at
  **rms 0.064 dex out-of-sample** — the constant-f + law model stands the referee test. **NEW systematic
  (corrects §10/ch.16 "no dead windows" phrasing):** below threshold, boost can PREVENT firing — Eb-drain
  hands off to momentum before θ crosses (sc fires at 2, NOT at 2.5–3, again at 3.5+); healthy runs, real
  fire-set gaps, mechanistically transparent (unlike kappa's §9a solver freezes). Diffuse long arms: f=2 fires
  at t≈5.04 Myr (horizon-dependence — 5 Myr is an operational choice); f=1@8Myr confirms the native peak is
  captured by 5 Myr (📏 rule-1 self-check ✅). Dense edge fires at every fine arm (θ 0.95–1.01) — theta5's
  NaNs were intermittent, ticket downgraded. **f_mix=4 adoption unchanged and strengthened** (now measured as
  the window's interior).
- **2026-07-02 (✅ MAINTAINER RULING — momentum-then-recollapse is acceptable physics; f_mix = 4 ADOPTED).**
  Maintainer, verbatim intent: a cloud that fires `cooling_balance`, hands off into the momentum phase, and
  subsequently recollapses is "completely fine" — fire-then-recollapse is an **outcome**, not a failure mode.
  Consequence: the f_mix decision gate (§10 point 5) is cleared and **f_mix = 4 is the adopted working value**
  (minimal single constant that fires the normal-GMC band; production default stays `cooling_boost_mode='none'`
  — 4 is the documented recommended setting). Still flagged as pathologies: the f=8 over-boost Eb-drain
  (momentum WITHOUT firing) and the dense-edge NaN rows. theta5b (fine bracket + diffuse stop_t=8) remains
  queued as the REFEREE-DEFENSE refinement — it measures the workable window around 4, it no longer gates the
  decision. Docs reconciled: ⭐⭐ VERDICT/OPEN/BEST-PATH, INDEX §0/§3, FINDINGS §10, REPRODUCE #29,
  CONTAMINATION register, storyline ch.16, figure/scorecard builder wording.
- **2026-07-02 (✅ THETA5 MATRIX RAN — the first fully rule-compliant `multiplier` calibration →
  `FINDINGS.md §10`, `runs/data/theta5_{summary,calibration}.csv`).** 32/32 arms reached 5 Myr or a
  physics end on Helix; θ = θ_max from `dictionary.jsonl`. **OPEN(1) is CLOSED.** Results: blowout
  under-read diffuse θ by ~2× (large_diffuse θ₀=0.535, peak at t≈4.9 — the old "diffuse needs f≈60"
  is dead; **it fires at f_mix=4** and survives to 5 Myr); multiplier θ₁-collapse law
  **f_fire ≈ 1.4·(0.95/θ₀)^1.8** (leverage θ∝f^0.55, knob-specific vs kappa's 0.27); **f_mix=4 fires
  the whole normal-GMC band** (θ_max at fire 0.96–1.04), f_mix=2 only the compact config; route-a =
  small_1e6 (de-conflated: same nCore as the firing diffuse config) + fail_repro (handoff untouched —
  §8b acceptance PASSES). **New failure modes:** fire-then-recollapse (every dense-core config that
  fires at f≥4 shell_collapses — §8a's question, now emergent → **maintainer physics call before
  pinning f_mix**) and over-boost Eb-drain (midrange@8 reaches momentum WITHOUT firing; dense edge
  NaNs — multiplier's gentler analog of §9a's dead windows). **NEXT: pin f_mix (2 vs 4 vs a finer
  (2,4] bracket) pending the recollapse call; diffuse stop_t=8 spot-check; dense-edge stiffness
  ticket.**
- **2026-07-01 (RECONCILIATION: pt2⇄pt3 merged, contamination register created, 📏 protocol codified,
  `multiplier` re-calibration matrix committed).** The parallel `feature/transition-trigger-pt3` line (819-sweep
  fold-in + `cooling_boost_kappa='auto'`) was merged into the post-PR#717 mainline; its FINDINGS section
  renumbered **§9** with post-merge flags (sweep `stop_t=2` vs 📏 rules; 'auto' = per-cloud lookup vs the
  single-constant + route-a decisions; the **§8e⇄§9 kappa tension** left OPEN, not silently resolved).
  **`CONTAMINATION.md` created** — rules (a)–(e), the full per-artifact register (what is quotable), the ⛔
  do-not-quote list (§14 numbers, blowout-θ f_κ-to-fire, the 63-cell grid values, pre-#715 fates, observer θ).
  **The 📏 standard protocol is now executable**: `runs/params/theta5/` (8 configs × f_mix {none,2,4,8}, all
  `stop_t 5`, 32 params validated through `read_param`), `runs/run_theta5.sbatch` (Helix array),
  `runs/harvest_theta_max.py` (θ_max from `dictionary.jsonl` accepted rows — the sanctioned measurement).
  INDEX rewritten (era map E1–E7, all 21 docs statused, branch archaeology incl. the stranded
  `feature/PdV-trigger-term` El-Badry-overlay commit `3e68143`). OPEN(1) re-written (the "✅ calibrated +
  cheap confirmation" claim retracted). **NEXT: run the theta5 matrix on HPC** → re-fit θ₀/p + the
  θ₁-collapse law for `multiplier` → pick the single physical f_mix → route-a boundary.
- **2026-07-01 (pt3: sweep folded in + `cooling_boost_kappa='auto'` shipped — merged into this line same
  day; see the ⚠️ post-merge flags in `FINDINGS.md §9` and `CONTAMINATION.md`).** The 819-combo grid ran on
  Helix 2026-06-29 (786/819 ok, 10h17m — `data/sweep_report.txt`, `data/summary.csv`; the fit step landed as
  `data/fkappa_nH_sweep.csv` + `fkappa_nH_sweep.png` in commit `880334f` but no doc recorded it until now).
  **De-conflation verdict: n_H-only REFUTED** — at fixed nCore the measured f_κ_fire spreads up to 32× across
  (mCloud, sfe); sfe is a strong secondary axis and mCloud dominates the dense end (1e7 M☉ fires at f_κ=1 for
  n≥3e3). **What does collapse it: the starting deficit** — log10 f_κ_fire = 0.041 + 3.755·log10(0.95/θ₁)
  over all 41 fired-above-1 cells (corr 0.968, rms 0.116 dex vs 0.21 dex for the best input-space fit), i.e. a
  universal leverage θ ∝ f_κ^0.266 — confirming §6's pessimistic developed-epoch exponent (~0.3), refuting the
  optimistic 0.63 snapshot estimate (`data/make_fkappa_theta1_collapse.py` → `fkappa_theta1_collapse.{csv,png}`).
  **Production mode shipped:** `cooling_boost_kappa='auto'` resolves at load (read_param Step 7 resolver) to the
  measured f_κ_fire via trilinear log-space interpolation of the 63-cell grid (`trinity/_input/fkappa_auto.py`;
  mCloud axis = pre-SFE `mCloud_input`; hull-clamped with warning; censored diffuse/high-SFE corner pinned at the
  sweep ceiling 64 with a may-not-fire warning; numeric values pass through untouched → default byte-identical).
  Tests: `test/test_fkappa_auto.py` (9). Acceptance: `runs/params/fkauto_verify.param` (1e5 M☉ GMC, sfe 0.03,
  n 1e3 — auto→12) must fire cooling_balance and reach momentum (`data/make_fkappa_auto_verify.py`, REPRODUCE #26).
  ⚠️ Written on the pt3 branch **without knowledge of the same-day pt2 entries below** (direction correction,
  knob correction, §8e) — the sweep's `stop_t=2`/fired-by-2-Myr metric violates 📏 rules 1+2, and 'auto' is a
  per-cloud f_κ(properties) lookup, in tension with the single-constant + route-a decisions. Status: **opt-in,
  PROVISIONAL** pending the 5 Myr/θ_max re-measurement.
- **2026-07-01 (✅ DIRECTION CORRECTED — θ is an OUTPUT; Rung A f_κ reinstated, θ_elbadry demoted to opt-in →
  `FINDINGS.md §8c`, `data/gate_prototype.csv`).** Maintainer steer: enforcing θ (`theta_target`/θ_elbadry)
  broke the causal structure — θ should *emerge* from a boosted cooling **mechanism**, not be set by hand.
  Evidence it matters: the PdV double-counting (§8b) is a **direct symptom of enforcement** — `L_loss=θ·L_mech`
  is blind to whether the loss is radiative or PdV, so it over-drains PdV-dominated clouds. **f_κ
  (`multiplier`) scales only the radiative channel, so it *cannot* over-drain a PdV-dominated bubble** —
  demonstrated: the PdV **regime-gate** prototype (`data/_theta_elbadry_gated_runner.py`) makes `fail_repro`
  expand to 500 pc (large_radius) instead of velocity_runaway, and the θ-gate ALONE does it (ebpeak irrelevant)
  — i.e. the gate merely re-derives, by hand, the selectivity f_κ has for free. **Corrected plan:** (1) Rung A
  f_κ (`multiplier`, already shipped, gated default-off) is PRIMARY; (2) El-Badry/Lancaster = **calibration
  target** for the *emergent* θ, not an enforced value; (3) set f_κ at a **physical** value and **accept
  diffuse route-a non-transition** (maintainer: "diffuse clouds may never enter momentum — the physics never
  allows it"); (4) massive/PdV clouds ride the PR #715 handoff, not θ; (5) **`theta_elbadry` stays as a
  documented opt-in override** (`THETA_ELBADRY_SPEC.md`). This RE-PROMOTES Rung A (scalar f_κ) — *not* Rung B
  (structural κ_mix, still SHELVED). §8/§8a/§8b results stand as evidence; the *direction* they were gathered
  under (enforce θ) is corrected here.
- **2026-07-01 (⚠️ COURSE-CORRECTION — merged PR #715 high-mass handoff; imposing El-Badry θ REVERSES it →
  `FINDINGS.md §8b`, `data/newcode_default_vs_theta.csv`).** Maintainer merged
  `bugfix/high-mass-cluster-transition-without-ebpeak` to `main`: phase 1b now **routes finite `Eb≤0` →
  momentum** (`classify_energy_collapse`, `ENERGY_HANDOFF_FLOOR=1e3`) instead of `ENERGY_COLLAPSED`; phase 1a
  gains `cooling_balance` parity. Merged main into this branch (code auto-merged clean; my Pb-fix coexists) and
  re-ran. **Result: on the merged code, `fail_repro` (5e9,n1e2) DEFAULT → large_radius (expands to 500 pc via
  momentum) — the fix works; but with `theta_elbadry` imposed it → velocity_runaway (v2=−500 inward collapse).
  Imposing El-Badry θ RE-BREAKS exactly the massive clouds PR #715 fixed** (same flip for pl2_steep: default
  expanding vs θ-imposed velocity_runaway). Root cause (regime error, ties to `HIMASS_HANDOFF_PLAN.md` §1): the
  high-mass turnover is **PdV/inertia-driven, not radiative** (radiative ~1% of L_mech; PdV/L_mech≈1.4).
  El-Badry θ is a *radiative* ratio — imposing θ·L_mech there adds a fake radiative sink ON TOP of the real PdV
  sink (double-counts the loss; PdV is already in `Edot_from_balance`) → crashes the bubble inward. **§8/§8a
  firing/threshold/max-gate results still hold; their massive-cloud FATE conclusions do NOT.** ⇒ **the
  `theta_elbadry` SPEC needs a REGIME GATE** (apply only where radiative dominates / `cooling_balance` engages
  natively; gate OFF when PdV/L_mech≳1 and defer those clouds to the handoff). **Stage B is NOT ready; the
  earlier "Stage A clean → Stage B" is retracted for the massive-cloud regime.**
- **2026-06-30 (STAGE A — θ_max SWEEP RAN → `FINDINGS.md §8a`, `data/sweep_tmax_fate.csv`; the SHELL_COLLAPSE
  question is now a maintainer physics call, NOT a knob).** Swept θ_max∈{0.80,0.85,0.90,0.95,0.99} × {pl2_steep,
  simple_cluster} to 5 Myr. **All 10 recollapse**, at the *same* fire (~0.01 Myr) & collapse (~0.06/0.13 Myr)
  times regardless of cap. Below 0.95 the transition still fires — via **`ebpeak` (PdV)**, because imposing
  θ≥~0.80 pushes `Edot_from_balance≤0` (stock's native θ~0.66 doesn't → stock never fires these; §6a). ⇒
  **reading (b) "θ_max too aggressive" is REFUTED — the cap is not the lever;** recollapse is intrinsic to these
  dense compact clouds transitioning. The fork is now clean: **(a) physical** recollapse of a weak-cluster/dense
  core (TRINITY's dedicated clean SHELL_COLLAPSED fate) vs **(b′) momentum-phase fidelity** (does TRINITY
  recollapse a bubble El-Badry keeps expanding?) — and (b′) is **outside this workstream's scope** (a momentum
  module question, not a trigger-design one). **Recommendation:** pick θ_max=0.95 or 0.99 (identical for dense
  clouds), treat SHELL_COLLAPSED as the correct fate for these configs, and **Stage B can proceed** pending the
  maintainer's OK on that physics reading. `SHELL_COLLAPSED` confirmed = shell recollapse (v2<0, R2<coll_r=1pc;
  `run_transition_phase.py:772/789`).
- **2026-06-30 (STAGE A — SHADOW RAN; 9 configs to ≥5 Myr → `FINDINGS.md §8`, `data/shadow_te_fate.csv`).**
  Ran the §3 El-Badry-θ logic via `data/_theta_elbadry_runner.py` (monkeypatch, no `trinity/` edit), trigger
  `cooling_balance,ebpeak`, λδv=3, θ_max=0.99. **Two of three Stage-A questions resolved by data:** (1) **§6
  `max(resolved,target)` gate SAFE** — resolved-wins **0/N** across all 9 configs, so El-Badry θ ≥ TRINITY's
  native θ everywhere and `max()`==direct assignment (no diffuse-end misbehavior). (2) **θ(n) + firing threshold
  behave** — n=10→θ=0.897 (energy-driven until ebpeak), n=100→0.965, n≥1e4→0.99(cap); n_fire≈48–50 confirmed.
  **(3) SHELL_COLLAPSE is CONFIRMED patch-induced (resolved from committed data, no new runs needed):** dense
  compact clouds (n≥1e4, θ=0.99) fire at t<0.02 Myr then **SHELL_COLLAPSE** (endcode 4 = *clean* fate, not error);
  diffuse well-powered reach STOPPING_TIME expanding (small_1e6→254 pc; diffuse_probe→139 pc); diffuse
  *under*-powered (large_diffuse_lowsfe sfe=0.01) collapses late (14 Myr). The stock baseline never fires these:
  §6a's committed `ebpeak_8config_xcheck.csv` shows native *radiative* θ (what `cooling_balance` tests) peaks
  **~0.66 compact / 0.17 diffuse** — far below 0.95 — so stock keeps them energy-driven; imposing El-Badry
  θ=0.99 (native 0.66→0.99) is what collapses them (resolved-wins=0 says the same from the shadow side). The
  dense-baseline full-runs (`data/_baseline_runner.py`) were **abandoned** — hours-scale + repeated container
  restarts — and are **not needed** for this conclusion. The 2 non-results (fail_repro mCloud 4.5e9
  energy-collapse; small_dense nCore 1e6 β-δ MonotonicError) are **pre-existing extremes, not patch-induced**.
  **What's left is a PHYSICS DECISION, not a run:** El-Badry says θ≈0.99 is *correct* for dense clouds, yet his
  θ→1 bubbles still expand (don't recollapse) — so is TRINITY's collapse (a) physical for these weak-cluster/dense
  configs, or (b) an artifact (θ_max=0.99 too aggressive, or the momentum phase mishandling a ~0-thermal-energy
  bubble)? **Cheapest discriminator = a θ_max∈{0.90,0.95,0.99} sweep on 2–3 dense configs** (separates "cap too
  high" from "collapses regardless"), far cheaper than baselines. **Stage B production stays BLOCKED until this
  physics call is made.**
- **2026-06-30 (CAPSTONE SPEC written — `THETA_ELBADRY_SPEC.md`; the path is off-paper-ready).** Consolidated
  every resolved decision into one implementation-ready spec for the gated `theta_elbadry` mode: **3 registry
  params** (`cooling_boost_mode='theta_elbadry'`, `cooling_boost_lambda_dv` default 0=off/set 3.0,
  `cooling_boost_theta_max`=0.99) + **1 branch** in `effective_Lloss_from_params` (`get_betadelta.py:360`) that
  computes θ=A_mix√(λδv·n_amb(R2))/(11/5+…) per step (A_mix=3.5; n_amb via `get_density_profile(R2)` pc⁻³→cm⁻³)
  and reuses the *verified* `theta_target` (1−θ) budget — no κ_mix port, no structural change. Byte-identical-off
  by construction (`mode='none'`→`Lcool+Lleak`). Pairs with `transition_trigger='cooling_balance,ebpeak'` (PdV).
  Flags the `max(resolved,target)` subtlety as a **test gate** (log where resolved wins; switch to direct
  θ_target if the diffuse end misbehaves). Test plan: unit (byte-identical-off + θ matches the calculator) →
  8-config ≥5 Myr on-runs (fate pattern vs n_fire≈50, first-crossing firing, resolved-wins fraction). NEXT =
  implement per the spec. No production code; no sims.
- **2026-06-30 (Lancaster 2021 Paper II read → λδv CALIBRATION resolved + route-a confirmed at the diffuse end;
  `LANCASTER_REFERENCE.md` §7).** Read ApJ 914, 90 (Lancaster+2021 Paper II, the 3D-sims validation). **Verified
  numbers:** Θ≡Ė_cool/Lw = **0.9–0.99** (retained 1−Θ~0.1–0.01, ∝t^{−1/2}, Eq 10) for ALL models; αp~1.2–4;
  fractal interface D~2.4–2.7; turbulent vt~200–400 km/s (their δv analogue); **density range nH≈40–2×10⁵**
  (GMC/clump, NOT diffuse). **Paper-ID fix:** ApJ 914, 90 is **Paper II (sims)**, not the theory paper (I had it
  backwards). **Decisive for route a/b:** Lancaster's plateau is **GMC-only (nH≳40)** and AGREES with El-Badry
  there (both Θ≈0.9–0.99, both rising with n) — but Lancaster **does NOT test diffuse ISM**, so it does not
  contradict El-Badry's √n drop at low n. ⇒ **route-a stands: diffuse clouds (nH≲50) genuinely stay
  energy-driven** (answers "can some clouds not transition" — yes, the diffuse ones). **λδv CALIBRATION:** to
  fire (θ≥0.95) across Lancaster's whole momentum-driven GMC range (down to nH~40) needs **λδv≈3–3.5**
  (n_fire≈48) — which is *also* El-Badry's own calibration value (A_mix=3.5 fit at λδv=3). **Adopt λδv≈3**
  (n_fire≈50). (λδv=1 → n_fire=143 would wrongly exclude Lancaster's nH~40–140 clouds.) No code; no sims.
- **2026-06-30 (Lancaster 2025 read + the PdV-inclusivity question answered; new doc `LANCASTER_REFERENCE.md`,
  imprint banners on both reference docs).** Read arXiv:2505.22730v1 (Lancaster 2025, "Co-Evolution of Wind
  Bubbles & Photoionized Gas I" — the CEM). **Confirms** Lancaster's `θ ≡ Ė_cool/Lw` = El-Badry's = TRINITY's,
  and the (1−θ) ED solution (Eq 1–3) is identical to El-Badry. **Landscape fix:** the θ~0.9–0.99 anchor is
  **Lancaster 2021c** (2104.07722), *not* this 2025 paper — so "2021" was right for that claim; the 2025 PDF is
  a newer, different paper. **PdV ANSWER (the branch's core question):** El-Badry/Lancaster/TRINITY θ is the
  **cooling fraction, PdV-EXCLUSIVE by definition**. Lancaster makes the split explicit — **θ (cooling) vs αp
  (momentum/PdV enhancement, ṗr/ṗw)**, coupled via the mixing velocity ⟨vout⟩; the energy→momentum transition
  is fundamentally the **αp→1 (momentum/PdV) budget**, not cooling alone. In source, TRINITY mirrors this:
  `cooling_balance` (θ≥0.95) uses Lloss only = **PdV-exclusive**; `ebpeak` is `Edot_from_balance = Lmech − Lloss
  − 4πR2²v2·Pb ≤ 0` (`get_betadelta.py:475`; code comment "PdV-inclusive"). **So the maintainer's intuition is
  right: for massive clusters (large Lw ⇒ large PdV) the PdV-inclusive `ebpeak` fires earlier and is the more
  physical transition.** Recommendation: pair the El-Badry θ_target (cooling boost) **with `ebpeak`**
  (transition_trigger='cooling_balance,ebpeak', first-fire) — the boost lowers retained energy (1−θ)Lmech and
  ebpeak checks whether PdV drains the rest; especially for massive clusters. Imprint banners added so future
  visits use `ELBADRY_REFERENCE.md`/`LANCASTER_REFERENCE.md` instead of re-reading PDFs. No code; no sims.
- **2026-06-30 (n-mapping RESOLVED + verified — `data/make_nmap_verify.py`).** Re-derived that El-Badry's `n`
  enters θ *only* as a proxy for the Weaver combo `R²Pb^{3/2} = K_W(1−θ)Ė_in ρ₀^{1/2}` (K_W=0.0383, verified).
  His L_int is interface-local (needs only R, Pb, λδv, T_pk). **Two faithful options:** (A) local cloud density
  `n_amb(R2)=get_density_profile(R2)` → his closed form (reuses A_mix=3.5); (B) direct `θ=L_int(R2,Pb)/Lmech`
  (no n-mapping; saturation emerges from Pb). **Verified (A) is faithful at equilibrium:** `n_eff/n_amb` median
  ~0.66–0.88 across all 6 cleanroom configs — the ~0.7 offset is exactly the dropped (1−θ)² cooling factor; it
  diverges (≤6 dex) only at the early-core/blowout extremes where El-Badry's late-time model is invalid anyway.
  **Recommend (A) for the first cut** (simplest, reuses calibration, θ_max ceiling handles the saturated dense
  core); (B) is the robust upgrade. Density is the local n_H at R2 (TRINITY's n is already n_H — matches
  El-Badry). Detail: `ELBADRY_REFERENCE.md` §7. No production code; no sims.
- **2026-06-30 (VERIFIED the θ_target mechanism + recorded the discrete-SN vs continuous-SB99 caveat).**
  Confirmed in source (`ELBADRY_REFERENCE.md` §9) that TRINITY's `cooling_boost_mode='theta_target'` IS
  El-Badry's (1−θ) budget: `Lloss = max(Lcool+leak, θ·Lmech)` (`get_betadelta.py:355`) ⇒ energy ODE gets
  `dEb/dt=(1−θ)Lmech−PdV`; trigger (`run_energy_implicit_phase.py:1207`) fires at `θ_eff=Lloss/Lgain≥0.95`,
  `Lgain=Lmech_total`; all three sites (residual/ODE/trigger) consistent; off=byte-identical. **Gap:**
  `cooling_boost_theta` is a *constant* — El-Badry's θ(λδv,n) needs a per-step density-dependent value (new mode
  `'theta_elbadry'` evaluating Eq 37/38 from λδv + n_amb(R2)) + a **θ_max<1 ceiling** (else (1−θ)→0 stalls the
  bubble at GMC density). **Maintainer caveat folded in (`ELBADRY_REFERENCE.md` §8):** El-Badry's *discrete-SN*
  machinery (Δt_SNe, §6.1 early-shock invalidity, the λδv estimate Eq 22-23) is obsolete for TRINITY's
  *continuous SB99* input — BUT the closed-form θ(λδv,n) is **Δt_SNe-independent** so it carries no SN-timescale
  baggage; calibrate λδv to Lancaster (not the Eq-22 estimate); winds-vs-SNe is second-order for θ. No code; no sims.
- **2026-06-30 (FULL El-Badry+2019 read → REVISED PLAN: use his analytic θ(λδv,n) as TRINITY's θ_target;
  new docs `ELBADRY_REFERENCE.md` + `data/make_elbadry_theta.py`).** Read the whole 32-page paper
  equation-by-equation (transcribed into `ELBADRY_REFERENCE.md` so future sessions skip the PDF). **The pivotal
  fact:** El-Badry's cooling efficiency **`θ ≡ L_int/Ė_in` IS TRINITY's trigger θ = L_cool/L_mech** (Ė_in =
  ESN/Δt_SNe = L_mech), and he gives a **3D-calibrated closed form** `θ = A_mix√(λδv·n)/(11/5 + A_mix√(λδv·n))`,
  A_mix=3.5 (Eq 37/38). Validated our calculator reproduces his fiducial θ(1,1)=0.61. **So the cleanest path is
  to feed his θ(λδv,n) straight into TRINITY's existing gated `cooling_boost_mode='theta_target'`** — El-Badry
  *himself* endorses this ("our solution … can be easily implemented in any application that uses the Weaver
  model", p.26). This **drops the κ_mix-into-the-Weaver-ODE injection** (the saturating/unstable path) entirely.
  **Three findings that reframe earlier work:** (1) **"dense θ low" is walked back** — El-Badry's √n (rising) +
  his own note "at molecular-cloud densities, high θ" + Lancaster θ~0.9-0.99 all agree dense clouds have HIGH θ;
  our self-consistent solve's dense θ~0.35 is the **outlier/artifact** (kprime, hard-max, wrong epoch). (2)
  **Parker conductivity is NEGLIGIBLE** (κ_P ≪ κ_S; El-Badry §3.1) — *correcting my prior-turn claim* it was a
  load-bearing missing piece. (3) **Saturation mainly affects Mhot (~15-20%) + early-time numerics, NOT θ** —
  *also tempering my prior-turn claim*; it's not the key to the cooling efficiency. **The "fate" picture (your
  framing) falls out cleanly:** at λδv=1 the firing threshold is ambient n≈143 cm⁻³ — GMC cores (≥1e2) fire,
  diffuse ISM doesn't, a falsifiable critical-density prediction. **Honest caveats (don't assume):** GMC θ rests
  on EXTRAPOLATING El-Badry's √n beyond his tested n≤10 (supported by Lancaster, unvalidated at n>10); the
  n-mapping (ambient density at the shell, NOT nCore) must be pinned; θ is a LATE-TIME (≥5 Myr) equilibrium;
  λδv∈[0.1,10] is the free knob to calibrate. No production code; no sims.
- **2026-06-30 (κ_mix TIME-RESOLVED θ — the blowout metric was the WRONG epoch; walks back "only 1/6 fires").**
  `make_kmix_theta_trajectory.py` re-solved κ_mix across ~14 rows/config of the implicit phase (not the single
  near-blowout row §2 used). **θ peaks EARLY (high Pb ⇒ κ_mix most dominant) and decays — blowout is the
  low-θ *tail*.** So §2 sampled the minimum: `be_sphere`/`midrange` (n=1e4) blowout θ≈0.23 but
  **trajectory-max 1.84/1.14** (would fire). Robust conclusions: λδv still saturates (not a dial); **dense
  (n≥1e5) stay low** (θ_max≲0.5) — that ceiling holds. **Open:** the decisive **early high-Pb epochs FAIL to
  solve** with the hard-max injection (0/4 early rows every config; the baseline OFF solve succeeds there) →
  mid-cloud firing is *plausible but unconfirmed*. Also found a **faithfulness bug** shared with SPEC §3:
  κ_mix ∝ n ∝ 1/T, so the κ_mix-regime kprime is **−1/T, not 0**. **Next step (the clear single one):** a
  **smooth-max + correct-kprime** injection (`κ_eff=κ_S(1+R^s)^{1/s}`, kprime `(1/T)[2.5−3.5R^s/(1+R^s)]`) that
  survives the early phase, then re-run §2b. Updated `KMIX_SELFCONSISTENT.md` (§2 superseded-banner, new §2b,
  §3 routes), SPEC §3 (kprime), FINDINGS, INDEX. No production code; no sims.
- **2026-06-30 (Pb-collapse fix APPLIED to production — the FIRST production code change in this workstream;
  maintainer-authorized).** Applied `PB_COLLAPSE_GUARD_FIX.md`: in `run_energy_implicit_phase.py` the
  phase-boundary reconciliation snapshot now **skips the Pb recompute on the `energy_collapsed` exit** (Eb<0 →
  `compute_R1_Pb` gave the garbage `Pb=−1.6×10¹⁸`) but **still `save_snapshot()`s** the last-healthy state so
  `ENERGY_COLLAPSED` (code 51) reaches the output. **The `else: save_snapshot()` was essential** — the
  failing-first test caught that skipping the whole block dropped the end code (`code None`); fixed, the test
  goes red→green. **Gates:** new `test/test_energy_collapse_snapshot.py` (heavy collapsing cloud, end-to-end)
  red on `main` then green; healthy-run regression **equivalent** — and surfaced a finding: **trinity is NOT
  bit-reproducible run-to-run** (two same-code runs differ in 3 SN-feedback terms `F_ram_SN/Lmech_SN/pdot_SN`
  at ~1e-22, BLAS-threading noise; **all physics fields bit-identical**), so the "byte-identical" gate is
  qualified accordingly; full `pytest` **596 passed**. Behaviour identical for every non-collapsing run (the
  change is an `if termination_reason != "energy_collapsed":` wrapper + an `else`). Reconciled `INDEX.md` and
  `PB_COLLAPSE_GUARD_FIX.md` (status → APPLIED).
- **2026-06-30 (SELF-CONSISTENT κ_mix injected into the REAL solver — decisive, tempers the GO; new doc
  `KMIX_SELFCONSISTENT.md`).** Built `data/make_kmix_selfconsistent.py`: monkeypatches the conduction in
  `bubble_luminosity.py` (RHS site :406) and re-runs the full production `get_bubbleproperties_pure()` with
  `κ_eff = κ_Spitzer·max(1, R)` on the 6 cleanroom configs (via `make_da_replay` state rebuild) + 2 fixtures.
  **Gates pass:** G1 identity **bit-identical (0.0)** off, G2 replay vs logged `bubble_Lloss` ≤7e-7. **Physics
  (the decisive part):** (1) κ_mix raises resolved θ in all 6 and the solver is stable across the sweep — GO
  confirmed self-consistently; BUT (2) θ **SATURATES by λδv≈0.01** (κ_mix swamps Spitzer at tiny λδv) → λδv is
  **not a tunable knob** ⇒ the "pin λδv to Lancaster" step is **RETIRED**; (3) the saturated θ is
  density-**mismatched** — diffuse overshoots (θ=1.54, fires) but mid/dense plateau **low** (0.23–0.35 ≪
  Lancaster 0.9–0.99), only **1/6** reaches the 0.95 trigger ⇒ **κ_mix alone does NOT transition the dense
  clouds**; (4) boundary finding — injecting κ_eff into the Spitzer boundary IC (:370) **diverges** (`dR2 ∝ C`
  blows past R1), so **RHS-only** is the stable/correct choice, **refining SPEC §3**. Net: κ_mix is a real but
  saturating, density-mismatched correction; gated production is **on hold** pending a strategy revision
  (combine with the θ_target cap? re-metric? boundary re-derive? — `KMIX_SELFCONSISTENT.md` §3). No production
  code touched (monkeypatch-only, no sims). Reconciled INDEX §2/§3 track, the spec (§3 + λδv-pin), and
  `KMIX_PROTOTYPE.md` §3.
- **2026-06-30 (two PLANS written — gated κ_mix impl+units spec, and the Pb-collapse fix; no code changed).**
  Per the maintainer's two asks: (1) **`KMIX_IMPLEMENTATION_SPEC.md`** — the design for wiring κ_mix. Key
  decision that neutralizes the units bug class: implement κ_mix as a **dimensionless multiplier** on the
  existing Spitzer term, `κ_eff = κ_Spitzer·max(1, R)`, `R = (λδv)·Pb_cgs/(C_th·T^(7/2))` computed entirely in
  cgs — so the solver's mixed AU/cgs RHS is untouched and **off ⇒ multiplier is literally 1.0 ⇒ bit-identical**.
  Verified the 3 conduction sites (`bubble_luminosity.py` :291 seed=leave-Spitzer, :370 boundary + :406 RHS =
  need κ_eff because `_T_INIT_BOUNDARY=3e4 K` sits *inside* the κ_mix layer). Gate params mirror
  `cooling_boost_mode`: `kappa_mix_mode='none'` + `kappa_mix_lambda_dv=0.0` (double off-switch). Gates:
  per-call bit-identical-off → self-consistent offline (all 8) → gated full-run byte-identical-off + θ
  calibration to Lancaster. (2) **`PB_COLLAPSE_GUARD_FIX.md`** — re-traced the heavy-run negative Pb: the
  earlier "line-1074-vs-865 ordering" guess was **wrong**. The garbage `Pb=−1.6×10¹⁸` is emitted by the
  **phase-boundary reconciliation snapshot** (`run_energy_implicit_phase.py:1269–1297`) that runs after the
  collapse `break` and recomputes `Pb_f=compute_R1_Pb(R2, Eb<0, …)` (:1273) from the negative collapse Eb, then
  `save_snapshot()` (:1297). Fix = skip reconciliation when `termination_reason=='energy_collapsed'` (one line,
  byte-identical for all non-collapsing runs); test plan = failing unit test (no negative Pb survives, code 51
  still propagates) + 8-config byte-identical regression + fail_repro end-to-end. Both queued behind the
  guardrail — **no production code touched.** Reconciled `INDEX.md` §2/§3 and `KMIX_PROTOTYPE.md` §2.
- **2026-06-30 (ran the 4 cal anchors in-container → κ_mix prototype on the full density span; GO firm).** The
  earlier "HPC needed" assumption was wrong — full sims fit in <60 min (each ~12 min). Ran cal_compact/mid/diffuse/
  dense (f_κ=1) + heavy fail_repro via background agents, monitored with a 10-min health loop; all completed
  cleanly (cal: STOPPING_TIME at t=0.3 Myr; heavy: ENERGY_COLLAPSED). Harvested Pb(t) →
  `runs/data/harvest_cal_*.csv`; pointed `data/make_kmix_prototype.py` at the 4 clean density anchors (nCore
  1e2–1e6). Result CONFIRMED + strengthened: **κ_mix/κ_Spitzer = 10³–10⁸ in the cool layer (2e4–2e5 K) at λδv=1**,
  T_cross 2.4–5.0×10⁶ K (far above the layer), fairly **uniform across density** → κ_mix matters generically;
  λδv is the sensitive knob (even λδv≪1 dominates) → calibrate to Lancaster, never crank. Heavy 5e9 **excluded**
  (energy-collapse, no implicit phase → no mixing layer; itself a finding). Updated `KMIX_PROTOTYPE.md` §2–§3,
  `INDEX.md` track. **Validation:** compact max θ=0.676 == the known baseline 0.667. No production code touched.
  **Side diagnosis (heavy negative Pb):** investigated `fail_repro` Pb=−1.6×10¹⁸ — NOT a results bug; it is the
  collapse signature (`Pb=(γ−1)Eb/V`, Eb<0 at `ENERGY_COLLAPSED`, ÷ tiny V), only in the terminal row, healthy
  runs clean. *(Source re-traced 06-30 — see the newer ledger entry above and `PB_COLLAPSE_GUARD_FIX.md`: the
  bad row is the post-loop reconciliation snapshot at `run_energy_implicit_phase.py:1269–1297`, not the
  line-1074-vs-865 ordering this entry originally guessed.)* **Not fixed** (production change, guardrail; low
  priority).
- **2026-06-29 (κ_mix OFFLINE PROTOTYPE — step 1 of Rung-B, GO; + master `INDEX.md`).** Built the offline scoping
  harness (`data/make_kmix_prototype.py` → `data/kmix_prototype.csv` + `kmix_prototype.png`; reads committed
  `runs/data/harvest_*.csv`, **no solver touched, no sims**) — the de-risk step the guardrail requires before any
  wiring. **Units handled explicitly** (the bug class): `Pb` is AU `Msun/Myr²/pc` → cgs via `/1.5454e12`
  (`Pb_cgs2au`); λδv pc·km/s → cm²/s ×3.086e23; dimensional self-check printed. Result: at the front
  (`n=Pb/k_B T`), `κ_mix/κ_Spitzer = λδv·Pb/(C_th·T^{7/2})`, and in the cool layer (2e4–2e5 K) **κ_mix dominates
  Spitzer by 10³–10⁹ even at λδv=1** across compact/diffuse/dense (T_cross 2.7e5–1.2e7 K, above the layer) →
  **GO**: κ_mix would restructure the front, it is not negligible. Equally: even λδv≪1 dominates, so λδv is the
  **sensitive magnitude knob** — calibrate to Lancaster, never crank. Coverage: **4 of 8 configs** (heavy harvest
  is a stub; the other 4 need HPC Pb(t)); the 3 covered span the regime range so the GO holds. Next (still
  pre-production): self-consistent re-solve with κ_mix injected, all 8 configs, byte-identical-off. Also added
  **`INDEX.md`** (master map: reading order, the doc timeline/phase/purpose table, the κ_mix implementation track).
- **2026-06-29 (maintainer MANUSCRIPT DRAFT verified + folded; new doc `KMIX_DIFFUSIVITY.md`).** Line-by-line
  checked a 2-section LaTeX draft ("A functional form for the conduction multiplier" + "Where the mixing
  diffusivity comes from") + claims table against our committed results. **~90% matches** (f_mix=f_κ^q with
  q≈0.3–0.4 < the El-Badry 1/2; `f_κ(n)=[θ_target/θ_0]^(1/q)` ≈4 compact/≈60 diffuse (the ≈60 anchor is dead
  per §10, 2026-07-02 — blowout-metric artifact); θ/(1−θ)=1.6√n folding the
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
  879 / not Weinberg) — the external handoff's `3e68143` El-Badry-overlay work was cherry-picked into this line on 2026-07-01 (entry below).
- **2026-06-29 (El-Badry+2019 PDF obtained — equations VERIFIED, overlay figure built).** The El-Badry PDF
  (1902.09547) was finally provided (earlier it 403'd, so `θ(n)` was a schematic stand-in — see FINDINGS §2).
  Read §5.2 and **verified the cooling-efficiency model line-by-line**: Eq 35 `L_int/Ė_th = (11/5)·θ/(1−θ)`
  ⇔ Eq 38 **`θ = ψ/(11/5+ψ)`** (ψ≡L_int/Ė_th); Eq 37 **`ψ = A_mix·(λδv)^½·n_H^½`**, **A_mix=3.5** (fit;
  1.7 first-principles); √n scaling REAL; θ time-independent (L_int∝t). **This retracts the earlier in-session
  skepticism that those equations might be confabulated — they are genuine.** **The real caveat is DOMAIN:**
  El-Badry calibrated this at **n_H,0 = 0.1–10 cm⁻³ and λδv = 0.1–10 pc·km/s** (Figs 6–7); GMC clouds at
  n=1e2–1e6 are **1–5 decades beyond** where it was tested, where θ_target is already saturated to ≈0.94–0.999
  (extrapolation, not measurement). Built `data/make_elbadry_overlay.py` → `elbadry_overlay.png`: our resolved
  θ_1D = 1−cool_at_blowout per config (the f_κ=1 Spitzer floor) overlaid on the verified Eq 37/38 band — **our
  points sit far below the target (gap 0.25→0.94 at n=1e2), entirely in the extrapolated GMC regime.** That gap
  is the cooling deficit mixing/κ_mix (or, as a stand-in, f_κ) must supply. Honest axis caveat: El-Badry's n is
  ambient, ours is nCore; the literature θ really depends on the (higher) interface density. No production code touched.
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
  at f_κ=4) / ~5-6 (mid, extrap.) / ~60 (diffuse, extrap.) → dead per §10 (blowout-metric artifact,
  2026-07-02) — answering the
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
  0.17–0.67 — i.e. **the diffuse cloud that needs f_κ~60 on the radiative-only trigger** (→ dead per §10,
  blowout-metric artifact, 2026-07-02) **is already ~0.65–0.85 on
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
  (needs `f_κ≈60`, extrapolated, at the viability edge → dead per §10, blowout-metric artifact, 2026-07-02). The **snapshot estimate was optimistic** — the
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

**Last updated:** 2026-07-02 (theta5 matrix ran — OPEN(1) closed, §10; live status in the re-entry ledger
above; pt2 merged to main via PR #717, pt3 merged into this line, docs reconciled — see `CONTAMINATION.md` +
`INDEX.md §5`). **Branch history:**
`feature/PdV-trigger-term` → `feature/PdV-trigger-term-pt2` (merged) ⇄ `feature/transition-trigger-pt3`
(merged). Original branch note:
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
  calibration (3 configs)**: θ(f_κ=1)=0.67/0.61/0.17, f_κ-to-fire ≈4/~5-6/~60 (compact measured/mid & diffuse
  extrap.) (⛔ CONTAMINATION #3; superseded by §10);
  **theta5 (the 📏 rule-compliant multiplier calibration, ran 2026-07-02 — §10):**
  `runs/data/theta5_summary.csv` (32-arm θ_max harvest) + `runs/data/theta5_calibration.csv` (per-config
  θ₀/p/f_fire + the θ₁-collapse fit), built by `runs/harvest_theta_max.py` (sanctioned θ_max harvester,
  reads `dictionary.jsonl`) → `runs/make_theta5_calibration.py` (turnkey reduction + scorecard);
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
