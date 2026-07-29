# FA_STATE_COUPLED — the state-coupled f_A: derive the density dependence instead of fitting it (single source of truth for the successor workstream)

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
> a committed artifact under `docs/dev/` (a CSV/table in `docs/dev/transition/pdv-trigger/data/`, or a
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

**Status (2026-07-22, created):** THE single plan doc for the **state-coupled f_A** workstream —
the successor the scalar-f_A stream's own Phase-6 tree names (row 3, `FINDINGS.md §15j`). Per the
maintainer's one-stream directive there are **no parallel plans**: the scalar-f_A history, evidence
and Phase 0–6 record stay in `SOURCE_TERM_DESIGN.md` (its §4 sketch is hereby **promoted here** and
must not be extended there); THIS doc plans only the successor. **Phase SC-0 COMPLETE 2026-07-25 → VERDICT: FAIL** (14 arms, `FINDINGS §15k`, `fa_state_screen.png`).
**TERMINAL 2026-07-25** — the FAIL is the pre-registered stop: SC-1…SC-5 are not to be started, no
production code is written, ruling clauses 2b/4 are resolved/moot below, and the f_mix retirement
ladder halts permanently at R0 (done). The plan stays open only as the entry point for a *new*
candidate law, which would restart at SC-0. Nothing here is awaiting a maintainer decision.
No derived candidate reproduces the measured doses: C1 spread 30× and nearly flat, C2 falsified by
2–8 dex, C3 (fitted baseline) 56×. **Per this plan's own pre-registered rule, SC-1 onward DO NOT
PROCEED and no production code is written.** The remainder of this doc is retained as the design of
record + the negative result's provenance. SC-1 onward are gated on the parent's Phase-6
maintainer ruling (§3 below). Nothing here touches production code until SC-1. **Phase-6 ruling
STARTED 2026-07-22: clause 1 RULED — f_mix RETAINED as an opt-in fallback, retirement deferred +
staged (R0→R2 ladder, §3); default stays `none` (clause 3). Clauses 2/4 (adopt scalar f_A as the
diagnostic knob; greenlight successor) await an explicit nod — but SC-0 may run regardless.**
**Clause 2 NARROWED + candidate set WIDENED 2026-07-22** (maintainer: "not sure the scalar should be
the calibration knob; maybe use the Lancaster values"): SC-0 now screens **three** candidates (§1) —
El-Badry L_int (C1), **Lancaster fractal area Eq 11 (C2)**, and the fitted scalar as baseline (C3) —
and the data picks. A back-of-envelope pre-screen (§1) finds C2 does NOT collapse the measured doses
at Lancaster's own d≈0.4–0.7 (3.4–16.8× spread vs C3's 5.4×), so nothing is pre-picked. Two
literature values (α_A, and what ℓ physically is) were needed for C2's fair test — **BOTH ANSWERED
2026-07-22 from maintainer-supplied L21b pp. 3–6 (`LANCASTER_REFERENCE §7c`): α_A is an order-unity
fudge factor (≈1) and ℓ is the physical cooling scale ℓ_cool (v_t(ℓ)t_cool = ℓ), NOT a grid scale — so
C2 is now CLOSED and PARAMETER-FREE (α_A≈1, d∈[0.4,0.7] both [V]), a stronger test than C1. SC-0 can
falsify it outright. Caveats: p=1/2 is our derivation, and the doses' demanded ℓ sits at/below L21b's
own Δx on the diffuse benches (§5).**

**Status addendum (2026-07-27, external review — `FINDINGS.md §17`):** the SC-0 FAIL and this
doc's TERMINAL state are **unaffected** (SC-0's targets are f_A-side doses, which the review
verified). But the parent premise "**f_mix was eliminated outright**" (§0 below, and `§15j`)
rests on a **metric artifact** — the bench6 fm Θ_cum omitted the f_mix boost; corrected, fm band
entry is ≈ 4⁺/8⁺/>8 and monotone (`§17`). The clause-1 outcome (f_mix RETAINED, default `none`)
is if anything reinforced, but its evidentiary record and the R0→R2 retirement premise must be
corrected and re-presented — see `SOURCE_TERM_DESIGN.md §3 "Phase 6 correction (X1–X4)"`.
**UPDATE 2026-07-28: X1–X4 are EXECUTED (`FINDINGS §18`/`§19`/`§20`).** Corrected band entry
bench3 ≈4 (measured) / bench2 >8 / bench1 >8 (extrapolated ≈8.2/11.9); on the tree's uniformity
metric f_mix ≈2.96× vs f_A 5.39×, i.e. the head-to-head **inverts** — as an estimate the fm≤8
grid cannot settle. SC-0's FAIL and this doc's TERMINAL state remain unaffected. The clause-1
re-presentation is in `SOURCE_TERM_DESIGN.md §3` under "Maintainer re-presentation".

## 0. Why this workstream exists (one paragraph)

The scalar f_A was measured to work — every clean L21b bench reaches the Θ band — but at a steeply
density-dependent dose: band entry **13.9 / 53.5 / 74.8** for n̄ = 5520/690/43 (spread 5.39×, fit
f_A(n̄) ≈ 315·n̄^−0.335; `FINDINGS §15j`), while f_mix was eliminated outright *(⛔ WITHDRAWN — that elimination was a metric artifact;
corrected `FINDINGS §18` 2026-07-28, where f_mix's band-entry spread ≈2.96× actually beats f_A's
5.39×. The hypothesis below stands on the f_A dose gradient alone, which is unaffected)*. A fitted f(n̄) is
exactly the kind of un-derived magic function this workstream's history warns against
(`INDEX.md §1.5`). Hypothesis to test: **the density dependence is not a free function but El-Badry
mixing-layer physics** — replace the scalar with f_A evaluated from the live bubble state via the
L_int closed form, leaving **one physical constant (λδv)** to serve the whole suite.

## 1. The object — THREE candidates SC-0 screens (do not pre-pick; 2026-07-22)

f_A is by definition an **area ratio** (A_eff/4πR₂²), so more than one literature law can supply it.
SC-0 screens all three against the same measured target (band-entry 13.9/53.5/74.8) and the **data
picks** — this replaces the earlier single-candidate framing (maintainer question 2026-07-22: "should
the scalar be the calibration, or should we use Lancaster's values?").

| # | candidate | form | free parameters | provenance |
|---|---|---|---|---|
| **C1** | **El-Badry mixing luminosity** | f_A = L_int^EB(R2,Pb;λδv) / (L2+L3)_prev | **λδv** (lit. ≈3–3.5) | `ELBADRY_REFERENCE §7` [V] |
| **C2** | **Lancaster fractal area** (Eq 11 — *literally* an area multiplier) — **CLOSED 2026-07-22, see §5** | f_A = α_A·(R₂/ℓ_cool)^d, ℓ_cool = [v_t(L)·t_cool]²/L | **α_A≈1 (order-unity, [V]) + d∈[0.4,0.7] ([V]) only — ℓ is NOT free** | `LANCASTER_REFERENCE §7c` Eq 11–13/22–23 [V] |
| **C3** | fitted scalar (baseline to beat) | f_A(n̄) ≈ 315·n̄^−0.335 | 2 fitted | `FINDINGS §15j` — measured, un-derived |

**C2 pre-screen (2026-07-22, back-of-envelope — NOT the SC-0 test).** Inverting Eq 11 at the blowout
radius (R_b ≈ rCloud = 5/10/20 pc, the verified diag end-states) for the measured doses, with α_A=1,
asks whether ONE inner scale ℓ serves all three benches: `ℓ = R_b / f_A^(1/d)`.

| d | ℓ(bench3) | ℓ(bench2) | ℓ(bench1) | spread |
|---|---|---|---|---|
| 0.4 | 0.0069 | 0.0005 | 0.0004 pc | 16.8× |
| **0.7** (top of lit. range) | 0.116 | 0.034 | 0.042 pc | **3.4×** |
| 1.0 | 0.360 | 0.187 | 0.267 pc | 1.9× |
| 1.5 (⚠️ far above lit.) | 0.865 | 0.704 | 1.127 pc | 1.6× |

**Read:** at Lancaster's *own* measured d≈0.4–0.7 the fractal law does **not** collapse the doses to a
single scale (3.4–16.8× spread, vs C3's 5.4×) — only d≈1.5, well outside the measured range, tightens
it. So **C2 is not a free win**; it is a real candidate that must be tested properly (along the
trajectory, not at one radius) before anyone prefers it. ⚠️ Caveats: R_b-at-blowout is a crude proxy
for the trajectory; the α_A=1 assumption is now **vindicated** ([V]: α_A is an order-unity fudge factor)
and ℓ is now **known to be ℓ_cool** (§5) — so this pre-screen's "spread of ℓ" must be re-read as "spread
of the ℓ the doses *demand*", to be compared against the ℓ_cool the physics *predicts* (SC-0's real job).
**New resolution finding (2026-07-22):** those demanded ℓ (0.116/0.034/0.042 pc at d=0.7) sit at
ℓ/Δx ≈ 5.8 / 0.85 / 0.28 versus L21b's own grid (Δx = 0.02/0.04/0.15 pc, Table 1 [V]) — i.e. **at or
below their resolution for the two diffuse benches**, the very regime we calibrate. Any C2 agreement
there is resolution-limited on the L21b side (`LANCASTER_REFERENCE §7c`).

Candidate C1's definition (the one-read swap at the two production edit sites,
`bubble_luminosity.py:435/845`):

```
f_A_state(t) = L_int^EB(R2, Pb; λδv) / (L2+L3)_resolved^(prev accepted step)
L_int^EB     = 4π·√(α·λδv) · R2² · Pb^(3/2) · √Λ(T_pk) / (k_B·T_pk),   T_pk ≈ 2×10⁴ K
```

(`ELBADRY_REFERENCE.md §7` option B — the direct form; §9 verified TRINITY's plumbing realizes the
El-Badry budget faithfully.) The knob's free physical constant is **λδv only** (literature anchor
λδv ≈ 3–3.5, `LANCASTER_REFERENCE.md §6`); the goal is a single λδv across the suite where the
scalar needed 14→75.

**Decision points SC-0 freezes (recommendation first, decide before SC-1):**

| # | decision | recommendation | why |
|---|---|---|---|
| D1 | closure lag | **previous accepted step** (lagged) | precedent: the §4-sketch q_w closure; avoids a new inner iteration in the hot loop |
| D2 | option A (n-mapped θ(λδv, n_amb)) vs **option B (direct L_int)** | **B** | no n-mapping; saturation emerges via Pb; robust at the early-core/late-blowout extremes where A diverges (`ELBADRY_REFERENCE §7`) |
| D3 | caps/floors | clamp f_A_state ∈ [1, f_cap≈256]; unchanged T<10^5.5 band gating; if (L2+L3)_prev → 0 (dense collapse) hold last finite value | keeps the dense-collapse stiffness (bench5_fa16_diag freeze, `§15h`) from amplifying; 256 = 2× the largest measured need |
| D4 | knob surface | sentinel string **`cooling_boost_fA='elbadry'`** resolved like `cooling_boost_kappa='auto'` | no new param; single-knob validator extends naturally; default '1.0' stays byte-identical |

## 2. Phase ladder (solver hot loop ⇒ full rule-5 ladder; gates BEFORE code)

**SC-0 — offline screen (read-only; MAY run pre-ruling). THE falsification gate.**
1. *Data prerequisite:* the committed traj CSVs lack Pb (`t_now,theta,Lcool,Lleak,Lmech,R2`), but
   every `dictionary.jsonl` logs `Pb` (reader vocabulary, `trinity_reader.py:165`). Either
   (a) maintainer re-harvests the bench1/2/3 fa1(+fa16) diag arms on Helix with a Pb+L2+L3 column
   set (extend `harvest_bench5.py` with `--extra-cols`), or (b) run the 3 diffuse fa1 diag arms
   locally (~20–45 min each, walltime evidence `data/bench5_durations.csv`) and harvest there.
   Commit as `runs/data/bench_state_traj/`.
2. *Offline calculator* `data/make_fa_state_screen.py`: along those trajectories evaluate **both
   derived candidates** (§1) and score them against the same target:
   - **C1** f_A = L_int^EB(R2,Pb;λδv)/(L2+L3)_prev for λδv ∈ {1, 2, 3, 3.5, 5};
   - **C2** f_A = α_A·(R2/ℓ)^d for d ∈ {0.4, 0.5, 0.6, 0.7} × the ℓ candidates of §5 (and, if the
     maintainer supplies it, the published α_A) — the trajectory version of the §1 pre-screen;
   - **C3** the fitted scalar, as the baseline both must beat.
   Score = blowout-window average f_A vs the **measured** band-entry doses 13.9/53.5/74.8, reported
   as the max/min spread of the implied free constant (λδv for C1, ℓ for C2). **PASS:** a candidate
   holds ONE constant across all three benches within a factor ~2 (the tolerance the p=3.33 law
   achieved) — that candidate is the physics and goes to SC-1. **FAIL (both):** neither closed form
   derives the curve ⇒ stop, record the spreads, and the fitted f_A(n̄) remains the honest shipped
   result — **no production code gets written.** Winner-takes-SC-1; if both pass, prefer the one with
   the smaller spread and the fewer un-imprinted constants.
3. Persist: `data/fa_state_screen.csv` + figure; register in REPRODUCE.

**SC-0 implementation notes (2026-07-22 — reuse, do not re-derive; units are this repo's declared
bug class).**
- **C1 needs NO new unit work.** `data/make_elbadry_theta.py` already carries the El-Badry closed form
  in *dimensionless* form and is validated against his Fig 7 (`theta(λδv=1, n=1, A_mix=3.5) = 0.61`):
  `X = A_mix·√(λδv·n)`, `θ_EB = X/(11/5 + X)` with λδv in pc·km/s and n in cm⁻³. Since option A's n is
  the *local ambient* density and every bench is `densPL_alpha=0` (uniform), **n = nCore = n̄_H exactly**
  (43.1 / 690 / 5520). Then, matching boosted θ to θ_EB:
  ```
  f_A^C1(t) = θ_EB(λδv, n̄) · L_mech(t) / (L2+L3)(t)          [all three factors already in the traj CSV]
  ```
  ⇒ import `theta()` from the validated builder; no L_int/k_B/Λ conversion is written by hand.
- **C2 DOES need one conversion** (P_b au→cgs and Λ(T_pk) at T_pk≈2×10⁴ K) for
  `t_cool = (k_B T_pk)²/(P_b Λ(T_pk))`. Use `trinity/_functions/unit_conversions.py` (`cvt`) and the
  bundled cooling table for Λ — **never hand-rolled constants** — and unit-test the single conversion
  against `t_cool = P/(n²Λ)` (Eq 13's other form, with n = P/(k_B T_pk)) as a cross-check before use.
- **Data (DONE 2026-07-22):** `runs/data/bench_state_traj/` — the three diffuse fa1 diag arms re-run
  locally (Helix unavailable) and harvested with the new `harvest_bench5.py --extra-cols`, so the CSVs
  carry `Pb, bubble_L2Conduction, bubble_L3Intermediate, bubble_dMdt, bubble_LTotal` alongside the
  standard six. Provenance: identical committed params (`runs/params/bench5/bench{1,2,3}_*__none_diag`),
  in-container (fidelity vs HPC was measured OK for bench5, `§15j`).

**SC-1 — wiring (gated on Phase-6 ruling + SC-0 PASS).** The one-read swap at the two edit sites +
registry sentinel resolver + validator extension + `test_fA_state_coupled.py`. Default `'1.0'`
stays the LITERAL float path (byte-identity preserved by construction, same guard style as today).

**SC-2 — gates (rerun the parent's Phase-3 pattern).** (i) default LITERAL byte-identity
(`dictionary.jsonl` sha256, pre==post); (ii) per-call equivalence: live f_A_state values vs the
SC-0 offline calculator on captured states; (iii) live sign checks (dMdt falls, θ rises, no
freeze/no-root regressions on the stiff edges `f1edge_*`).

**SC-3 — matrix (Helix).** The 9 theta5s configs × λδv {2, 3, 3.5, 5} = **36 arms** (no dose grid —
that is the point). Fire map + controls (`fail_repro`, `small_1e6` must stay cold).

**SC-4 — THE acceptance gate (Helix).** The 5-bench L21b suite × the same λδv grid, prod+diag =
**40 arms**, same blowout-window Θ_cum metric and harness as bench5/6. **PASS:** a single λδv
(target ≈3±1) lands Θ_cum ∈ [0.90, 0.99] on bench3/2/1 simultaneously, dense benches still fire,
controls cold, dex-vs-EB improves on the scalar's ≥0.85. **FAIL:** record how close (dex per
bench), keep the knob diagnostic-only.

**SC-5 — ship decision.** SC-4 PASS ⇒ the default-flip ruling package (one derived knob, one
physical constant, L21b-validated — the paper's f_A story completes). SC-4 FAIL ⇒ documented
negative + the scalar f_A(n̄) table stands as the calibration.

## 3. Phase 6 of the PARENT — the RULING (source of truth; SC-1+ waits on this)

This is THE Phase-6 ruling of record (this doc is the single place it lives; `SOURCE_TERM_DESIGN.md
§3 Phase 6` and any `FINDINGS.md §15k` point HERE, they do not restate it). Clauses tagged
**[RULED 2026-07-22]** are the maintainer's decision; **[pending]** clauses still want an explicit nod.

1. **f_mix — RETAINED as an opt-in fallback [RULED 2026-07-22, maintainer].** ~~bench6 eliminated
   f_mix as a *calibration* knob (never reaches the L21b band ≤8, wrong-sign dose-response on the
   diffuse benches, fm8 false-fires — `FINDINGS §15j`)~~ **[the stated GROUNDS are WITHDRAWN
   2026-07-28 — all three legs were metric artifacts or backwards; `FINDINGS §18`/`§19`. The
   RULING (retain as opt-in fallback) is unchanged and if anything strengthened; only its
   justification changes, and the R0→R2 retirement premise must be re-derived — see the
   maintainer re-presentation in `SOURCE_TERM_DESIGN.md §3`.]** f_mix was never eliminated as a
   *fallback*. It stays fully wired and
   supported for now: it is a valid opt-in mechanism AND the control arm the bench harness relies on.
   Nothing is removed while f_A is not yet the production path. **Retirement is deferred and STAGED
   (the "safely and slowly" ladder), each rung gated on the one before:**
   - **R0 (now) — DONE 2026-07-25** (registry `info` lines added to `cooling_boost_mode` +
     `cooling_boost_fmix`; help-text only, `exclude_from_snapshot=True`, no behavior change).
     **And R0 is now the TERMINAL rung**: SC-0 FAILED (§15k), so by the abort rule below the
     state-coupled f_A does not ship, R1/R2 never unlock, and **f_mix stays indefinitely.**
     `cooling_boost_mode='multiplier'` retained, opt-in, **inert by default** (default
     is `none`). Registry `info` for `cooling_boost_mode`/`cooling_boost_fmix` gains one line:
     "fallback — superseded for L21b *calibration* by f_A (`FA_STATE_COUPLED.md`); retained pending
     the state-coupled f_A shipping." No behavior change. (This is the only code touch clause 1
     authorizes now — a doc-string edit, byte-neutral.)
   - **R1:** only AFTER the state-coupled f_A ships as the production default (SC-5 PASS) AND ≥1
     release cycle of it running clean — mark `multiplier` **deprecated** in the registry (still
     works, emits a load-time deprecation warning). No removal yet.
   - **R2:** after ≥1 further cycle with nothing in-repo relying on it (grep the params/tests/docs),
     remove the `multiplier` branch — per the project rule, `git mv` the code + its arms into
     `docs/dev/to-be-removed/` for maintainer review, never a direct delete.
   - **Abort rule:** if the state-coupled f_A does NOT ship (SC-4/SC-5 FAIL), the ladder STOPS at R0
     — f_mix stays indefinitely and the retirement is void.
2. **Scalar f_A — NARROWED 2026-07-22 (maintainer challenged the wording; the challenge was right).**
   The earlier phrasing ("adopt the scalar as the calibration knob") over-claimed: it read as a
   *modelling commitment* to a fitted f(n̄), which is exactly the un-derived magic-function pattern
   `INDEX §1.5` warns against. Split into two, and only the first is asked for now:
   - **2a [recommended, low-stakes]:** the f_A(n̄) numbers are a **measurement of record** — quotable
     with HPC provenance as "the dose TRINITY needs per bench", i.e. the calibration *target* the
     derived candidates must reproduce (that is how SC-0 uses them). This commits to no model.
   - **2b [RESOLVED 2026-07-25 by SC-0's own pre-registration — NO f_A form ships]:** the answer was
     deferred to SC-0's output, and SC-0 returned FAIL on all three candidates (§15k): C1 misses by
     3.3× (band) / 4.5× (fire), C2 by 2–8 dex, C3 by 56× on `fire` — none holds one constant to the
     pre-agreed factor ~2. Per the SC-0 FAIL clause, *"neither closed form derives the curve ⇒ stop
     … no production code gets written."* So **the fitted f_A(n̄) stays a measurement of record
     (clause 2a), never a shipped model**, and TRINITY's production cooling is unchanged. Reopening
     2b requires a NEW candidate law, which restarts at SC-0 — not at SC-1.
3. **Production default — UNCHANGED [RULED 2026-07-22, implied by clause 1].** `cooling_boost_mode=none`,
   `cooling_boost_fA=1.0`, byte-identical. Keeping f_mix as an opt-in fallback presupposes no default
   flip now — so this is settled by clause 1.
4. **Successor — MOOT 2026-07-25 (nothing left to greenlight).** The clause gated SC-1+ on a
   greenlight plus an SC-0 PASS. SC-0 FAILED, so the ladder terminates at SC-0 by construction:
   SC-1 (wiring) through SC-5 (ship decision) are **not to be started**, and SC-4 is no longer a
   live bar because there is no candidate to put in front of it. No explicit nod is needed to
   *stop* — the stop was pre-registered. A future nod would only be needed to *restart*, and only
   with a new candidate law entering at SC-0.

Parent loose ends that stay in the PARENT's ledger (not this doc): the dMdt reducer re-run on the
Helix theta5s raw arms (`§15e` residue); Fig-17 re-digitization before quantitative fits; V_w
[I]-grade; `rosette-cf/figs/README.md` banner (other workstream).

## 4. Artifacts & reconciliation

Artifacts this plan will create: `runs/data/bench_state_traj/` (SC-0 data),
`data/{make_fa_state_screen.py, fa_state_screen.csv}` (SC-0), the SC-1 diff + tests, SC-3/SC-4
params + summaries under `runs/params/{sc_matrix,sc_bench}/` + `runs/data/`, REPRODUCE rows on
landing. Siblings to keep reconciled on every edit: `SOURCE_TERM_DESIGN.md` (§4 pointer + Phase-6
ruling), `FINDINGS.md` (new §15k+ entries), `INDEX.md` (this workstream's row), `PLAN.md` ledger,
`ELBADRY_REFERENCE.md`/`LANCASTER_REFERENCE.md` (imprints — read-only anchors here).

## 5. Literature asks — **BOTH ANSWERED 2026-07-22** (maintainer supplied L21b pp. 3–6)

Imprinted in `LANCASTER_REFERENCE.md §7c` (read that, not this summary, for the [V] detail).

1. **α_A — ANSWERED: an "order-unity parameter meant to account for any minor inconsistencies with
   this model"** (Eq 11 text). It is a fudge factor, not a measured constant ⇒ **α_A ≈ 1 is the right
   default** (the pre-screen's assumption was correct), and α_A must NOT be tuned to rescue a fit.
2. **ℓ — ANSWERED, then MEASURED (2026-07-25) — and the measurement reverses the optimistic reading.**
   ⚠️ **SC-0 result: ℓ_cool is unreachably small** — t_cool ≈ 0.03 yr in peak-cooling gas gives
   ℓ_cool ≈ 8×10⁻¹⁵ pc (p=1/2), and **for every p<1 it lies below every physical/numerical scale**
   (2.9×10⁻⁷ pc even at p=0, vs the ~5×10⁻⁷ pc conduction front and L21b's 0.02–0.15 pc grid). The
   cascade therefore never reaches ℓ_cool, so Eq 11's operative ℓ is set by the *truncation* scale
   (resolution in their sims), not by cooling. **C2 as literally specified predicts f_A ~ 10⁹–10²⁴ —
   falsified by 8+ orders of magnitude.** Unit cross-check passed exactly, so this is physics, not a
   conversion bug. C2 survives only if someone supplies an independent physical truncation scale.
   *(Superseded text, kept for the record: "ℓ is NOT free and NOT a grid scale.")* It is the
   **cooling scale ℓ_cool**, fixed by a cascade-vs-cooling balance (Eq 12–13 + text): the enthalpy flux
   grows toward smaller scales "until reaching the scale where **v_t(ℓ_cool)·t_cool = ℓ_cool**", with
   `t_cool = (k_B T_pk)²/(P Λ(T_pk))` (∝ R_b²). **⇒ C2 does NOT die on transferability** — the law is
   physical, not resolution-set, so it CAN be ported to a 1-D code.

**C2 is therefore CLOSED and (nearly) parameter-free.** With their initial spectrum `|v_k|² ∝ k^−4` and
`L_box = 2R_cloud` [V], and **p = 1/2 [D-grade: derived by us from that spectrum, NOT quoted — check
Paper I before any quantitative fit]**:

```
f_A^L21b = α_A · (R2 / ℓ_cool)^d ,   ℓ_cool = [ v_t(L) · t_cool ]² / L
   α_A ≈ 1 [V]      d ∈ [0.4, 0.7] [V]      L = L_box = 2 R_cloud [V]
   v_t(L) = Table-1 v_t = the α_vir=2 virial velocity [V-stated, Eq 23]
   t_cool = (k_B T_pk)² / (P_b Λ(T_pk)) [V Eq 13],  T_pk ≈ 2×10⁴ K (= El-Badry's T_pk)
```
Everything on the right is computable from TRINITY state — **no fitted parameter remains.** That makes
C2 a genuine *prediction* (a stronger test than C1, whose λδv is still a tuned constant): SC-0 can now
falsify or confirm it outright, with **zero freedom to fudge**.

⚠️ **Two live caveats carried into SC-0 (do not drop):**
- **p = 1/2 is ours, not Lancaster's** — if Paper I gives a different cascade index the ℓ_cool formula
  changes as `ℓ_cool ∝ [v_t t_cool]^{1/(1−p)}`. Ask for Paper I's §2 if C2 survives the first screen.
- **Resolution:** the ℓ our measured doses *demand* (0.116/0.034/0.042 pc at d=0.7) sits at
  ℓ/Δx ≈ 5.8/0.85/0.28 against L21b's own grid — **at or below their resolution on the two diffuse
  benches**. If SC-0 finds ℓ_cool there too, the agreement is resolution-limited on *their* side and
  must be reported as such (`LANCASTER_REFERENCE §7c`; Gentry & Krumholz 2019 caution).

**Side-effect corrections already applied to the imprint:** μ_H = **1.4271** (not 1.4 — 0.13% in radius,
inside the bench 2% gate ⇒ no param changes, but quote the exact value); Table 1 re-verified 12/12 with
two NEW columns (Δx, Resolution); v_t = virial velocity upgraded to [V]-stated; Eq 10 re-confirmed.
