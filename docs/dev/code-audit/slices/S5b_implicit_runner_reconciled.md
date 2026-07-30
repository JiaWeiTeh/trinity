# S5b implicit runner — reconciled

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

**Slice:** `trinity/phase1b_energy_implicit/run_energy_implicit_phase.py` (Phase 1b, the
implicit-phase runner).
**Inputs:** the three raw lens reports only (`S5b_implicit_runner_lens{A,B,C}.md`). No source was
read by this reconciler.
**Method:** align claim-by-claim on `file:line` + function; classify each disagreement by which pair
diverges; merge and dedupe; drop or demote lens findings that another lens plainly explains away.

Raw input: 36 (A) + 23 (B) + 45 (C) = **104 candidate findings** → **40 reconciled entries**
(1 × S1, 12 × S2, 17 × S3, 10 × S4), plus **17 explicitly dropped or demoted** (§4).

**Two structural facts that frame everything below.**

1. **The equation of motion is not in this slice.** Lens A establishes that `get_ODE_implicit_pure`
   delegates `dR2/dt` and `dv2/dt` to `get_ODE_Edot_pure(...)` (outside the slice) and substitutes
   only the third and fourth components. `compute_forces_pure` in this file therefore builds a force
   *inventory for output*, and whether those forces are the ones the integrator uses is unresolvable
   here. Most of Lens C's momentum/energy-equation expectations (C-03…C-08, C-15) are consequently
   **cross-slice checks, not findings against this file** — recorded in §5, not in the merged list.
2. **This is an operator-split runner.** All three lenses agree independently that `dEb/dt` and
   `dT0/dt` are frozen per segment (A §3, B F14, C §1), so `Eb` and `T0` are *linear in t* within a
   segment. Every finding about tolerances, event resolution and segment length has to be read
   against that: `dt_segment` is the **splitting cadence**, not an accuracy control, and LSODA's
   tolerances govern only two of the four states.

---

## 1. Coverage table — what each lens did and did not address

Silence is **not** corroboration. `—` means the lens never spoke to the item.

| # | Quantity / behaviour | Lens A (does) | Lens B (claims) | Lens C (should) | Status |
|---|---|---|---|---|---|
| 1 | ODE state order `y=[R2,v2,Eb,T0]` | ✔ | ✔ F11 | ✔ | corroborated (agree) |
| 2 | `dEb/dt`, `dT0/dt` frozen per segment | ✔ | ✔ F14 | ✔ (defining move) | corroborated (agree) |
| 3 | `out[2] == Ed_from_beta` (no second PdV subtraction) | ✔ index 2 replaced verbatim | ✔ | required (C-09) | corroborated **correct** |
| 4 | `dR2/dt == y[1]` literally | partial (fed live `y`; identity inside out-of-slice callee) | — | required (C-02) | **pending cross-slice** |
| 5 | `F_grav = G·M_sh(M_cl + M_sh/2)/R2²` | ✔ `:491`, literal `0.5` | ✔ F1 `:488` | ✔ C-06 | corroborated **correct** |
| 6 | `FOUR_PI == 4·π` | ✔ `4.0*np.pi` | — | required C-01 | corroborated **correct** |
| 7 | `P_drive = max(Pb, P_HII)` | ✔ `:529` | ✔ (implied) | ✔ | corroborated (agree) |
| 8 | `P_ram` in the energy phase | ✔ hard-coded `0.0` | contradictory (`F_ram` vs ":559 no ram") | must be absent (C-05) | contested → R-31 |
| 9 | `blowout ⟺ R2 > k·rCloud`; `ebpeak ⟺ Ėb_bal ≤ 0` | ✔ | ✔ F3/F4 | ✔ C-20/C-21 | corroborated **correct** |
| 10 | `r1_transition_decision` = OR, gated on membership, blowout precedence | ✔ | ✔ | ✔ C-23 | corroborated **correct** |
| 11 | `parse_transition_triggers` rejects unknown tokens | ✔ | ✔ | ✔ C-22 | corroborated **correct** |
| 12 | …but accepts an **empty** trigger set | ✔ A-31 | — | must reject (C-22) | corroborated → R-21 |
| 13 | `update_unconverged_streak` resets on converged | ✔ | ✔ N8 | ✔ C-27 | corroborated **correct** |
| 14 | `betadelta_phase_summary` guards `n==0` | ✔ | ✔ N10 | ✔ C-28 | corroborated **correct** (`clean` vacuously True at n=0 — noted only) |
| 15 | `classify_energy_collapse` 3-way partition | ✔ | ✔ (same partition) | wants **4-way** (C-16) | A=B ≠ C → R-06/R-07 |
| 16 | Cooling-balance criterion & direction | ✔ `(Lgain−Lloss)/Lgain < θ`; log says the inverse | **undocumented** (B-09) | `Ėb/Lgain ≤ θ` (C) | **A ≠ C, B silent** → **R-02** |
| 17 | No-root streak: routes or logs only? | ✔ **routes** (`break` at streak ≥ 50) | three contradictory comments | must be discriminated (C-32) | **A ≠ B** → **R-03** |
| 18 | Event terminality / direction | cannot tell; breaks on any `triggered` | **never documented** (B-16) | terminal + explicit direction (C-34) | ABC gap → **R-04** |
| 19 | Event path vs `classify_energy_collapse` | ✔ bypassed (A-08) | — | `Eb≤0` must be terminal (C-16) | corroborated → R-04 |
| 20 | `Pb` vintage: leak / betadelta / `Ed` | ✔ A-01, A-02 | leak freeze **documented** (A3); betadelta/`Ed` mismatch **not** | closure identity gate (C-12) | leak: A=B (demoted); `Ed`: single-lens → **R-05** |
| 21 | Scalar `atol` across the state vector | ✔ `1e-8` scalar, 4 states | value undocumented (N4) | must be per-component (C-36) | corroborated → R-15 |
| 22 | `max_step = 2e-5` vs `dt_segment ≤ 5e-2` | ✔ ≥2500 steps/segment | "ensures ≥5 steps" | — | corroborated → R-25 |
| 23 | `DT_SEGMENT_COLLAPSE (5e-5) < DT_SEGMENT_MIN (1e-4)` | ✔ measured | **predicted** from prose (FLAG-15) | ordering required (C-29) | corroborated → **R-14** |
| 24 | Adaptive controller effectively inert | ✔ 30/35 keys constant by construction | — | aliasing route (C-26) | corroborated, **different mechanism** → R-16 |
| 25 | Sign-flip handling in `compute_max_dex_change` | ✔ scores `1.0` | comments contradict | should force shrink | A ≠ B (stale comment) → R-17 |
| 26 | Zero-value skip in `compute_max_dex_change` | ✔ contributes nothing | — | must not contribute `0.0` (C-24) | corroborated → R-17 |
| 27 | `isCollapse` latch never cleared | ✔ A-14 | freeze documented, clearing not | mass must not *decrease* (C-42) | single-lens → R-10 |
| 28 | Shell-mass ratchet duplicated at two sites | ✔ (2nd omits `massDot`) | ✔ FLAG-13 | — | corroborated → R-23/R-34 |
| 29 | `P_ext` bare `except → 0.0` | ✔ A-11 | — | — | single-lens → R-11 |
| 30 | `PISM·k_B` units | needs `n·T` in AU (`pc⁻³ K`) | unit unstated (U5/U6) | `.param` declares `K cm⁻³` (C §2.2) | **contested** → R-29 |
| 31 | Reconciliation `try/except` swallows snapshot loss | ✔ A-17 | rationale documented (A15) | — | single-lens → R-12 |
| 32 | "All values at the same `t_now`" invariant | ✔ ≥6 concrete violations | ✔ FLAG-10 lists 5 | — | corroborated → R-23 |
| 33 | Transition-time resolution (±`dt_segment`) | ✔ post-ODE inline check, no refinement | — | must refine or bound (C-35) | corroborated → R-08 |
| 34 | `dt_segment` cap absolute vs `t_dyn` | ✔ cap `5e-2` Myr | ✔ N6 | must be relative (C-30) | corroborated → R-08 |
| 35 | `ENERGY_HANDOFF_FLOOR = 1e3` | ✔ = 1.901e46 erg; a **clamp**, not a trigger | value + cross-module coupling; unit unstated | should be relative (C-17) | ABC → R-27, C's trigger scenario dropped |
| 36 | Clock origin of `α, β, δ` | `cool_alpha = t_now·v2/R2` (form matches) | "preserves ODE continuity", no relation | must be **global cluster age** (C-11) | **open** → R-30 |
| 37 | `cool_alpha` refreshed after entry? | ✔ every segment `:798` | asks the question (B-19) | — | **resolved — dropped** |
| 38 | `_inflow_frac_thickness` semantics | ✔ bracket span, not measure | diagnostic only (R6) | thickness-weighted sum (C-18) | corroborated → R-32 |
| 39 | Momentum-equation term inventory | out of slice | vague (":621 pressure balance") | full spec (C-03…C-05) | **pending cross-slice** (§5) |
| 40 | Weaver / power-law asymptotics | — | — | C-12/C-13/C-14 | **single-lens gates** (§5) |
| 41 | `MAX_SEGMENTS` / streaks recorded as *numerical* exits | ✔ several exits set no end code | ✔ A16 ("real bug surface") | required (C-33/C-45) | corroborated → R-18 |
| 42 | `max()` work non-conservation | `max()` confirmed present | **silent** | must be documented (C-40) | single-lens → R-26 |
| 43 | Cooling-table refresh vs SN onset / age-file boundary | ✔ up to `5e-3` Myr stale, no forced refresh | interval documented | must force refresh (C-44) | corroborated → R-20 |
| 44 | `T_ode` vs `T` from the structure solve | both exist; no comparison | — | must not drift (C-41) | single-lens → R-28 |
| 45 | Purity of `_pure` functions | no aliasing found; RHS closes over **live** params | "no dict mutations during integration" | strict purity (C-39) | corroborated (latent) → R-39 |

**Never addressed by any lens** (recorded so a later pass does not mistake it for cleared):
the actual contents of `build_implicit_phase_events` / `check_event_termination`; the definition of
`effective_Lloss_from_params`; whether `params['t_now']` is the global SPS clock; the `.param`
schema unit for `PISM`; whether `F_HII`/`P_drive` reach `dictionary.jsonl`.

---

## 2. Divergence table

`AB` = code vs its own prose (doc-drift) · `AC` = code vs derived physics · `BC` = prose vs
literature · `ABC` = all three differ or all three flag a gap · `scope-creep` = agreed but
unsanctioned.

| # | Item (line) | A says | B says | C says | Class | Verdict |
|---|---|---|---|---|---|---|
| D1 | cooling-balance criterion (`:1296`) | `(Lgain−Lloss)/Lgain < 0.05`, i.e. `Lloss/Lgain > 0.95`; log message states the **inverse** | criterion never stated; only "default 0.05" | `Ėb/Lgain ≤ 0.05` (PdV **included**) | **AC** (+AB on the log line) | **R-02, top priority.** With B's F2 (`Ėb_bal = Lgain − Lloss − 4πR2²v2Pb`) the two triggers are *not* one quantity at two thresholds: `ebpeak` fires at `Lloss ≈ Lgain − PdV`, `cooling_balance` at `Lloss ≈ 0.95 Lgain`. In the Weaver limit PdV ≈ 0.55·Lw, so **ebpeak fires first** — inverting SPEC-014 Reading B's "ebpeak is strictly later". The paper settles which `Lloss` the published criterion uses. |
| D2 | no-root streak (`:116` vs `:741` vs `:840` vs `:865`) | at streak ≥ 50 sets `no_physical_root_handoff` and **breaks** | ":116 hands off" / ":741 log-only" / ":840 NOT a transition trigger" | must be discriminated from numerics by `Lcool/Lgain ≈ 1` | **AB** + **AC** | **R-03.** Code matches `:116`; `:741` and `:840` are stale and must be corrected. C's discriminator is absent → a root-find failure is reported as a physical handoff. |
| D3 | event terminality/direction (`:752`, `:1096`) | breaks on any `triggered`; cannot see terminal flags | **nothing documented at all** | terminal + explicit direction + start-of-segment re-test | **ABC** (gap) | **R-04.** Largest documentation gap in the slice; the actual defect is in `phase_events.py` (out of slice). A's `:1096` bypass of `classify_energy_collapse` is a *confirmed* defect regardless. |
| D4 | `classify_energy_collapse` partition (`:184`) | 3-way: non-finite → stop; `Eb≤0` → momentum; else None | identical 3-way | **4-way**, incl. `0 < Eb ≤ floor` → handoff | **AC** (A=B) | R-06/R-27. The code only hands off *after* `Eb` has already gone non-positive; with `Ėb` frozen, the overshoot is unbounded and is then clamped to a fixed `1e3`. |
| D5 | energy-collapsed guard rationale (`:1358`) | `energy_collapsed` = **non-finite** branch; that path saves a snapshot containing NaN/inf `Eb` | comment justifies the guard with "there `Eb < 0` → `Pb ~ −1.6e18`" | non-finite must be a hard error | **AB** | **R-07.** Comment is stale (pre-dates the momentum routing). The guard fires on the branch it describes wrongly, and the row it writes carries NaN, not `−1.6e18`. |
| D6 | `Pb` vintage for β vs `Ed` (`:826` vs `:939` vs `:992`) | β solved against the **previous** `Pb`; `Ed` built from the **new** `Pb` | ordering prose only justifies β-before-shell (which A confirms is satisfied) | closure identity `Ėb = Eb(3α−β)/t` must hold | **AC** (B silent) | R-05. Single-lens, high confidence in the reading; C-12 supplies a free run-level gate that would expose it. |
| D7 | leak `Pb`/`c_sound` one segment stale (`:813`) | defect (A-01, S2) | **documented** intentional "1-step frozen" (A3) | — | **none** (A=B) | **Demoted** to a documented approximation (folded into R-20). Undocumented residue: the leak mixes a *fresh* `R2` with the stale `Pb`. |
| D8 | `DT_SEGMENT_COLLAPSE` vs `DT_SEGMENT_MIN` (`:145`/`:113`/`:395`) | `5e-5 < 1e-4`; `max(dt/f, MIN)` **doubles** dt on the shrink branch | **predicted the hazard from prose alone** (FLAG-15) | clamp must be monotone | **AB** corroboration of a real numeric inversion | **R-14.** Strongest independent-corroboration pattern in the slice: B derived the ordering requirement blind, A measured the constants that violate it. |
| D9 | sign flip in `compute_max_dex_change` (`:312`/`:317`) | scores `1.0` (large change) | two comments prescribe opposite behaviour | must force shrink | **AB**, C supports A | **R-17** demoted from `silent-failure` to **doc-drift**: the code is correct, `:312` is stale. |
| D10 | zero-valued monitor key (`:315`) | skipped → contributes `0.0` dex | — | must not silently contribute `0.0` (C-24) | **AC** | R-17. Latent today (masked by D11); becomes live the moment D11 is fixed. |
| D11 | adaptive controller inert (`:1054`) | 30/35 keys **identical by construction** (capture point) | — | aliasing would kill it (C-26) | **AC**, different mechanisms | **R-16.** C's predicted *outcome* is real; C's predicted *cause* (aliasing) is not — A shows scalar payloads make the reference-holding harmless. Textbook case of the same defect found from two directions. |
| D12 | `atol = 1e-8` scalar (`:171`) | scalar over 4 states spanning ~15 decades | "relaxed from 1e-9", value unstated | must be per-component | **ABC** | R-15. Note the two lenses disagree on *which* direction hurts: C fears a large scalar hurting `v2`; A shows the shipped value is *small*, so the risk is an over-tight demand on `v2` near zero → step failure → **R-01**. A's account is the operative one. |
| D13 | solver failure handling (`:1080`/`:1085`) | free-text reason, `break`, **no** end code, **no** retry | — | numerical exits must be recorded as such (C-33/C-45) | **AC** | **R-01, only S1.** |
| D14 | `F_ram` (`:449`/`:538`/`:559`) | `F_ram = Pb·4πR2²` (a *drive* force); `P_ram` hard-coded `0.0` | ":559 no ram pressure in implicit phase" vs a populated `F_ram` | sweep drag must not live in `ForceProperties` (no `v2` arg) | **AB** naming | R-31 (S4). A resolves it: the name is a misnomer, not a double count — the double-count question itself is out of slice. |
| D15 | `ENERGY_HANDOFF_FLOOR` (`:181`) | a **clamp value** applied after `Eb ≤ 0` | `1e3`, matched to phase-1c, unit unstated | should be **relative**; absolute floor "fires immediately at the low-mass end" | **ABC** | R-27, **C's failure scenario dropped**: the floor is never a trigger, so it cannot fire early. The residual concern (A's) is that `1.9e46 erg` is injected into the handoff state and the overshoot magnitude is never logged. |
| D16 | velocity thresholds (`:143`/`:144`) | apply **only** when `v2 < 0`; they are inward-speed dt controls | ambiguous (`|v2|` vs "only during collapse") | `VELOCITY_THRESHOLD_COLLAPSE` should be ≈0; `EXTREME` ≤ `v_w` (C-43) | **AB** minor | **C-43 dropped**: A shows these are not collapse-*detection* thresholds (detection is `v2<0 ∧ R2<R2_prev`, which A and B agree on), so C's failure scenario cannot occur. |
| D17 | `PISM · k_B` (`:521`) | balances iff `PISM` is `n·T` in **AU** (`pc⁻³ K`) | unit unstated | `.param` declares `P/k_B` in **`K cm⁻³`** | **AC**, unresolved | **R-29, contested.** Both lenses agree `k_B` belongs there; they disagree on the length unit. If no `cm⁻³ → pc⁻³` conversion happens at ingestion this is a ~10⁵⁵ error that would fail loudly — so most likely fine, but nobody has checked. |
| D18 | `μ_conv/μ_ion` on the cloud term but not the ISM term (`:513`/`:521`) | asymmetry, possible defect (A-12, low) | — | `PISM` is a *total-particle* `P/k_B`; the cloud term converts `n_H → n_tot` | **AC**, C explains A | **Demoted to S4/dropped**: C's account makes the asymmetry expected, not a bug. |
| D19 | `F_ion_in` surface (`:535`) | pressure at `rShell`, area at `R2` | — | ambient term is `4πR2²·P_ext` at `R2` | **AC** | R-22. Small while the shell is thin; grows as `(R2/rShell)²`. |
| D20 | shadow "never drives the switch" (`:234`) | both the shadow CSV block **and** the `:1288` drive block exist | docstring asserts it unconditionally; drive path documented elsewhere, same citation | — | **AB** | R-19. The invariant holds only for the default trigger set; the byte-identical guarantee is scoped to that. |
| D21 | `Lgain` post-ODE vs `Lloss` pre-ODE (`:1233`) | confirmed | **documented** and justified ("`Lloss` changes slowly") | cooling refresh must be forced at SN / age-file boundaries (C-44) | **none (A=B)** + **AC** on the SN case | R-20 (S3): a declared approximation whose error is unbounded exactly where it is invoked. |
| D22 | grow/shrink share one 0.05-dex threshold | confirmed (no hysteresis) | gain 0.1 dex > deadband 0.05 dex ⇒ limit cycle (B-23) | hysteresis required (C-29) | **BC**, A supplies constants | R-33 (S4). **Demoted**: A's D11 shows `max_dex_change` is usually ~0, so `dt` almost certainly pins at `DT_SEGMENT_MAX` rather than oscillating. |
| D23 | `"unknown"` termination reason | **unreachable** (A-19) | ":1409 a real bug surface — surface it loudly" | — | **AB** | R-34 (S4 dead code). B's alarm is about a path that cannot be taken. |
| D24 | snapshot force list (`:8`) | `F_HII` **is** written; `F_ISM` is monitored but **never** written | docstring lists 4 forces; `ForceProperties` has 5 | — | **AB** | R-40 (S4). |
| D25 | `Eb` linear ⇒ mid-segment zero crossing | A-05 (from the code) | B-12 (from the prose) | C-16 (from the physics) | **ABC** | **R-06.** Three lenses, three routes, one concern — but the decisive fact (does the RHS recompute `Pb` from `y[2]`?) is out of slice. |

---

## 3. Merged, ranked findings

See the JSON block at the end. Ordering: S1, then S2 by expected impact on a number the code
reports, then S3, then S4.

- **Status:** 28 corroborated · 11 single-lens · 1 contested.
- **Divergence class:** 15 × AC (code vs derived physics) · 10 × AB (doc-drift) · 5 × ABC ·
  1 × BC · 9 × none (single-lens observations no other lens spoke to).
- **Confidence:** 21 high · 15 medium · 4 low. No `scope-creep` entries: everything the three
  lenses agreed on is sanctioned by the physics spec somewhere.

---

## 4. Dropped or demoted — and why

**Dropped as verified-correct** (a lens's expectation is met, per another lens's direct reading):

| Dropped | Why |
|---|---|
| C-01 `FOUR_PI` | A reads `FOUR_PI = 4.0*np.pi`. Exact. |
| C-06 `F_grav` half-mass factor | A (`:491`, literal `0.5`) and B (F1, `:488`) give C's expression verbatim. Triple agreement. |
| C-09 double PdV subtraction in the RHS | A: `out[2]` is `Ed_from_beta` verbatim; the third component of the inner call is discarded. C's 55 %-error scenario cannot occur. |
| C-20 / C-21 `evaluate_r1_shadow` | A: `R2 > k_blowout·rCloud` and `edot_balance ≤ 0`; `k` multiplies `rCloud`. Both directions correct. |
| C-23 `r1_transition_decision` | A: OR, gated on membership in `active_triggers`, blowout precedence. Exactly C's requirement. |
| C-27 `update_unconverged_streak` | A: returns 0 on converged, `+1` otherwise. |
| C-28 `betadelta_phase_summary` | A: `n == 0` guarded (`pct = 0.0`). Only residue: `clean` is vacuously `True` at `n==0` — noted, not filed. |
| C-37 `min_step` is LSODA-only | A and B both confirm `ODE_METHOD == 'LSODA'` and conditionally-built kwargs. Latent only if the method changes. |
| B-19 `cool_alpha` refreshed only at entry? | A: rewritten every segment at `:798` from the current `t_now·v2/R2`. Question answered; not a finding. |

**Demoted** (the concern survives, the framing does not):

| Item | Demotion |
|---|---|
| C-26 monitor-dict aliasing (S2 → folded into R-16) | A: the dict holds references, but the payloads are scalar floats, so before/after cannot co-mutate. The *outcome* C feared is real via a different route (capture point). |
| C-17 absolute `ENERGY_HANDOFF_FLOOR` "fires immediately at the low-mass end" (S3 → R-27) | A: the floor is a **clamp applied after** `Eb ≤ 0`, never a trigger. C's scale-dependence argument does not apply to triggering; it does apply to the injected magnitude. |
| C-43 velocity thresholds vs wind terminal speed (S4 → dropped) | A: the thresholds gate `dt_segment` and only when `v2 < 0`. They are not collapse detection (`v2<0 ∧ R2<R2_prev`, A=B) and never compared with `v_w`. |
| C-34 degenerate-event start-of-segment re-test (S2 → folded into R-04) | A: every inline exit test runs at the bottom of the segment body, i.e. before the next integration, so satisfied conditions *are* re-tested. Remains open only for the `solve_ivp` events themselves. |
| A-01 leak `Pb`/`c_sound` one segment stale (S2 → S3, in R-20) | B: documented as an intentional "1-step frozen" approximation with a stated rationale. Not undocumented drift. |
| A-12 missing `μ` factor on the ISM pressure term (S2 → dropped) | C: `PISM` is a total-particle `P/k_B`, so it needs no `n_H → n_tot` correction; the asymmetry is expected. Only the length-unit question survives (R-29). |
| B-03 sign-flip "skip" (S3 silent-failure → S3 doc-drift, in R-17) | A: the code scores a sign flip as `1.0` dex, i.e. correctly. Only the `:312` comment is wrong. |
| B-23 controller limit cycle (S4, low → R-33) | A: `max_dex_change` is ~0 on 30/35 keys, so `dt` pins at the cap rather than oscillating. |
| A-33 events built once before the loop (S4, low → folded into R-04) | Unresolvable in this slice by A's own admission; kept as the cross-slice question, not a standalone finding. |

**Not filed as findings against this file** (out of slice — see §5): C-03, C-04, C-05, C-07, C-08,
C-11 (partly), C-13, C-15, C-38.

---

## 5. Pending — questions this triangulation cannot close

1. **The actual WARPFIELD paper (Rahner+17/19) settles D1.** Does the published
   energy→momentum criterion compare *cooling losses* to *mechanical input*
   (`Lloss/Lgain > 1−ε`, what the code does) or the *net* bubble-energy rate
   (`Ėb/Lgain < ε`, what C derived)? The difference is the `PdV` term, ≈55 % of `L_w` in the Weaver
   limit, and it decides both the phase-end time and which of `cooling_balance`/`ebpeak` fires
   first. Lens C had no literature access and says so.
2. **`phase_events.py`** (`build_implicit_phase_events`, `check_event_termination`) — terminality,
   direction, root conditions, whether `cooling_balance` exists as *both* an event and an inline
   check (B's FLAG-07), and whether `cooling_balance_factory` being unused (A) means the event form
   is inert.
3. **`get_ODE_Edot_pure` / `compute_R1_Pb`** — whether `dR2/dt` is literally `y[1]` (C-02), whether
   `Pb` is recomputed from `y[2]` inside the RHS (decides the severity of R-06), whether the sweep-up
   drag and `pdot_w`/`pdot_SN` appear exactly once (C-03/C-04/C-05), and whether `Pb` uses the
   γ=5/3-only `2π` coefficient (C-07).
4. **Clock origin** — is `params['t_now']` the global cluster age the SPS table is indexed on?
   C-11 makes this load-bearing for `α`, `β`, `δ` and `Ṫ = δT/t`; neither A nor B can say.
5. **`PISM` schema unit** — `K cm⁻³` (C) or already-converted `K pc⁻³` (A's assumption)?
6. **Cheap global gates C proposes that would settle several items at once:** `Ėb·t/Eb − (3α−β) ≈ 0`
   in the constant-`L_mech` window (tests R-05 and the closure), `Eb(t)` linear within a segment
   (confirms R-06's premise), `Eb/(L_w t) → 5/11` on a gravity/radiation-free uniform run,
   and `α, β, δ → (η, 3η−1, −2η/7)` with `η = 3/(5−w)` on power-law clouds.

---

```json
[
  {
    "id": "S5b-R-01",
    "file": "trinity/phase1b_energy_implicit/run_energy_implicit_phase.py",
    "line": 1085,
    "class": "silent-failure",
    "severity": "S1",
    "claim": "A solve_ivp failure (exception, or sol.success False) ends the phase with only a free-text termination_reason: no SimulationEndCode, no SimulationEndReason, no EndSimulationDirectly, and no dt reduction or retry.",
    "evidence": "Lens A: :1080-1083 (except Exception) and :1085-1090 both set termination_reason and break, unlike :768-774, :1041-1044, :1152-1157, :1311-1313, :1321-1323, :1330-1332 which all set SimulationEndCode + EndSimulationDirectly. Lens C independently requires (C-33, C-45) that numerical cutoffs be recorded as a distinct fate with an explicit outcome/exit_code, never returned as a normal completion. Lens B records the same design intent from the other side (A16: '\"unknown\" means we fell through every known exit path -- a real bug surface, surface it loudly').",
    "expected": "A failed integration retries at dt_segment/ADAPTIVE_FACTOR down to DT_SEGMENT_MIN, and on final failure sets an explicit numerical-failure end code so downstream output records that the run did not complete physically.",
    "failure_scenario": "LSODA hits min_step=1e-6 Myr trying to satisfy the scalar atol=1e-8 as v2 crosses zero (see S5b-R-15). The phase returns 'solver_failed: Required step size is less than spacing between numbers.' with SimulationEndCode still holding whatever a previous phase set, and main hands off to the next phase as though the energy phase ended normally, from the last successful segment's state.",
    "repro": "Temporarily set ODE_MIN_STEP above ODE_MAX_STEP to force a failure, then assert params['SimulationEndCode'].value reflects a failure; it does not. Also assert no output row is produced past the failure point.",
    "confidence": "high",
    "lenses": ["A", "C"],
    "divergence": "AC",
    "status": "corroborated",
    "source_ids": ["S5b-A-04", "S5b-C-33", "S5b-C-45"]
  },
  {
    "id": "S5b-R-02",
    "file": "trinity/phase1b_energy_implicit/run_energy_implicit_phase.py",
    "line": 1296,
    "class": "divergence",
    "severity": "S2",
    "claim": "Three mutually inconsistent readings exist of the criterion that ends this phase on every default run: the code tests (Lgain-Lloss)/Lgain < 0.05 with Lloss = cooling + leak and NO PdV term; the log line states the inverse ('Lloss/Lgain ratio below threshold'); and the derived physics expects the net-energy form Edot_b/Lgain <= 0.05, which does include PdV.",
    "evidence": "Lens A: the test at :1296 is (Lgain - Lloss)/Lgain < threshold, i.e. Lloss/Lgain > 1 - threshold, while the log at :1298 says 'Lloss/Lgain ratio below {threshold}'. Lens B (FLAG-11/B-09): the criterion, its ratio and its inequality direction are documented NOWHERE, only the '(default 0.05)' threshold. Lens C: SPEC-013's criterion is L_loss/L_gain -> 1 restated as Edot_b/L_gain -> 0, and C explicitly warns that if the cooling_balance L_loss excludes the PdV work the two triggers stop being one quantity at two thresholds. Combining A's code formula with B's F2 (Edot_from_balance = Lmech - (bubble_LTotal + leak) - 4*pi*R2^2*v2*Pb) gives: ebpeak fires at Lloss ~ Lgain - PdV, cooling_balance at Lloss ~ 0.95*Lgain. With PdV ~ 0.55*Lw in the Weaver limit, ebpeak fires FIRST -- inverting SPEC-014 Reading B's 'ebpeak is strictly later'.",
    "expected": "The criterion written out in prose with its ratio and direction; the log message matched to the branch; and the choice of whether the PdV work belongs inside Lloss settled against the published WARPFIELD criterion, with the cooling_balance/ebpeak ordering made consistent with it.",
    "failure_scenario": "If the intended criterion is the net-energy form, the default phase 1b runs well past the point at which the bubble's energy peaks and ends late by the amount of trajectory over which Lloss climbs from ~0.45*Lgain to ~0.95*Lgain -- shifting the energy->momentum transition time, which is the code's headline prediction, on every config run today.",
    "repro": "Log Lgain, Lloss and Edot_from_balance per segment on param/simple_cluster.param; find the t at which Edot_from_balance first crosses 0 and the t at which (Lgain-Lloss)/Lgain first drops below 0.05, and report the gap. Then check the published criterion.",
    "confidence": "medium",
    "lenses": ["A", "B", "C"],
    "divergence": "AC",
    "status": "corroborated",
    "source_ids": ["S5b-A-10", "S5b-B-09", "S5b-C-20"]
  },
  {
    "id": "S5b-R-03",
    "file": "trinity/phase1b_energy_implicit/run_energy_implicit_phase.py",
    "line": 865,
    "class": "silent-failure",
    "severity": "S2",
    "claim": "The consecutive no-physical-root streak DOES end the phase at NO_ROOT_HANDOFF_STREAK = 50 (two comments say it does not), and the handoff is not discriminated from a mere root-find failure.",
    "evidence": "Lens A: :847-882 counts and logs, and at no_root_streak >= 50 sets termination_reason = 'no_physical_root_handoff' and breaks. Lens B (FLAG-01/B-01): :116-121 says the streak hands off to momentum, :741-742 says it is 'Log-only', :840-846 says it is 'a logged safety net, NOT a transition trigger (phase end stays owned by the cooling-balance event)'. A resolves the three-way contradiction in favour of :116-121, so :741-742 and :840-846 are stale. Lens C (C-32): losing the (beta,delta) root is the signature of catastrophic cooling but equally of a bad bracket or a bad initial guess; the free discriminator is that Lcool/Lgain must be ~1 at a genuine exit.",
    "expected": "Correct the two stale comments; and gate the handoff on the cooling balance corroborating it (Lcool/Lgain within a factor ~2 of 1), otherwise record an explicit numerical failure rather than a plausible-looking physical transition.",
    "failure_scenario": "A 50-segment run of bad brackets on a stiff config terminates the energy phase at a time set by the root-finder's health rather than by physics, and the output is indistinguishable from a genuine catastrophic-cooling transition. Compounded by S5b-R-18: this break sets no SimulationEndCode/Reason.",
    "repro": "Force the no-root condition on a high-mass config (B's A11 records a 1e6 Msun run pinned at beta=1); confirm termination_reason == 'no_physical_root_handoff' and record Lcool/Lgain at that segment.",
    "confidence": "high",
    "lenses": ["A", "B", "C"],
    "divergence": "AB",
    "status": "corroborated",
    "source_ids": ["S5b-B-01", "S5b-C-32", "S5b-A-16"]
  },
  {
    "id": "S5b-R-04",
    "file": "trinity/phase1b_energy_implicit/run_energy_implicit_phase.py",
    "line": 1096,
    "class": "state",
    "severity": "S2",
    "claim": "Event semantics are undocumented and unverifiable from this file; the runner breaks unconditionally on any triggered event, and that path bypasses classify_energy_collapse entirely, so an event firing at Eb <= 0 or non-finite Eb writes that value into params with no floor and no ENERGY_COLLAPSED code.",
    "evidence": "Lens A: the event block at :1096-1119 assigns Eb = float(event_result.y[2]) at :1103, calls apply_event_result(..., state_keys=['R2','v2','Eb','T0']) at :1117 and breaks at :1119 -- before the collapse classification at :1148; ode_events is also built once at :752 from a params that then mutates every segment. Lens B (FLAG-18/B-16): no event's terminal flag, crossing direction or root condition is documented anywhere in the slice, despite events being able to end the phase and set the recorded final state. Lens C (C-34): exit events must be terminal with explicit direction (+1 for R2-k*rCloud and R2-stop_r, -1 for Eb-floor and v2-threshold) and the runner must re-test exit conditions at the segment start because SciPy detects events only by a sign change within a step. C-16: the Eb<=0 branch must never fall through.",
    "expected": "Apply the collapse classification to the state accepted on the event path as well as the normal path; document each event's root condition, terminal flag and direction; and confirm whether any event closure captures params values at build time (if so it must be rebuilt per segment).",
    "failure_scenario": "An event (shell collapse, radius, stop_t) fires in the same segment in which Eb goes negative. The phase exits with that event's reason_code, Eb<0 in params, and the reconciliation at :1370 calls compute_R1_Pb(R2, Eb<0, ...) producing a negative or NaN Pb which is written to params['Pb'] and snapshotted.",
    "repro": "Assert classify_energy_collapse(params['Eb'].value) is None at phase exit for every termination_reason that came from an event. Separately, enumerate build_implicit_phase_events' returns and record terminal/direction for each.",
    "confidence": "high",
    "lenses": ["A", "B", "C"],
    "divergence": "ABC",
    "status": "corroborated",
    "source_ids": ["S5b-A-08", "S5b-A-33", "S5b-B-16", "S5b-C-34", "S5b-C-16"]
  },
  {
    "id": "S5b-R-05",
    "file": "trinity/phase1b_energy_implicit/run_energy_implicit_phase.py",
    "line": 826,
    "class": "state",
    "severity": "S2",
    "claim": "beta is solved against the previous segment's Pb, then Ed is built from beta together with the NEW Pb -- the implicit closure and the imposed dEb/dt use different pressures by construction.",
    "evidence": "Lens A: solve_betadelta_pure(...) at :826 reads a params whose 'Pb' still holds the value written at :939 of the previous segment; compute_R1_Pb overwrites params['Pb'] at :936-939; then :992 calls cool_beta_to_Ebdot_pure(beta, Pb, ...) with the new Pb and the old-Pb beta. Lens B's ordering prose (:822-824) only justifies beta-before-shell-structure, which A confirms is satisfied -- it says nothing about the Pb vintage, so the mismatch is undocumented. Lens C supplies the gate: the identity Edot_b = Eb*(3*alpha - beta)/t (C-12, derived and cross-checked against E_b = (5/11)L_w t) must hold in the constant-Lmech window; a vintage mismatch shows up there.",
    "expected": "compute_R1_Pb runs before the beta-delta solve, so the residual that defines beta and the Ed derived from beta use the same Pb.",
    "failure_scenario": "On stiff configs where Pb varies by more than the beta-delta residual tolerance across one dt_segment (f1edge_hidens-style high-density runs), the imposed dEb/dt is not the derivative the implicit closure converged, so Eb drifts off the closure by O(dPb/dt * dt) per segment over up to 5000 segments.",
    "repro": "Log Pb at :826 and at :939 per segment and assert equality; or recompute cool_beta_to_Ebdot_pure with the :826-vintage Pb and compare Ed. Then run C-12's identity check over the phase.",
    "confidence": "high",
    "lenses": ["A", "C"],
    "divergence": "AC",
    "status": "single-lens",
    "source_ids": ["S5b-A-02", "S5b-C-12"]
  },
  {
    "id": "S5b-R-06",
    "file": "trinity/phase1b_energy_implicit/run_energy_implicit_phase.py",
    "line": 624,
    "class": "numerical",
    "severity": "S2",
    "claim": "dEb/dt is frozen for the whole segment, so Eb is exactly linear in t and can cross zero mid-segment; the RHS keeps being evaluated with Eb < 0 for the remainder of the segment, and the collapse test only runs after the segment completes.",
    "evidence": "Independently reached by all three lenses. Lens A: get_ODE_implicit_pure returns [rd, vd, Ed, Td] with Ed/Td computed once at :992-993, hence Eb(t) = Eb0 + Ed*(t-t0) exactly; Eb is nevertheless passed back into get_ODE_Edot_pure as y[2] (:617-618); classify_energy_collapse is only called at :1148. Lens B (FLAG-14/B-12), from the prose alone: 'dEb/dt is constant over the segment, so Eb is linear in t and can cross zero mid-segment; the collapse check at :1140 is post-ODE', and asks whether the RHS recomputes Pb from y[2]. Lens C (C-16): a negative Eb gives Pb = Eb/[2pi(R2^3-R1^3)] < 0 and hence an inward 4*pi*R2^2*|Pb| force -- a sign inversion of the dominant driving term -- so the Eb<=0 branch must be terminal, never fall through.",
    "expected": "A terminal event on Eb (direction -1) inside ode_events, or a mid-segment guard, so the handoff at :1164 is taken at the crossing rather than after an arbitrary excursion into Eb < 0.",
    "failure_scenario": "A segment starting at Eb=1e4 with Ed=-1e8 and dt_segment=5e-4 Myr crosses zero after 1e-4 Myr; the remaining 4e-4 Myr is integrated with Eb<0. If the RHS derives Pb from y[2] the shell is pulled inward by a fictitious negative pressure (or the RHS returns NaN and the run takes the S5b-R-01 path) before the guard ever runs.",
    "repro": "Log min(sol.y[2]) versus sol.y[2,-1] per segment; find a segment whose minimum is negative but whose endpoint is not. Decisive prerequisite: determine whether get_ODE_Edot_pure derives Pb from y[2] or from the frozen snapshot.",
    "confidence": "medium",
    "lenses": ["A", "B", "C"],
    "divergence": "ABC",
    "status": "corroborated",
    "source_ids": ["S5b-A-05", "S5b-B-12", "S5b-C-16"]
  },
  {
    "id": "S5b-R-07",
    "file": "trinity/phase1b_energy_implicit/run_energy_implicit_phase.py",
    "line": 1137,
    "class": "state",
    "severity": "S2",
    "claim": "A non-finite Eb is written into params and persisted: the 'energy_collapsed' path saves a snapshot and appends a results row containing NaN/inf -- and the comment justifying that path's existence describes a condition (finite Eb < 0 producing Pb ~ -1.6e18) that its own documented partition routes elsewhere.",
    "evidence": "Lens A: params['Eb'].value = Eb at :1137 happens before classify_energy_collapse at :1148; on the 'stop' branch (:1149-1163) Eb is never repaired, :1365 routes 'energy_collapsed' to :1397-1402 which calls params.save_snapshot() unconditionally, and the same non-finite Eb is appended at :1345. Lens B (FLAG-09/B-07): classify_energy_collapse's docstring partitions non-finite -> 'stop'/ENERGY_COLLAPSED and finite Eb<=0 -> 'momentum', yet :1358-1361 and :1398-1401 justify the reconciliation skip with 'there Eb < 0 ... Pb ~ -1.6e18'. A confirms B's reading of the partition, so the comment is stale from before the momentum routing existed. Lens C (C-16): non-finite Eb must be a hard error.",
    "expected": "Either repair Eb on the non-finite path or skip the snapshot entirely; and re-key the guard's comment on the condition that actually reaches it.",
    "failure_scenario": "Any run whose bubble energy goes non-finite writes a final dictionary.jsonl row with Eb=NaN, poisoning any downstream reader that averages or interpolates the energy trajectory -- while the comment claims the garbage-row problem was solved.",
    "repro": "Assert np.isfinite(ImplicitPhaseResults.Eb).all() for a run terminating with 'energy_collapsed'; inspect the terminal Eb/Pb in dictionary.jsonl.",
    "confidence": "high",
    "lenses": ["A", "B"],
    "divergence": "AB",
    "status": "corroborated",
    "source_ids": ["S5b-A-06", "S5b-B-07"]
  },
  {
    "id": "S5b-R-08",
    "file": "trinity/phase1b_energy_implicit/run_energy_implicit_phase.py",
    "line": 114,
    "class": "numerical",
    "severity": "S2",
    "claim": "The phase-transition time is resolved only to +/- dt_segment with no refinement, dt_segment's cap is an absolute 5e-2 Myr rather than a fraction of a local timescale, and the controller that should shrink it is largely inert (S5b-R-16) -- so the headline output carries an unquoted error of order the maximum segment length.",
    "evidence": "Lens A: the cooling_balance / blowout / ebpeak tests are post-ODE inline checks at segment boundaries (:1288-:1300); nothing bisects or re-solves within the last segment; DT_SEGMENT_MAX = 5e-2 Myr (:114); dt_segment is capped there by next_dt_segment. Lens C (C-35): cooling_balance and ebpeak depend on Lcool, known only at segment boundaries, so those transitions are resolved only to +/- dt_segment unless refined. C-30: the cap must be relative (dt << R2/v2, dt << t, dt << t_cool) because alpha/beta/delta are logarithmic derivatives with respect to t and t_dyn ranges over decades across the shipped grid.",
    "expected": "Either a refinement step (bisect or re-solve the last segment) or an explicitly quoted uncertainty of order dt_segment on the transition time; and dt_segment = min(absolute cap, eps*R2/v2, eps*t).",
    "failure_scenario": "On a dense-cloud edge config where t_dyn ~ 1e-2 Myr, a segment at the 5e-2 Myr cap exceeds the dynamical time, so the frozen Edot/Tdot are stale by O(1) and the reported transition time is both biased and quantised at the segment length.",
    "repro": "Record dt_segment/(R2/v2) and dt_segment/t over docs/dev/performance/f1edge_hidens*.param; both should stay well below ~0.1. Then halve DT_SEGMENT_MAX and compare the reported transition time on the same config (project rule 5: separate processes, matched t).",
    "confidence": "medium",
    "lenses": ["A", "C"],
    "divergence": "AC",
    "status": "corroborated",
    "source_ids": ["S5b-C-35", "S5b-C-30", "S5b-A-27", "S5b-A-03"]
  },
  {
    "id": "S5b-R-09",
    "file": "trinity/phase1b_energy_implicit/run_energy_implicit_phase.py",
    "line": 847,
    "class": "state",
    "severity": "S2",
    "claim": "A no-physical-root beta-delta solve does not skip or restore: its beta/delta are written to params and drive Ed/Td for up to 49 consecutive segments before the handoff fires.",
    "evidence": "Lens A: beta, delta are taken unconditionally at :832-833; the no_physical_root branch at :847-879 only counts, logs and (at streak >= 50) breaks -- no continue, no restore; :885-886 write them to params and :992-993 build Ed and Td from them. Lens B's A7 documents the neighbouring intent -- on such a segment 'the structure values and the dMdt warm start hold at the last physical segment' -- which A confirms for bubble_properties (None -> updateDict skipped) but which does NOT hold for beta/delta. Lens C (C-32) requires that reaching the shrink floor while still unconverged be an explicit recorded failure, never a silent continue.",
    "expected": "Hold the last physical (beta, delta) on a no-root segment, matching what the surrounding logging claims about holding the last physical dMdt -- or skip the segment outright.",
    "failure_scenario": "Up to 49 segments are integrated with dEb/dt and dT0/dt derived from a (beta, delta) the solver itself flagged as unphysical; at dt_segment up to 5e-2 Myr that is up to ~2.45 Myr of trajectory.",
    "repro": "Log (beta, delta) alongside the no_physical_root flag per segment; on a no-root segment they differ from the previous segment's values.",
    "confidence": "medium",
    "lenses": ["A", "C"],
    "divergence": "AC",
    "status": "corroborated",
    "source_ids": ["S5b-A-16", "S5b-C-32"]
  },
  {
    "id": "S5b-R-10",
    "file": "trinity/phase1b_energy_implicit/run_energy_implicit_phase.py",
    "line": 1303,
    "class": "state",
    "severity": "S2",
    "claim": "isCollapse is a one-way latch that permanently freezes the shell mass; nothing in the module clears it when the shell re-expands.",
    "evidence": "Lens A: :1302-1303 sets params['isCollapse'].value = True when v2 < 0 and R2 < R2_prev, and nothing sets it back to False; both shell-mass blocks (:957-960 and :1188-1189) skip the mass-profile evaluation entirely when it is set, so shell_mass and shell_massDot freeze for the rest of the phase. Lens B documents the freeze-during-collapse and never-decrease invariants (:947-963, :1181-1194) but never documents clearing. Lens C (C-42) sanctions the never-decrease rule (a shell falling back moves through the evacuated interior and does not un-sweep) but says nothing that would sanction a permanent freeze through re-expansion.",
    "expected": "Clear the latch when the shell re-expands (v2 > 0 and R2 > R2_prev), or apply the freeze only while the collapse condition holds -- the never-decrease ratchet alone already prevents un-sweeping.",
    "failure_scenario": "A bubble that dips for one segment and then re-expands past rCloud keeps M_sh at the pre-dip value forever, so F_grav (proportional to M_sh*(M_cl + M_sh/2)) and the shell inertia are frozen while R2 grows by orders of magnitude.",
    "repro": "Count segments after the first isCollapse=True and assert params['shell_mass'] changes on a run where v2 recovers; it does not.",
    "confidence": "high",
    "lenses": ["A"],
    "divergence": "none",
    "status": "single-lens",
    "source_ids": ["S5b-A-14"]
  },
  {
    "id": "S5b-R-11",
    "file": "trinity/phase1b_energy_implicit/run_energy_implicit_phase.py",
    "line": 514,
    "class": "silent-failure",
    "severity": "S2",
    "claim": "A bare `except Exception` silently sets the ambient confining pressure to zero, removing the inward pressure term from the force budget with no log line.",
    "evidence": "Lens A: :509-515 wraps get_density_profile and the P_ext expression; any failure yields P_ext = 0.0, which makes F_ion_in = 0 at :535 and press_HII_in = 0 at :560. Neither B nor C addresses this block; B's force contract (:467-487) describes compute_forces_pure as purely computational with no failure mode.",
    "expected": "A density-profile failure at the shell radius should propagate, or at minimum log at WARNING with the exception and a counter.",
    "failure_scenario": "get_density_profile raises for rShell outside the tabulated profile (e.g. after blowout, rShell > rCloud with a profile that only spans the cloud). Every subsequent segment silently drops the inward ambient pressure term with no diagnostic at all.",
    "repro": "Add a counter inside the except and run a config in which rShell exceeds the profile support; the counter is non-zero and nothing appears in the log.",
    "confidence": "high",
    "lenses": ["A"],
    "divergence": "none",
    "status": "single-lens",
    "source_ids": ["S5b-A-11"]
  },
  {
    "id": "S5b-R-12",
    "file": "trinity/phase1b_energy_implicit/run_energy_implicit_phase.py",
    "line": 1394,
    "class": "silent-failure",
    "severity": "S2",
    "claim": "The phase-boundary reconciliation swallows every exception, and because save_snapshot() is the last statement inside the try, any failure loses the final snapshot with only a warning.",
    "evidence": "Lens A: the try opens at :1366 and closes at :1395 with `except Exception as e: logger.warning(...)`; params.save_snapshot() at :1394 is the final statement inside it, so any failure in get_current_sps_feedback, compute_R1_Pb, shell_structure_pure or compute_forces_pure aborts before it. Lens B documents why the reconciliation exists (A15: a bare save_snapshot would save stale derived values AND block the next phase's correct first snapshot via the duplicate guard) but not that its failure is silent.",
    "expected": "Save the snapshot outside the try, or re-raise.",
    "failure_scenario": "shell_structure_pure raises at the final state (e.g. a degenerate shell after the energy_to_momentum clamp sets Eb=1e3 and Pb collapses). The phase's last output row is missing entirely and the only trace is one WARNING line -- and per B's A15 the next phase's first snapshot may then also be affected.",
    "repro": "Force an exception inside the try and assert params.save_count increased; it does not.",
    "confidence": "high",
    "lenses": ["A"],
    "divergence": "none",
    "status": "single-lens",
    "source_ids": ["S5b-A-17"]
  },
  {
    "id": "S5b-R-13",
    "file": "trinity/phase1b_energy_implicit/run_energy_implicit_phase.py",
    "line": 1245,
    "class": "state",
    "severity": "S2",
    "claim": "The two Lloss branches feeding the termination test are semantically asymmetric: one passes a raw cooling luminosity with the real leak, the other passes an already-effective loss (possibly from an earlier segment) with the leak forced to zero.",
    "evidence": "Lens A: :1238-1243 uses _Lcool = bubble_props.bubble_LTotal with _leak = params['bubble_Leak']; :1245-1247 uses _Lcool = params['bubble_Lloss'] -- the value written at :930 from betadelta_result.L_loss, i.e. already an effective loss -- with _leak = 0.0; both are then fed to the same effective_Lloss_from_params. Lens B (FLAG-22/B-21) independently flags the naming: F2 writes the loss slot as 'bubble_LTotal + leak' while F10 writes it as 'Lcool + leak', and the prose asserts they are 'the SAME effective loss as the residual + ODE' -- so the check is whether the two names resolve to one value. They do not.",
    "expected": "Both branches pass the same kind of quantity (raw cooling luminosity) and the same leak.",
    "failure_scenario": "When the beta-delta solve returns bubble_properties=None -- the degraded/no-root path, exactly when the fallback matters most -- Lloss is a double-processed value carried over from an arbitrarily earlier segment with the leak dropped, and it drives the cooling_balance termination test at :1296.",
    "repro": "Force bubble_properties=None for one segment and compare the resulting Lloss with the value the if-branch would give from the same state.",
    "confidence": "medium",
    "lenses": ["A", "B"],
    "divergence": "AB",
    "status": "corroborated",
    "source_ids": ["S5b-A-15", "S5b-B-21"]
  },
  {
    "id": "S5b-R-14",
    "file": "trinity/phase1b_energy_implicit/run_energy_implicit_phase.py",
    "line": 395,
    "class": "numerical",
    "severity": "S3",
    "claim": "DT_SEGMENT_COLLAPSE (5e-5 Myr) is BELOW DT_SEGMENT_MIN (1e-4 Myr), so next_dt_segment's shrink branch -- max(dt/ADAPTIVE_FACTOR, DT_SEGMENT_MIN) -- doubles dt exactly when the controller asked for a reduction, undoing the collapse clamp after one segment.",
    "evidence": "Predicted and measured independently. Lens B (FLAG-15/B-13), from prose alone: three named durations (DT_SEGMENT_MIN, the 50-yr collapse duration, 'intermediate') are mapped onto two velocity bands, and 'if DT_SEGMENT_MIN > 5e-5 Myr, the extreme-collapse band uses a coarser step than the moderate band -- inverting the stated intent'; expected ordering DT_SEGMENT_MIN <= collapse <= DT_SEGMENT. Lens A measured the constants: DT_SEGMENT_MIN = 1e-4, DT_SEGMENT_COLLAPSE = 5e-5, and max(5e-5/1.25893, 1e-4) = 1e-4, a 2x increase on the shrink branch. Lens C (C-29) requires the controller be monotone in max_dex_change and clamped to [MIN, MAX].",
    "expected": "Either the floor becomes min(DT_SEGMENT_MIN, dt) so a shrink never grows dt, or DT_SEGMENT_COLLAPSE is not set below DT_SEGMENT_MIN.",
    "failure_scenario": "A collapsing shell with v2 < -150 pc/Myr is clamped to dt=5e-5; on the first segment where |v2| falls back below 50 pc/Myr with a large dex change, dt is doubled to 1e-4 -- resolution decreasing as the collapse accelerates, in the stiffest regime of the phase.",
    "repro": "assert next_dt_segment(5e-5, 0.5, 0) <= 5e-5 -- it returns 1e-4. Also assert DT_SEGMENT_MIN <= DT_SEGMENT_COLLAPSE.",
    "confidence": "high",
    "lenses": ["A", "B"],
    "divergence": "AB",
    "status": "corroborated",
    "source_ids": ["S5b-A-09", "S5b-B-13"]
  },
  {
    "id": "S5b-R-15",
    "file": "trinity/phase1b_energy_implicit/run_energy_implicit_phase.py",
    "line": 171,
    "class": "numerical",
    "severity": "S3",
    "claim": "A single scalar ODE_ATOL = 1e-8 is applied to four states spanning ~9-15 decades (R2 ~ 1e1 pc, v2 ~ 1e1-1e3 pc/Myr, Eb ~ 1e5-1e10 AU, T0 ~ 1e6-1e7 K), and its provenance is documented only as 'relaxed from 1e-9'.",
    "evidence": "Lens A: :170-173 and :1071-1077 -- scalar atol, min_step=1e-6 Myr, max_step=2e-5 Myr; for the large components rtol=1e-6 dominates, but for a component near zero the effective requirement is |err| < 1e-8 in that component's own units. Lens C (C-36) independently: 'the state spans ~9 decades, so a scalar ODE_ATOL cannot be right for all four components'. Lens B (N4/B-20): the value and the reason for the relaxation are nowhere recorded. Note the two lenses differ on direction: C fears a LARGE scalar destroying v2; A shows the shipped value is small, so the operative risk is an over-tight demand on v2 near zero.",
    "expected": "A per-component atol vector matched to each component's scale (e.g. [1e-8 pc, 1e-6 pc/Myr, 1e-3*Eb_scale, 1e-2 K]) or a non-dimensionalised state; and the atol history recorded with the run that forced the relaxation, per project rule 5.",
    "failure_scenario": "During a collapse v2 passes through 0 pc/Myr; LSODA must resolve v2 to 1e-8 pc/Myr absolute, drives h below min_step=1e-6 Myr, sol.success goes False -- and the phase then takes the S5b-R-01 path that ends with no end code.",
    "repro": "Run a config producing a v2 sign change inside a segment and check sol.message. Separately, tighten rtol 100x; if the trajectory moves more than the original tolerance implies, atol was binding on the wrong component.",
    "confidence": "medium",
    "lenses": ["A", "B", "C"],
    "divergence": "ABC",
    "status": "corroborated",
    "source_ids": ["S5b-A-26", "S5b-B-20", "S5b-C-36"]
  },
  {
    "id": "S5b-R-16",
    "file": "trinity/phase1b_energy_implicit/run_energy_implicit_phase.py",
    "line": 1054,
    "class": "state",
    "severity": "S3",
    "claim": "30 of the 35 ADAPTIVE_MONITOR_KEYS are byte-identical between values_before and values_after by construction, so the dex-based step controller only ever sees R2, v2, Eb, T0 and shell_mass.",
    "evidence": "Lens A: values_before is captured at :1054, AFTER every physics write of the segment (:785-:1009); values_after at :1198. Between them the only params writes are :1134-1138 (t_now, R2, v2, Eb, T0), :1169 (Eb on the momentum branch, which breaks) and :1196 (shell_mass). Pb, R1, cool_beta, cool_delta, all bubble_*, all shell_* except shell_mass, rShell and all F_* are unchanged; F_ISM is never written at all. Lens C (C-26) predicted the same OUTCOME from a different cause -- aliasing of the monitor dict -- rating it S2 ('the adaptive controller never shrinks, every run is silently integrated at DT_SEGMENT_MAX'). A rules out C's mechanism (the dict holds references, but the payloads are scalar floats, so before/after cannot co-mutate) while confirming a stronger one.",
    "expected": "Capture values_before at the top of the segment, before the beta-delta/shell/force block, so the monitored physics quantities can actually move -- or trim the list to the five keys that can.",
    "failure_scenario": "The controller cannot react to a bubble/shell/force excursion -- e.g. bubble_dMdt flipping sign, which compute_max_dex_change would score as 1.0 dex, cannot influence dt_segment at all -- so dt_segment tends to the cap and the operator-splitting error (S5b-R-08) is uncontrolled. Mitigating: the five surviving keys ARE the ODE state, which is a reasonable if narrower proxy, so the practical impact is smaller than C feared.",
    "repro": "Assert compute_max_dex_change(values_before, values_after, [k for k in ADAPTIVE_MONITOR_KEYS if k not in ('R2','v2','Eb','T0','shell_mass')]) == 0.0 on every segment of param/simple_cluster.param -- it holds.",
    "confidence": "high",
    "lenses": ["A", "C"],
    "divergence": "AC",
    "status": "corroborated",
    "source_ids": ["S5b-A-03", "S5b-C-26"]
  },
  {
    "id": "S5b-R-17",
    "file": "trinity/phase1b_energy_implicit/run_energy_implicit_phase.py",
    "line": 315,
    "class": "silent-failure",
    "severity": "S3",
    "claim": "compute_max_dex_change silently ignores any key that is exactly zero in either sample -- the largest possible relative change scores zero dex -- its zero test sits outside the try/except, and one of its two comments describes the opposite of the implemented sign-flip rule.",
    "evidence": "Lens A: :315 `if old_val == 0 or new_val == 0: continue` precedes the try at :323; the except at :326 catches (ValueError, ZeroDivisionError), neither of which np.log10 of a positive ratio can raise, so it is dead -- but an array-valued monitor entry would raise ValueError at :315 where nothing catches it. A sign flip is scored 1.0 (large change). Lens B (FLAG-04/B-03): :312 says 'Skip if values are missing, zero, or opposite signs' while :317-318 says 'Large change if sign flips' -- A resolves this in favour of :317-318, so :312 is stale prose, NOT a code defect. Lens C (C-24): an undefined key must not silently contribute 0.0, and an all-undefined result must force the minimum segment, not the maximum.",
    "expected": "Treat 0 -> non-zero as a maximal change; guard the zero test against arrays (or assert scalars); delete the dead except; correct the :312 comment.",
    "failure_scenario": "F_rad going to exactly 0.0 when shell_props.isDissolved flips, and F_HII going to 0.0 when n_IF_Str drops to 0, are both scored as no change. Latent today because S5b-R-16 already freezes those keys -- but it becomes live the moment R-16 is fixed, which is the dangerous ordering.",
    "repro": "assert compute_max_dex_change({'F_rad': 1e40}, {'F_rad': 0.0}, ['F_rad']) > 0 -- it returns 0.0. And assert a sign flip (+5 -> -5) returns 1.0 -- it does.",
    "confidence": "high",
    "lenses": ["A", "B", "C"],
    "divergence": "ABC",
    "status": "corroborated",
    "source_ids": ["S5b-A-24", "S5b-B-03", "S5b-C-24"]
  },
  {
    "id": "S5b-R-18",
    "file": "trinity/phase1b_energy_implicit/run_energy_implicit_phase.py",
    "line": 1290,
    "class": "state",
    "severity": "S3",
    "claim": "The blowout, ebpeak, cooling_balance and no_physical_root_handoff exits set only a local termination_reason, leaving SimulationEndCode/SimulationEndReason at whatever a previous phase wrote -- so numerical and physical exits are not distinguishable in the output.",
    "evidence": "Lens A: compare :768-774, :1041-1044, :1152-1157, :1311-1313, :1321-1323, :1330-1332 (which write both SimulationEndReason and SimulationEndCode) with :866, :1099, :1290, :1297 (which write only termination_reason). Lens C (C-33, C-45) requires numerical cutoffs -- MAX_SEGMENTS, streak handoffs, solver failure -- to be recorded as a distinct fate with disjoint physical/numerical categories. Lens B's A16 states the same intent for the '\"unknown\"' case.",
    "expected": "Set an explicit hand-off reason/code on every exit path, with physical exits and numerical cutoffs in disjoint categories, or clear SimulationEndReason on phase entry.",
    "failure_scenario": "A run that ends phase 1b via 'blowout' or 'no_physical_root_handoff' carries a stale reason (e.g. 'Stopping time reached') into the metadata of the hand-off snapshot, and a truncated run is post-processed as a physical transition -- contaminating phase-timeline statistics.",
    "repro": "Run with transition_trigger='blowout' and inspect params['SimulationEndReason'] at phase exit.",
    "confidence": "high",
    "lenses": ["A", "C"],
    "divergence": "AC",
    "status": "corroborated",
    "source_ids": ["S5b-A-30", "S5b-C-33", "S5b-C-45"]
  },
  {
    "id": "S5b-R-19",
    "file": "trinity/phase1b_energy_implicit/run_energy_implicit_phase.py",
    "line": 234,
    "class": "citation",
    "severity": "S3",
    "claim": "evaluate_r1_shadow's docstring asserts unconditionally that the R1 criteria are 'computed/logged, never drives the switch', with a byte-identical-output guarantee -- but an opt-in DRIVE path exists that ends the phase on blowout/ebpeak.",
    "evidence": "Lens B (FLAG-06/B-05): :234-236, :721-722 and :1256-1260 make the shadow claim; :276-281 and :1282-1287 document the drive path; both cite the same document (R1_SHADOW_PLAN.md). Lens A confirms both exist in code: the shadow CSV block at :1261-1280/:1435-1448 alongside the termination test at :1288 which breaks on 'blowout'/'ebpeak' when they are in active_triggers. Lens B's A17 already concedes the shadow code is provisional ('the future flip would replace the logging here with a real break').",
    "expected": "Scope the shadow docstring and the byte-identical guarantee to the default trigger set, or document the drive path as consuming a separate evaluation.",
    "failure_scenario": "An audit trusts the byte-identical claim for a non-default config where it does not hold; or a reader relies on 'never drives the switch' and is surprised when transition_trigger='r1' ends the phase on a geometric criterion.",
    "repro": "Run with transition_trigger='cooling_balance,blowout' and confirm termination_reason; compare dictionary.jsonl against the default run.",
    "confidence": "high",
    "lenses": ["A", "B"],
    "divergence": "AB",
    "status": "corroborated",
    "source_ids": ["S5b-B-05"]
  },
  {
    "id": "S5b-R-20",
    "file": "trinity/phase1b_energy_implicit/run_energy_implicit_phase.py",
    "line": 1233,
    "class": "state",
    "severity": "S3",
    "claim": "The inputs to the phase-ending ratio are of three different vintages -- Lgain re-fetched at the post-ODE t, Lloss from the pre-ODE bubble solve, the leak from a Pb/c_sound frozen one segment further back -- and the cooling tables behind Lloss may themselves be up to 5e-3 Myr stale with no forced refresh at the SN onset or the age-file boundaries.",
    "evidence": "Lens A: feedback_post = get_current_sps_feedback(t_now_new, ...) at :1230 while Lcool comes from the pre-step bubble_props and leak from :813 (computed with the previous segment's Pb and c_sound); the cooling-table refresh at :783-788 is gated on |t_prev - t_now| > 5e-3 Myr. Lens B documents all three as intentional (A3 'Pb and c_sound carried from the previous segment, 1-step frozen'; A4 'cannot cheaply recompute Lloss ... acceptable since Lloss changes slowly'; FLAG-12: the justification is weakest exactly at the SN turn-on the fresh-Lgain fetch exists to capture). Lens C (C-44): the cooling refresh must be short compared with the SPS variation timescale and FORCED across the age-indexed non-CIE file boundaries and the first SN, where the true Lcool steps discontinuously.",
    "expected": "Both sides of the ratio at the same t, or a documented bound on the error the one-segment lag introduces relative to the 0.05 threshold; and a forced cooling-structure refresh at known discontinuities.",
    "failure_scenario": "Across an SN turn-on Lgain jumps by ~an order of magnitude while Lloss and the cooling structure lag one segment, so the ratio can cross (or fail to cross) the threshold on a timing artefact rather than physics -- transitioning the phase at the wrong t.",
    "repro": "Log Lgain, Lloss and their source times for segments spanning an SN turn-on; recompute the ratio with a same-t Lloss and compare the transition time. Plot Lcool vs t and look for a plateau persisting past a known age-file boundary.",
    "confidence": "medium",
    "lenses": ["A", "B", "C"],
    "divergence": "AC",
    "status": "corroborated",
    "source_ids": ["S5b-B-10", "S5b-C-44", "S5b-A-01"]
  },
  {
    "id": "S5b-R-21",
    "file": "trinity/phase1b_energy_implicit/run_energy_implicit_phase.py",
    "line": 262,
    "class": "silent-failure",
    "severity": "S3",
    "claim": "An empty or whitespace-only transition_trigger silently yields frozenset() -- every transition trigger disabled, no error -- while None raises.",
    "evidence": "Lens A: :262 parts = {p.strip() for p in str(x).split(',') if p.strip()}; for '' or '  ' this is the empty set, which passes the unknown-token check at :266 and returns frozenset(); r1_transition_decision then always returns None and the cooling_balance test at :1296 is short-circuited. str(None) == 'None' becomes an unknown token and raises. Lens C (C-22) independently requires the parser to 'reject unknown tokens AND an empty result' at this trust boundary. Lens B records only the unknown-token contract.",
    "expected": "Reject an empty trigger set explicitly, and handle None the same way as ''.",
    "failure_scenario": "A .param with transition_trigger = '' runs the implicit phase all the way to MAX_SEGMENTS or stop_t with no phase transition and only an INFO-level completion line.",
    "repro": "assert parse_transition_triggers('') raises or is non-empty; it returns frozenset().",
    "confidence": "high",
    "lenses": ["A", "C"],
    "divergence": "AC",
    "status": "corroborated",
    "source_ids": ["S5b-A-31", "S5b-C-22"]
  },
  {
    "id": "S5b-R-22",
    "file": "trinity/phase1b_energy_implicit/run_energy_implicit_phase.py",
    "line": 535,
    "class": "regime",
    "severity": "S3",
    "claim": "F_ion_in evaluates the confining pressure at rShell but multiplies it by the area of R2.",
    "evidence": "Lens A: P_ext is computed from the density profile at shell_props.rShell (:510) and, on the ISM branch, gated on rShell >= rCloud (:520); F_ion_in = P_ext * FOUR_PI * R2**2 at :535 uses R2. Lens C's momentum equation writes the external term as -4*pi*R2^2*P_ext, i.e. the ambient pressure evaluated at the shell's outer surface R2. Lens B gives no units or radii for the force path at all (U6/FLAG-19).",
    "expected": "Pressure and area referring to the same surface, or the mismatch documented as a deliberate choice.",
    "failure_scenario": "In a thick-shell regime (rShell noticeably different from R2) the inward pressure force is scaled by the wrong area; the error grows as (R2/rShell)^2. Small while the thin-shell assumption holds.",
    "repro": "Log rShell/R2 per segment; where it departs from 1, F_ion_in is inconsistent by that ratio squared.",
    "confidence": "medium",
    "lenses": ["A", "C"],
    "divergence": "AC",
    "status": "corroborated",
    "source_ids": ["S5b-A-13"]
  },
  {
    "id": "S5b-R-23",
    "file": "trinity/phase1b_energy_implicit/run_energy_implicit_phase.py",
    "line": 1384,
    "class": "state",
    "severity": "S3",
    "claim": "The module's headline 'all values saved at consistent timestamps' invariant has at least six concrete exceptions, the sharpest being a final snapshot in which F_ion_in is freshly reconciled while press_HII_in -- the very pressure it is derived from -- is not.",
    "evidence": "Lens A: the reconciliation block writes R1, Pb, P_HII, F_HII, F_grav, F_ion_in, F_ram, F_rad, P_drive, P_ram (:1372-1393) but never n_IF, R_IF, press_HII_in, F_ram_wind, F_ram_SN, shell_mass, shell_massDot, c_sound, bubble_Leak, cool_beta or cool_delta -- all of which the in-loop block at :1002-1009 does write. Lens B (FLAG-10/B-08) independently enumerates five documented violations of the same invariant: beta/delta from the segment start on event termination and on the final append; the leak's one-segment-frozen Pb/c_sound; bubble_props held at the last physical segment on a no-root segment; the termination check mixing post-ODE Lgain with pre-ODE Lloss; and the energy-collapsed snapshot persisting the last healthy derived state against a new t.",
    "expected": "The reconciliation mirrors the in-loop write set (or declares which fields are intentionally left stale), and the docstring invariant is stated with its exceptions -- or a per-row consistency flag is emitted.",
    "failure_scenario": "The final dictionary.jsonl row fails F_ion_in == press_HII_in * 4*pi*R2^2, breaking any consistency check or plot that reconstructs the force budget from the recorded pressures; more generally, analysis code assuming simultaneity computes a force balance from a row that mixes two times.",
    "repro": "On the last output row assert abs(F_ion_in - press_HII_in*4*pi*R2**2) < tol; it fails whenever the shell structure changed over the last segment. Force a no-root segment and check whether the row's bubble_props correspond to t_now.",
    "confidence": "high",
    "lenses": ["A", "B"],
    "divergence": "AB",
    "status": "corroborated",
    "source_ids": ["S5b-A-18", "S5b-B-08"]
  },
  {
    "id": "S5b-R-24",
    "file": "trinity/phase1b_energy_implicit/run_energy_implicit_phase.py",
    "line": 1198,
    "class": "numerical",
    "severity": "S3",
    "claim": "The segment controller measures max_dex_change on the COMPLETED segment and applies the shrink to the NEXT one, so the first violating segment is always accepted with unbounded operator-splitting error.",
    "evidence": "Lens A: values_after and max_dex_change are computed at :1198-1202 and fed to next_dt_segment for the following segment; nothing rejects or re-integrates the segment just accepted, and the solver-failure paths (:1080, :1085) do not retry either. Lens C (C-31): 'accept-and-shrink lags the controller by one segment ... near the transition Lcool grows superexponentially, so the first violating segment is precisely the decisive one.'",
    "expected": "Reject-and-retry the violating segment at the smaller dt, or a documented bound on the accepted error.",
    "failure_scenario": "The reported transition time is set by the one segment the controller failed to resolve.",
    "repro": "Halve ADAPTIVE_THRESHOLD_DEX and re-run in a separate process at matched t; the transition time should move by less than the quoted precision.",
    "confidence": "medium",
    "lenses": ["A", "C"],
    "divergence": "AC",
    "status": "corroborated",
    "source_ids": ["S5b-C-31", "S5b-A-04"]
  },
  {
    "id": "S5b-R-25",
    "file": "trinity/phase1b_energy_implicit/run_energy_implicit_phase.py",
    "line": 173,
    "class": "numerical",
    "severity": "S3",
    "claim": "ODE_MAX_STEP = DT_SEGMENT_MIN/5 = 2e-5 Myr both over- and under-shoots its stated purpose: the comment's '>=5 steps per segment' guarantee fails at the 5e-5 Myr collapse clamp (<=3 steps), while at dt_segment = 5e-2 Myr it forces >=2500 internal steps, so dt_segment does not control integration accuracy at all.",
    "evidence": "Lens B (B-02), from prose: ':173 claims max_step ensures >=5 steps per segment; :145 sets the collapse segment duration to 50 yr = 5e-5 Myr, admitting at most 2.5 max-steps'. Lens A measured both ends: ODE_MAX_STEP = 2e-5, DT_SEGMENT_MAX = 5e-2, ratio 2500, each internal step calling get_ODE_implicit_pure -> get_ODE_Edot_pure with its third component discarded.",
    "expected": "Tie max_step to the active dt_segment (e.g. dt_segment/20) or let LSODA choose, and describe dt_segment as the beta-delta re-solve cadence rather than an accuracy control; qualify or correct the '>=5 steps' comment.",
    "failure_scenario": "No wrong number, but the stiffest regime (rapid collapse) gets the least-resolved integration while frozen Edot/Tdot span the whole segment; and up to 5000 segments x 2500 steps = 1.25e7 RHS evaluations, a third of each thrown away.",
    "repro": "Instrument sol.t.size per segment during a collapsing run and at dt_segment=5e-2; compare against the comment's claim.",
    "confidence": "high",
    "lenses": ["A", "B"],
    "divergence": "AB",
    "status": "corroborated",
    "source_ids": ["S5b-B-02", "S5b-A-27", "S5b-A-28"]
  },
  {
    "id": "S5b-R-26",
    "file": "trinity/phase1b_energy_implicit/run_energy_implicit_phase.py",
    "line": 529,
    "class": "divergence",
    "severity": "S3",
    "claim": "P_drive = max(Pb, P_HII) is not energy-conservative: when P_HII wins, the shell receives 4*pi*R2^2*P_HII*v2 of work while the bubble loses only 4*pi*R2^2*Pb*v2, and nothing in the module documents or accounts for the difference.",
    "evidence": "Lens C (C-40, SPEC-035 trap ii / SPEC-023 Reading A) states the imbalance and that it must be documented or accounted. Lens A confirms the construct exists (:529 P_drive = max(Pb, P_HII)) and that P_drive is written to params every segment. Lens B, which read every comment in the file, records no statement about it -- so it is indeed undocumented.",
    "expected": "An explicit note plus a diagnostic of the work imbalance, or an accounting term.",
    "failure_scenario": "Silent creation of mechanical energy whenever the photoionised branch wins -- i.e. exactly in the regime TRINITY presents as its advance over WARPFIELD.",
    "repro": "Integrate 4*pi*R2^2*(P_drive - Pb)*v2 dt over the energy phase and compare with the total injected energy.",
    "confidence": "medium",
    "lenses": ["C"],
    "divergence": "AC",
    "status": "single-lens",
    "source_ids": ["S5b-C-40"]
  },
  {
    "id": "S5b-R-27",
    "file": "trinity/phase1b_energy_implicit/run_energy_implicit_phase.py",
    "line": 1168,
    "class": "coefficient",
    "severity": "S3",
    "claim": "ENERGY_HANDOFF_FLOOR = 1e3 AU = 1.901e46 erg is a hard-coded absolute clamp applied AFTER Eb has gone non-positive; the discarded magnitude is never logged, and the constant is unit-less in prose.",
    "evidence": "Lens A: Eb = ENERGY_HANDOFF_FLOOR at :1168 and params['Eb'] at :1169, on the 'momentum' branch; using E_cgs2au from unit_conversions.py, 1 AU energy = 1.901e43 erg. Because Ed is frozen over the segment, the value being replaced is unbounded below. Lens B (F18/U5/B-17): the floor is documented as matching phase1c's ENERGY_FLOOR with NO unit stated, on a codebase whose own CLAUDE.md names units as a recurring bug class. Lens C (C-17) argues the floor should be relative (eps*L_gain*t or eps*Eb_peak) because Eb spans ~6 decades across the shipped sweep grid. NOTE: C's failure scenario ('an absolute floor fires immediately at the low-mass end') is DROPPED -- A shows the floor is never a trigger.",
    "expected": "Document the floor's units and its validity range against the smallest cluster in the shipped grid; log the magnitude of the negative Eb it replaces; and confirm it equals phase1c's ENERGY_FLOOR in the same units.",
    "failure_scenario": "At the low-mass end of param/paperII_grid_sweep.param, 1.9e46 erg may be a non-negligible fraction of the bubble's own peak energy, so the post-loop reconciliation computes Pb_f from an injected energy rather than from the state the integrator reached -- and the size of the overshoot is invisible.",
    "repro": "Record Eb at :1128 and at :1169 for a run ending in 'energy_to_momentum' and report the ratio; compare 1e3 AU against Eb_peak for the smallest and largest sweep cells.",
    "confidence": "medium",
    "lenses": ["A", "B", "C"],
    "divergence": "ABC",
    "status": "corroborated",
    "source_ids": ["S5b-A-07", "S5b-B-17", "S5b-C-17"]
  },
  {
    "id": "S5b-R-28",
    "file": "trinity/phase1b_energy_implicit/run_energy_implicit_phase.py",
    "line": 993,
    "class": "state",
    "severity": "S3",
    "claim": "The ODE-carried bubble temperature T0 (advanced by the frozen Td = delta*T/t) is never compared against the T the bubble-structure solve independently returns at the same state; a systematic drift would mean the conduction closure and the integrated T have decoupled.",
    "evidence": "Lens C (C-41): T is advanced as a bookkeeping variable while the structure solve independently determines T at xi = bubble_xi_Tb (SPEC-040/042). Lens A confirms both quantities exist -- T0 is state component 3 advanced linearly by Td from :993, and the residual bookkeeping at :910-930 records residual_T1_guess/residual_T2_guess -- but no comparison or reconciliation appears. Lens B's N14 confirms BetaDeltaResult carries T_bubble and T0 as raw diagnostics.",
    "expected": "|T_ode - T_solve|/T_solve small and non-drifting through the phase, checked and recorded.",
    "failure_scenario": "The cooling rate is evaluated at a temperature the structure does not support; Lcool is then wrong by the Lambda(T) sensitivity, which feeds the transition trigger directly.",
    "repro": "Log both T values per segment and plot their ratio over a full energy phase; commit the CSV under docs/dev/data/.",
    "confidence": "medium",
    "lenses": ["C"],
    "divergence": "none",
    "status": "single-lens",
    "source_ids": ["S5b-C-41"]
  },
  {
    "id": "S5b-R-29",
    "file": "trinity/phase1b_energy_implicit/run_energy_implicit_phase.py",
    "line": 521,
    "class": "units",
    "severity": "S3",
    "claim": "The two lenses that speak to PISM's storage convention disagree on its length unit: Lens A's dimensional analysis requires n*T in AU (pc^-3 K) for P_ext += PISM*k_B to balance, while Lens C reports the .param declares PISM as P/k_B in K cm^-3.",
    "evidence": "Lens A: ':521 P_ext += PISM * k_B ... balances only if PISM is stored as n*T in pc^-3 K; if it is stored as a pressure it is off by k_B'. Lens C (§2.2): 'PISM is declared as P/k_B in K cm^-3 (SPEC-092.4) -- must be multiplied by k_B somewhere upstream'. The two AGREE that the k_B multiply belongs there; they diverge on whether a cm^-3 -> pc^-3 conversion happens at ingestion. Neither lens could see the .param schema. Separately, A's original concern that the ISM term lacks the mu_convert/mu_ion factor its sibling carries is DEMOTED: C's account (PISM is a total-particle P/k_B while the cloud term converts n_H -> n_tot) makes the asymmetry expected.",
    "expected": "PISM converted to AU number density at .param ingestion, verified by a case in test/test_conventional_units.py.",
    "failure_scenario": "If no conversion happens, the ISM contribution to P_ext is wrong by (pc/cm)^3 ~ 3e55 -- which would fail loudly, so the likely truth is that it is fine and simply unverified. This only bites once rShell >= rCloud, i.e. post-blowout.",
    "repro": "Compare params['PISM'] * k_B against (mu_convert/mu_ion_shell) * nISM * k_B * T_ISM for the default schema values; they should agree to within the assumed ISM temperature.",
    "confidence": "low",
    "lenses": ["A", "C"],
    "divergence": "AC",
    "status": "contested",
    "source_ids": ["S5b-A-12", "S5b-C-15"]
  },
  {
    "id": "S5b-R-30",
    "file": "trinity/phase1b_energy_implicit/run_energy_implicit_phase.py",
    "line": 662,
    "class": "units",
    "severity": "S3",
    "claim": "Whether the t used for alpha = v2*t/R2 (and hence for beta, delta and dT/dt = delta*T/t) is the global cluster age or a phase-local clock is unresolved: the formula's FORM is agreed, its clock origin is asserted by nobody.",
    "evidence": "Lens A: params['cool_alpha'] = t_now * v2 / R2 at :662 and again every segment at :798 -- the form Lens C requires. Lens C (C-11): alpha, beta, delta and dT/dt = delta*T/t must all use the same GLOBAL cluster-age clock; a phase-local t makes alpha start at 0 and grow, corrupting the (beta,delta) root-find and hence Lcool and the transition time, with no visible error. Lens B says only 'update cool_alpha to match ODE-evolved v2 (preserves ODE continuity)' with no relation given (FLAG-16). Nothing in any lens establishes what params['t_now'] is anchored to.",
    "expected": "One t, threaded from the SPS clock; asserted in a test.",
    "failure_scenario": "If t_now were phase-local, alpha would start near zero at phase entry instead of relaxing to 3/(5-w), silently corrupting the closure.",
    "repro": "Extract alpha from snapshots; it must relax to 3/(5-w) = 0.6 for a uniform cloud, not start near zero at the phase start. Trace params['t_now'] to its origin.",
    "confidence": "medium",
    "lenses": ["C"],
    "divergence": "none",
    "status": "single-lens",
    "source_ids": ["S5b-C-11"]
  },
  {
    "id": "S5b-R-31",
    "file": "trinity/phase1b_energy_implicit/run_energy_implicit_phase.py",
    "line": 559,
    "class": "other",
    "severity": "S4",
    "claim": "'F_ram' names a bubble-pressure drive force (Pb*4*pi*R2^2), not a ram pressure, while the neighbouring comment says 'no ram pressure in implicit phase' and P_ram is hard-coded 0.0 -- so the output key F_ram means the opposite of what a consumer would assume.",
    "evidence": "Lens B (FLAG-05/B-04): :449 and :538 label a ram-pressure force computed from the bubble pressure and :8 lists F_ram among snapshot forces, while :559 asserts there is none in this phase. Lens A resolves it: F_ram = Pb * FOUR_PI * R2**2 at :539 ('a bubble-pressure force, not a ram pressure') and P_ram is passed hard-coded 0.0 at :559, written to params every segment and again at :1393, so params['P_ram'] is identically zero. Lens C (C-05) requires that the wind/SN ram pressures NOT enter P_drive in the energy phase -- which the hard-coded 0.0 is consistent with.",
    "expected": "Rename the field to reflect what it holds (a bubble-pressure drive force), or document the naming; drop the identically-zero P_ram or explain why it is persisted.",
    "failure_scenario": "A downstream force-budget consumer reads F_ram as a ram-pressure contribution and double-counts it against Pb, or reads P_ram == 0 as evidence that ram pressure is negligible rather than not modelled.",
    "repro": "Inspect the F_ram assignment in compute_forces_pure and the value of F_ram/P_ram in a dictionary.jsonl row from a phase-1b segment.",
    "confidence": "high",
    "lenses": ["A", "B"],
    "divergence": "AB",
    "status": "corroborated",
    "source_ids": ["S5b-B-04", "S5b-A-21"]
  },
  {
    "id": "S5b-R-32",
    "file": "trinity/phase1b_energy_implicit/run_energy_implicit_phase.py",
    "line": 212,
    "class": "other",
    "severity": "S4",
    "claim": "_inflow_frac_thickness returns the radial BRACKET of the inflowing zones, not their measure: a single inflowing cell scores exactly 0.0 and outflowing cells lying between the innermost and outermost inflowing cells are counted as inflow.",
    "evidence": "Lens A: :229-230 rneg = r[v<0]; return abs(rneg.max() - rneg.min())/rspan -- with one negative cell max == min so the result is 0.0. Lens C (C-18) independently specifies the correct form: sum(|dr| over inflow intervals)/(r_max - r_min), thickness-weighted rather than count-weighted, guarded for len<2, zero span, non-monotonic r and NaN. Severity held at S4 because Lens A (never read in this module; written to params['v_neg_frac_thick'] at :898) and Lens B (R6: 'NOT used in any physics') agree it is diagnostic only.",
    "expected": "Sum the widths of the inflowing intervals, per C-18; and state the frame convention (v<0 lab-frame vs v<v2 contact-discontinuity frame, C-19 -- unaddressed by A and B).",
    "failure_scenario": "The diagnostic reports 0.0 at the onset of inflow, which is exactly when it is wanted, and over-reports once inflow becomes patchy. No physics consequence today.",
    "repro": "assert _inflow_frac_thickness([1.0, -1.0, 1.0], [0.0, 0.5, 1.0]) > 0 -- it returns 0.0.",
    "confidence": "high",
    "lenses": ["A", "C"],
    "divergence": "AC",
    "status": "corroborated",
    "source_ids": ["S5b-A-32", "S5b-C-18", "S5b-C-19"]
  },
  {
    "id": "S5b-R-33",
    "file": "trinity/phase1b_energy_implicit/run_energy_implicit_phase.py",
    "line": 393,
    "class": "numerical",
    "severity": "S4",
    "claim": "The dt controller grows and shrinks on the SAME 0.05-dex threshold with no hysteresis, and its scale factor (10^0.1 ~ 1.26, i.e. 0.1 dex) is twice that deadband.",
    "evidence": "Lens A: the branches are 'Ddex > 0.05 -> dt/1.25893' and 'Ddex <= 0.05 -> dt*1.25893' -- one threshold, both directions. Lens B (B-23) does the arithmetic from prose: the gain exceeds the deadband that authorised it, admitting a limit cycle. Lens C (C-29) requires a lower grow threshold than the shrink threshold. DEMOTED from B's own S4/low: Lens A's S5b-R-16 shows max_dex_change is ~0 on 30 of 35 keys, so dt almost certainly pins at DT_SEGMENT_MAX rather than oscillating -- the limit cycle is unlikely until R-16 is fixed.",
    "expected": "Grow only below ADAPTIVE_THRESHOLD_DEX/m for some m > 1.",
    "failure_scenario": "After R-16 is fixed, shrink/grow limit cycling in dt_segment inflates the segment count without improving accuracy and injects a saw-tooth into the operator-splitting error.",
    "repro": "Log dt_segment and max_dex_change per segment on a steady stretch and look for a period-2 oscillation.",
    "confidence": "low",
    "lenses": ["B", "C"],
    "divergence": "BC",
    "status": "corroborated",
    "source_ids": ["S5b-B-23", "S5b-C-29"]
  },
  {
    "id": "S5b-R-34",
    "file": "trinity/phase1b_energy_implicit/run_energy_implicit_phase.py",
    "line": 699,
    "class": "deadcode",
    "severity": "S4",
    "claim": "A cluster of unreachable branches, unused values and duplicated computation: n_estimate, cooling_balance_factory, nISM, k_blowout's non-default, six unused imports, the 'unknown' termination path and its warning selector, both `tmax is not None` guards, a duplicated F_HII computation, a duplicated get_mass_profile call, a stale section banner, and the discarded third RHS component.",
    "evidence": "Lens A: n_estimate (:699) and cooling_balance_factory (:752) never referenced; nISM (:498) read and unused; imports scipy.optimize (:59), Dict/Optional/Tuple (:61), cvt (:66), ODEResult (:78), compute_derived_quantities (:79), BetaDeltaResult (:87) unreferenced; termination_reason == 'unknown' (:1410) unreachable because :1309 and :1040 already break on the time condition, so only segment_count >= MAX_SEGMENTS can end the loop; the `tmax is not None` guards at :1040/:1309 can never be False since :670 would have raised TypeError; F_HII computed identically at :985-986 and :536 with the first write overwritten at :998; get_mass_profile called at :1191 and again at :962 of the next iteration with the same R2, making the :964 ratchet a guaranteed no-op; :620-624 discards the third component of get_ODE_Edot_pure on every RHS call. Lens B (B-22) independently flags the stale 'Force Properties Dataclass' banner at :206-208 (the dataclass is at :439-451). Note B's A16 treats the unreachable 'unknown' path as a live 'real bug surface'.",
    "expected": "Delete or wire up; correct the banner. Note ruff's configured rule set here is F821/F811/F823/E9 only, so F401/F841 do not currently flag these -- per CLAUDE.md do not widen the rule set to clean up.",
    "failure_scenario": "",
    "repro": "Static; run ruff F401/F841 over this file locally without committing the config change.",
    "confidence": "high",
    "lenses": ["A", "B"],
    "divergence": "AB",
    "status": "corroborated",
    "source_ids": ["S5b-A-19", "S5b-A-20", "S5b-A-21", "S5b-A-22", "S5b-A-23", "S5b-A-28", "S5b-B-22"]
  },
  {
    "id": "S5b-R-35",
    "file": "trinity/phase1b_energy_implicit/run_energy_implicit_phase.py",
    "line": 943,
    "class": "silent-failure",
    "severity": "S4",
    "claim": "bubble_Tavg falls back to a hard-coded 1e6 K on any falsy value including a legitimately computed 0.0, while NaN is truthy and passes straight through into the sound speed.",
    "evidence": "Lens A: :943 `bubble_Tavg = params['bubble_Tavg'].value if params['bubble_Tavg'].value else 1e6`; Python truthiness makes 0.0 and None falsy and NaN truthy. Neither B nor C addresses this fallback.",
    "expected": "`if ... is not None` plus an explicit finiteness/positivity check.",
    "failure_scenario": "A degenerate bubble returning Tavg = 0.0 gets c_sound for a 1e6 K plasma, which then feeds get_leak_luminosity on the next segment; a NaN Tavg makes c_sound NaN silently for the rest of the phase.",
    "repro": "assert np.isfinite(params['c_sound'].value) and params['c_sound'].value > 0 after :944 on every segment.",
    "confidence": "medium",
    "lenses": ["A"],
    "divergence": "none",
    "status": "single-lens",
    "source_ids": ["S5b-A-25"]
  },
  {
    "id": "S5b-R-36",
    "file": "trinity/phase1b_energy_implicit/run_energy_implicit_phase.py",
    "line": 1026,
    "class": "other",
    "severity": "S4",
    "claim": "stop_at_rCloud_nSnap is off by one: the post-loop reconciliation snapshot past rCloud is written but never counted.",
    "evidence": "Lens A: the counter is incremented only inside the loop at :1023-1026, gated on params.save_count increasing, and the break at :768-775 happens before that segment's save -- but the post-loop reconciliation calls params.save_snapshot() at :1394 unconditionally, adding one more row while R2 > rCloud. Lens B documents the intended semantics (:761-763 'break BEFORE this iteration's save_snapshot fires'; :1019-1022 the counter increments only when the save actually wrote) without noting the reconciliation save.",
    "expected": "Exactly stop_at_rCloud_nSnap snapshots beyond rCloud in the output.",
    "failure_scenario": "With stop_at_rCloud_nSnap=1 the output contains 2 rows with R2 > rCloud.",
    "repro": "Run with stop_at_rCloud_nSnap=1 and count output rows with R2 > rCloud; the count is 2.",
    "confidence": "medium",
    "lenses": ["A"],
    "divergence": "none",
    "status": "single-lens",
    "source_ids": ["S5b-A-29"]
  },
  {
    "id": "S5b-R-37",
    "file": "trinity/phase1b_energy_implicit/run_energy_implicit_phase.py",
    "line": 1341,
    "class": "state",
    "severity": "S4",
    "claim": "On the no_physical_root_handoff exit the returned beta/delta arrays and params['cool_beta']/['cool_delta'] disagree, and the residual bookkeeping in the final snapshot is one segment stale.",
    "evidence": "Lens A: the break at :879 is upstream of :885-886, so params keeps the previous segment's beta/delta, while the trailing append at :1341-1349 uses the local beta/delta reassigned from the failed solve at :832-833; the same break skips the residual bookkeeping at :910-930, so betadelta_converged, residual_deltaT and residual_betaEdot are stale in the final row. Lens B's A6 documents only the benign case ('final append: beta/delta are from the last pre-ODE computation, best available').",
    "expected": "The results arrays and params agree at exit, or the local beta/delta are restored before the break.",
    "failure_scenario": "results.beta[-1] differs from the cool_beta recorded in the final dictionary.jsonl row on any run ending with no_physical_root_handoff.",
    "repro": "Run to a no_physical_root_handoff exit and compare results.beta[-1] with params['cool_beta'].value.",
    "confidence": "high",
    "lenses": ["A"],
    "divergence": "none",
    "status": "single-lens",
    "source_ids": ["S5b-A-34"]
  },
  {
    "id": "S5b-R-38",
    "file": "trinity/phase1b_energy_implicit/run_energy_implicit_phase.py",
    "line": 1251,
    "class": "other",
    "severity": "S4",
    "claim": "Object truthiness is used as a presence test for Param wrappers in at least six places, so a falsy wrapper (or a bare float stored instead of a wrapper) silently takes the default branch.",
    "evidence": "Lens A: :500 (PISM -> 0.0), :955 and :1186 (isCollapse -> False), :1241 (bubble_Leak -> 0.0), :1246 (bubble_Lloss -> 0.0), :1251 (phaseSwitch_LlossLgain -> literal 0.05) all use `if obj and hasattr(obj,'value')`. Speculative: it only bites if Param defines __bool__/__len__ in terms of its value, which no lens establishes.",
    "expected": "`if obj is not None and hasattr(obj, 'value')`.",
    "failure_scenario": "If Param ever gains a value-based __bool__, a stored phaseSwitch_LlossLgain of 0 falls back to the literal 0.05 -- changing the cooling_balance termination time.",
    "repro": "Store a bare float (not a Param) under 'PISM' and confirm compute_forces_pure silently uses PISM = 0.0.",
    "confidence": "low",
    "lenses": ["A"],
    "divergence": "none",
    "status": "single-lens",
    "source_ids": ["S5b-A-35"]
  },
  {
    "id": "S5b-R-39",
    "file": "trinity/phase1b_energy_implicit/run_energy_implicit_phase.py",
    "line": 1067,
    "class": "state",
    "severity": "S4",
    "claim": "The RHS lambda closes over the LIVE mutable params object beside a deliberately frozen ODESnapshot, so the '_pure' contract holds only by the accident that nothing writes params during the solve.",
    "evidence": "Lens A: :1067 `lambda t, y: get_ODE_implicit_pure(t, y, snapshot, params, Ed, Td)` -- snapshot is frozen at :1051 by create_ODE_snapshot, but the fourth argument (params_for_feedback) is the live dict; A finds no write between :1067 and :1079, so it is benign today. Lens B records the contract the code claims: 'Pure ODE functions: No dictionary mutations during integration' (:8). Lens C (C-39): purity is an explicit contract that must be testable by calling twice in a permuted order, and CLAUDE.md rule 5 records that trinity leaks module-level state in-process.",
    "expected": "Pass a frozen feedback view, matching the snapshot's intent; add the permuted-call purity test C-39 specifies.",
    "failure_scenario": "Benign today; if get_ODE_Edot_pure ever writes to params, or solve_ivp is driven concurrently, the segment's derivative becomes path-dependent and reproducibility across runs is lost.",
    "repro": "Call get_ODE_implicit_pure twice with shuffled ordering and other calls interleaved; assert bitwise-identical outputs and unchanged input objects.",
    "confidence": "low",
    "lenses": ["A", "C"],
    "divergence": "AC",
    "status": "corroborated",
    "source_ids": ["S5b-A-36", "S5b-C-39"]
  },
  {
    "id": "S5b-R-40",
    "file": "trinity/phase1b_energy_implicit/run_energy_implicit_phase.py",
    "line": 150,
    "class": "deadcode",
    "severity": "S4",
    "claim": "F_ISM appears in ADAPTIVE_MONITOR_KEYS but is never written anywhere in this module, and the module docstring's snapshot force list ('F_grav, F_ram, F_ion, F_rad') omits the HII-pressure force that is computed every segment.",
    "evidence": "Lens A: F_ISM is one of the 35 ADAPTIVE_MONITOR_KEYS (:150-167) and 'is never written anywhere in this module'; F_HII is written at :986 and :998 and again in the reconciliation at :1382. Lens B (FLAG-20/B-18): the docstring lists four forces while ForceProperties enumerates five, the extra being the outward HII pressure force from n_IF_Str.",
    "expected": "Drop F_ISM from the monitor list or write it; update the docstring's force list to match the fields actually persisted.",
    "failure_scenario": "F_ISM contributes nothing to the dex controller regardless of S5b-R-16; and a reader trusts a stale force inventory when reconstructing the shell force budget from dictionary.jsonl.",
    "repro": "Compare ForceProperties fields and ADAPTIVE_MONITOR_KEYS against the keys actually written by save_snapshot in this phase.",
    "confidence": "high",
    "lenses": ["A", "B"],
    "divergence": "AB",
    "status": "corroborated",
    "source_ids": ["S5b-B-18", "S5b-A-03"]
  }
]
```
