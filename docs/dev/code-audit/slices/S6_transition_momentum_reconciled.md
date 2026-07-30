# S6 transition + momentum — reconciled

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

**Status (2026-07-30):** 📗 reconciled report for slice S6 — merges the three raw lens reports
(`S6_transition_momentum_lensA.md`, `…lensB.md`, `…lensC.md`). Reconciler saw **only** those three
files; no source was read. Every line reference below is inherited from a lens and is therefore
**doubly unverified**.

Files in slice:
- **T** = `trinity/phase1c_transition/run_transition_phase.py`
- **M** = `trinity/phase2_momentum/run_momentum_phase.py`

Lens roles: **A** = what the code does (stripped source). **B** = what the code claims (comments and
docstrings only). **C** = what it should be (signatures + physics spec, **no literature access**).

---

## 1. Coverage table — who spoke about what

`—` = the lens never mentioned the quantity. **Silence is not corroboration.**

| # | Quantity / behaviour | Lens A (code) | Lens B (prose) | Lens C (physics) | Verdict |
|---|---|---|---|---|---|
| 1 | Momentum RHS `dv2/dt` assembly (M:452) | Explicit budget: `4πR2²(P_HII+P_ram−P_ext)/m − ṁv/m − G(M★+m/2)/R2² + F_rad/m`; term-by-term double-count check passed | F11 (terms named, no equation) | §1.2 canonical budget | **A = C**, no defect |
| 2 | Sweep-up `Ṁv` appears once | Once, as `−ṁ⁰v2/m⁰`; no separate `4πR2²ρv2²` | — | C-01 (S1) demanded exactly once | **A refutes C-01** → dropped |
| 3 | Feedback ram counted once | `P_ram` only inside `P_drive`; `F_ram`/`F_ram_wind`/`F_ram_SN` are diagnostics, not summed | — | C-04 (S1) double-count trap | **A refutes C-04** → dropped |
| 4 | `F_grav = G m(M★ + m/2)/R2²` | Identical at T:284, M:222, M:418 | — | C-06 (S1) requires exactly this | **A = C** → dropped, verified correct |
| 5 | `FOUR_PI` | `4.0·π` at T:98 / M:90 | — | C-26 (S1) requires 4π in both | **A = C** → dropped, verified correct |
| 6 | `F_rad` form | `f_absW,tot·(L/c)·(1 + Σ_{τ/κ}·κ_IR)` — C's form (ii), single expression | F22 "direct + IR-trapped", no formula | C-08: form (ii) preferred; warns against `(L/c)(1+τ_IR)` + a separate direct term | **A = C's preferred form** → demoted to a verification item |
| 7 | Radiation present in momentum phase | Yes, `+F_rad⁰/m⁰` unless `isDissolved` | Yes | C-05 (S1) requires it | **A refutes C-05** → dropped |
| 8 | `c_sound` identity | `bubble_Tavg` (else literal 1e6 K) → `get_soundspeed` | Units pc/Myr (T:200) | C-13: must be the **hot-interior** c_s | **A = B = C** → demoted to verification |
| 9 | `dEb/dt = min(Ed_bal, Ed_sc)`, `Ed_sc = −Eb c_s/R2` | Exactly this (T:238–245) | F1/F2 identical | §2.3 infers the same drain form (medium conf.) | **A = B = C**, corroborated |
| 10 | Fallback when `c_sound ≤ 0` or `R2 ≤ 0` | `Ed_sc = 0.0` ⇒ `Ed = min(Ed_bal, 0)` | — | — | **single-lens (A)** → R-17 |
| 11 | Transition `dR2/dt`, `dv2/dt` | **Delegated unchanged** to `get_ODE_Edot_pure` (energy-phase RHS) | F4 only | C-24/SPEC-022: transition drive must be `max(Pb, P_HII+P_ram)` | **A ≠ C, pending** → R-07 |
| 12 | Diagnostic `P_drive` | T:331 `max(Pb, P_HII+P_ram)`; M:265/445 `P_HII+P_ram` | `P_drive` never defined in prose | SPEC-022 phase-aware form | **A = C** for the diagnostic path; prose gap |
| 13 | `Pb`, `R1` in momentum | `Pb := pRam(...)` (M:585, :667), `R1 := R2` (M:588) | F7/F8/F9 + explicit "workaround" admission | §2.4: `R1` must be `R2`; `Pb` dropped | **A = B**; C sanctions `R1=R2` only → R-06 |
| 14 | `Eb` at hand-off | Forced `0.0` at M:511 and every segment M:571 | F10, `= 0` vs `≈ 0` inconsistency (B-14) | §2.4: drop, **but record** the discarded `E_b` | **A = B ≠ C** (recording) → R-05 |
| 15 | `ENERGY_FLOOR` magnitude | `1e3` (AU energy = M☉pc²/Myr²) | T:97, no unit given | C-20: needs `≪ 8e7` AU | **A = C** → dropped, verified adequate |
| 16 | Phase-exit criteria | Both present: `ram_fraction > 0.9` (T:763) **and** `Eb < 1e3` (T:769) + `energy_floor` event | B-01: three comments say floor, two say ram | C-11: exit on the force/pressure comparison | **A = B** (both exist); doc contradiction → R-21 |
| 17 | `R1` vs `R2` while `Pb → 0` | `compute_R1_Pb` opaque; A notes "if `Pb_post < 0` the fraction can exceed 1" | — | C-10 (S1): `R1 = √(ṗ/4πPb)` diverges, `V_b < 0`, `Pb` flips sign | **A supplies regime evidence, C supplies the mechanism** → R-01 |
| 18 | `ṗ = 2L/v` | `pRam` opaque (out of slice) | — | C-03 (S1) | **single-lens, out of slice** → R-02 |
| 19 | `P_ext` construction | `(μ_c/μ_i)·n(rShell)·k_B·T_ion` (if FABSi<1) `+ PISM·k_B` (if rShell ≥ rCloud) | "outside-shell inward pressure", "ISM pressure beyond cloud" | Budget has only `P_ext = k_B·PISM`; §1.4 says cloud thermal/turbulent pressure is **not** represented | **A = B; C does not sanction the `n·k_B·T_ion` term** → scope note in R-18 |
| 20 | `PISM·k_B` unit balance | A-16: balances iff `PISM` is `P/k_B` (conf. **low**) | No unit stated | C-27: SPEC-003 declares `PISM` in K cm⁻³ | **A = C → not a defect**; residual: cm⁻³→pc⁻³ at ingestion → R-31 |
| 21 | `P_ext` at frozen `rShell⁰` × live `R2²` | A-06 | — | C-15/C-19 (frozen coefficients, local ρ) | **A = C** → R-08 |
| 22 | Frozen `mShell`, `mShell_dot` | Frozen in snapshot; `mShell_dot` never refreshed post-step in T | — | C-15: splitting error controlled only by the dex threshold | **A = C** → folded into R-03 |
| 23 | Adaptive dex controller | A-12: 26 of 30 monitored keys provably unchanged between the two samples | F12–F14, B-13 (dead monitor groups in M) | C-14/C-16: must cover every frozen fast-varying quantity incl. SPS drivers | **A = C**, B corroborates the dead groups → R-03 |
| 24 | `compute_max_dex_change` degenerate cases | `None`→skip; `==0`→**skip**; sign flip→literal `1.0` dex (refines) | F12 only | C-14 (S2): predicted NaN silently disabling refinement | **A refutes the NaN mechanism**; residual zero-skip → R-28 (contested) |
| 25 | `max_step` | `DT_SEGMENT_MIN/5 = 1e-3/5 = 2e-4` Myr | F16: comment says **2e-5 Myr** "(≥5 steps/segment)" | C-28: must be below SPS sampling | **A ≠ B by 10×** → R-11 |
| 26 | `min_step` | `1e-6` Myr, LSODA-only | Stated, LSODA-gated | C-28: `MIN_STEP > 0` is an accuracy hazard | **A ≠ C** → R-32 |
| 27 | Velocity dt override | Gated `v2 < 0`, then `|v2| > 150` / `> 50` pc/Myr | Constants say `|v2|` (B-05) | C-21: must compare **signed** `v2`, pc/Myr, `|EXT|>|COLL|` | **A = C; B stale** → R-27 (doc-drift) |
| 28 | `DT_COLLAPSE` = 5e-4 Myr | `5e-4` | F15 "0.5 kyr = 5e-4 Myr" | C-29: `≲1e-3–1e-4` | **A = B = C** → dropped |
| 29 | Shell-mass monotone clamp / `isCollapse` freeze | T:547-549 / M:616-618, one-way latch | F20 word-identical in both; B-16 two collapse definitions + 1-segment lag | C-02: `Ṁv` must vanish for `v2<0` | **A = B**; A's clamp incidentally satisfies C-02 → R-09, C-02 demoted |
| 30 | Length-1 array unwrap | Present M:610-613, M:780-781; **absent** T:545, T:702 | "# Handle array returns" at M:609 only — 1 of 4 sites | — | **A = B, twin asymmetry** → R-12 |
| 31 | Final reconciliation content | T recomputes forces (T:850-859); M does **not** (M:886-894) | A's comment enumerates "Pb, shell structure, forces"; B's does not | §2.4 requires shell structure recomputed | **A = B, twin asymmetry** → R-13 |
| 32 | Final-block exception handling | M:899-908 enriched; T:861-862 bare | B-09: documented only in M | — | **A = B, twin asymmetry** → R-25 |
| 33 | Solver-failure / `max_segments` / `unknown` exits | No `SimulationEndCode`, no `EndSimulationDirectly` (A-11) | "unknown = a real bug surface" admitted in both files | C-17 (S2): segment-budget exhaustion must be a distinct recorded reason | **A = B = C** → R-04 |
| 34 | `MAX_SEGMENTS` | **5000 (T) vs 10000 (M)** | Constants block documented as *word-identical* | C: cap exhaustion is a numerical fate | **code diverges under identical prose** → folded into R-04 |
| 35 | `t_diss_onset` | Function-local in both; resets at the boundary | B-19: persistence threshold never stated in prose | — | A single-lens for the reset; **A refutes B-19's "counts segments" scenario** (it is `stop_t_diss` in Myr) → R-15 |
| 36 | `T0` | Re-stamped every segment from a phase-entry local | — | §2.4: `T0` must be **dropped**, not carried | **A = C** → R-16 |
| 37 | Two force implementations per file | A: confirmed; three copies of `F_rad`; two `ForceProperties` classes | B-06, B-15 ("same as in energy_implicit") | C-25/SPEC-007 closure | **A = B = C** → R-14 |
| 38 | Snapshot fields unread by the RHS | 6 of 18, incl. `Lmech_total`/`v_mech_total` shadowed by a live lookup | B-07 ("Gate all HII pressure" documented only in M) | — | **A = B**; A explains the gate is applied upstream → R-14 + R-29 |
| 39 | Live feedback in the RHS vs frozen `F_rad` | Feedback live at M:407; `F_rad` computed in the snapshot builder M:340-345 | B-04 exactly | — | **A = B, corroborated** → R-14 |
| 40 | Fate criteria (`coll_r`, `stop_r`, dissolution, rCloud) | Transcribed: `R2 < coll_r` (gated on `isCollapse`), `R2 > stop_r`, `shell_nMax < nISM` for `stop_t_diss` | Enumerated, same set | C-22/C-23: bare radius thresholds are the *wrong* criteria; escape needs `v2 > v_esc` | **A supplies the fact, only C calls it wrong** → R-23, R-24 |
| 41 | Momentum asymptote `R ∝ t^{2/(4−w)}` | — | — | C-18 | **single-lens, a test not a defect** → resolved table |
| 42 | Literature citations | — | **Zero citations anywhere** (B §3) | C-30: any eq. number would be unverifiable anyway | **B = C** → R-30 |

---

## 2. Divergence table

| Pair | Item | Reading |
|---|---|---|
| **A ≠ B** | `max_step`: code `1e-3/5 = 2e-4` Myr, comment "2e-5 Myr" | Doc-drift, factor 10. C's requirement (`< SPS sampling`) is met either way, so the **comment is the defect** (S3). Neither lens could see this alone. → R-11 |
| **A ≠ B** | Velocity thresholds: comments say `\|v2\|`, code gates `v2 < 0` first | C sides with the code (signed comparison is correct) ⇒ **stale comment** (S4). B's failure scenario ("no refinement for fast outward motion") is *not* supported by C. → R-27 |
| **A ≠ B** | `run_phase_transition` docstring states pure sound-crossing decay; code computes `min(...)` | Doc-drift; the module and ODE docstrings are current. → R-20 |
| **A ≠ B** | Momentum ODE docstring: "params for density profile lookup"; body also does a live feedback lookup | Doc-drift, momentum docstring is the stale twin. → R-26 |
| **A ≠ B** | B-19 "persistence duration never stated"; A shows it is `params['stop_t_diss']` in Myr | **B's concern resolved by A** → demoted |
| **A ≠ B** | B-20 "0/0 → nan in the ram fraction"; A shows a `P_total > 0` guard | **B's concern refuted by A** → dropped |
| **A ≠ B** | B-18 "two energy-floor thresholds"; A shows both read `ENERGY_FLOOR = 1e3` | **B's concern refuted by A** → dropped |
| **A ≠ B** | A-19 "duplicate final snapshot"; B documents a duplicate guard inside `save_snapshot` | **A's concern mitigated by B** → demoted to S4/contested → R-34 |
| **A ≠ C** | Adaptive dex controller cannot observe any frozen quantity ⇒ the operator-splitting error C says is "controlled only by `ADAPTIVE_THRESHOLD_DEX`" is controlled by **nothing** | Highest-value numerical finding in the slice. → R-03 |
| **A ≠ C** | Transition `dv2/dt` delegates to the energy-phase RHS; SPEC-022 wants `max(Pb, P_HII + P_ram)` during transition | Pending on `get_ODE_Edot_pure` (out of slice). → R-07 |
| **A ≠ C** | `Eb` zeroed with no record of the discarded energy | C sanctions the drop, not the silent drop. → R-05 |
| **A ≠ C** | `T0` carried frozen through both phases; C requires it dropped | → R-16 |
| **A ≠ C** | `P_ext` at frozen `rShell⁰` inside a live-`R2²` product; C wants local `ρ(R2)` | → R-08 |
| **A ≠ C** | No event/segment boundary at the `r_cloud` density discontinuity | → R-22 |
| **A ≠ C** | Re-collapse fate on `R2 < coll_r`; escape on `R2 > r_cloud` with no `v_esc` test | C calls both "the common wrong criterion". Fate labelling is partly out of slice. → R-23, R-24 |
| **A ≠ C** | `ODE_MIN_STEP = 1e-6 > 0` | C: an accuracy hazard. → R-32 |
| **A ≠ C** (**C unsure**) | `R1 = √(ṗ/4πPb)` divergence — the code exits the transition only at `Pb < P_ram/9`, i.e. **well past** `Pb = P_ram` where `R1 = R2` | Whether it blows up depends on a guard inside `compute_R1_Pb` (out of slice). A independently contemplates `Pb_post < 0`. → R-01 |
| **A ≠ C** (**C unsure**) | `ṗ = 2L/v` vs `L/v` | Lives in `get_bubbleParams.pRam`, invisible to A. → R-02 |
| **B ≠ C** | Zero literature citations for the defining `min()` decay law, the Strömgren balance, and the IR-trapping model | C: nothing to mis-cite, but nothing to validate against either. → R-30 |
| **A = B = C** | `F_grav`, `FOUR_PI`, `F_rad` form, `Ed_sc = −Eb c_s/R2`, `ENERGY_FLOOR` magnitude, `DT_COLLAPSE`, hot-interior `c_sound`, single `Ṁv` term, single feedback-ram count, radiation retained in the momentum phase | **Verified correct** — six of C's eight S1/S2 physics traps are *not* present. See §5. |
| **A = B, C silent/unsanctioned** | `P_ext ⊃ (μ_c/μ_i)·n(rShell)·k_B·T_ion` — an inward confinement term with no counterpart in C's derived budget, gated by a hard step at `FABSi == 1.0` | **Scope creep** (a term C's spec does not contain) *plus* a discontinuity. → R-18 |
| **A = B, C sanctions only half** | `Pb := pRam` in the momentum phase, applied **before** `shell_structure_pure` reads it | C endorses `R1 := R2`; nothing sanctions feeding a substituted `Pb` into `nShell0 = Pb/(k_B T_ion)`. The substitution is ≥9× at a ram-dominated exit. → R-06 |

---

## 3. Twin-runner comparison (the copy-paste-asymmetry hunt)

Both lenses were asked to diff the two runners. A's structural verdict: *"Both files are the same
program with the state vector shortened by one component and the energy machinery deleted"*, with
T:93–176 ≡ M:86–168 byte-for-byte apart from `MAX_SEGMENTS` and `ENERGY_FLOOR`. B's verdict:
*"B is the later copy-edit in the doc-comment layer but the stale copy in the ODE-docstring layer …
edits have flowed in both directions and neither file is uniformly ahead."* **These two independent
verdicts agree**, which is itself the most important corroboration in this section.

### 3.1 Same quantity, both runners — what each lens sees

| Quantity | Lens A (code) | Lens B (prose) | Signature |
|---|---|---|---|
| `F_grav` | identical (T:284 ≡ M:222 ≡ M:418) | identical | agree |
| `P_ext` incl. `FABSi < 1.0` branch and `PISM·k_B` | identical (T:306/314 ≡ M:246/254 ≡ M:427/438) | identical wording (B uses "ISM"/"ambient" interchangeably) | agree |
| `F_ion_in`, `F_HII`, `P_HII` formula | identical | identical | agree |
| `F_rad` | identical — **three** copies (T:344, M:278, M:343) | identical wording, incl. the snapshot-builder copy at M:339 | agree (duplication risk → R-14) |
| Adaptive-dt controller + all constants | identical | word-identical | agree |
| Velocity dt override | identical | word-identical | agree |
| ODE tolerances | identical | word-identical (**both carry the same wrong `2e-5` comment**) | agree — R-11 hits both twins |
| Termination block | identical | word-identical | agree |
| **`F_ram`** | **divergent**: `Pb·4πR2²` (T:338) vs `P_ram·4πR2²` (M:272) | **divergent**: T:260 "Ram pressure force (**from bubble pressure**)" vs M:195 bare | **divergence visible in BOTH layers** — documented, not silent |
| **`P_drive`** | **divergent**: `max(Pb, P_HII+P_ram)` (T:331) vs `P_HII+P_ram` (M:265) | **no prose divergence** — `P_drive` is never defined in either file | code-only, but **sanctioned by C's SPEC-022**; documentation gap only |
| **`MAX_SEGMENTS`** | **divergent**: 5000 (T) vs 10000 (M) | **prose identical** — the constants block is listed as word-identical | ⚠️ **copy-paste-asymmetry signature**: an undocumented 2× divergence in a termination-affecting constant |
| **length-1 array unwrap** | **divergent**: present M:610-613 / M:780-781, absent T:545 / T:702 | **divergent**: "# Handle array returns" at M:609 only — and absent from **both** adaptive-stepping copies | ⚠️ **strongest twin asymmetry**: a hardening applied at **1 of 4 structurally identical sites** |
| **final reconciliation content** | **divergent**: T recomputes the force set (T:850-859), M does not (M:886-894) | **divergent**: T's comment enumerates "Pb, shell structure, **forces**", M's says only "derived properties" | ⚠️ asymmetry visible in both layers; **momentum is the outlier** |
| **final-block exception handling** | **divergent**: M:899-908 enriched (type + traceback frame), T:861-862 bare | **divergent**: documented at M:896-898, nothing in T | ⚠️ asymmetry visible in both layers; **transition is the outlier** |
| **ODE `params` docstring** | no code divergence — **both** RHS use `params` for live feedback | **divergent**: T:200 "feedback interpolation" vs M:375 "density profile lookup" | prose-only; **momentum docstring is stale** |
| **"Gate all HII pressure"** | no code divergence — `include_PHII` gates `P_HII` in the *runner* of both files; the snapshot copy is never read | **divergent**: documented only at M:316 | prose-only; **B-07's concern is refuted by A** |
| monitor group headings (`Cooling`, `Bubble`) | identical key list; 26/30 keys inert in **both** files | copied verbatim into M where the module docstring says `Eb = Ed = Td = 0` | agree on the fact; consequence → R-03 |
| `ENERGY_FLOOR` / `Eb` machinery | present T, absent M | present T, absent M | phase-appropriate, expected |

### 3.2 Ranked twin findings and which runner is the outlier

1. **Length-1 array unwrap — 1 of 4 sites (R-12).** A and B reach it independently, from code and
   from a single comment. This is the textbook copy-paste asymmetry: a defensive fix landed in the
   momentum primary block and nowhere else, while both adaptive-stepping copies are *documented* as
   applying "the **same** guards as the primary shell mass block above" — a contract that the guard's
   absence makes false. **Outlier: the transition runner** (both of its sites), plus the momentum
   runner's own secondary site.
2. **Final reconciliation drops the force recompute (R-13).** Both layers diverge in the same
   direction. **Outlier: the momentum runner** — its last output row pairs final `R2/v2/t` with
   forces from the previous segment (up to 5e-2 Myr stale).
3. **`MAX_SEGMENTS` 5000 vs 10000 under byte-identical prose (folded into R-04).** The only divergence
   A sees that B's prose layer is entirely blind to. Plausibly deliberate (the momentum phase runs
   longer), but nothing documents it, and C-17 makes cap exhaustion a *fate-relevant* outcome.
   **Outlier: ambiguous** — the caps differ, the comments do not.
4. **`F_ram` means two different things (R-10).** Seen by both lenses in both layers, so it is **not**
   a silent copy-paste asymmetry — the transition runner's comment (T:260 "from bubble pressure")
   actively documents the odd behaviour. It is still a cross-phase output defect: inside one
   `ForceProperties` object the transition runner's `F_ram` disagrees with its own `P_ram`, and the
   output column changes meaning at the phase boundary. **Outlier: the transition runner.**
5. **Exception-detail enrichment (R-25).** Both layers; **outlier: the transition runner**. S4.
6. **Momentum ODE docstring stale (R-26).** Prose-only; **outlier: the momentum runner**. S4.

**Direction of drift.** Items 1 and 5 put the momentum runner ahead; items 2 and 6 put the transition
runner ahead. Neither file is uniformly the newer copy — so *every* byte-identical block between them
should be treated as un-reviewed since the split, exactly as Lens B concluded independently.

---

## 4. Merge / dedupe map

| Reconciled | Absorbed lens findings |
|---|---|
| R-01 | C-10 (+ A §2.3 step 20 ram-threshold arithmetic, A's `Pb_post < 0` note) |
| R-02 | C-03 |
| R-03 | A-12, C-15, C-16, B-13 |
| R-04 | A-11, C-17, A §7 (`unknown`), A §0 (`MAX_SEGMENTS` 5000/10000), B admissions |
| R-05 | A-02, B-14, B-01 (consequence half), C-12 (recording half) |
| R-06 | A §4.7 item 2, B-10 (half), B F7/F8 |
| R-07 | A §2.1, C-24, C-11 |
| R-08 | A-06, C-15, C-19 (half) |
| R-09 | A-08, B-16 |
| R-10 | A-01, B-10 (half), C-25 |
| R-11 | A §6 inventory, B-11, B F16 |
| R-12 | A-03, B-08 |
| R-13 | A-05 |
| R-14 | A-07, B-06, B-04, B-15, C-25 |
| R-15 | A-09 |
| R-16 | A-10, C-12 (`T0` half) |
| R-17 | A-14 |
| R-18 | A-15 (+ C §1.4 scope note) |
| R-19 | A-13 |
| R-20 | B-02 |
| R-21 | B-01 (prose half), B-18 (residual) |
| R-22 | C-19 (half) |
| R-23 | C-22 |
| R-24 | C-23 |
| R-25 | A-04, B-09 |
| R-26 | B-03 |
| R-27 | B-05, C-21 |
| R-28 | C-14 (narrowed) |
| R-29 | A-17, A-18, A-07 (dead-field half), B-07, B-13 |
| R-30 | B-12, C-30 |
| R-31 | A-16, C-27 |
| R-32 | C-28 |
| R-33 | B-17 |
| R-34 | A-19 |
| R-35 | B-21 |

---

## 5. Dropped or demoted, with reasons (the filter did its job)

**Dropped outright — another lens plainly explains why the concern is unfounded:**

| Lens finding | Sev. claimed | Why dropped |
|---|---|---|
| C-01 "sweep-up ram double-counted" | S1 | A's term-by-term transcription shows `Ṁv` appears **once**, as `−ṁ⁰v2/m⁰`, and no `4πR2²ρv2²` term exists. |
| C-04 "feedback ram double-counted" | S1 | A: `P_ram` enters only inside `P_drive`; `F_ram`/`F_ram_wind`/`F_ram_SN` are diagnostics that are never summed into the RHS. |
| C-05 "radiation dropped in the momentum phase" | S1 | A: `+F_rad⁰/m⁰` is in the momentum RHS. |
| C-06 "gravity self-gravity factor" | S1 | A: `G·m·(M★ + 0.5·m)/R2²`, identical at all three sites — exactly C's expected form. |
| C-26 "`FOUR_PI` must be 4π in both" | S1 | A's literal inventory: `4.0·π` at T:98 and M:90. |
| C-20 "`ENERGY_FLOOR` must be ≪ 8e7 AU" | S2 | A: `ENERGY_FLOOR = 1e3` M☉pc²/Myr² ≈ 1.9e46 erg, five orders below C's bar; and hitting it does terminate the phase. |
| C-02 "`Ṁv` must vanish for `v2 < 0`" | S2 | A: the monotone clamp and the `isCollapse` freeze both set `mShell_dot = 0.0` whenever the enclosed mass would decrease, so the shed-mass regime C feared is already excluded. (Residual: the *frozen* `ṁ⁰ > 0` combined with a mid-segment sign flip in `v2` gives a term of the **opposite** sign to C's concern — it decelerates infall, it does not accelerate it.) |
| C-21 "velocity thresholds must be signed, pc/Myr" | S3 | A: gated on `v2 < 0`, values 50/150 pc/Myr, `|EXTREME| > |COLLAPSE|` — all three of C's requirements met. |
| C-29 "`DT_SEGMENT_MIN` ≲ 1e-3–1e-4 Myr" | S3 | A: `DT_SEGMENT_MIN = 1e-3`, `DT_COLLAPSE = 5e-4` (which bypasses the floor downward). Marginally satisfied. |
| B-20 "`0/0 → nan` in the ram fraction" | S4 | A: the ratio is guarded by `P_total > 0`. |
| B-18 "two different energy-floor thresholds" | S4 | A: both the event (`build_transition_phase_events(energy_floor=1e3)`) and the post-segment test read the same `ENERGY_FLOOR = 1e3`. |
| B-19 "dissolution timer may count segments" | S4 | A: the test is `(t_now − t_diss_onset) >= params['stop_t_diss'].value` — absolute Myr, configurable. (The *reset at the phase boundary*, a different defect, survives as R-15.) |
| C-13 "`c_sound` may be the shell sound speed" | S3 | A: `T_for_sound = bubble_Tavg` → hot interior. Demoted to the verification list below. |
| C-08 "radiation double-count" | S2 | A: one expression, `f_abs·(L/c)·(1 + τ_IR)` = C's preferred form (ii). Demoted to the verification list below. |
| C-18 "momentum asymptote `t^{2/(4−w)}`" | S3 | A validation test, not a divergence claim; no lens reports a conflicting exponent in the slice. |
| B-07 "HII gate on the integration path only" | S3 | A: `include_PHII` gates `P_HII` in the **runner** of both files; the snapshot's copy of the flag is never read by the RHS, so reported and integrated `P_HII` are the same gated value. Survives only as dead-field hygiene (R-29). |

**Demoted (kept, but at lower severity or with the failure scenario struck out):**

- **C-14 → R-28 (S2 → S4, contested).** C predicted a NaN silently disabling refinement. A shows
  explicit guards: `None` → skip, `== 0` → skip, sign flip → a literal `1.0` dex which **exceeds**
  the `0.1` threshold and therefore *refines* — C's fail-safe direction is already implemented. Only
  the `== 0` skip survives, and it matters mainly because the momentum phase pins `Eb ≡ 0.0`.
- **B-05 → R-27 (S3 → S4).** The prose/code mismatch is real, but C sides with the **code**: a signed
  comparison is what the physics wants. B's failure scenario (no refinement during fast outward
  motion) is a *different*, legitimate concern that belongs to R-03, not to the constants' comments.
- **A-19 → R-34 (S4, contested).** B documents a duplicate guard inside `save_snapshot` that plausibly
  suppresses the second write. The surviving residual is the `_snapshots_after_rCloud` counter not
  being bumped by the final-block save.
- **A-16 + C-27 → R-31 (S3/S2 → S4).** The two lenses independently converge on the *same* reading
  (`PISM` is `P/k_B` in K cm⁻³, so `PISM·k_B` is correct). Not a defect; a schema check.
- **B-11 → R-11 (S3, reframed).** B's "≥5 steps per segment is unachievable" horns mostly dissolve:
  A shows `max_step = DT_SEGMENT_MIN/5` **by construction**, so the invariant holds. What survives is
  that the comment states the wrong number by 10×, plus a genuine cost note (a 5e-2 Myr segment is
  forced to ≥250 solver steps).

---

## 6. Pending items only the literature (or an out-of-slice file) can settle

Lens C had no arXiv/ADS access and refused to assert any equation number. These are the items where
an A ≠ C divergence is a **question**, not a confirmed defect:

1. **`ṗ = 2L/v` vs `L/v` (R-02).** Lives in `get_bubbleParams.pRam`, invisible to A. C derives the
   factor 2 from `L = ½Ṁv²` (high confidence in the derivation, no source). *Resolved by:* reading
   `pRam`, plus the WARPFIELD/Rahner momentum-injection definition. **Highest-value single check in
   the slice** — a missing 2 halves the entire momentum-phase drive.
2. **`R1` guard and the sign of `(R2³ − R1³)` (R-01).** `compute_R1_Pb` is out of slice. C's geometry
   is prefactor-free and high confidence; A independently contemplates `Pb_post < 0`. *Resolved by:*
   reading `compute_R1_Pb`, or asserting `R1 ≤ R2` at every transition snapshot.
3. **Which `P_drive` the transition phase actually integrates (R-07).** `get_ODE_Edot_pure` is out of
   slice. If it is the energy-phase RHS unchanged, its `P_drive` is `max(Pb, P_HII)` per SPEC-022 and
   the transition phase never sees the ram term in its equation of motion — while its own diagnostic
   `P_drive` at T:331 *does* include it.
4. **Radiation form and `f_absW,tot` semantics (verification).** A's expression matches C's preferred
   form (ii) **iff** `shell_fAbsorbedWeightedTotal ≈ 1 − e^{−τ_UV}` and
   `shell_tauKappaRatio·dust_KappaIR ≈ κ_IR M_sh/(4πR2²)`. Both live in `shell_structure`.
5. **`c_sound` constants (verification).** A confirms the hot-interior temperature source; C's
   0.05–0.2 `t_dyn` transition-duration prediction is a cheap end-to-end test of `get_soundspeed`'s
   `γ` and `μ`. Note the `bubble_Tavg == 0` falsy fallback silently substitutes 1e6 K (a 3.2× slower
   drain than C's 1e7 K reference).
6. **`PISM` schema unit and the cm⁻³ → pc⁻³ ingestion conversion (R-31).**

---

## 7. Merged, ranked candidate findings

```json
[
  {
    "id": "S6-R-01",
    "file": "trinity/phase1c_transition/run_transition_phase.py",
    "line": 507,
    "class": "divergence",
    "severity": "S1",
    "claim": "The transition phase keeps evaluating R1 and Pb from compute_R1_Pb long after the geometric energy->momentum boundary. Its ram-dominance exit fires only at P_ram/(Pb+P_ram) > 0.9, i.e. Pb < P_ram/9, whereas R1 = R2 already at 4*pi*R2^2*Pb = pdot_w (Pb ~ P_ram). If R1 is derived from sqrt(pdot/(4*pi*Pb)) without a R1 <= R2 guard, the bubble volume (4pi/3)(R2^3 - R1^3) goes negative and Pb = Eb/[2pi(R2^3-R1^3)] flips sign inside the phase, giving a large INWARD 'thermal' force.",
    "evidence": "Lens C section 2.1 (pure geometry, high confidence): R1 = sqrt(pdot_w/(4 pi Pb)) equals R2 exactly when 4 pi R2^2 Pb = pdot_w. Lens A: run_transition_phase.py:507 'R1, Pb = compute_R1_Pb(R2, Eb, Lmech_total, v_mech_total, gamma_adia)' every segment, with compute_R1_Pb opaque to Lens A; the only exit that responds to the Pb/P_ram ordering is :749-763 with RAM_DOMINANCE_THRESHOLD = 0.9. Lens A independently records at :756-759 that 'if Pb_post < 0 the fraction can exceed 1', i.e. a negative Pb is already contemplated by the code's own guard.",
    "expected": "Either a guard R1 = min(R1, R2) inside compute_R1_Pb, or a transition exit at 4*pi*R2^2*Pb <= pdot_total + 4*pi*R2^2*P_HII (the threshold-free branch switch), so the phase never integrates in the R1 > R2 regime.",
    "failure_scenario": "Between Pb = P_ram and Pb = P_ram/9 the code integrates a regime where the bubble volume can be negative. A sign-flipped Pb produces an inward force of hot-bubble magnitude at exactly the stiffest moment of the run; LSODA either chatters to min_step or accepts an inverted force budget, and the recorded Pb column goes negative with no warning.",
    "repro": "Assert R1 <= R2 and (R2**3 - R1**3) > 0 at every transition snapshot in dictionary.jsonl for param/simple_cluster.param and docs/dev/performance/f1edge_hidens*.param; log Pb sign and ram_fraction through the last 20 transition segments.",
    "confidence": "medium",
    "lenses": ["A", "C"],
    "divergence": "AC",
    "status": "single-lens",
    "source_ids": ["S6-C-10", "S6-A-01(context)"]
  },
  {
    "id": "S6-R-02",
    "file": "trinity/phase2_momentum/run_momentum_phase.py",
    "line": 225,
    "class": "coefficient",
    "severity": "S1",
    "claim": "The momentum injection rate behind pRam must be pdot = 2*Lmech_total/v_mech_total, not Lmech_total/v_mech_total. A missing factor 2 halves the entire momentum-phase drive.",
    "evidence": "Lens C S6-C-03, derived from L = 0.5*Mdot*v^2 and pdot = Mdot*v (SPEC-071); C notes the signature takes exactly (Lmech_total, v_mech_total) because that conversion is its purpose. Lens A cannot see it: get_bubbleParams.pRam is listed among the opaque out-of-slice symbols, and A only records the weak consistency 'F_ram_wind + F_ram_SN = pdot_W + pdot_SN, which equals 4*pi*R^2*pRam under the usual pRam = pdot/(4*pi*R^2)'. Lens B records no formula for pRam anywhere.",
    "expected": "pRam(R2, L, v) == 2*L/(v*4*pi*R2**2); recorded pdot_total == 2*Lmech_total/v_mech_total at every snapshot.",
    "failure_scenario": "R2 low by 2^(1/4) = 19% and swept mass low by ~60% in a uniform medium, across the whole published grid; the dispersal-vs-recollapse boundary moves systematically.",
    "repro": "Read get_bubbleParams.pRam; then check recorded pdot_total == 2*Lmech_total/v_mech_total in dictionary.jsonl.",
    "confidence": "low",
    "lenses": ["C"],
    "divergence": "AC",
    "status": "single-lens",
    "source_ids": ["S6-C-03"]
  },
  {
    "id": "S6-R-03",
    "file": "trinity/phase1c_transition/run_transition_phase.py",
    "line": 619,
    "class": "numerical",
    "severity": "S2",
    "claim": "The adaptive-timestep controller is nearly inert: values_before and values_after bracket only the ODE step, so 26 of the 30 ADAPTIVE_MONITOR_KEYS are provably unchanged between the two samples and contribute exactly 0 dex. The operator-splitting error of the frozen snapshot - which Lens C shows is controlled ONLY by ADAPTIVE_THRESHOLD_DEX and not at all by ODE_RTOL - is therefore controlled by nothing.",
    "evidence": "Lens A: values_before at run_transition_phase.py:619 (after every derived-quantity update), values_after at :709 (before the next iteration recomputes anything); the only params writes in between are :684-687 (t_now, R2, v2, Eb) and :707 (shell_mass). Same structure at run_momentum_phase.py:701/:788 with writes at :762-764 and :786. 'F_ISM' is never written by either file. In the momentum phase Eb is pinned to 0.0 and skipped by the '== 0' guard. Lens C S6-C-15/S6-C-16: the frozen mShell/mShell_dot/shell_props/c_sound introduce a splitting error invisible to ODE_RTOL, and a monitor set omitting the SPS drivers steps over the SN turn-on with a stale pdot. Lens B S6-B-13 corroborates from prose that the momentum runner retains 'Cooling parameters' and 'Bubble properties' monitor groups it never evolves.",
    "expected": "Either sample values_after after the next segment's derived-quantity recomputation (so Lmech, pdot, Pb, the optical depths and P_HII can actually register), or trim the key list to what can change and add an explicit driver-based segment bound (e.g. no segment may span a >0.1 dex change in Lmech_total).",
    "failure_scenario": "A single segment straddles the SN turn-on (~3-5 Myr, ~1 dex in Lmech) and is integrated at up to DT_SEGMENT_MAX = 5e-2 Myr with pre-SN momentum injection frozen in the diagnostic path; the controller sees only R2, v2, Eb, shell_mass moving smoothly and keeps growing dt by 10^0.1 per segment. Tightening ODE_RTOL does not help, which makes the error invisible to a convergence test.",
    "repro": "Log which key attains the max in compute_max_dex_change over a transition run (it will only ever be R2, v2, Eb or shell_mass); plot segment boundaries against Lmech(t) across the SN onset.",
    "confidence": "high",
    "lenses": ["A", "B", "C"],
    "divergence": "AC",
    "status": "corroborated",
    "source_ids": ["S6-A-12", "S6-C-15", "S6-C-16", "S6-B-13"]
  },
  {
    "id": "S6-R-04",
    "file": "trinity/phase1c_transition/run_transition_phase.py",
    "line": 646,
    "class": "silent-failure",
    "severity": "S2",
    "claim": "Numerical terminations are not recorded. Solver exception, solver failure, MAX_SEGMENTS exhaustion and the 'unknown' fall-through all exit with only a local termination_reason string and no SimulationEndCode / SimulationEndReason / EndSimulationDirectly, so a failed or budget-exhausted integration is indistinguishable from a normal phase completion. MAX_SEGMENTS additionally differs between the twins (5000 transition vs 10000 momentum) with byte-identical surrounding comments.",
    "evidence": "Lens A S6-A-11: run_transition_phase.py:641-644 (exception) and :646-648 (not sol.success / empty sol.t) set only the local string and break, while every other exit in the same loop (:472-479, :606-609, :780-783, :791-793, :800-802, :816-818) sets all three params keys; identical at run_momentum_phase.py:723-730; same for the max_segments/'unknown' fall-through at :870 / :916. Lens B records both files' own admission that \"'unknown' means we fell through every known exit path - a real bug surface, not a routine completion\" (T:872-873, M:918-919). Lens C S6-C-17: SPEC-100/SPEC-105 require segment-budget exhaustion to be a distinct recorded outcome, never merged with a physical fate. Lens A section 0 records MAX_SEGMENTS = 5000 (T) vs 10000 (M); Lens B lists the constants block as word-identical, so nothing documents the 2x difference.",
    "expected": "A distinct SimulationEndCode for each of solver_error, solver_failed, max_segments and unknown, set on all exit paths and surfaced in metadata.json; and either the two MAX_SEGMENTS values are unified or the difference is documented.",
    "failure_scenario": "LSODA fails mid-transition or the segment budget runs out during a stiff re-collapse; R2/v2/Eb retain pre-segment values, the final reconciliation block writes a normal-looking snapshot, and the run is later classified by a downstream fate label. Published dispersal-vs-recollapse statistics then silently include numerical artefacts.",
    "repro": "Force sol.success=False (e.g. min_step > max_step) and inspect params['SimulationEndCode'] after run_phase_transition returns; separately set MAX_SEGMENTS small and confirm the recorded outcome is not a physical fate.",
    "confidence": "high",
    "lenses": ["A", "B", "C"],
    "divergence": "AC",
    "status": "corroborated",
    "source_ids": ["S6-A-11", "S6-C-17"]
  },
  {
    "id": "S6-R-05",
    "file": "trinity/phase2_momentum/run_momentum_phase.py",
    "line": 511,
    "class": "state",
    "severity": "S2",
    "claim": "The momentum phase forces Eb to exactly 0.0 at entry and again at every segment, with no record of the discarded energy - and because the transition phase can exit on ram-pressure dominance rather than on the energy floor, the discarded Eb need not be small.",
    "evidence": "Lens A S6-A-02: run_momentum_phase.py:511 \"params['Eb'].value = 0.0\" at entry and :571 inside the loop; the transition phase can break at :763 on ram_fraction > 0.9 with Eb arbitrarily far above ENERGY_FLOOR = 1e3, since the ram test at :756-759 does not test Eb at all. Lens B S6-B-14 records the prose alternating between 'Eb = 0' (:3 Key Feature 1, :506) and 'Eb ~= 0' (:3 Overview, :462), and S6-B-01 documents the two competing exit criteria. Lens C section 2.4 sanctions DROPPING Eb ('total energy is explicitly not conserved - E_b is declared radiated/vented') but requires that 'a correct implementation records the discarded E_b so a global energy audit closes'.",
    "expected": "Record the discarded Eb (a params key and an output column) at the hand-off, and/or require the ram-dominated exit to also satisfy Eb below the floor. The docstrings should stop alternating between '=' and '~='.",
    "failure_scenario": "A wind-dominated configuration crosses ram_fraction = 0.9 while the bubble is still energetic; the energy vanishes from the budget with no accounting entry, and any energy-conservation diagnostic across the full run shows an unexplained jump at the phase boundary.",
    "repro": "Record Eb at the last transition snapshot and at the first momentum snapshot in dictionary.jsonl for param/simple_cluster.param and docs/dev/performance/f1edge_hidens*.param; compare against ENERGY_FLOOR and against which exit fired.",
    "confidence": "high",
    "lenses": ["A", "B", "C"],
    "divergence": "AC",
    "status": "corroborated",
    "source_ids": ["S6-A-02", "S6-B-14", "S6-B-01", "S6-C-12"]
  },
  {
    "id": "S6-R-06",
    "file": "trinity/phase2_momentum/run_momentum_phase.py",
    "line": 585,
    "class": "state",
    "severity": "S2",
    "claim": "At momentum entry Pb is overwritten with pRam(R2, ...) BEFORE shell_structure_pure reads it, and the shell inner-edge density is nShell0 = Pb/(k_B*T_ion). Since the transition exits at Pb < P_ram/9, the substitution multiplies Pb by more than 9 at the boundary, so the shell density, optical depths, F_rad and P_HII all step discontinuously at a boundary where Lens C requires the mechanical state to be continuous.",
    "evidence": "Lens A section 4.7 item 2: \"R1 -> forced to R2 (:588); Pb -> forced to pRam (:585). Both are overwrites of physically different phase1c quantities, and both happen BEFORE shell_structure_pure reads them, so the shell structure at the first momentum segment sees a different Pb than the last transition segment did at the same R2\"; the ram exit at :749-763 fires at P_ram/(Pb+P_ram) > 0.9, i.e. Pb < P_ram/9. Lens B F7/F8 and the admissions table: \":581 nShell0 = Pb/(k_B*T_ion)\", \":582 Without this, Pb = 0 in momentum phase would give n_IF -> 0\" - an explicitly stated workaround, not a physical identity. Lens C section 2.4 requires R2, v2, M_sh and dv2/dt continuous, Pb/R1 dropped, and shell structure RECOMPUTED as an algebraic function of (R2, M_sh, Q_i, L_bol) - notably not of Pb.",
    "expected": "Either the shell inner-edge density in the momentum phase is derived from the ram pressure explicitly (with the substitution documented as a model, not as 'Pb'), or nShell0 uses a formulation that does not require a bubble pressure. Under no reading should the transition's last shell structure and the momentum's first differ by a factor of ~9 at the same R2.",
    "failure_scenario": "n_shell0 jumps by >9x at the phase boundary, propagating into tau_UV/tau_IR, f_abs, F_rad and n_IF -> P_HII, so the drive itself is discontinuous. This defeats the whole purpose of the finite-duration transition phase (Lens C section 2.2), which exists precisely to make dv2/dt continuous into the momentum phase.",
    "repro": "Diff the last transition snapshot against the first momentum snapshot for Pb, shell_n0, shell_nMax, shell_tauKappaRatio, F_rad and P_HII at matched R2; PHYSICS_SPEC test T13 on dv2/dt across the boundary.",
    "confidence": "medium",
    "lenses": ["A", "B", "C"],
    "divergence": "AC",
    "status": "corroborated",
    "source_ids": ["S6-A-02(context)", "S6-B-10", "S6-C-12"]
  },
  {
    "id": "S6-R-07",
    "file": "trinity/phase1c_transition/run_transition_phase.py",
    "line": 231,
    "class": "regime",
    "severity": "S2",
    "claim": "The transition phase changes ONLY the energy equation: its dR2/dt and dv2/dt are delegated unchanged to get_ODE_Edot_pure, i.e. the energy-phase RHS. If that RHS uses the energy-phase drive max(Pb, P_HII), the transition phase never gets the ram term in its equation of motion - while the transition file's own diagnostic P_drive at :331 does compute max(Pb, P_HII + P_ram).",
    "evidence": "Lens A section 2.1: \"(rd, vd, Ed_bal) = get_ODE_Edot_pure(t, y, snapshot, params) at :231 ... So dR2/dt and dv2/dt are EXACTLY the phase-1 energy-driven expressions - the transition phase changes only the energy equation\", with get_ODE_Edot_pure listed as opaque/out-of-slice. Lens A section 4.2: the transition's P_drive (:331) is computed and stored but 'is not used by any expression here; it is a diagnostic only'. Lens C S6-C-24/SPEC-022: the transition drive must be max(Pb, P_HII + P_ram), and omitting P_ram 'makes the transition exit later and hands over at a lower driving pressure, producing a downward step in dv2/dt at transition->momentum'. Lens B never defines P_drive in prose in either file.",
    "expected": "The integrated transition drive and the recorded diagnostic P_drive must be the same expression; per SPEC-022 both should be max(Pb, P_HII + P_ram).",
    "failure_scenario": "The reported P_drive column includes P_ram while the trajectory was integrated without it, so the force budget shown in the paper's stacked-force figure is not the one that produced R2(t). The transition also exits later and hands over under-driven, adding a downward step in dv2/dt exactly where the phase was introduced to remove one.",
    "repro": "Read get_ODE_Edot_pure and identify its P_drive branch; then, at each transition snapshot, compare the recorded P_drive against mShell*dv2/dt reconstructed from consecutive rows (SPEC-007 force closure).",
    "confidence": "medium",
    "lenses": ["A", "C"],
    "divergence": "AC",
    "status": "corroborated",
    "source_ids": ["S6-C-24", "S6-C-11"]
  },
  {
    "id": "S6-R-08",
    "file": "trinity/phase2_momentum/run_momentum_phase.py",
    "line": 448,
    "class": "numerical",
    "severity": "S2",
    "claim": "Inside the momentum RHS the external pressure is evaluated at the frozen segment-start shell radius while the area it multiplies uses the live integration radius, so the confining term is systematically stale in one direction.",
    "evidence": "Lens A S6-A-06: run_momentum_phase.py:424 'rShell = snapshot.rShell'; :427 get_density_profile(np.array([rShell]), params); :437 'if rShell >= snapshot.rCloud'; then :448 'F_pressure = FOUR_PI * R2**2 * (P_drive - P_ext)' with R2 taken live from y at :397 - while P_ram (:421), F_grav (:418) and the feedback lookup (:407) are all live. Lens C S6-C-15/S6-C-19: the local rho(R2) must be evaluated at the current radius, and freezing it is an operator-splitting error controlled only by the dex threshold - with alpha = -2 the local density varies by 4 dex over a run.",
    "expected": "Either all radius-dependent factors in one force use the same radius, or the segment length is bounded so the induced error is below ODE_RTOL.",
    "failure_scenario": "With dt_segment at its 5e-2 Myr ceiling and v2 ~ 10-100 pc/Myr the shell moves 0.5-5 pc inside a segment; in a steep profile n(rShell_frozen) is wrong by a large factor and the -4*pi*R2^2*P_ext term lags an expanding shell systematically rather than randomly.",
    "repro": "Instrument get_ODE_momentum_pure to log rShell vs the live R2 at the last RHS call of each segment on an alpha = -2 config.",
    "confidence": "medium",
    "lenses": ["A", "C"],
    "divergence": "AC",
    "status": "corroborated",
    "source_ids": ["S6-A-06", "S6-C-15", "S6-C-19"]
  },
  {
    "id": "S6-R-09",
    "file": "trinity/phase2_momentum/run_momentum_phase.py",
    "line": 826,
    "class": "state",
    "severity": "S2",
    "claim": "isCollapse is a one-way latch with no reset path in either runner, is inherited across the phase boundary, and is set one segment late; once set, the shell mass and mShell_dot are frozen for the rest of the simulation even if the shell re-expands.",
    "evidence": "Lens A S6-A-08: set True at run_transition_phase.py:773 and run_momentum_phase.py:826 under 'v2 < 0 and R2 < R2_prev'; no assignment of False anywhere in either file; when true, :540-543 / :601-604 bypass get_mass_profile entirely and hold mShell = prev_mShell, mShell_dot = 0.0. Lens B S6-B-16: two different definitions of collapse coexist ('v2 < 0' for the timestep control at :724/:803 vs 'v2 < 0 AND R2 decreasing' for the detector at :771/:824), and by line order the detector runs AFTER the timestep control and the results store, so the flag consumed by the next segment's shell-mass freeze is one segment old.",
    "expected": "Either a reset when the shell resumes expanding (v2 > 0 and R2 > R2_prev), or a latch documented as terminal; and one definition of collapse, or two clearly named conditions.",
    "failure_scenario": "A shell that contracts transiently and then re-expands past its previous radius accretes no further mass for the rest of the run; mShell_dot is pinned at 0 so the -mdot*v2 drag term vanishes permanently, biasing v2 upward. Note the separate monotone clamp already reproduces the freeze while the shell is actually contracting, so the latch only changes the result after re-expansion.",
    "repro": "Run a config that oscillates once; log isCollapse, v2, R2-R2_prev and shell_mass per segment and confirm shell_mass stops tracking mass_profile(R2) after the first contraction segment.",
    "confidence": "high",
    "lenses": ["A", "B"],
    "divergence": "none",
    "status": "corroborated",
    "source_ids": ["S6-A-08", "S6-B-16"]
  },
  {
    "id": "S6-R-10",
    "file": "trinity/phase1c_transition/run_transition_phase.py",
    "line": 338,
    "class": "divergence",
    "severity": "S3",
    "claim": "The output key F_ram means two different physical quantities in the two sibling runners - 4*pi*R2^2*Pb in phase1c, 4*pi*R2^2*P_ram in phase2 - and inside the transition runner's own ForceProperties object F_ram disagrees with its sibling field P_ram. The divergence is present in the prose as well as the code, so it is deliberate, not a silent copy-paste slip.",
    "evidence": "Lens A S6-A-01: run_transition_phase.py:338 'F_ram = Pb * FOUR_PI * R2**2' with Pb from compute_R1_Pb at :507, versus run_momentum_phase.py:272 'F_ram = P_ram * FOUR_PI * R2**2' with P_ram = pRam(...) at :225; in phase2 Pb is defined as pRam (:585, :667) so the two forms coincide there, while in phase1c the same object also carries P_ram = pRam(...) at :329. Lens B S6-B-10 sees the same split in prose: T:260 'Ram pressure force (from bubble pressure)' vs M:195 bare 'Ram pressure force', and notes the exit criterion at :749 treats P_b and P_ram as distinct in the very same file. Lens C S6-C-25/SPEC-007 requires the two ForceProperties classes to share field semantics and to satisfy force closure.",
    "expected": "Either both runners compute 4*pi*R2^2*P_ram (matching the sibling diagnostics F_ram_wind = pdot_W and F_ram_SN = pdot_SN), or phase1c stores the bubble-pressure force under a distinct key.",
    "failure_scenario": "Any plot or sum of the F_ram column across the phase boundary shows a discontinuous jump of order Pb/P_ram that is an artefact of the key, not of the physics; a force-budget closure check F_ram == F_ram_wind + F_ram_SN passes in phase2 and fails in phase1c. The published stacked force-fraction figure then mixes conventions - wrong in exactly one phase.",
    "repro": "Run param/simple_cluster.param and compare params['F_ram'] against 4*pi*R2**2*params['P_ram'] and against F_ram_wind + F_ram_SN in the last transition snapshot and the first momentum snapshot.",
    "confidence": "high",
    "lenses": ["A", "B", "C"],
    "divergence": "AC",
    "status": "corroborated",
    "source_ids": ["S6-A-01", "S6-B-10", "S6-C-25"]
  },
  {
    "id": "S6-R-11",
    "file": "trinity/phase1c_transition/run_transition_phase.py",
    "line": 135,
    "class": "numerical",
    "severity": "S3",
    "claim": "The comment on the LSODA max_step constant states 2e-5 Myr; the code computes DT_SEGMENT_MIN/5 = 1e-3/5 = 2e-4 Myr. The stated number is wrong by a factor of 10 in both twins, which carry byte-identical comments.",
    "evidence": "Lens A numeric-literal inventory: \"1e-3/5 = 2e-4 | T:135 / M:127 | LSODA max_step\". Lens B F16 and S6-B-11: \"Both files carry the byte-identical justification '# Max step = 2e-5 Myr (ensures >=5 steps per segment)' (run_momentum_phase.py:127, run_transition_phase.py:135)\". Same lines, different numbers - neither lens could see this alone.",
    "expected": "The comment states the value the expression actually produces (DT_SEGMENT_MIN/5 = 2e-4 Myr), or the expression is replaced by the literal it claims.",
    "failure_scenario": "A maintainer sizing DT_SEGMENT_MIN or diagnosing solver cost reasons from a cap 10x smaller than the real one. Note that the '>=5 steps per segment' invariant IS structurally satisfied (max_step is defined as DT_SEGMENT_MIN/5), so Lens B's derived failure scenarios about that guarantee do not apply; what remains is the wrong number plus a genuine cost note - a 5e-2 Myr segment is forced to at least 250 solver steps.",
    "repro": "Read the two lines; no run needed.",
    "confidence": "high",
    "lenses": ["A", "B"],
    "divergence": "AB",
    "status": "corroborated",
    "source_ids": ["S6-B-11"]
  },
  {
    "id": "S6-R-12",
    "file": "trinity/phase1c_transition/run_transition_phase.py",
    "line": 545,
    "class": "divergence",
    "severity": "S3",
    "claim": "The length-1-array unwrap applied to mass_profile.get_mass_profile results exists at exactly one of the four structurally identical shell-mass update sites in the slice - and the two sites that lack it in the adaptive-stepping blocks are documented as applying 'the same guards as the primary shell mass block above'.",
    "evidence": "Lens A S6-A-03: run_momentum_phase.py:610-613 unwraps mShell_new/mShell_dot and :780-781 unwraps mShell_post; run_transition_phase.py:545 and :702 make the identical calls with the identical arguments and no unwrap, writing straight into params['shell_mass'].value / params['shell_massDot'].value at :552-553 and :707. Lens B S6-B-08 reaches the same conclusion from a single comment: '# Handle array returns' appears only at run_momentum_phase.py:609, absent from run_transition_phase.py:541-546 and from BOTH adaptive-stepping copies (:692-705 and :769-784), even though those copies carry the word-identical contract at T:693-694 / M:770-771.",
    "expected": "All four sites coerce the return type identically, or the 'same guards as above' comment is corrected.",
    "failure_scenario": "If get_mass_profile ever returns a length-1 ndarray for scalar input, an ndarray is stored into params['shell_mass'].value; the never-decrease comparison still works for length 1, so the ndarray propagates silently into create_ODE_snapshot, the dex monitor and the snapshot writer before failing or serialising oddly downstream.",
    "repro": "Assert isinstance(params['shell_mass'].value, float) after the first transition segment and after the first momentum segment; determine under what inputs get_mass_profile returns an array.",
    "confidence": "medium",
    "lenses": ["A", "B"],
    "divergence": "none",
    "status": "corroborated",
    "source_ids": ["S6-A-03", "S6-B-08"]
  },
  {
    "id": "S6-R-13",
    "file": "trinity/phase2_momentum/run_momentum_phase.py",
    "line": 886,
    "class": "divergence",
    "severity": "S3",
    "claim": "The momentum phase's final reconciliation refreshes Pb, R1 and shell structure but never recomputes the force set, so the last momentum snapshot pairs final R2/v2/t with forces from the previous segment. The transition twin does recompute them - and the two files' comments diverge in the same direction.",
    "evidence": "Lens A S6-A-05: run_momentum_phase.py:886-894 calls get_current_sps_feedback, pRam, shell_structure_pure and save_snapshot with no call to compute_forces_momentum_pure; run_transition_phase.py:850-859 does call compute_forces_pure and rewrites F_grav/F_ion_in/F_HII/F_ram/F_rad/P_HII/P_drive/P_ram before its save at :860. Lens B twin diff: T:828-832 'Recompute derived properties (Pb, shell structure, forces)' vs M:881-885 'Recompute derived properties' - B dropped the enumeration.",
    "expected": "Both final blocks refresh the same set of derived keys, or neither does. Neither currently refreshes press_HII_in, F_ram_wind, F_ram_SN, n_IF or R_IF.",
    "failure_scenario": "The final row of a momentum-terminating run reports forces evaluated at the previous segment's R2 and shell state - up to 5e-2 Myr of drift - while R2, v2, rShell and shell_mass in the same row are current.",
    "repro": "Compare the last two rows of dictionary.jsonl for a momentum-terminating run: F_grav should equal G*m*(M+m/2)/R2^2 with the row's own R2, and will not.",
    "confidence": "high",
    "lenses": ["A", "B"],
    "divergence": "none",
    "status": "corroborated",
    "source_ids": ["S6-A-05"]
  },
  {
    "id": "S6-R-14",
    "file": "trinity/phase2_momentum/run_momentum_phase.py",
    "line": 350,
    "class": "divergence",
    "severity": "S3",
    "claim": "The recorded force budget is not the integrated one. Four independent mechanisms: (a) the RHS re-fetches feedback live at :407 and shadows the snapshot's Lmech_total/v_mech_total, while the diagnostic forces are built from the segment-start feedback; (b) F_rad is computed inside create_momentum_snapshot and frozen for the segment while the feedback driving it is live; (c) each file contains two independent force implementations ~150 lines apart with no asserted equivalence; (d) F_rad exists in three separate copies across the slice.",
    "evidence": "Lens A S6-A-07: six of the eighteen MomentumODESnapshot fields are never read by the RHS, including Lmech_total (:350) and v_mech_total (:351), which :407-409 recompute live from get_current_sps_feedback(t, params) - while compute_forces_momentum_pure receives feedback.Lmech_total from :644-645. Lens A section 3.2: F_rad is computed inside the snapshot builder at :340-345, a third copy of the same expression (T:344, M:278, M:343). Lens B S6-B-04 independently flags the freshness contradiction between :339 (frozen F_rad) and :405-406 ('Use live feedback so SN turn-on events mid-segment are visible'); Lens B S6-B-06 flags the two-force-implementations split in both files, and S6-B-15 records the unpinned '(same as in energy_implicit)' claim at T:251, implying a third implementation elsewhere. Lens C S6-C-25/SPEC-007 requires the recorded forces to reproduce M_sh*dv2/dt.",
    "expected": "One force implementation per phase, or a pytest case asserting the reporting path and the integration path agree term-by-term at identical state; and a single freshness policy (live or frozen) for all feedback-derived terms.",
    "failure_scenario": "A supernova turning on mid-segment updates Lmech/pdot in the integrated ram term but not the frozen F_rad and not the recorded diagnostics, so the published force budget silently disagrees with R2(t), v2(t) while every run completes normally. Lens A verified the current copies agree expression-by-expression EXCEPT for F_ram and P_drive - which proves the drift mechanism is real and has already fired once.",
    "repro": "For one momentum segment, evaluate both paths at identical state and assert equality; separately, set up a run whose SN turn-on falls strictly inside a segment and compare v2 at segment end against a run with DT_SEGMENT forced small enough that the turn-on lands on a boundary (separate processes, matched t).",
    "confidence": "high",
    "lenses": ["A", "B", "C"],
    "divergence": "AC",
    "status": "corroborated",
    "source_ids": ["S6-A-07", "S6-B-04", "S6-B-06", "S6-B-15", "S6-C-25"]
  },
  {
    "id": "S6-R-15",
    "file": "trinity/phase2_momentum/run_momentum_phase.py",
    "line": 531,
    "class": "state",
    "severity": "S3",
    "claim": "The dissolution clock t_diss_onset is a function-local in both runners and is re-initialised to inf at the phase boundary, so accumulated time below nISM is lost when the transition phase hands off to the momentum phase.",
    "evidence": "Lens A S6-A-09: run_transition_phase.py:449 't_diss_onset = np.inf' and run_momentum_phase.py:531 identically; the test is '(t_now - t_diss_onset) >= params[\"stop_t_diss\"].value' at :813 / :866; nothing writes the onset time into params and the momentum phase re-initialises unconditionally. Lens B S6-B-19 confirms from prose that the persistence threshold is never stated in either file (but Lens A settles B's worry that the timer might count segments - it is absolute Myr from a param).",
    "expected": "The onset time lives in params (as isCollapse and isDissolved do) so the clock survives the phase change.",
    "failure_scenario": "A shell that has been below nISM for 0.99*stop_t_diss when the transition exits on ram_dominated must wait a further full stop_t_diss in the momentum phase before being declared dissolved - the run continues up to stop_t_diss longer than intended, and the recorded fate can differ from a single-phase reference.",
    "repro": "Config with a small stop_t_diss and a ram-dominated transition exit; compare the dissolution time against a run that stays in one phase.",
    "confidence": "high",
    "lenses": ["A"],
    "divergence": "none",
    "status": "single-lens",
    "source_ids": ["S6-A-09", "S6-B-19"]
  },
  {
    "id": "S6-R-16",
    "file": "trinity/phase1c_transition/run_transition_phase.py",
    "line": 491,
    "class": "state",
    "severity": "S3",
    "claim": "Both runners re-stamp params['T0'] every segment from a local captured once at phase entry, clobbering any value written by updateDict or the shell/bubble modules - and Lens C requires T0 to be dropped at the transition->momentum boundary, not carried.",
    "evidence": "Lens A S6-A-10: run_transition_phase.py:430 'T0 = params[\"T0\"].value' with the local never reassigned in the loop, and :491 'params[\"T0\"].value = T0' executed each segment; identically run_momentum_phase.py:509 and :572. T0 is in ADAPTIVE_MONITOR_KEYS (T:114 / M:106) where it can therefore never contribute a non-zero dex. Lens C section 2.4: 'Eb, Pb, R1, T0, bubble n(r)/T(r) profiles: dropped, not carried'.",
    "expected": "Either T0 is a genuine state variable and is evolved/read back, or it is not re-written at all - and it should not survive into the momentum phase, where Eb is identically 0.",
    "failure_scenario": "Every snapshot in a phase reports the same T0; anything that updates T0 between segments (feedback updateDict at :497, shell_structure_pure at :558) is overwritten on the next iteration, and the momentum phase carries a frozen phase-entry bubble temperature alongside Eb = 0 - a physically meaningless pair that downstream analysis may read.",
    "repro": "Log params['T0'] immediately after updateDict and again after the next :491 write.",
    "confidence": "high",
    "lenses": ["A", "C"],
    "divergence": "AC",
    "status": "corroborated",
    "source_ids": ["S6-A-10", "S6-C-12"]
  },
  {
    "id": "S6-R-17",
    "file": "trinity/phase1c_transition/run_transition_phase.py",
    "line": 240,
    "class": "regime",
    "severity": "S3",
    "claim": "In the fallback branch (c_sound <= 0 or R2 <= 0) the sound-crossing loss is set to 0.0 and the subsequent min() then clamps dEb/dt to be non-positive, silently deleting any energy injection instead of falling back to the unmodified energy balance.",
    "evidence": "Lens A S6-A-14: run_transition_phase.py:237-240 sets Ed_soundcrossing = -Eb*c_sound/R2 or 0.0; :245 'Ed = min(Ed_energy_balance, Ed_soundcrossing)'. With the fallback value 0.0 this becomes Ed = min(Ed_bal, 0). c_sound comes from :517 with a literal 1e6 K fallback at :516 that also fires when bubble_Tavg.value == 0 (falsy). Lens B F1/F5 confirm the intended model is min(Ed_energy_balance, Ed_soundcrossing) with the min() 'selecting whichever gives faster energy loss'; no prose covers the fallback.",
    "expected": "If the sound-crossing term is unavailable the energy equation falls back to the unmodified balance (Ed = Ed_energy_balance), not to a one-sided clamp - and the fallback should log.",
    "failure_scenario": "If get_soundspeed ever returns 0 the bubble energy is frozen from above for the whole segment and Eb decays only through Ed_bal - a different equation from the one the phase is meant to integrate, with no log line. Related: for Eb < 0 (integrator overshoot) Ed_sc flips sign and the leak becomes a source; the only protection is the energy_floor event.",
    "repro": "Force c_sound = 0 for one segment and compare the Eb trace against the unmodified energy-balance integration.",
    "confidence": "medium",
    "lenses": ["A"],
    "divergence": "none",
    "status": "single-lens",
    "source_ids": ["S6-A-14"]
  },
  {
    "id": "S6-R-18",
    "file": "trinity/phase1c_transition/run_transition_phase.py",
    "line": 300,
    "class": "regime",
    "severity": "S3",
    "claim": "P_ext is a step function of shell_fAbsorbedIon - the full ionized-gas pressure for FABSi = 1-eps and exactly zero at FABSi = 1, with no (1-FABSi) weighting - and any exception inside the density-profile lookup also maps to P_ext = 0.0 with no log. Separately, this n(rShell)*k_B*T_ion confinement term has no counterpart in Lens C's derived force budget.",
    "evidence": "Lens A S6-A-15: run_transition_phase.py:299-310 'if FABSi < 1.0: P_ext = (mu_convert/mu_ion_shell)*n(rShell)*k_B*TShell_ion else: P_ext = 0.0', identically run_momentum_phase.py:240-250 and, inside the integrated RHS, :425-434; the magnitude never depends on FABSi. The try/except at :307 / :247 converts any get_density_profile failure into P_ext = 0.0. Lens B section 4 records only that 'the ambient/ISM pressure term engages beyond the cloud edge' and lists an 'outside-shell inward pressure' block. Lens C section 1.2 lists only P_ext = k_B*PISM as the ambient inward term and states in section 1.4 that the cloud's own thermal/turbulent pressure is NOT represented - so the code carries an inward term the derived spec does not contain.",
    "expected": "A continuous dependence on the escaping fraction (or a documented reason the discontinuity is acceptable), a logged exception path, and a note reconciling the n(rShell)*k_B*T_ion term with the modelled physics.",
    "failure_scenario": "A run in which FABSi crosses 1.0 between segments sees the confining term -4*pi*R2^2*P_ext appear or vanish discontinuously in the integrated momentum equation, producing a kink in v2 that is a property of the branch, not the physics. A silent density-profile failure removes the confinement entirely for that segment.",
    "repro": "Log shell_fAbsorbedIon and P_ext per segment for a config that saturates absorption; assert the try/except never fires on the default configs.",
    "confidence": "medium",
    "lenses": ["A", "B"],
    "divergence": "scope-creep",
    "status": "single-lens",
    "source_ids": ["S6-A-15"]
  },
  {
    "id": "S6-R-19",
    "file": "trinity/phase2_momentum/run_momentum_phase.py",
    "line": 398,
    "class": "numerical",
    "severity": "S3",
    "claim": "The momentum RHS clamps R2 and mShell to 1e-10 locally without clamping the integrator state, so the returned RHS corresponds to a different radius/mass than the state being integrated, and F_grav at the clamp is ~1e20*G*M.",
    "evidence": "Lens A S6-A-13: run_momentum_phase.py:398 'R2 = max(R2, 1e-10)' rebinds the local unpacked from y; :415 'mShell = max(mShell, 1e-10)'; the returned rd = v2 (:451) is unaffected, so the solver state can keep decreasing while F_grav (:418) and P_ram (:421) are evaluated at the clamp. mShell_dot is not clamped, so the drag term mShell_dot*v2/1e-10 can dominate. The small_radius check at :842 only runs between segments.",
    "expected": "Terminate via an event before R2 reaches the clamp, or clamp consistently (state and RHS together).",
    "failure_scenario": "During a collapse the state R2 crosses zero inside a segment; the RHS returns a finite huge inward acceleration instead of failing, LSODA burns the segment down to min_step, and the recorded trajectory near R2 -> 0 is governed entirely by the clamp value rather than by physics.",
    "repro": "A collapsing config with coll_r below 1e-10 pc, or instrument the RHS to log when either max() is active.",
    "confidence": "medium",
    "lenses": ["A"],
    "divergence": "none",
    "status": "single-lens",
    "source_ids": ["S6-A-13"]
  },
  {
    "id": "S6-R-20",
    "file": "trinity/phase1c_transition/run_transition_phase.py",
    "line": 368,
    "class": "divergence",
    "severity": "S3",
    "claim": "run_phase_transition's docstring states the pre-min() energy model ('Energy decays on the sound-crossing timescale until it reaches a floor value, then momentum phase begins') while the module docstring (:3) and the ODE docstring (:200) state the current model, dE/dt = min(Ed_energy_balance, Ed_soundcrossing) - which is what the code computes.",
    "evidence": "Lens B S6-B-02: run_transition_phase.py:368 vs :3 and :200. Lens A section 2.1 confirms the code computes Ed = min(Ed_bal, -Eb*c_sound/R2) at :238-245, so the two current docstrings are right and the entry-point docstring is stale.",
    "expected": "The public function docstring states dE/dt = min(Ed_energy_balance, Ed_soundcrossing).",
    "failure_scenario": "A maintainer reading only the entry-point docstring believes energy decay is purely -Eb*c_sound/R2 and mis-diagnoses Ed behaviour driven by the energy-balance branch (attributing a slow decay to a wrong c_sound rather than to Lcool).",
    "repro": "Read the three docstrings side by side; no run needed.",
    "confidence": "high",
    "lenses": ["A", "B"],
    "divergence": "AB",
    "status": "corroborated",
    "source_ids": ["S6-B-02"]
  },
  {
    "id": "S6-R-21",
    "file": "trinity/phase1c_transition/run_transition_phase.py",
    "line": 765,
    "class": "divergence",
    "severity": "S3",
    "claim": "The transition phase documents two competing phase-ending criteria: three comments (:97, :368, :456) present the energy floor as the thing that ends the phase, while :746-749 calls ram-pressure dominance the 'Phase transition criterion' and :765 demotes the floor to a 'Safety fallback'. Both exist in the code, and which one fires is not documented anywhere.",
    "evidence": "Lens B S6-B-01: run_transition_phase.py:97, :368, :456 vs :746-749, :765. Lens A confirms both code paths exist and are independent: the ram exit at :749-763 (RAM_DOMINANCE_THRESHOLD = 0.9, re-bound inside the loop each iteration) and the floor exit at :766-769 (Eb < 1e3), plus an energy_floor terminal solve_ivp event built at :457 from the same constant. Lens A also settles Lens B's S6-B-18 worry: both floor checks read the same ENERGY_FLOOR = 1e3, so they cannot disagree. Lens C S6-C-11 wants the exit expressed as a force/pressure comparison, which the ram criterion is.",
    "expected": "One documented primary criterion with the other explicitly subordinate in every place it is mentioned; and a recorded field saying which exit fired.",
    "failure_scenario": "The exit that actually fires determines how much Eb is discarded at the hand-off (see S6-R-05) and where in the drive-ratio the momentum phase starts, but neither the prose nor the output identifies it. The RAM_DOMINANCE_THRESHOLD constant being re-bound inside the loop body also makes it look configurable when it is not.",
    "repro": "Instrument both exit paths and record which fires, plus Eb at exit relative to ENERGY_FLOOR, for param/simple_cluster.param and docs/dev/performance/f1edge_hidens*.param.",
    "confidence": "high",
    "lenses": ["A", "B"],
    "divergence": "AB",
    "status": "corroborated",
    "source_ids": ["S6-B-01", "S6-B-18"]
  },
  {
    "id": "S6-R-22",
    "file": "trinity/phase2_momentum/run_momentum_phase.py",
    "line": 427,
    "class": "numerical",
    "severity": "S3",
    "claim": "Nothing places a segment boundary or a solver event at the R2 = rCloud density discontinuity, even though the ambient density, the swept-mass slope and the PISM branch of P_ext all switch there.",
    "evidence": "Lens C S6-C-19: SPEC-060 makes rho discontinuous at r_cloud unless smoothed; a jump in the RHS mid-segment makes an adaptive stiff solver chatter or step over it. Lens A's transcription shows the only rCloud-aware logic is the P_ext branch ('if rShell >= rCloud: P_ext += PISM*k_B', T:314 / M:254 / M:438), the stop_at_rCloud snapshot counter (T:468-479 / M:547) and the rCloud snapshot bookkeeping - the event list built at T:457 is energy_floor / min_radius / velocity_runaway, with no rCloud crossing event.",
    "expected": "A terminal or non-terminal solve_ivp event at R2 = rCloud so the crossing lands on a segment boundary, or documented smoothing (the rcloud_smoothing machinery referenced by Lens C).",
    "failure_scenario": "A segment straddles the cloud edge with a frozen snapshot taken inside the cloud; the swept mass beyond r_cloud is then wrong, which changes the escape/stall verdict - the qualitative fate is decided at exactly this radius (Lens C section 4.4).",
    "repro": "Log the solver step size and mShell slope across the r_cloud crossing for docs/dev/performance/f1edge_lowdens*.param.",
    "confidence": "medium",
    "lenses": ["A", "C"],
    "divergence": "AC",
    "status": "single-lens",
    "source_ids": ["S6-C-19"]
  },
  {
    "id": "S6-R-23",
    "file": "trinity/phase2_momentum/run_momentum_phase.py",
    "line": 842,
    "class": "regime",
    "severity": "S3",
    "claim": "Re-collapse is declared on a bare radius threshold (R2 < coll_r, gated on the isCollapse latch) rather than on a dynamical/bound criterion, and stall is not distinguished from a turning point by the sign of the net force at v2 = 0.",
    "evidence": "Lens C S6-C-22: SPEC-103 - coll_r = 1 pc is scale-dependent; a 1e9 Msun cloud collapsing from 100 pc to 2 pc has manifestly collapsed but never trips it. The correct criterion is v2 < 0 sustained AND 0.5*v2^2 < G(M_cluster+M_sh)/R2. Lens A's transcription supplies the fact: the small_radius exit is 'R2 < coll_r, only while isCollapse' at run_transition_phase.py:791-793 / run_momentum_phase.py:842, and there is no bound/escape-velocity test anywhere in either runner.",
    "expected": "Fate assignment from (v2, dv2/dt, net force sign, v_esc), with any radius threshold used only as a numerical floor.",
    "failure_scenario": "Large clouds are never labelled 'collapse' and instead time out on stop_t, biasing the published fate statistics toward dispersal at the high-mass end of the grid.",
    "repro": "Cross-tabulate recorded fates against (min R2, sign of v2 at the end) for the paperII sweep.",
    "confidence": "medium",
    "lenses": ["A", "C"],
    "divergence": "AC",
    "status": "single-lens",
    "source_ids": ["S6-C-22"]
  },
  {
    "id": "S6-R-24",
    "file": "trinity/phase2_momentum/run_momentum_phase.py",
    "line": 849,
    "class": "regime",
    "severity": "S3",
    "claim": "Escape/blowout is never tested against the escape velocity: the runners stop on R2 > stop_r or on a snapshot count past rCloud, with no v2 > sqrt(2G(M_cluster+M_sh)/R2) condition.",
    "evidence": "Lens C S6-C-23 (SPEC-104, SPEC-032): inside a uniform cloud F_grav ~ M_sh^2/R2^2 ~ R2^4 grows faster than any driving term, so a shell can cross r_cloud sub-escape and still turn around. Lens A's transcription shows only large_radius (R2 > stop_r) and the stop_at_rCloud_nSnap counter (T:468-479 / M:547), which Lens C itself classifies as a run-length control rather than a fate. The mapping from these exits to a published fate label happens outside this slice (phase_events / _output).",
    "expected": "Any 'escaped/dispersed' fate label carries the velocity test; stop_at_rCloud_nSnap remains a run-length control.",
    "failure_scenario": "Shells that would have re-collapsed are recorded as escaped - a direct over-count of cloud dispersal, the paper's headline quantity.",
    "repro": "For every run terminated at the rCloud crossing or at stop_r, compute v2/v_esc at that snapshot; inspect how the SimulationEndCode set at those sites is rendered downstream.",
    "confidence": "low",
    "lenses": ["C"],
    "divergence": "AC",
    "status": "single-lens",
    "source_ids": ["S6-C-23"]
  },
  {
    "id": "S6-R-25",
    "file": "trinity/phase1c_transition/run_transition_phase.py",
    "line": 861,
    "class": "silent-failure",
    "severity": "S4",
    "claim": "A failure of the final phase-boundary reconciliation is swallowed as a warning in both runners - the run reports success while the final snapshot is silently missing - and the diagnostic quality of that warning was improved in the momentum twin only.",
    "evidence": "Lens A S6-A-04: run_momentum_phase.py:899-908 extracts the last traceback frame and logs 'type(e).__name__: msg at file:lineno' with a '<no message>' fallback, while run_transition_phase.py:861-862 logs only f'Phase-boundary reconciliation failed: {e}'; both blocks are bare try/except around the whole final reconciliation. Lens B S6-B-09 sees the same asymmetry in prose: M:896-898 documents the enriched handler, T:828-832 documents none.",
    "expected": "Identical diagnostics in both twins, and a failed final snapshot marked in the run's termination status rather than only in a log line.",
    "failure_scenario": "The final reconciliation raises (Lens B's comment names four plausible failure points: SPS lookup, pRam, shell_structure, save_snapshot); the run exits with a success termination_reason and the last record in the output is the earlier in-loop snapshot, with nothing marking the output as incomplete. In the transition runner an empty-message exception logs no information at all.",
    "repro": "Force shell_structure_pure to raise inside each final block; compare the two log lines and check the recorded termination status and the last dictionary.jsonl t.",
    "confidence": "high",
    "lenses": ["A", "B"],
    "divergence": "none",
    "status": "corroborated",
    "source_ids": ["S6-A-04", "S6-B-09"]
  },
  {
    "id": "S6-R-26",
    "file": "trinity/phase2_momentum/run_momentum_phase.py",
    "line": 375,
    "class": "divergence",
    "severity": "S4",
    "claim": "get_ODE_momentum_pure's docstring documents its params argument as 'Original params for density profile lookup' while the body also drives a live feedback lookup; the transition twin documents the same argument correctly and types it DescribedDict rather than dict.",
    "evidence": "Lens B S6-B-03: run_momentum_phase.py:375 vs :405-406 ('Use live feedback so SN turn-on events mid-segment are visible'); twin at run_transition_phase.py:200 'Original params dict for feedback interpolation'. Lens A confirms the body: get_current_sps_feedback(t, params) at :407 evaluated at every RHS call, and get_density_profile at :427 - so params serves both purposes and the docstring lists one.",
    "expected": "The momentum ODE docstring lists both uses and matches the twin's type annotation.",
    "failure_scenario": "A caller trusting the docstring passes a params object carrying only the density profile (a trimmed dict in a test harness or replay rig); the live feedback lookup either raises or silently falls back, changing the ODE's drive term.",
    "repro": "Read the two docstrings and the two bodies side by side.",
    "confidence": "high",
    "lenses": ["A", "B"],
    "divergence": "AB",
    "status": "corroborated",
    "source_ids": ["S6-B-03"]
  },
  {
    "id": "S6-R-27",
    "file": "trinity/phase1c_transition/run_transition_phase.py",
    "line": 105,
    "class": "divergence",
    "severity": "S4",
    "claim": "The velocity-based timestep constants are documented in terms of |v2| but the guard is gated on v2 < 0 first. The comment is the stale part: Lens C's derived expectation is a SIGNED comparison, so the code is right and the prose is wrong.",
    "evidence": "Lens B S6-B-05: constants at run_transition_phase.py:105-107 ('when |v2| exceeds threshold...') vs usage at :724 ('Only during collapse (negative velocity = inward motion)'); identically run_momentum_phase.py:97-99 vs :803. Lens A confirms the code: the override applies 'only for v2 < 0', then |v2| > 150 -> dt = 5e-4, elif |v2| > 50 -> dt = min(dt, 1e-3), in pc/Myr. Lens C S6-C-21 expects exactly 'v2 < -VELOCITY_THRESHOLD_COLLAPSE' with |EXTREME| > |COLLAPSE| in pc/Myr, and warns that comparing |v2| would wrongly fire during fast expansion.",
    "expected": "The constants' comments read 'when v2 < -threshold'.",
    "failure_scenario": "Documentation only. Lens B's failure scenario (no refinement for fast OUTWARD motion at breakout or SN turn-on) is a real but different concern that belongs to S6-R-03 - Lens C explicitly does not want the collapse path firing during expansion.",
    "repro": "Read the constants and the guard; no run needed.",
    "confidence": "high",
    "lenses": ["A", "B", "C"],
    "divergence": "AB",
    "status": "corroborated",
    "source_ids": ["S6-B-05", "S6-C-21"]
  },
  {
    "id": "S6-R-28",
    "file": "trinity/phase1c_transition/run_transition_phase.py",
    "line": 143,
    "class": "silent-failure",
    "severity": "S4",
    "claim": "compute_max_dex_change skips any monitored key whose old or new value is exactly 0, so a quantity that reaches zero is invisible to the step controller. (Lens C's stronger claim - that the function returns NaN and silently disables refinement - is refuted: the guards exist and a sign flip contributes a literal 1.0 dex, which exceeds the 0.1 threshold and therefore REFINES, the fail-safe direction Lens C asked for.)",
    "evidence": "Lens A section 1: '|log10(|new|/|old|)| with three guards: either value None -> skipped; either value == 0 -> skipped; sign flip (old>0) != (new>0) -> contributes the bare literal 1.0 dex and continues'; ValueError/ZeroDivisionError are caught per key and np.log10(0) is pre-empted by the == 0 guard. Lens C S6-C-14 predicted NaN at v2 = 0 and at Eb -> floor. In the momentum phase Eb is pinned to exactly 0.0 and is therefore permanently skipped (Lens A section 5).",
    "expected": "The zero case returns a refine-biased value rather than being skipped, and the 1.0-dex sign-flip constant is named rather than a bare literal.",
    "failure_scenario": "Low. Exact float zeros are rare for continuous quantities; the practical instance is the momentum phase's Eb == 0.0, which is inert anyway (see S6-R-03).",
    "repro": "Call with params_before v2 = +1 and params_after v2 = -1 (asserts the 1.0-dex path), and with params_after v2 = 0.0 (asserts the skip).",
    "confidence": "high",
    "lenses": ["A", "C"],
    "divergence": "AC",
    "status": "contested",
    "source_ids": ["S6-C-14"]
  },
  {
    "id": "S6-R-29",
    "file": "trinity/phase1c_transition/run_transition_phase.py",
    "line": 291,
    "class": "deadcode",
    "severity": "S4",
    "claim": "Dead reads, dead fields and unreachable guards throughout the slice: nISM read and unused in both compute_forces functions; six of eighteen MomentumODESnapshot fields never read by the RHS (nISM, n_IF, include_PHII, isCollapse, Lmech_total, v_mech_total); R1_post computed and unused; v2_from_alpha used only in an f-string; F_ISM in ADAPTIVE_MONITOR_KEYS but never written; dead imports (scipy.optimize T:51, Tuple T:53, unit_conversions as cvt T:57 and M:57); 'if tmax is not None' guards that can never be False; and P_HII/F_HII written twice per segment with identical values.",
    "evidence": "Lens A S6-A-17 and S6-A-18: nISM at run_transition_phase.py:291 / run_momentum_phase.py:232; snapshot fields at run_momentum_phase.py:350-365; R1_post at :752; v2_from_alpha at :391; 'if tmin >= tmax' at T:407 / M:488 executes before any None check, so the later guards at T:605/:779 and M:687/:832 are unreachable in their False branch. Lens B S6-B-07 flags the 'Gate all HII pressure' field documented at M:316 - Lens A explains it: include_PHII gates P_HII in the RUNNER of both files (T:563-569, M:634), so the snapshot copy is redundant and the reported and integrated P_HII are the same gated value. Lens B S6-B-13 flags the copied 'Cooling'/'Bubble' monitor headings in the momentum file.",
    "expected": "Remove, or use. Flag as pre-existing dead code rather than deleting opportunistically (project rule 3).",
    "failure_scenario": "",
    "repro": "",
    "confidence": "high",
    "lenses": ["A", "B"],
    "divergence": "none",
    "status": "corroborated",
    "source_ids": ["S6-A-17", "S6-A-18", "S6-A-07", "S6-B-07", "S6-B-13"]
  },
  {
    "id": "S6-R-30",
    "file": "trinity/phase1c_transition/run_transition_phase.py",
    "line": 3,
    "class": "citation",
    "severity": "S4",
    "claim": "The slice contains zero literature citations. The defining physics of phase 1c - dE/dt = min(Ed_energy_balance, -Eb/(R2/c_sound)) - is presented with no derivation and no reference; 'Stroemgren ionization balance' is invoked six times with no equation; 'IR-trapped' radiation pressure names no trapping model; 'the implicit phase's beta-derived Ed' names a formalism with no source.",
    "evidence": "Lens B S6-B-12 (B section 3: ~640 prose entries, zero references to any paper, equation number, thesis or textbook). Lens C S6-C-30 adds that literature access was blocked for this audit and that the Rahner MNRAS paper and thesis number equations differently, so any 'eq. N' would have to be audited on content anyway - which makes the absence of numbers less harmful than the absence of a derivation. Lens A confirms the min() form is exactly what the code computes and that Ed_sc is dimensionally sound (energy/time).",
    "expected": "The energy-decay law, the Stroemgren balance expression and the IR-trapping model each carry a reference or an in-repo derivation link. Prefer prefactor-free structural forms over hard-coded literature prefactors.",
    "failure_scenario": "The min() decay law cannot be validated against any published model, so a coefficient or sign error in either branch is undetectable by review. In particular the docstring's claim that the energy-balance branch dominates 'early on' and the sound-crossing branch 'once cooling becomes inefficient' is an unchecked assertion about the ordering of two rates; if the sound-crossing branch wins at the first step, the advertised continuity with the implicit phase silently does not hold.",
    "repro": "Log which branch of the min() is selected at each step for param/simple_cluster.param and docs/dev/performance/f1edge_hidens*.param; check the selected branch at t_transition_start.",
    "confidence": "high",
    "lenses": ["B", "C"],
    "divergence": "BC",
    "status": "corroborated",
    "source_ids": ["S6-B-12", "S6-C-30"]
  },
  {
    "id": "S6-R-31",
    "file": "trinity/phase1c_transition/run_transition_phase.py",
    "line": 314,
    "class": "units",
    "severity": "S4",
    "claim": "Verification item, not a defect: PISM is multiplied by k_B before being added to P_ext, which is correct if and only if PISM is stored as P/k_B in K*pc^-3 internal units. Lens A hypothesised this with low confidence; Lens C independently states SPEC-003 declares PISM in K cm^-3 - so the two lenses agree the code is right. What remains unverified by either is the cm^-3 -> pc^-3 conversion at .param ingestion.",
    "evidence": "Lens A S6-A-16: run_transition_phase.py:306 builds P_ext as (mu_convert/mu_ion_shell)*n*k_B*T, then :313-314 adds 'PISM * k_B' with no density, temperature or mu ratio; identically run_momentum_phase.py:254 and :438. Lens C S6-C-27: 'PISM is declared in K cm^-3 (P/k_B) and must be multiplied by k_B and converted to Msun pc^-1 Myr^-2 before use', and the default PISM = 0 means the term is invisible in every default run - only the sweep cells that set it are affected.",
    "expected": "Confirm the .param schema declares PISM in K cm^-3 and that ingestion converts cm^-3 -> pc^-3; then the k_B multiplication in the force functions is correct and no further factor is needed.",
    "failure_scenario": "If ingestion leaves PISM in K cm^-3, the ambient confinement is wrong by (pc/cm)^3 ~ 2.9e55 - loud. If it converts, the code is correct. Either way the default PISM = 0 hides it until a sweep sets it.",
    "repro": "Read the declared unit string for PISM in trinity/_input/; run with PISM = 1e6 K cm^-3 and check the recorded external force equals 4*pi*R2^2 * 1.4e-10 dyn cm^-2 in cgs.",
    "confidence": "medium",
    "lenses": ["A", "C"],
    "divergence": "none",
    "status": "corroborated",
    "source_ids": ["S6-A-16", "S6-C-27"]
  },
  {
    "id": "S6-R-32",
    "file": "trinity/phase1c_transition/run_transition_phase.py",
    "line": 134,
    "class": "numerical",
    "severity": "S4",
    "claim": "ODE_MIN_STEP is set to 1e-6 Myr (> 0), which Lens C classifies as an accuracy hazard: if LSODA needs a smaller step it either raises or silently accepts a larger one, and either way the requested tolerance is no longer met.",
    "evidence": "Lens A numeric inventory: min_step = 1e-6 at T:134 / M:126, passed only to LSODA ('min_step only supported by LSODA', Lens B A:626 / B:708). Lens C S6-C-28 expects ODE_MIN_STEP = 0 unless justified, and separately notes ODE_MAX_STEP must be below the SPS-table and cooling-file sampling - which the actual 2e-4 Myr (see S6-R-11) plausibly satisfies.",
    "expected": "min_step = 0, or a comment justifying the floor and stating what happens when the solver wants to go below it.",
    "failure_scenario": "At the stiffest moments (collapse, the max() branch crossing) the requested rtol/atol are quietly not met.",
    "repro": "Compare ODE_MAX_STEP against the SPS table dt; instrument for solver warnings at the branch crossing.",
    "confidence": "medium",
    "lenses": ["A", "C"],
    "divergence": "AC",
    "status": "single-lens",
    "source_ids": ["S6-C-28"]
  },
  {
    "id": "S6-R-33",
    "file": "trinity/phase1c_transition/run_transition_phase.py",
    "line": 399,
    "class": "state",
    "severity": "S4",
    "claim": "cool_alpha is re-derived once at phase entry as t_now*v2/R2 and never updated inside the segment loop, although it is a cooling parameter that depends on v2 and v2 evolves through the phase. There is also no guard on R2 == 0 or t_now == 0.",
    "evidence": "Lens A section 2.3: ':399 then overwrites cool_alpha := t_now*v2/R2 (dimensionless: Myr*pc/Myr/pc), i.e. alpha is re-derived from the ODE velocity, self-consistently. No guard on R2 == 0 or t_now == 0'; the entry block runs once. Lens B S6-B-17: the comment ':398 Update cool_alpha to match ODE-evolved v2 (preserves ODE continuity)' sits in the Initialization block, while 'Cooling parameters' appear in the per-segment monitor list (:117), implying they are expected to vary.",
    "expected": "State whether cool_alpha is a one-shot entry fixup or a per-segment quantity; if it depends on v2 and v2 evolves, the continuity claim holds only at t = t_entry.",
    "failure_scenario": "If get_ODE_Edot_pure consumes cool_alpha rather than recomputing it, Ed_energy_balance is evaluated with an entry-time cooling coefficient for the whole phase - and since the min() branch selection depends on Ed_energy_balance's magnitude, a stale cool_alpha can change WHICH branch is selected, not just its value.",
    "repro": "Check whether cool_alpha is recomputed anywhere inside the segment loop or inside get_ODE_Edot_pure; log cool_alpha and v2 per segment.",
    "confidence": "medium",
    "lenses": ["A", "B"],
    "divergence": "none",
    "status": "corroborated",
    "source_ids": ["S6-B-17"]
  },
  {
    "id": "S6-R-34",
    "file": "trinity/phase1c_transition/run_transition_phase.py",
    "line": 860,
    "class": "other",
    "severity": "S4",
    "claim": "When the loop exits via the reached_tmax check the final reconciliation block re-saves at the same t_now with the same state, and the _snapshots_after_rCloud counter is not bumped for that save, so the rCloud snapshot count and the actual snapshot count can disagree by one. (Whether a duplicate ROW appears depends on a duplicate guard inside save_snapshot that Lens A could not see.)",
    "evidence": "Lens A S6-A-19: run_transition_phase.py:593 save_snapshot() is followed at :605-611 by the 't_now >= tmax' break with no state change in between; :860 then saves again after recomputing the same derived quantities at the same R2, v2, Eb, t_now; identically run_momentum_phase.py:675 -> :687-693 -> :894. Lens B mitigates: both files document that the past-rCloud counter increments 'only when the save actually wrote (duplicate guard in save_snapshot can skip)' (T:595-596 / M:677-678) - so save_snapshot is documented as silently skipping duplicates, which would suppress the duplicate row but is itself a documented silent no-op in the output path.",
    "expected": "Skip the final save when the loop broke immediately after an in-loop save, or dedupe on t_now explicitly rather than relying on an undocumented guard.",
    "failure_scenario": "",
    "repro": "Run any config to stop_t and check for two identical t rows at the end of dictionary.jsonl, and compare _snapshots_after_rCloud against the actual number of rows with R2 > rCloud.",
    "confidence": "low",
    "lenses": ["A", "B"],
    "divergence": "AB",
    "status": "contested",
    "source_ids": ["S6-A-19"]
  },
  {
    "id": "S6-R-35",
    "file": "trinity/phase1c_transition/run_transition_phase.py",
    "line": 389,
    "class": "deadcode",
    "severity": "S4",
    "claim": "The transition runner ships two diagnostic blocks with no counterpart in the momentum twin - a '--- PHASE BOUNDARY DIAGNOSTIC ---' block in the initialization path and an 'Ed diagnostic at first segment (quantify the original discontinuity)' block inside the segment loop - the second being an explicit admission that a discontinuity existed at the implicit->transition boundary.",
    "evidence": "Lens B S6-B-21: run_transition_phase.py:389-396 and :520; no counterpart in run_momentum_phase.py. Lens A corroborates the initialization block's content: 'v2_from_alpha = cool_alpha*R2/t_now - the latter is computed purely for the log line at :392-394 and then discarded'.",
    "expected": "Either the discontinuity is resolved and the diagnostics are removed, or a comment records the current residual magnitude and why the instrumentation stays.",
    "failure_scenario": "The phrase 'the original discontinuity' leaves unstated whether the discontinuity is fixed - which is precisely the property the min() model at :3/:200 claims to deliver. The blocks also run on the first segment of every sweep member.",
    "repro": "Read what the :520 block computes and whether its output is still non-zero for param/simple_cluster.param.",
    "confidence": "medium",
    "lenses": ["A", "B"],
    "divergence": "none",
    "status": "corroborated",
    "source_ids": ["S6-B-21"]
  }
]
```
