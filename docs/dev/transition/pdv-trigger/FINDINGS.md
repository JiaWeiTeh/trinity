# PdV-trigger workstream — findings (verified, 4/4 live configs)

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

What looks like "three ways to boost cooling" is really **two cooling-magnitude approaches on opposite sides
of the structure solve, plus a separate trigger axis**. The key disambiguation: **"modify cooling like
El-Badry with κ" and "modify the conduction front k_f" are the *same* knob** (`cooling_boost_kappa`) — raising
the conduction coefficient *is* the 1D stand-in for more radiating surface / mixing. Every row is read from
source (knob `registry.py`, equation file:line); no assumptions.

| axis / approach | knob | what it changes (from source) | θ: imposed or **emergent**? | literature | status / verdict |
|---|---|---|---|---|---|
| **A. Outcome-side** — operate on `L_loss` *after* the structure solve (`effective_Lloss`, `get_betadelta.py:334`) | | | | | |
| · scalar multiplier | `cooling_boost_mode=multiplier`, f_mix | `L_loss = L_leak + f_mix·L_cool` (`:354`) | scaled (semi-imposed) | — | no single f_mix fires across density (1.4–3.8) |
| · **θ-target floor** ("sum like Lancaster θ") | `cooling_boost_mode=theta_target`, θ | `L_loss = max(L_cool+L_leak, θ·L_mech)` (`:356`) | **imposed** (top-down) | **Lancaster** θ≈0.9 | degenerate: constant θ=0.95 *is* the 0.95 trigger; θ(Da) refuted |
| **B. Mechanism-side** — operate on the conduction *inside* the structure; θ comes out | | | | | |
| · **κ_eff conduction multiplier** ("El-Badry κ" **=** "modify k_f / conduction front" — same knob) | `cooling_boost_kappa`, f_κ | `κ_eff = f_κ·C_th·T^(5/2)` at 3 sites (`bubble_luminosity.py:291/370/406`) → thicker front → more 10⁵–10⁶ K gas (more surface/mixing) | **emergent** (bottom-up) | **El-Badry** mixing (λδv↔κ_eff) | built/gated; f_κ≈4 (compact)…~60 (diffuse); side-effect: dMdt↑ |
| **C. Trigger-side** — *when* to transition, not *how much* it cools | | | | | |
| · PdV-inclusive trigger | `transition_trigger=ebpeak` | fire when `L_gain−L_loss−PdV ≤ 0` (`run_energy_implicit_phase.py:198,1206`) | n/a (timing) | El-Badry/Lancaster "cooling creeps up" | doesn't fire alone at f_κ=1; assist not substitute |

**A** imposes the result (Lancaster's θ lives here); **B** changes what *produces* the cooling so θ emerges
(El-Badry lives here, and it is the *same* knob as "modify the conduction front"); **C** is a different axis
(the transition criterion). **A and B must never be stacked** — the `max(·)` closure (§2 of the report) keeps
the loss single-count. Current direction = **B** (κ_eff), calibrated to a density-dependent target, with **C**
(PdV) as an optional timing assist.

**f_κ vs f_mix (row B vs row A — not the same knob; report §14).** f_mix scales the *output* L_cool
(`L_loss=L_leak+f_mix·L_cool`); f_κ scales the *conduction coefficient* (`κ_eff=f_κ·C_th·T^(5/2)`, L_cool
emerges). **(i) Exact by definition:** f_mix IS the L_cool multiplier, so `f_mix_equiv(f_κ) ≡
L_cool(f_κ)/L_cool(f_κ=1)` at matched-t — measured `f_κ=2 ⇒ f_mix_equiv 1.23–1.50` (developed→seed).
**(ii) The exponent is bracketed, not derived:** L_cool's effective exponent q (f_mix_equiv=f_κ^q) runs
~0.58 (seed)→~0.30 (developed), and the θ_blowout leverage (noisier — different t per f_κ) scatters 0.21–0.42.
The one *exact* anchor is `dMdt∝f_κ^(2/7)` (Weaver Eq 33), and the developed L_cool boost lands on it
(1.228≈2^(2/7)=1.219) — the mixing-layer relation `L~ṁ·enthalpy` (El-Badry/Fielding/Tan-Oh-Gronke). So
`f_mix_equiv≈f_κ^q`, q~0.3–0.6 bracketed below by 2/7, ⇒ `f_κ≈f_mix^(1/q) ≫ f_mix` (why §3's firing f_mix
1.4–3.8 and §13's firing f_κ ~4–60 look so different). **It is a heuristic magnitude map, NOT a closed-form
law.** **(iii) Deeper:** f_mix touches only L_cool; f_κ self-consistently raises dMdt∝f_κ^(2/7), drains Eb,
lowers Pb (θ emerges) — so even at matched L_cool the trajectories differ. **Magnitude-only; not interchangeable.**

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

> **Update (the merge, 2026-06-26):** the "density-dependent target" is now concrete — the **mechanism** is
> **κ_eff** (`cooling_boost_kappa`, **Rung A, built/gated**), which raises the *emergent* cooling in-structure
> (§6 — measured `bubble_LTotal` ×1.23–1.38); the **target** is `θ(n_H)` (El-Badry `λδv`=κ_eff + Lancaster);
> the **knob** is `f_κ(properties)`. So the goal is **enhanced, density-dependent cooling matched to obs/3D**,
> delivered by *calibrating* f_κ — not by a scalar floor and not by chasing evaporation suppression. See
> `PLAN.md` ⭐ synthesis and `RUNGB_SCOPING.md` §2a (the canonical θ/`λδv`/`f_κ`/0.95 reconciliation).

---

## 1. [data] Boost needed to reach the 0.95 trigger rises steeply as density falls

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

## 2. θ_lit(n) figure — `theta_vs_density.png` — SCHEMATIC overlay, gap NOT quantified

TRINITY's resolved `L_cool/L_mech` (= 1 − `cool_at_blowout`) vs ambient nCore is **real [data]** and rises
**0.250 (1e2) → 0.697 (1e6)**. The literature overlay is **[schematic / to-verify]**: El-Badry+2019
(arXiv:1902.09547) and Lancaster+2021 (arXiv:2104.07722) PDFs returned HTTP 403, so the band is an
arbitrary saturating stand-in, NOT digitized θ(n).

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
at ×3. Consistent with §1; confirms live that no constant `f_mix` fires cooling across the range.
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
  (diffuse, extrapolated)** — steeply density-dependent (only compact reaches 0.95 within the measured f_κ≤4
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
  *compact* case (fires by f_κ~2–4). **Net:** PdV (`ebpeak`) is an assist for transition *timing*, **not a
  substitute** for `κ_eff` (cooling *magnitude*) — complementary, downgraded from "PdV alone fixes f_κ~60."
  Artifacts: `data/ebpeak_trigger_test.csv` + `ebpeak_trigger_test.png` (+ `data/make_ebpeak_trigger_test.py`).
  No production code touched (default `transition_trigger=cooling_balance` unchanged).
- **8-config coverage [data]:** the f_κ=1 conclusion above is **2 live configs**, but it **generalizes to all 8**
  via the earlier frozen-trajectory screen (`make_ebpeak_8config_xcheck.py` → `ebpeak_8config_xcheck.csv/png`):
  all **6 normal** configs peak at PdV-inclusive **0.85–0.92** and never fire (only heavy-5e9 `fail_repro` 1.57
  and the `small_1e6` control 1.11 do; `large_diffuse_lowsfe` 1.02 barely, post-blowout). **Live-vs-frozen
  agrees to the digit** (simple_cluster live 0.911 == frozen 0.911). The f_κ-*dependence* (trade-off) is
  live-only and extended to `mid`=midrange_pl0 (running) + `dense`=small_dense_highsfe (stalled — nCore 1e6 is
  numerically stiff; frozen point used). HPC-deferred for the remaining configs.

## 7. Provenance
- Commits (`feature/PdV-trigger-term`): `6642ff4` matrix+comparator, `dc1c2fd` note patches, `17f9653`
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
