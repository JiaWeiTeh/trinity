# pdv-trigger workstream — master index (START HERE)

> ⚠️ **This document may be out of date — verify before trusting it.** It is a point-in-time map, not a maintained
> spec; the code and sibling docs move. **Re-check each row against the actual file before relying on it.**
>
> 🔄 **Living index — update on every visit.** When you add/rename/retire a doc or finish a task, update the tables
> below (and date it). Keep all banner paragraphs at the top.
>
> 💾 **Persist diagnostics — commit, don't re-run.** Every result has a committed builder + CSV + figure; see
> `REPRODUCE.md` for the result→command→artifact manifest.
>
> 🔗 **Cross-check the sibling docs.** This index is the hub; when a sibling's number/status changes, reconcile it
> here too. Never let two docs disagree.

---

## 0. What this workstream is (one paragraph)

TRINITY transitions a feedback bubble from **energy-driven** to **momentum-driven** when interface cooling drains
the mechanical luminosity (the `cooling_balance` trigger at θ = L_cool/L_mech ≥ 0.95 — the default
`transition_trigger`, threshold `phaseSwitch_LlossLgain=0.05`). This workstream asks: *what sets θ, what raises it
to the obs/3D values (Lancaster θ~0.9–0.99), and how does it depend on cloud properties?* TRINITY's native 1-D
θ under-shoots that band for most clouds, so realistic GMCs never fire the trigger — that is **the problem**.
**Current direction (corrected 2026-07-01):** θ is an **OUTPUT, not an input** — boost the cooling **mechanism**
and let the solved bubble produce θ, with El-Badry's closed form θ(λδv, n) and Lancaster's 0.9–0.99 band as the
**calibration target**; use a **single physical f constant** (not f(n)); **accept route-a** (→ nuanced 2026-07-02: theta5 showed the
canonical diffuse GMC *fires* at f_mix=4; route-a = `small_1e6` + `fail_repro`, a **θ₀-set** boundary, not a
density-set one — §10); massive/PdV clouds ride the PR #715 `Eb≤0→momentum` handoff. The tentative mechanism knob is
**`cooling_boost_mode='multiplier'`** (stable, radiative-only) — **its calibration was re-derived 2026-07-02**
(the 📏 theta5 matrix on Helix; `FINDINGS.md §10`, `runs/data/theta5_calibration.csv`): θ₁-collapse law
**f_fire ≈ 1.4·(0.95/θ₀)^1.82**, and **f_mix=4 fires the whole normal-GMC band** — **f_mix = 4 ADOPTED
(2026-07-02 maintainer ruling: momentum-then-recollapse is acceptable physics)**; theta5b refines the
workable window for the referee statement. The earlier θ₀/p had been fit
on the *other* knob, `cooling_boost_kappa`, which breaks down at f_κ=8 (FINDINGS §8e).
*History:* PdV question → f_κ Rung-A → structural κ_mix Rung-B (**shelved**, saturates) → impose-El-Badry-θ
detour (**demoted to opt-in**, double-counts PdV) → emergent-θ correction + knob correction (2026-07-01) →
**merged the parallel pt3 line** (819-run sweep θ₁-collapse law + `cooling_boost_kappa='auto'`, **PROVISIONAL**)
→ **theta5 matrix ran** (2026-07-02, §10).
Production changes to date (all gated/behavior-neutral): `cooling_boost_{mode,fmix,theta,kappa}` (+ pt3 `'auto'`
resolver), PR #715 handoff, Pb-collapse guard, `_MINT_LOG_TOL` log gate.

## 1. Read in this order (orientation) — updated 2026-07-01 (post pt2+pt3 reconciliation)

1. **this file** — the map (incl. the §1.5 staleness audit below).
2. **`CONTAMINATION.md`** — ⛔ what you may and may not quote (rules (a)–(e), the full artifact register,
   the §8e-vs-§9 tension resolution). **Read before quoting ANY number from this workstream.**
   **+ `MANIFEST.md`** — *which version is this file?* Generated ledger (`python make_manifest.py`): per
   artifact, last-updated date + commit, its producing script, and a ⚠️ STALE-RISK flag whenever a builder
   changed after its committed output (an output from an older builder version). Recency ≠ quotability —
   use MANIFEST for "is this current?", CONTAMINATION for "may I quote it?". **Regenerate the manifest in
   the same commit as any artifact change.**
3. `PLAN.md` → the **⭐⭐ CANONICAL SYNTHESIS + VERDICT** block (the current direction; supersedes all earlier
   synthesis) + the dated status ledger (newest first).
4. `FINDINGS.md` §8c (why enforce-θ was demoted), §8e (knob validation: kappa breaks; multiplier tentative),
   §9 (the pt3 819-sweep + θ₁-collapse law + 'auto', with post-merge flags), **§10 (the theta5 matrix,
   ran 2026-07-02 — the current headline result)**.
5. `ELBADRY_REFERENCE.md` + `LANCASTER_REFERENCE.md` — 📌 the two **imprint** reference docs (θ definition, the
   closed form, λδv≈3, the n-mapping, PdV) — the **calibration target** for emergent θ, not an enforced value.
6. `runs/README.md` **📏 STANDARD PROTOCOL** — the 8-config × ≥5 Myr × θ_max harness every new claim runs through.
7. `REPRODUCE.md` — result → `.param`/command → artifact manifest.

## 1.5 ⚠️ STALENESS AUDIT — docs that describe SUPERSEDED directions (read before trusting a conclusion)

The direction was **corrected on 2026-07-01** back to **emergent θ via a mechanism boost** (calibrate to
El-Badry/Lancaster) after the intermediate "**impose** El-Badry's θ" avenue (2026-06-30) was shown to
double-count PdV on massive clouds (`FINDINGS.md §8b/§8c`). The same day, the **KNOB CORRECTION** voided the
§14 validation (fit on `kappa`, run with `multiplier`) and §8e broke the `kappa` knob at f_κ=8. Several docs
predate one or more of these pivots and, read in isolation, point the wrong way. **Kept for provenance, flagged here:**

| doc | what's STALE in it | the correct current view |
|---|---|---|
| any doc/banner saying **"impose El-Badry θ as the trigger target"** is the direction (incl. earlier revisions of `PLAN.md`, `FINDINGS.md` taxonomy banner, `THETA_ELBADRY_SPEC.md` framing) | the *enforce-θ* framing | **demoted to an opt-in override** (2026-07-01) — it double-counts PdV (`FINDINGS.md §8b/§8c`). Direction = **emergent θ**, El-Badry as calibration target. |
| `F_KAPPA_FUNCTIONAL_FORM.md` **§14 numbers** (θ₀, p, f_κ_ideal, n_routeA; the θ_max=1.334/1.006 "validation") | blowout-θ₀ + kappa-fit + contaminated observer | **no §14 number is production-grade** (`CONTAMINATION.md` ⛔ #1–#2); the *program* (emergent θ, physical cap, route-a) stands; **re-derived 2026-07-02 → `FINDINGS.md §10` / `runs/data/theta5_calibration.csv`** |
| `FINDINGS.md` mid-doc "→ Calibration target (2026-06-29)" banner (f_κ(n_H)≈1.4×10²·n^−0.30; 48/9/3) | pre-sweep slope + pre-DECISION f_κ(n) framing | slope measured −0.60 (scorecard P1 ❌); superseded by the **single-constant DECISION** + the §9 θ₁-collapse law |
| `KAPPA_VALIDATION_PLAN.md` | its banner called the §8e⇄§9 kappa result an "**open tension with FINDINGS §9**" (⚡ #1) | that tension was **RESOLVED same-day** (`FINDINGS.md §9a`, `data/kappa_stability_map.csv`: non-monotonic breakdown windows); banner updated 2026-07-02. (Earlier staleness — "T3 ⏳ running", the 0.99/0.91/0.55 column mislabelled "the multiplier predictions" — was already fixed in-doc: T1–T5 all DONE; that column is the §14 **kappa-fit model** prediction, the multiplier runs the invalid (R5) *validation*.) |
| `SESSION_HANDOFF_2026-07-01.md` | frozen speculation snapshot (its own 🚨 banner); §5.2 re-anchors the θ peak on "~blowout" | historical record of the pt2 session — do not mine it for numbers; θ-peak epoch is config-dependent (`CONTAMINATION.md` ⚡ #3) |
| `RUNGB_SCOPING.md` | the **structural κ_mix injection** ("re-promoted", §8 gated production) as the path | the structural port is **SHELVED** (saturates/unstable, `KMIX_SELFCONSISTENT.md`); κ_mix survives only as physical *justification* for θ∝√(λδv·n) |
| `KMIX_SELFCONSISTENT.md` §2 | "dense θ plateaus low (~0.35) / only 1/6 fires" | **WALKED BACK** — wrong epoch (blowout) + buggy port; El-Badry+Lancaster agree **dense θ is HIGH (0.9–0.99)**. See §2b and `LANCASTER_REFERENCE.md` §7. |
| `KMIX_DIFFUSIVITY.md` / `KMIX_PROTOTYPE.md` | "calibrate λδv to Lancaster (value open)"; prototype Pb anchors from **0.3–1.0 Myr truncated** runs | **λδv≈3 is now pinned** (`LANCASTER_REFERENCE.md` §7); re-derive prototype Pb from ≥5 Myr runs before quoting numbers |
| `KMIX_IMPLEMENTATION_SPEC.md` | the κ_mix-into-the-ODE wiring design | **SHELVED** (banner in the doc); its dimensionless-multiplier *units* strategy is still reusable |
| `runs/README.md` §9-era verdicts | "heavy clouds collapse regardless" (fail_repro) | **pre-PR#715** dead-stop artifact — post-merge record is `data/newcode_default_vs_theta.csv` |
| any "Lancaster 2021c / ApJ 914, 91" / "ApJ 914,90 = theory" | paper-ID confusion | ApJ 914, **90 is Paper II (sims)** — the θ~0.9–0.99 anchor; see `LANCASTER_REFERENCE.md` §0 |

**Rule going forward (maintainer): whenever a decision is made, update the ⭐⭐ canonical synthesis AND this
audit AND `CONTAMINATION.md` AND the affected sibling together — never one in isolation.**

## 2. The docs — timeline, role, purpose, status

Eras (same labels as `CONTAMINATION.md`): **E1** PdV/f_mix screens+live edges (06-24→25) · **E2** κ_eff Rung-A/FM
probes (06-26→27) · **E3** kappa blowout-cal + ebpeak (06-28) · **E4** 819 sweep (ran 06-29; folded 07-01 from pt3)
· **E5** κ_mix Rung-B (06-29→30, shelved) · **E6** impose-θ detour (06-30→07-01, demoted) · **E7** PR#715 +
direction/knob/θ_max corrections + pt2⇄pt3 reconciliation (07-01) · **E8** theta5 protocol era (07-02→).

| doc | added | era | what it is meant to do | status |
|---|---|---|---|---|
| `PLAN.md` | 06-24 | all | living plan, ⭐⭐ synthesis, dated status ledger (the hub) | **live** |
| `NOTE_PATCHES.md` | 06-24 | E1 | the Paper-II note patches: don't-double-count, the f_mix convention fix | settled |
| `FINDINGS.md` | 06-25 | all | the verified findings ledger (§1–§9) + the 3-axis taxonomy | **live** |
| `KAPPA_EFF_SCOPING.md` | 06-25 | E2 | κ_eff Rung-A feasibility map + back-reaction result | settled |
| `RUNGB_SCOPING.md` | 06-26 | E5 | structural κ_mix scoping | 🛑 **SHELVED** (§1.5) |
| `REPRODUCE.md` | 06-28 | all | result→param→command→artifact map (now #1–#28) | **live** |
| `F_KAPPA_FUNCTIONAL_FORM.md` | 06-29 | E3/E4 | the f_κ / emergent-θ calibration program; §11–13 "don't-force-it"; §14 the (voided) calibration | **live — re-derivation DONE 2026-07-02 (FINDINGS §10); §14 numbers remain void** (§1.5) |
| `KMIX_DIFFUSIVITY.md` | 06-29 | E5 | maintainer manuscript draft verified; λδv origin | live (λδv recipe retired) |
| `KMIX_PROTOTYPE.md` | 06-29 | E5 | offline κ_mix go/no-go | live (⚠️ truncated anchors) |
| `KMIX_IMPLEMENTATION_SPEC.md` | 06-30 | E5 | κ_mix wiring design+units spec | ⏸ **SHELVED** (units strategy reusable) |
| `KMIX_SELFCONSISTENT.md` | 06-30 | E5 | κ_mix in the real solver (monkeypatch): saturation, §2b time-resolved | live (dense-low walked back) |
| `ELBADRY_REFERENCE.md` | 06-30 | E6→ | 📌 El-Badry+2019 distilled (every eq/number) | **live** (imprint) |
| `LANCASTER_REFERENCE.md` | 06-30 | E6→ | 📌 Lancaster distilled (θ~0.9–0.99, λδv≈3, route-a) | **live** (imprint) |
| `THETA_ELBADRY_SPEC.md` | 06-30 | E6 | the gated `theta_elbadry` mode spec (never merged to `trinity/`) | live (**opt-in override**; demoted 07-01) |
| `PB_COLLAPSE_GUARD_FIX.md` | 06-30 | E7 | Pb hygiene fix — applied + tested | **done** |
| `HIMASS_HANDOFF_PLAN.md` | 06-30 | E7 | the high-mass Eb≤0 dead-stop diagnosis → PR #715; deferred items (1a routing, pressure-crossover event) | **partially shipped** (PR #715 ✅; rest ⏳) |
| `ELBADRY_THETA_STORY.html` | 07-01 | E6/E7 | illustrated 9-chapter walkthrough incl. the correction chapter | narrative snapshot |
| `KAPPA_VALIDATION_PLAN.md` | 07-01 | E7 | the T1–T5 correct-knob validation working plan | **completed** (all done; see §1.5 for its stale lines) |
| `SESSION_HANDOFF_2026-07-01.md` | 07-01 | E7 | the pt2 session handoff (self-declared speculation + retractions R1–R6) | historical snapshot |
| `CONTAMINATION.md` | 07-01 | E7 | ⛔ the register: rules (a)–(e), per-artifact status, tensions (⚡ #1 resolved) | **live — read first** |
| `INDEX.md` (this file) | 06-30 | all | the map | **live** |

**Precursors one level up** (`docs/dev/transition/`): `P0.md`, `TRIGGER_PLAN.md`, `pshadow-design.md` — the
pre-pdv-trigger trigger-characterization story (F0–F5), each self-bannered ⛔ SUPERSEDED; and
**`PROVENANCE_PROTOCOL.md`** — the clean-baseline / separate-process contamination guard this workstream inherits.
Shared tooling: `../harness/` (`run_stamped.py` provenance-stamped launcher, `harvest.py` P0 trigger harvester);
the canonical 8-config base params live in `../cleanroom/configs/` (6) + `runs/make_theta5_params.py` (all 8).

## 3. The live thread — close the calibration loop on the RIGHT knob (updated 2026-07-01)

**Hard guardrail (maintainer): no production change before testing all 8 configs (📏 protocol: ≥5 Myr,
θ_max, separate processes).** The 8: `simple_cluster`, `midrange_pl0`, `be_sphere`, `pl2_steep`,
`large_diffuse_lowsfe`, `small_dense_highsfe` (6 cleanroom) + `fail_repro` (heavy 5e9) + `small_1e6` (control).

| step | what | status | where |
|---|---|---|---|
| direction | θ is an OUTPUT; mechanism boost; El-Badry/Lancaster = calibration target; single physical f; route-a | ✅ decided 07-01 | PLAN ⭐⭐, FINDINGS §8c |
| knob choice | `multiplier` tentative (kappa breaks @8 + slow; κ_mix shelved; theta_target double-counts) | 🟡 tentative | FINDINGS §8e |
| re-derive the `multiplier` calibration | **RAN on Helix 2026-07-02, 32/32 compliant**: θ₁-collapse law f_fire ≈ 1.4·(0.95/θ₀)^1.8; **f_mix=4 fires the whole normal-GMC band incl. the diffuse cloud** (blowout had under-read diffuse θ by 2×); route-a = small_1e6 + fail_repro | ✅ **done** | `FINDINGS.md §10`, `runs/data/theta5_calibration.csv` |
| pin the single f_mix | **✅ f_mix = 4 ADOPTED (2026-07-02)** — maintainer ruling: momentum-then-recollapse is acceptable physics (an outcome, not a failure); still pathological: f=8 Eb-drain-without-firing + dense-edge NaN (ticket open) | ✅ decided | PLAN ledger 07-02 ruling; `FINDINGS.md §10` |
| referee defense: "why exactly 4" + "why a constant" | fine-bracket matrix **committed & validated** (`runs/params/theta5b/`, 43 arms: f∈{2.5,3,3.5,4.5,5}×8 + diffuse stop_t=8); model-comparison table + out-of-sample law test designed | ⏳ **READY — run theta5b on HPC** (`runs/run_theta5b.sbatch`) | PLAN "REFEREE DEFENSE"; `pdvtrigger_report.html` §16.2 |
| resolve the §8e⇄§9 tension | both right: kappa breakdown is **non-monotonic in f_κ** (dead windows between firing bands; §8e's θ≈0.53 freeze reproduced on Helix) | ✅ resolved 07-01 from committed data | `FINDINGS.md §9a`, `data/kappa_stability_map.csv` |
| revalidate `'auto'` (pt3) | re-measure the 63-cell grid under the 📏 protocol (5 Myr, θ_max) or keep 'auto' opt-in-provisional | ⏳ open | FINDINGS §9 flags |
| acceptance target | normal-GMC-band clouds fire `cooling_balance` with emergent θ_max ∈ 0.9–0.99 and reach momentum; route-a clouds stay energy-driven **by design** (the boundary = the falsifiable output). **Measured 2026-07-02 (§10 point 4): the boundary is θ₀-based, NOT a clean density threshold** — small_1e6 (n=100) never fires through f=8 while large_diffuse (same n=100) fires at f=4 | the goal (boundary now measured) | LANCASTER_REFERENCE §7 + FINDINGS §10 |
| massive clouds | ride PR #715 `Eb≤0→momentum`; θ knobs must NOT touch them (§8b lesson) | ✅ shipped | HIMASS_HANDOFF_PLAN |

## 4. Data & figures

~50 builders + ~45 CSVs + ~45 figures under `data/`, `runs/data/` and the folder root. The canonical map is
**`REPRODUCE.md`** (result #1–#28 → builder/param/command/artifact); the quotability of every artifact is graded
in **`CONTAMINATION.md`** (do not quote a number without checking it there first). The HPC artifacts are the
819-combo sweep (`data/summary.csv` + reduction) and the 📏 `theta5` matrix (ran 2026-07-02;
`runs/data/theta5_{summary,calibration}.csv`).

## 5. Branch archaeology (which line holds what — mapped 2026-07-01)

| branch | state | contents |
|---|---|---|
| `feature/PdV-trigger-term-pt2` | **merged** via PR #717 → `main` (154e4da) | the whole pt2 session (E1–E7 docs/data above) |
| `feature/transition-trigger-pt3` | **merged into this line** (d222883) | the 819-sweep fold-in (`ca3b4c7`) + `cooling_boost_kappa='auto'` (`01b9616`) — written in parallel, without pt2's same-day corrections; reconciled in FINDINGS §9 + CONTAMINATION.md |
| `feature/PdV-trigger-term` (pt1) | stale, 4 remaining unmerged doc commits | `HANDOFF.md` (superseded by the pt2 handoff), storyline f_κ↔f_mix sharpening (superseded by F_KAPPA §14 framing). **`3e68143` (El-Badry-PDF eq verification + θ_1D overlay) was cherry-picked into this line 2026-07-01** (`elbadry_overlay.{csv,png}`, FINDINGS §2 update) |
| `fix/transition-trigger-pt2/pt3/pt3-figs` (06-16/17), `fix/transition-trigger-problem-pt4` (06-24) | ancient (≥122 behind main) | earlier transition-trigger attempts predating this workstream — historical only, do not develop on them |

*Index rewritten 2026-07-01 during the pt2⇄pt3 reconciliation. Update §2/§3/§5 whenever a doc, step, or branch changes.*
