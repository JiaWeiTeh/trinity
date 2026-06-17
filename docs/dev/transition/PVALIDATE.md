# P-validate results — `cooling_or_blowout` vs `instantaneous` (default decision)

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
> a committed artifact (a CSV/table under `docs/dev/data/`, or a force-added
> harness/figure in the relevant `docs/dev/<workstream>/` folder) — never left in
> `/tmp` or an untracked `outputs/`. A future visit must be able to reproduce or
> compare against the numbers **without re-running**; record the exact config +
> command that produced each artifact.

**About this document**
- **Status (verified 2026-06-17):** ✅ **COMPLETE** — P-validate run on fresh
  self-consistent **hybr** runs. **All gates pass in both regimes.** The evidence
  **supports flipping the default** to `cooling_or_blowout`; the flip itself is a
  **maintainer decision** (production behaviour) and has NOT been made here.
- **Type:** results — execution log for P-validate (winner-vs-current gate
  scorecard, continuity handoff, macro deltas, default decision).
- **Workstream:** `transition/` — the implicit→momentum transition trigger.
- **Where it sits:** `TRIGGER_PLAN.md` (plan) → `P0.md` (P0/P-sens) →
  `pshadow-design.md` (design) → P-promote (shipped) → **this (P-validate
  results)**.
- **Code it concerns:** the live `cooling_or_blowout` trigger
  (`trinity/phase_general/transition_shadow.py`, the 1b terminator in
  `run_energy_implicit_phase.py`), harvested from fresh hybr runs.
- **Linked files & data:** configs `docs/dev/transition/harness/val_*.param`;
  data `docs/dev/data/pvalidate_*.csv` (+ `transition_steep_long.csv`, the P0
  instantaneous baseline); figures `docs/dev/transition/figures/pvalidate_*.png`;
  harness `docs/dev/transition/harness/make_pvalidate_figures.py`.

## The runs (fresh hybr, 2026-06-17)
| config | model_name | trigger | fate | reused? |
|---|---|---|---|---|
| dense-flat (α=0) | `val_dense_flat_cob_hybr` | cooling_or_blowout | cooling transition @0.197 → momentum | fresh run |
| **steep (α=−2)** | `val_steep_cob_hybr` | cooling_or_blowout | **blowout @2.728 → 1c → momentum** | **fresh run (the crux)** |
| steep (α=−2) | (`transition_steep_long.csv`) | instantaneous | energy-driven to 4 Myr (never transitions) | **P0 baseline** |

**Only steep needs a fresh run under the new trigger** — for flat configs F0 fires
first and F4 never, so `cooling_or_blowout ≡ instantaneous` (verified in source:
`test_flat_returns_cooling_balance_in_both_modes`). The steep `instantaneous`
baseline is the committed P0 run `transition_steep_long.csv` (subagent-verified
byte-identical to pre-promote behaviour). A fresh steep `instantaneous` run was
attempted but **hung on an integrator-hard segment** (the steep config does
"excess ODE work"); the P0 CSV is the equivalent baseline, so it was not needed.

Reproduce the figures (no sim re-run):
`python docs/dev/transition/harness/make_pvalidate_figures.py`

## Gate scorecard — all green
| gate | dense-flat (cooling) | steep (blowout) |
|---|---|---|
| routes 1b→1c→momentum | ✅ reaches momentum | ✅ blowout→1c→momentum |
| fires at the Eb-peak | ✅ F0 @0.197 = Eb-peak | ✅ F4 @2.728 (Eb still rising → boundary peak) |
| no reset (fires pre-surge) | ✅ (run ends pre-surge) | ✅ 2.728 < WR surge 3.178 |
| continuity handoff (no P jump) | ✅ | ✅ **P_drive −0.15%** |
| retained-energy ∈ [0.1, 0.25] | ✅ **0.126** | ✅ **0.115** |
| adiabatic null (cooling never fires at Lloss→0) | ✅ unit-locked | ✅ unit-locked |

## Continuity handoff (steep, 1b→1c at blowout t=2.728 Myr)
| quantity | last implicit (pre-switch) | first transition (post) | Δ |
|---|---|---|---|
| R2 [pc] | 23.529 | 23.546 | +0.07% |
| v2 | 8.511 | 8.514 | +0.03% |
| Eb | 7.03e7 | 6.84e7 | −2.7% (draining via 1c) |
| **P_drive** | **862.77** | **861.49** | **−0.15%** |

`Eb/R2/v2/P_drive` hand off with no injected jump — the continuity requirement
(plan §P-promote: "a slightly earlier/later switch must not inject a pressure
jump") is met.

## Macro deltas (steep, at t=4 Myr: new vs current)
| quantity | instantaneous (current) | cooling_or_blowout (new) | Δ |
|---|---|---|---|
| transition | never (energy-driven to 4 Myr) | **blowout @2.728 → momentum** | qualitative |
| R2 [pc] | 37.41 | 32.75 | **−12.5%** |
| v2 | 14.82 | 6.06 | **−59%** |
| pdot_total (injection *rate*) | 2.82e5 | 2.82e5 | 0% (same cluster) |

The effect is **dynamical, not energetic**: the feedback budget (`pdot_total`) is
identical; the new trigger stops the energy-drive at blowout, leaving the steep
bubble smaller and slower by 4 Myr. (The exact shell-*momentum* delta `mass·v2`
is not pinned — the P0 baseline CSV lacks `shell_mass`, and the fresh baseline
hung; the crux shell mass is 1.02e6 M☉. R2/v2 carry the story.)

## Figures (`docs/dev/transition/figures/`)
| figure | shows |
|---|---|
| `pvalidate_continuity_handoff.png` | Eb/R2/v2/P_drive across the blowout switch — smooth, no jump (continuity gate) |
| `pvalidate_trigger_compare_steep.png` | steep R2(t)/v2(t), new vs current — overlap to blowout, then diverge (−12.5% R2, −59% v2) |
| `pvalidate_steep_F0_broken.png` | steep: ratio_F0 never reaches 0.05 (current trigger can't fire) while R2/rCloud crosses 1 (F4 fires) |
| `pvalidate_retained_scorecard.png` | retained-energy at firing vs the η≈0.1–0.25 band — new lands in-band both regimes; current can't fire for steep |
| `pvalidate_phase_timeline.png` | phase outcome per trigger — current leaves steep energy-driven forever; new routes it transition→momentum |

## Default decision (maintainer)
The pre-registered flip condition (plan §P-validate) is met: `cooling_or_blowout`
fires at the Eb-peak without reset across **both** regimes, and the
retained-energy cross-check lands in the literature band (0.115 / 0.126). So **the
evidence supports flipping the default**. Per the plan this is a maintainer call;
the default is still `instantaneous` and the new trigger remains selectable until
the maintainer decides.

## Caveats / honesty notes
- **Adiabatic-null scope:** the gate binds the *cooling* criterion only. F4 blowout
  is Lloss-independent by design and *will* fire at Lloss=0 — see the unit test
  `test_adiabatic_null_cooling_never_fires`. (Full pure-adiabatic Weaver-run
  validation still needs scaffolding: no cooling-off param, no Weaver analytic
  test exists.)
- **Cost reality:** hybr is integrator-hard on the steep config. The crux reached
  blowout (2.728 Myr) in ~40 min wall; the `instantaneous` baseline (must grind to
  4 Myr through the harder post-blowout regime) hung — hence the P0-CSV fallback.
- **Legacy solver is NOT faster here:** a legacy steep preview ran *slower* than
  hybr (contaminated trajectory anyway); abandoned.
