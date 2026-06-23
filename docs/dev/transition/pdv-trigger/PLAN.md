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

**Date:** 2026-06-23. **Branch:** `feature/PdV-trigger-term`. **Status:** scoping + evidence — **no
production change yet.** This note answers the maintainer's question ("add a PdV term to the
transition trigger — what was the argument against it, and is it still valid for larger clusters?")
and lays out how to plan/test it. Sibling priors (re-verify per banner): `../pt4/TRANSITION_FIX_SCOPING.md`
(Route 1), `../pt4/r1shadow/R1_FINDINGS.md`, `../../failed-large-clouds/PLAN.md` §6.

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
   integrated from `Edot_from_balance = Lmech − Lloss − 4πR2²·v2·Pb` (`get_betadelta.py:434`), which
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
| **Energy evolution** `get_betadelta.py:434` (`Edot_from_balance`), stored `residual_Edot2_guess` (`run_energy_implicit_phase.py:846-847`) | `Lmech − Lloss − 4πR2²·v2·Pb` | **yes** | how `Eb` actually evolves — PdV already drains the reservoir |
| **`cooling_balance` trigger** `run_energy_implicit_phase.py:1200` | `(Lmech − Lloss)/Lmech < 0.05`, `Lloss = bubble_LTotal (+leak)` | **no** | the default energy→momentum handoff; pure radiative |
| **`ebpeak` trigger** (opt-in) `evaluate_r1_shadow` `:208-209`, shadow `:1166-1184`, drive `:1192-1198` | `Edot_from_balance ≤ 0` | **yes** | "PdV in the trigger" — the net-energy turnover; default-off |

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

### Step 1 — Decisive new measurement: map the `PdV/Lmech = 1` boundary across the science grid
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

## Artifacts
- `data/pdv_regime_budget.csv` (+ `data/make_pdv_regime_table.py`) — the §Evidence table.
- Upstream (committed): `../cleanroom/data/c0_*_h0.csv`, `../../failed-large-clouds/data/budget_*.csv`,
  `../pt4/r1shadow/r1_shadow_summary.csv`.
</content>
</invoke>
