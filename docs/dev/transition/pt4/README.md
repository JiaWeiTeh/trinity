# pt4 — the transition-trigger problem: why it never fires under `hybr`, and what that means

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

**Date:** 2026-06-22. **Branch:** `fix/transition-trigger-problem-pt4`.
**Scope:** read-only investigation of H1–H4 — those four experiments changed **no production code**
(each is a monkeypatch / config / committed CSV under this folder). `pytest test/test_unit_conversions.py` green.

> **Update (2026-06-23):** the **R1** follow-up *did* ship production code — an **opt-in, default-off**
> `transition_trigger` keyword (`run_energy_implicit_phase.py`) wiring a blowout + PdV-inclusive
> **`ebpeak`** event as an inert shadow plus an opt-in drive, **byte-identical** with the default
> `cooling_balance`. The "no production code changed" scope above covers **H1–H4 only**; see
> `r1shadow/R1_FINDINGS.md`. The default flip and the heavy-cloud (phase-1a) Eb-peak handoff remain unbuilt.

This is the entry point tying together four hypotheses (H1–H4). Each has its own detailed doc;
this README is the synthesis + figure index. The sibling priors (`../cleanroom/FINDINGS.md`,
`../../failed-large-clouds/PLAN.md`) are reconfirmed and sharpened here — treat them as unverified.

---

## The question

Under the default solver `betadelta_solver=hybr`, the implicit→momentum cooling-balance trigger
(`trinity/phase1b_energy_implicit/run_energy_implicit_phase.py:1095`)
`(Lgain − Lloss)/Lgain < 0.05` **never fires (0/6 configs)** — runs stall in the implicit energy
phase to the 15 Myr cap. The maintainer challenged the prior "geometric blowout, not a bug"
conclusion with two hypotheses (H1, H2) and two follow-up experiments (H3, H4).

## Result in one sentence

**The trigger never fires because there is no cooling-balance event to fire on** — the
energy-conserving Weaver/Rahner interior is intrinsically too hot (T0 ~10⁶ K) to radiate enough.
It is **not** a `Lcool` bug (H1) and **not** the rCloud boundary (H2); the transition is
**geometric / profile-dependent**, which a single hardcoded scalar (0.05) cannot express. The
normal-cloud **stall** and the 5e9-cloud **crash** are **opposite extremes of one control
parameter, `PdV/Lmech`** (H3, H4).

## The four hypotheses and their verdicts

| # | hypothesis (maintainer) | verdict | key evidence | figures |
|---|---|---|---|---|
| **H1** | `Lcool` is a bug that "keeps surging up" | **Refuted as a bug** | Cooling integral byte-identical across every refactor. `Lcool` surges ~2× early (the "surging up") *then* collapses 4–9× (`n²V` dilution). Non-firing is the unbounded hybr β→+4 under-cooling, **not** the integral: legacy clamps β∈[0,1]→5/6 fire; hybr free β→0/6 fire. Ratio floors 0.28–0.49. | `h1_lloss_surge_collapse`, `h1_beta_clamp_divergence`, `h1_ratio_min_stats` |
| **H2** | "breaking rCloud = fail" / make rCloud infinite | **Refuted as a fix, vindicated as a diagnosis** | `R2 > rCloud` is a *clean phase switch* (`phase_events.py:218`, `is_simulation_ending=False`), not a fail. Cooling is set by **local density**, not rCloud proximity — matched-R2 test: identical ratio (0.4845) at the same absolute R2 despite 5.2× rCloud. Enlarging the cloud can't push the ratio below 0.05. | `h2_ratio_vs_rcloud`, `h2_matched_r2`, `h2_dip_vs_density_gradient` |
| **H3** | floor `Eb>0` so the bubble keeps expanding | **Eb-collapse is NOT the sole failure mode (refuted)** | Bit-identical no-op on 11/13 healthy/stall configs (Eb never collapses there). On 5e9 it does *not* rescue: `fail_repro` exposes the non-convergent implicit grind (R2 stuck); `fail_helix` hits a 2nd guard (`run_energy_phase.py:169`, "xi out of bounds" as Eb→0). | `h3_ebfloor_noop_and_grind` |
| **H4** | cap PdV at `min(PdV, κ·Lmech)` for an early window | **Survivable transient, not stillborn — but the rescue is non-physical** | PdV/Lmech stays super-critical (>1) for ~1.5–4.5e-3 Myr. tw=1e-3 too short (collapses); tw=3e-3 *splits* the clouds (helix self-sustains, repro re-collapses); tw=1e-2 self-sustains both — but only by injecting ~10× the proposed window of non-conserved energy. No-op on controls (max rel\|ΔEb\| ≤ 2.6e-5). | `h4_Eb_sweep_*`, `h4_pdvratio_sweep_*`, `h4_summary`, `h4_control_vs_collapse` |

## The connecting picture (the maintainer's "is there a connection?")

Yes — the transition-trigger stall and the failed-heavy-cloud crash are the **same energy budget at
opposite extremes**, sorted by `PdV/Lmech = 4πR2²·Pb·v2 / Lmech` (how fast the shell expands vs how
hard the wind drives) — see `h4_control_vs_collapse`:

| regime | `PdV/Lmech` | `Eb` behaviour | outcome |
|---|---|---|---|
| normal clouds (the stall) | **< 1** (decelerates, Weaver-like) | grows monotonically, never peaks | trigger never fires → **stall** |
| 5e9 clouds (the crash) | **> 1** (near free-expansion) | drained through zero | **`ENERGY_COLLAPSED`** (shipped guard) |

A **PdV-inclusive net-energy zero-crossing (the Eb-peak)** is the one criterion that unifies them:
it fires immediately for heavy clouds and *never in-cloud* for normal ones (5/6 have no Eb-peak), so
for normal clouds the operative transition stays **blowout**. H3/H4 confirm the heavy end needs
**momentum-driven continuation or added cooling**, not an energy/PdV prop — every artificial
"survives" is bought with non-conserved energy.

## The density-drop mechanism (the maintainer's direct question)

The cooling ratio bottoms out where the shell hits the **first steep density gradient** —
`Lloss ∝ n²V` turns over there. For flat (α=0) clouds that gradient is the cloud edge; for the steep
`pl2_steep` (α=−2) profile it is `rCore` (≈5% of rCloud), deep inside. The dip location tracks the
in-cloud density steepness monotonically (`h2_dip_vs_density_gradient`): 1× decline → edge; 14× (BE)
→ 0.72; 456× (steep) → 0.064. So the transition is **profile-dependent**, which a single scalar
threshold cannot express.

## What we do (maintainer decision, 2026-06-22: **consolidate & document**)

Retuning 0.05 is futile — there is no event to catch. The result *is* the deliverable: substrate
certified, under-cooling quantified across the regime, transition shown geometric/profile-dependent,
and the stall↔crash connection mapped. **Production is unchanged**; `ENERGY_COLLAPSED` correctly
diagnoses a real heavy-cloud breakdown (not a hidden continuation). Levers recorded for a future
maintainer call (none implemented):
- **Geometric trigger** — energy→momentum handoff at `R2 > k·rCloud` (profile-aware). Lets runs complete; epoch is geometric.
- **Leakage (`coverFraction<1`)** — cheapest *physical* lever; prior work fired the cooling trigger at Cf≈0.95, solver-healthy.
- **Mixing-layer cooling** into the betadelta solve (θ≈0.25) — the principled root fix; creates the missing 10⁵–10⁶ K radiating gas so a real cooling event exists. A modelling workstream (a bulk sink breaks the solver).

## Figure index (all in `figures/`, PDF+PNG, house style; regenerate via `make_pt4_figures.py`)

- `h1_lloss_surge_collapse` — `Lcool(t)`, all 6 hybr configs: surge ~2× then collapse 4–9× (peak dot).
- `h1_beta_clamp_divergence` — ratio(t) + β(t), hybr vs legacy (simple_cluster): the divergence is the β-clamp.
- `h1_ratio_min_stats` — cooling-ratio minimum per config, hybr (0.28–0.49, 0/6 fire) vs legacy (≤0, 5/6 fire).
- `h2_ratio_vs_rcloud` — ratio vs `R2/rCloud`: bottoms at the edge, recovers past blowout, never reaches 0.05.
- `h2_matched_r2` — baseline vs 5.2× cloud overlay identical at matched absolute R2 → cooling is local-density-set.
- `h2_dip_vs_density_gradient` — dip location vs in-cloud density steepness (explains the `pl2_steep` outlier).
- `h3_ebfloor_noop_and_grind` — flooring Eb: bit-identical no-op on simple_cluster; grinds (no rescue) on fail_repro.
- `h4_Eb_sweep_{fail_repro,fail_helix,mass_1e9}` — Eb(t) across the t_window sweep, cap-release marked.
- `h4_pdvratio_sweep_{fail_repro,fail_helix,mass_1e9}` — PdV/Lmech stays >1 after release → re-collapse unless cap is long.
- `h4_summary` — survived-window / self-sustained vs t_window (3e-3 splits the clouds; 1e-2 self-sustains both).
- `h4_control_vs_collapse` — PdV/Lmech: simple_cluster (control) <1, the 5e9 clouds cross 1.

## Artifacts & reproduce

- **Docs:** `H1_lcool_audit.md`, `H2_rcloud_audit.md`, `H3_eb_floor_experiment.md`, `H4_pdvcap_experiment.md`.
- **Harnesses (monkeypatch, production untouched):** `analyze_lcool_direction.py`, `trajectory_probe.py` (H1);
  `h2_*.py` (H2); `h3_variants.py`, `h3_run_variant.py`, `h3_run_matrix.sh`, `h3_analyze.py` (H3);
  `h4_variants.py`, `h4_run_variant.py`, `h4_run_matrix.sh`, `h4_analyze.py`, `h4_figures.py` (H4).
- **Data:** `H1_lcool_direction_summary.csv`, `h2_*.csv`, `h3_eval.csv`, `h4_eval.csv`, `h4_rows/`, `traj/*.csv`;
  upstream source `../cleanroom/data/c0_*_{h0,legacy}.csv`.
- **Regenerate all figures:** `python docs/dev/transition/pt4/make_pt4_figures.py` (pure reads of the committed CSVs).
- **Caveat on H3/H4:** both *inject non-conserved energy* (floor Eb / cap PdV) — they are **diagnostics** that
  isolate the failure mode, **not** production-fix candidates.
