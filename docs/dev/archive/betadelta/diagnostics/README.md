# docs/dev/archive/betadelta/diagnostics — betadelta hybr-solver diagnostics & plots (glossary)

> ⚠️ **This document may be out of date — verify before trusting it.** It is a
> point-in-time analysis/audit, not a maintained spec; the code moves faster
> than these notes (paths, line numbers, and "what shipped" status drift).
> **Any agent or person reading this: treat it as unverified. Flag that it may
> be stale and re-check each claim, snippet, and line reference against the
> current source before relying on it.**
>
> 🧊 **Frozen historical record — do not extend.** This workstream shipped or was
> superseded (see the Status line below); the doc is kept as evidence/history. Do
> not update or extend it — new work gets a new doc in an active workstream. The
> ⚠️ caveat above still applies: paths and line references reflect the code as it
> was when this was written.

Scratch diagnostics/plots for the β–δ **hybr** solver investigation (the
pole-free `g` metric, the four-arm promotion, and the negative-velocity /
WARPFIELD "Problem 2" study). **Not source** — regenerable. Canonical writeups:
`docs/dev/archive/betadelta/PHASE2_ARMS.md`, `docs/dev/archive/betadelta/stalling-energy-phase.md`,
`docs/dev/archive/betadelta/HYBR_PLAN.md`. Companion harness dir: `docs/dev/archive/betadelta/velstruct/`.

## Experiment families (by file prefix)

| prefix | phase | what it is | harness → data | plot script |
|---|---|---|---|---|
| `arms_*`     | 2.3 | **four-arm** shadow: 4 solver strategies side by side (A control / B metric / C cap+bounds / D **hybr**) | `arms.py` → `arms_*.jsonl` | `analyze_arms.py` |
| `probe_*`    | 2.1/2.2 | **probe**: (β,δ) residual *landscape* — a 7×7 scan + transects per segment | `probe.py` → `probe_*.jsonl` | `analyze_probe.py` → `betadelta_*` |
| `phase3_*`   | 3 | **hybr-vs-legacy** master-table summary (convergence / β-reach / transition / cost) | *(transcribed from the doc; no jsonl)* | `analyze_phase3.py` |
| `stalling_*` | 3/5 | **stall**: self-consistent `stop_t=4` sweep runs — source of the negative-velocity study | run → `docs/dev/data/stalling_*.csv` | `analyze_negvel.py` → `negvel_*` |
| `hunt_*` (h1–h6) | 6.0 | **hunt**: velocity-contamination sweep, 6 configs | `docs/dev/archive/betadelta/velstruct/hunt.py` → `docs/dev/data/hunt_*.csv` | `plot_hunt.py` → `hunt_*` |
| `negvel_*`   | 5/6 | the negative-velocity diagnosis figures (trigger / timeline / dmdt-lmech / feedback / profile / causal) | — | `analyze_negvel.py`, `reconstruct_vprofile.py` |

**Animations:**
- `make_arms_rootmap_gif.py` → `arms_rootmap_{simple1e5,mock4e3}.gif` — `arms_rootmap`
  revealed over segments: the REAL production-clamped roots (arm A = cage) vs hybr
  (arm D = no cage) + a residual-`g` convergence panel. Pure read of the jsonl → ~50 s.
- `make_rootmap_gif.py` → `rootmap_cage.gif` — the steep run's root-finding **with vs
  without the cage** over time. Six panels: LEFT the (β,δ) plane (cage box; hybr roots
  escape, the **real legacy/caged** roots ride the edge — NOT a geometric clip), a two-arm
  **residual-g** convergence panel, and **velocity v(r) vs physical radius [pc]**; RIGHT
  **velocity v(r) vs radial fraction** (inflow = v<0), Lmech(t), and **R2 & R_IF vs t**
  (the ionization front ~ R2 for this dense shell). Profile panels overlay cage vs no-cage.
  Frames are paced on a uniform **linear-t** grid (interpolated markers/curves; real
  segments kept as dots) so the over-dense early phase no longer dwells. A **pure read** of
  two committed csvs (`rootmap_cage_scalars.csv`, `rootmap_cage_profiles.csv.gz`) produced
  once by `tabulate_cage.py`; renders in seconds, needs only numpy+pandas+matplotlib+pillow.
- `tabulate_cage.py` → `rootmap_cage_{scalars.csv,profiles.csv.gz}` (+ a gitignored
  `rootmap_cage_table.npz` cache) — the slow half of the animation, factored out: for every
  steep-hybr segment it records both roots, both residuals, the recovered ODE-state T0, the
  ionization-front radius R_IF (a shell solve), and the re-solved v(r)/n(r) for **both** the
  no-cage (hybr) and the **REAL** caged (legacy, threaded-guess) solve. ~1.5 hr (the
  per-segment legacy solve dominates). Needs the venv.

**Cage counterfactual:** `cage_compare.py` → `cage_compare.png` — the REAL legacy
(clamped) solve vs hybr at two key segments, v vs r: at the WR surge the cage is
forced to a different in-box root that predicts NO inflow (it *hides* Problem 2).
The per-segment legacy solve is ~60 s (grids ~25 structure solves through the pole);
`cage_compare.png` is the static 2-segment deep-dive, while `tabulate_cage.py` now runs
that same real solve over **every** segment so `rootmap_cage.gif` animates the real cage
trajectory + profiles (its caged markers are no longer a geometric clip).

**Metrics:** `f` = legacy residual, `f_E=(E1−E2)/E1` — its denominator hits 0 near
the E_b peak (a **pole**). `g` = hybr residual, `g_E=(E1−E2)/Lmech_total` —
pole-free. Both: total = E-comp² + T-comp², converged < 1e-4. Legacy clamp box:
β∈[0,1], δ∈[−1,0].

## Config / run-name glossary

`cluster mass = sfe × mCloud`. Profile: `α_ρ` (densPL_alpha) = 0 flat, −2 steep
r⁻². All `dens_profile=densPL`.

| name(s) | mCloud [M☉] | sfe | cluster | α_ρ | nCore [cm⁻³] | what it is |
|---|---|---|---|---|---|---|
| **mock** (`mock4e3`, `mockfull`) | 3966 | 0.0085 | ~34 | 0 | 5e2 | tiny low-mass cloud (the `mockfull` example) |
| **simple** (`simple1e5`) | 1e5 input → 7e4 gas | 0.30 | 3e4 | 0 | 1e5† | the `simple_cluster` worked example (`param/simple_cluster.param`) |
| **typical** (`probe_cloud1e6`) | 1e6 | 0.01 | 1e4 | 0 | **1e3** | typical GMC, normal density (flat profile) |
| **flat** (master-table) | 1e6 | 0.01 | 1e4 | 0 | **1e5** | typical GMC, **dense** (flat profile, n=1e5) |
| **steep** (`cloudPL`) | 1e6 | 0.01 | 1e4 | −2 | 1e5 | steep r⁻² halo. `PL` = power-law |
| **cost** | 1e6 | 0.01 | 1e4 | 0 | 1e5 | flat but `stop_t=0.08` (throughput benchmark only) |
| **h1 base** | 1e6 | 0.01 | 1e4 | −2 | 1e5 | hunt baseline (= steep) |
| **h2 sfe10** | 1e6 | 0.10 | 1e5 | −2 | 1e5 | 10× stronger SN |
| **h3 sfe30** | 1e6 | 0.30 | 3e5 | −2 | 1e5 | strongest SN |
| **h4 dense** | 1e6 | 0.10 | 1e5 | −2 | **1e6** | dense halo (hand-off transient; excluded) |
| **h5 long** | 1e6 | 0.03 | 3e4 | −2 | 1e5 | `stop_t=8` — full WR→SN→decline |
| **h6 flat** | 1e6 | 0.30 | 3e5 | **0** | 1e3 | flat control (sfe 0.30) |

† `simple` sets no `nCore`/profile → uses `simple_cluster` defaults (n=1e5, α=0).

**Heads-up on `flat` vs `typical`:** both are flat profile (α=0); they differ by
density — master-table **flat** is n=1e5, **typical** is n=1e3. The probe run
`probe_cloud1e6` is the **typical** density (n=1e3), *not* the dense "flat".

## The .param files here

`probe_{cloud1e6,cloudPL,mock4e3,simple1e5}.param`, `arms_{mock4e3,simple1e5,smoke}.param`
(`*_smoke` = a tiny `stop_t` smoke test). The hunt params live in `docs/dev/archive/betadelta/velstruct/`.

## Data locations

- **canonical** (committed, read by the plot scripts): `docs/dev/data/stalling_*.csv`,
  `docs/dev/data/hunt_*.csv`.
- **scratch jsonl** (here): `arms_*.jsonl`, `probe_*.jsonl` — the per-segment shadow logs.
- **rootmap_cage tables** (here, committed): `rootmap_cage_scalars.csv` +
  `rootmap_cage_profiles.csv.gz` — the steep cage-vs-no-cage roots/residuals + v(r)/n(r)
  profiles per segment, written by `tabulate_cage.py`; `make_rootmap_gif.py` reads these.
- `reconstruct_vprofile.py` re-solves the real bubble structure (needs a venv with
  the pinned deps, numpy<2/scipy<2) to recover `v(r)`, which the stalling CSVs don't store.

## Figures index

Every tracked figure here, grouped by family (scratch artefacts `_frame_check.png`
and `rootmap_cage_profiles.npz` are gitignored — debug frame + the GIF profile cache).

**Four-arm shadow (Phase 2.3, `analyze_arms.py`):**
- `arms_summary.png` — per-arm convergence / β-reach / cost across configs (the headline).
- `arms_rootmap.png` — (β,δ) accepted roots, all arms × all segments, static (the GIFs animate this).
- `arms_residual.png` — accepted-root residual vs t, per arm.
- `arms_pareto.png` — cost vs convergence trade-off (hollow = dominated).
- `arms_rootmap_{simple1e5,mock4e3}.gif` — `arms_rootmap` revealed over time: cage (A) vs no-cage (D) + residual-`g` panel.

**Residual landscape / metric (Phase 2.1–2.2, `analyze_probe.py`):**
- `betadelta_gmap.png` — pole-free `g`-metric residual + feasibility, one panel per config.
- `betadelta_f_vs_g.png` — same scan, `f` (legacy, pole) vs `g` (hybr) side by side.

**Master table (Phase 3, `analyze_phase3.py`):**
- `phase3_headline.png` — hybr-vs-legacy convergence / β-reach / transition / cost.
- `phase3_regime.png` — outcome by regime (converge / stall / contaminate).

**Negative-velocity / Problem 2 (Phase 5–6, `analyze_negvel.py` + `reconstruct_vprofile.py`):**
- `negvel_trigger.png` — what flips v<0: β+δ≲−0.4 at an Lmech surge.
- `negvel_timeline.png` — inflow episodes against the feedback history.
- `negvel_dmdt_lmech.png` — dMdt vs Lmech (feedback leads the contamination).
- `negvel_feedback.png` — wind/SN decomposition of the driving surge.
- `negvel_profile.png` — reconstructed v(r): subsonic (Mach≈0.002), KE ~1e-6 of thermal, likely artefact.
- `negvel_causal.png` — causal ladder: measured ①–③ vs conjectural inflow ④.

**Velocity-contamination hunt (Phase 6.0, `docs/dev/archive/betadelta/velstruct/plot_hunt.py`):**
- `hunt_trigger.png` — trigger across the 6 configs (h1–h6).
- `hunt_massdep.png` — mass / sfe dependence.
- `hunt_dmdt_leads.png` — dMdt leads the velocity contamination.

**Cage counterfactual (`cage_compare.py` + `make_rootmap_gif.py`):**
- `cage_compare.png` — REAL legacy (clamped) solve vs hybr at key segments; the cage hides Problem 2.
- `rootmap_cage.gif` — steep run, cage vs no-cage over time: (β,δ) plane (real caged roots) +
  two-arm residual + density n(r) + velocity v(r) (cage vs no-cage) + Lmech(t). Pure CSV read.

## Status / log (2026-06-14)

- **Plot scripts read the canonical committed CSVs** (`docs/dev/data/stalling_*.csv`,
  `docs/dev/data/hunt_*.csv`); the earlier scratch duplicates were removed so there is
  one source of truth. The `*.jsonl` shadow logs (arms/probe) stay here — they are
  scratch-only and not duplicated upstream.
- **Negative-velocity figures are aligned with the canonical "Is the inflow physical?"
  finding**: the interior inflow is subsonic (Mach≈0.002), energetically negligible
  (~1e-6 of thermal), absent from the cooling/energy integrals, and **likely an ansatz
  artefact — still OPEN**. Plots present ①–③ as measured and the inflow rung ④ as conjectural.
- **The cage hides Problem 2.** `cage_compare.py` runs the real bounded (legacy) solver
  at the WR-surge segment: it is forced to a different *in-box* root that predicts NO
  interior inflow. The GIF's clamped square is only a geometric proxy for the box edge;
  the real caged prediction is `cage_compare.png`.
- **Both animations are now pure reads; the solve is factored out.**
  `make_arms_rootmap_gif.py` reads the jsonl (~50 s). `make_rootmap_gif.py` reads the
  committed `rootmap_cage_*` csvs and renders in seconds; the one-time ~1.5 hr solve
  (incl. the **real** per-segment legacy/caged solve and density) lives in
  `tabulate_cage.py`. Velocity/density share the radial-fraction x-axis with fixed
  y-limits so the inflow and the cage-vs-no-cage gap stay visible across frames.
- **T0 is an ODE state variable, not in the stalling CSV.** `tabulate_cage.py` recovers
  each segment's evolved T0 as `T_bubble` at the converged hybr root (gT≈0 there), then
  uses that shared target for the caged solve — otherwise the temperature residual is
  computed against the stale initial T0 and every segment looks unconverged.
- **Probe label fix:** `probe_cloud1e6` is the **typical** density (n=1e3, α=0), *not*
  the dense "flat" (n=1e5) — see the config glossary heads-up above.
- **Tracking note:** this README is the live tracker for the plotting/animation work so it
  does not collide with the canonical docs (`docs/dev/archive/betadelta/stalling-energy-phase.md`,
  `docs/dev/archive/betadelta/HYBR_PLAN.md`) that are being edited for Phase 6.1.
- **Pending (maintainer-owned):** the Phase-6.1 reject-and-hold counterfactual is running
  separately; once it lands, the planned figure is the treatment effect (arm A accept vs
  reject-and-hold: Δ dMdt / R2 / v2 / terminal momentum / transition time).
