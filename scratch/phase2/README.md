# scratch/phase2 — betadelta hybr-solver diagnostics & plots (glossary)

Scratch diagnostics/plots for the β–δ **hybr** solver investigation (the
pole-free `g` metric, the four-arm promotion, and the negative-velocity /
WARPFIELD "Problem 2" study). **Not source** — regenerable. Canonical writeups:
`analysis/BETADELTA_PHASE2_ARMS.md`, `analysis/stalling-energy-phase.md`,
`docs/dev/BETADELTA_HYBR_PLAN.md`. Companion harness dir: `scratch/phase6/`.

## Experiment families (by file prefix)

| prefix | phase | what it is | harness → data | plot script |
|---|---|---|---|---|
| `arms_*`     | 2.3 | **four-arm** shadow: 4 solver strategies side by side (A control / B metric / C cap+bounds / D **hybr**) | `arms.py` → `arms_*.jsonl` | `analyze_arms.py` |
| `probe_*`    | 2.1/2.2 | **probe**: (β,δ) residual *landscape* — a 7×7 scan + transects per segment | `probe.py` → `probe_*.jsonl` | `analyze_probe.py` → `betadelta_*` |
| `phase3_*`   | 3 | **hybr-vs-legacy** master-table summary (convergence / β-reach / transition / cost) | *(transcribed from the doc; no jsonl)* | `analyze_phase3.py` |
| `stalling_*` | 3/5 | **stall**: self-consistent `stop_t=4` sweep runs — source of the negative-velocity study | run → `analysis/data/stalling_*.csv` | `analyze_negvel.py` → `negvel_*` |
| `hunt_*` (h1–h6) | 6.0 | **hunt**: velocity-contamination sweep, 6 configs | `scratch/phase6/hunt.py` → `analysis/data/hunt_*.csv` | `plot_hunt.py` → `hunt_*` |
| `negvel_*`   | 5/6 | the negative-velocity diagnosis figures (trigger / timeline / dmdt-lmech / feedback / profile / causal) | — | `analyze_negvel.py`, `reconstruct_vprofile.py` |

**Animation:** `make_rootmap_gif.py` → `rootmap_cage.gif` — the steep run's (β,δ)
root-finding **with vs without the cage** (legacy box) over time, alongside the
re-solved bubble velocity-vs-density profile and Lmech(t). Frame = segment; all
three panels are the same timestamp. Needs the venv (numpy<2/scipy<2) + pillow.

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
(`*_smoke` = a tiny `stop_t` smoke test). The hunt params live in `scratch/phase6/`.

## Data locations

- **canonical** (committed, read by the plot scripts): `analysis/data/stalling_*.csv`,
  `analysis/data/hunt_*.csv`.
- **scratch jsonl** (here): `arms_*.jsonl`, `probe_*.jsonl` — the per-segment shadow logs.
- `reconstruct_vprofile.py` re-solves the real bubble structure (needs a venv with
  the pinned deps, numpy<2/scipy<2) to recover `v(r)`, which the CSVs don't store.
