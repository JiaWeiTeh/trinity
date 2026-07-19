# INTERIM ATLAS — Rosette C_f scan (P22 step 4)

> ⚠️ **PROVISIONAL / IN-CONTAINER — NOT HPC. INTERIM tier (PLAN §0.12).** Every
> figure here is a *preliminary* read of the 72-arm in-container scan. **Never a
> paper number; never fills a `TBD(HPC)`; does NOT satisfy or shrink P21** (the
> ~8000-run Helix survey stays the sole data SSOT). Re-confirm on HPC before any
> paper number. The 🔄/💾/🔗 banners of `docs/dev/rosette-cf/README.md` apply.

Produced by the maintainer's frozen reduction of the committed dicts (the README §11
close-out step). The **only quotable matcher** is the frozen `paper/rosette/matching/`
(`observables.py` + `likelihood.match_run`); the in-repo `harness/match_cf_scan.py` is
a fallback sanity read (POLICY diff verdict below).

## Exact commands
```
python docs/dev/rosette-cf/figs/reduce_cf_scan_frozen.py   # -> data/match_interim_cf_PISM1e5_frozen_2026-07-14{,_cells}.csv
python docs/dev/rosette-cf/figs/make_cf_atlas.py           # -> figs/A1..A6
```
Env: numpy<2 + matplotlib; **no scipy needed** for the reduction/atlas (they parse the
gzipped JSONL directly and call the frozen matcher). `trinity_reader`/`paper_Cf.py`
DO need scipy (absent in-container) — see the A1/A2 note.

## POLICY diff (step 1) verdict — CLEAN
The frozen `observables.py` and the fallback `match_cf_scan.py` agree on every **quotable**
constant: R2 ↔ 7.0 ± 1.0 pc, rShell ↔ 19.0 ± 2.0 pc, flat age window [1.5, 2.5] Myr,
radii-only default χ². The fallback additionally reports the **6.2 pc** cavity base (F-12 —
which the frozen file lacks but this task requires anyway) and *omits* the frozen marginal
`lnL_marg`, the velocity opt-in terms, and `SYS_FRAC`. None of those alter the χ² targets, so
the fallback keeps "sanity read" status and the frozen reduction (below) is authoritative — and
it **confirms** the fallback's qualitative picture.

## The reduced data (frozen matcher, both bases)
- `data/match_interim_cf_PISM1e5_frozen_2026-07-14.csv` — per-arm (72 rows): status, χ²_min,
  t_best, R2/rShell at best, overshoot, `lnL_marg`, on both `_7` and `_62` bases; `age_censored`.
- `..._cells.csv` — per (mass-pair × nCore × fmix × PHII) cell (24): χ² grid over C_f, `best_cf`,
  `edge_min`, `full3`, matched-t sealed overshoot, both bases.

## Figures
- **A1** `A1_cf_trajectories_{1e4_noPHII,1e5_yesPHII}.png` — decreasing-C_f cavity trajectories
  R2(t), grid nCore×fmix; obs bands (7±1 green, 6.2 dashed), age window; age-censored curves dimmed.
  *Interim stand-in for `figures/paper_Cf.plot_cf_trajectory`* (scipy absent → paper_Cf unrunnable
  in-container; same raw arrays, different wrapper — P21 runs the real script as a data-swap).
- **A2** `A2_cf_constraint_bothbases.png` — value-at-age R2 vs C_f against both bases; × = age-censored
  (final R2, not a match). *Interim stand-in for `plot_cf_constraint`.* NOTE: the P22 spec's
  "`--R_cavity` override in paper_Cf.py" does not exist (only `paper_Rosette.py` has `--R-cavity`);
  both bases are carried here in the atlas/reduction instead.
- **A3** `A3_match_map_base{7,62}.png` — frozen-χ² heatmap over (nCore × C_f) per mass×fmix×PHII panel;
  ★ best cell, hatched "cens" = age-censored (no minimum). All full-3 cells edge-min at C_f=0.70.
- **A4** `A4_overshoot_gradient.png` — R2(best-t) − target vs nCore per C_f, both bases; open markers =
  age-censored final R2 (extrapolation, not a match). Shows overshoot falling with lower C_f / higher nCore.
- **A5** `A5_best_match_ranking.{png,csv}` — top matches per base; the 2026-07-08 pilot appended, marked
  as a DIFFERENT corner (PISM=1e4, nCore=1e3) + a different (fallback) matcher — not a head-to-head row.
- **A6** `A6_coverage_map.png` — coverage of the full ~8100-run grid; interim samples one PISM, one nISM,
  PL0 only, 3/8 C_f, 3/5 nCore. The interim's 1e4/sfe.10 pair is an off-grid compact bracket, not a grid row.

## Headline (provisional; see PLAN §0.3 dated note 2026-07-14)
Frozen reduction **confirms** §0.12: 9/24 cells are full-3-point, **all edge-min at C_f=0.70**;
sealed (C_f=1) overshoots the cavity by **+38 to +47 pc** (matched-t) in the nCore 50–100 full-3
cells; the pilot's interior ~0.89 is **not** reproduced. Best radii-χ² corner (both bases):
**1e4/sfe0.10 pair, PHII off, C_f=0.70, nCore ≥ 100** (χ² ≈ 0–3; e.g. nCore=100/fmix4 → χ²=0.02,
R2=7.2, rShell=19.0). **Caveat:** that corner is the *less physical* mass pair (compact, high-SFE)
with PHII off — radii-only χ² alone does not select the physical corner. This is the D-15 argument.
