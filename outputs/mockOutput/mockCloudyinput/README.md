# mockCloudyinput

A complete, runnable example of the `trinity2CLOUDY` pipeline output —
kept here as a reference so anyone can see what a deck looks like
without running the driver.

## Contents

```
mockCloudyinput/
├── 1e5_sfe001_n1e3_PL0_yesPHII_232_momentum_t2p0197myr.in
├── 1e5_sfe001_n1e3_PL0_yesPHII_232_momentum_t2p0197myr.dlaw.txt
├── 1e5_sfe001_n1e3_PL0_yesPHII_current_run_context.md
├── 1e5_sfe001_n1e3_PL0_yesPHII_current_run_timeseries.csv
├── 1e5_sfe001_n1e3_PL0_yesPHII_bubble_profile_t0p1733myr.csv
├── 1e5_sfe001_n1e3_PL0_yesPHII_shell_PHII_profile_t1p9800myr.csv
├── make_current_run_plots.py
├── plot_last_bubble_pressure_profile.py
├── trinity_linelist.dat
└── cloudyinput.zip            # the three files above, bundled for download
```

| File             | Role                                                          |
|------------------|---------------------------------------------------------------|
| `*.in`           | the CLOUDY input deck (run with `cloudy -r <prefix>`)         |
| `*.dlaw.txt`     | the dlaw block on its own — paste into other decks            |
| `*_current_run_context.md` | provenance note and key current-run values        |
| `*_current_run_timeseries.csv` | age/radius/pressure time series from the current parent run |
| `*_bubble_profile_t0p1733myr.csv` | last non-stale hot-bubble density/temperature/pressure profile |
| `*_shell_PHII_profile_t1p9800myr.csv` | nearest-2-Myr ionised shell density/pressure profile |
| `make_current_run_plots.py` | regenerate the current-run PDF plots with `usetex=True` |
| `plot_last_bubble_pressure_profile.py` | plot last non-stale hot-bubble pressure profile |
| `trinity_linelist.dat` | the line list referenced by the deck's `save lines` lines |
| `cloudyinput.zip` | all three files bundled, for one-click download              |

## Where it came from

Generated from a TRINITY run named `1e5_sfe001_n1e3_PL0_yesPHII`
(1×10⁵ M☉ cloud, SFE = 0.01 → 1000 M☉ cluster, uniform power-law
profile with n_core = 10³ cm⁻³, covering fraction Cf = 0.77,
photoionisation pressure on, Z = Z☉). The deck is for snapshot 232 in
the **momentum** phase, at cluster age ≈ 2.02 Myr — the energy-driven
phase is over and the shell is coasting as a momentum-driven snowplow
at ≈ 27.7 pc.

To regenerate from a copy of that run directory:

```bash
python -m trinity._output.cloudy.trinity_to_cloudy \
    -F outputs/rosette_cf_survey_updated_0p77/1e5_sfe001_n1e3_PL0_yesPHII/ \
    --age 2.0 \
    --out outputs/mockOutput/mockCloudyinput/
```

The closest snapshot to age 2.0 Myr is index 232, at 2.0197 Myr — hence
the filename suffix `t2p0197myr`.

### Current parent-run diagnostics

The parent run directory still exists locally at
`outputs/rosette_cf_survey_updated_0p77/1e5_sfe001_n1e3_PL0_yesPHII/`,
but its current `dictionary.jsonl` appears to have drifted from the stream
that produced this committed deck: snapshot 232 is now at 2.4513 Myr, not
2.0197 Myr. The closest current snapshot to the deck age is index 223 at
2.0300 Myr.

The `*_current_run_*` CSV/plots therefore describe the current parent run,
not a byte-exact reconstruction of the committed CLOUDY deck's original
snapshot stream. The PDF plots use the TRINITY matplotlib style and show only
the interval `0 <= t <= 2.5 Myr`. The radius-vs-age plots overlay the
paper-facing Rosette anchors from `paper/rosette/main.tex`: the `1.5-2.5 Myr`
age window, the `r_bubble = 6.2 pc` wind cavity from Bruhweiler et al. (2010),
the `r_shell = 19 +/- 2 pc` HII outer edge from Celnik (1985), and an `11 pc`
CO-ring guide from Dent et al. (2009) as a separate molecular structure. The
`18-22 pc` dust-shell band remains a cross-check only; Planck XXXIV is a
magnetic-field/dust-shell source, not the primary radius target. Regenerate
the plots with:

```bash
./.venv/bin/python outputs/mockOutput/mockCloudyinput/make_current_run_plots.py
```

The script writes:

- `1e5_sfe001_n1e3_PL0_yesPHII_P_HII_vs_r_bubble.pdf`: `P_HII/k_B`
  versus `r_bubble` with a log y-axis
- `1e5_sfe001_n1e3_PL0_yesPHII_r_bubble_vs_age.pdf`: `r_bubble` versus age,
  with the Rosette wind-cavity observation
- `1e5_sfe001_n1e3_PL0_yesPHII_r_shell_vs_age.pdf`: `r_shell` versus age,
  with the Rosette HII/dust-shell and CO-ring observations

The current parent run still carries compact bubble-profile arrays at 2 Myr,
but those arrays are stale: the last snapshot where the saved bubble structure
changes is snapshot 154 at `t = 0.1733016 Myr`, during the implicit phase. The
`*_bubble_profile_t0p1733myr.csv` file is therefore the last genuinely computed
hot-bubble profile, not a 2 Myr profile. It includes `nH_cm_minus3`,
`temperature_K`, the simple `nH_T_K_cm_minus3`, and the composition-corrected
`P_total_over_kB_K_cm_minus3 = (mu_H / mu_ion) n_H T`.

Regenerate the matching PDF pressure-profile plot with:

```bash
MPLCONFIGDIR=/tmp/trinity-mpl ./.venv/bin/python outputs/mockOutput/mockCloudyinput/plot_last_bubble_pressure_profile.py
```

This writes `1e5_sfe001_n1e3_PL0_yesPHII_bubble_PHII_vs_r_bubble_t0p1733myr.pdf`
and `1e5_sfe001_n1e3_PL0_yesPHII_bubble_n_T_vs_r_bubble_t0p1733myr.pdf`.
The plotted pressure is the hot-bubble pressure divided by `k_B`,
`P/k_B = (mu_H / mu_ion) n_H T`, in `K cm^-3`.
The pressure panel uses a log y-axis and overlays the scalar `P_HII` from the
nearest snapshot to 2 Myr as a dashed horizontal reference.

For a pressure profile near 2 Myr, use the shell/HII file instead. The nearest
snapshot to exactly 2 Myr is snapshot 222 at `t = 1.9800298 Myr`; its ionised
shell profile runs from `R2 = 6.4086 pc` to `R_IF = 25.5246 pc`. The first
profile point matches the scalar `P_HII/k_B = 1.505e5 K cm^-3`; the final point
at the ionisation front carries the saved density jump and is correspondingly
much higher.

## Why this snapshot is interesting

Even in the momentum phase the dlaw is a **textbook ionisation front**,
but with the opposite weighting from a young, dense shell: here the
ionised gas fills almost the entire shell and the neutral region is a
thin outer skin. Walking through the 100-row block:

| Rows   | log r (dex)         | r (pc)          | log n_H (dex) | n_H (cm⁻³)   | Region                                   |
|--------|---------------------|-----------------|---------------|--------------|------------------------------------------|
| 1–58   | 19.3187 → 19.9310   | 6.75 → 27.65    | 0.788 → 0.883 | ≈6–8         | ionised interior (nearly the whole shell) |
| 58→59  | 19.93102 → 19.93103 | 27.647 → 27.648 | 0.883 → 3.184 | **8 → 1530** | **the ionisation front itself**           |
| 59–100 | 19.9310 → 19.9317   | 27.65 → 27.69   | 3.184 → 3.186 | ≈1530        | thin neutral skin at the shell's edge     |

The IF spans about **0.001 pc** of physical thickness with a **2.30 dex
(× 200) density jump**, sitting just **0.045 pc inside `rShell`**. The
dlaw preserves this verbatim — `dlaw.py`'s edge-detection threshold
catches the row pair as steep and densifies only the smooth spans on
either side, never inserting interpolated rows that would smear the front.

CLOUDY interpolates linearly between dlaw rows, so the deck presents this
as a near-step in density. CLOUDY then computes its own ionisation
balance and decides where it thinks the IF actually sits — comparing
TRINITY's IF location to CLOUDY's is itself a useful diagnostic of how
well the 1-D thin-shell prescription is doing.

## Geometry at a glance

All numbers in the dlaw are **log₁₀** (CLOUDY's `dlaw table radius`
convention):

```
radius 19.3187 19.9317      ← log₁₀(R2/cm)  log₁₀(rShell/cm)
              │                   │
              6.75 pc             27.69 pc        →  shell thickness 20.9 pc
```

For comparison: the cluster has Q(H) = 10⁴⁹·⁸¹ ≈ 6.5×10⁴⁹ photons/s.
That is enough to keep nearly the full 20.9 pc shell thickness ionised
at the low interior density (n_H ≈ 6–8 cm⁻³); only a ~0.05 pc neutral
skin survives at the outer edge, which is where the dlaw's density jump —
the ionisation front — sits.

## Before running CLOUDY

The deck's `table star` line carries a `<<<EDIT_ME>>>` sentinel:

```
table star "<<<EDIT_ME>>>" age = 2.0197e+06 years
```

CLOUDY needs the name of a CLOUDY-compiled SB99 atmosphere grid (a
`.mod` file built with CLOUDY's `compile stars` command). The pipeline
deliberately doesn't pick one for you — replace the sentinel by hand,
or regenerate with `--sb99 your_grid.mod`.

Full details (and why the TRINITY-side `1e6cluster_*.txt` files are NOT
a substitute) are in `trinity/_output/cloudy/README.md`.
