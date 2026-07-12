# Current Parent Run Context

Source run directory: `outputs/rosette_cf_survey_updated_0p77/1e5_sfe001_n1e3_PL0_yesPHII`

The committed CLOUDY deck is named for snapshot 232 at age 2.0197 Myr. In the current local parent run, snapshot 232 is at 2.451294 Myr. The closest current snapshot to 2.0197 Myr is index 223 at 2.030030 Myr.

At that closest current snapshot:

- phase: momentum
- r_bubble (R2): 6.43679 pc
- r_shell: 25.8428 pc
- R_IF: 25.8048 pc
- P_HII/k_B: 148400 K cm^-3

Current-run data and plotting helper:

- `1e5_sfe001_n1e3_PL0_yesPHII_current_run_timeseries.csv`
- `1e5_sfe001_n1e3_PL0_yesPHII_bubble_profile_t0p1733myr.csv`
- `1e5_sfe001_n1e3_PL0_yesPHII_shell_PHII_profile_t1p9800myr.csv`
- `make_current_run_plots.py`
- `plot_last_bubble_pressure_profile.py`

The PDF plots use `paper/_lib/trinity.mplstyle` and only include snapshots
with `0 <= t <= 2.5 Myr`. The H II pressure plot uses `r_bubble` on the x-axis
and a log y-axis. The radius-vs-age plots overlay the paper-facing Rosette
anchors from `paper/rosette/main.tex`: a 6.2 pc wind cavity (Bruhweiler+2010),
a 19 +/- 2 pc HII outer edge (Celnik 1985), and an 11 pc CO ring (Dent+2009)
as a separate molecular structure. The plot script forces `text.usetex = True`;
regenerate the PDFs with:

```bash
./.venv/bin/python outputs/mockOutput/mockCloudyinput/make_current_run_plots.py
```

It writes:

- `1e5_sfe001_n1e3_PL0_yesPHII_P_HII_vs_r_bubble.pdf`
- `1e5_sfe001_n1e3_PL0_yesPHII_r_bubble_vs_age.pdf`
- `1e5_sfe001_n1e3_PL0_yesPHII_r_shell_vs_age.pdf`

Bubble-profile caveat:

- The saved compact profile arrays are present at 2 Myr, but are stale there.
- Last snapshot where the bubble profile changes: index 154 at 0.1733016 Myr.
- The CSV profile is from that final non-stale implicit-phase structure.
- It includes both `nH_T_K_cm_minus3` and composition-corrected
  `P_total_over_kB_K_cm_minus3 = (mu_H / mu_ion) n_H T`.
- The matching pressure plot uses
  `P/k_B = (mu_H / mu_ion) n_H T` in `K cm^-3` and writes
  `1e5_sfe001_n1e3_PL0_yesPHII_bubble_PHII_vs_r_bubble_t0p1733myr.pdf`.
- Its y-axis is log-scaled and includes a dashed scalar `P_HII` reference from
  the nearest snapshot to 2 Myr.
- The same script also writes
  `1e5_sfe001_n1e3_PL0_yesPHII_bubble_n_T_vs_r_bubble_t0p1733myr.pdf`.

Shell/HII pressure profile:

- Nearest snapshot to exactly 2 Myr: index 222 at 1.9800298 Myr.
- The shell profile is valid there and is written through `R_IF`.
- Use `P_total_over_kB_K_cm_minus3 = (mu_H / mu_ion_shell) n_H T_shell`.
- The last row is the density jump at the ionisation front, not a smoothed value.
