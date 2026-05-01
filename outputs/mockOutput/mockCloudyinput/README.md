# mockCloudyinput

A complete, runnable example of the `trinity2CLOUDY` pipeline output —
kept here as a reference so anyone can see what a deck looks like
without running the driver.

## Contents

```
mockCloudyinput/
├── 1e5_sfe001_n20_PL0_yesPHII_208_implicit_t1p9783myr.in
├── 1e5_sfe001_n20_PL0_yesPHII_208_implicit_t1p9783myr.dlaw.txt
├── trinity_linelist.dat
└── cloudyinput.zip            # the three files above, bundled for download
```

| File             | Role                                                          |
|------------------|---------------------------------------------------------------|
| `*.in`           | the CLOUDY input deck (run with `cloudy -r <prefix>`)         |
| `*.dlaw.txt`     | the dlaw block on its own — paste into other decks            |
| `trinity_linelist.dat` | the line list referenced by the deck's `save lines` lines |
| `cloudyinput.zip` | all three files bundled, for one-click download              |

## Where it came from

Generated from a TRINITY run named `1e5_sfe001_n20_PL0_yesPHII`
(1×10⁵ M☉ cloud, SFE ≈ 0.0085 → ~850 M☉ cluster, n_core = 20 cm⁻³,
uniform power-law profile, photoionisation pressure on). The deck is
for snapshot 208 in the **implicit** phase, at cluster age ≈ 1.98 Myr —
i.e., the energy-driven phase is still active and the bubble has not
yet broken out.

To regenerate from a copy of that run directory:

```bash
python -m src._output.cloudy.trinity_to_cloudy \
    -F outputs/.../1e5_sfe001_n20_PL0_yesPHII/ \
    --age 2.0 \
    --out outputs/mockOutput/mockCloudyinput/
```

The closest snapshot to age 2.0 Myr is index 208, at 1.9783 Myr — hence
the filename suffix `t1p9783myr`.

## Why this snapshot is interesting

The dlaw block is a **textbook ionisation front**, exactly the case the
pipeline's IF-preserving densification was built for. Walking through
the block:

| Rows  | log r (dex) | r (pc)      | log n_H (dex) | n_H (cm⁻³) | Region                              |
|-------|-------------|-------------|---------------|------------|-------------------------------------|
| 1–51  | 20.0198 → 20.0606 | 33.85 → 37.11 | 1.077 → 1.085 | ≈12        | ionised inner shell (just inside R2) |
| 52    | 20.06061 → 20.06072 | 37.12 → 37.12 | 1.085 → 3.405 | **12 → 2540** | **the ionisation front itself**    |
| 53–100 | 20.0607 → 20.0613 | 37.12 → 37.16 | 3.405 → 3.407 | ≈2540       | neutral compressed snowplow         |

The IF spans about **0.001 pc** of physical thickness with a **2.3 dex
(× 210) density jump**. The dlaw preserves this verbatim — `dlaw.py`'s
edge-detection threshold catches the row pair as steep and densifies
only the smooth spans on either side, never inserting interpolated rows
that would smear the front.

CLOUDY interpolates linearly between dlaw rows, so the deck presents
this as a near-step in density. CLOUDY then computes its own ionisation
balance and decides where it thinks the IF actually sits — comparing
TRINITY's IF location to CLOUDY's is itself a useful diagnostic of how
well the 1-D thin-shell prescription is doing.

## Geometry at a glance

All numbers in the dlaw are **log₁₀** (CLOUDY's `dlaw table radius`
convention):

```
radius 20.0198 20.0613      ← log₁₀(R2/cm)  log₁₀(rShell/cm)
              │                   │
              33.85 pc            37.16 pc        →  shell thickness 3.3 pc
```

For comparison: the cluster has Q(H) = 10⁴⁹·⁸ ≈ 6×10⁴⁹ photons/s.
Static Strömgren in n_H = 12 cm⁻³ would be ~12 pc — but TRINITY's
expanding HII region with PHII pressure is balanced at ~37 pc here,
which is where the dlaw begins.

## Before running CLOUDY

The deck's `table star` line carries a `<<<EDIT_ME>>>` sentinel:

```
table star "<<<EDIT_ME>>>" age = 1.9783e+06 years
```

CLOUDY needs the name of a CLOUDY-compiled SB99 atmosphere grid (a
`.mod` file built with CLOUDY's `compile stars` command). The pipeline
deliberately doesn't pick one for you — replace the sentinel by hand,
or regenerate with `--sb99 your_grid.mod`.

Full details (and why the TRINITY-side `1e6cluster_*.txt` files are NOT
a substitute) are in `src/_output/cloudy/README.md`.
