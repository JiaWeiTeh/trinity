# mockCloudyinput

Sample output from the `trinity2CLOUDY` pipeline — kept here so anyone
can see what the driver produces without running it.

## Contents

```
mockCloudyinput/
├── 1e4_sfe010_n1e3_PL0_yesPHII_254_momentum_t1p9917myr.in
├── 1e4_sfe010_n1e3_PL0_yesPHII_254_momentum_t1p9917myr.dlaw.txt
└── trinity_linelist.dat
```

| File             | Role                                                          |
|------------------|---------------------------------------------------------------|
| `*.in`           | the CLOUDY input deck (run with `cloudy -r <prefix>`)         |
| `*.dlaw.txt`     | the dlaw block on its own — copy-paste between decks          |
| `trinity_linelist.dat` | the line list referenced by the deck's `save lines` lines |

## Where it came from

Generated from a TRINITY run named `1e4_sfe010_n1e3_PL0_yesPHII`
(1×10⁴ M☉ cloud, SFE = 0.010, n_core = 10³ cm⁻³, PL profile,
photoionisation pressure on). The deck is for snapshot 254 in the
`momentum` phase, at cluster age ≈ 1.99 Myr.

To regenerate from your own copy of that run directory:

```bash
python -m src._output.cloudy.trinity_to_cloudy \
    -F outputs/.../1e4_sfe010_n1e3_PL0_yesPHII/ \
    --age 2.0 \
    --out outputs/mockOutput/mockCloudyinput/
```

(Closest snapshot to age 2.0 Myr is 254, at 1.9917 Myr — hence the
filename suffix `t1p9917myr`.)

## Before running CLOUDY

The deck's `table star` line carries a `<<<EDIT_ME>>>` sentinel:

```
table star "<<<EDIT_ME>>>" age = 1.9917e+06 years
```

CLOUDY needs the name of a CLOUDY-compiled SB99 atmosphere grid (a
`.mod` file built with CLOUDY's `compile stars` command). The pipeline
deliberately doesn't pick one for you — replace the sentinel by hand,
or regenerate with `--sb99 your_grid.mod`. Full details (and why the
TRINITY-side `1e6cluster_*.txt` files are NOT a substitute) are in
`src/_output/cloudy/README.md`.
