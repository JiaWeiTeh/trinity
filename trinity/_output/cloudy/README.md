# trinity2CLOUDY

Convert a TRINITY run directory into one or more CLOUDY input decks.

## Quick start

From the repo root, either form works:

```bash
# As a module (any cwd that has 'trinity/' on PYTHONPATH)
python -m trinity._output.cloudy.trinity_to_cloudy \
    -F outputs/mockOutput/mockFullrun/ \
    --age 0.15

# As a plain file (must be cwd-independent — script bootstraps sys.path)
python trinity/_output/cloudy/trinity_to_cloudy.py \
    -F outputs/mockOutput/mockFullrun/ \
    --age 0.15
```

This picks the snapshot closest to cluster age 0.15 Myr and writes
`<run_dir>/cloudy/<model>_<idx>_<phase>_t<age>myr.in` plus a sidecar
`.dlaw.txt` (the dlaw block in isolation, for copy-paste between decks)
and a copy of `trinity_linelist.dat`.

Before running CLOUDY on the deck, replace the `<<<EDIT_ME>>>` sentinel
in the `table star` line with the name of your CLOUDY-compiled SB99
atmosphere grid. See [SB99 grid prerequisite](#sb99-grid-prerequisite)
below.

## What the script does

Given a TRINITY run directory and a cluster age, it picks the closest
snapshot, validates it, and emits a CLOUDY deck with every TRINITY-side
quantity filled in:

- geometry (`radius` inner / outer in log cm)
- ionising photon rate `Q(H)` from `Qi`
- cluster age in years (= `t_now − tSF`)
- shell density profile as a `dlaw table radius` block (with optional
  IF-preserving densification when sparse, and optional ambient ISM
  splicing past `rShell`)
- linear metallicity scale from `summary.ZCloud`

It does **not** run CLOUDY, compile SB99 atmosphere grids, or parse
CLOUDY output. CLOUDY-side prerequisites are deliberately out of scope
(see "Out of scope" below).

## Input

A standard TRINITY run directory:

```
<run_dir>/
├── <model>.param                    # raw input config (not parsed)
├── <model>_summary.txt              # full resolved config (parsed)
├── dictionary.jsonl                 # snapshot stream
├── metadata.json                    # run-invariant data
├── simulationEnd.txt                # success / failure status
└── ...                              # plots, debug logs (ignored)
```

The driver gates on `simulationEnd.txt: Status: SUCCESS`. To convert a
failed run anyway, pass `--force`.

## Output

```
<run_dir>/cloudy/                                            (or --out DIR)
├── <model>_<idx>_<phase>_t<age>myr.in        # deck (use this with cloudy -r)
├── <model>_<idx>_<phase>_t<age>myr.dlaw.txt  # dlaw block alone (for copy-paste)
├── trinity_linelist.dat                       # copy of bundled line list
└── manifest.json                              # only with --all
```

Filename floats use `p` instead of `.` (e.g. `t0p1482myr`) so prefixes
are filename-, shell-, and CLOUDY-safe.

## CLI flags

### Snapshot picker (exactly one required, no silent default)

| Flag                              | Description                                          |
|-----------------------------------|------------------------------------------------------|
| `--age MYR`                       | cluster age in Myr (canonical form)                  |
| `--t-now MYR`                     | raw simulation time (advanced)                       |
| `--index N`                       | Nth snapshot, `-1` = last                            |
| `--phase NAME [--pick first\|last]` | first or last snapshot in a named phase            |
| `--all`                           | one deck per snapshot, plus `manifest.json`          |

`--age` resolves the target snapshot via `t_target = age + tSF` →
closest match (never interpolated). The closing summary surfaces the
delta between requested age and the actual snapshot's age.

### IO

| Flag                              | Default                                              |
|-----------------------------------|------------------------------------------------------|
| `-F, --folder DIR`                | (required) TRINITY run directory                     |
| `--out DIR`                       | `<run_dir>/cloudy/`                                  |
| `--prefix NAME`                   | auto-built from snapshot                             |
| `--template PATH`                 | bundled `trinity2cloudy.in_template`                 |
| `--linelist PATH`                 | bundled `trinity_linelist.dat`                       |
| `--dry-run`                       | print rendered deck to stdout, no writes             |

`--all + --dry-run` is rejected (printing 178 decks isn't useful).

### Physics

| Flag                              | Default       | Effect                                                       |
|-----------------------------------|---------------|--------------------------------------------------------------|
| `--sb99 NAME`                     | `<<<EDIT_ME>>>` | substituted for `{{SB99_MOD}}` in the deck                  |
| `--radius-out PC`                 | `rShell`      | extends dlaw past `rShell` into ambient ISM                  |
| `--z-override FLOAT`              | `summary.ZCloud` | metallicity scale (linear, scales `metals and grains`)    |
| `--age-min YR`                    | `1e5`         | warn below this cluster age                                  |
| `--age-max YR`                    | `1e8`         | warn above this cluster age                                  |
| `--hard-age-bounds`               | off           | promote age-band warning to a hard error                     |
| `--min-rows N`                    | `10`          | densify dlaw to at least N rows                              |

### Safety

| Flag                              | Effect                                                       |
|-----------------------------------|--------------------------------------------------------------|
| `--force`                         | proceed even if `simulationEnd: Status` ≠ `SUCCESS`         |

## SB99 grid prerequisite

CLOUDY's `table star "<grid>.mod"` directive expects a **compiled
atmosphere grid** — a binary file built with CLOUDY's `compile stars`
command from raw STARBURST99 spectrum files. The 7-column ASCII tables
TRINITY itself reads (e.g. `1e6cluster_norot_Z0002_BH120.txt`) are
*integrated time series*, not SED grids, and **cannot** be used as
`table star` input.

The bundled template ships with `<<<EDIT_ME>>>` in the `table star`
line as a sentinel. Two ways to handle it:

1. Generate the deck, then hand-edit the line to point at your
   compiled grid: `table star "starburst99_z020_norot.mod" age = ...`
2. Pass `--sb99 starburst99_z020_norot.mod` at generation time. The
   driver substitutes the name and suppresses the closing-summary
   `TODO` reminder.

Identify the matching grid from your run's `<model>_summary.txt`:
`sps_path` names the TRINITY-side SPS input file (the bundled default
is solar metallicity, rotation = on, 1e6 Msun reference cluster).
`SB99_rotation` separately keys the non-CIE cooling tables, so make
sure your CLOUDY grid is compiled from the matching STARBURST99 model.

## Customising the template / line list

Edit the bundled files in place; they're regular text:

- `trinity/_output/cloudy/trinity2cloudy.in_template`
- `trinity/_output/cloudy/trinity_linelist.dat`

Or pass `--template PATH` and `--linelist PATH` to use your own.

The template uses `{{KEY}}` placeholders. The driver fills:

| Placeholder       | Source                                                 |
|-------------------|--------------------------------------------------------|
| `{{TITLE}}`       | computed: `TRINITY <model> phase=<phase> age=<X>Myr`   |
| `{{SB99_MOD}}`    | `--sb99` value (default sentinel)                      |
| `{{AGE_YR}}`      | `(t_now − tSF) × 10⁶`                                  |
| `{{LOG_QH}}`      | `log10(Qi) − log10(Myr→s)` (cluster Q(H) in /s, log10) |
| `{{LOG_RIN}}`     | `log10(R2[pc])` + log10(pc→cm)                         |
| `{{LOG_ROUT}}`    | `log10(rShell or --radius-out)` + log10(pc→cm)         |
| `{{ZREL}}`        | `summary.ZCloud` or `--z-override`                     |
| `{{DLAW_ROWS}}`   | dlaw rows (without header / footer)                    |
| `{{DLAW_BLOCK}}`  | full dlaw block (header + rows + footer)               |
| `{{PREFIX}}`      | filename prefix (filename-safe)                        |

Custom templates may use any subset. The strict renderer raises if any
`{{KEY}}` remains unsubstituted (typo guard). The `<<<EDIT_ME>>>`
sentinel does not match the `{{KEY}}` pattern, so it passes through.

## Z semantics

`metals and grains X` (the bundled template's syntax) scales the
abundance pattern set by `abundances HII region`. This is **not** the
same as `Z / Z_sun` unless the abundance pattern is solar. For typical
HII-region abundances it's a multiplicative factor on a sub-solar
default. If you want a strict `Z / Z_sun` scaling, edit the template
to use `abundances "GASS"` (solar) plus `metals X linear` and
`grains "ISM"` separately.

`--z-override` always wins over `summary.ZCloud`.

## CLOUDY syntax verification status

The bundled template uses best-guess forms for items I could not
verify against a live CLOUDY binary. They are pinned in module
defaults (`dlaw.py`) or in the template directly:

| Item                                       | Form pinned                              |
|--------------------------------------------|------------------------------------------|
| dlaw block header / row prefix / footer    | `dlaw table radius` / `continue ` / `end of dlaw` |
| Q(H) directive                             | `Q(H) {LOG_QH}` (no explicit `log`; relies on CLOUDY auto-log for >30) |
| `save last lines emissivity` external file | `save last lines emissivity ".emis" "trinity_linelist.dat"` |
| `stop efrac -2`                            | electron-fraction lower bound, log₁₀     |
| `metals and grains {ZREL}`                 | combined-directive shorthand             |

If any of these fail on first run, the fix is a one-line edit to
`trinity2cloudy.in_template` (or `dlaw.py` for the dlaw block syntax)
— no code change to the rest of the pipeline.

## Multi-age workflow

`--all` writes one self-consistent deck per snapshot — radius, age,
Q(H), and dlaw all matched to that snapshot. The `.dlaw.txt` sidecar
files let you copy a different snapshot's density profile into a deck
for experimentation, but be aware that more than the dlaw varies
between snapshots within a run:

| Field         | Variation in mockFullrun (0.01 → 0.30 Myr) |
|---------------|--------------------------------------------|
| `R2`          | 0.40 → 2.51 pc (6×)                        |
| `rShell`      | 0.45 → 19.78 pc (44×)                      |
| `age`         | 0.01 → 0.30 Myr (30×)                      |
| `Q(H)`        | ~constant (~0.006 dex over 0–0.3 Myr)      |

So pasting one snapshot's dlaw into a different snapshot's deck
without also updating `radius` and `age =` will produce a
physically inconsistent deck.

## Out of scope (deliberately)

- Running CLOUDY (`cloudy -r ...` is your job).
- Compiling SB99 atmosphere grids for CLOUDY (`compile stars`).
- Parsing CLOUDY output (`.con`, `.emis`, `.lines`).
- Reading SB99 from anywhere other than the cluster's `Qi` field
  (which TRINITY already pre-computed by the time the snapshot was
  written).

## Tests

```bash
pytest test/test_cloudy_dlaw.py \
       test/test_cloudy_run_loader.py \
       test/test_cloudy_snapshot_to_deck.py \
       test/test_cloudy_package_exports.py \
       test/test_cloudy_cli.py
```

144 tests covering dlaw construction, run-directory parsing, snapshot
validation, template rendering, snapshot picking, manifest writing,
status gating, and the e2e `mockFullrun → deck` round trip.

## Module layout

```
trinity/_output/cloudy/
├── __init__.py                      # re-exports public API
├── trinity_to_cloudy.py             # CLI entry point
├── dlaw.py                          # dlaw block builder
├── run_loader.py                    # parses run directory → RunBundle
├── snapshot_to_deck.py              # snapshot → template values
├── trinity2cloudy.in_template       # bundled CLOUDY template
└── trinity_linelist.dat             # bundled HII line list
```
