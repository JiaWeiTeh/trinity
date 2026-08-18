# examples

Three finished TRINITY runs, and a notebook that reads them. All are committed so that
`quickstart.ipynb` works on a fresh clone with no simulation to run first.

```
examples/
├── quickstart.ipynb     tutorial notebook (committed with its outputs, so GitHub renders it)
├── runs/                three finished runs, each dictionary.jsonl + metadata.json
│   ├── homogeneous/       uniform cloud
│   ├── powerlaw/          rho ~ r^-2
│   └── bonnor_ebert/      Bonnor-Ebert sphere
├── thin_run.py          de-duplicates, sorts and thins a run before committing it
├── export_web.sh        renders the notebook into the website
└── web/                 ← derived, gitignored: mirrors trinity-web's layout
```

`test/test_example_run.py` lives in the test suite rather than here, so `pytest` picks it up.

## Running the notebook

```bash
pip install -r requirements.txt jupyter
jupyter lab examples/quickstart.ipynb
```

To re-run it without opening the interface — useful when refreshing the committed
outputs, or if your Jupyter install is unhappy:

```bash
python -m jupyter nbconvert --to notebook --execute --inplace \
    --ExecutePreprocessor.timeout=300 examples/quickstart.ipynb
```

Do not set `MPLBACKEND=Agg` when you do: it overrides the inline backend and the
notebook is written out with no figures in it, silently.

The first cell walks up from the working directory to find the repository root and puts
it on `sys.path`, so the notebook runs whether you launch jupyter at the top level or
inside `examples/`, and without `pip install -e .` first — `requirements.txt` installs
the scientific stack but not TRINITY itself.

The reader wants the `.jsonl` path and picks up `metadata.json` from the same folder,
which is why a run directory has to travel as a whole.

## What the shipped runs are

The three tracked density-profile examples, identical in cloud mass (10^6 M☉) and
star-formation efficiency (1%) so that the profile is the only thing that differs. All
three run the full lifecycle — energy-driven, transition, momentum-driven — and end
differently: the uniform cloud is still expanding at the 15 Myr stopping time, while the
power-law and Bonnor-Ebert shells both collapse, at ~1.9 and ~4.0 Myr.

The README quickstart (`simple_cluster`) is deliberately *not* among them. It blows out of
its 1.7 pc cloud almost immediately and stays energy-driven for the rest of the run, so it
never demonstrates a phase transition or a stopping fate.

## Regenerating them

```bash
python run.py param/cloud_example_homogeneous.param
python run.py param/cloud_example_PL.param
python run.py param/cloud_example_BE.param

python examples/thin_run.py outputs/cloud_example_homogeneous examples/runs/homogeneous  --every 6
python examples/thin_run.py outputs/cloud_example_PL          examples/runs/powerlaw     --every 4
python examples/thin_run.py outputs/cloud_example_BE          examples/runs/bonnor_ebert --every 4

pytest test/test_example_run.py
```

`thin_run.py` does three things, and only the last one loses anything:

1. **De-duplicates.** Raw output repeats snapshots — the homogeneous run wrote 908 lines
   of which 424 were byte-identical repeats, leaving 483 distinct ones. The other two had
   none. This is a known defect in the snapshot writer, not something you did wrong.
2. **Sorts by time.** Snapshots are written in buffer-flush order, not chronological
   order, so anything plotting a raw run in file order gets a zig-zag.
3. **Thins.** Keeps every Nth of what remains, so the runs fit in a git repository. Each
   kept snapshot is byte-for-byte what TRINITY wrote; the first and last are always kept.

The `--every` values differ because the runs differ in length — 483 / 251 / 282 snapshots
after de-duplication. The values above land them at 82 / 64 / 72 snapshots, about 5.3 MB
in total. Raise them for a smaller repository, but re-run `pytest` afterwards:
`test_thinning_preserved_every_phase` fails if a phase gets thinned out of existence, and
`test_snapshots_are_chronological_and_unique` fails if raw output slips through unsorted.
- **Why not `simplify_npoints`.** Lowering it shrinks the profile arrays instead, and at 20
  points the code itself warns that the reconstruction has dropped below its R² ≥ 0.90
  fidelity bar (hundreds of warnings, some as low as R² = 0.42). Coarser time sampling
  costs nothing inside a snapshot; coarser profiles cost accuracy. Thin the snapshots.

De-duplicating and sorting are the two steps your own raw run will not match. Everything
else — the physics, the units, the profile arrays — is exactly what TRINITY wrote.

## Handing the notebook to the website

```bash
./examples/export_web.sh
cp -R examples/web/. ../trinity-web/
```

The export runs the notebook itself before converting it, and refuses to continue if any
cell came back empty. Converting an unexecuted notebook produces a page of code with no
results and no figures — which looks perfectly fine until you read it.

The export writes markdown, not HTML, so the website renders the notebook as one of its
own documentation pages — the site's typography, its code blocks and copy button, its
"on this page" rail, its KaTeX. `examples/web/` mirrors trinity-web's own layout
(`src/docs/` for the page, `public/notebook/` for the figures and the downloadable
`.ipynb`), so the copy above puts each file where it belongs in one command. It is
gitignored here because it is derived — the published copy belongs in trinity-web.

The run data deliberately does **not** go to the website. A visitor who downloads the
notebook needs the `trinity` package to run it anyway, so the site links to this repository
rather than serving a copy of the dataset.

## If the notebook breaks

`test/test_example_run.py` opens this run with the current reader on every `pytest` run. If
it fails after a change to the output schema, regenerate the example with the commands
above rather than adjusting the test.
