# examples

Three finished TRINITY runs and a notebook that reads them. Both are committed, so
`quickstart.ipynb` works on a fresh clone with no simulation to run first.

```
examples/
├── quickstart.ipynb     the tutorial notebook, committed with its outputs so GitHub shows it
├── runs/                three finished runs, each a dictionary.jsonl and a metadata.json
│   ├── homogeneous/       uniform cloud
│   ├── powerlaw/          n ~ r^-2
│   └── bonnor_ebert/      Bonnor-Ebert sphere
├── thin_run.py          sorts, de-duplicates and thins a run before it is committed
├── export_web.sh        turns the notebook into a page of the website
└── web/                 written by export_web.sh, gitignored
```

`test/test_example_run.py` opens the three runs with the current reader on every `pytest`
run.

## Running the notebook

```bash
pip install -r requirements.txt jupyter smplotlib
jupyter lab examples/quickstart.ipynb
```

`smplotlib` sets the plot style. Without it the notebook still runs, with matplotlib's
default style.

The first cell finds the top of the repository and puts it on `sys.path`. You can start
Jupyter from any folder inside the clone, and you do not have to install TRINITY as a
package.

To run every cell without opening the interface:

```bash
python -m jupyter nbconvert --to notebook --execute --inplace \
    --ExecutePreprocessor.timeout=300 examples/quickstart.ipynb
```

Leave `MPLBACKEND` unset when you do this. With `MPLBACKEND=Agg` the notebook is written
with no figures in it, and nothing warns you.

## The three runs

All three clouds have a mass of 10^6 Msun and a star-formation efficiency of 1%. They
differ in how the gas is arranged. The parameter files are
`param/cloud_example_homogeneous.param`, `param/cloud_example_PL.param` and
`param/cloud_example_BE.param`. The power-law and Bonnor-Ebert runs go through the
energy-driven, transition and momentum-driven phases; the uniform cloud is still
energy-driven when it stops at 15 Myr. The notebook's section "Three clouds, three
histories" shows how each of them ends.

`param/simple_cluster.param`, the run in the top-level README, is not one of them.

## Regenerating the runs

Run the three parameter files from the top of the repository. They are independent, so
they can run at the same time in three terminals:

```bash
python run.py param/cloud_example_homogeneous.param
python run.py param/cloud_example_PL.param
python run.py param/cloud_example_BE.param
```

Then thin each one into `examples/runs/` and run the test:

```bash
python examples/thin_run.py outputs/cloud_example_homogeneous examples/runs/homogeneous  --every N
python examples/thin_run.py outputs/cloud_example_PL          examples/runs/powerlaw     --every N
python examples/thin_run.py outputs/cloud_example_BE          examples/runs/bonnor_ebert --every N

pytest test/test_example_run.py
```

`thin_run.py` does three things and prints what each one did:

1. It drops repeated snapshots. A run can write the same snapshot more than once.
2. It sorts the snapshots by time. A run writes them in the order its buffer is flushed.
3. It keeps every Nth snapshot, and always the first and the last. Every snapshot it
   keeps is exactly what TRINITY wrote.

Replace each `N` with the number that leaves that run about 70 snapshots: its snapshot
count after de-duplication, divided by 70. `thin_run.py` prints both counts, so a first
pass with `--every 1` tells you what to divide. That holds the three runs to a few MB in
total. Run `pytest` afterwards: `test_thinning_preserved_every_phase` fails if a whole
phase was thinned away.

Thin the snapshots and leave `simplify_npoints` at its default. A lower `simplify_npoints`
makes the files smaller too, but it does so by coarsening the profiles inside every
snapshot.

## Sending the notebook to the website

```bash
./examples/export_web.sh
cp -R examples/web/. ../trinity-web/
```

The script runs the notebook first and stops if a code cell produced no output. It then
writes the notebook as a markdown page, with its figures and a copy of the `.ipynb`, into
`examples/web/`. That folder has the same layout as the trinity-web repository, so the
copy puts every file in its place.

The run data stays here. A visitor who downloads the notebook needs this repository to
run it, so the website links to the repository and serves no copy of the runs.

## If the test fails

`test/test_example_run.py` fails when the output format has moved on from the committed
runs. Regenerate the runs with the commands above and leave the test as it is.
