# examples

A finished TRINITY run, and a notebook that reads it. Both are committed so that
`quickstart.ipynb` works on a fresh clone with no simulation to run first.

```
examples/
├── quickstart.ipynb     tutorial notebook (committed with its outputs, so GitHub renders it)
├── lifecycle_run/       a finished run: dictionary.jsonl + metadata.json + the .param sidecar
├── thin_run.py          shrinks a run before committing it
├── export_web.sh        builds examples/web/ for the website
└── web/                 ← derived, gitignored: everything trinity-web needs, in one folder
```

`test/test_example_run.py` lives in the test suite rather than here, so `pytest` picks it up.

## Running the notebook

```bash
pip install -r requirements.txt jupyter
cd examples && jupyter lab quickstart.ipynb
```

The reader wants the `.jsonl` path and picks up `metadata.json` from the same folder,
which is why the run directory has to travel as a whole.

## What the shipped run is

`param/cloud_example_homogeneous.param` — a 10^6 M☉ uniform cloud at 1% star-formation
efficiency. It was chosen over the README quickstart (`simple_cluster`) for one reason:
`simple_cluster` blows out of its 1.7 pc cloud almost immediately and stays energy-driven
for the rest of the run, so it never demonstrates the phase transitions or the stopping
fate. This config runs the full lifecycle.

## Regenerating it

```bash
python run.py param/cloud_example_homogeneous.param
python examples/thin_run.py outputs/cloud_example_homogeneous examples/lifecycle_run --every 4
pytest test/test_example_run.py
```

Two things to know about the thinning step:

- **Why thin at all.** A complete run is tens of MB, most of it the twelve 1-D profile
  arrays carried in every snapshot — too much to commit. Thinning keeps every 4th snapshot
  and leaves each kept snapshot byte-for-byte as the run produced it. The first and last
  snapshots are always kept, so the run still starts and ends where it did.
- **Why not `simplify_npoints`.** Lowering it shrinks the profile arrays instead, and at 20
  points the code itself warns that the reconstruction has dropped below its R² ≥ 0.90
  fidelity bar (hundreds of warnings, some as low as R² = 0.42). Coarser time sampling
  costs nothing inside a snapshot; coarser profiles cost accuracy. Thin the snapshots.

The thinning is the one thing your own run will not reproduce byte-for-byte. Everything
else — the physics, the units, the profile arrays — is exactly what TRINITY wrote.

## Handing the notebook to the website

Execute the notebook and save it with its outputs, then:

```bash
./examples/export_web.sh
cp -R examples/web/. ../trinity-web/public/notebook/
```

`examples/web/` is the single hand-off folder: the rendered `quickstart.html` and a
downloadable copy of the notebook. It is gitignored here because it is derived — the
published copy belongs in trinity-web, not in both repositories.

The run data deliberately does **not** go to the website. A visitor who downloads the
notebook needs the `trinity` package to run it anyway, so the site links to this repository
rather than serving a copy of the dataset.

## If the notebook breaks

`test/test_example_run.py` opens this run with the current reader on every `pytest` run. If
it fails after a change to the output schema, regenerate the example with the commands
above rather than adjusting the test.
