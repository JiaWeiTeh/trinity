# TRINITY

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/)
<a href="https://jiaweiteh.github.io/trinity-web/" target="_blank" rel="noopener noreferrer"><img src="https://img.shields.io/badge/docs-trinity--web-brightgreen.svg" alt="Documentation"></a>

TRINITY is a feedback-driven bubble evolution code. For a given
giant-molecular-cloud mass, star-formation efficiency, density profile,
and ambient medium, it integrates the time evolution of an expanding
feedback bubble — shell radius, velocity, thermal state, and force
budget — and resolves the phase transitions and stopping fate of the
shell.

Full documentation: <https://jiaweiteh.github.io/trinity-web/>

## Repository layout

```
run.py         single entry point for individual runs and parameter sweeps
trinity/       the package: solver, evolution phases, bubble/shell/cloud physics, I/O
param/         .param config files (the tracked ones are worked examples)
lib/default/   bundled defaults — SB99 SPS table + cooling tables (quickstart runs out of the box)
paper/         scripts that regenerate published figures (see "Reproducing the figures")
docs/          Sphinx documentation source
test/          pytest test suite
tools/         small CLI utilities (param generation, audits, output comparisons)
```

## Requirements

Python 3.9 or newer and the scientific stack (NumPy, SciPy, Astropy,
Matplotlib, pandas) — installed via the command below. No compilation
step. Regenerating the publication-quality figures additionally needs a
LaTeX installation, since the plot style renders text with `text.usetex`.

## Quickstart

```bash
git clone https://github.com/JiaWeiTeh/trinity
cd trinity
pip install -r requirements.txt
python run.py param/simple_cluster.param
```

A run is configured by a `.param` file that overrides only the keys it
cares about; everything else falls back to the schema defaults. The
shipped example is just two lines:

```
mCloud    1e5
sfe       0.3
```

Other worked examples are tracked in `param/`: `cloud_example_PL.param`,
`cloud_example_BE.param`, and `cloud_example_homogeneous.param` cover the
three density profiles, and `sweep_example.param`,
`sweep_tuple_example.param`, and `sweep_hybrid_example.param` cover the
sweep syntaxes.

## Parameter sweeps

A `.param` file that uses list or tuple syntax is auto-detected as a
sweep and run across an in-process worker pool:

```bash
python run.py param/sweep_example.param --dry-run     # list the combinations, run nothing
python run.py param/sweep_example.param --workers 4   # run them across 4 workers
```

To scale across nodes on an HPC cluster (e.g. bwForCluster Helix /
bwUniCluster), emit a SLURM job array instead — one task per combination:

```bash
python run.py param/sweep_example.param --emit-jobs jobs/
# edit jobs/submit_sweep.sbatch: set --account / --partition / --time / --mem
sbatch jobs/submit_sweep.sbatch
python run.py --collect-report jobs/      # after the array finishes
```

Set an absolute `path2output` on a work/scratch filesystem for cluster
runs. See the [documentation](https://jiaweiteh.github.io/trinity-web/)
for the full workflow.

## Reproducing the figures

The method-paper figures regenerate from the post-processed `.npz`
bundles committed under `paper/methods/data/` — no raw simulation output
and no extra *data* downloads needed. The figure scripts do need the `[plots]`
extra (`pip install -e ".[plots]"`) and a LaTeX install (the plot style uses
`text.usetex`):

```bash
python paper/methods/make_figures.py           # all figures → paper/plots/
python paper/methods/make_figures.py teaser     # or one figure by short name
```

## Data on request

Raw simulation outputs, the full SPS/cooling libraries, and the figure
run-sets are not committed to the repository because of their size.
They are available on request — contact <jiaweiteh.astro@gmail.com>.

## Citation

If you use TRINITY in your research, please consider citing the method
paper, Teh et al. (2026) (arXiv [2605.27517](https://arxiv.org/abs/2605.27517)).
A BibTeX entry is available from
[ADS](https://ui.adsabs.harvard.edu/abs/2026arXiv260527517T/abstract).

## License

GPL v3 — see [`LICENSE`](LICENSE).
