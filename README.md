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

## Requirements

Python 3.9 or newer and the scientific stack (NumPy, SciPy, Astropy,
Matplotlib, pandas) — installed via the command below. No compilation step.

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

## Running on a cluster (SLURM)

On a laptop or a single node, a sweep runs across an in-process worker
pool (`--workers N`). To scale across nodes on an HPC cluster
(e.g. bwForCluster Helix / bwUniCluster), generate a SLURM job array
instead — one task per combination:

```bash
python run.py param/sweep_example.param --emit-jobs jobs/
# edit jobs/submit_sweep.sbatch: set --account / --partition / --time / --mem
sbatch jobs/submit_sweep.sbatch
python run.py --collect-report jobs/      # after the array finishes
```

Set an absolute `path2output` on a work/scratch filesystem for cluster
runs. See the [documentation](https://jiaweiteh.github.io/trinity-web/)
for the full workflow.

## License

GPL v3 — see [`LICENSE`](LICENSE).
