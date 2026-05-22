# TRINITY

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/)
[![Documentation](https://img.shields.io/badge/docs-trinity--web-brightgreen.svg)](https://jiaweiteh.github.io/trinity-web/)

TRINITY is a feedback-driven HII-region evolution code. For a given
giant-molecular-cloud mass, star-formation efficiency, density profile,
and ambient medium, it integrates the time evolution of an expanding
feedback bubble — shell radius, velocity, thermal state, and force
budget — and resolves the phase transitions and stopping fate of the
shell.

<!-- TODO: drop in an overview figure here, e.g. a radius/phase evolution plot.
     ![TRINITY shell evolution](docs/source/_static/overview.png) -->

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

## License

GPL v3 — see [`LICENSE`](LICENSE).

## Citation

If you use TRINITY in published work, please cite the TRINITY method
paper (Teh et al., in preparation).
