# TRINITY

[![CI](https://github.com/JiaWeiTeh/trinity/actions/workflows/ci.yml/badge.svg)](https://github.com/JiaWeiTeh/trinity/actions/workflows/ci.yml)
[![Docs](https://img.shields.io/badge/docs-trinity--web-brightgreen.svg)](https://jiaweiteh.github.io/trinity-web/)
[![arXiv](https://img.shields.io/badge/arXiv-2605.27517-b31b1b.svg)](https://arxiv.org/abs/2605.27517)
[![Python](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](LICENSE)

TRINITY is a feedback-driven bubble evolution code. For a given
giant-molecular-cloud mass, star-formation efficiency, density profile,
and ambient medium, it integrates the time evolution of an expanding
feedback bubble, i.e., shell radius, velocity, thermal state, and force
budget, and resolves the phase transitions and stopping fate of the
shell.

**Documentation: <https://jiaweiteh.github.io/trinity-web/>**

Developed by [Jia Wei Teh](https://jiaweiteh.github.io/) at Universität Heidelberg.

## Quickstart

```bash
git clone https://github.com/JiaWeiTeh/trinity
cd trinity
pip install -r requirements.txt
python run.py param/simple_cluster.param
```

Pure Python 3.9 or newer, with no compilation step. The
[Running TRINITY](https://jiaweiteh.github.io/trinity-web/?view=docs&page=running)
guide covers parameter files, parameter sweeps (including SLURM job arrays on a
cluster), and the output layout.

## Reproducing the figures

The method-paper figures regenerate from the post-processed `.npz`
bundles committed under `paper/methods/data/`, no raw simulation output
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
They are available on request. Please get in touch — the contact address is
on the [project site](https://jiaweiteh.github.io/trinity-web/#contact).

## Citation

If you use TRINITY in published work, please cite the method paper,
Teh et al. (2026), [arXiv:2605.27517](https://arxiv.org/abs/2605.27517):

```bibtex
@ARTICLE{2026arXiv260527517T,
       author = {{Teh}, Jia Wei and {Klessen}, Ralf S. and {Glover}, Simon C.~O. and {Kreckel}, Kathryn},
        title = "{TRINITY: A coupled model of winds, radiation, and photoionised gas in molecular clouds. I. Methods and validation}",
      journal = {arXiv e-prints},
         year = 2026,
        month = may,
          eid = {arXiv:2605.27517},
archivePrefix = {arXiv},
       eprint = {2605.27517},
       adsurl = {https://ui.adsabs.harvard.edu/abs/2026arXiv260527517T},
}
```

GitHub's *Cite this repository* button offers the same paper, from
[`CITATION.cff`](CITATION.cff).

## Contributing

See [`CONTRIBUTING.md`](CONTRIBUTING.md) for the development setup, the tests,
and a map of the repository.

## License

GPL v3: see [`LICENSE`](LICENSE).
