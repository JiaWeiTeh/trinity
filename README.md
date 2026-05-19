# TRINITY

TRINITY is a feedback-driven HII-region evolution code. For a given
giant-molecular-cloud mass, star-formation efficiency, density profile,
and ambient medium, it integrates the time evolution of an expanding
feedback bubble — shell radius, velocity, thermal state, and force
budget — and resolves the phase transitions and stopping fate of the
shell.

Full documentation: <https://jiaweiteh.github.io/trinity-web/>

## Quickstart

```bash
git clone https://github.com/JiaWeiTeh/trinity
cd trinity
pip install -r requirements.txt
python run.py param/simple_cluster.param
```

Requires Python 3.9 or newer. No compilation step.

## License

GPL v3 — see [`LICENSE`](LICENSE).

## Citation

If you use TRINITY in published work, please cite the TRINITY method
paper (Teh et al., in preparation).
