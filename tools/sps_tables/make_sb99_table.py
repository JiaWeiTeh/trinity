#!/usr/bin/env python3
"""Renewed Starburst99 tables for TRINITY from pySTARBURST99 (docs/dev/sps-tables/PLAN.md).

Subcommands (numpy only, except `run`, which needs pystarburst99 and so runs on Helix in the pysb99 env):

  run       --out DIR --Z MW --rot 1 --sed WM --mup 120 [--mtotal 1e6] [--speed DEFAULT] [--grid-override a b c]
            one pySTARBURST99 model on TRINITY's time grid (1e4 yr, then 1e5-yr steps to 1e8 yr); results saved in DIR
  table     --run DIR --out NEW.csv [--bands BANDS.csv]
            the full pySTARBURST99 table in TRINITY's layout (t, Qi, fi, Lbol, Lmech_total, pdot_W, Lmech_W, then L_FUV, L_LW,
            Q_FUV, Q_LW; log10 except t) and, with --bands, every band integral and self-check per age
  compare   --old OLD.csv --new NEW.csv --out DIFF.csv
            Gate R: new - old per column [dex] at matched t; RMS and max over t <= 3 Myr, 3-10 Myr, > 10 Myr
  hybrid    --old OLD.csv --new NEW.csv --out HYBRID.csv [--note TEXT ...]
            OLD's rows kept byte for byte, plus the four band columns scaled by OLD's own Ln (PLAN D-S7)
  selfcheck the band integrator on the Draine 1978 field (Sternberg et al. 2014 eq. 2): LW photon flux and 6-13.6 eV energy

Assumptions about pySTARBURST99's files are marked A1-A4 and gated in PLAN.md Phase 2 (G-S1, G-S2).
"""
import argparse
import datetime
import hashlib
import sys
from pathlib import Path

import numpy as np

HC_EV_A = 12398.419843       # h c [eV A]
HC_ERG_A = 1.98644586e-8     # h c [erg A]
C_A_S = 2.99792458e18        # c [A/s]
YR_S = 3.15576e7             # Julian year [s]
E_SN_ERG = 1e51              # kinetic energy per supernova (Leitherer et al. 1999 sect. 10)
HABING_CGS = 1.6e-3          # the Habing field's 6-13.6 eV energy flux [erg cm^-2 s^-1] (trinity/shell_structure/pdr.py)
LAM_LYC = 912.0              # Starburst99's Lyman edge [A]: ion_flux_HI counts photons at wave <= 912
BANDS = {                    # [A]; lower edge 912 so that L_i and the bands partition L_bol as the table's fi does
    'FUV': (LAM_LYC, HC_EV_A / 6.0),      # 6.0-13.6 eV, the Habing band (G0)
    'LW': (LAM_LYC, 1108.0),              # Sternberg et al. 2014's 912-1108 A Lyman-Werner band (1108 A = 11.19 eV) (I_UV)
    'HI06': (LAM_LYC, HC_EV_A / 11.0),    # 11.0-13.6 eV, Hosokawa & Inutsuka 2006's FUV band (their G_FUV counts photons)
}
OLD_COLS = ('t', 'Qi', 'fi', 'Lbol', 'Lmech_total', 'pdot_W', 'Lmech_W')
NEW_COLS = ('L_FUV', 'L_LW', 'Q_FUV', 'Q_LW')
GRID = (4.0, 8.0, 5.0)       # A3: log_age_start, log_age_end, log_age_step with log_intervals False -> 1e4 + k 1e5 yr


def trapz(y, x):
    """The trapezoid rule written out (numpy 1 names it trapz, numpy 2 trapezoid)."""
    y, x = np.asarray(y, float), np.asarray(x, float)
    return float(np.sum(0.5 * (y[1:] + y[:-1]) * np.diff(x)))


def band_integrals(wave, L_lam, lo, hi):
    """(energy [erg/s], photons [1/s]) of L_lam [erg/s/A] on wave [A, strictly increasing] between lo and hi [A].
    The band edges are added as nodes (log-linear interpolation of L_lam), then the trapezoid rule in lambda."""
    w, f = np.asarray(wave, float), np.asarray(L_lam, float)
    if np.any(np.diff(w) <= 0):
        raise ValueError('wave must increase strictly')
    if lo < w[0] or hi > w[-1]:
        raise ValueError(f'band {lo}-{hi} A lies outside the grid {w[0]}-{w[-1]} A')
    inside = (w > lo) & (w < hi)
    f_lo, f_hi = 10.0 ** np.interp([lo, hi], w, np.log10(np.maximum(f, 1e-300)))
    ww = np.concatenate(([lo], w[inside], [hi]))
    ff = np.concatenate(([f_lo], f[inside], [f_hi]))
    return trapz(ff, ww), trapz(ff * ww, ww) / HC_ERG_A


def pysb99_photons(wave, L_lam, limit):
    """Photons/s at wavelengths <= limit [A] counted as pySTARBURST99 counts them for ion_flux_*.txt (tools.compute_ion_flux): the
    trapezoid in frequency over the grid points at wave <= limit only, no node at the edge (G-S2 ii; h differs there: 1e-5 dex)."""
    w, f = np.asarray(wave, float), np.asarray(L_lam, float)
    m = w <= limit
    nu = C_A_S / w[m]
    return abs(trapz(f[m] * w[m] ** 2 / C_A_S / (HC_ERG_A / w[m]), nu))


def draine_field_check():
    """(LW photon flux [cm^-2 s^-1], 6-13.6 eV energy flux in Habing units) of the Draine 1978 field through band_integrals.
    Expected 2.07e7 (Sternberg et al. 2014 eq. 4) and about 1.7 (their footnote 4); computed 2026-11-03: 2.065e7 and 1.672."""
    lam = np.arange(900.0, 2100.0001, 0.05)
    F_nu = 1.068e-3 / lam - 1.719 / lam**2 + 6.853e2 / lam**3        # 4 pi I_nu [photons cm^-2 s^-1 Hz^-1], lambda in A
    E_lam = F_nu * C_A_S / lam**2 * HC_ERG_A / lam                     # [erg cm^-2 s^-1 A^-1]
    _, lw_photons = band_integrals(lam, E_lam, *BANDS['LW'])
    fuv_energy, _ = band_integrals(lam, E_lam, *BANDS['FUV'])
    return lw_photons, fuv_energy / HABING_CGS


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_table(path):
    """A TRINITY-layout CSV ('#' lines skipped, header row, log10 columns except t) as a dict of arrays, plus its data lines."""
    lines = [ln.rstrip('\n') for ln in Path(path).read_text().splitlines()]
    body = [ln for ln in lines if ln.strip() and not ln.lstrip().startswith('#')]
    names = [s.strip() for s in body[0].split(',')]
    data = np.array([[float(v) for v in ln.split(',')] for ln in body[1:]])
    return {n: data[:, i] for i, n in enumerate(names)}, body[1:]


def ln_of(tab):
    """L_n [erg/s] = L_bol (1 - 10**fi) of a TRINITY-layout table."""
    return 10.0 ** tab['Lbol'] * (1.0 - 10.0 ** tab['fi'])


def cmd_run(a):
    import pystarburst99                                  # the pysb99 env only (numpy >= 2); never TRINITY's env
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    start, end, step = a.grid_override or GRID
    settings = dict(output_folder=str(out), Z=a.Z, track_choice='GENEC', rotation=bool(a.rot), SED_library=a.sed,
                    spectra_library='WM', star_formation_option='ISF', M_total=a.mtotal, IMF_exponents=[1.3, 2.3],
                    IMF_mass_limits=[0.1, 0.5, a.mup], maximum_SN_mass=a.mup, run_speed_mode=a.speed, log_intervals=False,
                    log_age_start=start, log_age_end=end, log_age_step=step, plot_results_check=False, overwrite_results=True,
                    beta_tools=True)                      # required, a bool with no toml default (PLAN sect. 1 (a))
    sb = pystarburst99.Compiler(**settings)
    sb.population_synthesis()
    sb.save_results()                                     # empties the folder first, then writes the files PLAN sect. 1 lists
    version = getattr(pystarburst99, '__version__', 'unknown')
    (out / 'provenance.txt').write_text(
        f"pystarburst99 {version}; {datetime.datetime.now(datetime.timezone.utc).isoformat()}\n"
        f"settings: {settings}\nargv: {' '.join(sys.argv)}\n")
    t = np.loadtxt(out / 'SED_times.txt')
    want = 1e4 + 1e5 * np.arange(1000)
    ok = len(t) == 1000 and np.max(np.abs(t - want)) <= 0.5
    print(f"G-S1 grid: n={len(t)} first={t[0]:.6e} second={t[1] if len(t) > 1 else float('nan'):.6e} last={t[-1]:.6e} "
          f"max|dt|={np.max(np.abs(t[:min(len(t), 1000)] - want[:min(len(t), 1000)])):.3g} yr -> {'PASS' if ok else 'FAIL'}")


def at_ages(steps, values, ages):
    """Values on the internal time steps, at the output ages: linear in time between the two bracketing steps, exact on a
    step -- pySTARBURST99's own rule for the SEDs it saves (tools.interp_time); the files hold log10 values."""
    i = np.clip(np.searchsorted(steps, ages, side='right') - 1, 0, len(steps) - 2)
    w = (ages - steps[i]) / (steps[i + 1] - steps[i])
    lo, hi = values[i], values[i + 1]
    with np.errstate(invalid='ignore'):
        return np.where(w == 0, lo, np.where(w == 1, hi, (1 - w) * lo + w * hi))


def load_run(d):
    d = Path(d)
    t = np.loadtxt(d / 'SED_times.txt')                                           # the output ages, yr: the SED rows (A1)
    steps = np.loadtxt(d / 'timesteps.txt')                                       # the internal steps, yr: the rows below (A2)
    wave = np.loadtxt(d / 'SED_wavelength.txt')                                   # A (A1)
    sed = np.load(d / 'pySB_SED_stellar.npy')                                     # log10 L_lambda [erg/s/A], (n_t, n_wave) (A1)
    files = (('Lbol', 'bololum.txt'), ('Qi', 'ion_flux_HI.txt'), ('QHeI', 'ion_flux_HeI.txt'), ('QHeII', 'ion_flux_HeII.txt'),
             ('Lmech_W', 'windpower.txt'), ('pdot_W', 'windmom.txt'))             # log10 cgs, one row per internal step (A2)
    raw = {k: np.atleast_1d(np.loadtxt(d / f)) for k, f in files}
    sn = np.atleast_2d(np.loadtxt(d / 'SNrate.txt'))                              # [step yr, log10 rate per yr] (A4)
    shapes = {k: v.shape for k, v in raw.items()}
    shapes.update(t=t.shape, steps=steps.shape, sed=sed.shape, wave=wave.shape, SNrate=sn.shape)
    if (sed.shape != (len(t), len(wave)) or any(v.shape != steps.shape for v in raw.values())
            or t[0] < steps[0] or t[-1] > steps[-1]):
        raise SystemExit(f"A1/A2 FAIL: the SED is not one row per output age, or a file not one row per internal step "
                         f"covering the output ages: {shapes}")
    cols = {k: at_ages(steps, v, t) for k, v in raw.items()}                     # at the output ages, as the package does the SED
    return t, wave, sed, cols, sn


def cmd_table(a):
    t, wave, sed, cols, sn = load_run(a.run)
    rows, bands = [], []
    worst = {'bolo': 0.0, 'qi': 0.0, 'edge': 0.0}
    log_sn = np.interp(t, sn[:, 0], sn[:, 1], left=-30.0, right=sn[-1, 1])        # log10 SN rate [1/yr] at the output ages
    for i, ti in enumerate(t):
        L_lam = 10.0 ** sed[i]
        L_bol_sed = trapz(L_lam, wave)
        L_lyc, Q_lyc = band_integrals(wave, L_lam, wave[0], LAM_LYC)
        Q_pkg = pysb99_photons(wave, L_lam, LAM_LYC)                              # G-S2 (ii): the package's own count
        d_bolo = np.log10(L_bol_sed) - cols['Lbol'][i]
        d_qi = np.log10(max(Q_pkg, 1e-300)) - cols['Qi'][i]
        worst['bolo'] = max(worst['bolo'], abs(d_bolo))
        worst['qi'] = max(worst['qi'], abs(d_qi))
        worst['edge'] = max(worst['edge'], abs(np.log10(max(Q_lyc, 1e-300) / max(Q_pkg, 1e-300))))
        fi = np.log10(max(L_lyc, 1e-300) / 10.0 ** cols['Lbol'][i])          # no ionising flux at late ages: -300, not -inf
        b = {k: band_integrals(wave, L_lam, lo, hi) for k, (lo, hi) in BANDS.items()}
        L_sn = 10.0 ** log_sn[i] * E_SN_ERG / YR_S
        L_tot = np.log10(10.0 ** cols['Lmech_W'][i] + L_sn)
        rows.append((ti, cols['Qi'][i], fi, cols['Lbol'][i], L_tot, cols['pdot_W'][i], cols['Lmech_W'][i],
                     np.log10(b['FUV'][0]), np.log10(b['LW'][0]), np.log10(b['FUV'][1]), np.log10(b['LW'][1])))
        bands.append((ti, np.log10(L_bol_sed), cols['Lbol'][i], np.log10(max(Q_lyc, 1e-300)), cols['Qi'][i], cols['QHeI'][i],
                      cols['QHeII'][i], fi, b['FUV'][0], b['FUV'][1], b['LW'][0], b['LW'][1], b['HI06'][0], b['HI06'][1],
                      log_sn[i]))
    head = '# ' + Path(a.run).name + f' ({datetime.date.today()}); columns log10 cgs except t [yr]\n' + ','.join(OLD_COLS + NEW_COLS)
    fmt = '%.4e,' + ','.join(['%.4f'] * 10)
    np.savetxt(a.out, np.array(rows), fmt=fmt, header=head, comments='')
    if a.bands:
        np.savetxt(a.bands, np.array(bands), fmt='%.6e', delimiter=',', comments='',
                   header='t_yr,logLbol_sed,logLbol_file,logQi_sed,logQi_file,logQHeI,logQHeII,fi,L_FUV,Q_FUV,L_LW,Q_LW,L_HI06,Q_HI06,logSNrate')
    print(f"G-S2 (i) worst |dlog Lbol| = {worst['bolo']:.4f} dex -> {'PASS' if worst['bolo'] <= 0.005 else 'FAIL'}")
    print(f"G-S2 (ii) worst |dlog Qi| = {worst['qi']:.4f} dex -> {'PASS' if worst['qi'] <= 0.005 else 'FAIL'}")
    print(f"G-S2 (ii) report: the bands' 912 A edge (an interpolated node; also fi's) vs that count: worst |dlog Q| = "
          f"{worst['edge']:.4f} dex")
    for edge in (LAM_LYC, BANDS['LW'][1], BANDS['HI06'][1], BANDS['FUV'][1]):
        j = np.searchsorted(wave, edge)
        print(f"G-S2 (iv) edge {edge:.1f} A between grid points {wave[j - 1]:.2f} and {wave[j]:.2f} A")
    L_lam = 10.0 ** sed[np.argmin(np.abs(t - 1.01e6))]                            # the 1 Myr row: edge sensitivity
    for k in ('FUV', 'LW'):
        lo, hi = BANDS[k]
        e0 = band_integrals(wave, L_lam, lo, hi)[0]
        e_lo = band_integrals(wave, L_lam, lo + 2.0, hi)[0]
        e_hi = band_integrals(wave, L_lam, lo, hi - 2.0)[0]
        print(f"G-S2 (iv) {k} at ~1 Myr: lower edge +2 A changes L by {100 * (e_lo / e0 - 1):+.2f} %, upper edge -2 A by "
              f"{100 * (e_hi / e0 - 1):+.2f} %")


def cmd_compare(a):
    old, _ = read_table(a.old)
    new, _ = read_table(a.new)
    if len(old['t']) != len(new['t']) or np.max(np.abs(old['t'] - new['t'])) > 0.5:
        raise SystemExit(f"grids differ: {len(old['t'])} vs {len(new['t'])} rows, max|dt| = "
                         f"{np.max(np.abs(old['t'][:min(len(old['t']), len(new['t']))] - new['t'][:min(len(old['t']), len(new['t']))])):.3g} yr")
    t = old['t']
    cols = [c for c in OLD_COLS[1:] if c in new]
    diff = np.column_stack([t] + [new[c] - old[c] for c in cols])
    np.savetxt(a.out, diff, fmt='%.6e', delimiter=',', comments='', header='t_yr,' + ','.join('d_' + c for c in cols))
    windows = (('t <= 3 Myr', t <= 3e6), ('3-10 Myr', (t > 3e6) & (t <= 1e7)), ('> 10 Myr', t > 1e7))
    print(f"Gate R: {Path(a.new).name} - {Path(a.old).name} [dex]")
    print(f"{'column':12s} " + ' '.join(f"{w[0]:>22s}" for w in windows) + '   (RMS / max|d|)')
    verdict = True
    for c in cols:
        d = new[c] - old[c]
        cells = []
        for name, m in windows:
            rms, mx = np.sqrt(np.mean(d[m] ** 2)), np.max(np.abs(d[m]))
            cells.append(f"{rms:9.4f} /{mx:9.4f}")
            if c in ('Qi', 'fi', 'Lbol') and name.startswith('t <=') and not (rms <= 0.03 and mx <= 0.10):  # NaN fails
                verdict = False
        print(f"{c:12s} " + ' '.join(f"{s:>22s}" for s in cells))
    print(f"Gate R radiative columns (Qi, fi, Lbol; t <= 3 Myr; RMS <= 0.03, max <= 0.10 dex): {'PASS' if verdict else 'FAIL'}")
    print("winds and supernovae (Lmech_W, pdot_W, Lmech_total): report only, expected to differ (PLAN sect. 1)")


def cmd_hybrid(a):
    old, old_lines = read_table(a.old)
    new, _ = read_table(a.new)
    if len(old['t']) != len(new['t']) or np.max(np.abs(old['t'] - new['t'])) > 0.5:
        raise SystemExit('grids differ: run `compare` first')
    ln_old, ln_new = ln_of(old), ln_of(new)
    scaled = {c: np.log10(10.0 ** new[c] / ln_new * ln_old) for c in NEW_COLS}      # D-S7: pySTARBURST99's share x the old L_n
    L_FUV, L_LW, Q_FUV, Q_LW = (10.0 ** scaled[c] for c in NEW_COLS)
    if not (np.all(np.isfinite(list(scaled.values()))) and np.all(L_FUV < ln_old) and np.all(L_LW < L_FUV)
            and np.all(Q_LW < Q_FUV) and np.all(np.diff(old['t']) > 0)):
        raise SystemExit('G-S3 FAIL: a new column is not finite or not ordered (L_FUV < Ln, L_LW < L_FUV, Q_LW < Q_FUV)')
    head = [f"# {Path(a.out).name}: {Path(a.old).name} (sha256 {sha256(a.old)}) with four far-UV columns appended",
            f"# from {Path(a.new).name} (sha256 {sha256(a.new)}) as pySTARBURST99's share of L_n times this table's L_n (D-S7)",
            "# L_FUV: 912-2066 A, 6.0-13.6 eV [erg/s, log10]; L_LW: 912-1108 A, the Lyman-Werner band [erg/s, log10]; Q_FUV, Q_LW: photons/s in the same bands "
            "[log10]; the first seven columns are the shipped file's, byte for byte",
            f"# written {datetime.date.today()} by tools/sps_tables/make_sb99_table.py hybrid"]
    head += [f"# {n}" for n in (a.note or [])]
    out = head + [','.join(OLD_COLS + NEW_COLS)]
    for i, ln in enumerate(old_lines):
        out.append(ln + ',' + ','.join(f"{scaled[c][i]:.4f}" for c in NEW_COLS))
    Path(a.out).write_text('\n'.join(out) + '\n')
    check, _ = read_table(a.out)
    assert len(check['t']) == len(old['t']) and all(np.array_equal(check[c], old[c]) for c in OLD_COLS)
    if not (np.all(10.0 ** check['L_FUV'] < ln_of(check)) and np.all(check['L_LW'] < check['L_FUV'])
            and np.all(check['Q_LW'] < check['Q_FUV'])):      # the written (rounded) values, as the loader will read them
        raise SystemExit('G-S3 FAIL after rounding: a new column is not ordered (L_FUV < Ln, L_LW < L_FUV, Q_LW < Q_FUV)')
    print(f"G-S3: {a.out}: {len(old_lines)} rows, old columns identical, new columns ordered; sha256 {sha256(a.out)}")
    f = 10.0 ** scaled['L_FUV'] / ln_old
    print(f"f_FUV = L_FUV/Ln: {f.min():.3f}-{f.max():.3f}; mean over t <= 5.5 Myr {f[old['t'] <= 5.5e6].mean():.3f} (Paper III: 0.67)")


def cmd_selfcheck(_a):
    lw, hab = draine_field_check()
    ok = abs(lw / 2.07e7 - 1) <= 0.01 and 1.6 < hab < 1.8
    print(f"G-S2 (iii) Draine field: LW photons {lw:.4e} cm^-2 s^-1 (2.07e7), 6-13.6 eV {hab:.3f} Habing (1.6-1.8) -> "
          f"{'PASS' if ok else 'FAIL'}")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest='cmd', required=True)
    r = sub.add_parser('run')
    r.add_argument('--out', required=True)
    r.add_argument('--Z', default='MW')
    r.add_argument('--rot', type=int, default=1)
    r.add_argument('--sed', default='WM', choices=['WM', 'FW', 'PoWR'])
    r.add_argument('--mup', type=float, default=120.0)
    r.add_argument('--mtotal', type=float, default=1e6)
    r.add_argument('--speed', default='DEFAULT')
    r.add_argument('--grid-override', type=float, nargs=3, default=None, metavar=('START', 'END', 'STEP'))
    r.set_defaults(func=cmd_run)
    tb = sub.add_parser('table')
    tb.add_argument('--run', required=True)
    tb.add_argument('--out', required=True)
    tb.add_argument('--bands', default=None)
    tb.set_defaults(func=cmd_table)
    c = sub.add_parser('compare')
    c.add_argument('--old', required=True)
    c.add_argument('--new', required=True)
    c.add_argument('--out', required=True)
    c.set_defaults(func=cmd_compare)
    h = sub.add_parser('hybrid')
    h.add_argument('--old', required=True)
    h.add_argument('--new', required=True)
    h.add_argument('--out', required=True)
    h.add_argument('--note', action='append')
    h.set_defaults(func=cmd_hybrid)
    s = sub.add_parser('selfcheck')
    s.set_defaults(func=cmd_selfcheck)
    a = p.parse_args(argv)
    a.func(a)


if __name__ == '__main__':
    main()
