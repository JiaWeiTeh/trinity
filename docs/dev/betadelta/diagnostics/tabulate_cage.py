#!/usr/bin/env python3
"""Tabulate the steep run's cage-vs-no-cage roots + interior profiles per segment.

This is the slow half of the rootmap_cage animation, factored out so the GIF
(make_rootmap_gif.py) is a pure read of the table this writes. For every implicit
segment of the canonical steep hybr trajectory (docs/dev/data/stalling_steep_*.csv)
it records, at the SAME bubble state:

  no-cage (hybr) : the recorded (beta, delta) root, its residual g, the re-solved
                   interior profiles v(r), n(r), and the ionization-front radius
                   R_IF (a shell-structure solve; ~R2 for a dense absorbing shell).
  cage  (legacy) : the REAL bounded legacy solve (betadelta_solver='legacy'),
                   NOT a geometric clip -- its actual in-box optimum, residual g,
                   and profiles. The legacy guess is threaded from the cage's own
                   previous root (drift-capped continuity, like a driven run), so
                   this is a faithful caged *trajectory*, not a single shot.

Profiles are resampled onto a shared uniform radial-fraction grid (0=R1, 1=R2)
so the GIF curves are smooth. Density is converted to cgs (cm^-3).

The hybr structure solve is ~2 s/segment; the real legacy solve is ~60 s/segment
(it grids ~25 structure solves through the f-pole, L-BFGS-B on fallback), so a
full ~133-segment run is ~2 hr. Writes rootmap_cage_table.npz (gitignored).

REQUIRES the pinned deps (numpy<2, scipy<2) + the steep config probe_cloudPL.param:
  PYTHONPATH=<repo> python docs/dev/betadelta/diagnostics/tabulate_cage.py [--limit N] [--out PATH]
"""

import argparse
import csv
import logging
import time
from pathlib import Path

logging.disable(logging.CRITICAL)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from trinity._functions import unit_conversions as cvt  # noqa: E402
from trinity._input import read_param  # noqa: E402
from trinity._input.dictionary import updateDict  # noqa: E402
from trinity.sps.update_feedback import get_current_sps_feedback  # noqa: E402
import trinity.phase1b_energy_implicit.get_betadelta as GB  # noqa: E402
from trinity.cooling.non_CIE import read_cloudy as non_CIE  # noqa: E402
from trinity.shell_structure.shell_structure import shell_structure_pure  # noqa: E402
import trinity.main as tmain  # noqa: E402
import trinity.phase1_energy.run_energy_phase as RE  # noqa: E402

HERE = Path(__file__).resolve().parent
CSV = HERE.parents[1] / "analysis" / "data" / "stalling_steep_1e6_alpha-2.csv"
PARAM = HERE / "probe_cloudPL.param"
OUT = HERE / "rootmap_cage_table.npz"
NF = 600  # shared uniform radial-fraction grid resolution
BOX_B, BOX_D = (0.0, 1.0), (-1.0, 0.0)  # the cage (legacy clamp), for the seed only


class _InitDone(Exception):
    pass


def init_params(param_path):
    """Run the real init (cloud/SPS/CIE) but stop before phase 1a."""
    RE.run_energy = lambda params: (_ for _ in ()).throw(_InitDone())
    params = read_param.read_param(str(param_path))
    try:
        tmain.start_expansion(params)
    except _InitDone:
        pass
    return params


def set_state(params, row):
    """Set the segment's bubble state exactly as reconstruct_vprofile/cage_compare do
    (validated to match the CSV Pb/dMdt/v_min to the digit)."""
    t = float(row["t_now"])
    R2, v2, Eb = float(row["R2"]), float(row["v2"]), float(row["Eb"])
    params["t_now"].value = t
    updateDict(params, get_current_sps_feedback(t, params))
    c, h, n = non_CIE.get_coolingStructure(params)
    params["cStruc_cooling_nonCIE"].value = c
    params["cStruc_heating_nonCIE"].value = h
    params["cStruc_net_nonCIE_interpolation"].value = n
    params["current_phase"].value = "implicit"
    for k, val in (("R2", R2), ("v2", v2), ("Eb", Eb), ("cool_alpha", t / R2 * v2)):
        params[k].value = val
    # a sane warm-start seed for the dMdt fsolve (the recorded hybr value)
    params["bubble_dMdt"].value = float(row["bubble_dMdt"])
    return R2


def profiles(props, R2, f_grid):
    """Resample v(r) [pc/Myr] and n(r) [cm^-3] onto the shared radial-fraction grid."""
    r = np.asarray(props.bubble_r_arr, dtype=float)
    v = np.asarray(props.bubble_v_arr, dtype=float)
    n_cgs = np.asarray(props.bubble_n_arr, dtype=float) * cvt.ndens_au2cgs
    f = (r - props.R1) / (R2 - props.R1)
    o = np.argsort(f)
    f, v, n_cgs = f[o], v[o], n_cgs[o]
    return np.interp(f_grid, f, v), np.interp(f_grid, f, n_cgs)


def g_residual(det, Lmech_total):
    """The pole-free hybr residual g = gE^2 + gT^2 from detailed components."""
    gE = (det.Edot_from_beta - det.Edot_from_balance) / Lmech_total
    return float(gE) ** 2 + float(det.T_residual) ** 2


# committed-CSV column names (friendlier than the internal npz keys)
SCALAR_COLS = {
    "t": "t",
    "hb": "beta_nocage",
    "hd": "delta_nocage",
    "cb": "beta_cage",
    "cd": "delta_cage",
    "g_h": "g_nocage",
    "g_c": "g_cage",
    "f_c": "f_cage",
    "T0": "T0",
    "conv_c": "conv_cage",
    "cage_ok": "cage_ok",
    "Lmech_W": "Lmech_W",
    "Lmech_SN": "Lmech_SN",
    "Lmech_total": "Lmech_total",
    "R2": "R2",
    "v2": "v2",
    "R1_h": "R1_nocage",
    "R1_c": "R1_cage",
    "R_IF": "R_IF_nocage",
    "rShell": "rShell_nocage",
    "vmin_h": "vmin_nocage",
    "vmin_c": "vmin_cage",
    "c_sound": "c_sound",
}


def save_outputs(npz_path, data):
    """Write the npz cache (fast) AND the two committed CSVs (trackable, so the GIF
    reproduces from git after the container is gone). scalars = one row per segment;
    profiles = long format (one row per (segment, radial-fraction) sample)."""
    npz_path = Path(npz_path)
    np.savez(npz_path, **data)
    prefix = npz_path.stem.replace("_table", "")  # rootmap_cage_table.npz -> rootmap_cage

    scal = pd.DataFrame({csv_name: data[k] for k, csv_name in SCALAR_COLS.items()})
    scal.insert(0, "segment", np.arange(len(scal)))
    scalar_path = npz_path.with_name(f"{prefix}_scalars.csv")
    scal.to_csv(scalar_path, index=False, float_format="%.8g")

    f_grid, nseg, nf = data["f_grid"], len(data["t"]), len(data["f_grid"])
    prof = pd.DataFrame(
        {
            "segment": np.repeat(np.arange(nseg), nf),
            "f": np.tile(f_grid, nseg),
            "v_nocage": data["v_h"].ravel(),
            "n_nocage_cm3": data["n_h"].ravel(),
            "v_cage": data["v_c"].ravel(),
            "n_cage_cm3": data["n_c"].ravel(),
        }
    )
    # gzip the bulk profiles (80k float rows -- diffing them is meaningless);
    # pandas infers gzip from the .gz suffix on read and write alike.
    profile_path = npz_path.with_name(f"{prefix}_profiles.csv.gz")
    prof.to_csv(profile_path, index=False, float_format="%.6g")
    print(f"wrote {scalar_path.name} ({nseg} rows) + {profile_path.name} ({nseg * nf} rows)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--limit", type=int, default=None, help="only the first N segments (smoke test)"
    )
    ap.add_argument("--param", type=Path, default=PARAM, help="config .param for the run")
    ap.add_argument(
        "--csv",
        type=Path,
        default=CSV,
        help="hybr-trajectory CSV (implicit-phase rows; same columns as stalling_steep)",
    )
    ap.add_argument("--out", type=Path, default=OUT, help="npz path; CSV prefix derived from it")
    args = ap.parse_args()

    rows = list(csv.DictReader(open(args.csv)))
    if args.limit:
        rows = rows[: args.limit]
    f_grid = np.linspace(0.0, 1.0, NF)
    params = init_params(args.param)

    rec = {
        k: []
        for k in (
            "t",
            "hb",
            "hd",
            "cb",
            "cd",
            "g_h",
            "g_c",
            "f_c",
            "conv_c",
            "T0",
            "Lmech_W",
            "Lmech_SN",
            "Lmech_total",
            "R2",
            "v2",
            "c_sound",
            "R1_h",
            "R1_c",
            "R_IF",
            "rShell",
            "vmin_h",
            "vmin_c",
            "cage_ok",
        )
    }
    V_h, N_h, V_c, N_c = [], [], [], []
    nan = np.full(NF, np.nan)

    cage_guess = None  # threaded legacy guess (cage's own previous root)
    print(f"tabulating {len(rows)} segments -> {args.out}")
    t0 = time.perf_counter()
    for i, row in enumerate(rows):
        R2 = set_state(params, row)
        Lm = float(params["Lmech_total"].value)
        hb, hd = float(row["cool_beta"]), float(row["cool_delta"])

        # --- no-cage (hybr): the recorded root, re-solved for its structure ---
        det_h = GB.get_residual_detailed(hb, hd, params)
        ph = det_h.bubble_props
        # T0 is an ODE state variable (dT0/dt = (T0/t).delta) not stored in the CSV;
        # the converged hybr root has T_bubble == T0, so recover the segment's evolved
        # T0 as T_bubble at that root and use it as the shared target for both arms.
        T0 = float(det_h.T_bubble) if ph is not None else float(params["T0"].value)
        params["T0"].value = T0
        if ph is not None:
            vh, nh = profiles(ph, R2, f_grid)
            g_h = g_residual(GB.get_residual_detailed(hb, hd, params, bubble_props=ph), Lm)
            R1_h = ph.R1
            # ionization-front radius (shell solve) on the hybr trajectory; the shell
            # solver reads the current bubble R1/Pb/mass, so set them first (as the
            # implicit runner does before calling shell_structure_pure).
            params["R1"].value = ph.R1
            params["Pb"].value = ph.Pb
            params["bubble_mass"].value = ph.bubble_mass
            try:
                _sh = shell_structure_pure(params)
                R_IF, rShell = float(_sh.R_IF), float(_sh.rShell)
            except Exception:
                R_IF = rShell = np.nan
        else:
            vh, nh, g_h, R1_h, R_IF, rShell = nan, nan, np.nan, np.nan, np.nan, np.nan

        # --- cage (legacy): REAL bounded solve, threaded-guess continuity ---
        if cage_guess is None:  # both arms start from the same (clamped) point
            cage_guess = (float(np.clip(hb, *BOX_B)), float(np.clip(hd, *BOX_D)))
        params["betadelta_solver"].value = "legacy"
        res = GB.solve_betadelta_pure(cage_guess[0], cage_guess[1], params, "grid")
        params["betadelta_solver"].value = "hybr"
        pc = res.bubble_properties
        if pc is not None:
            cb, cd = float(res.beta), float(res.delta)
            cage_guess = (cb, cd)
            vc, nc = profiles(pc, R2, f_grid)
            # g from detailed components (the result's fields are None on the
            # legacy already-converged-input short-circuit path), props reused.
            g_c = g_residual(GB.get_residual_detailed(cb, cd, params, bubble_props=pc), Lm)
            f_c = float(res.total_residual)
            rec["cage_ok"].append(True)
            rec["conv_c"].append(bool(g_c < GB.RESIDUAL_THRESHOLD))
            R1_c, vmin_c = pc.R1, float(np.nanmin(vc))
        else:
            cb = cd = np.nan
            vc, nc, g_c, f_c, R1_c, vmin_c = nan, nan, np.nan, np.nan, np.nan, np.nan
            rec["cage_ok"].append(False)
            rec["conv_c"].append(False)

        for key, val in (
            ("t", float(row["t_now"])),
            ("hb", hb),
            ("hd", hd),
            ("cb", cb),
            ("cd", cd),
            ("g_h", float(g_h)),
            ("g_c", float(g_c)),
            ("f_c", float(f_c)),
            ("T0", T0),
            ("Lmech_W", float(row["Lmech_W"])),
            ("Lmech_SN", float(row["Lmech_SN"])),
            ("Lmech_total", float(row["Lmech_total"])),
            ("R2", R2),
            ("v2", float(row["v2"])),
            ("c_sound", float(row["c_sound"])),
            ("R1_h", float(R1_h)),
            ("R1_c", float(R1_c)),
            ("R_IF", float(R_IF)),
            ("rShell", float(rShell)),
            ("vmin_h", float(np.nanmin(vh))),
            ("vmin_c", float(vmin_c)),
        ):
            rec[key].append(val)
        V_h.append(vh)
        N_h.append(nh)
        V_c.append(vc)
        N_c.append(nc)

        if i % 5 == 0 or i == len(rows) - 1:
            dt = time.perf_counter() - t0
            print(
                f"  {i + 1}/{len(rows)}  t={float(row['t_now']):.3f}  "
                f"hybr=({hb:+.2f},{hd:+.2f}) cage=({cb:+.2f},{cd:+.2f})  "
                f"g_h={g_h:.1e} g_c={float(g_c):.1e}  [{dt / (i + 1):.1f} s/seg]"
            )

    out = {k: np.array(v) for k, v in rec.items()}
    out.update(
        f_grid=f_grid, v_h=np.array(V_h), n_h=np.array(N_h), v_c=np.array(V_c), n_c=np.array(N_c)
    )
    save_outputs(args.out, out)
    print(f"wrote {args.out}  ({len(rows)} segments, {time.perf_counter() - t0:.0f} s)")


if __name__ == "__main__":
    main()
