#!/usr/bin/env python3
"""Plot the last non-stale hot-bubble pressure profile."""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))

from trinity._functions import unit_conversions as cvt  # noqa: E402


RUN_DIR = (
    ROOT
    / "outputs"
    / "rosette_cf_survey_updated_0p77"
    / "1e5_sfe001_n1e3_PL0_yesPHII"
)
STYLE = ROOT / "paper" / "_lib" / "trinity.mplstyle"
PREFIX = "1e5_sfe001_n1e3_PL0_yesPHII"


def profile_signature(snapshot):
    keys = (
        "bubble_T_arr_r_arr",
        "log_bubble_T_arr",
        "bubble_n_arr_r_arr",
        "log_bubble_n_arr",
    )
    return json.dumps([snapshot.get(key) for key in keys], separators=(",", ":"))


def load_last_changing_profile():
    rows = [json.loads(line) for line in (RUN_DIR / "dictionary.jsonl").open()]
    last_change = 0
    previous = None
    for i, snapshot in enumerate(rows):
        current = profile_signature(snapshot)
        if current != previous:
            last_change = i
            previous = current
    two_myr_index = min(range(len(rows)), key=lambda i: abs(rows[i]["t_now"] - 2.0))
    return last_change, rows[last_change], two_myr_index, rows[two_myr_index]


idx, snap, two_myr_idx, two_myr_snap = load_last_changing_profile()
metadata = json.loads((RUN_DIR / "metadata.json").read_text())

r_pc = np.asarray(snap["bubble_T_arr_r_arr"], dtype=float)
temperature_k = 10.0 ** np.asarray(snap["log_bubble_T_arr"], dtype=float)
n_h_internal = 10.0 ** np.asarray(snap["log_bubble_n_arr"], dtype=float)

# bubble_n_arr stores hydrogen-nuclei density in internal units [pc^-3].
n_h_cgs = n_h_internal * cvt.ndens_au2cgs

# Convert n_H to total particle density for the hot ionised bubble. This matches
# Trinity's bubble-structure pressure relation in bubble_luminosity.py.
mu_factor = metadata["mu_convert"] / metadata["mu_ion"]
p_dyn_cm2 = mu_factor * n_h_cgs * cvt.K_B_CGS * temperature_k
p_over_kb = p_dyn_cm2 / cvt.K_B_CGS
p_hii_2myr_dyn_cm2 = two_myr_snap["P_HII"] * cvt.Pb_au2cgs
p_hii_2myr_over_kb = p_hii_2myr_dyn_cm2 / cvt.K_B_CGS

order = np.argsort(r_pc)
r_pc = r_pc[order]
n_h_cgs = n_h_cgs[order]
temperature_k = temperature_k[order]
p_dyn_cm2 = p_dyn_cm2[order]
p_over_kb = p_over_kb[order]

plt.style.use(str(STYLE))
plt.rcParams["text.usetex"] = True

fig, ax = plt.subplots(figsize=(4.8, 3.35))
ax.plot(r_pc, p_over_kb, color="#D55E00")
ax.axhline(
    p_hii_2myr_over_kb,
    color="#0072B2",
    ls="--",
    lw=1.2,
    label=rf"$P_{{\rm HII}}(t={two_myr_snap['t_now']:.2f}\,\mathrm{{Myr}})$",
)
ax.set_yscale("log")
ax.set_xlabel(r"$r_{\rm bubble}$ [pc]")
ax.set_ylabel(r"$P_{\rm hot}/k_{\rm B}$ [K cm$^{-3}$]")
ax.tick_params(which="both", direction="in", top=True, right=True)
if np.ptp(p_over_kb) / np.mean(p_over_kb) < 1e-8:
    values = np.array([np.mean(p_over_kb), p_hii_2myr_over_kb])
    ax.set_ylim(values.min() / 2.0, values.max() * 2.0)
ax.legend(loc="best", fontsize=8)
ax.text(
    0.04,
    0.08,
    rf"$t={snap['t_now']:.4f}\,$Myr",
    transform=ax.transAxes,
    ha="left",
    va="bottom",
)

age_token = f"{snap['t_now']:.4f}".replace(".", "p")
out = HERE / f"{PREFIX}_bubble_PHII_vs_r_bubble_t{age_token}myr.pdf"
fig.savefig(out)
plt.close(fig)

fig, (ax_n, ax_t) = plt.subplots(2, 1, figsize=(4.8, 5.0), sharex=True)
ax_n.plot(r_pc, n_h_cgs, color="#0072B2")
ax_n.set_yscale("log")
ax_n.set_ylabel(r"$n_{\rm H}$ [cm$^{-3}$]")
ax_n.tick_params(which="both", direction="in", top=True, right=True)
ax_n.text(
    0.04,
    0.08,
    rf"$t={snap['t_now']:.4f}\,$Myr",
    transform=ax_n.transAxes,
    ha="left",
    va="bottom",
)

ax_t.plot(r_pc, temperature_k, color="#D55E00")
ax_t.set_yscale("log")
ax_t.set_xlabel(r"$r_{\rm bubble}$ [pc]")
ax_t.set_ylabel(r"$T$ [K]")
ax_t.tick_params(which="both", direction="in", top=True, right=True)

profile_out = HERE / f"{PREFIX}_bubble_n_T_vs_r_bubble_t{age_token}myr.pdf"
fig.savefig(profile_out)
plt.close(fig)

print(f"snapshot_index={idx}")
print(f"age_myr={snap['t_now']:.16g}")
print(f"phase={snap['current_phase']}")
print(f"radius_pc=[{r_pc.min():.6g}, {r_pc.max():.6g}]")
print(f"P_dyn_cm2=[{p_dyn_cm2.min():.6e}, {p_dyn_cm2.max():.6e}]")
print(f"P_over_kB_K_cm-3=[{p_over_kb.min():.6e}, {p_over_kb.max():.6e}]")
print(f"P_HII_2Myr_snapshot_index={two_myr_idx}")
print(f"P_HII_2Myr_age_myr={two_myr_snap['t_now']:.16g}")
print(f"P_HII_2Myr_dyn_cm2={p_hii_2myr_dyn_cm2:.6e}")
print(f"P_HII_2Myr_over_kB_K_cm^-3={p_hii_2myr_over_kb:.6e}")
print(out)
print(profile_out)
