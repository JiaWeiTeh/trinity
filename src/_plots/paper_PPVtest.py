#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  6 18:21:40 2026

@author: Jia Wei Teh
"""
"""
Thin expanding spherical shell -> PV / PPV-style visualization

Physics
-------
Assume a spherically symmetric, purely radial expansion:
    v(r) = v_exp * r_hat

With the observer's line of sight (LOS) along +z, the LOS velocity is the dot product
with z_hat:
    v_los(x,y,z) = v · z_hat = v_exp * z/r = v_exp * cos(theta),
where r = sqrt(x^2+y^2+z^2) and cos(theta)=z/r.

For an infinitesimally thin shell at radius R (r=R), points on the shell satisfy:
    x^2 + y^2 + z^2 = R^2.

A central PV cut (a "slit" through the center) is commonly approximated by taking
a narrow range in y around 0. In the y=0 limit:
    z = ±sqrt(R^2 - x^2)
so
    v_los(x) = ± v_exp * sqrt(1 - x^2/R^2),
which is the classic "expansion ellipse" in a PV diagram.

What this script does
---------------------
- Creates a spherically symmetric 3D approximation from a 1D shell (radius R, speed v_exp)
  by sampling many points uniformly on the shell surface (optionally with small thickness).
- Computes v_los for each sampled point.
- Builds a PV diagram (x vs v_los) by selecting points in a narrow y-slit and binning
  counts into a 2D histogram (optically thin, uniform emissivity -> intensity ~ counts).
- Overlays the analytic expansion-ellipse boundary.

Run
---
    python thin_shell_ppv.py
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# --- output - save to project root's fig/ directory
FIG_DIR = Path(__file__).parent.parent.parent / "fig"
FIG_DIR.mkdir(parents=True, exist_ok=True)
SAVE_PDF = True

# ---------------- Parameters ----------------
R = 4.0          # shell radius (units)
v_exp = 2.0      # expansion speed (units/s)
t = 1.0          # time (s). Given for context; R is assumed to be the radius at this epoch.

# Sampling / plotting controls
N = 2_000_000           # number of Monte Carlo samples on the shell
shell_thickness = 0.06  # fraction of R used as thickness (0 -> infinitesimal shell)
slit_half_width = 0.15  # PV slit half-width in y (units). Smaller -> thinner ellipse rim.

nx, nv = 500, 400       # histogram resolution (x bins, velocity bins)

# -------------- Sample the shell --------------
rng = np.random.default_rng(2)

# Uniformly sample directions on a sphere: u = cos(theta) uniform in [-1,1], phi uniform [0,2pi]
u = rng.uniform(-1.0, 1.0, size=N)       # u = cos(theta)
phi = rng.uniform(0.0, 2*np.pi, size=N)

# Thin shell: r ~ R (optionally add a small thickness for realism / smoothing)
dR = shell_thickness * R
r = R + rng.uniform(-0.5*dR, 0.5*dR, size=N)

sin_theta = np.sqrt(1.0 - u*u)
x = r * sin_theta * np.cos(phi)
y = r * sin_theta * np.sin(phi)
z = r * u

# LOS velocity mapping (key equation)
v_los = v_exp * (z / r)  # = v_exp * cos(theta)

# -------------- Build PV diagram --------------
# Select a narrow slit around y=0 (a typical observational PV cut)
mask = np.abs(y) < slit_half_width
x_cut = x[mask]
v_cut = v_los[mask]

# Bin counts into a 2D histogram: intensity ~ counts for optically thin, uniform emissivity
x_edges = np.linspace(-R, R, nx + 1)
v_edges = np.linspace(-v_exp, v_exp, nv + 1)
H, _, _ = np.histogram2d(x_cut, v_cut, bins=[x_edges, v_edges])

# -------------- Plot --------------
plt.figure(figsize=(8, 5.5))
extent = [x_edges[0], x_edges[-1], v_edges[0], v_edges[-1]]

# imshow expects array indexed as [row, col] = [v_bin, x_bin], hence transpose
plt.imshow(H.T, origin="lower", aspect="auto", extent=extent)

plt.xlabel("Position x (units)")
plt.ylabel("LOS velocity v_los (units/s)")
plt.title(f"Thin expanding spherical shell: PV slice (|y| < {slit_half_width:.2f})")

# Analytic expansion ellipse boundary for a central cut (y=0)
xx = np.linspace(-R, R, 2000)
vv = v_exp * np.sqrt(np.clip(1.0 - (xx * xx) / (R * R), 0.0, 1.0))
plt.plot(xx,  vv, linewidth=1.5)
plt.plot(xx, -vv, linewidth=1.5)
plt.colorbar()
plt.tight_layout()
if SAVE_PDF:
    plt.savefig(FIG_DIR / 'PPVtest_thinShell.pdf', bbox_inches='tight')
    print(f"Saved: {FIG_DIR / 'PPVtest_thinShell.pdf'}")
plt.show()


#%%


#!/usr/bin/env python3
"""
Uniformly filled expanding sphere -> PVD (position–velocity diagram)

Model
-----
Geometry: a uniformly filled sphere of radius R with constant volume emissivity.
Kinematics: purely radial expansion with constant speed v_exp:
    v(r) = v_exp * r_hat

LOS velocity mapping (observer along +z):
    v_los(x,y,z) = v · z_hat = v_exp * z/r = v_exp * cos(theta),
where r = sqrt(x^2+y^2+z^2).

PVD definition
--------------
A PVD is intensity as a function of one sky position coordinate (here x along a slit)
and LOS velocity v_los, after integrating over the slit width in y and along z.
With Monte Carlo sampling and uniform emissivity, intensity ~ counts per (x, v) bin.

Compared to a thin shell, a filled sphere produces emission throughout the interior
of the classic expansion ellipse.
"""

import numpy as np
import matplotlib.pyplot as plt

# ---------------- Parameters ----------------
R = 4.0
v_exp = 2.0
t = 1.0  # context only

N = 3_000_000           # number of Monte Carlo samples in the volume
slit_half_width = 0.15  # slit half-width in y (units)
nx, nv = 500, 400       # histogram resolution

# -------------- Sample uniformly in VOLUME --------------
rng = np.random.default_rng(3)

# Uniform in volume => p(r) ∝ r^2: r = R * u^(1/3)
u = rng.uniform(0.0, 1.0, size=N)
r = R * u ** (1.0 / 3.0)

# Random direction
mu = rng.uniform(-1.0, 1.0, size=N)  # cos(theta)
phi = rng.uniform(0.0, 2*np.pi, size=N)
sin_theta = np.sqrt(1.0 - mu * mu)

x = r * sin_theta * np.cos(phi)
y = r * sin_theta * np.sin(phi)
z = r * mu

# -------------- LOS velocity mapping --------------
eps = 1e-12
v_los = np.zeros_like(r)
mask = r > eps
v_los[mask] = v_exp * (z[mask] / r[mask])  # = v_exp * cos(theta)

# -------------- Build PVD (x vs v_los) --------------
slit = np.abs(y) < slit_half_width
pos = x[slit]
vel = v_los[slit]

pos_edges = np.linspace(-R, R, nx + 1)
vel_edges = np.linspace(-v_exp, v_exp, nv + 1)

H, _, _ = np.histogram2d(pos, vel, bins=[pos_edges, vel_edges])
H = H.T  # velocity on vertical axis

img = np.log10(1.0 + H)

# -------------- Plot --------------
plt.figure(figsize=(8, 5.5))
extent = [pos_edges[0], pos_edges[-1], vel_edges[0], vel_edges[-1]]
im = plt.imshow(img, origin="lower", aspect="auto", extent=extent)

plt.xlabel("Position along slit (x) [units]")
plt.ylabel("LOS velocity v_los [units/s]")
plt.title(f"PVD of a uniformly filled expanding sphere (R={R}, v={v_exp}), slit |y|<{slit_half_width}")

cbar = plt.colorbar(im)
cbar.set_label("log10(1 + intensity)  [counts]")

# Overlay the maximum |v_los| boundary (central cut):
xx = np.linspace(-R, R, 2000)
vv = v_exp * np.sqrt(np.clip(1.0 - (xx * xx) / (R * R), 0.0, 1.0))
plt.plot(xx,  vv, linewidth=1.5)
plt.plot(xx, -vv, linewidth=1.5)

plt.tight_layout()
if SAVE_PDF:
    plt.savefig(FIG_DIR / 'PPVtest_filledSphere.pdf', bbox_inches='tight')
    print(f"Saved: {FIG_DIR / 'PPVtest_filledSphere.pdf'}")
plt.show()




#%%




