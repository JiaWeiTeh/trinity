 .. role:: small-caps
    :class: small-caps

Introduction to TRINITY
=======================

This is a guide for users of the TRINITY software package. TRINITY
is distributed under the terms of the GNU General Public License
v. 3.0; a copy of the license is included in the main TRINITY
directory. If you use TRINITY in any published work, please cite
the TRINITY method paper (in preparation) and acknowledge the code
as described in :doc:`publications`.


What Does TRINITY Do?
---------------------

TRINITY is a feedback-driven HII-region evolution code, meaning
that, for a specified giant-molecular-cloud mass, star-formation
efficiency, density profile, and ambient medium, it predicts the
time evolution of the shell radius, velocity, thermal state, and
force budget of the expanding feedback bubble. It also predicts the
phase transitions (energy-driven, transition, momentum-driven) that
the shell passes through, and the stopping fate of the bubble
(stall, dissolution, or escape from the cloud). In this regard,
TRINITY operates much like any other 1D shell-evolution code. The
main difference is that TRINITY treats the interior pressure as a
convex blend of a hot wind-blown bubble and the warm photo-ionised
gas, with a blending weight that is recorded at every integration
step so that the full thermal history of the shell is available to
the user, and it drives the simulation from a fully described state
dictionary that carries unit metadata alongside every recorded
quantity. The remainder of this section briefly describes the
major conceptual pieces of a TRINITY simulation. For a more
detailed description, readers are referred to :ref:`sec-physics`.


The Shell and its Environment
-----------------------------

TRINITY treats the region around a young stellar cluster as a
spherically symmetric two-zone flow. Inside an evacuated cavity, a
hot wind-blown bubble is bounded by an inner termination shock at
:math:`R_1` and by a thin swept-up shell at :math:`R_2`; outside
the shell the ambient cloud retains its initial density profile.
The shell carries essentially all of the swept-up mass, and its
motion is governed by the balance between the thermal pressure of
the hot interior, the ram pressure of the shocked wind, the
pressure of the photo-ionised gas behind the ionisation front, the
direct and dust-reprocessed radiation pressure, and the
gravitational pull of the cluster and the enclosed cloud mass.
Each of these terms is evaluated from first principles at every
time step; none is tabulated in advance.

Two analytic cloud profiles are supported: a single power-law
profile, ``densPL``, specified by a core density, core radius, and
power-law index :math:`\alpha`; and a Bonnor-Ebert profile,
``densBE``, constructed to be in hydrostatic equilibrium against
self-gravity. Both profiles are truncated at the cloud edge, beyond
which the shell expands into a uniform ambient medium of density
:math:`n_{\rm ISM}` (see :ref:`sec-physics`).


Stellar Feedback
----------------

Mechanical luminosity, ionising photon rate, and bolometric
luminosity are drawn from `Starburst99
<https://www.stsci.edu/science/starburst99/>`_ tables at the
metallicity of the stellar population and are resampled onto the
integration grid at the start of each run. Wind and supernova
momentum injection are tracked separately so that their relative
contribution to the ram-pressure term can be decomposed in
post-processing. Cooling of the hot bubble is handled through
collisional-ionisation-equilibrium curves, with an optional non-CIE
correction for the low-temperature regime; the user can disable
either the cooling or the non-CIE correction independently through
the parameter file. See :ref:`sec-parameters` for the full list of
feedback-related keywords.


Phase Transitions
-----------------

The shell passes through three dynamical regimes. In the
*energy-driven* phase, the hot bubble is effectively adiabatic and
drives the shell through its thermal pressure. In the
*momentum-driven* phase, the bubble has cooled enough that the
shell is pushed by the accumulated momentum of the wind and the
supernova ejecta. Between the two, a *transition* phase is
integrated in which the interior driving pressure is modelled as a
convex blend :math:`P_{\rm drive} = (1-w)\,P_b + w\,P_{\rm HII}`,
with :math:`w` set by the ratio of the cooling time to the
advection time. The blending weight is recorded at every step and
can be plotted directly from the output (see
:ref:`sec-visualization`).


Parameter Sweeps
----------------

TRINITY can evolve a single cloud or an entire grid of clouds. A
parameter file that contains one or more list-valued or
``tuple(...)``-valued inputs is interpreted as a sweep and is
dispatched to a parallel process pool automatically. Three sweep
modes are supported: a Cartesian sweep (every combination of the
listed values), a tuple sweep (only the enumerated parameter
combinations), and a hybrid sweep (Cartesian product of a tuple
directive with one or more list values). The same
``python run.py <file>`` invocation drives all three modes; see
:ref:`sec-running` for details.


Obtaining and Running TRINITY
-----------------------------

The TRINITY source is hosted at
https://github.com/JiaWeiTeh/trinity. TRINITY is pure Python; there
is no compilation step. To install, clone the repository and make
its root directory your working directory. TRINITY requires Python
3.9 or newer together with NumPy, SciPy, Astropy, pandas, and
Matplotlib.

Once the dependencies are in place, running a simulation is
extremely simple. Write a parameter file, then from the repository
root do::

    python run.py param/my_run.param

The code will integrate the shell to the stopping criterion and
write its output tree to the directory specified by ``path2output``.
See :ref:`sec-running` for a full description of the parameter
file format, the sweep modes, the command-line options, and the
on-disk output layout.


Organisation of This Manual
---------------------------

The remainder of this manual is organised as follows.
:ref:`sec-running` describes the command-line interface, the
parameter file syntax, the three sweep modes, and the on-disk
output. :ref:`sec-parameters` enumerates every input and output
parameter together with its unit, default, and short description.
:ref:`sec-architecture` documents the internal module layout and
the data flow between the orchestrator, phase solvers, and shared
physics modules. :ref:`sec-physics` sets out the governing
equations and the numerical scheme that integrates them.
:ref:`sec-trinity-reader` documents the Python API for loading and
analysing simulation output. :ref:`sec-visualization` describes the
plotting scripts shipped with TRINITY.


Contents
--------

.. toctree::
   :maxdepth: 2

   license
   running
   parameters
   architecture
   physics
   visualization
   trinity_reader
   publications
