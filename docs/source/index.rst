 .. role:: small-caps
    :class: small-caps

TRINITY
=======

TRINITY is a Python code for modelling the dynamical evolution of HII
regions and wind-blown bubbles around young stellar clusters. It solves
the coupled, one-dimensional equations of motion for a spherical shell
driven by stellar-wind mechanical luminosity, supernova injection,
ionising radiation, and radiation pressure, opposed by self-gravity and
the ram pressure of the ambient medium. Given a molecular cloud mass,
star-formation efficiency, and density profile, TRINITY integrates the
shell through the energy-driven, transition, and momentum-driven phases
until the shell either stalls, dissolves, or escapes the cloud.

The code is designed around a single parameter file, a single entry
point (``run.py``), and a fully described on-disk output: every
quantity recorded in ``dictionary.jsonl`` carries its own units and
human-readable description, so that downstream analysis scripts can
introspect the output without recourse to external metadata. Parameter
sweeps in Cartesian, tuple, and hybrid modes are supported natively
and execute in parallel via a process pool.

TRINITY is suitable both for single-cloud case studies and for
population-scale parameter surveys that span cloud mass, SFE, and
ambient density.


What TRINITY computes
---------------------

For each simulation, TRINITY evolves in time the shell radius
:math:`R_2(t)`, the shell velocity :math:`v_2(t)`, the bubble thermal
energy :math:`E_b(t)`, and a family of diagnostic pressures and
forces. Feedback is drawn from Starburst99 tables that are resampled
onto the integration grid, and cooling is handled through
collisional-ionisation-equilibrium curves with an optional non-CIE
correction. The solver switches between energy-driven and
momentum-driven regimes through a convex blend whose weight is
recorded at every step, so the full thermal history of the shell is
available to the user.

A complete list of recorded quantities is given in :ref:`sec-parameters`.


Obtaining and running TRINITY
-----------------------------

The source code lives at
https://github.com/JiaWeiTeh/trinity. TRINITY requires Python 3.9 or
newer together with NumPy, SciPy, Astropy, pandas, and Matplotlib. No
compilation step is required; the package is run directly from a
checkout of the repository.

A minimal simulation is launched with

.. code-block:: console

    python run.py param/my_run.param

where ``my_run.param`` is a plain-text parameter file. Chapter
:ref:`sec-running` describes the parameter file format, the three
sweep modes, the CLI options, and the on-disk output layout in full.


Citation
--------

If TRINITY contributes to a publication, please cite the papers listed
in :doc:`publications` and acknowledge the code as described in
:doc:`acknowledgements`.


Organisation of this manual
---------------------------

The manual is organised as follows. :ref:`sec-running` covers the
command-line interface, the parameter file format, parallel sweeps,
and the on-disk output. :ref:`sec-parameters` enumerates every input
and output parameter together with its units and default value.
:ref:`sec-architecture` describes the internal module layout, the
simulation phases, and the data flow between them. :ref:`sec-physics`
sets out the governing equations and the numerical scheme that
integrates them. :ref:`sec-trinity-reader` documents the Python API
for loading simulation output. :ref:`sec-visualization` and
:ref:`sec-analysis` describe the plotting and post-processing scripts
that ship with TRINITY.


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
   analysis
   publications
   acknowledgements
