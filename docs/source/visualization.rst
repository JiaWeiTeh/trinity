Visualization Tools
===================

TRINITY includes a suite of paper-quality visualization scripts for analyzing simulation outputs.
All scripts are located in ``src/_plots/`` and follow consistent conventions.

Common Features
---------------

All plotting scripts support:

- **Single-run mode**: Plot a single simulation from command line or config
- **Grid mode**: Generate parameter-space grids (mCloud × SFE)
- **CLI interface**: ``python paper_*.py <run_name>`` or ``python paper_*.py /path/to/data``
- **Consistent styling**: Uses ``trinity.mplstyle`` for publication-quality figures
- **Phase markers**: Shows transition (T) and momentum (M) phase boundaries

Usage Examples
--------------

.. code-block:: bash

   # Single run from command line
   python src/_plots/paper_feedback.py 1e7_sfe020_n1e4

   # From explicit path
   python src/_plots/paper_radiusEvolution.py /path/to/dictionary.jsonl

   # Grid mode (uses config at top of file)
   python src/_plots/paper_feedback.py

Force Budget Plots
------------------

paper_feedback.py
~~~~~~~~~~~~~~~~~

Stacked area plot showing relative importance of feedback forces as fractions of total:

- **Gravity** (black): Gravitational force
- **Ram pressure** (blue): Total ram pressure with wind/SN decomposition overlay

  - Blue forward hatch: Wind-attributed ram pressure
  - Yellow backward hatch: SN-attributed ram pressure

- **Photoionised gas** (red): Ionization pressure
- **Radiation** (purple): Direct and indirect radiation pressure
- **PISM** (white): External ISM pressure

paper_forceFraction.py
~~~~~~~~~~~~~~~~~~~~~~

Simplified force fraction plot showing the three mechanically distinct forces:

- **F_thermal**: Combined thermal driving pressure (hot bubble + warm HII)
- **F_rad**: Radiation pressure
- **F_grav**: Gravitational force

This is physically correct because these forces ARE additive in the momentum equation.

paper_dominantFeedback.py
~~~~~~~~~~~~~~~~~~~~~~~~~

Grid plot showing which feedback mechanism dominates across parameter space:

- X-axis: Cloud mass (mCloud)
- Y-axis: Star formation efficiency (SFE)
- Colors: Dominant force at each point

Force categories:

- **Gravity** (dark blue-gray)
- **Winds** (blue): Ram pressure from stellar winds
- **Supernovae** (yellow): Ram pressure from SN
- **Photoionised gas** (red)
- **Radiation** (purple)

Supports static plots and animated GIF generation with ``--movie`` flag.

Thermal Regime Plots
--------------------

paper_pressureEvolution.py
~~~~~~~~~~~~~~~~~~~~~~~~~~

Line plot showing pressure evolution over time:

- **P_b** (blue solid): Hot bubble pressure
- **P_IF** (red solid): Ionization front pressure
- **P_drive** (black dashed): Effective driving pressure (convex blend)
- **P_ext** (gray dotted): External pressure (optional)

Demonstrates the convex blend model transition:

.. math::

   P_{\rm drive} = (1 - w) P_b + w P_{\rm IF}

paper_thermalRegime.py
~~~~~~~~~~~~~~~~~~~~~~

Shows the blending weight w(t) which indicates thermal regime:

- **w ≈ 0**: Hot bubble dominates (energy-driven regime)
- **w ≈ 1**: Warm ionized gas dominates (HII-driven regime)

Supports both line plot and stacked area visualization modes (``--stacked`` flag).

Acceleration Decomposition
--------------------------

paper_accelerationDecomposition.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Multi-line plot showing acceleration components from force balance:

- **a_gas** (blue): Gas pressure acceleration
- **a_rad** (purple): Radiation pressure acceleration
- **a_grav** (black): Gravitational acceleration (negative = inward)
- **a_acc** (orange): Mass loading acceleration
- **a_net** (gray dashed): Net acceleration = sum of above

Uses symmetric log scale by default. The sign of a_net indicates:

- a_net > 0: Shell accelerating outward
- a_net ≈ 0: Quasi-equilibrium / coasting
- a_net < 0: Shell decelerating (may collapse)

paper_cancellationMetric.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Shows force cancellation metric:

.. math::

   \mathcal{C} = \frac{|a_{\rm net}|}{\sum |a_i|}

Interpretation:

- **C ≈ 1**: One force dominates completely
- **C ≈ 0**: Large cancellation, forces nearly balance
- **C ~ 0.1-0.3**: Typical quasi-equilibrium expansion

Shell Evolution Plots
---------------------

paper_radiusEvolution.py
~~~~~~~~~~~~~~~~~~~~~~~~

Shows shell radius R₂(t) evolution with phase markers.

paper_expansionVelocity.py
~~~~~~~~~~~~~~~~~~~~~~~~~~

Shows shell velocity v₂(t) evolution.

paper_momentum.py
~~~~~~~~~~~~~~~~~

Shows cumulative momentum evolution with force contributions.

Configuration
-------------

Each script has configuration variables at the top of the file:

.. code-block:: python

   # Parameter space
   mCloud_list = ["1e5", "5e5", "1e6", "5e6", "1e7", "5e7", "1e8"]
   ndens_list = ["1e3"]
   sfe_list = ["001", "005", "010", "020", "030", "050", "070", "080"]

   # Base directory for outputs
   BASE_DIR = Path.home() / "unsync/Code/Trinity/outputs/sweep_test_modified"

   # Smoothing
   SMOOTH_WINDOW = 11  # Moving average window (None to disable)

   # Optional single-run mode
   ONLY_M = "1e7"    # Set to None for grid mode
   ONLY_N = "1e4"
   ONLY_SFE = "010"

Output
------

All figures are saved to the ``fig/`` directory as PDF files with descriptive names:

- Single run: ``{script}_{run_name}.pdf``
- Grid: ``{script}_grid_{mass_range}_{sfe_range}_{ndens}.pdf``
