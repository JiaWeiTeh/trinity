Visualization Tools
===================

TRINITY includes a suite of paper-quality visualization scripts for analyzing simulation outputs.
All scripts are located in ``src/_plots/`` and follow consistent conventions.

Common Features
---------------

All plotting scripts support:

- **Folder-based input**: Auto-discover simulations using ``-F /path/to/outputs``
- **Density filtering**: Filter by core density with ``--nCore`` or ``-n``
- **Grid mode**: Generate parameter-space grids (mCloud × SFE)
- **Consistent styling**: Uses ``trinity.mplstyle`` for publication-quality figures
- **Phase markers**: Shows transition (T) and momentum (M) phase boundaries

All scripts use the ``trinity_reader`` module (see :ref:`sec-trinity-reader`) for data loading.

Usage Examples
--------------

.. code-block:: bash

   # Plot all simulations from a folder (auto-discovers mCloud/SFE grid)
   python src/_plots/paper_feedback.py -F /path/to/outputs/sweep_test

   # Filter by core density
   python src/_plots/paper_feedback.py -F /path/to/outputs --nCore 1e4

   # Specify output directory for figures
   python src/_plots/paper_feedback.py -F /path/to/outputs -o /path/to/figures

   # Single run from explicit path
   python src/_plots/paper_radiusEvolution.py /path/to/dictionary.jsonl

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

F_ram competes as a whole first, then subclassifies to wind (blue) or SN (yellow) if it wins.
For simulations that have ended (t > t_max), the final dominant feedback is persisted.

.. code-block:: bash

   # Plot from folder
   python src/_plots/paper_dominantFeedback.py -F /path/to/outputs --nCore 1e4

   # Custom time snapshots
   python src/_plots/paper_dominantFeedback.py -F /path/to/outputs --times 1.0 3.0 5.0

   # Generate animated GIF
   python src/_plots/paper_dominantFeedback.py -F /path/to/outputs --movie --dt 0.05

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

Scripts are configured via command-line arguments. Common options:

.. code-block:: bash

   # Required: folder containing simulations
   -F, --folder PATH     Path to folder with simulation subfolders

   # Filtering
   -n, --nCore VALUE     Filter by core density (e.g., "1e4")

   # Output
   -o, --output-dir PATH Directory to save figures (default: fig/)

   # Time selection (for grid plots)
   -t, --times VALUES    Time snapshots in Myr (e.g., 1.0 3.0 5.0)

   # Display options
   --smooth METHOD       Smoothing: 'none' or 'interp'
   --axis-mode MODE      'discrete' or 'continuous'

Output
------

All figures are saved to the ``fig/{folder_name}/`` directory as PDF files:

.. code-block:: text

   fig/
   └── sweep_test/
       ├── feedback_n1e4.pdf
       ├── dominantFeedback_n1e4_continuous_interp.pdf
       ├── radiusEvolution_1e7_sfe020_n1e4.pdf
       └── ...
