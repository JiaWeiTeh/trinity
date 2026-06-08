.. _sec-visualization:

Visualization Tools
===================

TRINITY's plotting code lives outside the installed ``trinity``
package, split by audience:

- **Published paper figures** are regenerated from the bundled
  ``paper/methods/data/*.npz`` files by a single entry point::

      python paper/make_figures.py            # all published figures
      python paper/make_figures.py teaser     # one figure (prefix match)

  Output lands in ``paper/plots/``. The figure scripts it drives live
  under ``paper/methods/figures/`` (with shared infrastructure in
  ``paper/_lib/``); see *Published Paper Figures* below.
- **Exploratory / personal scripts** — the broader catalogue below —
  live under ``scratch/``. They are not part of the installed package
  and are run directly, writing to ``fig/{folder_name}/``.

Each script is a thin wrapper around the :ref:`sec-trinity-reader`
API: it loads one or more simulations, extracts a few time series, and
renders them with a shared Matplotlib style sheet,
``paper/_lib/trinity.mplstyle``. The scripts double as worked
examples of how to drive the reader API from user code.

.. seealso::

   - :ref:`sec-trinity-reader` — the ``TrinityOutput`` API that every
     plotting script uses to load ``dictionary.jsonl`` data.
   - :ref:`sec-running` — how to produce the simulation outputs these
     scripts consume, including sweep folder layout.
   - :ref:`sec-parameters` — definitions of the parameters (``R2``,
     ``Pb``, ``F_*``, etc.) plotted below.


Common features
---------------

Most plotting scripts share a common parser
(``paper/_lib/cli.py``): they accept a folder of simulations
through ``-F`` and auto-discover the individual runs underneath, so a
single invocation can render either one simulation or a full
(mCloud × SFE) grid. Core density is selected through ``-n``/``--nCore``,
figure output is redirected through ``-o``, and the shared style sheet
keeps figures from different scripts visually consistent. Phase, cloud-
edge, and collapse markers are drawn via ``--show-*`` flags.

Usage examples
--------------

.. code-block:: bash

   # All runs in a folder (auto-discovers the mCloud/SFE grid)
   python paper/methods/figures/paper_feedback.py -F /path/to/outputs/sweep_test

   # Filter by core density
   python paper/methods/figures/paper_feedback.py -F /path/to/outputs --nCore 1e4

   # Custom output directory
   python paper/methods/figures/paper_feedback.py -F /path/to/outputs -o /path/to/figures

   # Single run from an explicit path
   python scratch/paper_radiusEvolution.py /path/to/dictionary.jsonl


Published Paper Figures
-----------------------

These are driven by ``paper/make_figures.py`` from the bundled
``paper/methods/data/*.npz`` files and rendered into ``paper/plots/``. Each row
maps a short name (usable as a prefix on the command line) to its
script and bundle:

.. list-table::
   :widths: 18 30 52
   :header-rows: 1

   * - Name
     - Script (``paper/methods/figures/``)
     - Figure
   * - ``density``
     - ``paper_densityProfile.py``
     - Density-profile ingredients (uniform, :math:`r^{-1}`,
       :math:`r^{-2}`, Bonnor-Ebert) plus the phase timeline.
   * - ``teaser``
     - ``paper_teaser.py``
     - Three-panel teaser: :math:`R_b`/:math:`v_b`, feedback-force
       decomposition, and the :math:`Q_i` photon budget.
   * - ``radiusComparison``
     - ``paper_radiusComparison.py``
     - :math:`R(t)` comparison of TRINITY against WARPFIELD and the
       analytic scaling laws (Weaver, Spitzer, momentum).
   * - ``rcloud_smoothing``
     - ``paper_rcloud_smoothing.py``
     - Cloud-edge density-smoothing schematic with before/after LSODA
       trajectories.

The remaining ``paper/methods/figures/`` script, ``paper_feedback.py``
(*Force Budget Plots* below), and the ``scratch/`` catalogue are run
directly rather than through ``make_figures.py``.


Force Budget Plots
------------------

paper_feedback.py
~~~~~~~~~~~~~~~~~

*(Lives under* ``paper/methods/figures/``\ *; run directly.)* Stacked-area plot
showing the relative importance of feedback forces as fractions of the
total:

- **Gravity** (black)
- **Ram pressure** (blue) with wind/SN decomposition overlay
- **Photoionised gas** (red)
- **Radiation** (purple): direct and reprocessed radiation pressure
- **PISM** (white): external ISM pressure

paper_forceFraction.py
~~~~~~~~~~~~~~~~~~~~~~

Simplified force-fraction plot showing the three mechanically distinct
forces that enter the momentum equation additively (so the fractions
sum to one): combined thermal driving pressure (``F_thermal``),
radiation (``F_rad``), and gravity (``F_grav``).

paper_dominantFeedback.py
~~~~~~~~~~~~~~~~~~~~~~~~~

Grid plot of which feedback mechanism dominates across parameter space
(x: cloud mass, y: SFE). ``F_ram`` competes as a whole first, then
subclassifies to wind (blue) or SN (yellow) if it wins. This script
carries extra flags beyond the shared set (see *Configuration*).

.. code-block:: bash

   python scratch/paper_dominantFeedback.py -F /path/to/outputs --nCore 1e4
   python scratch/paper_dominantFeedback.py -F /path/to/outputs --times 1.0 3.0 5.0
   python scratch/paper_dominantFeedback.py -F /path/to/outputs --movie --dt 0.05


Thermal Regime Plots
--------------------

paper_pressureEvolution.py
~~~~~~~~~~~~~~~~~~~~~~~~~~

Line plot of pressure evolution over time: hot bubble pressure
(``P_b``), ionization-front pressure (``P_IF``), the effective driving
pressure (``P_drive``, selected per phase from :math:`P_b`,
:math:`P_{\rm HII}`, and :math:`P_{\rm ram}`), and optionally the
external pressure.


Acceleration Decomposition
--------------------------

paper_accelerationDecomposition.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Multi-line plot of the acceleration components from the force balance:
gas-pressure (``a_gas``), radiation (``a_rad``), gravity (``a_grav``),
mass-loading (``a_acc``), and the net (``a_net``). Symmetric-log scale
by default; the sign of ``a_net`` indicates accelerating / coasting /
decelerating.

paper_cancellationMetric.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Shows the force-cancellation metric
:math:`\mathcal{C} = |a_{\rm net}| / \sum_i |a_i|`. Near 1, one force
dominates; near 0, forces nearly cancel.


Shell Evolution Plots
---------------------

paper_radiusEvolution.py
~~~~~~~~~~~~~~~~~~~~~~~~

Shell radius :math:`R_2(t)` with phase markers.

paper_expansionVelocity.py
~~~~~~~~~~~~~~~~~~~~~~~~~~

Shell velocity :math:`v_2(t)`.

paper_momentum.py
~~~~~~~~~~~~~~~~~

Cumulative momentum evolution with force contributions.


Configuration
-------------

Most scripts share a common parser (``paper/_lib/cli.py``),
which provides:

.. code-block:: bash

   # Input (use a folder OR a single dictionary.jsonl path)
   data                  Positional path to one dictionary.jsonl (optional)
   -F, --folder PATH     Folder of simulation subfolders to auto-discover

   # Filtering
   -n, --nCore VALUE     Filter by core density (e.g., "1e4")
   --mCloud VALUES       Filter by cloud mass (one or more)
   --sfe VALUES          Filter by star-formation efficiency (one or more)

   # Output / styling
   -o, --output-dir PATH Directory to save figures (default: fig/)
   --palette NAME        Colour palette
   --info                Print discovered runs and exit

   # Marker overlays
   --show-phase          Energy/transition/momentum phase boundaries
   --show-rcloud         Cloud-edge crossing
   --show-collapse       Collapse onset
   --show-noPHII         Overlay the no-PHII companion run
   --show-all-markers    Enable all of the above

``paper_dominantFeedback.py`` carries its own extra flags on top of the
shared set:

.. code-block:: bash

   -t, --times VALUES    Time snapshots in Myr (e.g., 1.0 3.0 5.0)
   --smooth METHOD       Smoothing: 'none' or 'interp'
   --axis-mode MODE      'discrete' or 'continuous'
   --movie               Render an animated GIF over time
   --dt VALUE            Frame spacing for --movie


Output
------

Standalone scripts (run directly) save their figures to the
``fig/{folder_name}/`` directory as PDF files; the ``make_figures.py``
entry point instead collects the published figures in ``paper/plots/``.
Standalone filenames include a parameter tag built from the (mCloud,
SFE, nCore) combination actually plotted, so runs with different
parameters never overwrite each other:

.. code-block:: text

   fig/
   └── sweep_test/
       ├── feedback_M1e7_sfe020_n1e4.pdf            # single run
       ├── feedback_M1e5-1e8_sfe001-080_n1e4.pdf    # grid of runs
       ├── radiusEvolution_M1e7_sfe020_n1e4.pdf
       └── ...
