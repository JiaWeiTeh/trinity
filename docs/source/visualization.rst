.. _sec-visualization:

Visualization Tools
===================

TRINITY's plotting code lives outside the installed ``trinity``
package, under ``paper/figures/``, with two entry points:

- **Published paper figures** are regenerated from the bundled
  ``paper/data/*.npz`` files by a single entry point::

      python paper/make_figures.py            # all published figures
      python paper/make_figures.py teaser     # one figure (prefix match)

  Output lands in ``paper/plots/``. The figure scripts it drives live
  under ``paper/figures/`` (with shared infrastructure in
  ``paper/figures/_lib/``); see *Published Paper Figures* below.
- **Run-directly scripts** — e.g. ``paper_feedback.py`` (below) — also
  live under ``paper/figures/`` but are invoked directly rather than
  through ``make_figures.py``, writing to ``fig/{folder_name}/``.

Each script is a thin wrapper around the :ref:`sec-trinity-reader`
API: it loads one or more simulations, extracts a few time series, and
renders them with a shared Matplotlib style sheet,
``paper/figures/_lib/trinity.mplstyle``. The scripts double as worked
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
(``paper/figures/_lib/cli.py``): they accept a folder of simulations
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
   python paper/figures/paper_feedback.py -F /path/to/outputs/sweep_test

   # Filter by core density
   python paper/figures/paper_feedback.py -F /path/to/outputs --nCore 1e4

   # Custom output directory
   python paper/figures/paper_feedback.py -F /path/to/outputs -o /path/to/figures

   # Single run from an explicit path
   python paper/figures/paper_feedback.py /path/to/dictionary.jsonl


Published Paper Figures
-----------------------

These are driven by ``paper/make_figures.py`` from the bundled
``paper/data/*.npz`` files and rendered into ``paper/plots/``. Each row
maps a short name (usable as a prefix on the command line) to its
script and bundle:

.. list-table::
   :widths: 18 30 52
   :header-rows: 1

   * - Name
     - Script (``paper/figures/``)
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

The remaining ``paper/figures/`` script, ``paper_feedback.py``
(*Force Budget Plots* below), is run directly rather than through
``make_figures.py``.


Force Budget Plots
------------------

paper_feedback.py
~~~~~~~~~~~~~~~~~

*(Lives under* ``paper/figures/``\ *; run directly.)* Stacked-area plot
showing the relative importance of feedback forces as fractions of the
total:

- **Gravity** (black)
- **Ram pressure** (blue) with wind/SN decomposition overlay
- **Photoionised gas** (red)
- **Radiation** (purple): direct and reprocessed radiation pressure
- **PISM** (white): external ISM pressure

Configuration
-------------

Most scripts share a common parser (``paper/figures/_lib/cli.py``),
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
       └── ...
