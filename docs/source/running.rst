.. highlight:: rest

.. _sec-running:

Running TRINITY
===============

Basic Runs
----------

Once TRINITY is installed, running a simulation is straightforward.
A simulation is fully specified by a single plain-text parameter
file, and the same entry point, ``run.py``, drives both single
runs and parameter sweeps. The only invocation a user normally
needs is, from the repository root::

    python run.py param/my_run.param

where ``my_run.param`` is a parameter file formatted as described
in :ref:`sec-parameters`. ``run.py`` inspects the file, decides
whether it describes one simulation or a grid of simulations, and
dispatches to either a single run or a parallel worker pool. No
separate script is needed for sweeps.

A minimal parameter file contains only three entries; everything
else falls back to the defaults listed in :ref:`sec-parameters`::

    model_name    my_first_run
    mCloud        1e6
    sfe           0.01

With this file in ``param/my_first_run.param``, run::

    python run.py param/my_first_run.param

and TRINITY will integrate the shell to the stopping criterion and
write the output tree described in *Output Data Model* below.


Paths and Output Directory
--------------------------

The parameter file passed to ``run.py`` may be given as an absolute
path or a path relative to the root of the repository; both of the
following are valid::

    python run.py param/example.param
    python run.py /home/user/my_params/custom.param

The destination of the simulation output is controlled by the
``path2output`` parameter. TRINITY creates the directory if it does
not already exist and populates it with three files::

    path2output/
    ├── dictionary.jsonl            # simulation state, one JSON object per snapshot
    ├── {model_name}_summary.txt    # human-readable parameter summary
    └── trinity.log                 # log file (written when log_file = True)

If ``path2output`` is set to the sentinel value ``def_dir`` (the
default), output is written into the current working directory.


Parameter Sweep Runs
--------------------

A parameter file that varies one or more inputs across a list of
values is interpreted as a sweep and executed in parallel through a
process pool. The detection is lexical: ``run.py`` treats the file as
a sweep whenever it encounters a multi-element list value such as
``mCloud [1e5, 1e6, 1e7]`` or a ``tuple(...)`` directive. No separate
script or flag is required, and no change to the command line is
needed; the same ``python run.py <file>`` invocation dispatches
either a single run or a full sweep.

Three sweep modes are supported. A *Cartesian* sweep uses list
syntax and generates every combination of the listed values. A
*tuple* sweep uses the ``tuple(name_1, name_2, ...)`` directive to
declare a set of explicit parameter combinations, so that only the
named points in parameter space are executed. A *hybrid* sweep
combines a tuple directive with one or more list sweeps and runs the
Cartesian product of the tuple combinations and the list values.

Cartesian Sweep Syntax
^^^^^^^^^^^^^^^^^^^^^^

Use list notation ``[val1, val2, ...]`` for parameters you want to vary. All
combinations will be generated:

.. code-block:: text
   :caption: param/sweep_example.param

    # Sweep parameters - generate all combinations
    mCloud    [1e5, 1e7, 1e8]
    sfe       [0.01, 0.10]
    nCore     [1e2, 1e3]

    # Fixed parameters - constant across all runs
    dens_profile    densPL
    densPL_alpha    0
    path2output     outputs/my_sweep

This generates a Cartesian product: 3 × 2 × 2 = 12 total simulations.

Tuple Sweep Syntax
^^^^^^^^^^^^^^^^^^

Use the ``tuple(...)`` directive when you want to run only **specific parameter
combinations**, rather than every Cartesian combination. Each ``[...]`` block
after the tuple defines one run:

.. code-block:: text
   :caption: param/sweep_tuple_example.param

    # Syntax: tuple(param1, param2, ...)  [val1, val2] [val1, val2] ...
    # Each [...] defines one combination to run.
    tuple(mCloud, sfe, nCore)   [1e5, 0.01, 1e2] [1e6, 0.05, 1e3] [1e7, 0.10, 1e4]

    # Fixed parameters
    dens_profile    densPL
    densPL_alpha    0
    nISM            0.1
    path2output     outputs/sweep_tuple_test

This runs exactly 3 simulations — one per tuple, with no Cartesian expansion.
Tuples are ideal for targeted scans along a physically meaningful curve
(e.g. fixed surface density) instead of a dense grid.

Hybrid Sweep Syntax
^^^^^^^^^^^^^^^^^^^

Tuple and list syntax can be combined in the same file. Each tuple combination
is crossed (Cartesian product) with the list sweeps:

.. code-block:: text
   :caption: param/sweep_hybrid_example.param

    # Explicit (mCloud, sfe) pairs
    tuple(mCloud, sfe)    [1e5, 0.01] [1e7, 0.10]

    # nCore is swept across each tuple combination
    nCore    [1e3, 1e4]

    dens_profile    densPL
    densPL_alpha    0
    path2output     outputs/sweep_hybrid_test

This yields 2 tuples × 2 ``nCore`` values = 4 runs:

- ``mCloud=1e5, sfe=0.01, nCore=1e3``
- ``mCloud=1e5, sfe=0.01, nCore=1e4``
- ``mCloud=1e7, sfe=0.10, nCore=1e3``
- ``mCloud=1e7, sfe=0.10, nCore=1e4``

Running Sweeps
^^^^^^^^^^^^^^

Sweep mode is auto-detected from the parameter file content — run the same way
as a single simulation:

.. code-block:: console

    # Preview combinations without running (dry run)
    python run.py param/sweep.param --dry-run

    # Run with 4 parallel workers
    python run.py param/sweep.param --workers 4

    # Run with automatic worker detection (see note below)
    python run.py param/sweep.param

    # Skip the interactive confirmation prompt
    python run.py param/sweep.param --yes

    # Enable verbose diagnostics (show all base params + debug logs)
    python run.py param/sweep.param --verbose

.. note::

   The default worker count is ``max(1, CPU count // 2 - 1)`` — conservative
   to keep laptops responsive while a sweep runs in the background. On HPC
   or workstations, pass ``--workers N`` to use more parallelism.

**Command-line options:**

.. list-table::
   :widths: 25 75
   :header-rows: 1

   * - Option
     - Description
   * - ``--dry-run``, ``-n``
     - Preview all parameter combinations without executing.
   * - ``--workers N``, ``-w``
     - Number of parallel worker processes (default: auto-detect).
   * - ``--yes``, ``-y``
     - Skip the interactive confirmation prompt.
   * - ``--verbose``, ``-v``
     - Enable verbose output (DEBUG-level logs, full base-param list).

Press ``Ctrl+C`` at any time to cancel the sweep cleanly; in-flight workers
are terminated and a report of completed/failed/cancelled runs is written to
the output directory. ``SIGTERM`` (e.g. from SLURM ``scancel`` or PBS walltime)
is handled the same way.

Pre-flight Validation
^^^^^^^^^^^^^^^^^^^^^

Before launching, ``run.py`` performs a GMC-parameter plausibility check on
every combination (cloud mass vs. core density vs. ISM density, cloud radius,
etc.). Invalid combinations are flagged up front — in ``--dry-run`` mode they
are listed with the specific errors, and in live mode they are listed before
the confirmation prompt so you can abort rather than waste compute.

Auto-Generated Run Names
^^^^^^^^^^^^^^^^^^^^^^^^

Each combination automatically receives a descriptive name following this convention:

.. code-block:: text

    {mCloud}_sfe{sfe*100:03d}_n{nCore}[_density-profile][_PHII]

Optional suffixes are appended only when the relevant parameter is explicitly
set in the sweep file (they stay off when the parameter is left to its
default in ``default.param``):

- Density profile: ``_PL{alpha}`` for ``dens_profile = densPL``
  (e.g. ``_PL0``, ``_PL-2``), or ``_BE{Omega}`` for ``dens_profile = densBE``
  (e.g. ``_BE14``).
- HII pressure: ``_yesPHII`` when ``include_PHII = True``, ``_noPHII`` when
  ``include_PHII = False``. Useful when sweeping the flag to compare runs
  with and without HII-region pressure.

**Examples:**

- ``1e5_sfe001_n1e2`` for ``mCloud=1e5, sfe=0.01, nCore=1e2``
- ``1e7_sfe010_n1e3`` for ``mCloud=1e7, sfe=0.10, nCore=1e3``
- ``1e7_sfe010_n1e4_noPHII`` for the same run with ``include_PHII = False``
- ``1e5_sfe001_n1e4_PL0_yesPHII`` for a power-law profile with HII on

Output files are organized into subdirectories of ``path2output``:

.. code-block:: text

    outputs/my_sweep/
    ├── 1e5_sfe001_n1e2/
    │   ├── dictionary.jsonl                    # Simulation output data
    │   ├── 1e5_sfe001_n1e2_summary.txt         # Parameter summary
    │   └── trinity.log                         # Per-run log file
    ├── 1e5_sfe001_n1e3/
    │   └── ...
    ├── sweep_report.txt                        # Human-readable sweep summary
    └── sweep_report.json                       # Machine-readable sweep summary


Logging
-------

The parameter reference in :ref:`sec-parameters` lists the four
logging parameters (``log_level``, ``log_console``, ``log_file``,
``log_colors``) and their defaults. This section covers the
conceptual ladder of log levels and an example of the output.

Log Levels
^^^^^^^^^^

Each level includes itself and all more severe levels:
``CRITICAL > ERROR > WARNING > INFO > DEBUG``. Setting
``log_level = INFO`` emits ``INFO``, ``WARNING``, ``ERROR``, and
``CRITICAL`` messages.

.. list-table::
   :widths: 15 45 40
   :header-rows: 1

   * - Level
     - Typical messages
     - When to use
   * - ``DEBUG``
     - Variable values, loop iterations, intermediate calculations,
       function entry/exit.
     - Development; debugging specific issues (default).
   * - ``INFO``
     - Phase transitions, major events (bubble burst, cloud edge
       reached), initialisation and completion markers.
     - Normal simulation runs.
   * - ``WARNING``
     - Values clamped to limits, fallback behaviour, unusual but
       non-critical conditions.
     - Production runs where only potential problems matter.
   * - ``ERROR``
     - Calculation failures, recoverable errors.
     - Silent runs where only actual errors matter.
   * - ``CRITICAL``
     - Unrecoverable failures, fatal errors.
     - When only simulation-stopping errors should print.


Example Output
^^^^^^^^^^^^^^

With ``log_level = INFO``:

.. code-block:: text

    2026-01-08 15:30:00 | INFO     | src.main | === TRINITY Simulation Starting ===
    2026-01-08 15:30:00 | INFO     | src.main | Model: test_simulation
    2026-01-08 15:30:01 | INFO     | src.sb99.read_SB99 | SB99 data loaded: 201 time points
    2026-01-08 15:30:03 | INFO     | src.phase1_energy | Entering energy-driven phase
    2026-01-08 15:30:15 | WARNING  | src.cooling | Temperature below minimum, clamping to 1e4 K
    2026-01-08 15:30:45 | INFO     | src.phase1_energy | Energy phase complete: 150 timesteps
    2026-01-08 15:35:00 | INFO     | src.main | === Simulation Finished ===


Output Data Model
-----------------

Each simulation writes its full state to ``dictionary.jsonl`` as a
stream of newline-delimited JSON objects, one per snapshot. Writes
are append-only, so the file remains readable after a crash — the
last line may be partial but every prior line is a complete snapshot.

Snapshot keys group into a handful of categories:

.. list-table::
   :widths: 30 70
   :header-rows: 1

   * - Category
     - Example keys
   * - Administrative
     - ``path2output``, ``model_name``, ``current_phase``,
       ``SimulationEndReason``
   * - Cloud setup
     - ``mCloud``, ``sfe``, ``mCluster``, ``rCloud``,
       ``initial_cloud_r_arr``, ``initial_cloud_n_arr``,
       ``initial_cloud_m_arr``
   * - Dynamical state
     - ``t_now``, ``R2``, ``v2``, ``Eb``, ``T0``, ``R1``, ``Pb``
   * - Feedback (SB99)
     - ``Lmech_W``, ``Lmech_SN``, ``Qi``, ``Lbol``, ``pdot``
   * - Forces
     - ``F_grav``, ``F_ram``, ``F_ram_wind``, ``F_ram_SN``,
       ``F_ion_out``, ``F_HII_St``, ``F_rad``
   * - Bubble profile
     - ``log_bubble_T_arr`` + ``bubble_T_arr_r_arr``,
       ``log_bubble_n_arr`` + ``bubble_n_arr_r_arr``,
       ``bubble_v_arr`` + ``bubble_v_arr_r_arr``
   * - Shell profile
     - ``log_shell_n_arr`` + ``shell_r_arr``,
       ``shell_grav_force_m`` + ``shell_grav_r``

A single snapshot row looks like:

.. code-block:: json

   {
     "snap_id": 42,
     "t_now": 1.523e-01,
     "current_phase": "energy",
     "R2": 2.48, "v2": 15.7, "Eb": 9.21e+06, "T0": 7.4e+06,
     "R1": 0.31, "Pb": 3.1e+04,
     "Lmech_W": 1.22e+11, "Lmech_SN": 0.0, "Qi": 4.5e+50, "Lbol": 1.1e+40,
     "F_grav": 9.3e+02, "F_ram": 1.6e+03, "F_rad": 7.2e+02,
     "log_shell_n_arr": [3.1, 3.2, ...], "shell_r_arr":  [2.48, 2.49, ...]
   }

For analysis, load snapshots through :ref:`sec-trinity-reader`,
which wraps ``dictionary.jsonl`` with a ``TrinityOutput`` container
and exposes time-series extraction, snapshot interpolation, phase
and time-range filtering, pandas conversion, and batch helpers for
sweep outputs.

Implementation notes — how the in-memory ``DescribedDict`` carries
state, how the buffer→flush pipeline writes to disk, and how signal
handlers preserve buffered data on early exit — are documented under
*Snapshot Persistence* in :ref:`sec-architecture`.


Troubleshooting
---------------

Most parameter errors are typos against the schema; the authoritative
list of valid keywords and defaults is ``src/_input/default.param``,
mirrored in :ref:`sec-parameters`. For issues and feature requests,
see https://github.com/JiaWeiTeh/trinity/issues.
