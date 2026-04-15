.. highlight:: rest

.. _sec-running:

Running TRINITY
===============

A TRINITY simulation is fully specified by a single plain-text
parameter file and is launched through a single entry point,
``run.py``. The entry point auto-detects whether the parameter file
describes one simulation or a parameter sweep, and dispatches to a
serial or a parallel worker pool accordingly. This chapter describes
the parameter file syntax, the three sweep modes (Cartesian, tuple,
and hybrid), the command-line options, and the on-disk layout of the
output produced by each run.

The only Python invocation a user normally needs is

.. code-block:: console

    python run.py param/my_run.param

from the root of the repository. The remainder of the chapter
unpacks what this command does.


A minimal run
-------------

A parameter file is a sequence of ``name value`` lines. The following
three parameters are sufficient to launch a simulation; all other
inputs fall back to the defaults listed in :ref:`sec-parameters`.

.. code-block:: text

    model_name    my_first_run
    mCloud        1e6
    sfe           0.01

Running ``python run.py param/my_run.param`` then integrates the
shell to the stopping criterion and writes the output tree described
in *Output Data Model* below.


Single Simulation Runs
----------------------

The parameter file passed to ``run.py`` may be given as either an
absolute path or a path relative to the root of the repository:

.. code-block:: console

    python run.py param/example.param
    python run.py /home/user/my_params/custom.param

The destination of the simulation output is controlled by the
``path2output`` parameter. TRINITY creates the directory if it does
not already exist and populates it with three files:

.. code-block:: text

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

    {mCloud}_sfe{sfe*100:03d}_n{nCore}

**Examples:**

- ``1e5_sfe001_n1e2`` for ``mCloud=1e5, sfe=0.01, nCore=1e2``
- ``1e7_sfe010_n1e3`` for ``mCloud=1e7, sfe=0.10, nCore=1e3``

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


Logging Configuration
---------------------

TRINITY provides a flexible logging system to help monitor simulation progress and debug issues.

Logging Parameters
^^^^^^^^^^^^^^^^^^

Configure logging in your parameter file:

.. code-block:: text

    log_level     INFO      # DEBUG, INFO, WARNING, ERROR, CRITICAL
    log_console   True      # Print to terminal
    log_file      True      # Write to .log file
    log_colors    True      # Color-coded terminal output

.. list-table::
   :widths: 20 15 65
   :header-rows: 1

   * - Parameter
     - Default
     - Description
   * - ``log_level``
     - ``INFO``
     - Controls how much detail you see. Think of it as a volume knob for logging.
   * - ``log_console``
     - ``True``
     - Print log messages to terminal during simulation.
   * - ``log_file``
     - ``True``
     - Save log messages to ``{path2output}/trinity.log``.
   * - ``log_colors``
     - ``True``
     - Color-code terminal output by severity level.


Log Levels
^^^^^^^^^^

Setting a log level shows **that level and everything more severe**:

.. raw:: html

   <style>
   .log-debug { color: #00CED1; font-weight: bold; }
   .log-info { color: #32CD32; font-weight: bold; }
   .log-warning { color: #FFA500; font-weight: bold; }
   .log-error { color: #FF4444; font-weight: bold; }
   .log-critical { color: #FF00FF; font-weight: bold; }
   </style>

.. raw:: html

   <table style="width:100%; border-collapse: collapse; margin: 1em 0;">
   <tr style="background: #f0f0f0;">
     <th style="padding: 8px; text-align: left; border: 1px solid #ddd;">Level</th>
     <th style="padding: 8px; text-align: left; border: 1px solid #ddd;">Shows</th>
     <th style="padding: 8px; text-align: left; border: 1px solid #ddd;">Use Case</th>
   </tr>
   <tr>
     <td style="padding: 8px; border: 1px solid #ddd;"><span class="log-debug">DEBUG</span></td>
     <td style="padding: 8px; border: 1px solid #ddd;">All messages</td>
     <td style="padding: 8px; border: 1px solid #ddd;">Development, debugging specific issues</td>
   </tr>
   <tr>
     <td style="padding: 8px; border: 1px solid #ddd;"><span class="log-info">INFO</span> ⭐</td>
     <td style="padding: 8px; border: 1px solid #ddd;">INFO + WARNING + ERROR + CRITICAL</td>
     <td style="padding: 8px; border: 1px solid #ddd;"><strong>Recommended</strong> - Normal simulation runs</td>
   </tr>
   <tr>
     <td style="padding: 8px; border: 1px solid #ddd;"><span class="log-warning">WARNING</span></td>
     <td style="padding: 8px; border: 1px solid #ddd;">WARNING + ERROR + CRITICAL</td>
     <td style="padding: 8px; border: 1px solid #ddd;">Production runs, only see potential problems</td>
   </tr>
   <tr>
     <td style="padding: 8px; border: 1px solid #ddd;"><span class="log-error">ERROR</span></td>
     <td style="padding: 8px; border: 1px solid #ddd;">ERROR + CRITICAL</td>
     <td style="padding: 8px; border: 1px solid #ddd;">Silent runs, only see actual errors</td>
   </tr>
   <tr>
     <td style="padding: 8px; border: 1px solid #ddd;"><span class="log-critical">CRITICAL</span></td>
     <td style="padding: 8px; border: 1px solid #ddd;">CRITICAL only</td>
     <td style="padding: 8px; border: 1px solid #ddd;">Only simulation-stopping errors</td>
   </tr>
   </table>


What Each Level Shows
"""""""""""""""""""""

.. raw:: html

   <p><span class="log-debug">DEBUG</span> - Variable values, loop iterations, intermediate calculations, function entry/exit</p>
   <p><span class="log-info">INFO</span> - Phase transitions, major events (bubble bursts, cloud edge reached), initialization, completion</p>
   <p><span class="log-warning">WARNING</span> - Values clamped to limits, fallback behavior, unusual but non-critical conditions</p>
   <p><span class="log-error">ERROR</span> - Calculation failures, unexpected conditions, recoverable errors</p>
   <p><span class="log-critical">CRITICAL</span> - Unrecoverable errors, fatal failures, simulation crashes</p>


Common Configurations
^^^^^^^^^^^^^^^^^^^^^

**Development/Debugging** (maximum detail):

.. code-block:: text

    log_level    DEBUG
    log_console  True
    log_file     False
    log_colors   True

**Normal Run** (recommended):

.. code-block:: text

    log_level    INFO
    log_console  True
    log_file     True
    log_colors   True

**Production/Batch Run** (minimal output):

.. code-block:: text

    log_level    WARNING
    log_console  False
    log_file     True
    log_colors   False


Example Log Output
^^^^^^^^^^^^^^^^^^

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

TRINITY writes simulation state to **JSONL** (JSON Lines) — one JSON object per
line, one line per snapshot. The format is append-only (O(1) flushes), streams
without loading into memory, and stays readable after a crash up to the last
complete line.

This section describes the in-memory ``DescribedDict`` that mirrors the file,
the keys contained in each snapshot, the on-disk layout, the save/flush
workflow, and how to reload snapshots from Python. For higher-level analysis,
use :ref:`trinity_reader <sec-trinity-reader>`.

Dictionary Structure
^^^^^^^^^^^^^^^^^^^^

Internally, TRINITY carries simulation state in a single ``DescribedDict``
(defined in ``src/_input/dictionary.py``). Each key maps to a
``DescribedItem`` object that wraps the raw value together with lightweight
metadata (a human-readable description and original units).

**In-memory layout:**

.. code-block:: python

    from src._input.dictionary import DescribedDict, DescribedItem

    params = DescribedDict()

    params["R2"]     = DescribedItem(0.0, info="Outer shell radius",      ori_units="pc")
    params["v2"]     = DescribedItem(0.0, info="Shell expansion velocity", ori_units="pc/Myr")
    params["Eb"]     = DescribedItem(0.0, info="Bubble thermal energy",    ori_units="Msun*pc**2/Myr**2")
    params["t_now"]  = DescribedItem(0.0, info="Current simulation time",  ori_units="Myr")

    # Access the raw value
    r = params["R2"].value           # -> float
    t = params["t_now"].value        # -> float

    # DescribedItem supports arithmetic/formatting directly
    area = 4 * 3.14159 * params["R2"] ** 2     # float result
    print(f"t = {params['t_now']:.3e} Myr")    # works via __format__

Each ``DescribedItem`` exposes three attributes:

.. list-table::
   :widths: 25 75
   :header-rows: 1

   * - Attribute
     - Meaning
   * - ``value``
     - The stored scalar, list, or numpy array.
   * - ``info``
     - Short human-readable description (seen in ``{model_name}_summary.txt``).
   * - ``ori_units``
     - Original-unit label (e.g. ``"pc"``, ``"Msun"``, ``"1/cm**3"``).

Additionally, a per-item ``exclude_from_snapshot`` flag marks keys that are
*not* persisted to disk — used for large auxiliary objects such as SB99
interpolation tables that can be rebuilt on load.

**What's in each snapshot:**

Snapshots are grouped into a handful of conceptual categories:

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

**On-disk form (one line of ``dictionary.jsonl``):**

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

Only the ``.value`` of each ``DescribedItem`` is written — ``info`` and
``ori_units`` live alongside the code and are reattached automatically when
you load a snapshot back in.

**Snapshot workflow (save → flush → disk):**

Snapshots are captured through a two-stage *buffer → flush* pipeline. Disk
writes stay cheap (append-only, O(1) per flush) and a crash can lose at most
``snapshot_interval`` steps of progress:

.. code-block:: text

    ┌──────────────────────────────────────────────────────────────────────┐
    │                       SIMULATION MAIN LOOP                           │
    │                                                                      │
    │   for each ODE step:                                                 │
    │       integrate physics ──► update params["R2"], ["v2"], ...         │
    │       params.save_snapshot()   ───────────────┐                      │
    │                                               │                      │
    │   at phase boundary / end of simulation:      │                      │
    │       params.flush()          ───────────────┐│                      │
    │                                              ││                      │
    └──────────────────────────────────────────────┼┼──────────────────────┘
                                                   ││
                                                   ▼▼
    ┌──────────────────────────────────────────────────────────────────────┐
    │                       DescribedDict internals                        │
    │                                                                      │
    │   save_snapshot():                                                   │
    │     1. Duplicate guard (skip if t_now & R2 unchanged)                │
    │     2. Serialise non-excluded keys → JSON-ready dict                 │
    │     3. Stage into self.previous_snapshot[str(save_count)]            │
    │     4. If save_count % snapshot_interval == 0 → call flush()         │
    │                                                                      │
    │   flush():                                                           │
    │     1. First flush of a fresh run: delete old dictionary.jsonl       │
    │     2. Append each pending snapshot as one JSON line                 │
    │     3. Clear self.previous_snapshot, bump flush_count                │
    │                                                                      │
    └──────────────────────────────────┬───────────────────────────────────┘
                                       │ append-only writes
                                       ▼
    ┌──────────────────────────────────────────────────────────────────────┐
    │                  {path2output}/dictionary.jsonl                      │
    │                                                                      │
    │   {"snap_id": 0, "t_now": 0.000, "R2": 0.01, ...}                    │
    │   {"snap_id": 1, "t_now": 0.003, "R2": 0.04, ...}                    │
    │   {"snap_id": 2, "t_now": 0.008, "R2": 0.09, ...}                    │
    │   ...                                                                │
    └──────────────────────────────────────────────────────────────────────┘

Stages in detail:

1. **Mutate the dict.** Physics modules update ``params["R2"].value``,
   ``params["Eb"].value``, etc. in place.
2. **Stage a snapshot.** ``params.save_snapshot()`` copies the current state
   (excluding any key marked ``exclude_from_snapshot=True``) into the
   in-memory buffer ``params.previous_snapshot``. A duplicate guard compares
   ``t_now`` + ``R2`` against the last saved entry and silently drops
   re-runs of the same step.
3. **Flush in batches.** Every ``snapshot_interval`` calls (default **10**),
   ``save_snapshot`` triggers ``flush()`` automatically. You can also call
   ``params.flush()`` manually at phase boundaries or after a critical event.
4. **Append to disk.** ``flush()`` opens ``dictionary.jsonl`` in append mode
   and writes one JSON line per pending snapshot, using ``NpEncoder`` to
   serialise numpy scalars and arrays. The first flush of a fresh run
   overwrites any existing file; subsequent flushes only append.
5. **Crash-safe handlers.** On construction, ``DescribedDict`` registers an
   ``atexit`` hook plus ``SIGINT``/``SIGTERM`` handlers. If the process
   exits — cleanly, via ``Ctrl+C``, or via ``kill`` / SLURM ``scancel`` — any
   buffered snapshots are flushed first and a termination debug report is
   written via ``src/_output/simulation_end.py``. ``SIGKILL`` (``kill -9``)
   and ``os._exit()`` bypass these hooks and can still lose the pending
   buffer; everything already on disk is always safe.

Because writes are append-only, the file is readable even after a crash —
the last line may be partial (one incomplete JSON object) but every prior
line is a complete snapshot.

You rarely call these APIs by hand; they are invoked by ``src/main.py`` and
the phase modules. The public-facing reader is ``trinity_reader``
(see :ref:`sec-trinity-reader`).

**Reloading a snapshot:**

.. code-block:: python

    from src._input.dictionary import DescribedDict

    # Load every snapshot into a dict keyed by id
    snapshots = DescribedDict.load_snapshots("/path/to/outputs/my_run")

    # Load one specific snapshot straight into a DescribedDict
    params = DescribedDict.load_snapshot("/path/to/outputs/my_run", snap_id=42)
    r_arr  = params["initial_cloud_r_arr"].value     # numpy array
    t_now  = params["t_now"].value                   # float

    # Convenience helper for the last snapshot
    params = DescribedDict.load_latest_snapshot("/path/to/outputs/my_run")

For most analysis work, prefer the higher-level
:ref:`trinity_reader <sec-trinity-reader>` API, which exposes the same data
as numpy arrays and pandas DataFrames.

Reading Output Data
^^^^^^^^^^^^^^^^^^^

The ``DescribedDict.load_snapshot`` helpers above give direct access to the
raw state — useful when you want the exact Python objects the simulation
worked with. For most analysis work, prefer the higher-level
``trinity_reader`` module, which layers a ``TrinityOutput`` container on top
of the same JSONL files and exposes:

- time-series extraction as numpy arrays (``output.get('R2')``),
- time-indexed snapshots with interpolation (``output.get_at_time(1.0)``),
- phase and time-range filtering (``output.filter(phase='energy')``),
- pandas conversion (``output.to_dataframe()``),
- batch utilities for sweep outputs (``find_all_simulations``,
  ``organize_simulations_for_grid``).

See :ref:`sec-trinity-reader` for the full API, plotting examples, and
details on the profile-array simplification applied to long 1-D arrays.


Troubleshooting
---------------

Common Issues
^^^^^^^^^^^^^

**"Parameter not found in default.param"**
    Your parameter file contains a typo or uses an outdated parameter name.
    Check :ref:`sec-parameters` for valid parameter names.

**"Only solar metallicity supported"**
    Currently, TRINITY only supports ``ZCloud = 1`` (solar metallicity).

**"Invalid density profile"**
    The ``dens_profile`` parameter must be either ``densPL`` or ``densBE``.

**Output directory not created**
    Ensure the parent directory exists. TRINITY creates the final directory but not parent paths.

Getting Help
^^^^^^^^^^^^

For issues and feature requests, visit:
https://github.com/JiaWeiTeh/trinity/issues


See Also
--------

- :ref:`sec-parameters` — complete reference of input parameters, units,
  defaults, and the ``# UNIT:`` annotation system used in ``default.param``.
- :ref:`sec-trinity-reader` — high-level ``TrinityOutput`` API for reading
  ``dictionary.jsonl`` files into numpy / pandas.
- :ref:`sec-visualization` — plotting scripts that consume
  sweep output directories.
- :ref:`sec-analysis` — post-processing analysis scripts (``src/_calc/``)
  that fit scaling relations to sweep results.
- :ref:`sec-architecture` — internal module layout and how ``run.py``
  drives ``main.start_expansion`` through the phase modules.
