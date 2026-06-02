.. highlight:: rest

.. _sec-running:

Running TRINITY
===============

Running a simulation
--------------------

A TRINITY run is fully specified by one plain-text parameter file, and
there is a single command — from the repository root::

    python run.py param/simple_cluster.param

The path may be absolute or relative to the repository root.
``run.py`` scans the file and dispatches automatically: if the file
contains list (``[...]``) or ``tuple(...)`` syntax it runs a parameter
sweep across a parallel worker pool, otherwise it runs a single
simulation. There is no separate command or flag for sweeps. On an HPC
cluster you can instead generate a SLURM job array with ``--emit-jobs``
(see *Running on a cluster (SLURM)* below).

Output is written to the directory named by the ``path2output``
parameter; the sentinel ``def_dir`` (the default) means the current
working directory. See *Outputs* below for the file layout.


Parameter-file formats
----------------------

A parameter file lists one ``keyword    value`` entry per line (see
:ref:`sec-parameters` for the full keyword reference). The *value*
syntax alone decides whether the file is a single run or a sweep:

.. code-block:: text
   :caption: param/sweep_hybrid_example.param

    # Plain key/value — fixed across every run
    dens_profile    densPL
    nISM            0.1
    path2output     outputs/demo

    # tuple(...) — only these explicit (mCloud, sfe) pairs are run
    tuple(mCloud, sfe)    [1e5, 0.01] [1e7, 0.10]

    # [list] — swept Cartesian-style across each tuple pair
    nCore    [1e3, 1e4]

The file above mixes all three value forms. How many simulations a
file generates depends only on which forms it uses:

.. list-table::
   :widths: 32 16 52
   :header-rows: 1

   * - Value syntax
     - Mode
     - Runs generated
   * - no ``[ ]``, no ``tuple()``
     - single
     - 1
   * - ``key [a, b, c]``
     - Cartesian
     - every combination (e.g. ``mCloud`` × ``sfe`` = 3 × 2 = 6)
   * - ``tuple(x, y) [..] [..]``
     - tuple
     - only the listed points, no expansion
   * - tuple **and** list together
     - hybrid
     - tuple points × list combinations

The hybrid example therefore runs 2 tuple pairs × 2 ``nCore`` values =
4 simulations. Single-purpose worked examples ship as
``param/sweep_example.param`` (Cartesian),
``param/sweep_tuple_example.param`` (tuple), and
``param/sweep_hybrid_example.param`` (hybrid).


Command-line flags
------------------

All flags are optional. Most take effect only in sweep mode; for a
single run, ``--dry-run`` prints the resolved file and exits without
running, while ``--workers`` and ``--yes`` are ignored:

.. list-table::
   :widths: 25 75
   :header-rows: 1

   * - Flag
     - Description
   * - ``--dry-run``, ``-n``
     - Preview all combinations (with any GMC warnings) without running.
   * - ``--workers N``, ``-w``
     - Parallel workers for the in-process sweep pool — or the array
       concurrency cap with ``--emit-jobs``. Default inside a SLURM job:
       the full allocation (``SLURM_CPUS_PER_TASK``, else
       ``SLURM_CPUS_ON_NODE`` / CPU affinity); on a laptop:
       ``max(1, CPU count // 2 - 1)``. Must be ``>= 1``; refused if it
       exceeds the cores available to this process.
   * - ``--yes``, ``-y``
     - Skip the interactive confirmation prompt.
   * - ``--verbose``, ``-v``
     - DEBUG-level logs and the full base-parameter list.
   * - ``--emit-jobs DIR``
     - Generate a SLURM job-array bundle in ``DIR`` (one task per
       combination) instead of running locally; requires a sweep file.
       Mutually exclusive with ``--collect-report`` (see below).
   * - ``--collect-report DIR``
     - Aggregate a finished ``--emit-jobs`` bundle into
       ``sweep_report.txt`` / ``.json``; needs no parameter file.

Before launching, ``run.py`` runs a GMC-parameter plausibility check
(cloud mass vs. core/ISM density, cloud radius, …) on every
combination; invalid ones are listed up front so you can abort rather
than waste compute. Press ``Ctrl+C`` — or send ``SIGTERM``, e.g. from
SLURM ``scancel`` — to cancel cleanly: in-flight workers are stopped
and a report of completed / failed / cancelled runs is written to the
output directory.


Running on a cluster (SLURM)
----------------------------

On a laptop or a single multi-core node, a sweep runs across an
in-process worker pool sized by ``--workers``. To scale across nodes on
an HPC cluster (e.g. bwForCluster Helix / bwUniCluster), generate a
SLURM **job array** instead — one array task per combination, so the
scheduler packs them across nodes and restarts failures independently::

    python run.py param/sweep_example.param --emit-jobs jobs/
    # edit jobs/submit_sweep.sbatch: --account, --partition, --time, --mem
    sbatch jobs/submit_sweep.sbatch
    python run.py --collect-report jobs/      # after the array finishes

Running the in-process pool on a *login* node is discouraged; ``run.py``
prints a warning when SLURM is detected without an active job.

``--emit-jobs DIR`` writes a self-contained, submittable bundle:

.. code-block:: text

    jobs/
    ├── params/<run_name>.param   # one per combination, absolute path2output
    ├── runs.tsv                  # param_path <TAB> output_dir; line N = array task N
    ├── manifest.json             # index: names, params, output dirs
    ├── submit_sweep.sbatch       # #SBATCH --array=1-N[%K]; one sim per task
    └── logs/                     # %A_%a.out per task

Each array task runs ``python run.py <combo>.param`` with one CPU and
math-library threads pinned to one (``OMP_NUM_THREADS=1`` …,
``MPLBACKEND=Agg``); parallelism comes from running many tasks, not from
threading one. Passing ``--workers K`` at emit time caps concurrency as
``--array=1-N%K``.

When the array finishes, ``--collect-report DIR`` reads each task's
``.exit_code`` / ``.duration`` sentinels and writes the same
``sweep_report.txt`` / ``.json`` as a local sweep, then prints a ready
``sbatch --array=<failed ids> jobs/submit_sweep.sbatch`` to rerun only
the failures.

Outputs land in the same ``path2output/<run_name>/`` layout as a local
sweep (see *Outputs* below). Bundled inputs (SPS, cooling tables,
``lib/default/``) resolve relative to the package, so the clone location
does not matter; only ``path2output`` follows the launch directory — set
it to an absolute path on a work/scratch filesystem for cluster runs.


Outputs
-------

File layout
^^^^^^^^^^^

A single run writes three files into ``path2output``:

.. code-block:: text

    path2output/
    ├── dictionary.jsonl            # simulation state, one JSON object per snapshot
    ├── {model_name}_summary.txt    # human-readable parameter summary
    └── trinity.log                 # log file (written when log_file = True)

A sweep writes those same three files into one subdirectory per
combination, plus two top-level reports:

.. code-block:: text

    outputs/my_sweep/
    ├── 1e5_sfe001_n1e3/
    │   ├── 1e5_sfe001_n1e3.param   # full resolved params for this run
    │   ├── dictionary.jsonl
    │   ├── 1e5_sfe001_n1e3_summary.txt
    │   └── trinity.log
    ├── 1e5_sfe001_n1e4/
    │   └── ...
    ├── sweep_report.txt            # human-readable sweep summary
    └── sweep_report.json           # machine-readable sweep summary

Auto-generated run names
^^^^^^^^^^^^^^^^^^^^^^^^

Each sweep combination is named automatically::

    {mCloud}_sfe{sfe*100:03d}_n{nCore}[_density-profile][_PHII][_other-swept-keys]

The optional suffixes appear only when the relevant parameter is set
explicitly in the sweep file (not when left at its ``default.param``
value):

- ``_PL{alpha}`` for ``dens_profile = densPL`` (e.g. ``_PL0``,
  ``_PL-2``), or ``_BE{Omega}`` for ``densBE`` (e.g. ``_BE14``).
- ``_yesPHII`` / ``_noPHII`` when ``include_PHII`` is set — useful when
  sweeping the flag to compare runs with and without HII pressure.
- Any *other* swept parameter without a curated slot above gets a
  generic ``_{key}{value}`` suffix so distinct combinations never
  collapse onto the same folder. snake_case keys become camelCase and
  decimal points in floats become ``p`` (minus signs are kept, as in
  ``_PL-2``). Examples: sweeping ``ZCloud = [0.5, 1.0]`` yields
  ``_ZCloud0p5`` / ``_ZCloud1p0``; ``coll_counter = [True, False]``
  yields ``_collCounterTrue`` / ``_collCounterFalse``. Multiple generic
  suffixes are emitted in sorted-key order for stability.

  Folder-name safety rules applied to generic values:

  - **Hard-rejected** with an immediate ``ValueError`` (no sweep runs):
    values containing ``/``, ``\``, ``..``, or any control character.
    This means **filepath-typed parameters cannot be swept** — set them
    once in your base param file. The check protects against
    silently nesting directories or escaping the sweep root.
  - **Sanitised** to ``-``: any character outside ``[A-Za-z0-9.+-]``
    (spaces, brackets, shell wildcards, unicode, ``=``, ``:`` …). The
    sweep still runs but with a safe folder name.
  - **Length-capped** at 200 characters for the full run name; the
    sweep aborts with a clear error if you cross it (reserve room for
    ``_modified`` / ``_summary.txt`` siblings within the 255-byte
    filesystem cap).

For example, ``1e7_sfe010_n1e4_noPHII`` is ``mCloud=1e7, sfe=0.10,
nCore=1e4`` with ``include_PHII = False``.

The folder name is only a unique human-readable handle: every run also
writes its full resolved parameter set to a per-run ``.param`` file
(plus the sweep-wide ``sweep_report.json``), so a run with *no* suffix
for some key still has that key recorded — it just took the
``default.param`` value. Master plot scripts that compare across
sweeps should read parameters from those sidecars rather than parse
them out of the folder name.

    2026-01-08 15:30:00 | INFO     | trinity.main | === TRINITY Simulation Starting ===
    2026-01-08 15:30:00 | INFO     | trinity.main | Model: test_simulation
    2026-01-08 15:30:01 | INFO     | trinity.sps.read_sps | SPS data processed: 201 time points, t_max=100.00 Myr
    2026-01-08 15:30:03 | INFO     | trinity.phase1_energy | Entering energy-driven phase
    2026-01-08 15:30:15 | WARNING  | trinity.cooling | Temperature below minimum, clamping to 1e4 K
    2026-01-08 15:30:45 | INFO     | trinity.phase1_energy | Energy phase complete: 150 timesteps
    2026-01-08 15:35:00 | INFO     | trinity.main | === Simulation Finished ===


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
   * - Feedback (SPS)
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

.. code-block:: text

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

For analysis, load snapshots through :ref:`sec-trinity-reader`, which
wraps ``dictionary.jsonl`` with a ``TrinityOutput`` container and
exposes time-series extraction, snapshot interpolation, phase and
time-range filtering, pandas conversion, and batch helpers for sweep
outputs. The in-memory ``DescribedDict`` and the buffer→flush pipeline
that produce the file are documented under *Snapshot Persistence* in
:ref:`sec-architecture`.


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

    2026-01-08 15:30:00 | INFO     | trinity.main | === TRINITY Simulation Starting ===
    2026-01-08 15:30:00 | INFO     | trinity.main | Model: test_simulation
    2026-01-08 15:30:01 | INFO     | trinity.sps.read_sps | SPS data processed: 201 time points
    2026-01-08 15:30:03 | INFO     | trinity.phase1_energy | Entering energy-driven phase
    2026-01-08 15:30:15 | WARNING  | trinity.cooling | Temperature below minimum, clamping to 1e4 K
    2026-01-08 15:30:45 | INFO     | trinity.phase1_energy | Energy phase complete: 150 timesteps
    2026-01-08 15:35:00 | INFO     | trinity.main | === Simulation Finished ===


Troubleshooting
---------------

Most parameter errors are typos against the schema; the authoritative
list of valid keywords and defaults is the ParamSpec registry
(``trinity/_input/registry.py``), from which ``trinity/_input/default.param``
is generated, mirrored in :ref:`sec-parameters`. For issues and
feature requests, see https://github.com/JiaWeiTeh/trinity/issues.
