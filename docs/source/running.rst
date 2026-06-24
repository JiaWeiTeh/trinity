.. include:: _deprecation.txt

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
parameter; the default sentinel ``def_dir`` resolves to
``outputs/<model_name>/`` under the current working directory. See
*Outputs* below for the file layout.


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
single run, ``--dry-run`` prints the raw parameter file and exits
without running, while ``--workers`` and ``--yes`` are ignored:

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

A single run writes these files into ``path2output``:

.. code-block:: text

    path2output/
    ├── dictionary.jsonl            # simulation state, one JSON object per snapshot
    ├── metadata.json               # run constants + termination + final-state blocks
    └── trinity.log                 # log file (written when log_file = True)

A sweep writes those same files into one subdirectory per combination,
adds a fully-resolved ``.param`` sidecar to each, and writes two
top-level reports:

.. code-block:: text

    outputs/my_sweep/
    ├── 1e5_sfe001_n1e3/
    │   ├── 1e5_sfe001_n1e3.param   # full resolved params for this run
    │   ├── dictionary.jsonl
    │   ├── metadata.json
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
    sibling filenames within the 255-byte filesystem cap).

For example, ``1e7_sfe010_n1e4_noPHII`` is ``mCloud=1e7, sfe=0.10,
nCore=1e4`` with ``include_PHII = False``.

The folder name is only a unique human-readable handle: every sweep run
also writes its full resolved parameter set to a per-run ``.param``
file (plus the sweep-wide ``sweep_report.json``), so a run with *no*
suffix for some key still has that key recorded — it just took the
``default.param`` value. Master plot scripts that compare across sweeps
should read parameters from those sidecars rather than parse them out of
the folder name.


Output data model
-----------------

dictionary.jsonl
^^^^^^^^^^^^^^^^

Each simulation writes its full state to ``dictionary.jsonl`` as a
stream of newline-delimited JSON objects, one per snapshot. Writes are
append-only and crash-safe (the run flushes buffered snapshots on a
clean exit, ``Ctrl+C``, or ``SIGTERM``), so the file remains readable
after a crash — the last line may be partial but every prior line is a
complete snapshot. Snapshots are saved *before* each ODE step, so all
values in one snapshot share a single ``t_now``.

Snapshot keys group into a handful of categories:

.. list-table::
   :widths: 30 70
   :header-rows: 1

   * - Category
     - Example keys
   * - Administrative
     - ``model_name``, ``current_phase``, ``SimulationEndReason``
   * - Cloud setup
     - ``mCloud``, ``mCluster``, ``rCloud``, ``nEdge``
   * - Dynamical state
     - ``t_now``, ``R2``, ``v2``, ``Eb``, ``T0``, ``R1``, ``Pb``
   * - Feedback (SPS)
     - ``Lmech_W``, ``Lmech_SN``, ``Qi``, ``Lbol``, ``pdot_total``
   * - Pressures
     - ``P_drive``, ``P_HII``, ``P_ram``
   * - Forces
     - ``F_grav``, ``F_ram``, ``F_ram_wind``, ``F_ram_SN``,
       ``F_HII``, ``F_rad``
   * - Bubble / shell profiles
     - ``log_bubble_T_arr`` + ``bubble_T_arr_r_arr``,
       ``bubble_v_arr`` + ``bubble_v_arr_r_arr``,
       ``log_shell_n_arr`` + ``shell_r_arr``

A handful of long 1-D profile arrays are downsampled before
serialisation to keep snapshots manageable. Each simplified array is
paired with its own abscissa (``*_r_arr``) and, where the values span
many decades, stored in :math:`\log_{10}` space (``log_*``). The
target point budget is set by ``simplify_npoints`` (see
:ref:`sec-parameters`). To recover a profile, linearly interpolate the
paired abscissa against the (possibly log-space) values.

The recommended way to read the file is the reader API in
:ref:`sec-trinity-reader`, which hides the JSONL layout, the per-key
unit metadata, and the legacy ``.json`` format behind a small set of
classes.

metadata.json
^^^^^^^^^^^^^

Run-constant parameters and end-of-run summaries live in a sibling
``metadata.json`` (current schema version 4) rather than being repeated
in every snapshot. The reader rehydrates the run constants into each
snapshot on load, so consumers never have to read this file directly —
but it is small and human-inspectable. Its top-level blocks are:

.. list-table::
   :widths: 24 76
   :header-rows: 1

   * - Block
     - Contents
   * - run constants
     - Every input parameter / set-once derived value that does not
       change after phase 0 (``mCloud``, ``sfe``, ``dens_profile``,
       physical constants, …). Membership is derived from the ParamSpec
       registry (see :ref:`sec-parameters`), so it always matches the
       schema.
   * - ``termination``
     - ``{exit_code, outcome, detail, timestamp, model_name}`` — how the
       run ended.
   * - ``final_state``
     - Every scalar / bool / string on the state dict at run end, in
       internal units (pc, Myr, pc⁻³). Long arrays are excluded — their
       final values are the last line of ``dictionary.jsonl``.
   * - ``termination_debug``
     - Last-two-snapshot diff, a NaN/Inf inventory, and physics sanity
       checks, for post-mortem debugging.

All writes go through an atomic helper, so an interrupted write can
never leave a corrupt file.

show_run
^^^^^^^^

For a quick human-readable view of a finished run — run context,
termination reason, and final state — without writing any plotting
code::

    python -m trinity._output.show_run path2output/

It reads ``metadata.json`` and pretty-prints a curated subset; pass
``--json`` for the full dump or ``--quiet`` for a scriptable exit code
(useful in batch loops over a sweep tree).


Logging
-------

The parameter reference in :ref:`sec-parameters` lists the four
logging parameters (``log_level``, ``log_console``, ``log_file``,
``log_colors``) and their defaults. This section covers the conceptual
ladder of log levels.

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
     - Development; debugging specific issues.
   * - ``INFO``
     - Phase transitions, major events (bubble burst, cloud edge
       reached), initialisation and completion markers.
     - Normal simulation runs (default).
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

With ``log_level = INFO``, console output looks like:

.. code-block:: text

    2026-01-08 15:30:00 | INFO     | trinity.main | === TRINITY Simulation Starting ===
    2026-01-08 15:30:00 | INFO     | trinity.main | Model: test_simulation
    2026-01-08 15:30:01 | INFO     | trinity.sps.read_sps | SPS data processed
    2026-01-08 15:30:03 | INFO     | trinity.phase1_energy | Entering energy-driven phase
    2026-01-08 15:30:45 | INFO     | trinity.main | === TRINITY Simulation Complete ===


Troubleshooting
---------------

Most parameter errors are typos against the schema; the authoritative
list of valid keywords and defaults is the ParamSpec registry
(``trinity/_input/registry.py``), from which ``trinity/_input/default.param``
is generated and mirrored in :ref:`sec-parameters`. For issues and
feature requests, see https://github.com/JiaWeiTeh/trinity/issues.
