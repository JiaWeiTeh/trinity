.. highlight:: rest

.. _sec-analysis:

Post-Processing Analysis (``_calc``)
====================================

The scripts in ``src/_calc/`` operate on completed TRINITY
parameter sweeps rather than on individual runs. Each script reads
the ``dictionary.jsonl`` files under a sweep directory, extracts a
particular diagnostic — a terminal quantity, a lifetime, or a scaling
law — fits the result across the sweep, and writes both a figure and
a CSV summary table to disk.

The module is intentionally kept independent of the main simulation
loop: the scripts can be re-run at any time against any existing
sweep directory, and they do not modify the underlying output. All
input/output goes through the :ref:`sec-trinity-reader` API, so the
scripts also serve as worked examples of how to post-process TRINITY
data in user code.


Quick Start
-----------

Run all analysis scripts on a sweep directory:

.. code-block:: bash

    python src/_calc/run_all.py -F /path/to/sweep_output

Run a single analysis (e.g. terminal momentum):

.. code-block:: bash

    python src/_calc/terminal_momentum.py -F /path/to/sweep_output

List available scripts:

.. code-block:: bash

    python src/_calc/run_all.py --list


Shared Conventions
------------------

All ``_calc/`` scripts follow the same conventions:

**CLI flags:**

.. list-table::
   :widths: 25 75
   :header-rows: 1

   * - Flag
     - Description
   * - ``-F / --folder``
     - Path to the sweep output directory tree (required).
   * - ``--nCore-ref``
     - Reference normalisation for :math:`n_c` [cm\ :sup:`-3`] (default: 1000).
   * - ``--mCloud-ref``
     - Reference normalisation for :math:`M_{\rm cloud}` [:math:`M_\odot`] (default: 10\ :sup:`5`).
   * - ``--sfe-ref``
     - Reference normalisation for :math:`\varepsilon` (default: 0.01).
   * - ``--sigma-clip``
     - Sigma threshold for outlier rejection in OLS fits (default: 3.0).
   * - ``--fmt``
     - Output figure format: ``pdf`` (default), ``png``, ``svg``, etc.

**Output directory:**  All figures and CSV files are written to ``./fig/<folder_name>/``
relative to the project root, where ``<folder_name>`` is the name of the input sweep
directory.

**Fitting method:**  Every power-law fit uses ordinary least-squares (OLS) in
:math:`\log_{10}` space with iterative sigma-clipping.  The design matrix
is built from the log-ratios of each varying parameter to its reference value:

.. math::

   \log_{10} Y = \log_{10} A
               + \alpha \log_{10}(n_c / n_0)
               + \beta  \log_{10}(M / M_0)
               + \gamma \log_{10}(\varepsilon / \varepsilon_0)

Parameters that are constant across the sweep (i.e. only one unique value)
are automatically excluded from the fit.

**Internal units:**  TRINITY uses :math:`M_\odot`, pc, Myr internally.
Conversions to CGS are provided by ``src/_functions/unit_conversions.py``.
Velocity conversion: 1 pc/Myr |approx| 0.978 km/s.

.. |approx| unicode:: U+2248


Scripts
-------

Each entry below summarises the physics the script captures, the
quantities it fits or computes, and the output files it writes.


``scaling_phases`` — Phase-transition timescales
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

TRINITY evolves expanding shells through three dynamical phases — an
energy-driven phase in which a hot stellar-wind bubble drives the shell
(Weaver et al. 1977, :math:`R \propto t^{3/5}`), a transition phase in
which radiative cooling shifts driving from thermal pressure to direct
momentum deposition and warm-ionised-gas pressure, and a momentum-driven
coasting phase (:math:`R \propto t^{1/2}`). The transition timescales
encode how quickly feedback energy is radiated away versus converted to
kinetic energy.

**Quantities fitted:**

.. list-table::
   :widths: 20 80
   :header-rows: 1

   * - Symbol
     - Description
   * - :math:`t_{\rm trans}`
     - Onset of the transition phase [Myr].
   * - :math:`t_{\rm trans,dur}`
     - Duration of the transition phase [Myr].
   * - :math:`t_{\rm mom}`
     - Onset of the momentum phase [Myr].

**Outputs:** ``scaling_phases_parity.{fmt}`` (parity plots),
``scaling_phases_results.csv`` (per-run timescale data),
``scaling_phases_fits.csv`` (fit coefficients and diagnostics).


``collapse_criterion`` — Minimum SFE for dispersal
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For each pair (:math:`n_c`, :math:`M_{\rm cloud}`) there exists a minimum
SFE :math:`\varepsilon_{\rm min}` below which feedback fails to disrupt
the cloud and the shell re-collapses. Because :math:`E_{\rm bind} \propto
\Sigma R^2` while the feedback luminosity scales as
:math:`\varepsilon M_{\rm cloud}`, the threshold tracks surface density:

.. math::

   \varepsilon_{\rm min} \propto \Sigma^\delta

This is the TRINITY analogue of the WARPFIELD headline result
(Rahner et al. 2019, Fig. 5).

**Quantities fitted:**

.. list-table::
   :widths: 25 75
   :header-rows: 1

   * - Symbol
     - Description
   * - :math:`\varepsilon_{\rm min}(n_c, M)`
     - Minimum SFE for dispersal (3-parameter fit).
   * - :math:`\varepsilon_{\rm min}(\Sigma)`
     - Same, using surface density as the single predictor.

**Outputs:** ``collapse_criterion_phase_diagram.{fmt}`` (expand/collapse
map in :math:`(M, \varepsilon)` space),
``collapse_criterion_epsmin_vs_sigma.{fmt}``,
``collapse_criterion_parity.{fmt}``,
``collapse_criterion_outcome_fraction.{fmt}`` (fraction collapsed vs SFE),
``collapse_criterion_results.csv``, ``collapse_criterion_fits.csv``.


``terminal_momentum`` — Terminal momentum per stellar mass
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The terminal radial momentum :math:`p_{\rm fin} = M_{\rm shell}\,v_{\rm shell}`
normalised by stellar mass :math:`M_*` quantifies the net momentum yield
of stellar feedback — the primary quantity used in sub-grid feedback
prescriptions for galaxy-scale simulations. For a single supernova
remnant in a uniform medium, resolved 3-D simulations give
:math:`p_{\rm fin}/M_* \approx 1000`–3000 km/s (Kim & Ostriker 2015);
TRINITY's multi-mechanism model (winds + radiation + HII + SNe) predicts
how the yield changes with cloud properties. With ``--decompose``, the
time-resolved force contributions are integrated to attribute the
momentum budget to individual mechanisms.

**Quantities fitted:**

.. list-table::
   :widths: 25 75
   :header-rows: 1

   * - Symbol
     - Description
   * - :math:`p_{\rm fin}/M_*`
     - Terminal specific momentum [km/s] (expanding and collapsing).
   * - Component impulses
     - Time-integrated contributions from each force (with ``--decompose``).

**Outputs:** ``terminal_momentum_parity.{fmt}``,
``terminal_momentum_vs_sfe.{fmt}``,
``terminal_momentum_decomposition.{fmt}`` (with ``--decompose``),
``terminal_momentum_results.csv``, ``terminal_momentum_fits.csv``.


``velocity_radius`` — Expansion law and self-similar index
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The power-law index :math:`\alpha = d\log v / d\log R` characterises the
expansion law. Classical self-similar solutions give constant
:math:`\alpha`:

.. list-table::
   :widths: 30 10 60
   :header-rows: 1

   * - Model
     - :math:`\alpha`
     - Regime
   * - Weaver wind bubble
     - :math:`-2/3`
     - Energy-driven (:math:`R \propto t^{3/5}`)
   * - Spitzer HII region
     - :math:`-3/2`
     - Pressure-driven (:math:`R \propto t^{4/7}`)
   * - Momentum-driven
     - :math:`-1`
     - Snowplough (:math:`R \propto t^{1/2}`)

The self-similar constant :math:`\eta = v\,t / R` lets observers estimate
a dynamical age :math:`t_{\rm dyn} = \eta\,R/v` from a single
:math:`(v, R)` measurement.

**Quantities computed:** :math:`\alpha_{\rm local}(R)` (instantaneous
index via finite differences); :math:`\alpha_{\rm phase}` (least-squares
fit within each phase); :math:`\eta(t)`; :math:`\eta` at fixed
:math:`R` and fixed :math:`t` (look-up values for observers).

**Outputs:** ``velocity_radius_gallery.{fmt}`` (v(R) trajectory gallery),
``velocity_radius_alpha.{fmt}``, ``velocity_radius_eta.{fmt}``,
``velocity_radius_results.csv``.


``dispersal_timescale`` — Cloud dispersal and feedback velocity
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The **dispersal time** :math:`t_{\rm disp}` is the first time the shell
reaches the cloud radius :math:`R_{\rm cloud}` with positive velocity;
the **feedback velocity** :math:`v_{\rm fb} = R_{\rm cloud} / t_{\rm disp}`
is directly comparable to the observational estimates of Chevance et al.
(2020), who measured 7–21 km/s across nearby galaxies. The SFE per
free-fall time, :math:`\varepsilon_{\rm ff} = \varepsilon \times
t_{\rm ff} / t_{\rm disp}`, connects the simulation outcome to
turbulence-regulated star-formation theories predicting
:math:`\varepsilon_{\rm ff} \approx 0.003`–0.01 (Krumholz & McKee 2005).

**Quantities fitted:**

.. list-table::
   :widths: 30 70
   :header-rows: 1

   * - Symbol
     - Description
   * - :math:`t_{\rm disp}`
     - Dispersal time [Myr] (expanding runs).
   * - :math:`t_{\rm disp}/t_{\rm ff}`
     - Dispersal time normalised by free-fall time.
   * - :math:`t_{\rm collapse}/t_{\rm ff}`
     - Collapse time normalised by free-fall time.
   * - :math:`v_{\rm fb}`
     - Feedback velocity [km/s].
   * - :math:`\varepsilon_{\rm ff}`
     - SFE per free-fall time.

**Observational bands:** Chevance et al. (2020) report
:math:`t_{\rm fb} = 1`–5 Myr and :math:`v_{\rm fb} = 7`–21 km/s;
Rahner et al. (2017) report :math:`t_{\rm collapse} = 2`–:math:`4\,t_{\rm ff}`.

**Outputs:** ``dispersal_timescale_vs_sfe.{fmt}``,
``dispersal_timescale_normalized.{fmt}`` (:math:`t/t_{\rm ff}` vs
:math:`\Sigma`), ``dispersal_timescale_feedback_velocity.{fmt}``,
``dispersal_timescale_epsilon_ff.{fmt}`` (histogram),
``dispersal_timescale_parity.{fmt}``,
``dispersal_timescale_collapse_map.{fmt}``,
``dispersal_timescale_results.csv``, ``dispersal_timescale_fits.csv``.


``energy_retention`` — Bubble energy retention fraction
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The energy retention fraction
:math:`\xi(t) = E_b(t) / \int_0^t L_w\,dt'` measures how efficiently the
mechanical luminosity of winds and supernovae thermalises in the bubble.
In the Weaver adiabatic limit :math:`\xi \approx 0.77`; real bubbles cool
radiatively, reducing :math:`\xi` to 0.01–0.5 depending on density and
luminosity. The script also tracks the full energy budget,

.. math::

   \frac{dE_b}{dt} = L_{\rm mech} - L_{\rm cool} - 4\pi R^2 P_b\,v - L_{\rm leak}

and decomposes the cumulative budget into fractional contributions from
thermal retention (:math:`f_\xi`), radiative cooling
(:math:`f_{\rm cool}`), PdV work (:math:`f_{\rm PdV}`), and energy
leakage (:math:`f_{\rm leak}`). The characteristic time :math:`t_{1/2}`
— when :math:`\xi` falls below 50% of its peak — is compared to the
Mac Low & McCray (1988) analytic cooling time

.. math::

   t_{\rm cool} \approx 16\,{\rm Myr}\;\left(\frac{Z}{Z_\odot}\right)^{-35/22}
                       n^{-8/11}\,L_{38}^{3/11}.

**Quantities fitted:**

.. list-table::
   :widths: 25 75
   :header-rows: 1

   * - Symbol
     - Description
   * - :math:`\xi` at 1 Myr
     - Early-time retention.
   * - :math:`\xi` at 3 Myr
     - Post-SN-onset retention.
   * - :math:`\xi_{\rm trans}`
     - Retention at the transition phase onset.
   * - :math:`\xi_{\rm disp}`
     - Retention at dispersal / peak-R (most relevant for sub-grid).
   * - :math:`t_{1/2}`
     - Cooling dominance timescale [Myr].
   * - :math:`f_{\rm cool}`
     - Fractional cooling loss at dispersal.
   * - :math:`f_{\rm PdV}`
     - Fractional PdV work at dispersal.

**Outputs:** ``energy_retention_evolution.{fmt}`` (:math:`\xi(t)`
gallery), ``energy_retention_budget.{fmt}`` (stacked energy budget),
``energy_retention_vs_params.{fmt}``,
``energy_retention_thalf_vs_tcool.{fmt}``,
``energy_retention_vs_radius.{fmt}``,
``energy_retention_parity.{fmt}``,
``energy_retention_results.csv``, ``energy_retention_fits.csv``.


Batch Runner (``run_all``)
--------------------------

The ``run_all.py`` script dispatches ``-F <folder>`` and shared flags
to every registered ``_calc/`` analysis script.

.. code-block:: bash

    # Run all scripts
    python src/_calc/run_all.py -F /data/sweep

    # Run only two
    python src/_calc/run_all.py -F /data/sweep --only scaling_phases energy_retention

    # Skip one
    python src/_calc/run_all.py -F /data/sweep --skip velocity_radius

    # Dry run (show commands without executing)
    python src/_calc/run_all.py -F /data/sweep --dry-run

    # Forward shared flags
    python src/_calc/run_all.py -F /data/sweep --nCore-ref 1e4 --fmt png

    # List available scripts
    python src/_calc/run_all.py --list


References
----------

.. [Weaver1977] Weaver, R. et al. (1977). "Interstellar bubbles." *ApJ*, 218, 377. `ADS <https://ui.adsabs.harvard.edu/abs/1977ApJ...218..377W>`_

.. [MacLow1988] Mac Low, M.-M. & McCray, R. (1988). "Superbubbles in disk galaxies." *ApJ*, 324, 776. `ADS <https://ui.adsabs.harvard.edu/abs/1988ApJ...324..776M>`_

.. [Rahner2017] Rahner, D. et al. (2017). "WARPFIELD: A semi-analytic model." *MNRAS*, 470, 4453. `ADS <https://ui.adsabs.harvard.edu/abs/2017MNRAS.470.4453R>`_

.. [Rahner2019] Rahner, D. et al. (2019). "WARPFIELD population synthesis." *MNRAS*, 483, 2547. `ADS <https://ui.adsabs.harvard.edu/abs/2019MNRAS.483.2547R>`_

.. [KimOstriker2015] Kim, C.-G. & Ostriker, E. C. (2015). "Momentum injection by SNe." *ApJ*, 802, 99. `ADS <https://ui.adsabs.harvard.edu/abs/2015ApJ...802...99K>`_

.. [Chevance2020] Chevance, M. et al. (2020). "Lifecycle of molecular clouds." *MNRAS*, 493, 2872. `ADS <https://ui.adsabs.harvard.edu/abs/2020MNRAS.493.2872C>`_

.. [KrumholzMcKee2005] Krumholz, M. R. & McKee, C. F. (2005). "A general theory of turbulence-regulated star formation." *ApJ*, 630, 250. `ADS <https://ui.adsabs.harvard.edu/abs/2005ApJ...630..250K>`_


See Also
--------

- :ref:`sec-physics` for the underlying equations of motion.
- :ref:`sec-trinity-reader` for the data loading API.
- :ref:`sec-visualization` for plotting scripts.
