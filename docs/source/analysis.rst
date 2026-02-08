.. highlight:: rest

.. _sec-analysis:

Post-Processing Analysis (``_calc``)
====================================

The ``src/_calc/`` module contains standalone analysis scripts that extract
scaling relations, timescales, and diagnostic quantities from TRINITY
parameter-sweep output.  Each script reads the ``dictionary.jsonl`` files
produced by TRINITY, fits power-law relations to the output, and generates
publication-quality figures plus CSV summary tables.


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


``scaling_phases`` — Phase-transition timescales
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. rubric:: Physics

TRINITY evolves expanding shells through distinct dynamical phases:

1. **Energy-driven phase** — the hot stellar-wind bubble dominates the
   driving pressure.  In the Weaver et al. (1977) adiabatic model, the
   shell radius grows as :math:`R \propto t^{3/5}`.

2. **Transition phase** — radiative cooling of the bubble becomes
   important and driving shifts from thermal pressure to direct momentum
   deposition and warm-ionised-gas pressure.

3. **Momentum-driven phase** — the bubble has cooled; the shell coasts
   under accumulated momentum (:math:`R \propto t^{1/2}`).

The *transition timescales* between these phases encode how quickly
feedback energy is radiated away versus converted to kinetic energy.

.. rubric:: Quantities fitted

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

.. rubric:: Output

- ``scaling_phases_parity.{fmt}`` — parity plots (fit vs TRINITY).
- ``scaling_phases_results.csv`` — per-run timescale data.
- ``scaling_phases_fits.csv`` — fit coefficients and diagnostics.


``collapse_criterion`` — Minimum SFE for dispersal
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. rubric:: Physics

Whether feedback can disrupt a cloud depends on the balance between the
energy/momentum injected by young stars and the gravitational binding
energy of the cloud.  For each pair (:math:`n_c`, :math:`M_{\rm cloud}`)
there exists a minimum SFE :math:`\varepsilon_{\rm min}` below which
the shell re-collapses.

The threshold scales with surface density because :math:`E_{\rm bind}
\propto \Sigma R^2` while feedback luminosity :math:`\propto \varepsilon
M_{\rm cloud}`, yielding:

.. math::

   \varepsilon_{\rm min} \propto \Sigma^\delta

This analysis is the TRINITY analogue of the WARPFIELD headline result
(Rahner et al. 2019, Fig. 5).

.. rubric:: Quantities fitted

.. list-table::
   :widths: 25 75
   :header-rows: 1

   * - Symbol
     - Description
   * - :math:`\varepsilon_{\rm min}(n_c, M)`
     - Minimum SFE for dispersal (3-parameter fit).
   * - :math:`\varepsilon_{\rm min}(\Sigma)`
     - Same, using surface density as the single predictor.

.. rubric:: Output

- ``collapse_criterion_phase_diagram.{fmt}`` — expand/collapse map in (:math:`M`, :math:`\varepsilon`) space.
- ``collapse_criterion_epsmin_vs_sigma.{fmt}`` — :math:`\varepsilon_{\rm min}` vs :math:`\Sigma`.
- ``collapse_criterion_parity.{fmt}`` — parity plot.
- ``collapse_criterion_outcome_fraction.{fmt}`` — fraction collapsed vs SFE.
- ``collapse_criterion_results.csv``, ``collapse_criterion_fits.csv``.


``terminal_momentum`` — Terminal momentum per stellar mass
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. rubric:: Physics

The terminal radial momentum :math:`p_{\rm fin} = M_{\rm shell}\,v_{\rm shell}`
normalised by the stellar mass :math:`M_*` quantifies the net momentum yield
of stellar feedback.  It is the primary quantity used in sub-grid feedback
prescriptions for galaxy-scale simulations.

For a single supernova remnant in a uniform medium, resolved 3D simulations
give :math:`p_{\rm fin}/M_* \approx 1000`–3000 km/s (Kim & Ostriker 2015).
TRINITY's multi-mechanism model (winds + radiation + HII + SNe) predicts
how this yield changes with cloud properties.

The optional ``--decompose`` flag integrates the time-resolved force
contributions to identify which mechanism dominates the momentum budget.

.. rubric:: Quantities fitted

.. list-table::
   :widths: 25 75
   :header-rows: 1

   * - Symbol
     - Description
   * - :math:`p_{\rm fin}/M_*`
     - Terminal specific momentum [km/s] (expanding and collapsing).
   * - Component impulses
     - Time-integrated contributions from each force (with ``--decompose``).

.. rubric:: Output

- ``terminal_momentum_parity.{fmt}`` — parity plot.
- ``terminal_momentum_vs_sfe.{fmt}`` — :math:`p_{\rm fin}/M_*` vs SFE.
- ``terminal_momentum_decomposition.{fmt}`` — force-component stacked bars (if ``--decompose``).
- ``terminal_momentum_results.csv``, ``terminal_momentum_fits.csv``.


``velocity_radius`` — Expansion law and self-similar index
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. rubric:: Physics

The power-law index :math:`\alpha = d\log v / d\log R` characterises
the expansion law.  Classical self-similar solutions give constant
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

The self-similar constant :math:`\eta = v\,t / R` allows observers
to estimate a dynamical age :math:`t_{\rm dyn} = \eta\,R/v` from a
single (v, R) measurement.

.. rubric:: Quantities computed

- :math:`\alpha_{\rm local}(R)` — instantaneous index via finite differences.
- :math:`\alpha_{\rm phase}` — least-squares fit within each phase.
- :math:`\eta(t)` — time-resolved self-similar constant.
- :math:`\eta` at fixed R and fixed t — look-up values for observers.

.. rubric:: Output

- ``velocity_radius_gallery.{fmt}`` — v(R) trajectory gallery.
- ``velocity_radius_alpha.{fmt}`` — :math:`\alpha` vs R.
- ``velocity_radius_eta.{fmt}`` — :math:`\eta` vs t.
- ``velocity_radius_results.csv``.


``dispersal_timescale`` — Cloud dispersal and feedback velocity
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. rubric:: Physics

The **dispersal time** :math:`t_{\rm disp}` is defined as the first time
the shell reaches the cloud radius :math:`R_{\rm cloud}` with positive
velocity.  The **feedback velocity** :math:`v_{\rm fb} = R_{\rm cloud} /
t_{\rm disp}` is directly comparable to the observational estimates of
Chevance et al. (2020) who measured 7–21 km/s across nearby galaxies.

The SFE per free-fall time, :math:`\varepsilon_{\rm ff} = \varepsilon
\times t_{\rm ff} / t_{\rm disp}`, connects the simulation outcome to
turbulence-regulated star-formation theories that predict
:math:`\varepsilon_{\rm ff} \approx 0.003`–0.01 (Krumholz & McKee 2005).

.. rubric:: Quantities fitted

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

.. rubric:: Observational bands

- Chevance et al. (2020): :math:`t_{\rm fb} = 1`–5 Myr, :math:`v_{\rm fb} = 7`–21 km/s.
- Rahner et al. (2017): :math:`t_{\rm collapse} = 2`–:math:`4\,t_{\rm ff}`.

.. rubric:: Output

- ``dispersal_timescale_vs_sfe.{fmt}`` — :math:`t_{\rm disp}` vs SFE.
- ``dispersal_timescale_normalized.{fmt}`` — :math:`t/t_{\rm ff}` vs :math:`\Sigma`.
- ``dispersal_timescale_feedback_velocity.{fmt}`` — :math:`v_{\rm fb}` vs :math:`\Sigma`.
- ``dispersal_timescale_epsilon_ff.{fmt}`` — :math:`\varepsilon_{\rm ff}` histogram.
- ``dispersal_timescale_parity.{fmt}`` — parity plot.
- ``dispersal_timescale_collapse_map.{fmt}`` — colour map of :math:`t_{\rm collapse}/t_{\rm ff}`.
- ``dispersal_timescale_results.csv``, ``dispersal_timescale_fits.csv``.


``energy_retention`` — Bubble energy retention fraction
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. rubric:: Physics

The energy retention fraction :math:`\xi(t) = E_b(t) / \int_0^t L_w\,dt'`
measures how efficiently the mechanical luminosity of winds and SNe
thermalises in the bubble.  In the Weaver adiabatic limit
:math:`\xi \approx 0.77`; real bubbles cool radiatively, reducing
:math:`\xi` to 0.01–0.5 depending on density and luminosity.

The script also tracks the **full energy budget**:

.. math::

   \frac{dE_b}{dt} = L_{\rm mech} - L_{\rm cool} - 4\pi R^2 P_b\,v - L_{\rm leak}

decomposing the cumulative budget into fractional contributions from
thermal retention (:math:`f_\xi`), radiative cooling (:math:`f_{\rm cool}`),
PdV work (:math:`f_{\rm PdV}`), and energy leakage (:math:`f_{\rm leak}`).

The characteristic time :math:`t_{1/2}` — when :math:`\xi` drops below
50% of its peak — is compared to the Mac Low & McCray (1988) analytic
cooling time:

.. math::

   t_{\rm cool} \approx 16\,{\rm Myr}\;\left(\frac{Z}{Z_\odot}\right)^{-35/22}
                       n^{-8/11}\,L_{38}^{3/11}

.. rubric:: Quantities fitted

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

.. rubric:: Output

- ``energy_retention_evolution.{fmt}`` — :math:`\xi(t)` gallery.
- ``energy_retention_budget.{fmt}`` — stacked energy budget.
- ``energy_retention_vs_params.{fmt}`` — :math:`\xi` at characteristic times vs :math:`\Sigma`.
- ``energy_retention_thalf_vs_tcool.{fmt}`` — :math:`t_{1/2}` vs analytic :math:`t_{\rm cool}`.
- ``energy_retention_vs_radius.{fmt}`` — :math:`\xi` vs bubble radius.
- ``energy_retention_parity.{fmt}`` — parity plot.
- ``energy_retention_results.csv``, ``energy_retention_fits.csv``.


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
