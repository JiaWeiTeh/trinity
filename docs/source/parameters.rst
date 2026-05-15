.. highlight:: rest

.. _sec-parameters:

Parameter Specifications
========================

Every TRINITY simulation is driven by a plain-text parameter file.
This chapter describes how such files are formatted and enumerates
every keyword that TRINITY recognises, grouped by physical role:
cloud and environment, stellar feedback, numerical solver, output,
and a handful of sweep-mode directives. For each keyword the
default value, unit, and short description are given. The same
information can be inspected at run time through the
``DescribedItem`` objects attached to every entry in the output
state dictionary (see :ref:`sec-running`, *Output Data Model*).


File Format
-----------

The canonical parameter schema (the set of valid keys and their default
values) lives at ``src/_input/default.param``. User-facing example
``.param`` files (one per run configuration) live under ``param/``;
see ``param/simple_cluster.param`` or ``param/rosette.param`` for
worked examples. Parameter files are generically formatted as a series
of entries of the form::

    keyword    value

Any line starting with ``#`` is considered a comment and is
ignored, and anything on a line after a ``#`` is similarly ignored.
Some general rules on keywords:

* Keywords may appear in any order.
* Many keywords have default values, indicated in parentheses in
  the list below. These keywords are optional and need not appear
  in the parameter file; missing keywords are set to their default
  values. Keywords that do not have a default are required.
* Keyword names are case-sensitive.
* Unless explicitly stated otherwise, the input unit system is CGS
  extended by :math:`M_\odot` for mass and Myr for time; the
  internal unit system in which values are reported on output is
  described below.
* A value written as a bracketed list (``mCloud [1e5, 1e6]``) or
  through a ``tuple(...)`` directive turns the file into a sweep;
  see :ref:`sec-running` for the full sweep syntax.

Supported Value Types
^^^^^^^^^^^^^^^^^^^^^

TRINITY parses values in the following order of precedence:

==================  ========================  ============================
Type                Example                   Notes
==================  ========================  ============================
Boolean             ``True``, ``False``       Case-sensitive
Scientific          ``1e6``, ``3.14e-2``      Standard notation
Fraction            ``5/3``                   Converted to float (1.6667)
Number              ``100``, ``0.01``         Integer or decimal
String              ``densPL``, ``my_model``  Fallback for text values
==================  ========================  ============================


Unit System
-----------

Input Units (CGS)
^^^^^^^^^^^^^^^^^

Parameters are specified in CGS units in the parameter file. TRINITY internally converts all quantities to astronomy units for numerical stability.

**Common input units:**

=================  ===========================
Quantity           Input Unit
=================  ===========================
Mass               :math:`M_\odot` (solar mass)
Length             pc (parsec) or cm
Time               Myr or s
Number density     cm\ :math:`^{-3}`
Velocity           km s\ :math:`^{-1}`
Temperature        K (Kelvin)
=================  ===========================

Internal Units
^^^^^^^^^^^^^^

TRINITY uses astronomy units internally: **[Msun, pc, Myr]**

The conversion is handled automatically based on the ``# UNIT:`` annotations in ``default.param``.

**Supported unit strings in annotations:**

.. code-block:: text

    # UNIT: [Msun]
    # UNIT: [cm**-3]
    # UNIT: [km * s**-1]
    # UNIT: [erg * s**-1 * cm**-1 * K**(-7/2)]


Parameter Reference
-------------------

Administrative Parameters
^^^^^^^^^^^^^^^^^^^^^^^^^

These parameters control simulation naming and output.

.. list-table::
   :widths: 25 15 60
   :header-rows: 1

   * - Parameter
     - Default
     - Description
   * - ``model_name``
     - ``default``
     - Prefix for all output filenames. Use alphanumeric characters and underscores only.
   * - ``path2output``
     - ``def_dir``
     - Output directory path. ``def_dir`` uses the current working directory.
   * - ``output_format``
     - ``JSON``
     - Output format. Currently only JSON is supported.
   * - ``simplify_npoints``
     - ``100``
     - Target number of points retained for the simplified profile arrays written
       into each snapshot (``bubble_T_arr``, ``bubble_n_arr``, ``bubble_dTdr_arr``,
       ``bubble_v_arr``, ``shell_grav_force_m``, ``shell_n_arr``). Larger values
       give higher-fidelity snapshots at the cost of larger output files. Clamped
       to ``>= 20`` (matches the coverage-skeleton chunk count); the first two
       simplify calls per implicit-phase snapshot log their reconstruction
       :math:`R^2` at ``INFO`` level so you can verify the chosen budget is faithful.

Logging Parameters
^^^^^^^^^^^^^^^^^^

Configure how TRINITY reports progress and diagnostics.

.. list-table::
   :widths: 25 15 60
   :header-rows: 1

   * - Parameter
     - Default
     - Description
   * - ``log_level``
     - ``DEBUG``
     - Logging verbosity: ``DEBUG``, ``INFO``, ``WARNING``, ``ERROR``, ``CRITICAL``. See :ref:`sec-running` for details.
   * - ``log_console``
     - ``False``
     - Enable terminal output for log messages.
   * - ``log_file``
     - ``True``
     - Write log messages to ``{path2output}/trinity.log``.
   * - ``log_colors``
     - ``True``
     - Color-code terminal output by severity level.


Physical Parameters
^^^^^^^^^^^^^^^^^^^

Core parameters defining the molecular cloud and star formation.

.. list-table::
   :widths: 25 15 15 45
   :header-rows: 1

   * - Parameter
     - Default
     - Unit
     - Description
   * - ``mCloud``
     - ``1e6``
     - :math:`M_\odot`
     - Total mass of the molecular cloud.
   * - ``sfe``
     - ``0.01``
     - --
     - Star formation efficiency (0 < sfe < 1). Fraction of cloud mass converted to stars.
   * - ``ZCloud``
     - ``1``
     - :math:`Z_\odot`
     - Cloud metallicity. **Currently only solar (1) is supported.**
   * - ``include_PHII``
     - ``True``
     - --
     - Include HII pressure (from Strömgren ionization balance in the shell) in :math:`P_{\rm drive}`. When ``False``, :math:`P_{\rm HII}` is set to zero.

**Derived quantities:**

- Cluster mass: :math:`M_{\rm cluster} = M_{\rm cloud} \times {\rm sfe}`
- Remaining cloud mass: :math:`M_{\rm cloud,after} = M_{\rm cloud} - M_{\rm cluster}`


Density Profile Parameters
^^^^^^^^^^^^^^^^^^^^^^^^^^

Define the radial density structure of the molecular cloud.

**Profile Selection:**

.. list-table::
   :widths: 25 15 60
   :header-rows: 1

   * - Parameter
     - Default
     - Description
   * - ``dens_profile``
     - ``densPL``
     - Density profile type: ``densPL`` (power-law) or ``densBE`` (Bonnor-Ebert)

**Common Parameters:**

.. list-table::
   :widths: 25 15 15 45
   :header-rows: 1

   * - Parameter
     - Default
     - Unit
     - Description
   * - ``nCore``
     - ``1e5``
     - cm\ :math:`^{-3}`
     - Core number density. For homogeneous clouds (``densPL_alpha=0``), this is the average density.
   * - ``nISM``
     - ``1``
     - cm\ :math:`^{-3}`
     - Ambient ISM number density.
   * - ``rCore``
     - ``0.01``
     - pc
     - Core radius. Not used for homogeneous clouds.

Power-Law Profile (densPL)
""""""""""""""""""""""""""

When ``dens_profile = densPL``, the density follows:

.. math::

    \rho(r) = \begin{cases}
    \rho_0 & r \leq r_0 \\
    \rho_0 \left(\frac{r}{r_0}\right)^\alpha & r_0 < r \leq r_{\rm cloud} \\
    \rho_{\rm ISM} & r > r_{\rm cloud}
    \end{cases}

.. list-table::
   :widths: 25 15 60
   :header-rows: 1

   * - Parameter
     - Default
     - Description
   * - ``densPL_alpha``
     - ``0``
     - Power-law exponent (:math:`-2 \leq \alpha \leq 0`). Special cases: ``0`` = homogeneous, ``-2`` = isothermal sphere.

Bonnor-Ebert Profile (densBE)
"""""""""""""""""""""""""""""

When ``dens_profile = densBE``, implements a Bonnor-Ebert sphere (`Ebert 1955 <https://ui.adsabs.harvard.edu/abs/1955ZA.....37..217E/abstract>`_; `Bonnor 1956 <https://ui.adsabs.harvard.edu/abs/1956MNRAS.116..351B/abstract>`_).

.. list-table::
   :widths: 25 15 60
   :header-rows: 1

   * - Parameter
     - Default
     - Description
   * - ``densBE_Omega``
     - ``14.1``
     - Density contrast :math:`\Omega = \rho_{\rm center}/\rho_{\rm edge}`. Values > 14.1 indicate gravitational instability.

.. note::

   Conditional parameters: ``densPL_alpha`` is ignored when using ``densBE``, and ``densBE_Omega`` is ignored when using ``densPL``.


Termination Parameters
^^^^^^^^^^^^^^^^^^^^^^

Conditions that end the simulation.

.. list-table::
   :widths: 25 15 15 45
   :header-rows: 1

   * - Parameter
     - Default
     - Unit
     - Description
   * - ``allowShellDissolution``
     - ``True``
     - --
     - Allow shell dissolution to terminate the simulation. If ``False``, the dissolution check is disabled.
   * - ``stop_t_diss``
     - ``1``
     - Myr
     - Duration ``shell_nMax`` must remain below ``nISM`` before dissolution is triggered.
   * - ``stop_r``
     - ``500``
     - pc
     - Maximum shell radius. Exceeding this triggers termination. Set to ``None`` to disable this condition.
   * - ``stop_v``
     - ``-1e4``
     - km/s
     - Velocity threshold for numerical instability detection.
   * - ``stop_t``
     - ``15``
     - Myr
     - Maximum simulation duration. Set to ``None`` to disable this condition.
   * - ``stop_at_rCloud_nSnap``
     - ``None``
     - --
     - Terminate after the shell crosses the cloud edge (R2 > rCloud).
       ``None`` disables.  ``0`` stops at the edge (only the energy-phase
       reconciliation snapshot at R2 = rCloud is recorded).  ``N > 0`` lets
       the implicit phase advance for ``N`` more segment-loop snapshots
       past the crossing before terminating; the implicit phase's
       end-of-phase reconciliation snapshot adds one extra past-rCloud
       sample, so the total snapshots with R2 ≥ rCloud is roughly
       ``N + 2`` (1 at-edge + ``N`` in-loop + 1 reconciliation).

.. note::

   Setting ``stop_r``, ``stop_t``, or ``stop_at_rCloud_nSnap`` to ``None``
   disables that termination condition, allowing the simulation to continue
   until other conditions are met (e.g., shell dissolution, collapse, or
   cloud boundary).

Collapse Parameters
^^^^^^^^^^^^^^^^^^^

.. list-table::
   :widths: 25 15 15 45
   :header-rows: 1

   * - Parameter
     - Default
     - Unit
     - Description
   * - ``coll_r``
     - ``1``
     - pc
     - Radius below which the cloud is considered completely collapsed.


Starburst99 Parameters
^^^^^^^^^^^^^^^^^^^^^^

Configure the stellar population synthesis model for feedback calculations.

.. list-table::
   :widths: 25 15 15 45
   :header-rows: 1

   * - Parameter
     - Default
     - Unit
     - Description
   * - ``SB99_mass``
     - ``1e6``
     - :math:`M_\odot`
     - Reference cluster mass used in SB99 files. Used for scaling.
   * - ``SB99_rotation``
     - ``1``
     - --
     - Include stellar rotation (1=yes, 0=no). Rotation extends lifetimes via internal mixing.
   * - ``SB99_BHCUT``
     - ``120``
     - :math:`M_\odot`
     - Black hole formation threshold. Stars above this ZAMS mass collapse directly to BH without SN.


Feedback Parameters
^^^^^^^^^^^^^^^^^^^

Control mass injection and energy thermalization from stellar feedback.

.. list-table::
   :widths: 25 15 15 45
   :header-rows: 1

   * - Parameter
     - Default
     - Unit
     - Description
   * - ``FB_mColdWindFrac``
     - ``0``
     - --
     - Fraction of cold material swept up by protostellar winds.
   * - ``FB_mColdSNFrac``
     - ``0``
     - --
     - Fraction of cold ejecta from supernovae.
   * - ``FB_thermCoeffWind``
     - ``1``
     - --
     - Thermalization efficiency for stellar wind kinetic energy.
   * - ``FB_thermCoeffSN``
     - ``1``
     - --
     - Thermalization efficiency for supernova ejecta.
   * - ``FB_vSN``
     - ``1e4``
     - km/s
     - Supernova ejecta velocity.


Phase Control Parameters
^^^^^^^^^^^^^^^^^^^^^^^^

Control transitions between simulation phases.

.. list-table::
   :widths: 30 15 55
   :header-rows: 1

   * - Parameter
     - Default
     - Description
   * - ``adiabaticOnlyInCore``
     - ``False``
     - Restrict adiabatic (energy-driven) phase to within core radius.
   * - ``immediate_leak``
     - ``True``
     - Transition immediately to momentum-driven phase when bubble bursts.
   * - ``phaseSwitch_LlossLgain``
     - ``0.05``
     - Threshold for :math:`(L_{\rm gain} - L_{\rm loss})/L_{\rm gain}` to trigger phase transition.
   * - ``use_adaptive_solver``
     - ``True``
     - Use the adaptive ODE solver for the energy-driven phase
       (``run_energy_phase_modified.py``). If ``False``, falls back to the
       original solver (``run_energy_phase.py``).


Cooling Parameters
^^^^^^^^^^^^^^^^^^

Parameters for radiative cooling calculations.

.. list-table::
   :widths: 25 15 60
   :header-rows: 1

   * - Parameter
     - Default
     - Description
   * - ``cool_alpha``
     - ``0.6``
     - Cooling parameter: :math:`\alpha = v_2 \cdot t_{\rm now} / R_2`
   * - ``cool_beta``
     - ``0.8``
     - Cooling parameter: :math:`\beta = -dP_b/dt`
   * - ``cool_delta``
     - ``-6/35``
     - Cooling parameter: :math:`\delta = dT/dt`


Path Configuration
^^^^^^^^^^^^^^^^^^

Specify paths to external data files.

.. list-table::
   :widths: 25 15 60
   :header-rows: 1

   * - Parameter
     - Default
     - Description
   * - ``path_cooling_CIE``
     - ``3``
     - Cooling curve for CIE (T > 10\ :sup:`5.5` K). Integer presets: 1=CLOUDY HII, 2=CLOUDY+grains, 3=Gnat & Ferland 2012, 4=Sutherland & Dopita 0.15Z.
   * - ``path_cooling_nonCIE``
     - ``def_dir``
     - Path to non-CIE cooling curves (T < 10\ :sup:`5.5` K).
   * - ``path_sps``
     - ``def_dir``
     - Path to Starburst99 stellar population files.


Physical Constants
^^^^^^^^^^^^^^^^^^

Standard physical constants. Typically not modified.

**Mean Molecular Weights:**

.. list-table::
   :widths: 20 25 55
   :header-rows: 1

   * - Parameter
     - Default
     - Description
   * - ``mu_atom``
     - ``1.273``
     - Neutral atomic gas (HI + He). :math:`\mu = 14/11`
   * - ``mu_ion``
     - ``0.609``
     - Fully ionized gas (H+ + He++). :math:`\mu = 14/23`
   * - ``mu_mol``
     - ``2.333``
     - Molecular gas (H2 + He). :math:`\mu = 14/6`
   * - ``mu_convert``
     - ``1.4``
     - Mass density conversion factor (n to :math:`\rho`).

**Temperature Constants:**

.. list-table::
   :widths: 20 15 15 50
   :header-rows: 1

   * - Parameter
     - Default
     - Unit
     - Description
   * - ``TShell_neu``
     - ``1e2``
     - K
     - Neutral shell temperature.
   * - ``TShell_ion``
     - ``1e4``
     - K
     - Ionized shell temperature.

**Dust Parameters:**

.. list-table::
   :widths: 20 15 20 45
   :header-rows: 1

   * - Parameter
     - Default
     - Unit
     - Description
   * - ``dust_sigma``
     - ``1.5e-21``
     - cm\ :math:`^2`
     - Dust cross-section at solar metallicity.
   * - ``dust_noZ``
     - ``0.05``
     - :math:`Z_\odot`
     - Metallicity below which dust is negligible.
   * - ``dust_KappaIR``
     - ``4``
     - cm\ :math:`^2`/g
     - Rosseland mean dust opacity :math:`\kappa_{\rm IR}`.

**Fundamental Constants:**

.. list-table::
   :widths: 20 20 20 40
   :header-rows: 1

   * - Parameter
     - Default
     - Unit
     - Description
   * - ``gamma_adia``
     - ``5/3``
     - --
     - Adiabatic index.
   * - ``caseB_alpha``
     - ``2.59e-13``
     - cm\ :math:`^3`/s
     - Case B recombination coefficient.
   * - ``C_thermal``
     - ``6e-7``
     - erg/(s cm K\ :sup:`7/2`)
     - Thermal conduction coefficient.
   * - ``c_light``
     - ``2.998e10``
     - cm/s
     - Speed of light.
   * - ``G``
     - ``6.674e-8``
     - cm\ :math:`^3`/(g s\ :math:`^2`)
     - Gravitational constant.
   * - ``k_B``
     - ``1.381e-16``
     - erg/K
     - Boltzmann constant.
   * - ``PISM``
     - ``0``
     - K cm\ :math:`^{-3}`
     - ISM pressure :math:`P/k_B`.

**Bubble Structure:**

.. list-table::
   :widths: 20 15 65
   :header-rows: 1

   * - Parameter
     - Default
     - Description
   * - ``bubble_xi_Tb``
     - ``0.98``
     - Relative radius :math:`\xi = r/R_2` for measuring bubble temperature.


Sweep Syntax
------------

Any parameter can also be given as a list (``[v1, v2, ...]``) or combined
into a ``tuple(...)`` directive to launch a parameter sweep. ``run.py``
auto-detects this from the file content, so the same script runs both single
simulations and sweeps.

See :ref:`sec-running` — *Parameter Sweep Runs* — for:

- Worked examples of **Cartesian**, **tuple**, and **hybrid** sweep syntax.
- Command-line options (``--dry-run``, ``--workers``, ``--yes``,
  ``--verbose``) and cancellation behaviour.
- Pre-flight GMC plausibility checks applied to every combination.
- Auto-generated run names and the resulting output directory layout.


Examples
--------

Minimal Parameter File
^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: text
   :caption: minimal.param

    model_name    my_simulation
    mCloud        1e6
    sfe           0.01

Power-Law Cloud
^^^^^^^^^^^^^^^

.. code-block:: text
   :caption: powerlaw.param

    # Model identification
    model_name      powerlaw_test
    path2output     outputs/powerlaw

    # Cloud properties
    mCloud          1e7
    sfe             0.05
    ZCloud          1

    # Power-law density profile
    dens_profile    densPL
    densPL_alpha    -1.5
    nCore           1e4
    rCore           0.5
    nISM            1

    # Termination
    stop_t          20
    stop_r          300

Bonnor-Ebert Sphere
^^^^^^^^^^^^^^^^^^^

.. code-block:: text
   :caption: bonnor_ebert.param

    # Model identification
    model_name      BE_sphere
    path2output     outputs/BE

    # Cloud properties
    mCloud          1e5
    sfe             0.02

    # Bonnor-Ebert profile
    dens_profile    densBE
    densBE_Omega    14.1
    nCore           1e5
    rCore           0.1

For sweep-style parameter files (``[v1, v2]`` list values and
``tuple(...)`` directives), see :ref:`sec-running` — *Parameter
Sweep Runs* — for worked Cartesian, tuple, and hybrid examples.


See Also
--------

- :ref:`sec-running` — how to execute single runs and parameter sweeps,
  including CLI flags, worker defaults, and output directory layout.
- :ref:`sec-physics` — equations and derivations behind ``dens_profile``,
  ``densPL_alpha``, ``densBE_Omega``, and the feedback parameters.
- :ref:`sec-trinity-reader` — reading back the JSONL output written by a
  simulation configured with these parameters.
- ``src/_input/default.param`` in the repository — the authoritative,
  fully-commented schema + defaults file. User ``.param`` files in
  ``param/`` override these defaults.
