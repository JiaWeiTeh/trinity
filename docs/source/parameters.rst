.. highlight:: rest

.. _sec-parameters:

Parameter Specifications
========================

Every TRINITY simulation is driven by a plain-text parameter file.
This chapter describes how such files are formatted and enumerates the
keywords TRINITY recognises, grouped by physical role: cloud and
environment, density profile, termination, stellar feedback, feedback
corrections, the numerical solver, cooling, and physical constants. For
each keyword the default value, unit, and short description are given.


Source of truth
---------------

The parameter schema — the complete set of keys, defaults, units, and
descriptions — is defined by the **ParamSpec registry** at
``trinity/_input/registry.py``. Everything else is derived from it:

* ``trinity/_input/default.param`` is *generated* from the registry
  (run ``python -m tools.gen_default_param --write`` after editing the
  registry) and is what ``read_param`` loads as the default layer.
* Which keys are written to ``metadata.json`` versus repeated in every
  snapshot is projected from per-spec flags (see :ref:`sec-running`,
  *Output data model*).
* The per-key ``info`` strings and ``ori_units`` labels surfaced by the
  reader (``output.info(verbose=True)``) come straight from the
  registry.

The tables below mirror the registry; when in doubt, the registry
wins. Worked example files live under ``param/`` (e.g.
``param/simple_cluster.param`` or ``param/cloud_example_PL.param``).


File format
-----------

A parameter file contains one ``keyword    value`` entry per line. A
``#`` starts a comment, either as a whole line or after a value.
Keyword names are case-sensitive and may appear in any order.

Keywords with a default (listed below) are optional; those without a
default are required. A value written as a bracketed list
(``mCloud [1e5, 1e6]``) or through a ``tuple(...)`` directive turns the
file into a sweep — see :ref:`sec-running` for the sweep syntax.

Supported value types
^^^^^^^^^^^^^^^^^^^^^^

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

Unit system
^^^^^^^^^^^

Inputs in the parameter file are CGS, extended by :math:`M_\odot`
(mass) and Myr (time). Common per-quantity units: pc for length,
cm\ :math:`^{-3}` for number density, km/s for velocity, K for
temperature. Internally TRINITY works in ``[Msun, pc, Myr]``;
conversion is automatic, driven by the ``# UNIT:`` annotations in
``default.param``.


Administrative parameters
-------------------------

.. list-table::
   :widths: 25 15 60
   :header-rows: 1

   * - Parameter
     - Default
     - Description
   * - ``model_name``
     - ``default``
     - Prefix for output naming. Use alphanumerics and underscores only.
   * - ``path2output``
     - ``def_dir``
     - Output directory. The sentinel ``def_dir`` resolves to
       ``outputs/<model_name>/`` under the current working directory; the
       directory is created automatically.
   * - ``output_format``
     - ``JSON``
     - Output format. Currently only JSON (JSONL) is supported.
   * - ``simplify_npoints``
     - ``100``
     - Target number of points retained for the downsampled profile
       arrays written into each snapshot (``bubble_T_arr``,
       ``bubble_n_arr``, ``bubble_dTdr_arr``, ``bubble_v_arr``,
       ``shell_grav_force_m``, ``shell_n_arr``). Larger values give
       higher-fidelity snapshots at the cost of larger files. Clamped to
       ``>= 20``.
   * - ``log_level``
     - ``DEBUG``
     - Verbosity: ``DEBUG``, ``INFO``, ``WARNING``, ``ERROR``,
       ``CRITICAL``. See :ref:`sec-running`.
   * - ``log_console``
     - ``False``
     - Enable terminal output for log messages.
   * - ``log_file``
     - ``True``
     - Write log messages to ``{path2output}/trinity.log``.
   * - ``log_colors``
     - ``True``
     - Colour-code terminal output by severity.


Physical parameters
--------------------

Core parameters defining the molecular cloud and star formation.

.. list-table::
   :widths: 25 15 15 45
   :header-rows: 1

   * - Parameter
     - Default
     - Unit
     - Description
   * - ``mCloud``
     - ``1e7``
     - :math:`M_\odot`
     - Total mass of the molecular cloud.
   * - ``sfe``
     - ``0.01``
     - --
     - Star formation efficiency (0 < sfe < 1). Fraction of cloud mass
       converted to stars.
   * - ``ZCloud``
     - ``1``
     - :math:`Z_\odot`
     - Cloud metallicity. **Currently only solar (1) is supported.**
   * - ``include_PHII``
     - ``True``
     - --
     - Include HII pressure (from Strömgren ionization balance in the
       shell) in the driving pressure. When ``False``,
       :math:`P_{\rm HII}` is set to zero.
   * - ``coverFraction``
     - ``1.0``
     - --
     - Closed fraction :math:`C_f` of the bubble wall (geometry-set
       energy leak). Hot gas vents through the open area
       :math:`(1-C_f)\,4\pi R_2^2` at the interior sound speed, draining
       bubble energy. ``1.0`` recovers the sealed (Weaver) bubble
       exactly; validated to ``0 < Cf <= 1``. Usable range ~0.9–0.99 —
       values near 0 drain the bubble within a step and stress the
       integrator. Only the energy leak is implemented so far; the
       matching mass sink is not yet modelled.
   * - ``nCore``
     - ``1e5``
     - cm\ :math:`^{-3}`
     - Core hydrogen-nuclei number density. For homogeneous clouds
       (``densPL_alpha = 0``) this is the average density.
   * - ``nISM``
     - ``1``
     - cm\ :math:`^{-3}`
     - Ambient ISM number density.
   * - ``rCore``
     - ``0.01``
     - pc
     - Core radius. Not used for homogeneous clouds.
   * - ``rCloud_max``
     - ``200``
     - pc
     - Maximum plausible cloud radius for the pre-run GMC validation. If
       the computed ``rCloud`` exceeds this, the run is rejected as
       implausibly diffuse for the given mass. Raise it to allow larger,
       more diffuse clouds.

**Derived quantities:**

- Cluster mass: :math:`M_{\rm cluster} = M_{\rm cloud} \times {\rm sfe}`
- Remaining cloud mass:
  :math:`M_{\rm cloud,after} = M_{\rm cloud} - M_{\rm cluster}`


Density profile parameters
--------------------------

.. list-table::
   :widths: 25 15 60
   :header-rows: 1

   * - Parameter
     - Default
     - Description
   * - ``dens_profile``
     - ``densPL``
     - Density profile type: ``densPL`` (power-law) or ``densBE``
       (Bonnor-Ebert).

Power-law profile (densPL)
^^^^^^^^^^^^^^^^^^^^^^^^^^

When ``dens_profile = densPL``, the density follows a flat core, a
power-law envelope, and the ambient ISM beyond the cloud edge:

.. math::

    n(r) = \begin{cases}
    n_{\rm core} & r \leq r_{\rm core} \\
    n_{\rm core} \left(\frac{r}{r_{\rm core}}\right)^\alpha & r_{\rm core} < r \leq r_{\rm cloud} \\
    n_{\rm ISM} & r > r_{\rm cloud}
    \end{cases}

.. list-table::
   :widths: 25 15 60
   :header-rows: 1

   * - Parameter
     - Default
     - Description
   * - ``densPL_alpha``
     - ``0``
     - Power-law exponent (:math:`-2 \leq \alpha \leq 0`). Special cases:
       ``0`` = homogeneous, ``-2`` = isothermal sphere.

Bonnor-Ebert profile (densBE)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

When ``dens_profile = densBE``, the cloud is an isothermal,
self-gravitating Bonnor-Ebert sphere (`Ebert 1955
<https://ui.adsabs.harvard.edu/abs/1955ZA.....37..217E/abstract>`_;
`Bonnor 1956
<https://ui.adsabs.harvard.edu/abs/1956MNRAS.116..351B/abstract>`_).

.. list-table::
   :widths: 25 15 60
   :header-rows: 1

   * - Parameter
     - Default
     - Description
   * - ``densBE_Omega``
     - ``14.1``
     - Density contrast :math:`\Omega = \rho_{\rm core}/\rho_{\rm edge}`.
       Values above the critical :math:`\Omega \approx 14.04` are
       gravitationally unstable.

.. note::

   ``densPL_alpha`` is ignored when using ``densBE``, and
   ``densBE_Omega`` is ignored when using ``densPL``. If you set
   ``dens_profile`` explicitly in a ``.param`` file, you must also set
   the matching companion (``densPL_alpha`` or ``densBE_Omega``) — the
   defaults are only assumed when ``dens_profile`` is left at its
   default too.


Termination parameters
----------------------

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
     - Allow shell dissolution to terminate the simulation. If ``False``,
       the dissolution check is disabled.
   * - ``stop_t_diss``
     - ``1``
     - Myr
     - Duration ``shell_nMax`` must remain below ``nISM`` before
       dissolution is triggered.
   * - ``stop_r``
     - ``500``
     - pc
     - Maximum shell radius. Set to ``None`` to disable.
   * - ``stop_t``
     - ``15``
     - Myr
     - Maximum simulation duration. Set to ``None`` to disable.
   * - ``stop_at_rCloud_nSnap``
     - ``None``
     - --
     - Terminate after the shell crosses the cloud edge (R2 > rCloud).
       ``None`` disables. ``0`` stops at the edge. ``N > 0`` lets the
       implicit phase advance ``N`` more segment-loop snapshots past the
       crossing before terminating.
   * - ``coll_r``
     - ``1``
     - pc
     - Radius below which the cloud is considered fully collapsed.


.. _sec-parameters-sps:

Stellar feedback (SPS)
----------------------

TRINITY reads time-evolving stellar feedback (ionizing photon rate,
bolometric and mechanical luminosities, wind and SN momentum injection,
…) from a stellar-population-synthesis (SPS) table.

.. list-table::
   :widths: 25 15 15 45
   :header-rows: 1

   * - Parameter
     - Default
     - Unit
     - Description
   * - ``sps_path``
     - ``def_path``
     - --
     - Path to an SPS data file. ``def_path`` loads the bundled default
       (see below). Otherwise points at any ``.txt`` / ``.csv`` file
       whose columns are described by ``sps_col_*`` declarations.
   * - ``sps_refmass``
     - ``def_value``
     - :math:`M_\odot`
     - Reference cluster mass for the feedback scaling
       :math:`f_{\rm mass} = M_{\rm cluster} / {\rm sps\_refmass}`.
       ``def_value`` resolves to ``1e6`` for the bundled file; a custom
       ``sps_path`` **requires** an explicit value.
   * - ``SB99_rotation``
     - ``1``
     - --
     - Keys the non-CIE cooling-table selection (rot vs norot). Only rot
       tables ship with the repo, so ``0`` requires a user-supplied
       ``sps_path`` plus matching cooling tables; ``0`` with
       ``sps_path = def_path`` raises at config-load.

When ``sps_path = def_path``, TRINITY loads the bundled file
``lib/default/sps/starburst99/1e6cluster_default.csv`` (solar
metallicity, rotation on, :math:`10^6\,M_\odot` reference cluster) using
the built-in 7-column SB99 preset. The default fallback only accepts
``ZCloud = 1.0`` and ``SB99_rotation = 1``, because the bundled non-CIE
cooling tables are rot-only.

Custom SPS files (``sps_col_*`` declarations)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The canonical column model is defined in ``trinity/sps/sps_columns.py``
(the single source of truth for the SPS loader). When ``sps_path`` is
an explicit file, describe its layout with one
``sps_col_<canonical>`` line per mapped column:

.. code-block:: text

    sps_col_<canonical>    <file_column>    <units>    <log|linear>

where:

* ``<canonical>`` is one of the canonical names in the table below.
* ``<file_column>`` is **either** a 0-based integer column index (works
  on any file), **or** a header-row name (the file must have a header).
* ``<units>`` is the declared unit of the column. The alias ``cgs`` is
  accepted as shorthand for each canonical's default cgs unit.
* ``<log|linear>`` declares whether file values are stored as
  :math:`\log_{10}` of the linear value.

The file may be ``.txt`` (whitespace-separated) or ``.csv``; the
delimiter is sniffed from the first data row. ``#`` lines and blank
lines are comments.

**Required canonicals** (loader will not start without them):

* ``t``, ``Qi``, ``Lbol``, ``Lmech_W``, ``pdot_W``
* **either** ``fi`` **or both** ``Li`` **and** ``Ln`` (the latter
  bypasses the hardcoded 13.6 eV ionizing-fraction split)
* **either** ``Lmech_total`` **or** ``Lmech_SN`` to drive the SN
  pipeline

**Optional canonicals** (the loader derives them when absent):
``Lmech_total``, ``Lmech_SN``, ``pdot_SN``, ``Mdot_SN``, ``v_SN``,
``Li``, ``Ln``.

Per-canonical recognised units and the AU unit each lands in:

.. list-table::
   :widths: 22 18 30 30
   :header-rows: 1

   * - Canonical
     - ``cgs`` alias
     - Other accepted ``<units>``
     - AU target
   * - ``t``
     - ``s``
     - ``yr``, ``Myr``
     - Myr
   * - ``Qi``
     - ``1/s``
     - ``1/Myr``
     - 1/Myr
   * - ``fi``
     - ``dimensionless``
     - --
     - dimensionless
   * - ``Lbol``, ``Lmech_*``, ``Li``, ``Ln``
     - ``erg/s``
     - ``L_sun``
     - :math:`M_\odot\,{\rm pc}^2/{\rm Myr}^3`
   * - ``pdot_W``, ``pdot_SN``
     - ``g*cm/s^2``
     - --
     - :math:`M_\odot\,{\rm pc}/{\rm Myr}^2`
   * - ``Mdot_SN``
     - ``g/s``
     - ``Msun/Myr``
     - :math:`M_\odot/{\rm Myr}`
   * - ``v_SN``
     - ``cm/s``
     - ``km/s``, ``pc/Myr``
     - pc/Myr

Mass-scaled canonicals (everything except ``t``, ``fi``, ``v_SN``) are
multiplied by :math:`f_{\rm mass}` after unit conversion.

**Example — headered file with the ``cgs`` alias.** A custom file
``my_sps.txt`` whose first line is
``time Qi fi Lbol Lmech_total pdot_W Lmech_W`` with data in
:math:`\log_{10}` cgs (except ``time`` linear yr and ``fi`` linear):

.. code-block:: text

    sps_path        /absolute/path/to/my_sps.txt
    sps_refmass     1e6

    sps_col_t            time         yr             linear
    sps_col_Qi           Qi           cgs            log
    sps_col_fi           fi           dimensionless  linear
    sps_col_Lbol         Lbol         cgs            log
    sps_col_Lmech_total  Lmech_total  cgs            log
    sps_col_pdot_W       pdot_W       cgs            log
    sps_col_Lmech_W      Lmech_W      cgs            log

The same columns can be mapped by 0-based integer index instead of
header name (``sps_col_t 0 yr linear`` …); indices and names may be
mixed across lines. If ``sps_col_*`` declarations are missing or
inconsistent while ``sps_path`` is set, the loader hard-errors at
startup with a fillable template listing exactly which canonicals are
missing.


Feedback corrections
--------------------

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
     - Fraction of cold material entrained by stellar winds.
   * - ``FB_mColdSNFrac``
     - ``0``
     - --
     - Fraction of cold material entrained in supernova ejecta.
   * - ``FB_thermCoeffWind``
     - ``1``
     - --
     - Thermalization efficiency for stellar-wind kinetic energy.
   * - ``FB_thermCoeffSN``
     - ``1``
     - --
     - Thermalization efficiency for supernova ejecta.
   * - ``FB_vSN``
     - ``1e4``
     - km/s
     - Supernova ejecta velocity.


Solver parameters
-----------------

.. list-table::
   :widths: 30 15 55
   :header-rows: 1

   * - Parameter
     - Default
     - Description
   * - ``phaseSwitch_LlossLgain``
     - ``0.05``
     - Threshold for :math:`(L_{\rm gain} - L_{\rm loss})/L_{\rm gain}`
       below which the implicit energy phase hands off to the transition
       phase.
   * - ``bubble_xi_Tb``
     - ``0.98``
     - Relative radius :math:`\xi = r/R_2` at which the bubble
       temperature is measured.
   * - ``cool_alpha``
     - ``0.6``
     - Cooling parameter :math:`\alpha = v_2 t_{\rm now}/R_2`.
   * - ``cool_beta``
     - ``0.8``
     - Cooling parameter :math:`\beta = -dP_b/dt`.
   * - ``cool_delta``
     - ``-6/35``
     - Cooling parameter :math:`\delta = dT/dt`.


Cooling tables
--------------

.. list-table::
   :widths: 25 15 60
   :header-rows: 1

   * - Parameter
     - Default
     - Description
   * - ``path_cooling_CIE``
     - ``3``
     - Selects the CIE (T > 10\ :sup:`5.5` K) cooling table. Integer
       presets bundled under ``lib/default/CIE/``:

       - ``1`` → ``coolingCIE_1_Cloudy.dat`` (CLOUDY HII)
       - ``2`` → ``coolingCIE_2_Cloudy_grains.dat`` (CLOUDY + grains)
       - ``3`` → ``coolingCIE_3_Gnat-Ferland2012.dat`` (Gnat & Ferland 2012)
   * - ``path_cooling_nonCIE``
     - ``def_dir``
     - Folder of non-CIE (T < 10\ :sup:`5.5` K) OPIATE/CLOUDY cubes.
       ``def_dir`` resolves to ``lib/default/opiate/``. Per-age files are
       selected at runtime from ``SB99_rotation`` + ``ZCloud``.


Physical constants
------------------

Standard physical constants. Typically not modified. Defaults are shown
as they appear in the registry (fractions where exact).

**Gas composition.** The helium fraction ``x_He`` (:math:`=n_{\rm He}/n_{\rm H}`)
and the helium ionisation states ``Z_He`` (hot bubble) and ``Z_He_shell``
(:math:`\sim10^4`\ K shell / HII region) are the single source of truth: the mean
masses per particle and electron factors below are all *derived* from them at
load. **All number densities** :math:`n` **in TRINITY are hydrogen-nuclei
densities** :math:`n_{\rm H}`, with mass density :math:`\rho=\mu_{\rm H}\,n_{\rm H}`.

.. list-table:: Composition inputs
   :widths: 20 16 64
   :header-rows: 1

   * - Parameter
     - Default
     - Description
   * - ``x_He``
     - ``0.1``
     - Helium-to-hydrogen number ratio :math:`n_{\rm He}/n_{\rm H}`.
   * - ``Z_He``
     - ``2``
     - He ionisation in the hot bubble (2 = doubly ionised, He\ :math:`^{2+}`).
   * - ``Z_He_shell``
     - ``1``
     - He ionisation in the :math:`\sim10^4`\ K shell (1 = singly ionised, He\ :math:`^{+}`).

**Derived mean masses per particle** (units of :math:`m_{\rm H}`) and electron
factors :math:`\chi_e=n_e/n_{\rm H}`:

.. list-table::
   :widths: 22 16 62
   :header-rows: 1

   * - Parameter
     - Default
     - Description
   * - ``mu_convert``
     - ``1.4``
     - :math:`\mu_{\rm H}=(1+4x_{\rm He})`, mass per H nucleus; use for
       :math:`\rho=\mu_{\rm H}n_{\rm H}` (ionisation-independent).
   * - ``mu_atom``
     - ``14/11``
     - Neutral atomic gas, :math:`\mu_{\rm H}/(1+x_{\rm He})\approx1.273`.
   * - ``mu_mol``
     - ``14/6``
     - Molecular gas, :math:`\mu_{\rm H}/(0.5+x_{\rm He})\approx2.333`.
   * - ``mu_ion``
     - ``14/23``
     - Hot bubble (He\ :math:`^{2+}`),
       :math:`\mu_{\rm H}/(2+x_{\rm He}(1+Z_{\rm He}))\approx0.609`.
   * - ``mu_ion_shell``
     - *(derived)*
     - :math:`\sim10^4`\ K shell (He\ :math:`^{+}`), :math:`=14/22\approx0.636`.
   * - ``chi_e``
     - *(derived)*
     - Bubble electron factor :math:`1+Z_{\rm He}\,x_{\rm He}=1.2`.
   * - ``chi_e_shell``
     - *(derived)*
     - Shell electron factor :math:`1+Z_{\rm He,shell}\,x_{\rm He}=1.1`.

**Temperatures, dust, and fundamental constants:**

.. list-table::
   :widths: 22 18 20 40
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
     - Ionised shell / HII-region temperature (He singly ionised; see
       ``Z_He_shell``).
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
     - ``6.6743e-8``
     - cm\ :math:`^3`/(g s\ :math:`^2`)
     - Gravitational constant.
   * - ``k_B``
     - ``1.380649e-16``
     - erg/K
     - Boltzmann constant.
   * - ``PISM``
     - ``0``
     - K cm\ :math:`^{-3}`
     - ISM pressure :math:`P/k_B`.


Examples
--------

Minimal parameter file
^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: text
   :caption: minimal.param

    model_name    my_simulation
    mCloud        1e6
    sfe           0.01

Power-law cloud
^^^^^^^^^^^^^^^

.. code-block:: text
   :caption: powerlaw.param

    model_name      powerlaw_test
    path2output     outputs/powerlaw

    mCloud          1e7
    sfe             0.05
    ZCloud          1

    dens_profile    densPL
    densPL_alpha    -1.5
    nCore           1e4
    rCore           0.5
    nISM            1

    stop_t          20
    stop_r          300

Bonnor-Ebert sphere
^^^^^^^^^^^^^^^^^^^

.. code-block:: text
   :caption: bonnor_ebert.param

    model_name      BE_sphere
    path2output     outputs/BE

    mCloud          1e5
    sfe             0.02

    dens_profile    densBE
    densBE_Omega    14.1
    nCore           1e5
    rCore           0.1

For sweep-style parameter files (list values and ``tuple(...)``
directives), see :ref:`sec-running`.
