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
state dictionary (see :ref:`sec-running`, *Snapshot data model*).


File Format
-----------

The canonical parameter schema — the authoritative, fully-commented
list of keys and defaults — lives at ``src/_input/default.param``;
the keywords below mirror it. Worked example files live under
``param/`` (see ``param/simple_cluster.param`` or
``param/cloud_example_PL.param``) and override those defaults.

A parameter file contains one ``keyword    value`` entry per line. A
``#`` starts a comment, either as a whole line or after a value.
Keyword names are case-sensitive and may appear in any order.

Keywords with a default (listed below) are optional; those without a
default are required. A value written as a bracketed list
(``mCloud [1e5, 1e6]``) or through a ``tuple(...)`` directive turns
the file into a sweep — see :ref:`sec-running` for the sweep syntax.

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

Inputs in the parameter file are CGS, extended by :math:`M_\odot`
(mass) and Myr (time). Common per-quantity units: pc for length,
cm\ :math:`^{-3}` for number density, km/s for velocity, K for
temperature. Internally TRINITY works in ``[Msun, pc, Myr]``;
conversion is automatic, driven by the ``# UNIT:`` annotations in
``default.param``. Example annotations:

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
   * - ``coll_r``
     - ``1``
     - pc
     - Radius below which the cloud is considered completely collapsed.

.. note::

   Setting ``stop_r``, ``stop_t``, or ``stop_at_rCloud_nSnap`` to ``None``
   disables that termination condition, allowing the simulation to continue
   until other conditions are met (e.g., shell dissolution, collapse, or
   cloud boundary).


Stellar Feedback (SPS) Parameters
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

TRINITY reads time-evolving stellar feedback (ionizing photon rate,
bolometric and mechanical luminosities, wind and SN momentum injection,
...) from a stellar-population-synthesis (SPS) data file. Two modes are
supported:

* **Bundled default (default)**: leave ``sps_path = def_path``. TRINITY
  loads ``lib/default/sps/starburst99/1e6cluster_default.csv`` — an SB99 grid at
  rotation=1, ZCloud=1 (solar, Z=0.014), BHCUT=120 :math:`M_\odot`,
  mass=:math:`10^6 M_\odot`, exported as CSV with the canonical
  7-column SB99 layout — and applies the ``LEGACY_SB99_COLUMN_MAP``
  positional preset internally. No ``sps_col_*`` declarations are
  required.
* **Custom SPS mode**: set ``sps_path`` to an actual file path, and
  describe the file's column layout with ``sps_col_*`` declarations
  (see the *Custom SPS files* subsection below).

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
     - Full path to an SPS data file. If ``def_path``, resolves to the
       bundled ``lib/default/sps/starburst99/1e6cluster_default.csv``. Otherwise
       points directly at any ``.txt`` or ``.csv`` file whose columns
       are described by the ``sps_col_*`` declarations.
   * - ``sps_refmass``
     - ``def_value``
     - :math:`M_\odot`
     - Reference cluster mass for the feedback scaling
       :math:`f_{\rm mass} = M_{\rm cluster} / {\rm sps\_refmass}`. If
       ``def_value``, copied from ``SB99_mass`` at startup so the
       bundled :math:`10^6 M_\odot` default scales correctly. Override
       when ``sps_path`` points at a file normalized to a different
       reference mass.
   * - ``SB99_mass``
     - ``1e6``
     - :math:`M_\odot`
     - Reference cluster mass the bundled SB99 grid is normalized
       against. Read by ``sps_refmass``'s ``def_value`` fallback; not
       used elsewhere under the bundled default.
   * - ``SB99_rotation``
     - ``1``
     - --
     - Informational under the bundled default
       (``lib/default/sps/starburst99/1e6cluster_default.csv`` is rotation=1).
       **Also used by the non-CIE cooling-table selection regardless
       of ``sps_path``**, so keep it consistent with your data choice.
   * - ``SB99_BHCUT``
     - ``120``
     - :math:`M_\odot`
     - Informational under the bundled default
       (``lib/default/sps/starburst99/1e6cluster_default.csv`` is BHCUT=120). BH
       formation threshold; stars above this ZAMS mass collapse
       directly to BH without SN.


Custom SPS files (``sps_col_*`` declarations)
"""""""""""""""""""""""""""""""""""""""""""""

When ``sps_path`` is set to an explicit file path, the file's column
layout must be described with one ``sps_col_<canonical>`` line per
mapped column. Each line has three whitespace-separated fields after
the key:

.. code-block:: text

    sps_col_<canonical>    <file_column>    <units>    <log|linear>

where:

* ``<canonical>`` is one of the canonical names in the table below.
* ``<file_column>`` is **either** a 0-based integer column index (works
  on any file, with or without a header), **or** a string name matching
  the file's header row (the file must have a header for name lookup to
  resolve).
* ``<units>`` is the declared unit of the column. The alias ``cgs`` is
  accepted as shorthand for each canonical's default cgs unit (see
  table below).
* ``<log|linear>`` declares whether file values are stored as
  :math:`\log_{10}` of the linear value.

The file can be ``.txt`` (whitespace-separated) or ``.csv`` (comma-
separated); the delimiter is sniffed from the first data row. Lines
beginning with ``#`` and blank lines are treated as comments.

**Required canonicals** (loader will not start without them):

* ``t``, ``Lbol``, ``Lmech_W``, ``Qi``, ``pdot_W``
* **either** ``fi`` **or both** ``Li`` **and** ``Ln`` (the latter
  bypasses SB99's hardcoded 13.6 eV ionizing threshold)
* **either** ``Lmech_total`` **or** ``Lmech_SN`` (one of them drives
  the SN pipeline)

**Optional canonicals** (loader derives them when absent):

``Lmech_total``, ``Lmech_SN``, ``pdot_SN``, ``Mdot_SN``, ``v_SN``,
``Li``, ``Ln``.

Per-canonical recognized units:

.. list-table::
   :widths: 20 20 30 30
   :header-rows: 1

   * - Canonical
     - ``cgs`` alias maps to
     - Other accepted ``<units>``
     - AU target (loader output)
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
     - :math:`M_\odot{\rm \cdot pc}^2/{\rm Myr}^3`
   * - ``pdot_W``, ``pdot_SN``
     - ``g*cm/s^2``
     - --
     - :math:`M_\odot{\rm \cdot pc}/{\rm Myr}^2`
   * - ``Mdot_SN``
     - ``g/s``
     - ``Msun/Myr``
     - :math:`M_\odot/{\rm Myr}`
   * - ``v_SN``
     - ``cm/s``
     - ``km/s``, ``pc/Myr``
     - pc/Myr

**Example: headered file with the ``cgs`` alias.** A custom SPS file
``my_sps.txt`` with first line
``time Qi fi Lbol Lmech_total pdot_W Lmech_W`` and numeric data below
in :math:`\log_{10}` cgs (except ``time`` in linear yr and ``fi`` as a
linear fraction):

.. code-block:: text

    sps_path        /absolute/path/to/my_sps.txt

    sps_col_t            time         yr             linear
    sps_col_Qi           Qi           cgs            log
    sps_col_fi           fi           dimensionless  linear
    sps_col_Lbol         Lbol         cgs            log
    sps_col_Lmech_total  Lmech_total  cgs            log
    sps_col_pdot_W       pdot_W       cgs            log
    sps_col_Lmech_W      Lmech_W      cgs            log

**Example: headerless file by integer index.** The same file with no
header row, columns mapped by 0-based index:

.. code-block:: text

    sps_path        /absolute/path/to/my_sps.txt

    sps_col_t            0    yr             linear
    sps_col_Qi           1    cgs            log
    sps_col_fi           2    dimensionless  linear
    sps_col_Lbol         3    cgs            log
    sps_col_Lmech_total  4    cgs            log
    sps_col_pdot_W       5    cgs            log
    sps_col_Lmech_W      6    cgs            log

Indices and header names can be mixed within a single ``.param``: each
``sps_col_*`` line is resolved independently.

If ``sps_col_*`` declarations are missing or inconsistent while
``sps_path`` is set, the loader hard-errors at startup with a fillable
template indicating exactly which canonicals are missing.


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
     - Selects the CIE (T > 10\ :sup:`5.5` K) cooling table. Integer
       presets under ZCloud=1: 1=CLOUDY HII, 2=CLOUDY+grains,
       3=Gnat & Ferland 2012; all bundled under
       ``lib/default/CIE/``. Under ZCloud=0.15 this is ignored
       and the loader auto-pins to
       ``lib/default/CIE/coolingCIE_4_Sutherland-Dopita1993.dat``.
   * - ``path_cooling_nonCIE``
     - ``def_dir``
     - Folder of non-CIE (T < 10\ :sup:`5.5` K) OPIATE/CLOUDY cubes.
       Sentinel ``def_dir`` resolves to ``lib/default/opiate/``.
       Per-age filenames inside follow the OPIATE grammar
       ``opiate_cooling_{rot|norot}_Z{1.00|0.15}_age{a}.dat`` and are
       selected at runtime from ``SB99_rotation`` + ``ZCloud``.
   * - ``path_sps``
     - ``def_dir``
     - SPS data directory anchor. Sentinel ``def_dir`` resolves to
       ``lib/default/sps/``, where the bundled
       ``1e6cluster_default.csv`` lives. Currently informational only —
       ``sps_path``'s ``def_path`` branch resolves directly to the
       bundled CSV. See the *Stellar Feedback (SPS) Parameters* section
       above.


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

For sweep-style parameter files (list values and ``tuple(...)``
directives), see :ref:`sec-running`.
