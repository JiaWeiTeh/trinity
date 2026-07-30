# S1 units & helpers — Lens B (what the code claims)

> ⚠️ **This document may be out of date — verify before trusting it.** It is a
> point-in-time analysis/audit, not a maintained spec; the code moves faster
> than these notes (paths, line numbers, and "what shipped" status drift).
> **Any agent or person reading this: treat it as unverified. Flag that it may
> be stale and re-check each claim, snippet, and line reference against the
> current source before relying on it.**
>
> 🔄 **Living plan — recheck and refine on every visit.** This is an evolving
> strategy doc, not a frozen record. Any agent or person who opens this file
> must, as part of the visit: (1) re-verify the claims and line references above
> against current source; (2) update anything that has drifted; (3) **rethink the
> strategy itself** — if a better ordering, gate, candidate, or experiment
> exists, revise the doc and note what changed and why (date it). Leave it better
> than you found it. **Keep all banner paragraphs at the top of every plan and
> analysis doc.**
>
> 💾 **Persist diagnostics — commit, don't re-run.** The container is ephemeral
> and full/hybr runs cost hours, so any diagnostic worth keeping must be saved as
> a committed artifact under `docs/dev/` (a CSV/table in `docs/dev/data/`, or a
> harness/figure in the relevant `docs/dev/<workstream>/` folder) — never left in
> `/tmp`, the local-only `scratch/`, or an untracked `outputs/`. A future visit must be able to reproduce or compare
> against the numbers **without re-running**; record the exact config + command
> that produced each artifact.
>
> 🔗 **Cross-check the sibling docs — keep the workstream self-consistent.** This file is one of
> several living docs for its workstream (its `PLAN.md`, `FINDINGS.md`, `runs/README.md`, `NOTE_PATCHES.md`,
> and any other notes in the same folder). They drift out of sync *with each other* as fast as they drift
> from the code. Any agent or person editing one MUST, as part of the visit, circle back through the
> siblings and reconcile: if a number, status, claim, or line reference here contradicts a sibling — or a
> sibling has gone stale — fix it (or flag it, dated) so no two docs in the workstream disagree. Never
> update one in isolation.

**Status (2026-07-29):** 📘 raw agent report — provenance for `FINDINGS.md`; unreconciled and unverified on its own.

**Method.** I read only the extracted prose (comments + docstrings) for the five files in slice S1.
I have not seen a single line of implementation. Everything below is a *claim the code makes about
itself*, recorded so another lens can test it against the implementation. Where I say a claim is
"arithmetically consistent", I mean the claim is consistent with other claims and with standard
constant values — **not** that the code does it.

Severity legend used in the JSON: **S1** wrong physics that would silently corrupt results ·
**S2** serious correctness risk · **S3** moderate — contradiction/ambiguity that a reader could
act on wrongly · **S4** minor / documentation-only.

Files in slice:

| file | what it claims to be |
|---|---|
| `trinity/_functions/unit_conversions.py` | frozen conversion constants + safe unit-string parser |
| `trinity/_functions/operations.py` | array search helpers + soundspeed |
| `trinity/_functions/cluster.py` | CPU/allocation detection for sweeps |
| `trinity/_functions/simplify.py` | curve downsampling + error metrics |
| `trinity/_functions/logging_setup.py` | logging configuration |
| `trinity/_functions/extract_example_snapshots.py` | snapshot extraction CLI |

---

## 1. The declared unit system

| # | claim | cite |
|---|---|---|
| 1.1 | "All conversions are to **'Astronomy Units' (AU)**: Mass: Msun (solar masses); Length: pc (parsecs); Time: Myr (megayears)." | `trinity/_functions/unit_conversions.py:3` |
| 1.2 | Constants are **hardcoded, not astropy at runtime**, "for SPEED — no import overhead and no repeated computation". | `trinity/_functions/unit_conversions.py:3` |
| 1.3 | Constants are "derived from astropy.units and **frozen** for safety". Verification against astropy is claimed for accuracy. | `trinity/_functions/unit_conversions.py:3` |
| 1.4 | Repo-wide convention: `import trinity._functions.unit_conversions as cvt`, then use **flat module-level names** (`cvt.cm2pc`, `cvt.pc2cm`, …). | `trinity/_functions/unit_conversions.py:3` |
| 1.5 | Naming convention: `<quantity>_cgs2au` / `<quantity>_au2cgs` for conversion factors; `<NAME>_CGS` for CGS physical constants. | `trinity/_functions/unit_conversions.py:3` |
| 1.6 | `CONV` (CGS→AU), `INV_CONV` (AU→CGS) and `CGS` (physical constants) are frozen dataclass instances and are "the immutable source of truth"; the flat names are **re-exports of their fields**, so `cvt.cm2pc` and `CONV.cm2pc` are *the same number*. | `trinity/_functions/unit_conversions.py:3`, `:246` |
| 1.7 | The flat layer is explicitly **NOT deprecated** — it is "the canonical call-site spelling". | `trinity/_functions/unit_conversions.py:232`, `:246` |
| 1.8 | `ConversionConstants`: "All constants convert **CGS → Astronomy Units [Msun, pc, Myr]**." | `trinity/_functions/unit_conversions.py:59` |
| 1.9 | `InverseConversionConstants`: "Inverse conversions: **Astronomy Units → CGS**". | `trinity/_functions/unit_conversions.py:157` |
| 1.10 | `PhysicalConstantsCGS` holds "fundamental physical constants, **not conversion factors**", "raw constants for calculations in CGS **before** converting to astronomy units". | `trinity/_functions/unit_conversions.py:189`, `:194` |
| 1.11 | `frozen=True` "prevents accidental modification": `CONV.cm2pc = 999` "Raises FrozenInstanceError". | `trinity/_functions/unit_conversions.py:59` |
| 1.12 | `__post_init__` "Verify that all constants are positive (sanity check)." | `trinity/_functions/unit_conversions.py:140` |
| 1.13 | For unit strings coming from `.param` files, use `convert2au("g*cm**2/s**2")`. | `trinity/_functions/unit_conversions.py:3` |
| 1.14 | Module was "Rewritten: January 2026 – Improved safety, accuracy, and maintainability". | `trinity/_functions/unit_conversions.py:3` |

**Direction, stated explicitly in the usage block** (`trinity/_functions/unit_conversions.py:3`) —
this is the anchor for every direction claim below:

```
r_pc = r_cm * cvt.cm2pc      # CGS -> astronomy units   (multiply a cm value to get pc)
r_cm = r_pc * cvt.pc2cm      # astronomy units -> CGS   (multiply a pc value to get cm)
E_au = E_erg * cvt.E_cgs2au
kT   = cvt.K_B_CGS * T       # CGS constants used directly, no conversion
```

---

## 2. Stated unit of every constant, with direction

Every row is "**multiply a value in the FROM unit by this constant to get the TO unit**", because
of the usage block in §1. All rows are inside `ConversionConstants`, whose docstring says the whole
class is CGS→AU.

| line | constant role | FROM | TO | direction claimed |
|---|---|---|---|---|
| `:73`,`:75` | Length; one length constant is documented as `= cm2pc / 1e-5` | cm (and, via the `/1e-5`, km) | pc | CGS → AU |
| `:77` | Time | s | Myr | CGS → AU |
| `:80` | Mass | g | Msun | CGS → AU |
| `:87` | Number density | 1/cm³ | 1/pc³ | CGS → AU |
| `:90` | Photon flux | 1/cm²/s | 1/pc²/Myr | CGS → AU |
| `:93` | Energy | erg | Msun·pc²/Myr² | CGS → AU |
| `:96` | Luminosity | erg/s | Msun·pc²/Myr³ | CGS → AU |
| `:99` | Momentum rate | g·cm/s² | Msun·pc/Myr² | CGS → AU |
| `:102` | Momentum rate derivative | g·cm/s³ | Msun·pc/Myr³ | CGS → AU |
| `:105` | Gravitational constant | cm³/g/s² | pc³/Msun/Myr² | CGS → AU |
| `:108`,`:109` | Velocity | **km/s** | pc/Myr | (not CGS) → AU |
| `:110` | Velocity | cm/s | pc/Myr | CGS → AU |
| `:112` | Force | g·cm/s² | Msun·pc/Myr² | CGS → AU |
| `:115` | Pressure | g/cm/s² | Msun/pc/Myr² | CGS → AU |
| `:118` | Boltzmann constant | erg/K | Msun·pc²/Myr²/K | CGS → AU |
| `:121` | Thermal conduction coefficient | g·cm/s³/K^(7/2) | Msun·pc/Myr³/K^(7/2) | CGS → AU |
| `:124` | Energy density rate | erg/cm³/s | Msun/pc/Myr³ | CGS → AU |
| `:127` | Cooling function | erg·cm³/s | Msun·pc⁵/Myr³ | CGS → AU |
| `:130` | Surface density | g/cm² | Msun/pc² | CGS → AU |
| `:133` | Gravitational potential | cm²/s² | pc²/Myr² | CGS → AU |
| `:136` | Gravitational force per unit mass | cm/s² | pc/Myr² | CGS → AU |

Module-level-only derived constants (explicitly **not** re-exports, `:254`):

| line | constant | FROM | TO | note |
|---|---|---|---|---|
| `:285` | `Pb_au2_KcmInv` | internal [Msun/pc/Myr²] | [K cm⁻³] | "= internal→cgs dyn/cm² via `Pb_au2cgs`, divided by k_B in cgs erg/K"; "≈ 4.6867e+03" |
| `:288` | `Mdot_au2Msunyr` | internal [Msun/Myr] | [Msun/yr] | "(1 Myr = 1e6 yr)" ⇒ factor must be 1e-6 |

Unit **tokens** understood by the parser:

| line | token semantics |
|---|---|
| `:366` | "Base unit conversion map" (cm, g, s, km, erg-components, …) |
| `:373`,`:374` | mean-molecular-weight unit: "dimensionless value × m_H [g] → Msun"; used for `mu_atom`, `mu_ion`, `mu_ion_shell`, `mu_mol`, `mu_convert` |
| `:376` | "Dimensionless units (no conversion needed)" |

**Dimensional cross-checks between the claims themselves** (all internally consistent, no finding):
`:121` g·cm/s³/K^(7/2) is indeed erg s⁻¹ cm⁻¹ K^(-7/2) (Spitzer κ = C·T^{5/2}); `:124` erg cm⁻³ s⁻¹
= g cm⁻¹ s⁻³ ↔ Msun pc⁻¹ Myr⁻³; `:127` erg cm³ s⁻¹ = g cm⁵ s⁻³ ↔ Msun pc⁵ Myr⁻³.

---

## 3. Stated numeric values

| line | quantity | stated value | claim-level check |
|---|---|---|---|
| `trinity/_functions/unit_conversions.py:316` | `convert2au("cm")` | `3.240779289444365e-19` | = 1/pc[cm]; consistent with pc = 3.0856775814913673e18 cm |
| `trinity/_functions/unit_conversions.py:316` | `convert2au("g*cm**2/s**2")` (erg) | `5.260183968837699e-44` | consistent with Msun=1.9884e33 g, pc as above, Myr = 3.15576e13 s |
| `trinity/_functions/unit_conversions.py:316` | `convert2au("km*s**-1")` | `1.022712165045695` | consistent (1 km/s = 1.0227 pc/Myr) |
| `trinity/_functions/unit_conversions.py:75` | a length constant | "= cm2pc / 1e-5" | i.e. ×1e5 ⇒ km→pc if the field is `km2pc` (1 km = 1e5 cm) |
| `trinity/_functions/unit_conversions.py:285` | `Pb_au2_KcmInv` | "≈ 4.6867e+03" | I reproduce 4686.7 from M_sun(IAU2015 nominal)/[pc·Myr²]/k_B(CODATA2018) — **consistent**; the code lens should confirm the *computed* value matches this comment |
| `trinity/_functions/unit_conversions.py:288` | 1 Myr | "1e6 yr" | ⇒ `Mdot_au2Msunyr` = 1e-6 |
| `trinity/_functions/operations.py:190` | ionisation split for μ | `T > 1e4 K` → `mu_ion`; `T <= 1e4 K` → `mu_atom` | boundary `T == 1e4` claimed **neutral** |
| `trinity/_functions/operations.py:86` | bubble outer-edge start temperature | `T_init = 3e4` (unit not stated; K implied) | — |
| `trinity/_functions/cluster.py:3` | oversubscription example | 64-core node, 4-core job → "~31 workers" | 64//2−1 = 31, i.e. the *laptop* formula applied to `os.cpu_count()` |
| `trinity/_functions/simplify.py:239`,`:298` | `_COVERAGE_CHUNKS` | 20 | capped at `nmin − 2` |
| `trinity/_functions/simplify.py:298`,`:581`,`:652` | mandatory-extremum threshold | 5 % of y-range | stated as ">" at `:298`, "≥" at `:652` |
| `trinity/_functions/simplify.py:298` | `nmin` default / floor | default 100; clamped to ≥ 20 | — |
| `trinity/_functions/simplify.py:298` | `grad_inc` default | 1.0 (Menger curvature threshold on rescaled axes) | — |
| `trinity/_functions/simplify.py:298` | `warn_below_r2` default | 0.9 | `None` disables |
| `trinity/_functions/simplify.py:298` | `dedup_tol` default | 1e-6 (relative to per-axis range) | "unless the input has more than ~10⁶ uniformly-sampled points" — consistent with 1/1e-6 |
| `trinity/_functions/simplify.py:760` | `max_rel_err` guard | skip points with `|y_orig| < 1e-30` | — |
| `trinity/_functions/simplify.py:760` | dex intuition | 0.01 dex ≈ 2 %; 0.1 dex ≈ 26 %; 1.0 dex = one decade | 10^0.01=1.023, 10^0.1=1.259 — consistent |
| `trinity/_functions/simplify.py:25` | micro-opt cost model | numpy-scalar fetch ≈ 100 ns, Python float ≈ 10 ns; "roughly halve the runtime" on million-point inputs | unverified perf claim |
| `trinity/_functions/simplify.py:656` | bisection-order equivalence | "verified against the BFS for n ∈ {2, 3, 4, …, 30 000}" | no persisted artifact cited |
| `trinity/_functions/logging_setup.py:122` | log-level integers | DEBUG 10, INFO 20, WARNING 30, ERROR 40, CRITICAL 50 | standard |

---

## 4. Formulas stated in prose

| line | formula as stated |
|---|---|
| `trinity/_functions/operations.py:190` | `c_s = sqrt(gamma_adia · k_B · T / mu)`, with `mu = mu_ion` for T > 1e4 K, `mu = mu_atom` for T ≤ 1e4 K, and explicitly **NOT** `mu_convert` (which is "mass per H nucleus"). Input `T` in K, output `c_s` in **pc/Myr**. |
| `trinity/_functions/unit_conversions.py:285` | `Pb_au2_KcmInv = Pb_au2cgs / k_B[cgs, erg/K]` |
| `trinity/_functions/unit_conversions.py:75` | `<len const> = cm2pc / 1e-5` |
| `trinity/_functions/unit_conversions.py:432` | `total_factor ← Π (base_factor_u)^(e_u)`, with `e_u → −e_u` when the unit is a denominator (`invert=True`, also `:469`) |
| `trinity/_functions/cluster.py:49` | non-SLURM default workers = `max(1, cpu//2 − 1)`; inside SLURM (`SLURM_JOB_ID` set) = full allocation |
| `trinity/_functions/simplify.py:507` | Menger curvature `κ_i = 1 / circumradius(P_{i−1}, P_i, P_{i+1})`; body works from "2× signed area of the triangle" and the three side lengths ⇒ `κ = 4A/(abc) = 2·|cross|/(abc)`; `kappa[i]` corresponds to original index `i+1`; array length `n−2` |
| `trinity/_functions/simplify.py:298` | arc length on rescaled axes `L = Σ sqrt((Δx/range_x)² + (Δy/range_y)²)`, divided into `nmin` equal bins (`:617` "Dividing total arc by nmin gives roughly nmin bins") |
| `trinity/_functions/simplify.py:298` | dedup rule: merge two consecutive samples **iff** `|Δx| ≤ dedup_tol·range_x` **AND** `|Δy| ≤ dedup_tol·range_y` |
| `trinity/_functions/simplify.py:298` | output size = `max(nmin, |mandatory_set|)` (also `:648`, `:721`) |
| `trinity/_functions/simplify.py:298` | coverage chunks = `min(_COVERAGE_CHUNKS, nmin − 2)`; cap inert for `nmin ≥ 22` (`:694`) |
| `trinity/_functions/simplify.py:124` | prominence of a max = descent from `y[p]` until a point more extreme than `y[p]` or the boundary; per-side shoulders from min-RMQ (max candidates) / max-RMQ (min candidates); walk ranges left `[pg+1, pm−1]`, right `[pm+1, ng−1]`, with `pg = −1 → lo = 0`, `ng = n → hi = n−1` (`:178`) |
| `trinity/_functions/simplify.py:25` | `prev_s[i] = max{ j < i : y[j] > y[i] }` (greater case) or `y[j] < y[i]` (less case), `−1` if none; `next_s[i] = min{ j > i : same }`, `n` if none |
| `trinity/_functions/simplify.py:760` | `compression = len(x_orig)/len(x_simp)`; `max_rel_err = max(|residual|/|y_orig|)` skipping `|y_orig| < 1e-30`; log metrics computed against a **log-linear** reconstruction (straight line in `(x, log10 y)`) |

---

## 5. Citations

| line | citation | what is attributed to it |
|---|---|---|
| `trinity/_functions/unit_conversions.py:3` | **astropy.units** | all conversion constants are "derived from astropy.units"; a bottom-of-file test verifies against astropy |
| `trinity/_functions/unit_conversions.py:59` | astropy.units | "Derived using astropy.units (see verification test at bottom)" |
| `trinity/_functions/unit_conversions.py:194` | **CODATA 2018 / IAU 2015 resolutions** | *collectively* attributed to: G, k_B, m_H, m_p, m_e, c, σ_SB, h, elementary charge (esu) |
| `trinity/_functions/simplify.py:87` | **Bender–Farach-Colton** | sparse-table O(1) RMQ construction ("two overlapping blocks of length 2**k") |
| `trinity/_functions/simplify.py:124` | topological persistence / **sublevel-set filtration** | prominence is claimed "equivalent to the persistence of the extremum in the sublevel-set filtration of y" — stated for *both* maxima and minima |
| `trinity/_functions/cluster.py:3` | SLURM on **bwForCluster Helix / bwUniCluster** | motivation for the env-var precedence order |
| `trinity/_functions/simplify.py:867` | **matplotlib log-y rendering** | justification for using a log-linear (not linear) reconstruction in the log metrics |

---

## 6. Contracts

### `unit_conversions.convert2au` (`:316`)
- Input: `unit_string : str or None`. `None` → returns 1 (dimensionless).
- Output: `float`, "Conversion factor to **multiply original value by**" — to astronomy units [Msun, pc, Myr].
- Raises `UnitConversionError` "If unit string contains unrecognized units or invalid syntax" (`:311` "Raised when unit conversion fails").
- Parsing: no `eval()`; `fractions.Fraction` used **for exponents only** (`:316`, `:459`); whitespace stripped (`:359`); split on `*` but not `**` (`:420`); split on `/` but not inside parentheses (`:390`); "Split into numerator and **denominators**" (plural, `:413`) ⇒ `a/b/c` puts both `b` and `c` in the denominator; exponents may be parenthesised, e.g. `"(-7/2)"` (`:457`); underscores allowed in unit names, e.g. `m_H` (`:437`).
- Edge cases **not** documented: empty string (a branch exists at `:362` but no stated behaviour), whitespace-only string, a parenthesised denominator *group* like `a/(b*c)`, repeated units, negative-value handling.

### `operations.find_nearest` (`:20`)
- "finds index idx in array for which array[idx] is closest to value". Input coerced to numpy array (`:23`).
- **Undocumented**: tie behaviour, empty array, NaN, whether the index or the value is returned.

### `operations.find_nearest_lower` (`:31`)
- Returns idx with `array[idx] ≤ value` and closest to value.
- Precondition: "Elements in array **need be** monotonically increasing or decreasing!"
- Monotonicity check is non-strict — "kind of, because includes equal values like `[1,2,3,3,4]`" (`:67`).
- Failure mode: raises (a monotonic error) which is **caught by `get_betadelta`** "as a penalised, retried trial"; logged at DEBUG, not printed, "so this firing is benign per-trial noise that must not spam stdout" (`:37`).
- **Admitted contract breach** at the boundary (`:56`): "when these happen, it means that the returned idx is actually higher than the value instead of the desired lower."

### `operations.find_nearest_higher` (`:147`)
- Returns idx with `array[idx] ≥ value` and closest to value.
- Precondition: "Elements in array **should be** monotonically increasing or decreasing" (softer wording than the `lower` sibling).
- "A shallow, localized numerical non-monotonicity (e.g. a sub-percent single-point spike, or a startup dip in the leading fraction) is tolerated; a deep or sustained-interior inversion still raises `MonotonicError`."
- Direction is determined **from the endpoints**, "robust to a tolerated local spike that would otherwise make the all-pairs `kindof_increasing()` return False" (`:161`).

### `operations._is_monotonic_or_tolerable` (`:102`)
- True if monotonic, **or** non-monotonic only as: (a) an isolated single-point spike — "any depth — a single point cannot be a physical inversion"; or (b) a shallow, localized multi-point wiggle.
- False for a **non-finite** profile and for deep or sustained interior non-monotonicity.
- Tuning knobs named but **valued only in code**: `MONOTONIC_RTOL` ("max relative drawdown treated as numerical noise", `:94`), `BOUNDARY_FRAC` ("leading fraction treated as a startup transient", `:95`), `MAX_SPIKE_LEN` ("longest wrong-direction run treated as an isolated spike", `:96`).
- Internals: "signed step in the intended direction; wrong-direction steps are < 0" (`:116`); "wrong-direction run covers steps `[start, end)`; values `L[start..end]`" (`:130`).

### `operations.get_soundspeed` (`:190`)
- In: `T : float (Units: K)`. Out: `c_s (Units: pc/Myr)`.
- μ selection as in §4. **The unit system of `k_B` and `mu` inside the formula is never stated.**

### `cluster.detect_allocated_cpus` (`:29`) / `get_optimal_workers` (`:49`)
- Returns `(n_cpus, source)`; `source` is a provenance label, e.g. `Workers: 4 (SLURM_CPUS_PER_TASK)`.
- Precedence: `SLURM_CPUS_PER_TASK` → `SLURM_CPUS_ON_NODE` → `os.sched_getaffinity` (Linux only, respects cgroup/cpuset) → `os.cpu_count()` (last resort).
- Rationale: `os.cpu_count()`/`multiprocessing.cpu_count()` report the whole node, not the granted cores.
- `get_optimal_workers`: full allocation inside SLURM (keyed on `SLURM_JOB_ID`); `max(1, cpu//2 − 1)` otherwise.
- Cross-module claim: "each worker spawns a full simulation subprocess; see the `OMP_NUM_THREADS=1` pinning in sweep_runner".

### `simplify._simplify` (`:298`)
- In: `x_arr`, `y_arr` (equal length — else `ValueError`), `nmin=100` (clamped ≥ 20), `grad_inc=1.0`, `warn_below_r2=0.9` (`None` disables), `dedup_tol=1e-6` ("Pass `0` to disable").
- Out: `(x_out, y_out)`, equal length. Size normally `nmin`; may be **larger** when mandatory features exceed the budget.
- Order contract: "Input may be ascending, descending, or non-monotonic in x. Output values are returned in the caller's **original positional order** … For non-monotonic input, output is a thinned subsequence in the input's original order." Restated at `:438` and implemented via `_restore` (`:482`).
- Degenerate input ("all samples stacked at one (x, y) point, or all-NaN") → "falls back to `nmin` uniformly spaced indices" (also `:612`).
- Empty arrays: "Nothing to simplify for empty arrays" (`:429`) — behaviour not documented in the docstring.
- Short/deduplicated input: "If the (deduplicated) array is already short enough, return it in caller order … if dedup collapsed a clump, the caller gets the meaningful subset rather than the duplicate-laden original" (`:492`).
- Priority tiers (`:641`): 1 endpoints, 2 high-prominence extrema (prominence DESC), 3 x-uniform coverage skeleton, 4 remaining merged pool in hierarchical-bisection order. `:702` adds that `idx_dist` (arc-length bin boundaries) is "promoted **ABOVE** bisection_pool", with `|idx_dist| ≈ nmin` "by construction, so taking it in x-order is fine — it never overflows the budget".
- Nesting guarantee: "the subset at any budget N is a superset of the subset at N − 1" (`:298`), restated at `:567` as "once a big dip or spike is in the output at budget N it is also in the output at N+1, N+2, …".
- Rescaling: curvature and arc length operate on `[0,1]`-rescaled axes; "sign-change detection and the post-hoc R² check still operate on **raw** arrays" (`:517`).

### `simplify._x_uniform_coverage_idx` (`:251`)
- In: `x` "Monotonically ascending x-values of the working array"; `pool_idx` sorted indices; `n_chunks` (default `_COVERAGE_CHUNKS`).
- Out: "Sorted, unique indices into `x` (**subset of `pool_idx`**)".
- Undocumented: empty `pool_idx`, `n_chunks ≤ 0`, chunks containing no pool point.

### `simplify._peak_prominences` (`:124`)
- In: `y` 1-D; `idx` indices of local extrema. Out: **non-negative** prominence per index.
- Complexity "O(n log n) total, fully deterministic".
- Negative results from rounding are clamped: "prominence is non-negative by definition" (`:229`).
- Empty-side handling: "if a side is empty (shouldn't happen for real extrema) treat its shoulder as **+inf** so the other side dominates" (`:196`).

### `simplify._simplify_error` (`:760`)
- Returns a dict with exactly: `max_abs_err`, `mean_abs_err`, `rms_err`, `max_rel_err`, `r_squared`, `compression`, `n_orig`, `n_simp`, `log_r_squared`, `log_rms_err`, `log_max_dex_err`, `log_mean_dex_err`.
- All four `log_*` fields are `nan` "if any `y <= 0`" (`:867` "any y ≤ 0").
- "sort the simplified curve internally so this works for descending or non-monotonic inputs too" (`:825`).

### `logging_setup` (`:122`, `:331`, `:368`, `:403`)
- `setup_logging(log_level='INFO', console_output=True, file_output=True, log_file_path=None, log_file_name=None, use_colors=True, format_string=None)`; default format `'%(asctime)s | %(levelname)s | %(name)s | %(message)s'`; auto filename `trinity_YYYYMMDD_HHMMSS.log`; returns the configured **root** logger; existing handlers removed to avoid duplicates (`:242`); colors only if output is a terminal (`:260`); log files plain text.
- `DedupWarningFilter` (`:80`): passes the FIRST occurrence of each unique rendered message at `min_level` and above and drops exact repeats; "Only *identical rendered text* collapses"; "Attach ONE instance per handler"; "State is per-process, so it resets every run/task — no cross-run leakage"; "malformed %-args -> never suppress" (`:104`). Named example warnings: "a super-critical Bonnor-Ebert sphere, `nEdge < nISM`, `rCloud > rCloud_max`".
- `setup_logging_from_params` (`:403`) maps `params['log_level'|'log_console'|'log_file'|'log_colors'|'path2output'].value` onto `setup_logging(log_level, console_output, file_output, use_colors, log_file_path)`.
- Third-party loggers are forced to INFO when TRINITY runs at DEBUG (`:310`).

### `extract_example_snapshots` (`:3`)
- Input: a folder containing `dictionary.jsonl`. Output: 6 single-snapshot `.jsonl` files into `outputs/mockOutput/<foldername>/`: `1_begin`, `2_energy`, `3_implicit`, `4_transition`, `5_momentum`, `6_final`.
- Phase-name claims: `'energy'`, `'implicit'`, `'transition'`, `'momentum'`.
- Selection rule: "the **second** snapshot of that phase is chosen for stability. If the second snapshot has already moved on to the next phase or marks termination, the **first** snapshot of the phase is used instead. If a phase never appears, that file is skipped **with a warning**."
- CLI: `python -m trinity._functions.extract_example_snapshots -F <folder>`.

---

## 7. Admissions of known debt (verbatim triggers)

| line | admission |
|---|---|
| `trinity/_functions/operations.py:56` | "Notes: boundary conditions, just in case. … **Not quite sure what to do with that for now**, but this part of the code **shouldnt need to run anyway**." |
| `trinity/_functions/operations.py:175` | identical text repeated verbatim in `find_nearest_higher` |
| `trinity/_functions/operations.py:79` | "**RETAINED FALLBACK**: … so this guard **may become unused by production**. It is kept deliberately … **do not remove it as 'dead code'**." |
| `trinity/_functions/operations.py:67` | "**kind of**, because includes equal values like `[1,2,3,3,4]`" |
| `trinity/_functions/operations.py:147` | "**should be** monotonically increasing or decreasing" (vs `:31` "**need be** … !") |
| `trinity/_functions/simplify.py:196` | "if a side is empty (**shouldn't happen** for real extrema) …" |
| `trinity/_functions/simplify.py:298` | the whole routine is labelled "**Heuristic** downsampling"; module docstring `:3` says "Heuristic downsampling" |
| `trinity/_functions/simplify.py:25` | "These are **micro-optimisations** but they roughly halve the runtime …"; "Output is **byte-identical** to the straightforward numpy-indexed version." (assertion, no artifact) |
| `trinity/_functions/simplify.py:656` | "The traversal is **byte-identical** to the queue version; verified against the BFS for n ∈ {2, …, 30 000}." (no artifact) |
| `trinity/_functions/simplify.py:617` | "Dividing total arc by nmin gives **roughly** nmin bins" |
| `trinity/_functions/simplify.py:239` | "**Internal constant** — the public `_simplify` signature is unchanged." |
| `trinity/_functions/unit_conversions.py:3` | "We use hardcoded constants (not astropy at runtime) for SPEED … **However**, we protect against accidental modification and provide verification against astropy for accuracy." |
| `trinity/_functions/unit_conversions.py:560` | "Test 6: Verify against astropy (**if available**)" |
| `trinity/_functions/cluster.py:3` | "**Why not just `os.cpu_count()`?**" — the whole module is a workaround for a platform limitation |

---

## 8. Flags

### 8.1 Prose contradicting prose

1. **The tolerance rule for non-monotonic arrays is stated three incompatible ways.**
   `operations.py:91`: "We tolerate only non-monotonicity that is **both** shallow (relative drawdown ≤ MONOTONIC_RTOL) **and** localized …". `operations.py:132`: "isolated single point: a numerical glitch, never a physical inversion → **tolerate regardless of depth**" (echoed by the docstring at `:102`, "any depth"). `operations.py:147`: "a **sub-percent** single-point spike … is tolerated". Depth-exempt vs depth-limited cannot both hold. `:96` also describes `MAX_SPIKE_LEN` as the "longest wrong-direction **run**", i.e. multi-point, while `:132` exempts only a *single* point.

2. **`find_nearest_higher`'s boundary note is a verbatim copy of `find_nearest_lower`'s** and says the returned idx is "actually **higher** than the value **instead of the desired lower**" (`:175`) — for `find_nearest_higher`, higher *is* the desired side. One of the two boundary branches is described by prose that cannot be about it.

3. **`find_nearest_lower` documents a contract it admits it can violate**: contract "array[idx] **smaller or equal** to value" (`:31`) vs "the returned idx is actually higher than the value instead of the desired lower" (`:56`).

4. **`ConversionConstants` claims "All constants convert CGS → Astronomy Units"** (`:59`) but the class contains a **km/s → pc/Myr** entry (`:109`) and a length constant derived as `cm2pc / 1e-5` (`:75`, i.e. km→pc). km and km/s are not CGS.

5. **`_simplify` claims sorting by x is harmless, and separately claims non-monotonic input is supported.** `:438`: "the rest of the algorithm is sequence-based (curvature on triplets, sign changes, cumulative arc length, peak persistence) and **is unaffected by the temporary reordering**". For a genuinely non-monotonic x (explicitly allowed by the contract at `:298`), sorting is a permutation that changes the point sequence, hence changes every one of those four sequence-based quantities. The claim is only defensible for ascending/descending input.

6. **`_simplify`'s nesting guarantee vs its own nmin-dependence.** `:298` item 5: "the subset at any budget N is a superset of the subset at N − 1". But the same docstring makes the arc-length bins depend on `nmin` (item 3: total arc "divided into `nmin` equal bins") and the coverage skeleton depend on `nmin` (item 4b cap at `nmin − 2`), and `:702` promotes those `nmin`-dependent bin boundaries **above** the bisection pool with `|idx_dist| ≈ nmin`. Changing N therefore changes most of the selected set.

7. **Priority order documented with 4 tiers at `:641`, 5 tiers at `:702`.** The `:641` list ends at "remaining merged-pool points, in hierarchical-bisection order"; `:702` inserts `idx_dist` above the bisection pool.

8. **`_peak_prominences` empty-side sentinel.** `:196` (inside the **MAX**-candidate block, `:170`) says "treat its shoulder as **+inf** so the other side dominates". For a maximum the shoulders come from a **min**-RMQ and the key col is the *higher* of the two shoulders, so a `+inf` shoulder makes the **empty** side dominate and drives the prominence to −inf, which `:229`'s "clamp tiny negative values" would then silently turn into 0. `+inf` is the correct sentinel for the **MIN** branch (`:203`, max-RMQ), not the MAX branch.

9. **dedup rule: "both" vs "OR".** `:298` "merged only when ***both*** `|Δx| ≤ …` and `|Δy| ≤ …`" vs `:462` "The **OR-on-Δ rule**". These reconcile by De Morgan (keep iff `|Δx| > tol` OR `|Δy| > tol`), but in a codebase whose declared bug class is unit/logic slips, the two labels sit 160 lines apart with no cross-reference.

10. **"Pass `0` to disable" dedup** (`:298`) vs the stated rule: at `dedup_tol = 0` the rule still merges consecutive samples with `Δx == 0 and Δy == 0`, i.e. exact duplicates are still folded. "Disable" is only true if the code special-cases 0.

11. **`nmin ≥ 20` clamp rationale is arithmetically wrong as written.** `:298` "Clamped to >= 20 (matches the coverage-skeleton chunk count **so endpoints + coverage always fit inside the budget**)" and `:498` "Enforce a floor of 20 samples (matches `_COVERAGE_CHUNKS`) so the algorithm has enough budget for **both endpoints and a meaningful coverage skeleton**". 2 endpoints + 20 chunks = 22 > 20; fitting is actually achieved by the separate `nmin − 2` cap (`:694`), not by the floor.

12. **5 % threshold strictness**: "prominence **exceeds** 5 %" (`:298`) vs "prominence **≥** 5 % of the y-range" (`:652`).

13. **Output can be shorter than `nmin`** (`:492`, deduplicated/short input) but the `nmin` contract only allows "normally `nmin`… may be **larger**" (`:298`).

14. **Docstring numbering vs body numbering** in `_simplify`: docstring calls arc-length sampling "3" (`:298`), the body calls it "Strategy **2**" (`:596`, and `:515` "the cumulative arc-length step (Strategy 2 below)"). Sign-change detection is docstring #2 but has no body "Strategy" header.

### 8.2 Constant whose stated unit conflicts with its stated direction
None found: every entry in §2 is dimensionally self-consistent with the CGS→AU direction anchored at `:3`. The only mismatches are the *category* mismatches in flag 4 (km, km/s inside a class documented as CGS-only).

### 8.3 Same quantity documented two ways
- **Momentum rate** (`:99`, g·cm/s² → Msun·pc/Myr²) and **Force** (`:112`, g·cm/s² → Msun·pc/Myr²) are the identical conversion under two names; they must be numerically equal, and any drift between them is invisible to a reader.
- `Pb_au2_KcmInv` (`:285`) is documented both as a stand-alone "internal → K cm⁻³" factor and as a composite "`Pb_au2cgs` / k_B[cgs]".
- μ-related mass is documented as "mean molecular weight unit: dimensionless × m_H [g] → Msun" (`:373`) in one file and as "mean mass per particle" / "mass per H nucleus" (`operations.py:190`) in another; only the second says which of the two physical definitions applies to which symbol.

### 8.4 Claims too vague to check from prose
- `MONOTONIC_RTOL`, `BOUNDARY_FRAC`, `MAX_SPIKE_LEN` (`operations.py:94`–`:96`) are described qualitatively with **no values**, so "sub-percent" (`:147`) and "shallow" (`:91`) cannot be checked against them from prose.
- `get_soundspeed` (`:190`) states the output unit (pc/Myr) and the input unit (K) but **not** the unit system of `k_B` or `mu` in the formula — the same expression yields cm/s in CGS. §2 + `:373` together imply μ must be in **Msun** and `k_B` must be the **AU-converted** constant (`:118`), not `K_B_CGS`.
- `T_init = 3e4` (`operations.py:86`) carries no unit.
- "Values from CODATA 2018 / IAU 2015 resolutions" (`:194`) is attributed to the whole block, including `m_H` (not a CODATA-tabulated constant) and Msun/pc/Myr-adjacent choices; the module docstring separately says "derived from astropy.units" (`:3`).
- "Verification against astropy for accuracy" (`:3`, `:59`) points at a suite that runs only under `python unit_conversions.py` (`:481`) and is itself conditional — "Test 6: Verify against astropy (**if available**)" (`:560`).
- Two "**byte-identical**" equivalence claims (`simplify.py:25`, `:656`) cite no committed harness, CSV, or figure.
- `logging_setup.py:122` tells the reader to call `setup_logging` "once at the start of your simulation (**in main.py**)", and `:368` uses `'trinity.phase1_energy.run_energy_phase'` as a live logger name — both are checkable existence claims that may be stale.

### 8.5 Claims outside the units domain worth handing on
- `operations.py:37` asserts a cross-module contract: the monotonicity raise "is caught by the beta-delta trial wrapper (`get_betadelta`) as a penalised, retried trial". If that catch ever narrows, the DEBUG-level logging becomes a silent failure.
- `cluster.py:49` keys the "use the **full** allocation" branch on `SLURM_JOB_ID`, while `:3` keys detection on `SLURM_CPUS_PER_TASK` / `SLURM_CPUS_ON_NODE`. A SLURM job where neither CPU var is exported falls through to `sched_getaffinity`/`os.cpu_count()` **and** takes the full-allocation branch — i.e. potentially the whole node, the exact failure the module says it prevents (mitigated only where cgroup/cpuset confinement makes `sched_getaffinity` correct).
- `logging_setup.py:80`: the dedup filter is documented as deliberately collapsing repeated physics warnings (`rCloud > rCloud_max`, `nEdge < nISM`, super-critical Bonnor-Ebert) to one line, with no stated suppressed-count report; and the per-process "seen" state has no stated bound for messages that embed changing values.
- `unit_conversions.py:254`: `Pb_au2_KcmInv` and `Mdot_au2Msunyr` are declared module-level-only "original definitions", so by the module's own description they are **outside** both the `frozen=True` protection (`:59`) and the positivity check (`:140`).
- `unit_conversions.py:3`: the abbreviation "AU" is used for the internal Msun/pc/Myr system, colliding with the standard astronomical unit (au ≈ 1.496e13 cm); it propagates into every public name (`cm2pc`… `convert2au`, `_cgs2au`, `_au2cgs`).

---

```json
[
  {
    "id": "S1-B-01",
    "file": "trinity/_functions/operations.py",
    "line": 56,
    "class": "silent-failure",
    "severity": "S3",
    "claim": "find_nearest_lower's contract (:31) is 'array[idx] smaller or equal to value and closest to value', but the boundary-condition comment admits 'when these happen, it means that the returned idx is actually higher than the value instead of the desired lower', followed by 'Not quite sure what to do with that for now' and 'this part of the code shouldnt need to run anyway'.",
    "evidence": "Docstring trinity/_functions/operations.py:31 vs comment trinity/_functions/operations.py:56.",
    "expected": "Either the boundary branch clamps/raises so the postcondition array[idx] <= value always holds, or the docstring states the exception. A 'lower' lookup that silently returns a higher bracket biases any table interpolation built on it.",
    "failure_scenario": "value falls at or outside an array endpoint; the helper returns the bracket above value; a cooling/SPS table lookup then interpolates from the wrong side with no error raised.",
    "repro": "Call find_nearest_lower on a monotonically increasing array with value below array[0] and with value above array[-1]; assert array[idx] <= value in both cases.",
    "confidence": "high"
  },
  {
    "id": "S1-B-02",
    "file": "trinity/_functions/operations.py",
    "line": 175,
    "class": "other",
    "severity": "S3",
    "claim": "find_nearest_higher's boundary note is a verbatim copy of find_nearest_lower's and states 'the returned idx is actually higher than the value instead of the desired lower' - but for find_nearest_higher, higher IS the desired side, so the prose cannot describe this function's branch.",
    "evidence": "trinity/_functions/operations.py:175 duplicates trinity/_functions/operations.py:56 word for word, including 'instead of the desired lower' and 'Not quite sure what to do with that for now'.",
    "expected": "The mirrored function's boundary branch should be described (and tested) in its own terms: the failure mode there is returning an index BELOW value.",
    "failure_scenario": "The copy-paste suggests the mirrored branch was never re-derived; if the index arithmetic was copied too, find_nearest_higher may return a lower bracket at the array edges without raising.",
    "repro": "Call find_nearest_higher with value above array[-1] and below array[0] on both increasing and decreasing arrays; assert array[idx] >= value.",
    "confidence": "high"
  },
  {
    "id": "S1-B-03",
    "file": "trinity/_functions/operations.py",
    "line": 91,
    "class": "numerical",
    "severity": "S3",
    "claim": "The tolerated-non-monotonicity rule is stated three incompatible ways: ':91' says tolerance requires BOTH shallow (relative drawdown <= MONOTONIC_RTOL) AND localized; ':132' and the docstring ':102' say an isolated single point is tolerated 'regardless of depth' / 'any depth'; ':147' says the tolerated case is a 'sub-percent single-point spike'. ':96' further describes MAX_SPIKE_LEN as the longest wrong-direction RUN, i.e. multi-point.",
    "evidence": "trinity/_functions/operations.py:91, :96, :102, :132, :147.",
    "expected": "One rule. Either single-point spikes are depth-exempt (then ':91' and ':147' are wrong) or they are bounded by MONOTONIC_RTOL (then ':102' and ':132' are wrong).",
    "failure_scenario": "A deep single-point inversion in T_array - which could be a dead-integrator artifact rather than dense-output noise - is accepted as 'tolerable' by the depth-exempt path, and the directional search proceeds on a corrupt temperature profile instead of raising MonotonicError.",
    "repro": "Build an increasing array with one interior point dropped by 50%; call _is_monotonic_or_tolerable and find_nearest_higher; check against each of the three documented rules.",
    "confidence": "high"
  },
  {
    "id": "S1-B-04",
    "file": "trinity/_functions/operations.py",
    "line": 190,
    "class": "units",
    "severity": "S3",
    "claim": "get_soundspeed documents c_s = sqrt(gamma_adia * k_B * T / mu) with T in K and the return value in pc/Myr, but never states the unit system of k_B or mu inside the formula. The identical expression with K_B_CGS (erg/K) and mu in grams returns cm/s, not pc/Myr.",
    "evidence": "trinity/_functions/operations.py:190. Cross-reference: unit_conversions.py:118 defines an AU Boltzmann constant (erg/K -> Msun*pc^2/Myr^2/K) and unit_conversions.py:373 says mu parameters are parsed as dimensionless * m_H[g] -> Msun.",
    "expected": "For the documented pc/Myr output, k_B must be the AU-converted constant and mu must be in Msun. The docstring should say so; a code lens should confirm the implementation uses the AU k_B, not cvt.K_B_CGS.",
    "failure_scenario": "If the CGS Boltzmann constant is used with mu in Msun (or vice versa), c_s is wrong by a fixed factor and every soundspeed-derived quantity (shell/bubble pressure balance, Mach numbers, phase transitions) is silently rescaled.",
    "repro": "Call get_soundspeed(1e4) and compare against sqrt(gamma*k_B_AU*1e4/mu_atom_Msun) in pc/Myr; separately check the value is ~10-15 pc/Myr scale for 1e4 K ionised gas, not ~1e6.",
    "confidence": "medium"
  },
  {
    "id": "S1-B-05",
    "file": "trinity/_functions/operations.py",
    "line": 161,
    "class": "divergence",
    "severity": "S3",
    "claim": "The two sibling lookups handle monotonicity differently by design: the tolerant check is documented as 'Tolerant monotonicity for find_nearest_higher' (:78) only, and find_nearest_higher determines direction 'use endpoints: robust to a tolerated local spike that would otherwise make the all-pairs kindof_increasing() return False' (:161), while find_nearest_lower has no such note (:44).",
    "evidence": "trinity/_functions/operations.py:44, :78, :161.",
    "expected": "Either both siblings tolerate the same class of numerical non-monotonicity and detect direction the same way, or the asymmetry is documented as intentional with the reason.",
    "failure_scenario": "The same T_array is accepted by find_nearest_higher and rejected (MonotonicError, then swallowed as a penalised beta-delta trial) by find_nearest_lower, making the trial outcome depend on which helper a code path happens to call.",
    "repro": "Feed an array with a single tolerated spike to both find_nearest_lower and find_nearest_higher and compare raise/no-raise and the chosen direction.",
    "confidence": "medium"
  },
  {
    "id": "S1-B-06",
    "file": "trinity/_functions/operations.py",
    "line": 20,
    "class": "other",
    "severity": "S4",
    "claim": "find_nearest is documented only as 'finds index idx in array for which array[idx] is closest to value'. Ties, empty input, and NaN entries are not specified, and the docstring does not say whether the index or the value is returned (the comments at :25/:27 say '# index' then '# return').",
    "evidence": "trinity/_functions/operations.py:20, :25, :27.",
    "expected": "Stated tie-breaking (lower index? lower value?), stated behaviour for empty arrays and NaN.",
    "failure_scenario": "A caller relying on lowest-index tie-breaking gets the other bracket after a data change; an empty array raises an opaque numpy error deep in a solver.",
    "repro": "find_nearest(np.array([1.0, 3.0]), 2.0) - which index? find_nearest(np.array([]), 1.0) - what happens?",
    "confidence": "high"
  },
  {
    "id": "S1-B-07",
    "file": "trinity/_functions/operations.py",
    "line": 79,
    "class": "deadcode",
    "severity": "S4",
    "claim": "Declared debt: 'RETAINED FALLBACK: the bubble-luminosity solver is moving to a solve_ivp event-based regime split that does not call find_nearest_higher, so this guard may become unused by production. It is kept deliberately ... do not remove it as dead code.'",
    "evidence": "trinity/_functions/operations.py:79.",
    "expected": "If production no longer calls it, the tolerance constants and the three-way-inconsistent rule (S1-B-03) are untested by any real run; the retained path needs its own pytest coverage or an explicit statement that it is only exercised by tests.",
    "failure_scenario": "The tolerance logic silently rots (nobody notices the rule contradiction) and then a grid-based fallback path re-activates it.",
    "repro": "grep the package for find_nearest_higher call sites reachable from run.py.",
    "confidence": "high"
  },
  {
    "id": "S1-B-08",
    "file": "trinity/_functions/unit_conversions.py",
    "line": 59,
    "class": "units",
    "severity": "S4",
    "claim": "ConversionConstants' docstring says 'All constants convert CGS -> Astronomy Units [Msun, pc, Myr]', but the class contains a km/s -> pc/Myr entry (:109) and a length constant defined as 'cm2pc / 1e-5' (:75), i.e. km -> pc. km and km/s are not CGS units.",
    "evidence": "trinity/_functions/unit_conversions.py:59 vs :75, :108, :109.",
    "expected": "Either the class docstring says 'CGS and common astronomical input units -> AU', or the km entries move out. A reader trusting 'All constants convert CGS' could apply the km/s factor to a cm/s value and be wrong by 1e5.",
    "failure_scenario": "A call site passes a cm/s velocity through the km/s->pc/Myr factor (or vice versa): velocity off by 1e5, silently.",
    "repro": "Check that the km/s constant equals the cm/s constant * 1e5, and that no call site applies the km/s factor to a CGS velocity.",
    "confidence": "high"
  },
  {
    "id": "S1-B-09",
    "file": "trinity/_functions/unit_conversions.py",
    "line": 99,
    "class": "units",
    "severity": "S4",
    "claim": "Two constants document the identical conversion under different names: ':99' 'Momentum rate: g*cm/s^2 -> Msun*pc/Myr^2' and ':112' 'Force: g*cm/s^2 -> Msun*pc/Myr^2'.",
    "evidence": "trinity/_functions/unit_conversions.py:99 and :112.",
    "expected": "The two hardcoded values must be exactly equal. If they differ in any digit, one call site is wrong and nothing in the module would detect it (the __post_init__ check at :140 only verifies positivity).",
    "failure_scenario": "A typo in one of the two literals gives force-budget terms and momentum-rate terms slightly different scalings - a classic force-budget bug that no positivity check catches.",
    "repro": "assert CONV.<force field> == CONV.<momentum rate field> exactly (and likewise for the corresponding INV_CONV fields).",
    "confidence": "high"
  },
  {
    "id": "S1-B-10",
    "file": "trinity/_functions/unit_conversions.py",
    "line": 194,
    "class": "citation",
    "severity": "S4",
    "claim": "'Values from CODATA 2018 / IAU 2015 resolutions' is attributed collectively to G, k_B, m_H, m_p, m_e, c, sigma_SB, h and the elementary charge in esu. m_H (hydrogen atom mass) is not a CODATA-tabulated constant; the module docstring (:3) separately says the constants are 'derived from astropy.units'.",
    "evidence": "trinity/_functions/unit_conversions.py:3, :194, :207.",
    "expected": "Per-constant provenance, especially for m_H (m_p + m_e - binding energy? astropy's u.u * 1.008? a rounded 1.6726e-24 = m_p?), because m_H sets the mu -> Msun conversion at :373 and therefore every mass density in the code.",
    "failure_scenario": "m_H silently set to the proton mass (0.054% low) or to 1 amu propagates a small systematic into every n<->rho conversion; the astropy verification test only runs under __main__ and only 'if available'.",
    "repro": "Compare M_H_CGS against astropy.constants m_p + m_e and against u.u.to('g'); assert which one the comment means.",
    "confidence": "medium"
  },
  {
    "id": "S1-B-11",
    "file": "trinity/_functions/unit_conversions.py",
    "line": 480,
    "class": "other",
    "severity": "S4",
    "claim": "The module's safety argument is 'hardcoded for speed BUT verified against astropy for accuracy' (:3, :59, 'See test suite at bottom for accuracy verification'), yet the suite is headed 'Test suite (run with: python unit_conversions.py)' (:481) and the astropy check is 'Test 6: Verify against astropy (if available)' (:560).",
    "evidence": "trinity/_functions/unit_conversions.py:3, :59, :481, :560.",
    "expected": "The accuracy claim should be backed by a pytest test that fails when a constant drifts from astropy, not by a __main__ block whose only accuracy check is skipped when astropy is missing.",
    "failure_scenario": "A hand-edited constant drifts from astropy and CI stays green, because the only comparison lives in a manually-invoked __main__ block.",
    "repro": "Check whether any test/test_*.py imports astropy and asserts the CONV/INV_CONV/CGS values; run pytest with astropy uninstalled and see whether coverage of the constants changes.",
    "confidence": "medium"
  },
  {
    "id": "S1-B-12",
    "file": "trinity/_functions/unit_conversions.py",
    "line": 316,
    "class": "other",
    "severity": "S4",
    "claim": "convert2au documents only two input contracts - None returns 1, unrecognised/invalid input raises UnitConversionError - but the body has an unstated empty-string branch (:362) and the parser description leaves several forms unspecified: a parenthesised denominator GROUP such as 'a/(b*c)' (parentheses are documented at :457 only for exponents, while :390 splits on / outside parentheses), whitespace-only strings, and repeated units.",
    "evidence": "trinity/_functions/unit_conversions.py:316, :359, :362, :390, :413, :457.",
    "expected": "Documented behaviour for '' (1 or raise?), and either support or an explicit rejection of grouped denominators - '.param' unit strings are a trust boundary.",
    "failure_scenario": "A .param unit string like 'erg/(cm**3*s)' parses to a factor that silently omits or mis-signs the 's' exponent, so a parameter enters the run off by many orders of magnitude with no error.",
    "repro": "convert2au(''), convert2au('   '), convert2au('g/(cm*s**2)') vs convert2au('g/cm/s**2') - the last two must agree.",
    "confidence": "medium"
  },
  {
    "id": "S1-B-13",
    "file": "trinity/_functions/unit_conversions.py",
    "line": 254,
    "class": "state",
    "severity": "S4",
    "claim": "'Pb_au2_KcmInv and Mdot_au2Msunyr below are original definitions (derived constants that only exist at module level), not re-exports.' By the module's own description these two therefore sit outside the frozen=True protection (:59) and outside the __post_init__ positivity check (:140) that the module advertises as its safety mechanism.",
    "evidence": "trinity/_functions/unit_conversions.py:59, :140, :249, :254, :285, :288.",
    "expected": "Either these live in a frozen container too, or the module docstring's 'we protect against accidental modification' is qualified to exclude them.",
    "failure_scenario": "Code assigns cvt.Pb_au2_KcmInv (module attribute, mutable) and every later pressure diagnostic in the process silently uses the altered value - exactly the failure frozen=True was introduced to prevent.",
    "repro": "In a REPL: import trinity._functions.unit_conversions as cvt; cvt.Pb_au2_KcmInv = 1.0 - does it raise?",
    "confidence": "high"
  },
  {
    "id": "S1-B-14",
    "file": "trinity/_functions/unit_conversions.py",
    "line": 3,
    "class": "units",
    "severity": "S4",
    "claim": "The module names its internal Msun/pc/Myr system 'Astronomy Units (AU)' and propagates 'au' through the public API (convert2au, _cgs2au, _au2cgs, Pb_au2cgs, Mdot_au2Msunyr). 'AU'/'au' is the standard symbol for the astronomical unit (1.495978707e13 cm).",
    "evidence": "trinity/_functions/unit_conversions.py:3, :246, :285, :288, :316.",
    "expected": "In a codebase whose declared recurring bug class is units, the internal system should not be abbreviated with a symbol that already denotes a length. At minimum the module docstring should call out the collision explicitly.",
    "failure_scenario": "A contributor reads 'convert to au' as 'convert to astronomical units' and applies or omits a 1.496e13 factor.",
    "repro": "",
    "confidence": "high"
  },
  {
    "id": "S1-B-15",
    "file": "trinity/_functions/simplify.py",
    "line": 438,
    "class": "state",
    "severity": "S3",
    "claim": "'Sort by x and work on an ascending copy ... the rest of the algorithm is sequence-based (curvature on triplets, sign changes, cumulative arc length, peak persistence) and is unaffected by the temporary reordering.' The same function's contract (:298) explicitly accepts non-monotonic x: 'Input may be ascending, descending, or non-monotonic in x.'",
    "evidence": "trinity/_functions/simplify.py:298 (Input/output contract) vs :438.",
    "expected": "Sorting is order-preserving-up-to-reversal only for monotonic input. For non-monotonic x the sort is a permutation that changes the point sequence, so curvature triplets, sign changes, arc length and persistence are all computed on a different curve than the caller supplied. Either the reordering claim is scoped to monotonic input, or non-monotonic support is withdrawn.",
    "failure_scenario": "A non-monotonic trajectory (e.g. a quantity plotted against a variable that reverses, or r(t) with a re-collapse) is thinned using features of a scrambled curve: real bends are missed and spurious ones introduced, while the output still looks plausible because ordering is restored on the way out.",
    "repro": "Simplify a curve whose x goes up then down (e.g. x = concatenate(linspace(0,1,500), linspace(1,0,500))) with a sharp feature only on the return leg; check whether that feature survives.",
    "confidence": "high"
  },
  {
    "id": "S1-B-16",
    "file": "trinity/_functions/simplify.py",
    "line": 196,
    "class": "sign",
    "severity": "S3",
    "claim": "In the MAX-candidate block (:170, shoulders from a min-RMQ) the comment says 'if a side is empty (shouldn't happen for real extrema) treat its shoulder as +inf so the other side dominates'. Under the prominence definition given at :124 (descend until a point more extreme than y[p]), the key col of a maximum is the HIGHER of the two shoulders, so a +inf shoulder makes the empty side dominate and yields prominence = y[p] - inf.",
    "evidence": "trinity/_functions/simplify.py:124 (definition), :170, :178, :196, :203 (MIN mirror), :229 (negative clamp).",
    "expected": "For a max candidate the empty-side sentinel should be -inf (or the side excluded) so the other side dominates; +inf is the correct sentinel only for the MIN branch (:203, max-RMQ). If +inf is used for maxima, the clamp at :229 ('clamp tiny negative values') would convert the resulting -inf/negative prominence to 0.",
    "failure_scenario": "An extremum adjacent to an array boundary gets prominence 0, so it never enters the mandatory set, and a genuine boundary feature (e.g. the first sharp bend after a phase transition) is silently dropped from the simplified output.",
    "repro": "Call _peak_prominences on y with a true maximum at index 1 (so the left walk range [pg+1, pm-1] can be empty) and check the returned prominence against the right-side descent.",
    "confidence": "medium"
  },
  {
    "id": "S1-B-17",
    "file": "trinity/_functions/simplify.py",
    "line": 529,
    "class": "sign",
    "severity": "S3",
    "claim": "The Menger curvature block computes '2x SIGNED area of the triangle formed by the triplet' (:529), then 'Menger curvature, len n-2' (:540), then 'Keep interior indices where curvature exceeds the threshold' kappa > grad_inc (:542, :298). Menger curvature is 1/circumradius = 4A/(abc) with A the unsigned area, i.e. 2*|cross|/(abc).",
    "evidence": "trinity/_functions/simplify.py:298 (kappa = reciprocal circumradius, kept where kappa > grad_inc), :507, :529, :532, :538, :540, :542.",
    "expected": "Two things the code lens should check: (a) the signed cross product is absolute-valued before the threshold test - otherwise only bends of one handedness are detected and the opposite-sign bends (equally sharp) are never kept; (b) the coefficient is 2*|cross|/(abc) when 'cross' is twice the area, not |cross|/(abc) or 4*|cross|/(abc) - a factor-2 error rescales the meaning of the documented grad_inc=1.0 default.",
    "failure_scenario": "Sign not stripped: every concave-down shock is retained and every concave-up one dropped (or vice versa), so the simplified curve systematically loses one class of feature. Wrong factor: grad_inc=1.0 selects 2x too many or too few bend points than the docstring implies.",
    "repro": "Feed a symmetric curve with one upward and one downward kink of equal sharpness; both should be retained. Separately compare kappa for three points on a circle of known radius R against 1/R.",
    "confidence": "medium"
  },
  {
    "id": "S1-B-18",
    "file": "trinity/_functions/simplify.py",
    "line": 298,
    "class": "numerical",
    "severity": "S3",
    "claim": "'Remaining slots are filled in hierarchical-bisection order (endpoints -> midpoint -> quartiles -> ...) so the subset at any budget N is a superset of the subset at N-1' (:298 item 5; restated at :567). The same docstring makes two selection stages depend on nmin: arc-length sampling divides total arc into 'nmin equal bins' (item 3, :617) and the coverage skeleton is capped at 'nmin - 2' chunks (item 4b, :694); and :702 promotes the nmin-dependent arc-length boundaries ABOVE the bisection pool, with '|idx_dist| ~= nmin by construction'.",
    "evidence": "trinity/_functions/simplify.py:298, :567, :617, :694, :702, :712.",
    "expected": "The nesting guarantee can hold only for the part of the selection that is nmin-independent. As documented, changing nmin re-bins the arc length and re-caps the coverage skeleton, so output(N) is not generally a superset of output(N-1).",
    "failure_scenario": "The stated anti-flicker property is relied on (e.g. comparing two runs written at different nmin, or a plot regenerated at a higher budget) and points appear/disappear anyway, so a diff between two simplified outputs shows changes that are artifacts of the budget rather than the physics.",
    "repro": "For a fixed curve, compute set(_simplify(x, y, nmin=N)) for N = 20..120 and assert set(N-1) is a subset of set(N).",
    "confidence": "medium"
  },
  {
    "id": "S1-B-19",
    "file": "trinity/_functions/simplify.py",
    "line": 641,
    "class": "divergence",
    "severity": "S4",
    "claim": "The priority order is documented with four tiers at :641 (1 endpoints, 2 high-prominence extrema DESC, 3 x-uniform coverage skeleton, 4 remaining merged-pool points in hierarchical-bisection order), but :702 states that idx_dist (the arc-length bin boundaries) is 'promoted ABOVE bisection_pool' - a fifth tier absent from the list.",
    "evidence": "trinity/_functions/simplify.py:641, :702.",
    "expected": "One authoritative statement of the priority order; the docstring at :298 also does not mention the arc-length promotion.",
    "failure_scenario": "A future edit 'restores' the documented 4-tier order and silently removes the arc-length promotion, which :709 warns 'silently undersamples high-gradient regions in either axis'.",
    "repro": "",
    "confidence": "high"
  },
  {
    "id": "S1-B-20",
    "file": "trinity/_functions/simplify.py",
    "line": 298,
    "class": "other",
    "severity": "S4",
    "claim": "dedup_tol is documented as 'Pass 0 to disable', but the stated merge rule is 'merged only when both |dx| <= dedup_tol*range_x and |dy| <= dedup_tol*range_y'. At dedup_tol = 0 that rule still merges consecutive samples with dx == 0 and dy == 0.",
    "evidence": "trinity/_functions/simplify.py:298 (dedup_tol parameter), :457.",
    "expected": "Either the code short-circuits the dedup pass when dedup_tol == 0 (making 'disable' true), or the docstring says 'exact duplicates are still collapsed'.",
    "failure_scenario": "A caller passes 0 expecting a literal pass-through for a length check or a bit-identical comparison, and gets a shorter array because exact duplicates were folded.",
    "repro": "_simplify(x, y, dedup_tol=0) on an array containing two exactly-equal consecutive samples; check whether both survive.",
    "confidence": "medium"
  },
  {
    "id": "S1-B-21",
    "file": "trinity/_functions/simplify.py",
    "line": 462,
    "class": "other",
    "severity": "S4",
    "claim": "The dedup rule is labelled 'The OR-on-Delta rule' at :462 while the parameter docstring says samples are 'merged only when *both* |dx| <= ... and |dy| <= ...' (AND).",
    "evidence": "trinity/_functions/simplify.py:298 (dedup_tol) vs :462.",
    "expected": "The two are reconcilable by De Morgan (keep iff |dx| > tol OR |dy| > tol), but the labels are 160 lines apart with no cross-reference. One phrasing should be used in both places.",
    "failure_scenario": "A maintainer implements the literal 'OR' merge condition, which would fold every vertical-drop and horizontal-plateau sample - precisely what both comments say must be preserved.",
    "repro": "",
    "confidence": "high"
  },
  {
    "id": "S1-B-22",
    "file": "trinity/_functions/simplify.py",
    "line": 498,
    "class": "other",
    "severity": "S4",
    "claim": "The nmin >= 20 floor is justified as 'matches _COVERAGE_CHUNKS ... so the algorithm has enough budget for both endpoints and a meaningful coverage skeleton' (:498) and 'so endpoints + coverage always fit inside the budget' (:298). 2 endpoints + 20 coverage chunks = 22 > 20.",
    "evidence": "trinity/_functions/simplify.py:239 (_COVERAGE_CHUNKS = 20), :298 (nmin doc), :498, :694 (cap at nmin - 2, 'at nmin >= 22 the cap is inert').",
    "expected": "The stated property is delivered by the separate nmin-2 cap, not by the floor. As written the rationale is arithmetically false and would mislead anyone changing either constant.",
    "failure_scenario": "Someone raises _COVERAGE_CHUNKS while trusting the floor rationale and removes the nmin-2 cap as 'redundant', after which the mandatory set exceeds nmin at every small budget.",
    "repro": "_simplify(x, y, nmin=20) - assert len(out) and compare against the documented 'Output size is normally nmin'.",
    "confidence": "high"
  },
  {
    "id": "S1-B-23",
    "file": "trinity/_functions/simplify.py",
    "line": 581,
    "class": "other",
    "severity": "S4",
    "claim": "The mandatory-extremum threshold is documented as strict at :298 ('Extrema whose prominence exceeds 5 % of the y-range are mandatory') and as non-strict at :652 ('prominence >= 5 % of the y-range').",
    "evidence": "trinity/_functions/simplify.py:298, :581, :652.",
    "expected": "One comparison operator, matching the code.",
    "failure_scenario": "Boundary-only; a feature sitting exactly at 5% is documented both as kept and as droppable.",
    "repro": "Construct a curve with an extremum of prominence exactly 0.05*(y-range) and check membership in the mandatory set.",
    "confidence": "high"
  },
  {
    "id": "S1-B-24",
    "file": "trinity/_functions/simplify.py",
    "line": 492,
    "class": "other",
    "severity": "S4",
    "claim": "The nmin contract says output is 'normally nmin points; it may be LARGER when the curve has more than nmin high-prominence extrema' (:298), but :492 documents a path returning fewer: 'If the (deduplicated) array is already short enough, return it in caller order ... if dedup collapsed a clump, the caller gets the meaningful subset rather than the duplicate-laden original.' The empty-array path (:429) is likewise undocumented.",
    "evidence": "trinity/_functions/simplify.py:298, :429, :492.",
    "expected": "The Returns section should state that len(out) <= len(in) always and may be < nmin for short or heavily deduplicated input.",
    "failure_scenario": "A caller preallocates or asserts len(out) == nmin and breaks on a short input.",
    "repro": "_simplify(x[:5], y[:5]) and _simplify([], []) - check returned lengths against the documented contract.",
    "confidence": "high"
  },
  {
    "id": "S1-B-25",
    "file": "trinity/_functions/simplify.py",
    "line": 124,
    "class": "citation",
    "severity": "S4",
    "claim": "'This is equivalent to the persistence of the extremum in the sublevel-set filtration of y' is stated for the routine as a whole, which handles both maxima and minima ('For each index p in idx (a local maximum or minimum of y)').",
    "evidence": "trinity/_functions/simplify.py:124.",
    "expected": "Minima persist in the sublevel-set filtration; maxima persist in the SUPERLEVEL-set filtration (equivalently the sublevel-set filtration of -y). The attribution as written is correct for only half the inputs.",
    "failure_scenario": "Documentation only - but it is the stated justification for the mirrored MIN branch, so a reader checking the MAX/MIN symmetry against the cited theory gets the wrong reference frame (see also S1-B-16).",
    "repro": "",
    "confidence": "medium"
  },
  {
    "id": "S1-B-26",
    "file": "trinity/_functions/simplify.py",
    "line": 25,
    "class": "other",
    "severity": "S4",
    "claim": "Two equivalence claims are asserted in comments with no cited artifact: 'Output is byte-identical to the straightforward numpy-indexed version' (:25) and 'The traversal is byte-identical to the queue version; verified against the BFS for n in {2, 3, 4, ..., 30 000}' (:656). Both accompany performance rewrites of hot paths.",
    "evidence": "trinity/_functions/simplify.py:25, :656.",
    "expected": "Per the project's own rule for perf 'free wins', a bit-identical claim needs a committed harness/CSV plus a value diff, and the reference implementation needs to still exist somewhere runnable. Neither comment names a test, a file, or a command.",
    "failure_scenario": "A later micro-optimisation breaks equivalence and nothing re-checks it, because the reference implementation the claim compares against is not in the tree.",
    "repro": "grep the pytest suite for a test that reimplements the naive prev/next-strictly-greater scan and the BFS bisection order and asserts equality.",
    "confidence": "medium"
  },
  {
    "id": "S1-B-27",
    "file": "trinity/_functions/simplify.py",
    "line": 825,
    "class": "other",
    "severity": "S4",
    "claim": "'np.interp requires ascending reference x; sort the simplified curve internally so this works for descending or NON-MONOTONIC inputs too.' Sorting only fixes the reference curve; the query grid is x_orig, and residuals are computed pointwise against y_orig.",
    "evidence": "trinity/_functions/simplify.py:760, :825, :832, :835.",
    "expected": "For non-monotonic x_orig a single x maps to several y_orig values, so the interpolated reconstruction cannot match more than one of them and every derived metric (r_squared, max_rel_err, all log_* fields) is meaningless rather than merely approximate. The claim should be scoped to ascending/descending input.",
    "failure_scenario": "Error metrics reported for a non-monotonic curve look poor (or spuriously fine) for reasons unrelated to the simplification quality, and the post-hoc R^2 UserWarning at :730 fires or fails to fire arbitrarily.",
    "repro": "_simplify_error on x that goes up then down; compare r_squared against a manual per-branch evaluation.",
    "confidence": "medium"
  },
  {
    "id": "S1-B-28",
    "file": "trinity/_functions/logging_setup.py",
    "line": 122,
    "class": "deadcode",
    "severity": "S4",
    "claim": "The setup_logging docstring instructs 'Call this function once at the start of your simulation (in main.py)', and set_log_level's example (:368) uses the logger name 'trinity.phase1_energy.run_energy_phase'. The project's documented single entry point is run.py.",
    "evidence": "trinity/_functions/logging_setup.py:122 (Notes), :368 (example), and the repo layout which names run.py as the single entry point.",
    "expected": "Both references should name modules that exist; a stale logger-name example is worse than none because set_log_level on a non-existent logger name silently creates and configures a logger nobody uses.",
    "failure_scenario": "A user follows the example to raise verbosity for the energy phase, a new logger object is created for a dead name, and nothing changes - with no error.",
    "repro": "grep for main.py and for the module path trinity/phase1_energy/run_energy_phase.py.",
    "confidence": "medium"
  },
  {
    "id": "S1-B-29",
    "file": "trinity/_functions/logging_setup.py",
    "line": 80,
    "class": "silent-failure",
    "severity": "S4",
    "claim": "DedupWarningFilter 'passes the FIRST occurrence of each unique message (at min_level and above) and drops exact repeats', with named examples that are physics diagnostics: 'a super-critical Bonnor-Ebert sphere, nEdge < nISM, rCloud > rCloud_max'. No suppressed-repeat count is mentioned, and the per-process 'seen' state has no stated bound.",
    "evidence": "trinity/_functions/logging_setup.py:80, :266.",
    "expected": "Either a final 'message X suppressed N times' summary, or an explicit statement that repeat frequency is not recoverable from the log. Also worth checking: the unique-message set grows without bound for messages that embed changing values (the docstring says such messages are deliberately NOT collapsed), which on a long run is a monotonically growing per-process cache.",
    "failure_scenario": "A run in which rCloud > rCloud_max fires on every step is indistinguishable in the log from one where it fired once, so a systematically invalid configuration reads as a single benign warning.",
    "repro": "Emit the same warning 10000 times through a handler carrying the filter and inspect the log for any indication of the count.",
    "confidence": "high"
  },
  {
    "id": "S1-B-30",
    "file": "trinity/_functions/cluster.py",
    "line": 49,
    "class": "regime",
    "severity": "S4",
    "claim": "get_optimal_workers keys the 'use the FULL allocation' branch on SLURM_JOB_ID being set, while detect_allocated_cpus derives the core count from SLURM_CPUS_PER_TASK -> SLURM_CPUS_ON_NODE -> os.sched_getaffinity -> os.cpu_count().",
    "evidence": "trinity/_functions/cluster.py:3 (precedence list and the '64-core node, 4-core job -> ~31 workers' example), :39, :49.",
    "expected": "Inside a SLURM job where neither SLURM_CPUS_* is exported and cgroup/cpuset confinement is absent or unreadable, detection falls through to os.cpu_count() (the whole node) AND the full-allocation branch is taken - producing exactly the oversubscription the module docstring says it prevents. The conservative max(1, cpu//2 - 1) halving does not apply in that branch.",
    "failure_scenario": "A job submitted without --cpus-per-task on a 64-core node spawns 64 simulation subprocesses inside a 4-core allocation and is killed for exceeding its cgroup.",
    "repro": "SLURM_JOB_ID=1 with SLURM_CPUS_PER_TASK and SLURM_CPUS_ON_NODE unset; call get_optimal_workers and compare to the granted core count.",
    "confidence": "medium"
  }
]
```
