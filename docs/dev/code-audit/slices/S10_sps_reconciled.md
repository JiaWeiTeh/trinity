# S10 SPS feedback — reconciled

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

**Slice:** `trinity/sps/read_sps.py`, `trinity/sps/sps_columns.py`, `trinity/sps/update_feedback.py`.
**Method:** blind triangulation. Lens A (code), Lens B (comments/docstrings), Lens C (physics spec),
each written in isolation. I have not read the source, the `lib/` tables, or the slice inputs — the
three lens reports are my entire evidence base. Where I derive algebra below, it is derived *from
the expressions Lens A transcribed*, not from source.

Input volume: A = 19 candidates, B = 28, C = 25 (72 total). Reconciled output: **24 items**, plus
**12 clearances** carried as first-class results and **10 candidates dropped or refuted**.

---

## 1. Coverage table

| Dimension | A (does) | B (claims) | C (should) | Converge? |
|---|---|---|---|---|
| Column index → quantity mapping | ✔ traced literal | ✔ transcribed preset | ◑ `[recalled]` SB99 files only | **A=B exact**; C not applicable |
| Declared units per column | ✔ | ✔ | ✔ | **A=B=C** |
| log/linear flag per column | ✔ | ✔ | ◑ pattern-level | **A=B exact**, C corroborates pattern |
| Conversion factor values | ✔ all 10 checked numerically | ◑ names targets only | ✔ derives required values | **A=C** (see CL-1 on a 2e-4 primitive offset) |
| De-log before convert (order) | ✔ | ✔ | ✔ | **A=B=C** |
| `f_mass` application count | ✔ traced, exponent bookkeeping | ◑ "applied by the caller" | ✔ derives extensive/intensive split | **A=C** |
| Wind quadruple identities | ✔ full algebra | — no formulas in prose | ✔ exact identities | **A=C** |
| SN quadruple / over-determination | ✔ precedence ladder traced | ◑ validator rules + admitted gap | ✔ 2-DoF theorem | **ABC** (defect) |
| Totals & effective velocity | ✔ | ✔ | ✔ | **A=B=C** |
| Off-grid `t > t_max` | ✔ raises | ✖ **undocumented** | ✔ must not clamp | **A=C**, B silent |
| Off-grid `t < t_min` | ✔ raises | ✖ undocumented | ✔ must clamp | resolved, see §3 |
| Interpolation scheme / positivity | ✔ cubic on linear y, no clamp | ◑ "cubic recommended" | ✔ needs monotone C¹ | **ABC** (defect) |
| `t` monotonicity + `t=0` prepend | ✔ | ✔ | ✔ | **A=B=C** |
| Header/delimiter sniffing | ✔ | ✔ | ◑ F.14 only | **A=B** |
| Validation & error paths | ✔ every branch | ✔ documented rules | ◑ | **A=B**, A refutes 3 B items |
| `SPSFeedback` interface | ✔ | ✔ | ✔ | **A=B=C** (fragility, not bug) |
| Table provenance | — out of scope | ✔ documented blank | ✔ what it must state | **B=C** (S3 doc) |
| Regime / IMF sampling validity | ◑ notes no bound | — | ✔ thresholds | **A=C** |
| Caching / state | ✔ | ✖ no claim | — | single-lens A |
| Dead code | ✔ | — | — | single-lens A |

✔ full · ◑ partial · ✖ explicit gap · — not covered

---

## 2. Column-mapping diff — the primary work

Lens A traced the hardcoded default `DEFAULT_SPS_COLUMN_MAP` (`sps_columns.py:166-174`).
Lens B transcribed the documented "Legacy SB99 7-column positional preset" (`sps_columns.py:152-165`).
Diffed index by index, including declared unit **and** log flag:

| idx | B — documented quantity + unit + log | A — code's assumed quantity + unit + log | Verdict |
|---|---|---|---|
| 0 | `t`, `yr`, **linear** | `t`, `yr`, **linear** → ×1e-6 → Myr | **agree** |
| 1 | `Qi`, `1/s`, **log10** | `Qi`, `1/s`, **log** → ×1/s2Myr = 3.1557e13 → 1/Myr | **agree** |
| 2 | `fi`, dimensionless, **log10** | `fi`, dimensionless, **log** → ×1 | **agree** |
| 3 | `Lbol`, `erg/s`, **log10** | `Lbol`, `erg/s`, **log** → ×L_cgs2au | **agree** |
| 4 | `Lmech_total`, `erg/s`, **log10** | `Lmech_total` (wind **+ SN**), `erg/s`, **log** | **agree** |
| 5 | `pdot_W`, `g*cm/s^2`, **log10** | `pdot_W` (**wind-only**), `g*cm/s^2`, **log** → ×pdot_cgs2au | **agree** |
| 6 | `Lmech_W`, `erg/s`, **log10** | `Lmech_W` (**wind-only**), `erg/s`, **log** | **agree** |

**Verdict: consistent.** All seven indices, all seven declared units, all seven log flags agree
exactly. There is no off-by-one, no log column read as linear, no linear column read as log, and no
unit mismatch between what the comment block promises and what the literal encodes. The log/linear
split is identical in both accounts: col 0 linear, cols 1–6 log10. Lens C corroborates the *pattern*
at type level (time in years and linear; rate columns logged; a fraction column logged and therefore
negative-valued; a per-column `log` flag is the right design because SB99's own `*.snr` mixes log and
linear in one file) but its per-file layout table is tagged `[recalled]` and describes SB99's separate
`*.quanta`/`*.power`/`*.snr` outputs, not a merged 7-column CSV — **so C's column ordering is not
evidence against the code and is not scored as a divergence.**

Three riders that survive the clean diff:

1. **Nothing else guards the mapping.** A reports `load_user_columns` bounds-checks an integer index
   (`0 <= idx < n_cols`) and stops there; `header_names`, which `_scan_layout` does recover, is
   consulted only on the string-name branch. So this diff is the only cross-check that exists, and
   it is a diff of two artefacts *in the same file written by the same author* — not independent
   confirmation that the bundled CSV's columns are actually in that order. **Neither lens read
   `lib/default/sps/starburst99/1e6cluster_default.csv`.** Agreement between comment and literal is
   necessary, not sufficient. (→ R-02, R-03)
2. **The masking failure mode is real and both A and B found it independently.** Indices 4 and 6 are
   both `erg/s`, both log, both luminosities. Swapping them makes `Lmech_SN_raw = Lmech_W −
   Lmech_total < 0` for the whole run, which the code turns into one WARNING and a clamp to zero —
   i.e. **SN feedback silently off, run completes**. The clamp that correctly handles log-rounding
   noise (C-10's requirement, which the code satisfies) is the same clamp that would hide a
   catastrophic swap. Fix is a diagnostic-strength change, not a clamp removal. (→ R-02)
3. **A found the literal unreferenced within the slice**; the live path is
   `params['sps_column_map'].value`, built from `.param` `sps_col_*` keys. B's prose says the preset
   is "injected as the column map for the bundled default file" — presumably by `read_param.py`,
   outside the slice. So the verified literal and the object actually injected are, strictly, not yet
   shown to be the same object. Cheap to close; recorded as R-17's rider.

---

## 3. The time-grid asymmetry — resolved

**The facts.** C derives an asymmetry: past `t_max`, clamping is wrong (a finite ~1e55 erg SN budget
becomes unbounded, changing the reported fate — C's headline S1); below `t_min`, clamping is right (a
burst is flat over 0–0.5 Myr, and backward cubic extrapolation can undershoot negative at the most
fragile solver step). A finds the code raises a hard `ValueError` in **both** directions, on a
**closed** interval `t_min <= t <= t_max`.

**Past `t_max`: not a defect — a clearance.** C's own ranked list of acceptable behaviours puts
"**(i) Refuse.** Raise/terminate with an explicit end-reason" *first*, above zeroing, above log-log
extrapolation, with clamping last and "wrong otherwise". The code implements C's top choice. C-04 is
cleared on physics. Two residuals, both S3 and both documentation-shaped, not physics-shaped: a bare
`ValueError` is not "an explicit end-reason" (whether the run driver converts it into a recorded
stopping fate or a lost run is outside this slice), and B confirms **no prose anywhere in the slice
states any off-grid policy** — B-13 asked "raises or extrapolates?" and A answers "raises, from
trinity's own guard, before scipy's". (→ R-07)

**Below `t_min`: not a defect either, for two independent reasons.**

*Reachability.* A reports `t_min` is **exactly 0.0 by construction** — `read_sps.py:263-264`
guarantees `t[0] == 0.0`, either because the file starts there or because 0.0 is prepended. So
`t < t_min` means *negative simulation time*. For a code whose cluster forms at t=0 that is
unreachable by any physical query.

*The clamp C wants is already there.* C-25's scenario is "SB99 grids commonly start at 1e4–1e5 yr, so
the loader must extrapolate below `t_min`; clamping to the first row is correct." The `t=0` prepend
**is** that clamp — A describes it as "a constant (zeroth-order) extension: `y_new[0] = y[0]`". The
code delivers C's required behaviour by materialising the clamp as a knot instead of as an off-grid
policy. C-25 is cleared *in substance*. Its concern re-enters by a different door, though: A, B and C
all independently note that a flat row butted against a steep rise, fed to an unconstrained cubic, is
the textbook ringing configuration — so the negative-driver-at-t=0 failure C-25 predicts can still
occur, via spline overshoot rather than via extrapolation. That is R-05, not a t_min policy defect.

**The `±1e-9` central difference: one defect, and it is not the one C predicted.** A reports
`update_feedback.py:184-185` computes `pdotdot_total` as `(fpdot_total(t+1e-9) −
fpdot_total(t−1e-9))/2e-9`. At `t == t_min == 0.0` this evaluates the spline at `−1e-9`, below
`interp1d`'s domain with default `bounds_error=True` ⇒ `ValueError`. Symmetrically at `t == t_max`.
The explicit guard admits a **closed** interval; the derivative needs an interval **open by 1e-9**.
So the lower bound is reachable **only by the code's own arithmetic**.

**Verdict: one defect, not two.** Remove the stencil problem and the lower-bound raise has no
reachable trigger and no physics cost (the required clamp lives in the prepend). Keep the stencil and
the lower-bound raise is not an independent policy error but the *symptom* of a single root cause: the
domain contract is inconsistent between the guard and its consumer. The fix is one change (one-sided
difference at the endpoints, a clamped stencil, or `CubicSpline(...).derivative()` which needs no
stencil at all) and it resolves both the `t=0` and the `t=t_max` crash. (→ R-01)

**Honest caveat on reachability, which I rank as the first thing to verify.** A's mechanism is
high-confidence, but taken literally it says the natural first ODE evaluation at `t = 0.0` crashes
every run — which the project's documented working quickstart contradicts. Either the solver never
queries exactly `0.0`, or something upstream of `get_current_sps_feedback` supplies t=0 feedback
another way. A's direct repro (`get_current_sps_feedback(float(sps_f['fQi'].x[0]), params)`) settles
the mechanism regardless of whether the solver ever does it, so run that first and treat the
*reachability* as medium confidence until it does.

---

## 4. Lens C's layout-independent checks, run against Lens A's transcription

C's point is that these need no knowledge of file layout, so they can be scored from A's transcription
alone. Two things are distinguishable and I score them separately: whether the **relation holds in the
code's construction**, and whether a **guard exists** to catch it if a table breaks it.

| C's check | Holds by construction? | Guard present? | Verdict |
|---|---|---|---|
| `v = 2L/ṗ` (wind) | **PASS** — exact | n/a (derived) | **pass** |
| `Ṁ = ṗ²/(2L)` (wind) | **PASS** — exact, `read_sps.py:214` | n/a (derived) | **pass** (but `Ṁ` is never exported) |
| `L_i + L_n = L_bol` | **PASS** in the derived path; `Li=Lbol·fi`, `Ln=Lbol·(1−fi)`; survives interpolation exactly | **absent** on the supplied-column path | **pass / not enforced** |
| `0 ≤ f_i ≤ 1` | **not determinable** — needs the table | **absent** (A enumerated every check; `fi` is in none) | **not determinable; guard fails** |
| `L_i/Q_i ≥ 13.6 eV` | **not determinable** — needs the table | **absent** | **not determinable; guard fails** |
| `v_w < c` | **not determinable** | **absent** — A: `velocity_wind` "is never range-checked", and it is not even exported | **not determinable; guard fails** |
| `L_w/L_bol ∈ [1e-4, 1e-2]` | **not determinable** | **absent** | **not determinable; guard fails** |

Notes that change the reading of this table:

- The two **exact identities pass by construction, and I verified them on A's collapsed output
  algebra rather than taking A's word.** With `a ≡ FB_mColdWindFrac`, `θ ≡ FB_thermCoeffWind`, A's
  wind block yields `Ṁ_out = (1+a)ṗ²/(2L)`, `v_out = √(θ/(1+a))·2L/ṗ`, `ṗ_out = √(θ(1+a))·ṗ`,
  `L_out = θL`. Then `2L_out/ṗ_out = √(θ/(1+a))·2L/ṗ = v_out` ✔; `Ṁ_out v_out = √(θ(1+a))·ṗ = ṗ_out`
  ✔; `½Ṁ_out v_out² = θL = L_out` ✔. All three closures hold on the *exported* pair, at every grid
  point. C-01's factor-2 concern is cleared.
- The four **not-determinable** rows all fail for the same reason and share one fix: **no physical
  plausibility gate of any kind exists after ingest.** A's exhaustive branch/clamp inventory finds
  positivity checked for exactly one quantity (`Lmech_SN_raw`) and finiteness checked once; nothing
  else. C's F-table supplies six cheap detectors (`fi∈[0,1]`, `v_w<c`, `L_w/L_bol`, `L_i/Q_i≥13.6 eV`,
  `Qi[-1]<Qi[0]`, `t[-1]∈[10,1e3] Myr`) that between them catch F.1, F.2, F.3, F.4, F.6, F.7, F.11 and
  F.15 — i.e. every silent S1 class C identified. This is consolidated as **R-03**, the single highest
  value-per-line change in the slice.
- The `v_w < c` gate deserves separate emphasis: it is simultaneously C's detector for an
  `L_w ↔ L_bol` swap (F.4) **and** the missing downstream sanity check A named as the second line of
  defence for the unguarded column mapping (A-01) **and** the guard C-09 wants against the
  `EPSILON`-denominator blowup (R-08 gives ~1e130 pc/Myr). Three independent findings, one assertion.
- `Ṁ` never leaves the module (A: "`Mdot_wind` and `Mdot_SN` are local variables, never exported").
  So C's Cauchy–Schwarz caveat — a derived `Ṁ_eff = ṗ²/(2L) ≤ Ṁ_true`, ambiguous by the population's
  velocity spread — is not propagated by this slice; any consumer needing `ρ_w = Ṁ/(4πr²v)` must
  reconstruct it and inherits the ambiguity. Recorded as a modelling note, not a defect.
- **Methodological warning for whoever runs these tests.** A establishes that `interp1d` is a linear
  operator on `y`, so the *additive* identities (`L_tot = L_W + L_SN`, `ṗ_tot = ṗ_W + ṗ_SN`,
  `L_i + L_n = L_bol`) survive off-grid to floating point, but the *multiplicative/ratio* identities
  do **not**: `2·fL_tot(t)/fṗ_tot(t)` equals the true grid-point velocity only *at knots*. C's A.4
  residual test (`|r| < 1e-12`) must therefore be evaluated at knots; on a dense grid it will show
  drift that is an interpolation artefact, not a bug.

### Does the pair {C's `v`-invariance argument, A's `f_mass` trace} close the double-scaling question?

**Within the slice, yes. Across the slice boundary, no — and that residual is the more dangerous
half.**

C's contribution is a negative result: `v = 2L/ṗ` is a ratio of two extensive quantities, so a
*uniform* double application of `f_mass` leaves `v` bitwise unchanged and is invisible to every ratio
test. C therefore says the invariance test is necessary but not sufficient and demands an absolute
check (`L(f)/L(1) == f`, or the B.2 mass/energy integrals).

A supplies exactly the absolute check C could not perform, by tracing rather than measuring: the only
mass dependence in the module is `if CANONICALS[canonical].mass_scaled: arr = arr * f_mass`, applied
per canonical, once, immediately after unit conversion and **before any derivation** — and A then
carries the `f_mass` exponent through every derived quantity (`Ṁ_W ⇒ f²/f¹ = f¹`; `v_W ⇒ f¹/f¹ = f⁰`,
correctly mass-independent; `ṗ_W^out, L_W^out ⇒ f¹`; `L_SN^raw ⇒ f¹`; `v_SN` flagged
`mass_scaled=False` from either source; `Ṁ_SN ⇒ f¹`; `Li/Ln ⇒ f¹` whether derived or read as columns).
Together with C's extensive/intensive classification — which A's `mass_scaled` flag table matches
exactly, including the subtle cases `fi=False` and `v_SN=False` — **the double-application question
is closed: C-03's F.5 and F.6 are both cleared.**

What is *not* closed is the question one layer up, which C-13 raises and A cannot see: `f_mass =
M_cluster / sps_refmass` is only meaningful if `sps_refmass` is the normalisation the table was
actually generated with. B confirms the slice documents no burst mass, no SPS version, no IMF, no
metallicity — the only hint is the filename `1e6cluster`. A per-M⊙-normalised table with
`sps_refmass = 1e6`, or a continuous-SF table (normalised to 1 M⊙/yr, dimensionally a different
object), is a 1e6-class error that is invisible to `v`-invariance, invisible to A's trace, and
invisible to every ratio check. That is R-09, and C's discriminators (`Qi(20 Myr) < 0.1·Qi(1 Myr)`;
`∫Ṁ dt < M_cluster`; `∫L_bol dt < 0.007 M c²`) are the cheap way to close it.

---

## 5. The over-determined SN block

**All three lenses converge, from three different directions.** This is the strongest ABC-corroborated
design finding in the slice.

- **C (derived, theorem):** `{L, ṗ, Ṁ, v}` are constrained by two definitions (`L = ½Ṁv²`, `ṗ = Ṁv`),
  so exactly two are free. A loader admitting more must either check the redundancy to table precision
  (`|r| ≲ 3e-3` for 3-decimal log columns) or declare which pair is primary. Silently accepting an
  inconsistent quadruple means the bubble's energy equation and its momentum equation are driven by
  two different physical winds.
- **A (traced):** the SN block admits **four** independent inputs — (`Lmech_SN` **or**
  `Lmech_total − Lmech_W`), `Mdot_SN`, `pdot_SN`, `v_SN` — resolved by a silent precedence ladder with
  **no cross-check anywhere**.
- **B (documented):** the validator demands `Lmech_total` **or** `Lmech_SN`, and its own comment admits
  the entry-point space is incomplete: "`Mdot_SN` alone is **not yet** a supported entry point here."
  The author knew the input space was over-wide and under-specified.

Two concrete manifestations, both traced by A and both predicted by C's theorem:

1. **A supplied `pdot_SN` is exported raw.** It is only `f_mass`-scaled; it never receives the
   `√(θ_S(1+b))` factor that the derived branch applies, while `Ṁ_SN` *is* multiplied by `(1+b)` and
   `L_SN^out` *is* multiplied by `θ_S`. The exported triple then violates `ṗ = Ṁv`. Independently,
   C-18 derives that mass loading must hold `L` fixed and recompute **both** `v` and `ṗ`, and warns
   that "reducing `v` but keeping the tabulated `ṗ` implicitly changes `L` by `(1+f)^(-1/2)` — an
   energy leak proportional to the loading". **That is exactly this code path.** A rated it S3, C
   rated the class S1/S2; reconciled to **S2 when a user table declares `sps_col_pdot_SN`, S3 in
   effect for default-preset runs which cannot reach it.** (→ R-04)
2. **A supplied `Mdot_SN` makes the validator-mandated `Lmech` column dead.** `Lmech_SN_raw` is built,
   possibly warned about, clamped — and then read only inside the `else` branch. The exported
   `Lmech_SN` becomes `½·θ_S·Ṁ_SN^col·v_SN²`, which need not equal `θ_S·L_SN^raw` by any bounded
   amount, with no message. So the interface simultaneously *requires* a column and *ignores* it.
   (→ R-06)

**The wind block passes the same test cleanly** — exactly two independent reads, everything else
derived, outputs rebuilt from the modified pair — so C-02 is half-cleared and the defect is
specifically an SN-block design defect, not a module-wide one (CL-3).

**Recommended resolution, merging all three lenses:** declare `(Lmech_SN, v_SN)` primary for the SN
channel (that is what the default path already does), and either reject a supplied `pdot_SN`/`Mdot_SN`
as over-determining, or accept them and cross-check `|2L/(ṗv) − 1| < 3e-3` and `|ṗ/(Ṁv) − 1| < 3e-3`
per C's tolerance, raising on failure. Relaxing the validator's `Lmech` requirement when `Mdot_SN` is
supplied is *not* sufficient on its own — it would remove the dead column but leave the quadruple
unchecked.

---

## 6. Divergence table — everything else

Class key: **AB** doc-drift · **AC** physics · **BC** mis-cited/recalled · **ABC** all three · **scope-creep** · **none**.

| # | Item | A | B | C | Class | Resolution | Status |
|---|---|---|---|---|---|---|---|
| 1 | Column mapping (7 indices) | literal traced | preset transcribed | `[recalled]`, per-file | **none** | identical; C inapplicable | corroborated |
| 2 | `t_max` policy | raises | undocumented | must not clamp | AB | code = C's top choice; prose silent | corroborated |
| 3 | `t_min` policy | raises | undocumented | must clamp | AC→none | prepend *is* the clamp; raise unreachable | corroborated |
| 4 | `±1e-9` stencil | crashes at both ends | predicted the risk | — | AB | real defect, R-01 | corroborated |
| 5 | SN over-determination | 4 inputs, no check | admits gap | 2 DoF theorem | **ABC** | R-04, R-06 | corroborated |
| 6 | Cubic on linear y, no ≥0 clamp | traced | "cubic recommended" (unsupported) | needs monotone C¹ | **ABC** | R-05 | corroborated |
| 7 | `EPSILON` role | denominator guard, 1e-100 | — | asks: floor or guard? | AC | **A answers C's open question H.2: denominator guard** | corroborated |
| 8 | Negative `Lmech_SN` | warn + clamp to 0 | "prose states no clamp" | must clip | AB | **B-05 refuted**; clamp exists; C-10 satisfied | dropped |
| 9 | `Lmech_W`/`pdot_W` required? | in `_REQUIRED_ALWAYS` | "never required" | — | AB | **B-22 refuted** | dropped |
| 10 | Int index bounds check | exists (`0≤idx<n_cols`) | "not documented" | — | AB | **B-17 refuted** | dropped |
| 11 | Doc order: `Mdot_SN` before `v_SN` | code binds `v_SN` first | doc lists it after | — | AB | **B-06 demoted** to doc-drift | dropped→R-16 |
| 12 | `v_wind` documented as output | not returned | listed in Notes | — | AB | doc-drift; **B-09 failure scenario refuted** (2L_out/ṗ_out = v_W exactly) | corroborated |
| 13 | log-space interpolation `log(0)` | interpolates linear y | — | NaN risk | AC | **C-20 inapplicable** | dropped |
| 14 | Bundled vs user loader divergence | single path | preset "injected" as a map | round-trip test | AC | **C-24 largely cleared**; narrowed to R-17 rider | corroborated |
| 15 | constant `fi` | `fi` is a column | — | constant `fi` is wrong | AC | **C-16 sub-claim refuted** | dropped |
| 16 | `t` strict monotonicity | `diffs<=0` rejected | enforced | must reject `<=` | AC | **C-08 cleared**; residual R-19 | corroborated |
| 17 | Validation before prepend | yes → `t[0]<0` reorders silently | — | — | AC | R-19 | single-lens |
| 18 | `FB_*` correction algebra | derived exactly | "no prose states the form" | mass-loading law | **ABC** | R-10; A's algebra **matches C's law** (CL-4) | corroborated |
| 19 | `FB_vSN` unconverted | raw `.value` | notes no documented conversion | canonical is pc/Myr | AB+C | R-14; resolution is cross-slice | corroborated |
| 20 | Provenance blank | — | version/IMF/Z/grid all absent | must be stated | BC | **S3 doc gap, not evidence of a code bug** | corroborated |
| 21 | `sps_refmass` vs table normalisation | out of slice | filename hint only | burst vs continuous-SF | BC | R-09 — the live 1e6-class risk | corroborated |
| 22 | Absolute AU constants | `1/L_cgs2au = 6.0241e29` | — | SPEC-091: `6.0255e29` | AC | **~2e-4, common to `C_L` and `C_p` ⇒ a primitive (M⊙) difference, not a code error** — see CL-1 | contested/benign |
| 23 | SB99 per-file layouts | — | merged 7-col CSV | `*.quanta`/`*.power`/`*.snr` | BC | `[recalled]` tag respected; **not evidence against the code** | contested/benign |
| 24 | `pdotdot_total` exists at all | computed, crash-prone | "for time evolution", no unit | never requested | **scope-creep** | R-18 | single-lens+ |
| 25 | `sps_f` cache not keyed on `f_mass` | traced | "no caching claim appears" | — | AB | R-23, low | single-lens |
| 26 | Low-`M_cluster` validity | no bound on `f_mass` | — | warn below ~1e4 M⊙ | AC | R-11 | corroborated |
| 27 | `SPSFeedback` positional protocol | hand-maintained, consistent *now* | 13 fields promised | swap undetectable | **ABC** | R-15 — fragility, not a live bug | corroborated |
| 28 | `'#'`-commented header | never detected | never detected | — | AB | R-20; A's message is clear, so loud not silent | corroborated |

---

## 7. Clearances (first-class results)

| ID | Cleared | Basis | Status |
|---|---|---|---|
| **CL-1** | All ten unit-conversion factors correct | A checked each numerically against `unit_conversions.py`. **C-07's two identity relations hold exactly by construction, not just numerically**: since `L_cgs2au = g2Msun·cm2pc²/s2Myr³` and `pdot_cgs2au = g2Msun·cm2pc/s2Myr²` are composed from the same three primitives, `C_L/C_p = 1/v_cms2au = C_v` and `C_p²/C_L = 1/(g2Msun/s2Myr) = C_Ṁ` identically. **Rider: do not "correct" these toward C's SPEC-091 values** — the ~2e-4 offset appears equally in `C_L` and `C_p`, i.e. it is a different M⊙ primitive (code 1.98841e33 g, the current IAU value; C's recalled 1.98892e33 is older), and adjusting one factor alone would *break* the exact internal relations. | A + reconciler |
| **CL-2** | `f_mass` applied exactly once, to the right columns | A traced the single multiplication site and carried the exponent through every derived quantity; the `mass_scaled` flag table matches C's extensive/intensive classification exactly, including `fi=False` and `v_SN=False`. Closes C-03/F.5/F.6 **within the slice**; does not close normalisation (R-09). | A + C |
| **CL-3** | Wind quadruple self-consistent | Exactly two independent reads; `Ṁ`, `v` derived; `ṗ_out`, `L_out` rebuilt from the modified pair. I re-derived all three closures on A's collapsed expressions — they hold exactly at every grid point. C-01 cleared; C-02 cleared for the wind. | A + reconciler |
| **CL-4** | Mass-loading law implemented correctly | C-18 requires `L` fixed, `ṗ ∝ √(1+f)`, `v ∝ 1/√(1+f)`, `Ṁ ∝ (1+f)`. A's algebra gives exactly that for the wind (`L_out = θ_W·L`, mass loading cancels) **and** for the default SN path (`L_SN^out = θ_S·L_SN^raw`). C flagged this path as "may be entirely untested"; it is nonetheless correct. Fails only on the supplied-`pdot_SN` branch (R-04). | A + C |
| **CL-5** | De-log before unit conversion | `if log: arr = 10**arr` then `arr * factor`. A verified, B documents the same order, C requires it. C-06 and trap F.1's catastrophic variant (`10**(arr/C)`) are absent. | A = B = C |
| **CL-6** | Totals built as sums, effective velocity from the totals | `L_tot^out = L_SN^out + L_W^out`, `ṗ_tot^out = ṗ_SN + ṗ_W^out`, `v_mech_total = 2L_tot/ṗ_tot`. The table's own `Lmech_total` column is *not* reused as the output total, removing the risk of an inconsistent tabulated total. Additive identities survive interpolation exactly. C-17 cleared. | A = B = C |
| **CL-7** | Strict `t` monotonicity enforced | `diffs <= 0` rejected — C-08 explicitly asked whether `>=` or `>` was used; it is the correct one. Duplicated rows are rejected. | A = C |
| **CL-8** | Negative `Lmech_SN` clipped, with a log message | `np.maximum(Lmech_SN_raw, 0)` — the only clamp in the module that announces itself. C-10 satisfied; **B-05's concern refuted**. Rider: the message cannot distinguish log-rounding noise from a column swap (R-02). | A vs B, C |
| **CL-9** | Off-grid policy is physics-correct in both directions | Past `t_max` the code refuses — C's top-ranked option, above zeroing and above clamping. Below `t_min` the physically-required clamp is delivered by the `t=0` prepend. C-04 and C-25 cleared in substance. | A + C |
| **CL-10** | `_L_SUN_ERG_S = 3.828e33` | Matches C-23's required IAU nominal value exactly. | A = C |
| **CL-11** | One ingestion path, not two | A finds the live path is always the user-map loader; B confirms the preset is *injected as a column map*. C-24's "two loaders diverge" risk does not exist as posed. | A + B |
| **CL-12** | Column mapping consistent | §2 — all 7 indices × unit × log flag agree between traced literal and documented preset. | A = B |

**Demoted or dropped (10):** B-05, B-06, B-09 (failure scenario only), B-17, B-22 — refuted by A's
trace. C-08, C-16 (constant-`fi` sub-claim), C-20, C-24 — refuted or made inapplicable. C-04/C-25 —
converted from defects into CL-9. B-12's provenance blank is retained but reclassified as S3
documentation per instruction, and merged into R-09 where it carries real correctness weight.

---

## 8. Merged ranked findings

```json
[
  {
    "id": "S10-R-01",
    "file": "trinity/sps/update_feedback.py",
    "line": 185,
    "class": "numerical",
    "severity": "S2",
    "claim": "The pdotdot_total central difference evaluates the spline at t +/- 1e-9, stepping outside the interpolation domain at exactly t_min and t_max - the endpoints the range guard three lines earlier explicitly admits. t_min is always exactly 0.0 by construction, so the code's own arithmetic makes the lower bound reachable.",
    "evidence": "A: update_feedback.py:156 guards with the CLOSED interval `t_min <= t <= t_max`; lines 184-185 then call fpdot_total(t+1e-9) and fpdot_total(t-1e-9). The ten interpolators are built at read_sps.py:341-354 with no fill_value and no bounds_error override, so scipy's default bounds_error=True raises outside [x[0], x[-1]]. read_sps.py:263-264 guarantees t[0] == 0.0. B-14 independently predicted the same failure from the docstring alone ('at t = 0 the prepended row a centred difference samples t < 0'). Reconciler: this is the ONLY reachable path to t < t_min - see the divergence analysis in section 3 - so the closed-vs-open domain mismatch is a single defect, not two.",
    "expected": "One-sided differences at the endpoints, a stencil clamped into [t_min, t_max], or scipy.interpolate.CubicSpline(...).derivative(), which needs no stencil. Raising past t_max remains correct policy (CL-9); only the endpoint arithmetic needs to change.",
    "failure_scenario": "The first ODE evaluation at t = 0.0 passes the range check and then dies with 'ValueError: A value in x_new is below the interpolation range'; symmetrically a run that reaches exactly t_max dies there instead of terminating with a fate.",
    "repro": "sps_f = get_interpolation(read_sps(1.0, params)); get_current_sps_feedback(float(sps_f['fQi'].x[0]), params)  # and again at float(sps_f['fQi'].x[-1])",
    "confidence": "high",
    "lenses": ["A", "B"],
    "divergence": "AB",
    "status": "corroborated",
    "source_ids": ["S10-A-02", "S10-B-14"]
  },
  {
    "id": "S10-R-02",
    "file": "trinity/sps/sps_columns.py",
    "line": 473,
    "class": "silent-failure",
    "severity": "S2",
    "claim": "An integer column index is never cross-checked against the file's own header row even when one was successfully parsed - only an array-bounds check runs. The comment-vs-literal diff in section 2 is the only thing guarding the mapping, and a same-unit swap of indices 4 and 6 is converted by the (otherwise correct) negative-SN clamp into a completed run with SN feedback silently off.",
    "evidence": "A: load_user_columns:473-480 takes the int branch and checks only `0 <= spec.file_column < n_cols`; header_names, recovered by _scan_layout:427-440, is consulted ONLY in the string branch at 490-497. The declared unit factor and log flag are then applied to whatever numbers sit at that index. Indices 4 (Lmech_total) and 6 (Lmech_W) are both erg/s, both log, both luminosities, so a swap is dimensionally undetectable; it makes Lmech_SN_raw = Lmech_W - Lmech_total < 0 for every row, which read_sps.py:203-208 turns into one WARNING plus np.maximum(...,0). B-01 independently identifies the same 4/6 swap as the preset's characteristic failure. Reconciler: A's traced literal and B's documented preset agree on all 7 indices/units/log flags, but neither lens read lib/default/sps/starburst99/1e6cluster_default.csv, so nothing has yet confirmed the FILE matches either.",
    "expected": "When header_names is non-empty, cross-check header_names[idx] against the canonical (or an alias table) and raise on mismatch. Independently, strengthen the negative-SN diagnostic so it distinguishes log-rounding noise (|L_SN| < 1e-3 of L_W, expected pre-onset) from a systematic sign inversion (negative on every row), which is a mapping error rather than a rounding artefact.",
    "failure_scenario": "sps_col_Lmech_W and sps_col_Lmech_total pointing at each other's indices produce a run that completes normally with one warning and zero supernova feedback for its entire life - the headline fate changes and nothing errors.",
    "repro": "Swap sps_col_Lmech_W and sps_col_Lmech_total indices in a .param, run `python run.py`, observe only the 'Negative SN mechanical luminosity detected; clamping to zero' warning and a completed run.",
    "confidence": "high",
    "lenses": ["A", "B"],
    "divergence": "none",
    "status": "corroborated",
    "source_ids": ["S10-A-01", "S10-B-01"]
  },
  {
    "id": "S10-R-03",
    "file": "trinity/sps/read_sps.py",
    "line": 174,
    "class": "silent-failure",
    "severity": "S2",
    "claim": "No physical plausibility gate of any kind runs after ingest. Finiteness is checked once and positivity for exactly one quantity (Lmech_SN_raw); fi is never bounded to [0,1], no velocity is range-checked, no ratio is checked, and the time axis is never checked for having been read in the right unit. Every one of Lens C's silent S1 traps is therefore undetected.",
    "evidence": "A's exhaustive branch/clamp inventory: read_sps.py:174 checks np.isfinite only; read_sps.py:203 sign-checks Lmech_SN_raw only; 'nothing validates positivity of Lmech_W, pdot_W, Lbol, or Qi'; velocity_wind (read_sps.py:215) 'has an obvious sanity window (a few 100 to a few 1000 km/s) and is never range-checked' - and is not even exported. C supplies the layout-independent detectors: Li/Qi >= 13.6 eV = 1.146e-54 AU (a theorem, expected band 1.3e-54..2.1e-54); 0 <= fi <= 1; v_w < c = 3.0664e5 pc/Myr; L_w/L_bol in [1e-4,1e-2]; pdot_w*c/L_bol in [0.02,2]; Qi[-1] < Qi[0] (rate vs cumulative shape test); t[-1] in [10,1e3] Myr (yr-vs-Myr). B-02 flags the specific fi risk: the code comment defensively insists col 2 is log-space, and if the file actually stores linear fi then 10**0.2..10**0.6 = 1.6..4 gives Li > Lbol and Ln < 0 for the whole run.",
    "expected": "A single post-ingest assertion block covering the six checks above, raising with the offending row index. Between them they catch C's traps F.1 (log read as linear), F.2 (linear read as log), F.3 (per-mass vs absolute), F.4 (L_w/L_bol swap), F.6 (intensive column scaled), F.7 (rate vs cumulative), F.11 (yr vs Myr) and F.15 (continuous-SF table) - the entire silent-S1 class.",
    "failure_scenario": "The log-read-as-linear direction gives Q = 53 and L = 40 in cgs, i.e. feedback ~1e50 times too weak; the bubble never expands and the run reports a plausible 'stalled/collapsed' verdict with no warning. The fi direction gives a negative non-ionising luminosity fed straight into the radiation force budget.",
    "repro": "After read_sps(1.0, params), assert 0 <= fi <= 1; assert (2*Lmech_W/pdot_W < 3.0664e5).all(); assert 1e-4 <= (Lmech_W/Lbol).max() <= 1e-2; assert (Li/Qi >= 1.146e-54).all(); assert Qi[-1] < Qi[0]; assert 10 <= t[-1] <= 1e3.",
    "confidence": "high",
    "lenses": ["A", "B", "C"],
    "divergence": "AC",
    "status": "corroborated",
    "source_ids": ["S10-A-01", "S10-A-06", "S10-B-02", "S10-C-11", "S10-C-14", "S10-C-16", "S10-C-19", "S10-C-15"]
  },
  {
    "id": "S10-R-04",
    "file": "trinity/sps/read_sps.py",
    "line": 241,
    "class": "divergence",
    "severity": "S2",
    "claim": "The SN block accepts up to four independent inputs for a two-degree-of-freedom system with no redundancy check. Concretely, a supplied pdot_SN column is exported raw - bypassing FB_thermCoeffSN and FB_mColdSNFrac - while Mdot_SN and Lmech_SN in the same block ARE modified by them, so the exported quadruple violates pdot = Mdot*v and silently breaks the mass-loading energy identity.",
    "evidence": "A: read_sps.py:241-242 uses cols['pdot_SN'] as-is (f_mass-scaled only), while line 234 applies (1+FB_mColdSNFrac) to Mdot_SN, lines 236-239 apply sqrt(FB_thermCoeffSN/(1+FB_mColdSNFrac)) to the velocity, and line 246 always recomputes Lmech_SN = 0.5*Mdot_SN*v_mod**2. The derived branch (line 244) gives pdot_SN = sqrt(theta_SN*(1+b))*pdot_raw; the column branch simply lacks that factor. C-02 derives the 2-DoF theorem and requires either exactly two canonicals per component or a cross-check to |residual| < 3e-3. C-18 derives the mass-loading law independently and warns in the abstract about precisely this pattern: 'reducing v but keeping the tabulated pdot implicitly changes L by (1+f)^(-1/2) - an energy leak proportional to the loading'. B-20 transcribes the validator's own admission that the entry-point space is incomplete ('Mdot_SN alone is not yet a supported entry point here').",
    "expected": "Declare (Lmech_SN, v_SN) the primary pair for the SN channel - which the default path already is - and either reject a supplied pdot_SN/Mdot_SN as over-determining, or accept it and assert |2L/(pdot*v) - 1| < 3e-3 and |pdot/(Mdot*v) - 1| < 3e-3 per row, raising on failure.",
    "failure_scenario": "With FB_thermCoeffSN = 0.3 and a declared sps_col_pdot_SN, the SN momentum injected exceeds what the energy budget implies by 1/sqrt(0.3*(1+b)) ~ 1.8x; the bubble's energy equation and its momentum equation are then driven by two different physical winds, and v_mech_total = 2L/pdot reports a velocity consistent with neither Lmech_SN nor Mdot_SN.",
    "repro": "Declare sps_col_pdot_SN with FB_thermCoeffSN != 1 and compare the exported pdot_SN against Mdot_SN*velocity_SN_modified.",
    "confidence": "high",
    "lenses": ["A", "B", "C"],
    "divergence": "ABC",
    "status": "corroborated",
    "source_ids": ["S10-A-03", "S10-C-02", "S10-C-18", "S10-B-20"]
  },
  {
    "id": "S10-R-05",
    "file": "trinity/sps/read_sps.py",
    "line": 341,
    "class": "numerical",
    "severity": "S2",
    "claim": "All ten interpolators are unconstrained cubic splines on LINEAR y over quantities spanning many decades, with no non-negativity clamp anywhere on the output, and the t=0 prepend deliberately creates a flat-then-steep first interval - the textbook ringing configuration - immediately before the solver's most fragile step.",
    "evidence": "A: read_sps.py:341-354 build interp1d(t_Myr, y, kind='cubic') on linear arrays (the log10 storage is undone at sps_columns.py:209 before interpolation); read_sps.py:263-274 prepend t=0.0 with y_new[0] = y[0], a zeroth-order constant extension; 'No positivity clamp is applied to any interpolated value.' C-05 derives why this matters: L_SN is exactly zero before ~3.5 Myr and jumps 1-2 dex within a fraction of a Myr, and an unconstrained cubic across a near-step undershoots negative on the zero side; a negative L_mech entering dE_b/dt removes energy from the bubble. C-25 warns about a negative driver at exactly t=0, which the prepend re-creates through ringing rather than through backward extrapolation. B-15 flags the same, and notes the docstring's 'cubic is recommended for small-value interpolations' is an unsupported claim.",
    "expected": "A monotone C1 scheme (PCHIP) - same order on smooth stretches, no overshoot at the SN step, guaranteed non-negative given non-negative data - or log-space interpolation with an explicit floor far below any dynamically relevant value. At minimum, clip interpolated Qi/Lbol/Lmech_*/pdot_* at zero and assert min >= 0 on a 1e4-point grid.",
    "failure_scenario": "Between two widely separated knots on the post-SN decline, or on the pre-onset side of the SN step, fQi or fLmech_SN returns a negative value; a negative ionising rate or negative injected luminosity propagates into the shell/bubble solve with no diagnostic.",
    "repro": "Build the interpolants and evaluate on numpy.linspace(t_min, t_max, 100000); assert (values >= 0).all() for fQi, fLbol, fLmech_W, fLmech_SN, fpdot_W, fpdot_SN. Check the first interval (0 -> t[1]) separately.",
    "confidence": "medium",
    "lenses": ["A", "B", "C"],
    "divergence": "ABC",
    "status": "corroborated",
    "source_ids": ["S10-A-07", "S10-B-15", "S10-C-05", "S10-C-25"]
  },
  {
    "id": "S10-R-06",
    "file": "trinity/sps/read_sps.py",
    "line": 233,
    "class": "deadcode",
    "severity": "S3",
    "claim": "When a Mdot_SN column is declared, the Lmech_SN/Lmech_total column that the validator still mandates is read, unit-converted, mass-scaled, differenced, possibly warned about, clamped - and then never used. The interface simultaneously requires a column and ignores it.",
    "evidence": "A: Lmech_SN_raw is assigned at read_sps.py:199/201 and clamped at 208; its only consumer is line 233, inside the `else` of `if 'Mdot_SN' in cols:` (line 230). Meanwhile sps_columns.py:299-303 raises unless Lmech_total or Lmech_SN is declared. Lmech_SN_final at line 246 is ALWAYS recomputed as 0.5*Mdot_SN*v_mod**2 and never takes the column value, so the exported SN luminosity is 0.5*theta_SN*Mdot_SN_col*v_SN^2, which need not equal theta_SN*L_SN_raw by any bounded amount. B-20 transcribes the validator's own admission of the gap.",
    "expected": "Either cross-check the two (per S10-R-04's tolerance) and raise on disagreement, or relax the validator so Lmech_total/Lmech_SN is not demanded when Mdot_SN is supplied. Removing the dead column alone is not sufficient - the quadruple still needs the redundancy check.",
    "failure_scenario": "A user supplies an accurate Mdot_SN table plus a validator-mandated Lmech_total column with different physics; the exported Lmech_SN silently follows Mdot_SN alone and can differ from the table's own SN luminosity by any amount, with no message.",
    "repro": "Declare both sps_col_Mdot_SN and sps_col_Lmech_SN with mutually inconsistent values; observe the returned Lmech_SN depends only on Mdot_SN.",
    "confidence": "high",
    "lenses": ["A", "B"],
    "divergence": "AB",
    "status": "corroborated",
    "source_ids": ["S10-A-04", "S10-B-20"]
  },
  {
    "id": "S10-R-07",
    "file": "trinity/sps/update_feedback.py",
    "line": 156,
    "class": "regime",
    "severity": "S3",
    "claim": "The off-grid policy is correct physics but is documented nowhere and is delivered as a bare ValueError rather than a recorded end-reason. Past t_max the code refuses - Lens C's top-ranked correct behaviour, above zeroing and far above clamping - but a raw exception is not the 'explicit end-reason' C asks for, and no prose in the slice states any policy at all.",
    "evidence": "A: update_feedback.py:156-159 raises ValueError('Time t=... outside SPS range ...') on a closed interval, in BOTH directions; there is no clamping, no zeroing, no extrapolation. C-04 derives that clamping past t_max would convert a finite ~1e55 erg SN budget into an unbounded source and change the reported fate, and ranks 'refuse with an explicit end-reason' first among acceptable behaviours. B-13 reports the slice's prose says nothing whatsoever about bounds_error, fill_value, extrapolation or clamping. Reconciler: the below-t_min raise is NOT a defect - t_min is exactly 0.0 by construction so t < t_min is unreachable for physical times, and C-25's required below-t_min clamping is already delivered by the t=0 prepend (a constant zeroth-order extension). See section 3.",
    "expected": "Document the policy in the module docstring, and convert the t > t_max raise into a recorded stopping fate ('SPS table exhausted at t = ...') so a long-running low-feedback cloud terminates with an auditable reason instead of a traceback. Whether the run driver already catches it is outside this slice and should be checked.",
    "failure_scenario": "A run that outlives the bundled table's t_max dies with an uncaught exception and loses its output, where the physics called for a clean, labelled termination. Separately, a maintainer reading only the docstrings cannot tell whether the code clamps, extrapolates or raises.",
    "repro": "Call get_current_sps_feedback(2*t_max, params) and inspect whether the exception is caught by the run driver and recorded as a fate.",
    "confidence": "medium",
    "lenses": ["A", "B", "C"],
    "divergence": "AB",
    "status": "corroborated",
    "source_ids": ["S10-B-13", "S10-C-04", "S10-C-25"]
  },
  {
    "id": "S10-R-08",
    "file": "trinity/sps/read_sps.py",
    "line": 214,
    "class": "silent-failure",
    "severity": "S3",
    "claim": "EPSILON = 1e-100 is used as a bare denominator guard, not as a log floor. It converts a zero or negative table entry into a ~1e100-magnitude finite number that passes every downstream isfinite check; and because de-logged columns are always positive, the guard can only ever fire on the case it handles worst.",
    "evidence": "A: read_sps.py:35 EPSILON = 1e-100; line 214 Mdot_wind = pdot**2/(2*np.maximum(Lmech_wind_raw, EPSILON)); line 215 velocity_wind = 2*Lmech_wind_raw/np.maximum(pdot_wind_raw, EPSILON); line 233 the same for v_SN**2. Line 174 checks only np.isfinite, which 1e100-scale values pass. For log-declared columns 10**x > 0 always, so all three np.maximum calls are unreachable; they fire only for linear-declared columns holding an exact 0, an underflowed value, or a negative. C-09 poses exactly this as an open question ('whether EPSILON is a log-floor (correct use) or a denominator guard (dangerous use)') and derives that at late times L_w and pdot_w go to zero TOGETHER, so 2L/eps for small-but-nonzero L gives an enormous, potentially superluminal velocity. A answers C's question: it is a denominator guard.",
    "expected": "Threshold-test rather than nudge: if pdot < threshold then set v = 0 and Mdot = 0. Validate that Lmech_W, pdot_W, Lbol and Qi are strictly positive after conversion and raise with the offending row index. The v_w < c assertion from S10-R-03 catches the blowup independently.",
    "failure_scenario": "A linear-declared pdot_W column with a 0.0 or a stray negative in its first row gives velocity_wind = 2L/1e-100 ~ 1e130 pc/Myr and a matching pdot - all finite, all silently interpolated, all fed to the shell momentum equation.",
    "repro": "Declare sps_col_pdot_W with 'linear' against a file whose first row holds 0 in that column; inspect velocity_wind.",
    "confidence": "medium",
    "lenses": ["A", "C"],
    "divergence": "AC",
    "status": "corroborated",
    "source_ids": ["S10-A-06", "S10-C-09"]
  },
  {
    "id": "S10-R-09",
    "file": "trinity/sps/sps_columns.py",
    "line": 152,
    "class": "citation",
    "severity": "S3",
    "claim": "The bundled table's provenance is a documented blank - SB99 is named with no version, IMF, metallicity, time grid, row count, or star-formation mode - and nothing validates that sps_refmass equals the normalisation the table was actually generated with. The missing documentation is S3; the unvalidated normalisation behind it is a live 1e6-class correctness risk that no ratio test can see.",
    "evidence": "B-12: the only provenance statements in the whole slice are 'Legacy SB99 7-column positional preset', 'canonical SB99 export layout', and the path lib/default/sps/starburst99/1e6cluster_default.csv; a search of the slice prose finds no IMF, no Kroupa/Chabrier/Salpeter, no Z, no metallicity, no grid range. The only quantitative hint is the filename '1e6cluster'. C-13 derives why this is more than a documentation issue: f_mass = M_cluster/sps_refmass is meaningful only if sps_refmass names the table's own normalisation; SB99's instantaneous-burst mode normalises to the burst mass in the .input file (1e6 Msun is the shipped default, not a law) while its continuous mode normalises to a star-formation RATE of 1 Msun/yr, a dimensionally different object whose Q(t) rises to a plateau. B-04 adds the matching schema-side check. Reconciler: this error class is invisible to v = 2L/pdot invariance AND to Lens A's f_mass trace (CL-2), because it lives one layer above the slice.",
    "expected": "Record SPS code + version, IMF form and mass limits, metallicity, rotation, star-formation mode and the burst normalisation in the bundled file's header or a docs entry; assert the schema default of sps_refmass equals the bundled table's normalisation; add C's shape discriminator Qi(20 Myr) < 0.1*Qi(1 Myr) to reject a continuous-SF table used as a burst table.",
    "failure_scenario": "A per-Msun-normalised or continuous-SF table used with sps_refmass = 1e6 gives every extensive driver a 1e6 error in either direction, with no diagnostic: downward the bubble stalls and the run reports a plausible collapse; upward it disperses every cloud and looks like a strong scientific result.",
    "repro": "Compare the sps_refmass schema default in trinity/_input against the bundled table's own normalisation; assert Qi[t~20 Myr] < 0.1*Qi[t~1 Myr]; assert trapz(Mdot dt) < M_cluster and trapz(Lbol dt, 0..40 Myr) < 0.007*M_cluster*c^2.",
    "confidence": "high",
    "lenses": ["B", "C"],
    "divergence": "BC",
    "status": "corroborated",
    "source_ids": ["S10-B-12", "S10-B-04", "S10-C-13", "S10-C-03"]
  },
  {
    "id": "S10-R-10",
    "file": "trinity/sps/read_sps.py",
    "line": 210,
    "class": "divergence",
    "severity": "S3",
    "claim": "The FB_* correction pipeline - the physics that turns raw SPS output into injected feedback - is documented only by name. No prose anywhere states the algebraic form of FB_thermCoeffWind, FB_mColdWindFrac, FB_thermCoeffSN or FB_mColdSNFrac, so the code is unverifiable against its own documentation. The algebra is in fact clean and matches Lens C's derived mass-loading law; it simply is not written down.",
    "evidence": "B-18: read_sps.py:39 names the four coefficients as 'Wind corrections' / 'SN corrections' and says only 'Thermal efficiency and cold mass corrections are applied to winds and SN'; the section markers say '=== WIND corrections (same math as the legacy path) ===' and '=== SN corrections (with user-override pluggability) ==='. A supplies the missing algebra by tracing: with a = FB_mColdWindFrac and theta = FB_thermCoeffWind, Mdot = (1+a)*pdot^2/(2L), v = sqrt(theta/(1+a))*2L/pdot, pdot_out = sqrt(theta*(1+a))*pdot, L_out = theta*L exactly - i.e. the thermalisation coefficient is a pure multiplier on the mechanical luminosity and mass loading has zero net effect on it. The SN default path collapses identically to L_SN_out = theta_SN*L_SN_raw. Reconciler: this matches C-18's derived requirement (L fixed, pdot *= sqrt(1+f), v /= sqrt(1+f), Mdot *= (1+f)) exactly - see CL-4.",
    "expected": "Write the four collapsed expressions above into the docstring. 'Same math as the legacy path' makes legacy parity the correctness bar for physics that has never been stated; a f vs (1-f) inversion in either cold-mass fraction would be unverifiable as documented.",
    "failure_scenario": "A future edit inverts (1+f) to (1-f) or moves theta from L to pdot; no documentation contradicts it, both defaults are 0/1 so the default test suite is insensitive, and the error surfaces only in a paper run that enables mass loading.",
    "repro": "Set FB_mColdWindFrac = 1.0 and assert pdot_W scales by exactly sqrt(2) while Lmech_W is unchanged; set FB_thermCoeffWind = 0.3 and assert Lmech_W scales by exactly 0.3.",
    "confidence": "high",
    "lenses": ["A", "B", "C"],
    "divergence": "AB",
    "status": "corroborated",
    "source_ids": ["S10-B-18", "S10-C-18"]
  },
  {
    "id": "S10-R-11",
    "file": "trinity/sps/read_sps.py",
    "line": 112,
    "class": "regime",
    "severity": "S3",
    "claim": "f_mass is validated only as finite and > 0. There is no warning when M_cluster = f_mass * sps_refmass falls below the IMF-sampling threshold, where linear scaling of an IMF-averaged table stops describing any individual cluster.",
    "evidence": "A: read_sps.py:112-115 checks isfinite and > 0 and nothing else; read_sps.py:172-173 is the only mass dependence in the module, an exact linear factor with no upper or lower bound on the extrapolation, and there is no metallicity or IMF axis of any kind (a single file, no interpolation between tables). C-12 derives the thresholds: N(>8 Msun) ~ M_cluster/100, and because Qi and L_w are dominated by the top few stars the effective sample is far smaller and the scatter far larger than Poisson - safe above 1e5 Msun; 10-30% scatter and an unsampled WR phase at 1e4-1e5; factor-of-few at 1e3-1e4; the IMF-averaged table describes no individual cloud below 1e3. C-22 adds that a smooth SN rate is qualitatively wrong below ~10 supernovae. C reports SPEC-073 flags param/paperII_grid_sweep.param reaching M_cluster = 100 Msun.",
    "expected": "A warning (not an error) at ingest keyed on M_cluster with the documented thresholds. The smooth-rate treatment itself should not change - it is correct for the 1-D model - but the validity limit belongs next to it.",
    "failure_scenario": "A published parameter grid contains cells the model cannot represent, presented with the same confidence as the valid cells.",
    "repro": "Run with M_cluster = 100 Msun (mCloud=1e4, sfe=0.01) and check whether any warning is emitted.",
    "confidence": "medium",
    "lenses": ["A", "C"],
    "divergence": "AC",
    "status": "corroborated",
    "source_ids": ["S10-C-12", "S10-C-22", "S10-A-18"]
  },
  {
    "id": "S10-R-12",
    "file": "trinity/sps/update_feedback.py",
    "line": 181,
    "class": "silent-failure",
    "severity": "S3",
    "claim": "v_mech_total = 2*Lmech_total/pdot_total divides with no zero or epsilon guard, unlike the three structurally identical divisions in read_sps.py which are all wrapped in np.maximum(..., EPSILON). Separately, this ratio is exact only at spline knots.",
    "evidence": "A: update_feedback.py:181 `v_mech_total = (2. * Lmech_total / pdot_total)[()]`, versus read_sps.py:215 `2 * Lmech_wind_raw / np.maximum(pdot_wind_raw, EPSILON)` and read_sps.py:233. pdot_total is a cubic spline of pdot_W + pdot_SN, so both interpolation undershoot (S10-R-05) and a genuinely low-momentum epoch can drive it to zero or negative. C-09/F.13 derives the same 0/0 hazard physically: as the massive stars die L_w and pdot_w go to zero together, and a bare guard gives an enormous velocity for a small-but-finite L. A also establishes that interp1d is a linear operator on y, so the ADDITIVE identities survive off-grid but the ratio does not: 2*fLmech_total(t)/fpdot_total(t) equals the true grid-point velocity only AT knots.",
    "expected": "The same threshold treatment recommended in S10-R-08 (if pdot_total below threshold, return v = 0), or an explicit raise rather than returning inf/nan. Any test of the v = 2L/pdot identity must be evaluated at knots, not on a dense grid.",
    "failure_scenario": "pdot_total interpolates to zero at some t; v_mech_total becomes inf (or nan if Lmech_total is also zero), is written into SPSFeedback and propagated into the force budget with no warning.",
    "repro": "Evaluate fpdot_total on a dense grid over the late-time decline and check for a zero crossing; then call get_current_sps_feedback there.",
    "confidence": "medium",
    "lenses": ["A", "C"],
    "divergence": "AC",
    "status": "corroborated",
    "source_ids": ["S10-A-05", "S10-C-09"]
  },
  {
    "id": "S10-R-13",
    "file": "trinity/sps/read_sps.py",
    "line": 192,
    "class": "silent-failure",
    "severity": "S3",
    "claim": "read_sps never calls validate_user_column_map, yet _read_sps_user unconditionally indexes six canonicals out of cols. If the param layer skips validation the user gets a bare KeyError instead of the carefully written fillable-template diagnostic that already exists.",
    "evidence": "A: read_sps.py:117-125 checks only that seven param KEYS exist; it never calls sps_columns.validate_user_column_map (defined at sps_columns.py:278 with _format_missing_template at 314-331). _read_sps_user then does cols['t'] (186), cols['Lbol'] and cols['fi'] (192, 194), cols['Lmech_total'] and cols['Lmech_W'] (201, 211), cols['pdot_W'] (212) with no guard. B-22's stronger claim - that Lmech_W and pdot_W are not required at all - is REFUTED by A: _REQUIRED_ALWAYS is {t, Qi, Lbol, Lmech_W, pdot_W}. The residual risk is only that the validator may never run.",
    "expected": "Call validate_user_column_map(column_map, filepath) at the top of _read_sps_user, or assert it has already run.",
    "failure_scenario": "A .param declaring sps_col_t/Qi/Lbol/Lmech_W/pdot_W but omitting sps_col_fi dies with KeyError: 'fi' at line 192 rather than the actionable 'missing sps_col_* for [...]' message that the module already contains.",
    "repro": "Call read_sps with a column_map lacking 'fi'.",
    "confidence": "medium",
    "lenses": ["A"],
    "divergence": "none",
    "status": "single-lens",
    "source_ids": ["S10-A-15"]
  },
  {
    "id": "S10-R-14",
    "file": "trinity/sps/read_sps.py",
    "line": 228,
    "class": "units",
    "severity": "S3",
    "claim": "params['FB_vSN'].value is consumed with no unit conversion, while the v_SN column path for the same physical quantity goes through the full UNIT_CONVERSIONS machinery. The two sources of one quantity have different unit contracts inside this module.",
    "evidence": "A: read_sps.py:225-228 - `if 'v_SN' in cols: velocity_SN_base = cols['v_SN']` (already converted via UNIT_CONVERSIONS['v_SN'], sps_columns.py:143-148) `else: velocity_SN_base = params['FB_vSN'].value`, used raw. The value then enters velocity_SN_base**2 at line 233 and velocity_SN_modified at 236, both of which require pc/Myr for Mdot_SN to come out in Msun/Myr. B-07 flags the same asymmetry from the docstrings and quantifies it: 1 km/s = 1.0227 pc/Myr, so an omitted km/s conversion is a deceptively small 2.3% velocity error, 4.6% in Mdot_SN.",
    "expected": "Convert FB_vSN through UNIT_CONVERSIONS['v_SN'], or assert/document that the param layer delivers it in pc/Myr. Resolution requires checking the .param schema - outside this slice.",
    "failure_scenario": "FB_vSN specified in km/s and used raw as pc/Myr gives v_SN 2.3% low, Mdot_SN 4.6% high and pdot_SN 2.3% high - too small to notice, too large to be right. A cm/s value used raw would be off by ~1e10.",
    "repro": "Check the declared unit of FB_vSN in the .param schema and whether read_param converts to AU before read_sps reads .value.",
    "confidence": "medium",
    "lenses": ["A", "B"],
    "divergence": "AB",
    "status": "corroborated",
    "source_ids": ["S10-A-12", "S10-B-07"]
  },
  {
    "id": "S10-R-15",
    "file": "trinity/sps/update_feedback.py",
    "line": 95,
    "class": "state",
    "severity": "S4",
    "claim": "SPSFeedback.__len__ hardcodes the literal 13 and __iter__ hardcodes the field list; both are hand-maintained duplicates of the 13-field dataclass declaration. They are consistent today, but a positional swap between same-unit fields is undetectable by any dimensional, range or finiteness check.",
    "evidence": "A: fields declared at update_feedback.py:66-78 (13), __iter__ at 82-87 re-lists all 13 by name, __len__ at 93-95 returns the literal 13, __getitem__ at 89-91 rebuilds list(self) on every index access; all three are currently consistent with each other and with read_sps.py's 11-element return order. B-24 documents the promised contract (unpacking, feedback[0] == feedback.t, len over the fields). C-21 supplies the reason it matters: Lmech_W, Lmech_SN and Lmech_total are all powers of order 1e40 erg/s and pdot_W/pdot_SN/pdot_total all forces of order 1e32 dyn, so only the sum rules and the distinct velocity bands (v_W ~ 2000 km/s vs v_SN ~ 3000-10000 km/s) can distinguish them.",
    "expected": "Derive all three from dataclasses.fields(self) / dataclasses.astuple(self); add a test asserting positional order matches named access for every field, plus Lmech_total == Lmech_W + Lmech_SN and pdot_total == pdot_W + pdot_SN.",
    "failure_scenario": "A 14th field is added; __len__ still returns 13 and __iter__ omits it, so a positional unpack silently drops the new quantity - or a reorder feeds SN luminosity where wind luminosity belongs, and every magnitude-based test still passes.",
    "repro": "assert len(fb) == len(dataclasses.fields(fb)); assert list(fb) == [getattr(fb, f.name) for f in dataclasses.fields(fb)].",
    "confidence": "high",
    "lenses": ["A", "B", "C"],
    "divergence": "ABC",
    "status": "corroborated",
    "source_ids": ["S10-A-13", "S10-B-24", "S10-C-21"]
  },
  {
    "id": "S10-R-16",
    "file": "trinity/sps/read_sps.py",
    "line": 39,
    "class": "other",
    "severity": "S4",
    "claim": "A cluster of docstring-vs-code and docstring-vs-docstring drift, each individually minor. Lens A resolves five of the six in favour of the code: the code is right and the prose is stale in every case.",
    "evidence": "(a) 'All arrays have t=0 prepended' vs the implementation comment 'idempotent - skip if the file already starts at t=0'; A confirms read_sps.py:263 prepends only when t[0] != 0.0, so array length is N or N+1 and the post-condition is t[0] == 0, not 'a row was prepended' [B-11]. (b) The module docstring says get_interpolation wraps the arrays 'on params[sps_f]' while the function returns a dict; A confirms it builds and returns the dict at 357-368 [B-27]. (c) The update_feedback module docstring says it updates the params dictionary; A's full branch inventory shows only reads of params['sps_f'] and a freshly built dataclass return [B-08]. (d) ColumnSpec's docstring restricts file_column to str for user files / int for the preset; A confirms parse_sps_col_value makes it an int whenever isdigit() and load_user_columns resolves ints positionally on ANY file [B-10]. (e) read_sps's Notes present a wind velocity as an output; A confirms it is never returned - though B-09's failure scenario is REFUTED, since 2*L_out/pdot_out = sqrt(theta/(1+a))*2L/pdot = v_W exactly, so a consumer recomputing it from the exported arrays gets the loader's own value [B-09]. (f) The documented fallback list orders the Mdot_SN derivation before the v_SN default; A confirms the code binds v_SN at 225-228 before using it at 233, so B-06 is doc-drift only, not a live bug. (g) The params contract omits sps_column_map and sps_refmass [B-23], and the 'cgs' example sps_col_Qi 0 cgs log places Qi at the preset's time index [B-28].",
    "expected": "Correct the seven statements. None changes behaviour; all of them mislead the next maintainer, and (f) in particular would lead a reader to believe the code divides by an unbound v_SN.",
    "failure_scenario": "",
    "repro": "",
    "confidence": "high",
    "lenses": ["A", "B"],
    "divergence": "AB",
    "status": "corroborated",
    "source_ids": ["S10-B-06", "S10-B-08", "S10-B-09", "S10-B-10", "S10-B-11", "S10-B-23", "S10-B-27", "S10-B-28"]
  },
  {
    "id": "S10-R-17",
    "file": "trinity/sps/sps_columns.py",
    "line": 166,
    "class": "deadcode",
    "severity": "S4",
    "claim": "Dead code and unused imports, plus one rider that matters for the column-mapping audit: DEFAULT_SPS_COLUMN_MAP - the literal whose contents section 2 verified against the documentation - is referenced nowhere in the slice, so the object actually injected for the bundled file has not been shown to be this literal.",
    "evidence": "A: sps_columns.py:166-174 DEFAULT_SPS_COLUMN_MAP is unreferenced within the slice; the live path is params['sps_column_map'].value (read_sps.py:129). Unused imports: `sys` (read_sps.py:25), `cvt` (read_sps.py:28 - no cvt. appears in the file), `updateDict` (update_feedback.py:13). CanonicalSpec.canonical_au_unit (sps_columns.py:56) is populated for all 13 canonicals and never read. read_sps.py:263's `len(t) == 0` disjunct is broken-if-reached (line 265 would index Qi[0] on an empty array) and unreachable in practice (_scan_layout raises unless one numeric row exists). B's prose says the preset is 'injected as the column map for the bundled default file', presumably by read_param.py - outside the slice.",
    "expected": "Remove the three dead imports and the len(t)==0 disjunct. Before touching DEFAULT_SPS_COLUMN_MAP, confirm which module imports it and that the injected map for the bundled file is exactly this literal - that confirmation is what makes the section 2 mapping diff binding on the default run path.",
    "failure_scenario": "",
    "repro": "grep -rn 'DEFAULT_SPS_COLUMN_MAP' trinity/ to find the injecting caller.",
    "confidence": "high",
    "lenses": ["A", "B"],
    "divergence": "none",
    "status": "corroborated",
    "source_ids": ["S10-A-14", "S10-A-09", "S10-C-24"]
  },
  {
    "id": "S10-R-18",
    "file": "trinity/sps/update_feedback.py",
    "line": 184,
    "class": "other",
    "severity": "S4",
    "claim": "pdotdot_total is scope creep: no lens's physics specification asks for a time derivative of the total momentum rate, no lens could name its consumer, its documented unit is missing, and it is computed by the crash-prone stencil of S10-R-01.",
    "evidence": "C's entire physics derivation - the identities, the totals, the effective velocity, the force budget - never requires dp/dt of the injection rate. B-14 reports the field is documented only as 'Numerical derivative of total momentum rate for time evolution' and, uniquely among the 13 SPSFeedback fields, carries no unit (A confirms the dimension is Msun*pc/Myr^3, consistent with cvt.pdotdot_cgs2au existing). A traces the computation but identifies no in-slice consumer.",
    "expected": "Identify the consumer and document it, or remove the field. If it is needed, take it from the spline's analytic derivative rather than a fixed 1e-9 stencil - which also fixes S10-R-01 and removes the noise a fixed absolute step produces across the abrupt SN turn-on.",
    "failure_scenario": "",
    "repro": "grep -rn 'pdotdot' trinity/ to find consumers.",
    "confidence": "medium",
    "lenses": ["A", "B", "C"],
    "divergence": "scope-creep",
    "status": "single-lens",
    "source_ids": ["S10-B-14", "S10-A-02"]
  },
  {
    "id": "S10-R-19",
    "file": "trinity/sps/read_sps.py",
    "line": 186,
    "class": "numerical",
    "severity": "S4",
    "claim": "validate_t_monotonic runs 77 lines before the t=0.0 prepend, so the prepend can undo the property just validated, and nothing checks t[0] >= 0.",
    "evidence": "A: read_sps.py:186 calls validate_t_monotonic on the raw column; the prepend happens at 263-274. If the table's first time is negative the check passes, 0.0 is inserted in front of it, and the array is non-monotonic - and scipy's interp1d defaults to assume_sorted=False, so it silently sorts x and y together rather than raising. A first time that is tiny but nonzero (e.g. 1e-30 Myr) instead creates a near-duplicate knot and an ill-conditioned first spline interval. C-08's requirement that duplicates be rejected IS met on the raw array (A: diffs <= 0 rejected) - this is the post-prepend gap only.",
    "expected": "Reject t[0] < 0 up front, or use `t[0] > 0.0` as the prepend condition, or re-validate after the prepend.",
    "failure_scenario": "A table with a negative first time or an offset time origin is silently reordered by scipy rather than rejected, moving the artificial constant-extension row into the interior of the grid.",
    "repro": "Ingest a table whose first time is negative and inspect fQi.x.",
    "confidence": "medium",
    "lenses": ["A"],
    "divergence": "none",
    "status": "single-lens",
    "source_ids": ["S10-A-10"]
  },
  {
    "id": "S10-R-20",
    "file": "trinity/sps/sps_columns.py",
    "line": 431,
    "class": "regime",
    "severity": "S4",
    "claim": "Header-row detection skips '#'-prefixed lines, so the standard SB99 convention of a commented header is never recognised - making header-name column specs unusable on exactly the files this module targets.",
    "evidence": "A: _scan_layout:431-440 walks upward from data_start, continues on blank and '#'-starting lines, examines only the first line that is neither, and breaks unconditionally; a header written as '# time Qi fi Lbol ...' is skipped, header_names stays empty, and load_user_columns:482-489 raises 'no header row was detected'. The token-count equality test at 437 also rejects a header whose column count differs (a leading '#' token, or a separate units row). B-16 documents the same rule from the prose. Reconciler: A's traced error message is clear and actionable, so this is loud, not silent - B-16's feared bare 'x is not in list' does not occur.",
    "expected": "Strip a leading '#' from candidate header lines before the token-count and non-numeric tests, or document that only uncommented headers are supported.",
    "failure_scenario": "A user writes sps_col_Qi 'Q_H 1/s log' against a normal SB99 file with a '#'-commented header and is told no header exists, when it visibly does.",
    "repro": "Run load_user_columns on a file whose only header line begins with '#', with any string-valued file_column.",
    "confidence": "high",
    "lenses": ["A", "B"],
    "divergence": "none",
    "status": "corroborated",
    "source_ids": ["S10-A-08", "S10-B-16"]
  },
  {
    "id": "S10-R-21",
    "file": "trinity/sps/sps_columns.py",
    "line": 130,
    "class": "units",
    "severity": "S4",
    "claim": "The luminosity and momentum-rate canonicals offer no AU pass-through unit, while t, Qi, Mdot_SN and v_SN all do. A table already written in code units cannot declare its luminosity or momentum columns.",
    "evidence": "A: UNIT_CONVERSIONS entries for Lbol/Lmech_W/Lmech_total/Lmech_SN/Li/Ln (lines 130-135) accept only 'erg/s', 'L_sun', 'cgs'; pdot_W/pdot_SN (136-137) only 'g*cm/s^2', 'cgs'. By contrast 't' has 'Myr': 1.0, 'Qi' has '1/Myr': 1.0, 'Mdot_SN' has 'Msun/Myr': 1.0, 'v_SN' has 'pc/Myr': 1.0. parse_sps_col_value rejects any unit not in the table. The canonical_au_unit strings for the missing entries are already declared at lines 69-86 (and are themselves never read - see S10-R-17).",
    "expected": "Add 'Msun*pc^2/Myr^3': 1.0 to the six luminosity entries and 'Msun*pc/Myr^2': 1.0 to the two pdot entries.",
    "failure_scenario": "",
    "repro": "parse_sps_col_value('Lbol', '3 Msun*pc^2/Myr^3 linear') raises ValueError.",
    "confidence": "high",
    "lenses": ["A"],
    "divergence": "none",
    "status": "single-lens",
    "source_ids": ["S10-A-11"]
  },
  {
    "id": "S10-R-22",
    "file": "trinity/sps/update_feedback.py",
    "line": 151,
    "class": "state",
    "severity": "S4",
    "claim": "params['sps_f'] caches ten interpolators baked at one specific f_mass; nothing in this module invalidates or re-derives them, and no docstring mentions the caching at all.",
    "evidence": "A: read_sps.py:172-173 multiplies every mass_scaled column by f_mass at load time, freezing it into the spline y-values; get_current_sps_feedback:151 reads params['sps_f'].value with no key, hash or f_mass check. t_min/t_max ARE re-derived per call from fQi.x, so they cannot go stale relative to the splines - only the mass normalisation can. B reports no caching claim appears anywhere in the slice's prose. A rates its own confidence low: whether a mass change mid-run or an in-process sweep occurs is outside this slice.",
    "expected": "Store f_mass alongside sps_f and assert it matches the current cluster mass, or rebuild on change; and document that sps_f is a cache.",
    "failure_scenario": "A run where the cluster mass changes (re-collapse / second SF event), or an in-process sweep reusing a params object across configurations, keeps using the previous mass's feedback with no warning.",
    "repro": "",
    "confidence": "low",
    "lenses": ["A", "B"],
    "divergence": "AB",
    "status": "single-lens",
    "source_ids": ["S10-A-16"]
  },
  {
    "id": "S10-R-23",
    "file": "trinity/sps/read_sps.py",
    "line": 24,
    "class": "other",
    "severity": "S4",
    "claim": "`import scipy` alone is relied on to make scipy.interpolate accessible; this works only via SciPy's lazy-subpackage __getattr__ (SciPy >= 1.9) or if another module happened to import scipy.interpolate first. Relatedly, kind='cubic' needs >= 4 points, so a 2- or 3-row SPS table raises from SciPy.",
    "evidence": "A: read_sps.py:24 `import scipy`, then lines 341-354 use scipy.interpolate.interp1d; there is no `import scipy.interpolate` or `from scipy.interpolate import interp1d` anywhere in the file. CLAUDE.md pins scipy<2 with no lower bound.",
    "expected": "`from scipy.interpolate import interp1d`, which is version-independent.",
    "failure_scenario": "On SciPy < 1.9, or in an import order where nothing else has pulled in scipy.interpolate, get_interpolation raises AttributeError: module 'scipy' has no attribute 'interpolate'.",
    "repro": "python -c \"import scipy; scipy.interpolate\" on the pinned scipy version.",
    "confidence": "medium",
    "lenses": ["A"],
    "divergence": "none",
    "status": "single-lens",
    "source_ids": ["S10-A-19"]
  },
  {
    "id": "S10-R-24",
    "file": "trinity/sps/sps_columns.py",
    "line": 236,
    "class": "other",
    "severity": "S4",
    "claim": "file_column parsing uses str.isdigit(), so '-1' and '+1' are not recognised as integers and fall through to the header-name branch, producing an error that names the wrong problem.",
    "evidence": "A: sps_columns.py:236-237 `file_column = (int(file_column_raw) if file_column_raw.isdigit() else file_column_raw)`. '-1'.isdigit() is False, so it becomes the string '-1' and load_user_columns:490-495 reports it as a missing header name; '007'.isdigit() is True and becomes 7. B-17's stronger claim that no bounds check exists for integer indices is REFUTED by A (0 <= idx < n_cols is checked at 474-479).",
    "expected": "Either accept negative indices explicitly or reject them with a message naming the actual problem.",
    "failure_scenario": "",
    "repro": "parse_sps_col_value('Qi', '-1 1/s log') yields ColumnSpec(file_column='-1', ...).",
    "confidence": "medium",
    "lenses": ["A"],
    "divergence": "none",
    "status": "single-lens",
    "source_ids": ["S10-A-17"]
  }
]
```
